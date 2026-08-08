//! fal.ai provider — the default primary. Async **job queue**: submit → poll/stream status →
//! fetch an expiring result URL → download bytes. Gives first-class access to the
//! control/edit models (FLUX Kontext, Qwen-Image-Edit) that keep a derived map aligned to the
//! albedo reference. Auth is `Authorization: Key <FAL_KEY>`; requests are driven off the URLs
//! fal returns (`status_url`/`response_url`/`cancel_url`) so path changes don't break us.

use super::{
    BackendParams, EstimateOutput, EstimateRequest, ImageData, ImageProvider, ImageRequest, ImageResponse, ImageSpec, JobProgress, OutputImage, ProviderError,
    ProviderId, RequestKind, UpscaleRequest,
};

const QUEUE_BASE: &str = "https://queue.fal.run";
/// Per-HTTP-request timeout (submit, one status poll, result download).
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(2);
/// Default overall submit→result deadline — big image models can take a while.
const DEFAULT_JOB_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(10);

/// fal.ai queue client. Holds one shared `reqwest::Client`.
pub struct FalProvider {
    /// Built once at construction; a builder failure (TLS/resolver init) is kept as text and
    /// surfaced as a typed error on first use — the constructor itself can't fail (the UI
    /// builds providers infallibly) and panicking is denied workspace-wide.
    http: Result<reqwest::Client, String>,
    api_key: String,
    /// Overall deadline for a single submit→result cycle.
    timeout: std::time::Duration,
    /// Per-endpoint input capabilities, probed once from fal's public OpenAPI and cached for
    /// the provider's lifetime (`None` = schema unavailable → permissive fallback).
    caps: tokio::sync::Mutex<std::collections::HashMap<String, Option<super::FalCaps>>>,
}

/// Submit response: the URLs to drive the rest of the lifecycle.
#[derive(serde::Deserialize)]
struct SubmitResponse {
    /// fal's id for the request (kept for logging/provenance; the lifecycle is URL-driven).
    #[allow(dead_code)]
    request_id: String,
    status_url: String,
    response_url: String,
    cancel_url: String,
    #[serde(default)]
    queue_position: Option<i64>,
}

/// Status poll response. Field names are fal's lowercase snake_case (`status`,
/// `queue_position`) — only the status VALUES are SCREAMING_SNAKE_CASE (on the enum), so no
/// struct-level rename here.
#[derive(serde::Deserialize)]
struct StatusResponse {
    status: FalStatus,
    #[serde(default)]
    queue_position: Option<i64>,
}

#[derive(serde::Deserialize, PartialEq, Eq, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum FalStatus {
    InQueue,
    InProgress,
    Completed,
    #[serde(other)]
    Other,
}

impl FalProvider {
    /// Build a provider from an API key read from the OS vault (never from settings state).
    pub fn new(api_key: String) -> Self {
        let http = super::http_client(REQUEST_TIMEOUT).map_err(|e| e.to_string());
        Self {
            http,
            api_key,
            timeout: DEFAULT_JOB_TIMEOUT,
            caps: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// The shared client, or the typed construction error kept from `new`.
    fn http(&self) -> Result<&reqwest::Client, ProviderError> {
        self.http.as_ref().map_err(|e| ProviderError::ClientBuild(e.clone()))
    }

    /// `Authorization` header value — fal uses `Key <FAL_KEY>`, not `Bearer`.
    fn auth_value(&self) -> String {
        format!("Key {}", self.api_key)
    }

    /// The endpoint's declared input capabilities (cached per model; `None` when the schema
    /// couldn't be fetched, in which case the request builder falls back to permissive).
    async fn endpoint_caps(&self, http: &reqwest::Client, model: &str) -> Option<super::FalCaps> {
        if let Some(cached) = self.caps.lock().await.get(model) {
            return cached.clone();
        }
        let fetched = match super::fetch_fal_caps(http, model).await {
            Ok(caps) => {
                tracing::debug!(%model, ?caps, "fal endpoint capabilities probed");
                Some(caps)
            }
            Err(e) => {
                tracing::warn!(%model, "fal schema probe failed ({e}); sending permissive params");
                None
            }
        };
        self.caps.lock().await.insert(model.to_string(), fetched.clone());
        fetched
    }

    /// Set the overall submit→result deadline (Settings → Generation → Job timeout). The
    /// per-request HTTP timeout stays short; this bounds the whole queued job, which for
    /// large image models can legitimately run for many minutes.
    #[must_use]
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Read the key from the environment (`FAL_KEY`, falling back to `FAL_API_KEY`).
    pub fn from_env() -> Option<Self> {
        std::env::var("FAL_KEY")
            .ok()
            .or_else(|| std::env::var("FAL_API_KEY").ok())
            .filter(|k| !k.is_empty())
            .map(Self::new)
    }

    /// Serialize the request body: model input JSON directly (no wrapper), references into
    /// `image_url`, common fields mapped, `BackendParams::Fal` + `extra` merged in. Only the
    /// fields the endpoint's schema declares are sent (`caps`; `None` = schema unknown →
    /// permissive) — model families differ (`image_size` vs `aspect_ratio`, quality, seed…)
    /// and fal silently ignores fields an endpoint doesn't declare.
    fn build_body(req: &ImageRequest, caps: Option<&super::FalCaps>) -> serde_json::Value {
        let permissive = super::FalCaps::permissive();
        let caps = caps.unwrap_or(&permissive);
        let mut body = serde_json::Map::new();
        body.insert("prompt".into(), req.prompt.clone().into());
        if caps.seed
            && let Some(seed) = req.seed
        {
            body.insert("seed".into(), seed.into());
        }
        if caps.num_images
            && let Some(n) = req.num_images
        {
            body.insert("num_images".into(), n.into());
        }
        // Size applies to BOTH kinds: text-to-image picks the output frame, and edits pin
        // the output to the reference's square size explicitly (gpt-image's `auto` may pick
        // a different frame and break 1:1 registration with the albedo).
        if let Some(spec) = &req.size {
            match spec {
                ImageSpec::Aspect(a) => {
                    if caps.aspect_ratio {
                        body.insert("aspect_ratio".into(), a.clone().into());
                    }
                    if caps.image_size
                        && let Some(preset) = Self::aspect_to_preset(a)
                    {
                        body.insert("image_size".into(), preset.into());
                    }
                }
                ImageSpec::Preset(p) => {
                    if caps.image_size {
                        body.insert("image_size".into(), p.clone().into());
                    }
                }
                ImageSpec::Pixels { width, height } => {
                    if caps.image_size {
                        body.insert("image_size".into(), super::obj(&[("width", (*width).into()), ("height", (*height).into())]));
                    } else if caps.aspect_ratio && width == height {
                        // Aspect-only endpoints can't take explicit pixels — at least pin the
                        // square aspect so the model doesn't pick a rectangle.
                        body.insert("aspect_ratio".into(), "1:1".into());
                    }
                }
                // MaxSquare is resolved to a concrete tier by the submit ladder in `generate`
                // before the body is built; treat a stray one as a square aspect request.
                ImageSpec::MaxSquare => {
                    if caps.aspect_ratio {
                        body.insert("aspect_ratio".into(), "1:1".into());
                    } else if caps.image_size {
                        body.insert("image_size".into(), "square_hd".into());
                    }
                }
            }
        }
        if caps.output_format
            && let Some(fmt) = req.output_format
        {
            body.insert(
                "output_format".into(),
                match fmt {
                    super::ImageFormat::Png => "png",
                    super::ImageFormat::Jpeg => "jpeg",
                }
                .into(),
            );
        }
        // References → image_url (first) / image_urls (all). Edits carry the albedo.
        if !req.references.is_empty() {
            let urls: Vec<serde_json::Value> = req.references.iter().map(|r| r.as_data_uri().into()).collect();
            if let Some(first) = urls.first() {
                body.insert("image_url".into(), first.clone());
            }
            body.insert("image_urls".into(), urls.into());
        }
        if let BackendParams::Fal {
            guidance_scale,
            num_inference_steps,
            strength,
            negative_prompt,
            sync_mode,
            quality,
            extra,
        } = &req.backend
        {
            if caps.guidance_scale
                && let Some(g) = guidance_scale
            {
                body.insert("guidance_scale".into(), (*g).into());
            }
            if caps.num_inference_steps
                && let Some(s) = num_inference_steps
            {
                body.insert("num_inference_steps".into(), (*s).into());
            }
            if caps.strength
                && let Some(s) = strength
            {
                body.insert("strength".into(), (*s).into());
            }
            if caps.negative_prompt
                && let Some(n) = negative_prompt
            {
                body.insert("negative_prompt".into(), n.clone().into());
            }
            if caps.sync_mode
                && let Some(sm) = sync_mode
            {
                body.insert("sync_mode".into(), (*sm).into());
            }
            if caps.quality
                && let Some(q) = quality
            {
                body.insert("quality".into(), q.clone().into());
            }
            if let Some(obj) = extra.as_object() {
                for (k, v) in obj {
                    body.insert(k.clone(), v.clone());
                }
            }
        }
        serde_json::Value::Object(body)
    }

    /// Square tiers for "highest resolution possible", largest first. 2880² is the gpt-image
    /// ceiling (8,294,400 total pixels); toolkit models that reject a tier at submit time
    /// (422, free) fall through to the next.
    const MAX_SQUARE_TIERS: [u32; 3] = [2880, 2048, 1024];

    /// The submit attempts for a request: `MaxSquare` text-to-image walks the tier ladder;
    /// everything else is a single attempt with the size as given.
    fn size_ladder(size: Option<&ImageSpec>, kind: RequestKind) -> Vec<Option<ImageSpec>> {
        match (size, kind) {
            (Some(ImageSpec::MaxSquare), RequestKind::TextToImage) => Self::MAX_SQUARE_TIERS
                .iter()
                .map(|s| Some(ImageSpec::Pixels { width: *s, height: *s }))
                .collect(),
            (Some(ImageSpec::MaxSquare), _) => vec![Some(ImageSpec::Aspect("1:1".into()))],
            (other, _) => vec![other.cloned()],
        }
    }

    /// fal's named `image_size` preset for an aspect-ratio string, when one exists.
    fn aspect_to_preset(aspect: &str) -> Option<&'static str> {
        match aspect {
            "1:1" => Some("square_hd"),
            "4:3" => Some("landscape_4_3"),
            "3:4" => Some("portrait_4_3"),
            "16:9" => Some("landscape_16_9"),
            "9:16" => Some("portrait_16_9"),
            _ => None,
        }
    }

    async fn api_error(resp: reqwest::Response) -> ProviderError {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        ProviderError::Api {
            status,
            message: text,
            kind: None,
        }
    }

    /// Consecutive poll/fetch failures tolerated before a job is failed. A submitted job must
    /// NEVER be resubmitted because of a poll blip — the orchestrator's retry would POST a
    /// brand-new PAID job while this one keeps running — so transient failures (transport,
    /// 408/429/5xx, undecodable bodies) are absorbed here and, once the budget is exhausted,
    /// surfaced as the non-retryable [`ProviderError::Job`] carrying the real error.
    const POLL_FAILURE_BUDGET: u32 = 5;

    /// Best-effort server-side cancel of an abandoned queue job (user cancel, deadline
    /// expiry, poll failure) — fal runs (and bills) accepted jobs to completion otherwise.
    /// Capped well below the per-request timeout so a dead network can't stall the exit path.
    async fn cancel_job(&self, http: &reqwest::Client, cancel_url: &str) {
        let put = http.put(cancel_url).header(reqwest::header::AUTHORIZATION, self.auth_value()).send();
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), put).await;
    }

    /// Poll a submitted queue job until COMPLETED (capped backoff, honoring cancel), then
    /// fetch and return the raw result JSON. Shared by generation and upscaling.
    async fn poll_and_fetch(
        &self,
        sub: &SubmitResponse,
        cancel: &tokio_util::sync::CancellationToken,
        progress: &super::ProgressFn,
    ) -> Result<serde_json::Value, ProviderError> {
        let http = self.http()?;
        progress(JobProgress::Queued { position: sub.queue_position });
        let deadline = tokio::time::Instant::now() + self.timeout;
        let mut backoff = std::time::Duration::from_millis(400);
        let mut failures: u32 = 0;
        let mut last_failure = String::new();
        loop {
            if tokio::time::Instant::now() > deadline {
                // The accepted job may still be running (and billing) server-side —
                // best-effort cancel, like the user-cancel path.
                self.cancel_job(http, &sub.cancel_url).await;
                return Err(ProviderError::Timeout(self.timeout));
            }
            // Race the status GET against cancellation: a hung connection would otherwise
            // pin cancel behind the full per-request timeout.
            let status_result = tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    self.cancel_job(http, &sub.cancel_url).await;
                    return Err(ProviderError::Canceled);
                }
                r = http.get(&sub.status_url)
                    .header(reqwest::header::AUTHORIZATION, self.auth_value())
                    .send() => r,
            };
            match status_result {
                Ok(resp) if resp.status().is_success() => match resp.json::<StatusResponse>().await {
                    Ok(st) => {
                        failures = 0;
                        match st.status {
                            FalStatus::Completed => break,
                            FalStatus::InProgress => progress(JobProgress::Running),
                            FalStatus::InQueue => progress(JobProgress::Queued { position: st.queue_position }),
                            FalStatus::Other => {}
                        }
                    }
                    Err(e) => {
                        failures += 1;
                        last_failure = format!("status decode: {e}");
                    }
                },
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    if !matches!(status, 408 | 429 | 500..=599) {
                        // A hard 4xx (revoked key, expired request id) can't heal by
                        // polling — fail now with the real status, not a misleading timeout.
                        self.cancel_job(http, &sub.cancel_url).await;
                        return Err(Self::api_error(resp).await);
                    }
                    failures += 1;
                    last_failure = Self::api_error(resp).await.to_string();
                }
                Err(e) => {
                    failures += 1;
                    last_failure = format!("status poll: {e}");
                }
            }
            if failures >= Self::POLL_FAILURE_BUDGET {
                self.cancel_job(http, &sub.cancel_url).await;
                return Err(ProviderError::Job(format!("status poll failed after {failures} attempts — {last_failure}")));
            }
            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    self.cancel_job(http, &sub.cancel_url).await;
                    return Err(ProviderError::Canceled);
                }
                () = tokio::time::sleep(backoff) => {}
            }
            backoff = (backoff * 2).min(std::time::Duration::from_secs(4));
        }
        // Result fetch. The job COMPLETED — transient failures here must not bubble out as
        // retryable either (a resubmit would run a brand-new paid job), so they get the same
        // bounded in-loop tolerance.
        let mut failures: u32 = 0;
        loop {
            let fetch = tokio::select! {
                biased;
                () = cancel.cancelled() => return Err(ProviderError::Canceled),
                r = http.get(&sub.response_url)
                    .header(reqwest::header::AUTHORIZATION, self.auth_value())
                    .send() => r,
            };
            let failure = match fetch {
                Ok(resp) if resp.status().is_success() => match resp.json::<serde_json::Value>().await {
                    Ok(raw) => return Ok(raw),
                    Err(e) => format!("result decode: {e}"),
                },
                Ok(resp) if matches!(resp.status().as_u16(), 408 | 429 | 500..=599) => Self::api_error(resp).await.to_string(),
                Ok(resp) => return Err(Self::api_error(resp).await),
                Err(e) => format!("result fetch: {e}"),
            };
            failures += 1;
            if failures >= Self::POLL_FAILURE_BUDGET {
                return Err(ProviderError::Job(format!("result fetch failed after {failures} attempts — {failure}")));
            }
            tokio::select! {
                biased;
                () = cancel.cancelled() => return Err(ProviderError::Canceled),
                () = tokio::time::sleep(std::time::Duration::from_millis(500)) => {}
            }
        }
    }

    /// The Topaz upscaler endpoint id.
    const TOPAZ_ENDPOINT: &'static str = "fal-ai/topaz/upscale/image";

    /// Build the Topaz request body: PNG data URI + model + computed factor. Face enhancement
    /// is EXPLICITLY disabled (its API default is true — face hallucination on texture panels
    /// is never wanted); output is PNG for lossless map data.
    fn build_upscale_body(req: &UpscaleRequest) -> Option<serde_json::Value> {
        use base64::Engine as _;
        let png = crate::maps::encode_png_rgba(&req.rgba, req.side)?;
        let data_uri = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(png));
        let factor = (req.target_side as f64 / req.side as f64).clamp(1.0, 4.0);
        Some(super::obj(&[
            ("image_url", data_uri.into()),
            ("model", req.model.clone().into()),
            ("upscale_factor", ((factor * 100.0).round() / 100.0).into()),
            ("output_format", "png".into()),
            ("face_enhancement", false.into()),
            ("crop_to_fill", false.into()),
        ]))
    }

    /// Build an estimator body: the source as a PNG `image_url`, plus any endpoint-specific
    /// params (Marigold's `ensemble_size`/`num_inference_steps`/`processing_res`) merged in.
    fn build_estimate_body(req: &EstimateRequest) -> Option<serde_json::Value> {
        use base64::Engine as _;
        let png = crate::maps::encode_png_rgba(&req.rgba, req.side)?;
        let data_uri = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(png));
        let mut body = serde_json::Map::new();
        body.insert("image_url".into(), data_uri.into());
        if let Some(extra) = req.extra.as_object() {
            for (k, v) in extra {
                body.insert(k.clone(), v.clone());
            }
        }
        Some(serde_json::Value::Object(body))
    }
}

#[async_trait::async_trait]
impl ImageProvider for FalProvider {
    #[tracing::instrument(skip(self, req, cancel, progress), fields(model = %req.model, ?req.map))]
    async fn generate(
        &self,
        req: &ImageRequest,
        cancel: tokio_util::sync::CancellationToken,
        progress: &super::ProgressFn,
    ) -> Result<ImageResponse, ProviderError> {
        let url = format!("{}/{}", QUEUE_BASE, req.model);
        let http = self.http()?;
        let caps = self.endpoint_caps(http, &req.model).await;

        // ── Submit (with the max-resolution tier ladder) ───────────────────
        let ladder = Self::size_ladder(req.size.as_ref(), req.kind);
        let mut sub: Option<SubmitResponse> = None;
        for (i, size) in ladder.iter().enumerate() {
            let mut attempt = req.clone();
            attempt.size = size.clone();
            let body = Self::build_body(&attempt, caps.as_ref());
            progress(JobProgress::Submitted);
            let submit = tokio::select! {
                biased;
                () = cancel.cancelled() => { return Err(ProviderError::Canceled); }
                r = http.post(&url)
                    .header(reqwest::header::AUTHORIZATION, self.auth_value())
                    .json(&body)
                    .send() => { r? }
            };
            let status = submit.status();
            if status.as_u16() == 422 && i + 1 < ladder.len() {
                // Validation rejection (free): this tier exceeds the model's limits — step down.
                tracing::info!(model = %req.model, ?size, "size tier rejected at submit; stepping down");
                continue;
            }
            if !status.is_success() {
                return Err(Self::api_error(submit).await);
            }
            sub = Some(submit.json().await?);
            break;
        }
        let Some(sub) = sub else { return Err(ProviderError::Empty) };
        let raw = self.poll_and_fetch(&sub, &cancel, progress).await?;
        let raw_images = raw.get("images").and_then(|i| i.as_array()).ok_or(ProviderError::Empty)?;
        let mut images = Vec::with_capacity(raw_images.len());
        for im in raw_images {
            let url = im.get("url").and_then(|u| u.as_str()).ok_or(ProviderError::Empty)?;
            let data = if let Some((mime, b64)) = super::parse_data_uri(url) {
                // sync_mode inline data-URI
                use base64::Engine as _;
                let bytes = base64::engine::general_purpose::STANDARD.decode(b64)?;
                ImageData::Base64 {
                    data: bytes,
                    mime: mime.to_string(),
                }
            } else {
                ImageData::Url(url.to_string())
            };
            images.push(OutputImage {
                data,
                width: im.get("width").and_then(|w| w.as_u64()).map(|w| w as u32),
                height: im.get("height").and_then(|h| h.as_u64()).map(|h| h as u32),
                content_type: im.get("content_type").and_then(|c| c.as_str()).map(str::to_string),
            });
        }
        if images.is_empty() {
            return Err(ProviderError::Empty);
        }
        progress(JobProgress::Done);
        Ok(ImageResponse {
            images,
            seed: raw.get("seed").and_then(|s| s.as_i64()).or(req.seed),
            cost: None, // fal doesn't report cost — the orchestrator estimates from config
            raw,
        })
    }

    /// Upscale one map via Topaz (`fal-ai/topaz/upscale/image`): submit the PNG as a data
    /// URI, poll the queue, download the single-file result.
    #[tracing::instrument(skip(self, req, cancel, progress), fields(?req.map, model = %req.model, side = req.side, target = req.target_side))]
    async fn upscale(
        &self,
        req: &UpscaleRequest,
        cancel: tokio_util::sync::CancellationToken,
        progress: &super::ProgressFn,
    ) -> Result<ImageResponse, ProviderError> {
        let body = Self::build_upscale_body(req).ok_or_else(|| ProviderError::Job("could not encode source PNG".into()))?;
        let url = format!("{}/{}", QUEUE_BASE, Self::TOPAZ_ENDPOINT);
        let http = self.http()?;
        progress(JobProgress::Submitted);
        let submit = tokio::select! {
            biased;
            () = cancel.cancelled() => { return Err(ProviderError::Canceled); }
            r = http.post(&url)
                .header(reqwest::header::AUTHORIZATION, self.auth_value())
                .json(&body)
                .send() => { r? }
        };
        if !submit.status().is_success() {
            return Err(Self::api_error(submit).await);
        }
        let sub: SubmitResponse = submit.json().await?;
        let raw = self.poll_and_fetch(&sub, &cancel, progress).await?;
        // Topaz returns a single `image: File`, not an `images` array.
        let file_url = raw.pointer("/image/url").and_then(|u| u.as_str()).ok_or(ProviderError::Empty)?.to_string();
        progress(JobProgress::Done);
        Ok(ImageResponse {
            images: vec![OutputImage {
                data: ImageData::Url(file_url),
                width: None,
                height: None,
                content_type: raw.pointer("/image/content_type").and_then(|c| c.as_str()).map(str::to_string),
            }],
            seed: None,
            cost: Some(super::topaz_cost_estimate(req.target_side)),
            raw,
        })
    }

    /// Run a depth/normal estimator (`fal-ai/imageutils/marigold-depth`,
    /// `fal-ai/image-preprocessors/depth-anything/v2`, a normals endpoint): submit the source
    /// as a PNG data URI, poll the queue, download the single-file result — the same
    /// single-`image` output shape as Topaz, not the `images` array `generate` returns.
    #[tracing::instrument(skip(self, req, cancel, progress), fields(?req.map, model = %req.model, side = req.side))]
    async fn estimate(
        &self,
        req: &EstimateRequest,
        cancel: tokio_util::sync::CancellationToken,
        progress: &super::ProgressFn,
    ) -> Result<ImageResponse, ProviderError> {
        let body = Self::build_estimate_body(req).ok_or_else(|| ProviderError::Job("could not encode source PNG".into()))?;
        let url = format!("{}/{}", QUEUE_BASE, req.model);
        let http = self.http()?;
        progress(JobProgress::Submitted);
        let submit = tokio::select! {
            biased;
            () = cancel.cancelled() => { return Err(ProviderError::Canceled); }
            r = http.post(&url)
                .header(reqwest::header::AUTHORIZATION, self.auth_value())
                .json(&body)
                .send() => { r? }
        };
        if !submit.status().is_success() {
            return Err(Self::api_error(submit).await);
        }
        let sub: SubmitResponse = submit.json().await?;
        let raw = self.poll_and_fetch(&sub, &cancel, progress).await?;
        let image = Self::select_estimate_image(&raw, &req.output).ok_or(ProviderError::Empty)?;
        progress(JobProgress::Done);
        Ok(ImageResponse {
            images: vec![image],
            seed: None,
            cost: None,
            raw,
        })
    }

    fn id(&self) -> ProviderId {
        ProviderId::Fal
    }
}

impl FalProvider {
    /// Pick the wanted image out of an estimator's result JSON per the declared output shape: a
    /// single top-level `image` (depth models), a named File field (MiDaS), or the `images[]`
    /// entry tagged with a `map_type` (PATINA). A `sync_mode` data URI is decoded inline.
    fn select_estimate_image(raw: &serde_json::Value, output: &EstimateOutput) -> Option<OutputImage> {
        let node = match output {
            EstimateOutput::Image => raw.get("image")?,
            EstimateOutput::Field(field) => raw.get(field)?,
            EstimateOutput::MapType(map_type) => raw
                .get("images")?
                .as_array()?
                .iter()
                .find(|im| im.get("map_type").and_then(|t| t.as_str()) == Some(map_type.as_str()))?,
        };
        let url = node.get("url")?.as_str()?;
        let data = if let Some((mime, b64)) = super::parse_data_uri(url) {
            use base64::Engine as _;
            ImageData::Base64 {
                data: base64::engine::general_purpose::STANDARD.decode(b64).ok()?,
                mime: mime.to_string(),
            }
        } else {
            ImageData::Url(url.to_string())
        };
        Some(OutputImage {
            data,
            width: node.get("width").and_then(|w| w.as_u64()).map(|w| w as u32),
            height: node.get("height").and_then(|h| h.as_u64()).map(|h| h as u32),
            content_type: node.get("content_type").and_then(|c| c.as_str()).map(str::to_string),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{BackendParams, ImageInput, ImageRequest, ImageSpec, RequestKind};

    fn edit_request() -> ImageRequest {
        ImageRequest {
            kind: RequestKind::ImageEdit,
            map: crate::data::MapId::Roughness,
            model: "fal-ai/flux-pro/kontext".into(),
            prompt: "make roughness".into(),
            references: vec![ImageInput::Bytes {
                data: vec![1, 2, 3, 4],
                mime: "image/png",
            }],
            size: Some(ImageSpec::Aspect("1:1".into())),
            seed: Some(4821),
            num_images: Some(1),
            output_format: Some(crate::providers::ImageFormat::Png),
            backend: BackendParams::Fal {
                guidance_scale: Some(3.5),
                num_inference_steps: None,
                strength: None,
                negative_prompt: None,
                sync_mode: Some(false),
                quality: Some("low".into()),
                extra: serde_json::Value::Null,
            },
        }
    }

    /// A trimmed copy of fal's real per-endpoint OpenAPI for `openai/gpt-image-2` (the
    /// schema-probe source): `image_size` + `quality`, no `aspect_ratio`, no `seed`.
    const GPT_IMAGE_2_SCHEMA: &str = r##"{
        "openapi": "3.0.4",
        "components": { "schemas": {
            "GptImage2Input": {
                "required": ["prompt"],
                "properties": {
                    "prompt": {"type": "string"},
                    "image_size": {"default": "landscape_4_3"},
                    "quality": {"type": "string", "default": "high"},
                    "num_images": {"type": "integer"},
                    "output_format": {"type": "string", "default": "png"},
                    "sync_mode": {"type": "boolean"}
                },
                "type": "object"
            }
        }},
        "paths": {
            "/openai/gpt-image-2": {
                "post": { "requestBody": { "required": true, "content": { "application/json": {
                    "schema": {"$ref": "#/components/schemas/GptImage2Input"}
                }}}}
            }
        }
    }"##;

    /// The schema probe must read gpt-image-2's real capability set: `image_size` and
    /// `quality` yes; `aspect_ratio`, `seed`, `guidance_scale` no.
    #[test]
    fn parse_fal_caps_reads_gpt_image_2_schema() {
        let raw: serde_json::Value = serde_json::from_str(GPT_IMAGE_2_SCHEMA).unwrap();
        let caps = crate::providers::parse_fal_caps(&raw, "openai/gpt-image-2").expect("caps parsed");
        assert!(caps.image_size);
        assert!(caps.quality);
        assert!(caps.num_images);
        assert!(caps.output_format);
        assert!(caps.sync_mode);
        assert!(!caps.aspect_ratio);
        assert!(!caps.seed);
        assert!(!caps.guidance_scale);
        assert!(!caps.negative_prompt);
    }

    /// With gpt-image-2's capabilities, the body sends `image_size`+`quality` and omits the
    /// fields the endpoint doesn't declare (aspect_ratio/seed/guidance) — the silent-ignore
    /// trap that once produced a 1024×768 "1:1" texture.
    #[test]
    fn body_respects_endpoint_capabilities() {
        let raw: serde_json::Value = serde_json::from_str(GPT_IMAGE_2_SCHEMA).unwrap();
        let caps = crate::providers::parse_fal_caps(&raw, "openai/gpt-image-2").unwrap();
        let mut req = edit_request();
        req.kind = RequestKind::TextToImage;
        req.references.clear();
        let body = FalProvider::build_body(&req, Some(&caps));
        assert_eq!(body.get("image_size").and_then(|v| v.as_str()), Some("square_hd"));
        assert_eq!(body.get("quality").and_then(|v| v.as_str()), Some("low"));
        assert_eq!(body.get("output_format").and_then(|v| v.as_str()), Some("png"));
        assert!(body.get("aspect_ratio").is_none(), "endpoint doesn't take aspect_ratio");
        assert!(body.get("seed").is_none(), "endpoint doesn't take seed");
        assert!(body.get("guidance_scale").is_none(), "endpoint doesn't take guidance_scale");
    }

    /// A text-to-image request must carry the aspect in BOTH dialects: `aspect_ratio` for
    /// FLUX-family endpoints and `image_size` for gpt-image-style ones (whose `auto` default
    /// otherwise returns arbitrary aspects like 1024×768 for a "1:1" texture).
    #[test]
    fn text_to_image_sends_both_size_dialects_when_schema_unknown() {
        let mut req = edit_request();
        req.kind = RequestKind::TextToImage;
        req.references.clear();
        let body = FalProvider::build_body(&req, None);
        assert_eq!(body.get("aspect_ratio").and_then(|v| v.as_str()), Some("1:1"));
        assert_eq!(body.get("image_size").and_then(|v| v.as_str()), Some("square_hd"));
    }

    /// The Topaz body carries the schema's exact fields: a PNG data URI, the model, a
    /// factor computed from source→target (clamped to the 4× cap), PNG output, and — always —
    /// face enhancement OFF (its API default is true; face hallucination on texture panels
    /// is never wanted).
    #[test]
    fn topaz_body_matches_schema() {
        let req = UpscaleRequest {
            map: crate::data::MapId::Albedo,
            rgba: vec![128u8; 8 * 8 * 4],
            side: 8,
            target_side: 32,
            model: "Standard V2".into(),
        };
        let body = FalProvider::build_upscale_body(&req).expect("body");
        assert!(
            body.get("image_url")
                .and_then(|v| v.as_str())
                .is_some_and(|u| u.starts_with("data:image/png;base64,")),
            "image_url must be a PNG data URI"
        );
        assert_eq!(body.get("model").and_then(|v| v.as_str()), Some("Standard V2"));
        assert_eq!(body.get("upscale_factor").and_then(|v| v.as_f64()), Some(4.0), "32/8 = 4×");
        assert_eq!(body.get("output_format").and_then(|v| v.as_str()), Some("png"));
        assert_eq!(body.get("face_enhancement").and_then(|v| v.as_bool()), Some(false));
        // Factor clamps to Topaz's 4× maximum.
        let big = UpscaleRequest { target_side: 100, ..req };
        let body = FalProvider::build_upscale_body(&big).expect("body");
        assert_eq!(body.get("upscale_factor").and_then(|v| v.as_f64()), Some(4.0));
    }

    /// An estimator body carries the source as a PNG data URI and merges the endpoint's extra
    /// params (Marigold's ensemble/steps) verbatim.
    #[test]
    fn estimate_body_carries_image_url_and_extra() {
        let req = EstimateRequest {
            map: crate::data::MapId::Displacement,
            rgba: vec![90u8; 8 * 8 * 4],
            side: 8,
            model: "fal-ai/imageutils/marigold-depth".into(),
            extra: crate::providers::obj(&[("ensemble_size", 10.into()), ("num_inference_steps", 12.into())]),
            output: EstimateOutput::Image,
        };
        let body = FalProvider::build_estimate_body(&req).expect("body");
        assert!(
            body.get("image_url").and_then(|v| v.as_str()).is_some_and(|u| u.starts_with("data:image/png;base64,")),
            "image_url must be a PNG data URI"
        );
        assert_eq!(body.get("ensemble_size").and_then(|v| v.as_i64()), Some(10));
        assert_eq!(body.get("num_inference_steps").and_then(|v| v.as_i64()), Some(12));
    }

    /// The output selector reads each endpoint's result shape: PATINA's `images[]` by
    /// `map_type`, and a depth model's single top-level `image`.
    #[test]
    fn select_estimate_image_reads_each_output_shape() {
        let patina: serde_json::Value = serde_json::from_str(
            r#"{"images":[
                {"url":"https://x/basecolor.png","map_type":"basecolor"},
                {"url":"https://x/normal.png","map_type":"normal","width":1024,"height":1024}
            ],"seed":7}"#,
        )
        .expect("json");
        let img = FalProvider::select_estimate_image(&patina, &EstimateOutput::MapType("normal".into())).expect("normal picked");
        match img.data {
            ImageData::Url(u) => assert!(u.ends_with("normal.png"), "got {u}"),
            ImageData::Base64 { .. } => panic!("expected a url, got inline base64"),
        }
        assert_eq!(img.width, Some(1024));
        assert!(
            FalProvider::select_estimate_image(&patina, &EstimateOutput::MapType("height".into())).is_none(),
            "an unrequested map_type is absent"
        );

        let depth: serde_json::Value = serde_json::from_str(r#"{"image":{"url":"https://x/d.png"}}"#).expect("json");
        let d = FalProvider::select_estimate_image(&depth, &EstimateOutput::Image).expect("image");
        match d.data {
            ImageData::Url(u) => assert!(u.ends_with("d.png"), "got {u}"),
            ImageData::Base64 { .. } => panic!("expected a url, got inline base64"),
        }
    }

    /// The published cost tiers by output megapixels.
    #[test]
    fn topaz_cost_tiers() {
        use crate::providers::topaz_cost_estimate;
        assert!((topaz_cost_estimate(4096) - 0.08).abs() < 1e-9, "4096² = 16.7MP → $0.08");
        assert!((topaz_cost_estimate(6000) - 0.16).abs() < 1e-9, "36MP → $0.16");
        assert!((topaz_cost_estimate(9000) - 0.32).abs() < 1e-9, "81MP → $0.32");
        assert!((topaz_cost_estimate(16384) - 1.36).abs() < 1e-9, "268MP → top tier");
    }

    /// "Max resolution" walks a descending square ladder for text-to-image (each rejected
    /// tier is a free 422 at submit) and pins 1:1 for other kinds.
    #[test]
    fn max_square_ladder_descends_for_text_to_image() {
        let ladder = FalProvider::size_ladder(Some(&ImageSpec::MaxSquare), RequestKind::TextToImage);
        let sides: Vec<u32> = ladder
            .iter()
            .map(|s| match s {
                Some(ImageSpec::Pixels { width, .. }) => *width,
                other => panic!("expected pixels tier, got {other:?}"),
            })
            .collect();
        assert_eq!(sides, vec![2880, 2048, 1024]);
        // Edits inherit the reference size; MaxSquare only pins the square aspect.
        let edit = FalProvider::size_ladder(Some(&ImageSpec::MaxSquare), RequestKind::ImageEdit);
        assert_eq!(edit, vec![Some(ImageSpec::Aspect("1:1".into()))]);
        // An explicit size is a single attempt, unchanged.
        let explicit = FalProvider::size_ladder(Some(&ImageSpec::Pixels { width: 2048, height: 2048 }), RequestKind::TextToImage);
        assert_eq!(explicit.len(), 1);
    }

    /// The queue's status payload uses lowercase keys with SCREAMING_SNAKE_CASE values —
    /// exactly what fal sends; a struct-level rename_all would make every poll fail to
    /// decode.
    #[test]
    fn status_response_decodes_fal_payloads() {
        let queued: StatusResponse = serde_json::from_str(r#"{"status":"IN_QUEUE","queue_position":3}"#).expect("queued decodes");
        assert_eq!(queued.status, FalStatus::InQueue);
        assert_eq!(queued.queue_position, Some(3));
        let running: StatusResponse = serde_json::from_str(r#"{"status":"IN_PROGRESS"}"#).expect("running decodes");
        assert_eq!(running.status, FalStatus::InProgress);
        let done: StatusResponse = serde_json::from_str(r#"{"status":"COMPLETED","response_url":"https://queue.fal.run/x"}"#).expect("done decodes");
        assert_eq!(done.status, FalStatus::Completed);
    }

    mod poll_and_fetch {
        use super::*;
        use crate::providers::test_http;

        fn submission(status_url: String, cancel_url: String) -> SubmitResponse {
            SubmitResponse {
                request_id: "req-1".into(),
                status_url,
                response_url: test_http::refused_url(),
                cancel_url,
                queue_position: None,
            }
        }

        /// A hard 4xx on the status poll (revoked key, expired request id) must fail fast
        /// with the real status instead of being silently ignored until the job deadline
        /// reports a misleading "timed out".
        #[tokio::test]
        async fn hard_4xx_fails_fast_with_the_real_status() {
            let (status_url, _requests) = test_http::serve("404 Not Found", "no such request");
            let provider = FalProvider::new("test-key".into());
            let sub = submission(format!("{status_url}/status"), test_http::refused_url());
            let progress = |_p: JobProgress| {};
            let err = provider
                .poll_and_fetch(&sub, &tokio_util::sync::CancellationToken::new(), &progress)
                .await
                .expect_err("404 must fail");
            match err {
                ProviderError::Api { status, message, .. } => {
                    assert_eq!(status, 404);
                    assert!(message.contains("no such request"), "body preserved, got {message:?}");
                }
                other => panic!("expected Api error, got {other:?}"),
            }
        }

        /// Transport-level poll failures are tolerated up to the consecutive budget and then
        /// surface as the NON-retryable `Job` error — never as retryable `Http`, which would
        /// make the orchestrator resubmit a duplicate paid job while this one keeps running.
        #[tokio::test]
        async fn exhausted_poll_budget_is_not_a_retryable_error() {
            let provider = FalProvider::new("test-key".into());
            let sub = submission(test_http::refused_url(), test_http::refused_url());
            let progress = |_p: JobProgress| {};
            let err = provider
                .poll_and_fetch(&sub, &tokio_util::sync::CancellationToken::new(), &progress)
                .await
                .expect_err("refused polls must fail");
            assert!(
                matches!(&err, ProviderError::Job(msg) if msg.contains("status poll failed")),
                "expected terminal Job error carrying the real failure, got {err:?}"
            );
        }

        /// Deadline expiry must perform the same best-effort server-side cancel as the
        /// user-cancel path — the accepted job keeps running (and billing) otherwise.
        #[tokio::test]
        async fn deadline_expiry_sends_best_effort_cancel() {
            let (status_url, _status_requests) = test_http::serve("200 OK", r#"{"status":"IN_QUEUE","queue_position":1}"#);
            let (cancel_url, cancel_requests) = test_http::serve("200 OK", "");
            let provider = FalProvider::new("test-key".into()).with_timeout(std::time::Duration::from_millis(1));
            let sub = submission(format!("{status_url}/status"), format!("{cancel_url}/cancel"));
            let progress = |_p: JobProgress| {};
            let err = provider
                .poll_and_fetch(&sub, &tokio_util::sync::CancellationToken::new(), &progress)
                .await
                .expect_err("deadline must expire");
            assert!(matches!(err, ProviderError::Timeout(_)), "got {err:?}");
            let head = cancel_requests
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("a cancel request must reach the server");
            assert!(head.starts_with("PUT "), "expected a PUT to cancel_url, got {head:?}");
        }

        /// Cancellation must interrupt an in-flight status request immediately — un-raced,
        /// a hung connection pins cancel behind the 2-minute per-request timeout.
        #[tokio::test]
        async fn cancel_interrupts_a_hung_status_poll() {
            let provider = FalProvider::new("test-key".into());
            let sub = submission(format!("{}/status", test_http::serve_hang()), test_http::refused_url());
            let cancel = tokio_util::sync::CancellationToken::new();
            let canceller = cancel.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                canceller.cancel();
            });
            let started = std::time::Instant::now();
            let progress = |_p: JobProgress| {};
            let err = provider.poll_and_fetch(&sub, &cancel, &progress).await.expect_err("cancelled");
            assert!(matches!(err, ProviderError::Canceled), "got {err:?}");
            assert!(
                started.elapsed() < std::time::Duration::from_secs(30),
                "cancel had to interrupt the hung poll, took {:?}",
                started.elapsed()
            );
        }
    }

    /// A Kontext edit must serialize the albedo reference into `image_url` (and `image_urls`),
    /// as a data URI, with the prompt/seed/format mapped through.
    #[test]
    fn kontext_edit_serializes_albedo_into_image_url() {
        let body = FalProvider::build_body(&edit_request(), None);
        let image_url = body.get("image_url").and_then(|v| v.as_str()).expect("image_url present");
        assert!(image_url.starts_with("data:image/png;base64,"), "image_url not a data URI");
        let image_urls = body.get("image_urls").and_then(|v| v.as_array()).expect("image_urls present");
        assert_eq!(image_urls.len(), 1);
        assert_eq!(body.get("prompt").and_then(|v| v.as_str()), Some("make roughness"));
        assert_eq!(body.get("seed").and_then(|v| v.as_i64()), Some(4821));
        assert_eq!(body.get("output_format").and_then(|v| v.as_str()), Some("png"));
        // The structural shape (keys only — the data URI value is huge) is stable.
        let mut keys: Vec<&str> = body.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        insta::assert_debug_snapshot!(keys);
    }
}
