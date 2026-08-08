//! AI image-generation providers (Provider Layer design doc): one `ImageProvider` trait over
//! three request kinds — text-to-image, image-to-image, image-edit — with backends for
//! **fal.ai** (the default primary; async queue), **OpenRouter** (chat-completions image
//! path), and a **Mock** that wraps the procedural renderer so the whole pipeline runs with
//! no key and no network. Lives in `lumagen-core` (no egui); the UI never imports a backend
//! directly.

mod fal;
mod mock;
mod openrouter;

pub use fal::FalProvider;
pub use mock::MockProvider;
pub use openrouter::OpenRouterProvider;

use crate::data::MapId;

/// Install the process-wide rustls crypto provider (`ring`) exactly once. reqwest is built
/// with `rustls-no-provider` (pure-Rust ring, no NASM on Windows), and constructing ANY
/// `reqwest::Client` before a provider is installed **panics** — inside a detached tokio
/// task that panic is silently swallowed. Idempotent and cheap; `http_client` calls it so no
/// call site can forget.
pub fn init_crypto() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if rustls::crypto::ring::default_provider().install_default().is_err() {
            // Another component installed a provider first — that's fine.
            tracing::debug!("rustls crypto provider was already installed");
        }
    });
}

/// The one way this crate builds an HTTP client: crypto provider installed, timeout applied.
/// A builder failure (TLS backend / resolver init) surfaces as a typed error — the tempting
/// `reqwest::Client::new()` fallback panics on exactly that failure class and would silently
/// drop the timeout besides.
pub(crate) fn http_client(timeout: std::time::Duration) -> Result<reqwest::Client, ProviderError> {
    init_crypto();
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| ProviderError::ClientBuild(e.to_string()))
}

/// The long-lived client for result downloads: one shared connection pool and TLS session
/// for the downloads that must happen immediately because fal URLs expire. Built once; a
/// build failure is cached and surfaced as the same typed error on every call.
pub(crate) fn download_client() -> Result<reqwest::Client, ProviderError> {
    static CLIENT: std::sync::OnceLock<Result<reqwest::Client, String>> = std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| http_client(std::time::Duration::from_mins(2)).map_err(|e| e.to_string()))
        .clone()
        .map_err(ProviderError::ClientBuild)
}

/// Build a JSON object from string-keyed pairs without `serde_json::json!`, whose macro
/// expansion hits the workspace-denied `Result::unwrap`.
pub(crate) fn obj(pairs: &[(&str, serde_json::Value)]) -> serde_json::Value {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect::<serde_json::Map<String, serde_json::Value>>()
        .into()
}

/// Split a `data:<mime>;base64,<payload>` URI into `(mime, payload)`.
pub(crate) fn parse_data_uri(url: &str) -> Option<(&str, &str)> {
    url.strip_prefix("data:")?.split_once(";base64,")
}

/// Sanitize an upstream error body for display/logging. fal's 4xx responses **echo the input
/// image** (`"input":"data:image/png;base64,<megabytes>"`), so a raw error can be many MB — and
/// rendering it in the UI makes egui tessellate tens of millions of glyph quads, overflowing its
/// index buffer and hard-crashing the wgpu backend. Cut any base64 payload down to a marker,
/// then cap the length so no error string can ever reach the UI oversized.
pub(crate) fn sanitize_error_body(body: &str) -> String {
    const CAP: usize = 600;
    // Replace each inline base64 payload (`;base64,<run>`) with a short marker.
    let mut cleaned = String::with_capacity(body.len().min(2048));
    let mut rest = body;
    while let Some(pos) = rest.find(";base64,") {
        let start = pos + ";base64,".len();
        cleaned.push_str(&rest[..start]);
        cleaned.push_str("<truncated>");
        // The base64 run ends at the next quote/brace/bracket (JSON) or the end.
        let end = rest[start..].find(['"', '}', ']']).map_or(rest.len(), |q| start + q);
        rest = &rest[end..];
    }
    cleaned.push_str(rest);
    // Cap the total length at a char boundary.
    match cleaned.char_indices().nth(CAP) {
        Some((cut, _)) => {
            let mut s = cleaned[..cut].to_string();
            s.push('…');
            s
        }
        None => cleaned,
    }
}

/// A text-only chat completion against any OpenAI-compatible endpoint (OpenAI, OpenRouter, a
/// local Ollama server, …). Used by the optional LLM-powered Material Description Check to
/// improve a material description. Returns the assistant's message text.
pub async fn chat_complete(endpoint: &str, api_key: &str, model: &str, system: &str, user: &str) -> Result<String, ProviderError> {
    if api_key.trim().is_empty() {
        return Err(ProviderError::Api {
            status: 0,
            message: "no LLM API key set".into(),
            kind: None,
        });
    }
    let http = http_client(std::time::Duration::from_mins(1))?;
    let base = endpoint.trim().trim_end_matches('/');
    let url = format!("{base}/chat/completions");
    let body = obj(&[
        ("model", model.into()),
        (
            "messages",
            vec![
                obj(&[("role", "system".into()), ("content", system.into())]),
                obj(&[("role", "user".into()), ("content", user.into())]),
            ]
            .into(),
        ),
        ("temperature", serde_json::Value::from(0.3)),
    ]);
    let resp = http
        .post(&url)
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {api_key}"))
        .json(&body)
        .send()
        .await?;
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        return Err(ProviderError::Api {
            status,
            message: resp.text().await.unwrap_or_default(),
            kind: None,
        });
    }
    let raw: serde_json::Value = resp.json().await?;
    let content = raw
        .pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
        .map(|s| s.trim().to_string())
        .ok_or(ProviderError::Empty)?;
    Ok(content)
}

/// Make a cheap, real authenticated request to check that a provider key/endpoint works, WITHOUT
/// running a paid generation. OpenRouter: `GET /api/v1/key` + a model-listing check for
/// `models[0]`. fal: an auth probe plus a schema probe for EVERY configured slug (the
/// text-to-image and edit slots). Returns a human-readable success line or a typed error.
pub async fn test_connection(id: ProviderId, api_key: &str, models: &[String]) -> Result<String, ProviderError> {
    let model = models.first().map(String::as_str).unwrap_or_default();
    if api_key.trim().is_empty() {
        return Err(ProviderError::Api {
            status: 0,
            message: "no API key set".into(),
            kind: None,
        });
    }
    let http = http_client(std::time::Duration::from_secs(15))?;
    match id {
        ProviderId::OpenRouter => {
            let resp = http
                .get("https://openrouter.ai/api/v1/key")
                .header(reqwest::header::AUTHORIZATION, format!("Bearer {api_key}"))
                .send()
                .await?;
            let status = resp.status().as_u16();
            if !resp.status().is_success() {
                return Err(ProviderError::Api {
                    status,
                    message: if status == 401 || status == 403 {
                        "invalid API key".into()
                    } else {
                        resp.text().await.unwrap_or_default()
                    },
                    kind: None,
                });
            }
            // Key accepted — also verify the configured model actually exists and can output
            // images (a typo'd custom model otherwise fails only at generation time).
            match openrouter_model_check(&http, model).await {
                ModelCheck::OkImage => Ok(format!("OpenRouter reachable · key valid · model {model} ✓")),
                ModelCheck::NoImageOutput => Err(ProviderError::Api {
                    status: 0,
                    message: format!("key valid, but model '{model}' can't output images — pick an image model"),
                    kind: None,
                }),
                ModelCheck::NotFound => Err(ProviderError::Api {
                    status: 0,
                    message: format!("key valid, but model '{model}' was not found on OpenRouter — check the id"),
                    kind: None,
                }),
                // The model listing being unreachable is not a key failure — report success
                // for what was actually verified.
                ModelCheck::Unknown => Ok("OpenRouter reachable · key valid (model list unavailable)".into()),
            }
        }
        ProviderId::Fal => {
            // The queue's request routes live under the BASE app id (owner/app) — a subpath
            // model like `fal-ai/flux-pro/kontext` must be trimmed to `fal-ai/flux-pro`, and
            // the request id must be UUID-shaped, or the route 405s before checking auth.
            let base_app = model.split('/').take(2).collect::<Vec<_>>().join("/");
            let url = format!("https://queue.fal.run/{base_app}/requests/00000000-0000-0000-0000-000000000000/status");
            let resp = http.get(&url).header(reqwest::header::AUTHORIZATION, format!("Key {api_key}")).send().await?;
            let status = resp.status().as_u16();
            if status == 401 || status == 403 {
                Err(ProviderError::Api {
                    status,
                    message: "invalid API key".into(),
                    kind: None,
                })
            } else if status == 404 || resp.status().is_success() {
                // 404 (request not found) or 2xx both mean auth was accepted. Also verify
                // every configured slug (t2i + edit) exists by probing its public schema.
                let mut verified = Vec::new();
                for m in models {
                    match fetch_fal_caps(&http, m).await {
                        Ok(_) => verified.push(format!("{m} ✓")),
                        Err(ProviderError::Api { .. }) => {
                            return Err(ProviderError::Api {
                                status: 0,
                                message: format!("key accepted, but model '{m}' was not found on fal — check the id"),
                                kind: None,
                            });
                        }
                        // Schema service unreachable ≠ key/model failure.
                        Err(_) => verified.push(format!("{m} (schema unavailable)")),
                    }
                }
                Ok(format!("fal.ai reachable · key accepted · {}", verified.join(" · ")))
            } else {
                Err(ProviderError::Api {
                    status,
                    message: resp.text().await.unwrap_or_default(),
                    kind: None,
                })
            }
        }
        ProviderId::Mock => Ok("Offline mock — no key or network needed".into()),
    }
}

/// What a fal endpoint's input schema accepts, probed from its **public per-endpoint
/// OpenAPI** (`fal.ai/api/openapi/queue/openapi.json?endpoint_id=…`). Model families take
/// different size dialects (`image_size` vs `aspect_ratio`) and different knobs; sending an
/// unsupported field is silently ignored (fal docs: "passing the wrong one will have no
/// effect") — so the request is built from what the endpoint actually declares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FalCaps {
    pub image_size: bool,
    pub aspect_ratio: bool,
    pub seed: bool,
    pub quality: bool,
    pub negative_prompt: bool,
    pub guidance_scale: bool,
    pub num_inference_steps: bool,
    pub strength: bool,
    pub num_images: bool,
    pub sync_mode: bool,
    pub output_format: bool,
}

impl FalCaps {
    /// The permissive fallback when the schema can't be fetched: send everything we have and
    /// let the endpoint ignore what it doesn't know.
    pub fn permissive() -> Self {
        Self {
            image_size: true,
            aspect_ratio: true,
            seed: true,
            quality: true,
            negative_prompt: true,
            guidance_scale: true,
            num_inference_steps: true,
            strength: true,
            num_images: true,
            sync_mode: true,
            output_format: true,
        }
    }
}

/// Fetch a fal endpoint's input capabilities from its public OpenAPI schema. `Err` with a
/// 404-ish status means the endpoint id doesn't exist on fal.
pub async fn fetch_fal_caps(http: &reqwest::Client, model: &str) -> Result<FalCaps, ProviderError> {
    let url = format!("https://fal.ai/api/openapi/queue/openapi.json?endpoint_id={model}");
    let resp = http.get(&url).send().await?;
    if !resp.status().is_success() {
        return Err(ProviderError::Api {
            status: resp.status().as_u16(),
            message: format!("no fal endpoint '{model}'"),
            kind: None,
        });
    }
    let raw: serde_json::Value = resp.json().await?;
    parse_fal_caps(&raw, model).ok_or_else(|| ProviderError::Job(format!("could not parse fal schema for '{model}'")))
}

/// Extract the POST input schema's property set from a fal per-endpoint OpenAPI document.
fn parse_fal_caps(raw: &serde_json::Value, model: &str) -> Option<FalCaps> {
    // The submit path is "/{model}"; JSON-pointer-escape it ("/" → "~1", "~" → "~0").
    let escaped = format!("/{model}").replace('~', "~0").replace('/', "~1");
    let schema_ref = raw
        .pointer(&format!("/paths/{escaped}/post/requestBody/content/application~1json/schema/$ref"))?
        .as_str()?;
    let name = schema_ref.rsplit('/').next()?;
    let props = raw.pointer(&format!("/components/schemas/{name}/properties"))?.as_object()?;
    let has = |k: &str| props.contains_key(k);
    Some(FalCaps {
        image_size: has("image_size"),
        aspect_ratio: has("aspect_ratio"),
        seed: has("seed"),
        quality: has("quality"),
        negative_prompt: has("negative_prompt"),
        guidance_scale: has("guidance_scale"),
        num_inference_steps: has("num_inference_steps"),
        strength: has("strength"),
        num_images: has("num_images"),
        sync_mode: has("sync_mode"),
        output_format: has("output_format"),
    })
}

/// Result of checking a model id against OpenRouter's public model listing.
enum ModelCheck {
    /// The model exists and lists "image" among its output modalities.
    OkImage,
    /// The model exists but cannot output images.
    NoImageOutput,
    /// No model with this id.
    NotFound,
    /// The listing couldn't be fetched/parsed — unknown either way.
    Unknown,
}

/// Look a model id up in OpenRouter's public `/api/v1/models` listing (no auth required).
async fn openrouter_model_check(http: &reqwest::Client, model: &str) -> ModelCheck {
    let Ok(resp) = http.get("https://openrouter.ai/api/v1/models").send().await else {
        return ModelCheck::Unknown;
    };
    if !resp.status().is_success() {
        return ModelCheck::Unknown;
    }
    let Ok(raw) = resp.json::<serde_json::Value>().await else {
        return ModelCheck::Unknown;
    };
    let Some(list) = raw.get("data").and_then(|d| d.as_array()) else {
        return ModelCheck::Unknown;
    };
    let Some(entry) = list.iter().find(|m| m.get("id").and_then(|i| i.as_str()) == Some(model)) else {
        return ModelCheck::NotFound;
    };
    let outputs_image = entry
        .pointer("/architecture/output_modalities")
        .and_then(|m| m.as_array())
        .is_some_and(|mods| mods.iter().any(|m| m.as_str() == Some("image")));
    if outputs_image { ModelCheck::OkImage } else { ModelCheck::NoImageOutput }
}

/// Which backend a provider talks to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ProviderId {
    /// fal.ai — the default primary (async job queue; control/edit models).
    Fal,
    /// OpenRouter — one key across many models, single blocking call, reports cost.
    OpenRouter,
    /// Offline procedural mock (no key, no network) — always available.
    Mock,
}

impl ProviderId {
    /// Human label for the UI.
    pub const fn label(self) -> &'static str {
        match self {
            ProviderId::Fal => "fal.ai",
            ProviderId::OpenRouter => "OpenRouter",
            ProviderId::Mock => "Offline mock",
        }
    }
}

/// What kind of generation this is. `TextToImage` has no reference; the others carry ≥1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestKind {
    /// Make an image from a prompt alone (a new albedo).
    TextToImage,
    /// Restyle/transform keeping structure (future).
    ImageToImage,
    /// Edit/derive from a reference image — every derived semantic map (albedo = reference).
    ImageEdit,
}

/// A reference image. Both backends accept a URL or base64, so model it once.
#[derive(Clone, Debug)]
pub enum ImageInput {
    /// A public https URL.
    Url(String),
    /// Raw bytes + MIME type → a data URI (fal) or a content part (OpenRouter).
    Bytes { data: Vec<u8>, mime: &'static str },
}

impl ImageInput {
    /// Render as a `data:` URI (used by both backends for inline references).
    pub fn as_data_uri(&self) -> String {
        match self {
            ImageInput::Url(u) => u.clone(),
            ImageInput::Bytes { data, mime } => {
                use base64::Engine as _;
                format!("data:{};base64,{}", mime, base64::engine::general_purpose::STANDARD.encode(data))
            }
        }
    }
}

/// Size expressed the way each backend likes; the adapter picks the right field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageSpec {
    /// Aspect ratio string, e.g. "1:1".
    Aspect(String),
    /// A named preset, e.g. "2K".
    Preset(String),
    /// Explicit pixel dimensions.
    Pixels { width: u32, height: u32 },
    /// The highest square resolution the model supports: the adapter tries the largest known
    /// tier and steps down on submit-time validation rejections (which cost nothing).
    MaxSquare,
}

/// Output container format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFormat {
    /// PNG (lossless; the default for map data).
    Png,
    /// JPEG.
    Jpeg,
}

/// A unified generation request. `model` is the fal model id or OpenRouter slug from
/// settings; `references` is empty for text-to-image and holds the albedo for derived maps.
#[derive(Clone, Debug)]
pub struct ImageRequest {
    /// Which kind of generation.
    pub kind: RequestKind,
    /// The map this request is for (drives the instruction template).
    pub map: MapId,
    /// fal model id or OpenRouter slug (from settings; never hardcoded as the only option).
    pub model: String,
    /// The full prompt/instruction (compiled base + material semantics + map template).
    pub prompt: String,
    /// Reference images (the albedo for derived maps); empty for text-to-image.
    pub references: Vec<ImageInput>,
    /// Requested size, if any.
    pub size: Option<ImageSpec>,
    /// Shared project seed so a map set stays coherent.
    pub seed: Option<i64>,
    /// How many images to generate (>1 for the variations grid).
    pub num_images: Option<u32>,
    /// Output container format.
    pub output_format: Option<ImageFormat>,
    /// Backend-specific knobs (typed escape hatch).
    pub backend: BackendParams,
}

/// Model-specific knobs that don't generalize, kept typed where common with an `extra`
/// escape hatch so a new model never needs a code change.
#[derive(Clone, Debug)]
pub enum BackendParams {
    /// fal.ai parameters.
    Fal {
        /// Guidance scale.
        guidance_scale: Option<f32>,
        /// Inference steps.
        num_inference_steps: Option<u32>,
        /// Deviation-from-input strength (real field name varies per model).
        strength: Option<f32>,
        /// Negative prompt.
        negative_prompt: Option<String>,
        /// Inline data-URI result instead of an expiring hosted URL.
        sync_mode: Option<bool>,
        /// Quality tier ("low" / "medium" / "high" / "auto") for models that take one
        /// (gpt-image family). Settings → Generation → Quality.
        quality: Option<String>,
        /// Long-tail model params merged into the request body.
        extra: serde_json::Value,
    },
    /// OpenRouter parameters.
    OpenRouter {
        /// Output modalities, e.g. `["image","text"]`.
        modalities: Vec<String>,
        /// `image_config` object (aspect_ratio, image_size, strength, …).
        image_config: serde_json::Value,
        /// Routing preferences.
        provider: Option<serde_json::Value>,
    },
}

/// How a finished image is delivered (fal → expiring URL; OpenRouter → base64).
#[derive(Clone, Debug)]
pub enum ImageData {
    /// A hosted URL (fal results expire — fetch immediately).
    Url(String),
    /// Inline base64 + MIME type.
    Base64 { data: Vec<u8>, mime: String },
}

impl ImageData {
    /// Materialize to bytes: base64-decode, or GET the URL. Fal URLs expire, so callers
    /// should call this as soon as the response lands.
    pub async fn into_bytes(self, http: &reqwest::Client) -> Result<(Vec<u8>, String), ProviderError> {
        match self {
            ImageData::Base64 { data, mime } => Ok((data, mime)),
            ImageData::Url(url) => {
                let resp = http.get(&url).send().await?;
                // Check the status BEFORE treating the body as image bytes: an expired fal
                // URL returns an error page that would otherwise surface as a baffling
                // "image decode" failure hiding the real status — keep the body as the
                // error message, it carries fal's reason.
                let status = resp.status();
                if !status.is_success() {
                    return Err(ProviderError::Api {
                        status: status.as_u16(),
                        message: sanitize_error_body(&resp.text().await.unwrap_or_default()),
                        kind: None,
                    });
                }
                let mime = resp
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("image/png")
                    .to_string();
                let bytes = resp.bytes().await?.to_vec();
                Ok((bytes, mime))
            }
        }
    }
}

/// One produced image.
#[derive(Clone, Debug)]
pub struct OutputImage {
    /// The image payload.
    pub data: ImageData,
    /// Width, if reported.
    pub width: Option<u32>,
    /// Height, if reported.
    pub height: Option<u32>,
    /// Content type, if reported.
    pub content_type: Option<String>,
}

/// A unified generation response.
#[derive(Clone, Debug)]
pub struct ImageResponse {
    /// The produced images.
    pub images: Vec<OutputImage>,
    /// The seed the model used, if reported.
    pub seed: Option<i64>,
    /// Real cost where the backend reports it (OpenRouter `usage.cost`; `None` for fal).
    pub cost: Option<f64>,
    /// The raw provider payload, for debugging/provenance.
    pub raw: serde_json::Value,
}

/// Progress of a queued/running job, surfaced to the UI.
#[derive(Clone, Copy, Debug)]
pub enum JobProgress {
    /// Submitted to the backend.
    Submitted,
    /// Waiting in the provider's queue.
    Queued {
        /// Position in queue, if known.
        position: Option<i64>,
    },
    /// Actively generating.
    Running,
    /// Finished.
    Done,
    /// Failed.
    Failed,
}

/// A progress sink that can cross task boundaries.
pub type ProgressFn = dyn Fn(JobProgress) + Send + Sync;

/// A request to upscale one map to a higher resolution (Step ③ Refine).
#[derive(Clone, Debug)]
pub struct UpscaleRequest {
    /// Which map is being upscaled (labeling/telemetry).
    pub map: MapId,
    /// Source RGBA8 pixels (square).
    pub rgba: Vec<u8>,
    /// Source side length.
    pub side: usize,
    /// Requested output side length. Backends clamp to their factor limits (Topaz: 4×).
    pub target_side: usize,
    /// Upscaler model name (e.g. Topaz "Standard V2", "CGI", "Wonder 3.5").
    pub model: String,
}

/// A request to run a single-image **estimator/preprocessor** — depth (Marigold, Depth
/// Anything) or surface normals — that takes one image and returns one image. Unlike
/// [`ImageRequest`] (prompt-driven, returns an `images` array) and [`UpscaleRequest`] (Topaz
/// factor), these fal endpoints take `{ image_url, … }` and return a single `image` File.
///
/// The depth path reconstructs the **macro** relief; a normals estimator reconstructs the
/// **micro** detail. Both hosted fal endpoints return 8-bit PNG — genuine `f32` macro relief
/// is recovered downstream by low-passing (averaging many 8-bit samples yields sub-LSB
/// precision, and the quantized high frequencies are exactly what the macro pass discards).
#[derive(Clone, Debug)]
pub struct EstimateRequest {
    /// Which map this estimation feeds (labeling/telemetry).
    pub map: MapId,
    /// Source RGBA8 pixels (square) — the albedo the estimator reads.
    pub rgba: Vec<u8>,
    /// Source side length.
    pub side: usize,
    /// The estimator endpoint id (e.g. `fal-ai/imageutils/marigold-depth`,
    /// `fal-ai/image-preprocessors/depth-anything/v2`, `fal-ai/patina`). Configurable per the
    /// same rule as generation slugs — never hardcoded as the only option.
    pub model: String,
    /// Endpoint-specific params merged into the request body (Marigold's `ensemble_size`,
    /// `num_inference_steps`, `processing_res`; PATINA's `maps`; `Null` sends none).
    pub extra: serde_json::Value,
    /// Which result the endpoint returns and how to pick the wanted image from it. Depth models
    /// return a single top-level `image`; PATINA returns a tagged `images[]` array.
    pub output: EstimateOutput,
}

/// How an estimator's response carries the wanted image — endpoints differ, so the caller
/// declares which shape to read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EstimateOutput {
    /// A single top-level `image` File — Marigold, Depth Anything.
    Image,
    /// A single named top-level File field, e.g. `normal_map` / `depth_map` — MiDaS.
    Field(String),
    /// The entry in an `images[]` array whose `map_type` equals this — PATINA (a multi-map
    /// PBR extractor returning one tagged image per requested map).
    MapType(String),
}

/// Estimated cost of one Topaz upscale by OUTPUT megapixels (fal's published tier table:
/// $0.08 ≤ 24MP, $0.16 ≤ 48MP, $0.32 ≤ 96MP, up to $1.36 for 512MP).
pub fn topaz_cost_estimate(target_side: usize) -> f64 {
    let mp = (target_side * target_side) as f64 / 1_000_000.0;
    if mp <= 24.0 {
        0.08
    } else if mp <= 48.0 {
        0.16
    } else if mp <= 96.0 {
        0.32
    } else {
        1.36
    }
}

/// The provider abstraction. Object-safe so the orchestrator can hold `Box<dyn ImageProvider>`.
#[async_trait::async_trait]
pub trait ImageProvider: Send + Sync {
    /// Run one generation to completion (fal submits + polls internally). Cancelable via the
    /// token; `progress` reports queue/running state back to the UI.
    async fn generate(&self, req: &ImageRequest, cancel: tokio_util::sync::CancellationToken, progress: &ProgressFn) -> Result<ImageResponse, ProviderError>;

    /// Upscale one image (Step ③ Refine). fal routes to Topaz; the mock resizes offline;
    /// backends without an upscaler return an error.
    async fn upscale(
        &self,
        _req: &UpscaleRequest,
        _cancel: tokio_util::sync::CancellationToken,
        _progress: &ProgressFn,
    ) -> Result<ImageResponse, ProviderError> {
        Err(ProviderError::Job("upscaling is not supported by this backend — fal.ai drives Topaz".into()))
    }

    /// Run a single-image estimator/preprocessor — depth (Marigold / Depth Anything) or
    /// surface normals — that returns one image. fal submits to the endpoint queue; the mock
    /// derives an offline stand-in from the albedo; backends without estimators error.
    async fn estimate(
        &self,
        _req: &EstimateRequest,
        _cancel: tokio_util::sync::CancellationToken,
        _progress: &ProgressFn,
    ) -> Result<ImageResponse, ProviderError> {
        Err(ProviderError::Job("depth/normal estimation is not supported by this backend".into()))
    }

    /// Which backend this is.
    fn id(&self) -> ProviderId;
}

/// The provider error type (rust-thiserror skill). Wraps HTTP, decode, API, job, cancel,
/// timeout, empty-result, and client-construction failures so the orchestrator can map them
/// onto `MapStatus::Error`.
#[derive(thiserror::Error, Debug)]
pub enum ProviderError {
    /// Transport/HTTP failure.
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON decode failure.
    #[error("decode: {0}")]
    Decode(#[from] serde_json::Error),

    /// The backend returned an error status.
    #[error("api {status}: {message}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Error message from the backend.
        message: String,
        /// Optional backend error kind.
        kind: Option<String>,
    },

    /// A queued job failed or was rejected.
    #[error("job failed: {0}")]
    Job(String),

    /// The operation was canceled.
    #[error("canceled")]
    Canceled,

    /// The operation exceeded its deadline.
    #[error("timed out after {0:?}")]
    Timeout(std::time::Duration),

    /// The response contained no image.
    #[error("no image in response")]
    Empty,

    /// base64 decode of an inline image failed.
    #[error("base64: {0}")]
    Base64(#[from] base64::DecodeError),

    /// The HTTP client could not be built (TLS backend / resolver initialization failed).
    /// Kept as text so provider constructors can cache the failure and surface it per call.
    #[error("http client: {0}")]
    ClientBuild(String),
}

/// Minimal std-only HTTP fixtures for provider tests (no wiremock dependency): each helper
/// runs a plain `TcpListener` on an OS thread and speaks just enough HTTP/1.1 for reqwest.
#[cfg(test)]
pub(crate) mod test_http {
    use std::io::{Read as _, Write as _};

    /// Serve a fixed response for every connection; returns the base URL plus a receiver
    /// yielding each request's first line (e.g. `"PUT /cancel HTTP/1.1"`).
    pub(crate) fn serve(status_line: &'static str, body: &'static str) -> (String, std::sync::mpsc::Receiver<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let addr = listener.local_addr().expect("listener addr");
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut buf = [0u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let head = String::from_utf8_lossy(&buf[..n]).lines().next().unwrap_or_default().to_string();
                let _ = tx.send(head);
                let resp = format!(
                    "HTTP/1.1 {status_line}\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(resp.as_bytes());
            }
        });
        (format!("http://{addr}"), rx)
    }

    /// Accept connections and never respond — a hung upstream that pins an un-raced await
    /// until the client's own request timeout.
    pub(crate) fn serve_hang() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let addr = listener.local_addr().expect("listener addr");
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                std::thread::spawn(move || {
                    let mut buf = [0u8; 4096];
                    let _ = stream.read(&mut buf);
                    std::thread::sleep(std::time::Duration::from_mins(10));
                });
            }
        });
        format!("http://{addr}")
    }

    /// A URL nothing listens on (bind, take the port, drop the listener) — connections are
    /// refused immediately, simulating a transport-level failure.
    pub(crate) fn refused_url() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let addr = listener.local_addr().expect("listener addr");
        drop(listener);
        format!("http://{addr}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod into_bytes {
        use super::*;

        /// An expired/forbidden result URL must surface the real HTTP status and the body's
        /// error text — not flow into the image decoder as a bogus "image decode" failure.
        #[tokio::test]
        async fn non_success_status_surfaces_as_api_error_with_body() {
            let (url, _requests) = test_http::serve("403 Forbidden", "url expired");
            let http = http_client(std::time::Duration::from_secs(10)).expect("client");
            let err = ImageData::Url(format!("{url}/result.png")).into_bytes(&http).await.expect_err("403 must fail");
            match err {
                ProviderError::Api { status, message, .. } => {
                    assert_eq!(status, 403);
                    assert!(message.contains("url expired"), "body text preserved, got {message:?}");
                }
                other => panic!("expected Api error, got {other:?}"),
            }
        }

        /// A 2xx response still materializes to bytes + MIME as before.
        #[tokio::test]
        async fn success_returns_body_and_mime() {
            let (url, _requests) = test_http::serve("200 OK", "not-really-png");
            let http = http_client(std::time::Duration::from_secs(10)).expect("client");
            let (bytes, mime) = ImageData::Url(format!("{url}/result.png")).into_bytes(&http).await.expect("2xx succeeds");
            assert_eq!(bytes, b"not-really-png");
            assert_eq!(mime, "application/json", "mime comes from the content-type header");
        }
    }

    /// A fal 4xx body echoes the multi-MB base64 input image; sanitizing must strip the payload
    /// and cap the length, while keeping the useful message and passing short bodies through —
    /// so a megabyte error can never reach the UI and overflow egui's index buffer.
    #[test]
    fn sanitize_error_body_strips_base64_and_caps_length() {
        let huge = "A".repeat(2_000_000);
        let body = format!(r#"{{"detail":[{{"msg":"Image dimensions are too large. Maximum 2048x2048","input":"data:image/png;base64,{huge}"}}]}}"#);
        let out = super::sanitize_error_body(&body);
        assert!(out.chars().count() <= 601, "must be capped, got {}", out.chars().count());
        assert!(out.contains("too large"), "keeps the useful message");
        assert!(out.contains("<truncated>"), "base64 payload replaced with a marker");
        assert!(!out.contains(&"A".repeat(1000)), "no long base64 run survives");
        // A short, base64-free body passes through unchanged.
        assert_eq!(super::sanitize_error_body("invalid API key"), "invalid API key");
    }

    /// The shared download client builds once and is reusable.
    #[test]
    fn download_client_builds() {
        assert!(download_client().is_ok());
        assert!(download_client().is_ok());
    }
}
