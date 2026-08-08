//! OpenRouter provider. Image generation (t2i / i2i / edit) rides on **chat completions** —
//! there is no dedicated image endpoint. One blocking `POST /api/v1/chat/completions` with
//! `modalities:["image","text"]` + `image_config`; reference images are `image_url` content
//! parts; the response carries inline base64 images and a real `usage.cost`.

use super::{
    BackendParams, ImageData, ImageProvider, ImageRequest, ImageResponse, ImageSpec, JobProgress, OutputImage, ProviderError, ProviderId, RequestKind,
};

const CHAT_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
/// OpenRouter's image path is one blocking chat call — the HTTP timeout IS the job timeout,
/// and big image models can legitimately run for several minutes.
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(10);

/// OpenRouter chat-completions image client.
pub struct OpenRouterProvider {
    /// Built once at construction; a builder failure (TLS/resolver init) is kept as text and
    /// surfaced as a typed error on first use — the constructor itself can't fail (the UI
    /// builds providers infallibly) and panicking is denied workspace-wide.
    http: Result<reqwest::Client, String>,
    api_key: String,
    referer: Option<String>,
    title: Option<String>,
}

impl OpenRouterProvider {
    /// Build from an API key read from the OS vault.
    pub fn new(api_key: String) -> Self {
        let http = super::http_client(REQUEST_TIMEOUT).map_err(|e| e.to_string());
        Self {
            http,
            api_key,
            referer: None,
            title: Some("Lumagen".into()),
        }
    }

    /// The shared client, or the typed construction error kept from `new`.
    fn http(&self) -> Result<&reqwest::Client, ProviderError> {
        self.http.as_ref().map_err(|e| ProviderError::ClientBuild(e.clone()))
    }

    /// Set the whole-call deadline (Settings → Generation → Job timeout). Rebuilds the HTTP
    /// client, since for this single-call provider the request timeout bounds the whole job.
    #[must_use]
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http = super::http_client(timeout).map_err(|e| e.to_string());
        self
    }

    /// Read the key from the environment (`OPENROUTER_API_KEY`).
    pub fn from_env() -> Option<Self> {
        std::env::var("OPENROUTER_API_KEY").ok().filter(|k| !k.is_empty()).map(Self::new)
    }

    /// Build the chat body. Text-to-image uses a plain string content; i2i/edit builds a
    /// parts array with the text instruction first, then each reference image.
    fn build_body(req: &ImageRequest) -> serde_json::Value {
        let content: serde_json::Value = if req.references.is_empty() {
            req.prompt.clone().into()
        } else {
            let mut parts = vec![super::obj(&[("type", "text".into()), ("text", req.prompt.clone().into())])];
            for r in &req.references {
                parts.push(super::obj(&[
                    ("type", "image_url".into()),
                    ("image_url", super::obj(&[("url", r.as_data_uri().into())])),
                ]));
            }
            parts.into()
        };

        let (modalities, mut image_config, provider) = match &req.backend {
            BackendParams::OpenRouter {
                modalities,
                image_config,
                provider,
            } => (modalities.clone(), image_config.clone(), provider.clone()),
            // Default when the caller didn't construct OpenRouter params.
            BackendParams::Fal { .. } => (vec!["image".into(), "text".into()], super::obj(&[("aspect_ratio", "1:1".into())]), None),
        };
        // For a text-to-image (new albedo), fold the requested size into image_config so the
        // model honors it; edits inherit the reference's dimensions, so size is left alone.
        if req.kind == RequestKind::TextToImage
            && let Some(spec) = &req.size
            && let Some(o) = image_config.as_object_mut()
        {
            match spec {
                ImageSpec::Aspect(a) => {
                    o.entry("aspect_ratio".to_string()).or_insert_with(|| a.clone().into());
                }
                ImageSpec::Preset(p) => {
                    o.entry("image_size".to_string()).or_insert_with(|| p.clone().into());
                }
                ImageSpec::Pixels { width, height } => {
                    o.entry("image_size".to_string())
                        .or_insert_with(|| super::obj(&[("width", (*width).into()), ("height", (*height).into())]));
                }
                // The chat image path has no reliable per-model size negotiation — pin the
                // square aspect and let the model pick its own maximum.
                ImageSpec::MaxSquare => {
                    o.entry("aspect_ratio".to_string()).or_insert_with(|| "1:1".into());
                }
            }
        }

        let mut body = super::obj(&[
            ("model", req.model.clone().into()),
            ("messages", vec![super::obj(&[("role", "user".into()), ("content", content)])].into()),
            ("modalities", modalities.into()),
        ]);
        if !image_config.is_null() {
            body["image_config"] = image_config;
        }
        if let Some(seed) = req.seed {
            body["seed"] = seed.into();
        }
        if let Some(p) = provider {
            body["provider"] = p;
        }
        body
    }

    /// A 2xx body with no images usually still says WHY: the model's text reply (a refusal,
    /// a moderation message, a model that ignored `modalities`) or an in-body `error` object
    /// (upstream failure after the request was accepted). Surface that — truncated, these can
    /// be whole paragraphs — instead of a bare "no image in response".
    fn no_image_error(raw: &serde_json::Value) -> ProviderError {
        const MAX_CHARS: usize = 300;
        let truncate = |s: &str| {
            let t = s.trim();
            if t.chars().count() > MAX_CHARS {
                let cut: String = t.chars().take(MAX_CHARS).collect();
                format!("{cut}…")
            } else {
                t.to_string()
            }
        };
        if let Some(err) = raw.get("error").or_else(|| raw.pointer("/choices/0/error")) {
            let message = err
                .get("message")
                .and_then(|m| m.as_str())
                .map_or_else(|| truncate(&err.to_string()), &truncate);
            return ProviderError::Job(format!("no image in response — provider error: {message}"));
        }
        if let Some(text) = raw
            .pointer("/choices/0/message/content")
            .and_then(|c| c.as_str())
            .map(str::trim)
            .filter(|t| !t.is_empty())
        {
            return ProviderError::Job(format!("no image in response — model replied: {}", truncate(text)));
        }
        ProviderError::Empty
    }
}

#[async_trait::async_trait]
impl ImageProvider for OpenRouterProvider {
    #[tracing::instrument(skip(self, req, cancel, progress), fields(model = %req.model, ?req.map))]
    async fn generate(
        &self,
        req: &ImageRequest,
        cancel: tokio_util::sync::CancellationToken,
        progress: &super::ProgressFn,
    ) -> Result<ImageResponse, ProviderError> {
        let http = self.http()?;
        progress(JobProgress::Submitted);
        progress(JobProgress::Running);
        let body = Self::build_body(req);

        let mut request = http
            .post(CHAT_URL)
            .header(reqwest::header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .json(&body);
        if let Some(r) = &self.referer {
            request = request.header("HTTP-Referer", r);
        }
        if let Some(t) = &self.title {
            request = request.header("X-Title", t);
        }

        // Single blocking call; cancel just drops the future (no server-side cancel).
        let resp = tokio::select! {
            biased;
            () = cancel.cancelled() => { return Err(ProviderError::Canceled); }
            r = request.send() => { r? }
        };
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Api {
                status,
                message: super::sanitize_error_body(&text),
                kind: None,
            });
        }
        // The multi-MB base64 payload transfers during the body read — race it against the
        // token too, or cancel has to wait for the whole download.
        let raw: serde_json::Value = tokio::select! {
            biased;
            () = cancel.cancelled() => { return Err(ProviderError::Canceled); }
            r = resp.json() => r?,
        };

        // Images at choices[0].message.images[].image_url.url (base64 data-URLs).
        let mut images = Vec::new();
        if let Some(choices) = raw.get("choices").and_then(|c| c.as_array()) {
            for choice in choices {
                if let Some(imgs) = choice.pointer("/message/images").and_then(|i| i.as_array()) {
                    for im in imgs {
                        let Some(url) = im.pointer("/image_url/url").and_then(|u| u.as_str()) else {
                            continue;
                        };
                        let Some((mime, b64)) = super::parse_data_uri(url) else {
                            continue;
                        };
                        use base64::Engine as _;
                        let data = base64::engine::general_purpose::STANDARD.decode(b64)?;
                        images.push(OutputImage {
                            data: ImageData::Base64 { data, mime: mime.to_string() },
                            width: None,
                            height: None,
                            content_type: None,
                        });
                    }
                }
            }
        }
        if images.is_empty() {
            return Err(Self::no_image_error(&raw));
        }
        let cost = raw.pointer("/usage/cost").and_then(|c| c.as_f64());
        progress(JobProgress::Done);
        Ok(ImageResponse {
            images,
            seed: req.seed,
            cost,
            raw,
        })
    }

    fn id(&self) -> ProviderId {
        ProviderId::OpenRouter
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{BackendParams, ImageInput, ImageRequest, ImageSpec, RequestKind};

    fn edit_request() -> ImageRequest {
        ImageRequest {
            kind: RequestKind::ImageEdit,
            map: crate::data::MapId::Metallic,
            model: "google/gemini-2.5-flash-image".into(),
            prompt: "make metallic".into(),
            references: vec![ImageInput::Bytes {
                data: vec![9, 8, 7],
                mime: "image/png",
            }],
            size: Some(ImageSpec::Aspect("1:1".into())),
            seed: Some(7),
            num_images: Some(1),
            output_format: None,
            backend: BackendParams::OpenRouter {
                modalities: vec!["image".into(), "text".into()],
                image_config: crate::providers::obj(&[("aspect_ratio", "1:1".into())]),
                provider: None,
            },
        }
    }

    /// An edit request must put the text part BEFORE the image part in the content array, and
    /// carry modalities + image_config + seed at the top level.
    #[test]
    fn openrouter_body_puts_text_before_image_part() {
        let body = OpenRouterProvider::build_body(&edit_request());
        let parts = body.pointer("/messages/0/content").and_then(|c| c.as_array()).expect("content parts array");
        assert_eq!(parts.len(), 2, "text + one image part");
        assert_eq!(parts[0].get("type").and_then(|t| t.as_str()), Some("text"), "first part must be text");
        assert_eq!(parts[0].get("text").and_then(|t| t.as_str()), Some("make metallic"));
        assert_eq!(
            parts[1].get("type").and_then(|t| t.as_str()),
            Some("image_url"),
            "second part must be the image"
        );
        let url = parts[1].pointer("/image_url/url").and_then(|u| u.as_str()).expect("image url");
        assert!(url.starts_with("data:image/png;base64,"));
        assert_eq!(body.get("seed").and_then(|v| v.as_i64()), Some(7));
        let modalities = body.get("modalities").and_then(|m| m.as_array()).expect("modalities");
        assert_eq!(modalities.len(), 2);
        // Structural shape (redacting the huge data URI) is stable.
        let mut redacted = body.clone();
        redacted["messages"][0]["content"][1]["image_url"]["url"] = "<data-uri>".into();
        insta::assert_debug_snapshot!(redacted);
    }

    mod no_image_error {
        use super::*;

        /// A refusal/moderation reply arrives as TEXT with no images — the explanation in
        /// `choices[0].message.content` must reach the user, not a generic "no image".
        #[test]
        fn surfaces_the_model_text_reply() {
            let raw = serde_json::json!({
                "choices": [{ "message": { "content": "I can't generate that image because…" } }]
            });
            let err = OpenRouterProvider::no_image_error(&raw);
            assert!(
                matches!(&err, ProviderError::Job(msg) if msg.contains("I can't generate that image")),
                "got {err:?}"
            );
        }

        /// A 200 body can carry the upstream failure as an `error` object (top-level or
        /// choice-level) — its message beats the model-text fallback.
        #[test]
        fn surfaces_in_body_error_objects() {
            let raw = serde_json::json!({ "error": { "code": 502, "message": "Provider returned error" } });
            let err = OpenRouterProvider::no_image_error(&raw);
            assert!(
                matches!(&err, ProviderError::Job(msg) if msg.contains("Provider returned error")),
                "got {err:?}"
            );

            let raw = serde_json::json!({ "choices": [{ "error": { "message": "upstream failed mid-generation" } }] });
            let err = OpenRouterProvider::no_image_error(&raw);
            assert!(
                matches!(&err, ProviderError::Job(msg) if msg.contains("upstream failed mid-generation")),
                "got {err:?}"
            );
        }

        /// A body with neither text nor an error object still reports the plain Empty error.
        #[test]
        fn falls_back_to_empty_when_the_body_says_nothing() {
            let raw = serde_json::json!({ "choices": [{ "message": {} }] });
            assert!(matches!(OpenRouterProvider::no_image_error(&raw), ProviderError::Empty));
        }

        /// Whole-paragraph replies are truncated so the toast stays readable.
        #[test]
        fn truncates_long_replies() {
            let long = "x".repeat(2000);
            let raw = serde_json::json!({ "choices": [{ "message": { "content": long } }] });
            let err = OpenRouterProvider::no_image_error(&raw);
            let msg = err.to_string();
            assert!(msg.chars().count() < 400, "reply must be truncated, got {} chars", msg.chars().count());
            assert!(msg.ends_with('…'), "truncation is marked: {msg:?}");
        }
    }
}
