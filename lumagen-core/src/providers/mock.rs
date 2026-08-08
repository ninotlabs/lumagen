//! Offline mock provider: wraps the procedural renderer (`maps::render_map`) so the entire
//! generation pipeline — orchestrator, concurrency, derive, install — runs end-to-end with
//! zero network and zero API key. This is the default when no provider key is set, and the
//! basis for testing the orchestrator without a network.

use super::{EstimateRequest, ImageData, ImageProvider, ImageRequest, ImageResponse, JobProgress, OutputImage, ProviderError, ProviderId};
use crate::data::MapId;
use crate::{derive, maps};

/// The offline provider. Always available; needs no key.
pub struct MockProvider;

#[async_trait::async_trait]
impl ImageProvider for MockProvider {
    async fn generate(
        &self,
        req: &ImageRequest,
        _cancel: tokio_util::sync::CancellationToken,
        progress: &super::ProgressFn,
    ) -> Result<ImageResponse, ProviderError> {
        progress(JobProgress::Submitted);
        progress(JobProgress::Running);
        // Render on a blocking thread so the runtime isn't stalled by the CPU rasterizer.
        // Honor num_images (variations): each candidate is the same map at a derived seed.
        let map = req.map;
        let seed = req.seed.unwrap_or(4821) as u32;
        let count = req.num_images.unwrap_or(1).max(1);
        let images = tokio::task::spawn_blocking(move || {
            (0..count)
                .map(|k| {
                    let s = seed.wrapping_add(k);
                    let layout = maps::build_layout(s);
                    let bytes = maps::render_map(map, &layout, s);
                    OutputImage {
                        data: ImageData::Base64 {
                            data: bytes,
                            mime: "image/png".into(),
                        },
                        width: Some(maps::TEX_SIZE as u32),
                        height: Some(maps::TEX_SIZE as u32),
                        content_type: Some("image/png".into()),
                    }
                })
                .collect::<Vec<_>>()
        })
        .await
        .map_err(|e| ProviderError::Job(format!("mock render join: {e}")))?;
        progress(JobProgress::Done);
        Ok(ImageResponse {
            images,
            seed: req.seed,
            cost: Some(0.0),
            raw: super::obj(&[("provider", "mock".into()), ("map", format!("{:?}", req.map).into())]),
        })
    }

    /// Offline "upscale": a plain Lanczos resize to the target side — free, keyless, and
    /// keeps the whole Refine flow exercisable (and testable) without a network.
    async fn upscale(
        &self,
        req: &super::UpscaleRequest,
        _cancel: tokio_util::sync::CancellationToken,
        progress: &super::ProgressFn,
    ) -> Result<ImageResponse, ProviderError> {
        progress(JobProgress::Submitted);
        progress(JobProgress::Running);
        let (rgba, side, target) = (req.rgba.clone(), req.side, req.target_side);
        let out = tokio::task::spawn_blocking(move || maps::resize_square_rgba(&rgba, side, target))
            .await
            .map_err(|e| ProviderError::Job(format!("mock upscale join: {e}")))?;
        progress(JobProgress::Done);
        Ok(ImageResponse {
            images: vec![OutputImage {
                data: ImageData::Base64 {
                    data: out,
                    mime: "image/png".into(),
                },
                width: Some(target as u32),
                height: Some(target as u32),
                content_type: Some("image/png".into()),
            }],
            seed: None,
            cost: Some(0.0),
            raw: super::obj(&[("provider", "mock".into()), ("upscale", true.into())]),
        })
    }

    /// Offline "estimate": derive a plausible depth/normal from the albedo so the entire
    /// depth→macro / normal→micro pipeline runs keyless. Depth maps derive a grayscale height;
    /// a normals request derives the tangent-space normal. Returned as a real PNG so the
    /// decode path is identical to the hosted endpoints'.
    async fn estimate(
        &self,
        req: &EstimateRequest,
        _cancel: tokio_util::sync::CancellationToken,
        progress: &super::ProgressFn,
    ) -> Result<ImageResponse, ProviderError> {
        progress(JobProgress::Submitted);
        progress(JobProgress::Running);
        let (rgba, side, map) = (req.rgba.clone(), req.side, req.map);
        let bytes = tokio::task::spawn_blocking(move || {
            let params = derive::DeriveParams::default();
            let out = if map == MapId::Normal {
                derive::derive_map(MapId::Normal, &rgba, None, side, &params)
            } else {
                derive::displacement_height_field_from_albedo(&rgba, side, &params).to_rgba8(false, 1.0, false)
            };
            maps::encode_png_rgba(&out, side)
        })
        .await
        .map_err(|e| ProviderError::Job(format!("mock estimate join: {e}")))?
        .ok_or_else(|| ProviderError::Job("mock estimate PNG encode failed".into()))?;
        progress(JobProgress::Done);
        Ok(ImageResponse {
            images: vec![OutputImage {
                data: ImageData::Base64 {
                    data: bytes,
                    mime: "image/png".into(),
                },
                width: Some(req.side as u32),
                height: Some(req.side as u32),
                content_type: Some("image/png".into()),
            }],
            seed: None,
            cost: Some(0.0),
            raw: super::obj(&[("provider", "mock".into()), ("estimate", format!("{:?}", req.map).into())]),
        })
    }

    fn id(&self) -> ProviderId {
        ProviderId::Mock
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The offline estimator returns a decodable PNG for both a depth (height) and a normals
    /// request, so the whole depth pipeline is exercisable with no key or network.
    #[tokio::test]
    async fn mock_estimate_returns_decodable_png_for_depth_and_normal() {
        let side = maps::TEX_SIZE;
        let albedo = vec![120u8; side * side * 4];
        for map in [MapId::Displacement, MapId::Normal] {
            let req = EstimateRequest {
                map,
                rgba: albedo.clone(),
                side,
                model: "mock".into(),
                extra: serde_json::Value::Null,
                output: crate::providers::EstimateOutput::Image,
            };
            let progress = |_p: JobProgress| {};
            let resp = MockProvider
                .estimate(&req, tokio_util::sync::CancellationToken::new(), &progress)
                .await
                .expect("mock estimate succeeds");
            let img = resp.images.into_iter().next().expect("one image");
            match img.data {
                ImageData::Base64 { data, mime } => {
                    assert_eq!(mime, "image/png");
                    assert!(image::load_from_memory(&data).is_ok(), "{map:?} estimate must be a decodable PNG");
                }
                ImageData::Url(u) => panic!("expected inline base64, got url {u}"),
            }
        }
    }
}
