//! Local Depth Anything V2 depth via [`tract`](https://github.com/sonos/tract) — pure-Rust
//! ONNX inference, so this is an **offline, true-raw-float** depth source with no hosted 8-bit
//! round-trip and no native runtime to link. Feature-gated (`onnx`): the hosted Marigold /
//! Depth Anything path needs none of this.
//!
//! The model file (a Depth Anything V2 ONNX export) is supplied at runtime via
//! `DepthReconstruct.local_model_path` — placed in the cache or downloaded once. Inference runs
//! on CPU; the loaded model is cached per path so repeated materials don't re-optimize it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use tract_onnx::prelude::*;

use crate::height::HeightField;
use crate::providers::ProviderError;

/// DINOv2 patch size is 14; Depth Anything V2 wants a square input that's a multiple of it.
/// 518 = 37 × 14 is the standard inference resolution.
const INPUT_SIDE: usize = 518;
/// ImageNet normalization the model was trained with.
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A loaded, optimized, runnable model.
struct Engine {
    model: TypedRunnableModel<TypedModel>,
}

/// Loaded models cached by path — optimizing a model is expensive, and one material generation
/// shouldn't pay it twice.
static ENGINES: OnceLock<Mutex<HashMap<PathBuf, Arc<Engine>>>> = OnceLock::new();

/// Load (or fetch the cached) engine for a model file, pinning the input to `[1,3,518,518]`
/// f32 so tract can fully optimize the graph.
fn engine_for(path: &Path) -> Result<Arc<Engine>, ProviderError> {
    let cache = ENGINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().map_err(|_| ProviderError::Job("onnx engine cache poisoned".into()))?;
    if let Some(engine) = map.get(path) {
        return Ok(Arc::clone(engine));
    }
    if !path.exists() {
        return Err(ProviderError::Job(format!(
            "local depth model not found at {} — place a Depth Anything V2 ONNX export there",
            path.display()
        )));
    }
    let model = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|e| ProviderError::Job(format!("onnx load {}: {e}", path.display())))?
        .with_input_fact(0, f32::fact([1, 3, INPUT_SIDE, INPUT_SIDE]).into())
        .map_err(|e| ProviderError::Job(format!("onnx input fact: {e}")))?
        .into_optimized()
        .map_err(|e| ProviderError::Job(format!("onnx optimize: {e}")))?
        .into_runnable()
        .map_err(|e| ProviderError::Job(format!("onnx runnable: {e}")))?;
    let engine = Arc::new(Engine { model });
    map.insert(path.to_path_buf(), Arc::clone(&engine));
    Ok(engine)
}

/// Preprocess an RGBA8 albedo into the model's `[1,3,518,518]` normalized NCHW input tensor.
fn preprocess(albedo: &[u8], side: usize) -> Result<Tensor, ProviderError> {
    let resized = crate::maps::resize_square_rgba(albedo, side, INPUT_SIDE);
    let (px, _) = resized.as_chunks::<4>();
    let hw = INPUT_SIDE * INPUT_SIDE;
    let mut data = vec![0f32; 3 * hw];
    for (i, p) in px.iter().enumerate() {
        for c in 0..3 {
            let v = p[c] as f32 / 255.0;
            data[c * hw + i] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }
    let tensor: Tensor = tract_ndarray::Array4::from_shape_vec((1, 3, INPUT_SIDE, INPUT_SIDE), data)
        .map_err(|e| ProviderError::Job(format!("onnx input shape: {e}")))?
        .into();
    Ok(tensor)
}

/// Run local Depth Anything V2 on the albedo and return a normalized macro-ready height field
/// at `side`. The model's relative (affine-invariant) depth is robustly renormalized to `0..1`;
/// `invert` flips the near/far convention. Genuine `f32` throughout — no 8-bit round-trip.
pub fn reconstruct_local_depth(model_path: &Path, albedo: &[u8], side: usize, invert: bool) -> Result<HeightField, ProviderError> {
    let engine = engine_for(model_path)?;
    let input = preprocess(albedo, side)?;
    let result = engine
        .model
        .run(tvec!(input.into()))
        .map_err(|e| ProviderError::Job(format!("onnx inference: {e}")))?;
    let view = result
        .first()
        .ok_or_else(|| ProviderError::Job("onnx produced no output".into()))?
        .to_array_view::<f32>()
        .map_err(|e| ProviderError::Job(format!("onnx output: {e}")))?;
    // The output is a single-channel depth map (shape [1,H,W] or [1,1,H,W]); flatten it and
    // recover the square side from the element count.
    let out: Vec<f32> = view.iter().copied().collect();
    let out_side = (out.len() as f64).sqrt().round() as usize;
    let field = HeightField::from_vec(out, out_side).ok_or_else(|| ProviderError::Job(format!("onnx depth output not square ({out_side}² ≠ len)")))?;
    let mut hf = field.resized(side).normalized_robust(0.005, 0.995);
    if invert {
        hf.invert();
    }
    Ok(hf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Preprocessing produces the exact `[1,3,518,518]` NCHW shape the model expects, with
    /// ImageNet normalization applied (a mid-gray pixel maps near each channel's `(0.5-μ)/σ`).
    #[test]
    fn preprocess_shapes_and_normalizes() {
        let side = 64;
        let albedo = vec![128u8; side * side * 4];
        let tensor = preprocess(&albedo, side).expect("preprocess");
        assert_eq!(tensor.shape(), [1, 3, INPUT_SIDE, INPUT_SIDE]);
        let view = tensor.to_array_view::<f32>().expect("f32 view");
        let expected_r = (128.0 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        assert!((view[[0, 0, 0, 0]] - expected_r).abs() < 1e-3, "red channel normalized");
    }

    /// A missing model path fails with a clear, actionable error rather than panicking.
    #[test]
    fn missing_model_reports_a_clear_error() {
        let albedo = vec![10u8; 32 * 32 * 4];
        let err = reconstruct_local_depth(Path::new("does-not-exist.onnx"), &albedo, 32, false).expect_err("missing model must error");
        assert!(matches!(&err, ProviderError::Job(msg) if msg.contains("not found")), "unexpected: {err:?}");
    }
}
