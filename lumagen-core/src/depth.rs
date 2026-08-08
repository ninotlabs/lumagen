//! Depth reconstruction — turn estimator outputs into the **macro** displacement height.
//!
//! Architecture (the depth=macro / normal=micro split): the displacement carries the coarse
//! geometry-scale relief, while the finest surface texture is carried by the normal. So depth
//! reconstruction here produces a low-frequency `f32` [`HeightField`]; the crossover frequency
//! partitions the spectrum — everything below it becomes displacement, everything above it is
//! left for the micro normal, so geometry and shading never double-count the same feature.
//!
//! Two depth models run per material ("run both, pick best"): [`blend_best_of`] keeps the
//! robust global structure from one (Depth Anything) and the finer macro detail from the other
//! (Marigold). Both hosted endpoints return 8-bit — genuine `f32` precision is recovered by the
//! macro low-pass, and the local ONNX engine (opt-in) skips the 8-bit round-trip entirely.

use crate::height::HeightField;

/// Decode an estimator's output image (a Marigold / Depth Anything PNG, or the local ONNX
/// engine's 16-bit render) into a normalized `0..1` [`HeightField`]. Reads full 16-bit
/// precision when the source carries it (`to_luma16`), robustly renormalizes the affine range
/// depth models emit, and applies `invert` for the near/far convention. `None` if the bytes
/// aren't a decodable image.
pub fn decode_depth(bytes: &[u8], invert: bool) -> Option<HeightField> {
    let img = image::load_from_memory(bytes).ok()?;
    let side = img.width().min(img.height()) as usize;
    if side == 0 {
        return None;
    }
    // Estimators mirror the (square) albedo, but guard a non-square return by squaring it.
    let square = if img.width() == img.height() {
        img
    } else {
        img.resize_exact(side as u32, side as u32, image::imageops::FilterType::Lanczos3)
    };
    // to_luma16 preserves a genuine 16-bit source and widens an 8-bit one — honest either way.
    let luma = square.to_luma16();
    let data: Vec<f32> = luma.as_raw().iter().map(|&v| v as f32 / 65535.0).collect();
    let mut hf = HeightField::from_vec(data, side)?;
    hf = hf.normalized_robust(0.005, 0.995);
    if invert {
        hf.invert();
    }
    Some(hf)
}

/// Best-of-both depth: the robust global structure from `global` (Depth Anything) plus the
/// finer macro detail from `detail` (Marigold), split at `blend_radius`. `detail` is resized to
/// `global`'s side if they differ. This is the "run both, pick best" blend — each model
/// contributes the band it's strongest in.
pub fn blend_best_of(global: &HeightField, detail: &HeightField, blend_radius: usize) -> HeightField {
    let detail = if detail.side() == global.side() {
        detail.clone()
    } else {
        detail.resized(global.side())
    };
    let side = global.side();
    let g_low = global.low_pass(blend_radius);
    let d_low = detail.low_pass(blend_radius);
    let (g, d, dl) = (g_low.samples(), detail.samples(), d_low.samples());
    // global's low band + detail's high-band residual.
    let data: Vec<f32> = (0..side * side).map(|i| (g[i] + (d[i] - dl[i])).clamp(0.0, 1.0)).collect();
    HeightField::from_vec(data, side).unwrap_or_else(|| global.clone())
}

/// Reduce a full-spectrum depth to the MACRO band that ships as displacement: everything finer
/// than `crossover_radius` is dropped (to be carried by the micro normal instead). A thin
/// wrapper over [`HeightField::low_pass`] that names the intent in the pipeline.
pub fn to_macro(depth: &HeightField, crossover_radius: usize) -> HeightField {
    depth.low_pass(crossover_radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a grayscale RGBA PNG carrying a horizontal ramp.
    fn ramp_png(side: usize) -> Vec<u8> {
        let mut rgba = vec![0u8; side * side * 4];
        for y in 0..side {
            for x in 0..side {
                let v = (x * 255 / (side - 1)) as u8;
                rgba[(y * side + x) * 4..][..4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        crate::maps::encode_png_rgba(&rgba, side).expect("png")
    }

    /// Decoding reads the gradient, robustly normalizes to `0..1`, and honors `invert`.
    #[test]
    fn decode_depth_normalizes_and_inverts() {
        let side = 32;
        let png = ramp_png(side);
        let hf = decode_depth(&png, false).expect("decode");
        assert_eq!(hf.side(), side);
        assert!(hf.samples()[0] < 0.05, "left edge ~0, got {}", hf.samples()[0]);
        assert!(hf.samples()[side - 1] > 0.95, "right edge ~1, got {}", hf.samples()[side - 1]);
        let inv = decode_depth(&png, true).expect("decode");
        assert!(inv.samples()[0] > 0.95, "invert flips the near/far convention");
        assert!(decode_depth(&[1, 2, 3], false).is_none(), "junk isn't an image");
    }

    /// The blend follows the global model's macro while injecting the detail model's fine
    /// ripple — so a fine feature present only in `detail` survives in the result.
    #[test]
    fn blend_injects_detail_high_frequency() {
        let side = 48;
        // global: smooth ramp (no ripple). detail: same ramp + a fine per-pixel ripple.
        let mut g = vec![0.0f32; side * side];
        let mut d = vec![0.0f32; side * side];
        for y in 0..side {
            for x in 0..side {
                let base = x as f32 / (side - 1) as f32;
                g[y * side + x] = base * 0.8 + 0.1;
                let ripple = if (x + y) % 2 == 0 { 0.06 } else { -0.06 };
                d[y * side + x] = (base * 0.8 + 0.1 + ripple).clamp(0.0, 1.0);
            }
        }
        let global = HeightField::from_vec(g, side).expect("sq");
        let detail = HeightField::from_vec(d, side).expect("sq");
        let blended = blend_best_of(&global, &detail, 4);
        // The high-frequency ripple (present only in detail) must appear in the blend: adjacent
        // samples differ by roughly the ripple amplitude, which the smooth global lacks.
        let i = 10 * side + 10;
        let ripple_in_blend = (blended.samples()[i] - blended.samples()[i + 1]).abs();
        let ripple_in_global = (global.samples()[i] - global.samples()[i + 1]).abs();
        assert!(ripple_in_blend > ripple_in_global + 0.03, "detail ripple must survive the blend");
    }

    /// The macro reduction removes fine ripple, leaving a smooth field near the local mean.
    #[test]
    fn to_macro_removes_fine_ripple() {
        let side = 48;
        let mut data = vec![0.0f32; side * side];
        for (i, v) in data.iter_mut().enumerate() {
            *v = 0.5 + if i % 2 == 0 { 0.2 } else { -0.2 };
        }
        let hf = HeightField::from_vec(data, side).expect("sq");
        let macro_ = to_macro(&hf, 4);
        let i = 20 * side + 20;
        assert!((macro_.samples()[i] - 0.5).abs() < 0.08, "fine ripple removed, got {}", macro_.samples()[i]);
    }
}
