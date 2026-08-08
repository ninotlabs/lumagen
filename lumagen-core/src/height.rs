//! High-precision height fields — the `f32` representation that carries **true** geometric
//! precision from derivation/estimation all the way to the encoder, so displacement (and the
//! normal/AO derived from it) is quantized to 16-bit or float exactly **once**, at export.
//!
//! This is what makes "16-bit" real instead of an 8-bit value widened by ×257: an 8-bit
//! source has 256 distinct levels no matter how wide the container, so promoting it to 16-bit
//! invents nothing and the displacement still terraces. Keep the field in `f32` from birth
//! (a derived luminance-blur, or a depth model's affine output) and the 65,536 levels a
//! 16-bit PNG can hold are genuinely distinct — smooth relief, no banding.
//!
//! The UI/display path stays RGBA8 (a monitor can't show 16-bit anyway); this type is the
//! parallel high-precision buffer used for the geometry maps and their export, mirroring the
//! `MasterPixels` pattern the provider path already uses for full-resolution color masters.

use std::path::Path;

use crate::error::{Error, Result};

/// A single-channel height/displacement field in normalized `0.0..=1.0`, stored as `side²`
/// `f32` samples in row-major order. The canonical high-precision geometric buffer:
/// derivation and depth estimation both produce one, normal/AO derive from it, and export
/// quantizes it once (16-bit PNG or 32-bit-float EXR).
#[derive(Clone, Debug, PartialEq)]
pub struct HeightField {
    /// `side²` samples, row-major, nominally in `0.0..=1.0` (callers may hold out-of-range
    /// values transiently; every encoder clamps at the quantization boundary).
    data: Vec<f32>,
    /// Side length; the field is always square.
    side: usize,
}

/// 4×4 Bayer threshold matrix (values `0..16`) for ordered dithering when a height must be
/// quantized to 8-bit — it trades banding for imperceptible high-frequency noise, so a smooth
/// gradient that would terrace at 8-bit stays visually smooth. Unused on the 16-bit/EXR paths,
/// where 65,536+ levels make dithering pointless.
const BAYER4: [[f32; 4]; 4] = [[0.0, 8.0, 2.0, 10.0], [12.0, 4.0, 14.0, 6.0], [3.0, 11.0, 1.0, 9.0], [15.0, 7.0, 13.0, 5.0]];

/// The Bayer threshold at `(x, y)` normalized to `0.0..1.0`.
fn bayer4(x: usize, y: usize) -> f32 {
    BAYER4[y % 4][x % 4] / 16.0
}

/// Smoothstep on `0..1` — cubic ease used by the seam blend so the crossfade has no visible
/// kink where the band meets untouched interior.
fn smoothstep(x: f32) -> f32 {
    x * x * (3.0 - 2.0 * x)
}

/// One separable box-blur pass (horizontal then vertical) with edge-clamped windows, computed
/// with a sliding running sum so cost is O(side²) regardless of `radius`. Both window bounds
/// are monotonic in the scan index, so the running sum only ever sheds on the left and gains
/// on the right.
fn box_blur_separable(src: &[f32], side: usize, radius: usize) -> Vec<f32> {
    let n = side as i64;
    let r = radius as i64;
    let window = |i: i64| ((i - r).max(0), (i + r).min(n - 1));

    // Horizontal pass over contiguous rows.
    let mut tmp = vec![0.0f32; side * side];
    for y in 0..side {
        let base = y * side;
        let (mut lo, mut hi) = window(0);
        let mut sum: f32 = (lo..=hi).map(|x| src[base + x as usize]).sum();
        for x in 0..n {
            let (nlo, nhi) = window(x);
            while lo < nlo {
                sum -= src[base + lo as usize];
                lo += 1;
            }
            while hi < nhi {
                hi += 1;
                sum += src[base + hi as usize];
            }
            tmp[base + x as usize] = sum / (hi - lo + 1) as f32;
        }
    }

    // Vertical pass over strided columns.
    let mut dst = vec![0.0f32; side * side];
    for x in 0..side {
        let (mut lo, mut hi) = window(0);
        let mut sum: f32 = (lo..=hi).map(|y| tmp[y as usize * side + x]).sum();
        for y in 0..n {
            let (nlo, nhi) = window(y);
            while lo < nlo {
                sum -= tmp[lo as usize * side + x];
                lo += 1;
            }
            while hi < nhi {
                hi += 1;
                sum += tmp[hi as usize * side + x];
            }
            dst[y as usize * side + x] = sum / (hi - lo + 1) as f32;
        }
    }
    dst
}

impl HeightField {
    /// Wrap `data` as a `side²` field. Returns `None` if the length doesn't match `side²`, so
    /// an inconsistent field can never be constructed silently.
    pub fn from_vec(data: Vec<f32>, side: usize) -> Option<Self> {
        (data.len() == side * side).then_some(Self { data, side })
    }

    /// A flat field of `value` at `side²`.
    pub fn filled(side: usize, value: f32) -> Self {
        Self {
            data: vec![value; side * side],
            side,
        }
    }

    /// Side length (the field is square).
    pub fn side(&self) -> usize {
        self.side
    }

    /// The raw `f32` samples, row-major.
    pub fn samples(&self) -> &[f32] {
        &self.data
    }

    /// Resample to `out × out` with a high-quality convolution (Lanczos3 via SIMD
    /// `fast_image_resize`, the same filter/backend the color path uses so preview and export
    /// agree), clamping any filter overshoot back into `0..1`. A no-op when already `out`;
    /// falls back to the source unscaled if the resizer rejects the buffer (never panics).
    #[must_use]
    pub fn resized(&self, out: usize) -> Self {
        if out == 0 || out == self.side {
            return self.clone();
        }
        use fast_image_resize as fr;
        // `fast_image_resize` works on byte buffers with a declared pixel type; F32 is the raw
        // native-endian bytes of each sample. The buffer is produced and consumed in-process,
        // so native endianness round-trips exactly.
        let bytes: Vec<u8> = self.data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let Ok(src) = fr::images::Image::from_vec_u8(self.side as u32, self.side as u32, bytes, fr::PixelType::F32) else {
            return self.clone();
        };
        let mut dst = fr::images::Image::new(out as u32, out as u32, fr::PixelType::F32);
        let mut resizer = fr::Resizer::new();
        if resizer
            .resize(
                &src,
                &mut dst,
                &fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Lanczos3)),
            )
            .is_err()
        {
            return self.clone();
        }
        let data: Vec<f32> = dst
            .into_vec()
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]).clamp(0.0, 1.0))
            .collect();
        Self { data, side: out }
    }

    /// Robustly renormalize to fill `0..1`, mapping the `[lo_pct, hi_pct]` percentile band onto
    /// the full range (percentiles, not min/max, so a few outlier samples don't crush the
    /// usable contrast). This is how a depth model's affine-invariant output — arbitrary scale
    /// and offset per image — becomes a usable height. A degenerate (flat) field is returned
    /// unchanged.
    #[must_use]
    pub fn normalized_robust(&self, lo_pct: f32, hi_pct: f32) -> Self {
        if self.data.is_empty() {
            return self.clone();
        }
        let mut sorted = self.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let pick = |p: f32| sorted[((p.clamp(0.0, 1.0) * (n - 1) as f32).round() as usize).min(n - 1)];
        let lo = pick(lo_pct);
        let hi = pick(hi_pct);
        let span = hi - lo;
        if span <= f32::EPSILON {
            return self.clone();
        }
        let data = self.data.iter().map(|v| ((v - lo) / span).clamp(0.0, 1.0)).collect();
        Self { data, side: self.side }
    }

    /// Invert the height in place (`h → 1 - h`) — for depth models whose near/far convention
    /// reads raised where the surface should recede.
    pub fn invert(&mut self) {
        for v in &mut self.data {
            *v = 1.0 - *v;
        }
    }

    /// Low-pass the field to its **macro** structure: three iterations of a separable
    /// sliding-window box blur (≈ Gaussian) at `radius` pixels. `0` is a no-op. This is the
    /// operation that turns an 8-bit-sourced depth into smooth `f32` macro relief — averaging
    /// many samples yields genuine sub-LSB precision, and the quantized high frequencies (where
    /// 8-bit banding lives) are discarded here, to be carried instead by the micro normal.
    #[must_use]
    pub fn low_pass(&self, radius: usize) -> Self {
        if radius == 0 || self.side < 2 {
            return self.clone();
        }
        let mut data = self.data.clone();
        for _ in 0..3 {
            data = box_blur_separable(&data, self.side, radius);
        }
        Self { data, side: self.side }
    }

    /// The high-frequency residual `self - low_pass(radius)`, centered at `0.5` — the detail a
    /// low-pass discards. Used to graft one depth source's fine macro detail onto another's
    /// robust global structure (best-of-both), and to derive a micro signal.
    #[must_use]
    pub fn high_pass(&self, radius: usize) -> Self {
        let low = self.low_pass(radius);
        let data = self.data.iter().zip(&low.data).map(|(v, l)| (v - l + 0.5).clamp(0.0, 1.0)).collect();
        Self { data, side: self.side }
    }

    /// Blend narrow opposite border bands so the tile wraps exactly (the `f32` port of
    /// `export::make_seamless`): each band crossfades toward the average of itself and the
    /// mirrored opposite band, then the outermost row/column pairs are forced identical.
    /// Operates in float so the seam fix costs no precision.
    pub fn make_seamless(&mut self, margin: usize) {
        let size = self.side;
        if margin == 0 || size < 8 {
            return;
        }
        let m = margin.min(size / 4).max(1);
        let denom = (m.saturating_sub(1)).max(1) as f32;
        let idx = |x: usize, y: usize| y * size + x;
        for horizontal in [true, false] {
            for lane in 0..size {
                for i in 0..m {
                    let t = smoothstep(i as f32 / denom);
                    let (near, far) = if horizontal {
                        (idx(i, lane), idx(size - 1 - i, lane))
                    } else {
                        (idx(lane, i), idx(lane, size - 1 - i))
                    };
                    let a = self.data[near];
                    let b = self.data[far];
                    let common = 0.5 * (a + b);
                    self.data[near] = common * (1.0 - t) + a * t;
                    self.data[far] = common * (1.0 - t) + b * t;
                }
            }
        }
        for horizontal in [true, false] {
            for lane in 0..size {
                let (near, far) = if horizontal {
                    (idx(0, lane), idx(size - 1, lane))
                } else {
                    (idx(lane, 0), idx(lane, size - 1))
                };
                let e = 0.5 * (self.data[near] + self.data[far]);
                self.data[near] = e;
                self.data[far] = e;
            }
        }
    }

    /// Quantize to an 8-bit grayscale RGBA buffer for the UI preview (and the 8-bit
    /// displacement fallback), honoring `invert`/`strength`. Ordered (Bayer) dithering breaks
    /// up the 8-bit banding so the preview matches what the 16-bit export actually looks like;
    /// pass `dither = false` for an exact, reproducible mapping.
    pub fn to_rgba8(&self, invert: bool, strength: f32, dither: bool) -> Vec<u8> {
        let size = self.side;
        let mut out = vec![0u8; size * size * 4];
        for (i, p) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let mut v = self.data[i] * strength;
            if invert {
                v = 1.0 - v;
            }
            let mut scaled = v.clamp(0.0, 1.0) * 255.0;
            if dither {
                scaled += bayer4(i % size, i / size) - 0.5;
            }
            let g = scaled.round().clamp(0.0, 255.0) as u8;
            *p = [g, g, g, 255];
        }
        out
    }

    /// Encode as a true 16-bit grayscale PNG (65,536 real levels) to any sink. Big-endian
    /// samples per the PNG spec; values clamped to `0..1` at the boundary. `finish()` is
    /// explicit so a truncated final write surfaces as an error rather than being swallowed by
    /// `Drop`.
    fn encode_png16<W: std::io::Write>(&self, sink: W) -> Result<()> {
        let size = self.side;
        let mut encoder = png::Encoder::new(sink, size as u32, size as u32);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Sixteen);
        // Height is linear data, not color — signal gamma 1.0 so importers don't sRGB-decode it.
        encoder.set_source_gamma(png::ScaledFloat::from_scaled(100_000));
        let mut writer = encoder.write_header().map_err(|e| Error::Encode(e.to_string()))?;
        let mut data = vec![0u8; size * size * 2];
        for (out, v) in data.chunks_exact_mut(2).zip(&self.data) {
            let q = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
            out.copy_from_slice(&q.to_be_bytes());
        }
        writer.write_image_data(&data).map_err(|e| Error::Encode(e.to_string()))?;
        writer.finish().map_err(|e| Error::Encode(e.to_string()))
    }

    /// Write a true 16-bit grayscale PNG (the Unreal/Unity displacement fallback when EXR
    /// isn't wanted).
    pub fn write_png16(&self, path: &Path) -> Result<()> {
        self.encode_png16(std::fs::File::create(path).map_err(Error::Io)?)
    }

    /// Encode a true 16-bit grayscale PNG to an in-memory byte vector.
    pub fn to_png16_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.encode_png16(&mut out)?;
        Ok(out)
    }

    /// Decode a grayscale PNG back into a field at its native side, values **as stored**
    /// (`0..1`, NOT renormalized) — the inverse of [`Self::to_png16_bytes`] for document reload,
    /// preserving 16-bit precision via `to_luma16`. `None` if the bytes aren't a decodable image.
    pub fn from_png_bytes(bytes: &[u8]) -> Option<Self> {
        let img = image::load_from_memory(bytes).ok()?;
        let side = img.width().min(img.height()) as usize;
        if side == 0 {
            return None;
        }
        let square = if img.width() == img.height() {
            img
        } else {
            img.resize_exact(side as u32, side as u32, image::imageops::FilterType::Lanczos3)
        };
        let luma = square.to_luma16();
        let data: Vec<f32> = luma.as_raw().iter().map(|&v| v as f32 / 65535.0).collect();
        Self::from_vec(data, side)
    }

    /// Write a 32-bit-float grayscale EXR (the default displacement format — the precision
    /// Blender's adaptive displacement and a clean normal/AO derivation want). The three RGB
    /// channels carry the same height so any DCC reads it as luminance.
    pub fn write_exr(&self, path: &Path) -> Result<()> {
        let size = self.side;
        let data = &self.data;
        exr::prelude::write_rgb_file(path, size, size, |x, y| {
            let v = data.get(y * size + x).copied().unwrap_or(0.0);
            (v, v, v)
        })
        .map_err(|e| Error::Encode(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// A horizontal `0..1` ramp `side` columns wide (at most `side` distinct levels).
    fn ramp(side: usize) -> HeightField {
        let mut data = vec![0.0f32; side * side];
        for y in 0..side {
            for x in 0..side {
                data[y * side + x] = x as f32 / (side - 1) as f32;
            }
        }
        HeightField::from_vec(data, side).expect("ramp is square")
    }

    /// A monotone `0..1` gradient over every sample — `side²` distinct input levels, so the
    /// encoder's precision (not the input's) is what caps the output level count.
    fn fine_gradient(side: usize) -> HeightField {
        let n = (side * side) as f32;
        let mut data = vec![0.0f32; side * side];
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f32 / (n - 1.0);
        }
        HeightField::from_vec(data, side).expect("square")
    }

    /// The whole point: a smooth ramp encoded as 16-bit PNG carries **far more than 256**
    /// distinct levels — the precision the old `8-bit ×257` path could never hold. Decoded
    /// back through the `png` crate to prove the file really is 16-bit, not just labeled so.
    #[test]
    fn png16_has_far_more_than_256_levels() {
        let side = 256;
        let hf = fine_gradient(side);
        let bytes = hf.to_png16_bytes().expect("encode");

        let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        let mut reader = decoder.read_info().expect("read info");
        let mut buf = vec![0u8; reader.output_buffer_size().expect("buffer size")];
        let info = reader.next_frame(&mut buf).expect("frame");
        assert_eq!(info.bit_depth, png::BitDepth::Sixteen, "must be a real 16-bit PNG");
        assert_eq!(info.color_type, png::ColorType::Grayscale);

        let valid = info.line_size * info.height as usize;
        let real_levels: HashSet<u16> = buf[..valid].chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
        assert!(real_levels.len() > 256, "true 16-bit ramp has {} levels", real_levels.len());

        // Contrast: the discredited "widen 8-bit by ×257" path can only ever produce ≤256
        // distinct values across the identical ramp — this is the banding we're eliminating.
        let fake_levels: HashSet<u16> = hf.samples().iter().map(|v| ((v.clamp(0.0, 1.0) * 255.0).round() as u16) * 257).collect();
        assert!(fake_levels.len() <= 256, "×257 path is capped at 256, got {}", fake_levels.len());
        assert!(real_levels.len() > fake_levels.len() * 4, "16-bit must resolve far more than the 8-bit path");
    }

    /// Robust normalization maps an arbitrary affine range (a depth model's output) onto
    /// `0..1` without letting outliers crush contrast.
    #[test]
    fn normalized_robust_fills_unit_range() {
        // Values centered around 100 with a span of ~4, plus one wild outlier.
        let side = 16;
        let mut data = vec![0.0f32; side * side];
        for (i, v) in data.iter_mut().enumerate() {
            *v = 98.0 + (i % side) as f32 / (side as f32) * 4.0;
        }
        data[0] = 1000.0; // outlier the percentile clamp must ignore
        let hf = HeightField::from_vec(data, side).expect("square");
        let norm = hf.normalized_robust(0.01, 0.99);
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for &v in norm.samples() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(lo <= 0.001, "low end reaches 0, got {lo}");
        assert!(hi >= 0.999, "high end reaches 1, got {hi}");
    }

    /// The float seam blend leaves opposite borders exactly equal, so the exported tile wraps.
    #[test]
    fn seamless_makes_opposite_borders_equal() {
        let side = 32;
        let mut data = vec![0.0f32; side * side];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i * 37) % 251) as f32 / 251.0;
        }
        let mut hf = HeightField::from_vec(data, side).expect("square");
        hf.make_seamless(8);
        let s = hf.samples();
        for x in 0..side {
            assert!((s[x] - s[(side - 1) * side + x]).abs() < f32::EPSILON, "row edge mismatch at x={x}");
        }
        for y in 0..side {
            assert!((s[y * side] - s[y * side + side - 1]).abs() < f32::EPSILON, "col edge mismatch at y={y}");
        }
    }

    /// Resizing preserves the square invariant and keeps samples in range (no Lanczos
    /// overshoot leaks past the clamp).
    #[test]
    fn resized_keeps_invariant_and_range() {
        let hf = ramp(64);
        let up = hf.resized(256);
        assert_eq!(up.side(), 256);
        assert_eq!(up.samples().len(), 256 * 256);
        assert!(up.samples().iter().all(|v| (0.0..=1.0).contains(v)), "resample stays in 0..1");
        // No-op path returns an equal field.
        assert_eq!(hf.resized(64), hf);
    }

    /// EXR export writes a non-empty float file.
    #[test]
    fn exr_export_writes_a_file() {
        let hf = ramp(16);
        let path = std::env::temp_dir().join(format!("lumagen_height_exr_{}.exr", std::process::id()));
        hf.write_exr(&path).expect("write exr");
        let len = std::fs::metadata(&path).expect("exr exists").len();
        assert!(len > 0, "exr must be non-empty");
        let _ = std::fs::remove_file(&path);
    }

    /// Dithered 8-bit quantization stays in bounds and honors invert.
    #[test]
    fn to_rgba8_dither_in_bounds_and_inverts() {
        let hf = ramp(32);
        let plain = hf.to_rgba8(false, 1.0, false);
        let inverted = hf.to_rgba8(true, 1.0, false);
        assert_eq!(plain.len(), 32 * 32 * 4);
        // First pixel is 0.0 → black plain, white inverted.
        assert_eq!(&plain[0..4], &[0, 0, 0, 255]);
        assert_eq!(&inverted[0..4], &[255, 255, 255, 255]);
        // Dithered path never leaves the byte range.
        let dithered = hf.to_rgba8(false, 1.0, true);
        assert_eq!(dithered.len(), 32 * 32 * 4);
    }

    /// Low-pass removes high-frequency detail (toward the local mean) and leaves a flat field
    /// unchanged.
    #[test]
    fn low_pass_smooths_and_preserves_flat() {
        let side = 32;
        let mut data = vec![0.0f32; side * side];
        for (i, v) in data.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 0.0 } else { 1.0 }; // max high-frequency
        }
        let hf = HeightField::from_vec(data, side).expect("square");
        let lp = hf.low_pass(4);
        let mid = 16 * side + 16;
        assert!(
            (lp.samples()[mid] - 0.5).abs() < 0.15,
            "high freq flattened toward mean, got {}",
            lp.samples()[mid]
        );
        // A constant field is (numerically) unchanged.
        let flat = HeightField::filled(side, 0.3);
        assert!(flat.low_pass(5).samples().iter().all(|v| (v - 0.3).abs() < 1e-4), "flat stays flat");
    }

    /// Length/side mismatches can't build a field.
    #[test]
    fn from_vec_rejects_mismatched_length() {
        assert!(HeightField::from_vec(vec![0.0; 10], 4).is_none());
        assert!(HeightField::from_vec(vec![0.0; 16], 4).is_some());
    }
}
