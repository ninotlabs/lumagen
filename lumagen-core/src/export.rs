//! Export: writes the map set to disk as per-map PNGs (+ a 16-bit PNG or 32-bit-float EXR
//! displacement), a contact sheet, and a `manifest.json`. Honors the engine preset
//! (normal-convention green-flip, ORM/MaskMap channel packing), the `filename_pattern`
//! tokens (`$material_$map_$res`), the requested final size (SIMD resize), and sRGB-vs-linear
//! per map. Runs off the UI thread.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use crate::data::{MAPS, MapId, map_def};
use crate::error::{Error, Result};

/// Engine export convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineConvention {
    /// Separate maps, OpenGL normals.
    Blender,
    /// ORM packed (R=AO, G=Roughness, B=Metallic), DirectX normals.
    Unreal,
    /// MaskMap (Metallic+Smoothness), OpenGL normals.
    UnityUrp,
    /// MaskMap (AO/Smooth/Metallic), OpenGL normals.
    UnityHdrp,
    /// metalRough packed, OpenGL normals.
    Gltf,
    /// Separate maps, OpenGL normals.
    Generic,
}

impl EngineConvention {
    /// From the display name used in Settings.
    pub fn from_name(name: &str) -> Self {
        match name {
            "Unreal" => EngineConvention::Unreal,
            "Unity URP" => EngineConvention::UnityUrp,
            "Unity HDRP" => EngineConvention::UnityHdrp,
            "glTF" => EngineConvention::Gltf,
            "Generic" => EngineConvention::Generic,
            _ => EngineConvention::Blender,
        }
    }

    /// Whether normals should be flipped to DirectX (green−).
    pub const fn directx_normal(self) -> bool {
        matches!(self, EngineConvention::Unreal)
    }

    /// Human display name for the manifest/README.
    pub const fn display_name(self) -> &'static str {
        match self {
            EngineConvention::Blender => "Blender",
            EngineConvention::Unreal => "Unreal",
            EngineConvention::UnityUrp => "Unity URP",
            EngineConvention::UnityHdrp => "Unity HDRP",
            EngineConvention::Gltf => "glTF",
            EngineConvention::Generic => "Generic",
        }
    }

    /// The channel-packing scheme this engine uses (for the manifest/README).
    pub const fn packing(self) -> &'static str {
        match self {
            EngineConvention::Unreal => "ORM (R=AO, G=Roughness, B=Metallic)",
            EngineConvention::UnityUrp | EngineConvention::UnityHdrp => "MaskMap (R=Metallic, G=Occlusion, B=0, A=Smoothness)",
            EngineConvention::Gltf => "metalRough (G=Roughness, B=Metallic)",
            EngineConvention::Blender | EngineConvention::Generic => "Separate maps",
        }
    }

    /// The normal-map convention (for the manifest/README).
    pub const fn normal_name(self) -> &'static str {
        normal_convention(self.directx_normal())
    }
}

/// Display string for a normal convention (`ExportJob::directx_normal` can override the
/// engine default, so the manifest/README key off the effective flag, not the engine).
const fn normal_convention(directx: bool) -> &'static str {
    if directx { "DirectX (Y-)" } else { "OpenGL (Y+)" }
}

/// Everything needed for one export run.
pub struct ExportJob {
    /// The 8 maps' RGBA8 pixels (any may be empty if not generated).
    pub maps: [Option<Vec<u8>>; 8],
    /// Per-map source side length. `0` falls back to `src_size` — maps generated at full
    /// model resolution export from their native-size masters, not the working copy.
    pub map_sizes: [usize; 8],
    /// Working size of the input maps (square) — the fallback source size.
    pub src_size: usize,
    /// Output size (square) — resize if different from `src_size`.
    pub out_size: usize,
    /// Material name for the `$material` token.
    pub material: String,
    /// Resolution string for the `$res` token (e.g. "4k").
    pub res: String,
    /// Filename pattern with `$material`/`$map`/`$res` tokens.
    pub pattern: String,
    /// Engine convention.
    pub engine: EngineConvention,
    /// Whether to flip normals to DirectX (Y−). Defaults from the engine preset but can be
    /// overridden by the explicit Settings → Export normal-convention choice.
    pub directx_normal: bool,
    /// Displacement as 16-bit PNG (vs 8-bit) when not writing EXR.
    pub displacement_16bit_png: bool,
    /// Displacement as 32-bit-float EXR (the default for Blender/Generic); 16-bit PNG otherwise.
    pub displacement_exr: bool,
    /// High-precision displacement source (0..1 `f32`) from the derivation/depth path. When
    /// present, the displacement map is encoded from this — a **true** 16-bit PNG / 32-bit-float
    /// EXR — instead of the 8-bit RGBA in `maps`, which can only ever carry 256 levels. `None`
    /// falls back to the 8-bit buffer (imported/materialized displacement with no float master).
    pub displacement_height: Option<crate::height::HeightField>,
    /// Shared project seed (recorded in the manifest).
    pub seed: u32,
    /// The compiled prompt / description used (recorded in the manifest for provenance).
    pub prompt: String,
    /// Border band (px, at `out_size`) blended so opposite edges wrap exactly; 0 disables.
    pub seam_margin: usize,
}

/// One map's entry in the manifest.
#[derive(serde::Serialize)]
struct ManifestMap {
    map: String,
    file: String,
    colorspace: String,
    bit_depth: u8,
}

/// The manifest describing what was written.
#[derive(serde::Serialize)]
struct Manifest<'a> {
    material: &'a str,
    engine: String,
    normal_convention: String,
    channel_packing: String,
    resolution: String,
    seed: u32,
    prompt: &'a str,
    maps: Vec<ManifestMap>,
    extra_files: Vec<String>,
}

/// Expand `$material`/`$map`/`$res` tokens in the filename pattern into a sanitized stem.
fn expand(pattern: &str, material: &str, map: &str, res: &str) -> String {
    sanitize_stem(&pattern.replace("$material", material).replace("$map", map).replace("$res", res))
}

/// Make an expanded filename stem safe to join under the chosen export folder. Pattern and
/// material name round-trip through `.lumagen` documents unsanitized, so a loaded document
/// could otherwise escape `out_dir` (`..`, `C:\..` — `Path::join` replaces the base on an
/// absolute component) or fail `File::create` mid-export (`<>:"|?*`, reserved device names).
fn sanitize_stem(stem: &str) -> String {
    // Reserved Win32 device names — reserved with any extension, so `con.png` is included.
    const RESERVED: [&str; 22] = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6",
        "LPT7", "LPT8", "LPT9",
    ];
    let mut s: String = stem
        .chars()
        .map(|c| {
            if matches!(c, '/' | '\\' | '<' | '>' | ':' | '"' | '|' | '?' | '*') || c.is_control() {
                '_'
            } else {
                c
            }
        })
        .collect();
    // Win32 strips trailing dots/spaces at create time, silently renaming the file.
    s.truncate(s.trim_end_matches(['.', ' ']).len());
    // A device name is reserved by its prefix before the first dot, case-insensitively.
    let device = s.split('.').next().unwrap_or("");
    if RESERVED.iter().any(|r| device.eq_ignore_ascii_case(r)) {
        s.insert(0, '_');
    }
    if s.is_empty() { "untitled".to_owned() } else { s }
}

/// Resize an RGBA8 buffer to `out` (Lanczos3 via fast_image_resize when sizes differ).
fn resize_rgba(px: &[u8], src: usize, out: usize) -> Result<Vec<u8>> {
    if src == out {
        return Ok(px.to_vec());
    }
    use fast_image_resize as fr;
    let src_img = fr::images::Image::from_vec_u8(src as u32, src as u32, px.to_vec(), fr::PixelType::U8x4).map_err(|e| Error::Encode(e.to_string()))?;
    let mut dst = fr::images::Image::new(out as u32, out as u32, fr::PixelType::U8x4);
    let mut resizer = fr::Resizer::new();
    resizer
        .resize(
            &src_img,
            &mut dst,
            &fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Lanczos3)),
        )
        .map_err(|e| Error::Encode(e.to_string()))?;
    Ok(dst.into_vec())
}

/// Write an RGBA8 buffer as an 8-bit PNG. Color maps are tagged sRGB; data maps are left
/// untagged (linear) so an engine's import pipeline reads them correctly.
fn write_png(path: &Path, rgba: &[u8], size: usize, srgb: bool) -> Result<()> {
    let mut encoder = png::Encoder::new(std::fs::File::create(path).map_err(Error::Io)?, size as u32, size as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    if srgb {
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    } else {
        // Linear/data map: signal gamma 1.0 so tools don't apply an sRGB decode.
        encoder.set_source_gamma(png::ScaledFloat::from_scaled(100_000));
    }
    finish_png(encoder, rgba)
}

/// Write the header + one frame and `finish()`: Drop swallows the final IEND write's error,
/// so only an explicit `finish()` surfaces a truncated file.
fn finish_png<W: std::io::Write>(encoder: png::Encoder<'_, W>, data: &[u8]) -> Result<()> {
    let mut writer = encoder.write_header().map_err(|e| Error::Encode(e.to_string()))?;
    writer.write_image_data(data).map_err(|e| Error::Encode(e.to_string()))?;
    writer.finish().map_err(|e| Error::Encode(e.to_string()))
}

/// Write a height map as a 32-bit-float grayscale EXR (the default displacement format — the
/// precision Blender's adaptive displacement and a clean normal/AO derivation want). The 8-bit
/// luma is promoted to normalized float; a true high-precision source would carry more bits.
fn write_exr(path: &Path, height_rgba: &[u8], size: usize) -> Result<()> {
    let (px, _) = height_rgba.as_chunks::<4>();
    exr::prelude::write_rgb_file(path, size, size, |x, y| {
        let v = px.get(y * size + x).map_or(0.0f32, |p| p[0] as f32 / 255.0);
        (v, v, v)
    })
    .map_err(|e| Error::Encode(e.to_string()))?;
    Ok(())
}

/// Composite a grid of every present map into a single contact-sheet PNG (256 px cells).
fn write_contact_sheet(path: &Path, resized: &[Option<Vec<u8>>; 8], out_size: usize) -> Result<()> {
    let present: Vec<&Vec<u8>> = MAPS.iter().filter_map(|d| resized[d.id.index()].as_ref()).collect();
    if present.is_empty() {
        return Ok(());
    }
    let cols = 4usize;
    let rows = present.len().div_ceil(cols);
    let cell = 256usize;
    let sheet_w = cols * cell;
    let sheet_h = rows * cell;
    let mut sheet = vec![0u8; sheet_w * sheet_h * 4];
    for p in sheet.as_chunks_mut::<4>().0 {
        *p = [24, 26, 30, 255];
    }
    for (slot, px) in present.iter().enumerate() {
        let thumb = resize_rgba(px, out_size, cell)?;
        let cx = (slot % cols) * cell;
        let cy = (slot / cols) * cell;
        for (row, src) in thumb.chunks_exact(cell * 4).enumerate() {
            let dst = ((cy + row) * sheet_w + cx) * 4;
            if let Some(dst_row) = sheet.get_mut(dst..dst + cell * 4) {
                dst_row.copy_from_slice(src);
            }
        }
    }
    let mut encoder = png::Encoder::new(std::fs::File::create(path).map_err(Error::Io)?, sheet_w as u32, sheet_h as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    finish_png(encoder, &sheet)
}

/// Write a height map as a 16-bit grayscale PNG (the Unreal/Unity displacement fallback).
fn write_png16(path: &Path, height_rgba: &[u8], size: usize) -> Result<()> {
    let mut encoder = png::Encoder::new(std::fs::File::create(path).map_err(Error::Io)?, size as u32, size as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Sixteen);
    // Expand 8-bit luma to 16-bit (×257 maps 0..255 onto 0..65535), big-endian per PNG.
    let mut data = vec![0u8; size * size * 2];
    let (px, _) = height_rgba.as_chunks::<4>();
    for (out, p) in data.as_chunks_mut::<2>().0.iter_mut().zip(px) {
        *out = ((p[0] as u16) * 257).to_be_bytes();
    }
    finish_png(encoder, &data)
}

/// Flip the green channel of a normal map in place (OpenGL ⇄ DirectX).
fn flip_normal_green(rgba: &mut [u8]) {
    for p in rgba.as_chunks_mut::<4>().0 {
        p[1] = 255u8.saturating_sub(p[1]);
    }
}

fn smoothstep(x: f32) -> f32 {
    x * x * (3.0 - 2.0 * x)
}

/// Blend narrow opposite border bands so the tile wraps exactly (port of the reference
/// pipeline's `seamless_u8`): each band cross-fades toward the average of itself and the
/// mirrored opposite band, then the outermost row/column pairs are made identical.
fn make_seamless(rgba: &mut [u8], size: usize, margin: usize) {
    if margin == 0 || size < 8 {
        return;
    }
    let m = margin.min(size / 4).max(1);
    let denom = (m.saturating_sub(1)).max(1) as f32;
    let idx = |x: usize, y: usize| (y * size + x) * 4;
    // Cross-fade each band toward the average of itself and the mirrored opposite band —
    // the horizontal (left/right) pass first, then the vertical pass over the blended data.
    for horizontal in [true, false] {
        for lane in 0..size {
            for i in 0..m {
                let t = smoothstep(i as f32 / denom);
                let (near, far) = if horizontal {
                    (idx(i, lane), idx(size - 1 - i, lane))
                } else {
                    (idx(lane, i), idx(lane, size - 1 - i))
                };
                for c in 0..4 {
                    let a = rgba[near + c] as f32;
                    let b = rgba[far + c] as f32;
                    let common = 0.5 * (a + b);
                    rgba[near + c] = (common * (1.0 - t) + a * t).round().clamp(0.0, 255.0) as u8;
                    rgba[far + c] = (common * (1.0 - t) + b * t).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    // Force exact equality of the outermost column pairs, then the row pairs.
    for horizontal in [true, false] {
        for lane in 0..size {
            let (near, far) = if horizontal {
                (idx(0, lane), idx(size - 1, lane))
            } else {
                (idx(lane, 0), idx(lane, size - 1))
            };
            for c in 0..4 {
                let e = (rgba[near + c] as u16 + rgba[far + c] as u16).div_ceil(2) as u8;
                rgba[near + c] = e;
                rgba[far + c] = e;
            }
        }
    }
}

/// Re-normalize a tangent-space normal map in place (port of the reference pipeline): decode
/// to vectors, clamp +Z positive, renormalize to unit length, re-encode. Repairs the slight
/// denormalization that resizing and seam blending introduce.
fn renormalize_normal(rgba: &mut [u8]) {
    for p in rgba.as_chunks_mut::<4>().0 {
        let mut v = [p[0] as f32 / 127.5 - 1.0, p[1] as f32 / 127.5 - 1.0, p[2] as f32 / 127.5 - 1.0];
        v[2] = v[2].max(0.02);
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
        for (c, out) in v.iter().enumerate() {
            p[c] = ((out / len + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Build the engine's packed channel map, if the engine packs and every required source map is
/// present. Returns `(filename_suffix, rgba)`.
fn pack_for_engine(engine: EngineConvention, resized: &[Option<Vec<u8>>; 8], out_size: usize) -> Option<(&'static str, Vec<u8>)> {
    /// Assemble `n` output pixels from a per-pixel channel recipe.
    fn pack(n: usize, pixel: impl Fn(usize) -> [u8; 4]) -> Vec<u8> {
        let mut out = vec![0u8; n * 4];
        for (i, p) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            *p = pixel(i);
        }
        out
    }
    let n = out_size * out_size;
    let ch = |id: MapId| Some(resized[id.index()].as_ref()?.as_chunks::<4>().0);
    match engine {
        EngineConvention::Unreal => {
            let (a, r, m) = (ch(MapId::Ao)?, ch(MapId::Roughness)?, ch(MapId::Metallic)?);
            Some(("orm", pack(n, |i| [a[i][0], r[i][0], m[i][0], 255])))
        }
        EngineConvention::UnityUrp | EngineConvention::UnityHdrp => {
            // MaskMap: R=Metallic, G=Occlusion, B=0, A=Smoothness (inverted roughness).
            let (m, a, r) = (ch(MapId::Metallic)?, ch(MapId::Ao)?, ch(MapId::Roughness)?);
            Some(("maskmap", pack(n, |i| [m[i][0], a[i][0], 0, 255u8.saturating_sub(r[i][0])])))
        }
        EngineConvention::Gltf => {
            // glTF metallicRoughness: G=Roughness, B=Metallic (R/A unused).
            let (r, m) = (ch(MapId::Roughness)?, ch(MapId::Metallic)?);
            Some(("metalRough", pack(n, |i| [255, r[i][0], m[i][0], 255])))
        }
        EngineConvention::Blender | EngineConvention::Generic => None,
    }
}

/// Write a README describing the exported set (engine, colorspace, packing, normal convention).
fn write_readme(path: &Path, job: &ExportJob, maps: &[ManifestMap]) -> Result<()> {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "# {}\n", job.material);
    let _ = writeln!(s, "PBR material set exported by Lumagen — every map pixel-aligned to the albedo.\n");
    let _ = writeln!(s, "- **Engine preset:** {}", job.engine.display_name());
    let _ = writeln!(s, "- **Normal convention:** {}", normal_convention(job.directx_normal));
    let _ = writeln!(s, "- **Channel packing:** {}", job.engine.packing());
    let _ = writeln!(s, "- **Resolution:** {}", job.res);
    let _ = writeln!(s, "- **Seed:** {}\n", job.seed);
    let _ = writeln!(s, "## Maps\n");
    for m in maps {
        let _ = writeln!(s, "- **{}** ({}, {}-bit) — `{}`", m.map, m.colorspace, m.bit_depth, m.file);
    }
    std::fs::write(path, s).map_err(Error::Io)?;
    Ok(())
}

/// Run the export: write per-map files (+ packed variants per engine) + manifest. Returns the
/// list of files written. Creates `out_dir` if needed.
#[tracing::instrument(skip(job), fields(material = %job.material, out = %out_dir.display()))]
pub fn export_material(job: &ExportJob, out_dir: &Path) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(out_dir).map_err(Error::Io)?;
    let mut written = Vec::new();
    let mut manifest_maps: Vec<ManifestMap> = Vec::new();

    // Resize all present maps once, each from its own source size (native master or working),
    // then seam-blend the borders so the exported tile wraps exactly.
    let mut resized: [Option<Vec<u8>>; 8] = Default::default();
    for (i, m) in job.maps.iter().enumerate() {
        if let Some(px) = m {
            let src = if job.map_sizes[i] == 0 { job.src_size } else { job.map_sizes[i] };
            let mut out = resize_rgba(px, src, job.out_size)?;
            make_seamless(&mut out, job.out_size, job.seam_margin);
            resized[i] = Some(out);
        }
    }

    // The high-precision displacement source, resized + seam-blended in float so its geometry
    // matches the 8-bit maps' — a true 16-bit/EXR displacement quantized exactly once.
    let disp_field: Option<crate::height::HeightField> = job.displacement_height.as_ref().map(|hf| {
        let mut hf = hf.resized(job.out_size);
        hf.make_seamless(job.seam_margin);
        hf
    });

    for def in &MAPS {
        let Some(px) = &resized[def.id.index()] else { continue };
        // Only the normal map is transformed for output (and `resized` must stay pristine
        // for the contact sheet); every other map exports the shared buffer without a copy.
        let px: Cow<'_, [u8]> = if def.id == MapId::Normal {
            let mut fixed = px.clone();
            if job.directx_normal {
                flip_normal_green(&mut fixed);
            }
            // Resize + seam blending denormalize slightly — repair to unit vectors.
            renormalize_normal(&mut fixed);
            Cow::Owned(fixed)
        } else {
            Cow::Borrowed(px.as_slice())
        };
        let srgb = def.colorspace.eq_ignore_ascii_case("sRGB");
        let slug = map_def(def.id).name.to_lowercase().replace(' ', "_");
        let stem = expand(&job.pattern, &job.material, &slug, &job.res);

        let (file, bit_depth) = if def.id == MapId::Displacement && job.displacement_exr {
            let path = out_dir.join(format!("{stem}.exr"));
            // Real f32 EXR from the height master when we have one; the 8-bit ×257 fallback is
            // only reached for an imported/materialized displacement with no float source.
            match &disp_field {
                Some(hf) => hf.write_exr(&path)?,
                None => write_exr(&path, &px, job.out_size)?,
            }
            written.push(path);
            (format!("{stem}.exr"), 32u8)
        } else if def.id == MapId::Displacement && job.displacement_16bit_png {
            let path = out_dir.join(format!("{stem}.png"));
            match &disp_field {
                Some(hf) => hf.write_png16(&path)?,
                None => write_png16(&path, &px, job.out_size)?,
            }
            written.push(path);
            (format!("{stem}.png"), 16)
        } else {
            let path = out_dir.join(format!("{stem}.png"));
            write_png(&path, &px, job.out_size, srgb)?;
            written.push(path);
            (format!("{stem}.png"), 8)
        };
        manifest_maps.push(ManifestMap {
            map: def.name.to_string(),
            file,
            colorspace: def.colorspace.to_string(),
            bit_depth,
        });
    }

    let mut extra_files = Vec::new();
    if let Some((suffix, packed)) = pack_for_engine(job.engine, &resized, job.out_size) {
        let stem = expand(&job.pattern, &job.material, suffix, &job.res);
        let path = out_dir.join(format!("{stem}.png"));
        write_png(&path, &packed, job.out_size, false)?;
        written.push(path);
        extra_files.push(format!("{stem}.png"));
    }

    let cs = out_dir.join("contact_sheet.png");
    write_contact_sheet(&cs, &resized, job.out_size)?;
    written.push(cs);
    extra_files.push("contact_sheet.png".to_string());

    let readme = out_dir.join("README.md");
    write_readme(&readme, job, &manifest_maps)?;
    written.push(readme);
    extra_files.push("README.md".to_string());

    let manifest = Manifest {
        material: &job.material,
        engine: job.engine.display_name().to_string(),
        normal_convention: normal_convention(job.directx_normal).to_string(),
        channel_packing: job.engine.packing().to_string(),
        resolution: job.res.clone(),
        seed: job.seed,
        prompt: &job.prompt,
        maps: manifest_maps,
        extra_files,
    };
    let manifest_path = out_dir.join("manifest.json");
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).map_err(Error::from)?).map_err(Error::Io)?;
    written.push(manifest_path);

    Ok(written)
}

/// A simple edge-continuity metric: mean absolute difference between opposite border pixels
/// (0 = perfectly seamless). Lower is better; used for the "Seams OK" status.
pub fn seam_score(rgba: &[u8], size: usize) -> f32 {
    if size == 0 {
        return 0.0;
    }
    let (px, _) = rgba.as_chunks::<4>();
    let mut acc = 0.0f32;
    let mut n = 0u32;
    for x in 0..size {
        let top = px[x];
        let bottom = px[(size - 1) * size + x];
        for c in 0..3 {
            acc += (top[c] as f32 - bottom[c] as f32).abs();
            n += 1;
        }
        let left = px[x * size];
        let right = px[x * size + (size - 1)];
        for c in 0..3 {
            acc += (left[c] as f32 - right[c] as f32).abs();
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { acc / n as f32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(size: usize, v: u8) -> Vec<u8> {
        let mut px = vec![0u8; size * size * 4];
        for p in px.as_chunks_mut::<4>().0 {
            *p = [v, v, v, 255];
        }
        px
    }

    #[test]
    fn export_writes_maps_and_manifest() {
        let size = crate::maps::TEX_SIZE;
        let mut maps: [Option<Vec<u8>>; 8] = Default::default();
        maps[MapId::Albedo.index()] = Some(solid(size, 200));
        maps[MapId::Roughness.index()] = Some(solid(size, 128));
        let job = ExportJob {
            maps,
            map_sizes: [0; 8],
            src_size: size,
            out_size: size,
            material: "test".into(),
            res: "4k".into(),
            pattern: "$material_$map_$res".into(),
            engine: EngineConvention::Blender,
            directx_normal: false,
            displacement_16bit_png: false,
            displacement_exr: false,
            displacement_height: None,
            seed: 4821,
            prompt: "painted hull".into(),
            seam_margin: 0,
        };
        // Unique per process — concurrent test runs must not share output paths.
        let dir = std::env::temp_dir().join(format!("lumagen_export_writes_maps_and_manifest_{}", std::process::id()));
        let written = export_material(&job, &dir).expect("export");
        assert!(written.iter().any(|p| p.ends_with("manifest.json")));
        assert!(written.iter().any(|p| p.ends_with("contact_sheet.png")));
        assert!(written.iter().any(|p| p.ends_with("README.md")));
        assert!(written.iter().any(|p| p.to_string_lossy().contains("albedo")));
        assert!(written.iter().any(|p| p.to_string_lossy().contains("roughness")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// End-to-end proof that a displacement backed by an `f32` height master exports as a TRUE
    /// 16-bit PNG — far more than the 256 levels the 8-bit ×257 fallback can hold. The
    /// precision must survive the whole export path (resize + seam + encode), not just the
    /// encoder in isolation.
    #[test]
    fn displacement_height_master_exports_true_16bit() {
        use std::collections::HashSet;

        let side = 128;
        // A fine f32 gradient — every sample a distinct level, so the encoder's precision is
        // what caps the output level count, not the input's.
        let n = (side * side) as f32;
        let data: Vec<f32> = (0..side * side).map(|i| i as f32 / (n - 1.0)).collect();
        let hf = crate::height::HeightField::from_vec(data, side).expect("square");

        let mut maps: [Option<Vec<u8>>; 8] = Default::default();
        maps[MapId::Albedo.index()] = Some(solid(side, 180));
        maps[MapId::Displacement.index()] = Some(hf.to_rgba8(false, 1.0, false));

        let job = ExportJob {
            maps,
            map_sizes: [0; 8],
            src_size: side,
            out_size: side,
            material: "grad".into(),
            res: "1k".into(),
            pattern: "$map".into(),
            engine: EngineConvention::Generic,
            directx_normal: false,
            displacement_16bit_png: true,
            displacement_exr: false,
            displacement_height: Some(hf),
            seed: 1,
            prompt: String::new(),
            seam_margin: 0,
        };
        let dir = std::env::temp_dir().join(format!("lumagen_disp_true_16bit_{}", std::process::id()));
        export_material(&job, &dir).expect("export");

        let png = std::fs::read(dir.join("displacement.png")).expect("displacement png");
        let decoder = png::Decoder::new(std::io::Cursor::new(png));
        let mut reader = decoder.read_info().expect("read info");
        let mut buf = vec![0u8; reader.output_buffer_size().expect("buffer size")];
        let info = reader.next_frame(&mut buf).expect("frame");
        assert_eq!(info.bit_depth, png::BitDepth::Sixteen, "displacement must export as real 16-bit");
        let valid = info.line_size * info.height as usize;
        let levels: HashSet<u16> = buf[..valid].chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
        assert!(levels.len() > 256, "true 16-bit displacement resolves {} levels (>256)", levels.len());

        let _ = std::fs::remove_dir_all(&dir);
    }

    mod sanitize_stem {
        use rstest::rstest;

        use super::super::sanitize_stem;

        #[rstest]
        #[case::separators_replaced("a/b\\c", "a_b_c")]
        #[case::windows_invalid_chars_replaced("a<b>c:d\"e|f?g*h", "a_b_c_d_e_f_g_h")]
        #[case::absolute_prefix_neutralized("C:\\evil\\albedo", "C__evil_albedo")]
        #[case::traversal_collapses_to_plain_name("..\\..\\up", ".._.._up")]
        #[case::control_chars_replaced("a\u{1}b\nc", "a_b_c")]
        #[case::reserved_device_name_prefixed("CON", "_CON")]
        #[case::reserved_device_with_extension_prefixed("con.backup", "_con.backup")]
        #[case::trailing_dots_and_spaces_trimmed("name.. ", "name")]
        #[case::empty_falls_back("", "untitled")]
        #[case::only_invalid_chars_fall_back("...", "untitled")]
        #[case::clean_name_unchanged("hull_albedo_4k", "hull_albedo_4k")]
        fn hostile_stems_become_safe_filenames(#[case] input: &str, #[case] expected: &str) {
            assert_eq!(sanitize_stem(input), expected);
        }
    }

    #[test]
    fn directx_flips_normal_green() {
        let mut px = solid(4, 128);
        let before = px[1];
        flip_normal_green(&mut px);
        assert_eq!(px[1], 255 - before);
    }

    #[test]
    fn seamless_edges_score_zero() {
        let px = solid(8, 128);
        assert!(seam_score(&px, 8).abs() < f32::EPSILON);
    }

    /// The seam blend must leave opposite borders byte-identical (the reference pipeline's
    /// guarantee), even on noisy input.
    #[test]
    fn seam_blend_makes_opposite_borders_identical() {
        let size = 32;
        let mut px = vec![0u8; size * size * 4];
        for (i, p) in px.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let v = ((i * 37) % 251) as u8;
            *p = [v, v.wrapping_add(91), v.wrapping_mul(3), 255];
        }
        make_seamless(&mut px, size, 8);
        let idx = |x: usize, y: usize| (y * size + x) * 4;
        for x in 0..size {
            assert_eq!(
                px[idx(x, 0)..idx(x, 0) + 4],
                px[idx(x, size - 1)..idx(x, size - 1) + 4],
                "row mismatch at x={x}"
            );
        }
        for y in 0..size {
            assert_eq!(
                px[idx(0, y)..idx(0, y) + 4],
                px[idx(size - 1, y)..idx(size - 1, y) + 4],
                "col mismatch at y={y}"
            );
        }
        assert!(seam_score(&px, size).abs() < f32::EPSILON, "blended tile must score perfectly seamless");
    }

    /// Renormalization must output unit-length vectors with +Z (a legal tangent-space normal).
    #[test]
    fn renormalize_produces_unit_normals() {
        let mut px = vec![10u8, 240, 255, 255, 128, 128, 60, 255];
        renormalize_normal(&mut px);
        for p in px.as_chunks::<4>().0 {
            let v = [p[0] as f32 / 127.5 - 1.0, p[1] as f32 / 127.5 - 1.0, p[2] as f32 / 127.5 - 1.0];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((len - 1.0).abs() < 0.02, "not unit length: {len}");
            assert!(v[2] > 0.0, "Z must stay positive");
        }
    }
}
