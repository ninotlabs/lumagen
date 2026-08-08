//! The document model + `.lumagen` persistence. A `Document` is what the user made: material
//! identity, the per-map state and versions, prompt overrides, and export choices — separate
//! from `UiPrefs` (how the app is set up, which stays in eframe Storage). The `.lumagen` file
//! is a single-file container: a versioned MessagePack manifest (rmp-serde, `#[serde(default)]`
//! everywhere for forward-compat) holding the map pixels as PNG sections, zstd-compressed.
//! Keys are never serialized into it (they live in the OS vault).

use std::collections::HashMap;
use std::io::{Read as _, Write as _};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::data::MapId;
use crate::error::{Error, Result};

/// Current container schema version. Bump when the manifest shape changes; older files are
/// read through `#[serde(default)]` so new fields don't break them.
pub const SCHEMA_VERSION: u32 = 1;

const MAGIC: &[u8; 8] = b"LUMAGEN\x01";

/// Decompression bound for the manifest: proportional to the compressed payload, with a
/// generous floor. The manifest's bulk is PNG sections that are already deflate-compressed
/// (zstd gains ≈ nothing on them), so any manifest `save` can produce stays far below
/// compressed × 32 — only a crafted zstd bomb exceeds the bound. The floor keeps small
/// documents (whose MessagePack keys and metadata do compress hard) trivially inside it.
const MANIFEST_DECOMPRESS_FLOOR: u64 = 64 * 1024 * 1024;
const MANIFEST_DECOMPRESS_RATIO: u64 = 32;

/// One map's stored state: its versions (as PNG sections) and the non-destructive adjust params.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MapDoc {
    /// Non-destructive strength.
    pub strength: f32,
    /// Invert flag.
    pub invert: bool,
    /// Normal green-channel flip (OpenGL ⇄ DirectX).
    pub flip_y: bool,
    /// AO radius.
    pub radius: f32,
    /// 16-bit displacement flag.
    pub bit_depth_16: bool,
    /// Fidelity to albedo.
    pub fidelity: f32,
    /// Macro ↔ micro detail routing.
    pub detail_routing: f32,
    /// The active version index.
    pub active_version: usize,
    /// Whether this map has image data.
    pub has_image: bool,
    /// Whether the user locked/approved this map (locked maps skip regeneration).
    pub locked: bool,
    /// AI ⇄ derived path (true = AI-generated, false = deterministically derived).
    #[serde(default)]
    pub ai_path: bool,
    /// The seed of each version in the history (index+1 = the version number). Persisted so the
    /// history chips survive a save/reload; the active version's pixels are in `png`.
    #[serde(default)]
    pub version_seeds: Vec<u32>,
    /// Side length of the stored pixels. `0` (legacy documents) means the document's
    /// `tex_size` working size; otherwise this is the map's full-resolution master side.
    #[serde(default)]
    pub px_size: usize,
    /// PNG-encoded pixels for the active version (RGBA8, `px_size`² — the highest-resolution
    /// source available at save time).
    #[serde(with = "serde_bytes")]
    pub png: Vec<u8>,
    /// PNG-encoded working-size preview (RGBA8, `tex_size`²), stored alongside a
    /// full-resolution master so opening a document paints instantly from ~100 KB
    /// previews while the master decodes in the background. Empty when `png` is already
    /// working-size, or in documents saved before this field existed (those fall back to
    /// the full master decode and gain previews on their next save).
    #[serde(default, with = "serde_bytes")]
    pub preview_png: Vec<u8>,
}

impl Default for MapDoc {
    fn default() -> Self {
        Self {
            strength: 1.0,
            invert: false,
            flip_y: false,
            radius: 0.35,
            bit_depth_16: false,
            fidelity: 0.96,
            detail_routing: 10.0,
            active_version: 0,
            has_image: false,
            locked: false,
            ai_path: false,
            version_seeds: Vec::new(),
            px_size: 0,
            png: Vec::new(),
            preview_png: Vec::new(),
        }
    }
}

/// The document: everything the user made for one material.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Document {
    /// Container schema version.
    pub schema_version: u32,
    /// Material name (slug).
    pub material_name: String,
    /// Project name.
    pub project_name: String,
    /// Material type/class.
    pub mat_type: String,
    /// The semantic description driving generation.
    pub description: String,
    /// Shared project seed.
    pub seed: u32,
    /// Requested generation size (0 = model default / auto, else a square side in px).
    pub gen_size: usize,
    /// Engine preset name for export.
    pub engine: String,
    /// Export filename pattern.
    pub export_pattern: String,
    /// Working texture size (square).
    pub tex_size: usize,
    /// Per-map state.
    pub maps: HashMap<String, MapDoc>,
    /// Prompt overrides (base, final, negative, per-map templates).
    pub base_template: String,
    /// Final-requirement override.
    pub final_requirement: String,
    /// Negative-prompt override.
    pub negative_prompt: String,
    /// Per-map template overrides.
    pub map_templates: HashMap<String, String>,
    /// The depth-reconstructed displacement height master, stored as a true 16-bit PNG so the
    /// high-precision displacement survives save/load — the 8-bit RGBA master in `maps` can't
    /// carry it. Empty when the displacement wasn't depth-reconstructed (derived-on-demand at
    /// export instead). Its square side is `displacement_height_side`.
    #[serde(default, with = "serde_bytes")]
    pub displacement_height_png16: Vec<u8>,
    /// Side length of `displacement_height_png16` (0 when absent).
    #[serde(default)]
    pub displacement_height_side: usize,
}

impl Default for Document {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            material_name: String::new(),
            project_name: String::new(),
            mat_type: String::new(),
            description: String::new(),
            seed: 4821,
            // 0 = Max: always ask the model for the highest resolution it supports.
            gen_size: 0,
            engine: "Blender".into(),
            export_pattern: "$material_$map_$res".into(),
            tex_size: crate::maps::TEX_SIZE,
            maps: HashMap::new(),
            base_template: String::new(),
            final_requirement: String::new(),
            negative_prompt: String::new(),
            map_templates: HashMap::new(),
            displacement_height_png16: Vec::new(),
            displacement_height_side: 0,
        }
    }
}

impl Document {
    /// The key for a map in the `maps` map (its `MapId::index()` as a stable string).
    fn key(id: MapId) -> String {
        id.index().to_string()
    }

    /// Store a map's pixels + adjust params. `side` is the pixel buffer's square side —
    /// the working size, or a full-resolution master. On encode failure (e.g. a buffer
    /// that doesn't match `side`² RGBA) nothing is recorded, so a save can never claim
    /// `has_image` over an empty PNG section.
    pub fn set_map_sized(&mut self, id: MapId, rgba: &[u8], side: usize, params: MapDocParams) -> Result<()> {
        let png = encode_png(rgba, side)?;
        // A working-size preview rides along with every master so opens paint instantly
        // and the master can decode in the background. Preview failure only degrades the
        // open path back to the full master decode — it never fails the save.
        let preview_png = if side > self.tex_size {
            let work = crate::maps::resize_square_rgba(rgba, side, self.tex_size);
            encode_png(&work, self.tex_size).unwrap_or_else(|e| {
                tracing::warn!(?id, "working-size preview encode failed: {e}");
                Vec::new()
            })
        } else {
            Vec::new()
        };
        self.maps.insert(Self::key(id), params.into_map_doc(png, preview_png, side));
        Ok(())
    }

    /// Store a map's pixels at the document working size (`tex_size`).
    pub fn set_map(&mut self, id: MapId, rgba: &[u8], params: MapDocParams) -> Result<()> {
        self.set_map_sized(id, rgba, self.tex_size, params)
    }

    /// Store a map from an ALREADY-ENCODED PNG section — the save passthrough for a master
    /// that was opened from disk and never touched: no decode, no re-encode, byte-identical
    /// write-back. `preview_png` must be the working-size preview for that section (encode
    /// one from the working pixels when the source document predates previews).
    pub fn set_map_encoded(&mut self, id: MapId, png: Vec<u8>, preview_png: Vec<u8>, side: usize, params: MapDocParams) {
        self.maps.insert(Self::key(id), params.into_map_doc(png, preview_png, side));
    }

    /// Read a map's stored state, decoding its pixels back to RGBA8 (at `px_size`², falling
    /// back to `tex_size` for legacy documents).
    pub fn get_map(&self, id: MapId) -> Option<(Vec<u8>, MapDoc)> {
        self.maps
            .get(&Self::key(id))
            .and_then(|doc| decode_png(&doc.png).ok().map(|rgba| (rgba, doc.clone())))
    }

    /// Read a map's working-size preview (RGBA8 at `tex_size`²), when the document carries
    /// one — the fast path for opening and library cards. `None` for working-size maps and
    /// pre-preview documents; callers fall back to the full `get_map` decode.
    pub fn get_map_preview(&self, id: MapId) -> Option<Vec<u8>> {
        let doc = self.maps.get(&Self::key(id))?;
        if doc.preview_png.is_empty() {
            return None;
        }
        decode_png(&doc.preview_png).ok().filter(|rgba| rgba.len() == self.tex_size * self.tex_size * 4)
    }

    /// The stored pixel side for a map (`tex_size` when the legacy field is absent).
    pub fn map_px_size(&self, id: MapId) -> usize {
        self.maps
            .get(&Self::key(id))
            .map_or(self.tex_size, |d| if d.px_size == 0 { self.tex_size } else { d.px_size })
    }

    /// Serialize to the `.lumagen` container (zstd-compressed MessagePack with a magic
    /// header). Written to a sibling temp file that is renamed over the target once the
    /// bytes are synced, so a crash or disk-full mid-write can't destroy the previous save.
    #[tracing::instrument(skip(self), fields(path = %path.display()))]
    pub fn save(&self, path: &Path) -> Result<()> {
        let manifest = rmp_serde::to_vec_named(self).map_err(|e| Error::Serialize(e.to_string()))?;
        let compressed = zstd::bulk::compress(&manifest, 3).map_err(Error::Io)?;
        let tmp = sibling_tmp_path(path);
        write_container(&tmp, path, &compressed).map_err(|e| {
            // A failed write must not leave the temp behind; the previous save is untouched.
            let _ = std::fs::remove_file(&tmp);
            Error::Io(e)
        })
    }

    /// Load a `.lumagen` container. The decompression bound scales with the compressed
    /// payload (see `MANIFEST_DECOMPRESS_RATIO`), so any document `save` produced loads
    /// while a zstd bomb still errors instead of exhausting memory.
    #[tracing::instrument(fields(path = %path.display()))]
    pub fn load(path: &Path) -> Result<Self> {
        let mut file = std::fs::File::open(path).map_err(Error::Io)?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic).map_err(Error::Io)?;
        if &magic != MAGIC {
            return Err(Error::InvalidState("not a .lumagen file".into()));
        }
        let mut compressed = Vec::new();
        file.read_to_end(&mut compressed).map_err(Error::Io)?;
        let cap = MANIFEST_DECOMPRESS_FLOOR.max((compressed.len() as u64).saturating_mul(MANIFEST_DECOMPRESS_RATIO));
        // Streaming decode grows the buffer to the actual manifest size instead of
        // pre-allocating the cap; `take(cap + 1)` turns a bomb into a typed error.
        let decoder = zstd::stream::read::Decoder::new(compressed.as_slice()).map_err(Error::Io)?;
        let mut manifest = Vec::new();
        decoder.take(cap.saturating_add(1)).read_to_end(&mut manifest).map_err(Error::Io)?;
        if manifest.len() as u64 > cap {
            return Err(Error::InvalidState(format!("document manifest exceeds the {cap}-byte decompression bound")));
        }
        let doc: Document = rmp_serde::from_slice(&manifest).map_err(|e| Error::Serialize(e.to_string()))?;
        // A newer-format file must not silently decode to defaults (`#[serde(default)]`
        // fills unknown shapes) and then be overwritten stripped by the next save.
        if doc.schema_version > SCHEMA_VERSION {
            return Err(Error::InvalidState(format!(
                "document schema v{} is newer than this app's v{SCHEMA_VERSION}; update Lumagen to open it",
                doc.schema_version
            )));
        }
        Ok(doc)
    }

    /// Store the depth-reconstructed displacement height master as a 16-bit PNG section, so a
    /// TRUE 16-bit/EXR displacement survives save/load.
    pub fn set_displacement_height(&mut self, height: &crate::height::HeightField) -> Result<()> {
        self.displacement_height_png16 = height.to_png16_bytes()?;
        self.displacement_height_side = height.side();
        Ok(())
    }

    /// Read back the stored displacement height master, if any (decoded 16-bit-precise, values
    /// as stored). `None` when the document carries no reconstructed height.
    pub fn displacement_height(&self) -> Option<crate::height::HeightField> {
        if self.displacement_height_png16.is_empty() {
            return None;
        }
        crate::height::HeightField::from_png_bytes(&self.displacement_height_png16)
    }

    /// Serialize to pretty JSON (for diff/VCS export).
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Error::from)
    }
}

/// Adjust params captured when storing a map (avoids a partial `MapDoc` at the call site).
#[derive(Clone, Debug, Default)]
pub struct MapDocParams {
    /// Strength.
    pub strength: f32,
    /// Invert.
    pub invert: bool,
    /// Flip Y.
    pub flip_y: bool,
    /// Radius.
    pub radius: f32,
    /// 16-bit.
    pub bit_depth_16: bool,
    /// Fidelity.
    pub fidelity: f32,
    /// Detail routing.
    pub detail_routing: f32,
    /// Active version.
    pub active_version: usize,
    /// Locked/approved.
    pub locked: bool,
    /// AI ⇄ derived path (true = AI).
    pub ai_path: bool,
    /// The seed of each version in the history.
    pub version_seeds: Vec<u32>,
}

impl MapDocParams {
    /// Build the stored entry around already-encoded PNG sections. `has_image` follows the
    /// section: a successful encode is never empty (the header alone is bytes), and an empty
    /// passthrough must not claim an image.
    fn into_map_doc(self, png: Vec<u8>, preview_png: Vec<u8>, px_size: usize) -> MapDoc {
        MapDoc {
            strength: self.strength,
            invert: self.invert,
            flip_y: self.flip_y,
            radius: self.radius,
            bit_depth_16: self.bit_depth_16,
            fidelity: self.fidelity,
            detail_routing: self.detail_routing,
            active_version: self.active_version,
            has_image: !png.is_empty(),
            locked: self.locked,
            ai_path: self.ai_path,
            version_seeds: self.version_seeds,
            px_size,
            png,
            preview_png,
        }
    }
}

// ── Container write ────────────────────────────────────────────────────────

/// A temp path in the same directory as `path`: rename is only atomic within one volume,
/// so the temp must be a sibling of the destination. The pid suffix keeps concurrent
/// processes from clobbering each other's in-progress write.
fn sibling_tmp_path(path: &Path) -> std::path::PathBuf {
    let mut name = path
        .file_name()
        .map_or_else(|| std::ffi::OsString::from("document"), std::ffi::OsStr::to_os_string);
    name.push(format!(".{}.tmp", std::process::id()));
    path.with_file_name(name)
}

/// Write magic + payload to `tmp`, sync, and rename it over `path`. The previous file at
/// `path` is only replaced once the new bytes are durable (`std::fs::rename` replaces an
/// existing destination on Windows).
fn write_container(tmp: &Path, path: &Path, compressed: &[u8]) -> std::io::Result<()> {
    {
        let mut file = std::fs::File::create(tmp)?;
        file.write_all(MAGIC)?;
        file.write_all(compressed)?;
        // Sync before rename so the rename can never promote a not-yet-durable file to
        // being the only copy.
        file.sync_all()?;
    }
    // The handle is closed before the rename: Windows can refuse to move an open file.
    std::fs::rename(tmp, path)
}

// ── PNG section codec ──────────────────────────────────────────────────────

fn encode_png(rgba: &[u8], size: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, size as u32, size as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|e| Error::Encode(e.to_string()))?;
        writer.write_image_data(rgba).map_err(|e| Error::Encode(e.to_string()))?;
    }
    Ok(out)
}

/// Largest master side the app can produce (the Refine flow's 16 K upscale ceiling);
/// bounds what a stored PNG section may declare.
const MAX_MAP_SIDE: u32 = 16_384;
/// The output-buffer ceiling that follows from it (RGBA8 at `MAX_MAP_SIDE`²).
const MAX_PNG_OUT_BYTES: usize = (MAX_MAP_SIDE as usize) * (MAX_MAP_SIDE as usize) * 4;

/// Decode a stored map PNG section to RGBA8, with the same hostile-input limits `load()`
/// applies — for callers (the app's open worker) that defer master decodes and hold the
/// raw section bytes themselves.
pub fn decode_map_png(png_bytes: &[u8]) -> Result<Vec<u8>> {
    decode_png(png_bytes)
}

fn decode_png(png_bytes: &[u8]) -> Result<Vec<u8>> {
    // A `.lumagen` can be corrupt or hostile, so the decoder's internal allocations are
    // capped at the app's real maximum instead of trusting the header.
    let decoder = png::Decoder::new_with_limits(std::io::Cursor::new(png_bytes), png::Limits { bytes: MAX_PNG_OUT_BYTES });
    let mut reader = decoder.read_info().map_err(|e| Error::Decode(e.to_string()))?;
    let out_size = reader.output_buffer_size().ok_or_else(|| Error::Decode("unknown png size".into()))?;
    // `Limits` doesn't cover this caller-side buffer, so header-declared dimensions must
    // be validated before allocating — a hostile header can otherwise force a huge
    // allocation and abort the process.
    if out_size > MAX_PNG_OUT_BYTES {
        return Err(Error::Decode(format!(
            "png declares {out_size} output bytes, above the {MAX_PNG_OUT_BYTES}-byte ({MAX_MAP_SIDE}² RGBA) map ceiling"
        )));
    }
    let mut buf = vec![0u8; out_size];
    let info = reader.next_frame(&mut buf).map_err(|e| Error::Decode(e.to_string()))?;
    buf.truncate(info.buffer_size());
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique per-process/per-test temp dir, removed on drop (including panic unwind) so
    /// concurrent suite runs can't race on a shared fixture path.
    struct TestDir(std::path::PathBuf);

    impl TestDir {
        fn new(test: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("lumagen_doc_{}_{test}", std::process::id()));
            std::fs::create_dir_all(&dir).expect("create test dir");
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Deterministic xorshift64* stream: cheap to generate and incompressible enough that
    /// neither PNG deflate nor zstd can shrink it meaningfully.
    fn pseudo_random_bytes(len: usize) -> Vec<u8> {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut out = Vec::with_capacity(len + 8);
        while out.len() < len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.extend_from_slice(&state.to_le_bytes());
        }
        out.truncate(len);
        out
    }

    /// A minimal PNG (signature + IHDR + empty IDAT) declaring arbitrary dimensions; only
    /// the header must be readable for `decode_png`'s pre-allocation checks to fire.
    fn png_with_header_dims(width: u32, height: u32) -> Vec<u8> {
        fn crc32(bytes: &[u8]) -> u32 {
            let mut crc = 0xFFFF_FFFF_u32;
            for &b in bytes {
                crc ^= u32::from(b);
                for _ in 0..8 {
                    crc = if crc & 1 == 1 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
                }
            }
            !crc
        }
        fn chunk(out: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) {
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            out.extend_from_slice(&kind);
            out.extend_from_slice(data);
            let mut crc_input = kind.to_vec();
            crc_input.extend_from_slice(data);
            out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
        }
        let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        // 8-bit RGBA, deflate, adaptive filtering, no interlace.
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
        chunk(&mut out, *b"IHDR", &ihdr);
        chunk(&mut out, *b"IDAT", &[]);
        out
    }

    #[test]
    fn document_round_trip_preserves_maps_and_fields() {
        let size = crate::maps::TEX_SIZE;
        let mut doc = Document {
            material_name: "painted_hull".into(),
            description: "test material".into(),
            seed: 999,
            ..Default::default()
        };
        let rgba = vec![128u8; size * size * 4];
        doc.set_map(
            MapId::Roughness,
            &rgba,
            MapDocParams {
                strength: 1.5,
                flip_y: true,
                ai_path: true,
                version_seeds: vec![111, 222, 333],
                ..Default::default()
            },
        )
        .expect("set map");

        let dir = TestDir::new("round_trip");
        let path = dir.path().join("doc.lumagen");
        doc.save(&path).expect("save");
        let loaded = Document::load(&path).expect("load");

        assert_eq!(loaded.material_name, "painted_hull");
        assert_eq!(loaded.description, "test material");
        assert_eq!(loaded.seed, 999);
        let (pixels, mapdoc) = loaded.get_map(MapId::Roughness).expect("map present");
        assert_eq!(pixels.len(), size * size * 4);
        assert!(pixels.iter().all(|&b| b == 128));
        assert!((mapdoc.strength - 1.5).abs() < f32::EPSILON);
        assert!(mapdoc.flip_y);
        // The AI-path flag and version history metadata survive the round-trip.
        assert!(mapdoc.ai_path, "ai_path not persisted");
        assert_eq!(mapdoc.version_seeds, vec![111, 222, 333], "version seeds not persisted");
    }

    /// The depth-reconstructed displacement height survives save/load at 16-bit precision, so a
    /// reopened material still exports a TRUE 16-bit/EXR displacement.
    #[test]
    fn displacement_height_round_trips_at_16bit() {
        let side = 64;
        let n = (side * side) as f32;
        let data: Vec<f32> = (0..side * side).map(|i| i as f32 / (n - 1.0)).collect();
        let hf = crate::height::HeightField::from_vec(data, side).expect("square");

        let mut doc = Document::default();
        doc.set_displacement_height(&hf).expect("set height");
        let dir = TestDir::new("disp_height");
        let path = dir.path().join("h.lumagen");
        doc.save(&path).expect("save");

        let loaded = Document::load(&path).expect("load");
        let back = loaded.displacement_height().expect("height restored");
        assert_eq!(back.side(), side);
        for (a, b) in hf.samples().iter().zip(back.samples()) {
            assert!((a - b).abs() < 2.0 / 65535.0, "16-bit precision preserved: {a} vs {b}");
        }
        // A document with no reconstructed height reports none.
        assert!(Document::default().displacement_height().is_none());
    }

    #[test]
    fn load_rejects_non_lumagen_file() {
        let dir = TestDir::new("not_a_doc");
        let path = dir.path().join("not_a_doc.bin");
        std::fs::write(&path, b"not a lumagen file").expect("write");
        assert!(Document::load(&path).is_err());
    }

    #[test]
    fn save_replaces_existing_file_without_leaving_temp() {
        let dir = TestDir::new("atomic_save");
        let path = dir.path().join("mat.lumagen");
        let first = Document {
            material_name: "first".into(),
            ..Default::default()
        };
        first.save(&path).expect("first save");
        let second = Document {
            material_name: "second".into(),
            ..Default::default()
        };
        second.save(&path).expect("second save over existing file");
        let loaded = Document::load(&path).expect("load");
        assert_eq!(loaded.material_name, "second");
        // No `.tmp` sibling may survive a successful save — a leftover would mean the
        // write path isn't write-temp-then-rename.
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .expect("read dir")
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temp files left behind: {leftovers:?}");
    }

    #[test]
    fn save_into_missing_directory_errors() {
        let dir = TestDir::new("save_missing_dir");
        let path = dir.path().join("nope").join("mat.lumagen");
        let doc = Document::default();
        let err = doc.save(&path).expect_err("save into a missing directory must fail");
        assert!(matches!(err, Error::Io(_)), "expected Io error, got {err:?}");
    }

    #[test]
    fn manifest_larger_than_64_mib_round_trips() {
        let dir = TestDir::new("large_manifest");
        // 96 MiB of incompressible payload puts the decompressed manifest past the 64 MiB
        // floor, so loading must rely on the compressed-size-scaled bound.
        let payload = pseudo_random_bytes(96 * 1024 * 1024);
        let mut doc = Document {
            material_name: "big".into(),
            ..Default::default()
        };
        doc.maps.insert(
            Document::key(MapId::Albedo),
            MapDoc {
                has_image: true,
                px_size: 4096,
                png: payload.clone(),
                ..Default::default()
            },
        );
        let path = dir.path().join("big.lumagen");
        doc.save(&path).expect("save");
        let loaded = Document::load(&path).expect("load of a >64 MiB manifest");
        let map = loaded.maps.get(&Document::key(MapId::Albedo)).expect("map present");
        assert_eq!(map.png.len(), payload.len());
        // Identity check without a 96 MiB assert_eq! diff on failure.
        assert!(map.png == payload, "payload bytes diverged after round-trip");
    }

    #[test]
    fn masters_store_a_working_size_preview_and_working_maps_do_not() {
        let mut doc = Document::default();
        let side = crate::maps::TEX_SIZE * 2;
        let master = vec![128u8; side * side * 4];
        doc.set_map_sized(MapId::Albedo, &master, side, MapDocParams::default()).expect("set master");
        let preview = doc.get_map_preview(MapId::Albedo).expect("master carries a preview");
        assert_eq!(preview.len(), crate::maps::TEX_SIZE * crate::maps::TEX_SIZE * 4, "preview is working-size RGBA");

        let working = vec![64u8; crate::maps::TEX_SIZE * crate::maps::TEX_SIZE * 4];
        doc.set_map(MapId::Roughness, &working, MapDocParams::default()).expect("set working");
        assert!(doc.get_map_preview(MapId::Roughness).is_none(), "working-size maps need no separate preview");
    }

    #[test]
    fn pre_preview_documents_fall_back_to_the_master_decode() {
        // A document written before `preview_png` existed: the field is empty, so the
        // fast path must decline and the full `get_map` decode must still work.
        let mut doc = Document::default();
        let side = crate::maps::TEX_SIZE * 2;
        doc.set_map_sized(MapId::Albedo, &vec![200u8; side * side * 4], side, MapDocParams::default())
            .expect("set");
        if let Some(map) = doc.maps.get_mut(&Document::key(MapId::Albedo)) {
            map.preview_png = Vec::new();
        }
        assert!(doc.get_map_preview(MapId::Albedo).is_none());
        let (rgba, _) = doc.get_map(MapId::Albedo).expect("full decode still available");
        assert_eq!(rgba.len(), side * side * 4);
    }

    #[test]
    fn load_rejects_newer_schema_version() {
        let dir = TestDir::new("newer_schema");
        let doc = Document {
            schema_version: SCHEMA_VERSION + 1,
            material_name: "future".into(),
            ..Default::default()
        };
        let path = dir.path().join("future.lumagen");
        doc.save(&path).expect("save");
        let err = Document::load(&path).expect_err("newer schema must be rejected, not decoded to defaults");
        assert!(err.to_string().contains("newer"), "unexpected error: {err}");
    }

    #[test]
    fn decode_png_rejects_hostile_header_dimensions() {
        // 100000² RGBA would be a ~40 GB caller-side allocation if the header were trusted.
        let hostile = png_with_header_dims(100_000, 100_000);
        let err = decode_png(&hostile).expect_err("hostile dimensions must be rejected");
        assert!(matches!(err, Error::Decode(_)), "expected Decode error, got {err:?}");
        assert!(err.to_string().contains("ceiling"), "unexpected error: {err}");
    }

    #[test]
    fn set_map_with_mismatched_buffer_errors_and_records_nothing() {
        let mut doc = Document::default();
        let err = doc
            .set_map(MapId::Albedo, &[0u8; 16], MapDocParams::default())
            .expect_err("a buffer that doesn't match tex_size² RGBA must fail");
        assert!(matches!(err, Error::Encode(_)), "expected Encode error, got {err:?}");
        // The failed encode must not record an entry claiming has_image over empty bytes.
        assert!(doc.maps.is_empty(), "failed encode recorded a map entry");
    }

    /// Secrets sweep: a serialized Document must never contain an API key. The struct has no
    /// key field by construction, but this guards against a future field accidentally
    /// serializing a secret (keys live only in the OS vault per the keyring skill).
    #[test]
    fn serialized_document_contains_no_secret_markers() {
        let doc = Document {
            material_name: "hull".into(),
            description: "painted hull".into(),
            ..Default::default()
        };
        let json = doc.to_json().expect("serialize");
        for marker in ["api_key", "fal_key", "openrouter", "secret", "token", "Bearer", "FAL_KEY"] {
            assert!(!json.contains(marker), "document leaked marker {marker}");
        }
    }
}
