//! Central application state.

use std::collections::HashMap;

use eframe::egui::{self, TextureHandle, TextureOptions};
use serde::{Deserialize, Serialize};

use crate::data::{MapId, PRESETS};
use crate::maps::{self, Layout};
use crate::render::{self, IoResult, IoTask, Output, RenderRequest, RenderTarget, RenderWorker};
use crate::secrets::Provider;
use crate::widgets::{ToastKind, Toasts};

/// Storage key under which UI preferences are persisted via eframe Storage (RON).
pub const PREFS_KEY: &str = "ui_prefs";

/// User preferences that survive a restart. Small by design — eframe Storage is for "how
/// the app was set up," not the working document. Every field has a `#[serde(default)]`
/// so adding a preference never breaks an existing file.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct UiPrefs {
    pub theme: usize,
    pub nav_preset: usize,
    pub accent: String,
    pub autosave: bool,
    pub library_location: String,
    pub watch_folders: String,
    /// Chosen provider model slugs (empty falls back to the built-in defaults on load).
    /// `fal_model` is the historical name for the text-to-image slot.
    pub fal_model: String,
    /// fal image-edit endpoint (empty derives one from the t2i slot on load).
    pub fal_model_edit: String,
    pub or_model: String,
    pub llm_model: String,
    /// Cost-per-image estimate driver (string so the settings field round-trips verbatim).
    pub cost_per_image: String,
    /// Whether the first-run intro has been seen (it only shows once).
    pub intro_seen: bool,
    /// Per-job deadline in seconds (empty keeps the built-in default).
    pub job_timeout: String,
    /// Topaz upscaler model (empty keeps the built-in default).
    pub upscale_model: String,
    /// Upscale target side in px (0 keeps the built-in default).
    pub upscale_target: usize,
    /// Default map-path policy index (None keeps the built-in default).
    pub map_paths: Option<usize>,
    /// Known project names — projects are lightweight labels, so an empty project (no
    /// materials yet, or none saved) must survive a restart via prefs, not only via docs.
    pub projects: Vec<String>,
    /// The active project new materials are filed under.
    pub current_project: String,
    /// Generation quality tier index (None keeps the built-in default).
    pub quality: Option<usize>,
    /// Export size index (None keeps the built-in default).
    pub final_size: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Library,
    Workspace,
    Settings,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MapStatus {
    Empty,
    Queued,
    Generating,
    Ready,
    Locked,
    Overridden,
    Error,
}

impl MapStatus {
    pub fn has_image(self) -> bool {
        matches!(self, MapStatus::Ready | MapStatus::Locked | MapStatus::Overridden)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    Map2d,
    Material3d,
    Tiled,
    Sheet,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PreviewMesh {
    Sphere,
    Plane,
    Cube,
}

impl PreviewMesh {
    pub fn name(self) -> &'static str {
        match self {
            PreviewMesh::Sphere => "Sphere",
            PreviewMesh::Plane => "Plane",
            PreviewMesh::Cube => "Cube",
        }
    }
}

#[derive(Clone)]
pub struct Version {
    pub n: u32,
    pub seed: u32,
    /// The exact RGBA8 pixels for this version, kept for recent versions so restoring is exact.
    /// `None` means the version is restorable only by re-render/re-derive from the seed (older
    /// versions whose pixels were evicted, or versions loaded from disk).
    pub pixels: Option<std::sync::Arc<Vec<u8>>>,
    /// This version's full-resolution master `(rgba, side)`, kept for the most recent versions
    /// so upscaling is non-destructive: restoring a version brings its native resolution back.
    /// The pixels are `Arc`-shared with `master_pixels`/save/derive jobs — masters are only
    /// ever replaced wholesale, never mutated in place, so sharing is sound and restoring or
    /// saving never copies the (up to 16K²) buffer.
    pub master: Option<(std::sync::Arc<Vec<u8>>, usize)>,
}

#[derive(Clone)]
pub struct MapState {
    pub status: MapStatus,
    pub versions: Vec<Version>,
    pub active_version: usize,
    /// AI ⇄ derived path for this map. Defaults per the design split (roughness/metallic/
    /// emission = AI; normal/AO/transparency/displacement = derived); the per-map toggle in the
    /// workspace can promote/demote any map, and the choice drives `GenerationPlan.paths`.
    pub derive_path: lumagen_core::derive::DerivePath,
    // adjust (non-destructive)
    pub strength: f32,
    pub invert: bool,
    pub flip_y: bool,
    pub radius: f32,
    pub bit_depth_16: bool,
    pub fidelity: f32,
    pub detail_routing: f32, // macro ↔ micro split (10 = displacement owns 10%)
}

impl MapState {
    fn new(id: MapId) -> Self {
        Self {
            status: MapStatus::Empty,
            versions: Vec::new(),
            active_version: 0,
            derive_path: lumagen_core::derive::default_path(id),
            strength: 1.0,
            invert: false,
            flip_y: false,
            radius: 0.35,
            bit_depth_16: crate::data::map_def(id).bits == 16,
            fidelity: 0.96,
            detail_routing: 10.0,
        }
    }

    /// The non-destructive derivation params (the adjust sliders) as a core `DeriveParams`.
    pub fn derive_params(&self) -> lumagen_core::derive::DeriveParams {
        lumagen_core::derive::DeriveParams {
            strength: self.strength,
            invert: self.invert,
            flip_y: self.flip_y,
            radius: self.radius,
            detail_routing: self.detail_routing,
        }
    }
}

// ── Library data ─────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct LibMaterial {
    pub slug: String,
    pub class: String,
    /// The project this material belongs to (empty = unassigned).
    pub project: String,
    pub res: String,
    pub coverage: u8,
    pub stale_index: Option<usize>,
    pub seed: u32,
    /// The backing `.lumagen` file, when one exists on disk. Drives the grid preview: a real
    /// document's preview is its saved albedo — never the procedural stand-in.
    pub doc_path: Option<std::path::PathBuf>,
}

#[derive(Clone)]
pub struct LibProject {
    pub name: String,
}

// ── Settings state ───────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SettingsCategory {
    Providers,
    Generation,
    Prompts,
    Presets,
    Export,
    Library,
    Interface,
}

impl SettingsCategory {
    pub const ALL: [SettingsCategory; 7] = [
        SettingsCategory::Providers,
        SettingsCategory::Generation,
        SettingsCategory::Prompts,
        SettingsCategory::Presets,
        SettingsCategory::Export,
        SettingsCategory::Library,
        SettingsCategory::Interface,
    ];
    pub fn name(self) -> &'static str {
        match self {
            SettingsCategory::Providers => "Providers & API keys",
            SettingsCategory::Generation => "Generation defaults",
            SettingsCategory::Prompts => "Prompts & templates",
            SettingsCategory::Presets => "Material presets",
            SettingsCategory::Export => "Export defaults",
            SettingsCategory::Library => "Library",
            SettingsCategory::Interface => "Interface",
        }
    }
}

/// Which backend would serve a generation right now.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActiveProvider {
    Fal,
    OpenRouter,
    Mock,
}

/// The resolved active backend: provider + the model each request kind would use.
#[derive(Clone, Debug)]
pub struct ActiveBackend {
    pub provider: ActiveProvider,
    /// Text-to-image model (the source albedo).
    pub t2i: String,
    /// Image-edit model (every AI-path map, albedo-referenced).
    pub edit: String,
    /// Whether this is a paid live backend (false = offline mock).
    pub live: bool,
}

/// Outcome of a provider "Test connection" run, shown inline in the settings card so the
/// result can't be missed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TestStatus {
    /// The request is in flight.
    Pending,
    /// The provider answered and the key was accepted.
    Ok(String),
    /// The test failed (bad key, network, …).
    Failed(String),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PromptTab {
    Base,
    PerMap,
    Negative,
    Suffixes,
    Tokens,
}

pub struct SettingsState {
    pub category: SettingsCategory,
    pub return_to: Screen,
    // providers — the API keys themselves live only in the OS vault (keyring skill); these
    // are a non-persisted edit buffer plus a "is a key set" flag for the UI.
    pub fal_key_input: String,
    pub fal_key_set: bool,
    pub show_fal_key: bool,
    pub openrouter_key_input: String,
    pub openrouter_key_set: bool,
    pub show_openrouter_key: bool,
    pub openai_key_input: String,
    pub openai_key_set: bool,
    pub show_openai_key: bool,
    pub llm_key_input: String,
    pub llm_key_set: bool,
    pub show_llm_key: bool,
    pub cost_per_image: String,
    pub max_concurrency: String,
    pub retries: String,
    pub endpoint: String,
    /// Inline per-provider "Test connection" status (not persisted).
    pub test_status: HashMap<Provider, TestStatus>,
    /// fal text-to-image endpoint for the source albedo (fal splits t2i and edit endpoints).
    pub fal_model_t2i: String,
    /// fal image-edit endpoint shared by every AI-path map (albedo as reference).
    pub fal_model_edit: String,
    pub or_model: String,
    pub llm_model: String,
    // generation defaults
    pub api_size: usize,
    pub final_size: usize,
    pub quality: usize,
    pub seam_margin: String,
    pub background_opaque: bool,
    /// Overall per-job deadline in seconds (submit → result). Big image models can take
    /// several minutes; string so the settings field round-trips verbatim.
    pub job_timeout: String,
    // prompts
    pub prompt_tab: PromptTab,
    pub base_template: String,
    pub final_requirement: String,
    pub negative_prompt: String,
    pub map_templates: HashMap<MapId, String>,
    pub editor_map: MapId,
    pub preview_preset: String,
    pub preview_map: MapId,
    // presets
    pub preset_texts: Vec<String>,
    // export defaults
    pub filename_pattern: String,
    pub normal_convention: usize,
    /// Topaz upscaler model for Step ③ Refine (schema-fixed enum on fal's side).
    pub upscale_model: String,
    /// Upscale target side in px (4096 / 8192 / 16384). Targets beyond a single 4× pass
    /// chain multiple passes.
    pub upscale_target: usize,
    /// Default map-path policy for new materials: 0 = AI for semantic maps (roughness ·
    /// metallic · emission), 1 = AI for every map (like the reference script), 2 = derived
    /// for every map (free). Each map's Path section overrides per material.
    pub map_path_policy: usize,
    // library
    pub library_location: String,
    pub watch_folders: String,
    pub autosave: bool,
    // interface
    pub theme: usize,
    pub nav_preset: usize,
    pub accent: String,
}

impl SettingsState {
    fn new() -> Self {
        let mut map_templates = HashMap::new();
        for id in MapId::ALL {
            map_templates.insert(id, crate::data::default_map_template(id).to_string());
        }
        Self {
            category: SettingsCategory::Providers,
            return_to: Screen::Workspace,
            fal_key_input: String::new(),
            fal_key_set: false,
            show_fal_key: false,
            openrouter_key_input: String::new(),
            openrouter_key_set: false,
            show_openrouter_key: false,
            openai_key_input: String::new(),
            openai_key_set: false,
            show_openai_key: false,
            llm_key_input: String::new(),
            llm_key_set: false,
            show_llm_key: false,
            cost_per_image: "0.04".into(),
            max_concurrency: "3".into(),
            retries: "3".into(),
            endpoint: "https://api.openai.com/v1".into(),
            test_status: HashMap::new(),
            fal_model_t2i: "openai/gpt-image-2".into(),
            fal_model_edit: "openai/gpt-image-2/edit".into(),
            or_model: "google/gemini-2.5-flash-image".into(),
            llm_model: "gpt-5.4-mini".into(),
            api_size: 0,
            final_size: 0,
            // High quality by default — the reference pipeline's setting, and the difference
            // is very visible. Drop to low in Settings → Generation for cheap test runs.
            quality: 2,
            seam_margin: "64".into(),
            background_opaque: true,
            job_timeout: "600".into(),
            prompt_tab: PromptTab::Base,
            base_template: crate::data::BASE_TEMPLATE.into(),
            final_requirement: crate::data::FINAL_REQUIREMENT.into(),
            negative_prompt: crate::data::NEGATIVE_PROMPT.into(),
            map_templates,
            editor_map: MapId::Roughness,
            preview_preset: "painted_hull".into(),
            preview_map: MapId::Roughness,
            preset_texts: PRESETS.iter().map(|p| p.desc.to_string()).collect(),
            filename_pattern: "$material_$map_$res".into(),
            normal_convention: 0,
            upscale_model: "High Fidelity V2".into(),
            upscale_target: 4096,
            map_path_policy: 0,
            library_location: "~/Lumagen/Library".into(),
            watch_folders: "~/Textures, ~/Downloads/pbr".into(),
            autosave: true,
            theme: 0,
            nav_preset: 0,
            accent: "Blue".into(),
        }
    }

    /// The per-job deadline as a duration, clamped to a sane range (60 s – 1 h). Falls back
    /// to 10 minutes when the field doesn't parse.
    pub fn job_timeout_duration(&self) -> std::time::Duration {
        let secs = self.job_timeout.trim().parse::<u64>().unwrap_or(600).clamp(60, 3600);
        std::time::Duration::from_secs(secs)
    }

    pub fn compiled_preview(&self) -> String {
        let desc = self
            .preset_texts
            .get(PRESETS.iter().position(|p| p.slug == self.preview_preset).unwrap_or(0))
            .cloned()
            .unwrap_or_default();
        let map_tpl = self.map_templates.get(&self.preview_map).cloned().unwrap_or_default();
        crate::data::compile_prompt(&self.base_template, &self.final_requirement, &desc, &map_tpl)
    }

    /// The `(is-set flag, edit buffer)` slot for a provider's API key.
    fn key_slot(&mut self, provider: Provider) -> (&mut bool, &mut String) {
        match provider {
            Provider::Fal => (&mut self.fal_key_set, &mut self.fal_key_input),
            Provider::OpenRouter => (&mut self.openrouter_key_set, &mut self.openrouter_key_input),
            Provider::OpenAi => (&mut self.openai_key_set, &mut self.openai_key_input),
            Provider::Llm => (&mut self.llm_key_set, &mut self.llm_key_input),
        }
    }
}

// ── New-material wizard ──────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CreateStage {
    Choice,
    Generate,
    Import,
}

pub struct CreateState {
    pub stage: CreateStage,
    /// The material's name (slug). Empty = auto-derive from type + seed.
    pub name: String,
    pub mat_type: String,
    pub prompt: String,
    pub negative: String,
    pub seed: String,
    /// Export size choice (0 = 1024², 1 = 2048², 2 = 4096²) — applied to the export setting.
    pub resolution: usize,
}

impl CreateState {
    fn new() -> Self {
        Self {
            stage: CreateStage::Choice,
            name: String::new(),
            mat_type: "Painted".into(),
            prompt: "painted orbital-station hull, modular armor plates, recessed seams, structural ribs, sparse cool-blue emissive strips, subtle grime and edge wear".into(),
            negative: "seams, text, watermark, perspective, baked lighting".into(),
            seed: "4821".into(),
            resolution: 2,
        }
    }
}

// ── App state ────────────────────────────────────────────────────────────

pub struct AppState {
    pub screen: Screen,
    pub selected: MapId,
    pub view: ViewMode,
    pub active_step: usize,
    pub material_name: String,
    pub project_name: String,
    pub mat_type: String,
    pub description: String,
    pub seed: u32,
    pub spend: f64,
    pub unsaved: bool,
    pub engine: String,

    /// Per-map state, indexed by `MapId::index()` — a total lookup, so `map`/`map_mut`
    /// never fail.
    pub maps: [MapState; 8],
    pub layout: Layout,
    pub textures: HashMap<MapId, TextureHandle>,
    pub thumbs: HashMap<MapId, TextureHandle>,
    pub lib_previews: HashMap<String, TextureHandle>,
    /// CPU-side RGBA8 pixels per map (working size), kept so the wgpu PBR preview can sample
    /// the real maps without reading back GPU textures. Populated on generate/import.
    pub map_pixels: HashMap<MapId, std::sync::Arc<Vec<u8>>>,
    /// Full-resolution square masters `(rgba, side)` per map — the model's native output,
    /// kept for export quality (the working copy above is a 400² preview downscale).
    /// Pixels are `Arc`-shared (never mutated in place) so save/derive/version snapshots
    /// clone the handle, not the buffer.
    pub master_pixels: HashMap<MapId, (std::sync::Arc<Vec<u8>>, usize)>,
    /// High-precision displacement height masters (`0..1` `f32`) from the depth-model macro
    /// path, keyed by map (only Displacement uses it today). Parallel to `master_pixels`: the
    /// 8-bit master drives the UI/contact sheet, this drives a TRUE 16-bit/EXR export.
    pub master_height: HashMap<MapId, lumagen_core::height::HeightField>,
    /// GPU textures of the masters, uploaded on demand so the 2D viewport shows native
    /// resolution instead of the soft 400² working copy.
    pub master_textures: HashMap<MapId, TextureHandle>,
    /// Generation size request: 0 = Max (the highest square the model supports, found by
    /// stepping down a tier ladder on free submit rejections), else an explicit square side.
    /// Per material; persisted in the document.
    pub gen_size: usize,

    // generation
    pub queue: Vec<MapId>,
    pub generating: bool,
    /// Parent cancellation token for the in-flight generation batch; cancelled on new runs.
    pub gen_cancel: Option<tokio_util::sync::CancellationToken>,
    /// Per-map child tokens for the in-flight batch, so a single map can be cancelled from the
    /// job queue without killing the rest.
    pub gen_map_cancel: HashMap<MapId, tokio_util::sync::CancellationToken>,
    /// Maps that completed in the current batch (for progress display).
    pub gen_done: usize,
    /// Total maps expected in the current batch.
    pub gen_total: usize,
    /// Set by `start_generate_all`/`reroll` to tell the app to launch the batch on the runtime.
    pub pending_generate: bool,
    /// Set by the variations modal to tell the app to launch a 4-candidate variation run for
    /// this map on the runtime. Cleared once launched.
    pub pending_variations: Option<MapId>,
    /// Set by a Settings "Test connection" button; the app runs a real cheap request for this
    /// provider on the tokio runtime and toasts the outcome. Cleared once launched.
    pub pending_test: Option<Provider>,
    /// Set by the "Generate from prompt" wizard; the app runs a text-to-image albedo generation
    /// on the tokio runtime. Cleared once launched.
    pub pending_albedo_gen: bool,
    /// Maps queued for a Topaz upscale (Step ③ Refine); the app launches them on the runtime.
    pub pending_upscale: Vec<MapId>,
    /// Derived-path maps waiting for the albedo's upscale to land before re-deriving from
    /// the new master (Step ③ Refine, free path).
    pub pending_rederive: Vec<MapId>,
    /// Maps with an upscale in flight (drives the spinners; results land as Ready outcomes).
    pub upscaling: std::collections::HashSet<MapId>,
    /// Cancellation token for the in-flight albedo generation (long jobs need a Cancel).
    pub albedo_cancel: Option<tokio_util::sync::CancellationToken>,
    /// Set by the Description Check "Enhance with AI" button; the app runs an LLM review of the
    /// description on the tokio runtime and replaces the description with the result.
    pub pending_enhance: bool,
    /// Cancellation token for the in-flight variations run — closing the modal cancels the
    /// paid provider call instead of letting it bill and stream into a closed UI.
    pub variations_cancel: Option<tokio_util::sync::CancellationToken>,
    /// The description text captured when the LLM enhance launched. The result auto-applies
    /// only if the text is still identical — edits typed while the call was in flight win.
    pub enhance_baseline: Option<String>,
    /// Monotonic material-session counter. Worker/render results are stamped with it at
    /// submit; a result carrying an older epoch is dropped at the drain instead of being
    /// installed into whichever material happens to be open now.
    pub material_epoch: u64,
    /// Document saves queued on the worker but not yet reported back (`DocSaved`); app exit
    /// drains these so closing the window can't abandon a queued save.
    pub pending_saves: usize,
    /// A library open deferred behind the unsaved-changes confirm; the modal resolves it.
    pub pending_open: Option<String>,
    /// A new-material create deferred behind the unsaved-changes confirm.
    pub pending_new: Option<(String, u32)>,
    /// One-shot consent from the unsaved-changes modal: the next guarded switch proceeds.
    pub discard_confirmed: bool,
    /// The slug whose document is opening on the worker (drives the library-card spinner
    /// and blocks further opens until `DocOpened` lands).
    pub opening: Option<String>,
    /// Maps whose full-resolution masters are decoding on the worker right now (requested
    /// on demand by the 2D native view / Export / Refine). Export/Refine gate on this.
    pub masters_loading: std::collections::HashSet<MapId>,
    /// Still-ENCODED master sections from the opened document `(png, stored preview,
    /// native side)`. Decoded lazily when something needs pixels; written back
    /// byte-identical on save so untouched masters cost no decode and no re-encode.
    /// Invalidated the moment a map's master changes (generation/upscale/override/…).
    pub master_encoded: HashMap<MapId, render::EncodedMaster>,
    // Seam-score cache: recomputed only when the albedo's seed changes (rendering the albedo
    // every frame to re-measure it is too expensive).
    seam_cache: f32,
    seam_cache_seed: Option<u32>,
    /// The last albedo-generation failure, shown persistently in the viewport/right panel
    /// (a transient toast alone made failures look like a silent fake success).
    pub albedo_error: Option<String>,
    // The folder the last export wrote to; revealed on a successful export.
    pub last_export_dir: Option<std::path::PathBuf>,
    // Bounded undo/redo history of document snapshots (Ctrl+Z / Ctrl+Shift+Z).
    undo_stack: Vec<lumagen_core::document::Document>,
    redo_stack: Vec<lumagen_core::document::Document>,
    // Geometric maps whose adjust sliders changed and need an off-thread re-derive; flushed
    // after a short debounce so dragging a slider doesn't re-derive on every frame.
    adjust_dirty: std::collections::HashSet<MapId>,
    adjust_dirty_at: f64,

    // background render worker (created on first `logic` pass, which has the Context)
    render_worker: Option<RenderWorker>,
    // slugs with a library-preview render already in flight, so we don't resubmit per frame
    lib_pending: std::collections::HashSet<String>,
    // variation preview keys (var_*) with a render in flight
    pub var_pending: std::collections::HashSet<String>,
    // variation candidate RGBA pixels (var_*) kept so promoting one installs the real pixels
    // rather than re-rendering from a seed
    pub var_pixels: std::collections::HashMap<String, Vec<u8>>,
    // variation candidate full-resolution masters (var_* → (rgba, side))
    pub var_masters: std::collections::HashMap<String, (Vec<u8>, usize)>,
    // whether the vault keys have been requested this session (settings opens the load once)
    pub keys_loaded: bool,
    // whether the on-disk library has been scanned this session (library opens the scan once)
    pub library_scanned: bool,
    // a library scan is in flight on the worker (the grid shows its placeholder meanwhile)
    pub library_scanning: bool,

    // viewport
    pub mesh: PreviewMesh,
    pub hdri: String,
    pub tile_mode: usize, // 0=1×1 1=2×2 2=3×3
    pub zoom: f32,
    pub pan: egui::Vec2,
    pub light_angle: f32,
    /// 3D camera orbit yaw (drag X), radians.
    pub orbit_yaw: f32,
    /// 3D camera orbit pitch (drag Y), radians, clamped shy of the poles.
    pub orbit_pitch: f32,
    /// 3D camera distance (scroll to dolly).
    pub orbit_radius: f32,
    /// UV repeat factor in the 3D view (inspect tiling density without leaving 3D).
    pub tiling3d: f32,
    /// Normal-map strength in the 3D view (preview-only; export bakes via per-map Strength).
    pub normal_strength3d: f32,
    /// Emission intensity in the 3D view (preview-only; engines scale emissive at import).
    pub emission_boost3d: f32,
    /// The egui texture id of the wgpu PBR preview's offscreen target, registered each frame the
    /// 3D view is shown. `None` until the first 3D frame renders.
    pub preview_tex: Option<egui::TextureId>,

    // modals
    pub show_intro: bool,
    pub show_create: bool,
    pub show_new_project: bool,
    pub new_project_name: String,
    pub create: CreateState,
    pub show_confirm: bool,
    pub show_variations: Option<MapId>,
    pub show_export: bool,
    /// The Edit-material modal (rename + export/generation size after creation).
    pub show_edit_material: bool,
    /// Name buffer for the Edit-material modal (applied — slugified — on Save).
    pub edit_material_name: String,
    /// Project buffer for the Edit-material modal (applied on Save; new names become
    /// projects — this is also how existing project-less materials get filed).
    pub edit_material_project: String,
    pub export_pattern: String,
    pub show_assist: bool,
    pub assist_map: MapId,
    /// The Refine modal (Step ③): pick target/model and upscale a selection of maps.
    pub show_refine: bool,
    /// Maps ticked in the Refine modal (rebuilt from `upscale_targets` when it opens).
    pub refine_selection: std::collections::HashSet<MapId>,
    pub queue_expanded: bool,
    /// Before/after split compare is open (albedo vs the selected map), with a draggable split.
    pub show_compare: bool,
    /// Horizontal split position (0..1) of the compare divider.
    pub compare_split: f32,
    /// Compare mode: 0 = slide (split divider), 1 = overlay (opacity blend).
    pub compare_mode: usize,
    /// Overlay opacity (0 = albedo only, 1 = map only).
    pub compare_alpha: f32,

    // library
    pub library: Vec<LibMaterial>,
    pub projects: Vec<LibProject>,
    pub lib_nav: String,
    pub lib_class: String,
    pub lib_search: String,
    pub lib_sort: usize,

    pub settings: SettingsState,
    pub toasts: Toasts,
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppState {
    pub fn new() -> Self {
        let mut maps: [MapState; 8] = std::array::from_fn(|i| MapState::new(MapId::ALL[i]));
        // The albedo starts ready with its first version.
        let albedo = &mut maps[MapId::Albedo.index()];
        albedo.status = MapStatus::Ready;
        albedo.versions.push(Version {
            n: 1,
            seed: 4821,
            pixels: None,
            master: None,
        });

        Self {
            screen: Screen::Library,
            selected: MapId::Albedo,
            view: ViewMode::Map2d,
            active_step: 1,
            material_name: "painted_hull".into(),
            project_name: String::new(),
            mat_type: "Painted".into(),
            description: "painted orbital-station hull with modular armor plates, recessed seams, and sparse cool-blue emissive strips.".into(),
            seed: 4821,
            spend: 0.0,
            unsaved: true,
            engine: "Blender".into(),
            maps,
            layout: maps::build_layout(4821),
            textures: HashMap::new(),
            thumbs: HashMap::new(),
            lib_previews: HashMap::new(),
            map_pixels: HashMap::new(),
            master_pixels: HashMap::new(),
            master_height: HashMap::new(),
            master_textures: HashMap::new(),
            gen_size: 0,
            queue: Vec::new(),
            generating: false,
            gen_cancel: None,
            gen_map_cancel: HashMap::new(),
            gen_done: 0,
            gen_total: 0,
            pending_generate: false,
            pending_variations: None,
            pending_test: None,
            pending_albedo_gen: false,
            pending_upscale: Vec::new(),
            pending_rederive: Vec::new(),
            upscaling: std::collections::HashSet::new(),
            albedo_cancel: None,
            pending_enhance: false,
            variations_cancel: None,
            enhance_baseline: None,
            material_epoch: 0,
            pending_saves: 0,
            pending_open: None,
            pending_new: None,
            discard_confirmed: false,
            opening: None,
            masters_loading: std::collections::HashSet::new(),
            master_encoded: HashMap::new(),
            render_worker: None,
            lib_pending: std::collections::HashSet::new(),
            var_pixels: std::collections::HashMap::new(),
            var_masters: std::collections::HashMap::new(),
            var_pending: std::collections::HashSet::new(),
            keys_loaded: false,
            library_scanned: false,
            library_scanning: false,
            seam_cache: 0.0,
            seam_cache_seed: None,
            albedo_error: None,
            last_export_dir: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            adjust_dirty: std::collections::HashSet::new(),
            adjust_dirty_at: 0.0,
            mesh: PreviewMesh::Sphere,
            hdri: "Studio Soft".into(),
            tile_mode: 1,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            light_angle: 0.0,
            orbit_yaw: 0.5,
            orbit_pitch: 0.3,
            orbit_radius: 3.2,
            tiling3d: 1.0,
            normal_strength3d: 1.0,
            emission_boost3d: 1.0,
            preview_tex: None,
            show_intro: true,
            show_create: false,
            show_new_project: false,
            new_project_name: String::new(),
            create: CreateState::new(),
            show_confirm: false,
            show_variations: None,
            show_export: false,
            show_edit_material: false,
            edit_material_name: String::new(),
            edit_material_project: String::new(),
            export_pattern: "$material_$map_$res".into(),
            show_assist: false,
            assist_map: MapId::Roughness,
            show_refine: false,
            refine_selection: std::collections::HashSet::new(),
            queue_expanded: false,
            show_compare: false,
            compare_split: 0.5,
            compare_mode: 0,
            compare_alpha: 0.5,
            // No demo seed data: the library starts empty (a friendly empty state invites the
            // first material) and fills from the on-disk scan + the user's own creations.
            library: Vec::new(),
            projects: Vec::new(),
            lib_nav: "All materials".into(),
            lib_class: "All".into(),
            lib_search: String::new(),
            lib_sort: 0,
            settings: SettingsState::new(),
            toasts: Toasts::new(),
        }
    }

    /// The best texture to DISPLAY for a map in the 2D/tiled viewport: the native-resolution
    /// master (uploaded lazily) when one exists, else the working-size texture. Returns the
    /// texture id + source side length.
    pub fn display_texture(&mut self, ctx: &egui::Context, id: MapId) -> Option<(egui::TextureId, usize)> {
        // A still-encoded master decodes on demand the first time the viewport wants
        // native resolution; the working texture shows meanwhile and the badge reads
        // "decoding native resolution…" until the GPU upload below takes over.
        if !self.master_pixels.contains_key(&id) && self.master_encoded.contains_key(&id) {
            self.request_master_decode(ctx, id);
        }
        if let Some(m) = self.master_pixels.get(&id) {
            // GPU backends cap texture sides (often 8192 or 16384; test backends less) — a
            // 16K master must be downscaled for DISPLAY or the upload panics. The badge
            // still reports the true native side.
            let max_side = ctx.input(|i| i.max_texture_side).max(512);
            let tex = self.master_textures.entry(id).or_insert_with(|| {
                let (px, side) = (m.0.as_slice(), m.1);
                let img = if side > max_side {
                    let shown = maps::resize_square_rgba(px, side, max_side);
                    egui::ColorImage::from_rgba_unmultiplied([max_side, max_side], &shown)
                } else {
                    egui::ColorImage::from_rgba_unmultiplied([side, side], px)
                };
                ctx.load_texture(format!("{id:?}_master"), img, TextureOptions::LINEAR)
            });
            return Some((tex.id(), m.1));
        }
        self.textures.get(&id).map(|t| (t.id(), maps::TEX_SIZE))
    }

    // Total lookup — `maps` holds one slot per `MapId` variant, indexed by `MapId::index()`.
    pub fn map(&self, id: MapId) -> &MapState {
        &self.maps[id.index()]
    }
    pub fn map_mut(&mut self, id: MapId) -> &mut MapState {
        &mut self.maps[id.index()]
    }

    /// Whether a map has real image content. The albedo is deliberately not special-cased —
    /// the procedural placeholder must never masquerade as a generated result while a real
    /// generation is in flight (or has failed).
    pub fn has_map(&self, id: MapId) -> bool {
        self.map(id).status.has_image()
    }

    /// How many recent versions keep their pixels in memory for exact restore (older ones fall
    /// back to re-render from the seed). Bounds memory: a 4K RGBA version is ~64 MB.
    const MAX_VERSION_PIXELS: usize = 6;
    /// How many recent versions keep their full-resolution MASTER (8K/16K masters are huge;
    /// three is enough to A/B upscaler modes non-destructively).
    const MAX_VERSION_MASTERS: usize = 3;

    /// Push a new version for `id`, make it active, and (optionally) keep its exact pixels
    /// (+ master) for restore — evicting the oldest kept payloads beyond the caps.
    fn push_version(&mut self, id: MapId, seed: u32, pixels: Option<std::sync::Arc<Vec<u8>>>, master: Option<(std::sync::Arc<Vec<u8>>, usize)>) {
        let m = self.map_mut(id);
        let n = m.versions.len() as u32 + 1;
        m.versions.push(Version { n, seed, pixels, master });
        m.active_version = m.versions.len() - 1;
        Self::evict_over_cap(&mut m.versions, Self::MAX_VERSION_PIXELS, |v| &mut v.pixels);
        Self::evict_over_cap(&mut m.versions, Self::MAX_VERSION_MASTERS, |v| &mut v.master);
    }

    /// Clear the oldest kept payloads so at most `cap` versions retain one.
    fn evict_over_cap<T>(versions: &mut [Version], cap: usize, payload: impl Fn(&mut Version) -> &mut Option<T>) {
        let kept: Vec<usize> = versions.iter_mut().enumerate().filter_map(|(i, v)| payload(v).is_some().then_some(i)).collect();
        for &i in kept.iter().take(kept.len().saturating_sub(cap)) {
            *payload(&mut versions[i]) = None;
        }
    }

    /// Install RGBA8 pixels for a map: upload its full texture + thumbnail and cache the pixels.
    fn install_pixels(&mut self, ctx: &egui::Context, id: MapId, px: std::sync::Arc<Vec<u8>>) {
        let full: &[u8] = &px;
        self.textures
            .insert(id, ctx.load_texture(format!("{id:?}_full"), render::full_image(full), TextureOptions::LINEAR));
        let thumb = crate::maps::make_thumb(full);
        self.thumbs.insert(
            id,
            ctx.load_texture(format!("{id:?}_thumb"), render::thumb_image(&thumb), TextureOptions::LINEAR),
        );
        self.map_pixels.insert(id, px);
    }

    /// Replace (or clear) a map's full-resolution master. Always invalidates the encoded
    /// save-passthrough section and the stale master GPU texture — a superseded master must
    /// never write back into the next save or keep displaying.
    fn set_master(&mut self, id: MapId, master: Option<(std::sync::Arc<Vec<u8>>, usize)>) {
        self.master_encoded.remove(&id);
        match master {
            Some(m) => {
                self.master_pixels.insert(id, m);
            }
            None => {
                self.master_pixels.remove(&id);
            }
        }
        self.master_textures.remove(&id);
    }

    /// Reset every map slot to empty and drop all pixel/texture stores — the baseline for
    /// installing a different material.
    fn reset_maps_and_stores(&mut self) {
        for id in MapId::ALL {
            let m = self.map_mut(id);
            m.status = MapStatus::Empty;
            m.versions.clear();
            m.active_version = 0;
        }
        self.textures.clear();
        self.thumbs.clear();
        self.map_pixels.clear();
        self.master_pixels.clear();
        self.master_height.clear();
        self.master_encoded.clear();
        self.master_textures.clear();
    }

    pub fn generated_all(&self) -> bool {
        MapId::DERIVED.iter().all(|id| self.map(*id).status.has_image())
    }

    /// How many maps currently hold image content (the library card's coverage pips).
    fn coverage(&self) -> u8 {
        MapId::ALL.iter().filter(|id| self.map(**id).status.has_image()).count() as u8
    }

    pub fn toast(&mut self, msg: impl Into<String>, kind: ToastKind) {
        self.toasts.push(msg, kind);
    }

    /// Log an error and surface it to the user as an error toast.
    fn error_toast(&mut self, log: &str, toast: impl Into<String>) {
        tracing::error!("{log}");
        self.toast(toast, ToastKind::Error);
    }

    /// Snapshot the persisted UI preferences out of the live settings.
    pub fn prefs(&self) -> UiPrefs {
        UiPrefs {
            theme: self.settings.theme,
            nav_preset: self.settings.nav_preset,
            accent: self.settings.accent.clone(),
            autosave: self.settings.autosave,
            library_location: self.settings.library_location.clone(),
            watch_folders: self.settings.watch_folders.clone(),
            fal_model: self.settings.fal_model_t2i.clone(),
            fal_model_edit: self.settings.fal_model_edit.clone(),
            or_model: self.settings.or_model.clone(),
            llm_model: self.settings.llm_model.clone(),
            cost_per_image: self.settings.cost_per_image.clone(),
            intro_seen: !self.show_intro,
            job_timeout: self.settings.job_timeout.clone(),
            upscale_model: self.settings.upscale_model.clone(),
            upscale_target: self.settings.upscale_target,
            map_paths: Some(self.settings.map_path_policy),
            projects: self.projects.iter().map(|p| p.name.clone()).collect(),
            current_project: self.project_name.clone(),
            quality: Some(self.settings.quality),
            final_size: Some(self.settings.final_size),
        }
    }

    /// Apply restored preferences onto the live settings. Empty stored strings keep the
    /// built-in defaults so an older prefs file (before these fields existed) is clean.
    pub fn apply_prefs(&mut self, prefs: UiPrefs) {
        // Persisted indices feed straight into 2-entry arrays (settings.rs) — an out-of-range
        // value from a tampered/newer prefs file must clamp, not panic the Settings screen.
        self.settings.theme = prefs.theme.min(1);
        self.settings.nav_preset = prefs.nav_preset.min(1);
        self.settings.accent = prefs.accent;
        self.settings.autosave = prefs.autosave;
        self.settings.library_location = prefs.library_location;
        self.settings.watch_folders = prefs.watch_folders;
        if !prefs.fal_model.is_empty() {
            self.settings.fal_model_t2i = prefs.fal_model;
        }
        if !prefs.fal_model_edit.is_empty() {
            self.settings.fal_model_edit = prefs.fal_model_edit;
        } else if !self.settings.fal_model_t2i.is_empty() {
            // Migration from the single-slot era: derive the edit endpoint from the t2i one.
            // Edit-capable ids (kontext / qwen / explicit …/edit) serve both; split families
            // (gpt-image, nano-banana) get their `/edit` sibling.
            let t2i = self.settings.fal_model_t2i.clone();
            self.settings.fal_model_edit = if t2i.ends_with("/edit") || t2i.contains("kontext") || t2i.contains("qwen-image-edit") {
                t2i
            } else {
                format!("{t2i}/edit")
            };
        }
        if !prefs.or_model.is_empty() {
            self.settings.or_model = prefs.or_model;
        }
        if !prefs.llm_model.is_empty() {
            self.settings.llm_model = prefs.llm_model;
        }
        if !prefs.cost_per_image.is_empty() {
            self.settings.cost_per_image = prefs.cost_per_image;
        }
        if !prefs.job_timeout.is_empty() {
            self.settings.job_timeout = prefs.job_timeout;
        }
        if !prefs.upscale_model.is_empty() && prefs.upscale_model != "Standard V2" {
            // "Standard V2" was the old built-in default, snapshotted into every prefs file —
            // treat it as unset so the High Fidelity V2 default applies (it can still be
            // re-picked in Settings, which persists any non-default choice).
            self.settings.upscale_model = prefs.upscale_model;
        }
        if matches!(prefs.upscale_target, 4096 | 8192 | 16384) {
            self.settings.upscale_target = prefs.upscale_target;
        }
        if let Some(p) = prefs.map_paths {
            self.settings.map_path_policy = p.min(2);
        }
        for name in prefs.projects {
            if !name.is_empty() && !self.projects.iter().any(|p| p.name == name) {
                self.projects.push(LibProject { name });
            }
        }
        if !prefs.current_project.is_empty() {
            self.project_name = prefs.current_project;
        }
        if let Some(q) = prefs.quality {
            self.settings.quality = q.min(3);
        }
        if let Some(f) = prefs.final_size {
            // 5 export sizes (4096/2048/1024/8192/16384) — the clamp must cover all of them
            // or a persisted 8K/16K choice is silently dropped.
            self.settings.final_size = f.min(4);
        }
        if prefs.intro_seen {
            self.show_intro = false;
        }
    }

    /// The path the current default policy gives `id`: 0 = AI for the semantic maps only,
    /// 1 = AI for every map (like the reference script), 2 = derived for every map.
    pub fn policy_path(&self, id: MapId) -> lumagen_core::derive::DerivePath {
        use lumagen_core::derive::DerivePath;
        if id == MapId::Albedo {
            return DerivePath::Ai;
        }
        match self.settings.map_path_policy {
            1 => DerivePath::Ai,
            2 => DerivePath::Derived,
            _ => lumagen_core::derive::default_path(id),
        }
    }

    /// Stamp the default policy onto every map of the open material (used at material
    /// creation and by the explicit "Apply to this material" action — each map's Path
    /// toggle can still override afterwards). Maps demoted to derived re-derive their
    /// previews from the albedo (free); promotions only affect the next generation.
    pub fn apply_path_policy(&mut self, now: f64) {
        use lumagen_core::derive::DerivePath;
        for id in MapId::ALL {
            if id == MapId::Albedo {
                continue;
            }
            let path = self.policy_path(id);
            let was = std::mem::replace(&mut self.map_mut(id).derive_path, path);
            if was == DerivePath::Ai && path == DerivePath::Derived {
                self.mark_adjust_dirty(id, now);
            }
        }
    }

    pub fn add_spend(&mut self, amount: f64) {
        self.spend += amount;
    }

    /// Max undo depth. Snapshots are cheap (working textures are `TEX_SIZE`, not 4K).
    const MAX_UNDO: usize = 12;

    /// Capture the current document onto the undo stack (call before a mutating action). Clears
    /// the redo stack, since a new action forks the history.
    pub fn snapshot(&mut self) {
        let doc = self.to_document();
        self.undo_stack.push(doc);
        if self.undo_stack.len() > Self::MAX_UNDO {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    /// Undo the last snapshotted action (Ctrl+Z). Textures re-upload from the restored pixel
    /// store on the next `ensure_textures` pass.
    pub fn undo(&mut self) {
        let Some(prev) = self.undo_stack.pop() else {
            self.toast("Nothing to undo", ToastKind::Info);
            return;
        };
        let current = self.to_document();
        self.redo_stack.push(current);
        self.apply_document(&prev);
        self.unsaved = true;
        self.toast("Undo", ToastKind::Info);
    }

    /// Redo the last undone action (Ctrl+Shift+Z).
    pub fn redo(&mut self) {
        let Some(next) = self.redo_stack.pop() else {
            self.toast("Nothing to redo", ToastKind::Info);
            return;
        };
        let current = self.to_document();
        self.undo_stack.push(current);
        self.apply_document(&next);
        self.unsaved = true;
        self.toast("Redo", ToastKind::Info);
    }

    /// The image backend a generation run would actually use right now, mirroring
    /// `provider_config`'s resolution order: fal (vault key or env), then OpenRouter, then the
    /// offline mock.
    pub fn active_backend(&self) -> ActiveBackend {
        let env_set = |v: &str| std::env::var(v).map(|k| !k.is_empty()).unwrap_or(false);
        if self.settings.fal_key_set || env_set("FAL_KEY") || env_set("FAL_API_KEY") {
            ActiveBackend {
                provider: ActiveProvider::Fal,
                t2i: self.settings.fal_model_t2i.clone(),
                edit: self.settings.fal_model_edit.clone(),
                live: true,
            }
        } else if self.settings.openrouter_key_set || env_set("OPENROUTER_API_KEY") {
            ActiveBackend {
                provider: ActiveProvider::OpenRouter,
                t2i: self.settings.or_model.clone(),
                edit: self.settings.or_model.clone(),
                live: true,
            }
        } else {
            ActiveBackend {
                provider: ActiveProvider::Mock,
                t2i: "offline mock".into(),
                edit: "offline mock".into(),
                live: false,
            }
        }
    }

    /// The configured cost per image (Settings → Providers), falling back to the built-in price.
    pub fn cost_per_image(&self) -> f64 {
        self.settings
            .cost_per_image
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|c| *c >= 0.0)
            .unwrap_or(crate::data::PRICE_PER_IMAGE)
    }

    /// (paid request count, estimated cost) to generate the currently-missing maps — only the
    /// AI-path maps cost money; derived maps are computed locally for free.
    pub fn generate_estimate(&self) -> (usize, f64) {
        use lumagen_core::derive::DerivePath;
        let n = MapId::DERIVED
            .iter()
            .filter(|id| !self.map(**id).status.has_image() && self.map(**id).derive_path == DerivePath::Ai)
            .count();
        (n, n as f64 * self.cost_per_image())
    }

    /// Cancel the whole in-flight generation batch (the job-queue "Stop" control). Cancels
    /// every per-map token individually — with extended batches there can be several parent
    /// launches, so the single parent token alone doesn't cover all running maps.
    pub fn cancel_generation(&mut self) {
        if let Some(c) = self.gen_cancel.take() {
            c.cancel();
        }
        for (_, token) in self.gen_map_cancel.drain() {
            token.cancel();
        }
        for id in MapId::DERIVED {
            if matches!(self.map(id).status, MapStatus::Queued | MapStatus::Generating) {
                self.map_mut(id).status = MapStatus::Empty;
            }
        }
        self.generating = false;
        self.queue.clear();
        self.toast("Generation stopped", ToastKind::Info);
    }

    /// Cancel a single map's in-flight generation (its job-queue "cancel" control) — this really
    /// signals that map's provider future via its child token, so it stops and is not charged.
    pub fn cancel_map(&mut self, ctx: &egui::Context, id: MapId) {
        let launched = if let Some(token) = self.gen_map_cancel.remove(&id) {
            token.cancel();
            true // the future reports back as canceled, advancing the batch counter
        } else {
            false
        };
        self.map_mut(id).status = MapStatus::Empty;
        self.queue.retain(|q| *q != id);
        if !launched && self.generating {
            // Cancelled before its task launched — no future will report, so shrink the batch.
            self.gen_total = self.gen_total.saturating_sub(1);
            self.finish_batch_if_done(ctx);
        }
    }

    /// The `.lumagen` path for the current material, under the configured library location.
    pub fn document_path(&self) -> std::path::PathBuf {
        let base = Self::shelve(&self.settings.library_location);
        let name = if self.material_name.is_empty() {
            "material"
        } else {
            self.material_name.as_str()
        };
        base.join(format!("{name}.lumagen"))
    }

    /// Short label for the configured export size (the library card's res badge).
    pub fn export_res_short(&self) -> &'static str {
        match self.settings.final_size {
            0 => "4K",
            1 => "2K",
            3 => "8K",
            4 => "16K",
            _ => "1K",
        }
    }

    /// Rename the open material: moves the saved `.lumagen` document (when present), rekeys
    /// the library entry + preview, and points future saves at the new slug. Returns false
    /// (with a toast) when the name is empty, unchanged, or collides with another material.
    pub fn rename_material(&mut self, new_slug: String) -> bool {
        if new_slug.is_empty() || new_slug == self.material_name {
            return false;
        }
        if self.library.iter().any(|m| m.slug == new_slug) {
            self.toast(format!("A material named “{new_slug}” already exists"), ToastKind::Error);
            return false;
        }
        let old_path = self.document_path();
        let old_slug = std::mem::replace(&mut self.material_name, new_slug.clone());
        let new_path = self.document_path();
        if old_path.exists()
            && let Err(e) = std::fs::rename(&old_path, &new_path)
        {
            // Not fatal — the next save writes the new path; the old file just lingers.
            tracing::warn!("renaming document failed: {e}");
        }
        if let Some(entry) = self.library.iter_mut().find(|m| m.slug == old_slug) {
            entry.slug.clone_from(&new_slug);
            if entry.doc_path.is_some() {
                entry.doc_path = Some(new_path);
            }
        }
        if let Some(preview) = self.lib_previews.remove(&old_slug) {
            self.lib_previews.insert(new_slug, preview);
        }
        self.lib_pending.remove(&old_slug);
        self.unsaved = true;
        true
    }

    /// The document's metadata (identity, prompts, export choices) with no map payloads.
    fn document_meta(&self) -> lumagen_core::document::Document {
        use lumagen_core::document::Document;
        let mut doc = Document {
            material_name: self.material_name.clone(),
            project_name: self.project_name.clone(),
            mat_type: self.mat_type.clone(),
            description: self.description.clone(),
            seed: self.seed,
            gen_size: self.gen_size,
            engine: self.engine.clone(),
            export_pattern: self.export_pattern.clone(),
            base_template: self.settings.base_template.clone(),
            final_requirement: self.settings.final_requirement.clone(),
            negative_prompt: self.settings.negative_prompt.clone(),
            ..Default::default()
        };
        for (k, v) in &self.settings.map_templates {
            doc.map_templates.insert(format!("{}", k.index()), v.clone());
        }
        doc
    }

    /// The per-map document params for a map with image content.
    fn map_doc_params(&self, id: MapId) -> lumagen_core::document::MapDocParams {
        use lumagen_core::derive::DerivePath;
        let m = self.map(id);
        lumagen_core::document::MapDocParams {
            strength: m.strength,
            invert: m.invert,
            flip_y: m.flip_y,
            radius: m.radius,
            bit_depth_16: m.bit_depth_16,
            fidelity: m.fidelity,
            detail_routing: m.detail_routing,
            active_version: m.active_version,
            locked: m.status == MapStatus::Locked,
            ai_path: m.derive_path == DerivePath::Ai,
            version_seeds: m.versions.iter().map(|v| v.seed).collect(),
        }
    }

    /// Serialize the current material + maps + prompt overrides into a `Document` (core).
    /// Working-size pixels only — used by the undo stack and tests; the on-disk save goes
    /// through the worker with full-resolution masters (`save_document`).
    pub fn to_document(&self) -> lumagen_core::document::Document {
        let mut doc = self.document_meta();
        for id in MapId::ALL {
            if !self.map(id).status.has_image() {
                continue;
            }
            // Persist the real current pixels (generated / imported / derived) — never the
            // albedo as a stand-in for a missing map.
            let Some(rgba) = self.export_pixels(id) else { continue };
            // Encode failure loses this map from the snapshot/round-trip — loud, never silent.
            if let Err(e) = doc.set_map(id, &rgba, self.map_doc_params(id)) {
                tracing::error!(?id, "map PNG encode failed while building document: {e}");
            }
        }
        doc
    }

    /// Expand a leading `~` in a configured path to the user's home directory.
    fn shelve(p: &str) -> std::path::PathBuf {
        if let Some(rest) = p.strip_prefix("~/")
            && let Some(home) = directories::BaseDirs::new().map(|b| b.home_dir().to_path_buf())
        {
            return home.join(rest);
        }
        std::path::PathBuf::from(p)
    }

    /// Save the current document to its `.lumagen` path — asynchronously, on the worker:
    /// PNG-encoding full-resolution masters would stall the frame loop. The outcome arrives
    /// as `IoResult::DocSaved`. Returns the target path.
    #[tracing::instrument(skip(self, ctx))]
    pub fn save_document(&mut self, ctx: &egui::Context, announce: bool) -> std::path::PathBuf {
        let path = self.document_path();
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            self.toast(format!("Save failed: {e}"), ToastKind::Error);
            return path;
        }
        let doc = self.document_meta();
        // Each map's best bytes: an untouched opened master passes its ENCODED section
        // through (no decode, no re-encode — the autosave win); a fresh/decoded-and-newer
        // master encodes from pixels; else the working copy.
        let mut maps = Vec::new();
        for id in MapId::ALL {
            if !self.map(id).status.has_image() {
                continue;
            }
            let params = self.map_doc_params(id);
            if let Some((png, preview, side)) = self.master_encoded.get(&id) {
                let working = self.map_pixels.get(&id).cloned().unwrap_or_else(|| std::sync::Arc::new(Vec::new()));
                maps.push(render::SaveMap {
                    id,
                    source: render::SaveMapSource::Encoded {
                        png: std::sync::Arc::clone(png),
                        preview: std::sync::Arc::clone(preview),
                        working,
                    },
                    side: *side,
                    params,
                });
            } else if let Some(master) = self.master_pixels.get(&id) {
                maps.push(render::SaveMap {
                    id,
                    source: render::SaveMapSource::Pixels(std::sync::Arc::clone(&master.0)),
                    side: master.1,
                    params,
                });
            } else if let Some(px) = self.map_pixels.get(&id) {
                // The working copy is already Arc-held; `export_pixels` would deep-clone it.
                maps.push(render::SaveMap {
                    id,
                    source: render::SaveMapSource::Pixels(std::sync::Arc::clone(px)),
                    side: maps::TEX_SIZE,
                    params,
                });
            } else if let Some(rgba) = self.export_pixels(id) {
                maps.push(render::SaveMap {
                    id,
                    source: render::SaveMapSource::Pixels(std::sync::Arc::new(rgba)),
                    side: maps::TEX_SIZE,
                    params,
                });
            }
        }
        self.submit_io(
            ctx,
            IoTask::SaveDoc(Box::new(render::SaveJob {
                path: path.clone(),
                doc,
                maps,
                displacement_height: self.master_height.get(&MapId::Displacement).cloned(),
                announce,
            })),
        );
        // Exit waits for queued saves (drain_pending_saves_on_exit) — closing the window
        // right after Ctrl+S/autosave must not abandon the write.
        self.pending_saves += 1;
        // Optimistic: DocSaved flips it back on failure.
        self.unsaved = false;
        // Keep the library entry's pips honest: refresh coverage/class/project on save, point
        // it at the saved document, and drop the cached grid preview so it re-renders from
        // the document's real albedo.
        let coverage = self.coverage();
        let (slug, class, project) = (self.material_name.clone(), self.mat_type.clone(), self.project_name.clone());
        if let Some(entry) = self.library.iter_mut().find(|m| m.slug == slug) {
            entry.coverage = coverage;
            entry.class = class;
            entry.project = project;
            entry.seed = self.seed;
            entry.doc_path = Some(path.clone());
        }
        self.lib_previews.remove(&slug);
        self.lib_pending.remove(&slug);
        path
    }

    /// Autosave the document when the preference is on. Silent on success (a toast per
    /// generated map would be noise); failures surface via `DocSaved`.
    pub fn autosave(&mut self, ctx: &egui::Context) {
        if !self.settings.autosave || self.material_name.trim().is_empty() {
            return;
        }
        let path = self.save_document(ctx, false);
        tracing::info!("autosave dispatched → {}", path.display());
    }

    /// Drop the `var_*` preview textures the variations modal inserted into `lib_previews`.
    /// Without this, every variations session permanently accumulates 4 full GPU textures.
    pub fn evict_variation_previews(&mut self) {
        self.lib_previews.retain(|k, _| !k.starts_with("var_"));
        self.var_pending.retain(|k| !k.starts_with("var_"));
        self.var_pixels.retain(|k, _| !k.starts_with("var_"));
        self.var_masters.retain(|k, _| !k.starts_with("var_"));
        self.pending_variations = None;
    }

    /// Promote a variation candidate into the map slot: install its real pixels (and its
    /// full-resolution master) as the map's content, push a version, mark ready. The other
    /// maps stay untouched.
    pub fn promote_variation(&mut self, ctx: &egui::Context, map: MapId, slot: usize) {
        let key = format!("var_{:?}_{}", map, slot);
        let Some(rgba) = self.var_pixels.get(&key).cloned() else { return };
        self.snapshot();
        let px = std::sync::Arc::new(rgba);
        self.install_pixels(ctx, map, std::sync::Arc::clone(&px));
        let master = self.var_masters.get(&key).cloned().map(|(mpx, side)| (std::sync::Arc::new(mpx), side));
        self.set_master(map, master);
        let seed = self.seed;
        let master_arc = self.master_pixels.get(&map).cloned();
        self.push_version(map, seed, Some(px), master_arc);
        self.map_mut(map).status = MapStatus::Ready;
        self.unsaved = true;
    }

    // ── Texture management (async: rasterize off-thread, upload on the UI thread) ──

    /// Lazily create the render worker the first time we have a `Context` to hand it.
    fn worker(&mut self, ctx: &egui::Context) -> &RenderWorker {
        self.render_worker.get_or_insert_with(|| RenderWorker::spawn(ctx.clone()))
    }

    /// The sender a generation task uses to push finished maps back and wake the loop.
    pub fn generation_sender(&mut self, ctx: &egui::Context) -> render::GenerationHandle {
        self.worker(ctx).generation_sender()
    }

    /// Submit a blocking I/O task (vault access, export) to the background worker, off the UI
    /// thread. Public wrapper so UI modules don't touch the private worker accessor.
    pub fn submit_io(&mut self, ctx: &egui::Context, task: IoTask) {
        self.worker(ctx).submit_io(task);
    }

    /// The current albedo's RGBA8 pixels at the working size (the reference for every
    /// derived/semantic map and the source for derivation). Rendered on demand.
    pub fn albedo_rgba(&self) -> Vec<u8> {
        maps::render_map(MapId::Albedo, &self.layout, self.seed)
    }

    /// The RGBA8 pixels for a map at the working size, for the wgpu preview. Generated/imported
    /// maps come from the `map_pixels` cache; the albedo falls back to the procedural render and
    /// is cached so the preview doesn't re-rasterize it every frame.
    pub fn map_rgba(&mut self, id: MapId) -> Option<std::sync::Arc<Vec<u8>>> {
        if let Some(px) = self.map_pixels.get(&id) {
            return Some(std::sync::Arc::clone(px));
        }
        if id == MapId::Albedo {
            let px = std::sync::Arc::new(self.albedo_rgba());
            self.map_pixels.insert(MapId::Albedo, std::sync::Arc::clone(&px));
            return Some(px);
        }
        None
    }

    /// The albedo pixels to export/derive from: the real imported/generated albedo if present,
    /// else the procedural render.
    pub fn export_albedo(&self) -> Vec<u8> {
        self.map_pixels.get(&MapId::Albedo).map(|a| (**a).clone()).unwrap_or_else(|| self.albedo_rgba())
    }

    /// The RGBA8 pixels to write for `id` on export: the real generated/imported/promoted pixels
    /// when present; else a fresh deterministic derivation for a geometric map; else `None` for a
    /// semantic (AI) map that was never generated (so export never writes the albedo as a
    /// stand-in for a missing roughness/metallic/emission map).
    pub fn export_pixels(&self, id: MapId) -> Option<Vec<u8>> {
        if let Some(px) = self.map_pixels.get(&id) {
            return Some((**px).clone());
        }
        if id == MapId::Albedo {
            return Some(self.export_albedo());
        }
        use lumagen_core::derive::DerivePath;
        let is_geometric =
            self.map(id).derive_path == DerivePath::Derived || matches!(id, MapId::Normal | MapId::Ao | MapId::Transparency | MapId::Displacement);
        if is_geometric {
            let albedo = self.export_albedo();
            let params = self.map(id).derive_params();
            return Some(lumagen_core::derive::derive_map(id, &albedo, None, maps::TEX_SIZE, &params));
        }
        None
    }

    /// Whether a map is a geometric (deterministically derived) map whose adjust sliders
    /// re-derive live pixels.
    fn is_geometric(id: MapId) -> bool {
        matches!(id, MapId::Normal | MapId::Ao | MapId::Transparency | MapId::Displacement)
    }

    /// Mark a geometric map dirty so its adjust sliders re-derive it (debounced) — the
    /// mechanism that makes the "adjust" sliders actually change the on-screen pixels.
    pub fn mark_adjust_dirty(&mut self, id: MapId, now: f64) {
        // Only derived-path maps re-derive live; an AI-path map's pixels must never be
        // stomped by a local derivation (any map can be promoted to AI now).
        if id != MapId::Albedo && self.map(id).derive_path == lumagen_core::derive::DerivePath::Derived {
            self.adjust_dirty.insert(id);
            self.adjust_dirty_at = now;
        }
    }

    /// After a short debounce, submit an off-thread re-derive for every dirty geometric map so
    /// the 2D/3D preview reflects the current strength/invert/flip_y/radius/detail_routing.
    pub fn flush_adjust(&mut self, ctx: &egui::Context) {
        if self.adjust_dirty.is_empty() {
            return;
        }
        let now = ctx.input(|i| i.time);
        if now - self.adjust_dirty_at < 0.12 {
            ctx.request_repaint(); // keep the loop ticking so the debounce fires
            return;
        }
        let ids: Vec<MapId> = self.adjust_dirty.drain().collect();
        let albedo = std::sync::Arc::new(self.export_albedo());
        for id in ids {
            let params = self.map(id).derive_params();
            let epoch = self.material_epoch;
            self.submit_io(
                ctx,
                IoTask::Derive {
                    id,
                    albedo: std::sync::Arc::clone(&albedo),
                    params,
                    size: maps::TEX_SIZE,
                    epoch,
                },
            );
        }
    }

    /// Open the OS file manager at `path` (reveal-in-folder after a successful export).
    fn reveal_in_folder(path: &std::path::Path) {
        #[cfg(target_os = "windows")]
        let _ = std::process::Command::new("explorer").arg(path).spawn();
        #[cfg(target_os = "macos")]
        let _ = std::process::Command::new("open").arg(path).spawn();
        #[cfg(all(unix, not(target_os = "macos")))]
        let _ = std::process::Command::new("xdg-open").arg(path).spawn();
    }

    /// The albedo edge-continuity score, cached and recomputed only when the seed/layout
    /// changes (recomputing the albedo every frame is too expensive).
    pub fn seam_score(&mut self) -> f32 {
        if self.seam_cache_seed != Some(self.seed) {
            let score = lumagen_core::export::seam_score(&self.export_albedo(), maps::TEX_SIZE);
            self.seam_cache = score;
            self.seam_cache_seed = Some(self.seed);
        }
        self.seam_cache
    }

    /// Enqueue a map render (full + thumbnail) off the UI thread.
    pub fn request_map_render(&mut self, ctx: &egui::Context, id: MapId, seed: u32) {
        let epoch = self.material_epoch;
        self.worker(ctx).submit(RenderRequest {
            target: RenderTarget::Map(id),
            seed,
            epoch,
        });
    }

    /// Enqueue a library-preview render off the UI thread.
    pub fn request_lib_preview(&mut self, ctx: &egui::Context, slug: String, seed: u32) {
        let epoch = self.material_epoch;
        self.worker(ctx).submit(RenderRequest {
            target: RenderTarget::LibPreview(slug),
            seed,
            epoch,
        });
    }

    /// Load every provider's API key from the OS vault, off the UI thread. Called once at
    /// startup; results arrive via `process_worker_results`.
    pub fn request_load_keys(&mut self, ctx: &egui::Context) {
        for provider in [Provider::Fal, Provider::OpenRouter, Provider::OpenAi, Provider::Llm] {
            self.worker(ctx).submit_io(IoTask::LoadKey(provider));
        }
    }

    /// Save a provider's API key to the OS vault, off the UI thread. The key moves from the
    /// edit buffer to the vault and the buffer is cleared, so no plaintext lingers in state.
    /// Saving an emptied field deletes the stored key — clearing must not be a silent no-op.
    pub fn save_key(&mut self, ctx: &egui::Context, provider: Provider) {
        let (set, input) = self.settings.key_slot(provider);
        let stored = *set;
        let key = std::mem::take(input);
        if key.is_empty() {
            if stored {
                self.worker(ctx).submit_io(IoTask::DeleteKey(provider));
            }
            return;
        }
        self.worker(ctx).submit_io(IoTask::StoreKey(provider, key));
    }

    /// Drain finished work (renders + I/O + generated maps) and apply it. Called from `logic`
    /// every frame — the only GPU-touching part (`load_texture`) runs here; the expensive
    /// rasterization, vault I/O, and provider calls already happened off-thread.
    pub fn process_worker_results(&mut self, ctx: &egui::Context) {
        let Some(worker) = self.render_worker.as_mut() else { return };
        for out in worker.drain() {
            match out {
                Output::Rendered(res) => self.install_render(ctx, res),
                Output::Io(io) => self.apply_io(ctx, io),
                Output::Generated(outcome) => self.apply_generated(ctx, outcome),
                Output::Async(render::AsyncMsg::Toast { msg, kind }) => self.toast(msg, kind),
                Output::Async(render::AsyncMsg::SetDescription(d)) => {
                    // Auto-apply only if the text is untouched since the enhance launched —
                    // edits the user typed while the LLM ran must never be clobbered.
                    if self.enhance_baseline.take().as_deref() == Some(self.description.as_str()) {
                        self.description = d;
                        self.unsaved = true;
                    } else {
                        self.toast("Description edited while enhancing — suggestion discarded", ToastKind::Info);
                    }
                }
                Output::Async(render::AsyncMsg::TestResult { provider, result }) => {
                    let status = match result {
                        Ok(msg) => TestStatus::Ok(msg),
                        Err(e) => TestStatus::Failed(e),
                    };
                    self.settings.test_status.insert(provider, status);
                }
            }
        }
    }

    /// A handle a tokio network task uses to toast its result back to the UI (waking the loop).
    pub fn async_handle(&mut self, ctx: &egui::Context) -> render::AsyncHandle {
        self.worker(ctx).async_handle(ctx.clone())
    }

    /// Queue a real "test connection" for `provider`; the app picks up `pending_test` and runs
    /// the request on the tokio runtime. The result lands both as a toast and as the inline
    /// status in the provider's settings card (set to Pending immediately for feedback).
    pub fn test_connection(&mut self, ctx: &egui::Context, provider: Provider) {
        // Ensure the worker (and its async channel) exists before the app spawns the task.
        let _ = self.worker(ctx);
        self.settings.test_status.insert(provider, crate::state::TestStatus::Pending);
        self.pending_test = Some(provider);
    }

    /// Apply a finished generated/derived map: upload its texture, mark it ready, accumulate
    /// spend, and advance the batch. Failures mark the map errored and toast.
    fn apply_generated(&mut self, ctx: &egui::Context, outcome: lumagen_core::generate::MapOutcome) {
        use lumagen_core::generate::MapOutcome;
        match outcome {
            MapOutcome::Ready { map, rgba, master, height, cost } => {
                let was_upscale = self.upscaling.remove(&map);
                // Only a generation result may retire generation state — an upscale finishing
                // mid-regeneration must not clobber the regen's cancel token or progress.
                if !was_upscale {
                    self.gen_map_cancel.remove(&map);
                }
                let px = std::sync::Arc::new(rgba);
                self.install_pixels(ctx, map, std::sync::Arc::clone(&px));
                self.set_master(map, master.map(|(mpx, side)| (std::sync::Arc::new(mpx), side)));
                // Keep the f32 displacement master for a TRUE 16-bit/EXR export. A displacement
                // result WITHOUT one (derived, upscaled, imported) supersedes any stale height.
                if let Some(h) = height {
                    self.master_height.insert(map, h);
                } else if map == MapId::Displacement {
                    self.master_height.remove(&map);
                }
                let seed = self.seed;
                let master_arc = self.master_pixels.get(&map).cloned();
                self.push_version(map, seed, Some(px), master_arc);
                self.map_mut(map).status = MapStatus::Ready;
                if map == MapId::Albedo {
                    self.albedo_error = None;
                    if !was_upscale {
                        self.albedo_cancel = None;
                    }
                }
                // Charge the real provider cost; derived + mock maps report 0.0, so they add
                // nothing — a purely procedural/offline result is never billed.
                let charge = cost.unwrap_or_else(|| self.cost_per_image());
                self.add_spend(charge);
                self.unsaved = true;
                if map == MapId::Albedo {
                    // A standalone albedo (text-to-image or upscale) result — not batched.
                    self.seam_cache_seed = None;
                    // Step ① done; the flow moves to Generate (unless the set is complete).
                    if !was_upscale && !self.generated_all() {
                        self.active_step = 1;
                    }
                    // Derived maps that waited for this new master re-derive from it now.
                    if was_upscale && !self.pending_rederive.is_empty() {
                        let ids = std::mem::take(&mut self.pending_rederive);
                        self.spawn_rederives(ctx, ids);
                    }
                    self.autosave(ctx);
                } else if was_upscale {
                    // An upscale is not part of a generation batch — no counter to advance.
                    if self.upscaling.is_empty() {
                        self.autosave(ctx);
                    }
                } else {
                    self.gen_done += 1;
                    self.finish_batch_if_done(ctx);
                }
            }
            MapOutcome::Failed { map, error } => {
                // A failed variations run must not touch the underlying (perfectly good) map
                // or the generation batch counters — it only closes out the variations state.
                let var_prefix = format!("var_{:?}_", map);
                if self.var_pending.iter().any(|k| k.starts_with(&var_prefix)) {
                    self.var_pending.retain(|k| !k.starts_with(&var_prefix));
                    self.variations_cancel = None;
                    if error.contains("canceled") {
                        tracing::info!(?map, "variations run canceled");
                    } else {
                        tracing::error!(?map, "variations run failed: {error}");
                        self.toast(format!("Variations failed: {error}"), ToastKind::Error);
                    }
                    return;
                }
                let was_upscale = self.upscaling.remove(&map);
                if !was_upscale {
                    self.gen_map_cancel.remove(&map);
                }
                if was_upscale {
                    // A failed upscale leaves the map's existing (perfectly good) pixels and
                    // status untouched — it only surfaces the error.
                    if map == MapId::Albedo && !self.pending_rederive.is_empty() {
                        // Re-derives waiting on the new albedo master have nothing to gain now.
                        for id in std::mem::take(&mut self.pending_rederive) {
                            self.upscaling.remove(&id);
                        }
                    }
                    tracing::error!(?map, "upscale failed: {error}");
                    self.toast(format!("{} upscale failed: {}", crate::data::map_def(map).name, error), ToastKind::Error);
                    return;
                }
                if error.contains("canceled") {
                    // A user-cancelled map is not an error and is not charged.
                    if self.map(map).status != MapStatus::Empty {
                        self.map_mut(map).status = MapStatus::Empty;
                    }
                    if map == MapId::Albedo {
                        self.albedo_cancel = None;
                    }
                    tracing::info!(?map, "map generation canceled");
                } else {
                    tracing::error!(?map, "map generation failed: {error}");
                    self.map_mut(map).status = MapStatus::Error;
                    if map == MapId::Albedo {
                        // Keep the failure on screen (viewport + right panel), not only in a
                        // 3-second toast — a missed toast made failures look like fake successes.
                        self.albedo_error = Some(error.clone());
                        self.albedo_cancel = None;
                    }
                    self.toast(format!("{} failed: {}", crate::data::map_def(map).name, error), ToastKind::Error);
                }
                if map != MapId::Albedo {
                    self.gen_done += 1;
                    self.finish_batch_if_done(ctx);
                }
            }
            MapOutcome::Variations {
                map,
                candidates,
                masters,
                cost,
            } => {
                for (k, rgba) in candidates.iter().enumerate() {
                    let key = format!("var_{:?}_{}", map, k);
                    self.lib_previews
                        .insert(key.clone(), ctx.load_texture(key.clone(), render::full_image(rgba), TextureOptions::LINEAR));
                    self.var_pixels.insert(key.clone(), rgba.clone());
                    if let Some(Some((mpx, side))) = masters.get(k) {
                        self.var_masters.insert(key.clone(), (mpx.clone(), *side));
                    }
                    self.var_pending.remove(&key);
                }
                self.variations_cancel = None;
                let charge = cost.unwrap_or_else(|| self.cost_per_image() * candidates.len() as f64);
                self.add_spend(charge);
            }
        }
    }

    /// When every map in the batch has reported (ready or failed), close out the run.
    fn finish_batch_if_done(&mut self, ctx: &egui::Context) {
        if self.generating && self.gen_done >= self.gen_total {
            self.generating = false;
            self.gen_cancel = None;
            self.gen_map_cancel.clear();
            if self.generated_all() {
                self.active_step = 2;
                tracing::info!("generation finished: all 7 maps aligned to albedo");
                self.toast("7 maps generated · aligned to albedo", ToastKind::Success);
            } else {
                let failed = MapId::DERIVED.iter().filter(|id| self.map(**id).status == MapStatus::Error).count();
                if failed > 0 {
                    self.toast(format!("{} map{} failed", failed, if failed == 1 { "" } else { "s" }), ToastKind::Error);
                }
            }
            self.autosave(ctx);
        }
    }

    fn install_render(&mut self, ctx: &egui::Context, res: render::RenderResult) {
        match res.target {
            RenderTarget::Map(id) => {
                if res.epoch != self.material_epoch {
                    tracing::info!(?id, "dropping map render finished for a previous material session");
                    return;
                }
                let name = format!("{:?}_{}", id, res.seed);
                self.textures.insert(
                    id,
                    ctx.load_texture(format!("{}_full", name), render::full_image(&res.full), TextureOptions::LINEAR),
                );
                self.thumbs.insert(
                    id,
                    ctx.load_texture(format!("{}_thumb", name), render::thumb_image(&res.thumb), TextureOptions::LINEAR),
                );
            }
            // Library previews are keyed by slug, not material session — they stay valid
            // across workspace switches.
            RenderTarget::LibPreview(slug) => {
                self.lib_previews.insert(
                    slug.clone(),
                    ctx.load_texture(format!("lib_{slug}"), render::full_image(&res.full), TextureOptions::LINEAR),
                );
                self.lib_pending.remove(&slug);
            }
        }
    }

    fn apply_io(&mut self, ctx: &egui::Context, io: IoResult) {
        match io {
            IoResult::KeyLoaded(provider, key) => {
                let (set, input) = self.settings.key_slot(provider);
                *set = key.is_some();
                if let Some(k) = key {
                    *input = k;
                }
            }
            IoResult::KeyStored(provider, ok) => {
                *self.settings.key_slot(provider).0 = ok;
                if ok {
                    self.toast("API key saved to your OS keychain", ToastKind::Success);
                } else {
                    self.toast("Couldn't reach the system keychain — key not saved", ToastKind::Error);
                }
            }
            IoResult::KeyDeleted(provider, ok) => {
                if ok {
                    *self.settings.key_slot(provider).0 = false;
                    self.toast("API key removed from your OS keychain", ToastKind::Success);
                } else {
                    self.toast("Couldn't reach the system keychain — key not removed", ToastKind::Error);
                }
            }
            IoResult::Exported(result) => match result {
                Ok(count) => {
                    tracing::info!("export wrote {count} files");
                    self.toast(format!("Exported {} files", count), ToastKind::Success);
                    if let Some(dir) = self.last_export_dir.take() {
                        Self::reveal_in_folder(&dir);
                    }
                }
                Err(e) => {
                    self.error_toast(&format!("export failed: {e}"), format!("Export failed: {e}"));
                    self.last_export_dir = None;
                }
            },
            IoResult::AlbedoImported { result, epoch } => {
                if epoch != self.material_epoch {
                    tracing::info!("dropping albedo import finished for a previous material session");
                    return;
                }
                match result {
                    Ok((rgba, name)) => self.import_albedo(ctx, rgba, &name),
                    Err(e) => self.error_toast(&format!("albedo import failed: {e}"), format!("Import failed: {e}")),
                }
            }
            IoResult::Derived { id, full, thumb, size, epoch } => {
                if epoch != self.material_epoch {
                    tracing::info!(?id, "dropping re-derive finished for a previous material session");
                    return;
                }
                if size > maps::TEX_SIZE {
                    // A Refine re-derive at the albedo master's resolution: install as the
                    // map's new master + working preview, versioned like an upscale.
                    self.upscaling.remove(&id);
                    let working = maps::resize_square_rgba(&full, size, maps::TEX_SIZE);
                    let px = std::sync::Arc::new(working);
                    self.install_pixels(ctx, id, std::sync::Arc::clone(&px));
                    let master = (std::sync::Arc::new(full), size);
                    self.set_master(id, Some(master.clone()));
                    let seed = self.seed;
                    self.push_version(id, seed, Some(px), Some(master));
                    self.map_mut(id).status = MapStatus::Ready;
                    self.unsaved = true;
                    if self.upscaling.is_empty() && self.pending_rederive.is_empty() {
                        self.autosave(ctx);
                    }
                } else {
                    // A live adjust re-derive: refresh the pixels/textures in place, no version bump.
                    self.map_pixels.insert(id, std::sync::Arc::new(full.clone()));
                    self.textures.insert(
                        id,
                        ctx.load_texture(format!("{id:?}_adj_full"), render::full_image(&full), TextureOptions::LINEAR),
                    );
                    self.thumbs.insert(
                        id,
                        ctx.load_texture(format!("{id:?}_adj_thumb"), render::thumb_image(&thumb), TextureOptions::LINEAR),
                    );
                    self.unsaved = true;
                }
            }
            IoResult::FolderImported { result, epoch } => {
                if epoch != self.material_epoch {
                    tracing::info!("dropping folder import finished for a previous material session");
                    return;
                }
                match result {
                    Ok(imported) => self.install_imported_folder(ctx, imported),
                    Err(e) => self.error_toast(&format!("folder import failed: {e}"), format!("Import failed: {e}")),
                }
            }
            IoResult::LibraryScanned(result) => {
                self.library_scanning = false;
                match result {
                    Ok(found) => self.install_scanned(found),
                    Err(e) => tracing::warn!("library scan failed: {e}"),
                }
            }
            IoResult::DocOpened { slug, epoch, result } => {
                if epoch != self.material_epoch {
                    tracing::info!(slug, "dropping document open finished for a previous material session");
                    return;
                }
                self.opening = None;
                match result {
                    Ok(opened) => {
                        // Entry-derived identity first (class/project fallbacks), then the
                        // document's own fields win — same order as the synchronous path.
                        if let Some(m) = self.library.iter().find(|m| m.slug == slug).cloned() {
                            self.adopt_entry_identity(&m);
                        }
                        self.apply_opened_document(*opened);
                        self.screen = Screen::Workspace;
                        self.unsaved = false;
                        self.toast(format!("Opened {}", slug), ToastKind::Info);
                    }
                    Err(e) => self.error_toast(&format!("could not open {slug}: {e}"), format!("Couldn't open {slug}: {e}")),
                }
            }
            IoResult::MasterDecoded { id, epoch, result } => {
                if epoch != self.material_epoch {
                    tracing::info!(?id, "dropping master decode finished for a previous material session");
                    return;
                }
                self.masters_loading.remove(&id);
                match result {
                    Ok((rgba, side)) => {
                        let master = (std::sync::Arc::new(rgba), side);
                        self.master_pixels.insert(id, master.clone());
                        self.master_textures.remove(&id);
                        let av = self.map(id).active_version;
                        if let Some(v) = self.map_mut(id).versions.get_mut(av) {
                            v.master = Some(master);
                        }
                    }
                    Err(e) => {
                        // The working copy stays usable; only native-res export/refine of
                        // this map is degraded — surfaced, not silent.
                        tracing::error!(?id, "full-resolution master decode failed: {e}");
                        self.toast(
                            format!("{} native resolution unavailable — using the working copy", crate::data::map_def(id).name),
                            ToastKind::Error,
                        );
                    }
                }
            }
            IoResult::DocPreview { slug, full } => {
                self.lib_previews.insert(
                    slug.clone(),
                    ctx.load_texture(format!("lib_{slug}"), render::full_image(&full), TextureOptions::LINEAR),
                );
                self.lib_pending.remove(&slug);
            }
            IoResult::DocSaved { path, result, announce } => {
                self.pending_saves = self.pending_saves.saturating_sub(1);
                match result {
                    Ok(()) => {
                        tracing::info!("saved document to {}", path.display());
                        if announce {
                            self.toast(format!("Saved {} → {}", self.material_name, path.display()), ToastKind::Success);
                        }
                    }
                    Err(e) => {
                        self.unsaved = true;
                        self.error_toast(&format!("save failed: {e}"), format!("Save failed: {e}"));
                    }
                }
            }
            IoResult::MapOverridden { id, result, epoch } => {
                if epoch != self.material_epoch {
                    tracing::info!(?id, "dropping map override finished for a previous material session");
                    return;
                }
                match result {
                    Ok(rgba) => {
                        self.snapshot();
                        let px = std::sync::Arc::new(rgba);
                        self.install_pixels(ctx, id, std::sync::Arc::clone(&px));
                        let seed = self.seed;
                        // A user override replaces the master too (the file IS the new source).
                        self.set_master(id, None);
                        self.push_version(id, seed, Some(px), None);
                        self.map_mut(id).status = if id == MapId::Albedo { MapStatus::Ready } else { MapStatus::Overridden };
                        if id == MapId::Albedo {
                            self.seam_cache_seed = None;
                        }
                        self.unsaved = true;
                        self.toast(format!("{} replaced with your file", crate::data::map_def(id).name), ToastKind::Success);
                    }
                    Err(e) => self.error_toast(&format!("map override failed: {e}"), format!("Replace failed: {e}")),
                }
            }
        }
    }

    /// Install a material scanned from a folder of loose map images: reset the workspace, install
    /// each decoded map as the user's own pixels, and register the material in the library.
    fn install_imported_folder(&mut self, ctx: &egui::Context, imported: render::ImportedFolder) {
        self.snapshot();
        self.reset_maps_and_stores();
        // Never adopt another material's slug — autosave would overwrite its document.
        // (Re-importing a folder named like the currently open material is intentional.)
        let slug = self.unique_slug(&imported.slug, true);
        if slug != imported.slug {
            self.toast(format!("Named {slug} — \"{}\" already exists in the library", imported.slug), ToastKind::Info);
        }
        self.material_name.clone_from(&slug);
        let seed = self.seed;
        for (id, rgba) in imported.maps {
            let px = std::sync::Arc::new(rgba);
            self.install_pixels(ctx, id, std::sync::Arc::clone(&px));
            self.push_version(id, seed, Some(px), None);
            self.map_mut(id).status = MapStatus::Ready;
        }
        if !self.library.iter().any(|m| m.slug == slug) {
            let coverage = self.coverage();
            self.library.insert(
                0,
                LibMaterial {
                    slug: slug.clone(),
                    class: "Custom".into(),
                    project: self.project_name.clone(),
                    res: "4K".into(),
                    coverage,
                    stale_index: None,
                    seed,
                    doc_path: None,
                },
            );
        }
        self.seam_cache_seed = None;
        self.screen = Screen::Workspace;
        self.unsaved = true;
        self.toast("Imported material from folder", ToastKind::Success);
        self.autosave(ctx);
    }

    /// Ensure every map with image data has a render in flight (or already uploaded), and
    /// kick off any missing library previews — all off the UI thread.
    pub fn ensure_textures(&mut self, ctx: &egui::Context) {
        for id in MapId::ALL {
            let needs = self.has_map(id) && !self.textures.contains_key(&id);
            if needs {
                // Prefer the real generated/imported/loaded pixels; only fall back to the
                // procedural stand-in when a map has no stored pixels yet.
                if let Some(px) = self.map_pixels.get(&id).cloned() {
                    self.install_pixels(ctx, id, px);
                } else {
                    let seed = self.map(id).versions.last().map(|v| v.seed).unwrap_or(self.seed);
                    self.request_map_render(ctx, id, seed);
                }
            }
        }
        // Library previews: submit any that are missing and not already queued. A material
        // with a saved document gets its REAL albedo as the preview; only doc-less entries
        // (fresh imports not yet saved) fall back to the procedural render.
        for index in 0..self.library.len() {
            let seed = self.library[index].seed;
            let slug = &self.library[index].slug;
            if self.lib_previews.contains_key(slug) || self.lib_pending.contains(slug) {
                continue;
            }
            let slug = slug.clone();
            self.lib_pending.insert(slug.clone());
            match self.library[index].doc_path.clone() {
                Some(path) => self.submit_io(ctx, IoTask::DocPreview { slug, path, seed }),
                None => self.request_lib_preview(ctx, slug, seed),
            }
        }
    }

    // ── Generation ───────────────────────────────────────────────────────

    /// Start generating every map that isn't ready/locked/overridden. Marks the batch and
    /// asks the app to launch it on the tokio runtime (`pending_generate`).
    #[tracing::instrument(skip(self))]
    pub fn start_generate_all(&mut self) {
        if self.generating {
            return;
        }
        // Every map derives from the albedo — never generate a set against the procedural
        // stand-in while the real albedo is still generating (or failed).
        if !self.has_map(MapId::Albedo) {
            self.toast("Generate the albedo first — every map derives from it", ToastKind::Cost);
            return;
        }
        tracing::info!("generation started");
        let missing: Vec<MapId> = MapId::DERIVED.iter().copied().filter(|id| !self.map(*id).status.has_image()).collect();
        if missing.is_empty() {
            self.toast("All 7 maps already generated", ToastKind::Info);
            return;
        }
        self.snapshot();
        self.begin_batch(missing);
    }

    /// Batch setup: mark maps queued (they transition to Generating as the provider picks
    /// them up) and flag the app to launch them. While a batch is already running this
    /// **extends** it — each launch is its own tokio task, so single-map generations run
    /// concurrently instead of being blocked until the previous one finishes.
    fn begin_batch(&mut self, maps: Vec<MapId>) {
        if self.generating {
            self.gen_total += maps.len();
        } else {
            if let Some(c) = self.gen_cancel.take() {
                c.cancel();
            }
            self.gen_total = maps.len();
            self.gen_done = 0;
            self.generating = true;
        }
        for id in &maps {
            self.map_mut(*id).status = MapStatus::Queued;
        }
        // `queue` is the launch buffer: the app drains it into the next spawned task.
        for id in maps {
            if !self.queue.contains(&id) {
                self.queue.push(id);
            }
        }
        self.active_step = 1;
        self.queue_expanded = true;
        self.pending_generate = true;
    }

    /// Restore a prior version's pixels into the map slot (non-destructive: the version list
    /// is untouched, only the active pointer moves and textures re-render from that seed).
    pub fn restore_version(&mut self, ctx: &egui::Context, id: MapId, index: usize) {
        let m = self.map(id);
        if index >= m.versions.len() || index == m.active_version {
            return;
        }
        let seed = m.versions[index].seed;
        let n = m.versions[index].n;
        let pixels = m.versions[index].pixels.clone();
        let master = m.versions[index].master.clone();
        self.snapshot();
        self.map_mut(id).active_version = index;
        // Non-destructive: reinstate the version's own master when it was kept (recent
        // versions keep theirs, so upscales roll back losslessly); otherwise drop the stale
        // one rather than export mismatched pixels.
        self.set_master(id, master);
        if id == MapId::Albedo {
            self.seed = seed;
            self.layout = maps::build_layout(seed);
            self.seam_cache_seed = None;
        }
        if let Some(px) = pixels {
            // Exact restore from the kept pixels.
            self.install_pixels(ctx, id, px);
        } else {
            self.textures.remove(&id);
            self.thumbs.remove(&id);
            self.map_pixels.remove(&id);
            if Self::is_geometric(id) {
                // Re-derive from the albedo + current params (aligned) rather than the procedural
                // stand-in.
                let albedo = std::sync::Arc::new(self.export_albedo());
                let params = self.map(id).derive_params();
                let epoch = self.material_epoch;
                self.submit_io(
                    ctx,
                    IoTask::Derive {
                        id,
                        albedo,
                        params,
                        size: maps::TEX_SIZE,
                        epoch,
                    },
                );
            } else if id != MapId::Albedo {
                self.request_map_render(ctx, id, seed);
            }
            // Albedo with no kept pixels re-renders procedurally from the seed via ensure_textures.
        }
        self.unsaved = true;
        self.toast(format!("Restored v{} of {}", n, crate::data::map_def(id).name), ToastKind::Info);
    }

    /// Launch a real albedo generation for the current description/seed: clear any stale
    /// placeholder pixels, show a genuine Generating state, and queue the provider call.
    pub fn generate_albedo_now(&mut self) {
        self.albedo_error = None;
        self.textures.remove(&MapId::Albedo);
        self.thumbs.remove(&MapId::Albedo);
        self.map_pixels.remove(&MapId::Albedo);
        self.set_master(MapId::Albedo, None);
        self.seam_cache_seed = None;
        self.map_mut(MapId::Albedo).status = MapStatus::Generating;
        self.active_step = 0;
        self.pending_albedo_gen = true;
        self.unsaved = true;
        let ab = self.active_backend();
        self.toast(
            if ab.live {
                format!("Generating albedo via {}…", ab.t2i)
            } else {
                "Generating albedo offline (no API key set)…".to_string()
            },
            ToastKind::Info,
        );
    }

    /// The export target side in pixels (Settings → Generation → Final size).
    pub fn export_side(&self) -> usize {
        match self.settings.final_size {
            0 => 4096,
            1 => 2048,
            3 => 8192,
            4 => 16384,
            _ => 1024,
        }
    }

    /// The best source pixels for upscaling a map: master else working copy `(rgba, side)`.
    pub fn upscale_source(&self, id: MapId) -> Option<(std::sync::Arc<Vec<u8>>, usize)> {
        if let Some(m) = self.master_pixels.get(&id) {
            return Some(m.clone());
        }
        self.export_pixels(id).map(|px| (std::sync::Arc::new(px), maps::TEX_SIZE))
    }

    /// The upscale target side (Refine setting: 4096 / 8192 / 16384). Targets beyond a
    /// single 4× pass are reached by chaining passes; the source side only matters for the
    /// chain length (capped at 3 passes = 64×).
    pub fn upscale_target(&self, src_side: usize) -> usize {
        let target = self.settings.upscale_target.clamp(1024, 16384);
        // With the 3-pass cap, very small sources top out at src × 64.
        target.min(src_side.saturating_mul(64))
    }

    /// `(passes, estimated cost)` to upscale a source of `src_side` to the Refine target —
    /// each chained pass is billed by its own output tier.
    pub fn upscale_cost_estimate(&self, src_side: usize) -> (usize, f64) {
        let target = self.upscale_target(src_side);
        let chain = lumagen_core::generate::upscale_chain(src_side, target);
        let cost = chain.iter().map(|s| lumagen_core::providers::topaz_cost_estimate(*s)).sum();
        (chain.len(), cost)
    }

    /// A map's current native (master) side, falling back to the working size.
    pub fn master_side(&self, id: MapId) -> usize {
        self.master_pixels.get(&id).map(|m| m.1).unwrap_or(maps::TEX_SIZE)
    }

    /// Whether Refine treats `id` as a Topaz upscale (albedo + AI-path maps) rather than a
    /// free re-derive from the albedo master.
    pub fn refine_via_topaz(&self, id: MapId) -> bool {
        id == MapId::Albedo || self.map(id).derive_path == lumagen_core::derive::DerivePath::Ai
    }

    /// The side derived-path maps will re-derive at after this Refine run: the albedo's
    /// final resolution (its upscale target if an albedo upscale is selected/in flight,
    /// else its current master side). Derived maps always follow the albedo — deriving
    /// larger than the albedo source would just interpolate noise.
    pub fn rederive_target(&self) -> usize {
        let side = self.master_side(MapId::Albedo);
        let albedo_growing = self.refine_selection.contains(&MapId::Albedo) || self.upscaling.contains(&MapId::Albedo);
        if albedo_growing { self.upscale_target(side) } else { side }
    }

    /// The maps Refine can improve: every map with real content still below its target —
    /// the Topaz target for the albedo + AI-path maps, the albedo's resolution for
    /// derived-path maps (which re-derive locally, free).
    pub fn upscale_targets(&self) -> Vec<MapId> {
        MapId::ALL
            .iter()
            .copied()
            .filter(|id| {
                if !self.has_map(*id) || self.upscaling.contains(id) {
                    return false;
                }
                let side = self.master_side(*id);
                if self.refine_via_topaz(*id) {
                    side < self.upscale_target(side)
                } else {
                    side < self.rederive_target()
                }
            })
            .collect()
    }

    /// Launch a Refine selection: AI-path maps (and the albedo) go to Topaz; derived-path
    /// maps re-derive from the albedo master at its native side — immediately if the albedo
    /// is settled, or queued until its own upscale lands so they pick up the new master.
    pub fn request_refine(&mut self, ctx: &egui::Context, sel: Vec<MapId>) {
        let (topaz, derived): (Vec<MapId>, Vec<MapId>) = sel.into_iter().partition(|id| self.refine_via_topaz(*id));
        let albedo_growing = topaz.contains(&MapId::Albedo) || self.upscaling.contains(&MapId::Albedo);
        self.request_upscale(ctx, topaz);
        if derived.is_empty() {
            return;
        }
        if albedo_growing {
            for id in derived {
                if self.upscaling.insert(id) {
                    self.pending_rederive.push(id);
                }
            }
        } else {
            self.spawn_rederives(ctx, derived);
        }
    }

    /// Submit worker-side re-derives of `ids` from the current albedo master at its native
    /// resolution. Results come back as `IoResult::Derived` with the master side and install
    /// as new masters (a free, deterministic "upscale" that stays pixel-aligned).
    fn spawn_rederives(&mut self, ctx: &egui::Context, ids: Vec<MapId>) {
        let Some(am) = self.master_pixels.get(&MapId::Albedo).cloned() else {
            for id in &ids {
                self.upscaling.remove(id);
            }
            self.toast("Re-derive skipped — the albedo has no full-resolution master yet", ToastKind::Cost);
            return;
        };
        let (albedo, side) = am;
        if side <= maps::TEX_SIZE {
            for id in &ids {
                self.upscaling.remove(id);
            }
            self.toast("Derived maps already match the albedo resolution", ToastKind::Info);
            return;
        }
        let mut queued = 0;
        for id in ids {
            self.upscaling.insert(id);
            let params = self.map(id).derive_params();
            let epoch = self.material_epoch;
            self.submit_io(
                ctx,
                IoTask::Derive {
                    id,
                    albedo: std::sync::Arc::clone(&albedo),
                    params,
                    size: side,
                    epoch,
                },
            );
            queued += 1;
        }
        if queued > 0 {
            self.toast(
                format!("Re-deriving {queued} map{} from the {side}² albedo…", if queued == 1 { "" } else { "s" }),
                ToastKind::Info,
            );
        }
    }

    /// Queue an upscale for `targets` (deduped against in-flight ones); the app launches the
    /// jobs on the tokio runtime. Results come back through the generation channel as new
    /// masters + refreshed previews (and a version bump).
    pub fn request_upscale(&mut self, ctx: &egui::Context, targets: Vec<MapId>) {
        // Ensure the worker/channel exists before the tasks are spawned.
        let _ = self.worker(ctx);
        let mut queued = 0;
        for id in targets {
            if self.has_map(id) && !self.upscaling.contains(&id) {
                self.upscaling.insert(id);
                self.pending_upscale.push(id);
                queued += 1;
            }
        }
        if queued > 0 {
            let (label, live) = {
                let ab = self.active_backend();
                (self.settings.upscale_model.clone(), ab.live)
            };
            self.toast(
                if live {
                    format!("Upscaling {queued} map{} via Topaz {label}…", if queued == 1 { "" } else { "s" })
                } else {
                    format!(
                        "Upscaling {queued} map{} offline (resize only — no API key)…",
                        if queued == 1 { "" } else { "s" }
                    )
                },
                ToastKind::Info,
            );
        }
    }

    /// Cancel the in-flight albedo generation (the generating hero's Cancel button). The
    /// provider future stops via its token; the albedo returns to its empty state.
    pub fn cancel_albedo_generation(&mut self) {
        if let Some(c) = self.albedo_cancel.take() {
            c.cancel();
        }
        self.pending_albedo_gen = false;
        if self.map(MapId::Albedo).status == MapStatus::Generating {
            self.map_mut(MapId::Albedo).status = MapStatus::Empty;
        }
        self.toast("Albedo generation cancelled", ToastKind::Info);
    }

    /// Re-roll a single map (non-destructive: pushes a version).
    ///
    /// For the albedo this launches a **real** text-to-image generation via the configured
    /// provider (the offline mock only when no key is set) at a fresh seed.
    #[tracing::instrument(skip(self), fields(?id))]
    pub fn reroll(&mut self, id: MapId) {
        tracing::info!(?id, "map re-roll requested");
        if id == MapId::Albedo {
            self.snapshot();
            let mut rng = maps::Rng::new(self.seed.wrapping_add(1));
            self.seed = 4000 + (rng.next_f32() * 4000.0) as u32;
            self.layout = maps::build_layout(self.seed);
            self.generate_albedo_now();
            return;
        }
        // A derived-map re-roll needs a real albedo to derive from.
        if !self.has_map(MapId::Albedo) {
            self.toast("Generate the albedo first — every map derives from it", ToastKind::Cost);
            return;
        }
        // Already in flight for this map — don't double-charge it.
        if matches!(self.map(id).status, MapStatus::Queued | MapStatus::Generating) {
            self.toast(format!("{} is already generating", crate::data::map_def(id).name), ToastKind::Info);
            return;
        }
        self.snapshot();
        // Runs alongside anything already generating (begin_batch extends the running batch).
        self.begin_batch(vec![id]);
    }

    pub fn progress(&self) -> (usize, usize, Option<MapId>) {
        let total = if self.gen_total > 0 { self.gen_total } else { 7 };
        let done = self.gen_done;
        // The map actually in flight right now (falling back to the first still-queued one).
        let current = MapId::DERIVED
            .iter()
            .copied()
            .find(|id| self.map(*id).status == MapStatus::Generating)
            .or_else(|| MapId::DERIVED.iter().copied().find(|id| self.map(*id).status == MapStatus::Queued));
        (done, total, current)
    }

    /// Pan/zoom deliberately survive the switch: maps are pixel-aligned to the albedo, so
    /// staying on the same spot lets the user compare the maps at one location. Fit /
    /// double-click reset the view; a material switch resets it in `begin_material_session`.
    pub fn select(&mut self, id: MapId) {
        self.selected = id;
    }

    /// The current prompt state as a `PromptPack` (base, final, negative, per-map templates).
    pub fn to_prompt_pack(&self) -> lumagen_core::promptpack::PromptPack {
        let mut pack = lumagen_core::promptpack::PromptPack {
            base_template: self.settings.base_template.clone(),
            final_requirement: self.settings.final_requirement.clone(),
            negative_prompt: self.settings.negative_prompt.clone(),
            ..Default::default()
        };
        for (k, v) in &self.settings.map_templates {
            pack.map_templates.insert(format!("{}", k.index()), v.clone());
        }
        pack
    }

    /// Apply a prompt pack onto the current settings (merge over the active pack).
    pub fn apply_prompt_pack(&mut self, pack: lumagen_core::promptpack::PromptPack) {
        if !pack.base_template.is_empty() {
            self.settings.base_template = pack.base_template;
        }
        if !pack.final_requirement.is_empty() {
            self.settings.final_requirement = pack.final_requirement;
        }
        if !pack.negative_prompt.is_empty() {
            self.settings.negative_prompt = pack.negative_prompt;
        }
        for (k, v) in pack.map_templates {
            if let Ok(idx) = k.parse::<usize>()
                && let Some(id) = MapId::ALL.get(idx)
            {
                self.settings.map_templates.insert(*id, v);
            }
        }
    }

    /// Scan the configured library location for real materials (.lumagen documents),
    /// SYNCHRONOUSLY on the calling thread — headless/test path only. The UI goes through
    /// `request_library_scan` so the paint thread never blocks on per-document loads.
    pub fn scan_library(&mut self) {
        let dir = self.library_dir();
        if !dir.exists() {
            return;
        }
        match lumagen_core::library::scan_documents(&dir) {
            Ok(found) => self.install_scanned(found),
            Err(e) => {
                tracing::warn!("library scan failed for {}: {e}", dir.display());
            }
        }
    }

    /// The library root: `LUMAGEN_LIBRARY_DIR` when set (tests point it at a temp dir so UI
    /// runs never touch the developer's real library), else the configured location with
    /// `~` expanded.
    pub fn library_dir(&self) -> std::path::PathBuf {
        match std::env::var("LUMAGEN_LIBRARY_DIR") {
            Ok(dir) if !dir.trim().is_empty() => std::path::PathBuf::from(dir),
            _ => Self::shelve(&self.settings.library_location),
        }
    }

    /// Queue the on-disk library scan on the worker: loading every `.lumagen` document is
    /// heavyweight I/O (zstd + MessagePack per file) that must never run on the paint path.
    /// Results land via `IoResult::LibraryScanned`; `library_scanning` gates a placeholder.
    pub fn request_library_scan(&mut self, ctx: &egui::Context) {
        let dir = self.library_dir();
        if !dir.exists() {
            return;
        }
        self.library_scanning = true;
        self.submit_io(ctx, IoTask::ScanLibrary { dir });
    }

    /// Install scan results: non-empty results replace the in-memory library list; an empty
    /// scan keeps the current entries (fresh imports not yet saved).
    fn install_scanned(&mut self, found: Vec<lumagen_core::library::ScannedMaterial>) {
        if found.is_empty() {
            tracing::debug!("library scan: no documents found");
            return;
        }
        tracing::info!("library scan: {} material(s)", found.len());
        self.library = found
            .into_iter()
            .map(|m| LibMaterial {
                slug: m.slug,
                class: if m.class.is_empty() { "Custom".into() } else { m.class },
                project: m.project,
                res: "4K".into(),
                coverage: m.coverage,
                stale_index: m.stale_index,
                seed: m.seed,
                doc_path: Some(m.path),
            })
            .collect();
        // Projects come from the documents themselves — list every distinct one.
        for m in &self.library {
            if !m.project.is_empty() && !self.projects.iter().any(|x| x.name == m.project) {
                self.projects.push(LibProject { name: m.project.clone() });
            }
        }
    }

    /// Start a fresh material session: invalidate every in-flight async result targeting the
    /// previous material and clear per-material transient state. The generation channel is
    /// swapped (old senders deliver nowhere) and worker results are epoch-stamped, so a slow
    /// provider result from the previous material can never install — or autosave — into the
    /// one being opened. Undo history is per material and is cleared with it.
    fn begin_material_session(&mut self) {
        self.material_epoch = self.material_epoch.wrapping_add(1);
        if let Some(w) = self.render_worker.as_mut() {
            w.reset_generation_channel();
        }
        if let Some(c) = self.gen_cancel.take() {
            c.cancel();
        }
        if let Some(c) = self.albedo_cancel.take() {
            c.cancel();
        }
        if let Some(c) = self.variations_cancel.take() {
            c.cancel();
        }
        for (_, c) in self.gen_map_cancel.drain() {
            c.cancel();
        }
        self.generating = false;
        self.gen_done = 0;
        self.gen_total = 0;
        self.pending_generate = false;
        self.pending_albedo_gen = false;
        self.pending_upscale.clear();
        self.pending_rederive.clear();
        self.upscaling.clear();
        self.evict_variation_previews();
        self.show_variations = None;
        self.enhance_baseline = None;
        self.albedo_error = None;
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.opening = None;
        self.masters_loading.clear();
        self.master_encoded.clear();
        self.zoom = 1.0;
        self.pan = egui::Vec2::ZERO;
    }

    /// A library-unique slug for adopting `desired` as the current material's identity —
    /// imports and creates must never silently take over an existing material's slug (the
    /// next autosave would overwrite that material's document). `allow_current` exempts the
    /// currently open material (re-importing into it is intentional).
    fn unique_slug(&self, desired: &str, allow_current: bool) -> String {
        let taken = |s: &str| {
            if allow_current && s == self.material_name {
                return false;
            }
            self.library.iter().any(|m| m.slug == s)
        };
        let base = desired.trim();
        if !taken(base) {
            return base.to_string();
        }
        let mut n = 2usize;
        loop {
            let candidate = format!("{base}_{n}");
            if !taken(&candidate) {
                return candidate;
            }
            n += 1;
        }
    }

    /// Whether results are pending on the generation channel, which no sender wakes the UI
    /// for — the app keeps scheduling repaints while this holds so completions surface
    /// without user input (worker + async-message channels wake the loop themselves).
    pub fn async_work_pending(&self) -> bool {
        self.albedo_cancel.is_some() || self.variations_cancel.is_some() || !self.var_pending.is_empty()
    }

    /// Block (bounded) until document saves queued on the worker have landed. Called from
    /// app exit — closing the window must not abandon a save that was only queued.
    pub fn drain_pending_saves_on_exit(&mut self, max_wait: std::time::Duration) {
        let deadline = std::time::Instant::now() + max_wait;
        while self.pending_saves > 0 {
            let Some(worker) = self.render_worker.as_mut() else { return };
            for out in worker.drain() {
                if let render::Output::Io(IoResult::DocSaved { path, result, .. }) = out {
                    self.pending_saves = self.pending_saves.saturating_sub(1);
                    match result {
                        Ok(()) => tracing::info!("exit drain: saved {}", path.display()),
                        Err(e) => tracing::error!("exit drain: save failed for {}: {e}", path.display()),
                    }
                }
            }
            if self.pending_saves == 0 {
                return;
            }
            if std::time::Instant::now() >= deadline {
                tracing::error!("exiting with {} document save(s) still pending", self.pending_saves);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
    }

    /// Whether a material switch must first pass the unsaved-changes confirm — same contract
    /// as the window-close path. Only real pixels are worth a prompt: a fresh procedural
    /// workspace has nothing to lose. Consumes the modal's one-shot consent when set.
    fn needs_discard_confirm(&mut self) -> bool {
        self.unsaved && !self.map_pixels.is_empty() && !std::mem::take(&mut self.discard_confirmed)
    }

    /// Adopt a library entry's identity (slug / class / project) as the open material's.
    fn adopt_entry_identity(&mut self, m: &LibMaterial) {
        self.material_name.clone_from(&m.slug);
        self.mat_type.clone_from(&m.class);
        if !m.project.is_empty() {
            self.project_name.clone_from(&m.project);
        }
    }

    /// Open a library material into the workspace (used by the library grid on click). A
    /// real `.lumagen` opens ASYNCHRONOUSLY: the worker parses it and ships working-size
    /// pixels first (`DocOpened` — near-instant via stored previews), then streams each
    /// full-resolution master (`MasterDecoded`). The click never blocks the frame loop;
    /// `opening` drives the library spinner until the workspace appears.
    #[tracing::instrument(skip(self, ctx))]
    pub fn open_material(&mut self, ctx: &egui::Context, slug: &str) {
        if self.needs_discard_confirm() {
            self.pending_open = Some(slug.to_string());
            return;
        }
        // One open at a time — a second click while a document is loading is dropped
        // (the card spinner shows why).
        if self.opening.is_some() {
            return;
        }
        self.begin_material_session();
        let entry = self.library.iter().find(|m| m.slug == slug).cloned();
        // The entry's recorded document path wins — a file whose stem differs from its
        // material name must not open as an empty workspace.
        let path = entry
            .as_ref()
            .and_then(|m| m.doc_path.clone())
            .unwrap_or_else(|| self.library_dir().join(format!("{slug}.lumagen")));
        if path.exists() {
            // Identity is applied when `DocOpened` lands (from the document itself), so a
            // failed load leaves the current workspace untouched.
            self.opening = Some(slug.to_string());
            let epoch = self.material_epoch;
            self.submit_io(
                ctx,
                IoTask::OpenDoc {
                    slug: slug.to_string(),
                    path,
                    epoch,
                },
            );
        } else {
            if let Some(m) = &entry {
                self.adopt_entry_identity(m);
            }
            // No saved document yet for this entry — start a fresh workspace at its seed so
            // the previous material's maps never masquerade as this one.
            let seed = entry.map(|m| m.seed).filter(|s| *s != 0).unwrap_or(self.seed);
            self.reset_maps_and_stores();
            self.seed = seed;
            self.layout = maps::build_layout(seed);
            self.seam_cache_seed = None;
            self.map_mut(MapId::Albedo).versions.push(Version {
                n: 1,
                seed,
                pixels: None,
                master: None,
            });
            self.map_mut(MapId::Albedo).status = MapStatus::Ready;
            self.screen = Screen::Workspace;
            self.unsaved = false;
            self.toast(format!("Opened {}", slug), ToastKind::Info);
        }
    }

    /// Synchronous open — headless/test path (no worker, blocks the calling thread on the
    /// full document decode). The UI goes through `open_material`, which streams via the
    /// worker so a click never freezes the frame loop.
    pub fn open_material_sync(&mut self, slug: &str) {
        if self.needs_discard_confirm() {
            self.pending_open = Some(slug.to_string());
            return;
        }
        self.begin_material_session();
        let entry = self.library.iter().find(|m| m.slug == slug).cloned();
        if let Some(m) = &entry {
            self.adopt_entry_identity(m);
        }
        let path = entry
            .and_then(|m| m.doc_path)
            .unwrap_or_else(|| self.library_dir().join(format!("{slug}.lumagen")));
        if !path.exists() {
            return;
        }
        match lumagen_core::document::Document::load(&path) {
            Ok(doc) => {
                self.apply_document(&doc);
                self.screen = Screen::Workspace;
                self.unsaved = false;
            }
            Err(e) => {
                let chain = render::error_chain(&e);
                self.error_toast(&format!("could not load {}: {chain}", path.display()), format!("Couldn't open {slug}: {chain}"));
            }
        }
    }

    /// Create a new project (a named group in the library sidebar) and make it the current
    /// project. No-op with a toast if the name is empty or already exists.
    pub fn create_project(&mut self, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            self.toast("Give the project a name", ToastKind::Info);
            return;
        }
        if self.projects.iter().any(|p| p.name.eq_ignore_ascii_case(name)) {
            self.toast(format!("{} already exists", name), ToastKind::Info);
            return;
        }
        self.projects.push(LibProject { name: name.to_string() });
        self.project_name = name.to_string();
        self.lib_nav = name.to_string();
        self.unsaved = true;
        self.toast(format!("Project {} created — new materials land in it", name), ToastKind::Success);
    }

    /// Start a brand-new material from the create wizard: reset every map to empty, set the
    /// identity/seed, register it in the library, and mark the albedo **Generating** — the
    /// real result arrives from the provider. No placeholder pixels and no fake history
    /// entry: the workspace shows a generating state until the actual albedo lands.
    pub fn new_material_from_create(&mut self, name: String, seed: u32) {
        if self.needs_discard_confirm() {
            self.pending_new = Some((name, seed));
            return;
        }
        self.begin_material_session();
        // A create must never adopt an existing material's slug — the first autosave would
        // overwrite that material's document.
        let name = self.unique_slug(&name, false);
        self.reset_maps_and_stores();
        self.apply_path_policy(0.0);
        self.material_name.clone_from(&name);
        self.seed = seed;
        self.layout = maps::build_layout(seed);
        self.seam_cache_seed = None;
        self.albedo_error = None;
        self.map_mut(MapId::Albedo).status = MapStatus::Generating;
        // Step ① (Albedo) is in progress, not done.
        self.active_step = 0;
        // Register in the library (newest first) if not already present.
        if !self.library.iter().any(|m| m.slug == name) {
            self.library.insert(
                0,
                LibMaterial {
                    slug: name,
                    class: self.mat_type.clone(),
                    project: self.project_name.clone(),
                    res: "4K".into(),
                    coverage: 0,
                    stale_index: None,
                    seed,
                    doc_path: None,
                },
            );
        }
        self.screen = Screen::Workspace;
        self.unsaved = true;
    }

    /// Install an imported albedo image (decoded to RGBA8 at the working size) as the source
    /// albedo, replacing the procedural one. Pushes a version and switches to the workspace.
    pub fn import_albedo(&mut self, ctx: &egui::Context, rgba: Vec<u8>, name: &str) {
        self.snapshot();
        let seed = self.seed;
        let px = std::sync::Arc::new(rgba);
        self.install_pixels(ctx, MapId::Albedo, std::sync::Arc::clone(&px));
        self.push_version(MapId::Albedo, seed, Some(px), None);
        self.map_mut(MapId::Albedo).status = MapStatus::Ready;
        // The imported albedo has different edges than the procedural one — recompute the seam.
        self.seam_cache_seed = None;
        if !name.trim().is_empty() {
            // Never adopt another material's slug — autosave would overwrite its document.
            let slug = self.unique_slug(name, true);
            if slug != name.trim() {
                self.toast(format!("Named {slug} — \"{}\" already exists in the library", name.trim()), ToastKind::Info);
            }
            self.material_name = slug;
        }
        self.screen = Screen::Workspace;
        self.unsaved = true;
        self.toast("Albedo imported", ToastKind::Success);
        self.autosave(ctx);
    }

    /// Restore everything from a document EXCEPT pixels: identity, prompts, per-map params,
    /// statuses, and version-history metadata. Shared by the synchronous apply (tests) and
    /// the worker-streamed apply — pixels install differently in each.
    fn apply_document_shell(&mut self, doc: &lumagen_core::document::Document) {
        self.material_name.clone_from(&doc.material_name);
        self.project_name.clone_from(&doc.project_name);
        if !doc.mat_type.is_empty() {
            self.mat_type.clone_from(&doc.mat_type);
        }
        self.description.clone_from(&doc.description);
        self.seed = doc.seed;
        self.gen_size = doc.gen_size;
        self.engine.clone_from(&doc.engine);
        self.export_pattern.clone_from(&doc.export_pattern);
        self.settings.base_template.clone_from(&doc.base_template);
        self.settings.final_requirement.clone_from(&doc.final_requirement);
        self.settings.negative_prompt.clone_from(&doc.negative_prompt);
        for (k, v) in &doc.map_templates {
            if let Ok(idx) = k.parse::<usize>()
                && let Some(id) = MapId::ALL.get(idx)
            {
                self.settings.map_templates.insert(*id, v.clone());
            }
        }
        self.layout = maps::build_layout(self.seed);
        self.reset_maps_and_stores();
        // Restore the reconstructed displacement height (16-bit section) so a reopened material
        // still exports a TRUE 16-bit/EXR displacement instead of falling back to 8-bit/re-derive.
        if let Some(hf) = doc.displacement_height() {
            self.master_height.insert(MapId::Displacement, hf);
        }
        let fallback_seed = self.seed;
        for id in MapId::ALL {
            // Maps the document never touched keep the default-policy path.
            let policy = self.policy_path(id);
            let Some(mapdoc) = doc.maps.get(&format!("{}", id.index())) else {
                self.map_mut(id).derive_path = policy;
                continue;
            };
            {
                let m = self.map_mut(id);
                m.strength = mapdoc.strength;
                m.invert = mapdoc.invert;
                m.flip_y = mapdoc.flip_y;
                m.radius = mapdoc.radius;
                m.bit_depth_16 = mapdoc.bit_depth_16;
                m.fidelity = mapdoc.fidelity;
                m.detail_routing = mapdoc.detail_routing;
                m.active_version = mapdoc.active_version;
                // The document's per-map path wins (the Path toggle persists per material);
                // a never-generated map falls back to the default policy.
                m.derive_path = if mapdoc.ai_path {
                    lumagen_core::derive::DerivePath::Ai
                } else if mapdoc.has_image {
                    lumagen_core::derive::DerivePath::Derived
                } else {
                    policy
                };
                // Rebuild the version-history metadata so the chips survive the reload.
                m.versions = if mapdoc.version_seeds.is_empty() {
                    vec![Version {
                        n: 1,
                        seed: fallback_seed,
                        pixels: None,
                        master: None,
                    }]
                } else {
                    mapdoc
                        .version_seeds
                        .iter()
                        .enumerate()
                        .map(|(i, s)| Version {
                            n: i as u32 + 1,
                            seed: *s,
                            pixels: None,
                            master: None,
                        })
                        .collect()
                };
                m.active_version = mapdoc.active_version.min(m.versions.len().saturating_sub(1));
                if mapdoc.has_image {
                    m.status = if mapdoc.locked { MapStatus::Locked } else { MapStatus::Ready };
                }
            }
        }
    }

    /// Restore workspace state from a loaded document, decoding pixels inline — the
    /// SYNCHRONOUS path (tests/headless). Documents store the highest-resolution source
    /// (`px_size`); a full-res master splits into master + working-size downscale;
    /// unexpected sizes are skipped (re-render/derive) instead of crashing the upload.
    fn apply_document(&mut self, doc: &lumagen_core::document::Document) {
        self.apply_document_shell(doc);
        for id in MapId::ALL {
            let Some(mapdoc) = doc.maps.get(&format!("{}", id.index())) else { continue };
            if !mapdoc.png.is_empty()
                && let Some((rgba, _)) = doc.get_map(id)
            {
                let side = doc.map_px_size(id);
                if rgba.len() == side * side * 4 {
                    let mut master_arc = None;
                    let working = if side == maps::TEX_SIZE {
                        rgba
                    } else {
                        let work = maps::resize_square_rgba(&rgba, side, maps::TEX_SIZE);
                        let master = (std::sync::Arc::new(rgba), side);
                        self.master_pixels.insert(id, master.clone());
                        master_arc = Some(master);
                        work
                    };
                    let px = std::sync::Arc::new(working);
                    self.map_pixels.insert(id, std::sync::Arc::clone(&px));
                    let av = self.map(id).active_version;
                    if let Some(v) = self.map_mut(id).versions.get_mut(av) {
                        v.pixels = Some(px);
                        v.master = master_arc;
                    }
                } else {
                    tracing::warn!(?id, len = rgba.len(), side, "skipping stored pixels with unexpected size");
                }
            }
        }
        self.seam_cache_seed = None;
    }

    /// Install a document opened on the worker: params + working-size pixels immediately
    /// (the workspace paints without waiting), full-resolution masters as their
    /// `MasterDecoded` messages stream in — `masters_loading` gates Export/Refine and
    /// defers saves until the set is complete.
    fn apply_opened_document(&mut self, opened: render::OpenedDoc) {
        let render::OpenedDoc {
            doc,
            working,
            encoded_masters,
            pending_masters,
        } = opened;
        self.apply_document_shell(&doc);
        for (id, work, _native) in working {
            let px = std::sync::Arc::new(work);
            self.map_pixels.insert(id, std::sync::Arc::clone(&px));
            let av = self.map(id).active_version;
            if let Some(v) = self.map_mut(id).versions.get_mut(av) {
                // The master attaches to this version when its decode lands.
                v.pixels = Some(px);
            }
        }
        // Masters stay encoded until the native view / Export / Refine asks for pixels —
        // and pass through byte-identical on save while untouched.
        self.master_encoded = encoded_masters.into_iter().collect();
        // Only legacy masters (decoded during the open to build working copies) stream in.
        self.masters_loading = pending_masters.into_iter().collect();
        self.seam_cache_seed = None;
    }

    /// Request decodes for every still-encoded master that has no pixels yet — Export and
    /// Refine call this when their modals open; the existing `masters_loading` gate holds
    /// their launch buttons until the batch lands. Decodes run in parallel on the worker.
    pub fn ensure_masters_decoded(&mut self, ctx: &egui::Context) {
        let jobs: Vec<(MapId, std::sync::Arc<Vec<u8>>, usize)> = self
            .master_encoded
            .iter()
            .filter(|(id, _)| !self.master_pixels.contains_key(id) && !self.masters_loading.contains(id))
            .map(|(id, (png, _, side))| (*id, std::sync::Arc::clone(png), *side))
            .collect();
        if jobs.is_empty() {
            return;
        }
        for (id, ..) in &jobs {
            self.masters_loading.insert(*id);
        }
        let epoch = self.material_epoch;
        self.submit_io(ctx, IoTask::DecodeMasters { jobs, epoch });
    }

    /// Request one master's decode (the 2D native view path) — a no-op when the pixels
    /// are already present, already decoding, or the map has no encoded master.
    fn request_master_decode(&mut self, ctx: &egui::Context, id: MapId) {
        if self.master_pixels.contains_key(&id) || self.masters_loading.contains(&id) {
            return;
        }
        let Some((png, _, side)) = self.master_encoded.get(&id) else { return };
        let job = (id, std::sync::Arc::clone(png), *side);
        self.masters_loading.insert(id);
        let epoch = self.material_epoch;
        self.submit_io(ctx, IoTask::DecodeMasters { jobs: vec![job], epoch });
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn unique_slug_never_adopts_another_materials_identity() {
        let mut s = AppState::new();
        s.library.clear();
        s.library.push(LibMaterial {
            slug: "rusty_hull".into(),
            class: "Metal".into(),
            project: String::new(),
            res: "4K".into(),
            coverage: 8,
            stale_index: None,
            seed: 1,
            doc_path: None,
        });
        // A colliding import/create gets a numbered suffix instead of overwriting.
        assert_eq!(s.unique_slug("rusty_hull", false), "rusty_hull_2");
        assert_eq!(s.unique_slug("fresh_paint", false), "fresh_paint");
        // Re-importing into the currently open material keeps its own slug.
        s.material_name = "rusty_hull".into();
        assert_eq!(s.unique_slug("rusty_hull", true), "rusty_hull");
        // ... but a CREATE named like the open material still uniquifies.
        assert_eq!(s.unique_slug("rusty_hull", false), "rusty_hull_2");
    }

    #[test]
    fn apply_prefs_clamps_persisted_indices() {
        let mut s = AppState::new();
        let prefs = UiPrefs {
            theme: 99,
            nav_preset: 42,
            ..Default::default()
        };
        s.apply_prefs(prefs);
        assert!(s.settings.theme <= 1, "theme index must clamp to the 2-entry array");
        assert!(s.settings.nav_preset <= 1, "nav_preset index must clamp to the 2-entry array");
    }

    #[test]
    fn switching_materials_with_unsaved_pixels_defers_to_the_confirm() {
        let mut s = AppState::new();
        s.library.push(LibMaterial {
            slug: "other_mat".into(),
            class: "Metal".into(),
            project: String::new(),
            res: "4K".into(),
            coverage: 0,
            stale_index: None,
            seed: 2,
            doc_path: None,
        });
        s.material_name = "current".into();
        s.unsaved = true;
        s.map_pixels.insert(MapId::Albedo, std::sync::Arc::new(vec![0; 4]));
        let ctx = egui::Context::default();
        s.open_material(&ctx, "other_mat");
        // Deferred, not switched: the confirm modal resolves pending_open.
        assert_eq!(s.pending_open.as_deref(), Some("other_mat"));
        assert_eq!(s.material_name, "current");
        // The modal's consent flag lets the same call through.
        s.discard_confirmed = true;
        s.open_material(&ctx, "other_mat");
        assert_eq!(s.material_name, "other_mat");
        assert!(s.pending_open.is_some(), "the deferred request is untouched until the modal resolves it");
    }

    #[test]
    fn material_switch_bumps_epoch_and_clears_session_state() {
        let mut s = AppState::new();
        let before = s.material_epoch;
        s.undo_stack.push(s.to_document());
        s.gen_cancel = Some(tokio_util::sync::CancellationToken::new());
        s.albedo_cancel = Some(tokio_util::sync::CancellationToken::new());
        s.upscaling.insert(MapId::Albedo);
        s.new_material_from_create("epoch_probe".into(), 9);
        assert_eq!(s.material_epoch, before + 1, "each switch invalidates in-flight results");
        assert!(s.undo_stack.is_empty(), "undo history is per material");
        assert!(s.gen_cancel.is_none() && s.albedo_cancel.is_none() && s.upscaling.is_empty());
    }

    #[test]
    fn job_timeout_parses_and_clamps() {
        let mut s = AppState::new();
        assert_eq!(s.settings.job_timeout_duration().as_secs(), 600, "default is 10 minutes");
        s.settings.job_timeout = "1200".into();
        assert_eq!(s.settings.job_timeout_duration().as_secs(), 1200);
        s.settings.job_timeout = "5".into();
        assert_eq!(s.settings.job_timeout_duration().as_secs(), 60, "clamped to the floor");
        s.settings.job_timeout = "999999".into();
        assert_eq!(s.settings.job_timeout_duration().as_secs(), 3600, "clamped to the ceiling");
        s.settings.job_timeout = "not a number".into();
        assert_eq!(s.settings.job_timeout_duration().as_secs(), 600, "junk falls back to the default");
    }

    #[test]
    fn cost_per_image_parses_or_falls_back() {
        let mut s = AppState::new();
        s.settings.cost_per_image = "0.12".into();
        assert!((s.cost_per_image() - 0.12).abs() < 1e-9);
        s.settings.cost_per_image = "not a number".into();
        assert!((s.cost_per_image() - crate::data::PRICE_PER_IMAGE).abs() < 1e-9);
    }

    #[test]
    fn generate_estimate_counts_only_missing_ai_maps() {
        let mut s = AppState::new();
        s.settings.cost_per_image = "0.10".into();
        // Fresh material: roughness / metallic / emission are AI + missing => 3 paid requests.
        let (reqs, cost) = s.generate_estimate();
        assert_eq!(reqs, 3);
        assert!((cost - 0.30).abs() < 1e-9);
        // Marking one AI map done drops the count; the derived maps never counted.
        s.map_pixels.insert(MapId::Roughness, std::sync::Arc::new(vec![0u8; 4]));
        s.map_mut(MapId::Roughness).status = MapStatus::Ready;
        assert_eq!(s.generate_estimate().0, 2);
    }

    #[test]
    fn export_pixels_sources_real_or_derived_never_albedo_standin() {
        let s = AppState::new();
        // The albedo always has pixels (procedural or imported).
        assert!(s.export_pixels(MapId::Albedo).is_some());
        // A geometric map derives from the albedo even without stored pixels.
        assert!(s.export_pixels(MapId::Normal).is_some());
        // A semantic map with no generated pixels yields None — never the albedo as a stand-in.
        assert!(s.export_pixels(MapId::Roughness).is_none());
    }

    #[test]
    fn document_round_trip_preserves_identity_and_real_pixels() {
        let mut s = AppState::new();
        s.material_name = "test_hull".into();
        s.description = "painted hull with bare steel bolts".into();
        s.seed = 4242;
        // Install a real (non-albedo) roughness map with distinctive pixels.
        let px = std::sync::Arc::new(vec![64u8; maps::TEX_SIZE * maps::TEX_SIZE * 4]);
        s.map_pixels.insert(MapId::Roughness, px);
        s.map_mut(MapId::Roughness).status = MapStatus::Ready;
        s.map_mut(MapId::Roughness).versions.push(Version {
            n: 1,
            seed: 4242,
            pixels: None,
            master: None,
        });

        let doc = s.to_document();
        let mut s2 = AppState::new();
        s2.apply_document(&doc);

        assert_eq!(s2.material_name, "test_hull");
        assert_eq!(s2.description, "painted hull with bare steel bolts");
        assert_eq!(s2.seed, 4242);
        assert!(s2.map(MapId::Roughness).status.has_image());
        // The real roughness pixels round-tripped — not the albedo as a stand-in.
        let restored = s2.map_pixels.get(&MapId::Roughness).expect("roughness pixels restored");
        assert!(restored.iter().all(|&b| b == 64), "roughness pixels were not preserved");
    }

    #[test]
    fn imported_albedo_persists_and_restores() {
        // A distinctive imported albedo must survive save/load (regression for the bug where
        // Ctrl+S wrote the procedural render and discarded the user's source).
        let mut s = AppState::new();
        let albedo = std::sync::Arc::new(vec![201u8; maps::TEX_SIZE * maps::TEX_SIZE * 4]);
        s.map_pixels.insert(MapId::Albedo, albedo);
        let doc = s.to_document();
        let mut s2 = AppState::new();
        s2.apply_document(&doc);
        let restored = s2.map_pixels.get(&MapId::Albedo).expect("albedo restored");
        assert!(restored.iter().all(|&b| b == 201), "imported albedo was not preserved across save/load");
    }
}
