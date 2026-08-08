//! Background map renderer. CPU rasterization + thumbnailing are too expensive for the UI
//! frame thread, so this module owns a single dedicated worker thread: the UI enqueues
//! [`RenderRequest`]s and drains finished [`RenderResult`]s each frame, calling
//! `request_repaint` so a finished job surfaces immediately. Only the cheap
//! `Context::load_texture` upload stays on the UI thread.

use std::sync::mpsc::{Receiver, Sender};

use eframe::egui;

use crate::data::MapId;
use crate::maps::{self, TEX_SIZE, THUMB_SIZE};

/// A unit of off-thread work. `target` identifies where the result lands. `epoch` is the
/// material session the request belongs to — a result from a previous session (the user
/// switched materials while it was in flight) is dropped instead of installed.
#[derive(Clone, Debug)]
pub struct RenderRequest {
    pub target: RenderTarget,
    pub seed: u32,
    pub epoch: u64,
}

/// A blocking I/O task (e.g. a keyring vault read/write) to run off the UI thread. These
/// share the render worker's thread so the app has exactly one background worker.
pub enum IoTask {
    /// Load a provider's API key from the OS vault.
    LoadKey(crate::secrets::Provider),
    /// Store a provider's API key into the OS vault.
    StoreKey(crate::secrets::Provider, String),
    /// Remove a provider's API key from the OS vault (the user cleared the field).
    DeleteKey(crate::secrets::Provider),
    /// Export the map set to a folder (writes many MB — off the UI thread). `derive_missing`
    /// lists geometric maps to derive on the worker at the albedo source's native resolution
    /// (deriving at 4K on the UI thread would freeze it); `albedo_src` is that source.
    Export {
        /// The prepared job (maps already sourced from full-res masters where available).
        job: Box<lumagen_core::export::ExportJob>,
        dir: std::path::PathBuf,
        /// Geometric maps to derive on the worker from `albedo_src`.
        derive_missing: Vec<(MapId, lumagen_core::derive::DeriveParams)>,
        albedo_src: Option<(std::sync::Arc<Vec<u8>>, usize)>,
    },
    /// Build + write the `.lumagen` document off the UI thread (PNG-encoding full-resolution
    /// masters is far too slow for the frame loop).
    SaveDoc(Box<SaveJob>),
    /// Decode an imported albedo image off the UI thread (PNG/JPEG → RGBA8 at the working size).
    ImportAlbedo { path: std::path::PathBuf, epoch: u64 },
    /// Re-derive one geometric map from the albedo with the given adjust params, off the UI
    /// thread (drives the live "adjust" sliders). The result installs without pushing a version.
    Derive {
        id: MapId,
        /// The source albedo (shared, so the buffer isn't copied per request).
        albedo: std::sync::Arc<Vec<u8>>,
        params: lumagen_core::derive::DeriveParams,
        size: usize,
        epoch: u64,
    },
    /// Scan a folder of loose map images (albedo + suffixed maps) into a material, off the UI
    /// thread (the "Import from folder" action).
    ImportFolder { dir: std::path::PathBuf, epoch: u64 },
    /// Scan the on-disk library for `.lumagen` documents, off the UI thread (loading every
    /// document blocks for the whole library and must never run on the paint path).
    ScanLibrary { dir: std::path::PathBuf },
    /// Open a `.lumagen` document off the UI thread: the manifest parse + working-size
    /// pixels come back first (`DocOpened`, fast via stored previews); full-resolution
    /// masters stay ENCODED until something needs them — the click stays instant.
    OpenDoc { slug: String, path: std::path::PathBuf, epoch: u64 },
    /// Decode a batch of encoded masters on demand (native 2D view, Export, Refine) —
    /// in parallel across cores under a memory budget, each result streaming back as its
    /// own `MasterDecoded` the moment it finishes.
    DecodeMasters {
        /// `(map, stored PNG section, native side)` per master to decode.
        jobs: Vec<(MapId, std::sync::Arc<Vec<u8>>, usize)>,
        epoch: u64,
    },
    /// Render a library-grid preview from a saved document's REAL albedo (falling back to the
    /// procedural render at `seed` only when the document holds no albedo pixels).
    DocPreview {
        slug: String,
        path: std::path::PathBuf,
        /// Fallback procedural seed.
        seed: u32,
    },
    /// Decode a user's own image file to override one map (the "Replace file" action).
    OverrideMap { id: MapId, path: std::path::PathBuf, epoch: u64 },
}

/// Everything the worker needs to build and write a `.lumagen` document: the metadata-only
/// document plus each map's bytes (fresh pixels to encode, or an untouched master's
/// already-encoded section to pass through) and params.
pub struct SaveJob {
    pub path: std::path::PathBuf,
    /// The document with identity/prompt fields set and NO maps yet.
    pub doc: lumagen_core::document::Document,
    pub maps: Vec<SaveMap>,
    /// Whether to toast on success (explicit Ctrl+S yes; silent autosave no).
    pub announce: bool,
}

/// One map's contribution to a `SaveJob`.
pub struct SaveMap {
    pub id: MapId,
    pub source: SaveMapSource,
    /// The stored pixel side (native master side, or the working size).
    pub side: usize,
    pub params: lumagen_core::document::MapDocParams,
}

/// Where a saved map's bytes come from.
pub enum SaveMapSource {
    /// Raw RGBA pixels to PNG-encode (fresh generations and working copies).
    Pixels(std::sync::Arc<Vec<u8>>),
    /// An untouched master opened from disk: its PNG section writes back byte-identical —
    /// no decode, no re-encode, which is what makes autosaves of opened materials cheap.
    Encoded {
        /// The stored full-resolution PNG section.
        png: std::sync::Arc<Vec<u8>>,
        /// The stored working-size preview (empty for pre-preview documents — the worker
        /// then encodes one from `working` so old files self-heal to fast opens).
        preview: std::sync::Arc<Vec<u8>>,
        /// The working-size pixels, for preview generation when `preview` is empty.
        working: std::sync::Arc<Vec<u8>>,
    },
}

/// A material scanned from a folder of loose map images.
pub struct ImportedFolder {
    /// Material slug (the folder name).
    pub slug: String,
    /// The decoded maps found (RGBA8 at the working size), keyed by map.
    pub maps: Vec<(MapId, Vec<u8>)>,
}

/// Where a finished render should be installed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderTarget {
    /// A material map (full texture + thumbnail) for `MapId`.
    Map(MapId),
    /// A library-grid preview for this material slug. Slug-keyed (not index-keyed) so a
    /// library insert/remove while the render is in flight can't land the preview under
    /// a different material.
    LibPreview(String),
}

/// A finished render: full-size and thumbnail RGBA8 pixels, produced off-thread.
pub struct RenderResult {
    pub target: RenderTarget,
    pub seed: u32,
    /// Material session the render was requested in (stale map renders are dropped).
    pub epoch: u64,
    pub full: Vec<u8>,
    pub thumb: Vec<u8>,
}

/// The result of a finished blocking I/O task.
pub enum IoResult {
    /// A loaded key: `None` means "not set" or "vault unavailable".
    KeyLoaded(crate::secrets::Provider, Option<String>),
    /// A store attempt: `true` if it landed in the vault.
    KeyStored(crate::secrets::Provider, bool),
    /// A delete attempt: `true` if the entry is gone from the vault.
    KeyDeleted(crate::secrets::Provider, bool),
    /// An export finished: the files written, or the error.
    Exported(Result<usize, String>),
    /// An imported albedo decoded: RGBA8 pixels + a display name, or the error.
    AlbedoImported { result: Result<(Vec<u8>, String), String>, epoch: u64 },
    /// A re-derived map. At the working size it's a live adjust refresh (no version bump);
    /// larger sizes are Refine re-derives that install as new full-resolution masters.
    Derived {
        id: MapId,
        /// RGBA8 pixels at `size × size`.
        full: Vec<u8>,
        thumb: Vec<u8>,
        size: usize,
        epoch: u64,
    },
    /// A folder scanned into a material, or the error.
    FolderImported { result: Result<ImportedFolder, String>, epoch: u64 },
    /// The on-disk library scan finished (worker-side full-document loads).
    LibraryScanned(Result<Vec<lumagen_core::library::ScannedMaterial>, String>),
    /// A library preview rendered from a saved document's albedo (RGBA8 at working size).
    DocPreview { slug: String, full: Vec<u8> },
    /// A document save finished (worker-side build + write).
    DocSaved {
        path: std::path::PathBuf,
        result: Result<(), String>,
        /// Whether success deserves a toast.
        announce: bool,
    },
    /// A user's own image decoded to override one map (RGBA8), or the error.
    MapOverridden { id: MapId, result: Result<Vec<u8>, String>, epoch: u64 },
    /// A document opened on the worker: identity/params plus working-size pixels, ready to
    /// install. Full-resolution masters follow as `MasterDecoded` messages.
    DocOpened {
        slug: String,
        epoch: u64,
        /// The opened payload, or the load error (source chain included).
        result: Result<Box<OpenedDoc>, String>,
    },
    /// One full-resolution master finished decoding for the document just opened.
    MasterDecoded {
        id: MapId,
        epoch: u64,
        /// `(rgba, side)` at native resolution, or the decode error.
        result: Result<(Vec<u8>, usize), String>,
    },
}

/// A document opened on the worker. `doc` keeps identity, prompts, and per-map params; the
/// PNG payloads it carried have been taken out (working pixels decoded, master sections
/// handed over encoded) so no redundant multi-hundred-MB buffers ride along.
pub struct OpenedDoc {
    /// The parsed document, pixel payloads stripped.
    pub doc: lumagen_core::document::Document,
    /// Per-map working-size RGBA8 (`tex_size`²) + the native side of its master
    /// (equal to `tex_size` when the map has no master).
    pub working: Vec<(MapId, Vec<u8>, usize)>,
    /// Each master's still-encoded payloads, keyed by map. Kept encoded until the 2D
    /// native view / Export / Refine needs pixels — and written back byte-identical on
    /// save, so untouched masters cost no decode and no re-encode.
    pub encoded_masters: Vec<(MapId, EncodedMaster)>,
    /// Masters already decoded during the open (legacy documents without previews pay the
    /// full decode to build working copies) — these stream in right behind `DocOpened`.
    pub pending_masters: Vec<MapId>,
}

/// Work the single background worker pulls off its queue.
enum Task {
    Render(RenderRequest),
    Io(IoTask),
}

/// A lightweight message from a tokio task (network work) to the UI: e.g. a "test connection"
/// result. Delivered through the same single drain point as renders/I-O/generation.
pub enum AsyncMsg {
    /// Raise a toast on the UI thread.
    Toast { msg: String, kind: crate::widgets::ToastKind },
    /// Replace the material description (the LLM Description Check result).
    SetDescription(String),
    /// A finished provider connection test — drives the inline status in Settings.
    TestResult {
        provider: crate::secrets::Provider,
        /// Human-readable success line, or the error text.
        result: Result<String, String>,
    },
}

/// A finished unit of work, delivered back to the UI thread.
pub enum Output {
    Rendered(RenderResult),
    Io(IoResult),
    /// A map finished generating (semantic via provider or geometric via derive), pushed from
    /// the tokio generation task and drained alongside renders/I/O at the single drain point.
    Generated(lumagen_core::generate::MapOutcome),
    /// A message from a tokio network task (e.g. a "test connection" outcome).
    Async(AsyncMsg),
}

/// Handle the UI thread holds: send work in, drain results out.
pub struct RenderWorker {
    tx: Sender<Task>,
    rx: Receiver<Output>,
    /// The generation (tokio) task pushes finished maps here; drained with the rest.
    gen_rx: tokio::sync::mpsc::UnboundedReceiver<lumagen_core::generate::MapOutcome>,
    /// Kept so the app can hand clones to generation tasks.
    gen_tx: tokio::sync::mpsc::UnboundedSender<lumagen_core::generate::MapOutcome>,
    /// Tokio network tasks (e.g. test-connection) push UI messages here; drained with the rest.
    msg_rx: tokio::sync::mpsc::UnboundedReceiver<AsyncMsg>,
    /// Kept so the app can hand `AsyncHandle`s to tokio tasks.
    msg_tx: tokio::sync::mpsc::UnboundedSender<AsyncMsg>,
}

impl RenderWorker {
    /// Spawn the single worker thread. `ctx` is a cheap clone the worker uses to wake the
    /// UI loop when a result lands (the standard eframe async pattern).
    pub fn spawn(ctx: egui::Context) -> Self {
        let (req_tx, req_rx) = std::sync::mpsc::channel::<Task>();
        let (res_tx, res_rx) = std::sync::mpsc::channel::<Output>();
        let (gen_tx, gen_rx) = tokio::sync::mpsc::unbounded_channel();
        let (msg_tx, msg_rx) = tokio::sync::mpsc::unbounded_channel();
        match std::thread::Builder::new().name("lumagen-worker".into()).spawn({
            let worker_ctx = ctx.clone();
            move || run_worker(req_rx, res_tx, worker_ctx)
        }) {
            Ok(_handle) => {}
            Err(e) => {
                // A failure to spawn the worker thread is unrecoverable for this app; log
                // it rather than panic the UI thread. Requests will simply never complete,
                // which the UI surfaces as maps stuck in "generating".
                tracing::error!("failed to spawn background worker thread: {e}");
            }
        }
        Self {
            tx: req_tx,
            rx: res_rx,
            gen_rx,
            gen_tx,
            msg_rx,
            msg_tx,
        }
    }

    /// Enqueue a render. Non-blocking; the worker picks it up off-thread.
    pub fn submit(&self, req: RenderRequest) {
        let target = format!("{:?}", req.target);
        if self.tx.send(Task::Render(req)).is_err() {
            tracing::warn!("worker channel closed; dropping render request for {target}");
        }
    }

    /// Enqueue a blocking I/O task (vault access). Non-blocking at the call site.
    pub fn submit_io(&self, task: IoTask) {
        if self.tx.send(Task::Io(task)).is_err() {
            tracing::warn!("worker channel closed; dropping I/O task");
        }
    }

    /// The sender a generation task uses to push finished maps back (and wake the loop).
    pub fn generation_sender(&self) -> GenerationHandle {
        GenerationHandle { tx: self.gen_tx.clone() }
    }

    /// A handle a tokio network task uses to push a UI message (toast) back and wake the loop.
    pub fn async_handle(&self, ctx: egui::Context) -> AsyncHandle {
        AsyncHandle { tx: self.msg_tx.clone(), ctx }
    }

    /// Swap the generation channel for a fresh pair. Tasks launched before the swap hold
    /// the old sender, so their late results deliver nowhere — the structural guarantee
    /// that a material switch can never install a previous material's generated maps.
    pub fn reset_generation_channel(&mut self) {
        let (gen_tx, gen_rx) = tokio::sync::mpsc::unbounded_channel();
        self.gen_tx = gen_tx;
        self.gen_rx = gen_rx;
    }

    /// Drain every finished output (renders + I/O + generated maps + async messages) without
    /// blocking. Called once per frame from `logic` — the single drain point.
    pub fn drain(&mut self) -> Vec<Output> {
        let mut out: Vec<Output> = self.rx.try_iter().collect();
        while let Ok(outcome) = self.gen_rx.try_recv() {
            out.push(Output::Generated(outcome));
        }
        while let Ok(msg) = self.msg_rx.try_recv() {
            out.push(Output::Async(msg));
        }
        out
    }
}

/// A handle a tokio network task holds to report a result back to the UI as a toast. Carries a
/// `Context` clone so a delivered message wakes the frame loop regardless of the active screen.
#[derive(Clone)]
pub struct AsyncHandle {
    tx: tokio::sync::mpsc::UnboundedSender<AsyncMsg>,
    ctx: egui::Context,
}

impl AsyncHandle {
    /// Push a toast to the UI thread and request a repaint so it surfaces immediately.
    pub fn toast(&self, msg: impl Into<String>, kind: crate::widgets::ToastKind) {
        let _ = self.tx.send(AsyncMsg::Toast { msg: msg.into(), kind });
        self.ctx.request_repaint();
    }

    /// Replace the material description on the UI thread (LLM enhance result).
    pub fn set_description(&self, description: String) {
        let _ = self.tx.send(AsyncMsg::SetDescription(description));
        self.ctx.request_repaint();
    }

    /// Report a provider connection-test outcome (inline status in Settings).
    pub fn test_result(&self, provider: crate::secrets::Provider, result: Result<String, String>) {
        let _ = self.tx.send(AsyncMsg::TestResult { provider, result });
        self.ctx.request_repaint();
    }
}

/// A handle the app hands to a generation task so it can push finished maps back to the UI.
/// An active generation batch already repaints every frame (the pulsing status dots), so
/// results are drained continuously without an extra wake.
#[derive(Clone)]
pub struct GenerationHandle {
    tx: tokio::sync::mpsc::UnboundedSender<lumagen_core::generate::MapOutcome>,
}

impl GenerationHandle {
    /// The raw channel sender for the orchestrator.
    pub fn into_sender(self) -> tokio::sync::mpsc::UnboundedSender<lumagen_core::generate::MapOutcome> {
        self.tx
    }
}

/// Render an error with its full source chain (`context: cause: root`). The core error
/// types don't restate their source in `Display` (that would double-print in anyhow
/// chains), so any boundary that flattens an error into a toast/log String must walk
/// `source()` — or the user sees "I/O error" with the actual cause stripped.
pub fn error_chain(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut src = e.source();
    while let Some(s) = src {
        out.push_str(": ");
        out.push_str(&s.to_string());
        src = s.source();
    }
    out
}

/// Send one finished output to the UI and wake it. `false` when the UI hung up — the
/// worker should shut down cleanly.
fn send_output(tx: &Sender<Output>, ctx: &egui::Context, out: Output) -> bool {
    if tx.send(out).is_err() {
        return false;
    }
    ctx.request_repaint();
    true
}

/// Worker loop: pull tasks, do the blocking work off-thread, ship results back, wake the UI.
fn run_worker(rx: Receiver<Task>, tx: Sender<Output>, ctx: egui::Context) {
    while let Ok(task) = rx.recv() {
        // Opens and batch master decodes stream MULTIPLE outputs — handled outside the
        // one-task-one-output match below.
        let task = match task {
            Task::Io(IoTask::OpenDoc { slug, path, epoch }) => {
                if !handle_open_doc(slug, &path, epoch, &tx, &ctx) {
                    return;
                }
                continue;
            }
            Task::Io(IoTask::DecodeMasters { jobs, epoch }) => {
                if !handle_decode_masters(jobs, epoch, &tx, &ctx) {
                    return;
                }
                continue;
            }
            other => other,
        };
        let output = match task {
            Task::Render(req) => {
                let map_id = match &req.target {
                    RenderTarget::Map(id) => *id,
                    RenderTarget::LibPreview(_) => MapId::Albedo,
                };
                let layout = maps::build_layout(req.seed);
                let full = maps::render_map(map_id, &layout, req.seed);
                let thumb = maps::make_thumb(&full);
                Output::Rendered(RenderResult {
                    target: req.target,
                    seed: req.seed,
                    epoch: req.epoch,
                    full,
                    thumb,
                })
            }
            Task::Io(IoTask::LoadKey(provider)) => Output::Io(IoResult::KeyLoaded(provider, crate::secrets::load(provider))),
            Task::Io(IoTask::StoreKey(provider, key)) => Output::Io(IoResult::KeyStored(provider, crate::secrets::store(provider, &key))),
            Task::Io(IoTask::DeleteKey(provider)) => Output::Io(IoResult::KeyDeleted(provider, crate::secrets::delete(provider))),
            Task::Io(IoTask::Export {
                mut job,
                dir,
                derive_missing,
                albedo_src,
            }) => {
                // Derive the geometric maps at the albedo source's native resolution — full
                // detail into the export, off the UI thread.
                if let Some((albedo, side)) = albedo_src {
                    for (id, params) in derive_missing {
                        if id == MapId::Displacement {
                            // Keep the displacement in full f32 so export writes a TRUE
                            // 16-bit/EXR file; the 8-bit buffer is only for the contact sheet.
                            let hf = lumagen_core::derive::displacement_height_field_from_albedo(&albedo, side, &params);
                            job.maps[id.index()] = Some(hf.to_rgba8(false, 1.0, false));
                            job.map_sizes[id.index()] = side;
                            job.displacement_height = Some(hf);
                        } else {
                            let rgba = lumagen_core::derive::derive_map(id, &albedo, None, side, &params);
                            job.maps[id.index()] = Some(rgba);
                            job.map_sizes[id.index()] = side;
                        }
                    }
                }
                let result = lumagen_core::export::export_material(&job, &dir)
                    .map(|files| files.len())
                    .map_err(|e| error_chain(&e));
                Output::Io(IoResult::Exported(result))
            }
            Task::Io(IoTask::SaveDoc(job)) => {
                let SaveJob { path, mut doc, maps, announce } = *job;
                // A map that fails to encode fails the whole save — a document silently
                // missing a map would masquerade as a good backup. Untouched masters pass
                // their stored PNG section through byte-identical (no decode/re-encode).
                let result = maps
                    .into_iter()
                    .try_for_each(|m| match m.source {
                        SaveMapSource::Pixels(px) => doc.set_map_sized(m.id, &px, m.side, m.params),
                        SaveMapSource::Encoded { png, preview, working } => {
                            let preview = if preview.is_empty() {
                                // A pre-preview document self-heals here: one cheap 400²
                                // encode and every future open is fast.
                                lumagen_core::maps::encode_png_rgba(&working, TEX_SIZE).unwrap_or_default()
                            } else {
                                (*preview).clone()
                            };
                            doc.set_map_encoded(m.id, (*png).clone(), preview, m.side, m.params);
                            Ok(())
                        }
                    })
                    .and_then(|()| doc.save(&path))
                    .map_err(|e| error_chain(&e));
                Output::Io(IoResult::DocSaved { path, result, announce })
            }
            Task::Io(IoTask::ImportAlbedo { path, epoch }) => {
                let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "imported".into());
                let result = decode_albedo(&path).map(|rgba| (rgba, name));
                Output::Io(IoResult::AlbedoImported { result, epoch })
            }
            Task::Io(IoTask::Derive {
                id,
                albedo,
                params,
                size,
                epoch,
            }) => {
                let full = lumagen_core::derive::derive_map(id, &albedo, None, size, &params);
                // Thumbnails come from a working-size copy — make_thumb assumes TEX_SIZE input.
                let working = if size == TEX_SIZE {
                    full.clone()
                } else {
                    maps::resize_square_rgba(&full, size, TEX_SIZE)
                };
                let thumb = maps::make_thumb(&working);
                Output::Io(IoResult::Derived { id, full, thumb, size, epoch })
            }
            Task::Io(IoTask::ImportFolder { dir, epoch }) => Output::Io(IoResult::FolderImported {
                result: import_folder_maps(&dir),
                epoch,
            }),
            Task::Io(IoTask::ScanLibrary { dir }) => {
                let result = lumagen_core::library::scan_documents(&dir).map_err(|e| error_chain(&e));
                Output::Io(IoResult::LibraryScanned(result))
            }
            Task::Io(IoTask::DocPreview { slug, path, seed }) => {
                // Documents store the albedo at its native px_size (masters can be 2880²+),
                // so downscale to the working size. The stored working-size preview skips
                // the full master decode entirely.
                let full = lumagen_core::document::Document::load(&path)
                    .ok()
                    .and_then(|doc| {
                        if let Some(preview) = doc.get_map_preview(MapId::Albedo) {
                            return Some(preview);
                        }
                        let side = doc.map_px_size(MapId::Albedo);
                        doc.get_map(MapId::Albedo)
                            .map(|(rgba, _)| (rgba, side))
                            .filter(|(rgba, side)| *side >= 4 && rgba.len() == side * side * 4)
                            .map(|(rgba, side)| {
                                if side == TEX_SIZE {
                                    rgba
                                } else {
                                    maps::resize_square_rgba(&rgba, side, TEX_SIZE)
                                }
                            })
                    })
                    .unwrap_or_else(|| {
                        let layout = maps::build_layout(seed);
                        maps::render_map(MapId::Albedo, &layout, seed)
                    });
                Output::Io(IoResult::DocPreview { slug, full })
            }
            Task::Io(IoTask::OverrideMap { id, path, epoch }) => Output::Io(IoResult::MapOverridden {
                id,
                epoch,
                result: decode_albedo(&path),
            }),
            // Intercepted before this match (they stream multiple outputs); never reached.
            Task::Io(IoTask::OpenDoc { .. } | IoTask::DecodeMasters { .. }) => continue,
        };
        if !send_output(&tx, &ctx, output) {
            return;
        }
    }
}

/// An untouched master's encoded payloads: `(stored PNG section, stored working-size
/// preview — may be empty for pre-preview documents, native side)`.
pub type EncodedMaster = (std::sync::Arc<Vec<u8>>, std::sync::Arc<Vec<u8>>, usize);

/// One master decode job: `(map, stored PNG section, native side)`.
type DecodeJob = (MapId, std::sync::Arc<Vec<u8>>, usize);

/// Split decode jobs into greedy waves bounded by decoded-output bytes and core count —
/// a 16K master is 1 GiB raw, so eight in flight at once would spike past 8 GiB.
fn decode_waves(jobs: Vec<DecodeJob>) -> Vec<Vec<DecodeJob>> {
    const BUDGET_BYTES: usize = 2 << 30;
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut waves = Vec::new();
    let mut wave: Vec<DecodeJob> = Vec::new();
    let mut bytes = 0usize;
    for job in jobs {
        let out = job.2 * job.2 * 4;
        if !wave.is_empty() && (bytes + out > BUDGET_BYTES || wave.len() >= workers) {
            waves.push(std::mem::take(&mut wave));
            bytes = 0;
        }
        bytes += out;
        wave.push(job);
    }
    if !wave.is_empty() {
        waves.push(wave);
    }
    waves
}

/// Decode one stored master section, validating the pixel count against its declared side.
fn decode_master(png: &[u8], native: usize) -> Result<Vec<u8>, String> {
    match lumagen_core::document::decode_map_png(png) {
        Ok(rgba) if rgba.len() == native * native * 4 => Ok(rgba),
        Ok(rgba) => Err(format!("master pixel size mismatch: {} bytes for {native}²", rgba.len())),
        Err(e) => Err(error_chain(&e)),
    }
}

/// Decode a batch of masters across cores (budgeted waves), streaming each result back as
/// its own `MasterDecoded` the moment it lands. Returns `false` when the UI hung up.
fn handle_decode_masters(jobs: Vec<DecodeJob>, epoch: u64, tx: &Sender<Output>, ctx: &egui::Context) -> bool {
    for wave in decode_waves(jobs) {
        let hung_up = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|s| {
            for (id, png, native) in wave {
                let tx = tx.clone();
                let ctx = ctx.clone();
                let hung_up = &hung_up;
                s.spawn(move || {
                    let result = decode_master(&png, native).map(|rgba| (rgba, native));
                    if !send_output(&tx, &ctx, Output::Io(IoResult::MasterDecoded { id, epoch, result })) {
                        hung_up.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                });
            }
        });
        if hung_up.load(std::sync::atomic::Ordering::Relaxed) {
            return false;
        }
    }
    true
}

/// Open a document for the UI: manifest parse + working-size pixels ship in one
/// `DocOpened` (near-instant when the document carries previews). Masters are handed over
/// still ENCODED — nothing decodes until the 2D native view, Export, or Refine asks —
/// except in legacy pre-preview documents, whose masters must decode here (in parallel
/// waves, off the UI thread) to produce working copies at all; those stream back
/// immediately as `MasterDecoded`. Returns `false` when the UI hung up.
fn handle_open_doc(slug: String, path: &std::path::Path, epoch: u64, tx: &Sender<Output>, ctx: &egui::Context) -> bool {
    use lumagen_core::document::{Document, decode_map_png};
    let mut doc = match Document::load(path) {
        Ok(d) => d,
        Err(e) => {
            return send_output(
                tx,
                ctx,
                Output::Io(IoResult::DocOpened {
                    slug,
                    epoch,
                    result: Err(error_chain(&e)),
                }),
            );
        }
    };
    let tex = doc.tex_size;
    let mut working: Vec<(MapId, Vec<u8>, usize)> = Vec::new();
    let mut encoded_masters: Vec<(MapId, EncodedMaster)> = Vec::new();
    // Legacy masters (no stored preview): the working copy can only come from the full
    // decode, so these decode before `DocOpened` — in parallel waves below.
    let mut legacy_jobs: Vec<DecodeJob> = Vec::new();
    for id in MapId::ALL {
        let Some(mapdoc) = doc.maps.get_mut(&format!("{}", id.index())) else {
            continue;
        };
        if mapdoc.png.is_empty() {
            continue;
        }
        let native = if mapdoc.px_size == 0 { tex } else { mapdoc.px_size };
        if native > tex {
            let png = std::sync::Arc::new(std::mem::take(&mut mapdoc.png));
            let preview_png = std::sync::Arc::new(std::mem::take(&mut mapdoc.preview_png));
            let preview = if preview_png.is_empty() {
                None
            } else {
                decode_map_png(&preview_png).ok().filter(|p| p.len() == tex * tex * 4)
            };
            match preview {
                Some(p) => working.push((id, p, native)),
                None => legacy_jobs.push((id, std::sync::Arc::clone(&png), native)),
            }
            encoded_masters.push((id, (png, preview_png, native)));
        } else {
            match decode_map_png(&mapdoc.png) {
                Ok(rgba) if rgba.len() == native * native * 4 => {
                    working.push((id, rgba, native));
                    mapdoc.png = Vec::new();
                }
                Ok(rgba) => tracing::warn!(?id, len = rgba.len(), native, "skipping stored pixels with mismatched size"),
                Err(e) => tracing::warn!(?id, "skipping undecodable stored pixels: {e}"),
            }
        }
    }
    // Parallel legacy decode: working copies for `DocOpened` now, the full masters kept to
    // stream right behind it (they are already paid for).
    let mut ready: Vec<(MapId, Vec<u8>, usize)> = Vec::new();
    for wave in decode_waves(legacy_jobs) {
        let results = std::thread::scope(|s| {
            let handles: Vec<_> = wave
                .into_iter()
                .map(|(id, png, native)| (id, native, s.spawn(move || decode_master(&png, native))))
                .collect();
            handles
                .into_iter()
                .map(|(id, native, h)| match h.join() {
                    Ok(res) => (id, res, native),
                    Err(_) => (id, Err("master decode thread panicked".to_string()), native),
                })
                .collect::<Vec<_>>()
        });
        for (id, res, native) in results {
            match res {
                Ok(rgba) => {
                    working.push((id, maps::resize_square_rgba(&rgba, native, tex), native));
                    ready.push((id, rgba, native));
                }
                Err(e) => {
                    tracing::warn!(?id, "skipping undecodable stored master: {e}");
                    encoded_masters.retain(|(mid, _)| *mid != id);
                }
            }
        }
    }
    let pending_masters: Vec<MapId> = ready.iter().map(|r| r.0).collect();
    let opened = OpenedDoc {
        doc,
        working,
        encoded_masters,
        pending_masters,
    };
    if !send_output(
        tx,
        ctx,
        Output::Io(IoResult::DocOpened {
            slug,
            epoch,
            result: Ok(Box::new(opened)),
        }),
    ) {
        return false;
    }
    for (id, rgba, native) in ready {
        if !send_output(
            tx,
            ctx,
            Output::Io(IoResult::MasterDecoded {
                id,
                epoch,
                result: Ok((rgba, native)),
            }),
        ) {
            return false;
        }
    }
    true
}

/// Decode an image file (PNG/JPEG/etc.) to RGBA8 at the working `TEX_SIZE` — center-cropped
/// to square + resized, so non-square sources don't distort. Runs off the UI thread.
fn decode_albedo(path: &std::path::Path) -> Result<Vec<u8>, String> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(lumagen_core::maps::fit_to_working_rgba(&img, TEX_SIZE))
}

/// Scan a folder of loose map images into a material by matching map-name suffixes in filenames
/// (`*_albedo`, `*_roughness`, …, with common alternates). Requires an albedo. Runs off-thread.
fn import_folder_maps(dir: &std::path::Path) -> Result<ImportedFolder, String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let files: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| matches!(e.to_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp" | "tga" | "bmp"))
                .unwrap_or(false)
        })
        .collect();
    // Primary suffix + alternates per map.
    let matchers: [(MapId, &[&str]); 8] = [
        (MapId::Albedo, &["albedo", "basecolor", "base_color", "diffuse", "color"]),
        (MapId::Roughness, &["roughness", "rough"]),
        (MapId::Metallic, &["metallic", "metalness", "metal"]),
        (MapId::Normal, &["normal", "nrm"]),
        (MapId::Displacement, &["displacement", "height", "disp"]),
        (MapId::Ao, &["ambientocclusion", "occlusion", "_ao", "ao_"]),
        (MapId::Emission, &["emission", "emissive"]),
        (MapId::Transparency, &["transparency", "opacity", "alpha"]),
    ];
    let mut maps = Vec::new();
    for (id, keys) in matchers {
        let hit = files.iter().find(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| {
                    let s = s.to_lowercase();
                    keys.iter().any(|k| s.contains(k))
                })
                .unwrap_or(false)
        });
        if let Some(path) = hit {
            match decode_albedo(path) {
                Ok(rgba) => maps.push((id, rgba)),
                Err(e) => tracing::warn!("skipping {}: {e}", path.display()),
            }
        }
    }
    if !maps.iter().any(|(id, _)| *id == MapId::Albedo) {
        return Err("no albedo image found in the folder".into());
    }
    let slug = dir.file_name().and_then(|n| n.to_str()).unwrap_or("imported").to_string();
    Ok(ImportedFolder { slug, maps })
}

/// Full-size (`TEX_SIZE`) egui image for upload on the UI thread.
pub fn full_image(px: &[u8]) -> egui::ColorImage {
    egui::ColorImage::from_rgba_unmultiplied([TEX_SIZE, TEX_SIZE], px)
}

/// Thumbnail image for upload on the UI thread.
pub fn thumb_image(px: &[u8]) -> egui::ColorImage {
    egui::ColorImage::from_rgba_unmultiplied([THUMB_SIZE, THUMB_SIZE], px)
}
