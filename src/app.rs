//! Root application: screen routing, keyboard model, modals, toasts.

use eframe::egui;

use crate::data::{MAPS, MapId};
use crate::state::{AppState, Screen, UiPrefs, ViewMode};
use crate::theme;
use crate::widgets::ToastKind;

pub struct LumagenApp {
    pub state: AppState,
    /// Set when the OS asks to close with unsaved work; drives the confirm dialog.
    show_close_confirm: bool,
    /// The app's single tokio runtime (network + async file I/O). Generation and provider
    /// futures run on it; CPU rasterization stays on the std::thread render worker.
    runtime: tokio::runtime::Handle,
    /// The wgpu render state, when the wgpu backend is active. The 3D preview renders through it.
    wgpu_render_state: Option<eframe::egui_wgpu::RenderState>,
}

impl LumagenApp {
    /// Fallible per the eframe creator contract: an unrecoverable startup failure (no async
    /// runtime at all) returns an error for `run_native` to report instead of aborting.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        theme::apply(&cc.egui_ctx);
        egui_extras::install_image_loaders(&cc.egui_ctx);
        // Report which GPU/backend wgpu actually selected — a software adapter (llvmpipe)
        // is the usual cause of a laggy frame loop.
        if let Some(rs) = &cc.wgpu_render_state {
            let info = rs.adapter.get_info();
            tracing::info!(
                "wgpu adapter: {} · backend {:?} · device {:?} · {:?}",
                info.name,
                info.backend,
                info.device_type,
                info.vendor
            );
        } else {
            tracing::warn!("no wgpu render state — running on the glow fallback?");
        }
        let mut state = AppState::new();
        // A missing or malformed persisted prefs value just yields defaults, so first run
        // and schema changes are clean.
        if let Some(prefs) = cc.storage.and_then(|s| eframe::get_value::<UiPrefs>(s, crate::state::PREFS_KEY)) {
            state.apply_prefs(prefs);
        }
        // Apply the restored accent + theme to the live style (theme::apply already ran with the
        // defaults, so re-apply now that we know the user's choices).
        let accent = state.settings.accent.clone();
        crate::theme::set_accent_name(&accent);
        crate::theme::set_theme(state.settings.theme);
        crate::theme::apply(&cc.egui_ctx);
        // pre-render the albedo so the workspace starts populated
        let seed = state.seed;
        state.layout = crate::maps::build_layout(seed);
        let keep_alive = |rt: tokio::runtime::Runtime| {
            let handle = rt.handle().clone();
            std::mem::forget(rt); // keep the runtime alive until process exit
            handle
        };
        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("lumagen-tokio")
            .build()
        {
            Ok(rt) => keep_alive(rt),
            Err(e) => {
                // Runtime construction only fails on OS resource exhaustion; log and keep a
                // current-thread fallback so the app still runs (generation may be disabled).
                tracing::error!("multi-thread tokio runtime failed ({e}); using current-thread");
                match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                    Ok(rt) => keep_alive(rt),
                    Err(e2) => {
                        // Even the fallback failed — surface a real startup error through
                        // eframe's fallible creator instead of aborting the process.
                        tracing::error!("tokio runtime unavailable: {e2}");
                        return Err(format!("could not start an async runtime: {e2}").into());
                    }
                }
            }
        };
        Ok(Self {
            state,
            show_close_confirm: false,
            runtime,
            wgpu_render_state: cc.wgpu_render_state.clone(),
        })
    }

    /// Spawn a future on the app's tokio runtime.
    pub fn spawn<F>(&self, fut: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.runtime.spawn(fut)
    }

    /// Build the image provider **and** the model slugs (text-to-image for the albedo, image
    /// edit for the maps — fal splits these into separate endpoints) + backend params from
    /// settings: fal.ai primary, OpenRouter alternate (one slug serves both), mock when no
    /// key. The key comes from the OS vault — env vars are a dev fallback. A one-off
    /// synchronous vault read at batch launch is acceptable (the batch is async).
    fn provider_config(
        state: &AppState,
    ) -> (
        std::sync::Arc<dyn lumagen_core::providers::ImageProvider>,
        String,
        String,
        lumagen_core::providers::BackendParams,
    ) {
        use lumagen_core::providers::{BackendParams, FalProvider, MockProvider, OpenRouterProvider};
        use std::sync::Arc;
        let s = &state.settings;
        let fal_key = crate::secrets::load(crate::secrets::Provider::Fal)
            .or_else(|| std::env::var("FAL_KEY").ok().filter(|k| !k.is_empty()))
            .or_else(|| std::env::var("FAL_API_KEY").ok().filter(|k| !k.is_empty()));
        let or_key = crate::secrets::load(crate::secrets::Provider::OpenRouter).or_else(|| std::env::var("OPENROUTER_API_KEY").ok().filter(|k| !k.is_empty()));
        // Quality tier for models that take one (gpt-image family) — Settings → Generation.
        let quality = ["low", "medium", "high", "auto"].get(s.quality).copied().unwrap_or("high").to_string();
        let fal_backend = || BackendParams::Fal {
            guidance_scale: Some(3.5),
            num_inference_steps: None,
            strength: None,
            negative_prompt: None,
            sync_mode: Some(false),
            quality: Some(quality.clone()),
            extra: serde_json::Value::Null,
        };
        // Overall submit→result deadline (Settings → Generation → Job timeout): big image
        // models can legitimately take several minutes.
        let timeout = s.job_timeout_duration();
        if let Some(key) = fal_key {
            (
                Arc::new(FalProvider::new(key).with_timeout(timeout)),
                s.fal_model_t2i.clone(),
                s.fal_model_edit.clone(),
                fal_backend(),
            )
        } else if let Some(key) = or_key {
            let mut cfg = serde_json::Map::new();
            cfg.insert("aspect_ratio".to_string(), serde_json::Value::String("1:1".to_string()));
            let backend = BackendParams::OpenRouter {
                modalities: vec!["image".to_string(), "text".to_string()],
                image_config: serde_json::Value::Object(cfg),
                provider: None,
            };
            (
                Arc::new(OpenRouterProvider::new(key).with_timeout(timeout)),
                s.or_model.clone(),
                s.or_model.clone(),
                backend,
            )
        } else {
            (Arc::new(MockProvider), s.fal_model_t2i.clone(), s.fal_model_edit.clone(), fal_backend())
        }
    }

    /// Build the prompt context (base/final/description + the user's per-map templates +
    /// negative) that carries the Settings → Prompts edits into every request.
    fn prompt_context(state: &AppState) -> lumagen_core::generate::PromptContext {
        let s = &state.settings;
        lumagen_core::generate::PromptContext {
            base_template: s.base_template.clone(),
            final_requirement: s.final_requirement.clone(),
            description: state.description.clone(),
            map_templates: std::array::from_fn(|i| s.map_templates.get(&MapId::ALL[i]).cloned().unwrap_or_default()),
            negative: s.negative_prompt.clone(),
        }
    }

    fn retries(&self) -> usize {
        self.state.settings.retries.parse().unwrap_or(3)
    }

    /// The reference every generated map is locked to: the REAL albedo pixels (generated /
    /// imported / promoted), plus the full-resolution master when one exists. `albedo_rgba()`
    /// renders the procedural stand-in and must never be the reference once a real albedo
    /// exists.
    fn albedo_reference(&self) -> (Vec<u8>, Option<(Vec<u8>, usize)>) {
        (
            self.state.export_albedo(),
            self.state.master_pixels.get(&MapId::Albedo).map(|m| (m.0.as_ref().clone(), m.1)),
        )
    }

    /// Bridge a "generate" request from the UI onto the tokio runtime. Builds the provider,
    /// prompt, and plan from settings + per-map state, generates only the queued target maps,
    /// and streams finished maps back through the worker's generation channel.
    #[tracing::instrument(skip_all)]
    fn launch_generation(&mut self, ctx: &egui::Context) {
        use lumagen_core::generate::{DepthReconstruct, GenerationPlan, MapGenerator};
        use lumagen_core::providers::ImageSpec;

        // Maps are image edits — use the edit-endpoint slug.
        let (provider, _t2i, model, backend) = Self::provider_config(&self.state);
        let prompt = Self::prompt_context(&self.state);
        let max_conc = self.state.settings.max_concurrency.parse::<usize>().unwrap_or(3).max(1);
        let retries = self.retries();
        let paths = std::array::from_fn(|i| self.state.maps[i].derive_path);
        let derive_params = std::array::from_fn(|i| self.state.maps[i].derive_params());
        // Depth-model reconstruction is configured from the displacement map's own controls:
        // the depth-model picker, the Invert toggle, and the macro↔micro split slider (0..100
        // → a ~0.4%..3.4%-of-side crossover — a larger split keeps the displacement broader).
        let disp = &self.state.maps[MapId::Displacement.index()];
        let depth = DepthReconstruct {
            depth_models: disp.depth_mode.models(),
            invert: disp.invert,
            crossover_radius: 0.004 + (disp.detail_routing / 100.0).clamp(0.0, 1.0) * 0.03,
            ..DepthReconstruct::default()
        };
        let plan = GenerationPlan {
            paths,
            model,
            backend,
            max_concurrency: max_conc,
            retries,
            size: Some(ImageSpec::Aspect("1:1".into())),
            derive_params,
            depth,
        };
        let (albedo, albedo_master) = self.albedo_reference();
        let size = crate::maps::TEX_SIZE;
        let seed = self.state.seed as i64;
        // Drain the launch buffer — only the newly queued maps go into this task, so a later
        // single-map generation extends the batch with its own concurrent task instead of
        // relaunching (or being blocked by) the maps already in flight.
        let targets = if self.state.queue.is_empty() {
            MapId::DERIVED.to_vec()
        } else {
            std::mem::take(&mut self.state.queue)
        };
        let handle = self.state.generation_sender(ctx);
        let cancel = tokio_util::sync::CancellationToken::new();
        self.state.gen_cancel = Some(cancel.clone());
        // A per-map child token so the job queue can cancel a single map. Extend (don't
        // replace) — earlier launches' tokens must stay cancellable.
        let tokens: std::collections::HashMap<_, _> = targets.iter().map(|id| (*id, cancel.child_token())).collect();
        self.state.gen_map_cancel.extend(tokens.iter().map(|(k, v)| (*k, v.clone())));

        let generator = MapGenerator::new(provider);
        self.spawn(async move {
            generator
                .generate_material(albedo, albedo_master, size, targets, prompt, plan, seed, tokens, handle.into_sender(), cancel)
                .await;
        });
    }

    /// Launch a 4-candidate variation run for one map on the tokio runtime. The candidates
    /// stream back as a `Variations` outcome and land in the variations modal's previews.
    #[tracing::instrument(skip(self, ctx), fields(?map))]
    fn launch_variations(&mut self, ctx: &egui::Context, map: MapId) {
        use lumagen_core::generate::{GenerationPlan, MapGenerator};

        // Variations are image edits — use the edit-endpoint slug.
        let (provider, _t2i, model, backend) = Self::provider_config(&self.state);
        let prompt = Self::prompt_context(&self.state);
        let retries = self.retries();
        let plan = GenerationPlan {
            model,
            backend,
            retries,
            ..Default::default()
        };
        let (albedo, albedo_master) = self.albedo_reference();
        let seed = self.state.seed as i64;
        let handle = self.state.generation_sender(ctx);
        let cancel = tokio_util::sync::CancellationToken::new();
        // Held so closing the variations modal (or switching materials) can cancel the
        // paid run instead of letting it bill into a closed UI.
        self.state.variations_cancel = Some(cancel.clone());

        let generator = MapGenerator::new(provider);
        self.spawn(async move {
            generator
                .generate_variations(albedo, albedo_master, map, 4, prompt, plan, seed, handle.into_sender(), cancel)
                .await;
        });
    }

    /// Generate a new albedo from the material description via the provider (text-to-image), on
    /// the tokio runtime. Offline (mock) this renders a procedural albedo at no cost.
    #[tracing::instrument(skip_all)]
    fn launch_albedo_generation(&mut self, ctx: &egui::Context) {
        use lumagen_core::generate::{GenerationPlan, MapGenerator};
        use lumagen_core::providers::ImageSpec;

        // The albedo is text-to-image — use the t2i-endpoint slug.
        let (provider, model, _edit, backend) = Self::provider_config(&self.state);
        let retries = self.retries();
        // The requested generation size: Max (0) walks the highest-supported tier ladder;
        // otherwise the explicit square side.
        let size_spec = if self.state.gen_size == 0 {
            ImageSpec::MaxSquare
        } else {
            ImageSpec::Pixels {
                width: self.state.gen_size as u32,
                height: self.state.gen_size as u32,
            }
        };
        let plan = GenerationPlan {
            model,
            backend,
            retries,
            size: Some(size_spec),
            ..Default::default()
        };
        let description = self.state.description.clone();
        let negative = self.state.settings.negative_prompt.clone();
        let seed = self.state.seed as i64;
        let handle = self.state.generation_sender(ctx);
        let cancel = tokio_util::sync::CancellationToken::new();
        // Keep the token so the generating hero's Cancel can stop a long-running job.
        self.state.albedo_cancel = Some(cancel.clone());
        let generator = MapGenerator::new(provider);
        self.spawn(async move {
            generator.generate_albedo(description, negative, plan, seed, handle.into_sender(), cancel).await;
        });
    }

    /// Launch Topaz upscales for `targets` on the tokio runtime (Step ③ Refine): each map's
    /// best source (master, else working copy) goes to the upscaler; results stream back as
    /// Ready outcomes with new full-resolution masters.
    #[tracing::instrument(skip(self, ctx), fields(n = targets.len()))]
    fn launch_upscale(&mut self, ctx: &egui::Context, targets: Vec<MapId>) {
        use lumagen_core::generate::MapGenerator;
        let (provider, _t2i, _edit, _backend) = Self::provider_config(&self.state);
        let model = self.state.settings.upscale_model.clone();
        let retries = self.retries();
        for id in targets {
            let Some((rgba, side)) = self.state.upscale_source(id) else {
                self.state.upscaling.remove(&id);
                continue;
            };
            let target = self.state.upscale_target(side);
            if side >= target {
                self.state.upscaling.remove(&id);
                self.state
                    .toast(format!("{} is already at the export size", crate::data::map_def(id).name), ToastKind::Info);
                continue;
            }
            let handle = self.state.generation_sender(ctx);
            let cancel = tokio_util::sync::CancellationToken::new();
            let generator = MapGenerator::new(std::sync::Arc::clone(&provider));
            let model = model.clone();
            self.spawn(async move {
                // Unshared source buffers unwrap for free; a shared master is copied here on
                // the runtime worker instead of stalling the UI thread with a 16K² memcpy.
                let rgba = std::sync::Arc::try_unwrap(rgba).unwrap_or_else(|arc| (*arc).clone());
                generator
                    .upscale_map(id, rgba, side, target, model, retries, handle.into_sender(), cancel)
                    .await;
            });
        }
    }

    /// Run the optional LLM-powered Material Description Check on the tokio runtime, replacing the
    /// description with the model's improved version. Uses the stored LLM key + configured
    /// endpoint/model (OpenAI-compatible).
    #[tracing::instrument(skip_all)]
    fn launch_enhance_description(&mut self, ctx: &egui::Context) {
        let buf = self.state.settings.llm_key_input.clone();
        let key = if buf.trim().is_empty() {
            crate::secrets::load(crate::secrets::Provider::Llm).unwrap_or_default()
        } else {
            buf
        };
        let endpoint = self.state.settings.endpoint.clone();
        let model = self.state.settings.llm_model.clone();
        let description = self.state.description.clone();
        // The result only auto-applies if the text still matches this baseline — edits
        // typed while the LLM runs win over the suggestion.
        self.state.enhance_baseline = Some(description.clone());
        let handle = self.state.async_handle(ctx);
        if key.trim().is_empty() {
            handle.toast("Add an LLM API key in Settings → Providers first", ToastKind::Error);
            return;
        }
        let system = "You are a PBR material description assistant. Rewrite the user's material \
description, keeping their intent and wording, but add only PBR-relevant details that are \
missing: which regions are bare metal vs dielectric, coating/paint finish, glass or transparent \
parts, rubber/plastic, wear/oxidation/grime, emitters (lights/LEDs), macro relief \
(plates/bays/ribs), micro relief (seams/screws/grille), and confirm it is seamless/tileable. \
Invent no new subjects. Return ONLY the improved description as plain prose, no preamble."
            .to_string();
        self.spawn(async move {
            match lumagen_core::providers::chat_complete(&endpoint, &key, &model, &system, &description).await {
                Ok(improved) if !improved.trim().is_empty() => {
                    handle.set_description(improved);
                    handle.toast("Description enhanced by AI", ToastKind::Success);
                }
                Ok(_) => handle.toast("The model returned no changes", ToastKind::Info),
                Err(e) => handle.toast(format!("Enhance failed — {e}"), ToastKind::Error),
            }
        });
    }

    /// Run a real, cheap "test connection" request for a provider on the tokio runtime and toast
    /// the actual outcome. The key is the just-typed edit buffer if present, else the OS vault,
    /// else an environment variable.
    #[tracing::instrument(skip(self, ctx), fields(?provider))]
    fn launch_test_connection(&mut self, ctx: &egui::Context, provider: crate::secrets::Provider) {
        use crate::secrets::Provider;
        use lumagen_core::providers::ProviderId;
        let (id, models, buf_key) = match provider {
            // Validate BOTH fal slots — the albedo (t2i) and maps (edit) endpoints.
            Provider::Fal => (
                ProviderId::Fal,
                vec![self.state.settings.fal_model_t2i.clone(), self.state.settings.fal_model_edit.clone()],
                self.state.settings.fal_key_input.clone(),
            ),
            Provider::OpenRouter => (
                ProviderId::OpenRouter,
                vec![self.state.settings.or_model.clone()],
                self.state.settings.openrouter_key_input.clone(),
            ),
            // No other provider is an image backend; treat it as OpenRouter for the check.
            Provider::OpenAi | Provider::Llm => (ProviderId::OpenRouter, vec![self.state.settings.or_model.clone()], String::new()),
        };
        let key = if buf_key.trim().is_empty() {
            crate::secrets::load(provider)
                .or_else(|| match provider {
                    Provider::Fal => std::env::var("FAL_KEY").ok().or_else(|| std::env::var("FAL_API_KEY").ok()),
                    Provider::OpenRouter => std::env::var("OPENROUTER_API_KEY").ok(),
                    _ => None,
                })
                .unwrap_or_default()
        } else {
            buf_key
        };
        let handle = self.state.async_handle(ctx);
        self.spawn(async move {
            // Report both ways: a toast for immediate visibility and the inline status in
            // the provider card so the outcome persists past the toast's 3 seconds.
            match lumagen_core::providers::test_connection(id, &key, &models).await {
                Ok(msg) => {
                    handle.test_result(provider, Ok(msg.clone()));
                    handle.toast(msg, ToastKind::Success);
                }
                Err(e) => {
                    handle.test_result(provider, Err(e.to_string()));
                    handle.toast(format!("Test failed — {e}"), ToastKind::Error);
                }
            }
        });
    }

    fn handle_keyboard(&mut self, ctx: &egui::Context) {
        let mut actions: Vec<KeyAction> = Vec::new();
        // Shortcuts must not fire while a text field is being edited. Read the focus state
        // BEFORE `ctx.input` (calling a `ctx.*` accessor inside the closure would re-lock the
        // Context and deadlock). Note: `RawInput::focused` is the OS *window* focus — using it
        // here silenced every shortcut whenever the window was focused.
        let typing = ctx.egui_wants_keyboard_input();
        ctx.input(|i| {
            if typing {
                return;
            }
            if i.key_pressed(egui::Key::Tab) {
                actions.push(KeyAction::Toggle2d3d);
            }
            if i.key_pressed(egui::Key::T) {
                actions.push(KeyAction::Tiled);
            }
            if i.key_pressed(egui::Key::F) {
                actions.push(KeyAction::Fit);
            }
            for def in &MAPS {
                let key = match def.key {
                    '1' => egui::Key::Num1,
                    '2' => egui::Key::Num2,
                    '3' => egui::Key::Num3,
                    '4' => egui::Key::Num4,
                    '5' => egui::Key::Num5,
                    '6' => egui::Key::Num6,
                    '7' => egui::Key::Num7,
                    '8' => egui::Key::Num8,
                    _ => egui::Key::Num1,
                };
                if i.key_pressed(key) {
                    actions.push(KeyAction::SelectMap(def.id));
                }
            }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::G) {
                actions.push(KeyAction::GenerateAll);
            }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::E) {
                actions.push(KeyAction::Export);
            }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::Comma) {
                actions.push(KeyAction::Settings);
            }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::L) {
                actions.push(KeyAction::Library);
            }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::S) {
                actions.push(KeyAction::Save);
            }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::Z) {
                actions.push(if i.modifiers.shift { KeyAction::Redo } else { KeyAction::Undo });
            }
        });
        for action in actions {
            match action {
                KeyAction::Toggle2d3d => {
                    if self.state.screen == Screen::Workspace {
                        self.state.view = if self.state.view == ViewMode::Material3d {
                            ViewMode::Map2d
                        } else {
                            ViewMode::Material3d
                        };
                    }
                }
                KeyAction::Tiled => {
                    if self.state.screen == Screen::Workspace {
                        self.state.view = ViewMode::Tiled;
                    }
                }
                KeyAction::Fit => {
                    self.state.zoom = 1.0;
                    self.state.pan = egui::Vec2::ZERO;
                }
                KeyAction::SelectMap(id) => {
                    if self.state.screen == Screen::Workspace {
                        self.state.select(id);
                    }
                }
                KeyAction::GenerateAll => {
                    if self.state.screen == Screen::Workspace && !self.state.generating {
                        if self.state.has_map(crate::data::MapId::Albedo) {
                            self.state.show_assist = true;
                        } else {
                            self.state.toast("Generate the albedo first — every map derives from it", ToastKind::Cost);
                        }
                    }
                }
                KeyAction::Export => {
                    if self.state.screen == Screen::Workspace {
                        if self.state.generated_all() && self.state.has_map(crate::data::MapId::Albedo) {
                            self.state.active_step = 3;
                            self.state.show_export = true;
                        } else {
                            self.state.toast("Generate the maps first", ToastKind::Cost);
                        }
                    }
                }
                KeyAction::Settings => {
                    if self.state.screen != Screen::Settings {
                        self.state.settings.return_to = self.state.screen;
                        self.state.screen = Screen::Settings;
                    }
                }
                KeyAction::Library => {
                    self.state.screen = Screen::Library;
                }
                KeyAction::Undo => {
                    if self.state.screen == Screen::Workspace {
                        self.state.undo();
                    }
                }
                KeyAction::Redo => {
                    if self.state.screen == Screen::Workspace {
                        self.state.redo();
                    }
                }
                KeyAction::Save => {
                    // Async: the worker encodes + writes; the result toast arrives via DocSaved.
                    let path = self.state.save_document(ctx, true);
                    tracing::info!("save dispatched → {}", path.display());
                }
            }
        }
    }

    /// Veto an OS close request while there is unsaved work; run the confirm dialog.
    /// Every close request is vetoed while the dialog is up — a second window-X must not
    /// slip past the confirm and discard the work the first one was asking about.
    fn handle_close_request(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.viewport().close_requested()) && self.state.unsaved {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.show_close_confirm = true;
        }
        if self.show_close_confirm {
            let mut stay = false;
            let mut discard = false;
            egui::Window::new("Unsaved changes")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label(format!("{} has unsaved changes.", self.state.material_name));
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Keep editing").clicked() {
                            stay = true;
                        }
                        if ui.button("Discard and quit").clicked() {
                            discard = true;
                        }
                    });
                });
            if stay {
                self.show_close_confirm = false;
            } else if discard {
                self.show_close_confirm = false;
                self.state.unsaved = false;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }
}

enum KeyAction {
    Toggle2d3d,
    Tiled,
    Fit,
    SelectMap(MapId),
    GenerateAll,
    Export,
    Settings,
    Library,
    Save,
    Undo,
    Redo,
}

impl eframe::App for LumagenApp {
    /// Per-frame non-drawing work: advance the generation state machine, upload any
    /// textures that became ready, and read the keyboard. Runs before each `ui` and also
    /// when the window is hidden but a repaint was requested.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Load stored API keys from the OS vault once at startup (off the UI thread), so key
        // state is correct everywhere — not only after the Settings screen was opened.
        if !self.state.keys_loaded {
            self.state.keys_loaded = true;
            self.state.request_load_keys(ctx);
        }
        self.state.ensure_textures(ctx);
        self.state.process_worker_results(ctx);
        // Re-derive any geometric maps whose adjust sliders changed (debounced, off-thread).
        self.state.flush_adjust(ctx);
        // Launch queued work (set by the UI this frame) on the tokio runtime.
        if self.state.pending_generate {
            self.state.pending_generate = false;
            self.launch_generation(ctx);
        }
        if let Some(map) = self.state.pending_variations.take() {
            self.launch_variations(ctx, map);
        }
        if let Some(provider) = self.state.pending_test.take() {
            self.launch_test_connection(ctx, provider);
        }
        if self.state.pending_albedo_gen {
            self.state.pending_albedo_gen = false;
            self.launch_albedo_generation(ctx);
        }
        if !self.state.pending_upscale.is_empty() {
            let targets = std::mem::take(&mut self.state.pending_upscale);
            self.launch_upscale(ctx, targets);
        }
        // Upscale results arrive over the generation channel; keep the loop draining.
        if !self.state.upscaling.is_empty() {
            ctx.request_repaint();
        }
        if self.state.pending_enhance {
            self.state.pending_enhance = false;
            self.launch_enhance_description(ctx);
        }
        // While a batch runs, keep the frame loop alive on every screen so generation results
        // are drained promptly (the incidental status-dot repaint only fires on the Workspace).
        if self.state.generating {
            ctx.request_repaint();
        }
        // Standalone albedo generations and variation runs report over the generation
        // channel, whose senders don't wake the loop — poll gently until they land.
        if self.state.async_work_pending() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        self.handle_keyboard(ctx);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.handle_close_request(&ctx);

        match self.state.screen {
            Screen::Library => crate::library::show(ui, &mut self.state),
            Screen::Workspace => crate::workspace::show(ui, &mut self.state, self.wgpu_render_state.as_ref()),
            Screen::Settings => crate::settings::show(ui, &mut self.state),
        }

        crate::modals::show_all(&ctx, &mut self.state);
        self.state.toasts.show(&ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, crate::state::PREFS_KEY, &self.state.prefs());
    }

    fn on_exit(&mut self) {
        // A save queued on the worker milliseconds before close must still land — the
        // worker thread dies with the process, so wait (bounded) for pending writes.
        self.state.drain_pending_saves_on_exit(std::time::Duration::from_secs(15));
    }
}
