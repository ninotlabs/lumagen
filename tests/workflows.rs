//! End-to-end workflow test: create project → create material → the albedo generates for
//! REAL through the orchestrator (offline mock provider, real channels, real state
//! application; the workspace shows a genuine Generating state, never a placeholder posing
//! as the result) → generate all maps → autosave → rescan → reload → export the files.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::disallowed_methods)]

use eframe::egui;
use lumagen::data::MapId;
use lumagen::state::{AppState, MapStatus, Screen};

fn unique_tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("lumagen_wf_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir
}

/// Step the harness — sleeping between frames so background workers can deliver results —
/// until `cond` holds. Returns `false` on timeout (~8 s).
fn wait_until(harness: &mut egui_kittest::Harness<'_, AppState>, mut cond: impl FnMut(&AppState) -> bool) -> bool {
    for _ in 0..400 {
        harness.step();
        if cond(harness.state()) {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    false
}

/// Mirror `LumagenApp`'s generation bridging (mock provider) inside the harness closure.
fn bridge_generation(state: &mut AppState, ctx: &egui::Context, rt: &tokio::runtime::Handle) {
    use lumagen_core::generate::{GenerationPlan, MapGenerator, PromptContext};
    use lumagen_core::providers::MockProvider;
    let prompt_ctx = |state: &AppState| PromptContext {
        base_template: state.settings.base_template.clone(),
        final_requirement: state.settings.final_requirement.clone(),
        description: state.description.clone(),
        map_templates: std::array::from_fn(|i| state.settings.map_templates.get(&MapId::ALL[i]).cloned().unwrap_or_default()),
        negative: state.settings.negative_prompt.clone(),
    };
    if state.pending_albedo_gen {
        state.pending_albedo_gen = false;
        let plan = GenerationPlan {
            retries: 0,
            ..Default::default()
        };
        let description = state.description.clone();
        let negative = state.settings.negative_prompt.clone();
        let seed = state.seed as i64;
        let handle = state.generation_sender(ctx);
        let cancel = tokio_util::sync::CancellationToken::new();
        let generator = MapGenerator::new(std::sync::Arc::new(MockProvider));
        rt.spawn(async move {
            generator.generate_albedo(description, negative, plan, seed, handle.into_sender(), cancel).await;
        });
    }
    if !state.pending_upscale.is_empty() {
        let targets = std::mem::take(&mut state.pending_upscale);
        let model = state.settings.upscale_model.clone();
        for id in targets {
            let Some((rgba, side)) = state.upscale_source(id) else {
                state.upscaling.remove(&id);
                continue;
            };
            let target = state.upscale_target(side);
            let handle = state.generation_sender(ctx);
            let cancel = tokio_util::sync::CancellationToken::new();
            let generator = MapGenerator::new(std::sync::Arc::new(MockProvider));
            let model = model.clone();
            rt.spawn(async move {
                // Mirrors the app: unwrap the shared handle (or copy) off the UI thread.
                let rgba = std::sync::Arc::try_unwrap(rgba).unwrap_or_else(|arc| (*arc).clone());
                generator.upscale_map(id, rgba, side, target, model, 0, handle.into_sender(), cancel).await;
            });
        }
    }
    if state.pending_generate {
        state.pending_generate = false;
        let plan = GenerationPlan {
            paths: std::array::from_fn(|i| state.maps[i].derive_path),
            derive_params: std::array::from_fn(|i| state.maps[i].derive_params()),
            retries: 0,
            ..Default::default()
        };
        let prompt = prompt_ctx(state);
        let albedo = state.export_albedo();
        let albedo_master = state.master_pixels.get(&MapId::Albedo).map(|m| (m.0.as_ref().clone(), m.1));
        let seed = state.seed as i64;
        // Drain the launch buffer, exactly like the app does — later requests extend the
        // batch with their own task.
        let targets = std::mem::take(&mut state.queue);
        let handle = state.generation_sender(ctx);
        let cancel = tokio_util::sync::CancellationToken::new();
        let generator = MapGenerator::new(std::sync::Arc::new(MockProvider));
        rt.spawn(async move {
            generator
                .generate_material(
                    albedo,
                    albedo_master,
                    lumagen::maps::TEX_SIZE,
                    targets,
                    prompt,
                    plan,
                    seed,
                    std::collections::HashMap::new(),
                    handle.into_sender(),
                    cancel,
                )
                .await;
        });
    }
}

#[test]
fn full_workflow_create_generate_save_scan_export() {
    let tmp = unique_tmp("e2e");
    let lib_dir = tmp.join("library");
    let export_dir = tmp.join("export");
    std::fs::create_dir_all(&export_dir).expect("mkdir export");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime");
    let rt = runtime.handle().clone();

    // ── 1. Create a project and a material, as the wizard does ──
    let mut state = AppState::new();
    state.show_intro = false;
    state.settings.library_location = lib_dir.display().to_string();
    state.settings.autosave = true;
    state.create_project("Test Project");
    assert_eq!(state.project_name, "Test Project");
    state.mat_type = "Metal".into();
    state.new_material_from_create("test_hull_777".into(), 777);
    assert_eq!(state.material_name, "test_hull_777");
    assert!(matches!(state.screen, Screen::Workspace));
    let entry = state.library.iter().find(|m| m.slug == "test_hull_777").expect("registered in library");
    assert_eq!(entry.project, "Test Project");
    assert_eq!(entry.seed, 777);

    // The albedo is honestly *generating* — no placeholder result, no fake history entry.
    assert_eq!(state.map(MapId::Albedo).status, MapStatus::Generating);
    assert!(state.map(MapId::Albedo).versions.is_empty(), "no fake v1 before the real result");
    assert!(!state.has_map(MapId::Albedo));

    state.start_generate_all();
    assert!(!state.generating, "generate-all must be blocked while the albedo is generating");

    state.pending_albedo_gen = true;

    // ── 2. Drive the state in a headless frame loop; bridge generation like the app does ──
    let mut harness = egui_kittest::Harness::builder().with_size(egui::Vec2::new(1200.0, 800.0)).build_ui_state(
        move |ui, state: &mut AppState| {
            let ctx = ui.ctx().clone();
            state.ensure_textures(&ctx);
            state.process_worker_results(&ctx);
            bridge_generation(state, &ctx, &rt);
            lumagen::workspace::show(ui, state, None);
        },
        state,
    );

    assert!(wait_until(&mut harness, |s| s.has_map(MapId::Albedo)), "albedo generation did not complete");
    assert_eq!(harness.state().map(MapId::Albedo).versions.len(), 1, "the real result is v1");

    // Single-map generations run CONCURRENTLY: request roughness, then metallic while the
    // first is still in flight — the second must extend the batch, not be blocked.
    harness.state_mut().reroll(MapId::Roughness);
    assert!(harness.state().generating);
    harness.state_mut().reroll(MapId::Metallic);
    assert_eq!(harness.state().gen_total, 2, "second map extends the running batch");
    assert_eq!(harness.state().map(MapId::Metallic).status, MapStatus::Queued);
    let pair_done = wait_until(&mut harness, |s| s.has_map(MapId::Roughness) && s.has_map(MapId::Metallic) && !s.generating);
    assert!(pair_done, "concurrent single-map generations did not both complete");

    harness.state_mut().start_generate_all();
    assert!(harness.state().generating, "generate-all runs once the albedo is real");
    let done = wait_until(&mut harness, |s| !s.generating && s.generated_all());
    assert!(done, "generation did not complete: {:?}", harness.state().queue);
    harness.step();
    harness.step();

    for id in MapId::DERIVED {
        assert!(harness.state().map(id).status.has_image(), "{id:?} missing");
        assert!(harness.state().map_pixels.contains_key(&id), "{id:?} pixels missing");
    }
    assert!(harness.state().spend.abs() < 1e-9, "offline mock must not be billed");

    // ── 3. Step ③ Refine: upscale everything (mock = offline resize, free) ──
    let up_targets = harness.state().upscale_targets();
    assert!(up_targets.contains(&MapId::Albedo), "albedo is an upscale target");
    assert!(up_targets.contains(&MapId::Roughness), "AI maps are upscale targets");
    assert!(!up_targets.contains(&MapId::Normal), "derived maps re-derive at export — not upscaled");
    // 400² source, 4096² Refine target: two chained passes (400 → 1600 → 4096).
    let expected_side = harness.state().upscale_target(400);
    assert_eq!(expected_side, 4096);
    assert_eq!(lumagen_core::generate::upscale_chain(400, expected_side), vec![1600, 4096]);
    for &id in &up_targets {
        harness.state_mut().upscaling.insert(id);
        harness.state_mut().pending_upscale.push(id);
    }
    let upscaled = wait_until(&mut harness, |s| {
        s.upscaling.is_empty() && up_targets.iter().all(|id| s.master_pixels.get(id).map(|m| m.1) == Some(expected_side))
    });
    assert!(upscaled, "upscales did not complete");
    assert_eq!(
        harness.state().map(MapId::Albedo).versions.len(),
        2,
        "the upscale lands as a new version (the original stays restorable)"
    );
    assert!(harness.state().spend.abs() < 1e-9, "offline upscale must not be billed");

    // ── 4. Autosave wrote the .lumagen; a fresh session scans and reloads it ──
    // Saves are ASYNC (worker-side PNG encode + write), and earlier partial autosaves may
    // land first — wait until the on-disk document carries all 8 maps.
    let doc_path = lib_dir.join("test_hull_777.lumagen");
    let persisted = wait_until(&mut harness, |_| {
        doc_path.exists()
            && lumagen_core::library::scan_documents(&lib_dir).is_ok_and(|found| found.iter().any(|m| m.slug == "test_hull_777" && m.coverage == 8))
    });
    assert!(persisted, "autosave never persisted all 8 maps to {}", doc_path.display());
    let saved_entry = harness.state().library.iter().find(|m| m.slug == "test_hull_777").expect("entry");
    assert!(saved_entry.doc_path.is_some(), "library entry points at the saved document");

    // ── 4a. The ASYNC open path: the worker ships working pixels (from stored previews)
    // in one DocOpened; masters ride along ENCODED and decode only on demand, in parallel;
    // an untouched master passes back into the next save byte-identical. ──
    {
        // The worker already exists, so the Context here is only a formal argument.
        let ctx = egui::Context::default();
        harness.state_mut().open_material(&ctx, "test_hull_777");
        assert!(harness.state().opening.is_some(), "the open runs on the worker, not in the click");
        let opened = wait_until(&mut harness, |s| s.opening.is_none() && s.masters_loading.is_empty() && s.generated_all());
        assert!(opened, "async open did not complete");
        {
            let s = harness.state();
            assert!(s.has_map(MapId::Albedo), "async open restores the albedo");
            assert!(!s.unsaved, "a freshly opened document is not dirty");
            // Lazy masters: the open itself decodes only the ~KB previews. (The harness
            // paints the workspace, whose 2D viewport may have already requested the
            // DISPLAYED map's master on demand — so only the others must still be lazy.)
            let undecoded = MapId::ALL.iter().filter(|id| !s.master_pixels.contains_key(id)).count();
            assert!(undecoded >= 7, "the open must not decode masters eagerly ({undecoded}/8 still encoded)");
            assert_eq!(
                s.master_encoded.get(&MapId::Albedo).map(|m| m.2),
                Some(expected_side),
                "the albedo master rides along encoded at its native side"
            );
        }
        // On-demand decode — what the Export/Refine modals trigger; parallel on the worker.
        harness.state_mut().ensure_masters_decoded(&ctx);
        let decoded = wait_until(&mut harness, |s| {
            s.masters_loading.is_empty() && s.master_pixels.get(&MapId::Albedo).map(|m| m.1) == Some(expected_side)
        });
        assert!(decoded, "on-demand master decode did not complete");
        // Save passthrough: untouched masters write back from their encoded sections
        // (no re-encode); the sync reload below re-validates the written document.
        harness.state_mut().save_document(&ctx, false);
        assert!(wait_until(&mut harness, |s| s.pending_saves == 0), "passthrough save did not complete");
    }

    let mut fresh = AppState::new();
    fresh.settings.library_location = lib_dir.display().to_string();
    fresh.scan_library();
    let scanned = fresh.library.iter().find(|m| m.slug == "test_hull_777").expect("scan finds the material");
    assert_eq!(scanned.coverage, 8, "all 8 maps persisted");
    assert_eq!(scanned.project, "Test Project", "project persisted");
    assert!(scanned.doc_path.is_some(), "scan records the document path for real previews");
    fresh.open_material_sync("test_hull_777");
    assert!(matches!(fresh.screen, Screen::Workspace));
    assert_eq!(fresh.seed, 777);
    assert!(fresh.generated_all(), "loaded document restores all maps");
    assert!(fresh.has_map(MapId::Albedo), "loaded document restores the real albedo");

    // ── 5. Export the full set (what the export modal submits) ──
    use lumagen_core::export::{EngineConvention, ExportJob};
    let mut maps: [Option<Vec<u8>>; 8] = Default::default();
    for id in MapId::ALL {
        maps[id.index()] = fresh.export_pixels(id);
        assert!(maps[id.index()].is_some(), "{id:?} has no export pixels");
    }
    let job = ExportJob {
        maps,
        map_sizes: [0; 8],
        src_size: lumagen::maps::TEX_SIZE,
        out_size: 1024,
        material: fresh.material_name.clone(),
        res: "1k".into(),
        pattern: "$material_$map_$res".into(),
        engine: EngineConvention::Generic,
        directx_normal: false,
        displacement_16bit_png: true,
        displacement_exr: true,
        displacement_height: None,
        seed: fresh.seed,
        prompt: fresh.description.clone(),
        seam_margin: 64,
    };
    let files = lumagen_core::export::export_material(&job, &export_dir).expect("export succeeds");
    assert!(files.len() >= 9, "expected 8 maps + manifest, got {}", files.len());
    let names: Vec<String> = std::fs::read_dir(&export_dir)
        .expect("read export dir")
        .flatten()
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .collect();
    for expected in ["test_hull_777_albedo_1k.png", "test_hull_777_roughness_1k.png", "manifest.json"] {
        assert!(names.iter().any(|n| n == expected), "missing {expected} in {names:?}");
    }

    let _ = std::fs::remove_dir_all(&tmp);
}
