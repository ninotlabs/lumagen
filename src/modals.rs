//! Modals: intro, new-material wizard, variations, export, Description Check (spec §9).

use eframe::egui::{self, Align, Color32, Frame, Layout, Pos2, Rect, RichText, Sense, Stroke, Ui, Vec2};

use lumagen_core::derive::DerivePath;

use crate::data::{INGREDIENTS, MATERIAL_TYPES, MapId, map_def};
use crate::state::{AppState, CreateStage};
use crate::theme::*;
use crate::widgets::{self, ToastKind};

pub fn show_all(ctx: &egui::Context, state: &mut AppState) {
    if state.show_intro {
        intro(ctx, state);
    }
    if state.show_create {
        create_material(ctx, state);
    }
    if state.show_new_project {
        new_project(ctx, state);
    }
    if state.show_confirm {
        confirm_generate(ctx, state);
    }
    if state.show_variations.is_some() {
        variations(ctx, state);
    }
    if state.show_export {
        export_modal(ctx, state);
    }
    if state.show_assist {
        assist(ctx, state);
    }
    if state.show_compare {
        compare(ctx, state);
    }
    if state.show_refine {
        refine(ctx, state);
    }
    if state.show_edit_material {
        edit_material(ctx, state);
    }
    if state.pending_open.is_some() || state.pending_new.is_some() {
        unsaved_switch(ctx, state);
    }
}

/// The unsaved-changes confirm for material switches (open-from-library / create-new).
/// Same contract as the window-close dialog: no exit path discards work silently.
fn unsaved_switch(ctx: &egui::Context, state: &mut AppState) {
    let mut keep = false;
    let mut save_first = false;
    let mut discard = false;
    egui::Window::new("Unsaved changes")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            ui.label(format!("{} has unsaved changes.", state.material_name));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Keep editing").clicked() {
                    keep = true;
                }
                if ui.button("Save, then switch").clicked() {
                    save_first = true;
                }
                if ui.button("Discard and switch").clicked() {
                    discard = true;
                }
            });
        });
    if keep {
        state.pending_open = None;
        state.pending_new = None;
    } else if save_first || discard {
        if save_first {
            // Queued on the worker; exit/switch is safe because the document path is
            // captured now and the write is drained on app exit.
            state.save_document(ctx, false);
        } else {
            state.unsaved = false;
        }
        state.discard_confirmed = true;
        if let Some(slug) = state.pending_open.take() {
            state.pending_new = None;
            state.open_material(ctx, &slug);
        } else if let Some((name, seed)) = state.pending_new.take() {
            state.new_material_from_create(name, seed);
        }
    }
}

/// Edit material — the wizard's choices aren't locked in: rename, and retarget the export
/// and generation sizes after creation (start at 4K, decide on 8K later).
fn edit_material(ctx: &egui::Context, state: &mut AppState) {
    let mut close = false;
    let mut save = false;
    let dismissed = scrim(ctx, "edit_material", 520.0, |ui| {
        if modal_header(ui, "Edit material") {
            close = true;
        }
        Frame::new().inner_margin(egui::Margin::symmetric(18, 14)).show(ui, |ui| {
            widgets::field_label(ui, "Name", None);
            ui.add(
                egui::TextEdit::singleline(&mut state.edit_material_name)
                    .margin(egui::Margin::symmetric(8, 6))
                    .desired_width(f32::INFINITY),
            );
            ui.label(RichText::new("Renaming moves the saved .lumagen document with it.").size(10.0).color(TEXT_LO));
            ui.add_space(10.0);
            widgets::field_label(ui, "Project", None);
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut state.edit_material_project)
                        .margin(egui::Margin::symmetric(8, 6))
                        .hint_text(RichText::new("No project").color(TEXT_LO))
                        .desired_width(220.0),
                );
                if !state.projects.is_empty() {
                    egui::ComboBox::from_id_salt("edit_project_pick")
                        .selected_text(RichText::new("existing").size(12.0).color(TEXT_MID))
                        .width(110.0)
                        .show_ui(ui, |ui| {
                            for project in &state.projects {
                                let selected = state.edit_material_project == project.name;
                                if ui.selectable_label(selected, project.name.as_str()).clicked() {
                                    state.edit_material_project.clone_from(&project.name);
                                }
                            }
                        });
                }
            });
            ui.label(
                RichText::new("A new name becomes a new project. This is also how existing materials get filed into a project.")
                    .size(10.0)
                    .color(TEXT_LO),
            );
            ui.add_space(10.0);
            widgets::field_label(ui, "Material type", None);
            widgets::chip_row(ui, &MATERIAL_TYPES, &mut state.mat_type);
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    crate::settings::indexed_combo(
                        ui,
                        "edit_export_size",
                        "Export size",
                        &crate::settings::FINAL_SIZE_LABELS,
                        120.0,
                        &mut state.settings.final_size,
                    );
                });
                ui.add_space(16.0);
                ui.vertical(|ui| {
                    widgets::field_label(ui, "Generation size", None);
                    crate::workspace::gen_size_combo(ui, "edit_gen_size", &mut state.gen_size);
                });
            });
            ui.add_space(6.0);
            ui.label(
                RichText::new("Sizes apply from the next generation/export on — existing pixels keep their resolution (Step ③ Refine upscales them).")
                    .size(10.0)
                    .color(TEXT_LO),
            );
        });
        modal_footer(ui, |ui| {
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if widgets::primary_button(ui, "Save changes").clicked() {
                    save = true;
                }
                if widgets::ghost_button(ui, "Cancel").clicked() {
                    close = true;
                }
            });
        });
    });
    if save {
        let slug = slugify(&state.edit_material_name);
        if !slug.is_empty() && slug != state.material_name {
            state.rename_material(slug);
        }
        let project = state.edit_material_project.trim().to_string();
        state.project_name.clone_from(&project);
        if !project.is_empty() && !state.projects.iter().any(|p| p.name == project) {
            state.projects.push(crate::state::LibProject { name: project.clone() });
        }
        let name = state.material_name.clone();
        let res = state.export_res_short().to_string();
        let class = state.mat_type.clone();
        if let Some(entry) = state.library.iter_mut().find(|m| m.slug == name) {
            entry.res = res;
            entry.class = class;
            entry.project = project;
        }
        state.unsaved = true;
        state.show_edit_material = false;
        // Persist immediately — the project lives inside the .lumagen document, so an edit
        // that never hits another autosave must not evaporate on restart.
        state.save_document(ctx, false);
        state.toast("Material updated", ToastKind::Success);
    } else if close || dismissed {
        state.show_edit_material = false;
    }
}

/// Step ③ Refine: pick the upscale target (4K/8K/16K) and Topaz model, tick the maps to
/// upscale, see per-map pass counts and costs, and launch. Each result lands as a NEW
/// version (non-destructive — restore from History to roll back and A/B other models).
fn refine(ctx: &egui::Context, state: &mut AppState) {
    // Kick off decodes for still-encoded masters; masters_decoding_hold gates the footer.
    state.ensure_masters_decoded(ctx);
    let mut close = false;
    let mut launch = false;
    let dismissed = scrim(ctx, "refine", 620.0, |ui| {
        if modal_header(ui, "Refine — upscale (Topaz)") {
            close = true;
        }
        Frame::new().inner_margin(egui::Margin::symmetric(18, 14)).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    widgets::field_label(ui, "Target", None);
                    const TARGETS: [(usize, &str); 3] = [(4096, "4096² (4K)"), (8192, "8192² (8K)"), (16384, "16384² (16K)")];
                    let sel = TARGETS
                        .iter()
                        .find(|(v, _)| *v == state.settings.upscale_target)
                        .map(|(_, l)| *l)
                        .unwrap_or("4096² (4K)");
                    egui::ComboBox::from_id_salt("refine_target").selected_text(sel).width(150.0).show_ui(ui, |ui| {
                        for (v, l) in TARGETS {
                            ui.selectable_value(&mut state.settings.upscale_target, v, l);
                        }
                    });
                });
                ui.add_space(14.0);
                ui.vertical(|ui| {
                    widgets::field_label(ui, "Topaz model", None);
                    egui::ComboBox::from_id_salt("refine_model")
                        .selected_text(state.settings.upscale_model.clone())
                        .width(170.0)
                        .show_ui(ui, |ui| {
                            for m in crate::settings::TOPAZ_MODELS {
                                ui.selectable_value(&mut state.settings.upscale_model, m.to_string(), m);
                            }
                        });
                });
            });
            ui.add_space(4.0);
            ui.label(
                RichText::new("High Fidelity V2 (default), CGI and Standard V2 are precision models (safe for data maps); Wonder/Redefine/Bloom invent detail — best on the albedo. Each result is a new version: restore from History to try another model.")
                    .size(10.5)
                    .color(TEXT_LO),
            );
            ui.add_space(10.0);
            widgets::section_header(ui, "Maps");
            ui.add_space(6.0);
            let candidates: Vec<MapId> = MapId::ALL.iter().copied().filter(|id| state.has_map(*id)).collect();
            if candidates.is_empty() {
                ui.label(RichText::new("No maps to upscale yet — generate the material first.").size(12.0).color(TEXT_LO));
            }
            let ab_live = state.active_backend().live;
            for id in candidates {
                let def = map_def(id);
                let side = state.master_side(id);
                let topaz = state.refine_via_topaz(id);
                let target = if topaz { state.upscale_target(side) } else { state.rederive_target() };
                let at_target = side >= target;
                let busy = state.upscaling.contains(&id);
                ui.horizontal(|ui| {
                    let mut on = state.refine_selection.contains(&id) && !at_target && !busy;
                    let enabled = !at_target && !busy;
                    if ui.add_enabled(enabled, egui::Checkbox::new(&mut on, "")).changed() {
                        if on {
                            state.refine_selection.insert(id);
                        } else {
                            state.refine_selection.remove(&id);
                        }
                    }
                    ui.label(RichText::new(def.name).size(12.5));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let (text, color) = if busy {
                            ("working…".to_string(), GENERATING)
                        } else if at_target {
                            let note = if topaz { "at target" } else { "matches albedo" };
                            (format!("{side}² {} {note}", crate::icon::CONFIRM), READY)
                        } else if topaz {
                            let (passes, _) = state.upscale_cost_estimate(side);
                            let via = if ab_live { "Topaz" } else { "offline resize" };
                            let passes_str = if passes > 1 { format!(" · {passes} passes") } else { String::new() };
                            (format!("{side}² → {target}²{passes_str} · {via}"), TEXT_MID)
                        } else {
                            (format!("{side}² → {target}² · re-derive"), TEXT_MID)
                        };
                        ui.label(RichText::new(text).monospace().size(11.0).color(color));
                    });
                });
                ui.add_space(2.0);
            }
            ui.add_space(4.0);
        });
        // Recompute the footer summary (selection may have just changed).
        let sel = refine_effective_selection(state);
        let ab_live = state.active_backend().live;
        modal_footer(ui, |ui| {
            ui.label(
                RichText::new(format!(
                    "{} map{} selected{}",
                    sel.len(),
                    if sel.len() == 1 { "" } else { "s" },
                    if ab_live { "" } else { " · offline resize" }
                ))
                .monospace()
                .size(12.0)
                .color(TEXT_MID),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let launch_label = format!("{} Refine {} map{}", crate::icon::GENERATE, sel.len(), if sel.len() == 1 { "" } else { "s" });
                if !masters_decoding_hold(ui, state) && !sel.is_empty() && widgets::primary_button(ui, &launch_label).clicked() {
                    launch = true;
                }
                if widgets::ghost_button(ui, "Cancel").clicked() {
                    close = true;
                }
            });
        });
    });
    if launch {
        let sel = refine_effective_selection(state);
        state.show_refine = false;
        let ctx2 = ctx.clone();
        state.request_refine(&ctx2, sel);
    } else if close || dismissed {
        state.show_refine = false;
    }
}

/// Export/Refine read full-resolution masters. While any are still decoding, render the
/// footer hold (label + spinner) in place of the action button and return true — launching
/// early would silently run on the 400² working copies.
fn masters_decoding_hold(ui: &mut Ui, state: &AppState) -> bool {
    let n = state.masters_loading.len();
    if n == 0 {
        return false;
    }
    ui.label(
        RichText::new(format!("Decoding {n} full-resolution map(s)…"))
            .monospace()
            .size(11.5)
            .color(TEXT_MID),
    );
    ui.spinner();
    true
}

/// THE definition of the Refine modal's "selected" set: checked ∧ has content ∧ not busy
/// ∧ not already at target. The checkbox display, the footer count, and the launched set
/// all agree on it — what the modal shows is exactly what launches (and gets billed).
fn refine_effective_selection(state: &AppState) -> Vec<MapId> {
    MapId::ALL
        .iter()
        .copied()
        .filter(|id| state.refine_selection.contains(id) && state.has_map(*id) && !state.upscaling.contains(id))
        .filter(|id| {
            let side = state.master_side(*id);
            let target = if state.refine_via_topaz(*id) {
                state.upscale_target(side)
            } else {
                state.rederive_target()
            };
            side < target
        })
        .collect()
}

/// Dimmed, centered modal. `id` must be unique per modal. Returns true when the user
/// dismissed it by clicking the backdrop or pressing Escape — the caller treats that the
/// same as its close action, so a modal never traps input. Built on `egui::Modal`, whose
/// backdrop only catches clicks *outside* the dialog.
fn scrim(ctx: &egui::Context, id: &'static str, width: f32, contents: impl FnOnce(&mut Ui)) -> bool {
    let modal = egui::Modal::new(egui::Id::new(("modal", id)))
        .backdrop_color(Color32::from_black_alpha(110))
        .frame(
            Frame::new()
                .fill(BG_1())
                .stroke(Stroke::new(1.0_f32, BORDER_2))
                .corner_radius(10)
                .shadow(egui::epaint::Shadow {
                    offset: [0, 24],
                    blur: 80,
                    spread: 0,
                    color: Color32::from_black_alpha(150),
                }),
        )
        .show(ctx, |ui| {
            ui.set_width(width);
            contents(ui);
        });
    modal.should_close()
}

fn modal_header(ui: &mut Ui, title: &str) -> bool {
    let mut close = false;
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add_space(14.0);
        ui.label(RichText::new(title).size(14.0).strong());
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.add_space(14.0);
            if ui
                .add(egui::Button::new(crate::icon::text(crate::icon::CLOSE, 15.0, TEXT_LO)).frame(false))
                .clicked()
            {
                close = true;
            }
        });
    });
    ui.add_space(6.0);
    ui.separator();
    close
}

fn modal_footer(ui: &mut Ui, contents: impl FnOnce(&mut Ui)) {
    ui.separator();
    Frame::new().inner_margin(egui::Margin::symmetric(18, 13)).show(ui, |ui| {
        ui.horizontal(|ui| {
            contents(ui);
        });
    });
}

fn exp_row(ui: &mut Ui, label: &str, value: &str, cost: bool) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(12.0).color(TEXT_MID));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(RichText::new(value).monospace().size(12.0).color(if cost { COST } else { TEXT_HI }));
        });
    });
    ui.add_space(2.0);
    ui.separator();
    ui.add_space(2.0);
}

// ── Intro ────────────────────────────────────────────────────────────────

fn intro(ctx: &egui::Context, state: &mut AppState) {
    let mut close = false;
    let dismissed = scrim(ctx, "intro", 560.0, |ui| {
        ui.add(
            crate::brand::splash()
                .fit_to_exact_size(Vec2::new(560.0, 350.0))
                .corner_radius(egui::CornerRadius { nw: 8, ne: 8, sw: 0, se: 0 }),
        );
        ui.add_space(16.0);
        ui.vertical_centered(|ui| {
            ui.add_space(0.0);
            ui.label(
                RichText::new("Turn one seamless albedo into a full, pixel-aligned PBR map set.\nEvery map locked to the source. Every prompt yours.")
                    .size(12.5)
                    .color(TEXT_MID),
            );
            ui.add_space(14.0);
            ui.label(RichText::new("1–8 switch map · Tab 2D/3D · T tiled").monospace().size(11.0).color(TEXT_MID));
            ui.add_space(4.0);
            ui.label(
                RichText::new("Ctrl+G generate all · Ctrl+E export · Ctrl+S save")
                    .monospace()
                    .size(11.0)
                    .color(TEXT_MID),
            );
            ui.add_space(18.0);
            if widgets::primary_button(ui, "Get started").clicked() {
                close = true;
            }
            ui.add_space(26.0);
        });
    });
    if close || dismissed {
        state.show_intro = false;
    }
}

// ── New material / source albedo ─────────────────────────────────────────

/// Turn a typed material name into a filesystem-safe slug: lowercase, spaces/dashes to
/// underscores, everything else alphanumeric-only.
fn slugify(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.trim().to_lowercase().chars() {
        match c {
            'a'..='z' | '0'..='9' => out.push(c),
            ' ' | '-' | '_' | '.' if !out.ends_with('_') => out.push('_'),
            _ => {}
        }
    }
    out.trim_matches('_').to_string()
}

fn create_material(ctx: &egui::Context, state: &mut AppState) {
    let mut close = false;
    let dismissed = scrim(ctx, "create", 560.0, |ui| {
        if modal_header(ui, "Source albedo") {
            close = true;
        }
        Frame::new()
            .inner_margin(egui::Margin::symmetric(18, 14))
            .show(ui, |ui| match state.create.stage {
                CreateStage::Choice => {
                    ui.horizontal(|ui| {
                        let w = (ui.available_width() - 14.0) / 2.0;
                        for (i, (icon, title, desc, is_gen)) in [
                            (crate::icon::GENERATE, "Generate from prompt", "Create a new seamless albedo with AI", true),
                            (crate::icon::IMPORT, "Import image", "Use an existing seamless texture", false),
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            if i > 0 {
                                ui.add_space(14.0);
                            }
                            let resp = Frame::new()
                                .fill(BG_2())
                                .stroke(Stroke::new(1.0_f32, BORDER))
                                .corner_radius(R_CARD)
                                .inner_margin(egui::Margin::same(16))
                                .show(ui, |ui| {
                                    ui.set_width(w - 34.0);
                                    ui.vertical_centered(|ui| {
                                        ui.label(RichText::new(icon).size(24.0).color(accent()));
                                        ui.add_space(6.0);
                                        ui.label(RichText::new(title).size(13.0).strong());
                                        ui.add_space(4.0);
                                        ui.label(RichText::new(desc).size(11.0).color(TEXT_LO));
                                    });
                                })
                                .response
                                .interact(Sense::click());
                            if resp.hovered() {
                                ui.painter()
                                    .rect_stroke(resp.rect, R_CARD, Stroke::new(1.0_f32, accent()), egui::StrokeKind::Inside);
                            }
                            if resp.clicked() {
                                if is_gen {
                                    state.create.stage = CreateStage::Generate;
                                } else {
                                    state.create.stage = CreateStage::Import;
                                }
                            }
                        }
                    });
                    ui.add_space(8.0);
                }
                CreateStage::Generate => {
                    widgets::field_label(ui, "Material name", None);
                    ui.add(
                        egui::TextEdit::singleline(&mut state.create.name)
                            .margin(egui::Margin::symmetric(8, 6))
                            .hint_text(RichText::new("e.g. hull_panels — empty picks a name from type + seed").color(TEXT_LO))
                            .desired_width(f32::INFINITY),
                    );
                    ui.add_space(10.0);
                    widgets::field_label(ui, "Material type", None);
                    widgets::chip_row(ui, &MATERIAL_TYPES, &mut state.create.mat_type);
                    ui.add_space(10.0);
                    widgets::field_label(ui, "Project", None);
                    egui::ComboBox::from_id_salt("create_project")
                        .selected_text(if state.project_name.is_empty() {
                            "No project".to_string()
                        } else {
                            state.project_name.clone()
                        })
                        .width(220.0)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut state.project_name, String::new(), "No project");
                            for project in &state.projects {
                                ui.selectable_value(&mut state.project_name, project.name.clone(), &project.name);
                            }
                        });
                    ui.add_space(10.0);
                    widgets::field_label(ui, "Prompt", None);
                    ui.add(
                        egui::TextEdit::multiline(&mut state.create.prompt)
                            .margin(egui::Margin::symmetric(8, 6))
                            .desired_rows(3)
                            .desired_width(f32::INFINITY),
                    );
                    ui.add_space(8.0);
                    widgets::field_label(ui, "Negative", None);
                    ui.add(
                        egui::TextEdit::singleline(&mut state.create.negative)
                            .margin(egui::Margin::symmetric(8, 6))
                            .desired_width(f32::INFINITY),
                    );
                    ui.add_space(8.0);
                    widgets::field_label(ui, "Generation size", None);
                    crate::workspace::gen_size_combo(ui, "create_gen_size", &mut state.gen_size);
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            widgets::field_label(ui, "Seed", None);
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::TextEdit::singleline(&mut state.create.seed)
                                        .margin(egui::Margin::symmetric(8, 6))
                                        .font(egui::TextStyle::Monospace)
                                        .desired_width(140.0),
                                );
                                if widgets::ghost_button(ui, crate::icon::RANDOMIZE).on_hover_text("Randomize").clicked() {
                                    let mut rng = crate::maps::Rng::new(9999);
                                    state.create.seed = format!("{}", (rng.next_f32() * 9999.0) as u32);
                                }
                            });
                        });
                        ui.add_space(20.0);
                        ui.vertical(|ui| {
                            widgets::field_label(ui, "Export size", None);
                            let opts = ["1024²", "2048²", "4096²", "8192²", "16384²"];
                            egui::ComboBox::from_id_salt("create_res")
                                .selected_text(opts[state.create.resolution.min(opts.len() - 1)])
                                .show_ui(ui, |ui| {
                                    for (i, o) in opts.iter().enumerate() {
                                        ui.selectable_value(&mut state.create.resolution, i, *o);
                                    }
                                });
                        });
                    });
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("Always seamless / tileable. The model generates at its native size (~1K); Lumagen upscales to the export size.")
                            .size(10.5)
                            .color(TEXT_LO),
                    );
                }
                CreateStage::Import => {
                    ui.vertical_centered(|ui| {
                        ui.add_space(20.0);
                        crate::icon::label(ui, crate::icon::IMPORT, 30.0, accent());
                        ui.add_space(10.0);
                        ui.label(RichText::new("Import a seamless albedo").size(14.0).strong());
                        ui.add_space(6.0);
                        ui.label(
                            RichText::new("PNG or JPEG. It's resized to the working size and becomes the source every map derives from.")
                                .size(11.5)
                                .color(TEXT_LO),
                        );
                        ui.add_space(16.0);
                        if widgets::primary_button(ui, &format!("{} Choose image…", crate::icon::IMPORT)).clicked()
                            && let Some(path) = rfd::FileDialog::new().add_filter("Image", &["png", "jpg", "jpeg"]).pick_file()
                        {
                            // Decode off the UI thread; the result installs via apply_io.
                            let epoch = state.material_epoch;
                            state.submit_io(ctx, crate::render::IoTask::ImportAlbedo { path, epoch });
                            close = true;
                        }
                        ui.add_space(8.0);
                        if widgets::ghost_button(ui, "Back").clicked() {
                            state.create.stage = CreateStage::Choice;
                        }
                        ui.add_space(10.0);
                    });
                }
            });
        if state.create.stage == CreateStage::Generate {
            modal_footer(ui, |ui| {
                ui.label(RichText::new("1 AI request").monospace().size(12.0).color(TEXT_MID));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if widgets::primary_button(ui, &format!("{} Generate albedo", crate::icon::GENERATE)).clicked() {
                        close = true;
                        let seed: u32 = state.create.seed.parse().unwrap_or(4821);
                        state.mat_type.clone_from(&state.create.mat_type);
                        state.description.clone_from(&state.create.prompt);
                        // The user's name (slug-sanitized), falling back to type + seed; made
                        // unique so a repeat name never overwrites an existing material.
                        let typed = slugify(&state.create.name);
                        let mut name = if typed.is_empty() {
                            format!("{}_{}", slugify(&state.create.mat_type), seed)
                        } else {
                            typed
                        };
                        if state.library.iter().any(|m| m.slug == name) {
                            name = format!("{name}_{seed}");
                        }
                        // Wire the wizard's export size into the real export setting
                        // (wizard order 1024/2048/4096/8192/16384 → final_size order
                        // 4096/2048/1024/8192/16384).
                        state.settings.final_size = [2, 1, 0, 3, 4][state.create.resolution.min(4)];
                        state.new_material_from_create(name, seed);
                        state.pending_albedo_gen = true;
                        let ab = state.active_backend();
                        state.toast(
                            if ab.live {
                                format!("Generating albedo via {}…", ab.t2i)
                            } else {
                                "Generating albedo offline (no API key set)…".to_string()
                            },
                            ToastKind::Info,
                        );
                    }
                    if widgets::ghost_button(ui, "Cancel").clicked() {
                        close = true;
                    }
                });
            });
        }
    });
    if close || dismissed {
        state.show_create = false;
    }
}

// ── Before / after compare ───────────────────────────────────────────────

/// Registration compare: the source albedo against the selected map, at native (master)
/// resolution — either a draggable split divider or an opacity overlay, which is the
/// clearest way to spot even sub-pixel misalignment between channels.
fn compare(ctx: &egui::Context, state: &mut AppState) {
    let mut close = false;
    let dismissed = scrim(ctx, "compare", 720.0, |ui| {
        if modal_header(ui, "Compare vs albedo") {
            close = true;
        }
        Frame::new().inner_margin(egui::Margin::symmetric(18, 14)).show(ui, |ui| {
            let after = state.selected;
            let after_name = map_def(after).name;
            // Native-resolution textures (masters when available) — comparing 400² previews
            // hides exactly the subtle misregistration this tool exists to reveal.
            let albedo_tex = state.display_texture(ctx, MapId::Albedo);
            let after_tex = state.display_texture(ctx, after);
            ui.horizontal(|ui| {
                widgets::segmented(ui, &["Slide", "Overlay"], &mut state.compare_mode);
                ui.label(RichText::new(format!("albedo · {after_name}")).size(12.0).color(TEXT_MID));
            });
            ui.add_space(10.0);
            let side = (ui.available_width()).min(560.0);
            let (rect, resp) = ui.allocate_exact_size(Vec2::new(side, side), Sense::click_and_drag());
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            if state.compare_mode == 0 {
                // ── Slide: albedo left of the divider, map right ──
                if resp.dragged()
                    && let Some(pos) = resp.interact_pointer_pos()
                {
                    state.compare_split = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
                }
                let split_x = rect.min.x + rect.width() * state.compare_split;
                if let Some((tex, _)) = albedo_tex {
                    let left = Rect::from_min_max(rect.min, Pos2::new(split_x, rect.max.y));
                    ui.painter().with_clip_rect(left).image(tex, rect, uv, Color32::WHITE);
                }
                if let Some((tex, _)) = after_tex {
                    let right = Rect::from_min_max(Pos2::new(split_x, rect.min.y), rect.max);
                    ui.painter().with_clip_rect(right).image(tex, rect, uv, Color32::WHITE);
                }
                ui.painter()
                    .line_segment([Pos2::new(split_x, rect.min.y), Pos2::new(split_x, rect.max.y)], Stroke::new(2.0_f32, accent()));
                let handle = Rect::from_center_size(Pos2::new(split_x, rect.center().y), Vec2::splat(18.0));
                ui.painter().circle_filled(handle.center(), 9.0, accent());
                ui.painter().circle_stroke(handle.center(), 9.0, Stroke::new(1.5_f32, Color32::WHITE));
                for (text, x, align) in [
                    ("albedo", rect.min.x + 8.0, egui::Align2::LEFT_TOP),
                    (after_name, rect.max.x - 8.0, egui::Align2::RIGHT_TOP),
                ] {
                    let galley = ui.painter().layout_no_wrap(text.to_string(), egui::FontId::monospace(11.0), TEXT_HI);
                    let w = galley.size().x + 12.0;
                    let bx = if align == egui::Align2::LEFT_TOP { x } else { x - w };
                    ui.painter().rect_filled(
                        Rect::from_min_size(Pos2::new(bx, rect.min.y + 8.0), Vec2::new(w, 20.0)),
                        4.0,
                        Color32::from_black_alpha(150),
                    );
                    ui.painter().galley(Pos2::new(bx + 6.0, rect.min.y + 11.0), galley, TEXT_HI);
                }
            } else {
                // ── Overlay: the map blended over the albedo at adjustable opacity ──
                if let Some((tex, _)) = albedo_tex {
                    ui.painter().image(tex, rect, uv, Color32::WHITE);
                }
                if let Some((tex, _)) = after_tex {
                    let a = (state.compare_alpha * 255.0).round().clamp(0.0, 255.0) as u8;
                    ui.painter().image(tex, rect, uv, Color32::from_white_alpha(a));
                }
                if resp.dragged()
                    && let Some(pos) = resp.interact_pointer_pos()
                {
                    state.compare_alpha = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
                }
            }
            ui.painter().rect_stroke(rect, 4.0, Stroke::new(1.0_f32, BORDER), egui::StrokeKind::Inside);
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if state.compare_mode == 0 {
                    ui.label(RichText::new("Split").size(11.5).color(TEXT_LO));
                    ui.add(egui::Slider::new(&mut state.compare_split, 0.0..=1.0).show_value(false));
                } else {
                    ui.label(RichText::new(format!("{} opacity", after_name)).size(11.5).color(TEXT_LO));
                    ui.add(egui::Slider::new(&mut state.compare_alpha, 0.0..=1.0).show_value(false));
                    ui.label(
                        RichText::new("drag on the image to scrub · features must sit exactly on top of each other")
                            .size(10.5)
                            .color(TEXT_LO),
                    );
                }
            });
        });
        modal_footer(ui, |ui| {
            let native = state.master_pixels.get(&state.selected).map(|m| m.1).unwrap_or(crate::maps::TEX_SIZE);
            ui.label(RichText::new(format!("Comparing at native resolution · {native}²")).size(12.0).color(TEXT_MID));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if widgets::ghost_button(ui, "Close").clicked() {
                    close = true;
                }
            });
        });
    });
    if close || dismissed {
        state.show_compare = false;
    }
}

// ── New project ──────────────────────────────────────────────────────────

fn new_project(ctx: &egui::Context, state: &mut AppState) {
    let mut close = false;
    let mut create = false;
    let dismissed = scrim(ctx, "new_project", 420.0, |ui| {
        if modal_header(ui, "New project") {
            close = true;
        }
        Frame::new().inner_margin(egui::Margin::symmetric(18, 14)).show(ui, |ui| {
            ui.label(
                RichText::new("Projects group materials in your library. The new project becomes the current one.")
                    .size(12.0)
                    .color(TEXT_MID),
            );
            ui.add_space(12.0);
            widgets::field_label(ui, "Project name", None);
            let resp = ui.add(
                egui::TextEdit::singleline(&mut state.new_project_name)
                    .margin(egui::Margin::symmetric(8, 6))
                    .hint_text(RichText::new("e.g. Orbital Station").color(TEXT_LO))
                    .desired_width(f32::INFINITY),
            );
            // Enter submits.
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                create = true;
            }
        });
        modal_footer(ui, |ui| {
            ui.label(RichText::new("Shown in the library sidebar").size(12.0).color(TEXT_MID));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if widgets::primary_button(ui, "Create project").clicked() {
                    create = true;
                }
                if widgets::ghost_button(ui, "Cancel").clicked() {
                    close = true;
                }
            });
        });
    });
    if create {
        let name = state.new_project_name.trim().to_string();
        if name.is_empty() {
            state.toast("Give the project a name", ToastKind::Info);
        } else {
            state.create_project(&name);
            state.new_project_name.clear();
            state.show_new_project = false;
        }
    } else if close || dismissed {
        state.new_project_name.clear();
        state.show_new_project = false;
    }
}

// ── Confirm generation ───────────────────────────────────────────────────

fn confirm_generate(ctx: &egui::Context, state: &mut AppState) {
    let mut run = false;
    let mut close = false;
    let dismissed = scrim(ctx, "confirm", 420.0, |ui| {
        if modal_header(ui, "Confirm generation") {
            close = true;
        }
        Frame::new().inner_margin(egui::Margin::symmetric(18, 14)).show(ui, |ui| {
            ui.label(
                RichText::new(format!(
                    "Generate the 7 derived maps from {} albedo at 4096². Every map is locked to the albedo for pixel alignment.",
                    state.material_name
                ))
                .size(12.0)
                .color(TEXT_MID),
            );
            ui.add_space(12.0);
            let (reqs, _) = state.generate_estimate();
            let quality = ["low", "medium", "high", "auto"].get(state.settings.quality).copied().unwrap_or("high");
            exp_row(ui, "AI requests", &format!("{} × 1 image", reqs), false);
            exp_row(ui, "Quality", quality, false);
        });
        modal_footer(ui, |ui| {
            ui.label(RichText::new("Runs on your own API key").size(12.0).color(TEXT_MID));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if widgets::primary_button(ui, "Generate 7 maps").clicked() {
                    run = true;
                }
                if widgets::ghost_button(ui, "Cancel").clicked() {
                    close = true;
                }
            });
        });
    });
    if run {
        state.show_confirm = false;
        state.start_generate_all();
    } else if close || dismissed {
        state.show_confirm = false;
    }
}

// ── Variations ───────────────────────────────────────────────────────────

fn variations(ctx: &egui::Context, state: &mut AppState) {
    let Some(map) = state.show_variations else { return };
    let def = map_def(map);
    let var_key = |k: usize| format!("var_{map:?}_{k}");
    // Kick off the 4-candidate run once per modal open; the candidates land in lib_previews
    // under var_ keys. Guarded by pending_variations + var_pending so the immediate-mode
    // frame loop doesn't resubmit while the modal stays open.
    let already = (0..4).any(|k| state.lib_previews.contains_key(&var_key(k)));
    let in_flight = state.pending_variations == Some(map) || (0..4).any(|k| state.var_pending.contains(&var_key(k)));
    if !already && !in_flight {
        for k in 0..4 {
            state.var_pending.insert(var_key(k));
        }
        state.pending_variations = Some(map);
    }
    let mut promoted: Option<usize> = None;
    let mut close = false;
    let dismissed = scrim(ctx, "variations", 560.0, |ui| {
        if modal_header(ui, &format!("Variations — {}", def.name)) {
            close = true;
        }
        Frame::new().inner_margin(egui::Margin::symmetric(18, 14)).show(ui, |ui| {
            ui.label(
                RichText::new("Four candidates from the same albedo lock. Click one to promote it into the map slot — the other six maps stay untouched.")
                    .size(12.0)
                    .color(TEXT_MID),
            );
            ui.add_space(12.0);
            let w = (ui.available_width() - 10.0) / 2.0;
            egui::Grid::new("var_grid").num_columns(2).spacing(Vec2::splat(10.0)).show(ui, |ui| {
                for k in 0..4 {
                    if let Some(tex) = state.lib_previews.get(&var_key(k)) {
                        let resp = ui.add(egui::Image::new(tex).fit_to_exact_size(Vec2::splat(w)).corner_radius(6.0).sense(Sense::click()));
                        if resp.hovered() {
                            ui.painter()
                                .rect_stroke(resp.rect, 6.0, Stroke::new(2.0_f32, accent()), egui::StrokeKind::Inside);
                        }
                        if resp.clicked() {
                            promoted = Some(k);
                        }
                    }
                    if k % 2 == 1 {
                        ui.end_row();
                    }
                }
            });
        });
        modal_footer(ui, |ui| {
            ui.label(RichText::new("4 AI requests").monospace().size(12.0).color(TEXT_MID));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if widgets::ghost_button(ui, "Close").clicked() {
                    close = true;
                }
            });
        });
    });
    if let Some(k) = promoted {
        state.promote_variation(ctx, map, k);
        state.show_variations = None;
        state.evict_variation_previews();
        state.toast(format!("{} variation promoted", def.name), ToastKind::Cost);
    } else if close || dismissed {
        // Closing cancels the paid in-flight run — nothing may keep billing into a closed
        // UI, and reopening starts a fresh run instead of double-charging beside a zombie.
        if let Some(c) = state.variations_cancel.take() {
            c.cancel();
        }
        state.show_variations = None;
        state.evict_variation_previews();
    }
}

// ── Export ───────────────────────────────────────────────────────────────

fn export_modal(ctx: &egui::Context, state: &mut AppState) {
    // Kick off decodes for still-encoded masters; masters_decoding_hold gates the footer.
    state.ensure_masters_decoded(ctx);
    // Export writes whatever maps exist — a simple texture needn't have the full set.
    let present_maps: Vec<MapId> = MapId::ALL.iter().copied().filter(|id| state.has_map(*id)).collect();
    let present = present_maps.len();
    let plural = if present == 1 { "" } else { "s" };
    let mut close = false;
    let mut confirmed = false;
    let dismissed = scrim(ctx, "export", 560.0, |ui| {
        if modal_header(ui, "Export material") {
            close = true;
        }
        Frame::new().inner_margin(egui::Margin::symmetric(18, 14)).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    crate::settings::indexed_combo(
                        ui,
                        "export_size",
                        "Export size",
                        &crate::settings::FINAL_SIZE_LABELS,
                        120.0,
                        &mut state.settings.final_size,
                    );
                });
                ui.add_space(16.0);
                ui.vertical(|ui| {
                    widgets::field_label(ui, "Source", None);
                    let side = state.master_pixels.get(&MapId::Albedo).map(|m| m.1).unwrap_or(crate::maps::TEX_SIZE);
                    ui.label(RichText::new(format!("albedo {side}² native")).monospace().size(11.5).color(TEXT_MID));
                });
            });
            // Honesty check: exporting above the sources' native resolution is a plain resize,
            // not new detail. Derived maps re-derive at the albedo's native side, so their
            // effective source is the albedo master.
            {
                let out_size = state.export_side();
                let albedo_side = state.master_side(MapId::Albedo);
                let low: Vec<&str> = MapId::ALL
                    .iter()
                    .filter(|id| state.has_map(**id))
                    .filter_map(|id| {
                        let src = if state.master_pixels.contains_key(id) {
                            state.master_side(*id)
                        } else if *id != MapId::Albedo && state.map(*id).derive_path == DerivePath::Derived {
                            albedo_side
                        } else {
                            crate::maps::TEX_SIZE
                        };
                        (src < out_size).then(|| map_def(*id).name)
                    })
                    .collect();
                if !low.is_empty() {
                    ui.add_space(6.0);
                    let names = if low.len() <= 3 { format!(" ({})", low.join(", ")) } else { String::new() };
                    ui.label(
                        RichText::new(format!(
                            "{} {} map{}{names} below {out_size}² — plain resize on export. Refine (Step ③) first for real detail.",
                            crate::icon::WARNING,
                            low.len(),
                            if low.len() == 1 { "" } else { "s" },
                        ))
                        .size(11.0)
                        .color(COST),
                    )
                    .on_hover_text(low.join(", "));
                }
            }
            ui.add_space(12.0);
            let normal = if state.settings.normal_convention == 1 {
                "DirectX (Y−)"
            } else {
                "OpenGL (Y+)"
            };
            exp_row(ui, "Normal convention", normal, false);
            exp_row(ui, "Colorspace", "Albedo sRGB · data Linear", false);
            exp_row(ui, "Format", "PNG per map · EXR displacement", false);
            ui.add_space(8.0);
            widgets::field_label(ui, "Filename pattern", None);
            ui.add(
                egui::TextEdit::singleline(&mut state.export_pattern)
                    .margin(egui::Margin::symmetric(8, 6))
                    .font(egui::TextStyle::Monospace)
                    .desired_width(f32::INFINITY),
            );
            ui.add_space(6.0);
            let preview_res = state.export_res_short().to_lowercase();
            let preview_material = state.material_name.clone();
            let names_preview: Vec<String> = present_maps
                .iter()
                .take(4)
                .map(|id| {
                    let slug = map_def(*id).name.to_lowercase().replace(' ', "_");
                    state
                        .export_pattern
                        .replace("$material", &preview_material)
                        .replace("$map", &slug)
                        .replace("$res", &preview_res)
                        + ".png"
                })
                .collect();
            Frame::new()
                .fill(BG_0())
                .stroke(Stroke::new(1.0_f32, BORDER))
                .corner_radius(4)
                .inner_margin(egui::Margin::same(9))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    for n in &names_preview {
                        ui.label(RichText::new(n).monospace().size(11.5).color(accent()));
                    }
                    let extra = present.saturating_sub(names_preview.len());
                    let more = if extra > 0 { format!("…+{extra} more · ") } else { String::new() };
                    ui.label(
                        RichText::new(format!("{more}contact_sheet.png · manifest.json"))
                            .monospace()
                            .size(11.0)
                            .color(TEXT_LO),
                    );
                });
        });
        modal_footer(ui, |ui| {
            ui.label(
                RichText::new(format!("{present} map file{plural} · contact sheet · manifest.json · README.md"))
                    .size(12.0)
                    .color(TEXT_MID),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if !masters_decoding_hold(ui, state)
                    && widgets::primary_button(ui, &format!("{} Export {present} map{plural}", crate::icon::DOWNLOAD)).clicked()
                {
                    confirmed = true;
                }
            });
        });
    });
    // Only the explicit Export button runs the export — the header ✕, backdrop, and Escape
    // just close.
    if confirmed {
        state.show_export = false;
        run_export(ctx, state);
    } else if close || dismissed {
        state.show_export = false;
    }
}

/// Pick a destination folder and export the map set for real (per-map PNGs + engine packing +
/// manifest), off the UI thread. Every map exports from its highest-resolution source: the
/// model's native master when one exists, the working copy otherwise — and geometric maps
/// re-derive on the worker at the albedo source's full resolution.
fn run_export(ctx: &egui::Context, state: &mut AppState) {
    let Some(dir) = rfd::FileDialog::new().set_title("Export maps to folder").pick_folder() else {
        return;
    };
    use lumagen_core::derive::DerivePath;
    use lumagen_core::export::{EngineConvention, ExportJob};
    let working = crate::maps::TEX_SIZE;

    // The albedo source for worker-side geometric derivation: master if present.
    let albedo_src: Option<(std::sync::Arc<Vec<u8>>, usize)> = match state.master_pixels.get(&MapId::Albedo) {
        Some(m) => Some((std::sync::Arc::clone(&m.0), m.1)),
        None => state.has_map(MapId::Albedo).then(|| (std::sync::Arc::new(state.export_albedo()), working)),
    };

    let mut maps: [Option<Vec<u8>>; 8] = Default::default();
    let mut map_sizes: [usize; 8] = [0; 8];
    let mut derive_missing: Vec<(MapId, lumagen_core::derive::DeriveParams)> = Vec::new();
    for id in MapId::ALL {
        if !state.has_map(id) {
            continue;
        }
        let geometric_derived = id != MapId::Albedo && state.map(id).derive_path == DerivePath::Derived;
        if let Some(master) = state.master_pixels.get(&id) {
            // ExportJob wants owned buffers; the copy runs once per export, off the hot path.
            maps[id.index()] = Some(master.0.as_ref().clone());
            map_sizes[id.index()] = master.1;
        } else if geometric_derived && albedo_src.is_some() {
            derive_missing.push((id, state.map(id).derive_params()));
        } else {
            maps[id.index()] = state.export_pixels(id);
            map_sizes[id.index()] = working;
        }
    }

    let out_size = state.export_side();
    // Engine presets are hidden for now — exports are plain per-map PNGs (Generic
    // convention, EXR displacement); the Settings → Export normal convention still applies.
    let engine = EngineConvention::Generic;
    let displacement_exr = true;
    let directx_normal = state.settings.normal_convention == 1;
    let job = ExportJob {
        maps,
        map_sizes,
        src_size: working,
        out_size,
        material: state.material_name.clone(),
        res: format!("{}k", out_size / 1024),
        pattern: state.export_pattern.clone(),
        engine,
        directx_normal,
        displacement_16bit_png: state.map(MapId::Displacement).bit_depth_16,
        displacement_exr,
        // The f32 height master from the depth-model path exports as TRUE 16-bit/EXR. When the
        // displacement is instead derived on demand, this is None here and the worker
        // (render.rs) fills it from the albedo at native resolution (deriving the f32 height at
        // 4K on the UI thread would freeze it).
        displacement_height: state.master_height.get(&MapId::Displacement).cloned(),
        seed: state.seed,
        prompt: state.description.clone(),
        seam_margin: state.settings.seam_margin.trim().parse::<usize>().unwrap_or(64).min(out_size / 4),
    };
    state.last_export_dir = Some(dir.clone());
    // Off the UI thread (derivation + resize + writing many MB).
    state.submit_io(
        ctx,
        crate::render::IoTask::Export {
            job: Box::new(job),
            dir,
            derive_missing,
            albedo_src,
        },
    );
}

// ── Material Description Check (spec §9) ─────────────────────────────────

/// Append an ingredient's suggested addition to the description as a fresh sentence.
fn append_ingredient(description: &mut String, addition: &str) {
    let trimmed = description.trim_end().to_string();
    *description = format!("{trimmed} {addition}.");
}

fn assist(ctx: &egui::Context, state: &mut AppState) {
    let mut apply_and_generate = false;
    let mut generate_anyway = false;
    let mut close = false;
    let dismissed = scrim(ctx, "assist", 640.0, |ui| {
        if modal_header(ui, "Material Description Check") {
            close = true;
        }
        let (est_reqs, _) = state.generate_estimate();
        // The check can be taller than the window — scroll the body so the header and the
        // footer (with the Generate buttons) always stay reachable. The window auto-sizes, so
        // the scroll area must claim its height explicitly (min == max) or it collapses.
        let body_max_h = (ctx.content_rect().height() - 240.0).max(220.0);
        Frame::new().inner_margin(egui::Margin::symmetric(18, 14)).show(ui, |ui| {
            egui::ScrollArea::vertical()
                .max_height(body_max_h)
                .min_scrolled_height(body_max_h)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    let present = crate::data::detect_ingredients(&state.description);
                    let defined = present.iter().filter(|p| **p).count();
                    let total = present.len();
                    ui.horizontal(|ui| {
                        let w = ui.available_width() - 90.0;
                        let (rect, _) = ui.allocate_exact_size(Vec2::new(w, 8.0), Sense::hover());
                        ui.painter().rect_filled(rect, 4.0, BG_3());
                        let frac = defined as f32 / total as f32;
                        let col = if frac > 0.66 {
                            READY
                        } else if frac > 0.33 {
                            COST
                        } else {
                            DANGER
                        };
                        ui.painter().rect_filled(Rect::from_min_size(rect.min, Vec2::new(w * frac, 8.0)), 4.0, col);
                        ui.label(RichText::new(format!("{}/{} covered", defined, total)).monospace().size(12.0));
                    });
                    ui.add_space(10.0);
                    widgets::derive_note(
                        ui,
                        "Your description drives all 7 maps — fixing gaps here beats regenerating the whole set on a weak description.",
                    );
                    ui.add_space(10.0);

                    widgets::field_label(ui, "Description", None);
                    ui.add(
                        egui::TextEdit::multiline(&mut state.description)
                            .margin(egui::Margin::symmetric(8, 6))
                            .desired_rows(2)
                            .desired_width(f32::INFINITY),
                    );
                    ui.add_space(6.0);
                    if state.settings.llm_key_set && widgets::secondary_button(ui, &format!("{} Enhance with AI", crate::icon::GENERATE)).clicked() {
                        state.pending_enhance = true;
                        state.toast("Asking the model to improve the description…", ToastKind::Info);
                    }
                    ui.add_space(8.0);

                    let missing: Vec<usize> = present.iter().enumerate().filter(|(_, p)| !**p).map(|(i, _)| i).collect();
                    let ok: Vec<usize> = present.iter().enumerate().filter(|(_, p)| **p).map(|(i, _)| i).collect();

                    if !missing.is_empty() {
                        widgets::section_header(ui, &format!("Suggested additions ({})", missing.len()));
                        ui.add_space(6.0);
                        let mut clicked_add: Option<usize> = None;
                        for &i in &missing {
                            let ing = &INGREDIENTS[i];
                            Frame::new()
                                .fill(BG_2())
                                .stroke(Stroke::new(1.0_f32, BORDER))
                                .corner_radius(6)
                                .inner_margin(egui::Margin::same(10))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        crate::icon::label(ui, crate::icon::WARNING, 13.0, COST);
                                        ui.add_space(6.0);
                                        ui.vertical(|ui| {
                                            ui.set_width((ui.available_width() - 80.0).max(60.0));
                                            ui.label(RichText::new(ing.label).size(12.5).strong());
                                            ui.label(RichText::new(format!("{}. Add: “{}.”", ing.reason, ing.addition)).size(11.0).color(TEXT_MID));
                                            ui.label(RichText::new(format!("protects {}", ing.protects)).monospace().size(10.0).color(accent()));
                                        });
                                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                            if widgets::secondary_button(ui, &format!("{} Add", crate::icon::ADD)).clicked() {
                                                clicked_add = Some(i);
                                            }
                                        });
                                    });
                                });
                            ui.add_space(6.0);
                        }
                        if let Some(i) = clicked_add {
                            append_ingredient(&mut state.description, INGREDIENTS[i].addition);
                        }
                        ui.add_space(4.0);
                    }

                    widgets::section_header(ui, &format!("Defined ({})", ok.len()));
                    ui.add_space(6.0);
                    ui.horizontal_wrapped(|ui| {
                        if ok.is_empty() {
                            ui.label(RichText::new("none yet").size(11.0).color(TEXT_LO));
                        }
                        for &i in &ok {
                            widgets::chip(ui, &format!("{} {}", crate::icon::CONFIRM, INGREDIENTS[i].label), true);
                        }
                    });
                    ui.add_space(10.0);

                    widgets::section_header(ui, "Compiled prompt sent to the model");
                    ui.add_space(6.0);
                    let cur = state.assist_map;
                    egui::ComboBox::from_id_salt("assist_map").selected_text(map_def(cur).name).show_ui(ui, |ui| {
                        for id in MapId::DERIVED {
                            ui.selectable_value(&mut state.assist_map, id, map_def(id).name);
                        }
                    });
                    ui.add_space(6.0);
                    Frame::new()
                        .fill(BG_0())
                        .stroke(Stroke::new(1.0_f32, BORDER))
                        .corner_radius(6)
                        .inner_margin(egui::Margin::same(12))
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            if state.map(state.assist_map).derive_path == DerivePath::Derived {
                                ui.label(
                                    RichText::new(format!(
                                        "{} is computed locally from the albedo (aligned, no cost) — no prompt is sent to the model.",
                                        map_def(state.assist_map).name
                                    ))
                                    .size(11.5)
                                    .color(TEXT_MID),
                                );
                            } else {
                                // The same template resolution the request uses, so the preview matches.
                                let tpl = state.settings.map_templates.get(&state.assist_map).cloned().unwrap_or_default();
                                let compiled = crate::data::compile_prompt_with(
                                    &state.settings.base_template,
                                    &state.settings.final_requirement,
                                    &state.description,
                                    &tpl,
                                    state.seed,
                                    "4096",
                                );
                                egui::ScrollArea::vertical().max_height(140.0).id_salt("assist_compiled").show(ui, |ui| {
                                    ui.label(RichText::new(compiled).monospace().size(11.0).color(TEXT_MID));
                                });
                            }
                        });
                });
        });
        let missing_count = crate::data::detect_ingredients(&state.description).iter().filter(|p| !**p).count();
        modal_footer(ui, |ui| {
            ui.label(
                RichText::new(format!("{} AI request{}", est_reqs, if est_reqs == 1 { "" } else { "s" }))
                    .monospace()
                    .size(12.0)
                    .color(TEXT_MID),
            )
            .on_hover_text("Runs on your own API key");
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let apply_label = if missing_count > 0 {
                    format!("Apply {} & Generate", missing_count)
                } else {
                    "Generate 7 maps".to_string()
                };
                if widgets::primary_button(ui, &apply_label).clicked() {
                    apply_and_generate = true;
                }
                if missing_count > 0 && widgets::secondary_button(ui, "Generate anyway").clicked() {
                    generate_anyway = true;
                }
                if widgets::ghost_button(ui, "Cancel").clicked() {
                    close = true;
                }
            });
        });
    });
    if apply_and_generate {
        let present = crate::data::detect_ingredients(&state.description);
        for (i, ing) in INGREDIENTS.iter().enumerate() {
            if !present[i] {
                append_ingredient(&mut state.description, ing.addition);
            }
        }
        state.show_assist = false;
        state.show_confirm = true;
    } else if generate_anyway {
        state.show_assist = false;
        state.show_confirm = true;
    } else if close || dismissed {
        state.show_assist = false;
    }
}
