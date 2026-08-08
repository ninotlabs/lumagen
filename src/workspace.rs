//! Workspace screen (spec §5): top bar, maps panel, viewport, properties, job queue.

use eframe::egui::{self, Color32, Frame, Pos2, Rect, RichText, Sense, Stroke, Ui, Vec2};

use lumagen_core::derive::DerivePath;

use crate::data::{MATERIAL_TYPES, MapId, map_def};
use crate::state::{AppState, MapStatus, PreviewMesh, Screen, ViewMode};
use crate::theme::*;
use crate::widgets::{self, ToastKind};

/// Flag a map's adjust change for the debounced off-thread re-derive.
fn mark_adjust(state: &mut AppState, ui: &Ui, id: MapId) {
    let now = ui.input(|i| i.time);
    state.mark_adjust_dirty(id, now);
}

pub fn show(ui: &mut egui::Ui, state: &mut AppState, render_state: Option<&eframe::egui_wgpu::RenderState>) {
    let ctx = ui.ctx().clone();
    top_bar(ui, state);
    bottom_bar(ui, state);
    left_panel(ui, state);
    right_panel(ui, &ctx, state);
    central(ui, state, render_state);
}

// ── Top bar ──────────────────────────────────────────────────────────────

fn top_bar(ui: &mut Ui, state: &mut AppState) {
    egui::Panel::top("topbar")
        .exact_size(48.0)
        .frame(Frame::new().fill(BG_1()).stroke(Stroke::new(1.0_f32, BORDER)))
        .show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.add_space(8.0);
                if widgets::brand_logo(ui, 22.0).on_hover_text("Back to Library").clicked() {
                    state.screen = Screen::Library;
                }
                ui.add_space(4.0);
                let lib = ui.link(RichText::new("Library").size(13.0).color(TEXT_MID));
                if lib.clicked() {
                    state.screen = Screen::Library;
                }
                if !state.project_name.is_empty() {
                    ui.label(RichText::new("›").color(BORDER_2));
                    let proj = ui.link(RichText::new(state.project_name.clone()).size(13.0).color(TEXT_MID));
                    if proj.clicked() {
                        state.screen = Screen::Library;
                    }
                }
                ui.label(RichText::new("›").color(BORDER_2));
                ui.label(RichText::new(state.material_name.clone()).size(13.0).strong());
                if widgets::icon_button(ui, crate::icon::EDIT, "Edit material — rename, export & generation size").clicked() {
                    state.edit_material_name.clone_from(&state.material_name);
                    state.edit_material_project.clone_from(&state.project_name);
                    state.show_edit_material = true;
                }
                if state.unsaved {
                    let (rect, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
                    ui.painter().circle_filled(rect.center(), 3.0, COST);
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Default);
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(12.0);
                    if widgets::primary_button(ui, &format!("{}  Export", crate::icon::EXPORT)).clicked() {
                        if state.generated_all() && state.has_map(MapId::Albedo) {
                            state.active_step = 3;
                            state.show_export = true;
                        } else {
                            state.toast("Generate the maps first", ToastKind::Cost);
                        }
                    }
                    ui.add_space(8.0);
                    if widgets::icon_button(ui, crate::icon::SETTINGS, "Settings  (Ctrl+,)").clicked() {
                        state.settings.return_to = Screen::Workspace;
                        state.screen = Screen::Settings;
                    }
                    ui.add_space(8.0);
                    let ab = state.active_backend();
                    let short = ab.edit.split_once('/').map(|(_, rest)| rest.to_string()).unwrap_or_else(|| ab.edit.clone());
                    widgets::pill(ui, |ui| {
                        ui.label(crate::icon::text(crate::icon::GENERATE, 11.0, if ab.live { TEXT_MID } else { COST }));
                        ui.label(RichText::new(short).size(12.0).strong());
                    })
                    .on_hover_text(if ab.live {
                        format!("Albedo: {} · Maps: {} (Settings → Providers)", ab.t2i, ab.edit)
                    } else {
                        "No API key set — generation runs the offline renderer. Add a key in Settings → Providers.".into()
                    });

                    // Every step is clickable, so the pipeline can be walked back and forth
                    // (re-roll the albedo after refining, re-open Refine after an export, …).
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        let available = ui.available_width();
                        let stepper_w = 470.0;
                        if available > stepper_w {
                            ui.add_space((available - stepper_w) / 2.0);
                        }
                        if let Some(step) = widgets::flow_stepper(ui, state.active_step) {
                            match step {
                                0 => {
                                    state.active_step = 0;
                                    state.selected = MapId::Albedo;
                                }
                                1 => {
                                    state.active_step = 1;
                                }
                                2 => {
                                    if MapId::ALL.iter().any(|id| state.has_map(*id)) {
                                        state.active_step = 2;
                                        state.refine_selection = state.upscale_targets().into_iter().collect();
                                        state.show_refine = true;
                                    } else {
                                        state.toast("Nothing to refine yet — generate the albedo first", ToastKind::Cost);
                                    }
                                }
                                _ => {
                                    if state.generated_all() && state.has_map(MapId::Albedo) {
                                        state.active_step = 3;
                                        state.show_export = true;
                                    } else {
                                        state.toast("Generate the maps first", ToastKind::Cost);
                                    }
                                }
                            }
                        }
                    });
                });
            });
        });
}

// ── Left panel: Maps ─────────────────────────────────────────────────────

fn left_panel(ui: &mut Ui, state: &mut AppState) {
    egui::Panel::left("maps")
        .exact_size(264.0)
        .resizable(false)
        .frame(Frame::new().fill(BG_1()).stroke(Stroke::new(1.0_f32, BORDER)))
        .show(ui, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.add_space(10.0);
                widgets::section_header(ui, "Material");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(10.0);
                    let add = ui
                        .add(egui::Button::new(crate::icon::text(crate::icon::ADD, 15.0, TEXT_MID)).frame(false))
                        .on_hover_text("New / change albedo");
                    if add.clicked() {
                        state.create.stage = crate::state::CreateStage::Choice;
                        state.show_create = true;
                    }
                });
            });

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(2.0);
                source_tile(ui, state);
                let mut last_group = "";
                for def in &crate::data::MAPS[1..] {
                    if def.group != last_group {
                        last_group = def.group;
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            ui.add_space(8.0);
                            ui.label(RichText::new(def.group.to_uppercase()).size(10.0).color(TEXT_LO).strong());
                        });
                        ui.add_space(2.0);
                    }
                    map_row(ui, state, def.id);
                }
            });

            ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                ui.add_space(4.0);
                Frame::new()
                    .fill(BG_1())
                    .stroke(Stroke::new(1.0_f32, BORDER))
                    .inner_margin(egui::Margin::same(10))
                    .show(ui, |ui| {
                        // The parent is bottom-up; stack the footer's own contents top-down so
                        // the button leads and the fine print follows.
                        ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                            ui.set_width(ui.available_width());
                            let (reqs, _) = state.generate_estimate();
                            let albedo_ready = state.has_map(MapId::Albedo);
                            let label = if state.generating {
                                format!("{} Generating…", crate::icon::GENERATE)
                            } else if !albedo_ready {
                                format!("{} Waiting for albedo…", crate::icon::GENERATE)
                            } else {
                                format!("{} Generate all maps", crate::icon::GENERATE)
                            };
                            let btn = widgets::primary_button(ui, &label);
                            if btn.clicked() && !state.generating {
                                if albedo_ready {
                                    state.show_assist = true;
                                } else {
                                    state.toast("Generate the albedo first — every map derives from it", ToastKind::Cost);
                                }
                            }
                            ui.add_space(4.0);
                            ui.label(
                                RichText::new(if reqs == 0 {
                                    "all maps derived locally".to_string()
                                } else {
                                    format!("{} AI request{}", reqs, if reqs == 1 { "" } else { "s" })
                                })
                                .monospace()
                                .size(11.0)
                                .color(TEXT_MID),
                            );
                            ui.add_space(2.0);
                            let regen = ui.add(egui::Button::new(RichText::new("Regenerate missing only").size(11.0).color(TEXT_LO)).frame(false));
                            if regen.clicked() && !state.generating {
                                state.start_generate_all();
                            }
                            if state.generated_all() {
                                ui.add_space(2.0);
                                let up = ui.add(egui::Button::new(RichText::new("Refine · upscale maps…").size(11.0).color(TEXT_LO)).frame(false));
                                if up.clicked() {
                                    state.refine_selection = state.upscale_targets().into_iter().collect();
                                    state.show_refine = true;
                                }
                            }
                        });
                    });
            });
        });
}

/// Paint the 2px accent bar that marks the selected row/tile.
fn selection_bar(ui: &Ui, rect: Rect) {
    let bar = Rect::from_min_size(rect.left_top(), Vec2::new(2.0, rect.height()));
    ui.painter().rect_filled(bar, 0.0, accent());
}

fn source_tile(ui: &mut Ui, state: &mut AppState) {
    let selected = state.selected == MapId::Albedo;
    let status = state.map(MapId::Albedo).status;
    let stroke = if selected { accent() } else { BORDER };
    let frame = Frame::new()
        .fill(BG_2())
        .stroke(Stroke::new(1.0_f32, stroke))
        .corner_radius(R_CARD)
        .inner_margin(egui::Margin::same(8));
    let resp = frame
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                if let Some(tex) = state.thumbs.get(&MapId::Albedo) {
                    ui.add(egui::Image::new(tex).fit_to_exact_size(Vec2::splat(52.0)).corner_radius(5.0));
                } else {
                    let (rect, _) = ui.allocate_exact_size(Vec2::splat(52.0), Sense::hover());
                    ui.painter().rect_stroke(rect, 5.0, Stroke::new(1.0_f32, BORDER_2), egui::StrokeKind::Inside);
                }
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Albedo").size(13.0).strong());
                        Frame::new()
                            .fill(accent_dim())
                            .corner_radius(3)
                            .inner_margin(egui::Margin::symmetric(5, 2))
                            .show(ui, |ui| {
                                ui.label(RichText::new("SOURCE").size(8.5).color(accent_text()).strong());
                            });
                    });
                    let sub = match status {
                        MapStatus::Generating => "generating…".to_string(),
                        MapStatus::Error => "generation failed".to_string(),
                        s if s.has_image() => format!("seed {} · exports {}", state.seed, export_res_label(state)),
                        _ => "not generated".to_string(),
                    };
                    ui.label(RichText::new(sub).size(11.0).color(if status == MapStatus::Error { DANGER } else { TEXT_LO }));
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    widgets::status_dot(ui, status, ui.input(|i| i.time));
                });
            });
        })
        .response
        .interact(Sense::click());
    if resp.clicked() {
        state.select(MapId::Albedo);
    }
    if selected {
        selection_bar(ui, resp.rect);
    }
}

fn map_row(ui: &mut Ui, state: &mut AppState, id: MapId) {
    let def = map_def(id);
    let status = state.map(id).status;
    let selected = state.selected == id;
    let has = status.has_image();

    let frame = Frame::new()
        .fill(if selected { BG_2() } else { Color32::TRANSPARENT })
        .corner_radius(R_CARD)
        .inner_margin(egui::Margin::symmetric(8, 6));
    let resp = frame
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let (thumb_rect, _) = ui.allocate_exact_size(Vec2::splat(40.0), Sense::hover());
                if has {
                    if let Some(tex) = state.thumbs.get(&id) {
                        ui.painter()
                            .image(tex.id(), thumb_rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                        ui.painter()
                            .rect_stroke(thumb_rect, 5.0, Stroke::new(1.0_f32, BORDER), egui::StrokeKind::Inside);
                    }
                } else {
                    ui.painter()
                        .rect_stroke(thumb_rect, 5.0, Stroke::new(1.0_f32, BORDER_2), egui::StrokeKind::Inside);
                }
                if status == MapStatus::Generating {
                    widgets::status_dot(ui, MapStatus::Generating, ui.input(|i| i.time));
                }

                ui.vertical(|ui| {
                    ui.add_space(2.0);
                    ui.label(RichText::new(def.name).size(12.5));
                    let path = if state.map(id).derive_path == DerivePath::Ai { "AI" } else { "derived" };
                    let sub = if has {
                        format!("{} · {} · {}-bit", path, def.colorspace, def.bits)
                    } else {
                        match status {
                            MapStatus::Queued => format!("{} · queued…", path),
                            MapStatus::Generating => format!("{} · generating…", path),
                            _ => format!("{} · not generated", path),
                        }
                    };
                    ui.label(RichText::new(sub).monospace().size(10.5).color(TEXT_LO));
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    widgets::status_dot(ui, status, ui.input(|i| i.time));
                });
            });
        })
        .response
        .interact(Sense::click());

    if resp.clicked() {
        state.select(id);
    }
    if selected {
        selection_bar(ui, resp.rect);
    }
    // Hover actions. Gate on the pointer being inside the row RECT, not on `hovered()`:
    // the Foreground-order Area below takes pointer priority as soon as the cursor reaches
    // its buttons, which turns `hovered()` false and made the actions flicker/misfire.
    let row_hovered = ui.ctx().pointer_hover_pos().is_some_and(|p| resp.rect.contains(p));
    if row_hovered {
        egui::Area::new(egui::Id::new(("rowbtns", id)))
            .order(egui::Order::Foreground)
            .fixed_pos(resp.rect.right_top() + Vec2::new(-64.0, 8.0))
            .show(ui.ctx(), |ui| {
                Frame::new().fill(BG_2()).corner_radius(4).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .add(egui::Button::new(crate::icon::text(crate::icon::REGENERATE, 13.0, TEXT_MID)).frame(false))
                            .on_hover_text("Re-generate")
                            .clicked()
                        {
                            state.reroll(id);
                        }
                        // Locking asserts "this content is approved" — an empty/queued map
                        // has no content to approve, and Locked would flip has_image() true,
                        // opening the export gate for a procedural stand-in. Hide the toggle
                        // until real pixels exist.
                        if status.has_image() {
                            let lock_icon = if status == MapStatus::Locked {
                                crate::icon::LOCKED
                            } else {
                                crate::icon::UNLOCKED
                            };
                            if ui
                                .add(egui::Button::new(crate::icon::text(lock_icon, 12.0, TEXT_MID)).frame(false))
                                .on_hover_text("Lock / approve")
                                .clicked()
                            {
                                let ms = state.map_mut(id);
                                let locked = ms.status != MapStatus::Locked;
                                ms.status = if locked { MapStatus::Locked } else { MapStatus::Ready };
                                state.toast(if locked { format!("{} locked", def.name) } else { "Unlocked".into() }, ToastKind::Info);
                            }
                        }
                    });
                });
            });
    }
}

// ── Center: viewport ─────────────────────────────────────────────────────

/// Shared cursor-anchored pan/zoom for the 2D and tiled viewports: drag pans; scroll zooms
/// exponentially (a notch is a gentle nudge, a fast flick accelerates) anchored so the
/// point under the cursor stays put; double-click resets. Up to 32× — close pixel
/// inspection of 4K+ masters needs far more than a "fits on screen" ceiling.
fn pan_zoom(ui: &Ui, state: &mut AppState, avail: Rect, resp: &egui::Response) {
    if resp.dragged() {
        state.pan += resp.drag_delta();
    }
    if resp.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            let old_zoom = state.zoom;
            let new_zoom = (old_zoom * (scroll * 0.0018).exp()).clamp(0.25, 32.0);
            if (new_zoom - old_zoom).abs() > f32::EPSILON {
                if let Some(pointer) = resp.hover_pos() {
                    // Keep the point under the cursor stationary across the zoom change.
                    let center = avail.center() + state.pan;
                    let rel = pointer - center;
                    state.pan += rel * (1.0 - new_zoom / old_zoom);
                }
                state.zoom = new_zoom;
            }
        }
    }
    if resp.double_clicked() {
        state.zoom = 1.0;
        state.pan = Vec2::ZERO;
    }
}

fn central(ui: &mut Ui, state: &mut AppState, render_state: Option<&eframe::egui_wgpu::RenderState>) {
    egui::CentralPanel::default().frame(Frame::new().fill(BG_0())).show(ui, |ui| {
        viewport_toolbar(ui, state);
        ui.separator();
        let avail = ui.available_rect_before_wrap();
        // Everything derives from the albedo — until a REAL one exists (generated or
        // imported), every view shows its lifecycle state instead of a stand-in image.
        if !state.has_map(MapId::Albedo) {
            albedo_hero(ui, state, avail);
            return;
        }
        match state.view {
            ViewMode::Map2d => view_2d(ui, state, avail),
            ViewMode::Material3d => view_3d(ui, state, avail, render_state),
            ViewMode::Tiled => view_tiled(ui, state, avail),
            ViewMode::Sheet => view_sheet(ui, state),
        }
    });
}

/// Viewport state for a material whose albedo doesn't exist yet: a genuine generating
/// spinner while the provider call runs, the persistent error + retry when it failed, and a
/// generate call-to-action when nothing has been requested.
fn albedo_hero(ui: &mut Ui, state: &mut AppState, avail: Rect) {
    let status = state.map(MapId::Albedo).status;
    let ab = state.active_backend();
    let model = ab.t2i.clone();
    let rect = Rect::from_center_size(avail.center(), Vec2::new(520.0, 260.0));
    ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
        ui.vertical_centered(|ui| match status {
            MapStatus::Generating => {
                ui.add_space(34.0);
                widgets::status_dot(ui, MapStatus::Generating, ui.input(|i| i.time));
                ui.add_space(10.0);
                ui.label(RichText::new("Generating albedo…").size(16.0).strong());
                ui.add_space(6.0);
                ui.label(
                    RichText::new(format!("via {} · seed {}", model, state.seed))
                        .monospace()
                        .size(11.5)
                        .color(TEXT_MID),
                );
                ui.add_space(4.0);
                ui.label(
                    RichText::new("Large models can take several minutes. The result streams in here — every other map is generated from it afterwards.")
                        .size(12.0)
                        .color(TEXT_LO),
                );
                ui.add_space(10.0);
                if widgets::ghost_button(ui, "Cancel").clicked() {
                    state.cancel_albedo_generation();
                }
            }
            MapStatus::Error => {
                ui.add_space(24.0);
                crate::icon::label(ui, crate::icon::WARNING, 26.0, DANGER);
                ui.add_space(8.0);
                ui.label(RichText::new("Albedo generation failed").size(16.0).strong());
                ui.add_space(6.0);
                if let Some(err) = &state.albedo_error {
                    ui.label(RichText::new(err.clone()).size(11.5).color(TEXT_MID));
                }
                ui.add_space(6.0);
                ui.label(
                    RichText::new("Check the model id and API key in Settings › Providers — if it timed out, raise the job timeout in Settings › Generation.")
                        .size(11.5)
                        .color(TEXT_LO),
                );
                ui.add_space(12.0);
                if widgets::primary_button(ui, &format!("{} Try again", crate::icon::REGENERATE)).clicked() {
                    state.generate_albedo_now();
                }
            }
            _ => {
                ui.add_space(40.0);
                ui.label(RichText::new("No albedo yet").size(16.0).strong());
                ui.add_space(6.0);
                ui.label(
                    RichText::new("Generate the source albedo from the prompt in the right panel — every map derives from it.")
                        .size(12.0)
                        .color(TEXT_MID),
                );
                ui.add_space(12.0);
                if widgets::primary_button(ui, &format!("{} Generate albedo", crate::icon::GENERATE)).clicked() {
                    state.generate_albedo_now();
                }
            }
        });
    });
}

fn viewport_toolbar(ui: &mut Ui, state: &mut AppState) {
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add_space(12.0);
        let options = ["2D map", "3D material", "Tiled", "All maps"];
        let mut sel = match state.view {
            ViewMode::Map2d => 0,
            ViewMode::Material3d => 1,
            ViewMode::Tiled => 2,
            ViewMode::Sheet => 3,
        };
        if widgets::segmented(ui, &options, &mut sel) {
            state.view = match sel {
                0 => ViewMode::Map2d,
                1 => ViewMode::Material3d,
                2 => ViewMode::Tiled,
                _ => ViewMode::Sheet,
            };
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(12.0);
            match state.view {
                ViewMode::Map2d => {
                    ui.label(
                        RichText::new(format!("{}%", (state.zoom * 100.0).round() as i32))
                            .monospace()
                            .size(11.0)
                            .color(TEXT_LO),
                    );
                    if small_tool(ui, &format!("{} Fit", crate::icon::FIT)).clicked() {
                        state.zoom = 1.0;
                        state.pan = Vec2::ZERO;
                    }
                    if small_tool(ui, &format!("{} Before / after", crate::icon::BEFORE_AFTER)).clicked() {
                        state.show_compare = true;
                    }
                }
                ViewMode::Material3d => {
                    if small_tool(ui, &format!("{} Rotate light", crate::icon::ORBIT)).clicked() {
                        state.light_angle += 0.5;
                    }
                    ui.add(
                        egui::DragValue::new(&mut state.tiling3d)
                            .speed(0.05)
                            .range(0.5..=16.0)
                            .prefix("tile ×")
                            .max_decimals(1),
                    )
                    .on_hover_text("UV repeats across the mesh — drag to change");
                    ui.add(
                        egui::DragValue::new(&mut state.normal_strength3d)
                            .speed(0.02)
                            .range(0.0..=4.0)
                            .prefix("normal ×")
                            .max_decimals(2),
                    )
                    .on_hover_text("Normal-map strength in the preview (0 = flat). Export bakes strength via the map's Adjust panel.");
                    ui.add(
                        egui::DragValue::new(&mut state.emission_boost3d)
                            .speed(0.05)
                            .range(0.0..=16.0)
                            .prefix("glow ×")
                            .max_decimals(1),
                    )
                    .on_hover_text("Emission intensity in the preview. Engines scale emissive at import (e.g. Bevy's emissive color/exposure weight).");
                    egui::ComboBox::from_id_salt("hdri")
                        .selected_text(RichText::new(state.hdri.clone()).size(12.0).color(TEXT_MID))
                        .show_ui(ui, |ui| {
                            for h in ["Studio Soft", "Warehouse", "Outdoor Noon", "Night City"] {
                                ui.selectable_value(&mut state.hdri, h.to_string(), h);
                            }
                        });
                    let meshes = ["Sphere", "Plane", "Cube"];
                    let mut m = match state.mesh {
                        PreviewMesh::Sphere => 0,
                        PreviewMesh::Plane => 1,
                        PreviewMesh::Cube => 2,
                    };
                    if widgets::segmented(ui, &meshes, &mut m) {
                        state.mesh = match m {
                            0 => PreviewMesh::Sphere,
                            1 => PreviewMesh::Plane,
                            _ => PreviewMesh::Cube,
                        };
                    }
                }
                ViewMode::Tiled => {
                    ui.label(RichText::new("seam check").monospace().size(11.0).color(TEXT_LO));
                    let tiles = ["1×1", "2×2", "3×3"];
                    let mut t = state.tile_mode;
                    if widgets::segmented(ui, &tiles, &mut t) {
                        state.tile_mode = t;
                    }
                }
                ViewMode::Sheet => {
                    ui.label(RichText::new("8 channels · click a cell to focus").monospace().size(11.0).color(TEXT_LO));
                }
            }
        });
    });
    ui.add_space(4.0);
}

fn small_tool(ui: &mut Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(label).size(12.0).color(TEXT_MID))
            .fill(BG_2())
            .stroke(Stroke::new(1.0_f32, BORDER)),
    )
}

fn view_2d(ui: &mut Ui, state: &mut AppState, avail: Rect) {
    let id = state.selected;
    let def = map_def(id);
    if !state.has_map(id) {
        empty_hero(ui, state, def.name);
        return;
    }
    if id == MapId::Transparency {
        widgets::paint_checkerboard(ui.painter(), avail, 22.0);
    }
    let resp = ui.interact(avail, egui::Id::new("vp2d"), Sense::click_and_drag());
    pan_zoom(ui, state, avail, &resp);

    // Display the native-resolution master when one exists — the 400² working copy is only
    // a preview fallback and reads far softer than the actual generation.
    let ctx = ui.ctx().clone();
    if let Some((tex_id, _side)) = state.display_texture(&ctx, id) {
        let fit = (avail.width().min(avail.height()) * 0.82).min(avail.height() - 32.0);
        let size = fit * state.zoom;
        let center = avail.center() + state.pan;
        let rect = Rect::from_center_size(center, Vec2::splat(size));
        // Clip to the viewport — a panned/zoomed image must never draw over the panels.
        let painter = ui.painter().with_clip_rect(avail);
        painter.rect_filled(rect.translate(Vec2::new(0.0, 12.0)), 4.0, Color32::from_black_alpha(90));
        painter.image(tex_id, rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        painter.rect_stroke(rect, 4.0, Stroke::new(1.0_f32, BORDER), egui::StrokeKind::Inside);
    }
    map_badges(ui, state, avail);
    hint_badge(ui, avail, "drag to pan · scroll to zoom · F to fit");
}

fn view_tiled(ui: &mut Ui, state: &mut AppState, avail: Rect) {
    let id = state.selected;
    let def = map_def(id);
    if !state.has_map(id) {
        empty_hero(ui, state, def.name);
        return;
    }
    if id == MapId::Transparency {
        widgets::paint_checkerboard(ui.painter(), avail, 22.0);
    }
    let n = match state.tile_mode {
        0 => 1,
        1 => 2,
        _ => 3,
    };
    let resp = ui.interact(avail, egui::Id::new("vp_tiled"), Sense::click_and_drag());
    pan_zoom(ui, state, avail, &resp);
    let ctx = ui.ctx().clone();
    if let Some((tex_id, _)) = state.display_texture(&ctx, id) {
        let cell = (avail.width().min(avail.height()) * 0.72 / n as f32).min(360.0) * state.zoom;
        let total = cell * n as f32;
        let origin = avail.center() + state.pan - Vec2::splat(total / 2.0);
        // Clip to the viewport — panned/zoomed tiles must never draw over the panels.
        let painter = ui.painter().with_clip_rect(avail);
        painter.rect_filled(
            Rect::from_min_size(origin, Vec2::splat(total)).translate(Vec2::new(0.0, 10.0)),
            4.0,
            Color32::from_black_alpha(90),
        );
        for r in 0..n {
            for c in 0..n {
                let rect = Rect::from_min_size(origin + Vec2::new(c as f32 * cell, r as f32 * cell), Vec2::splat(cell));
                painter.image(tex_id, rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
            }
        }
        painter.rect_stroke(
            Rect::from_min_size(origin, Vec2::splat(total)),
            4.0,
            Stroke::new(1.0_f32, BORDER),
            egui::StrokeKind::Inside,
        );
    }
    map_badges(ui, state, avail);
    hint_badge(ui, avail, "tiled repetition — drag to pan · scroll to zoom");
}

fn view_3d(ui: &mut Ui, state: &mut AppState, avail: Rect, render_state: Option<&eframe::egui_wgpu::RenderState>) {
    // wgpu PBR preview: render the mesh into an offscreen target via a paint callback, then
    // show that target as an egui::Image. Placeholder when wgpu isn't available (glow backend).
    ui.painter().rect_filled(avail, 0.0, Color32::from_rgb(0x10, 0x12, 0x16));

    let resp = ui.interact(avail, egui::Id::new("vp3d"), Sense::drag());
    if resp.dragged() {
        // The navigation preset (Blender/Maya) sets the drag direction.
        let dir = if state.settings.nav_preset == 1 { -1.0 } else { 1.0 };
        let delta = resp.drag_delta();
        state.orbit_yaw -= delta.x * 0.008 * dir;
        state.orbit_pitch = (state.orbit_pitch + delta.y * 0.008 * dir).clamp(-1.45, 1.45);
    }
    if resp.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            state.orbit_radius = (state.orbit_radius * (-scroll * 0.002).exp()).clamp(1.4, 12.0);
        }
    }
    if resp.double_clicked() {
        state.orbit_yaw = 0.5;
        state.orbit_pitch = 0.3;
        state.orbit_radius = 3.2;
    }
    let angle = state.light_angle;

    let Some(rs) = render_state else {
        ui.painter().text(
            avail.center(),
            egui::Align2::CENTER_CENTER,
            "3D preview needs the wgpu backend",
            egui::FontId::proportional(13.0),
            TEXT_MID,
        );
        return;
    };

    let generated = state.generated_all();
    // Feed the GPU the best pixels it can afford: an already-decoded master up to 4096²
    // uploads once (the callback's per-slot cache keys on Arc identity) and shades the
    // mesh at native detail; bigger or still-encoded masters keep the 400² working copy —
    // the preview never triggers a decode or blows VRAM on 16K buffers.
    const PREVIEW_MAX_NATIVE: usize = 4096;
    let best = |state: &mut AppState, id: MapId| -> (Option<std::sync::Arc<Vec<u8>>>, u32) {
        if let Some((px, side)) = state.master_pixels.get(&id)
            && *side <= PREVIEW_MAX_NATIVE
        {
            return (Some(std::sync::Arc::clone(px)), *side as u32);
        }
        (state.map_rgba(id), crate::maps::TEX_SIZE as u32)
    };
    let (albedo, albedo_side) = best(state, MapId::Albedo);
    let when_generated = |state: &mut AppState, id: MapId| {
        if generated { best(state, id) } else { (None, crate::maps::TEX_SIZE as u32) }
    };
    let (normal, normal_side) = when_generated(state, MapId::Normal);
    let (roughness, rough_side) = when_generated(state, MapId::Roughness);
    let (metallic, metal_side) = when_generated(state, MapId::Metallic);
    let (ao, ao_side) = when_generated(state, MapId::Ao);
    let (emission, emission_side) = when_generated(state, MapId::Emission);
    let map_sides = [albedo_side, normal_side, rough_side, metal_side, ao_side, emission_side];
    let env = crate::preview::EnvPreset::from_hdri(&state.hdri);

    let ppp = ui.ctx().pixels_per_point();
    let phys = [(avail.width() * ppp).max(1.0).round() as u32, (avail.height() * ppp).max(1.0).round() as u32];

    // Register the offscreen target (created/owned by the callback's PreviewResources) as an
    // egui texture, then draw it. Registration happens here, before the callback's prepare runs
    // for this frame, so the callback renders into the same view the image samples.
    let tex_id = {
        let mut renderer = rs.renderer.write();
        crate::preview::register_offscreen(&rs.device, &mut renderer, phys)
    };
    state.preview_tex = Some(tex_id);

    let callback = crate::preview::PreviewCallback {
        mesh: state.mesh,
        angle,
        orbit: [state.orbit_yaw, state.orbit_pitch],
        radius: state.orbit_radius,
        tiling: state.tiling3d,
        normal_strength: state.normal_strength3d,
        emission_boost: state.emission_boost3d,
        has_maps: generated,
        albedo,
        normal,
        roughness,
        metallic,
        ao,
        emission,
        map_sides,
        env,
        phys_size: phys,
    };
    ui.painter().add(eframe::egui_wgpu::Callback::new_paint_callback(avail, callback));
    ui.painter()
        .image(tex_id, avail, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);

    let hint = if generated {
        format!("Full material · {}", state.hdri)
    } else {
        "Albedo only — generate maps to complete".to_string()
    };
    hint_badge(ui, avail, &hint);
    let text = format!("{} {} · drag to orbit · scroll to dolly", crate::icon::MATERIAL_3D, state.mesh.name());
    let galley = ui.painter().layout_no_wrap(text, egui::FontId::monospace(10.5), TEXT_MID);
    let badge_pos = avail.min + Vec2::new(14.0, avail.height() - 34.0);
    let rect2 = Rect::from_min_size(badge_pos, Vec2::new(galley.size().x + 16.0, 22.0));
    ui.painter().rect_filled(rect2, 4.0, Color32::from_black_alpha(160));
    ui.painter().rect_stroke(rect2, 4.0, Stroke::new(1.0_f32, BORDER), egui::StrokeKind::Inside);
    ui.painter().galley(rect2.min + Vec2::new(8.0, 5.0), galley, TEXT_MID);
}

fn view_sheet(ui: &mut Ui, state: &mut AppState) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.add_space(14.0);
        let cols = 4;
        let cell_w = (ui.available_width() - 44.0 - (cols as f32 - 1.0) * 14.0) / cols as f32;
        let time = ui.input(|i| i.time);
        egui::Grid::new("sheet").num_columns(cols).spacing(Vec2::splat(14.0)).show(ui, |ui| {
            for (i, def) in crate::data::MAPS.iter().enumerate() {
                let id = def.id;
                let has = state.has_map(id);
                let status = state.map(id).status;
                let frame = Frame::new().fill(BG_1()).stroke(Stroke::new(1.0_f32, BORDER)).corner_radius(R_CARD);
                let resp = frame
                    .show(ui, |ui| {
                        // Grid cells flow left-to-right — force the image + caption to stack.
                        ui.vertical(|ui| {
                            ui.set_width(cell_w);
                            ui.spacing_mut().item_spacing = Vec2::ZERO;
                            let (img_rect, _) = ui.allocate_exact_size(Vec2::new(cell_w, cell_w), Sense::hover());
                            if has {
                                if id == MapId::Transparency {
                                    widgets::paint_checkerboard(ui.painter(), img_rect, 14.0);
                                }
                                if let Some(tex) = state.textures.get(&id) {
                                    ui.painter()
                                        .image(tex.id(), img_rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                                }
                            } else if status == MapStatus::Generating {
                                ui.painter().text(
                                    img_rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    "…",
                                    egui::FontId::proportional(28.0),
                                    GENERATING.gamma_multiply(0.6 + 0.4 * (time * 5.0).sin() as f32),
                                );
                            } else {
                                ui.painter().text(
                                    img_rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    format!("{} — empty", def.name),
                                    egui::FontId::proportional(11.0),
                                    TEXT_LO,
                                );
                            }
                            let (row, _) = ui.allocate_exact_size(Vec2::new(cell_w, 26.0), Sense::hover());
                            ui.painter().text(
                                Pos2::new(row.min.x + 8.0, row.center().y),
                                egui::Align2::LEFT_CENTER,
                                def.name,
                                egui::FontId::proportional(11.0),
                                TEXT_HI,
                            );
                            ui.painter().text(
                                Pos2::new(row.max.x - 8.0, row.center().y),
                                egui::Align2::RIGHT_CENTER,
                                format!("{} {}b", def.colorspace, def.bits),
                                egui::FontId::monospace(9.5),
                                TEXT_LO,
                            );
                        });
                    })
                    .response
                    .interact(Sense::click());
                if resp.clicked() {
                    state.select(id);
                    state.view = ViewMode::Map2d;
                }
                if resp.hovered() {
                    ui.painter()
                        .rect_stroke(resp.rect, R_CARD, Stroke::new(1.0_f32, accent()), egui::StrokeKind::Inside);
                }
                if (i + 1) % cols == 0 {
                    ui.end_row();
                }
            }
        });
    });
}

fn empty_hero(ui: &mut Ui, state: &mut AppState, name: &str) {
    let avail = ui.available_rect_before_wrap();
    let w = 380.0;
    let h = 200.0;
    let rect = Rect::from_center_size(avail.center(), Vec2::new(w, h));
    ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(8.0);
            ui.label(RichText::new(format!("{} not generated yet", name)).size(16.0).strong());
            ui.add_space(8.0);
            ui.label(
                RichText::new("Every map is derived from the albedo and locked to it, so all channels stay pixel-aligned. Generate the set to fill this in.")
                    .size(12.0)
                    .color(TEXT_MID),
            );
            ui.add_space(14.0);
            if widgets::primary_button(ui, &format!("{} Generate all maps", crate::icon::GENERATE)).clicked() {
                state.show_assist = true;
            }
        });
    });
}

/// The configured export resolution (Settings → Generation → Final size) as a label.
fn export_res_label(state: &AppState) -> &'static str {
    crate::settings::FINAL_SIZE_LABELS.get(state.settings.final_size).copied().unwrap_or("1024²")
}

fn map_badges(ui: &mut Ui, state: &AppState, avail: Rect) {
    let id = state.selected;
    let def = map_def(id);
    let pos = avail.min + Vec2::new(14.0, avail.height() - 34.0);
    let mut x = pos.x;
    let badges: Vec<(String, bool)> = {
        let mut v = vec![(def.name.to_string(), true)];
        match state.master_pixels.get(&id) {
            Some(m) => v.push((format!("{0}×{0} px", m.1), false)),
            None if state.masters_loading.contains(&id) => v.push(("decoding native resolution…".into(), false)),
            None => match state.master_encoded.get(&id) {
                Some((.., side)) => v.push((format!("{0}×{0} px on file", side), false)),
                None => v.push(("400×400 px (preview)".into(), false)),
            },
        }
        v.push((format!("export {}", export_res_label(state)), false));
        v.push((
            if def.bits == 16 {
                format!("{} · 16-bit", def.colorspace)
            } else {
                def.colorspace.to_string()
            },
            false,
        ));
        if id == MapId::Normal {
            let convention = if state.settings.normal_convention == 1 {
                "DirectX (Y−)"
            } else {
                "OpenGL (Y+)"
            };
            v.push((convention.into(), false));
        }
        v
    };
    for (text, strong) in badges {
        let galley = ui
            .painter()
            .layout_no_wrap(text, egui::FontId::monospace(11.0), if strong { TEXT_HI } else { TEXT_MID });
        let w = galley.size().x + 16.0;
        let rect = Rect::from_min_size(Pos2::new(x, pos.y), Vec2::new(w, 22.0));
        ui.painter().rect_filled(rect, 4.0, Color32::from_black_alpha(170));
        ui.painter().rect_stroke(rect, 4.0, Stroke::new(1.0_f32, BORDER), egui::StrokeKind::Inside);
        ui.painter()
            .galley(rect.min + Vec2::new(8.0, 4.0), galley, if strong { TEXT_HI } else { TEXT_MID });
        x += w + 7.0;
    }
}

fn hint_badge(ui: &mut Ui, avail: Rect, text: &str) {
    let galley = ui.painter().layout_no_wrap(text.to_string(), egui::FontId::proportional(11.0), TEXT_LO);
    let w = galley.size().x + 18.0;
    let pos = avail.max - Vec2::new(14.0 + w, avail.height() - 14.0 - 26.0);
    let rect = Rect::from_min_size(pos, Vec2::new(w, 26.0));
    ui.painter().rect_filled(rect, 4.0, Color32::from_black_alpha(150));
    ui.painter().rect_stroke(rect, 4.0, Stroke::new(1.0_f32, BORDER), egui::StrokeKind::Inside);
    ui.painter().galley(rect.min + Vec2::new(9.0, 6.0), galley, TEXT_LO);
}

// ── Right panel: Properties / Generate ───────────────────────────────────

fn right_panel(ui: &mut Ui, ctx: &egui::Context, state: &mut AppState) {
    egui::Panel::right("props")
        .exact_size(312.0)
        .resizable(false)
        .frame(Frame::new().fill(BG_1()).stroke(Stroke::new(1.0_f32, BORDER)))
        .show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(8.0);
                let id = state.selected;
                if id == MapId::Albedo {
                    albedo_props(ui, ctx, state);
                } else if !state.has_map(id) {
                    empty_map_props(ui, state, id);
                } else {
                    derived_map_props(ui, ctx, state, id);
                }
            });
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                Frame::new()
                    .fill(BG_1())
                    .stroke(Stroke::new(1.0_f32, BORDER))
                    .inner_margin(egui::Margin::symmetric(14, 11))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("This map").size(11.5).color(TEXT_MID));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                let id = state.selected;
                                let derived = id != MapId::Albedo && state.map(id).derive_path == DerivePath::Derived;
                                let (path, color) = if id == MapId::Albedo {
                                    ("source", TEXT_LO)
                                } else if derived {
                                    ("derived", READY)
                                } else {
                                    ("AI model", TEXT_MID)
                                };
                                ui.label(RichText::new(path).monospace().size(11.5).color(color));
                            });
                        });
                    });
            });
        });
}

fn albedo_props(ui: &mut Ui, ctx: &egui::Context, state: &mut AppState) {
    let status = state.map(MapId::Albedo).status;
    ui.horizontal(|ui| {
        ui.add_space(14.0);
        widgets::section_header(ui, "Source albedo");
    });
    ui.add_space(6.0);
    if status.has_image() {
        ui.horizontal(|ui| {
            ui.add_space(14.0);
            if let Some(tex) = state.thumbs.get(&MapId::Albedo) {
                ui.add(egui::Image::new(tex).fit_to_exact_size(Vec2::splat(120.0)).corner_radius(6.0));
            }
        });
        ui.add_space(8.0);
        section(ui, |ui| {
            widgets::derive_note(
                ui,
                "This is the anchor. All 7 derived maps are generated from it under a structural lock, guaranteeing pixel alignment across the set.",
            );
        });
    } else if status == MapStatus::Generating {
        let model = state.active_backend().t2i;
        section(ui, |ui| {
            widgets::derive_note(ui, &format!("Generating the albedo via {model}… the result appears here and in the viewport."));
        });
    } else if status == MapStatus::Error {
        section(ui, |ui| {
            Frame::new()
                .fill(BG_2())
                .stroke(Stroke::new(1.0_f32, DANGER))
                .corner_radius(4)
                .inner_margin(egui::Margin::symmetric(10, 9))
                .show(ui, |ui| {
                    ui.label(RichText::new("Generation failed").size(11.5).color(DANGER).strong());
                    if let Some(err) = &state.albedo_error {
                        ui.label(RichText::new(err.clone()).size(11.0).color(TEXT_MID));
                    }
                    ui.label(
                        RichText::new("Check the model id and API key in Settings › Providers.")
                            .size(10.5)
                            .color(TEXT_LO),
                    );
                });
        });
    }
    section(ui, |ui| {
        widgets::section_header(ui, "Generation");
        ui.add_space(8.0);
        widgets::field_label(ui, "Material type", None);
        widgets::chip_row(ui, &MATERIAL_TYPES, &mut state.mat_type);
        ui.add_space(8.0);
        widgets::field_label(ui, "Prompt", None);
        ui.add(
            egui::TextEdit::multiline(&mut state.description)
                .margin(egui::Margin::symmetric(8, 6))
                .desired_rows(3)
                .desired_width(f32::INFINITY)
                .font(egui::TextStyle::Body),
        );
        ui.add_space(6.0);
        widgets::field_label(ui, "Seed", Some(&format!("{} {}", state.seed, crate::icon::LOCKED)));
        ui.add_space(6.0);
        widgets::field_label(ui, "Generation size", None);
        gen_size_combo(ui, "ws_gen_size", &mut state.gen_size);
        ui.add_space(6.0);
        let ab = state.active_backend();
        widgets::field_label(ui, "Model", None);
        match ab.provider {
            crate::state::ActiveProvider::Fal => {
                widgets::model_field(ui, "ws_albedo_model", &mut state.settings.fal_model_t2i, &crate::settings::FAL_T2I_MODELS);
            }
            crate::state::ActiveProvider::OpenRouter => {
                widgets::model_field(ui, "ws_albedo_model", &mut state.settings.or_model, &crate::settings::OR_MODELS);
            }
            crate::state::ActiveProvider::Mock => {
                ui.label(RichText::new("offline mock — add an API key in Settings › Providers").size(11.0).color(TEXT_LO));
            }
        }
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            match status {
                MapStatus::Generating => {
                    widgets::status_dot(ui, MapStatus::Generating, ui.input(|i| i.time));
                    ui.label(RichText::new("Generating…").size(12.0).color(TEXT_MID));
                }
                MapStatus::Error | MapStatus::Empty => {
                    // Same seed retry — the prompt above is editable before trying again.
                    if widgets::primary_button(ui, &format!("{} Generate albedo", crate::icon::GENERATE)).clicked() {
                        state.generate_albedo_now();
                    }
                }
                _ => {
                    // A fresh seed + a real provider run (never the procedural stand-in).
                    if widgets::secondary_button(ui, &format!("{} Re-generate albedo", crate::icon::REGENERATE)).clicked() {
                        state.reroll(MapId::Albedo);
                    }
                }
            }
            if widgets::ghost_button(ui, crate::icon::REPLACE_FILE)
                .on_hover_text("Replace with your own file")
                .clicked()
                && let Some(path) = rfd::FileDialog::new().add_filter("Image", &["png", "jpg", "jpeg", "webp"]).pick_file()
            {
                let epoch = state.material_epoch;
                state.submit_io(
                    ctx,
                    crate::render::IoTask::OverrideMap {
                        id: MapId::Albedo,
                        path,
                        epoch,
                    },
                );
            }
        });
    });
    if status.has_image() {
        upscale_section(ui, state, MapId::Albedo);
    }
    if !state.map(MapId::Albedo).versions.is_empty() {
        history_section(ui, ctx, state, MapId::Albedo);
    }
}

/// The version-history section: chips + restore-on-click.
fn history_section(ui: &mut Ui, ctx: &egui::Context, state: &mut AppState, id: MapId) {
    let versions: Vec<(u32, u32)> = state.map(id).versions.iter().map(|v| (v.n, v.seed)).collect();
    let active = state.map(id).active_version.min(versions.len().saturating_sub(1));
    let mut restore: Option<usize> = None;
    section(ui, |ui| {
        widgets::section_header(ui, "History");
        ui.add_space(6.0);
        restore = version_chips(ui, &versions, active);
    });
    if let Some(i) = restore {
        state.restore_version(ctx, id, i);
    }
}

fn empty_map_props(ui: &mut Ui, state: &mut AppState, id: MapId) {
    let def = map_def(id);
    ui.horizontal(|ui| {
        ui.add_space(14.0);
        widgets::section_header(ui, def.name);
    });
    section(ui, |ui| {
        let chain = if matches!(id, MapId::Ao | MapId::Normal) {
            " (and the displacement chain)"
        } else {
            ""
        };
        widgets::derive_note(
            ui,
            &format!("Not generated yet. {} is derived from the albedo{} and locked to it.", def.name, chain),
        );
    });
    section(ui, |ui| {
        ui.set_width(ui.available_width());
        let derived = state.map(id).derive_path == DerivePath::Derived;
        if !derived {
            maps_model_field(ui, state);
            ui.add_space(8.0);
        }
        let label = if derived {
            format!("{} Derive {}", crate::icon::GENERATE, def.name)
        } else {
            format!("{} Generate {}", crate::icon::GENERATE, def.name)
        };
        if widgets::primary_button(ui, &label).clicked() {
            state.reroll(id);
        }
    });
}

fn derived_map_props(ui: &mut Ui, ctx: &egui::Context, state: &mut AppState, id: MapId) {
    let def = map_def(id);
    let status = state.map(id).status;
    let suffix = match status {
        MapStatus::Overridden => " · your file",
        MapStatus::Locked => " · locked",
        _ => "",
    };
    ui.horizontal(|ui| {
        ui.add_space(14.0);
        widgets::section_header(ui, &format!("{}{}", def.name, suffix));
    });
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.add_space(14.0);
        if let Some(tex) = state.thumbs.get(&id) {
            ui.add(egui::Image::new(tex).fit_to_exact_size(Vec2::splat(120.0)).corner_radius(6.0));
        }
    });

    section(ui, |ui| {
        widgets::section_header(ui, "Generation");
        ui.add_space(8.0);
        let mut fidelity = state.map(id).fidelity;
        widgets::field_label(ui, "Fidelity to albedo", Some(&format!("{}%", (fidelity * 100.0).round() as i32)));
        if ui.add(egui::Slider::new(&mut fidelity, 0.6..=1.0).show_value(false)).changed() {
            state.map_mut(id).fidelity = fidelity;
        }
        slider_legend(ui, "Allow detail", "Keep aligned");
        ui.add_space(4.0);
        widgets::field_label(ui, "Seed", Some(&format!("{} · shared {}", state.seed, crate::icon::LOCKED)));
        if state.map(id).derive_path == DerivePath::Ai {
            ui.add_space(6.0);
            maps_model_field(ui, state);
        }
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if widgets::secondary_button(ui, &format!("{} Re-generate", crate::icon::REGENERATE)).clicked() {
                state.reroll(id);
            }
            if widgets::secondary_button(ui, &format!("{} 4 variations", crate::icon::VARIATIONS)).clicked() {
                state.show_variations = Some(id);
            }
        });
    });

    upscale_section(ui, state, id);

    section(ui, |ui| {
        ui.horizontal(|ui| {
            widgets::section_header(ui, "Path");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new("per map · saved with the material").size(10.0).color(TEXT_LO));
            });
        });
        ui.add_space(6.0);
        let mut ai = state.map(id).derive_path == DerivePath::Ai;
        let path_label = if ai { "AI (generated)" } else { "Derived (computed)" };
        if widgets::toggle_row(ui, &mut ai, path_label).changed() {
            state.map_mut(id).derive_path = if ai { DerivePath::Ai } else { DerivePath::Derived };
            if !ai {
                // Demoted → refresh the preview from the albedo right away (free).
                mark_adjust(state, ui, id);
            }
            state.unsaved = true;
        }
        ui.label(
            RichText::new("AI = generated by the model from the albedo. Derived = computed locally, pixel-aligned by construction. Default for new materials: Settings → Generation.")
                .size(10.0)
                .color(TEXT_LO),
        );
    });

    let derived = state.map(id).derive_path == DerivePath::Derived;

    // Adjust — the sliders re-derive derived-path maps live (debounced, off-thread). An
    // AI-path map's pixels must never be stomped by a local derivation, so it gets no
    // derive sliders.
    if derived {
        section(ui, |ui| {
            ui.horizontal(|ui| {
                widgets::section_header(ui, "Adjust");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new("non-destructive · live").size(10.0).color(TEXT_LO));
                });
            });
            ui.add_space(8.0);
            let mut strength = state.map(id).strength;
            widgets::field_label(ui, "Strength", Some(&format!("{:.2}", strength)));
            if ui.add(egui::Slider::new(&mut strength, 0.0..=2.0).show_value(false)).changed() {
                state.map_mut(id).strength = strength;
                mark_adjust(state, ui, id);
            }
            if id == MapId::Normal {
                let mut flip = state.map(id).flip_y;
                if widgets::toggle_row(ui, &mut flip, "Flip Y (OpenGL / DirectX)").changed() {
                    state.map_mut(id).flip_y = flip;
                    mark_adjust(state, ui, id);
                }
            }
            if id == MapId::Displacement {
                ui.add_space(4.0);
                bit_depth_combo(ui, state, "bitdepth");
            }
            if id == MapId::Ao {
                let mut radius = state.map(id).radius;
                widgets::field_label(ui, "Radius", Some(&format!("{:.2}", radius)));
                if ui.add(egui::Slider::new(&mut radius, 0.0..=1.0).show_value(false)).changed() {
                    state.map_mut(id).radius = radius;
                    mark_adjust(state, ui, id);
                }
            }
            ui.add_space(4.0);
            let mut invert = state.map(id).invert;
            if widgets::toggle_row(ui, &mut invert, "Invert").changed() {
                state.map_mut(id).invert = invert;
                mark_adjust(state, ui, id);
            }
        });
    } else if id == MapId::Displacement {
        // The AI displacement is depth-model-reconstructed: its controls (depth model, near/far,
        // bit depth, macro/micro split) drive the reconstruction and apply on the next
        // Re-generate — they are not live re-derives.
        section(ui, |ui| {
            depth_model_combo(ui, state);
            ui.add_space(6.0);
            bit_depth_combo(ui, state, "bitdepth_ai");
            ui.add_space(6.0);
            let mut invert = state.map(id).invert;
            if widgets::toggle_row(ui, &mut invert, "Invert depth (near ⇄ far)").changed() {
                state.map_mut(id).invert = invert;
            }
            ui.add_space(8.0);
            let mut split = state.map(id).detail_routing;
            widgets::field_label(ui, "Macro / micro split", Some(&format!("{:.0} / {:.0}", split, 100.0 - split)));
            if ui.add(egui::Slider::new(&mut split, 0.0..=100.0).show_value(false)).changed() {
                state.map_mut(MapId::Displacement).detail_routing = split;
                state.map_mut(MapId::Normal).detail_routing = split;
            }
            slider_legend(ui, "Displacement (macro)", "Normal (micro)");
            ui.add_space(6.0);
            widgets::derive_note(
                ui,
                "Depth reconstruction: the depth model estimates macro relief; the Normal map carries micro detail. These apply on the next Re-generate.",
            );
        });
    } else {
        section(ui, |ui| {
            widgets::derive_note(
                ui,
                "This map is on the AI path. Change it with Re-generate or 4 variations above — the adjust sliders apply to derived-path maps only.",
            );
        });
    }

    // Detail routing re-derives both maps live, so it only applies on the derived path.
    if derived && matches!(id, MapId::Normal | MapId::Displacement) {
        section(ui, |ui| {
            widgets::section_header(ui, "Detail routing");
            ui.add_space(8.0);
            let mut split = state.map(id).detail_routing;
            widgets::field_label(ui, "Macro / micro split", Some(&format!("{:.0} / {:.0}", split, 100.0 - split)));
            if ui.add(egui::Slider::new(&mut split, 0.0..=100.0).show_value(false)).changed() {
                state.map_mut(MapId::Normal).detail_routing = split;
                state.map_mut(MapId::Displacement).detail_routing = split;
                mark_adjust(state, ui, MapId::Normal);
                mark_adjust(state, ui, MapId::Displacement);
            }
            slider_legend(ui, "Displacement (macro)", "Normal (micro)");
            ui.add_space(6.0);
            widgets::derive_note(
                ui,
                "Mirrors your prompt doc: displacement owns the largest ~10% of form; normal owns the ~90% micro detail.",
            );
        });
    }

    section(ui, |ui| {
        widgets::section_header(ui, "Override");
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if widgets::secondary_button(ui, &format!("{} Replace file", crate::icon::REPLACE_FILE)).clicked()
                && let Some(path) = rfd::FileDialog::new().add_filter("Image", &["png", "jpg", "jpeg", "webp"]).pick_file()
            {
                let epoch = state.material_epoch;
                state.submit_io(ctx, crate::render::IoTask::OverrideMap { id, path, epoch });
            }
            if widgets::ghost_button(ui, "Reset to AI").clicked() {
                // Undo an override: geometric maps re-derive from the albedo; AI maps re-roll.
                if matches!(id, MapId::Normal | MapId::Ao | MapId::Transparency | MapId::Displacement) {
                    state.map_mut(id).status = MapStatus::Ready;
                    mark_adjust(state, ui, id);
                    state.toast("Reset to derived default", ToastKind::Info);
                } else {
                    state.reroll(id);
                    state.toast("Regenerating from the model…", ToastKind::Info);
                }
            }
        });
    });

    history_section(ui, ctx, state, id);
}

/// Depth-model picker for an AI-path displacement — drives `DepthReconstruct.depth_models`
/// (Blend runs both models best-of-both; or force Marigold / Depth Anything). Applies on the
/// next Re-generate.
fn depth_model_combo(ui: &mut Ui, state: &mut AppState) {
    use crate::state::DepthMode;
    widgets::field_label(ui, "Depth model", None);
    let mut mode = state.map(MapId::Displacement).depth_mode;
    let resp = egui::ComboBox::from_id_salt("depth_model")
        .selected_text(mode.label())
        .show_ui(ui, |ui| {
            for m in [DepthMode::Blend, DepthMode::Marigold, DepthMode::DepthAnything, DepthMode::Local] {
                ui.selectable_value(&mut mode, m, m.label());
            }
        });
    if resp.response.changed() {
        state.map_mut(MapId::Displacement).depth_mode = mode;
    }
}

/// Displacement export bit depth. Only affects export precision — no live re-derive needed.
fn bit_depth_combo(ui: &mut Ui, state: &mut AppState, salt: &str) {
    widgets::field_label(ui, "Bit depth", None);
    let mut b16 = state.map(MapId::Displacement).bit_depth_16;
    let resp = egui::ComboBox::from_id_salt(salt)
        .selected_text(if b16 { "16-bit" } else { "8-bit" })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut b16, false, "8-bit");
            ui.selectable_value(&mut b16, true, "16-bit");
        });
    if resp.response.changed() {
        state.map_mut(MapId::Displacement).bit_depth_16 = b16;
    }
}

/// Version-history chips; clicking an inactive chip returns its index for restore.
/// Fixed-size chips in a wrapped, capped-height scroll area so a long history can't
/// overflow the panel.
fn version_chips(ui: &mut Ui, versions: &[(u32, u32)], active: usize) -> Option<usize> {
    const CHIP: Vec2 = Vec2::new(100.0, 26.0);
    let mut clicked: Option<usize> = None;
    egui::ScrollArea::vertical().max_height(148.0).auto_shrink([false, true]).show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(6.0, 6.0);
            for (i, (n, seed)) in versions.iter().enumerate() {
                let on = i == active;
                let (rect, resp) = ui.allocate_exact_size(CHIP, Sense::click());
                let painter = ui.painter();
                painter.rect_filled(rect, 4.0, BG_2());
                painter.rect_stroke(rect, 4.0, Stroke::new(1.0_f32, if on { accent() } else { BORDER }), egui::StrokeKind::Inside);
                painter.text(
                    Pos2::new(rect.min.x + 9.0, rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    format!("v{}", n),
                    egui::FontId::monospace(11.0),
                    if on { accent_text() } else { TEXT_HI },
                );
                painter.text(
                    Pos2::new(rect.max.x - 9.0, rect.center().y),
                    egui::Align2::RIGHT_CENTER,
                    format!("{}", seed),
                    egui::FontId::monospace(10.0),
                    TEXT_LO,
                );
                if !on {
                    if resp.hovered() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                    }
                    if resp.clicked() {
                        clicked = Some(i);
                    }
                }
            }
        });
    });
    clicked
}

/// Step ③ Refine, per map: the Topaz upscale for AI-path maps, or the free follow-the-albedo
/// re-derive for derived-path maps. Hidden work never replaces good pixels (a failed upscale
/// leaves the map untouched).
fn upscale_section(ui: &mut Ui, state: &mut AppState, id: MapId) {
    if !state.has_map(id) {
        return;
    }
    let src_side = state.master_side(id);
    let topaz = state.refine_via_topaz(id);
    let target = if topaz { state.upscale_target(src_side) } else { state.rederive_target() };
    let model = state.settings.upscale_model.clone();
    section(ui, |ui| {
        ui.horizontal(|ui| {
            widgets::section_header(ui, "Upscale");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let tag = if topaz {
                    format!("Topaz · {model}")
                } else {
                    "derived · follows albedo".to_string()
                };
                ui.label(RichText::new(tag).size(10.0).color(TEXT_LO));
            });
        });
        ui.add_space(6.0);
        if state.upscaling.contains(&id) {
            ui.horizontal(|ui| {
                widgets::status_dot(ui, MapStatus::Generating, ui.input(|i| i.time));
                let verb = if topaz { "Upscaling" } else { "Re-deriving" };
                ui.label(RichText::new(format!("{verb} to {target}²…")).size(12.0).color(TEXT_MID));
            });
        } else if src_side >= target {
            let note = if topaz { "At export size" } else { "Matches the albedo" };
            ui.label(
                RichText::new(format!("{} {note} · {src_side}² native", crate::icon::CONFIRM))
                    .size(12.0)
                    .color(READY),
            );
        } else {
            let (label, note) = if topaz {
                let (passes, _) = state.upscale_cost_estimate(src_side);
                let offline = if state.active_backend().live { "" } else { " · offline resize" };
                let passes_str = if passes > 1 { format!(" ({passes} passes)") } else { String::new() };
                (
                    format!("{} Upscale {src_side}² → {target}²{passes_str}{offline}", crate::icon::GENERATE),
                    "A new version — the current one stays in History, so you can roll back or A/B another Topaz model.",
                )
            } else {
                (
                    format!("{} Re-derive {src_side}² → {target}²", crate::icon::GENERATE),
                    "Recomputed from the full-resolution albedo master — pixel-aligned by construction.",
                )
            };
            if widgets::secondary_button(ui, &label).clicked() {
                let ctx = ui.ctx().clone();
                state.request_refine(&ctx, vec![id]);
            }
            ui.label(RichText::new(note).size(10.0).color(TEXT_LO));
        }
    });
}

/// The image-edit model shared by every AI-path map (one model for all channels — never one
/// per map), editable in place with custom ids.
fn maps_model_field(ui: &mut Ui, state: &mut AppState) {
    let ab = state.active_backend();
    widgets::field_label(ui, "Maps model (shared by all AI maps)", None);
    match ab.provider {
        crate::state::ActiveProvider::Fal => {
            widgets::model_field(ui, "ws_maps_model", &mut state.settings.fal_model_edit, &crate::settings::FAL_EDIT_MODELS);
        }
        crate::state::ActiveProvider::OpenRouter => {
            widgets::model_field(ui, "ws_maps_model", &mut state.settings.or_model, &crate::settings::OR_MODELS);
        }
        crate::state::ActiveProvider::Mock => {
            ui.label(RichText::new("offline mock — add an API key in Settings › Providers").size(11.0).color(TEXT_LO));
        }
    }
}

/// The generation-size selector: what square resolution the model is asked to render.
/// Shared between the workspace albedo panel and the create wizard.
pub fn gen_size_combo(ui: &mut Ui, salt: &str, gen_size: &mut usize) {
    const OPTS: [(usize, &str); 4] = [
        (0, "Max (highest the model supports)"),
        (1024, "1024²"),
        (2048, "2048²"),
        (2880, "2880² (gpt-image max)"),
    ];
    let label = OPTS
        .iter()
        .find(|(v, _)| v == gen_size)
        .map(|(_, l)| (*l).to_string())
        .unwrap_or_else(|| format!("{gen_size}²"));
    egui::ComboBox::from_id_salt(salt)
        .selected_text(RichText::new(label).size(12.0))
        .width(220.0)
        .show_ui(ui, |ui| {
            for (v, l) in OPTS {
                ui.selectable_value(gen_size, v, l);
            }
        });
}

fn slider_legend(ui: &mut Ui, left: &str, right: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(left).size(10.0).color(TEXT_LO));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(right).size(10.0).color(TEXT_LO));
        });
    });
}

fn section(ui: &mut Ui, contents: impl FnOnce(&mut Ui)) {
    Frame::new().inner_margin(egui::Margin::symmetric(14, 12)).stroke(Stroke::NONE).show(ui, |ui| {
        ui.set_width(ui.available_width());
        contents(ui);
    });
    ui.add_space(2.0);
    ui.separator();
    ui.add_space(2.0);
}

// ── Bottom bar: job queue ────────────────────────────────────────────────

fn bottom_bar(ui: &mut Ui, state: &mut AppState) {
    let height = if state.queue_expanded && state.generating { 132.0 } else { 32.0 };
    egui::Panel::bottom("jobbar")
        .exact_size(height)
        .frame(Frame::new().fill(BG_1()).stroke(Stroke::new(1.0_f32, BORDER)))
        .show(ui, |ui| {
            ui.add_space(3.0);
            ui.horizontal(|ui| {
                ui.add_space(12.0);
                let time = ui.input(|i| i.time);
                let (done, total, current) = state.progress();
                if state.generating {
                    widgets::status_dot(ui, MapStatus::Generating, time);
                    let name = current.map(|c| map_def(c).name).unwrap_or("");
                    ui.label(
                        RichText::new(format!("Generating {} of {} — {}", done + 1, total, name))
                            .size(11.5)
                            .color(TEXT_MID),
                    );
                    let (rect, _) = ui.allocate_exact_size(Vec2::new(150.0, 6.0), Sense::hover());
                    ui.painter().rect_filled(rect, 3.0, BG_3());
                    let frac = (state.gen_done as f32 / total.max(1) as f32).clamp(0.0, 1.0);
                    ui.painter()
                        .rect_filled(Rect::from_min_size(rect.min, Vec2::new(rect.width() * frac, 6.0)), 3.0, GENERATING);
                    ui.add_space(10.0);
                    if ui.add(egui::Button::new(RichText::new("Stop").size(10.5).color(DANGER)).frame(false)).clicked() {
                        state.cancel_generation();
                    }
                } else {
                    widgets::status_dot(ui, MapStatus::Ready, time);
                    let label = if state.generated_all() { "Ready · 8 maps aligned" } else { "Ready" };
                    ui.label(RichText::new(label).size(11.5).color(TEXT_MID));
                }

                if !state.upscaling.is_empty() {
                    ui.add_space(14.0);
                    widgets::status_dot(ui, MapStatus::Generating, time);
                    ui.label(
                        RichText::new(format!(
                            "Upscaling {} map{}…",
                            state.upscaling.len(),
                            if state.upscaling.len() == 1 { "" } else { "s" }
                        ))
                        .size(11.5)
                        .color(TEXT_MID),
                    );
                }

                ui.add_space(14.0);
                // Seam score = edge-continuity of the albedo (lower = more seamless); cached
                // so the albedo isn't re-rendered every frame.
                if state.has_map(MapId::Albedo) {
                    let score = state.seam_score();
                    let (seam_label, seam_ok) = if score < 8.0 {
                        (format!("Seams OK · {:.1}", score), true)
                    } else {
                        (format!("Seam edges · {:.1}", score), false)
                    };
                    widgets::status_dot(ui, if seam_ok { MapStatus::Ready } else { MapStatus::Queued }, time);
                    ui.label(RichText::new(seam_label).size(11.5).color(TEXT_MID));
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(12.0);
                    let ts = crate::maps::TEX_SIZE;
                    let tex_bytes = (state.textures.len() + state.lib_previews.len()) * ts * ts * 4
                        + state.thumbs.len() * crate::maps::THUMB_SIZE * crate::maps::THUMB_SIZE * 4;
                    let tex_mb = tex_bytes as f32 / (1024.0 * 1024.0);
                    ui.label(
                        RichText::new(format!("textures {:.0} MB · {}² working", tex_mb, ts))
                            .monospace()
                            .size(11.0)
                            .color(TEXT_LO),
                    );
                    ui.add_space(10.0);
                    let arrow = if state.queue_expanded { crate::icon::EXPAND } else { crate::icon::COLLAPSE };
                    let toggle = ui.add(egui::Button::new(RichText::new(format!("Job queue {}", arrow)).size(11.5).color(TEXT_LO)).frame(false));
                    if toggle.clicked() {
                        state.queue_expanded = !state.queue_expanded;
                    }
                });
            });

            if state.queue_expanded && state.generating {
                ui.add_space(6.0);
                ui.separator();
                ui.add_space(4.0);
                for id in MapId::DERIVED {
                    let status = state.map(id).status;
                    if !matches!(status, MapStatus::Queued | MapStatus::Generating) {
                        continue;
                    }
                    ui.horizontal(|ui| {
                        ui.add_space(16.0);
                        widgets::status_dot(ui, status, ui.input(|i| i.time));
                        ui.label(RichText::new(map_def(id).name).size(11.5).color(TEXT_HI));
                        ui.label(RichText::new(format!("seed {}", state.seed)).monospace().size(10.5).color(TEXT_LO));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add_space(16.0);
                            let cancel = ui.add(egui::Button::new(RichText::new("cancel").size(10.5).color(TEXT_LO)).frame(false));
                            if cancel.clicked() {
                                // Cancels the provider future via its child token, so the
                                // request stops and isn't charged.
                                let ctx = ui.ctx().clone();
                                state.cancel_map(&ctx, id);
                                state.toast(format!("{} cancelled", map_def(id).name), ToastKind::Info);
                            }
                        });
                    });
                }
            }
        });
}
