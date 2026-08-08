//! Reusable design-system widgets (spec §14.2): status dots, coverage pips,
//! chips, stepper, pills, cards, switches, segmented controls, toasts.

use eframe::egui::{self, Color32, Frame, Pos2, Rect, RichText, Sense, Stroke, Ui, Vec2};

use crate::state::MapStatus;
use crate::theme::*;

// ── Status dot ───────────────────────────────────────────────────────────

pub fn status_color(s: MapStatus) -> Color32 {
    match s {
        MapStatus::Empty => BORDER_2,
        MapStatus::Queued => COST,
        MapStatus::Generating => GENERATING,
        MapStatus::Ready => READY,
        MapStatus::Locked => READY,
        MapStatus::Overridden => accent(),
        MapStatus::Error => DANGER,
    }
}

pub fn status_dot(ui: &mut Ui, status: MapStatus, time: f64) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
    let mut color = status_color(status);
    let mut radius = 4.0;
    if status == MapStatus::Generating {
        let pulse = 0.5 + 0.5 * (time * 6.0).sin() as f32;
        color = color.gamma_multiply(0.45 + 0.55 * pulse);
        radius = 3.0 + 1.4 * pulse;
        ui.ctx().request_repaint();
    }
    let center = rect.center();
    if status == MapStatus::Locked {
        ui.painter().circle_stroke(center, 5.4, Stroke::new(1.2_f32, READY.gamma_multiply(0.45)));
    }
    ui.painter().circle_filled(center, radius, color);
}

// ── Coverage pips ────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Pip {
    Present,
    Stale,
    Missing,
}

fn pip_color(pip: Pip) -> Color32 {
    match pip {
        Pip::Present => READY,
        Pip::Stale => COST,
        Pip::Missing => BG_3(),
    }
}

/// Paint coverage pips into an explicit rect (for painter-driven rows, e.g. the library
/// tiles, where widget layout must not run).
pub fn paint_coverage_pips(painter: &egui::Painter, rect: Rect, pips: &[Pip]) {
    let n = pips.len().max(1) as f32;
    let gap = 3.0;
    let w = ((rect.width() - gap * (n - 1.0)) / n).max(2.0);
    for (i, pip) in pips.iter().enumerate() {
        let x = rect.min.x + i as f32 * (w + gap);
        painter.rect_filled(Rect::from_min_size(Pos2::new(x, rect.min.y), Vec2::new(w, rect.height())), 2.0, pip_color(*pip));
    }
}

pub fn coverage_pips(ui: &mut Ui, pips: &[Pip]) {
    let total_w = ui.available_width();
    let n = pips.len().max(1) as f32;
    let gap = 3.0;
    let w = ((total_w - gap * (n - 1.0)) / n).max(4.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = gap;
        for pip in pips {
            let (rect, _) = ui.allocate_exact_size(Vec2::new(w, 4.0), Sense::hover());
            ui.painter().rect_filled(rect, 2.0, pip_color(*pip));
        }
    });
}

// ── Chips ────────────────────────────────────────────────────────────────

/// A selectable chip (pill). Returns true when clicked.
pub fn chip(ui: &mut Ui, label: &str, on: bool) -> bool {
    let text_color = if on { accent_text() } else { TEXT_MID };
    let fill = if on { accent_dim() } else { BG_2() };

    let galley = ui.painter().layout_no_wrap(label.to_owned(), egui::FontId::proportional(11.5), text_color);
    let size = galley.size() + Vec2::new(20.0, 12.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let stroke_color = if on {
        accent()
    } else if resp.hovered() {
        BORDER_2
    } else {
        BORDER
    };
    ui.painter().rect_filled(rect, 20.0, fill);
    ui.painter()
        .rect_stroke(rect, 20.0, Stroke::new(1.0_f32, stroke_color), egui::StrokeKind::Inside);
    ui.painter().galley(rect.min + Vec2::new(10.0, 6.0), galley, text_color);
    resp.clicked()
}

pub fn chip_row(ui: &mut Ui, items: &[&str], selected: &mut String) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.y = 6.0;
        // egui 0.34: item_spacing.x no longer separates siblings in a wrapped row — add it
        // explicitly between chips.
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                ui.add_space(6.0);
            }
            if chip(ui, item, selected == item) {
                *selected = (*item).to_string();
            }
        }
    });
}

// ── Buttons ──────────────────────────────────────────────────────────────

/// Primary accent button.
pub fn primary_button(ui: &mut Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(label).color(Color32::WHITE).strong())
            .fill(accent())
            .stroke(Stroke::new(1.0_f32, accent())),
    )
}

/// Secondary (bordered) button.
pub fn secondary_button(ui: &mut Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(label).color(TEXT_HI))
            .fill(BG_2())
            .stroke(Stroke::new(1.0_f32, BORDER_2)),
    )
}

/// Ghost (borderless) button.
pub fn ghost_button(ui: &mut Ui, label: &str) -> egui::Response {
    ui.add(egui::Button::new(RichText::new(label).color(TEXT_MID)).frame(false))
}

/// Small square icon button.
pub fn icon_button(ui: &mut Ui, label: &str, tip: &str) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(label).color(TEXT_MID).size(14.0))
            .fill(BG_2())
            .stroke(Stroke::new(1.0_f32, BORDER))
            .min_size(Vec2::splat(30.0)),
    )
    .on_hover_text(tip)
}

// ── Pill (top-bar info chip) ─────────────────────────────────────────────

pub fn pill(ui: &mut Ui, contents: impl FnOnce(&mut Ui)) -> egui::Response {
    Frame::new()
        .fill(BG_2())
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(R_CTL)
        .inner_margin(egui::Margin::symmetric(10, 6))
        .show(ui, |ui| {
            // Force left-to-right inside the pill so content order is stable even when the
            // pill sits in a right-to-left top-bar layout.
            ui.horizontal(|ui| contents(ui));
        })
        .response
}

// ── Segmented control ────────────────────────────────────────────────────

pub fn segmented(ui: &mut Ui, options: &[&str], selected: &mut usize) -> bool {
    let mut changed = false;
    Frame::new()
        .fill(BG_2())
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(R_CTL)
        .inner_margin(egui::Margin::same(2))
        .show(ui, |ui| {
            // egui 0.34 lays horizontal rows out on a grid where item_spacing.x no longer
            // separates siblings — space the segments explicitly instead.
            for (i, opt) in options.iter().enumerate() {
                if i > 0 {
                    ui.add_space(2.0);
                }
                let on = *selected == i;
                let btn = if on {
                    egui::Button::new(RichText::new(*opt).size(12.0).color(Color32::WHITE))
                        .fill(accent())
                        .corner_radius(3)
                } else {
                    egui::Button::new(RichText::new(*opt).size(12.0).color(TEXT_MID)).frame(false).corner_radius(3)
                };
                if ui.add(btn).clicked() && !on {
                    *selected = i;
                    changed = true;
                }
            }
        });
    changed
}

// ── Toggle switch ────────────────────────────────────────────────────────

pub fn switch(ui: &mut Ui, on: &mut bool) -> egui::Response {
    let (rect, mut resp) = ui.allocate_exact_size(Vec2::new(32.0, 18.0), Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    let fill = if *on { accent() } else { BG_3() };
    ui.painter().rect_filled(rect, 10.0, fill);
    let x = if *on { rect.max.x - 9.0 } else { rect.min.x + 9.0 };
    ui.painter().circle_filled(Pos2::new(x, rect.center().y), 7.0, Color32::WHITE);
    resp
}

pub fn toggle_row(ui: &mut Ui, on: &mut bool, label: &str) -> egui::Response {
    // Return the switch's response (not the row container's) so `.changed()` propagates;
    // the label is a click target too.
    ui.horizontal(|ui| {
        let mut resp = switch(ui, on);
        let text = ui.add(egui::Label::new(RichText::new(label).size(12.0).color(TEXT_MID)).sense(Sense::click()));
        if text.clicked() {
            *on = !*on;
            resp.mark_changed();
        }
        resp
    })
    .inner
}

// ── Model field ──────────────────────────────────────────────────────────

/// An editable model-id field: free-text entry (any id works) plus a suggestions dropdown
/// that fills it. Adapts to narrow panels by stacking the dropdown under the editor.
pub fn model_field(ui: &mut Ui, salt: &str, value: &mut String, suggestions: &[(&str, &str)]) {
    let narrow = ui.available_width() < 420.0;
    let editor = |ui: &mut Ui, value: &mut String, width: f32| {
        ui.add(
            egui::TextEdit::singleline(value)
                .margin(egui::Margin::symmetric(8, 6))
                .font(egui::TextStyle::Monospace)
                .desired_width(width),
        );
    };
    let combo = |ui: &mut Ui, value: &mut String| {
        egui::ComboBox::from_id_salt(salt.to_owned())
            .selected_text(RichText::new("suggestions").size(12.0).color(TEXT_MID))
            .width(130.0)
            .show_ui(ui, |ui| {
                for (slug, name) in suggestions {
                    let row = format!("{slug}   ·  {name}");
                    if ui.selectable_label(value == slug, RichText::new(row).monospace().size(11.5)).clicked() {
                        *value = (*slug).to_string();
                    }
                }
            });
    };
    if narrow {
        editor(ui, value, ui.available_width());
        combo(ui, value);
    } else {
        ui.horizontal(|ui| {
            let w = (ui.available_width() - 150.0).clamp(140.0, 380.0);
            editor(ui, value, w);
            combo(ui, value);
        });
    }
}

// ── Panels & sections ────────────────────────────────────────────────────

/// Uppercase section header like the mockup's `.panel-h` / `.sec h4`.
pub fn section_header(ui: &mut Ui, label: &str) {
    ui.label(RichText::new(label.to_uppercase()).size(10.5).color(TEXT_LO).strong());
}

/// Bordered card frame. Fills the available column width so stacked cards line up.
pub fn card(ui: &mut Ui, contents: impl FnOnce(&mut Ui)) -> egui::InnerResponse<()> {
    Frame::new()
        .fill(BG_1())
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(8)
        .inner_margin(egui::Margin::same(16))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            contents(ui);
        })
}

/// "Derive note" — accent-left info callout.
pub fn derive_note(ui: &mut Ui, text: &str) {
    // Wider left margin makes room for the accent bar, which is painted left of the text —
    // the label stays a direct child of a vertical layout so it wraps at the panel width.
    Frame::new()
        .fill(BG_2())
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(4)
        .inner_margin(egui::Margin {
            left: 18,
            right: 10,
            top: 9,
            bottom: 9,
        })
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(11.0).color(TEXT_LO));
            let inner = ui.min_rect();
            let bar = Rect::from_min_size(Pos2::new(inner.min.x - 10.0, inner.min.y), Vec2::new(2.0, inner.height()));
            ui.painter().rect_filled(bar, 1.0, accent());
        });
}

// ── Field label with optional mono value ─────────────────────────────────

pub fn field_label(ui: &mut Ui, label: &str, value: Option<&str>) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(11.5).color(TEXT_MID));
        if let Some(v) = value {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(v).monospace().size(11.5).color(TEXT_HI));
            });
        }
    });
}

// ── Stepper (top-bar flow ①–④) ───────────────────────────────────────────

/// The ①②③④ pipeline stepper. Every step is a click target — the workflow is free to move
/// back and forth, not a one-way street. Returns the step clicked this frame, if any.
pub fn flow_stepper(ui: &mut Ui, active: usize) -> Option<usize> {
    let steps = ["Albedo", "Generate", "Refine", "Export"];
    let mut clicked = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        for (i, name) in steps.iter().enumerate() {
            let done = i < active;
            let current = i == active;
            let text_color = if current {
                accent_text()
            } else if done {
                READY
            } else {
                TEXT_LO
            };
            let (rect, resp) = ui.allocate_exact_size(Vec2::new(96.0, 26.0), Sense::click());
            if resp.clicked() {
                clicked = Some(i);
            }
            let fill = if current {
                accent_dim()
            } else if resp.hovered() {
                BG_3()
            } else {
                Color32::TRANSPARENT
            };
            let _ = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
            ui.painter().rect_filled(rect, 20.0, fill);
            let c = Pos2::new(rect.min.x + 16.0, rect.center().y);
            // Phosphor CHECK — U+2713 is missing from egui's default fonts and drew a tofu box.
            let (num, nfill) = if done {
                (crate::icon::CONFIRM.to_string(), Color32::TRANSPARENT)
            } else if current {
                ((i + 1).to_string(), accent())
            } else {
                ((i + 1).to_string(), Color32::TRANSPARENT)
            };
            ui.painter().circle_filled(c, 8.5, nfill);
            ui.painter()
                .circle_stroke(c, 8.5, Stroke::new(1.2_f32, if current { accent() } else { text_color }));
            ui.painter().text(
                c,
                egui::Align2::CENTER_CENTER,
                num,
                egui::FontId::proportional(9.5),
                if current { Color32::WHITE } else { text_color },
            );
            ui.painter().text(
                Pos2::new(rect.min.x + 30.0, rect.center().y),
                egui::Align2::LEFT_CENTER,
                *name,
                egui::FontId::proportional(12.0),
                text_color,
            );
            if i < steps.len() - 1 {
                // "›" is in egui's default font; "→" is not and rendered as a tofu box.
                ui.label(RichText::new("›").size(12.0).color(BORDER_2));
            }
        }
    });
    clicked
}

// ── Toasts ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ToastKind {
    Info,
    Cost,
    Success,
    /// A failure (provider, I/O, save, export) — shown in the danger color.
    Error,
}

pub struct Toast {
    pub message: String,
    pub kind: ToastKind,
    pub created: f64,
}

pub struct Toasts {
    pub items: Vec<Toast>,
}

impl Default for Toasts {
    fn default() -> Self {
        Self::new()
    }
}

impl Toasts {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }
    pub fn push(&mut self, message: impl Into<String>, kind: ToastKind) {
        self.items.push(Toast {
            message: message.into(),
            kind,
            created: now(),
        });
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        let t = now();
        self.items.retain(|toast| t - toast.created < 3.0);
        if self.items.is_empty() {
            return;
        }
        egui::Area::new(egui::Id::new("toasts"))
            .order(egui::Order::Foreground)
            .anchor(egui::Align2::RIGHT_BOTTOM, Vec2::new(-18.0, -44.0))
            .show(ctx, |ui| {
                for toast in &self.items {
                    let age = (t - toast.created) as f32;
                    let fade = ((3.0 - age) / 0.4).clamp(0.0, 1.0).min(age / 0.15).clamp(0.0, 1.0);
                    let bar = match toast.kind {
                        ToastKind::Info => accent(),
                        ToastKind::Cost => COST,
                        ToastKind::Success => READY,
                        ToastKind::Error => DANGER,
                    };
                    Frame::new()
                        .fill(BG_2().gamma_multiply(0.97))
                        .stroke(Stroke::new(1.0_f32, BORDER_2))
                        .corner_radius(6)
                        .inner_margin(egui::Margin::symmetric(14, 11))
                        .show(ui, |ui| {
                            ui.scope(|ui| {
                                ui.set_min_width(220.0);
                                ui.set_opacity(fade);
                                ui.horizontal(|ui| {
                                    let (rect, _) = ui.allocate_exact_size(Vec2::new(3.0, 20.0), Sense::hover());
                                    ui.painter().rect_filled(rect, 1.5, bar);
                                    ui.add_space(6.0);
                                    ui.label(RichText::new(&toast.message).size(12.0).color(TEXT_HI));
                                });
                            });
                        });
                    ui.add_space(8.0);
                }
            });
        ctx.request_repaint();
    }
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ── Brand logo ───────────────────────────────────────────────────────────

pub fn brand_logo(ui: &mut Ui, size: f32) -> egui::Response {
    // The real four-quadrant brand mark (embedded PNG, cached by the bytes loader).
    ui.add(crate::brand::mark(size).sense(Sense::click()))
}

// ── Checkerboard painter ─────────────────────────────────────────────────

pub fn paint_checkerboard(painter: &egui::Painter, rect: Rect, cell: f32) {
    painter.rect_filled(rect, 0.0, CHECKER_B);
    if cell <= 0.0 {
        return;
    }
    // Only tessellate cells that fall inside the painter's clip region. A background drawn over
    // a very large rect (a native-resolution or zoomed image) would otherwise emit one rect per
    // cell across the whole area — millions of rects that overflow egui's per-frame index buffer
    // and hard-crash the wgpu backend. The base fill above already covers the full rect, so the
    // clipped cells are all that's ever visible. Indices stay aligned to `rect`'s origin so the
    // pattern doesn't swim as the clip changes.
    let vis = rect.intersect(painter.clip_rect());
    if !vis.is_positive() {
        return;
    }
    let col0 = (((vis.min.x - rect.min.x) / cell).floor() as i32).max(0);
    let row0 = (((vis.min.y - rect.min.y) / cell).floor() as i32).max(0);
    let col1 = ((vis.max.x - rect.min.x) / cell).ceil() as i32 + 1;
    let row1 = ((vis.max.y - rect.min.y) / cell).ceil() as i32 + 1;
    for r in row0..row1 {
        for c in col0..col1 {
            if (r + c) % 2 == 0 {
                let rct = Rect::from_min_size(rect.min + Vec2::new(c as f32 * cell, r as f32 * cell), Vec2::splat(cell)).intersect(rect);
                painter.rect_filled(rct, 0.0, CHECKER_A);
            }
        }
    }
}
