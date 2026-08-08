//! `lumagen-core` — the egui-free core of Lumagen: the data model and prompt templates, the
//! procedural mock renderer, deterministic map derivation (height/normal/AO/opacity), the AI
//! provider layer (fal.ai / OpenRouter / mock), document persistence, and export. Splitting
//! this out of the UI crate makes it unit-testable without a GPU or window.

// Tests are exempt from the deny-unwrap/expect/panic gate: inside `#[cfg(test)]` a failed
// unwrap IS the test failure. `disallowed_methods` is the config-level lint backing the same
// rule and needs the same exemption.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::disallowed_methods))]

pub mod data;
pub mod depth;
pub mod derive;
pub mod document;
pub mod error;
pub mod export;
pub mod generate;
pub mod height;
pub mod library;
pub mod maps;
pub mod promptpack;
pub mod providers;

pub use error::{Error, Result};
