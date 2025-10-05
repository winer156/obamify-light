#![warn(clippy::all, rust_2018_idioms)]

mod app;
pub use app::ObamifyApp;
#[cfg(target_arch = "wasm32")]
pub use app::worker_entry;
