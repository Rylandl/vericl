//! Shared panic-catching helper for reference-twin execution.
//!
//! A panicking reference execution is a reported finding (e.g. an
//! out-of-bounds access a GPU backend would silently clamp via WGSL
//! robustness), not a harness crash. This was originally hand-written once
//! in `conform.rs` as `run_reference`; it now lives here so the
//! macro-generated `conformance_case` (see `vericl-macros`) and `conform.rs`
//! (demo-defects mode) share one implementation instead of two copies that
//! could drift.

use std::panic::{AssertUnwindSafe, catch_unwind};

/// Run `f`, catching a panic without the default hook's stderr noise.
/// Returns `Ok(T)` on success or `Err(message)` with the panic payload
/// converted to a string.
pub fn catch_reference_panic<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    result.map_err(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "reference panicked".to_string())
    })
}
