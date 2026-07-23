//! Documents — as an executable, empirical test — the f64/wgpu soundness
//! landmine that dictates why `axpy_f64`'s conformance lane is cubecl-cpu and
//! never wgpu.
//!
//! WGSL has no f64. cubecl 0.10 nonetheless compiles and launches an f64
//! kernel on the wgpu/Metal backend with **no compile error and no panic** —
//! and then produces silently *wrong* results (not even an f32 demotion:
//! genuine garbage, because the host uploads 8-byte f64 elements into a buffer
//! the WGSL kernel indexes as if it held a different element type). The
//! macro-derived f64 twin computes correctly, so a differential run against
//! wgpu MUST diverge. This test pins that: if it ever *stops* diverging, the
//! platform assumption behind the cpu-only f64 lane has changed and must be
//! re-examined.
#![cfg(feature = "wgpu")]

use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;
use vericl_examples::axpy_f64_vericl;

#[test]
fn f64_axpy_silently_diverges_on_wgpu() {
    let client = WgpuRuntime::client(&Default::default());
    let sizes = [7usize, 256, 1000, 1027];

    let mut any_diverged = false;
    for &n in &sizes {
        // conformance_case runs the correct f64 twin AND launches the f64
        // kernel on wgpu, then compares. The twin is in-bounds so it never
        // panics; wgpu's silent corruption surfaces purely as compare failure.
        let outcome = axpy_f64_vericl::conformance_case::<WgpuRuntime>(&client, n, 0xF64D, 256);
        assert!(
            outcome.reference_panic.is_none(),
            "n={n}: f64 twin panicked unexpectedly: {:?}",
            outcome.reference_panic
        );
        if !outcome.pass() {
            any_diverged = true;
        }
    }

    assert!(
        any_diverged,
        "f64 axpy on wgpu unexpectedly MATCHED the f64 reference twin across {sizes:?} — WGSL \
         has no f64, so this should silently diverge. If wgpu/naga gained real f64 support, the \
         cpu-only f64 lane assumption (README \"f64 support\") needs revisiting."
    );
}
