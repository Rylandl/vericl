//! End-to-end vectorized differential conformance (design-line-vector.md §4.4,
//! §9, §11 V4). Exercises the macro-generated `conformance_case` for
//! `Array<Vector<f32, 4>>` kernels: flat-scalar input generation (`lines * W`
//! scalars per array), the launch spliced at the pinned vectorization `W`,
//! `lines`-thread dispatch, flat-scalar readback, and the flat-scalar compare
//! with per-lane `(line, lane)` divergence reporting.
//!
//! Two tiers, matching the GT test's finding (`line_shim_gpu_ground_truth.rs`):
//! - `vec_add`/`vec_scale`: per-lane `+`/`*` are correctly rounded, so the twin
//!   is bit-exact with the GPU — these pass under any honest tolerance.
//! - `vec_madd` (`a*a + b`): fusable to one FMA rounding on the backend, a
//!   ≤1-ULP per-lane gap the declared `compare(abs = …)` covers (the same
//!   float-contraction story a scalar `a*a+b` kernel has). This is the explicit
//!   "div/transcendental/FMA needs a tolerance" example the GT finding demands.
//!
//! The negative controls: `vec_add_off_by_one` (a bounds over-read caught by the
//! reference panicking, backend-independent) and `vec_madd_bitexact` (a
//! too-tight tolerance whose divergence is reported with the lane named).
//!
//! Runs on wgpu and, behind `--features cpu`, cubecl-cpu. The FMA-contraction
//! reporting test is wgpu-only: whether `a*a+b` fuses is a backend property
//! (Metal fuses — design §6), so the bit-exact negative control is asserted on
//! the lane known to exhibit it.
#![cfg(feature = "wgpu")]

use cubecl::Runtime;
use cubecl::prelude::ComputeClient;
use cubecl::wgpu::WgpuRuntime;
use vericl::describe_case_outcome;
use vericl_examples::{
    vec_add_off_by_one_vericl, vec_add_vericl, vec_madd_bitexact_vericl, vec_madd_vericl,
    vec_scale_vericl,
};

// `n` is the LINE count (design §2.3): buffers are `n * 4` scalars, the launch
// dispatches `n` threads. Sweeps 1 line up to a multi-cube dispatch.
const LINE_COUNTS: [usize; 6] = [1, 3, 7, 64, 257, 1000];
const CUBE_DIM: u32 = 64;
const SEED: u64 = 0x5EC0_1DED;

/// The correctly-rounded elementwise kernels (`vec_add`, `vec_scale`) and the
/// tolerance-covered `vec_madd` all pass through the full vectorized launch/I/O
/// path at every swept size, and the off-by-one is caught. Backend-generic so
/// both the wgpu and cpu lanes share one body.
fn conformance_lane<R: Runtime>(client: &ComputeClient<R>, lane: &str) {
    for &n in &LINE_COUNTS {
        let o = vec_add_vericl::conformance_case::<R>(client, n, SEED ^ n as u64, CUBE_DIM);
        assert!(o.pass(), "[{lane}] vec_add n={n}: {}", describe_case_outcome(&o));

        let o = vec_scale_vericl::conformance_case::<R>(client, n, SEED ^ 0xA1 ^ n as u64, CUBE_DIM);
        assert!(o.pass(), "[{lane}] vec_scale n={n}: {}", describe_case_outcome(&o));

        let o = vec_madd_vericl::conformance_case::<R>(client, n, SEED ^ 0xB2 ^ n as u64, CUBE_DIM);
        assert!(
            o.pass(),
            "[{lane}] vec_madd n={n} diverged under its declared abs tolerance: {}",
            describe_case_outcome(&o)
        );
    }

    // Negative control (design §11 V4): the off-by-one (`<= out.len()`) over-read
    // is caught by the reference twin panicking on `out[out.len()]` when the
    // multi-cube dispatch runs a thread at `ABSOLUTE_POS == out.len()` (100 lines
    // -> 128 threads under cube_dim 64). Backend-independent (it is the twin, not
    // the GPU, that panics — WGSL robustness would silently clamp).
    let o = vec_add_off_by_one_vericl::conformance_case::<R>(client, 100, SEED, CUBE_DIM);
    assert!(!o.pass(), "[{lane}] vec_add_off_by_one was NOT caught");
    let msg = describe_case_outcome(&o);
    assert!(msg.contains("bounds"), "[{lane}] off-by-one should be a bounds finding: {msg}");
}

#[test]
fn vector_conformance_wgpu() {
    let client = WgpuRuntime::client(&Default::default());
    conformance_lane(&client, "wgpu");
}

/// Second lane (`--features cpu`): the same all-`Vector<f32,4>` elementwise
/// kernels on cubecl-cpu. A disagreement between lanes is a finding, not
/// something to average away.
#[cfg(feature = "cpu")]
#[test]
fn vector_conformance_cpu() {
    let client = cubecl::cpu::CpuRuntime::client(&Default::default());
    conformance_lane(&client, "cpu");
}

/// Per-lane `(line, lane)` divergence reporting (design §9), end to end, on
/// wgpu/Metal where `a*a+b` fuses. The bit-exact `vec_madd_bitexact` fails on the
/// FMA-contraction gap the honest `vec_madd` tolerance covers; the failing report
/// must NAME the divergent lane (`out[line=.., lane=..]`), proving the flat
/// compare's per-lane reporting is wired through the real generated
/// `conformance_case`, not just a flat offset.
#[test]
fn per_lane_divergence_is_reported_by_line_and_lane_wgpu() {
    let client = WgpuRuntime::client(&Default::default());
    let n = 257usize; // lines
    let outcome =
        vec_madd_bitexact_vericl::conformance_case::<WgpuRuntime>(&client, n, 0xF3A, CUBE_DIM);
    assert!(
        !outcome.pass(),
        "vec_madd_bitexact should be caught: a bit-exact tolerance cannot cover the FMA gap"
    );
    assert!(outcome.reference_panic.is_none(), "this is a value divergence, not a bounds panic");
    let (label, report) =
        outcome.reports.iter().find(|(_, r)| !r.pass).expect("a failing report");
    assert!(
        label.contains("line=") && label.contains("lane="),
        "the divergence must name (line, lane); got label `{label}`"
    );
    // The label decodes exactly from the report's worst flat index.
    let w = report.worst.as_ref().expect("a worst mismatch");
    assert_eq!(
        label,
        &format!("out[line={}, lane={}]", w.index / 4, w.index % 4),
        "label must match the worst mismatch's decoded lane"
    );
    println!("caught vec_madd_bitexact divergence: {}", describe_case_outcome(&outcome));
}
