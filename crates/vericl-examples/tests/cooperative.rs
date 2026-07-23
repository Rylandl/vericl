//! Cooperative-kernel differential probe (shared-memory milestone M5,
//! docs/design-shared-memory.md §4.6): the *generated* phase-split twin of a
//! cooperative kernel, run against the real kernel on wgpu/Metal, must be
//! **bit-exact** (the twin sums in the identical tree order, so at equal f32
//! precision it reproduces the reduction exactly).
//!
//! These kernels are deliberately NOT in `vericl::suite!` yet (the A↔B claim
//! coupling is M6 and the suite wiring is M7). This file exercises the
//! macro-generated `conformance_case` end-to-end on real hardware without
//! touching the evidence manifest.

use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;
use vericl_examples::{block_sum_reduce_vericl, grid_stride_reduce_vericl};

const CUBE_DIM: u32 = 256;
const SEED: u64 = 0xC00B_E5A7;

/// Every compared partial is bit-exact (an exact-comparison pass with
/// `max_ulp == 0`), and the reference did not panic.
fn assert_bit_exact(outcome: &vericl::CaseOutcome, ctx: &str) {
    assert!(
        outcome.reference_panic.is_none(),
        "{ctx}: reference panicked: {:?}",
        outcome.reference_panic
    );
    assert!(!outcome.reports.is_empty(), "{ctx}: no compared outputs");
    for (name, r) in &outcome.reports {
        assert!(
            r.pass && r.max_ulp == Some(0),
            "{ctx}: output `{name}` not bit-exact: pass={} mismatches={} max_ulp={:?}",
            r.pass,
            r.mismatches,
            r.max_ulp
        );
    }
}

#[test]
fn block_sum_reduce_is_bit_exact_vs_wgpu() {
    let client = WgpuRuntime::client(&Default::default());
    for &n in &[1usize, 3, 200, 256, 257, 512, 1000, 4096] {
        let outcome =
            block_sum_reduce_vericl::conformance_case::<WgpuRuntime>(&client, n, SEED, CUBE_DIM);
        assert_bit_exact(&outcome, &format!("block_sum_reduce n={n}"));
    }
}

#[test]
fn grid_stride_reduce_is_bit_exact_vs_wgpu() {
    let client = WgpuRuntime::client(&Default::default());
    for &n in &[1usize, 3, 200, 256, 257, 512, 1000, 4096, 65536] {
        let outcome =
            grid_stride_reduce_vericl::conformance_case::<WgpuRuntime>(&client, n, SEED, CUBE_DIM);
        assert_bit_exact(&outcome, &format!("grid_stride_reduce n={n}"));
    }
}

/// The single-source-of-truth guard (docs/design-shared-memory.md §9 risk 5):
/// launching a cooperative kernel with a `cube_dim` other than its pinned
/// `cooperative(cube_dim = …)` value is a harness bug and panics loudly, rather
/// than silently binding `CUBE_DIM` to a block size the launch does not use.
#[test]
#[should_panic(expected = "pinned to cube_dim = 256")]
fn mismatched_cube_dim_panics() {
    let client = WgpuRuntime::client(&Default::default());
    let _ = block_sum_reduce_vericl::conformance_case::<WgpuRuntime>(&client, 256, SEED, 128);
}
