//! Honest-fallback tier regression (docs/design-shared-memory.md §6, M6): with
//! `prove: false`, a cooperative kernel's phase-split twin is still faithful
//! *only* under intra-phase race freedom — but nothing proved it this run. The
//! suite must therefore inject the EXPLICIT `intra-phase-race-freedom` assumed
//! claim and have the tested claim depend on it, never record a silent green
//! cooperative pass. This is the same posture as `prove: false` omitting the
//! bounds proof rather than faking one.
//!
//! Kept in its own evidence file so it never collides with the primary
//! `evidence/vericl.json` (which records the strong tier for the same kernel).
//!
//! Usage: `cargo test` verifies `evidence/cooperative_fallback.json`;
//! `VERICL_UPDATE=1 cargo test` regenerates it. No z3 needed (prove: false).

use vericl_examples::*;

vericl::suite! {
    runtime: cubecl::wgpu::WgpuRuntime,
    kernels: [block_sum_reduce],
    evidence: "evidence/cooperative_fallback.json",
    prove: false,
}
