//! Conformance suite: differential-tests the honest example kernels (GPU vs.
//! the macro-derived sequential reference) and maintains the evidence
//! manifest. `cargo test` is the whole CI story (README "Locked decisions").
//!
//! Usage:
//!   cargo test                     verify evidence/vericl.json (fails on
//!                                   missing, stale, or mismatched evidence)
//!   VERICL_UPDATE=1 cargo test     regenerate evidence/vericl.json
//!   cargo test --features cpu      also adds the cubecl-cpu lane's claims
//!
//! Deliberately defective kernels (`axpy_off_by_one`, `sum_racy`) stay OUT
//! of this suite — they belong to the `conform` binary's demo-defects mode,
//! which shows the checks catching them on purpose.

use vericl_examples::*;

vericl::suite! {
    runtime: cubecl::wgpu::WgpuRuntime,
    kernels: [axpy, xorshift_step, mix_u32, fir3],
    evidence: "evidence/vericl.json",
    extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime),
}
