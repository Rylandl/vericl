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
    kernels: [
        axpy, xorshift_step, mix_u32, fir3, flatten_decode_scale,
        gain_kernel, fir_pair_kernel,
        // Cooperative (workgroup-shared-memory) reduction — the shared-memory
        // milestone (docs/design-shared-memory.md). Carries the triple: tested
        // (differential, race-freedom dependency cited) + proved smt-oob-freedom
        // + proved smt-race-freedom, on BOTH lanes (wgpu + cpu feature).
        //
        // `grid_stride_reduce` is deliberately NOT suite-wired: it reads the
        // `CUBE_COUNT` builtin for its grid stride, which the cubecl-cpu backend
        // does not support ("Unsupported builtin was used: CubeCount") — exactly
        // why the production `reduce_rssi` passes the grid width as a runtime
        // scalar instead. It stays a fully-tested clean-room example (bit-exact
        // vs wgpu in `tests/cooperative.rs`; race-free + in-bounds proved in the
        // lib unit tests), just outside the multi-LANE suite so the cpu lane
        // stays green.
        block_sum_reduce,
    ],
    evidence: "evidence/vericl.json",
    extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime),
}
