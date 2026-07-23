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
        // Array-value-dependent indices (offset table / gather) — the last
        // Tier-2 prover gap (docs/dogfood-2026-07.md). `y[i] = x[offsets[i]]`
        // carries `tested` (bit-exact differential) + `proved` (3-obligation
        // SMT bounds), the latter only reachable because the element-range
        // assume models the loaded offset `< x.len()`.
        gather_copy,
        // match / Switch (quick-wins batch 1): a `match` on the scalar `mode`
        // lowers to `Branch::Switch`, modeled by the prover as an exhaustive
        // if-chain. Carries `tested` (differential) + `proved` (6-obligation
        // SMT bounds, 3 arms × {x read, y write}).
        select_mode,
        // Length-relationship assume (quick-wins batch 1): `y.len() + 4 <=
        // x.len()` discharges the forward read `x[i + 4]` under a `i < y.len()`
        // guard. Carries `tested` + `proved` (3-obligation SMT bounds).
        offset_window,
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
        // Cooperative v1.1 acceptance example — the `emitter_powers_multi_rx`
        // shape (minus 2-D dispatch): a #[comptime] `n_emitters`, a `uses(...)`
        // helper in phase 0 (`square_sample`), and a workgroup-uniform
        // `terminate!()` padding guard, all at once. Carries the full triple
        // (tested + proved smt-oob-freedom + proved smt-race-freedom) on both
        // lanes — the cooperative v1.1 extensions landing together on the real
        // reduction shape (docs/design-shared-memory.md §7.4).
        emitter_reduce,
    ],
    evidence: "evidence/vericl.json",
    extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime),
}
