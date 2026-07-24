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
        // --- Quick-wins batch 2 (macro-leaning) ---
        // Feature 1 (verified host shims): the flagship u32-RNG-output →
        // unit-interval-f32 kernel, `y[i] = cast_from(x[i] >> 8) / 2^24` via a
        // composed helper using the GPU-verified `cast_from` shim. Bit-exact
        // (max_ulp=0) + proved bounds. `mul_hi_map` exercises the verified
        // `mul_hi` shim (exact u32 high word) + proved bounds.
        unit_interval_map,
        mul_hi_map,
        // Feature 2 (helper-level wrapping): a NON-wrapping kernel composing the
        // WRAPPING `lcg_step` helper (`z*a+b`, wrap-on-overflow) — the interaction
        // rule in action. Exact u32 + proved bounds.
        lcg_map,
        // Feature 3 (comptime! block evaluation): `comptime!(extra + 2)` is
        // evaluated at expansion (extra is #[comptime]-pinned) and used as a
        // shift amount. Exact u32 + proved bounds.
        comptime_shift,
        // Cooperative v1.1 acceptance example — the `emitter_powers_multi_rx`
        // shape (minus 2-D dispatch): a #[comptime] `n_emitters`, a `uses(...)`
        // helper in phase 0 (`square_sample`), and a workgroup-uniform
        // `terminate!()` padding guard, all at once. Carries the full triple
        // (tested + proved smt-oob-freedom + proved smt-race-freedom) on both
        // lanes — the cooperative v1.1 extensions landing together on the real
        // reduction shape (docs/design-shared-memory.md §7.4).
        emitter_reduce,
        // --- Vector<P, N> elementwise (design-line-vector.md §11 V5) ---
        // Clean-room vectorized elementwise add over `Array<Vector<f32, 4>>`
        // (width pinned via `instantiate(N = 4)`). The vectorized differential
        // path (flat-scalar gen of `lines*4` scalars, launch spliced at
        // vectorization 4, flat-scalar per-lane compare) + the whole-vector
        // line-granular bounds proof carry `tested` (bit-exact — a vec-4 add is 4
        // correctly-rounded scalar adds) + `proved` (3-obligation SMT bounds, N
        // absent from the obligation). The `sizes` are line counts; the pinned
        // width is recorded in the claim config (§9). Generalizes the scalar
        // elementwise shortlist to its true vector element type.
        vec_add,
        // --- Core `Slice` (docs/design-view-slice.md) ---
        // The #2 ecosystem gap's tractable half. A slice access lowers to a
        // checked `origin[offset + i]`, so bounds proving is the ordinary
        // origin obligation, UNMODIFIED (deliverable B is a no-op for the
        // prover, §5). The twin maps a slice to a Rust subslice (`&arr[a..b]`) —
        // bit-exact (a slice adds no numeric op, §6) with Rust as the soundness
        // oracle for slice-creation validity and mutable aliasing (§4.3/§4.4).
        //
        // `windowed_slice_sum`: dynamic-offset slice creation + `for v in slice`
        // iteration (`RangeLoop` over `x[i+j]`, §2.2) + length. Bit-exact
        // windowed sum + proved bounds.
        windowed_slice_sum,
        // `slice_gather_copy`: gather through a `to_slice()` of an element-
        // assumed offset table — the element assume transfers through the slice
        // via origin-id keying, for free (§5.4). Exact + proved (3 obligations).
        slice_gather_copy,
        // `windowed_helper_kernel`: the dominant composition usage — a
        // `#[vericl::helper]` taking a `&Slice<F>` param (§10), called with the
        // idiomatic `&x.slice(a, b)` form. Exact + proved.
        windowed_helper_kernel,
    ],
    evidence: "evidence/vericl.json",
    extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime),
}
