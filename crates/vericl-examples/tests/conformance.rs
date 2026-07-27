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

use cubecl::Runtime;
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
        // why the production reduction kernel passes the grid width as a runtime
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
        // Cooperative v1.1 acceptance example — the multi-receiver reduction
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
        // `slice_scale_inplace`: the mutable-**write** path end-to-end (F1,
        // round-9). Every slice example above reads; this scales in place through
        // `y.slice_mut(ABSOLUTE_POS, ABSOLUTE_POS + 1)` (the twin's `&mut
        // y[i..i+1]`), one element per thread so the per-thread windows are
        // disjoint (deterministic differential) and the origin write obligation
        // proves with no assume. Exact (`max_ulp = 0`, single multiply) + proved
        // (2 obligations: the slice write + its read). The multi-element
        // `slice_mut(a,b)[j]` window and the sequential-vs-overlapping aliasing
        // convention are `sequential_slice_mut_scale` + `scratchpad/slicemut`
        // (twin unit test + scratch compile-fail control; not suite-wired).
        slice_scale_inplace,
        // --- Shim-and-small-gate batch (2026-07 coverage re-census) ---
        // The GPU-verified `fma` shim: `cubecl::prelude::fma` is a free
        // function that panics on a host call, and the obvious `a*b + c`
        // substitute rounds twice where it rounds once. Both kernels are
        // bit-exact (`max_ulp = 0`) — the tier the ground truth earns outside
        // the subnormal domain (Metal flushes denormals; both `gen(...)` ranges
        // keep every operand and result well clear of it).
        //
        // `fma_poly3_map`: three NESTED fma's (Horner) inside a composed
        // `#[vericl::helper]` — the shim rewrite's helper site and its
        // post-order nesting.
        fma_poly3_map,
        // `fma_two_product_residual`: `fma(h, x, -(h*x))`, the exact rounding
        // error of a rounded product — the shape whose unfused rewrite is
        // identically zero, i.e. the reason this is a shim and not a
        // source-level rewrite.
        fma_two_product_residual,
        // `CastToF32` source extension. `index_ramp_map` casts the
        // `usize`-typed `ABSOLUTE_POS` (cubecl's `AddressType`, u32 storage);
        // `bernoulli_indicator_map` casts a `bool` predicate (`true → 1.0`) —
        // the Bernoulli value map, the third distribution core. Both bit-exact.
        index_ramp_map,
        bernoulli_indicator_map,
        // `wrapping` on a TUPLE-returning helper: a non-wrapping kernel
        // destructuring a wrapping two-word counter step. Exact u32 + proved
        // bounds; the per-item interaction rule is unchanged.
        counter_split_map,
        // --- Struct-typed #[comptime] parameters (docs/design-struct-comptime.md) ---
        // A `#[comptime] cfg: StageCfg` whose type is declared with
        // `vericl::config! { … }`. The config never reaches the device — CubeCL
        // re-emits `cfg.field`/`cfg.method()` as host Rust at expansion time —
        // so the IR is byte-identical to the same kernel written with plain
        // comptime scalars, and the prover needs no notion of a config at all.
        // What the milestone adds is soundness, not capability: each of these
        // folds `<StageCfg as vericl::ConfigIdentity>::CONFIG_HASH` into its
        // recorded identity, so a config METHOD BODY edit makes this evidence
        // correctly stale (the hole measured in design §5.1 left it "fresh").
        //
        // `config_window_sum`: fields + a depth-2 config method as a LOOP BOUND
        // + a `comptime!` block over the same config (enum dispatch). Bit-exact
        // (`max_ulp = 0`, plain adds and one multiply by a constant — no fma
        // contraction shape) + proved bounds.
        config_window_sum,
        // `config_mode_scale`: the same config type pinned at a different value,
        // exercising the enum's other arm and a config method NAMED `dot` —
        // which the receiver-blind float-method check used to reject with a
        // message about `F::dot` (design R6's measured false positive). Bit-exact
        // + proved.
        config_mode_scale,
    ],
    evidence: "evidence/vericl.json",
    extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime),
}

/// NEGATIVE CONTROL for the `fma` shim — the measured answer to "why not just
/// rewrite `fma(a, b, c)` to `a*b + c` in the twin?".
///
/// `fma_two_product_residual` (suite-wired) and `unfused_two_product_residual`
/// differ by exactly that rewrite: `fma(hi, xi, -product)` vs
/// `hi*xi - product`, where `product` is the rounded `hi*xi`. This test runs
/// BOTH on the same inputs and pins the gap **on the device**:
///
/// - the fused kernel returns the true rounding residual of `hi*xi` — non-zero
///   on essentially every input, and bit-exactly what the `fma` shim computes;
/// - the unfused kernel returns identically `0.0`, and its twin (which computes
///   the unfused expression too) matches it bit-exactly.
///
/// So the rewrite is not a tolerance question, it is a different function, and
/// the difference is 100% of the answer. Both twins are faithful — each models
/// the kernel that was actually written — which is exactly the property the
/// shim buys: vericl reproduces `fma` as `fma`, not as an algebraic
/// paraphrase.
///
/// **Measured backend detail, corrected here** (it is not the same fact as the
/// FMA-contraction one recorded for `vec_madd_bitexact`): Metal contracts
/// `a*b + c` into a single fused instruction when `c` is *independent* — that is
/// what `vec_madd_bitexact`'s bit-exact failure and the ground-truth probe's
/// "unfused kernel differs from the fused intrinsic on 0 of 7972 triples"
/// both show. Here the addend IS the product, so common-subexpression
/// elimination collapses `t - t` to zero *before* any contraction can apply,
/// and the unfused kernel is genuinely unfused on the GPU. Both behaviours are
/// pinned: this test would fail if either changed.
#[test]
fn fused_and_unfused_residual_kernels_compute_different_functions_on_gpu() {
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    let n = 4096usize;
    let seed = 0x5EED_1234u64;

    // Both kernels' twins track their own GPU kernel bit-exactly (`max_ulp = 0`).
    let fused =
        fma_two_product_residual_vericl::conformance_case::<cubecl::wgpu::WgpuRuntime>(
            &client, n, seed, 64,
        );
    assert!(fused.pass(), "the fused kernel's shim-routed twin must be bit-exact: {fused:?}");
    let unfused =
        unfused_two_product_residual_vericl::conformance_case::<cubecl::wgpu::WgpuRuntime>(
            &client, n, seed, 64,
        );
    assert!(
        unfused.pass(),
        "the unfused kernel's twin must also be faithful — vericl models what was written, and \
         if THIS fails the backend has started contracting `t - t` and the finding recorded in \
         `host_shims::fma_f32` must be re-measured: {unfused:?}"
    );

    // …and they are different functions. Compare the two twins' VALUES over
    // the declared input range (the differential above only says each twin
    // matches its own kernel; this says the two kernels are not the same
    // computation). Inputs are drawn independently of `conformance_case`'s
    // stream on purpose — the claim is about the functions over the declared
    // `gen(...)` range, not about one particular draw.
    let m = 1024usize;
    let h: Vec<f32> = (0..m).map(|i| 0.0009765625 + (i as f32) * 0.000_97).collect();
    let x: Vec<f32> = (0..m).map(|i| 1.0 + (i as f32) * 3.997).collect();
    let mut y_fused = vec![0f32; m];
    let mut y_unfused = vec![0f32; m];
    fma_two_product_residual_vericl::reference(&h, &x, &mut y_fused, m);
    unfused_two_product_residual_vericl::reference(&h, &x, &mut y_unfused, m);
    assert!(
        y_unfused.iter().all(|v| *v == 0.0),
        "the unfused rewrite collapses to exactly zero, by construction"
    );
    let nonzero = y_fused.iter().filter(|v| **v != 0.0).count();
    assert!(
        nonzero > m / 2,
        "the fused residual must be genuinely non-zero on most inputs, else this control is \
         vacuous (got {nonzero} of {m})"
    );
    println!(
        "fma discrimination: fused residual non-zero on {nonzero}/{m} inputs, unfused rewrite \
         exactly 0.0 on all {m} — a 100% relative error, not a tolerance gap"
    );
}
