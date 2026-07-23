//! Example kernels for the vericl first release.
//!
//! Two honest kernels (one float with a declared tolerance, one integer with
//! exact comparison) and two deliberately defective twins whose defects the
//! differential check must catch (README outcome 4).

use cubecl::prelude::*;

/// Generic saxpy — the flagship `instantiate(...)` example: `F` is pinned to
/// `f32` below, monomorphizing the reference twin, `conformance_case`'s
/// launch, and `kernel_definition`'s IR extraction all at that one concrete
/// type (see the `instantiate(...)` contract clause in the README).
///
/// Tolerance rationale (a first real VeriCL finding): the wgpu/Metal backend
/// contracts `a*x + y` (fma / fast-math), so under cancellation no useful ULP
/// bound exists — divergence up to ~27k ULP was observed. The honest claim is
/// an absolute error bound derived from the declared input ranges: one
/// rounding of `alpha*x` with |alpha| <= 4 and |x| <= 100 is at most
/// ulp(400) ≈ 3.1e-5, so abs = 1e-4 covers contraction with margin.
#[vericl::kernel(
    assumes(
        x.len() == y.len(),
        alpha.abs() <= 4.0,
        x.iter().all(|v| v.abs() <= 100.0),
        y.iter().all(|v| v.abs() <= 100.0)
    ),
    compare(abs = 1e-4),
    gen(alpha in -4.0..=4.0, x in -100.0..=100.0, y in -100.0..=100.0),
    instantiate(F = f32)
)]
#[cube(launch)]
pub fn axpy<F: Float + CubeElement>(alpha: F, x: &Array<F>, y: &mut Array<F>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
    }
}

/// A small windowed FIR (up to 3 taps), generic over its element type *and*
/// pinning the active tap count via `#[comptime]` — the milestone's
/// headline case: a genuinely generic + comptime kernel that still lands a
/// **proved** out-of-bounds-freedom claim (see
/// `fir3_kernel_definition_is_provably_in_bounds` below), not just a
/// differential one. Deliberately avoids a loop-carried accumulator (the
/// dogfood survey's Tier-2 prover gap, docs/dogfood-2026-07.md) by fully
/// guarding each of the (at most) two extra taps with its own condition
/// rather than a `for k in 0..taps` loop.
///
/// Guards are written as a single `&&`-composed condition (`taps > 1 &&
/// ABSOLUTE_POS >= 1`) — the natural, idiomatic shape. This used to require
/// two nested `if`s instead (see `crates/vericl-ir/src/prover.rs`'s
/// `nested_if_guard_still_proves` test, which pins that the nested-if shape
/// still proves too): the SMT bounds prover didn't model `&&`-composed
/// branch conditions and reported `OutOfSubset` for them, confirmed
/// empirically at the time by collapsing this exact kernel. Boolean
/// condition composition (`Operator::And`/`Or`/`Not`, all eagerly evaluated
/// by CubeCL rather than lowered to nested branches — see
/// docs/ir-research.md §3) is now modeled, so the collapsed form proves.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0),
    instantiate(F = f32, taps = 3)
)]
#[cube(launch)]
pub fn fir3<F: Float>(x: &Array<F>, y: &mut Array<F>, #[comptime] taps: u32) {
    if ABSOLUTE_POS < y.len() {
        let mut acc = x[ABSOLUTE_POS];
        if taps > 1 && ABSOLUTE_POS >= 1 {
            acc += x[ABSOLUTE_POS - 1];
        }
        if taps > 2 && ABSOLUTE_POS >= 2 {
            acc += x[ABSOLUTE_POS - 2];
        }
        y[ABSOLUTE_POS] = acc;
    }
}

/// Same shape as `fir3`, pinned at a different comptime value purely to
/// demonstrate that changing an `instantiate(...)` value changes kernel
/// identity (`SOURCE_HASH`) — see
/// `fir3_identity_changes_with_instantiate_value` below. Not part of the
/// conformance suite. Deliberately kept on the *nested*-`if` guard shape
/// (rather than following `fir3`'s move to `&&`) so the two kernels'
/// source text differs only in the pinned `taps` value and this guard
/// shape, not in both — the hash-differs test is about `instantiate(...)`
/// specifically, and nested-`if` provability is independently pinned by
/// `crates/vericl-ir/src/prover.rs`'s `nested_if_guard_still_proves`.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0),
    instantiate(F = f32, taps = 1)
)]
#[cube(launch)]
pub fn fir3_alt<F: Float>(x: &Array<F>, y: &mut Array<F>, #[comptime] taps: u32) {
    if ABSOLUTE_POS < y.len() {
        let mut acc = x[ABSOLUTE_POS];
        #[allow(clippy::collapsible_if)]
        if taps > 1 {
            if ABSOLUTE_POS >= 1 {
                acc += x[ABSOLUTE_POS - 1];
            }
        }
        #[allow(clippy::collapsible_if)]
        if taps > 2 {
            if ABSOLUTE_POS >= 2 {
                acc += x[ABSOLUTE_POS - 2];
            }
        }
        y[ABSOLUTE_POS] = acc;
    }
}

/// One xorshift32 step per element — integer, bit-exact, RNG-flavored
/// (the non-Substrate-shaped example).
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(exact)
)]
#[cube(launch)]
pub fn xorshift_step(x: &Array<u32>, y: &mut Array<u32>) {
    if ABSOLUTE_POS < y.len() {
        let mut s = x[ABSOLUTE_POS];
        s ^= s << 13u32;
        s ^= s >> 17u32;
        s ^= s << 5u32;
        y[ABSOLUTE_POS] = s;
    }
}

/// Murmur3 `fmix32`-style integer mixer, one element per thread — integer,
/// bit-exact, and relies on wrap-on-overflow: the multiplies use large odd
/// constants that routinely overflow `u32`, and WGSL wraps on overflow where
/// Rust's default (debug) arithmetic panics. Same finding class as the fma
/// story in the README ("A first finding"): the fix is the declared
/// `wrapping` contract clause below, which folds the reference twin's
/// `*`/`>>` to `wrapping_mul`/`wrapping_shr` — not a silent approximation.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(exact),
    wrapping
)]
#[cube(launch)]
pub fn mix_u32(x: &Array<u32>, y: &mut Array<u32>) {
    if ABSOLUTE_POS < y.len() {
        let mut h = x[ABSOLUTE_POS];
        h ^= h >> 16u32;
        h *= 0x85ebca6bu32;
        h ^= h >> 13u32;
        h *= 0xc2b2ae35u32;
        h ^= h >> 16u32;
        y[ABSOLUTE_POS] = h;
    }
}

/// Flat 1-D `ABSOLUTE_POS` decoded into a `(row, col)` pair via `/`/`%`,
/// recombined into a write index, and scaled — the milestone's div/mod
/// headline (docs/dogfood-2026-07.md Tier-2 gap #2, candidate
/// `flatten_decode_scale`): a clean-room stand-in for the dogfood survey's
/// "flat 1-D → row/col decode" shape (7/22 kernels blocked on it), pinning
/// the div/mod prover boundary with a genuine, publicly-committable
/// example. `width` is a plain runtime `u32` parameter, not a `#[comptime]`
/// constant — the div/mod modeling has to hold for a symbolic divisor, not
/// just a literal.
///
/// The guard is a single `ABSOLUTE_POS < y.len()` (the same axpy-shaped
/// bound as every other honest kernel here) plus `width >= 1u32`, which is
/// what actually matters for the *proof*: it's exactly the fact
/// `vericl-ir`'s div/mod side-obligation needs to discharge (divisor
/// nonzero; both operands nonnegative comes for free — `ABSOLUTE_POS` and
/// `width` are both unsigned leaves). The write index is the *recombined*
/// `row * width + col`, not `ABSOLUTE_POS` directly — proving it in bounds
/// requires the SMT solver to derive `row * width + col == ABSOLUTE_POS`
/// from the SMT-LIB `div`/`mod` (Euclidean) axioms and combine that with
/// the `ABSOLUTE_POS < y.len()` guard, which is the actual boundary this
/// example pins (see `flatten_decode_scale_kernel_definition_is_provably_in_bounds`
/// below). Euclidean division coincides with Rust's/WGSL's truncated
/// semantics exactly when both operands are nonnegative (see
/// `vericl-ir`'s module docs) — true here by construction, so the
/// differential reference and the real kernel compute identically.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-4),
    gen(x in -100.0..=100.0, y in 0.0..=0.0, width in 1..=64, scale in 0.1..=4.0)
)]
#[cube(launch)]
pub fn flatten_decode_scale(x: &Array<f32>, y: &mut Array<f32>, width: u32, scale: f32) {
    if ABSOLUTE_POS < y.len() && width >= 1u32 {
        let w = width as usize;
        let row = ABSOLUTE_POS / w;
        let col = ABSOLUTE_POS % w;
        y[row * w + col] = x[ABSOLUTE_POS] * scale;
    }
}

// ---------------------------------------------------------------------------
// Kernel composition: #[vericl::helper] + uses(...) — the milestone this
// section demonstrates (docs/dogfood-2026-07.md Tier-1 gap #2, blocking
// 16/22 dogfooded kernels). See README "Kernel composition" for the design.
//
// A note on why every helper below is fully monomorphized via its own
// instantiate(...) clause rather than left generic (a first draft of this
// design tried the latter): `FLOAT_METHOD_WHITELIST`'s host-safety proof
// (crates/vericl-macros' module doc) relies on Rust preferring an inherent
// method over a trait method for a *concrete* receiver type. A bound-but-
// unsubstituted generic type parameter `F: Float` has no inherent methods
// at all, so a whitelisted call like `.abs()`/`.sqrt()` inside a still-
// generic host fn resolves purely through the `Float`/`Numeric` trait,
// which (confirmed by reading cubecl-core's `impl_unary_func!` macro, and
// empirically: a scratch generic `fn g<F: Float>(x: F) -> F { x.sqrt() }`
// panics on host calling `g(2.5f32)`, as does `.abs()`) is exactly the
// unverified `unexpanded!()` panic path the whitelist exists to keep out of
// a twin. Monomorphizing via instantiate(...) — reusing the exact same
// resolve_instantiate/FloatMethodCheck machinery a kernel already uses —
// closes that gap the same proven way instead of introducing a second,
// weaker safety story. The cost: a helper's twin is pinned to one concrete
// type (only `f32` is supported anywhere in vericl v0 today, so this is
// free in practice); a `#[comptime]` parameter, unlike a generic type
// parameter, is never pinned by a helper's instantiate(...) — it stays an
// ordinary pass-through parameter, since it carries no host-callability
// hazard (it's just a plain integer value) and the caller's own twin
// already has the concrete pinned value in hand to pass along.

/// Scales one scalar by a gain factor — the simplest possible helper (no
/// array parameters, no other helper calls), reused by two kernels below
/// (`gain_kernel` directly, `fir_pair_scaled` transitively).
#[vericl::helper(instantiate(F = f32))]
#[cube]
pub fn single_tap<F: Float>(a: F, gain: F) -> F {
    a * gain
}

/// A 2-tap FIR pair returning a tuple — the milestone's headline "helper
/// returning a tuple" shape (docs/dogfood-2026-07.md's suggested candidate
/// example). Tuple returns are plain Rust; no special handling needed.
#[vericl::helper(instantiate(F = f32))]
#[cube]
pub fn fir_pair<F: Float>(a: F, b: F) -> (F, F) {
    (a + b, a - b)
}

/// Calls two OTHER `#[vericl::helper]`-annotated functions —
/// helper-calling-helper composition, supported via the exact same
/// `uses(...)` rewrite mechanism a kernel gets (no special-casing). Its
/// twin's identity recursively folds in both `fir_pair`'s and
/// `single_tap`'s (see `fir_pair_scaled_vericl::identity_hash`).
#[vericl::helper(instantiate(F = f32), uses(fir_pair, single_tap))]
#[cube]
pub fn fir_pair_scaled<F: Float>(a: F, b: F, gain: F) -> (F, F) {
    let sum_diff: (F, F) = fir_pair::<F>(a, b);
    (single_tap::<F>(sum_diff.0, gain), single_tap::<F>(sum_diff.1, gain))
}

/// Reads a value AND its right neighbor — unlike the helpers above, the
/// array access genuinely lives *inside the helper's own body*, not the
/// caller's. Pins the prover-composition boundary (see
/// `tap_pair_guarded_kernel`/`tap_pair_unguarded_kernel` below): whether the
/// SMT bounds prover, walking a kernel's inlined IR, discharges an
/// obligation that only exists because of what a composed helper does.
#[vericl::helper(instantiate(F = f32))]
#[cube]
pub fn tap_pair<F: Float>(x: &Array<F>, idx: usize) -> F {
    x[idx] + x[idx + 1]
}

/// Composed kernel A: calls `single_tap` directly. Wired into
/// `vericl::suite!` — carries both `tested` (differential) and `proved`
/// (SMT bounds) claims, same as any honest non-composed kernel; composition
/// needed zero prover changes (cube expansion inlines the helper's IR
/// directly into this kernel's own `Scope` — see
/// `crates/vericl-ir/src/prover.rs`'s module doc, "Soundness notes").
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0, gain in -4.0..=4.0),
    instantiate(F = f32),
    uses(single_tap)
)]
#[cube(launch)]
pub fn gain_kernel<F: Float + CubeElement>(x: &Array<F>, y: &mut Array<F>, gain: F) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = single_tap::<F>(x[ABSOLUTE_POS], gain);
    }
}

/// Fix 3 regression pin (round-2 adversarial review, `UsesRewriteFold`
/// multi-segment call bypass): identical to `gain_kernel` above, except the
/// call to `single_tap` is `self::`-qualified. Before the fix, a
/// multi-segment path bypassed both the rewrite AND the unlisted-callee
/// rejection entirely, so the twin silently called the ORIGINAL `#[cube]
/// fn single_tap` host-side instead of `single_tap_vericl_ref` — never
/// caught, since (for this host-safe helper) both compute the same answer,
/// making it invisible to a black-box differential check; see
/// `self_path_gain_kernel_twin_matches_hand_computed` below, which pins the
/// fix at the AST level via `gain_kernel_twin_matches_hand_computed`'s own
/// expected values instead. Not suite-wired (no new evidence entry needed
/// — this exists purely to pin the fix, same precedent as
/// `tap_pair_guarded_kernel` below).
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0, gain in -4.0..=4.0),
    instantiate(F = f32),
    uses(single_tap)
)]
#[cube(launch)]
pub fn self_path_gain_kernel<F: Float + CubeElement>(x: &Array<F>, y: &mut Array<F>, gain: F) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = self::single_tap::<F>(x[ABSOLUTE_POS], gain);
    }
}

/// Composed kernel B: calls `fir_pair_scaled`, which itself calls
/// `fir_pair` and `single_tap` — two levels of composition end to end, and
/// `single_tap` reused across both `gain_kernel` (directly) and this kernel
/// (transitively). Two `&mut Array` outputs, one per tuple element — the
/// macro-generated `conformance_case`/comparison machinery already handles
/// N output buffers generically, so this needed no new machinery either.
/// Wired into `vericl::suite!`: the composed kernel carrying tested + proved
/// claims the milestone asks for.
#[vericl::kernel(
    assumes(x.len() == sum_out.len(), x.len() == diff_out.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, sum_out in 0.0..=0.0, diff_out in 0.0..=0.0, gain in -4.0..=4.0),
    instantiate(F = f32),
    uses(fir_pair_scaled)
)]
#[cube(launch)]
pub fn fir_pair_kernel<F: Float + CubeElement>(
    x: &Array<F>,
    sum_out: &mut Array<F>,
    diff_out: &mut Array<F>,
    gain: F,
) {
    if ABSOLUTE_POS + 1 < x.len() {
        let s_d: (F, F) = fir_pair_scaled::<F>(x[ABSOLUTE_POS], x[ABSOLUTE_POS + 1], gain);
        sum_out[ABSOLUTE_POS] = s_d.0;
        diff_out[ABSOLUTE_POS] = s_d.1;
    }
}

/// Prover-composition positive control (docs/dogfood-2026-07.md-style, not
/// wired into `vericl::suite!` — mirrors the `stepped_loop_*` precedent
/// below of a kernel that exists purely to pin a prover finding, not to
/// carry evidence): the guard `ABSOLUTE_POS + 1 < x.len()` covers BOTH
/// reads `tap_pair`'s own body performs (`x[idx]`, `x[idx + 1]`) even
/// though those accesses live inside the composed helper, not here. Must
/// discharge `Proved` — see
/// `tap_pair_guarded_kernel_kernel_definition_is_provably_in_bounds` below.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0),
    instantiate(F = f32),
    uses(tap_pair)
)]
#[cube(launch)]
pub fn tap_pair_guarded_kernel<F: Float + CubeElement>(x: &Array<F>, y: &mut Array<F>) {
    if ABSOLUTE_POS + 1 < x.len() {
        y[ABSOLUTE_POS] = tap_pair::<F>(x, ABSOLUTE_POS);
    }
}

/// Prover-composition negative control: same shape as
/// `tap_pair_guarded_kernel` and the same helper (`tap_pair` — demonstrating
/// helper reuse together with the kernel above), but the guard only
/// establishes `ABSOLUTE_POS < x.len()`, one short of what `tap_pair`'s own
/// unguarded `x[idx + 1]` read needs at the top position. Must `Refuted`,
/// never `Proved` — the obligation genuinely lives inside the helper's
/// body, and the prover must not silently drop it just because it's
/// composed rather than written directly in the kernel.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0),
    instantiate(F = f32),
    uses(tap_pair)
)]
#[cube(launch)]
pub fn tap_pair_unguarded_kernel<F: Float + CubeElement>(x: &Array<F>, y: &mut Array<F>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = tap_pair::<F>(x, ABSOLUTE_POS);
    }
}

/// DEFECTIVE: boundary guard is `<=`, reading and writing one element past
/// the end. WGSL robustness silently clamps this on the GPU, so a GPU-only
/// test can pass; the sequential reference panics deterministically.
#[vericl::kernel(
    assumes(x.len() == y.len(), alpha.abs() <= 4.0),
    compare(abs = 1e-4),
    gen(alpha in -4.0..=4.0, x in -100.0..=100.0, y in -100.0..=100.0)
)]
#[cube(launch)]
pub fn axpy_off_by_one(alpha: f32, x: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS <= y.len() {
        y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
    }
}

/// DEFECTIVE: unsynchronized accumulation — every thread read-modify-writes
/// `y[0]` with no atomics. The sequential reference computes the true sum;
/// the GPU result is whatever the race leaves behind.
#[vericl::kernel(
    assumes(y.len() == 1),
    compare(max_ulp = 0),
    gen(x in 0.5..=1.5, y in 0.0..=0.0, len(y = 1))
)]
#[cube(launch)]
pub fn sum_racy(x: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < x.len() {
        y[0] += x[ABSOLUTE_POS];
    }
}

/// REGRESSION (adversarial soundness review, Bug 1 — see
/// `vericl_ir::prover::process_range_loop`): `range_stepped` with a negative
/// step produces a descending loop (`start > end` numerically). The SMT
/// prover must reject this outright rather than silently assert ascending
/// bounds, which for a real descending loop are unsatisfiable and would make
/// every obligation inside vacuously "provable". This kernel's body is an
/// ordinary in-bounds copy — even so it must not prove: the loop *shape* is
/// outside the modeled subset, independent of whether the body happens to be
/// safe. Not exercised by `conform`'s differential/evidence pipeline
/// (never GPU-launched); used only by the prover regression tests below.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(exact)
)]
#[cube(launch)]
pub fn stepped_loop_descending_copy(x: &Array<u32>, y: &mut Array<u32>) {
    let n = y.len() as i32;
    for i in range_stepped(n - 1, -1, -1) {
        let idx = i as usize;
        y[idx] = x[idx];
    }
}

/// REGRESSION (adversarial soundness review, Bug 1): the exact vacuous-proof
/// shape the review demonstrated — a runtime-bounded, negative-step loop
/// whose body writes far out of bounds (`y[100000]`). Before the fix,
/// `process_range_loop` asserted ascending bounds for this descending loop,
/// making the SMT context infeasible and vacuously discharging the
/// `y[100000]`/`x[100000]` obligations as "proved" even though a real
/// (sequential) execution of this loop panics out-of-bounds. Must now
/// return `OutOfSubset`, never `Proved`. Not exercised by `conform`'s
/// differential/evidence pipeline (never GPU-launched); used only by the
/// prover regression tests below.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(exact)
)]
#[cube(launch)]
pub fn stepped_loop_oob_write(x: &Array<u32>, y: &mut Array<u32>) {
    let n = y.len() as i32;
    for _i in range_stepped(n - 1, -1, -1) {
        y[100000] = x[100000];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validates the macro-derived twin against independently written scalar
    /// code — guarding the source-to-reference derivation itself.
    #[test]
    fn xorshift_twin_matches_handwritten() {
        let x: Vec<u32> = vec![1, 0xDEADBEEF, u32::MAX, 42, 0];
        let mut y = vec![0u32; x.len()];
        xorshift_step_vericl::reference(&x, &mut y, 8);
        for (i, &v) in x.iter().enumerate() {
            let mut s = v;
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            assert_eq!(y[i], s, "index {i}");
        }
    }

    /// The twin honors the guard: threads past the guard write nothing.
    #[test]
    fn axpy_twin_respects_guard() {
        let x = vec![1.0f32; 3];
        let mut y = vec![10.0f32; 3];
        axpy_vericl::reference(2.0, &x, &mut y, 256); // threads >> len
        assert_eq!(y, vec![12.0; 3]);
    }

    #[test]
    fn assumes_predicate_is_executable() {
        assert!(axpy_vericl::check_assumes(1.0, &[0.0; 4], &[0.0; 4]));
        assert!(!axpy_vericl::check_assumes(1.0, &[0.0; 4], &[0.0; 3])); // len mismatch
        assert!(!axpy_vericl::check_assumes(9.0, &[0.0; 4], &[0.0; 4])); // alpha out of range
        assert!(!axpy_vericl::check_assumes(1.0, &[500.0; 4], &[0.0; 4])); // element out of range
    }

    #[test]
    fn identity_hashes_are_distinct_per_kernel() {
        assert_ne!(axpy_vericl::SOURCE_HASH, axpy_off_by_one_vericl::SOURCE_HASH);
        assert_ne!(axpy_vericl::SOURCE_HASH, xorshift_step_vericl::SOURCE_HASH);
    }

    /// Independently written murmur3 fmix32, used only to cross-check the
    /// macro-derived `wrapping` twin below — kept deliberately separate from
    /// the kernel body so the check is not circular.
    fn fmix32(mut h: u32) -> u32 {
        h ^= h >> 16;
        h = h.wrapping_mul(0x85ebca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2ae35);
        h ^= h >> 16;
        h
    }

    /// The `wrapping` clause matters: the twin's `*`/`>>` are folded to
    /// `wrapping_mul`/`wrapping_shr`, so it must NOT panic on inputs that
    /// overflow `u32` multiplication — even though this test runs under
    /// `cargo test`'s default dev profile, which has `overflow-checks =
    /// true`. Without the fold (or with a checked `*`), this would panic on
    /// `u32::MAX` and friends; a reference panic here would be a semantics
    /// mismatch, not a kernel bug (same class as `axpy_off_by_one`, but
    /// caused by the wrong arithmetic model instead of a bounds bug).
    #[test]
    fn mix_u32_wraps_without_panicking() {
        let x: Vec<u32> = vec![u32::MAX, 0, 1, 0x9E37_79B9, 0xFFFF_FFFF, u32::MAX / 2 + 1];
        let mut y = vec![0u32; x.len()];
        mix_u32_vericl::reference(&x, &mut y, x.len()); // must not panic
    }

    /// Cross-checks the macro-derived twin against the handwritten
    /// `fmix32` above, including the murmur3 `fmix32(0) == 0` vector.
    #[test]
    fn mix_u32_twin_matches_handwritten_fmix32() {
        assert_eq!(fmix32(0), 0);

        let x: Vec<u32> = vec![0, 1, 42, 0xDEAD_BEEF, u32::MAX, 0x9E37_79B9];
        let mut y = vec![0u32; x.len()];
        mix_u32_vericl::reference(&x, &mut y, x.len());
        for (i, &v) in x.iter().enumerate() {
            assert_eq!(y[i], fmix32(v), "index {i}");
        }
    }

    /// The `wrapping` clause is part of the recorded contract.
    #[test]
    fn wrapping_is_recorded_in_the_contract() {
        assert!(mix_u32_vericl::contract().wrapping);
        assert!(!xorshift_step_vericl::contract().wrapping);
    }

    /// `assumes(x.len() == y.len())` is recognized as a structured
    /// `LenEq` assume, and `BUFFER_PARAMS` records buffer-registration order
    /// — both are what the SMT bounds prover needs and has no other way to
    /// recover from the IR alone.
    #[test]
    fn structured_assumes_and_buffer_params_are_generated() {
        assert_eq!(
            axpy_vericl::contract().structured_assumes,
            &[vericl::StructuredAssume::LenEq { a: "x", b: "y" }]
        );
        assert_eq!(axpy_vericl::BUFFER_PARAMS, &[("x", false), ("y", true)]);

        assert_eq!(
            sum_racy_vericl::contract().structured_assumes,
            &[vericl::StructuredAssume::LenEqConst { a: "y", value: 1 }]
        );
    }

    /// `kernel_definition()` builds a real `KernelDefinition` (no
    /// client/device/runtime) that the SMT bounds prover can discharge —
    /// the first machine-checked property (README "First release" outcome
    /// 3), exercised end-to-end here for the guarded, in-bounds case.
    #[test]
    fn kernel_definition_is_provably_in_bounds() {
        let def = axpy_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = axpy_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 3),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// `axpy_off_by_one`'s IR-level bounds obligation is genuinely violated
    /// (not just its differential/reference-panic check) — the SMT prover
    /// refutes it independently of the differential harness.
    #[test]
    fn off_by_one_kernel_definition_refutes() {
        let def = axpy_off_by_one_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = axpy_off_by_one_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Refuted { .. } => {}
            other => panic!("expected Refuted, got {other:?}"),
        }
    }

    /// `xorshift_step` and `mix_u32` both prove in spite of their bitwise/
    /// wrapping-integer bodies: every index used is a bare `ABSOLUTE_POS`,
    /// so the value-computation ops (outside the prover's modeled subset)
    /// never end up needed for an obligation (see vericl-ir's module docs).
    #[test]
    fn xorshift_and_mix_u32_prove_despite_bitwise_bodies() {
        for (def, buffer_params) in [
            (xorshift_step_vericl::kernel_definition(), xorshift_step_vericl::BUFFER_PARAMS),
            (mix_u32_vericl::kernel_definition(), mix_u32_vericl::BUFFER_PARAMS),
        ] {
            let buffers: Vec<vericl_ir::BufferParam> = buffer_params
                .iter()
                .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
                .collect();
            let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
            match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
                vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
                other => panic!("expected Proved, got {other:?}"),
            }
        }
    }

    /// `sum_racy`'s `y[0]` access proves given `assumes(y.len() == 1)` —
    /// the race is a differential finding, not a bounds one; the two claim
    /// kinds are cleanly separate (see README "Claims and trust
    /// boundaries").
    #[test]
    fn sum_racy_bounds_prove_independently_of_its_race() {
        let def = sum_racy_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = sum_racy_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEqConst { a: "y", value: 1 }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            // x[pos] read, y[0] read, y[0] write.
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 3),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// REGRESSION (adversarial soundness review, Bug 1): a `range_stepped`
    /// loop — here a runtime-bounded descending copy that is, by
    /// construction, entirely in-bounds — must still be rejected as
    /// `OutOfSubset`, never `Proved`. Before the fix, `process_range_loop`
    /// never read `rl.step` and unconditionally asserted ascending bounds
    /// (`start <= i < end`); for this real descending loop `start > end`
    /// numerically, so those assertions are unsatisfiable, the SMT context
    /// becomes infeasible, and every obligation inside would discharge
    /// vacuously as "proved" regardless of the body. The fix rejects any
    /// `rl.step.is_some()` outright.
    #[test]
    fn stepped_range_loop_is_out_of_subset() {
        let def = stepped_loop_descending_copy_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = stepped_loop_descending_copy_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("stepped") || reason.contains("range_stepped"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset, got {other:?}"),
        }
    }

    /// REGRESSION (adversarial soundness review, Bug 1): the exact
    /// vacuous-proof shape the review demonstrated — a runtime-bounded,
    /// negative-step loop whose body writes/reads `[100000]`, far outside
    /// any plausible buffer length. Before the fix this returned
    /// `Proved { obligations: 2 }` (the SMT context was infeasible, so both
    /// the `x[100000]` read and `y[100000]` write discharged vacuously)
    /// even though a real sequential execution of this loop panics
    /// out-of-bounds — a false soundness claim. Must now return
    /// `OutOfSubset`.
    #[test]
    fn stepped_loop_cannot_vacuously_prove() {
        let def = stepped_loop_oob_write_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = stepped_loop_oob_write_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("stepped") || reason.contains("range_stepped"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (not Proved — see doc comment), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // instantiate(...): generic + #[comptime] monomorphization (fir3).
    // -----------------------------------------------------------------

    /// Independently written 3-tap (at most) windowed sum, used only to
    /// cross-check the macro-derived, instantiate(...)-monomorphized twin —
    /// kept deliberately separate from the kernel body so the check is not
    /// circular. Mirrors `fmix32` above for `mix_u32`.
    fn fir_handwritten(x: &[f32], taps: u32) -> Vec<f32> {
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                let mut acc = v;
                if taps > 1 && i >= 1 {
                    acc += x[i - 1];
                }
                if taps > 2 && i >= 2 {
                    acc += x[i - 2];
                }
                acc
            })
            .collect()
    }

    /// `fir3`'s twin (F -> f32, `#[comptime] taps` removed from the
    /// signature and bound as a `let taps: u32 = 3;` const per
    /// `instantiate(F = f32, taps = 3)`) matches the independent
    /// hand-written computation above — the same guard against a circular
    /// derivation check as `xorshift_twin_matches_handwritten`/
    /// `mix_u32_twin_matches_handwritten_fmix32`.
    #[test]
    fn fir3_twin_matches_handwritten() {
        let x: Vec<f32> = vec![1.0, -2.0, 3.5, 0.25, -7.0, 10.0, 0.0, -1.5];
        let mut y = vec![0.0f32; x.len()];
        // reference() takes no `taps` parameter — it's a const now.
        fir3_vericl::reference(&x, &mut y, x.len());
        let expected = fir_handwritten(&x, 3);
        for (i, (&got, &want)) in y.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "index {i}: got {got}, want {want}");
        }
    }

    /// The twin honors the guard exactly like `axpy`'s: threads past the
    /// guard write nothing to `y`.
    #[test]
    fn fir3_twin_respects_guard() {
        let x = vec![1.0f32; 3];
        let mut y = vec![9.0f32; 3];
        fir3_vericl::reference(&x, &mut y, 256); // threads >> len
        assert_eq!(y, vec![1.0, 2.0, 3.0]); // taps=3: x[0], x[1]+x[0], x[2]+x[1]+x[0]
    }

    /// `instantiate(...)`'s pinned values are part of the recorded contract
    /// (`Contract::instantiate`/`ContractRecord::instantiate`), in clause
    /// declaration order — separate from `assumes`/`wrapping`, mirroring how
    /// `wrapping_is_recorded_in_the_contract` pins that clause's field.
    #[test]
    fn instantiate_is_recorded_in_the_contract() {
        assert_eq!(fir3_vericl::contract().instantiate, &["F = f32", "taps = 3"]);
        assert_eq!(fir3_alt_vericl::contract().instantiate, &["F = f32", "taps = 1"]);
        // A non-generic, non-comptime kernel has an empty instantiate list.
        assert!(axpy_off_by_one_vericl::contract().instantiate.is_empty());
    }

    /// Changing an `instantiate(...)` value changes kernel identity: `fir3`
    /// and `fir3_alt` are byte-identical source except for the pinned
    /// `taps` value, and the instantiation value is part of the raw
    /// contract attribute tokens `SOURCE_HASH` covers — so the two hashes
    /// must differ. This is the source-level counterpart to
    /// `identity_hashes_are_distinct_per_kernel` above, specifically for
    /// the instantiate(...) clause.
    #[test]
    fn fir3_identity_changes_with_instantiate_value() {
        assert_ne!(fir3_vericl::SOURCE_HASH, fir3_alt_vericl::SOURCE_HASH);
    }

    /// The milestone's headline result: `fir3` is genuinely generic
    /// (`F: Float`) *and* has a `#[comptime]` parameter, monomorphized via
    /// `instantiate(F = f32, taps = 3)`, *and* uses `&&`-composed guards
    /// (`taps > 1 && ABSOLUTE_POS >= 1`) — and its bounds obligations still
    /// discharge as `Proved`, not merely `OutOfSubset`. Achieved by (a)
    /// avoiding a loop-carried accumulator (see the kernel's doc comment):
    /// each extra tap is its own guarded condition, the same shape the
    /// prover already handles for `axpy`/`sum_racy`; and (b) boolean
    /// condition composition (`vericl-ir`'s `Operator::And` modeling).
    #[test]
    fn fir3_kernel_definition_is_provably_in_bounds() {
        let def = fir3_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = fir3_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            // x[pos] read, y[pos] write, guarded x[pos-1]/x[pos-2] reads.
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 4),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // flatten_decode_scale: the div/mod prover-expansion milestone kernel.
    // -----------------------------------------------------------------

    /// The twin honors the guard: threads past `y.len()` write nothing.
    #[test]
    fn flatten_decode_scale_twin_respects_guard() {
        let x = vec![1.0f32; 4];
        let mut y = vec![9.0f32; 4];
        flatten_decode_scale_vericl::reference(&x, &mut y, 2, 3.0, 256); // threads >> len
        // pos 0: row=0,col=0 -> idx 0; pos 1: row=0,col=1 -> idx 1;
        // pos 2: row=1,col=0 -> idx 2; pos 3: row=1,col=1 -> idx 3 (idx ==
        // pos throughout, by construction of the row/col decode/recombine).
        assert_eq!(y, vec![3.0, 3.0, 3.0, 3.0]);
    }

    /// Independently computes the same row/col decode + scale via plain
    /// integer division/remainder (not derived from the kernel body), to
    /// cross-check the macro-derived twin — same guard against a circular
    /// derivation check as `fir_handwritten`/`fmix32` above.
    #[test]
    fn flatten_decode_scale_twin_matches_handwritten() {
        let x: Vec<f32> = (0..12).map(|i| i as f32 - 5.0).collect();
        let mut y = vec![0.0f32; x.len()];
        let width = 4usize;
        let scale = 2.5f32;
        flatten_decode_scale_vericl::reference(&x, &mut y, width as u32, scale, x.len());
        for (pos, (&xv, &yv)) in x.iter().zip(y.iter()).enumerate() {
            let row = pos / width;
            let col = pos % width;
            let idx = row * width + col;
            assert_eq!(idx, pos, "row/col recombine should be the identity at pos {pos}");
            assert_eq!(yv, xv * scale, "index {pos}");
        }
    }

    /// `assumes(x.len() == y.len())` is still the only structured assume —
    /// `width`'s nonzero-ness is established by the kernel's own `width >=
    /// 1u32` runtime guard (a path condition), not a declared assume; see
    /// the kernel's doc comment.
    #[test]
    fn flatten_decode_scale_structured_assumes() {
        assert_eq!(
            flatten_decode_scale_vericl::contract().structured_assumes,
            &[vericl::StructuredAssume::LenEq { a: "x", b: "y" }]
        );
    }

    /// The div/mod milestone's headline result: the write index is the
    /// *recombined* `row * width + col`, not a bare `ABSOLUTE_POS` — proving
    /// it in bounds requires the SMT solver to derive `row * width + col ==
    /// ABSOLUTE_POS` from the `div`/`mod` (Euclidean) axioms and combine
    /// that with the `ABSOLUTE_POS < y.len()` guard. Before div/mod
    /// modeling, `Arithmetic::Div`/`Modulo` tainted their output, `row`/
    /// `col` were unbound, and this was `OutOfSubset` at the write index.
    #[test]
    fn flatten_decode_scale_kernel_definition_is_provably_in_bounds() {
        let def = flatten_decode_scale_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = flatten_decode_scale_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            // x[pos] read, y[row*width+col] write.
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Kernel composition: #[vericl::helper] + uses(...).
    // -----------------------------------------------------------------

    /// The `single_tap`/`fir_pair` helper twins compute exactly what their
    /// (very simple) bodies say — guards against the twin derivation itself
    /// going wrong for a helper, same purpose as `fir_handwritten`/`fmix32`
    /// above for kernels.
    #[test]
    fn helper_twins_match_hand_computed() {
        assert_eq!(single_tap_vericl_ref(3.0f32, 2.0f32), 6.0);
        assert_eq!(fir_pair_vericl_ref(5.0f32, 2.0f32), (7.0, 3.0));
    }

    /// `fir_pair_scaled` calls two OTHER helpers (`fir_pair`, `single_tap`)
    /// — its twin must genuinely compose their `_vericl_ref` twins, not
    /// silently fall back to something else. Cross-checked two ways: against
    /// a hand-composed value, and against literally calling the other two
    /// twins directly and combining them the same way the helper's own body
    /// does — the latter is only a meaningful check because it's the exact
    /// call chain `uses(fir_pair, single_tap)`'s rewrite is supposed to
    /// produce.
    #[test]
    fn fir_pair_scaled_twin_composes_its_own_helpers() {
        let (a, b, gain) = (5.0f32, 2.0f32, 10.0f32);
        let got = fir_pair_scaled_vericl_ref(a, b, gain);

        let (sum, diff) = fir_pair_vericl_ref(a, b);
        let expected = (single_tap_vericl_ref(sum, gain), single_tap_vericl_ref(diff, gain));
        assert_eq!(got, expected);
        assert_eq!(got, (70.0, 30.0));
    }

    /// `tap_pair`'s twin reads its own two elements — the shape the
    /// composition-prover tests below rely on.
    #[test]
    fn tap_pair_twin_matches_hand_computed() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(tap_pair_vericl_ref(&x, 0), 3.0);
        assert_eq!(tap_pair_vericl_ref(&x, 2), 7.0);
    }

    /// `gain_kernel`'s twin honors its guard and matches a hand computation
    /// that never goes through `single_tap_vericl_ref` at all — guards
    /// against the composed kernel's *own* derivation (ABSOLUTE_POS rewrite,
    /// instantiate(...) substitution, uses(...) rewrite all combined) being
    /// wrong in a way an isolated helper-twin test wouldn't catch.
    #[test]
    fn gain_kernel_twin_matches_hand_computed() {
        let x = vec![1.0f32, -2.0, 3.0];
        let mut y = vec![0.0f32; x.len()];
        gain_kernel_vericl::reference(&x, &mut y, 2.0, x.len());
        assert_eq!(y, vec![2.0, -4.0, 6.0]);
    }

    /// Threads past the guard write nothing — same discipline as every
    /// other kernel's twin.
    #[test]
    fn gain_kernel_twin_respects_guard() {
        let x = vec![1.0f32; 3];
        let mut y = vec![9.0f32; 3];
        gain_kernel_vericl::reference(&x, &mut y, 2.0, 256);
        assert_eq!(y, vec![2.0, 2.0, 2.0]);
    }

    /// Fix 3 regression pin: `self_path_gain_kernel`'s twin (`self::`-
    /// qualified `single_tap` call) must produce byte-identical results to
    /// `gain_kernel`'s (bare call) — the same expected values as
    /// `gain_kernel_twin_matches_hand_computed` above, for the same inputs.
    /// Pre-fix, this would have *coincidentally* still passed (the
    /// bypassed, un-rewritten path called the original `#[cube] fn
    /// single_tap`, which is host-safe and computes the same thing as
    /// `single_tap_vericl_ref`) — the differential can't distinguish the
    /// two; see `uses_rewrite_fold_rewrites_self_qualified_helper_call` in
    /// `vericl-macros` for the AST-level pin that actually catches the
    /// bypass.
    #[test]
    fn self_path_gain_kernel_twin_matches_hand_computed() {
        let x = vec![1.0f32, -2.0, 3.0];
        let mut y = vec![0.0f32; x.len()];
        self_path_gain_kernel_vericl::reference(&x, &mut y, 2.0, x.len());
        assert_eq!(y, vec![2.0, -4.0, 6.0]);
    }

    /// Fix 3 regression pin: `self_path_gain_kernel`'s bounds proof
    /// discharges identically to `gain_kernel`'s (same obligation count) —
    /// the `self::`-qualified call composes through the prover exactly like
    /// the bare call, since both inline the same helper IR into the same
    /// kernel `Scope` either way.
    #[test]
    fn self_path_gain_kernel_definition_is_provably_in_bounds() {
        let def = self_path_gain_kernel_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = self_path_gain_kernel_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// `fir_pair_kernel`'s twin (two-level composition: kernel ->
    /// `fir_pair_scaled` -> `fir_pair`/`single_tap`, two `&mut Array`
    /// outputs) matches an independent hand computation.
    #[test]
    fn fir_pair_kernel_twin_matches_hand_computed() {
        let x = vec![1.0f32, 3.0, -2.0, 5.0];
        let mut sum_out = vec![0.0f32; x.len()];
        let mut diff_out = vec![0.0f32; x.len()];
        let gain = 2.0f32;
        fir_pair_kernel_vericl::reference(&x, &mut sum_out, &mut diff_out, gain, x.len());
        // Guard is ABSOLUTE_POS + 1 < x.len(), so only positions 0..len-2
        // write; last position(s) stay at their initial value.
        for pos in 0..x.len() - 2 {
            let (a, b) = (x[pos], x[pos + 1]);
            assert_eq!(sum_out[pos], (a + b) * gain, "sum_out[{pos}]");
            assert_eq!(diff_out[pos], (a - b) * gain, "diff_out[{pos}]");
        }
        assert_eq!(sum_out[x.len() - 1], 0.0);
        assert_eq!(diff_out[x.len() - 1], 0.0);
    }

    /// `uses(...)` is recorded on the contract, and `tap_pair` is reused
    /// verbatim by two different kernels (the milestone's explicit "two
    /// kernels using the same helper" ask) — both list exactly `["tap_pair"]`.
    #[test]
    fn uses_clause_is_recorded_and_helper_is_reused_by_two_kernels() {
        assert_eq!(gain_kernel_vericl::contract().uses, &["single_tap"]);
        assert_eq!(fir_pair_kernel_vericl::contract().uses, &["fir_pair_scaled"]);
        assert_eq!(tap_pair_guarded_kernel_vericl::contract().uses, &["tap_pair"]);
        assert_eq!(tap_pair_unguarded_kernel_vericl::contract().uses, &["tap_pair"]);
        // A non-composing kernel's uses() is empty.
        assert!(axpy_vericl::contract().uses.is_empty());
        // A helper composing two other helpers records both, in clause order.
        assert_eq!(fir_pair_scaled_vericl::USES, &["fir_pair", "single_tap"]);
    }

    // -----------------------------------------------------------------
    // Composition identity: uses(...) must make a helper body change
    // visible in the composing kernel's (and helper's) recorded identity,
    // without leaking into anything that doesn't use() it.
    // -----------------------------------------------------------------

    /// A composing kernel's *recorded* identity (`identity()`) is NOT the
    /// same as its own compile-time-only `SOURCE_HASH` — composition
    /// genuinely changes what gets recorded, and does so by exactly folding
    /// in the declared helper's own `identity_hash()` (verified by
    /// reproducing the combine independently via
    /// `vericl::combine_source_hash` and asserting byte-for-byte equality,
    /// not just "differs from something").
    #[test]
    fn composed_kernel_identity_folds_in_its_helpers_hash() {
        let recorded = gain_kernel_vericl::identity().source_hash;
        assert_ne!(recorded, gain_kernel_vericl::SOURCE_HASH);
        let expected = vericl::combine_source_hash(
            gain_kernel_vericl::SOURCE_HASH,
            &[single_tap_vericl::identity_hash()],
        );
        assert_eq!(recorded, expected);
    }

    /// A NON-composing kernel's recorded identity is an exact pass-through
    /// of its own `SOURCE_HASH` — proving `identity()` folds in exactly the
    /// `uses(...)`-declared set and nothing else, regardless of how many
    /// unrelated helpers exist elsewhere in this same crate. This is the
    /// "changing a helper's unused sibling changes neither hash" guarantee,
    /// checked structurally (identity() provably can't see an undeclared
    /// helper at all) rather than by an actual source edit + rebuild —
    /// which was additionally exercised by hand (not committed; see the
    /// task's verification report) by editing `single_tap`'s body and
    /// confirming `gain_kernel`'s identity()/ir_hash moved while
    /// `axpy`'s/`flatten_decode_scale`'s did not, then reverting.
    #[test]
    fn unused_helper_does_not_affect_an_unrelated_kernels_identity() {
        assert_eq!(axpy_vericl::identity().source_hash, axpy_vericl::SOURCE_HASH);
        assert_eq!(
            flatten_decode_scale_vericl::identity().source_hash,
            flatten_decode_scale_vericl::SOURCE_HASH,
        );
    }

    /// Helper-calling-helper: `fir_pair_scaled`'s OWN `identity_hash()`
    /// recursively folds in both `fir_pair`'s and `single_tap`'s hashes —
    /// verified by reproducing the combine independently, exactly as for
    /// the kernel-level test above.
    #[test]
    fn helper_calling_helper_identity_is_recursive() {
        let recorded = fir_pair_scaled_vericl::identity_hash();
        assert_ne!(recorded, fir_pair_scaled_vericl::SOURCE_HASH);
        let expected = vericl::combine_source_hash(
            fir_pair_scaled_vericl::SOURCE_HASH,
            &[fir_pair_vericl::identity_hash(), single_tap_vericl::identity_hash()],
        );
        assert_eq!(recorded, expected);
    }

    /// The two-level composition case (`fir_pair_kernel` -> `fir_pair_scaled`
    /// -> `fir_pair`/`single_tap`): the KERNEL's own recorded identity only
    /// ever combines with its *direct* dependency's already-recursive
    /// `identity_hash()` — it never needs to know about `fir_pair`/
    /// `single_tap` by name at all, yet a change two levels deep still
    /// reaches it, because `fir_pair_scaled_vericl::identity_hash()` (used
    /// here) already covers its own `uses(...)` the same way (pinned by
    /// `helper_calling_helper_identity_is_recursive` above).
    #[test]
    fn composed_kernel_identity_is_recursive_through_the_helper_chain() {
        let recorded = fir_pair_kernel_vericl::identity().source_hash;
        let expected = vericl::combine_source_hash(
            fir_pair_kernel_vericl::SOURCE_HASH,
            &[fir_pair_scaled_vericl::identity_hash()],
        );
        assert_eq!(recorded, expected);
    }

    // -----------------------------------------------------------------
    // Composition + the SMT bounds prover (README claim: composition needs
    // zero prover changes, since cube expansion inlines a used helper's IR
    // directly into the composing kernel's own Scope).
    // -----------------------------------------------------------------

    /// Positive control: `tap_pair`'s own `x[idx]`/`x[idx + 1]` reads live
    /// entirely inside the composed helper's body, not the kernel's — and
    /// the guard `ABSOLUTE_POS + 1 < x.len()` the KERNEL establishes still
    /// gets combined with them correctly. Must discharge `Proved` — this is
    /// the milestone's positive composition-prover result.
    #[test]
    fn tap_pair_guarded_kernel_definition_is_provably_in_bounds() {
        let def = tap_pair_guarded_kernel_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = tap_pair_guarded_kernel_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Proved { .. } => {}
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// Negative control: same helper, same kernel shape, but the guard only
    /// covers `ABSOLUTE_POS < x.len()` — one short of what `tap_pair`'s own
    /// unguarded `x[idx + 1]` read needs at the top position. Must
    /// `Refuted`, proving the obligation from inside the helper's body is
    /// genuinely walked and not silently dropped because it's composed
    /// rather than written directly in the kernel — the milestone's
    /// negative composition-prover result.
    #[test]
    fn tap_pair_unguarded_kernel_definition_refutes() {
        let def = tap_pair_unguarded_kernel_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = tap_pair_unguarded_kernel_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Refuted { .. } => {}
            other => panic!("expected Refuted, got {other:?}"),
        }
    }

    /// The suite-wired composed kernels also prove — carrying both tested
    /// (differential, via `vericl::suite!`) and proved claims is the
    /// milestone's "composed kernel carries tested + proved claims" ask.
    #[test]
    fn suite_wired_composed_kernels_prove() {
        let gain_buffers: Vec<vericl_ir::BufferParam> = gain_kernel_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        match vericl_ir::prove_bounds_freedom(
            &gain_kernel_vericl::kernel_definition(),
            &gain_buffers,
            &[vericl_ir::Assume::LenEq { a: "x", b: "y" }],
        ) {
            vericl_ir::ProveResult::Proved { .. } => {}
            other => panic!("expected gain_kernel to prove, got {other:?}"),
        }

        let fir_buffers: Vec<vericl_ir::BufferParam> = fir_pair_kernel_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        match vericl_ir::prove_bounds_freedom(
            &fir_pair_kernel_vericl::kernel_definition(),
            &fir_buffers,
            &[
                vericl_ir::Assume::LenEq { a: "x", b: "sum_out" },
                vericl_ir::Assume::LenEq { a: "x", b: "diff_out" },
            ],
        ) {
            vericl_ir::ProveResult::Proved { .. } => {}
            other => panic!("expected fir_pair_kernel to prove, got {other:?}"),
        }
    }
}
