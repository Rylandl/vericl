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
}
