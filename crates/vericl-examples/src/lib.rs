//! Example kernels for the vericl first release.
//!
//! Two honest kernels (one float with a declared tolerance, one integer with
//! exact comparison) and two deliberately defective twins whose defects the
//! differential check must catch (README outcome 4).

use cubecl::prelude::*;

/// f32 saxpy.
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
    gen(alpha in -4.0..=4.0, x in -100.0..=100.0, y in -100.0..=100.0)
)]
#[cube(launch)]
pub fn axpy(alpha: f32, x: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
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
}
