//! Output comparison between a reference execution and a realization under test.

use serde::{Deserialize, Serialize};

use crate::contract::Compare;

/// Result of comparing two output buffers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompareReport {
    /// `true` iff every element was within tolerance (`mismatches == 0`).
    pub pass: bool,
    /// Number of elements compared.
    pub checked: usize,
    /// Number of elements outside the declared tolerance.
    pub mismatches: usize,
    /// Worst observed f32 ULP distance (f32 comparisons only, mismatched or not).
    pub max_ulp: Option<u64>,
    /// First (or worst, for f32) mismatch, for diagnostics.
    pub worst: Option<Mismatch>,
}

/// A single element where two buffers diverged, for diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Mismatch {
    /// Element index at which the divergence occurred.
    pub index: usize,
    /// The reference (expected) value, widened to f64.
    pub expected: f64,
    /// The observed (actual) value, widened to f64.
    pub actual: f64,
    /// ULP distance; `None` when NaN is involved or comparison is integral.
    pub ulp: Option<u64>,
}

/// Monotonic integer mapping of f32 for ULP distance. Adjacent floats map to
/// adjacent integers; +0.0 and -0.0 coincide.
fn ordered_f32(x: f32) -> i64 {
    let b = x.to_bits() as i64;
    if (b >> 31) & 1 == 1 { 0x8000_0000 - b } else { b }
}

/// ULP distance between two f32 values. `None` if either is NaN.
pub fn ulp_distance_f32(a: f32, b: f32) -> Option<u64> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    Some((ordered_f32(a) - ordered_f32(b)).unsigned_abs())
}

/// Compare f32 buffers under a declared maximum ULP tolerance.
pub fn compare_f32(expected: &[f32], actual: &[f32], max_ulp: u32) -> CompareReport {
    assert_eq!(expected.len(), actual.len(), "buffer length mismatch");
    let mut mismatches = 0usize;
    let mut observed_max: u64 = 0;
    let mut worst: Option<Mismatch> = None;

    for (i, (&e, &a)) in expected.iter().zip(actual).enumerate() {
        match ulp_distance_f32(e, a) {
            Some(d) => {
                observed_max = observed_max.max(d);
                if d > max_ulp as u64 {
                    mismatches += 1;
                    if worst.as_ref().is_none_or(|w| Some(d) > w.ulp) {
                        worst = Some(Mismatch {
                            index: i,
                            expected: e as f64,
                            actual: a as f64,
                            ulp: Some(d),
                        });
                    }
                }
            }
            None => {
                mismatches += 1;
                if worst.is_none() {
                    worst = Some(Mismatch {
                        index: i,
                        expected: e as f64,
                        actual: a as f64,
                        ulp: None,
                    });
                }
            }
        }
    }

    CompareReport {
        pass: mismatches == 0,
        checked: expected.len(),
        mismatches,
        max_ulp: Some(observed_max),
        worst,
    }
}

/// Compare f32 buffers under `|expected - actual| <= abs + rel * |expected|`.
/// NaN on either side is a failure. ULP distance is still recorded for
/// diagnostics.
pub fn compare_f32_absrel(expected: &[f32], actual: &[f32], abs: f32, rel: f32) -> CompareReport {
    assert_eq!(expected.len(), actual.len(), "buffer length mismatch");
    let mut mismatches = 0usize;
    let mut observed_max_ulp: u64 = 0;
    let mut worst: Option<(f32, Mismatch)> = None; // keyed by excess over bound

    for (i, (&e, &a)) in expected.iter().zip(actual).enumerate() {
        if let Some(d) = ulp_distance_f32(e, a) {
            observed_max_ulp = observed_max_ulp.max(d);
        }
        let bound = abs + rel * e.abs();
        let diff = (e - a).abs();
        // NaN anywhere (including inf - inf) must fail, so test diff for NaN
        // explicitly rather than relying on comparison direction.
        let fails = e.is_nan() || a.is_nan() || diff.is_nan() || diff > bound;
        if fails {
            mismatches += 1;
            let excess = if diff.is_nan() { f32::INFINITY } else { diff - bound };
            if worst.as_ref().is_none_or(|(w, _)| excess > *w) {
                worst = Some((
                    excess,
                    Mismatch {
                        index: i,
                        expected: e as f64,
                        actual: a as f64,
                        ulp: ulp_distance_f32(e, a),
                    },
                ));
            }
        }
    }

    CompareReport {
        pass: mismatches == 0,
        checked: expected.len(),
        mismatches,
        max_ulp: Some(observed_max_ulp),
        worst: worst.map(|(_, m)| m),
    }
}

/// Monotonic integer mapping of f64 for ULP distance. Adjacent floats map to
/// adjacent integers; +0.0 and -0.0 coincide. Uses `i128` because the f64 bit
/// pattern is 64-bit — the sign-flip (`2^63 - b`) and the subtraction of two
/// ordered values would overflow `i64`.
fn ordered_f64(x: f64) -> i128 {
    let b = x.to_bits() as i128;
    if (b >> 63) & 1 == 1 { (1i128 << 63) - b } else { b }
}

/// ULP distance between two f64 values. `None` if either is NaN. The distance
/// is saturated to `u64::MAX` (the full f64 range spans ~2^64 ordered values,
/// marginally more than `u64` can hold) — a saturated ULP distance is only
/// ever reported for values already astronomically far apart, well past any
/// tolerance a `max_ulp` contract would declare.
pub fn ulp_distance_f64(a: f64, b: f64) -> Option<u64> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    let d = (ordered_f64(a) - ordered_f64(b)).unsigned_abs();
    Some(u64::try_from(d).unwrap_or(u64::MAX))
}

/// Compare f64 buffers under a declared maximum ULP tolerance. Mirror of
/// [`compare_f32`]: NaN on either side is always a failure, and the worst
/// observed ULP distance is recorded for diagnostics.
pub fn compare_f64(expected: &[f64], actual: &[f64], max_ulp: u32) -> CompareReport {
    assert_eq!(expected.len(), actual.len(), "buffer length mismatch");
    let mut mismatches = 0usize;
    let mut observed_max: u64 = 0;
    let mut worst: Option<Mismatch> = None;

    for (i, (&e, &a)) in expected.iter().zip(actual).enumerate() {
        match ulp_distance_f64(e, a) {
            Some(d) => {
                observed_max = observed_max.max(d);
                if d > max_ulp as u64 {
                    mismatches += 1;
                    if worst.as_ref().is_none_or(|w| Some(d) > w.ulp) {
                        worst = Some(Mismatch { index: i, expected: e, actual: a, ulp: Some(d) });
                    }
                }
            }
            None => {
                mismatches += 1;
                if worst.is_none() {
                    worst = Some(Mismatch { index: i, expected: e, actual: a, ulp: None });
                }
            }
        }
    }

    CompareReport {
        pass: mismatches == 0,
        checked: expected.len(),
        mismatches,
        max_ulp: Some(observed_max),
        worst,
    }
}

/// Compare f64 buffers under `|expected - actual| <= abs + rel * |expected|`.
/// Mirror of [`compare_f32_absrel`]: NaN on either side (including `inf - inf`)
/// is a failure, tested explicitly rather than via comparison direction. ULP
/// distance is still recorded for diagnostics.
pub fn compare_f64_absrel(expected: &[f64], actual: &[f64], abs: f64, rel: f64) -> CompareReport {
    assert_eq!(expected.len(), actual.len(), "buffer length mismatch");
    let mut mismatches = 0usize;
    let mut observed_max_ulp: u64 = 0;
    let mut worst: Option<(f64, Mismatch)> = None; // keyed by excess over bound

    for (i, (&e, &a)) in expected.iter().zip(actual).enumerate() {
        if let Some(d) = ulp_distance_f64(e, a) {
            observed_max_ulp = observed_max_ulp.max(d);
        }
        let bound = abs + rel * e.abs();
        let diff = (e - a).abs();
        // NaN anywhere (including inf - inf) must fail, so test diff for NaN
        // explicitly rather than relying on comparison direction.
        let fails = e.is_nan() || a.is_nan() || diff.is_nan() || diff > bound;
        if fails {
            mismatches += 1;
            let excess = if diff.is_nan() { f64::INFINITY } else { diff - bound };
            if worst.as_ref().is_none_or(|(w, _)| excess > *w) {
                worst = Some((
                    excess,
                    Mismatch { index: i, expected: e, actual: a, ulp: ulp_distance_f64(e, a) },
                ));
            }
        }
    }

    CompareReport {
        pass: mismatches == 0,
        checked: expected.len(),
        mismatches,
        max_ulp: Some(observed_max_ulp),
        worst: worst.map(|(_, m)| m),
    }
}

/// Bit-exact comparison for u32 buffers.
pub fn compare_exact_u32(expected: &[u32], actual: &[u32]) -> CompareReport {
    assert_eq!(expected.len(), actual.len(), "buffer length mismatch");
    let mut mismatches = 0usize;
    let mut worst: Option<Mismatch> = None;
    for (i, (&e, &a)) in expected.iter().zip(actual).enumerate() {
        if e != a {
            mismatches += 1;
            if worst.is_none() {
                worst = Some(Mismatch {
                    index: i,
                    expected: e as f64,
                    actual: a as f64,
                    ulp: None,
                });
            }
        }
    }
    CompareReport {
        pass: mismatches == 0,
        checked: expected.len(),
        mismatches,
        max_ulp: None,
        worst,
    }
}

/// Dispatch a declared [`Compare`] mode against an `f32` buffer pair. Used by
/// the macro-generated `conformance_case` (one call per compared `&mut
/// Array<f32>` parameter), which knows the element type at expansion time
/// but not which `compare(...)` mode the author declared.
///
/// Panics if `compare` is [`Compare::Exact`] — that mode is for integer
/// kernels; an f32 array needs `max_ulp` or `abs`/`rel`. This is a contract-
/// authoring bug caught the first time the kernel's evidence is generated,
/// not a runtime data problem.
#[doc(hidden)] // generated-code plumbing (macro-emitted conformance_case)
pub fn compare_f32_with(compare: Compare, expected: &[f32], actual: &[f32]) -> CompareReport {
    match compare {
        Compare::MaxUlpF32(max_ulp) => compare_f32(expected, actual, max_ulp),
        Compare::AbsRelF32 { abs, rel } => compare_f32_absrel(expected, actual, abs, rel),
        Compare::Exact => panic!(
            "compare(exact) is for integer kernels; an f32 array needs `max_ulp = N` or \
             `abs = X[, rel = Y]`"
        ),
        Compare::MaxUlpF64(_) | Compare::AbsRelF64 { .. } => panic!(
            "internal error: an f64 compare mode ({}) reached an f32 array — the macro should \
             have emitted the f32 compare variant for an f32 kernel",
            compare.describe()
        ),
    }
}

/// Dispatch a declared [`Compare`] mode against a `u32` buffer pair. See
/// [`compare_f32_with`]; panics if `compare` is not [`Compare::Exact`].
#[doc(hidden)] // generated-code plumbing (macro-emitted conformance_case)
pub fn compare_u32_with(compare: Compare, expected: &[u32], actual: &[u32]) -> CompareReport {
    match compare {
        Compare::Exact => compare_exact_u32(expected, actual),
        other => panic!(
            "compare({}) is for float kernels; a u32 array only supports compare(exact)",
            other.describe()
        ),
    }
}

/// Dispatch a declared [`Compare`] mode against an `f64` buffer pair — the
/// f64 counterpart of [`compare_f32_with`], selected by the macro-generated
/// `conformance_case` when the compared `&mut Array<f64>` element type is f64
/// (the macro emits the `*F64` compare variants for an f64 kernel, so this
/// never legitimately sees a `*F32` mode).
///
/// Panics on [`Compare::Exact`] (integer mode) or on an f32 compare variant
/// reaching an f64 kernel — both are contract-authoring bugs caught the first
/// time evidence is generated, not runtime data problems.
#[doc(hidden)] // generated-code plumbing (macro-emitted conformance_case)
pub fn compare_f64_with(compare: Compare, expected: &[f64], actual: &[f64]) -> CompareReport {
    match compare {
        Compare::MaxUlpF64(max_ulp) => compare_f64(expected, actual, max_ulp),
        Compare::AbsRelF64 { abs, rel } => compare_f64_absrel(expected, actual, abs, rel),
        Compare::Exact => panic!(
            "compare(exact) is for integer kernels; an f64 array needs `max_ulp = N` or \
             `abs = X[, rel = Y]`"
        ),
        Compare::MaxUlpF32(_) | Compare::AbsRelF32 { .. } => panic!(
            "internal error: an f32 compare mode ({}) reached an f64 array — the macro should \
             have emitted the f64 compare variant for an f64 kernel",
            compare.describe()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_basics() {
        assert_eq!(ulp_distance_f32(1.0, 1.0), Some(0));
        assert_eq!(ulp_distance_f32(0.0, -0.0), Some(0));
        assert_eq!(ulp_distance_f32(1.0, f32::from_bits(1.0f32.to_bits() + 1)), Some(1));
        // across zero: smallest positive vs smallest negative subnormal = 2 ulps
        assert_eq!(
            ulp_distance_f32(f32::from_bits(1), -f32::from_bits(1)),
            Some(2)
        );
        assert_eq!(ulp_distance_f32(f32::NAN, 1.0), None);
    }

    #[test]
    fn compare_reports() {
        let r = compare_f32(&[1.0, 2.0], &[1.0, 2.0], 0);
        assert!(r.pass);
        let r = compare_f32(&[1.0], &[f32::from_bits(1.0f32.to_bits() + 2)], 1);
        assert!(!r.pass);
        assert_eq!(r.worst.unwrap().ulp, Some(2));
        let r = compare_exact_u32(&[1, 2, 3], &[1, 9, 3]);
        assert!(!r.pass);
        assert_eq!(r.worst.unwrap().index, 1);
    }

    #[test]
    fn ulp_basics_f64() {
        assert_eq!(ulp_distance_f64(1.0, 1.0), Some(0));
        assert_eq!(ulp_distance_f64(0.0, -0.0), Some(0));
        assert_eq!(ulp_distance_f64(1.0, f64::from_bits(1.0f64.to_bits() + 1)), Some(1));
        // across zero: smallest positive vs smallest negative subnormal = 2 ulps
        assert_eq!(ulp_distance_f64(f64::from_bits(1), -f64::from_bits(1)), Some(2));
        assert_eq!(ulp_distance_f64(f64::NAN, 1.0), None);
    }

    #[test]
    fn compare_reports_f64() {
        let r = compare_f64(&[1.0, 2.0], &[1.0, 2.0], 0);
        assert!(r.pass);
        let r = compare_f64(&[1.0], &[f64::from_bits(1.0f64.to_bits() + 2)], 1);
        assert!(!r.pass);
        assert_eq!(r.worst.unwrap().ulp, Some(2));
    }

    /// f64's absrel mirror of the f32 edges: NaN and `inf - inf` both fail even
    /// when a naive `diff > bound` comparison would be `false` for a NaN diff.
    #[test]
    fn absrel_f64_nan_and_inf_edges() {
        // exact match passes with a tiny bound
        let r = compare_f64_absrel(&[1.0, -3.5], &[1.0, -3.5], 1e-12, 0.0);
        assert!(r.pass);
        // NaN on either side fails
        assert!(!compare_f64_absrel(&[f64::NAN], &[1.0], 1e9, 1.0).pass);
        assert!(!compare_f64_absrel(&[1.0], &[f64::NAN], 1e9, 1.0).pass);
        // inf - inf is NaN and must fail regardless of how large the bound is
        assert!(!compare_f64_absrel(&[f64::INFINITY], &[f64::INFINITY], f64::MAX, 1.0).pass);
        // a real f64-scale rounding difference is within a justified abs bound
        let e = 400.0f64;
        let a = e + 5e-14; // ~ ulp(400) in f64
        assert!(compare_f64_absrel(&[e], &[a], 1e-12, 0.0).pass);
    }
}
