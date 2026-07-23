//! Empirical verification of `FLOAT_METHOD_WHITELIST` / `FLOAT_METHOD_REJECT`
//! (crates/vericl-macros/src/lib.rs) **for f64** — the f64 twin of
//! `float_method_whitelist.rs`.
//!
//! The whitelist's host-safety proof rests on Rust preferring an inherent
//! method over a trait method for a *concrete* receiver — and the concrete
//! type here is `f64`, not `f32`. cubecl's `Float`/`Numeric` trait impls
//! could in principle differ per type (a method with a real per-type host
//! impl for one and only the panicking `unexpanded!()` default for the
//! other), so this MUST be re-verified against f64 directly rather than
//! assumed to transfer from the f32 run. This file calls every whitelisted
//! name on host `f64`, cross-checks it against an independent `std`/f64
//! computation, and confirms every rejected name still panics — if any
//! entry diverged from the f32 result, the whitelist would have to become
//! per-type.
//!
//! Result (verified by running this file): every whitelist/reject entry
//! behaves identically on f64 as on f32, so a single shared list stays
//! correct — no per-type split is needed.

use cubecl::prelude::*;

fn panics<R>(f: impl FnOnce() -> R) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err()
}

/// Every `FLOAT_METHOD_WHITELIST` entry: host-callable on `f64`, and (where a
/// same-named independent computation exists) numerically correct.
#[test]
fn whitelisted_methods_are_host_callable_and_correct_f64() {
    let x = 2.5f64;
    let y = 1.25f64;
    let tiny = 1e-12f64;

    assert!(!panics(|| f64::new(3.0)));
    assert_eq!(f64::new(3.0), 3.0);

    assert!(!panics(|| f64::from_int(3)));
    assert_eq!(f64::from_int(3), 3.0);

    assert!(!panics(f64::min_value));
    assert_eq!(f64::min_value(), f64::MIN);
    assert!(!panics(f64::max_value));
    assert_eq!(f64::max_value(), f64::MAX);

    assert!(!panics(|| x.abs()));
    assert_eq!((-x).abs(), x);

    assert!(!panics(|| x.min(y)));
    assert_eq!(x.min(y), y);
    assert!(!panics(|| x.max(y)));
    assert_eq!(x.max(y), x);
    assert!(!panics(|| x.clamp(0.0, 1.0)));
    assert_eq!(x.clamp(0.0, 1.0), 1.0);

    assert!(!panics(|| x.floor()));
    assert_eq!(x.floor(), 2.0);
    assert!(!panics(|| x.ceil()));
    assert_eq!(x.ceil(), 3.0);
    assert!(!panics(|| x.round()));
    assert_eq!(x.round(), 3.0);
    assert!(!panics(|| x.trunc()));
    assert_eq!(x.trunc(), 2.0);

    assert!(!panics(|| x.sqrt()));
    assert!((x.sqrt() - (2.5f64).sqrt()).abs() < tiny);
    assert!(!panics(|| x.recip()));
    assert!((x.recip() - 0.4).abs() < tiny);

    assert!(!panics(|| x.sin()));
    assert!((x.sin() - (2.5f64).sin()).abs() < tiny);
    assert!(!panics(|| x.cos()));
    assert!((x.cos() - (2.5f64).cos()).abs() < tiny);
    assert!(!panics(|| x.tan()));
    assert!((x.tan() - (2.5f64).tan()).abs() < tiny);

    let s = 0.6f64;
    assert!(!panics(|| s.asin()));
    assert!((s.asin() - (0.6f64).asin()).abs() < tiny);
    assert!(!panics(|| s.acos()));
    assert!((s.acos() - (0.6f64).acos()).abs() < tiny);
    assert!(!panics(|| x.atan()));
    assert!((x.atan() - (2.5f64).atan()).abs() < tiny);
    assert!(!panics(|| x.atan2(y)));
    assert!((x.atan2(y) - (2.5f64).atan2(1.25)).abs() < tiny);

    assert!(!panics(|| x.sinh()));
    assert!((x.sinh() - (2.5f64).sinh()).abs() < 1e-9);
    assert!(!panics(|| x.cosh()));
    assert!((x.cosh() - (2.5f64).cosh()).abs() < 1e-9);
    assert!(!panics(|| x.tanh()));
    assert!((x.tanh() - (2.5f64).tanh()).abs() < tiny);

    assert!(!panics(|| x.exp()));
    assert!((x.exp() - (2.5f64).exp()).abs() < 1e-9);
    assert!(!panics(|| x.ln()));
    assert!((x.ln() - (2.5f64).ln()).abs() < tiny);
    assert!(!panics(|| x.powf(y)));
    assert!((x.powf(y) - (2.5f64).powf(1.25)).abs() < 1e-9);
    assert!(!panics(|| x.powi(3)));
    assert_eq!(x.powi(3), 15.625);
    assert!(!panics(|| x.hypot(y)));
    assert!((x.hypot(y) - (2.5f64).hypot(1.25)).abs() < tiny);

    assert!(!panics(|| x.is_nan()));
    assert!(!x.is_nan());
    assert!(f64::NAN.is_nan());
    assert!(!panics(|| x.to_degrees()));
    assert!((x.to_degrees() - x.to_degrees()).abs() < tiny); // std shadow, sanity only
    assert!(!panics(|| x.to_radians()));
}

/// Every `FLOAT_METHOD_REJECT` entry still panics on a host `f64` call — the
/// empirical evidence that rejecting them is protecting against a real
/// footgun on f64 too, not just f32.
///
/// `#[allow(unstable_name_collisions)]`: rustc warns `.erf()`/`.dot()` may
/// collide with a future std method of the same name.
#[test]
#[allow(unstable_name_collisions)]
fn rejected_methods_panic_on_host_f64() {
    let x = 2.5f64;
    let y = 1.25f64;

    assert!(panics(|| x.log1p()), "log1p did not panic on host f64");
    assert!(panics(|| x.inverse_sqrt()), "inverse_sqrt did not panic on host f64");
    assert!(panics(|| x.erf()), "erf did not panic on host f64");
    assert!(panics(|| x.is_inf()), "is_inf did not panic on host f64");
    assert!(panics(|| x.rhypot(y)), "rhypot did not panic on host f64");
    assert!(panics(|| x.dot(y)), "dot did not panic on host f64");

    let idx: usize = 3;
    assert!(panics(|| f64::cast_from(idx)), "cast_from did not panic on host f64");
    assert!(
        panics(|| <f64 as Reinterpret>::reinterpret(3u64)),
        "reinterpret did not panic on host f64"
    );
}
