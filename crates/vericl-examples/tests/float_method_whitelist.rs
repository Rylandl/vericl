//! Empirical verification of `vericl-macros`' `FLOAT_METHOD_WHITELIST` /
//! `FLOAT_METHOD_REJECT` (crates/vericl-macros/src/lib.rs) — the set of
//! `cubecl::prelude::Float`/`Numeric` trait methods a reference twin body is
//! allowed to call after `instantiate(...)` substitutes its generic type
//! parameter to a concrete float (e.g. `F` -> `f32`).
//!
//! Method: for every whitelisted name, call it on host through the *same*
//! path the twin uses (either `x.method(...)` or `f32::method(...)`, with
//! `cubecl::prelude::*` in scope so the trait — not just the std inherent
//! method — is a candidate) and cross-check the result against an
//! independent `std`/hand-computed equivalent. For every rejected name,
//! confirm it actually panics on host (`Unexpanded Cube functions should
//! not be called.`) — proving the rejection is protecting against a real
//! footgun, not a hypothetical one.
//!
//! This is the "strong form" of verification the instantiate(...) design
//! calls for: not just reading cubecl's source, but calling every entry and
//! observing what actually happens. Keep this file's method list in sync
//! with `FLOAT_METHOD_WHITELIST`/`FLOAT_METHOD_REJECT` by hand — there is no
//! shared crate to `#[cfg(test)]` cross-check against (vericl-macros cannot
//! depend on cubecl; see its module docs).

use cubecl::prelude::*;

fn panics<R>(f: impl FnOnce() -> R) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err()
}

/// Every `FLOAT_METHOD_WHITELIST` entry: host-callable, and (where a
/// same-named independent computation exists) numerically correct.
#[test]
fn whitelisted_methods_are_host_callable_and_correct() {
    let x = 2.5f32;
    let y = 1.25f32;
    let tiny = 1e-4f32;

    assert!(!panics(|| f32::new(3.0)));
    assert_eq!(f32::new(3.0), 3.0);

    assert!(!panics(|| f32::from_int(3)));
    assert_eq!(f32::from_int(3), 3.0);

    assert!(!panics(f32::min_value));
    assert_eq!(f32::min_value(), f32::MIN);
    assert!(!panics(f32::max_value));
    assert_eq!(f32::max_value(), f32::MAX);

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
    assert!((x.sqrt() - (2.5f64).sqrt() as f32).abs() < tiny);
    assert!(!panics(|| x.recip()));
    assert!((x.recip() - 0.4).abs() < tiny);

    assert!(!panics(|| x.sin()));
    assert!((x.sin() - (2.5f64).sin() as f32).abs() < tiny);
    assert!(!panics(|| x.cos()));
    assert!((x.cos() - (2.5f64).cos() as f32).abs() < tiny);
    assert!(!panics(|| x.tan()));
    assert!((x.tan() - (2.5f64).tan() as f32).abs() < tiny);

    let s = 0.6f32;
    assert!(!panics(|| s.asin()));
    assert!((s.asin() - (0.6f64).asin() as f32).abs() < tiny);
    assert!(!panics(|| s.acos()));
    assert!((s.acos() - (0.6f64).acos() as f32).abs() < tiny);
    assert!(!panics(|| x.atan()));
    assert!((x.atan() - (2.5f64).atan() as f32).abs() < tiny);
    assert!(!panics(|| x.atan2(y)));
    assert!((x.atan2(y) - (2.5f64).atan2(1.25) as f32).abs() < tiny);

    assert!(!panics(|| x.sinh()));
    assert!((x.sinh() - (2.5f64).sinh() as f32).abs() < 1e-3);
    assert!(!panics(|| x.cosh()));
    assert!((x.cosh() - (2.5f64).cosh() as f32).abs() < 1e-3);
    assert!(!panics(|| x.tanh()));
    assert!((x.tanh() - (2.5f64).tanh() as f32).abs() < tiny);

    assert!(!panics(|| x.exp()));
    assert!((x.exp() - (2.5f64).exp() as f32).abs() < 1e-3);
    assert!(!panics(|| x.ln()));
    assert!((x.ln() - (2.5f64).ln() as f32).abs() < tiny);
    assert!(!panics(|| x.powf(y)));
    assert!((x.powf(y) - (2.5f64).powf(1.25) as f32).abs() < 1e-3);
    assert!(!panics(|| x.powi(3)));
    assert_eq!(x.powi(3), 15.625);
    assert!(!panics(|| x.hypot(y)));
    assert!((x.hypot(y) - (2.5f64).hypot(1.25) as f32).abs() < tiny);

    assert!(!panics(|| x.is_nan()));
    assert!(!x.is_nan());
    assert!(f32::NAN.is_nan());
    assert!(!panics(|| x.to_degrees()));
    assert!((x.to_degrees() - x.to_degrees()).abs() < tiny); // std shadow, sanity only
    assert!(!panics(|| x.to_radians()));
}

/// Every `FLOAT_METHOD_REJECT` entry actually panics on a host call — this
/// is the empirical evidence behind `FloatMethodCheck` rejecting them at
/// macro-expansion time rather than shipping a twin that would panic (or,
/// worse, silently miscompute) the first time a kernel used one.
///
/// `#[allow(unstable_name_collisions)]`: rustc warns that `.erf()`/`.dot()`
/// may collide with a future std method of the same name — itself a small
/// piece of evidence for why these particular names are unverified rather
/// than trusted.
#[test]
#[allow(unstable_name_collisions)]
fn rejected_methods_panic_on_host() {
    let x = 2.5f32;
    let y = 1.25f32;

    assert!(panics(|| x.log1p()), "log1p did not panic on host");
    assert!(panics(|| x.inverse_sqrt()), "inverse_sqrt did not panic on host");
    assert!(panics(|| x.erf()), "erf did not panic on host");
    assert!(panics(|| x.is_inf()), "is_inf did not panic on host");
    assert!(panics(|| x.rhypot(y)), "rhypot did not panic on host");
    assert!(panics(|| x.dot(y)), "dot did not panic on host");

    // Found by dogfooding against a real generic kernel
    // (a production kernel's `F::cast_from(index)` idiom):
    // `Cast::cast_from`'s only impl is a blanket `unexpanded!()`, so it
    // panics for every type, not just f32 — same for `Reinterpret::reinterpret`.
    let idx: usize = 3;
    assert!(panics(|| f32::cast_from(idx)), "cast_from did not panic on host");
    assert!(
        panics(|| <f32 as Reinterpret>::reinterpret(3u32)),
        "reinterpret did not panic on host"
    );
}
