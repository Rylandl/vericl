//! Round-10 review, moderate 6 — a **scalar type alias** in `#[comptime]`
//! position.
//!
//! `#[vericl::kernel]` classifies a comptime parameter as a *config* by syntax:
//! anything that is not a written-out scalar primitive. A `#[proc_macro_attribute]`
//! has no name resolution, so `type Taps = u32;` looked like a struct-typed
//! config and produced a `ConfigIdentity`-not-implemented error telling the
//! author to wrap `Taps` in `vericl::config!` — advice that is impossible to
//! follow for an alias and wrong about what the type is.
//!
//! rustc *can* see through the alias, and that is where the resolution now
//! happens: `vericl` implements `ConfigIdentity` for each scalar primitive with
//! a constant naming the type. This file pins both halves — the kernel
//! compiles and computes, and the alias is not identity-invisible.

use cubecl::prelude::*;
use vericl::ConfigIdentity;

pub type Taps = u32;

#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(max_ulp = 0),
    gen(x in -1.0..=1.0, y in 0.0..=0.0),
    instantiate(taps = 3)
)]
#[cube(launch)]
pub fn alias_scaled(x: &Array<f32>, y: &mut Array<f32>, #[comptime] taps: Taps) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = x[ABSOLUTE_POS] * f32::cast_from(taps);
    }
}

#[test]
fn a_scalar_alias_comptime_parameter_compiles_and_computes() {
    let x = vec![2.0f32, -1.0];
    let mut y = vec![0.0f32; x.len()];
    alias_scaled_vericl::reference(&x, &mut y, x.len());
    assert_eq!(y, vec![6.0f32, -3.0], "the pinned alias value must reach the twin");
}

/// The alias is not identity-invisible: the scalar identity names the concrete
/// type, so retargeting `type Taps = u32;` at another primitive moves the
/// folded hash and re-stales stored evidence.
#[test]
fn scalar_identities_are_distinct_per_type() {
    assert_eq!(<u32 as ConfigIdentity>::CONFIG_HASH, "vericl-scalar:u32");
    assert_ne!(
        <u32 as ConfigIdentity>::CONFIG_HASH,
        <u64 as ConfigIdentity>::CONFIG_HASH,
        "retargeting a scalar alias must move the kernel's recorded identity"
    );
    // And the kernel really folds it (not merely `SOURCE_HASH`).
    assert_eq!(
        alias_scaled_vericl::identity().source_hash,
        vericl::combine_source_hash(
            alias_scaled_vericl::SOURCE_HASH,
            &[<Taps as ConfigIdentity>::CONFIG_HASH.to_string()],
        ),
        "the recorded identity must be SOURCE_HASH folded with the alias's scalar identity"
    );
}
