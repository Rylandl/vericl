//! Round-10 review, major 3 — the bare-`fma` shadow hole, as a **runnable**
//! regression.
//!
//! The review's probe put a user's own `#[cube] fn fma(a, b, c) { a*b + c +
//! 1000.0 }` beside a kernel that called `fma(...)` bare. On the `#[cube]` side
//! that item wins over `cubecl::prelude`'s glob-imported intrinsic (a glob
//! import is the weakest binding in Rust), but `ShimRewriteFold`'s shadowing
//! guard only knew about `uses(...)` names and `collect_locals`' `PatIdent`s —
//! item names in the enclosing scope are invisible to a proc macro. So the twin
//! silently took the GPU-verified host shim and computed **5.0** where the
//! device computed **1005.0**; only the differential lane caught it.
//!
//! Both halves of the fix are pinned here:
//!
//! 1. a bare `fma` is no longer rewritten at all — it reaches the ordinary
//!    undeclared-call classification, so the probe's kernel is now a COMPILE
//!    error naming both fixes (structural regression:
//!    `vericl-macros`' `bare_fma_is_rejected_with_both_fixes_named`); and
//! 2. the fix this file exercises — an explicit `uses(fma)` **wins over the
//!    shim**, so composing a helper that happens to be named `fma` is a legal
//!    program that computes the user's function on both sides. That case used to
//!    be rejected outright as "ambiguous".
//!
//! The intrinsic stays reachable in the very same item under its qualified
//! spelling, which is what makes the two meanings expressible side by side.

use cubecl::prelude::*;

/// A user function that happens to be named `fma`. The `+ 1000.0` is what makes
/// a substitution unmissable: source semantics for `(1, 2, 3)` is `1006.0`, the
/// intrinsic's is `5.0`.
///
/// Deliberately a chain of ADDITIONS rather than the review probe's `a*b + c +
/// 1000.0`: on Metal the shader compiler contracts an independent `a*b + c`
/// back into one fused instruction (the `vec_madd_bitexact` finding, unrelated
/// to this test), which costs 2 ulp against an unfused twin and would make this
/// file fail for a reason that has nothing to do with name resolution. Pure
/// additions have no contraction opportunity, so `max_ulp = 0` isolates the one
/// question here: WHICH `fma` did each side resolve?
#[vericl::helper]
#[cube]
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    a + b + c + 1000.0
}

/// `uses(fma)` composes the helper above; `cubecl::prelude::fma` in the same
/// body still reaches the GPU-verified host shim. Their sum is `1006 + 5`.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(max_ulp = 0),
    gen(x in -1.0..=1.0, y in 0.0..=0.0),
    uses(fma)
)]
#[cube(launch)]
pub fn fma_helper_and_intrinsic(x: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        let v = x[ABSOLUTE_POS];
        y[ABSOLUTE_POS] = fma(v, 2.0f32, 3.0f32) + cubecl::prelude::fma(v, 2.0f32, 3.0f32);
    }
}

#[test]
fn a_uses_declared_helper_named_fma_wins_over_the_shim() {
    let x = vec![1.0f32];
    let mut y = vec![0.0f32; 1];
    fma_helper_and_intrinsic_vericl::reference(&x, &mut y, 1);
    // helper: 1 + 2 + 3 + 1000 = 1006; intrinsic shim: 1*2 + 3 = 5.
    assert_eq!(
        y,
        vec![1011.0f32],
        "the twin must take the USER's `fma` for the bare call (1006) and the verified shim \
         for the qualified one (5) — a twin computing 10.0 has silently substituted the shim \
         for the helper, which is exactly the round-10 finding"
    );
}

/// The device is the arbiter: if the twin and the kernel disagreed about which
/// `fma` the bare call means, this differential case would fail — which is how
/// the original defect was found in the first place.
#[cfg(feature = "wgpu")]
#[test]
fn the_differential_lane_agrees_on_the_gpu() {
    use cubecl::Runtime;
    use cubecl::wgpu::WgpuRuntime;

    let client = WgpuRuntime::client(&Default::default());
    let outcome =
        fma_helper_and_intrinsic_vericl::conformance_case::<WgpuRuntime>(&client, 64, 0xC0FE, 256);
    assert!(
        outcome.pass(),
        "twin and kernel must resolve `fma` the same way: {}",
        vericl::describe_case_outcome(&outcome)
    );
}
