//! The `vericl::cube_struct!` milestone's **pre-registered residual**, tested
//! rather than asserted away (`docs/design-cubetype-args.md` §13, risk 2).
//!
//! Rust permits an inherent `impl` for a local type **anywhere in the crate**, so
//! a `#[cube] impl P { … }` written outside a `vericl::cube_struct! { … }`
//! invocation is invisible to both halves of what that macro provides: the
//! block's `STRUCT_HASH` (so an edit there does not move a kernel's recorded
//! identity) and gate CS4 (so the method is not rejected). There is no fix at
//! macro scope — a `#[proc_macro]` only ever sees the tokens it is handed, and
//! whole-crate name resolution is not available to it.
//!
//! This is `vericl::config!`'s risk 3 one parameter position over, and it is
//! **worse in consequence**, which is why it gets its own file rather than a
//! sentence. For a config, an out-of-block method that is not host-callable
//! fails as a loud twin panic. For a runtime struct, `#[cube]` emits *both* a
//! host body and an expanded device body, so the two can compute different
//! things and the failure mode is a **numeric divergence** — quieter, and caught
//! by a different backstop.
//!
//! The residual is accepted because the backstops are real, and this file pins
//! them instead of leaving them as prose:
//!
//! 1. `out_of_block_cube_impl_is_caught_by_the_differential_lane` — the
//!    divergence half. The out-of-block method's value reaches a compared
//!    output, so the differential harness sees any host/device disagreement.
//!    Here the two agree (`self.a * self.b` on both sides), and the case passes
//!    — which is the honest state of affairs: the backstop is the *lane*, not a
//!    claim that the hole cannot be entered.
//! 2. `out_of_block_impls_do_not_move_struct_hash` — the identity half, as an
//!    executable fact: two byte-identical `vericl::cube_struct!` blocks with
//!    *different* out-of-block impls hash identically. This is the residual, in
//!    one assertion, so it can never be quietly forgotten.
//! 3. `the_declared_struct_still_carries_a_real_hash` — the discrimination that keeps
//!    (2) from being an excuse: moving the same impl INSIDE the block is a
//!    compile error naming the measured `[3,6,9,12]` -> `[4,5,6,7]` divergence
//!    (pinned in `vericl-macros`' `impl_blocks_and_cube_attributes_are_rejected`;
//!    restated here as the documented boundary).
//!
//! 4. `forged_struct_identity_is_a_complete_bypass` — the round-11 addition, and
//!    a residual of a *different kind*: `StructIdentity` is a public, unsealed
//!    trait, so a hand-written impl for a type the macro never saw bypasses the
//!    mechanism entirely rather than partially. Written so that it stops
//!    compiling if the trait is ever sealed.
//!
//! The backstop for (2) beyond this file: a struct-derived value that reaches
//! the device is in the IR, so `Identity::ir_hash` moves even when `source_hash`
//! does not.

use vericl::StructIdentity;
use vericl_examples::{EvasivePair, cube_struct_out_of_block_evasion_vericl};

/// (1) The divergence half, through the *real* harness path.
#[cfg(feature = "wgpu")]
#[test]
fn out_of_block_cube_impl_is_caught_by_the_differential_lane() {
    use cubecl::Runtime;
    use cubecl::wgpu::WgpuRuntime;

    let client = WgpuRuntime::client(&Default::default());
    let outcome = cube_struct_out_of_block_evasion_vericl::conformance_case::<WgpuRuntime>(
        &client, 256, 0xC0FE, 256,
    );
    assert!(
        outcome.reference_panic.is_none(),
        "this probe's out-of-block method IS host-callable — a panic here would mean the probe \
         drifted into the config-style residual instead: {outcome:?}"
    );
    assert!(
        outcome.pass(),
        "the host and device bodies of the out-of-block `#[cube] impl` agree here, so the case \
         must pass — the point is that the differential lane is what would catch them disagreeing, \
         not that the hole is closed: {outcome:?}"
    );
}

/// (1b) The same claim without a GPU: the twin really does call the
/// out-of-block host method, so a divergence introduced there is a divergence
/// the lane compares. If the twin did not reach it, (1) would be vacuous.
#[test]
fn the_twin_reaches_the_out_of_block_method() {
    let x: Vec<u32> = (0..8u32).collect();
    let mut y = vec![0u32; x.len()];
    cube_struct_out_of_block_evasion_vericl::reference(&x, &mut y, x.len());
    let expected: Vec<u32> = x.iter().map(|v| v * 3).collect();
    assert_eq!(
        y, expected,
        "the twin must execute the out-of-block `fold()` — that it does is exactly why an edit \
         there is a numeric divergence rather than a compile error"
    );
}

// (2) The identity half. Two `vericl::cube_struct!` blocks with byte-identical
// tokens, differing only in an impl written OUTSIDE each block.

#[allow(dead_code)]
mod residual_a {
    // `vericl::cube_struct!` emits CubeCL's `CubeType`/`CubeLaunch` derives,
    // which expand to unqualified `CubeType`/`KernelBuilder`/… paths — so the
    // block needs `cubecl::prelude` in scope, exactly as a `#[cube]` item does.
    use cubecl::prelude::*;

    vericl::cube_struct! {
        pub struct C {
            pub k: u32,
        }
    }

    /// Outside the block: neither hashed nor gated.
    impl C {
        pub fn extra(&self) -> u32 {
            self.k * 2
        }
    }
}

#[allow(dead_code)]
mod residual_b {
    use cubecl::prelude::*;

    vericl::cube_struct! {
        pub struct C {
            pub k: u32,
        }
    }

    /// Outside the block, and DIFFERENT from `residual_a`'s.
    impl C {
        pub fn extra(&self) -> u32 {
            self.k * 3
        }
    }
}

#[allow(dead_code)]
mod residual_c {
    use cubecl::prelude::*;

    vericl::cube_struct! {
        /// One field RENAMED — an edit INSIDE the block.
        pub struct C {
            pub kk: u32,
        }
    }
}

/// (2) The residual, as an assertion: an out-of-block impl changes what a kernel
/// using the struct would compute, and does not move `STRUCT_HASH`.
///
/// If this test ever starts FAILING because the hashes differ, the residual has
/// been closed and `docs/design-cubetype-args.md` §13 risk 2, this file's module
/// doc, and `vericl-macros`' `cube_struct` module doc must be updated to say so.
/// It is written to be sensitive in that direction on purpose.
#[test]
fn out_of_block_impls_do_not_move_struct_hash() {
    // The two out-of-block impls genuinely differ…
    assert_ne!(
        residual_a::C { k: 5 }.extra(),
        residual_b::C { k: 5 }.extra(),
        "the two out-of-block impls must compute different things for this test to mean anything"
    );
    // …and the in-block halves are byte-identical, so the hashes match: the
    // out-of-block impl contributes nothing to identity. THIS IS THE RESIDUAL.
    assert_eq!(
        <residual_a::C as StructIdentity>::STRUCT_HASH,
        <residual_b::C as StructIdentity>::STRUCT_HASH,
        "documented residual (design §13 risk 2): an impl outside the vericl::cube_struct! block \
         is invisible to STRUCT_HASH"
    );

    // Discrimination — the gate is not vacuous: an edit INSIDE the block does
    // move the hash.
    assert_ne!(
        <residual_a::C as StructIdentity>::STRUCT_HASH,
        <residual_c::C as StructIdentity>::STRUCT_HASH,
        "an in-block declaration edit MUST move STRUCT_HASH"
    );

    // And the runtime-struct trait is not silently the config one: `C`'s single
    // field is a `u32`, so it is comptime-usable and carries BOTH traits with
    // the same hash — one type, both positions (design §6, as **corrected** in
    // round 11: the second impl is emitted only for a declared type whose whole
    // field shape is integer/bool/char/unit-enum, because CubeCL `Debug`-formats
    // a comptime parameter and derives `Hash`/`Eq` over it, and `f32` is none of
    // those).
    assert_eq!(
        <residual_a::C as StructIdentity>::STRUCT_HASH,
        <residual_a::C as vericl::ConfigIdentity>::CONFIG_HASH,
        "a comptime-usable cube_struct! type's STRUCT_HASH and CONFIG_HASH are one hash of one block"
    );
}

// (4) The FORGED-IDENTITY bypass (round-11 review, LOW 6).
//
// `StructIdentity` is a public, unsealed trait. Nothing stops an author writing
// the impl by hand for a type `vericl::cube_struct!` never saw — and that is not
// a narrow gap in one gate, it is a complete bypass of the mechanism: no gate
// runs on the type, its `#[cube] impl` methods are unrestricted, and its
// recorded identity is a constant that by construction never goes stale.
//
// It is recorded rather than closed. A `#[proc_macro]` cannot seal a trait, and
// VeriCL's guarantee has never been "an author cannot lie to their own evidence
// file" — it is "an author who does not lie gets an identity that moves when the
// meaning does". Every gate is aimed at accidental drift.

/// A type `vericl::cube_struct!` never saw, wearing the trait anyway.
#[derive(Clone, Copy)]
struct Forged {
    #[allow(dead_code)]
    k: u32,
}

impl StructIdentity for Forged {
    const STRUCT_HASH: &'static str = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
}

/// (4) The acknowledgment test, written to FAIL IF THE HOLE IS CLOSED.
///
/// If `StructIdentity` is ever sealed, this file stops compiling — at which
/// point `vericl-macros`' `cube_struct` residual section and this comment are
/// wrong and must be rewritten. That is the point of writing it down as code.
#[test]
fn forged_struct_identity_is_a_complete_bypass() {
    // The forged hash is whatever the author typed, and it covers nothing: the
    // type has no declaration block, no gates, and no emitted constructor.
    assert_eq!(<Forged as StructIdentity>::STRUCT_HASH, "sha256:0000000000000000000000000000000000000000000000000000000000000000");
    // Discrimination against the honest path: a real declared struct's hash is
    // derived from tokens, so two different declarations differ — the forged one
    // cannot differ from itself no matter what is edited.
    assert_ne!(
        <Forged as StructIdentity>::STRUCT_HASH,
        <residual_a::C as StructIdentity>::STRUCT_HASH,
        "a declared struct's hash must not collide with a hand-written constant"
    );
    assert_ne!(
        <residual_a::C as StructIdentity>::STRUCT_HASH,
        <residual_c::C as StructIdentity>::STRUCT_HASH,
        "…and the honest path is the one where an edit moves the hash"
    );
}

/// (3) The documented boundary the residual sits against: the same `#[cube]
/// impl` written INSIDE the block is rejected, so the residual is a
/// crate-scoping limitation and not a decision to allow struct methods.
///
/// The evasion probe's type is declared with `vericl::cube_struct!` and carries
/// a real `STRUCT_HASH`, so the kernel's identity does cover the struct's
/// *fields*; what it cannot cover is the impl. Both facts in one place.
#[test]
fn the_declared_struct_still_carries_a_real_hash() {
    let h = <EvasivePair as StructIdentity>::STRUCT_HASH;
    assert!(h.starts_with("sha256:"), "a declared struct must carry a real block hash: {h}");
    assert_ne!(
        h,
        <residual_a::C as StructIdentity>::STRUCT_HASH,
        "different blocks must hash differently"
    );
    assert_ne!(
        cube_struct_out_of_block_evasion_vericl::identity().source_hash,
        cube_struct_out_of_block_evasion_vericl::SOURCE_HASH,
        "the evasion kernel still folds its struct's STRUCT_HASH — the residual is the IMPL, not \
         the declaration"
    );
}
