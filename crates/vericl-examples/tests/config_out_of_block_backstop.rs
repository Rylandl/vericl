//! The `vericl::config!` milestone's **pre-registered residual**, tested rather
//! than asserted away (`docs/design-struct-comptime.md` §13, risk 3).
//!
//! Rust permits an inherent `impl` for a local type **anywhere in the crate**,
//! so a second `impl` block written outside a `vericl::config! { … }`
//! invocation is invisible to both halves of what that macro provides: the
//! block's `CONFIG_HASH` (so an edit there does not move a kernel's recorded
//! identity) and the method-body gates (so a call to a non-host-callable
//! intrinsic there is not a compile error). There is no fix at macro scope — a
//! `#[proc_macro]` only ever sees the tokens it is handed, and whole-crate name
//! resolution is not available to it.
//!
//! The residual is accepted because both halves fail LOUDLY, and this file pins
//! that claim instead of leaving it as prose:
//!
//! 1. `out_of_block_cube_impl_panics_loudly_in_the_twin` — the host-callability
//!    half. The reference twin calls the out-of-block host method, CubeCL's
//!    `fma` is `unexpanded!()` on host, and the twin panics. Moving the same
//!    method INTO the `vericl::config!` block turns this into a compile-time
//!    rejection at `fma`'s own span (gate G3; pinned by
//!    `vericl-macros`' `reject_listed_call_in_a_config_method_is_rejected`).
//! 2. `differential_lane_reports_the_out_of_block_panic` — the same failure
//!    through the *real* harness path: `conformance_case` catches it, records
//!    it as `reference_panic`, and the case does not pass. Nothing is swallowed.
//! 3. `out_of_block_impls_do_not_move_config_hash` — the identity half, stated
//!    as an executable fact: two byte-identical `vericl::config!` blocks with
//!    *different* out-of-block impls hash identically. This is the residual, in
//!    one assertion, so it can never be quietly forgotten.
//!
//! The backstop for (3) beyond this file: a config-derived value that reaches
//! the device is a constant in the IR, so `Identity::ir_hash` moves even when
//! `source_hash` does not (design §3, §5.1).

use vericl_examples::config_out_of_block_evasion_vericl;

/// (1) The host-callability half of the residual: loud, at the twin.
#[test]
#[should_panic(expected = "Unexpanded Cube functions")]
fn out_of_block_cube_impl_panics_loudly_in_the_twin() {
    let x = vec![1.0f32, 2.0, 3.0];
    let mut y = vec![0.0f32; x.len()];
    config_out_of_block_evasion_vericl::reference(&x, &mut y, x.len());
}

/// (2) The same failure through the differential harness: recorded as a
/// reference panic and a failing case, never swallowed.
#[cfg(feature = "wgpu")]
#[test]
fn differential_lane_reports_the_out_of_block_panic() {
    use cubecl::Runtime;
    use cubecl::wgpu::WgpuRuntime;

    let client = WgpuRuntime::client(&Default::default());
    let outcome =
        config_out_of_block_evasion_vericl::conformance_case::<WgpuRuntime>(&client, 64, 0xC0FE, 256);
    let panic_msg = outcome
        .reference_panic
        .as_ref()
        .expect("the twin must panic — that is the backstop this test exists to pin");
    assert!(
        panic_msg.contains("Unexpanded Cube functions"),
        "the recorded panic must name the actual cause: {panic_msg}"
    );
    assert!(!outcome.pass(), "a panicking twin must never produce a passing case");
    let rendered = vericl::describe_case_outcome(&outcome);
    assert!(
        rendered.contains("reference execution panicked"),
        "the harness must report it in the human-readable outcome: {rendered}"
    );
}

// (3) The identity half. Two `vericl::config!` blocks with byte-identical
// tokens, differing only in an impl written OUTSIDE each block.

#[allow(dead_code)]
mod residual_a {
    vericl::config! {
        #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
        pub struct C {
            pub k: u32,
        }

        impl C {
            pub fn k(&self) -> u32 {
                self.k
            }
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
    vericl::config! {
        #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
        pub struct C {
            pub k: u32,
        }

        impl C {
            pub fn k(&self) -> u32 {
                self.k
            }
        }
    }

    /// Outside the block, and DIFFERENT from `residual_a`'s.
    impl C {
        pub fn extra(&self) -> u32 {
            self.k * 3
        }
    }
}

/// (3) The residual, as an assertion: an out-of-block impl changes what a
/// kernel using the config would compute, and does not move `CONFIG_HASH`.
///
/// If this test ever starts FAILING because the hashes differ, the residual has
/// been closed and `docs/design-struct-comptime.md` §13 risk 3, this file's
/// module doc, and `vericl-macros`' `config` module doc must be updated to say
/// so. It is written to be sensitive in that direction on purpose.
#[test]
fn out_of_block_impls_do_not_move_config_hash() {
    use vericl::ConfigIdentity;

    // The two out-of-block impls genuinely differ…
    assert_ne!(
        residual_a::C { k: 5 }.extra(),
        residual_b::C { k: 5 }.extra(),
        "the two out-of-block impls must compute different things for this test to mean anything"
    );
    // …and the in-block halves are byte-identical, so the hashes match: the
    // out-of-block impl contributes nothing to identity. THIS IS THE RESIDUAL.
    assert_eq!(
        <residual_a::C as ConfigIdentity>::CONFIG_HASH,
        <residual_b::C as ConfigIdentity>::CONFIG_HASH,
        "documented residual (design §13 risk 3): an impl outside the vericl::config! block is \
         invisible to CONFIG_HASH"
    );

    // Discrimination — the gate is not vacuous: an edit INSIDE the block does
    // move the hash. (`residual_c` differs from `residual_a` only in `k()`'s
    // body, which is inside the block.)
    assert_ne!(
        <residual_a::C as ConfigIdentity>::CONFIG_HASH,
        <residual_c::C as ConfigIdentity>::CONFIG_HASH,
        "an in-block method body edit MUST move CONFIG_HASH"
    );
}

#[allow(dead_code)]
mod residual_c {
    vericl::config! {
        #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
        pub struct C {
            pub k: u32,
        }

        impl C {
            pub fn k(&self) -> u32 {
                self.k + 1
            }
        }
    }
}
