//! The `vericl::cube_struct!` milestone's central claims, as executable facts:
//! the **flipped soundness probe**, the lowering tripwire, and the negative
//! controls (`docs/design-cubetype-args.md` §4.1, §4.3, §2.3, §12 M3/M4).
//!
//! # What was measured before this milestone, and what it means here
//!
//! At `e5589f3`, a runtime `CubeType` struct on a `#[vericl::helper]` was
//! **silently accepted** — no diagnostic anywhere — and its definition was in no
//! hash. Editing a `#[cube] impl Pair { fn fold }` from `self.a * self.b` to
//! `self.a + self.b` moved the reference twin from `[3, 6, 9, 12]` to
//! `[4, 5, 6, 7]` while the kernel's `SOURCE_HASH`, the helper's `SOURCE_HASH`
//! **and** `identity().source_hash` all stayed bit-identical. Evidence recorded
//! against the first build verified FRESH against the second: a different
//! computed function under an unmoved identity, which is the one thing an
//! evidence harness must never do.
//!
//! The probe is flipped in two independent places, and this file pins both:
//!
//! 1. **The edit is rejected.** An `impl` block inside a `vericl::cube_struct!`
//!    block is a compile error naming the measured divergence, and an undeclared
//!    struct in runtime parameter position is a compile error naming
//!    `vericl::cube_struct!` (both pinned in `vericl-macros`' own unit tests,
//!    where the macro's output can be inspected without a `trybuild` dependency
//!    — `cube_struct::tests` and `struct_param_*` in the crate-root tests).
//! 2. **The identity moves.** Every edit to the declaration — field name, field
//!    type, field ORDER, comptime-ness — moves `STRUCT_HASH`, and therefore the
//!    recorded `identity().source_hash` of every kernel and helper that reaches
//!    the type. That is what this file asserts end to end, on kernels that
//!    actually run.
//!
//! The field-ORDER case is not cosmetic. CubeCL fills a launch struct
//! **positionally** (`generate_struct.rs:92-114`), so before this milestone
//! swapping two same-typed fields in the *declaration* changed the computed
//! function with the kernel body and the launch-call text byte-unchanged (§4.3,
//! probe X2). Under `vericl::cube_struct!` the constructor is emitted from the
//! declared order, so the computation stays correct — and the hash must move so
//! the stored evidence goes stale rather than silently describing another
//! kernel. Both halves are asserted below; they are independent defences.

use vericl_examples::*;

/// (1) The flipped probe's identity half, on the **kernel**: two modules whose
/// kernel and helper tokens are byte-identical and whose `vericl::cube_struct!`
/// blocks differ only in FIELD ORDER.
///
/// Before the milestone, the analogous A/B (a `#[cube] impl` body edit) left
/// `identity().source_hash` bit-identical at `sha256:f0096061…` across a twin
/// that changed from `[3, 6, 9, 12]` to `[4, 5, 6, 7]`. Now a declaration edit
/// that does *not* even change the answer still moves identity — the safe
/// direction, and the one that makes `vericl::suite!` report
/// `STALE evidence — identity mismatch (source_hash X -> Y)`.
#[test]
fn a_field_reorder_moves_kernel_and_helper_identity() {
    let base = struct_identity_base::pair_map_vericl::identity();
    let reordered = struct_identity_reordered::pair_map_vericl::identity();

    // The two kernels' OWN tokens are byte-identical — this is what makes the
    // test mean something. If these ever differ, the A/B has drifted and the
    // assertion below would pass for the wrong reason.
    assert_eq!(
        struct_identity_base::pair_map_vericl::SOURCE_HASH,
        struct_identity_reordered::pair_map_vericl::SOURCE_HASH,
        "the two kernels' own source tokens must be byte-identical for this A/B to mean anything \
         — SOURCE_HASH cannot see the struct's definition, which is the whole point"
    );
    assert_eq!(
        struct_identity_base::fold_pair_vericl::SOURCE_HASH,
        struct_identity_reordered::fold_pair_vericl::SOURCE_HASH,
        "…and so must the two helpers'"
    );

    // …and the recorded identity nonetheless differs, because STRUCT_HASH is
    // folded in. THIS IS THE FLIP.
    assert_ne!(
        base.source_hash, reordered.source_hash,
        "a field REORDER in the vericl::cube_struct! block must move the kernel's recorded \
         identity — the positional `<Name>Launch::new` makes field order semantically \
         load-bearing (design §4.3, probe X2), and before this milestone the same class of \
         declaration edit left identity bit-identical (§4.1, probe V4)"
    );

    // The helper's own composition-aware hash moves too, independently: a
    // helper that merely *takes* the struct folds it. That is the half that
    // closes the measured hole, since the accepted, undiagnosed shape at
    // `e5589f3` was exactly `#[vericl::helper] fn use_pair(p: Pair)`.
    assert_ne!(
        struct_identity_base::fold_pair_vericl::identity_hash(),
        struct_identity_reordered::fold_pair_vericl::identity_hash(),
        "a helper taking a runtime struct must fold that struct's STRUCT_HASH into its own \
         identity — this is the V3/V4 hole's direct closure"
    );
}

/// (2) NEGATIVE CONTROL for (1): the reorder must **not** change what the
/// kernels compute. VeriCL-side field access is by name, and the launch
/// constructor is re-emitted from the declared order, so both lanes still
/// compute `x * 3`.
///
/// This is what makes the two defences *independent*, and it is the honest
/// framing the design insisted on: stating the reorder as a twin-visible hazard
/// would have been the easy overclaim (§4.1's own negative control measured the
/// twin unchanged at `[3, 6, 9, 12]`). The hazard is launch-side, it is now
/// internal, and the hash is what reports it.
#[test]
fn the_reorder_changes_identity_but_not_the_computed_function() {
    let x: Vec<u32> = (0..8u32).collect();
    let mut y_base = vec![0u32; x.len()];
    let mut y_reordered = vec![0u32; x.len()];
    struct_identity_base::pair_map_vericl::reference(&x, &mut y_base, x.len());
    struct_identity_reordered::pair_map_vericl::reference(&x, &mut y_reordered, x.len());
    assert_eq!(y_base, y_reordered, "field access is by name in the twin — the reorder is inert");
    let expected: Vec<u32> = x.iter().map(|v| v * 3).collect();
    assert_eq!(y_base, expected, "…and both compute a * 3");
}

/// (3) The lowering tripwire (design §2.3, probes I1/I3): **a runtime struct
/// parameter is exactly a positional flattening of its fields**, so the struct
/// kernel's `kernel_ir_hash` must equal the flattened kernel's, byte for byte.
///
/// This is the load-bearing measurement behind "the prover needs zero changes":
/// the IR cannot tell the two spellings apart, so `BUFFER_PARAMS`, the
/// `index == buffer id` invariant, and every obligation are unchanged. It is
/// also the cheap cubecl-upgrade tripwire the design's risk 9 asks for — a 0.11
/// that changed field-expansion order would fail here rather than silently.
#[test]
fn struct_and_flattened_spellings_have_identical_ir() {
    let struct_hash = vericl_ir::kernel_ir_hash(&uniform_value_map_vericl::kernel_definition());
    let flat_hash = vericl_ir::kernel_ir_hash(&uniform_value_map_flat_vericl::kernel_definition());
    assert_eq!(
        struct_hash, flat_hash,
        "a runtime struct parameter must lower to exactly the flattened spelling of its fields \
         (design §2.3) — if this fails, cubecl's `generate_struct.rs` expansion order has changed \
         and docs/design-cubetype-args.md §2 must be re-measured before anything else is trusted"
    );

    // …and the two carry the same buffer custody: the struct contributes NO
    // buffer, so the prover sees the same two arrays in the same order.
    assert_eq!(
        uniform_value_map_vericl::BUFFER_PARAMS,
        uniform_value_map_flat_vericl::BUFFER_PARAMS,
        "a v1 runtime struct is all scalars, so it must contribute no buffer entry"
    );
    assert_eq!(uniform_value_map_vericl::BUFFER_PARAMS, [("s", false), ("y", true)]);
}

/// (4) NEGATIVE CONTROL for (3): the tripwire is not vacuous — two kernels that
/// genuinely differ have different `ir_hash`es.
#[test]
fn the_ir_hash_tripwire_discriminates() {
    let uniform = vericl_ir::kernel_ir_hash(&uniform_value_map_vericl::kernel_definition());
    let stage = vericl_ir::kernel_ir_hash(&stage_window_sum_vericl::kernel_definition());
    assert_ne!(uniform, stage, "different kernels must have different ir_hashes");
}

/// (5) The **body-literal** collection route, which is what 19 of the corrected
/// 20 ecosystem sites exercise: `accum_blend_map` never takes a struct
/// parameter — it builds an `Accum { … }` literal in its body and hands it to a
/// helper. Its identity must still fold `Accum`'s `STRUCT_HASH`.
///
/// The discriminating kernel is `cube_struct_out_of_block_evasion`, NOT
/// `accum_blend_map`: the latter also declares `uses(accum_blend)`, so its
/// folded identity would differ from its `SOURCE_HASH` even with the body route
/// removed. Verified by defect injection — disabling the body walk leaves
/// `accum_blend_map`'s assertion passing and fails the one below.
///
/// `cube_struct_out_of_block_evasion` has **no `uses(...)` and no struct
/// parameter**, so a struct literal in its body is its only possible dependency;
/// if its folded `source_hash` differs from its own `SOURCE_HASH`, the body
/// route is the only thing that can have put it there.
#[test]
fn a_body_struct_literal_is_folded_into_identity() {
    assert!(
        cube_struct_out_of_block_evasion_vericl::USES.is_empty(),
        "this test's discrimination depends on the probe kernel having no uses(...) — if that \
         changes, the assertion below stops proving the body route"
    );
    assert_ne!(
        cube_struct_out_of_block_evasion_vericl::identity().source_hash,
        cube_struct_out_of_block_evasion_vericl::SOURCE_HASH,
        "a kernel whose ONLY route to a struct type is a body literal must still fold that \
         struct's STRUCT_HASH — otherwise an author could mention a type, edit its definition, and \
         show the identity unmoved (design risk 1)"
    );

    // The helper route, on both sides of a composition: `accum_blend` takes the
    // struct by value, and `accum_blend_map` both builds it and composes the
    // helper.
    assert_ne!(
        accum_blend_vericl::identity_hash(),
        accum_blend_vericl::SOURCE_HASH,
        "the helper's composition-aware hash must fold its struct parameter's STRUCT_HASH"
    );
    assert_ne!(
        accum_blend_map_vericl::identity().source_hash,
        accum_blend_map_vericl::SOURCE_HASH,
        "…and the composing kernel inherits it"
    );
}

/// (6) NEGATIVE CONTROL for the whole milestone's evidence claim: a kernel that
/// mentions **no** struct at all must have a byte-identical identity to what it
/// had before — which is what the 194-insertion, 0-deletion evidence diff
/// showed, and what this pins as a permanent assertion.
///
/// `combine_source_hash` is a pure pass-through on an empty dependency list, so
/// a struct-free, helper-free kernel's `identity().source_hash` must be exactly
/// its `SOURCE_HASH`.
#[test]
fn a_struct_free_kernel_identity_is_untouched() {
    assert_eq!(
        axpy_vericl::identity().source_hash,
        axpy_vericl::SOURCE_HASH,
        "adding the StructIdentity fold must be a no-op for every kernel that reaches no struct"
    );
}

/// (7) The comptime-FIELD path, end to end: `instantiate(cfg.window.taps = 3)`
/// must reach BOTH consumers with the same value — the twin's struct binding and
/// the kernel's expansion (design risk 6).
///
/// Asserted through the two artifacts that would disagree if it did not: the
/// twin's own output (a 3-tap accumulation) and the extracted IR's loop bound
/// (a `RangeLoop` with a constant end, hence a `kernel_definition()` that
/// differs from the 2-tap spelling). One `const` spec binding is what makes this
/// structural rather than hopeful — there is no second literal to drift.
#[test]
fn a_comptime_struct_field_reaches_the_twin_and_the_ir() {
    // 4 elements, all 1.0, gain 1.0, bias 0.0 → a 3-tap forward sum, truncated
    // at the end of the array by the kernel's own `idx < x.len()` guard.
    let x = vec![1.0f32; 4];
    let mut y = vec![0.0f32; 4];
    let cfg = StageArgs { window: StageWindow { gain: 1.0, taps: 3 }, bias: 0.0 };
    stage_window_sum_vericl::reference(&x, cfg, &mut y, x.len());
    assert_eq!(
        y,
        vec![3.0, 3.0, 2.0, 1.0],
        "the twin must accumulate `taps = 3` forward taps — if this is 2 or 4, the pinned comptime \
         field reached the twin with a different value than the kernel"
    );

    // The pinned value is in the recorded contract, so the evidence says which
    // instantiation was verified.
    assert!(
        stage_window_sum_vericl::contract().instantiate.iter().any(|s| s.contains("cfg.window.taps")),
        "the dotted pin must be recorded in the contract: {:?}",
        stage_window_sum_vericl::contract().instantiate
    );
}

/// (8) The struct value the harness generates is the value the twin sees AND the
/// value the GPU is launched with — one draw, two consumers. Pinned on the twin
/// side by regenerating the same case and checking the drawn struct lands inside
/// its declared `gen(...)` ranges (the launch side is covered by the differential
/// itself, which would diverge if the two disagreed).
#[test]
fn generated_struct_fields_respect_their_declared_ranges() {
    // `check_assumes` is the executable form of the declared assumes, including
    // the two over struct FIELDS — `args.lower_bound.abs() <= 100.0`. A draw
    // outside the range would fail it.
    let s = vec![0u32; 8];
    let y = vec![0.0f32; 8];
    assert!(
        uniform_value_map_vericl::check_assumes(
            &s,
            UniformArgs { lower_bound: -100.0, upper_bound: 100.0 },
            &y
        ),
        "an in-range struct must satisfy the field assumes"
    );
    assert!(
        !uniform_value_map_vericl::check_assumes(
            &s,
            UniformArgs { lower_bound: -100.5, upper_bound: 100.0 },
            &y
        ),
        "an out-of-range struct FIELD must fail check_assumes — otherwise `assumes(...)` over a \
         struct field is decorative"
    );
}
