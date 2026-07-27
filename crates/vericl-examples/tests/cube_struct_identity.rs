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
/// the kernel's **expansion** (design risk 6).
///
/// # Why this test looks the way it does (round-11 review, MODERATE 3)
///
/// Its previous form asserted the twin's arithmetic and that the string
/// `"cfg.window.taps"` appears in the recorded contract, then *claimed in its
/// own doc comment* that the extracted IR's loop bound differs from another
/// spelling's — an assertion it never made. Both of the things it did check stay
/// true if the pinned field never reaches the IR at all: the twin reads the
/// field from its own struct binding, and the contract string is copied from the
/// attribute tokens. The claim and the measurement were two different claims.
///
/// The measurement is now the A/B the claim describes: `stage_window_sum` and
/// `stage_window_sum_taps5` have byte-identical bodies and differ only in the
/// pinned value, so their `kernel_ir_hash`es must differ — and `kernel_ir_hash`
/// skips `kernel_name` (test (3) above is the standing proof: two differently
/// named kernels hash identically), so the two names cannot be what moves it.
#[test]
fn a_comptime_struct_field_reaches_the_twin_and_the_ir() {
    // --- the twin half ------------------------------------------------------
    // 4 elements, all 1.0, gain 1.0, bias 0.0 → a 3-tap forward sum, truncated
    // at the end of the array by the kernel's own `idx < x.len()` guard.
    let x = vec![1.0f32; 4];
    let mut y3 = vec![0.0f32; 4];
    let cfg = StageArgs { window: StageWindow { gain: 1.0, taps: 3 }, bias: 0.0 };
    stage_window_sum_vericl::reference(&x, cfg, &mut y3, x.len());
    assert_eq!(
        y3,
        vec![3.0, 3.0, 2.0, 1.0],
        "the twin must accumulate `taps = 3` forward taps — if this is 2 or 4, the pinned comptime \
         field reached the twin with a different value than the kernel"
    );

    // The twin of the 5-tap sibling, driven by ITS pin, must differ — the twin
    // is generated from the same `instantiate(...)` the kernel is.
    let x5 = vec![1.0f32; 8];
    let mut y5 = vec![0.0f32; 8];
    let cfg5 = StageArgs { window: StageWindow { gain: 1.0, taps: 5 }, bias: 0.0 };
    stage_window_sum_taps5_vericl::reference(&x5, cfg5, &mut y5, x5.len());
    assert_eq!(y5[0], 5.0, "the 5-tap twin must sum five taps: {y5:?}");

    // --- the IR half: THE assertion the doc comment has always claimed ------
    let ir3 = vericl_ir::kernel_ir_hash(&stage_window_sum_vericl::kernel_definition());
    let ir5 = vericl_ir::kernel_ir_hash(&stage_window_sum_taps5_vericl::kernel_definition());
    assert_ne!(
        ir3, ir5,
        "two kernels with byte-identical bodies, pinned at taps = 3 and taps = 5, must extract to \
         DIFFERENT IR — the comptime field becomes the `RangeLoop`'s constant bound. If these are \
         equal, the pinned value never reached the CompilationArg the IR is built from, and every \
         `proved` claim on this kernel was discharged against a loop the kernel does not have"
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
/// value the GPU is launched with — one draw, two consumers.
///
/// # Why this test looks the way it does (round-11 review, MODERATE 3)
///
/// Its previous form hand-built two `UniformArgs` values and checked
/// `check_assumes` accepted one and rejected the other. That is a test of
/// `check_assumes`, not of generation: it never called the generator, so it held
/// for any `gen(...)` ranges whatsoever, including ranges that draw outside the
/// declared assumes. It now inspects the values `generate_case` ACTUALLY draws.
///
/// Three things are asserted, and each fails under a different injection: every
/// drawn field lies inside its declared `gen(...)` range (fails if a range is
/// widened or the draw ignores it), the draws SPREAD across the range (fails if
/// a range is narrowed, or if the field is silently drawn from a constant), and
/// the pinned comptime field carries its `instantiate(...)` value into every
/// generated case (fails if the pin is dropped from the spec).
#[test]
fn generated_struct_fields_respect_their_declared_ranges() {
    // --- flat struct, two f32 fields, `gen(… in -100.0..=100.0)` ------------
    let (mut lo_min, mut lo_max) = (f32::INFINITY, f32::NEG_INFINITY);
    for seed in 0..64u64 {
        let (_s, args, _y) = uniform_value_map_vericl::generate_case(16, seed);
        for (name, v) in [("lower_bound", args.lower_bound), ("upper_bound", args.upper_bound)] {
            assert!(
                (-100.0..=100.0).contains(&v),
                "drawn `args.{name}` = {v} is outside its declared gen(...) range -100.0..=100.0"
            );
        }
        lo_min = lo_min.min(args.lower_bound);
        lo_max = lo_max.max(args.lower_bound);
    }
    // Non-vacuity: the draw must actually move across the declared range. A
    // narrowed range, or a field wired to a constant, fails here.
    assert!(
        lo_min < -80.0 && lo_max > 80.0,
        "64 draws of `args.lower_bound` spanned only [{lo_min}, {lo_max}] — that is not the \
         declared -100.0..=100.0 range"
    );

    // --- nested struct + pinned comptime field ------------------------------
    let (mut gain_min, mut gain_max) = (f32::INFINITY, f32::NEG_INFINITY);
    for seed in 0..64u64 {
        let (_x, cfg, _y) = stage_window_sum_vericl::generate_case(16, seed);
        assert!(
            (0.5..=2.0).contains(&cfg.window.gain),
            "drawn `cfg.window.gain` = {} is outside 0.5..=2.0",
            cfg.window.gain
        );
        assert!(
            (-1.0..=1.0).contains(&cfg.bias),
            "drawn `cfg.bias` = {} is outside -1.0..=1.0",
            cfg.bias
        );
        assert_eq!(
            cfg.window.taps, 3,
            "the `instantiate(cfg.window.taps = 3)` pin must be carried into EVERY generated case \
             — a drawn or defaulted value here means the twin ran a different kernel than the GPU"
        );
        gain_min = gain_min.min(cfg.window.gain);
        gain_max = gain_max.max(cfg.window.gain);
    }
    assert!(
        gain_min < 0.7 && gain_max > 1.8,
        "64 draws of the NESTED `cfg.window.gain` spanned only [{gain_min}, {gain_max}] — that is \
         not the declared 0.5..=2.0 range"
    );

    // …and the sibling pinned at 5 carries ITS value, so the assertion above is
    // about the pin and not about the number 3 being a default.
    let (_x, cfg5, _y) = stage_window_sum_taps5_vericl::generate_case(16, 0);
    assert_eq!(cfg5.window.taps, 5, "the 5-tap sibling's pin must reach its generated case too");

    // Every drawn case satisfies the declared assumes by construction — that is
    // `generate_case`'s resample contract, asserted here rather than assumed.
    let (s, args, y) = uniform_value_map_vericl::generate_case(8, 7);
    assert!(
        uniform_value_map_vericl::check_assumes(&s, args, &y),
        "a generated case must satisfy the kernel's own assumes(...)"
    );
    assert!(
        !uniform_value_map_vericl::check_assumes(
            &s,
            UniformArgs { lower_bound: -100.5, upper_bound: 100.0 },
            &y
        ),
        "…and an out-of-range struct FIELD must fail check_assumes, or `assumes(...)` over a \
         struct field is decorative"
    );
}

/// (9) MODERATE 2, the "works" direction: a `vericl::cube_struct!` type in
/// **`#[comptime]` parameter position** — the design's §6 "one type, both
/// positions", which shipped as an unconditional claim and was measured false
/// for every declared type in round 11 (`ConfigIdentity` was emitted, but
/// CubeCL's own requirement — a comptime parameter is `Debug`-formatted and its
/// `CompilationArg` derives `Hash`/`Eq` — was not met by any of them).
///
/// `ScaleCfg` is all-integer, so `cube_struct!` now emits those derives and the
/// `ConfigIdentity` impl for it, and the same type serves both positions. That
/// this module compiles at all is the primary assertion; the rest pins that the
/// two positions really do share one identity.
#[test]
fn an_all_integer_cube_struct_serves_both_parameter_positions() {
    let x: Vec<u32> = (0..8u32).collect();
    let mut y = vec![0u32; x.len()];
    // Comptime position: `instantiate(c = ScaleCfg { m: 2, n: 3 })`.
    scale_cfg_comptime_vericl::reference(&x, &mut y, x.len());
    assert_eq!(y, x.iter().map(|v| v * 2 + 3).collect::<Vec<_>>(), "{y:?}");

    // Runtime position: the same type, generated per case.
    let (gx, cfg, _) = scale_cfg_runtime_vericl::generate_case(8, 11);
    assert!((1..=4).contains(&cfg.m) && (0..=9).contains(&cfg.n), "{cfg:?}");
    let mut gy = vec![0u32; gx.len()];
    scale_cfg_runtime_vericl::reference(&gx, cfg, &mut gy, gx.len());
    assert_eq!(gy, gx.iter().map(|v| v * cfg.m + cfg.n).collect::<Vec<_>>());

    // One block, one hash, both positions: each kernel folds `ScaleCfg`'s hash
    // into its own identity, through a different trait.
    for id in [
        scale_cfg_comptime_vericl::identity().source_hash,
        scale_cfg_runtime_vericl::identity().source_hash,
    ] {
        assert!(id.starts_with("sha256:"), "{id}");
    }
    assert_ne!(
        scale_cfg_comptime_vericl::identity().source_hash,
        scale_cfg_comptime_vericl::SOURCE_HASH,
        "the comptime-position kernel must fold ScaleCfg's CONFIG_HASH into its identity"
    );
    assert_ne!(
        scale_cfg_runtime_vericl::identity().source_hash,
        scale_cfg_runtime_vericl::SOURCE_HASH,
        "the runtime-position kernel must fold ScaleCfg's STRUCT_HASH into its identity"
    );
}

/// (10) MODERATE 2, the other direction, and the round-11 enum fix.
///
/// A **float-field** declared struct cannot occupy `#[comptime]` position at any
/// price (`f32` is neither `Hash` nor `Eq`), so `cube_struct!` withholds
/// `ConfigIdentity` from it and the author lands on that trait's
/// `on_unimplemented` note naming the reason — instead of three raw rustc trait
/// errors pointing at `#[cube(launch)]`, which is what the unconditional impl
/// produced. There is no `trybuild` harness here, so the negative direction is
/// pinned where it is decidable: `vericl-macros`' own
/// `config_identity_and_the_comptime_derives_track_field_hashability`.
///
/// What this test can assert end to end is the sibling fix: a declared unit
/// **enum** as a `#[cube(comptime)]` FIELD — a shape CS2 has always admitted and
/// which did not compile at all before round 11.
#[test]
fn a_unit_enum_comptime_field_declares_launches_and_pins() {
    let x = vec![2.0f32, -1.0, 0.5, 4.0];
    let mut y = vec![0.0f32; x.len()];
    let p = BlendCfg { gain: 3.0, mode: Blend::Double };
    blend_mode_map_vericl::reference(&x, p, &mut y, x.len());
    assert_eq!(y, x.iter().map(|v| v * 3.0).collect::<Vec<_>>(), "{y:?}");

    // The pin reaches every generated case, and the drawn runtime field stays
    // in its declared range.
    let (_gx, gp, _gy) = blend_mode_map_vericl::generate_case(8, 3);
    assert_eq!(gp.mode, Blend::Double, "the pinned comptime enum field must reach the case");
    assert!((0.5..=2.0).contains(&gp.gain), "drawn gain {} out of range", gp.gain);

    // The enum carries ConfigIdentity (comptime positions) and NOT
    // StructIdentity — asserted by the fact that this compiles at all plus the
    // macro-level test; here we pin that the owning struct folds a hash.
    assert_ne!(
        blend_mode_map_vericl::identity().source_hash,
        blend_mode_map_vericl::SOURCE_HASH,
        "the kernel must fold BlendCfg's STRUCT_HASH"
    );
}

/// (11) LOW 5: a declared STRUCT as a `#[cube(comptime)]` field, pinned WHOLE.
///
/// Unlocked by the same round-11 derive detection as (10) — CubeCL's generated
/// `CompilationArg` derives `Hash`/`Eq` over every comptime field, and before
/// round 11 `cube_struct!` emitted neither. The whole-value pin is the honest v1
/// surface: there is no `gen(p.win.taps in …)` form, and the nested alias for
/// this path now points at a marker type whose name says so, instead of at
/// `Win__VericlSpec` (which made the mismatch a raw `E0308`).
#[test]
fn a_declared_struct_comptime_field_is_pinned_whole() {
    // taps = 3, stride = 2, gain = 1.0 → x[i] + x[i+2] + x[i+4], truncated.
    let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mut y = vec![0.0f32; x.len()];
    let p = WinHost { gain: 1.0, win: Win { taps: 3, stride: 2 } };
    strided_window_sum_vericl::reference(&x, p, &mut y, x.len());
    assert_eq!(y[0], 0.0 + 2.0 + 4.0, "{y:?}");
    assert_eq!(y[1], 1.0 + 3.0 + 5.0, "{y:?}");
    assert_eq!(y[7], 7.0, "the guard must truncate at the end: {y:?}");

    // The whole pinned struct reaches every generated case…
    let (_gx, gp, _gy) = strided_window_sum_vericl::generate_case(16, 5);
    assert_eq!(gp.win, Win { taps: 3, stride: 2 }, "the whole-struct pin must reach the case");
    assert!((0.5..=2.0).contains(&gp.gain), "drawn gain {} out of range", gp.gain);

    // …and it reaches the extracted IR: `taps`/`stride` become the loop's
    // constants, so this kernel's ir_hash differs from the 3-tap unit-stride
    // `stage_window_sum`'s. (A weaker check than (7)'s A/B, but it is the same
    // property: a comptime field that never reached the CompilationArg would
    // make every proved obligation here describe a loop the kernel lacks.)
    assert_ne!(
        vericl_ir::kernel_ir_hash(&strided_window_sum_vericl::kernel_definition()),
        vericl_ir::kernel_ir_hash(&stage_window_sum_vericl::kernel_definition()),
    );
}
