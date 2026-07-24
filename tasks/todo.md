# VeriCL build plan

Working toward the four first-release outcomes in README.md.

## M0 — De-risk CubeCL 0.10 (critical path) — DONE
- [x] Workspace scaffold: `crates/vericl`, `crates/vericl-macros`, `crates/vericl-examples`
- [x] Vanilla `#[cube(launch)]` axpy compiles and runs on wgpu (Metal): exact match, n=1027
- [x] 0.10 API confirmed: `ABSOLUTE_POS` is `usize`, scalars pass directly as launch args,
      `create_from_slice` / `read_one(handle)` / `ArrayArg::from_raw_parts(handle, len)`

## M1 — Contract macro (`#[vericl::kernel]`) — DONE
- [x] Passes the cube kernel through untouched; generates sibling `<name>_vericl` module
- [x] Parses `assumes(...)` and `compare(exact | max_ulp = N | abs = X[, rel = Y])`
- [x] Sequential scalar reference twin: `ABSOLUTE_POS` → loop var, `&Array<T>` → `&[T]`
- [x] `check_assumes` — assumes clauses as an executable predicate (iterator exprs work)
- [x] `SOURCE_HASH` identity (source tokens + contract + vericl version)
- [x] Out-of-subset constructs rejected at compile time (topology, SharedMemory, plane_*,
      comptime, vectors, `return`, generics, #[comptime] params)

## M2 — Core library — DONE
- [x] `compare`: ULP distance, max-ulp compare, abs+rel compare (NaN always fails), exact u32
- [x] `rng`: SplitMix64, dependency-free seeded generation
- [x] `evidence`: manifest, claims (proved/tested/assumed kinds + trusted list), `verify`
      with hard staleness rejection
- [x] vericl core has NO cubecl dependency (independence of the reference/evidence layer)

## M3 — Conformance path — DONE
- [x] `conform update|check|demo-defects` binary; evidence at `evidence/vericl.json`
- [x] axpy + xorshift_step pass differentially on wgpu/Metal across 7 sizes
- [x] Staleness demo verified: mutated kernel → `check` fails with identity mismatch, exit 1
- [x] `axpy_off_by_one` caught: reference panics on OOB where WGSL robustness silently clamps
      (and passes at sizes that are multiples of the cube dim — the classic missed bug)
- [x] `sum_racy` caught: GPU race leaves ~0.78 where the true sum is ~4117
- [x] Twin-derivation guarded by unit tests against handwritten scalar code

## M4 — In progress (delegated to Sonnet 5 dev agents)
- [x] cubecl-cpu as additional differential lane — DONE (works on macOS arm64 via prebuilt
      MLIR/LLVM; second Tested claim per kernel, trusted list records shared front-end,
      wgpu-only evidence byte-identical; spot-checked: clippy 0, check passes)
- [x] Wrapping-arithmetic `wrapping` contract clause — DONE (syn `Fold` over the reference twin
      body only — the `#[cube]` kernel is re-emitted untouched; `+`/`-`/`*` → `wrapping_{add,
      sub,mul}`, `<<`/`>>` and compound-assign forms → masked-amount `wrapping_{shl,shr}`
      matching WGSL. Integer-only subset gate (u32/i32/u64/i64 params) rejects mixed float/int
      kernels at compile time, since the fold is untyped. `Contract`/`ContractRecord` gained
      `wrapping: bool`; already covered by `SOURCE_HASH` (hashes the raw, unparsed contract
      attribute tokens). New `mix_u32` murmur3-fmix32-style example kernel, `wrapping` +
      `compare(exact)`, wired into both conform.rs lanes; passes bit-exact against wgpu and
      cubecl-cpu across all 7 sizes. clippy 0 (default + `--features cpu`); `cargo test
      --workspace` and `-p vericl-examples --features cpu` pass; `conform update` → `check` →
      `demo-defects` all pass)
- [x] CubeCL IR access research — DONE, findings + validated prototypes in docs/ir-research.md
      and docs/prototypes/. Headlines: IR extractable with zero client (call `<name>::expand`
      with hand-built KernelBuilder + AddressType::U32.register); deterministic SHA-256 via
      Scope's curated Hash impl (never use == on Scope — Allocator PartialEq is Rc identity);
      solver decision: easy-smt + subprocess z3 (validated UNSAT/SAT on the axpy obligation).
- [x] SMT bounds checking over cubecl IR — DONE, new cubecl-dependent `crates/vericl-ir` (kept out
      of core `vericl` by design). Recursive walker over `Scope.instructions` with an SMT push/pop
      path-condition stack; QF_LIA via easy-smt + subprocess z3. Values are substituted expression
      trees, not per-variable SMT constants — only genuine leaves (`AbsolutePos`, integer
      `GlobalScalar`s, per-buffer `Length`, `RangeLoop` induction vars) get a declared constant.
      Unsupported ops (`Bitwise`, `Atomic`, float arithmetic) taint their output instead of
      aborting the whole kernel, so `xorshift_step`/`mix_u32` still prove even though their bodies
      use unmodeled bitwise ops — those values never feed an index expression (every index is a
      bare `ABSOLUTE_POS`); a tainted value only fails, explicitly, at the obligation/branch site
      that actually needs it. `RangeLoop` modeled as a fresh var in `[start, end)` with no
      unrolling, guarded against loop-carried mutation (accumulators) which a single symbolic pass
      can't soundly represent — rejected as out-of-subset rather than mismodeled. Macro gained
      `kernel_definition()` (adapts the prototype's zero-client `KernelBuilder` recipe) and
      `BUFFER_PARAMS` (array param name + is-output, in registration order — the single point of
      custody vericl-ir needs to map IR `input(i)`/`output(j)` back to param names); `assumes(...)`
      clauses of the form `A.len() == B.len()` / `A.len() == <literal>` are additionally parsed
      into `vericl::StructuredAssume` for the prover to bind `Length` variables from — unrecognized
      clauses stay string-only (sound: fewer constraints never cause a false Proved). conform.rs
      adds a `Proved`/`smt-oob-freedom` claim (config: solver + version, `QF_LIA`, obligation
      count) for axpy/xorshift_step/mix_u32; `axpy_off_by_one` REFUTES with a printed
      counterexample (position == length) in `demo-defects`, `sum_racy`'s bounds separately PROVE
      (`LenEqConst` from `assumes(y.len() == 1)`) even though its race still fails differentially
      — the two claim kinds stay visibly distinct in the demo output. 9 vericl-ir unit tests (hash
      determinism, the `Scope==Scope` Allocator-identity trap pinned per the research doc, guarded/
      unguarded/loop positive+negative prover controls) plus 6 new vericl-examples integration
      tests exercising the full macro → IR → prover path; `cargo test --workspace`, `-p
      vericl-examples --features cpu`, clippy 0 (default + `--features cpu`), and the staleness
      cycle (mutate → `check` fails exit 1 reporting both `source_hash` and `ir_hash` mismatches →
      revert → `check` passes) all verified end to end.
- [x] IR-level identity hash — DONE, same agent/crate as SMT above (shared extraction plumbing).
      `vericl_ir::kernel_ir_hash` reproduces the research doc's validated `sha256:3ae1a32f...` for
      axpy exactly. `Identity` gained `ir_hash: Option<String>` (`#[serde(default)]`, `None` only
      for evidence produced without IR access — core `vericl` still can't compute it, by design);
      the harness sets it after computing it via vericl-ir, so `verify()`'s existing whole-`Identity`
      comparison now catches IR-level drift (e.g. a CubeCL-upgrade codegen change with no source
      diff) in addition to source-level drift, with both hashes reported on mismatch.
- [x] Absorb per-kernel GPU glue into generated code — DONE, see "Roadmap" item 4 below for the
      full writeup (`gen(...)` clause, `conformance_case`, `vericl::suite!`). Standalone
      `vericl check` CLI remains not done (superseded by the `cargo test` CI story — see README
      CI story row and Roadmap item 6).
- [x] Adversarial soundness review of the SMT bounds prover — DONE. One CRITICAL confirmed bug:
      `process_range_loop` (crates/vericl-ir/src/prover.rs) never read `rl.step`, so a
      `range_stepped` (CubeCL stepped-range) loop — including a genuinely descending loop where
      `start > end` numerically — got the same unconditional ascending `start <= i (<)= end`
      assertions as an ordinary `for`. For a real descending loop that makes the SMT context
      infeasible, so every obligation inside discharges vacuously as "proved" regardless of the
      body — demonstrated false-Proved: a negative-step loop body writing `y[100000]` returned
      `Proved { obligations: 2 }` although a real (sequential) run of that loop panics
      out-of-bounds. Fixed by rejecting any `rl.step.is_some()` outright as `OutOfSubset`
      ("stepped range loop (range_stepped) is outside the vericl v0 subset...") before any bounds
      assertion is pushed, per the "rejected rather than silently approximated" principle —
      stepped/descending loops are not modeled, not approximated. Regression tests: vericl-ir
      `prover::tests::stepped_range_loop_is_out_of_subset` (bare `#[cube(launch)]` + KernelBuilder,
      same layer as the existing loop-carry test) and, in the stronger macro-integration form,
      vericl-examples `tests::stepped_range_loop_is_out_of_subset` +
      `tests::stepped_loop_cannot_vacuously_prove` (the latter is the exact `y[100000]` vacuous-
      proof shape from the review; confirmed by temporarily disabling the guard that it reproduces
      `Proved { obligations: 2 }` pre-fix). The three good kernels (axpy, xorshift_step, mix_u32)
      use no stepped loops, so their obligation counts are unaffected — `evidence/vericl.json` is
      byte-identical before/after the fix. Also fixed one cosmetic issue found in the same review:
      `conform.rs`'s `describe_outcome` hard-coded the bounds/WGSL-robustness narrative onto every
      `reference_panic`, which would mislabel e.g. a `wrapping` kernel's reference twin panicking
      on division-by-zero as a bounds defect; now gated on the panic message containing "index out
      of bounds", with a neutral "divergent semantics or defect" framing otherwise (OOB wording for
      `axpy_off_by_one` unchanged — verified via `demo-defects` output diff). All other attack
      surfaces the review probed survived without changes needed: u32 wraparound both directions,
      tainted conditions, `IfElse` negation, loop-carried mutation (including a local-array bypass
      attempt), length aliasing, the `wrapping` fold on real GPU across profiles, and the no-z3
      error path. One accepted low-severity gap: `Identity` does not record the CubeCL crate
      version — mitigated by the exact `=0.10.0` pin in `Cargo.toml` and the documented trust
      boundary; folding the cubecl version into `Identity` is future work.

## Decisions made during build
- Reference execution: macro-generated twin (independent — shares only source text), with
  cubecl-cpu later as a secondary, shared-front-end lane.
- Kernel identity v0: source tokens + contract + vericl version. IR-level hash deferred until
  IR access is wired for SMT work anyway.
- Comparison model: `Exact`, `MaxUlpF32`, and `AbsRelF32 {abs, rel}` — the last added after
  the fma finding (below); tolerances must be justified by `assumes` input ranges.
- vericl core does not depend on cubecl, by design.

## Review

**All four first-release outcomes demonstrated** on wgpu/Metal, CubeCL 0.10.0 pinned.

Notable finding (first real one): wgpu/Metal contracts `a*x + y` into fma; under cancellation
divergence from the strict-rounding reference reached ~27k ULP. A ULP tolerance is the wrong
claim shape for contracted float kernels; an abs+rel bound derived from declared input ranges
is honest. This drove the `AbsRelF32` comparison mode and is written up in the README.

Verification: 11 unit tests pass, clippy clean, `conform update` → `check` → mutate → `check`
(fails stale) → revert → `check` (passes) cycle exercised end-to-end; both defective kernels
caught deterministically.

## Roadmap (agreed 2026-07-22)

1. [DONE 2026-07-22] Dogfooded privately against 22 production kernels — full findings in
   docs/dogfood-2026-07.md. Headline: generics block 20/22, composition 16/22, comptime 15/22;
   Tensor/2D speculation withdrawn (zero uses); wrapping clause independently validated by a
   real kernel; terminate!() latent soundness gap found and banned. Roadmap below reordered
   accordingly — new order: instantiate() clause (generics+comptime), composition, prover
   div/mod + loop-carry refinement, shared-memory reductions last.
2. [DONE 2026-07-22] `instantiate(...)` contract clause — generic (`<F: Float>`) kernel and
   `#[comptime]` parameter support, monomorphized at declared concrete values (roadmap item 1,
   unblocks 20/22 dogfooded kernels). Design: `instantiate(F = f32, taps = 3)` — one clause per
   kernel (v0), type params get concrete types, `#[comptime]` params get concrete literal
   values. Gating replaces the old blanket "generic kernels are outside the vericl v0 subset"
   rejection with three targeted errors: a kernel with generics/comptime and no clause ("add
   one, e.g. `instantiate(F = f32, N = 8)`"); a clause on a kernel with neither ("unused
   instantiation is a contract lie"); and a duplicate clause. Only plain type generic params
   are supported (lifetimes/const generics/where-clauses still rejected outright).
   Monomorphization: the generic ident is substituted token-wise into the twin's signature
   (via a substituted, reparsed param list feeding the *same* `classify_param`/`NumKind`/
   `gen(...)` machinery every other kernel already uses — no downstream function needed to
   learn about instantiate() at all) and body (extending the existing `transform_body`
   ABSOLUTE_POS/banned-ident walk with a substitution map); `#[comptime]` params are removed
   from the twin signature and bound as `let name: ty = value;` consts at the top of
   `reference`/`check_assumes` (loop-invariant by construction); `kernel_definition()` calls
   `expand::<f32, ...>(...)` and `conformance_case` calls `launch::<f32, ..., R>(...)` with
   comptime values spliced in at their declared parameter position (cubecl keeps a comptime
   param in its original position with its plain type — confirmed from cubecl-macros'
   `generate/launch.rs`). Two new syn `Fold` passes over the twin body (added to the existing
   unconditional block-reparse, so they cost nothing for kernels that don't need them):
   `StripUnrollFold` removes the perf-only `#[unroll]`/`#[unroll(n)]` statement attribute from
   twin loops (invalid in plain Rust) and errors on any *other* statement attribute instead of
   silently dropping it; `FloatMethodCheck` rejects any call (`.method()` or `Type::method()`)
   to a name on `FLOAT_METHOD_REJECT`.
   **Float-method host-callability** (the CRITICAL research item) — empirically verified (not
   just read from source) via `crates/vericl-examples/tests/float_method_whitelist.rs`, which
   calls every candidate on host `f32` and cross-checks against `std`/confirms a panic:
   `FLOAT_METHOD_WHITELIST` (new, abs, min, max, clamp, floor/ceil/round/trunc, sqrt, recip,
   sin/cos/tan/asin/acos/atan/atan2, sinh/cosh/tanh, exp, ln, powf, powi, hypot, is_nan,
   to_degrees, to_radians, from_int, min_value, max_value) are host-safe — most because Rust's
   inherent-method resolution always prefers `std`'s own `f32` method over the trait's
   `unexpanded!()`-panicking default, a few (`new`, `from_int`, `min_value`, `max_value`) via a
   real per-type implementation. `FLOAT_METHOD_REJECT` (log1p, inverse_sqrt, erf, is_inf,
   rhypot, magnitude, normalize, dot, mul_hi, saturating_add, saturating_sub, from_int_128,
   from_vec, cast_from, reinterpret) panic on host (`Unexpanded Cube functions should not be
   called.`) and are rejected at macro time naming the method — `cast_from`/`reinterpret` were
   *added* to this list mid-task, found by the private dogfood validation below (see its
   entry), a genuine example of real-code dogfooding sharpening a whitelist built first from
   source reading alone. Separately (also found via dogfooding, recorded in code comments, not
   yet reflected in the whitelist since it's a different axis): `new`/`from_int` additionally
   require a *compile-time-constant* argument even in GPU-expand context — passing either a
   genuinely runtime-computed value compiles (for `from_int`) or doesn't (for `new`) but panics
   or fails independent of vericl the moment it's actually expanded/launched. Host-callable and
   expand-runtime-safe are different, currently-undocumented-until-now axes; worth a dedicated
   `FLOAT_METHOD_CONST_ONLY` distinction as follow-up if a dogfooded kernel needs it.
   Examples: `axpy` converted to `axpy<F: Float + CubeElement>` with `instantiate(F = f32)` —
   the flagship shows the feature (the `+ CubeElement` bound is required by cubecl itself for
   any kernel with a bare scalar generic parameter, unrelated to vericl — confirmed against
   cubecl's own `kernel_with_generics` test pattern, where the bound lives on the *caller*
   instead since cubecl's own test never has vericl's generated code calling `launch` with an
   already-concrete type). New `fir3<F: Float>(x: &Array<F>, y: &mut Array<F>, #[comptime]
   taps: u32)` — a clean-room windowed FIR, taps 1..=3 selected by the comptime `taps` value —
   is the milestone's headline: genuinely generic *and* comptime, and its bounds obligations
   still discharge `Proved` (4 obligations), not merely `OutOfSubset`, by deliberately avoiding
   a loop-carried accumulator (guarding each extra tap with its own nested `if` rather than a
   `for k in 0..taps` loop — confirmed empirically that collapsing to `if taps > 1 &&
   ABSOLUTE_POS >= 1` turns this from `Proved` into `OutOfSubset`: the prover does not compose
   `&&`-joined branch conditions, only nested `if`s, individually, on its path-condition stack
   — now a `#[allow(clippy::collapsible_if)]` with that exact finding recorded in a comment).
   [UPDATED 2026-07-22, see roadmap item 5 below: boolean condition composition is now modeled.
   `fir3` has since moved to the `&&`-composed form (`taps > 1 && ABSOLUTE_POS >= 1`) as its
   primary/public shape and still proves; the `#[allow(clippy::collapsible_if)]`/nested-if
   workaround this paragraph describes was removed from the public example. The nested-if
   shape's provability (a genuinely different code path in the prover — the SMT push/pop
   path-condition stack, rather than an `Operator::And` term) remains independently pinned, now
   as a `vericl-ir` unit test rather than a second public example:
   `prover::tests::nested_if_guard_still_proves`.]
   `fir3_alt` (same shape, `taps = 1`) exists solely to show instantiate() changes
   `SOURCE_HASH`. `suite!` runs `axpy`/`fir3` unchanged alongside the pre-existing kernels — no
   suite-side change needed, proving the "monomorphize once, everything downstream just works"
   design. `Contract`/`ContractRecord` gained `instantiate: &[&str]`/`Vec<String>`
   (`#[serde(default)]` on the record field so evidence written before this feature still
   loads).
   Private validation (per the README's private-codebase policy: never committed, described here only
   by construct class): one real generic + `#[comptime]` production launch kernel, blocked
   *only* by the generics/comptime gate (no composition, no shared-memory topology) per the
   survey, passed differentially on wgpu end-to-end across 5 sizes after `instantiate(...)`
   annotation; its bounds proof is honestly `OutOfSubset` on a pre-existing, separately
   documented div/mod-index gap (nothing to do with instantiate()). Needed two documented,
   semantics-preserving adaptations to compile at all under the v0 subset — both are genuine
   subset-gap findings fed back into the public whitelist/rejection lists above, not
   workarounds-of-convenience: (1) a `usize -> F` runtime index conversion via `F::cast_from`
   (blanket-`unexpanded!()`, panics on host for every type) replaced with a small precomputed
   lookup-table array read (the same pattern the kernel's own other float lookups already use);
   (2) two `comptime!(...)` macro blocks (still separately banned, unrelated to instantiate())
   dropped as a no-semantic-change codegen-hint removal, same class as the pre-existing
   `#[unroll]`-dropping precedent in that private workspace.
   Verification: `cargo test --workspace` and `-p vericl-examples --features cpu` both green;
   `cargo clippy --workspace --all-targets` zero warnings on both feature sets; `VERICL_UPDATE=1
   cargo test` then plain `cargo test` green (fresh evidence, `axpy`+`fir3` both carry
   `tested`+`proved` claims); `conform demo-defects` exits 0; the stale-evidence cycle (mutate
   `fir3`'s guard → fails naming both hashes → revert → passes) exercised end to end; the
   no-instantiate and unused-instantiate errors demonstrated in the scratchpad (not committed);
   the private dogfood suite (a widening 32x32->64-bit product-split kernel, a counter-based
   block-cipher RNG kernel, and a cos/sin synthesis kernel — construct classes only, per the
   private-codebase policy) green end to end, `dogfood-rejects` still fails to build with
   its generics-blocked kernel now naming the *new* targeted "add instantiate(...)" error
   instead of the old blanket one (confirming the replacement fires correctly) while its
   topology-blocked variant is unaffected.
3. CI: DEFERRED per Ryland (2026-07-22) — no GitHub Actions or remote execution for now;
   everything stays local. The CI story is `cargo test --workspace` (+ `--features cpu`) run
   locally. A workflow existed briefly and was removed in ff675ec (recoverable from e869646);
   do not re-add remote CI without an explicit ask.
4. [x] Ergonomics: absorb per-kernel GPU launch glue into the macro — DONE. `#[vericl::kernel]`
   gained a `gen(...)` contract clause (`name in lo..=hi` per parameter, elementwise for arrays;
   optional `len(name = N)` to pin an array's generated length instead of the case size — needed
   by `sum_racy`'s `assumes(y.len() == 1)`) and now generates `<name>_vericl::conformance_case`,
   which draws inputs via vericl's `SplitMix64` (declaration order, deterministic, resampling up
   to 64 times against `check_assumes` before erroring), runs the reference and the real kernel
   with standard 1D dispatch, and compares every `&mut Array` param against the reference
   (reporting the param name on mismatch). Deliberate ergonomic decision: a float parameter with
   no declared `gen(...)` range is a **compile-time error**, not a silent unbounded default —
   unbounded float generation produces NaN/inf-adjacent garbage and tolerances no
   `compare(abs = ...)` can honestly justify, and that's far more useful caught at authoring time
   than as a confusing runtime NaN mismatch. New `vericl::suite!` (proc-macro in `vericl-macros` —
   chosen over `macro_rules!` in core because the DSL's several optional, order-independent,
   defaulted fields need real parsing with error spans, which is exactly what `parse_contract`
   already does for the kernel attribute; `vericl-macros` still never depends on `cubecl` itself,
   it only emits tokens that reference `::cubecl::`/`::vericl_ir::` paths at the call site, same as
   `kernel_definition()` already did) expands to a `#[test] fn vericl_conformance()`: runs every
   listed kernel's `conformance_case` across the declared sizes, discharges the SMT bounds proof
   when `prove` is enabled (default; missing z3 is now an actionable compile-time-style panic
   naming the `brew`/`apt` install command, not a silent skip), and assembles evidence exactly in
   the existing schema — `VERICL_UPDATE` set writes it, otherwise it verifies against what's on
   disk and panics with the problem list, so `cargo test` is the whole CI story. Multi-lane
   (`--features cpu`) is an optional `extra_lane: (cfg(...), RuntimePath)` DSL field folded into
   the *same* test via `#[cfg(...)]` on a block, rather than a second hand-written `#[test]` —
   two independent tests sharing one evidence file would race under `cargo test`'s unordered
   execution and try to write two different claim shapes to the same manifest. `evidence/vericl.json`
   moved from the workspace root to `crates/vericl-examples/evidence/vericl.json`
   (`CARGO_MANIFEST_DIR`-relative, the idiomatic cargo convention, instead of a hand-counted
   `../../` from the harness binary). `vericl` core gained `catch_reference_panic` (the
   panic-hook-silencing helper, moved out of `conform.rs`), `describe_case_outcome` +
   `CaseOutcome::pass` (`CaseOutcome.report: Option<CompareReport>` became
   `reports: Vec<(String, CompareReport)>` — one entry per compared `&mut Array` param, so a
   multi-output kernel's mismatch names the offending param), `compare_f32_with`/`compare_u32_with`
   (dispatch a declared `Compare` against a known element type), `differential_config`/
   `proved_config` (claim `config` JSON, shared instead of duplicated), and the `trust` module
   (`reference_twin_trust`, `backend_buffer_trust`, `GPU_HARDWARE_TRUST`, `proved_bounds_trust`,
   `shared_frontend_lane_trust` — the wording `conform.rs` used to hand-duplicate). `verify()`
   gained a downgrade check: a stored `Proved` claim with no matching claim in the current build
   (e.g. `prove: false`, or z3 going missing) is now a reported problem, not a silent pass — with
   regression tests `dropped_proved_claim_is_a_downgrade` /
   `retained_proved_claim_is_not_a_downgrade`. `conform.rs` shrank to demo-defects mode only (729
   → 149 lines), reusing `conformance_case` for the defect kernels too; `tests/conformance.rs`
   (new, 22 lines) replaces the old `update`/`check` machinery — 729 lines of hand-written
   per-kernel harness in the examples crate became 171, and that 171 no longer grows per kernel
   (adding a 4th honest kernel to the suite is one name in `kernels: [...]`, not a new ~100-line
   `run_*` function). Verification: `cargo test --workspace` green without `VERICL_UPDATE` (fresh
   evidence committed), `--features cpu` variant green (evidence gains the cpu-lane claims only
   when the feature + `VERICL_UPDATE` produce it; the default evidence shape is unchanged),
   `cargo clippy --workspace --all-targets` zero warnings on both feature sets, `conform`
   demo-defects still exits 0, the stale-evidence negative test (mutate `axpy`'s guard → `cargo
   test` fails naming both `source_hash` and `ir_hash` → revert → passes) exercised end to end,
   and the float-without-`gen` compile rejection demonstrated in a standalone scratch crate.
   Standalone `vericl` CLI remains future work (see README CI story row).
5. [DONE 2026-07-22] Three prover-subset expansions (roadmap item 4's div/mod +
   loop-carry refinement, plus a boolean-composition gap found while building `fir3` above),
   sound-by-construction, plus the `flatten_decode_scale` public example kernel
   (docs/dogfood-2026-07.md candidate #1) — Tier-2 table there annotated `[implemented]`.
   All three land in `crates/vericl-ir/src/prover.rs`; see that file's module docs for the full
   soundness argument behind each (this entry summarizes).

   **IR findings** (docs/ir-research.md §3, validated empirically the same way as the original
   IR-access research — extracting IR for small probe kernels, not read from source alone):
   CubeCL 0.10 lowers `&&`/`||`/`!` to **eager** `Operator::And`/`Or(BinaryOperator)` and
   `Operator::Not(UnaryOperator)` over already-evaluated `Bool` sub-expressions — *not* to
   nested branches as speculated in the task brief. `/`/`%` lower to ordinary
   `Arithmetic::Div`/`Modulo(BinaryOperator)`, no different in IR shape from `Add`/`Sub`/`Mul`.

   **Boolean condition composition:** `And`/`Or`/`Not` modeled directly as SMT `and`/`or`/`not`
   over recursively-resolved operands (`Prover::bool_binary`/`bool_unary`), plus `Bool` constants
   in `constant_expr` (a natural companion, strictly sound). A tainted sub-condition taints the
   whole composed condition, same discipline as everywhere else. `fir3` converted from a nested-
   `if` workaround to the natural `&&` form (see the [UPDATED] note on roadmap item 2 above) and
   still proves (4 obligations, unchanged). Tests: `prover::tests::{and,or,not}_guard_proves`
   (positive) and `and_guard_insufficient_refutes`/`or_guard_insufficient_refutes` (negative — an
   `&&`/`||` guard whose arms don't actually protect the access still `Refuted`, confirming
   composition doesn't over-prove) plus `nested_if_guard_still_proves` (regression pinning the
   *other* condition-composition shape, the path-condition stack, moved here from being
   implicitly covered by `fir3`'s old shape).

   **Div/mod-derived indices:** `Arithmetic::Div`/`Modulo` modeled with SMT-LIB `div`/`mod`
   (Euclidean) — but only when a solver-discharged internal side-obligation (divisor nonzero,
   both operands nonnegative, checked fresh via `Prover::try_discharge` under the *live* path
   conditions, not inferred from the operands' declared unsigned types) actually proves;
   otherwise the result is left tainted, never hard-errored (`Prover::divmod_int`). This
   side-obligation is deliberately not counted in the public `obligations` total (it's an
   internal modeling precondition, not a bounds check). z3 handles a symbolic (non-constant)
   divisor fine in practice, including deriving `a == b*(div a b) + (mod a b)` from the theory's
   own axioms — load-bearing for `flatten_decode_scale` below, which recombines a decoded index
   and relies on the solver connecting it back to the original guard. Tests:
   `prover::tests::div_guarded_proves`/`mod_guarded_proves` (positive),
   `div_unguarded_divisor_is_out_of_subset` (negative/taint: divisor possibly zero → the
   dependent index obligation fails as `OutOfSubset`, never `Proved`),
   `div_index_unbounded_refutes` (negative/refute: a genuinely-unsafe decode where the divisor
   guard discharges but nothing bounds the resulting index → `Refuted`, asserted to be
   specifically the `y` write obligation).

   **Loop-carry refinement:** replaces the old wholesale "reject any loop that reassigns a
   variable bound outside it" with tainting exactly the reassigned (carried) variables — via the
   same `memo`/taint machinery as any other unsupported construct — for the loop body's whole
   walk (`Prover::carried_stack`, consulted by `bind_out`/`taint_out`) and, defensively, again
   after the loop returns. Everything else in the loop, and every other loop, is still modeled
   exactly as before. CAREFUL design point honored: the induction-variable handling and the
   stepped-loop rejection (`rl.step.is_some()`) are untouched, run first, and are unaffected by
   this refinement — reran `stepped_range_loop_is_out_of_subset` (`vericl-ir` and
   `vericl-examples`) and `stepped_loop_cannot_vacuously_prove` (the exact vacuous-proof shape
   from the earlier adversarial review) after the change: unchanged pass, obligation counts
   unaffected. `scope_reassigns_any` (found the *first* carried variable, used only to reject)
   replaced by `scope_reassigned_vars`/`collect_reassigned_vars` (collects the *whole* set, used
   to taint). Tests: `prover::tests::loop_carried_accumulator_unused_as_index_proves` (positive —
   the regression this refinement exists for: an accumulator whose index/branch expressions
   never touch carried state now proves, 2 obligations) and
   `loop_carried_accumulator_used_as_index_is_out_of_subset` (renamed/updated negative control —
   an index literally derived from the carried accumulator is still never `Proved`; the reason
   string changed from a wholesale "loop-carried" rejection to a use-site "write index... depends
   on a construct outside the vericl v0 subset", since the loop itself is no longer rejected
   outright).

   **`flatten_decode_scale`** (`crates/vericl-examples/src/lib.rs`): 1-D dispatch, `row =
   ABSOLUTE_POS / width`, `col = ABSOLUTE_POS % width` (a plain runtime `u32` parameter, not
   `#[comptime]` — the modeling has to hold for a symbolic divisor), guarded write at the
   *recombined* `row * width + col` scaled by a factor. Contract: `assumes(x.len() ==
   y.len())`, `compare(abs = 1e-4)`, `gen(x in -100.0..=100.0, y in 0.0..=0.0, width in 1..=64,
   scale in 0.1..=4.0)`. Wired into `vericl::suite!` alongside the other honest kernels — no
   suite-side change needed beyond adding the name. Carries both a `tested` (differential,
   wgpu, 7 sizes) and a `proved` (`smt-oob-freedom`, 2 obligations: the `x` read and the
   recombined-index `y` write) claim in `evidence/vericl.json` — the milestone headline, per the
   task brief. Twin-derivation guarded by `flatten_decode_scale_twin_matches_handwritten`
   (independent row/col arithmetic, same pattern as `fir_handwritten`/`fmix32`) and
   `_twin_respects_guard`.

   Verification: full existing prover regression suite green, unchanged — 21/21 `vericl-ir` unit
   tests (10 pre-existing, one renamed/updated in place for the loop-carry refinement + 11 new: 3
   boolean-composition positive, 2 negative, 1 nested-if regression pin, 3 div positive/negative/
   refute controls, 1 mod positive, 1 loop-carry positive) plus 23/23 `vericl-examples` lib tests
   (19 pre-existing + 4 new
   `flatten_decode_scale_*`); `cargo test --workspace` and `-p vericl-examples --features cpu`
   both green; `cargo clippy --workspace --all-targets` zero warnings on both feature sets (one
   `clippy::nonminimal_bool` fix needed on the deliberately-non-simplified `!` test guard);
   `VERICL_UPDATE=1 cargo test` (default features) then plain `cargo test` green — fresh evidence
   for all five honest kernels including `flatten_decode_scale`'s new `tested`+`proved` pair; a
   `--features cpu` `VERICL_UPDATE=1` pass was run and verified green too, then the *default*
   `VERICL_UPDATE=1` was run last (after it) to leave the committed evidence in the default
   (non-cpu) shape, per the "run VERICL_UPDATE as the LAST thing you do" staleness-guard lesson
   from the earlier adversarial review; `conform demo-defects` exits 0, output unchanged (neither
   defect kernel touches `&&`/div/mod/loop-carry).
6. [DONE 2026-07] Next proved property: race-freedom via two-thread symbolic reduction — the
   shared-memory milestone (`docs/design-shared-memory.md`), delivered M1–M7. `smt-race-freedom`
   is now a live second proved property alongside `smt-oob-freedom`: a GPUVerify-style two-thread
   reduction (`t1 != t2` over one cube; per-phase write-write / read-write / inter-cube
   single-writer disjointness + barrier uniformity in QF_LIA) over the CubeCL IR
   (`crates/vericl-ir/src/prover.rs`, `prove_race_freedom`/`prove_cooperative`). The cooperative
   twin is a macro-derived **phase-split** reference (`crates/vericl-macros/src/coop.rs`), gated by
   a `cooperative(cube_dim = N)` clause. **M6 — the coupling**: a cooperative `tested` differential
   claim always makes its dependence on race freedom explicit — discharged (cites the
   `smt-race-freedom` proof), assumed (an injected `intra-phase-race-freedom` `assumed` claim when
   `prove: false` or the proof is out-of-subset), or refused — never silently green. One sound
   two-thread walk backs BOTH the `smt-oob-freedom` (bounds deferred by the single-thread walk) and
   `smt-race-freedom` claims via the split `prove_cooperative` returns (`CooperativeProof`). A
   declared-reference fallback (`reference = fn`) carries a distinct, strictly weaker
   `differential-declared-reference` check string (candidate #3, §4.4). `verify()`'s downgrade check
   already covers the new claim kind (keyed on the `check` string;
   `dropped_proved_race_freedom_claim_is_a_downgrade` pins it). **M7 — validation**: clean-room
   `block_sum_reduce` + `grid_stride_reduce` wired into `vericl::suite!`, each carrying the triple
   `tested` (race dependency discharged) + proved `smt-oob-freedom` + proved `smt-race-freedom`
   (both lanes: wgpu, and cpu feature); a cooperative defective twin `block_sum_reduce_racy` (the
   overlapping `tile[tid] += tile[tid+1]` stride) REFUTES `smt-race-freedom` with a two-thread
   counterexample (`t1 == t2 + 1`) in `conform demo-defects`, exit 0. Private dogfood: the
   production `Σ|iq|²` reduction shape annotated cooperative + instantiate + full
   contract lands the whole triple on the real shape (5 documented adaptations, 2 new walls —
   comptime loop bound, caller-supplied grid width — and the predicted fma tolerance finding; see
   `docs/dogfood-2026-07.md` shared-memory addendum, and `vericl-dogfood`). Resolves the README
   "open decision" on ordering: race-freedom is the gateway and is now delivered.
7. Later: prover follow-ups. [The unbounded-integer-overflow gap this line led with — a divisor
   provably nonzero in unbounded QF_LIA that wraps to exactly zero via `u32` overflow (`a * b ==
   2^32`), and the wider class of a wrapped value feeding any index/guard/loop-bound — is now
   DONE; see roadmap item 14 below. It was closed WITHOUT the full QF_BV rewrite, via a faithful
   finite-width model in QF_LIA (design decision + rationale in item 14 and the prover's
   "Bounded-integer overflow model" module doc), so the div/mod gap is no longer "known-inert on
   naga"; it taints, `OutOfSubset`.] Remaining: fold cubecl version into Identity; upstream
   conversation with tracel-ai; standalone `vericl check` CLI (README CI story row); a
   `FLOAT_METHOD_CONST_ONLY` distinction if a dogfooded kernel needs a runtime `new`/`from_int`.
   [The `f64` instantiation tier this line previously listed as hypothetical debt is DONE — see
   roadmap item 11; the production codebase validates at f64 on cubecl-cpu, which drove it.]
   [A *full* QF_BV model is no longer on the critical path after item 14, but remains a possible
   future precision upgrade — it would model `Mul` wraparound exactly (item 14 taints a possibly-
   wrapping `Mul` rather than modeling `(a*b) mod 2^W`, which is QF_NIA-hard in the current
   encoding) instead of conservatively.]
8. [DONE 2026-07-22] Kernel composition — `#[vericl::helper]` + a kernel-side `uses(...)` clause
   (roadmap item 3 per docs/dogfood-2026-07.md, the last Tier-1 macro gate, unblocking 16/22
   dogfooded kernels). Design: `#[vericl::helper(instantiate(...), uses(...))]` on a non-launch
   `#[cube]` device fn generates a host twin `fn <name>_vericl_ref(...)` plus a `<name>_vericl`
   module (`SOURCE_HASH`, `USES`, `identity_hash`/`identity_hash_at`); a kernel's own
   `uses(helperA, helperB)` clause rewrites its twin's calls `helperA(...)` -> its own twin's
   calls to `helperA_vericl_ref(...)` (turbofish preserved on rewrite, dropped on the callee side
   since the twin target is always monomorphized — see below); helpers may call other helpers via
   their own `uses(...)`, the identical mechanism, so helper-calling-helper needed no special
   casing.

   **Design override from the original brief, found and fixed mid-task (approved by the
   orchestrating session):** the brief called for a helper's generic type parameter to stay a
   plain Rust generic in the twin. Empirically falsified before implementing it: cubecl-core's
   `Float`/`Numeric` method traits (`impl_unary_func!`) give most methods only the panicking
   `unexpanded!()` default (e.g. `impl Sqrt for f32 {}`, no override) — `FLOAT_METHOD_WHITELIST`'s
   host-safety proof relies entirely on Rust preferring an inherent method over a trait method for
   a *concrete* receiver, a preference that does not exist for a bound-but-unsubstituted generic
   type parameter. Verified directly: a scratch `fn g<F: Float>(x: F) -> F { x.sqrt() }` panics on
   host calling `g(2.5f32)` (confirmed via `catch_unwind`), as does `.abs()`; only the small
   per-type-overridden associated-fn subset (`new`, `from_int`, `min_value`, `max_value`) is safe
   generically. Fix: a helper's generic type parameter(s) must be monomorphized via its own
   `instantiate(...)` clause exactly like a kernel's — required whenever the helper has generic
   type params, reusing `resolve_instantiate`/`transform_body`/`FloatMethodCheck` unchanged (now
   parameterized by `item_kind: &str` for kernel- vs. helper-flavored error text). `#[comptime]`
   parameters are unaffected by this finding (plain values, no trait dispatch) and stay ordinary
   pass-through parameters in a helper's twin signature, per the original design — the caller's
   own twin already has the pinned value in hand to pass along. Cost: one concrete type per
   helper (today, `f32` is the only type any part of vericl v0 supports, so this is free in
   practice).

   **Unlisted-callee detection** (`uses(...)`'s call-expression scan, `UsesRewriteFold`): a
   `#[proc_macro_attribute]` invocation cannot see whether some other bare ident in scope names a
   `#[cube]` fn, a `#[vericl::helper]`-annotated one, a host-safe free function, or nothing at all
   — no whole-crate visibility. Classifies every bare (single-segment, e.g. not `Type::method`)
   call in a twin body into three buckets: `uses(...)`-listed -> rewritten to `_vericl_ref`
   (turbofish stripped — the target is always monomorphized, confirmed necessary empirically
   while building the examples: a real generic call site often needs `foo::<F>(...)` for
   inference even though the twin's target has zero generics after substitution); a local binding
   (collected by a `syn::visit::Visit` walk over every `Pat::Ident` in the body plus the fn's own
   params, deliberately over-inclusive of nested scoping — a spurious local match only ever
   avoids flagging something real rustc still gets the final word on) or a tiny explicit allowlist
   (`KNOWN_HOST_SAFE_FREE_FNS`, currently just `range_stepped`, grown by demand) -> left alone;
   anything else -> a targeted compile error naming the function and suggesting `uses(...)` +
   `#[vericl::helper]`, replacing what would otherwise be a confusing type/resolution error deep
   in cubecl's generated code (the original, untouched item really is in scope under that name,
   since `#[vericl::helper]`/`#[vericl::kernel]` always re-emit it — so the fallback isn't
   "cannot find function", it's a genuinely confusing signature mismatch). Verified this
   classification is complete for the existing example suite (only `range_stepped` needed the
   allowlist) and exercises the rejection path correctly on a deliberately-unlisted call (scratch,
   not committed).

   **Identity and the drift hazard:** helpers get their own `SOURCE_HASH` (same recipe as a
   kernel's — source tokens + raw contract tokens + vericl version); a composing kernel's/helper's
   `identity()`/`identity_hash()` additionally folds `SOURCE_HASH` with every `uses(...)`-listed
   dependency's own (already-recursive) `identity_hash_at(depth)` via a new core function,
   `vericl::combine_source_hash` (SHA-256; the one place core `vericl` now depends on `sha2` —
   still zero `cubecl` dependency, the constraint that actually matters). Recursion composes
   without double-counting: a kernel/helper only ever combines with its *direct* dependencies'
   hashes, and each of those already recursively covers its own `uses(...)`, so a change N levels
   deep still reaches the top without re-hashing the same content redundantly at each level (a
   diamond dependency being hashed into two different parents' combines is correct, not a bug —
   it's exactly what should happen for two independent parents of the same changed child).
   Regression-tested (`crates/vericl-examples/src/lib.rs`'s `#[cfg(test)]` block):
   `composed_kernel_identity_folds_in_its_helpers_hash` /
   `helper_calling_helper_identity_is_recursive` /
   `composed_kernel_identity_is_recursive_through_the_helper_chain` reproduce the combine
   independently via `combine_source_hash` and assert byte-for-byte equality (not just
   "differs"); `unused_helper_does_not_affect_an_unrelated_kernels_identity` asserts a
   non-composing kernel's `identity()` is an exact pass-through of its own `SOURCE_HASH`
   regardless of how many helpers exist elsewhere in the crate — structural proof `identity()`
   only ever sees the `uses(...)`-declared set. Additionally verified by hand (not committed,
   since it needs a real source edit + rebuild a `#[test]` can't do in one process): edited
   `single_tap`'s body, reran `cargo test -p vericl-examples --lib`, and confirmed
   `gain_kernel_vericl::identity().source_hash` AND its `ir_hash` (via `vericl_ir::kernel_ir_hash`)
   both moved while `axpy`'s and `flatten_decode_scale`'s (unrelated, non-composing) stayed
   byte-identical, then reverted — the exact "helper body changes, kernel source doesn't, kernel
   identity must" hazard the design brief called out, closed and empirically confirmed shut.
   `ir_hash` already covers this too, independently: cube expansion inlines a used helper's real
   IR into the composing kernel's own `Scope`, so `ir_hash` moved in the same hand-edit check —
   `identity()`'s job is making the *source-level* hash honor composition the same way, not
   duplicating what IR-level identity already gave for free.

   **Recursion:** cycles are possible in a Rust fn call graph (mutual recursion compiles) —
   verified empirically that `#[cube]` itself does not reject it either (a self-recursive and a
   two-function mutually-recursive `#[cube] fn`, both compile cleanly; the former only draws
   rustc's ordinary `unconditional_recursion` *lint warning*, the latter not even that — no
   upstream backstop to lean on). `register_and_check_cycle` (vericl-macros) maintains a
   process-local registry of every `uses(...)` edge seen so far in the compilation and DFS-checks
   for a cycle reachable from each new declaration's dependencies back to itself, on every
   kernel's/helper's own macro invocation. This is provably complete for a cycle written in
   ordinary top-to-bottom source (the last node in the cycle to be macro-expanded always closes
   it, and by construction every other node has already registered by then) but not a
   soundness-critical guarantee in general, since one macro invocation cannot see another's output
   directly — documented as best-effort, not silently assumed complete. Verified against BOTH
   shapes by hand (scratch, not committed, since there's no compile-fail harness yet — same
   precedent as the existing `wrapping`-subset rejection): a helper listing itself in its own
   `uses(...)` and a two-helper mutual cycle (`uses(...)` declared on both sides) were both
   rejected at compile time, at the second-processed item, naming the exact cycle path (e.g.
   `cyc_b -> cyc_a -> cyc_b`). Backstop for the acknowledged residual gap (added after review):
   the runtime hash-combine is depth-guarded (`vericl::check_helper_composition_depth`, 32
   levels, panics naming the offending item) so a cycle that somehow slips the compile-time check
   fails loudly instead of hanging — direct unit test
   (`crates/vericl/src/contract.rs::tests::helper_composition_depth_guard_trips_at_the_threshold`)
   pins the guard's own threshold behavior, since no compiling cycle could be constructed to
   exercise it end-to-end (every cycle tried was caught first, as expected).

   **Instantiation mismatch across a `uses(...)` edge** (e.g. a kernel pinned `F = f32` calling a
   helper pinned `F = f64`): not caught by vericl-macros (no cross-invocation visibility, same
   limitation as cycle detection), but checked empirically what happens instead — ordinary Rust
   type-checking in the generated twin produces an `E0308` at the exact call-site argument plus a
   "function defined here" note, both landing on real, comprehensible source spans (the call
   expression's own span, and the callee name's span, deliberately preserved from the original
   `fn` item through every token substitution) rather than pointing into opaque macro-internal
   code. It does not spell out "these two instantiate(...) clauses disagree" on its own; mitigated
   by the generated twin's doc comment always stating its pinned concrete type. Documented as a
   residual in `vericl-macros::helper`'s doc comment rather than left silently unaddressed.

   **Prover:** needed zero changes, confirmed rather than assumed — cube expansion inlines a
   `uses(...)`-listed helper's IR directly into the composing kernel's own `Scope` (the same
   mechanism cubecl itself already uses for ordinary, non-vericl kernel composition), so the
   existing walker over `kernel_definition()` already sees everything a helper's body does.
   Positive/negative pair (`crates/vericl-examples/src/lib.rs`): `tap_pair` is a helper whose OWN
   body reads `x[idx]` and `x[idx + 1]`; `tap_pair_guarded_kernel` establishes `ABSOLUTE_POS + 1 <
   x.len()` before calling it and `Proved`s; `tap_pair_unguarded_kernel` (same helper, same
   shape) only establishes `ABSOLUTE_POS < x.len()` — one short of what the helper's own
   unguarded second read needs — and `Refuted`s, proving the obligation living inside the composed
   helper's body is genuinely walked, not silently dropped because it's composed rather than
   written directly in the kernel.

   **Public examples** (`crates/vericl-examples/src/lib.rs`, wired into `tests/conformance.rs`'s
   `suite!`): `single_tap` (pure scalar, reused directly by two kernels — `gain_kernel` and,
   transitively via `fir_pair_scaled`, `fir_pair_kernel`) and `fir_pair` (tuple-returning 2-tap
   pair, the milestone's suggested shape) are composed by `fir_pair_scaled`
   (`uses(fir_pair, single_tap)`, one level of helper-calling-helper) into `fir_pair_kernel` (two
   `&mut Array` outputs — the existing N-output machinery needed no changes either). `gain_kernel`
   and `fir_pair_kernel` both carry `tested` (differential, wgpu, 5 sizes) + `proved`
   (`smt-oob-freedom`) claims in `evidence/vericl.json` — the milestone's "composed kernel carries
   tested + proved claims" ask. `tap_pair`/`tap_pair_guarded_kernel`/`tap_pair_unguarded_kernel`
   (prover-only positive/negative pair above, not suite-wired — mirrors the existing
   `stepped_loop_*` precedent for a kernel that exists purely to pin a prover finding).

   **Private dogfood validation** (per the README's private-codebase policy: never committed, described
   here only by construct class): the survey's own "inner-loop-with-single-helper shape"
   candidate — a two-chain FIR-convolution device fn (which the private source itself already calls
   "the first proof that #[cube] Level-1 composition works") and the single-emitter inner-loop
   kernel (the `#[cube(launch)]` entry point that calls it exactly once) from the private kernel crate —
   generic (`F: Float`), one `#[comptime]` param, exactly one helper call, no shared memory, per
   the survey's own gap ranking the least-blocked composed shape. Copied UNCHANGED (no adaptation
   needed for either body) and passed differentially on wgpu end-to-end across 5 sizes. Its bounds
   proof — the real headline finding — is `Proved` (54 obligations), not merely `OutOfSubset`: the
   helper's `#[unroll] for j in 0..8` loop carries four float accumulators, but per the existing
   loop-carry refinement (docs/dogfood-2026-07.md Tier-2 gap #1) only they get tainted, since
   nothing they touch is ever used as an index — every access inside the composed helper's body
   stayed provable. Predicted "usize runtime param" wall did NOT surface (this kernel's only
   non-array param is a `#[comptime] usize`, already covered by `instantiate(...)`); no NEW wall
   surfaced either — composition landed on the first real composed kernel tried, with zero
   adaptations to either function body. One genuine, previously-undocumented finding: the
   kernel's own body destructures the helper's tuple return with ordinary `let (a, b) =
   helper(...)` and compiles fine as-is, whereas the identical pattern failed
   (`Unsupported local pat: Pat::Tuple`) when tried between two plain `#[cube]` device fns while
   building the public `fir_pair_scaled` example (worked around there with `.0`/`.1` field access
   instead) — so tuple-`let` destructuring of a composed call's return is specifically a
   device-fn-calling-device-fn cubecl limitation, not a general composition one; noted for a
   future README update if a public helper-calling-helper example ever wants the more natural
   form. Separately, and NOT a composition finding: running the full private dogfood suite
   surfaced that `instantiate_subset.rs`'s cos/sin-synthesis-kernel div/mod bounds-proof-out-of-subset
   test (written for roadmap item 5's div/mod prover milestone, before div/mod modeling existed)
   now returns `Refuted` instead of its hardcoded `OutOfSubset`-or-`Proved` expectation, because
   that milestone's div/mod modeling has since landed and this specific test passes zero
   `assumes` (so the solver can freely pick `fsteps.len() == 0` as a real counterexample) —
   confirmed unrelated to this task (reproduces identically with none of this task's new files
   present) and out of scope to fix here (a different milestone's private test going stale, not a
   composition bug); left as-is and flagged here rather than silently worked around.

   Verification: `cargo test --workspace` and `-p vericl-examples --features cpu` both green (77
   tests total: 15 vericl core + 37 vericl-examples lib + 1 conformance + 2 float-whitelist + 21
   vericl-ir + 1 vericl-macros, unchanged pass counts across both feature sets); `cargo clippy
   --workspace --all-targets` zero warnings on both feature sets; `VERICL_UPDATE=1` run for
   `--features cpu` first, verified, then the default (non-cpu) `VERICL_UPDATE=1` run LAST,
   verified — evidence gained `gain_kernel`/`fir_pair_kernel` (`tested`+`proved` each), all five
   pre-existing kernels' evidence unchanged; `conform demo-defects` exits 0 unchanged (composition
   touches neither defective kernel); the helper-drift identity regression exercised both
   structurally (four dedicated tests, above) and by hand (real source edit + rebuild, reverted);
   the prover composed-kernel positive (`tap_pair_guarded_kernel` -> Proved) and negative
   (`tap_pair_unguarded_kernel` -> Refuted) tests pass; the private dogfood suite's new
   `composition_subset.rs` (2 tests: differential pass on wgpu across 5 sizes, bounds `Proved` 54
   obligations) passes; `dogfood-rejects` still fails to build with the same class of expected
   rejections (generics/topology/usize gates), unaffected.
9. [DONE 2026-07-22] Adversarial soundness review round 2 (SMT prover + instantiate()/uses(...)
   hardening) — DONE. One CRITICAL confirmed bug, one MEDIUM, one LOW, plus a docs-only finding
   pair; all four fixed and regression-tested. Same posture as the round-1 review above: every
   fix closes a real, demonstrated hole rather than a hypothetical one — the reviewer's scratch
   repro crate (path-deps on this repo, never committed here per the private-scratch precedent)
   reproduced each bug against the pre-fix build and was re-run against the fixed build to confirm
   the new verdict.

   **CRITICAL — branch-scoped value-map rollback in `process_branch`
   (`crates/vericl-ir/src/prover.rs`).** Root cause: `self.smt.push()/pop()` scopes *path
   conditions* around an `If`/`IfElse` arm, but `self.memo` (the `VariableKind` -> symbolic-value
   map) was mutated in place with no save/restore at all — a variable reassigned inside one arm
   was treated as certain in the other arm, and after the branch closed, unconditionally. Three
   confirmed false-`Proved` manifestations, each independently demonstrated: (1) a variable
   clamped to a safe value inside an `If` with no `else` leaked past the branch close (a
   near-impossible guard clamping `idx` to `0` made an unrelated, unguarded, genuinely-unbounded
   later use of `idx` look safe); (2) the `else` arm of an `IfElse` saw the `if` arm's writes (both
   arms walked sequentially against the same unscoped map, no reset between them); (3) a
   post-`IfElse` read resolved to whichever arm was walked *last* (the `else` arm's write always
   won, regardless of which arm the real, per-thread execution actually took). Confirmed with a
   real OOB write on Metal (`crates/vericl-ir/src/prover.rs`'s pre-fix behavior on the reviewer's
   `if_else_merge_bug_kernel`: `Proved { obligations: 1 }` with `y.len() == 1` assumed, while a
   real 4-thread wgpu/Metal dispatch of the same kernel — declared length 1, backing allocation 4
   — wrote past the declared length at indices 1–3).

   Fix: `process_branch`'s `If`/`IfElse` cases now snapshot `self.memo` (a full `HashMap` clone —
   `SExpr` is `Copy`, so this is cheap) before walking an arm, restore that snapshot before walking
   the *other* arm (fixing manifestation 2), and restore it once more after the construct, then
   explicitly taint (`None`) every variable written *anywhere* in either arm rather than trusting
   either arm's leftover value (fixing 1 and 3) — no if/else value merging in v0; a variable set to
   the identical value in both arms still taints, deliberately conservative. "Written anywhere in
   either arm" is tracked by a new `Prover::write_log_stack: Vec<HashSet<VariableKind>>` (one frame
   per currently-open arm) and a new `Prover::set_var` helper that both writes `self.memo` and
   records the write into the top-of-stack frame — the single point every genuine variable write
   (`bind_out`, `taint_out`, and the loop-carry pre/post taint in `process_range_loop`, all
   rerouted through it) goes through, as opposed to `value_of`'s read-only resolution caching,
   which must NOT be logged (logging it would spuriously re-taint e.g. `ABSOLUTE_POS` the first
   time a branch happens to be where it's lazily resolved, breaking unrelated obligations after the
   branch). Composes correctly for nested branches with no special-casing: an inner branch's own
   merge step re-applies its taints through `set_var` too, which — since the inner frame is already
   popped by then — logs into whatever frame is now on top (the *enclosing* arm's), so a write two
   levels deep still reaches the outermost merge. (First implementation attempt used a raw
   `memo.insert` for the merge's own taint-application loop instead of `set_var`, which passed
   every single-level test but failed the nested-branch regression test below immediately — a
   genuine catch, not a hypothetical one; fixed by routing that loop through `set_var` too.) Full
   soundness argument in the module doc's new "Branch-scoped write taint (If/IfElse)" bullet.

   Regression tests (`crates/vericl-ir/src/prover.rs::tests`, new "Branch-scoped write taint"
   section): `branch_write_does_not_leak_past_if` (manifestation 1 — now `OutOfSubset`, reason
   names the write index and `y`), `if_arm_write_does_not_leak_into_else_arm` (manifestation 2, the
   task's exact `if pos >= HUGE { idx = 0 } else { y[idx] = v }` shape with `y.len() == 1` assumed
   — now `Refuted` on genuine grounds, not a leaked value that happened to still be unsafe),
   `post_ifelse_merge_taints_branch_written_vars` (manifestation 3 — now `OutOfSubset`), and two
   nested-composition tests: `nested_branches_restore_correctly` (a write two levels deep, inside
   an `IfElse` nested in an outer `If`'s only arm, must still reach the outer merge's taint set —
   this is the test that caught the `set_var`-routing bug above) and
   `nested_branch_write_does_not_leak_into_outer_sibling` (the other half — the same two-level-deep
   write, now inside the outer `IfElse`'s `if` arm, must not leak into the outer's own `else` arm,
   a true sibling — `Refuted` on the sibling's genuinely-unbounded `idx`). Positive controls: the
   full pre-existing 21-test
   `vericl-ir` suite (axpy/fir3/flatten_decode_scale/composed kernels and every If/IfElse-using
   kernel that does NOT write a branch-arm variable into an index) passes unchanged, and all seven
   suite kernels' obligation counts are byte-identical (confirmed by `git diff` on
   `crates/vericl-examples/evidence/vericl.json` being empty — not merely "same numbers", the
   evidence file itself never needed regenerating): axpy=3, xorshift_step=2, mix_u32=2, fir3=4,
   flatten_decode_scale=2, gain_kernel=2, fir_pair_kernel=4.

   **MEDIUM — `instantiate(...)` substitution namespace collision (`crates/vericl-macros/src/
   lib.rs`).** `subst_type_tokens`/`transform_body`'s `instantiate(F = f32)` substitution is purely
   lexical (a `TokenTree::Ident` string match), with no notion of Rust's separate type/value
   namespaces — a local binding legally named `F` (type parameter and local live in different
   namespaces in the *original* kernel) or named like the concrete type (`let f32 = ...`) gets
   silently rewritten right along with the type parameter, producing a twin that computes something
   different from the real kernel with no compile-time signal. Demonstrated: the reviewer's
   `f_name_collision_kernel` (`let f32 = x[ABSOLUTE_POS]; let F = F::new(999.0);` — two genuinely
   distinct locals in the real kernel) had its twin silently write the second, shadowing local's
   value (`999.0`) instead of the first (`x[ABSOLUTE_POS]`) on every input. Fix: new
   `check_instantiate_local_collisions` (called from both `expand` and `expand_helper`, right after
   `params` is classified, on the ORIGINAL pre-substitution body) reuses `collect_locals` to scan
   for any local/parameter whose name equals either an `instantiate(...)` generic type parameter's
   own ident or its pinned concrete type's bare ident (when the concrete type reduces to a single
   identifier — the only shape a local's name could ever collide with), and rejects with a targeted
   error naming the collision and its role (`"local binding \`F\` collides with kernel
   \`name\`'s type parameter under instantiate(...) — rename the local; outside the vericl v0
   subset"`). Deliberately conservative (flags a local merely *named* either sensitive string, not
   only the narrower "a second, independent binding already uses the resulting name" condition
   that's the only shape that's actually unsound) — same "reject rather than silently approximate"
   posture as everything else in this project. Tests: `crates/vericl-macros/src/lib.rs::tests`
   `instantiate_local_collision_is_rejected` / `instantiate_no_collision_is_accepted` /
   `instantiate_empty_subst_is_always_accepted` (unit-level, direct calls into the checker); the
   reviewer's exact `f_name_collision_kernel` re-run against the fixed build now fails to compile
   with this targeted error (confirmed in the scratch crate, not committed — compile-fail
   demonstrated per the existing `wrapping`-subset-rejection precedent, no trybuild harness yet).

   **LOW — multi-segment call bypass in `UsesRewriteFold` (`crates/vericl-macros/src/lib.rs`).**
   The fold only ever inspected `p.path.segments.len() == 1` — a multi-segment call to a declared
   helper (e.g. `self::triple::<F>(x)`, reached via a `self::`-qualified path) skipped BOTH the
   rewrite-to-`_vericl_ref` AND the unlisted-callee rejection entirely, silently falling through to
   call the ORIGINAL, un-rewritten `#[cube]` item host-side — invisible to a black-box differential
   check whenever the original happens to be host-safe (as it was in the reviewer's repro,
   confirmed: `self_path_call_kernel_vericl::reference` produced the numerically-correct answer
   either way, since `triple`'s body is host-safe arithmetic — the bypass is a real hole but not
   one a differential-only check could ever have caught). Fix: `fold_expr_call` now also handles
   `segments.len() > 1`, rewriting when the LAST segment matches a `uses(...)`-declared name —
   turbofish stripped (same reasoning as the single-segment case) and **the whole path prefix
   dropped**, not just the last segment renamed in place. Dropping the prefix is necessary, not
   merely simpler: the twin body lives one module level deeper than the original call site (nested
   in the generated `<name>_vericl` module, which does `use super::*;`), so a prefix meaningful at
   the ORIGINAL call site — `self::`, above all — does not still mean the same thing one level
   down; the rewritten bare target is reachable via that same glob import regardless, exactly the
   mechanism the single-segment case already relies on. A multi-segment call whose last segment
   does NOT match a declared helper (`f32::max(...)`, an unrelated module path, ...) is left
   completely untouched — a **documented residual** (in `UsesRewriteFold`'s doc comment), not a
   soundness gap this fold's rejection guarantee covers: a multi-segment call to an unlisted,
   genuinely-cross-module helper is a case this fold cannot distinguish from a legitimate external
   call. Tests: `crates/vericl-macros/src/lib.rs::tests`
   `uses_rewrite_fold_rewrites_self_qualified_helper_call` (asserts the resulting AST directly —
   bare `triple_vericl_ref(x)`, turbofish stripped; chosen over a black-box differential/GPU probe
   specifically because those can't distinguish "correctly rewritten" from "bypassed but
   coincidentally correct", as the repro above demonstrates) and
   `uses_rewrite_fold_leaves_non_matching_multi_segment_call_untouched` (`f32::max(a, b)`
   byte-for-byte unchanged). Also added a real macro-pipeline regression,
   `crates/vericl-examples/src/lib.rs`'s `self_path_gain_kernel` (identical to `gain_kernel` except
   `self::single_tap::<F>(...)` in place of the bare call; not suite-wired, no new evidence entry
   needed — exists purely to pin the fix, same precedent as `tap_pair_guarded_kernel`):
   `self_path_gain_kernel_twin_matches_hand_computed` (same expected output as
   `gain_kernel_twin_matches_hand_computed`) and `self_path_gain_kernel_definition_is_provably_in_
   bounds` (`Proved { obligations: 2 }`, matching `gain_kernel`'s own count).

   **Docs-only findings** (README "CubeCL semantics findings", new subsection under "Proved
   claims"): (a) CubeCL 0.10 lowers `&&`/`||` to **eager**, unconditionally-evaluated instructions
   inside a kernel body, not short-circuiting branches — a guard shaped `idx_ok && x[idx] > 0.0`
   does not protect the `x[idx]` read the way it would in host Rust (the prover already refutes an
   insufficiently-guarded access composed this way; WGSL's own robustness, which silently clamps
   rather than traps, can mask the effect at runtime on wgpu specifically); (b) naga's
   division-by-zero fallback is dividend-preserving (`a / 0 == a`, `a % 0 == 0`, confirmed
   empirically), not trapping — noted in roadmap item 7 above as the concrete motivation for the
   still-open QF_BV wrapping-model item, since it's what makes the
   unbounded-overflow-feeding-div/mod gap (a `u32` multiplication provably nonzero in QF_LIA but
   wrapping to exactly `0`) currently harmless-in-practice on wgpu/Metal specifically, rather than
   a live crash risk — not something to be relied on in general. `uses(...)` declaration-order
   hash sensitivity (same dependency *set*, different `SOURCE_HASH`/`identity()` on reorder —
   confirmed via the reviewer's `diamond_kernel`/`diamond_kernel_reordered` scratch pair) is now
   documented in three places: `crates/vericl/src/contract.rs`'s `combine_source_hash` doc (the
   central, authoritative explanation), the macro-generated `identity()` doc comment (what a user
   actually sees hovering over their own generated code), and README's "Identity and composition"
   paragraph — all noting it's the safe direction (spurious staleness only, never silently drops a
   real change).

   **Surfaces the review probed that survived without changes needed** (re-run against the fixed
   build, scratch crate not committed): eager `&&`-RHS array access
   (`and_rhs_has_array_access`, refutes correctly — the read is genuinely unconditional in IR, and
   the prover already models it that way, per the docs finding above); div-chain and mod-chain
   composition (`a/b/c`, `(a%b)%c)` — both `Proved { obligations: 2 }`, unaffected); loop-carry
   shadowing (`shadowed_carry` — an inner loop-local named the same as an outer carried
   accumulator, `Proved`, since cubecl allocates distinct `VariableKind` ids per binding regardless
   of surface-name reuse) and a carried variable feeding its own loop's bound
   (`carried_own_bound` — correctly `OutOfSubset` on the range-loop's `end` resolution, since
   `rl.end`'s `Variable` is read once at loop-entry time in the IR itself, before any in-body
   taint applies); `wrapping` kernel composing a non-`wrapping` helper (`add_one_u32` overflow
   inside a composed helper's twin — panics loudly on overflow, a LOUD differential failure, not a
   silent wrong pass, since `#[vericl::helper]` rejects a `wrapping` clause outright and Rust's
   default checked arithmetic panics on debug-profile overflow); read-before-write inside a
   loop-carried accumulator (`read_before_write_carry` — correctly `OutOfSubset`, the pre-loop
   taint applies before the body walk starts, so a read of the carried variable before its own
   first write in program order never sees the stale pre-loop value).

   Verification: the three branch-scoping regression tests + the two nested-composition tests all
   pass; `cargo test --workspace` and `-p vericl-examples --features cpu` both green (89 tests
   default: 15 vericl core + 39 vericl-examples lib + 1 conformance + 2 float-whitelist + 26
   vericl-ir + 6 vericl-macros — vericl-ir gained 5, vericl-macros gained 5, vericl-examples lib
   gained 2 over their round-1-review-era counts; `--features cpu` variant identical pass count);
   `cargo clippy
   --workspace --all-targets` zero warnings on both feature sets; `evidence/vericl.json` byte-
   identical (`git diff` empty) — no `VERICL_UPDATE` run was needed, since none of the four fixes
   changed any existing suite-wired kernel's source tokens (Fix 1 touches only
   `crates/vericl-ir/src/prover.rs`; Fixes 2/3 only change macro-expansion-time *rejection*/
   *rewrite* behavior, not any example kernel's own source; the one new kernel,
   `self_path_gain_kernel`, is deliberately NOT suite-wired, so it never touches evidence);
   `conform demo-defects` exits 0, output unchanged (neither defective kernel touches branches-
   with-arm-writes, instantiate(), or uses()). The reviewer's scratch repro crate (path-deps on
   this repo) was re-run in full against the fixed build: `if_merge_bug`/`if_else_merge_bug`
   (manifestations 1/2) now `OutOfSubset`/`Refuted` respectively (were both falsely `Proved`);
   `post_ifelse_merge`/`post_ifelse_false_proved` (manifestation 3 shapes) now `OutOfSubset` (one
   of the two was already incidentally `Refuted` pre-fix, for the wrong reason — a leaked value
   that happened to still be unsafe, not a correctly-scoped one); `f_name_collision_kernel` now
   fails to compile with the targeted collision error (was a silent wrong twin); the ground-truth
   GPU probe (`probe3_ground_truth`, real 4-thread wgpu/Metal dispatch) independently confirms the
   underlying OOB write the manifestation-2/3 kernels' pre-fix `Proved` verdict was wrong about.

10. [DONE 2026-07-23] Adversarial soundness review round 3 (cooperative/shared-memory periphery) —
    DONE. **Verdict: no false-`Proved` this round** — the reviewer's 17-kernel battery produced no
    wrong verdict on any *core* surface; both findings are on the periphery (a macro-level
    definedness ban and identity bookkeeping), neither a prover soundness hole. Both fixed and
    regression-tested here, same posture as rounds 1–2 (every fix closes a demonstrated hole; the
    reviewer's scratch shapes reproduced against the pre-fix build and re-run against the fixed one,
    scratch not committed — path in the verification report).

    **Surfaces the review probed that held without changes needed** (17-kernel battery, cooperative
    and non-cooperative): barrier-uniformity taint — a `sync_cube()` under a thread-varying `if
    tid < half` and a barrier inside a thread-varying-trip-count loop both stay `OutOfSubset` with
    the barrier-divergence reason (round-2 risk 2 analog, held); the two-thread shared-memory race
    walker — `block_sum_reduce`/`grid_stride_reduce` still `Proved` race-free with unchanged
    obligation counts, and `block_sum_reduce_racy`'s overlapping `tile[tid]+=tile[tid+1]` stride
    still `Refuted` with a two-thread counterexample; the tainted shared/global **index** discipline
    — a gather-tainted or otherwise unmodelled index at a shared/global access is `OutOfSubset` at
    the access site, never `Proved` and never silently modelled; shared-tile **definedness** — the
    poison twin still fires on a never-written cell rather than masking it as 0.0 (the property F1
    hardens); `CUBE_DIM` pinning — the `cooperative(cube_dim = N)`↔launch assertion still refuses a
    mismatched block size; the round-1/2 fixes re-probed under cooperative bodies (branch-scoped
    write taint, `instantiate(...)` namespace-collision rejection, the `uses(...)` rewrite/rejection)
    all held; and the §6 honesty coupling — a cooperative differential with neither a race-freedom
    proof nor the injected `assumed` clause is still *refused*, not recorded.

    **Design-intended residual the reviewer confirmed (not a bug):** a genuinely racy cooperative
    kernel whose racy access rides a **tainted** index (one the two-thread walker cannot prove
    disjoint) does not get a false `Proved` race-freedom claim — it falls to the labeled fallback
    tier (`OutOfSubset` on the proof + the differential carrying the explicit `assumed`
    "intra-phase race freedom" clause, or refusal), exactly the §5.3/§6 posture. The gap is
    *documented and labeled*, never a silent green — the reviewer confirmed the tier boundary holds.

    **F1 (MEDIUM) — paren-evasion of the shared-tile compound-assignment poison ban
    (`crates/vericl-macros/src/coop.rs`).** `SharedCompoundAssignCheck` classified a compound-assign
    target only as `Expr::Index { expr: Expr::Path }`, so `(tile)[tid] += 1.0` (a parenthesised
    index base) slipped past the ban that keeps a read-modify-write out of the shared-memory subset
    (§4.5). The twin's poison `SharedTile` then read the never-written cell as its `Default` (`0.0`)
    — green on Metal (which zero-inits), but the kernel is UB on a non-zeroing backend, the exact
    definedness-masking §9 risk 3 warns about, hidden behind a passing differential. Fix: peel
    `Expr::Paren`/`Expr::Group` (new `unwrap_paren_group`, mirroring `expr_is_pure_alias`'s existing
    Paren/Group recursion, coop.rs) at *both* LHS levels — the whole target (`(tile[tid]) += …`) and
    the index base (`(tile)[tid] += …`). The reviewer's exact paren kernel is now rejected with the
    unchanged poison-ban wording. Tests (`coop::tests`): `paren_compound_assign_into_shared_tile_is_
    rejected` (the exact evasion), `nested_paren_..._is_rejected` (`((tile))[i]` and `(tile[i])`
    both peeled), `bare_..._is_rejected` (positive control, pre-existing behaviour). All four fail
    against the pre-fix matcher and pass after (confirmed by temporary in-place revert).

    **F1 audit — the same paren-evasion pattern everywhere the macros classify an expression by
    shape** (the reviewer asked for a sweep, not a point fix): (1) **`UsesRewriteFold` callee
    detection** (`crates/vericl-macros/src/lib.rs`) — **was vulnerable**: `(helper)(x)` matched
    neither the `Expr::Path` rewrite-to-`_vericl_ref` nor the unlisted-callee rejection, so a
    parenthesised helper call silently called the original `#[cube]` item host-side (invisible to a
    black-box differential, exactly the round-2 multi-segment bypass class). Fixed by peeling the
    callee (`peel_paren_group`) before classifying; tests `uses_rewrite_fold_rewrites_parenthesised_
    helper_call` (`((self::triple::<F>))(x)` → bare `triple_vericl_ref(x)`) and `..._rejects_
    parenthesised_unlisted_call`, both failing pre-fix. (2) **`check_instantiate_local_collisions` /
    `single_ident_string`** — **not vulnerable**: the type-value side is gated to `Expr::Path` only
    by `resolve_instantiate` (a parenthesised `instantiate(F = (f32))` is rejected upstream before
    reaching the classifier), and the local-name side uses `visit_pat_ident`, which already recurses
    through `Pat::Paren`/`Pat::Tuple`. (3) **`WrappingFold` LHS handling** — **not vulnerable**: it
    keys on the `Expr::Binary` op and rewrites the whole `#left` wholesale (no index-vs-path shape
    classification), and folds children post-order, so `(x) += y` / `(x += y)` are handled without a
    shape gate to evade. (4) **the pure-alias check `expr_is_pure_alias`** — **already recursive**
    (Paren/Group arms, coop.rs) — it is the reference the fix mirrors.

    **F2 (LOW-MEDIUM) — declared-reference identity did not cover the reference body
    (`crates/vericl-macros/src/lib.rs`).** A kernel's `SOURCE_HASH` covers its own tokens + the
    contract attribute tokens; the `reference = <path>` clause put only the reference's *path text*
    in those tokens, never its body — so drifting the referenced fn's body left the kernel's recorded
    identity byte-identical (demonstrated on `block_sum_reduce_declared`), contradicting §4.4's
    promise that the reference fn's own source hash is recorded. Fix: a new `#[vericl::reference]`
    attribute for the plain host fn used as a declared reference — it derives no twin and no
    `#[cube]` machinery (it *is* the reference); it generates a sibling `<name>_vericl` module
    holding the fn's own `SOURCE_HASH` (over its tokens), plus a sibling accessor
    `<name>_vericl_reference_source_hash()`. The kernel's `reference = fn` clause now folds that hash
    into `identity()` via the same `vericl::combine_source_hash` runtime path `uses(...)` uses, so a
    reference-**body** drift moves the kernel's recorded identity. The clause **requires** the
    annotation: it calls the sibling accessor, so an un-annotated reference fails to compile at the
    `reference = …` clause span with `cannot find function \`<name>_vericl_reference_source_hash\``
    (naming the attribute requirement, no misleading `cargo add` help — the reason the accessor is a
    sibling, not nested in the module). `block_sum_declared_ref` now carries `#[vericl::reference]`.
    Regression test `crates/vericl-examples/src/lib.rs::declared_reference_body_is_part_of_kernel_
    identity` (structural: `identity()` folds in exactly the reference's hash, and the derived-twin
    sibling `block_sum_reduce` stays a pass-through — the fold is scoped to declared references); the
    inverted probe (edit the reference body → the real kernel's identity moves `dec33577…` →
    `07e8bd42…`) and the annotation-missing error were demonstrated in scratch (see verification
    report). Macro-level tests (`vericl-macros`): `reference_macro_generates_source_hash_module_and_
    accessor`, `reference_macro_hash_tracks_the_body`, `reference_macro_rejects_cube_fn`,
    `reference_macro_rejects_arguments`. Design §4.4 status note + README declared-reference
    paragraph updated.

    Verification: full workspace green (`cargo test --workspace` — vericl 20, vericl-examples lib 47
    + integration 7 [conformance 1, cooperative 3, cooperative_fallback 1, float_method_whitelist 2],
    vericl-ir 46, vericl-macros 15; vericl-macros +6 and vericl-examples lib +1 over round-2 counts);
    `-p vericl-examples --features cpu` identical pass count; `cargo clippy --workspace --all-targets`
    and `-p vericl-examples --features cpu --all-targets` both zero warnings; `evidence/vericl.json`
    and `evidence/cooperative_fallback.json` byte-identical (`git diff` empty — F1 changes only
    macro-time *rejection*, F2 only the identity fold of a NON-suite-wired kernel; `block_sum_reduce`,
    the one suite-wired cooperative kernel, declares no `reference` so its identity is unchanged, so
    no `VERICL_UPDATE`/regeneration was needed); `conform demo-defects` exits 0, output unchanged
    (all bounds/race/differential defects still caught; neither defective kernel uses a paren
    compound-assign or a declared reference).

11. [DONE 2026-07-23] **f64 support** — the `f64` instantiation tier (roadmap item 7's former
    "hypothetical debt"), driven by the real demand that the private production codebase validates
    at f64 on cubecl-cpu (docs/dogfood-2026-07.md). f64 is an instantiate *tier*, not a new subset —
    `wrapping` stays integer-only, and the bounds/race provers, composition, and cooperative paths
    are untouched.

    **Critical platform finding (verified empirically FIRST, since it shapes everything).** WGSL has
    no f64. cubecl 0.10 launching an f64 kernel on `WgpuRuntime` is **not** a compile error and
    **not** a runtime panic — it silently returns **wrong** results (not an f32 demotion: genuine
    garbage — a probe's worst element was 526.99 where the correct f64 value is 1776.99; the host
    uploads 8-byte f64 into a buffer WGSL indexes at a different element size). This silent-corruption
    landmine is exactly the failure class the project exists to catch, so it is pinned by a test
    (`tests/f64_wgpu_unsound.rs`, asserting the f64 kernel *diverges* from its correct twin on wgpu)
    and wgpu is never an f64 lane. cubecl-cpu runs f64 correctly at full precision (probe:
    `max_abs_diff = 0`, distinct from the f32-demoted value at 17 sig figs). **Design consequence:**
    an f64 kernel has NO front-end-independent execution lane on this machine (wgpu broken, cubecl-cpu
    shares CubeCL's front end), so the macro-derived twin is the *sole* independent leg — its
    independence is load-bearing. Recorded honestly in the f64 evidence trusted list
    (`host CPU execution hardware` + the explicit shared-front-end caveat) via a new
    `frontend_independent: false` suite declaration; README "f64 support" states it loudly.

    **Core compare** (`crates/vericl/src/compare.rs`): `ulp_distance_f64` (i128 ordered-map,
    saturating to u64), `compare_f64` (max_ulp), `compare_f64_absrel`, `compare_f64_with` (dispatch)
    — mirror the f32 impls including NaN-always-fails and the `inf - inf` edge; f32 unit tests ported
    to f64 (`ulp_basics_f64`, `compare_reports_f64`, `absrel_f64_nan_and_inf_edges`).
    **`Compare`** (`contract.rs`) gained `MaxUlpF64(u32)` / `AbsRelF64 { abs: f64, rel: f64 }` (f64
    tolerances stored at f64 precision, described `f64 …`) — additive, so all existing f32/integer
    evidence is byte-identical. **rng** (`rng.rs`): `next_f64_range`/`fill_f64` via the 53-bit path
    (`>> 11`, `/ 2^53`), the f64 analog of the 24-bit f32 path; `f64_uses_full_precision` proves the
    draws exceed f32.

    **Macro** (`vericl-macros`): `NumKind::F64` + `NumKind::is_float()`; f64 draw codegen
    (`next_f64_range`/`fill_f64`); the mandatory-range-for-floats rule and all gen/compare error
    messages extended to f64; compare dispatch (lib.rs + coop.rs) adds `NumKind::F64 ->
    compare_f64_with`. The `Compare` VALUE is precision-aware: `parse_contract` keeps building the
    f32/default tokens verbatim (so non-f64 kernels are byte-identical) and additionally records a
    precision-agnostic `CompareMode`; `expand` computes the compared `&mut Array` outputs' float kind
    and, only for an f64 kernel, rebuilds the tokens via `compare_tokens_f64` (a kernel mixing f32 and
    f64 outputs is rejected — one compare mode can't serve two precisions). `instantiate(F = f64)`
    needed no resolver change (`f64` parses as a concrete type token-wise).

    **Whitelist re-verified on f64, not assumed** (`tests/float_method_whitelist_f64.rs`): every
    `FLOAT_METHOD_WHITELIST` entry is host-callable + numerically correct on `f64`, and every
    `FLOAT_METHOD_REJECT` entry panics on `f64` — identical to the f32 result, so a single shared
    whitelist stays correct (NO per-type split needed). Same reason: inherent-method preference for a
    concrete `f64` receiver, and real per-type `f64` impls for the associated fns.

    **Example + suite**: `axpy_f64` (byte-for-byte `axpy` with `instantiate(F = f64)`,
    `compare(abs = 1e-12)` justified from the ranges) in `crates/vericl-examples/src/lib.rs`, plus lib
    tests (`axpy_f64_twin_is_full_precision`, `axpy_f64_compare_is_recorded_as_f64`,
    `axpy_f64_kernel_definition_is_provably_in_bounds` — 3 obligations, same as f32). Suite wiring: a
    SECOND `suite!` invocation (`tests/conformance_f64.rs`, runtime `cubecl::cpu::CpuRuntime`,
    `evidence/vericl_f64.json`, `frontend_independent: false`, `#[cfg(feature = "cpu")]`) — the
    conformance.rs + cooperative_fallback.rs "one suite, one manifest" precedent, honoring M6's
    two-`#[test]`s-must-not-share-one-evidence-file constraint. `axpy_f64` carries `tested`
    (differential, cpu) + `proved` `smt-oob-freedom` (3 obligations). New `suite!` field
    `frontend_independent` (default `true` = unchanged for every existing suite; `false` swaps
    `GPU_HARDWARE_TRUST` for `HOST_HARDWARE_TRUST` + `shared_frontend_lane_trust` on the primary lane).

    **Private dogfood** (`vericl-dogfood`, never committed; reported by construct class only): the
    production pure-cos/sin synthesis shape — whose private validation story IS f64-based
    (a private cubecl-cpu f64 proof: cos/sin bit-exact to host libm at f64 on
    cubecl-cpu) — annotated `instantiate(F = f64)` as a pinned-f64 instance (identical body
    to the existing f32 clean-room kernel; only the pinned type changed). `tests/f64_cpu.rs` (new
    `cpu` feature) validates three ways: (1) differential on cubecl-cpu vs the f64 twin passes across
    sizes; (2) bounds PROVED with declared comptime-implied lengths (4 obligations); (3) the twin is
    **bit-exact** to host libm f64 cos/sin — the production expectation — closing the loop kernel(cpu)
    == twin == host libm f64. No new subset wall surfaced; `instantiate(...)` monomorphized cleanly at
    f64.

    Verification: `cargo test --workspace` (default/wgpu) green — `evidence/vericl.json` verified
    UNCHANGED (byte-identical `git diff` empty; f64 added no kernels to it, and the compare-token
    rebuild fires only for f64 kernels); `cargo test -p vericl-examples --features cpu` green
    (conformance.rs's cpu extra-lane + conformance_f64.rs both pass); `cargo clippy --workspace
    --all-targets` and `--features cpu` both zero warnings (forced fresh, not cached). Counts: vericl
    core 25 (+f64 compare/rng tests), vericl-examples lib 50 (+3 axpy_f64), integration adds
    `float_method_whitelist_f64` (2), `f64_wgpu_unsound` (1), `conformance_f64` (1, cpu-only).
    `VERICL_UPDATE=1` was run LAST and ONLY on the new `conformance_f64` test binary (the sole
    evidence that changed), leaving `vericl.json`/`cooperative_fallback.json` untouched, per the
    staleness-guard lesson.

12. [DONE 2026-07-23] **Array-value-dependent indices (offset tables / gather)** — the last Tier-2
    prover gap (docs/dogfood-2026-07.md, ≥5 kernels), via element-range `assumes(...)`. All prover
    work in `crates/vericl-ir/src/prover.rs`; see that file's "Element-range assumptions" +
    "Write invalidation" module docs for the full soundness argument.

    **New assume forms.** `StructuredAssume::ElemsBelowLen { arr, len_of }` / `ElemsBelowConst {
    arr, bound }` (mirrored as `vericl_ir::Assume`), parsed from `A.iter().all(|v| (*v as usize) <
    B.len())` / `… < N` — the LHS normalized through parens/`as _` casts/`*` deref down to exactly
    the closure binding (so `*v + 1 < N` is correctly NOT recognized), strict `<` only (a `<=`
    admits `v == bound`, not a valid in-bounds guarantee, so it stays string-only). Unrecognized
    clauses stay string-only, sound (fewer constraints).

    **Prover encoding.** When an element assume covers global array `A`, a read `A[i]` — its own
    `0 <= i < A.len()` obligation still emitted and discharged as today — binds its output to a
    FRESH symbol `v` with `v < bound` (and `0 <= v` iff the element type is unsigned, a sound type
    fact; a signed element models `v < bound` alone so `0 <= index` stays a real proof). The ONLY
    case array contents get a model; everything else stays tainted. Gathers `x[offsets[i]]` and
    nested `a[b[i]]` prove (the fresh symbol flows through modeled arithmetic + is exactly the
    inner index the next layer needs); a wrong/too-loose bound REFUTES with the `elem…` symbol at
    the boundary.

    **Write invalidation** (both directions, tested): a write `A[j] = …` invalidates `A`'s assume
    for every *subsequent* read (`elem_invalidated`, monotonic); a read *before* a write keeps its
    model. For a loop, a body that writes `A` anywhere invalidates `A` for the whole body *before*
    the walk (a later iteration's write happens-before an earlier read — the in-order rule alone
    would be unsound). All conservative (only ever removes a model). Existing kernels are
    byte-identical (the invalidation set stays empty when no element assume is declared).

    **gen ergonomics.** The element bound doubles as the `gen(...)` range for the index array when
    it has no explicit `gen(arr in …)` — `Const(n)` draws `[0, n)`, `Len(b)` draws `[0, b.len())`
    (only when `b` is declared before `arr`, else it falls back to the resample loop's actionable
    panic). Stated once, in `assumes(...)`.

    **Examples.** `gather_copy` (`y[i] = x[offsets[i]]`, suite-wired: bit-exact differential +
    3-obligation `smt-oob-freedom` proof); `nested_gather` (`data[inner[outer[i]]]`, prover-only
    composition control); `gather_oob` (stale const bound `< 16` vs `x.len() == 8`, a `conform`
    demo-defect that REFUTES with `elem == x.len()`).

    **Private dogfood** (private-codebase policy — construct classes only). The offset-table source-anchor
    shape from a production coherent-accumulate primitive: the pure gather core `source[offsets[i]]`
    PROVES (3 obligations, no adaptation beyond the assume); the faithful additive anchor
    `out_idx + offsets[e]` is honestly REFUTED (the element assume bounds the offset but not the
    sum — a new "implicit host-side buffer-sizing invariant" finding needing a length-*relationship*
    assume beyond v0, same class as the div/mod and cooperative-store findings). Gen derivation
    validated (no `gen(offsets in …)` needed).

    **Soundness bar.** Taint over guess; write-invalidation tested both directions; a negative twin
    per positive; obligation counts for existing suite kernels unchanged; full suite (both feature
    sets) + clippy (both) + demo-defects green; `VERICL_UPDATE` run LAST (only `gather_copy` added
    an evidence entry).

13. [DONE 2026-07-23] Adversarial soundness review round 4 (element-range assume recognizer) —
    DONE. **Verdict: exactly one CONFIRMED defect** — a recognizer-level false-`Proved` in the
    element-range `assumes(...)` shape (item 12), fixed and regression-tested here. The
    **prover-core machinery survived every mission** the reviewer threw at it: the SMT bounds
    encoding, the fresh-symbol element model, `elem_bounds`/`elem_invalidated` write-invalidation
    (both directions), signed-vs-unsigned non-negativity, gather/nested-gather composition, and the
    `gather_oob` refutation all held with unchanged obligation counts. The single hole was upstream
    of the prover, in how the macro *decides which clause text becomes a structured constraint* —
    the prover faithfully proved what it was (wrongly) told.

    **D1 (CRITICAL) — arbitrary-cast peeling on the element-assume closure LHS
    (`crates/vericl-macros/src/lib.rs`).** `recognize_elem_assume` normalized the closure LHS via
    `peel_to_ident`, which peeled *any* `Expr::Cast`. A truncating chain
    `offsets.iter().all(|v| (*v as u8 as usize) < x.len())` was therefore recognized as
    `ElemsBelowLen { offsets, x }`, so the prover modeled the **un-truncated** offset `< x.len()`
    while the executable `check_assumes` evaluated the **truncation**. A contract-satisfying input
    (offset 256, `x.len() == 8`: `256 as u8 as usize == 0 < 8` is true) then earned a `Proved{3}`
    bounds certificate for a gather that reads `x[256]` — a real out-of-bounds read the reviewer
    reproduced in the twin (`check_assumes == true`, `Proved{3}`, twin panic "index out of bounds:
    len 8, index 256"). Fix: LHS cast peeling is now gated on **value-preservation** given the
    iterated array's element type. New `IntKind` (signedness + bit width; `usize`/`isize` pinned to
    their portable 32-bit floor — vericl runs its differential lane on 32-/64-bit hosts, so 32 is
    the certainly-safe minimum for a cast *target*) and `int_kind_of_type`; new `lhs_is_binding`
    replaces `peel_to_ident` and peels a cast only when the element type is a known unsigned integer
    and the target is a known unsigned integer at least as wide, **checked per step against the
    element width** (so `as u8 as usize` from `u32` is rejected at the `u8` step, 8 < 32, even
    though the outer `as usize` alone would pass). Any other cast — narrowing, signed source, signed
    target, unknown element/target type — leaves the whole clause string-only (sound: the prover
    never receives the bound and cannot prove the gather from it). The bare/deref binding with no
    cast is still recognized for any element type (a signed element still models `v < bound` alone,
    no unsound `0 <= v`), so nothing shipped regresses. Pinned decision on the multi-step
    `(*v as u64 as usize)` from `u32`: **accepted** — the value stays ≤ 2^32-1 through the `u64`
    widening, so the final `as usize` is value-preserving on every host; `u64 as usize` (from a
    `u64` element) is **rejected** as it could truncate on a 32-bit host.

    **RHS verified independently, left as-is (`peel_cast_paren`).** The reviewer flagged RHS
    truncation as the *safe* direction; I re-derived it per recognized shape rather than trusting
    the claim. For `ElemsBelowLen` (RHS `B.len() as T`) and `ElemsBelowConst` (RHS `N as T`), a cast
    of the non-negative bound satisfies `(bound as T) <= bound`, so peeling it can only make the
    executed bound **≤** the modeled bound — i.e. the model is at most *weaker* than the contract,
    never stronger. A weaker model admits *more* inputs than reality, so it can never mint a false
    `Proved` (the exact reverse of the LHS hazard). No RHS cast shape was found where the model is
    *stronger* than the contract, so no gate is needed there; the reasoning is recorded in the
    `peel_cast_paren` doc comment.

    **Blast radius — why this class is CRITICAL.** No shipped kernel used a truncating cast: the
    three gather kernels (`gather_copy`, `nested_gather`, `gather_oob`) all use the value-preserving
    `(*v as usize)` on `Array<u32>`/`Array<u64>` element arrays, still recognized identically, so
    `evidence/vericl.json` is byte-identical and every obligation count is unchanged. And a
    *suite-wired* kernel that hit this bug would have a second line of defense — the differential
    lane runs real inputs against the twin, and a truncating clause makes the twin itself panic OOB
    (as the reviewer's repro shows), so the differential would catch it. But a **proved-only**
    kernel (`proved` claim without `tested` — e.g. `nested_gather`, or any production kernel wired
    for the SMT proof alone) has **no such backstop**: the false `Proved` would be the only signal,
    and it would be green. That is precisely why a recognizer false-`Proved` is critical even though
    nothing shipped tripped it — the proof is load-bearing exactly where the differential is absent.

    **Tests.** Macro recognizer (`vericl-macros`, white-box over `recognize_assume`):
    `elem_assume_truncating_cast_chain_is_string_only` (the reviewer's exact repro — asserts `None`,
    no `ElemsBelow*` emitted), `elem_assume_single_narrowing_cast_is_string_only`,
    `elem_assume_value_preserving_forms_recognized` (shipped `as usize`, width-equal, widen-to-u64,
    bare/deref, by-ref binding, extra parens — all still recognized),
    `elem_assume_widening_chain_through_u64_is_recognized` (the pinned chain decision),
    `elem_assume_u64_element_to_usize_is_string_only` (host-portability rejection + width-equal
    accept), `elem_assume_signed_element_cast_rejected_bare_recognized`,
    `elem_assume_unknown_element_type_gates_cast_only`, `int_kind_classification`. Prover backstop
    (`vericl-examples`): `gather_copy_is_not_provable_without_element_assume` — WITHOUT the
    element-range assume the gather is NOT `Proved` (OutOfSubset/Refuted), pinning that a string-only
    clause cannot be laundered into a certificate.

    **Verification.** `cargo test --workspace` green (vericl 25, vericl-examples lib 56 [+1],
    vericl-ir 53, vericl-macros 23 [+8], integration all pass); `-p vericl-examples --features cpu`
    green with the cpu lane + evidence check (`evidence/vericl.json` byte-identical, `git diff`
    empty — no shipped kernel's contract changed, so no `VERICL_UPDATE` needed);
    `cargo clippy --workspace --all-targets` and `-p vericl-examples --features cpu --all-targets`
    both zero warnings; `conform` demo-defects exits 0 with output unchanged (all bounds/race/
    differential defects still caught — `gather_oob` still `Refuted` with the element symbol at the
    boundary, its `(*v as usize) < 16` clause still recognized). Reviewer's scratch repro reproduced
    against the pre-fix build and confirmed string-only + non-provable after; scratch not committed.

14. [DONE 2026-07-23] **Unbounded-integer overflow gap** — the "known-inert-on-naga" item (roadmap
    item 7): the prover modeled integers in unbounded QF_LIA, so `divisor = a * b` guarded by
    `a >= 1 && b >= 1` proved nonzero while real `u32` multiplication wraps `65536 * 65536 == 2^32`
    to exactly `0`; inert only because naga's div-by-zero fallback is dividend-preserving (a backend
    behavior, not a guarantee). All prover work in `crates/vericl-ir/src/prover.rs`; the full
    soundness argument + design rationale is that file's new "Bounded-integer overflow model" module
    doc (this entry summarizes).

    **Design decision — approach (b), a *faithful finite-width model in QF_LIA*, NOT (a) full
    QF_BV.** (a) would model wraparound for free but rewrite every existing encoding — bounds
    obligations, the length/element/gather assumes, the two-thread race walk, the cooperative leaves,
    div/mod — in a file hardened across four adversarial-review rounds, and thread signed-vs-unsigned
    bitvector-comparison discipline through every comparison: a large, review-hungry rewrite for a
    ~30-obligation/10-kernel suite that already solves sub-millisecond. (b) preserves every encoding:
    the change is confined to leaf declaration and the three arithmetic handlers (plus `Cast`).
    The refinement that makes (b) both sound AND non-disruptive: rather than the naive "taint any
    arithmetic that might overflow" (which breaks the legitimate guarded `x[pos+1]` pattern, whose
    wrap is benign because guard and index share the term), make the model **faithful** — every
    non-tainted modeled integer term equals the real (wrapping) hardware value, or is tainted.
    (i) Leaves declared in their type range `[type_min, type_max]` (a sound type fact — `usize`/
    positions/lengths are `u32` per `AddressType::U32`); (ii) `Add`/`Sub` folded back into range by
    an exact single-wrap `ite` (operands in range ⟹ at most one wrap); (iii) `Mul` carries a
    no-overflow side-obligation (`(a*b) mod 2^W` is QF_NIA-hard, so bind the plain product only when
    it provably cannot wrap, else taint — same discipline as div/mod); (iv) `Div`/`Modulo` unchanged
    but now sound-under-wrap because operands are faithful; (v) `Cast` passes value-preserving casts
    through, gates narrowing/sign-flip on a fits-in-destination side-obligation. **Consumer
    enumeration → completeness by invariant:** because every non-tainted term == the real value,
    EVERY consumer (index/bounds obligation, divisor, branch/loop guard, loop bound, race index,
    element-assume bound) reads the true value; a possibly-diverging term is tainted and the existing
    taint discipline already fails at whichever consumer needs it — so no consumer can be reached by
    a wrapped-but-untainted value, with no per-consumer casework.

    **The round-2 construction flips.** `a * b` divisor (`a,b >= 1`) → `checked_mul`'s side-obligation
    `a*b <= u32::MAX` fails (`a == b == 65536`) → `a*b` taints → modulo divisor taints → dependent
    guard `OutOfSubset`, never `Proved` (`prover::tests::mul_overflow_divisor_is_out_of_subset`). The
    `a + b == 2^32` variant is caught too — faithful `Add` gives the divisor term `0`, so the div
    nonzero check fails (`add_overflow_divisor_is_out_of_subset`).

    **Verdict changes on the suite (each justified).** `flatten_decode_scale` KEEPS `Proved`
    (2 obligations, unchanged) with NO assume strengthening — the anticipated "real finding" did not
    materialize: the leaf bound `ABSOLUTE_POS <= u32::MAX` plus the Euclidean fact `row*width <=
    ABSOLUTE_POS` discharges the `Mul` no-overflow side-obligation, and `row*width + col == ABSOLUTE_POS`
    is faithful. `fir_pair_kernel` (suite-wired) DID change: its guard `ABSOLUTE_POS + 1 < x.len()`
    silently relied on no-wrap to also cover the `x[ABSOLUTE_POS]` read (`pos+1 < len ⟹ pos < len`
    holds at every reachable dispatch but NOT at `pos == u32::MAX`, where `pos+1` wraps to `0`, the
    guard passes, and `x[pos]` is OOB — the faithful model `Refuted`s it there). Strengthened to
    `ABSOLUTE_POS < x.len() && ABSOLUTE_POS + 1 < x.len()` (genuinely more correct; safe at every
    reachable dispatch either way) → `Proved` again, 4 obligations unchanged. `tap_pair_guarded_kernel`
    (prover-only control) strengthened identically. Every OTHER suite kernel: counts/verdicts
    unchanged — `evidence/vericl.json` diff is exactly `fir_pair_kernel`'s two identity hashes
    (source + IR), obligation count still 4. The `wrapping` rule is explicit and needs no prover code:
    the `wrapping` clause never reaches the prover (which proves BOUNDS); a wrapped index is still OOB,
    so a `wrapping` kernel is treated identically (its value arithmetic already taints, its indices
    stay non-wrapping or become `OutOfSubset`).

    **Tests** (`crates/vericl-ir/src/prover.rs::tests`, +9): the round-2 regression
    (`mul_overflow_divisor_is_out_of_subset`) + `add_overflow_divisor`; per-consumer negatives —
    wrapped index (`mul_overflow_index_is_out_of_subset`), wrapped guard-of-different-index
    (`add_overflow_guard_refutes`, a genuine `Refuted` catching the danger), wrapped loop bound
    (`wrapped_loop_bound_is_out_of_subset`); positive controls — guard-bounded product proves
    (`guard_bounded_mul_proves`), the strengthened shifted-read proves
    (`shifted_read_selfguard_strengthened_proves`) while the lone-guard form `Refuted`s
    (`shifted_read_selfguard_refutes_at_type_max`, the `fir_pair` finding at the prover level),
    faithful underflow surfaces the true wrapped index (`sub_underflow_unguarded_refutes`).

    **Solver-time impact: negligible.** The 62 prover unit tests (all pure SMT, no GPU) run in 0.10s
    vs the pre-change 53 in 0.12s — the added `ite`s and side-obligations are cheap; z3 collapses the
    wrap `ite` under path facts. No kernel proof approaches a second.

    **Private dogfood spot-check** (private-codebase policy: the private source READ ONLY, `~/code/vericl-dogfood`
    writable-private, construct classes only): reran the production kernels' bounds/race/cooperative
    proofs against the overflow-model prover. **No production kernel flipped verdict** — the
    counter-RNG, div/mod-index, offset-table-gather, composition, and cooperative-reduction shapes all
    keep their prior `Proved`/`OutOfSubset`/`Refuted` verdicts and obligation counts (their
    divisors/indices are guard-bounded, comptime-pinned, or bare `ABSOLUTE_POS`, so the no-overflow
    side-obligations discharge and no wrap is reachable; none used an unbounded scalar product as a
    divisor/index). No new subset wall. Two PRE-EXISTING dogfood-test statenesses surfaced during the
    rerun, both unrelated to this milestone and confirmed so (one an `ElemsBelow*`-assume match left
    non-exhaustive since roadmap item 12; one a counter-RNG bounds test still asserting the
    pre-loop-carry-refinement `OutOfSubset` verdict from roadmap item 5 — verified pre-existing by
    stashing this milestone's changes and rebuilding, which reproduced the same `Proved{6}`); both
    corrected in `vericl-dogfood` to the true current verdicts, same precedent as the roadmap-item-8
    dogfood-staleness note.

    **Verification.** `cargo test --workspace` green (vericl 25, vericl-examples lib 56, vericl-ir 62
    [+9], vericl-macros 23, integration all pass); `-p vericl-examples --features cpu` green
    (conformance + f64 + cooperative_fallback + cooperative lanes); `cargo clippy --workspace
    --all-targets` and `--features cpu` both zero warnings; `conform demo-defects` exits 0, all
    defects still caught (`axpy_off_by_one`/`gather_oob` `Refuted`, `sum_racy` bounds `Proved`,
    `block_sum_reduce_racy` race `Refuted`); `evidence/vericl.json` diff is fir_pair_kernel's two
    identity hashes only, `evidence/vericl_f64.json` + `evidence/cooperative_fallback.json`
    byte-identical. `VERICL_UPDATE=1` run for `--features cpu` first (verified), then default LAST
    (committed default shape), per the staleness-guard lesson.

15. [DONE 2026-07-23] Adversarial soundness review round 5 (cooperative `AbsolutePos`
    recomposition) — DONE. **Verdict: exactly one CONFIRMED CRITICAL**, a cooperative-mode
    false-`Proved` where the `AbsolutePos` recomposition bypassed the faithful-integer invariant.
    Every other surface the reviewer probed **survived** (audited below). Prover-only change,
    confined to `crates/vericl-ir/src/prover.rs`; the full soundness argument lives in that file's
    "Cooperative mode" + "Bounded-integer overflow model" module docs (this entry summarizes).

    **D1 (CRITICAL) — unwrapped `AbsolutePos` in cooperative mode (`builtin_value`, ~1160).** The
    cooperative `AbsolutePos` was built as `cube_pos*cube_dim + unit_pos` with raw `smt.times`/
    `smt.plus`, the one integer term in the file constructed outside the faithful handlers. `cube_pos`
    is a *full*-`u32` leaf (`[0, 2^32)`), so that raw sum can exceed `2^32`, which real hardware wraps
    — the model's `abs_pos` was the **unwrapped over-value**, violating the module invariant that
    every non-tainted modeled integer term equals the real hardware value. A guard `ABSOLUTE_POS <
    output.len()` then forced `cube_pos*cube_dim < len` in the model and so transferred a bound onto
    `cube_pos` that hardware never honors. Reviewer's repro (a cooperative kernel guarding on
    `ABSOLUTE_POS` but indexing the *unguarded* `output[CUBE_POS]`) earned a false **`Proved{3}`**,
    while a `cube_pos = 2^24`, `unit_pos = 0`, `cube_dim = 256` dispatch computes `abs_pos = 2^32 ≡
    0 < len` (guard passes) and writes `output[2^24]` — wildly OOB for any `len <= 2^24` on a
    CUDA-class backend (WGSL robustness would clamp; the certificate is still unsound).

    **Fix — exact modular recomposition in QF_LIA (`abs_pos_sym`), NOT the taint route.**
    `AbsolutePos` in cooperative mode is now a fresh in-range `u32` leaf `abs_pos` asserted
    `abs_pos = cube_pos*cube_dim + unit_pos − k*2^32` for a fresh wrap count `k >= 0` (additionally
    `k <= cube_dim − 1`, the tight constant ceiling: `cube_pos <= 2^32−1` ∧ `unit_pos <= cube_dim−1`
    ⟹ raw sum `<= 2^32*cube_dim − 1` ⟹ `k = ⌊raw/2^32⌋ <= cube_dim−1`; not needed for soundness but
    cheap). Both products are variable×constant (`cube_dim`, `2^32` are constants), hence **LINEAR —
    QF_LIA, no QF_NIA**. This is *exact*: `abs_pos ∈ [0, 2^32)` congruent to the raw sum mod `2^32`
    is its unique residue = the true hardware value, so multiple wraps are handled and no unwrapped
    over-value can leak a bound. The taint route (a `checked_mul`-style no-overflow side-obligation
    on the recomposition) was rejected as decided: `cube_pos` is full-range so the product *always*
    can wrap, tainting `abs_pos` unconditionally and destroying **every** `ABSOLUTE_POS`-guarded
    cooperative proof — far too coarse when the exact encoding costs one leaf + one `k` + one linear
    equality. The plain-walk (`coop == None`) `AbsolutePos` is unchanged (a bare fresh `u32` leaf —
    already faithful for an opaque position; there is no recomposition to be faithful to).

    **Predeclaration (soundness-critical for the race walk).** Unlike the pre-fix raw term (a pure
    `SExpr`, no declarations), `abs_pos_sym` emits a `declare-const` + assertions, so it is
    **predeclared at the outermost SMT scope** — via `predeclare_coop_leaves` (bounds walk) and at
    the top of each thread's `race_walk` (per-thread, reading that thread's `UnitPos = t`). A lazy
    first resolution inside a branch arm would scope its declaration to that arm and drop it on the
    matching `pop`, leaving a *deferred* cross-thread race obligation whose recorded guard/index
    mentions `ABSOLUTE_POS` referencing an undeclared symbol — the exact hazard `race_setup` already
    predeclares buffer lengths against. Confirmed live in `conform demo-defects`: the
    `block_sum_reduce_racy` two-thread counterexample now shows both threads' predeclared symbols
    (`abs_pos7`/`abs_wrap8` for `t1`, `abs_pos9`/`abs_wrap10` for `t2`), each recomposition exact
    (`t1=1 ⟹ abs_pos=1`, `t2=0 ⟹ abs_pos=0`, both `abs_wrap=0`).

    **Regression test (`cooperative_abspos_guard_cubepos_index_refutes`, +1).** The reviewer's exact
    repro, made permanent. Under the fix it flips `Proved{3}` → **`Refuted`** (the honest verdict —
    the OOB is genuinely reachable, not merely `OutOfSubset`), with the witness `cube_pos=16843009,
    abs_wrap=1, abs_pos=16843008, len_output=16843009`: `abs_pos = 16843009*256 − 2^32 = 16843008 =
    len−1 < len` (guard satisfied) while `cube_pos = 16843009 = len` (index == length, OOB). Asserts
    the obligation is on `output` and the counterexample exhibits the large `cube_pos`.

    **Sibling-hunt audit (independent, per task).** Grepped every `smt.times`/`plus`/`sub`/`negate`/
    `div`/`modulo`/`ite` on integer terms and classified each: the only raw integer arithmetic
    *outside* a faithful handler was this one defect. All others are load-bearing and correct —
    `checked_mul`'s `times` (no-overflow side-obligation), `wrapping_binary`'s `plus`/`sub` (wrapped
    by `wrap_to_range`), `wrap_to_range`'s own `plus`/`sub`/`ite` (the single-wrap correction),
    `divmod_int`'s `div`/`modulo` (nonzero + nonnegativity side-obligation), and `constant_expr`/
    `int_const`'s `negate` (exact negative literals). Independently confirms the reviewer found no
    other site; documented in the module doc. Corrected the doc invariant claim to state the modular
    recomposition explicitly (the "Cooperative mode" bullet, the `Leaves` sub-bullet, and
    `prove_bounds_freedom_cooperative`'s rustdoc all now say `(CubePos*cube_dim + UnitPos) mod 2^32`,
    not the raw identity).

    **Survived surfaces (reviewer probed, held — confirmed real, no code change).** `wrap_to_range`'s
    `ite` re-validated exhaustively against true hardware wrapping for u8 **and** i8 (all `256^2`
    operand pairs × {Add, Sub} match — `wrapcheck.py`). `checked_mul` boundary-exact and load-bearing:
    `a,b < 65536 ⟹ Proved` but `a,b < 65537 ⟹ OutOfSubset` (product can hit `2^32`), the `a*b == 2^32
    ≡ 0` divisor construction still taints (`mul_overflow_divisor_is_out_of_subset`,
    `guard_bounded_mul_proves`). `cast_int` value-preservation gating intact
    (`adv_u64_narrow_*`-shaped `cast_int` fits-obligation). Faithful Add/Sub chains exact — `(pos−1)+1
    == pos`, unguarded `x[pos−1]` refutes at the true `2^32−1` (`sub_underflow_unguarded_refutes`).
    Element-assume + faithful-overflow interaction — `offsets[i]+1` can equal `x.len()` off-by-one,
    still refutes; plain gather still `Proved{3}` (`gather_with_element_assume_proves`,
    `nested_gather_composes_and_proves`, write-invalidation both directions). Guard-strengthening
    findings from round 2 still real (`shifted_read_selfguard_refutes_at_type_max` /
    `_strengthened_proves`). None of these flipped.

    **Verification.** `cargo test --workspace` green (vericl 25, vericl-examples lib 56, vericl-ir 63
    [+1], vericl-macros 23, integration all pass — `cooperative.rs` 3, `cooperative_fallback.rs` 1);
    `-p vericl-examples --features cpu` green (conformance + f64 + cooperative_fallback + cooperative
    lanes). **Shipped cooperative proofs unchanged (prover-only change; identities untouched):**
    `cooperative_shared_load_proves` `Proved{5}`, `block_sum_reduce_is_race_free` `Proved{19}`
    (bounds 8 + ww 6 + rw 4 + intercube 1 = 8+11), `grid_stride_reduce_is_race_free` `Proved{16}`,
    both `*_defers_to_m3` still `OutOfSubset`, `cooperative_undersized_tile_refutes` still `Refuted`
    on `unit_pos`. `cargo clippy --workspace --all-targets` and `--features cpu --all-targets` both
    zero warnings. `conform demo-defects` exits 0, output semantically unchanged. **Evidence
    byte-identical** — `git status` shows only `prover.rs` modified, no `evidence/*.json` touched
    (`ir_hash`/`source_hash` are macro/IR-derived, untouched by a prover change), and the suite's
    evidence-verify lanes pass without `VERICL_UPDATE`. **Private dogfood re-verified unchanged**
    (`vericl-dogfood`, path-dep on this checkout; private-codebase IP rules — construct classes only, no
    committed IP): the cooperative grid-stride reduction still `Proved` bounds(oob)=8 race=8 (ww=3 rw=4
    intercube=1) uniformity=2 phases=3, whole `dogfood-kernels` suite green (composition, instantiate,
    prover_subset, shmem_probe/min/conformance/reduction); `dogfood-rejects` still fails `cargo
    build` by design (compile-fail fixtures, unaffected by a prover-runtime change). All five rounds'
    regression tests green.

## Round-6 adversarial review (2026-07-23) — CLEAN

Verdict: MERGE-READY, no confirmed or suspected soundness defect — the second clean round
(rounds 1,2,4,5 each found one critical; round 3 clean). All four v1.1 surfaces held under
attack with real machinery: terminate modeling (eager bounds fire before not_cond — a
pre-terminate unguarded store Refutes; wider-than-len terminate bound Refutes; uniformity
verified before assertion so rejection never pollutes context), comptime baking (pinned
values confirmed identical across IR/twin/launch from one source; load-bearing bound flips
Proved/Refuted exactly at the boundary), helper rejection (recursive token scan caught
every evasion: aliasing, parens, nesting), and the barrier-count lane check (None-path
audited — unreachable from evidence-producing flows; rejection-only so cannot false-Prove).
Known benign asymmetry recorded: a global-array-read terminate condition is genuinely
uniform and the twin accepts it, while the prover taints the read and goes OutOfSubset —
suite degrades to the labeled assumed-race-freedom tier, never a false Proved.

## Ecosystem survey (2026-07-23) — tracel-ai's own CubeCL kernel libraries

Public-code counterpart to the private dogfood — full report in
`docs/ecosystem-survey-2026-07.md`. Ran VeriCL against tracel-ai's open-source kernel
libraries at VeriCL's pinned `cubecl = "=0.10.0"`. Work in a sibling workspace
(`/Users/ryland/code/vericl-ecosystem-survey`); no vericl-repo source changes beyond the
survey doc and this addendum; no commits.

**Mapping finding (premise correction).** The named targets are NOT crates in `tracel-ai/cubecl`
at 0.10.0 — the cubecl 0.10.0 meta-crate ships only `cubecl-std` as a kernel library. For this
generation the algorithm kernels live in a separate repo, `tracel-ai/cubek` (published `cubek`
v0.2.0, pins cubecl 0.10.0): `cubek-random`, `cubek-reduce`, `cubek-matmul`, `cubek-convolution`,
`cubek-std`. `burn` v0.21.0 consumes both; `burn-cubecl/src/kernel/*` are host wrappers with zero
`#[cube]`. Also: cubecl-random is a **Tausworthe-88 + LCG** hybrid, not Philox (does not weaken
the "proven this shape before" premise — it is `xorshift_step`/`mix_u32`-shaped).

**Gap map (464 device `#[cube]` items; ecosystem-wide gate ranking, item×gate incidences).**
1 `Line`/`Vector` **148** · 2 `View`/`Slice` **128** · 3 `comptime!{}` blocks 120 · 4 `match`/Switch 119 ·
5 `plane_*` 88 · 6 rejected methods (`cast_from`/`mul_hi`/…) 82 · 7 custom `CubeType` struct params 68 ·
8 cmma/Matrix 62 · 9 2-D topology 39 · 10 `Tensor` 32 · 11 `SharedMemory` 24 (supported only in the
1-D cooperative scalar subset) · 12 `select()` 9 · 13 `Atomic` 1. Launch entry-points are very few
(cubek-random 1, cubek-reduce 2, cubek-matmul 4, cubek-std 0, cubek-conv 0) and all maximally gated
— the annotatable content is the reusable scalar device helpers underneath, not the dispatch sites.

**THE HEADLINE — the frontier flipped vs the private codebase.** The private dogfood found *zero* `Line`/`Vector`/
`Slice`/`plane_*`/`Atomic`/`Tensor` and withdrew Tensor/2-D speculation (correct for that codebase). The
ecosystem's own libraries are the mirror image: `Line`/`Vector` is the #1 gap, `View`/`Slice` #2. The
two disagree because they occupy different layers (private codebase: 1-D scalar app kernels; cubek/cubecl:
the vectorized tensor-algebra layer). **Recommended next milestone: `Line`/`Vector` element
support (twin = length-N lane array; per-lane compare; bounds over the outer index), scoped first to
1-D vectorized elementwise + reduction shapes where the topology/proof machinery already exists, with
`View`/`Slice` as the immediate follow-on.** This is the change that converts "VeriCL proves the
reusable scalar cores of tracel-ai's kernels" into "VeriCL proves tracel-ai's kernels".

**Shortlist — 8 kernels, full tested + proved pair, on TWO differential lanes (wgpu/WGSL/Metal +
cubecl-cpu).** All bodies verbatim from upstream (MIT/Apache-2.0, cited); `*_map` drivers are 1-D
glue; contracts ours. Evidence: `vericl-ecosystem-survey/annotated/evidence/vericl.json`.
- cubek-random RNG core: `taus_step_0/1/2` (via `taus0/1/2_map`, composition/helper-calling-helper,
  `Proved{2}` each) · `lcg_step` (via `lcg_map`, **wrapping**, `Proved{2}`) · `combined_taus_lcg`
  (cubek's full per-value output `taus0^taus1^taus2^lcg`, **wrapping + `uses(...)` together**,
  `Proved{5}`). All `compare(exact)`, bit-exact on both lanes.
- cubecl-std: `to_degrees`/`to_radians` (**generic** `instantiate(F=f32)` + composition, `abs`
  tolerances derived from input range, `Proved{2}`) · `shift_right` (`#[comptime]` bool pass-through,
  `exact`, `Proved{2}`).
Confirms composition, `instantiate`, `wrapping`, `#[comptime]` params all land on real upstream code
with zero adaptation. Positive result: `wrapping` + `uses(...)` co-exist in one kernel.

**Findings classified.**
- VeriCL gaps: (1) `Line`/`Vector`+`View` (the frontier, above). (2) `cast_from` blocks cubek-random's
  `u32→f32` converters (`to_unit_interval_*`) — the exact seam between the provable integer core and
  the float-conversion boundary; verified clean macro rejection on the real body. (3) `wrapping` is
  kernel-only, so cubek's wrap-intent `lcg_step` cannot be a `#[vericl::helper]` — inline-into-a-
  wrapping-kernel is the faithful path (proves); a helper-level `wrapping` would let the LFSR/LCG steps
  compose end-to-end. Residual, low urgency.
- Implicit invariant: none in the shortlist. One observation outside it — cubek-reduce `shared_sum`'s
  prose-only "caller must zero the output" obligation (same class as the dogfood findings), not
  annotatable today (`Atomic`+`Vector`+`View`+generic).
- Real upstream bugs: none (mature library code; bit-exact on two backends, provably in-bounds).
- Negative controls (discrimination proven, `annotated/src/bin/negatives.rs`, exit 0): `lcg_map_oob`
  (`<=` guard) `Refuted` at `abs_pos==len` while honest `lcg_map` `Proved{2}` (both directions);
  `lcg_map_nowrap` differential catches the checked twin panicking on overflow. Macro-gate rejections
  captured on real bodies (`cast_from`; `Array<Vector<…>>` element).

Roadmap consequence: elevate `Line`/`Vector` (+ `View`/`Slice`) to the next milestone slot — it is the
demand-ranked #1/#2 gap across tracel-ai's own libraries, ahead of any remaining scalar-tier
follow-up. Upstream-conversation-worthy: the RNG-core proof result, and the `wrapping`-on-helper /
runtime-`cast_from` expressiveness gaps.

## Queued (deferred by Ryland, 2026-07-23): upstream f64 disclosure

Report the cubecl 0.10 wgpu f64 silent-corruption bug (pinned by
crates/vericl-examples/tests/f64_wgpu_unsound.rs) to tracel-ai — QUEUED, do not send
without Ryland's explicit go. Pre-send verification checklist:
1. Reproduce against cubecl main and any 0.11 pre-release (fixed? rejected? still silent?).
2. Check the SPIR-V compilation path (cubecl-wgpu spirv feature) — f64 is expressible there;
   behavior may differ from the WGSL/naga path.
3. Draft framing: "still present at <commit>" or "fixed on main, published 0.10 affected —
   consider advisory"; include the minimal repro + the corruption-not-demotion diagnostic
   (axpy expected 1776.99, got 526.99; cpu lane bit-exact).
The broader upstream package (7 compiler/runtime findings + docs/ecosystem-survey-2026-07.md)
goes with it when Ryland green-lights contact.

## Agreed sequencing (Ryland, 2026-07-23)

Quick wins -> Line/Vector -> certificates + IR-interpreter cross-check (parallel with
View/Slice) -> cubecl-0.11 upgrade drill -> productization gate. plane_* deferred.
Ladder decision: take Rung A (cvc5/Alethe certificates + counterexample validation) and
Rung B (IR reference interpreter + fuzz cross-check); skip Verus/Lean tiers (documented
rationale: cost, capability reduction, and no authoritative CubeCL spec to anchor them).

Quick-wins batch 1 (prover-leaning): match/Switch modeling; length-relationship assumes.
Quick-wins batch 2 (macro-leaning): verified cast_from/mul_hi host shims; helper-level
wrapping; comptime! block evaluation under instantiate. Review round 7 after both.

## Quick-wins batch 2 (macro-leaning) — DONE 2026-07-23

Three features at the seven-round soundness standard, motivated by
`docs/ecosystem-survey-2026-07.md` (§3a residuals + §4 recommendation #3). Round-7 review
attacks batches 1+2 together.

**Feature 1 — verified `cast_from` / `mul_hi` host shims.** New `crates/vericl/src/host_shims.rs`
(vericl core, no cubecl dep). `Cast::cast_from`/`Numeric::mul_hi` are `unexpanded!()` on host
(they panic — the reason they were `FLOAT_METHOD_REJECT`ed, blocking 82 surveyed kernels incl.
cubek-random's u32→f32 converters). A new `ShimRewriteFold` (runs in both the kernel and helper
twin pipelines, BEFORE `FloatMethodCheck`) rewrites recognized intrinsic CALLS to GPU-verified
shims: `f32::cast_from(x)` → `::vericl::host_shims::cast_to_f32(x)` (source type resolved by Rust
trait dispatch, `CastToF32` impl'd for u32/i32/f32-identity — the surveyed set; an unsupported
source is a `CastToF32: not satisfied` compile error in the twin, loud); `T::mul_hi(a,b)` and
`a.mul_hi(b)` → `::vericl::host_shims::mul_hi(a,b)` (`MulHi` trait, u32 only). An UNRECOGNIZED
`cast_from` (non-f32 target, e.g. `u32::cast_from` / `f64::cast_from`, or a qualified-self path) is
left unrewritten and still rejected BY NAME by `FloatMethodCheck` (which now skips only
`vericl`-rooted paths so the shim's own `mul_hi` last-segment isn't self-rejected). A bare
single-segment `mul_hi(a,b)` is left alone (it cannot be the intrinsic — always `T::mul_hi`/
`x.mul_hi`), so a `uses(...)` helper named `mul_hi` is not hijacked.
  **Shim set (scoped to survey demand):** `cast_from_u32_f32`, `cast_from_i32_f32`,
  `cast_from_f32_f32` (identity), `mul_hi_u32`. Reject everything else (by name for wrong cast
  target; by trait-bound error for unsupported source/operand type).
  **GPU ground truth (the load-bearing verification — GPU-defined semantics, verified against GPU
  not std).** `crates/vericl-examples/tests/host_shim_gpu_ground_truth.rs` runs the REAL intrinsic
  in real `#[cube]` kernels and asserts the shim matches bit-for-bit across boundary + random
  inputs (u32/i32 incl. the >2^24 rounding-sensitive range; mul_hi full range). **FINDING /
  result: on wgpu/Metal AND cubecl-cpu, `cast_from` u32→f32 and i32→f32 match Rust `x as f32`
  bit-for-bit (both round-to-nearest-even), and `mul_hi` u32 matches `((a as u64)*(b as u64))>>32`
  bit-for-bit — NO divergence between backends, and none from `as f32`.** (So the "verify, don't
  assume the rounding mode" concern resolved to agreement; documented in the shim module + test.)
  **Prover:** unchanged — cast_from produces a float (tainted anyway) and mul_hi's high word taints
  via the existing `Arithmetic` catch-all (`_ => None`, prover.rs); neither feeds an index in the
  examples, so bounds still `Proved` (taint is fine v1, documented).
  **Flagship example:** `unit_interval_map` (`crates/vericl-examples/src/lib.rs`) — the u32-RNG →
  unit-interval-f32 kernel `y[i] = cast_from(x[i] >> 8) / 2^24` via a composed `to_unit_interval`
  helper (Lemire's technique / cubek `to_unit_interval_closed_open` shape), bit-exact (max_ulp=0),
  `Proved{2}`. Plus `mul_hi_map` (exact u32, `Proved{3}`).

**Feature 2 — helper-level `wrapping`.** `#[vericl::helper(wrapping, ...)]` now accepted (was
rejected). `WrappingFold` applied to the helper twin body under the same integer-only gate as
kernels (every value param + the RETURN type must be u32/i32/u64/i64; the untyped fold must not
touch float math — a float param/return is rejected). **Interaction rule decided + documented +
tested (ergonomics-first): each item's `wrapping` governs ONLY its own body; integers cross the
helper boundary as plain values.** So (a) a NON-wrapping kernel freely uses a wrapping helper — the
flagship `lcg_map` (`y[i] = lcg_step(x[i])`, non-wrapping kernel, `#[vericl::helper(wrapping)]
lcg_step` = `z*a+b`); and (b) a wrapping kernel using a non-wrapping helper gets the helper's
CHECKED arithmetic, which panics loudly on overflow — the round-3 behavior KEPT (not forced
clause-matching, which would wrongly reject `taus_step`-style shift/xor helpers that never
overflow). Both halves pinned: `lcg_step_twin_wraps_on_overflow` (wrapping helper never panics) +
`nonwrapping_helper_twin_panics_on_overflow` (the `lcg_step_checked` negative-control helper's
twin panics on overflow, vs its wrapping sibling wrapping cleanly).

**Feature 3 — `comptime! { EXPR }` block evaluation.** Was blanket-banned (`comptime` ∈
`BANNED_IDENTS`). New token-level pre-pass `rewrite_comptime_blocks` (runs before `transform_body`
in kernel + helper) strips `comptime! { EXPR }` → `(EXPR)`/`{EXPR}` (host Rust the twin re-runs —
exactly what cube does at expansion) IFF every bare value identifier EXPR references is a
`#[comptime]` parameter (concrete under instantiate) or a literal; multi-segment paths / method
names / field accessors are not runtime values and are allowed. Rejected BY NAME otherwise: a
reference to a runtime scalar/array/local (names it), or a nested macro invocation (opaque tokens
can't be validated — e.g. `comptime!(assert!(...))` rejected). A leftover bare `comptime` ident
still hits the `BANNED_IDENTS` ban. Example: `comptime_shift` (`shift = comptime!(extra + 2)`,
`extra` pinned via `instantiate(extra = 1)`), exact u32, `Proved{2}`.
  **Coverage (honest):** UNLOCKS comptime! blocks that are pure host arithmetic/logic over scalar
  `#[comptime]` params + literals — the shape vericl's subset can actually have (surveyed real uses
  like `comptime!(layout.num_rows * layout.num_cols)` on scalar comptime, `comptime!(extra+1)`,
  `comptime!(a.min(b))`). REMAINS REJECTED (correctly, out of subset): comptime! over custom
  CubeType struct params (`comptime!(t.config.tile_size.m())` — the dominant real shape, but needs
  struct comptime params vericl doesn't support), nested macros, and any runtime-value reference.
  Of the survey's 120 comptime!-block incidences, essentially all are struct-typed (matmul/reduce
  tile config) so remain gated on the unrelated CubeType gap; the scalar-comptime shape this
  unlocks is the one vericl kernels can express.

**Ecosystem validation (verbatim cubek shapes, `/Users/ryland/code/vericl-ecosystem-survey/
annotated`, non-destructively — backed up + restored byte-identical, coordinated with the
ecosystem-survey agent; workspace is not a git repo).** Closes survey §3a residuals 2 & 3 on the
real bodies: `combined_taus_lcg` recomposed to `uses(taus_step_0/1/2, lcg_step)` with `lcg_step` a
`#[vericl::helper(wrapping)]` (dropping the kernel's own `wrapping` clause — a NON-wrapping kernel
composing a wrapping helper), still `Proved{5}` + bit-exact on wgpu+cpu (identical to the old inline
form); `to_unit_interval_closed_open` (verbatim base.rs 191-197, `f32::cast_from`) added as a helper
+ driver, compiles (was §3a rejection), 0-ULP on wgpu+cpu, `Proved{2}`; negatives bin still exits 0.
(Noted a pre-existing drift for the survey agent: their `negatives.rs` match on `StructuredAssume`
lacks the `LenPlusConstLe` arm batch-1 added — their point-in-time snapshot; not my change to make.)

**Files.** `crates/vericl/src/host_shims.rs` (new); `crates/vericl/src/lib.rs` (+`pub mod
host_shims`); `crates/vericl-macros/src/lib.rs` (`ShimRewriteFold`, `rewrite_comptime_blocks` +
`ComptimeRefCheck` + `validate_comptime_expr`, `HelperSpec.wrapping` + helper wrapping gate + fold,
`FloatMethodCheck` vericl-path skip, generated `unused_parens`/`unused_braces` allow);
`crates/vericl-examples/src/lib.rs` (5 kernels + 3 helpers: `to_unit_interval`+`unit_interval_map`,
`mul_hi_map`, `lcg_step`+`lcg_map`, `lcg_step_checked` [neg control], `comptime_shift`; 7 new twin
guards); `crates/vericl-examples/tests/host_shim_gpu_ground_truth.rs` (new, GPU ground truth);
`crates/vericl-examples/tests/conformance.rs` (4 suite kernels); `crates/vericl-examples/evidence/
vericl.json` (+4 kernels, additive only — 255 insertions, 0 removals, no cpu-lane leakage);
`README.md` (comptime! accuracy fix). `docs/` unchanged (survey doc is the ecosystem-survey agent's
point-in-time snapshot).

**Verification.** `cargo test --workspace` green (244 tests, 0 failed; +4 host_shims unit,
+11 vericl-macros batch-2, +7 vericl-examples twin guards, +1 GPU ground-truth wgpu);
`-p vericl-examples --features cpu` green incl. the cpu GPU-ground-truth lane
(`shims_match_cpu_ground_truth`, no backend divergence) + the cpu conformance lane;
`cargo clippy --workspace --all-targets` zero warnings on BOTH feature sets; `conform demo-defects`
exits 0 (unchanged); the 4 new suite kernels each carry `tested` (differential) + `proved`
(smt-oob-freedom) claims; existing kernels' evidence byte-identical (suite obligation counts
unchanged; +4 kernels justified). `VERICL_UPDATE` run cpu-then-default LAST per the staleness
lesson — committed evidence is default (non-cpu) shape. Negative test per positive (macro-level
white-box + example-level twin guards + the neg-control helper). No prover changes (taint suffices
v1, documented).

## Round-7 adversarial review (2026-07-23) — MERGE-READY

Verdict: **MERGE-READY, no false-`Proved`** — the review attacked batches 1+2 together (the
shim/comptime/helper-wrapping surface) plus the whole prior stack. **No confirmed CRITICAL; no
suspected soundness defect.** Two LOW findings were closed here (one defense-in-depth, one
pre-existing blast-radius-class), and one pre-existing loud-only hazard was classified and queued.
The surfaces the reviewer probed **held under attack with real machinery**: signed-scrutinee
default reachability (a `match` on a signed value with an unreachable-looking default still walked
and defended), the divmod nonzero/nonnegativity gate protecting the `Switch` lowering, the
taint-union write machinery across branch/switch merges, huge-`K`/zero-length `LenPlusConstLe`
handling (the `K` literal never wraps the length leaf), shim discrimination proven by an injected
1-ULP defect (the twin's shim vs the real intrinsic caught it), and the comptime-laundering
rejections (a runtime-value or nested-macro `comptime!` still rejected by name). None flipped.

**F1 (LOW, defense-in-depth) — paren-wrapped intrinsic evasion of the shim rewrite / float-method
reject (`crates/vericl-macros/src/lib.rs`).** `(f32::cast_from)(x)` / `(u32::mul_hi)(a, b)` (a
parenthesised callee) slipped past `ShimRewriteFold::rewrite_call` and `FloatMethodCheck::
fold_expr_call` — both inspected only `Expr::Path` at the immediate callee, so a `Expr::Paren`/
`Expr::Group` wrapper classified as neither a recognized intrinsic (→ shim rewrite) nor a rejected
name. The result was never silent — the un-rewritten `unexpanded!()` intrinsic panics loudly
host-side in the twin — but the round-3/4 paren-peel standard (F1 audit, the `UsesRewriteFold`
precedent) must apply uniformly. Fix: peel `Paren`/`Group` on the callee in **both** folds before
classifying — `ShimRewriteFold::rewrite_call` now peels in place via the existing `peel_paren_group`
(identity for the twin; byte-identical for the common bare callee), and `FloatMethodCheck::
fold_expr_call` peels **borrow-only** via `peel_paren` (that fold is a pure check and must not mutate
the tree). Defense-in-depth: `ShimRewriteFold` runs first and already un-parens the callee, but
`FloatMethodCheck` now rejects correctly in isolation / under any reordering. Tests (`vericl-macros`,
+2): `shim_rewrite_peels_parenthesised_callee` (`(f32::cast_from)(x)`, `(u32::mul_hi)(a,b)`, and
doubly-nested, all rewrite identically to the bare form) and `float_method_check_peels_
parenthesised_callee` (`(f32::inverse_sqrt)(x)` and `(u32::mul_hi)(a,b)` reject identically bare vs
parenthesised, with a non-rejected `(some_helper)(x)` negative control) — both fail pre-fix.

**F1 sweep — newer scans since the round-4 audit (the reviewer asked for a sweep, not a point
fix).** The round-4 F1 audit table (roadmap item 10) covers the older sites; the scans added since
(comptime, coop, composition) were re-audited and are **NOT paren-vulnerable**: (1)
**`rewrite_comptime_blocks`** operates at the TOKEN level and recurses into **every** `Group`
(delimiter-agnostic), so a `comptime!` inside any paren/brace/bracket group is still found and
rewritten — paren-safe by construction, not by a shape gate. (2) **`ComptimeRefCheck`** (a
`syn::Visit`) reaches inner `ExprPath`/`Macro` nodes THROUGH `Expr::Paren`/`Expr::Group` via the
visitor's default recursion, so `comptime!{ (runtime_val) + 1 }` still rejects the wrapped
`runtime_val` and a wrapped nested macro still rejects — paren-safe by construction. (3) The
positive assume-recognizers (`recognize_elem_assume`, the `len`/`all` matchers) and attribute
parsers (`reference = <path>`, `instantiate` type-value classification) key on exact `Expr::Path`/
`Expr::MethodCall` shapes, but non-recognition there is **conservative** (a rejected clause or a
string-only/tainted model — the prover simply never receives the fact), never an evasion of a
rejection. Only the two intrinsic-classifying folds had the dangerous direction (non-peel = evade a
reject/rewrite), and both are now peeled.

**F2 (LOW, pre-existing round-4 blast-radius class) — degenerate element bound vacuously proves
(`crates/vericl-ir/src/prover.rs`).** `ElemsBelowConst { bound: 0 }` on an **unsigned** array makes
every modeled read `offsets[i]` carry `0 <= v ∧ v < 0` (`model_element_read`'s `declare_leaf` gives
the unsigned `0 <= v`, the bound gives `v < 0`) — an infeasible per-read context that vacuously
discharges every obligation emitted under the read (the read's OWN bounds obligation included),
minting a confirmed false **`Proved{3}`** on an otherwise-OOB gather (demonstrated by temporarily
disabling the new gate: both degenerate shapes returned `Proved{3}`; scratch not committed). The
length-only infeasibility gate does not cover element bounds by design — element assumes assert
nothing global (they only populate `elem_bounds`), so the outermost length check-sat is SAT. Fix
(soundly, minimally, in the gate): after the length check-sat, `assert_element_bounds_feasible`
probes each array carrying bounds with a fresh **witness** element of that array's scalar type
(`buffer_tys[id]`, threaded from `def.buffers[id].ty` — the same signedness/width `declare_leaf`
would give a real read: `0 <= w` for unsigned, `type_min <= w` for signed, plus `w < b` for every
recorded bound). It tests exactly "could ANY element satisfy the bounds" — which is precisely what
modeling a read assumes — so an unsatisfiable-for-any-element bound (unsigned `< 0`, or a chained
`len_of` pinned `0`) makes the probe UNSAT → `OutOfSubset` "contradictory assumptions". A signed
element bounded `< 0` is genuinely satisfiable (`w = -1`), so it is NOT rejected (its later index
obligation fails honestly on `0 <= idx` instead). The probe is isolated in a `push`/`pop` with the
witness declared **raw** (a fixed `__elem_feas_{id}` name, outside `self.declared`/`self.fresh`), so
a satisfiable bound perturbs neither the walk nor any later counterexample — confirmed by
`gather_oob`'s `demo-defects` counterexample staying byte-identical (`elem5=8`, no `__elem_feas`
leak). The reviewer-flagged comment ("tests exactly the assumption feasibility") was corrected to
state precisely that the length check-sat tests *length-fact* feasibility only, with element-range
feasibility handled by the separate witness gate. Tests (`vericl-ir::prover::tests`, +3):
`degenerate_zero_element_bound_is_out_of_subset` (the reviewer's bound-0 vacuous shape → OutOfSubset,
was `Proved{3}`), `satisfiable_constant_element_bound_still_proves` (`offsets[·] < 8` with `x.len()`
pinned `8` → `Proved{3}`, the witness is a no-op), `element_bound_forced_to_zero_by_length_chain_
is_out_of_subset` (the chained `ElemsBelowLen{offsets,x}` + `x.len()==0` case that guard (1) misses
and the element witness catches).

**Classified + queued (not fixed here) — predeclaration hazard in the loop handlers.** The reviewer
noted that a symbol first resolved lazily inside a loop-body arm can, in the same shape the round-5
`abs_pos_sym` predeclaration fix addressed, reference a scope-dropped declaration. In the current
loop handlers this surfaces as a **loud `SolverError`** only (z3 errors on the undeclared symbol —
never a silent `Proved`), it is **pre-existing** (predates v1.1, unrelated to batches 1/2), and no
evidence-producing flow reaches it. Classified LOW, **queued as cleanup, not urgency** — the same
predeclaration discipline `predeclare_coop_leaves`/`race_walk` already apply, extended to the loop
leaves, is the eventual tidy-up; it cannot mint a false certificate as-is.

**Verification.** `cargo test --workspace` green (vericl 29, vericl-examples lib 75, vericl-ir 87
[+3], vericl-macros 46 [+2], integration all pass — conformance 1, conformance_f64 0/1, cooperative
6, cooperative_fallback 1, f64_wgpu_unsound 1, float_method_whitelist 2, float_method_whitelist_f64
2, host_shim_gpu_ground_truth 1); `-p vericl-examples --features cpu` green incl. the 34s cpu
conformance/evidence lane + the cpu GPU-ground-truth lane (host_shim_gpu_ground_truth 2). `cargo
clippy --workspace --all-targets` and `-p vericl-examples --features cpu --all-targets` both zero
warnings. `conform demo-defects` exits 0, output unchanged — every defect still caught, `gather_oob`
still `Refuted` with a byte-identical element-symbol counterexample. **Evidence byte-identical** —
`git status` shows only `prover.rs` + `lib.rs` (macros) modified, no `evidence/*.json` touched; the
cpu conformance evidence-verify lane passes **without `VERICL_UPDATE`** (the witness addition is a
no-op for every satisfiable suite assume — F1 only changes macro-time *rejection*/rewrite of a
paren shape no shipped kernel uses, F2 only adds an isolated feasibility probe that never enters the
walk). All seven rounds' regression tests green (spot-checked: `mul_overflow_divisor_is_out_of_
subset`, `add_overflow_guard_refutes`, `cooperative_abspos_guard_cubepos_index_refutes`,
`shifted_read_selfguard_refutes_at_type_max`, `elem_assume_truncating_cast_chain_is_string_only`,
`uses_rewrite_fold_rewrites_parenthesised_helper_call`, `gather_copy_is_not_provable_without_
element_assume`).

## Line/Vector element milestone — V1–V6 DONE (2026-07-23)

Delivered `Vector<P, N>` support for the **vectorized elementwise class** per
`docs/design-line-vector.md` (the ecosystem survey's #1 gate, 148/464 device items). At the pins it is
`Vector<P, N>` with a **comptime** width (not the pre-0.10 launch-dynamic `Line<T>`), so it pins per
contract via `instantiate(N = W)` exactly as a generic float pins via `instantiate(F = f32)`. Six
milestones; round-8 will attack this work, so every positive has a negative control.

**V1 — soundness guard + prover confirmation (prover).** `is_modeled_int` now requires
`ty.vector_size() == 1` (`crates/vericl-ir/src/prover.rs`): `Type::Vector(u32,N).is_int()` is `true`
(a vector's *storage* is integer), so without the guard a 4-lane vector was eligible to be modeled as
ONE scalar SMT `Int` — the design's pre-registered round-8 risk 1. Whole-vector indexing lowers to
`vector_size: 0` (width in the list's `Type`) and `.len()` is line-granular, so the bounds obligation
is the scalar one unmodified — `N` never enters it. Tests: `is_modeled_int_rejects_vector_integers`
(predicate), `whole_vector_vec_add_proves`/`..._unguarded_refutes`, `vector_offsets_gather_is_out_of_
subset` (a `Vector<u32,N>` value under `ElemsBelowConst` lands `OutOfSubset`, never a vacuous per-lane
bound).

**V2 — the `vericl::Line<T,W>` shim + GPU ground truth (core + test).** `crates/vericl/src/line.rs`:
`Line<T, W>` = `[T; W]`, every op a per-lane map, the full v1 op surface (arith/bitwise, splat/ctor,
per-lane compare → `Line<bool,W>`, `count_ones`, width query, lane index, the per-lane float-math
whitelist). Every op **GPU-ground-truth-verified** bit-exact against a real `Vector<_,N>` kernel on
wgpu (+cpu) — `crates/vericl-examples/tests/line_shim_gpu_ground_truth.rs`. Finding: Metal `f32 /` is
not correctly-rounded (≤1 ULP), reclassified to the vec==scalar tier (GPU vec op == GPU per-lane
scalar op, bit-exact) with a ≤2-ULP twin bound — the same legitimate float gap `compare(abs=…)` covers.

**V3 — vector recognition + `instantiate(N=W)` (macro).** `Array<Vector<F,N>>` params recognized;
twin element mapped to `&[::vericl::Line<F,W>]`; the `Vector` head rewritten to `::vericl::Line` in
twin bodies (so `Vector::new(s)` splat → `Line::new(s)`); the IR face uses `Vector<F, Const<W>>`; the
`Vector` ban lifted only under the vector gate. Cross-lane reduces (`dot`/`magnitude`/`normalize`)
rejected by name; unpinned width rejected with the targeted "add `instantiate(N = W)`" message.

**V4 — vectorized launch/I/O + gen + compare (macro + core).** `build_vector_conformance_items`
(`crates/vericl-macros/src/lib.rs`): scalar I/O throughout (design §4.4) — `gen` draws `lines*W` flat
scalars per array reshaped to `Line`s, buffers sized `lines*W`, the `launch` splices `W` as the
vectorization `usize` after `cube_dim` (a `Size` generic is monomorphized OUT of the launch turbofish,
unlike `expand` — the `launch_generic_types` field), readback flat, twin `Line` output flattened, the
flat-scalar compare reports a divergence per lane `(line = i/W, lane = i%W)`. New evidence config
`differential_vector_config` records `vector_width` + `sizes_unit: "lines"`; the kernel module exposes
`VECTOR_WIDTH`, which `suite!` reads to select it. Conformance element-type gate: only f32/f64 vector
outputs (§8.3); mixed scalar/vector-array kernels rejected. Examples: clean-room `vec_add`,
`vec_scale` (splat), `vec_madd` (`a*a+b`, the explicit FMA-contraction `compare(abs=…)` tolerance
example — probe: 259/1028 lanes diverge bit-exact, covered by abs). Tests
(`crates/vericl-examples/tests/vector_conformance.rs`): `vec_add`/`vec_scale`/`vec_madd` conformance
passes on wgpu (+cpu) across sizes; `vec_add_off_by_one` (`<= out.len()`) caught (reference panic);
`vec_madd_bitexact` (`compare(max_ulp=0)`) caught with the lane NAMED (`out[line=244, lane=0]`).
Macro-gate tests: integer-vector output rejected, mixed-array rejected, `gen` on a vector kernel now
accepted.

**V5 — per-lane comptime-unroll acceptance + public example (prover + suite).** The prover now accepts
a comptime-unrolled `for j in 0..W` affine-in-lane write into a **register** vector: `register_vector_
lane` + `constant_lane_index` route a `LocalConst`/`LocalMut` `Vector` list with a `Constant` lane
index to `process_register_lane`, which carries NO buffer obligation and taints the lane value (model
(b), lane contents unmodeled) — the only per-lane shape the 148 use (IR validated: the unrolled `v[j]`
lowers to `index: Constant(UInt(j))`). A **data-dependent** (runtime) register-vector lane index is
`OutOfSubset` with the §8.3 "only comptime-unrolled lane loops" message; a per-lane loaded value used
to index another array (`x[v[j]]`) refuses at the global index — never a bound minted from a vector
value. Tests: `lane_unroll_comptime_pattern_proves` (`Proved{1}`), `lane_runtime_index_is_out_of_
subset_with_targeted_message`, `lane_divergent_gather_is_out_of_subset`. `vec_add` wired into
`vericl::suite!` at `N=4`: `tested` (bit-exact per-lane differential) + `proved` (3-obligation
line-granular bounds), width recorded in the claim config; suite green, evidence regenerated (only the
`vec_add` entry added — existing 16 entries byte-identical).

**V6 — survey-kernel generalization (dogfood).** The shortlist's already-provable f32 elementwise
`to_degrees_map` re-annotated at its real `Vector<f32, 4>` element type in the survey workspace
(`vericl-ecosystem-survey/annotated`, non-destructive: modified in place, generated/confirmed, then
restored byte-identical). `to_degrees_map_vec` carries the full pair — `tested` (per-lane differential,
`vector_width=4`) + `proved` (2-obligation bounds), `instantiate: N=4`; the scalar `to_degrees_map`
still proves. "Proves the scalar core" becomes "proves the vectorized kernel" for the elementwise class.

**Honest reach (design §0.5, §12, non-soundness).** Vector is the #1 *gate incidence* but only 13/148
items trip it ALONE (mostly framework impls). v1 generalizes the provable elementwise shortlist to its
true vector element type; the whole-kernel unlock (reductions/matmul launch sites) needs `View`/`Slice`
(#2 gap) + `Atomic` + `comptime!` + `match` — a documented, non-silent boundary. Deferred with targeted
rejections: cross-lane reductions, `SharedMemory<Vector>`, reinterpret-slice (`vector_size≠0`), vector
`cast_from`/`wrapping`, single-clause width sweep.

## Round-8 adversarial review (2026-07-23) — CLEAN

Verdict: Vector milestone merge-ready, no critical — fourth consecutive clean round
(3, 6, 7, 8). Surfaces held under injection: the is_modeled_int vector guard is
load-bearing at cast_int (cast_is_value_preserving would pass a Vector<u32,4> as a
scalar without it) and pinned by its direct predicate test; register-lane taint covers
read AND write through the branch write-log; the width is single-sourced across all
three faces (twin literal / Const<W> expand / folded launch) with factor-W mis-sizing
caught loudly lane-named; mixed vector/scalar-array kernels rejected by name and length
symbols opaque-in-units (no unit-confusion bound transfer); GT tiers discriminate
(injected comparison defect caught); compound-assign GPU sharing empirically confirmed;
empty() is zero-init on cubecl 0.10 + Metal so the twin's zero convention is
bit-faithful. Coverage nuances recorded, not defects: the two gather attack-shape tests
are saved by register-lane taint rather than the guard (only the predicate test
discriminates guard removal); powi/clamp/is_nan derive from GT'd primitives rather than
carrying dedicated GT rows (same posture as the scalar whitelist).

## Core `Slice` element milestone — S1–S6 DONE (2026-07-23)

Delivered core `Slice` support per `docs/design-view-slice.md` — the ecosystem survey's **#2 gate
(128/464 device items)**, whose tractable half is the core `Slice` (an *addressing view* over a base
buffer, distinct from the deferred `View`/`VirtualLayout` strided machinery). Headline: a `slice[i]`
access lowers to a checked `origin[offset + i]` (frontend-only, no buffer/metadata/id), so **the prover
change is ZERO** — B is a no-op; the work is the twin + macro gate. Six milestones, each with a negative
control (round-9 will attack this).

**S1 — prover confirmation (prover, no production change).** Reproduced every
`scratchpad/linevec/src/bin/sliceprove.rs` §5.2 verdict as `crates/vericl-ir/src/prover.rs::tests` (+10):
`to_slice`/dynamic-offset/const-offset(+origin-len)/**nested**(additive offset, §2.5)/**iteration**
(RangeLoop over `origin[offset+i]`, §2.2)/**gather-through-slice**(element assume transfers by origin-id
keying, §5.4) all `Proved`; `to_slice_oob`/const-offset-without-origin-len `Refuted`; gather-without-
assume `OutOfSubset`; and the reinterpret-slice (`with_vector_size`) `OutOfSubset` with the
`check_trivial_vectorization` message asserted **verbatim** (the standing risk-3 control). Confirms
"slices are transparent to the walker" on the unmodified prover.

**S2 — Rust-subslice twin + slice gate (macro).** `SliceRewriteFold` (`crates/vericl-macros/src/lib.rs`,
the `ShimRewriteFold`/vector-head-rewrite sibling) rewrites on the twin only: `x.slice(a,b)` →
`&(x)[(a)..(b)]`, `slice_mut` → `&mut …`, `to_slice()`/`to_slice_mut()` → `&(x)[..]`/`&mut (x)[..]`,
`for item in slice` → `for &item in …` (by-value iteration), nested slices compose relatively. A
`&Slice<F,ReadOnly>` helper param maps to `&[F]` (ArrayRef) and `&SliceMut<F>`/`&Slice<F,ReadWrite>` to
`&mut [F]` (ArrayMut) — `elem_of_slice`, reusing the Array twin representation (no new `ParamKind`, so
zero downstream match-arm churn — a reported refinement of the design's `SliceRef`/`SliceMut` variants).
Slices are helper-only, never launch args (kernel-path targeted reject). The bare `Slice` type ident
stays in `BANNED_IDENTS` (a v1 slice body never contains it), alongside the newly-banned `View`/
`VirtualLayout`/`Coordinates`/`StridedLayout`/`LinearView`/`VirtualView`/`ViewOperations` idents.
Tests (+8 macro): the four creations, nested, dynamic offset, for-loop by-value, the `&x.slice(a,b)`
redundant-outer-ref collapse (the idiomatic `&Slice` helper-arg form), param classification, laundering
rejects, and an end-to-end `expand` reject.

**S3 — aliasing + laundering (macro).** `slice_mut` → `&mut x[a..b]` makes **the borrow checker the
aliasing oracle**: sequential mutable slices compile (the dominant shape), two simultaneously-live
overlapping `&mut` subslices of one origin do **not** (`E0499`). Laundering (`as_mut_unchecked`/
`downcast`/`downcast_unchecked`/`try_cast_slice`) and reinterpret (`with_vector_size`/`into_vectorized`)
methods are rejected with targeted messages. `unused_mut` added to the twin allow-list so the sequential
mutable-slice twin is warning-clean. Negative control: `scratchpad/slicemut` (overlap `E0499` vs
sequential compiles), `scratchpad/viewslice/slicemut_aliasing_out.txt`.

**S4/S5 — public examples + suite (examples).** Three clean-room kernels wired into `vericl::suite!`
(`crates/vericl-examples`), each carrying `tested` (bit-exact, `max_ulp=0`) + `proved`: `windowed_slice_sum`
(dynamic-offset creation + `for v in slice` iteration + length, `Proved{2}`), `slice_gather_copy`
(gather through a `to_slice()` of an element-assumed table, the assume transfers free, `Proved{3}`),
`windowed_helper_kernel` (the dominant composition — a `#[vericl::helper] fn window_edge_sum(w: &Slice<F>)`
called `&x.slice(a,b)`, `Proved{3}`). Lib tests (+7) pin the twins value-for-value, the prover obligation
counts, and the no-relationship backstop. Evidence regenerated: only the 3 new entries added, existing 17
byte-identical (cpu-first, default-last — committed wgpu-only format).

**S6 — round-9 risk-1 twin-validity control.** cubecl does NOT bounds-check slice creation; Rust's
`&arr[a..b]` panics, so the twin is the slice-creation validity oracle (§4.4). Test
`windowed_slice_creation_panics_when_x_undersized`: with `x.len() == y.len()` (violating the
`y.len()+4<=x.len()` contract) the `tested` twin **panics** on the out-of-range `&x[pos..pos+4]` while the
`proved` claim (over the *accesses*, `x[pos..pos+4)`) still holds — the two claims' scopes are documented
so neither over-claims. The prover side is `slice_dyn_offset_proves` (proves the guarded access though the
created slice may exceed the origin).

**Survey-count correction + honest coverage.** `docs/dogfood-2026-07.md` (new addendum) + README (new
"Core `Slice`" section) record that the "128" is really **~25 real core-slice creators + a `ReadOnly`/
`ReadWrite`-ident tail + the deferred `View` machinery** (`scratchpad/viewslice/split.py`): core `Slice`
is **necessary but rarely sufficient** — only ~10 of the ~25 trip no other gate, all impls/traits/test-
launchers, not 1-D launch kernels. Post-`Slice` frontier ranking: `plane_*` → custom cube structs
(`CubeType`-arg) → 2-D topology → `Tensor`/`View`.

**Ecosystem validation (non-destructive).** `slice_window_read` — the `x.slice(start,end)[j]` addressing
view the cubek attention (`reader/query.rs`) and convolution (`async_copy.rs`) readers build over their
inputs, isolated to a 1-D glue kernel (its real callers are co-gated by `View`/`downcast`/2-D/matmul) —
annotated in `vericl-ecosystem-survey/annotated`, confirmed end-to-end (`tested` wgpu + cpu, `proved`
`{2}`, clean append), then the survey workspace **restored byte-identical** (the V6 convention).

**Deferred with targeted rejections:** the `View`/`VirtualLayout`/`Coordinates` strided machinery,
reinterpret-slice (`vector_size≠0`, also unrunnable on wgpu), `Tensor` slice origins, `SharedMemory`-slice
race-freedom (a cooperative-slice milestone), and `split_at_mut`-recognized disjoint mutable slicing (v1.1).

**Counts:** vericl-ir 104 (+10), vericl-macros 59 (+8), vericl-examples lib 88 (+7). Clippy 0 (default +
cpu). conform demo-defects exit 0. Full `cargo test --workspace` green; evidence append-only, existing
entries byte-identical.

## Round-9 adversarial review (2026-07-24) — CLEAN

Verdict: core `Slice` milestone merge-ready, no critical — **fifth consecutive clean round**
(3, 6, 7, 8, 9). The four round-9 findings are **coverage/doc items, not soundness defects**: no
attack produced a false `Proved`, a silent accept, or a wrong twin. The slice design's central
conservatism held under injection — a slice access is *indistinguishable in the IR* from a hand-written
`origin[offset+i]`, so it rides the existing origin obligation with zero prover change, and every place
that could have leaked the addressing view into a bound instead keys off the origin's own
`Metadata::Length` leaf (never the derived `slice.len() = end−start`, never `BufferLength`).

Surfaces held under injection / discriminate-by-injection evidence:
- **Write lane is load-bearing, not decorative.** The new `slice_mut` write obligation
  (`slice_mut_write_proves`) discharges through the same walker as a read; its negative twin
  (`slice_mut_write_unguarded_refutes`) shows an unguarded `slice_mut` write **Refutes** — removing the
  guard flips the verdict, so the write bound is not vacuously satisfied. The example
  `slice_scale_inplace` carries the same `Proved{2}` end-to-end (write + its read) and bit-exact
  differential on both lanes.
- **Creation-validity conservatism (risk-1) is deliberate and documented, not a gap.** cubecl does NOT
  bounds-check slice *creation*; the prover proves only the guarded *accesses* (`offset+i < origin_len`)
  and leaves `end ≤ origin_len` to the twin, which **panics** on an out-of-range `&arr[a..b]`
  (`windowed_slice_creation_panics_when_x_undersized`, `slice_dyn_offset_proves` on the prover side). The
  two claims' scopes are split on purpose — `tested` catches invalid creation, `proved` covers accesses —
  so neither over-claims. Attacking "create out of bounds but guard each access" yields the honest
  answer (twin panic + accesses proved), never a false whole-slice `Proved`.
- **Mutable-aliasing oracle is the borrow checker, and the pair discriminates.** Sequential mutable
  slices compile (`sequential_slice_mut_scale` twin, committed); two simultaneously-live *overlapping*
  `slice_mut` views of one origin fail `E0499` (`scratchpad/slicemut/overlap.rs`, reverified this round).
  The scratch pair (sequential compiles / overlap `E0499`) is the discriminating control.
- **Reinterpret gate + element-assume-through-slice unchanged and still sound.** The `with_vector_size`
  `OutOfSubset` message stands verbatim (risk-3 control); the gather element assume transfers through a
  `to_slice()` by origin-id keying (`Proved{3}`) and drops to `OutOfSubset` without it (never a false
  `Proved`).

Findings closed this round (F1 the only substantive one; F2–F4 doc/coverage hygiene):

- **F1 — mutable-write path now has end-to-end coverage.** Was: every committed slice example *read*;
  the `slice_mut` write lane had a twin rewrite and a design but zero committed exercise. Added
  `slice_scale_inplace` (`crates/vericl-examples`), a per-thread disjoint-window in-place scale through
  `y.slice_mut(ABSOLUTE_POS, ABSOLUTE_POS+1)`, suite-wired carrying `tested`(bit-exact) + `proved`{2}
  (the one new evidence entry; existing entries byte-identical, cpu-first/default-last). A wider
  `slice_mut(a,b)[j]` multi-element window with the sequential-two-slice_mut aliasing convention is
  `sequential_slice_mut_scale` + its twin test (not suite-wired: single-threaded fixed-8 layout, and a
  disjoint wide window is unprovable — the `start = i*W` stride's `checked_mul` overflow side-obligation
  is unbounded → `OutOfSubset` — while an overlapping wide window is a write-order-dependent race, so the
  suite differential stays one element wide). Prover unit twin of the write lowering:
  `slice_mut_write_proves`/`_unguarded_refutes` (`crates/vericl-ir`). Overlapping-rejected control:
  `scratchpad/slicemut/overlap.rs` (`E0499`), referenced from the example/test docs.
- **F2 — docs state the as-built aliasing behavior honestly.** `docs/design-view-slice.md` §4.3, §8.3
  (the two mutable-slice rows), and §11 S3 promised a *macro-authored targeted message* for
  overlapping/disjoint mutable-slice aliasing; the implementation deliberately ships rustc's
  `E0499`/`E0502` as the oracle (the macro detection was never built). Preserved the design text and
  appended `[as-built]` annotations recording that the borrow checker IS the (sound) rejection and the
  prettified message is deferred; §8.4 now lists the prettification as future work. README aliasing
  sentence tightened to name `E0499`/`E0502` explicitly (dogfood doc was already honest).
- **F3 — the `View`/`Layout` ban list now covers the whole cubecl-std export surface.** Swept the
  vendored source (`vericl-ecosystem-survey/cubecl/crates/cubecl-std`, `tensor/view/` + `tensor/layout/`,
  2026-07-23); extended `BANNED_IDENTS` from 7 view idents to 28, so a kernel naming any layout/view type
  gets the **targeted** out-of-subset message instead of loud-failing deeper (opaque twin error / the
  reinterpret gate). Makes the README/design "rejected with targeted errors" claim true for the surface,
  not just `View`. Final list below. Test: `view_layout_export_surface_is_banned` (samples 27 idents).
- **F4 — `SliceRewriteFold`'s name+arity keying is flagged as latent fragility with a required guard.**
  The fold recognizes a slice creator by **method name + arity**, never receiver type (a proc-macro has
  no type info). Sound TODAY only because no View-like type with a same-named `.slice(a,b)`/`.to_slice()`
  method (different strided semantics) can reach a compilable twin — the whole `View`/`Layout` surface is
  a banned ident (F3). **REQUIRED WORK: any future `View` milestone that un-bans a `View`-like type MUST
  add a receiver-type guard to `SliceRewriteFold` first** (fire the rewrite only for genuine core-`Slice`
  receivers, mirroring the vector gate); otherwise `view.slice(a,b)` is silently mis-rewritten to a 1-D
  `&view[a..b]`. Recorded impossible-to-miss: the fold's own doc comment
  (`crates/vericl-macros/src/lib.rs`), `docs/design-view-slice.md` §8.4, and this line.

**F3 final ban list (28 view/layout idents).** Existing 7 kept; 21 added. Scope = the public
**type/trait** surface an author could name in a kernel body, from cubecl-std 0.10's `tensor/view/` +
`tensor/layout/`. Views: `View`, `VirtualView`, `VirtualViewMut`, `LinearView`, `VirtualTensor`,
`ConcreteLayout`, `ViewOperations`, `ViewOperationsMut`. As-view creation traits: `AsView`, `AsViewMut`,
`AsTensorView`, `AsTensorViewMut`. Layouts: `Layout`, `VirtualLayout`, `StridedLayout`, `SliceLayout`,
`PermutedLayout`, `PlainLayout`, `SimpleLayout`, `FixedDimLayout`, `MatrixBatchLayout`, `LinearLayout`,
`LinearViewLayout`, `Chain`, `Coordinates`, `IntoDyn`, `IntoDynLayout`, `IntoDyn2Layout`. Deliberately
**excluded** (documented in the ban-list comment): the codegen/launch plumbing (`*Expand`,
`*CompilationArg`, `*Launch`, `*Arg` — macro-generated, never hand-written), the
`Coords{1..5}{d,i}`/`CoordsDyn` tuple aliases (bare tuples, covered by the `Coordinates` trait ban), and
the `Sealed` marker (too generic to ban safely).

**Counts:** vericl-ir 106 (+2: `slice_mut_write_proves`/`_unguarded_refutes`), vericl-macros 60 (+1:
`view_layout_export_surface_is_banned`), vericl-examples lib 91 (+3: `slice_scale_inplace` twin+proof,
`sequential_slice_mut_scale` twin). Clippy 0 (default + cpu). conform demo-defects 0. Full
`cargo test --workspace` green (both feature sets). Evidence: one new entry (`slice_scale_inplace`),
existing entries byte-identical, other two evidence files untouched (cpu-first, default-last —
committed wgpu-only format).

## Round-9 adversarial review (2026-07-23) — CLEAN, + F1-F4 closed (2026-07-24)

Verdict: core Slice milestone sound and merge-ready, no false-Proved / silent-wrong-twin —
fifth consecutive clean round (3, 6, 7, 8, 9). Discrimination proven by injection: a
twin-only to_slice off-by-one was caught by BOTH the hand-computed test and the wgpu
differential. The creation-validity seam is conservative and honest: whenever the twin's
subslice creation does not panic, every access is provably in-origin-bounds — no shape
exists where the twin silently accepts what the GPU would overrun (the twin over-rejects
loudly, never under-rejects). Aliasing oracle: overlapping slice_mut = E0499, origin-write
conflicts = E0502, laundering banned by name, the &&[f32] intermediate case deref-coerces
silent-CORRECT. Rewrite shape coverage held (paren/chain/nested/stored-variable forms);
name-only .slice rewrite sound today solely because cubecl-std View cannot reach a
compilable twin. Findings closed post-review: F1 mutable-write end-to-end coverage
(slice_scale_inplace suite kernel + twin/prover/sequential tests; overlapping compile-fail
demonstrated in scratch), F2 docs annotated [as-built] (borrow checker IS the aliasing
rejection; prettified messages future work), F3 View/layout ban list extended to the full
cubecl-std export surface, F4 the receiver-blind .slice rewrite recorded as REQUIRED WORK
(receiver-type guard) in any future View milestone.

## Assurance ladder Rung A — counterexample validation DONE; certificates DEFERRED (2026-07-24)

Rung A attacks the largest opaque component in the trusted base — the subprocess SMT solver.
Two halves: (1) validate every `sat`-derived refutation in Rust (cheap dual); (2) independently
checkable proof certificates for `unsat` (moves the solver out of trust for `Proved` too).
**(1) shipped and unconditional; (2) blocked on tooling availability, documented not faked** —
the sanctioned outcome. Full decision record: `docs/certificates-decision.md`.

**Counterexample validation (SHIPPED).** Every `Refuted`-producing `check-sat` — `check_obligation`
(bounds) and `check_race` (two-thread data race) — now re-checks the solver's model before the
verdict is reported (`crates/vericl-ir/src/prover.rs`, `Prover::validate_counterexample`). Mechanism:
a vericl-side mirror of z3's assertion stack (`Prover::asserts`), kept exactly parallel by routing
all 13 push / 13 pop / 33 assert sites through the new `s_push`/`s_pop`/`s_assert` wrappers; at a
`sat` point its flatten IS the live assertion set (negated obligation + path conditions + assumes +
leaf type-range facts). The model is read back with `get_value` and each live assertion is evaluated
by a small **total interpreter** over the exact emitted SMT-LIB subset (`eval_sexpr`/`eval_op`,
free functions): int `+`/`-`/`*`/`div`/`mod` (Euclidean via i128 `checked_div_euclid`/`checked_rem_
euclid`), `ite`, `<`/`<=`/`>`/`>=`/`=`, `and`/`or`/`not`, literals + declared consts. A model that
fails any live assertion fails **closed** to `SolverError` — never a silent/spurious `Refuted`
(documented invariant on `ProveResult::Refuted`). For a refutation the solver's `sat` verdict thereby
leaves the trusted base; what remains is the ~120-line auditable Rust interpreter + vericl's encoding.

**Tests (5 new, `prover::tests`):** `genuine_refutation_passes_counterexample_validation` (end-to-end
positive — a real OOB refutes and survives validation, not flipped to SolverError);
`cex_interp_arithmetic_and_comparisons`; `cex_interp_flags_a_model_that_violates_an_assertion` (the
synthetic invalid-model negative — a fabricated model with `idx == len` is flagged `false`);
`cex_interp_div_mod_ite_and_negatives` (Euclidean div/mod, unary negate, ite, div-by-zero guard);
`cex_interp_rejects_unbound_and_unsupported` (unbound const + `bvand` → Err, fail-closed). All 106
pre-existing prover tests pass unchanged (every existing Refuted test now routes through validation).

**Verification.** Zero capability loss: vericl-examples lib 91/91, every `Proved` obligation count
unchanged. Zero regressions: vericl-ir 111, vericl 36, vericl-macros 60, conformance (wgpu, incl.
evidence check) 1, cooperative 6, cooperative_fallback 1, vector_conformance 2, float_method_whitelist
2, f64_wgpu_unsound 1, conformance_f64 (cpu) 1, float_method_whitelist_f64 (cpu) 2. Demo-defects: exit
0, all defects caught, every refutation now printed "validated in-checker" with byte-identical
counterexamples (`elem5=8`, race `t1=1,t2=0`, etc. unchanged). Clippy both feature sets: zero.
**Evidence byte-identical** — validation touches only `Refuted` verdicts, which produce no evidence
entry, and adds no field to any `Proved`-claim config; the conformance evidence check passed without
`VERICL_UPDATE`. Solver-time impact: **zero** added solver work on the `Proved` path (validation runs
only on `sat`); on a `Refuted`, one extra `get_value` + µs-scale Rust eval, immeasurable against the
~7-8 ms z3 spawn baseline (measured).

**Certificates (DEFERRED — blocked, verified 2026-07-24).** Design prior: cvc5 + Alethe + Carcara, an
OPTIONAL `certify: true` suite lane (re-solve each `unsat` in a fresh non-incremental cvc5 context with
proof production, check the Alethe cert with Carcara, fail closed). Blockers at pinned versions:
**cvc5 not available** (not on PATH; no Homebrew formula — `brew search cvc5` → only `cc65`);
**Carcara not on crates.io** (`index.crates.io/ca/rc/carcara` → NoSuchKey; git-only, no stable library
API); **z3's own proof format has no independent QF_LIA checker** (fails the whole point). With neither
tool present the required round-trip + corrupted-certificate tests cannot be written or run — shipping
the plumbing would be the half-checked lane the task forbids. z3 stays trusted for `Proved` claims,
recorded honestly. Path forward enumerated in `docs/certificates-decision.md`.

## Assurance ladder Rung B — IR reference interpreter + fuzz cross-check DONE (2026-07-24)

Rung B attacks the deepest remaining trust gap: **model fidelity** — does VeriCL's model of what a
CubeCL IR instruction *means* match what CubeCL actually executes? Everything the prover proves and
the identity hash covers is stated over the IR, so a wrong wrapping rule / off-by-one bound / misread
div-mod would let the prover certify a property the hardware violates with no test catching it. The
deliverable is a **third, independent implementation** of the modeled semantics that turns "trust the
IR model" into an empirically cross-checked, continuously-run property. Full standalone writeup:
`docs/interpreter.md`; README trusted-base note updated (the model-fidelity risk now has an empirical
cross-check).

**Architecture choice.** New module `crates/vericl-ir/src/interp.rs` (not a new crate). Rationale:
the interpreter consumes the same `KernelDefinition` (a cubecl type) the prover consumes, so it must
live in vericl-ir — the sole cubecl-facing library crate (the locked "isolate all IR-facing code in
one crate" decision). Independence from the other two implementations is the point and is structural:
the macro **twin** rewrites source *tokens* (vericl-macros, never sees IR); the **prover** encodes the
IR *symbolically* into QF_LIA + z3 (prover.rs, never runs the kernel); the **interpreter** *executes*
the IR concretely over real inputs (interp.rs, separate walk, no shared code with the prover's
`value_of`/encoding). `interpret_dispatch(def, inputs) -> Outcome` runs `AbsolutePos = 0..num_threads`
sequentially (threads independent in the non-cooperative subset, exactly as the twin loops), applying
writes to the buffers.

**Subset covered (v0, non-cooperative scalar 1-D).** Real finite-width semantics — integers wrap
exactly (`wrapping_add/sub/mul`, width-masked shifts, truncated div/mod), floats are IEEE-754
`f32`/`f64` (separate `Mul`+`Add` stay strict, explicit `Fma` fuses) — which is what makes it
bit-exact with the twin and a faithful GPU stand-in. Covers arithmetic (incl. `MulHi` and float
transcendentals), comparisons, `And`/`Or`/`Not`, bitwise (`&`/`|`/`^`/`<<`/`>>`/`!`/`count_ones`/…),
`Cast`/`Reinterpret`/`Select`, `Metadata::Length`, checked+unchecked `Index`/`IndexAssign` (bounds
**reported**, never panicked), `If`/`IfElse`/`Switch`/`RangeLoop`/bare `Loop` (instruction-budget
guarded), 1-D topology builtins, local + const arrays. **Excluded and reported `Unsupported` (never
guessed):** all cooperative constructs (`SharedArray`/`Shared`/`sync_cube` — a faithful cooperative
interpreter needs a lock-step multi-thread phase model; scoped out honestly as future work), atomics,
plane/warp, cooperative-matrix, TMA, tensor metadata, `Vector<_, N>` indexing, stepped/descending
loops. `interp.rs` has 11 unit tests over **real `#[cube]`-expanded IR** (via the KernelBuilder recipe
from docs/ir-research.md §1) per construct — axpy, off-by-one (OOB *reported*, not panicked), xorshift,
mix (wrap-on-overflow), div/mod decode, gather + out-of-range-offset OOB, switch, forward read, range
loop, and cooperative-rejection.

**Cross-check harness (deliverable 2).** `crates/vericl-examples/tests/interp_crosscheck.rs`, 15 tests:
the interpreter runs over each honest kernel's real `kernel_definition()` and is compared **bit-for-bit**
against the macro twin over many random inputs × 9 sizes — `axpy`, `xorshift_step`, `mix_u32`,
`flatten_decode_scale`, `gather_copy`, `select_mode`, `offset_window`, `fir3`, `gain_kernel`, `lcg_map`,
`comptime_shift`, `mul_hi_map`, `unit_interval_map` (mixed u32→f32), plus a guard-boundary case
(threads ≫ len). One kernel, `xorshift_step` (exact integer, no fma ambiguity), is checked **three-way**
— interpreter vs twin vs a live wgpu/Metal launch on 1027 elements — all bit-identical. Float kernels
compare interp↔twin bit-exact (both strict-rounding); interp≈GPU follows transitively (twin≈GPU is the
conformance suite's job, at the tolerance it records — the fma-contraction reason the float three-way
isn't asserted bit-exact).

**Fuzz lane (deliverable 3).** `crates/vericl-ir/src/fuzz.rs`. A seeded (dependency-free SplitMix64)
generator produces random in-subset kernels from a small grammar over the modeled constructs (guarded
indexing, arithmetic/bitwise chains, div/mod by a nonzero divisor, `if`/`if-else`/`switch`, bounded
`for` loops with a carried accumulator, gathers through a valid offset table), across 7 shape templates
(5 safe → expected `Proved`, 2 defective → expected `Refuted`). Each kernel is realized **two ways from
the same AST**: lowered to hand-built IR (the same instruction shapes `#[cube]` emits — `Index`/
`IndexAssign`, `Arithmetic`/`Bitwise`/`Comparison`, `Operator::And`, `Branch::{If,Switch,RangeLoop}`,
`Metadata::Length`) and evaluated directly by an independent tree-walking reference that never touches
the IR. Two cross-checks per kernel: **(a)** reference ≡ interpreter on valid + adversarial random
inputs (same output or same OOB); **(b)** prover verdict ⇄ interpreter — a `Proved` kernel probed
exhaustively over assume-satisfying inputs must never OOB (an OOB = CRITICAL model-fidelity finding),
a `Refuted` kernel's counterexample (or the shape's minimal witness) replayed must exhibit the OOB.
Any disagreement is a `Finding` with full reproduction, never silently reconciled. Deterministic subset
(400 kernels / 4,800 agreement inputs, prover on) is wired into `cargo test` (`deterministic_corpus_
has_no_critical_findings`, ~2.8s) plus a prover-free 2,000-kernel agreement test; the **full corpus
(20,000 kernels, 320,000 agreement inputs, 14,285 Proved / 5,715 Refuted, prover on) runs behind
`VERICL_FUZZ=1` and produced ZERO findings in 215.0s** (`#[ignore]`d `full_corpus`, release).
As-built lowering fidelity note: the fuzz IR *mirrors* cubecl shapes but is hand-built (random
structure can't be macro-expanded at runtime) — its primary value is soundness at scale
(prover↔interpreter, reference↔interpreter); *genuine*-cubecl-IR fidelity is anchored by cross-check 2
and the interp unit tests, both over real `#[cube]` expansion.

**Two real bugs the cross-check caught during construction (fixed, not patched around).** (1)
`KernelBuilder`'s `create_local_mut` *pools and reuses* a mutable local once the returned
`ManagedVariable` `Rc` is dropped — which the hand-lowering did immediately by dereffing to a plain
`Variable` — so every generated local collapsed to `local(0)`/`local(1)`, clobbering intermediates
(index resolved to the stored value → phantom OOB). The reference↔interpreter check flagged it at once;
fixed by using `create_local_restricted` (always a fresh `LocalMut`, never pooled). (2) The
Refuted-replay allocated `vec![0u32; len]` from the solver's counterexample, but z3 may assign a `len`
anywhere in the u32 range → a ~16 GB allocation → SIGKILL. Fixed by capping materialized lengths and
falling back to the shape's minimal OOB witness. Both are harness/lowering bugs, not vericl-model
findings — but they demonstrate the cross-check's discriminating power on first contact.

**Negative controls (deliberate semantics-bug injection; run + confirmed, not committed).** Three bugs
injected into `interp.rs`, each caught immediately, then reverted: (i) integer `Add`→`wrapping_sub` —
caught by the fuzz `InterpVsReference` check (and, tellingly, the *float* axpy public test stayed green,
correctly, since axpy's `+` is a float op on a different code path — evidence the fuzz lane covers the
integer arithmetic the float example kernels don't); (ii) float `Add`→`Sub` — caught by the public
`interp_axpy_matches_twin` bit-exact check; (iii) a bounds check that clamps instead of reporting (hides
OOB) — caught by both the `gather_out_of_range_offset_reports_oob` unit test and the fuzz
`RefutedButNoOob` path.

**Findings.** NONE. Across 20,000 fuzz kernels (both prover verdicts, 320k inputs), 13 public example
kernels bit-exact vs the twin, and a live GPU three-way, no interpreter/twin/prover disagreement
surfaced in the modeled subset. The task flagged this as the work most likely to surface a real
model-fidelity discrepancy; it did not — an honest, corroborating null result (the injected-bug
negatives establish the checks *would* have caught one).

**Verification (zero regressions).** vericl-ir 124 lib tests (111 pre-existing + 11 interp + 2 fuzz;
`full_corpus` `#[ignore]`d), vericl 36, vericl-macros 60, vericl-examples lib 91 unchanged, conformance
(wgpu, incl. evidence check — verified WITHOUT `VERICL_UPDATE`, **evidence byte-identical**) 1,
interp_crosscheck 15, plus cooperative/f64/whitelist/vector/shim suites all green; cpu-lane conformance
+ interp_crosscheck green under `--features cpu`. `cargo clippy --workspace --all-targets` zero warnings
on both feature sets. `conform demo-defects` exits 0, output unchanged. New files:
`crates/vericl-ir/src/interp.rs`, `crates/vericl-ir/src/fuzz.rs`,
`crates/vericl-examples/tests/interp_crosscheck.rs`, `docs/interpreter.md`; `crates/vericl-ir/src/lib.rs`
gained the two `mod` + re-export lines. No prover/twin/evidence code touched — the interpreter only
*reads* `KernelDefinition`s the existing machinery already produces.

## CubeCL-0.11 upgrade drill — DONE 2026-07-24

Rehearsed the promised upgrade against **cubecl git main =
`0.11.0-pre.1` @ `870666bf46e1c370d3aae08e3dcb9d9a74ed90c8`** (no 0.11 on
crates.io yet — 0.10.0 still latest). Full write-up:
**`docs/upgrade-drill-2026-07.md`**. Disposable drill copy left in place at
`/Users/ryland/code/vericl-upgrade-drill` (not a git repo; `README.DRILL.md`
inside says it's deletable — `rm -rf` it any time). Canonical repo untouched;
`=0.10.0` pin intact.

**Isolation held — the headline structural result.** Only `vericl-ir` breaks.
`crates/vericl` (36 tests) and `crates/vericl-macros` (60 tests) pass on main
**unchanged** (neither depends on cubecl). So a 0.11 upgrade is a `vericl-ir`
porting job, not a workspace rewrite — the README "isolate all IR-facing code
in one crate" decision paid off exactly as intended.

**Toolchain finding:** cubecl `0.11.0-pre.1` bumped **MSRV to rustc ≥ 1.95**
(`cubecl-zspace`); drill machine had 1.94.0. Drilled with a locally-installed
`1.97.1` (default toolchain left unchanged). The real upgrade must bump
`rust-version` + README toolchain note.

**Breakage catalog (vericl-ir, 89 first-pass errors: prover 50, fuzz 23,
interp 17, hash 1):**
- `Scope` is now interior-mutable (`instructions`/`locals` are `RefCell<…>`);
  `const_arrays` gone; `create_local_restricted` gone; `AddressType::register`
  takes `&Scope`. Mechanical (`.borrow()`/`.get_mut()`), ~20 sites.
- **SSA value model rebuilt:** `Variable`→`Value`, `VariableKind`→`ValueKind`,
  and `ValueKind` collapsed to `Value{id}`|`Constant` — the rich kind enum
  (GlobalInputArray/LocalMut/Builtin/…) is GONE; semantics moved to ops
  (`ReadBuiltin`/`ReadScalar`/`global_args`). **Architectural** for the prover.
- **Memory model rebuilt:** `Operator::Index/IndexAssign/Unchecked*` removed →
  new `Operation::Memory(Memory::Index/Load/Store)` **pointer** model;
  `IndexOperands` now carries **`checked: bool`** + `unroll_factor` (the IR
  itself now models bounds-checking — directly in vericl's domain).
  **Architectural** for the prover's bounds analysis.
- `Metadata::Length`/`Rank` removed; `BufferLength.var`→`list`.
- `Arithmetic::Modulo` split into `Rem` (truncated) + `ModFloor` (floored);
  operand structs `BinaryOperator`/`UnaryOperator`→`BinaryOperands`/
  `UnaryOperands`; `KernelDefinition.buffers`→`Vec<BufferInfo>`,
  `scalars`→`Vec<ScalarInfo>`; launch `ArrayArg`→`BufferArg`;
  `ComputeClient<S,C>`→`ComputeClient<R>`.

**4a — axpy `ir_hash` changes: YES, guaranteed** (drift tripwire fires as
designed). `Scope::Hash` now also folds `locals`; body IR + `Value` + `buffers`
all restructured → the stored `sha256:3ae1a32f…` cannot match; every
`evidence/*.json` `ir_hash` goes stale on upgrade until re-verified. Identity
*mechanism* ports with **1 line** changed (RefCell `get_mut`) — proven to
compile (`f64probe/src/bin/hash_port.rs`). Bonus: main adds a **`TypeHash`**
schema-drift trait (ideal future tripwire) but its recursion overflows on
non-flat IR types (`Type` is self-recursive) — usable only on flat leaf kinds
today; worth an upstream report.

**4b — f64-on-wgpu silent corruption: STILL REPRODUCES on main (disclosure
headline).** Self-contained probe (`f64probe/src/bin/f64_wgpu.rs`, cubecl-only):
cpu lane bit-exact; wgpu lane silently diverges (elem0 got 100 vs 102.5, worst
err 2.5e3 at n=1027) with **no compile error, no panic, no naga/wgpu
diagnostic**. Not fixed, not rejected — same silent-corruption class as 0.10.
→ **Queued-disclosure checklist item 1 = DONE: "still present at `870666bf`
(0.11.0-pre.1); published 0.10 affected — consider advisory + a launch-time
f64-on-wgpu reject."** Item 2 (SPIR-V path) still open — drill covered the
naga/WGSL/Metal path only. **Still queued; do NOT contact anyone without
Ryland's explicit go.**

**4c — other findings on main:** eager `&&` — eager `Operator::And/Or/Not`
persist, but main **added short-circuit machinery + a `short_circuit` regression
test** for side-effecting RHS (finding partially addressed; guarded-array-read
case needs an IR re-probe during the real port). division/modulo — IR
`Modulo`→`Rem`+`ModFloor` (rounding clarified); naga div-by-zero runtime
fallback not re-probed (backend behavior). `terminate!()` host expansion /
`CUBE_COUNT` on cpu — live above the IR layer, not re-verifiable while the
prover is un-ported; deferred to the real upgrade's re-verification pass.

**Effort estimate to port vericl-ir to 0.11:** hash core ~0.5 h (proven);
hash harness ~1–2 h; interp ~1–2 days; fuzz ~1–2 days; **prover ~1–2 weeks +
full 7-round soundness re-review** (both its SSA-value and memory models
changed — architectural, characterized-not-rewritten per protocol); re-stamp
evidence ~1 h. Recommended playbook (order, re-verify list, dedicated review
round) in `docs/upgrade-drill-2026-07.md`.

## Productization gate — DONE 2026-07-24

The final roadmap item before first outside users. Ergonomics-first
(familiar workflows, actionable errors, zero new concepts) treated as a
first-class gate. Five deliverables; publishing itself deliberately NOT done
(gated on Ryland). No evidence touched (byte-identical); zero regressions.

**1 — Real user guide (`docs/guide.md`, 13 sections, ~750 lines).** The path
from "I have a CubeCL kernel" to "`cargo test` verifies evidence" for a
competent Rust/GPU dev who has never seen this repo: installation (z3 per-OS
table + the three-crate `Cargo.toml`, with *why* each of `vericl`/`vericl-ir`/
`cubecl` is needed), a first verified kernel (axpy) end to end, the contract
clauses built up (minimal `assumes`+`compare`, the four compare modes, `gen`,
`wrapping`), `instantiate(...)`, composition (`helper`+`uses`), cooperative
kernels, the `suite!` field reference, the `VERICL_UPDATE` workflow, reading an
evidence file (the four claim categories + identity/staleness + lane
independence, from the real `axpy` entry), a rejection reference (VeriCL's own
+ the two rustc-mediated cases + run-time panics, each with a fix), and an
honest "what VeriCL does not do". Linked prominently from `README.md` (callout
after the intro). **Every ```rust snippet is a verbatim excerpt of a kernel in
`crates/vericl-examples/src/lib.rs` or a `suite!` in `tests/` — all compiled by
the green `cargo test --workspace` run** (signatures + attributes verified
byte-identical against source); the only non-excerpt blocks are TOML, JSON
(from the real `evidence/vericl.json`), and `text` error output.

**2 — Error-message audit.** Swept every user-facing `syn::Error`/`panic!`/
`.expect` site across `vericl-macros/{lib,suite,coop}.rs` (114 sites cataloged
verbatim with span + verdict). The vast majority already passed the bar (name
the construct + say what to do + right span). Fixed the ones that failed
(before → after):

| Site | Before | After |
|---|---|---|
| lib.rs catch-all (highest-traffic rejection) | "`X` is outside the vericl v0 kernel subset; …rejected rather than silently approximated (see README "First release")" — no remediation | now lists the supported subset (affine indexing, bounded `for`/`match`, `&Array`/`Slice`, `instantiate`, `cooperative`) and points at the guide's rejection reference for `X` |
| lib.rs plain-param-name | "expected a plain parameter name" | adds: "destructuring/tuple/`_` patterns … are outside the subset; give each parameter a simple name (e.g. `x: &Array<f32>`)" |
| lib.rs unsupported param type | "unsupported parameter type in the vericl v0 subset" | now names what IS supported (scalar / `&Array` / core `&Slice` / `#[comptime]` scalar) |
| lib.rs kernel-returns-value | "kernels must not return a value" | adds: "…writes through an `&mut Array<T>` output, not a return value (a value-returning `#[cube]` fn is a `#[vericl::helper]`)" |
| lib.rs `#[vericl::reference]` no-args (A24) | span = `call_site()` | span = the offending argument tokens |
| suite.rs duplicate field (C5) | span = `call_site()` | span = the duplicate field's own value tokens |
| suite.rs evidence-write (E8) | bare `.expect("write vericl evidence manifest")` | actionable panic naming the path + "check the `evidence:` path is writable and its parent exists" |
| coop.rs shared compound-assign (D4) | span = `call_site()` | span = the offending `tile[i] OP= …` expression (captured the `ExprBinary` span in the visitor) |

Two BADSPAN cases (`check_instantiate_local_collisions` B8, cross-barrier local
D6) were left at the coarse span **deliberately**: both messages already name
the exact offending identifier (the actionable info), and threading spans
through their shared string-only collectors (`collect_locals` is also used by
`UsesRewriteFold`; `IdentRefCollector` by several passes) is disproportionate
regression risk for rare/advanced rejections — documented here as an accepted
pre-1.0 gap. The two **rustc-mediated** cases got doc-comment cross-references
where the error lands, not silent acceptance: E0499/E0502 overlapping-mutable-
slice (doc on `SliceRewriteFold`, naming the exact rustc codes) and the
missing-`#[vericl::kernel]`-annotation → "cannot find `<name>_vericl`"
resolution error (doc on the `suite!` module). Both are also in the guide's
rejection reference. Consistency note left as-is: coop.rs says "v1"/"v1.1"
subset where lib.rs says "v0" — intentional (cooperative is a later tier), not
churned.

**3 — API stability pass (core `vericl` crate).** The genuine public surface a
user touches is small: the four macros, `Compare`, and the evidence-reading
types (`Manifest`/`Entry`/`Claim`/`ClaimKind`/`ClaimResult`/`ContractRecord`/
`Identity`/`verify`) + the compare/ulp utilities + `SplitMix64`. Everything
else that is `pub` is there ONLY because macro-generated code references it
across the crate boundary at the user's call site — it cannot be `pub(crate)`
(that would break `suite!`), so it was marked `#[doc(hidden)]` instead
(compile-invisible, zero regression, reversible): the `trust`/`host_shims`/
`panic`/`shared`/`line` plumbing modules, `Line`, `SharedTile`,
`catch_reference_panic`, all the `*_config` claim builders +
`race_freedom_assumption_claim` + `RaceDependency` + the two `*_CHECK` consts,
`combine_source_hash`/`check_helper_composition_depth`/
`MAX_HELPER_COMPOSITION_DEPTH`/`StructuredAssume`, and the `compare_*_with`
dispatchers. `#![warn(missing_docs)]` added to the core crate and satisfied
(zero warnings) — every remaining visible type/field/variant/method now has a
doc comment (`Manifest`, `Entry` + fields, `Claim`/`ClaimResult` + variants,
`CompareReport`/`Mismatch` fields, `Compare` variant fields, `Identity`/
`ContractRecord` fields, `SplitMix64` methods, `Compare::describe`,
`Contract::record`). No compat break — the doc-hidden items stay `pub` and
callable; the pre-1.0 note in `docs/release-checklist.md` records that they are
plumbing, not API. A crate-root comment block documents the surface split.

**4 — Publication readiness (PREPARED, not published).** crates.io metadata on
all three published crates (`description` [already present], `license`,
`repository = https://github.com/Rylandl/vericl`, `keywords`, `categories`);
`vericl` sets `readme = "../../README.md"` (cargo copies it into the package —
verified in the dry-run tarball, 53 KB README present). **`vericl-ir` made
publishable** (removed `publish = false`) — the critical finding: the `suite!`
macro emits `::vericl_ir::` at the user's call site (even with `prove: false`,
for the IR identity hash), so it MUST be on crates.io or no external `suite!`
compiles. `vericl-examples` kept `publish = false`. `cargo publish --workspace
--dry-run --allow-dirty` passes for all three (packaged together via a local
temp registry so inter-crate deps resolve; `vericl` correctly resolves
`vericl-macros = "0.1.0"`, path stripped). `docs/release-checklist.md` records
the exact publish sequence (workspace command, or leaf-first order + the
expected standalone-dry-run chicken-and-egg), a from-crates.io smoke test, and
the decisions that stay Ryland's (whether/when to publish, crate-name
ownership, version choice, the `=0.10.0` cubecl pin's user-facing consequence,
yank policy).

**5 — This record.**

**Verification (all green, reported):** `cargo test --workspace` = 283 passed /
0 failed / 1 ignored, zero warnings; `cargo test -p vericl-examples --features
cpu` all green (incl. the f64-on-cpu suite); `cargo clippy --workspace
--all-targets` clean; `cargo clippy -p vericl-examples --all-targets --features
cpu` clean; `conform` demo-defects exit 0; `evidence/*.json` byte-identical
(git clean). Files changed: `Cargo.toml` (+repository), the three published
crates' `Cargo.toml` (metadata; `vericl-ir` un-`publish=false`d),
`vericl-macros/src/{lib,suite,coop}.rs` (messages/spans/doc cross-refs),
`vericl/src/{lib,compare,contract,evidence,rng}.rs` (doc-hide + docs +
missing_docs lint), `README.md` (guide callout). New: `docs/guide.md`,
`docs/release-checklist.md`.

### Roadmap state — the agreed sequencing is COMPLETE

With the productization gate done, the agreed roadmap (README first-release
outcomes + the 2026-07-22 roadmap + the foundation/Slice/interpreter rungs) is
delivered end to end, and VeriCL is ready for first outside users (publishing
gated on Ryland). Remaining **known** items, all previously recorded, none
blocking first users:

1. **Prover port for cubecl 0.11** when it ships (0.11.0-pre.1 drilled
   2026-07-24; `vericl`/`vericl-macros` unaffected, `vericl-ir` is the porting
   job — playbook in `docs/upgrade-drill-2026-07.md`; the real upgrade also
   bumps MSRV to rustc ≥ 1.95 per cubecl-zspace).
2. **The queued f64-on-wgpu silent-corruption disclosure** (still reproduces at
   0.11.0-pre.1; item 1 of the checklist done, SPIR-V path still open). **Do
   NOT contact anyone without Ryland's explicit go.**
3. **The post-`Slice` subset frontier:** `plane_*` reductions, then custom
   `CubeType`-struct arguments, then 2-D topology (then `Tensor` + the deferred
   `View`/`VirtualLayout` machinery) — each rejected explicitly today, not
   approximated.
4. **Proof certificates** (independently-checkable `unsat` proofs to move the z3
   binary out of the trusted base) when the tooling exists — cvc5 + Alethe +
   Carcara are not available at the pinned versions; decision record in
   `docs/certificates-decision.md`.

Smaller carried debts (unchanged): fold the cubecl crate version into
`Identity`; a standalone `vericl check` CLI (superseded by the `cargo test`
story for v0); a `FLOAT_METHOD_CONST_ONLY` distinction if a dogfooded kernel
needs a runtime `new`/`from_int`; a full QF_BV overflow model as a precision
upgrade; VeriCL-authored (buffer-named) diagnostics for the two rustc-mediated
rejections (E0499 aliasing, missing-annotation accessor).

## Coverage re-census (2026-07-24) — both surveys re-run against today's VeriCL

Both original surveys re-scored against today's gates, and — the part that
makes it a census rather than a paper exercise — **every item the gates said was
reachable was actually annotated and run**. Full reports in the two survey docs'
own re-census addenda. All work in the private sibling workspaces; the only
vericl-repo changes are those two addenda, this record, and three pre-existing
private-identifier leaks fixed in passing (below). No commits, no evidence
touched, `cargo test --workspace` green before and after.

**Private codebase (22 kernels): 2/22 → 6/22 faithful; 21/22 with a validated
artifact.** Faithful = body byte-for-byte the private source, only the contract
attribute added. Plus 3 at one-named-construct substitution (9/22 near-faithful),
12 distilled-core validated, 1 with no honest annotation at any fidelity, and
**0 annotatable-but-unattempted**. Three new evidence manifests, **513
machine-checked obligations**, all re-verified by a plain `cargo test`;
composition exercised **four levels deep** (60 obligations through the chain).
Every Tier-1 and Tier-2 gate in the original table is closed and blocks nothing.
The survey's own predicted "next wall" (`usize` scalar params) blocks **0/22** in
the predicted form — no launch kernel takes a runtime `usize` scalar; it
resurfaced as a `cast_from` *source-type* wall instead.

**Private residual walls (measured):** 1. bare `fma(...)` — 12/22, and cubecl's
`fma` is **not host-callable**, so the rejection is correct and a helper is no
escape; the `a*b+c` rewrite is measurably unsound for the shape that needs it
(fused residual 8.23e-7 → exactly 0). 2. `cast_from` with a `usize` source —
13/22, sole for 1. 3. `wrapping` on a tuple-returning helper — 3/22, sole for 2.
4. non-`f32`-target / `bool`-source `cast_from`. 5. **new**: implicit
*injectivity* of a table used as a write index — no assume form can express it.
6. 2-D dispatch. Walls 1+2 both live in one ~30-line device helper that is the
sole gate on 12 of 22; fixing both takes faithful coverage 6/22 → ~18/22.

**Two new private findings.** (a) An implicit host-side reach invariant on four
kernels: an unbounded anchor underflows, the wrapped value feeds an interleaving
`*2` whose `checked_mul` side-obligation cannot discharge, the index becomes
unmodelable → `OutOfSubset`. Adding the guard one production kernel already
carries flips all four to `Proved{55/60/61/64}`. **VeriCL residual: surface the
failed `checked_mul` side-obligation instead of the generic out-of-subset text** —
the same defect yields an actionable counterexample or a bare `OutOfSubset`
depending only on buffer size. (b) A differential FAILURE that is a genuine
implicit invariant: a table-loaded output slot collides for ~half of drawn
inputs (16/32 elements diverge, worst ~2.1e9 ULP); correct upstream only because
the table is injective.

**Three recorded walls measured as GONE**, each on real private code:
helper-calling-helper tuple destructuring (the `Pat::Tuple` residual is
**closed**), `#[comptime]` params in cooperative kernels (shared-memory wall #1
**lifted** — full oob+race triple), and "cannot bind a free scalar near the cube
count" (not a macro gate; the constraint is semantic).

**Ecosystem (464 items): denominator and all thirteen v0 counts reproduced
exactly, then re-scored.** Items with zero blocking gates 103 → 119 (+16) on the
v0-lineage gate set; 35 → 49 (+14) under a corrected set applied symmetrically.
Both are reported because the **v0 gate list under-counted blockers** — it never
looked for struct-typed comptime params, the broad `CubeType` parameter shape, or
`intrinsic!`, which is why it showed 103 gate-free items while the survey could
shortlist only 8. Genuinely new annotatable non-test plain `fn`s: **4**, all via
the `cast_from` shim.

**The re-census adds sole-blocker counts — the honest reach number the original
lacked** (incidence answers "how common", not "what would removing this
unlock"). 127 of 464 items have exactly one blocking gate.

**Two shipped gates measured to unlock nothing here.** `match`/Switch (v0 rank
#4, 119 incidences): of ~270 `match` expressions inside `#[cube]` items, **exactly
4 have integer-literal arms and all 4 are in cubecl's own conformance suite** —
zero in cubek's nine crates, zero in cubecl-std, zero in burn-cubecl
(independently confirmed by a second scan returning 0 hits). `comptime!{}`
blocks: all 12 lexically admissible items are blocked by something else, 9 of
them by a struct-typed `#[comptime]` param. Both milestones are correct and
useful; this corpus does not exercise them. Core `Slice` is the same story.

**Spot-validation: §4 residual #3 is CLOSED, measured.** Six kernels added
non-destructively to the survey crate (original 8 evidence entries byte-identical,
per-entry SHA-256 verified), all green on **both** lanes:
`to_unit_interval_open_map` (`max_ulp=3`, `Proved{2}`), `uniform_value_map`
(`abs=1e-4`, `Proved{5}` — composition + helper-level `wrapping` + the shim in
one kernel), `normal_box_muller_map` (`abs=1e-2`, `Proved{6}`),
`kernel_switch_simple` (`Proved{3}`), `slice_select` (`Proved{2}`), and
`bernoulli_value_map` cleanly **rejected** (`bool: CastToF32 is not satisfied`).
Findings: `to_unit_interval_open` is 1 ULP on wgpu (backend lowers `/8388609.0`
to a reciprocal-multiply; the sibling divides by an exact `2^24`) — the shim is
not implicated; **new residual: bool-source `cast_from`, 15 sites across cubek,
GPU ground truth already measured** (`true → 1.0`, `false → 0.0`, both lanes), fix
is one trait impl per target type; and the upstream slice tests are blocked by the
single-designated-thread `if UNIT_POS == 0` idiom, **not** by `Slice`.

### The post-re-census frontier ranking — the recorded one is overturned

The Slice milestone recorded (1) `plane_*`, (2) `CubeType`-arg, (3) 2-D, (4)
`Tensor`+`View`. Measured sole-blocker counts:

| Rank | Gate | Items | **Sole** |
|---:|---|---:|---:|
| 1 | struct-typed `#[comptime]` / `comptime_type!` | 243 | **38** |
| 2 | `View`/`Layout` machinery | 110 | **45** |
| 3 | custom `CubeType` struct args | 141 | 8 (the only sole bucket that is plain `fn`s) |
| 4 | cmma / `Matrix` | 62 | 6 |
| 5 | **`plane_*`** | 88 | **2** |
| 6 | **2-D topology** | 39 | **1** |
| 7 | `Tensor` 32 · reinterpret 18 · `select()` 9 · rejected methods 5 · `Atomic` 1 | | **0 each** |

`plane_*` — the recorded #1 next milestone — sole-blocks 2 items. 2-D sole-blocks
1. `Tensor` sole-blocks 0. Ranking by incidence was ranking by the wrong number.

### RECOMMENDED NEXT MILESTONE (with its measured justification)

**A shim-and-small-gate batch, before any large subset milestone.** Justification,
all measured this pass:

1. **`fma`/`mul_add` host shim** (GPU-ground-truthed, exactly as `cast_to_f32`
   and `mul_hi` were): unblocks **12 of the 22** private kernels. cubecl's `fma`
   is not host-callable, so this is a shim, not a whitelist entry, and the naive
   `a*b+c` substitution is measurably unsound for the shape that needs it.
2. **`CastToF32` source/target extension** — `usize` source (sole blocker for 1
   private kernel, part of the complex blocking 13) and `bool` source
   (**15 ecosystem sites, GPU ground truth already in hand**, and the only thing
   between VeriCL and the third of the three cubek-random distribution cores),
   plus non-`f32` targets.
3. **`wrapping` on a tuple-of-integers-returning helper** — sole blocker for 2
   private kernels.
4. **Prover diagnostic**: surface the undischarged `checked_mul` side-obligation
   instead of the generic `OutOfSubset`.

Together these take private faithful coverage from **6/22 to ~18-19/22** and close
the last ecosystem distribution core — for four small, well-understood, already-
ground-truthed changes. Every one of them was surfaced by running real code, not
by speculation.

**Then, for ecosystem reach: struct-typed `#[comptime]` params + the `CubeType`
struct-arg story** (243 items / 38 sole, and 141 / 8-all-plain-`fn`s) — one
underlying capability: letting a `#[cube]` item take a comptime-known
struct/enum. **`plane_*` should be de-prioritised** from its recorded #1 slot: 88
incidences but 2 sole-blockers.

Two further residuals recorded, not scheduled: a permutation/injectivity element
assume (the private write-index finding above), and accepting the 1-D
single-designated-thread `if UNIT_POS == 0` idiom outside `cooperative(...)`.

### Policy-gate fix (private-codebase policy, README)

Three pre-existing private-identifier leaks in the public repo were found by a
sweep and genericized in passing — one private type/field name in
`docs/dogfood-2026-07.md`, and three private kernel names in this file (build
record and composition-validation note). Wording only; no claim changed. A
full-repo sweep for the private project name and its kernel/file identifiers is
now clean.
