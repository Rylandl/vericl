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
   Private validation (per the README's Substrate policy: never committed, described here only
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
   the private dogfood suite (`mulhilo32_kernel`, `philox4x32_two_kernel`,
   `synth_freqshift_cw_kernel`) green end to end, `dogfood-rejects` still fails to build with
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
6. Next proved property: race-freedom via two-thread symbolic reduction (the sum_racy class
   proved, not just differentially caught).
7. Later: QF_BV wrapping model in the prover — needed for, among other things, the
   unbounded-overflow-feeding-div/mod gap found in the round-2 adversarial review (roadmap item 9
   below): a divisor provably nonzero in unbounded QF_LIA can still wrap to exactly zero via `u32`
   overflow (e.g. `a * b == 2^32`), which the current div/mod side-obligation does not model.
   Currently **known-inert on wgpu/Metal specifically** because naga's division-by-zero fallback
   is dividend-preserving rather than trapping (`a / 0 == a`, `a % 0 == 0` — confirmed empirically,
   see README "CubeCL semantics findings"), so the resulting index is wrong but not itself a crash
   on today's one supported backend; not something to rely on in general, and the first concrete
   motivation for this item rather than a purely speculative one. Also: fold cubecl version into
   Identity; upstream conversation with tracel-ai; standalone `vericl check` CLI (README CI story
   row); array-value-dependent indices (offset tables / gather) via quantified assumes
   (docs/dogfood-2026-07.md Tier-2 gap #3, still open); a `FLOAT_METHOD_CONST_ONLY` distinction
   if a dogfooded kernel needs a runtime `new`/`from_int`; an `f64` instantiation tier if a
   dogfooded kernel ever needs one (see roadmap item 8's "one concrete type per helper" note —
   today only `f32` is supported anywhere in vericl v0, so this is purely hypothetical debt, not
   a known gap).
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

   **Private dogfood validation** (per the README's Substrate policy: never committed, described
   here only by construct class): the survey's own "inner-loop-with-single-helper shape"
   candidate — `fir_conv_two_chain` (a `#[cube]` device fn Substrate's own source already calls
   "the first proof that #[cube] Level-1 composition works") and `inner_loop_kernel` (the
   `#[cube(launch)]` entry point that calls it exactly once) from `substrate-kernels/src/lib.rs` —
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
   surfaced that `instantiate_subset.rs`'s `synth_freqshift_cw_kernel_bounds_proof_is_out_of_subset_div_mod`
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
   repro crate (path-deps on this repo, never committed here per the Substrate-scratch precedent)
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
