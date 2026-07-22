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
5. Next proved property: race-freedom via two-thread symbolic reduction (the sum_racy class
   proved, not just differentially caught).
6. Later: QF_BV wrapping model in the prover; fold cubecl version into Identity; upstream
   conversation with tracel-ai; standalone `vericl check` CLI (README CI story row); prover
   div/mod-index modeling (unblocks proving kernels like the private `synth_freqshift_cw`
   validation above); kernel composition (roadmap item 3 per docs/dogfood-2026-07.md); a
   `FLOAT_METHOD_CONST_ONLY` distinction if a dogfooded kernel needs a runtime `new`/`from_int`.
