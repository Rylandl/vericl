# CubeCL upgrade drill — 2026-07-24

Rehearsal of the "cubecl-0.11 upgrade" that `tasks/todo.md` and the README's
health-check story call for. No `0.11` exists on crates.io yet (0.10.0 is the
latest, verified today), so the drill repoints CubeCL to **git main** — the
best available proxy for what the next release breaks.

## What was drilled, where

- **Drilled commit:** `github.com/tracel-ai/cubecl` @
  `870666bf46e1c370d3aae08e3dcb9d9a74ed90c8` (HEAD on 2026-07-24). `cargo`
  resolves it as **`0.11.0-pre.1`** — i.e. this literally is the in-progress
  next release, not an arbitrary main snapshot.
- **Isolated copy:** `/Users/ryland/code/vericl-upgrade-drill` (a sibling, NOT
  a git repo, disposable — see its `README.DRILL.md`; delete with
  `rm -rf /Users/ryland/code/vericl-upgrade-drill`). It is an `rsync` of the
  workspace source (excluding `target/` and `.git/`) with the workspace
  `Cargo.toml` cubecl dep repointed to the git rev above, plus one throwaway
  probe crate `crates/f64probe` that depends only on cubecl (never on vericl).
- **The canonical repo `/Users/ryland/code/vericl` was not touched** beyond
  adding this doc and the `tasks/todo.md` record. Its `=0.10.0` pin is intact.

## Headline results

| Question | Answer |
|---|---|
| Toolchain | **MSRV bumped: cubecl `0.11.0-pre.1` requires rustc ≥ 1.95** (`cubecl-zspace`). Drill machine had 1.94.0 → hard error. Drilled with a locally-installed `1.97.1` toolchain (`cargo +1.97.1`, default toolchain left unchanged). |
| vericl core (`crates/vericl`) | **Green, unchanged.** 36/36 tests pass on main. Has no cubecl dep — the evidence/compare layer is provably upgrade-independent. |
| vericl-macros (`crates/vericl-macros`) | **Green, unchanged.** 60/60 tests pass on main. Has no cubecl dep — the macro layer is provably cubecl-version-independent (a README promise, now demonstrated). |
| vericl-ir (`crates/vericl-ir`) | **89 first-pass compile errors** against main. The IR was substantially restructured (see catalog). This is the *only* crate that breaks — the isolation decision held. |
| axpy `ir_hash` (4a) | **Changes — guaranteed.** Drift tripwire fires exactly as designed. (Structural proof below; the identity core ports with 1 line changed.) |
| f64-on-wgpu silent corruption (4b) | **STILL REPRODUCES on main. Not fixed, not rejected — silent, no diagnostic.** Headline for the queued disclosure. |
| eager `&&` (4c) | Eager `Operator::And/Or/Not` persist; main **added short-circuit machinery + a `short_circuit` regression test** for side-effecting RHS. Partially addressed — vericl's guarded-array-read case needs re-verification. |
| division/modulo rounding (4c) | IR `Arithmetic::Modulo` **split into `Rem` + `ModFloor`**; naga's div-by-zero runtime fallback not re-probed (backend behavior, unlikely to have moved). |

## The isolation architecture held

The whole point of "isolate all IR-facing code in one crate" is that a CubeCL
upgrade's blast radius is bounded. Measured on main:

- `crates/vericl` (core) and `crates/vericl-macros`: **zero changes, all tests
  green.** Neither depends on cubecl.
- `crates/vericl-ir`: the sole crate that breaks. All 89 errors live here.
- `crates/vericl-examples`: breaks only transitively (it depends on
  `vericl-ir`); its own `#[cube(...)]` kernel bodies use the stable frontend.

So an upgrade is a `vericl-ir` porting project, not a workspace-wide rewrite.

## Breakage catalog (vericl-ir vs cubecl 0.11.0-pre.1)

First-pass error distribution (compiler stops at the lib; test-module and
cascade errors are additional): **prover.rs 50, fuzz.rs 23, interp.rs 17,
hash.rs 1.**

### A. Scope became interior-mutable (pervasive, mechanical)
- `Scope.instructions: Vec<Instruction>` → **`RefCell<Vec<Instruction>>`**;
  `Scope.locals` likewise `RefCell<Vec<Value>>`. Every `&scope.instructions`
  iterate / index / `.len()` now needs `.borrow()` (or `.get_mut()` when a
  `&mut Scope` is held). ~20 sites across all four files.
- `Scope.const_arrays` field **removed**.
- `AddressType::register` now takes `&Scope` (not `&mut Scope`) — consistent
  with the interior-mutability refactor.
- `Scope::create_local_restricted` **removed** (9 hits in prover.rs, all in the
  test harness that hand-builds kernels).

### B. The SSA value model was rebuilt (`Variable` → `Value`) (architectural)
- `cubecl::ir::Variable` → **`Value`**; `VariableKind` → **`ValueKind`**.
- Crucially, `ValueKind` is now just **`Value { id }` | `Constant(..)`** — the
  rich 0.10 `VariableKind` (`GlobalInputArray`, `GlobalOutputArray`, `LocalMut`,
  `LocalConst`, `ConstantScalar`, `Builtin`, …) is **gone**. Semantics that were
  carried by the variable's *kind* now live in operations: builtins via
  `Operator::ReadBuiltin(Builtin)`, scalars via `Operator::ReadScalar(Id)`,
  buffers via `GlobalStateInner.global_args` + the typemap.
- Impact: the prover's entire variable-classification logic — "is this SSA
  value an input array? an output array? `ABSOLUTE_POS`? a local?" — has no
  direct analog and must be reconstructed from the new operation-carried model.
  This is the single deepest change.

### C. Memory model: direct index/assign → pointer-based load/store (architectural)
- 0.10: `Operator::Index / UncheckedIndex / IndexAssign / UncheckedIndexAssign`
  (each a `BinaryOperator`). All **removed from `Operator`**.
- main: a new **`Operation::Memory(Memory)`** category, where
  `Memory::Index(IndexOperands)` computes an address (`&list[index]`),
  `Memory::Load(Value)` reads through a pointer, `Memory::Store(StoreOperands{
  ptr, value })` writes, `Memory::CopyMemory(..)`. `Operator::CopyMemory /
  CopyMemoryBulk` also gone.
- `IndexOperands` = `{ list: Value, index: Value, unroll_factor: usize,
  checked: bool }`. **The IR now carries a `checked: bool` per index and an
  `unroll_factor` "Adjustment factor for bounds check".**
- Impact on the prover: its central abstraction — find each array index, prove
  `index < len` — must now (a) match `Operation::Memory` instead of `Operator`,
  and (b) follow the pointer produced by `Memory::Index` to the `Load`/`Store`
  that consumes it, rather than reading a self-contained index/assign op. This
  is a rewrite of the bounds-analysis core, not a rename. **Upside:** the new
  `checked` flag is exactly vericl's domain (prove OOB-freedom ⇒ elide checks);
  a ported prover may be able to *read* `checked` and even cross-check it.

### D. Metadata restructured (mechanical-ish, prover-facing)
- `Metadata::Length` and `Metadata::Rank` **removed**.
- `Metadata::BufferLength` field `var` → **`list`**; `Stride`/`Shape` now
  `{ dim, list }`. The prover models `array.len()` via `BufferLength`, so its
  length-resolution must switch to `{ list }`.

### E. Arithmetic / operand structs renamed (mechanical, wide)
- Operand payload structs renamed: `BinaryOperator` → **`BinaryOperands`**
  (`.lhs/.rhs`, output now on `Instruction.out`), `UnaryOperator` →
  **`UnaryOperands`** (`.input`), plus new `IndexOperands`, `StoreOperands`,
  `CopyMemoryOperands`. The removed `cubecl::ir::{BinaryOperator, UnaryOperator,
  IndexOperator, IndexAssignOperator}` imports must be repointed.
- `Arithmetic::Modulo(..)` **split** into **`Rem(BinaryOperands)`** (truncated
  remainder) and **`ModFloor(BinaryOperands)`** (floored). The interp/prover
  `%` modeling must choose the right one (see finding 4c).

### F. Builder / launch harness churn (test-code + fuzz synthesis)
- `KernelBuilder::input_array` / `output_array` **removed** → `buffer(value_ty:
  Type) -> Value`, `tensor(..)`, `scalar(storage: StorageType) -> Id`.
- `LaunchArg::expand_output` **removed** (input/output distinction handled via
  `inplace` now); `Array`'s `ArrayCompilationArg` type is gone.
- Launch surface for `Array<T>`/slice params is now **`BufferArg::from_raw_parts
  (handle, len)`** (was `ArrayArg::from_raw_parts`); scalar launch args grew a
  dtype (`ScalarArg::new(val, dtype)`), though the `#[cube(launch)]`-generated
  path still accepts a raw scalar. `ComputeClient<R::Server, R::Channel>` →
  **`ComputeClient<R>`**.
- `KernelDefinition.buffers`: `Vec<KernelArg>` → **`Vec<BufferInfo>`** (`{ id,
  value, has_extended_meta }`); `scalars`: `Vec<ScalarKernelArg>` →
  **`Vec<ScalarInfo>`**. Still `Serialize`, so `hash.rs`'s serde fold survives —
  but the serialized *bytes* differ (feeds 4a).
- fuzz.rs hand-builds IR via these APIs and the old `If`/`RangeLoop`/index
  structs, so it must be rewritten to the new builder + memory model.

## 4a — does axpy's `ir_hash` change? YES, and it must

Stored baseline (0.10.0, reproduced from `evidence/vericl.json` and the
research doc): axpy = `sha256:3ae1a32f63226aac7b8cb2d19851e3bcad00626ac98b5116131e91f48772e722`.

`kernel_ir_hash` = `SHA256( Scope::hash-bytes ‖ serde(buffers) ‖ serde(scalars)
‖ cube_dim )`. On main it is **guaranteed** to differ, for ≥4 independent reasons:

1. **`Scope`'s hand-written `Hash` impl changed:** it now folds in `locals`
   (`self.depth; self.instructions.borrow(); self.locals.borrow()`) — 0.10 did
   not. Different bytes even for identical IR.
2. **The body IR is structurally different types:** axpy's
   `y[i] = alpha*x[i]+y[i]` is now `Memory::Index/Load/Store` +
   `Arithmetic(BinaryOperands)` + `ReadBuiltin`/`ReadScalar`, vs 0.10's
   `Operator::Index/IndexAssign` + rich `VariableKind`. Different hashed bytes.
3. **`buffers` serialize as `BufferInfo`** (`{id,value,has_extended_meta}`) vs
   `KernelArg` — different JSON folded into the digest.
4. **`Value` (was `Variable`)** has a different serde/hash shape (`{kind, ty}`
   with `ValueKind = Value{id}|Constant`).

This is the drift tripwire working exactly as designed: on a real 0.10→0.11
upgrade **every stored `ir_hash` in `evidence/*.json` goes stale**, and
`conform check` reports `ir_hash` mismatches until each kernel is re-verified
and its evidence re-stamped. The identity *mechanism* itself ports cleanly:
`hash.rs`'s core (`kernel_ir_hash` + the source-loc normalizer + the `Branch`
walker) compiles against main with **one** semantic change — `scope.instructions`
is a `RefCell`, so the normalizer uses `.get_mut()` (proved: drill bin
`crates/f64probe/src/bin/hash_port.rs` compiles). The `Branch` arms are
byte-identical (payloads became `Box<_>` but `&mut` auto-derefs).

### TypeHash — the schema-drift tripwire, with a caveat
CubeCL main added a **`TypeHash`** trait (`pub cubecl::ir::TypeHash`) derived on
the IR types: `type_hash()` is a stable FNV hash of a type's *structure*
(variant/field names + types) that changes iff the schema drifts. This is the
ideal forward-looking tripwire for exactly this drill — pin it and an IR
restructure fails a cheap unit test instead of a wall of `rustc` errors.
**Caveat found:** `type_hash()`'s derived recursion has **no cycle guard**, and
`Type` is self-recursive (`Type::Vector/Pointer/Atomic/Array(Intern<Type>)`), so
calling it on any non-flat IR type (`Scope`, `Instruction`, `Operation`,
`Operator`, `Value`, `Type`, …) **stack-overflows**. Only flat leaf enums work
today, e.g. on the drilled commit `UIntKind=0x4392f585cd6c36bf`,
`IntKind=0xab28d521ed0c8c10`, `FloatKind=0x2362ec1b32639b03`,
`ElemType=0x21c68afe634312bb` (drill bin `ir_typehash.rs`). Worth an upstream
report; until fixed, a vericl tripwire can pin only the flat kinds.

## 4b — f64 on wgpu: STILL silently corrupts on main (disclosure headline)

Self-contained probe `crates/f64probe/src/bin/f64_wgpu.rs` (depends only on
cubecl main; a `#[cube(launch)]` f64 axpy launched on both backends, compared to
a host-f64 reference), n=1027, α=2.5:

```
[cpu]  bit-exact match (f64 correct), worst abs err 0
[wgpu] DIVERGES, worst abs err 2.5675e3 at i=1026 (got 1126, expected 3693.5);
       elem0 got 100 (input, untouched) expected 102.5
[wgpu] launch completed without panic
```

- **No compile error, no runtime panic, no naga/wgpu diagnostic of any kind.**
  A second run grepping all of stdout+stderr for `f64|double|naga|valid|
  unsupported|wgsl|shader|feature` surfaced nothing.
- The corruption signature is identical in character to the 0.10 diagnosis
  (8-byte f64 uploaded into a buffer the WGSL kernel indexes at a different
  element size — some elements untouched, others garbage), i.e. **genuine
  silent corruption, not an f32 demotion.**
- cubecl-cpu remains the correct f64 lane (bit-exact), consistent with the
  README's cpu-only f64 story.
- Nuance: cubecl's own `runtime_tests` acknowledge the problem at the
  test-selection level (a portable-subset comment: "wgpu rejects
  i8/i16/u8/u16/f64"), so upstream *knows* — but there is still **no runtime
  guard**; the launch path silently corrupts rather than erroring.

**Disclosure checklist update (`tasks/todo.md`):**
- Item 1 ("reproduce against main / pre-release"): **DONE — still present at
  `870666bf` (0.11.0-pre.1). Framing = "still present on main; published 0.10
  affected — consider advisory + a launch-time reject/error for f64-on-wgpu."**
- Item 2 (SPIR-V path): **not covered** — this drill exercised the default
  wgpu/naga/WGSL/Metal path (the one vericl uses). SPIR-V untested (Metal
  doesn't use it); still open per the checklist.
- Item 3 (draft framing / minimal repro): the probe above is a ready minimal
  repro (`f64_wgpu.rs`, ~90 LOC, no vericl deps).
- Still queued — **do NOT contact anyone without Ryland's explicit go.**

## 4c — other documented findings on main

- **Eager `&&` / `||`:** `Operator::And/Or(BinaryOperands)` and
  `Operator::Not(UnaryOperands)` still exist as eager operators (pure operands
  take the eager path — main's `short_circuit.rs` says so explicitly). But main
  **added short-circuit lowering + a `runtime_tests/short_circuit.rs`
  regression test** that pins `&&`/`||` to short-circuit when the RHS has side
  effects (via a side-channel array write in a helper `#[cube] fn`). Net: the
  docs-only finding ("a guard `idx_ok && x[idx]` doesn't protect the read")
  is **partially addressed** — whether a *guarded array read* is now
  short-circuited vs still eager depends on whether cubecl now treats an index
  as a side effect, which needs an IR-extraction re-probe (not cheaply doable
  while the prover is un-ported). Flag for re-verification during the real
  upgrade; the prover already *refutes* the unsafe case, so this is a
  precision / story-accuracy question, not a soundness regression.
- **Division/modulo rounding:** the IR `Arithmetic::Modulo` **split into `Rem`
  (truncated) and `ModFloor` (floored)** — upstream clarified modulo semantics.
  The prover/interp `%` modeling must pick the correct variant. Naga's
  dividend-preserving div-by-zero *runtime* fallback (`a/0==a`, `a%0==0`) is a
  backend behavior, not an IR change, and was **not re-probed** (cheaply
  possible with a wgpu arithmetic probe if the disclosure needs it; low value
  since it is naga's behavior, not cubecl's, and unlikely to have moved).
- **`terminate!()` host expansion / `CUBE_COUNT` on cpu:** these are
  macro/host-shim and cpu-runtime findings that live above the `vericl-ir` IR
  layer; **not re-verifiable in this drill** without the ported macro-launch
  path. Deferred to the real upgrade's re-verification pass.

## Effort estimate (to port vericl-ir to 0.11)

Ordered by the natural dependency/isolation gradient:

| Subsystem | Change class | Estimate |
|---|---|---|
| Toolchain bump to rustc ≥ 1.95 | config | ~0 (install; note MSRV in README/CI) |
| `hash.rs` core (`kernel_ir_hash`) | mechanical (RefCell) | **~0.5 h** — proven: 1 line |
| `hash.rs` test harness (`build_axpy` etc.) | builder/launch API | ~1–2 h |
| `interp.rs` (reference interpreter) | Value model + memory model + Rem/ModFloor + Metadata | **~1–2 days** |
| `fuzz.rs` (IR synthesis + corpus) | builder + memory model rewrite | ~1–2 days |
| `prover.rs` (SMT OOB/race prover) | **architectural**: SSA-value reclassification (B) + pointer-based memory model (C) + Metadata (D) | **~1–2 weeks**, plus re-running the full 7-round soundness review — no proof can be trusted until re-validated end-to-end |
| Re-stamp all `evidence/*.json` `ir_hash`/`source_hash` | expected drift | ~1 h (mechanical `VERICL_UPDATE=1` per suite, after the above) |

**Boundary the drill stopped at (per protocol):** the `prover.rs` port is
architectural — both the SSA-value model (§B) and the memory model (§C) changed
fundamentally, so it was *characterized, not rewritten*. `hash.rs` core was the
one mechanical fix carried through (and proven to compile). `interp.rs` and
`fuzz.rs` are large-but-mechanical ports gated on the same two upstream changes.

## Recommended upgrade playbook

1. **Wait for a tagged 0.11 release** (this drilled `0.11.0-pre.1`; the API may
   still move before release). Bump the pin deliberately, one version.
2. **Bump MSRV to the release's requirement (≥1.95 as of this drill).** Update
   `rust-version`, the README's toolchain note, and any local CI.
3. **Confirm the free wins first:** `cargo test -p vericl` and
   `-p vericl-macros` should pass untouched (they did on main). If either
   breaks, something leaked the isolation boundary — investigate before going on.
4. **Port `vericl-ir` bottom-up, module by module, per-crate with timeouts:**
   `hash.rs` (trivial) → `interp.rs` → `fuzz.rs` → `prover.rs` (the long pole).
   Land `hash.rs` first so identity/`conform check` works early.
5. **Re-anchor the prover on the new model:** rebuild variable classification
   from `global_args`/`ReadBuiltin`/`ReadScalar` (§B) and re-express bounds
   analysis over `Memory::Index → Load/Store` pointers (§C). Evaluate reading
   `IndexOperands.checked` directly — it may simplify or cross-check the prover.
6. **Re-validate soundness, don't assume it.** Re-run the full multi-round
   review (the negative controls, the terminate/eager-`&&`/wrapping probes) —
   the prover's proofs are only as good as the model they sit on, and the model
   changed. Re-verify findings 4c (eager-`&&` guarded read; Rem vs ModFloor) at
   the IR level once extraction works again.
7. **Re-stamp evidence last:** run `VERICL_UPDATE=1` per suite to refresh the
   now-stale `ir_hash`/`source_hash` values (4a), one binary at a time.
8. **Adopt `TypeHash` as a standing tripwire** (once the upstream recursion
   overflow is fixed, or scoped to flat leaf kinds until then): pin the IR type
   hashes so the *next* schema drift trips a unit test, not a build wall.
9. **Schedule a dedicated review round** for the upgrade itself — treat the
   prover re-anchoring as a soundness-critical change, not a mechanical bump.

## Files / tests run in the drill

- Baseline @ 0.10.0 (drill copy): `vericl-ir` 124 tests pass, `vericl-macros`
  60 pass, `vericl-examples` compiles (lib+bins+tests).
- Against main @ `870666bf` (rustc 1.97.1): `vericl` 36 pass, `vericl-macros`
  60 pass; `vericl-ir` = 89 first-pass compile errors (catalog above).
- Probes (`crates/f64probe/`, cubecl-only): `f64_wgpu` (4b, ran — cpu bit-exact,
  wgpu silent divergence), `ir_typehash` (TypeHash values + overflow finding),
  `hash_port` (hash.rs core compiles against main).
- Canonical repo `/Users/ryland/code/vericl`: untouched except this doc and the
  `tasks/todo.md` record; `=0.10.0` pin intact, git otherwise clean.
