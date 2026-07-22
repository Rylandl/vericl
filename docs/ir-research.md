# CubeCL 0.10 IR access — research findings (July 2026)

Validated empirically with three working prototypes against the pinned `cubecl =0.10.0`
(`z3 0.20.2`, `easy-smt 0.3.2`). Prototype sources preserved in
[prototypes/ir_extraction.rs](prototypes/ir_extraction.rs) and
[prototypes/smt_bounds_check.rs](prototypes/smt_bounds_check.rs). This informs the IR-level
identity hash and SMT bounds-checking milestones. File:line citations refer to the CubeCL
v0.10.0 source tree.

## 1. IR extraction needs zero client/runtime/device

`#[cube(launch)] fn axpy(...)` generates `pub mod axpy` containing `pub fn expand(&mut Scope,
...)` (visibility follows the original fn). Calling `expand` directly with a hand-built
`KernelBuilder` yields the full `KernelDefinition` — no `ComputeClient` needed:

```rust
let mut builder = KernelBuilder::default();
builder.runtime_properties(Default::default());
AddressType::U32.register(&mut builder.scope); // REQUIRED: usize/isize storage type; panics without it
let alpha = <f32 as LaunchArg>::expand(&Default::default(), &mut builder);
let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
let y = <Array<f32> as LaunchArg>::expand_output(&ArrayCompilationArg { inplace: None }, &mut builder);
axpy::expand(&mut builder.scope, alpha, x, y);
let def: KernelDefinition = builder.build(KernelSettings::default());
```

The generated `CubeKernel::define()` calls `builder.device_properties(client.properties())`,
but `Scope.properties` is read in exactly one place: the `comptime!` intrinsic
(cubecl-core `frontend/comptime.rs:11`) — which vericl-macros already bans. With vericl's
`default-features = false` pin, no backend crate compiles at all; only cubecl-core/-ir/
-runtime/-macros are in the graph, so this path is natural, not a workaround.

Trade-off: this couples to `KernelBuilder`'s registration plumbing (small, stable across the
0.10 line) rather than the blessed `define()` path. Cover with the "survives a CubeCL upgrade"
health check; `TypeHash` (below) is a cheap schema-drift tripwire.

## 2. Determinism and hashing

- IR types all derive serde (`cubecl-core` requests `cubecl-ir/serde` unconditionally).
- Expansion is deterministic: variable ids come from a plain counter driven by execution
  order of the generated `expand()` code; identical source ⇒ identical IR.
- **TRAP: never use `==`/`assert_eq!` on `Scope`.** `Allocator::PartialEq` is `Rc::ptr_eq`
  (reference identity, `allocator.rs:37-42`) — two identical builds always compare unequal.
  Pin this with a unit test so a future CubeCL "fix" doesn't silently change identity semantics.
- The safe mechanism: `Scope`'s hand-written `Hash` (`scope.rs:77-90`) hashes only the
  semantic fields (instructions, locals, shared, const_arrays, ...) and skips the allocator/
  caches. Drive it with a custom `Hasher` forwarding to SHA-256. `KernelArg`/`ScalarKernelArg`
  only derive `Serialize` — fold their serialized bytes into the same hasher.
- Validated: `sha256:3ae1a32f...` for axpy reproduced across repeated runs, fresh processes,
  and a full `cargo clean` rebuild; flipping `<` to `<=` changed the hash; reverting restored it.
- `Instruction.source_loc` is `None` unless CubeCL's `debug_symbols` cfg is on; normalize to
  `None` before hashing anyway (absolute paths would otherwise leak into identity).
- CubeCL's own `KernelId` is a compilation-cache key (TypeId-based), not a content hash — not
  usable for identity.

## 3. IR shapes for the bounds walker

- Index ops: `Operator::{Index,UncheckedIndex,IndexAssign,UncheckedIndexAssign}` with
  `IndexOperator { list, index, vector_size, unroll_factor }`. Non-1 vector/unroll only for
  `Line`/vectorized code (banned) — assert defensively, reject otherwise.
- Loops: `Branch::RangeLoop { i, start, end, step, inclusive, scope }` maps to `for i in
  start..end`. Bare `Branch::Loop` (break-terminated) has no static bound — reject for v0.
- `ABSOLUTE_POS` → `VariableKind::Builtin(Builtin::AbsolutePos)`, typed per AddressType (u32).
- Lengths: `Metadata::Length { var }` is the caller-declared logical length (`y.len()` lowers
  to it). `Metadata::BufferLength` is the physical allocation — conflating them makes the
  checker unsound once inplace/aliasing exists. Key strictly off `Length`; reject inplace
  buffers for v0.
- Structure: `Scope.instructions: Vec<Instruction>`, `Operation::{Arithmetic, Comparison,
  Operator, Metadata, Branch, ...}`; `LocalConst{id}` is assigned once (SSA-ish). Walker =
  recursive descent with a path-condition stack; each Index/IndexAssign emits the obligation
  `0 <= index < Length(list)` under current path conditions + contract assumes.

axpy's actual trace (from the prototype):

```
binding(0) = output(1).len()
binding(1) = AbsolutePos < binding(0)
if(binding(1)) {
    binding(2) = input(0)[AbsolutePos]
    binding(3) = scalar<f32>(0) * binding(2)
    binding(4) = output(1)[AbsolutePos]
    binding(5) = binding(3) + binding(4)
    output(1)[AbsolutePos] = binding(5)
}
```

## 4. Solver: easy-smt + subprocess z3 (decided)

Measured comparison: `z3` FFI crate built in 4.45s but only after manual
`LIBRARY_PATH=/opt/homebrew/lib` (linker couldn't find libz3 even with Homebrew z3 installed);
`easy-smt` built in 0.79s, pure Rust, ~2 tiny deps, zero setup. The subprocess model also
keeps the solver an external, independently versioned **trusted component** — capture
`z3 --version` in the evidence manifest, same trust posture as backend codegen. v0 formulas
are plain QF_LIA; SMT-LIB2 text is sufficient. CI/dev machines need `z3` on PATH
(`brew install z3`; present at /opt/homebrew/bin/z3 on this machine).

End-to-end validation: the axpy guard obligation (`assumes(x.len()==y.len())`,
`0 <= pos < num_threads`, guard `pos < y.len()` ⟹ in-bounds access of both arrays) proved
UNSAT; removing the assumes clause flipped it to SAT (counterexample) — the contract's assumes
are load-bearing for the proof exactly as the claim model requires.

## 5. Implementation risks

1. `Allocator::PartialEq` Rc-identity trap (pin with a test).
2. `KernelBuilder` plumbing is undocumented internals — health-check on upgrade.
3. Only `RangeLoop` is boundable; reject `Loop`.
4. `Length` vs `BufferLength` distinction is a soundness edge.
5. Assert `vector_size`/`unroll_factor` trivial rather than assuming.
