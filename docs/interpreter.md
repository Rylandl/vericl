# The IR reference interpreter and the model-fidelity cross-check

*(Foundation rung B, 2026-07-24. See `tasks/todo.md` for the as-built record.)*

## Why this exists

Everything VeriCL proves and tests is stated over the CubeCL **IR**: the SMT
bounds/race prover (`crates/vericl-ir/src/prover.rs`) reasons about a
symbolic encoding of `KernelDefinition.body`, and the identity hash is taken
over the same IR. That makes one component load-bearing and, until now,
unchecked: **does VeriCL's model of what a CubeCL instruction *means* match
what CubeCL actually executes?** A subtle mismatch — a wrong wrapping rule, an
off-by-one bound, a misread div/mod — would let the prover certify a property
the hardware violates, with no test catching it.

The interpreter attacks that gap empirically. It is a **third, independent
implementation** of the modeled cube semantics:

| Implementation | Lives in | Reads | Strategy |
|---|---|---|---|
| macro **twin** | `crates/vericl-macros` | kernel *source tokens* | rewrite to a host `reference(...)` fn |
| **prover** | `crates/vericl-ir/src/prover.rs` | the *IR* | symbolic QF_LIA encoding + z3 |
| **interpreter** | `crates/vericl-ir/src/interp.rs` | the *IR* | concrete execution over real inputs |

All three consume the *same* `KernelDefinition`. Running the interpreter
against the twin (two independent semantics concur) and against the prover (a
`Proved` kernel the interpreter can drive out of bounds would be a fidelity
defect) is a standing empirical check that the model is faithful.

## What the interpreter is

`vericl_ir::interpret_dispatch(def, inputs) -> Outcome` executes the IR body
concretely, one thread at a time over `AbsolutePos = 0..num_threads` (threads
are independent in the non-cooperative subset, exactly as the sequential twin
runs them), applying every write to the buffers and returning either the final
buffers, a **reported** out-of-bounds access, a reported division by zero, or
an `Unsupported` verdict for a construct outside its subset.

Deliberate design points:

- **Real finite-width semantics.** Integers wrap exactly (`wrapping_add/sub/mul`,
  width-masked shifts, truncated-toward-zero div/mod); floats use IEEE-754
  `f32`/`f64` (separate `Mul`+`Add` stay strict, an explicit `Fma` fuses). This
  is what makes it a faithful stand-in for the GPU and bit-exact with the twin.
- **Reports OOB, never panics.** An out-of-bounds index read/write (checked or
  unchecked) and a divide-by-zero become a structured `Outcome`, with the array
  name, index, length, and thread — so a defect surfaces as data, not a crash.
- **Fails closed on anything unmodeled.** A construct outside the subset is
  rejected up front (`Unsupported`) rather than approximated.

## What it covers (v0)

Non-cooperative, scalar, 1-D kernels:

- arithmetic (`Add`/`Sub`/`Mul`/`Div`/`Modulo`/`Neg`/`Abs`/`Min`/`Max`/`Fma`,
  `MulHi`, and the float transcendentals `sqrt`/`recip`/`floor`/…);
- comparisons and boolean composition (`And`/`Or`/`Not`);
- bitwise ops (`&`/`|`/`^`/`<<`/`>>`/`!`/`count_ones`/`leading`/`trailing`);
- `Cast` (`as`-semantics), `Reinterpret` (bitcast), `Select` (ternary);
- `Metadata::Length`; `Index`/`IndexAssign` (checked and unchecked) with bounds
  reporting;
- control flow: `If`, `IfElse`, `Switch` (`match`), `RangeLoop` (ascending,
  unit or positive step), and the bare `Loop` (break-terminated, guarded by an
  instruction budget);
- topology builtins for a 1-D dispatch (`AbsolutePos`/`UnitPos`/`CubePos`/
  `CubeDim`/`CubeCount` and their X/Y/Z forms);
- local scratch arrays (`LocalArray`) and constant arrays.

## What it excludes (reported, never guessed)

- **All cooperative constructs**: `SharedMemory`/`SharedArray`, `sync_cube` and
  the other barriers. A faithful cooperative interpreter needs a lock-step
  multi-thread phase model; that is future work, so a cooperative kernel is
  rejected `Unsupported` up front rather than mis-executed single-threaded.
- Atomics, plane/warp ops, cooperative-matrix, TMA, tensor metadata
  (`Rank`/`Stride`/`Shape`), vectorized (`Vector<_, N>`) indexing, and
  stepped/descending range loops.

## The cross-checks

**1. Public examples — interpreter ≡ twin (bit-exact), on real `#[cube]` IR.**
`crates/vericl-examples/tests/interp_crosscheck.rs` runs the interpreter over
each honest kernel's actual `kernel_definition()` and compares, bit-for-bit,
against the macro twin over many random inputs and sizes: `axpy`,
`xorshift_step`, `mix_u32`, `flatten_decode_scale`, `gather_copy`,
`select_mode`, `offset_window`, `fir3`, `gain_kernel`, `lcg_map`,
`comptime_shift`, `mul_hi_map`, `unit_interval_map`, plus a guard-boundary case.
One kernel (`xorshift_step`, exact integer) is checked **three-way**:
interpreter vs twin vs a real wgpu/Metal GPU launch — all bit-identical.

**2. Fuzz lane — interpreter ⇄ reference and prover ⇄ interpreter.**
`crates/vericl-ir/src/fuzz.rs` generates random in-subset kernels from a small
grammar (guarded indexing, arithmetic/bitwise chains, div/mod by a nonzero
divisor, `if`/`switch`/bounded loops, gathers through a valid offset table),
each realized two ways from the same AST — lowered to hand-built IR (the same
instruction shapes `#[cube]` emits) and evaluated directly by an independent
tree-walking reference. For each kernel:

- **(a)** the AST reference and the IR interpreter must produce the same output
  (or the same OOB) on many random inputs, valid and adversarial;
- **(b)** the prover's verdict is cross-checked against the interpreter: if the
  prover says `Proved`, no assume-satisfying input may drive the interpreter out
  of bounds; if it says `Refuted`, its counterexample (or the shape's minimal
  witness) replayed through the interpreter must exhibit the OOB.

Any disagreement is surfaced as a `Finding` with full reproduction detail,
never silently reconciled. A deterministic subset (400 kernels, prover on) runs
in `cargo test`; the full corpus (20,000 kernels) runs behind `VERICL_FUZZ=1`.
As built, the full corpus — 20,000 kernels, 320,000 agreement inputs, 14,285
`Proved`, 5,715 `Refuted` — produced **zero** findings in 215s.

The hand-built fuzz IR *mirrors* CubeCL's shapes but is not produced by
`#[cube]` (random structure cannot be macro-expanded at runtime). Its primary
value is soundness at scale — thousands of independent prover-vs-interpreter and
reference-vs-interpreter checks. Fidelity to *genuine* CubeCL IR is anchored
separately by cross-check #1 and the `interp.rs` unit tests, both of which run
the interpreter over real `#[cube]`-expanded `KernelDefinition`s.

## What agreement does and does not establish

**Does** (empirical, honest):

- Shrinks model-fidelity risk. Two (public examples: three) independent
  implementations of cube semantics concur on the actual kernels; the prover's
  `Proved`/`Refuted` verdicts are corroborated by concrete execution at scale.
- Discriminates against real bugs. Injecting a deliberate semantics bug into the
  interpreter — a flipped integer add, a flipped float add, a bounds check that
  hides an OOB — is caught immediately by the cross-checks (negative controls,
  run and confirmed for rung B; not committed).

**Does not** (the claim-taxonomy honesty standard):

- It is **not a proof** and mints no `Proved` claim. Agreement is a `Tested`
  observation over specific inputs, not a universally-quantified guarantee.
- It does not verify anything below the IR: CubeCL's front-end expansion and
  backend codegen, the driver, and the hardware remain **Trusted** (the
  three-way GPU leg tests, but does not verify, that layer).
- A shared blind spot common to twin, interpreter, and reference (all three
  wrong the same way) would not be caught by their mutual agreement — which is
  precisely why the prover is the fourth, differently-constructed leg, and why
  the negative controls exercise the harness's discriminating power directly.

In short: the interpreter turns "trust that VeriCL's IR model matches CubeCL"
into an empirically-checked, continuously-run property — without ever claiming
that empirical check is a proof.
