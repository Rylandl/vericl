# VeriCL

> One kernel contract. Equivalent implementations.

VeriCL is a conformance and evidence harness for [CubeCL](https://github.com/tracel-ai/cubecl)
compute kernels. You write a kernel once in Rust, attach the assumptions and properties that
matter, and VeriCL produces the artifacts and checks needed to support clearly bounded claims
about it: differential test results across backends, machine-checked safety properties, and an
evidence manifest that goes stale when anything it depends on changes.

## Status

Exploratory. This document records the problem, the design decisions that are locked in, and the
scope of the first release. Decisions marked **open** are genuinely undecided; everything else is
settled unless the first release proves it wrong. The original backend-neutral ideation charter is
archived at [docs/ideation-charter.md](docs/ideation-charter.md).

## Problem

Accelerated kernels are hard to trust for reasons beyond the arithmetic in their bodies: indexing
and layout conventions differ between implementations, boundary behavior is implicit, parallel
execution introduces collisions and ordering differences, optimizations change numerical behavior,
and reference implementations drift away from the accelerated code they supposedly describe. Tests
demonstrate selected cases without explaining the scope of the guarantee, and formal results can
prove a model without establishing that deployed code implements it.

The usual failure mode is not a wrong artifact but silent disagreement between artifacts that each
look reasonable in isolation. VeriCL keeps the kernel's intended behavior, its executable
realizations, and the evidence about them mechanically connected, so that disagreement is detected
instead of accumulated.

## Locked decisions

| Decision | Choice |
|---|---|
| Implementation language | Rust |
| Kernel substrate | CubeCL (`#[cube]` kernels) |
| Authoring experience | Plain CubeCL kernels plus a `#[vericl(...)]` attribute for contracts — no new notation |
| Point of custody | The annotated Rust kernel function; every other artifact is derived from or checked against it |
| Kernel identity | Content hash of the expanded CubeCL IR plus the contract plus the toolchain versions |
| Independent comparison | Scalar CPU reference execution derived from the same kernel definition, differentially tested against GPU runs |
| First machine-checked property | Out-of-bounds freedom for a supported kernel subset, discharged by an SMT solver over the CubeCL IR |
| Numerical stance (v1) | Exact comparison for integer kernels; floating-point kernels declare a per-kernel tolerance that is recorded as an assumption in the evidence |
| Evidence format | A manifest binding every result to the kernel identity it was produced from; both human- and machine-readable |
| CI story | Conformance runs under plain `cargo test` (`vericl::suite!` generates the test); `VERICL_UPDATE=1 cargo test` regenerates evidence. A standalone `vericl check` CLI is future work — the `cargo test` path fully covers "fails on missing, stale, or mismatched evidence" for v0 |

### Why CubeCL

A `#[cube]` kernel is written in a subset of Rust whose semantics parallel ordinary Rust. That
makes the central idea concrete instead of aspirational: the kernel function itself is the single
point of custody, and a scalar reference implementation can be derived from the same definition
rather than hand-maintained alongside it. CubeCL also compiles one kernel through its own IR to
multiple backends (wgpu/WGSL, CUDA, ROCm/HIP, SPIR-V), so cross-target differential comparison
falls out of the design rather than being engineered per backend. Its IR is accessible from Rust,
which gives static checking a well-defined substrate.

The cost is coupling to a young, fast-moving project. Mitigations: pin the CubeCL version,
isolate all IR-facing code in one crate, and treat "survives a CubeCL upgrade" as a recurring
health check rather than a surprise.

### The contract attribute (implemented)

```rust
#[vericl::kernel(
    assumes(
        x.len() == y.len(),
        alpha.abs() <= 4.0,
        x.iter().all(|v| v.abs() <= 100.0),
        y.iter().all(|v| v.abs() <= 100.0)
    ),
    compare(abs = 1e-4),
    gen(alpha in -4.0..=4.0, x in -100.0..=100.0, y in -100.0..=100.0),
    instantiate(F = f32)
)]
#[cube(launch)]
pub fn axpy<F: Float + CubeElement>(alpha: F, x: &Array<F>, y: &mut Array<F>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
    }
}
```

From this single definition VeriCL derives, in a generated `axpy_vericl` module: the untouched
CubeCL kernel; a sequential scalar `reference` twin (`ABSOLUTE_POS` becomes a loop variable,
`&Array<T>` becomes `&[T]`, and — per `instantiate(...)` below — the generic type parameter is
substituted to its pinned concrete type) sharing no CubeCL machinery; the `assumes` clauses as an
executable `check_assumes` predicate; a `SOURCE_HASH` identity that evidence binds to; and — from
the `gen(...)` clause — a `conformance_case` function that generates inputs, runs the reference and
the real kernel, and compares them, so no kernel needs hand-written GPU launch/input-gen glue.
Kernels using constructs the twin cannot model (`UNIT_POS`, `SharedMemory`, `plane_*`, `comptime!`
blocks, vectors, `return`) are rejected at compile time rather than silently approximated. Kernel
*composition* — calling another `#[cube]` fn — is supported via `#[vericl::helper]` and a
kernel-side `uses(...)` clause; see "Kernel composition" below.

### The `instantiate(...)` clause: monomorphizing generic + `#[comptime]` kernels

Real CubeCL kernels are overwhelmingly generic over their element type (`<F: Float>`) and use
`#[comptime]` parameters for unroll counts, tap counts, and feature toggles — a July 2026 dogfooding
survey against a private 22-kernel production codebase found generics blocking 20/22 kernels and
`#[comptime]` blocking 15/22 (see `docs/dogfood-2026-07.md`). `instantiate(...)` names a concrete
value for every generic type parameter and every `#[comptime]` parameter the kernel declares —
`instantiate(F = f32, taps = 3)` — and VeriCL monomorphizes everything it derives at those values:

- **Reference twin**: the generic type ident is substituted token-wise wherever it appears in the
  twin's signature and body (`F` -> `f32`); `#[comptime]` parameters are removed from the twin's
  signature entirely and instead bound as `let name: ty = value;` consts at the top of `reference`
  (before the `ABSOLUTE_POS` loop — they're loop-invariant by construction) and `check_assumes`.
  The perf-only `#[unroll]`/`#[unroll(n)]` statement attribute is stripped from twin loops (it isn't
  valid plain Rust); any *other* statement attribute is a compile error, not a silent drop.
- **`conformance_case`**: launches via `<name>::launch::<f32, R>(...)`, with `#[comptime]` values
  spliced in at their declared parameter position — CubeCL keeps a comptime param in its original
  position with its plain type, it's only non-const params that get wrapped for the runtime.
- **`kernel_definition()`** (the IR the SMT prover and `ir_hash` see): calls the CubeCL-generated
  `expand::<f32>(...)` with the same turbofish and comptime values, exactly mirroring a real call
  site.
- **Contract identity**: instantiation values are part of the raw contract attribute tokens, so
  `SOURCE_HASH` already changes when they change; `Contract`/`ContractRecord` additionally record
  the pinned values as strings (`instantiate: ["F = f32", "taps = 3"]`) purely for evidence
  legibility.

A kernel with generic type parameters and/or `#[comptime]` parameters and **no** `instantiate(...)`
clause is a targeted compile error telling you to add one; an `instantiate(...)` clause on a kernel
with neither is also an error (an unused instantiation is a contract lie). v0 supports exactly one
`instantiate(...)` clause per kernel — multiple instantiations of the same kernel body is future
work — and only plain type generic parameters (no lifetimes, no const generics, no where-clauses).

**Float-method host-callability.** After substitution the twin's body may call `Float`/`Numeric`
trait methods (`F::new(x)`, `x.sqrt()`, ...) resolved through `cubecl::prelude`'s traits. Most of
these are safe to call on the host: either they have a real per-type implementation (`Float::new`)
or they share a name with a `std` `f32` inherent method, which Rust's method resolution always
prefers over a trait method regardless of which traits are `use`-imported. A few are *not* safe —
`log1p`, `inverse_sqrt`, `erf`, and `is_inf` have no such shadow and panic
(`Unexpanded Cube functions should not be called.`) if called on the host at all. VeriCL verified
this empirically (`crates/vericl-examples/tests/float_method_whitelist.rs` calls every candidate
method on `f32` and either cross-checks it against `std` or confirms it panics) and rejects, at
macro time, any twin body calling a method outside the verified whitelist:
`error: host-callability of 'F::erf' in the reference twin is unverified — outside the vericl v0
subset`. This is an explicit rejection, not a best-effort attempt — a twin that silently miscomputes
or panics on a method vericl never verified is exactly the failure mode this project exists to
prevent.

### Kernel composition: `#[vericl::helper]` and `uses(...)`

Real kernels call other `#[cube]` functions — the July 2026 dogfooding survey found this blocking
16/22 production kernels, the largest gap after generics/`#[comptime]` (see
`docs/dogfood-2026-07.md`). `#[vericl::helper]` extends the same derivation story to non-launch
`#[cube]` device functions:

```rust
#[vericl::helper(instantiate(F = f32))]
#[cube]
pub fn single_tap<F: Float>(a: F, gain: F) -> F {
    a * gain
}

#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0, gain in -4.0..=4.0),
    instantiate(F = f32),
    uses(single_tap)
)]
#[cube(launch)]
pub fn gain_kernel<F: Float + CubeElement>(x: &Array<F>, y: &mut Array<F>, gain: F) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = single_tap(x[ABSOLUTE_POS], gain);
    }
}
```

`#[vericl::helper]` re-emits the `#[cube]` function untouched and generates a host twin
`fn single_tap_vericl_ref(...)` plus a `single_tap_vericl` module carrying its own `SOURCE_HASH`.
The kernel's `uses(single_tap)` clause rewrites its twin's calls to `single_tap(...)` into calls to
`single_tap_vericl_ref(...)`; a call to a function that's neither `uses(...)`-listed, a local
binding, nor a small allowlist of known host-safe free functions is a targeted compile error
naming the function and suggesting `uses(...)` + `#[vericl::helper]`, instead of the confusing
type error that would otherwise surface deep in cubecl's generated code. Helpers may call other
`#[vericl::helper]`-annotated functions via their own `uses(...)` clause — the identical mechanism,
so helper-calling-helper needs no special casing. `#[comptime]` parameters on a helper stay
ordinary pass-through parameters (the caller's own twin already has the pinned value in hand to
pass along); `ABSOLUTE_POS` and every other topology builtin are banned in a helper's body — a pure
device function reading global thread position would make its twin's calling convention ambiguous
(the dogfood survey found zero helpers using topology, so this costs nothing real).

**A helper's generic type parameter must be monomorphized via its own `instantiate(...)`, exactly
like a kernel's — it cannot be left generic**, even though an early draft of this design tried
that. The reason is the same Float-method-whitelist story above, taken one step further: on a
*concrete* receiver (`x: f32`), Rust prefers the inherent `f32::sqrt` over the trait method, which
is what makes the whitelist host-safe. On a still-generic, merely-bound receiver (`x: F` with
`F: Float`), there is no inherent method to prefer — the call resolves purely through the `Float`
trait, whose default body is the same `unexpanded!()` panic the whitelist exists to keep out.
Verified empirically (not just reasoned about): a scratch `fn g<F: Float>(x: F) -> F { x.sqrt() }`
panics on host calling `g(2.5f32)`, as does `.abs()` — reading cubecl-core's `impl_unary_func!`
macro confirms why (`impl Sqrt for f32 {}` inherits the panicking default rather than overriding
it). Monomorphizing a helper via its own `instantiate(...)` reuses the exact machinery already
verified safe for kernels instead of introducing a second, weaker safety story. The practical cost
is small: a helper's twin is pinned to one concrete type (today, `f32` is the only type any part of
vericl v0 supports, so this is free in practice — revisit if/when an `f64` tier is added).

**Identity and composition.** A kernel's `SOURCE_HASH` constant only ever covers its own source
tokens, computed at macro-expansion time — it cannot see a change to a helper's body, since that
lives in a separate macro invocation vericl-macros has no way to observe. `<kernel>_vericl::identity()`
closes this gap at ordinary Rust runtime: it folds `SOURCE_HASH` together with every `uses(...)`-listed
helper's own `identity_hash()` (via `vericl::combine_source_hash`, a small SHA-256 combine — the
one place core `vericl` depends on `sha2`, still with no `cubecl` dependency), and a helper's
`identity_hash()` recursively folds in its *own* `uses(...)` the same way, so a change two levels
deep in a helper-call chain still moves the top-level kernel's recorded identity. This is defense
in depth alongside, not instead of, the IR-level hash: cube expansion inlines a used helper's real
IR directly into the composing kernel's own `Scope`, so `ir_hash` already reflects a helper body
change too — `identity()` makes the source-level hash honor composition the same way rather than
leaving that half silently stale. A helper (or kernel) whose `uses(...)` graph is cyclic — including
the degenerate case of listing itself — is rejected at compile time on a best-effort basis: a
process-local registry accumulates every `uses(...)` edge seen so far in the compilation and checks
for a cycle on each new declaration, which reliably catches any cycle written in ordinary top-to-
bottom source (the last node in a cycle to be macro-expanded always closes it, and by definition
every other node has already registered by then) but is not a soundness-critical guarantee, since a
`#[proc_macro_attribute]` invocation cannot see other invocations' output directly. `#[cube]` itself
does not help here — verified empirically that both direct and mutual recursion between `#[cube]`
functions compile cleanly today (the former only draws rustc's ordinary `unconditional_recursion`
lint *warning*). As a backstop for the residual gap, the runtime hash-combine is depth-guarded
(32 levels) and panics naming the offending item rather than hanging, should a cycle ever slip past
the compile-time check.

The SMT bounds prover needed zero changes for composition: cube expansion inlines a used helper's
IR directly into the composing kernel's own `Scope`, so the existing walker over
`kernel_definition()` already sees everything a helper's body does — a guarded array access inside
a composed helper discharges exactly like one written directly in the kernel, and an unguarded one
refutes the same way (see `crates/vericl-examples/src/lib.rs`'s `tap_pair_guarded_kernel`/
`tap_pair_unguarded_kernel` for the pinned positive/negative pair).

### The `gen(...)` clause: ergonomic by being explicit

`gen(...)` declares, per parameter, how `conformance_case` draws inputs: `name in lo..=hi` for a
scalar or (applied elementwise) an array, and an optional `len(name = N)` to pin an array's
generated length to a constant instead of the case size — needed by kernels like `sum_racy`, whose
`assumes(y.len() == 1)` requires `gen(..., len(y = 1))`. Integer parameters left out of `gen(...)`
default to full-range generation; **float parameters with no declared range are a compile error**,
not a silent default. This is a deliberate ergonomic decision: an unbounded float draw produces
NaN/inf-adjacent garbage and tolerances no `compare(abs = ...)` can honestly justify, and the
failure is far more useful caught at authoring time (`error: parameter alpha is a float with no
declared gen(...) range`) than surfacing later as a confusing NaN mismatch or an unprovable
tolerance at run time. Generated inputs are drawn from vericl's `SplitMix64` in kernel-parameter
declaration order (not `gen(...)` clause order) for determinism, then checked against
`check_assumes(...)`; a rejected draw resamples (same RNG stream) up to 64 times before erroring
with the kernel name, so a persistent failure means the declared ranges are inconsistent with the
kernel's own `assumes(...)`, not a runtime fluke.

### Suites: `vericl::suite!`

```rust
vericl::suite! {
    runtime: cubecl::wgpu::WgpuRuntime,
    kernels: [axpy, xorshift_step, mix_u32],
    evidence: "evidence/vericl.json",
}
```

Expands to `#[test] fn vericl_conformance()`: builds the client, runs every listed kernel's
`conformance_case` across the declared sizes, discharges the SMT bounds proof via `vericl-ir`
(`prove: false` omits proved claims instead of ever recording a fake or skipped one), and
assembles the evidence manifest. With `VERICL_UPDATE` set (any value), it writes the manifest;
otherwise it loads what's on disk, calls `vericl::verify`, and panics with the problem list on any
mismatch — so `cargo test` is the whole CI story. The evidence path is relative to
`CARGO_MANIFEST_DIR`. An optional `extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime)`
folds an additional differential lane (sharing CubeCL's front end, so recorded as *not
independent* — only the macro-derived sequential twin is) into the same test, appending claims to
the same entries before the manifest is finalized, so one suite invocation always produces exactly
one manifest.

### A first finding: why `compare(abs = ...)` exists

The very first differential run caught the wgpu/Metal backend contracting `a*x + y` into a fused
multiply-add: under catastrophic cancellation (`alpha*x ≈ -y`) the observed divergence from the
strict-rounding reference reached ~27,000 ULP. No useful ULP bound exists for this kernel on this
backend — the honest claim is an absolute error bound (`|e-a| <= abs + rel*|e|`) derived from the
declared input ranges in `assumes(...)`. The tolerance is part of the contract and is recorded as
an assumption in the evidence, exactly as the claim model requires.

## Claims and trust boundaries

VeriCL must say exactly what a result establishes. These are different claims and are never
presented as interchangeable:

- **Proved** — a property discharged by a checker over the kernel IR, under stated assumptions.
- **Tested** — behavior observed on specific inputs, on a specific backend, driver, and device.
- **Assumed** — declared constraints (input ranges, tolerances) that evidence depends on but does
  not establish.
- **Trusted** — components outside the checked boundary: CubeCL's backend code generation, the
  driver, the hardware. Source-level evidence never silently implies these are verified.

Every evidence entry records which of these categories each part of its claim falls into, and the
assumptions travel with the result. Evidence that no longer matches the kernel identity it was
produced from is rejected, not warned about.

### Proved claims

The first proved claim is live: out-of-bounds freedom for `axpy`, `xorshift_step`, and `mix_u32`,
discharged in QF_LIA by z3 (subprocess, via `easy-smt`) over each kernel's CubeCL IR — every
`Index`/`IndexAssign` obligation negated and checked UNSAT, with anything outside the supported
subset (bare loops, `Switch`, vectorized indexing, float-valued indices) reported explicitly rather
than silently skipped. The z3 binary, its bounds-obligation encoding, and CubeCL's front-end
expansion are recorded as trusted for this claim, since the proof is about the IR and codegen below
it stays covered only by the tested differential claims. Kernel identity now also carries an
IR-level content hash alongside the source-level one, so evidence goes stale on either kind of
drift. `axpy_off_by_one` REFUTES with a counterexample exhibiting the out-of-bounds position, and
`sum_racy`'s bounds PROVE even though its differential check correctly fails — the race is a
distinct, differential finding, never conflated with the bounds claim.

## First release

The first release demonstrates one complete, honest path from kernel intent to executable artifact
and evidence. It is done when:

1. **Contract and identity** — a kernel can be annotated with assumptions, and VeriCL assigns it a
   stable identity; changing the kernel, contract, or toolchain invalidates dependent evidence.
2. **Differential conformance** — generated inputs run against the scalar reference and at least
   one GPU backend, with counterexamples reported on divergence, and `vericl check` enforces this
   in CI.
3. **One proved property** — out-of-bounds freedom is machine-checked for a defined kernel subset
   (affine index expressions, bounded loops, known launch dimensions), with kernels outside the
   subset rejected explicitly rather than silently approximated.
4. **Honest examples** — at least two example kernels (one Substrate-motivated but independently
   written, one generic, e.g. a counter-based RNG or prefix sum), each paired with a deliberately
   defective twin whose defect the appropriate check catches and reports usefully.

Breadth — more backends, more property classes, richer numeric models, proof assistants — is
explicitly deferred. A narrow path with honest claims is sufficient.

## Relationship to prior art

- **GPUVerify** — the closest neighbor: static race and bounds analysis for CUDA/OpenCL, now
  essentially unmaintained and disconnected from any Rust or CubeCL workflow. VeriCL's checked
  property list starts narrower, but its evidence is bound to a live, multi-backend source of
  custody rather than a one-shot analysis.
- **Alive2 / translation validation** — validates compiler transformations; VeriCL does not verify
  CubeCL's codegen and records it as trusted instead. Translation validation of CubeCL backends
  would shrink that trusted boundary and is a natural later stage.
- **Verus, Kani** — Rust-level verification tools. Because the reference execution is ordinary
  Rust, these are candidate engines for proving properties of the reference itself in a later
  release, without changing VeriCL's core concepts.
- **Exo, Halide** — correct-by-construction scheduling for kernels authored in their own
  languages; VeriCL instead meets CubeCL developers in the language they already use and checks
  after the fact.

## Relationship to Substrate

Substrate is an early adopter supplying kernels with real demands around determinism, indexing,
replay, and numerical comparison. VeriCL contains no RF-specific concepts, does not require
Substrate, and must demonstrate its value on at least one unrelated example before claiming
general usefulness. Substrate-specific policy lives in Substrate or an integration layer.

Substrate kernels inform requirements and are dogfooded against VeriCL privately; proprietary
Substrate kernel implementations are never committed to this repository. Every example in the
public validation suite is generic or independently written — "Substrate-motivated" means a
re-derived kernel exercising the same shape, never a copy.

## Non-goals

- Verifying arbitrary Rust programs, or anything that is not a CubeCL kernel.
- Verifying CubeCL's compiler backends, drivers, or hardware — these are trusted and recorded as
  such.
- Guaranteeing bit-identical floating-point results across backends without explicit per-kernel
  support and evidence for that claim.
- Proving performance or algorithmic appropriateness.
- Recovering intent from arbitrary existing kernels automatically.
- Hiding assumptions to present a simpler correctness badge.

## Open decisions

- Whether the scalar reference execution is a derived interpretation of the cube function or a
  macro-generated twin function — decide when implementation reveals which stays honest with less
  machinery.
- The floating-point comparison model beyond declared per-kernel tolerances.
- The exact supported kernel subset for the bounds checker, and how it grows.
- Report format details; whether evidence manifests are committed or regenerated in CI.
- Whether later property classes (race freedom on shared memory, reduction-order sensitivity) come
  before or after a second proved property on the reference side via Kani/Verus.

Material choices get recorded with their alternatives and the claim boundary they create.

## Naming

**VeriCL** = verification for CubeCL. The `-CL` suffix deliberately ties the name to the substrate
this project committed to rather than staying backend-neutral; it was chosen only after the
CubeCL-only scope (see "Locked decisions") was locked in. The tagline — *one kernel contract,
equivalent implementations* — is now literal: one annotated CubeCL kernel, with its reference
execution and GPU realizations demonstrably equivalent under stated assumptions.

The project's working name during early, backend-neutral exploration was **Equik**.
[docs/ideation-charter.md](docs/ideation-charter.md), linked above under "Status", predates the
rename and still refers to the project by that name — it is an archived historical document and
is left as originally written rather than updated to match.
