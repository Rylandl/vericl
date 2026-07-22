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
| CI story | A `vericl check` command that fails on missing, stale, or mismatched evidence |

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
    compare(abs = 1e-4)
)]
#[cube(launch)]
pub fn axpy(alpha: f32, x: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
    }
}
```

From this single definition VeriCL derives, in a generated `axpy_vericl` module: the untouched
CubeCL kernel; a sequential scalar `reference` twin (`ABSOLUTE_POS` becomes a loop variable,
`&Array<T>` becomes `&[T]`) sharing no CubeCL machinery; the `assumes` clauses as an executable
`check_assumes` predicate; and a `SOURCE_HASH` identity that evidence binds to. Kernels using
constructs the twin cannot model (`UNIT_POS`, `SharedMemory`, `plane_*`, `comptime`, vectors,
`return`) are rejected at compile time rather than silently approximated.

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
