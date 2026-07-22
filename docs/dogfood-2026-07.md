# Dogfood findings — July 2026

VeriCL was run against a private production CubeCL codebase (22 kernels, RF signal-processing
domain; implementations stay private per the Substrate policy in the README — this document
records only generic construct classes and counts). Method: survey every kernel's shape against
VeriCL's v0 gates; annotate a coverage-selected subset in a private sibling workspace; run the
full suite (differential on wgpu/Metal + SMT bounds proofs) where possible; record exact
rejection errors where not.

## What worked on real code

- A counter-based RNG primitive passed end-to-end: bit-exact differential across all sizes on
  the production backend (wgpu/Metal — which the codebase's pre-existing validation had never
  exercised; it validates via cubecl-cpu at f64 against vendor implementations), plus a
  4-obligation z3 `proved` bounds claim.
- A second RNG kernel **independently rediscovered the wrapping finding**: its key-schedule
  addition is intended to wrap, the derived twin's checked arithmetic panicked, and the
  `wrapping` contract clause resolved it. The clause was designed from a synthetic example and
  validated itself against production code unprompted.
- The prover was honestly `OutOfSubset` (loop-carried accumulator) rather than falsely `proved`
  on the kernel it couldn't cover.

## Subset gaps, ranked by how many of the 22 kernels each blocks

Tier 1 — macro gates (compile-time rejection):

| Gap | Blocks | Notes |
|---|---|---|
| Generic kernels (`<F: Float>`) | 20/22 | The dominant gap by far; production kernels are float-generic almost universally |
| Kernel composition (calling other `#[cube]` fns) | 16/22 | Twin derivation has no route through helper calls |
| `#[comptime]` parameters | 15/22 | Used for unroll counts, feature toggles, emitter counts |
| `usize` scalar params | 0 today, next wall | Every current use hides behind `#[comptime]`; surfaces the moment that gate lifts |
| Shared-memory topology (`UNIT_POS`, `CUBE_DIM`, `SharedMemory`, `sync_cube`) | 2/22 directly | The reduction-kernel shape; both also blocked by the gates above |

Tier 2 — prover gates (twin + differential fine; proof honestly unavailable):

| Gap | Shapes | Notes |
|---|---|---|
| Loop-carried accumulators | 8 | Rejection is sound but coarse: could be refined to reject only when carried state feeds an index |
| `/`, `%`-derived indices (flat 1-D → row/col decode) | 7 | QF_LIA models nonneg div/mod fine; this is implementable, not fundamental |
| Array-value-dependent indices (offset tables / gather) | ≥5 | Needs element-range assumptions (e.g. quantified assumes) to be provable |

Notable non-findings: zero uses of `Tensor`, `Line`/`Vector`, `Slice`, `plane_*`, or `Atomic`;
all dispatch is 1-D in-kernel. Earlier roadmap speculation about Tensor/2D support is
**withdrawn** — demand-driven scoping working as intended.

Latent soundness gap found and fixed: `terminate!()` was absent from the banned-construct list;
outside `#[cube]` it expands to an empty block, so a twin would silently fall through a
guard. Unreachable today (all users also hit other gates) but now banned explicitly.

## Candidate public example kernels (clean-room, generic)

1. `flatten_decode_scale` — `/`,`%` decode of `ABSOLUTE_POS` into (row, col) then an
   axpy-shaped write; pins the div/mod prover boundary.
2. `gather_copy` — `output[i] = input[offsets[i]]` with element-range assumes; pins the
   value-dependent-index boundary.
3. `block_sum_reduce` (aspirational) — minimal shared-memory tree reduction; the design target
   for lifting the topology gate.

## Resulting roadmap order

1. Generic kernel support via an `instantiate(...)` contract clause (monomorphized twin,
   launch, and IR at declared concrete types) — unblocks 20/22.
2. `#[comptime]` parameter pinning through the same clause — unblocks most of the 15/22.
3. Kernel composition (annotated helpers contribute twins) — unblocks 16/22.
4. Prover: div/mod index modeling (cheap), loop-carry refinement, then value-dependent
   indices via quantified assumes.
5. Shared-memory reduction shape — last, hardest, and the gateway to race-freedom proofs.
