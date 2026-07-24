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
| Kernel composition (calling other `#[cube]` fns) | 16/22 | **[implemented, 2026-07]** `#[vericl::helper]` + a kernel-side `uses(...)` clause — see `crates/vericl-macros/src/lib.rs`'s composition doc and README "Kernel composition" |
| `#[comptime]` parameters | 15/22 | Used for unroll counts, feature toggles, emitter counts |
| `usize` scalar params | 0 today, next wall | Every current use hides behind `#[comptime]`; surfaces the moment that gate lifts |
| Shared-memory topology (`UNIT_POS`, `CUBE_DIM`, `SharedMemory`, `sync_cube`) | 2/22 directly | The reduction-kernel shape; both also blocked by the gates above |

Tier 2 — prover gates (twin + differential fine; proof honestly unavailable):

| Gap | Shapes | Notes |
|---|---|---|
| Loop-carried accumulators | 8 | **[implemented, 2026-07]** Refined per this table's own suggestion: carried variables are tainted (not the whole loop rejected), so an accumulator whose index/branch expressions don't touch carried state now proves — see `vericl-ir`'s `process_range_loop`/`loop_carried_accumulator_unused_as_index_proves` |
| `/`, `%`-derived indices (flat 1-D → row/col decode) | 7 | **[implemented, 2026-07]** Modeled via SMT-LIB `div`/`mod` (Euclidean) behind a solver-discharged nonzero+nonnegative side-obligation — see `vericl-ir`'s `divmod_int` and the public `flatten_decode_scale` example (candidate #1 below) |
| Array-value-dependent indices (offset tables / gather) | ≥5 | **[implemented, 2026-07]** Element-range `assumes(...)` — `A.iter().all(\|v\| (*v as usize) < B.len())` / `… < N` — let a read `A[i]` produce a value modeled as a fresh symbol bounded by the assume (the ONLY case array contents get a model), so `x[offsets[i]]` and nested `a[b[i]]` gathers prove; a write to `A` invalidates it. See `vericl-ir`'s "Element-range assumptions" + the public `gather_copy` example (candidate #2 below) |

Notable non-findings: zero uses of `Tensor`, `Line`/`Vector`, `Slice`, `plane_*`, or `Atomic`;
all dispatch is 1-D in-kernel. Earlier roadmap speculation about Tensor/2D support is
**withdrawn** — demand-driven scoping working as intended. **[Vector superseded, 2026-07]** the
broader ecosystem survey (`docs/ecosystem-survey-2026-07.md` §1) later found `Vector<P, N>` to be the
**#1 gate incidence** across tracel-ai's own kernel libraries (148/464 device items), so it graduated
from non-finding to a delivered milestone — see the Line/Vector addendum below and
`docs/design-line-vector.md`.

Latent soundness gap found and fixed: `terminate!()` was absent from the banned-construct list;
outside `#[cube]` it expands to an empty block, so a twin would silently fall through a
guard. Unreachable today (all users also hit other gates) but now banned explicitly.

## Candidate public example kernels (clean-room, generic)

1. `flatten_decode_scale` — `/`,`%` decode of `ABSOLUTE_POS` into (row, col) then an
   axpy-shaped write; pins the div/mod prover boundary. **[implemented, 2026-07]** —
   `crates/vericl-examples/src/lib.rs`, wired into `vericl::suite!`; carries both a `tested`
   (differential) and `proved` (2-obligation SMT bounds) claim in `evidence/vericl.json`.
2. `gather_copy` — `output[i] = input[offsets[i]]` with element-range assumes; pins the
   value-dependent-index boundary. **[implemented, 2026-07]** —
   `crates/vericl-examples/src/lib.rs`, wired into `vericl::suite!`; carries a `tested`
   (bit-exact differential — a gather is a pure permutation) and a `proved` (3-obligation SMT
   bounds) claim. The element bound is stated once: it doubles as the `gen(...)` range for
   `offsets` (drawn in `[0, x.len())`), so the differential lane exercises satisfying offset
   tables with no separate `gen(offsets in …)` clause. Its negative twin `gather_oob` (a stale
   constant bound `< 16` looser than `x.len() == 8`) is `Refuted` with the fresh element symbol
   pinned at the boundary — a `conform` demo-defect. A `nested_gather` example
   (`data[inner[outer[i]]]`) pins that element assumes compose across index layers.
3. `block_sum_reduce` (aspirational) — minimal shared-memory tree reduction; the design target
   for lifting the topology gate.

## Resulting roadmap order

1. Generic kernel support via an `instantiate(...)` contract clause (monomorphized twin,
   launch, and IR at declared concrete types) — unblocks 20/22.
2. `#[comptime]` parameter pinning through the same clause — unblocks most of the 15/22.
3. [DONE 2026-07] Kernel composition (annotated helpers contribute twins) — unblocks 16/22.
4. Prover: div/mod index modeling (cheap), loop-carry refinement, then value-dependent
   indices via quantified assumes.
5. Shared-memory reduction shape — last, hardest, and the gateway to race-freedom proofs.

## Addendum (late July 2026, post div/mod + composition milestones)

- **Implicit-invariant finding on a real kernel**: once div/mod indices became modelable, a
  production flat-index-decode kernel's bounds went from unanalyzable to *refuted* — with a
  genuine counterexample (an empty parameter table admits an OOB read). The kernel's in-bounds
  behavior had always depended on caller-side buffer-sizing discipline that nothing in the
  kernel declared. With the invariant stated as contract assumes (`table.len() == <comptime
  count>`), the proof discharges. Exactly the charter's "boundary behavior can be implicit"
  failure class, surfaced on real code by a subset expansion.
- **Composition validated with zero adaptations**: a production inner-loop kernel + its FIR
  helper ran unchanged through helper/uses annotation — differential pass on wgpu, bounds
  Proved (54 obligations) with obligations walked inside the composed helper body. The
  predicted "usize runtime param" wall did not surface.
- New residual: tuple-destructuring of a helper call's return (`let (a, b) = helper(...)`)
  works in launch-kernel twins but not yet in device-fn-calling-device-fn twins
  (`Pat::Tuple` unsupported at that site) — queued.

## Addendum (shared-memory milestone, July 2026)

The production tree-reduction kernel (the `Σ|iq|²` grid-stride reduction shape) was annotated
`cooperative(cube_dim = 256)` + `instantiate(F = f32)` with a full contract and run through the
whole shared-memory path — the phase-split cooperative twin (differential vs wgpu) AND both SMT
proofs (out-of-bounds freedom + two-thread race freedom). **Result: the full triple lands** —
differential pass, `smt-oob-freedom` proved (8 obligations), `smt-race-freedom` proved (8: 3
write-write, 4 read-write, 1 inter-cube single-writer; 3 phases, 2 barrier-uniformity checks) — the
real acceptance test for the milestone passing on the real shape, not just the clean-room example.

Five adaptations were needed, each a genuine subset boundary the exercise surfaced (construct
classes only, per the Substrate policy):

1. **Comptime loop bound → buffer-derived.** The cooperative subset rejects `#[comptime]`
   parameters; a comptime element-count loop bound had to become `let n = buf.len() / K` derived
   from an interleaved buffer length. **New wall**: a cooperative kernel cannot take a comptime loop
   bound — it must be recoverable from a buffer length.
2. **Caller-supplied grid width → `CUBE_COUNT` builtin.** A runtime `num_cubes: u32` scalar that
   must equal the launch cube count is not expressible: the cooperative launch model *owns*
   `cube_count` (it derives it from the case size and sizes the per-cube output to it), and there is
   no way to bind a free scalar parameter to that value. The grid-stride must read `CUBE_COUNT`.
   **New wall**, and the reason the clean-room example was written against `CUBE_COUNT` from the
   start.
3. **Named tile-length const → literal matching `cube_dim`.** A symbolic tile length keyed off
   `CUBE_DIM` is a v1.1 tier; the tile literal must equal the pinned `cube_dim`.
4. **Unguarded single-writer store → explicitly bounds-guarded.** The production store wrote
   `out[CUBE_POS]` under only `tid == 0`, relying on the host sizing the output to exactly the
   launched cube count — an implicit invariant the bounds prover cannot see, so the store was an
   undischargeable OOB obligation until an explicit `CUBE_POS < out.len()` guard was added. Same
   "boundary behavior can be implicit" class as the div/mod dogfood finding above — surfaced again,
   this time on the cooperative store. Both a provability fix and a genuine hardening.
5. **Generic pinned via `instantiate(F = f32)`** (v0 is f32-only; production also runs an f64 host
   oracle lane).

- **Float-associativity finding (predicted, confirmed).** Unlike the clean-room `grid_stride_reduce`
  (a single `data[k]*data[k]` product, bit-exact vs wgpu at `max_ulp = 0`), the real per-sample
  term `re*re + im*im` (two products + an add) is fma-**contracted** by wgpu/naga, so the strict-f32
  phase-split twin diverges by ~1 ULP per contracted op. `max_ulp = 0` failed at n=258 with a single
  1-ULP miss; the honest claim is a relative bound (`compare(abs = 1e-2, rel = 1e-6)`), justified by
  the reduction depth and the fact that a sum of non-negative squares has no cancellation. Exactly
  the §4.6 / risk-6 prediction in `docs/design-shared-memory.md`, and the same finding class as
  axpy's original fma story — the phase model itself introduces no divergence, only the backend
  contraction does.
- **No new wall in the prover.** The div-derived loop bound (`buf.len() / 2`) and the
  complex-interleaved indices (`buf[k*2]`, `buf[k*2+1]`) discharged their bounds through the existing
  div/mod modeling with no changes — the two-thread walk proved the whole real shape in-subset.

## Addendum (element-range / offset-table milestone, July 2026)

The array-value-dependent-index gap was closed with element-range `assumes(...)` (this table's own
suggestion), and validated against a real offset-table shape from the private workspace (construct
classes only, per the Substrate policy):

- **Pure offset-table gather PROVES.** A production coherent-accumulate primitive reads a per-emitter
  source anchor out of a `&Array<u32>` offset table and uses it as a source index. Distilled to its
  value-dependent-index core — `out[i] = source[base_offsets[i]]`, one lane per output sample, the FIR
  conv / Doppler phasor / complex gain / emitter runtime-loop all dropped (each a *separately*-named
  composition or comptime subset gap, none bearing on the index) — it `Proved` in bounds with a single
  element-range assume (`base_offsets.iter().all(|v| (*v as usize) < source.len())`), 3 obligations, no
  prover adaptation beyond stating the assume. The value-dependent index the tool refused to reason
  about now lands on the real shape.
- **The faithful additive anchor surfaces a new implicit-invariant finding.** The real anchor is not a
  pure gather but a *sum*, `base_src = out_idx + base_offsets[e]`. The element-range assume bounds the
  loaded offset (`base_offsets[i] < source.len()`) but says nothing about `out_idx + base_offsets[i]`,
  so the shape is honestly `Refuted` (counterexample: `out_idx` near `out.len()`, offset near
  `source.len()`, the sum overruns). In the production kernel that sum stays in bounds only because the
  resident buffer is host-sized with headroom (`ResidentPipeline`'s `shifted_len` contract) — an
  implicit caller-side invariant nothing in the kernel declares. Exactly the "boundary behavior can be
  implicit" class already found on the div/mod flat-index decode and the cooperative single-writer
  store. The residual: expressing it needs a length *relationship* assume (`out.len() + max_offset <=
  source.len()`), an inequality/sum shape beyond v0's `LenEq`/`LenEqConst`/element-range recognizer —
  queued, and the honest `Refuted` (never a false `Proved`) is the correct v0 verdict meanwhile.
- **Gen ergonomics validated.** Neither distilled kernel needed a `gen(offsets in …)` clause: the
  element bound derives the offset table's generation range (`[0, source.len())`) automatically, so the
  differential lane draws satisfying tables from the single `assumes(...)` statement.

## Addendum (Line/Vector element milestone, July 2026)

The `Vector<P, N>` SIMD element type — the ecosystem survey's #1 gate incidence (148/464 device items,
`docs/ecosystem-survey-2026-07.md` §1) — was delivered for the **vectorized elementwise class**, per
`docs/design-line-vector.md` (V1–V6). At the pinned versions it is `Vector<P, N>` with a **comptime**
width `N` (not the pre-0.10 launch-dynamic `Line<T>`), so it pins per contract exactly as `instantiate`
already pins a generic float. Landed as six milestones:

| Row | Status | Notes |
|---|---|---|
| **Whole-vector 1-D bounds** | **supported** | Proved *unmodified*: whole-vector indexing lowers to `vector_size: 0` (width in the list's `Type`) and `.len()` is line-granular, so the obligation is the scalar one — `N` never enters it. Plus the **one** soundness guard: `is_modeled_int` now requires `vector_size() == 1`, so a `Vector<u32, N>` value (whose *storage* `is_int()`) can never be modeled as a single scalar SMT `Int` (round-8 risk 1). |
| **Pinned lane-array twin** | **supported** | `vericl::Line<T, W>` = `[T; W]`, every op a per-lane map, each **GPU-ground-truth-verified** bit-exact against a real `Vector<_, N>` kernel on wgpu (+cpu) — `tests/line_shim_gpu_ground_truth.rs`. Finding: Metal `f32 /` is not correctly-rounded (≤1 ULP), the same legitimate float divergence a scalar `/` has, covered by `compare(abs=…)`. |
| **Vectorized launch / I/O + gen + compare** | **supported** | Scalar I/O throughout: `gen` draws `lines*W` flat scalars per array, buffers sized `lines*W`, the launch splices `W` as the vectorization, the flat-scalar compare reports a divergence per lane `(line = i/W, lane = i%W)`. `vec_madd` (`a*a+b`) is the explicit FMA-contraction tolerance example. |
| **Per-lane comptime-unroll** | **supported** | A comptime-unrolled `for j in 0..W` affine-in-lane write into a *register* vector proves (constant lane index into a register vector carries no buffer obligation, lane contents tainted). A **data-dependent** (runtime) register-vector lane index, and a per-lane value used to index another array, stay `OutOfSubset` — the only per-lane shape the 148 use is the comptime-unrolled one. |
| **Public example** | **supported** | Clean-room `vec_add` wired into `vericl::suite!` at `N = 4`: `tested` (bit-exact per-lane differential) + `proved` (3-obligation line-granular bounds), the pinned width recorded in the claim config. |
| **Survey-kernel generalization (V6)** | **validated** | The shortlist's already-provable f32 elementwise `to_degrees_map` re-annotated at its real `Vector<f32, 4>` element type in the survey workspace: the full `tested` (per-lane, width 4) + `proved` (2-obligation bounds) pair — "proves the scalar core" becomes "proves the vectorized kernel" for the elementwise class. |

Honest reach (design §0.5, §12): Vector is the #1 *gate incidence* but rarely the *only* gate — only
13/148 items trip it alone, mostly framework impls. v1's value is **generalizing the already-provable
elementwise shortlist to its true vector element type**; the whole-kernel unlock (reductions/matmul)
needs `View`/`Slice` (the #2 gap) + `Atomic` + `comptime!` + `match`, a documented, non-silent boundary.
Deferred with targeted rejections: cross-lane reductions (`dot`/`magnitude`/`normalize`, GPU-defined
summation order), `SharedMemory<Vector>` cooperative reductions, reinterpret-slice (`vector_size≠0`),
vector `cast_from`/`wrapping`, and a single-clause width sweep.
