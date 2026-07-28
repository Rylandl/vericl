# Can I bring this kernel?

> A per-kernel-class answer, checked against the code rather than the ambition.

This page exists to stop you wasting an afternoon. VeriCL's supported subset is narrow and
deliberately so — every construct outside it is a compile error rather than a silent approximation
(see [the rejection reference](guide.md#12-reading-rejections)). But "narrow" is not useful
guidance. What you want to know is whether *your* kernel is the kind of kernel this tool has
anything to say about.

So: the rows below are kernel classes a working GPU programmer recognizes, not VeriCL's internal
feature list. Every cell is backed by an example in `crates/vericl-examples/src/lib.rs`, a test in
`crates/vericl-examples/tests/`, or a committed evidence entry — cited by name. Where a claim is
weaker than it looks, the caveat is on the row, not in a footnote.

---

## The 5-second version

**Bring it if it is a 1-D elementwise, gather, stencil, RNG/hash, or shared-memory tree-reduction
kernel over `Array<T>`.** Those are supported, exercised, and carry committed evidence.

**Image-space 2-D/3-D kernels are supported as of the dispatch milestone** — elementwise,
transpose and *branch-free clamped* stencils, opted in with
`dispatch(cube_dim = (16, 16), extents = (w, h))`. Read the row and
[its section](#image-space-2-d--3-d-dispatch) before you assume that covers your image kernel: 2-D
shared-memory tiles are still rejected, and the enabling length assume is binary, so a
`w x h x d` volume index is differential-only.

**Do not bring it yet if it needs 2-D shared-memory tiles, atomics, `plane_*` subgroup ops, or
`Tensor`/`View`/`cmma` tiling.** The first three are on the gap-closure plan below; the fourth is
deliberately out.

---

## How to read the columns

These four words mean four different things, and conflating them is the failure mode this whole
project is against.

- **Differential-tested** — the kernel ran on a real backend and agreed, within your declared
  tolerance, with a scalar twin derived from the same source, on generated inputs. This is an
  observation about the inputs that were drawn. It is not a proof, and because the twin is derived
  from your kernel, a bug present in both is invisible to it.
- **Bounds-proved** — `smt-oob-freedom`: z3 discharged, over the CubeCL IR, that no in-bounds input
  produces an out-of-range buffer access. Quantified over all inputs satisfying your `assumes(...)`.
- **Race-proved** — `smt-race-freedom`: a GPUVerify-style two-thread symbolic reduction showed no
  intra-phase write-write or read-write collision, and no barrier divergence. **Only cooperative
  kernels get this.** For every other row it reads *not checked* — which is a real gap, not a
  formality; see [Races outside cooperative kernels](#races-outside-cooperative-kernels).
- **Status** — supported / partial / planned / out.

Scope of the whole table: **CubeCL 0.10 pinned** (`cubecl = "=0.10.0"`), backend `wgpu<wgsl>` with an
opt-in `cubecl-cpu` second lane (`--features cpu`). Committed evidence today is **37 entries across
four manifests, 132 machine-checked obligations, 22 of them race obligations**.

---

## The matrix

| Kernel class | Differential-tested | Bounds-proved | Race-proved | Status |
|---|---|---|---|---|
| **Elementwise map, 1-D** (`Array<T>`) | yes — `axpy`, `select_mode`, `flatten_decode_scale`, `index_ramp_map`, `bernoulli_indicator_map` | yes | not checked | **supported** |
| **Vectorized elementwise** (`Array<Vector<P,N>>`) | yes — `vec_add` (evidence), `vec_scale`, `vec_madd` (test only) | `vec_add` only | not checked | **partial** — one width pinned per contract; only width 4 is exercised; no cross-lane ops |
| **Gather / permutation** (read indirection) | yes — `gather_copy`, `slice_gather_copy`, `nested_gather` | yes, *only with an element-range assume* | not checked | **supported** for reads — **injectivity is not expressible**, so a permutation used as a *write* index can be proved in-bounds and still be wrong |
| **Windowed / stencil, 1-D via slices** | yes — `fir3`, `offset_window`, `windowed_slice_sum`, `windowed_helper_kernel`, `slice_scale_inplace` | yes, *only with a length-relationship assume* | not checked | **supported** — overlapping mutable slices are caught by rustc E0499, not a VeriCL message |
| **Tree / grid-stride reduction** (shared memory) | yes — `block_sum_reduce`, `emitter_reduce` (evidence); `grid_stride_reduce`, `comptime_window_reduce`, `composed_sq_reduce` (test only) | yes (8 obligations each) | **yes** (11 obligations each) | **supported** — opt-in `cooperative(cube_dim = N)`; **1-D only**; power-of-two `cube_dim`; one recognized tree-loop shape |
| **RNG / hash / bit-mixing** | yes — `xorshift_step`, `mix_u32`, `lcg_map`, `counter_split_map`, `unit_interval_map`, `mul_hi_map`, `uniform_value_map` | yes | not checked | **supported** — integer overflow requires the `wrapping` clause |
| **Scatter-add / histogram** (atomics) | no | no | no | **PLANNED** — [M-B](#the-gap-closure-plan). `Atomic*` rejected at compile time today |
| **Image-space 2-D dispatch — elementwise / transpose** | yes — `elementwise2d_scale`, `transpose2d`, `topology_report2d` (evidence, six image shapes) | yes | not checked | **supported** — opt-in `dispatch(cube_dim = (Wx, Wy), extents = (w, h))`; needs an `A.len() == (w as usize) * (h as usize)` assume; flat `ABSOLUTE_POS`/`CUBE_POS`/`CUBE_COUNT` rejected inside the clause |
| **Image-space 2-D stencil / blur** (branch-free clamp) | yes — `box_blur3x3` (evidence, six image shapes, bit-exact) | yes (10 obligations) | not checked | **supported** — the clamp must be `u32::min`/`u32::max`, **not** an `if`; an `if`-based clamp is tainted by branch write-taint and is `OutOfSubset` |
| **3-D dispatch** (`cube_dim` 3-tuple) | yes — `elementwise3d_scale` (test only, six volume shapes) | **no** | not checked | **partial** — the launch, twin and per-axis leaves are all rank-3; but a `w*h*d` length fact is not expressible (the product assume is binary), so a volume index is `OutOfSubset` |
| **2-D shared-memory tiles** (tiled matmul, separable filters with a tile) | no | no | no | **PLANNED** — `dispatch(...)` and `cooperative(...)` are mutually exclusive in v1, with a targeted error. Measured: the intra-cube half is cheap, the inter-cube write-disjointness half times out in z3 at 180 s and needs a new pattern recognizer ([design §8](design-2d-dispatch.md)) |
| **Subgroup / warp reductions** (`plane_*`) | no | no | no | **PLANNED** — [M-C](#the-gap-closure-plan). `plane_*` rejected at compile time today |
| **Tiled matmul / conv / attention** (`Tensor`/`View`/`cmma`) | no | no | no | **out of scope**, with rationale below. `View`/`Layout` and `Tensor` params rejected with targeted errors; **`cmma` is not** — it fails downstream instead |
| **Framework-generic trait kernels** | element-type generics only — `axpy`, `fir3`, `gain_kernel`, `vec_add` | same | not checked | **partial** — `<F: Float>` pinned by `instantiate(...)` works; a user-defined `#[cube] trait` does not, and is not rejected with a targeted error |
| **Struct-arg kernels** (`cube_struct!` / `config!`) | yes — `uniform_value_map`, `stage_window_sum`, `accum_blend_map`, `config_window_sum`, `config_mode_scale` | yes | not checked | **supported** — scalar fields only; buffer-valued fields deferred; the identity traits are unsealed and forgeable |
| **f64 kernels** | **cubecl-cpu lane only** — `axpy_f64` | yes (3 obligations) | not checked | **partial** — WGSL has no f64 and the wgpu lane is *silently wrong*, not merely absent |

---

## Row detail and caveats

### Elementwise map, 1-D

The core case. `&Array<T>`/`&mut Array<T>`, affine `ABSOLUTE_POS` indexing, bounded `for`, `match`
(lowered to `Branch::Switch` — `select_mode` proves 6 obligations across three arms), symbolic `/`
and `%` (`flatten_decode_scale`), and `#[comptime]`/generic parameters pinned via `instantiate(...)`.
All suite-wired with committed evidence and re-verified on every `cargo test`.

An independent IR interpreter cross-checks the twin against the same `KernelDefinition` the prover
consumes, bit-for-bit, for 13 of these kernels (`tests/interp_crosscheck.rs`). That shrinks
*model-fidelity* risk empirically. It is still a `tested` observation, not a proof — see
[docs/interpreter.md](interpreter.md).

### Vectorized elementwise

`Vector<P, N>` as an `Array` element type. The twin uses `vericl::Line<T, W>`, a lane array whose
every operation is a per-lane map, because the real host-side `Vector` stores only one element and
is useless as a reference.

- **Width.** There is no fixed set of allowed widths — the requirement is that you *pin* one, with
  `instantiate(N = W)`, and one width per contract is the rule. But **4 is the only width with GPU
  ground truth or any test coverage today** (every committed vector example and
  `tests/line_shim_gpu_ground_truth.rs` pin 4). Treat other widths as unexercised.
- **Only `vec_add` carries evidence.** `vec_scale` and `vec_madd` are covered by
  `vector_conformance_wgpu`/`_cpu` but have no committed bounds proof.
- **No cross-lane operations.** `dot`, `magnitude`, `normalize`, `VectorSum` are rejected by name —
  they are order-sensitive and need their own ground-truth and tolerance story.
- **Also rejected:** integer-vector outputs (the differential compares f32/f64 lanes only), a
  `Vector` array mixed with a plain scalar `Array`, a `Vector` array plus a runtime struct parameter,
  vector `wrapping`, and vectors in cooperative kernels.
- **Measured backend finding:** Metal does not correctly-round f32 `/` (up to 1 ULP against the
  host). Recorded in the shim ground-truth test, not hidden in a tolerance.

### Gather / permutation

Read-side indirection works and proves, *provided you declare the element range*. This is not
incidental: `gather_copy_is_not_provable_without_element_assume` is a standing negative control, and
`gather_oob` — whose declared element bound is looser than the buffer length — is `Refuted` with the
offending element symbol pinned.

**The caveat that matters.** VeriCL has no way to express that a table is *injective*. If your
kernel uses a loaded value as a **write** index (scatter through a permutation), the bounds proof
still discharges, the kernel is genuinely in-bounds, and it can still be wrong — two threads land on
the same output slot. This was measured on real code: a table-loaded output slot collided for roughly
half of drawn inputs, 16 of 32 elements diverged, worst case ~2.1e9 ULP. The differential caught it;
nothing proved it. A permutation/injectivity assume form is a recorded residual and is
[explicitly not in the gap-closure plan](#explicitly-out).

### Windowed / stencil, 1-D via slices

Dynamic-offset `x.slice(i, i+4)`, iteration over a slice, `.to_slice()`, `&Slice<F>` as a
`#[vericl::helper]` parameter, and the mutable write path via `y.slice_mut(...)`.

- **You must declare the length relationship.** `y[i] = x[i] + x[i+4]` is only provable with an
  assume tying `x.len()` to `y.len()`; `gen(len(x = n + 4))` generates inputs that satisfy it.
  `offset_window_is_not_provable_without_relationship` is the negative control.
- **Aliasing is delegated to rustc, by design.** Overlapping simultaneously-live `&mut` subslices
  surface as borrow-checker **E0499**/**E0502** on the generated twin, not as a VeriCL-authored,
  buffer-named diagnostic. That is the intended catch and it is sound; the diagnostic quality is
  deferred work. Disjoint simultaneously-live mutable slices (the `split_at_mut` shape) are rejected
  by the same mechanism even though they are safe — recognizing them is also deferred.
- Slice type-punning and reinterpret methods (`downcast`, `try_cast_slice`, `with_vector_size`, …)
  are rejected by name; a core `Slice` is not a launch argument.

### Tree / grid-stride reduction (shared memory)

The only class that gets a **race proof**, and the strongest row in the table.

Opt in with `cooperative(cube_dim = N)`. The clause and the topology must agree in both directions —
using `SharedMemory`/`sync_cube`/`UNIT_POS` without the clause is an error that names the fix, and
declaring the clause on a kernel with no cooperative topology is also an error (an unused
cooperative declaration is a contract lie).

Constraints, all enforced:

- **1-D only.** `COOP_ALLOWED` is exactly `ABSOLUTE_POS`, `UNIT_POS`, `CUBE_POS`, `CUBE_DIM`,
  `CUBE_COUNT`, `SharedMemory`, `sync_cube`, `terminate`. Every `_X`/`_Y`/`_Z` variant stays banned
  under the clause, as do `sync_units`, `sync_storage`, `plane_*`, `Atomic*`.
- **`cube_dim` must be a power of two and non-zero.** The tree reduction's halving only covers a
  block cleanly for a power of two. This is checked when the value is an integer literal; a
  non-literal expression is left to the launch, so the check is best-effort by construction.
  Launching at a `cube_dim` other than the pinned one panics.
- **One `SharedMemory` tile**, one non-cooperative accumulation loop before the first barrier, and
  **one recognized tree-loop shape** (`while half > 0 { …; sync_cube(); half /= c }`). A differently
  shaped tree loop is `OutOfSubset` — never a wrong `Proved`.
- **Barriers must be visible at the kernel's top level.** `SharedMemory` or `sync_cube` inside a
  `#[vericl::helper]` is rejected unconditionally. The prover cross-checks the source barrier count
  against the helper-inlined IR count, so a helper that hides a barrier is caught.
- **No barrier divergence.** A `sync_cube()` under a thread-varying condition, or in a loop with a
  thread-varying trip count, is `OutOfSubset`. Uniform *conditional* barriers (`if CUBE_POS < n`) are
  rejected too — deferred, not unsound.
- **Per-thread state cannot live across a barrier.** Only pure topology aliases (`let tid = UNIT_POS
  as usize`) may cross; a genuinely stateful local is `OutOfSubset`.
- **Uninitialized shared reads panic.** The twin's `SharedTile` tracks written-ness per cell and
  panics on a poison read (`shared_read_before_write`), rather than reading a plausible zero.
- **`grid_stride_reduce` is excluded from the suite** — it uses the `CubeCount` builtin, which
  cubecl-cpu does not support. It is bit-exact against wgpu in `tests/cooperative.rs` but has no
  evidence entry.
- **`prove: false` does not silently downgrade.** The cooperative fallback manifest carries an
  explicit `assumed` / `intra-phase-race-freedom` claim with `status: "declared"` that the `tested`
  claim depends on. There is no such thing as a quietly green cooperative pass.

### RNG / hash / bit-mixing

Well covered, and several kernels compare at `exact` or `max_ulp = 0` rather than a tolerance:
xorshift, Murmur3 `fmix32`, an LCG step, a two-word counter split, `u32 → [0,1)` uniform, `mul_hi`,
a Bernoulli indicator, and a uniform value map.

- **Integer overflow needs the `wrapping` clause.** WGSL wraps; a debug-mode Rust twin panics. The
  clause is per-kernel and also available per-helper, and it is recorded in the contract.
- The host shims underneath (`cast_from`, `mul_hi`, `fma`) are the load-bearing pieces and are pinned
  bit-for-bit against the real intrinsics on both lanes by `tests/host_shim_gpu_ground_truth.rs`.
  Measured: casts and `mul_hi` are bit-exact with zero divergences; `fma` over 21996 triples is
  exact on cubecl-cpu everywhere, and on wgpu/Metal outside a 4974-triple flush-to-zero domain that
  the local model reproduces exactly.
- `vericl::rng` (`SplitMix64`) is the *host* input generator, not a verified kernel. It carries no
  claim in any manifest and never runs on a device; it is a trusted harness component.

### Scatter-add / histogram (atomics) — PLANNED

Rejected today: any identifier with the `Atomic` prefix trips the ban list and produces the generic
out-of-subset error. (One precision note: the prefix is capital-`A`, so `Atomic::fetch_add` — the
spelling the ecosystem actually uses — is caught by it, while a hypothetical lowercase free
`atomic_add(...)` would instead be caught by the undeclared-call rejection. Both are rejections;
they are different messages.)

This is [M-B](#the-gap-closure-plan). The honesty question it has to answer before it ships is on the
record: a **float** atomic add has no defined accumulation order, so a differential against a
sequential twin with one fixed order compares against one of many legal results. Integer and
bit-exact-associative accumulations do not have this problem. A design that quietly picks the twin's
order and calls the agreement a pass would be exactly the kind of claim this project exists to
prevent.

### Image-space 2-D / 3-D dispatch

Opt in with a `dispatch(...)` clause and index with the per-axis builtins:

```rust
#[vericl::kernel(
    dispatch(cube_dim = (16, 16), extents = (w, h)),
    assumes(inp.len() == out.len(), inp.len() == (w as usize) * (h as usize)),
    compare(max_ulp = 0),
    gen(inp in -100.0..=100.0, out in 0.0..=0.0)
)]
#[cube(launch)]
pub fn box_blur3x3(inp: &Array<f32>, out: &mut Array<f32>, w: u32, h: u32) { … }
```

The clause un-bans `ABSOLUTE_POS_X/Y/Z`, `UNIT_POS_X/Y/Z`, `CUBE_POS_X/Y/Z`, `CUBE_DIM_X/Y/Z` and
`CUBE_COUNT_X/Y/Z` for the axes its `cube_dim` arity enables, and keeps flat `CUBE_DIM` and
`UNIT_POS`. It is required exactly when the body uses one of them, and rejected when it does not —
the same biconditional `cooperative(...)` has.

**X is the fastest-varying axis.** `inp[(y * w + x) as usize]` is a row-major image;
`inp[(x * h + y) as usize]` is *in bounds* and *transposed*, and the proof will not catch it — a
transposed image is a functional bug, not a memory-safety one, and the auto-derived twin mirrors
the same index math, so the differential does not catch it either unless you supply an independent
`reference = fn`. A *different* transposition — swapping the `dispatch(extents = ...)` clause
(`extents = (h, w)`) against a body that guards `ABSOLUTE_POS_X < w`, `ABSOLUTE_POS_Y < h` — **is**
rejected at compile time by the clause/body consistency gate (design §13 risk 6); neither lane can
see it, because the twin grid and the launch derive from the same clause. That gate is conservative:
it checks only axes guarded with the canonical `ABSOLUTE_POS_a < <extent>` form, so an unguarded
axis (or one bounded by `w - 1`, a `min`, …) is a documented residual.

Four things are narrower than "2-D works" would suggest, and each is a measurement:

- **Flat `ABSOLUTE_POS`, `CUBE_POS` and `CUBE_COUNT` are rejected inside the clause.** In a
  multi-axis dispatch `ABSOLUTE_POS` is *not* `CUBE_POS * CUBE_DIM + UNIT_POS`: it linearizes the
  global thread grid, the other linearizes cube-major-then-unit. Swept on hardware over 722 launch
  shapes the identity held in 189 and broke in 533 — 912 of 960 threads violate it at the image-like
  `CubeCount(5,3,1) x CubeDim(8,8,1)`. Index with the per-axis builtins, or drop the clause and use
  the flat 1-D form throughout.
- **The clamp must be branch-free.** `let mut x2 = x; if x + 1 < w { x2 = x + 1; }` writes a mutable
  local inside a branch arm, which branch write-taint correctly taints, so the neighbour index is
  unmodelable. Write `let x2 = u32::min(x + 1, w - 1);` and `let x0 = u32::max(x, 1) - 1;` — both
  lower to arithmetic the prover models exactly.
- **The length assume is the enabling fact, and it is binary.** `abs_y * w` has no Euclidean parent
  the way a flat kernel's `row * w` does, so without `A.len() == (w as usize) * (h as usize)` its
  no-overflow side-obligation is unprovable and *every* 2-D kernel that indexes an array is
  `OutOfSubset`. It must be written widen-then-multiply: `A.len() == (w * h) as usize` multiplies in
  `u32` and then widens, so the executable predicate tests the wrapped product while the model
  asserts the mathematical one — a false `Proved` at `w = 2, h = 2147483649`, rejected by name. And
  because it is binary, a rank-3 volume index (`len == w*h*d`) has no expressible fact and stays
  differential-only.
- **`dispatch(...)` excludes `cooperative(...)`, `Vector<P, W>`, and a runtime `cube_struct!`
  parameter**, each with a targeted error naming the reason.

**Honest reach.** Of the 464 surveyed ecosystem device items, 39 name 2-D topology and **1** is
sole-blocked by it; of the 22 private dogfood kernels, 2 are blocked and **0** solely. This is a
capability-and-soundness milestone, not a coverage one — it is the shape external users most expect
to work, and it is one of the two remaining walls in the private corpus, but a page of green image
kernels is more persuasive than the number deserves.

Design and measurements: [docs/design-2d-dispatch.md](design-2d-dispatch.md).

### Subgroup / warp reductions (`plane_*`) — PLANNED

Rejected today by the `plane_` prefix, always.

This is [M-C](#the-gap-closure-plan), and its honesty question is also already on the record: the
plane/subgroup width is **decided by the device**, not by the kernel. A sequential twin cannot model
a plane reduction without committing to a width, and a width the device does not honor makes the
twin a different function from the kernel. Whatever the design settles on, the pinned width has to be
part of the recorded contract, and a launch on a device with a different width has to fail rather
than pass.

### Tiled matmul / conv / attention (`Tensor` / `View` / `cmma`) — out

Three different mechanisms, and they are not equally strong:

- **`View`/`Layout`** — 28 identifiers banned by name with a targeted error, swept from cubecl-std
  0.10 and pinned by a test that asserts every one of them is rejected.
- **`Tensor`** — a `&Tensor<f32>` *parameter* is rejected by parameter classification with a message
  naming the supported parameter shapes. `VirtualTensor`, `AsTensorView`, `AsTensorViewMut` are on
  the ban list.
- **`cmma`** — **not rejected by a targeted VeriCL error.** There is no `cmma` string anywhere in the
  macro crate. A `cmma::execute(...)` call is a multi-segment path and hits a documented residual:
  the twin leaves it untouched, so it fails downstream as a twin compile error or an
  `Unexpanded Cube functions should not be called.` panic at twin run time, not at a VeriCL span with
  an actionable message. Improving that diagnostic is worth doing; the class stays out either way.

**Why out, not planned.** Two independent reasons. First, the `View` machinery is `Arc<dyn>`
dynamic-dispatch coordinate-to-offset layout; modeling it soundly enough for a bounds claim to *mean*
anything is a larger effort than the entire current prover. Second, VeriCL's differential leg
compares against a twin derived from the same source — for a tiled matmul the honest twin is the
naive triple loop, which accumulates in a genuinely different order, so the tolerance story is its
own research project rather than a per-kernel `compare(...)`. Shipping this class would mean shipping
a claim weaker than the ones on this page pretend to be.

### Framework-generic trait kernels

Half of this works and half does not, and the difference is worth stating precisely.

**Works:** plain *type* generic parameters bounded by CubeCL's own traits — `<F: Float + CubeElement>`,
`<F: Float>`, `<N: Size>` — pinned to a concrete type or value by `instantiate(F = f32)` /
`instantiate(N = 4)`. The generic ident is substituted token-wise into the twin; `#[comptime]`
parameters are bound as consts. Exercised by most of the table.

**Does not work:**

- `where`-clauses, lifetime parameters, and const generic parameters, each with a targeted error.
- More than one `instantiate(...)` per kernel — one monomorphization per kernel body in v0.
- **A kernel generic over a user-defined `#[cube] trait`.** There is no trait-item handling anywhere
  in the macro crate and no example. This is the same residual class as `cmma`: a trait-method call
  resolves as a multi-segment path, is left untouched in the twin, and fails downstream rather than
  at a VeriCL span.
- `#[vericl::helper]` applies to free functions only, not to inherent-impl or trait methods.

This last point is the real ceiling for the CubeCL ecosystem, and the survey measured it: of 89
gate-free items in one scoring pass, 71 were impl/trait items — and the survey's own caveat is that
every impl/trait reach number is an *over*-estimate, because the signature scan does not measure
`&self` receivers at all. Read any "VeriCL could annotate N ecosystem items" number with that in mind.

### Struct-arg kernels

Both positions work, and both are soundness milestones rather than conveniences — before them, a
config type's method body could change what the kernel computed while leaving the recorded identity
bit-identical.

- **Runtime struct arguments** — declare the type in a `vericl::cube_struct! { … }` block. Suite-wired:
  `uniform_value_map`, `stage_window_sum` (nested struct + a `#[cube(comptime)]` field),
  `accum_blend_map` (a struct literal built in the kernel body).
- **Struct-typed `#[comptime]` configs** — declare the type *and every one of its impl blocks* in a
  `vericl::config! { … }` block. Suite-wired: `config_window_sum`, `config_mode_scale`.

Caveats:

- **Scalar fields only** — `f32/f64/u32/i32/u64/i64/usize/bool`, a struct declared in the same block,
  or a `#[cube(comptime)]` integer/bool/unit-enum field. **Buffer-valued fields (`Array`, `Tensor`)
  are deferred** with a targeted message explaining what they would need.
- By value or `&P`; **`&mut P` is rejected** (it would declare an output that cannot exist while
  buffer fields are deferred).
- **Impl blocks must be inside the declaration block.** An `impl` written outside it is invisible to
  the identity hash and to every gate. This is a pre-registered residual with tests, not a fixed
  hole; the differential lane is the only backstop, and for `config!` the twin at least panics
  loudly.
- Field-type paths must be single segment (a module alias could otherwise make an `f32` field pass
  the integer check and get the integer draw path).
- `cfg_attr` is rejected outright in both declaration macros — it re-spells every by-name gate at
  once, and one such split was measured producing a false `Proved`.
- A comptime **enum field** compiles and enters the compilation argument (two pins are two compiled
  kernels), but the v1 subset gives the kernel *body* no way to branch on it.
- **`StructIdentity` and `ConfigIdentity` are public, unsealed traits.** A hand-written impl claiming
  a constant hash is a complete identity bypass — pinned as an executable fact by
  `forged_struct_identity_is_a_complete_bypass`. This is not the threat model (the guarantee is "an
  author who does not lie gets an identity that moves when the meaning does"), but you should know
  it is there.

### f64 kernels

Supported in the type system, the twin, input generation at full 53-bit precision, and the tolerance
record — with one platform fact that dominates everything else.

**WGSL has no f64, and the failure is silent.** CubeCL 0.10 compiles and launches an f64 kernel on
wgpu/Metal with no compile error and no panic, then produces garbage — not an f32 demotion, genuinely
wrong values. `tests/f64_wgpu_unsound.rs::f64_axpy_silently_diverges_on_wgpu` is a standing tripwire
that *requires* divergence: if wgpu ever gains real f64, that test fails and the cpu-only assumption
gets re-examined.

Consequences:

- The only honest lane is `cubecl::cpu::CpuRuntime`, behind `--features cpu`.
- cubecl-cpu shares CubeCL's front end with the kernel under test, so **the f64 suite records the
  weak lane claim** — derived from its `runtime: cubecl::cpu::CpuRuntime`, not declared by hand — and
  the macro-derived twin is the sole independent leg. This is recorded in the manifest, not assumed
  away.
- A kernel with both f32 and f64 `&mut Array` outputs is rejected — one `compare(...)` mode cannot
  honestly serve two float precisions.
- Exactly one f64 example exists (`axpy_f64`). Treat f64 breadth as unexercised.

---

## Races outside cooperative kernels

Worth its own heading, because the *not checked* in the race column is load-bearing.

VeriCL discharges `smt-race-freedom` **only** for kernels carrying `cooperative(cube_dim = N)`. An
ordinary 1-D elementwise kernel that writes `y[0] += x[i]` from every thread is a data race, and
VeriCL will happily prove it in-bounds — the example `sum_racy` does exactly this, and its bounds
obligations discharge cleanly. It is caught by the differential lane diverging, which is a
probabilistic catch on drawn inputs, not a proof.

The usual reason this is fine is that a one-output-slot-per-thread kernel cannot race by
construction. But that is *your* argument about *your* kernel, and VeriCL is not currently making it
for you. If your non-cooperative kernel has any cross-thread write aliasing, no claim on this page
covers it.

---

## The gap-closure plan

VeriCL's supported subset was ranked by two measured corpora — a private production DSP codebase and
a 464-item CubeCL ecosystem survey. That ranking was right for what it measured and wrong as a guide
for outside users: it ranks by what those two corpora happen to contain, and neither contains a
histogram. The plan below deliberately ranks by *recognizability to a working GPU programmer*
instead, and says so rather than retrofitting a survey number to justify it.

No dates. Each milestone lands with the same gates as the ones before it: examples, negative
controls, committed evidence, and an adversarial review.

| | Milestone | State |
|---|---|---|
| **M-A** | **2-D / 3-D dispatch** — image-space kernels, the `_X`/`_Y`/`_Z` position builtins | **landed** — [docs/design-2d-dispatch.md](design-2d-dispatch.md); the deferred half (2-D shared-memory tiles) is M-E below |
| **M-B** | **Atomics** — scatter-add and histogram | queued; must resolve float-atomic-add ordering honesty first |
| **M-C** | **`plane_*` subgroup reductions** | queued; must resolve the device-decided-width question first |
| **M-D** | Loop `break` semantics on the twin side, and struct buffer fields | queued |
| **M-E** | **2-D cooperative tiles** — `dispatch(...)` x `cooperative(...)`, i.e. tiled matmul's shape | queued with its cost already measured: the intra-cube race obligation discharges in <10 ms, the inter-cube one times out at 180 s and needs a 2-D write-pattern recognizer ([design §8](design-2d-dispatch.md)) |

### Explicitly out

- **`Tensor` / `View` / `cmma` tiling** — rationale [above](#tiled-matmul--conv--attention-tensor--view--cmma--out).
- **A permutation/injectivity assume form** — the scatter-correctness gap in the gather row. Recorded,
  measured on real code, not scheduled.

---

## If your kernel is not on this page

Two things are true at once: the subset is narrow, and the rejection is the product. A construct
VeriCL cannot model faithfully produces a compile error naming the construct and pointing at
[the rejection reference](guide.md#12-reading-rejections) — not a twin that quietly approximates it
and a green test run.

Read [What VeriCL does not do](guide.md#13-what-vericl-does-not-do) before you rely on a green run,
whatever row you are in.
