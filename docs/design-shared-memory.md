# Shared-memory milestone — design (July 2026)

The implementable design for VeriCL's workgroup-cooperative kernel shape: `UNIT_POS`, `CUBE_POS`,
`CUBE_DIM`, `SharedMemory`, `sync_cube`, grid-stride loops, tree reductions. Two deliverables:

- **A. The reference twin for barrier-cooperative kernels** — the current sequential
  loop-over-`ABSOLUTE_POS` twin is meaningless under shared memory (§1). A *phase-splitting*
  transformation replaces it (§4).
- **B. Race-freedom proving** — a GPUVerify-style two-thread reduction over the CubeCL IR, a new
  `proved`/`smt-race-freedom` claim alongside `smt-oob-freedom` (§5).

Everything below marked "validated" was checked empirically against the pinned `cubecl =0.10.0`
(z3 4.x on PATH), the same posture as [docs/ir-research.md](ir-research.md). Probe sources are
preserved in the scratchpad (`shmem_probe.rs`, `*.smt2`, `ir_dumps.txt`) and were run in the
private `vericl-dogfood` workspace (cubecl already compiled there). **Clean-room kernels only** —
no private source was probed; the two motivating private kernels (a grid-stride sum-of-squares
reduction and a multi-receiver row-per-workgroup reduction) are described here by *shape* only,
per the README private-codebase policy.

File:line citations to `crates/vericl-ir/src/prover.rs`, `crates/vericl-macros/src/lib.rs`, and the
`cubecl-ir-0.10.0` / `cubecl-core-0.10.0` source trees are current as of `397cd60`.

---

## 0. Headline recommendation

1. **Twin (A): phase-splitting transformation, macro-derived.** Split the body at each `sync_cube()`
   into barrier-delimited *segments*; the twin runs, per cube, per segment, per `unit_pos`, that
   segment — with cross-barrier per-thread locals promoted to `[T; cube_dim]` and shared memory a
   single per-cube array. **Candidate #2 (thread-loop inversion) is the same transformation seen
   from the AST; they are unified, not rivals.** Candidate #3 (declared external reference) ships as
   a **complement** — the honest fallback for kernels outside the transformable subset, carrying a
   distinct, weaker claim label. **Validated bit-exactly on wgpu/Metal** (§4.6).

2. **Race-freedom (B): two-thread symbolic reduction, new `proved` check.** Duplicate the value
   environment for two symbolic threads `t1 ≠ t2` in one cube; between each pair of barriers, prove
   no write-write / read-write conflict on the same shared index (and on global arrays via
   `ABSOLUTE_POS`-disjointness), plus barrier-uniformity. **All obligation classes validated as
   plain QF_LIA, discharged by z3, with racy negative controls firing SAT** (§5.5).

3. **The A↔B coupling is the crux and must be mechanical.** The phase-split twin is a *faithful*
   reference **only when the kernel is intra-phase race-free** — which is exactly what B proves. A
   `tested` differential claim on a cooperative kernel therefore lists "intra-phase race freedom" in
   its **assumed** section *unless* the `proved`/`smt-race-freedom` claim is present, in which case
   it moves to a discharged dependency. Three honesty tiers, never blurred (§6).

4. **v1 subset = the reduction shape**, pinned by the clean-room `block_sum_reduce` and
   `grid_stride_reduce` probes: 1-D topology, one non-cooperative accumulation loop, one
   uniform-trip-count cooperative tree loop, single-writer global store. `terminate!()`, 2-D
   dispatch, and non-uniform barriers are explicitly rejected with targeted errors and deferred
   (§7).

This resolves the README "open decision" (*"Whether later property classes (race freedom on shared
memory …) come before …"*) — race-freedom is the **gateway**, and roadmap item 6 (*"race-freedom
via two-thread symbolic reduction"*) is this document.

---

## 1. Why the current twin is meaningless under shared memory

The v0 twin (README "contract attribute", `crates/vericl-macros/src/lib.rs` `conformance_case`,
lib.rs:2773) executes the body **once per thread, sequentially, over `ABSOLUTE_POS = 0..num_threads`**,
with each `&Array<T>` a `&[T]` and no cross-thread state. That model has three fatal gaps for a
cooperative kernel:

- **No shared state.** `SharedMemory` is per-*workgroup* storage shared by all its threads. A
  loop-over-`ABSOLUTE_POS` twin has no per-cube arena; each "thread" would see a fresh `tile`, so a
  tree reduction reads back nothing.
- **No barrier semantics.** `sync_cube()` means "every thread in the cube reaches this point before
  any proceeds." Running threads one-at-a-time to completion violates it: thread 0 finishes the
  whole body (including the tree reduction reading `tile[tid+half]`) before thread 1 has written
  `tile[1]`.
- **Per-workgroup outputs.** The output of a reduction is one partial *per workgroup*
  (`partials[CUBE_POS]`), not one value per thread. The flat `ceil(n/cube_dim)`-threads launch model
  (lib.rs:2782) doesn't fit; the harness must launch a chosen `(cube_count, cube_dim)` and size the
  output to `cube_count`.

Accordingly the macro **bans** the whole vocabulary today (`BANNED_IDENTS`, lib.rs:41): `UNIT_POS`,
`CUBE_POS`, `CUBE_DIM`, `CUBE_COUNT`, `SharedMemory`, `sync_cube`, `terminate`, and `plane_*`. The
milestone lifts that ban *for a new cooperative mode* with a genuinely different twin.

---

## 2. IR construct catalog (validated)

Probed by expanding two clean-room kernels through the zero-client `KernelBuilder` recipe
(ir-research.md §1) and walking `def.body`. Full dumps in `ir_dumps.txt`.

### 2.1 Topology builtins

`VariableKind::Builtin(Builtin::…)` (`cubecl-ir-0.10.0/src/variable.rs:100`). The 1-D names appear
directly as index/arithmetic operands, typed `UInt(U32)`:

| Surface | `Builtin` variant | Notes |
|---|---|---|
| `UNIT_POS` | `UnitPos` | thread id within cube; **per-thread leaf** |
| `CUBE_POS` | `CubePos` | workgroup id; **cube-uniform leaf** |
| `CUBE_DIM` | `CubeDim` | block size; **cube-uniform leaf** |
| `ABSOLUTE_POS` | `AbsolutePos` | global thread id; per-thread leaf. `= CUBE_POS*CUBE_DIM + UNIT_POS` in 1-D |
| `CUBE_COUNT` | `CubeCount` | grid width; passed here as a runtime `GlobalScalar` (`num_cubes`) |

The `X`/`Y`/`Z` variants (`UnitPosX`, …) and cluster/plane builtins are **out of subset** (1-D
only). `let tid = UNIT_POS as usize` produced **no cast instruction** — cubecl folds it, and
`Builtin(UnitPos)` appears directly as the `Index`/`IndexAssign` index. So a cooperative walker sees
raw `Builtin(UnitPos)` etc. at index sites, not a renamed local.

Current prover: `value_of` (prover.rs:950) models only `AbsolutePos` (prover.rs:956) and integer
`GlobalScalar` (prover.rs:959) as leaves; `UnitPos`/`CubePos`/`CubeDim` fall through to `_ => None`
(prover.rs:968) → tainted. **These must become modeled leaves.**

### 2.2 Shared memory

`SharedMemory::<f32>::new(256)` → a `VariableKind::SharedArray { id, length: 256, unroll_factor: 1,
alignment: None }` (`variable.rs:73`), registered in `Scope.shared: Vec<Variable>` (hashed into
identity, `scope.rs:84`). **There is no allocation instruction** — the tile appears only as the
`list` operand of `Index`/`IndexAssign`. The `new(size)` argument must be a `usize` literal/const
(`SharedMemory::new(256usize)`; a bare `256` is `i32` and fails to compile — validated).

Indexing is the *same* `Operator::{Index,IndexAssign}` shape as global arrays, distinguished only by
the `list.kind` being `SharedArray` instead of `GlobalInputArray/GlobalOutputArray`:

```
LocalConst{9}  = OP Index      list=SharedArray{id:0,length:256,..} index=Builtin(UnitPos)
SharedArray{id:0,..} = OP IndexAssign index=Builtin(UnitPos) value=LocalConst{12}
```

Current prover: `buffer_of` (prover.rs:651) accepts only `GlobalInputArray/GlobalOutputArray`; a
`SharedArray` list hits the `other =>` arm (prover.rs:655) → `OutOfSubset`. **Validated**: the
current prover on `block_sum_reduce` returns exactly `OutOfSubset { reason: "indexing into
\`SharedArray { id: 0, length: 256, .. }\` (not a global input/output array) is outside the vericl
v0 subset" }`.

Two shared arrays are distinguished by `id` (probe A used id 0; probe B's tile got id 8 — ids share
the global variable counter, they are not a dense per-shared index).

### 2.3 The barrier

`sync_cube()` → `Operation::Synchronization(Synchronization::SyncCube)`
(`cubecl-ir-0.10.0/src/synchronization.rs:10`), an instruction with **no `out`** and no operands.
`SyncPlane`/`SyncStorage`/`SyncAsyncProxyShared` are sibling variants, all **out of subset** for v1.
Current prover: caught by the `_ => taint_out` catchall (prover.rs:403) — a no-op that neither
rejects nor records the barrier; the cooperative walker must instead treat it as a **phase
boundary**.

### 2.4 Loops — both reduction loops are `Branch::Loop`, not `RangeLoop`

`while cond { … }` lowers to a **break-terminated `Branch::Loop`** (`branch.rs`), with a canonical
desugared shape:

```
LOOP {
  c   = <cond>
  nc  = Not c
  IF nc { BRANCH Break }     // leading break-guard
  … body …
  <loop-update>              // e.g. half = half / 2
}
```

Two distinct roles surfaced in the probes:

- **Non-cooperative accumulation loop** (grid-stride phase 1): `while k < n { local += data[k]*data[k];
  k += stride }`. **Contains no `SyncCube`.** Carries an induction var (`k`, `LocalMut`) and a float
  accumulator (`local`, `LocalMut`). Trip count is **data-dependent and per-thread** (different `k`
  start per `ABSOLUTE_POS`) — which is *fine*, because there is no barrier inside.
- **Cooperative tree loop**: `while half > 0 { if tid < half { … }; sync_cube(); half /= 2 }`.
  **Contains a `SyncCube`.** The carried control var `half = CUBE_DIM/2` halves each iteration —
  **cube-uniform** (no `UnitPos` dependence), so the trip count (`log2(CUBE_DIM)`) is identical on
  every thread.

Current prover: `Branch::Loop` is rejected wholesale (prover.rs:803). **Validated**: the current
prover on `grid_stride_reduce` returns `OutOfSubset { reason: "\`Branch::Loop\`
(unbounded/break-terminated loop) is outside the vericl v0 subset" }` (it hits the grid-stride loop
before reaching shared memory).

### 2.5 `terminate!()` (used by the second private kernel, deferred)

`terminate!()` → `Branch::Return` nested inside a structured `if` (validated):

```
c = OP GreaterEqual(CubePos, GlobalScalar(0))   // row >= n_rows
IF c { BRANCH Return }
```

The prover already treats `Branch::Return => Ok(())` (prover.rs:810) as a no-op; the macro **bans**
`terminate` (lib.rs:77) because outside `#[cube]` it expands to an empty block, so a sequential twin
would fall through the guard. For a *cooperative* twin a workgroup-uniform `terminate!()` means
"skip this whole cube," which the phase-split model *can* represent — but only once we can prove the
terminate condition is cube-uniform (else it is barrier divergence). **Deferred to v1.1** (§7.4).

---

## 3. The two motivating shapes (generic description)

Both private kernels — a grid-stride sum-of-squares reduction and a multi-receiver row-per-workgroup
reduction — are the **identical reduction skeleton**, which the clean-room probes reproduce faithfully:

```
tid = UNIT_POS
// PHASE 0: per-thread accumulate a strided slice of a global array into a register `local`
local = 0
k = <start>                       // ABSOLUTE_POS (grid-stride) or tid (block-stride)
while k < n { local += f(global[k…]); k += <stride> }   // stride = CUBE_DIM*CUBE_COUNT or CUBE_DIM
tile = SharedMemory::new(BLOCK)   // BLOCK a comptime power of two, == launch cube_dim
tile[tid] = local
sync_cube()                       // BARRIER 1
// PHASE 1..log2: tree reduction, one barrier per level
half = CUBE_DIM / 2
while half > 0 {
    if tid < half { tile[tid] = tile[tid] + tile[tid + half] }
    sync_cube()                   // BARRIER per level
    half /= 2
}
// FINAL PHASE: single-writer store of the per-cube result
if tid == 0 { out[CUBE_POS] = tile[0] }
```

Differences that scope the milestone tiers:

- The grid-stride reduction is the **v1 target**: grid-stride over `CUBE_DIM*num_cubes`, one partial per cube,
  no `terminate!()`. `block_sum_reduce`/`grid_stride_reduce` clean-room probes match it.
- The multi-receiver reduction adds **v1.1** features: one workgroup per row (`row = CUBE_POS`,
  block-stride by `CUBE_DIM`), a **workgroup-uniform `terminate!()`** padding guard (its own doc
  argues the guard is workgroup-uniform and precedes every `sync_cube`, so barrier-safe — exactly
  the property B must *prove*, not assume), **2-D dispatch** padding, and **helper composition** in
  phase 0 (a per-row sample helper calling a per-tap sample helper, which cube-expansion already inlines into
  the scope, so the walker sees it — the existing composition story, README "Kernel composition").

Neither kernel writes shared memory under a non-uniform barrier; both are intra-phase race-free by
construction. That is the class v1 targets.

---

## 4. Deliverable A — the reference twin

### 4.1 The phase-splitting execution model

Partition the body into a sequence of **segments** `S0, S1, …, Sm` delimited by `sync_cube()` calls
at statement level. Under intra-phase race freedom (B) and barrier non-divergence, GPU semantics
are **exactly**:

```
for cube in 0..cube_count:
    <fresh per-cube shared arrays>
    for seg in 0..=m:
        for unit_pos in 0..cube_dim:
            execute segment `seg` for (cube, unit_pos)
        // implicit barrier between segments
```

The inner "for each `unit_pos`, run the segment" is faithful **because the segment contains no
barrier** (barriers are the segment delimiters), so within a segment thread execution order is
unobservable *iff* the segment is race-free — precisely B's guarantee.

**Per-thread state that crosses a barrier** (a local written in `Si`, read in `Sj>i`) must be
promoted to a `[T; cube_dim]` array indexed by `unit_pos`, since each thread keeps its own copy
across the segment boundary. State confined to one segment stays an ordinary scalar. In the
reduction shape:

- `local`, `k` (phase 0) are per-thread but **consumed within phase 0** (`tile[tid] = local` is
  before barrier 1) → stay scalars in the phase-0 `unit_pos` loop. No promotion needed.
- `tile` is **shared** → one per-cube array, visible to every `unit_pos`.
- `half` crosses barriers but is **cube-uniform** → thread-loop inversion (§4.2) keeps it a single
  scalar hoisted around the per-`unit_pos` segment loops.

### 4.2 Cooperative loops (barrier inside a loop) and candidate #2

A loop that **contains** a barrier cannot be run per-thread-to-completion. The tree loop's body is
`{ if tid<half {…}; sync_cube(); half/=2 }` — one segment before the barrier (the `if`), one after
(`half/=2`). Because the trip count is **cube-uniform**, we invert the thread loop *inside* the
barrier loop:

```
half = cube_dim / 2                       // uniform, single scalar
while half > 0 {                          // outer: uniform trip count
    for unit_pos in 0..cube_dim: run segment-A(unit_pos)   // the `if tid<half` tree step
    // barrier
    half /= 2                             // segment-B (uniform update; run once)
}
```

**Candidate #2 ("thread-loop inversion") is exactly this, and it is the same transformation as
candidate #1 viewed from the AST.** #1 says "split at barriers, iterate threads within each
segment"; #2 says "hoist a `for unit_pos` loop around each segment and promote cross-barrier locals
to `vec![_; cube_dim]`." They produce the identical twin. **They are unified — there is no design
choice between them.** The uniform-trip-count requirement is what makes both well-defined: a
*non-uniform* barrier loop (some threads iterate more than others) is barrier divergence — rejected
(§7.3).

### 4.3 Which `sync_cube` positions are supportable

| Position of `sync_cube` | v1 | Reason |
|---|---|---|
| Top level of the body | **accept** | clean segment boundary |
| Inside a **uniform-trip-count** loop, at statement level | **accept** | thread-loop inversion (§4.2); tree reduction |
| Inside a non-uniform loop | **reject** | barrier divergence — trip count differs per thread |
| Inside an `if`/`else` under a **non-uniform** condition | **reject** | barrier divergence — some threads skip it |
| Inside an `if` under a **cube-uniform** condition | v1.1 | whole cube takes it together — safe but adds a case |
| Inside a helper called from the body | v1.1 | inlined into the scope, but segment-splitting across an inlined call boundary needs care |

"Uniform" = provably independent of `UnitPos`/`AbsolutePos` (§5.4). v1 recognizes the two accepted
shapes structurally (top-level barriers; the `while half>0 { …; sync_cube(); half/=2 }` uniform
loop); broader uniformity is proved by B's machinery and can widen the accepted set later without a
twin-model change.

### 4.4 Candidate #3 — declared external reference (the complement)

A `reference = my_sequential_fn` contract clause lets the author supply a hand-written sequential
reference for a cooperative kernel the transformation cannot derive (e.g. a data-dependent barrier
count, or a kernel outside v1's shape). The evidence labels it honestly:

> `tested` — differential vs. **author-supplied reference (not derived from kernel source)**.

This is a **distinct, strictly weaker claim** than the derived twin: the derived twin *is* the
kernel's own structure re-executed sequentially (custody preserved — README "point of custody"),
whereas an author-supplied reference is a separate artifact that can silently drift from the kernel.
The claim taxonomy must never blur them (§6): the `check` string differs
(`differential-derived-twin` vs `differential-declared-reference`), and the declared-reference
variant records the reference fn's own source hash so its drift is at least visible in identity.

> **Status (round-3 adversarial review, F2 — implemented):** the reference fn's source hash is now
> recorded as promised. The `reference = fn` clause **requires** that fn to carry the new
> `#[vericl::reference]` attribute, which generates a sibling `<fn>_vericl` module holding the
> reference's own `SOURCE_HASH` (over its tokens). The kernel folds that hash into `identity()` via
> the same `vericl::combine_source_hash` runtime path `uses(...)` uses, so a drift in the reference
> **body** — not merely the clause path text carried in the attribute tokens — moves the kernel's
> recorded identity (`crates/vericl-macros/src/lib.rs::expand_reference`; regression test
> `block_sum_reduce_declared`/`declared_reference_body_is_part_of_kernel_identity`). Before F2 the
> clause folded in *nothing*, so `SOURCE_HASH` saw only the reference's *path text* and a body drift
> left identity byte-identical. A `reference = fn` naming an un-annotated fn is a compile error at
> the clause span, naming the missing `#[vericl::reference]` accessor.

**#3 is a complement, not a rival.** It keeps custody *honest* (an explicit "we did not derive
this") for kernels #1/#2 can't reach, instead of stretching the phase-split transform past where it
is sound. Recommendation: ship #1/#2 as the primary path and #3 as the fallback, gated so a kernel
*inside* the transformable subset cannot silently opt into the weaker claim (that would be a
custody downgrade with no signal).

### 4.5 Twin faithfulness depends on B (the coupling, previewed)

The phase-split twin picks **one** intra-segment thread order (`unit_pos = 0,1,2,…`). If a segment
has a write-write or read-write race on shared/global memory, that order is *one* possible GPU
outcome among several the hardware may pick — so the differential test is not a valid equivalence
check. The twin is faithful **iff every segment is race-free**, which is deliverable B. Full coupling
design in §6.

A second, subtler faithfulness obligation: **shared-memory definedness.** GPU shared memory is
*uninitialized* until written. If a segment reads `tile[i]` that no earlier segment wrote, the twin
(which would otherwise zero-init `tile`) and the GPU (garbage) diverge — and it is a real bug. The
twin therefore **poison-initializes** shared memory (a read of a never-written cell panics, exactly
like the OOB-read panic the bounds twin already uses — README "First finding" / `axpy_off_by_one`),
surfacing the bug instead of masking it. For the reduction shape definedness holds by construction
(every `tile[tid]` for `tid∈[0,cube_dim)` is written in phase 0 before any tree read of
`tile[tid+half]<cube_dim`), so the poison never fires — but the check is what keeps a *different*
cooperative kernel honest.

### 4.6 Feasibility probe — validated bit-exactly on wgpu/Metal

I hand-wrote the phase-split twins of `block_sum_reduce` and `grid_stride_reduce` (mechanically,
following §4.1–4.2 — `shmem_probe.rs::{block_sum_reduce_twin, grid_stride_reduce_twin}`) and ran
them differentially against the **real cooperative kernels launched on wgpu/Metal**:

```
block_sum_reduce   n∈{1,3,200,256,257,512,1000,4096}, cube_dim=256:
    every partial BIT-EXACT (gpu.to_bits() == twin.to_bits())
grid_stride_reduce cube_count∈{1,2,4,8,16}, cube_dim=256, n=4096:
    every partial BIT-EXACT
```

**Bit-exact, not merely within tolerance** — because the phase-split twin sums in the identical tree
order the GPU does, so at equal precision (f32) it reproduces the value exactly. This is the
strongest possible evidence that the phase model is faithful: the derived sequential reference is
not an approximation of the reduction, it *is* the reduction, re-associated in the same order. (A
kernel whose GPU backend contracts to FMA or reorders — e.g. the grid-stride reduction at `F=f32` vs an `f64`
reference lane — would need the usual `compare(abs=…)` tolerance, README "First finding"; the
phase model itself introduces no additional divergence.)

---

## 5. Deliverable B — race-freedom proving

### 5.1 The two-thread abstraction

Data races are **pairwise**: a workgroup has a race iff *some* pair of threads conflicts. So it
suffices to prove race-freedom for **two arbitrary distinct symbolic threads** `t1 ≠ t2` in one cube
— the standard GPUVerify reduction (README "Relationship to prior art"). Fully symbolic `t1, t2`
over `[0, CUBE_DIM)` cover every concrete pair, so UNSAT-for-the-symbolic-pair ⟹ race-free for all
pairs. This is sound and, crucially, keeps the query in **QF_LIA** — the exact theory the existing
prover already uses.

### 5.2 Duplicated value environment

Extend `Prover` to carry **two memos**, one per thread, sharing uniform leaves:

| Leaf | thread 1 | thread 2 | shared? |
|---|---|---|---|
| `UnitPos` | fresh `t1` (`0≤t1<CUBE_DIM`) | fresh `t2` | **no** (`t1 ≠ t2` asserted) |
| `AbsolutePos` | `CUBE_POS*CUBE_DIM + t1` | `CUBE_POS*CUBE_DIM + t2` | **no** (derived from `t1`/`t2`) |
| `CubePos`, `CubeDim`, `CubeCount`/`num_cubes`, `Length(buf)`, integer `GlobalScalar` | one SExpr | **same** SExpr | **yes** (cube-uniform) |

`CUBE_DIM` should be **bound to the contract-pinned `cube_dim` constant** (§7.1) so it is a concrete
number (256), making the shared-length obligation `tile[tid+half] < 256` trivially concrete; a
stronger tier can leave it symbolic with `CUBE_DIM ≤ shared_len` as an assume.

The single-thread walk already builds *terms* not per-variable symbols (prover.rs module docs), so
"duplicate the environment" is: run the existing walk **twice**, once resolving `UnitPos→t1`, once
`→t2`, recording the symbolic **index term** at every shared/global access, then emit cross-thread
obligations from the two recorded term sets.

### 5.3 The obligation set, per barrier interval

Partition the body into phases at `SyncCube` (the same segmentation as A). Within each phase,
collect for each thread its set of shared/global **accesses** tagged read/write with the symbolic
index term. Emit, per phase:

- **Shared write-write**: for every pair of writes `(w1 by t1, w2 by t2)` to the *same* shared
  array, obligation `t1≠t2 ⟹ idx(w1) ≠ idx(w2)`. Negation checked SAT: `t1≠t2 ∧ guard1 ∧ guard2 ∧
  idx(w1)=idx(w2)` — UNSAT ⟹ no WW race.
- **Shared read-write**: for every `(write by t1, read by t2)` on the same array, obligation
  `t1≠t2 ⟹ idx(write) ≠ idx(read)`. (Symmetric; also `t2` writes vs `t1` reads.)
- **Global write-write / read-write**: same, on `GlobalOutputArray`. Same-cube pairs use the
  `AbsolutePos = CUBE_POS*CUBE_DIM + t_i` encoding; a store at `out[ABSOLUTE_POS]` is disjoint across
  threads because `t1≠t2`. **Inter-cube** global races (two threads in *different* cubes) are a
  second, lighter two-thread pair with `CUBE_POS1 ≠ CUBE_POS2` and the shared leaves *not* shared —
  v1 handles the two provable-by-construction cases: `out[ABSOLUTE_POS]` (disjoint since global ids
  are unique) and single-writer `out[CUBE_POS]` under a `tid==0` guard (disjoint across cubes, one
  writer within).
- **No cross-*phase* obligations** — the barrier orders phase `p` writes before phase `p+1` reads,
  so they cannot race. This is what makes the set small and per-phase local.

Reads never race with reads (omitted). Benign races (both threads write the *same value*) are **not**
exploited in v1 — a same-value WW still fails; deliberately conservative, matching the "reject rather
than approximate" posture (prover.rs module docs).

A tainted index (depends on array contents — a gather, or any unmodeled construct) means we cannot
prove disjointness → the obligation fails as **`OutOfSubset`** at that access site (never silently,
never `Proved`) — the identical taint discipline as the bounds walker (prover.rs:670, the
"read index … depends on a construct outside the vericl v0 subset" site).

### 5.4 Barrier uniformity (the third class) and cross-thread taint

Every `sync_cube()` must be reached by **all threads together**. v1 obligations:

- **Enclosing-condition uniformity**: every `if`/loop condition enclosing a `sync_cube` must be
  **thread-invariant**. Primary mechanism is *dataflow taint*, reusing the existing memo: mark
  `UnitPos`/`AbsolutePos` as **thread-varying**; a value tainted-by-thread-varying that gates a
  barrier ⟹ reject. This needs no SMT. An SMT cross-check is also available for the affine cases
  (prove `cond(t1) = cond(t2)`); v1 uses the dataflow form as authoritative.
- **Cooperative-loop trip-count uniformity**: the carried control variable's recurrence and the
  loop guard must both be thread-invariant (the `half` recurrence is `CubeDim/2`, halving — no
  `UnitPos`, uniform).

"**Taint cross-thread**" means: a per-thread local computed from `UnitPos` is *thread-varying* (fine
— modeled as a function of `t_i`, differs between threads by construction); a value from array
contents is *tainted/unmodeled* (as in single-thread). Only the first is relevant to uniformity; only
the second blocks a race obligation. They are different axes and the design keeps them separate: a
thread-varying-but-modeled index is exactly what the race obligation reasons about; a tainted index
is what it gives up on.

### 5.5 Feasibility probe — all classes validated in QF_LIA

SMT-LIB obligations for the tree-reduction phase (`*.smt2`), discharged by z3:

| Obligation | Encoding | z3 |
|---|---|---|
| Shared WW race-free | `t1≠t2 ∧ t1<H ∧ t2<H ∧ t1=t2` | **unsat** ✓ (no race) |
| Shared RW race-free | `t1≠t2 ∧ t1<H ∧ t2<H ∧ (t1=t2 ∨ t1=t2+H)` | **unsat** ✓ (no race) |
| Racy control `tile[t]=tile[t+1]` | `t1≠t2 ∧ t1=t2+1` | **sat** ✓ (race found) |
| Shared bounds `tile[tid+half]<256` | `t1<H ∧ 1≤H≤D/2 ∧ D≤256 ∧ (t1+H≥256)` | **unsat** ✓ (in bounds) |
| Barrier uniformity (tree guard `H>0`) | `cond(t)=(H>0)`, `cond(t1)≠cond(t2)` | **unsat** ✓ (uniform) |
| Barrier divergence (`tid<half`) | `cond(t)=(t<H)`, `cond(t1)≠cond(t2)` | **sat** ✓ (divergent — reject) |

All plain linear integer arithmetic — no new theory, no bitvectors, symbolic `H`/`D`. The racy and
divergent negative controls fire SAT, so the obligations don't vacuously over-prove. **The
two-thread encoding is tractable for exactly the shape v1 targets.**

### 5.6 The new claim

A `proved`/`smt-race-freedom` claim, sibling to `smt-oob-freedom`
(`crates/vericl/src/evidence.rs:45` `ClaimKind::Proved`; the discriminator is the `check` string,
evidence.rs:33). Its `config` records: solver + `z3 --version` (trusted component, prover.rs:204),
`QF_LIA`, phase count, and per-phase obligation counts (WW/RW/uniformity). Trusted list unchanged
(z3, cube front-end). `verify()`'s existing downgrade check (evidence.rs:242 — a stored `Proved`
with no matching current claim is a reported problem) extends to it for free.

---

## 6. The A↔B coupling — the honesty design

This is the crux the adversarial review will attack. The rule:

> **A `tested` differential claim on a cooperative kernel is meaningful only under intra-phase race
> freedom + barrier non-divergence. The evidence must express that dependency, never assume it
> silently.**

Three tiers, each a distinct claim shape:

| Tier | Twin | Race-freedom | Differential claim records |
|---|---|---|---|
| **Strong** | derived phase-split twin | `proved`/`smt-race-freedom` present | `tested` (derived twin); dependency **discharged** — cites the race-freedom claim |
| **Honest fallback** | derived phase-split twin | *not* proved | `tested` (derived twin) with **`assumed`: "intra-phase race freedom + barrier non-divergence"** in its assumptions |
| **Weakest** | author-supplied reference (#3) | n/a | `tested` (**author-supplied reference, not derived**); still may carry the race-freedom assumption |

Mechanically: when the suite runs a cooperative kernel, it checks whether the `smt-race-freedom`
proof `Proved`. If yes, the differential `Claim.config` lists race-freedom as a *discharged*
dependency (pointing at the proved claim's `check`). If no (proof `OutOfSubset`, or `prove:false`),
the harness **injects an `Assumed` claim** "intra-phase race freedom + barrier non-divergence" into
the entry and the differential claim's assumptions reference it — exactly as `compare(abs=…)`
tolerances already travel as assumptions (README "Claims and trust boundaries"). The claim is never
silently upgraded: without the proof or the assumption, a cooperative differential result is
**refused**, not recorded (the same posture as `prove:false` omitting a proved claim rather than
faking one, README suite section).

Why this is the right severity: a cooperative differential *pass* against a race-free-by-luck-of-order
twin looks identical to a genuinely-equivalent one — the exact "silent disagreement between
artifacts that each look reasonable" failure the project exists to catch (README "Problem"). Making
race-freedom a named, visible dependency is what stops a green cooperative test from over-claiming.

Note the dependency direction: **A depends on B, not vice versa.** B (race-freedom) is a standalone
`proved` property valid regardless of whether the twin exists. A (the differential claim) is the one
that borrows B's guarantee. The `sum_racy` precedent (README "Proved claims": bounds `Proved` while
the differential *correctly fails*) shows the two claim kinds already stay independent; here the
differential claim additionally *cites* the proved one when present.

---

## 7. The v1 subset boundary — what's accepted, what's rejected

### 7.1 Contract additions

- **`cooperative(cube_dim = 256)`** — declares the kernel as workgroup-cooperative and pins the
  launch block size. `cube_dim` becomes the `CUBE_DIM` binding for the prover (a concrete constant)
  and the per-thread loop bound for the twin. Must be a power of two ≥ the tree reduction's needs
  and `≤` every `SharedMemory::new(N)` length in the body (checked; else the shared store is OOB).
- The suite launches a cooperative kernel with `CubeCount = cube_count` (a declared or per-case
  value) and `CubeDim = cube_dim`, sizing each `&mut Array` output to `cube_count` (the partials).
  This replaces the flat `ceil(n/cube_dim)` model (lib.rs:2782) *for cooperative kernels only*.
- Optional **`reference = fn`** (§4.4) for the declared-reference fallback.

### 7.2 Accepted (v1)

1-D topology (`UNIT_POS`, `CUBE_POS`, `CUBE_DIM`, `ABSOLUTE_POS`, `CUBE_COUNT`-as-scalar);
`SharedMemory::<f32>::new(N)` with comptime `N`, 1-D affine indexing; `sync_cube()` at statement
level (top-level or in a uniform-trip-count loop under uniform conditions); one non-cooperative
accumulation loop (any trip count, no barrier inside); one uniform cooperative tree loop;
single-writer global store guarded by `tid==0`. Helper composition in phase 0 (already inlined).

### 7.3 Rejected, with targeted errors

| Construct | Error (macro or prover) |
|---|---|
| `sync_cube` under a thread-varying condition | prover: `"sync_cube() under a non-uniform condition (barrier divergence) is outside the vericl v0 subset"` |
| Barrier inside a non-uniform-trip-count loop | prover: `"sync_cube() inside a loop with a thread-varying trip count (barrier divergence) …"` |
| Shared index that is tainted / non-affine | prover: `"shared write index for \`tile[...]\` depends on a construct outside the vericl v0 subset"` (same site as prover.rs:691) |
| `sync_plane`/`sync_storage`/`plane_*`/`Atomic` | macro: existing `BANNED_IDENTS`/`BANNED_PREFIXES` (lib.rs:64,80) |
| 2-D/3-D topology (`UNIT_POS_X`, `CUBE_POS_Y`, …) | macro: existing `BANNED_IDENTS` (lib.rs:47) — 1-D only |
| `terminate!()` | macro: existing ban (lib.rs:77) — deferred to v1.1 (§7.4) |
| Multiple shared arrays with aliasing indices | v1: reject >1 `SharedArray` id per kernel (single tile); relax later |
| A cooperative differential result with neither a race-freedom proof nor an explicit assumption | harness: **refused** (§6) |

### 7.4 Deferred (v1.1+, not rejected-forever)

Workgroup-uniform `terminate!()` (needs a proved-uniform terminate condition + twin "skip cube");
2-D dispatch padding; barriers inside helpers; multiple shared tiles; inter-cube global races beyond
the two provable-by-construction cases; the multi-receiver reduction's full shape (composition +
terminate + 2-D). Each widens an axis of the same design without changing the phase model or the
two-thread encoding.

> **Status (cooperative v1.1 extensions — IMPLEMENTED).** Three of these axes
> now land, together, on the multi-receiver reduction shape (minus 2-D):
> - **`#[comptime]` parameters** in a cooperative kernel — cube-uniform by
>   construction, threaded through the phase-split twin as `let` consts and baked
>   into the IR (the easiest uniformity case; no phase-splitter interaction).
> - **`uses(...)` composition** — a helper runs as a **barrier-free** unit inside
>   one segment. The soundness crux is enforced on **both lanes**: the twin lane
>   rejects a `#[vericl::helper]` that contains `sync_cube`/`SharedMemory` with a
>   targeted error ("barriers … must be visible at the cooperative kernel's top
>   level"), and the prover independently compares the (helper-inlined) IR's
>   `SyncCube` count against the twin's declared `COOP_BARRIER_COUNT` — a hidden
>   helper barrier inflates the IR count and is rejected. (Barriers *inside*
>   helpers themselves remain deferred: helpers must stay barrier-free.) Shared
>   tiles cannot cross a helper boundary (twin-local `SharedTile`).
> - **Workgroup-uniform `terminate!()`** — accepted only under a proven-uniform
>   condition at the top level before any barrier (the "skip the whole cube"
>   guard, un-banned in cooperative mode only). The twin models it as a cube-level
>   `continue`; the prover as a `!cond` path condition (uniformity verified by the
>   thread-varying taint machinery, before-any-barrier enforced), which is
>   **load-bearing** for the single-writer store bound in the multi-receiver reduction
>   (`powers[CUBE_POS]` with no explicit guard). Non-uniform or post-barrier
>   terminate stays rejected on both lanes.
>
> Still deferred: **2-D dispatch padding** (the one multi-receiver-reduction
> feature this round does not lift — the kernel body is 1-D in `CUBE_POS`, so a
> 1-D launch annotates it; 2-D only raises the workgroup ceiling), multiple
> shared tiles, barriers inside helpers, and wider inter-cube races.

---

## 8. Implementation plan (agent-sized milestones)

Each milestone is independently verifiable and lands behind the existing test posture (`cargo test
--workspace`, clippy 0, evidence regenerated *last*). B and A are separable — **B first** (it is the
gateway and A depends on it).

**M1 — Model the cooperative leaves + shared arrays in the bounds walker (no races yet).**
`value_of`: add `UnitPos`/`CubePos`/`CubeDim`/`CubeCount` as leaves (`CubeDim` bound to the pinned
`cube_dim` constant; `UnitPos` a fresh `[0,CUBE_DIM)` symbol; `AbsolutePos` recomputed as
`CubePos*CubeDim+UnitPos` when cooperative). `buffer_of`/`emit_obligation`: accept `SharedArray`
lists, keying bounds off the `length` in the `VariableKind` (a compile-time constant, not a runtime
`Length`). *Verify*: bounds of a single-thread symbolic pass over `block_sum_reduce` `Proved` (shared
`tile[tid+half] < 256` discharges given `CUBE_DIM=256`); a deliberately-oversized `SharedMemory::new`
vs `cube_dim` `Refuted`.

**M2 — Recognize the two loop shapes (`Branch::Loop`).** Detect the canonical `while` desugaring
(leading break-guard, §2.4). For a **non-cooperative** loop (no `SyncCube` inside): model like a
`RangeLoop` — induction var symbolic, loop guard as an in-body path condition, carried accumulator
tainted (reuse `scope_reassigned_vars`, prover.rs:1020). For a **cooperative** loop: require uniform
trip count (M4) and hand its body to the phase walker (M3). *Verify*: `grid_stride_reduce`'s phase-0
loop's `data[k]` read `Proved` under `n ≤ data.len()`; a bare `Branch::Loop` with no break-guard
still `OutOfSubset` (unchanged).

**M3 — The two-thread race walker + `smt-race-freedom` claim.** Duplicated environment (§5.2), phase
segmentation at `SyncCube`, per-phase WW/RW obligation emission (§5.3), the new `Proved` check
(§5.6). *Verify*: `block_sum_reduce`/`grid_stride_reduce` `Proved` race-free (obligation counts
recorded); a racy variant (`tile[tid]=tile[tid+1]` with no barrier between generations) `Refuted`
with a two-thread counterexample; matches the `*.smt2` probe verdicts.

**M4 — Barrier uniformity.** Thread-varying taint of `UnitPos`/`AbsolutePos`; reject a barrier under
a thread-varying condition or in a thread-varying loop (§5.4, §7.3). *Verify*: a kernel with
`sync_cube()` inside `if tid < half` `OutOfSubset` with the barrier-divergence reason; the tree loop
(`half` uniform) accepted.

**M5 — The phase-split twin + `cooperative(...)` clause (macro).** New cooperative twin-derivation
mode: `cooperative(cube_dim=…)` gate; emit the per-cube/per-segment/per-`unit_pos` twin with
cross-barrier per-thread locals promoted and shared arrays as per-cube `Vec`s (poison-init, §4.5);
the new launch/output model (§7.1). *Verify*: the differential probe of §4.6 reproduced through the
*generated* twin (bit-exact vs wgpu); a hand-written twin cross-check unit test (the
`*_twin_matches_handwritten` precedent).

**M6 — The coupling + declared-reference fallback.** Wire §6: differential claim cites the
race-freedom proof when present, injects the `Assumed` "intra-phase race freedom" claim otherwise,
refuses a cooperative differential with neither; add `reference = fn` (#3) with its distinct
`differential-declared-reference` check string and reference-source-hash. *Verify*: a cooperative
kernel evidence entry shows the `proved`+`tested`(+discharged-dependency) triple; forcing
`prove:false` flips the `tested` claim to carry the explicit `assumed` race-freedom clause; a
declared-reference kernel records the weaker check string.

**M7 — Public example + private dogfood.** A clean-room `block_sum_reduce` public example wired into
`vericl::suite!` carrying `tested`+`proved`(race-freedom)+`proved`(bounds); dogfood the grid-stride reduction
privately (construct-class only, per README policy) to confirm the real shape lands. *Verify*: suite
green, evidence regenerated last; dogfood differential + both proofs pass on the real reduction
shape.

---

## 9. Open risks, ranked

1. **Cooperative-loop recognition brittleness (high).** v1 keys on the canonical `while` desugaring
   (`Branch::Loop` with a leading break-guard, uniform halving control). A kernel that writes its
   tree loop differently (e.g. a `for level in 0..log2n` `RangeLoop`, or a manual index recurrence)
   won't match the structural recognizer and falls to `OutOfSubset`. Mitigation: recognize by the
   *uniformity property* (thread-invariant guard + recurrence) rather than the exact syntax, and add
   the `RangeLoop`-shaped tree as a second accepted form. **Attack surface**: an adversarial reviewer
   will hand a semantically-identical tree loop in a shape v1 doesn't recognize — the honest answer
   is `OutOfSubset`, not a wrong `Proved`, which is acceptable but must be *demonstrated* not
   assumed.

2. **Barrier-uniformity soundness (high).** If the uniformity taint is too loose, a barrier under a
   subtly thread-dependent condition slips through and the phase model is unsound (the twin assumes
   non-divergence). Mitigation: conservative dataflow (any `UnitPos`/`AbsolutePos` reachability taints
   the condition), plus the SMT cross-check for affine conditions; negative controls in the test
   suite (the `uniform_bad.smt2` shape). This is the analog of the round-2 branch-scoping bug
   (prover.rs module docs) — it *will* be adversarially probed.

3. **Shared-memory definedness masking (medium).** If the twin zero-inits shared memory instead of
   poisoning, an uninitialized-read bug is silently masked (twin reads 0, GPU reads garbage).
   Mitigation: poison-init + a definedness obligation (§4.5). Low-probability for the reduction shape
   (definedness holds by construction) but a real hole for other cooperative kernels.

4. **Inter-cube global races (medium).** v1 proves same-cube race freedom and handles only the two
   provable-by-construction inter-cube global cases (`out[ABSOLUTE_POS]`, single-writer
   `out[CUBE_POS]`). A cooperative kernel with a genuinely cube-crossing global write pattern is
   `OutOfSubset` in v1. Mitigation: the second two-thread pair with `CUBE_POS1≠CUBE_POS2` is a
   straightforward extension; scope it explicitly so the gap is *documented*, not silent.

5. **`CUBE_DIM` pinning vs runtime block size (medium).** Binding `CUBE_DIM` to the pinned
   `cube_dim` is sound only if the launch actually uses that block size. Mitigation: the suite's
   cooperative launch uses `cube_dim` from the clause (single source of truth); a mismatch is a
   harness bug, not a silent unsoundness, but worth an assertion. The stronger symbolic-`CUBE_DIM`
   tier removes the dependency at the cost of harder obligations.

6. **Float non-associativity beyond the tree order (low).** The phase-split twin is bit-exact *at
   equal precision in the same order* (§4.6), but an `f64` reference lane, or a backend that
   contracts/reorders the tree, needs `compare(abs=…)` — the tolerance must be justified by input
   ranges, same as every float kernel. Not a new risk, but cooperative kernels make the summation
   depth (`log2(cube_dim)` + grid-stride length) larger, so tolerances need care.

7. **cubecl upgrade drift (low, standing).** The `Branch::Loop` desugaring, `SharedArray` shape, and
   `Synchronization::SyncCube` are internals; a cubecl upgrade could change them. Mitigation: the
   existing "survives a CubeCL upgrade" health check + the IR-level identity hash (which now covers
   `scope.shared`, scope.rs:84) already trips on codegen drift.

---

## 10. Roadmap impact

- **Resolves** the README open decision on ordering ("race freedom on shared memory … before or
  after a second proved property"): race-freedom is the **gateway** and is designed here; it is
  roadmap item 6 (`tasks/todo.md`).
- **Confirms** the dogfood ranking (`docs/dogfood-2026-07.md`): the shared-memory reduction shape is
  "last, hardest, and the gateway to race-freedom proofs" — and the `block_sum_reduce` candidate #3
  is the validated v1 example (probed bit-exact here).
- **New public claim kind**: `proved`/`smt-race-freedom`, the second machine-checked property, joins
  `smt-oob-freedom`. The claim taxonomy gains a third differential flavor
  (`differential-declared-reference`) for the author-supplied-reference fallback — kept strictly
  distinct from the derived twin.
- **Does not** need QF_BV, the `f64` tier, or 2-D dispatch — all obligations are QF_LIA, and the v1
  subset is 1-D f32, matching everything else in vericl v0. The multi-receiver reduction's full shape
  (composition + `terminate!()` + 2-D) is a well-scoped v1.1 that widens axes without touching the
  phase model or the two-thread encoding.
