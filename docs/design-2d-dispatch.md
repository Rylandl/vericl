# 2-D / 3-D dispatch — design (July 2026)

The implementable design for per-axis topology in kernel bodies — `ABSOLUTE_POS_X/Y/Z`,
`CUBE_POS_X/Y/Z`, `UNIT_POS_X/Y/Z`, `CUBE_DIM_X/Y/Z`, `CUBE_COUNT_X/Y/Z` — and for multi-axis
`CubeCount`/`CubeDim` launches. This is the last unclaimed item on the recorded post-`Slice`
frontier (`tasks/todo.md:2302-2305`), it is the **#1 shape an external user expects to work and
finds rejected** (image-space kernels), and it is the **weakest measured corpus unlock of any
VeriCL milestone so far**: `docs/ecosystem-survey-2026-07.md` measures 39 items naming 2-D topology
with **1 sole-blocker of 464**, and `docs/dogfood-2026-07.md` measures 2/22 private kernels blocked
with **0 sole** — though 2-D is one of only **two** walls the private corpus has left.

Two deliverables:

- **A. The generalization.** This is not a green-field milestone. Every mechanism it needs already
  exists — the round-5 exact-modular recomposition, the faithful Add/Sub wrap, the `checked_mul`
  side-obligation, branch write-taint, the phase-split twin. The design's job is the **soundness
  edges** where the 1-D machinery does not survive the generalization, and §2 finds one that is not
  a detail: **the identity VeriCL's cooperative prover encodes is false in 2-D**, in 533 of 722
  measured launch shapes.
- **B. The latent hole.** A VeriCL kernel proved and differentially green today carries **no claim
  at all** about the launch *shape* the user chooses. `conformance_case` hardcodes a 1-D dispatch
  (`lib.rs:6243-6244`); nothing records that, nothing checks it, and the flat `ABSOLUTE_POS` a
  1-D-authored kernel indexes with is a **u32** row-major flatten over the whole grid (§1.3,
  measured in the generated WGSL) which stops being injective at `2^32` threads — reachable only on
  a multi-axis grid. §3.3 states the hole and its exact reachability threshold; §10.4 closes the
  recordable half of it.

Everything marked *measured* was checked empirically against the pinned `cubecl =0.10.0` (z3 4.16.0
on PATH, wgpu 29.0.4 / Metal on an Apple M3), the same posture as
[design-line-vector.md](design-line-vector.md), [design-shared-memory.md](design-shared-memory.md),
[design-struct-comptime.md](design-struct-comptime.md) and
[design-cubetype-args.md](design-cubetype-args.md). Probe sources are preserved in the scratchpad
(`scratchpad/design2d/probe/src/bin/{ir_axes,axis_order,identity_sweep,blur2d,limits,prover2d}.rs`,
`scratchpad/design2d/probeA/topo/src/main.rs`, and the nine hand-written SMT-LIB files
`scratchpad/design2d/smt/p{1,1b,2a,2b,2c,2d,2e,2f,3,4,4b,5}*.smt2`); the consolidated run is
`scratchpad/design2d/RESULTS.txt` and the captured WGSL is
`scratchpad/design2d/probeA/wgsl_dump.txt`. Reference shapes are **clean-room / upstream-public
only** (cubecl 0.10.0 — MIT/Apache-2.0), per the README policy; the box blur and transpose in §6 are
textbook shapes written for this document.

File:line citations to `crates/vericl-macros/src/{lib,coop,suite}.rs`,
`crates/vericl-ir/src/{prover,interp}.rs`, `crates/vericl/src/evidence.rs` and the
`cubecl-{core,ir,wgpu,cpp,cpu,spirv,runtime,macros}-0.10.0` trees are current as of `22e4349`.

---

## 0. Headline recommendation

1. **The axis-ordering footgun the task asked me to hunt does not exist — and the one that does is
   somewhere else entirely.** `CubeCount::Static(x,y,z)` reaches `dispatch_workgroups(x,y,z)`
   reaches `MTLSize{width:x,height:y,depth:z}` reaches `workgroup_id.x/.y/.z` reaches
   `CUBE_POS_X/Y/Z` with **no transposition at any layer**
   (`cubecl-wgpu/src/compute/stream.rs:577-585`, `wgpu-hal/src/metal/command.rs:1771-1778`,
   `cubecl-wgpu/src/compiler/wgsl/compiler.rs:342-400`), and a kernel that self-reports its whole
   topology tuple confirms it: **0 violations in 1 212 threads across 6 launch shapes** including
   `(3,2,2)×(2,3,2)`, with every observed per-axis maximum equal to its launched extent (§1.4).
   Every flattening is **row-major with X fastest**, consistently across WGSL, CUDA/HIP, SPIR-V and
   the CPU runtime (§1.3). The real footgun is item 2.

2. **`ABSOLUTE_POS == CUBE_POS * CUBE_DIM + UNIT_POS` — the identity `abs_pos_sym` encodes as
   soundness-critical since round 5 — is FALSE in 2-D, and this is the centre of the milestone.**
   `ABSOLUTE_POS` linearizes the *global thread coordinate* over the whole grid; `CUBE_POS*CUBE_DIM
   + UNIT_POS` linearizes *cube-major, then unit-within-cube*. Those are different orderings.
   Swept on hardware over **722 launch shapes**, the identity **held in 189 and broke in 533** — 912
   of 960 threads violate it at the image-like `(5,3,1)×(8,8,1)`. It holds for every thread **iff
   `CUBE_DIM_Y == CUBE_DIM_Z == 1`** (or the degenerate single-column-of-cubes corner); the exact
   algebraic predicate agrees with hardware **722/722** (§2.2). Consequence: the round-5
   recomposition **must be re-expressed per axis**, and the flat `ABSOLUTE_POS` must not be modeled
   from the per-axis leaves at all (item 4).

3. **The round-5 defect transplants to each axis unchanged, and so does its fix.** An *unwrapped*
   per-axis recomposition `abs_x = cube_x*Wx + unit_x` under a guard `ABSOLUTE_POS_X < out.len()`
   yields a false `Proved` on a bare `out[CUBE_POS_X]` write — **UNSAT, measured** (`p1`). The
   exact-modular form with a **fresh wrap counter per axis** turns it into the honest `Refuted` —
   **SAT with the witness `cube_x=16843009, wrap_x=1, abs_x=16843008, len_out=16843009`** (`p1b`),
   literally round 5's witness shape on the X axis. Both products stay variable×constant because
   `CUBE_DIM_X` is pinned by the contract clause, so the encoding stays **QF_LIA** (§4.3).

4. **The three flat builtins are rejected in per-axis mode, because modeling them is genuinely
   nonlinear — measured, not asserted.** `ABSOLUTE_POS`'s stride is `CUBE_COUNT_X*CUBE_DIM_X` with
   `CUBE_COUNT_X` a *runtime* value, so `ABSOLUTE_POS_Y * (CUBE_COUNT_X*CUBE_DIM_X)` is
   variable×variable; ditto `CUBE_POS_Y*CUBE_COUNT_X` and `CUBE_COUNT_X*CUBE_COUNT_Y`. The
   corresponding inter-cube disjointness query **times out in z3 at 180 s** (`p5`), where the 1-D
   case is a pattern match costing no SMT at all. v1 therefore rejects flat
   `ABSOLUTE_POS`/`CUBE_POS`/`CUBE_COUNT` inside a `dispatch(...)` kernel (R1) and keeps flat
   `CUBE_DIM` / `UNIT_POS`, which stay a constant and a pinned-coefficient linear form (§4.4).

5. **Decided design: one new contract clause `dispatch(cube_dim = (…), extents = (…))`, per-axis
   leaves with per-axis exact modular recomposition, two newly-modeled arithmetic ops, one new
   structured assume, and a nested-loop twin.** The clause is `cooperative(cube_dim = N)`'s
   precedent moved one position over: pinning the per-axis cube dims is what keeps every
   recomposition linear, exactly as pinning `N` did in 1-D. `Arithmetic::Min`/`Max` become exact
   `ite` terms — six lines, no new obligation — because that is what the *only* branch-free stencil
   clamp lowers to (`binding(5) = binding(3).min(binding(4))`, measured) and because the `if`-based
   clamp is killed by round-2 branch write-taint (§3.2). The new assume
   `out.len() == (w as usize) * (h as usize)` is what ties the 2-D extents to a buffer length;
   without it the write obligation is genuinely **SAT** (`p2e`), with it the whole 3×3 clamped
   stencil is **UNSAT in 0.20 s** (`p2c`) — §4.

6. **Honest reach: v1 unlocks at most 1 of 464 ecosystem items and 0 additional private kernels on
   its own, and that is measured.** 39 ecosystem items name 2-D topology; **1** is sole-blocked by
   it (`docs/ecosystem-survey-2026-07.md:345`). Two of the 22 private dogfood kernels are blocked,
   **0 solely** (`docs/dogfood-2026-07.md:285`) — but 2-D is one of only **two** walls that corpus
   has left after the shim batch took it from 6/22 to 19/22, so it is a real ceiling there. The case
   for the milestone is **capability + the §3.3 hole**, not corpus arithmetic, and §11 says so
   plainly. This is the same shape as `cube_struct!` ("v1 unlocks zero of the corrected 20") and
   should be reviewed as such.

---

## 1. IR and backend reality — the per-axis topology catalog (validated)

### 1.1 The 31 `Builtin` variants and their prelude constants

`cubecl-ir-0.10.0/src/variable.rs:97-132` declares exactly 31 `#[repr(u8)]` variants; the
discriminant is load-bearing (cubecl-cpu indexes a `[Option<Value>; 31]` by `builtin as usize`,
`cubecl-cpu-0.10.0/src/compiler/visitor/args_manager.rs:23,309-316`). The prelude constants are
generated 1:1 in `cubecl-core-0.10.0/src/frontend/topology.rs` by two macros, and **the macro picks
the element type**:

```rust
macro_rules! constant {        // topology.rs:10-29   -> u32
    ($ident:ident, $var:expr, $doc:expr) => {
        pub const $ident: u32 = 2;
        ...
                    u32::as_type(scope).storage_type(),
macro_rules! constant_usize {  // topology.rs:31-50   -> usize (== AddressType)
        pub const $ident: usize = 2;
        ...
                    usize::as_type(scope).storage_type(),
```

| constant | `topology.rs` | `Builtin` | frontend type | v1 status |
|---|---|---|---|---|
| `ABSOLUTE_POS` | :268 | `AbsolutePos` | **`usize`** (→ `AddressType`) | **reject** in `dispatch(...)`, R1 |
| `ABSOLUTE_POS_X/Y/Z` | :276/:284/:292 | `AbsolutePosX/Y/Z` | `u32` | **accept** |
| `UNIT_POS` | :76 | `UnitPos` | `u32` | accept (linear in pinned dims) |
| `UNIT_POS_X/Y/Z` | :84/:92/:100 | `UnitPosX/Y/Z` | `u32` | **accept** |
| `CUBE_POS` | :172 | `CubePos` | **`usize`** | **reject** in `dispatch(...)`, R1 |
| `CUBE_POS_X/Y/Z` | :180/:188/:196 | `CubePosX/Y/Z` | `u32` | **accept** |
| `CUBE_DIM` | :140 | `CubeDim` | `u32` | accept (a pinned numeral) |
| `CUBE_DIM_X/Y/Z` | :148/:156/:164 | `CubeDimX/Y/Z` | `u32` | **accept** (pinned numerals) |
| `CUBE_COUNT` | :236 | `CubeCount` | **`usize`** | **reject** in `dispatch(...)`, R1 |
| `CUBE_COUNT_X/Y/Z` | :244/:252/:260 | `CubeCountX/Y/Z` | `u32` | **accept** |
| `PLANE_DIM`, `PLANE_POS`, `UNIT_POS_PLANE` | :52/:60/:68 | `PlaneDim`, `PlanePos`, `UnitPosPlane` | `u32` | reject (unchanged, R8) |
| `CUBE_CLUSTER_DIM{,_X,_Y,_Z}` | :108–:132 | `CubeClusterDim*` | `u32` | reject (unchanged, R8) |
| `CUBE_POS_CLUSTER{,_X,_Y,_Z}` | :204–:228 | `CubePosCluster*` | `u32` | reject (unchanged, R8) |

Three properties matter downstream, and none of them is cosmetic:

- **The flat position/count builtins are `usize`, the per-axis ones are `u32`.** `usize` resolves
  through `scope.resolve_type::<usize>()` (`cubecl-core-0.10.0/src/frontend/element/uint.rs:87-89`)
  to the kernel's `AddressType`, i.e. its *width is launch-configurable* (U32 or U64); the per-axis
  builtins are hard `u32`. This is why every index expression in a per-axis kernel is written
  `inp[(y * w + x) as usize]` (measured throughout §6) while a 1-D kernel writes
  `inp[ABSOLUTE_POS]` bare, and it is why the twin's loop variables must be `u32`, not `usize`
  (§4.7).
- **Outside a `#[cube]` fn every one of these constants is the literal `2`.** Host code reading
  `ABSOLUTE_POS_X` silently gets `2`. The existing ban is what has protected the twin from this; the
  design must keep an explicit rewrite for every per-axis ident it un-bans (§4.7, R2).
- **`CUBE_POS_CLUSTER*` folds to `1` on WGSL** (`compiler.rs:366-369`) **and to `0` on
  CUDA/SPIR-V** (`cubecl-cpp/shared/base.rs:1775`, `cubecl-spirv/globals.rs:67-70`) — an upstream
  cross-backend semantic divergence in cubecl 0.10. It changes nothing for v1 (the cluster builtins
  stay rejected) but it is recorded here because it is the kind of thing the upgrade drill should
  watch.

### 1.2 The flat builtins are IR primitives; every backend expands them itself

`Variable::builtin($var, …)` is the whole frontend (`topology.rs:21-26`) — the IR carries the
`Builtin` variant opaquely and nothing lowers it. Grepping `Builtin::` across `cubecl-opt-0.10.0`
finds only two *analysis* sites (`analyses/uniformity.rs:217-247`,
`analyses/integer_range.rs:157-167`); neither mutates the IR. Confirmed directly from an extracted
`KernelDefinition` (`ir_axes`, no client/device — the `docs/prototypes/ir_extraction.rs` recipe):

```text
binding(0) = AbsolutePosX < scalar<u32>(0) : (u32, u32) -> (bool)
binding(3) = AbsolutePosY * scalar<u32>(0) : (u32, u32) -> (u32)
binding(4) = binding(3) + AbsolutePosX     : (u32, u32) -> (u32)
```

**So the flat/axis relationship exists only below the IR, in each backend independently.** That is
the fact the prover model rests on: VeriCL sees `AbsolutePos` and `AbsolutePosX` as two unrelated
opaque leaves, and *any* relationship between them is a modeling choice VeriCL makes, not something
the IR forces. It is also why §2's broken identity is invisible to anything that reads only the IR.

`KernelDefinition.cube_dim` does carry the launch block dims, and it is the **only** place the IR
records them — measured: `KernelSettings::default().cube_dim(CubeDim::new_1d(256))` and
`.cube_dim(CubeDim::new_2d(16,16))` produce **byte-identical bodies** and differ only in
`def.cube_dim` (`ir_axes`). This matters twice: it is what `kernel_ir_hash` would have to fold to
make a dispatch-rank change move the identity (§10.4), and it is what makes the pinned-cube-dim
clause implementable at all.

### 1.3 The exact flattening formulas, source-cited and measured

**WGSL** — `cubecl-wgpu-0.10.0/src/compiler/wgsl/body.rs:19-25`, and the real emitted text from
`scratchpad/design2d/probeA/wgsl_dump.txt:47-49`:

```wgsl
let workgroup_id_no_axis = (u32(num_workgroups.y) * u32(num_workgroups.x) * u32(workgroup_id.z)) + (u32(num_workgroups.x) * u32(workgroup_id.y)) + u32(workgroup_id.x);
let workgroup_size_no_axis = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y * WORKGROUP_SIZE_Z;
let id = (u32(global_id.z) * u32(num_workgroups.x) * u32(WORKGROUP_SIZE_X) * u32(num_workgroups.y) * u32(WORKGROUP_SIZE_Y)) + (u32(global_id.y) * u32(num_workgroups.x) * u32(WORKGROUP_SIZE_X)) + u32(global_id.x);
```

Written in VeriCL's vocabulary, with `Ga = CUBE_COUNT_a * CUBE_DIM_a` the global grid extent on
axis `a`:

```
ABSOLUTE_POS = ( ABSOLUTE_POS_X
               + ABSOLUTE_POS_Y * Gx
               + ABSOLUTE_POS_Z * Gx * Gy )               mod 2^AddressBits
CUBE_POS     = ( CUBE_POS_X
               + CUBE_POS_Y * CUBE_COUNT_X
               + CUBE_POS_Z * CUBE_COUNT_X * CUBE_COUNT_Y ) mod 2^AddressBits
UNIT_POS     =   UNIT_POS_X
               + UNIT_POS_Y * CUBE_DIM_X
               + UNIT_POS_Z * CUBE_DIM_X * CUBE_DIM_Y
CUBE_DIM     =   CUBE_DIM_X   * CUBE_DIM_Y   * CUBE_DIM_Z
CUBE_COUNT   =   CUBE_COUNT_X * CUBE_COUNT_Y * CUBE_COUNT_Z
ABSOLUTE_POS_a = ( CUBE_POS_a * CUBE_DIM_a + UNIT_POS_a )  mod 2^32     (per axis a)
```

`AddressBits` is 32 by default (`AddressType::U32`, what `ir_extraction.rs` and the VeriCL macro
register) and the `u32(...)` casts in the emitted text above are that choice made visible. **The
`mod` is not decoration** — WGSL u32 arithmetic wraps, so the flatten really is modular, and §3.3
is the consequence.

Every other backend agrees on the algebra:

| backend | `ABSOLUTE_POS` | cite |
|---|---|---|
| WGSL | emitted `u32` arithmetic, above | `wgsl/body.rs:23` |
| CUDA / HIP | `(z*ccx*cdx*ccy*cdy) + (y*ccx*cdx) + x`, same coefficients | `cubecl-cpp/shared/kernel.rs:307-321` |
| SPIR-V | Horner form `((z*size_y + y)*size_x + x)`, `size_a = groups_a*cube_dim.a` | `cubecl-spirv/globals.rs:173-203` |
| Metal via `cubecl-cpp` (`msl` feature, **off by default**) | native `thread_index_in_grid`, no expansion | `cubecl-cpp/metal/dialect.rs:511-524` |
| CPU (MLIR) | emitted arithmetic, same strides | `cubecl-cpu/src/compiler/visitor/mod.rs:233-353` |

The Metal-via-`cubecl-cpp` row is the one to keep an eye on: it delegates to the hardware builtin
rather than reproducing the formula, so it agrees *by convention*, not by construction. It is not
what VeriCL uses — `cubecl-wgpu`'s `msl` feature is off by default
(`cubecl-wgpu-0.10.0/Cargo.toml:41`) and `vericl-examples` enables plain `cubecl/wgpu`, so Metal is
reached through WGSL → naga → MSL and gets `body.rs:23`'s arithmetic.

Measured against all of it (`axis_order`, 6 launch shapes, 1 212 threads, every thread
self-reporting all twelve of its topology values and the host checking each formula per thread):

```text
########## image-like 2-D : CubeCount(5, 3, 1) CubeDim(8, 8, 1)  (960 threads)
  observed: CUBE_DIM=64 (x,y,z)=(8,8,1)   CUBE_COUNT=15 (x,y,z)=(5,3,1)
  [dims] OK - no transposition at the launch boundary
  unwritten slots           : 0
  (2) ABSOLUTE_POS == x + y*Gx + z*Gx*Gy    violations: 0 / 960
  (3) UNIT_POS     == ux + uy*Wx + uz*Wx*Wy violations: 0 / 960
  (4) CUBE_POS     == bx + by*Nx + bz*Nx*Ny violations: 0 / 960
  (5) abs_a == cube_a*dim_a + unit_a        violations: 0 / 960
```

### 1.4 Axis ordering — the footgun that isn't

Three independent checks, all negative:

1. **The launch boundary.** Every configuration reports `CUBE_DIM_a` and `CUBE_COUNT_a` equal to the
   component launched on axis `a`, including the deliberately asymmetric `(3,2,1)×(4,2,1)` and
   `(3,2,2)×(2,3,2)` (`axis_order`, `[dims] OK` in 6/6).
2. **The observed extents.** At `(5,3,1)×(8,8,1)`: `max(abs_x)=39 = Gx-1`, `max(abs_y)=23 = Gy-1`,
   `max(cube_x)=4 = Nx-1`, `max(cube_y)=2 = Ny-1`. A transposition anywhere would show up as a
   swapped maximum; none does.
3. **The code path.** `CubeCount::Static(x,y,z)` → `pass.dispatch_workgroups(x,y,z)`
   (`cubecl-wgpu/src/compute/stream.rs:577-585`) → `MTLSize{width:x,height:y,depth:z}`
   (`wgpu-hal/src/metal/command.rs:1771-1778`); `Builtin::CubePosX → workgroup_id.x` is a straight
   map (`cubecl-wgpu/src/compiler/wgsl/compiler.rs:342-400`).

One stale comment is worth flagging so a future reader does not mistake it for a constraint —
`cubecl-wgpu-0.10.0/src/compiler/wgsl/body.rs:6-9` says *"the body assumes that the kernel will run
on a 2D grid … but with Z=1"*, and the code 14 lines below includes the `global_id.z` term. The
3-D measurements pass; the comment is outdated, not a limit.

**What v1 must still do about ordering.** The absence of a transposition is exactly why the
*rejection* wording matters more than a runtime check: a user who assumes Y is fastest (the
row-major-array intuition) will write `inp[x * h + y]` and get a silently transposed image that is
still perfectly in bounds — a *functional* bug the differential lane catches and the prover, by
design, does not. §10.3 R9 makes the guide state the convention next to the clause rather than
leaving it to be discovered.

### 1.5 Hardware limits, measured

`limits`, on this machine:

```text
runtime = "wgpu<wgsl>"
    max_cube_count:      (65535, 65535, 65535)
    max_cube_dim:        (1024, 1024, 1024)
    max_units_per_cube:  1024
    max_shared_memory_size: 32768
```

| bound | WebGPU default | this adapter (Metal) | where |
|---|---|---|---|
| `max_compute_workgroups_per_dimension` | 65535 | 65535 (hardcoded by wgpu-hal, not a Metal limit) | `wgpu-types-29.0.4/src/limits.rs:407`; `wgpu-hal/src/metal/adapter.rs:1232` |
| `max_compute_workgroup_size_x/y` | 256 | 1024 | `limits.rs:404-405`; `adapter.rs:1229-1230` |
| `max_compute_workgroup_size_z` | **64** | **1024** | `limits.rs:406`; `adapter.rs:1231` |
| `max_compute_invocations_per_workgroup` | 256 | 1024 | `limits.rs:403`; `adapter.rs:1227` |

Failure modes, measured by pushing past each:

- `CubeDim` over-large: cubecl validates it (`cubecl-runtime-0.10.0/src/validation.rs:9-45`, called
  from `cubecl-wgpu/src/compute/server.rs:157-158`) — but only against the *adapter's* limits, so
  `(1,1,65)` and `(32,16,1)` and `(1024,1,1)` all **launch fine here** while they would be rejected
  on a WebGPU-default device. A `dispatch(cube_dim = …)` clause tuned on Metal can therefore fail on
  another backend; R5's wording says so.
- `CubeCount` over-large: **cubecl does not check it at all** — `ResourceLimitError` has no
  `CubeCount` variant (`cubecl-runtime-0.10.0/src/server/base.rs:211-251`) — and it fails late and
  loudly in wgpu-core:
  ```text
  --- cube_count 65536 in x (over the 65535 cap): CubeCount(65536, 1, 1) CubeDim(1, 1, 1)
      wgpu error: Validation Error … Each current dispatch group size dimension
      ([65536, 1, 1]) must be less or equal to 65535
      PANIC: called `Result::unwrap()` on an `Err` value: CallError
  ```
  Loud is the right behaviour, and it is the **round-5 hardware-witness territory generalized**: the
  per-axis ceiling is 65535, so a 1-D dispatch tops out at `65535 × 1024 ≈ 6.7e7` threads while a
  2-D one reaches `(6.7e7)^2 ≈ 4.5e15` — the first time a VeriCL-shaped launch can exceed `2^32`
  threads at all (§3.3).
- **Indirect** dispatch is *not* CPU-validated; an over-limit `CubeCount::Dynamic` is rewritten to
  `(0,0,0)` by a generated shader — a **silent no-op**
  (`wgpu-core-29.0.4/src/indirect_validation/dispatch.rs:60-74`). v1 does not use `Dynamic`; R10
  keeps it that way.

Two upstream hazards found while cataloguing, neither owned here:

- **`cubecl-opt`'s integer-range analysis asserts `CUBE_COUNT_a` is the constant `cube_dim.a`** —
  `cubecl-opt-0.10.0/src/analyses/integer_range.rs:162-165` maps `Builtin::CubeCountX =>
  Range::constant(opt.cube_dim.x)`. That is simply the wrong quantity, asserted as a *constant*
  rather than a bound. It is **latent in 0.10** (no consumer of `Ranges` exists outside that file),
  but if it is ever wired into a bounds-check-elimination pass, kernels reading `CUBE_COUNT_*`
  miscompile. Pre-registered as risk 8.
- **`CubeDim::new::<R>(client, working_units)` returns a 2-D `CubeDim`** —
  `cubecl-runtime-0.10.0/src/server/base.rs:1011` is `Self::new_2d(plane_size, plane_count)`. Any
  code that treats "the recommended default cube dim" as 1-D is wrong. VeriCL never calls it
  (`lib.rs:6244` constructs `new_1d` from the suite's `cube_dim` field), which is worth stating
  because it is the reason the §3.3 hole is *latent* rather than live in the suite.
- **`cube_count_spread` silently turns a 1-D request into a multi-axis grid** above the per-axis cap
  (`cubecl-runtime-0.10.0/src/server/base.rs:1091-1117`, reached from `CubeCountSelection::new`).
  VeriCL does not route through it (`lib.rs:6243` builds `CubeCount::Static(count,1,1)` directly),
  but a user who does gets a multi-axis grid — and therefore §2's broken identity — without asking
  for one. Pre-registered as risk 2.

---

## 2. The identity that breaks — the round-5 lesson's 2-D shape (measured)

### 2.1 `ABSOLUTE_POS != CUBE_POS * CUBE_DIM + UNIT_POS`

`crates/vericl-ir/src/prover.rs:1633-1684` (`abs_pos_sym`) and the module docs' "Cooperative mode"
bullet (`:348-384`) state the cooperative model's central equation:

> `AbsolutePos` is *recomputed* from the 1-D identity `CubePos*cube_dim + UnitPos` … **That
> recomputation is the exact *modular* one** …

The **1-D** qualifier in that sentence is doing more work than it looks. Writing `Na` for
`CUBE_COUNT_a`, `Wa` for `CUBE_DIM_a`, `ba` for `CUBE_POS_a`, `ua` for `UNIT_POS_a`:

```
A := ABSOLUTE_POS                   = gz·Gx·Gy + gy·Gx + gx     with ga = ba·Wa + ua, Ga = Na·Wa
B := CUBE_POS·CUBE_DIM + UNIT_POS   = (bz·Nx·Ny + by·Nx + bx)·Wx·Wy·Wz + uz·Wx·Wy + uy·Wx + ux

A − B =  uz·Wx·Wy·(Nx·Ny − 1)
       + uy·Wx·(Nx − 1)
       − by·Nx·Wx·Wy·(Wz − 1)
       − bx·Wx·(Wy·Wz − 1)
```

The `bz` terms cancel; the other four do not. `A` is a row-major index over the **global thread
grid**; `B` is a **cube-blocked (tiled)** index. They are the same permutation only when the tiling
is trivial.

### 2.2 Exactly when it holds — 722/722

`identity_sweep` launches a reporting kernel at every `(CubeCount, CubeDim)` in
`{1,2,3}^3 × {1,2,3}^3` with `≤ 400` threads and compares `CUBE_POS*CUBE_DIM + UNIT_POS` against
each thread's own `ABSOLUTE_POS`:

```text
configurations swept        : 722
identity held (all threads) : 189
identity broke somewhere    : 533
exact algebraic predicate agrees: 722/722
sufficient condition `cube_dim.y == 1 && cube_dim.z == 1`: held in 81/81 such configurations
```

The four free indices range independently, so "equal for every thread" requires each term above to
vanish on its own range. That gives the exact predicate, which agrees with hardware **722/722**:

```
(Wz == 1 ∨ Nx·Ny == 1) ∧ (Wy == 1 ∨ Nx == 1) ∧ (Ny == 1 ∨ Wz == 1) ∧ (Nx == 1 ∨ (Wy == 1 ∧ Wz == 1))
```

which collapses to two useful statements:

- **If `Nx > 1` (more than one cube along X — every realistic dispatch), the identity holds iff
  `CUBE_DIM_Y == CUBE_DIM_Z == 1`.** The cube must be 1-D along X. Nothing about the *cube count*
  matters: `(3,2,1) × (4,1,1)` holds with **0/24** violations, `(3,1,1) × (4,2,1)` breaks with
  **16/24**.
- The other solutions are the degenerate single-cube (`Nx=Ny=Nz=1`) and single-column-of-cubes
  (`Nx=1`) corners. No design should lean on them.

The measured failure is not marginal. At the image-like `(5,3,1)×(8,8,1)`: **912 of 960** threads
violate it, first at `id=8` where `ABSOLUTE_POS=8` but `CUBE_POS*CUBE_DIM+UNIT_POS = 1*64+0 = 64`.

> **`cooperative(cube_dim = N)` is safe, and this is why.** `coop.rs`'s launch and
> `conformance_case` both build `CubeDim::new_1d(cube_dim)` (`coop.rs:1260`, `lib.rs:6244`), so
> `Wy = Wz = 1` holds by construction for every cooperative kernel VeriCL launches. The shipped
> `abs_pos_sym` is correct **because the launch is 1-D**, not because the identity is universal —
> and today nothing in the tree records that dependency. §10.4 correction 1 writes it down.

### 2.3 Why this is a soundness fact, not trivia

Round 5's lesson, in the record's own words, is that *a bound asserted on a derived quantity
transfers backwards onto its components in the model — ask whether hardware honors that transfer*.
The 2-D shape is the same mechanism with an extra step: if a per-axis milestone kept
`abs_pos_sym` as-is and merely *added* per-axis leaves, then a kernel that reads both
`ABSOLUTE_POS` and `UNIT_POS_Y` would be modeled with `ABSOLUTE_POS` pinned to a value the hardware
does not produce, and a guard on it would transfer a bound onto `CUBE_POS`/`UNIT_POS` that is not
merely unhonoured but *arithmetically wrong*. That is a false `Proved` with no wraparound involved
at all.

The design closes it in the only two ways available: the flat `ABSOLUTE_POS` is **rejected** in
per-axis mode (R1, §4.4), and `abs_pos_sym`'s own precondition is **written into the code** as a
`debug_assert`-and-doc pair plus a clause-level gate, so a future 2-D cooperative milestone cannot
reach it accidentally (§10.4 correction 1).

### 2.4 The `u32` flatten wraps — `ABSOLUTE_POS` stops being injective at `2^32` threads

Two measurements compose into one claim, and the claim is the §3.3 hole:

1. The flatten's arithmetic is `u32` — visible in the emitted text (`u32(global_id.z) * … +
   u32(global_id.x)`, `wgsl_dump.txt:49`), and WGSL `u32` arithmetic wraps.
2. The formula itself is exact — **0 violations in 1 212 threads over 6 shapes**, plus 722 further
   shapes in `identity_sweep` where `ABSOLUTE_POS` was a bijection onto `0..n` in every one.

Therefore `ABSOLUTE_POS = (row-major grid index) mod 2^32`, and it ceases to be injective exactly
when `Gx·Gy·Gz ≥ 2^32`. With the measured limits (§1.5: `Na ≤ 65535`, `Wx·Wy·Wz ≤ 1024`), a
**reachable** witness on this adapter is `CubeCount(2048, 2048, 1) × CubeDim(32, 32, 1)` — 1024
units per cube (at the cap), 2048 cubes per axis (well under 65535), `Gx = Gy = 65536`, `Gx·Gy =
2^32`, so the last thread's `ABSOLUTE_POS` wraps to `0` and collides with the first. A 1-D dispatch
**cannot** reach it (`65535 × 1024 < 2^32`), which is why this has never been reachable in VeriCL
before.

*Honesty about what was and was not run.* The mechanism (u32) and the formula (exact) are both
measured; the `2^32`-thread dispatch itself was **not** run on the probe machine — a 4.3-billion-
thread launch risks a GPU watchdog reset on a display-connected device, and the claim does not
need it. It is stated as a derivation from two measurements, and §13 risk 4 records that the
end-to-end witness is outstanding.

---

## 3. What VeriCL does today (measured), and the hole hunt

### 3.1 Three independent gates, all currently closed

| layer | mechanism | file:line |
|---|---|---|
| macro | all 12 per-axis idents + the 4 bare ones in `BANNED_IDENTS`, under the comment `// topology other than ABSOLUTE_POS` | `lib.rs:45-65` |
| macro | the 4 bare ones un-banned **only** under `cooperative(...)` (`COOP_ALLOWED`); "The X/Y/Z variants … stay banned even here (1-D only, §7.3)" | `coop.rs:55-73`, `lib.rs:754-763` |
| prover | `builtin_value`: non-cooperative models only `AbsolutePos`; cooperative adds the four 1-D ones; `// X/Y/Z, plane, cluster builtins: out of the 1-D subset. _ => None` | `prover.rs:1690-1710` |
| interpreter | per-axis variants are *recognised* but pinned 1-D: `AbsolutePosY \| AbsolutePosZ => 0`, `CubeDimY \| CubeDimZ => 1` | `interp.rs:518-537` |
| launch | `conformance_case` hardcodes `CubeCount::Static(count,1,1)` / `CubeDim::new_1d(cube_dim)` | `lib.rs:6242-6245`, `coop.rs:1259-1260` |

The interpreter row is the interesting one: `interp.rs:520-529` already *matches* on
`AbsolutePosX/Y/Z` etc. and answers `0`/`1` for the Y and Z axes. That is correct **only for a 1-D
launch** and is exactly the shape round 11 named a *classification split* — VeriCL's view of the
topology and the device's view diverge the moment a 2-D launch happens. It is not reachable today
(the macro ban means no 2-D kernel exists), but a milestone that lifts the ban and forgets this file
turns a fail-closed rejection into a silently-wrong cross-check. §12 M5 owns it.

### 3.2 Verdicts today — every 2-D shape is `OutOfSubset` at the first guard

`prover2d` builds four kernels' `KernelDefinition`s and runs the real
`vericl_ir::prove_bounds_freedom` at `22e4349`:

| kernel | verdict |
|---|---|
| `ew2d` — `if x < w && y < h { out[(y*w+x) as usize] = … }` | `OutOfSubset { reason: "`if` condition depends on a construct outside the vericl v0 subset" }` |
| `clamp_if` — the same, with `let mut x2 = x; if x+1 < w { x2 = x+1; }` | same |
| `clamp_min` — the same, with `u32::min(x+1, w-1)` | same |
| `flat_decode` — the v0 workaround, `row = ABSOLUTE_POS / w; col = … % w; out[row*w+col]` | **`Proved { obligations: 2 }`** |

**There is no false `Proved` today.** The prover fails closed at the very first guard, because
`AbsolutePosX` taints, so `x < w` taints, so the `if` condition cannot resolve. The macro ban is the
first gate and the prover taint is the backstop, and both hold — this milestone is *not* a
"discovered hole in the accepted path" milestone the way `cube_struct!` was. Its hole is §3.3, which
lives outside the accepted path entirely.

The IR shapes in that run decide two design questions outright:

- **`clamp_if` hits round-2 branch write-taint.** The extracted IR is
  `local(3) = AbsolutePosX; … if(binding(5)) { local(3) = binding(6) }`, i.e. a variable written
  inside an arm. Per `prover.rs`'s "Branch-scoped write taint" discipline, `local(3)` is tainted
  after the branch closes, so the neighbour index is unmodelable **even with per-axis leaves
  modeled**. The idiomatic `if`-based stencil clamp is therefore *not* provable, and no amount of
  per-axis leaf work changes that.
- **`clamp_min` lowers to an `Arithmetic::Min` instruction** — `binding(5) =
  binding(3).min(binding(4)) : (u32, u32) -> (u32)` — which `process_arithmetic` currently drops on
  its `_ => None` arm (`prover.rs:1947`). So the *branch-free* clamp is unmodelable only because
  two arithmetic ops are missing, and adding them is six lines (§4.5).

Without §4.5, the 2-D coverage story is "elementwise and transpose only, no stencil"; with it, the
whole clamped-stencil class proves (§7). That is the single highest-leverage line in the design.

### 3.3 D1 — the latent hole: a proved 1-D kernel says nothing about the launch shape

Today's `#[vericl::kernel]` contract has **no launch-shape term at all**. `conformance_case` picks a
1-D dispatch; `differential_config` records `sizes`, `seed`, `cube_dim` — a *scalar* `cube_dim`
(`evidence.rs:177-184`) — and `proved_config` records solver, logic, obligation count
(`evidence.rs:212-218`). Nothing anywhere says "this evidence was produced under
`CubeCount::Static(n,1,1)`", and nothing constrains the user's own `kernel::launch::<R>(…)` call,
which is an ordinary public cubecl launch fn taking any `CubeCount`/`CubeDim`.

For the *value* claims this is benign and provably so: on any launch shape, `ABSOLUTE_POS` is the
row-major flatten and — as long as it does not wrap — a bijection onto `0..num_threads`, so the
twin's `for ABSOLUTE_POS in 0..num_threads` models exactly the same multiset of positions, and the
prover's opaque `u32` leaf stays faithful. **Measured**: `identity_sweep`'s 722 shapes all had
`ABSOLUTE_POS` bijective onto `0..n`; `blur2d`'s `box_blur3x3_flat` (a flat-`ABSOLUTE_POS` kernel)
is bit-exact against its twin at six image shapes.

The hole is the wrap. Composing §2.4: at `Gx·Gy·Gz ≥ 2^32` — reachable **only** on a multi-axis
grid, and reachable on this adapter at `CubeCount(2048,2048,1) × CubeDim(32,32,1)` — two distinct
threads receive the same `ABSOLUTE_POS`. A kernel whose evidence reads `Proved{n}` +
`differential PASS` then has:

- its **bounds** claim still true (each thread's index is still `< len`, individually);
- its **differential** claim still true (it was measured at 1-D sizes ≤ 65536);
- and a **write-write data race** on `out[ABSOLUTE_POS]` that no claim covers — the plain
  (non-cooperative) walk has no race lane at all, by design.

This is narrow, it needs a deliberate and enormous launch, and it has never been reachable before
because a 1-D dispatch cannot exceed `2^32` threads under the measured per-axis caps. It is
nonetheless a real gap between what the evidence says and what the user can do with the kernel, and
a milestone that *introduces* multi-axis launches is the right place to close the recordable half of
it. §10.4 correction 2 adds the launch shape to the differential config and a `num_threads < 2^32`
precondition to the twin's own contract; §13 risk 4 keeps the residual visible.

### 3.4 An adjacent hole this design surfaces but does not own

`suite!`'s 1-D launch has an unrecorded ceiling: `conformance_case` computes `__vericl_count =
(n as u32).div_ceil(cube_dim).max(1)` and dispatches `CubeCount::Static(count, 1, 1)`
(`lib.rs:6242-6243`). With the measured 65535-per-axis cap, any `sizes:` entry above
`65535 × cube_dim` (16.7 M at the default 256) produces a dispatch wgpu rejects, surfacing as the
`CallError` panic in §1.5 rather than a VeriCL diagnostic. The current suite tops out at 65536, so
nothing trips it. It is **not** owned here — the fix is a size check in `suite!`, orthogonal to
per-axis topology — but a 2-D milestone that teaches users to think about grid extents will make it
reachable sooner, so it is recorded rather than left to be rediscovered.

---

## 4. The decided design

### 4.1 Shape — what the user writes

```rust
#[vericl::kernel(
    dispatch(cube_dim = (16, 16), extents = (w, h)),
    assumes(inp.len() == out.len(), out.len() == (w as usize) * (h as usize)),
    compare(exact),
    gen(inp in -100.0..=100.0, out in 0.0..=0.0)
)]
#[cube(launch)]
pub fn box_blur3x3(inp: &Array<f32>, out: &mut Array<f32>, w: u32, h: u32) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x < w && y < h {
        let x0 = u32::max(x, 1) - 1;
        let x2 = u32::min(x + 1, w - 1);
        let y0 = u32::max(y, 1) - 1;
        let y2 = u32::min(y + 1, h - 1);
        let mut acc = 0f32;
        acc += inp[(y0 * w + x0) as usize];
        /* … seven more … */
        acc += inp[(y2 * w + x2) as usize];
        out[(y * w + x) as usize] = acc * 0.111111111f32;
    }
}
```

```rust
vericl::suite! {
    runtime: cubecl::wgpu::WgpuRuntime,
    kernels: [box_blur3x3],
    evidence: "evidence/vericl.json",
    sizes: [(37, 19), (64, 64), (1, 1), (255, 257)],   // (w, h) tuples in a 2-D suite
}
```

Four things are new: the `dispatch(...)` clause; per-axis idents legal in the body; the
`A.len() == (w as usize) * (h as usize)` assume shape; and tuple `sizes`. Everything else — `gen`,
`compare`, `assumes`, `uses`, `instantiate`, `wrapping`, the evidence flow — is unchanged.

### 4.2 The `dispatch(...)` clause, and why the cube dims must be pinned

```text
dispatch(cube_dim = (Wx, Wy[, Wz]), extents = (e0, e1[, e2]))
```

- **`cube_dim`** is a 2- or 3-tuple of positive **integer literals**. The tuple's arity *is* the
  dispatch rank: a 2-tuple means `CubeDim::new_2d(Wx, Wy)` / `CubeCount::Static(_, _, 1)` and makes
  the Z-axis builtins out of subset; a 3-tuple enables all three. The literals become the prover's
  `CUBE_DIM_X/Y/Z` numerals, the launch's `CubeDim`, and the twin's per-axis loop strides — one
  source of truth, exactly as `cooperative(cube_dim = N)` is today (`lib.rs:3498-3540`,
  `prover.rs:687-694`).
- **`extents`** names the kernel's own runtime `u32` parameters carrying the problem extents. The
  harness derives `CubeCount::Static(ceil(e0/Wx), ceil(e1/Wy), ceil(e2/Wz))`, binds them from the
  case size, and sizes un-pinned buffers to their product.

**Why pinning is load-bearing, not ergonomic.** Every recomposition and every flat form in §1.3
multiplies by some cube dim. With `Wa` a *numeral*, `cube_a·Wa`, `uy·Wx` and `uz·Wx·Wy` are all
variable×constant — **LINEAR, QF_LIA**, precisely the property round 5's fix depended on ("Both
products are variable×constant (`cube_dim` and `2^32` are constants), hence LINEAR — QF_LIA, no
QF_NIA"). With `Wa` a runtime value they are variable×variable, and the whole model degrades to
QF_NIA for facts that are asserted *globally* rather than probed as side-obligations — which is the
one place a solver `unknown` would be silently costly. Pinning is the same trade `cooperative(...)`
already made, for the same reason.

**Why the counts are *not* pinned.** `CUBE_COUNT_X/Y/Z` stay free `u32` leaves. Pinning them would
bind the proof to one launch shape (`docs/design-shared-memory.md` §9 risk 5's exact hazard, one
level worse because the count varies per differential case), and leaving them free costs nothing:
no v1 obligation needs a count bound. **Rejected explicitly**: asserting the hardware ceiling
`CUBE_COUNT_a ≤ 65535` as a leaf fact. It is true on wgpu (§1.5) and false on CUDA
(`2^31 − 1` on X), so it is a *backend-specific* fact, and the round-5 discipline is that the model
asserts only what hardware universally honours.

**Clause gates (D1–D8), all macro-authored, all accumulating:**

| gate | rule |
|---|---|
| D1 | `cube_dim` entries must be integer literals ≥ 1 (a non-literal has no numeral for the prover, and the twin's loop stride would be runtime) |
| D2 | `Wx·Wy·Wz ≤ 1024`, the measured `max_units_per_cube` ceiling on the reference backend; the message names the WebGPU default of 256 |
| D3 | `extents` arity must equal `cube_dim` arity |
| D4 | each `extents` name must be a declared `u32` **runtime** parameter of this kernel (not `#[comptime]`, not an array, not a struct field in v1) |
| D5 | the clause is required iff the body names any per-axis builtin (R2/R3, the `cooperative(...)` biconditional generalized — `lib.rs:3498`, `coop.rs:118-137`) |
| D6 | `dispatch(...)` and `cooperative(...)` are mutually exclusive in v1 (R4) |
| D7 | at most one `dispatch(...)` clause |
| D8 | the body must not name flat `ABSOLUTE_POS` / `CUBE_POS` / `CUBE_COUNT` (R1) |

### 4.3 Prover model — per-axis leaves, per-axis exact modular recomposition

A third entry point beside `prove_bounds_freedom` and `prove_bounds_freedom_cooperative`:

```rust
pub fn prove_bounds_freedom_dispatch(
    def: &KernelDefinition,
    buffers: &[BufferParam],
    assumes: &[Assume],
    cube_dim: [u32; 3],          // the pinned clause tuple, Z = 1 for a 2-D clause
) -> ProveResult
```

carrying `self.dispatch: Option<[u32; 3]>` alongside the existing `self.coop: Option<u32>`. In that
mode `builtin_value` answers:

| builtin | model | logic |
|---|---|---|
| `CubeDimX/Y/Z` | the numeral `Wa` | constant |
| `CubeDim` | the numeral `Wx·Wy·Wz` | constant |
| `UnitPosX/Y/Z` | fresh leaf `unit_a ∈ [0, Wa)` | linear |
| `UnitPos` | `unit_x + unit_y·Wx + unit_z·Wx·Wy` — **a term, not a leaf**; cannot wrap (bounded by the constant `Wx·Wy·Wz`) | linear |
| `CubePosX/Y/Z` | fresh `u32`-range leaf `cube_a` | — |
| `CubeCountX/Y/Z` | fresh `u32`-range leaf `count_a` | — |
| `AbsolutePosX/Y/Z` | fresh `u32`-range leaf `abs_a` **pinned by the per-axis exact modular recomposition** | linear |
| `AbsolutePos`, `CubePos`, `CubeCount` | unreachable — R1 rejects them at the macro; the prover keeps `None` as the fail-closed backstop | — |

**The recomposition, per axis** (`abs_pos_axis_sym(a)`, the direct generalization of
`abs_pos_sym`, `prover.rs:1653-1684`):

```
declare abs_a   with  0 ≤ abs_a  ≤ 2^32 − 1
declare wrap_a  with  0 ≤ wrap_a ≤ Wa − 1
assert  abs_a = cube_a · Wa + unit_a − wrap_a · 2^32
```

Both products are variable×constant. The ceiling `wrap_a ≤ Wa − 1` is the same tight-but-not-
soundness-critical bound round 5 derived (`cube_a ≤ 2^32−1 ∧ unit_a ≤ Wa−1 ⟹ raw ≤ 2^32·Wa − 1`).
The encoding is **exact**: a value in `[0, 2^32)` congruent to the raw sum mod `2^32` is its unique
residue, i.e. the true hardware value, so the module invariant — *every non-tainted modeled integer
term equals the real hardware value* — is preserved.

**Measured, both directions** (`smt/p1*.smt2`, both < 10 ms):

| probe | encoding | verdict | meaning |
|---|---|---|---|
| `p1` | unwrapped `abs_x = cube_x·256 + unit_x`, guard `abs_x < len`, obligation `out[CUBE_POS_X]` | **unsat** | **false `Proved`** — round 5's defect, per axis |
| `p1b` | exact modular with `wrap_x` | **sat**, `cube_x=16843009, unit_x=0, abs_x=16843008, wrap_x=1, len_out=16843009` | honest `Refuted`; `16843009·256 − 2^32 = 16843008 = len−1 < len` while `cube_x = len` |

**Predeclaration is mandatory and is three symbols per axis.** `abs_pos_axis_sym` emits
`declare-const`s and assertions, so — exactly as round 5 established for `abs_pos_sym` — every leaf
it can reach must be declared at the **outermost** SMT scope, or a lazy first resolution inside a
branch arm scopes the declaration to that arm and `pop` drops it. `predeclare_dispatch_leaves`
declares, for each axis the clause enables: `unit_a`, `cube_a`, `count_a`, `abs_a`, `wrap_a` — five
per axis, up to fifteen. An unused leaf is a harmless free nonnegative constant, precisely as
`predeclare_coop_leaves` already argues (`prover.rs:1571-1592`).

> **Round-7's queued predeclaration hazard is now on the critical path.** `tasks/todo.md:1552-1655`
> classified the same lazy-declaration shape as still latent in the **loop** handlers, LOW because
> no evidence-producing flow reaches it. A 2-D kernel whose per-axis leaf is first resolved inside a
> grid-stride `while` body is exactly such a flow. §12 M2 fixes it rather than inheriting it, and
> §13 risk 3 pre-registers it as an attack surface.

### 4.4 The flat builtins in per-axis mode: rejected, and the measurement that decides it

Modeling `ABSOLUTE_POS` from the per-axis leaves requires `abs_y · (count_x · Wx)` — variable ×
variable, since `count_x` is a runtime leaf. Same for `CUBE_POS` (`cube_y · count_x`) and
`CUBE_COUNT` (`count_x · count_y`). Three options were considered and two rejected:

| Alternative | Why not |
|---|---|
| Model them in QF_NIA anyway | **Measured**: the 2-D analogue of the inter-cube single-writer gate — "two threads in different cubes cannot write the same `out[abs_y·w + abs_x]`" — **times out in z3 at 180 s** (`p5`). The 1-D version of the same check is a pattern match on `VariableKind::Builtin(AbsolutePos)` costing **no SMT at all** (`prover.rs:3003-3005`). Trading an O(1) pattern for a 180 s timeout is not a trade |
| Pin `CUBE_COUNT` per axis in the clause too, making everything linear | Binds the certificate to one launch shape, and the count legitimately varies per differential case (`ceil(w/Wx)` at each `sizes` entry). This is `design-shared-memory.md` §9 risk 5 with a per-case multiplier |
| **Reject the three flat builtins inside `dispatch(...)`** (chosen) | The two addressing schemes are different permutations (§2.1), so mixing them in one kernel is a coherence question the *author* should answer, not a modeling question VeriCL should paper over. Costs nothing measured: an image kernel indexes `y·w + x` with its own runtime width, never `ABSOLUTE_POS` |

Flat `CUBE_DIM` and `UNIT_POS` are **kept**: with the dims pinned they are a numeral and a
pinned-coefficient linear form respectively, and `UNIT_POS` cannot wrap (its value is bounded by the
constant `Wx·Wy·Wz`, so no `wrap_to_range` correction is needed and none is emitted).

### 4.5 Two newly-modeled ops: `Arithmetic::Min` / `Max` as exact `ite`

```rust
Arithmetic::Max(b) => self.minmax_int(b, &out.ty, /* is_max */ true),
Arithmetic::Min(b) => self.minmax_int(b, &out.ty, /* is_max */ false),
```

resolving both operands under the existing `is_modeled_int` gate and emitting
`ite(l >= r, l, r)` / `ite(l <= r, l, r)`. **No side-obligation, no wrap correction, no new leaf**:
the result is *one of the operands*, both of which are in range by the leaf/faithful-term invariant,
so the term is exact and in range by construction. Floats are excluded by the pre-existing
`is_modeled_int(&out.ty)` guard at `prover.rs:1926` (and integer min/max has no NaN question).

This is the difference between "2-D elementwise and transpose" and "the whole stencil class":

- The **`if`-based** clamp is unrecoverable — round-2 branch write-taint tainted `local(3)` and that
  is the correct, deliberate behaviour (§3.2). It is not worth relaxing for this milestone.
- The **branch-free** clamp is the only remaining spelling, it lowers to `Arithmetic::Min`/`Max`
  (measured IR, §3.2), and with the `ite` handler the full 3×3 clamped stencil obligation is
  **UNSAT in 0.20 s** (`p2c`, §7).

R11's wording tells a user who wrote the `if` form to switch spellings, naming both.

### 4.6 The new structured assume — and the round-4 gate it needs

**`Assume::LenEqProduct { a, x, y }`**, recognized from
`A.len() == (x as usize) * (y as usize)` where `x`,`y` are runtime `u32` scalar parameters, asserted
as `len_a = x · y` over the two scalar leaves and `A`'s length leaf.

It is **necessary and sufficient**, measured:

| probe | context | verdict |
|---|---|---|
| `p2e` | existing assume vocabulary only (`inp.len() == out.len()`), obligation `y·w + x < out.len()` | **sat** — Refuted; nothing ties extents to a length |
| `p2a` | with `len == w·h`, the `checked_mul` side-obligation `0 ≤ abs_y·w ≤ 2^32−1` | **unsat** in 0.13 s — the row stride is *bound*, not tainted |
| `p2b` | with `len == w·h`, the write obligation `0 ≤ wrap(y·w + x) < out.len()` | **unsat** — Proved |
| `p2f` | the weaker one-directional `w·h ≤ out.len()` | **unsat** — also proves; the safer direction, kept as a sibling shape |

`p2a` is the crux: the `checked_mul` side-obligation on the *row stride* is what decides whether a
2-D kernel is provable at all, and it discharges because `abs_y ≤ h−1 ⟹ abs_y·w ≤ w·h − w = len −
w ≤ len ≤ 2^32−1`. Without the product assume there is no such chain — this is precisely what the
1-D `flatten_decode_scale` example gets *for free* from Euclidean division (`row·w ≤ ABSOLUTE_POS`,
`tasks/todo.md:1183`) and what genuine 2-D loses (§5).

**The round-4 gate (R6), and its measured witness.** The assume must be recognized only in the
**widen-then-multiply** spelling. Written the other way — `out.len() == (w * h) as usize` — the
executable `check_assumes` evaluates the **wrapped `u32`** product while a naive recognizer asserts
the **mathematical** one, and that gap is a false `Proved` of exactly round 4's shape ("the
recognized form must *imply* the claimed property"). Measured witness (`p3`, sat in 0.13 s):

```text
w = 2 , h = 2147483649 , k = 1 , len_real = 2 , abs_x = 0 , abs_y = 1
```

`w·h = 4 294 967 298`; wrapped to `u32` that is `2`, so a **length-2** buffer satisfies the host
predicate while the model asserts `len = 4 294 967 298`. The index `abs_y·w + abs_x = 2` then proves
in bounds against the model and is out of bounds against reality. R6 rejects the outer-cast spelling
by name, at the cast's own span, and — per round 4 — the recognizer peels an LHS/RHS cast **only**
when it is value-preserving for the operand's own type, reusing `cast_is_value_preserving`'s
existing width/signedness logic rather than a fresh one.

**Deliberately not implied by the clause.** The harness *will* size buffers to `w·h`, which makes
the assume true on the differential lane — but the prover must not learn it from the clause.
Asserting a fact that only the harness's launch guarantees is the same category error as pinning the
cube count (§4.2): a hand-launched kernel with an over-sized or under-sized buffer would carry a
certificate that never applied to it. The user states it, `check_assumes` tests it, evidence records
it — the ordinary path for every length assumption VeriCL has.

**Evidence honesty.** A `len == x·y` assertion is nonlinear, so `proved_config`'s hardcoded
`"logic": "QF_LIA"` (`evidence.rs:212-218`) becomes wrong for these kernels. §10.4 correction 3
makes the field reflect the emitted context (`"QF_NIA"` when any `LenEqProduct` is in scope). The
counterexample validator needs **no** change: `eval_sexpr` already interprets integer `*`
(`prover.rs:23`).

### 4.7 Twin treatment — a nested loop over the grid, in flat order

The ordinary twin rewrites `ABSOLUTE_POS` to a loop variable over `0..num_threads`
(`lib.rs:729-735`, `:4137-4142`). The 2-D twin is the same idea with one loop per enabled axis:

```rust
pub fn reference(inp: &[f32], out: &mut [f32], w: u32, h: u32,
                 grid: (u32, u32, u32)) {
    for __vericl_abs_z in 0..grid.2 {
        for __vericl_abs_y in 0..grid.1 {
            for __vericl_abs_x in 0..grid.0 {
                /* body, with ABSOLUTE_POS_X -> __vericl_abs_x, … */
            }
        }
    }
}
```

Four properties, each decided rather than assumed:

1. **Loop order is Z outer → Y → X inner, which reproduces the flat `ABSOLUTE_POS` order exactly**
   (§1.3's row-major flatten). So the twin's write-ordering convention for aliasing writes is
   *unchanged* from the 1-D twin, and a kernel ported from flat to per-axis addressing keeps the
   same reference semantics. Any other nesting would silently change the convention.
2. **The loops range over the GRID, not the image** (`grid.0 = ceil(w/Wx)·Wx ≥ w`), so the padding
   threads run the guard exactly as on device. A twin that looped `0..w` would model a *different*
   kernel — one with no padding — and would agree with the device only by luck.
3. **The loop variables are `u32`, not `usize`**, matching `ABSOLUTE_POS_X`'s frontend type (§1.1).
   Index expressions carry the author's own `as usize`, as they must in the kernel.
4. **`CUBE_DIM_a` rewrites to the pinned literal; `CUBE_COUNT_a` rewrites to a derived binding**
   (`grid.a / Wa`). `UNIT_POS_a` and `CUBE_POS_a` rewrite to `__vericl_abs_a % Wa` and
   `__vericl_abs_a / Wa` — the per-axis decomposition, which is exact for every axis
   (**measured 0 violations / 1 212 threads**, check (5) in §1.3) and needs none of §2's broken
   flat identity. Flat `CUBE_DIM` and `UNIT_POS` rewrite to the corresponding constant / linear
   combination. This is why the twin can support `UNIT_POS_a` while the flat `ABSOLUTE_POS` is
   rejected: the per-axis relations are individually sound; only the flat *cross-axis* one is not.

**Measured bit-exact** at six image shapes × three kernels (§6).

### 4.8 Launch, `gen(...)`, `sizes`, and evidence

| surface | 1-D today | 2-D |
|---|---|---|
| `suite!` `sizes` | `[usize]` | `[(usize, usize)]` / `[(usize, usize, usize)]`, arity = the clause's |
| `generate_case` | `(n, seed)` | `(extents, seed)`; extents bind the named `u32` params, never drawn |
| default buffer length | `n` | `∏ extents` |
| `gen(len(name = N))` | pins one buffer | unchanged — a transpose's output is the same product, an integral image's is `(w+1)(h+1)` |
| `CubeCount` | `Static(ceil(n/cd), 1, 1)` | `Static(ceil(e0/Wx), ceil(e1/Wy), ceil(e2/Wz))` |
| `CubeDim` | `new_1d(cube_dim)` (suite field) | `new_2d/3d(clause literals)`; the suite's `cube_dim` field is **rejected** together with `dispatch(...)` (R7) — two sources of truth for one launch parameter is the hazard `cooperative(...)` already avoids by asserting them equal (`coop.rs:1016`) |
| `num_threads` | `count · cube_dim` | not passed; the twin takes the `grid` triple |
| evidence config | `differential_config(sizes, seed, cube_dim)` | `differential_dispatch_config(sizes, seed, cube_dim: [u32;3], rank)` with `sizes_unit: "extents"` — the `differential_vector_config` precedent (`evidence.rs:194-208`), which added `sizes_unit: "lines"` for the same reason |

### 4.9 Rejected alternatives

| Alternative | Why not |
|---|---|
| **Model the flat builtins from the per-axis leaves in QF_NIA** | `p5` times out at 180 s on the query the 1-D path answers with a pattern match; and it would re-import §2's broken identity into the model |
| **Pin `CUBE_COUNT` per axis in the clause** | binds the certificate to one launch shape, and the count varies per differential case by construction (`ceil(e/W)`) |
| **Assert the hardware ceiling `CUBE_COUNT_a ≤ 65535`** | backend-specific (wgpu 65535, CUDA `2^31−1`); the model asserts only universally-honoured facts (round 5) |
| **Derive the `len == w·h` fact from the `dispatch(extents = …)` clause** | asserts something only the harness's own launch guarantees; a hand-launched kernel would carry a certificate that never applied. Same category error as pinning the counts |
| **Keep `abs_pos_sym` and just add per-axis leaves alongside** | the 2-D `ABSOLUTE_POS` is not `CUBE_POS·CUBE_DIM + UNIT_POS` (533/722 shapes), so the flat leaf would be pinned to a value hardware never produces — a false `Proved` with no wraparound involved |
| **A flat twin loop `for i in 0..grid_total { x = i % Gx; … }`** | needs a runtime `Gx` division in the twin where the nested form needs none, and re-introduces the div/mod modeling question on the *reference* side. The nested form is also what makes the guard structure textually identical to the kernel's |
| **Relax round-2 branch write-taint so the `if`-based clamp models** | that taint is a confirmed-critical fix (round 2, three false-`Proved` manifestations). Adding `Min`/`Max` gets the same coverage at zero soundness cost |
| **Support 2-D via a `#[comptime]` width and flat `ABSOLUTE_POS`** | already works today (`box_blur3x3_flat` is bit-exact and `flat_decode` is `Proved{2}`) and is documented as the v0 workaround — but it forces a `/`+`%` per thread, cannot express a non-power-of-two tile, and gives up the hardware's own 2-D scheduling. It is the *baseline*, not the design |

---

## 5. Row-major decode/encode — what the existing machinery already covers, and what it does not

The task asked whether the canonical `y·W + x` ↔ `i/W`, `i%W` arithmetic is covered **both ways**.
Measured, the answer is asymmetric, and the asymmetry is the whole reason 2-D needs a new assume.

**Decode → re-encode: already `Proved`, and free.** `flatten_decode_scale`
(`crates/vericl-examples/src/lib.rs:225-238`) writes `y[row·w + col]` with
`row = ABSOLUTE_POS / w`, `col = ABSOLUTE_POS % w`, and is `Proved{2}` in the shipped suite. An
independent clean-room re-spelling of the same shape (`prover2d`'s `flat_decode`, different
parameter names and a literal scale) returns **`Proved { obligations: 2 }`** against the real
`prove_bounds_freedom` at `22e4349`, so the property is the shape's, not the example's. Two facts
make it work and neither survives translation to per-axis form:

- the div/mod side-obligation discharges because `w ≥ 1` is *in the guard* and both operands are
  unsigned (`prover.rs:267-291`);
- the `checked_mul` side-obligation on `row·w` discharges from the **Euclidean fact**
  `w · (a div w) ≤ a` together with the leaf bound `ABSOLUTE_POS ≤ 2^32−1` — the product is bounded
  by a quantity the model already has (`tasks/todo.md:1183`). The `Add` back to `row·w + col` is
  then faithful and z3 recovers `row·w + col == ABSOLUTE_POS`, so the guard transfers.

**Encode from independent axes: a genuinely new boundary.** `abs_y · w` has no Euclidean parent.
`abs_y` and `w` are unrelated leaves, so `checked_mul`'s `0 ≤ abs_y·w ≤ 2^32−1` is *unprovable* from
the guard alone — measured **sat** in `p2e`, i.e. the model correctly refuses. The only fact that
bounds it is a relation between the extents and a buffer length, which is exactly `LenEqProduct`
(§4.6), and with it the obligation is **unsat in 0.13 s** (`p2a`).

Two consequences worth stating plainly:

1. **The product assume is not ergonomics, it is the enabling fact.** A v1 that shipped per-axis
   leaves without it would produce `OutOfSubset` on every 2-D kernel that indexes an array — i.e.
   all of them — which is the "safe but useless" outcome the brief warned about.
2. **The same hazard already exists elsewhere and this milestone does not widen it.** A wide
   `slice_mut` window's `start = i·W` stride is unprovable for the identical reason
   (`tasks/todo.md:1869-1873`). The per-axis case is the *first* one with a natural contract shape
   that discharges it, which is a small argument that `LenEqProduct` is worth having beyond 2-D.

**Transpose exercises both directions in one kernel** — `out[x·h + y] = inp[y·w + x]` — with two
independent `checked_mul` obligations against two different extents. Both discharge under
`inp.len() == (w as usize)·(h as usize)` and `out.len() == inp.len()`, and the kernel is exact on
hardware at all six shapes (§6).

---

## 6. Ground-truth probe (validated)

`scratchpad/design2d/probe/src/bin/blur2d.rs`: four clean-room kernels, each hand-twinned as a
nested grid loop per §4.7, compared **bit-for-bit** (`f32::to_bits`) against wgpu/Metal at six image
shapes chosen so `w ≠ h`, neither is a multiple of the cube dim, and the degenerate `1×1` and the
thin `3×129` / `129×3` cases are covered.

- **`box_blur3x3`** — 2-D dispatch, per-axis guard, clamped 3×3 neighbourhood, row-major `y·w + x`.
  Nine `f32` adds in a fixed order then one multiply by `0.111111111f32`; inputs are small exact
  integers so the nine-term sum is itself exact and the single multiply is the only rounding, and it
  is the *same* single rounding on both sides. No mul-add chain, so no FMA-contraction question.
- **`transpose`** — `out[(x·h + y)] = inp[(y·w + x)]`, `u32` elements so bit-exactness is
  unambiguous.
- **`elementwise2d`** — the coverage floor.
- **`box_blur3x3_flat`** — the *same* blur addressed by flat `ABSOLUTE_POS` with a `/`+`%` decode,
  i.e. the v0 workaround, run against the *same* twin. Included so the design can state what 2-D
  dispatch buys and what it does not.

```text
=== w=37 h=19 cube_dim=(8,8) cube_count=(5,3) grid=(40,24) threads=960 image=703 ===
  box_blur3x3  (2-D dispatch) bit-exact: true
  box_blur3x3_flat (1-D, v0 workaround) bit-exact: true
  transpose    (2-D dispatch) exact: true
  elementwise2d(2-D dispatch) bit-exact: true
=== w=255 h=257 cube_dim=(16,4) cube_count=(16,65) grid=(256,260) threads=66560 image=65535 ===
  box_blur3x3  (2-D dispatch) bit-exact: true
  box_blur3x3_flat (1-D, v0 workaround) bit-exact: true
  transpose    (2-D dispatch) exact: true
  elementwise2d(2-D dispatch) bit-exact: true

=== ALL BIT-EXACT: true ===
```

(Six shapes × four kernels = **24 / 24 bit-exact**, `0` differing bits; full run in `RESULTS.txt`.)

**What this establishes, three things at once.** (1) The §4.7 nested-loop twin is the right model —
including the padding threads, which at `37×19 / (8,8)` are 257 of 960 and all take the `else`
branch. (2) There is no float-ordering or contraction divergence introduced by 2-D dispatch. (3) The
flat workaround and the per-axis form compute **the same function** against **the same twin**, which
is what makes §11's "what v1 buys" claim a measurement rather than a slogan.

**One obligation, hand-discharged.** For `box_blur3x3` at the bottom-right neighbour — the worst
index in the kernel — the obligation the milestone must produce is
`0 ≤ min(y+1, h−1)·w + min(x+1, w−1) < inp.len()` under the live facts `abs_x < w`, `abs_y < h`,
`inp.len() == w·h`, with `Min` modeled as an exact `ite`, `Add`/`Sub` wrap-faithful, and the
`checked_mul` side-obligation discharged. Written out verbatim in
`scratchpad/design2d/smt/p2c_neighbour_clamped.smt2` and discharged **unsat in 0.20 s**. Its
negative control — the same read with the clamp deleted, `inp[(y+1)·w + (x+1)]` — is **sat in
0.14 s** with the minimal witness `w=1, h=1, x=0, y=0`, i.e. a `1×1` image where the kernel reads
`inp[2]` from a length-1 buffer (`p2d`). The gate discriminates.

---

## 7. Prover treatment measured end to end — the SMT ledger

Every encoding decision in §4 was written out as SMT-LIB and run against z3 4.16.0. This is the
complete ledger; all twelve are in `scratchpad/design2d/smt/`.

| # | question | logic | verdict | time | meaning |
|---|---|---|---|---|---|
| `p1` | unwrapped per-axis recomposition, round-5 attack | QF_LIA | **unsat** | <0.01 s | **false `Proved`** — the defect transplants per axis |
| `p1b` | exact modular per-axis recomposition, same attack | QF_LIA | **sat** + witness | <0.01 s | honest `Refuted` — the fix transplants too |
| `p2a` | `checked_mul` side-obligation on the row stride `abs_y·w` | QF_NIA | **unsat** | 0.13 s | the product binds instead of tainting |
| `p2b` | 2-D write obligation `y·w + x < out.len()` | QF_NIA | **unsat** | <0.01 s | elementwise-2D proves |
| `p2c` | 3×3 clamped neighbour, `Min` as exact `ite` | QF_NIA | **unsat** | 0.20 s | the stencil class proves |
| `p2d` | **negative control** — the same, unclamped | QF_NIA | **sat** + `w=1,h=1,x=0,y=0` | 0.14 s | the gate discriminates |
| `p2e` | `p2b` **without** the product assume | QF_NIA | **sat** | 0.04 s | the assume is *necessary* |
| `p2f` | the weaker `w·h ≤ out.len()` direction | QF_NIA | **unsat** | <0.01 s | the safe direction also proves |
| `p3` | round-4 shape: `(w*h) as usize` vs `(w as usize)*(h as usize)` | QF_NIA | **sat** + `w=2, h=2147483649, len=2` | 0.13 s | R6 is mandatory |
| `p4` | 2-D cooperative race: two units of a `(16,16)` cube writing `tile[uy·16 + ux]` | QF_LIA | **unsat** | <0.01 s | tile indexing is injective, cheaply |
| `p4b` | **negative control** — the racy `tile[ux]` | QF_LIA | **sat** + `(0,0)` vs `(0,1)` | <0.01 s | the gate discriminates |
| `p5` | 2-D inter-cube single-writer for `out[abs_y·w + abs_x]` | QF_NIA | **TIMEOUT at 180 s** | — | the flat-builtin rejection (§4.4) and the v1.1 line (§8) |

Three readings:

- **Nothing in the v1 path is slow.** The worst v1 query is 0.20 s. The only timeout is `p5`, which
  v1 does not emit (it belongs to the race walk, which `dispatch(...)` kernels do not enter).
- **The logic label moves.** Eight of the twelve are genuinely QF_NIA because the product assume and
  the `checked_mul` side-obligation are nonlinear. `checked_mul` already emits variable×variable
  products today, so `proved_config`'s hardcoded `"QF_LIA"` was already a slight over-claim; §10.4
  correction 3 makes it honest rather than introducing the problem.
- **Both negative controls fire.** Per round 8's "which test actually discriminates?", each positive
  result is paired with the smallest defect injection that must flip it — the deleted clamp (`p2d`)
  and the dropped `y` term (`p4b`) — and both do.

---

## 8. Cooperative 2-D — measured tractability, and where the v1.1 line falls

2-D workgroups + shared memory is the full image-tile pattern and the obvious next ask. Measured
rather than guessed, it splits cleanly into a tractable half and an intractable one, and the split
is where v1 stops.

**Tractable — the intra-cube half.** The two-thread race abstraction generalizes by replacing the
scalar `t1 ≠ t2` with a **tuple** distinctness `(t1x,t1y,t1z) ≠ (t2x,t2y,t2z)`, i.e. a disjunction
of three inequalities. Shared-tile indices stay linear because the cube dims are pinned. `p4`
discharges the canonical `tile[uy·Wx + ux]` write-write obligation **unsat in <0.01 s**, and `p4b`
refutes the racy `tile[ux]` variant with the two-thread witness `(0,0)` vs `(0,1)`. The
phase-split twin generalizes the same way: `coop.rs`'s `for cube { for seg { for unit_pos { … } } }`
(`coop.rs:11-17`) becomes a per-axis nest, with the same segment boundaries.

**Intractable in v1 — the inter-cube half.** `check_intercube_global` (`prover.rs:2975-3052`) proves
global-write disjointness across cubes by recognizing exactly two *patterns*:
`out[ABSOLUTE_POS]` (globally unique) and single-writer `out[CUBE_POS]`. Both patterns are gone in
per-axis mode (R1 rejects the flat builtins), and the natural 2-D replacement
`out[ABSOLUTE_POS_Y·w + ABSOLUTE_POS_X]` cannot be discharged by SMT — `p5` **times out at 180 s**
because `abs_y·w` is variable×variable on both threads. It *is* provable by a **pattern** (the
guard `x < w ∧ y < h` makes `(y,x) ↦ y·w + x` injective on the guarded domain), but that is a new
recognizer with its own soundness argument, not a generalization of an existing one.

**Therefore: `dispatch(...)` and `cooperative(...)` are mutually exclusive in v1 (R4, D6).** The
rejection points at this section rather than saying "unsupported", and §10.5 records the three
concrete pieces v1.1 needs: the tuple-distinctness race setup, the per-axis phase-split twin, and
the 2-D inter-cube write-pattern recognizer. The measurement says the first two are cheap and the
third is the real work — which is a much better place to start a milestone from than a guess.

**One thing v1 must do now to keep v1.1 reachable.** `abs_pos_sym`'s correctness *depends* on the
launch being 1-D in the cube (§2.2), and today nothing says so. §10.4 correction 1 adds the
statement to `prover.rs`'s module docs, to `prove_bounds_freedom_cooperative`'s rustdoc, and — as a
`REQUIRED WORK` note in the round-9 F4 style — to the site a v1.1 implementer will actually touch.

---

## 9. Compatibility matrix

Every cell measured unless marked. "PASS" = wgpu/Metal differential green at the listed sizes.

| Feature × 2-D/3-D dispatch | v1 | Evidence |
|---|---|---|
| 2-D elementwise (`out[y·w+x] = f(inp[y·w+x])`) | **support** | §6 `elementwise2d` bit-exact 6/6; `p2b` unsat |
| 2-D transpose (`out[x·h+y] = inp[y·w+x]`) | **support** | §6 exact 6/6; two independent `checked_mul`s discharge (§5) |
| 3×3 / 5×5 clamped stencil, blur | **support** | §6 `box_blur3x3` bit-exact 6/6; `p2c` unsat 0.20 s; needs `Min`/`Max` (§4.5) |
| stencil with an **`if`-based** clamp | **reject** (targeted, R11) | round-2 branch write-taint tains the clamped local (measured IR, §3.2); rewrite branch-free |
| 3-D dispatch (`cube_dim` 3-tuple) | **support** | same machinery; `axis_order` `(3,2,2)×(2,3,2)` 0 violations / 144 |
| grid-stride 2-D loop over `CUBE_COUNT_a · CUBE_DIM_a` | **support** | `CubeCountX/Y/Z` are free `u32` leaves; the loop is the existing `Branch::Loop` break-guard shape. Requires §12 M2's predeclaration fix |
| flat `ABSOLUTE_POS` / `CUBE_POS` / `CUBE_COUNT` inside `dispatch(...)` | **reject** (targeted, R1) | §2.2 (533/722 shapes break the identity), `p5` timeout |
| flat `CUBE_DIM` / `UNIT_POS` inside `dispatch(...)` | **support** | a pinned numeral and a pinned-coefficient linear form; neither can wrap |
| `CUBE_POS_X/Y/Z`, `UNIT_POS_X/Y/Z` | **support** | per-axis decomposition exact, 0 violations / 1 212 threads (check (5), §1.3) |
| `#[comptime]` params / generics × `dispatch(...)` | **support** (inherited) | orthogonal: `instantiate(...)` is unchanged; a `#[comptime]` param may **not** be an `extents` name (D4) |
| `vericl::config!` comptime struct × `dispatch(...)` | **support** (inherited) | the clause reads kernel *parameters*, not config fields |
| `vericl::cube_struct!` runtime struct × `dispatch(...)` | **support**, with one restriction | a struct **field** may not be an `extents` name in v1 (D4) — the `gen(p.field)` two-segment grammar exists but the launch-side extent plumbing would need it too; R12 |
| **`Vector<P, W>`** × `dispatch(...)` | **reject** (targeted, R13) → v1.1 | a `Vector` kernel's `sizes` are *line* counts (`differential_vector_config`, `evidence.rs:194-208`) while a 2-D suite's are *extents*; two units in one config with no decided reconciliation. Round 8's "keep multi-unit quantities opaque in units" says decide it, not guess it |
| core `Slice` × `dispatch(...)` | **support** (inherited) | a slice access is indistinguishable in the IR from `origin[offset + i]` (`docs/design-view-slice.md`); the origin obligation is per-axis-agnostic |
| **gather** (`inp[offsets[i]]`) × 2-D | **support** (inherited) | `ElemsBelowLen` is a content assume over an array; the *index* being 2-D-derived changes nothing |
| **cooperative** (`cooperative(cube_dim = N)`) × `dispatch(...)` | **reject** (targeted, R4) → v1.1 | §8: intra-cube measured tractable (`p4`/`p4b`), inter-cube measured **not** (`p5` 180 s timeout) |
| `SharedMemory` × `dispatch(...)` | **reject** (inherited via R4) | `SharedMemory` requires `cooperative(...)`, which R4 excludes |
| `uses(...)` composition | **support** | a helper's twin cannot read topology at all (`lib.rs:702-710`, banned) — per-axis positions are passed as plain `u32` arguments, exactly like `ABSOLUTE_POS` today |
| `wrapping` × `dispatch(...)` | **support** | the clause is about integer *ops* in the body; per-axis positions add none. A wrapped **index** is still out of bounds (`prover.rs:259-266`) — unchanged |
| `assumes(A.len() == B.len())`, `LenEqConst`, `LenPlusConstLe` | **support** (inherited) | unchanged; asserted over the same length leaves |
| `assumes(A.len() == (w as usize) * (h as usize))` | **support** (new shape) | `p2a`/`p2b` unsat; §4.6 |
| `assumes(A.len() == (w * h) as usize)` | **reject** (targeted, R6) | `p3` sat — false `Proved` with `w=2, h=2147483649, len=2` |
| `gen(len(name = N))` × `dispatch(...)` | **support** | unchanged; the default becomes `∏ extents` |
| f64 lane × `dispatch(...)` | **support** (inherited) | the compare tier still comes from the `ArrayMut` element type (`lib.rs:3390-3419`); topology is `u32` regardless |
| `suite!` `cube_dim:` field × `dispatch(...)` | **reject** (targeted, R7) | two sources of truth for one launch parameter; the clause wins |
| IR interpreter cross-check × `dispatch(...)` | **support**, after M5 | `interp.rs:520-529` currently pins Y/Z to `0`/`1` — correct for 1-D, a classification split for 2-D. M5 makes it grid-aware; until then the cross-check lane must report `Unsupported`, not guess |
| `kernel_ir_hash` × a `cube_dim` change | **support** (already, via `SOURCE_HASH`); **improved** in M6 | `hash.rs:80` already folds `def.cube_dim` — but `kernel_definition()` builds with `KernelSettings::default()` (`lib.rs:4375`), whose `cube_dim` is `(1,1,1)` (measured, §1.2), so the field is **constant across every VeriCL kernel** and contributes nothing today, including for `cooperative(cube_dim = N)`. A clause edit still stales evidence, because the attribute tokens are in `SOURCE_HASH` (`lib.rs:3699-3703`). M6 threads the pinned dims into `KernelSettings` so `ir_hash` moves too (§10.4 correction 4) |
| indirect dispatch (`CubeCount::Dynamic`) | **reject** (targeted, R10) | over-limit becomes a silent `(0,0,0)` no-op (`wgpu-core/src/indirect_validation/dispatch.rs:60-74`) — a launch that does nothing while every claim stays green |

No silent gaps: every feature is supported, deferred-with-rejection, or out with the rejection site
named.

---

## 10. The v1 subset boundary

### 10.1 Contract / macro additions

1. `dispatch(cube_dim = (…), extents = (…))` — new clause, parsed in `parse_contract`
   (`lib.rs:480-685`), stored as `ContractSpec::dispatch: Option<(Span, [u32; 3], u8, Vec<Ident>)>`
   (span, pinned dims, rank, extent parameter names).
2. `Assume::LenEqProduct { a, x, y }` — sixth structured-assume variant
   (`prover.rs:544-567`), mirrored in `vericl::StructuredAssume` (`contract.rs`).
3. `prove_bounds_freedom_dispatch(def, buffers, assumes, cube_dim: [u32; 3])` — third prover entry
   point.
4. `Arithmetic::Min` / `Arithmetic::Max` modeled (`prover.rs:1931-1948`).
5. `vericl::differential_dispatch_config(sizes, seed, cube_dim, rank)` (`evidence.rs`).
6. `suite!` `sizes:` accepts tuples; `cube_dim:` becomes invalid alongside a `dispatch(...)` kernel.

### 10.2 Accepted (v1)

A `#[vericl::kernel]` declaring `dispatch(cube_dim = (W…), extents = (e…))` where:

- every `cube_dim` entry is a positive integer literal and their product is `≤ 1024` (D1, D2); and
- `extents` has the same arity as `cube_dim`, and every name is a declared runtime `u32` parameter
  of this kernel (D3, D4); and
- the kernel declares no `cooperative(...)` clause (D6); and
- the body names at least one per-axis builtin (D5) and no flat
  `ABSOLUTE_POS`/`CUBE_POS`/`CUBE_COUNT` (D8).

Inside such a body the following become legal, for each axis the rank enables:
`ABSOLUTE_POS_X/Y/Z`, `CUBE_POS_X/Y/Z`, `UNIT_POS_X/Y/Z`, `CUBE_DIM_X/Y/Z`, `CUBE_COUNT_X/Y/Z`,
plus flat `CUBE_DIM` and `UNIT_POS`. Everything else about the body is governed by the existing
subset.

### 10.3 Rejected, with targeted errors

**R1 — a flat topology builtin inside a `dispatch(...)` kernel** (macro-authored, at the ident's
span):

> ``error: `ABSOLUTE_POS` is outside the vericl v0 subset in a `dispatch(...)` kernel — in a multi-axis dispatch it is NOT `CUBE_POS * CUBE_DIM + UNIT_POS`, it is the row-major flatten of the whole thread grid, `ABSOLUTE_POS_X + ABSOLUTE_POS_Y * (CUBE_COUNT_X * CUBE_DIM_X) + …` (measured: the two disagree for 912 of 960 threads at CubeCount(5,3,1) x CubeDim(8,8,1), and for 533 of 722 launch shapes swept). Its stride is a *runtime* product, so modeling it needs nonlinear arithmetic the prover does not use for global facts. Index with the per-axis builtins — `inp[(ABSOLUTE_POS_Y * w + ABSOLUTE_POS_X) as usize]` — or drop the `dispatch(...)` clause and use the flat 1-D form throughout (docs/design-2d-dispatch.md §2, §4.4)``

The same text, with the leading ident substituted, covers `CUBE_POS`
(*"…is the row-major flatten of the cube grid, `CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X + …`"*) and
`CUBE_COUNT` (*"…is `CUBE_COUNT_X * CUBE_COUNT_Y * CUBE_COUNT_Z`, a product of two runtime
values"*).

**R2 — a per-axis builtin with no `dispatch(...)` clause** (macro-authored, at the ident's span;
replaces today's generic out-of-subset message for these 12 idents, and is the direct analogue of
`COOP_CONSTRUCTS`'s targeted message at `lib.rs:775-784`):

> ``error: `ABSOLUTE_POS_X` is a per-axis topology builtin outside the ordinary vericl v0 subset; add a `dispatch(cube_dim = (Wx, Wy), extents = (w, h))` clause to `#[vericl::kernel(...)]` to opt this kernel into the multi-axis twin and the per-axis prover model (docs/design-2d-dispatch.md §4.2). The cube dimensions must be pinned literals: they are what keeps every position recomposition linear``

**R3 — `dispatch(...)` on a kernel that uses no per-axis builtin** (macro-authored, at the clause's
span):

> ``error: this kernel declares `dispatch(...)` but its body reads no per-axis topology builtin — a dispatch clause changes the launch shape, the reference twin's iteration space and the recorded evidence, so declaring one for a kernel that does not use it is a contract lie. Remove the clause, or index with `ABSOLUTE_POS_X`/`ABSOLUTE_POS_Y``

**R4 — `dispatch(...)` together with `cooperative(...)`** (macro-authored, at the second clause's
span):

> ``error: `dispatch(...)` and `cooperative(...)` are mutually exclusive in the vericl v1 subset — 2-D workgroups with shared memory are deferred. The intra-cube half is measured tractable (a two-thread tile-write obligation over per-axis unit ids discharges in under 10 ms), but the inter-cube half is not: proving `out[ABSOLUTE_POS_Y * w + ABSOLUTE_POS_X]` is written by exactly one thread across cubes needs a variable-by-variable product, and z3 times out on it at 180 s where the 1-D `out[ABSOLUTE_POS]` case is an O(1) pattern match. It needs a 2-D write-pattern recognizer, not a generalization (docs/design-2d-dispatch.md §8)``

**R5 — an invalid `cube_dim` tuple** (macro-authored, at the offending entry's span; three
sub-messages sharing a prefix):

> ``error: `dispatch(cube_dim = ...)` takes 2 or 3 positive integer *literals* — `Wx` is not a literal. The prover binds each entry as an SMT numeral and the reference twin uses it as a loop stride, so a runtime value would leave both undefined``
>
> ``error: `dispatch(cube_dim = (32, 64))` has 2048 units per cube, above the 1024 this milestone accepts. 1024 is the measured `max_units_per_cube` on the reference backend (wgpu 29 / Metal); the WebGPU default is 256, and a clause tuned above it will launch here and be rejected elsewhere. cubecl validates the same bound at launch (cubecl-runtime `validation.rs`), so this is the early, named form of a failure you would otherwise hit at run time``

**R6 — a wrapping product-assume spelling** (macro-authored, at the cast's span):

> ``error: `out.len() == (w * h) as usize` multiplies in u32 and then widens, so the executable `check_assumes` predicate tests the WRAPPED product while the prover would assert the mathematical one. Measured, those disagree: at `w = 2, h = 2147483649` the wrapped product is 2, so a length-2 buffer satisfies the clause while the model believes the length is 4294967298 — and an index of 2 then proves in bounds against a buffer that does not have it. Write `out.len() == (w as usize) * (h as usize)`: widening first makes the two agree at every input``

**R7 — `suite!`'s `cube_dim:` field alongside a `dispatch(...)` kernel** (macro-authored, at the
field's span):

> ``error: kernel `box_blur3x3` declares `dispatch(cube_dim = (16, 16))`, so the suite's `cube_dim:` field has nothing to set — two sources of truth for one launch parameter is how a proof gets bound to a block size the launch does not use. Remove `cube_dim:` from this suite, or remove the `dispatch(...)` clause from every kernel it lists``

**R8 — plane / cluster builtins** (unchanged; `BANNED_IDENTS` + `BANNED_PREFIXES`,
`lib.rs:45-65,146`). Recorded here only to state that `dispatch(...)` does **not** lift them:
`PLANE_DIM`, `PLANE_POS`, `UNIT_POS_PLANE`, `CUBE_CLUSTER_DIM*` and `CUBE_POS_CLUSTER*` stay
rejected, the last two additionally because cubecl 0.10 folds them to **different constants on
different backends** (`1` on WGSL, `0` on CUDA/SPIR-V — §1.1).

**R9 — the axis-convention note** (not an error; a required guide/doc addition, recorded as a
rejection-adjacent obligation because omitting it is the milestone's most likely *user* defect). The
guide's new 2-D section must state, at the point the clause is introduced:

> **X is the fastest-varying axis.** `ABSOLUTE_POS_X` moves along a row; a row-major image of width
> `w` is indexed `inp[(y * w + x) as usize]`. Writing `inp[(x * h + y) as usize]` is *in bounds* and
> *transposed* — VeriCL's proof will not catch it, because a transposed image is a functional bug,
> not a memory-safety one. The differential lane will.

**R10 — `CubeCount::Dynamic`** (macro-authored, wherever a future indirect-dispatch surface appears;
recorded now because the milestone is what makes it tempting):

> ``error: indirect dispatch (`CubeCount::Dynamic`) is outside the vericl v0 subset — an over-limit indirect count is silently rewritten to (0,0,0) by wgpu's validation shader, so the kernel does nothing while every recorded claim stays green. Compute the cube count on the host from the `dispatch(extents = ...)` parameters``

**R11 — an `if`-based clamp in a `dispatch(...)` kernel** (macro-authored, at the `if`'s span; a
*targeted upgrade* of the existing generic taint diagnostic rather than a new gate — the shape is
already unprovable, this just names it):

> ``error: a mutable local assigned inside an `if` cannot feed an index expression in the vericl v0 subset — a variable written in a branch arm is tainted after the arm closes (adversarial review round 2: not doing so was a confirmed false-`Proved` on a real out-of-bounds write). For a stencil clamp write it branch-free: `let x2 = u32::min(x + 1, w - 1);` and `let x0 = u32::max(x, 1) - 1;`. Both lower to arithmetic the prover models exactly``

**R12 — a `cube_struct!` field named in `extents`** (macro-authored, at the name's span):

> ``error: `dispatch(extents = ...)` names runtime `u32` *parameters* of this kernel; `p.width` is a struct field. The launch harness derives the cube count from the extents before it builds the struct's launch argument, so a field would have to be read twice from two places. Pass the extents as loose `u32` parameters``

### 10.4 Wording and gate corrections landing with v1

Three pre-existing statements this milestone makes wrong, or reachable, and must fix:

1. **`abs_pos_sym`'s unstated 1-D precondition** (`prover.rs:1633-1684`, module docs `:348-384`,
   `prove_bounds_freedom_cooperative`'s rustdoc `:673-694`). The recomposition
   `(CubePos*cube_dim + UnitPos) mod 2^32` is the true hardware value **only when the launch is 1-D
   in the cube** (`CUBE_DIM_Y == CUBE_DIM_Z == 1`, §2.2, 722/722). Today that holds because
   `coop.rs:1260` and `lib.rs:6244` both build `CubeDim::new_1d`, and *nothing says so*. All three
   sites gain the statement, and — in the round-9 F4 style, in the place a future implementer will
   look — a `REQUIRED WORK:` note at `abs_pos_sym` itself: *any 2-D cooperative milestone MUST
   replace this with the per-axis recompositions before enabling a multi-axis `CubeDim`.*
2. **The differential config does not record the launch shape** (`evidence.rs:177-184`). It records
   a *scalar* `cube_dim` and no cube count, so evidence cannot distinguish the shape it was produced
   under (§3.3). `differential_dispatch_config` records `cube_dim: [u32;3]` and `rank`; the 1-D
   `differential_config` gains `"rank": 1` so old and new evidence are comparable. This closes the
   recordable half of D1; the residual — that a user may launch a 1-D-authored kernel on a
   `≥ 2^32`-thread grid where `ABSOLUTE_POS` aliases — becomes an explicit line in
   `docs/guide.md` §12 ("What VeriCL does not do") and §13 risk 4.
3. **`proved_config`'s hardcoded `"logic": "QF_LIA"`** (`evidence.rs:212-218`). Nonlinear terms are
   already emitted (`checked_mul`'s variable×variable product), and `LenEqProduct` makes it routine.
   The field becomes the logic actually in force for that kernel.
4. **The extracted IR is built at a `cube_dim` the kernel is never launched with**
   (`lib.rs:4375`). `kernel_definition()` calls `builder.build(KernelSettings::default())`, and
   `KernelSettings::default()`'s cube dim is `CubeDim { x: 1, y: 1, z: 1 }` (measured, `ir_axes`) —
   so the `def.cube_dim` that `kernel_ir_hash` dutifully folds (`hash.rs:80`) is the **same constant
   for every kernel in the tree**, `cooperative(cube_dim = 256)` included. Today this is harmless:
   the IR *body* is byte-identical across cube dims (measured, §1.2), and a clause edit stales
   evidence anyway through `SOURCE_HASH`'s attribute tokens. But it is round 11's classification
   split in miniature — VeriCL extracts and hashes IR under settings the launch does not use — and a
   milestone whose whole contract *is* the launch shape should not leave it. M6 passes the pinned
   dims (or the suite's 1-D `cube_dim`) into `KernelSettings`, which makes `ir_hash` an independent
   tripwire on the dispatch shape rather than a constant.

### 10.5 Deferred (v1.1+, rejected with a pointer, not rejected forever)

| Deferral | Why | Measured basis |
|---|---|---|
| **2-D cooperative tiles** (`dispatch` × `cooperative`) | the intra-cube half is cheap; the inter-cube write-disjointness half needs a new *pattern* recognizer because SMT does not close it | `p4` unsat <0.01 s, `p4b` sat, **`p5` 180 s timeout** (§8) |
| **flat `ABSOLUTE_POS` in per-axis mode** | needs nonlinear global facts; and it is the identity that breaks | §2.2 (533/722), `p5` |
| **`Vector<P, W>` × `dispatch(...)`** | two `sizes_unit` conventions (lines vs extents) in one evidence config, undecided | round 8's units discipline; `evidence.rs:194-208` |
| **`cube_struct!` field as an extent** | needs the launch harness to read a field before building the struct arg | R12; the `gen(p.field)` grammar exists (`design-cubetype-args.md` §5.5) but is generation-side only |
| **runtime (non-literal) `cube_dim`** | the prover needs numerals to stay linear (§4.2) and the twin needs a loop stride | `p5` shows what happens when a coefficient goes runtime |
| **an `if`-based stencil clamp** | round-2 branch write-taint, a confirmed-critical fix nobody should relax for ergonomics | measured IR, §3.2; R11 gives the branch-free spelling |
| **`CubeCount::Dynamic`** | silent `(0,0,0)` no-op above the limit | `wgpu-core/src/indirect_validation/dispatch.rs:60-74` |
| **a `suite!` size ceiling check** (§3.4) | orthogonal to per-axis topology; owned by whoever touches `suite!`'s launch math next | `lib.rs:6242-6243`, measured 65535/axis |

---

## 11. Coverage projection — measured, not estimated

**Of the 464 surveyed ecosystem device items, v1 unlocks at most 1.** 39 items name 2-D / multi-axis
topology; the sole-blocker count is **1** (`docs/ecosystem-survey-2026-07.md:345`, reconfirmed after
the struct-comptime correction at `design-struct-comptime.md:801`). The other 38 each carry a co-gate
this milestone does not touch — the survey's own ranking puts 2-D at **rank 6** of the sole-blocker
table, below `plane_*` (2) and far below `View`/`Layout` (45).

**Of the 22 private dogfood kernels, v1 unlocks 0 on its own — and it is one of the two remaining
walls.** 2-D dispatch blocks 2/22 and is sole blocker for **0**
(`docs/dogfood-2026-07.md:285`). But the shim batch closed walls 1–4 and took faithful coverage from
6/22 to **19/22** (`tasks/todo.md:2466-2553`), so of the six recorded walls only **#5 (an
injectivity/permutation element assume)** and **#6 (2-D dispatch)** are left, and 2-D is named among
the three still-non-faithful items (`dogfood-2026-07.md:397`). Closing it does not by itself move
the 19; closing it *and* #5 finishes the corpus.

**What v1 does buy, stated as kernel classes rather than counts.** Measured against the §6 probe and
the §7 ledger:

| class | v1 | what carries it |
|---|---|---|
| 2-D / 3-D elementwise | differential **PASS** + **`Proved`** | `p2b` |
| transpose / permutation | differential **PASS** + **`Proved`** | §5, §6 |
| clamped stencil, blur, separable filters, Sobel/gradient | differential **PASS** + **`Proved`** | `p2c`, and only with `Min`/`Max` (§4.5) |
| stencil with a runtime-varying radius | differential PASS, `OutOfSubset` | the loop bound is a carried variable feeding an index |
| 2-D reduction / integral image | differential PASS, `OutOfSubset` | carried accumulators feeding indices; unchanged from 1-D |
| tiled matmul, 2-D shared-memory tiles | **rejected** (R4) | §8 |

**The honest framing.** This is a **capability-and-soundness milestone, not a coverage milestone**,
and the reviewer should hold it to that standard. Its case is: (a) it is the shape external users
most expect and most reliably hit — image-space kernels are the canonical "second GPU program"; (b)
it is the last unclaimed item on the recorded post-`Slice` frontier
(`tasks/todo.md:2302-2305`), the two items ahead of it having landed or been measured down; (c) it
closes one of the two remaining private walls; and (d) it surfaces D1 (§3.3) and three stale
statements (§10.4) that are wrong *today*, independent of whether the feature ships. The last two
milestones each discovered that their headline gate "does not exist; there is a hole instead"
(`tasks/todo.md:2943-2944, 3171-3173`); this one discovered that its headline gate is real, that
today's rejection is **correctly** fail-closed with no false `Proved` anywhere (§3.2), and that the
hole is one layer up, in what the evidence *records* rather than in what the prover *models*.

**What dominates after v1.** Unchanged: `View`/`Layout` (45 sole), trait-generic and
associated-type parameters (8), cmma (6). 2-D was rank 6 before this milestone and its removal does
not reorder anything above it.

---

## 12. Implementation plan (agent-sized milestones)

Each milestone leaves the tree green and the full example suite passing. The chain is strictly
ordered — M1 supplies the clause M2 reads, M2 supplies the prover mode M3's obligations need, M4
supplies the twin M6's differential exercises — with the single exception that M5 and M7 may be
swapped.

**M1 — the `dispatch(...)` clause + gates D1–D8, macro side only.**
`parse_contract` (`lib.rs:480-685`) gains the arm; `ContractSpec` gains the field; the biconditional
gate (D5) mirrors `cooperative(...)`'s (`lib.rs:3498`, `coop.rs:118-137`); `transform_body`
(`lib.rs:711-811`) gains a `dispatch: Option<u8>` parameter that un-bans the per-axis idents for the
enabled rank only, keeping the flat three banned (D8, R1). No twin, no prover, no launch yet — a
kernel with the clause compiles and its `contract()` reports the dims.
*Verify:* each of D1–D8 has a compile-fail test asserting the **exact** error string at the **exact**
span (a runtime `cube_dim` → R5a; `(32,64)` → R5b; `extents = (w)` with a 2-tuple `cube_dim` → D3;
`extents = (n)` where `n: usize` → D4; `dispatch` + `cooperative` → R4; `ABSOLUTE_POS` in the body →
R1; a 2-tuple clause with `ABSOLUTE_POS_Z` in the body → out-of-rank); a **negative control**
removing each gate must let its probe compile again; a kernel with no clause still gets **R2's new
targeted message** for `ABSOLUTE_POS_X` (not the generic one); and `cargo test --workspace` is green
with `evidence/vericl.json` **byte-unchanged**.

**M2 — the prover's per-axis mode.**
New `prove_bounds_freedom_dispatch` + `self.dispatch: Option<[u32;3]>`;
`abs_pos_axis_sym(axis)` modelled line-for-line on `abs_pos_sym` (`prover.rs:1653-1684`);
`predeclare_dispatch_leaves` modelled on `predeclare_coop_leaves` (`:1577-1592`), declaring five
leaves per enabled axis at the outermost scope; `builtin_value` (`:1690-1710`) gains the per-axis
arm. **Also fixes round 7's queued predeclaration hazard in the loop handlers** — a per-axis leaf
first resolved inside a grid-stride loop body is exactly the flow that makes it reachable
(`tasks/todo.md:1552-1655`).
*Verify:* `p1`'s exact repro is a permanent regression test per axis — the unwrapped form must
`Proved`, the shipped form must **`Refuted`** with a witness exhibiting `wrap_a = 1` and
`cube_a ≥ len` (the round-5 test `cooperative_abspos_guard_cubepos_index_refutes` cloned to X and
Y); a **negative control** deleting the `− wrap_a·2^32` term must flip those to `Proved`; a leaf
first resolved inside a `RangeLoop` body must still be in scope for an obligation after the loop
(the predeclaration test, which must **fail** with the fix reverted); and every existing prover test
is byte-identical.

**M3 — `Arithmetic::Min`/`Max` + `Assume::LenEqProduct` + the R6 cast gate.**
Two arms at `prover.rs:1931-1948` emitting `ite`; the sixth `Assume` variant and its assertion; the
recognizer in `lib.rs`'s assume classifier with cast peeling gated on `cast_is_value_preserving`
per operand type (round 4's rule, reused not reinvented); `check_assumes` emits the widened product;
`proved_config`'s logic field (§10.4 correction 3).
*Verify:* `p2a`–`p2f` and `p3` reproduced as prover-level tests with the same verdicts and the same
witnesses (`p2d`'s `w=1,h=1`; `p3`'s `w=2, h=2147483649`); R6's compile-fail asserts the exact text
at the cast's span; a **negative control** removing the cast gate must let `(w * h) as usize` be
recognized and must make an OOB kernel `Proved` — i.e. the gate is shown load-bearing, not
decorative; `Min`/`Max` on **floats** must still taint (the `is_modeled_int` guard, its own
predicate test); and `flatten_decode_scale` stays `Proved{2}`.

**M4 — the multi-axis twin.**
`coop.rs`'s structure is the model, but this is a *new*, simpler mode in `lib.rs`: the nested grid
loop of §4.7, the per-axis ident rewrites (including `UNIT_POS_a → __vericl_abs_a % Wa`,
`CUBE_POS_a → __vericl_abs_a / Wa`), `u32` loop variables, and the `grid` triple parameter replacing
`num_threads`.
*Verify:* the §6 kernels under `#[vericl::kernel]` reproduce the probe's **24/24 bit-exact** result
at the same six shapes; the loop nest is Z→Y→X and a test asserts the twin's write order equals the
flat `ABSOLUTE_POS` order for an aliasing-write kernel (a **negative control** swapping the nest to
X→Y→Z must fail it); the grid, not the image, is iterated (a test with `w=37, Wx=8` must execute 40
columns and 3 padding threads per row must take the `else`); and a helper called from a 2-D kernel
still cannot name any topology ident (`lib.rs:702-710` unchanged).

**M5 — the interpreter and the cross-check lane.**
`interp.rs:518-537` currently answers `AbsolutePosY => 0`, `CubeDimY => 1` — correct for 1-D, a
round-11-style classification split for 2-D. `ThreadCtx` (`:386-406`) gains the grid triple and
per-axis fields; `Inputs` gains the dims. A kernel the interpreter cannot place must return
`Unsupported`, never a guessed axis value.
*Verify:* an interpreter run of the §6 blur over a 2-D grid matches the twin exactly; a **negative
control** reverting to the pinned `0`/`1` answers must make that test fail (proving the test
discriminates); `run_corpus`'s 20 000-kernel fuzz cross-check is re-run and stays at **zero
findings**; and a 2-D kernel presented with 1-D `Inputs` reports `Unsupported`.

**M6 — launch, `sizes`, evidence, identity.**
`conformance_case`'s per-axis cube count and `CubeDim::new_2d/3d`; `generate_case` taking an extents
tuple; `suite!`'s tuple `sizes` and the R7 gate; `differential_dispatch_config`; and §10.4
correction 4 — `kernel_definition()`'s `KernelSettings::default()` (`lib.rs:4375`) gains
`.cube_dim(…)` from the clause, so the `def.cube_dim` that `hash.rs:80` already folds stops being
the constant `(1,1,1)`.
*Verify:* the identity improvement is an A/B — two kernels with **byte-identical bodies** differing
only in `dispatch(cube_dim = (16,16))` vs `(8,32)` must have **different** `kernel_ir_hash`es
(measured §1.2: their IR *bodies* are identical, so today they collide — the test must **fail**
before the fix and pass after, which is what makes it discriminate); a **cooperative** kernel's
`ir_hash` must likewise move on a `cube_dim` edit, closing the same gap for the shipped 1-D path;
every existing kernel's `identity().source_hash` must be **byte-identical** to before, while its
`ir_hash` legitimately moves once (a one-time, reviewed evidence re-record, called out as such);
`VERICL_UPDATE=1` produces exactly the new entries with every *other* field byte-identical under
per-entry canonical-JSON SHA-256, run `--features cpu` first then default last (the staleness-guard
lesson); and `vericl::obligation_count_changes` reports the new counts rather than absorbing them
(round 11's risk-8 warning).

**M7 — examples, guide, coverage, and the §10.4 corrections.**
Three example kernels in `crates/vericl-examples/src/lib.rs` wired into `tests/conformance.rs`
(elementwise-2D, transpose, box blur) plus one deliberately-defective 2-D kernel in the
`conform demo-defects` binary (an unclamped stencil, which must `Refuted` with the `p2d` witness
shape); the guide's 2-D section including **R9's axis-convention note**; `docs/coverage.md`'s 2-D
rows; and all three §10.4 corrections, each with a test.
*Verify:* the suite reports PASS + `Proved` for all three; the defect kernel is caught; correction 1
is asserted by a test that reads the doc text (the `REQUIRED WORK` marker must be present — the
round-9 F4 pattern); correction 2 by an evidence-shape test; correction 3 by a `Proved` claim whose
recorded logic reads `QF_NIA` for a `LenEqProduct` kernel and `QF_LIA` for `axpy`.

Ordering: M1 before M2 because the prover mode needs the pinned dims; M3 before M4 only in the sense
that M4's differential is worth more once M3's obligations discharge; M5 after M4 so the cross-check
compares against a twin that exists; M6 last of the plumbing because the identity fold should be
exercised by kernels that run.

---

## 13. Open risks, ranked (pre-registered for review round 12)

1. **The per-axis recomposition is now three copies of the round-5 encoding, and a fourth site
   (`abs_pos_sym`) still carries the 1-D one (high).** M2 adds `abs_pos_axis_sym` for X, Y and Z
   while `abs_pos_sym` stays for cooperative mode, so there are four constructions of the same
   equation and only the clause gate keeps them apart. Round 5's own lesson was *audit every
   construction site of the thing the invariant quantifies over*. **Attack surface**: reach
   `abs_pos_sym` from a `dispatch(...)` walk (or reach `abs_pos_axis_sym` with a stale
   `self.coop`), then guard on one flattening and index with the other — the exact 2-D analogue of
   round 5's `ABSOLUTE_POS`-guarded / `CUBE_POS`-indexed repro. Also worth probing: a `dispatch`
   walk where `self.dispatch` is `Some` **and** `self.coop` is `Some` through some path R4 does not
   cover. *Mitigation*: make the two modes structurally exclusive (`enum TopologyMode { Flat,
   Dispatch([u32;3]), Coop(u32) }` rather than two `Option`s), re-run the round-5 sibling hunt over
   every `smt.times`/`plus`/`sub` on an integer term, and add the per-axis regression tests M2
   specifies. **Currently the sharpest open question for review.**

2. **`cube_count_spread` can hand a user a multi-axis grid they did not ask for (high, inherited).**
   `cubecl-runtime-0.10.0/src/server/base.rs:1091-1117` halves X and doubles Y (then Y→Z) until
   each axis fits `max_cube_count`, and `CubeCountSelection::new` returns `Approx` with *more* cubes
   than requested. A user who sizes their dispatch through `cubecl_core::calculate_cube_count_
   elemwise` — the idiomatic cubecl helper — gets a 2-D grid above ~65535 cubes, and therefore §2's
   broken identity, in a kernel VeriCL certified as 1-D. **Attack surface**: a 1-D `Proved` kernel
   launched via the spread helper at a size that trips the split, then indexed with both
   `ABSOLUTE_POS` and `CUBE_POS`. *Mitigation*: the R1 wording and correction 1 tell the story, but
   the honest fix is that `differential_config` records the launch shape (correction 2) so at least
   the evidence names what it measured. A stronger option — a `debug_assert` in the generated launch
   glue — is **not** available because VeriCL does not own the user's launch call.

3. **Round 7's queued predeclaration hazard becomes reachable, and M2 is the only thing that closes
   it (high, inherited).** `tasks/todo.md:1552-1655` classified lazy leaf declaration inside the
   **loop** handlers as latent-LOW because no evidence-producing flow reached it. A grid-stride 2-D
   kernel resolving `ABSOLUTE_POS_Y` for the first time inside a `while` body is that flow, and the
   symptom is a `SolverError` on an undeclared symbol — or, worse, a scoped range fact silently
   dropped after the `pop`, which is the *soundness* half. **Attack surface**: a 2-D kernel whose
   only use of a per-axis builtin is inside a loop body, with an obligation after the loop.
   *Mitigation*: `predeclare_dispatch_leaves` declares all fifteen unconditionally, and M2's
   predeclaration test must **fail** with the fix reverted.

4. **The D1 residual: evidence still cannot constrain the user's launch (medium).** Correction 2
   makes the differential config *record* the shape it measured; nothing makes the user honour it,
   and `ABSOLUTE_POS` aliasing above `2^32` threads (§2.4/§3.3) remains uncovered by any claim.
   **Attack surface**: a `Proved{n}` 1-D kernel launched at `CubeCount(2048,2048,1) ×
   CubeDim(32,32,1)` with an `out[ABSOLUTE_POS] += …` body — a write-write race with every claim
   green. *Mitigation*: the guide §12 line, and the honest scoping that VeriCL's claims are about
   the kernel *as launched by the suite*. **The end-to-end witness was not dispatched** (a
   4.3-billion-thread launch risks a GPU watchdog reset on a display-connected device); the claim
   rests on two measurements composed (§2.4), and a reviewer with a headless device should run it.

5. **`LenEqProduct` puts a nonlinear fact in the *global* assertion context, where an `unknown` is
   costly (medium).** `checked_mul`'s products are probed in a `push`/`pop` and an `unknown` merely
   taints. A global `len = x·y` is in force for every obligation and for the **infeasible-assumption
   guard** (`prover.rs:336-347`), where an `unknown` is treated as "not provably contradictory" and
   allowed through. **Attack surface**: a kernel with two product assumes over overlapping scalars
   (`a.len() == w*h`, `b.len() == h*w`, `a.len() != b.len()`) that is genuinely contradictory but
   returns `unknown`, vacuously discharging everything — the round-1/round-7 infeasible-context trap
   with a nonlinear twist. *Mitigation*: measured worst case in the v1 ledger is 0.20 s and the
   feasibility probe is one extra `check-sat`, but the milestone must add a **degenerate-product
   feasibility test** in the `assert_element_bounds_feasible` family, and treat `unknown` on the
   *product* feasibility probe as `OutOfSubset` rather than allowed-through.

6. **`extents` names a parameter, and the twin, the launch and the prover each read it separately
   (medium).** D4 checks the name is a `u32` runtime parameter; nothing checks the three consumers
   agree on *which* one, and a kernel with two `u32` parameters of similar name is an easy
   transposition. **Attack surface**: `dispatch(extents = (h, w))` — the names swapped relative to
   the body's use — which launches a transposed grid, still indexes in bounds, and produces a
   wrong image. *Mitigation*: the differential lane catches it (measured: the §6 twin iterates the
   *grid*, so a swapped extent changes which threads are padding), and a test must assert exactly
   that; the deeper fix is that the twin's `grid` triple is derived from the same clause tokens the
   launch uses, which M4/M6 must share rather than re-derive.

7. **A 2-tuple clause silently makes the Z builtins out-of-rank, and "out of rank" is a new
   rejection category (medium).** `ABSOLUTE_POS_Z` under `cube_dim = (16,16)` is *always zero* on
   hardware, so accepting it would be harmless — and rejecting it is a strictly-narrower choice that
   users will hit. **Attack surface**: a 3-D-authored kernel whose clause was edited to 2-D, where
   accepting `_Z` would silently change the kernel's meaning from "a Z-strided walk" to "a constant
   0". *Mitigation*: reject (chosen), with wording that says the axis is not enabled by *this
   clause* rather than that the builtin is unsupported; a compile-fail test per axis.

8. **cubecl upgrade drift, and one upstream bug already in the blast radius (low, standing).** The
   flattening formulas, the `usize`/`u32` split, and the axis mapping are properties of
   `cubecl-{wgpu,cpp,spirv,cpu}-0.10.0`. **Attack surface**: a 0.11 that changes the flatten's
   coefficient order or widens `AddressType` by default. Separately,
   `cubecl-opt-0.10.0/src/analyses/integer_range.rs:162-165` asserts `CUBE_COUNT_a` is the constant
   `cube_dim.a` — wrong, currently dead (no consumer of `Ranges` exists in 0.10), and a real
   miscompile risk for `CUBE_COUNT_*`-reading kernels the moment it is wired into a
   bounds-check-elimination pass. *Mitigation*: the `identity_sweep` predicate (722/722) and the
   `axis_order` formula checks are cheap asserts that fail loudly if the mapping changes; add both
   to `docs/upgrade-drill-2026-07.md`'s checklist, and add a standing note that VeriCL's `Proved`
   claim is conditional on trusted codegen — which already covers the `integer_range` bug, but
   should name it while it exists.

9. **Coverage honesty: 1 sole-blocker is the weakest reach any milestone has shipped with (low).**
   §11 says so, but a green suite full of pretty image kernels is persuasive in a way the number is
   not. **Attack surface**: a README/guide edit that implies 2-D "unlocks image processing"
   generally, when the measured unlock is 1 ecosystem item and 0 additional private kernels.
   *Mitigation*: M7's guide text must carry the measured numbers, and `docs/coverage.md`'s new rows
   must distinguish `Proved` classes from `differential-only` ones (the stencil-with-runtime-radius
   and 2-D-reduction rows in §11 exist precisely so the table is not all green).

---

## 14. Roadmap impact

- **Clears the last unclaimed item on the recorded post-`Slice` frontier.** The ranking at
  `tasks/todo.md:2302-2305` was `plane_*` → `CubeType`-arg → 2-D → `Tensor`/`View`. `plane_*` was
  measured down to 2 sole-blockers and de-prioritised, `cube_struct!` landed in round 11, and this
  is the third. After it, the recorded frontier is empty and the *measured* one — `View`/`Layout` at
  45 sole-blockers — is unambiguously next.
- **Leaves one private wall standing.** With 2-D closed, the private corpus's only remaining wall is
  #5, the injectivity/permutation element assume (`tasks/todo.md:2340-2348`), which is a
  contract-vocabulary milestone rather than a subset one and is now the smallest remaining step to
  finishing that corpus.
- **Puts a second, harder milestone on the board with its cost already measured.** 2-D cooperative
  tiles (§8) is the tiled-matmul shape everyone eventually wants. This design measured its two
  halves separately and found the expensive one (`p5`, 180 s timeout) before anyone scoped it, so a
  future plan starts from "write a 2-D inter-cube write-pattern recognizer" rather than from "port
  the race walker".
- **Adds three prover capabilities that outlive the milestone.** `Arithmetic::Min`/`Max` as exact
  `ite` is useful anywhere a clamp appears; `LenEqProduct` is the first contract shape that
  discharges a variable×variable stride, which the `slice_mut` window case
  (`tasks/todo.md:1869-1873`) has wanted since the Slice milestone; and the per-axis exact modular
  recomposition is the template for any future derived-builtin modeling.
- **What this design does *not* need.** No new macro (the clause lives in `#[vericl::kernel]`), no
  new trait, no change to the counterexample validator (`eval_sexpr` already handles integer `*`),
  no change to `compare`, `gen`'s grammar, `uses(...)`, `instantiate(...)`, `config!`,
  `cube_struct!`, the `Vector` path, the `Slice` path, or the f64 lane. The prover delta is one
  entry point, one predeclaration function, one per-axis symbol builder, two arithmetic arms and one
  assume variant — which is what "a generalize-the-existing-machinery milestone" should look like
  when the generalization is honest about the one identity that does not survive it.

