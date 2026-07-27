# Runtime `CubeType` struct arguments — design (July 2026)

The implementable design for the re-ranked frontier's #1 shape: a `#[cube]` item taking a **runtime**
(non-`#[comptime]`) custom struct — `data: MyStruct` or `&MyStruct` where `MyStruct` derives
`CubeType`/`CubeLaunch`. `docs/ecosystem-survey-2026-07.md`'s post-correction addendum ranks it
**141 / 464** items, **28 sole-blocker, all 28 plain non-test `fn`s**, and
`docs/design-struct-comptime.md` §4.4/§11 names it the next milestone.

Two deliverables:

- **A. The census correction.** The 28 is **20** — the classifier double-counts the struct-comptime
  params it had already demoted — and **8 of the 28 are gate-free on today's unmodified VeriCL**
  (one of them ported end to end here, §3.1). Of the corrected 20, a v1 unlocks **zero**.
- **B. The soundness milestone.** A runtime `CubeType` struct is **silently accepted today** on the
  helper path, with the struct's definition in **no hash** — the `docs/design-struct-comptime.md`
  §5.1 hole reproduced verbatim, live at `e5589f3` (§4.1). This design closes it.

Everything marked *measured* was checked empirically against the pinned `cubecl =0.10.0` (z3 4.16.0
on PATH, wgpu 29 / Metal on an Apple M3), the same posture as
[design-line-vector.md](design-line-vector.md), [design-shared-memory.md](design-shared-memory.md)
and [design-struct-comptime.md](design-struct-comptime.md). Probe sources are preserved in the
scratchpad (`scratchpad/cubetype/probe/src/bin/{equiv,arrays}.rs`,
`scratchpad/cubetype/vcl/src/bin/{ir,irarr,reject,rettuple,dlocal,arraynew,gt,site21}.rs` and
`vcl/src/main.rs`, plus the classifier re-runs `{dump28,refine28,structs,fieldcensus,recount,launchfns}.py`);
the consolidated run is `scratchpad/cubetype/RESULTS.txt`. Reference shapes are **clean-room /
upstream-public only** (cubecl 0.10.0, cubek — MIT/Apache-2.0), per the README policy.

File:line citations to `crates/vericl-macros/src/{lib,config,suite,coop}.rs`,
`crates/vericl/src/{contract,evidence}.rs`, `crates/vericl-ir/src/prover.rs` and the
`cubecl-{core,macros}-0.10.0` trees are current as of `e5589f3`.

---

## 0. Headline recommendation

1. **API-reality correction 1 — the 28 is 20, and 8 of the 28 need nothing.** The re-census's broad
   `CubeType`-param check strips `#[...]` attributes before typing a parameter
   (`classify.py:441`), so every `#[comptime] cfg: SomeConfig` it had just *demoted* out of
   "struct-typed `#[comptime]` param" is re-flagged under "custom `CubeType` param (broad)". Eight of
   the 28 have no other offending parameter. Re-running the classifier with that one rule fixed and
   everything else byte-identical: gate-free items **89 → 97**, gate-free non-test `fn`s
   **12 → 20**, and the gate's sole-blocker count **28 → 20** (§3.1). The plain-function margin over
   the next bucket collapses from **9×** to **1.8×** (cmma 11, `plane_*` 9). One of the eight,
   cubek-std's `interleaved_load_zeros`, is ported clean-room here and passes the wgpu/Metal
   differential with **0 / 32** bit-differences on today's unmodified VeriCL.

2. **API-reality correction 2 — there is no struct-of-buffers in the ecosystem.**
   `docs/design-struct-comptime.md:331-334` scopes this milestone as needing "a twin representation
   for a struct-of-buffers". Measured over the six survey crates: **165** `CubeType`/`CubeLaunch`
   struct definitions, **25** of which derive `CubeLaunch` (the only ones eligible as a runtime
   *launch* argument), and **zero** of those 25 has an `Array` or `Tensor` field. Twenty definitions
   *do* carry `Array`/`Slice`/`SharedMemory` fields — and all twenty are `CubeType`-only device
   aggregates (`RowWise`, `RegisterTile`, `StridedTile`, …) built inside a kernel, never launched
   (§3.3). The struct-of-buffers is *expressible* and works (X6–X8), it is simply **unused**.

3. **The lowering model, measured three independent ways: a runtime struct parameter is exactly a
   positional flattening of its fields.** On Metal, a struct-of-scalars kernel and the same kernel
   with the fields spelled as loose parameters are **bit-exact equal** (X1), and so are the
   array-carrying versions (X6). The `KernelDefinition`s agree buffer-for-buffer and scalar-for-scalar
   (I2–I3). And `kernel_ir_hash` is **byte-identical** — `sha256:e0312c05…` for
   `Bundle { a: Array<f32>, k: u32, b: Array<u32>, eps: f32 }` sitting between two plain array
   parameters and for its six-parameter flattened spelling (I3). **The prover needs zero changes**,
   for the same reason as struct comptime but by a different route: there the struct never reaches
   the IR, here it is dissolved before the IR exists (§2, §7).

4. **There is a live soundness hole, and it is the milestone.** Three parameter positions, three
   different behaviours on today's VeriCL: `p: &MyStruct` is cleanly rejected (V1); `p: MyStruct` on a
   **kernel** is rejected by the wrong gate with a message blaming `gen(...)` (V2); `p: MyStruct` on a
   **helper** is accepted with **no diagnostic at all** (V3). The accepted path has no identity
   coverage: with a `#[cube] impl Pair { fn fold(&self) -> u32 }` edited from `self.a * self.b` to
   `self.a + self.b`, the twin goes from `[3, 6, 9, 12]` to `[4, 5, 6, 7]` while the kernel's
   `SOURCE_HASH`, the helper's `SOURCE_HASH` **and** `identity().source_hash` all stay bit-identical
   (V4, §4.1). On the launch side there is a second, sharper hazard: `XLaunch::new` is **positional**,
   so swapping two same-typed fields in the definition changes the computed function with the kernel
   body *and the launch-call text* unchanged (X2).

5. **Decided design: `vericl::cube_struct! { … }` + a `StructIdentity` fold + one new `ParamKind`,
   and the twin is the user's own struct.** The declaration macro is the `vericl::config!` precedent
   moved one position over (§5, §6). Because the macro must see the declaration anyway to hash it, it
   *also* knows the field list — which is what makes the flattening desugar possible at all, since a
   token-only macro otherwise cannot know what `p` contains. And because a `#[derive(CubeType)]`
   struct of scalars is still an ordinary Rust struct, the twin needs **no generated mirror type**:
   it binds the author's own struct and the body tokens are unchanged (V6, §5.4). Array fields are
   **deferred**, with their foundation measured rather than guessed (§10.5).

6. **Honest reach: v1 unlocks 0 of the corrected 20, and that is measured site by site.** Every one
   carries a co-gate this feature does not touch: `Sequence` (3), device aggregates with
   `Slice`/`SharedMemory`/`View` fields (6), trait-generic or associated-type parameters (8), a
   runtime enum reached through a `#[cube] impl` method (2), cmma (1) — §11. Nineteen of the twenty
   are `#[cube]` **helpers**, not launch entry points, so their struct arguments are device-local
   values, not launch arguments. **This is a soundness milestone, exactly like struct comptime**: the
   corpus pays for it with the 8-site census correction and with a hole that is closed rather than
   documented.

---

## 1. API reality — the `#[derive(CubeType)]` / `#[derive(CubeLaunch)]` mechanism catalog

Catalogued from `cubecl-macros-0.10.0/src/generate/cube_type/generate_struct.rs` (401 lines), the
sole struct generator; enums go through `generate_enum.rs` (897 L, comptime enums) and
`generate_runtime_enum.rs` (533 L, payload-carrying runtime enums). Every claim below is either a
source citation or a probe.

### 1.1 What each derive emits

`generate(with_launch)` (`generate_struct.rs:12-35`) is a clean either/or — the two derives share no
generated item:

| Derive | Generated items | Cite |
|---|---|---|
| `CubeType` | `<Name>Expand` companion struct (one field per field, `<T as CubeType>::ExpandType`, or the plain type for a comptime field); `Clone` + `__expand_clone_method`; `impl CubeType { type ExpandType = <Name>Expand }`; `impl IntoMut`/`CubeDebug` | `:38-49`, `:51-75`, `:137-149`, `:292-321`, `:338-348` |
| `CubeLaunch` | `<Name>Launch<..generics.., R: Runtime>`; its positional `fn new(...)`; `<Name>CompilationArg` with hand-written `Clone`/`Hash`/`PartialEq`/`Debug`; `impl LaunchArg` with `register`/`expand`/`expand_output` | `:77-90`, `:92-114`, `:158-219`, `:221-290` |

Two corrections to `docs/design-struct-comptime.md:319-324`, which described the same machinery from a
reading rather than a probe. **The branch is exclusive, not additive** (`generate_struct.rs:12-35`):
`CubeLaunch` emits *only* the launch items and no `CubeType` impl, which is why the corpus always
writes both derives together and why this design's macro emits the pair (§5.2). And
`<Name>Launch` **carries no lifetime** — `parse/cube_type/parse_struct.rs:55-60`; `ArrayArg<R>` and
`TensorArg<R>` are lifetime-free too (`array/launch.rs:26`, `:125`), so `XLaunch<'_, R>` does not
name anything. Two further names in that paragraph's neighbourhood do not exist at 0.10.0 at all:
`LaunchArgExpand` (`expand`/`expand_output` are methods on `LaunchArg` itself,
`cubecl-core-0.10.0/src/frontend/element/base.rs:199-223`) and `ArgSettings` (only
`ScalarArgSettings` survives).

The load-bearing body is `launch_arg_impl` (`:221-290`): `expand` and `expand_output` build the
`<Name>Expand` value by calling `<FieldTy as LaunchArg>::expand(&arg.field, builder)` **once per
field, in declaration order**, on the *same* `KernelBuilder`. That is the source-level statement of
the flattening §2 measures. A `#[cube(comptime)]` field is not expanded at all — it is
`arg.field.clone()`, lifted straight out of the `CompilationArg` (`:227-236`).

`launch_arg_where` (`:323-336`) requires `LaunchArg` of **every non-comptime field type**. So:

- **`CubeType` alone is not enough for a runtime kernel parameter.** It generates no `LaunchArg`
  impl, and cubecl's blanket scalar impl is `impl<T: ScalarArgSettings> LaunchArg for T`
  (`cubecl-core-0.10.0/src/frontend/element/numeric.rs:129`), which a struct does not satisfy.
  Worse, that blanket impl **masks cubecl's own guidance**: `LaunchArg` carries
  `#[diagnostic::on_unimplemented(note = "Consider using `#[derive(CubeLaunch)]` on `{Self}`")]`
  (`base.rs:198`), and rustc never reaches it — the reported error is the scalar chain, i.e.
  "``OnlyType: ScalarArgSettings`` is not satisfied … the trait `CubeElement` is not implemented".
  A VeriCL user who bypassed R1 would get that message, which is a reason for R1 to fire at macro
  time rather than to rely on rustc.
- **`CubeType` alone *is* enough for a device-local value** — a struct built inside a kernel and
  passed to a helper, which is what 19 of the corrected 20 sites do (§3.2).
- **A generic struct with a bare generic scalar field does not compile under the natural bound.**
  `compilation_ty` (`generate_struct.rs:158-163`) is generated with the struct's plain generics and
  **without** `launch_arg_where`, so `struct P<F: Float> { data: Array<F>, scale: F }` needs
  `F: Float + CubeElement` and produces 30 errors otherwise. cubecl's own `CubeLaunch` structs all
  sidestep it by being non-generic or by marking the generic field `#[cube(comptime)]`. CS3 (§5.3)
  rejects generic declared structs for an identity reason; this is a second, independent one.

### 1.2 The `#[cube(comptime)]` field attribute — pervasive, and the design's sharpest asymmetry

A field may be marked `#[cube(comptime)]`, making it a host-side constant of the *runtime* type.
`launch_field`/`launch_new_arg`/`compilation_arg_field` (`:355-391`) keep the plain type for such a
field; `expand_type_impl` (`:292-321`) keeps it plain in the expand companion too.

Measured over the six survey crates: **88 of 165** struct definitions (53%) carry at least one
comptime field, and **169 of 396** fields (43%) are comptime. `Swizzle` and `NoopLayout` derive
`CubeLaunch` with **every** field comptime; `ReduceRequirements` — the parameter type of corrected
site 15 — is a `CubeType` struct with exactly one field, `#[cube(comptime)] coordinates: bool`, read
in the body as `requirements.coordinates.comptime()`.

Two consequences, both measured (X3):

- a comptime field **keeps its positional slot** in `XLaunch::new` but takes the plain host value:
  `HalfComptimeLaunch::new(scale: f32, bias: u32)`;
- a comptime field's type must be `Hash + Eq` — `XCompilationArg`'s hand-written `Hash`
  (`:196-209`) hashes every comptime field, so an `f32` comptime field **fails to compile**. This is
  why every comptime field in the corpus is a `u32`, a `bool` or a unit enum.

### 1.3 Launch-side construction is positional

`launch_new` (`:92-114`) emits `fn new(#(#args),*) -> Self` in declaration order, with runtime fields
typed `<Ty as LaunchArg>::RuntimeArg<R>` and comptime fields typed plainly. In cubecl 0.10 a scalar's
`RuntimeArg` is the scalar itself (`numeric.rs:130`), so the corpus writes
`UniformLaunch::new(self.lower_bound, self.upper_bound)` (`cubek-random/src/uniform.rs:73-75`) —
two bare `f32`s, positionally. Nested `CubeLaunch` structs compose the same way (X4), and an array
field takes `ArrayArg::from_raw_parts(handle, len)` (X6).

**This positional constructor is the identity hazard of §4.3.** Nothing about `new(a, b)` names the
fields it fills.

### 1.4 `Sequence<T>` and `FastDivmod`, since the corpus leads with them

`Sequence<T>` (`cubecl-core-0.10.0/src/frontend/container/sequence/base.rs:19-22`) is a
**comptime-length** container: its expand type is `Rc<RefCell<Vec<T::ExpandType>>>` (`:146-150`) and
indexing calls `index.constant().expect("Sequence index must be constant")` (`:124-133`). Its
`LaunchArg` (`sequence/launch.rs:67-104`) maps element-wise, so the length is baked into the
`CompilationArg` and hence into the `KernelId` — **a different length is a different kernel**.
Measured: a 3-element `Sequence<Array<f32>>` parameter produces three separate `Read` buffers, and it
flattens identically when nested in a struct.

`FastDivmod` is a `CubeType` **enum** (`cubecl-std/src/fast_math.rs:11-21`,
`Fast { divisor, multiplier, shift_right }` / `Fallback { divisor }`) with a **hand-written**
`LaunchArg` in a private module (`:101-169`) whose `RuntimeArg<R> = I` — the caller passes only the
divisor. It contributes **no buffer** and either **three** `u32` scalar slots (fast path) or
**one** (fallback), chosen from *device* u64 support at register time (`:120-122`). Measured on
Metal: `scalar U32 count=3`. That a declared field's scalar footprint can be device-dependent is a
direct argument for CS2's closed field-type list (§5.3) — a design that assumed one slot per field
would be wrong on some devices and right on others.

Both are out of v1 (§10.3, §10.5); they are named here because together they account for 11 of the
25 `CubeLaunch` definitions' runtime fields and 3 of the corrected 20 sites.

---

## 2. Lowering reality — flattening, measured (validated)

### 2.1 On the GPU (X1, X6, GT)

`y[i] = x[i] * p.scale + p.bias` with `Affine { scale: f32, bias: f32 }`, against the same body with
two loose `f32` parameters, `n = 8` on Metal:

```text
P1 struct  = c0800000 c0180000 bf400000 3f600000 40200000 40840000 40b80000 40ec0000
P1 flat    = c0800000 c0180000 bf400000 3f600000 40200000 40840000 40b80000 40ec0000
P1 VERDICT : BIT-EXACT EQUAL
```

With array fields (`Bundle { a: Array<f32>, k: u32, b: Array<u32>, eps: f32 }` between a leading
plain array and a trailing output), also bit-exact. And on the real ecosystem shape (GT, §8),
**0 / 4096** bit-differences.

### 2.2 In the `KernelDefinition` (I2, I3)

Built client-free with the macro's own recipe (`lib.rs:3814-3827`):

```text
--- struct in the middle: 4 buffers        --- flattened: 4 buffers
      [0] id:0 Read      F32   (lead)            [0] id:0 Read      F32
      [1] id:1 Read      F32   (s.a)             [1] id:1 Read      F32
      [2] id:2 Read      U32   (s.b)             [2] id:2 Read      U32
      [3] id:3 ReadWrite F32   (out)             [3] id:3 ReadWrite F32
      scalars: ["F32x1", "U32x1"]                scalars: ["F32x1", "U32x1"]
```

Buffer ids come from `KernelBuilder::buffer_id` = `buffers.len() + tensor_maps.len()`
(`cubecl-core-0.10.0/src/compute/builder.rs:38-40`), assigned in `expand` call order — inputs and
outputs share **one** id space — so a struct's buffers land **in place**, at the struct's own
parameter slot, in field declaration order.

Scalars follow a different, three-part rule, and the design depends on knowing it precisely:
they are **grouped by `StorageType`** via a per-type counter (`builder.rs:31-36`); the groups are
**sorted by `StorageType`** in the `KernelDefinition` (`codegen/integrator.rs:113`) and in the packed
info buffer (`codegen/scalars.rs:30-44`); and **within** a group the index is `builder.scalar()` call
order, i.e. declaration order. Group order is therefore independent of declaration order —
`{ i: u32, a: f32, j: u32, b: f32 }` still emits the `f32` group first. A struct's scalar fields and
the kernel's loose scalar parameters share the same counters. This is why the design never lets
`BUFFER_PARAMS`-style positional custody reach the scalar side: on the buffer side position is
meaning, on the scalar side it is not.

### 2.3 In `kernel_ir_hash` (I1, I3)

```text
ir_hash flat            = sha256:58a9d55c02c77517f3485b63bb3b1f568b43fd36df3826121c34934bc6e2b83a
ir_hash struct (middle) = sha256:58a9d55c02c77517f3485b63bb3b1f568b43fd36df3826121c34934bc6e2b83a
ir_hash struct (first)  = sha256:58a9d55c02c77517f3485b63bb3b1f568b43fd36df3826121c34934bc6e2b83a

ir_hash struct  (with Array fields) = sha256:e0312c052f16a5f258770a16f90c29423ceb2bd52cd8f161ea7c495bd9e034d7
ir_hash flat    (with Array fields) = sha256:e0312c052f16a5f258770a16f90c29423ceb2bd52cd8f161ea7c495bd9e034d7
```

**The IR cannot tell a struct parameter from its flattened spelling** — including when the struct
sits first rather than in the middle, and including when it carries buffers. This is the §7 argument
for zero prover changes, and it is a measurement, not an inference.

### 2.4 Counterexample hunting — what was tried against the flattening claim

Pre-done, because a review round would do it:

| Attack | Result |
|---|---|
| struct with **zero** array fields (scalars only) | X1 bit-exact |
| struct with array fields **interleaved** with plain array parameters | X6 bit-exact, I3 identical IR |
| struct parameter **first** rather than in the middle | I1 identical IR |
| **nested** `CubeLaunch` struct | X4 works, composes positionally |
| **two** struct parameters of the same type | X5 works |
| struct taken **by value** rather than by reference | X5 works |
| `&mut Struct` whose field is the output buffer | X7 works — **every** field routes through `expand_output` (`generate/kernel.rs:223-229`), so all array fields become `ReadWrite`; scalars are unaffected |
| an array field the body **never reads** | X8 still bound — and the shift is *observable*: an unread field consumes a binding index and moves every later buffer's id down one. Expansion is driven by the `CompilationArg`, never by body usage |
| two struct parameters of the **same** type | no deduplication — five distinct buffers for `(&Arrays, &Arrays, &mut Array)` |
| struct field read inside a `#[vericl::helper]`-shaped `#[cube]` helper | no effect on the layout; bindings are created entirely at the top-level parameter list, and `&`/`&mut` on a helper parameter is stripped (`parse/kernel.rs:767-786`) |
| scalar field **access order** ≠ declaration order | index follows declaration order; the emitted WGSL reads `scalars_u32[1]` before `[0]` when the body does |
| a `#[cube(comptime)]` field mixed with runtime fields | X3 works; slot kept, plain type, `Hash` required |
| an `f32` `#[cube(comptime)]` field | **does not compile** — `XCompilationArg: Hash` |

The one place flattening is *not* faithful is the launch **call**, and that is §4.3.

---

## 3. What the 28 actually are — and the three corrections that forces

### 3.1 Correction 1: the gate double-counts the params it already demoted

`broad_cubetype_params` (survey `classify.py:407-453`) computes
`q = re.sub(r'#\[[^\]]*\]', '', p)` and then types `q`, so a `#[comptime] scheme: QuantScheme` is
indistinguishable from a runtime `scheme: QuantScheme`. The struct-comptime addendum demoted the
`struct-typed #[comptime] param` row to supported; this row silently re-adds the same parameters.

Eight of the 28 have **no** other offending parameter:

```text
cubecl-std/quant/dequantize.rs:78          cast_masked             #[comptime] QuantScheme
cubek-reduce/components/global/base.rs:22  reduce_count            #[comptime] VectorizationMode, VectorSize
cubek-reduce/components/readers/base.rs:86 fill_coordinate_vector  #[comptime] VectorizationMode
cubek-std/stage/stage_memory/swizzle.rs:40 as_swizzle_object       #[comptime] SwizzleMode
cubek-std/tile/compute/matmul/interleaved.rs:107 interleaved_load_zeros  #[comptime] InterleavedMatmul
cubek-std/tile/compute/matmul/register.rs:10     register_execute        #[comptime] RegisterMatmul
cubek-std/tile/compute/matmul/register.rs:185    register_load_zeros     #[comptime] RegisterMatmul, StageIdent
cubek-std/tile/data/interleaved.rs:93            interleaved_allocate_acc #[comptime] InterleavedMatmul, MatrixLayout
```

Re-running the classifier with that one rule fixed (skip `#[comptime]`-attributed parameters) and
everything else byte-identical:

| | with the double count | corrected |
|---|---:|---:|
| items tripping zero blocking gates | 89 | **97** |
| …of which plain non-test `fn` | 12 | **20** |
| items with exactly one blocking gate | 158 | **174** |

and the frontier re-ranks a second time:

| gate | sole | sole non-test `fn` |
|---|---:|---:|
| View/Layout machinery | 57 | 0 |
| `plane_*` | 21 | **9** |
| **custom `CubeType` param (broad)** | **20** | **20** |
| `comptime_type!` | 18 | 0 |
| cmma / `Matrix` | 17 | **11** |
| `CubeType`-arg (v0 name list) | 12 | 5 |
| `comptime!{}` out of subset | 10 | 4 |

It is still #1 for plain functions, but by **1.8×**, not the recorded 9×.

**Correction 1 is closed end to end, not asserted.** `interleaved_load_zeros` — one of the eight,
`config.tile_size.m() * config.tile_size.n()` driving the body — was ported clean-room with
`vericl::config!` and run on today's unmodified VeriCL:

```text
twin[0..4] = [6.0, 6.0, 6.0, 6.0]   gpu[0..4] = [6.0, 6.0, 6.0, 6.0]
bit-differences = 0 / 32     VERDICT: PASS — gate-free today, no new feature
identity sha256:69ad24c65e1b0e467c767ad23abb5265ae170b70109f3a11b3e5ee4666aff58a
ir_hash  sha256:d9cd1f1c97e86db37803d56bdbab1290f741368ddb425503781a452637e9d158
```

**Correction 1b — the classifier never checks return types.** Two of the 20 now-gate-free `fn`s
return a `CubeType` struct (`-> Swizzle`, `-> Tile<A, Sc, ReadWrite>`), so the honest gate-free count
is **18**. Of the 20 sole-blocked, three return tuples and two return `CubeType` containers. VeriCL
does not gate helper return types either (V5: `-> (u32, u32)` and `-> Pair` both compile today), so
this is a hole on both sides of the ledger, and §10.4 closes VeriCL's half.

### 3.2 Correction 2: these are two different mechanisms, and the corpus uses the *other* one

| | **launch struct** | **device-local aggregate** |
|---|---|---|
| derive | `CubeType + CubeLaunch` | `CubeType` only |
| where it comes from | a launch argument, built on the host | a struct literal inside a kernel |
| cubecl machinery | `XLaunch`, `XCompilationArg`, `impl LaunchArg` | `XExpand` and nothing else |
| VeriCL today | rejected (V1) or misdiagnosed (V2) | **silently accepted** (V3, V6) |
| corpus definitions | 25 | 140 |
| of the corrected 20 sites | 1 (`reduce_kernel`, and by associated type) | 19 |

Nineteen of the twenty corrected sites are `#[cube]` **helpers**. Their struct arguments are values a
caller built on the device — `StridedTile`, `GlobalIterator`, `StridedStageMemory`, `Specializer`,
`PartitionScheduler`, `ReduceRequirements`, the `*Job` structs. Only `CubeMapping` and `RuntimeArgs`
among the named types derive `CubeLaunch` at all.

That matters twice. It means the corpus reach of the *launch* half is near zero (§11). And it means
the **already-accepted** half is the half the corpus uses — measured working, and measured unhashed
(§4.1).

### 3.3 Correction 3: no struct-of-buffers exists — but it is expressible

Field census over the six survey crates:

```text
CubeType/CubeLaunch struct definitions            165   (+19 enums)
  derive CubeLaunch (eligible as a launch arg)     25
  derive CubeType only (device-internal)          140
fields, all definitions                           396   of which #[cube(comptime)]  169 (43%)
definitions with >= 1 comptime field               88 / 165
definitions with a field of another CubeType       38 / 165   (nested)
definitions with an Array/Slice/SharedMemory field 20 — every one CubeType-ONLY
CubeLaunch definitions with an Array/Tensor field   0
```

The 25 `CubeLaunch` definitions by runtime-field composition: **11** carry a
`View`/`Sequence`/`VirtualLayout` field, **7** carry a nested struct or `FastDivmod`, **5** have
runtime fields that are **all scalars** (`Uniform`, `Normal`, `Bernoulli`, `SimpleLayout`,
`TestPerTensorScaleLayout`), and **2** are entirely comptime. Runtime-field type histogram:
`u32` 16, `View` 7, `Sequence` 7, `f32` 5, `usize` 4, `VirtualLayout` 4, `ComptimeOption` 4,
`FastDivmod` 4, then singletons.

So the ecosystem's runtime launch struct is a **parameter block of scalars**, or a *layout* object
whose contents are `View`/`Sequence` — never a bundle of buffers. `docs/design-struct-comptime.md`'s
"twin representation for a struct-of-buffers" names a shape the corpus does not contain. It is
nonetheless entirely expressible (X6–X8: it compiles, it binds one buffer per array field, `&mut`
gives `ReadWrite`, an unread field is still bound), which is why §10.5 defers it *with* a measured
foundation instead of leaving it open.

### 3.4 Co-occurrence — what the compatibility matrix must cover

Of the 101 fn-parsable items with a runtime non-scalar parameter vs the 71 without:

| feature | with (n=101) | without (n=71) |
|---|---:|---:|
| generic type params | **94 (93%)** | 48 (68%) |
| `Line`/`Vector` | **46 (46%)** | 17 (24%) |
| View/Layout machinery | 44 (44%) | 27 (38%) |
| struct `#[comptime]` param | 49 (49%) | 43 (61%) |
| `match` | 28 (28%) | 14 (20%) |
| `comptime!` block | 25 (25%) | 8 (11%) |
| cmma / `Matrix` | 18 (18%) | 8 (11%) |
| `plane_*` / `sync_*` | 14 (14%) | 7 (10%) |
| `Sequence` | 5 (5%) | **0 (0%)** |
| `SharedMemory` | 2 (2%) | 2 (3%) |

Generics (93%) and `Vector` (46%) must co-work. Runtime non-scalar parameters per item: 1 → 37,
2 → 31, 3 → 19, 4 → 9, and a tail to 10.

### 3.5 The corrected 20, site by site

| # | site | runtime struct shape | co-gate that still blocks |
|---|---|---|---|
| 1 | `cubecl-std` contiguous/base.rs:70 | `Sequence<FastDivmod>`, `Sequence<usize>` | `Sequence` |
| 2 | `cubek-conv` spatial.rs:214 | `Sequence<From>` | `Sequence` + `Sequence` return |
| 3 | `cubek-conv` tma_im2col.rs:136 | `Sequence<FastDivmod<u32>>` | `Sequence` + tuple-of-`Sequence` return |
| 4 | `cubek-conv` async_full_cyclic.rs:181 | Job / `GlobalIterator` / `StridedStageMemory` / `RuntimeArgs` | `SharedMemory` + `View` fields |
| 5 | `cubek-matmul` partition/matmul.rs:145 | `Args::State`, `(u32, u32)` | associated type |
| 6 | `cubek-matmul` double_buffer:17 | `LJ`/`RJ`/`S::Barrier`/`Specializer` | trait-generic + associated type |
| 7 | `cubek-matmul` double_buffer:60 | 10 params, `SMM::*` | trait-generic + associated type |
| 8 | `cubek-matmul` double_buffer:151 | 9 params, `SMM::*` | trait-generic + associated type |
| 9–10 | `cubek-matmul` async_{full,partial}_cyclic | Job / `GlobalIterator` / stage memory | `SharedMemory` + `View` fields |
| 11–12 | `cubek-matmul` cube_mapping.rs:10,16 | `&CubeMapping` | runtime **enum** field + `#[cube] impl` method + tuple return |
| 13 | `cubek-reduce` instructions/base.rs:442 | `&R`, `R::SharedAccumulator` | trait-generic + `SharedMemory` |
| 14 | `cubek-reduce` instructions/mean.rs:18 | `&SI` | trait-generic |
| 15 | `cubek-reduce` readers/base.rs:65 | `ReduceRequirements` (one comptime field) | **return** `Value<Vector<u32, N>>` |
| 16 | `cubek-reduce` launch/base.rs:115 | `RA::Input` / `RA::Output` | associated type |
| 17 | `cubek-std` mma/writer.rs:193 | `MmaDefinition<A, B, CD>` | cmma |
| 18–20 | `cubek-std` register.rs:76,137,161 | `&StridedTile` (a `Slice` field) | `Slice`/`View` field |

Site 15 is the closest miss: its parameter type is admissible under v1 (a struct whose only field is
`#[cube(comptime)] coordinates: bool`), and only the return type blocks it.

---

## 4. The defects (measured)

### 4.1 The identity hole — the milestone's centre of gravity

`SOURCE_HASH = sha256(fn tokens ‖ "||contract:" ‖ attr tokens ‖ "||vericl:" ‖ version)`
(`lib.rs:3232-3239`). A struct type's **definition** is in none of those inputs — exactly the
`docs/design-struct-comptime.md` §5.1 shape, one position over.

`scratchpad/cubetype/vcl/src/main.rs`: a kernel builds `Pair { a: x[i], b: 3 }` and calls a
`#[vericl::helper] fn use_pair(p: Pair) -> u32 { p.fold() }`, with
`#[cube] impl Pair { fn fold(&self) -> u32 { … } }`. Built twice, changing only the method body:

```text
                       twin reference([1,2,3,4])   kernel SOURCE_HASH   helper SOURCE_HASH   identity().source_hash
self.a * self.b        [3, 6, 9, 12]               ac9b4005…            4d819c5e…            f0096061…
self.a + self.b        [4, 5, 6, 7]                ac9b4005…            4d819c5e…            f0096061…
```

A different computed function under a bit-identical recorded identity, on today's tree, with **no
gate anywhere in the path**. Evidence recorded against the first build verifies FRESH against the
second.

**Negative control, run rather than assumed:** a *field reorder* of the same struct leaves the twin
at `[3, 6, 9, 12]`. VeriCL-side field access is by name, so the reorder hazard is **launch-side
only** — §4.3. Stating it the other way round would have been the easy overclaim.

### 4.2 Three parameter positions, three behaviours, none of them right

| written | today | site |
|---|---|---|
| `fn k(p: &Pair, …)` kernel or helper | ``error: reference parameters must be `&Array<T>`, `&mut Array<T>`, or a core `&Slice<T>`/`&SliceMut<T>` (helper params) in the vericl v0 subset`` | `lib.rs:2270-2276` — correct, one error, right span |
| `fn k(p: Pair, …)` **kernel** | ``error: gen(...) v0 only supports f32/f64/u32/i32/u64/i64 scalar parameters; `p: Pair` is outside that set`` | `lib.rs:2283-2286` classifies it `Scalar`; the message comes from `build_gen_field` `:5122-5133` and blames the wrong clause |
| `fn h(p: Pair) -> u32` **helper** | **nothing** | helpers never reach `build_gen_field`; `classify_param`'s `Type::Path(_) => Scalar` catch-all swallows it |

`docs/design-struct-comptime.md:640` records the second message as "already correct". It is not: it
tells the author to fix `gen(...)`, when the parameter class is the problem. §10.4 replaces it.

### 4.3 The positional-constructor hazard (launch side)

`XLaunch::new` fills fields by position (`generate_struct.rs:92-114`). Two same-typed fields swapped
in the *definition*, with the kernel body and the launch-call text byte-unchanged:

```text
Affine        { scale, bias }  ->  y = x*3.25 - 0.75   c0800000 c0180000 bf400000 …
AffineSwapped { bias, scale }  ->  y = x*(-0.75)+3.25  40800000 40680000 40500000 …
same body `x[i] * p.scale + p.bias`, same call `…Launch::new(scale, bias)`
```

Under this design VeriCL itself emits the constructor from the declared field order, so the hazard
becomes *internal* — which is precisely why the declaration must be hashed: if the macro generates
`PLaunch::new(a, b)` from a field order it read at expansion time, an edit to that order must move
the recorded identity, or the harness and the kernel disagree silently.

### 4.4 An adjacent hole this design surfaces but does not own

Probing the corpus's `RowWise` shape (a device aggregate with a register `Array::new` field) the
twin panics `Unexpanded Cube functions should not be called.`. **Isolated with no struct anywhere
in the kernel** (`vcl/src/bin/arraynew.rs`), the same panic occurs: `Array::new(…)` in a kernel body
compiles under `#[vericl::kernel]` and fails only at twin runtime. That is a pre-existing ungated
device-only call in the `docs/design-struct-comptime.md` §5.2 family, **not** a struct defect, and
this design does not claim it. It is recorded here because it is the first thing an implementer will
hit when they reach for §10.5's array fields, and because a v1 that rejected array fields *without*
saying this would look like it had closed something it had not.

---

## 5. The decided design

### 5.1 Shape — what the user writes

```rust
vericl::cube_struct! {
    /// Runtime parameter block. Fields are scalars, nested declared structs,
    /// or `#[cube(comptime)]` constants.
    pub struct Uniform {
        pub lower_bound: f32,
        pub upper_bound: f32,
        #[cube(comptime)]
        pub inclusive: bool,
    }
}

#[vericl::kernel(
    assumes(s.len() == y.len(), args.lower_bound.abs() <= 100.0),
    compare(abs = 1e-4),
    gen(args.lower_bound in -100.0..=100.0, args.upper_bound in -100.0..=100.0, y in 0.0..=0.0),
    instantiate(args.inclusive = false),
    uses(to_unit_interval_closed_open)
)]
#[cube(launch)]
pub fn uniform_value_map(s: &Array<u32>, args: &Uniform, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        let scale = args.upper_bound - args.lower_bound;
        let unit = if comptime!(args.inclusive) {
            to_unit_interval_closed(s[ABSOLUTE_POS])
        } else {
            to_unit_interval_closed_open(s[ABSOLUTE_POS])
        };
        y[ABSOLUTE_POS] = unit * scale + args.lower_bound;
    }
}
```

`args.lower_bound` and `args.upper_bound` are runtime scalars — generated per case, uploaded as
launch scalars, and read by both the twin and the device from the same tokens. `args.inclusive` is a
comptime field: it never reaches the GPU, it is pinned once by `instantiate(...)`, and it selects a
branch at expansion time in the kernel and at `let`-binding time in the twin.

`vericl::cube_struct!` re-emits the declaration verbatim, adds `#[derive(CubeType, CubeLaunch)]`
itself, hashes the whole block, and emits `impl vericl::StructIdentity` (and `ConfigIdentity`, §6)
per declared type.

### 5.2 Why an item macro, and why the macro must own the derives

Three reasons, in decreasing order of force:

1. **Only the declaration site has the tokens.** `SOURCE_HASH` structurally cannot cover a type
   defined elsewhere (§4.1). This is the identical argument that made `vericl::config!` an item
   macro (`docs/design-struct-comptime.md` §6.2).
2. **The macro needs the field list, not just its hash.** A token-only attribute macro on the kernel
   cannot know what `args` contains — so it cannot emit `UniformLaunch::new(…)`, cannot build the
   twin's binding, and cannot resolve `gen(args.lower_bound in …)`. The declaration block is what
   supplies that. The identity requirement and the implementability requirement are the *same*
   requirement, which is why this design has one mechanism rather than two.
3. **Owning the derives closes the derive-set escape.** If the author wrote `#[derive(CubeType)]`
   themselves, dropping `CubeLaunch` would change the type from launchable to device-local with the
   kernel's tokens unchanged. The macro emits the pair; a user-written `CubeType`/`CubeLaunch` derive
   inside the block is rejected as redundant (CS5).

### 5.3 What `vericl::cube_struct!` does, and its gates

Steps, mirroring `config.rs:330-382`:

1. parse the block as a `syn::File`; collect declared struct names;
2. run gates CS1–CS8 (below), accumulating **all** errors before returning;
3. `hash = sha256(ts.to_string())` over the whole block — the same granularity as `CONFIG_HASH`
   (`config.rs:360-374`) and as a kernel's own `SOURCE_HASH`;
4. re-emit each declared struct **verbatim, prefixed with `#[derive(::cubecl::prelude::CubeType,
   ::cubecl::prelude::CubeLaunch)]`**;
5. emit `impl ::vericl::StructIdentity for T { const STRUCT_HASH: &'static str = #hash; }` and
   `impl ::vericl::ConfigIdentity for T { const CONFIG_HASH: &'static str = #hash; }` per type.

| # | gate | why |
|---|---|---|
| **CS1** | the block must declare at least one `struct` | otherwise nothing gets a `StructIdentity` and the macro is a no-op that looks like a declaration (config G1) |
| **CS2** | every field type must be a scalar primitive or a struct declared in **this** block; `#[cube(comptime)]` fields additionally admit a unit-only enum declared in this block | a type declared in a *different* block would contribute meaning without contributing to the hash (config G6). `Array`/`Slice`/`View`/`Sequence`/`SharedMemory`/`Tensor` fields are the §10.5 deferral |
| **CS3** | no generics on a declared struct | `impl<T> StructIdentity for P<T>` gives every instantiation one hash (config G5) |
| **CS4** | no `impl` block, and no `#[cube]` anywhere in the block | a `#[cube]` method's body runs as host Rust in the twin and as expanded device code in the kernel, and nothing reconciles them — measured as the very edit that moves the twin without moving any hash (§4.1). A plain `impl` cannot be called from a `#[cube]` body at all. The escape hatch is a free `#[vericl::helper]` taking the struct, which *is* twin-generated and gated |
| **CS5** | only `std` derives; `CubeType`/`CubeLaunch` are emitted by the macro and rejected if written | a custom derive's *definition* decides the type's impls and the hash covers only the invocation (config G11); an author-chosen derive set is a silent capability switch (§5.2) |
| **CS6** | no macro invocation inside the block | a macro's tokens are opaque to `syn`'s visitors, so CS2–CS5 would be evaded wholesale (config G8) |
| **CS7** | only `struct`/`enum`/`use` items — no `fn`, `const`, `static`, `mod`, `trait`, `macro_rules!` | v1 has no method surface, so any other item is either dead or an unhashed escape (config G7) |
| **CS8** | a `use` may not rebind `core`/`std`/`alloc`/`Self`/a primitive name, and may not be a glob | CS2 resolves field types **by name** (config G12, from round 10) |

CS4 is the one that will draw fire, because the corpus uses methods heavily
(`RowWise::new_filled`, `StridedTile::new_contiguous`, `CubeMapping::cube_pos_to_xyz`). It is the
right v1 line anyway: those methods are on device aggregates carrying `Slice`/`Array` fields that
CS2 already defers, so admitting methods would buy nothing measurable and would import the entire
host/device body-divergence problem (§13, risk 2).

### 5.4 Twin treatment — the struct *is* the twin

`#[derive(CubeType)]` is purely additive: it generates companions and leaves the declared struct an
ordinary Rust struct. So for a struct whose runtime fields are scalars, the twin needs **no generated
mirror type at all**:

| kernel construct | twin |
|---|---|
| parameter `args: &P` / `args: P` | `args: P` — the author's own struct, by value |
| field read `args.lower_bound` | the same tokens; ordinary Rust field access |
| nested `args.inner.k` | the same tokens |
| `#[cube(comptime)] args.inclusive` | the pinned value, materialised in the `let` binding that builds `args` |
| passing `args` to a `#[vericl::helper]` | the helper's twin takes `P` too (`lib.rs:4336-4346` already types a helper's non-array params by their written type) |

`ref_params` (`lib.rs:3328-3346`) gains one arm: `ParamKind::Struct(ty) => quote!(#name: #ty)`.
`pred_params` (`:3363-3376`) and `ref_arg_names` (`:3349-3361`) follow. Nothing in `transform_body`
changes — the body tokens reach the twin unmodified, which is the property both the helper design and
the config design rest on: *one token stream, two consumers*.

Measured (V6): a device-local `Acc { sum: f32, n: u32 }` built in a kernel body and passed to a
helper already runs this way today, GPU vs twin **0 / 64** bit-differences. The design does not
invent the twin mapping; it gates and hashes the one that already works.

### 5.5 Launch, `gen(...)`, and `instantiate(...)`

The harness must construct `PLaunch::new(f1, …, fn)` positionally. The macro emits it from the field
order it read in `cube_struct!`, so it never guesses:

- **runtime field** → the generated value, exactly as a loose scalar parameter of that type is
  generated today (`lib.rs:5122-5200`): a float field **requires** a `gen(p.field in lo..=hi)`
  range, an integer field defaults to full-range;
- **comptime field** → the `instantiate(p.field = …)` pinned tokens, subject to the existing
  pinnable-expression gate (`is_pinnable_config_expr`, `lib.rs:2157-2185`).

`GenEntry::parse` (`lib.rs:287-323`) and `InstantiateEntry` (`:336-344`) take an `Ident`; both grow a
**two-segment dotted** form `param.field`. Depth is capped at the nesting the block declares, so
`p.inner.k` resolves through CS2's declared-type graph.

`assumes(...)` needs **no change** at the executable layer: `check_assumes` receives the same `P` the
twin does, so `args.lower_bound.abs() <= 100.0` is ordinary Rust (`lib.rs:3795-3798`). At the
prover layer nothing changes either, because a v1 struct contributes **no buffers** — `BUFFER_PARAMS`
(`lib.rs:3804`, consumed at `prover.rs:530-534`) is emitted unchanged, and `array_param_names`
(`lib.rs:3430-3437`) sees no new names.

### 5.6 Rejected alternatives

| Alternative | Why not |
|---|---|
| **Flatten in the twin too** (rewrite `p.scale` → `p_scale` in the body) | breaks the one-token-stream property the whole macro rests on; fails the moment `p` is passed to a helper or shadowed, and needs alias analysis the macro cannot do |
| **Generate a twin mirror type unconditionally** | dead weight for v1 — the author's struct already *is* the twin (V6). The mirror becomes necessary only when array fields land (§10.5), and generating it early would freeze a shape before its requirements are known |
| **An attribute on the struct (`#[vericl::cube_struct] struct P { … }`)** | an attribute macro sees only the item it decorates, so it could not reject the sibling `impl` CS4 exists to reject; the item-macro form is the config precedent and the same argument applies verbatim |
| **A blanket `impl<T: CubeType> StructIdentity for T`** | would make every undeclared struct silently valid — the exact hole being closed |
| **No declaration macro; hash the struct's `TypeId`/name** | a name is not a definition; the §4.1 A/B changes the body and not the name |
| **Accept struct params but forbid the device-local literal form** | the device-local form is the one the corpus uses (19 of 20) *and* the one accepted today; forbidding it would be a regression dressed as a gate |

---

## 6. Identity and hashing treatment

**The rule.** Every named struct type reachable from a kernel's or helper's **signature or body** —
parameter type, return type, or a struct-literal expression — must be declared with
`vericl::cube_struct!`, and its `STRUCT_HASH` folds into `identity()`.

The body clause is what closes §4.1: the offending probe passes the struct by parameter *and*
constructs it with a literal, and either route alone would have left a hole. `syn` sees a struct
literal as `Expr::Struct`, so collection is a body walk with no ambiguity, in the same visitor family
as `ComptimeRefCheck` (`lib.rs:876-899`).

**The fold.** `combine_source_hash` (`contract.rs:334-345`), already order-sensitive and already
tested for determinism and dependency-sensitivity (`contract.rs:308-320`):

```rust
__vericl_id.source_hash = ::vericl::combine_source_hash(
    SOURCE_HASH,
    &[ /* uses(...) helpers */, /* reference fn */,
       /* CONFIG_HASH per struct-comptime param type */,
       <Uniform as ::vericl::StructIdentity>::STRUCT_HASH.to_string() ],
);
```

Struct hashes append **after** the config hashes, in first-appearance order (signature left-to-right,
then body in source order), deduplicated — the dependency list is built at `lib.rs:3260-3288` and
grows one section. A `STRUCT_HASH` is a leaf `const`, so `MAX_HELPER_COMPOSITION_DEPTH`
(`contract.rs:356-377`) is untouched.

**Enforcement is by naming, not by a trait bound**, exactly as `ConfigIdentity` does
(`lib.rs:3284-3288`): the generated `identity()` body names
`<P as StructIdentity>::STRUCT_HASH`, so an undeclared type fails `E0277` with a
`#[diagnostic::on_unimplemented]` message at the parameter's own span (R1).

**One type, both positions.** `is_config_comptime_type` (`lib.rs:2106-2116`) requires
`ConfigIdentity` of *any* non-scalar `#[comptime]` parameter type, so `cube_struct!` emits
`ConfigIdentity` **as well as** `StructIdentity` with the same hash. A declared struct may therefore
also be a `#[comptime]` parameter — the corpus shape `docs/design-struct-comptime.md` §9 records as
"config type deriving `CubeType`". It is not a replacement for `vericl::config!`, because CS4 leaves
a `cube_struct!` type with no methods and the ecosystem's comptime configs are method-heavy
(132 config methods surveyed); the two macros are for the two halves of that split.

The converse does **not** hold: `vericl::config!` does not emit `StructIdentity`, because a config
type's methods are gated for *host*-callability (config G3/G4) and nothing has checked them as device
code. R6 is that rejection.

**Nested types** follow config's rule (CS2): every type reachable from a declared struct's fields
must be declared in the *same* block, so one hash covers the whole reachable graph. Cross-block
nesting is the §10.5 `deps(...)` deferral.

**Evidence surface.** `ContractRecord` (`contract.rs:150-171`) gains no field; the struct hash mixes
into `Identity.source_hash`, which `verify()` already diffs field-by-field with a rendered
`source_hash X -> Y` (`evidence.rs:399-402`). Existing evidence for struct-free kernels is
**byte-unchanged** — an M-level verification criterion, not an aspiration (§12, M4).

---

## 7. Prover treatment — zero changes in v1, and a measured plan for v1.1

**v1 changes nothing — and that is discharged, not asserted.** §8's proof leg hands the *flattened*
kernel's `BUFFER_PARAMS` and structured assumes to `prove_bounds_freedom` against the *struct* lane's
`KernelDefinition` and gets `Proved { obligations: 2 }`, identical to the flattened lane's. A struct
of scalar fields contributes no buffers, and §2.3 measured that the IR is byte-identical to the
flattened spelling. `BUFFER_PARAMS` (`lib.rs:3477-3530`), the
`BufferParam` bridge (`prover.rs:530-534`, `suite.rs:375`/`:515`), `array_param_names`
(`lib.rs:3430-3437`), `recognize_assume` (`lib.rs:4637-4645`) and `emit_obligation`
(`prover.rs:2456-2469`) are all untouched. `ParamKind::Struct` simply pushes nothing, the way
`Scalar` and `Comptime` already do.

**v1.1, when array fields land**, has one decision and it is already measured. Buffer ids are
assigned in `expand` call order and a struct flattens **in place** (§2.2), so `index == buffer id` —
the invariant `prover.rs:525-529` documents — survives if `BUFFER_PARAMS` gains one entry per array
field with a **dotted synthesised name**:

```text
BUFFER_PARAMS = [("lead", false), ("s.a", false), ("s.b", false), ("out", true)]
```

That keeps the tuple shape `(&str, bool)`, keeps `suite.rs` untouched, and makes
`buffer_id_by_name` (`prover.rs:1557-1567`) resolve `assumes(s.a.len() == out.len())` for free. The
macro-side cost is two-segment receivers in `len_call_target` (`lib.rs:4931`) and
`array_param_names`. The alternative — a nested `BUFFER_PARAMS` shape — ripples into
`suite.rs:375`/`:515` and the interpreter's `Buffer` (`interp.rs:164-171`) for no gain.

---

## 8. Ground-truth probe (validated)

One real ecosystem shape, restored rather than invented. `cubek-random/src/uniform.rs:9` declares
`#[derive(CubeLaunch, CubeType)] struct Uniform { lower_bound: f32, upper_bound: f32 }`, and the
survey's clean-room port `uniform_value_map` explicitly **dropped** it — the port's own comment lists
"the trait-impl + `args: Uniform` CubeType wrapper" among the dropped constructs and respells the two
bounds as loose `f32` parameters.

`scratchpad/cubetype/vcl/src/bin/gt.rs` runs both spellings side by side: **lane A** is the shipped
`#[vericl::kernel]` port, whose generated `reference` twin is the oracle; **lane B** is the identical
body with `args: &Uniform`, launched with `UniformLaunch::new(lower, upper)`. Same five helpers
(`taus_step_0/1/2`, `lcg_step`, `to_unit_interval_closed_open`), same inputs, `n = 4096`:

```text
GPU(A flattened) vs GPU(B struct) : bit-differences = 0 / 4096
twin(A)          vs GPU(B struct) : worst |e-a| = 7.629e-06   (declared abs = 1e-4)
twin(A)          vs GPU(A flat)   : worst |e-a| = 7.629e-06
BUFFER_PARAMS(A) = [("s0",false),("s1",false),("s2",false),("s3",false),("y",true)]
```

Three things at once: the struct form **is** the flattened kernel bit for bit; the twin VeriCL
already generates is a valid oracle for the struct form at the declared tolerance; and the worst
error reproduces the survey's independently recorded **7.63e-6** for this kernel, including its
FMA-contraction explanation.

**The proof leg closes too** (`scratchpad/cubetype/vcl/src/bin/prove.rs`). On a reduced pair of the
same kernel, the flattened lane's `BUFFER_PARAMS` and structured assumes were handed to
`vericl_ir::prove_bounds_freedom` **twice** — once against the flattened `KernelDefinition` and once
against the struct lane's:

```text
BUFFER_PARAMS   = [("s", false), ("y", true)]
structured      = [LenEq { a: "s", b: "y" }]
ir_hash flat    = sha256:7646a3799ee76e300f0e6df0c1e880633b410539ec15ec311bbf626975c3aa9d
ir_hash struct  = sha256:7646a3799ee76e300f0e6df0c1e880633b410539ec15ec311bbf626975c3aa9d
prove(flat lane)= Proved { obligations: 2 }
prove(struct IR)= Proved { obligations: 2 }
```

Same IR, same buffer custody, same two obligations discharged. That is §7's "zero prover changes"
as a measurement rather than an argument.

**Coverage cross-check (honesty).** This probe demonstrates the *mechanism*, not reach. `Uniform` is
one of only five all-scalar `CubeLaunch` structs in the corpus, and upstream reaches it through
`#[cube] impl PrngRuntime for Uniform` — an impl block VeriCL structurally cannot annotate
(`lib.rs:2518`, `:3528` parse `ItemFn`). What v1 buys here is that a *clean-room port* no longer has
to drop the wrapper, and that the port's identity covers the wrapper's definition. §11 does not count
it as an unlocked site.

---

## 9. Compatibility matrix

Every cell measured unless marked. "PASS" = wgpu/Metal differential green at the listed sizes.

| Feature × runtime struct param | v1 | Evidence |
|---|---|---|
| struct of scalar fields, by reference | **support** | X1 bit-exact vs flattened; I1 identical `ir_hash` |
| struct of scalar fields, by value | **support** | X5 |
| **nested** declared struct (scalars) | **support** | X4; CS2 requires same-block declaration |
| `#[cube(comptime)]` field | **support** | X3 — slot kept, plain host type, pinned via `instantiate(p.f = …)` |
| `f32` `#[cube(comptime)]` field | **reject** (rustc-mediated) | X3 — `XCompilationArg: Hash`; CS2 admits integer/bool/unit-enum comptime fields only |
| two struct params of the same type | **support** | X5 |
| struct param **anywhere** in the parameter list | **support** | I1 — position does not change the IR |
| device-local struct literal + `#[vericl::helper]` param | **support** (this is what the corpus does) | V6 PASS 0/64; V3 shows it is accepted today, ungated — v1 gates and hashes it |
| **generic** kernel (93% co-occurrence) × struct param | **support** | orthogonal: the struct type itself may not be generic (CS3), the *kernel* may — `instantiate(F = f32)` unchanged |
| **`Vector`/`Line`** (46% co-occurrence) | **support** (inherited) | a `Vector` is an array *element* type (`lib.rs:2370`), never a param type; the struct contributes scalars only |
| core `Slice` × struct param | **support** (inherited) | slices fold into `ArrayRef`/`ArrayMut` (`lib.rs:2262-2268`); no interaction |
| **cooperative** (`cooperative(cube_dim = N)`) | **support** | a struct param is cube-uniform by construction (a launch scalar); threads through the two `ParamKind` matches `coop.rs:1042-1078`, `:1112-1190` and `UniformCtx::is_uniform` `:695-701` |
| `SharedMemory` × struct param | **support** (orthogonal) | `SharedMemory` is a local, never a param (`coop.rs:250-277`) |
| `wrapping` × struct param | **support** | the clause is about integer *ops* in the body (`lib.rs:3005-3021`); a struct param adds none |
| `uses(...)` composition — helper takes the struct | **support** | V6; the helper's twin types it by its written type (`lib.rs:4336-4346`) |
| `uses(...)` composition — helper takes a struct **field** | **support** | an ordinary scalar argument |
| `assumes(...)` over a struct field | **support, executable-only** | the twin holds the real struct; the prover-visible recognizers take array `.len()` and element bounds only (`lib.rs:4637-4645`) — a **pre-existing** limit shared with scalars |
| `gen(p.field in lo..=hi)` | **support** (new grammar) | §5.5; the two-segment form is the only contract-surface change |
| `instantiate(p.field = …)` for a comptime field | **support** (new grammar) | §5.5, reusing `is_pinnable_config_expr` `lib.rs:2157-2185` |
| f64 lane | **support** (inherited) | field precision is per-field; the compare tier still comes from the `ArrayMut` element type (`lib.rs:3390-3419`) |
| declared struct also used as a `#[comptime]` param | **support** | §6 — `cube_struct!` emits `ConfigIdentity` too |
| `vericl::config!` type used as a **runtime** struct param | **reject** (targeted, R6) | a config's methods are gated host-callable, not device-callable |
| **`Array`/`Tensor` field** | **reject** (targeted, R3) → v1.1 | X6–X8 measured working; I3 measured identical IR; deferred on scope, §10.5 |
| **`Slice`/`View`/`Sequence`/`SharedMemory` field** | **reject** (targeted, R3) | the View/Layout and `Sequence` milestones own these |
| **`#[cube] impl` method** on a declared struct | **reject** (new gate, R4) | §4.1 measured the divergence this creates |
| **generic** struct type (`P<T>`) | **reject** (targeted, R5) | CS3; one hash per instantiation otherwise |
| **enum** as a runtime struct param (payload-carrying) | **reject** (targeted, R7) | `generate_runtime_enum.rs`; blocks corrected sites 11–12 |
| `&mut P` struct param | **reject** (targeted, R8) | meaningful only once a field is a buffer (X7); v1 has none, so it would be a no-op that looks like an output |
| struct **return** type from a helper | **reject** (targeted, R2) | V5 measured: accepted today with no gate; v1 closes it rather than growing it |

No silent gaps: every feature is supported, deferred-with-rejection, or out with the rejection site
named.

---

## 10. The v1 subset boundary

### 10.1 Contract / macro additions

1. `vericl::cube_struct! { … }` — new item macro, `crates/vericl-macros/src/cube_struct.rs`,
   modelled on `config.rs`.
2. `vericl::StructIdentity` — new trait in `crates/vericl/src/contract.rs`, one associated
   `const STRUCT_HASH: &'static str`, with `#[diagnostic::on_unimplemented]`.
3. `ParamKind::Struct(Type)` — fifth variant (`lib.rs:2043-2062`).
4. `gen(...)` / `instantiate(...)` accept a two-segment `param.field` name.

### 10.2 Accepted (v1)

A parameter `p: P` or `p: &P` on a `#[vericl::kernel]` or `#[vericl::helper]`, and a struct-literal
`P { … }` in a kernel or helper body, where:

- `P` is declared inside a `vericl::cube_struct! { … }` block; and
- every field of `P` is a scalar primitive (`f32`/`f64`/`u32`/`i32`/`u64`/`i64`/`usize`/`bool`), a
  struct declared in the same block, or a `#[cube(comptime)]` integer/bool/unit-enum; and
- every runtime float field has a `gen(p.field in lo..=hi)` range and every comptime field has an
  `instantiate(p.field = …)` pinnable value.

Everything else about the body is governed by the existing subset.

### 10.3 Rejected, with targeted errors

**R1 — a struct type not declared with `vericl::cube_struct!`** (rustc-mediated, via
`#[diagnostic::on_unimplemented]` on `StructIdentity`, at the parameter's or literal's span):

> ``error[E0277]: `Pair` is used as a runtime CubeType parameter but is not declared with a `vericl::cube_struct!` block``
> `label: not a vericl cube struct`
> ``note: wrap the struct declaration in `vericl::cube_struct! { … }` so vericl can fold the struct's definition into kernel identity, emit the CubeType/CubeLaunch derives, and build the launch argument from the declared field order — a field reorder or type change would otherwise alter what the kernel computes while leaving its recorded identity bit-identical``

**R2 — a struct or tuple return type on a kernel or helper** (macro-authored, at the return type's
span; replaces today's silent acceptance, V5):

> ``error: a `#[vericl::helper]` may not return a struct or tuple in the vericl v0 subset — the twin and the device body would each construct the value, and vericl has no per-field comparison for a returned aggregate; return the fields as separate scalars, or take a `&mut Array<T>` out-parameter``

**R3 — a non-scalar field in a `cube_struct!` block** (macro-authored, at the field type's span):

> ``error: a vericl cube struct field must be a scalar (f32/f64/u32/i32/u64/i64/usize/bool), a struct declared in this same block, or a `#[cube(comptime)]` integer/bool/unit-enum — `a: Array<f32>` is a buffer-valued field, which is deferred: it lowers to its own kernel binding (measured: one buffer per array field, flattened in place at the struct's parameter slot), so it needs a twin mirror type holding `&[T]`, a per-field entry in the compared-buffer set, and a `gen(len(p.a = N))` form. None of those exist yet``

**R4 — an `impl` block or a `#[cube]` attribute inside a `cube_struct!` block** (macro-authored, at
the item's span):

> ``error: a vericl cube struct block declares fields only — an `impl` block is outside the v0 subset. A `#[cube]` method's body runs as ordinary host Rust in the reference twin and as expanded device code in the kernel, and nothing reconciles the two (measured: editing such a method changed the twin from [3,6,9,12] to [4,5,6,7] with every recorded hash bit-identical). Write the operation as a `#[vericl::helper]` free function taking the struct — a helper's twin is generated from the same tokens the device gets, and its body is gated``

**R5 — a generic declared struct** (macro-authored, at the generics' span):

> ``error: a vericl cube struct may not be generic — `StructIdentity` would give every instantiation the same STRUCT_HASH, so a change reachable only through one type argument would be invisible to kernel identity; declare the concrete shapes you launch``

**R6 — a `vericl::config!` type used as a runtime (non-`#[comptime]`) parameter** (rustc-mediated;
`config!` does not emit `StructIdentity`, so R1's text applies, with an added note):

> ``note: `TileCfg` is declared with `vericl::config!`, which gates its methods for HOST-callability because a comptime config runs on the host. A runtime parameter is device data; declare it with `vericl::cube_struct!` instead (a `cube_struct!` type may also be used as a #[comptime] parameter — the reverse is not sound)``

**R7 — an enum as a runtime parameter type** (macro-authored, at the parameter's span):

> ``error: a payload-carrying runtime enum parameter is outside the vericl v0 subset — CubeCL lowers it to a tag plus every variant's payload, and the twin would need a matching host discriminant model; a `#[cube(comptime)]` unit enum field inside a `vericl::cube_struct!` type is supported instead``

**R8 — `&mut P`** (macro-authored, at the parameter's span):

> ``error: a runtime struct parameter must be taken by value or by shared reference — `&mut P` differs from `&P` only for buffer-valued fields (a `&mut` struct's array field becomes a ReadWrite binding), and buffer-valued fields are deferred (see the field-type error), so `&mut` would declare an output that cannot exist``

### 10.4 Wording and gate corrections landing with v1

Three pre-existing bugs this milestone makes reachable and must fix:

1. `lib.rs:2283-2286` — the `Type::Path(_) => Scalar` catch-all swallows every by-value struct. It
   must narrow to the written-out scalar set, or R1/R7 never fire and V2's misleading `gen(...)`
   message survives.
2. `lib.rs:5122-5133` — the `gen(...)` message blames the wrong clause for a struct parameter (V2).
   With (1) fixed it becomes unreachable for structs; keep it for genuinely unsupported scalars.
3. Helper return types are ungated (V5) — R2.

### 10.5 Deferred (v1.1+, rejected with a pointer, not rejected forever)

| Deferral | Why | Measured basis |
|---|---|---|
| `Array<T>` / `Tensor<T>` fields | needs a generated twin mirror (`P { a: &'a [f32] }`), dotted `BUFFER_PARAMS` entries, per-field compare-tier selection, and a `gen(len(p.a = N))` form | X6–X8 and I3: it works, it flattens in place, the IR is identical — the deferral is scope, not risk. **Zero corpus instances** (§3.3) |
| `Slice`/`View`/`Sequence` fields | owned by the View/Layout and `Sequence` milestones; 11 of 25 `CubeLaunch` definitions and 3 of the corrected 20 sites wait on them | §3.3, §3.5 |
| `#[cube]` methods on a declared struct ("struct helpers") | the twin would have to be generated from the method body the way `#[vericl::helper]` already is — a real v1.1, since the machinery exists | §4.1 measured the divergence; CS4 rejects it meanwhile |
| payload-carrying runtime enums | `generate_runtime_enum.rs`; blocks corrected sites 11–12 together with the `#[cube] impl` method they call | §3.5 |
| cross-block nested types (`deps(...)`) | CS2 requires one block per reachable type, mirroring config's identical rule | `docs/design-struct-comptime.md` §7 |
| struct/tuple **return** types | R2 rejects rather than grows; needs per-field comparison semantics that no compare mode has | V5 |
| structurally-recognised `assumes` over struct fields | the recognizers take array `.len()` and element bounds; a scalar field is no different from a scalar param, which has the same pre-existing limit | `lib.rs:4284-4286` |
| `Array::new` in a kernel body | pre-existing ungated device-only call, surfaced here but **not owned** here (§4.4) | D2, D2′ |

---

## 11. Coverage projection — measured, not estimated

**Of the corrected 20 sole-blocker sites, v1 unlocks zero.** §3.5 lists the co-gate for each. By
category: `Sequence` 3, device aggregates with `Slice`/`SharedMemory`/`View` fields 6, trait-generic
or associated-type parameters 8, runtime enum reached through a `#[cube] impl` method 2, cmma 1.
Site 15 (`new_coordinates`) is the only one whose *parameter* v1 admits; its
`-> Value<Vector<u32, N>>` return blocks it.

**What the corpus does pay:**

- **+8 gate-free non-test `fn`s, for free** — the census correction (§3.1), one of them ported and
  passing today. Measured `fn_nontest` **12 → 20**, with the return-type caveat taking the honest
  figure to **18**.
- **One live soundness hole closed** (§4.1), plus two diagnostic corrections (§10.4) and one
  adjacent hole documented rather than inherited (§4.4).
- **The dogfood surface**: the clean-room port of `Uniform`/`Normal`/`Bernoulli` no longer has to
  drop the `args:` wrapper (§8), and the wrapper's definition becomes part of the port's identity.

**What dominates after v1.** For plain annotatable functions, the ranking becomes cmma/`Matrix` (11
sole non-test `fn`s), `plane_*` (9), `CubeType`-arg on the v0 name list (5), `comptime!{}` out of
subset (4). For the *residual* of this gate specifically, the ranked co-gates are trait-generic and
associated-type parameters (8 sites — and that is the `impl`/`trait` item wall in another costume,
since those types are defined by traits VeriCL cannot annotate), then buffer-valued struct fields (6),
then `Sequence` (3).

**The honest framing.** This is the second consecutive milestone where the #1-ranked frontier gate
turns out to be a classifier artifact plus a live soundness hole rather than a coverage unlock. That
is not a reason to skip it — an ungated, unhashed surface that already compiles is strictly worse
than an unsupported one — but it should retire "custom `CubeType` param (broad)" as a *reach*
argument. The reach argument now belongs to the `impl`/`trait` item wall, which
`docs/ecosystem-survey-2026-07.md` already names as the single largest open roadmap question.

---

## 12. Implementation plan (agent-sized milestones)

Each milestone leaves the tree green and the full example suite passing. The chain is strictly
ordered — M1 supplies the trait M2 requires, M3 supplies the running kernel M4's identity test
exercises, M5 needs M3's launch path to feed — with the single exception that M6 and M7 may be
swapped.

**M1 — `vericl::cube_struct!` declaration macro + gates CS1–CS8.**
New `crates/vericl-macros/src/cube_struct.rs` modelled on `config.rs:330-382`: parse the block,
collect declared names, run the eight gates accumulating all errors, hash
`ts.to_string()`, re-emit verbatim with the macro-supplied derives, emit `StructIdentity` +
`ConfigIdentity`.
*Verify:* a positive block (`Uniform` from §5.1) compiles and its `STRUCT_HASH` equals its
`CONFIG_HASH`; each of CS1–CS8 has a compile-fail test asserting the **exact** error string at the
**exact** span (an `Array<f32>` field → R3, an `impl` block → R4, `struct P<T>` → R5, a
`macro_rules!` → CS6, `use crate::evil as core;` → CS8); a **negative control** removing each gate in
turn must let the corresponding probe compile again; and an edit to any field name, field type, field
**order**, or derive list must move `STRUCT_HASH` (the field-order case is the §4.3 hazard and must
be asserted explicitly).

**M2 — `ParamKind::Struct` + classification, and the catch-all narrowing.**
Add the variant (`lib.rs:2043-2062`), classify `p: P`/`p: &P` in `classify_param`
(`lib.rs:2255-2293`), and **narrow the `Type::Path(_) => Scalar` catch-all at `:2283-2286`** to the
written-out scalar set. Fill in all 15 exhaustive `ParamKind` matches; audit the 13 `_`-catch-all /
`matches!` filters (`lib.rs:2933`, `:2942`, `:3189`, `:3303`, `:3381`, `:3399`, `:3434`, `:3446`,
`:5018`, `:5049`, `:5059`, `:5224`, `:5262`, `coop.rs:441`) and decide each explicitly.
*Verify:* V1's `&Pair` error is replaced by R1 at the same span; V2's misleading `gen(...)` message
no longer fires for a struct; **V3's silent acceptance now fails to compile** (this is the defect the
milestone exists to close, and its regression test is a compile-fail asserting R1's text); a
by-value `u32` parameter still classifies as `Scalar` (negative control on the narrowing); and
`cargo test -p vericl-examples --test conformance` is green with `evidence/vericl.json`
**byte-unchanged**.

**M3 — twin + `check_assumes` + launch wiring.**
`ref_params`/`pred_params`/`ref_arg_names` (`lib.rs:3328-3376`) gain the `Struct` arm; the launch
argument becomes a macro-emitted `PLaunch::new(…)` in declared field order; `comptime_bindings`
(`:3378-3386`) materialises comptime fields into the twin's struct value.
*Verify:* the §8 ground-truth kernel, rewritten with `args: &Uniform` under `#[vericl::kernel]`,
passes the differential at `abs = 1e-4` with worst `|e-a|` reproducing **7.629e-6**, and its
`kernel_ir_hash` equals the flattened version's; a **negative control** swapping two fields in the
`cube_struct!` declaration without touching the kernel must both move `STRUCT_HASH` (M1) and produce
a differential FAIL if the identity fold is disabled — i.e. the two defences are shown to be
independent.

**M4 — identity folding + body-literal collection.**
Collect struct types from the signature **and** from `Expr::Struct` literals in the body; fold
`STRUCT_HASH` into `identity()` via `combine_source_hash` after the config hashes
(`lib.rs:3260-3288`).
*Verify:* the §4.1 A/B — the `Pair::fold` edit that today leaves `identity().source_hash` at
`f0096061…` must now move it, and `vericl::suite!` must report
`STALE evidence — identity mismatch (source_hash X -> Y)`; a kernel that merely *mentions* no struct
must have a **byte-identical** `identity()`; and the whole example suite's `evidence/vericl.json`
must be unchanged.

**M5 — `gen(p.field in …)` and `instantiate(p.field = …)`.**
Two-segment names in `GenEntry::parse` (`lib.rs:287-323`) and `InstantiateEntry` (`:336-344`),
resolved through the declared field graph in `resolve_gen_entries` (`:4997-5079`) and
`resolve_instantiate` (`:2514-2721`).
*Verify:* a float field with no range produces the same "no declared gen(...) range" error as a loose
float parameter, naming `p.field`; an unknown field name errors at its own span; a comptime field
pinned with a non-pinnable expression hits the existing gate (`is_pinnable_config_expr`); and a
`gen(p.field …)` naming a *comptime* field is rejected as a category error.

**M6 — composition: helpers, cooperative, generics.**
Thread `ParamKind::Struct` through `expand_helper` (`lib.rs:4021`, twin signature `:4336-4346`), the
two cooperative matches (`coop.rs:1042-1078`, `:1112-1190`) and `UniformCtx::is_uniform`
(`coop.rs:695-701`).
*Verify:* the V6 device-local shape (`Acc` built in-body, passed to a helper) passes the differential
**and** now folds the struct's hash; a cooperative kernel with a struct parameter passes at
`cube_dim = 256` for `n ∈ {256, 1024, 4096}`; a generic kernel `instantiate(F = f32)` with a struct
parameter passes; and a `wrapping` kernel with a struct parameter is unchanged.

**M7 — README/guide surface, examples, evidence.**
Two example kernels in `crates/vericl-examples/src/lib.rs` wired into `tests/conformance.rs` — one
launch struct (the §8 shape) and one device-local aggregate (the V6 shape) — plus the identity
regression test in the `config_out_of_block_backstop.rs` family and the guide section.
*Verify:* the suite reports PASS for both new kernels with `Proved` obligations on the launch one;
`VERICL_UPDATE=1` produces exactly two new evidence entries with every pre-existing entry
byte-identical under per-entry canonical-JSON SHA-256.

Ordering: M1 before M2 because `StructIdentity` must exist before a parameter can require it; M4
after M3 so the fold is exercised by a kernel that actually runs; M5 after M3 because `gen` needs the
launch path to feed.

---

## 13. Open risks, ranked (pre-registered for review round 11)

1. **The body-literal collector is a token walk, and a struct type can enter a body without a
   struct-literal expression (high).** M4 collects `Expr::Struct` and signature types. A value can
   also arrive from a helper's return (R2 rejects that), from a `let` binding whose type is written
   (`let p: Pair = …`), from a type-ascribed closure, or from a path expression naming a unit struct
   (`Unit` with no braces). **Attack surface**: a kernel that names a struct type in a `let`
   annotation or as a unit struct, edits its definition, and shows the identity unmoved — the direct
   descendant of round 10's P1/P5b/P7. *Mitigation*: collect from `Type` positions **and**
   `Expr::Struct` **and** `Expr::Path` whose final segment resolves to a declared name, and add a
   compile-fail test per route. **Currently the sharpest open question for review.**

2. **CS4 rejects `#[cube]` methods, but Rust lets an `impl` block live anywhere in the crate
   (high, inherited).** This is `vericl::config!`'s pre-registered risk 3, verbatim: a second `impl`
   written *outside* the `cube_struct!` invocation is invisible to the hash and to every gate. For a
   runtime struct it is worse than for a config, because a `#[cube] impl` outside the block gives the
   device a method the twin does not have, and the failure is a *numeric divergence* rather than a
   panic. **Attack surface**: declare `P` in a block, write `#[cube] impl P { fn f(&self) … }`
   outside it, call `p.f()` from a kernel, edit the method. *Mitigation*: the differential lane
   catches the divergence for any value that reaches an output, and `ir_hash` moves whenever the
   value reaches the device — but a host-only effect is not covered. A backstop test in the
   `config_out_of_block_backstop.rs` family must pin the residual **as a passing test asserting the
   hole exists**, the way config's does.

3. **`STRUCT_HASH` is per-block, so two types in one block share a hash (medium).** Identical to
   `CONFIG_HASH` (`config.rs:368-374`) and accepted for the same reason — but a runtime struct's
   hash also authorises the *launch constructor's field order*, so a shared hash means an edit to
   struct `A` marks kernels using only struct `B` as stale. **Attack surface**: a false-stale report
   that trains users to re-record evidence reflexively. *Mitigation*: accepted for v1 (fail-loud
   beats fail-silent); a per-type hash is a mechanical v1.1 change.

4. **The macro emits the launch constructor positionally from a field order it read at expansion
   time (medium).** If `cube_struct!` and the kernel are in different crates and only one is
   recompiled, cargo's own recompilation tracking is what makes them agree. **Attack surface**: a
   stale `.rlib` with a reordered struct. *Mitigation*: `STRUCT_HASH` is a `const` in the declaring
   crate, so a stale dependency yields a stale hash and a STALE evidence report rather than a silent
   swap; a test must prove that by rebuilding only one side.

5. **`gen(p.field …)`'s two-segment grammar collides with nothing today, but `instantiate` shares a
   parser with generic-type pinning (medium).** `instantiate(F = f32, p.inclusive = false)` mixes a
   type pin and a field pin in one clause. **Attack surface**: `instantiate(p = Pair { … })` — pinning
   a whole *runtime* struct, which is meaningless — must be rejected, not silently treated as a
   generic pin. *Mitigation*: an explicit category error, with a compile-fail test.

6. **A `#[cube(comptime)]` field is comptime on the device but an ordinary field on the host
   (medium).** The twin's struct carries it as a plain value; the device bakes it into the kernel.
   They agree only because `instantiate` pins one expression. **Attack surface**: a comptime field
   whose pinned value differs between the twin binding and the launch argument (e.g. an expression
   evaluated twice). *Mitigation*: the pinned expression flows through `is_pinnable_config_expr`
   (`lib.rs:2157-2185`), which round 10 hardened; a test must assert the *same* tokens reach both
   consumers.

7. **Zero measured corpus unlock invites scope creep toward array fields (low).** The pull to "just
   also do buffers, it's measured working" is real and §10.5 documents the foundation. **Attack
   surface**: a v1 that quietly grows R3 into support without the twin mirror, the compare-tier
   selection at `lib.rs:3390-3419`, or the dotted `BUFFER_PARAMS`. *Mitigation*: R3's wording names
   all four missing pieces explicitly, so growing it requires deleting a list rather than relaxing a
   check.

8. **`KernelLauncher` accumulates scalars and metadata in thread-locals under `cubecl/std`
   (medium, latent).** With the `std` feature on, `KernelLauncher::with_info`/`with_scope` route
   through `std::thread_local!` statics rather than through the launcher
   (`cubecl-core-0.10.0/src/compute/launcher.rs:17-22`, `:48-56`), and only `into_bindings`
   (`:114-124`) drains them. A launcher fed via `LaunchArg::register` but **never launched** leaks
   its scalars into the *next real launch* — measured, as two probe kernels silently returning
   garbage from a recycled buffer, with `std` off producing correct values from identical code.
   **VeriCL is not affected today**: `kernel_definition()` introspects through `KernelBuilder` and
   `LaunchArg::expand` (`lib.rs:3814-3827`) and never constructs a `KernelLauncher`, and the
   conformance path uses the real `launch`, which drains. **Attack surface**: any future
   register-then-introspect harness — for instance a `gen(...)`-side path that wanted a
   `CompilationArg` without launching — would silently corrupt the following launch. *Mitigation*:
   record the constraint here, and keep the `CompilationArg` construction hand-built (the field-wise
   form, which the probes verified equals `register()`'s output) rather than obtained by registering.

9. **cubecl upgrade drift (low, standing).** The flattening model, the positional constructor, and
   the comptime-field `Hash` requirement are all properties of
   `cubecl-macros-0.10.0/src/generate/cube_type/generate_struct.rs`. **Attack surface**: a 0.11 that
   changes field-expansion order or makes `XLaunch::new` named. *Mitigation*: the same tripwire the
   upgrade drill uses — the I1/I3 `ir_hash` equalities (`sha256:58a9d55c…`, `sha256:e0312c05…`) are
   cheap asserts that fail loudly if the generator changes, and `docs/upgrade-drill-2026-07.md`'s
   checklist gains this doc's §2.

---

## 14. Roadmap impact

- **Corrects the frontier a second time.** "Custom `CubeType` param (broad)" is **20** sole-blocker,
  not 28, and 8 of the 28 need nothing at all. The measured gate-free non-test `fn` count moves
  **12 → 20** with zero code, and the ranking margin over cmma and `plane_*` collapses from 9× to
  1.8×. `docs/ecosystem-survey-2026-07.md`'s addendum should carry this correction the way it
  carried the struct-comptime one.
- **Retires this gate as a reach argument, and promotes the real one.** Every remaining site is
  co-gated, and 8 of the 20 are co-gated by trait-generic/associated-type parameters — the
  `impl`/`trait` item wall wearing a different hat. That wall is now unambiguously the largest open
  roadmap question.
- **Confirms the soundness-milestone pattern.** Two consecutive #1-ranked gates have resolved to
  "the gate does not exist; there is a hole instead". The pattern is worth naming: a lexical gate
  list measures what a classifier can see, and what it cannot see is exactly what compiles.
- **Corrects `docs/design-struct-comptime.md` §4.4.** Of its four named prerequisites, the
  struct-of-buffers twin is a shape the corpus does not contain (§3.3), the `LaunchArg` construction
  path is a positional `XLaunch::new` the macro can emit from the declaration it already must hash
  (§5.5), and per-field comparison semantics are not needed until array fields land (§10.5). It also
  missed a fifth: the struct's **definition must be folded into identity**, or the feature ships with
  the exact hole `vericl::config!` was built to close.
- **Does not** need prover, interpreter, `suite!`, or evidence-schema changes — measured, not
  assumed (§2.3, §7).
