# Line/Vector element support — design (July 2026)

The implementable design for VeriCL's #1 ecosystem gap: vectorized (SIMD) element types, the
gate that blocks **148 / 464** device `#[cube]` items in tracel-ai's own kernel libraries
(`docs/ecosystem-survey-2026-07.md` §1). Two deliverables:

- **A. A vectorized reference twin** — the current twin maps `&Array<f32>` → `&[f32]` and cannot
  represent a `Vector<f32, N>` element at all (the macro bans the `Vector`/`Line` idents outright,
  `crates/vericl-macros/src/lib.rs:70`). A **pinned lane-array** twin replaces it (§4).
- **B. Vectorized bounds proving** — this turns out to be *almost entirely already done*: the
  existing bounds walker proves whole-vector array kernels **unmodified**, plus **one** necessary
  soundness guard (§5).

Everything marked "validated" was checked empirically against the pinned `cubecl =0.10.0`
(z3 4.16.0 on PATH, wgpu 29 / Metal on an Apple M3), the same posture as
[docs/design-shared-memory.md](design-shared-memory.md) and [docs/ir-research.md](ir-research.md).
Probe sources are preserved in the scratchpad
(`scratchpad/linevec/src/bin/{ir,gt,dbg,prove,tycheck}.rs`, `scratchpad/linevec/smt/*.smt2`) and were
built against a clean `cubecl =0.10.0` crate plus a path-dep on `vericl-ir`. Reference kernels are
**clean-room / upstream-public** only (cubecl-core's own `runtime_tests/vector.rs`, cubek v0.2.0,
MIT/Apache-2.0, attributed) — no private source was probed, per the README policy.

File:line citations to `crates/vericl-ir/src/prover.rs`, `crates/vericl-macros/src/lib.rs`, and the
`cubecl-{core,ir}-0.10.0` / `cubek-0.2.0` source trees are current as of `3a0803d`.

---

## 0. Headline recommendation

1. **It is `Vector<P, N>`, not `Line<T>` — and `N` is comptime, not launch-dynamic.** At the pinned
   versions **`Line<T>` and the `line_size` builtin do not exist** (0 `Line<` tokens across the
   entire cubecl+cubek tree; the 8 `line_size` hits are ordinary local `let`s in cubek-std). The
   SIMD element type is `Vector<P: Scalar, N: Size>` where the width `N` is a **compile-time generic**
   (`Const<W>`, or `DynamicSize` resolved once per compilation). `Line<T>` was the *pre-0.10* name.
   This is decisive scoping: **pin `N` per contract via `instantiate`** — the width is a type-level
   fact cubecl itself carries in the IR `Type`, and the pinned prior in the brief is correct (§1).

2. **Twin (A): pinned lane-array host type, `instantiate(F=f32, N=4)`.** `Vector<F, N>` becomes a
   VeriCL-provided host `Line<f32, W>` = `[f32; W]` with element-wise ops; `&Array<Vector<F,N>>`
   becomes `&[Line<f32, W>]`; the body tokens are unchanged (`out[p] = a[p] + b[p]` works via
   `Line: Add`). The real cubecl `Vector<f32,N>` host value stores **only one element**
   (`base.rs:12` "Comptime vectors only support 1 element"), so it is useless as a twin — VeriCL
   **must** supply its own lane-array shim. I/O stays scalar-`f32`: the harness uploads a flat
   `&[f32]`, launches with the pinned vectorization, reads back scalars, and compares against the
   twin's flattened lanes. **Validated bit-exact on wgpu/Metal** (§4.5, §6).

3. **Prover (B): the whole-vector bounds obligation is already correct — plus one soundness guard.**
   Ran the *current, unmodified* `prove_bounds_freedom` on a real `Array<Vector<f32,4>>` kernel's IR:
   **`Proved { obligations: 3 }`**, with the unguarded variant correctly `Refuted` (boundary
   counterexample `abs_pos=0, len=0`). Whole-vector indexing lowers to `vector_size: 0` in the
   `IndexOperator` (the width lives in the *list's Type*), so it passes the existing trivial-vector
   assertion; `.len()` is line-granular, so the obligation is exactly the scalar one. The **one**
   necessary change is a soundness guard: `is_modeled_int` must also require `vector_size()==1`,
   because `Type::Vector(u32,4).is_int()` returns `true` today and would let a 4-lane vector be
   mis-modeled as a single scalar integer (§5.3). This is the round-8 attack surface (§11 risk 1).

4. **v1 subset = pinned-width, whole-vector 1-D elementwise + comptime-unrolled per-lane ops**, with
   scalar I/O and `View`/`Slice` explicitly out. Reinterpret-slice access, data-dependent per-lane
   cross-array indexing, vector `wrapping`, and `SharedMemory<Vector>` are rejected with targeted
   errors and deferred (§8, §10).

5. **Honest reach**: Vector is the #1 *gate incidence* but rarely the *only* gate. Of the 148,
   **13** trip Vector with no other blocking gate (21 counting the now-shimmed rejected-methods),
   and those 13 are mostly framework impls, not launchable kernels — Vector travels with `View`
   (52), `comptime!` (57), `match` (49), `plane_*` (31). v1's real value is **generalizing the
   already-provable scalar shortlist to its true vectorized element type** and the vectorized
   elementwise shape; **whole-kernel reach needs `View`/`Slice` (#2 gap) as the immediate
   follow-on** (§12).

---

## 1. The name correction: `Vector<P,N>` at the pins, width is comptime

The brief describes "`Line<T>` … dynamically line-sized at launch via `line_size`". That is the
**pre-0.10** API. At the versions VeriCL pins (`cubecl 0.10.0`, `cubek 0.2.0`), validated by
`grep` over the full source trees:

| Probe | Result |
|---|---|
| `Line<` / `Line::` tokens tree-wide (cubecl + cubek) | **0** (8 `line_size` hits are all local `let line_size = …` in cubek-std, not a type) |
| `Vector<…>` occurrences (target crates) | **~450**, e.g. `Vector<E, N>` ×59, `Vector<T, N>` ×43, `Vector<u32, N>` ×10 |
| The type definition | `cubecl-core/src/frontend/container/vector/base.rs:11`: `pub struct Vector<P: Scalar, N: Size>` |

`N: Size` is a **type-level width** (`frontend/element/base.rs:556`). Its two implementors:

- `Const<const N: usize>` (`base.rs:554,566`) — a compile-time constant; `try_value_const() → Some(N)`.
- `DynamicSize<Marker>` (`base.rs:578`) — resolved from the scope at expansion (`scope.resolve_size`);
  `try_value_const() → None`. This is the "launch-dynamic" flavor — but it is still **monomorphized
  per compilation**: the launcher picks the width at launch-config time and the IR is generated for
  that one resolved width.

Either way, the width is fixed for a given kernel compilation, and cubecl bakes it into the IR
`Type`: `CubePrimitive::as_type` for `Vector<P,N>` is `P::as_type(scope).with_vector_size(N)`
(`base.rs:299-301`). A `Vector<f32,4>` **is** `Type::Vector(Scalar(Float(F32)), 4)` in the IR.

**Consequence for VeriCL.** Pin the width per contract, exactly as `instantiate` already pins the
generic float `F`. `instantiate` is a purely lexical token substitution
(`crates/vericl-macros/src/lib.rs:553` `GenericSubst = HashMap<String, TokenStream2>`,
`subst_type_tokens` at `:890`), and the width is just another generic param to pin. This keeps the
twin **monomorphic** at width `W` and the prover **concrete** — the pinned prior is right.

*Multi-width.* A kernel that must run at widths `{1,4,8}` gets **one `instantiate` per width** in v1
(the macro supports exactly one `instantiate` clause today, `:433`), i.e. one contract each — the
same posture as instantiating a generic kernel at `f32` vs `f64` separately. A single-clause
width-sweep (`instantiate(N in {1,4,8})` expanding to three conformance items) is a mechanical
extension, deferred (§10.4). One width per contract is the v1 rule.

---

## 2. IR construct catalog (validated)

Probed by expanding four clean-room kernels through the zero-client `KernelBuilder` recipe
(ir-research.md §1) at width `Const<4>` and walking `def.body`
(`scratchpad/linevec/src/bin/ir.rs`).

### 2.1 A vectorized buffer is a `Vector`-typed `Type`; the width is NOT on the arg

```
buffer id=0 vis=Read       ty=Vector(Scalar(Float(F32)), 4)
buffer id=2 vis=ReadWrite  ty=Vector(Scalar(Float(F32)), 4)
```

The `ArrayCompilationArg` carries **only** `inplace: Option<Id>`
(`frontend/container/array/launch.rs:15`) — there is no vectorization field. All vectorization is
type-level. At **launch** the width is passed to the generated `expand`/`launch` as one extra
`usize` argument (validated: `vec_add::expand::<Const<4>>(&mut scope, 4usize, a, b, out)` — the
`4usize` is the width). The harness must supply the pinned `W` there (§4.4).

### 2.2 Whole-vector array indexing — `vector_size: 0`, width in the list type

`out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS]` over `Array<Vector<f32,4>>`:

```
binding(0) = output(2).len()                 : (vector<f32,4>) -> (u32)
binding(1) = AbsolutePos < binding(0)         : (u32,u32) -> (bool)
if(binding(1)) {
    binding(2) = input(0)[AbsolutePos]        : (vector<f32,4>, u32, u32, u32) -> (vector<f32,4>)
    binding(3) = input(1)[AbsolutePos]        : (vector<f32,4>, u32, u32, u32) -> (vector<f32,4>)
    binding(4) = binding(2) + binding(3)      : Arithmetic::Add over vector<f32,4> operands
    output(2)[AbsolutePos] = binding(4)
}
```

The `Index` JSON: `{ list: GlobalInputArray(0) ty=Vector(f32,4), index: Builtin(AbsolutePos),
vector_size: 0, unroll_factor: 1 }`. **`vector_size` is `0`, not `4`** — the width is carried in
`list.ty`/the output `ty`, not the `IndexOperator`. `Arithmetic::Add` is the ordinary variant over
vector-typed operands (§5.2). This is the axpy shape with a vector element type.

### 2.3 `.len()` counts **lines** (outer elements); the launch length is in **scalars**

`Metadata::Length { var: GlobalOutputArray(2) ty=Vector(f32,4) }` lowers `.len()`. Two independent
validations that it returns the **line** count (physical-scalars / `N`):

- `runtime_tests/metadata.rs:224` (`test_buffer_len_vectorized`): a 32-scalar buffer with `N=4`
  gives `buffer_len() == 8`.
- **My wgpu probe** (`scratchpad/linevec/src/bin/dbg.rs`, decisive): the guard
  `if ABSOLUTE_POS < out.len()` writes **nothing** when the array length is passed as `lines`, and
  is **exactly correct** when passed as `lines*N` (scalars). So `from_raw_parts` takes the **scalar
  count**, and `.len()` = `scalars / N` = lines. The physical byte size is `len_lines * N * sizeof(P)`.

This is the crux for soundness: the outer bounds obligation `pos < len` is at **line granularity**,
and `N` never enters it. The harness must size the buffer to `lines*N` scalars and pass `lines*N` as
the length; it dispatches `lines` threads.

### 2.4 Per-lane access indexes a **local** vector; output type is the scalar

`v[i]` where `v: Vector<f32,4>` is a register value (`lane_read`):

```
binding(1) = cast<vector<f32,4>>(f32(5.0))     // Vector::new(5.0) splat, see 2.5
for(local(2) in u32(0)..u32(4)) {
    binding(3) = binding(1)[local(2)]           : (vector<f32,4>, …) -> (f32)   // list = LocalConst(1)
    output(0)[local(2)] = binding(3)            // write into scalar Array<f32>
}
```

Per-lane **write** via `RuntimeCell` (`lane_write`):

```
binding(1) = output(0)[u32(0)]        // read whole vector, vector_size=0
local(2)   = binding(1)               // Copy to a LocalMut
local(2)[u32(0)] = f32(5.0)           // IndexAssign into the LOCAL vector, list=LocalMut, vector_size=0
output(0)[u32(0)] = local(2)          // write whole vector back
```

Both lane forms have **`vector_size: 0`** too. The distinguisher between "whole vector out of an
array" and "one lane out of a register vector" is the **list kind + output type**, not the
`IndexOperator`:

| Access | list.kind | list.ty | output.ty |
|---|---|---|---|
| whole-vector array read | `GlobalInput/OutputArray` | `Vector(f32,4)` | `Vector(f32,4)` |
| per-lane register read | `LocalConst`/`LocalMut` | `Vector(f32,4)` | `f32` (scalar) |

The current `array_ref` (`prover.rs:2016`) rejects `LocalConst`/`LocalMut` lists via its `other =>`
arm, so **per-lane register indexing already lands as `OutOfSubset`** unless we choose to model it
(§5.4) — a clean, honest default.

### 2.5 `Vector::new(scalar)` (splat) is a `Cast`

`Vector::new(s)` / `Vector::<f32,N>::new(5.0)` lowers to `Operator::Cast` from the scalar `f32` to
the vector type: `binding = cast<vector<f32,4>>(scalar<f32>(0))`. The splat is a GPU-defined
broadcast — a value-producing op whose result is a vector, never an index (§5.2 leaves it tainted).

### 2.6 `IndexOperator.vector_size` / `unroll_factor` — what is non-trivial

The prover currently *asserts* `vector_size ∈ {0,1}` and `unroll_factor == 1` defensively
(`check_trivial_vectorization`, `prover.rs:1998`). The probes show **all v1 vector shapes are
`vector_size: 0`**. Non-trivial `vector_size` is set in exactly one place: `index_expand` with an
explicit `Some(vector_size)` (`frontend/operation/base.rs:104`), reached only from the **reinterpret-
slice** machinery (`cubecl-std/src/reinterpret_slice.rs`, `Slice::with_line_size`/`into_vectorized`)
— a `View`/`Slice` feature that re-reads memory at a *different* vectorization than the array's
declared type. So the defensive assertion is **exactly right and needs no change**: it accepts the
whole-vector/lane shapes (`vector_size=0`) and rejects reinterpret-vectorized access, which belongs
to the `View`/`Slice` gap (§8). The "current defensive assertion becomes real modeling" ask resolves
to: *we now understand it precisely* — `vector_size≠0` ⇔ reinterpret-slice ⇔ out of subset, and the
rejection is honest, not a placeholder.

---

## 3. What the 148 actually do (usage shapes)

Sampled from cubek-reduce, cubek-random, cubecl-std, and cubecl-core's own `runtime_tests`:

- **Whole-vector elementwise** dominates: `out[p] = a[p] OP b[p]`, `if cond { input[0] } else { input[1] }`
  (`runtime_tests/vector.rs:112`), comparisons `lhs[0].less_than(rhs[0])` → `Vector<bool,N>`
  (`vector.rs:194`). Pure SIMD map, no cross-lane interaction.
- **Comptime-unrolled per-lane** into a **local** vector, affine in the lane:
  `for j in 0..N::value() { coordinates[j] = first + j as u32 }`
  (cubek-reduce `components/readers/base.rs:95`). The lane index `j` is a comptime `0..N` loop var;
  the written value is affine-in-lane; it indexes a **register** vector, **not** a data-dependent
  gather into another array. `RuntimeCell::store_at` / `to_vectorized` are the write forms.
- **Comptime width queries**: `output.vector_size()`, `input_vector_size / output_vector_size`
  reshaping (cubek-reduce writers) — the reshaping is `View`-tied reinterpret (§8).
- **The reduction shape** (cubek-reduce) is `SharedMemory<Vector<T,N>>` + `Atomic::fetch_add`
  cross-cube + `View` inputs + generic + `match` — i.e. Vector is one of *five* gates on it (§12).

**Design read:** data-dependent per-lane indexing into *other* arrays (the case that would force
scalar-expansion in the prover) **does not occur** in the sampled 148; the per-lane pattern that
exists is comptime-unrolled affine-in-lane into locals. This is what makes the cheap prover model
(§5.4) honest for v1.

---

## 4. Deliverable A — the pinned lane-array twin

### 4.1 The host `Line` type (the shim)

VeriCL ships a host type `vericl::Line<T, const W: usize>` wrapping `[T; W]`, implementing the API
surface the twin body needs (§7): element-wise `Add/Sub/Mul/Div/Neg` (and `*Assign`), bitwise ops,
per-lane comparisons returning `Line<bool, W>`, `new`(splat), `fill`, `empty`, `from_int`,
`Index<usize>`/`IndexMut<usize>` (lane access), and per-lane math methods. Every op is a
straight element-wise map:

```rust
impl<T: Add<Output=T>+Copy, const W: usize> Add for Line<T, W> {
    fn add(self, r: Self) -> Self { Line(core::array::from_fn(|j| self.0[j] + r.0[j])) }
}
```

This is a shim in the exact sense of the `cast_from`/`mul_hi` host shims
(`docs/ecosystem-survey-2026-07.md` update note): its per-lane semantics must be **GPU-ground-truth-
verified**, not assumed. §6 does that.

### 4.2 Twin type mapping

The macro pins `(F, W)` from the contract's `instantiate(F = f32, N = 4)` and rewrites element types:

| Kernel type | Twin type |
|---|---|
| `&Array<Vector<F, N>>` | `&[::vericl::Line<f32, 4>]` |
| `&mut Array<Vector<F, N>>` | `&mut [::vericl::Line<f32, 4>]` |
| `Vector<F, N>` (local, return) | `::vericl::Line<f32, 4>` |
| `Vector<bool, N>` | `::vericl::Line<bool, 4>` |

This extends the twin param map (`crates/vericl-macros/src/lib.rs:2315-2318`,
`ParamKind::ArrayRef(elem) => &[#elem]`). Because `instantiate` is lexical, `F`→`f32` and `N`→width
substitution is mechanical; the one structural addition is recognizing the `Vector<_, _>` type
constructor (a new `ParamKind::ArrayRef` element shape) and mapping the `Vector` head + its `N`
argument to `Line<_, W>`. The body tokens (`out[p] = a[p] + b[p]`, `Vector::new(x)`, `v[j]`) survive
unchanged because `Line` implements the same operator/method surface.

### 4.3 Per-lane arithmetic in the twin

Whole-vector `a[p] + b[p]` is an element-wise map over the `W` lanes — **bit-exact** with GPU SIMD
at equal precision because a vec-`W` add *is* `W` independent scalar `f32` adds, no reordering, no
cross-lane coupling (§6 proves it). Splat `Vector::new(s)` → `Line::splat(s)` (all lanes `s`). Per-
lane read/write `v[j]` → `Line`'s `Index`/`IndexMut` at lane `j`. Cross-lane reductions (`VectorSum`,
`dot`, `magnitude`) have a GPU-defined summation order and are **not** in the v1 twin surface (§7,
deferred).

### 4.4 The launch / I/O model (scalar throughout)

The harness never materializes `Line` on the GPU side:

1. **Generate** inputs as flat `Vec<f32>` of `lines * W` scalars (the existing `gen` draw, extended
   to accept a vector element type by drawing `W` scalars per line, §9).
2. **Upload** the flat scalar bytes (`f32::as_bytes`), `ArrayArg::from_raw_parts(handle, lines*W)`.
3. **Launch** `kernel::launch::<…, R>(&client, count, dim, W /*vectorization*/, args…)` with the
   pinned width `W` spliced as the vectorization `usize` (§2.1); dispatch `lines` threads
   (`count*dim ≥ lines`), output sized to `lines*W` scalars.
4. **Read back** flat `Vec<f32>`; **reshape** to `Vec<Line<f32,W>>` (chunks of `W`).
5. **Twin** runs on `&[Line<f32,W>]`; its output flattens back to `Vec<f32>`.
6. **Compare** flat scalar vs flat scalar with the existing `compare_f32_with` (§9).

No `CubeElement`-for-`Vector` is required; I/O stays in scalar-`f32` land end to end.

### 4.5 Faithfulness — bit-exact at equal precision, tolerance only for contraction

Because the lane-array twin performs the identical per-lane scalar ops in the identical order the
GPU does, at equal precision (f32) it reproduces the GPU value **bit-for-bit** for elementwise
kernels (§6). The only divergence source is the *same* one every float twin has — a fused
multiply-add the backend contracts (`a*a+b` → one rounding) that the twin computes as two roundings
— handled by the existing `compare(abs=…)` tolerance, justified from input ranges exactly as the
scalar path (`docs/ecosystem-survey-2026-07.md` `to_degrees`). **The vector model introduces zero
additional divergence.**

---

## 5. Deliverable B — vectorized bounds proving (almost already done)

### 5.1 The whole-vector obligation is the scalar obligation

Traced against the IR (§2.2) and confirmed by running the **real, unmodified**
`prove_bounds_freedom` (`prover.rs:624`) on the `Array<Vector<f32,4>>` `vec_add` kernel's IR
(`scratchpad/linevec/src/bin/prove.rs`):

```
vec_add (guarded)       => Proved { obligations: 3 }
vec_add_oob (unguarded) => Refuted { obligation: "0 <= index < a.len() (read access to `a`)",
                                     counterexample: "len_a=0, len_out=0, len_b=0, abs_pos=0" }
```

Why it already works, step by step:

- `check_trivial_vectorization` (`prover.rs:1998`) accepts `vector_size == 0` (the whole-vector case).
- `array_ref` (`:2016`) accepts the `GlobalInput/OutputArray` list — the `Vector` element type does
  not change the list *kind*.
- `Metadata::Length` binds a **line-granular** length symbol (§2.3); the obligation
  `0 <= AbsolutePos < len` is emitted at line granularity — `N` never appears.
- `value_of(AbsolutePos)` is the existing leaf; the guard `AbsolutePos < out.len()` is the existing
  path condition.

So **B is a no-op for whole-vector bounds**, modulo §5.3. The hand-written z3 confirms the same
obligation independently:

| Obligation (`scratchpad/linevec/smt/*.smt2`) | Encoding | z3 |
|---|---|---|
| line-granular store, guarded `pos < len_lines` | `pos≥0 ∧ len≥0 ∧ pos<len ∧ ¬(0≤pos<len)` | **unsat** ✓ (in bounds) |
| unguarded negative control | drop the guard | **sat** ✓ witness `pos=0, len=0` |

### 5.2 Vector-typed values are (correctly) tainted

Vector arithmetic (`Arithmetic::Add` over `vector<f32,4>` operands, §2.2), the splat `Cast`
(§2.5), and per-lane comparison results are **value**-producing ops whose results this checker has no
scalar model for. They fall through `value_of`'s `_ => None` (`prover.rs:3217`) → tainted, exactly
like `xorshift_step`'s bitwise ops today (module docs). This is sound because these values **never
feed an index** in the v1 shapes (every index is a bare `AbsolutePos`/comptime lane). A tainted value
that *did* reach an index or a branch condition fails explicitly at that use site as `OutOfSubset` —
the existing discipline, unchanged.

### 5.3 The one necessary change: guard `is_modeled_int` with `vector_size()==1`

`is_modeled_int` (`prover.rs:3254`) is `ty.is_int() && !ty.is_bool()`. **Trap, confirmed
empirically** (`scratchpad/linevec/src/bin/tycheck.rs`):

```
Vector(u32,4).is_int()  = true      // cubecl-ir type.rs:523: !is_semantic() && storage.is_int()
Vector(u32,4).is_bool() = false
=> is_modeled_int(Vector(u32,4)) = true   // TRAP
```

So a `Vector<u32,4>` value is currently *eligible* to be modeled as **one** SMT `Int` (its lane
width, 32 bits). Two live sites would act on it: `value_of`'s integer `GlobalScalar` leaf
(`:3208`) and `model_element_read`'s element-range modeling (`:2081`, which declares a scalar `elem`
leaf bounded `< b`). Modeling a 4-lane vector as a single integer is **unsound** — if that value ever
reached an index (a per-lane gather, or a future construct), the bound would be a fiction. The fix:

```rust
fn is_modeled_int(ty: &Type) -> bool { ty.is_int() && !ty.is_bool() && ty.vector_size() == 1 }
```

`Type::vector_size()` is `1` for `Scalar`, `N` for `Vector` (`cubecl-ir type.rs:487`, validated), so
this admits only scalar integers. Effect: vector integer reads become **tainted-but-unmodeled**
(never a scalar-int leaf), which is exactly what §5.2 requires. This is the single load-bearing
soundness edit and the pre-registered round-8 target (§11 risk 1).

### 5.4 Prover model decision: outer-index-only (b), not scalar-expansion (a)

Two candidate encodings for a `Vector` value:

- **(a) scalar expansion** — model a `Vector<u32,W>` as `W` separate SMT integer terms. Exact; would
  let per-lane divergent indexing into other arrays discharge. But it `W`-multiplies term counts and
  requires modeling every lane op lane-wise.
- **(b) outer-index-only** — bound the *outer* (line) index as the existing machinery does; treat
  lane **contents** as tainted (the §5.3 guard). Cheap; covers every whole-vector shape and the
  comptime-unrolled affine-in-lane pattern; **cannot** express data-dependent per-lane index
  divergence into other arrays.

**Decision: (b).** The §3 survey shows data-dependent per-lane cross-array indexing does not occur in
the 148 (the per-lane pattern is affine-in-lane into locals). So (b) is honest: it proves the
dominant elementwise shapes and rejects the case it cannot express as `OutOfSubset` rather than
approximating it. Concretely, **per-lane register indexing** (`v[j]` with `list: LocalConst/LocalMut`)
already lands as `OutOfSubset` via `array_ref`'s `other =>` arm (§2.4) — v1 keeps that rejection with
a Vector-specific message (§8). If a real kernel later needs lane-content reasoning, (a) is a scoped
extension over (b) without reworking it (comptime-unroll the `W` lanes at the recognizer).

### 5.5 Interaction with `vector_size`/`unroll_factor`, specified exactly

`check_trivial_vectorization` stays as-is and its meaning is now pinned (§2.6):

- `vector_size ∈ {0, 1}` ∧ `unroll_factor == 1` — accepted (all whole-vector/lane shapes are `0`).
- `vector_size ∉ {0,1}` **or** `unroll_factor ≠ 1` — `OutOfSubset`: reinterpret-slice /
  cross-vectorization access, a `View`/`Slice` feature (§8). Message updated to name it (§8).

---

## 6. Ground-truth feasibility probe (validated)

Hand-wrote the lane-array twin for two real `Array<Vector<f32,4>>` kernel shapes and ran them
differentially against the **real kernels launched on wgpu/Metal**
(`scratchpad/linevec/src/bin/gt.rs`):

```
vec_addmul  out = (a+b)*b   (no fma contraction possible)
    lines ∈ {1, 3, 7, 64, 257}, N=4:   every scalar BIT-EXACT (gpu.to_bits() == twin.to_bits()), 0 mismatches
vec_madd    out = a*a + b    (fma-contractible)
    lines ∈ {1, 3, 7, 64, 257}, N=4:   diffs appear (≈1 representable step), growing with size
```

**Bit-exact for the faithful case**, and the *only* divergence is the FMA contraction of `a*a+b`
(single vs two roundings) — the identical, well-understood float-contraction issue a scalar `a*a+b`
kernel has, handled by `compare(abs=…)`. This is the design's decisive evidence, in the shared-memory
doc's mold (§4.6 there): the lane-array reference is not an approximation of the SIMD computation, it
**is** the computation, lane by lane.

A subtle probe finding, load-bearing for §2.3 and §4.4: passing the array length as `lines` (not
`lines*N`) makes the GPU write **nothing** (guard `pos < len()` with `len() = lines/N` fails for all
threads). Only `lines*N` scalars is correct. The harness's launch-length units are **scalars**; the
prover's obligation units are **lines**. Both are now pinned by probe.

---

## 7. API surface — per-op verification plan

The `Vector<P,N>` surface (`cubecl-core/src/frontend/container/vector/{base,ops}.rs`), grouped by
what the twin's `Line<T,W>` must do and how each is verified. "GT" = requires GPU ground-truth (a
row in the host-shim verification test, `tests/host_shim_gpu_ground_truth.rs` precedent).

| Op group | Members | Twin `Line` behavior | Verification |
|---|---|---|---|
| **Arithmetic** | `Add Sub Mul Div` + `*Assign`, `Neg` | per-lane scalar op | **GT: bit-exact** (§6). f32/f64 exact; contraction via `compare(abs=…)` |
| **Splat / ctor** | `new(x)`, `fill(x)`, `empty`, `zeroed`, `from_int` | all lanes = `x` (`new`/`fill`); zero (`empty`/`zeroed`) | GT: `new`/`fill` (Cast splat, §2.5); `empty`/`zeroed` = zero-init |
| **Lane index** | `v[j]` read, `store_at(j,x)` / `v[j]=x` (RuntimeCell) | `[T;W]` Index/IndexMut | GT (whole-vector round-trip); prover: local-vector index ⇒ `OutOfSubset` in v1 unless comptime-unrolled |
| **Comparison** | `equal not_equal less_than greater_than less_equal greater_equal` → `Vector<bool,N>`; `and`/`or` (bool vec) | per-lane bool | GT: per-lane compare; result tainted in prover |
| **Bitwise** | `BitAnd/Or/Xor`, `Shl/Shr` + assign, `count_ones` | per-lane | GT: bit-exact (integer, like `mix_u32`) |
| **Width query** | `size()` / `vector_size()` | comptime `W` const | comptime, no runtime op (folds to `W`) |
| **Math (unary)** | `abs sqrt exp ln sin cos … floor ceil round` | per-lane, reuse `FLOAT_METHOD_WHITELIST` semantics per lane | GT per lane = the existing scalar whitelist rows, applied lane-wise |
| **Cross-lane reduce** | `VectorSum`, `dot`, `magnitude`, `normalize` | GPU-order-defined sum | **deferred** (§8) — order-sensitive, needs its own GT + tolerance story |
| **Rejected** | `cast_from`/`reinterpret` on vectors, `mul_hi` vec | — | already `FLOAT_METHOD_REJECT` (`lib.rs:176`); vector cast_from needs a vector shim (v1.1) |

v1 twin surface = arithmetic + splat/ctor + comparison + bitwise + width query + per-lane math
(lane-wise reuse of the scalar whitelist). Lane index is accepted in the twin (host `[T;W]`) but the
prover rejects a *local-vector* index site unless it is a comptime-unrolled `0..W` loop (§5.4).
Cross-lane reductions and vector `cast_from` are deferred.

---

## 8. The v1 subset boundary

### 8.1 Contract additions

- **`instantiate(F = f32, N = 4)`** — the existing `instantiate` clause gains the width param. `N`
  (the kernel's `Size` generic) pins to a concrete width `W`; the macro emits `Line<f32, W>` in the
  twin and splices `W` as the launch vectorization (§4.2, §4.4). One clause ⇒ one width per contract.
- No new launch clause is needed for the *non*-cooperative vector case: the standard 1-D dispatch
  (`ceil(lines/cube_dim)`) applies with `lines` threads and `lines*W`-scalar buffers.

### 8.2 Accepted (v1)

1-D topology (`ABSOLUTE_POS`), `Array<Vector<F, N>>` / `&mut Array<Vector<F, N>>` with `F ∈ {f32,f64}`
(numeric-vector u32/i32 accepted for bounds; compare gate §9) and comptime-pinned `N = Const<W>`;
whole-vector elementwise arithmetic/bitwise/comparison; splat `Vector::new`/`fill`; per-lane math
(whitelist, lane-wise); comptime-unrolled per-lane access into a **local** vector (`for j in 0..W`);
`uses(...)` composition with vector-typed helpers at the pinned width (§10). Bounds proved by the
existing walker + the §5.3 guard.

### 8.3 Rejected, with targeted errors

| Construct | Error site & message |
|---|---|
| Reinterpret / cross-vectorization access (`vector_size≠0` in `IndexOperator`) | prover `check_trivial_vectorization`: `"reinterpret-vectorized indexing (vector_size={v}) is a View/Slice construct outside the vericl v0 subset"` |
| Data-dependent per-lane index into another array | prover `array_ref`: `"per-lane indexing into `{list}` (a register vector) is outside the vericl v0 subset; only comptime-unrolled lane loops are supported"` |
| `SharedMemory<Vector<…>>` | macro: `SharedMemory` stays banned outside cooperative mode (`lib.rs`); cooperative+vector is v1.1 (§10.4) |
| Cross-lane reduce (`VectorSum`/`dot`/`magnitude`/`normalize`) | macro `FLOAT_METHOD_REJECT`-style: `"cross-lane reduction '{m}' has a GPU-defined summation order not yet verified — outside the vericl v0 subset"` |
| Vector `cast_from` / `reinterpret` | existing `FLOAT_METHOD_REJECT` (`lib.rs:176`) — unchanged |
| Vector element with `F ∉ {f32,f64}` in a compared `&mut` output | macro conformance gate (extend `lib.rs:4217`): `"conformance_case v1 compares f32/f64 vector outputs; `Array<Vector<{F},N>>` is outside that set"` |
| Unpinned `N` (a `DynamicSize`/generic left un-instantiated) | macro: `"a Vector<_, N> kernel requires instantiate(N = W) to pin the vector width; N is unpinned"` |
| `View`/`Slice`/`Tensor<Vector>` params | macro: `View`/`Slice`/`Tensor` stay banned (§12; the #2 gap) |

### 8.4 Deferred (v1.1+, not rejected-forever)

Vector `wrapping` (per-lane wrapping fold — the RNG cores, §9); `SharedMemory<Vector>` cooperative
reductions (cubek-reduce shape); cross-lane reductions with a verified order + tolerance; vector
`cast_from`/`from_int` host shim; a single-clause **width sweep** (`instantiate(N in {1,4,8})`);
`View`/`Slice`-carried vectorized launch entry points (the whole-kernel unlock, §12).

---

## 9. Comparison / gen / evidence

- **`gen`** (`lib.rs:3960`): recognize a `Vector<F, N>` array element. Draw `lines * W` scalars per
  array (the existing `fill_f32`/`fill_f64` path over the flat buffer); ranges apply per lane
  (element-wise, as they already do for scalar arrays). Reject an un-pinned width at the same site as
  the current `"gen(...) v0 only supports … array elements"` gate (`lib.rs:4010`) — now with the
  vector-aware message (§8.3).
- **Per-lane comparison reporting**: the flat-scalar compare (§4.4) already yields the failing
  scalar index `i`; report it as `(line = i / W, lane = i % W)` so a divergence names the lane, not
  just a flat offset. This is a `compare_*_with` reporting tweak, not a new mechanism.
- **Evidence config**: the `Claim.config` for a vectorized kernel records `vector_width = W`
  (alongside the existing precision/tolerance), so a width is part of identity and a re-run at a
  different width is a visibly different claim. The `ir_hash` already covers the vector `Type`
  (the width is in `scope`), so a width change moves identity for free.

---

## 10. Compatibility matrix — every existing feature × Vector

| Feature | Status | Detail |
|---|---|---|
| **Standard 1-D bounds proof** | **supported** | proved unmodified (§5.1) + the `is_modeled_int` guard (§5.3) |
| **`instantiate` (generics)** | **supported** | width `N` pins as another generic; lexical subst + `Vector`→`Line` head rewrite (§4.2) |
| **`compare(abs=…)` tolerances** | **supported** | flat-scalar compare; per-lane reporting (§9) |
| **`gen(...)`** | **supported** | draw `lines*W` scalars, ranges per lane (§9) |
| **Overflow / faithful integer model** | **supported (unchanged)** | vector ints tainted via the `vector_size()==1` guard; the wrapping/checked-mul machinery only ever sees scalars (§5.3) |
| **`uses(...)` composition** | **supported** | vector-typed helper inlined at the pinned width; helper twin maps `Vector`→`Line` identically (§4.2). Helper must be pinned at the caller's `W` |
| **`#[comptime]` params** | **supported (orthogonal)** | comptime params are cube-uniform scalars; `vector_size()`/`N::value()` fold to `W` |
| **div / mod indices** | **supported (orthogonal)** | index arithmetic is scalar (`ABSOLUTE_POS`, lane loops); unaffected by vector element type |
| **`wrapping`** | **deferred (v1.1)** | needs a per-lane wrapping `Line` variant + prover per-lane wrap; scalar `wrapping` unaffected. Motivating case (RNG LCG vector core) is `View`-gated anyway |
| **Gather element-range assumes** | **N/A in v1** | assumes model *scalar* array elements; a `Vector`-typed offsets array cannot be bounded per-lane (`model_element_read` guarded off by `vector_size()==1`, §5.3) ⇒ its use `OutOfSubset` |
| **Cooperative (SharedMemory/`sync_cube`)** | **deferred (v1.1)** | whole-vector shared bounds work (line-granular, §2.3), but the phase-split twin + two-thread race obligations need lane-array shared state; the cubek-reduce target also needs `Atomic` + `View` |
| **Declared-reference fallback (`reference = fn`)** | **supported** | a hand-written vector reference works over `&[Line<f32,W>]`; the weaker claim label is unchanged (`docs/design-shared-memory.md` §4.4) |
| **`View`/`Slice`/`Tensor`** | **out (the #2 gap)** | reinterpret-slice is the only `vector_size≠0` source (§2.6); whole-kernel reach needs it (§12) |

No silent gaps: every feature is supported, deferred-with-rejection, or N/A with the rejection site
named.

---

## 11. Implementation plan (agent-sized milestones)

Each milestone lands behind the existing posture (`cargo test --workspace`, clippy 0, evidence
regenerated **last**). Ordered so the soundness guard and prover confirmation come first (they are
tiny and gate everything), then the twin/macro work.

**V1 — Soundness guard + prover confirmation (prover).** Add `&& ty.vector_size() == 1` to
`is_modeled_int` (`prover.rs:3254`); update the `check_trivial_vectorization` reject message to name
reinterpret-slice (§8.3). *Verify*: the `scratchpad/linevec/src/bin/prove.rs` result reproduced as a
unit test (whole-vector `vec_add` `Proved{3}`, unguarded `Refuted`); a `Array<Vector<u32,4>>` +
`ElemsBelowConst` assume no longer models the vector read as a scalar leaf (a targeted regression
proving the guard fires).

**V2 — The host `Line<T,W>` shim + its ground-truth test (macro/runtime).** Ship `vericl::Line`
(§4.1) with the §7 v1 op surface. Add its rows to the host-shim ground-truth test
(`tests/host_shim_gpu_ground_truth.rs`): each lane op bit-exact vs a real `Vector` kernel on wgpu
(+cpu lane). *Verify*: the §6 `vec_addmul` bit-exact result reproduced through `Line`; `vec_madd`
diff bounded by the declared `compare(abs=…)`.

**V3 — Vector element recognition in the twin + `instantiate(N=W)` (macro).** Recognize
`Array<Vector<F,N>>` params; map to `&[Line<f32,W>]`; pin `(F,W)`; rewrite the `Vector` head → `Line`
in the twin body; lift the `Vector` ban under the new gate (`lib.rs:70`). *Verify*: a clean-room
`vec_add` kernel's generated twin compiles and matches a hand-written `Line` twin
(`*_twin_matches_handwritten` precedent).

**V4 — The vectorized launch/I/O + gen + compare (macro).** Splice `W` as the launch vectorization
(§4.4); size buffers to `lines*W`; extend `gen` to draw `lines*W` scalars (§9); flat-scalar compare
with per-lane `(line, lane)` reporting; extend the conformance element-type gate for vector outputs
(§8.3). *Verify*: end-to-end `conformance_case` on the clean-room `vec_add` passes on wgpu (+cpu)
across sizes; an off-by-one variant `Refuted`/diff-caught with the lane named.

**V5 — Per-lane comptime-unroll acceptance + public example (prover + example).** Accept the
`for j in 0..W` comptime-unrolled local-vector lane pattern (§3, §5.4); reject data-dependent
per-lane and reinterpret with the §8.3 messages. Wire a clean-room vectorized elementwise example
into `vericl::suite!` carrying `tested` + `proved`(bounds) at `N=4`. *Verify*: suite green, evidence
regenerated last; a lane-divergent negative control `OutOfSubset` with the targeted message.

**V6 — Generalize the survey shortlist to vectors (dogfood-style).** Re-annotate one already-provable
scalar shortlist kernel (e.g. a trig or elementwise map) at its real `Vector<f32,4>` element type and
confirm the full `tested`+`proved` pair, demonstrating v1 converts "proves the scalar core" into
"proves the vectorized kernel" for the elementwise class. *Verify*: bit-exact both lanes, bounds
`Proved`, width recorded in evidence.

B (V1) and A (V2–V4) are separable; V1 first because it is the soundness gate and is a one-line diff
plus a regression.

---

## 12. Open risks, ranked (pre-registered round-8 targets)

1. **The `is_modeled_int` vector-as-scalar trap (high).** `Type::Vector(u32,4).is_int()` is `true`
   (§5.3, validated). Without the `vector_size()==1` guard, a vector integer value can be modeled as
   one scalar SMT `Int` and mint a false bound. **Attack surface**: round-8 will hand a kernel where a
   `Vector<u32,N>` value reaches an index or an element-range assume; the honest answer is
   `OutOfSubset`/tainted, never a `Proved` on a fictional per-lane bound. Mitigation is the guard +
   the V1 regression that proves it fires. This is the analog of the round-2 branch-scoping leak —
   *it will be probed*, so the negative control is mandatory, not optional.

2. **Twin/GPU lane-op faithfulness beyond bit-exact arithmetic (high).** §6 proves elementwise
   arithmetic bit-exact, but the `Line` shim asserts per-lane semantics for the *whole* v1 op surface
   (splat, comparison, bitwise, per-lane math). A shim op that silently differs from the GPU lane op
   (e.g. a splat that the GPU implements as a broadcast-with-conversion, or a per-lane `round` with a
   different tie rule) makes the twin unfaithful. Mitigation: the V2 ground-truth test covers **every**
   v1 op row bit-exact on wgpu **and** cpu, same as the `cast_from`/`mul_hi` shims; no op ships to the
   twin surface without a GT row. **Attack surface**: round-8 hands a kernel using an un-GT'd lane op;
   the answer must be a macro-time reject (op not on the verified list), not a trusted twin.

3. **Reinterpret-slice / cross-vectorization masquerading as in-subset (medium).** The only
   `vector_size≠0` source is reinterpret-slice (§2.6), a `View`/`Slice` construct. If a kernel reaches
   the prover with a reinterpret access the macro didn't catch, `check_trivial_vectorization` must
   reject it (not silently accept a mis-sized bound). Mitigation: the assertion is retained and its
   message names the construct; a reinterpret probe (`cubecl-std reinterpret_slice`) is a negative
   control. **Attack surface**: a kernel that reads an `Array<f32>` as `Vector<f32,4>` via a slice —
   must be `OutOfSubset`, and the line-vs-scalar length confusion (§6) must not produce a wrong bound.

4. **Width-pinning vs the actual launch width (medium).** Binding the twin/obligation to `W` is sound
   only if the launch uses width `W`. Mitigation: the launch vectorization is sourced from the single
   `instantiate(N=W)` clause (one source of truth, §4.4), exactly as the cooperative `cube_dim`
   pinning (`docs/design-shared-memory.md` §9 risk 5). A mismatch is a harness bug, not silent
   unsoundness, but warrants an assertion that the passed vectorization equals `W`.

5. **`.len()` line-vs-scalar unit confusion (medium).** The launch length is in scalars, the
   obligation in lines (§2.3, §6). A harness that passes the wrong unit either writes nothing (probe
   §6) or, worse, over-dispatches. Mitigation: the harness computes `lines` and `lines*W` from one
   `W`; a unit test pins that a vectorized launch of `lines` lines dispatches `lines` threads and
   sizes the buffer to `lines*W` — the exact bug the §6 probe caught.

6. **Coverage over-claim (medium, non-soundness).** Vector is the #1 gate incidence but only **13/148**
   items trip it alone (§0.5), mostly framework impls. Claiming v1 "unlocks the 148" would be dishonest.
   Mitigation: the doc and evidence frame v1 as *generalizing the elementwise shortlist to vectors* and
   name `View`/`Slice` as the whole-kernel prerequisite (§12 below). **Attack surface**: round-8 asks
   "which whole launch entry points does v1 verify end-to-end?" — the honest answer is *the vectorized
   elementwise class + the generalized shortlist*, not the reduction/matmul launch sites (those need
   View + Atomic + comptime! + match).

7. **cubecl upgrade drift (low, standing).** The `Vector` `Type`, `vector_size: 0` index shape, and
   the launch vectorization `usize` are internals; an upgrade (e.g. a rename back to `Line`, or moving
   the width onto the arg) could change them. Mitigation: the existing "survives a CubeCL upgrade"
   health check + the `ir_hash` (which covers the vector `Type`) trip on drift; the §2 probes are the
   schema tripwire to re-run on upgrade.

---

## 13. Roadmap impact

- **Confirms** the ecosystem survey's recommendation (`docs/ecosystem-survey-2026-07.md` §4):
  Line/Vector is the next frontier, and — the survey's own qualifier — its full value is realized
  **with** `View`/`Slice`. This design delivers the Vector half (twin + the one prover guard) as a
  self-contained, low-risk milestone, and pins `View`/`Slice` as the immediate follow-on.
- **Corrects the scope**: not `Line<T>`/launch-dynamic `line_size` (a pre-0.10 API absent at the
  pins) but `Vector<P,N>` with a **comptime-pinned** width — which is why it composes with
  `instantiate` rather than needing a new runtime-vectorization mechanism.
- **Does not** need QF_BV, a new claim kind, or the two-thread machinery: whole-vector bounds are the
  existing QF_LIA obligation at line granularity (proved unmodified). The net prover change is one
  guarded predicate.
- **Widens** the provable set from "the scalar cores of tracel-ai's kernels" toward "their vectorized
  elementwise kernels", with the reduction/matmul launch sites explicitly gated behind View + Atomic
  + comptime! + match — a documented, non-silent boundary.
