# View/Slice element support — design (July 2026)

The implementable design for VeriCL's #2 ecosystem gap: the `View`/`Slice` family, which the
survey counts at **128 / 464** device `#[cube]` items (`docs/ecosystem-survey-2026-07.md` §1) and
which `docs/design-line-vector.md` §0.5/§12/§13 names as the whole-kernel follow-on to Vector.
Two deliverables:

- **A. A reference twin for slices** — the current twin maps `&Array<f32>` → `&[f32]`; a slice
  becomes a **Rust subslice** (`&arr[start..end]` / `&mut arr[start..end]`), which is the exact,
  bit-faithful, aliasing-checked host mapping (§4, §6).
- **B. Slice bounds proving** — this turns out to be **almost entirely already done**: after
  lowering, a slice access is not a new IR construct at all but an ordinary checked `Index`/
  `IndexAssign` on the **origin buffer** with index `Add(offset, i)`, which the existing bounds
  walker + faithful integer model discharge **unmodified** (§5, validated).

Everything marked "validated" was checked empirically against the pinned `cubecl =0.10.0`
(z3 4.16.0 on PATH, wgpu 29 / Metal on an Apple M3), the same posture as
[docs/design-line-vector.md](design-line-vector.md) and [docs/design-shared-memory.md](design-shared-memory.md).
Probe sources are preserved in the scratchpad
(`scratchpad/linevec/src/bin/{sliceir,sliceprove,reinterp,slicegt}.rs`,
`scratchpad/viewslice/{slice_tile_offset*.smt2,split.py}`), built against a clean `cubecl =0.10.0`
crate plus a path-dep on `vericl-ir`. Reference kernels are **clean-room / upstream-public** only
(cubecl-core's own `runtime_tests/slice.rs`, MIT/Apache-2.0, attributed) — no private source was
probed, per the README policy.

File:line citations to `crates/vericl-ir/src/prover.rs`, `crates/vericl-macros/src/lib.rs`, and the
`cubecl-{core,std}-0.10.0` source trees are current as of `bc6442e`.

---

## 0. Headline recommendation

1. **It is `Slice<E: CubePrimitive, IO: SliceVisibility = ReadOnly>`, and it is an *addressing view*
   over a base buffer, not a buffer.** At the pins, a slice is a frontend-only triple
   `(origin ∈ {Array, Tensor, SharedMemory}, offset: usize, length: usize)`
   (`cubecl-core/.../slice/base.rs:23,61`). Created by `arr.slice(start, end)` /
   `arr.slice_mut(start, end)` / `arr.to_slice()` (`base.rs:218`, `operator.rs:228`), with
   `offset = start`, `length = end − start`. **`IO` is a *type-level* read/read-write marker**
   (`ReadOnly`/`ReadWrite`, `base.rs:13-15`) — the pre-0.10 comptime-status param is gone; the
   0.10 `Slice` carries visibility in the type. `SliceMut<E>` is `Slice<E, ReadWrite>`
   (`slice/mod.rs:7`).

2. **The decisive IR fact (validated): a slice access lowers to `origin[offset + index]`.** A
   `slice[i]` read expands to `index' = add(offset, i)` then a **checked `Operator::Index` on the
   origin buffer** (`base.rs:345-372`, `read_offset::expand`); a `slice_mut[i] = v` write expands to
   an `IndexAssign`/`UncheckedIndexAssign` on the origin (`base.rs:419-435`). The slice itself emits
   **no buffer, no metadata node, no separate id** — the prover cannot even distinguish
   `arr.slice(2,5)[i]` from a hand-written `arr[2+i]` (§2.1, IR-confirmed). `slice.len()` returns the
   tracked `end − start` value, **not** a `Metadata::Length` node (§2.3). Nested slices compose
   offsets **additively** (`operator.rs:168`, IR-confirmed §2.5).

3. **Prover (B): the slice bounds obligation is the ordinary origin obligation — already discharged
   unmodified.** Running the *current, unmodified* `prove_bounds_freedom` on real slice kernels'
   IR: `to_slice()`-based whole-slice access ⟹ **`Proved{2}`**, a dynamic-offset slice
   `input.slice(pos, pos+4)[0]` ⟹ **`Proved{2}`**, a constant-offset slice under a known origin
   length ⟹ **`Proved{2}`**, a **nested** slice ⟹ **`Proved{2}`**, a gather **through a slice of an
   element-assumed array** ⟹ **`Proved{3}`** (the element-assume transfers for free), and every
   unguarded/under-constrained variant correctly **`Refuted`**/`OutOfSubset` (§5, all validated).
   The offset arithmetic composes through the faithful integer model; the origin length is the
   existing `Metadata::Length` leaf. **B is a no-op for the prover** modulo the macro allowing slice
   syntax through.

4. **Twin (A): slices are Rust subslices; the borrow checker is a sound aliasing oracle.**
   `&Slice<F, ReadOnly>` → `&[f32]`, `&Slice<F, ReadWrite>` → `&mut [f32]`; `.slice(a,b)` →
   `&x[a..b]`, `.slice_mut(a,b)` → `&mut x[a..b]`, `.to_slice()` → `&x[..]`, `slice[i]`/`slice.len()`/
   `for item in slice` map directly. **Bit-exact on wgpu/Metal** for two real slice shapes (a slice
   introduces zero numeric op — it reads the same origin scalars in the same order, §6). Rust's
   slice-creation validity (`start ≤ end ≤ len`) and mutable-aliasing rules are enforced *by the
   twin at compile/generation time* — the honest treatment of the mutable-aliasing danger zone (§4.3).

5. **v1 subset = core `Slice` (create / index / len / iterate / nest / compose), NOT the `View`
   machinery.** The survey's "128" conflates two very different populations (§3, an API-reality
   correction): the tractable **core `Slice`** (~25 items literally call `.slice()`/`.slice_mut()`/
   `to_slice`) and the intractable **`cubecl-std` `View`/`VirtualLayout`/`Coordinates` machinery** — an
   `Arc<dyn ViewOperations>` **dynamic-dispatch** abstraction over strided/permuted/chained layouts
   (`cubecl-std/.../view/base.rs:21,32`). v1 delivers core `Slice`. **`View`, reinterpret-slice
   (`with_vector_size`, already "not supported on wgpu"), and multi-dim layouts are rejected and
   deferred** (§8).

6. **Honest reach.** Core `Slice` removes the #2 gate *incidence*, and combined with Vector
   generalizes the elementwise+slice shortlist — but it unlocks **few new whole launchable kernels on
   its own**: of the core-slice items, only ~10 trip no other blocking gate, and those are
   impls/traits/test-launchers, not 1-D launch kernels; the rest are co-gated by `plane_*` / `match` /
   `comptime!` / `CubeType`-arg / `cmma` (§3, §6.3, cross-tabulated). Slice is **necessary but rarely
   sufficient**; the post-Slice frontier is `plane_*` + custom cube structs (`CubeType`-arg) + 2-D
   topology (§12).

---

## 1. The API correction: `Slice<E, IO>` is an addressing view, `View` is a separate beast

The brief asks which of `slice()`/`slice_mut()`, `View` with offsets/shapes/strides, reinterpret-
slice, `to_slice()`, and `SharedMemory` slices the 128 use. Validated by reading the 0.10 source and
expanding real kernels:

| Construct | Reality at the pins |
|---|---|
| `Slice<E, IO=ReadOnly>` | frontend triple `(origin, offset, length)`, `IO ∈ {ReadOnly, ReadWrite}` type-level marker (`slice/base.rs:23,61`); `SliceMut<E> = Slice<E, ReadWrite>` (`mod.rs:7`) |
| `arr.slice(start, end)` / `slice_mut(start, end)` | `offset = start`, `length = end − start` (`base.rs:218-233`); origin one of `Tensor`/`Array`/`SharedMemory` (`SliceOrigin`, `base.rs:31`) |
| `arr.to_slice()` / `to_slice_mut()` | whole-buffer slice: `offset = 0`, `length = arr.len()` (a real `Metadata::Length`) (`operator.rs:127-135`) |
| **offsets** | **IR values**, not metadata: `offset`/`length` are `NativeExpand<usize>` computed at the slice site and only *materialize* when the slice is indexed (as `Add(offset, i)`) |
| **`slice.len()`** | returns the tracked `end − start` value (`base.rs:239` `len(&self) { self.length }`), **not** a `Metadata::Length`/`BufferLength` node |
| nested `slice.slice(s,e)` | `offset' = s + self.offset` (additive), `length' = e − s` (`operator.rs:168-169`) |
| reinterpret-slice (`with_vector_size::<N2>` / `into_vectorized`) | rescales `offset`/`length` and sets `vector_size = Some(N2)`; **"only works with `launch_unchecked`, not supported on wgpu"** (`base.rs:82-121`, verbatim source warning) |
| `SharedMemory` slices | supported origin (`operator.rs:16-42`); the reduction-tile case |
| **`View<E, C, IO>`** (cubecl-std) | a **different type**: `Arc<dyn ViewOperationsExpand>` dynamic dispatch over a `Coordinates` space `C` + a `VirtualLayout` (strided/permuted/chained) (`view/base.rs:21,32`, `layout/*`). Multi-dim, runtime-polymorphic. **Not** the core `Slice`. |

**Consequence for VeriCL.** The core `Slice` is a thin addressing view whose every operation lowers
to an origin access — it composes with the existing machinery almost for free (§5). The `View`
machinery is a full strided-tensor-view abstraction with `Arc<dyn>` dispatch and arbitrary
coordinate→offset layouts; modeling it soundly is a separate, much larger effort (§8, deferred). The
pinned prior in the brief — "slice()/slice_mut(), View with offsets/shapes/strides, reinterpret-slice
(the vector_size≠0 source the prover currently rejects)" — is confirmed, with the sharpening that
**`View` and core `Slice` are distinct types with distinct tractability**, and reinterpret-slice is
`vector_size≠0` **and** already unrunnable on wgpu.

---

## 2. IR construct catalog (validated)

Probed by expanding seven clean-room slice kernels through the zero-client `KernelBuilder` recipe
(ir-research.md §1) and walking `def.body` (`scratchpad/linevec/src/bin/sliceir.rs`). Buffer ids: the
`input` is `GlobalInputArray(0)`, `output` is `GlobalOutputArray(1)`.

### 2.1 A slice access *is* an origin access — `origin[offset + index]`

`input.slice(2,3); output[0] = slice[0]` (`slice_select`):

```
if(UnitPos == 0) {
    binding(1) = u32(2) + u32(0)          // offset(2) + index(0)
    binding(2) = input(0)[binding(1)]     // checked Index on the ORIGIN, id=0
    output(1)[u32(0)] = binding(2)
}
```

The slice produced **no buffer, no offset IR value beyond the add, no metadata**. `input.slice(2,3)[0]`
is byte-for-byte the IR of `input[2 + 0]`. The read op is a **checked `Operator::Index`** on the
origin (the frontend passes `checked = true`, `base.rs:345`, and the origin is a global array so
`expand_index_native` emits `Operator::Index`, `indexation.rs:114-119`).

### 2.2 Slice iteration is a `RangeLoop` over origin accesses

`for item in input.slice(2,4) { sum += item }` (`slice_for`):

```
for(local(2) in u32(0)..u32(2)) {         // 0 .. length (length = 4−2 = 2, folded)
    binding(3) = u32(2) + local(2)        // offset(2) + i
    binding(4) = input(0)[binding(3)]     // origin read
    local(1) = local(1) + binding(4)
}
```

The `Iterable` impl (`base.rs:284-317`) lowers `for item in slice` to `RangeLoop(0 .. length)` reading
`slice[i]`. The loop bound *is the slice length* and the body is the origin access `origin[offset+i]`.
`RangeLoop` is already the boundable loop the walker models (ir-research.md §3).

### 2.3 `slice.len()` is the tracked `end − start`, NOT a metadata node

`output[0] = input.slice(2,4).len() as u32` (`slice_len`) lowers to `output(1)[0] = u32(2)` — the
length **folded to the constant `2` (= 4−2)**. There is **no `Metadata::Length` instruction**.
`slice.len()` returns the stored `length` value (`base.rs:239`), which for a dynamic slice is the
`Arithmetic::Sub(end, start)` result and for `to_slice()` is the origin's `Metadata::Length`. This is
the load-bearing distinction the ir-research doc flagged (`Metadata::Length` vs `BufferLength`): for a
slice, the *outer* obligation is against the **origin** length (a real `Metadata::Length`), while
`slice.len()` is a derived arithmetic value — the walker never confuses the two because it keys the
obligation off the origin's own length leaf (§5.2).

### 2.4 A mutable-slice write to an **Array** origin is `UncheckedIndexAssign`

`output.slice_mut(2,3); slice[0] = input[0]` (`slice_mut_assign`):

```
binding(1) = input(0)[u32(0)]
binding(2) = u32(2) + u32(0)
unchecked output(1)[binding(2)] = binding(1)     // UncheckedIndexAssign on the origin
```

`write_offset::expand` hardcodes `checked = false` for an `Array` origin (`base.rs:486`), `true` for
`Tensor` (`:478`), `false` for `SharedMemory` (`:494`). So an Array-origin slice write is a **raw,
unchecked** store — the backend does **not** clamp it. The bounds obligation is therefore
*safety*-critical, not merely correctness-critical, for these writes. The prover emits the obligation
for `UncheckedIndexAssign` exactly as for `IndexAssign` (`prover.rs:1959-1965`), so this is handled.

### 2.5 `to_slice()` (offset 0) and nested slices (additive offsets)

`to_slice_pos` — `let s = input.to_slice(); if pos < s.len() { output[pos] = s[pos] }`:

```
binding(0) = input(0).len()               // Metadata::Length(input)
binding(1) = binding(0) - u32(0)          // slice.len() = origin_len − offset(0)
binding(2) = AbsolutePos < binding(1)     // guard
if(binding(2)) {
    binding(3) = u32(0) + AbsolutePos     // offset(0) + pos
    binding(4) = input(0)[binding(3)]     // origin read
    output(1)[AbsolutePos] = binding(4)
}
```

`slice_nested` — `input.slice(1,10).slice(1,3)[0]` lowers to `binding(1) = 1 + 1` (inner start `1` +
outer offset `1`), `binding(2) = 2 + 0`, `input(0)[binding(2)]` — **offset composes additively** and
the origin is unchanged. `slice_dyn` — `input.slice(pos, pos+4)[0]` lowers to `input(0)[pos + 0]`
(dynamic offset `AbsolutePos` composes into the index add). Every shape is `origin[Σ offsets + index]`.

### 2.6 Reinterpret-slice is the only `vector_size ≠ 0` source, and it is already rejected

`input.slice(0,4).with_vector_size::<Const<2>>()[0]` (`reinterp`) emits an origin read with
**`vector_size = Some(2)`**:

```
binding(2) = (0*2) + 0
binding(3) = input(0)[binding(2)]         // ty vector<f32,4> -> vector<f32,2>, vector_size = Some(2)
```

Fed to the *unmodified* prover: **`OutOfSubset { reason: "reinterpret-vectorized indexing
(vector_size=2, unroll_factor=1) is a View/Slice reinterpret-slice construct outside the vericl v0
subset" }`** — `check_trivial_vectorization` (`prover.rs:1998`) fires with the message the Vector
milestone pre-wrote for exactly this (`design-line-vector.md` §2.6). So reinterpret-slice needs **no
new rejection**: it is already `OutOfSubset`, and — independently — it is unrunnable on wgpu
(`base.rs:87`), so there is no ground truth to compare against even if it were modeled.

---

## 3. What the 128 actually are — the core-Slice / View-machinery split (API-reality correction)

The survey's "View/Slice: 128" (`ecosystem-survey-2026-07.md` §1) is a single regex gate
(`\bView<|\bSlice<|LinearView|VirtualView|StridedLayout|ReadWrite|ReadOnly`). Re-running the
classifier with the two populations separated (`scratchpad/viewslice/split.py`, reusing `classify.py`)
sharpens it decisively:

| Signal | Count | Meaning |
|---|---|---|
| Items literally calling `.slice()`/`.slice_mut()`/`to_slice()` | **~25** | genuinely *create* a core slice — v1's real target |
| Items mentioning only core-`Slice` idents (no `View`/layout) | ~90 | inflated by matmul `Stage<ES, ReadOnly>` type-params and reader traits reusing the `ReadOnly`/`ReadWrite` markers |
| Items using the `View`/`VirtualLayout`/`Coordinates`/strided machinery | ~14 pure + 38 mixed | `Arc<dyn>` dynamic-dispatch tensor views — **deferred** (§8) |
| Reinterpret-slice (`with_vector_size`/`into_vectorized`/`reinterpret`) | ~10 | rejected (§2.6); 4 of them are the cubecl-std reinterpret tests, unrunnable on wgpu |

Cross-tabulating the core-slice items against the *other* blocking gates (what still blocks them after
Slice lands):

- Only **10** core-slice items trip **no other** gate — and every one is an `impl`/`trait`/test-
  launcher with `1d = False` (e.g. `Stage<ES, ReadOnly> for Filled`, `LoadStageFamily<ReadOnly>`,
  `launch_test_{1,2,3}`), **not** a clean 1-D launch kernel.
- The rest are co-gated by `cmma`/`Matrix` (matmul tiles), `plane_*` (warp reductions), `match`,
  `comptime!`, `CubeType`-arg (custom cube structs), and `Tensor`.

**Design read.** Core `Slice` is a real primitive that ~25 items use directly, and it is what the
Vector kernels' readers/writers index *through* — but Slice on its own converts almost no whole
launch entry-point from "unannotatable" to "annotatable", because those entry-points are
simultaneously gated on `plane_*` / `match` / `comptime!` / custom structs. This is the same honesty
posture the Vector doc took (`design-line-vector.md` §0.5, risk 6): Slice's v1 value is
**generalizing the provable elementwise+slice shortlist to its true addressing form and unblocking the
readers/writers**, not "unlocking the 128" (§12).

---

## 4. Deliverable A — the Rust-subslice twin

### 4.1 Type and expression mapping

The twin already maps array params through `ParamKind::{ArrayRef, ArrayMut}(elem)` →
`&[elem]` / `&mut [elem]` (`lib.rs:2536-2544`). Slices reuse this representation exactly — a slice
*is* a Rust slice:

| Kernel construct | Twin |
|---|---|
| `&Slice<F, ReadOnly>` (helper param) | `&[f32]` |
| `&mut Slice<F, ReadWrite>` / `&SliceMut<F>` (helper param) | `&mut [f32]` |
| `arr.slice(a, b)` | `&arr[a..b]` |
| `arr.slice_mut(a, b)` | `&mut arr[a..b]` |
| `arr.to_slice()` / `to_slice_mut()` | `&arr[..]` / `&mut arr[..]` |
| `slice[i]` (read) | `slice[i]` |
| `slice_mut[i] = v` | `slice[i] = v` |
| `slice.len()` / `slice.is_empty()` | `slice.len()` / `slice.is_empty()` |
| `for item in slice { … }` | `for &item in slice { … }` |
| `s.slice(a, b)` (nested) | `&s[a..b]` |

The body tokens survive unchanged because a Rust slice implements the same `Index`/`len()`/`IntoIterator`
surface. The macro's `.slice()`/`.slice_mut()`/`.to_slice()` **method** calls rewrite to Rust range
expressions (`x.slice(a,b)` → `&x[a..b]`) — a token rewrite in the same class as the Vector
milestone's `Vector::new(x)` → `Line::new(x)` head rewrite (`lib.rs:601-616`). This is a *slice gate*
mirroring the existing *vector gate*: `"Slice"` leaves `BANNED_IDENTS` (`lib.rs:72`) and the `.slice*`
methods are recognized **only** for a slice-typed receiver, exactly as `"Vector"` is lifted only for a
recognized `Vector<F,N>` element (`lib.rs:611`).

Note: at the pins, a core `Slice` is **not** a top-level launch argument — launch entry points take
`Array`/`Tensor`/`View`, and slices are created *inside* the kernel or passed to `#[vericl::helper]`
functions. So slice params appear on **helpers** (the composition case, §10), and `.slice()` calls
appear in **bodies**; both are covered above. No new launch-arg plumbing is needed.

### 4.2 Faithfulness — bit-exact, because a slice is pure addressing

A slice introduces **zero numeric operations**: `slice[i]` reads `origin[offset+i]`, the identical
scalar the GPU reads, in the identical order. So the twin inherits the array twin's already-proven
bit-exactness (`design-line-vector.md` §6, `host_shim_gpu_ground_truth`) with **no new divergence
source** — no FMA, no reordering, no cross-lane coupling. §6 confirms this on wgpu/Metal directly.

### 4.3 Mutable-slice aliasing — the danger zone, and the honest treatment

The brief flags overlapping mutable slices as the danger zone: cube may permit overlapping `slice_mut`
views WGSL-side that Rust's borrow rules forbid. The reality (`operator.rs:44-70,138-158`): a
`slice_mut` call clones the origin's expand `Variable` into a fresh `SliceExpand`, so cube **does**
let two `slice_mut` views of the same origin be simultaneously live (overlapping or not) — the type
system does not stop it, and the backend does not enforce single-writer between them.

**Decision: the twin uses Rust's borrow checker as a sound aliasing oracle.** `slice_mut(a,b)` →
`&mut x[a..b]`. Three cases follow mechanically:

- **Sequential** mutable slices (each created, used, and dropped before the next): the twin compiles;
  sound; the dominant real shape.
- **Simultaneously-live mutable slices of the same origin** — Rust rejects two `&mut` into the same
  `Vec` at once, so the derived twin **does not compile**. This is the correct outcome, split two ways:
  - *Overlapping ranges* — genuinely unsafe (write-order-dependent; a data race if the two live views
    are written from different threads). Rejecting it is *right*; the borrow error **is** the
    rejection, surfaced by the macro as a targeted message (§8.3) rather than an opaque borrow-check
    diagnostic.
  - *Disjoint ranges* (the `split_at_mut` pattern) — safe in principle, but `&mut x[0..k]` +
    `&mut x[k..n]` held at once still fails the borrow checker. **Deferred to v1.1** (recognize
    `split_at_mut`-shaped disjoint slicing); a v1 kernel that does this gets a targeted defer message,
    not a silent accept.

> **[as-built] (F2, round-9).** The two sub-bullets above describe the *aspired* v1 behavior — the
> macro detecting simultaneously-live mutable slices and emitting a prettified targeted message. That
> detection was **not** built for v1. **As shipped, the borrow checker's own diagnostic IS the
> rejection**, verbatim: an overlapping-live pair yields rustc `E0499` ("cannot borrow `x` as mutable
> more than once"), and a launder that would defeat the oracle is caught separately (`as_mut_unchecked`
> is a banned method, §8.3). This is fully sound — the borrow error rejects exactly the unsafe program,
> which is the whole decision above — it is only *less pretty* than a macro-authored message. The
> targeted **prettification** of the borrow error (naming the buffer, pointing at the two `slice_mut`
> spans, distinguishing overlapping-reject from disjoint-defer) is deferred future work, tracked in
> §8.4. The positive/negative controls are `scratchpad/slicemut/{sequential_ok,overlap}.rs` (sequential
> compiles; overlapping fails `E0499`), referenced from the example test docs.

Read-only slices (`Slice<E, ReadOnly>` from `.slice()`/`.to_slice()` — the overwhelmingly common
case, §3) are `&[T]` and alias freely, no borrow issue.

*Alternative considered and rejected for v1:* desugar every `slice_mut(a,b)[i]=v` to an origin write
`origin[a+i]=v` in the twin (mirroring the IR), which avoids the borrow conflict entirely. Rejected
because it **discards the aliasing oracle** — it would silently accept overlapping-live mutable writes
that Rust (and safety) forbid, trading the free soundness net for a small gain in expressible kernels.
v1 keeps the conservative borrow-checked mapping; the desugaring is a v1.1 option if a determinism-safe
overlapping pattern actually appears.

### 4.4 Slice-creation validity is a twin-enforced fact

cubecl's `__expand_new` does **no** bounds check at creation: `arr.slice(2, 100)` on a length-5 array
produces a slice with `offset=2, length=98` and reads `arr[2+i]` out of bounds (the trait doc claims
checked-mode clamps `end` to `len`, but the code does not — `base.rs:218-233`). Rust's `&arr[2..100]`
**panics**. So the twin *is* the slice-validity check: an invalid slice makes the differential
`tested` lane's twin panic at generation time, exactly as an out-of-bounds `arr[i]` twin does today.
This is complementary to the `proved` lane, which independently proves each **origin** access
(`offset+i < origin_len`) is in bounds (§5). The two lanes together are sound with no extra machinery:
the twin catches invalid *creation*, the prover proves in-bounds *access*.

---

## 5. Deliverable B — slice bounds proving (already done, validated)

### 5.1 The slice obligation is the origin obligation

Because a slice access lowers to `Index(origin, Add(offset, i))` (§2), the existing walker's
obligation for it is **exactly** `0 ≤ offset + i < Length(origin)` — the ordinary origin bounds check.
Step by step, all pre-existing machinery:

- `array_ref` (`prover.rs:2017`) classifies the `Index.list` — the origin is a
  `GlobalInput/OutputArray` (or a `SharedArray` for a shared-tile slice), which it already accepts.
- `array_len_and_name` binds the bound to the origin's **`Metadata::Length`** leaf
  (`prover.rs:2780-2786`) — never `BufferLength`, which stays unmodeled (the ir-research soundness
  edge, honored).
- `value_of(Add(offset, i))` (`prover.rs:3253`) resolves the index via the **faithful wrapping `Add`**
  (`wrapping_binary`, `prover.rs:1718`): `offset` traces to its constants / `Sub`(end,start) /
  `Metadata::Length`, `i` to `AbsolutePos` or a `RangeLoop` induction var. All modeled leaves.
- `emit_obligation` (`prover.rs:2187`) emits `0 ≤ idx < len` under the live path conditions + assumes.

### 5.2 Validated on the unmodified prover

Running the **current, unmodified** `prove_bounds_freedom` (`scratchpad/linevec/src/bin/sliceprove.rs`):

```
to_slice_pos   (whole-slice, guarded pos<len, LenEq)   => Proved { obligations: 2 }
to_slice_oob   (whole-slice, UNGUARDED)                => Refuted (len_input=0, abs_pos=0)
slice_dyn      (offset = AbsolutePos, guarded, LenEq)  => Proved { obligations: 2 }
slice_select_ap(const offset 2, origin len == 5)       => Proved { obligations: 2 }
slice_select_ap(const offset 2, no assume)             => Refuted (input.len() could be 0)
slice_nested_ap(offsets 1+1 composed, len == 5)        => Proved { obligations: 2 }
gather_slice   (x[offsets.to_slice()[pos]], ElemsBelowLen{offsets,x}) => Proved { obligations: 3 }
gather_slice   (no element assume)                     => OutOfSubset (gather index tainted)
reinterp       (with_vector_size::<2>)                 => OutOfSubset (vector_size=2, §2.6)
```

Every result is the sound one. The hand z3 confirms the offset-composed case independently
(`scratchpad/viewslice/slice_tile_offset*.smt2`), the realistic tiled shape
`input.slice(tile*T, tile*T+T)[j]` ⟹ `input[tile*T + j]`:

| Obligation | Encoding | z3 |
|---|---|---|
| tiled offset, with `input.len() == K*T` | `0≤tile<K ∧ 0≤j<T ∧ len=K*T ∧ ¬(0≤ tile*T+j < len)` | **unsat** ✓ (in bounds) |
| same, **without** the length fact | drop `len = K*T` | **sat** ✓ witness `tile=0,j=0,K=1,len=0` |

The length relationship is load-bearing (not vacuously proved), exactly as the array path requires.

### 5.3 What is and isn't automatic — the `slice.len()`-guard limitation

Because cubecl does not guarantee `end ≤ origin_len` at creation (§4.4), a guard `i < slice.len()`
(= `i < end − start`) does **not** by itself bound `offset + i < origin_len` — the prover would
correctly **Refute** (find `end > origin_len`) rather than assume validity. Two consequences, both
honest:

- **Whole-slice (`to_slice()`) and offset-known accesses prove cleanly** (offset 0 ⟹
  `slice.len() = Metadata::Length(origin)`; a dynamic offset like `AbsolutePos` composes and is bounded
  by the same guard as a plain gather). This is the `Proved{2}` shortlist above.
- **Constant/computed-offset slices need the origin length constrained** — a guard against
  `origin.len()`, an `assumes(origin.len() == N)`, or an `assumes(A.len() + K ≤ B.len())`
  (`LenPlusConstLe`, `prover.rs:513`) — else `Refuted`. This is sound and conservative; it is the same
  posture the array path takes for `arr[k]` with an unknown `arr.len()`.

**Decision: no prover change for v1.** The prover *cannot* recognize a "slice was created" event
(§2.1 — the IR is indistinguishable from `arr[offset+i]`), so a slice-validity assumption cannot be
injected at the IR level anyway. The Rust-subslice twin (§4.4) is the validity check; the prover
proves the concrete origin accesses. A future `Assume::SliceValid { … }` recognized at the *macro*
level (which does see `arr.slice(a,b)`) could inject `end ≤ origin_len` to make `slice.len()`-guarded
accesses provable without an origin-length assume — a scoped v1.1 extension (§8.4), not a v1
requirement.

### 5.4 The element-assume transfers through a slice, soundly and for free

`gather_slice` above (`Proved{3}`) demonstrates the composition the brief asks about: a slice of an
element-assumed array. Because the slice read lowers to a read of the **origin** buffer id, and
`model_element_read` (`prover.rs:2127`) keys the assume off that id, a read `offsets.to_slice()[pos]`
is modeled by `offsets`'s `ElemsBelowLen`/`ElemsBelowConst` bound automatically — **no slice-specific
code**. Soundness: a slice reads a *subset* of the origin's elements, all of which satisfy the assume,
so the transferred bound holds. Write-invalidation is symmetric — a `slice_mut` write to origin id
invalidates that origin's assume for subsequent reads (`elem_invalidated`, `prover.rs:2170`), so an
in-place scatter through a mutable slice is handled. This is the design's cleanest composition: origin-
id keying makes slices transparent to the gather machinery.

---

## 6. Ground-truth feasibility probe (validated)

Hand-wrote the Rust-subslice twin for two real slice-using kernel shapes and ran them differentially
against the **real kernels launched on wgpu/Metal** (`scratchpad/linevec/src/bin/slicegt.rs`):

```
windowed_sum  out[p] = Σ input.slice(p, p+W)            (slice creation + iteration + dynamic offset)
    m ∈ {1,3,7,64,257}, W=4:   every scalar BIT-EXACT (0 mismatches)
tile_last     out[p] = input.slice(p*W, p*W+W)[W-1]      (strided tile offset + index)
    m ∈ {1,3,7,64,257}, W=4:   every scalar BIT-EXACT (0 mismatches)
```

**Bit-exact across every size for both shapes.** This is the design's decisive evidence, in the
shared-memory / line-vector doc mold: the Rust-subslice reference is not an approximation of the slice
access, it **is** the access — `&input[p..p+W]` reads exactly `input[p], input[p+1], …` in order, the
same scalars the GPU `origin[offset+i]` reads. A slice adds no rounding and no reordering, so the twin
is bit-faithful by construction, and the FMA-contraction caveat that the array/vector twins carry does
not even arise for pure-addressing kernels.

### 6.3 Coverage cross-check (honesty)

The `split.py` cross-tab (§3) is the coverage evidence: core-slice items are dominated by co-gates.
The two GT kernels above are *constructed* clean-room shapes that isolate slices; the real cubek
readers that use `.slice()` also use `plane_*`/`comptime!`/custom tiles, so they do not become whole-
kernel-annotatable from Slice alone. v1's demonstrable end-to-end reach is **the slice-carrying
elementwise/windowed class + the generalized shortlist**, not the matmul/reduce launch sites (§12).

---

## 7. API surface — per-construct verification plan

The core `Slice` surface (`cubecl-core/.../slice/{base,operator}.rs`), grouped by what the twin must
do and how each is verified. "GT" = requires a GPU ground-truth row.

| Construct | Members | Twin behavior | Verification |
|---|---|---|---|
| **Create (read)** | `slice(a,b)`, `to_slice()` | `&x[a..b]`, `&x[..]` | GT: §6 bit-exact; prover: origin access §5 |
| **Create (write)** | `slice_mut(a,b)`, `to_slice_mut()` | `&mut x[a..b]`, `&mut x[..]` | GT: bit-exact write-back; borrow-checked aliasing §4.3 |
| **Index** | `slice[i]` read, `slice[i]=v` write | Rust `Index`/`IndexMut` | GT: §6; prover: `origin[offset+i]` obligation §5 |
| **Length** | `len()`, `is_empty()` | `.len()`, `.is_empty()` | comptime/value; prover: tracked `end−start` (§2.3) |
| **Iterate** | `for item in slice` | `for &item in slice` | GT: §6 `windowed_sum`; prover: `RangeLoop` over origin (§2.2) |
| **Nest** | `s.slice(a,b)` | `&s[a..b]` | prover: additive offset (§2.5), `Proved` |
| **Origins** | `Array`/`Tensor`/`SharedMemory` slice | array/shared twin; Tensor deferred | Array/Shared: supported; Tensor: with the `Tensor` gap |
| **Reinterpret** | `with_vector_size`, `into_vectorized` | — | **rejected**: `vector_size≠0` (§2.6); unrunnable on wgpu |
| **Downcast** | `downcast`, `downcast_unchecked`, `as_mut_unchecked` | — | **rejected**: unsafe type-punning, `ReinterpretSlice` internals (§8) |
| **View** | `View::new`, `.view(layout)`, `Coordinates`, `VirtualLayout` | — | **rejected/deferred**: `Arc<dyn>` strided machinery (§8) |

v1 twin surface = create/index/len/iterate/nest over `Array` and (cooperative) `SharedMemory` origins,
read-only slices fully, mutable slices under the borrow-checked aliasing rule. Reinterpret, downcast,
and the `View` machinery are rejected.

---

## 8. The v1 subset boundary

### 8.1 Contract / macro additions

- **Slice gate** — `"Slice"` leaves `BANNED_IDENTS` (`lib.rs:72`) under a recognizer that fires only
  for slice-typed values, mirroring the vector gate (`lib.rs:611`); the `.slice()`/`.slice_mut()`/
  `.to_slice()`/`.to_slice_mut()` methods and `Slice<F, ReadOnly>`/`SliceMut<F>` helper param types are
  recognized and rewritten to Rust range expressions / `&[_]`/`&mut [_]` (§4.1).
- No new launch clause: slices are internal / helper-only (§4.1); the standard 1-D dispatch applies,
  and the origin `Array`/`SharedMemory` is launched exactly as today.

### 8.2 Accepted (v1)

1-D topology (`ABSOLUTE_POS`) and cooperative mode; `arr.slice(a,b)`/`slice_mut`/`to_slice`/
`to_slice_mut` over an `Array<F>` or (cooperative) `SharedMemory<F>` origin, `F ∈ {f32,f64}` for
compared outputs (numeric u32/i32 for bounds/gather); `slice[i]` read/write; `slice.len()`/
`is_empty()`; `for item in slice`; **nested** slices; **read-only** slices without restriction and
**mutable** slices under the borrow-checked aliasing rule (§4.3); `uses(...)` composition with helpers
taking `&Slice<F>`/`&SliceMut<F>` (§10); element-assume gathers **through** a slice (§5.4). Bounds
proved by the existing walker (§5).

### 8.3 Rejected, with targeted errors

| Construct | Error site & message |
|---|---|
| Reinterpret / cross-vectorization slice (`with_vector_size`/`into_vectorized`, `vector_size≠0`) | prover `check_trivial_vectorization` (already in place, `prover.rs:2003`): `"reinterpret-vectorized indexing (vector_size={v}, unroll_factor={u}) is a View/Slice reinterpret-slice construct outside the vericl v0 subset"` |
| Simultaneously-live mutable slices of one origin (overlapping) | **[as-built]** the twin's borrow checker: rustc `E0499` on the two live `&mut (x)[..]` subslices (`scratchpad/slicemut/overlap.rs`). The macro-authored message in the original cell (`"two mutable slices of `{buf}` are live at once; …"`) was **not built** — the raw `E0499` IS the (sound) rejection; prettifying it is §8.4 future work (F2). |
| Simultaneously-live disjoint mutable slices (`split_at_mut` shape) | **[as-built]** same borrow checker (`E0499`): held-at-once disjoint `&mut` subslices still fail NLL. The macro-authored defer message (`"… need split_at_mut recognition — deferred to vericl v1.1"`) was **not built**; `split_at_mut` recognition is §8.4 deferred work (F2). |
| `slice.downcast(...)` / `downcast_unchecked` / `as_mut_unchecked` | macro `FLOAT_METHOD_REJECT`-style: `"slice type-punning method '{m}' (ReinterpretSlice internals) is outside the vericl v0 subset"` |
| `View<_, _, _>` / `View::new` / `.view(layout)` / `VirtualLayout` / `Coordinates` / `StridedLayout` | macro: `"tensor `View`/`VirtualLayout` (multi-dim strided views) is the cubecl-std layout machinery, outside the vericl v0 subset"` |
| `Tensor<F>` slice origin | macro: `Tensor` stays banned (the separate `Tensor` gap); an `Array`/`SharedMemory` origin is required |
| `slice.len()`-guarded access with an unconstrained origin length | prover: `Refuted` (honest — the origin length is load-bearing, §5.3), not a silent accept |

### 8.4 Deferred (v1.1+, not rejected-forever)

`Assume::SliceValid` injected at the macro level to make `slice.len()`-guarded accesses provable
without an origin-length assume (§5.3); `split_at_mut`-recognized disjoint mutable slicing (§4.3);
`Tensor` slice origins (with the Tensor gap); `SharedMemory<Vector>` slice reductions (needs Vector +
cooperative, §10); the `View`/`VirtualLayout`/`Coordinates` strided-view machinery (a separate large
milestone); reinterpret-slice (blocked on wgpu support upstream anyway).

**Deferred macro-ergonomics / correctness work carried from round-9 (as-built gaps, not soundness
gaps):**

- **[F2] Prettify the mutable-aliasing rejection.** As-built, overlapping/disjoint simultaneously-live
  mutable slices are rejected by the twin's borrow checker as raw `E0499`/`E0502` (§4.3 [as-built],
  §8.3). The deferred work is a macro pass that detects the simultaneously-live pattern and emits the
  buffer-named, span-pointed targeted message the §8.3 cells originally described — and that
  distinguishes overlapping (reject-forever) from disjoint `split_at_mut`-shaped (defer, recognizable).
  Sound today; only the diagnostic quality is deferred.
- **[F4] A receiver-type guard on `SliceRewriteFold` is REQUIRED before any `View` milestone.** The
  slice twin rewrite (`SliceRewriteFold::rewrite_method`, `crates/vericl-macros/src/lib.rs`) recognizes
  a slice creator by **method name + arity** (`slice`/`slice_mut`/`to_slice`/`to_slice_mut`), never by
  receiver type. This is sound *only* while no View-like type with a same-named `.slice(a, b)` /
  `.to_slice()` method (different, multi-dim strided semantics) can reach a compilable twin — which
  holds today precisely because the whole `View`/`Layout` surface is a banned ident (§8.3, F3). The
  moment a future milestone un-bans a `View`-like type, `view.slice(a, b)` would be **silently**
  mis-rewritten to a 1-D Rust subslice `&view[a..b]`. **Un-banning `View` MUST be accompanied by a
  receiver-type guard** that fires the rewrite only for genuine core-`Slice` receivers (mirroring the
  vector gate). This is a make-it-impossible-to-miss note: it lives here, at the fold's doc comment, and
  in `tasks/todo.md`.

---

## 9. Comparison / gen / evidence

- **`gen`** (`lib.rs:3960`): unchanged. Slices are internal views over `Array` origins; the origin
  arrays are drawn exactly as today. A helper taking `&Slice<F>` is exercised through the caller kernel
  whose `Array` origin `gen` already draws.
- **Comparison**: unchanged flat-scalar `compare_f32_with`. A slice writes into its origin buffer, which
  is compared as the ordinary output array; there is no separate slice buffer to compare.
- **Evidence config**: no new field. The slice offsets/lengths are in-body IR the `ir_hash` already
  covers (they are `scope` instructions), so a change to a slice bound moves identity for free; a
  slice-carrying kernel's `Claim` is otherwise identical to an array kernel's.

---

## 10. Compatibility matrix — every existing feature × Slice

| Feature | Status | Detail |
|---|---|---|
| **Standard 1-D bounds proof** | **supported (unchanged)** | slice access = origin `Index(origin, offset+i)`; proved unmodified (§5, validated) |
| **`instantiate` (generics)** | **supported** | slice element `F`/origin generics pin lexically, as today |
| **Vector elements (`Slice<Vector<F,N>>`)** | **supported (whole-vector) / rejected (reinterpret)** | whole-vector slice = line-granular `origin[offset+i]` on `Array<Vector>` + the line-vector twin (`&[Line<f32,W>]`); reinterpret (`with_vector_size`) is the `vector_size≠0` reject (§2.6). Composes with `design-line-vector.md` §5 |
| **`compare(abs=…)` tolerances** | **supported** | flat-scalar compare of the origin buffer; a slice adds no numeric op (§6) |
| **`gen(...)`** | **supported (unchanged)** | draws the origin arrays; slices are internal views (§9) |
| **Overflow / faithful integer model** | **supported (unchanged)** | offset arithmetic `offset+i` uses the faithful wrapping `Add` (`prover.rs:1718`); a wrapped offset that could exceed the origin is `Refuted`, never `Proved` — a wrapped index is still out of bounds |
| **`uses(...)` composition** | **supported** | the dominant real usage — a `#[vericl::helper] fn f(s: &Slice<F>) -> F` inlines via `UsesRewriteFold`; helper twin maps `&Slice<F>` → `&[f32]`; the caller's origin/offset flow through inlining so the origin obligation is emitted at the inlined access |
| **`#[comptime]` params** | **supported (orthogonal)** | comptime bounds fold to constants (`slice.len()` → const, §2.3) |
| **div / mod indices** | **supported (orthogonal)** | slice offset/index arithmetic composes with div/mod-derived indices via the existing `value_of` |
| **Gather element-range assumes** | **supported** | the assume transfers through the slice via origin-id keying — `Proved{3}`, validated (§5.4); write-invalidation is origin-keyed too |
| **`wrapping`** | **supported (unchanged)** | `wrapping` clause does not reach the prover; slice value arithmetic is tainted, slice indices are non-wrapping or `Refuted` (same rule as any kernel) |
| **Cooperative (SharedMemory slices)** | **supported (bounds) / v1.1 (race)** | `shared.slice(a,b)[i]` = `SharedArray[offset+i]`, bounded by the compile-time shared length (`array_ref` handles `SharedArray`); the two-thread race walk over shared-slice *writes* wants its own validation and the reduction target also needs `plane_*`/`Atomic`/`match` — race-through-slice deferred |
| **`match` / Switch** | **supported (orthogonal)** | a slice access inside a `match` arm is bounded under that arm's path condition, as any access (`process_switch`) |
| **Declared-reference fallback (`reference = fn`)** | **supported** | a hand-written reference over `&[f32]` origins works; weaker claim label unchanged |
| **Mutable slice aliasing** | **borrow-checked** | read-only: free; mutable sequential: supported; mutable simultaneously-live: rejected (overlapping) or deferred (disjoint) via the twin borrow checker (§4.3) |
| **`View`/`VirtualLayout`/`Coordinates`/`Tensor`** | **out (deferred, §8)** | the `Arc<dyn>` strided-view machinery / the `Tensor` gap |

No silent gaps: every feature is supported, deferred-with-rejection, or out with the rejection site
named.

---

## 11. Implementation plan (agent-sized milestones)

Each milestone lands behind the existing posture (`cargo test --workspace`, clippy 0, evidence
regenerated **last**). Ordered so the (already-passing) prover confirmation is pinned first, then the
twin/macro work, since B is essentially free and A is where the effort is.

**S1 — Prover confirmation + regression pins (prover).** No production change to the walker. Add unit
tests reproducing `scratchpad/linevec/src/bin/sliceprove.rs`: `to_slice_pos`/`slice_dyn`/
`slice_nested`/`slice_select` (with/without origin-length assume) as `Proved`/`Refuted` controls, the
`gather_slice` element-assume-through-slice as `Proved{3}`, and the reinterpret kernel as the
`OutOfSubset` negative control. *Verify*: the suite reproduces every §5.2 verdict; the reinterpret
message is asserted verbatim (guards against a future edit weakening it).

**S2 — Slice gate + method/param rewrite (macro).** Lift `"Slice"` from `BANNED_IDENTS` under a
slice recognizer (mirror the vector gate, `lib.rs:611`); rewrite `.slice(a,b)`/`.slice_mut(a,b)`/
`.to_slice()`/`.to_slice_mut()` → Rust range expressions; add `ParamKind::SliceRef`/`SliceMut`
mapping `&Slice<F,ReadOnly>`/`&SliceMut<F>` → `&[f32]`/`&mut [f32]` for helper params. *Verify*: a
clean-room `windowed_sum` kernel's generated twin compiles and matches a hand-written subslice twin
(`*_twin_matches_handwritten` precedent).

**S3 — The mutable-aliasing gate (macro).** Detect simultaneously-live mutable slices of one origin
and emit the §8.3 targeted messages (overlapping = reject, disjoint = defer) instead of surfacing a
raw borrow error; keep sequential mutable slices compiling. *Verify*: a sequential `slice_mut_assign`
kernel passes; an overlapping-live variant emits the targeted reject; a `split_at_mut`-shaped variant
emits the targeted defer.

> **[as-built] (F2, round-9).** S3 shipped **partially**: the *soundness* half landed and the
> *ergonomics* half is deferred. The sequential-mutable-slice half works (the twin compiles and runs —
> pinned by the committed `sequential_slice_mut_scale` twin test), and the rejection half works via the
> borrow checker (overlapping-live ⟹ `E0499`, `scratchpad/slicemut/overlap.rs`). What was **not** built
> is the macro *detection* that replaces the raw borrow error with a prettified §8.3 message and splits
> overlapping-reject from disjoint-defer — deferred to §8.4. So as-built S3 is: sequential passes
> (committed twin test), overlapping rejected (borrow checker, scratch compile-fail control), disjoint
> also rejected-not-deferred (borrow checker) — every case sound, the diagnostic just unprettified.

**S4 — End-to-end slice conformance + bit-exact GT (macro/runtime).** Wire the §6 `windowed_sum` /
`tile_last` shapes through `conformance_case` (twin vs wgpu, flat-scalar compare); add a
`host_shim`-style row asserting slice iteration/index bit-exact. *Verify*: the §6 bit-exact result
reproduced through the generated pipeline across sizes; an off-by-one window variant diff-caught.

**S5 — Slice example in the suite + public dogfood (example).** Wire a clean-room slice example
(`windowed_sum` at `F=f32`) into `vericl::suite!` carrying `tested` + `proved`(bounds); re-annotate one
already-provable shortlist kernel to read its inputs through a `to_slice()` and confirm the full pair.
*Verify*: suite green, evidence regenerated last; a `slice.len()`-guarded-without-origin-length variant
`Refuted` (the §5.3 honesty control).

**S6 — Vector × Slice generalization (prover + example).** Confirm a whole-vector `Slice<Vector<f32,4>>`
elementwise kernel proves (line-granular origin access) and its twin (`&[Line<f32,4>]`) is bit-exact,
demonstrating the two #-1/#-2 gaps compose. *Verify*: bit-exact, bounds `Proved`, reinterpret variant
`OutOfSubset`.

B (S1) is confirmation-only and first because it de-risks the whole design; A (S2–S6) is the real
work and is macro/twin-side.

---

## 12. Open risks, ranked (pre-registered round-9 targets)

1. **Slice-validity vs the transparent prover (high).** The prover proves the *origin* access
   `offset+i < origin_len`; it does **not** verify `end ≤ origin_len` at creation, because the IR of
   `arr.slice(a,b)[i]` is indistinguishable from `arr[a+i]` (§2.1). Soundness rests on the twin
   panicking on an invalid `&arr[a..b]` (§4.4). **Attack surface**: round-9 hands a kernel whose slice
   is created out of the origin's bounds but whose *accesses* are individually guarded so the prover
   proves them — the honest answer is that the `tested` twin panics (invalid slice caught there), and
   the `proved` claim is over the accesses only; the two claims' scopes must be documented so neither
   over-claims. Mitigation: the §5.3 decision + the twin-validity §4.4 argument + an explicit test that
   an over-long slice makes the twin panic while the prover proves the guarded accesses. **This will be
   probed** — the negative control (twin panic) is mandatory.

2. **Mutable-aliasing oracle completeness (high).** §4.3 leans on Rust's borrow checker to reject
   overlapping-live mutable slices. A kernel that defeats the oracle — e.g. two `slice_mut` views
   whose overlap is only realized through an index computed at runtime, or an `as_mut_unchecked`
   round-trip that launders a `ReadOnly` slice into `ReadWrite` — could hold aliased mutable views the
   twin does *not* reject. **Attack surface**: round-9 hands an aliased mutable-slice write the borrow
   checker misses. Mitigation: `as_mut_unchecked`/`downcast*` are rejected outright (§8.3); a runtime-
   computed overlap still lowers to two `UncheckedIndexAssign` on the same origin, so the *bounds*
   claim is unaffected (both writes are individually bounds-checked), and the cross-write *ordering*
   is a data-race concern that only arises cooperatively (deferred, §10). The v1 claim is bounds +
   differential, neither of which an in-thread overlap unsound-ifies; document that v1 does not claim
   write-ordering determinism for overlapping mutable slices.

3. **Reinterpret-slice masquerading as in-subset (medium).** The only `vector_size≠0` source is
   reinterpret-slice (§2.6). If a kernel reaches the prover with a reinterpret access the macro didn't
   catch, `check_trivial_vectorization` must reject it (validated: `OutOfSubset` with the exact
   message). **Attack surface**: a kernel that reads an `Array<f32>` as `Vector<f32,4>` via
   `into_vectorized().with_vector_size` — must be `OutOfSubset`, and the offset-rescaling `mul`/`div`
   the reinterpret emits (`base.rs:104-116`) must not be mistaken for a valid bound. Mitigation: the
   assertion is retained and the reinterpret probe is the standing negative control; separately, the
   construct is unrunnable on wgpu, so it can never appear in a passing differential kernel.

4. **`View`/`Slice` count over-claim (medium, non-soundness).** The "128" is ~25 real core-slice
   creators + a `ReadOnly`/`ReadWrite`-ident tail + the `View` machinery (§3), and even the core-slice
   items are co-gated (§6.3). Claiming v1 "unlocks the 128" would be dishonest. **Attack surface**:
   round-9 asks "which whole launch entry-points does Slice v1 verify end-to-end?" — the honest answer
   is *the slice-carrying elementwise/windowed class + the generalized shortlist + the Vector readers*,
   not matmul/reduce launch sites (those need `plane_*` + `match` + `comptime!` + custom structs).
   Mitigation: §3/§6.3/§12 frame it exactly; the evidence names the co-gates.

5. **`slice.len()` line-vs-origin-length confusion (medium).** `slice.len()` is `end−start`, not
   `Metadata::Length(origin)` (§2.3). A future change that keyed a slice access's bound off
   `slice.len()` instead of the origin's length would be unsound (the slice length can exceed what the
   origin can service if the slice is invalid). Mitigation: the walker keys the obligation off the
   origin's own `Metadata::Length` leaf (§5.1) — never `slice.len()` — and never `BufferLength`; a
   test pins that `to_slice()` binds `Metadata::Length(origin)` and `slice(a,b)` binds the origin
   length, not the `Sub`.

6. **SharedMemory-slice cooperative interaction (medium).** A `shared.slice(a,b)[i]` bounds-proves
   (§10), but the two-thread race walk over shared-slice *writes* is unvalidated, and the offset `a`
   composing with `UnitPos`-derived indices could interact with the modular `AbsolutePos`
   recomposition (`design-shared-memory.md`). Mitigation: race-through-shared-slice is explicitly
   deferred to a cooperative-slice milestone (the reduction target needs `plane_*`/`Atomic`/`match`
   anyway); v1 accepts shared-slice *bounds* only, and a race probe over a shared-slice write is the
   deferral's entry criterion.

7. **cubecl upgrade drift (low, standing).** The `Slice` triple, the `origin[offset+index]` lowering,
   the `vector_size` reinterpret field, and the `Array`-origin `UncheckedIndexAssign` are internals; an
   upgrade (a rename, a clamp added at creation, a move of the offset onto the arg) could change them.
   Mitigation: the existing "survives a CubeCL upgrade" health check + the `ir_hash`; the §2 probes are
   the schema tripwire to re-run on upgrade.

---

## 13. Roadmap impact

- **Confirms and sharpens** the survey's #2-gap recommendation
  (`ecosystem-survey-2026-07.md` §4, `design-line-vector.md` §12/§13): the whole-kernel value of
  Vector is realized **with** the addressing view its readers index through — and this design delivers
  the **core `Slice`** half as a **near-zero-prover-cost** milestone (B is confirmation-only; the work
  is the twin/macro), while **correcting the scope** by splitting the tractable core `Slice` from the
  intractable `View`/`VirtualLayout` machinery the survey's regex folded together.
- **Corrects the coverage framing**: core `Slice` is necessary but rarely sufficient (§3, §6.3) — it
  generalizes the elementwise/windowed shortlist and unblocks the Vector readers, but whole matmul/
  reduce launch sites stay gated behind `plane_*` + `match` + `comptime!` + custom cube structs.
- **The honest post-Slice frontier ranking** (from the gate histograms, §3):
  1. **`plane_*`** (warp/subgroup reductions — `plane_sum`/`plane_broadcast`, ~89 items) — the #3 real
     gate and the next whole-kernel blocker for reductions.
  2. **`CubeType`-arg / custom cube structs** (`RuntimeCell`, `RowWise`, `Accumulator`, matmul tiles,
     ~68 items) — the abstraction layer matmul/reduce are built from.
  3. **2-D topology** (`CUBE_POS_X/Y`, `ABSOLUTE_POS_X/Y`, ~38 items) — matmul tiling.
  4. `Tensor` (multi-dim strided inputs, ~32) and the `View` machinery (deferred here).
  `match` and `comptime!`-params are already largely supported (`process_switch`; comptime folding),
  so those histogram counts overstate the real remaining block. `Atomic` and `cmma` are specialized,
  low-incidence paths.
- **Does not** need QF_BV, a new claim kind, a new IR construct, or new solver machinery: a slice
  access is the existing QF_LIA origin obligation, proved unmodified. The net **prover** change for v1
  is **zero**; the net **macro** change is a slice gate + a Rust-subslice twin + the mutable-aliasing
  gate — a bounded, well-scoped milestone.
