# Ecosystem survey — tracel-ai's own CubeCL kernel libraries (July 2026)

VeriCL was run against tracel-ai's **open-source** CubeCL kernel libraries — the same
rigor as the private dogfood against a production RF/signal-processing codebase
(`docs/dogfood-2026-07.md`), but on public code, so
findings and construct citations are public with attribution. This survey answers one
strategic question the private dogfood could not: **when VeriCL meets the CubeCL
ecosystem's own kernels (not one private RF codebase), which gap is the real frontier?**

Method: map every `#[cube]`/`#[cube(launch)]` function in the targets against VeriCL's v0
gates, rank the gaps by how many kernels each blocks (the ecosystem-demand ranking), then
run the full VeriCL flow — differential on wgpu/Metal **and** cubecl-cpu, plus every
applicable SMT proof — on the provable-today shortlist. All work was done in a sibling
workspace (`/Users/ryland/code/vericl-ecosystem-survey`); the vericl repo was not modified
except for this document and a `tasks/todo.md` addendum. No commits, no git-state changes.

## 0. Where the kernel libraries actually live (a mapping finding)

The task named `crates/cubecl-random`, `crates/cubecl-reduce`, `crates/cubecl-std`,
`cubecl-matmul`, `cubecl-convolution` as if they were crates in `tracel-ai/cubecl`. At the
version VeriCL pins (`cubecl = "=0.10.0"`), **only `cubecl-std` is in the cubecl repo.** The
`cubecl` 0.10.0 meta-crate re-exports exactly `cubecl-core`, `cubecl-std` (the "standard
library"), the runtimes, IR and runtime — there is no matmul/reduce/random/convolution crate
in the cubecl workspace at v0.9.0, v0.10.0, or `main`.

For the 0.10.0 generation the algorithm kernels live in a **separate repo**,
`tracel-ai/cubek` (published as `cubek` v0.2.0, which pins `cubecl = "0.10.0"` exactly), as
crates `cubek-random`, `cubek-reduce`, `cubek-matmul`, `cubek-convolution`, `cubek-std`,
`cubek-attention`, `cubek-quant`, `cubek-fft`, `cubek-interpolate`. `tracel-ai/burn`
(v0.21.0, pinning cubecl 0.10.0 + cubek 0.2.0) is the tensor framework that consumes both;
`burn-cubecl/src/kernel/{prng,reduce,matmul,conv}` are thin host-side wrappers with **zero
`#[cube]` functions** — they delegate into `cubek`.

So the ecosystem VeriCL 0.10.0 would actually consume is three repos, and the survey targets
resolve to:

| Task target | Real location (matching cubecl 0.10.0) |
|---|---|
| cubecl-random | `cubek v0.2.0` `crates/cubek-random` |
| cubecl-reduce | `cubek v0.2.0` `crates/cubek-reduce` |
| cubecl-std | `cubecl v0.10.0` `crates/cubecl-std` (+ a separate `cubek-std`, matmul-tile infra) |
| cubecl-matmul / cubecl-convolution | `cubek v0.2.0` `crates/cubek-matmul` / `crates/cubek-convolution` |

Sources used (exact refs): `cubecl` tag `v0.10.0` (commit `7cf2037`); `cubek` tag `v0.2.0`
(commit `f91effc`, `cubecl = "0.10.0"`). Solver: z3 4.16.0. GPU: wgpu 29 / WGSL / Metal.

**"Philox" correction.** The task described cubecl-random as "Philox-family". It is not:
cubek-random uses a **combined Tausworthe-88 + LCG** hybrid (`taus_step` ×3 + `lcg_step`,
XOR-combined — the classic Marsaglia/NVIDIA GPU generator), not a Philox counter cipher.
This does not weaken the "VeriCL has proven this shape before" premise — the Tausworthe/LCG
steps are pure `u32` bit-ops and a wrapping multiply, exactly the shape VeriCL proved with
`xorshift_step` and `mix_u32`.

## 1. The gap map

Inventory over the target crates, classified per `#[cube]` device item (fn / trait-impl /
trait-decl) by the VeriCL gate it trips. An item can trip several gates; the histogram counts
item×gate incidences. `#[cube(comptime)]` host-only functions (pure compile-time helpers, not
device code) are excluded from the device count.

**Denominator.** 464 device `#[cube]` items across the six target crates (172 are `fn` items;
the rest are trait/impl blocks — Layout impls, ReduceInstruction impls, etc.). Launch
entry-points are strikingly few: cubek-random **1**, cubek-reduce 2, cubek-matmul 4,
cubecl-std 16 (10 in tests), cubek-std **0**, cubek-convolution **0** (dispatch delegates
elsewhere). The libraries are overwhelmingly generic *device-function machinery* composed at a
handful of maximally-gated generic launch sites — not a catalogue of standalone kernels.

### Ecosystem-wide gate ranking (the demand signal)

| Rank | VeriCL gap | Device items blocked | VeriCL status |
|---:|---|---:|---|
| 1 | **`Line`/`Vector` (vectorized elements)** | 148 | **unsupported** |
| 2 | **`View`/`Slice` (tensor views, `ReadWrite`)** | 128 | **unsupported** |
| 3 | `comptime!{…}` blocks / `comptime_type!` | 120 | unsupported (distinct from `#[comptime]` *params*, which ARE supported) |
| 4 | `match` / `Switch` | 119 | unsupported |
| 5 | `plane_*` (plane/warp ops) | 88 | unsupported |
| 6 | rejected `Float`/`Numeric` methods (`cast_from`, `mul_hi`, …) | 82 | unsupported (macro-rejected, whitelist) |
| 7 | custom `CubeType` struct params (`Accumulator`, `RowWise`, `FastDivmod`, …) | 68 | unsupported |
| 8 | cmma / `Matrix` / MMA fragments (tensor cores) | 62 | unsupported |
| 9 | 2-D / multi-axis topology (`new_2d`, `*_POS_X/Y`) | 39 | unsupported |
| 10 | `Tensor<…>` | 32 | unsupported |
| 11 | `SharedMemory` | 24 | supported **only** in the 1-D cooperative *scalar* reduction subset |
| 12 | `select()` | 9 | unsupported |
| 13 | `Atomic` | 1 | unsupported |

### Per-crate character

- **cubek-random** (12 device fns, 1 launch): the RNG **cores are scalar and clean**; the
  wrapper is fully gated. 5 scalar `u32` step functions (`taus_step`, `taus_step_0/1/2`,
  `lcg_step`) trip **no** gate. The two `u32→f32` converters trip `cast_from`. Everything
  above them (`prng_kernel`, the three `PrngRuntime::inner_loop` impls) is `Vector` + `View` +
  generic `<E: Numeric, N: Size>` + trait-associated dispatch — the full stack.
- **cubek-reduce** (63 device items, 2 launch): a trait-composed `ReduceInstruction<P>`
  framework. The tree-reduction *shape* matches VeriCL's cooperative subset, but the element
  type is `SharedMemory<Vector<T,N>>` (vectorized, not scalar `f32`), the cross-cube combine is
  `Atomic::fetch_add`, inputs are `View`/`Tensor`, reduce steps `match` on a comptime enum, and
  everything is generic. `plane_*` (21), `CubeType`-args (33), `comptime!`-blocks (26) dominate.
- **cubecl-std** (76 device items, 16 launch incl. tests): tensor layout / view / coordinate
  infrastructure (generic + `Line`/`Slice`/`Tensor`), plus a few genuinely scalar utilities —
  `to_degrees`/`to_radians` (trigonometry) and `shift_right` (swizzle) are clean; `FastDivmod`
  is `match` + `mul_hi` + generic.
- **cubek-std** (117 device items, **0** launch): pure matmul-tile infrastructure — cmma/MMA
  fragments, plane-vector tiles, softmax over `WhiteboxFragment`. **Not** general utilities; 59
  items trip cmma/Matrix. Zero annotatable kernels.
- **cubek-matmul** (166 device items, 4 launch) / **cubek-convolution** (30 device items, 0
  launch): heavy-gate crates, as expected — `Line`/`Vector` + `comptime!` + `match` + `plane_*`
  + `View` everywhere. The only near-clean `fn`s are index-decode helpers
  (`cube_pos_to_m_n_batch`, `div_mod_seq`), but they operate on comptime structs
  (`CubeMapping`, `Sequence<FastDivmod>`), not plain arrays/scalars, so they are not
  annotatable. (Counted, not fought, per the brief.)

### The headline: the frontier flipped relative to the private codebase

The private dogfood found **zero** uses of `Line`/`Vector`, `Slice`, `plane_*`,
`Atomic`, `Tensor`, and explicitly *withdrew* Tensor/2-D roadmap speculation as demand-driven
scoping. tracel-ai's own libraries are the **mirror image**: `Line`/`Vector` is the single
most common gate (148), `View`/`Slice` second (128), `plane_*` and `Tensor` and cmma all
heavily present. The two codebases disagree about the frontier because they occupy different
layers: the private codebase writes 1-D application kernels over scalar arrays; cubek/cubecl write the
*vectorized tensor-algebra layer* underneath a framework. **For ecosystem reach, the next
frontier is `Line`/`Vector` (+ `View`/`Slice`) support — not the 1-D scalar expansions the
private survey implied.** See §4.

## 2. The shortlist — provable today

Eight kernels are annotatable within the current subset; all landed the full **tested**
(differential) + **proved** (`smt-oob-freedom`) pair, on **two** differential lanes
(wgpu/WGSL/Metal and cubecl-cpu). Device-function *bodies* are copied verbatim from the cited
upstream source (MIT/Apache-2.0); the thin `*_map` launch drivers are survey glue (a 1-D
elementwise driver, the same shape as VeriCL's own `xorshift_step`), and the VeriCL contracts
are ours. Evidence: `vericl-ecosystem-survey/annotated/evidence/vericl.json`.

| Kernel (upstream body) | Source (crate/file:line, v0.2.0 / v0.10.0) | VeriCL features exercised | compare | Differential (wgpu+cpu) | Proved bounds |
|---|---|---|---|---|---|
| `taus_step_0/1/2` via `taus0/1/2_map` | cubek-random `base.rs:157-179` | **composition** (`uses`), helper-calling-helper | `exact` | PASS ×7 sizes | `Proved{2}` |
| `lcg_step` via `lcg_map` (inlined) | cubek-random `base.rs:181-187` | **wrapping** (u32 `z*a+b`) | `exact` | PASS ×7 | `Proved{2}` |
| `combined_taus_lcg` | cubek-random `uniform.rs:48-53` (per-value core) | **wrapping + composition together** | `exact` | PASS ×7 | `Proved{5}` |
| `to_degrees` via `to_degrees_map` | cubecl-std `trigonometry.rs:19-22` | **generic** (`instantiate(F=f32)`) + composition | `abs=1e-2` | PASS ×7 | `Proved{2}` |
| `to_radians` via `to_radians_map` | cubecl-std `trigonometry.rs:33-36` | generic + composition | `abs=1e-5` | PASS ×7 | `Proved{2}` |
| `shift_right` via `shift_right_map` | cubecl-std `swizzle.rs:102-109` | **`#[comptime]` param** (pass-through bool) | `exact` | PASS ×7 | `Proved{2}` |

Each entry carries a stable `source_hash` + `ir_hash` in the evidence manifest, the twin
reference recorded as "vericl-macros sequential twin", and the cpu lane recorded (honestly) as
**not** front-end-independent. Every proof is z3 4.16.0 / QF_LIA.

Notes on the individual results:

- **The RNG core is the headline.** cubek-random's reusable numeric heart — the Tausworthe
  LFSR step (shared, then specialized ×3) and the LCG step — proves in bounds and matches
  wgpu/Metal *and* cubecl-cpu bit-for-bit. The LFSR steps compose exactly the way cubek itself
  structures them (`taus_step_0` calls `taus_step`), so VeriCL's `#[vericl::helper]` +
  `uses(...)` mechanism models real upstream composition unchanged. `combined_taus_lcg`
  reassembles cubek's full per-value output (`taus0(s0) ^ taus1(s1) ^ taus2(s2) ^ lcg(s3)`,
  `uniform.rs:48-53`) and confirms that **`wrapping` and `uses(...)` co-exist in one kernel** —
  the wrapping fold rewrites the inline LCG's `*`/`+` while leaving the composed helper calls
  and the XOR untouched.
- **`to_degrees`/`to_radians`** exercise the generic path on real cubecl-std code: `F: Float`
  monomorphized at `f32`, body `val * F::new(const)`, using only the whitelisted `F::new`.
  Tolerances are derived honestly from the declared input range (a single multiply by a
  constant, no fma contraction possible: `|val| ≤ 1000 ⇒` one f32 rounding covered with margin).
  In practice both matched both backends exactly; the tolerance is what is *guaranteed*.
- **`shift_right`** is the smallest win: a scalar `u32` helper whose `#[comptime] bool` selects
  the shift direction, pinned via the caller passing the literal (no helper `instantiate`
  needed — a comptime param carries no host-callability hazard). The `shift < 32` assume is
  load-bearing (Rust `>>` panics at ≥32; WGSL masks) and doubles as the generation bound.

## 3. Findings, classified

Per the survey standard, every finding is classified — real upstream bug / implicit-invariant
(undocumented contract) / VeriCL gap — and not over-claimed.

### 3a. VeriCL gaps (the frontier signals)

- **`Line`/`Vector` + `View`/`Slice` is the dominant ecosystem gap** (148 + 128 items). This is
  the single most important survey output — see §1 headline and §4.
- **`cast_from` blocks cubek-random's `u32→f32` converters.** `to_unit_interval_closed_open` /
  `_open` (cubek-random `base.rs:191-206`) — the functions that turn the RNG's `u32` output into
  a float in `[0,1)` — use `f32::cast_from`, on VeriCL's `FLOAT_METHOD_REJECT` list.
  Annotating one is a clean macro-time rejection (verified on the real body):
  > `error: host-callability of 'F::cast_from' in the reference twin is unverified — outside the
  > vericl v0 subset; verified host-callable Float/Numeric methods are: new, from_int, …`

  This is the exact seam between what VeriCL proves and what it cannot: the integer generator
  core proves; the float-conversion boundary is out of subset. A `usize/u32 → F` numeric cast is
  pervasive in this ecosystem (part of the 82 rejected-method incidences), and a verified
  host-safe `cast_from`/`from_int`-for-runtime story would unlock the distribution kernels' scalar
  cores (`Uniform`/`Normal`/`Bernoulli` `inner_loop`) once `Vector`/`View` are also handled.
- **`wrapping` is kernel-only, so cubek's `lcg_step` cannot be a composed helper.** `lcg_step`'s
  `z*a+b` is wrap-on-overflow by intent (cubek even annotates the analogous thread-seed line
  `#[allow(arithmetic_overflow)]`, `base.rs:135`). `#[vericl::helper]` rejects the `wrapping`
  clause (it is a kernel-only contract), so a helper twin for `lcg_step` computes checked
  arithmetic and panics on overflow — demonstrated as a negative control (§3d, item 2). The
  faithful path is to inline `lcg_step`'s body into a `wrapping` kernel, which is what `lcg_map`
  and `combined_taus_lcg` do. **Residual:** a `wrapping` (or per-method wrap) capability on
  `#[vericl::helper]` would let the LFSR/LCG steps be modeled compositionally end-to-end, matching
  how cubek factors them. Low urgency (the inline path is faithful and proves), but a real
  expressiveness gap surfaced on real code.

### 3b. Implicit-invariant findings

None in the shortlist: the RNG steps and trig helpers have no caller-maintained bounds
invariant — every access is a guarded `ABSOLUTE_POS` read/write and the proofs discharge from
the stated `len` assumes alone.

One implicit-invariant *observation* outside the annotatable set, worth recording: cubek-reduce's
`shared_sum` (`routines/shared_sum.rs:22-27`) documents in prose that *"This doesn't set the
value of output to 0 before computing… It is the responsibility of the caller"* — a genuine
undocumented-in-the-type caller obligation, the same "boundary behavior can be implicit" class
the private dogfood hit. It is not annotatable today (the kernel is `Atomic` + `Vector` +
`View` + generic), so this is a note, not a proof.

### 3c. Real upstream bugs

**None found.** The kernels in the annotatable shortlist are correct: bit-exact across two
backends and provably in-bounds. This is the honest, expected result for mature library code —
recorded so the survey's discrimination claims (§3d) are not mistaken for an absence of testing.

### 3d. Negative controls (discrimination proven)

Two deliberately-defective variants of the shortlist, plus the positive control, confirm the
checks discriminate rather than rubber-stamp (`annotated/src/bin/negatives.rs`, exit 0 = all
caught):

1. **Bounds refutation.** `lcg_map_oob` (a `<=` off-by-one guard) → `Refuted`, counterexample
   `abs_pos == len` (position at the boundary). The honest `lcg_map` `Proved{2}` on the same
   run — discrimination in both directions, so the `smt-oob-freedom` proofs above are not
   vacuous.
2. **Wrapping necessity.** `lcg_map_nowrap` (`lcg_step` body without the `wrapping` clause) →
   the checked reference twin panics (`attempt to multiply with overflow`) on every size, caught
   deterministically by the differential lane. Confirms the `wrapping` finding (§3a) is real:
   cubek's wrap-on-overflow LCG is unfaithful to a checked twin.

Macro-gate rejections were also verified on real upstream bodies: `to_unit_interval_closed_open`
(`cast_from`, §3a) and a `Vector`-element array kernel (`error: gen(...) v0 only supports
f32/f64/u32/i32/u64/i64 array elements; … Array<Line<u32>> is outside that set`) — both rejected
cleanly at macro time with actionable messages, never silently approximated.

## 4. Recommendation: the next frontier is `Line`/`Vector` (with `View`)

The private dogfood's demand-driven scoping was correct *for that codebase* and drove the right
milestones (generics, composition, div/mod, cooperative reductions) — all of which this survey
confirms landing cleanly on public code (composition and `instantiate` in particular ran on real
cubek/cubecl bodies with zero adaptation). But the ecosystem's own libraries send a different,
unambiguous demand signal:

- **`Line`/`Vector`: 148 device items** — #1 gap, and the element type of essentially every
  cubek-random distribution kernel, every cubek-reduce reduction, and the cubek-matmul/std tiles.
  VeriCL's cooperative reduction support already has the right *shape* (tree reduction, tid==0
  store) but is pinned to scalar `SharedMemory<f32>`; the real reductions are
  `SharedMemory<Vector<T,N>>`. A `Line<T,N>` element model — twin as a length-`N` lane array,
  bounds obligations over the *outer* index, per-lane differential compare — is the single change
  that would move the most ecosystem kernels from OutOfSubset toward analyzable.
- **`View`/`Slice`: 128 device items** — #2 gap, the input/output abstraction over which those
  vectorized kernels index. `Vector` without `View` reaches the scalar cores but not the launch
  entry-points (which take `LinearView<Vector<…>>`); the two together are what unlock a whole
  kernel rather than its numeric heart.
- A distant-but-real third for the RNG family specifically: a verified **runtime `cast_from` /
  `from_int`** host-safety story (§3a), which — combined with `Vector` — would take the
  `Uniform`/`Normal`/`Bernoulli` `inner_loop` bodies from fully-gated to shortlist.

Everything below that (`plane_*`, cmma, `Tensor`, 2-D topology, `match`, `comptime!` blocks)
is matmul/attention-tier machinery — large, hard, and lower leverage per unit of subset work
than `Line`/`Vector`. Recommend **`Line`/`Vector` element support as the next milestone**, scoped
first to the 1-D vectorized elementwise + reduction shapes (where VeriCL already has the topology
and proof machinery), with `View`/`Slice` as the immediate follow-on. That is the change that
converts "VeriCL proves the reusable scalar cores of tracel-ai's kernels" into "VeriCL proves
tracel-ai's kernels".

## Appendix — reproduction

- Workspace: `/Users/ryland/code/vericl-ecosystem-survey` (`cubecl@v0.10.0`, `cubek@v0.2.0`,
  `burn@v0.21.0` blobless; `annotated/` the path-dep annotation crate; `classify.py` the gate
  classifier).
- Full flow: `cd annotated && VERICL_UPDATE=1 cargo test --features cpu --test conformance`
  (writes evidence, both lanes), then `cargo test --features cpu --test conformance` (verifies).
- Negative controls: `cargo run --bin negatives` (exit 0 = all defects caught).
- Gate map: `python3 classify.py`.

---

**Update (2026-07-23, post-survey):** §3a residuals 2 and 3 are closed. Quick-wins batch 2
added verified `cast_from`/`mul_hi` host shims (GPU-ground-truth-verified bit-exact on both
wgpu/Metal and cubecl-cpu — u32/i32→f32 is round-to-nearest-even everywhere, matching Rust
`as f32`; no backend divergence) and helper-level `wrapping`. Re-validated against the
verbatim cubek shapes in the survey workspace: `combined_taus_lcg` recomposed with
`lcg_step` as a wrapping helper (Proved{5}, bit-exact both lanes) and
`to_unit_interval_closed_open` with its verbatim `f32::cast_from` body (0-ULP, Proved{2}).
The dominant remaining ecosystem gaps are unchanged: Line/Vector, View/Slice, and
struct-typed comptime params (the majority shape among the 120 comptime! incidences).

---

## Re-census (2026-07-24) — the 464 items re-scored against today's gates

Re-run of this survey against today's VeriCL, to measure what the `Vector`, core-`Slice`,
`match`/Switch, `comptime!`-block and `cast_from`/`mul_hi`-shim milestones actually bought. The
private-codebase half is in `docs/dogfood-2026-07.md`'s own re-census addendum. Classifier:
`vericl-ecosystem-survey/classify.py`, updated in place with the v0 gate list retained verbatim as
`GATES_V0`; both gate sets run on the same item-capture path so the diff is meaningful.

**Denominator and baseline re-verified before any gate change**: 464 device `#[cube]` items (172
`fn`, 292 impl/trait; 153 `#[cube(comptime)]` host fns excluded), every per-crate and per-launch
count matching §1, and `GATES_V0` reproducing all thirteen recorded counts exactly — 148 / 128 / 120
/ 119 / 88 / 82 / 68 / 62 / 39 / 32 / 24 / 9 / 1, zero drift. Two independent implementations of the
classifier were run and cross-checked; where they disagree, the range is given rather than a
falsely precise number.

### Before → after

| v0 gate | v0 | Today |
|---|---:|---|
| `Line`/`Vector` | 148 | **43 blocking** (still-out shapes) + a supported-class band of 2 (lower) to 91 (upper); the old regex was also carrying 25 core-slice creators and 49 `line_size`/`vector_size` plumbing items |
| `View`/`Slice` | 128 | **110** View/Layout machinery + **21** `Slice`/`SliceMut` type ident + **18** reinterpret; ~70 of the `ReadOnly`/`ReadWrite` mentions are marker-only and **not** a gate |
| `comptime!{}` blocks / `comptime_type!` | 120 | **71** out-of-subset `comptime!{}` + **53** `comptime_type!`; only **12** are lexically admissible |
| `match` / Switch | 119 | **0 blocking — supported** |
| `plane_*` | 88 | 88 |
| rejected Float/Numeric methods | 82 | **5** residual + **6** non-`f32`-target `cast_from`; 7 shimmed; 44 generic-target, fate decided by the author's `instantiate(...)` |
| custom `CubeType` struct params | 68 | 68 on the original name list; **141** on a broad parameter-type check |
| cmma 62 · 2-D 39 · `Tensor` 32 · `SharedMemory` 24 · `select()` 9 · `Atomic` 1 | | unchanged |
| *(never measured at v0)* | — | **struct-typed `#[comptime]` param 243** · broad `CubeType` param 141 · `intrinsic!` 7 |

### Sole-blocker counts — the honest reach number

The v0 survey ranked gaps by *incidence*. Incidence is the wrong number for "what would removing
this unlock", because an item usually trips several gates. The re-census adds the sole-blocker count
(items where the gate is the only blocking gate tripped); 127 of 464 items have exactly one.

| Gate | Items | **Sole** | Sole non-test `fn` |
|---|---:|---:|---:|
| struct-typed `#[comptime]` param | 243 | **38** | 0 |
| `View`/`Layout` machinery | 110 | **45** | 0 |
| custom `CubeType` param (broad) | 141 | 8 | **8** |
| `CubeType`-arg (v0 name list) | 68 | 8 | 1 |
| cmma / `Matrix` | 62 | 6 | 0 |
| `comptime_type!` | 53 | 4 | 0 |
| `SharedMemory` | 24 | 4 | 0 |
| `Slice`/`SliceMut` type ident | 21 | 4 | 0 |
| **`plane_*`** | **88** | **2** | 2 |
| `comptime!{}` out of subset | 71 | 2 | 0 |
| `Vector` shape still out of subset | 43 | 2 | 0 |
| 2-D / multi-axis topology | 39 | **1** | 0 |
| `Tensor` 32 · reinterpret 18 · `select()` 9 · residual rejected methods 5 · `Atomic` 1 | | **0 each** | 0 |

### Two gates that unlock nothing here — measured, not estimated

**`match`/Switch (v0 rank #4, 119 items) unlocks zero ecosystem items.** Of ~270 `match`
expressions inside `#[cube]` item spans, **exactly 4 have integer-literal arms, and all 4 are in
cubecl's own language-conformance suite** (`cubecl-core/src/runtime_tests/branch.rs`) — zero in
cubek's nine crates, zero in cubecl-std, zero in burn-cubecl. An independent scan of the six target
crates for an integer-literal match arm inside any `#[cube]` device item returned 0 hits. The rest
scrutinise a comptime-typed enum (`MatrixLayout`, `StageIdent`, `ConvolutionOperation`, …, the
largest bucket), a `CubeType` enum (`FastDivmod`'s `match self`, the second largest), or
`Option`/`bool`. `Branch::Switch` modeling is correct and useful; **this corpus does not exercise
it.**

**`comptime!{}` blocks unlock zero on their own.** All 12 lexically admissible items are blocked by
something else — 9 of the 12 by a struct-typed `#[comptime]` param, exactly as the 2026-07-23 update
predicted. That shape is now measured: **243 of 464 items, the single largest blocking gate in the
corpus.**

Core `Slice` is the same story, and consistent with the Slice milestone's own "necessary but rarely
sufficient": of ~84 slice-creation sites inside `#[cube]` items, every one in cubek/cubecl-std
production code is on a `View`/`LinearView`/`StridedStageMemory`, on launch machinery, or internal to
an abstraction. The bare-`Array`/`SharedMemory` calls in plain `#[cube]` fns are all in cubecl's
conformance suite.

### Items within reach

| Bucket | v0 | Today | Δ |
|---|---:|---:|---:|
| Items with zero blocking gates (v0-lineage gate set) | 103 | **119** | **+16** |
| …of which non-test `fn` | 18 | **41** | **+23** |
| Items with zero blocking gates (corrected gate set, applied to both sides) | 35 | **49** | **+14** |
| …of which non-test `fn` | 10 | 13 | +3 |
| Genuinely new annotatable non-test plain `fn`s (blocked at v0, clear today) | — | **4** | all four via the `cast_from` shim |

Both rows are reported because the v0 gate list **under-counted blockers** — it never looked for
struct-typed comptime params, the broad `CubeType` parameter shape, or `intrinsic!`. That is why the
v0 survey showed 103 gate-free items yet could shortlist only 8. Under the corrected set applied
symmetrically, the honest movement is 35 → 49.

### Spot-validation — §4 residual #3 is closed, measured not predicted

§4 predicted that a verified runtime `cast_from` "would take the `Uniform`/`Normal`/`Bernoulli`
`inner_loop` bodies from fully-gated to shortlist". Two of the three now carry the full
tested+proved pair on **both** differential lanes; the third produced a clean, actionable rejection.
Six new kernels, added non-destructively to the survey crate (the original 8 evidence entries are
byte-identical, verified by per-entry canonical-JSON SHA-256):

| Kernel (upstream body) | Source | compare | wgpu/Metal | cubecl-cpu | Proved |
|---|---|---|---|---|---|
| `to_unit_interval_open_map` | cubek-random `base.rs:199-206` | `max_ulp = 3` | PASS | PASS | `Proved{2}` |
| `uniform_value_map` | cubek-random `uniform.rs` per-value core | `abs = 1e-4` | PASS | PASS | `Proved{5}` |
| `normal_box_muller_map` | cubek-random `normal.rs:65-70` | `abs = 1e-2` | PASS | PASS | `Proved{6}` |
| `bernoulli_value_map` | cubek-random `bernoulli.rs` per-value core | — | **rejected at compile time** | — | — |
| `kernel_switch_simple` | cubecl-core `runtime_tests/branch.rs` | `max_ulp = 0` | PASS | PASS | `Proved{3}` |
| `slice_select` | cubecl-core `runtime_tests/slice.rs` | `max_ulp = 0` | PASS | PASS | `Proved{2}` |

- **`uniform_value_map` is the composite result**: three `taus_step_*` helpers, `lcg_step` as a
  helper-level `wrapping` helper, the `cast_from` shim (including its f32→f32 identity arm), and the
  affine map — composition + helper `wrapping` + shim in one kernel. Its negative control shows why
  helper-level `wrapping` was *necessary* rather than convenient: a **float** kernel cannot carry the
  integer-only `wrapping` clause, so the survey's original "inline the body into a wrapping kernel"
  workaround is unavailable here. Measured wgpu worst `|e−a| = 7.63e-6` (FMA contraction, ~⅓ of
  elements); cpu bit-exact.
- **`to_unit_interval_open` is not bit-exact, unlike its sibling** — 1 ULP on wgpu, 0 on cpu. The
  mechanism was identified rather than guessed: the backend lowers `/ 8388609.0` to a
  reciprocal-multiply. The sibling divides by `2^24` (exact), this one by `2^23 + 1` (inexact). The
  `cast_from` shim is not implicated.
- **`normal_box_muller_map`'s open-interval domain invariant** is established *structurally*, not by
  the differential lane: verified exhaustively over all 2^23 distinct converter outputs
  (`[1.19e-7, 0.9999999]`, both endpoints excluded). It is load-bearing — the `_closed_open` sibling
  returns exactly `0.0` for its 256 smallest inputs, giving `ln = -inf` and NaN — but at a density of
  256/2^32 per draw the suite's ~144k draws would hit it 0.0086 times in expectation. Recorded as a
  known limit of the lane.
- **`bernoulli_value_map` rejection** is `bool: CastToF32 is not satisfied` — the shim rewrite fires
  (the target segment *is* `f32`), and the closed `CastToF32` source set rejects at the twin's own
  call-site span. Loud and actionable, but rustc-mediated rather than a VeriCL-authored message.
- **New residual: bool-source `cast_from`.** GPU ground truth already measured (`true → 1.0`,
  `false → 0.0`, bit-identical on wgpu/Metal *and* cubecl-cpu), so the fix is one trait impl per
  target type. Not a Bernoulli one-off: **15 bool-source `cast_from` sites** across cubek — branchless
  masking/predication in cubek-std softmax/rowwise/whitebox, cubek-matmul/convolution async-copy,
  cubek-reduce, cubek-quant. Most target `u32`/`i32`/a generic `E`, so a full fix needs
  bool→{u32,i32,f32}.
- **`slice_select` surfaced a mis-attribution**: the upstream slice tests are not blocked by `Slice`
  at all. All five use the single-designated-thread idiom `if UNIT_POS == 0`, and the measured
  rejection names `UNIT_POS` and says nothing about `.slice(2, 3)`. That idiom — not slicing — is the
  gate, and it is one VeriCL could plausibly accept as a 1-D special case.
- Discrimination re-proven on the new claims (`annotated/src/bin/recensus_negatives.rs`, exit 0): an
  off-by-one guard on the 5-buffer composition is `Refuted` with a counterexample while the three
  honest kernels `Proved{2}/{5}/{6}` on the same run, and a non-wrapping `lcg_step` makes the twin
  panic at every size.

### The post-re-census frontier ranking (measured)

The Slice milestone's recorded frontier ranking was (1) `plane_*`, (2) `CubeType`-arg, (3) 2-D
topology, (4) `Tensor` + `View`. **The sole-blocker measurement overturns it.**

1. **struct-typed `#[comptime]` params / `comptime_type!`** — 243 items, 38 sole-blocker. The
   single largest blocking gate in the corpus, and the thing standing behind the `comptime!`-block
   support already shipped.
2. **`View`/`Layout` machinery** — 110 items, **45 sole-blocker**, the largest single-gate unlock.
3. **custom `CubeType` struct args** — 141 items broad, 8 sole — and those 8 are the **only**
   sole-blocker bucket that is plain non-test `fn`s rather than impl/trait items.
4. cmma / `Matrix` — 62 items, 6 sole.
5. **`plane_*` — 88 items but only 2 sole-blocker.** The #2 gate by incidence is near the bottom by
   reach. Removing it alone unlocks two functions.
6. **2-D topology — 39 items, 1 sole-blocker.**
7. **`Tensor` (32), reinterpret-slice (18), `select()` (9), the residual rejected methods (5), and
   `Atomic` (1) sole-block *zero* items each.** Removing any of them alone unlocks nothing.

Items 1–3 are one underlying capability — letting a `#[cube]` item take a comptime-known
struct/enum, and letting the twin model a dispatching view. That, not `plane_*`, is where the
ecosystem's remaining mass sits.

### Where the classification is honestly imprecise

- The "all-`Vector` elementwise class" is a whole-signature property regex cannot decide: 133–134
  items name `Vector`/`Line`, 91 avoid the explicitly-deferred shapes (upper bound), 2 trip nothing
  at all (lower bound), and **0** of those are launch entry points.
- The broad `CubeType` parameter check parses `fn` signatures only, so the 292 impl/trait items'
  `&self` receivers go unmeasured — every impl/trait reach number is an over-estimate.
- `comptime!{}` admissibility is a lexical approximation of the macro's real `syn`-level check;
  calibrating it against real bodies moved the admissible count from 22 to 12.
- `cast_from` has three buckets, not two: 7 definitely shimmed, 6 definitely still rejected, and 44
  with a generic target whose fate depends on the author's `instantiate(...)` — reported separately,
  never counted as unlocked.
- Bare free-function calls to undeclared helpers are not gated (that needs whole-crate visibility),
  so a few items read "clean" while calling something blocked; every reach number assumes callees get
  annotated or inlined.
- `SharedMemory` (24) is kept wholly blocking because regex cannot recognise the 1-D cooperative
  scalar subset — conservative, i.e. reach-understating.
- The View pure/mixed split does not reproduce the Slice addendum's ~14 pure + 38 mixed; the *mixed*
  figure lands near 38 in every variant, the *pure* figure is dominated by whether `Coordinates` is in
  the signal set. Reported as measured with the definition stated, not tuned to match.
