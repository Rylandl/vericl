# Atomics milestone — design (July 2026)

The implementable design for VeriCL's **atomic** kernel shape: scatter-add, histograms,
atomic max/min — the highest-*recognizability* PLANNED gap ([M-B](coverage.md#the-gap-closure-plan)).
It is a curious milestone: measured against VeriCL's two corpora it sole-blocks **zero** items
(`Atomic` is 1/464 ecosystem items and 1/22 private dogfood kernels, neither solely blocked —
[ecosystem-survey §](ecosystem-survey-2026-07.md), [dogfood §](dogfood-2026-07.md)), yet a histogram
is one of the first three kernels a working GPU programmer writes. The gap-closure plan ranks by
recognizability on purpose ([coverage.md](coverage.md#the-gap-closure-plan)); this is that plan's
clearest instance.

The milestone lives or dies on one honesty question, and this document makes it the spine:

> **Integer** atomic add/sub/max/min/and/or/xor is **order-independent** — the final value in a bin
> is the same regardless of how the hardware interleaves the contending threads. So the twin can be a
> sequential fold, the differential is **bit-exact**, and the prover may treat the atomic location as
> **race-exempt**. **Float** atomic add/sub is **order-dependent** — fp addition is not associative,
> the GPU result depends on the interleaving, and a sequential twin with one fixed order compares
> against *one of many legal results*. A design that quietly picks the twin's order and calls the
> agreement a pass is exactly the silent-disagreement failure VeriCL exists to prevent.

Everything below marked **measured** was checked against the pinned `cubecl =0.10.0` on this machine
(Apple M3, wgpu/Metal, `SHADER_FLOAT32_ATOMIC` present; z3 4.16.0 on PATH). Probe sources and their
raw output are preserved in the session scratchpad
(`scratchpad/designatomics/{probe,fgpu,fspread,ir_dump,proverprobe}.rs`, `mixed_access.smt2`,
`FINDINGS.txt`). **Clean-room kernels only** — no private source was probed; the one private dogfood
atomic kernel is described by shape only, per the README private-codebase policy.

File:line citations to `crates/vericl-ir/src/prover.rs`, `crates/vericl-macros/src/lib.rs`, and the
`cubecl-ir-0.10.0` / `cubecl-core-0.10.0` / `cubecl-wgpu-0.10.0` source trees are current as of
`8c19539`.

---

## 0. Headline recommendation

1. **v1 = integer scatter-add / histogram / atomic-max-min, global, 1-D, f32-free.** The
   order-independent integer reductions (`fetch_add`, `fetch_sub`, `fetch_max`, `fetch_min`,
   `fetch_and`, `fetch_or`, `fetch_xor`) get the **strong** claim triple: a **bit-exact** differential
   against a derived sequential-fold twin (`tested`), plus **bounds** (`proved`/`smt-oob-freedom`),
   plus **race-exemption** (`proved`/`smt-atomic-race-freedom`, the new check). **Measured
   bit-exact and deterministic on wgpu/Metal** (§9).

2. **Float atomics are rejected from the differential lane in v1, with a measured message.** Float
   atomic add is genuinely nondeterministic on wgpu/Metal — **19/19 runs bit-distinct** in every
   configuration probed, with a run-to-run spread that **grows with contention**: 11 ULP at 4 096
   adds/bin up to 93 ULP at ~1 M adds/bin for benign positive values, and **389 ULP** under
   cancellation (§1.3, §9). No fixed tolerance is sound (the spread is data- and contention-dependent
   and has no a-priori provable bound short of the per-bin population, which is the injectivity gap
   VeriCL does not have — §1.4). The honest claim shape the measurement dictates is **(c) reject from
   the differential**, not (a) a declared tolerance. Float atomics may still be *bounds-* and
   *race-*provable in a later proved-only mode (§8.4), but that is strictly weaker than VeriCL's
   custody standard and is **not** v1.

3. **Bounds come almost for free; the race check is the real work but is small.** The atomic location
   lowers as an ordinary `Operator::Index` into a normal buffer (§3), so the existing bounds walker
   already discharges it: a clean-room atomic **scatter-add proves today with the current prover, zero
   changes, 3 obligations** (§9, `proverprobe.rs`). The two additions are (i) one **sound bounds
   extension** for the `bins[key % n_bins]` idiom (modulo-range for a tainted dividend, §5.2) and (ii)
   the **race-exemption + mixed-access** discipline in the race walker (§6), whose obligation shape is
   validated in QF_LIA (§6.4).

4. **The A↔B coupling is inverted from shared memory, and simpler.** For cooperative kernels the
   *differential* borrows the race proof. For **global** atomics the race property is not even an SMT
   query — it is an **access-discipline** fact (an `Array<Atomic<T>>` can *only* be touched atomically,
   so all-atomic ⟹ race-exempt regardless of the index pattern, §6.1). The two-thread SMT machinery is
   needed only for the **mixed** atomic-vs-plain case, which within cubecl's type system arises only in
   **shared** atomic tiles (the block-histogram shape) — deferred to v1.1 (§8.5).

5. **Do not trust `atomic_type_usage`.** cubecl-wgpu's WGSL backend registers only
   `LoadStore | Add` and reports `MinMax = false` for **every** type (`wgsl.rs:83-113`) — yet integer
   `atomicMax` **compiled and ran and matched the sequential host max exactly** (§9). The feature flag
   under-reports the backend's real capability; v1 must gate on VeriCL's own subset and the pinned
   backend's *measured* behaviour, never on the flag (§4.2, risk 3).

This resolves M-B's stated precondition ("must resolve float-atomic-add ordering honesty first",
[coverage.md](coverage.md#the-gap-closure-plan)): the honesty is resolved by *rejecting* float from the
differential on measured grounds, and shipping the integer reductions at full strength.

---

## 1. The spine — integer order-independence vs float order-dependence (measured)

### 1.1 Why integer atomic reductions are order-independent

An atomic RMW `bin.fetch_op(v)` applied by a set of threads leaves the bin holding
`op(init, v_1, v_2, …, v_k)` for the multiset `{v_i}` the threads contributed. For
`op ∈ {+, −, max, min, &, |, ^}` on a fixed-width integer:

- `+`/`−`: the result is `init + Σ v_i` (subtraction accumulates as `init − Σ subtrahend`). Two's-complement integer addition is associative **and** commutative — it wraps, but wrapping is itself associative — so the sum is independent of order. (WGSL wraps; VeriCL's twin needs the `wrapping` clause, exactly as for every other integer kernel — README "Integer overflow".)
- `max`/`min`: commutative, associative, idempotent — the extreme of a multiset is order-free.
- `&`/`|`/`^`: bitwise, commutative, associative — order-free.

So a **sequential fold in any order — including the twin's `ABSOLUTE_POS = 0,1,2,…` order — reproduces
the GPU result bit-for-bit.** This is the integer analogue of the shared-memory reduction's
"the twin *is* the reduction, re-associated in the same order" (design-shared-memory §4.6), except here
we do not even need the same order: any order agrees.

**Measured (`probe.rs`, `FINDINGS.txt`), wgpu/Metal, M3, 1 048 576 scatters into 256 bins
(~4 096 adds/bin), cube_dim 256:**

- Integer histogram: **bit-identical across 8 runs**; total adds recorded = exactly 1 048 576.
- Integer atomic max: **matches the sequential host `fold(max)` exactly**.

The order-independence is not assumed from the derive — it is observed on hardware.

### 1.2 The two accumulators that are *not* order-independent

`Swap`/`exchange` and `CompareAndSwap` are **order-dependent even for integers**: the final stored value
is the last writer's, which the hardware chooses. Their common use (lock-free update loops) is a
data-dependent unbounded loop anyway. Both are **deferred** (§8.3), not part of the bit-exact set.

### 1.3 Why float atomic add is order-dependent — and by how much (measured)

Floating-point addition is **not associative**: `(a+b)+c ≠ a+(b+c)` in general. A float atomic-add bin
accumulates its summands in whatever order the hardware schedules the contending threads, so different
runs give different f32 results.

**This is not hypothetical on VeriCL's target backend.** f32 atomic add runs on M3/Metal
(`SHADER_FLOAT32_ATOMIC` is exposed; `wgsl.rs:108-113`). Measured (`fgpu.rs`, 20 runs each, 1 048 576
scatters):

| distribution | adds/bin | runs bit-distinct | ULP spread | abs spread | rel spread |
|---|---:|---:|---:|---:|---:|
| positive O(1) `[0.1,1.1)` | 4 096 | 19/19 | 11 | 0.0027 | 1.1e-6 |
| positive O(1) | 65 536 | 19/19 | 27 | 0.105 | 2.7e-6 |
| positive O(1) | 1 048 576 | 19/19 | 93 | 5.81 | 9.2e-6 |
| signed, near-cancelling `±1` | 4 096 | 19/19 | 29 | 0.00022 | 3.1e-6 |
| signed, near-cancelling | 1 048 576 | 19/19 | **389** | 0.0119 | 2.8e-5 |
| wide dynamic range `[1e-3,1e6]` | 4 096 | 19/19 | 9 | **144** | 7.8e-7 |
| wide dynamic range | 1 048 576 | 19/19 | 33 | **135 168** | 2.7e-6 |

Every configuration is nondeterministic (19/19 distinct). The spread **grows with contention**
(adds/bin) and its character depends on the value distribution: modest ULPs but enormous *absolute*
spread for wide-dynamic-range data (max error vs an f64 reference reached ~40 M in the 1-bin case),
hundreds of ULPs under cancellation. A backend-independent cross-check with a full random-permutation
model (`fspread.rs`) is *worse* (it is the adversarial order the hardware does not quite reach): 58 ULP
at 4 096, up to 429 ULP at 262 144 for positive O(1), 652 ULP at 65 536 under cancellation. The GPU is
milder than fully-random because atomic contention is partially serialized by warp/threadgroup, but it
is emphatically **not** bit-stable.

### 1.4 Why no sound tolerance exists (the fma precedent, inverted)

The fma finding (round 10, [coverage.md](coverage.md#rng--hash--bit-mixing)) succeeded because the
divergence domain was *characterizable*: a bounded flush-to-zero region the local model reproduces
exactly. Float atomic add is the opposite — its divergence is **unbounded a priori**:

- The standard forward-error bound for summing `n` values each ≤ `M` is `|computed − exact| ≤
  (n−1)·ε·Σ|x_i|`, so the spread between two orders is `≤ 2·(n−1)·ε·Σ|x_i|` per bin.
- A **sound** declared tolerance must hold for *every* input the assumes admit. Without a bound on the
  **per-bin population** `n`, the worst case is *all* `N` scatters landing in one bin, giving
  `n = N` and `Σ|x_i| ≤ N·M` — a bound of `2·(N−1)·ε·N·M`, **quadratic in N**. At `N = 10^6`,
  `ε ≈ 6e-8`, `M = 1` that is ≈ `1.2e8`: uselessly loose, and it would mask any real bug.
- A **tight** tolerance would need the per-bin population — which is the scatter *distribution*, i.e.
  the injectivity/permutation fact VeriCL explicitly does **not** have (the scatter-correctness gap,
  "recorded, measured, not scheduled" — [coverage.md](coverage.md#explicitly-out)).

So the choice among the task's four options is forced by measurement:

- **(a) declared tolerance** — rejected: no sound tight bound exists; the only provable bound is
  quadratic and useless. Asserting a tolerance I did not measure to hold universally would be the
  precise dishonesty round 10/11/12 keep catching.
- **(b) permutation-invariant comparison** — there is no stable f32 reference to be invariant *to*
  (the f32 sum genuinely differs by order). An **f64** reference exists, but comparing f32-GPU to an
  f64 twin needs a tolerance, and that tolerance is exactly the unbounded (a) again.
- **(d) declared determinism assumptions** — a kernel cannot *assume away* hardware nondeterminism;
  the measurement shows the hardware really is nondeterministic.
- **(c) reject from the differential, keep the proofs** — the honest shape. Bounds and race-exemption
  are order-independent facts that stay fully meaningful; only the functional-equivalence differential
  is unavailable. v1 takes (c) in its strongest form: **reject float atomics outright** with the
  measured message, and record a **proved-only** float mode as a deferred, explicitly-weaker option
  (§8.4).

This mirrors two existing honesty stances: `sum_racy` (bounds `Proved` while the differential *correctly*
cannot certify — [coverage.md](coverage.md#races-outside-cooperative-kernels)) and `plane_*`
(the device decides something the twin cannot model — [coverage.md](coverage.md#subgroup--warp-reductions-plane_--planned)).

---

## 2. API catalog (validated)

`Atomic<Inner>` (`cubecl-core-0.10.0/src/frontend/element/atomic.rs:16`) wraps a numeric primitive and
**disables normal operations** on it — the type itself is the access-discipline that §6.1 leans on.
Two method sets, split by trait bound:

| Method | AtomicOp (IR) | Available for | Order-independent? | v1 |
|---|---|---|---|---|
| `fetch_add(v) -> old` | `Add` | `Scalar: Numeric` (int + float) | int: **yes**; float: **no** | int only |
| `fetch_sub(v) -> old` | `Sub` | `Scalar: Numeric` | int: **yes**; float: **no** | int only |
| `fetch_max(v) -> old` | `Max` | `Scalar: Numeric` | **yes** | int only (float unsupported by backend, §4.2) |
| `fetch_min(v) -> old` | `Min` | `Scalar: Numeric` | **yes** | int only (float unsupported) |
| `fetch_and/or/xor(v) -> old` | `And`/`Or`/`Xor` | `Scalar: Int` | **yes** | int (v1) |
| `load() -> v` | `Load` | `Scalar: Numeric` | n/a (pure read) | accept as read |
| `store(v)` | `Store` | `Scalar: Numeric` | **no** (last-writer) | reject (§8.3) |
| `swap(v) -> old` | `Swap` | `Scalar: Numeric` | **no** (last-writer) | reject (§8.3) |
| `compare_exchange_weak(cmp,v) -> old` | `CompareAndSwap` | `Scalar: Int` | **no** (+ retry loop) | reject (§8.3) |

Method-name note for the rejection layer: the ecosystem spelling is `Atomic::fetch_add`
(`atomic.rs:71`), method calls on an `Atomic` value; the **capital-`A` `Atomic` ident** is what the
current ban catches (`BANNED_PREFIXES`, `lib.rs:175`), as [coverage.md](coverage.md#scatter-add--histogram-atomics--planned)
already records.

---

## 3. IR representation — how the location lowers (validated)

The load-bearing question: *is the atomic location a normal buffer slot the prover already
bounds-checks?* **Yes.** Dumped IR of `bins[b].fetch_add(1u32)` for `bins: &mut Array<Atomic<u32>>`
(`ir_dump.rs`, `FINDINGS.txt`):

```
binding(3) = output(1)[binding(2)]        Operator::Index   list=GlobalOutputArray(1)  index=binding(2)
                                          -> a value of type  atomic<u32>
binding(4) = atomic_add(binding(3), u32(1))   Operation::Atomic(AtomicOp::Add)  lhs=binding(3)  rhs=u32(1)
```

Three facts follow, each decisive for the design:

1. **Addressing is an ordinary `Operator::Index`** into a `GlobalOutputArray` — the exact shape
   `process_index` (`prover.rs:2806`) already resolves and bounds-checks. The `AtomicOp` carries **no
   index**; it consumes the already-indexed pointer `binding(3)`. So the atomic Index emits the same
   `0 <= index < bins.len()` obligation as any array read, **for free** (measured — §9).
2. **The buffer's storage type is `Atomic(UInt(U32))`** (`type.rs:92`), and the Index result is
   `atomic<u32>`. cubecl's own bounds-check pass keys on exactly this (`checked_io.rs:61`,
   `if op.list.ty.is_atomic()`). The list's `VariableKind` is still `GlobalOutputArray`, so
   `array_ref`/`length_of` (`prover.rs:2746,1497`) recognise it unchanged.
3. **The pairing is structural.** An `Atomic<T>` pointer is produced by an Index and consumed by
   exactly one following `AtomicOp` (`self.into()` → pointer → op; the type forbids anything else). So
   the prover can pair `Index(atomic result)` with its `AtomicOp` deterministically to recover
   (buffer, index, read/write, atomic) — no dataflow guesswork.

Current prover behaviour on this IR (unchanged, and correct as far as it goes): the Index emits the
bounds obligation and records a **read** access; `model_element_read` returns false because the
`atomic<u32>` out is not `is_modeled_int` (`prover.rs:2861`), so the pointer is tainted; the `AtomicOp`
hits `_ => taint_out` (`prover.rs:2276`), tainting the returned old value. For a histogram the old
value is unused, so nothing downstream fails. The two gaps this leaves — the access is recorded as a
*read* not a *write*, and the race walker's inter-cube gate rejects the non-disjoint index — are what
§5–§6 close.

---

## 4. Backend reality (measured) — the surprising part

### 4.1 What runs

On M3/Metal via wgpu (VeriCL's default backend, `WgpuRuntime::client(&Default::default())`):
integer `fetch_add` ✓, integer `fetch_max` ✓ (bit-exact vs host), f32 `fetch_add` ✓ (but
nondeterministic, §1.3). All three exercised end-to-end in `probe.rs`.

### 4.2 What the feature flag *says* vs what the backend *does*

`client.properties().features.atomic_type_usage(...)` reports, for **i32, u32, and f32 alike**:
`LoadStore = true, Add = true, MinMax = false`. Source: `cubecl-wgpu-0.10.0/src/backend/wgsl.rs:83-113`
registers only `LoadStore | Add` for `[I32, U32]`, and f32 the same behind `SHADER_FLOAT32_ATOMIC`;
**`MinMax` is registered nowhere on the WGSL path.** The Vulkan path *does* register
`AtomicUsage::all()` for i32/u32 (`vulkan.rs:294`) — so the flag is backend-shaped, not spec-shaped.

Yet `fetch_max` on `Atomic<u32>` **compiled, launched, and matched the sequential host max exactly**
(§9). WGSL has `atomicMax` for `atomic<i32>`/`atomic<u32>`; cubecl emits it and it runs — the
`atomic_type_usage` map simply **under-reports** it.

**Design consequence:** VeriCL must not gate atomic support on `atomic_type_usage` (it is wrong in the
conservative direction — it would reject a working `atomicMax`). Gate on VeriCL's own subset, and let
the conformance suite's **differential lane** be the backstop that catches any backend where an atomic
op silently miscompiles (a wrong value diverges from the bit-exact twin, exactly as for every other
kernel). A standing tripwire test (the shape of `f64_wgpu_unsound`) should assert that integer
`atomicMax` on the pinned backend equals the sequential twin — so a future cubecl that *adds* a
refusal-gate keyed on the false flag fails loudly instead of silently dropping the kernel (risk 3).

---

## 5. Deliverable A — the twin, and the two bounds facts

### 5.1 The integer twin: a sequential fold (bit-exact)

The derived twin runs the body once per thread over `ABSOLUTE_POS = 0..num_threads` — the **existing
flat twin** (`lib.rs` `conformance_case`) — with one addition: an `Atomic<T>` output array is modeled
as a plain `&mut [T]`, and `bin.fetch_add(v)` becomes `{ let old = bins[i]; bins[i] = old.wrapping_add(v); old }`
(and analogously for sub/max/min/and/or/xor). Because these ops are order-independent (§1.1), the
sequential order the twin happens to pick reproduces the GPU multiset result **bit-for-bit** — no new
faithfulness obligation, no tolerance. This is *strictly simpler* than the phase-split cooperative twin:
no barriers, no promotion, no per-cube arena.

**Location initial value — a shared drawn input, not an automatic zero.** Unlike `SharedMemory`
(genuinely uninitialized → poison-init, design-shared-memory §4.5), an `Array<Atomic<T>>` output is a
device buffer the harness **fills with a drawn initial value before launch, and the twin's `&mut [T]`
starts from the identical draw** — so both lanes accumulate onto the same defined base and bit-exactness
holds regardless of what that base is. This is the honest general design; "zero-fill" would be a subtle
bug for `fetch_max`/`fetch_min`, whose identity is **type-dependent** (max-identity is the type minimum —
`0` for `u32` but `i32::MIN` for `i32`; min-identity is the type maximum). A histogram draws
`gen(bins in 0..=0)` (zeros, the add-identity); an atomic-max kernel draws the correct extremum, and the
twin uses the same drawn buffer either way. Treating the initial state as a shared input — never an
implicit zero — is what keeps the twin faithful across ops. (The GPU buffer must be explicitly
initialized: device memory is not zero by default; the harness writes the draw. Measured: `probe.rs`
initializes `bins` from a host slice on every run, and the twin matches.)

### 5.2 Bounds fact #1 — scatter-add with an element assume proves **today**

Measured (`proverprobe.rs`, real `prove_bounds_freedom`, z3 4.16.0): a clean-room atomic scatter

```rust
if i < idx.len() { bins[idx[i] as usize].fetch_add(val[i]); }
```

with `assumes(idx elements < bins.len(), idx.len() == val.len())` →
**`Proved { obligations: 3 }`**, with the **current prover, no changes.** The gather element-range
machinery (`model_element_read`, `prover.rs:2861`; `Assume::ElemsBelowLen`) applies to the atomic buffer
verbatim: `bins[idx[i]]` is a nested gather whose inner index `idx[i]` is modeled `< bins.len()` by the
assume, so the outer obligation discharges. The negative control (assume removed) → `Refuted` with a
two-value counterexample. This is probe #6's "one bounds obligation hand-discharged", and it lands
without touching the prover.

### 5.3 Bounds fact #2 — the `bins[key % n_bins]` idiom needs one sound extension

Measured: `bins[ABSOLUTE_POS % 256]` with `assumes(bins.len() == 256)` → **`Proved { obligations: 1 }`**
(the prover already models `X % C ∈ [0,C)` when `X` is a *modeled* value). But the classic histogram
`bins[idx[i] % 256]` → **`OutOfSubset { "read index for bins[...] depends on a construct outside the
vericl v0 subset" }`**, because `idx[i]` is a **tainted array-content read** and `tainted % const` is
tainted.

The fix is small and unconditionally sound: **model `X % C` (C a positive integer constant) as a fresh
symbol in `[0, C)` even when `X` is tainted** — the range `0 ≤ (X % C) < C` holds for every `X`,
independent of whether `X` is modeled. This is a one-site change in the modulo handling
(the value the mod produces gets a bounded fresh symbol instead of taint-propagating), it can never
mint a false `Proved` (it only ever *narrows* the counterexample search, like every assume), and it
turns the most recognizable histogram idiom from `OutOfSubset` into `Proved`. It is the cleanest
capability-per-line in the milestone.

Both idioms therefore reach `Proved`:
- **scatter-by-precomputed-index** (`bins[idx[i]]`, `ElemsBelowLen`): ships **now**, 0 prover LOC.
- **modulo-histogram** (`bins[idx[i] % NB]`): ships behind the one modulo-range extension.

### 5.4 The write-invalidation soundness item (real, small)

The current prover records the atomic Index as a **read** (§3). An atomic RMW is a read-*modify-write*;
for **bounds** read-vs-write is identical, but for **element-assume invalidation** it matters: an
`ElemsBelowLen{bins, …}` assume (bins elements used as indices elsewhere) is invalidated only by a
*write* to bins (`prover.rs` "Write invalidation"). If the atomic RMW stays modeled as a read, that
assume survives a write it should not → a latent false `Proved` for a kernel that both atomically
updates `bins` **and** uses `bins`' contents as indices. A pure histogram never does this (the assume
subject is `idx`, not `bins`), so it does not bite today — but the M-B implementation **must tag the
atomic RMW's location access as a write** for invalidation. Pinned as a pre-registered risk (risk 2)
with a negative-control kernel.

---

## 6. Deliverable B — the race check (the spine's other half)

### 6.1 Global atomic arrays: race-exemption is an access-discipline fact, not an SMT query

The task frames the race extension as "an atomic op to L is race-exempt vs *other atomic ops* to L".
For a **global** `Array<Atomic<T>>` this reduces to something even cleaner than a two-thread query:

> An `Array<Atomic<T>>` parameter can be accessed **only** atomically (the `Atomic` type disables
> normal ops — `atomic.rs:12`). Every atomic op to any location in it is mutually defined with every
> other atomic op to that location (that is what "atomic" *means*). Therefore **if every access to a
> buffer is atomic, the buffer is race-free for any index pattern** — no disjointness needed.

So the v1 global-atomic race obligation is discharged **structurally**, per buffer:

1. every access to buffer `B` in the kernel IR is an atomic op (holds by construction when `B`'s
   parameter type is `Array<Atomic<T>>` and nothing aliases it — §6.3); and
2. no *other* buffer parameter aliases `B` (§6.3).

This is why the current inter-cube gate (`check_intercube_global`, `prover.rs:3405`), which demands the
global-output index be `ABSOLUTE_POS`-disjoint or a `tid==0` single-writer, must be **extended to
EXEMPT atomic writes**: a histogram's `bins[idx[i]]` is neither disjoint nor single-writer, and under
the *non-atomic* gate it is correctly `OutOfSubset`; the atomic semantics are exactly what make the
collision defined. The SMT queries in §6.4(C,D) show the collision is genuinely reachable (`sat` as a
disjointness query) — which is *why* it must be exempted, not made to discharge.

**No new SMT theory, no two-thread walk, for the global-atomic v1 case.** This is a deliberate
narrowing: the strong measured result (integer histograms) needs only the structural check plus the
bounds obligation.

### 6.2 The mixed-access obligation — where the two-thread SMT walk earns its keep

A **plain** (non-atomic) access to `L` concurrent with an **atomic** op to `L` **is** a race
(mixed-access). Within cubecl's type system this **cannot happen inside one kernel through a single
`Array<Atomic<T>>` parameter** — the type forbids the plain access. It arises in exactly two ways:

- **Aliasing across parameters** — one buffer bound both as `Array<Atomic<T>>` and as `Array<T>`
  (§6.3): handled by the non-aliasing gate, not the SMT walk.
- **Shared atomic tiles** — a `SharedMemory<Atomic<T>>` block-histogram tile touched *atomically* in
  one phase and *plainly* in another (e.g. atomic-add during accumulation, plain-read during the
  merge). Cross-**phase** mixed access is ordered by the barrier and safe (design-shared-memory §5.3,
  "no cross-phase obligations"). **Same-phase** mixed access — an atomic write and a plain read of the
  same tile with no barrier between — is the real race, and it is the one that needs the two-thread
  disjointness query.

The extension to the existing race walker is small and localized:

- Add `is_atomic: bool` to `Access` (`prover.rs:1225`), set when the access is an atomic-typed Index
  paired with its `AtomicOp` (§3). Set `is_write = true` for RMW/Store atomics (Load stays a read).
- In `emit_race_obligations` (`prover.rs:3279`), for a same-array same-phase pair:
  - **both atomic** → **skip** (exempt); count it as an `atomic_exempt` for the report.
  - **one atomic, one plain** → emit `check_race` (the existing disjointness query,
    `prover.rs:3346`): UNSAT ⟹ provably disjoint (safe mixed), SAT ⟹ mixed-access race (`Refuted`,
    two-thread counterexample) — never a silent pass.
  - **both plain** → unchanged.

`check_race` is reused verbatim; only the *pairing rule* in `emit_race_obligations` changes. Reads
never race with reads; a same-value benign race is still conservatively a race (unchanged posture).

### 6.3 Aliasing — the assumption made explicit

VeriCL's race identity treats distinct buffer ids as never aliasing (`RaceArray`, `prover.rs:1215`).
For atomics this assumption must be **stated in the contract**, because it is load-bearing for §6.1's
"nothing aliases `B`": if a launch bound the same device handle to both an `Array<Atomic<T>>` and an
`Array<T>` parameter, a plain write could race the atomic RMW and no obligation would fire. The launch
harness binds distinct handles to distinct parameters (the ordinary case), so the assumption holds; v1
records it as an `assumed` claim ("distinct buffer parameters do not alias") on any atomic kernel,
travelling the same way tolerance and race-freedom assumptions already do
(design-shared-memory §6). It is honest and visible rather than silent.

### 6.4 Feasibility — the mixed-access obligation validated in QF_LIA (measured)

Hand-written SMT-LIB (`mixed_access.smt2`), discharged by z3 4.16.0 (`FINDINGS.txt`):

| Obligation | Encoding | z3 | Meaning |
|---|---|---|---|
| Safe mixed (disjoint indices) | `t1≠t2 ∧ 0≤t1,t2<D ∧ t1=t2` | **unsat** | provably disjoint — no mixed race |
| Unsafe mixed (data-dependent atomic write vs plain read) | `t1≠t2 ∧ 0≤b<D ∧ b=t2` | **sat** | mixed race reachable ⟹ **reject** |
| Atomic-vs-atomic, data-dependent bins | `t1≠t2 ∧ 0≤b1,b2<D ∧ b1=b2` | **sat** | collide *as a WW query* — hence must be **exempted**, not discharged |
| Global histogram, two threads, same bin | `a1≠a2 ∧ 0≤k1,k2<L ∧ k1=k2` | **sat** | only atomics make it defined; the non-atomic gate rejects it |

All plain linear integer arithmetic — no new theory. The safe case is provable; the unsafe cases are
`sat` (the walker will not vacuously over-prove). Rows 3–4 are the affirmative evidence for §6.1: a
histogram's writes genuinely collide, so exemption (not disjointness) is the only sound treatment.

---

## 7. The contract surface + claim shapes

### 7.1 Declaring an atomic kernel

No new *clause* is required for the common case: an atomic kernel is a plain 1-D kernel whose output is
`&mut Array<Atomic<T>>`. The macro un-bans `Atomic` **only when** the output parameter type carries it
(the same biconditional posture `cooperative(...)` and `dispatch(...)` use — required when used,
rejected when not). The existing `assumes(...)` machinery addresses the atomic buffer exactly as it does
any array: `assumes(idx.iter().all(|&b| (b as usize) < bins.len()))` for scatter-by-index, or
`assumes(bins.len() == N)` for the modulo-histogram — the histogram's bins are bounded by the index's
range, and the gather element-assume applies (§5.2). Integer atomic kernels use `compare(exact)` /
`compare(max_ulp = 0)` (bitwise), the natural mode given order-independence, plus the `wrapping` clause
for add/sub (WGSL wraps).

### 7.2 The claims recorded

| Kernel class | `proved`/smt-oob-freedom | `proved`/smt-atomic-race-freedom | `tested` (differential) |
|---|---|---|---|
| integer scatter-add / histogram / max-min | yes (bounds, §5) | yes (structural exemption, §6.1) + `assumed`: non-aliasing (§6.3) | **bit-exact** vs sequential-fold twin |
| shared/block histogram (v1.1) | yes | yes (mixed-access SMT, §6.2) | bit-exact vs phase-split+fold twin |
| float atomic add/sub | (proved-only mode, §8.4) | (proved-only mode) | **refused** — recorded reason "float-atomic order-nondeterministic; no stable reference" |

The new check string is `smt-atomic-race-freedom`, a sibling of `smt-race-freedom`
(evidence.rs `ClaimKind::Proved`, discriminated by the `check` string). Its `config` records: the
per-buffer exemption count, the mixed-access obligation count (0 for global v1), the non-aliasing
assumption, and — critically — the **element type is integer**, so the bit-exact differential claim is
legible as "integer-atomic bit-exact" rather than a tolerance. A float atomic kernel that reaches the
harness produces a **refused** differential entry (no `tested` claim minted), the same posture as a
cooperative differential with neither a race proof nor an assumption (design-shared-memory §6) — never a
silently green float atomic run.

---

## 8. The v1 subset boundary

### 8.1 Accepted (v1)

1-D topology; `&mut Array<Atomic<T>>` for **integer** `T` (`u32`/`i32`; `u64`/`i64` only where the
backend registers them — measured per-launch, not assumed); the order-independent ops `fetch_add`,
`fetch_sub`, `fetch_max`, `fetch_min`, `fetch_and`, `fetch_or`, `fetch_xor`, and `load` (as a read);
scatter-by-index bounded by an element-range assume, or modulo-histogram behind the §5.3 extension;
`compare(exact | max_ulp = 0)` with `wrapping` for add/sub. Bounds proved; race-exemption proved (with
the non-aliasing `assumed`); differential bit-exact.

### 8.2 Rejected, with targeted errors

| Construct | Error |
|---|---|
| `Atomic<f32>` / any float atomic output | macro: **"float atomic ops have no defined accumulation order (measured nondeterministic on wgpu/Metal, up to ~390 ULP under contention); rejected from the differential — see docs/design-atomics.md §1"** |
| `swap` / `store` / `compare_exchange_weak` | macro: "last-writer / CAS atomics are order-dependent (final value depends on interleaving) and are outside the vericl v0 subset (§8.3)" |
| `bins[idx[i] % N]` **before** the §5.3 extension | prover: existing "read index for `bins[...]` depends on a construct outside the vericl v0 subset" |
| an `Array<Atomic<T>>` also read/written non-atomically via an aliasing `Array<T>` param | harness: the non-aliasing `assumed` makes this a refused/loud case, not a silent pass (§6.3) |
| `Atomic` used without the output carrying it (spurious) | macro: biconditional gate, "the `Atomic` vocabulary is enabled only for a `&mut Array<Atomic<_>>` output" |
| `SharedMemory<Atomic<T>>` (block histogram) | prover: "shared atomic tiles (block-histogram) are deferred to vericl v1.1 (§8.5)" |
| 2-D atomic dispatch | requires `dispatch(...)` × atomics (§8.5) |

### 8.3 Deferred — order-dependent atomics (`swap`, `store`, `CAS`)

`swap`/`store` are last-writer (order-dependent final value); `compare_exchange_weak` is order-dependent
**and** its idiomatic use is a data-dependent retry loop (`Branch::Loop`, already `OutOfSubset`). These
are not bit-exact-twinnable and are deferred with the same honesty as float add — the difference is that
integers *could* one day get a proved-only mode too, but the retry-loop shape blocks it independently.

### 8.4 Deferred — float atomics, proved-only mode

A future mode could admit a float atomic kernel for its **bounds** and **race-exemption** proofs (both
order-independent, fully meaningful) while **refusing** the differential — a genuinely useful "this
histogram cannot go out of bounds and its bins cannot race, but we cannot certify its values"
statement. It is **strictly weaker** than VeriCL's custody standard (a green run with no functional
check), so it is gated behind an explicit opt-in and is **not** v1. Recording it here keeps the door
open without over-claiming.

### 8.5 Deferred — shared/block histogram and 2-D (v1.1)

The block-histogram (per-cube shared atomic tile, then global merge) needs: (i) `SharedMemory<Atomic<T>>`
support (the type compiles — `Atomic<Inner>: CubePrimitive` — but shared-atomic backend support is
**unmeasured** here and device-gated; a probe is required before claiming it); (ii) the same-phase
**mixed-access** SMT obligation of §6.2 (the plain-read-of-atomically-written-tile merge, safe only
across a barrier); (iii) the cooperative twin (design-shared-memory) composed with the fold twin.
2-D atomic dispatch is `dispatch(...)` × atomics — the image-histogram shape — a mechanical composition
once both land. Each widens an axis without changing the integer/float honesty spine.

---

## 9. Ground truth (measured, preserved in scratchpad)

| Probe | What it establishes | Result |
|---|---|---|
| `probe.rs` | integer histogram determinism + total | **8/8 runs bit-identical**, total = 1 048 576 exactly |
| `probe.rs` | integer atomic max vs sequential twin | **matches host `fold(max)` exactly** |
| `probe.rs` | reported atomic support | i32/u32/f32 = LoadStore+Add, **MinMax=false** (yet max ran — §4.2) |
| `probe.rs` / `fgpu.rs` | float atomic add nondeterminism | **19/19 runs bit-distinct** every config; 11–93 ULP (positive), up to **389 ULP** (cancelling), abs spread to **135 168** (wide range) |
| `fspread.rs` | backend-independent worst-case spread | up to **652 ULP** (cancelling, 65 536/bin); grows ~√N |
| `ir_dump.rs` | atomic location lowering | normal `Operator::Index` → `atomic<u32>` ptr, then `AtomicOp::Add`; buffer `Atomic(U32)` |
| `proverprobe.rs` | scatter-add bounds (element assume) | **`Proved{obligations:3}`**, current prover, 0 changes |
| `proverprobe.rs` | modulo-histogram bounds | modeled key **`Proved{1}`**; tainted key **`OutOfSubset`** (→ §5.3 extension) |
| `mixed_access.smt2` | mixed-access obligation shape | **unsat, sat, sat, sat** (safe provable; unsafe/exempt reachable) |

Together: the integer path is **bit-exact, deterministic, and mostly already-proved**; the float path is
**measured-nondeterministic and untolerable**; the race property is **structural for global, SMT for
mixed**; and the bounds are **free except for one sound modulo extension**.

---

## 10. Implementation plan (agent-sized milestones)

Each lands behind the existing posture (`cargo test --workspace`, clippy 0, evidence regenerated
*last*). The bounds and twin work is independent of the race work; **integer-first**, float never.

**M1 — Un-ban `Atomic` behind the output-type biconditional (macro).** `BANNED_PREFIXES` keeps
`Atomic` banned *unless* a `&mut Array<Atomic<T>>` output is present; reject float `T`, and
`swap`/`store`/`CAS`, with the §8.2 messages. *Verify*: an integer-atomic kernel passes the gate; a
float one and a `swap` one are rejected by name; a spurious `Atomic` ident with no atomic output is
rejected.

**M2 — The integer fold twin (macro).** Model `Array<Atomic<T>>` output as `&mut [T]`; lower each RMW to
its wrapping fold step (§5.1); zero-init (not poison — §5.1). *Verify*: the generated twin of a
clean-room `histogram_u32` and `scatter_add_u32` reproduces wgpu **bit-exact** across the shapes of §9
(the `*_twin_matches_handwritten` precedent), and the atomic-max twin matches host.

**M3 — Bounds: tag atomic RMW as a write; add the modulo-range extension (prover).** Pair the atomic
Index with its `AtomicOp` (§3); record the access `is_write=true` so element-assume invalidation is
correct (§5.4); model `X % C ∈ [0,C)` for tainted `X` (§5.3). *Verify*: scatter-add `Proved` (already
green — regression-pin it); modulo-histogram flips `OutOfSubset → Proved{1}`; a kernel that atomically
writes `bins` and reuses `bins` as indices is **not** falsely `Proved` (the risk-2 negative control).

**M4 — Race-exemption: the structural global check + the new `proved` claim (prover).** Exempt atomic
writes in `check_intercube_global` (§6.1); verify per-buffer all-atomic + non-aliasing; emit
`smt-atomic-race-freedom` with the exemption/obligation counts and the non-aliasing `assumed`.
*Verify*: `histogram_u32` `Proved` race-exempt (0 SMT race obligations, N exemptions recorded); a kernel
mixing an atomic and an aliasing plain write is refused loudly.

**M5 — The coupling + claim wiring (harness).** Integer atomic kernel evidence shows the
`tested`(bit-exact) + `proved`(bounds) + `proved`(race-exempt, with non-aliasing `assumed`) triple; a
float atomic kernel that reaches the harness is **refused** (no `tested` minted), with the recorded
reason. *Verify*: a public `histogram_u32` example carries the triple; forcing a float variant produces
the refusal, not a green run.

**M6 — Mixed-access SMT (prover, v1.1 gateway).** `Access.is_atomic`; the atomic-vs-plain pairing rule
in `emit_race_obligations` reusing `check_race` (§6.2). *Verify*: the §6.4 obligation verdicts
reproduced through the walker on a clean-room same-phase mixed kernel (safe `Proved`, data-dependent
`Refuted`). Ships with the shared-tile probe of §8.5 or stands alone as the aliasing check's SMT form.

**M7 — Public example + private dogfood.** A clean-room `histogram_u32` (scatter-add + max) wired into
`vericl::suite!` carrying the triple; dogfood the one private atomic kernel by shape only (README
policy) to confirm the real shape lands. *Verify*: suite green, evidence regenerated last.

---

## 11. Open risks, ranked (pre-registered for review round 13)

1. **Float rejection is a *policy* boundary an adversarial reviewer will push (high).** The measured
   nondeterminism (§1.3) justifies rejecting float from the *differential*; the reviewer will ask "then
   why not ship the proved-only float mode in v1?" The honest answer is custody (§8.4): a green run
   with no functional check is weaker than VeriCL's standard, so it is opt-in and deferred — but this is
   a *judgement*, not a measurement, and must be defended as one, not smuggled in. **Attack surface:** a
   float histogram handed to v1 must produce the *refusal* wording and no `tested` claim; a test pins
   it.

2. **Atomic RMW modeled as a read is a latent false `Proved` (high, concrete).** Until M3 tags the
   atomic location access as a write, an `ElemsBelowLen{bins,…}` assume survives an atomic write to
   `bins`. Pure histograms do not trigger it, so it will *look* fine. **Attack surface:** the reviewer
   hands a kernel that atomically updates `bins` and *also* indexes with `bins[j]` elsewhere under an
   element-assume on `bins`; without the write-tag it is falsely `Proved`. The risk-2 negative control
   must fire `Refuted`/invalidated. This is the round-8 "modeled where it must be tainted" shape.

3. **`atomic_type_usage` is wrong, and VeriCL must not trust it (medium).** MinMax reports false while
   `atomicMax` runs (§4.2). If VeriCL ever gates on the flag it will reject working kernels; if a future
   cubecl *adds* a codegen refusal keyed on the flag, VeriCL's `atomicMax` kernels break at the cubecl
   layer. **Mitigation:** gate on VeriCL's subset + a standing tripwire asserting integer `atomicMax` ==
   sequential twin on the pinned backend, so drift is loud. Cubecl-upgrade drill item.

4. **The modulo-range extension must be exactly `X % C, C` a positive *constant* (medium).** `X % Y`
   with a runtime/variable `Y` is *not* bounded by a constant, and `X % 0` is UB. **Attack surface:** a
   kernel with `bins[idx[i] % n_bins]` where `n_bins` is a *runtime scalar* must stay `OutOfSubset`
   (the bound is `< n_bins`, not a constant `bins.len()` fact) unless a separate `n_bins <= bins.len()`
   assume form is added — which v1 does **not** have. Pin both: constant-C proves, variable-C rejects.

5. **Non-aliasing is an assumption, and the harness must honor it (medium).** §6.1's structural
   race-exemption depends on "nothing aliases `bins`". The launch binds distinct handles, but an
   `ArrayArg::Alias` (cubecl supports output aliasing an input, `array/launch.rs`) could bind the same
   buffer to two params. **Mitigation:** the non-aliasing `assumed` is recorded on every atomic kernel;
   if the launch uses an alias, the harness must refuse rather than silently proceed. Pin an
   aliasing-launch negative control.

6. **Shared-atomic backend support is unmeasured (medium, scoping).** §8.5's block histogram assumes
   `SharedMemory<Atomic<T>>` works on wgpu/Metal; the type compiles but the backend behaviour was not
   probed here. **Mitigation:** measure before claiming v1.1 lands it; keep it deferred and explicit
   until then, exactly as `plane_*` width is deferred.

7. **cubecl upgrade drift on the atomic IR (low, standing).** `AtomicOp`'s variant set, the
   `StorageType::Atomic` typing, and the "Index-produces-pointer, AtomicOp-consumes-it" pairing are
   internals a cubecl upgrade could change (e.g. folding the index into the AtomicOp). **Mitigation:**
   the IR-level identity hash + the "survives a CubeCL upgrade" health check already trip on codegen
   drift; the M3 pairing logic must fail *loudly* (not silently mis-pair) if the shape changes.

---

## 12. Roadmap impact

- **Resolves** M-B's precondition ("must resolve float-atomic-add ordering honesty first",
  [coverage.md](coverage.md#the-gap-closure-plan)): resolved by measurement — integer ships bit-exact,
  float is refused from the differential with a measured reason.
- **New public claim kind**: `proved`/`smt-atomic-race-freedom`, the third machine-checked property,
  joining `smt-oob-freedom` and `smt-race-freedom`. For global atomics it is *structural* (per-buffer
  access discipline), a genuinely new proof shape that is not a two-thread SMT query.
- **Reuses more than it adds.** Bounds: the gather element-assume + one modulo line. Twin: the flat
  twin + a fold lowering. Race (global): a structural check, no SMT. Only the v1.1 shared-tile
  mixed-access reuses the full two-thread walk. The milestone is small *because* the honest scope is
  narrow.
- **Does not** need a tolerance model, QF_BV, the f64 tier, or 2-D dispatch. The v1 subset is 1-D
  integer f32-free, matching everything else in vericl v0. The coverage page's M-B row moves from
  PLANNED to landed for integer scatter-add / histogram / atomic-max-min; float atomics, `swap`/`CAS`,
  and block/2-D histograms are recorded as deferred with their reasons measured, not asserted.
```
