# Struct-typed `#[comptime]` parameters — design (July 2026)

The implementable design for the re-census's #1-ranked ecosystem shape: a `#[cube]` item taking
`#[comptime] config: SomeConfig` where `SomeConfig` is a user struct/enum
(`docs/ecosystem-survey-2026-07.md` re-census: **243 / 464** items, **38** sole-blocker). The
secondary question — whether the same mechanism cheaply covers plain runtime `CubeType` struct args
— is answered too (**it does not**; §4.4).

Everything marked *measured* was checked empirically against the pinned `cubecl =0.10.0`
(z3 4.16.0 on PATH, wgpu 29 / Metal on an Apple M3), the same posture as
[design-line-vector.md](design-line-vector.md) and [design-shared-memory.md](design-shared-memory.md).
Probe sources are preserved in the scratchpad (`scratchpad/structct/src/bin/{api,bounds,ir,gt,gt2,gt3,gt4,neg,rej,identity,vcl_shapes,traps}.rs`,
the `cfgproto`/`cfgrt`/`cfgtest` prototype crates, and `classify_nostructct.py`); the consolidated
run is `scratchpad/structct/RESULTS.txt`. Reference shapes are **clean-room / upstream-public only**
(cubecl 0.10.0, cubek, burn — MIT/Apache-2.0), per the README policy.

File:line citations to `crates/vericl-macros/src/lib.rs`, `crates/vericl/src/contract.rs`,
`crates/vericl-macros/src/suite.rs` and the `cubecl-{core,macros,runtime}-0.10.0` trees are current
as of `c43a53c`.

---

## 0. Headline recommendation

1. **The feature is not missing. It already works — by accident, ungated and unclaimed.** The
   brief's premise ("`instantiate(...)` currently pins scalar comptime values") is a **third**
   API-reality correction in the design-line-vector tradition, and the largest one yet:
   `classify_param` only rejects *reference* comptime types (`lib.rs:2020-2027`), and
   `resolve_instantiate` stores a comptime value as **raw tokens** (`lib.rs:2409-2410`). Measured on
   today's unmodified VeriCL at `c43a53c`: **seven** struct-comptime shapes — plain fields, depth-2
   method chains, `comptime!` blocks over a config, enum + `match` dispatch, config-driven loop
   bounds, `uses(...)` helper composition, and a `const fn` pinned value — all compile, all pass the
   wgpu/Metal differential, and all carry `Proved` SMT bounds (§2, `RESULTS.txt`). So do
   cooperative × config, `Vector` × config, `assumes(...)` × config, generic-over-a-config-trait
   (`C: TileConfig`), path-qualified struct literals, and a non-`Copy` `Vec<u32>` comptime value.
   **Zero new capability is needed. The milestone is a hardening-and-claiming milestone.**

2. **The IR is byte-identical to the scalar case, so the prover needs zero changes — measured, not
   argued.** A struct-comptime kernel, a *depth-2-method-chain* struct-comptime kernel, and the
   plain-comptime-scalar kernel with the same numbers all produce the **same
   `kernel_ir_hash`**: `sha256:c92d99bf…` for all three (§3). `cfg.m` is `Constant(UInt(3))` in the
   `RangeLoop` end; `cfg.tile_size().n()` is `Constant(UInt(8))` in the `Mul`. There is no
   struct in the IR at all, and cubecl **cannot** put one there: lowering a comptime value to a
   `Variable` goes through `NativeExpand::from_lit`, which requires `Into<ConstantValue>`
   (`cubecl-core-0.10.0/src/frontend/element/base.rs:467-474`) — a struct fails to compile rather
   than silently becoming a GPU value. This is the Slice precedent, but stronger: not "the prover
   handles it", but "the prover cannot tell the difference".

3. **There are exactly three real defects, and they are the whole milestone.**
   (a) **The identity hole** — `SOURCE_HASH` covers the kernel's own tokens + the contract attribute
   tokens + the vericl version (`lib.rs:2942-2949`); a config type's **definition** is in neither.
   Measured: editing a config method from `self.m * self.n` to `self.m + self.n` changes the kernel
   from ×24 to ×11 and leaves `SOURCE_HASH` **bit-identical** at `sha256:dd3d0579…`, with
   `contract.instantiate` still printing `"cfg = TileCfg { m : 3, n : 8 }"` — recorded evidence
   stays "fresh" while describing a different kernel (§5.1).
   (b) **The config-method gate hole** — `FloatMethodCheck` walks the kernel body it is handed; a
   config method body lives in another item and is invisible. A `#[cube]` config method calling
   `fma` compiles, and only fails at *runtime* as a twin panic (§5.2).
   (c) **Unsound pinned expressions** — `instantiate(cfg = cfg_from_env())` is accepted today with
   no gate at all (§5.3).

4. **Decided design: `vericl::config! { … }` + three gates. `instantiate(...)` is unchanged.**
   The config type **and its impl blocks** are wrapped in one `vericl::config!` item macro, which
   re-emits them verbatim, hashes the whole token block, emits
   `impl vericl::ConfigIdentity for T { const CONFIG_HASH: … }`, and runs the host-callability check
   over every method body. The kernel folds `<T as ConfigIdentity>::CONFIG_HASH` into `identity()`
   via the existing `combine_source_hash` — the exact `uses(...)` / `reference = path` precedent
   (§6, §7). **Prototyped and validated**: the same config-method edit that leaves `SOURCE_HASH`
   unmoved moves `CONFIG_HASH` from `sha256:3ada1666…` to `sha256:4b537fa4…`; an `fma` in a config
   method body is rejected at the right span at compile time; an undeclared config type gets a
   targeted `#[diagnostic::on_unimplemented]` message (§6.4).

5. **Honest reach: this unlocks 38 ecosystem items and ZERO plain functions — and that is measured,
   not estimated.** Re-running the survey classifier with the struct-comptime row demoted from
   blocking to supported (`scratchpad/structct/classify_nostructct.py`): gate-free items
   **51 → 89** (+38, exactly the recorded sole-blocker count), but `fn_nontest` **12 → 12,
   unchanged**. All 38 are impl blocks (29) or trait definitions (9), and `#[vericl::kernel]` /
   `#[vericl::helper]` parse `ItemFn` (`lib.rs:2518`, `:3528`) — VeriCL structurally cannot annotate
   any of them. The value is (i) making the ecosystem's dominant *idiom* honestly expressible in
   clean-room ports and dogfood code, and (ii) turning an ungated accident into a claimed, tested,
   identity-covered surface. The frontier re-ranks decisively: **`custom CubeType param (broad)`
   goes from 8 to 28 sole-blocker, all 28 plain non-test `fn`s** — that, not struct comptime, is the
   next plain-function unlock (§11).

6. **The ecosystem's construction reality kills any "richer instantiate syntax" ambition, and that
   is a *simplification*.** 66% of real config values are computed from runtime tensor shapes,
   strides, dtypes and device properties; in cubek matmul/attention the config **does not exist on
   the host at all** — it is built inside the kernel by `expand_config(&comptime::device_properties(), …)`
   (§4.3). No attribute syntax can express that, and VeriCL's zero-client IR extraction cannot
   support `device_properties()` anyway (`docs/ir-research.md` §1). So `instantiate(...)` stays
   exactly as it is — an arbitrary token expression — and v1 *gates* which expressions are honest
   rather than growing new grammar (§10.1).

---

## 1. API reality, part 1 — cubecl 0.10's comptime-struct surface

Catalogued against the pinned registry tree (`cubecl-{core,macros,runtime}-0.10.0`), verified
byte-identical to the survey workspace's `cubecl` checkout at `7cf2037`.

### 1.1 There is no `CubeType`, no derive, and no bound the macro imposes

The entire comptime/runtime discrimination at declaration time is one function
(`cubecl-macros-0.10.0/src/parse/kernel.rs:767-775`): a `#[comptime]` param's type is left
**untouched**, while a runtime param's becomes `<T as CubeType>::ExpandType`. No trait bounds are
injected into the generated generics (`:731-755`).

The bounds that *do* apply are implied by generated code, and were minimized empirically
(`scratchpad/structct/src/bin/bounds.rs`):

| Context | Required of the comptime type | Where it bites |
|---|---|---|
| `#[cube(launch)]` kernel param | `Clone + Debug + Hash + PartialEq + Eq` (+ `Send + Sync + 'static`, which must be *written* for a generic `C`) | hand-written `{Kernel}Info` impls, `generate/kernel.rs:522-571`; `KernelId::info<I: 'static + PartialEq + Eq + Hash + Debug + Send + Sync>`, `cubecl-runtime-0.10.0/src/id.rs:147` |
| bare `#[cube]` helper param | **`Clone` only** | `expression.rs:227-231` (`#name.clone()`) |
| `CubeType` | **not required** | — |
| `Copy` | **not required** | cubecl's own `Vec<u32>` and `Operation<U>` comptime params |

A consequence worth recording: **a comptime config field can never be a float.** `Hash + Eq` on the
config type is mandatory for `#[cube(launch)]`, and `f32: Eq` does not hold — measured as a hard
compile error. Comptime configs are integer/bool/enum-valued by construction.

`CubeComptime` exists (`cubecl-core-0.10.0/src/frontend/element/base.rs:149-164`) but is a **blanket
alias** (`impl<T> CubeComptime for T where T: Debug + Hash + Eq + Clone + Copy`), referenced nowhere
in cubecl-macros; its own doc says a type "doesn't need to implement `CubeComptime` to be used as a
comptime argument". There is no `#[derive(CubeComptime)]` — only `#[derive_cube_comptime]`
(`cubecl-macros-0.10.0/src/lib.rs:154-164`), pure sugar for `#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]`.

### 1.2 Field and method access are re-emitted as ordinary host Rust

- `cfg.field` → `quote![#base.#field.clone()]` with `base.as_const()` yielding `cfg.clone()`
  (`generate/expression.rs:125-130`). A host field read executed while building the IR.
- `cfg.method(args)` → **the original `syn::ExprMethodCall`, verbatim**
  (`parse/expression.rs:131-159`: `Expression::Verbatim { tokens: quote![#method] }`, taken whenever
  the receiver and all args are const). Not `__expand_method`. This is the single most important
  fact for the twin (§8).
- `match cfg` with a const scrutinee → a plain Rust `match`, arm patterns untouched
  (`parse/expression.rs:300-316`, `generate/expression.rs:615-633`). No `Branch::Switch`.
- `if cfg.pred()` with a const condition → a plain Rust `if`; the dead branch never enters the IR
  (`generate/expression.rs:318-330`).
- `comptime! { … }` → `Expression::Verbatim` — entirely un-expanded host Rust
  (`parse/statement.rs:136-141`).

### 1.3 Launch and identity

Comptime args are **not** wrapped: only non-const params get rewritten to `RuntimeArg<T, R>`
(`generate/launch.rs:249-258`), and comptime args never touch the `KernelLauncher` — they are passed
straight into the kernel-struct constructor (`:147-161`). There is no `CompileTimeArg`/`ComptimeArg`
type in 0.10.

The comptime value participates in cubecl's own kernel cache key through the generated `{Kernel}Info`
struct, whose `Hash`/`Eq`/`Debug` are hand-written field-wise delegations
(`generate/kernel.rs:522-571`), fed to `KernelId::info`. Two caveats for the doc's risk register:
the `Info` impls carry **no `where` bounds**, so a missing `Hash` surfaces as an error *inside*
generated code; and `KernelId::stable_hash()`/`stable_format()` feed both `Hash` **and** `Debug` into
a **persistent, cross-process on-disk cache key** (`cubecl-runtime-0.10.0/src/id.rs:119-141`,
consumed at `cubecl-wgpu-0.10.0/src/backend/base.rs:44`).

---

## 2. API reality, part 2 — VeriCL already accepts all of it (measured)

The decisive probe. Every row below is today's unmodified VeriCL at `c43a53c`, run end-to-end on
wgpu/Metal with the SMT bounds prover (`scratchpad/structct/src/bin/{gt,gt2,gt3,gt4}.rs`).

| # | Shape | `instantiate(...)` | differential | proof |
|---|---|---|---|---|
| F1 | plain field access `cfg.m + cfg.n` | `cfg = TileCfg { m: 3, n: 8 }` | PASS | `Proved{2}` |
| F2 | depth-2 method chain `cfg.tile_size().total()` | `cfg = StageCfg { tile: TileCfg { m: 3, n: 8 }, k: 2 }` | PASS | `Proved{2}` |
| F3 | `comptime!(cfg.m() + cfg.n)` | `cfg = TileCfg { m: 3, n: 8 }` | PASS | `Proved{2}` |
| F4 | enum param + `match` dispatch | `mode = Mode::Triple` | PASS | `Proved{2}` |
| F5 | config **method** as a loop bound | `cfg = LoopCfg { taps: 3 }` | PASS | `Proved{3}` |
| F6 | `uses(...)` helper taking the same config | `cfg = TileCfg { m: 3, n: 8 }` | PASS | `Proved{2}` |
| F7 | `const fn` pinned value | `cfg = default_cfg()` | PASS | `Proved{2}` |
| I1 | **cooperative** (`cube_dim = 256`) × config-driven accumulation | `cfg = WinCfg { taps: 3 }` | PASS | — |
| I2 | config type that also derives `CubeType` and has a `#[cube]` method | `cfg = DualCfg { m: 3 }` | PASS | — |
| J1 | `assumes(x.len() >= cfg.stride())` (host-only surface) | `cfg = Cfg { stride: 4 }` | PASS | — |
| J2 | `Vector<f32, 4>` × config | `N = 4, cfg = VecCfg { k: 3 }` | PASS | — |
| K1 | generic over a config **trait**, both pinned | `C = Cfg3, cfg = Cfg3 { s: 5 }` | PASS | — |
| K2 | path-qualified struct literal | `cfg = nested::Deep { v: 7 }` | PASS | — |
| K3 | non-`Copy` heap value | `w = vec![2u32, 3, 5]` | PASS | — |

**Why it works.** Three independent pieces of existing machinery line up:

- `classify_param`'s comptime branch rejects only `Type::Reference`; every other type falls through
  to `ParamKind::Comptime(ty)` (`lib.rs:2020-2027`). The error text *says* "must be plain scalar
  types", but nothing enforces it — a **documentation/wording bug, not a gate** (§10.3).
- `resolve_instantiate` treats a comptime entry as opaque tokens:
  `comptime_values.insert(key, entry.value.to_token_stream())` (`lib.rs:2409-2410`). `InstantiateEntry`
  parses the value as a `syn::Expr` (`lib.rs:325-333`), and `Expr` parsing admits struct literals;
  a brace group is one token tree, so inner commas do not split the punctuated list.
- The twin binds it as `let #name: #ty = #value;` (`lib.rs:3041-3050`) — the config is host Rust in
  the twin, host Rust in `check_assumes`, and a plain positional argument in the `expand()` call
  (`lib.rs:3180-3190`).

**And `comptime!` blocks already accept it.** `ComptimeRefCheck` bans *bare non-comptime value
idents* and *nested macro invocations* (`lib.rs:856-905`); its doc explicitly notes "method names,
and field accessors are not runtime values and are allowed" (`lib.rs:791`). So
`comptime!(cfg.tile_size().m())` is admissible today (F3, measured).

---

## 3. IR reality — the prover needs zero changes (measured)

`scratchpad/structct/src/bin/ir.rs` expands four kernels through the zero-client `KernelBuilder`
recipe (`docs/ir-research.md` §1) and hashes each with `vericl_ir::kernel_ir_hash`:

```
k_struct  cfg=TileCfg{m:3,n:8}            sha256:c92d99bff97c7050830c4b8fce38022264a8da5c…
k_scalar  m=3 n=8                         sha256:c92d99bff97c7050830c4b8fce38022264a8da5c…
k_methods cfg=StageCfg{tile:{3,8},k:99}   sha256:c92d99bff97c7050830c4b8fce38022264a8da5c…
k_enum    mode=Triple                     sha256:9a39b56fff01d95d8c48630f46b2a1b6a9493f63…

struct-field == comptime-scalar : true
method-chain == comptime-scalar : true
```

The three are not merely equivalent, they are **the same bytes**. Concretely, `for i in 0..cfg.m`
lowers to `RangeLoop { end: Constant(UInt(3)) }` and `ABSOLUTE_POS * cfg.tile_size().n()` to
`Mul { rhs: Constant(UInt(8)) }`. The enum kernel's `match mode { Double => 2, Triple => 3 }` lowers
to a bare `Constant(UInt(3))` — no `Branch::Switch`, no branch at all.

**Consequence.** Every prover input — `Operator::Index`, `RangeLoop` bounds, path conditions,
`Metadata::Length` — is exactly what a comptime-scalar kernel produces. The bounds walker, the
race-freedom walker, the taint tracker, the counterexample renderer: unchanged. This is the strongest
form of the Slice precedent.

Discrimination, so the claim is not vacuous (`scratchpad/structct/src/bin/neg.rs`):

| Control | Expected | Measured |
|---|---|---|
| L1: config-driven read `x[i + cfg.stride()]`, no length relation | REFUTE | `REFUTED 0 <= index < x.len() (read access to \`x\`) \| len_x=4294967292, abs_pos=4294967291`; differential also FAILS (twin `index out of bounds`) |
| L2: same kernel + `assumes(y.len() + 4 <= x.len())` + `gen(len(x = n + 4))` | PROVE | `Proved{2}` |
| L3: identical kernel pinned `stride: 5` against the literal-4 assume | REFUTE | `REFUTED …`; differential FAILS |
| L2 vs L3 `SOURCE_HASH` | differ | `0c4aa039…` vs `269d4460…` — differ |

---

## 4. What the 243 sites actually are — and the three corrections that forces

Exhaustive re-classification of the census scope (the six crates behind the 464/243/38 numbers),
plus a broad scan of all of `cubecl` + `burn` + `cubek`. Within the 243 sites: **417 struct-typed
comptime param occurrences across 63 distinct type strings**.

### 4.1 Definition shapes — the fields-only struct is ~absent

| Shape | distinct types | occurrences | share |
|---|---:|---:|---:|
| enum | 26 | 140 | 34% |
| nested struct (contains other config types) | 16 | 139 | 33% |
| methods **and** nested | 7 | 55 | 13% |
| trait-generic / associated type (`Self::Config`, `GMM::Config`) | 4 | 48 | 12% |
| impl-method-heavy only | 2 | 9 | 2% |
| **plain fields-only struct** | **3** | **5** | **1.2%** |
| unresolved aliases | 5 | 21 | 5% |

**Correction 1: a design that only handles `cfg.field` on a fields-only struct addresses 1.2% of the
corpus.** The dominant types are `GlobalReaderConfig` (63 occurrences, 9 fields of which 5 are
themselves config types), `MatrixLayout` (45), `StageMemoryConfig` (12 fields / 9 methods),
`SharedGlobalMatmulConfig<S>` (7 fields / 11 methods). Nesting reaches depth 3. Method-chain access
is not an edge case: of ~1500 accessing mentions, **139 are `.field.method()` (depth 2)** and 16 are
depth 3 (`config.global_config.stage_config().elements_in_stage_m()`). The ubiquitous
`config.tile_size.m()` comes from `define_3d_size_base!` (`cubek-std/src/size.rs:14-95`).

Derives: **53 of 63 types (341 of 417 occurrences, 82%) carry plain Rust derives and no `CubeType`
at all**; the canonical spelling is `#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]` (32 types).
Four derive `CubeType`; there are **zero** explicit `impl CubeComptime` anywhere, and only 6 files
use `#[derive_cube_comptime]`. This confirms §1.1 from the demand side.

### 4.2 No config method is `#[cube]` — the brief's assumption, verified exhaustively

Scan over **65 concrete config type names** across all three checkouts: **52 impl blocks, 132
functions; 0 carry `#[cube]`**, and inside those bodies there are 0 `comptime!`, 0 `unexpanded`,
0 `Line`/`Vector`, 0 device intrinsic calls. The config *traits* (`GlobalConfig`, `StageConfig`,
`BatchConfig`, the attention trio) are all plain traits bounded
`Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static` with no `#[cube]`. The
separation is deliberate and visible side-by-side in one file:
`cubek-matmul/src/components/global/base.rs:212` `pub trait GlobalConfig:` (plain) vs `:56 #[cube] pub trait GlobalMatmul`.

**Every method reachable on a real comptime config is plain host-callable Rust returning
`u32`/`bool`/another config.** The brief's "no `unexpanded!()` trap because these are the USER's own
structs" is correct for the ecosystem as it stands — but it is a *property of the corpus, not of the
type system*, and VeriCL must not rely on it. A `#[cube]` config method is expressible (measured, I2)
and a non-host-callable one is a real hazard (§5.2).

One measurement correction to the census itself: the classifier's regex misses **reference-typed**
comptime params. There are **17 more** in the corpus (`#[comptime] config: &Self::Config` ×16,
`&S::Config` ×1), in `burn-cubecl/src/kernel/pool/` and cubek. Real incidence is ~3% higher than 243.

### 4.3 Construction: two-thirds of configs are runtime-derived, and matmul's never touches the host

Of 139 non-test config construction sites (±10%, regex heuristic): **92 (66%) are runtime/host-derived**
(tensor shapes, strides, `client`, `device_props`, `num_sms`, dtypes, discovered vector sizes),
31 (22%) all-literal, 16 (12%) host-symbolic.

More decisive: **of the 243 sites only 5 are `#[cube(launch…)]` entry points**; the other 238 receive
an already-built config. And cubek's matmul/attention kernels do not receive the config as a launch
argument at all — they receive a *blueprint* and build the config inside the kernel body at expand
time (`cubek-matmul/src/components/batch/partitioned_matmul/matmul.rs:57-82`):

```rust
let vector_size_lhs = Args::view_lhs(&state).vector_size();     // from the actual tensor
let device_props = comptime::device_properties();               // queried from the GPU
let config = comptime!(PartitionedBatchMatmulFamily::<…>::expand_config(
    &device_props, &blueprint, &dtypes, &vector_sizes));
if comptime!(config.is_err()) { push_validation_error(…); comptime!(return); }
```

9 `comptime::device_properties()` call sites, 20 fallible `expand_config` implementations chaining
batch → global → stage → tile.

**Correction 2: no attribute grammar can express the dominant construction path**, and VeriCL's IR
extraction could not honour it anyway — the zero-client `KernelBuilder` leaves `Scope.properties`
as `None`, and `comptime::device_properties()` is the one intrinsic that reads it
(`docs/ir-research.md` §1). This is *not* a gap to close; it is a boundary to state (§10.2).

### 4.4 The `CubeType` struct-arg question — answered: not the same mechanism

The comptime path is a **bypass**: zero generated types, value stored verbatim in the kernel struct.
`#[derive(CubeType)]` generates an `XExpand` companion + `impl CubeType`;
`#[derive(CubeLaunch)]` additionally generates `XLaunch<'_, R>`, `XCompilationArg` (with its own
hand-written `Hash`/`Eq`/`Debug`), and an `impl LaunchArg` whose `register`/`expand`/`expand_output`
fan out per field — and every field type must itself be `LaunchArg`
(`cubecl-macros-0.10.0/src/generate/cube_type/generate_struct.rs:12-289`). They share only the
id-material pattern.

Measured against VeriCL today: `#[derive(CubeType)] struct Pair { a: f32, b: f32 }` as a **runtime**
kernel arg is rejected with a clean, correct message —
`gen(...) v0 only supports f32/f64/u32/i32/u64/i64 scalar parameters; \`p: Pair\` is outside that
set` — plus rustc's `Pair: ScalarArgSettings` unsatisfied (it needs `CubeLaunch`, not `CubeType`).

**Correction 3: runtime `CubeType`/`CubeLaunch` struct args are a separate milestone.** They need a
twin representation for a struct-of-buffers, a `gen(...)` story for structured launch data, a
`LaunchArg` construction path, and comparison semantics per field. None of that is touched here. That
milestone is now the ranked next one (§11).

### 4.5 Co-occurrence — what the compatibility matrix must cover

Of the 243 sites with a struct comptime param vs the 221 without:

| feature | with (n=243) | without (n=221) |
|---|---:|---:|
| generic type params | **187 (76%)** | 142 (64%) |
| `Line`/`Vector` | **118 (48%)** | 58 (26%) |
| `match` | 84 (34%) | 35 (15%) |
| `comptime!` block | 55 (22%) | 28 (12%) |
| View/Layout machinery | 25 (10%) | 51 (23%) |
| `plane_*` / `sync_*` | 17 (7%) | 11 (5%) |
| `SharedMemory` | 11 (4.5%) | 9 (4%) |
| cmma / `Matrix` | 6 (2.5%) | 12 (5%) |

Generics (76%) and `Vector` (48%) are the two features that **must** co-work; both measured PASS
(K1, J2). Cooperative, shared memory, slices and cmma are each ≤7% and *not* enriched relative to
baseline — struct comptime is largely orthogonal to them.

Multiple config params per function: 1 → 221 (76%), 2 → 57 (20%), 3 → 8, 4 → 6. Param names:
`config` 192, `layout` 42, `ident` 20, `blueprint` 14.

### 4.6 The 38 sole-blocked items

Recomputed exactly: **29 `impl` blocks + 9 `trait` definitions, zero free functions.** By crate:
cubek-matmul 28, cubek-convolution 6, cubek-std 4. This matches the re-census's own recorded "sole
non-test `fn` = 0" and is load-bearing for §11.

---

## 5. The three real defects (measured)

### 5.1 The identity hole — the milestone's centre of gravity

`SOURCE_HASH = sha256(fn tokens ‖ "||contract:" ‖ attr tokens ‖ "||vericl:" ‖ version)`
(`lib.rs:2942-2949`). A config type's definition is in **neither** input.

`scratchpad/structct/src/bin/identity.rs`, one kernel `y[i] = x[i] * cfg.total()` pinned
`cfg = TileCfg { m: 3, n: 8 }`, built twice with only the *method body* changed:

| | base (`self.m * self.n` → 24) | alt (`self.m + self.n` → 11) |
|---|---|---|
| `SOURCE_HASH` | `sha256:dd3d05798f24431e…` | `sha256:dd3d05798f24431e…` **identical** |
| `identity().source_hash` | `sha256:dd3d05798f24431e…` | **identical** |
| `contract.instantiate` | `["cfg = TileCfg { m : 3, n : 8 }"]` | **identical** |
| `ir_hash` | `sha256:df4b3ed4c33f808b…` | `sha256:6720a5c0ebf1e252…` (moves) |
| differential | PASS | PASS |

Both variants are *honest kernels*; the defect is purely one of identity. Stored evidence for the
base variant passes `verify()` against the alt build.

**Scope of the hole.** `ir_hash` catches every config-derived quantity that reaches the device, but:
(i) it is only populated when the suite runs with `prove: true` (`suite.rs:512-513`), and it is
`Option<String>` with `#[serde(default)]` (`contract.rs:176-194`) — two `None`s compare equal;
(ii) it does not cover the **host-only** surface, i.e. a config used in `assumes(...)` (measured
working, J1 — `assumes(x.len() >= cfg.stride())` compiles and is checked, and correctly stays
string-only rather than reaching the prover). So `ir_hash` is a *partial* mitigation, not the fix.

The hole is a strict superset of a known-and-solved problem: `uses(...)` had exactly this shape
("a kernel's `SOURCE_HASH` cannot by itself reflect a change to a helper's body"), solved by
`combine_source_hash` (`contract.rs:226-270`).

**When the hole is absent.** If the pinned expression is a self-contained literal construction *and*
the body only does field access, `SOURCE_HASH` is complete by construction. That is 1.2% of the
corpus (§4.1) — too small to be the v1 boundary, but it is the right *fallback* boundary if the
`vericl::config!` mechanism is ever rejected in review.

### 5.2 The config-method gate hole

`FloatMethodCheck` (`lib.rs:1296-1303`) and `FLOAT_METHOD_REJECT` (`lib.rs:248-265`) walk the
kernel/helper body they are handed. A config method body is a different item.

Measured (`scratchpad/structct/src/bin/gt2.rs`, I3): a `#[cube]` config method whose body is
`fma(v, 2.0, 1.0)` compiles cleanly, and the failure appears only at *runtime*:

```
I3 trap  diff=FAIL n=64: reference execution panicked
         (Unexpanded Cube functions should not be called. ) — divergent semantics or defect
```

Loud, not silent — the harness's `catch_reference_panic` does its job. But it is a runtime failure
where every comparable VeriCL gate is a compile-time rejection with an authored message, and it
depends on the differential lane being run at all.

**And the same name-based gate produces a false positive in the other direction.** A config method
whose name collides with the reject list is rejected with a *wrong* message
(`scratchpad/structct/src/bin/rej.rs`, M4):

```
error: host-callability of `F::dot` in the reference twin is unverified — outside the vericl v0
       subset; verified host-callable Float/Numeric methods are: new, from_int, …
  --> cfg.dot()
```

`dot`, `normalize`, `magnitude` are plausible config-method names; `FloatMethodCheck` is name-based
and cannot see that the receiver is a config.

### 5.3 Unsound pinned expressions

`instantiate(...)` accepts an arbitrary `syn::Expr` with no purity or determinism requirement.
Measured accepted today, ungated:

- `instantiate(cfg = cfg_from_env())` where the fn reads `std::env::args()` — compiles
  (`scratchpad/structct/src/bin/traps.rs`, H2).
- A config method with interior mutability. `scratchpad/structct/src/bin/gt.rs` F8 pins
  `cfg = ImpureCfg` whose `scale()` returns an incrementing counter. The pinned expression is
  evaluated **three times** per run (twin, kernel expansion, `kernel_definition()`), so the twin
  gets 1 and the kernel gets 2:

  ```
  F8 impure  diff=FAIL n=1: 1/1 elements diverge, expected 6.405361 got 12.810722 (8388608 ulp)
             proof=Proved{2}   (CALLS after the run: 3)
  ```

  Caught by the differential lane — as a numeric divergence, not as a diagnosis. Note the proof still
  says `Proved{2}`: the IR is internally consistent, so the *bounds* claim is true of whichever
  variant was expanded. The evidence would be self-consistent and wrong.

There is a third, subtler member of this family that no lane catches: an expression that is
deterministic *within* a process but varies *across* builds (a `const fn` reading `option_env!`, a
value derived from `cfg!(...)`). Both hashes stay put and both lanes pass; only the config-definition
hash of §6 would move, and only if the varying part lives in the config declaration.

---

## 6. The decided design

### 6.1 Shape

Three pieces, no new contract grammar:

```rust
vericl::config! {
    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    pub struct TileCfg { pub m: u32, pub n: u32 }

    impl TileCfg {
        pub fn total(&self) -> u32 { self.m * self.n }
    }
}

#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(max_ulp = 0),
    gen(x in -10.0..=10.0, y in 0.0..=0.0),
    instantiate(cfg = TileCfg { m: 3, n: 8 })     // unchanged grammar
)]
#[cube(launch)]
pub fn scaled(x: &Array<f32>, y: &mut Array<f32>, #[comptime] cfg: TileCfg) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = x[ABSOLUTE_POS] * f32::cast_from(cfg.total());
    }
}
```

### 6.2 Why an item macro and not an attribute

The natural first instinct is `#[vericl::config]` on the type. It does not work: an inherent `impl`
block is a **separate item**, so an attribute on the struct cannot see (hash, or gate) the method
bodies — which is where both defects live. Two impl blocks also cannot each define the same
associated const. `vericl::config! { … }` receives the struct *and* every impl block as one token
stream, which is exactly the unit that determines the kernel's meaning.

It also composes with the existing precedent rather than inventing one:

| Precedent | Annotation required on the dependency | Folded into identity by |
|---|---|---|
| `uses(helper)` | `#[vericl::helper]` | `combine_source_hash(SOURCE_HASH, [helper::identity_hash_at(1)])` (`lib.rs:3434`) |
| `reference = path` | `#[vericl::reference]` | `combine_source_hash(…, [<ref>_vericl_reference_source_hash()])` (`lib.rs:3556`) |
| **struct comptime** | **`vericl::config! { … }`** | **`combine_source_hash(…, [<T as ConfigIdentity>::CONFIG_HASH])`** |

### 6.3 What `vericl::config!` does

1. Re-emits every item **verbatim** (no rewriting — a config is ordinary host Rust and must stay so).
2. Computes `sha256` over the **whole** token block (type + every impl block + attributes/derives).
3. Emits, per declared struct/enum, `impl ::vericl::ConfigIdentity for T { const CONFIG_HASH: &'static str = …; }`.
4. Runs the host-callability check over every method body in the block — the same closed
   `FLOAT_METHOD_REJECT` list plus the free-function forms (`fma`, `cast_from`, …), with a
   config-specific message.
5. Rejects a block with no struct/enum, and (v1) a block declaring a type whose fields are not
   `Hash + Eq`-able — matching cubecl's own hard requirement (§1.1) with an authored message instead
   of a derive-macro error.

### 6.4 Prototype — validated, not proposed

`scratchpad/structct/cfgproto` (proc-macro) + `cfgrt` (trait) + `cfgtest` (driver), built on the
same toolchain (rustc 1.94.0):

| Claim | Measured |
|---|---|
| the config-method edit that leaves `SOURCE_HASH` unmoved moves `CONFIG_HASH` | `sha256:3ada1666…` → `sha256:4b537fa4…`; value `24` → `11` |
| an `fma` in a config method body is rejected at compile time, at the right span | `error: config method body calls \`fma\`, which is not verified host-callable` → pointing at `fma` |
| an undeclared config type gets a targeted message | `error[E0277]: \`Undeclared\` is used as a struct-typed #[comptime] parameter but is not declared with a \`vericl::config!\` block` / `label: not a vericl config type` / note naming the fix |

The undeclared-type message uses `#[diagnostic::on_unimplemented]` on `ConfigIdentity`, which is
stable on the pinned toolchain and renders with the right span and a `help` pointing at the type
definition.

### 6.5 Rejected alternatives

| Alternative | Why not |
|---|---|
| **Richer `instantiate` grammar** (a dedicated struct-literal parser, a `config(...)` clause) | Solves nothing measured. `Expr` already parses every real form (§2, K1–K3), and the ecosystem's dominant construction path cannot be written in an attribute at all (§4.3). New grammar is pure surface area. |
| **Hash the config *value*** (via its `Hash` impl, which cubecl already requires) | Does not close the hole: both §5.1 variants have the identical value `TileCfg { m: 3, n: 8 }`. The definition, not the value, is what moved. |
| **Make `ir_hash` unconditional and call it done** | Necessary but insufficient — it misses the host-only `assumes(...)` surface (§5.1), and it is IR-level while VeriCL's declared identity model is source-level (`contract.rs:117`). Adopted as a *complementary* hardening (M4). |
| **Restrict v1 to fields-only configs with literal pins** (no new machinery needed) | Sound, and the honest fallback, but addresses 1.2% of the corpus (§4.1) and none of the method-chain idiom. Recorded as the retreat position if `vericl::config!` fails review. |
| **`#[vericl::config]` attribute on the type** | Cannot see the impl blocks (§6.2). |
| **Auto-derive `ConfigIdentity` via a blanket impl / no annotation at all** | Then nothing hashes the definition and nothing gates the method bodies — i.e. today's state, which is the defect. |

---

## 7. Identity and hashing treatment

**The pinned expression is already hashed and needs no work.** It lives in the contract attribute
tokens, which `SOURCE_HASH` covers verbatim (`lib.rs:2944-2946`). Measured: pinning `stride: 4` vs
`stride: 5` on otherwise identical kernels gives `0c4aa039…` vs `269d4460…` (§3). Token text is the
right granularity — it is what `Contract::instantiate` already documents ("the instantiation *values*
are already part of `SOURCE_HASH` (they're in the raw contract attribute tokens the hash covers)",
`contract.rs:136-141`). No serialization of an arbitrary struct value is needed, and none would be
better: two spellings of the same value (`TileCfg { m: 3, n: 8 }` vs `default_cfg()`) are
*different contracts* and should hash differently.

**The config definition is the new dependency.** `identity()` gains one dependency expression per
distinct struct-typed comptime param type, in declared parameter order, deduplicated:

```rust
__vericl_id.source_hash = ::vericl::combine_source_hash(
    SOURCE_HASH,
    &[ /* uses(...) helpers */, /* reference fn */,
       <TileCfg as ::vericl::ConfigIdentity>::CONFIG_HASH.to_string() ],
);
```

`combine_source_hash` is order-sensitive by design (`contract.rs:236-270`) and already tested for
determinism and dependency-sensitivity (`contract.rs:308-320`). A config hash is a **leaf** (a
`const`, not a recursive `identity_hash_at`), so it does not interact with `MAX_COMPOSITION_DEPTH`.

**Nested config types.** `StageCfg { tile: TileCfg, … }` — if both are declared in the same
`vericl::config!` block, one hash covers both. If `TileCfg` is declared in a *separate* block, the
kernel's param type is `StageCfg` and only `StageCfg`'s hash is folded, so an edit to `TileCfg`'s
methods is invisible again. **v1 rule: a `vericl::config!` block must declare every config type
reachable from a kernel's comptime param types** — enforced by requiring each type named in a
`#[comptime]` position to have `ConfigIdentity`, and (v1.1) by an optional `deps(TileCfg)` form
inside `vericl::config!` that folds a sibling block's hash. Recorded as a §10.4 deferral with a
targeted error, not as a silent gap.

**Evidence surface.** `ContractRecord` gains no field: the config hash is folded into
`Identity::source_hash`, which `verify()` already compares field-by-field with a rendered
`source_hash X -> Y` diff (`evidence.rs:385-425`). Existing evidence for non-config kernels is
byte-unchanged.

---

## 8. Twin treatment

**The twin needs no changes at all**, and this is the cheapest face VeriCL has ever had — the brief's
prior is correct and is now measured.

A `#[comptime] cfg: T` param is dropped from the twin signature and bound as
`let cfg: TileCfg = TileCfg { m: 3, n: 8 };` at the top of `reference`, `check_assumes`, and each
cooperative phase segment (`lib.rs:3041-3050`, `:3244`, `:3443`). The body tokens are unchanged, so
`cfg.total()` in the twin is the *same host method call* the `#[cube]` side re-emits verbatim
(§1.2). One token stream, one host function, two consumers — the twin and the expansion cannot
disagree about a pure config.

Three consequences worth stating:

1. **No `unexpanded!()` trap for a plain config** — confirmed exhaustively against the real corpus
   (§4.2: 132 config methods, 0 `#[cube]`, 0 device-only calls). The trap is *expressible* but not
   *present*, so v1 gates it at declaration (§6.3 step 4) rather than trusting the corpus.
2. **The twin is where impurity surfaces.** F8 measured the mechanism: the pinned expression is
   evaluated once per consumer, so a non-deterministic config produces a numeric divergence the
   differential lane catches (§5.3). v1 adds a compile-time gate on the *expression form* (§10.3) so
   the common cases fail early with an authored message instead of as an ULP report.
3. **`compare(...)` and `gen(...)` are untouched.** A comptime param is excluded from `gen(...)` by
   construction (`lib.rs:4871`) and from the compared-buffer set. Config-driven *sizes* are a
   different matter and are deferred (§10.4).

---

## 9. Compatibility matrix

Every cell measured unless marked. "PASS" = wgpu/Metal differential green at the listed sizes.

| Feature × struct comptime | v1 | Evidence |
|---|---|---|
| plain field access | **support** | F1 PASS, `Proved{2}` |
| method call, depth 1 | **support** | F3/F5 PASS |
| method call, depth 2–3 | **support** | F2 PASS, IR byte-identical to scalar (§3) |
| enum param + `match` dispatch | **support** | F4 PASS, `Proved{2}`; lowers to a constant, no `Switch` |
| `comptime!` block over a config | **support** | F3 PASS |
| `comptime!` calling a **free fn** on a config (`comptime![packing_mask(scheme)]`) | **reject** (targeted) | M3 measured: `ComptimeRefCheck` rejects the bare `packing` ident. Deferred to v1.1 (§10.4) |
| config as a loop bound | **support** | F5 PASS, `Proved{3}` |
| config in index arithmetic | **support** | L2 `Proved{2}`; L1 correctly `REFUTED` |
| **generic type params** (76% co-occurrence) | **support** | K1 PASS — `instantiate(C = Cfg3, cfg = Cfg3 { s: 5 })` pins both faces |
| **`Vector`/`Line`** (48% co-occurrence) | **support** | J2 PASS at `N = 4` |
| `uses(...)` **composition** | **support** | F6 PASS; helper comptime params stay pass-through (`lib.rs:2413-2419`), caller supplies the pinned value — the ecosystem's 52%-threading shape |
| **cooperative** (`cooperative(cube_dim = N)`) | **support** | I1 PASS at `cube_dim = 256`, n ∈ {256, 1024, 4096}. A comptime value is cube-uniform by construction (`lib.rs:2762-2768`) |
| `SharedMemory` sized from a config | **reject** (targeted) | cubecl needs a literal-or-const size; a config-derived size is expressible but the phase-split twin's tile allocation is pinned by `cooperative(cube_dim)`. Deferred (§10.4) |
| core `Slice` × config | **support** (inherited) | slices are an addressing view; the config contributes constants only (`design-view-slice.md` §5) |
| `assumes(...)` referencing a config | **support, string-only** | J1 PASS; the clause is recorded but not structured — the recognizers take integer literals only (`lib.rs:4284-4286`, pinned by `lib.rs:5999`), a **pre-existing** limitation shared with comptime scalars |
| `gen(...)` sized from a config | **reject** (targeted) | not measured as working; `gen(len(x = n + K))` takes a literal `K`. Deferred (§10.4) |
| config type deriving `CubeType` | **support** | I2 PASS — the const path re-emits the host method regardless |
| config method that is `#[cube]` | **reject** (new gate) | I3: compiles today, fails at runtime with `Unexpanded Cube functions…`. §6.3 step 4 makes it a compile error |
| **runtime `CubeType`/`CubeLaunch` struct arg** | **reject** (unchanged) | separate mechanism (§4.4); today's `gen(...)` message is already correct |
| `&Config` reference param | **reject** (reworded) | M1: rejected today with misleading "must be plain scalar types … Array" text (§10.3) |
| multiple config params (24% of sites) | **support** | `resolve_instantiate` handles N comptime entries; not separately probed beyond 2 |
| `Vec<u32>` / non-`Copy` comptime value | **support** | K3 PASS |
| path-qualified config type | **support** | K2 PASS |
| f64 lane | **support** (inherited) | comptime configs cannot be float-valued (§1.1); the config is orthogonal to output precision |

---

## 10. The v1 subset boundary

### 10.1 Accepted

A `#[comptime] name: T` parameter where:

- `T` is a struct or enum declared inside a `vericl::config! { … }` block (or a generic parameter
  whose `instantiate(...)`-pinned concrete type is), and
- the `instantiate(...)` value is one of the two **pinnable expression forms**:
  - a **literal construction** — a struct/enum literal, unit variant, or nested composition thereof,
    whose every leaf is a literal or a path to a `const`; or
  - a **call to a `const fn`** with literal arguments.

Everything else about the body is already governed by the existing subset: field access, method
calls (any depth), `match`, `if`, `comptime!` blocks, loop bounds, index arithmetic.

The two pinnable forms cover every measured real shape that a launch site can express (§2, F1–F7,
K1–K3) and exclude the §5.3 hazards by construction. `const fn` is admitted deliberately: it is
Rust's own purity guarantee, it matches the `default_cfg()` idiom (F7), and its body is hashed
whenever the fn lives in a `vericl::config!` block.

### 10.2 Rejected, with exact wording

**R1 — a config type not declared with `vericl::config!`** (rustc-mediated, via
`#[diagnostic::on_unimplemented]`; the prototype's measured rendering):

> ``error[E0277]: `TileCfg` is used as a struct-typed #[comptime] parameter but is not declared with a `vericl::config!` block``
> `label: not a vericl config type`
> ``note: wrap the type AND its impl blocks in `vericl::config! { … }` so vericl can fold the config's definition into kernel identity and gate its method bodies for host-callability``

**R2 — a non-pinnable `instantiate(...)` value** (macro-authored, at the value's span):

> ``error: instantiate(...) value for #[comptime] parameter `cfg` must be a literal construction (e.g. `TileCfg { m: 3, n: 8 }`, `Mode::Triple`) or a `const fn` call with literal arguments — `cfg_from_env()` is an arbitrary host expression, which vericl cannot pin: it is evaluated separately for the reference twin and for kernel expansion, so a value that varies between the two (or between builds) makes the recorded evidence describe a kernel that was never run``

**R3 — a `#[cube]` method on a config type** (macro-authored, inside `vericl::config!`, at the
`#[cube]` attribute's span):

> ``error: a `#[cube]` attribute on a vericl config type's impl block is outside the vericl v0 subset — a comptime config's methods run in the reference twin as ordinary host Rust, so the twin would call the host body while the device gets the expanded one; keep config methods plain (the CubeCL ecosystem's own config types are all plain Rust — 132 methods surveyed, 0 annotated `#[cube]`)``

**R4 — a non-host-callable call in a config method body** (macro-authored, at the callee's span; the
prototype's measured form, with the reject-list note added):

> ``error: config method body calls `fma`, which is not verified host-callable — a comptime config's methods run in the reference twin as ordinary host Rust, so every call in them must be host-callable; use the vericl host shim (`vericl::fma`) or compute the value on the host before pinning it``

**R5 — a reference-typed comptime param** (replaces today's misleading text at `lib.rs:2020-2027`;
17 real ecosystem sites use this form):

> ``error: a #[comptime] parameter must be taken by value, not by reference — `&TileCfg` cannot be pinned by instantiate(...) (the pinned expression's lifetime is the twin binding's, not the kernel's); change the parameter to `cfg: TileCfg` (a comptime config is `Clone` by CubeCL's own requirement, so passing by value is free at expansion)``

**R6 — a config method name colliding with the float-method reject list** (replaces the false
positive measured in M4). `FloatMethodCheck` gains receiver awareness: a method call whose receiver
resolves to a `#[comptime]` parameter name is exempt from the name-based reject list (it is gated by
R4 at the declaration instead). If receiver resolution is ambiguous, the existing error is emitted
with an added note:

> `note: if `dot` is a method on the #[comptime] parameter `cfg`, its host-callability is gated where the config is declared (`vericl::config!`), not here — rename the local or file this shape`

**R7 — a config-derived `SharedMemory` size / `gen(...)` length** — deferred forms, §10.4.

### 10.3 Wording and gate corrections landing with v1

Three are pre-existing bugs the milestone must fix because it makes them reachable in practice:

1. `lib.rs:2020-2027` — the comptime type error says "must be plain scalar types … (Array is not
   supported as a comptime parameter)" while only rejecting references. Replace with R5.
2. `lib.rs:2306` — the missing-`instantiate` error suggests `instantiate(F = f32, N = 8)`. Add a
   struct example so a config kernel's first error names the right shape.
3. `FloatMethodCheck` receiver-blindness (R6).

### 10.4 Deferred (v1.1+, rejected with a pointer, not rejected forever)

| Deferral | Why | Measured basis |
|---|---|---|
| `comptime!` calling a **free host fn** on a config (`comptime![packing_mask(scheme)]`) | `ComptimeRefCheck` bans bare non-comptime idents by design; admitting arbitrary free fns needs the same host-callability gate `vericl::config!` applies to methods. Natural v1.1: admit calls to fns declared inside a `vericl::config!` block. | M3; the shape is real (`cubecl-std/src/quant/dequantize.rs:33`) and sole-blocks 9 items / 3 plain fns once struct comptime is off the gate list (§11) |
| Cross-block nested config types (`deps(...)`) | §7; v1 requires one block per reachable type | — |
| `SharedMemory::new(cfg.tile_size())` | The phase-split twin's tile is pinned by `cooperative(cube_dim)`; a config-derived size needs a second pin | not probed as working |
| `gen(len(x = cfg.k()))` | The `gen` length forms take literals | not probed as working |
| Config-driven `assumes` recognized *structurally* (so `y.len() + cfg.k() <= x.len()` reaches the prover) | The recognizers take integer literals (`lib.rs:4284-4286`, pinned by `lib.rs:5999`) — a **pre-existing** limit shared with comptime scalars, not new here. Fixing it means substituting pinned comptime values into `assumes` tokens before recognition, which is only possible for literal-valued expressions. | J1 (string-only, measured); L3 shows the literal/config drift hazard this would remove |
| A config value derived from device properties (`comptime::device_properties()`) | Structurally out: VeriCL's zero-client IR extraction leaves `Scope.properties: None` (`docs/ir-research.md` §1) | §4.3 |
| Runtime `CubeType`/`CubeLaunch` struct args | Separate mechanism (§4.4), separate milestone | H5, §11 |
| `#[cube]` items that are `impl`/`trait` members | `#[vericl::kernel]`/`#[vericl::helper]` parse `ItemFn` (`lib.rs:2518`, `:3528`). This is what actually gates the 38 (§4.6) | §11 |

---

## 11. Coverage projection — measured, not estimated

The survey classifier's rule is "any `#[comptime] name: T` with `T` outside
`{u8…usize, i8…isize, bool, f32, f64, char}` → **blocking**" (`classify.py:514-517`). §2 measures
that rule to be **wrong**: the shapes it flags compile, pass the differential, and prove.

Re-run with that row demoted from `blocking` to `supported`
(`scratchpad/structct/classify_nostructct.py`, everything else byte-identical):

| Bucket | with the false gate | without it | Δ |
|---|---:|---:|---:|
| Items tripping zero blocking gates (v1-full) | 51 | **89** | **+38** |
| …with the extra return/unbounded-loop screen | 49 | **87** | +38 |
| …of which **plain non-test `fn`** | 12 | **12** | **0** |
| …of which impl/trait non-test | 33 | 71 | +38 |
| Items with exactly one blocking gate | 127 | 158 | +31 |

The `+38` matches the recorded sole-blocker count exactly — a clean consistency check.

**Sole-blocker frontier, before → after:**

| Gate | items | sole (before) | sole (after) | sole non-test `fn` (after) |
|---|---:|---:|---:|---:|
| **custom `CubeType` param (broad)** | 141 | 8 | **28** | **28** |
| View/Layout machinery | 110 | 45 | **57** | 0 |
| `comptime_type!` | 53 | 4 | **18** | 0 |
| `plane_*` | 88 | 2 | **14** | 2 |
| `comptime!{}` out of subset | 71 | 2 | **9** | **3** |
| `CubeType`-arg (v0 name list) | 68 | 8 | 8 | 1 |
| cmma / `Matrix` | 62 | 6 | 6 | 0 |
| `Slice`/`SliceMut` ident | 21 | 4 | 6 | 0 |
| `Vector` shape out of subset | 43 | 2 | 4 | 0 |
| `SharedMemory` 4 · 2-D 1 · `intrinsic!` 2 · non-f32 `cast_from` 1 | | | unchanged | |

**Honest reading.**

1. **This milestone unlocks 38 ecosystem items and zero plain functions.** All 38 are impl blocks
   (29) or trait definitions (9) that VeriCL's `ItemFn`-based macros structurally cannot annotate
   (§4.6, §10.4). Anyone reading "243 items, the single largest blocking gate" as a reach number is
   reading the wrong number — the same lesson the re-census taught about `match` (119 items,
   0 unlocked) and `plane_*` (88 items, 2 sole).
2. **The real deliverable is soundness and claimability, not reach.** An ungated accidental capability
   with a silent identity hole is worse than an unsupported one: it produces evidence that looks
   valid and can be stale. The private dogfood corpus does not exercise struct comptime at all (its
   ranked walls are `fma`, `cast_from` sources, tuple `wrapping`, an injectivity assume, 2-D
   dispatch), so this is purely an ecosystem-facing and future-proofing milestone.
3. **`custom CubeType param (broad)` is now unambiguously the next milestone**: 28 sole-blocker,
   **all 28 plain non-test `fn`s** — a 3.5× jump, and the only bucket in the table that is plain
   functions. §4.4 scopes it as genuinely separate work.
4. **Two smaller co-gates become worth their own line items**: `comptime!{}` out of subset
   (2 → 9 sole, 3 plain fns), whose dominant unsupported form is the free-fn-in-`comptime!` shape
   (§10.4), and `plane_*` (2 → 14 sole, 2 plain fns).

---

## 12. Implementation plan (agent-sized milestones)

Each milestone is independently verifiable and leaves the tree green.

**M1 — `vericl::config!` + `ConfigIdentity`.**
New `vericl::config!` proc macro in `vericl-macros` (function-like, alongside `suite!`), new
`ConfigIdentity` trait in `crates/vericl/src/contract.rs` with `#[diagnostic::on_unimplemented]`.
Emits items verbatim + one `impl ConfigIdentity` per declared struct/enum, hashing the whole token
block. Port the prototype at `scratchpad/structct/cfgproto/src/lib.rs`.
*Verify:* unit tests that (a) two blocks differing only in a method body produce different
`CONFIG_HASH`; (b) a whitespace-only difference produces the same hash as the token stream
(document the granularity either way); (c) a block with no struct/enum errors; (d) the emitted items
are token-identical to the input.

**M2 — config method-body gate (R3, R4).**
Run the host-callability check over every method body in a `vericl::config!` block; reject `#[cube]`
on any impl inside it.
*Verify:* the I3 shape (`fma` in a config method) now fails at **compile** time at the callee's span,
with the R4 wording; a `#[cube] impl` fails with R3; a plain config with 10 ordinary methods compiles
unchanged. Negative control: temporarily remove the check and confirm I3 reverts to the measured
runtime `Unexpanded Cube functions…` twin panic.

**M3 — kernel-side wiring (R1) + identity folding.**
`expand` collects the distinct struct-typed comptime param types (after `instantiate(...)`
substitution, so a generic `C` resolves to its pinned concrete type), emits a
`<T as ConfigIdentity>::CONFIG_HASH` reference per type, and folds them into `identity()` via
`combine_source_hash` alongside the `uses(...)` and `reference` deps.
*Verify:* the §5.1 A/B — the config-method edit that today leaves `SOURCE_HASH` at `dd3d0579…` must
now move `identity().source_hash`, and `vericl::suite!` must report
`STALE evidence — identity mismatch (source_hash X -> Y)`. A non-config kernel's `identity()` must be
**byte-identical** to today's (`combine_source_hash` is pass-through with no deps, `contract.rs:308`);
re-run the full example suite and confirm `evidence/vericl.json` is unchanged.

**M4 — pinnable-expression gate (R2) + `ir_hash` hardening.**
Classify the `instantiate(...)` value for a struct-typed comptime param as literal-construction /
`const fn` call / other; reject "other" with R2. Separately, populate `Identity::ir_hash`
unconditionally (it needs only `kernel_ir_hash`, not z3), so a `prove: false` suite still carries
IR-level identity.
*Verify:* H2 (`cfg_from_env()`) and F8 (`ImpureCfg` — its unit-struct pin is literal, so gate on the
*method* purity via M2 and record the residual) produce R2 or are documented as residual; F1–F7 and
K1–K3 all still compile. Confirm the `ir_hash`-unconditional change moves no existing verdict and
that old manifests still load (`#[serde(default)]`).

**M5 — wording and gate corrections (R5, R6, §10.3).**
Replace the comptime-type error text; add receiver awareness to `FloatMethodCheck`; extend the
missing-`instantiate` suggestion.
*Verify:* M1 (`&Cfg`) emits R5; M4-collision (`cfg.dot()`) compiles; a genuine `x.dot(y)` on a float
still rejects; the existing `float_method_whitelist{,_f64}.rs` tests are unchanged.

**M6 — public surface: suite-wired examples + tests.**
Add to `crates/vericl-examples`: a fields-only config kernel, a depth-2 method-chain kernel, an
enum-dispatch kernel, a config-driven loop-bound kernel, a `uses(...)`-composed config kernel, and a
cooperative × config kernel — all wired into `vericl::suite!` with `tested` + `proved` on both lanes.
Plus non-suite negative controls mirroring L1/L3/I3/H2.
*Verify:* `cargo test` green on wgpu and `--features cpu`; new evidence entries carry the folded
config hash; the pre-existing 8+ evidence entries are byte-identical (per-entry canonical-JSON
SHA-256, the re-census's own method).

**M7 — docs + re-census.**
`docs/guide.md` section; README subset table row; re-run `classify.py` with the struct-comptime row
demoted and record the measured delta (§11) in `docs/ecosystem-survey-2026-07.md` as a correction to
the re-census, explicitly flagging that the 243/38 row measured a gate that did not exist.
*Verify:* the guide's example compiles as a doctest; the recorded numbers reproduce.

---

## 13. Open risks, ranked (pre-registered for review round 10)

**Round 10's scope is this milestone *plus* the already-landed shim-and-small-gate batch**
(`fma` host shim, `CastToF32` source extension, tuple-returning `wrapping`, the `checked_mul` prover
diagnostic — `tasks/todo.md`, "Shim-and-small-gate batch — DONE 2026-07-27"), the same "one review
over two related batches" pattern round 7 used for quick-wins 1+2.

1. **The pinnable-expression gate is a recognizer, and a recognizer defect is critical-class.**
   R2 decides which expressions are honest. If it admits something impure (a `const fn` that is
   `const` only by name and reads `option_env!`; a path to a `static mut`; a literal construction
   whose leaf is a `const` that itself came from a build script) the evidence is self-consistent and
   wrong — F8 shows the failure mode, and note that its proof still reported `Proved{2}`. **Attack
   surface:** every "literal construction" leaf form. The round-4 recognizer lesson applies verbatim:
   the recognized form must *imply* the claimed property. Mitigation to review: make the classifier
   strict-by-construction (an explicit allowlist of leaf `Expr` variants) and treat anything
   unclassified as R2, never as accepted.
2. **`vericl::config!` hashes token text, so it is blind to what the block does not contain.**
   A config method calling a *free function* defined outside the block (`fn packing(c: Cfg) -> u32`)
   is neither hashed nor gated. This is the exact `uses(...)` problem one level down. v1 must either
   reject a config method body containing any call to a path not declared in the block, or state the
   residual loudly. **Currently unresolved — this is the sharpest open question for review.**
3. **The `#[cube]`-config-method gate is name-based and can be evaded.** R3 checks for a `#[cube]`
   attribute on impl blocks *inside* the `vericl::config!` block. A second impl block outside it
   (`impl TileCfg { #[cube] fn sneaky() }`) is invisible to both the hash and the gate. Rust allows
   inherent impls for a local type anywhere in the crate. Mitigation: none clean at macro scope;
   candidate is a `#[deny]`-style lint or accepting the residual with the I3 runtime failure as the
   backstop. **Pre-registered as a known limit, not as solved.**
4. **Folding `CONFIG_HASH` changes `identity()` for every config kernel — verify it changes nothing
   else.** `combine_source_hash` is pass-through with no deps, so non-config kernels must be
   byte-identical. A regression here silently invalidates every stored manifest. M3's verification
   step is the guard; review should re-derive it independently.
5. **cubecl's persistent on-disk kernel cache keys on the config's `Hash` *and* `Debug`**
   (`cubecl-runtime-0.10.0/src/id.rs:119-141`). A config type with a nondeterministic `Debug` (a
   pointer, a `HashMap` iteration order) poisons that cache across processes — a *cubecl* correctness
   issue VeriCL inherits and does not control. v1 should note it; a derive-only `Debug` requirement
   is a candidate gate.
6. **The `assumes`-literal / config drift hazard (L3) is real and stays open in v1.** A kernel pinned
   `stride: 5` alongside `assumes(y.len() + 4 <= x.len())` is a contract lie the *recognizers* cannot
   see. Both lanes caught it here (REFUTED + differential FAIL), which is the system working — but
   only because the mismatch was large enough to go out of bounds. A drift in the safe direction
   (assuming `+ 8` while pinning `5`) is a weaker-than-necessary claim nothing flags. §10.4's
   structured config-assume is the fix; review should decide whether it belongs in v1.
7. **Reference-typed comptime params are 17 real sites and v1 rejects them.** R5's rationale (the
   pinned expression's lifetime) should be pressure-tested: `&'static` config references may in fact
   be pinnable, in which case R5 is over-strict and should become a `&'static`-only acceptance.
8. **Receiver-aware `FloatMethodCheck` (R6) risks the opposite error.** Exempting method calls whose
   receiver is a comptime param name could exempt a genuine float method if a comptime param shadows
   or aliases. The existing `check_instantiate_local_collisions` machinery (`lib.rs:1055`) is the
   precedent for getting this right; review should attack the shadowing cases.
9. **The measured "it already works" table is 14 shapes, not a proof of totality.** Multi-config
   kernels beyond 2 params, depth-3 chains, enums *with data* in `match` arms
   (`GlobalOrder::SwizzleRow(w)`), and `Option`-typed configs were not probed end to end. Review
   should pick two and run them.
10. **Coverage honesty.** §11's headline is "+38 items, 0 plain functions". If that reads as a
    disappointing milestone, the temptation is to quote 243. The doc must not, and review should
    check that no downstream artifact does.

---

## 14. Roadmap impact

- **Struct-typed `#[comptime]` params leave the frontier ranking** — not because they were unlocked,
  but because they were **never a gate**. The re-census's #1 row measured the classifier, not VeriCL.
  §11's re-run is the correction; `docs/ecosystem-survey-2026-07.md` should carry it.
- **The recorded post-re-census ranking (1) struct comptime, (2) View/Layout, (3) `CubeType` args
  is superseded.** Measured, with the false gate removed: **(1) custom `CubeType` struct args —
  28 sole, all 28 plain non-test `fn`s**; (2) View/Layout — 57 sole, 0 plain fns; (3) `comptime_type!`
  — 18 sole; (4) `plane_*` — 14 sole, 2 plain fns; (5) `comptime!{}` out of subset — 9 sole, 3 plain
  fns. For *annotatable plain functions*, `CubeType` args are now 9× the next-best bucket.
- **The impl/trait item wall is the real ceiling.** 71 of the 89 gate-free items — and every one of
  the 38 this milestone touches — are `impl`/`trait` members that `#[vericl::kernel]`/
  `#[vericl::helper]` cannot be applied to. Whether to grow `ItemImpl`/`ItemTrait` support is now the
  single largest open roadmap question, and it is orthogonal to every construct-level gate remaining.
- **This milestone's value is measured in soundness, not items.** It converts an ungated accidental
  capability — with a demonstrated identity hole, a demonstrated compile-time gate hole, and a
  demonstrated unsound-pin surface — into a claimed, tested, identity-covered one, and it does so
  with one new macro, one new trait, and zero changes to the twin, the prover, or the IR.
