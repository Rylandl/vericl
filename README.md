# VeriCL

> One kernel contract. Equivalent implementations.

VeriCL is a conformance and evidence harness for [CubeCL](https://github.com/tracel-ai/cubecl)
compute kernels. You write a kernel once in Rust, attach the assumptions and properties that
matter, and VeriCL produces the artifacts and checks needed to support clearly bounded claims
about it: differential test results across backends, machine-checked safety properties, and an
evidence manifest that goes stale when anything it depends on changes.

> **New to VeriCL? Start with the [user guide](docs/guide.md).** It walks a competent Rust/GPU
> developer from a plain CubeCL kernel to `cargo test`-verified evidence in under ten minutes of
> reading — installation, annotating a first kernel, the `suite!` block, the `VERICL_UPDATE`
> workflow, reading an evidence file, what each rejection means, and an honest "what VeriCL does not
> do". This README is the design record and changelog; the guide is the manual.

## What kernels can VeriCL verify today?

Bring it if it is a **1-D elementwise, gather, stencil, RNG/hash, or shared-memory tree-reduction**
kernel over `Array<T>` — those are supported, exercised, and carry committed evidence. Vectorized
`Vector<P, N>` elementwise, `cube_struct!`/`config!` struct arguments, and f64 work with caveats worth
reading first. Image-space **2-D/3-D dispatch** — elementwise, transpose, branch-free clamped
stencils — is supported behind a `dispatch(...)` clause, with a narrower boundary than "2-D works"
suggests (read the row). **Do not bring it yet** if it needs 2-D *shared-memory tiles*, atomics, or
`plane_*` subgroup reductions — those are rejected at compile time today and are the next three
milestones. `Tensor`/`View`/`cmma` tiling is deliberately out of scope, with reasons.

**[docs/coverage.md](docs/coverage.md) is the per-kernel-class matrix** — differential-tested,
bounds-proved, race-proved, and status for each class, every cell cited to a real example or test,
with the caveats on the row rather than in a footnote. It also carries the gap-closure plan. Read it
before you spend an afternoon annotating a kernel VeriCL will reject.

## Status

Exploratory. This document records the problem, the design decisions that are locked in, and the
scope of the first release. Decisions marked **open** are genuinely undecided; everything else is
settled unless the first release proves it wrong. The original backend-neutral ideation charter is
archived at [docs/ideation-charter.md](docs/ideation-charter.md).

## Problem

Accelerated kernels are hard to trust for reasons beyond the arithmetic in their bodies: indexing
and layout conventions differ between implementations, boundary behavior is implicit, parallel
execution introduces collisions and ordering differences, optimizations change numerical behavior,
and reference implementations drift away from the accelerated code they supposedly describe. Tests
demonstrate selected cases without explaining the scope of the guarantee, and formal results can
prove a model without establishing that deployed code implements it.

The usual failure mode is not a wrong artifact but silent disagreement between artifacts that each
look reasonable in isolation. VeriCL keeps the kernel's intended behavior, its executable
realizations, and the evidence about them mechanically connected, so that disagreement is detected
instead of accumulated.

## Locked decisions

| Decision | Choice |
|---|---|
| Implementation language | Rust |
| Kernel framework | CubeCL (`#[cube]` kernels) |
| Authoring experience | Plain CubeCL kernels plus a `#[vericl(...)]` attribute for contracts — no new notation |
| Point of custody | The annotated Rust kernel function; every other artifact is derived from or checked against it |
| Kernel identity | Content hash of the expanded CubeCL IR plus the contract plus the toolchain versions |
| Independent comparison | Scalar CPU reference execution derived from the same kernel definition, differentially tested against GPU runs |
| First machine-checked property | Out-of-bounds freedom for a supported kernel subset, discharged by an SMT solver over the CubeCL IR |
| Numerical stance (v1) | Exact comparison for integer kernels; floating-point kernels declare a per-kernel tolerance that is recorded as an assumption in the evidence |
| Evidence format | A manifest binding every result to the kernel identity it was produced from; both human- and machine-readable |
| CI story | Conformance runs under plain `cargo test` (`vericl::suite!` generates the test); `VERICL_UPDATE=1 cargo test` regenerates evidence. A standalone `vericl check` CLI is future work — the `cargo test` path fully covers "fails on missing, stale, or mismatched evidence" for v0 |

### Why CubeCL

A `#[cube]` kernel is written in a subset of Rust whose semantics parallel ordinary Rust. That
makes the central idea concrete instead of aspirational: the kernel function itself is the single
point of custody, and a scalar reference implementation can be derived from the same definition
rather than hand-maintained alongside it. CubeCL also compiles one kernel through its own IR to
multiple backends (wgpu/WGSL, CUDA, ROCm/HIP, SPIR-V), so cross-target differential comparison
falls out of the design rather than being engineered per backend. Its IR is accessible from Rust,
which gives static checking a well-defined foundation.

The cost is coupling to a young, fast-moving project. Mitigations: pin the CubeCL version,
isolate all IR-facing code in one crate, and treat "survives a CubeCL upgrade" as a recurring
health check rather than a surprise.

### The contract attribute (implemented)

```rust
#[vericl::kernel(
    assumes(
        x.len() == y.len(),
        alpha.abs() <= 4.0,
        x.iter().all(|v| v.abs() <= 100.0),
        y.iter().all(|v| v.abs() <= 100.0)
    ),
    compare(abs = 1e-4),
    gen(alpha in -4.0..=4.0, x in -100.0..=100.0, y in -100.0..=100.0),
    instantiate(F = f32)
)]
#[cube(launch)]
pub fn axpy<F: Float + CubeElement>(alpha: F, x: &Array<F>, y: &mut Array<F>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
    }
}
```

From this single definition VeriCL derives, in a generated `axpy_vericl` module: the untouched
CubeCL kernel; a sequential scalar `reference` twin (`ABSOLUTE_POS` becomes a loop variable,
`&Array<T>` becomes `&[T]`, and — per `instantiate(...)` below — the generic type parameter is
substituted to its pinned concrete type) sharing no CubeCL machinery; the `assumes` clauses as an
executable `check_assumes` predicate; a `SOURCE_HASH` identity that evidence binds to; and — from
the `gen(...)` clause — a `conformance_case` function that generates inputs, runs the reference and
the real kernel, and compares them, so no kernel needs hand-written GPU launch/input-gen glue.
Kernels using constructs the twin cannot model (`UNIT_POS`, `SharedMemory`, `plane_*`, vectors,
`return`) are rejected at compile time rather than silently approximated. A `comptime! { … }` block
is evaluated at expansion when it depends only on `#[comptime]` parameters + literals, and rejected
by name otherwise (see "comptime! block evaluation" below). Kernel *composition* — calling another
`#[cube]` fn — is supported via `#[vericl::helper]` and a kernel-side `uses(...)` clause; see "Kernel
composition" below.

### The `instantiate(...)` clause: monomorphizing generic + `#[comptime]` kernels

Real CubeCL kernels are overwhelmingly generic over their element type (`<F: Float>`) and use
`#[comptime]` parameters for unroll counts, tap counts, and feature toggles — a July 2026 dogfooding
survey against a private 22-kernel production codebase found generics blocking 20/22 kernels and
`#[comptime]` blocking 15/22 (see `docs/dogfood-2026-07.md`). `instantiate(...)` names a concrete
value for every generic type parameter and every `#[comptime]` parameter the kernel declares —
`instantiate(F = f32, taps = 3)` — and VeriCL monomorphizes everything it derives at those values:

- **Reference twin**: the generic type ident is substituted token-wise wherever it appears in the
  twin's signature and body (`F` -> `f32`); `#[comptime]` parameters are removed from the twin's
  signature entirely and instead bound as `let name: ty = value;` consts at the top of `reference`
  (before the `ABSOLUTE_POS` loop — they're loop-invariant by construction) and `check_assumes`.
  The perf-only `#[unroll]`/`#[unroll(n)]` statement attribute is stripped from twin loops (it isn't
  valid plain Rust); any *other* statement attribute is a compile error, not a silent drop.
- **`conformance_case`**: launches via `<name>::launch::<f32, R>(...)`, with `#[comptime]` values
  spliced in at their declared parameter position — CubeCL keeps a comptime param in its original
  position with its plain type, it's only non-const params that get wrapped for the runtime.
- **`kernel_definition()`** (the IR the SMT prover and `ir_hash` see): calls the CubeCL-generated
  `expand::<f32>(...)` with the same turbofish and comptime values, exactly mirroring a real call
  site.
- **Contract identity**: instantiation values are part of the raw contract attribute tokens, so
  `SOURCE_HASH` already changes when they change; `Contract`/`ContractRecord` additionally record
  the pinned values as strings (`instantiate: ["F = f32", "taps = 3"]`) purely for evidence
  legibility.

A kernel with generic type parameters and/or `#[comptime]` parameters and **no** `instantiate(...)`
clause is a targeted compile error telling you to add one; an `instantiate(...)` clause on a kernel
with neither is also an error (an unused instantiation is a contract lie). v0 supports exactly one
`instantiate(...)` clause per kernel — multiple instantiations of the same kernel body is future
work — and only plain type generic parameters (no lifetimes, no const generics, no where-clauses).

**Float-method host-callability.** After substitution the twin's body may call `Float`/`Numeric`
trait methods (`F::new(x)`, `x.sqrt()`, ...) resolved through `cubecl::prelude`'s traits. Most of
these are safe to call on the host: either they have a real per-type implementation (`Float::new`)
or they share a name with a `std` `f32` inherent method, which Rust's method resolution always
prefers over a trait method regardless of which traits are `use`-imported. A few are *not* safe —
`log1p`, `inverse_sqrt`, `erf`, and `is_inf` have no such shadow and panic
(`Unexpanded Cube functions should not be called.`) if called on the host at all. VeriCL verified
this empirically (`crates/vericl-examples/tests/float_method_whitelist.rs` calls every candidate
method on `f32` and either cross-checks it against `std` or confirms it panics) and rejects, at
macro time, any twin body calling a method outside the verified whitelist:
`error: host-callability of 'F::erf' in the reference twin is unverified — outside the vericl v0
subset`. This is an explicit rejection, not a best-effort attempt — a twin that silently miscomputes
or panics on a method vericl never verified is exactly the failure mode this project exists to
prevent.

The whitelist was re-verified **on `f64`** the same empirical way
(`crates/vericl-examples/tests/float_method_whitelist_f64.rs`) rather than assumed to transfer from
`f32` — cubecl's `Float`/`Numeric` trait impls could in principle differ per type. Result: every
whitelisted method is host-callable and numerically correct on `f64`, and every rejected method
panics on `f64`, exactly as on `f32`, so a single shared whitelist stays correct (no per-type
split needed). The reason is the same: for a *concrete* `f64` receiver Rust prefers the inherent
`f64::method` over the trait's `unexpanded!()` default, and the associated fns (`new`, `from_int`,
`min_value`, `max_value`) have real per-type `f64` impls.

### Struct-typed `#[comptime]` parameters: `vericl::config! { … }`

A `#[comptime]` parameter's type does not have to be a scalar. CubeCL lets a `#[cube]` item take
`#[comptime] cfg: SomeConfig` for a user struct or enum, and re-emits `cfg.field` / `cfg.method()`
as **plain host Rust executed while the IR is built** — so the config never reaches the device, only
the constants it computes do. This is the CubeCL ecosystem's dominant configuration idiom (243 of
464 surveyed items; `docs/ecosystem-survey-2026-07.md`).

VeriCL accepted that shape before this milestone — but *ungated and unclaimed*, with three measured
defects (`docs/design-struct-comptime.md` §5). The decisive one: a config type's **definition** is in
neither input of `SOURCE_HASH` (the kernel's own tokens; the contract attribute tokens), so editing a
config method body from `self.m * self.n` to `self.m + self.n` changed the kernel from ×24 to ×11 and
left the recorded identity **bit-identical**. Stored evidence stayed "fresh" while describing a
different kernel.

The declaration form closes it. Wrap the config type **and every one of its impl blocks** in one
`vericl::config!` invocation:

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
    instantiate(cfg = TileCfg { m: 3, n: 8 })        // unchanged grammar
)]
#[cube(launch)]
pub fn scaled(x: &Array<f32>, y: &mut Array<f32>, #[comptime] cfg: TileCfg) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = x[ABSOLUTE_POS] * f32::cast_from(cfg.total());
    }
}
```

An item macro rather than an attribute for a structural reason: an inherent `impl` is a *separate
item*, so `#[vericl::config]` on the type could not see — let alone hash or gate — the method bodies,
which is where the defects live.

**What the declaration buys.**

- **Identity.** The macro hashes the whole token block into
  `impl vericl::ConfigIdentity for TileCfg { const CONFIG_HASH }`, and the kernel folds that const
  into `identity()` via `combine_source_hash` — the same treatment `uses(...)` gives a helper body
  and `reference = path` gives a declared reference. A config method-body edit now makes the stored
  evidence correctly stale.
- **A required declaration, not an optional one.** A struct-typed `#[comptime]` parameter whose type
  is *not* declared with `vericl::config!` no longer compiles:
  `` error[E0277]: `TileCfg` is used as a struct-typed #[comptime] parameter but is not declared with
  a `vericl::config!` block `` — via `#[diagnostic::on_unimplemented]`, with the label on the
  parameter and a help pointing at the type definition.
- **Gated method bodies.** Every body in the block is checked for host-callability with the same
  closed reject list the kernel body gets, so an `fma` in a config method is a **compile** error at
  the callee's span instead of a run-time twin panic. `#[cube]` anywhere in the block is rejected
  outright.
- **A pinnable pinned expression.** The `instantiate(...)` value for a config parameter must be a
  literal construction (a struct/enum literal, a path to a `const`) or a `const fn` call. Two halves:
  a strict-by-construction syntactic allowlist with a VeriCL-authored message, plus a generated
  `const` binding of the value at its own type, so rustc must const-evaluate it —
  `instantiate(cfg = cfg_from_env())` is `error[E0015]: cannot call non-const function
  'cfg_from_env' in constants` at the value's span. Without that gate the pinned expression is
  evaluated once per consumer (twin, `expand()`, IR extraction), and an impure one makes them
  disagree: measured, an incrementing counter gave the twin `1` and the kernel `2` — an 8388608-ULP
  divergence whose *proof* still said `Proved{2}`, because the IR was internally consistent with
  whichever variant was expanded.

**The subset, exactly.** Accepted: field access, method calls at any depth, `match`/`if` on a
comptime scrutinee, `comptime!` blocks over a config, config-driven loop bounds and index
arithmetic, nested config types (declared in the same block), `uses(...)` composition, generics,
`Vector`, cooperative kernels, and a type alias for a scalar in `#[comptime]` position. Rejected,
each with a targeted message: a config type not declared with `vericl::config!`; a non-pinnable
`instantiate(...)` value; `#[cube]` on a config impl; a non-host-callable call in a config method; a
**reference**-typed `#[comptime]` parameter (it must be taken by value); a generic config type; a
field or **return type** that is neither a scalar primitive nor declared in the same block; a call or
associated-item read that resolves *outside* the block in any syntactic form — `helper(x)`,
`Self::helper(x)`, `self.helper()`, `Self::K`, or a user extension trait's `self.m.boost()`; an
**impure** reach (`std::env`, `std::process`, `std::time`, `std::fs`, `std::io`, `rand`-likes, and
`std::mem` for target-dependence); a **custom derive**; a `use` that rebinds `core`/`std`/`alloc`/a
primitive name, or a glob `use`; and, inside the block, a `static`, a `mod`, or any macro invocation
— including the `macro_rules!`-generated config families CubeCL's own `cubek-std` uses, which must be
written out so that what is hashed is what the type actually is. The prover is untouched: a
struct-comptime kernel's IR is *byte-identical* to the same kernel written with plain comptime
scalars.

**Two limits worth knowing before you start.** A config type must be *declared* inside a
`vericl::config!` block, so a **third-party** config type is inexpressible in v1 — Rust's orphan rule
would let you write `impl ConfigIdentity for TheirCfg`, but VeriCL never emits a bare impl, because a
hash over tokens you did not write certifies nothing and the gates would have nothing to walk. The
workaround is a clean-room port of the parts you use (that is exactly what the ecosystem
spot-validation does with `cubek-std`'s `TileSize`). And `const`-evaluable **does not mean
source-determined**: a pin derived from `option_env!`/`cfg!` is const-evaluable and can still differ
between two builds of identical source. VeriCL hashes the pin's *expression text* (it is inside the
contract-attribute tokens), not the environment that resolved it, so such evidence is per-build
deterministic rather than per-source reproducible — `ir_hash` under `prove: true` is what catches the
drift. If cross-build reproducibility matters, write the value out.

**The residual, stated plainly.** Rust allows an inherent `impl` for a local type anywhere in the
crate, so a second impl block written *outside* the `vericl::config!` invocation escapes both the
hash and the gates. There is no fix at macro scope. It is accepted because it fails loudly — the twin
panics with `Unexpanded Cube functions should not be called.`, which the differential harness catches
and reports, and a config-derived value that reaches the device still moves `ir_hash` — and both
halves are pinned by tests
(`crates/vericl-examples/tests/config_out_of_block_backstop.rs`), including one assertion whose whole
job is to state the residual so it cannot be quietly forgotten.

**Honest reach.** This milestone's value is soundness, not coverage. Re-running the ecosystem
classifier with the (measured non-existent) struct-comptime gate removed moves gate-free items from
51 to 89 — and plain non-test functions from 12 to **12, unchanged**: all 38 are `impl` blocks or
`trait` definitions, which `#[vericl::kernel]`/`#[vericl::helper]` (both `ItemFn`-based)
structurally cannot annotate.

### Runtime `CubeType` struct parameters: `vericl::cube_struct! { … }`

The other parameter position. CubeCL also lets a `#[cube]` item take a **runtime** (non-`#[comptime]`)
struct — `args: &MyStruct` where `MyStruct` derives `CubeType`/`CubeLaunch` — and lowers it as a
**positional flattening of its fields** at that parameter's own slot, in field declaration order.
That is measured three independent ways (`docs/design-cubetype-args.md` §2): the GPU output is
bit-exact against the same kernel with the fields spelled as loose parameters, the `KernelDefinition`
agrees buffer-for-buffer and scalar-for-scalar, and `kernel_ir_hash` is byte-identical. **The prover
needs zero changes**, and the equality is re-asserted in-repo as a cubecl-upgrade tripwire
(`tests/cube_struct_identity.rs::struct_and_flattened_spellings_have_identical_ir`).

VeriCL accepted half of this shape before the milestone — the **helper** half — with *no diagnostic
at all* and its definition in *no hash*. With a `#[cube] impl Pair { fn fold }` edited from
`self.a * self.b` to `self.a + self.b`, the reference twin went from `[3, 6, 9, 12]` to
`[4, 5, 6, 7]` while the kernel's `SOURCE_HASH`, the helper's `SOURCE_HASH` **and**
`identity().source_hash` all stayed bit-identical: evidence recorded against the first build verified
FRESH against the second. There is a second, launch-side hazard in the same family —
`<Name>Launch::new` fills fields **by position**, so swapping two same-typed fields in the
*declaration* changed the computed function with the kernel body and the launch-call text
byte-unchanged.

The declaration form closes both:

```rust
vericl::cube_struct! {
    pub struct UniformArgs {
        pub lower_bound: f32,
        pub upper_bound: f32,
    }
}

#[vericl::kernel(
    assumes(s.len() == y.len(), args.lower_bound.abs() <= 100.0),
    compare(abs = 1e-4),
    gen(args.lower_bound in -100.0..=100.0, args.upper_bound in -100.0..=100.0, y in 0.0..=0.0),
    uses(to_unit_interval)
)]
#[cube(launch)]
pub fn uniform_value_map(s: &Array<u32>, args: &UniformArgs, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        let scale = args.upper_bound - args.lower_bound;
        y[ABSOLUTE_POS] = to_unit_interval(s[ABSOLUTE_POS]) * scale + args.lower_bound;
    }
}
```

Note what is *not* written: the `#[derive(CubeType, CubeLaunch)]`. The macro owns the derive set,
because an author-chosen one is a silent capability switch — dropping `CubeLaunch` turns the type from
launchable to device-local with every kernel's tokens unchanged. It also emits `Clone`/`Copy`, since
the generated twin binds the struct by value.

**The twin is your own struct.** A `#[derive(CubeType)]` struct of scalars is still an ordinary Rust
struct, so there is no generated mirror type: the twin takes `args: UniformArgs`, `args.lower_bound`
is ordinary field access, and the body tokens reach the twin unmodified. One token stream, two
consumers — the same property `#[vericl::helper]` and `vericl::config!` rest on.

**The v1 boundary.** A field must be one of:

| field kind | admitted | why |
|---|---|---|
| runtime scalar | `f32` `f64` `u32` `i32` `u64` `i64` | generated exactly as a loose scalar parameter of that type is; `usize`/`bool` are comptime-only because there is no scalar draw for them |
| nested struct | any struct declared in the **same** block | one block is one `STRUCT_HASH`, so a sibling-block type would contribute meaning without contributing to identity |
| `#[cube(comptime)]` | integer, `bool`, `char`, a unit enum declared in the same block, or another declared struct whose fields are all of those (pinned **whole**) | it keeps its positional launch slot but takes the plain host type; no float anywhere in the shape, because CubeCL's generated `CompilationArg` derives `Hash`/`Eq` and `f32` is neither |

Field types are written **unqualified** (`u32`, not `sm::u32`): VeriCL resolves a field type by its
last path segment, so a qualified path is rejected rather than trusted. A field may carry only the
bare `#[cube(comptime)]` marker and doc comments, and `#[cfg_attr(…)]` is rejected anywhere in the
block — rustc expands it *after* the macro has classified the attribute, which is exactly how the
macro and the compiler come to disagree about which fields are comptime.

**One type, both positions — when the fields allow it.** If every field in the transitive shape is
integer/`bool`/`char`/unit-enum, the macro also emits `Debug`/`PartialEq`/`Eq`/`Hash` (what CubeCL
requires of a `#[comptime]` parameter) and `ConfigIdentity`, so the same declaration serves `p: &T`
*and* `#[comptime] p: T` under one hash. A float field anywhere makes the comptime position
unavailable at any price, and the `ConfigIdentity` diagnostic says so rather than surfacing as three
raw trait errors.

`Array`/`Tensor`/`Slice`/`View`/`Sequence`/`SharedMemory` fields are **deferred**, with a rejection
that names all four missing pieces (a twin mirror type holding `&[T]`, a per-field entry in the
compared-buffer set, per-field compare-tier selection, and a `gen(len(p.a = N))` form). They are
measured working at the CubeCL level — the deferral is scope, not risk — and there are **zero**
instances in the surveyed ecosystem.

Also rejected, each with its own message: an `impl` block or `#[cube]` method inside the block (that
is the measured divergence, verbatim); a generic declared struct; `&mut P`; a struct or enum
**return** type from a kernel or helper (a tuple of scalars stays supported — it is destructured at
the call site); a payload-carrying runtime enum; `wrapping` together with a struct parameter; and a
`Vector` kernel with a struct parameter.

**The contract surface grows by one thing: dotted names.** `gen(p.field in lo..=hi)` and
`instantiate(p.comptime_field = …)`, at any declared nesting depth (`gen(cfg.window.gain in …)`).
Every runtime field needs a range and every comptime field needs a pin — and that is checked by
**rustc**, not by the macro: the clauses are emitted as a struct literal of a generated spec type, so
a missing range is `E0063: missing field` naming the field and a misspelled one is `E0560`. That is
also why it works across a crate boundary, where a macro-time registry could not.

**The residual, stated plainly.** The same one `vericl::config!` has, and worse in consequence: a
`#[cube] impl` written *outside* the block escapes both the hash and the gates, and because `#[cube]`
emits a host body *and* a device body, the failure mode is a numeric divergence rather than a panic.
There is no fix at macro scope. It is accepted because the differential lane catches any divergence
that reaches an output and `ir_hash` moves whenever the value reaches the device — both pinned by
`crates/vericl-examples/tests/cube_struct_out_of_block_backstop.rs`, including one assertion whose
whole job is to state the residual so it cannot be quietly forgotten.

**Honest reach.** As with struct-comptime, the value here is soundness, not coverage: of the ecosystem
sites this gate was blocking, a v1 unlocks **zero** — every one carries a co-gate this feature does
not touch (`Sequence`, device aggregates with `Slice`/`SharedMemory`/`View` fields, trait-generic or
associated-type parameters, cmma). What the corpus pays is a **census correction** (the gate's
sole-blocker count is 20, not 28, and 8 of the 28 needed nothing at all) and one live soundness hole
closed. See `docs/ecosystem-survey-2026-07.md`'s addendum.

### f64 support: the cubecl-cpu-only tier

`instantiate(F = f64)` monomorphizes a generic kernel at `f64` exactly like `F = f32`: the twin
becomes `&[f64]`/`alpha: f64` and computes at full f64 precision, `conformance_case` launches
`<f64, R>`, and `kernel_definition()` extracts the IR at `f64`. Input generation uses
`SplitMix64`'s 53-bit `next_f64_range`/`fill_f64` (the f64 analog of the 24-bit f32 path), a float
parameter without a `gen(...)` range is the same compile error as for f32, and the compare mode is
recorded honestly at f64 precision — `compare(abs = 1e-12)` on an f64 kernel becomes
`Compare::AbsRelF64` (an f64 tolerance stored at f64 precision, described `f64 |e-a| <= …`), never
silently narrowed to the f32 variant. The flagship example is `axpy_f64` — byte-for-byte `axpy`
with `instantiate(F = f64)`.

**The platform caveat, stated loudly because it is a soundness landmine.** WGSL has no `f64`, so an
f64 kernel *cannot* run on the wgpu/Metal backend — but cubecl 0.10 does **not** reject it. Verified
empirically: launching an f64 kernel on `WgpuRuntime` produces **no compile error and no runtime
panic**, and then returns **silently wrong results** — not even an f32 demotion (which would at least
be a recognizable rounding), but genuine garbage, because the host uploads 8-byte f64 elements into a
buffer the WGSL kernel indexes at a different element size. A green-looking launch that quietly
computes the wrong answer is precisely the failure class VeriCL exists to catch, so this is pinned by
a test (`crates/vericl-examples/tests/f64_wgpu_unsound.rs`, which asserts the f64 kernel *diverges*
from its correct twin on wgpu) and never used as an execution lane. cubecl-cpu, by contrast, runs
f64 correctly at full precision (verified: bit-exact to a host f64 computation).

The consequence for the trust boundary is real and worth naming. For an **f32** kernel, wgpu and
cubecl-cpu are two genuinely different backends, so the wgpu lane is an execution path independent of
cpu (and the cpu extra-lane is recorded as *not* independent because it shares CubeCL's front end).
For an **f64** kernel on this machine there is **no front-end-independent execution lane at all**:
wgpu is unusable, and cubecl-cpu shares CubeCL's front end (macro expansion + IR) with the kernel
under test. So the macro-derived sequential twin is the **sole** independent leg, which makes its
independence *load-bearing* rather than a redundant cross-check. The f64 suite records this in the
evidence trusted list explicitly — `host CPU execution hardware` (not the f32 lanes' "GPU hardware"),
plus the standing shared-front-end caveat "this lane is NOT an independent reference; only the
vericl-macros sequential twin is independent of CubeCL" — via a `frontend_independent: false` suite
declaration. f64 kernels therefore get their own `suite!` invocation on `cubecl::cpu::CpuRuntime`
with its own evidence file (`crates/vericl-examples/tests/conformance_f64.rs` →
`evidence/vericl_f64.json`), the same "one suite, one manifest" precedent as `conformance.rs` and
`cooperative_fallback.rs`; it is `#[cfg(feature = "cpu")]`, so it is exercised under `cargo test
--features cpu`. `axpy_f64` there carries a `tested` (differential, cpu) claim and a `proved`
`smt-oob-freedom` claim (3 obligations — bounds freedom is about buffer `Length`, so the f64 element
type is irrelevant to the proof). Everything else — `wrapping` (still integer-only), the bounds
prover, kernel composition — is unchanged; f64 is an instantiate tier, not a new subset.

### Kernel composition: `#[vericl::helper]` and `uses(...)`

Real kernels call other `#[cube]` functions — the July 2026 dogfooding survey found this blocking
16/22 production kernels, the largest gap after generics/`#[comptime]` (see
`docs/dogfood-2026-07.md`). `#[vericl::helper]` extends the same derivation story to non-launch
`#[cube]` device functions:

```rust
#[vericl::helper(instantiate(F = f32))]
#[cube]
pub fn single_tap<F: Float>(a: F, gain: F) -> F {
    a * gain
}

#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0, gain in -4.0..=4.0),
    instantiate(F = f32),
    uses(single_tap)
)]
#[cube(launch)]
pub fn gain_kernel<F: Float + CubeElement>(x: &Array<F>, y: &mut Array<F>, gain: F) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = single_tap(x[ABSOLUTE_POS], gain);
    }
}
```

`#[vericl::helper]` re-emits the `#[cube]` function untouched and generates a host twin
`fn single_tap_vericl_ref(...)` plus a `single_tap_vericl` module carrying its own `SOURCE_HASH`.
The kernel's `uses(single_tap)` clause rewrites its twin's calls to `single_tap(...)` into calls to
`single_tap_vericl_ref(...)`; a call to a function that's neither `uses(...)`-listed, a local
binding, nor a small allowlist of known host-safe free functions is a targeted compile error
naming the function and suggesting `uses(...)` + `#[vericl::helper]`, instead of the confusing
type error that would otherwise surface deep in cubecl's generated code. Helpers may call other
`#[vericl::helper]`-annotated functions via their own `uses(...)` clause — the identical mechanism,
so helper-calling-helper needs no special casing. `#[comptime]` parameters on a helper stay
ordinary pass-through parameters (the caller's own twin already has the pinned value in hand to
pass along); `ABSOLUTE_POS` and every other topology builtin are banned in a helper's body — a pure
device function reading global thread position would make its twin's calling convention ambiguous
(the dogfood survey found zero helpers using topology, so this costs nothing real).

**A helper's generic type parameter must be monomorphized via its own `instantiate(...)`, exactly
like a kernel's — it cannot be left generic**, even though an early draft of this design tried
that. The reason is the same Float-method-whitelist story above, taken one step further: on a
*concrete* receiver (`x: f32`), Rust prefers the inherent `f32::sqrt` over the trait method, which
is what makes the whitelist host-safe. On a still-generic, merely-bound receiver (`x: F` with
`F: Float`), there is no inherent method to prefer — the call resolves purely through the `Float`
trait, whose default body is the same `unexpanded!()` panic the whitelist exists to keep out.
Verified empirically (not just reasoned about): a scratch `fn g<F: Float>(x: F) -> F { x.sqrt() }`
panics on host calling `g(2.5f32)`, as does `.abs()` — reading cubecl-core's `impl_unary_func!`
macro confirms why (`impl Sqrt for f32 {}` inherits the panicking default rather than overriding
it). Monomorphizing a helper via its own `instantiate(...)` reuses the exact machinery already
verified safe for kernels instead of introducing a second, weaker safety story. The practical cost
is small: a helper's twin is pinned to one concrete type (today, `f32` is the only type any part of
vericl v0 supports, so this is free in practice — revisit if/when an `f64` tier is added).

**Identity and composition.** A kernel's `SOURCE_HASH` constant only ever covers its own source
tokens, computed at macro-expansion time — it cannot see a change to a helper's body, since that
lives in a separate macro invocation vericl-macros has no way to observe. `<kernel>_vericl::identity()`
closes this gap at ordinary Rust runtime: it folds `SOURCE_HASH` together with every `uses(...)`-listed
helper's own `identity_hash()` (via `vericl::combine_source_hash`, a small SHA-256 combine — the
one place core `vericl` depends on `sha2`, still with no `cubecl` dependency), and a helper's
`identity_hash()` recursively folds in its *own* `uses(...)` the same way, so a change two levels
deep in a helper-call chain still moves the top-level kernel's recorded identity. This is defense
in depth alongside, not instead of, the IR-level hash: cube expansion inlines a used helper's real
IR directly into the composing kernel's own `Scope`, so `ir_hash` already reflects a helper body
change too — `identity()` makes the source-level hash honor composition the same way rather than
leaving that half silently stale. **`uses(...)`'s declaration order is folded into the combine, so
purely reordering a `uses(a, b)` clause to `uses(b, a)` — the same dependency *set* — changes
`SOURCE_HASH` and `identity()`, even though nothing about the kernel's actual behavior changed.**
This is a safe direction to be sensitive in (it only ever causes spurious "stale evidence, please
re-run" churn, never lets real drift through unnoticed) but is worth knowing before reordering a
`uses(...)` list expecting evidence to stay untouched. A helper (or kernel) whose `uses(...)` graph is cyclic — including
the degenerate case of listing itself — is rejected at compile time on a best-effort basis: a
process-local registry accumulates every `uses(...)` edge seen so far in the compilation and checks
for a cycle on each new declaration, which reliably catches any cycle written in ordinary top-to-
bottom source (the last node in a cycle to be macro-expanded always closes it, and by definition
every other node has already registered by then) but is not a soundness-critical guarantee, since a
`#[proc_macro_attribute]` invocation cannot see other invocations' output directly. `#[cube]` itself
does not help here — verified empirically that both direct and mutual recursion between `#[cube]`
functions compile cleanly today (the former only draws rustc's ordinary `unconditional_recursion`
lint *warning*). As a backstop for the residual gap, the runtime hash-combine is depth-guarded
(32 levels) and panics naming the offending item rather than hanging, should a cycle ever slip past
the compile-time check.

The SMT bounds prover needed zero changes for composition: cube expansion inlines a used helper's
IR directly into the composing kernel's own `Scope`, so the existing walker over
`kernel_definition()` already sees everything a helper's body does — a guarded array access inside
a composed helper discharges exactly like one written directly in the kernel, and an unguarded one
refutes the same way (see `crates/vericl-examples/src/lib.rs`'s `tap_pair_guarded_kernel`/
`tap_pair_unguarded_kernel` for the pinned positive/negative pair).

### The `gen(...)` clause: ergonomic by being explicit

`gen(...)` declares, per parameter, how `conformance_case` draws inputs: `name in lo..=hi` for a
scalar or (applied elementwise) an array, and an optional `len(name = N)` to pin an array's
generated length to a constant instead of the case size — needed by kernels like `sum_racy`, whose
`assumes(y.len() == 1)` requires `gen(..., len(y = 1))`. Integer parameters left out of `gen(...)`
default to full-range generation; **float parameters with no declared range are a compile error**,
not a silent default. This is a deliberate ergonomic decision: an unbounded float draw produces
NaN/inf-adjacent garbage and tolerances no `compare(abs = ...)` can honestly justify, and the
failure is far more useful caught at authoring time (`error: parameter alpha is a float with no
declared gen(...) range`) than surfacing later as a confusing NaN mismatch or an unprovable
tolerance at run time. Generated inputs are drawn from vericl's `SplitMix64` in kernel-parameter
declaration order (not `gen(...)` clause order) for determinism, then checked against
`check_assumes(...)`; a rejected draw resamples (same RNG stream) up to 64 times before erroring
with the kernel name, so a persistent failure means the declared ranges are inconsistent with the
kernel's own `assumes(...)`, not a runtime fluke.

### The `cooperative(...)` clause: workgroup shared-memory reductions

```rust
#[vericl::kernel(
    assumes(input.iter().all(|v| v.abs() <= 1000.0)),
    compare(max_ulp = 0),
    gen(input in -1000.0..=1000.0),
    cooperative(cube_dim = 256)
)]
#[cube(launch)]
pub fn block_sum_reduce(input: &Array<f32>, output: &mut Array<f32>) {
    let tid = UNIT_POS as usize;
    let mut tile = SharedMemory::<f32>::new(256usize);
    /* load into tile; */ sync_cube();
    let mut half = CUBE_DIM as usize / 2;
    while half > 0usize {
        if tid < half { tile[tid] = tile[tid] + tile[tid + half]; }
        sync_cube();
        half /= 2usize;
    }
    if tid == 0usize && CUBE_POS < output.len() { output[CUBE_POS] = tile[0usize]; }
}
```

The `cooperative(cube_dim = N)` clause opts a kernel into the workgroup-cooperative shape —
`UNIT_POS`/`CUBE_POS`/`CUBE_DIM`/`CUBE_COUNT`, `SharedMemory`, `sync_cube()`, grid-stride loops,
tree reductions — which the ordinary loop-over-`ABSOLUTE_POS` twin cannot model (a sequential
per-thread twin has no per-workgroup shared arena and no barrier semantics). It swaps in a
**phase-split twin**: the body is split at each `sync_cube()` into barrier-delimited segments, run
per cube, per segment, per `unit_pos`, with `SharedMemory` a per-cube **poison-initialised** tile
(a read of a never-written cell panics rather than masking an uninitialised-read bug with a zero).
`cube_dim` pins the launch block size *and* the prover's `CUBE_DIM` binding (a single source of
truth — a launch with a different block size panics loudly rather than binding `CUBE_DIM` to a value
the launch does not use). The suite sizes each `&mut Array` output to `cube_count` (one partial per
workgroup) and launches `(cube_count, cube_dim)`. The v1 subset is the 1-D reduction shape
(one non-cooperative accumulation loop, one uniform-trip-count tree loop, single-writer `tid == 0`
store); anything else — a barrier under a thread-varying condition (barrier divergence), a
non-uniform tree loop, multiple tiles — is rejected with a targeted error, never mis-modelled.
Design: `docs/design-shared-memory.md`.

### The `dispatch(...)` clause: 2-D / 3-D image-space kernels

```rust
#[vericl::kernel(
    dispatch(cube_dim = (16, 16), extents = (w, h)),
    assumes(inp.len() == out.len(), inp.len() == (w as usize) * (h as usize)),
    compare(max_ulp = 0),
    gen(inp in -100.0..=100.0, out in 0.0..=0.0)
)]
#[cube(launch)]
pub fn box_blur3x3(inp: &Array<f32>, out: &mut Array<f32>, w: u32, h: u32) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x < w && y < h {
        let x0 = u32::max(x, 1u32) - 1u32;
        let x2 = u32::min(x + 1u32, w - 1u32);
        /* … y0, y2, nine adds … */
        out[(y * w + x) as usize] = acc * 0.111111111f32;
    }
}
```

`dispatch(cube_dim = (Wx, Wy[, Wz]), extents = (e0, e1[, e2]))` is `cooperative(cube_dim = N)`'s
precedent moved one position over: the pinned literal dims are the single source of truth for the
launch's `CubeDim`, the twin's per-axis loop strides, and the prover's `CUBE_DIM_X/Y/Z` numerals —
and pinning them is what keeps every position recomposition **linear**. `extents` names the kernel's
own runtime `u32` extent parameters; the harness binds them from each case's declared size, derives
`CubeCount::Static(ceil(e0/Wx), …)`, and sizes un-pinned buffers to their product. The twin becomes a
**nested grid loop**, `for z { for y { for x { … } } }` over the *grid* (so padding threads run the
guard exactly as on device) in Z→Y→X order (which is the flat `ABSOLUTE_POS` order, so a kernel
ported from flat to per-axis addressing keeps the same aliasing-write convention).

Three things are load-bearing and each is measured, not asserted:

- **The flat `ABSOLUTE_POS`/`CUBE_POS`/`CUBE_COUNT` are rejected inside the clause.** The identity
  the 1-D cooperative model encodes — `ABSOLUTE_POS == CUBE_POS * CUBE_DIM + UNIT_POS` — is **false**
  in a multi-axis dispatch: swept on hardware over 722 launch shapes it held in 189 and broke in 533,
  and 912 of 960 threads violate it at the image-like `CubeCount(5,3,1) x CubeDim(8,8,1)`. The
  per-axis relations (`ABSOLUTE_POS_a == CUBE_POS_a * CUBE_DIM_a + UNIT_POS_a`) *are* exact — 0
  violations in 1 212 threads — so those are what the twin and the prover both use, each through the
  **same single** exact-modular recomposition the round-5 fix introduced. Flat `CUBE_DIM` and
  `UNIT_POS` are kept: a numeral and a pinned-coefficient linear form, neither of which can wrap.
- **`assumes(A.len() == (w as usize) * (h as usize))` is the enabling fact, not sugar.**
  `ABSOLUTE_POS_Y * w` has no Euclidean parent, so without it the row stride's no-overflow
  side-obligation is unprovable and *every* 2-D kernel that indexes an array is `OutOfSubset`. It
  must be widen-then-multiply: the outer-cast spelling `(w * h) as usize` tests the **wrapped**
  product in `check_assumes` while the model asserts the mathematical one — a false `Proved` at
  `w = 2, h = 2147483649` with a length-2 buffer — and is rejected at the cast by name.
- **The stencil clamp must be branch-free.** `if x + 1 < w { x2 = x + 1; }` writes a mutable local in
  a branch arm, which round-2 branch write-taint correctly taints; `u32::min(x + 1, w - 1)` computes
  the identical function and lowers to an `Arithmetic::Min` the prover models as an exact `ite`.

`dispatch(...)` excludes `cooperative(...)` (2-D shared-memory tiles are deferred: the intra-cube
race obligation discharges in under 10 ms, the inter-cube one times out in z3 at 180 s and needs a
new write-pattern recognizer), `Vector<P, W>` (two `sizes_unit` conventions, undecided), and a
runtime `cube_struct!` parameter. **Honest reach: 1 of 464 surveyed ecosystem items is sole-blocked
by 2-D and 0 additional private kernels are unlocked** — this is a capability-and-soundness
milestone, not a coverage one. Design: `docs/design-2d-dispatch.md`.

### Vector (SIMD) element support: `Array<Vector<P, N>>`

```rust
#[vericl::kernel(
    assumes(a.len() == out.len(), b.len() == out.len()),
    compare(abs = 1e-6),
    gen(a in -100.0..=100.0, b in -100.0..=100.0, out in 0.0..=0.0),
    instantiate(N = 4)          // pin the lane width, exactly as instantiate(F = f32)
)]
#[cube(launch)]
pub fn vec_add<N: Size>(
    a: &Array<Vector<f32, N>>,
    b: &Array<Vector<f32, N>>,
    out: &mut Array<Vector<f32, N>>,
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}
```

`Vector<P, N>` is CubeCL's SIMD element type — a length-`N` lane vector. Its width `N` is a
**compile-time** generic (`N: Size`), so it pins per contract via `instantiate(N = W)` just as a
generic float pins via `instantiate(F = f32)`; one width per contract. The reference twin maps
`Array<Vector<P, N>>` → `&[vericl::Line<P, W>]`, a host lane-array shim whose every op is a **per-lane**
map — and a vector-`W` op *is* `W` independent scalar ops with no cross-lane coupling or reordering, so
at equal precision the twin reproduces the GPU value **bit-for-bit** for the correctly-rounded
elementwise ops. Every lane op is GPU-ground-truth-verified bit-exact on wgpu (and cubecl-cpu) —
nothing reaches the twin surface unverified. I/O stays scalar throughout: `gen` draws `lines*W` flat
scalars (the range applies per lane), the launch is spliced at the pinned vectorization `W`, and a
divergence is reported per lane `(line, lane)`. A fusable expression like `a*a + b` (`vec_madd`) gets
the same `compare(abs = …)` an ordinary scalar `a*a+b` would — the one legitimate float divergence (an
FMA the backend contracts, or Metal's not-correctly-rounded `f32 /`), never a vector-model error.

Bounds are proved by the existing walker unmodified: whole-vector indexing lowers to a `vector_size: 0`
access whose width lives in the element `Type`, and `.len()` is **line**-granular, so the obligation
`0 <= ABSOLUTE_POS < out.len()` is the scalar one — `N` never enters it. The one soundness guard is
that a `Vector<u32, N>` value (whose *storage* is integer) can never be modeled as a single scalar
integer. A comptime-unrolled `for j in 0..W` affine-in-lane write into a register vector is accepted;
data-dependent per-lane indexing, cross-lane reductions (`dot`/`magnitude`/`normalize`),
reinterpret-slice, and `SharedMemory<Vector>` are rejected with targeted errors.

**Honest coverage.** This is the **vectorized elementwise class** — the immediate generalization of the
already-provable scalar shortlist to its true vector element type (e.g. an f32 `to_degrees` map at
`Vector<f32, 4>`). `Vector` is the #1 gate incidence in tracel-ai's kernel libraries, but rarely the
*only* gate: whole-kernel reach for the reduction/matmul launch sites additionally needs `View`/`Slice`
(the #2 gap), `Atomic`, `comptime!`, and `match` — the documented, non-silent follow-on. Design:
`docs/design-line-vector.md`.

### Core `Slice` (addressing views): `arr.slice(a, b)`

```rust
#[vericl::kernel(
    assumes(y.len() + 4 <= x.len()),      // the window fits: x is 4 longer than y
    compare(max_ulp = 0),
    gen(x in -10.0..=10.0, y in 0.0..=0.0, len(x = n + 4))
)]
#[cube(launch)]
pub fn windowed_slice_sum(x: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        let mut acc = f32::new(0.0);
        for v in x.slice(ABSOLUTE_POS, ABSOLUTE_POS + 4) {   // a slice window
            acc += v;
        }
        y[ABSOLUTE_POS] = acc;
    }
}
```

A core `Slice<E, IO>` is a pure **addressing view** `(origin, offset, length)`, not a buffer:
`arr.slice(a, b)[i]` lowers to a checked `origin[a + i]` — the slice emits no buffer, no metadata, no
separate id (the prover cannot even distinguish `arr.slice(2,5)[i]` from a hand-written `arr[2+i]`). So
**bounds proving is the ordinary origin obligation, discharged by the existing walker unmodified**:
`to_slice()`, dynamic and constant offsets, **nested** slices (offsets compose additively),
**iteration** (a `RangeLoop` over `origin[offset+i]`), and a **gather through a slice** of an
element-assumed array (the assume transfers for free by origin-id keying) all `Proved`; unguarded or
under-constrained variants `Refuted`/`OutOfSubset`.

The reference twin maps a slice to a **Rust subslice** — `arr.slice(a, b)` → `&arr[a..b]`,
`slice_mut` → `&mut arr[a..b]`, `to_slice()` → `&arr[..]`, `for item in slice` → `for &item in …`. A
slice introduces **zero numeric ops**, so the twin is **bit-exact** on wgpu and cubecl-cpu. Two Rust
guarantees become the soundness net that cubecl itself lacks: an out-of-range `&arr[a..b]` **panics** in
the tested twin (cubecl does *not* bounds-check slice creation), and the **borrow checker is the
aliasing oracle** — sequential mutable slices compile, but two simultaneously-live overlapping `&mut`
subslices of one origin do not. That aliasing rejection is, as-built, rustc's own `E0499`/`E0502` on the
generated twin (a buffer-named vericl-authored message is future work, `docs/design-view-slice.md` §8.4);
it is the borrow checker itself, not a macro pass, that rejects the unsafe program. Slice type-punning
(`as_mut_unchecked`/`downcast*`), reinterpret-slice
(`with_vector_size`, `vector_size ≠ 0` — also unrunnable on wgpu upstream), and the `View`/`VirtualLayout`/
`Coordinates` strided-tensor machinery (a separate `Arc<dyn>` abstraction, **not** core `Slice`) are
rejected with targeted errors. Slices are helper-only, not launch args; a `#[vericl::helper]` taking a
`&Slice<F>` param (the dominant real usage) maps it to `&[f32]`.

**Honest coverage.** Core `Slice` is the tractable half of the survey's #2 gate — whose "128" is really
**~25 real core-slice creators + a `ReadOnly`/`ReadWrite`-ident tail + the deferred `View` machinery**
(the single regex conflated them). It is **necessary but rarely sufficient**: of the ~25 creators, only
~10 trip no other gate, and every one is an `impl`/`trait`/test-launcher, not a 1-D launch kernel. v1's
reach is the **slice-carrying elementwise/windowed class + the generalized shortlist + the Vector
readers**, not the matmul/reduce launch sites. The honest post-`Slice` frontier: `plane_*` reductions,
then custom cube structs (`CubeType`-arg), then 2-D topology, then `Tensor` + the deferred `View`
machinery. Design: `docs/design-view-slice.md`.

### Suites: `vericl::suite!`

```rust
vericl::suite! {
    runtime: cubecl::wgpu::WgpuRuntime,
    kernels: [axpy, xorshift_step, mix_u32],
    evidence: "evidence/vericl.json",
}
```

Expands to `#[test] fn vericl_conformance()`: builds the client, runs every listed kernel's
`conformance_case` across the declared sizes, discharges the SMT bounds proof via `vericl-ir`
(`prove: false` omits proved claims instead of ever recording a fake or skipped one), and
assembles the evidence manifest. With `VERICL_UPDATE` set (any value), it writes the manifest;
otherwise it loads what's on disk, calls `vericl::verify`, and panics with the problem list on any
mismatch — so `cargo test` is the whole CI story. The evidence path is relative to
`CARGO_MANIFEST_DIR`. An optional `extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime)`
folds an additional differential lane (sharing CubeCL's front end, so recorded as *not
independent* — only the macro-derived sequential twin is) into the same test, appending claims to
the same entries before the manifest is finalized, so one suite invocation always produces exactly
one manifest.

### A first finding: why `compare(abs = ...)` exists

The very first differential run caught the wgpu/Metal backend contracting `a*x + y` into a fused
multiply-add: under catastrophic cancellation (`alpha*x ≈ -y`) the observed divergence from the
strict-rounding reference reached ~27,000 ULP. No useful ULP bound exists for this kernel on this
backend — the honest claim is an absolute error bound (`|e-a| <= abs + rel*|e|`) derived from the
declared input ranges in `assumes(...)`. The tolerance is part of the contract and is recorded as
an assumption in the evidence, exactly as the claim model requires.

### GPU-verified host shims: `fma`, `cast_from`, `mul_hi`

Some CubeCL intrinsics have no host implementation at all. `Cast::cast_from`, `Numeric::mul_hi`
and the free function `cubecl::prelude::fma` are all `unexpanded!()` — calling one from ordinary
Rust panics — so the derived reference twin cannot simply call them, and a `#[vericl::helper]` is
no escape hatch (the helper's twin would panic too). Writing them out by hand is worse: the
semantics are *GPU*-defined.

So `#[vericl::kernel]` / `#[vericl::helper]` rewrite a recognized intrinsic call in the twin (and
only the twin — the `#[cube]` item is re-emitted untouched) to a shim in
`vericl::host_shims`, and **every shim is pinned against the real intrinsic run in a real `#[cube]`
kernel**, on wgpu and on cubecl-cpu, in
`crates/vericl-examples/tests/host_shim_gpu_ground_truth.rs`. An unrecognized spelling is not
guessed at — it is rejected by name, and an unsupported operand type is a `CastToF32`/`MulHi`/`Fma`
trait error in the twin. Loud, never a silently wrong value.

| Intrinsic | Shim | Measured tier |
|---|---|---|
| `f32::cast_from(x)`, `x: u32`/`i32`/`usize`/`bool`/`f32` | `cast_to_f32` | bit-exact, both lanes (`bool` is exactly `true → 1.0`, `false → +0.0`; `usize` is cubecl's `AddressType`, verified across the whole u32 domain including `> 2^24`) |
| `T::mul_hi(a, b)` / `a.mul_hi(b)`, `T = u32` | `mul_hi` | bit-exact, both lanes |
| `cubecl::prelude::fma(a, b, c)`, `f32` | `fma` (Rust `f32::mul_add`) | bit-exact on cubecl-cpu everywhere; bit-exact on wgpu/Metal outside a characterized flush-to-zero domain (4974 of 21996 probe triples) — see below |

**`a * b + c` is not a substitute for `fma(a, b, c)`, and this is measured, not asserted.** The
unfused form rounds twice where `fma` rounds once, and for the two-product idiom
`fma(h, x, -(h*x))` — the exact rounding error of a rounded product, the primitive under every
compensated accumulator — the unfused rewrite is identically `0.0` where the fused answer is the
entire signal. Over the ground-truth corpus the naive host substitute would diverge from the real
GPU intrinsic on 8508 of 21996 triples on wgpu and 3782 of 21996 on cubecl-cpu; the public
`fma_two_product_residual` / `unfused_two_product_residual` example pair shows the same gap on the
device (residual non-zero on 1023 of 1024 inputs vs exactly zero).

**The one divergence class, recorded rather than smoothed over.** On wgpu/Metal the `fma` shim is
bit-exact on 17022 of 21996 probe triples. The other 4974 are a **flush-to-zero domain**, and on it
the device's answer is given exactly by:

```text
metal_fma(a, b, c) =
    let (a, b, c) = (ftz a, ftz b, ftz c)           // subnormal operands -> ±0
    if 0 < |exact(a*b + c)| < f32::MIN_POSITIVE     // EXACT, before rounding
        then ±0 with the sign of the exact value
        else fma(a, b, c)
```

The underflow decision is made on the **exact pre-rounding magnitude**, not on the rounded result.
The distinction is measurable, not pedantic: `fma(2^-126, 2^-126, -2^-126)` has all-normal operands
and rounds to the normal `-2^-126` on the host, while the device returns `-0`. The ground-truth
test asserts this model over *every* triple on both lanes (not just where host and device already
disagree), requires exactly one of "this model" / "no flush at all" to explain each lane with zero
mismatches, and is discriminated by ten injected mutations of the model — all of which fail it.
A kernel computing in the normal range is bit-exact (`compare(max_ulp = 0)`); one that genuinely
computes at or below the underflow boundary needs a tolerance.

**Only the qualified spelling is rewritten.** `fma` is glob-imported from `cubecl::prelude`, and a
glob import is the weakest binding in Rust: a `uses(...)` helper, a local binding, or **an ordinary
item named `fma` anywhere in the kernel's scope** wins over it on the `#[cube]` side, and a proc
macro cannot see the enclosing scope. VeriCL therefore does not rewrite a **bare** `fma(a, b, c)` at
all — measured, a `#[cube] fn fma` declared beside a kernel produced a twin computing `5.0` where
the device computed `1005.0`. A bare call falls through to the ordinary undeclared-call rejection,
which names both fixes: write `cubecl::prelude::fma(...)` for the intrinsic (that spelling compiles
inside `#[cube]` and *is* rewritten to the verified shim), or `uses(fma)` if you mean your own
`#[vericl::helper]`. An explicit `uses(fma)` declaration wins over the shim, so composing a helper
that happens to be named `fma` is a legal program.

## Claims and trust boundaries

VeriCL must say exactly what a result establishes. These are different claims and are never
presented as interchangeable:

- **Proved** — a property discharged by a checker over the kernel IR, under stated assumptions.
- **Tested** — behavior observed on specific inputs, on a specific backend, driver, and device.
- **Assumed** — declared constraints (input ranges, tolerances) that evidence depends on but does
  not establish.
- **Trusted** — components outside the checked boundary: CubeCL's backend code generation, the
  driver, the hardware. Source-level evidence never silently implies these are verified.

Every evidence entry records which of these categories each part of its claim falls into, and the
assumptions travel with the result. Evidence that no longer matches the kernel identity it was
produced from is rejected, not warned about.

### Proved claims

The first proved claim is live: out-of-bounds freedom for `axpy`, `xorshift_step`, and `mix_u32`,
discharged in QF_LIA by z3 (subprocess, via `easy-smt`) over each kernel's CubeCL IR — every
`Index`/`IndexAssign` obligation negated and checked UNSAT, with anything outside the supported
subset (unbounded `while`/`loop`, vectorized indexing, float-valued indices) reported explicitly
rather than silently skipped. The z3 binary, its bounds-obligation encoding, and CubeCL's front-end
expansion are recorded as trusted for this claim, since the proof is about the IR and codegen below
it stays covered only by the tested differential claims. Kernel identity now also carries an
IR-level content hash alongside the source-level one, so evidence goes stale on either kind of
drift. `axpy_off_by_one` REFUTES with a counterexample exhibiting the out-of-bounds position, and
`sum_racy`'s bounds PROVE even though its differential check correctly fails — the race is a
distinct, differential finding, never conflated with the bounds claim.

**Counterexample validation (the solver's `sat` verdict is not trusted for refutations).** Every
`REFUTED` verdict — bounds and two-thread data-race alike — is *independently re-checked in plain
Rust* before it is reported. The solver's model is read back and evaluated against the obligation's
entire live assertion set (the negated obligation, the path conditions, the assumes, and the leaf
type-range facts) by a small total interpreter over the exact SMT-LIB subset vericl emits; a model
that does not actually satisfy those assertions never becomes a `Refuted` — it fails **closed** to a
solver error, never a silent (possibly spurious) refutation. So for a refutation the solver's `sat`
verdict leaves the trusted base: what remains trusted is the ~120-line, unit-tested Rust
interpreter (checked directly against a synthetic invalid-model negative) plus vericl's own
encoding. This runs unconditionally, including in the defect demos, and adds no solver work on the
`Proved` path (it only runs on `sat`). The dual for `Proved` claims — independently *checkable proof
certificates* for `unsat`, which would move the solver binary out of the trusted base for proofs
too — is designed but currently deferred: it requires cvc5 + Alethe + the Carcara checker, none of
which are available at the pinned toolchain versions here (cvc5 is not packaged and Carcara is not a
crates.io dependency). The honest decision record, and the path to enabling it, are in
`docs/certificates-decision.md`; until then the z3 binary remains trusted for `Proved` claims, and
is recorded as such in evidence.

**Model fidelity: an independent IR interpreter cross-check (the trusted "CubeCL front-end
expansion" is now empirically checked).** Everything above is stated over the CubeCL IR, so one
component is load-bearing and was, until now, unchecked: does VeriCL's *model* of what an IR
instruction means match what CubeCL actually executes? A concrete **reference interpreter**
(`crates/vericl-ir/src/interp.rs`) closes that gap empirically. It is a *third, independent*
implementation of the modeled semantics — the twin rewrites source tokens, the prover encodes the IR
symbolically, and the interpreter *executes* the same `KernelDefinition` concretely over real inputs
with true finite-width wrapping arithmetic and IEEE-754 floats, **reporting** (never panicking on)
any out-of-bounds index or divide-by-zero. Two cross-checks run in `cargo test`: on every honest
example kernel the interpreter agrees **bit-for-bit** with the twin over the kernel's real
`kernel_definition()` IR (one kernel, `xorshift_step`, is checked three-way against a live
wgpu/Metal launch too); and a seeded **fuzz lane** generates random in-subset kernels and cross-checks
the prover's verdict against concrete execution — a `Proved` kernel that the interpreter can drive out
of bounds on an assume-satisfying input would be a critical fidelity finding, and a `Refuted` kernel's
counterexample, replayed, must exhibit the OOB. The full corpus (20,000 random kernels, 320,000
inputs, prover on) produces **zero** disagreements; a deterministic subset runs by default and the
full run behind `VERICL_FUZZ=1`. This *shrinks* model-fidelity risk empirically — it is **not** a
proof and mints no `Proved` claim (agreement is a `Tested` observation, and CubeCL codegen/driver/
hardware stay Trusted). Injected-bug negative controls confirm the cross-check catches real semantics
defects. Full scope, exclusions (cooperative/shared-memory kernels are out of the v0 interpreter
subset), and the exact "what agreement does and does not establish" are in `docs/interpreter.md`.

**Array-value-dependent indices (offset tables / gather).** The prover recognizes two *element-range*
`assumes(...)` shapes over an integer index array, in addition to the length shapes (`A.len() ==
B.len()`, `A.len() == N`): `A.iter().all(|v| (*v as usize) < B.len())` and `A.iter().all(|v| *v < N)`
(with/without the deref and `usize` cast normalized; only the strict `<` — a `<=` is not a valid
in-bounds guarantee and stays string-only). Under such an assume, a read `A[i]` — whose *own* index
obligation still has to discharge — produces a value modeled as a fresh symbol bounded by the assume,
instead of the usual taint. This is the **only** case array *contents* get a model, and it is what
lets a gather `y[i] = x[offsets[i]]` prove in bounds (`gather_copy`, wired into the suite: bit-exact
differential + a 3-obligation `smt-oob-freedom` proof), with nested gathers `a[b[i]]` composing
automatically. It stays sound the same way a length assume does — the proof is conditional on an
assumed claim that the executable `check_assumes` predicate tests at generation time (so the
differential lane only runs offset tables satisfying it, and the bound doubles as `offsets`' `gen(...)`
range, stated once). A write to `A`'s elements invalidates the assumption for every subsequent read of
`A` (including across loop iterations), and a *wrong* (too-loose) bound does not hide a bug: `gather_oob`
(a stale constant bound looser than the indexed array) REFUTES with the fresh element symbol pinned at
the boundary.

**`match` on integers (`Branch::Switch`).** A Rust `match` on an integer scrutinee lowers to a
`Branch::Switch`, which the prover models as an exhaustive if-chain: each case arm is bounds-checked
under its own path condition `value == case_i`, and the default arm under the conjunction of all
`value != case_i` (so a case set that fully covers a bounded scrutinee's range makes the default
provably unreachable). Branch-scoped write taint is the same machinery as `if`/`else`, generalized to
N+1 arms — a per-arm write is never merged across arms, so it cannot leak past the switch. A
thread-varying scrutinee with a `sync_cube()` inside an arm is barrier divergence, rejected exactly
like any other conditional barrier. `select_mode` (a `match` on a scalar `mode`) is wired into the
suite with a tested + a 6-obligation proved claim. The reference twin re-emits the `match` verbatim
(host Rust `match` is the reference), so the differential lane needs no special handling.

**Length-relationship assume (`A.len() + K <= B.len()`).** A third recognized `assumes(...)` shape (an
integer literal `K`; the `A.len() <= B.len()` `K = 0` case included) — the "additive anchor" host-side
buffer-sizing invariant. The prover asserts `len_a + K <= len_b` directly, which — combined with a
guard `i < A.len()` — discharges a forward/offset read `B[i + K]` in bounds. Unlike the element-range
proxy, the recognized relation `<=` maps onto the modeled `<=` verbatim (the source clause *is* the
constraint, with no index-validity reinterpretation), so `<=` is exactly correct here where only `<`
was sound for the element case. The recognizer is strict (only the two literal shapes; `<`, `>=`,
non-literal `K`, subtraction, and any other arithmetic stay string-only). `offset_window`
(`y[i] = x[i] + x[i + 4]` with `y.len() + 4 <= x.len()`) is wired into the suite with a tested + a
3-obligation proved claim.

**Overflow soundness (finite-width integer semantics).** The bounds proof models integer
arithmetic *faithfully to hardware wraparound*: every non-tainted modeled integer term equals the
real (wrapping) `u32`/etc. value at every input, so an index, a div/mod divisor, a branch/loop
guard, and a loop bound all read the true value — a term that could diverge from hardware is
tainted instead, and fails explicitly at whichever site needs it. Leaves are declared in their
type's range (a `u32` really is in `[0, 2^32)`), `Add`/`Sub` are modeled exactly under wraparound,
and `Mul` carries a no-overflow side-obligation (bind the product only when it provably cannot
wrap, else taint). This closes the overflow-into-zero-divisor gap the round-2 review found (below):
a divisor `a * b` that is provably nonzero in unbounded arithmetic but wraps to `65536 * 65536 ==
2^32 ≡ 0` on hardware now taints — `OutOfSubset`, never `Proved`. A genuinely non-wrapping chain
still proves: `flatten_decode_scale`'s `row*width + col` proves in bounds because the leaf bound
`ABSOLUTE_POS <= u32::MAX` plus `row*width <= ABSOLUTE_POS` discharges the no-overflow
side-obligation, with no assume strengthening needed. The chosen approach keeps the existing QF_LIA
encodings (bounds, length/element assumes, div/mod, the race walk) intact rather than rewriting to
QF_BV — the design rationale is in `crates/vericl-ir/src/prover.rs`'s "Bounded-integer overflow
model" module doc. One honest consequence surfaced on our own suite: `fir_pair_kernel`'s guard
`ABSOLUTE_POS + 1 < x.len()` silently relied on no-wrap to also cover its `x[ABSOLUTE_POS]` read
(the implication `pos + 1 < len ⟹ pos < len` holds at every reachable dispatch but not at the
adversarial `pos == u32::MAX`, where `pos + 1` wraps to `0`); it was strengthened to state `pos <
x.len() && pos + 1 < x.len()` explicitly (safe at every reachable dispatch either way, and now
provable). A `wrapping`-clause kernel declares wrap intent for its *values*; its *indices* still may
not wrap (a wrapped index is still out of bounds), so the prover treats it exactly like any other
kernel.

The second proved claim is **data-race freedom** (`smt-race-freedom`), for the cooperative
shared-memory kernels. It is discharged by a GPUVerify-style two-thread symbolic reduction: two
arbitrary distinct threads `t1 ≠ t2` of one cube are walked, and within each barrier-delimited phase
every shared/global write is proved not to collide (same index) with another thread's write
(write-write) or read (read-write), plus barrier uniformity and inter-cube single-writer
disjointness — all in QF_LIA, UNSAT meaning race-free, SAT a real race reported with a two-thread
counterexample. `block_sum_reduce` and `grid_stride_reduce` PROVE race-free and in-bounds; the
demo-defects `block_sum_reduce_racy` (an overlapping `tile[tid] += tile[tid+1]` stride) REFUTES with
a two-thread counterexample (`t1 == t2 + 1`). The one two-thread walk discharges *both* the race
obligations and the tree-reduction bounds obligations that the single-thread bounds walk defers, so
a cooperative kernel earns both a `smt-race-freedom` and a `smt-oob-freedom` proved claim from it,
each with its own honest obligation count.

**The differential↔race-freedom coupling (the honesty rule).** A phase-split twin picks *one*
intra-segment thread order, so it is a faithful reference **only** when every segment is race-free —
which is exactly what `smt-race-freedom` proves. A cooperative kernel's `tested` differential claim
therefore always makes that dependency explicit, in one of three never-blurred tiers: when race
freedom is **proved**, the tested claim's config cites it as a *discharged* dependency (pointing at
the proved claim); when it is **not** proved (`prove: false`, or the proof is out-of-subset), the
suite injects an explicit `assumed` claim — "intra-phase race freedom + barrier non-divergence" —
and the tested claim depends on *that* instead; a cooperative differential result with neither the
proof nor the assumption is **refused**, not recorded (the same posture as `prove: false` omitting a
proved claim rather than faking one). A green cooperative test can never silently over-claim: the
thing that makes it valid is always a named, visible dependency. A hand-written reference supplied
via `reference = fn` (for a kernel the transform cannot derive) carries a distinct, strictly weaker
`differential-declared-reference` check string, since it is a separate artifact that can drift from
the kernel — never conflated with the derived twin. That reference fn must carry the
`#[vericl::reference]` attribute (a compile error names it otherwise); the attribute records the
reference's own source hash, which the kernel folds into its `identity()`, so a drift in the
reference **body** — not just the `reference = fn` clause path text — moves the kernel's recorded
identity (round-3 adversarial review, F2).

### CubeCL semantics findings

Two upstream CubeCL/WGSL behaviors surfaced while adversarially reviewing the SMT prover (round 2,
see `tasks/todo.md`) that are worth knowing on their own, independent of VeriCL:

- **`&&`/`||` are eager inside a `#[cube]` kernel body, not short-circuiting.** CubeCL 0.10 lowers
  both operands of `a && b` (and `a || b`) to ordinary, unconditionally-evaluated instructions
  *before* combining them into a single boolean — there is no branch, so the right-hand side
  executes even when the left-hand side alone would already decide the result. A guard shaped
  `idx_ok && x[idx] > 0.0` does **not** protect the `x[idx]` read the way the same expression would
  in host Rust: the read happens on every thread, guard or not. VeriCL's prover models this
  correctly — a guard's `&&` composes as SMT `and` over both operands' obligations, which are
  already unconditional in the IR, so an insufficiently-guarded access still `Refuted`s — but on
  WGSL the backend's own robustness (out-of-bounds reads/writes silently clamp rather than trap)
  can mask the effect at runtime, exactly the kind of gap a differential-only check (no static
  prover) would miss entirely.
- **naga's division-by-zero fallback is dividend-preserving, not trapping.** On the wgpu/Metal
  backend, `a / 0` (and `a % 0`) does not trap or return a fixed sentinel — it returns `a` unchanged
  (confirmed empirically: `ABSOLUTE_POS / 0` returns `ABSOLUTE_POS`; `ABSOLUTE_POS % 0` returns
  `0`). One consequence: a divisor that's provably nonzero in unbounded integer arithmetic but
  wraps to exactly zero via `u32` overflow (e.g. `a * b` where `a * b == 2^32`) does not itself
  crash on this backend — the resulting index is merely wrong, not a hardware fault. This
  overflow-into-zero-divisor shape *was* a known out-of-subset gap (harmless in practice only
  because of naga's fallback, never a guarantee to rely on); it is now **closed** by the
  finite-width overflow model (see "Overflow soundness" above and the prover's "Bounded-integer
  overflow model" module doc): the `Mul` no-overflow side-obligation fails for `a * b == 2^32`, so
  the divisor taints and the dependent access is `OutOfSubset` rather than falsely `Proved` — no
  longer relying on the backend's dividend-preserving behavior.

## First release

The first release demonstrates one complete, honest path from kernel intent to executable artifact
and evidence. It is done when:

1. **Contract and identity** — a kernel can be annotated with assumptions, and VeriCL assigns it a
   stable identity; changing the kernel, contract, or toolchain invalidates dependent evidence.
2. **Differential conformance** — generated inputs run against the scalar reference and at least
   one GPU backend, with counterexamples reported on divergence, and `vericl check` enforces this
   in CI.
3. **One proved property** — out-of-bounds freedom is machine-checked for a defined kernel subset
   (affine index expressions, bounded loops, known launch dimensions), with kernels outside the
   subset rejected explicitly rather than silently approximated.
4. **Honest examples** — at least two example kernels (one motivated by a private production kernel
   but independently written, one generic, e.g. a counter-based RNG or prefix sum), each paired with a
   deliberately defective twin whose defect the appropriate check catches and reports usefully.

Breadth — more backends, more property classes, richer numeric models, proof assistants — is
explicitly deferred. A narrow path with honest claims is sufficient.

## Relationship to prior art

- **GPUVerify** — the closest neighbor: static race and bounds analysis for CUDA/OpenCL, now
  essentially unmaintained and disconnected from any Rust or CubeCL workflow. VeriCL's checked
  property list starts narrower, but its evidence is bound to a live, multi-backend source of
  custody rather than a one-shot analysis.
- **Alive2 / translation validation** — validates compiler transformations; VeriCL does not verify
  CubeCL's codegen and records it as trusted instead. Translation validation of CubeCL backends
  would shrink that trusted boundary and is a natural later stage.
- **Verus, Kani** — Rust-level verification tools. Because the reference execution is ordinary
  Rust, these are candidate engines for proving properties of the reference itself in a later
  release, without changing VeriCL's core concepts.
- **Exo, Halide** — correct-by-construction scheduling for kernels authored in their own
  languages; VeriCL instead meets CubeCL developers in the language they already use and checks
  after the fact.

## Private dogfooding

VeriCL was developed and validated against a private production RF/signal-processing codebase whose
kernels place real demands on determinism, indexing, replay, and numerical comparison. That codebase
is where the requirements came from and where VeriCL is exercised against non-toy kernels — but its
kernel implementations never enter this repository. Every example in the public validation suite is
generic or independently written from scratch; a "dogfood-motivated" example is a clean-room kernel
that re-derives the same *shape*, never a copy. VeriCL itself carries no domain-specific concepts,
does not depend on that codebase, and must demonstrate its value on unrelated examples before
claiming general usefulness.

## Non-goals

- Verifying arbitrary Rust programs, or anything that is not a CubeCL kernel.
- Verifying CubeCL's compiler backends, drivers, or hardware — these are trusted and recorded as
  such.
- Guaranteeing bit-identical floating-point results across backends without explicit per-kernel
  support and evidence for that claim.
- Proving performance or algorithmic appropriateness.
- Recovering intent from arbitrary existing kernels automatically.
- Hiding assumptions to present a simpler correctness badge.

## Open decisions

- Whether the scalar reference execution is a derived interpretation of the cube function or a
  macro-generated twin function — decide when implementation reveals which stays honest with less
  machinery.
- The floating-point comparison model beyond declared per-kernel tolerances.
- The exact supported kernel subset for the bounds checker, and how it grows.
- Report format details; whether evidence manifests are committed or regenerated in CI.
- Whether later property classes (race freedom on shared memory, reduction-order sensitivity) come
  before or after a second proved property on the reference side via Kani/Verus.

Material choices get recorded with their alternatives and the claim boundary they create.

## Naming

**VeriCL** = verification for CubeCL. The `-CL` suffix deliberately ties the name to the foundation
this project committed to rather than staying backend-neutral; it was chosen only after the
CubeCL-only scope (see "Locked decisions") was locked in. The tagline — *one kernel contract,
equivalent implementations* — is now literal: one annotated CubeCL kernel, with its reference
execution and GPU realizations demonstrably equivalent under stated assumptions.

The project's working name during early, backend-neutral exploration was **Equik**.
[docs/ideation-charter.md](docs/ideation-charter.md), linked above under "Status", predates the
rename and still refers to the project by that name — it is an archived historical document and
is left as originally written rather than updated to match.
