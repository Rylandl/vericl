# VeriCL user guide

> From "I have a CubeCL kernel" to "`cargo test` verifies evidence" — in one sitting.

VeriCL is a conformance-and-evidence harness for [CubeCL](https://github.com/tracel-ai/cubecl)
compute kernels. You write a kernel once, attach the assumptions and properties that matter in a
`#[vericl::kernel(...)]` attribute, and VeriCL derives — from that single definition — a scalar
reference twin, generated input, a differential test against a real GPU backend, and (where the
kernel is in the supported subset) a machine-checked out-of-bounds-freedom proof. All of it runs
under plain `cargo test`, and it produces an **evidence manifest** that goes stale the moment the
kernel, its contract, or the toolchain changes.

This guide is written for a competent Rust/GPU developer who has never seen this repository. It
assumes you can already write and launch a `#[cube(launch)]` kernel. If you can't, read CubeCL's own
docs first; VeriCL sits *on top of* an ordinary CubeCL kernel and never replaces it.

If you want the design rationale rather than the how-to, the [README](../README.md) is a
charter-and-changelog; this document is the manual.

**Before you start, check whether your kernel is in scope:
[docs/coverage.md](coverage.md) — "Can I bring this kernel?"** — a per-kernel-class matrix
(elementwise, gather, stencil, reduction, RNG, atomics, 2-D, `plane_*`, tiling, f64, …) giving the
honest status of each, cited to real examples, plus the gap-closure plan for the classes that are not
supported yet. It will save you an afternoon if your kernel is one VeriCL rejects.

---

## Contents

1. [What you get, in one paragraph](#1-what-you-get-in-one-paragraph)
2. [Installation](#2-installation)
3. [Your first verified kernel](#3-your-first-verified-kernel)
4. [The contract clauses, built up](#4-the-contract-clauses-built-up)
5. [Generic and `#[comptime]` kernels: `instantiate(...)`](#5-generic-and-comptime-kernels-instantiate)
   - [5.1 Struct-typed `#[comptime]` config parameters: `vericl::config!`](#51-struct-typed-comptime-config-parameters-vericlconfig)
   - [5.2 Runtime struct parameters: `vericl::cube_struct!`](#52-runtime-struct-parameters-vericlcube_struct)
6. [Kernel composition: `#[vericl::helper]` + `uses(...)`](#6-kernel-composition-vericlhelper--uses)
7. [Cooperative kernels: shared-memory reductions](#7-cooperative-kernels-shared-memory-reductions)
8. [Image-space kernels: `dispatch(...)`](#8-image-space-kernels-dispatch)
9. [The `suite!` block](#9-the-suite-block)
10. [The `VERICL_UPDATE` workflow](#10-the-vericl_update-workflow)
11. [Reading an evidence file](#11-reading-an-evidence-file)
12. [Reading rejections](#12-reading-rejections)
13. [What VeriCL does not do](#13-what-vericl-does-not-do)
14. [Where to go next](#14-where-to-go-next)

---

## 1. What you get, in one paragraph

You add one attribute to a CubeCL kernel and list its name in a `suite!` block. On `cargo test`,
VeriCL: (a) generates random inputs that satisfy your declared `assumes(...)`, (b) runs the kernel on
a real GPU backend **and** runs an independently-derived scalar reference twin, (c) compares them
under a tolerance *you declared* and reports the first divergence with the buffer name and element
index, (d) discharges an SMT out-of-bounds-freedom proof over the kernel's CubeCL IR (if the kernel
is in the supported subset), and (e) writes all of that — bound to a content hash of the kernel — to
a JSON evidence file. Re-running `cargo test` **re-verifies** the evidence: any drift in the kernel,
the contract, or the toolchain is reported as a stale-evidence failure, not silently accepted.

The claims VeriCL records are never blurred together. A *tested* result ("agreed on these inputs, on
this backend") is a different thing from a *proved* result ("no in-bounds input can go out of
bounds"), which is different again from an *assumed* constraint and a *trusted* component. Section 10
explains each.

---

## 2. Installation

### 2.1 Rust

VeriCL builds with a recent stable Rust toolchain. Install via [rustup](https://rustup.rs) if you
haven't. CubeCL and wgpu are slow to compile unoptimized, so a dev profile with `opt-level = 1` is
worth setting (this repository does).

### 2.2 z3 (required for proofs)

The out-of-bounds-freedom and race-freedom **proofs** are discharged by the
[z3](https://github.com/Z3Prover/z3) SMT solver, invoked as a subprocess. VeriCL calls the `z3`
binary on your `PATH`. Install it:

| Platform | Command |
|---|---|
| macOS (Homebrew) | `brew install z3` |
| Debian / Ubuntu | `sudo apt install z3` |
| Fedora | `sudo dnf install z3` |
| Arch | `sudo pacman -S z3` |
| Windows (winget) | `winget install z3` |
| conda (any OS) | `conda install -c conda-forge z3` |

Verify it's found:

```console
$ z3 --version
Z3 version 4.16.0 - 64 bit
```

If `z3` is not on `PATH` when a suite has proofs enabled (the default), the test panics with an
actionable message naming the install command — it never silently skips the proof and records
"tested only". If you deliberately don't want proofs (for example on a machine without z3), set
`prove: false` in the `suite!` block (section 8) and VeriCL will omit the proved claims rather than
fake them.

### 2.3 A GPU backend

The differential test needs a real backend. On macOS/Windows/Linux with a GPU, the `wgpu` backend
(Metal/Vulkan/DX12) works out of the box. If you have no GPU, the `cubecl-cpu` backend runs on the
host CPU (it shares CubeCL's front end, so it is a weaker cross-check — see section 10 — but it
lets everything compile and run).

### 2.4 Cargo dependencies

Add three crates to your `Cargo.toml`:

```toml
[dependencies]
vericl = "0.1"
vericl-ir = "0.1"
cubecl = { version = "0.10", default-features = false, features = ["wgpu"] }
```

Why three:

- **`vericl`** — the macros (`#[vericl::kernel]`, `vericl::suite!`, …) and the evidence types. This
  crate deliberately has **no** CubeCL dependency, so your reference and evidence layer stays
  independent of the pipeline under test.
- **`vericl-ir`** — the IR extraction, identity hashing, and SMT prover. The `suite!` macro emits
  calls into this crate at your call site, so you must depend on it directly even though you never
  write `vericl_ir::` yourself. (It is a required dependency even with `prove: false`, because the
  IR-level identity hash is computed from it.)
- **`cubecl`** — your kernels are CubeCL kernels. Pick a backend feature (`wgpu` and/or `cpu`).
  CubeCL is pinned to an exact version by VeriCL (`=0.10.0`), so your `cubecl = "0.10"` resolves to
  that same version; a mismatched CubeCL is a compile error, not a silent incompatibility.

> Version note: until VeriCL is published to crates.io you can point these at a git revision or a
> local path instead (`vericl = { git = "https://github.com/Rylandl/vericl" }`); the crate names and
> the three-dependency shape are the same.

---

## 3. Your first verified kernel

Here is the whole thing, start to finish. We'll use a scaled vector add (`y := alpha*x + y`), the
canonical "saxpy". The only addition to an ordinary CubeCL kernel is the `#[vericl::kernel(...)]`
attribute above the usual `#[cube(launch)]`:

```rust
use cubecl::prelude::*;

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

Read the attribute top to bottom:

- **`assumes(...)`** — the conditions the kernel is claimed correct under. `x.len() == y.len()` is a
  buffer-length invariant; `alpha.abs() <= 4.0` and the two `iter().all(...)` clauses bound the input
  magnitudes. These are ordinary Rust boolean expressions. They become an executable predicate that
  generated inputs must satisfy, and a length invariant the prover can use.
- **`compare(abs = 1e-4)`** — how the reference and the GPU result are compared. `abs = X` means
  "pass when `|expected - actual| <= X`". (More modes in section 4.)
- **`gen(...)`** — how inputs are drawn. `alpha in -4.0..=4.0` draws a scalar in that inclusive
  range; `x in -100.0..=100.0` draws each *element* of the array in that range.
- **`instantiate(F = f32)`** — this kernel is generic over `F: Float`, so VeriCL needs a concrete
  type to monomorphize the twin and the launch at. Pin it to `f32`. (Section 5.)

Now list the kernel in a suite, in a normal integration test file (`tests/conformance.rs`):

```rust
use vericl_examples::*; // <- your own crate

vericl::suite! {
    runtime: cubecl::wgpu::WgpuRuntime,
    kernels: [axpy],
    evidence: "evidence/vericl.json",
}
```

Seed the evidence once, then verify it on every subsequent run:

```console
$ VERICL_UPDATE=1 cargo test        # writes evidence/vericl.json
$ cargo test                        # verifies it — this is your CI check
```

That's the whole loop. The first command generates the evidence file; the second re-runs everything
and fails if anything drifted. Commit `evidence/vericl.json` alongside your code — it is the record
of what was checked and under which assumptions.

### Why `abs`, and not an exact or ULP match?

The very first differential run of `axpy` caught the wgpu/Metal backend contracting `alpha*x + y`
into a fused multiply-add. Under catastrophic cancellation (`alpha*x ≈ -y`) the divergence from a
strict-rounding reference reached ~27,000 ULP — so no ULP bound is honest for this kernel on this
backend. The honest claim is an **absolute** error bound justified by the input ranges you declared:
one rounding of `alpha*x` with `|alpha| <= 4` and `|x| <= 100` is at most `ulp(400) ≈ 3.1e-5`, so
`abs = 1e-4` covers the contraction with margin. This is the general shape of a float tolerance in
VeriCL: **declared, and justified by `assumes(...)`**, never a magic number.

---

## 4. The contract clauses, built up

Start from the simplest possible contract and add one clause at a time.

### 4.1 The minimum: `assumes` + `compare`

An integer kernel that is bit-exact needs nothing but a length assumption and an exact compare. No
`gen(...)` is required — integer parameters default to full-range generation:

```rust
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(exact)
)]
#[cube(launch)]
pub fn xorshift_step(x: &Array<u32>, y: &mut Array<u32>) {
    if ABSOLUTE_POS < y.len() {
        let mut s = x[ABSOLUTE_POS];
        s ^= s << 13u32;
        s ^= s >> 17u32;
        s ^= s << 5u32;
        y[ABSOLUTE_POS] = s;
    }
}
```

`compare(exact)` is bit-for-bit equality and is the only mode for integer kernels.

### 4.2 The compare modes

| Clause | Meaning | Use for |
|---|---|---|
| `compare(exact)` | bit-for-bit equality | integer kernels |
| `compare(max_ulp = N)` | ULP distance `<= N` | float kernels the backend rounds identically to your twin |
| `compare(abs = X)` | `\|e - a\| <= X` | float kernels the backend may contract/reorder |
| `compare(abs = X, rel = Y)` | `\|e - a\| <= X + Y*\|e\|` | float kernels where the error scales with magnitude |

NaN on either side is always a failure, in every float mode. A tolerance is part of the contract and
is recorded in the evidence — pick the tightest one your input ranges honestly justify.

### 4.3 `gen(...)`: declaring how inputs are drawn

`gen(...)` declares, per parameter, how the conformance test draws inputs:

- `name in lo..=hi` — a scalar (or, for an array, applied to each element) drawn uniformly in that
  inclusive range.
- `len(name = N)` — pin an array's generated length to a constant `N` instead of the case size.
  Needed when an assumption constrains a length, e.g. a kernel with `assumes(y.len() == 1)` needs
  `gen(..., len(y = 1))`.

Two ergonomic rules to know:

- **Integer parameters left out of `gen(...)` default to full-range generation.** That's why
  `xorshift_step` above needs no `gen(...)`.
- **A float parameter with no declared range is a compile error**, not a silent default. Unbounded
  float generation produces NaN/inf-adjacent garbage and tolerances no `compare(abs = ...)` can
  honestly justify — so VeriCL makes you declare the range:

  ```text
  error: kernel `foo`: parameter `alpha` is a float with no declared gen(...) range — declare
  `gen(alpha in lo..=hi)`; unbounded float generation produces NaN/inf-adjacent garbage and
  un-provable tolerances
  ```

Generated inputs are drawn deterministically from a seeded PRNG in kernel-parameter declaration
order, checked against your `assumes(...)`, and resampled (same stream) up to 64 times if a draw is
rejected. A persistent failure means your declared ranges are inconsistent with your `assumes(...)`,
and the test says so by name.

### 4.4 `wrapping`: WGSL overflow semantics

WGSL wraps integer arithmetic on overflow, where Rust's default (debug) arithmetic panics. A kernel
that relies on wraparound — an integer hash/mixer with large odd multiplier constants, say —
declares `wrapping`, which folds the *reference twin's* `+`/`-`/`*`/`<<`/`>>` to their
`wrapping_*` forms. The `#[cube]` kernel itself is re-emitted untouched.

```rust
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(exact),
    wrapping
)]
#[cube(launch)]
pub fn mix_u32(x: &Array<u32>, y: &mut Array<u32>) {
    if ABSOLUTE_POS < y.len() {
        let mut h = x[ABSOLUTE_POS];
        h ^= h >> 16u32;
        h *= 0x85ebca6bu32;
        h ^= h >> 13u32;
        h *= 0xc2b2ae35u32;
        h ^= h >> 16u32;
        y[ABSOLUTE_POS] = h;
    }
}
```

`wrapping` is integer-only: every parameter must be an integer scalar or integer array when it's
declared (the fold is untyped and must not silently touch float math). Note that `wrapping` declares
wrap intent for *values* — a wrapped *index* is still out of bounds, so the prover treats a
`wrapping` kernel exactly like any other for bounds purposes.

---

## 5. Generic and `#[comptime]` kernels: `instantiate(...)`

Real kernels are usually generic over their element type (`<F: Float>`) and use `#[comptime]`
parameters for unroll/tap counts and feature toggles. VeriCL cannot derive a host twin from a still-
generic body (a trait-bound-but-unsubstituted `F::sqrt()` resolves to a panicking default rather
than the inherent `f32::sqrt`), so it requires you to pin every generic type and every `#[comptime]`
parameter to a concrete value with `instantiate(...)`:

```rust
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0),
    instantiate(F = f32, taps = 3)
)]
#[cube(launch)]
pub fn fir3<F: Float>(x: &Array<F>, y: &mut Array<F>, #[comptime] taps: u32) {
    if ABSOLUTE_POS < y.len() {
        let mut acc = x[ABSOLUTE_POS];
        if taps > 1 && ABSOLUTE_POS >= 1 {
            acc += x[ABSOLUTE_POS - 1];
        }
        if taps > 2 && ABSOLUTE_POS >= 2 {
            acc += x[ABSOLUTE_POS - 2];
        }
        y[ABSOLUTE_POS] = acc;
    }
}
```

`instantiate(F = f32, taps = 3)` names a concrete type for the `F` generic and a concrete value for
the `#[comptime] taps`. VeriCL monomorphizes everything it derives at those values: the twin becomes
`&[f32]`, the launch calls `axpy::launch::<f32, R>`, and the IR is extracted at `f32`. The pinned
values are part of the kernel's identity, so changing them re-stales the evidence.

Rules:

- v0 supports **exactly one** `instantiate(...)` clause per kernel (one monomorphization).
- Only plain type generics (`<F: Float>`) — no lifetimes, no const generics, no where-clauses.
- A generic/comptime kernel with **no** `instantiate(...)` is a targeted compile error telling you to
  add one; an `instantiate(...)` on a kernel with neither is also an error (an unused instantiation
  is a contract lie).
- Not every host float method is safe to call in the twin. A verified whitelist (`sqrt`, `abs`,
  `sin`, `exp`, `powf`, …) is allowed; a few (`erf`, `log1p`, `inverse_sqrt`, `is_inf`, …) panic on
  the host and are rejected at macro time by name, rather than silently miscomputing.

The `f64` tier works identically: `instantiate(F = f64)` monomorphizes at full f64 precision. One
platform caveat, stated loudly: **WGSL has no f64**, and CubeCL launches an f64 kernel on the
wgpu/Metal backend with no error and silently wrong results. So an f64 kernel's differential lane
must be `cubecl-cpu`, never wgpu (see section 8 and the README's "f64 support" section).

### 5.1 Struct-typed `#[comptime]` config parameters: `vericl::config!`

A `#[comptime]` parameter's type does not have to be a scalar. CubeCL lets you pass a whole
configuration struct or enum, and evaluates every `cfg.field` / `cfg.method()` as **ordinary host
Rust while the IR is built** — the config never reaches the GPU; only the constants it computes do.
This is how the CubeCL ecosystem configures nearly everything.

VeriCL supports it, with one requirement: **the config type and all of its impl blocks must be
declared inside a `vericl::config! { … }` block.**

```rust
vericl::config! {
    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    pub struct WindowCfg { pub taps: u32, pub gain: u32 }

    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    pub enum Weighting { Flat, Doubled }

    // A nested config: `WindowCfg` must be declared in this SAME block.
    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    pub struct StageCfg { pub window: WindowCfg, pub weighting: Weighting }

    impl WindowCfg {
        pub fn taps(&self) -> u32 { self.taps }
    }

    impl StageCfg {
        pub fn taps(&self) -> u32 { self.window().taps() }
        pub fn window(&self) -> WindowCfg { self.window }
        pub fn scale(&self) -> u32 {
            let base = self.window().gain;
            match self.weighting { Weighting::Flat => base, Weighting::Doubled => 2u32 * base }
        }
    }
}

#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(max_ulp = 0),
    gen(x in -10.0..=10.0, y in 0.0..=0.0),
    instantiate(cfg = StageCfg { window: WindowCfg { taps: 3, gain: 2 }, weighting: Weighting::Doubled })
)]
#[cube(launch)]
pub fn config_window_sum(x: &Array<f32>, y: &mut Array<f32>, #[comptime] cfg: StageCfg) {
    if ABSOLUTE_POS < y.len() {
        let mut acc = x[ABSOLUTE_POS];
        for j in 1..cfg.taps() {                    // a config method as a loop bound
            let idx = ABSOLUTE_POS + j as usize;
            if idx < x.len() { acc += x[idx]; }
        }
        let scale = comptime!(cfg.scale());         // a comptime! block over the config
        y[ABSOLUTE_POS] = acc * f32::cast_from(scale);
    }
}
```

`instantiate(...)`'s grammar is unchanged — a config is pinned exactly like a scalar.

**Why the declaration is required.** A kernel's `SOURCE_HASH` covers its own tokens and its contract
attribute's. A config type's *definition* is in neither. Before `vericl::config!` existed, editing
`total()` from `self.m * self.n` to `self.m + self.n` changed a kernel from ×24 to ×11 and left its
recorded identity bit-identical — the evidence still looked fresh. `vericl::config!` hashes the whole
block into a `CONFIG_HASH` that the kernel folds into its identity, so that edit now re-stales the
evidence, exactly the way editing a `uses(...)` helper's body does.

**If you forget it**, the error names the fix:

```text
error[E0277]: `TileCfg` is used as a struct-typed #[comptime] parameter but is not declared with a
              `vericl::config!` block
   |
   | pub fn k(x: &Array<f32>, y: &mut Array<f32>, #[comptime] cfg: TileCfg) {
   |                                                               ^^^^^^^ not a vericl config type
   = note: wrap the type AND its impl blocks in `vericl::config! { … }` so vericl can fold the
           config's definition into kernel identity and gate its method bodies for host-callability
```

**What you can pin.** A literal construction — a struct/enum literal, a unit variant, a path to a
`const`, nested compositions of those — or a call to a `const fn`. Anything else is rejected. The
reason is not tidiness: the pinned expression is evaluated *separately* for the reference twin, for
kernel expansion, and for IR extraction, so a value that differs between them produces evidence
describing a kernel that was never run. `const` is Rust's own guarantee that it cannot:

```text
error[E0015]: cannot call non-const function `cfg_from_env` in constants
   |
   |     instantiate(cfg = cfg_from_env())
   |                       ^^^^^^^^^^^^^^
```

**The caveat on const-evaluable pins, stated exactly.** `const` guarantees the value is the same for
every evaluation *within a build*. It does not guarantee it is a function of the source alone: a
`const` derived from `option_env!` or `cfg!` is const-evaluable and can still differ between two
builds of identical source.

```rust
pub const BUILD_M: u32 = match option_env!("MY_BUILD_M") { Some(_) => 9, None => 4 };
pub const ENV_CFG: PinCfg = PinCfg { m: BUILD_M };     // accepted: it really is a const
```

What VeriCL records is honest about this rather than silent: the pin's **expression text**
(`instantiate(cfg = ENV_CFG)`) is inside the contract-attribute tokens `SOURCE_HASH` hashes, so
editing the pin re-stales the evidence — but the *environment that resolved it* is not hashed, and
cannot be. So evidence produced by such a build is per-build deterministic, not per-source
reproducible: rebuilding the same source with a different `MY_BUILD_M` produces a kernel with the
same recorded identity and different behavior, and `ir_hash` — populated on every run, `prove: true`
or not, since extracting the expanded IR needs no solver — is what catches it. If reproducibility across builds matters to you, do not derive a pin from the build
environment — write the value out.

**What a config method body may contain.** Ordinary host Rust: field reads, arithmetic, `if`,
`match`, `let`, loops, calls into the pure part of `core`/`std`, to a primitive's associated
functions (`u32::max`) and inherent methods (`x.pow(2)`), and to anything else the same block
declares. What it may **not** contain, and why:

| Rejected in a `vericl::config!` block | Why |
|---|---|
| a call to a non-host-callable intrinsic (`fma`, `cast_from`, `mul_hi`, …) | it runs in the reference twin as host Rust and would panic there; you get a compile error at the callee instead |
| `#[cube]` on any impl or method | the twin would call the host body while the device gets the expanded one |
| a call to a **function declared outside the block** — as `helper(x)`, as `Self::helper(x)`, **or as `self.helper()`** | its body is neither hashed nor gated — move the function into the block |
| a read of a `const`/`static` declared outside the block, **including an associated `Self::K`** | same reason: the kernel's meaning would depend on something `CONFIG_HASH` cannot see |
| a method reached through a **user extension trait** (`self.m.boost()` with `impl Boost for u32` elsewhere) | same reason again, in the shape that is easiest to miss — only the `std` inherent surface is admitted on a receiver the block cannot type |
| anything **impure**: `std::env`, `std::process`, `std::time`, `std::fs`, `std::io`, `rand`, … (and `std::mem`, for target-dependence) | a config method is evaluated separately for the twin, for kernel expansion and for IR extraction; an answer that can differ between them makes the recorded evidence describe a kernel that was never run |
| a **custom derive** (`#[derive(CubeType)]`, `#[derive(serde::Serialize)]`) | the derive's *definition* decides what impls the type gets, and the hash covers only the invocation — the same reason a macro cannot declare a config type. `std` derives are fine |
| a `use` that **rebinds** `core`/`std`/`alloc`/a primitive name, or a glob `use` | the gates resolve path roots by name, so `use my::evil as core;` would re-point the whole standard-library allowance at user code |
| a generic config type (`Cfg<S>`) | one block is one hash, so every instantiation would share it |
| a field whose type — or a method's **return type** — is not a scalar primitive, `Self`, or another type declared in the same block | a nested config in a *sibling* block would escape the hash, and a return type whose methods live elsewhere would be ungated at the kernel's call site |
| a `static`, a `mod`, or any macro invocation (including `macro_rules!`-generated config types) | their contents are opaque to the gates, so hashing the block would not cover what the type is |

Each of these is a targeted message, and each exists so that the tokens VeriCL hashed really are the
tokens that determine what the kernel computes.

**Chains stop at the first link.** On the *kernel* side, a method call whose receiver **is** a config
parameter is exempt from VeriCL's Float/Numeric name list — its host-callability was already checked
where the config is declared, which is strictly stronger. That exemption covers one link and no
more: `cfg.dot()` compiles, `cfg.window().dot()` is rejected at `dot`'s own span, because the
justification applies to the method the config declares and not to whatever it returns. Write the
whole chain as one more config method (`cfg.window_dot()`), which is the better program anyway — the
computation is then hashed and gated instead of half-and-half.

**One residual, stated up front.** Rust lets you write an inherent `impl` for your own type anywhere
in the crate, and a `impl MyCfg { … }` written *outside* the `vericl::config!` block is invisible to
both the hash and the gates — a proc macro only sees the tokens it is handed. Keep every impl for a
config type inside its block. If you do not, the failure is loud rather than silent: a
non-host-callable call reached that way panics in the twin with `Unexpanded Cube functions should
not be called.`, which the differential lane reports as a failure. The *in-block* half cannot reach
into it, though — a config method may only call and read what the block declares, in every syntactic
form (bare call, `Self::`-qualified, and method syntax on `self` or on a field).

**Third-party config types are out of scope in v1, and that is a real limit.** A config type must be
*declared* inside a `vericl::config!` block. Rust's orphan rule would let you write
`impl ConfigIdentity for TheirCfg` in your own crate, but VeriCL never emits a bare impl: a hash over
tokens you did not write would certify nothing, and the gates would have nothing to walk. If you need
a config type from another crate, port the parts you use into a `vericl::config!` block of your own —
a clean-room port. It is more work, and it is also the only version that means anything, because the
hash then covers code that is actually in your repository.

**A type alias for a scalar works.** `type Taps = u32;` in `#[comptime]` position compiles: VeriCL's
macro cannot see through the alias (a proc macro has no name resolution), so it classifies `Taps` as
a config, but the requirement it emits — `<Taps as ConfigIdentity>::CONFIG_HASH` — is resolved by
*rustc*, which can. Scalars carry an identity naming the type, so retargeting the alias at another
primitive re-stales the evidence. An alias for a **struct** still needs that struct declared with
`vericl::config!`, and the error says so.

### 5.2 Runtime struct parameters: `vericl::cube_struct!`

Section 5.1 is about a struct that never reaches the device. This one is about a struct that does.

CubeCL lets a `#[cube]` item take a **runtime** struct — `args: &MyStruct` (or by value) where
`MyStruct` derives `CubeType`/`CubeLaunch`. It is lowered as a *positional flattening of its fields*
at that parameter's own slot: the same kernel with the fields spelled as loose parameters produces
bit-exact GPU output, an identical `KernelDefinition`, and a byte-identical `kernel_ir_hash`. So it
is a spelling, not a new capability — but the spelling is the ecosystem's, and before this milestone
VeriCL accepted half of it silently and hashed none of it.

Declare the type with `vericl::cube_struct!`:

```rust
use cubecl::prelude::*;   // the block emits CubeCL's derives, so the prelude must be in scope

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

You write no derives. `vericl::cube_struct!` emits `CubeType`, `CubeLaunch`, `Clone` and `Copy`
itself, and **rejects them if you write them** — an author-chosen derive set is a silent capability
switch (dropping `CubeLaunch` turns the type from launchable to device-local with your kernel's
tokens unchanged), and `Clone`/`Copy` are what let the generated twin bind the struct by value.

**And, when it can, four more.** If every field in the type's transitive shape is an integer, `bool`,
`char`, a unit enum declared in the same block, or another such struct, the macro *also* emits
`Debug`, `PartialEq`, `Eq` and `Hash` — the four CubeCL requires before a type may appear in
`#[comptime]` position — plus the `ConfigIdentity` impl. **That is the whole recipe**: such a type
serves *both* positions, `p: &T` and `#[comptime] p: T`, from one declaration and one hash. A type
with an `f32`/`f64` field anywhere gets neither, because no derive set can give `f32` `Hash` or `Eq`;
naming it in `#[comptime]` position is a `ConfigIdentity` error whose note says so. (Unlike the four
above, these are ordinary `std` derives — if you write one yourself, the macro simply does not
duplicate it.)

**Why the declaration is mandatory.** Two measured reasons, both about identity:

1. A struct type's *definition* is in neither input of `SOURCE_HASH`. Before this milestone, a
   `#[vericl::helper] fn use_pair(p: Pair)` compiled with **no diagnostic at all**, and editing a
   `#[cube] impl Pair { fn fold }` from `self.a * self.b` to `self.a + self.b` moved the twin from
   `[3, 6, 9, 12]` to `[4, 5, 6, 7]` while every recorded hash stayed bit-identical.
2. CubeCL fills a launch struct **by position**, so reordering two same-typed fields in the
   *declaration* changed what the kernel computed with the body and the launch call unchanged.
   VeriCL now emits that constructor from the order it hashed, so the reorder stays *correct* — and
   `STRUCT_HASH` moving is what makes your stored evidence correctly stale.

**The twin is your own struct.** No mirror type is generated: the twin takes `args: UniformArgs`,
reads `args.lower_bound` with the same tokens the device gets, and hands the whole value to a
`#[vericl::helper]` that takes `UniformArgs` too.

**What a field may be.**

- a runtime scalar: `f32`, `f64`, `u32`, `i32`, `u64`, `i64`. Each needs a
  `gen(p.field in lo..=hi)` range, generated exactly as a loose scalar parameter of that type is.
  (`usize`/`bool` are comptime-only: VeriCL has no scalar draw for them.)
- another struct declared **in the same block** — nested to any depth, with dotted clauses to match
  (`gen(cfg.window.gain in 0.5..=2.0)`).
- `#[cube(comptime)] pub taps: u32` — an integer, `bool`, `char`, a unit enum declared in the same
  block, or another declared struct whose own fields are all of those. It keeps its positional launch
  slot but takes the plain host type, never reaches the device, and is pinned once with
  `instantiate(cfg.window.taps = 3)`. A struct-typed one is pinned **whole**
  (`instantiate(p.win = Win { taps: 3, stride: 2 })`) — there is no per-sub-field `gen`/`instantiate`
  surface beneath it, and writing one is an `E0560` naming a type called
  `…__is_a_comptime_field_pinned_whole_by_instantiate`. No float anywhere in the shape: CubeCL's
  generated `CompilationArg` derives `Hash`/`Eq`, and `f32` is neither.

Write the field type **unqualified** — `u32`, not `sm::u32`, and `Inner`, not `other::Inner`. VeriCL
resolves a field type by the name of its last path segment, so a qualified path is rejected rather
than trusted: the tail of `sm::u32` says nothing about what it resolves to.

A field may carry only the bare `#[cube(comptime)]` marker and doc comments — every other attribute,
including `#[cfg]` and `#[cfg_attr]`, is rejected by name. `cfg_attr` in particular is rejected
*anywhere* in the block: rustc expands it after VeriCL has already classified the attribute, so it
would let the macro and the compiler disagree about which fields are comptime.

Buffer-valued fields (`Array`, `Tensor`, `Slice`, `View`, `Sequence`, `SharedMemory`) are
**deferred**, and the rejection names all four missing pieces rather than waving at "not supported".
Pass the buffer as its own `&Array<T>` / `&mut Array<T>` parameter.

**Field coverage is checked by rustc, not by VeriCL.** The macro annotating your kernel never learns
the struct's fields — it only sees the names you wrote in `gen(...)`/`instantiate(...)`. It emits them
as a literal of a generated spec type, so a field with no range is
**E0063: missing field `upper_bound`** and a misspelled one is **E0560**, both naming the field. The
same `const` binding is what const-evaluates every pinned comptime value (E0015 if it is impure) and
what guarantees `generate_case` and the IR extraction read the *same* tokens.

**Also rejected**, each with its own message: an `impl` block or `#[cube]` method inside the block; a
generic declared struct; `&mut P`; a struct or enum **return** type from a kernel or helper (a tuple
of scalars is fine — it is destructured at the call site); a payload-carrying runtime enum;
`wrapping` together with a struct parameter; and a `Vector` kernel with a struct parameter.

**The residual.** A `#[cube] impl` written *outside* the block escapes both the hash and the gates —
Rust allows an inherent impl anywhere in the crate and a proc macro sees only the tokens it is
handed. Worse than the config case, because `#[cube]` emits a host body *and* a device body, so the
failure is a numeric divergence rather than a panic. The differential lane is what catches it, and
`ir_hash` moves whenever the value reaches the device. Keep operations on a declared struct in
`#[vericl::helper]` free functions, where the twin is generated from the same tokens the device gets
and the body is gated.

---

---

## 6. Kernel composition: `#[vericl::helper]` + `uses(...)`

Kernels call other `#[cube]` functions. To let VeriCL follow the call into a device helper, annotate
the helper with `#[vericl::helper]` and declare the dependency on the calling kernel with `uses(...)`:

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
        y[ABSOLUTE_POS] = single_tap::<F>(x[ABSOLUTE_POS], gain);
    }
}
```

`#[vericl::helper]` re-emits the `#[cube]` function untouched and generates a host twin for it. The
kernel's `uses(single_tap)` clause rewrites its twin's call to `single_tap(...)` into a call to the
helper's twin. Helpers can call other helpers via their own `uses(...)` — the same mechanism, no
special-casing.

Two things to know:

- A helper with generic type parameters must be monomorphized via its **own** `instantiate(...)`,
  exactly like a kernel (same host-callability reason as section 5).
- A call in the twin body to a function that is neither `uses(...)`-listed, a local binding, nor a
  known host-safe free function is a targeted compile error naming the function and suggesting you
  add it to `uses(...)` and annotate it `#[vericl::helper]` — instead of a confusing type error deep
  in generated code.

Composition also carries into **identity**: a kernel's recorded identity folds in each used helper's
own identity hash (recursively), so a change two levels deep in a helper's body still re-stales the
top-level kernel's evidence. The bounds prover needs nothing special — CubeCL inlines a used helper's
IR into the kernel's own scope, so an obligation living inside a composed helper's body is walked
exactly as if it were written in the kernel.

---

## 7. Cooperative kernels: shared-memory reductions

A workgroup-cooperative kernel — one that uses `UNIT_POS`/`CUBE_DIM`, `SharedMemory`, and
`sync_cube()` barriers — cannot be modeled by the ordinary per-thread twin (a sequential twin has no
per-workgroup shared arena and no barriers). Opt into the cooperative machinery with
`cooperative(cube_dim = N)`:

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
    if ABSOLUTE_POS < input.len() {
        tile[tid] = input[ABSOLUTE_POS];
    } else {
        tile[tid] = 0.0f32;
    }
    sync_cube();

    let mut half = CUBE_DIM as usize / 2;
    while half > 0usize {
        if tid < half {
            let a = tile[tid];
            let b = tile[tid + half];
            tile[tid] = a + b;
        }
        sync_cube();
        half /= 2usize;
    }

    if tid == 0usize && CUBE_POS < output.len() {
        output[CUBE_POS] = tile[0usize];
    }
}
```

`cooperative(cube_dim = 256)` swaps in a **phase-split twin**: the body is split at each
`sync_cube()` into barrier-delimited segments, run per cube, per segment, per thread, with the shared
tile modeled as a per-cube array whose cells start **poisoned** — a read of a never-written cell
panics (a definedness bug surfaces as a reported finding, not a silent zero). `cube_dim` pins the
launch block size *and* the prover's `CUBE_DIM` binding.

A cooperative kernel earns two proved claims where the shape is in subset: `smt-oob-freedom` (bounds)
and `smt-race-freedom` (a GPUVerify-style two-thread symbolic reduction proving no two threads
collide within a barrier-delimited phase). Because the phase-split twin picks one intra-segment
thread order, its differential result is honest **only** under race freedom — so a cooperative tested
claim always names its dependence on race freedom explicitly (discharged by the proof, or carried as
an explicit assumption if the proof is disabled), and is refused if it has neither. The v1 subset is
the 1-D reduction shape; anything outside it (a barrier under a thread-varying condition, a
non-uniform tree loop, multiple tiles) is rejected with a targeted error rather than mis-modeled.

Cooperative kernels output one partial per workgroup, so the suite sizes each `&mut Array` output to
the cube count. Design detail lives in `docs/design-shared-memory.md`.

---

## 8. Image-space kernels: `dispatch(...)`

An image kernel wants to say "this thread is at `(x, y)`", not "this thread is number 4 217". Opt
into the multi-axis machinery with a `dispatch(...)` clause and index with the per-axis builtins:

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
        let y0 = u32::max(y, 1u32) - 1u32;
        let y2 = u32::min(y + 1u32, h - 1u32);
        let mut acc = 0f32;
        acc += inp[(y0 * w + x0) as usize];
        /* … seven more … */
        acc += inp[(y2 * w + x2) as usize];
        out[(y * w + x) as usize] = acc * 0.111111111f32;
    }
}
```

- **`cube_dim`** is a 2- or 3-tuple of positive integer **literals**, and its arity *is* the dispatch
  rank. Those literals are the single source of truth for three consumers: the launch's `CubeDim`,
  the twin's per-axis loop strides, and the prover's `CUBE_DIM_X/Y/Z` numerals. They must be pinned
  because that is what keeps every position recomposition linear — the same trade
  `cooperative(cube_dim = N)` already makes. Their product must be `<= 1024`.
- **`extents`** names this kernel's own runtime `u32` parameters carrying the problem extents. The
  harness binds them from each case's declared size, derives
  `CubeCount::Static(ceil(w/Wx), ceil(h/Wy), …)`, and sizes un-pinned buffers to their product.

The clause is required exactly when the body reads a per-axis builtin, and rejected when it does not
— the same biconditional `cooperative(...)` has, for the same reason: a clause changes the launch
shape, the twin's iteration space and the recorded evidence, so declaring an unused one is a
contract lie.

### 8.1 X is the fastest-varying axis

`ABSOLUTE_POS_X` moves along a row; a row-major image of width `w` is indexed
`inp[(y * w + x) as usize]`. Writing `inp[(x * h + y) as usize]` is *in bounds* and *transposed* —
**VeriCL's proof will not catch it**, because a transposed image is a functional bug, not a
memory-safety one, and the auto-derived reference twin mirrors the *same* index math, so the
differential does not see it either unless you supply an independent `reference = fn`.

A **different** mistake — transposing the *clause*, `dispatch(extents = (h, w))`, against a body
that still guards `ABSOLUTE_POS_X < w` and `ABSOLUTE_POS_Y < h` — IS rejected, at compile time, by
the clause/body consistency gate (design §13 risk 6): the differential cannot catch it because the
twin's grid and the GPU launch derive from the same clause, so a swap moves both together. That
gate is **conservative** — it only cross-checks axes the body guards with the canonical
`ABSOLUTE_POS_a < <extent>` form; an axis with no such guard, or one whose bound is a non-bare
expression (`w - 1`, a `min`), is not checked (see §13).

This is not a convention VeriCL chose. `CubeCount::Static(x, y, z)` reaches
`dispatch_workgroups(x, y, z)` reaches `workgroup_id.x/.y/.z` with no transposition at any layer,
and every flatten is row-major with X fastest, consistently across WGSL, CUDA/HIP, SPIR-V and the
CPU runtime — measured with 0 violations in 1 212 threads across 6 launch shapes.

### 8.2 The length assume is what makes any of it provable

`out.len() == (w as usize) * (h as usize)` is not ergonomics. A 1-D kernel that decodes
`row = ABSOLUTE_POS / w` gets `row * w <= ABSOLUTE_POS` for free from Euclidean division, so its row
stride is bounded. `ABSOLUTE_POS_Y * w` has no such parent: `abs_y` and `w` are unrelated leaves, so
the multiply's no-overflow side-obligation is *unprovable* and the index is `OutOfSubset`. With the
product assume, `abs_y <= h-1` gives `abs_y * w <= w*h - w = len - w`, and the whole 3x3 clamped
stencil discharges.

**Write it widen-then-multiply.** `out.len() == (w * h) as usize` multiplies in `u32` and then
widens, so the executable `check_assumes` predicate tests the **wrapped** product while the prover
would assert the mathematical one. Measured, those disagree: at `w = 2, h = 2147483649` the wrapped
product is `2`, so a length-2 buffer satisfies the clause while the model believes the length is
4 294 967 298 — and an index of 2 then proves in bounds against a buffer that does not have it. That
spelling is rejected by name, at the cast.

Because the assume is *binary*, a rank-3 volume index needs `len == w*h*d`, which has no expressible
form: a 3-D dispatch runs, launches and twins correctly, but its bounds claim is `OutOfSubset`.

### 8.3 Clamp branch-free, or not at all

The idiomatic stencil clamp

```rust
let mut x2 = x;
if x + 1 < w { x2 = x + 1; }        // rejected: x2 is tainted after the arm
```

writes a mutable local inside a branch arm. VeriCL taints such a variable once the arm closes —
adversarial review round 2 found that *not* doing so was a confirmed false `Proved` on a real
out-of-bounds write — so the neighbour index built from `x2` is unmodelable, and no amount of
per-axis machinery changes that. Write it branch-free instead:

```rust
let x2 = u32::min(x + 1u32, w - 1u32);
let x0 = u32::max(x, 1u32) - 1u32;
```

Both compute the identical function under the guard `x < w`, and both lower to a single arithmetic
instruction the prover models exactly.

### 8.4 What a 2-D suite looks like

A dispatch kernel's cases are per-axis **extents**, not thread counts, so they need their own
`suite!` with tuple `sizes:` — and no `cube_dim:` field, because the clause already pins it:

```rust
vericl::suite! {
    runtime: cubecl::wgpu::WgpuRuntime,
    kernels: [elementwise2d_scale, transpose2d, box_blur3x3, topology_report2d],
    evidence: "evidence/vericl_2d.json",
    sizes: [(37, 19), (64, 64), (1, 1), (3, 129), (129, 3), (255, 257)],
}
```

The claim's config records `sizes_unit: "extents"`, the full pinned `cube_dim` triple and the `rank`,
so a reader can tell what launch shape the evidence was produced under — which 1-D evidence could not
say before this milestone. Because the product assume is nonlinear, the proved claim's recorded
`logic` reads `QF_NIA` for these kernels rather than `QF_LIA`.

### 8.5 What `dispatch(...)` does not admit

Each of these is a targeted compile error naming the reason, not a silent approximation:

| Written | Why not |
|---|---|
| flat `ABSOLUTE_POS` / `CUBE_POS` / `CUBE_COUNT` inside the clause | in a multi-axis dispatch `ABSOLUTE_POS != CUBE_POS * CUBE_DIM + UNIT_POS` — measured, the two disagree for 912 of 960 threads at `CubeCount(5,3,1) x CubeDim(8,8,1)`, and in 533 of 722 swept launch shapes. Flat `CUBE_DIM` and `UNIT_POS` **are** kept |
| a per-axis builtin with no clause | add the clause; the message says so |
| `ABSOLUTE_POS_Z` under a 2-tuple `cube_dim` | the arity is the rank; a Z read under a rank-2 launch is a constant 0, and accepting it would silently change a strided walk into one |
| `dispatch(...)` + `cooperative(...)` | 2-D shared-memory tiles are deferred: the intra-cube race obligation discharges in under 10 ms, the inter-cube one times out in z3 at 180 s |
| `dispatch(...)` + `Vector<P, W>` | a vector suite's sizes are *lines*, a dispatch suite's are *extents*; two units in one evidence config, undecided |
| `dispatch(...)` + a runtime `cube_struct!` parameter | CubeCL flattens a struct's fields into the same scalar registration counter the extents are numbered by, and this macro cannot see the field types |
| a non-literal `cube_dim` entry, or a product above 1024 | the prover needs numerals and the twin needs a loop stride; 1024 is the measured `max_units_per_cube` on wgpu/Metal (the WebGPU default is 256) |
| `gen(w in …)` for an extent | an extent is bound from the case size, never drawn |

Design and measurements: `docs/design-2d-dispatch.md`.

---

## 9. The `suite!` block

`vericl::suite!` expands to a single `#[test] fn vericl_conformance()`. It runs every listed kernel's
conformance case across the declared sizes, discharges the SMT proofs, and assembles the evidence
manifest. The full field set:

```rust
vericl::suite! {
    runtime: cubecl::wgpu::WgpuRuntime,          // required: the backend to run on
    kernels: [axpy, xorshift_step, mix_u32],     // required: kernels to check
    evidence: "evidence/vericl.json",            // required: manifest path (relative to crate root)
    // --- optional fields, with their defaults ---
    // sizes: [1, 7, 256, 1000, 1027, 4096, 65536],
    // seed: 0xE901,
    // cube_dim: 256,
    // prove: true,
    // frontend_independent: <derived from `runtime:` — see below>,
    // extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime),
}
```

- **`runtime`** — the backend runtime path. `cubecl::wgpu::WgpuRuntime` for GPU;
  `cubecl::cpu::CpuRuntime` for the host CPU backend.
- **`kernels`** — the list of kernel names. Each must carry `#[vericl::kernel]` (see section 12 for
  the error you get if one doesn't). Every kernel in one suite must agree on the case *unit* — all
  1-D (scalar `sizes`) or all `dispatch(...)` of the same rank (tuple `sizes`, section 8.4); a
  mismatch is a compile error naming the kernel. Adding a fourth honest kernel is one name here, not new
  boilerplate.
- **`evidence`** — the manifest path, relative to `CARGO_MANIFEST_DIR` (your crate root).
- **`sizes`** — the buffer sizes to test. Defaults to a spread from 1 to 65536 including
  non-multiples of `cube_dim` (which is where off-by-one and clamping bugs hide). In a
  `dispatch(...)` suite these are per-axis **extents** tuples instead — `[(37, 19), (64, 64), …]` —
  and `cube_dim:` must be absent (the clause pins it; declaring it twice is rejected).
- **`prove`** — whether to run the SMT proofs. Default `true`; set `false` to omit proved claims (and
  drop the z3 requirement) rather than fake them.
- **`extra_lane`** — an additional differential lane behind a `cfg`, e.g. the `cubecl-cpu` backend
  under `--features cpu`. It is folded into the *same* test (two independent tests sharing one
  evidence file would race), and its claims are recorded as *not* front-end-independent (see section
  10). A cpu extra-lane appears only when you build with that feature.
- **`frontend_independent`** — whether the primary runtime is an execution lane independent of the
  CubeCL front end the kernel goes through. **Normally you omit it**: it is derived from `runtime:`
  (`WgpuRuntime` → independent, trusted list records "GPU hardware"; `CpuRuntime` → shared front end,
  trusted list records "host CPU execution hardware" plus the explicit caveat that only the derived
  twin is independent). It is not a defaulted bool — writing `true` on `CpuRuntime` is a compile
  error, and an unrecognized runtime must declare which it is, because neither default is safe there.
  A literal `true`/`false` only. Declaring `false` on a recognized-independent runtime is always
  allowed: downgrading to the weaker claim cannot overstate anything.

One suite invocation always produces exactly one manifest. Use a second `suite!` in a second test
file for a kernel that needs a different runtime (the f64-on-cpu case is
`tests/conformance_f64.rs` → `evidence/vericl_f64.json`).

---

## 10. The `VERICL_UPDATE` workflow

There is no separate CLI. Conformance is a `cargo test` citizen.

```console
# Seed or regenerate the evidence (writes evidence/*.json):
$ VERICL_UPDATE=1 cargo test

# Verify against committed evidence — the CI check (fails on missing/stale/mismatched):
$ cargo test

# Also exercise the cubecl-cpu lane, if the suite declares an extra_lane:
$ cargo test --features cpu
```

The mental model:

- **`VERICL_UPDATE=1 cargo test`** runs everything and *writes* the manifest. It refuses to store
  failing evidence — if a differential check or proof fails, it panics telling you to fix the kernel
  or contract first, so you can never bake a red result into the record.
- **`cargo test`** (no env var) runs everything and *verifies* against the committed manifest,
  **completely**: the stored claim set has to be this build's claim set, field for field. See
  "What `verify` compares" in section 11 for the table and the normalization rules.

You commit the evidence files. A reviewer diffing a PR sees exactly which claims changed. A change to
the kernel body without re-running `VERICL_UPDATE` fails with an identity mismatch naming both the
source and IR hash — the whole point.

> Tip: when regenerating with multiple feature sets, run the *default* `VERICL_UPDATE=1 cargo test`
> **last**, so the committed evidence is left in the default (non-cpu) shape. A `--features cpu`
> update leaves cpu-lane claims in the default manifest, and the next plain `cargo test` reports each
> of them as a claim the build did not produce. The reverse direction is fine and is the intended
> steady state: running `cargo test --features cpu` against default-shape evidence prints the extra
> lane's claims as a note (`vericl note — N item(s) of evidence produced by this build are not
> recorded in …`) and passes.

---

## 11. Reading an evidence file

An evidence manifest is JSON: a `vericl_version`, a `provenance` record, and a list of `entries`, one
per kernel.

The `provenance` record is the **verification environment** the file was produced in — the toolchain,
the pinned crate versions, the solver, the execution lanes, and the device:

```json
"provenance": {
  "rustc": "rustc 1.94.0 (4a4ef493e 2026-03-02)",
  "target": "aarch64-apple-darwin",
  "vericl": "0.1.0",
  "vericl_ir": "0.1.0",
  "vericl_macros": "0.1.0",
  "cubecl": "=0.10.0",
  "z3": "z3 Z3 version 4.16.0 - 64 bit",
  "lanes": ["\"wgpu<wgsl>\""],
  "device": "Metal"
}
```

Every one of those changes what the evidence *means* while leaving kernel identity bit-identical: the
reference twin is compiled by that rustc, the kernel under test goes through that cubecl, the proofs
were discharged by that z3, and `"wgpu<wgsl>"` on Metal and on Vulkan are two different code
generators reporting one name. So an evidence file carried to a different environment is **stale**,
in the same class as a kernel edit, and `verify` says so with the field and both values. What it does
**not** cover is stated in the `vericl::provenance` module docs — `RUSTFLAGS`, cargo profile,
transitive dependency versions, and GPU driver builds are all invisible to it.

Here is `axpy`'s entry (abridged):

```json
{
  "kernel": "axpy",
  "identity": {
    "source_hash": "sha256:0f202b53…",
    "vericl_version": "0.1.0",
    "ir_hash": "sha256:3ae1a32f…"
  },
  "contract": {
    "assumes": ["x.len() == y.len()", "alpha.abs() <= 4.0", "…"],
    "compare": "f32 |e-a| <= 1e-4 + 0e0*|e|",
    "wrapping": false,
    "instantiate": ["F = f32"],
    "uses": []
  },
  "claims": [
    {
      "kind": "tested",
      "check": "differential",
      "backend": "\"wgpu<wgsl>\"",
      "config": { "cube_dim": 256, "seed": 59649, "sizes": [1, 7, 256, 1000, 1027, 4096, 65536],
                  "reference": "vericl-macros sequential twin" },
      "result": { "status": "pass" }
    },
    {
      "kind": "proved",
      "check": "smt-oob-freedom",
      "config": { "logic": "QF_LIA", "obligations": 3, "solver": "z3 Z3 version 4.16.0 - 64 bit" },
      "result": { "status": "pass" }
    }
  ],
  "trusted": [
    "rustc codegen of the reference twin",
    "vericl-macros source-to-reference derivation",
    "\"wgpu<wgsl>\" buffer upload/readback integrity",
    "GPU hardware",
    "the solver binary (z3 …) discharging the SMT bounds obligations",
    "…"
  ]
}
```

### The four claim categories

VeriCL's whole discipline is that these mean different things and are never presented as
interchangeable:

- **`tested`** — behavior *observed* on specific generated inputs, on a specific backend, driver, and
  device. `axpy`'s `differential` claim: the GPU output matched the reference twin, within the
  declared tolerance, across all listed sizes. It says nothing about inputs not drawn.
- **`proved`** — a property *machine-checked* by a solver over the kernel's IR, under the stated
  assumptions. `axpy`'s `smt-oob-freedom` claim: every array index provably stays in bounds for
  *every* in-bounds dispatch (3 obligations discharged UNSAT in QF_LIA by z3). Cooperative kernels
  can additionally carry `smt-race-freedom`.
- **`assumed`** — a *declared* constraint that the other claims lean on but do **not** establish. The
  `compare` tolerance and the input ranges are assumptions; a cooperative kernel with proofs disabled
  carries an explicit `intra-phase-race-freedom` assumed claim rather than silently trusting it.
- **`trusted`** — components *outside* the checked boundary, listed in each entry's `trusted` array:
  CubeCL's backend codegen, the driver, the GPU hardware, and — for a proof — the z3 binary and
  VeriCL's own obligation encoding. Source-level evidence never silently implies these are verified.

### `identity` and staleness

The `identity` binds the claims to the exact kernel they were produced from: a `source_hash` (source
tokens + contract + vericl version, composition-aware for `uses(...)` kernels) and an `ir_hash`
(content hash of the expanded CubeCL IR). `verify` rejects any entry whose stored identity differs
from the freshly built one — that is what "stale evidence" means. Both hashes are reported on a
mismatch, so a source edit and a codegen change are distinguishable.

### What `verify` compares

Everything the manifest records. The normalization is deliberate in both directions — insensitive
where order carries no information, sensitive where it does:

| part | compared as | order |
|---|---|---|
| `vericl_version` | exact | — |
| `provenance` | exact per field; `lanes` subset-checked | `lanes` preserved |
| entries | keyed by kernel name; a duplicate name is refused | insensitive |
| `identity` | exact, per field, all reported on a mismatch | — |
| `contract` | exact, per field, each named | **sensitive** (authored, hash-covered) |
| claim set | multiset on `(kind, check, backend)`, then `(kind, check)` | insensitive |
| claim `config` | structural JSON diff to a dotted path | objects insensitive, **arrays sensitive** |
| claim `result` | exact | — |
| `trusted` | a **set** — order- and duplicate-insensitive | insensitive |

Claim order is an artifact of the pipeline (tested is pushed first, a cooperative kernel *inserts* its
tested claim at the front, an extra lane appends), so it is ignored. A `sizes` array is a declared
sequence, so it is not: reordering it re-stales the evidence. Sensitivity is the safe direction —
being wrong there costs one regeneration, being wrong the other way lets a real change through.

The property, stated once: **the stored claim set must be this build's claim set**. A claim the file
records that the build does not produce, a claim the build produces that the file does not record, a
mutated backend / seed / size list / solver / obligation count / result, an erased trust dependency —
each is a named problem with the stored and current values shown. There is exactly one exemption, and
it is scoped by the provenance record rather than by shape: a claim or trust entry contributed by an
execution lane the stored `lanes` says did not run (the `extra_lane` case), which is printed as a
note.

### Independence of lanes

The differential twin is derived by VeriCL's macros and shares **only source text** with the kernel —
it is genuinely independent of CubeCL's pipeline. A `cubecl-cpu` extra lane, by contrast, shares
CubeCL's front end (macro expansion + IR) with the kernel under test, so it is recorded as **not** an
independent reference. For an f64 kernel — where wgpu is unusable — the macro-derived twin is the
*sole* independent leg.

Which of the two an entry records is **derived from `runtime:`**, not defaulted: `WgpuRuntime`
resolves to the independent lane (trusted list records "GPU hardware"), `CpuRuntime` to the shared
front end (trusted list records "host CPU execution hardware" plus an explicit caveat that only the
derived twin is independent). Writing `frontend_independent: true` on `CpuRuntime` is a compile
error, and a runtime VeriCL does not recognize must declare which it is — neither default is safe
there, so the macro asks. The field takes a literal `true`/`false` only; a claim that depends on a
runtime value cannot be checked when it is made.

---

## 12. Reading rejections

VeriCL rejects constructs it cannot faithfully model, at compile time, rather than silently
approximating them. Rejections come in three flavors: **VeriCL's own** targeted messages, a couple of
**rustc-mediated** cases VeriCL deliberately delegates to the compiler, and **run-time** panics. Here
are the common ones and what to do.

### VeriCL compile-time rejections

| You see | It means | Do |
|---|---|---|
| `` `UNIT_POS` is a workgroup-cooperative construct outside the ordinary vericl v0 subset; add a `cooperative(cube_dim = N)` clause `` | You used shared-memory/barrier topology in an ordinary kernel | Add `cooperative(cube_dim = N)` (section 7) |
| `` … has generic type parameters and/or #[comptime] parameters but no instantiate(...) clause `` | A generic or `#[comptime]` kernel needs a pinned value | Add `instantiate(F = f32, …)` (section 5) |
| `` …declares instantiate(...) but has no generic … to instantiate — remove the clause `` | An `instantiate(...)` on a non-generic kernel | Remove it (an unused instantiation is a contract lie) |
| `` parameter `alpha` is a float with no declared gen(...) range `` | A float input with no range | Add `gen(alpha in lo..=hi)` (section 4.3) |
| `` call to `foo` in the reference twin is not recognized as a local binding, a declared helper, … `` | The twin calls a function VeriCL can't follow | Annotate `foo` with `#[vericl::helper]` and add it to `uses(foo)` (section 6) |
| `` host-callability of `F::erf` in the reference twin is unverified `` | A float method that panics on the host | Use a whitelisted method, or precompute it (section 5) |
| `` `<construct>` is outside the vericl v0 kernel subset; … Rewrite the kernel within the supported subset … or see the rejection reference in docs/guide.md `` | A construct VeriCL doesn't model (`return`, `plane_*`, `PLANE_DIM`, `Atomic`, `View`, `terminate!`, …) | Rewrite within the supported subset, below |
| `` `ABSOLUTE_POS_X` is a per-axis topology builtin outside the ordinary vericl v0 subset; add a `dispatch(cube_dim = (Wx, Wy), extents = (w, h))` clause `` | You indexed by axis in a kernel with no dispatch clause | Add the clause (section 8) |
| `` `ABSOLUTE_POS` is outside the vericl v0 subset in a `dispatch(...)` kernel — in a multi-axis dispatch it is NOT `CUBE_POS * CUBE_DIM + UNIT_POS` … `` | You mixed the flat and the per-axis addressing schemes | Index with the per-axis builtins, or drop the clause and stay flat (section 8.5) |
| `` this kernel declares `dispatch(...)` but its body reads no per-axis topology builtin `` | An unused dispatch clause — it still changes the launch shape and the evidence | Remove the clause, or index by axis |
| `` kernel `k` declares no `&mut Array<T>` output parameter, so there is nothing for the differential to compare `` | The kernel writes no output buffer. Every case would report zero compared parameters, and `all()` over zero reports is `true` — a passing `tested` claim that established nothing | Add an `&mut Array<T>` output, or make it a `#[vericl::helper]` |
| `` gen(...) pins `len(y) = 0` `` | A pinned buffer length of zero compares no elements — and `gen(...)` is not in the contract record, so the claim would advertise the suite's sizes having compared nothing | Use a positive length |
| `` suite!: `kernels: []` declares a conformance suite over no kernels `` | A suite that checks nothing and prints `vericl evidence OK` | List a kernel, or delete the `suite!` |
| `` suite!: `sizes: []` declares a differential over no cases `` | `all()` over zero outcomes is `true`: every kernel would record a passing `tested` claim having executed nothing | Declare a size, or omit the field for the default list |
| `` suite!: a `sizes:` entry of 0 runs a case that compares ZERO elements `` | Same vacuity one level down | Use a positive size (`1` is the honest degenerate case) |
| `` suite!: this suite's `runtime:` … shares CubeCL's front end … would record a claim that is not true `` | `frontend_independent: true` on a runtime that is not an independent lane | Remove the field — it is derived from `runtime:` |
| `` suite!: vericl does not recognize this runtime … `` | A runtime VeriCL has not measured; neither lane-independence default is safe | Declare `frontend_independent: true` or `false` (the message says what each records) |
| `` suite!: `frontend_independent:` takes a literal `true` or `false`, not an expression `` | A claim selected by a runtime value cannot be checked when it is made | Write the literal |
| `` `ABSOLUTE_POS_Z` names the Z axis, which this kernel's `dispatch(...)` clause does not enable `` | A rank-3 read under a 2-tuple `cube_dim` | Widen the clause to a 3-tuple, or drop the Z read (section 8.5) |
| `` `out.len() == (w * h) as usize` multiplies in u32 and then widens … `` | The wrapping product spelling — a measured false `Proved` | Write `out.len() == (w as usize) * (h as usize)` (section 8.2) |
| `` `dispatch(cube_dim = ...)` takes 2 or 3 positive integer *literals* `` / `` … has 2048 units per cube, above the 1024 `` | A runtime or over-large cube dim | Pin literals whose product is `<= 1024` (section 8) |
| `` `dispatch(...)` and `cooperative(...)` are mutually exclusive `` | 2-D shared-memory tiles are deferred, with the cost measured | See section 8.5 and `docs/design-2d-dispatch.md` §8 |

The **supported v0 kernel subset** is: affine `ABSOLUTE_POS` indexing; bounded `for` and `match`;
`&Array<T>`/`&mut Array<T>` and core `Slice`; `#[comptime]` and generic parameters pinned via
`instantiate(...)`; the `wrapping` clause for integer overflow; behind
`cooperative(cube_dim = N)` — workgroup shared memory with barriers; and behind
`dispatch(cube_dim = (…), extents = (…))` — per-axis 2-D/3-D topology (section 8). Constructs
*outside* it are rejected rather than approximated: unbounded `while`/`loop`, stepped/descending
range loops, `return`, `plane_*` reductions (including the `PLANE_DIM`/`PLANE_POS` constants and the
whole `CUBE_*_CLUSTER*` family), `Atomic*`, the `View`/`Layout` strided-tensor machinery,
`terminate!()` outside the cooperative uniform guard, and 2-D *shared-memory tiles* are all future
work.

For the same boundary organized by **kernel class** rather than by construct — "is my gather / my
stencil / my histogram / my 2-D image kernel in scope, and what exactly is proved about it?" — see
[docs/coverage.md](coverage.md), which also records which classes are planned and which are
deliberately out.

### Rustc-mediated rejections (delegated to the compiler, by design)

Four safety catches are enforced by rustc on the *generated twin* or on generated `const` items, not
by a VeriCL message — this is deliberate (the compiler is a stronger oracle than a macro pass), so
recognize them for what they are:

- **Overlapping mutable slices** surface as a borrow-checker error **E0499** ("cannot borrow … as
  mutable more than once at a time") or **E0502** on your `.slice_mut(...)` calls. That is the
  intended aliasing catch — VeriCL maps a mutable slice to a Rust `&mut [_]` subslice precisely so the
  borrow checker rejects a genuinely-unsafe overlapping-write kernel. A VeriCL-authored, buffer-named
  diagnostic for this is future work (`docs/design-view-slice.md` §8.4).
- **A kernel listed in `suite!` without its `#[vericl::kernel]` attribute** surfaces as a plain rustc
  resolution error at the `suite!` site ("failed to resolve: use of undeclared … `<name>_vericl`", or
  "cannot find function `conformance_case`"). The `suite!` macro can't see whether a name is an
  annotated kernel, so it can't pre-empt this. The fix is always: add `#[vericl::kernel(...)]` (and
  `#[cube(launch)]`) to the kernel, or remove the name from `kernels:`.

- **A struct-typed `#[comptime]` parameter whose type is not declared with `vericl::config!`**
  surfaces as **E0277** on the `ConfigIdentity` trait, with a VeriCL-authored
  `#[diagnostic::on_unimplemented]` message, the label on the parameter's type and a help pointing at
  the type's definition. Rustc renders it because the requirement *is* a trait bound — that is what
  makes the declaration impossible to skip (section 5.1).
- **A non-const-evaluable `instantiate(...)` value for a config parameter** surfaces as **E0015**
  ("cannot call non-const function … in constants") at the value's own span, from the `const` binding
  VeriCL generates for each pinned config value. The syntactic half of that gate is a
  VeriCL-authored message; const-evaluability is delegated because Rust's own `const` rules are
  exactly the purity guarantee needed (section 5.1).

### Run-time panics

- **`proved claims require z3 on PATH (macOS: brew install z3; …)`** — a suite with `prove: true` (the
  default) but no `z3`. Install z3 (section 2.2) or set `prove: false`.
- **`gen(...) could not produce inputs satisfying assumes(...) after 64 resample attempts`** — your
  declared `gen(...)` ranges are inconsistent with your `assumes(...)`. Widen the ranges or relax the
  assumption so a draw can satisfy it.
- **`STALE evidence — identity mismatch`** — the kernel/contract/IR/version changed without renewing
  evidence. Re-run `VERICL_UPDATE=1 cargo test` (after reviewing that the change was intended).
- **`evidence downgraded — stored evidence has a proved … claim that the current build did not
  produce`** — you lost a proof (z3 missing, or `prove: false`) that the committed evidence has.
  Restore z3/prove, or regenerate the evidence if the downgrade is intended.

---

## 13. What VeriCL does not do

Read this section before you rely on a green run. VeriCL is deliberately narrow, and its honesty
depends on you knowing the boundary.

- **It does not verify CubeCL's backends, drivers, or hardware.** The proof is about the CubeCL IR;
  the codegen below it, the driver, and the GPU are **trusted** and recorded as such. A `proved`
  claim is not a guarantee against a codegen or hardware bug.
- **A `tested` claim is not a proof.** It is agreement on the *generated inputs*, on *one* backend and
  device. It says nothing about inputs not drawn, or about a different GPU. Only a `proved` claim
  quantifies over all in-bounds inputs, and only for the property it names (today: out-of-bounds
  freedom and race freedom).
- **It does not prove functional correctness.** VeriCL proves out-of-bounds freedom and (for
  cooperative kernels) race freedom. It does **not** prove your kernel computes the right answer — the
  differential test checks the kernel against a twin *derived from the same source*, so a bug present
  in both is invisible to it. (An independent IR interpreter cross-check shrinks *model*-fidelity risk
  empirically, but it, too, is a `tested` observation, not a proof.)
- **Image transposition is only partly caught.** Two distinct mistakes: (1) transposing the
  `dispatch(extents = ...)` *clause* relative to the body's per-axis guards — `extents = (h, w)`
  with a body guarding `ABSOLUTE_POS_X < w`, `ABSOLUTE_POS_Y < h` — is **rejected at compile time**
  by the clause/body consistency gate (design §13 risk 6); neither the differential nor the bounds
  proof can see it, because the twin grid and the GPU launch share the clause and a swap moves both
  together. That gate is *conservative*: it cross-checks only axes the body guards with the
  canonical `ABSOLUTE_POS_a < <extent>` form, so an axis with no such guard, or one bounded by a
  non-bare expression (`w - 1`, a `min(...)`), is **not** checked. (2) Transposing the *index math*
  itself — `inp[(x * h + y)]` where the clause and guards agree — is a functional bug the bounds
  proof does not target, and the auto-derived twin mirrors the same math, so the differential does
  not catch it either unless you supply an independent `reference = fn`.
- **It does not guarantee bit-identical floats across backends.** Float equivalence is claimed only
  within your declared per-kernel tolerance, and recorded as an assumption.
- **It does not verify arbitrary Rust, or anything that isn't a CubeCL kernel.**
- **It does not recover intent from an existing kernel automatically, or prove performance or
  algorithmic appropriateness.**
- **The supported kernel subset is narrow (section 12).** Whole classes of real kernels — `plane_*`
  reductions, 2-D *shared-memory tiles*, `Tensor`/`View` strided machinery, atomics — are out of scope
  for v0 and rejected explicitly, not approximated. (Custom `CubeType` struct arguments and 2-D/3-D
  dispatch *were* on this list and are now supported, sections 5.2 and 8;
  [docs/coverage.md](coverage.md) is the maintained per-class status.)
- **No claim constrains the launch shape you choose.** VeriCL's claims are about the kernel *as
  launched by the suite*: `kernel::launch::<R>(…)` is an ordinary public CubeCL entry point taking
  any `CubeCount`/`CubeDim`, and nothing stops you calling it differently. For the value claims this
  is benign — `ABSOLUTE_POS` is the row-major flatten of the grid, and as long as it does not wrap it
  is a bijection onto `0..num_threads`, exactly the twin's iteration space. The residual is the wrap:
  above `2^32` threads two distinct threads receive the same `ABSOLUTE_POS`, and an
  `out[ABSOLUTE_POS] = …` kernel then has a write-write race no claim covers. That is reachable only
  on a multi-axis grid (a 1-D dispatch tops out at `65535 x 1024 < 2^32` under the measured per-axis
  caps) and needs a deliberate, enormous launch — on the reference adapter,
  `CubeCount(2048, 2048, 1) x CubeDim(32, 32, 1)`. Differential claims now *record* the launch shape
  they were produced under (`rank`, and the full `cube_dim` triple for a dispatch kernel); nothing
  makes a hand-written launch honour it. Note also that `cubecl_core::calculate_cube_count_elemwise`
  and `cube_count_spread` will silently turn a 1-D request above ~65535 cubes into a *multi-axis*
  grid, so a kernel certified as 1-D can reach a 2-D launch without asking for one.
- **`f64` has no front-end-independent lane on a wgpu-only machine.** WGSL has no f64; the honest lane
  is cubecl-cpu, which shares CubeCL's front end. For an f64 kernel the macro-derived twin is the sole
  independent reference.

None of these are hidden: every trusted component is listed in the evidence, every assumption travels
with the result, and every out-of-subset construct is a compile error rather than a silent
approximation. That is the point — a simpler-looking correctness badge would be a dishonest one.

---

## 14. Where to go next

- **`README.md`** — the design decisions, the claim model, and the CubeCL-semantics findings behind
  each clause.
- **`docs/design-shared-memory.md`** — the cooperative/phase-split twin and the two-thread race proof.
- **`docs/design-view-slice.md`** — core `Slice` support and the aliasing story.
- **`docs/design-line-vector.md`** — `Vector<P, N>` (SIMD) element support.
- **`docs/interpreter.md`** — the independent IR interpreter cross-check and exactly what its
  agreement does and does not establish.
- **`docs/certificates-decision.md`** — why solver proof certificates are deferred, and the path to
  enabling them.
- **The example kernels** in `crates/vericl-examples/src/lib.rs` — every construct in this guide has a
  real, tested example there, wired into `crates/vericl-examples/tests/conformance.rs`.
