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

---

## Contents

1. [What you get, in one paragraph](#1-what-you-get-in-one-paragraph)
2. [Installation](#2-installation)
3. [Your first verified kernel](#3-your-first-verified-kernel)
4. [The contract clauses, built up](#4-the-contract-clauses-built-up)
5. [Generic and `#[comptime]` kernels: `instantiate(...)`](#5-generic-and-comptime-kernels-instantiate)
6. [Kernel composition: `#[vericl::helper]` + `uses(...)`](#6-kernel-composition-vericlhelper--uses)
7. [Cooperative kernels: shared-memory reductions](#7-cooperative-kernels-shared-memory-reductions)
8. [The `suite!` block](#8-the-suite-block)
9. [The `VERICL_UPDATE` workflow](#9-the-vericl_update-workflow)
10. [Reading an evidence file](#10-reading-an-evidence-file)
11. [Reading rejections](#11-reading-rejections)
12. [What VeriCL does not do](#12-what-vericl-does-not-do)
13. [Where to go next](#13-where-to-go-next)

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

## 8. The `suite!` block

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
    // frontend_independent: true,
    // extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime),
}
```

- **`runtime`** — the backend runtime path. `cubecl::wgpu::WgpuRuntime` for GPU;
  `cubecl::cpu::CpuRuntime` for the host CPU backend.
- **`kernels`** — the list of kernel names. Each must carry `#[vericl::kernel]` (see section 11 for
  the error you get if one doesn't). Adding a fourth honest kernel is one name here, not new
  boilerplate.
- **`evidence`** — the manifest path, relative to `CARGO_MANIFEST_DIR` (your crate root).
- **`sizes`** — the buffer sizes to test. Defaults to a spread from 1 to 65536 including
  non-multiples of `cube_dim` (which is where off-by-one and clamping bugs hide).
- **`prove`** — whether to run the SMT proofs. Default `true`; set `false` to omit proved claims (and
  drop the z3 requirement) rather than fake them.
- **`extra_lane`** — an additional differential lane behind a `cfg`, e.g. the `cubecl-cpu` backend
  under `--features cpu`. It is folded into the *same* test (two independent tests sharing one
  evidence file would race), and its claims are recorded as *not* front-end-independent (see section
  10). A cpu extra-lane appears only when you build with that feature.
- **`frontend_independent`** — set `false` for a suite whose primary runtime shares CubeCL's front
  end with the kernel (the f64/cubecl-cpu case), so the trusted list records "host CPU execution
  hardware" and the shared-front-end caveat honestly, rather than implying an independent GPU lane.

One suite invocation always produces exactly one manifest. Use a second `suite!` in a second test
file for a kernel that needs a different runtime (the f64-on-cpu case is
`tests/conformance_f64.rs` → `evidence/vericl_f64.json`).

---

## 9. The `VERICL_UPDATE` workflow

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
- **`cargo test`** (no env var) runs everything and *verifies* against the committed manifest. It
  fails, with the problem list, if the evidence is missing, if the kernel's identity has drifted
  (source, contract, IR, or vericl version), or if a claim now fails. It also catches a **downgrade**:
  if the committed evidence has a proved claim the current build no longer produces (z3 went missing,
  or `prove: false`), that is a reported problem, not a silent pass.

You commit the evidence files. A reviewer diffing a PR sees exactly which claims changed. A change to
the kernel body without re-running `VERICL_UPDATE` fails CI with an identity mismatch naming both the
source and IR hash — the whole point.

> Tip: when regenerating with multiple feature sets, run the *default* `VERICL_UPDATE=1 cargo test`
> **last**, so the committed evidence is left in the default (non-cpu) shape. Running a
> `--features cpu` update last would leave cpu-lane claims in the default manifest and the next plain
> `cargo test` would report them as unexpected.

---

## 10. Reading an evidence file

An evidence manifest is JSON: a `vericl_version` and a list of `entries`, one per kernel. Here is
`axpy`'s entry (abridged):

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

### Independence of lanes

The differential twin is derived by VeriCL's macros and shares **only source text** with the kernel —
it is genuinely independent of CubeCL's pipeline. A `cubecl-cpu` extra lane, by contrast, shares
CubeCL's front end (macro expansion + IR) with the kernel under test, so it is recorded as **not** an
independent reference. For an f64 kernel — where wgpu is unusable — the macro-derived twin is the
*sole* independent leg, which is why the f64 suite declares `frontend_independent: false` and records
"host CPU execution hardware" honestly.

---

## 11. Reading rejections

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
| `` `<construct>` is outside the vericl v0 kernel subset; … Rewrite the kernel within the supported subset … or see the rejection reference in docs/guide.md `` | A construct VeriCL doesn't model (`return`, `plane_*`, `Atomic`, `View`, `terminate!`, …) | Rewrite within the supported subset, below |

The **supported v0 kernel subset** is: affine `ABSOLUTE_POS` indexing; bounded `for` and `match`;
`&Array<T>`/`&mut Array<T>` and core `Slice`; `#[comptime]` and generic parameters pinned via
`instantiate(...)`; the `wrapping` clause for integer overflow; and — behind
`cooperative(cube_dim = N)` — workgroup shared memory with barriers. Constructs *outside* it are
rejected rather than approximated: unbounded `while`/`loop`, stepped/descending range loops,
`return`, `plane_*` reductions, `Atomic*`, the `View`/`Layout` strided-tensor machinery,
`terminate!()` outside the cooperative uniform guard, and 2-D topology are all future work.

### Rustc-mediated rejections (delegated to the compiler, by design)

Two safety catches are enforced by rustc on the *generated twin*, not by a VeriCL message — this is
deliberate (the compiler is a stronger oracle than a macro pass), so recognize them for what they
are:

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

## 12. What VeriCL does not do

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
- **It does not guarantee bit-identical floats across backends.** Float equivalence is claimed only
  within your declared per-kernel tolerance, and recorded as an assumption.
- **It does not verify arbitrary Rust, or anything that isn't a CubeCL kernel.**
- **It does not recover intent from an existing kernel automatically, or prove performance or
  algorithmic appropriateness.**
- **The supported kernel subset is narrow (section 11).** Whole classes of real kernels — `plane_*`
  reductions, custom `CubeType` struct arguments, 2-D topology, `Tensor`/`View` strided machinery,
  atomics — are out of scope for v0 and rejected explicitly, not approximated.
- **`f64` has no front-end-independent lane on a wgpu-only machine.** WGSL has no f64; the honest lane
  is cubecl-cpu, which shares CubeCL's front end. For an f64 kernel the macro-derived twin is the sole
  independent reference.

None of these are hidden: every trusted component is listed in the evidence, every assumption travels
with the result, and every out-of-subset construct is a compile error rather than a silent
approximation. That is the point — a simpler-looking correctness badge would be a dishonest one.

---

## 13. Where to go next

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
