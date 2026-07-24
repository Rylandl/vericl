# Release checklist

The exact sequence to publish VeriCL to crates.io, plus the decisions that stay with the maintainer.
Everything up to (but not including) `cargo publish` has been prepared and dry-run-verified;
publishing itself is **gated on Ryland** and is deliberately not automated.

## What ships

| Crate | crates.io | Notes |
|---|---|---|
| `vericl` | **published** | the macros + evidence types; front page is the workspace `README.md` |
| `vericl-macros` | **published** | proc-macros; `vericl` depends on it |
| `vericl-ir` | **published** | IR/SMT plumbing — **required by every `suite!` call site**, so it must be public even though users never write `vericl_ir::` themselves |
| `vericl-examples` | **not published** | `publish = false`; example kernels + tests, depends on backend features |

Why `vericl-ir` is published: the `vericl::suite!` macro emits `::vericl_ir::` paths (IR hashing +
the SMT prover) that compile at the *user's* call site — even with `prove: false`, because the
IR-level identity hash still comes from it. A consumer crate therefore depends on `vericl`,
`vericl-ir`, **and** `cubecl` (with a backend feature). This is documented in `docs/guide.md` §2.4.

## Pre-publish state (already done)

- [x] crates.io metadata on all three published crates: `description`, `license`
      (`MIT OR Apache-2.0`), `repository` (`https://github.com/Rylandl/vericl`), `keywords`,
      `categories`. `vericl` additionally sets `readme = "../../README.md"` (cargo copies it into the
      package as `README.md` — verified in the dry-run tarball).
- [x] `vericl-ir` made publishable (removed `publish = false`).
- [x] Inter-crate deps carry both a path and a version (`{ path = …, version = "0.1.0" }` in
      `[workspace.dependencies]`), so `cargo publish` rewrites them to registry version deps
      automatically. Verified: the generated `vericl` manifest resolves `vericl-macros = "0.1.0"`.
- [x] `cargo publish --workspace --dry-run --allow-dirty` passes for all three crates (packages them
      together with a local temp registry so the inter-crate deps resolve). `vericl-examples` is
      correctly excluded.
- [x] Full workspace test green (default + `--features cpu`), clippy clean on both, demo-defects
      exit 0, evidence byte-identical.

## Publish sequence (maintainer runs these)

`cargo publish` for a crate with an internal dependency needs that dependency already on the index,
so **order matters**. Two options:

### Option A — one workspace command (cargo ≥ 1.90)

```console
$ cargo publish --workspace
```

cargo computes the dependency order and publishes `vericl-macros`, `vericl-ir`, and `vericl`
together, waiting for the index between them. This is the recommended path on the pinned toolchain
(cargo 1.94).

### Option B — one crate at a time (explicit order)

```console
$ cargo publish -p vericl-macros
$ cargo publish -p vericl-ir          # independent of vericl-macros; order between these two is free
# wait for the crates.io index to update (usually < 1 min)
$ cargo publish -p vericl             # resolves vericl-macros from the index
```

Note: `cargo publish -p vericl --dry-run` on its own will fail with "no matching package named
`vericl-macros`" until `vericl-macros` is actually published — that is expected, not a defect
(a standalone dry-run resolves deps against the registry). Use Option A's `--workspace --dry-run` to
verify `vericl` before anything is live.

## Post-publish smoke test (do this once, from crates.io)

The dry-runs verify packaging in the workspace context. Confirm the *published* crates work for a
real external consumer:

```console
$ cargo new /tmp/vericl-smoke && cd /tmp/vericl-smoke
$ cargo add vericl vericl-ir
$ cargo add cubecl --no-default-features --features wgpu
# add the axpy kernel + a one-kernel suite! from docs/guide.md §3
$ VERICL_UPDATE=1 cargo test && cargo test
```

If this compiles and passes, the three-dependency story in the guide is confirmed end to end from
the registry.

## Decisions that remain the maintainer's

These are deliberately **not** decided here — they are Ryland's calls:

1. **Whether to publish at all, and when.** This checklist prepares everything; it does not pull the
   trigger.
2. **Crate-name ownership on crates.io.** `vericl`, `vericl-macros`, and `vericl-ir` must be
   available/owned. Verify (`cargo owner --list <name>` post-publish) and reserve the names before a
   third party can. If any name is taken, a rename ripples through the macro-emitted `::vericl::` /
   `::vericl_ir::` paths and must be decided before publishing.
3. **The version number.** Everything is `0.1.0` today. A first public release could stay `0.1.0`
   (signaling pre-1.0, breaking-change-allowed — appropriate given the "pre-1.0 API notes" below) or
   jump to a chosen number. The version is workspace-wide (`[workspace.package].version`).
4. **The `cubecl` pin.** VeriCL pins `cubecl = "=0.10.0"` exactly, by design. Publishing locks
   external users to that exact CubeCL until the prover is ported to a newer CubeCL (a known roadmap
   item). Confirm this is the intended coupling to ship.
5. **Yanking / re-publishing policy** if an early version needs to be withdrawn.

## Pre-1.0 API notes (compat)

The published `vericl` surface is intentionally small (macros + `Compare` + the evidence-reading
types; see the crate-root docs). Everything else that is `pub` is generated-code plumbing marked
`#[doc(hidden)]`. Before 1.0, the following may still change and are **not** covered by a stability
promise:

- The `#[doc(hidden)]` plumbing (`differential_config`, the `trust::*` helpers, `combine_source_hash`,
  `StructuredAssume`, `Line`, `SharedTile`, …) — it is `pub` only because macro-generated code at the
  user's call site references it; treat it as private.
- The evidence JSON schema. `ContractRecord`/`Identity` already use `#[serde(default)]` on
  newer fields so older manifests still load, but the schema is not yet frozen.
- The exact wording of rejection messages and the boundaries of the supported kernel subset (both
  grow as the prover and twin grow).
