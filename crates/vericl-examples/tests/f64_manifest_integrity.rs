//! FIX 3 (round-13A) — verify `evidence/vericl_f64.json`'s INTEGRITY in the
//! **default** `cargo test` run, where the f64 lane itself cannot execute.
//!
//! # The gap this closes
//!
//! `conformance_f64.rs` is `#![cfg(feature = "cpu")]`: WGSL has no f64, so the
//! kernel can only run honestly on cubecl-cpu, and the whole suite compiles to
//! nothing under the default (wgpu-only) `cargo test`. That left
//! `vericl_f64.json` checked *only* under `cargo test --features cpu`, and
//! `evidence_tamper.rs` only checks it against itself — so a plain `cargo test`
//! (the README's "the whole CI story") would happily pass while the committed
//! f64 manifest described a **different kernel, contract, or toolchain** than
//! the source in this tree.
//!
//! # What is checkable without the cpu runtime
//!
//! A kernel's identity (`source_hash` + `ir_hash`), its contract, and the
//! toolchain fingerprint are all derivable from source + codegen alone — no
//! launch, no z3, no cpu backend. `kernel_definition()` builds the expanded IR
//! (the same call the `suite!` runner folds into `ir_hash`), and it works in the
//! default build: the in-crate unit test
//! `axpy_f64_kernel_definition_is_provably_in_bounds` already relies on that.
//! So this recomputes each f64 kernel's identity + contract from *this tree's*
//! source and refuses the manifest if it does not match — catching a source
//! edit, a contract edit, a vericl/cubecl bump, or a rustc/target change that
//! left the committed f64 evidence stale.
//!
//! # What is NOT checked here (and why that is honest)
//!
//! The differential *tested* claim itself — that cubecl-cpu's f64 output agrees
//! with the sequential twin over the recorded sizes — is a statement about a
//! **run**, and that run needs the cpu backend. It is verified only under
//! `cargo test --features cpu` (by `conformance_f64.rs`). This file deliberately
//! does not assert the claim's result; it asserts the manifest is *bound to the
//! right kernel and toolchain*, which is the part a default run can prove.

use vericl::{Provenance, Manifest};
use vericl_examples::*;

fn f64_manifest() -> Manifest {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("evidence/vericl_f64.json");
    Manifest::load(&path).unwrap_or_else(|e| panic!("load {}: {e}", path.display()))
}

/// The identity + contract the committed f64 manifest records must be the
/// identity + contract this tree's `axpy_f64` source produces right now. Both
/// halves of the identity are recomputed independently: `source_hash` (kernel
/// tokens + contract + vericl version) and `ir_hash` (the expanded CubeCL IR,
/// via the same `kernel_ir_hash(kernel_definition())` the runner uses).
#[test]
fn f64_manifest_identity_and_contract_match_this_source() {
    let m = f64_manifest();
    let entry = m
        .entries
        .iter()
        .find(|e| e.kernel == "axpy_f64")
        .expect("vericl_f64.json must carry the axpy_f64 entry");

    // Fresh identity, derived from source with no cpu runtime.
    let mut fresh = axpy_f64_vericl::identity();
    fresh.ir_hash = Some(vericl_ir::kernel_ir_hash(&axpy_f64_vericl::kernel_definition()));

    assert_eq!(
        entry.identity.source_hash, fresh.source_hash,
        "vericl_f64.json's source_hash does not match this tree's axpy_f64 — the kernel source or \
         contract changed without regenerating f64 evidence (VERICL_UPDATE=1 cargo test --features cpu)"
    );
    assert_eq!(
        entry.identity.ir_hash, fresh.ir_hash,
        "vericl_f64.json's ir_hash does not match this tree's expanded axpy_f64 IR — a codegen-level \
         change (e.g. a cubecl bump) left the f64 evidence stale; regenerate under --features cpu"
    );
    assert_eq!(
        entry.identity.vericl_version, fresh.vericl_version,
        "vericl_f64.json was written by a different vericl version"
    );

    // The contract, field for field, recomputed from source.
    assert_eq!(
        entry.contract,
        axpy_f64_vericl::contract().record(),
        "vericl_f64.json's contract record does not match this tree's axpy_f64 contract"
    );
}

/// The toolchain fingerprint fields a default run CAN re-derive (rustc, target,
/// the crate versions, the cubecl pin) must match the environment running now.
/// The lane/device/z3/salt_scheme fields are set by the cpu-lane runner and are
/// not re-derivable here — they are covered by the `--features cpu` verify — so
/// this checks only what it can honestly recompute.
#[test]
fn f64_manifest_toolchain_fingerprint_is_current() {
    let m = f64_manifest();
    let cur = Provenance::current();
    let p = &m.provenance;

    assert!(p.is_recorded(), "vericl_f64.json carries no provenance record");
    assert_eq!(p.rustc, cur.rustc, "vericl_f64.json's rustc is stale — regenerate under --features cpu");
    assert_eq!(p.target, cur.target, "vericl_f64.json's target triple is stale");
    assert_eq!(p.vericl, cur.vericl, "vericl_f64.json's vericl version is stale");
    assert_eq!(p.cubecl, cur.cubecl, "vericl_f64.json's cubecl pin is stale");
    assert_eq!(
        p.vericl_ir,
        vericl_ir::VERSION,
        "vericl_f64.json's vericl-ir version is stale"
    );
    // The lane recorded is the cpu lane (WGSL cannot run f64); pin that it is
    // the shared-front-end lane and not, say, a wgpu lane hand-edited in.
    assert_eq!(
        p.lanes,
        vec!["\"cpu\"".to_string()],
        "vericl_f64.json must record exactly the cubecl-cpu lane as its (sole, non-independent) \
         execution lane — WGSL has no f64, so a wgpu lane here would be a fiction"
    );
}
