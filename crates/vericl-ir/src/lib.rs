//! CubeCL IR access for vericl: kernel identity hashing and an SMT-checked
//! out-of-bounds-freedom prover over the CubeCL IR (`Scope`).
//!
//! This crate is the only place in vericl that depends on `cubecl` beyond the
//! macro/example layer's own dependency — `crates/vericl` (the core evidence
//! and comparison library) stays cubecl-free by design (see README "Locked
//! decisions": "isolate all IR-facing code in one crate"). It does not
//! depend on `crates/vericl` either, so the interface here is deliberately
//! minimal and self-describing rather than sharing types with the contract
//! layer; callers (the conformance harness) translate between the two.

/// The `vericl-ir` crate version, recorded in an evidence manifest's
/// verification-environment fingerprint (`vericl::Provenance`).
///
/// `vericl` core cannot read it — it does not depend on this crate (the
/// dependency runs the other way, and only through the `suite!`-generated code
/// at the user's call site), so the `suite!` runner reads it from here.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

mod fuzz;
mod hash;
mod interp;
mod prover;

pub use fuzz::{CorpusReport, Finding, FindingKind, run_corpus};
pub use hash::kernel_ir_hash;
pub use interp::{
    Buffer, Dispatch3, Inputs, Oob, Outcome, ScalarBinding, Val, interpret_dispatch,
};
pub use prover::{
    Assume, BufferParam, CooperativeObligations, CooperativeProof, ProveResult,
    SMT_OOB_FREEDOM_CHECK, SMT_RACE_FREEDOM_CHECK, prove_bounds_freedom,
    prove_bounds_freedom_cooperative, prove_bounds_freedom_dispatch, prove_cooperative,
    prove_race_freedom, z3_version,
};
