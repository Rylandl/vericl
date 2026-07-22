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

mod hash;
mod prover;

pub use hash::kernel_ir_hash;
pub use prover::{Assume, BufferParam, ProveResult, prove_bounds_freedom, z3_version};
