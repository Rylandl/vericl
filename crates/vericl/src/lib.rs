//! VeriCL: conformance and evidence harness for CubeCL compute kernels.
//!
//! One annotated kernel is the single point of custody. From it, the
//! `#[vericl::kernel]` macro derives a sequential scalar reference twin, an
//! executable `assumes` predicate, and a source-level identity hash. This
//! crate provides the comparison, input-generation, and evidence-manifest
//! machinery around those generated artifacts.
//!
//! Claim vocabulary (see README): **proved** / **tested** / **assumed** /
//! **trusted** are distinct and never presented interchangeably.
//!
//! This crate deliberately does not depend on CubeCL: the reference twin and
//! the evidence layer must stay independent of the pipeline under test.

pub mod compare;
pub mod contract;
pub mod evidence;
pub mod rng;

pub use compare::{
    CompareReport, Mismatch, compare_exact_u32, compare_f32, compare_f32_absrel, ulp_distance_f32,
};
pub use contract::{Compare, Contract, ContractRecord, Identity, StructuredAssume};
pub use evidence::{CaseOutcome, Claim, ClaimKind, ClaimResult, Entry, Manifest, verify};
pub use rng::SplitMix64;
pub use vericl_macros::kernel;

/// VeriCL version, recorded in every identity and manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
