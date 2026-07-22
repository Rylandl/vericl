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
pub mod panic;
pub mod rng;
pub mod trust;

pub use compare::{
    CompareReport, Mismatch, compare_exact_u32, compare_f32, compare_f32_absrel,
    compare_f32_with, compare_u32_with, ulp_distance_f32,
};
pub use contract::{Compare, Contract, ContractRecord, Identity, StructuredAssume};
pub use evidence::{
    CaseOutcome, Claim, ClaimKind, ClaimResult, Entry, Manifest, describe_case_outcome,
    differential_config, proved_config, verify,
};
pub use panic::catch_reference_panic;
pub use rng::SplitMix64;
pub use trust::{
    GPU_HARDWARE_TRUST, backend_buffer_trust, proved_bounds_trust, reference_twin_trust,
    shared_frontend_lane_trust,
};
pub use vericl_macros::{kernel, suite};

/// VeriCL version, recorded in every identity and manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
