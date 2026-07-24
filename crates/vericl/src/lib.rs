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
#![warn(missing_docs)]

// ---------------------------------------------------------------------------
// Public surface. What a user's own code touches is small and stable:
//
//   * the macros — `#[vericl::kernel]`, `#[vericl::helper]`,
//     `#[vericl::reference]`, and `vericl::suite!` (the authoring interface);
//   * the comparison vocabulary — [`Compare`];
//   * the evidence-reading types — [`Manifest`], [`Entry`], [`Claim`],
//     [`ClaimKind`], [`ClaimResult`], [`ContractRecord`], [`Identity`], and
//     [`verify`] — for anyone parsing or checking an `evidence/*.json` file
//     programmatically.
//
// Everything else exported below is `pub` only because the *generated* code
// (the `suite!` test body and the derived reference twins) references it across
// the crate boundary at the user's call site — it is not an API a user calls
// directly, and is marked `#[doc(hidden)]` so it does not clutter the rustdoc.
// See docs/guide.md and docs/release-checklist.md ("API surface") for the
// stability policy on these items before 1.0.
// ---------------------------------------------------------------------------

pub mod compare;
pub mod contract;
pub mod evidence;
pub mod rng;

// Plumbing modules: referenced only by macro-generated twin / suite code
// (`::vericl::host_shims::…`, `::vericl::trust::…`, `::vericl::Line`), never by
// user code — the kernel author writes `Vector<P, N>`/`SharedMemory` and the
// macro derives the `Line`/`SharedTile` twin.
#[doc(hidden)]
pub mod host_shims;
#[doc(hidden)]
pub mod line;
#[doc(hidden)]
pub mod panic;
#[doc(hidden)]
pub mod shared;
#[doc(hidden)]
pub mod trust;

// --- User-facing re-exports ---
pub use compare::{
    CompareReport, Mismatch, compare_exact_u32, compare_f32, compare_f32_absrel, compare_f64,
    compare_f64_absrel, ulp_distance_f32, ulp_distance_f64,
};
pub use contract::{Compare, Contract, ContractRecord, Identity};
pub use evidence::{
    CaseOutcome, Claim, ClaimKind, ClaimResult, Entry, Manifest, describe_case_outcome, verify,
};
pub use rng::SplitMix64;
pub use vericl_macros::{helper, kernel, reference, suite};

// --- Generated-code plumbing (pub for cross-crate use, hidden from docs) ---
#[doc(hidden)]
pub use compare::{compare_f32_with, compare_f64_with, compare_u32_with};
#[doc(hidden)]
pub use contract::{
    MAX_HELPER_COMPOSITION_DEPTH, StructuredAssume, check_helper_composition_depth,
    combine_source_hash,
};
#[doc(hidden)]
pub use evidence::{
    RACE_FREEDOM_ASSUMPTION_CHECK, RaceDependency, SMT_RACE_FREEDOM_CHECK,
    cooperative_differential_config, differential_config, differential_vector_config,
    proved_bounds_cooperative_config, proved_config, proved_race_config,
    race_freedom_assumption_claim,
};
#[doc(hidden)]
pub use line::Line;
#[doc(hidden)]
pub use panic::catch_reference_panic;
#[doc(hidden)]
pub use shared::SharedTile;
#[doc(hidden)]
pub use trust::{
    GPU_HARDWARE_TRUST, HOST_HARDWARE_TRUST, backend_buffer_trust, proved_bounds_trust,
    proved_race_freedom_trust, reference_twin_trust, shared_frontend_lane_trust,
};

/// VeriCL version, recorded in every identity and manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
