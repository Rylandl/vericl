//! f64 conformance suite — the differential + bounds-proof evidence for the
//! f64 example kernel(s), run on the **cubecl-cpu** lane.
//!
//! Why a separate suite (not a lane on `conformance.rs`): an f64 kernel cannot
//! run on the primary wgpu lane — WGSL has no f64, and cubecl 0.10 launches an
//! f64 kernel on wgpu/Metal with no error and no panic but silently wrong
//! results (verified empirically; see the README "f64 support" section and
//! `tests/f64_wgpu_unsound.rs`). cubecl-cpu runs f64 at full precision, so it is
//! the honest execution lane. A second `suite!` invocation with its own
//! evidence file is the established precedent (`conformance.rs` +
//! `cooperative_fallback.rs`), and satisfies the M6 constraint that two
//! `#[test]`s must not share one evidence file.
//!
//! This whole file is `#[cfg(feature = "cpu")]`: under the default `cargo test`
//! (wgpu only) it compiles to nothing and `evidence/vericl_f64.json` is not
//! touched. It is verified/regenerated under `cargo test --features cpu`.
//!
//! Trust note: the cpu lane shares cubecl's front end (macro expansion + IR)
//! with the kernel under test, exactly like the f32 cpu extra-lane. Unlike f32
//! — where wgpu is a genuinely different backend — f64 has NO front-end-
//! independent execution lane on this platform, so the macro-derived sequential
//! twin (`reference_twin_trust`) is the sole independent leg here.
//!
//! Usage:
//!   cargo test --features cpu                  verify evidence/vericl_f64.json
//!   VERICL_UPDATE=1 cargo test --features cpu  regenerate it
#![cfg(feature = "cpu")]

use vericl_examples::*;

vericl::suite! {
    runtime: cubecl::cpu::CpuRuntime,
    kernels: [axpy_f64],
    evidence: "evidence/vericl_f64.json",
    // cubecl-cpu is the ONLY honest f64 backend (WGSL has no f64), and it
    // shares CubeCL's front end with the kernel under test — so this lane is
    // not front-end-independent. The trusted list records that explicitly
    // (HOST_HARDWARE_TRUST + the shared-front-end caveat) instead of implying
    // a GPU/independent execution lane; only the derived twin is independent.
    frontend_independent: false,
}
