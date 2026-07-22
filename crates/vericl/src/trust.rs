//! Shared trusted-component wording for evidence entries (README "Claims and
//! trust boundaries"). Centralized here so `conform.rs`'s demo-defects mode
//! and the `vericl::suite!`-generated conformance runner never hand-maintain
//! two copies of the same strings.

/// Trust entries every differential entry carries, independent of backend or
/// runtime.
pub fn reference_twin_trust() -> Vec<String> {
    vec![
        "rustc codegen of the reference twin".to_string(),
        "vericl-macros source-to-reference derivation".to_string(),
    ]
}

/// Buffer upload/readback integrity for a specific backend, e.g.
/// `"wgpu<wgsl> buffer upload/readback integrity"`.
pub fn backend_buffer_trust(backend: &str) -> String {
    format!("{backend} buffer upload/readback integrity")
}

/// The GPU hardware itself is always trusted, never verified.
pub const GPU_HARDWARE_TRUST: &str = "GPU hardware";

/// Trust entries added when a `Proved`/`smt-oob-freedom` claim is folded into
/// an entry.
pub fn proved_bounds_trust(solver: &str) -> Vec<String> {
    vec![
        format!("the solver binary ({solver}) discharging the SMT bounds obligations"),
        "vericl-ir's obligation encoding (0 <= index < Length(array) in QF_LIA over the \
         CubeCL IR)"
            .to_string(),
        "cubecl front-end expansion (the proof is about the IR; codegen below the IR remains \
         covered only by the tested differential claims)"
            .to_string(),
    ]
}

/// Trust wording for an additional differential lane that shares CubeCL's
/// front end with the kernel under test (e.g. the `cpu` runtime lane) — not
/// an independent reference, unlike the macro-derived sequential twin.
pub fn shared_frontend_lane_trust(backend: &str) -> String {
    format!(
        "{backend} runtime shares CubeCL's front end (macro expansion + IR) with the kernel \
         under test — this lane is NOT an independent reference; only the vericl-macros \
         sequential twin is independent of CubeCL"
    )
}
