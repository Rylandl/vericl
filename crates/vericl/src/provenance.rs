//! The **verification-environment fingerprint** recorded in every evidence
//! manifest.
//!
//! # Why evidence needs one
//!
//! [`Identity`](crate::Identity) binds a claim to the *kernel* it was produced
//! from — its source tokens, its contract, its expanded IR. It says nothing
//! about the machine that produced it. But a differential claim is a statement
//! about what a specific compiler emitted, running against a specific backend,
//! and a proved claim is a statement about what a specific solver binary
//! discharged. Move any of those and the sentence the evidence file appears to
//! be making is no longer the sentence that was measured.
//!
//! Concretely, all of these change what the evidence means while leaving every
//! kernel identity bit-identical:
//!
//! * a different **rustc** — the reference twin is ordinary Rust that rustc
//!   compiles; the twin is one of the two legs the differential compares;
//! * a different **cubecl** — the whole pipeline under test, pinned (`=0.10.0`)
//!   precisely because it is not a detail;
//! * a different **z3** — the binary that returned `unsat` for every bounds
//!   obligation;
//! * a different **target triple** or **device** — `"wgpu<wgsl>"` on Metal and
//!   on Vulkan are two different code generators behind one name.
//!
//! So an evidence file carried to a different toolchain is **stale**, in the
//! same class as a kernel edit, and [`verify`](crate::verify) says so rather
//! than accepting it silently.
//!
//! # What it does *not* cover
//!
//! Stated plainly, in the spirit of the round-10 caveat precedent (a recorded
//! boundary is worth more than an implied one):
//!
//! * **Build-script and environment residuals.** [`RUSTC_VERSION`] is captured
//!   by this crate's own `build.rs` from cargo's `RUSTC`. It does not cover
//!   `RUSTFLAGS`, `-C target-cpu`, cargo profile/opt-level, `cfg` features of
//!   *other* crates, or a `[patch]`ed dependency that keeps its version
//!   string. Two builds with identical fingerprints can therefore still differ
//!   — the fingerprint refuses the differences it can see, and claims nothing
//!   about the rest.
//! * **Transitive dependency versions.** Only the four versions that decide
//!   meaning are recorded (`vericl`, `vericl-ir`, `vericl-macros`, `cubecl`).
//!   `serde`, `wgpu`'s own patch version, and the rest are not.
//! * **Driver / OS version.** [`Provenance::device`] records what the runtime
//!   exposes (for wgpu, the selected backend — Metal / Vulkan / DX12), not the
//!   GPU driver build. A driver update is invisible here.
//! * **The solver's own soundness.** `z3 <version>` is recorded, and z3 is a
//!   trusted component either way (see `trust::proved_bounds_trust`).
//!
//! # Where the z3 version lives
//!
//! It is recorded in **two** places, deliberately, and they are not redundant:
//! [`Provenance::z3`] is the environment fact ("this file was produced with
//! this solver"), while a proved claim's `config.solver` is the claim-scoped
//! fact ("*this obligation set* was discharged by it"). A manifest may contain
//! claims produced with no solver at all (`prove: false`), which is why the
//! environment field is an `Option` and is `None` when nothing was proved:
//! recording a solver that discharged nothing would manufacture staleness out
//! of an unused binary on `PATH`.

use serde::{Deserialize, Serialize};

/// The `rustc` that compiled this crate, as `rustc -vV`'s first line (e.g.
/// `"rustc 1.89.0 (abcdef012 2025-06-01)"`). Captured by `build.rs`; see the
/// module docs for what that does and does not guarantee.
pub const RUSTC_VERSION: &str = env!("VERICL_RUSTC_VERSION");

/// Cargo's target triple for this compilation (e.g. `"aarch64-apple-darwin"`).
/// Captured by `build.rs` from cargo's `TARGET`.
pub const TARGET: &str = env!("VERICL_TARGET");

/// The **pinned** CubeCL requirement from the workspace manifest, verbatim.
///
/// `vericl` core is deliberately cubecl-free (README "Locked decisions"), so it
/// cannot read cubecl's `CARGO_PKG_VERSION`. The pin is an exact-version
/// requirement (`=0.10.0`) that the workspace owns and upgrades deliberately,
/// which makes the string itself the right thing to record — and
/// `cubecl_pin_matches_the_workspace_manifest` below fails the test suite if
/// this constant and `Cargo.toml` ever disagree, so it cannot drift silently.
pub const CUBECL_PINNED: &str = "=0.10.0";

/// The verification environment an evidence manifest was produced in.
///
/// Schema-additive: every field is `#[serde(default)]`, so a manifest written
/// before this record existed still *loads* (as [`Provenance::default`], which
/// [`Provenance::is_recorded`] reports as absent) instead of hard-failing
/// deserialization for a programmatic consumer. It does not still *verify* —
/// see [`verify`](crate::verify).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// [`RUSTC_VERSION`] as recorded when the manifest was written.
    #[serde(default)]
    pub rustc: String,
    /// [`TARGET`] as recorded when the manifest was written.
    #[serde(default)]
    pub target: String,
    /// The `vericl` crate version.
    #[serde(default)]
    pub vericl: String,
    /// The `vericl-ir` crate version (filled in by the `suite!`-generated
    /// runner, which is the only place that can see it — core does not depend
    /// on `vericl-ir`). Empty for a manifest built by [`Manifest::new`].
    ///
    /// [`Manifest::new`]: crate::Manifest::new
    #[serde(default)]
    pub vericl_ir: String,
    /// The `vericl-macros` crate version, as the `suite!` macro's own
    /// `CARGO_PKG_VERSION` at the time it expanded. A proc-macro crate cannot
    /// export a constant, so the macro emits the literal instead. Empty for a
    /// manifest built by [`Manifest::new`].
    ///
    /// [`Manifest::new`]: crate::Manifest::new
    #[serde(default)]
    pub vericl_macros: String,
    /// [`CUBECL_PINNED`] as recorded when the manifest was written.
    #[serde(default)]
    pub cubecl: String,
    /// The solver that discharged this manifest's proved claims, e.g.
    /// `"z3 4.16.0"`. `None` when nothing was proved (`prove: false`) — see
    /// the module docs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z3: Option<String>,
    /// The execution lanes that ran, primary first, each as the backend
    /// identity the runtime reported (the same string a claim's `backend`
    /// field carries). One entry for an ordinary suite; two when a
    /// `suite!`'s `extra_lane` is `cfg`-enabled.
    ///
    /// Order is meaningful (primary first) and preserved.
    #[serde(default)]
    pub lanes: Vec<String>,
    /// Device identity for the primary lane, where the runtime exposes one
    /// cheaply: CubeCL's `ComputeClient::info()`, which for wgpu is the
    /// selected graphics backend (`Metal`, `Vulkan`, `Dx12`) and for
    /// cubecl-cpu is `()`. `None` when the runner did not record one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

impl Provenance {
    /// Everything `vericl` core can determine on its own: the compiling
    /// toolchain, the target, its own version, and the CubeCL pin. The
    /// `suite!`-generated runner fills in [`vericl_ir`](Self::vericl_ir),
    /// [`vericl_macros`](Self::vericl_macros), [`z3`](Self::z3),
    /// [`lanes`](Self::lanes), and [`device`](Self::device) — facts only the
    /// call site can see.
    pub fn current() -> Self {
        Self {
            rustc: RUSTC_VERSION.to_string(),
            target: TARGET.to_string(),
            vericl: crate::VERSION.to_string(),
            vericl_ir: String::new(),
            vericl_macros: String::new(),
            cubecl: CUBECL_PINNED.to_string(),
            z3: None,
            lanes: Vec::new(),
            device: None,
        }
    }

    /// Whether this manifest carries a provenance record at all. `false` for
    /// evidence written before the record existed (which deserializes to
    /// [`Provenance::default`], every string empty).
    pub fn is_recorded(&self) -> bool {
        !self.rustc.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`CUBECL_PINNED`] is hand-written (core cannot read cubecl's own
    /// version), so it is only trustworthy if it cannot drift from the
    /// workspace pin it claims to mirror. This reads the workspace manifest and
    /// fails the moment the two disagree — the whole reason a hand-written
    /// constant is acceptable here.
    #[test]
    fn cubecl_pin_matches_the_workspace_manifest() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("Cargo.toml");
        let text = std::fs::read_to_string(&root)
            .unwrap_or_else(|e| panic!("workspace manifest at {}: {e}", root.display()));
        let line = text
            .lines()
            .map(str::trim)
            .find(|l| l.starts_with("cubecl = "))
            .unwrap_or_else(|| panic!("no `cubecl = ` line in {}", root.display()));
        assert!(
            line.contains(&format!("version = \"{CUBECL_PINNED}\"")),
            "vericl::provenance::CUBECL_PINNED is `{CUBECL_PINNED}` but the workspace pins \
             `{line}` — update the constant in the same change as the pin, or evidence records a \
             cubecl version it was not built against"
        );
    }

    /// The build script must have produced something usable — an empty or
    /// obviously-unfilled constant would make the fingerprint match everywhere
    /// and quietly verify nothing.
    #[test]
    fn build_script_captured_a_real_toolchain() {
        assert!(RUSTC_VERSION.starts_with("rustc "), "rustc version: {RUSTC_VERSION:?}");
        assert!(!TARGET.is_empty() && TARGET != "<unknown>", "target: {TARGET:?}");
    }

    /// The absent-record sentinel: a legacy manifest deserializes to the
    /// default, which must not be mistaken for a real fingerprint.
    #[test]
    fn default_provenance_is_not_recorded_but_current_is() {
        assert!(!Provenance::default().is_recorded());
        assert!(Provenance::current().is_recorded());
    }
}
