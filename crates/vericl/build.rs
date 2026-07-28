//! Captures the *compiling* toolchain into two constants the evidence
//! manifest's provenance record reads back (`vericl::provenance`).
//!
//! Why a build script rather than shelling out to `rustc` from the test: a
//! build script is invoked by cargo with `RUSTC` set to the compiler that is
//! actually building this crate, so the recorded version is the one that
//! produced the reference twin — not whatever happens to be first on `PATH`
//! when the test runs. Cargo's build-script fingerprint includes the compiler
//! version, so a toolchain change re-runs this and moves the constant.
//!
//! `TARGET` is cargo's own target triple for this compilation, which is the
//! honest answer for "what was this evidence produced for" on a cross build.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    // `rustc -vV`'s first line is `rustc <semver> (<hash> <date>)` — the whole
    // line, not just the semver, because a nightly's hash/date is exactly the
    // part that distinguishes two builds claiming the same version number.
    let version = Command::new(&rustc)
        .arg("-vV")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().map(|l| l.trim().to_string()))
        .filter(|l| !l.is_empty())
        .unwrap_or_else(|| format!("<unknown: `{rustc} -vV` produced no version line>"));
    println!("cargo:rustc-env=VERICL_RUSTC_VERSION={version}");

    let target = std::env::var("TARGET").unwrap_or_else(|_| "<unknown>".to_string());
    println!("cargo:rustc-env=VERICL_TARGET={target}");
}
