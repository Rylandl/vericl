//! The evidence manifest: every claim bound to the kernel identity it was
//! produced from. Evidence that no longer matches the current build is
//! rejected, not warned about.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::compare::CompareReport;
use crate::contract::{ContractRecord, Identity};

/// The evidence manifest — the serialized form of an `evidence/*.json` file.
///
/// One [`Entry`] per kernel, each binding its [`Claim`]s to the [`Identity`]
/// they were produced from. Load one from disk with [`Manifest::load`] and
/// check it against a freshly built manifest with [`verify`]; `vericl::suite!`
/// does exactly this on every `cargo test` run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    /// The `vericl` version that produced this manifest.
    pub vericl_version: String,
    /// One entry per kernel in the suite.
    pub entries: Vec<Entry>,
}

/// One kernel's evidence: its identity, contract, established claims, and the
/// components the entry trusts rather than checks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Entry {
    /// The kernel's name.
    pub kernel: String,
    /// The identity every claim below is bound to; a mismatch is stale evidence.
    pub identity: Identity,
    /// The contract (assumptions + comparison semantics) claims were produced under.
    pub contract: ContractRecord,
    /// What each check established, tagged by [`ClaimKind`].
    pub claims: Vec<Claim>,
    /// Components this evidence trusts rather than checks (README "Claims and
    /// trust boundaries").
    pub trusted: Vec<String>,
}

/// A single claim. `kind` states what the result establishes — these are
/// never interchangeable (see README "Claims and trust boundaries").
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Claim {
    /// What this result establishes (proved / tested / assumed).
    pub kind: ClaimKind,
    /// Which check produced this claim (e.g. "differential").
    pub check: String,
    /// Backend identity as reported at test time, for tested claims.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// Check configuration: seeds, sizes, case counts.
    pub config: serde_json::Value,
    /// The outcome of the check ([`ClaimResult`]).
    pub result: ClaimResult,
}

/// Which of the four claim categories a result falls into (README "Claims and
/// trust boundaries").
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ClaimKind {
    /// Property discharged by a checker (none yet in v0; reserved for the
    /// SMT bounds milestone).
    Proved,
    /// Behavior observed on specific inputs on a specific backend.
    Tested,
    /// Declared constraint the other claims depend on but do not establish.
    Assumed,
}

/// The outcome recorded for a [`Claim`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum ClaimResult {
    /// The check passed.
    Pass,
    /// The check failed, with a human-readable explanation.
    Fail {
        /// What diverged / why the check failed.
        detail: String,
    },
    /// The claim is a recorded assumption; nothing was executed.
    Declared,
}

/// One differential case outcome, folded into a claim's detail on failure.
///
/// `reports` carries one `(param name, CompareReport)` pair per compared
/// `&mut Array` parameter, in declaration order — a kernel with multiple mut
/// arrays (e.g. two output buffers) gets one report per array, so a mismatch
/// can be attributed to the specific parameter that diverged rather than
/// merged into a single anonymous report. Empty when `reference_panic` is
/// set (nothing was compared).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaseOutcome {
    /// A label for the case (e.g. `"n=256"`).
    pub case: String,
    /// One `(param name, report)` per compared `&mut Array` parameter.
    pub reports: Vec<(String, CompareReport)>,
    /// Set when the reference execution panicked (e.g. an out-of-bounds
    /// access the GPU backend would silently clamp).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_panic: Option<String>,
}

impl CaseOutcome {
    /// `true` iff the reference didn't panic and every compared parameter's
    /// report passed.
    pub fn pass(&self) -> bool {
        self.reference_panic.is_none() && self.reports.iter().all(|(_, r)| r.pass)
    }
}

/// Human-readable description of one case outcome, for print output and
/// claim failure detail. Shared by `conform.rs`'s demo-defects mode and the
/// `vericl::suite!`-generated conformance runner.
pub fn describe_case_outcome(o: &CaseOutcome) -> String {
    if let Some(p) = &o.reference_panic {
        // Only an "index out of bounds" panic is the WGSL-robustness story
        // (a GPU backend would silently clamp an out-of-bounds access that
        // panics sequentially). Any other panic (e.g. a `wrapping` kernel's
        // reference twin still dividing by zero) is a divergent-semantics/
        // defect finding of a different kind and must not be mislabeled as
        // a bounds issue.
        return if p.contains("index out of bounds") {
            format!(
                "{}: reference execution panicked ({p}) — the kernel accesses outside its \
                 declared bounds; GPU backends (WGSL robustness) would silently clamp this",
                o.case
            )
        } else {
            format!(
                "{}: reference execution panicked ({p}) — divergent semantics or defect; see \
                 message",
                o.case
            )
        };
    }
    let failing: Vec<String> = o
        .reports
        .iter()
        .filter(|(_, r)| !r.pass)
        .map(|(param, r)| {
            let worst = r
                .worst
                .as_ref()
                .map(|w| {
                    format!(
                        " worst at [{}]: expected {} got {}{}",
                        w.index,
                        w.expected,
                        w.actual,
                        w.ulp.map(|u| format!(" ({u} ulp)")).unwrap_or_default()
                    )
                })
                .unwrap_or_default();
            format!(
                "{} `{param}`: {}/{} elements diverge from reference{worst}",
                o.case, r.mismatches, r.checked
            )
        })
        .collect();
    if failing.is_empty() {
        format!("{}: pass", o.case)
    } else {
        failing.join("; ")
    }
}

/// `config` JSON for a differential (`Tested`) claim. Shared by `conform.rs`
/// and the `vericl::suite!`-generated runner so the field names/shape can
/// never drift between hand-written and generated code.
#[doc(hidden)] // generated-code plumbing (suite! claim config builder)
pub fn differential_config(sizes: &[usize], seed: u64, cube_dim: u32) -> serde_json::Value {
    serde_json::json!({
        "sizes": sizes,
        "seed": seed,
        "cube_dim": cube_dim,
        "reference": "vericl-macros sequential twin",
    })
}

/// `config` JSON for a *vectorized* differential (`Tested`) claim
/// (design-line-vector.md §9). Identical to [`differential_config`] but records
/// the pinned lane width `W`, so a re-run at a different width is a visibly
/// different claim, and the `sizes` are documented as **line** counts (each
/// line is `W` scalars — the buffer is `sizes[i] * W` scalars). The twin
/// operates on `Line<P, W>` lane arrays, front-end-independently of the GPU's
/// SIMD `Vector<P, W>`, so the reference wording is width-aware.
#[doc(hidden)] // generated-code plumbing (suite! claim config builder)
pub fn differential_vector_config(
    sizes: &[usize],
    seed: u64,
    cube_dim: u32,
    vector_width: u32,
) -> serde_json::Value {
    serde_json::json!({
        "sizes": sizes,
        "sizes_unit": "lines",
        "seed": seed,
        "cube_dim": cube_dim,
        "vector_width": vector_width,
        "reference": "vericl-macros sequential Line<P, W> lane-array twin",
    })
}

/// `config` JSON for a `Proved`/`smt-oob-freedom` claim.
#[doc(hidden)] // generated-code plumbing (suite! claim config builder)
pub fn proved_config(solver: &str, obligations: usize) -> serde_json::Value {
    serde_json::json!({
        "solver": solver,
        "logic": "QF_LIA",
        "obligations": obligations,
    })
}

/// The `check` string of the injected assumption a cooperative differential
/// claim depends on when race freedom is *not* proved (the honest-fallback
/// tier, docs/design-shared-memory.md §6). Distinct from the `smt-race-freedom`
/// proved-claim check the strong tier cites — the two must never be conflated.
#[doc(hidden)] // generated-code plumbing (cooperative claim wiring)
pub const RACE_FREEDOM_ASSUMPTION_CHECK: &str = "intra-phase-race-freedom";

/// The `check` string of the `Proved` race-freedom claim, duplicated here
/// (core cannot depend on `vericl-ir`, by design) so a cooperative differential
/// claim's `depends_on` can cite it. Kept byte-identical to
/// `vericl_ir::SMT_RACE_FREEDOM_CHECK` — the suite asserts both agree.
#[doc(hidden)] // generated-code plumbing (cooperative claim wiring)
pub const SMT_RACE_FREEDOM_CHECK: &str = "smt-race-freedom";

/// How a cooperative differential (`tested`) claim records its dependency on
/// intra-phase race freedom + barrier non-divergence (docs/design-shared-
/// memory.md §6). The phase-split twin picks one intra-segment thread order, so
/// it is a faithful reference *only* under race freedom; that dependency is
/// always made explicit, never assumed silently.
///
/// Generated-code plumbing: a parameter of [`cooperative_differential_config`],
/// set by the `suite!` runner — not an API user code constructs.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaceDependency {
    /// Strong tier: the `smt-race-freedom` proof is present and discharged the
    /// dependency. The tested claim cites that proved claim's `check`.
    Discharged,
    /// Honest-fallback tier: race freedom was not proved (prove disabled, or
    /// the proof came back `OutOfSubset`), so it travels as an explicit
    /// [`race_freedom_assumption_claim`] the tested claim depends on.
    Assumed,
}

/// `config` JSON for a cooperative kernel's differential (`tested`) claim,
/// carrying the race-freedom dependency coupling (docs/design-shared-memory.md
/// §6). `reference` describes the reference twin (derived phase-split, or an
/// author-supplied declared reference). `dependency` records whether the twin's
/// faithfulness is discharged by the `smt-race-freedom` proof or rests on the
/// injected assumption.
#[doc(hidden)] // generated-code plumbing (suite! claim config builder)
pub fn cooperative_differential_config(
    sizes: &[usize],
    seed: u64,
    cube_dim: u32,
    reference: &str,
    dependency: RaceDependency,
) -> serde_json::Value {
    let depends_on = match dependency {
        RaceDependency::Discharged => serde_json::json!({
            "property": "intra-phase race freedom + barrier non-divergence",
            "check": SMT_RACE_FREEDOM_CHECK,
            "status": "discharged-by-proof",
        }),
        RaceDependency::Assumed => serde_json::json!({
            "property": "intra-phase race freedom + barrier non-divergence",
            "check": RACE_FREEDOM_ASSUMPTION_CHECK,
            "status": "assumed-undischarged",
        }),
    };
    serde_json::json!({
        "sizes": sizes,
        "seed": seed,
        "cube_dim": cube_dim,
        "reference": reference,
        "depends_on": depends_on,
    })
}

/// The `Assumed` claim injected into a cooperative kernel's entry when race
/// freedom is not proved (the honest-fallback tier, §6). Travels exactly as a
/// `compare(abs=…)` tolerance does — a declared constraint the tested claim
/// leans on but does not itself establish. A cooperative differential result
/// with neither this assumption nor the `smt-race-freedom` proof is refused,
/// never recorded silently.
#[doc(hidden)] // generated-code plumbing (cooperative claim wiring)
pub fn race_freedom_assumption_claim() -> Claim {
    Claim {
        kind: ClaimKind::Assumed,
        check: RACE_FREEDOM_ASSUMPTION_CHECK.to_string(),
        backend: None,
        config: serde_json::json!({
            "statement": "intra-phase race freedom + barrier non-divergence (undischarged — the \
                          phase-split twin is a faithful reference only if every barrier-delimited \
                          segment is race-free; this was not proved for this kernel/run)",
        }),
        result: ClaimResult::Declared,
    }
}

/// `config` JSON for a cooperative kernel's `Proved`/`smt-oob-freedom` claim.
/// Unlike the ordinary bounds proof, a cooperative kernel's tree-reduction
/// bounds obligations are discharged by the two-thread cooperative walk (the
/// single-thread bounds walk defers a barrier-carrying loop) — recorded here so
/// the provenance is explicit.
#[doc(hidden)] // generated-code plumbing (suite! claim config builder)
pub fn proved_bounds_cooperative_config(solver: &str, obligations: usize) -> serde_json::Value {
    serde_json::json!({
        "solver": solver,
        "logic": "QF_LIA",
        "obligations": obligations,
        "discharged_by": "two-thread cooperative walk (a barrier-carrying tree loop is deferred \
                          by the single-thread bounds walk and discharged here)",
    })
}

/// `config` JSON for a `Proved`/`smt-race-freedom` claim (docs/design-shared-
/// memory.md §5.6): solver, QF_LIA, phase count, and the per-class obligation
/// counts (write-write / read-write / inter-cube single-writer / barrier
/// uniformity). `obligations` is the total of the three SMT-checked race
/// classes.
#[doc(hidden)] // generated-code plumbing (suite! claim config builder)
#[allow(clippy::too_many_arguments)]
pub fn proved_race_config(
    solver: &str,
    obligations: usize,
    phases: usize,
    write_write: usize,
    read_write: usize,
    intercube: usize,
    uniformity: usize,
) -> serde_json::Value {
    serde_json::json!({
        "solver": solver,
        "logic": "QF_LIA",
        "obligations": obligations,
        "phases": phases,
        "write_write": write_write,
        "read_write": read_write,
        "intercube_single_writer": intercube,
        "barrier_uniformity": uniformity,
    })
}

impl Manifest {
    /// A manifest over `entries`, stamped with the current `vericl` version.
    pub fn new(entries: Vec<Entry>) -> Self {
        Self {
            vericl_version: crate::VERSION.to_string(),
            entries,
        }
    }

    /// Write the manifest as pretty JSON to `path`, creating parent directories.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self).unwrap() + "\n")
    }

    /// Read a manifest from `path` (an `evidence/*.json` file).
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// Verify stored evidence against the current build's freshly produced
/// manifest. Returns human-readable problems; empty means the evidence stands.
pub fn verify(stored: &Manifest, current: &Manifest) -> Vec<String> {
    let mut problems = Vec::new();

    for cur in &current.entries {
        match stored.entries.iter().find(|e| e.kernel == cur.kernel) {
            None => problems.push(format!(
                "kernel `{}`: no stored evidence — run update",
                cur.kernel
            )),
            Some(st) => {
                if st.identity != cur.identity {
                    // Report every mismatched identity field, not just the
                    // first — a kernel edit typically changes both the
                    // source-level and IR-level hash together, and both
                    // must be visible in the failure, not just whichever
                    // field happens to differ.
                    let mut fields = Vec::new();
                    if st.identity.source_hash != cur.identity.source_hash {
                        fields.push(format!(
                            "source_hash {} -> {}",
                            st.identity.source_hash, cur.identity.source_hash
                        ));
                    }
                    if st.identity.ir_hash != cur.identity.ir_hash {
                        fields.push(format!(
                            "ir_hash {} -> {}",
                            st.identity.ir_hash.as_deref().unwrap_or("<none>"),
                            cur.identity.ir_hash.as_deref().unwrap_or("<none>"),
                        ));
                    }
                    if st.identity.vericl_version != cur.identity.vericl_version {
                        fields.push(format!(
                            "vericl_version {} -> {}",
                            st.identity.vericl_version, cur.identity.vericl_version
                        ));
                    }
                    problems.push(format!(
                        "kernel `{}`: STALE evidence — identity mismatch ({}) (kernel source, \
                         contract, IR, or vericl version changed without renewing evidence)",
                        cur.kernel,
                        fields.join(", ")
                    ));
                    // Identity mismatch invalidates everything else about the entry.
                    continue;
                }
                if st.contract != cur.contract {
                    problems.push(format!(
                        "kernel `{}`: contract record drifted without identity change (bug?)",
                        cur.kernel
                    ));
                }
                // Downgrade check: a `Proved` claim recorded in stored
                // evidence must still be produced by the current build.
                // Silently dropping one (e.g. `prove: false`, or z3 going
                // missing) is a downgrade, not a pass — proved and "never
                // claimed" are never presented as interchangeable (README
                // "Claims and trust boundaries").
                for st_claim in st.claims.iter().filter(|c| c.kind == ClaimKind::Proved) {
                    let still_present = cur
                        .claims
                        .iter()
                        .any(|c| c.kind == ClaimKind::Proved && c.check == st_claim.check);
                    if !still_present {
                        problems.push(format!(
                            "kernel `{}`: evidence downgraded — stored evidence has a proved \
                             `{}` claim that the current build did not produce (prove disabled, \
                             or z3 unavailable?)",
                            cur.kernel, st_claim.check
                        ));
                    }
                }
                for claim in st.claims.iter().chain(&cur.claims) {
                    if let ClaimResult::Fail { detail } = &claim.result {
                        problems.push(format!(
                            "kernel `{}`: {} `{}` claim failed: {}",
                            cur.kernel,
                            match claim.kind {
                                ClaimKind::Proved => "proved",
                                ClaimKind::Tested => "tested",
                                ClaimKind::Assumed => "assumed",
                            },
                            claim.check,
                            detail
                        ));
                    }
                }
            }
        }
    }

    for st in &stored.entries {
        if !current.entries.iter().any(|e| e.kernel == st.kernel) {
            problems.push(format!(
                "kernel `{}`: stored evidence for a kernel that no longer exists in this build",
                st.kernel
            ));
        }
    }

    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(kernel: &str, hash: &str) -> Entry {
        Entry {
            kernel: kernel.into(),
            identity: Identity {
                source_hash: hash.into(),
                vericl_version: crate::VERSION.into(),
                ir_hash: None,
            },
            contract: ContractRecord {
                assumes: vec![],
                compare: "exact".into(),
                wrapping: false,
                instantiate: vec![],
                uses: vec![],
            },
            claims: vec![],
            trusted: vec![],
        }
    }

    #[test]
    fn stale_identity_is_rejected() {
        let stored = Manifest::new(vec![entry("k", "aaa")]);
        let current = Manifest::new(vec![entry("k", "bbb")]);
        let problems = verify(&stored, &current);
        assert_eq!(problems.len(), 1);
        assert!(problems[0].contains("STALE"));
    }

    #[test]
    fn matching_evidence_passes() {
        let stored = Manifest::new(vec![entry("k", "aaa")]);
        let current = Manifest::new(vec![entry("k", "aaa")]);
        assert!(verify(&stored, &current).is_empty());
    }

    #[test]
    fn missing_and_orphaned_entries_flagged() {
        let stored = Manifest::new(vec![entry("gone", "x")]);
        let current = Manifest::new(vec![entry("new", "y")]);
        let problems = verify(&stored, &current);
        assert_eq!(problems.len(), 2);
    }

    fn proved_claim() -> Claim {
        Claim {
            kind: ClaimKind::Proved,
            check: "smt-oob-freedom".into(),
            backend: None,
            config: serde_json::json!({}),
            result: ClaimResult::Pass,
        }
    }

    /// A `Proved` claim on file that the current build no longer produces
    /// (e.g. `prove: false`, or z3 went missing) is a downgrade and must be
    /// caught, not silently accepted as "fewer claims, but nothing failed".
    #[test]
    fn dropped_proved_claim_is_a_downgrade() {
        let mut stored_entry = entry("k", "aaa");
        stored_entry.claims.push(proved_claim());
        let stored = Manifest::new(vec![stored_entry]);
        // Current build: same identity, but no proved claim at all.
        let current = Manifest::new(vec![entry("k", "aaa")]);
        let problems = verify(&stored, &current);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("downgraded"), "{problems:?}");
        assert!(problems[0].contains("smt-oob-freedom"), "{problems:?}");
    }

    /// The downgrade check keys on the claim's `check` string, so it covers
    /// the new `smt-race-freedom` proved claim (docs/design-shared-memory.md
    /// §5.6/§6) exactly like `smt-oob-freedom`: a stored race-freedom proof the
    /// current build no longer produces (prove disabled, z3 gone, or the
    /// cooperative walk regressed) is a downgrade, not a silent pass. This is
    /// the coupling's safety net — a cooperative tested claim cites this proof
    /// as its discharged dependency, so losing it must not go unnoticed.
    #[test]
    fn dropped_proved_race_freedom_claim_is_a_downgrade() {
        let race_claim = Claim {
            kind: ClaimKind::Proved,
            check: SMT_RACE_FREEDOM_CHECK.into(),
            backend: None,
            config: serde_json::json!({}),
            result: ClaimResult::Pass,
        };
        let mut stored_entry = entry("coop_k", "aaa");
        stored_entry.claims.push(proved_claim()); // smt-oob-freedom
        stored_entry.claims.push(race_claim);
        let stored = Manifest::new(vec![stored_entry]);
        // Current build keeps bounds but drops the race-freedom proof.
        let mut current_entry = entry("coop_k", "aaa");
        current_entry.claims.push(proved_claim());
        let current = Manifest::new(vec![current_entry]);
        let problems = verify(&stored, &current);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("downgraded"), "{problems:?}");
        assert!(problems[0].contains("smt-race-freedom"), "{problems:?}");
    }

    /// The same proved claim present on both sides is not a downgrade.
    #[test]
    fn retained_proved_claim_is_not_a_downgrade() {
        let mut stored_entry = entry("k", "aaa");
        stored_entry.claims.push(proved_claim());
        let mut current_entry = entry("k", "aaa");
        current_entry.claims.push(proved_claim());
        let stored = Manifest::new(vec![stored_entry]);
        let current = Manifest::new(vec![current_entry]);
        assert!(verify(&stored, &current).is_empty());
    }

    #[test]
    fn case_outcome_pass_and_describe() {
        let ok = CaseOutcome { case: "n=4".into(), reports: vec![], reference_panic: None };
        assert!(ok.pass());
        assert_eq!(describe_case_outcome(&ok), "n=4: pass");

        let bad_report = CompareReport { pass: false, checked: 4, mismatches: 1, max_ulp: None, worst: None };
        let bad = CaseOutcome {
            case: "n=4".into(),
            reports: vec![("y".to_string(), bad_report)],
            reference_panic: None,
        };
        assert!(!bad.pass());
        let msg = describe_case_outcome(&bad);
        assert!(msg.contains('y'), "{msg}");
        assert!(msg.contains("1/4"), "{msg}");
    }

    /// REGRESSION (adversarial soundness review, Bug 2 — cosmetic; moved
    /// here from `conform.rs` along with `describe_case_outcome` itself so
    /// the macro-generated `conformance_case` and `conform.rs`'s
    /// demo-defects mode can't drift). Only an "index out of bounds" panic
    /// gets the WGSL-robustness/bounds narrative.
    #[test]
    fn describe_outcome_labels_oob_panic_with_bounds_story() {
        let o = CaseOutcome {
            case: "n=4".into(),
            reports: vec![],
            reference_panic: Some("index out of bounds: the len is 4 but the index is 4".into()),
        };
        let msg = describe_case_outcome(&o);
        assert!(msg.contains("GPU backends (WGSL robustness) would silently clamp this"), "{msg}");
        assert!(msg.contains("index out of bounds"), "{msg}");
    }

    /// A non-bounds panic (division by zero, the motivating `wrapping`-
    /// clause case from the review) must NOT get the bounds/WGSL-robustness
    /// narrative — that would misattribute the cause.
    #[test]
    fn describe_outcome_labels_non_oob_panic_neutrally() {
        let o = CaseOutcome {
            case: "n=4".into(),
            reports: vec![],
            reference_panic: Some("attempt to divide by zero".into()),
        };
        let msg = describe_case_outcome(&o);
        assert!(!msg.contains("WGSL robustness"), "{msg}");
        assert!(!msg.contains("accesses outside its declared"), "{msg}");
        assert!(msg.contains("divergent semantics or defect"), "{msg}");
        assert!(msg.contains("attempt to divide by zero"), "{msg}");
    }
}
