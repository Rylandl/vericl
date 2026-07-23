//! The evidence manifest: every claim bound to the kernel identity it was
//! produced from. Evidence that no longer matches the current build is
//! rejected, not warned about.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::compare::CompareReport;
use crate::contract::{ContractRecord, Identity};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub vericl_version: String,
    pub entries: Vec<Entry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Entry {
    pub kernel: String,
    pub identity: Identity,
    pub contract: ContractRecord,
    pub claims: Vec<Claim>,
    /// Components this evidence trusts rather than checks.
    pub trusted: Vec<String>,
}

/// A single claim. `kind` states what the result establishes — these are
/// never interchangeable (see README "Claims and trust boundaries").
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Claim {
    pub kind: ClaimKind,
    /// Which check produced this claim (e.g. "differential").
    pub check: String,
    /// Backend identity as reported at test time, for tested claims.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// Check configuration: seeds, sizes, case counts.
    pub config: serde_json::Value,
    pub result: ClaimResult,
}

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum ClaimResult {
    Pass,
    Fail { detail: String },
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
    pub case: String,
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
pub fn differential_config(sizes: &[usize], seed: u64, cube_dim: u32) -> serde_json::Value {
    serde_json::json!({
        "sizes": sizes,
        "seed": seed,
        "cube_dim": cube_dim,
        "reference": "vericl-macros sequential twin",
    })
}

/// `config` JSON for a `Proved`/`smt-oob-freedom` claim.
pub fn proved_config(solver: &str, obligations: usize) -> serde_json::Value {
    serde_json::json!({
        "solver": solver,
        "logic": "QF_LIA",
        "obligations": obligations,
    })
}

impl Manifest {
    pub fn new(entries: Vec<Entry>) -> Self {
        Self {
            vericl_version: crate::VERSION.to_string(),
            entries,
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self).unwrap() + "\n")
    }

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
