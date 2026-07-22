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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaseOutcome {
    pub case: String,
    pub report: Option<CompareReport>,
    /// Set when the reference execution panicked (e.g. an out-of-bounds
    /// access the GPU backend would silently clamp).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_panic: Option<String>,
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
}
