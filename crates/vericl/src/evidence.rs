//! The evidence manifest: every claim bound to the kernel identity it was
//! produced from. Evidence that no longer matches the current build is
//! rejected, not warned about.

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::compare::CompareReport;
use crate::contract::{ContractRecord, Identity};
use crate::provenance::Provenance;

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
    /// The verification environment this manifest was produced in — toolchain,
    /// crate versions, solver, execution lanes. See [`Provenance`].
    ///
    /// `#[serde(default)]` so an evidence file written before the fingerprint
    /// existed still loads for a programmatic consumer; [`verify`] refuses it
    /// rather than accepting a file whose toolchain is unknown.
    #[serde(default)]
    pub provenance: Provenance,
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
    /// `true` iff the reference didn't panic, **something was actually
    /// compared**, and every compared parameter's report passed.
    ///
    /// # The non-vacuity clauses
    ///
    /// `reports.iter().all(…)` alone is `true` over an empty list, and
    /// `CompareReport { pass: true, checked: 0 }` is what a zero-length buffer
    /// produces — so the plain "nothing diverged" reading of this method has
    /// two ways to be green while comparing nothing:
    ///
    /// * **no compared parameter at all** — a kernel declaring no `&mut Array`
    ///   output. `#[vericl::kernel]` now rejects that shape at compile time, so
    ///   this clause is a backstop rather than the primary gate;
    /// * **a zero-element comparison** — reachable from a size that evaluates
    ///   to `0` (`sizes: [0]`, or a `gen(len(y = 0))` pin). Both literal
    ///   spellings are rejected by the macros; a size behind a `const` or an
    ///   arbitrary expression is not, and lands here.
    ///
    /// A case that compared nothing is not a passing case, so both are `false`
    /// and [`describe_case_outcome`] says which one happened. This is the same
    /// discipline the suite applies to `kernels: []` and `sizes: []`: an empty
    /// set is refused, never reported as agreement.
    pub fn pass(&self) -> bool {
        self.reference_panic.is_none()
            && !self.reports.is_empty()
            && self.reports.iter().all(|(_, r)| r.pass && r.checked > 0)
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
    // The two vacuity shapes [`CaseOutcome::pass`] refuses. Named explicitly:
    // "0/0 elements diverge" would otherwise read as agreement.
    if o.reports.is_empty() {
        return format!(
            "{}: NOTHING WAS COMPARED — the case produced no compared parameter at all, so \
             agreement here is vacuous",
            o.case
        );
    }
    if let Some((param, _)) = o.reports.iter().find(|(_, r)| r.checked == 0) {
        return format!(
            "{}: NOTHING WAS COMPARED — parameter `{param}` has zero elements, so agreement here \
             is vacuous (a size or a `gen(len(...))` pin evaluated to 0?)",
            o.case
        );
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
        // The launch shape this evidence was produced under (§10.4 correction
        // 2). A 1-D suite dispatches `CubeCount::Static(n, 1, 1)`; recording the
        // rank is what lets a reader tell this claim apart from a
        // `differential_dispatch_config` one instead of inferring it from the
        // absence of a field.
        "rank": 1,
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

/// `config` JSON for a *multi-axis* (2-D/3-D) differential (`Tested`) claim
/// (docs/design-2d-dispatch.md §4.8, §10.4 correction 2). The
/// [`differential_vector_config`] precedent, for the same reason it exists:
/// this claim's `sizes` are **extents** tuples, not thread counts, and saying
/// so is what keeps two units from being read as one.
///
/// It also closes the recordable half of the D1 hole (§3.3): the 1-D
/// [`differential_config`] records a *scalar* `cube_dim` and no cube count at
/// all, so evidence could not distinguish the launch shape it was produced
/// under. Here the full pinned `cube_dim` triple and the `rank` are recorded,
/// and `differential_config` gains `"rank": 1` so old and new evidence are
/// comparable.
#[doc(hidden)] // generated-code plumbing (suite! claim config builder)
pub fn differential_dispatch_config(
    sizes: &[[usize; 3]],
    seed: u64,
    cube_dim: [u32; 3],
    rank: u8,
) -> serde_json::Value {
    // Report each case at the clause's own arity — a rank-2 suite's sizes are
    // (w, h) pairs, and padding them to triples in the record would invent a
    // third extent the contract never mentions.
    let sizes: Vec<Vec<usize>> =
        sizes.iter().map(|e| e[..rank as usize].to_vec()).collect();
    serde_json::json!({
        "sizes": sizes,
        "sizes_unit": "extents",
        "seed": seed,
        "cube_dim": cube_dim[..rank as usize].to_vec(),
        "rank": rank,
        "reference": "vericl-macros sequential multi-axis grid twin",
    })
}

/// `config` JSON for a `Proved`/`smt-oob-freedom` claim.
///
/// `logic` is the logic actually in force for this kernel, not a constant
/// (§10.4 correction 3): a `LenEqProduct` assume puts a genuinely nonlinear
/// `len = x*y` in the global assertion context, so `QF_LIA` would be wrong
/// there. (`checked_mul` already emitted variable×variable products under a
/// `push`/`pop`, so the hardcoded label was a slight over-claim before this
/// milestone too — this makes it honest rather than introducing the problem.)
#[doc(hidden)] // generated-code plumbing (suite! claim config builder)
pub fn proved_config_with_logic(
    solver: &str,
    obligations: usize,
    logic: &str,
) -> serde_json::Value {
    serde_json::json!({
        "solver": solver,
        "logic": logic,
        "obligations": obligations,
    })
}

/// `config` JSON for a `Proved`/`smt-oob-freedom` claim in the linear-integer
/// case — [`proved_config_with_logic`] at `QF_LIA`.
#[doc(hidden)] // generated-code plumbing (suite! claim config builder)
pub fn proved_config(solver: &str, obligations: usize) -> serde_json::Value {
    proved_config_with_logic(solver, obligations, "QF_LIA")
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
    /// A manifest over `entries`, stamped with the current `vericl` version and
    /// the part of the verification environment this crate can see on its own
    /// ([`Provenance::current`]). Use [`Manifest::with_provenance`] to supply
    /// the rest (solver, lanes, device, sibling crate versions) — the
    /// `suite!`-generated runner does.
    pub fn new(entries: Vec<Entry>) -> Self {
        Self {
            vericl_version: crate::VERSION.to_string(),
            provenance: Provenance::current(),
            entries,
        }
    }

    /// A manifest over `entries` with a fully-populated verification-environment
    /// record.
    ///
    /// Generated-code plumbing: the `suite!` runner is the only place that can
    /// see the solver version, the execution lanes, the device, and the
    /// `vericl-ir` / `vericl-macros` versions.
    #[doc(hidden)]
    pub fn with_provenance(entries: Vec<Entry>, provenance: Provenance) -> Self {
        Self {
            vericl_version: crate::VERSION.to_string(),
            provenance,
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

// ---------------------------------------------------------------------------
// Verification
//
// The completeness property, stated once, because everything below is an
// instance of it:
//
//   **The stored file must not claim anything this build does not produce.**
//
// Every tamper class is an instance of that sentence being violated — a tested
// claim deleted from the build while the file still advertises it, a backend
// swapped, a `sizes` list shortened, a trust dependency erased, a passing claim
// typed into the file by hand. Each is a *problem*: `verify` refuses.
//
// The converse — this build producing MORE than the file records — is also a
// mismatch (the file records a claim SET, and a different set is a different
// statement), with **one** exemption, scoped by the provenance record rather
// than by shape: a claim or trust entry contributed by an execution LANE that
// the stored provenance says did not run. That is what lets a single manifest
// serve both `cargo test` and `cargo test --features cpu`, where the second
// `cfg`-enables a whole extra execution lane and adds one claim plus one trust
// entry per kernel. Those additions are reported by [`unrecorded_evidence`] as
// printed notes — never silent, and never able to mask a loss, a mutation, or a
// failure on the lanes the file does record.
//
// `trusted` is where the asymmetry bites hardest and is worth stating on its
// own: a component this build trusts that the file OMITS makes the recorded
// claim look STRONGER than the one established (fewer things taken on faith),
// so the missing direction is a problem there too.
// ---------------------------------------------------------------------------

/// Both directions of a stored-vs-current manifest comparison.
struct Comparison {
    /// The stored file claims something this build does not support.
    problems: Vec<String>,
    /// This build produces evidence the stored file does not record.
    unrecorded: Vec<String>,
}

fn kind_word(k: ClaimKind) -> &'static str {
    match k {
        ClaimKind::Proved => "proved",
        ClaimKind::Tested => "tested",
        ClaimKind::Assumed => "assumed",
    }
}

fn result_word(r: &ClaimResult) -> &'static str {
    match r {
        ClaimResult::Pass => "pass",
        ClaimResult::Fail { .. } => "fail",
        ClaimResult::Declared => "declared",
    }
}

fn claim_label(c: &Claim) -> String {
    match &c.backend {
        Some(b) => format!("{} `{}` (backend {b})", kind_word(c.kind), c.check),
        None => format!("{} `{}`", kind_word(c.kind), c.check),
    }
}

/// Render a pair of JSON values for one diff line, each truncated so a long
/// `statement` string cannot bury the rest of the report.
///
/// Truncating the two sides *independently from the start* is what a first
/// attempt does, and it has a defect worth avoiding: two values that agree on
/// their first `MAX` characters and differ after render **identically**, so the
/// line reads `stored X -> current X` for a difference that is really there.
/// The values that hit this are exactly the ones where it matters most — the
/// whole-array blob an unequal-length `sizes` diff produces.
///
/// So the window is centred on the **first divergence** instead: if the common
/// prefix is longer than the budget, both sides are shown from a little before
/// the point where they part company, elided at the front with `…`.
fn render_json_pair(stored: &serde_json::Value, current: &serde_json::Value) -> (String, String) {
    const MAX: usize = 160;
    /// Characters of agreeing context to keep before the divergence.
    const LEAD: usize = 24;

    let a: Vec<char> = stored.to_string().chars().collect();
    let b: Vec<char> = current.to_string().chars().collect();
    if a.len() <= MAX && b.len() <= MAX {
        return (a.into_iter().collect(), b.into_iter().collect());
    }
    let common = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    let start = if common + 1 > MAX { common.saturating_sub(LEAD) } else { 0 };
    let window = |v: &[char]| {
        let head = if start > 0 { "…" } else { "" };
        let tail = if v.len() > start + MAX { "…" } else { "" };
        let body: String = v.iter().skip(start).take(MAX).collect();
        format!("{head}{body}{tail}")
    };
    (window(&a), window(&b))
}

/// Structural diff of two claim `config` values as `path: stored X -> current Y`
/// lines.
///
/// **Normalization.** Object keys are compared as a *set* (the union, sorted) —
/// `serde_json`'s `Map` is a `BTreeMap` in this build, so both sides are already
/// canonically key-ordered on load and key order carries no information.
/// **Arrays are order-sensitive**: a config array is a declared sequence
/// (`sizes`, `cube_dim`), and reordering `sizes:` is a different declaration
/// even though the same cases run. Sensitivity is the safe direction here — it
/// can only ask for a regeneration after a cosmetic edit, never let a real
/// change through (the same argument `combine_source_hash` already records for
/// `uses(...)` order).
fn json_diff(path: &str, stored: &serde_json::Value, current: &serde_json::Value, out: &mut Vec<String>) {
    match (stored, current) {
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
            let mut keys: Vec<&str> = a.keys().chain(b.keys()).map(String::as_str).collect();
            keys.sort_unstable();
            keys.dedup();
            for k in keys {
                let p = if path.is_empty() { k.to_string() } else { format!("{path}.{k}") };
                match (a.get(k), b.get(k)) {
                    (Some(x), Some(y)) => json_diff(&p, x, y, out),
                    (Some(x), None) => {
                        let (r, _) = render_json_pair(x, &serde_json::Value::Null);
                        out.push(format!("{p}: stored {r} -> current <absent>"))
                    }
                    (None, Some(y)) => {
                        let (_, r) = render_json_pair(&serde_json::Value::Null, y);
                        out.push(format!("{p}: stored <absent> -> current {r}"))
                    }
                    (None, None) => unreachable!("key came from one of the two maps"),
                }
            }
        }
        (serde_json::Value::Array(a), serde_json::Value::Array(b)) if a.len() == b.len() => {
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                json_diff(&format!("{path}[{i}]"), x, y, out);
            }
        }
        _ => {
            if stored != current {
                let (a, b) = render_json_pair(stored, current);
                out.push(format!("{path}: stored {a} -> current {b}"));
            }
        }
    }
}

/// Match a stored entry's claims against the current build's.
///
/// **Normalization: claim order is meaningless and is ignored.** The order
/// claims appear in is an artifact of the pipeline (tested pushed first, then
/// proved; the cooperative branch *inserts* its tested claim at index 0; an
/// extra lane appends), not a property of the kernel. Claims are matched as a
/// multiset in three passes, narrowing:
///
/// 1. **whole claim equal** — identical `(kind, check, backend, config,
///    result)`. This pass exists for attribution, not detection: when a group
///    shares a key, first-fit on the key alone can pair a stored claim with a
///    *different* current claim of the same key and then blame the wrong one
///    for the config diff. Matching identical content first leaves only the
///    genuinely-changed claims for the later passes;
/// 2. key `(kind, check, backend)` — the common case, and the one that keeps
///    two same-`check` claims from two execution lanes apart;
/// 3. `(kind, check)` for whatever is left — so a *changed* backend is reported
///    as one field diff rather than as an unrelated removal plus addition.
///
/// Whether a difference is reported at all does not depend on the passes:
/// every stored claim ends up paired (and any field diff on the pair is
/// reported) or in `stored_only`, and every unpaired current claim is in
/// `current_only`. The passes decide *which* claim a diff is attributed to.
///
/// Returns `(pairs, stored_only, current_only)`, `pairs` sorted by stored index
/// so the report order is deterministic.
fn pair_claims(stored: &[Claim], current: &[Claim]) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let mut taken = vec![false; current.len()];
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut unpaired: Vec<usize> = (0..stored.len()).collect();

    // Each pass keeps whatever it could not match for the next, looser one.
    let matchers: [fn(&Claim, &Claim) -> bool; 3] = [
        |s, c| s == c,
        |s, c| s.kind == c.kind && s.check == c.check && s.backend == c.backend,
        |s, c| s.kind == c.kind && s.check == c.check,
    ];
    for matches in matchers {
        let mut still = Vec::new();
        for si in unpaired {
            let s = &stored[si];
            match current.iter().enumerate().find(|(ci, c)| !taken[*ci] && matches(s, c)) {
                Some((ci, _)) => {
                    taken[ci] = true;
                    pairs.push((si, ci));
                }
                None => still.push(si),
            }
        }
        unpaired = still;
    }

    pairs.sort_unstable();
    let current_only = (0..current.len()).filter(|i| !taken[*i]).collect();
    (pairs, unpaired, current_only)
}

/// Field-level diff of the contract record.
///
/// **Normalization: order-SENSITIVE on all three lists.** `assumes`,
/// `instantiate` and `uses` are authored clause lists whose order is already
/// covered by `SOURCE_HASH` (and, for `uses`, by `combine_source_hash`), so a
/// reorder is already an identity change — being sensitive here keeps the two
/// checks from disagreeing about the same fact.
fn contract_problems(kernel: &str, st: &ContractRecord, cur: &ContractRecord, out: &mut Vec<String>) {
    let mut note = |field: &str, a: String, b: String| {
        if a != b {
            out.push(format!(
                "kernel `{kernel}`: contract field `{field}` drifted without an identity change: \
                 stored {a} -> current {b} — `source_hash` covers the contract tokens, so this \
                 should be unreachable; treat it as a vericl bug rather than as evidence to renew"
            ));
        }
    };
    note("assumes", format!("{:?}", st.assumes), format!("{:?}", cur.assumes));
    note("compare", format!("{:?}", st.compare), format!("{:?}", cur.compare));
    note("wrapping", st.wrapping.to_string(), cur.wrapping.to_string());
    note("instantiate", format!("{:?}", st.instantiate), format!("{:?}", cur.instantiate));
    note("uses", format!("{:?}", st.uses), format!("{:?}", cur.uses));
}

/// Compare the verification-environment fingerprints, returning the execution
/// lanes present in this run that the stored evidence does not record (the
/// additive-lane exemption set for the `trusted` check).
fn provenance_problems(
    stored: &Provenance,
    current: &Provenance,
    problems: &mut Vec<String>,
    unrecorded: &mut Vec<String>,
) -> Vec<String> {
    if !stored.is_recorded() {
        problems.push(
            "STALE evidence — the stored manifest carries no verification-environment record (it \
             predates the provenance fingerprint). Evidence whose toolchain is unknown is not \
             merely older, it is unverifiable: regenerate it with `VERICL_UPDATE=1 cargo test`"
                .to_string(),
        );
        return Vec::new();
    }

    let mut fields: Vec<String> = Vec::new();
    let mut cmp = |name: &str, a: &str, b: &str| {
        if a != b {
            fields.push(format!("{name} `{a}` -> `{b}`"));
        }
    };
    cmp("rustc", &stored.rustc, &current.rustc);
    cmp("target", &stored.target, &current.target);
    cmp("vericl", &stored.vericl, &current.vericl);
    cmp("vericl-ir", &stored.vericl_ir, &current.vericl_ir);
    cmp("vericl-macros", &stored.vericl_macros, &current.vericl_macros);
    cmp("cubecl", &stored.cubecl, &current.cubecl);
    cmp(
        "z3",
        stored.z3.as_deref().unwrap_or("<none>"),
        current.z3.as_deref().unwrap_or("<none>"),
    );
    cmp(
        "device",
        stored.device.as_deref().unwrap_or("<none>"),
        current.device.as_deref().unwrap_or("<none>"),
    );
    if !fields.is_empty() {
        problems.push(format!(
            "STALE evidence — the verification environment changed ({}). This evidence was \
             produced by a different toolchain/solver/device than the one running now, so what it \
             measured is not what this build measures. Regenerate it here (`VERICL_UPDATE=1 cargo \
             test`), or check out the environment that produced it",
            fields.join("; ")
        ));
    }

    // Lanes are SUBSET-checked rather than compared: a lane the stored evidence
    // recorded that is missing now is a loss (and every claim it produced is
    // separately reported as stored-only), while a lane this run adds is the
    // additive `extra_lane` case — strictly more evidence, so a note.
    for lane in &stored.lanes {
        if !current.lanes.contains(lane) {
            problems.push(format!(
                "STALE evidence — execution lane {lane} is recorded in the stored evidence but did \
                 not run in this build (a `cfg` feature that was enabled when the evidence was \
                 produced is off now?)"
            ));
        }
    }

    // The lane list is the ONLY input to the exemption, so it is also the only
    // thing an attacker editing the stored file would go after: every lane
    // deleted from it widens the exempt set by one, and a claim or trust entry
    // from an exempt lane is a note instead of a refusal. Two guards close that,
    // and they are why the exemption is safe to have at all.
    //
    // (1) A manifest that records claims must record the lane that produced
    //     them, whenever this run has lanes that could be exempted by its
    //     silence. (Both sides empty is not a risk and not a `suite!` shape:
    //     nothing can be exempt when there is no current lane to exempt, so a
    //     hand-built `Manifest::new` manifest is compared at maximum strictness
    //     rather than refused.)
    if stored.lanes.is_empty() && !current.lanes.is_empty() {
        problems.push(
            "the stored evidence records no execution lane at all — the lane list is what scopes \
             the additive-lane exemption, so an empty one would excuse every difference on every \
             backend. Regenerate with `VERICL_UPDATE=1`"
                .to_string(),
        );
        return Vec::new();
    }
    // (2) The PRIMARY lane is never exempt, whatever the list says. The
    //     exemption exists for an `extra_lane` that a `cfg` feature switched on;
    //     the lane a suite always runs is not that, and letting it in would
    //     excuse an erased `"<primary>" buffer upload/readback integrity` trust
    //     entry (which starts with the backend name) as an "unrecorded lane".
    let primary = current.lanes.first();
    if let (Some(st_primary), Some(cur_primary)) = (stored.lanes.first(), primary) {
        if st_primary != cur_primary {
            problems.push(format!(
                "STALE evidence — the PRIMARY execution lane changed: stored {st_primary} -> \
                 current {cur_primary}. The primary lane is the one every claim in this manifest \
                 was measured on"
            ));
        }
    }
    let exempt: Vec<String> = current
        .lanes
        .iter()
        .filter(|l| Some(*l) != primary && !stored.lanes.contains(l))
        .cloned()
        .collect();
    if !exempt.is_empty() {
        unrecorded.push(format!(
            "execution lane(s) {} ran but are not recorded in the stored evidence (stored lanes: \
             {}) — their claims are additional evidence, not a mismatch",
            exempt.join(", "),
            stored.lanes.join(", ")
        ));
    }
    exempt
}

fn compare_manifests(stored: &Manifest, current: &Manifest) -> Comparison {
    let mut problems: Vec<String> = Vec::new();
    let mut unrecorded: Vec<String> = Vec::new();

    // A manifest with no entries passes every check below vacuously — each one
    // quantifies over `entries`. `suite!` rejects `kernels: []` at compile time,
    // but `verify` is public API that anything can call, so it refuses the shape
    // here too rather than returning "no problems" for an empty file.
    if current.entries.is_empty() {
        problems.push(
            "this build produced NO kernel entries — every check below quantifies over the entry \
             list, so an empty manifest verifies vacuously. Refused rather than reported as OK"
                .to_string(),
        );
    }
    if stored.entries.is_empty() && current.entries.is_empty() {
        problems.push(
            "the stored evidence file contains no entries — there is nothing recorded to verify \
             against"
                .to_string(),
        );
    }

    // Entries are matched by kernel NAME (entry order is meaningless — it is
    // just the `kernels:` list order). That makes a duplicated name lossy: the
    // second copy would never be looked at. `suite!` cannot produce one; a
    // hand-edited file can.
    for (side, m) in [("stored", stored), ("current", current)] {
        let mut seen = BTreeSet::new();
        for e in &m.entries {
            if !seen.insert(e.kernel.as_str()) {
                problems.push(format!(
                    "{side} manifest has more than one entry for kernel `{}` — entries are keyed \
                     by kernel name, so a duplicate hides everything after the first",
                    e.kernel
                ));
            }
        }
    }

    if stored.vericl_version != current.vericl_version {
        problems.push(format!(
            "STALE evidence — manifest `vericl_version` {} -> {}",
            stored.vericl_version, current.vericl_version
        ));
    }

    let exempt_lanes =
        provenance_problems(&stored.provenance, &current.provenance, &mut problems, &mut unrecorded);

    for cur in &current.entries {
        let Some(st) = stored.entries.iter().find(|e| e.kernel == cur.kernel) else {
            problems.push(format!("kernel `{}`: no stored evidence — run update", cur.kernel));
            continue;
        };

        if st.identity != cur.identity {
            // Report every mismatched identity field, not just the first — a
            // kernel edit typically changes both the source-level and IR-level
            // hash together, and both must be visible in the failure, not just
            // whichever field happens to differ.
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
                "kernel `{}`: STALE evidence — identity mismatch ({}) (kernel source, contract, \
                 IR, or vericl version changed without renewing evidence)",
                cur.kernel,
                fields.join(", ")
            ));
            // Identity mismatch invalidates everything else about the entry.
            continue;
        }

        contract_problems(&cur.kernel, &st.contract, &cur.contract, &mut problems);

        // --- claims: the full set, normalized ---
        let (pairs, stored_only, current_only) = pair_claims(&st.claims, &cur.claims);

        for (si, ci) in pairs {
            let (s, c) = (&st.claims[si], &cur.claims[ci]);
            // When the backend is what moved, name the claim without it — the
            // dedicated line below carries both values.
            let label = if s.backend == c.backend {
                claim_label(s)
            } else {
                format!("{} `{}`", kind_word(s.kind), s.check)
            };
            if s.backend != c.backend {
                problems.push(format!(
                    "kernel `{}`: {label} — backend changed: stored {} -> current {}",
                    cur.kernel,
                    s.backend.as_deref().unwrap_or("<none>"),
                    c.backend.as_deref().unwrap_or("<none>"),
                ));
            }
            let mut cfg = Vec::new();
            json_diff("config", &s.config, &c.config, &mut cfg);
            for line in cfg {
                problems.push(format!("kernel `{}`: {label} — {line}", cur.kernel));
            }
            // A `Fail` on either side is reported in full (with its detail) by
            // the failure pass below; reporting the status change too would say
            // the same thing twice.
            let either_failed = matches!(s.result, ClaimResult::Fail { .. })
                || matches!(c.result, ClaimResult::Fail { .. });
            if !either_failed && s.result != c.result {
                problems.push(format!(
                    "kernel `{}`: {label} — result changed: stored {} -> current {}",
                    cur.kernel,
                    result_word(&s.result),
                    result_word(&c.result),
                ));
            }
        }

        for si in stored_only {
            let s = &st.claims[si];
            if s.kind == ClaimKind::Proved {
                // Keep the downgrade wording: losing a proof is the specific
                // regression this check was originally written for, and
                // "proved" and "never claimed" are never interchangeable
                // (README "Claims and trust boundaries").
                problems.push(format!(
                    "kernel `{}`: evidence downgraded — stored evidence has a proved `{}` claim \
                     that the current build did not produce (prove disabled, or z3 unavailable?)",
                    cur.kernel, s.check
                ));
            } else {
                problems.push(format!(
                    "kernel `{}`: {} is recorded in the stored evidence but this build did not \
                     produce it — the file claims more than the build supports (a check was \
                     removed, or the claim was written into the file by hand)",
                    cur.kernel,
                    claim_label(s)
                ));
            }
        }

        for ci in current_only {
            let c = &cur.claims[ci];
            // The additive-lane exemption, keyed on the stored provenance's
            // lane list — the same one `trusted` below uses, and the only way
            // a claim this build produced is allowed to be absent from the
            // file. Everything else is a mismatch: the file records a claim SET
            // and this build's set is different.
            let from_unrecorded_lane =
                c.backend.as_deref().is_some_and(|b| exempt_lanes.iter().any(|l| l == b));
            if from_unrecorded_lane {
                unrecorded.push(format!(
                    "kernel `{}`: {} was produced by an execution lane the stored evidence does \
                     not record",
                    cur.kernel,
                    claim_label(c)
                ));
            } else {
                problems.push(format!(
                    "kernel `{}`: {} was produced by this build but is MISSING from the stored \
                     evidence — the recorded claim set is not this build's claim set (a claim was \
                     deleted from the file, or new evidence needs recording: VERICL_UPDATE=1)",
                    cur.kernel,
                    claim_label(c)
                ));
            }
        }

        // --- trusted: a SET, and the asymmetry inverts (see the module note) ---
        //
        // Duplicate-insensitive on purpose: it is a list of components taken on
        // faith, and whether one was pushed twice says nothing about the claim.
        let st_trust: BTreeSet<&str> = st.trusted.iter().map(String::as_str).collect();
        let cur_trust: BTreeSet<&str> = cur.trusted.iter().map(String::as_str).collect();
        for missing in cur_trust.difference(&st_trust) {
            if exempt_lanes.iter().any(|l| missing.starts_with(l.as_str())) {
                unrecorded.push(format!(
                    "kernel `{}`: trusted component contributed by an unrecorded lane: `{missing}`",
                    cur.kernel
                ));
            } else {
                problems.push(format!(
                    "kernel `{}`: trusted component `{missing}` is produced by this build but \
                     MISSING from the stored evidence — an omitted trust dependency makes the \
                     recorded claim look stronger than the one actually established",
                    cur.kernel
                ));
            }
        }
        for extra in st_trust.difference(&cur_trust) {
            problems.push(format!(
                "kernel `{}`: stored evidence records a trusted component this build does not \
                 produce: `{extra}`",
                cur.kernel
            ));
        }

        // --- failures ---
        //
        // Reported per SIDE, and labelled with the backend. Both matter:
        //
        // * a failure this build produced and one the *file* records are
        //   different statements — `VERICL_UPDATE` refuses to write failing
        //   evidence, so a stored `Fail` means the file was hand-edited or
        //   written by an older version, which is worth its own sentence
        //   rather than being phrased as if this run is failing;
        // * the backend is part of the identity of a failure. Two lanes
        //   diverging the same way produce the same `detail`, and a message
        //   that omitted the backend would collapse them into one line —
        //   under-reporting how much is broken.
        for (side, claims) in [("stored", &st.claims), ("current", &cur.claims)] {
            for claim in claims.iter() {
                let ClaimResult::Fail { detail } = &claim.result else { continue };
                problems.push(if side == "current" {
                    format!(
                        "kernel `{}`: {} FAILED in this build: {detail}",
                        cur.kernel,
                        claim_label(claim)
                    )
                } else {
                    format!(
                        "kernel `{}`: the stored evidence records a FAILING {}: {detail} — \
                         `VERICL_UPDATE` refuses to write failing evidence, so this entry was \
                         hand-edited or written by an older vericl",
                        cur.kernel,
                        claim_label(claim)
                    )
                });
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

    Comparison { problems, unrecorded }
}

/// Verify stored evidence against the current build's freshly produced
/// manifest. Returns human-readable problems; empty means the evidence stands.
///
/// # What is compared
///
/// Everything the manifest records, normalized (see below):
///
/// | part | compared as | order |
/// |---|---|---|
/// | manifest `vericl_version` | exact | — |
/// | [`Provenance`] | exact per field; `lanes` subset-checked | `lanes` preserved |
/// | entries | keyed by kernel name; duplicates refused | insensitive |
/// | [`Identity`] | exact, per field | — |
/// | [`ContractRecord`] | exact, per field | **sensitive** (authored, hash-covered) |
/// | [`Claim`] set | multiset on `(kind, check, backend)`, then `(kind, check)` | insensitive |
/// | claim `config` | structural JSON diff | objects insensitive, **arrays sensitive** |
/// | claim `result` | exact | — |
/// | `trusted` | **set** (order- and duplicate-insensitive) | insensitive |
///
/// # The property it enforces
///
/// **The stored claim set must be this build's claim set, field for field.**
/// A claim the file records that the build does not produce, a claim the build
/// produces that the file does not record, a mutated backend / config / result,
/// an erased trust dependency — all are problems.
///
/// There is exactly **one** exemption, and it is scoped by the provenance
/// record rather than by shape: a claim or trust entry contributed by an
/// execution lane the stored [`Provenance::lanes`] says did not run. That is
/// the `suite!` `extra_lane` case (`cargo test --features cpu` adds a whole
/// second backend), and those additions are reported by
/// [`unrecorded_evidence`] as printed notes rather than dropped.
pub fn verify(stored: &Manifest, current: &Manifest) -> Vec<String> {
    compare_manifests(stored, current).problems
}

/// Evidence produced by an execution lane the stored manifest does **not**
/// record — [`verify`]'s one exemption, surfaced rather than dropped.
///
/// `cargo test --features cpu` `cfg`-enables a suite's `extra_lane`, which adds
/// one differential claim and one trust entry per kernel. A manifest committed
/// from a default (wgpu-only) run does not have them, and refusing it for that
/// would make one of the two configurations permanently red. Those additions
/// are strictly more evidence on a lane the file never spoke about, so they are
/// exempt — and reported here, printed by the `suite!`-generated runner on
/// every verifying run, so "this evidence file is missing a whole execution
/// lane" is a line on screen rather than a silence.
///
/// Nothing else appears here. A claim added or removed on a lane the file
/// *does* record is a [`verify`] problem.
pub fn unrecorded_evidence(stored: &Manifest, current: &Manifest) -> Vec<String> {
    compare_manifests(stored, current).unrecorded
}

/// Per-kernel **proof-scope changes** between stored and freshly built
/// evidence, as human-readable `kernel `k`: check N -> M` lines. Empty means
/// every proved claim discharges the same number of obligations it did before.
///
/// # Why this exists (round-11 review, risk-8 residual)
///
/// [`verify`] refuses stale evidence and refuses a *dropped* proved claim, but
/// the `VERICL_UPDATE` path refuses nothing — by construction, since its job is
/// to rewrite the file. That leaves one shape unaccounted for: a change that
/// keeps every claim present and passing while **shrinking what it proves**. A
/// kernel whose bounds proof went from 12 obligations to 2 because a walk
/// started bailing out early still records a passing `smt-oob-freedom` claim,
/// and a routine `VERICL_UPDATE=1 cargo test` would absorb the regression into
/// the committed manifest with nothing on screen.
///
/// This is deliberately a *printed warning* and not a refusal: an obligation
/// count legitimately moves whenever the kernel body changes, so failing on it
/// would make ordinary work impossible. What it buys is that the change is
/// never invisible — the number appears in the update output next to the kernel
/// it belongs to, where the author is already looking, and a drop they did not
/// intend is one line rather than a diff they were not going to read.
///
/// A claim that disappeared entirely is reported as `N -> <none>`, which is the
/// same regression at its limit (and is what [`verify`]'s downgrade check would
/// have caught on the *verify* path).
pub fn obligation_count_changes(stored: &Manifest, current: &Manifest) -> Vec<String> {
    fn obligations(c: &Claim) -> Option<u64> {
        c.config.get("obligations")?.as_u64()
    }
    let mut out = Vec::new();
    for cur in &current.entries {
        let Some(st) = stored.entries.iter().find(|e| e.kernel == cur.kernel) else {
            continue; // a brand-new kernel has nothing to compare against
        };
        for st_claim in &st.claims {
            let Some(old) = obligations(st_claim) else { continue };
            match cur.claims.iter().find(|c| c.kind == st_claim.kind && c.check == st_claim.check) {
                Some(cur_claim) => match obligations(cur_claim) {
                    Some(new) if new != old => out.push(format!(
                        "kernel `{}`: `{}` obligations {old} -> {new}",
                        cur.kernel, st_claim.check
                    )),
                    Some(_) => {}
                    None => out.push(format!(
                        "kernel `{}`: `{}` obligations {old} -> <not recorded>",
                        cur.kernel, st_claim.check
                    )),
                },
                None => out.push(format!(
                    "kernel `{}`: `{}` obligations {old} -> <none> (the claim is gone)",
                    cur.kernel, st_claim.check
                )),
            }
        }
    }
    out
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

    /// The round-11 risk-8 residual: `VERICL_UPDATE` refuses nothing, so a
    /// proof-scope regression that keeps every claim present and passing would
    /// otherwise be absorbed into the committed manifest silently. Every
    /// direction of the comparison, including the negative controls that make
    /// it non-vacuous.
    #[test]
    fn obligation_count_changes_are_surfaced_per_kernel() {
        fn with(kernel: &str, check: &str, obligations: u64) -> Entry {
            let mut e = entry(kernel, "aaa");
            e.claims.push(Claim {
                kind: ClaimKind::Proved,
                check: check.into(),
                backend: None,
                config: proved_config("z3 4.13", obligations as usize),
                result: ClaimResult::Pass,
            });
            e
        }

        // A shrink — the regression this exists for.
        let stored = Manifest::new(vec![with("k", "smt-oob-freedom", 12)]);
        let current = Manifest::new(vec![with("k", "smt-oob-freedom", 2)]);
        let msgs = obligation_count_changes(&stored, &current);
        assert_eq!(msgs.len(), 1, "{msgs:?}");
        assert!(msgs[0].contains("kernel `k`"), "{msgs:?}");
        assert!(msgs[0].contains("smt-oob-freedom"), "{msgs:?}");
        assert!(msgs[0].contains("12 -> 2"), "{msgs:?}");

        // A growth is reported too — an unexpected INCREASE is as much a
        // signal that the kernel is not what the author thinks it is.
        let grown = Manifest::new(vec![with("k", "smt-oob-freedom", 40)]);
        assert!(obligation_count_changes(&stored, &grown)[0].contains("12 -> 40"));

        // Unchanged: silent.
        assert!(obligation_count_changes(&stored, &stored).is_empty());

        // The claim disappearing entirely is the same regression at its limit.
        let dropped = Manifest::new(vec![entry("k", "aaa")]);
        let d = obligation_count_changes(&stored, &dropped);
        assert_eq!(d.len(), 1, "{d:?}");
        assert!(d[0].contains("<none>"), "{d:?}");

        // A brand-new kernel has nothing to compare against, and a claim with
        // no `obligations` key (a differential claim) is not a proof scope.
        let fresh = Manifest::new(vec![with("newly_added", "smt-oob-freedom", 5)]);
        assert!(obligation_count_changes(&stored, &fresh).is_empty());
        let mut tested_only = entry("k", "aaa");
        tested_only.claims.push(Claim {
            kind: ClaimKind::Tested,
            check: "differential".into(),
            backend: None,
            config: differential_config(&[4], 1, 64),
            result: ClaimResult::Pass,
        });
        let t = Manifest::new(vec![tested_only]);
        assert!(obligation_count_changes(&t, &t).is_empty());

        // Per-kernel, not global: two kernels, one moved.
        let stored2 =
            Manifest::new(vec![with("a", "smt-oob-freedom", 3), with("b", "smt-oob-freedom", 9)]);
        let current2 =
            Manifest::new(vec![with("a", "smt-oob-freedom", 3), with("b", "smt-oob-freedom", 1)]);
        let m2 = obligation_count_changes(&stored2, &current2);
        assert_eq!(m2.len(), 1, "{m2:?}");
        assert!(m2[0].contains("kernel `b`"), "{m2:?}");
    }

    // -----------------------------------------------------------------------
    // COMPLETE-MANIFEST VERIFICATION — one regression per tamper class the
    // external consumer review listed, each with the untampered pair as its
    // own negative control (before: clean; after: named problem).
    //
    // `tampered(f)` returns `(problems, unrecorded)` for a realistic entry with
    // `f` applied to the STORED side — i.e. someone edited the committed
    // evidence file — and asserts the *untampered* pair is clean first, so no
    // test in this block can pass by the comparison being broken.
    // -----------------------------------------------------------------------

    /// A realistic entry: the shape every non-cooperative suite kernel records
    /// — one tested differential claim on a backend, one proved bounds claim,
    /// and the trust list those two imply.
    fn realistic_entry() -> Entry {
        let mut e = entry("axpy", "sha256:aaa");
        e.identity.ir_hash = Some("sha256:ir-aaa".into());
        e.contract.assumes = vec!["x.iter().all(| v | v.abs() <= 1000.0)".into()];
        e.contract.compare = "f32 max_ulp=0".into();
        e.claims = vec![
            Claim {
                kind: ClaimKind::Tested,
                check: "differential".into(),
                backend: Some("\"wgpu<wgsl>\"".into()),
                config: differential_config(&[1, 7, 256], 0xE901, 256),
                result: ClaimResult::Pass,
            },
            Claim {
                kind: ClaimKind::Proved,
                check: "smt-oob-freedom".into(),
                backend: None,
                config: proved_config("z3 4.16.0", 3),
                result: ClaimResult::Pass,
            },
        ];
        e.trusted = crate::reference_twin_trust();
        e.trusted.push(crate::backend_buffer_trust("\"wgpu<wgsl>\""));
        e.trusted.push(crate::GPU_HARDWARE_TRUST.to_string());
        e.trusted.extend(crate::proved_bounds_trust("z3 4.16.0"));
        e
    }

    fn recorded(entries: Vec<Entry>, lanes: &[&str]) -> Manifest {
        let mut p = crate::Provenance::current();
        p.vericl_ir = "0.1.0".into();
        p.vericl_macros = "0.1.0".into();
        p.z3 = Some("z3 4.16.0".into());
        p.lanes = lanes.iter().map(|s| s.to_string()).collect();
        p.device = Some("Metal".into());
        Manifest::with_provenance(entries, p)
    }

    /// Apply `tamper` to the stored side of an otherwise-matching pair, after
    /// pinning that the untampered pair verifies clean.
    fn tampered(tamper: impl FnOnce(&mut Entry)) -> (Vec<String>, Vec<String>) {
        let current = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);
        let clean = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);
        assert!(
            verify(&clean, &current).is_empty(),
            "NEGATIVE CONTROL BROKEN — the untampered pair must verify clean: {:?}",
            verify(&clean, &current)
        );
        assert!(unrecorded_evidence(&clean, &current).is_empty());

        let mut e = realistic_entry();
        tamper(&mut e);
        let stored = recorded(vec![e], &["\"wgpu<wgsl>\""]);
        (verify(&stored, &current), unrecorded_evidence(&stored, &current))
    }

    fn only_problem(problems: &[String]) -> &str {
        assert_eq!(problems.len(), 1, "expected exactly one problem: {problems:#?}");
        &problems[0]
    }

    /// TAMPER CLASS 1 — a tested claim the file advertises that the build no
    /// longer produces (a check removed from the suite, or a hand-written
    /// claim). Before this change `verify` only checked *proved* claims for
    /// this, so a deleted differential was invisible.
    #[test]
    fn tamper_tested_claim_no_longer_produced_is_caught() {
        // Stored keeps the tested claim; current drops it.
        let mut cur_entry = realistic_entry();
        cur_entry.claims.retain(|c| c.kind != ClaimKind::Tested);
        let current = recorded(vec![cur_entry], &["\"wgpu<wgsl>\""]);
        let stored = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);
        let p = verify(&stored, &current);
        let msg = only_problem(&p);
        assert!(msg.contains("tested `differential`"), "{msg}");
        assert!(msg.contains("did not produce it"), "{msg}");
    }

    /// TAMPER CLASS 1b — the same claim deleted from the FILE while the build
    /// still produces it. The recorded claim set is not this build's claim set,
    /// so it is a problem too (and NOT an exempt lane, since the backend is one
    /// the stored provenance records).
    #[test]
    fn tamper_tested_claim_deleted_from_the_file_is_caught() {
        let (p, u) = tampered(|e| e.claims.retain(|c| c.kind != ClaimKind::Tested));
        let msg = only_problem(&p);
        assert!(msg.contains("tested `differential`"), "{msg}");
        assert!(msg.contains("MISSING from the stored evidence"), "{msg}");
        assert!(u.is_empty(), "{u:?}");
    }

    /// TAMPER CLASS 2 — the backend a tested claim was measured on is changed.
    /// Reported as one field diff (not an unrelated removal + addition) because
    /// the pairing falls back to `(kind, check)`.
    #[test]
    fn tamper_backend_changed_is_caught() {
        let (p, _) = tampered(|e| e.claims[0].backend = Some("\"cuda\"".into()));
        let msg = only_problem(&p);
        assert!(msg.contains("backend changed"), "{msg}");
        assert!(msg.contains("\"cuda\""), "{msg}");
        assert!(msg.contains("wgpu<wgsl>"), "{msg}");
    }

    /// TAMPER CLASS 3 — the `sizes` the differential actually ran over are
    /// altered (here: the two large cases dropped, so the file advertises
    /// coverage the run never had). The diff names the field AND both values.
    #[test]
    fn tamper_sizes_altered_is_caught() {
        let (p, _) = tampered(|e| e.claims[0].config = differential_config(&[1], 0xE901, 256));
        let msg = only_problem(&p);
        assert!(msg.contains("config.sizes"), "{msg}");
        assert!(msg.contains("stored [1]"), "{msg}");
        assert!(msg.contains("current [1,7,256]"), "{msg}");
    }

    /// TAMPER CLASS 3b — the other config fields the review named: seed,
    /// cube_dim, solver, obligation count. All are structurally diffed, so each
    /// is one named line rather than an opaque "config differs".
    #[test]
    fn tamper_seed_cube_dim_solver_and_obligations_are_each_named() {
        let (p, _) = tampered(|e| e.claims[0].config = differential_config(&[1, 7, 256], 1, 256));
        assert!(only_problem(&p).contains("config.seed"), "{p:?}");

        let (p, _) = tampered(|e| e.claims[0].config = differential_config(&[1, 7, 256], 0xE901, 64));
        assert!(only_problem(&p).contains("config.cube_dim"), "{p:?}");

        let (p, _) = tampered(|e| e.claims[1].config = proved_config("z3 4.8.7", 3));
        let msg = only_problem(&p);
        assert!(msg.contains("config.solver"), "{msg}");
        assert!(msg.contains("4.8.7"), "{msg}");

        let (p, _) = tampered(|e| e.claims[1].config = proved_config("z3 4.16.0", 99));
        let msg = only_problem(&p);
        assert!(msg.contains("config.obligations"), "{msg}");
        assert!(msg.contains("stored 99 -> current 3"), "{msg}");
    }

    /// TAMPER CLASS 4 — a trust dependency erased from the file. This is the
    /// direction that makes evidence look STRONGER than it is (fewer components
    /// taken on faith), so it is a problem even though the file now says
    /// *less*.
    #[test]
    fn tamper_trust_dependency_erased_is_caught() {
        let (p, u) = tampered(|e| e.trusted.retain(|t| !t.contains("solver binary")));
        let msg = only_problem(&p);
        assert!(msg.contains("MISSING from the stored evidence"), "{msg}");
        assert!(msg.contains("solver binary"), "{msg}");
        assert!(msg.contains("stronger"), "{msg}");
        assert!(u.is_empty(), "{u:?}");

        // The other direction — a trust entry in the file the build does not
        // produce — is also reported, with its own wording.
        let (p, _) = tampered(|e| e.trusted.push("a component nobody trusts".into()));
        assert!(only_problem(&p).contains("does not produce"), "{p:?}");
    }

    /// TAMPER CLASS 5 — an arbitrary passing claim typed into the file. The
    /// build never produced it, so the file claims more than the build
    /// supports.
    #[test]
    fn tamper_arbitrary_passing_claim_added_is_caught() {
        let (p, _) = tampered(|e| {
            e.claims.push(Claim {
                kind: ClaimKind::Tested,
                check: "formally-verified-by-vibes".into(),
                backend: Some("\"wgpu<wgsl>\"".into()),
                config: serde_json::json!({"trust me": true}),
                result: ClaimResult::Pass,
            })
        });
        let msg = only_problem(&p);
        assert!(msg.contains("formally-verified-by-vibes"), "{msg}");
        assert!(msg.contains("did not produce it"), "{msg}");

        // …including one that impersonates a PROVED claim, which keeps the
        // downgrade wording (proved and "never claimed" are not interchangeable).
        let (p, _) = tampered(|e| {
            e.claims.push(Claim {
                kind: ClaimKind::Proved,
                check: "smt-race-freedom".into(),
                backend: None,
                config: proved_config("z3 4.16.0", 12),
                result: ClaimResult::Pass,
            })
        });
        let msg = only_problem(&p);
        assert!(msg.contains("downgraded"), "{msg}");
        assert!(msg.contains("smt-race-freedom"), "{msg}");
    }

    /// TAMPER CLASS 6 — a claim's recorded RESULT flipped (a `declared`
    /// assumption relabelled as a `pass`, say). The status change is its own
    /// named line.
    #[test]
    fn tamper_result_status_changed_is_caught() {
        let mut cur_entry = realistic_entry();
        cur_entry.claims.push(race_freedom_assumption_claim());
        let current = recorded(vec![cur_entry.clone()], &["\"wgpu<wgsl>\""]);
        assert!(verify(&recorded(vec![cur_entry.clone()], &["\"wgpu<wgsl>\""]), &current).is_empty());

        let mut st_entry = cur_entry;
        st_entry.claims.last_mut().unwrap().result = ClaimResult::Pass;
        let stored = recorded(vec![st_entry], &["\"wgpu<wgsl>\""]);
        let p = verify(&stored, &current);
        let msg = only_problem(&p);
        assert!(msg.contains("result changed"), "{msg}");
        assert!(msg.contains("stored pass -> current declared"), "{msg}");
    }

    /// TAMPER CLASS 7 — the contract record. Order-sensitive on `assumes` (an
    /// authored, hash-covered list), and every field named individually.
    #[test]
    fn tamper_contract_fields_are_each_named() {
        let (p, _) = tampered(|e| e.contract.compare = "f32 max_ulp=4".into());
        let msg = only_problem(&p);
        assert!(msg.contains("contract field `compare`"), "{msg}");
        assert!(msg.contains("max_ulp=4"), "{msg}");

        let (p, _) = tampered(|e| e.contract.wrapping = true);
        assert!(only_problem(&p).contains("contract field `wrapping`"), "{p:?}");

        let (p, _) = tampered(|e| e.contract.assumes.clear());
        assert!(only_problem(&p).contains("contract field `assumes`"), "{p:?}");

        let (p, _) = tampered(|e| e.contract.uses = vec!["b".into(), "a".into()]);
        assert!(only_problem(&p).contains("contract field `uses`"), "{p:?}");
    }

    /// TAMPER CLASS 8 — the IR hash, which `prove: false` evidence used to
    /// leave `null`. Now that it is always populated, blanking it is caught by
    /// the identity comparison and named as the `ir_hash` field.
    #[test]
    fn tamper_ir_hash_blanked_is_caught() {
        let (p, _) = tampered(|e| e.identity.ir_hash = None);
        let msg = only_problem(&p);
        assert!(msg.contains("STALE"), "{msg}");
        assert!(msg.contains("ir_hash <none> -> sha256:ir-aaa"), "{msg}");
    }

    /// A whole entry duplicated in the file: entries are keyed by kernel name,
    /// so a second copy would never be looked at.
    #[test]
    fn duplicate_kernel_entries_are_refused() {
        let current = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);
        let stored = recorded(vec![realistic_entry(), realistic_entry()], &["\"wgpu<wgsl>\""]);
        let p = verify(&stored, &current);
        assert!(p.iter().any(|m| m.contains("more than one entry for kernel `axpy`")), "{p:#?}");
    }

    /// The vacuous manifest: nothing to quantify over, so every check is
    /// trivially satisfied. Refused rather than reported OK.
    #[test]
    fn an_empty_manifest_does_not_verify_vacuously() {
        let empty = recorded(vec![], &["\"wgpu<wgsl>\""]);
        let p = verify(&empty, &empty);
        assert!(p.iter().any(|m| m.contains("NO kernel entries")), "{p:#?}");
        assert!(p.iter().any(|m| m.contains("nothing recorded to verify against")), "{p:#?}");

        // NEGATIVE CONTROL: one real entry and the vacuity messages are gone.
        let real = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);
        assert!(verify(&real, &real).is_empty());
    }

    // ---- provenance (the verification-environment fingerprint) ----

    /// Evidence from another toolchain is STALE-class, and the message names
    /// the field and both values.
    #[test]
    fn a_different_toolchain_is_stale_not_silently_accepted() {
        let current = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);
        for (label, tamper) in [
            ("rustc", Box::new(|p: &mut crate::Provenance| p.rustc = "rustc 1.0.0 (old)".into())
                as Box<dyn FnOnce(&mut crate::Provenance)>),
            ("target", Box::new(|p: &mut crate::Provenance| p.target = "x86_64-unknown-linux-gnu".into())),
            ("cubecl", Box::new(|p: &mut crate::Provenance| p.cubecl = "=0.9.0".into())),
            ("vericl-ir", Box::new(|p: &mut crate::Provenance| p.vericl_ir = "0.0.9".into())),
            ("vericl-macros", Box::new(|p: &mut crate::Provenance| p.vericl_macros = "0.0.9".into())),
            ("z3", Box::new(|p: &mut crate::Provenance| p.z3 = Some("z3 4.8.7".into()))),
            ("device", Box::new(|p: &mut crate::Provenance| p.device = Some("Vulkan".into()))),
        ] {
            let mut stored = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);
            tamper(&mut stored.provenance);
            let p = verify(&stored, &current);
            let msg = only_problem(&p);
            assert!(msg.contains("STALE"), "{label}: {msg}");
            assert!(msg.contains("verification environment changed"), "{label}: {msg}");
            assert!(msg.contains(label), "{label}: {msg}");
        }
    }

    /// Evidence written before the fingerprint existed still LOADS (schema is
    /// additive) but does not still VERIFY.
    #[test]
    fn evidence_without_a_provenance_record_is_refused_but_still_parses() {
        let json = serde_json::json!({
            "vericl_version": crate::VERSION,
            "entries": [],
        });
        let legacy: Manifest = serde_json::from_value(json).expect("additive schema still loads");
        assert!(!legacy.provenance.is_recorded());
        let current = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);
        let p = verify(&legacy, &current);
        assert!(p.iter().any(|m| m.contains("no verification-environment record")), "{p:#?}");
    }

    /// The ONE exemption: an execution lane the stored evidence does not
    /// record contributes claims and trust entries that are notes, not
    /// problems. This is `cargo test --features cpu` against a manifest
    /// committed from a default run.
    #[test]
    fn an_extra_execution_lane_is_additional_evidence_not_a_mismatch() {
        let stored = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);

        let mut cur_entry = realistic_entry();
        cur_entry.claims.push(Claim {
            kind: ClaimKind::Tested,
            check: "differential".into(),
            backend: Some("\"cpu\"".into()),
            config: differential_config(&[1, 7, 256], 0xE901, 256),
            result: ClaimResult::Pass,
        });
        cur_entry.trusted.push(crate::shared_frontend_lane_trust("\"cpu\""));
        let current = recorded(vec![cur_entry], &["\"wgpu<wgsl>\"", "\"cpu\""]);

        let p = verify(&stored, &current);
        assert!(p.is_empty(), "an added lane must not be a mismatch: {p:#?}");
        let u = unrecorded_evidence(&stored, &current);
        assert_eq!(u.len(), 3, "{u:#?}");
        assert!(u.iter().any(|m| m.contains("execution lane(s) \"cpu\"")), "{u:#?}");
        assert!(u.iter().any(|m| m.contains("tested `differential` (backend \"cpu\")")), "{u:#?}");
        assert!(u.iter().any(|m| m.contains("trusted component contributed")), "{u:#?}");
    }

    /// ATTACK ON THE EXEMPTION ITSELF. The lane list is its only input, so it
    /// is the thing to edit: every lane deleted from the stored file widens the
    /// exempt set by one. Emptying it entirely would exempt every backend —
    /// including an erased `"<primary>" buffer upload/readback integrity` trust
    /// entry, which starts with the backend name and would match the
    /// `starts_with` test. Refused, so the exemption cannot be opened up.
    #[test]
    fn emptying_the_stored_lane_list_does_not_widen_the_exemption() {
        let mut st_entry = realistic_entry();
        st_entry.trusted.retain(|t| !t.contains("buffer upload/readback"));
        let mut stored = recorded(vec![st_entry], &[]);
        assert!(stored.provenance.lanes.is_empty());
        let current = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);
        let p = verify(&stored, &current);
        assert!(p.iter().any(|m| m.contains("records no execution lane at all")), "{p:#?}");
        // …and with the lane list restored, the erasure is still caught (i.e.
        // the refusal above is not the only thing standing between the tamper
        // and a green run).
        stored.provenance.lanes = vec!["\"wgpu<wgsl>\"".into()];
        let p = verify(&stored, &current);
        assert!(p.iter().any(|m| m.contains("buffer upload/readback")), "{p:#?}");
        assert!(p.iter().any(|m| m.contains("stronger")), "{p:#?}");
    }

    /// The PRIMARY lane is never exempt, whatever the lane list says. The
    /// exemption is for a `cfg`-enabled `extra_lane`; the lane a suite always
    /// runs is not that.
    #[test]
    fn the_primary_lane_is_never_exempt() {
        // Stored records only the cpu lane; this build's PRIMARY is wgpu. The
        // wgpu claim is missing from the file and must not be excused.
        let mut st_entry = realistic_entry();
        st_entry.claims[0].backend = Some("\"cpu\"".into());
        let stored = recorded(vec![st_entry], &["\"cpu\""]);
        let current = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);
        let p = verify(&stored, &current);
        assert!(p.iter().any(|m| m.contains("PRIMARY execution lane changed")), "{p:#?}");
        assert!(unrecorded_evidence(&stored, &current).is_empty(), "the primary lane is not exempt");
    }

    /// ADVERSARIAL-REVIEW REGRESSION (round-13 pre-review, CRITICAL as filed).
    /// The reviewer's exact counterexample, at struct level: delete a kernel's
    /// tested claim from the file AND empty `provenance.lanes`, so the deleted
    /// claim's backend lands in the exempt set and the removal reads as a note.
    /// It verified GREEN against the pre-guard code.
    #[test]
    fn deleting_a_claim_and_its_lane_marker_together_is_still_refused() {
        let mut st_entry = realistic_entry();
        st_entry.claims.retain(|c| c.kind != ClaimKind::Tested);
        let mut stored = recorded(vec![st_entry.clone()], &[]);
        let current = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);

        let p = verify(&stored, &current);
        assert!(p.iter().any(|m| m.contains("records no execution lane at all")), "{p:#?}");
        assert!(
            p.iter().any(|m| m.contains("tested `differential`")
                && m.contains("MISSING from the stored evidence")),
            "the deleted claim must still be reported, not exempted: {p:#?}"
        );
        assert!(unrecorded_evidence(&stored, &current).is_empty(), "nothing may be exempt here");

        // The same attack with a PLAUSIBLE lane list rather than an empty one:
        // name a lane that did not run, so the deleted claim's backend is still
        // "unrecorded". Caught by the primary-lane rule.
        stored.provenance.lanes = vec!["\"cpu\"".into()];
        let p = verify(&stored, &current);
        assert!(p.iter().any(|m| m.contains("PRIMARY execution lane changed")), "{p:#?}");
        assert!(
            p.iter().any(|m| m.contains("MISSING from the stored evidence")),
            "{p:#?}"
        );
    }

    /// ADVERSARIAL-REVIEW REGRESSION (finding 5). Two lanes failing the same way
    /// produce the same `detail`; a message keyed without the backend collapses
    /// them into one line and under-reports how much is broken.
    #[test]
    fn two_lanes_failing_identically_are_two_reported_failures() {
        let fail = |backend: &str| Claim {
            kind: ClaimKind::Tested,
            check: "differential".into(),
            backend: Some(backend.into()),
            config: differential_config(&[1, 7, 256], 0xE901, 256),
            result: ClaimResult::Fail { detail: "n=256 `y`: 4/256 elements diverge".into() },
        };
        let mut e = entry("k", "aaa");
        e.claims = vec![fail("\"wgpu<wgsl>\""), fail("\"cpu\"")];
        let m = recorded(vec![e], &["\"wgpu<wgsl>\"", "\"cpu\""]);
        let p = verify(&m, &m);
        let failures: Vec<&String> = p.iter().filter(|s| s.contains("FAILED in this build")).collect();
        assert_eq!(failures.len(), 2, "one line per failing lane: {p:#?}");
        assert!(failures.iter().any(|s| s.contains("wgpu<wgsl>")), "{failures:#?}");
        assert!(failures.iter().any(|s| s.contains("\"cpu\"")), "{failures:#?}");
    }

    /// ADVERSARIAL-REVIEW REGRESSION (finding 5, second half). A failure the
    /// FILE records is not a failure this build produced, and must not be
    /// phrased as one — `VERICL_UPDATE` refuses to write failing evidence, so a
    /// stored `Fail` means the file was edited.
    #[test]
    fn a_failure_recorded_in_the_file_is_reported_as_the_files_failure() {
        let mut st_entry = realistic_entry();
        st_entry.claims[0].result = ClaimResult::Fail { detail: "invented".into() };
        let stored = recorded(vec![st_entry], &["\"wgpu<wgsl>\""]);
        let current = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);
        let p = verify(&stored, &current);
        assert!(p.iter().any(|m| m.contains("the stored evidence records a FAILING")), "{p:#?}");
        assert!(!p.iter().any(|m| m.contains("FAILED in this build")), "{p:#?}");
    }

    /// ADVERSARIAL-REVIEW REGRESSION (finding 3). Two long values agreeing on a
    /// long prefix must not RENDER identically — `stored X -> current X` for a
    /// real difference is a message that actively misleads. The window is
    /// centred on the first divergence.
    #[test]
    fn a_diff_late_in_a_long_value_is_still_visible_in_the_message() {
        let long = |tail: &str| {
            serde_json::json!({ "statement": format!("{}{tail}", "context ".repeat(40)) })
        };
        let mut st_entry = realistic_entry();
        st_entry.claims[0].config = long("AAA");
        let stored = recorded(vec![st_entry], &["\"wgpu<wgsl>\""]);
        let mut cur_entry = realistic_entry();
        cur_entry.claims[0].config = long("BBB");
        let current = recorded(vec![cur_entry], &["\"wgpu<wgsl>\""]);

        let p = verify(&stored, &current);
        let msg = only_problem(&p);
        assert!(msg.contains("AAA"), "the stored side's difference must be visible: {msg}");
        assert!(msg.contains("BBB"), "the current side's difference must be visible: {msg}");
        // …and the two rendered sides are not the same string.
        let (a, b) = msg.split_once(" -> current ").expect("a two-sided diff line");
        assert_ne!(a.rsplit("stored ").next(), Some(b), "{msg}");
    }

    /// ADVERSARIAL-REVIEW REGRESSION (finding 2). When several claims share a
    /// key, the content-equal pass must pair the unchanged ones first, so the
    /// config diff is attributed to the claim that actually changed rather than
    /// to whichever happened to come first in the array.
    #[test]
    fn a_config_diff_is_attributed_to_the_claim_that_changed() {
        let claim = |backend: &str, seed: u64| Claim {
            kind: ClaimKind::Tested,
            check: "differential".into(),
            backend: Some(backend.into()),
            config: differential_config(&[1, 7, 256], seed, 256),
            result: ClaimResult::Pass,
        };
        // Same key on both, different array order, and only the cpu one moved.
        let mut st_entry = entry("k", "aaa");
        st_entry.claims = vec![claim("\"cpu\"", 99), claim("\"wgpu<wgsl>\"", 1)];
        let mut cur_entry = entry("k", "aaa");
        cur_entry.claims = vec![claim("\"wgpu<wgsl>\"", 1), claim("\"cpu\"", 1)];
        let stored = recorded(vec![st_entry], &["\"wgpu<wgsl>\"", "\"cpu\""]);
        let current = recorded(vec![cur_entry], &["\"wgpu<wgsl>\"", "\"cpu\""]);

        let p = verify(&stored, &current);
        let msg = only_problem(&p);
        assert!(msg.contains("config.seed"), "{msg}");
        assert!(msg.contains("backend \"cpu\""), "blamed the wrong lane: {msg}");
        assert!(!msg.contains("wgpu"), "the unchanged lane must not be blamed: {msg}");
    }

    /// The exemption is NOT a general "extra claims are fine" hole: the same
    /// added claim on a lane the stored evidence DOES record is a problem.
    /// (Negative control for the test above.)
    #[test]
    fn the_lane_exemption_does_not_excuse_an_added_claim_on_a_recorded_lane() {
        let stored = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\"", "\"cpu\""]);
        let mut cur_entry = realistic_entry();
        cur_entry.claims.push(Claim {
            kind: ClaimKind::Tested,
            check: "differential".into(),
            backend: Some("\"cpu\"".into()),
            config: differential_config(&[1, 7, 256], 0xE901, 256),
            result: ClaimResult::Pass,
        });
        let current = recorded(vec![cur_entry], &["\"wgpu<wgsl>\"", "\"cpu\""]);
        let p = verify(&stored, &current);
        assert!(p.iter().any(|m| m.contains("MISSING from the stored evidence")), "{p:#?}");
    }

    /// A lane the stored evidence records that did NOT run is a loss, not an
    /// exemption — the reverse direction of the test above.
    #[test]
    fn a_lane_that_stopped_running_is_refused() {
        let mut st_entry = realistic_entry();
        st_entry.claims.push(Claim {
            kind: ClaimKind::Tested,
            check: "differential".into(),
            backend: Some("\"cpu\"".into()),
            config: differential_config(&[1, 7, 256], 0xE901, 256),
            result: ClaimResult::Pass,
        });
        let stored = recorded(vec![st_entry], &["\"wgpu<wgsl>\"", "\"cpu\""]);
        let current = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);
        let p = verify(&stored, &current);
        assert!(p.iter().any(|m| m.contains("execution lane \"cpu\"")), "{p:#?}");
        assert!(p.iter().any(|m| m.contains("did not produce it")), "{p:#?}");
    }

    // ---- normalization: the deliberate order choices ----

    /// Claim ORDER is meaningless (the pipeline pushes tested-then-proved, the
    /// cooperative branch inserts at 0, an extra lane appends) and is ignored.
    #[test]
    fn claim_order_is_not_significant() {
        let current = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);
        let mut st_entry = realistic_entry();
        st_entry.claims.reverse();
        let stored = recorded(vec![st_entry], &["\"wgpu<wgsl>\""]);
        assert!(verify(&stored, &current).is_empty());
    }

    /// Entry ORDER is meaningless (it is the `kernels:` list order) and is
    /// ignored; `trusted` ORDER and DUPLICATES are meaningless (it is a set of
    /// components taken on faith) and are ignored.
    #[test]
    fn entry_order_and_trusted_order_and_duplicates_are_not_significant() {
        let other = {
            let mut e = realistic_entry();
            e.kernel = "fir3".into();
            e
        };
        let current = recorded(vec![realistic_entry(), other.clone()], &["\"wgpu<wgsl>\""]);

        let mut shuffled = realistic_entry();
        shuffled.trusted.reverse();
        shuffled.trusted.push(shuffled.trusted[0].clone()); // a duplicate says nothing
        let stored = recorded(vec![other, shuffled], &["\"wgpu<wgsl>\""]);
        assert!(verify(&stored, &current).is_empty(), "{:#?}", verify(&stored, &current));
    }

    /// Config ARRAY order IS significant: `sizes: [7, 1]` is a different
    /// declaration from `sizes: [1, 7]`. Sensitivity is the safe direction —
    /// it can only ask for a regeneration, never let a real change through.
    #[test]
    fn config_array_order_is_significant() {
        let (p, _) = tampered(|e| e.claims[0].config = differential_config(&[256, 7, 1], 0xE901, 256));
        // Same length, so the diff is per-index — which is more useful than a
        // whole-array dump: it names exactly which positions moved.
        assert_eq!(p.len(), 2, "{p:#?}");
        assert!(p[0].contains("config.sizes[0]: stored 256 -> current 1"), "{p:#?}");
        assert!(p[1].contains("config.sizes[2]: stored 1 -> current 256"), "{p:#?}");
    }

    /// Config OBJECT key order is not significant — `serde_json`'s map is
    /// key-ordered on load, and a JSON object is unordered by definition. The
    /// same fields written in a different textual order must verify clean.
    #[test]
    fn config_object_key_order_is_not_significant() {
        let current = recorded(vec![realistic_entry()], &["\"wgpu<wgsl>\""]);
        let mut st_entry = realistic_entry();
        // Byte-reversed key order in the source text, same content.
        st_entry.claims[1].config = serde_json::from_str(
            r#"{"obligations": 3, "logic": "QF_LIA", "solver": "z3 4.16.0"}"#,
        )
        .unwrap();
        let stored = recorded(vec![st_entry], &["\"wgpu<wgsl>\""]);
        assert!(verify(&stored, &current).is_empty(), "{:#?}", verify(&stored, &current));
    }

    /// A nested config object (the cooperative `depends_on` coupling) diffs to
    /// a dotted path, so "the dependency was quietly relabelled discharged" is
    /// one readable line.
    #[test]
    fn a_nested_config_field_diffs_to_a_dotted_path() {
        let honest = cooperative_differential_config(
            &[256],
            1,
            256,
            "twin",
            RaceDependency::Assumed,
        );
        let overclaimed = cooperative_differential_config(
            &[256],
            1,
            256,
            "twin",
            RaceDependency::Discharged,
        );
        let mut cur_entry = realistic_entry();
        cur_entry.claims[0].config = honest;
        let current = recorded(vec![cur_entry], &["\"wgpu<wgsl>\""]);

        let mut st_entry = realistic_entry();
        st_entry.claims[0].config = overclaimed;
        let stored = recorded(vec![st_entry], &["\"wgpu<wgsl>\""]);

        let p = verify(&stored, &current);
        assert!(p.iter().any(|m| m.contains("config.depends_on.status")), "{p:#?}");
        assert!(p.iter().any(|m| m.contains("discharged-by-proof")), "{p:#?}");
        assert!(p.iter().any(|m| m.contains("assumed-undischarged")), "{p:#?}");
    }

    /// An added / removed config KEY is named, in both directions, rather than
    /// being an opaque whole-object diff.
    #[test]
    fn an_added_or_removed_config_key_is_named() {
        let (p, _) = tampered(|e| {
            e.claims[1].config = serde_json::json!({"solver": "z3 4.16.0", "obligations": 3})
        });
        assert!(only_problem(&p).contains("config.logic: stored <absent>"), "{p:?}");

        let (p, _) = tampered(|e| {
            let o = e.claims[1].config.as_object_mut().unwrap();
            o.insert("nonsense".into(), serde_json::json!(true));
        });
        assert!(only_problem(&p).contains("config.nonsense: stored true -> current <absent>"), "{p:?}");
    }

    #[test]
    fn case_outcome_pass_and_describe() {
        let good_report =
            CompareReport { pass: true, checked: 4, mismatches: 0, max_ulp: None, worst: None };
        let ok = CaseOutcome {
            case: "n=4".into(),
            reports: vec![("y".to_string(), good_report)],
            reference_panic: None,
        };
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

    // -----------------------------------------------------------------------
    // Vacuity backstops (external consumer review — the "zero outcomes,
    // `all()` is true" shape, at the per-case level).
    // -----------------------------------------------------------------------

    /// A case with no compared parameter at all compares nothing, so it is not
    /// a pass — and the description says so instead of printing "pass".
    #[test]
    fn a_case_that_compared_no_parameter_is_not_a_pass() {
        let vacuous = CaseOutcome { case: "n=4".into(), reports: vec![], reference_panic: None };
        assert!(!vacuous.pass(), "an empty report list is agreement over nothing");
        let msg = describe_case_outcome(&vacuous);
        assert!(msg.contains("NOTHING WAS COMPARED"), "{msg}");
        assert!(!msg.contains(": pass"), "{msg}");
    }

    /// A zero-element comparison (`sizes: [0]`, or a `gen(len(y = 0))` pin
    /// behind a const the macro cannot fold) reports `checked: 0, pass: true`
    /// — "0/0 elements diverge" is agreement over nothing.
    #[test]
    fn a_case_that_compared_zero_elements_is_not_a_pass() {
        let empty = CompareReport { pass: true, checked: 0, mismatches: 0, max_ulp: None, worst: None };
        let vacuous = CaseOutcome {
            case: "n=0".into(),
            reports: vec![("y".to_string(), empty)],
            reference_panic: None,
        };
        assert!(!vacuous.pass());
        let msg = describe_case_outcome(&vacuous);
        assert!(msg.contains("NOTHING WAS COMPARED"), "{msg}");
        assert!(msg.contains("`y`"), "{msg}");

        // NEGATIVE CONTROL: one real element compared is a real pass.
        let one = CompareReport { pass: true, checked: 1, mismatches: 0, max_ulp: None, worst: None };
        let real = CaseOutcome {
            case: "n=1".into(),
            reports: vec![("y".to_string(), one)],
            reference_panic: None,
        };
        assert!(real.pass());
        assert_eq!(describe_case_outcome(&real), "n=1: pass");
    }

    /// A mixed entry — one real parameter and one empty one — must not be
    /// rescued by the real one. Every compared parameter has to have compared
    /// something.
    #[test]
    fn one_empty_parameter_sinks_a_case_with_a_real_one() {
        let good = CompareReport { pass: true, checked: 8, mismatches: 0, max_ulp: None, worst: None };
        let empty = CompareReport { pass: true, checked: 0, mismatches: 0, max_ulp: None, worst: None };
        let o = CaseOutcome {
            case: "n=8".into(),
            reports: vec![("y".to_string(), good), ("z".to_string(), empty)],
            reference_panic: None,
        };
        assert!(!o.pass());
        assert!(describe_case_outcome(&o).contains("`z`"));
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
