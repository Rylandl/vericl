//! Tamper regressions against the **real committed evidence files**.
//!
//! The unit tests in `vericl::evidence` exercise `verify` on synthetic
//! manifests. This file does the same thing to `evidence/vericl.json` and its
//! three siblings as they are actually committed — 37 entries and 71 claims of
//! real recorded evidence — because a check that only works on hand-built
//! fixtures is a check whose shape was chosen to fit the test.
//!
//! Every case is the same experiment: load the committed manifest, confirm it
//! verifies **clean against itself** (the negative control — without it, a
//! broken comparison would make every case below pass), then edit one thing in
//! a copy and confirm `verify` names it. The edits are made on the JSON, the
//! way a person editing a committed evidence file would make them, and reparsed
//! — so they also pin that the schema round-trips.
//!
//! No GPU, no z3, no kernel launch: this is pure manifest arithmetic and runs
//! in milliseconds on any machine.

use std::path::PathBuf;

use vericl::{Manifest, unrecorded_evidence, verify};

/// Every evidence file this workspace commits. `cooperative_fallback.json` is
/// the `prove: false` one and `vericl_f64.json` the shared-front-end lane, so
/// the set spans all three suite shapes.
const EVIDENCE: &[&str] = &[
    "evidence/vericl.json",
    "evidence/vericl_2d.json",
    "evidence/vericl_f64.json",
    "evidence/cooperative_fallback.json",
];

fn path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn load(rel: &str) -> Manifest {
    Manifest::load(&path(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// The committed JSON as a `serde_json::Value`, for edits that are easier to
/// express textually than through the typed API.
fn load_json(rel: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path(rel)).unwrap()).unwrap()
}

fn reparse(v: serde_json::Value) -> Manifest {
    serde_json::from_value(v).expect("a tampered manifest must still be a valid manifest")
}

/// Load `rel`, apply `tamper` to the STORED copy, and return the problems.
/// Asserts the untampered pair is clean first.
fn tamper(rel: &str, tamper: impl FnOnce(&mut serde_json::Value)) -> Vec<String> {
    let pristine = load(rel);
    assert!(
        verify(&pristine, &pristine).is_empty(),
        "NEGATIVE CONTROL BROKEN — {rel} must verify clean against itself: {:#?}",
        verify(&pristine, &pristine)
    );
    let mut json = load_json(rel);
    tamper(&mut json);
    verify(&reparse(json), &pristine)
}

fn entries(v: &mut serde_json::Value) -> &mut Vec<serde_json::Value> {
    v["entries"].as_array_mut().unwrap()
}

fn claims(v: &mut serde_json::Value, entry: usize) -> &mut Vec<serde_json::Value> {
    entries(v)[entry]["claims"].as_array_mut().unwrap()
}

fn one(problems: &[String]) -> &str {
    assert_eq!(problems.len(), 1, "expected exactly one problem: {problems:#?}");
    &problems[0]
}

// ---------------------------------------------------------------------------
// The negative control, on its own, for every committed file.
// ---------------------------------------------------------------------------

/// Each committed manifest verifies clean against itself, records a provenance
/// fingerprint, and is non-empty. Without this, every tamper case below could
/// be passing for the wrong reason.
#[test]
fn every_committed_manifest_verifies_clean_against_itself() {
    for rel in EVIDENCE {
        let m = load(rel);
        assert!(!m.entries.is_empty(), "{rel} has no entries");
        assert!(
            m.provenance.is_recorded(),
            "{rel} carries no verification-environment record — regenerate it"
        );
        assert!(!m.provenance.lanes.is_empty(), "{rel} records no execution lane");
        assert!(verify(&m, &m).is_empty(), "{rel}: {:#?}", verify(&m, &m));
        assert!(unrecorded_evidence(&m, &m).is_empty(), "{rel}");
    }
}

/// Fix 3, on the committed artifacts: EVERY entry carries an IR hash, including
/// the `prove: false` suite that used to record `ir_hash: null` because the
/// hash was computed inside the prover branch.
#[test]
fn every_committed_entry_records_an_ir_hash() {
    for rel in EVIDENCE {
        for e in load(rel).entries {
            assert!(
                e.identity.ir_hash.as_deref().is_some_and(|h| h.starts_with("sha256:")),
                "{rel}: kernel `{}` has no ir_hash — IR extraction needs no solver, so \
                 `prove: false` is not a reason to omit it",
                e.kernel
            );
        }
    }
}

// ---------------------------------------------------------------------------
// One case per tamper class the external consumer review listed.
// ---------------------------------------------------------------------------

/// CLASS 1 — a tested claim removed from the build while the file still
/// advertises it. Before this arc, `verify` checked only *proved* claims for
/// presence, so deleting a differential from the current build was invisible.
#[test]
fn a_tested_claim_the_build_no_longer_produces_is_caught() {
    let pristine = load("evidence/vericl.json");
    // "current build" = the committed evidence minus one kernel's differential.
    let mut json = load_json("evidence/vericl.json");
    let removed = claims(&mut json, 0).remove(0);
    assert_eq!(removed["kind"], "tested");
    let current = reparse(json);

    let problems = verify(&pristine, &current);
    let msg = one(&problems);
    assert!(msg.contains("kernel `axpy`"), "{msg}");
    assert!(msg.contains("tested `differential`"), "{msg}");
    assert!(msg.contains("did not produce it"), "{msg}");
}

/// CLASS 1b — the same claim deleted from the FILE. The recorded claim set is
/// not this build's claim set either way.
#[test]
fn a_tested_claim_deleted_from_the_file_is_caught() {
    let problems = tamper("evidence/vericl.json", |v| {
        claims(v, 0).remove(0);
    });
    let msg = one(&problems);
    assert!(msg.contains("tested `differential`"), "{msg}");
    assert!(msg.contains("MISSING from the stored evidence"), "{msg}");
}

/// CLASS 2 — the backend a claim was measured on is rewritten. Reported as one
/// field diff, not as an unrelated removal plus addition.
#[test]
fn a_changed_backend_is_caught() {
    let problems = tamper("evidence/vericl.json", |v| {
        claims(v, 0)[0]["backend"] = serde_json::json!("\"cuda<ptx>\"");
    });
    let msg = one(&problems);
    assert!(msg.contains("backend changed"), "{msg}");
    assert!(msg.contains("cuda<ptx>"), "{msg}");
    assert!(msg.contains("wgpu<wgsl>"), "{msg}");
}

/// CLASS 3 — the sizes the differential ran over are shortened, so the file
/// advertises coverage the run never had. The problem names the field and both
/// values.
#[test]
fn altered_sizes_are_caught_and_both_values_shown() {
    let problems = tamper("evidence/vericl.json", |v| {
        claims(v, 0)[0]["config"]["sizes"] = serde_json::json!([1, 7]);
    });
    let msg = one(&problems);
    assert!(msg.contains("config.sizes"), "{msg}");
    assert!(msg.contains("stored [1,7]"), "{msg}");
    assert!(msg.contains("current [1,7,256,1000,1027,4096,65536]"), "{msg}");
}

/// CLASS 3b — the rest of the recorded configuration: seed, cube_dim, solver,
/// obligation count, and (on the 2-D suite) the per-axis extents and rank.
#[test]
fn every_recorded_configuration_field_is_compared() {
    /// `(evidence file, the dotted path the diff must name, the edit)`.
    type Case = (&'static str, &'static str, Box<dyn Fn(&mut serde_json::Value)>);
    let cases: Vec<Case> = vec![
        (
            "evidence/vericl.json",
            "config.seed",
            Box::new(|v| claims(v, 0)[0]["config"]["seed"] = serde_json::json!(1)),
        ),
        (
            "evidence/vericl.json",
            "config.cube_dim",
            Box::new(|v| claims(v, 0)[0]["config"]["cube_dim"] = serde_json::json!(64)),
        ),
        (
            "evidence/vericl.json",
            "config.solver",
            Box::new(|v| claims(v, 0)[1]["config"]["solver"] = serde_json::json!("z3 4.8.7")),
        ),
        (
            "evidence/vericl.json",
            "config.obligations",
            Box::new(|v| claims(v, 0)[1]["config"]["obligations"] = serde_json::json!(1)),
        ),
        (
            "evidence/vericl.json",
            "config.logic",
            Box::new(|v| claims(v, 0)[1]["config"]["logic"] = serde_json::json!("QF_NIA")),
        ),
        (
            "evidence/vericl_2d.json",
            "config.rank",
            Box::new(|v| claims(v, 0)[0]["config"]["rank"] = serde_json::json!(3)),
        ),
        (
            "evidence/vericl_2d.json",
            "config.sizes[0][1]",
            Box::new(|v| claims(v, 0)[0]["config"]["sizes"][0][1] = serde_json::json!(20)),
        ),
        (
            "evidence/vericl_2d.json",
            "config.cube_dim[0]",
            Box::new(|v| claims(v, 0)[0]["config"]["cube_dim"][0] = serde_json::json!(8)),
        ),
        (
            "evidence/cooperative_fallback.json",
            "config.depends_on.status",
            Box::new(|v| {
                claims(v, 0)[0]["config"]["depends_on"]["status"] =
                    serde_json::json!("discharged-by-proof")
            }),
        ),
    ];
    for (file, field, edit) in cases {
        let problems = tamper(file, edit);
        let msg = one(&problems);
        assert!(msg.contains(field), "{file} / {field}: {msg}");
    }
}

/// CLASS 4 — a trust dependency erased. The direction that makes evidence look
/// STRONGER than it is, so it is a problem even though the file now says less.
#[test]
fn an_erased_trust_dependency_is_caught() {
    let problems = tamper("evidence/vericl.json", |v| {
        let t = entries(v)[0]["trusted"].as_array_mut().unwrap();
        let before = t.len();
        t.retain(|s| !s.as_str().unwrap().contains("GPU hardware"));
        assert_eq!(t.len(), before - 1, "the fixture must actually have this entry");
    });
    let msg = one(&problems);
    assert!(msg.contains("GPU hardware"), "{msg}");
    assert!(msg.contains("MISSING from the stored evidence"), "{msg}");
    assert!(msg.contains("stronger"), "{msg}");
}

/// CLASS 4b — the solver trust line, which is the one a reader would use to
/// judge what a `proved` claim rests on.
#[test]
fn an_erased_solver_trust_line_is_caught() {
    let problems = tamper("evidence/vericl.json", |v| {
        entries(v)[0]["trusted"]
            .as_array_mut()
            .unwrap()
            .retain(|s| !s.as_str().unwrap().contains("solver binary"));
    });
    assert!(one(&problems).contains("solver binary"), "{problems:#?}");
}

/// CLASS 5 — an arbitrary passing claim typed into the file, in both the
/// plausible (`tested`) and the maximally dishonest (`proved`) spelling.
#[test]
fn an_arbitrary_passing_claim_added_to_the_file_is_caught() {
    let problems = tamper("evidence/vericl.json", |v| {
        claims(v, 0).push(serde_json::json!({
            "kind": "tested",
            "check": "exhaustively-verified",
            "backend": "\"wgpu<wgsl>\"",
            "config": { "cases": "all of them" },
            "result": { "status": "pass" }
        }));
    });
    let msg = one(&problems);
    assert!(msg.contains("exhaustively-verified"), "{msg}");
    assert!(msg.contains("did not produce it"), "{msg}");

    let problems = tamper("evidence/vericl.json", |v| {
        claims(v, 0).push(serde_json::json!({
            "kind": "proved",
            "check": "smt-total-correctness",
            "config": { "solver": "z3 4.16.0", "obligations": 9001 },
            "result": { "status": "pass" }
        }));
    });
    let msg = one(&problems);
    assert!(msg.contains("downgraded"), "{msg}");
    assert!(msg.contains("smt-total-correctness"), "{msg}");
}

/// CLASS 5b — a whole entry invented, and a whole entry deleted.
#[test]
fn an_invented_or_deleted_entry_is_caught() {
    let problems = tamper("evidence/vericl.json", |v| {
        let mut fake = entries(v)[0].clone();
        fake["kernel"] = serde_json::json!("a_kernel_that_does_not_exist");
        entries(v).push(fake);
    });
    assert!(one(&problems).contains("no longer exists in this build"), "{problems:#?}");

    let problems = tamper("evidence/vericl.json", |v| {
        entries(v).remove(0);
    });
    assert!(one(&problems).contains("no stored evidence"), "{problems:#?}");
}

/// CLASS 6 — the recorded result flipped. `cooperative_fallback.json` carries
/// the `assumed` / `declared` race-freedom claim, which is exactly the one an
/// author would be tempted to relabel.
#[test]
fn a_flipped_result_status_is_caught() {
    let problems = tamper("evidence/cooperative_fallback.json", |v| {
        let c = claims(v, 0);
        assert_eq!(c[1]["kind"], "assumed");
        c[1]["result"] = serde_json::json!({ "status": "pass" });
    });
    let msg = one(&problems);
    assert!(msg.contains("result changed"), "{msg}");
    assert!(msg.contains("stored pass -> current declared"), "{msg}");
}

/// CLASS 7 — identity. Both hashes, independently, and the `uses(...)`
/// composition hash by extension.
#[test]
fn a_tampered_identity_is_stale() {
    for (field, label) in [("source_hash", "source_hash"), ("ir_hash", "ir_hash")] {
        let problems = tamper("evidence/vericl.json", |v| {
            entries(v)[0]["identity"][field] = serde_json::json!("sha256:0000");
        });
        let msg = one(&problems);
        assert!(msg.contains("STALE"), "{label}: {msg}");
        assert!(msg.contains(label), "{label}: {msg}");
    }

    // The pre-arc shape: `ir_hash: null`. Now that it is always populated,
    // blanking it is caught rather than being the normal state of affairs.
    let problems = tamper("evidence/vericl.json", |v| {
        entries(v)[0]["identity"]["ir_hash"] = serde_json::Value::Null;
    });
    assert!(one(&problems).contains("ir_hash <none> ->"), "{problems:#?}");
}

/// CLASS 8 — the contract record, field by field.
#[test]
fn a_tampered_contract_field_is_named() {
    for (field, value) in [
        ("compare", serde_json::json!("f32 max_ulp=1000")),
        ("wrapping", serde_json::json!(true)),
        ("assumes", serde_json::json!([])),
        ("instantiate", serde_json::json!(["F = f64"])),
        ("uses", serde_json::json!(["something"])),
    ] {
        let problems = tamper("evidence/vericl.json", |v| {
            entries(v)[0]["contract"][field] = value;
        });
        let msg = one(&problems);
        assert!(msg.contains(&format!("contract field `{field}`")), "{field}: {msg}");
    }
}

/// CLASS 9 — the verification-environment fingerprint. An evidence file from
/// another toolchain is stale-class, not silently accepted.
#[test]
fn a_tampered_provenance_record_is_stale() {
    for (field, value) in [
        ("rustc", serde_json::json!("rustc 1.70.0 (90c541806 2023-05-31)")),
        ("target", serde_json::json!("x86_64-pc-windows-msvc")),
        ("cubecl", serde_json::json!("=0.9.0")),
        ("vericl", serde_json::json!("0.0.1")),
        ("vericl_ir", serde_json::json!("0.0.1")),
        ("vericl_macros", serde_json::json!("0.0.1")),
        ("salt_scheme", serde_json::json!("fnv1a-name^splitmix-case/v2")),
        ("device", serde_json::json!("Vulkan")),
    ] {
        let problems = tamper("evidence/vericl.json", |v| {
            v["provenance"][field] = value;
        });
        let msg = one(&problems);
        assert!(msg.contains("STALE"), "{field}: {msg}");
        assert!(msg.contains("verification environment changed"), "{field}: {msg}");
    }
}

/// CLASS 9b — the fingerprint removed entirely (a pre-fingerprint file, or one
/// with the block deleted to dodge the check). The file still PARSES — the
/// schema addition is back-compatible — and does not still verify.
#[test]
fn evidence_with_the_provenance_block_removed_still_parses_and_is_refused() {
    let problems = tamper("evidence/vericl.json", |v| {
        v.as_object_mut().unwrap().remove("provenance");
    });
    let msg = one(&problems);
    assert!(msg.contains("no verification-environment record"), "{msg}");
    assert!(msg.contains("VERICL_UPDATE"), "{msg}");
}

/// CLASS 10 — a duplicated entry. Entries are keyed by kernel name, so the
/// second copy would never be looked at; a hand-edited file could hide one
/// there.
#[test]
fn a_duplicated_entry_is_refused() {
    let problems = tamper("evidence/vericl.json", |v| {
        let dup = entries(v)[0].clone();
        entries(v).push(dup);
    });
    assert!(one(&problems).contains("more than one entry for kernel"), "{problems:#?}");
}

/// FIX 8 (round-13A) — an unknown key spliced into the manifest is refused at
/// PARSE time, before `verify` ever runs, so a doctored file cannot smuggle a
/// blessing vericl never issued past a consumer that reads it. The reviewer's
/// exact injected strings: an `"audit"` line at manifest level and a
/// `"summary": "PROVED CORRECT"` inside a claim. Without `deny_unknown_fields`
/// serde parses-and-drops them, and the junk rides along in the file the human
/// actually opens.
#[test]
fn unknown_manifest_keys_are_refused_at_parse_time() {
    // The junk keys, at each schema level a consumer would trust.
    type Inject = fn(&mut serde_json::Value);
    let injections: &[(&str, Inject)] = &[
        ("audit @ manifest", |v| {
            v["audit"] = serde_json::json!("independently certified correct by a third party");
        }),
        ("summary @ manifest", |v| {
            v["summary"] = serde_json::json!("PROVED CORRECT");
        }),
        ("summary @ claim", |v| {
            claims(v, 0)[0]["summary"] = serde_json::json!("PROVED CORRECT");
        }),
        ("audit @ entry", |v| {
            entries(v)[0]["audit"] = serde_json::json!("independently certified");
        }),
        ("blessing @ provenance", |v| {
            v["provenance"]["blessing"] = serde_json::json!("shipped by vericl");
        }),
        ("badge @ identity", |v| {
            entries(v)[0]["identity"]["badge"] = serde_json::json!("gold");
        }),
        ("note @ contract", |v| {
            entries(v)[0]["contract"]["note"] = serde_json::json!("hand-blessed");
        }),
    ];
    for (label, inject) in injections {
        // NEGATIVE CONTROL: the untouched file parses.
        let mut json = load_json("evidence/vericl.json");
        assert!(
            serde_json::from_value::<Manifest>(json.clone()).is_ok(),
            "{label}: the pristine file must still parse"
        );
        inject(&mut json);
        let err = serde_json::from_value::<Manifest>(json)
            .expect_err(&format!("{label}: an unknown key must be refused, not parsed-and-dropped"));
        assert!(
            err.to_string().contains("unknown field"),
            "{label}: the refusal must name the unknown field: {err}"
        );
    }
}

// ---------------------------------------------------------------------------
// Normalization: the deliberate order choices, on the real files.
// ---------------------------------------------------------------------------

/// Meaningless order is meaningless: reversing the entry list, each entry's
/// claim list, and each entry's trusted list — and duplicating a trusted entry
/// — must all still verify clean.
#[test]
fn meaningless_order_and_duplication_do_not_make_evidence_stale() {
    for rel in EVIDENCE {
        let pristine = load(rel);
        let mut json = load_json(rel);
        entries(&mut json).reverse();
        for e in entries(&mut json) {
            e["claims"].as_array_mut().unwrap().reverse();
            let t = e["trusted"].as_array_mut().unwrap();
            t.reverse();
            let first = t[0].clone();
            t.push(first);
        }
        let shuffled = reparse(json);
        assert!(verify(&shuffled, &pristine).is_empty(), "{rel}: {:#?}", verify(&shuffled, &pristine));
    }
}

/// Meaningful order is meaningful: a reordered `sizes` list is a different
/// declaration and re-stales the evidence. Sensitivity here can only ask for a
/// regeneration, never let a real change through.
#[test]
fn a_reordered_sizes_list_is_significant() {
    let problems = tamper("evidence/vericl.json", |v| {
        let s = claims(v, 0)[0]["config"]["sizes"].as_array_mut().unwrap();
        s.reverse();
    });
    assert!(!problems.is_empty(), "reordering `sizes` must be visible");
    assert!(problems.iter().all(|m| m.contains("config.sizes[")), "{problems:#?}");
}

/// Whitespace and key order in the JSON are not content: a manifest re-emitted
/// with its object keys in a different textual order verifies clean.
#[test]
fn json_object_key_order_is_not_content() {
    for rel in EVIDENCE {
        let pristine = load(rel);
        // A compact, differently-ordered re-emission of the same content.
        let compact = serde_json::to_string(&pristine).unwrap();
        let round_tripped: Manifest = serde_json::from_str(&compact).unwrap();
        assert!(verify(&round_tripped, &pristine).is_empty(), "{rel}");
    }
}

// ---------------------------------------------------------------------------
// The one exemption, on the real files (round-13A fixes 1+2). The committed
// conformance manifest is the SUPERSET of lanes: it records the cpu `extra_lane`
// (produced under `--features cpu`), so a default `cargo test` verifies the
// wgpu lane and NOTES the recorded-but-unexercised cpu lane, while `--features
// cpu` verifies both. The two directions are not symmetric: a lane the file
// records but this run skipped is a note; a lane this run produced but the file
// omits is a problem (the strip-detection).
// ---------------------------------------------------------------------------

/// The non-primary (cpu) lane the committed conformance manifest records under
/// `--features cpu`. Panics with an actionable message if the file was not
/// regenerated to record it — fixes 1+2 RECORD the cpu lane rather than
/// exempting-and-hiding it, so its absence is a real problem, not a test skip.
fn committed_cpu_lane() -> String {
    let m = load("evidence/vericl.json");
    m.provenance.lanes.get(1).cloned().unwrap_or_else(|| {
        panic!(
            "evidence/vericl.json records only {:?} — fixes 1+2 require the cpu lane to be RECORDED \
             (regenerate: VERICL_UPDATE=1 cargo test --features cpu)",
            m.provenance.lanes
        )
    })
}

/// Remove the cpu lane, its per-entry claims, and its per-entry trust from a
/// manifest JSON — either a wgpu-only "current run" derived from the committed
/// file, or a "stripped file".
fn without_cpu_lane(json: &mut serde_json::Value, cpu: &str) {
    json["provenance"]["lanes"].as_array_mut().unwrap().retain(|l| l.as_str() != Some(cpu));
    for e in entries(json) {
        e["claims"].as_array_mut().unwrap().retain(|c| c["backend"].as_str() != Some(cpu));
        e["trusted"].as_array_mut().unwrap().retain(|t| {
            let s = t.as_str().unwrap();
            !(s == cpu || s.starts_with(&format!("{cpu} ")))
        });
    }
}

/// SKIPPED-lane direction, on the committed file: a default `cargo test` (wgpu
/// only) against the manifest committed under `--features cpu`. The cpu lane it
/// records is NOT re-verified here — a NOTE, never a problem, because the lane
/// is strictly verified under its own feature.
#[test]
fn a_skipped_cpu_lane_is_a_note_on_the_committed_file() {
    let cpu = committed_cpu_lane();
    let stored = load("evidence/vericl.json"); // records wgpu + cpu
    let mut json = load_json("evidence/vericl.json");
    without_cpu_lane(&mut json, &cpu);
    let wgpu_only_run = reparse(json);

    let problems = verify(&stored, &wgpu_only_run);
    assert!(problems.is_empty(), "a skipped cpu lane must not be a problem: {problems:#?}");
    let notes = unrecorded_evidence(&stored, &wgpu_only_run);
    assert!(
        notes.iter().any(|n| n.contains("did not exercise") && n.contains(&cpu)),
        "the unexercised cpu lane must be a note: {notes:#?}"
    );
    assert!(
        notes.iter().any(|n| n.contains(&format!("(backend {cpu})"))),
        "the skipped cpu claim(s) must be notes: {notes:#?}"
    );
}

/// STRIPPED-lane direction, on the committed file: delete the cpu lane + its
/// claims + its trust and run `--features cpu` against the stripped file. The
/// cpu evidence the build produces is MISSING — a problem SET, not the pre-fix
/// `0 problems, all notes` the reviewer demonstrated (fixes 1+2).
#[test]
fn a_stripped_cpu_lane_is_a_problem_on_the_committed_file() {
    let cpu = committed_cpu_lane();
    let full_run = load("evidence/vericl.json"); // the --features cpu build
    let mut json = load_json("evidence/vericl.json");
    without_cpu_lane(&mut json, &cpu);
    let stripped_file = reparse(json);

    let problems = verify(&stripped_file, &full_run);
    assert!(
        problems.iter().any(|m| m.contains(&format!("execution lane {cpu} ran")) && m.contains("NOT recorded")),
        "the stripped lane must be a problem: {problems:#?}"
    );
    assert!(
        problems.iter().any(|m| m.contains(&format!("(backend {cpu})"))
            && m.contains("MISSING from the stored evidence")),
        "the cpu claim(s) must be problems: {problems:#?}"
    );
    assert!(
        problems.iter().any(|m| m.contains("stronger")),
        "the cpu trust must be a problem: {problems:#?}"
    );
    assert!(
        unrecorded_evidence(&stripped_file, &full_run).is_empty(),
        "nothing may be exempted in the strip direction"
    );
}

/// ADVERSARIAL-REVIEW REGRESSION (round-13 pre-review, filed CRITICAL), on the
/// real committed manifest and in the reviewer's own words: *"delete a `Tested`
/// claim from the committed evidence file and have `verify()` report zero
/// problems, simply by also removing the corresponding string from
/// `stored.provenance.lanes`."*
///
/// The lane list scopes the (now stored-only) exemption, and the attacker
/// controls it. Deleting the wgpu tested claim and editing `provenance.lanes`
/// so its backend looks "unrecorded" must still be refused: the empty-lane guard
/// (an empty list cannot make a build's lanes verify against nothing) and the
/// primary-lane guard (a plausible `["cpu"]` list changes the PRIMARY, itself a
/// problem) both close it. Two edits, one green run — against the code as first
/// written; caught here.
#[test]
fn deleting_a_claim_and_its_lane_marker_together_is_still_refused() {
    let pristine = load("evidence/vericl.json");

    for lanes in [serde_json::json!([]), serde_json::json!(["\"cpu\""])] {
        let mut json = load_json("evidence/vericl.json");
        let removed = claims(&mut json, 0).remove(0);
        assert_eq!(removed["kind"], "tested", "the fixture must remove the differential claim");
        json["provenance"]["lanes"] = lanes.clone();
        let stored = reparse(json);

        let problems = verify(&stored, &pristine);
        assert!(
            problems.iter().any(|m| m.contains("MISSING from the stored evidence")),
            "lanes {lanes}: the deleted tested claim must still be reported: {problems:#?}"
        );
        assert!(
            unrecorded_evidence(&stored, &pristine)
                .iter()
                .all(|n| !n.contains("tested `differential`")),
            "lanes {lanes}: the deletion must not be downgraded to a note"
        );
    }
}

/// The same attack aimed at `trusted` rather than at a claim: the lane-scoped
/// trust strings (`backend_buffer_trust`, `shared_frontend_lane_trust`) are
/// formatted as `"{backend} …"`, so they match the exemption's `starts_with`
/// test by construction. Erasing one while deleting the lane marker must not
/// downgrade it either.
#[test]
fn erasing_a_lane_scoped_trust_entry_with_its_lane_marker_is_still_refused() {
    let pristine = load("evidence/vericl.json");
    let mut json = load_json("evidence/vericl.json");
    let t = entries(&mut json)[0]["trusted"].as_array_mut().unwrap();
    let before = t.len();
    t.retain(|s| !s.as_str().unwrap().contains("buffer upload/readback"));
    assert_eq!(t.len(), before - 1, "the fixture must actually have this entry");
    json["provenance"]["lanes"] = serde_json::json!([]);
    let stored = reparse(json);

    let problems = verify(&stored, &pristine);
    assert!(
        problems.iter().any(|m| m.contains("buffer upload/readback") && m.contains("stronger")),
        "{problems:#?}"
    );
}

/// A tested claim on a lane the file DOES record that this run exercised but
/// did NOT produce is still a loss (not the skipped-lane note): the recorded
/// claim set is not this build's. Here the committed cpu lane runs but the
/// stored file carries an *extra* cpu claim the build never emitted.
#[test]
fn a_recorded_lane_claim_the_build_did_not_emit_is_a_problem() {
    let cpu = committed_cpu_lane();
    let current = load("evidence/vericl.json"); // the --features cpu build
    let mut json = load_json("evidence/vericl.json");
    // Append a SECOND cpu claim to entry 0 — a claim on the recorded cpu lane
    // that this build does not produce.
    let extra = {
        let mut c = claims(&mut json, 0)[0].clone();
        c["backend"] = serde_json::json!(cpu);
        c["check"] = serde_json::json!("hand-added-extra");
        c
    };
    claims(&mut json, 0).push(extra);
    let stored = reparse(json);

    let problems = verify(&stored, &current);
    assert!(
        problems.iter().any(|m| m.contains("hand-added-extra") && m.contains("did not produce it")),
        "an extra claim on a recorded, exercised lane must be a problem: {problems:#?}"
    );
    // It is NOT downgraded to a note (the lane ran, so it is not "skipped").
    assert!(
        unrecorded_evidence(&stored, &current)
            .iter()
            .all(|n| !n.contains("hand-added-extra")),
        "the extra claim must not be exempted"
    );
}
