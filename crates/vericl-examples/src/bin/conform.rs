//! Conformance harness: differential-tests the example kernels (GPU vs the
//! macro-derived sequential reference) and maintains the evidence manifest.
//!
//! Usage:
//!   conform update        regenerate evidence/vericl.json from a fresh run
//!   conform check         verify stored evidence against the current build
//!                         (stale identity, missing entries, or failing
//!                         checks are hard errors)
//!   conform demo-defects  run the deliberately defective kernels and show
//!                         that the checks catch them (exits 0 iff caught)
//!
//! With `--features cpu`, a second differential lane runs on
//! `cubecl::cpu::CpuRuntime` and appends an additional `Tested` claim to
//! each good kernel's evidence entry. That lane shares CubeCL's front end
//! (macro expansion + IR) with the kernel under test, so it is recorded as
//! *not independent* in the entry's `trusted` list — the macro-generated
//! sequential twin remains the only independent reference.
//!
//! v0 caveat: `check` verifies whichever lanes the *current* build produces
//! against the union of claims already on file; it does not diff the claim
//! *sets*. So `check` run with fewer/more `--features` than the `update`
//! that wrote the evidence will not detect the mismatch (a missing lane is
//! not reported as missing). `update` and `check` must be run with the same
//! `--features` for the evidence to mean what it says.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use cubecl::prelude::*;
use vericl::{
    CaseOutcome, Claim, ClaimKind, ClaimResult, Compare, CompareReport, Contract, Entry, Manifest,
    SplitMix64, StructuredAssume,
};
use vericl_examples::*;

type R = cubecl::wgpu::WgpuRuntime;

const CUBE_DIM: u32 = 256;
const SIZES: &[usize] = &[1, 7, 256, 1000, 1027, 4096, 65536];
const SEED: u64 = 0xE9_01;

fn dispatch(n: usize) -> (CubeCount, CubeDim, usize) {
    let count = (n as u32).div_ceil(CUBE_DIM).max(1);
    (
        CubeCount::Static(count, 1, 1),
        CubeDim::new_1d(CUBE_DIM),
        (count * CUBE_DIM) as usize,
    )
}

/// Run the sequential reference, catching panics (an out-of-bounds access in
/// the reference is a *finding*, not a harness crash).
fn run_reference(f: impl FnOnce()) -> Result<(), String> {
    // A panicking reference is a reported finding; keep the default hook's
    // stderr noise out of the report.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    result.map_err(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "reference panicked".to_string())
    })
}

fn compare_outputs(compare: Compare, expected_f32: Option<(&[f32], &[f32])>, expected_u32: Option<(&[u32], &[u32])>) -> CompareReport {
    match (compare, expected_f32, expected_u32) {
        (Compare::MaxUlpF32(max), Some((e, a)), _) => vericl::compare_f32(e, a, max),
        (Compare::AbsRelF32 { abs, rel }, Some((e, a)), _) => {
            vericl::compare_f32_absrel(e, a, abs, rel)
        }
        (Compare::Exact, _, Some((e, a))) => vericl::compare_exact_u32(e, a),
        _ => panic!("comparison mode does not match buffer types"),
    }
}

/// The SMT out-of-bounds-freedom proof for one kernel: the IR content hash
/// (for `Identity::ir_hash`), the `Proved` claim ready to fold into an
/// evidence entry, and the raw `vericl_ir::ProveResult` for pretty-printing
/// (demo-defects wants the counterexample text; the main flow just needs
/// pass/fail).
struct ProofOutcome {
    ir_hash: String,
    solver: String,
    claim: Claim,
    result: vericl_ir::ProveResult,
}

/// Run vericl-ir's SMT bounds prover over one kernel's IR. This is the same
/// call for every kernel — good or defective — so it's shared rather than
/// duplicated per call site; only what's done with the result differs
/// (folded into an evidence entry for the three good kernels, printed
/// standalone in `demo-defects`).
fn prove_kernel(
    def: &KernelDefinition,
    buffer_params: &[(&str, bool)],
    structured_assumes: &[StructuredAssume],
) -> ProofOutcome {
    let ir_hash = vericl_ir::kernel_ir_hash(def);
    let buffers: Vec<vericl_ir::BufferParam> = buffer_params
        .iter()
        .map(|&(name, is_output)| vericl_ir::BufferParam { name, is_output })
        .collect();
    let assumes: Vec<vericl_ir::Assume> = structured_assumes
        .iter()
        .map(|a| match *a {
            StructuredAssume::LenEq { a, b } => vericl_ir::Assume::LenEq { a, b },
            StructuredAssume::LenEqConst { a, value } => vericl_ir::Assume::LenEqConst { a, value },
        })
        .collect();
    let solver = vericl_ir::z3_version()
        .map(|v| format!("z3 {v}"))
        .unwrap_or_else(|| "z3 (version unavailable — is z3 on PATH?)".to_string());

    let result = vericl_ir::prove_bounds_freedom(def, &buffers, &assumes);
    let (obligations, claim_result) = match &result {
        vericl_ir::ProveResult::Proved { obligations } => (*obligations, ClaimResult::Pass),
        vericl_ir::ProveResult::Refuted { obligation, counterexample } => (
            0,
            ClaimResult::Fail {
                detail: format!("REFUTED: {obligation} — counterexample: {counterexample}"),
            },
        ),
        vericl_ir::ProveResult::OutOfSubset { reason } => {
            (0, ClaimResult::Fail { detail: format!("outside the vericl v0 subset: {reason}") })
        }
        vericl_ir::ProveResult::SolverError { detail } => {
            (0, ClaimResult::Fail { detail: format!("solver error: {detail}") })
        }
    };

    let claim = Claim {
        kind: ClaimKind::Proved,
        check: "smt-oob-freedom".into(),
        backend: None,
        config: serde_json::json!({
            "solver": solver,
            "logic": "QF_LIA",
            "obligations": obligations,
        }),
        result: claim_result,
    };

    ProofOutcome { ir_hash, solver, claim, result }
}

struct KernelRun {
    contract: Contract,
    outcomes: Vec<CaseOutcome>,
    /// The SMT bounds proof for this kernel, when the caller ran one (only
    /// the three good kernels in the main update/check flow — the
    /// defective kernels are proved separately in `demo-defects`, printed
    /// standalone rather than folded into a persisted entry).
    proof: Option<ProofOutcome>,
}

impl KernelRun {
    fn passed(&self) -> bool {
        self.outcomes
            .iter()
            .all(|o| o.reference_panic.is_none() && o.report.as_ref().is_some_and(|r| r.pass))
    }

    /// Build this run's result into a `Tested` claim for `backend`. Shared
    /// by every lane (wgpu, and cpu when the `cpu` feature is on) so a
    /// kernel's evidence entry can carry one claim per lane it was run on.
    fn claim(&self, backend: &str) -> Claim {
        let detail = self
            .outcomes
            .iter()
            .filter(|o| o.reference_panic.is_some() || o.report.as_ref().is_none_or(|r| !r.pass))
            .map(describe_outcome)
            .collect::<Vec<_>>()
            .join("; ");
        let result = if self.passed() {
            ClaimResult::Pass
        } else {
            ClaimResult::Fail { detail }
        };
        Claim {
            kind: ClaimKind::Tested,
            check: "differential".into(),
            backend: Some(backend.to_string()),
            config: serde_json::json!({
                "sizes": SIZES,
                "seed": SEED,
                "cube_dim": CUBE_DIM,
                "reference": "vericl-macros sequential twin",
            }),
            result,
        }
    }

    fn into_entry(self, backend: &str) -> Entry {
        let claim = self.claim(backend);
        let mut identity = self.contract.identity();
        let mut claims = vec![claim];
        let mut trusted = vec![
            "rustc codegen of the reference twin".into(),
            "vericl-macros source-to-reference derivation".into(),
            "wgpu buffer upload/readback integrity".into(),
            "GPU hardware".into(),
        ];
        if let Some(proof) = self.proof {
            identity.ir_hash = Some(proof.ir_hash);
            claims.push(proof.claim);
            trusted.push(format!("the solver binary ({}) discharging the SMT bounds obligations", proof.solver));
            trusted.push(
                "vericl-ir's obligation encoding (0 <= index < Length(array) in QF_LIA over the \
                 CubeCL IR)"
                    .into(),
            );
            trusted.push(
                "cubecl front-end expansion (the proof is about the IR; codegen below the IR \
                 remains covered only by the tested differential claims)"
                    .into(),
            );
        }
        Entry {
            kernel: self.contract.kernel.to_string(),
            identity,
            contract: self.contract.record(),
            claims,
            trusted,
        }
    }
}

fn describe_outcome(o: &CaseOutcome) -> String {
    if let Some(p) = &o.reference_panic {
        // Only an "index out of bounds" panic is the WGSL-robustness story
        // (a GPU backend would silently clamp an out-of-bounds access that
        // panics sequentially). Any other panic (e.g. `wrapping`'s reference
        // twin still dividing by zero) is a divergent-semantics/defect
        // finding of a different kind and must not be mislabeled as a
        // bounds issue.
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
    match &o.report {
        Some(r) if !r.pass => {
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
                "{}: {}/{} elements diverge from reference{}",
                o.case, r.mismatches, r.checked, worst
            )
        }
        _ => format!("{}: pass", o.case),
    }
}

// ---------------------------------------------------------------------------
// Per-kernel differential runs (GPU launch signatures are kernel-specific).
// ---------------------------------------------------------------------------

fn gpu_f32_axpy_like<R: Runtime, L>(client: &ComputeClient<R>, launch: L, alpha: f32, x: &[f32], y: &[f32], n: usize) -> Vec<f32>
where
    L: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, f32, ArrayArg<R>, ArrayArg<R>),
{
    let (count, dim, _) = dispatch(n);
    let x_h = client.create_from_slice(f32::as_bytes(x));
    let y_h = client.create_from_slice(f32::as_bytes(y));
    launch(
        client,
        count,
        dim,
        alpha,
        unsafe { ArrayArg::from_raw_parts(x_h, x.len()) },
        unsafe { ArrayArg::from_raw_parts(y_h.clone(), y.len()) },
    );
    f32::from_bytes(&client.read_one(y_h).unwrap()).to_vec()
}

fn run_axpy<R: Runtime>(client: &ComputeClient<R>) -> KernelRun {
    let mut rng = SplitMix64::new(SEED);
    let mut outcomes = Vec::new();
    for &n in SIZES {
        let alpha = rng.next_f32_range(-4.0, 4.0);
        let x = rng.fill_f32(n, -100.0, 100.0);
        let y0 = rng.fill_f32(n, -100.0, 100.0);
        assert!(axpy_vericl::check_assumes(alpha, &x, &y0));

        let (_, _, threads) = dispatch(n);
        let mut y_ref = y0.clone();
        let panicked = run_reference(|| axpy_vericl::reference(alpha, &x, &mut y_ref, threads));

        let outcome = match panicked {
            Err(p) => CaseOutcome { case: format!("n={n}"), report: None, reference_panic: Some(p) },
            Ok(()) => {
                let y_gpu = gpu_f32_axpy_like(
                    client,
                    axpy::launch::<R>,
                    alpha, &x, &y0, n,
                );
                CaseOutcome {
                    case: format!("n={n}"),
                    report: Some(compare_outputs(
                        axpy_vericl::contract().compare,
                        Some((&y_ref, &y_gpu)),
                        None,
                    )),
                    reference_panic: None,
                }
            }
        };
        outcomes.push(outcome);
    }
    let proof = prove_kernel(
        &axpy_vericl::kernel_definition(),
        axpy_vericl::BUFFER_PARAMS,
        axpy_vericl::contract().structured_assumes,
    );
    KernelRun { contract: axpy_vericl::contract(), outcomes, proof: Some(proof) }
}

fn run_axpy_off_by_one<R: Runtime>(client: &ComputeClient<R>) -> KernelRun {
    let mut rng = SplitMix64::new(SEED ^ 0xDEF0);
    let mut outcomes = Vec::new();
    for &n in SIZES {
        let alpha = rng.next_f32_range(-4.0, 4.0);
        let x = rng.fill_f32(n, -100.0, 100.0);
        let y0 = rng.fill_f32(n, -100.0, 100.0);

        let (_, _, threads) = dispatch(n);
        let mut y_ref = y0.clone();
        let panicked =
            run_reference(|| axpy_off_by_one_vericl::reference(alpha, &x, &mut y_ref, threads));

        let outcome = match panicked {
            Err(p) => CaseOutcome { case: format!("n={n}"), report: None, reference_panic: Some(p) },
            Ok(()) => {
                let y_gpu = gpu_f32_axpy_like(
                    client,
                    axpy_off_by_one::launch::<R>,
                    alpha, &x, &y0, n,
                );
                CaseOutcome {
                    case: format!("n={n}"),
                    report: Some(compare_outputs(
                        axpy_off_by_one_vericl::contract().compare,
                        Some((&y_ref, &y_gpu)),
                        None,
                    )),
                    reference_panic: None,
                }
            }
        };
        outcomes.push(outcome);
    }
    KernelRun { contract: axpy_off_by_one_vericl::contract(), outcomes, proof: None }
}

fn run_xorshift<R: Runtime>(client: &ComputeClient<R>) -> KernelRun {
    let mut rng = SplitMix64::new(SEED ^ 0xBEEF);
    let mut outcomes = Vec::new();
    for &n in SIZES {
        let x = rng.fill_u32(n);
        let y0 = vec![0u32; n];
        assert!(xorshift_step_vericl::check_assumes(&x, &y0));

        let (count, dim, threads) = dispatch(n);
        let mut y_ref = y0.clone();
        let panicked = run_reference(|| xorshift_step_vericl::reference(&x, &mut y_ref, threads));

        let outcome = match panicked {
            Err(p) => CaseOutcome { case: format!("n={n}"), report: None, reference_panic: Some(p) },
            Ok(()) => {
                let x_h = client.create_from_slice(u32::as_bytes(&x));
                let y_h = client.create_from_slice(u32::as_bytes(&y0));
                xorshift_step::launch::<R>(
                    client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(x_h, n) },
                    unsafe { ArrayArg::from_raw_parts(y_h.clone(), n) },
                );
                let y_gpu = u32::from_bytes(&client.read_one(y_h).unwrap()).to_vec();
                CaseOutcome {
                    case: format!("n={n}"),
                    report: Some(compare_outputs(
                        xorshift_step_vericl::contract().compare,
                        None,
                        Some((&y_ref, &y_gpu)),
                    )),
                    reference_panic: None,
                }
            }
        };
        outcomes.push(outcome);
    }
    let proof = prove_kernel(
        &xorshift_step_vericl::kernel_definition(),
        xorshift_step_vericl::BUFFER_PARAMS,
        xorshift_step_vericl::contract().structured_assumes,
    );
    KernelRun { contract: xorshift_step_vericl::contract(), outcomes, proof: Some(proof) }
}

fn run_mix_u32<R: Runtime>(client: &ComputeClient<R>) -> KernelRun {
    let mut rng = SplitMix64::new(SEED ^ 0x1357);
    let mut outcomes = Vec::new();
    for &n in SIZES {
        let x = rng.fill_u32(n);
        let y0 = vec![0u32; n];
        assert!(mix_u32_vericl::check_assumes(&x, &y0));

        let (count, dim, threads) = dispatch(n);
        let mut y_ref = y0.clone();
        let panicked = run_reference(|| mix_u32_vericl::reference(&x, &mut y_ref, threads));

        let outcome = match panicked {
            Err(p) => CaseOutcome { case: format!("n={n}"), report: None, reference_panic: Some(p) },
            Ok(()) => {
                let x_h = client.create_from_slice(u32::as_bytes(&x));
                let y_h = client.create_from_slice(u32::as_bytes(&y0));
                mix_u32::launch::<R>(
                    client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(x_h, n) },
                    unsafe { ArrayArg::from_raw_parts(y_h.clone(), n) },
                );
                let y_gpu = u32::from_bytes(&client.read_one(y_h).unwrap()).to_vec();
                CaseOutcome {
                    case: format!("n={n}"),
                    report: Some(compare_outputs(
                        mix_u32_vericl::contract().compare,
                        None,
                        Some((&y_ref, &y_gpu)),
                    )),
                    reference_panic: None,
                }
            }
        };
        outcomes.push(outcome);
    }
    let proof = prove_kernel(
        &mix_u32_vericl::kernel_definition(),
        mix_u32_vericl::BUFFER_PARAMS,
        mix_u32_vericl::contract().structured_assumes,
    );
    KernelRun { contract: mix_u32_vericl::contract(), outcomes, proof: Some(proof) }
}

fn run_sum_racy<R: Runtime>(client: &ComputeClient<R>) -> KernelRun {
    let mut rng = SplitMix64::new(SEED ^ 0xACE);
    let mut outcomes = Vec::new();
    for &n in &[4096usize, 65536] {
        let x = rng.fill_f32(n, 0.5, 1.5);
        let y0 = vec![0f32];
        assert!(sum_racy_vericl::check_assumes(&x, &y0));

        let (count, dim, threads) = dispatch(n);
        let mut y_ref = y0.clone();
        let panicked = run_reference(|| sum_racy_vericl::reference(&x, &mut y_ref, threads));

        let outcome = match panicked {
            Err(p) => CaseOutcome { case: format!("n={n}"), report: None, reference_panic: Some(p) },
            Ok(()) => {
                let x_h = client.create_from_slice(f32::as_bytes(&x));
                let y_h = client.create_from_slice(f32::as_bytes(&y0));
                sum_racy::launch::<R>(
                    client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(x_h, n) },
                    unsafe { ArrayArg::from_raw_parts(y_h.clone(), 1) },
                );
                let y_gpu = f32::from_bytes(&client.read_one(y_h).unwrap()).to_vec();
                CaseOutcome {
                    case: format!("n={n}"),
                    report: Some(compare_outputs(
                        sum_racy_vericl::contract().compare,
                        Some((&y_ref, &y_gpu)),
                        None,
                    )),
                    reference_panic: None,
                }
            }
        };
        outcomes.push(outcome);
    }
    KernelRun { contract: sum_racy_vericl::contract(), outcomes, proof: None }
}

// ---------------------------------------------------------------------------

fn manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../evidence/vericl.json")
}

fn print_run(run: &KernelRun) {
    let status = if run.passed() { "PASS" } else { "FAIL" };
    println!("  [{status}] {} ({})", run.contract.kernel, run.contract.compare.describe());
    for o in &run.outcomes {
        println!("      {}", describe_outcome(o));
    }
}

/// Run the good kernels on the CubeCL CPU runtime and fold each result into
/// the matching wgpu-lane entry as an additional `Tested` claim.
///
/// Trust accounting: unlike the wgpu lane, this lane does not add
/// independent evidence about kernel *semantics* — `CpuRuntime` and the
/// subject under test both go through CubeCL's own front end (macro
/// expansion into CubeCL IR), so a front-end bug would reproduce identically
/// on both. It is genuinely useful for catching *backend-specific*
/// divergence (e.g. wgpu-only codegen bugs), but it is not a second
/// independent reference — only the vericl-macros sequential twin, which
/// shares no CubeCL machinery, is that.
#[cfg(feature = "cpu")]
fn add_cpu_lane(entries: &mut [Entry]) {
    type CpuR = cubecl::cpu::CpuRuntime;

    let device = Default::default();
    let client = CpuR::client(&device);
    let backend = format!("{:?}", CpuR::name(&client));
    println!("vericl conformance — additional lane, backend {backend}");
    let runs = vec![run_axpy(&client), run_xorshift(&client), run_mix_u32(&client)];
    for run in &runs {
        print_run(run);
    }
    for run in &runs {
        let Some(entry) = entries.iter_mut().find(|e| e.kernel == run.contract.kernel) else {
            continue;
        };
        entry.claims.push(run.claim(&backend));
        entry.trusted.push(format!(
            "CPU runtime ({backend}) shares CubeCL's front end (macro expansion + IR) with the \
             kernel under test — this lane is NOT an independent reference; only the \
             vericl-macros sequential twin is independent of CubeCL"
        ));
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "check".into());
    let device = Default::default();
    let client = R::client(&device);
    let backend = format!("{:?}", R::name(&client));

    match mode.as_str() {
        "update" | "check" => {
            println!("vericl conformance — backend {backend}");
            let runs = vec![run_axpy(&client), run_xorshift(&client), run_mix_u32(&client)];
            for run in &runs {
                print_run(run);
            }
            // `mut` is only exercised when the cpu lane appends claims below.
            #[cfg_attr(not(feature = "cpu"), allow(unused_mut))]
            let mut entries: Vec<Entry> =
                runs.into_iter().map(|r| r.into_entry(&backend)).collect();

            #[cfg(feature = "cpu")]
            add_cpu_lane(&mut entries);

            let current = Manifest::new(entries);

            if mode == "update" {
                if current.entries.iter().any(|e| {
                    e.claims
                        .iter()
                        .any(|c| matches!(c.result, ClaimResult::Fail { .. }))
                }) {
                    eprintln!("refusing to store failing evidence");
                    std::process::exit(1);
                }
                let path = manifest_path();
                current.save(&path).expect("write manifest");
                println!("evidence written to {}", path.display());
            } else {
                let stored = match Manifest::load(&manifest_path()) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("no stored evidence ({e}); run `conform update` first");
                        std::process::exit(1);
                    }
                };
                let problems = vericl::verify(&stored, &current);
                if problems.is_empty() {
                    println!("evidence OK: identities match, all claims pass");
                } else {
                    for p in &problems {
                        eprintln!("PROBLEM: {p}");
                    }
                    std::process::exit(1);
                }
            }
        }
        "demo-defects" => {
            println!("vericl defect demo — backend {backend}");
            println!("(each kernel below is deliberately defective; the check must FAIL)\n");
            let runs = vec![run_axpy_off_by_one(&client), run_sum_racy(&client)];
            let mut all_caught = true;
            for run in &runs {
                print_run(run);
                if run.passed() {
                    eprintln!(
                        "  !! defect in `{}` was NOT caught — this is a vericl failure",
                        run.contract.kernel
                    );
                    all_caught = false;
                } else {
                    println!("  -> defect caught");
                }
            }

            // The SMT bounds proof is a separate claim from the differential
            // checks above (proved vs. tested — see README "Claims and
            // trust boundaries"), so it is deliberately not folded into
            // `run.passed()`/`all_caught` the same way; each kernel's
            // outcome is printed and reasoned about on its own terms.
            println!(
                "\nSMT bounds proofs (a separate claim from the differential checks above):"
            );

            let ob_proof = prove_kernel(
                &axpy_off_by_one_vericl::kernel_definition(),
                axpy_off_by_one_vericl::BUFFER_PARAMS,
                axpy_off_by_one_vericl::contract().structured_assumes,
            );
            match &ob_proof.result {
                vericl_ir::ProveResult::Refuted { obligation, counterexample } => {
                    println!("  [REFUTED] axpy_off_by_one: {obligation}");
                    println!("    counterexample: {counterexample}");
                    println!("  -> bounds defect caught (an out-of-bounds access is reachable)");
                }
                other => {
                    eprintln!(
                        "  !! axpy_off_by_one's bounds check did NOT refute ({other:?}) — this \
                         is a vericl failure"
                    );
                    all_caught = false;
                }
            }

            let racy_proof = prove_kernel(
                &sum_racy_vericl::kernel_definition(),
                sum_racy_vericl::BUFFER_PARAMS,
                sum_racy_vericl::contract().structured_assumes,
            );
            match &racy_proof.result {
                vericl_ir::ProveResult::Proved { obligations } => {
                    println!(
                        "  [PROVED] sum_racy: {obligations} obligation(s) — the race above is a \
                         differential finding; sum_racy's array accesses are not out of bounds"
                    );
                }
                other => {
                    eprintln!(
                        "  !! sum_racy's bounds check did not prove ({other:?}) — this is a \
                         vericl failure"
                    );
                    all_caught = false;
                }
            }

            std::process::exit(if all_caught { 0 } else { 1 });
        }
        other => {
            eprintln!("unknown mode `{other}`; use update | check | demo-defects");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REGRESSION (adversarial soundness review, Bug 2 — cosmetic): only an
    /// "index out of bounds" reference panic gets the WGSL-robustness/bounds
    /// narrative; any other panic (e.g. `wrapping`'s reference twin still
    /// dividing by zero) must be reported neutrally instead of being
    /// mislabeled as a bounds defect.
    #[test]
    fn describe_outcome_labels_oob_panic_with_bounds_story() {
        let o = CaseOutcome {
            case: "n=4".into(),
            report: None,
            reference_panic: Some("index out of bounds: the len is 4 but the index is 4".into()),
        };
        let msg = describe_outcome(&o);
        assert!(msg.contains("GPU backends (WGSL robustness) would silently clamp this"), "{msg}");
        assert!(msg.contains("index out of bounds"), "{msg}");
    }

    /// A non-bounds panic (division by zero, the motivating `wrapping`-clause
    /// case from the review) must NOT get the bounds/WGSL-robustness
    /// narrative — that would misattribute the cause.
    #[test]
    fn describe_outcome_labels_non_oob_panic_neutrally() {
        let o = CaseOutcome {
            case: "n=4".into(),
            report: None,
            reference_panic: Some("attempt to divide by zero".into()),
        };
        let msg = describe_outcome(&o);
        assert!(!msg.contains("WGSL robustness"), "{msg}");
        assert!(!msg.contains("accesses outside its declared"), "{msg}");
        assert!(msg.contains("divergent semantics or defect"), "{msg}");
        assert!(msg.contains("attempt to divide by zero"), "{msg}");
    }
}
