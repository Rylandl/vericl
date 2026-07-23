//! Demo-defects binary: runs the deliberately defective example kernels
//! (`axpy_off_by_one`, `sum_racy`) and shows that vericl's checks catch
//! them on purpose.
//!
//! Everything conform.rs used to also do — per-kernel GPU launch/input-gen
//! glue, and evidence update/check — is now handled by the macro-generated
//! `conformance_case` (see `#[vericl::kernel]`'s `gen(...)` clause) and the
//! `vericl::suite!`-driven `tests/conformance.rs`, run under plain `cargo
//! test` (README "Locked decisions": conformance is a `cargo test`
//! citizen, not a separate CLI). The defective kernels stay OUT of that
//! suite on purpose — they belong here, as the showcase of checks catching
//! real defects.
//!
//! Usage:
//!   conform     run the defect demo; exits 0 iff every defect is caught

use cubecl::prelude::*;
use vericl_examples::*;

type R = cubecl::wgpu::WgpuRuntime;

const CUBE_DIM: u32 = 256;
const AXPY_SIZES: &[usize] = &[1, 7, 256, 1000, 1027, 4096, 65536];
const SUM_SIZES: &[usize] = &[4096, 65536];
const SEED: u64 = 0xE9_01;

/// Run the SMT out-of-bounds-freedom prover for one kernel — the same
/// wiring `vericl::suite!` generates for the honest kernels in
/// `tests/conformance.rs`, hand-written here since the defect kernels are
/// deliberately excluded from that suite.
fn prove_kernel(
    def: &KernelDefinition,
    buffer_params: &[(&str, bool)],
    structured_assumes: &[vericl::StructuredAssume],
) -> vericl_ir::ProveResult {
    let buffers: Vec<vericl_ir::BufferParam> = buffer_params
        .iter()
        .map(|&(name, is_output)| vericl_ir::BufferParam { name, is_output })
        .collect();
    let assumes: Vec<vericl_ir::Assume> = structured_assumes
        .iter()
        .map(|a| match *a {
            vericl::StructuredAssume::LenEq { a, b } => vericl_ir::Assume::LenEq { a, b },
            vericl::StructuredAssume::LenEqConst { a, value } => {
                vericl_ir::Assume::LenEqConst { a, value }
            }
        })
        .collect();
    vericl_ir::prove_bounds_freedom(def, &buffers, &assumes)
}

/// Run the two-thread race-freedom prover for a cooperative kernel — the
/// deterministic catch for a missing-barrier data race (docs/design-shared-
/// memory.md §5). Same shape as `prove_kernel`, but the race walk instead of
/// the single-thread bounds walk.
fn prove_race_kernel(
    def: &KernelDefinition,
    buffer_params: &[(&str, bool)],
    cube_dim: u32,
) -> vericl_ir::ProveResult {
    let buffers: Vec<vericl_ir::BufferParam> = buffer_params
        .iter()
        .map(|&(name, is_output)| vericl_ir::BufferParam { name, is_output })
        .collect();
    vericl_ir::prove_race_freedom(def, &buffers, &[], cube_dim)
}

/// Print every case outcome and return whether they all passed.
fn print_outcomes(kernel: &str, compare_desc: &str, outcomes: &[vericl::CaseOutcome]) -> bool {
    let pass = outcomes.iter().all(vericl::CaseOutcome::pass);
    println!("  [{}] {kernel} ({compare_desc})", if pass { "PASS" } else { "FAIL" });
    for o in outcomes {
        println!("      {}", vericl::describe_case_outcome(o));
    }
    pass
}

/// Report one defective kernel's already-run differential cases, and record
/// (via `all_caught`) whether the defect was caught — a passing run here
/// would itself be a vericl failure, since these kernels are known-broken
/// on purpose.
fn demo_defect(kernel: &str, compare_desc: &str, outcomes: &[vericl::CaseOutcome], all_caught: &mut bool) {
    let pass = print_outcomes(kernel, compare_desc, outcomes);
    if pass {
        eprintln!("  !! defect in `{kernel}` was NOT caught — this is a vericl failure");
        *all_caught = false;
    } else {
        println!("  -> defect caught");
    }
}

fn main() {
    let device = Default::default();
    let client = R::client(&device);
    let backend = format!("{:?}", R::name(&client));
    println!("vericl defect demo — backend {backend}");
    println!("(each kernel below is deliberately defective; the check must FAIL)\n");

    let mut all_caught = true;

    let axpy_outcomes: Vec<_> = AXPY_SIZES
        .iter()
        .map(|&n| {
            axpy_off_by_one_vericl::conformance_case::<R>(&client, n, SEED ^ 0xDEF0 ^ (n as u64), CUBE_DIM)
        })
        .collect();
    demo_defect(
        "axpy_off_by_one",
        &axpy_off_by_one_vericl::contract().compare.describe(),
        &axpy_outcomes,
        &mut all_caught,
    );

    let racy_outcomes: Vec<_> = SUM_SIZES
        .iter()
        .map(|&n| sum_racy_vericl::conformance_case::<R>(&client, n, SEED ^ 0xACE ^ (n as u64), CUBE_DIM))
        .collect();
    demo_defect(
        "sum_racy",
        &sum_racy_vericl::contract().compare.describe(),
        &racy_outcomes,
        &mut all_caught,
    );

    // The SMT bounds proof is a separate claim from the differential checks
    // above (proved vs. tested — see README "Claims and trust boundaries"),
    // so it is deliberately not folded into `all_caught` the same way; each
    // kernel's outcome is printed and reasoned about on its own terms.
    println!("\nSMT bounds proofs (a separate claim from the differential checks above):");

    let ob_def = axpy_off_by_one_vericl::kernel_definition();
    match prove_kernel(&ob_def, axpy_off_by_one_vericl::BUFFER_PARAMS, axpy_off_by_one_vericl::contract().structured_assumes) {
        vericl_ir::ProveResult::Refuted { obligation, counterexample } => {
            println!("  [REFUTED] axpy_off_by_one: {obligation}");
            println!("    counterexample: {counterexample}");
            println!("  -> bounds defect caught (an out-of-bounds access is reachable)");
        }
        other => {
            eprintln!(
                "  !! axpy_off_by_one's bounds check did NOT refute ({other:?}) — this is a \
                 vericl failure"
            );
            all_caught = false;
        }
    }

    let racy_def = sum_racy_vericl::kernel_definition();
    match prove_kernel(&racy_def, sum_racy_vericl::BUFFER_PARAMS, sum_racy_vericl::contract().structured_assumes) {
        vericl_ir::ProveResult::Proved { obligations } => {
            println!(
                "  [PROVED] sum_racy: {obligations} obligation(s) — the race above is a \
                 differential finding; sum_racy's array accesses are not out of bounds"
            );
        }
        other => {
            eprintln!(
                "  !! sum_racy's bounds check did not prove ({other:?}) — this is a vericl \
                 failure"
            );
            all_caught = false;
        }
    }

    // Cooperative (shared-memory) defect: a missing-barrier reduction step is a
    // *data race* the two-thread race walker proves REFUTED — the deterministic
    // catch (docs/design-shared-memory.md §5.5 / §8 M7). Unlike sum_racy above
    // (a differential-only race finding), this one is caught by proof, with a
    // two-thread counterexample. The differential twin *usually* also diverges
    // from the racy GPU result, but that is nondeterministic, so the proof is
    // the reliable catch and drives `all_caught`, not the differential.
    println!(
        "\nCooperative race-freedom proof (a proved defect, unlike sum_racy's differential-only \
         race):"
    );
    let racy_coop_def = block_sum_reduce_racy_vericl::kernel_definition();
    match prove_race_kernel(&racy_coop_def, block_sum_reduce_racy_vericl::BUFFER_PARAMS, CUBE_DIM) {
        vericl_ir::ProveResult::Refuted { obligation, counterexample } => {
            println!("  [REFUTED] block_sum_reduce_racy: {obligation}");
            println!("    two-thread counterexample: {counterexample}");
            println!(
                "  -> race defect caught by proof (an overlapping `tile[tid] += tile[tid+1]` \
                 reduction stride lets adjacent threads race within one generation; the \
                 phase-split twin would be an unfaithful reference here)"
            );
        }
        other => {
            eprintln!(
                "  !! block_sum_reduce_racy's race check did NOT refute ({other:?}) — this is a \
                 vericl failure"
            );
            all_caught = false;
        }
    }

    // Best-effort differential (nondeterministic — see the note above): run it
    // and report, but do NOT let it drive the exit code.
    let racy_coop_outcomes: Vec<_> = SUM_SIZES
        .iter()
        .map(|&n| {
            block_sum_reduce_racy_vericl::conformance_case::<R>(
                &client,
                n,
                SEED ^ 0xC00B ^ (n as u64),
                CUBE_DIM,
            )
        })
        .collect();
    let racy_coop_diff_pass = racy_coop_outcomes.iter().all(vericl::CaseOutcome::pass);
    if racy_coop_diff_pass {
        println!(
            "  (differential agreed with the racy GPU run this time — nondeterministic; the proof \
             refutation above is the reliable catch)"
        );
    } else {
        println!("  (differential ALSO diverged this run — the race changed the GPU result)");
        for o in &racy_coop_outcomes {
            println!("      {}", vericl::describe_case_outcome(o));
        }
    }

    std::process::exit(if all_caught { 0 } else { 1 });
}
