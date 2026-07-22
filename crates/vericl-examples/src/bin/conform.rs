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

    std::process::exit(if all_caught { 0 } else { 1 });
}
