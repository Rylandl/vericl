//! The three §10.4 corrections that land with the 2-D/3-D dispatch milestone,
//! each with the test the design asked for (docs/design-2d-dispatch.md §12 M7).
//!
//! These are statements that were *wrong today*, independent of whether the
//! feature shipped — so each test is written to fail if the correction is
//! reverted, not merely to observe the corrected state.

use cubecl::prelude::*;
use vericl_examples::*;

// ---------------------------------------------------------------------------
// Correction 4 — the extracted IR is built at the cube dim the kernel is
// actually LAUNCHED with.
//
// `kernel_definition()` used to call `KernelSettings::default()`, whose cube dim
// is `CubeDim { x: 1, y: 1, z: 1 }` — so the `def.cube_dim` that
// `kernel_ir_hash` folds was the *same constant for every kernel in the tree*,
// `cooperative(cube_dim = 256)` included, and contributed nothing at all.
// ---------------------------------------------------------------------------

/// Two kernels with **byte-identical bodies** differing only in their pinned
/// `dispatch(cube_dim = …)`. This is the A/B the correction needs: the IR body
/// really is identical across cube dims (measured in the design's §1.2 probe),
/// so before the fix these two collided on `ir_hash` and the fold was a no-op.
#[vericl::kernel(
    dispatch(cube_dim = (16, 16), extents = (w, h)),
    assumes(out.len() == (w as usize) * (h as usize)),
    compare(exact)
)]
#[cube(launch)]
pub fn ab_dispatch_16x16(out: &mut Array<u32>, w: u32, h: u32) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x < w && y < h {
        out[(y * w + x) as usize] = 1u32;
    }
}

#[vericl::kernel(
    dispatch(cube_dim = (8, 32), extents = (w, h)),
    assumes(out.len() == (w as usize) * (h as usize)),
    compare(exact)
)]
#[cube(launch)]
pub fn ab_dispatch_8x32(out: &mut Array<u32>, w: u32, h: u32) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x < w && y < h {
        out[(y * w + x) as usize] = 1u32;
    }
}

#[test]
fn ir_hash_moves_on_a_dispatch_cube_dim_edit() {
    let a = ab_dispatch_16x16_vericl::kernel_definition();
    let b = ab_dispatch_8x32_vericl::kernel_definition();

    // The premise, restated as an assertion: the bodies ARE identical. If this
    // ever stops holding, the test below would pass for the wrong reason (the
    // body, not the cube dim, would be moving the hash) and this line says so.
    assert_eq!(
        format!("{}", a.body),
        format!("{}", b.body),
        "the A/B premise: two kernels differing only in `dispatch(cube_dim = …)` have \
         byte-identical IR bodies, so `ir_hash` can only distinguish them via `def.cube_dim`"
    );
    assert_eq!(a.cube_dim, CubeDim::new_2d(16, 16));
    assert_eq!(b.cube_dim, CubeDim::new_2d(8, 32));

    assert_ne!(
        vericl_ir::kernel_ir_hash(&a),
        vericl_ir::kernel_ir_hash(&b),
        "§10.4 correction 4: `kernel_definition()` must build at the clause's pinned cube dim, so \
         `ir_hash` is an independent tripwire on the dispatch shape. With \
         `KernelSettings::default()` restored these two collide — the bodies are identical and \
         `def.cube_dim` is the constant (1,1,1) for every kernel in the tree."
    );
}

/// The same gap, closed for the shipped **1-D cooperative** path: a cooperative
/// kernel's `ir_hash` must move on a `cooperative(cube_dim = N)` edit too.
/// Before the correction it could not — every kernel was extracted at `(1,1,1)`.
#[vericl::kernel(compare(max_ulp = 0), gen(input in -10.0..=10.0), cooperative(cube_dim = 64))]
#[cube(launch)]
pub fn ab_coop_64(input: &Array<f32>, output: &mut Array<f32>) {
    let tid = UNIT_POS as usize;
    let mut tile = SharedMemory::<f32>::new(64usize);
    tile[tid] = 0.0f32;
    if ABSOLUTE_POS < input.len() {
        tile[tid] = input[ABSOLUTE_POS];
    }
    sync_cube();
    if tid == 0usize && CUBE_POS < output.len() {
        output[CUBE_POS] = tile[0usize];
    }
}

#[vericl::kernel(compare(max_ulp = 0), gen(input in -10.0..=10.0), cooperative(cube_dim = 128))]
#[cube(launch)]
pub fn ab_coop_128(input: &Array<f32>, output: &mut Array<f32>) {
    let tid = UNIT_POS as usize;
    let mut tile = SharedMemory::<f32>::new(64usize);
    tile[tid] = 0.0f32;
    if ABSOLUTE_POS < input.len() {
        tile[tid] = input[ABSOLUTE_POS];
    }
    sync_cube();
    if tid == 0usize && CUBE_POS < output.len() {
        output[CUBE_POS] = tile[0usize];
    }
}

#[test]
fn ir_hash_moves_on_a_cooperative_cube_dim_edit() {
    let a = ab_coop_64_vericl::kernel_definition();
    let b = ab_coop_128_vericl::kernel_definition();
    assert_eq!(
        format!("{}", a.body),
        format!("{}", b.body),
        "the A/B premise for the cooperative path"
    );
    assert_ne!(
        vericl_ir::kernel_ir_hash(&a),
        vericl_ir::kernel_ir_hash(&b),
        "§10.4 correction 4 closes the same gap for the shipped 1-D cooperative path"
    );
}

// ---------------------------------------------------------------------------
// Corrections 2 and 3 — the evidence records the launch shape it measured, and
// the logic it actually used.
// ---------------------------------------------------------------------------

fn load(rel: &str) -> serde_json::Value {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!("could not read {} ({e}) — run `VERICL_UPDATE=1 cargo test` first", p.display())
    }))
    .expect("evidence is valid JSON")
}

fn claim<'a>(m: &'a serde_json::Value, kernel: &str, kind: &str) -> &'a serde_json::Value {
    m["entries"]
        .as_array()
        .expect("entries")
        .iter()
        .find(|e| e["kernel"] == kernel)
        .unwrap_or_else(|| panic!("no entry for `{kernel}`"))["claims"]
        .as_array()
        .expect("claims")
        .iter()
        .find(|c| c["kind"] == kind)
        .unwrap_or_else(|| panic!("no `{kind}` claim on `{kernel}`"))
}

/// §10.4 correction 2: the differential config records the launch shape it was
/// produced under. Before this milestone it recorded a *scalar* `cube_dim` and
/// no rank at all, so evidence could not distinguish a 1-D dispatch from any
/// other — the recordable half of the D1 hole (§3.3).
#[test]
fn the_differential_config_records_the_launch_shape() {
    let two_d = load("evidence/vericl_2d.json");
    let cfg = &claim(&two_d, "box_blur3x3", "tested")["config"];
    assert_eq!(cfg["rank"], 2, "a rank-2 dispatch records its rank");
    assert_eq!(
        cfg["cube_dim"],
        serde_json::json!([16, 16]),
        "the FULL pinned per-axis cube dim, at the clause's own arity"
    );
    assert_eq!(
        cfg["sizes_unit"], "extents",
        "a 2-D suite's sizes are per-axis extents, not thread counts — the \
         `differential_vector_config` precedent, for the same units reason"
    );
    assert_eq!(
        cfg["sizes"],
        serde_json::json!([[37, 19], [64, 64], [1, 1], [3, 129], [129, 3], [255, 257]]),
        "each case at the clause's arity — padding to triples would invent an extent"
    );

    // And the 1-D side gains `rank: 1`, so old and new evidence are comparable
    // rather than distinguished only by a missing field.
    let one_d = load("evidence/vericl.json");
    let axpy_cfg = &claim(&one_d, "axpy", "tested")["config"];
    assert_eq!(axpy_cfg["rank"], 1, "a 1-D suite records rank 1 explicitly");
    assert!(axpy_cfg.get("sizes_unit").is_none(), "1-D sizes are thread counts, the default unit");
}

/// §10.4 correction 3: `proved_config`'s `logic` is the logic actually in force,
/// not the hardcoded `QF_LIA` it used to be. A `LenEqProduct` assume puts a
/// genuinely nonlinear `len = x*y` in the *global* assertion context.
///
/// Both directions are asserted — a test that only checked the `QF_NIA` side
/// would pass if the field were hardcoded the other way.
#[test]
fn the_proved_config_records_the_logic_actually_used() {
    let two_d = load("evidence/vericl_2d.json");
    assert_eq!(
        claim(&two_d, "box_blur3x3", "proved")["config"]["logic"],
        "QF_NIA",
        "a kernel with a `len == w*h` product assume is not QF_LIA"
    );
    let one_d = load("evidence/vericl.json");
    assert_eq!(
        claim(&one_d, "axpy", "proved")["config"]["logic"],
        "QF_LIA",
        "a kernel with only linear length facts still records QF_LIA — otherwise the field would \
         be a different constant rather than a measurement"
    );
}

/// §11's honesty requirement, as a test rather than a paragraph: the shipped 2-D
/// evidence must contain a class that is `tested` but NOT `proved`, so the table
/// is not all green by construction. `elementwise3d_scale` is that class — a
/// rank-3 volume index needs `len == w*h*d`, which the binary product assume
/// cannot express.
#[test]
fn elementwise3d_scale_is_out_of_subset_for_want_of_a_triple_product_assume() {
    let def = elementwise3d_scale_vericl::kernel_definition();
    let buffers: Vec<vericl_ir::BufferParam> = elementwise3d_scale_vericl::BUFFER_PARAMS
        .iter()
        .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
        .collect();
    let assumes: Vec<vericl_ir::Assume> = elementwise3d_scale_vericl::contract()
        .structured_assumes
        .iter()
        .map(|a| match *a {
            vericl::StructuredAssume::LenEq { a, b } => vericl_ir::Assume::LenEq { a, b },
            other => panic!("unexpected assume {other:?}"),
        })
        .collect();
    let cd = elementwise3d_scale_vericl::DISPATCH_CUBE_DIM.expect("declares dispatch(...)");
    match vericl_ir::prove_bounds_freedom_dispatch(&def, &buffers, &assumes, cd) {
        vericl_ir::ProveResult::OutOfSubset { reason } => {
            println!("rank-3 volume index: OutOfSubset — {reason}");
        }
        vericl_ir::ProveResult::Proved { obligations } => panic!(
            "a `w*h*d` volume index Proved{{{obligations}}} with only `inp.len() == out.len()` in \
             scope — either the assume vocabulary grew a triple product (update this test and \
             docs/coverage.md's 3-D row) or something is unsound"
        ),
        other => panic!("expected OutOfSubset for the rank-3 volume index, got {other:?}"),
    }
}
