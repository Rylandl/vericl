//! 2-D / 3-D dispatch — GPU ground truth through the **generated** twin, and
//! the twin's own decided properties (docs/design-2d-dispatch.md §4.7, §6).
//!
//! The design's §6 probe hand-wrote both the kernels and their nested-grid-loop
//! twins and measured **24 / 24 bit-exact** against wgpu/Metal. This file
//! re-establishes the same count through the machinery that now derives the
//! twin, the launch, and the grid automatically: four kernels — three rank-2 at
//! six image shapes plus the rank-3 one at six volume shapes — every case
//! compared bit-for-bit.
//!
//! The three suite-wired kernels are also run at these shapes by
//! `tests/conformance_2d.rs`; the duplication is deliberate. This file's claim
//! is the design's ground-truth *count* reproduced end to end (rank 3 included,
//! which no suite covers), and it is where the twin's structural properties are
//! pinned — those need direct `reference(...)` calls, not a suite run.

use cubecl::Runtime;
use vericl_examples::*;

type R = cubecl::wgpu::WgpuRuntime;

/// The design's six image shapes: `w != h`, neither a multiple of the cube dim,
/// plus the degenerate `1x1` and the thin `3x129` / `129x3` cases.
const IMAGE_SHAPES: [[usize; 3]; 6] =
    [[37, 19, 1], [64, 64, 1], [1, 1, 1], [3, 129, 1], [129, 3, 1], [255, 257, 1]];

/// Six volume shapes for the rank-3 kernel, chosen on the same principles
/// (unequal extents, none a multiple of the `(8, 8, 4)` cube, the degenerate
/// `1x1x1`, and two thin ones).
const VOLUME_SHAPES: [[usize; 3]; 6] =
    [[7, 5, 3], [16, 16, 8], [1, 1, 1], [3, 3, 17], [17, 3, 3], [33, 9, 5]];

/// **24 / 24 bit-exact, through the generated twin.**
///
/// Each case runs the macro-generated `conformance_case`, which draws the
/// inputs, binds the extents from the case, runs the derived nested-grid twin,
/// launches the kernel at `CubeCount::Static(ceil(e0/Wx), ceil(e1/Wy),
/// ceil(e2/Wz))` with the clause's pinned `CubeDim`, and compares — at
/// `max_ulp = 0` for the f32 kernels and `exact` for the u32 one, i.e. bitwise
/// in both cases.
///
/// What the count establishes, three things at once (§6): the nested-loop twin
/// is the right model *including* the padding threads (at `37x19 / (16,16)`
/// they are 1 536 - 703 = 833 of 1 536); 2-D dispatch introduces no
/// float-ordering or FMA-contraction divergence; and rank 3 is genuinely just
/// rank 2 with another loop.
#[test]
fn twenty_four_of_twenty_four_bit_exact_through_the_generated_twin() {
    let client = R::client(&Default::default());
    let seed = 0x2D15_0A7Cu64;
    let mut total = 0usize;
    let mut exact = 0usize;

    for e in IMAGE_SHAPES {
        for (name, outcome) in [
            (
                "elementwise2d_scale",
                elementwise2d_scale_vericl::conformance_case::<R>(&client, e, seed),
            ),
            ("transpose2d", transpose2d_vericl::conformance_case::<R>(&client, e, seed)),
            ("box_blur3x3", box_blur3x3_vericl::conformance_case::<R>(&client, e, seed)),
        ] {
            total += 1;
            let pass = outcome.pass();
            if pass {
                exact += 1;
            }
            println!(
                "  {name:<20} w={:<4} h={:<4} bit-exact: {pass}  ({})",
                e[0],
                e[1],
                vericl::describe_case_outcome(&outcome)
            );
            assert!(pass, "{name} at {e:?} is not bit-exact: {outcome:?}");
        }
    }

    for e in VOLUME_SHAPES {
        let outcome = elementwise3d_scale_vericl::conformance_case::<R>(&client, e, seed);
        total += 1;
        let pass = outcome.pass();
        if pass {
            exact += 1;
        }
        println!(
            "  {:<20} w={:<4} h={:<4} d={:<4} bit-exact: {pass}",
            "elementwise3d_scale", e[0], e[1], e[2]
        );
        assert!(pass, "elementwise3d_scale at {e:?} is not bit-exact: {outcome:?}");
    }

    println!("=== bit-exact through the generated twin: {exact}/{total} ===");
    assert_eq!((exact, total), (24, 24), "the design's §6 ground-truth count");
}

/// `uses(...)` composition under a `dispatch(...)` clause — §9 lists it as
/// "support (a helper's twin cannot read topology at all; per-axis positions are
/// passed as plain `u32` arguments)", and a supported capability with no
/// compiled instance is the round-11 MODERATE-3 defect. This is the instance:
/// bit-exact against the generated twin at every image shape, and `Proved`.
#[test]
fn uses_composition_works_under_a_dispatch_clause() {
    let client = R::client(&Default::default());
    for e in IMAGE_SHAPES {
        let outcome = checkerboard2d_vericl::conformance_case::<R>(&client, e, 0xC0FFEEu64);
        assert!(outcome.pass(), "checkerboard2d at {e:?}: {outcome:?}");
    }

    let def = checkerboard2d_vericl::kernel_definition();
    let buffers: Vec<vericl_ir::BufferParam> = checkerboard2d_vericl::BUFFER_PARAMS
        .iter()
        .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
        .collect();
    let assumes: Vec<vericl_ir::Assume> = checkerboard2d_vericl::contract()
        .structured_assumes
        .iter()
        .map(|a| match *a {
            vericl::StructuredAssume::LenEq { a, b } => vericl_ir::Assume::LenEq { a, b },
            vericl::StructuredAssume::LenEqProduct { a, x_scalar, y_scalar, .. } => {
                vericl_ir::Assume::LenEqProduct { a, x_scalar, y_scalar }
            }
            other => panic!("unexpected assume {other:?}"),
        })
        .collect();
    let cd = checkerboard2d_vericl::DISPATCH_CUBE_DIM.expect("declares dispatch(...)");
    match vericl_ir::prove_bounds_freedom_dispatch(&def, &buffers, &assumes, cd) {
        vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
        other => panic!("a composed 2-D kernel must still prove: {other:?}"),
    }
}

/// §4.7 property 2 + §13 risk 6, in one measurement: **the twin's loops range
/// over the GRID, not the image**, and the grid comes from the extents in the
/// order the clause names them.
///
/// With `w = 37, h = 19` and the pinned `(16, 16)` cube the grid is
/// `(48, 32)` — larger than the image on both axes, so the extra threads run the
/// guard and take the `else`. Two directions are asserted:
///
/// * **Padding is inert.** An even larger grid `(64, 64)` produces the identical
///   output, i.e. the guard really does neutralize every padding thread. (A twin
///   that looped `0..w` would also pass this half — it is the control, not the
///   discriminator.)
/// * **The grid is load-bearing.** A grid derived from *swapped* extents,
///   `(32, 48)`, produces a DIFFERENT output: `grid.0 = 32 < w = 37`, so image
///   columns 32..36 are never visited. This is the discriminator for both
///   properties at once — it can only differ if the twin's bounds are the grid
///   — and it is exactly risk 6's attack (`dispatch(extents = (h, w))` with the
///   names swapped relative to the body's use), shown to change the twin's
///   answer rather than sliding through in bounds.
#[test]
fn twin_iterates_the_grid_and_a_swapped_extent_changes_which_threads_are_padding() {
    let (w, h) = (37u32, 19u32);
    let inp: Vec<f32> = (0..(w * h)).map(|i| (i % 97) as f32 - 40.0).collect();

    let run = |grid: (u32, u32, u32)| {
        let mut out = vec![0f32; (w * h) as usize];
        box_blur3x3_vericl::reference(&inp, &mut out, w, h, grid);
        out
    };

    let exact_grid = run((48, 32, 1));
    let bigger_grid = run((64, 64, 1));
    assert_eq!(
        exact_grid, bigger_grid,
        "padding threads must take the `else` — a larger grid cannot change the result"
    );

    let swapped_grid = run((32, 48, 1));
    assert_ne!(
        exact_grid, swapped_grid,
        "the twin's loop bounds must be the GRID: with grid.0 = 32 < w = 37, image columns \
         32..36 are never visited, so a swapped `extents = (h, w)` clause must produce a \
         different reference — this is what makes the differential lane catch risk 6"
    );
    let untouched = (32..37)
        .filter(|x| swapped_grid[*x as usize] == 0.0) // row 0, so the index IS x
        .count();
    assert_eq!(untouched, 5, "exactly the 5 unvisited columns of row 0 stay at their initial 0.0");
}

/// §4.7 property 1: **the twin's write order for aliasing writes is the flat
/// `ABSOLUTE_POS` order** — Z outer, Y, X innermost, X fastest-varying.
///
/// `diag_alias_write2d` writes `out[x + y] = 100*x + y`, so slot 1 is written by
/// both `(x=1, y=0)` and `(x=0, y=1)`. Under the required Y-outer/X-inner
/// nesting the later visit is `(0, 1)` and the slot ends at `1`; under the
/// opposite X-outer/Y-inner nesting it would be `(1, 0)` and the slot would end
/// at `100`. Both are asserted — the second is the negative control, and it is
/// what makes this test discriminate rather than merely pass.
///
/// This matters beyond aesthetics: a kernel ported from flat `ABSOLUTE_POS`
/// addressing to per-axis addressing must keep the same reference semantics, and
/// the reference semantics for aliasing writes *is* the iteration order.
#[test]
fn twin_write_order_is_the_flat_absolute_pos_order() {
    let (w, h) = (4u32, 3u32);
    let mut out = vec![0u32; (w * h) as usize];
    // Grid = image here (both extents are multiples of the pinned (2, 2) cube),
    // so the order under test is not confounded by padding.
    diag_alias_write2d_vericl::reference(&mut out, w, h, (4, 4, 1));

    // Slot k's last writer under Z->Y->X is the (x, y) on `x + y == k` with the
    // largest y (X innermost means y advances last).
    let expected_row_major: Vec<u32> = (0..(w * h))
        .map(|k| {
            (0..w)
                .flat_map(|x| (0..h).map(move |y| (x, y)))
                .filter(|(x, y)| x + y == k && *x < w && *y < h)
                .max_by_key(|(x, y)| (*y, *x))
                .map(|(x, y)| x * 100 + y)
                .unwrap_or(0)
        })
        .collect();
    assert_eq!(
        out, expected_row_major,
        "the twin must visit in flat ABSOLUTE_POS order (Z outer, X innermost)"
    );

    // NEGATIVE CONTROL: the opposite nesting picks the largest x instead.
    let expected_column_major: Vec<u32> = (0..(w * h))
        .map(|k| {
            (0..w)
                .flat_map(|x| (0..h).map(move |y| (x, y)))
                .filter(|(x, y)| x + y == k && *x < w && *y < h)
                .max_by_key(|(x, y)| (*x, *y))
                .map(|(x, y)| x * 100 + y)
                .unwrap_or(0)
        })
        .collect();
    assert_ne!(
        expected_row_major, expected_column_major,
        "the control must actually distinguish the two nestings, or this test is vacuous"
    );
    assert_ne!(
        out, expected_column_major,
        "an X-outer/Y-inner nest would change the aliasing-write convention silently"
    );
}

/// §4.7 property 4: `CUBE_POS_a`, `UNIT_POS_a`, `CUBE_DIM_a` and `CUBE_COUNT_a`
/// are bound in the twin by the **per-axis** decomposition — `abs_a % Wa`,
/// `abs_a / Wa`, the pinned literal, and `grid_a / Wa` — each of which is exact
/// on hardware for every launch shape (measured 0 violations / 1 212 threads).
///
/// `topology_report2d` writes each thread's own reconstruction of a value that
/// is only correct if all four bindings agree with the device's. Bit-exact
/// against wgpu at three shapes here; the flat cross-axis identity that breaks
/// in 2-D is never used, because the flat builtins are rejected inside the
/// clause.
#[test]
fn per_axis_decomposition_agrees_with_the_device() {
    let client = R::client(&Default::default());
    for e in [[37usize, 19, 1], [64, 64, 1], [3, 129, 1]] {
        let outcome = topology_report2d_vericl::conformance_case::<R>(&client, e, 0x7C0Fu64);
        assert!(
            outcome.pass(),
            "the twin's per-axis topology bindings must match the device at {e:?}: {outcome:?}"
        );
    }
}
