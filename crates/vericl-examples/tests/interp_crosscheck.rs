//! Public-example cross-check: the concrete IR interpreter (`vericl_ir::
//! interpret_dispatch`) vs the macro-derived reference twin, on the SAME
//! `KernelDefinition` the prover consumes, over many random inputs.
//!
//! This is the model-fidelity anchor that uses **real** `#[cube]`-expanded IR
//! (via each kernel's generated `kernel_definition()`), not the hand-built IR of
//! the `vericl-ir` fuzz lane. Agreement here means two independent
//! implementations of cube semantics — the token-rewrite twin and the concrete
//! IR interpreter — concur on the actual example kernels. It shrinks
//! model-fidelity risk empirically; it is not a proof (see docs/interpreter.md).
//!
//! The twin (`<name>_vericl::reference`) is validated against GPU/cubecl-cpu by
//! the conformance suite (`tests/conformance.rs`); this test establishes
//! interpreter ≡ twin bit-for-bit, so interpreter ≈ GPU follows transitively
//! for the tolerance the conformance suite records. One kernel (`axpy`) is
//! additionally checked three-way against a real wgpu launch below.

use vericl_examples::*;
use vericl_ir::{Buffer, Dispatch3, Inputs, Outcome, ScalarBinding, interpret_dispatch};

// ---- a tiny deterministic RNG (no external dep) ----------------------------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x2545_F491_4F6C_DD1D)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn u32(&mut self) -> u32 {
        self.next() as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
    /// A float in `[lo, hi]`.
    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        let t = (self.next() >> 11) as f32 / (1u64 << 53) as f32;
        lo + t * (hi - lo)
    }
}

fn interp(def: &cubecl::prelude::KernelDefinition, inputs: Inputs) -> Vec<Buffer> {
    match interpret_dispatch(def, &inputs) {
        Outcome::Completed { buffers } => buffers,
        other => panic!("interpreter did not complete on an honest kernel: {other:?}"),
    }
}

/// Bit-exact f32 comparison (interpreter and twin run identical ops, so their
/// outputs must be bit-identical, not merely close).
fn assert_f32_bits(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{ctx}: element {i}: interp {g} ({:#x}) != twin {w} ({:#x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

const SIZES: &[usize] = &[1, 2, 3, 7, 8, 16, 33, 64, 100];

#[test]
fn interp_axpy_matches_twin() {
    let def = axpy_vericl::kernel_definition();
    let mut rng = Rng::new(1);
    for &n in SIZES {
        for _ in 0..8 {
            let alpha = rng.f32(-4.0, 4.0);
            let x: Vec<f32> = (0..n).map(|_| rng.f32(-100.0, 100.0)).collect();
            let y0: Vec<f32> = (0..n).map(|_| rng.f32(-100.0, 100.0)).collect();
            let mut y_ref = y0.clone();
            axpy_vericl::reference(alpha, &x, &mut y_ref, n);
            let out = interp(
                &def,
                Inputs {
                    buffers: vec![Buffer::f32("x", &x, false), Buffer::f32("y", &y0, true)],
                    scalars: vec![ScalarBinding::f32(0, alpha)],
                    cube_dim: 256,
                    num_threads: n as u32,
                    dispatch: None,
                },
            );
            assert_f32_bits(&out[1].as_f32(), &y_ref, &format!("axpy n={n}"));
        }
    }
}

#[test]
fn interp_xorshift_step_matches_twin() {
    let def = xorshift_step_vericl::kernel_definition();
    let mut rng = Rng::new(2);
    for &n in SIZES {
        let x: Vec<u32> = (0..n).map(|_| rng.u32()).collect();
        let mut y_ref = vec![0u32; n];
        xorshift_step_vericl::reference(&x, &mut y_ref, n);
        let out = interp(
            &def,
            Inputs {
                buffers: vec![Buffer::u32("x", &x, false), Buffer::u32("y", &vec![0u32; n], true)],
                scalars: vec![],
                cube_dim: 256,
                num_threads: n as u32,
                dispatch: None,
            },
        );
        assert_eq!(out[1].as_u32(), y_ref, "xorshift n={n}");
    }
}

#[test]
fn interp_mix_u32_matches_twin() {
    let def = mix_u32_vericl::kernel_definition();
    let mut rng = Rng::new(3);
    for &n in SIZES {
        let x: Vec<u32> = (0..n).map(|_| rng.u32()).collect();
        let mut y_ref = vec![0u32; n];
        mix_u32_vericl::reference(&x, &mut y_ref, n);
        let out = interp(
            &def,
            Inputs {
                buffers: vec![Buffer::u32("x", &x, false), Buffer::u32("y", &vec![0u32; n], true)],
                scalars: vec![],
                cube_dim: 256,
                num_threads: n as u32,
                dispatch: None,
            },
        );
        assert_eq!(out[1].as_u32(), y_ref, "mix_u32 n={n}");
    }
}

#[test]
fn interp_flatten_decode_scale_matches_twin() {
    let def = flatten_decode_scale_vericl::kernel_definition();
    let mut rng = Rng::new(4);
    for &n in SIZES {
        for _ in 0..4 {
            let width = 1 + rng.below(64);
            let scale = rng.f32(0.1, 4.0);
            let x: Vec<f32> = (0..n).map(|_| rng.f32(-100.0, 100.0)).collect();
            let mut y_ref = vec![0.0f32; n];
            flatten_decode_scale_vericl::reference(&x, &mut y_ref, width, scale, n);
            let out = interp(
                &def,
                Inputs {
                    buffers: vec![Buffer::f32("x", &x, false), Buffer::f32("y", &vec![0.0; n], true)],
                    scalars: vec![ScalarBinding::u32(0, width), ScalarBinding::f32(0, scale)],
                    cube_dim: 256,
                    num_threads: n as u32,
                    dispatch: None,
                },
            );
            assert_f32_bits(&out[1].as_f32(), &y_ref, &format!("flatten n={n} w={width}"));
        }
    }
}

#[test]
fn interp_gather_copy_matches_twin() {
    let def = gather_copy_vericl::kernel_definition();
    let mut rng = Rng::new(5);
    for &n in SIZES {
        let m = 1 + rng.below(64); // x.len()
        let x: Vec<f32> = (0..m).map(|_| rng.f32(-10.0, 10.0)).collect();
        let offsets: Vec<u32> = (0..n).map(|_| rng.below(m)).collect();
        let mut y_ref = vec![0.0f32; n];
        gather_copy_vericl::reference(&x, &offsets, &mut y_ref, n);
        let out = interp(
            &def,
            Inputs {
                buffers: vec![
                    Buffer::f32("x", &x, false),
                    Buffer::u32("offsets", &offsets, false),
                    Buffer::f32("y", &vec![0.0; n], true),
                ],
                scalars: vec![],
                cube_dim: 256,
                num_threads: n as u32,
                dispatch: None,
            },
        );
        assert_f32_bits(&out[2].as_f32(), &y_ref, &format!("gather n={n}"));
    }
}

#[test]
fn interp_select_mode_matches_twin() {
    let def = select_mode_vericl::kernel_definition();
    let mut rng = Rng::new(6);
    for &n in SIZES {
        for mode in 0..4u32 {
            let x: Vec<f32> = (0..n).map(|_| rng.f32(-10.0, 10.0)).collect();
            let mut y_ref = vec![0.0f32; n];
            select_mode_vericl::reference(mode, &x, &mut y_ref, n);
            let out = interp(
                &def,
                Inputs {
                    buffers: vec![Buffer::f32("x", &x, false), Buffer::f32("y", &vec![0.0; n], true)],
                    scalars: vec![ScalarBinding::u32(0, mode)],
                    cube_dim: 256,
                    num_threads: n as u32,
                    dispatch: None,
                },
            );
            assert_f32_bits(&out[1].as_f32(), &y_ref, &format!("select_mode n={n} mode={mode}"));
        }
    }
}

#[test]
fn interp_offset_window_matches_twin() {
    let def = offset_window_vericl::kernel_definition();
    let mut rng = Rng::new(7);
    for &n in SIZES {
        // x must be 4 longer than y (the length-relationship assume).
        let x: Vec<f32> = (0..n + 4).map(|_| rng.f32(-10.0, 10.0)).collect();
        let mut y_ref = vec![0.0f32; n];
        offset_window_vericl::reference(&x, &mut y_ref, n);
        let out = interp(
            &def,
            Inputs {
                buffers: vec![Buffer::f32("x", &x, false), Buffer::f32("y", &vec![0.0; n], true)],
                scalars: vec![],
                cube_dim: 256,
                num_threads: n as u32,
                dispatch: None,
            },
        );
        assert_f32_bits(&out[1].as_f32(), &y_ref, &format!("offset_window n={n}"));
    }
}

#[test]
fn interp_fir3_matches_twin() {
    // taps is #[comptime]-pinned in the kernel definition; the twin bakes the
    // same value, so no runtime scalar appears.
    let def = fir3_vericl::kernel_definition();
    let mut rng = Rng::new(8);
    for &n in SIZES {
        let x: Vec<f32> = (0..n).map(|_| rng.f32(-10.0, 10.0)).collect();
        let mut y_ref = vec![0.0f32; n];
        fir3_vericl::reference(&x, &mut y_ref, n);
        let out = interp(
            &def,
            Inputs {
                buffers: vec![Buffer::f32("x", &x, false), Buffer::f32("y", &vec![0.0; n], true)],
                scalars: vec![],
                cube_dim: 256,
                num_threads: n as u32,
                dispatch: None,
            },
        );
        assert_f32_bits(&out[1].as_f32(), &y_ref, &format!("fir3 n={n}"));
    }
}

#[test]
fn interp_gain_kernel_matches_twin() {
    let def = gain_kernel_vericl::kernel_definition();
    let mut rng = Rng::new(9);
    for &n in SIZES {
        let gain = rng.f32(-4.0, 4.0);
        let x: Vec<f32> = (0..n).map(|_| rng.f32(-100.0, 100.0)).collect();
        let mut y_ref = vec![0.0f32; n];
        gain_kernel_vericl::reference(&x, &mut y_ref, gain, n);
        let out = interp(
            &def,
            Inputs {
                buffers: vec![Buffer::f32("x", &x, false), Buffer::f32("y", &vec![0.0; n], true)],
                scalars: vec![ScalarBinding::f32(0, gain)],
                cube_dim: 256,
                num_threads: n as u32,
                dispatch: None,
            },
        );
        assert_f32_bits(&out[1].as_f32(), &y_ref, &format!("gain_kernel n={n}"));
    }
}

#[test]
fn interp_lcg_map_matches_twin() {
    let def = lcg_map_vericl::kernel_definition();
    let mut rng = Rng::new(10);
    for &n in SIZES {
        let x: Vec<u32> = (0..n).map(|_| rng.u32()).collect();
        let mut y_ref = vec![0u32; n];
        lcg_map_vericl::reference(&x, &mut y_ref, n);
        let out = interp(
            &def,
            Inputs {
                buffers: vec![Buffer::u32("x", &x, false), Buffer::u32("y", &vec![0u32; n], true)],
                scalars: vec![],
                cube_dim: 256,
                num_threads: n as u32,
                dispatch: None,
            },
        );
        assert_eq!(out[1].as_u32(), y_ref, "lcg_map n={n}");
    }
}

#[test]
fn interp_comptime_shift_matches_twin() {
    let def = comptime_shift_vericl::kernel_definition();
    let mut rng = Rng::new(11);
    for &n in SIZES {
        let x: Vec<u32> = (0..n).map(|_| rng.u32()).collect();
        let mut y_ref = vec![0u32; n];
        comptime_shift_vericl::reference(&x, &mut y_ref, n);
        let out = interp(
            &def,
            Inputs {
                buffers: vec![Buffer::u32("x", &x, false), Buffer::u32("y", &vec![0u32; n], true)],
                scalars: vec![],
                cube_dim: 256,
                num_threads: n as u32,
                dispatch: None,
            },
        );
        assert_eq!(out[1].as_u32(), y_ref, "comptime_shift n={n}");
    }
}

#[test]
fn interp_mul_hi_map_matches_twin() {
    let def = mul_hi_map_vericl::kernel_definition();
    let mut rng = Rng::new(12);
    for &n in SIZES {
        let a: Vec<u32> = (0..n).map(|_| rng.u32()).collect();
        let b: Vec<u32> = (0..n).map(|_| rng.u32()).collect();
        let mut y_ref = vec![0u32; n];
        mul_hi_map_vericl::reference(&a, &b, &mut y_ref, n);
        let out = interp(
            &def,
            Inputs {
                buffers: vec![
                    Buffer::u32("a", &a, false),
                    Buffer::u32("b", &b, false),
                    Buffer::u32("y", &vec![0u32; n], true),
                ],
                scalars: vec![],
                cube_dim: 256,
                num_threads: n as u32,
                dispatch: None,
            },
        );
        assert_eq!(out[2].as_u32(), y_ref, "mul_hi_map n={n}");
    }
}

#[test]
fn interp_unit_interval_map_matches_twin() {
    // Mixed element types: u32 input, f32 output.
    let def = unit_interval_map_vericl::kernel_definition();
    let mut rng = Rng::new(13);
    for &n in SIZES {
        let x: Vec<u32> = (0..n).map(|_| rng.u32()).collect();
        let mut y_ref = vec![0.0f32; n];
        unit_interval_map_vericl::reference(&x, &mut y_ref, n);
        let out = interp(
            &def,
            Inputs {
                buffers: vec![Buffer::u32("x", &x, false), Buffer::f32("y", &vec![0.0; n], true)],
                scalars: vec![],
                cube_dim: 256,
                num_threads: n as u32,
                dispatch: None,
            },
        );
        assert_f32_bits(&out[1].as_f32(), &y_ref, &format!("unit_interval_map n={n}"));
    }
}

/// Three-way: interpreter vs twin vs a real wgpu GPU launch, on identical
/// inputs, for an exact integer kernel (no float contraction, so all three must
/// agree bit-for-bit — the strongest form of the cross-check). `xorshift_step`
/// is `compare(exact)`; the interpreter, the twin, and the GPU are three
/// independent realizations of its semantics.
#[test]
fn interp_vs_twin_vs_gpu_xorshift() {
    use cubecl::prelude::*;
    type R = cubecl::wgpu::WgpuRuntime;

    let def = xorshift_step_vericl::kernel_definition();
    let mut rng = Rng::new(99);
    let n = 1027usize; // deliberately not a multiple of the cube dim
    let x: Vec<u32> = (0..n).map(|_| rng.u32()).collect();

    // (1) twin
    let mut y_twin = vec![0u32; n];
    xorshift_step_vericl::reference(&x, &mut y_twin, n);

    // (2) interpreter over the real kernel_definition() IR
    let out = interp(
        &def,
        Inputs {
            buffers: vec![Buffer::u32("x", &x, false), Buffer::u32("y", &vec![0u32; n], true)],
            scalars: vec![],
            cube_dim: 256,
            num_threads: n as u32,
            dispatch: None,
        },
    );
    let y_interp = out[1].as_u32();

    // (3) real GPU launch on wgpu/Metal
    let client = R::client(&Default::default());
    let x_h = client.create_from_slice(u32::as_bytes(&x));
    let y_h = client.create_from_slice(u32::as_bytes(&vec![0u32; n]));
    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);
    xorshift_step::launch::<R>(
        &client,
        CubeCount::Static(cube_count, 1, 1),
        CubeDim::new_1d(cube_dim),
        unsafe { ArrayArg::from_raw_parts(x_h, n) },
        unsafe { ArrayArg::from_raw_parts(y_h.clone(), n) },
    );
    let bytes = client.read_one(y_h).unwrap();
    let y_gpu = u32::from_bytes(&bytes).to_vec();

    assert_eq!(y_interp, y_twin, "interpreter vs twin");
    assert_eq!(y_interp, y_gpu, "interpreter vs GPU");
    assert_eq!(y_twin, y_gpu, "twin vs GPU");
}

/// The dispatch can launch more threads than the buffer is long; the guard
/// `ABSOLUTE_POS < y.len()` must filter them in the interpreter exactly as it
/// does in the twin (no spurious OOB, no extra writes).
#[test]
fn interp_respects_guard_when_threads_exceed_length() {
    let def = axpy_vericl::kernel_definition();
    let n = 5usize;
    let alpha = 2.0f32;
    let x = vec![1.0f32; n];
    let y0 = vec![10.0f32; n];
    let mut y_ref = y0.clone();
    axpy_vericl::reference(alpha, &x, &mut y_ref, 256); // 256 threads, len 5
    let out = interp(
        &def,
        Inputs {
            buffers: vec![Buffer::f32("x", &x, false), Buffer::f32("y", &y0, true)],
            scalars: vec![ScalarBinding::f32(0, alpha)],
            cube_dim: 256,
            num_threads: 256,
            dispatch: None,
        },
    );
    assert_f32_bits(&out[1].as_f32(), &y_ref, "axpy guard n=5 threads=256");
}

// ---------------------------------------------------------------------------
// 2-D / 3-D dispatch (docs/design-2d-dispatch.md §12 M5).
//
// The interpreter's topology answers used to be pinned to a 1-D launch
// (`AbsolutePosY => 0`, `CubeDimY => 1`) — correct then, and a round-11-style
// *classification split* the moment a multi-axis launch exists. These two tests
// pin both halves: with the launch supplied the interpreter agrees with the twin
// thread-for-thread; without it, a multi-axis kernel is `Unsupported` rather
// than silently mis-placed.
// ---------------------------------------------------------------------------

/// The grid a `dispatch(cube_dim = (Wx, Wy))` kernel is launched at for image
/// extents `(w, h)` — the same derivation `conformance_case` and the twin share.
fn grid_for(cube_dim: [u32; 3], e: [u32; 3]) -> Dispatch3 {
    let g = [
        e[0].div_ceil(cube_dim[0]).max(1) * cube_dim[0],
        e[1].div_ceil(cube_dim[1]).max(1) * cube_dim[1],
        e[2].div_ceil(cube_dim[2]).max(1) * cube_dim[2],
    ];
    Dispatch3 { cube_dim, grid: g }
}

/// The §6 clamped blur, interpreted over a real 2-D grid, must match the
/// generated twin bit-for-bit — the third independent semantics implementation
/// agreeing with the first two on the milestone's headline kernel.
///
/// NEGATIVE CONTROL, and the reason this test discriminates: reverting
/// `builtin` to the old pinned `AbsolutePosY => 0` answers makes every thread
/// read row 0, so the interpreted image is a vertical smear of the first row and
/// this comparison fails at once for any `h > 1`. Two of the four shapes here
/// have `h > 1` on purpose.
#[test]
fn interp_box_blur3x3_matches_twin_over_a_2d_grid() {
    let def = box_blur3x3_vericl::kernel_definition();
    let cube_dim = box_blur3x3_vericl::DISPATCH_CUBE_DIM.expect("a dispatch kernel");
    let mut rng = Rng::new(0x2D);
    for [w, h] in [[37u32, 19u32], [8, 8], [1, 1], [3, 33]] {
        let n = (w * h) as usize;
        let inp: Vec<f32> = (0..n).map(|_| rng.f32(-100.0, 100.0)).collect();
        let out0 = vec![0f32; n];

        let d = grid_for(cube_dim, [w, h, 1]);
        let mut twin = out0.clone();
        box_blur3x3_vericl::reference(&inp, &mut twin, w, h, (d.grid[0], d.grid[1], d.grid[2]));

        let got = interp(
            &def,
            Inputs {
                buffers: vec![
                    Buffer::f32("inp", &inp, false),
                    Buffer::f32("out", &out0, true),
                ],
                scalars: vec![ScalarBinding::u32(0, w), ScalarBinding::u32(1, h)],
                cube_dim: cube_dim[0] * cube_dim[1] * cube_dim[2],
                num_threads: d.grid[0] * d.grid[1] * d.grid[2],
                dispatch: Some(d),
            },
        );
        assert_f32_bits(&got[1].as_f32(), &twin, &format!("box_blur3x3 {w}x{h}"));
    }
}

/// The rank-3 kernel, same check — three loops, three axis leaf sets.
#[test]
fn interp_elementwise3d_matches_twin_over_a_3d_grid() {
    let def = elementwise3d_scale_vericl::kernel_definition();
    let cube_dim = elementwise3d_scale_vericl::DISPATCH_CUBE_DIM.expect("a dispatch kernel");
    let mut rng = Rng::new(0x3D);
    for [w, h, dd] in [[7u32, 5u32, 3u32], [8, 8, 4], [1, 1, 1], [3, 3, 9]] {
        let n = (w * h * dd) as usize;
        let inp: Vec<f32> = (0..n).map(|_| rng.f32(-100.0, 100.0)).collect();
        let out0 = vec![0f32; n];

        let d = grid_for(cube_dim, [w, h, dd]);
        let mut twin = out0.clone();
        elementwise3d_scale_vericl::reference(
            &inp,
            &mut twin,
            w,
            h,
            dd,
            (d.grid[0], d.grid[1], d.grid[2]),
        );

        let got = interp(
            &def,
            Inputs {
                buffers: vec![
                    Buffer::f32("inp", &inp, false),
                    Buffer::f32("out", &out0, true),
                ],
                scalars: vec![
                    ScalarBinding::u32(0, w),
                    ScalarBinding::u32(1, h),
                    ScalarBinding::u32(2, dd),
                ],
                cube_dim: cube_dim[0] * cube_dim[1] * cube_dim[2],
                num_threads: d.grid[0] * d.grid[1] * d.grid[2],
                dispatch: Some(d),
            },
        );
        assert_f32_bits(&got[1].as_f32(), &twin, &format!("elementwise3d {w}x{h}x{dd}"));
    }
}

/// A multi-axis kernel presented with **1-D `Inputs`** must report
/// `Unsupported`, never a guessed axis value. This is the fail-closed half of
/// M5: the interpreter would otherwise answer `ABSOLUTE_POS_Y => 0` for every
/// thread and produce a plausible, wrong image.
#[test]
fn interp_rejects_a_multi_axis_kernel_without_a_launch() {
    let def = box_blur3x3_vericl::kernel_definition();
    let n = 64usize;
    let inp = vec![1.0f32; n];
    let out0 = vec![0f32; n];
    match interpret_dispatch(
        &def,
        &Inputs {
            buffers: vec![Buffer::f32("inp", &inp, false), Buffer::f32("out", &out0, true)],
            scalars: vec![ScalarBinding::u32(0, 8), ScalarBinding::u32(1, 8)],
            cube_dim: 64,
            num_threads: 64,
            dispatch: None,
        },
    ) {
        Outcome::Unsupported { reason } => assert!(
            reason.contains("per-axis topology builtin"),
            "the rejection must name the cause: {reason}"
        ),
        other => panic!(
            "a multi-axis kernel with a 1-D launch must be Unsupported, not silently placed at \
             y = z = 0: {other:?}"
        ),
    }
}
