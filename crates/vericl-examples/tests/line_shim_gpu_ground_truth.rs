//! GPU ground-truth verification for `vericl::Line<T, W>` — the load-bearing
//! empirical check behind Deliverable A (the pinned lane-array twin,
//! design-line-vector.md §4, §6, §7; round-8 risk 2).
//!
//! Every op the reference twin's `Line` calls is pinned against the REAL
//! `Vector<_, N>` intrinsic run in a real `#[cube]` kernel on wgpu (and, behind
//! `--features cpu`, cubecl-cpu). No op reaches the twin surface without a row
//! here. The I/O is scalar throughout (design §4.4): a flat `&[f32]`/`&[u32]` of
//! `lines * W` scalars is uploaded, the kernel is launched at the pinned
//! vectorization `W` dispatching `lines` threads, and the flat readback is
//! reshaped into `Line`s and compared lane-by-lane.
//!
//! **Two verification tiers, matching where GPU semantics are defined:**
//!
//! - **Bit-exact (`Line` shim == GPU, exactly).** For the algebraically-exact
//!   ops — arithmetic (`+ - * /`, neg), bitwise (`& | ^ << >>`, `count_ones`),
//!   splat (`new`/`fill`), per-lane comparison, and the exact float ops
//!   (`abs`/`floor`/`ceil`/`round`/`trunc`) — a vec-`W` op *is* `W` independent
//!   scalar ops with no cross-lane coupling and no reordering, so the lane-array
//!   twin reproduces the GPU value bit-for-bit (design §4.5, §6). A single
//!   mismatch is a FINDING and fails the test: this is exactly the trap in
//!   round-8 risk 2 (a splat the GPU implements as a broadcast-with-conversion,
//!   a `round` with a different tie rule, a comparison whose true-lane encoding
//!   is not `1`).
//!
//! - **Vector-faithfulness (GPU vec op == GPU per-lane scalar op, exactly).** For
//!   the transcendentals — the full unary whitelist `sqrt recip sin cos tan asin
//!   acos atan sinh cosh tanh exp ln to_degrees to_radians` and the binary
//!   `powf atan2 hypot`, plus `div` — whose GPU value legitimately differs from
//!   `std`'s at the last ULP (handled by `compare(abs=…)` at the conformance
//!   layer, the same as the scalar path — see `float_method_whitelist.rs`), the
//!   *vector-specific* claim to verify is that the GPU computes a
//!   `Vector<f32, N>` math op as `N` independent scalar ops. We launch the vector
//!   kernel and a scalar `Array<f32>` kernel on the identical flat data and
//!   assert their outputs are **bit-identical** — proving no lane coupling and no
//!   distinct vector implementation. The twin's per-lane reuse of the scalar
//!   whitelist is then faithful by construction (design §7 "Math (unary)"). A
//!   twin-vs-GPU ULP sanity bound is also checked. (`min`/`max` are algebraic
//!   selects, so they are verified in the bit-exact tier instead.)
//!
//! Ops the twin surface exposes that derive from the above rather than carrying a
//! separate row: the compound-assign forms (`+=`/`&=`/…, the same per-lane op as
//! their value form), `clamp` (a `min`/`max` composition), `powi` (repeated
//! `*`), the constant-splat constructors (`from_int`/`min_value`/`max_value`/
//! `zeroed`, the same broadcast as `new`/`fill`), and lane index `v[j]` (plain
//! `[T; W]` indexing). `empty()` is uninitialized on the GPU and modeled as zero
//! by convention (not bit-exact-verifiable); a kernel reading it before writing
//! is caught by the differential compare, not masked here.
//!
//! **Empirical result on wgpu/Metal (Apple M3):** every bit-exact op (arithmetic
//! `+ - *`, neg, splat/fill, all bitwise, `count_ones`, all six comparisons,
//! `abs`/`floor`/`ceil`/`round`/`trunc`) matched the `Line` shim bit-for-bit with
//! zero mismatches, and every `Vector<f32, 4>` transcendental (plus `div`)
//! equalled its scalar `Array<f32>` output bit-for-bit — no lane coupling, no
//! vector-specific implementation.
//!
//! **FINDINGS (GPU semantics that differ from a naive host op, documented not
//! averaged away):**
//! 1. `f32 /` (division) is **not correctly-rounded** on Metal: it differs from
//!    the host's correctly-rounded `/` by ≤1 ULP. This is the same class of
//!    legitimate float divergence as an FMA contraction — the `compare(abs=…)`
//!    tolerance covers it at the conformance layer — so `div` is verified in the
//!    vec-vs-scalar tier (GPU vec == GPU scalar, bit-exact) with a ≤2-ULP
//!    twin bound, NOT bit-exact against the twin. `+`, `-`, `*` ARE correctly
//!    rounded and bit-exact (matching design §6's `vec_addmul` result, which used
//!    only `+`/`*`).
//! 2. `Vector::<u32, N>::cast_from(bool_vec)` encodes a true lane as `1u32` (the
//!    comparison readback relies on this; pinned by the comparison rows).
#![cfg(feature = "wgpu")]

use cubecl::prelude::*;
use vericl::Line;

const W: usize = 4;

// Launch/twin function-pointer aliases (keep the per-op case tables readable and
// clippy's `type_complexity` lint happy).
type VecBinLaunch<R> =
    fn(&ComputeClient<R>, CubeCount, CubeDim, usize, ArrayArg<R>, ArrayArg<R>, ArrayArg<R>);
type VecUnaryLaunch<R> = fn(&ComputeClient<R>, CubeCount, CubeDim, usize, ArrayArg<R>, ArrayArg<R>);
type ScalarUnaryLaunch<R> = fn(&ComputeClient<R>, CubeCount, CubeDim, ArrayArg<R>, ArrayArg<R>);
type ScalarBinLaunch<R> =
    fn(&ComputeClient<R>, CubeCount, CubeDim, ArrayArg<R>, ArrayArg<R>, ArrayArg<R>);
type CmpTwin = fn(Line<f32, W>, Line<f32, W>) -> Line<bool, W>;
type UnaryTwin = fn(Line<f32, W>) -> Line<f32, W>;
type BinTwin = fn(Line<f32, W>, Line<f32, W>) -> Line<f32, W>;

// ===========================================================================
// Kernels — the real Vector<_, N> intrinsics, one per op.
// ===========================================================================

macro_rules! vec_binary_f32 {
    ($name:ident, $lhs:ident, $rhs:ident, $body:expr) => {
        #[cube(launch_unchecked)]
        fn $name<N: Size>(
            a: &Array<Vector<f32, N>>,
            b: &Array<Vector<f32, N>>,
            out: &mut Array<Vector<f32, N>>,
        ) {
            if ABSOLUTE_POS < out.len() {
                let $lhs = a[ABSOLUTE_POS];
                let $rhs = b[ABSOLUTE_POS];
                out[ABSOLUTE_POS] = $body;
            }
        }
    };
}

vec_binary_f32!(k_add, x, y, x + y);
vec_binary_f32!(k_sub, x, y, x - y);
vec_binary_f32!(k_mul, x, y, x * y);
vec_binary_f32!(k_div, x, y, x / y);

#[cube(launch_unchecked)]
fn k_neg<N: Size>(a: &Array<Vector<f32, N>>, out: &mut Array<Vector<f32, N>>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = -a[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
fn k_splat<N: Size>(s: f32, out: &mut Array<Vector<f32, N>>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = Vector::new(s);
    }
}

#[cube(launch_unchecked)]
fn k_fill<N: Size>(s: f32, a: &Array<Vector<f32, N>>, out: &mut Array<Vector<f32, N>>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS].fill(s);
    }
}

macro_rules! vec_binary_u32 {
    ($name:ident, $lhs:ident, $rhs:ident, $body:expr) => {
        #[cube(launch_unchecked)]
        fn $name<N: Size>(
            a: &Array<Vector<u32, N>>,
            b: &Array<Vector<u32, N>>,
            out: &mut Array<Vector<u32, N>>,
        ) {
            if ABSOLUTE_POS < out.len() {
                let $lhs = a[ABSOLUTE_POS];
                let $rhs = b[ABSOLUTE_POS];
                out[ABSOLUTE_POS] = $body;
            }
        }
    };
}

vec_binary_u32!(k_and, x, y, x & y);
vec_binary_u32!(k_or, x, y, x | y);
vec_binary_u32!(k_xor, x, y, x ^ y);
vec_binary_u32!(k_shl, x, y, x << y);
vec_binary_u32!(k_shr, x, y, x >> y);

#[cube(launch_unchecked)]
fn k_count_ones<N: Size>(a: &Array<Vector<u32, N>>, out: &mut Array<Vector<u32, N>>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS].count_ones();
    }
}

// Comparisons: the bool-vector result is cast to a u32 vector for readback
// (the GPU's true-lane encoding is pinned below to be 1u32).
macro_rules! vec_cmp {
    ($name:ident, $method:ident) => {
        #[cube(launch_unchecked)]
        fn $name<N: Size>(
            a: &Array<Vector<f32, N>>,
            b: &Array<Vector<f32, N>>,
            out: &mut Array<Vector<u32, N>>,
        ) {
            if ABSOLUTE_POS < out.len() {
                let c = a[ABSOLUTE_POS].$method(b[ABSOLUTE_POS]);
                out[ABSOLUTE_POS] = Vector::<u32, N>::cast_from(c);
            }
        }
    };
}

vec_cmp!(k_less_than, less_than);
vec_cmp!(k_greater_than, greater_than);
vec_cmp!(k_less_equal, less_equal);
vec_cmp!(k_greater_equal, greater_equal);
vec_cmp!(k_equal, equal);
vec_cmp!(k_not_equal, not_equal);

macro_rules! vec_unary_f32 {
    ($name:ident, $val:ident, $body:expr) => {
        #[cube(launch_unchecked)]
        fn $name<N: Size>(a: &Array<Vector<f32, N>>, out: &mut Array<Vector<f32, N>>) {
            if ABSOLUTE_POS < out.len() {
                let $val = a[ABSOLUTE_POS];
                out[ABSOLUTE_POS] = $body;
            }
        }
    };
}

vec_unary_f32!(k_abs, v, v.abs());
vec_unary_f32!(k_floor, v, v.floor());
vec_unary_f32!(k_ceil, v, v.ceil());
vec_unary_f32!(k_round, v, v.round());
vec_unary_f32!(k_trunc, v, v.trunc());
// The full unary math whitelist (design §7 "Math (unary)"), applied lane-wise.
vec_unary_f32!(k_sqrt, v, v.sqrt());
vec_unary_f32!(k_recip, v, v.recip());
vec_unary_f32!(k_sin, v, v.sin());
vec_unary_f32!(k_cos, v, v.cos());
vec_unary_f32!(k_tan, v, v.tan());
vec_unary_f32!(k_asin, v, v.asin());
vec_unary_f32!(k_acos, v, v.acos());
vec_unary_f32!(k_atan, v, v.atan());
vec_unary_f32!(k_sinh, v, v.sinh());
vec_unary_f32!(k_cosh, v, v.cosh());
vec_unary_f32!(k_tanh, v, v.tanh());
vec_unary_f32!(k_exp, v, v.exp());
vec_unary_f32!(k_ln, v, v.ln());
vec_unary_f32!(k_to_degrees, v, v.to_degrees());
vec_unary_f32!(k_to_radians, v, v.to_radians());

macro_rules! vec_binary_math_f32 {
    ($name:ident, $lhs:ident, $rhs:ident, $body:expr) => {
        #[cube(launch_unchecked)]
        fn $name<N: Size>(
            a: &Array<Vector<f32, N>>,
            b: &Array<Vector<f32, N>>,
            out: &mut Array<Vector<f32, N>>,
        ) {
            if ABSOLUTE_POS < out.len() {
                let $lhs = a[ABSOLUTE_POS];
                let $rhs = b[ABSOLUTE_POS];
                out[ABSOLUTE_POS] = $body;
            }
        }
    };
}

vec_binary_math_f32!(k_powf, x, y, x.powf(y));
vec_binary_math_f32!(k_atan2, x, y, x.atan2(y));
vec_binary_math_f32!(k_hypot, x, y, x.hypot(y));
vec_binary_math_f32!(k_min, x, y, x.min(y));
vec_binary_math_f32!(k_max, x, y, x.max(y));

// Scalar counterparts for the transcendental vec-vs-scalar equivalence.
macro_rules! scalar_unary_f32 {
    ($name:ident, $val:ident, $body:expr) => {
        #[cube(launch_unchecked)]
        fn $name(a: &Array<f32>, out: &mut Array<f32>) {
            if ABSOLUTE_POS < out.len() {
                let $val = a[ABSOLUTE_POS];
                out[ABSOLUTE_POS] = $body;
            }
        }
    };
}

scalar_unary_f32!(s_sqrt, v, f32::sqrt(v));
scalar_unary_f32!(s_recip, v, f32::recip(v));
scalar_unary_f32!(s_sin, v, f32::sin(v));
scalar_unary_f32!(s_cos, v, f32::cos(v));
scalar_unary_f32!(s_tan, v, f32::tan(v));
scalar_unary_f32!(s_asin, v, f32::asin(v));
scalar_unary_f32!(s_acos, v, f32::acos(v));
scalar_unary_f32!(s_atan, v, f32::atan(v));
scalar_unary_f32!(s_sinh, v, f32::sinh(v));
scalar_unary_f32!(s_cosh, v, f32::cosh(v));
scalar_unary_f32!(s_tanh, v, f32::tanh(v));
scalar_unary_f32!(s_exp, v, f32::exp(v));
scalar_unary_f32!(s_ln, v, f32::ln(v));
scalar_unary_f32!(s_to_degrees, v, f32::to_degrees(v));
scalar_unary_f32!(s_to_radians, v, f32::to_radians(v));

macro_rules! scalar_binary_f32 {
    ($name:ident, $lhs:ident, $rhs:ident, $body:expr) => {
        #[cube(launch_unchecked)]
        fn $name(a: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>) {
            if ABSOLUTE_POS < out.len() {
                let $lhs = a[ABSOLUTE_POS];
                let $rhs = b[ABSOLUTE_POS];
                out[ABSOLUTE_POS] = $body;
            }
        }
    };
}

scalar_binary_f32!(s_powf, x, y, f32::powf(x, y));
scalar_binary_f32!(s_atan2, x, y, f32::atan2(x, y));
scalar_binary_f32!(s_hypot, x, y, f32::hypot(x, y));

// Scalar `/` counterpart: GPU f32 division is not correctly-rounded on Metal
// (a ≤1-ULP gap vs host, the same legitimate float divergence as an FMA
// contraction, handled by `compare(abs=…)`), so `div` is verified in the
// vec-vs-scalar-equivalence tier rather than bit-exact against the twin.
#[cube(launch_unchecked)]
fn s_div(a: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] / b[ABSOLUTE_POS];
    }
}

// ===========================================================================
// Launch harness (scalar I/O, pinned width W, `lines` threads).
// ===========================================================================

fn launch_dims(lines: usize) -> (CubeCount, CubeDim) {
    let cube_dim = 64u32;
    let count = (lines as u32).div_ceil(cube_dim).max(1);
    (CubeCount::Static(count, 1, 1), CubeDim::new_1d(cube_dim))
}

/// Binary `T`-vector kernel: two flat `[T]` inputs → one flat `[T]` output.
fn run_binary<R, T, F>(client: &ComputeClient<R>, a: &[T], b: &[T], launch: F) -> Vec<T>
where
    R: Runtime,
    T: CubeElement + Default + Clone,
    F: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, usize, ArrayArg<R>, ArrayArg<R>, ArrayArg<R>),
{
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let lines = n / W;
    let ha = client.create_from_slice(T::as_bytes(a));
    let hb = client.create_from_slice(T::as_bytes(b));
    let ho = client.create_from_slice(T::as_bytes(&vec![T::default(); n]));
    let (count, dim) = launch_dims(lines);
    launch(client, count, dim, W, unsafe { ArrayArg::from_raw_parts(ha, n) }, unsafe {
        ArrayArg::from_raw_parts(hb, n)
    }, unsafe { ArrayArg::from_raw_parts(ho.clone(), n) });
    T::from_bytes(&client.read_one(ho).unwrap()).to_vec()
}

/// Unary `T`-vector kernel: one flat `[T]` input → one flat `[T]` output.
fn run_unary<R, T, F>(client: &ComputeClient<R>, a: &[T], launch: F) -> Vec<T>
where
    R: Runtime,
    T: CubeElement + Default + Clone,
    F: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, usize, ArrayArg<R>, ArrayArg<R>),
{
    let n = a.len();
    let lines = n / W;
    let ha = client.create_from_slice(T::as_bytes(a));
    let ho = client.create_from_slice(T::as_bytes(&vec![T::default(); n]));
    let (count, dim) = launch_dims(lines);
    launch(client, count, dim, W, unsafe { ArrayArg::from_raw_parts(ha, n) }, unsafe {
        ArrayArg::from_raw_parts(ho.clone(), n)
    });
    T::from_bytes(&client.read_one(ho).unwrap()).to_vec()
}

/// A comparison kernel: two flat `[f32]` inputs → one flat `[u32]` output.
fn run_cmp<R, F>(client: &ComputeClient<R>, a: &[f32], b: &[f32], launch: F) -> Vec<u32>
where
    R: Runtime,
    F: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, usize, ArrayArg<R>, ArrayArg<R>, ArrayArg<R>),
{
    let n = a.len();
    let lines = n / W;
    let ha = client.create_from_slice(f32::as_bytes(a));
    let hb = client.create_from_slice(f32::as_bytes(b));
    let ho = client.create_from_slice(u32::as_bytes(&vec![0u32; n]));
    let (count, dim) = launch_dims(lines);
    launch(client, count, dim, W, unsafe { ArrayArg::from_raw_parts(ha, n) }, unsafe {
        ArrayArg::from_raw_parts(hb, n)
    }, unsafe { ArrayArg::from_raw_parts(ho.clone(), n) });
    u32::from_bytes(&client.read_one(ho).unwrap()).to_vec()
}

/// A scalar `Array<f32>` unary kernel (no vectorization arg).
fn run_scalar_unary<R, F>(client: &ComputeClient<R>, a: &[f32], launch: F) -> Vec<f32>
where
    R: Runtime,
    F: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, ArrayArg<R>, ArrayArg<R>),
{
    let n = a.len();
    let ha = client.create_from_slice(f32::as_bytes(a));
    let ho = client.create_from_slice(f32::as_bytes(&vec![0f32; n]));
    let (count, dim) = launch_dims(n);
    launch(client, count, dim, unsafe { ArrayArg::from_raw_parts(ha, n) }, unsafe {
        ArrayArg::from_raw_parts(ho.clone(), n)
    });
    f32::from_bytes(&client.read_one(ho).unwrap()).to_vec()
}

/// A scalar `Array<f32>` binary kernel (no vectorization arg).
fn run_scalar_binary<R, F>(client: &ComputeClient<R>, a: &[f32], b: &[f32], launch: F) -> Vec<f32>
where
    R: Runtime,
    F: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, ArrayArg<R>, ArrayArg<R>, ArrayArg<R>),
{
    let n = a.len();
    let ha = client.create_from_slice(f32::as_bytes(a));
    let hb = client.create_from_slice(f32::as_bytes(b));
    let ho = client.create_from_slice(f32::as_bytes(&vec![0f32; n]));
    let (count, dim) = launch_dims(n);
    launch(client, count, dim, unsafe { ArrayArg::from_raw_parts(ha, n) }, unsafe {
        ArrayArg::from_raw_parts(hb, n)
    }, unsafe { ArrayArg::from_raw_parts(ho.clone(), n) });
    f32::from_bytes(&client.read_one(ho).unwrap()).to_vec()
}

// ===========================================================================
// Twin (Line) computation + reshape helpers.
// ===========================================================================

fn as_lines<T: Copy + Default>(flat: &[T]) -> Vec<Line<T, W>> {
    flat.chunks_exact(W).map(|c| Line(<[T; W]>::try_from(c).unwrap())).collect()
}

fn flat<T: Copy>(lines: Vec<Line<T, W>>) -> Vec<T> {
    lines.into_iter().flat_map(|l| l.0).collect()
}

fn twin_binary<T: Copy + Default>(
    a: &[T],
    b: &[T],
    f: impl Fn(Line<T, W>, Line<T, W>) -> Line<T, W>,
) -> Vec<T> {
    flat(as_lines(a).into_iter().zip(as_lines(b)).map(|(x, y)| f(x, y)).collect())
}

fn twin_unary<T: Copy + Default>(a: &[T], f: impl Fn(Line<T, W>) -> Line<T, W>) -> Vec<T> {
    flat(as_lines(a).into_iter().map(f).collect())
}

// ===========================================================================
// Comparators.
// ===========================================================================

fn assert_bit_exact_f32(op: &str, lane: &str, gpu: &[f32], twin: &[f32]) {
    assert_eq!(gpu.len(), twin.len(), "[{lane}] {op}: length mismatch");
    for i in 0..gpu.len() {
        assert_eq!(
            gpu[i].to_bits(),
            twin[i].to_bits(),
            "[{lane}] {op} diverged at line={} lane={}: gpu={} twin={} (FINDING: the Line shim \
             does not match the GPU intrinsic — fix the shim to match the GPU, never the test)",
            i / W,
            i % W,
            gpu[i],
            twin[i]
        );
    }
}

fn assert_bit_exact_u32(op: &str, lane: &str, gpu: &[u32], twin: &[u32]) {
    assert_eq!(gpu.len(), twin.len(), "[{lane}] {op}: length mismatch");
    for i in 0..gpu.len() {
        assert_eq!(
            gpu[i], twin[i],
            "[{lane}] {op} diverged at line={} lane={}: gpu={} twin={} (FINDING: fix the shim)",
            i / W,
            i % W,
            gpu[i],
            twin[i]
        );
    }
}

fn max_ulp_f32(a: &[f32], b: &[f32]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x.to_bits() as i64 - y.to_bits() as i64).unsigned_abs())
        .max()
        .unwrap_or(0)
}

// ===========================================================================
// Input generators.
// ===========================================================================

struct Xs(u64);
impl Xs {
    fn new(seed: u64) -> Self {
        Xs(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// In `[-range, range)`.
    fn f32(&mut self, range: f32) -> f32 {
        ((self.next_u64() >> 40) as f32 / 16_777_216.0) * (2.0 * range) - range
    }
    /// In `[lo, hi)`.
    fn f32_pos(&mut self, lo: f32, hi: f32) -> f32 {
        lo + ((self.next_u64() >> 40) as f32 / 16_777_216.0) * (hi - lo)
    }
    fn u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

fn gen_f32(n: usize, seed: u64, range: f32) -> Vec<f32> {
    let mut x = Xs::new(seed);
    (0..n).map(|_| x.f32(range)).collect()
}
fn gen_pos_f32(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut x = Xs::new(seed);
    (0..n).map(|_| x.f32_pos(lo, hi)).collect()
}
fn gen_u32(n: usize, seed: u64) -> Vec<u32> {
    let mut x = Xs::new(seed);
    (0..n).map(|_| x.u32()).collect()
}

// The line counts the harness sweeps (1 up to a multi-cube dispatch).
const LINE_COUNTS: [usize; 5] = [1, 3, 7, 64, 257];

// ===========================================================================
// The verification body (run per backend lane).
// ===========================================================================

fn verify_lane<R: Runtime>(client: &ComputeClient<R>, lane: &str) {
    for &lines in &LINE_COUNTS {
        let n = lines * W;

        // ---- arithmetic (bit-exact) ----
        let a = gen_f32(n, 0x0A11 ^ lines as u64, 4.0);
        // divisor kept away from zero so `/` is well-defined everywhere.
        let b = gen_pos_f32(n, 0x0B22 ^ lines as u64, 1.0, 5.0);

        let gpu = run_binary(client, &a, &b, |c, cc, cd, w, x, y, o| unsafe {
            k_add::launch_unchecked::<R>(c, cc, cd, w, x, y, o)
        });
        assert_bit_exact_f32("add", lane, &gpu, &twin_binary(&a, &b, |x, y| x + y));

        let gpu = run_binary(client, &a, &b, |c, cc, cd, w, x, y, o| unsafe {
            k_sub::launch_unchecked::<R>(c, cc, cd, w, x, y, o)
        });
        assert_bit_exact_f32("sub", lane, &gpu, &twin_binary(&a, &b, |x, y| x - y));

        let gpu = run_binary(client, &a, &b, |c, cc, cd, w, x, y, o| unsafe {
            k_mul::launch_unchecked::<R>(c, cc, cd, w, x, y, o)
        });
        assert_bit_exact_f32("mul", lane, &gpu, &twin_binary(&a, &b, |x, y| x * y));

        let gpu = run_unary(client, &a, |c, cc, cd, w, x, o| unsafe {
            k_neg::launch_unchecked::<R>(c, cc, cd, w, x, o)
        });
        assert_bit_exact_f32("neg", lane, &gpu, &twin_unary(&a, |x| -x));

        // `div`: FINDING — GPU f32 division is not correctly-rounded on Metal
        // (≤1 ULP vs the host's correctly-rounded `/`), so it is verified like a
        // transcendental: the GPU vec op equals the GPU scalar op bit-for-bit (no
        // vector-specific divergence), and the twin is within a small ULP bound
        // (the same legitimate float gap `compare(abs=…)` covers on the scalar
        // path). It must NOT be asserted bit-exact against the twin.
        let gpu_vec = run_binary(client, &a, &b, |c, cc, cd, w, x, y, o| unsafe {
            k_div::launch_unchecked::<R>(c, cc, cd, w, x, y, o)
        });
        let gpu_scalar = run_scalar_binary(client, &a, &b, |c, cc, cd, x, y, o| unsafe {
            s_div::launch_unchecked::<R>(c, cc, cd, x, y, o)
        });
        assert_bit_exact_f32("div (vec==scalar)", lane, &gpu_vec, &gpu_scalar);
        let twin_div = twin_binary(&a, &b, |x, y| x / y);
        let ulp = max_ulp_f32(&gpu_vec, &twin_div);
        assert!(
            ulp <= 2,
            "[{lane}] div: twin diverged from GPU by {ulp} ULP (expected the ≤1-ULP \
             non-correctly-rounded-division gap, not a vector-model error)"
        );

        // ---- splat / fill (bit-exact) ----
        let s = 3.5f32;
        let gpu = run_unary(client, &a, |c, cc, cd, w, _x, o| unsafe {
            // splat ignores its input; reuse the unary harness's output buffer.
            k_splat::launch_unchecked::<R>(c, cc, cd, w, s, o)
        });
        assert_bit_exact_f32("splat_new", lane, &gpu, &vec![s; n]);

        let gpu = run_unary(client, &a, |c, cc, cd, w, x, o| unsafe {
            k_fill::launch_unchecked::<R>(c, cc, cd, w, s, x, o)
        });
        assert_bit_exact_f32("fill", lane, &gpu, &twin_unary(&a, |x| x.fill(s)));

        // ---- bitwise (bit-exact) ----
        let ua = gen_u32(n, 0xB1A5 ^ lines as u64);
        // shift amounts kept in [0, 32) so `<<`/`>>` are defined per lane.
        let mut xu = Xs::new(0x5417 ^ lines as u64);
        let ub: Vec<u32> = (0..n).map(|_| xu.u32() % 32).collect();

        let gpu = run_binary(client, &ua, &ub, |c, cc, cd, w, x, y, o| unsafe {
            k_and::launch_unchecked::<R>(c, cc, cd, w, x, y, o)
        });
        assert_bit_exact_u32("bitand", lane, &gpu, &twin_binary(&ua, &ub, |x, y| x & y));

        let gpu = run_binary(client, &ua, &ub, |c, cc, cd, w, x, y, o| unsafe {
            k_or::launch_unchecked::<R>(c, cc, cd, w, x, y, o)
        });
        assert_bit_exact_u32("bitor", lane, &gpu, &twin_binary(&ua, &ub, |x, y| x | y));

        let gpu = run_binary(client, &ua, &ub, |c, cc, cd, w, x, y, o| unsafe {
            k_xor::launch_unchecked::<R>(c, cc, cd, w, x, y, o)
        });
        assert_bit_exact_u32("bitxor", lane, &gpu, &twin_binary(&ua, &ub, |x, y| x ^ y));

        let gpu = run_binary(client, &ua, &ub, |c, cc, cd, w, x, y, o| unsafe {
            k_shl::launch_unchecked::<R>(c, cc, cd, w, x, y, o)
        });
        assert_bit_exact_u32("shl", lane, &gpu, &twin_binary(&ua, &ub, |x, y| x << y));

        let gpu = run_binary(client, &ua, &ub, |c, cc, cd, w, x, y, o| unsafe {
            k_shr::launch_unchecked::<R>(c, cc, cd, w, x, y, o)
        });
        assert_bit_exact_u32("shr", lane, &gpu, &twin_binary(&ua, &ub, |x, y| x >> y));

        let gpu = run_unary(client, &ua, |c, cc, cd, w, x, o| unsafe {
            k_count_ones::launch_unchecked::<R>(c, cc, cd, w, x, o)
        });
        assert_bit_exact_u32("count_ones", lane, &gpu, &twin_unary(&ua, |x| x.count_ones()));

        // ---- comparison (bit-exact; true-lane == 1u32 pinned here) ----
        // Force some exact-equality lanes so `equal`/`less_equal` exercise ties.
        let mut ca = a.clone();
        for i in (0..n).step_by(3) {
            ca[i] = b[i];
        }
        let cmp_cases: [(&str, VecBinLaunch<R>, CmpTwin); 6] = [
            ("less_than", |c, cc, cd, w, x, y, o| unsafe { k_less_than::launch_unchecked::<R>(c, cc, cd, w, x, y, o) }, |x, y| x.less_than(y)),
            ("greater_than", |c, cc, cd, w, x, y, o| unsafe { k_greater_than::launch_unchecked::<R>(c, cc, cd, w, x, y, o) }, |x, y| x.greater_than(y)),
            ("less_equal", |c, cc, cd, w, x, y, o| unsafe { k_less_equal::launch_unchecked::<R>(c, cc, cd, w, x, y, o) }, |x, y| x.less_equal(y)),
            ("greater_equal", |c, cc, cd, w, x, y, o| unsafe { k_greater_equal::launch_unchecked::<R>(c, cc, cd, w, x, y, o) }, |x, y| x.greater_equal(y)),
            ("equal", |c, cc, cd, w, x, y, o| unsafe { k_equal::launch_unchecked::<R>(c, cc, cd, w, x, y, o) }, |x, y| x.equal(y)),
            ("not_equal", |c, cc, cd, w, x, y, o| unsafe { k_not_equal::launch_unchecked::<R>(c, cc, cd, w, x, y, o) }, |x, y| x.not_equal(y)),
        ];
        for (name, klaunch, twinf) in cmp_cases {
            let gpu = run_cmp(client, &ca, &b, klaunch);
            let twin: Vec<u32> = as_lines(&ca)
                .into_iter()
                .zip(as_lines(&b))
                .flat_map(|(x, y)| twinf(x, y).0.map(|t| t as u32))
                .collect();
            assert_bit_exact_u32(name, lane, &gpu, &twin);
        }

        // ---- exact float ops (bit-exact) ----
        // Values spanning tie points and both signs for round/trunc.
        let m = gen_f32(n, 0x77E5 ^ lines as u64, 6.0);
        let exact: [(&str, VecUnaryLaunch<R>, UnaryTwin); 5] = [
            ("abs", |c, cc, cd, w, x, o| unsafe { k_abs::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |v| v.abs()),
            ("floor", |c, cc, cd, w, x, o| unsafe { k_floor::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |v| v.floor()),
            ("ceil", |c, cc, cd, w, x, o| unsafe { k_ceil::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |v| v.ceil()),
            ("round", |c, cc, cd, w, x, o| unsafe { k_round::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |v| v.round()),
            ("trunc", |c, cc, cd, w, x, o| unsafe { k_trunc::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |v| v.trunc()),
        ];
        for (name, klaunch, twinf) in exact {
            let gpu = run_unary(client, &m, klaunch);
            assert_bit_exact_f32(name, lane, &gpu, &twin_unary(&m, twinf));
        }

        // ---- unary math whitelist: GPU vec == GPU scalar (bit-exact), twin
        // within ULP. The input range [0.05, 0.95] is valid for EVERY whitelist
        // unary (asin/acos need [-1,1]; sqrt/ln need > 0; recip needs != 0), so a
        // single set covers all 15 — every op the twin surface exposes has a row.
        let p = gen_pos_f32(n, 0x7A11 ^ lines as u64, 0.05, 0.95);
        let transc: [(&str, VecUnaryLaunch<R>, ScalarUnaryLaunch<R>, UnaryTwin); 15] = [
            ("sqrt", |c, cc, cd, w, x, o| unsafe { k_sqrt::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_sqrt::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.sqrt()),
            ("recip", |c, cc, cd, w, x, o| unsafe { k_recip::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_recip::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.recip()),
            ("sin", |c, cc, cd, w, x, o| unsafe { k_sin::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_sin::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.sin()),
            ("cos", |c, cc, cd, w, x, o| unsafe { k_cos::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_cos::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.cos()),
            ("tan", |c, cc, cd, w, x, o| unsafe { k_tan::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_tan::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.tan()),
            ("asin", |c, cc, cd, w, x, o| unsafe { k_asin::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_asin::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.asin()),
            ("acos", |c, cc, cd, w, x, o| unsafe { k_acos::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_acos::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.acos()),
            ("atan", |c, cc, cd, w, x, o| unsafe { k_atan::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_atan::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.atan()),
            ("sinh", |c, cc, cd, w, x, o| unsafe { k_sinh::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_sinh::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.sinh()),
            ("cosh", |c, cc, cd, w, x, o| unsafe { k_cosh::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_cosh::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.cosh()),
            ("tanh", |c, cc, cd, w, x, o| unsafe { k_tanh::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_tanh::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.tanh()),
            ("exp", |c, cc, cd, w, x, o| unsafe { k_exp::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_exp::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.exp()),
            ("ln", |c, cc, cd, w, x, o| unsafe { k_ln::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_ln::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.ln()),
            ("to_degrees", |c, cc, cd, w, x, o| unsafe { k_to_degrees::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_to_degrees::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.to_degrees()),
            ("to_radians", |c, cc, cd, w, x, o| unsafe { k_to_radians::launch_unchecked::<R>(c, cc, cd, w, x, o) }, |c, cc, cd, x, o| unsafe { s_to_radians::launch_unchecked::<R>(c, cc, cd, x, o) }, |v| v.to_radians()),
        ];
        for (name, kvec, kscalar, twinf) in transc {
            let gpu_vec = run_unary(client, &p, kvec);
            let gpu_scalar = run_scalar_unary(client, &p, kscalar);
            // Vector-specific claim: the GPU computes the vec op as W scalar ops.
            assert_bit_exact_f32(&format!("{name} (vec==scalar)"), lane, &gpu_vec, &gpu_scalar);
            // Sanity: the Line twin (std per-lane) is close to the GPU value — the
            // same last-ULP transcendental gap the scalar conformance path has.
            let twin = twin_unary(&p, twinf);
            let ulp = max_ulp_f32(&gpu_vec, &twin);
            assert!(
                ulp <= 4096,
                "[{lane}] {name}: Line twin diverged from GPU by {ulp} ULP (expected a small \
                 transcendental gap, not a vector-model error)"
            );
        }

        // ---- binary math: powf/atan2/hypot (vec==scalar tier), min/max
        // (bit-exact — algebraic selects). Two positive operand sets.
        let q = gen_pos_f32(n, 0x9B12 ^ lines as u64, 0.05, 0.95);
        let bin_transc: [(&str, VecBinLaunch<R>, ScalarBinLaunch<R>, BinTwin); 3] = [
            ("powf", |c, cc, cd, w, x, y, o| unsafe { k_powf::launch_unchecked::<R>(c, cc, cd, w, x, y, o) }, |c, cc, cd, x, y, o| unsafe { s_powf::launch_unchecked::<R>(c, cc, cd, x, y, o) }, |x, y| x.powf(y)),
            ("atan2", |c, cc, cd, w, x, y, o| unsafe { k_atan2::launch_unchecked::<R>(c, cc, cd, w, x, y, o) }, |c, cc, cd, x, y, o| unsafe { s_atan2::launch_unchecked::<R>(c, cc, cd, x, y, o) }, |x, y| x.atan2(y)),
            ("hypot", |c, cc, cd, w, x, y, o| unsafe { k_hypot::launch_unchecked::<R>(c, cc, cd, w, x, y, o) }, |c, cc, cd, x, y, o| unsafe { s_hypot::launch_unchecked::<R>(c, cc, cd, x, y, o) }, |x, y| x.hypot(y)),
        ];
        for (name, kvec, kscalar, twinf) in bin_transc {
            let gpu_vec = run_binary(client, &p, &q, kvec);
            let gpu_scalar = run_scalar_binary(client, &p, &q, kscalar);
            assert_bit_exact_f32(&format!("{name} (vec==scalar)"), lane, &gpu_vec, &gpu_scalar);
            let ulp = max_ulp_f32(&gpu_vec, &twin_binary(&p, &q, twinf));
            assert!(
                ulp <= 4096,
                "[{lane}] {name}: Line twin diverged from GPU by {ulp} ULP (expected a small \
                 transcendental gap, not a vector-model error)"
            );
        }
        // min/max are algebraic (a select), correctly rounded -> bit-exact.
        let m2 = gen_f32(n, 0x33CD ^ lines as u64, 6.0);
        let gpu = run_binary(client, &m, &m2, |c, cc, cd, w, x, y, o| unsafe {
            k_min::launch_unchecked::<R>(c, cc, cd, w, x, y, o)
        });
        assert_bit_exact_f32("min", lane, &gpu, &twin_binary(&m, &m2, |x, y| x.min(y)));
        let gpu = run_binary(client, &m, &m2, |c, cc, cd, w, x, y, o| unsafe {
            k_max::launch_unchecked::<R>(c, cc, cd, w, x, y, o)
        });
        assert_bit_exact_f32("max", lane, &gpu, &twin_binary(&m, &m2, |x, y| x.max(y)));
    }
}

#[test]
fn line_ops_match_wgpu_ground_truth() {
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    verify_lane(&client, "wgpu");
}

/// Second lane (`--features cpu`): the cubecl-cpu backend. A disagreement between
/// backends, or between a backend and the shim, is a FINDING to document and
/// resolve by matching the intended target — never something to average away.
#[cfg(feature = "cpu")]
#[test]
fn line_ops_match_cpu_ground_truth() {
    let client = cubecl::cpu::CpuRuntime::client(&Default::default());
    verify_lane(&client, "cpu");
}
