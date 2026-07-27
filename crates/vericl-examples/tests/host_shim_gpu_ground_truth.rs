//! GPU ground-truth verification for `vericl::host_shims` — the load-bearing
//! empirical check behind Feature 1 (verified `cast_from`/`mul_hi` host shims).
//!
//! Unlike `float_method_whitelist.rs` (which cross-checks host methods against
//! `std`), these intrinsics have **GPU-defined** semantics — the u32→f32
//! rounding mode in particular is whatever the hardware does, not what `std`'s
//! intuition says it "should" be. So each shim is pinned bit-for-bit against
//! the REAL intrinsic run in a real `#[cube]` kernel on wgpu (and, where the
//! backend supports it, cubecl-cpu — the `--features cpu` lane), across
//! boundary + random inputs. If a future backend's semantics diverge, this
//! test fails loudly and the shim — not the test — must change to match the GPU.
//!
//! Empirical result (recorded here and in `crates/vericl/src/host_shims.rs`):
//! on wgpu/Metal, `cast_from` u32→f32 and i32→f32 match Rust `as f32`
//! bit-for-bit across the full range (including >2^24 where rounding is
//! observable; both round to nearest, ties to even), and `mul_hi` u32 matches
//! `((a as u64) * (b as u64)) >> 32` bit-for-bit. **No divergence from the
//! shim was found on either lane.**
//!
//! Measured 2026-07-25 (shim-and-small-gate batch), same method:
//!
//! - **usize→f32** matches Rust `x as f32` bit-for-bit over the full u32
//!   addressing domain on both lanes, and so does the real idiom
//!   `f32::cast_from(ABSOLUTE_POS)` (`ABSOLUTE_POS` is `usize`-typed).
//! - **bool→f32** is exactly `true → 1.0` / `false → +0.0` on both lanes.
//! - **fma f32** — 7972 triples (boundary, ±0, ±inf, NaN, subnormal,
//!   cancellation-heavy two-product residuals, random):
//!   - **cubecl-cpu (LLVM `llvm.fma`): 0 divergences, bit-exact everywhere,
//!     subnormals included.**
//!   - **wgpu/Metal (WGSL `fma()`): 0 divergences with no subnormal operand or
//!     result; 78 divergences in the subnormal domain, ALL exactly
//!     flush-to-zero** (`gpu == ftz(fma(ftz a, ftz b, ftz c))`, checked as a
//!     model, not eyeballed). Metal flushes denormals; the host does not. That
//!     is the honest tier boundary and it is asserted, not averaged away.
//!   - Discrimination (why the verdict is not vacuous): the naive host
//!     substitute `a*b + c` would diverge from the real GPU intrinsic on
//!     **3654/7972 triples on wgpu and 3576/7972 on cpu** — ~46% of the corpus.
//!   - Backend note, measured in passing: with three INDEPENDENT operands (this
//!     probe's shape) the UNFUSED `a*b + c` *kernel* agrees with the fused
//!     intrinsic on **0 of 7972** triples' worth of difference on wgpu — i.e.
//!     Metal's shader compiler contracts `a*b + c` back into an FMA — while on
//!     cubecl-cpu the two kernels genuinely differ on 3576 triples. So "just
//!     write `a*b + c` in the twin" is wrong on *both* lanes, for two different
//!     reasons.
//!
//!     That contraction is **operand-shape-dependent**, and the scope matters:
//!     when the addend is the same product (`fma(h, x, -(h*x))` written
//!     unfused as `h*x - product`), common-subexpression elimination collapses
//!     `t - t` to zero *before* contraction can apply, and the unfused kernel
//!     really is unfused on Metal — measured separately by
//!     `fused_and_unfused_residual_kernels_compute_different_functions_on_gpu`
//!     in `tests/conformance.rs`. Neither fact generalizes to the other.
#![cfg(feature = "wgpu")]

use cubecl::prelude::*;
use vericl::host_shims;

#[cube(launch)]
fn cast_u32_f32_kernel(x: &Array<u32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = f32::cast_from(x[ABSOLUTE_POS]);
    }
}

#[cube(launch)]
fn cast_i32_f32_kernel(x: &Array<i32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = f32::cast_from(x[ABSOLUTE_POS]);
    }
}

#[cube(launch)]
fn mulhi_u32_kernel(a: &Array<u32>, b: &Array<u32>, y: &mut Array<u32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = a[ABSOLUTE_POS].mul_hi(b[ABSOLUTE_POS]);
    }
}

/// `usize` source: the u32 payload is widened to the kernel's `usize` (cubecl's
/// `AddressType`, `U32` for any buffer this size) and then cast — so the probe
/// covers the full u32 domain of the addressing regime, not just `0..n`.
#[cube(launch)]
fn cast_usize_f32_kernel(x: &Array<u32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        let u = usize::cast_from(x[ABSOLUTE_POS]);
        y[ABSOLUTE_POS] = f32::cast_from(u);
    }
}

/// The *idiom* the `usize` source exists for: `f32::cast_from(ABSOLUTE_POS)`
/// (`ABSOLUTE_POS` is `usize`-typed in cubecl 0.10 — `constant_usize!`), the
/// generic index-to-float conversion. Verified separately from the widened
/// probe above so the real shape is pinned, not just a synthetic one.
#[cube(launch)]
fn cast_abspos_f32_kernel(y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = f32::cast_from(ABSOLUTE_POS);
    }
}

/// `bool` source: a GPU-computed predicate, cast to f32.
#[cube(launch)]
fn cast_bool_f32_kernel(x: &Array<u32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        let b = x[ABSOLUTE_POS] > 0u32;
        y[ABSOLUTE_POS] = f32::cast_from(b);
    }
}

/// The real fused multiply-add intrinsic (`cubecl::prelude::fma`, a free
/// function — cubecl 0.10 defines no `Float::fma` method or associated form).
#[cube(launch)]
fn fma_f32_kernel(a: &Array<f32>, b: &Array<f32>, c: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = fma(a[ABSOLUTE_POS], b[ABSOLUTE_POS], c[ABSOLUTE_POS]);
    }
}

/// The UNFUSED rewrite, run on the same GPU with the same inputs — the
/// discrimination control. If `a*b + c` were a faithful substitute for `fma`,
/// this kernel would agree with `fma_f32_kernel` everywhere; the test asserts
/// it does *not* (and reports how often), which is what makes the shim's
/// fusedness a measured property rather than an assumption.
#[cube(launch)]
fn mul_add_unfused_kernel(a: &Array<f32>, b: &Array<f32>, c: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = a[ABSOLUTE_POS] * b[ABSOLUTE_POS] + c[ABSOLUTE_POS];
    }
}

fn launch1<R, In, Out, F>(client: &ComputeClient<R>, xs: &[In], run: F) -> Vec<Out>
where
    R: Runtime,
    In: CubeElement,
    Out: CubeElement + Default + Clone,
    F: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, cubecl::server::Handle, cubecl::server::Handle),
{
    let n = xs.len();
    let xh = client.create_from_slice(In::as_bytes(xs));
    let yh = client.create_from_slice(Out::as_bytes(&vec![Out::default(); n]));
    let count = CubeCount::Static((n as u32).div_ceil(64).max(1), 1, 1);
    run(client, count, CubeDim::new_1d(64), xh, yh.clone());
    Out::from_bytes(&client.read_one(yh).unwrap()).to_vec()
}

fn cast_u32<R: Runtime>(client: &ComputeClient<R>, xs: &[u32]) -> Vec<f32> {
    let n = xs.len();
    launch1(client, xs, |c, count, dim, xh, yh| {
        cast_u32_f32_kernel::launch::<R>(c, count, dim, unsafe { ArrayArg::from_raw_parts(xh, n) }, unsafe {
            ArrayArg::from_raw_parts(yh, n)
        });
    })
}

fn cast_i32<R: Runtime>(client: &ComputeClient<R>, xs: &[i32]) -> Vec<f32> {
    let n = xs.len();
    launch1(client, xs, |c, count, dim, xh, yh| {
        cast_i32_f32_kernel::launch::<R>(c, count, dim, unsafe { ArrayArg::from_raw_parts(xh, n) }, unsafe {
            ArrayArg::from_raw_parts(yh, n)
        });
    })
}

fn mulhi<R: Runtime>(client: &ComputeClient<R>, a: &[u32], b: &[u32]) -> Vec<u32> {
    let n = a.len();
    let ah = client.create_from_slice(u32::as_bytes(a));
    let bh = client.create_from_slice(u32::as_bytes(b));
    let yh = client.create_from_slice(u32::as_bytes(&vec![0u32; n]));
    let count = CubeCount::Static((n as u32).div_ceil(64).max(1), 1, 1);
    mulhi_u32_kernel::launch::<R>(
        client,
        count,
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(ah, n) },
        unsafe { ArrayArg::from_raw_parts(bh, n) },
        unsafe { ArrayArg::from_raw_parts(yh.clone(), n) },
    );
    u32::from_bytes(&client.read_one(yh).unwrap()).to_vec()
}

fn u32_probe_inputs() -> Vec<u32> {
    // Boundary values, especially around 2^24 where u32→f32 rounding is
    // observable (integers above 2^24 are not all exactly representable).
    let mut xs: Vec<u32> = vec![
        0, 1, 2, 0x00FF_FFFF, 0x0100_0000, 0x0100_0001, 0x0100_0002, 0x0100_0003, 0x0100_0005,
        0x7FFF_FFFF, 0x8000_0000, 0x8000_0001, 0xFFFF_FFFF, 0xFFFF_FF80, 0xFFFF_FF7F, 16_777_217,
        16_777_219, 33_554_435,
    ];
    let mut s = 0x1234_5678u32;
    for _ in 0..3000 {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        xs.push(s);
    }
    xs
}

fn i32_probe_inputs() -> Vec<i32> {
    let mut xs: Vec<i32> = vec![
        0, 1, -1, i32::MIN, i32::MAX, i32::MIN + 1, -16_777_217, 16_777_217, 16_777_219,
        -16_777_219, 33_554_435, -33_554_435,
    ];
    let mut s = 0x9E37_79B9u32;
    for _ in 0..3000 {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        xs.push(s as i32);
    }
    xs
}

fn cast_abspos<R: Runtime>(client: &ComputeClient<R>, n: usize) -> Vec<f32> {
    let yh = client.create_from_slice(f32::as_bytes(&vec![0f32; n]));
    let count = CubeCount::Static((n as u32).div_ceil(64).max(1), 1, 1);
    cast_abspos_f32_kernel::launch::<R>(client, count, CubeDim::new_1d(64), unsafe {
        ArrayArg::from_raw_parts(yh.clone(), n)
    });
    f32::from_bytes(&client.read_one(yh).unwrap()).to_vec()
}

/// Run both the fused and the unfused kernel over the same triples.
fn fma_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    a: &[f32],
    b: &[f32],
    c: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let n = a.len();
    let count = CubeCount::Static((n as u32).div_ceil(64).max(1), 1, 1);
    let run = |unfused: bool| -> Vec<f32> {
        let ah = client.create_from_slice(f32::as_bytes(a));
        let bh = client.create_from_slice(f32::as_bytes(b));
        let ch = client.create_from_slice(f32::as_bytes(c));
        let yh = client.create_from_slice(f32::as_bytes(&vec![0f32; n]));
        let args = (
            unsafe { ArrayArg::from_raw_parts(ah, n) },
            unsafe { ArrayArg::from_raw_parts(bh, n) },
            unsafe { ArrayArg::from_raw_parts(ch, n) },
            unsafe { ArrayArg::from_raw_parts(yh.clone(), n) },
        );
        if unfused {
            mul_add_unfused_kernel::launch::<R>(
                client,
                count.clone(),
                CubeDim::new_1d(64),
                args.0,
                args.1,
                args.2,
                args.3,
            );
        } else {
            fma_f32_kernel::launch::<R>(
                client,
                count.clone(),
                CubeDim::new_1d(64),
                args.0,
                args.1,
                args.2,
                args.3,
            );
        }
        f32::from_bytes(&client.read_one(yh).unwrap()).to_vec()
    };
    (run(false), run(true))
}

/// Triples covering: exact small values, ±0, infinities, NaN, the extremes of
/// the normal range, **subnormals** (where a flush-to-zero backend would
/// diverge from `std`), and — the load-bearing family — **cancellation-heavy**
/// two-product residuals `fma(hi, x, -(hi*x))`, where the fused and unfused
/// results differ by 100% of the answer.
fn fma_probe_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    let mut push = |x: f32, y: f32, z: f32| {
        a.push(x);
        b.push(y);
        c.push(z);
    };

    // Boundary / special values.
    let specials = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        f32::MIN_POSITIVE,      // smallest normal
        f32::MIN_POSITIVE / 2.0, // subnormal
        f32::from_bits(1),       // smallest subnormal
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        16_777_216.0, // 2^24
        16_777_217.0,
        1e-30,
        1e30,
    ];
    for &x in &specials {
        for &y in &specials {
            push(x, y, 1.0);
            push(x, y, -1.0);
            push(x, y, 0.0);
        }
    }

    // Cancellation-heavy: the two-product residual `fma(hi, x, -(hi*x))`, whose
    // fused value is the EXACT rounding error of `hi*x` and whose unfused value
    // is identically zero. This is the family the private double-single phase
    // accumulator is built on.
    let mut s = 0x5EED_1234u32;
    for _ in 0..1500 {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        // A "step"-shaped mantissa in [2^-8, 2^-6) and an exact integer index.
        let hi = f32::from_bits((0x3B80_0000u32) | (s >> 9));
        let x = ((s % 100_000) as f32).floor() + 1.0;
        push(hi, x, -(hi * x));
        // Near-cancellation at a shifted exponent, both signs.
        push(-hi, x, hi * x);
    }

    // Uniform random bit patterns, restricted to finite values so the majority
    // of the corpus exercises ordinary arithmetic rather than NaN propagation.
    let mut t = 0xA5A5_5A5Au32;
    let mut next = || {
        loop {
            t = t.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = f32::from_bits(t);
            if v.is_finite() {
                return v;
            }
        }
    };
    for _ in 0..2000 {
        let (x, y, z) = (next(), next(), next());
        push(x, y, z);
    }
    // Random values in a tame exponent band (products stay finite), so the
    // corpus is not dominated by overflow-to-infinity.
    for _ in 0..2000 {
        let scale = |v: f32| (v.to_bits() % 1000) as f32 / 97.0 - 5.0;
        let (x, y, z) = (scale(next()), scale(next()), scale(next()));
        push(x, y, z);
    }

    (a, b, c)
}

fn mulhi_probe_inputs() -> (Vec<u32>, Vec<u32>) {
    let mut a: Vec<u32> = vec![0, 1, 0xFFFF_FFFF, 0x8000_0000, 65536, 0x1234_5678, u32::MAX];
    let mut b: Vec<u32> = vec![0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 2, 65536, 0x9ABC_DEF0, u32::MAX];
    let mut s3 = 0xDEAD_BEEFu32;
    let mut s4 = 0xCAFE_BABEu32;
    for _ in 0..3000 {
        s3 = s3.wrapping_mul(1664525).wrapping_add(1013904223);
        s4 = s4.wrapping_mul(22695477).wrapping_add(1);
        a.push(s3);
        b.push(s4);
    }
    (a, b)
}

/// Verify all three shims against the given runtime's real intrinsics.
fn verify_lane<R: Runtime>(client: &ComputeClient<R>, lane: &str) {
    let xs = u32_probe_inputs();
    let gpu = cast_u32(client, &xs);
    for (i, &x) in xs.iter().enumerate() {
        assert_eq!(
            host_shims::cast_from_u32_f32(x).to_bits(),
            gpu[i].to_bits(),
            "[{lane}] cast_from u32->f32 diverged at x={x} (0x{x:08x}): shim={} gpu={}",
            host_shims::cast_from_u32_f32(x),
            gpu[i]
        );
    }

    let is = i32_probe_inputs();
    let gpu_i = cast_i32(client, &is);
    for (i, &x) in is.iter().enumerate() {
        assert_eq!(
            host_shims::cast_from_i32_f32(x).to_bits(),
            gpu_i[i].to_bits(),
            "[{lane}] cast_from i32->f32 diverged at x={x}: shim={} gpu={}",
            host_shims::cast_from_i32_f32(x),
            gpu_i[i]
        );
    }

    let (a, b) = mulhi_probe_inputs();
    let gpu_m = mulhi(client, &a, &b);
    for i in 0..a.len() {
        assert_eq!(
            host_shims::mul_hi_u32(a[i], b[i]),
            gpu_m[i],
            "[{lane}] mul_hi u32 diverged at a={} b={}: shim={} gpu={}",
            a[i],
            b[i],
            host_shims::mul_hi_u32(a[i], b[i]),
            gpu_m[i]
        );
    }

    // --- usize source (AddressType) --------------------------------------
    // Full u32 domain, widened to the kernel's `usize` on the GPU side.
    let gpu_us: Vec<f32> = launch1(client, &xs, |c, count, dim, xh, yh| {
        cast_usize_f32_kernel::launch::<R>(
            c,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(xh, xs.len()) },
            unsafe { ArrayArg::from_raw_parts(yh, xs.len()) },
        );
    });
    for (i, &x) in xs.iter().enumerate() {
        assert_eq!(
            host_shims::cast_from_usize_f32(x as usize).to_bits(),
            gpu_us[i].to_bits(),
            "[{lane}] cast_from usize->f32 diverged at x={x}: shim={} gpu={}",
            host_shims::cast_from_usize_f32(x as usize),
            gpu_us[i]
        );
    }
    // The idiom itself: `f32::cast_from(ABSOLUTE_POS)`.
    let gpu_ap = cast_abspos(client, 4096);
    for (i, &g) in gpu_ap.iter().enumerate() {
        assert_eq!(
            host_shims::cast_from_usize_f32(i).to_bits(),
            g.to_bits(),
            "[{lane}] cast_from(ABSOLUTE_POS) diverged at i={i}: shim={} gpu={g}",
            host_shims::cast_from_usize_f32(i)
        );
    }

    // --- bool source ------------------------------------------------------
    let bxs: Vec<u32> = (0..512u32).map(|i| i % 3).collect(); // mixes 0 / nonzero
    let gpu_b: Vec<f32> = launch1(client, &bxs, |c, count, dim, xh, yh| {
        cast_bool_f32_kernel::launch::<R>(
            c,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(xh, bxs.len()) },
            unsafe { ArrayArg::from_raw_parts(yh, bxs.len()) },
        );
    });
    for (i, &x) in bxs.iter().enumerate() {
        let want = host_shims::cast_from_bool_f32(x > 0);
        assert_eq!(
            want.to_bits(),
            gpu_b[i].to_bits(),
            "[{lane}] cast_from bool->f32 diverged at ({x} > 0): shim={want} gpu={}",
            gpu_b[i]
        );
    }

    // --- fma --------------------------------------------------------------
    verify_fma_lane(client, lane);
}

/// Bit-exact float comparison with the one concession GPU ground truth
/// requires: any NaN equals any NaN (the payload/quiet-bit of a NaN produced by
/// a hardware FMA is not specified by IEEE-754 and differs across backends;
/// the *value* is NaN on both sides, which is what the twin must reproduce).
fn same_f32(shim: f32, gpu: f32) -> bool {
    if shim.is_nan() && gpu.is_nan() {
        return true;
    }
    shim.to_bits() == gpu.to_bits()
}

/// `fma` ground truth.
///
/// **Discrimination** (what makes the bit-exact verdict below non-vacuous): the
/// same triples are also scored against the naive host substitute `a*b + c`.
/// The corpus is only meaningful if that substitute would FAIL loudly here, and
/// the assertion below requires it to.
///
/// The UNFUSED `a*b + c` **kernel** is launched too, and its result reported —
/// a measured backend property rather than a gate (see the module header:
/// on wgpu/Metal the shader compiler contracts `a*b + c` back into an FMA, so
/// the two kernels agree; that is exactly why the *host* twin has to be fused).
fn verify_fma_lane<R: Runtime>(client: &ComputeClient<R>, lane: &str) {
    let (a, b, c) = fma_probe_inputs();
    let (gpu, gpu_unfused) = fma_gpu(client, &a, &b, &c);

    // Subnormal ("denormal") flush-to-zero model: the divergence class this
    // probe actually finds on wgpu/Metal. `ftz` is what a flush-to-zero backend
    // does to a value at an operand or a result boundary.
    let is_sub = |v: f32| v != 0.0 && v.abs() < f32::MIN_POSITIVE;
    let ftz = |v: f32| if is_sub(v) { 0.0f32.copysign(v) } else { v };

    let mut normal_domain: Vec<usize> = Vec::new();
    let mut subnormal_domain: Vec<usize> = Vec::new();
    let mut model_misses: Vec<usize> = Vec::new();
    for i in 0..a.len() {
        let shim = host_shims::fma_f32(a[i], b[i], c[i]);
        if same_f32(shim, gpu[i]) {
            continue;
        }
        if is_sub(a[i]) || is_sub(b[i]) || is_sub(c[i]) || is_sub(shim) || is_sub(gpu[i]) {
            subnormal_domain.push(i);
            // Does the flush-to-zero model explain it exactly?
            let modeled = ftz(host_shims::fma_f32(ftz(a[i]), ftz(b[i]), ftz(c[i])));
            if !same_f32(modeled, gpu[i]) {
                model_misses.push(i);
            }
        } else {
            normal_domain.push(i);
        }
    }

    // Would the naive `a*b + c` host substitute pass this probe? It must not.
    let naive_wrong = (0..a.len()).filter(|&i| !same_f32(a[i] * b[i] + c[i], gpu[i])).count();
    let kernel_unfused_diff =
        (0..a.len()).filter(|&i| !same_f32(gpu[i], gpu_unfused[i])).count();
    eprintln!(
        "[{lane}] fma ground truth over {} triples: shim-vs-GPU divergences = {} in the \
         normal domain, {} in the subnormal domain ({} of those NOT explained by \
         flush-to-zero). Discrimination: the naive host `a*b + c` would diverge from the \
         GPU on {} triples. Backend note: the unfused `a*b + c` KERNEL differs from the \
         fused intrinsic on {} of {} triples.",
        a.len(),
        normal_domain.len(),
        subnormal_domain.len(),
        model_misses.len(),
        naive_wrong,
        kernel_unfused_diff,
        a.len()
    );
    assert!(
        naive_wrong > a.len() / 100,
        "[{lane}] the probe corpus does not discriminate a fused shim from the naive \
         `a*b + c` (only {naive_wrong} of {} triples would catch it) — the bit-exact \
         verdict would be vacuous",
        a.len()
    );

    let fmt = |i: usize| {
        format!(
            "a={:e}(0x{:08x}) b={:e}(0x{:08x}) c={:e}(0x{:08x}) shim={:e}(0x{:08x}) \
             gpu={:e}(0x{:08x})",
            a[i],
            a[i].to_bits(),
            b[i],
            b[i].to_bits(),
            c[i],
            c[i].to_bits(),
            host_shims::fma_f32(a[i], b[i], c[i]),
            host_shims::fma_f32(a[i], b[i], c[i]).to_bits(),
            gpu[i],
            gpu[i].to_bits()
        )
    };

    // TIER 1 (bit-exact): no operand and no result subnormal. Any divergence
    // here is a FINDING that changes the shim, not the test.
    assert!(
        normal_domain.is_empty(),
        "[{lane}] fma shim diverged from the real GPU intrinsic on {} of {} triples with no \
         subnormal operand or result. This is a FINDING, not a flake: the shim is pinned to \
         GPU semantics, so either the shim changes to match the hardware or this domain \
         leaves the bit-exact tier.\nFirst divergences:\n  {}",
        normal_domain.len(),
        a.len(),
        normal_domain.iter().take(10).map(|&i| fmt(i)).collect::<Vec<_>>().join("\n  ")
    );

    // TIER 2 (documented divergence): the subnormal domain. Whatever divergence
    // exists there must be EXACTLY flush-to-zero — a characterized backend
    // property, not an unexplained mismatch.
    assert!(
        model_misses.is_empty(),
        "[{lane}] {} subnormal-domain divergences are NOT explained by flush-to-zero. The \
         documented boundary in `host_shims::fma_f32` claims they all are; that claim is \
         now false and must be re-derived.\nFirst unexplained:\n  {}",
        model_misses.len(),
        model_misses.iter().take(10).map(|&i| fmt(i)).collect::<Vec<_>>().join("\n  ")
    );
    if subnormal_domain.is_empty() {
        eprintln!(
            "[{lane}] fma subnormal domain: no divergences at all — this lane is bit-exact \
             everywhere, subnormals included (tier 1 covers the whole corpus here)."
        );
    } else {
        eprintln!(
            "[{lane}] fma subnormal domain: {} divergences, ALL exactly flush-to-zero \
             (documented tier-2 boundary).",
            subnormal_domain.len()
        );
    }
}

#[test]
fn shims_match_wgpu_ground_truth() {
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    verify_lane(&client, "wgpu");
}

/// Second lane (`--features cpu`): the cubecl-cpu backend. If the two backends
/// ever disagree with each other or with the shim, this is a FINDING, not
/// something to average away — the shim is pinned to the GPU semantics, and a
/// divergence must be documented and resolved by matching the intended target.
#[cfg(feature = "cpu")]
#[test]
fn shims_match_cpu_ground_truth() {
    let client = cubecl::cpu::CpuRuntime::client(&Default::default());
    verify_lane(&client, "cpu");
}
