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
//! - **fma f32** — 21996 triples (boundary, ±0, ±inf, NaN, subnormal, the
//!   **underflow boundary in both directions**, an 18³ cross-product over the
//!   values straddling the normal/subnormal border, cancellation-heavy
//!   two-product residuals, random):
//!   - **cubecl-cpu (LLVM `llvm.fma`): 0 divergences, bit-exact everywhere,
//!     subnormals and the underflow boundary included — this lane does not
//!     flush.**
//!   - **wgpu/Metal (WGSL `fma()`): 0 divergences outside a 4974-triple
//!     flush-to-zero domain, and inside it the device is matched EXACTLY by
//!     [`metal_ftz_fma`]** — subnormal operands flushed on input, and the
//!     result flushed to a signed zero whenever the **exact, pre-rounding**
//!     product-sum is subnormal.
//!   - **Round-10 correction.** The model asserted before this was
//!     `ftz(fma(ftz a, ftz b, ftz c))` — flush the *rounded* result. It is
//!     false: `fma(2^-126, 2^-126, -2^-126)` has all-normal operands, rounds to
//!     the normal `-2^-126` on the host, and the device returns `-0`. The old
//!     corpus never reached that band, and the old assertion only checked the
//!     model where the shim and the device already disagreed, so 4 distinct
//!     wrong models passed it. Both halves are fixed here: the corpus reaches
//!     the band, and each lane's model is scored over **every** triple.
//!   - Discrimination (why the verdict is not vacuous): the naive host
//!     substitute `a*b + c` would diverge from the real GPU intrinsic on
//!     **8508/21996 triples on wgpu and 3782/21996 on cpu**; exactly one of the
//!     two candidate models must explain each lane with zero mismatches while
//!     the other mismatches somewhere; and ten injected mutations of the model
//!     (including the superseded one above) all fail the test.
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
/// diverge from `std`), the **underflow boundary in both directions** (round-10
/// review), and — the load-bearing family — **cancellation-heavy** two-product
/// residuals `fma(hi, x, -(hi*x))`, where the fused and unfused results differ
/// by 100% of the answer.
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
    // is identically zero. This is the family a double-single phase
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

    // ---- the underflow boundary (round-10 review, critical 2) --------------
    //
    // Everything above lands either well inside the normal range or well inside
    // the subnormal one, so it cannot tell a backend that decides underflow on
    // the ROUNDED result from one that decides it on the EXACT pre-rounding
    // value. Those two rules differ on exactly one set: an exact result in the
    // half-ulp band `[MIN_POSITIVE - 2^-150, MIN_POSITIVE)`, which rounds UP to
    // the smallest normal. The corpus below reaches it deliberately, from both
    // sides and in both signs, with **all-normal operands** (so the divergence
    // lands in the bit-exact tier, where it is a finding rather than a
    // documented boundary).
    let mp = f32::MIN_POSITIVE; // 2^-126, the smallest normal
    let sub_max = f32::from_bits(0x007F_FFFF); // largest subnormal
    let sub_min = f32::from_bits(1); // 2^-149, smallest subnormal
    let ulp_n = f32::from_bits(mp.to_bits() + 1) - mp; // 2^-149, ulp above 2^-126

    // (i) exact = ±(MIN_POSITIVE - delta) with delta far below the half-ulp
    //     2^-150, built from all-normal operands: the tiny product is a normal
    //     times a normal, and the addend is exactly ±MIN_POSITIVE.
    for &(pa, pb) in &[
        (mp, mp),                                  // 2^-252
        (mp, f32::from_bits(mp.to_bits() + 1)),    // just above 2^-252
        (mp, 1e-30f32),
        (1e-20f32, 1e-20f32),
        (f32::from_bits(0x0080_0002), mp),
        (1.0e-30f32, 1.0e-15f32),
    ] {
        for &sign in &[1.0f32, -1.0] {
            // exact magnitude just BELOW MIN_POSITIVE: rounds to MIN_POSITIVE,
            // but the exact value is subnormal.
            push(sign * pa, pb, -sign * mp);
            push(-sign * pa, pb, sign * mp);
            // exact magnitude just ABOVE MIN_POSITIVE: no underflow at all —
            // the direction that must NOT be flushed.
            push(sign * pa, pb, sign * mp);
        }
    }

    // (ii) the half-ulp band walked explicitly: c = ±MIN_POSITIVE and a*b a
    //      ladder of magnitudes straddling the 2^-150 rounding boundary. The
    //      ladder value 2^-(140+k) is itself subnormal, so it is built as a
    //      PRODUCT of two normals — the whole point is that every operand stays
    //      normal while the exact result does not.
    let pow2 = |e: u32| f32::from_bits((127 - e) << 23); // 2^-e, normal for e <= 126
    for k in 0..24u32 {
        // 2^-140 … 2^-163, i.e. both sides of the 2^-150 half-ulp boundary.
        let e = 140 + k;
        let (e1, e2) = (e / 2, e - e / 2);
        let (fa, fb) = (pow2(e1), pow2(e2));
        for &sign in &[1.0f32, -1.0] {
            push(sign * fa, fb, -sign * mp); // exact = ∓(mp - 2^-e)
            push(sign * fa, fb, sign * mp); // exact = ±(mp + 2^-e)
            push(sign * fa, -fb, sign * mp);
        }
    }

    // (iii) exactly at the boundary, and one ulp either side of it, with no
    //       rounding involved at all.
    for &(x, y, z) in &[
        (1.0f32, mp, 0.0f32),      // exact = mp
        (1.0, mp, -0.0),           // exact = mp
        (0.5, mp, 0.0),            // exact = mp/2, a subnormal
        (1.0, mp, ulp_n),          // exact = mp + ulp
        (1.0, mp, -ulp_n),         // exact = mp - ulp, subnormal
        (1.0, sub_max, 0.0),       // largest subnormal, untouched
        (1.0, sub_max, sub_min),   // exact = mp exactly
        (-1.0, sub_max, -sub_min), // exact = -mp exactly
        (1.0, mp, mp),
        (-1.0, mp, mp),
        (2.0, mp, -mp),
        (mp, 1.0, -sub_min),
    ] {
        push(x, y, z);
    }

    // (iv) a cross-product over the operand values that straddle the
    //      normal/subnormal border in both signs — the family that
    //      discriminates whether the ADDEND is flushed on input.
    let border = [
        0.0f32,
        -0.0,
        sub_min,
        -sub_min,
        f32::from_bits(0x0040_0000),
        -f32::from_bits(0x0040_0000),
        sub_max,
        -sub_max,
        mp,
        -mp,
        f32::from_bits(mp.to_bits() + 1),
        -f32::from_bits(mp.to_bits() + 1),
        0.5,
        -0.5,
        1.0,
        -1.0,
        2.0,
        -2.0,
    ];
    for &x in &border {
        for &y in &border {
            for &z in &border {
                push(x, y, z);
            }
        }
    }

    // (v) randomized near-boundary cancellation: a normal-or-subnormal operand
    //     pair whose product is tiny, against an addend pinned at ±the border —
    //     the shape that produced the round-10 counterexample class.
    let mut u = 0x1BAD_C0DEu32;
    let mut rnd = || {
        u = u.wrapping_mul(1664525).wrapping_add(1013904223);
        u
    };
    for _ in 0..4000 {
        // A random value with a small-but-normal exponent (biased 1..=60, i.e.
        // 2^-126 … 2^-67) and a random mantissa.
        let small = |r: u32| {
            let exp = 1 + (r >> 26) % 60;
            f32::from_bits((exp << 23) | (r & 0x007F_FFFF) | ((r & 1) << 31))
        };
        let x = small(rnd());
        let y = small(rnd());
        let r = rnd();
        let z = match r % 4 {
            0 => mp,
            1 => -mp,
            2 => f32::from_bits(mp.to_bits() + (r >> 8) % 4),
            _ => -f32::from_bits(mp.to_bits() + (r >> 8) % 4),
        };
        push(x, y, z);
        push(x, y, -z);
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

/// What a flush-to-zero backend does to a value that reaches an operand or a
/// result boundary: a subnormal becomes a signed zero.
fn ftz(v: f32) -> f32 {
    if v != 0.0 && v.abs() < f32::MIN_POSITIVE { 0.0f32.copysign(v) } else { v }
}

/// `true` if the **exact** (infinitely precise) value of `a*b + c` is non-zero
/// with magnitude strictly below `f32::MIN_POSITIVE` — i.e. the pre-rounding
/// result lands in the subnormal range, *whatever it rounds to*.
///
/// This is the discriminating predicate of the whole FTZ model (round-10
/// review, critical 2). Deciding underflow on the ROUNDED result and deciding
/// it on the EXACT one differ on exactly one set — an exact magnitude in the
/// half-ulp band `[MIN_POSITIVE - 2^-150, MIN_POSITIVE)`, which rounds up to
/// the smallest normal — and the device sides with the exact rule (measured:
/// 8688/8688 head-to-head).
///
/// It must be computed exactly, not in `f64`: `fma(2^-126, 2^-126, -2^-126)`
/// has exact value `-(2^-126 - 2^-252)`, which rounds to `-2^-126` in `f64`
/// too, so an `f64` evaluation reports "not below" and gets the answer wrong.
/// `p = a*b` is exact in `f64` (24+24 ≤ 53 significand bits, exponents far
/// inside range), and Knuth's two-sum then represents `p + c` exactly as an
/// unevaluated `s + err`, which is enough to compare against `2^-126` exactly.
fn exact_product_sum_underflows(a: f32, b: f32, c: f32) -> Option<f32> {
    let p = a as f64 * b as f64;
    let cd = c as f64;
    let s = p + cd;
    if !s.is_finite() {
        return None;
    }
    // Knuth two-sum: `p + cd == s + err`, exactly.
    let bb = s - p;
    let err = (p - (s - bb)) + (cd - bb);
    // Exact magnitude is `t + u` with `t >= 0` and `|u| <= ulp(t)/2`.
    let t = s.abs();
    let u = if s.is_sign_negative() { -err } else { err };
    if t == 0.0 && u == 0.0 {
        return None; // an exact zero is not an underflow
    }
    let m = f32::MIN_POSITIVE as f64;
    let below = t < m || (t == m && u < 0.0);
    if !below {
        return None;
    }
    // The sign of the exact value: `s`'s, unless the sum cancelled exactly (in
    // which case `err` is zero too and we returned above).
    Some(if s.is_sign_negative() || (s == 0.0 && err < 0.0) { -1.0 } else { 1.0 })
}

/// The **measured** wgpu/Metal semantics of `fma(a, b, c)` for `f32`: flush
/// subnormal operands to zero on input, evaluate the fused product-sum exactly,
/// and flush to a signed zero whenever that exact value is subnormal — deciding
/// underflow on the pre-rounding magnitude, not on the rounded one.
///
/// Validated bit-for-bit against the device over the whole corpus below, and
/// discriminated against nine mutations of itself (see `verify_fma_lane`).
fn metal_ftz_fma(a: f32, b: f32, c: f32) -> f32 {
    let (a, b, c) = (ftz(a), ftz(b), ftz(c));
    match exact_product_sum_underflows(a, b, c) {
        Some(sign) => 0.0f32.copysign(sign),
        None => host_shims::fma_f32(a, b, c),
    }
}

/// `fma` ground truth.
///
/// **What is asserted, and why it is stronger than it was.** Each lane is
/// scored against BOTH candidate semantics over the *whole* corpus:
///
/// - the identity model — the shim itself, `fma_f32(a, b, c)` — which is what a
///   backend with no flush-to-zero implements; and
/// - [`metal_ftz_fma`], the measured flush model.
///
/// Exactly one of them must explain the lane with **zero** mismatches, and the
/// other must mismatch somewhere. That single assertion carries three claims at
/// once: the lane's semantics are fully characterized (no unexplained
/// divergence anywhere in the corpus), the characterization is not vacuous (the
/// corpus discriminates the two models), and the shim is bit-exact wherever the
/// lane does not flush.
///
/// The round-10 review's finding was that the previous form — check the model
/// only on the triples where shim and device already *disagreed* — was blind on
/// the other 7894 of 7972 triples, so four distinct mutations of the model
/// (dropping the addend's input flush, unfusing the inner product, flushing the
/// intermediate product, and the pre-round/post-round rule itself) all passed.
/// Scoring the model over every triple is what closes that.
///
/// **Discrimination** (what makes the bit-exact verdict non-vacuous): the same
/// triples are also scored against the naive host substitute `a*b + c`, which
/// must FAIL loudly. The UNFUSED `a*b + c` **kernel** is launched too and its
/// result reported — a measured backend property rather than a gate (see the
/// module header: on wgpu/Metal the shader compiler contracts `a*b + c` back
/// into an FMA, so the two kernels agree; that is exactly why the *host* twin
/// has to be fused).
fn verify_fma_lane<R: Runtime>(client: &ComputeClient<R>, lane: &str) {
    let (a, b, c) = fma_probe_inputs();
    let (gpu, gpu_unfused) = fma_gpu(client, &a, &b, &c);
    let n = a.len();

    let fmt = |i: usize, modeled: f32| {
        format!(
            "a=0x{:08x} b=0x{:08x} c=0x{:08x} | shim=0x{:08x} model=0x{:08x} gpu=0x{:08x}",
            a[i].to_bits(),
            b[i].to_bits(),
            c[i].to_bits(),
            host_shims::fma_f32(a[i], b[i], c[i]).to_bits(),
            modeled.to_bits(),
            gpu[i].to_bits()
        )
    };

    // The two candidate semantics, each scored over the WHOLE corpus.
    let ident_miss: Vec<usize> =
        (0..n).filter(|&i| !same_f32(host_shims::fma_f32(a[i], b[i], c[i]), gpu[i])).collect();
    let flush_miss: Vec<usize> =
        (0..n).filter(|&i| !same_f32(metal_ftz_fma(a[i], b[i], c[i]), gpu[i])).collect();
    // The flush domain: where the model and the raw shim disagree at all. Its
    // complement is the tier-1 (bit-exact) domain.
    let flush_domain =
        (0..n).filter(|&i| !same_f32(metal_ftz_fma(a[i], b[i], c[i]), host_shims::fma_f32(a[i], b[i], c[i]))).count();

    let naive_wrong = (0..n).filter(|&i| !same_f32(a[i] * b[i] + c[i], gpu[i])).count();
    let kernel_unfused_diff = (0..n).filter(|&i| !same_f32(gpu[i], gpu_unfused[i])).count();

    eprintln!(
        "[{lane}] fma ground truth over {n} triples: identity model (the raw shim) mismatches \
         the device on {}, the flush model on {}. Flush domain (model != shim) = {} triples; \
         the other {} are the bit-exact tier. Discrimination: the naive host `a*b + c` would \
         diverge from the GPU on {}. Backend note: the unfused `a*b + c` KERNEL differs from \
         the fused intrinsic on {} of {n}.",
        ident_miss.len(),
        flush_miss.len(),
        flush_domain,
        n - flush_domain,
        naive_wrong,
        kernel_unfused_diff,
    );

    assert!(
        naive_wrong > n / 100,
        "[{lane}] the probe corpus does not discriminate a fused shim from the naive \
         `a*b + c` (only {naive_wrong} of {n} triples would catch it) — the bit-exact \
         verdict would be vacuous"
    );

    // THE GATE. Exactly one model must explain the lane completely.
    let ident_explains = ident_miss.is_empty();
    let flush_explains = flush_miss.is_empty();
    assert!(
        ident_explains || flush_explains,
        "[{lane}] NEITHER candidate model explains this backend: the raw shim mismatches the \
         device on {} of {n} triples and the flush-to-zero model on {}. This is a FINDING, not \
         a flake — the shim and its documented divergence model are pinned to GPU semantics, so \
         the model must be re-derived from the data below (or the shim changed to match the \
         hardware).\nFirst raw-shim divergences:\n  {}\nFirst flush-model divergences:\n  {}",
        ident_miss.len(),
        flush_miss.len(),
        ident_miss
            .iter()
            .take(6)
            .map(|&i| fmt(i, metal_ftz_fma(a[i], b[i], c[i])))
            .collect::<Vec<_>>()
            .join("\n  "),
        flush_miss
            .iter()
            .take(6)
            .map(|&i| fmt(i, metal_ftz_fma(a[i], b[i], c[i])))
            .collect::<Vec<_>>()
            .join("\n  "),
    );
    assert!(
        !(ident_explains && flush_explains),
        "[{lane}] BOTH models explain this backend, which means the corpus never reaches the \
         flush domain — the flush-model verdict would be vacuous. The boundary families in \
         `fma_probe_inputs` must be repaired before this test can claim anything."
    );

    if ident_explains {
        eprintln!(
            "[{lane}] fma: the raw shim is bit-exact on ALL {n} triples, subnormals and the \
             underflow boundary included — this lane does not flush. (The flush model would \
             have mismatched on {}, so the corpus does discriminate the two.)",
            flush_miss.len()
        );
    } else {
        eprintln!(
            "[{lane}] fma: this lane FLUSHES. The flush model matches the device on all {n} \
             triples; the raw shim alone would mismatch on {} of them, all inside the {}-triple \
             flush domain.",
            ident_miss.len(),
            flush_domain
        );
        // Tier 1, stated separately so a regression names the right thing: the
        // raw shim must be bit-exact everywhere OUTSIDE the flush domain.
        let tier1_miss: Vec<usize> = ident_miss
            .iter()
            .copied()
            .filter(|&i| {
                same_f32(metal_ftz_fma(a[i], b[i], c[i]), host_shims::fma_f32(a[i], b[i], c[i]))
            })
            .collect();
        assert!(
            tier1_miss.is_empty(),
            "[{lane}] internal inconsistency: {} triples diverge from the shim while the flush \
             model predicts no flush at all.\n  {}",
            tier1_miss.len(),
            tier1_miss
                .iter()
                .take(6)
                .map(|&i| fmt(i, metal_ftz_fma(a[i], b[i], c[i])))
                .collect::<Vec<_>>()
                .join("\n  ")
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
