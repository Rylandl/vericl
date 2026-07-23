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
