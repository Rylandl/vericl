//! De-risk binary: vanilla CubeCL 0.10 axpy on wgpu (Metal), no vericl involvement.

use cubecl::prelude::*;

#[cube(launch)]
fn axpy(alpha: f32, x: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
    }
}

fn main() {
    type R = cubecl::wgpu::WgpuRuntime;
    let device = Default::default();
    let client = R::client(&device);

    let n = 1027usize; // deliberately not a multiple of the cube dim
    let alpha = 2.5f32;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    let x_handle = client.create_from_slice(f32::as_bytes(&x));
    let y_handle = client.create_from_slice(f32::as_bytes(&y));

    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);

    axpy::launch::<R>(
        &client,
        CubeCount::Static(cube_count, 1, 1),
        CubeDim::new_1d(cube_dim),
        alpha,
        unsafe { ArrayArg::from_raw_parts(x_handle, n) },
        unsafe { ArrayArg::from_raw_parts(y_handle.clone(), n) },
    );

    let bytes = client.read_one(y_handle).unwrap();
    let result = f32::from_bytes(&bytes);

    let mut max_abs_diff = 0f32;
    for i in 0..n {
        let expected = alpha * x[i] + y[i];
        max_abs_diff = max_abs_diff.max((result[i] - expected).abs());
    }
    println!(
        "runtime={:?} n={n} max_abs_diff={max_abs_diff}",
        R::name(&client)
    );
    assert_eq!(max_abs_diff, 0.0, "GPU axpy diverged from CPU expectation");
    println!("vanilla axpy OK");
}
