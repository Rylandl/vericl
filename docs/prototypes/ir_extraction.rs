//! Prototype: extract the CubeCL IR (`Scope`) for a kernel with ZERO
//! ComputeClient / Runtime / device involved. Validates the API path for
//! VeriCL's "IR-level kernel identity" milestone.

use cubecl::prelude::*;
use sha2::{Digest, Sha256};

#[cube(launch)]
pub fn axpy(alpha: f32, x: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
    }
}

/// Build the KernelDefinition for `axpy` with no client/device/runtime at all.
fn build_axpy_ir() -> KernelDefinition {
    let mut builder = KernelBuilder::default();
    // Static, backend-independent properties -- no device query.
    builder.runtime_properties(Default::default());
    // Deliberately DO NOT call builder.device_properties(...): that's the
    // only KernelBuilder method that needs a real ComputeClient, and it's
    // only consulted by `comptime`-based device-feature queries, which are
    // outside VeriCL's kernel subset anyway.

    // Register how `usize`/`isize` (ABSOLUTE_POS, .len(), indices) map to
    // concrete storage types. This is plain data (AddressType::U32/U64), no
    // device query -- KernelSettings::default() picks AddressType::U32.
    AddressType::U32.register(&mut builder.scope);

    let alpha = <f32 as LaunchArg>::expand(&Default::default(), &mut builder);
    let x = <Array<f32> as LaunchArg>::expand(
        &ArrayCompilationArg { inplace: None },
        &mut builder,
    );
    let y = <Array<f32> as LaunchArg>::expand_output(
        &ArrayCompilationArg { inplace: None },
        &mut builder,
    );

    // Call the macro-generated `expand` function directly -- this is the
    // literal IR-building trace of the kernel body.
    axpy::expand(&mut builder.scope, alpha, x, y);

    builder.build(KernelSettings::default())
}

/// A tiny deterministic content hash over the parts of KernelDefinition that
/// define semantics (buffers/scalars/cube_dim/body), skipping `options`
/// (kernel_name is a debug label, not semantic).
struct ShaHasher(Sha256);
impl core::hash::Hasher for ShaHasher {
    fn finish(&self) -> u64 {
        unimplemented!("write-only hasher")
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
}

fn content_hash(def: &KernelDefinition) -> String {
    use core::hash::{Hash, Hasher};
    let mut h = ShaHasher(Sha256::new());
    // Scope derives std::hash::Hash directly (see cubecl-ir/src/scope.rs) --
    // drive it with our SHA256-backed Hasher for a stable content hash.
    def.body.hash(&mut h);
    // KernelArg/ScalarKernelArg only derive Serialize (no std::hash::Hash),
    // so fold their serialized bytes into the same hasher. serde field order
    // is fixed by struct definition order, so this is deterministic too.
    h.write(&serde_json::to_vec(&def.buffers).unwrap());
    h.write(&serde_json::to_vec(&def.scalars).unwrap());
    def.cube_dim.hash(&mut h);
    format!("sha256:{:x}", h.0.finalize())
}

fn main() {
    let def = build_axpy_ir();

    println!("=== Display (pretty-printed IR) ===");
    println!("cube_dim = {:?}", def.cube_dim);
    println!("buffers = {:?}", def.buffers.iter().map(|b| (b.id, b.visibility)).collect::<Vec<_>>());
    println!("scalars = {:?}", def.scalars);
    println!("body = {}", def.body);

    println!("\n=== serde_json (Scope) ===");
    let json = serde_json::to_string_pretty(&def.body).expect("Scope is Serialize");
    println!("{json}");

    println!("\n=== content hash (Rust Hash -> SHA256) ===");
    let hash1 = content_hash(&def);
    println!("{hash1}");

    // Determinism check: rebuild the IR again within the same process.
    //
    // NOTE: `Scope`'s *derived* PartialEq is a trap for this purpose --
    // `Scope` embeds `Allocator`, and `Allocator`'s hand-written PartialEq is
    // `Rc::ptr_eq` on its internal counters/pools, i.e. REFERENCE identity,
    // not value identity. Two structurally-identical `Scope`s built from two
    // separate `KernelBuilder`s will therefore always compare unequal via
    // `==`/derive(PartialEq). Confirmed below: this assertion fails even
    // though the printed IR (see above) and the content hash are identical.
    let def2 = build_axpy_ir();
    let derived_eq = def.body == def2.body;
    println!(
        "\nderived PartialEq (Scope == Scope) across separate builds: {derived_eq} \
         (expected false -- Allocator::eq is Rc::ptr_eq, a known trap)"
    );

    // The *custom* std::hash::Hash impl on Scope (see cubecl-ir/src/scope.rs)
    // explicitly hashes only semantic fields (instructions/locals/etc.) and
    // skips allocator/typemap/debug/runtime_properties/modes/properties.
    // That is the mechanism VeriCL should build identity on top of.
    let hash2 = content_hash(&def2);
    assert_eq!(hash1, hash2, "content hash should be stable across rebuilds");
    println!("content hash (custom Hash impl, same process, rebuilt twice): OK, hashes match");
}
