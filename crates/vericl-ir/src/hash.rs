//! IR-level kernel identity: a deterministic SHA-256 content hash over the
//! semantic parts of a [`KernelDefinition`] (body, buffers, scalars,
//! cube_dim). Adapted from docs/prototypes/ir_extraction.rs — see
//! docs/ir-research.md §2 for the validated findings this implementation
//! relies on.

use cubecl::ir::{Branch, Instruction, Operation, Scope};
use cubecl::prelude::KernelDefinition;
use sha2::{Digest, Sha256};

/// A `std::hash::Hasher` that forwards every byte written to it into a
/// running SHA-256 digest. Write-only: `finish()` is never called by
/// `Hash::hash`, only used to drive a hasher whose output we read from the
/// wrapped `Sha256` directly.
struct ShaHasher(Sha256);

impl core::hash::Hasher for ShaHasher {
    fn finish(&self) -> u64 {
        unimplemented!("write-only hasher")
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
}

/// Recursively clear `Instruction.source_loc` on every instruction in
/// `scope`, including inside nested branch scopes. `source_loc` is `None`
/// unless CubeCL's `debug_symbols` cfg is on (which vericl never enables),
/// but normalizing explicitly means identity can never depend on it even if
/// that default changes — absolute source paths must not leak into a
/// content hash that's supposed to travel between machines.
fn normalize_source_locs(scope: &mut Scope) {
    for inst in &mut scope.instructions {
        inst.source_loc = None;
        for nested in nested_scopes_mut(inst) {
            normalize_source_locs(nested);
        }
    }
}

/// The child `Scope`s directly nested inside one instruction's operation, if
/// any (branch bodies). Empty for straight-line operations.
fn nested_scopes_mut(inst: &mut Instruction) -> Vec<&mut Scope> {
    match &mut inst.operation {
        Operation::Branch(b) => match b {
            Branch::If(if_) => vec![&mut if_.scope],
            Branch::IfElse(ie) => vec![&mut ie.scope_if, &mut ie.scope_else],
            Branch::Switch(sw) => {
                let mut scopes = vec![&mut sw.scope_default];
                scopes.extend(sw.cases.iter_mut().map(|(_, s)| s));
                scopes
            }
            Branch::RangeLoop(rl) => vec![&mut rl.scope],
            Branch::Loop(l) => vec![&mut l.scope],
            Branch::Return | Branch::Break | Branch::Unreachable => vec![],
        },
        _ => vec![],
    }
}

/// Content hash of a [`KernelDefinition`]'s semantics: `"sha256:<hex>"`.
///
/// Drives `Scope`'s hand-written `Hash` impl (which already skips the
/// `Allocator`/caches — see docs/ir-research.md §2 and the
/// `scope_hash_is_allocator_identity_independent` test below) with a custom
/// `Hasher` that forwards to SHA-256, after normalizing `source_loc`. Folds
/// in the serialized buffer/scalar `KernelArg` metadata (only `Serialize`,
/// not `Hash`) and `cube_dim`. Deliberately skips `options` — `kernel_name`
/// is a debug label, not semantic.
pub fn kernel_ir_hash(def: &KernelDefinition) -> String {
    use core::hash::{Hash, Hasher};

    let mut body = def.body.clone();
    normalize_source_locs(&mut body);

    let mut h = ShaHasher(Sha256::new());
    body.hash(&mut h);
    h.write(&serde_json::to_vec(&def.buffers).expect("KernelArg is Serialize"));
    h.write(&serde_json::to_vec(&def.scalars).expect("ScalarKernelArg is Serialize"));
    def.cube_dim.hash(&mut h);
    format!("sha256:{:x}", h.0.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::prelude::*;

    #[cube(launch)]
    fn ir_hash_test_axpy(alpha: f32, x: &Array<f32>, y: &mut Array<f32>) {
        if ABSOLUTE_POS < y.len() {
            y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
        }
    }

    #[cube(launch)]
    fn ir_hash_test_axpy_off_by_one(alpha: f32, x: &Array<f32>, y: &mut Array<f32>) {
        if ABSOLUTE_POS <= y.len() {
            y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
        }
    }

    #[cube(launch)]
    fn ir_hash_test_xorshift(x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS < y.len() {
            let mut s = x[ABSOLUTE_POS];
            s ^= s << 13u32;
            y[ABSOLUTE_POS] = s;
        }
    }

    fn build_axpy() -> KernelDefinition {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        AddressType::U32.register(&mut builder.scope);
        let alpha = <f32 as LaunchArg>::expand(&Default::default(), &mut builder);
        let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        ir_hash_test_axpy::expand(&mut builder.scope, alpha, x, y);
        builder.build(KernelSettings::default())
    }

    fn build_axpy_off_by_one() -> KernelDefinition {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        AddressType::U32.register(&mut builder.scope);
        let alpha = <f32 as LaunchArg>::expand(&Default::default(), &mut builder);
        let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        ir_hash_test_axpy_off_by_one::expand(&mut builder.scope, alpha, x, y);
        builder.build(KernelSettings::default())
    }

    fn build_xorshift() -> KernelDefinition {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        AddressType::U32.register(&mut builder.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<u32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        ir_hash_test_xorshift::expand(&mut builder.scope, x, y);
        builder.build(KernelSettings::default())
    }

    /// Hash determinism: two independent extractions of the same kernel
    /// definition, in the same process, produce the same hash.
    #[test]
    fn hash_is_deterministic_across_extractions() {
        let h1 = kernel_ir_hash(&build_axpy());
        let h2 = kernel_ir_hash(&build_axpy());
        assert_eq!(h1, h2);
        assert!(h1.starts_with("sha256:"));
    }

    /// Different kernels hash differently — including a one-token change
    /// (`<` vs `<=`) that source-level identity also catches, proving the
    /// IR hash is at least as sensitive as source identity for this case.
    #[test]
    fn hash_differs_between_kernels() {
        let axpy = kernel_ir_hash(&build_axpy());
        let off_by_one = kernel_ir_hash(&build_axpy_off_by_one());
        let xorshift = kernel_ir_hash(&build_xorshift());
        assert_ne!(axpy, off_by_one);
        assert_ne!(axpy, xorshift);
        assert_ne!(off_by_one, xorshift);
    }

    /// TRAP (docs/ir-research.md §2, §5.1): `Allocator::PartialEq` is
    /// `Rc::ptr_eq` on internal counters/pools, i.e. reference identity, not
    /// value identity — so two structurally-identical `Scope`s built from
    /// separate `KernelBuilder`s always compare UNEQUAL via derived
    /// `PartialEq`, even though their content hash (and printed IR) match.
    /// Pinned here exactly as the research doc recommends: if a future
    /// CubeCL upgrade changes `Allocator::eq` to value equality, this
    /// assertion flips and must be noticed (and this comment revisited) —
    /// do NOT use `==`/`assert_eq!` on `Scope` for identity purposes; the
    /// hand-written `Hash` impl `kernel_ir_hash` drives is the sound
    /// mechanism.
    #[test]
    fn scope_partial_eq_is_allocator_identity_not_value_equality() {
        let def1 = build_axpy();
        let def2 = build_axpy();
        assert_ne!(
            def1.body, def2.body,
            "expected Scope::eq to be Allocator::ptr_eq (false across separate builds) — \
             if this now passes, CubeCL's Allocator::PartialEq changed to value equality; \
             update this pin and reconsider whether `==` on Scope is now safe"
        );
        // The content hash is unaffected by the trap: it agrees across the
        // same two (PartialEq-unequal) builds.
        assert_eq!(kernel_ir_hash(&def1), kernel_ir_hash(&def2));
    }
}
