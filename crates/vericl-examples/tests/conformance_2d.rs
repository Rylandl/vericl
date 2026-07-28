//! 2-D dispatch conformance suite — the differential + bounds-proof evidence
//! for the multi-axis example kernels (docs/design-2d-dispatch.md).
//!
//! Why a separate suite (not more kernels in `conformance.rs`): a
//! `dispatch(...)` kernel's cases are **per-axis extents**, `(w, h)` tuples,
//! while an ordinary kernel's are scalar thread counts. Those are different
//! units, and one `sizes:` field cannot carry both — round 8's units discipline
//! says decide it rather than paper over it, so the two shapes get two suites
//! and two evidence files. A second `suite!` invocation with its own evidence
//! file is the established precedent (`conformance_f64.rs`,
//! `cooperative_fallback.rs`) and satisfies the constraint that two `#[test]`s
//! must not share one manifest.
//!
//! The six shapes are the design's own ground-truth set, chosen so `w != h`,
//! neither is a multiple of the cube dim, and the degenerate `1x1` and the thin
//! `3x129` / `129x3` cases are covered.
//!
//! Usage:
//!   cargo test                     verify evidence/vericl_2d.json
//!   VERICL_UPDATE=1 cargo test     regenerate it

use vericl_examples::*;

vericl::suite! {
    runtime: cubecl::wgpu::WgpuRuntime,
    kernels: [
        // 2-D elementwise — the coverage floor: per-axis guard, row-major
        // `y*w + x`, and the `out.len() == (w as usize) * (h as usize)` product
        // assume that makes the row stride provable at all.
        elementwise2d_scale,
        // Transpose — decode AND encode, two independent `checked_mul`
        // side-obligations against two different extents.
        transpose2d,
        // The clamped 3x3 box blur — the whole stencil class, reachable only
        // because `Arithmetic::Min`/`Max` are modeled as exact `ite` terms (the
        // branch-free clamp is the only spelling round-2 branch write taint
        // leaves standing).
        box_blur3x3,
        // Every per-axis builtin the clause admits, plus the two flat ones it
        // keeps (`CUBE_DIM`, `UNIT_POS`) — the differential is what checks the
        // twin's `abs_a % Wa` / `abs_a / Wa` bindings against the device's own
        // values.
        topology_report2d,
    ],
    evidence: "evidence/vericl_2d.json",
    // Per-axis EXTENTS, not thread counts. No `cube_dim:` field: each kernel's
    // `dispatch(cube_dim = (16, 16))` clause is the single source of truth for
    // the block size, and declaring it twice is rejected (R7).
    sizes: [(37, 19), (64, 64), (1, 1), (3, 129), (129, 3), (255, 257)],
}
