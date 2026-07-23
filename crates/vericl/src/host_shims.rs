//! Host shims for GPU-defined numeric intrinsics whose reference twin cannot
//! call the real `cubecl::prelude::Cast`/`Numeric` methods on host.
//!
//! `Cast::cast_from` and `Numeric::mul_hi` are `unexpanded!()` on host (they
//! panic — see `FLOAT_METHOD_REJECT` in vericl-macros and the empirical proof
//! in `crates/vericl-examples/tests/float_method_whitelist.rs`), so a derived
//! twin cannot invoke them directly. Their semantics are *GPU-defined*
//! (particularly the u32→f32 rounding mode), so — unlike the
//! `FLOAT_METHOD_WHITELIST` methods, which are cross-checked against `std` — a
//! host shim for one of these must be validated against **GPU ground truth**,
//! not against `std`'s intuition of what the operation "should" do.
//!
//! `#[vericl::kernel]`/`#[vericl::helper]` rewrite a recognized intrinsic call
//! in the twin body to the matching shim here:
//!
//! - `f32::cast_from(x)`  →  [`cast_to_f32`]`(x)` (source type resolved by
//!   Rust trait dispatch via [`CastToF32`] — u32, i32, and f32-identity are the
//!   verified source types; any other source is a `CastToF32: not satisfied`
//!   compile error in the twin, loud, never a silent wrong value);
//! - `T::mul_hi(a, b)` / `a.mul_hi(b)`  →  [`mul_hi`]`(a, b)` (via [`MulHi`];
//!   u32 is the verified type, other types are a `MulHi: not satisfied` error).
//!
//! **GPU ground truth (the load-bearing verification).** Every shim below is
//! pinned bit-exactly against the real intrinsic run in a real `#[cube]` kernel
//! on wgpu (and, where the backend supports it, cubecl-cpu) across boundary +
//! random inputs — see `crates/vericl-examples/tests/host_shim_gpu_ground_truth.rs`.
//! The empirical result recorded there:
//!
//! - **u32→f32** on wgpu/Metal matches Rust `x as f32` bit-for-bit across the
//!   full range, *including* values above 2^24 where rounding is observable
//!   (both round to nearest, ties to even). No divergence from `as f32`.
//! - **i32→f32** on wgpu/Metal matches Rust `x as f32` bit-for-bit (same
//!   round-to-nearest-even, including the negative >2^24 magnitude range).
//! - **mul_hi u32** on wgpu/Metal matches `((a as u64) * (b as u64)) >> 32`
//!   bit-for-bit (the high word of the 64-bit unsigned product).
//!
//! If a future backend's rounding or high-word semantics were to diverge from
//! these, the ground-truth test fails loudly and the shim — not the test —
//! must be changed to match the GPU (the intrinsic's semantics are whatever the
//! hardware does, and the twin must reproduce that, not `std`'s convention).

// ---------------------------------------------------------------------------
// cast_from → f32
// ---------------------------------------------------------------------------

/// `u32 → f32`, verified equal to the GPU `f32::cast_from(x: u32)`
/// (round-to-nearest-even; matches Rust `as f32`).
#[inline]
pub fn cast_from_u32_f32(x: u32) -> f32 {
    x as f32
}

/// `i32 → f32`, verified equal to the GPU `f32::cast_from(x: i32)`
/// (round-to-nearest-even; matches Rust `as f32`).
#[inline]
pub fn cast_from_i32_f32(x: i32) -> f32 {
    x as f32
}

/// `f32 → f32` identity — `f32::cast_from(x: f32)` is the no-op same-type cast
/// (`instantiate(...)` can pin a generic `F::cast_from` where the source is
/// already `f32`).
#[inline]
pub fn cast_from_f32_f32(x: f32) -> f32 {
    x
}

/// The set of source types for which `f32::cast_from(source)` has a
/// GPU-verified host shim. Deliberately closed: an unsupported source type
/// produces a `the trait bound \`_: CastToF32\` is not satisfied` error in the
/// generated twin (loud, at the twin's own call-site span), never a silently
/// approximated value. Grow it only by adding a GPU-verified shim + impl.
pub trait CastToF32 {
    /// The value cast to `f32` with GPU-matching semantics.
    fn vericl_cast_to_f32(self) -> f32;
}

impl CastToF32 for u32 {
    #[inline]
    fn vericl_cast_to_f32(self) -> f32 {
        cast_from_u32_f32(self)
    }
}

impl CastToF32 for i32 {
    #[inline]
    fn vericl_cast_to_f32(self) -> f32 {
        cast_from_i32_f32(self)
    }
}

impl CastToF32 for f32 {
    #[inline]
    fn vericl_cast_to_f32(self) -> f32 {
        cast_from_f32_f32(self)
    }
}

/// The twin's target for a rewritten `f32::cast_from(x)` — dispatches to the
/// GPU-verified per-source shim via [`CastToF32`]. The macro emits this rather
/// than a source-specific function name because it cannot know the argument's
/// concrete type at expansion; Rust's trait resolution supplies it, and an
/// unsupported source is a clean compile error (see [`CastToF32`]).
#[inline]
pub fn cast_to_f32<S: CastToF32>(x: S) -> f32 {
    x.vericl_cast_to_f32()
}

// ---------------------------------------------------------------------------
// mul_hi
// ---------------------------------------------------------------------------

/// High 32 bits of the 64-bit unsigned product `a * b` — verified equal to the
/// GPU `u32::mul_hi(a, b)`.
#[inline]
pub fn mul_hi_u32(a: u32, b: u32) -> u32 {
    (((a as u64) * (b as u64)) >> 32) as u32
}

/// The set of types for which `mul_hi(a, b)` has a GPU-verified host shim
/// (v1: `u32` only). An unsupported type is a `MulHi: not satisfied` compile
/// error in the twin — the same closed-set, loud-over-silent discipline as
/// [`CastToF32`]. (cubecl also defines `mul_hi` for `i32`/`usize`/`isize`; only
/// `u32` is in the surveyed demand and is verified here — extend by adding a
/// GPU-verified shim + impl.)
pub trait MulHi {
    /// The high word of the full-width product `self * other`, GPU-matching.
    fn vericl_mul_hi(self, other: Self) -> Self;
}

impl MulHi for u32 {
    #[inline]
    fn vericl_mul_hi(self, other: Self) -> Self {
        mul_hi_u32(self, other)
    }
}

/// The twin's target for a rewritten `T::mul_hi(a, b)` / `a.mul_hi(b)` —
/// dispatches via [`MulHi`]. Emitted uniformly for both the path and method
/// call forms (the macro cannot always know the operand type at expansion; the
/// trait resolves it).
#[inline]
pub fn mul_hi<T: MulHi>(a: T, b: T) -> T {
    a.vericl_mul_hi(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Boundary values where u32→f32 rounding is observable (> 2^24). Pins the
    /// shim's own arithmetic; the GPU-equality half lives in
    /// `vericl-examples/tests/host_shim_gpu_ground_truth.rs`.
    #[test]
    fn cast_from_u32_rounds_to_nearest_even() {
        assert_eq!(cast_from_u32_f32(0), 0.0);
        assert_eq!(cast_from_u32_f32(1 << 24), 16_777_216.0);
        // 2^24 + 1 is not representable; ties-to-even rounds to 2^24.
        assert_eq!(cast_from_u32_f32((1 << 24) + 1), 16_777_216.0);
        // 2^24 + 3 rounds up to 2^24 + 4.
        assert_eq!(cast_from_u32_f32((1 << 24) + 3), 16_777_220.0);
        assert_eq!(cast_from_u32_f32(u32::MAX), 4_294_967_296.0);
    }

    #[test]
    fn cast_from_i32_signed() {
        assert_eq!(cast_from_i32_f32(-1), -1.0);
        assert_eq!(cast_from_i32_f32(i32::MIN), -2_147_483_648.0);
        assert_eq!(cast_from_i32_f32(-((1 << 24) + 1)), -16_777_216.0);
    }

    #[test]
    fn mul_hi_u32_high_word() {
        assert_eq!(mul_hi_u32(0, u32::MAX), 0);
        assert_eq!(mul_hi_u32(u32::MAX, u32::MAX), u32::MAX - 1);
        assert_eq!(mul_hi_u32(1 << 16, 1 << 16), 1); // 2^32 >> 32
        assert_eq!(mul_hi_u32(0x8000_0000, 2), 1);
    }

    #[test]
    fn dispatch_matches_named() {
        assert_eq!(cast_to_f32(7u32), cast_from_u32_f32(7));
        assert_eq!(cast_to_f32(-7i32), cast_from_i32_f32(-7));
        assert_eq!(cast_to_f32(2.5f32), 2.5);
        assert_eq!(mul_hi(0xDEAD_BEEFu32, 0xCAFEu32), mul_hi_u32(0xDEAD_BEEF, 0xCAFE));
    }
}
