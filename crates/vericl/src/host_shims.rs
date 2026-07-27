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
//!   Rust trait dispatch via [`CastToF32`] — u32, i32, usize, bool, and
//!   f32-identity are the verified source types; any other source is a
//!   `CastToF32: not satisfied` compile error in the twin, loud, never a silent
//!   wrong value);
//! - `T::mul_hi(a, b)` / `a.mul_hi(b)`  →  [`mul_hi`]`(a, b)` (via [`MulHi`];
//!   u32 is the verified type, other types are a `MulHi: not satisfied` error);
//! - `fma(a, b, c)` (the free function `cubecl::prelude::fma`, the only form
//!   cubecl 0.10 defines)  →  [`fma`]`(a, b, c)` (via [`Fma`]; f32 is the
//!   verified type). **`a * b + c` is not a substitute** — see [`fma_f32`].
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
//! - **usize→f32** matches Rust `x as f32` bit-for-bit over the whole u32
//!   addressing domain, on wgpu and cubecl-cpu, including the real idiom
//!   `f32::cast_from(ABSOLUTE_POS)`.
//! - **bool→f32** is exactly `true → 1.0`, `false → +0.0` on both lanes.
//! - **fma f32** is bit-exact on cubecl-cpu everywhere, and bit-exact on
//!   wgpu/Metal on every triple with no subnormal operand or result. **One
//!   measured divergence class: Metal flushes subnormals to zero** (78 of 7972
//!   probe triples, all exactly `ftz(fma(ftz a, ftz b, ftz c))`), so the shim is
//!   bit-exact *outside* the subnormal domain and flush-to-zero-divergent
//!   inside it. See [`fma_f32`] and the ground-truth test's header.
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

/// `usize → f32`, verified equal to the GPU `f32::cast_from(x: usize)`.
///
/// **Why `as f32` is right on both sides of the width question.** In a `#[cube]`
/// kernel `usize` is not the host's 64-bit `usize`: it is cubecl's
/// `AddressType`, which is `U32` for every buffer of at most `u32::MAX`
/// elements and `U64` above that (`cubecl_ir::AddressType::from_len`). So the
/// GPU-side conversion is `u32 → f32` in the addressing regime any real kernel
/// runs in, and `u64 → f32` in the (unreachable-in-practice) wide regime. Rust's
/// `usize as f32` agrees with **both**: it is round-to-nearest-even from the
/// integer value, and for values `< 2^32` that is bit-identical to `u32 as f32`
/// (verified against the real intrinsic, below). This is the shape behind
/// `f32::cast_from(ABSOLUTE_POS)` and `f32::cast_from(x.len())` — both
/// `usize`-typed in cubecl 0.10.
#[inline]
pub fn cast_from_usize_f32(x: usize) -> f32 {
    x as f32
}

/// `bool → f32`, verified equal to the GPU `f32::cast_from(b: bool)`:
/// `true → 1.0`, `false → 0.0` (measured on wgpu and cubecl-cpu, exactly, both
/// with a `+0.0` sign bit). The shape behind a predicate-weighted accumulate
/// (`acc += f32::cast_from(x[i] < t)`) and behind a Bernoulli draw's
/// `f32::cast_from(u < p)`.
#[inline]
pub fn cast_from_bool_f32(b: bool) -> f32 {
    if b {
        1.0
    } else {
        0.0
    }
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

impl CastToF32 for usize {
    #[inline]
    fn vericl_cast_to_f32(self) -> f32 {
        cast_from_usize_f32(self)
    }
}

impl CastToF32 for bool {
    #[inline]
    fn vericl_cast_to_f32(self) -> f32 {
        cast_from_bool_f32(self)
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

// ---------------------------------------------------------------------------
// fma
// ---------------------------------------------------------------------------

/// Fused multiply-add `a*b + c` with a **single** rounding — verified equal to
/// the GPU `cubecl::prelude::fma(a, b, c)`.
///
/// **`a * b + c` is not a substitute, and that is the whole point of this
/// shim.** The unfused expression rounds twice; the fused one rounds once, and
/// the difference is not a rounding-noise detail but the entire signal for the
/// two-product idiom `fma(hi, x, -(hi*x))`, which extracts the *exact* residual
/// of a rounded product. Measured on a real double-single phase step
/// (`hi = 0.013371337`, `x = 4097.0`): the fused residual is `8.23e-7`, the
/// unfused rewrite is **exactly `0.0`** — a 100% relative error, silently, for
/// every input. A twin that rewrote `fma` to `a*b + c` would not be an
/// approximation of the kernel, it would be a different algorithm.
///
/// Rust's `f32::mul_add` is IEEE-754 `fusedMultiplyAdd`: one rounding,
/// guaranteed by the language (a real FMA instruction on aarch64 and on x86-64
/// with the `fma` target feature; a correctly-rounded software fallback
/// otherwise). That is the same operation WGSL `fma()` and LLVM's `llvm.fma`
/// name — the two lowerings cubecl 0.10 emits for `Arithmetic::Fma`
/// (`cubecl-wgpu` `wgsl::Instruction::Fma` → `fma(a, b, c)`; `cubecl-cpu`
/// → `llvm_ods::intr_fma`).
///
/// **Measured tier (the honest one, not the hoped-for one).** Against the real
/// intrinsic over 7972 triples — boundary, ±0, ±inf, NaN, subnormal,
/// cancellation-heavy two-product residuals, random:
///
/// - cubecl-cpu: **bit-exact everywhere**, subnormals included.
/// - wgpu/Metal: **bit-exact on every triple with no subnormal operand or
///   result**; on the 78 triples that touch the subnormal range the backend
///   **flushes denormals to zero** and the host does not. The divergence is not
///   approximate — it is exactly `gpu == ftz(fma_f32(ftz a, ftz b, ftz c))`,
///   asserted as a model in the ground-truth test rather than tolerated as
///   noise. A twin whose kernel genuinely computes in the subnormal range must
///   therefore compare with a tolerance, not `max_ulp = 0`; every other twin is
///   bit-exact. This is the same shape of finding as the `to_unit_interval`
///   Metal reciprocal-multiply (1 ULP): a real backend property, recorded.
///
/// **Discrimination.** If this shim were the naive `a*b + c`, the ground-truth
/// test would fail on 3654/7972 triples on wgpu and 3576/7972 on cubecl-cpu.
/// The verdict above is not vacuous.
#[inline]
pub fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// The set of types for which `fma(a, b, c)` has a GPU-verified host shim
/// (v1: `f32` only). An unsupported type is a `Fma: not satisfied` compile
/// error in the twin — the same closed-set, loud-over-silent discipline as
/// [`CastToF32`] and [`MulHi`]. (cubecl's `fma` is generic over every
/// `CubePrimitive`; only `f32` is in the surveyed demand and ground-truthed
/// here. `f64` is a one-impl extension once measured on the cubecl-cpu lane —
/// the only honest f64 backend, see the README "f64 support" section.)
pub trait Fma {
    /// `self * b + c` with a single rounding, GPU-matching.
    fn vericl_fma(self, b: Self, c: Self) -> Self;
}

impl Fma for f32 {
    #[inline]
    fn vericl_fma(self, b: Self, c: Self) -> Self {
        fma_f32(self, b, c)
    }
}

/// The twin's target for a rewritten `fma(a, b, c)` — dispatches via [`Fma`].
/// The macro cannot know the operand type at expansion (the intrinsic is a free
/// function with no type qualifier at all), so the trait resolves it.
#[inline]
pub fn fma<T: Fma>(a: T, b: T, c: T) -> T {
    a.vericl_fma(b, c)
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

    /// `usize` sources: identical to the `u32` shim across the whole u32 range
    /// (the `AddressType::U32` regime every real kernel runs in), including
    /// above 2^24 where the rounding is observable.
    #[test]
    fn cast_from_usize_matches_u32_over_the_address_range() {
        for x in [0usize, 1, (1 << 24) + 1, (1 << 24) + 3, u32::MAX as usize] {
            assert_eq!(
                cast_from_usize_f32(x).to_bits(),
                cast_from_u32_f32(x as u32).to_bits(),
                "usize and u32 shims must agree at {x}"
            );
        }
        // Above the u32 addressing regime the value is a genuine 64-bit one
        // (AddressType::U64); `as f32` is still round-to-nearest-even.
        assert_eq!(cast_from_usize_f32(1usize << 40), 1_099_511_627_776.0);
    }

    #[test]
    fn cast_from_bool_is_one_or_zero() {
        assert_eq!(cast_from_bool_f32(true), 1.0);
        assert_eq!(cast_from_bool_f32(false), 0.0);
        // Positive zero, not negative zero — the GPU intrinsic's bit pattern.
        assert_eq!(cast_from_bool_f32(false).to_bits(), 0u32);
    }

    /// The load-bearing property: one rounding, not two. `fma(hi, x, -(hi*x))`
    /// is the exact residual of the rounded product; the unfused rewrite
    /// collapses to exactly zero. (The same numbers recorded in `fma_f32`'s
    /// doc, measured on a real double-single phase-step shape.)
    #[test]
    fn fma_is_fused_not_mul_then_add() {
        let hi = 0.013_371_337f32;
        let x = 4097.0f32;
        let p_hi = hi * x;
        let fused = fma_f32(hi, x, -p_hi);
        let unfused = hi * x - p_hi;
        assert_ne!(fused, 0.0, "the fused residual is genuinely non-zero");
        assert_eq!(unfused, 0.0, "the unfused rewrite collapses to exactly zero");
        // Pin the magnitude so a regression that silently unfuses is loud.
        assert!((fused.abs() - 8.23e-7).abs() < 1e-9, "residual was {fused:e}");
    }

    #[test]
    fn fma_boundary_values() {
        assert_eq!(fma_f32(2.0, 3.0, 4.0), 10.0);
        assert_eq!(fma_f32(0.0, 0.0, 0.0), 0.0);
        assert_eq!(fma_f32(1.0, -1.0, 1.0), 0.0);
        assert!(fma_f32(f32::INFINITY, 1.0, 0.0).is_infinite());
        assert!(fma_f32(f32::NAN, 1.0, 0.0).is_nan());
    }

    #[test]
    fn dispatch_matches_named() {
        assert_eq!(cast_to_f32(7u32), cast_from_u32_f32(7));
        assert_eq!(cast_to_f32(-7i32), cast_from_i32_f32(-7));
        assert_eq!(cast_to_f32(2.5f32), 2.5);
        assert_eq!(cast_to_f32(9usize), cast_from_usize_f32(9));
        assert_eq!(cast_to_f32(true), 1.0);
        assert_eq!(mul_hi(0xDEAD_BEEFu32, 0xCAFEu32), mul_hi_u32(0xDEAD_BEEF, 0xCAFE));
        assert_eq!(fma(2.0f32, 3.0, 4.0), fma_f32(2.0, 3.0, 4.0));
    }
}
