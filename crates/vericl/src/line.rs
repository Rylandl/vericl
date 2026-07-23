//! `Line<T, W>` — the pinned lane-array host shim for CubeCL's `Vector<P, N>`
//! element type (design-line-vector.md §4.1, §7).
//!
//! The real `cubecl::prelude::Vector<P, N>` host value stores **only one
//! element** (`base.rs:12` "Comptime vectors only support 1 element"), so it is
//! useless as a reference twin. VeriCL therefore supplies its own host type: a
//! `[T; W]` lane array whose every operation is a straight **per-lane** map. A
//! whole-vector op on `Vector<f32, N>` is `N` independent scalar ops with no
//! cross-lane coupling and no reordering, so at equal precision the lane-array
//! twin reproduces the GPU value **bit-for-bit** for the correctly-rounded
//! elementwise ops (`+ - *`, neg, bitwise, compare, exact float ops; design
//! §4.5, §6). The only divergences are the same ones every scalar float twin has
//! and that `compare(abs=…)` already covers: an FMA the backend contracts, and —
//! a ground-truth FINDING recorded in the GT test — `f32 /` division, which
//! Metal does **not** correctly-round (≤1 ULP vs the host's `/`). These are
//! per-lane float facts, not vector-model errors: the GT test confirms the GPU
//! vec op equals the GPU per-lane scalar op bit-for-bit.
//!
//! This is a *shim* in the exact sense of the `cast_from`/`mul_hi` shims in
//! [`crate::host_shims`]: its per-lane semantics are not assumed, they are
//! **GPU-ground-truth-verified**, bit-exact, against real `Vector<_, N>` kernels
//! on wgpu (and, behind `--features cpu`, cubecl-cpu). Every op group below has a
//! row in `crates/vericl-examples/tests/line_shim_gpu_ground_truth.rs`; no op
//! reaches the reference-twin surface without one (round-8 risk 2).
//!
//! The op surface mirrors the v1 subset of `cubecl-core`'s
//! `frontend/container/vector/{base,ops}.rs` (design §7): elementwise
//! arithmetic/bitwise, splat/constructors, per-lane comparison → `Line<bool, W>`,
//! `count_ones`, width query, lane index, and the per-lane float math whitelist.
//! Cross-lane reductions (`VectorSum`/`dot`/`magnitude`/`normalize`) and vector
//! `cast_from`/`reinterpret`/`wrapping` are **deliberately absent** — they are
//! deferred (design §7, §8.4) and the macro rejects them by name (design §8.3).
//!
//! The `vericl` crate does not depend on CubeCL (see the crate docs), so this
//! type is pure host Rust; the generated twin — which lives in the downstream
//! kernel crate where `cubecl::prelude` is in scope — calls `Line`'s ops after
//! the macro rewrites the `Vector` head to `Line`.

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Index, IndexMut, Mul, MulAssign, Neg, Not, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

/// A width-`W` lane array. `Line<T, W>` is to VeriCL's twin what
/// `cubecl::prelude::Vector<T, N>` is to a kernel: a SIMD element whose ops are
/// per-lane. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Line<T, const W: usize>(pub [T; W]);

// ---------------------------------------------------------------------------
// Constructors (splat / fill / empty / zeroed) — generic over the lane type.
// ---------------------------------------------------------------------------

impl<T: Copy, const W: usize> Line<T, W> {
    /// Splat: every lane set to `val`. Mirrors `Vector::new(val)` (which lowers
    /// to a `Cast` broadcast on the GPU, design §2.5).
    #[inline]
    pub fn new(val: T) -> Self {
        Line([val; W])
    }

    /// `Vector::fill(self, value)` — replace every lane with `value`. The GPU
    /// form takes `self` and returns a fully-filled vector (the prior contents
    /// are irrelevant), so this ignores `self`.
    #[inline]
    #[must_use]
    pub fn fill(self, value: T) -> Self {
        Line([value; W])
    }

    /// The raw lanes.
    #[inline]
    pub fn into_array(self) -> [T; W] {
        self.0
    }
}

impl<T: Copy + Default, const W: usize> Line<T, W> {
    /// `Vector::zeroed()` — every lane is the type's zero. GPU `zeroed` is
    /// `bytemuck::Zeroable`, i.e. all-bits-zero, which for the numeric lane
    /// types this shim serves equals `Default::default()`.
    #[inline]
    pub fn zeroed() -> Self {
        Line([T::default(); W])
    }

    /// `Vector::empty()` — an uninitialized vector on the GPU. The twin has no
    /// notion of uninitialized memory and models it as zero (design §7:
    /// "empty/zeroed = zero-init"). A kernel that *reads* an `empty()` lane
    /// before writing it is relying on unspecified data; the differential
    /// compare catches any resulting divergence rather than this masking it.
    #[inline]
    pub fn empty() -> Self {
        Line([T::default(); W])
    }
}

// ---------------------------------------------------------------------------
// Lane index — `v[j]` read / `v[j] = x` write (design §7 "Lane index").
// ---------------------------------------------------------------------------

impl<T, const W: usize> Index<usize> for Line<T, W> {
    type Output = T;
    #[inline]
    fn index(&self, lane: usize) -> &T {
        &self.0[lane]
    }
}

impl<T, const W: usize> IndexMut<usize> for Line<T, W> {
    #[inline]
    fn index_mut(&mut self, lane: usize) -> &mut T {
        &mut self.0[lane]
    }
}

impl<T: Copy, const W: usize> Line<T, W> {
    /// `RuntimeCell::store_at(j, x)` — write one lane (design §2.4). Provided as
    /// a method alongside `IndexMut` because the GPU per-lane write form is a
    /// `store_at` call, not always an index-assign.
    #[inline]
    pub fn store_at(&mut self, lane: usize, value: T) {
        self.0[lane] = value;
    }

    /// The width `W`, mirroring `Vector::size()` / `Vector::vector_size()`
    /// (design §7 "Width query"). Comptime on the GPU; a plain `usize` here.
    #[inline]
    pub fn size(&self) -> usize {
        W
    }

    /// Alias of [`size`](Self::size) — cubecl exposes both names.
    #[inline]
    pub fn vector_size(&self) -> usize {
        W
    }
}

// ---------------------------------------------------------------------------
// Elementwise arithmetic (Add/Sub/Mul/Div/Neg + assign forms). Per-lane, in
// the exact order the GPU performs it — bit-exact at equal precision (§4.5).
// ---------------------------------------------------------------------------

macro_rules! impl_line_binop {
    ($trait:ident, $method:ident, $bound:ident, $op:tt) => {
        impl<T: $bound<Output = T> + Copy, const W: usize> $trait for Line<T, W> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self {
                Line(core::array::from_fn(|i| self.0[i] $op rhs.0[i]))
            }
        }
    };
}

impl_line_binop!(Add, add, Add, +);
impl_line_binop!(Sub, sub, Sub, -);
impl_line_binop!(Mul, mul, Mul, *);
impl_line_binop!(Div, div, Div, /);
impl_line_binop!(BitAnd, bitand, BitAnd, &);
impl_line_binop!(BitOr, bitor, BitOr, |);
impl_line_binop!(BitXor, bitxor, BitXor, ^);
impl_line_binop!(Shl, shl, Shl, <<);
impl_line_binop!(Shr, shr, Shr, >>);

macro_rules! impl_line_assign {
    ($trait:ident, $method:ident, $bound:ident, $op:tt) => {
        impl<T: $bound + Copy, const W: usize> $trait for Line<T, W> {
            #[inline]
            fn $method(&mut self, rhs: Self) {
                for i in 0..W {
                    self.0[i] $op rhs.0[i];
                }
            }
        }
    };
}

impl_line_assign!(AddAssign, add_assign, AddAssign, +=);
impl_line_assign!(SubAssign, sub_assign, SubAssign, -=);
impl_line_assign!(MulAssign, mul_assign, MulAssign, *=);
impl_line_assign!(DivAssign, div_assign, DivAssign, /=);
impl_line_assign!(BitAndAssign, bitand_assign, BitAndAssign, &=);
impl_line_assign!(BitOrAssign, bitor_assign, BitOrAssign, |=);
impl_line_assign!(BitXorAssign, bitxor_assign, BitXorAssign, ^=);
impl_line_assign!(ShlAssign, shl_assign, ShlAssign, <<=);
impl_line_assign!(ShrAssign, shr_assign, ShrAssign, >>=);

impl<T: Neg<Output = T> + Copy, const W: usize> Neg for Line<T, W> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Line(core::array::from_fn(|i| -self.0[i]))
    }
}

impl<T: Not<Output = T> + Copy, const W: usize> Not for Line<T, W> {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Line(core::array::from_fn(|i| !self.0[i]))
    }
}

// ---------------------------------------------------------------------------
// Per-lane comparison → Line<bool, W>  (design §7 "Comparison").
// ---------------------------------------------------------------------------

impl<T: PartialEq + Copy, const W: usize> Line<T, W> {
    /// Per-lane `==`, mirroring `Vector::equal`.
    #[inline]
    #[must_use]
    pub fn equal(self, other: Self) -> Line<bool, W> {
        Line(core::array::from_fn(|i| self.0[i] == other.0[i]))
    }

    /// Per-lane `!=`, mirroring `Vector::not_equal`.
    #[inline]
    #[must_use]
    pub fn not_equal(self, other: Self) -> Line<bool, W> {
        Line(core::array::from_fn(|i| self.0[i] != other.0[i]))
    }
}

impl<T: PartialOrd + Copy, const W: usize> Line<T, W> {
    /// Per-lane `<`, mirroring `Vector::less_than`.
    #[inline]
    #[must_use]
    pub fn less_than(self, other: Self) -> Line<bool, W> {
        Line(core::array::from_fn(|i| self.0[i] < other.0[i]))
    }

    /// Per-lane `>`, mirroring `Vector::greater_than`.
    #[inline]
    #[must_use]
    pub fn greater_than(self, other: Self) -> Line<bool, W> {
        Line(core::array::from_fn(|i| self.0[i] > other.0[i]))
    }

    /// Per-lane `<=`, mirroring `Vector::less_equal`.
    #[inline]
    #[must_use]
    pub fn less_equal(self, other: Self) -> Line<bool, W> {
        Line(core::array::from_fn(|i| self.0[i] <= other.0[i]))
    }

    /// Per-lane `>=`, mirroring `Vector::greater_equal`.
    #[inline]
    #[must_use]
    pub fn greater_equal(self, other: Self) -> Line<bool, W> {
        Line(core::array::from_fn(|i| self.0[i] >= other.0[i]))
    }
}

impl<const W: usize> Line<bool, W> {
    /// Per-lane logical AND of two bool vectors, mirroring `Vector::<bool>::and`.
    #[inline]
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        Line(core::array::from_fn(|i| self.0[i] && other.0[i]))
    }

    /// Per-lane logical OR of two bool vectors, mirroring `Vector::<bool>::or`.
    #[inline]
    #[must_use]
    pub fn or(self, other: Self) -> Self {
        Line(core::array::from_fn(|i| self.0[i] || other.0[i]))
    }
}

// ---------------------------------------------------------------------------
// Integer-lane methods: count_ones, from_int, min_value/max_value. Per type,
// closed set (each has a GT row for its GPU intrinsic where one exists).
// ---------------------------------------------------------------------------

macro_rules! impl_line_int {
    ($($t:ty),+ $(,)?) => {$(
        impl<const W: usize> Line<$t, W> {
            /// Per-lane population count → a `u32` line, mirroring
            /// `Vector::count_ones` (which returns `Vector<u32, N>`).
            #[inline]
            #[must_use]
            pub fn count_ones(self) -> Line<u32, W> {
                Line(core::array::from_fn(|i| self.0[i].count_ones()))
            }

            /// `Vector::from_int(val)` — splat a constant across every lane.
            #[inline]
            pub fn from_int(val: i64) -> Self {
                Line([val as $t; W])
            }

            /// `Vector::min_value()` — the type minimum, splatted.
            #[inline]
            pub fn min_value() -> Self {
                Line([<$t>::MIN; W])
            }

            /// `Vector::max_value()` — the type maximum, splatted.
            #[inline]
            pub fn max_value() -> Self {
                Line([<$t>::MAX; W])
            }
        }
    )+};
}

impl_line_int!(u32, i32, u64, i64);

// ---------------------------------------------------------------------------
// Per-lane float math — the FLOAT_METHOD_WHITELIST (vericl-macros) applied
// lane-wise. On the scalar twin `x.sqrt()` is std's inherent `f32::sqrt`, which
// `tests/float_method_whitelist.rs` already pins bit-exact to the GPU; a vec-W
// math op is W of those scalar ops, so per-lane reuse is faithful by
// construction (design §7 "Math (unary)"). Implemented for both float precisions
// the twin can pin via `instantiate(F = f32 | f64)`.
//
// `new` (splat) is the generic `Line::new` above; the per-type block below adds
// the `from_int`/`min_value`/`max_value` whitelist constructors and the math
// methods. Each unary/binary method forwards to the std inherent method of the
// same name, one lane at a time.
// ---------------------------------------------------------------------------

/// Emit a batch of per-lane unary `self -> Self` float methods that forward to
/// the std inherent method of the same name.
macro_rules! line_unary_math {
    ($($m:ident),+ $(,)?) => {$(
        #[inline]
        #[must_use]
        pub fn $m(self) -> Self {
            Line(core::array::from_fn(|i| self.0[i].$m()))
        }
    )+};
}

/// Emit a batch of per-lane binary `self, Self -> Self` float methods.
macro_rules! line_binary_math {
    ($($m:ident),+ $(,)?) => {$(
        #[inline]
        #[must_use]
        pub fn $m(self, other: Self) -> Self {
            Line(core::array::from_fn(|i| self.0[i].$m(other.0[i])))
        }
    )+};
}

macro_rules! impl_line_float {
    ($t:ty) => {
        impl<const W: usize> Line<$t, W> {
            /// `Vector::from_int(val)` — splat an integer-valued constant.
            #[inline]
            pub fn from_int(val: i64) -> Self {
                Line([val as $t; W])
            }

            /// `min_value` / `max_value` (whitelist ctors).
            #[inline]
            pub fn min_value() -> Self {
                Line([<$t>::MIN; W])
            }
            #[inline]
            pub fn max_value() -> Self {
                Line([<$t>::MAX; W])
            }

            line_unary_math!(abs, floor, ceil, round, trunc, sqrt, recip, sin, cos, tan,
                asin, acos, atan, sinh, cosh, tanh, exp, ln, to_degrees, to_radians);

            /// Per-lane `is_nan` → a bool line (whitelist `is_nan`).
            #[inline]
            #[must_use]
            pub fn is_nan(self) -> Line<bool, W> {
                Line(core::array::from_fn(|i| self.0[i].is_nan()))
            }

            /// Per-lane `powi` (integer exponent, shared across lanes).
            #[inline]
            #[must_use]
            pub fn powi(self, n: i32) -> Self {
                Line(core::array::from_fn(|i| self.0[i].powi(n)))
            }

            line_binary_math!(min, max, powf, atan2, hypot);

            /// Per-lane `clamp` with vector bounds (whitelist `clamp`).
            #[inline]
            #[must_use]
            pub fn clamp(self, min: Self, max: Self) -> Self {
                Line(core::array::from_fn(|i| self.0[i].clamp(min.0[i], max.0[i])))
            }
        }
    };
}

impl_line_float!(f32);
impl_line_float!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_is_per_lane() {
        let a = Line([1.0f32, 2.0, 3.0, 4.0]);
        let b = Line([10.0f32, 20.0, 30.0, 40.0]);
        assert_eq!((a + b).0, [11.0, 22.0, 33.0, 44.0]);
        assert_eq!((b - a).0, [9.0, 18.0, 27.0, 36.0]);
        assert_eq!((a * b).0, [10.0, 40.0, 90.0, 160.0]);
        assert_eq!((b / a).0, [10.0, 10.0, 10.0, 10.0]);
        assert_eq!((-a).0, [-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn assign_forms_are_per_lane() {
        let mut a = Line([1.0f32, 2.0, 3.0, 4.0]);
        a += Line([1.0f32; 4]);
        assert_eq!(a.0, [2.0, 3.0, 4.0, 5.0]);
        a *= Line([2.0f32; 4]);
        assert_eq!(a.0, [4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn splat_and_index() {
        let mut v = Line::<f32, 4>::new(7.0);
        assert_eq!(v.0, [7.0; 4]);
        assert_eq!(v[2], 7.0);
        v[1] = 9.0;
        assert_eq!(v.0, [7.0, 9.0, 7.0, 7.0]);
        v.store_at(3, 5.0);
        assert_eq!(v[3], 5.0);
        assert_eq!(v.size(), 4);
        assert_eq!(v.vector_size(), 4);
        assert_eq!(Line::<f32, 4>::zeroed().0, [0.0; 4]);
        assert_eq!(v.fill(2.0).0, [2.0; 4]);
    }

    #[test]
    fn comparisons_yield_bool_lines() {
        let a = Line([1.0f32, 5.0, 3.0, 4.0]);
        let b = Line([2.0f32, 2.0, 3.0, 8.0]);
        assert_eq!(a.less_than(b).0, [true, false, false, true]);
        assert_eq!(a.equal(b).0, [false, false, true, false]);
        assert_eq!(a.greater_equal(b).0, [false, true, true, false]);
        let c = a.less_than(b);
        let d = a.greater_than(b);
        assert_eq!(c.or(d).0, [true, true, false, true]);
        assert_eq!(c.and(d).0, [false, false, false, false]);
    }

    #[test]
    fn bitwise_and_count_ones() {
        let a = Line([0b1010u32, 0xFFu32, 0u32, 0xF0F0u32]);
        let b = Line([0b0110u32, 0x0Fu32, 0xFFu32, 0x0F0Fu32]);
        assert_eq!((a & b).0, [0b0010, 0x0F, 0, 0]);
        assert_eq!((a | b).0, [0b1110, 0xFF, 0xFF, 0xFFFF]);
        assert_eq!((a ^ b).0, [0b1100, 0xF0, 0xFF, 0xFFFF]);
        assert_eq!(Line([1u32, 2, 4, 7]).count_ones().0, [1, 1, 1, 3]);
        assert_eq!((Line([1u32; 4]) << Line([1u32, 2, 3, 4])).0, [2, 4, 8, 16]);
    }

    #[test]
    fn per_lane_math_matches_scalar_std() {
        let v = Line([1.0f32, 4.0, 9.0, 16.0]);
        assert_eq!(v.sqrt().0, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(Line([-1.0f32, 2.0, -3.0, 4.0]).abs().0, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(Line([1.5f32, 2.4, -0.5, 3.9]).floor().0, [1.0, 2.0, -1.0, 3.0]);
        // Lane-wise == W scalar std ops.
        for (i, &x) in v.0.iter().enumerate() {
            assert_eq!(v.exp().0[i], x.exp());
            assert_eq!(v.ln().0[i], x.ln());
        }
        assert_eq!(Line([1.0f32, 2.0, 3.0, 4.0]).min(Line([2.0f32, 1.0, 5.0, 0.0])).0, [1.0, 1.0, 3.0, 0.0]);
        assert_eq!(Line::<f32, 4>::from_int(3).0, [3.0; 4]);
    }

    #[test]
    fn f64_math_is_full_precision() {
        let a = 1.0f64 + 2f64.powi(-40);
        let v = Line::<f64, 2>::new(a);
        assert_eq!(v.0, [a; 2]);
        assert_ne!(a, (a as f32) as f64);
    }
}
