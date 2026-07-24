//! Example kernels for the vericl first release.
//!
//! Two honest kernels (one float with a declared tolerance, one integer with
//! exact comparison) and two deliberately defective twins whose defects the
//! differential check must catch (README outcome 4).

use cubecl::prelude::*;

/// Generic saxpy — the flagship `instantiate(...)` example: `F` is pinned to
/// `f32` below, monomorphizing the reference twin, `conformance_case`'s
/// launch, and `kernel_definition`'s IR extraction all at that one concrete
/// type (see the `instantiate(...)` contract clause in the README).
///
/// Tolerance rationale (a first real VeriCL finding): the wgpu/Metal backend
/// contracts `a*x + y` (fma / fast-math), so under cancellation no useful ULP
/// bound exists — divergence up to ~27k ULP was observed. The honest claim is
/// an absolute error bound derived from the declared input ranges: one
/// rounding of `alpha*x` with |alpha| <= 4 and |x| <= 100 is at most
/// ulp(400) ≈ 3.1e-5, so abs = 1e-4 covers contraction with margin.
#[vericl::kernel(
    assumes(
        x.len() == y.len(),
        alpha.abs() <= 4.0,
        x.iter().all(|v| v.abs() <= 100.0),
        y.iter().all(|v| v.abs() <= 100.0)
    ),
    compare(abs = 1e-4),
    gen(alpha in -4.0..=4.0, x in -100.0..=100.0, y in -100.0..=100.0),
    instantiate(F = f32)
)]
#[cube(launch)]
pub fn axpy<F: Float + CubeElement>(alpha: F, x: &Array<F>, y: &mut Array<F>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
    }
}

/// The f64 flagship: `axpy` with `instantiate(F = f64)` instead of `f32`.
/// Identical body, generic over `F` exactly like `axpy` — the only change is
/// the pinned concrete type — demonstrating that the `instantiate(...)`
/// machinery monomorphizes the twin, launch, and IR at f64 with no per-type
/// special-casing in the kernel author's code.
///
/// **Differential lane is cubecl-cpu, never wgpu.** WGSL has no f64: cubecl
/// 0.10 launches an f64 kernel on the wgpu/Metal backend with no compile
/// error and no panic, but the results are silently *wrong* (not even an f32
/// demotion — genuine garbage; verified empirically, see the README "f64
/// support" section and `tests/f64_wgpu_unsound.rs`). So this kernel's suite
/// lane is `cubecl::cpu::CpuRuntime` (`tests/conformance_f64.rs`), where f64
/// runs at full precision. Both cpu and wgpu share cubecl's front end, and
/// wgpu is unusable for f64 anyway, so there is currently **no**
/// front-end-independent execution lane for an f64 kernel on this platform —
/// the macro-derived sequential twin is the sole independent leg, which makes
/// its independence load-bearing.
///
/// Tolerance rationale: with `|alpha| <= 4` and `|x| <= 100`, `|alpha*x| <=
/// 400`, and one f64 rounding at that scale is at most `ulp(400) ≈ 5.7e-14`,
/// so `abs = 1e-12` covers a rounding (and any fma contraction the backend
/// might apply) with wide margin — the same claim shape as `axpy`'s, one
/// precision tier finer. In practice cubecl-cpu matches the strict-f64 twin
/// bit-for-bit here (no contraction observed), so the tolerance is never
/// approached; it is declared to stay honest about what is *guaranteed*, not
/// what is merely observed.
#[vericl::kernel(
    assumes(
        x.len() == y.len(),
        alpha.abs() <= 4.0,
        x.iter().all(|v| v.abs() <= 100.0),
        y.iter().all(|v| v.abs() <= 100.0)
    ),
    compare(abs = 1e-12),
    gen(alpha in -4.0..=4.0, x in -100.0..=100.0, y in -100.0..=100.0),
    instantiate(F = f64)
)]
#[cube(launch)]
pub fn axpy_f64<F: Float + CubeElement>(alpha: F, x: &Array<F>, y: &mut Array<F>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
    }
}

/// A small windowed FIR (up to 3 taps), generic over its element type *and*
/// pinning the active tap count via `#[comptime]` — the milestone's
/// headline case: a genuinely generic + comptime kernel that still lands a
/// **proved** out-of-bounds-freedom claim (see
/// `fir3_kernel_definition_is_provably_in_bounds` below), not just a
/// differential one. Deliberately avoids a loop-carried accumulator (the
/// dogfood survey's Tier-2 prover gap, docs/dogfood-2026-07.md) by fully
/// guarding each of the (at most) two extra taps with its own condition
/// rather than a `for k in 0..taps` loop.
///
/// Guards are written as a single `&&`-composed condition (`taps > 1 &&
/// ABSOLUTE_POS >= 1`) — the natural, idiomatic shape. This used to require
/// two nested `if`s instead (see `crates/vericl-ir/src/prover.rs`'s
/// `nested_if_guard_still_proves` test, which pins that the nested-if shape
/// still proves too): the SMT bounds prover didn't model `&&`-composed
/// branch conditions and reported `OutOfSubset` for them, confirmed
/// empirically at the time by collapsing this exact kernel. Boolean
/// condition composition (`Operator::And`/`Or`/`Not`, all eagerly evaluated
/// by CubeCL rather than lowered to nested branches — see
/// docs/ir-research.md §3) is now modeled, so the collapsed form proves.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0),
    instantiate(F = f32, taps = 3)
)]
#[cube(launch)]
pub fn fir3<F: Float>(x: &Array<F>, y: &mut Array<F>, #[comptime] taps: u32) {
    if ABSOLUTE_POS < y.len() {
        let mut acc = x[ABSOLUTE_POS];
        if taps > 1 && ABSOLUTE_POS >= 1 {
            acc += x[ABSOLUTE_POS - 1];
        }
        if taps > 2 && ABSOLUTE_POS >= 2 {
            acc += x[ABSOLUTE_POS - 2];
        }
        y[ABSOLUTE_POS] = acc;
    }
}

/// Same shape as `fir3`, pinned at a different comptime value purely to
/// demonstrate that changing an `instantiate(...)` value changes kernel
/// identity (`SOURCE_HASH`) — see
/// `fir3_identity_changes_with_instantiate_value` below. Not part of the
/// conformance suite. Deliberately kept on the *nested*-`if` guard shape
/// (rather than following `fir3`'s move to `&&`) so the two kernels'
/// source text differs only in the pinned `taps` value and this guard
/// shape, not in both — the hash-differs test is about `instantiate(...)`
/// specifically, and nested-`if` provability is independently pinned by
/// `crates/vericl-ir/src/prover.rs`'s `nested_if_guard_still_proves`.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0),
    instantiate(F = f32, taps = 1)
)]
#[cube(launch)]
pub fn fir3_alt<F: Float>(x: &Array<F>, y: &mut Array<F>, #[comptime] taps: u32) {
    if ABSOLUTE_POS < y.len() {
        let mut acc = x[ABSOLUTE_POS];
        #[allow(clippy::collapsible_if)]
        if taps > 1 {
            if ABSOLUTE_POS >= 1 {
                acc += x[ABSOLUTE_POS - 1];
            }
        }
        #[allow(clippy::collapsible_if)]
        if taps > 2 {
            if ABSOLUTE_POS >= 2 {
                acc += x[ABSOLUTE_POS - 2];
            }
        }
        y[ABSOLUTE_POS] = acc;
    }
}

/// One xorshift32 step per element — integer, bit-exact, RNG-flavored
/// (the non-Substrate-shaped example).
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(exact)
)]
#[cube(launch)]
pub fn xorshift_step(x: &Array<u32>, y: &mut Array<u32>) {
    if ABSOLUTE_POS < y.len() {
        let mut s = x[ABSOLUTE_POS];
        s ^= s << 13u32;
        s ^= s >> 17u32;
        s ^= s << 5u32;
        y[ABSOLUTE_POS] = s;
    }
}

/// Murmur3 `fmix32`-style integer mixer, one element per thread — integer,
/// bit-exact, and relies on wrap-on-overflow: the multiplies use large odd
/// constants that routinely overflow `u32`, and WGSL wraps on overflow where
/// Rust's default (debug) arithmetic panics. Same finding class as the fma
/// story in the README ("A first finding"): the fix is the declared
/// `wrapping` contract clause below, which folds the reference twin's
/// `*`/`>>` to `wrapping_mul`/`wrapping_shr` — not a silent approximation.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(exact),
    wrapping
)]
#[cube(launch)]
pub fn mix_u32(x: &Array<u32>, y: &mut Array<u32>) {
    if ABSOLUTE_POS < y.len() {
        let mut h = x[ABSOLUTE_POS];
        h ^= h >> 16u32;
        h *= 0x85ebca6bu32;
        h ^= h >> 13u32;
        h *= 0xc2b2ae35u32;
        h ^= h >> 16u32;
        y[ABSOLUTE_POS] = h;
    }
}

/// Flat 1-D `ABSOLUTE_POS` decoded into a `(row, col)` pair via `/`/`%`,
/// recombined into a write index, and scaled — the milestone's div/mod
/// headline (docs/dogfood-2026-07.md Tier-2 gap #2, candidate
/// `flatten_decode_scale`): a clean-room stand-in for the dogfood survey's
/// "flat 1-D → row/col decode" shape (7/22 kernels blocked on it), pinning
/// the div/mod prover boundary with a genuine, publicly-committable
/// example. `width` is a plain runtime `u32` parameter, not a `#[comptime]`
/// constant — the div/mod modeling has to hold for a symbolic divisor, not
/// just a literal.
///
/// The guard is a single `ABSOLUTE_POS < y.len()` (the same axpy-shaped
/// bound as every other honest kernel here) plus `width >= 1u32`, which is
/// what actually matters for the *proof*: it's exactly the fact
/// `vericl-ir`'s div/mod side-obligation needs to discharge (divisor
/// nonzero; both operands nonnegative comes for free — `ABSOLUTE_POS` and
/// `width` are both unsigned leaves). The write index is the *recombined*
/// `row * width + col`, not `ABSOLUTE_POS` directly — proving it in bounds
/// requires the SMT solver to derive `row * width + col == ABSOLUTE_POS`
/// from the SMT-LIB `div`/`mod` (Euclidean) axioms and combine that with
/// the `ABSOLUTE_POS < y.len()` guard, which is the actual boundary this
/// example pins (see `flatten_decode_scale_kernel_definition_is_provably_in_bounds`
/// below). Euclidean division coincides with Rust's/WGSL's truncated
/// semantics exactly when both operands are nonnegative (see
/// `vericl-ir`'s module docs) — true here by construction, so the
/// differential reference and the real kernel compute identically.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-4),
    gen(x in -100.0..=100.0, y in 0.0..=0.0, width in 1..=64, scale in 0.1..=4.0)
)]
#[cube(launch)]
pub fn flatten_decode_scale(x: &Array<f32>, y: &mut Array<f32>, width: u32, scale: f32) {
    if ABSOLUTE_POS < y.len() && width >= 1u32 {
        let w = width as usize;
        let row = ABSOLUTE_POS / w;
        let col = ABSOLUTE_POS % w;
        y[row * w + col] = x[ABSOLUTE_POS] * scale;
    }
}

// ---------------------------------------------------------------------------
// Array-value-dependent indices (offset tables / gather) — the last Tier-2
// prover gap (docs/dogfood-2026-07.md, ≥5 kernels). A read of an *offset* array
// produces a value the checker cannot know, so it is normally tainted and any
// `x[offsets[i]]` index is `OutOfSubset`. An element-range `assumes(...)` clause
// — `offsets.iter().all(|v| (*v as usize) < x.len())` — lets that read be
// modeled as a fresh symbol bounded by the assumption, so the gather's inner
// index obligation discharges (see `crates/vericl-ir/src/prover.rs`'s
// "Element-range assumptions"). The bound is stated once: it doubles as the
// `gen(...)` range for `offsets`, so the differential lane draws satisfying
// offset tables with no separate `gen(offsets in ..)` clause.
// ---------------------------------------------------------------------------

/// Gather: `y[i] = x[offsets[i]]` — the milestone's headline
/// (docs/dogfood-2026-07.md candidate #2), pinning the array-value-dependent
/// index boundary with a publicly-committable example. The element-range assume
/// `offsets.iter().all(|v| (*v as usize) < x.len())` is what makes the inner
/// `x[offsets[i]]` read provable: without it the loaded offset is opaque and the
/// index is `OutOfSubset`. `offsets.len() == y.len()` covers the *outer*
/// `offsets[i]` read under the `ABSOLUTE_POS < y.len()` guard. The comparison is
/// bit-exact (`max_ulp = 0`): a gather is a pure memory permutation, no
/// arithmetic to contract. `offsets` needs no `gen(...)` range — it is derived
/// from the element assume (drawn in `[0, x.len())`), so the whole differential
/// lane exercises in-bounds offset tables automatically. Wired into
/// `vericl::suite!` — carries `tested` (differential) + `proved` (3-obligation
/// SMT bounds) claims. See `gather_copy_kernel_definition_is_provably_in_bounds`.
#[vericl::kernel(
    assumes(
        offsets.len() == y.len(),
        offsets.iter().all(|v| (*v as usize) < x.len()),
    ),
    compare(max_ulp = 0),
    gen(x in -10.0..=10.0, y in 0.0..=0.0)
)]
#[cube(launch)]
pub fn gather_copy(x: &Array<f32>, offsets: &Array<u32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = x[offsets[ABSOLUTE_POS] as usize];
    }
}

/// Nested / two-level gather: `y[i] = data[inner[outer[i]]]`. Pins that element
/// assumes *compose* — the fresh symbol `outer[i]` yields (bounded `< inner`) is
/// exactly the index `inner[·]` needs, whose own fresh symbol (bounded `<
/// data`) is what `data[·]` needs — with no special casing in the prover.
/// Prover-only control (like `tap_pair_guarded_kernel`), not suite-wired; see
/// `nested_gather_kernel_definition_is_provably_in_bounds`.
#[vericl::kernel(
    assumes(
        outer.len() == y.len(),
        outer.iter().all(|v| (*v as usize) < inner.len()),
        inner.iter().all(|v| (*v as usize) < data.len()),
    ),
    compare(max_ulp = 0),
    gen(data in -10.0..=10.0, y in 0.0..=0.0)
)]
#[cube(launch)]
pub fn nested_gather(
    data: &Array<f32>,
    inner: &Array<u32>,
    outer: &Array<u32>,
    y: &mut Array<f32>,
) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = data[inner[outer[ABSOLUTE_POS] as usize] as usize];
    }
}

/// DEFECTIVE (bounds): the declared element bound is a stale constant (`< 16`)
/// looser than the data array it indexes (`x.len() == 8`), so an offset in
/// `[8, 16)` reads out of bounds. The bounds prover models the offset value
/// `< 16` and *refutes* the `x[offsets[i]]` obligation with the fresh element
/// symbol pinned at the boundary (`elem == x.len()`). This is the
/// value-dependent-index defect the demo catches deterministically by proof
/// (unlike a differential run, where the OOB surfaces only for offsets that
/// happen to be drawn `>= 8`); it belongs to the `conform` binary's
/// demo-defects mode, not the honest suite.
#[vericl::kernel(
    assumes(
        offsets.len() == y.len(),
        x.len() == 8,
        offsets.iter().all(|v| (*v as usize) < 16),
    ),
    compare(max_ulp = 0),
    gen(x in -10.0..=10.0, y in 0.0..=0.0, len(x = 8))
)]
#[cube(launch)]
pub fn gather_oob(x: &Array<f32>, offsets: &Array<u32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = x[offsets[ABSOLUTE_POS] as usize];
    }
}

// ---------------------------------------------------------------------------
// match / Switch (quick-wins batch 1) — a Rust `match` on an integer scalar
// lowers to `Branch::Switch`, modeled by the prover as an exhaustive if-chain
// (crates/vericl-ir/src/prover.rs, "Switch modeling"). Each arm is bounds-
// checked under its own path condition `mode == case_i`; the default under the
// conjunction of the negations.
// ---------------------------------------------------------------------------

/// Mode-selected elementwise op: a `match` on the scalar `mode` selects the
/// transform applied to each element (identity / negate / double). The guard
/// `ABSOLUTE_POS < y.len()` bounds every arm's `x`/`y` access; the `match`
/// lowers to a `Branch::Switch` whose three arms (case 0, case 1, default) the
/// prover bounds-checks individually. Every op is a single exact f32 operation
/// (copy, negate, ×2 — no fma contraction possible), so the comparison is
/// bit-exact (`max_ulp = 0`). `mode` is drawn across `0..=3` so the two cases
/// and the default arm are all exercised on the differential lane. Wired into
/// `vericl::suite!` — carries `tested` (differential) + `proved` (6-obligation
/// SMT bounds: 3 arms × {`x` read, `y` write}) claims.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(max_ulp = 0),
    gen(mode in 0..=3, x in -10.0..=10.0, y in 0.0..=0.0)
)]
#[cube(launch)]
pub fn select_mode(mode: u32, x: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        match mode {
            0 => {
                y[ABSOLUTE_POS] = x[ABSOLUTE_POS];
            }
            1 => {
                y[ABSOLUTE_POS] = -x[ABSOLUTE_POS];
            }
            _ => {
                y[ABSOLUTE_POS] = x[ABSOLUTE_POS] * 2.0f32;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Length-relationship assume (quick-wins batch 1) — `A.len() + K <= B.len()`,
// the "additive anchor" host-side buffer-sizing invariant. An offset read
// `x[i + K]` guarded only by `i < y.len()` is in bounds exactly when
// `y.len() + K <= x.len()`; the new `StructuredAssume::LenPlusConstLe` supplies
// that relationship for the prover.
// ---------------------------------------------------------------------------

/// Offset-window sum: `y[i] = x[i] + x[i + 4]`, guarded by `i < y.len()`. The
/// forward read `x[i + 4]` needs `i + 4 < x.len()`, which the length-
/// relationship assume `y.len() + 4 <= x.len()` supplies (combined with the
/// guard `i < y.len()`). This is the additive-anchor shape from the dogfood
/// findings, isolated to the pure length relationship. `gen(... len(x = n + 4))`
/// sizes `x` four elements longer than the dispatch/`y`, satisfying the
/// declared invariant on every differential case. A single f32 add is bit-exact
/// vs the backend (`max_ulp = 0`). Wired into `vericl::suite!` — carries
/// `tested` (differential) + `proved` (3-obligation SMT bounds: `x[i]`,
/// `x[i + 4]`, `y[i]`) claims.
#[vericl::kernel(
    assumes(y.len() + 4 <= x.len()),
    compare(max_ulp = 0),
    gen(x in -10.0..=10.0, y in 0.0..=0.0, len(x = n + 4))
)]
#[cube(launch)]
pub fn offset_window(x: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = x[ABSOLUTE_POS] + x[ABSOLUTE_POS + 4usize];
    }
}

// ---------------------------------------------------------------------------
// Core `Slice` (docs/design-view-slice.md) — the #2 ecosystem gap's tractable
// half. A slice is a pure *addressing view*: `arr.slice(a, b)[i]` lowers to a
// checked `origin[a + i]` (§2.1), so bounds proving is the ordinary origin
// obligation, unchanged (§5, deliverable B). The twin maps a slice to a Rust
// subslice (`&arr[a..b]`), which is bit-exact (a slice adds no numeric op,
// §4.2/§6) and makes Rust the soundness oracle for slice-creation validity
// (an out-of-range `&arr[a..b]` PANICS in the tested twin, §4.4) and for
// mutable aliasing (overlapping live `&mut` subslices do not compile, §4.3).
// ---------------------------------------------------------------------------

/// Windowed sum through a slice: `y[i] = Σ x.slice(i, i+4)`, guarded `i <
/// y.len()`. Exercises the whole core-slice surface end to end — dynamic-offset
/// slice **creation**, **iteration** (`for v in slice`, a `RangeLoop` over
/// `x[i + j]`, §2.2), and **length** — proving that the addressing view is
/// transparent to both lanes. The reads `x[i .. i+4)` are bounded by the same
/// `y.len() + 4 <= x.len()` length relationship the indexed `offset_window`
/// uses, combined with the `i < y.len()` guard; `gen(... len(x = n + 4))` sizes
/// `x` four longer so every window is in range and the twin's `&x[i..i+4]`
/// never panics. The window sum is a fixed left-to-right sequence of exact f32
/// adds (no fma contraction), bit-exact vs the backend (`max_ulp = 0`, the
/// design's §6 `windowed_sum` ground-truth result). Wired into `vericl::suite!`
/// — carries `tested` (differential) + `proved` (SMT bounds) claims.
#[vericl::kernel(
    assumes(y.len() + 4 <= x.len()),
    compare(max_ulp = 0),
    gen(x in -10.0..=10.0, y in 0.0..=0.0, len(x = n + 4))
)]
#[cube(launch)]
pub fn windowed_slice_sum(x: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        let mut acc = f32::new(0.0);
        for v in x.slice(ABSOLUTE_POS, ABSOLUTE_POS + 4) {
            acc += v;
        }
        y[ABSOLUTE_POS] = acc;
    }
}

/// Gather **through** a slice of an element-assumed array: `y[i] =
/// x[offsets.to_slice()[i]]` — the `gather_copy` shape, but reading the offset
/// table through a whole-buffer `to_slice()` (offset 0). Because the slice read
/// lowers to a read of the **origin** buffer id, the `offsets.iter().all(...)`
/// element assume keys off that same id and transfers **for free** (§5.4), so
/// the loaded index is modeled `< x.len()` with no slice-specific prover code.
/// A slice adds no numeric op, so the copy is bit-exact (`max_ulp = 0`). Wired
/// into `vericl::suite!` — carries `tested` + `proved` (3-obligation SMT
/// bounds: `offsets[i]` read, `x[·]` read, `y[i]` write). This doubles as the
/// "re-annotate an already-provable kernel to read through a `to_slice()`"
/// demonstration (the addressing view is transparent to the gather machinery).
#[vericl::kernel(
    assumes(
        offsets.len() == y.len(),
        offsets.iter().all(|v| (*v as usize) < x.len()),
    ),
    compare(max_ulp = 0),
    gen(x in -10.0..=10.0, y in 0.0..=0.0)
)]
#[cube(launch)]
pub fn slice_gather_copy(x: &Array<f32>, offsets: &Array<u32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        let s = offsets.to_slice();
        y[ABSOLUTE_POS] = x[s[ABSOLUTE_POS] as usize];
    }
}

/// Reads the edges of a slice window — the composition boundary probe for
/// slices, analogous to `tap_pair` for arrays. The array access lives inside
/// the helper's own body (`w[0] + w[3]`), over a `&Slice<F>` param (the
/// idiomatic real-cubek helper-param form; the twin maps `&Slice<F>` -> `&[f32]`,
/// §10). Pins that the SMT bounds prover, walking the caller's inlined IR,
/// discharges the origin obligation that only exists because of what a composed
/// slice helper does.
#[vericl::helper(instantiate(F = f32))]
#[cube]
pub fn window_edge_sum<F: Float>(w: &Slice<F>) -> F {
    w[0] + w[3]
}

/// Composed slice kernel: `y[i] = window_edge_sum(&x.slice(i, i+4))` — the
/// dominant real slice-composition usage (a `#[vericl::helper]` taking a slice,
/// §3, §10). The idiomatic `&x.slice(a, b)` argument's redundant outer `&`
/// collapses in the twin so it matches the helper twin's `&[f32]` param. Under
/// the same `y.len() + 4 <= x.len()` relationship, both edge reads (`x[i]`,
/// `x[i+3]`) prove in the caller's inlined IR. Bit-exact single-add sum
/// (`max_ulp = 0`). Wired into `vericl::suite!` — carries `tested` + `proved`.
#[vericl::kernel(
    assumes(y.len() + 4 <= x.len()),
    compare(max_ulp = 0),
    gen(x in -10.0..=10.0, y in 0.0..=0.0, len(x = n + 4)),
    uses(window_edge_sum)
)]
#[cube(launch)]
pub fn windowed_helper_kernel(x: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = window_edge_sum::<f32>(&x.slice(ABSOLUTE_POS, ABSOLUTE_POS + 4));
    }
}

/// In-place scale **through a mutable slice** — the write-path end-to-end
/// example (F1, round-9). Every other committed slice example reads a
/// `Slice`/`to_slice()`; this one exercises the `slice_mut` **write** lane the
/// twin maps to `&mut arr[a..b]` (design §2.4/§4.1). Each thread owns the
/// **disjoint** single-element window `y.slice_mut(ABSOLUTE_POS, ABSOLUTE_POS +
/// 1)` and scales it in place (`s[0] = s[0] * alpha`): the exact write-path
/// mirror of `windowed_slice_sum`'s dynamic-offset read window, one element wide
/// so the per-thread windows never overlap — the differential is deterministic
/// (thread `i` is the sole writer of `y[i]`) and **bit-exact** (`max_ulp = 0`: a
/// single correctly-rounded multiply on both lanes). Guarded `ABSOLUTE_POS <
/// y.len()`, the write lowers to the ordinary origin obligation `IndexAssign(y,
/// ABSOLUTE_POS + 0)` and **proves** with no assume (deliverable B is a no-op
/// for the prover on the write path too, §5). Wired into `vericl::suite!` —
/// carries `tested` (differential) + `proved` (SMT bounds).
///
/// A **multi-element** `slice_mut(a, b)[j]` write window and the
/// sequential-vs-overlapping aliasing convention are exercised by
/// `sequential_slice_mut_scale` below (a wider window cannot be a disjoint
/// *provable* suite differential: disjoint blocks need a `start = i*W` stride
/// whose `checked_mul` overflow side-obligation is unbounded, so it would be
/// `OutOfSubset`, and an overlapping window is a write-order-dependent race — so
/// the suite example stays one element wide, and the wider window is a twin
/// unit test at a fixed length).
#[vericl::kernel(
    compare(max_ulp = 0),
    gen(y in -100.0..=100.0, alpha in -4.0..=4.0)
)]
#[cube(launch)]
pub fn slice_scale_inplace(y: &mut Array<f32>, alpha: f32) {
    if ABSOLUTE_POS < y.len() {
        let mut s = y.slice_mut(ABSOLUTE_POS, ABSOLUTE_POS + 1);
        s[0] = s[0] * alpha;
    }
}

/// The S3-milestone mutable-aliasing convention control (docs/design-view-slice.md
/// §4.3, §11 S3, F1 round-9) **and** the multi-element `slice_mut(a, b)[j]` write
/// window. Thread 0 scales two **disjoint, sequentially-created** mutable windows
/// of one origin: `y.slice_mut(0, 4)` (used and dropped) then `y.slice_mut(4, 8)`.
/// Because each `&mut (y)[a..b]` twin is created, used, and dropped before the
/// next, the borrow checker (the aliasing ORACLE, §4.3) accepts the twin under
/// NLL — **sequential mutable slices compile**, the dominant real shape. This is
/// the *positive* control of the pair; the *negative* control is
/// `scratchpad/slicemut/overlap.rs`, where two **simultaneously-live overlapping**
/// `slice_mut` views of one origin fail to compile with rustc `E0499` — the borrow
/// error IS the rejection (as-built; the prettified §8.3 macro message is deferred,
/// §8.4). The pair is `scratchpad/slicemut/{sequential_ok,overlap}.rs`.
///
/// Not suite-wired: it is single-threaded (thread 0) over a fixed 8-wide layout, so
/// its twin panics on a shorter origin — the multi-size differential does not apply.
/// Its twin is pinned at a fixed length by `sequential_slice_mut_scale_twin_scales_two_windows`.
#[vericl::kernel(
    compare(max_ulp = 0),
    gen(y in -100.0..=100.0, alpha in -4.0..=4.0)
)]
#[cube(launch)]
pub fn sequential_slice_mut_scale(y: &mut Array<f32>, alpha: f32) {
    if ABSOLUTE_POS == 0 {
        let mut lo = y.slice_mut(0, 4);
        for j in 0..4usize {
            lo[j] = lo[j] * alpha;
        }
        let mut hi = y.slice_mut(4, 8);
        for j in 0..4usize {
            hi[j] = hi[j] * alpha;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel composition: #[vericl::helper] + uses(...) — the milestone this
// section demonstrates (docs/dogfood-2026-07.md Tier-1 gap #2, blocking
// 16/22 dogfooded kernels). See README "Kernel composition" for the design.
//
// A note on why every helper below is fully monomorphized via its own
// instantiate(...) clause rather than left generic (a first draft of this
// design tried the latter): `FLOAT_METHOD_WHITELIST`'s host-safety proof
// (crates/vericl-macros' module doc) relies on Rust preferring an inherent
// method over a trait method for a *concrete* receiver type. A bound-but-
// unsubstituted generic type parameter `F: Float` has no inherent methods
// at all, so a whitelisted call like `.abs()`/`.sqrt()` inside a still-
// generic host fn resolves purely through the `Float`/`Numeric` trait,
// which (confirmed by reading cubecl-core's `impl_unary_func!` macro, and
// empirically: a scratch generic `fn g<F: Float>(x: F) -> F { x.sqrt() }`
// panics on host calling `g(2.5f32)`, as does `.abs()`) is exactly the
// unverified `unexpanded!()` panic path the whitelist exists to keep out of
// a twin. Monomorphizing via instantiate(...) — reusing the exact same
// resolve_instantiate/FloatMethodCheck machinery a kernel already uses —
// closes that gap the same proven way instead of introducing a second,
// weaker safety story. The cost: a helper's twin is pinned to one concrete
// type (only `f32` is supported anywhere in vericl v0 today, so this is
// free in practice); a `#[comptime]` parameter, unlike a generic type
// parameter, is never pinned by a helper's instantiate(...) — it stays an
// ordinary pass-through parameter, since it carries no host-callability
// hazard (it's just a plain integer value) and the caller's own twin
// already has the concrete pinned value in hand to pass along.

/// Scales one scalar by a gain factor — the simplest possible helper (no
/// array parameters, no other helper calls), reused by two kernels below
/// (`gain_kernel` directly, `fir_pair_scaled` transitively).
#[vericl::helper(instantiate(F = f32))]
#[cube]
pub fn single_tap<F: Float>(a: F, gain: F) -> F {
    a * gain
}

/// A 2-tap FIR pair returning a tuple — the milestone's headline "helper
/// returning a tuple" shape (docs/dogfood-2026-07.md's suggested candidate
/// example). Tuple returns are plain Rust; no special handling needed.
#[vericl::helper(instantiate(F = f32))]
#[cube]
pub fn fir_pair<F: Float>(a: F, b: F) -> (F, F) {
    (a + b, a - b)
}

/// Calls two OTHER `#[vericl::helper]`-annotated functions —
/// helper-calling-helper composition, supported via the exact same
/// `uses(...)` rewrite mechanism a kernel gets (no special-casing). Its
/// twin's identity recursively folds in both `fir_pair`'s and
/// `single_tap`'s (see `fir_pair_scaled_vericl::identity_hash`).
///
/// Uses the natural **tuple-`let` destructuring** of the composed call's return
/// (`let (sum, diff) = fir_pair::<F>(a, b)`) — the residual noted in
/// docs/dogfood-2026-07.md (the `.0`/`.1` workaround) is closed: the *untyped*
/// tuple pattern is desugared by cube's own `Desugar` pass and preserved by the
/// twin's `uses(...)` rewrite, in a device-fn-calling-device-fn body. (A *typed*
/// tuple `let (a, b): (F, F) = …` remains a cubecl desugar limitation — its
/// `Desugar` matches a bare `Pat::Tuple` but not one wrapped in `Pat::Type` — and
/// is out of vericl's scope, since vericl re-emits the `#[cube]` body untouched.)
#[vericl::helper(instantiate(F = f32), uses(fir_pair, single_tap))]
#[cube]
pub fn fir_pair_scaled<F: Float>(a: F, b: F, gain: F) -> (F, F) {
    let (sum, diff) = fir_pair::<F>(a, b);
    (single_tap::<F>(sum, gain), single_tap::<F>(diff, gain))
}

/// Reads a value AND its right neighbor — unlike the helpers above, the
/// array access genuinely lives *inside the helper's own body*, not the
/// caller's. Pins the prover-composition boundary (see
/// `tap_pair_guarded_kernel`/`tap_pair_unguarded_kernel` below): whether the
/// SMT bounds prover, walking a kernel's inlined IR, discharges an
/// obligation that only exists because of what a composed helper does.
#[vericl::helper(instantiate(F = f32))]
#[cube]
pub fn tap_pair<F: Float>(x: &Array<F>, idx: usize) -> F {
    x[idx] + x[idx + 1]
}

/// Composed kernel A: calls `single_tap` directly. Wired into
/// `vericl::suite!` — carries both `tested` (differential) and `proved`
/// (SMT bounds) claims, same as any honest non-composed kernel; composition
/// needed zero prover changes (cube expansion inlines the helper's IR
/// directly into this kernel's own `Scope` — see
/// `crates/vericl-ir/src/prover.rs`'s module doc, "Soundness notes").
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0, gain in -4.0..=4.0),
    instantiate(F = f32),
    uses(single_tap)
)]
#[cube(launch)]
pub fn gain_kernel<F: Float + CubeElement>(x: &Array<F>, y: &mut Array<F>, gain: F) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = single_tap::<F>(x[ABSOLUTE_POS], gain);
    }
}

/// Fix 3 regression pin (round-2 adversarial review, `UsesRewriteFold`
/// multi-segment call bypass): identical to `gain_kernel` above, except the
/// call to `single_tap` is `self::`-qualified. Before the fix, a
/// multi-segment path bypassed both the rewrite AND the unlisted-callee
/// rejection entirely, so the twin silently called the ORIGINAL `#[cube]
/// fn single_tap` host-side instead of `single_tap_vericl_ref` — never
/// caught, since (for this host-safe helper) both compute the same answer,
/// making it invisible to a black-box differential check; see
/// `self_path_gain_kernel_twin_matches_hand_computed` below, which pins the
/// fix at the AST level via `gain_kernel_twin_matches_hand_computed`'s own
/// expected values instead. Not suite-wired (no new evidence entry needed
/// — this exists purely to pin the fix, same precedent as
/// `tap_pair_guarded_kernel` below).
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0, gain in -4.0..=4.0),
    instantiate(F = f32),
    uses(single_tap)
)]
#[cube(launch)]
pub fn self_path_gain_kernel<F: Float + CubeElement>(x: &Array<F>, y: &mut Array<F>, gain: F) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = self::single_tap::<F>(x[ABSOLUTE_POS], gain);
    }
}

/// Composed kernel B: calls `fir_pair_scaled`, which itself calls
/// `fir_pair` and `single_tap` — two levels of composition end to end, and
/// `single_tap` reused across both `gain_kernel` (directly) and this kernel
/// (transitively). Two `&mut Array` outputs, one per tuple element — the
/// macro-generated `conformance_case`/comparison machinery already handles
/// N output buffers generically, so this needed no new machinery either.
/// Wired into `vericl::suite!`: the composed kernel carrying tested + proved
/// claims the milestone asks for.
///
/// The guard is `ABSOLUTE_POS < x.len() && ABSOLUTE_POS + 1 < x.len()` — BOTH
/// conjuncts, deliberately. The single `ABSOLUTE_POS + 1 < x.len()` this kernel
/// carried before the overflow-soundness milestone silently relied on `+ 1` not
/// wrapping to also cover the `x[ABSOLUTE_POS]` read (`pos + 1 < len ⟹ pos <
/// len` holds only when `pos + 1` does not overflow — i.e. at every reachable
/// dispatch, but NOT at the adversarial `pos == u32::MAX`, where `pos + 1`
/// wraps to `0`, the guard passes, and `x[pos]` is out of bounds). Once the
/// prover models `u32` wraparound faithfully (crates/vericl-ir's
/// "Bounded-integer overflow model"), that latent reliance is exposed as a
/// `Refuted` on the `x[ABSOLUTE_POS]` read at `pos == u32::MAX`. Stating both
/// conjuncts makes the two reads each provable from their own guard and
/// excludes the wrap point — the honest strengthening (safe at every reachable
/// dispatch either way; see the README "Overflow soundness" note).
#[vericl::kernel(
    assumes(x.len() == sum_out.len(), x.len() == diff_out.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, sum_out in 0.0..=0.0, diff_out in 0.0..=0.0, gain in -4.0..=4.0),
    instantiate(F = f32),
    uses(fir_pair_scaled)
)]
#[cube(launch)]
pub fn fir_pair_kernel<F: Float + CubeElement>(
    x: &Array<F>,
    sum_out: &mut Array<F>,
    diff_out: &mut Array<F>,
    gain: F,
) {
    if ABSOLUTE_POS < x.len() && ABSOLUTE_POS + 1 < x.len() {
        let s_d: (F, F) = fir_pair_scaled::<F>(x[ABSOLUTE_POS], x[ABSOLUTE_POS + 1], gain);
        sum_out[ABSOLUTE_POS] = s_d.0;
        diff_out[ABSOLUTE_POS] = s_d.1;
    }
}

/// Prover-composition positive control (docs/dogfood-2026-07.md-style, not
/// wired into `vericl::suite!` — mirrors the `stepped_loop_*` precedent
/// below of a kernel that exists purely to pin a prover finding, not to
/// carry evidence): the guard `ABSOLUTE_POS < x.len() && ABSOLUTE_POS + 1 <
/// x.len()` covers BOTH reads `tap_pair`'s own body performs (`x[idx]`,
/// `x[idx + 1]`) even though those accesses live inside the composed helper,
/// not here. Both conjuncts are stated for the same reason `fir_pair_kernel`
/// (above) states them: under the faithful `u32` overflow model a lone
/// `ABSOLUTE_POS + 1 < x.len()` no longer implies `ABSOLUTE_POS < x.len()` at
/// the adversarial `pos == u32::MAX` wrap point (where the helper's own
/// `x[idx]` read would be out of bounds). Must discharge `Proved` — see
/// `tap_pair_guarded_kernel_definition_is_provably_in_bounds` below.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0),
    instantiate(F = f32),
    uses(tap_pair)
)]
#[cube(launch)]
pub fn tap_pair_guarded_kernel<F: Float + CubeElement>(x: &Array<F>, y: &mut Array<F>) {
    if ABSOLUTE_POS < x.len() && ABSOLUTE_POS + 1 < x.len() {
        y[ABSOLUTE_POS] = tap_pair::<F>(x, ABSOLUTE_POS);
    }
}

/// Prover-composition negative control: same shape as
/// `tap_pair_guarded_kernel` and the same helper (`tap_pair` — demonstrating
/// helper reuse together with the kernel above), but the guard only
/// establishes `ABSOLUTE_POS < x.len()`, one short of what `tap_pair`'s own
/// unguarded `x[idx + 1]` read needs at the top position. Must `Refuted`,
/// never `Proved` — the obligation genuinely lives inside the helper's
/// body, and the prover must not silently drop it just because it's
/// composed rather than written directly in the kernel.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(abs = 1e-5),
    gen(x in -10.0..=10.0, y in 0.0..=0.0),
    instantiate(F = f32),
    uses(tap_pair)
)]
#[cube(launch)]
pub fn tap_pair_unguarded_kernel<F: Float + CubeElement>(x: &Array<F>, y: &mut Array<F>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = tap_pair::<F>(x, ABSOLUTE_POS);
    }
}

/// DEFECTIVE: boundary guard is `<=`, reading and writing one element past
/// the end. WGSL robustness silently clamps this on the GPU, so a GPU-only
/// test can pass; the sequential reference panics deterministically.
#[vericl::kernel(
    assumes(x.len() == y.len(), alpha.abs() <= 4.0),
    compare(abs = 1e-4),
    gen(alpha in -4.0..=4.0, x in -100.0..=100.0, y in -100.0..=100.0)
)]
#[cube(launch)]
pub fn axpy_off_by_one(alpha: f32, x: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS <= y.len() {
        y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
    }
}

/// DEFECTIVE: unsynchronized accumulation — every thread read-modify-writes
/// `y[0]` with no atomics. The sequential reference computes the true sum;
/// the GPU result is whatever the race leaves behind.
#[vericl::kernel(
    assumes(y.len() == 1),
    compare(max_ulp = 0),
    gen(x in 0.5..=1.5, y in 0.0..=0.0, len(y = 1))
)]
#[cube(launch)]
pub fn sum_racy(x: &Array<f32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < x.len() {
        y[0] += x[ABSOLUTE_POS];
    }
}

/// REGRESSION (adversarial soundness review, Bug 1 — see
/// `vericl_ir::prover::process_range_loop`): `range_stepped` with a negative
/// step produces a descending loop (`start > end` numerically). The SMT
/// prover must reject this outright rather than silently assert ascending
/// bounds, which for a real descending loop are unsatisfiable and would make
/// every obligation inside vacuously "provable". This kernel's body is an
/// ordinary in-bounds copy — even so it must not prove: the loop *shape* is
/// outside the modeled subset, independent of whether the body happens to be
/// safe. Not exercised by `conform`'s differential/evidence pipeline
/// (never GPU-launched); used only by the prover regression tests below.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(exact)
)]
#[cube(launch)]
pub fn stepped_loop_descending_copy(x: &Array<u32>, y: &mut Array<u32>) {
    let n = y.len() as i32;
    for i in range_stepped(n - 1, -1, -1) {
        let idx = i as usize;
        y[idx] = x[idx];
    }
}

/// REGRESSION (adversarial soundness review, Bug 1): the exact vacuous-proof
/// shape the review demonstrated — a runtime-bounded, negative-step loop
/// whose body writes far out of bounds (`y[100000]`). Before the fix,
/// `process_range_loop` asserted ascending bounds for this descending loop,
/// making the SMT context infeasible and vacuously discharging the
/// `y[100000]`/`x[100000]` obligations as "proved" even though a real
/// (sequential) execution of this loop panics out-of-bounds. Must now
/// return `OutOfSubset`, never `Proved`. Not exercised by `conform`'s
/// differential/evidence pipeline (never GPU-launched); used only by the
/// prover regression tests below.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(exact)
)]
#[cube(launch)]
pub fn stepped_loop_oob_write(x: &Array<u32>, y: &mut Array<u32>) {
    let n = y.len() as i32;
    for _i in range_stepped(n - 1, -1, -1) {
        y[100000] = x[100000];
    }
}

// ===================================================================
// Cooperative (workgroup shared-memory) reduction kernels — the
// shared-memory milestone M5 clean-room probes (docs/design-shared-
// memory.md §3, §4.6). Deliberately NOT wired into `vericl::suite!` yet:
// the coupling of the differential claim to the race-freedom proof is M6
// and the suite wiring is M7. They exist here so the *generated*
// phase-split cooperative twin (coop.rs) can be differential-tested
// bit-exact against wgpu (see the `*_coop_*` tests below).
// ===================================================================

/// `block_sum_reduce` — the v1 reduction shape (docs/design-shared-memory.md
/// §3): block-strided load into a per-cube `SharedMemory` tile, one barrier,
/// a uniform tree reduction (one barrier per level), and a single-writer
/// per-cube partial store guarded by `tid == 0`. One partial per workgroup, so
/// the output is sized `cube_count` (not the flat thread count) — the
/// cooperative launch/output model (§7.1).
///
/// The macro's `cooperative(cube_dim = 256)` clause opts this kernel into the
/// phase-split twin: the body is split at each `sync_cube()` into segments, run
/// per cube / per segment / per `unit_pos`, with the tile a poison-initialised
/// per-cube array (§4.5). Bit-exact vs wgpu because the twin sums in the
/// identical tree order (§4.6).
#[vericl::kernel(
    assumes(input.iter().all(|v| v.abs() <= 1000.0)),
    compare(max_ulp = 0),
    gen(input in -1000.0..=1000.0),
    cooperative(cube_dim = 256)
)]
#[cube(launch)]
pub fn block_sum_reduce(input: &Array<f32>, output: &mut Array<f32>) {
    let tid = UNIT_POS as usize;
    let mut tile = SharedMemory::<f32>::new(256usize);
    if ABSOLUTE_POS < input.len() {
        tile[tid] = input[ABSOLUTE_POS];
    } else {
        tile[tid] = 0.0f32;
    }
    sync_cube();

    let mut half = CUBE_DIM as usize / 2;
    while half > 0usize {
        if tid < half {
            let a = tile[tid];
            let b = tile[tid + half];
            tile[tid] = a + b;
        }
        sync_cube();
        half /= 2usize;
    }

    if tid == 0usize && CUBE_POS < output.len() {
        output[CUBE_POS] = tile[0usize];
    }
}

/// `comptime_window_reduce` — the cooperative **#[comptime] parameter** example
/// (docs/design-shared-memory.md §7.4, lifted to v1.1). Each thread sums up to
/// `taps` (a `#[comptime]` value, pinned to 3 by `instantiate(taps = 3)`)
/// consecutive input samples starting at `ABSOLUTE_POS`, then feeds the usual
/// tree reduction. The `#[comptime] taps` drives the **accumulation loop bound**
/// — the exact deferral concern (a comptime value in a loop bound). It is
/// **cube-uniform by construction** (the same compile-time constant for every
/// thread — the easiest uniformity case), so the phase-split twin treats it as
/// an ordinary uniform value (bound as a `let` const) and the two-thread walk /
/// uniform-trip-count check see nothing thread-varying. Bounds prove (the `idx <
/// input.len()` guard bounds each read) and the shape is race-free (each
/// `tile[tid]` written once before the tree combine). Bit-exact vs wgpu (the
/// per-thread sum is a plain add loop in the identical order — no fma).
#[vericl::kernel(
    assumes(input.iter().all(|v| v.abs() <= 100.0)),
    compare(max_ulp = 0),
    gen(input in -100.0..=100.0),
    instantiate(taps = 3),
    cooperative(cube_dim = 256)
)]
#[cube(launch)]
pub fn comptime_window_reduce(input: &Array<f32>, output: &mut Array<f32>, #[comptime] taps: u32) {
    let tid = UNIT_POS as usize;
    let mut local = 0.0f32;
    for j in 0..taps {
        let idx = ABSOLUTE_POS + j as usize;
        if idx < input.len() {
            local += input[idx];
        }
    }

    let mut tile = SharedMemory::<f32>::new(256usize);
    tile[tid] = local;
    sync_cube();

    let mut half = CUBE_DIM as usize / 2;
    while half > 0usize {
        if tid < half {
            let a = tile[tid];
            let b = tile[tid + half];
            tile[tid] = a + b;
        }
        sync_cube();
        half /= 2usize;
    }

    if tid == 0usize && CUBE_POS < output.len() {
        output[CUBE_POS] = tile[0usize];
    }
}

/// A barrier-free device-fn helper used inside a cooperative kernel's phase 0
/// (`composed_sq_reduce` below) — the cooperative-**composition** example
/// (docs/design-shared-memory.md §7.4, lifted to v1.1). It contains no
/// `sync_cube()` and no `SharedMemory` (both rejected in a helper with a
/// targeted error), so it is a pure barrier-free unit the phase splitter can run
/// inside a single segment. Cube inlines its IR into the composing kernel's
/// scope, so the two-thread walk sees the squared value directly.
#[vericl::helper(instantiate(F = f32))]
#[cube]
pub fn square_sample<F: Float>(v: F) -> F {
    v * v
}

/// `composed_sq_reduce` — a cooperative reduction whose phase-0 per-thread load
/// is computed by a `#[vericl::helper]` (`square_sample`), the cooperative
/// **composition** example. Identical reduction skeleton to `block_sum_reduce`;
/// the only difference is the tile is loaded with `square_sample(input[pos])`
/// (a helper call) instead of a bare read. The helper is barrier-free, so the
/// phase split stays local — the twin runs the (rewritten `_vericl_ref`) call
/// inside phase 0, and the prover proves bounds + race-freedom over the inlined
/// IR exactly as for `block_sum_reduce`, with the extra barrier-count check
/// confirming the helper introduced no `sync_cube()`.
#[vericl::kernel(
    assumes(input.iter().all(|v| v.abs() <= 100.0)),
    compare(max_ulp = 0),
    gen(input in -100.0..=100.0),
    cooperative(cube_dim = 256),
    uses(square_sample)
)]
#[cube(launch)]
pub fn composed_sq_reduce(input: &Array<f32>, output: &mut Array<f32>) {
    let tid = UNIT_POS as usize;
    let mut tile = SharedMemory::<f32>::new(256usize);
    if ABSOLUTE_POS < input.len() {
        tile[tid] = square_sample::<f32>(input[ABSOLUTE_POS]);
    } else {
        tile[tid] = 0.0f32;
    }
    sync_cube();

    let mut half = CUBE_DIM as usize / 2;
    while half > 0usize {
        if tid < half {
            let a = tile[tid];
            let b = tile[tid + half];
            tile[tid] = a + b;
        }
        sync_cube();
        half /= 2usize;
    }

    if tid == 0usize && CUBE_POS < output.len() {
        output[CUBE_POS] = tile[0usize];
    }
}

/// `emitter_reduce` — the acceptance example for the cooperative v1.1
/// extensions, the `emitter_powers_multi_rx` shape (docs/design-shared-memory.md
/// §3, minus its 2-D dispatch) reduced to 1-D: it exercises **all three v1.1
/// cooperative extensions at once**, plus shared memory.
///
/// - **`#[comptime]` parameter** (`n_emitters`, pinned to 4): the number of
///   active workgroups, driving the padding guard.
/// - **workgroup-uniform `terminate!()`**: `if CUBE_POS >= n_emitters {
///   terminate!() }` at the top level before any barrier — the "skip the whole
///   cube" padding guard. `CUBE_POS` is cube-uniform, so the whole workgroup
///   terminates together (barrier-safe). Padding cubes (launched beyond
///   `n_emitters`) produce no output; the phase-split twin models this as a
///   cube-level `continue`, and the prover as a `!(CUBE_POS >= n_emitters)` path
///   condition (uniformity verified by the thread-varying taint machinery).
/// - **`uses(...)` composition**: phase 0 loads via the barrier-free
///   `square_sample` helper.
///
/// Bit-exact vs wgpu: `square_sample` is a single product (no fma) and the tree
/// sums in the identical order; padding cubes leave the zero-initialised output
/// untouched on both lanes. Bounds prove (the store keeps its explicit
/// `CUBE_POS < output.len()` guard) and the shape is race-free. The only
/// `emitter_powers` feature left out is 2-D dispatch (per the milestone scope).
#[vericl::kernel(
    assumes(input.iter().all(|v| v.abs() <= 100.0)),
    compare(max_ulp = 0),
    gen(input in -100.0..=100.0),
    instantiate(n_emitters = 4),
    cooperative(cube_dim = 256),
    uses(square_sample)
)]
#[cube(launch)]
pub fn emitter_reduce(input: &Array<f32>, output: &mut Array<f32>, #[comptime] n_emitters: u32) {
    if CUBE_POS >= n_emitters as usize {
        terminate!();
    }

    let tid = UNIT_POS as usize;
    let mut tile = SharedMemory::<f32>::new(256usize);
    if ABSOLUTE_POS < input.len() {
        tile[tid] = square_sample::<f32>(input[ABSOLUTE_POS]);
    } else {
        tile[tid] = 0.0f32;
    }
    sync_cube();

    let mut half = CUBE_DIM as usize / 2;
    while half > 0usize {
        if tid < half {
            let a = tile[tid];
            let b = tile[tid + half];
            tile[tid] = a + b;
        }
        sync_cube();
        half /= 2usize;
    }

    if tid == 0usize && CUBE_POS < output.len() {
        output[CUBE_POS] = tile[0usize];
    }
}

/// `grid_stride_reduce` — the `reduce_rssi`-shaped reduction (docs/design-
/// shared-memory.md §3): a *non-cooperative* grid-stride accumulation loop
/// (`while k < data.len()`, no barrier inside — the shape §4 requires be
/// transformable, appearing before the first barrier) squares and sums a
/// strided slice into a per-thread `local`, which then feeds the same tree
/// reduction as `block_sum_reduce`. Uses the `CUBE_COUNT` builtin for the
/// grid stride (validated runtime value on wgpu), so no extra parameter is
/// needed and the twin's `CUBE_COUNT` binds to the launch cube_count.
///
/// NOT wired into `vericl::suite!` (unlike `block_sum_reduce`): the cubecl-cpu
/// backend does not support the `CUBE_COUNT` builtin ("Unsupported builtin was
/// used: CubeCount"), so this kernel cannot run on the suite's `--features cpu`
/// lane — exactly the portability reason the production `reduce_rssi` passes the
/// grid width as an explicit runtime scalar rather than reading `CUBE_COUNT`.
/// It remains fully covered: bit-exact vs wgpu in `tests/cooperative.rs`, and
/// race-free + in-bounds proved in `grid_stride_reduce_definition_is_race_free`.
#[vericl::kernel(
    assumes(data.iter().all(|v| v.abs() <= 100.0)),
    compare(max_ulp = 0),
    gen(data in -100.0..=100.0),
    cooperative(cube_dim = 256)
)]
#[cube(launch)]
pub fn grid_stride_reduce(data: &Array<f32>, partials: &mut Array<f32>) {
    let tid = UNIT_POS as usize;
    let stride = CUBE_DIM as usize * CUBE_COUNT;
    let n = data.len();
    let mut k = ABSOLUTE_POS;
    let mut local = 0.0f32;
    while k < n {
        local += data[k] * data[k];
        k += stride;
    }

    let mut tile = SharedMemory::<f32>::new(256usize);
    tile[tid] = local;
    sync_cube();

    let mut half = CUBE_DIM as usize / 2;
    while half > 0usize {
        if tid < half {
            let a = tile[tid];
            let b = tile[tid + half];
            tile[tid] = a + b;
        }
        sync_cube();
        half /= 2usize;
    }

    if tid == 0usize && CUBE_POS < partials.len() {
        partials[CUBE_POS] = tile[0usize];
    }
}

/// A deliberately-buggy cooperative kernel that **reads shared memory before
/// writing it** (`tile[tid]` is read in `tile[tid] + input[ABSOLUTE_POS]`
/// before any thread has written `tile[tid]`). On the GPU this reads
/// uninitialised shared memory (garbage); the phase-split twin poison-
/// initialises the tile (docs/design-shared-memory.md §4.5), so its
/// `reference` **panics loudly** on the poison read instead of masking the bug
/// with a zero — demonstrated by `shared_read_before_write_twin_panics_on_
/// poison` below. Not suite-wired (it is a defect probe, never GPU-launched
/// for evidence).
#[vericl::kernel(
    assumes(input.iter().all(|v| v.abs() <= 100.0)),
    compare(max_ulp = 0),
    gen(input in -100.0..=100.0),
    cooperative(cube_dim = 256)
)]
#[cube(launch)]
pub fn shared_read_before_write(input: &Array<f32>, output: &mut Array<f32>) {
    let tid = UNIT_POS as usize;
    let mut tile = SharedMemory::<f32>::new(256usize);
    // BUG: `tile[tid]` is read here before any write to it.
    let acc = tile[tid] + input[ABSOLUTE_POS];
    tile[tid] = acc;
    sync_cube();
    if tid == 0usize && CUBE_POS < output.len() {
        output[CUBE_POS] = tile[0usize];
    }
}

/// DEFECTIVE cooperative twin — the racy variant of `block_sum_reduce`
/// (docs/design-shared-memory.md §5.5 / §8 M7). Its reduction generation uses
/// the **overlapping neighbor stride** `tile[tid] = tile[tid] + tile[tid + 1]`
/// under `tid < 255` instead of the correct disjoint `tile[tid] += tile[tid +
/// half]` under `tid < half`: thread `t` reads `tile[t + 1]` while thread `t +
/// 1` concurrently writes `tile[t + 1]`, and no barrier can separate reads from
/// writes *within* one generation — an intra-phase read-write race (`t1 == t2 +
/// 1` collides). All accesses are bounds-safe (the `tid < 255` guard keeps the
/// neighbor read in range), so the two-thread race walker refutes
/// `smt-race-freedom` on the **race**, not on bounds, printing a two-thread
/// counterexample — the deterministic catch. Because the twin serialises the
/// generation into one arbitrary thread order (and the GPU does not), the GPU
/// differential *usually* diverges too, but that is nondeterministic, so the
/// proof refutation is the reliable finding. Not suite-wired (a defect probe;
/// lives in `conform`'s demo-defects mode).
#[vericl::kernel(
    assumes(input.iter().all(|v| v.abs() <= 100.0)),
    compare(max_ulp = 0),
    gen(input in -100.0..=100.0),
    cooperative(cube_dim = 256)
)]
#[cube(launch)]
pub fn block_sum_reduce_racy(input: &Array<f32>, output: &mut Array<f32>) {
    let tid = UNIT_POS as usize;
    let mut tile = SharedMemory::<f32>::new(256usize);
    if ABSOLUTE_POS < input.len() {
        tile[tid] = input[ABSOLUTE_POS];
    } else {
        tile[tid] = 0.0f32;
    }
    sync_cube();
    // BUG: an intra-level neighbor combine — thread `t` reads `tile[t + 1]`
    // while thread `t + 1` writes it. The correct reduction uses a disjoint
    // `tid + half` stride under `tid < half`, so read and write sets never
    // overlap; this overlapping `tid + 1` stride makes adjacent threads race.
    if tid < 255usize {
        let neighbor = tile[tid + 1usize];
        tile[tid] = tile[tid] + neighbor;
    }
    sync_cube();
    if tid == 0usize && CUBE_POS < output.len() {
        output[CUBE_POS] = tile[0usize];
    }
}

/// Hand-written sequential reference for the declared-reference demonstrator
/// below (candidate #3, docs/design-shared-memory.md §4.4/§6). Deliberately a
/// SEPARATE artifact from the kernel — the whole point of #3 is a reference the
/// phase-split transform did not derive, for a cooperative kernel outside the
/// transformable subset. Signature matches the cooperative twin's:
/// `(inputs..., outputs..., cube_count, cube_dim)`.
///
/// `#[vericl::reference]` records this fn's own `SOURCE_HASH` (over its tokens)
/// in a sibling `block_sum_declared_ref_vericl` module, which the kernel below
/// folds into its `identity()` — so a drift in THIS body moves the kernel's
/// recorded identity (round-3 adversarial review F2). The annotation is
/// required by the `reference = …` clause.
#[vericl::reference]
pub fn block_sum_declared_ref(input: &[f32], output: &mut [f32], cube_count: usize, cube_dim: usize) {
    for (c, slot) in output.iter_mut().enumerate().take(cube_count) {
        let mut tile = vec![0.0f32; cube_dim];
        for (tid, cell) in tile.iter_mut().enumerate() {
            let abs = c * cube_dim + tid;
            *cell = if abs < input.len() { input[abs] } else { 0.0 };
        }
        let mut half = cube_dim / 2;
        while half > 0 {
            for tid in 0..half {
                tile[tid] += tile[tid + half];
            }
            half /= 2;
        }
        *slot = tile[0];
    }
}

/// Declared-reference demonstrator (candidate #3, docs/design-shared-memory.md
/// §4.4/§6): identical reduction shape to `block_sum_reduce`, but its reference
/// is the author-supplied `block_sum_declared_ref` (via `reference = …`) rather
/// than a derived phase-split twin. A *strictly weaker, distinct* claim — the
/// tested claim carries `differential-declared-reference`, not `differential`,
/// because a hand-written reference is a separate artifact that can silently
/// drift from the kernel (the custody the derived twin preserves is given up).
/// Not suite-wired; exercised by `block_sum_reduce_declared_uses_the_declared_
/// reference` below. NB a kernel *inside* the transformable subset (as this one
/// is) opting into the weaker claim is only allowed *explicitly*, via the
/// clause — never silently (§4.4 gate).
#[vericl::kernel(
    assumes(input.iter().all(|v| v.abs() <= 1000.0)),
    compare(max_ulp = 0),
    gen(input in -1000.0..=1000.0),
    cooperative(cube_dim = 256),
    reference = block_sum_declared_ref
)]
#[cube(launch)]
pub fn block_sum_reduce_declared(input: &Array<f32>, output: &mut Array<f32>) {
    let tid = UNIT_POS as usize;
    let mut tile = SharedMemory::<f32>::new(256usize);
    if ABSOLUTE_POS < input.len() {
        tile[tid] = input[ABSOLUTE_POS];
    } else {
        tile[tid] = 0.0f32;
    }
    sync_cube();

    let mut half = CUBE_DIM as usize / 2;
    while half > 0usize {
        if tid < half {
            let a = tile[tid];
            let b = tile[tid + half];
            tile[tid] = a + b;
        }
        sync_cube();
        half /= 2usize;
    }

    if tid == 0usize && CUBE_POS < output.len() {
        output[CUBE_POS] = tile[0usize];
    }
}

// ===========================================================================
// Quick-wins batch 2 (macro-leaning): verified host shims (cast_from / mul_hi),
// helper-level `wrapping`, and comptime! block evaluation.
// ===========================================================================

// --- Feature 1: verified cast_from / mul_hi host shims --------------------
//
// `Cast::cast_from` / `Numeric::mul_hi` are `unexpanded!()` on host, so a twin
// cannot call them directly. `#[vericl::kernel]`/`#[vericl::helper]` rewrite a
// recognized call to a GPU-verified `::vericl::host_shims::` shim, pinned
// bit-exactly against the real intrinsic on wgpu in
// `tests/host_shim_gpu_ground_truth.rs`.

/// A `u32` → `f32` unit-interval `[0, 1)` converter — the reusable heart of a
/// GPU RNG's float output (Lemire's upper-24-bits technique,
/// <https://lemire.me/blog/2017/02/28/how-many-floating-point-numbers-are-in-the-interval-01/>;
/// the same shape cubek-random's `to_unit_interval_closed_open` uses). Its
/// `f32::cast_from(shifted)` is exactly what used to be `FLOAT_METHOD_REJECT`ed;
/// now it rewrites to the verified `cast_to_f32` shim. Both the u32→f32 cast
/// (round-to-nearest-even) and the divide by 2^24 (an exact power-of-two scale)
/// are bit-exact on GPU and host, so the composing kernel compares at 0 ULP.
#[vericl::helper]
#[cube]
pub fn to_unit_interval(int_random: u32) -> f32 {
    let shifted = int_random >> 8u32; // keep the upper 24 bits
    f32::cast_from(shifted) / 16777216.0 // / 2^24
}

/// FLAGSHIP (Feature 1): the u32-RNG-output → unit-interval-f32 kernel that
/// completes the RNG story end-to-end — a `u32` stream in, a `[0, 1)` `f32`
/// stream out, via the verified `cast_from` shim inside a composed helper.
/// Bit-exact (`max_ulp = 0`) against the GPU, bounds `Proved`.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(max_ulp = 0),
    gen(y in 0.0..=0.0),
    uses(to_unit_interval)
)]
#[cube(launch)]
pub fn unit_interval_map(x: &Array<u32>, y: &mut Array<f32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = to_unit_interval(x[ABSOLUTE_POS]);
    }
}

/// Feature 1 (`mul_hi`): the high 32 bits of the full-width `u32` product — the
/// core of a fixed-point / fast-division multiply (cubecl-std's `FastDivmod`
/// uses `mul_hi`). `a.mul_hi(b)` rewrites to the verified `mul_hi` shim
/// (`(a·b) >> 32`); the result is exact `u32`, so `compare(exact)`, bounds
/// `Proved` (the high word is not used as an index).
#[vericl::kernel(
    assumes(a.len() == b.len(), a.len() == y.len()),
    compare(exact)
)]
#[cube(launch)]
pub fn mul_hi_map(a: &Array<u32>, b: &Array<u32>, y: &mut Array<u32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = a[ABSOLUTE_POS].mul_hi(b[ABSOLUTE_POS]);
    }
}

// --- Feature 2: helper-level `wrapping` -----------------------------------
//
// `#[vericl::helper(wrapping)]` applies the same `WrappingFold` a kernel's
// `wrapping` clause applies, to the helper twin's own body (integer-only gate).
// Interaction rule (each item governs ONLY its own body; integers cross the
// boundary as plain integers): a NON-wrapping kernel may freely use a wrapping
// helper — the flagship below.

/// A linear-congruential-generator step, `z*a + b` (Numerical Recipes
/// constants) — wrap-on-overflow by intent, matching WGSL. As a
/// `#[vericl::helper(wrapping)]` its twin's `z*a + b` folds to
/// `z.wrapping_mul(a).wrapping_add(b)`; without `wrapping` the checked twin
/// would panic on overflow (the negative control below).
#[vericl::helper(wrapping)]
#[cube]
pub fn lcg_step(z: u32) -> u32 {
    let a = 1664525u32;
    let b = 1013904223u32;
    z * a + b
}

/// FLAGSHIP (Feature 2): a NON-wrapping kernel composing the WRAPPING `lcg_step`
/// helper. The kernel body has no arithmetic of its own (checked semantics are
/// fine for it); the helper wraps internally, matching the GPU. Exact `u32`
/// compare, bounds `Proved`. This is the interaction rule in action — the
/// helper declares its own wrap semantics and integers cross the call boundary
/// as plain values, so no `wrapping` clause is needed (or allowed) on the
/// kernel.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(exact),
    uses(lcg_step)
)]
#[cube(launch)]
pub fn lcg_map(x: &Array<u32>, y: &mut Array<u32>) {
    if ABSOLUTE_POS < y.len() {
        y[ABSOLUTE_POS] = lcg_step(x[ABSOLUTE_POS]);
    }
}

/// NEGATIVE CONTROL (Feature 2 interaction rule, the other half): the SAME LCG
/// body WITHOUT the `wrapping` clause. Its twin computes CHECKED `z*a + b`, so a
/// `wrapping` kernel that composed it would get a loud overflow panic in the
/// twin instead of the GPU's wrap — the round-3 behavior kept by design (each
/// item governs only its own body; a non-wrapping helper's checked arithmetic
/// panics rather than silently diverging). Not suite-wired; pinned by
/// `nonwrapping_helper_twin_panics_on_overflow` below. The faithful path is the
/// `wrapping` clause (`lcg_step` above).
#[vericl::helper]
#[cube]
pub fn lcg_step_checked(z: u32) -> u32 {
    z * 1664525u32 + 1013904223u32
}

// --- Feature 3: comptime! block evaluation --------------------------------
//
// A `comptime! { EXPR }` block whose expression references only #[comptime]
// parameters (concrete under instantiate) + literals is evaluated at expansion
// by stripping the wrapper to `EXPR` (host Rust the twin re-runs identically);
// a block referencing a runtime value is rejected by name.

/// Feature 3: a right-shift whose amount is derived in a `comptime!` block from
/// the `#[comptime] extra` parameter (`shift = extra + 2`, pinned to 3). The
/// block is evaluated at expansion — `extra` is compile-time-known — so the
/// twin re-emits `(extra + 2)` over its `let extra = 1;` const. Exact `u32`
/// compare, bounds `Proved`.
#[vericl::kernel(
    assumes(x.len() == y.len()),
    compare(exact),
    instantiate(extra = 1)
)]
#[cube(launch)]
pub fn comptime_shift(x: &Array<u32>, y: &mut Array<u32>, #[comptime] extra: u32) {
    if ABSOLUTE_POS < y.len() {
        let shift = comptime!(extra + 2u32);
        y[ABSOLUTE_POS] = x[ABSOLUTE_POS] >> shift;
    }
}

/// Clean-room vectorized elementwise add over `Array<Vector<f32, N>>`
/// (design-line-vector.md §11 V3/V4). The width is pinned to `4` via
/// `instantiate(N = 4)`, so the reference twin is monomorphized to
/// `&[::vericl::Line<f32, 4>]` and the `Vector` element ban is lifted under the
/// vector gate. Reference kernel shape is upstream-public (cubecl-core's own
/// `runtime_tests/vector.rs`; MIT/Apache-2.0). The `gen(...)` range applies per
/// lane: each case draws `lines * 4` flat scalars in it (design §4.4, §9). A
/// vector-`W` add is `W` correctly-rounded scalar adds, so the twin is bit-exact
/// with the GPU — `compare(abs = 1e-6)` is generous headroom, not required.
#[vericl::kernel(
    assumes(a.len() == out.len(), b.len() == out.len()),
    compare(abs = 1e-6),
    gen(a in -100.0..=100.0, b in -100.0..=100.0, out in 0.0..=0.0),
    instantiate(N = 4)
)]
#[cube(launch)]
pub fn vec_add<N: Size>(
    a: &Array<Vector<f32, N>>,
    b: &Array<Vector<f32, N>>,
    out: &mut Array<Vector<f32, N>>,
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

/// Clean-room vectorized scale-by-splat: `out[p] = a[p] * Vector::new(s)`. Its
/// body contains a `Vector` ident (the splat constructor), so it exercises the
/// V3 `Vector`->`::vericl::Line` head rewrite inside the twin body (design §4.3),
/// which `vec_add` above does not. The scalar `s` is drawn once per case; `a` is
/// drawn per-lane. Per-lane `*` is correctly rounded, so the twin is bit-exact.
#[vericl::kernel(
    assumes(a.len() == out.len()),
    compare(abs = 1e-6),
    gen(s in -8.0..=8.0, a in -100.0..=100.0, out in 0.0..=0.0),
    instantiate(N = 4)
)]
#[cube(launch)]
pub fn vec_scale<N: Size>(
    s: f32,
    a: &Array<Vector<f32, N>>,
    out: &mut Array<Vector<f32, N>>,
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] * Vector::new(s);
    }
}

/// Clean-room vectorized `out[p] = a[p] * a[p] + b[p]` — the **vec_madd-class
/// tolerance example** (design §4.5, §6). Unlike `vec_add`/`vec_scale` (whose
/// per-lane `+`/`*` are individually correctly rounded and therefore bit-exact),
/// `a*a + b` is fusable: a backend that contracts it to a single fused
/// multiply-add rounds once where the twin (two ops) rounds twice, a ≤1-ULP
/// per-lane gap — the identical, well-understood float-contraction divergence a
/// *scalar* `a*a+b` kernel has, handled by the declared `compare(abs = …)`. The
/// abs bound is justified from the `gen` ranges: with `|a|,|b| ≤ 8`, `|a*a+b| ≤
/// 72`, and one f32 rounding of a value ≤ 72 is ≤ ulp(72)/2 ≈ 4e-6, so `abs =
/// 1e-5` is generous headroom. The vector model itself adds **zero** divergence
/// (design §4.5): this tolerance is a per-lane float fact, not a lane-coupling
/// artifact.
#[vericl::kernel(
    assumes(a.len() == out.len(), b.len() == out.len()),
    compare(abs = 1e-5),
    gen(a in -8.0..=8.0, b in -8.0..=8.0, out in 0.0..=0.0),
    instantiate(N = 4)
)]
#[cube(launch)]
pub fn vec_madd<N: Size>(
    a: &Array<Vector<f32, N>>,
    b: &Array<Vector<f32, N>>,
    out: &mut Array<Vector<f32, N>>,
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] * a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

/// DEFECTIVE (tolerance too tight): the same `a*a + b` as `vec_madd`, but with a
/// bit-exact `compare(max_ulp = 0)`. On a backend that fuses `a*a + b` to one FMA
/// rounding (Metal does — design §6), the two-rounding twin diverges by up to a
/// representable step per lane, so a bit-exact tolerance FAILS — and the
/// differential check reports the divergence naming the specific `(line, lane)`
/// (design §9). It is the negative control for per-lane divergence reporting:
/// the fix is the honest `compare(abs = …)` on `vec_madd`, justified from the
/// input ranges. Kept OUT of the conformance suite.
#[vericl::kernel(
    assumes(a.len() == out.len(), b.len() == out.len()),
    compare(max_ulp = 0),
    gen(a in -8.0..=8.0, b in -8.0..=8.0, out in 0.0..=0.0),
    instantiate(N = 4)
)]
#[cube(launch)]
pub fn vec_madd_bitexact<N: Size>(
    a: &Array<Vector<f32, N>>,
    b: &Array<Vector<f32, N>>,
    out: &mut Array<Vector<f32, N>>,
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] * a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

/// DEFECTIVE (bounds): the boundary guard is `<=`, reading and writing one
/// **line** past the end. Exactly the `axpy_off_by_one` defect (design §11 V4
/// negative control) at a `Vector<f32, 4>` element type: WGSL robustness clamps
/// the over-read on the GPU, but the `Line`-array reference twin indexes
/// `out[out.len()]` and panics deterministically — the differential check catches
/// it. Kept OUT of the conformance suite (it belongs to a negative-control test),
/// like its scalar sibling.
#[vericl::kernel(
    assumes(a.len() == out.len(), b.len() == out.len()),
    compare(abs = 1e-6),
    gen(a in -100.0..=100.0, b in -100.0..=100.0, out in 0.0..=0.0),
    instantiate(N = 4)
)]
#[cube(launch)]
pub fn vec_add_off_by_one<N: Size>(
    a: &Array<Vector<f32, N>>,
    b: &Array<Vector<f32, N>>,
    out: &mut Array<Vector<f32, N>>,
) {
    if ABSOLUTE_POS <= out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vericl::Line;

    /// Validates the macro-derived twin against independently written scalar
    /// code — guarding the source-to-reference derivation itself.
    #[test]
    fn xorshift_twin_matches_handwritten() {
        let x: Vec<u32> = vec![1, 0xDEADBEEF, u32::MAX, 42, 0];
        let mut y = vec![0u32; x.len()];
        xorshift_step_vericl::reference(&x, &mut y, 8);
        for (i, &v) in x.iter().enumerate() {
            let mut s = v;
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            assert_eq!(y[i], s, "index {i}");
        }
    }

    /// V3 acceptance (design §11): the macro-derived twin for the vectorized
    /// `vec_add` — mapped to `&[Line<f32, 4>]` via `instantiate(N = 4)` and the
    /// `Vector`->`Line` head rewrite — matches an independently hand-written
    /// `Line` twin lane-for-lane, and honours the `ABSOLUTE_POS < out.len()`
    /// guard. This is the `*_twin_matches_handwritten` precedent extended to a
    /// vector element type.
    #[test]
    fn vec_add_twin_matches_handwritten_line() {
        let a = vec![
            Line([1.0f32, 2.0, 3.0, 4.0]),
            Line([-1.0f32, 0.5, 100.0, -7.5]),
            Line([0.0f32, 0.0, 0.0, 0.0]),
        ];
        let b = vec![
            Line([10.0f32, 20.0, 30.0, 40.0]),
            Line([1.0f32, 1.0, 1.0, 1.0]),
            Line([9.0f32, 8.0, 7.0, 6.0]),
        ];
        let mut out = vec![Line([-1.0f32; 4]); a.len()];
        // Dispatch far more threads than lines: the guard must leave nothing
        // out of place and the twin must not run past `out.len()`.
        vec_add_vericl::reference(&a, &b, &mut out, 256);

        // Independent hand-written lane-array reference.
        let want: Vec<Line<f32, 4>> = a
            .iter()
            .zip(&b)
            .map(|(x, y)| Line(core::array::from_fn(|j| x.0[j] + y.0[j])))
            .collect();
        assert_eq!(out, want);
        // And bit-exact per lane (elementwise add is correctly rounded).
        for (o, w) in out.iter().zip(&want) {
            for j in 0..4 {
                assert_eq!(o.0[j].to_bits(), w.0[j].to_bits(), "lane {j}");
            }
        }
    }

    /// V3 body-rewrite acceptance: `vec_scale`'s body uses `Vector::new(s)`,
    /// which the macro rewrites to `::vericl::Line::new(s)` in the twin. The
    /// derived twin computes `a[p] * splat(s)` lane-for-lane, matching a
    /// hand-written `Line` reference — confirming the `Vector` head rewrite
    /// inside the body (not just the signature) is correct.
    #[test]
    fn vec_scale_twin_rewrites_splat_and_matches_handwritten() {
        let s = 2.5f32;
        let a = vec![
            Line([1.0f32, 2.0, 3.0, 4.0]),
            Line([-2.0f32, 0.0, 8.0, -1.5]),
        ];
        let mut out = vec![Line([0.0f32; 4]); a.len()];
        vec_scale_vericl::reference(s, &a, &mut out, 64);

        let want: Vec<Line<f32, 4>> =
            a.iter().map(|x| Line(core::array::from_fn(|j| x.0[j] * s))).collect();
        assert_eq!(out, want);
    }

    /// The vectorized `vec_add` twin honours the guard: with fewer threads than
    /// lines, lines past the dispatch keep their initial contents.
    #[test]
    fn vec_add_twin_respects_guard() {
        let a = vec![Line([1.0f32; 4]); 4];
        let b = vec![Line([2.0f32; 4]); 4];
        let mut out = vec![Line([-9.0f32; 4]); 4];
        vec_add_vericl::reference(&a, &b, &mut out, 2); // only 2 of 4 lines
        assert_eq!(out[0], Line([3.0f32; 4]));
        assert_eq!(out[1], Line([3.0f32; 4]));
        assert_eq!(out[2], Line([-9.0f32; 4])); // untouched
        assert_eq!(out[3], Line([-9.0f32; 4])); // untouched
    }

    /// The vectorized `vec_add` kernel's IR proves bounds-free exactly like the
    /// scalar axpy — 3 obligations (a read, b read, out write), line-granular,
    /// with `N` never entering the obligation (design §5.1). Confirms
    /// `kernel_definition()` builds valid vector IR under `instantiate(N = 4)`.
    #[test]
    fn vec_add_kernel_definition_is_provably_in_bounds() {
        let def = vec_add_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = vec_add_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [
            vericl_ir::Assume::LenEq { a: "a", b: "out" },
            vericl_ir::Assume::LenEq { a: "b", b: "out" },
        ];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 3),
            other => panic!("expected Proved {{ obligations: 3 }}, got {other:?}"),
        }
    }

    /// The contract records the pinned width in `instantiate` — a re-run at a
    /// different width is a visibly different claim.
    #[test]
    fn vec_add_contract_records_pinned_width() {
        assert_eq!(vec_add_vericl::contract().instantiate, &["N = 4"]);
    }

    /// The twin honors the guard: threads past the guard write nothing.
    #[test]
    fn axpy_twin_respects_guard() {
        let x = vec![1.0f32; 3];
        let mut y = vec![10.0f32; 3];
        axpy_vericl::reference(2.0, &x, &mut y, 256); // threads >> len
        assert_eq!(y, vec![12.0; 3]);
    }

    /// `axpy_f64`'s twin is monomorphized to `f64` (`&[f64]`, `alpha: f64`)
    /// by `instantiate(F = f64)` and computes at full f64 precision — a value
    /// finer than f32 can represent round-trips through the twin exactly,
    /// proving the twin is genuinely f64 and not silently f32.
    #[test]
    fn axpy_f64_twin_is_full_precision() {
        let x = vec![1.0f64; 3];
        let mut y = vec![10.0f64; 3];
        axpy_f64_vericl::reference(2.0, &x, &mut y, 256); // threads >> len
        assert_eq!(y, vec![12.0f64; 3]);

        // A value that is NOT representable in f32: the twin must preserve it.
        let a = 1.0f64 + 2f64.powi(-40); // distinct from its own f32 round-trip
        assert_ne!(a, (a as f32) as f64);
        let x2 = vec![1.0f64];
        let mut y2 = vec![0.0f64];
        axpy_f64_vericl::reference(a, &x2, &mut y2, 1);
        assert_eq!(y2[0], a); // a*1 + 0, exact in f64
    }

    /// The compare mode is recorded at f64 precision (`AbsRelF64`, described
    /// as `f64 ...`), not silently narrowed to the f32 variant — the whole
    /// point of the macro's `compare_tokens_f64` path.
    #[test]
    fn axpy_f64_compare_is_recorded_as_f64() {
        match axpy_f64_vericl::contract().compare {
            vericl::Compare::AbsRelF64 { abs, rel } => {
                assert_eq!(abs, 1e-12);
                assert_eq!(rel, 0.0);
            }
            other => panic!("expected AbsRelF64, got {other:?}"),
        }
        assert!(axpy_f64_vericl::contract().compare.describe().starts_with("f64 "));
        // The f32 flagship stays f32 — no cross-contamination.
        assert!(matches!(axpy_vericl::contract().compare, vericl::Compare::AbsRelF32 { .. }));
        assert_eq!(axpy_f64_vericl::contract().instantiate, &["F = f64"]);
    }

    /// The SMT bounds prover discharges `axpy_f64` exactly like the f32
    /// flagship (3 obligations) — bounds freedom is about buffer `Length`, so
    /// the f64 element type is irrelevant to the proof; f64 support did not
    /// weaken it.
    #[test]
    fn axpy_f64_kernel_definition_is_provably_in_bounds() {
        let def = axpy_f64_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = axpy_f64_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 3),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    #[test]
    fn assumes_predicate_is_executable() {
        assert!(axpy_vericl::check_assumes(1.0, &[0.0; 4], &[0.0; 4]));
        assert!(!axpy_vericl::check_assumes(1.0, &[0.0; 4], &[0.0; 3])); // len mismatch
        assert!(!axpy_vericl::check_assumes(9.0, &[0.0; 4], &[0.0; 4])); // alpha out of range
        assert!(!axpy_vericl::check_assumes(1.0, &[500.0; 4], &[0.0; 4])); // element out of range
    }

    #[test]
    fn identity_hashes_are_distinct_per_kernel() {
        assert_ne!(axpy_vericl::SOURCE_HASH, axpy_off_by_one_vericl::SOURCE_HASH);
        assert_ne!(axpy_vericl::SOURCE_HASH, xorshift_step_vericl::SOURCE_HASH);
    }

    /// Independently written murmur3 fmix32, used only to cross-check the
    /// macro-derived `wrapping` twin below — kept deliberately separate from
    /// the kernel body so the check is not circular.
    fn fmix32(mut h: u32) -> u32 {
        h ^= h >> 16;
        h = h.wrapping_mul(0x85ebca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2ae35);
        h ^= h >> 16;
        h
    }

    /// The `wrapping` clause matters: the twin's `*`/`>>` are folded to
    /// `wrapping_mul`/`wrapping_shr`, so it must NOT panic on inputs that
    /// overflow `u32` multiplication — even though this test runs under
    /// `cargo test`'s default dev profile, which has `overflow-checks =
    /// true`. Without the fold (or with a checked `*`), this would panic on
    /// `u32::MAX` and friends; a reference panic here would be a semantics
    /// mismatch, not a kernel bug (same class as `axpy_off_by_one`, but
    /// caused by the wrong arithmetic model instead of a bounds bug).
    #[test]
    fn mix_u32_wraps_without_panicking() {
        let x: Vec<u32> = vec![u32::MAX, 0, 1, 0x9E37_79B9, 0xFFFF_FFFF, u32::MAX / 2 + 1];
        let mut y = vec![0u32; x.len()];
        mix_u32_vericl::reference(&x, &mut y, x.len()); // must not panic
    }

    /// Cross-checks the macro-derived twin against the handwritten
    /// `fmix32` above, including the murmur3 `fmix32(0) == 0` vector.
    #[test]
    fn mix_u32_twin_matches_handwritten_fmix32() {
        assert_eq!(fmix32(0), 0);

        let x: Vec<u32> = vec![0, 1, 42, 0xDEAD_BEEF, u32::MAX, 0x9E37_79B9];
        let mut y = vec![0u32; x.len()];
        mix_u32_vericl::reference(&x, &mut y, x.len());
        for (i, &v) in x.iter().enumerate() {
            assert_eq!(y[i], fmix32(v), "index {i}");
        }
    }

    /// The `wrapping` clause is part of the recorded contract.
    #[test]
    fn wrapping_is_recorded_in_the_contract() {
        assert!(mix_u32_vericl::contract().wrapping);
        assert!(!xorshift_step_vericl::contract().wrapping);
    }

    /// `assumes(x.len() == y.len())` is recognized as a structured
    /// `LenEq` assume, and `BUFFER_PARAMS` records buffer-registration order
    /// — both are what the SMT bounds prover needs and has no other way to
    /// recover from the IR alone.
    #[test]
    fn structured_assumes_and_buffer_params_are_generated() {
        assert_eq!(
            axpy_vericl::contract().structured_assumes,
            &[vericl::StructuredAssume::LenEq { a: "x", b: "y" }]
        );
        assert_eq!(axpy_vericl::BUFFER_PARAMS, &[("x", false), ("y", true)]);

        assert_eq!(
            sum_racy_vericl::contract().structured_assumes,
            &[vericl::StructuredAssume::LenEqConst { a: "y", value: 1 }]
        );
    }

    /// `kernel_definition()` builds a real `KernelDefinition` (no
    /// client/device/runtime) that the SMT bounds prover can discharge —
    /// the first machine-checked property (README "First release" outcome
    /// 3), exercised end-to-end here for the guarded, in-bounds case.
    #[test]
    fn kernel_definition_is_provably_in_bounds() {
        let def = axpy_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = axpy_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 3),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// `axpy_off_by_one`'s IR-level bounds obligation is genuinely violated
    /// (not just its differential/reference-panic check) — the SMT prover
    /// refutes it independently of the differential harness.
    #[test]
    fn off_by_one_kernel_definition_refutes() {
        let def = axpy_off_by_one_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = axpy_off_by_one_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Refuted { .. } => {}
            other => panic!("expected Refuted, got {other:?}"),
        }
    }

    /// `xorshift_step` and `mix_u32` both prove in spite of their bitwise/
    /// wrapping-integer bodies: every index used is a bare `ABSOLUTE_POS`,
    /// so the value-computation ops (outside the prover's modeled subset)
    /// never end up needed for an obligation (see vericl-ir's module docs).
    #[test]
    fn xorshift_and_mix_u32_prove_despite_bitwise_bodies() {
        for (def, buffer_params) in [
            (xorshift_step_vericl::kernel_definition(), xorshift_step_vericl::BUFFER_PARAMS),
            (mix_u32_vericl::kernel_definition(), mix_u32_vericl::BUFFER_PARAMS),
        ] {
            let buffers: Vec<vericl_ir::BufferParam> = buffer_params
                .iter()
                .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
                .collect();
            let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
            match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
                vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
                other => panic!("expected Proved, got {other:?}"),
            }
        }
    }

    /// `sum_racy`'s `y[0]` access proves given `assumes(y.len() == 1)` —
    /// the race is a differential finding, not a bounds one; the two claim
    /// kinds are cleanly separate (see README "Claims and trust
    /// boundaries").
    #[test]
    fn sum_racy_bounds_prove_independently_of_its_race() {
        let def = sum_racy_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = sum_racy_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEqConst { a: "y", value: 1 }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            // x[pos] read, y[0] read, y[0] write.
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 3),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// REGRESSION (adversarial soundness review, Bug 1): a `range_stepped`
    /// loop — here a runtime-bounded descending copy that is, by
    /// construction, entirely in-bounds — must still be rejected as
    /// `OutOfSubset`, never `Proved`. Before the fix, `process_range_loop`
    /// never read `rl.step` and unconditionally asserted ascending bounds
    /// (`start <= i < end`); for this real descending loop `start > end`
    /// numerically, so those assertions are unsatisfiable, the SMT context
    /// becomes infeasible, and every obligation inside would discharge
    /// vacuously as "proved" regardless of the body. The fix rejects any
    /// `rl.step.is_some()` outright.
    #[test]
    fn stepped_range_loop_is_out_of_subset() {
        let def = stepped_loop_descending_copy_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = stepped_loop_descending_copy_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("stepped") || reason.contains("range_stepped"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset, got {other:?}"),
        }
    }

    /// REGRESSION (adversarial soundness review, Bug 1): the exact
    /// vacuous-proof shape the review demonstrated — a runtime-bounded,
    /// negative-step loop whose body writes/reads `[100000]`, far outside
    /// any plausible buffer length. Before the fix this returned
    /// `Proved { obligations: 2 }` (the SMT context was infeasible, so both
    /// the `x[100000]` read and `y[100000]` write discharged vacuously)
    /// even though a real sequential execution of this loop panics
    /// out-of-bounds — a false soundness claim. Must now return
    /// `OutOfSubset`.
    #[test]
    fn stepped_loop_cannot_vacuously_prove() {
        let def = stepped_loop_oob_write_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = stepped_loop_oob_write_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("stepped") || reason.contains("range_stepped"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (not Proved — see doc comment), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // instantiate(...): generic + #[comptime] monomorphization (fir3).
    // -----------------------------------------------------------------

    /// Independently written 3-tap (at most) windowed sum, used only to
    /// cross-check the macro-derived, instantiate(...)-monomorphized twin —
    /// kept deliberately separate from the kernel body so the check is not
    /// circular. Mirrors `fmix32` above for `mix_u32`.
    fn fir_handwritten(x: &[f32], taps: u32) -> Vec<f32> {
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                let mut acc = v;
                if taps > 1 && i >= 1 {
                    acc += x[i - 1];
                }
                if taps > 2 && i >= 2 {
                    acc += x[i - 2];
                }
                acc
            })
            .collect()
    }

    /// `fir3`'s twin (F -> f32, `#[comptime] taps` removed from the
    /// signature and bound as a `let taps: u32 = 3;` const per
    /// `instantiate(F = f32, taps = 3)`) matches the independent
    /// hand-written computation above — the same guard against a circular
    /// derivation check as `xorshift_twin_matches_handwritten`/
    /// `mix_u32_twin_matches_handwritten_fmix32`.
    #[test]
    fn fir3_twin_matches_handwritten() {
        let x: Vec<f32> = vec![1.0, -2.0, 3.5, 0.25, -7.0, 10.0, 0.0, -1.5];
        let mut y = vec![0.0f32; x.len()];
        // reference() takes no `taps` parameter — it's a const now.
        fir3_vericl::reference(&x, &mut y, x.len());
        let expected = fir_handwritten(&x, 3);
        for (i, (&got, &want)) in y.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "index {i}: got {got}, want {want}");
        }
    }

    /// The twin honors the guard exactly like `axpy`'s: threads past the
    /// guard write nothing to `y`.
    #[test]
    fn fir3_twin_respects_guard() {
        let x = vec![1.0f32; 3];
        let mut y = vec![9.0f32; 3];
        fir3_vericl::reference(&x, &mut y, 256); // threads >> len
        assert_eq!(y, vec![1.0, 2.0, 3.0]); // taps=3: x[0], x[1]+x[0], x[2]+x[1]+x[0]
    }

    /// `instantiate(...)`'s pinned values are part of the recorded contract
    /// (`Contract::instantiate`/`ContractRecord::instantiate`), in clause
    /// declaration order — separate from `assumes`/`wrapping`, mirroring how
    /// `wrapping_is_recorded_in_the_contract` pins that clause's field.
    #[test]
    fn instantiate_is_recorded_in_the_contract() {
        assert_eq!(fir3_vericl::contract().instantiate, &["F = f32", "taps = 3"]);
        assert_eq!(fir3_alt_vericl::contract().instantiate, &["F = f32", "taps = 1"]);
        // A non-generic, non-comptime kernel has an empty instantiate list.
        assert!(axpy_off_by_one_vericl::contract().instantiate.is_empty());
    }

    /// Changing an `instantiate(...)` value changes kernel identity: `fir3`
    /// and `fir3_alt` are byte-identical source except for the pinned
    /// `taps` value, and the instantiation value is part of the raw
    /// contract attribute tokens `SOURCE_HASH` covers — so the two hashes
    /// must differ. This is the source-level counterpart to
    /// `identity_hashes_are_distinct_per_kernel` above, specifically for
    /// the instantiate(...) clause.
    #[test]
    fn fir3_identity_changes_with_instantiate_value() {
        assert_ne!(fir3_vericl::SOURCE_HASH, fir3_alt_vericl::SOURCE_HASH);
    }

    /// The milestone's headline result: `fir3` is genuinely generic
    /// (`F: Float`) *and* has a `#[comptime]` parameter, monomorphized via
    /// `instantiate(F = f32, taps = 3)`, *and* uses `&&`-composed guards
    /// (`taps > 1 && ABSOLUTE_POS >= 1`) — and its bounds obligations still
    /// discharge as `Proved`, not merely `OutOfSubset`. Achieved by (a)
    /// avoiding a loop-carried accumulator (see the kernel's doc comment):
    /// each extra tap is its own guarded condition, the same shape the
    /// prover already handles for `axpy`/`sum_racy`; and (b) boolean
    /// condition composition (`vericl-ir`'s `Operator::And` modeling).
    #[test]
    fn fir3_kernel_definition_is_provably_in_bounds() {
        let def = fir3_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = fir3_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            // x[pos] read, y[pos] write, guarded x[pos-1]/x[pos-2] reads.
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 4),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // flatten_decode_scale: the div/mod prover-expansion milestone kernel.
    // -----------------------------------------------------------------

    /// The twin honors the guard: threads past `y.len()` write nothing.
    #[test]
    fn flatten_decode_scale_twin_respects_guard() {
        let x = vec![1.0f32; 4];
        let mut y = vec![9.0f32; 4];
        flatten_decode_scale_vericl::reference(&x, &mut y, 2, 3.0, 256); // threads >> len
        // pos 0: row=0,col=0 -> idx 0; pos 1: row=0,col=1 -> idx 1;
        // pos 2: row=1,col=0 -> idx 2; pos 3: row=1,col=1 -> idx 3 (idx ==
        // pos throughout, by construction of the row/col decode/recombine).
        assert_eq!(y, vec![3.0, 3.0, 3.0, 3.0]);
    }

    /// Independently computes the same row/col decode + scale via plain
    /// integer division/remainder (not derived from the kernel body), to
    /// cross-check the macro-derived twin — same guard against a circular
    /// derivation check as `fir_handwritten`/`fmix32` above.
    #[test]
    fn flatten_decode_scale_twin_matches_handwritten() {
        let x: Vec<f32> = (0..12).map(|i| i as f32 - 5.0).collect();
        let mut y = vec![0.0f32; x.len()];
        let width = 4usize;
        let scale = 2.5f32;
        flatten_decode_scale_vericl::reference(&x, &mut y, width as u32, scale, x.len());
        for (pos, (&xv, &yv)) in x.iter().zip(y.iter()).enumerate() {
            let row = pos / width;
            let col = pos % width;
            let idx = row * width + col;
            assert_eq!(idx, pos, "row/col recombine should be the identity at pos {pos}");
            assert_eq!(yv, xv * scale, "index {pos}");
        }
    }

    /// `assumes(x.len() == y.len())` is still the only structured assume —
    /// `width`'s nonzero-ness is established by the kernel's own `width >=
    /// 1u32` runtime guard (a path condition), not a declared assume; see
    /// the kernel's doc comment.
    #[test]
    fn flatten_decode_scale_structured_assumes() {
        assert_eq!(
            flatten_decode_scale_vericl::contract().structured_assumes,
            &[vericl::StructuredAssume::LenEq { a: "x", b: "y" }]
        );
    }

    /// The div/mod milestone's headline result: the write index is the
    /// *recombined* `row * width + col`, not a bare `ABSOLUTE_POS` — proving
    /// it in bounds requires the SMT solver to derive `row * width + col ==
    /// ABSOLUTE_POS` from the `div`/`mod` (Euclidean) axioms and combine
    /// that with the `ABSOLUTE_POS < y.len()` guard. Before div/mod
    /// modeling, `Arithmetic::Div`/`Modulo` tainted their output, `row`/
    /// `col` were unbound, and this was `OutOfSubset` at the write index.
    #[test]
    fn flatten_decode_scale_kernel_definition_is_provably_in_bounds() {
        let def = flatten_decode_scale_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = flatten_decode_scale_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            // x[pos] read, y[row*width+col] write.
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Array-value-dependent indices (offset tables / gather).
    // -----------------------------------------------------------------

    /// Both element-range assume shapes are recognized: the outer-read length
    /// tie AND the `offsets[·] < x.len()` element bound (the new
    /// `ElemsBelowLen`).
    #[test]
    fn gather_copy_structured_assumes() {
        assert_eq!(
            gather_copy_vericl::contract().structured_assumes,
            &[
                vericl::StructuredAssume::LenEq { a: "offsets", b: "y" },
                vericl::StructuredAssume::ElemsBelowLen { arr: "offsets", len_of: "x" },
            ]
        );
    }

    /// The derived sequential twin performs the gather its body says — guards
    /// the twin derivation for the value-dependent-index shape.
    #[test]
    fn gather_copy_twin_matches_hand_computed() {
        let x = vec![10.0f32, 20.0, 30.0, 40.0];
        let offsets = vec![3u32, 1, 0, 2];
        let mut y = vec![0.0f32; 4];
        gather_copy_vericl::reference(&x, &offsets, &mut y, 4);
        assert_eq!(y, vec![40.0, 20.0, 10.0, 30.0]);
    }

    /// The milestone's headline: `y[i] = x[offsets[i]]` proves in bounds only
    /// because `offsets[i]`'s value is modeled `< x.len()` by the element-range
    /// assume — the value-dependent index the checker never used to reach.
    #[test]
    fn gather_copy_kernel_definition_is_provably_in_bounds() {
        let def = gather_copy_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = gather_copy_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [
            vericl_ir::Assume::LenEq { a: "offsets", b: "y" },
            vericl_ir::Assume::ElemsBelowLen { arr: "offsets", len_of: "x" },
        ];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            // offsets[pos] read, x[elem] read, y[pos] write.
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 3),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// Element assumes compose: `data[inner[outer[i]]]` proves with an assume
    /// on each index layer, no special casing.
    #[test]
    fn nested_gather_kernel_definition_is_provably_in_bounds() {
        let def = nested_gather_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = nested_gather_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [
            vericl_ir::Assume::LenEq { a: "outer", b: "y" },
            vericl_ir::Assume::ElemsBelowLen { arr: "outer", len_of: "inner" },
            vericl_ir::Assume::ElemsBelowLen { arr: "inner", len_of: "data" },
        ];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            // outer[pos], inner[elem], data[elem] reads + y[pos] write.
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 4),
            other => panic!("expected Proved (nested gather composes), got {other:?}"),
        }
    }

    /// Round-4 adversarial-review backstop for the recognizer soundness fix.
    /// The truncating-cast repro `offsets.iter().all(|v| (*v as u8 as usize) <
    /// x.len())` is now left string-only (see the macro-crate regression
    /// `elem_assume_truncating_cast_chain_is_string_only`), so the prover
    /// receives NO `ElemsBelowLen` for `offsets`. This pins the consequence that
    /// makes that sound: WITHOUT the element-range assume, the gather is NOT
    /// provable — the loaded offset stays opaque and `x[offsets[i]]` is out of
    /// subset, never `Proved`. So a truncating clause can never be laundered
    /// into a false bounds certificate: the only thing that discharges the inner
    /// obligation is the structured assume the fix now withholds.
    #[test]
    fn gather_copy_is_not_provable_without_element_assume() {
        let def = gather_copy_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = gather_copy_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        // Only the length assume — the element-range assume is withheld, exactly
        // as it is for a string-only (unrecognized) element clause. Any non-Proved
        // outcome (OutOfSubset with the opaque offset, or Refuted) is an honest
        // pass; only a Proved would mean a false bounds certificate was minted.
        let assumes = [vericl_ir::Assume::LenEq { a: "offsets", b: "y" }];
        if let vericl_ir::ProveResult::Proved { .. } =
            vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes)
        {
            panic!("gather must NOT prove without the element-range assume");
        }
    }

    /// The defect twin: a stale constant bound (`< 16`) looser than `x.len() ==
    /// 8` — the prover models the offset `< 16` and refutes `x[offsets[i]]`,
    /// with the element symbol at the boundary in the counterexample.
    #[test]
    fn gather_oob_kernel_definition_refutes_with_element_symbol() {
        let def = gather_oob_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = gather_oob_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [
            vericl_ir::Assume::LenEq { a: "offsets", b: "y" },
            vericl_ir::Assume::LenEqConst { a: "x", value: 8 },
            vericl_ir::Assume::ElemsBelowConst { arr: "offsets", bound: 16 },
        ];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Refuted { obligation, counterexample } => {
                assert!(obligation.contains('x'), "unexpected obligation: {obligation}");
                assert!(
                    counterexample.contains("elem"),
                    "counterexample should exhibit the element symbol: {counterexample}"
                );
            }
            other => panic!("expected Refuted (offset overruns x), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // match / Switch (quick-wins batch 1).
    // -----------------------------------------------------------------

    /// The derived twin re-emits the `match` and selects the same arm the GPU
    /// does — guards the twin derivation for the Switch shape (mode 0 =
    /// identity, mode 1 = negate, default = double).
    #[test]
    fn select_mode_twin_matches_hand_computed() {
        let x = vec![1.0f32, -2.0, 3.0, -4.0];
        let expect = |mode: u32, f: fn(f32) -> f32| {
            let mut y = vec![0.0f32; 4];
            select_mode_vericl::reference(mode, &x, &mut y, 4);
            let want: Vec<f32> = x.iter().copied().map(f).collect();
            assert_eq!(y, want, "mode {mode}");
        };
        expect(0, |v| v);
        expect(1, |v| -v);
        expect(2, |v| v * 2.0); // default arm
        expect(7, |v| v * 2.0); // any un-listed value → default
    }

    /// A guarded `match` proves in bounds: each of the three arms (case 0,
    /// case 1, default) is bounds-checked under its own path condition, 3 arms
    /// × {`x` read, `y` write} = 6 obligations.
    #[test]
    fn select_mode_kernel_definition_is_provably_in_bounds() {
        let def = select_mode_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = select_mode_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 6),
            other => panic!("expected Proved (guarded match), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Length-relationship assume (quick-wins batch 1).
    // -----------------------------------------------------------------

    /// The length-relationship clause is recognized into the structured shape.
    #[test]
    fn offset_window_structured_assumes() {
        assert_eq!(
            offset_window_vericl::contract().structured_assumes,
            &[vericl::StructuredAssume::LenPlusConstLe { a: "y", k: 4, b: "x" }]
        );
    }

    /// The derived twin performs the offset-window sum its body says.
    #[test]
    fn offset_window_twin_matches_hand_computed() {
        // x is longer than y by the window (4); y indexes 0..4.
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let mut y = vec![0.0f32; 4];
        offset_window_vericl::reference(&x, &mut y, 4);
        // y[i] = x[i] + x[i + 4]
        assert_eq!(y, vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// The milestone's headline: `x[i + 4]` proves in bounds only because the
    /// length relationship `y.len() + 4 <= x.len()` is declared (3 obligations:
    /// `x[i]`, `x[i + 4]`, `y[i]`).
    #[test]
    fn offset_window_kernel_definition_is_provably_in_bounds() {
        let def = offset_window_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = offset_window_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenPlusConstLe { a: "y", k: 4, b: "x" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 3),
            other => panic!("expected Proved (offset window), got {other:?}"),
        }
    }

    /// Backstop: WITHOUT the length relationship, the forward read is NOT
    /// provable — a string-only (unrecognized) length clause cannot be
    /// laundered into a bounds certificate. Only `x.len() == y.len()` is given,
    /// which proves `x[i]` but leaves `x[i + 4]` unbounded.
    #[test]
    fn offset_window_is_not_provable_without_relationship() {
        let def = offset_window_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = offset_window_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        if let vericl_ir::ProveResult::Proved { .. } =
            vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes)
        {
            panic!("offset window must NOT prove without the length relationship");
        }
    }

    // -----------------------------------------------------------------
    // Core `Slice` (docs/design-view-slice.md).
    // -----------------------------------------------------------------

    /// The derived slice twin performs the windowed sum its body says —
    /// `y[i] = Σ x.slice(i, i+4)` — via the Rust-subslice `&x[i..i+4]` mapping
    /// and `for &v in slice` by-value iteration (§4.1). Pins the slice twin
    /// derivation at the value level (the `*_twin_matches_hand_computed`
    /// precedent).
    #[test]
    fn windowed_slice_sum_twin_matches_hand_computed() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // len 8 >= 4 + 4
        let mut y = vec![0.0f32; 4];
        windowed_slice_sum_vericl::reference(&x, &mut y, 4);
        // y[i] = x[i] + x[i+1] + x[i+2] + x[i+3]
        assert_eq!(y, vec![10.0, 14.0, 18.0, 22.0]);
    }

    /// Round-9 risk-1 mandatory negative control (docs/design-view-slice.md
    /// §4.4, §12.1): the twin is the slice-**creation** validity oracle cubecl
    /// lacks. If a caller violates the `y.len() + 4 <= x.len()` contract (here
    /// `x.len() == y.len()`), the twin's `&x[pos..pos+4]` slice creation is
    /// out of range for the largest `pos` and **PANICS** — even though each
    /// individual origin *access* the `proved` claim covers is in bounds. The
    /// two claims' scopes are deliberately split: `proved` is over the
    /// accesses, `tested` independently catches invalid creation. (The prover
    /// side of this same risk — proving guarded accesses while the created
    /// slice may exceed the origin — is `slice_dyn_offset_proves` in
    /// crates/vericl-ir.)
    #[test]
    fn windowed_slice_creation_panics_when_x_undersized() {
        let x = vec![1.0f32; 8]; // == y.len(): violates y.len() + 4 <= x.len()
        let mut y = vec![0.0f32; 8];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            windowed_slice_sum_vericl::reference(&x, &mut y, 8);
        }));
        assert!(
            r.is_err(),
            "the twin MUST panic on the out-of-range slice creation — Rust's \
             slice-validity is the soundness net cubecl does not provide (§4.4)"
        );
    }

    /// The windowed-slice bounds prove (2 obligations: the `RangeLoop` origin
    /// read `x[i+j]` and the `y[i]` write) only because the `y.len() + 4 <=
    /// x.len()` relationship is declared — the slice access is the ordinary
    /// origin obligation, discharged UNMODIFIED (deliverable B, §5).
    #[test]
    fn windowed_slice_sum_definition_is_provably_in_bounds() {
        let def = windowed_slice_sum_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = windowed_slice_sum_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenPlusConstLe { a: "y", k: 4, b: "x" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved (windowed slice), got {other:?}"),
        }
    }

    /// Backstop: without the length relationship, the window read is unbounded
    /// — a slice does not launder a bound the origin cannot service (§5.3).
    #[test]
    fn windowed_slice_sum_not_provable_without_relationship() {
        let def = windowed_slice_sum_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = windowed_slice_sum_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        if let vericl_ir::ProveResult::Proved { .. } =
            vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes)
        {
            panic!("windowed slice must NOT prove without the length relationship");
        }
    }

    /// The gather-through-slice twin reads `y[i] = x[offsets.to_slice()[i]]`
    /// via the `&offsets[..]` whole-slice mapping.
    #[test]
    fn slice_gather_copy_twin_matches_hand_computed() {
        let x = vec![10.0f32, 20.0, 30.0, 40.0];
        let offsets = vec![3u32, 1, 0, 2];
        let mut y = vec![0.0f32; 4];
        slice_gather_copy_vericl::reference(&x, &offsets, &mut y, 4);
        assert_eq!(y, vec![40.0, 20.0, 10.0, 30.0]);
    }

    /// The element assume transfers **through** the slice via origin-id keying
    /// (§5.4): the gather proves (3 obligations) only with the element bound.
    #[test]
    fn slice_gather_copy_definition_is_provably_in_bounds() {
        let def = slice_gather_copy_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = slice_gather_copy_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [
            vericl_ir::Assume::LenEq { a: "offsets", b: "y" },
            vericl_ir::Assume::ElemsBelowLen { arr: "offsets", len_of: "x" },
        ];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 3),
            other => panic!("expected Proved (slice gather), got {other:?}"),
        }
    }

    /// The slice-param helper twin maps `&Slice<F>` -> `&[f32]` and computes
    /// what its body says (`w[0] + w[3]`) — composition support at the twin
    /// level (§10).
    #[test]
    fn window_edge_sum_helper_twin_maps_slice_to_host_slice() {
        assert_eq!(window_edge_sum_vericl_ref(&[10.0f32, 1.0, 2.0, 40.0]), 50.0);
    }

    /// The composed slice kernel's twin calls the helper twin on the collapsed
    /// `&x[i..i+4]` argument, computing `x[i] + x[i+3]`.
    #[test]
    fn windowed_helper_kernel_twin_matches_hand_computed() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut y = vec![0.0f32; 4];
        windowed_helper_kernel_vericl::reference(&x, &mut y, 4);
        // y[i] = x[i] + x[i+3]
        assert_eq!(y, vec![1.0 + 4.0, 2.0 + 5.0, 3.0 + 6.0, 4.0 + 7.0]);
    }

    /// F1 (round-9): the mutable-**write** slice twin scales in place through the
    /// `&mut y[i..i+1]` mapping of `y.slice_mut(ABSOLUTE_POS, ABSOLUTE_POS + 1)`,
    /// writing `s[0] = s[0] * alpha` = `y[i] * alpha`. Pins the write-path twin
    /// derivation at the value level (the `*_twin_matches_hand_computed`
    /// precedent, now on the `slice_mut` write lane). Also confirms the guard is
    /// honored: with more threads than elements, elements are each written once.
    #[test]
    fn slice_scale_inplace_twin_matches_hand_computed() {
        let mut y = vec![1.0f32, -2.0, 3.0, -4.0, 5.0];
        // Dispatch more threads than elements: the guard must leave nothing out
        // of place and thread i must be the sole writer of y[i].
        slice_scale_inplace_vericl::reference(&mut y, 2.0, 64);
        assert_eq!(y, vec![2.0, -4.0, 6.0, -8.0, 10.0]);
    }

    /// F1 (round-9): the mutable-slice **write** obligation proves end to end at
    /// the example level. `s[0] = s[0] * alpha` lowers to `IndexAssign(y,
    /// ABSOLUTE_POS + 0)` — the ordinary origin **write** obligation — plus the
    /// `s[0]` read of the same index, both discharged by the guard `ABSOLUTE_POS <
    /// y.len()` with **no assume**. (The prover-unit twin of this — a `slice_mut`
    /// write proving/refuting through the lowering — is `slice_mut_write_*` in
    /// crates/vericl-ir.)
    #[test]
    fn slice_scale_inplace_definition_is_provably_in_bounds() {
        let def = slice_scale_inplace_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = slice_scale_inplace_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &[]) {
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved (slice_mut write + read), got {other:?}"),
        }
    }

    /// F1 (round-9), the S3 mutable-aliasing convention **positive** control
    /// (docs/design-view-slice.md §4.3, §11 S3): two **sequential** disjoint
    /// `slice_mut` windows of one origin — `y.slice_mut(0, 4)` then
    /// `y.slice_mut(4, 8)` — each with a `[j]` write loop. The very fact this twin
    /// **compiles** is the control: sequential mutable slices are accepted by the
    /// borrow checker (the aliasing oracle) under NLL, because each `&mut (y)[a..b]`
    /// is dropped before the next is created. The **negative** control is the
    /// scratch compile-fail demo `scratchpad/slicemut/overlap.rs`, where two
    /// simultaneously-live *overlapping* `slice_mut` views fail rustc `E0499` — the
    /// borrow error is the (as-built) rejection (§8.3 [as-built], §8.4 for the
    /// deferred prettified message). This test also value-checks the two `[j]`
    /// windows scaled correctly.
    #[test]
    fn sequential_slice_mut_scale_twin_scales_two_windows() {
        let mut y = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // len 8
        sequential_slice_mut_scale_vericl::reference(&mut y, 10.0, 1);
        // Both disjoint windows [0,4) and [4,8) scaled by 10 (thread 0 only).
        assert_eq!(y, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    }

    // -----------------------------------------------------------------
    // Kernel composition: #[vericl::helper] + uses(...).
    // -----------------------------------------------------------------

    /// The `single_tap`/`fir_pair` helper twins compute exactly what their
    /// (very simple) bodies say — guards against the twin derivation itself
    /// going wrong for a helper, same purpose as `fir_handwritten`/`fmix32`
    /// above for kernels.
    #[test]
    fn helper_twins_match_hand_computed() {
        assert_eq!(single_tap_vericl_ref(3.0f32, 2.0f32), 6.0);
        assert_eq!(fir_pair_vericl_ref(5.0f32, 2.0f32), (7.0, 3.0));
    }

    /// `fir_pair_scaled` calls two OTHER helpers (`fir_pair`, `single_tap`)
    /// — its twin must genuinely compose their `_vericl_ref` twins, not
    /// silently fall back to something else. Cross-checked two ways: against
    /// a hand-composed value, and against literally calling the other two
    /// twins directly and combining them the same way the helper's own body
    /// does — the latter is only a meaningful check because it's the exact
    /// call chain `uses(fir_pair, single_tap)`'s rewrite is supposed to
    /// produce.
    ///
    /// Also pins the tuple-destructuring residual (docs/dogfood-2026-07.md) as
    /// closed: `fir_pair_scaled`'s body now uses `let (sum, diff) = fir_pair(a,
    /// b)` (device-fn-calling-device-fn), and this passing test confirms the twin
    /// derives correctly from that natural form — no `.0`/`.1` workaround.
    #[test]
    fn fir_pair_scaled_twin_composes_its_own_helpers() {
        let (a, b, gain) = (5.0f32, 2.0f32, 10.0f32);
        let got = fir_pair_scaled_vericl_ref(a, b, gain);

        let (sum, diff) = fir_pair_vericl_ref(a, b);
        let expected = (single_tap_vericl_ref(sum, gain), single_tap_vericl_ref(diff, gain));
        assert_eq!(got, expected);
        assert_eq!(got, (70.0, 30.0));
    }

    /// `tap_pair`'s twin reads its own two elements — the shape the
    /// composition-prover tests below rely on.
    #[test]
    fn tap_pair_twin_matches_hand_computed() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(tap_pair_vericl_ref(&x, 0), 3.0);
        assert_eq!(tap_pair_vericl_ref(&x, 2), 7.0);
    }

    /// `gain_kernel`'s twin honors its guard and matches a hand computation
    /// that never goes through `single_tap_vericl_ref` at all — guards
    /// against the composed kernel's *own* derivation (ABSOLUTE_POS rewrite,
    /// instantiate(...) substitution, uses(...) rewrite all combined) being
    /// wrong in a way an isolated helper-twin test wouldn't catch.
    #[test]
    fn gain_kernel_twin_matches_hand_computed() {
        let x = vec![1.0f32, -2.0, 3.0];
        let mut y = vec![0.0f32; x.len()];
        gain_kernel_vericl::reference(&x, &mut y, 2.0, x.len());
        assert_eq!(y, vec![2.0, -4.0, 6.0]);
    }

    /// Threads past the guard write nothing — same discipline as every
    /// other kernel's twin.
    #[test]
    fn gain_kernel_twin_respects_guard() {
        let x = vec![1.0f32; 3];
        let mut y = vec![9.0f32; 3];
        gain_kernel_vericl::reference(&x, &mut y, 2.0, 256);
        assert_eq!(y, vec![2.0, 2.0, 2.0]);
    }

    /// Fix 3 regression pin: `self_path_gain_kernel`'s twin (`self::`-
    /// qualified `single_tap` call) must produce byte-identical results to
    /// `gain_kernel`'s (bare call) — the same expected values as
    /// `gain_kernel_twin_matches_hand_computed` above, for the same inputs.
    /// Pre-fix, this would have *coincidentally* still passed (the
    /// bypassed, un-rewritten path called the original `#[cube] fn
    /// single_tap`, which is host-safe and computes the same thing as
    /// `single_tap_vericl_ref`) — the differential can't distinguish the
    /// two; see `uses_rewrite_fold_rewrites_self_qualified_helper_call` in
    /// `vericl-macros` for the AST-level pin that actually catches the
    /// bypass.
    #[test]
    fn self_path_gain_kernel_twin_matches_hand_computed() {
        let x = vec![1.0f32, -2.0, 3.0];
        let mut y = vec![0.0f32; x.len()];
        self_path_gain_kernel_vericl::reference(&x, &mut y, 2.0, x.len());
        assert_eq!(y, vec![2.0, -4.0, 6.0]);
    }

    /// Fix 3 regression pin: `self_path_gain_kernel`'s bounds proof
    /// discharges identically to `gain_kernel`'s (same obligation count) —
    /// the `self::`-qualified call composes through the prover exactly like
    /// the bare call, since both inline the same helper IR into the same
    /// kernel `Scope` either way.
    #[test]
    fn self_path_gain_kernel_definition_is_provably_in_bounds() {
        let def = self_path_gain_kernel_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = self_path_gain_kernel_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// `fir_pair_kernel`'s twin (two-level composition: kernel ->
    /// `fir_pair_scaled` -> `fir_pair`/`single_tap`, two `&mut Array`
    /// outputs) matches an independent hand computation.
    #[test]
    fn fir_pair_kernel_twin_matches_hand_computed() {
        let x = vec![1.0f32, 3.0, -2.0, 5.0];
        let mut sum_out = vec![0.0f32; x.len()];
        let mut diff_out = vec![0.0f32; x.len()];
        let gain = 2.0f32;
        fir_pair_kernel_vericl::reference(&x, &mut sum_out, &mut diff_out, gain, x.len());
        // Guard is ABSOLUTE_POS + 1 < x.len(), so only positions 0..len-2
        // write; last position(s) stay at their initial value.
        for pos in 0..x.len() - 2 {
            let (a, b) = (x[pos], x[pos + 1]);
            assert_eq!(sum_out[pos], (a + b) * gain, "sum_out[{pos}]");
            assert_eq!(diff_out[pos], (a - b) * gain, "diff_out[{pos}]");
        }
        assert_eq!(sum_out[x.len() - 1], 0.0);
        assert_eq!(diff_out[x.len() - 1], 0.0);
    }

    /// `uses(...)` is recorded on the contract, and `tap_pair` is reused
    /// verbatim by two different kernels (the milestone's explicit "two
    /// kernels using the same helper" ask) — both list exactly `["tap_pair"]`.
    #[test]
    fn uses_clause_is_recorded_and_helper_is_reused_by_two_kernels() {
        assert_eq!(gain_kernel_vericl::contract().uses, &["single_tap"]);
        assert_eq!(fir_pair_kernel_vericl::contract().uses, &["fir_pair_scaled"]);
        assert_eq!(tap_pair_guarded_kernel_vericl::contract().uses, &["tap_pair"]);
        assert_eq!(tap_pair_unguarded_kernel_vericl::contract().uses, &["tap_pair"]);
        // A non-composing kernel's uses() is empty.
        assert!(axpy_vericl::contract().uses.is_empty());
        // A helper composing two other helpers records both, in clause order.
        assert_eq!(fir_pair_scaled_vericl::USES, &["fir_pair", "single_tap"]);
    }

    // -----------------------------------------------------------------
    // Composition identity: uses(...) must make a helper body change
    // visible in the composing kernel's (and helper's) recorded identity,
    // without leaking into anything that doesn't use() it.
    // -----------------------------------------------------------------

    /// A composing kernel's *recorded* identity (`identity()`) is NOT the
    /// same as its own compile-time-only `SOURCE_HASH` — composition
    /// genuinely changes what gets recorded, and does so by exactly folding
    /// in the declared helper's own `identity_hash()` (verified by
    /// reproducing the combine independently via
    /// `vericl::combine_source_hash` and asserting byte-for-byte equality,
    /// not just "differs from something").
    #[test]
    fn composed_kernel_identity_folds_in_its_helpers_hash() {
        let recorded = gain_kernel_vericl::identity().source_hash;
        assert_ne!(recorded, gain_kernel_vericl::SOURCE_HASH);
        let expected = vericl::combine_source_hash(
            gain_kernel_vericl::SOURCE_HASH,
            &[single_tap_vericl::identity_hash()],
        );
        assert_eq!(recorded, expected);
    }

    /// A NON-composing kernel's recorded identity is an exact pass-through
    /// of its own `SOURCE_HASH` — proving `identity()` folds in exactly the
    /// `uses(...)`-declared set and nothing else, regardless of how many
    /// unrelated helpers exist elsewhere in this same crate. This is the
    /// "changing a helper's unused sibling changes neither hash" guarantee,
    /// checked structurally (identity() provably can't see an undeclared
    /// helper at all) rather than by an actual source edit + rebuild —
    /// which was additionally exercised by hand (not committed; see the
    /// task's verification report) by editing `single_tap`'s body and
    /// confirming `gain_kernel`'s identity()/ir_hash moved while
    /// `axpy`'s/`flatten_decode_scale`'s did not, then reverting.
    #[test]
    fn unused_helper_does_not_affect_an_unrelated_kernels_identity() {
        assert_eq!(axpy_vericl::identity().source_hash, axpy_vericl::SOURCE_HASH);
        assert_eq!(
            flatten_decode_scale_vericl::identity().source_hash,
            flatten_decode_scale_vericl::SOURCE_HASH,
        );
    }

    /// Helper-calling-helper: `fir_pair_scaled`'s OWN `identity_hash()`
    /// recursively folds in both `fir_pair`'s and `single_tap`'s hashes —
    /// verified by reproducing the combine independently, exactly as for
    /// the kernel-level test above.
    #[test]
    fn helper_calling_helper_identity_is_recursive() {
        let recorded = fir_pair_scaled_vericl::identity_hash();
        assert_ne!(recorded, fir_pair_scaled_vericl::SOURCE_HASH);
        let expected = vericl::combine_source_hash(
            fir_pair_scaled_vericl::SOURCE_HASH,
            &[fir_pair_vericl::identity_hash(), single_tap_vericl::identity_hash()],
        );
        assert_eq!(recorded, expected);
    }

    /// The two-level composition case (`fir_pair_kernel` -> `fir_pair_scaled`
    /// -> `fir_pair`/`single_tap`): the KERNEL's own recorded identity only
    /// ever combines with its *direct* dependency's already-recursive
    /// `identity_hash()` — it never needs to know about `fir_pair`/
    /// `single_tap` by name at all, yet a change two levels deep still
    /// reaches it, because `fir_pair_scaled_vericl::identity_hash()` (used
    /// here) already covers its own `uses(...)` the same way (pinned by
    /// `helper_calling_helper_identity_is_recursive` above).
    #[test]
    fn composed_kernel_identity_is_recursive_through_the_helper_chain() {
        let recorded = fir_pair_kernel_vericl::identity().source_hash;
        let expected = vericl::combine_source_hash(
            fir_pair_kernel_vericl::SOURCE_HASH,
            &[fir_pair_scaled_vericl::identity_hash()],
        );
        assert_eq!(recorded, expected);
    }

    // -----------------------------------------------------------------
    // Composition + the SMT bounds prover (README claim: composition needs
    // zero prover changes, since cube expansion inlines a used helper's IR
    // directly into the composing kernel's own Scope).
    // -----------------------------------------------------------------

    /// Positive control: `tap_pair`'s own `x[idx]`/`x[idx + 1]` reads live
    /// entirely inside the composed helper's body, not the kernel's — and
    /// the guard `ABSOLUTE_POS + 1 < x.len()` the KERNEL establishes still
    /// gets combined with them correctly. Must discharge `Proved` — this is
    /// the milestone's positive composition-prover result.
    #[test]
    fn tap_pair_guarded_kernel_definition_is_provably_in_bounds() {
        let def = tap_pair_guarded_kernel_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = tap_pair_guarded_kernel_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Proved { .. } => {}
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// Negative control: same helper, same kernel shape, but the guard only
    /// covers `ABSOLUTE_POS < x.len()` — one short of what `tap_pair`'s own
    /// unguarded `x[idx + 1]` read needs at the top position. Must
    /// `Refuted`, proving the obligation from inside the helper's body is
    /// genuinely walked and not silently dropped because it's composed
    /// rather than written directly in the kernel — the milestone's
    /// negative composition-prover result.
    #[test]
    fn tap_pair_unguarded_kernel_definition_refutes() {
        let def = tap_pair_unguarded_kernel_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = tap_pair_unguarded_kernel_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        let assumes = [vericl_ir::Assume::LenEq { a: "x", b: "y" }];
        match vericl_ir::prove_bounds_freedom(&def, &buffers, &assumes) {
            vericl_ir::ProveResult::Refuted { .. } => {}
            other => panic!("expected Refuted, got {other:?}"),
        }
    }

    /// The suite-wired composed kernels also prove — carrying both tested
    /// (differential, via `vericl::suite!`) and proved claims is the
    /// milestone's "composed kernel carries tested + proved claims" ask.
    #[test]
    fn suite_wired_composed_kernels_prove() {
        let gain_buffers: Vec<vericl_ir::BufferParam> = gain_kernel_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        match vericl_ir::prove_bounds_freedom(
            &gain_kernel_vericl::kernel_definition(),
            &gain_buffers,
            &[vericl_ir::Assume::LenEq { a: "x", b: "y" }],
        ) {
            vericl_ir::ProveResult::Proved { .. } => {}
            other => panic!("expected gain_kernel to prove, got {other:?}"),
        }

        let fir_buffers: Vec<vericl_ir::BufferParam> = fir_pair_kernel_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        match vericl_ir::prove_bounds_freedom(
            &fir_pair_kernel_vericl::kernel_definition(),
            &fir_buffers,
            &[
                vericl_ir::Assume::LenEq { a: "x", b: "sum_out" },
                vericl_ir::Assume::LenEq { a: "x", b: "diff_out" },
            ],
        ) {
            vericl_ir::ProveResult::Proved { .. } => {}
            other => panic!("expected fir_pair_kernel to prove, got {other:?}"),
        }
    }

    // =================================================================
    // Cooperative reduction kernels (shared-memory milestone M5).
    // =================================================================

    /// Independently-written sequential block-strided tree reduction — the
    /// cross-check reference for `block_sum_reduce`'s macro-derived twin. NOT
    /// derived from the kernel tokens: written by hand from the reduction
    /// algorithm, matching the GPU's tree order (so the check is bit-exact and
    /// not circular — same posture as `fmix32` / `xorshift_twin_matches_
    /// handwritten`).
    fn handwritten_block_sum(input: &[f32], cube_count: usize, cube_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; cube_count];
        for (c, slot) in out.iter_mut().enumerate() {
            let mut tile = vec![0.0f32; cube_dim];
            for (tid, cell) in tile.iter_mut().enumerate() {
                let abs = c * cube_dim + tid;
                *cell = if abs < input.len() { input[abs] } else { 0.0 };
            }
            let mut half = cube_dim / 2;
            while half > 0 {
                for tid in 0..half {
                    tile[tid] += tile[tid + half];
                }
                half /= 2;
            }
            *slot = tile[0];
        }
        out
    }

    /// Independently-written grid-stride squared-sum reduction — the
    /// cross-check reference for `grid_stride_reduce`'s twin.
    fn handwritten_grid_stride(data: &[f32], cube_count: usize, cube_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; cube_count];
        let stride = cube_dim * cube_count;
        for (c, slot) in out.iter_mut().enumerate() {
            let mut tile = vec![0.0f32; cube_dim];
            for (tid, cell) in tile.iter_mut().enumerate() {
                let mut k = c * cube_dim + tid;
                let mut local = 0.0f32;
                while k < data.len() {
                    local += data[k] * data[k];
                    k += stride;
                }
                *cell = local;
            }
            let mut half = cube_dim / 2;
            while half > 0 {
                for tid in 0..half {
                    tile[tid] += tile[tid + half];
                }
                half /= 2;
            }
            *slot = tile[0];
        }
        out
    }

    /// Independently-written windowed (comptime-`taps`) block reduction — the
    /// cross-check reference for `comptime_window_reduce`'s twin. `taps` is
    /// baked into the kernel's twin as a `let` const (cube-uniform), so it is a
    /// plain parameter here.
    fn handwritten_comptime_window(
        input: &[f32],
        cube_count: usize,
        cube_dim: usize,
        taps: u32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; cube_count];
        for (c, slot) in out.iter_mut().enumerate() {
            let mut tile = vec![0.0f32; cube_dim];
            for (tid, cell) in tile.iter_mut().enumerate() {
                let abs = c * cube_dim + tid;
                let mut local = 0.0f32;
                for j in 0..taps {
                    let idx = abs + j as usize;
                    if idx < input.len() {
                        local += input[idx];
                    }
                }
                *cell = local;
            }
            let mut half = cube_dim / 2;
            while half > 0 {
                for tid in 0..half {
                    tile[tid] += tile[tid + half];
                }
                half /= 2;
            }
            *slot = tile[0];
        }
        out
    }

    /// The comptime-parameter cooperative twin reproduces the independent
    /// handwritten windowed reduction bit-for-bit (`taps = 3` pinned by
    /// `instantiate(...)`). Pins that a `#[comptime]` value threads through the
    /// phase-split twin as a `let` const with no per-thread divergence.
    #[test]
    fn comptime_window_reduce_twin_matches_handwritten() {
        let cube_dim = 256usize;
        for &n in &[1usize, 3, 200, 256, 257, 512, 1000, 4096] {
            let input: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect();
            let cube_count = n.div_ceil(cube_dim).max(1);
            let mut got = vec![0.0f32; cube_count];
            comptime_window_reduce_vericl::reference(&input, &mut got, cube_count, cube_dim);
            let want = handwritten_comptime_window(&input, cube_count, cube_dim, 3);
            for c in 0..cube_count {
                assert_eq!(
                    got[c].to_bits(),
                    want[c].to_bits(),
                    "n={n} cube {c}: twin={} handwritten={}",
                    got[c],
                    want[c]
                );
            }
        }
    }

    /// The comptime-parameter cooperative kernel proves data-race free AND
    /// in-bounds: the `#[comptime] taps` accumulation loop bound is cube-uniform
    /// by construction, and each `input[idx]` read is bounded by its own `idx <
    /// input.len()` guard. Verifies the deferral's stated fact (a comptime loop
    /// bound is the easiest uniformity case) rather than assuming it.
    #[test]
    fn comptime_window_reduce_definition_is_race_free_and_in_bounds() {
        let def = comptime_window_reduce_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = comptime_window_reduce_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        match vericl_ir::prove_race_freedom(&def, &buffers, &[], 256) {
            vericl_ir::ProveResult::Proved { .. } => {}
            other => panic!("expected race-free Proved, got {other:?}"),
        }
    }

    /// Independently-written squared-sum block reduction — the cross-check
    /// reference for `composed_sq_reduce`'s twin (which calls the `square_sample`
    /// helper in phase 0). Written by hand from the algorithm, not the tokens.
    fn handwritten_composed_sq(input: &[f32], cube_count: usize, cube_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; cube_count];
        for (c, slot) in out.iter_mut().enumerate() {
            let mut tile = vec![0.0f32; cube_dim];
            for (tid, cell) in tile.iter_mut().enumerate() {
                let abs = c * cube_dim + tid;
                *cell = if abs < input.len() { input[abs] * input[abs] } else { 0.0 };
            }
            let mut half = cube_dim / 2;
            while half > 0 {
                for tid in 0..half {
                    tile[tid] += tile[tid + half];
                }
                half /= 2;
            }
            *slot = tile[0];
        }
        out
    }

    /// The composed cooperative twin (calling the rewritten `square_sample_
    /// vericl_ref` inside phase 0) reproduces the independent handwritten
    /// squared-sum reduction bit-for-bit. Pins that helper composition threads
    /// through the phase-split twin.
    #[test]
    fn composed_sq_reduce_twin_matches_handwritten() {
        let cube_dim = 256usize;
        for &n in &[1usize, 3, 200, 256, 257, 512, 1000, 4096] {
            let input: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect();
            let cube_count = n.div_ceil(cube_dim).max(1);
            let mut got = vec![0.0f32; cube_count];
            composed_sq_reduce_vericl::reference(&input, &mut got, cube_count, cube_dim);
            let want = handwritten_composed_sq(&input, cube_count, cube_dim);
            for c in 0..cube_count {
                assert_eq!(
                    got[c].to_bits(),
                    want[c].to_bits(),
                    "n={n} cube {c}: twin={} handwritten={}",
                    got[c],
                    want[c]
                );
            }
        }
    }

    /// The composed cooperative kernel proves data-race free AND in-bounds: cube
    /// inlines the barrier-free `square_sample` helper's IR into the kernel's
    /// scope, so the two-thread walk sees the squared load directly and the
    /// barrier structure is identical to `block_sum_reduce` (the helper adds no
    /// `sync_cube()`). Pins the prover lane of cooperative composition.
    #[test]
    fn composed_sq_reduce_definition_is_race_free_and_in_bounds() {
        let def = composed_sq_reduce_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = composed_sq_reduce_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        match vericl_ir::prove_race_freedom(&def, &buffers, &[], 256) {
            vericl_ir::ProveResult::Proved { .. } => {}
            other => panic!("expected race-free Proved, got {other:?}"),
        }
    }

    /// Independently-written reference for `emitter_reduce` (the acceptance
    /// example: comptime + composition + terminate + shared). Padding cubes
    /// (`c >= n_emitters`) are skipped — the workgroup-uniform terminate — and
    /// leave the zero-initialised output; active cubes reduce a squared block.
    fn handwritten_emitter_reduce(
        input: &[f32],
        cube_count: usize,
        cube_dim: usize,
        n_emitters: u32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; cube_count];
        for (c, slot) in out.iter_mut().enumerate() {
            if c >= n_emitters as usize {
                continue; // terminate!() — skip the whole cube
            }
            let mut tile = vec![0.0f32; cube_dim];
            for (tid, cell) in tile.iter_mut().enumerate() {
                let abs = c * cube_dim + tid;
                *cell = if abs < input.len() { input[abs] * input[abs] } else { 0.0 };
            }
            let mut half = cube_dim / 2;
            while half > 0 {
                for tid in 0..half {
                    tile[tid] += tile[tid + half];
                }
                half /= 2;
            }
            *slot = tile[0];
        }
        out
    }

    /// The `emitter_reduce` twin (comptime `n_emitters = 4` + `square_sample`
    /// composition + workgroup-uniform `terminate!()`) reproduces the independent
    /// handwritten reference bit-for-bit. The sizes include `cube_count > 4`, so
    /// the terminate actually skips padding cubes (both twin and reference leave
    /// them zero) — pinning that the twin's cube-level `continue` models the
    /// "skip the whole cube" semantics.
    #[test]
    fn emitter_reduce_twin_matches_handwritten() {
        let cube_dim = 256usize;
        // n = 4096 gives cube_count = 16 > n_emitters = 4 (12 cubes skipped).
        for &n in &[1usize, 3, 256, 512, 1000, 4096] {
            let input: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect();
            let cube_count = n.div_ceil(cube_dim).max(1);
            let mut got = vec![0.0f32; cube_count];
            emitter_reduce_vericl::reference(&input, &mut got, cube_count, cube_dim);
            let want = handwritten_emitter_reduce(&input, cube_count, cube_dim, 4);
            for c in 0..cube_count {
                assert_eq!(
                    got[c].to_bits(),
                    want[c].to_bits(),
                    "n={n} cube {c}: twin={} handwritten={}",
                    got[c],
                    want[c]
                );
            }
        }
    }

    /// `emitter_reduce` proves data-race free AND in-bounds. The prover models
    /// the workgroup-uniform `terminate!()` as a `!(CUBE_POS >= n_emitters)` path
    /// condition (uniformity verified, before any barrier), the `square_sample`
    /// helper inlines with no extra barrier (the barrier-count check passes), and
    /// the store's explicit guard bounds it. Pins the combined v1.1 shape on the
    /// proof lane.
    #[test]
    fn emitter_reduce_definition_is_race_free_and_in_bounds() {
        let def = emitter_reduce_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = emitter_reduce_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        match vericl_ir::prove_cooperative(
            &def,
            &buffers,
            &[],
            256,
            emitter_reduce_vericl::COOP_BARRIER_COUNT,
        ) {
            vericl_ir::CooperativeProof::Proved(_) => {}
            other => panic!("expected Proved (race-free + in-bounds), got {other:?}"),
        }
    }

    /// The macro-derived phase-split twin of `block_sum_reduce` reproduces the
    /// independent handwritten reduction bit-for-bit, across sizes that stress
    /// the tail-guard (`n` not a multiple of `cube_dim`) and multi-cube
    /// (`cube_count > 1`). Guards the source-to-twin derivation itself.
    #[test]
    fn block_sum_reduce_twin_matches_handwritten() {
        let cube_dim = 256usize;
        for &n in &[1usize, 3, 200, 256, 257, 512, 1000, 4096] {
            let input: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect();
            let cube_count = n.div_ceil(cube_dim).max(1);
            let mut got = vec![0.0f32; cube_count];
            block_sum_reduce_vericl::reference(&input, &mut got, cube_count, cube_dim);
            let want = handwritten_block_sum(&input, cube_count, cube_dim);
            for c in 0..cube_count {
                assert_eq!(
                    got[c].to_bits(),
                    want[c].to_bits(),
                    "n={n} cube {c}: twin={} handwritten={}",
                    got[c],
                    want[c]
                );
            }
        }
    }

    /// Same for `grid_stride_reduce`, including a small-`cube_count` /
    /// large-`data` configuration so the pre-barrier grid-stride accumulation
    /// loop runs many iterations (the `while k < data.len()` shape §4 requires
    /// be transformable).
    #[test]
    fn grid_stride_reduce_twin_matches_handwritten() {
        let cube_dim = 256usize;
        // (cube_count, data_len) pairs — the second column forces multi-iter.
        for &(cube_count, n) in &[(1usize, 300usize), (2, 700), (4, 4096), (3, 5000)] {
            let data: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.25 - 1.0).collect();
            let mut got = vec![0.0f32; cube_count];
            grid_stride_reduce_vericl::reference(&data, &mut got, cube_count, cube_dim);
            let want = handwritten_grid_stride(&data, cube_count, cube_dim);
            for c in 0..cube_count {
                assert_eq!(
                    got[c].to_bits(),
                    want[c].to_bits(),
                    "cube_count={cube_count} n={n} cube {c}: twin={} handwritten={}",
                    got[c],
                    want[c]
                );
            }
        }
    }

    /// Shared-memory definedness (docs/design-shared-memory.md §4.5): the
    /// generated twin poison-initialises shared memory, so a kernel that reads
    /// `tile[tid]` before writing it makes the reference **panic loudly**
    /// rather than silently reading a zero the GPU would read as garbage.
    #[test]
    #[should_panic(expected = "poison")]
    fn shared_read_before_write_twin_panics_on_poison() {
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];
        shared_read_before_write_vericl::reference(&input, &mut output, 1, 256);
    }

    /// Twin/prover subset agreement: the exact clean-room kernels whose twin
    /// M5 derives are also accepted by the race-freedom and cooperative bounds
    /// provers (the two lanes cover the *same* kernels — §4.3). `block_sum_
    /// reduce` proves data-race free and in-bounds at `cube_dim = 256`.
    #[test]
    fn block_sum_reduce_definition_is_race_free_and_in_bounds() {
        let def = block_sum_reduce_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = block_sum_reduce_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        // The race walk discharges data-race freedom AND the tree-reduction
        // bounds obligations that the single-thread cooperative bounds walk
        // defers (a `Branch::Loop` carrying `sync_cube()` is `OutOfSubset` in
        // the plain walk — see the prover module docs).
        match vericl_ir::prove_race_freedom(&def, &buffers, &[], 256) {
            vericl_ir::ProveResult::Proved { .. } => {}
            other => panic!("expected race-free Proved, got {other:?}"),
        }
    }

    /// Same agreement check for `grid_stride_reduce` (its extra pre-barrier
    /// grid-stride accumulation loop is modeled by the same walker).
    #[test]
    fn grid_stride_reduce_definition_is_race_free() {
        let def = grid_stride_reduce_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = grid_stride_reduce_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        match vericl_ir::prove_race_freedom(&def, &buffers, &[], 256) {
            vericl_ir::ProveResult::Proved { .. } => {}
            other => panic!("expected race-free Proved, got {other:?}"),
        }
    }

    /// The demo-defects racy twin `block_sum_reduce_racy` is REFUTED by the
    /// two-thread race walker (M7): the overlapping `tile[tid] += tile[tid + 1]`
    /// stride is a read-write race between adjacent threads, caught with a
    /// two-thread counterexample (`t1 == t2 + 1`) — deterministically, unlike
    /// the nondeterministic GPU differential divergence. Pins that the same
    /// clean `block_sum_reduce` shape, once made racy, flips Proved -> Refuted.
    #[test]
    fn block_sum_reduce_racy_definition_refutes_race_freedom() {
        let def = block_sum_reduce_racy_vericl::kernel_definition();
        let buffers: Vec<vericl_ir::BufferParam> = block_sum_reduce_racy_vericl::BUFFER_PARAMS
            .iter()
            .map(|(name, is_output)| vericl_ir::BufferParam { name, is_output: *is_output })
            .collect();
        match vericl_ir::prove_race_freedom(&def, &buffers, &[], 256) {
            vericl_ir::ProveResult::Refuted { obligation, counterexample } => {
                assert!(
                    obligation.contains("read-write race"),
                    "unexpected obligation: {obligation}"
                );
                assert!(
                    counterexample.contains("t1") && counterexample.contains("t2"),
                    "expected a two-thread counterexample: {counterexample}"
                );
            }
            other => panic!("expected Refuted (neighbor-stride race), got {other:?}"),
        }
    }

    /// Declared-reference fallback (candidate #3, docs/design-shared-memory.md
    /// §4.4/§6, M6): `block_sum_reduce_declared` opts into the author-supplied
    /// `block_sum_declared_ref` via `reference = …`. The generated `reference`
    /// forwards to it (so a kernel outside the transformable subset is
    /// supportable), and the `DECLARED_REFERENCE` const flips true so the suite
    /// tags the tested claim with the strictly-weaker
    /// `differential-declared-reference` check string — never the derived-twin
    /// `differential`. The derived-twin sibling (`block_sum_reduce`) keeps it
    /// false. The forwarded reference agrees bit-for-bit with the derived twin
    /// here (the two shapes are identical), demonstrating the fallback is a
    /// faithful — but separately-authored, hence weaker-claim — reference.
    #[test]
    fn block_sum_reduce_declared_uses_the_declared_reference() {
        const { assert!(block_sum_reduce_declared_vericl::DECLARED_REFERENCE) };
        const { assert!(!block_sum_reduce_vericl::DECLARED_REFERENCE) };
        assert_eq!(block_sum_reduce_declared_vericl::COOPERATIVE_CUBE_DIM, Some(256));

        let cube_dim = 256usize;
        for &n in &[1usize, 3, 257, 512] {
            let input: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect();
            let cube_count = n.div_ceil(cube_dim).max(1);
            // The declared kernel's forwarding reference == the hand-written fn.
            let mut got = vec![0.0f32; cube_count];
            block_sum_reduce_declared_vericl::reference(&input, &mut got, cube_count, cube_dim);
            let mut want = vec![0.0f32; cube_count];
            block_sum_declared_ref(&input, &mut want, cube_count, cube_dim);
            assert_eq!(got, want, "n={n}: forwarding must call the declared reference");
            // …and it agrees bit-for-bit with the DERIVED twin of the same shape.
            let mut derived = vec![0.0f32; cube_count];
            block_sum_reduce_vericl::reference(&input, &mut derived, cube_count, cube_dim);
            for c in 0..cube_count {
                assert_eq!(got[c].to_bits(), derived[c].to_bits(), "n={n} cube {c}");
            }
        }
    }

    /// Round-3 adversarial review F2 (inverted probe): a declared-reference
    /// kernel's *recorded* identity now folds in the `#[vericl::reference]`
    /// fn's own `SOURCE_HASH`, so a drift in the reference BODY moves the
    /// kernel's identity — the exact leak the reviewer found (identity stayed
    /// byte-identical because `SOURCE_HASH` only ever saw the `reference =
    /// <path>` clause text, never the referenced body). Verified structurally,
    /// by reproducing the combine independently (same posture as the
    /// `uses(...)` composition-identity tests above): `identity()` provably
    /// folds in exactly `block_sum_declared_ref_vericl::identity_hash()`, which
    /// is that reference's own `SOURCE_HASH`. An actual body-edit-moves-the-hash
    /// run was additionally done by hand (scratch, not committed — see the
    /// verification report). The derived-twin sibling `block_sum_reduce` (no
    /// `reference = …`) stays a pass-through, so the fold is scoped to declared
    /// references only.
    #[test]
    fn declared_reference_body_is_part_of_kernel_identity() {
        // The reference module composes nothing — its identity_hash is exactly
        // its own SOURCE_HASH.
        assert_eq!(
            block_sum_declared_ref_vericl::identity_hash(),
            block_sum_declared_ref_vericl::SOURCE_HASH,
        );
        const { assert!(block_sum_declared_ref_vericl::IS_VERICL_REFERENCE) };

        // The kernel's recorded identity is NOT its own SOURCE_HASH: the
        // reference's hash is genuinely folded in (so a reference-body drift
        // moves it), and by exactly `combine_source_hash(SOURCE_HASH, [ref])`.
        let recorded = block_sum_reduce_declared_vericl::identity().source_hash;
        assert_ne!(recorded, block_sum_reduce_declared_vericl::SOURCE_HASH);
        let expected = vericl::combine_source_hash(
            block_sum_reduce_declared_vericl::SOURCE_HASH,
            &[block_sum_declared_ref_vericl::identity_hash()],
        );
        assert_eq!(recorded, expected);

        // The derived-twin sibling declares no `reference = …`, so its recorded
        // identity is an exact pass-through of its own SOURCE_HASH — the fold is
        // scoped to declared-reference kernels only.
        assert_eq!(
            block_sum_reduce_vericl::identity().source_hash,
            block_sum_reduce_vericl::SOURCE_HASH,
        );
    }

    // ---------------------------------------------------------------------
    // Quick-wins batch 2: twin-derivation guards for the new example kernels.
    // Each pins the macro-derived twin (which now routes cast_from/mul_hi
    // through host shims, wraps a helper's body, or strips a comptime! block)
    // against independently hand-written scalar code.
    // ---------------------------------------------------------------------

    /// Feature 1: the `cast_from` shim path. The `to_unit_interval` helper twin
    /// must compute `(int_random >> 8) as f32 / 2^24` exactly — the same value
    /// the verified `cast_from_u32_f32` shim + exact power-of-two divide give.
    #[test]
    fn to_unit_interval_twin_matches_hand_computed() {
        for &v in &[0u32, 1, 255, 256, 0x00FF_FFFF, 0x0100_0000, 0xFFFF_FFFF, 0xDEAD_BEEF] {
            let got = to_unit_interval_vericl_ref(v);
            let want = ((v >> 8) as f32) / 16_777_216.0;
            assert_eq!(got.to_bits(), want.to_bits(), "int_random={v}");
            assert!((0.0..1.0).contains(&got), "out of [0,1): {got}");
        }
    }

    /// Feature 1: the flagship kernel twin routes through the helper and honors
    /// the guard (threads past `y.len()` write nothing).
    #[test]
    fn unit_interval_map_twin_matches_and_respects_guard() {
        let x: Vec<u32> = vec![0, 0x0100_0000, 0xFFFF_FFFF, 42];
        let mut y = vec![-1.0f32; x.len()];
        unit_interval_map_vericl::reference(&x, &mut y, 256); // threads >> len
        for (i, &v) in x.iter().enumerate() {
            assert_eq!(y[i].to_bits(), (((v >> 8) as f32) / 16_777_216.0).to_bits());
        }
    }

    /// Feature 1: the `mul_hi` shim path — the twin computes the high word of
    /// the full-width u32 product, `(a*b) >> 32`.
    #[test]
    fn mul_hi_map_twin_matches_hand_computed() {
        let a: Vec<u32> = vec![0, u32::MAX, 0x8000_0000, 65536, 0x1234_5678];
        let b: Vec<u32> = vec![u32::MAX, u32::MAX, 2, 65536, 0x9ABC_DEF0];
        let mut y = vec![0u32; a.len()];
        mul_hi_map_vericl::reference(&a, &b, &mut y, a.len());
        for i in 0..a.len() {
            let want = (((a[i] as u64) * (b[i] as u64)) >> 32) as u32;
            assert_eq!(y[i], want, "a={} b={}", a[i], b[i]);
        }
    }

    /// Feature 2: the WRAPPING `lcg_step` helper twin folds `z*a + b` to
    /// wrap-on-overflow — matching WGSL. The hand-written reference uses
    /// `wrapping_*` explicitly; a checked twin would panic on these inputs.
    #[test]
    fn lcg_step_twin_wraps_on_overflow() {
        for &z in &[0u32, 1, 0xFFFF_FFFF, 0x1234_5678, u32::MAX / 2] {
            let got = lcg_step_vericl_ref(z);
            let want = z.wrapping_mul(1664525).wrapping_add(1013904223);
            assert_eq!(got, want, "z={z}");
        }
    }

    /// Feature 2: the NON-wrapping kernel composing the wrapping helper — the
    /// interaction rule. The kernel twin calls the helper twin (which wraps);
    /// the result matches the fully-wrapping hand computation, and no overflow
    /// panic occurs despite the kernel itself being non-wrapping.
    #[test]
    fn lcg_map_twin_matches_hand_computed() {
        let x: Vec<u32> = vec![0, 1, 0xFFFF_FFFF, 0xDEAD_BEEF, u32::MAX];
        let mut y = vec![0u32; x.len()];
        lcg_map_vericl::reference(&x, &mut y, x.len());
        for (i, &z) in x.iter().enumerate() {
            assert_eq!(y[i], z.wrapping_mul(1664525).wrapping_add(1013904223), "z={z}");
        }
    }

    /// Feature 2 interaction rule (the loud half): a NON-wrapping helper's twin
    /// computes CHECKED arithmetic, so it panics on overflow rather than
    /// silently diverging from the GPU (which wraps). This is why a wrapping
    /// kernel composing a non-wrapping helper gets a loud signal, kept per
    /// round-3. `lcg_step`'s WRAPPING twin, by contrast, never panics
    /// (`lcg_step_twin_wraps_on_overflow`).
    #[test]
    fn nonwrapping_helper_twin_panics_on_overflow() {
        // z * 1664525 overflows u32 for large z; the checked twin panics.
        let panicked = std::panic::catch_unwind(|| lcg_step_checked_vericl_ref(u32::MAX)).is_err();
        assert!(panicked, "a non-wrapping helper twin must panic (not wrap) on overflow");
        // The wrapping sibling on the same input does NOT panic — it wraps.
        let wrapped = lcg_step_vericl_ref(u32::MAX);
        assert_eq!(wrapped, u32::MAX.wrapping_mul(1664525).wrapping_add(1013904223));
    }

    /// Feature 3: the `comptime!` block is evaluated at expansion — the twin
    /// shifts by `extra + 2` with `extra` pinned to 1, i.e. `>> 3`.
    #[test]
    fn comptime_shift_twin_evaluates_comptime_block() {
        let x: Vec<u32> = vec![0, 8, 0xFFFF_FFFF, 1024, 0x1234_5678];
        let mut y = vec![0u32; x.len()];
        comptime_shift_vericl::reference(&x, &mut y, x.len());
        for (i, &v) in x.iter().enumerate() {
            assert_eq!(y[i], v >> 3, "x={v}"); // extra(1) + 2 == 3
        }
    }
}
