//! Kernel contracts: the declared assumptions and comparison semantics that
//! evidence is produced under.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// How outputs of two realizations are compared.
///
/// The comparison mode is part of the contract: a tolerance is a declared
/// assumption recorded in the evidence, never an implementation detail.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Compare {
    /// Bit-exact equality. The only mode for integer kernels.
    Exact,
    /// Maximum permitted ULP distance between f32 results. NaN on either side
    /// is always a failure.
    MaxUlpF32(u32),
    /// Pass when `|expected - actual| <= abs + rel * |expected|`.
    ///
    /// The honest tolerance shape when a backend may contract or reorder
    /// float operations (e.g. fma): under cancellation no useful ULP bound
    /// exists, but an absolute bound derived from the declared input ranges
    /// does. The bound is part of the contract and must be justified by the
    /// `assumes(...)` clauses.
    AbsRelF32 {
        /// Absolute tolerance term.
        abs: f32,
        /// Relative tolerance term (scaled by `|expected|`).
        rel: f32,
    },
    /// f64 counterpart of [`Compare::MaxUlpF32`] — maximum permitted ULP
    /// distance between f64 results. The macro emits this (rather than the
    /// f32 variant) for an f64 kernel; see the `NumKind::F64` handling in
    /// `vericl-macros`.
    MaxUlpF64(u32),
    /// f64 counterpart of [`Compare::AbsRelF32`]. Tolerances are stored at f64
    /// precision (an f64 tolerance rounded to f32 would be a dishonest record
    /// of a bound the author declared for an f64 kernel).
    AbsRelF64 {
        /// Absolute tolerance term (f64 precision).
        abs: f64,
        /// Relative tolerance term (scaled by `|expected|`).
        rel: f64,
    },
}

impl Compare {
    /// A short human-readable description of the comparison mode, as recorded
    /// in the evidence `contract.compare` field (e.g. `"f32 max_ulp=0"`).
    pub fn describe(&self) -> String {
        match self {
            Compare::Exact => "exact".to_string(),
            Compare::MaxUlpF32(n) => format!("f32 max_ulp={n}"),
            Compare::AbsRelF32 { abs, rel } => {
                format!("f32 |e-a| <= {abs:e} + {rel:e}*|e|")
            }
            Compare::MaxUlpF64(n) => format!("f64 max_ulp={n}"),
            Compare::AbsRelF64 { abs, rel } => {
                format!("f64 |e-a| <= {abs:e} + {rel:e}*|e|")
            }
        }
    }
}

/// An `assumes(...)` clause the macro recognized as a specific structured
/// shape, in addition to keeping it in `Contract::assumes` as a
/// pretty-printed string. This is the data the SMT bounds prover
/// (`vericl-ir`) binds buffer `Length` variables from — it has no other way
/// to relate a `.len()` constraint to a specific declared assumption.
/// Buffer identity is by parameter name; the harness maps names to the IR's
/// `input(i)`/`output(j)` positions via the macro-generated `BUFFER_PARAMS`.
///
/// Recognizing more clause shapes only ever adds provable obligations —
/// never silently loosens one — so growing this enum is always sound.
///
/// Generated-code plumbing: constructed by `#[vericl::kernel]` and consumed by
/// the `suite!`/`conform` prover wiring — not an API user code calls.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredAssume {
    /// `A.len() == B.len()`.
    LenEq { a: &'static str, b: &'static str },
    /// `A.len() == <int literal>`.
    LenEqConst { a: &'static str, value: u64 },
    /// `A.iter().all(|v| (*v as usize) < B.len())` — every element of the
    /// integer array `A` is a valid index into `B` (an array-value-dependent
    /// / gather index bound). Unlike the length assumptions above, this is a
    /// *content* claim; the SMT prover uses it to model a read `A[i]` as a
    /// fresh symbol `< B.len()` (see `vericl-ir`'s "Element-range
    /// assumptions"), and it is invalidated for `A` by any write to `A`'s
    /// elements.
    ElemsBelowLen { arr: &'static str, len_of: &'static str },
    /// `A.iter().all(|v| *v < N)` — every element of `A` is below the integer
    /// literal `N` (the constant-bound sibling of `ElemsBelowLen`).
    ElemsBelowConst { arr: &'static str, bound: u64 },
    /// `A.len() + K <= B.len()` (and the `K = 0` case `A.len() <= B.len()`) for
    /// an integer literal `K` — a length *relationship* between two array
    /// parameters (the "additive anchor" host-side buffer-sizing invariant). The
    /// SMT prover asserts `len_a + K <= len_b`, which — combined with a guard
    /// `i < A.len()` — discharges an offset read `B[i + K]` in bounds. Unlike
    /// the element-range forms above, the recognized Rust relation `<=` maps
    /// *directly* onto the modeled `<=`: the source clause IS the constraint,
    /// asserted verbatim (there is no index-validity reinterpretation, so `<=`
    /// is exactly correct here where only `<` was sound for the element case).
    LenPlusConstLe { a: &'static str, k: u64, b: &'static str },
    /// `A.len() == (x as usize) * (y as usize)` for two runtime `u32` scalar
    /// parameters — the fact that ties a 2-D/3-D dispatch's extents to a buffer
    /// length (docs/design-2d-dispatch.md §4.6). Without it a 2-D write
    /// obligation `y*w + x < out.len()` is genuinely SAT (measured, `p2e`) and
    /// the `checked_mul` side-obligation on the row stride `y*w` has nothing to
    /// bound it — this assume is the *enabling fact* of the whole milestone, not
    /// ergonomics.
    ///
    /// `x_scalar`/`y_scalar` are the operands' IR `GlobalScalar` ids; the prover
    /// has no way to recover a scalar parameter's name from the IR (scalars are
    /// just `scalar<u32>(id)`), exactly as it has none for buffers, so the macro
    /// carries the mapping here the way `BUFFER_PARAMS` carries the buffer one.
    ///
    /// **Only the widen-then-multiply spelling is ever recognized.** Written
    /// `A.len() == (x * y) as usize` the executable `check_assumes` tests the
    /// WRAPPED u32 product while this asserts the mathematical one — a false
    /// `Proved` with the measured witness `x = 2, y = 2147483649, len = 2`. That
    /// spelling is rejected at the macro (R6), never silently ignored.
    LenEqProduct {
        a: &'static str,
        x: &'static str,
        y: &'static str,
        x_scalar: u32,
        y_scalar: u32,
    },
}

/// Static contract metadata generated by `#[vericl::kernel]`.
#[derive(Debug, Clone, Copy)]
pub struct Contract {
    /// Kernel function name.
    pub kernel: &'static str,
    /// Hash of the kernel source tokens + contract + vericl version.
    ///
    /// v0 identity is source-level; an IR-level hash is a recorded upgrade
    /// path (see README "Locked decisions").
    pub source_hash: &'static str,
    /// Pretty-printed `assumes(...)` clauses.
    pub assumes: &'static [&'static str],
    /// The subset of `assumes(...)` the macro could parse into a structured
    /// shape (see [`StructuredAssume`]). Unrecognized clauses are simply
    /// absent here — they still appear in `assumes` above.
    pub structured_assumes: &'static [StructuredAssume],
    /// Declared comparison semantics.
    pub compare: Compare,
    /// Whether the `wrapping` clause is declared: the reference twin's
    /// integer `+`, `-`, `*`, `<<`, `>>` (and their compound-assign forms)
    /// use wrapping arithmetic instead of Rust's default checked/panicking
    /// behavior, matching WGSL's wrap-on-overflow semantics. A declared
    /// contract clause, not a silent approximation — see README "A first
    /// finding" for the analogous fma story.
    pub wrapping: bool,
    /// Pretty-printed `instantiate(...)` entries (`"F = f32"`, `"taps =
    /// 3"`), one per generic type parameter or `#[comptime]` parameter the
    /// kernel declares. Empty for a non-generic, non-comptime kernel. The
    /// instantiation *values* are already part of `SOURCE_HASH` (they're in
    /// the raw contract attribute tokens the hash covers) — this field
    /// exists purely so evidence records what a kernel was monomorphized
    /// at, the same way `wrapping` records a declared clause.
    pub instantiate: &'static [&'static str],
    /// Names of the `#[vericl::helper]`-annotated functions this kernel
    /// declares via `uses(...)` (kernel composition). `[]` for a kernel that
    /// calls no helpers. Purely for evidence legibility — `identity()`
    /// (not `identity`/this struct) is what actually folds each used
    /// helper's identity into the recorded source hash; see its doc.
    pub uses: &'static [&'static str],
}

/// Serializable form of a [`Contract`] for the evidence manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContractRecord {
    /// Pretty-printed `assumes(...)` clauses.
    pub assumes: Vec<String>,
    /// The declared comparison mode ([`Compare::describe`]).
    pub compare: String,
    /// Whether the `wrapping` clause was declared.
    pub wrapping: bool,
    /// See [`Contract::instantiate`]. `[]` for a non-generic, non-comptime
    /// kernel. `#[serde(default)]` so evidence written before this field
    /// existed still loads (as `[]`) instead of hard-failing deserialization
    /// — unlike a source/IR hash change, adding this field never changes
    /// `SOURCE_HASH` for a kernel that doesn't use `instantiate(...)`, so an
    /// old manifest for such a kernel is otherwise still perfectly valid.
    #[serde(default)]
    pub instantiate: Vec<String>,
    /// See [`Contract::uses`]. `#[serde(default)]` for the same reason as
    /// `instantiate` above — old evidence for a non-composing kernel is
    /// unaffected by this field's addition.
    #[serde(default)]
    pub uses: Vec<String>,
}

/// Identity a piece of evidence is bound to. Any mismatch between a stored
/// identity and the currently built kernel makes the evidence stale.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Identity {
    /// Hash of the kernel source tokens + contract + vericl version (composition-
    /// aware for a kernel with `uses(...)`; see `<kernel>_vericl::identity()`).
    pub source_hash: String,
    /// The `vericl` version that produced the evidence.
    pub vericl_version: String,
    /// Content hash of the expanded CubeCL IR (see `vericl-ir::kernel_ir_hash`).
    ///
    /// `None` only for evidence produced without IR access — this crate is
    /// deliberately cubecl-free (see README "Locked decisions": "isolate
    /// all IR-facing code in one crate"), so `Contract::identity()` can
    /// never compute this itself; the harness sets it after computing it
    /// via `vericl-ir`. With both hashes populated, `verify()`'s whole-
    /// `Identity` comparison catches IR-level drift (e.g. a CubeCL upgrade
    /// that changes codegen without changing kernel source) in addition to
    /// source-level drift.
    #[serde(default)]
    pub ir_hash: Option<String>,
}

impl Contract {
    /// Source-level identity only — `ir_hash` is left `None` here since this
    /// crate cannot depend on cubecl to compute it (see [`Identity::ir_hash`]);
    /// callers with IR access fill it in afterward.
    pub fn identity(&self) -> Identity {
        Identity {
            source_hash: self.source_hash.to_string(),
            vericl_version: crate::VERSION.to_string(),
            ir_hash: None,
        }
    }

    /// The serializable [`ContractRecord`] form of this contract, for the manifest.
    pub fn record(&self) -> ContractRecord {
        ContractRecord {
            assumes: self.assumes.iter().map(|s| s.to_string()).collect(),
            compare: self.compare.describe(),
            wrapping: self.wrapping,
            instantiate: self.instantiate.iter().map(|s| s.to_string()).collect(),
            uses: self.uses.iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Identity of a struct/enum used as a struct-typed `#[comptime]` kernel
/// parameter (a *config type*), implemented by the `vericl::config! { … }` item
/// macro — and by `vericl::cube_struct! { … }` for the declared types that can
/// actually stand in that position (see "Cube structs in comptime position"
/// below).
///
/// # Why this trait exists
///
/// A kernel's `SOURCE_HASH` covers its own tokens plus the contract attribute
/// tokens (see [`Contract::source_hash`]). A config type's *definition* — its
/// fields and, decisively, its **method bodies** — is in neither: it lives in
/// separate items the kernel's `#[proc_macro_attribute]` invocation cannot
/// see. Measured consequence before this trait existed: editing a config
/// method from `self.m * self.n` to `self.m + self.n` changed the kernel from
/// ×24 to ×11 while leaving `SOURCE_HASH` bit-identical, so stored evidence
/// stayed "fresh" while describing a different kernel
/// (`docs/design-struct-comptime.md` §5.1).
///
/// [`CONFIG_HASH`](ConfigIdentity::CONFIG_HASH) is a SHA-256 over the **whole**
/// `vericl::config!` token block (every declared type, every impl block, every
/// method body), which a kernel folds into its recorded identity via
/// [`combine_source_hash`] — exactly the way `uses(...)` folds a helper's hash
/// and `reference = path` folds a declared reference's. Requiring the trait is
/// also what *forces* the declaration: a struct-typed `#[comptime]` parameter
/// whose type is not wrapped in `vericl::config!` fails to compile, with the
/// `#[diagnostic::on_unimplemented]` message below naming the fix.
///
/// # Do not implement this by hand
///
/// A hand-written impl can claim any hash it likes, including a constant one —
/// which reintroduces exactly the identity hole the trait closes. Only
/// `vericl::config!` derives a hash that actually covers the definition. See
/// `docs/guide.md` §5.1 and the README's struct-comptime section.
///
/// # Scalar type aliases
///
/// The macro classifies a `#[comptime]` parameter as a config by *syntax*: any
/// type that is not a written-out scalar primitive. A **type alias** for a
/// scalar (`type Taps = u32;`) is therefore classified as a config, because a
/// `#[proc_macro_attribute]` sees only the tokens of the item it annotates and
/// has no name resolution — it cannot know that `Taps` *is* `u32` (round-10
/// review, moderate 6).
///
/// rustc can, though, and the impls below are where that resolution happens:
/// each scalar primitive carries a `CONFIG_HASH` naming the type itself, so
/// `#[comptime] taps: Taps` compiles and folds `"vericl-scalar:u32"`. This is
/// the honest identity for a scalar — a primitive has no user-authored
/// definition that could drift — and it is not a weakening: changing the alias
/// to `type Taps = u64;` moves the folded hash and re-stales the evidence, which
/// is exactly what an identity fold is for. An alias to anything *else* still
/// hits the `#[diagnostic::on_unimplemented]` message below, which names the
/// alias case explicitly so the diagnosis is not misleading.
///
/// # Cube structs in comptime position (round-11 correction)
///
/// `vericl::cube_struct!` also emits this trait — but **only** for the types
/// that can genuinely occupy `#[comptime]` position, which is not all of them.
/// The trait is one of two requirements; the other is CubeCL's, and it is not
/// negotiable: a comptime parameter is `Debug`-formatted and its
/// `CompilationArg` derives `Hash`/`Eq`. Measured (round-11 review) — a
/// two-`u32` `cube_struct!` type in comptime position failed with `no method
/// named 'hash'`, `no method named 'eq'` and "doesn't implement `Debug`" while
/// its `ConfigIdentity` impl was present and correct.
///
/// So `cube_struct!` now emits those four derives itself for a declared type
/// whose transitive field shape is entirely integers/`bool`/`char`/unit enums,
/// and emits `ConfigIdentity` for exactly the same set. A struct with an
/// `f32`/`f64` field anywhere gets neither, because no derive set can give
/// `f32` `Hash` or `Eq` — such a type is a runtime parameter type only, and
/// naming it in comptime position lands on the note below rather than on three
/// raw trait errors pointing at `#[cube(launch)]`.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is used as a struct-typed #[comptime] parameter but is not declared with a `vericl::config!` block",
    label = "not a vericl config type",
    note = "wrap the type AND its impl blocks in `vericl::config! {{ … }}` so vericl can fold the config's definition into kernel identity and gate its method bodies for host-callability",
    note = "if `{Self}` is a TYPE ALIAS, note that vericl's macro cannot see through it (a proc macro has no name resolution): an alias for a scalar primitive resolves here automatically, but an alias for a struct/enum needs that underlying type declared with `vericl::config!`",
    note = "if `{Self}` is declared with `vericl::cube_struct!`, note that only a declared type whose fields are ALL integer/bool/char/unit-enum (transitively) can occupy #[comptime] position: CubeCL Debug-formats a comptime parameter and derives Hash/Eq over it, and `f32`/`f64` is none of those — so a float-field cube struct is a RUNTIME parameter type only. Pass it as `p: T` / `p: &T` and pin the integer parts with `#[cube(comptime)]` fields, or declare a separate all-integer type for the comptime half"
)]
pub trait ConfigIdentity {
    /// SHA-256 (as `"sha256:<hex>"`) over the entire `vericl::config!` token
    /// block that declared this type. Folded into the kernel's recorded
    /// `source_hash` by [`combine_source_hash`].
    const CONFIG_HASH: &'static str;
}

/// Identity of a struct used as a **runtime** (non-`#[comptime]`) `CubeType`
/// kernel/helper parameter, or constructed as a struct literal in a kernel or
/// helper body — implemented **only** by the `vericl::cube_struct! { … }` item
/// macro.
///
/// # Why this trait exists
///
/// This is [`ConfigIdentity`]'s argument moved one parameter position over, and
/// it closes a hole that was **live** rather than hypothetical. A kernel's
/// `SOURCE_HASH` covers its own tokens plus the contract attribute tokens; a
/// runtime struct type's *definition* — its field names, field types, field
/// **order**, and any `#[cube] impl` method reachable from the body — is in
/// neither.
///
/// Measured before this trait existed (`docs/design-cubetype-args.md` §4.1,
/// probe V3/V4): a `#[vericl::helper] fn use_pair(p: Pair) -> u32` was accepted
/// with **no diagnostic at all**, and editing `#[cube] impl Pair { fn fold }`
/// from `self.a * self.b` to `self.a + self.b` moved the reference twin from
/// `[3, 6, 9, 12]` to `[4, 5, 6, 7]` while the kernel's `SOURCE_HASH`, the
/// helper's `SOURCE_HASH` **and** `identity().source_hash` all stayed
/// bit-identical. Recorded evidence verified FRESH against a different computed
/// function.
///
/// There is a second, launch-side hazard the hash also covers: CubeCL generates
/// `<Name>Launch::new(…)` **positionally** in field-declaration order
/// (`generate_struct.rs:92-114`), so swapping two same-typed fields in the
/// *declaration* changes what the kernel computes with the kernel body and the
/// launch-call text byte-unchanged (§4.3, probe X2). Under
/// `vericl::cube_struct!` VeriCL emits that constructor itself from the declared
/// field order — so the reorder stays *correct*, and
/// [`STRUCT_HASH`](StructIdentity::STRUCT_HASH) moving is what makes the stored
/// evidence correctly stale.
///
/// # Do not implement this by hand
///
/// A hand-written impl can claim any hash it likes, including a constant one,
/// which reintroduces exactly the hole the trait closes. Only
/// `vericl::cube_struct!` derives a hash that covers the definition — and only
/// it emits the `CubeType`/`CubeLaunch` derives and the launch constructor from
/// the field order it hashed.
///
/// # Enums
///
/// `vericl::cube_struct!` emits `StructIdentity` for the **structs** it
/// declares, never for an enum: a payload-carrying runtime enum lowers to a tag
/// plus every variant's payload and has no twin model in the v1 subset, and a
/// unit enum's place in the subset is as a `#[cube(comptime)]` *field* type or a
/// `#[comptime]` parameter (where it needs no `CubeType` derive at all — though
/// it does need `Clone`/`Copy`/`Debug`/`PartialEq`/`Eq`/`Hash`, which the macro
/// emits for it as of round 11; without them the shape did not compile).
/// A declared enum therefore gets [`ConfigIdentity`] only, and naming it in
/// runtime parameter position lands on the message below.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is used as a runtime CubeType parameter but is not declared with a `vericl::cube_struct!` block",
    label = "not a vericl cube struct",
    note = "wrap the struct declaration in `vericl::cube_struct! {{ … }}` so vericl can fold the struct's definition into kernel identity, emit the CubeType/CubeLaunch derives, and build the launch argument from the declared field order — a field reorder or type change would otherwise alter what the kernel computes while leaving its recorded identity bit-identical",
    note = "if `{Self}` is declared with `vericl::config!`, note that a config type is NOT a runtime parameter type: `vericl::config!` gates its methods for HOST-callability because a comptime config runs on the host, while a runtime parameter is device data. Declare it with `vericl::cube_struct!` instead — a `cube_struct!` type may ALSO be used as a #[comptime] parameter when every one of its fields is integer/bool/char/unit-enum (CubeCL Debug-formats a comptime parameter and derives Hash/Eq over it, which no float field can satisfy), and the reverse is never sound",
    note = "if `{Self}` is an ENUM, a payload-carrying runtime enum parameter is outside the vericl v0 subset (CubeCL lowers it to a tag plus every variant's payload, and the twin would need a matching host discriminant model); a `#[cube(comptime)]` unit-enum FIELD inside a `vericl::cube_struct!` type is supported instead"
)]
pub trait StructIdentity {
    /// SHA-256 (as `"sha256:<hex>"`) over the entire `vericl::cube_struct!`
    /// token block that declared this type. Folded into the kernel's or
    /// helper's recorded `source_hash` by [`combine_source_hash`].
    const STRUCT_HASH: &'static str;
}

/// Scalar primitives carry an identity naming the type itself — see
/// [`ConfigIdentity`]'s "Scalar type aliases" section for why these exist and
/// why a constant is the right value here (unlike a hand-written impl for a
/// *struct*, which would hide a real definition).
macro_rules! scalar_config_identity {
    ($($t:ty),* $(,)?) => {
        $(impl ConfigIdentity for $t {
            const CONFIG_HASH: &'static str = concat!("vericl-scalar:", stringify!($t));
        })*
    };
}
scalar_config_identity!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, bool, char, f32, f64);

/// Fold a kernel's or helper's own (compile-time) source hash together with
/// the already-computed identity hashes of every helper it directly
/// `uses(...)`, producing the hash actually recorded as identity.
///
/// This is the runtime half of kernel-composition identity (see
/// `crates/vericl-macros`' `#[vericl::helper]`/`uses(...)` design): a
/// kernel's `SOURCE_HASH` constant only ever covers its own source tokens
/// (computable at macro-expansion time), so it cannot by itself reflect a
/// change to a helper's body — that only exists as a separate, sibling
/// `SOURCE_HASH` constant vericl has no way to fold in at compile time
/// (macro invocations cannot see each other's output). Folding the used
/// helpers' hashes in here, at ordinary Rust runtime, closes that gap:
/// `<kernel>_vericl::identity()` calls this with `deps` built from each
/// `uses(...)`-listed helper's own `identity_hash()` (itself computed the
/// same way, recursively — see that function's generated doc), so a change
/// two levels deep in the helper-call graph still changes the top-level
/// kernel's recorded `source_hash`. An empty `deps` (the overwhelmingly
/// common case: a kernel/helper with no `uses(...)` clause) is a pure
/// pass-through, so this is a no-op for every kernel that doesn't compose —
/// existing evidence for such kernels is unaffected.
///
/// Recognizing more dependencies only ever changes the combined hash, never
/// silently drops a real change — the reverse (a dependency change that
/// fails to move the combined hash) would be the unsound direction, and
/// isn't possible here since every `deps` entry is mixed into the digest.
///
/// **`deps`'s order matters, not just its contents.** `deps` is fed to the
/// hasher in whatever order the caller passes it — which, for the generated
/// `identity()`, is exactly the order the corresponding `uses(...)` clause
/// listed its helpers in. Reordering a `uses(a, b)` clause to `uses(b, a)`
/// (the same dependency *set*) therefore changes the resulting hash, even
/// though nothing about the kernel's or helper's actual behavior changed.
/// This is deliberately the safe direction to be sensitive in — it can only
/// ever cause spurious "evidence is stale, please re-run" churn after a
/// purely cosmetic reordering, never let a real dependency change through
/// unnoticed — but is worth knowing before reordering a `uses(...)` list
/// expecting evidence to stay untouched.
#[doc(hidden)] // generated-code plumbing (called by `<kernel>_vericl::identity()`)
pub fn combine_source_hash(local: &str, deps: &[String]) -> String {
    if deps.is_empty() {
        return local.to_string();
    }
    let mut hasher = Sha256::new();
    hasher.update(local.as_bytes());
    for d in deps {
        hasher.update(b"||uses:");
        hasher.update(d.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

/// The deepest helper-composition chain `combine_source_hash`'s generated
/// callers (`identity()`/`identity_hash()`) will recurse through before
/// panicking (see [`check_helper_composition_depth`]) — a runtime backstop,
/// independent of and in addition to vericl-macros' best-effort
/// compile-time `uses(...)` cycle check (which cannot see across separate
/// macro invocations with full reliability; see that check's doc). 32 is
/// far beyond any plausible legitimate helper-call depth (v0's own examples
/// nest at most two deep) and cheap to check on every call.
#[doc(hidden)] // generated-code plumbing (runtime cycle backstop)
pub const MAX_HELPER_COMPOSITION_DEPTH: u32 = 32;

/// Panics with an actionable message identifying `name` if `depth` has
/// reached [`MAX_HELPER_COMPOSITION_DEPTH`]. Called by every macro-generated
/// `identity()`/`identity_hash()` before it recurses into its own
/// `uses(...)`-listed dependencies. Without this, a recursive or
/// mutually-recursive `uses(...)` declaration that slipped past
/// vericl-macros' compile-time cycle check would make this combine hang
/// forever chasing its own tail instead of failing loudly and namely — a
/// silent hang is strictly worse than a clear panic naming the offending
/// item.
#[doc(hidden)] // generated-code plumbing (called by `<kernel>_vericl::identity()`)
pub fn check_helper_composition_depth(name: &str, depth: u32) {
    assert!(
        depth < MAX_HELPER_COMPOSITION_DEPTH,
        "vericl: helper composition depth exceeded {MAX_HELPER_COMPOSITION_DEPTH} while \
         combining source-hash identity through `{name}` — this almost always means a \
         recursive or mutually-recursive uses(...) declaration slipped past vericl-macros' \
         compile-time cycle check; fix the uses(...) clauses so the helper-call graph is \
         acyclic"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_source_hash_is_pass_through_with_no_deps() {
        assert_eq!(combine_source_hash("local-hash", &[]), "local-hash");
    }

    #[test]
    fn combine_source_hash_is_deterministic_and_dep_sensitive() {
        let a = combine_source_hash("local", &["dep-a".to_string()]);
        let a_again = combine_source_hash("local", &["dep-a".to_string()]);
        let b = combine_source_hash("local", &["dep-b".to_string()]);
        let two_deps = combine_source_hash("local", &["dep-a".to_string(), "dep-b".to_string()]);
        assert_eq!(a, a_again, "same inputs must hash identically");
        assert_ne!(a, "local", "folding in a dep must change the hash");
        assert_ne!(a, b, "a different dep hash must change the combined hash");
        assert_ne!(a, two_deps, "an extra dep must change the combined hash");
    }

    /// Direct test of the runtime depth guard's own threshold behavior —
    /// independent of whether a real compiling cycle can be constructed to
    /// exercise it end to end (see `crates/vericl-macros`'
    /// `register_and_check_cycle` doc: every cycle constructed by hand
    /// during this milestone's verification was in fact caught at compile
    /// time, so this guard is believed unreachable via vericl-macros'
    /// generated code in practice — this test pins the backstop itself
    /// works correctly regardless).
    #[test]
    fn helper_composition_depth_guard_trips_at_the_threshold() {
        for d in 0..MAX_HELPER_COMPOSITION_DEPTH {
            check_helper_composition_depth("ok", d); // must not panic
        }
        let trapped = std::panic::catch_unwind(|| {
            check_helper_composition_depth("cyclic_helper", MAX_HELPER_COMPOSITION_DEPTH);
        });
        assert!(trapped.is_err(), "depth == MAX must panic");
        let msg = *trapped.unwrap_err().downcast::<String>().expect("panic payload is a String");
        assert!(msg.contains("cyclic_helper"), "panic message should name the offending item: {msg}");
    }
}
