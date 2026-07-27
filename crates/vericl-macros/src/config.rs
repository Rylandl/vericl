//! `vericl::config! { … }` — the declaration form for a struct-typed
//! `#[comptime]` kernel parameter's type (a *config type*).
//!
//! # What it is for
//!
//! CubeCL lets a `#[cube]` item take `#[comptime] cfg: SomeConfig` where
//! `SomeConfig` is an ordinary user struct/enum: field reads and method calls
//! on it are re-emitted as **plain host Rust** executed while the IR is built
//! (`docs/design-struct-comptime.md` §1.2), so the value never reaches the
//! device — only the constants it computes do. VeriCL accepted that shape
//! before this macro existed, but ungated and unclaimed, with three measured
//! defects (design §5):
//!
//! 1. **the identity hole** — a config type's *definition* is in neither of
//!    `SOURCE_HASH`'s inputs (the kernel's own tokens, the contract attribute
//!    tokens), so editing a config method body from `self.m * self.n` to
//!    `self.m + self.n` changed the kernel from ×24 to ×11 and left the
//!    recorded identity bit-identical;
//! 2. **the config-method gate hole** — the kernel-body walkers
//!    (`FloatMethodCheck` and friends) are handed the kernel's body; a config
//!    method body is a different item and is invisible to them, so an
//!    `unexpanded!()` intrinsic in one failed only at run time;
//! 3. **unsound pinned expressions** — `instantiate(cfg = cfg_from_env())` was
//!    accepted with no purity requirement at all (gated on the kernel side, see
//!    `is_pinnable_config_expr` in the crate root).
//!
//! Wrapping the type **and every one of its impl blocks** in one item macro is
//! what makes 1 and 2 fixable: an attribute on the type could not see the impl
//! blocks (they are separate items), and that is exactly where both defects
//! live. This macro receives all of it as one token stream, which is the unit
//! that determines the kernel's meaning.
//!
//! # What it does
//!
//! - **Re-emits every item verbatim.** A config is ordinary host Rust and must
//!   stay so; nothing here rewrites a token.
//! - **Hashes the whole block** (every type, every impl block, every method
//!   body, every attribute/derive) into
//!   `impl ::vericl::ConfigIdentity for T { const CONFIG_HASH }`, one impl per
//!   declared struct/enum. The kernel folds that const into its recorded
//!   identity via `::vericl::combine_source_hash` — the same treatment
//!   `uses(...)` gives a helper and `reference = path` gives a declared
//!   reference.
//! - **Gates every body in the block** for host-callability and for hash
//!   coverage (see "The gates" below).
//!
//! **Hash granularity.** The hash is over `TokenStream::to_string()`, so
//! whitespace, line breaks and ordinary `//` comments do **not** move it, while
//! any token change — including a doc comment, which tokenizes to a `#[doc]`
//! attribute — does. This is deliberately the identical granularity as a
//! kernel's own `SOURCE_HASH` (over `ItemFn::to_token_stream().to_string()`),
//! so the two halves of a config kernel's identity are sensitive to exactly the
//! same class of edit.
//!
//! # The gates
//!
//! Each is *strict by construction*: an unrecognized form is rejected, never
//! accepted. The rejections are what let VeriCL claim that the tokens it hashed
//! are the tokens that determine the kernel.
//!
//! | # | Gate | Why |
//! |---|---|---|
//! | G1 | the block must declare at least one struct or enum | otherwise nothing gets a `ConfigIdentity` and the macro is a no-op that looks like a declaration |
//! | G2 | no `#[cube]` anywhere in the block (design R3) | a `#[cube]` config method runs as host Rust in the twin and as an expanded body on the device — the twin would call a different function than the kernel |
//! | G3 | no call to a [`crate::FLOAT_METHOD_REJECT`] name in any body (design R4) | those are `unexpanded!()` on host: a config method calling one panics in the twin at run time (measured), where every comparable VeriCL gate is a compile-time rejection |
//! | G4 | every call — **path form and method form** — must resolve into the block, to a primitive-qualified path, or to `core`/`std`/`alloc` (design risk 2) | a function defined *outside* the block is neither hashed nor gated — the `uses(...)` problem one level down |
//! | G5 | a declared config type may not be generic | `impl<S> ConfigIdentity for Cfg<S>` would give every instantiation the same hash, so a change in `S`'s own block would be invisible |
//! | G6 | every field/const type must be a scalar primitive, an array/tuple of those, or a type declared in **this** block (design §7) | a nested config declared in a *different* block would contribute its methods to the kernel's meaning without contributing to its hash |
//! | G7 | only `struct`/`enum`/`impl`/`trait`/`fn`/`const`/`use` items | a `static` (interior mutability), a `mod` (unhashed contents), or a `macro_rules!` re-opens the escape G4 closes |
//! | G8 | no macro invocation in a body | a macro's tokens are opaque to `syn`'s visitors, so `anything!(fma(a, b, c))` would evade G3 and G4 wholesale |
//! | G9 | every path *expression* must be a local, `self`, a name declared in the block, or a primitive-/`core`-/`std`-qualified path — and for a `Self::X`/`T::X` path, the **tail** must be an associated item this block declares | a bare `SOME_CONST`, or an associated `const` in an out-of-block impl, is a value the kernel's meaning depends on and the hash cannot see |
//! | G10 | no `core`/`std`/`alloc` path into an impure module ([`IMPURE_STD_MODULES`]), and no rand-like crate root | a config method is evaluated separately for the twin, for expansion and for IR extraction, so an environment-, clock- or randomness-dependent answer makes the three disagree |
//! | G11 | only `std` derives ([`STD_DERIVES`]) | a custom derive's *definition* decides what impls the type has, and the hash covers only the invocation — the unhashed-impl sibling of G7 |
//! | G12 | a `use` may not rebind an allowlisted path root, and may not be a glob | G4/G9 resolve roots BY NAME, so `use crate::evil as core;` would re-point the whole standard-library allowance at user code |
//! | G13 | every declared `fn` must return a primitive, an array/tuple of those, `Self`, a block-declared type, or nothing | the value crosses into the kernel body, where only the FIRST link of a chain rooted at a config parameter is exempt from the Float/Numeric name list |
//! | G14 | no `#[cfg_attr(…)]` anywhere in the block | every gate above reads attributes AS WRITTEN and rustc expands `cfg_attr` afterwards, so `#[cfg_attr(all(), cube)]` re-spells G2 and `#[cfg_attr(all(), derive(Evil))]` re-spells G11 (shared implementation in [`crate::decl_block`]) |
//!
//! G4's method-call half, G9's tail check, G10, G11 and G12 all come from the
//! round-10 adversarial review, which measured each of them as a live escape;
//! G12's derive-name half and G14 come from round 11, likewise measured.
//!
//! # The residual, precisely
//!
//! **Rust permits an inherent `impl` for a local type anywhere in the crate**,
//! so a *second* impl block written outside the `vericl::config!` invocation is
//! invisible to both the hash and every gate above. There is no macro-scope fix:
//! a `#[proc_macro]` sees only the tokens it is handed. This is the design's
//! pre-registered risk 3, and it is accepted with a loud backstop rather than
//! silently:
//!
//! - a non-host-callable body reached that way fails **loudly at run time** —
//!   the reference twin calls the host function and CubeCL's `unexpanded!()`
//!   panics ("Unexpanded Cube functions should not be called."), which the
//!   differential harness catches and reports as a divergence rather than
//!   swallowing (pinned by
//!   `crates/vericl-examples/tests/config_out_of_block_backstop.rs`);
//! - an *identity* drift reached that way still moves `ir_hash` whenever the
//!   affected value reaches the device, since the config's constants are folded
//!   into the IR (design §3);
//! - and G4/G9 mean the *in-block* half cannot silently call into it. **This
//!   was false until round 10** and is stated here as an enforced property
//!   rather than an assumption, because two measured escapes went through it:
//!   `self.combine()` in method syntax with `combine` in an out-of-block impl
//!   (two blocks with byte-identical tokens computed ×24 and ×11 with identical
//!   `CONFIG_HASH`es), and `Self::K` reading an out-of-block associated const
//!   (×24 vs ×15, same identity). G4 now resolves the *receiver's type* for a
//!   method call and G9 the *tail* of a qualified path, so reaching an
//!   out-of-block impl really does require the author to write the call on the
//!   kernel side, in tokens `SOURCE_HASH` already covers.
//!
//! One residual of a *different* kind, recorded here so it is never mistaken
//! for something the gates cover: `::vericl::ConfigIdentity` is a **public,
//! unsealed** trait, so `impl vericl::ConfigIdentity for MyOwnType { const
//! CONFIG_HASH: &'static str = "sha256:0000…"; }` for a type this macro never
//! saw is a **complete bypass of the mechanism** — no gate runs on the type, and
//! its recorded identity is a constant the author chose, which by construction
//! never goes stale. A `#[proc_macro]` cannot seal a trait, and VeriCL's
//! guarantee has never been "an author cannot lie to their own evidence file";
//! it is "an author who does not lie gets an identity that moves when the
//! meaning does". Every gate above is aimed at *accidental* drift. The same
//! bypass exists for `StructIdentity` and is stated in `cube_struct`'s residual
//! section, with a compiling acknowledgment test.
//!
//! Three narrower residuals of the same family, stated rather than papered
//! over: a trait `impl` for a declared type written outside the block (including
//! an operator-trait impl, so `self.a + self.b` on config-typed fields); the
//! bodies of `core`/`std` items G4/G9 allow (not user code, so not an identity
//! concern); and a method call on a receiver whose type the block cannot
//! resolve, where the [`STD_METHOD_ALLOWLIST`] name check is the whole gate
//! (sound because those names are `std` inherent methods, which win over any
//! user trait — see that constant's doc). See `docs/design-struct-comptime.md`
//! §13 risk 3.
//!
//! # The orphan-rule consequence
//!
//! Every config type must be *declared* inside a `vericl::config!` block, so a
//! **third-party** type cannot be one: `impl ConfigIdentity for TheirCfg` in
//! your crate is what Rust's orphan rule permits, but this macro never emits a
//! bare impl — it emits the declaration and the impl together, because a hash
//! over tokens you did not write would certify nothing. A comptime config from
//! another crate is therefore inexpressible in v1. The workaround is a
//! clean-room port: declare the type and the methods you need inside a
//! `vericl::config!` block of your own (that is exactly what the survey's
//! `tile_size_window_scale` does with `cubek-std`'s `TileSize`), which is more
//! work and is also the only version that means anything — the hash then covers
//! code that is actually in your repository.

use std::collections::{HashMap, HashSet};

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Expr, Ident, Item, Type};

use crate::FLOAT_METHOD_REJECT;
// The macro-agnostic half of the round-10 gate hardening, shared with
// `vericl::cube_struct!` rather than re-typed there (see `decl_block`'s module
// doc for exactly which gates are shared and which deliberately are not).
use crate::decl_block::{
    EXTERNAL_ROOTS, PRIMITIVE_TYPES, STD_DERIVES, block_hash, derive_paths, is_std_derive,
    render_path, std_derive_list,
};

/// G10 (purity): `core`/`std`/`alloc` **modules** a config body may not reach
/// into, by second path segment. G4/G9 admit the standard library because its
/// bodies are not user code and therefore not an identity concern — but "not an
/// identity concern" is only half the requirement. A config method also has to
/// be a *function of its source*: it is evaluated separately for the reference
/// twin, for kernel expansion and for IR extraction (see `is_pinnable_config_expr`
/// in the crate root), so a method whose answer depends on the environment makes
/// those three disagree and the recorded evidence describe a kernel that was
/// never run.
///
/// Measured (round-10 review, probe P2): `std::env::var("…")` in a config method
/// passed every one of G1–G9, and flipping the environment variable changed the
/// twin's answer from `2.0` to `4.0` with the kernel's recorded identity
/// unmoved.
///
/// The list is a denylist rather than an allowlist because the *pure*
/// computational surface of `core`/`std` is the large, open part
/// (`core::cmp`, `core::f32::consts`, `u32::max`, arithmetic, `Option`
/// combinators) and the impure part is small and enumerable. `mem` is on it not
/// for impurity but for target-dependence: `size_of::<usize>()` is 8 on the
/// host and 4 in a kernel's addressing regime, so a config method built on it
/// would not describe the kernel it configures.
const IMPURE_STD_MODULES: &[&str] = &[
    "env",
    "process",
    "fs",
    "io",
    "net",
    "thread",
    "time",
    "sync",
    "os",
    "sys",
    "backtrace",
    "panic",
    "hint",
    "intrinsics",
    "arch",
    "simd",
    "ptr",
    "cell",
    "mem",
    "collections",
    "rc",
    "task",
    "future",
    "pin",
    "ffi",
    "path",
    "random",
    "alloc",
];

/// Crate roots that are *never* admissible in a config body and whose rejection
/// deserves a purity diagnosis rather than the generic "not declared in this
/// block" one. (G4/G9 already reject every root outside [`EXTERNAL_ROOTS`]; this
/// only improves the message for the shapes an author is most likely to try.)
const IMPURE_CRATE_ROOTS: &[&str] =
    &["rand", "fastrand", "getrandom", "chrono", "uuid", "instant", "web_time"];

/// Method names a config body may call on a receiver whose type is **not** a
/// type this block declares — the `std` inherent surface.
///
/// **Why a name list is sound here, and only here.** Rust resolves an
/// *inherent* method before any trait method, regardless of which traits are in
/// scope — the same argument [`crate::FLOAT_METHOD_WHITELIST`] rests on. So for
/// a receiver of primitive type, a name on this list always resolves to the
/// `std` inherent method, and a user's `trait Boost { fn pow(…) }` cannot
/// silently win. A name that is *not* on this list is exactly the dangerous
/// case: `self.m.boost()` resolves to whatever extension trait happens to be in
/// scope — code the block neither hashes nor gates (round-10 review, probe P5a,
/// where an out-of-block `impl Boost for u32` turned `m` into `m * 7`).
///
/// Names that dispatch through a **user-extensible conversion trait** are
/// deliberately absent — `into`, `from`, `try_into`, `try_from`, `as_ref`,
/// `to_owned`, `borrow`, `deref`. Those are the one family where a user impl
/// for a primitive really can be reached without an ambiguity error, so they
/// stay out even though they are "std".
///
/// The derive-provided names (`clone`, `eq`, `cmp`, …) are trait methods, not
/// inherent ones, but a second trait declaring the same name makes the call
/// ambiguous at the *call site* (E0034) rather than silently rebinding it — a
/// loud failure, which is the standard this list is held to.
const STD_METHOD_ALLOWLIST: &[&str] = &[
    // --- integer inherent ---------------------------------------------------
    "pow", "checked_add", "checked_sub", "checked_mul", "checked_div", "checked_rem",
    "checked_neg", "checked_pow", "checked_shl", "checked_shr", "wrapping_add", "wrapping_sub",
    "wrapping_mul", "wrapping_div", "wrapping_rem", "wrapping_neg", "wrapping_pow",
    "wrapping_shl", "wrapping_shr", "saturating_add", "saturating_sub", "saturating_mul",
    "saturating_pow", "overflowing_add", "overflowing_sub", "overflowing_mul", "count_ones",
    "count_zeros", "leading_zeros", "trailing_zeros", "leading_ones", "trailing_ones",
    "rotate_left", "rotate_right", "swap_bytes", "reverse_bits", "to_be", "to_le", "div_euclid",
    "rem_euclid", "div_ceil", "div_floor", "next_multiple_of", "next_power_of_two",
    "is_power_of_two", "ilog2", "ilog10", "isqrt", "abs_diff", "unsigned_abs", "signum",
    "is_positive", "is_negative",
    // --- shared numeric -----------------------------------------------------
    "abs", "min", "max", "clamp",
    // --- float inherent -----------------------------------------------------
    "floor", "ceil", "round", "trunc", "fract", "copysign", "mul_add", "powi", "powf", "sqrt",
    "exp", "exp2", "ln", "log", "log2", "log10", "cbrt", "hypot", "sin", "cos", "tan", "asin",
    "acos", "atan", "atan2", "sin_cos", "exp_m1", "ln_1p", "sinh", "cosh", "tanh", "asinh",
    "acosh", "atanh", "recip", "to_degrees", "to_radians", "is_nan", "is_infinite", "is_finite",
    "is_normal", "is_sign_positive", "is_sign_negative", "to_bits", "total_cmp",
    // --- bool / char inherent ----------------------------------------------
    "then", "then_some", "is_ascii", "to_ascii_lowercase", "to_ascii_uppercase", "is_alphabetic",
    "is_numeric", "is_ascii_digit",
    // --- Option/Result combinators (their closures are gated in-block) ------
    "unwrap", "unwrap_or", "unwrap_or_else", "unwrap_or_default", "expect", "is_some", "is_none",
    "is_ok", "is_err", "map_or", "and_then", "or_else", "ok", "ok_or", "filter", "take",
    // --- std-derive-provided ------------------------------------------------
    "clone", "eq", "ne", "lt", "le", "gt", "ge", "cmp", "partial_cmp", "hash", "fmt", "default",
];

/// Everything the block declares, by name — the allowlist G4/G6/G9 resolve
/// against.
#[derive(Default)]
struct Declared {
    /// `struct`/`enum` names — each gets a `ConfigIdentity` impl.
    config_types: Vec<Ident>,
    /// `struct`/`enum`/`trait` names — the type namespace G6 accepts.
    types: HashSet<String>,
    /// Free `fn` and `const` names declared at block level — the value
    /// namespace G4/G9 accept as single-segment paths.
    values: HashSet<String>,
    /// Type (or trait) name -> every associated item name this block declares
    /// for it: inherent and in-block-trait `fn`/`const`s, enum variants, and the
    /// items an allowlisted `#[derive]` contributes. This is what G4/G9 check
    /// the TAIL of a `Self::X` / `T::X` path against, and what G4's method-call
    /// resolution checks a `.m()` against once the receiver's type is known.
    assoc: HashMap<String, HashSet<String>>,
    /// Type name -> (field name -> declared field type). Drives receiver-type
    /// resolution: `self.window.taps()` needs `window`'s type to know which
    /// type `taps` must be declared on.
    fields: HashMap<String, HashMap<String, Type>>,
    /// (type name, fn name) -> declared return type — the other half of receiver
    /// resolution, so a chain `self.window().taps()` resolves link by link.
    returns: HashMap<(String, String), Option<Type>>,
}

impl Declared {
    fn declares_assoc(&self, ty: &str, item: &str) -> bool {
        self.assoc.get(ty).is_some_and(|s| s.contains(item))
    }
}

pub(crate) fn expand(ts: TokenStream2) -> syn::Result<TokenStream2> {
    let file: syn::File = syn::parse2(ts.clone()).map_err(|e| {
        syn::Error::new(
            e.span(),
            format!(
                "vericl::config! takes a block of ordinary Rust items — the config type(s) and \
                 every one of their impl blocks: {e}"
            ),
        )
    })?;

    let mut errors: Vec<syn::Error> = Vec::new();
    let declared = collect_declared(&file, &mut errors);

    if declared.config_types.is_empty() && errors.is_empty() {
        return Err(syn::Error::new(
            ts.span(),
            "vericl::config! { … } must declare at least one struct or enum — it exists to give a \
             struct-typed #[comptime] parameter's type a `ConfigIdentity` (the hash of its whole \
             definition) and to gate its method bodies; a block with no type declaration does \
             neither, so writing one would record a guarantee that was never made",
        ));
    }

    check_field_types(&file, &declared, &mut errors);
    check_return_types(&file, &declared, &mut errors);
    check_no_cube_attr(&file, &mut errors);
    check_derives(&file, &mut errors);
    // G12 — shared with `vericl::cube_struct!`'s CS8 (round-10 probe P5b).
    crate::decl_block::check_use_items(
        &file,
        "vericl::config!",
        "G4/G9 resolve a call/read",
        &mut errors,
    );
    // G14 — shared with `vericl::cube_struct!`'s CS11 (round-11 review): a
    // `cfg_attr` makes the attribute set G2/G11 classify against and the one
    // rustc expands two different sets, which re-spells both gates at once.
    crate::decl_block::check_no_cfg_attr(&file, "vericl::config!", &mut errors);
    gate_bodies(&file, &declared, &mut errors);

    if let Some(combined) = errors.into_iter().reduce(|mut a, b| {
        a.combine(b);
        a
    }) {
        return Err(combined);
    }

    // The hash covers the WHOLE block: every declared type, every impl block,
    // every method body. This is the input `SOURCE_HASH` structurally cannot
    // have (design §5.1) and the reason this macro is an item macro rather than
    // an attribute (design §6.2).
    let hash = block_hash(&ts);

    let impls = declared.config_types.iter().map(|n| {
        quote! {
            impl ::vericl::ConfigIdentity for #n {
                const CONFIG_HASH: &'static str = #hash;
            }
        }
    });

    // Items re-emitted VERBATIM: `#ts`, not a re-serialization of the parsed
    // `syn::File`. A config is ordinary host Rust, and the `#[cube]` side must
    // see exactly what the author wrote.
    Ok(quote! {
        #ts
        #(#impls)*
    })
}

/// G1/G5/G7: walk the top-level items, collecting the declared names and
/// rejecting item kinds and generic config types.
///
/// Two passes: the type namespace must be complete before impl blocks are
/// resolved (an `impl Tr for T` may name a trait declared further down).
fn collect_declared(file: &syn::File, errors: &mut Vec<syn::Error>) -> Declared {
    let mut d = collect_types(file, errors);
    collect_assoc_items(file, &mut d);
    d
}

fn collect_types(file: &syn::File, errors: &mut Vec<syn::Error>) -> Declared {
    let mut d = Declared::default();
    for item in &file.items {
        match item {
            Item::Struct(s) => {
                reject_generics(&s.generics, &s.ident, errors);
                d.types.insert(s.ident.to_string());
                d.config_types.push(s.ident.clone());
                let name = s.ident.to_string();
                let map = d.fields.entry(name.clone()).or_default();
                for (i, f) in s.fields.iter().enumerate() {
                    let key = match &f.ident {
                        Some(id) => id.to_string(),
                        None => i.to_string(),
                    };
                    map.insert(key, f.ty.clone());
                }
                for (dv, items) in derived_assoc_items(&s.attrs) {
                    let _ = dv;
                    d.assoc.entry(name.clone()).or_default().extend(items);
                }
            }
            Item::Enum(e) => {
                reject_generics(&e.generics, &e.ident, errors);
                d.types.insert(e.ident.to_string());
                d.config_types.push(e.ident.clone());
                let name = e.ident.to_string();
                let assoc = d.assoc.entry(name.clone()).or_default();
                for v in &e.variants {
                    assoc.insert(v.ident.to_string());
                }
                for (dv, items) in derived_assoc_items(&e.attrs) {
                    let _ = dv;
                    d.assoc.entry(name.clone()).or_default().extend(items);
                }
            }
            Item::Trait(t) => {
                d.types.insert(t.ident.to_string());
                let assoc = d.assoc.entry(t.ident.to_string()).or_default();
                for it in &t.items {
                    match it {
                        syn::TraitItem::Fn(f) => {
                            assoc.insert(f.sig.ident.to_string());
                        }
                        syn::TraitItem::Const(c) => {
                            assoc.insert(c.ident.to_string());
                        }
                        syn::TraitItem::Type(ty) => {
                            assoc.insert(ty.ident.to_string());
                        }
                        _ => {}
                    }
                }
            }
            Item::Fn(f) => {
                d.values.insert(f.sig.ident.to_string());
            }
            Item::Const(c) => {
                d.values.insert(c.ident.to_string());
            }
            Item::Impl(_) | Item::Use(_) => {}
            // Targeted, because this is the ecosystem's dominant declaration
            // spelling: `cubek-std/src/size.rs` generates `TileSize`,
            // `PartitionSize`, `StageSize` … from one `define_3d_size_base!`
            // (design §4.1). The invocation's tokens are in the block, but the
            // macro's *definition* is not, so hashing them would not cover what
            // the type is — and none of the body gates can walk an unexpanded
            // macro.
            Item::Macro(m) => errors.push(syn::Error::new(
                m.mac.path.span(),
                "a macro invocation cannot declare a config type inside vericl::config! — the \
                 invocation's tokens are hashed but the MACRO's definition is not, so an edit to \
                 the macro would change what the config type is while leaving CONFIG_HASH (and \
                 every kernel's recorded identity) unmoved, and none of the method-body gates can \
                 walk an unexpanded macro. Write the expansion out inside the block (this is the \
                 real cost of the guarantee for macro-generated config families such as CubeCL's \
                 own `define_3d_size_base!`), or keep the macro-generated type outside vericl and \
                 pin the values it would compute with instantiate(...)",
            )),
            other => errors.push(syn::Error::new(
                other.span(),
                "only `struct`, `enum`, `impl`, `trait`, `fn`, `const` and `use` items are \
                 allowed inside vericl::config! — every other item kind either hides state the \
                 config hash cannot see (`static`, and in particular interior mutability) or \
                 hides tokens the method-body gates cannot walk (`mod`, `macro_rules!`, a type \
                 alias to an undeclared type); declare it outside the block and, if a config \
                 method needs it, move that function INTO the block",
            )),
        }
    }
    d
}

/// The type name an `impl` block's self type names, when it is a plain path
/// this block could have declared (`impl TileCfg`, `impl Trait for TileCfg`).
fn impl_self_ty_name(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(tp) if tp.qself.is_none() => {
            tp.path.segments.last().map(|s| s.ident.to_string())
        }
        Type::Paren(p) => impl_self_ty_name(&p.elem),
        Type::Reference(r) => impl_self_ty_name(&r.elem),
        _ => None,
    }
}

/// Pass 2: every associated item the block's `impl` blocks declare, per type —
/// the resolution table G4's method check and G9's path-tail check consult.
fn collect_assoc_items(file: &syn::File, d: &mut Declared) {
    for item in &file.items {
        let Item::Impl(ii) = item else { continue };
        let Some(ty_name) = impl_self_ty_name(&ii.self_ty) else { continue };
        // `impl Tr for T` where `Tr` is declared here: the trait's own items
        // (including defaults) are reachable on `T` too.
        if let Some((_, trait_path, _)) = &ii.trait_ {
            if let Some(tr) = trait_path.segments.last() {
                let tr = tr.ident.to_string();
                if let Some(items) = d.assoc.get(&tr).cloned() {
                    d.assoc.entry(ty_name.clone()).or_default().extend(items);
                }
            }
        }
        for it in &ii.items {
            match it {
                syn::ImplItem::Fn(f) => {
                    let m = f.sig.ident.to_string();
                    d.assoc.entry(ty_name.clone()).or_default().insert(m.clone());
                    let ret = match &f.sig.output {
                        syn::ReturnType::Default => None,
                        syn::ReturnType::Type(_, t) => Some((**t).clone()),
                    };
                    d.returns.insert((ty_name.clone(), m), ret);
                }
                syn::ImplItem::Const(c) => {
                    d.assoc.entry(ty_name.clone()).or_default().insert(c.ident.to_string());
                }
                syn::ImplItem::Type(t) => {
                    d.assoc.entry(ty_name.clone()).or_default().insert(t.ident.to_string());
                }
                _ => {}
            }
        }
    }
}

/// The associated item names an allowlisted `#[derive(...)]` contributes, so a
/// derived `TileCfg::default()` / `self.clone()` resolves against the block.
/// Unknown derives are *not* filtered here — [`check_derives`] rejects them.
fn derived_assoc_items(attrs: &[syn::Attribute]) -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    for path in derive_paths(attrs) {
        let name = render_path(&path);
        if let Some((_, items)) = STD_DERIVES.iter().find(|(n, _)| *n == name) {
            out.push((name, items.iter().map(|s| s.to_string()).collect()));
        }
    }
    out
}

/// G11 (custom derives): a `#[derive]` outside the `std` set is the unhashed-impl
/// sibling of G7's macro rejection. A derive macro is a `proc_macro_derive`: the
/// *invocation* (`#[derive(Foo)]`) is in the tokens `CONFIG_HASH` covers, but
/// `Foo`'s **definition** — which decides what impls, methods and associated
/// consts the config type actually has — is not. An edit there changes what the
/// kernel computes with `CONFIG_HASH` unmoved, and none of the body gates can
/// walk code that does not exist until rustc expands it.
///
/// The `std` derives are admitted because their expansion is fixed by the
/// language, contributes no user-authored body, and is already covered by
/// [`STD_DERIVES`]' associated-item table.
fn check_derives(file: &syn::File, errors: &mut Vec<syn::Error>) {
    let mut check = |attrs: &[syn::Attribute]| {
        for path in derive_paths(attrs) {
            let name = render_path(&path);
            if is_std_derive(&name) {
                continue;
            }
            errors.push(syn::Error::new(
                path.span(),
                format!(
                    "`#[derive({name})]` inside a vericl::config! block is outside the vericl v1 \
                     struct-comptime subset — vericl::config! hashes the block's tokens, and a \
                     custom derive's tokens are only its INVOCATION: the derive macro's own \
                     definition decides what impls and associated items the config type has, so \
                     an edit there would change what the kernel computes while leaving \
                     CONFIG_HASH (and every kernel's recorded identity) unmoved, and none of the \
                     method-body gates can walk code that does not exist until rustc expands it. \
                     This is the same reason a macro invocation cannot declare a config type \
                     (gate G7). Allowed derives: {}. If the type needs a derive from another \
                     crate (`CubeType`, `serde::Serialize`, …), keep that type outside vericl and \
                     pin the values it would compute with instantiate(...)",
                    std_derive_list()
                ),
            ));
        }
    };
    for item in &file.items {
        match item {
            Item::Struct(s) => check(&s.attrs),
            Item::Enum(e) => check(&e.attrs),
            Item::Trait(t) => check(&t.attrs),
            Item::Fn(f) => check(&f.attrs),
            Item::Const(c) => check(&c.attrs),
            Item::Impl(i) => check(&i.attrs),
            _ => {}
        }
    }
}

fn reject_generics(generics: &syn::Generics, name: &Ident, errors: &mut Vec<syn::Error>) {
    if generics.params.is_empty() {
        return;
    }
    errors.push(syn::Error::new(
        generics.span(),
        format!(
            "a generic config type (`{name}<…>`) is outside the vericl v1 struct-comptime subset \
             — one `vericl::config!` block hashes to one CONFIG_HASH, so every instantiation of \
             `{name}<…>` would carry the SAME identity and a change inside the type argument's \
             own block would be invisible to the kernel's recorded identity; declare a concrete \
             config type instead (the kernel's own generics are unaffected: \
             `instantiate(C = MyCfg, cfg = MyCfg {{ … }})` pins both faces and folds `MyCfg`'s \
             hash)"
        ),
    ));
}

/// G6: a field's (or associated const's) type must be a scalar primitive, an
/// array/tuple of allowed types, or a type declared in **this** block.
///
/// This is what turns the design's §7 "a `vericl::config!` block must declare
/// every config type reachable from a kernel's comptime param types" from a
/// documented rule into an enforced one: `StageCfg { tile: TileCfg }` with
/// `TileCfg` declared in a *sibling* block would fold only `StageCfg`'s hash,
/// leaving `TileCfg`'s method bodies outside the kernel's identity entirely.
fn check_field_types(file: &syn::File, declared: &Declared, errors: &mut Vec<syn::Error>) {
    let mut check = |ty: &Type, owner: &str| {
        if !is_allowed_field_type(ty, declared) {
            errors.push(syn::Error::new(
                ty.span(),
                format!(
                    "`{owner}` has a field/const of a type vericl::config! cannot account for — \
                     a config type's fields must be scalar primitives (u8..u128, i8..i128, \
                     usize/isize, bool, char, f32/f64), arrays/tuples of those, or ANOTHER type \
                     declared in THIS SAME vericl::config! block; a config type declared in a \
                     different block contributes its method bodies to the kernel's meaning but \
                     not to its CONFIG_HASH, which is the exact identity hole this macro exists \
                     to close (docs/design-struct-comptime.md §7) — move the nested type into \
                     this block"
                ),
            ));
        }
    };
    for item in &file.items {
        match item {
            Item::Struct(s) => {
                for f in s.fields.iter() {
                    check(&f.ty, &s.ident.to_string());
                }
            }
            Item::Enum(e) => {
                for v in &e.variants {
                    for f in v.fields.iter() {
                        check(&f.ty, &format!("{}::{}", e.ident, v.ident));
                    }
                }
            }
            Item::Const(c) => check(&c.ty, &c.ident.to_string()),
            _ => {}
        }
    }
}

/// G13 (return types): every `fn` the block declares must return a scalar
/// primitive, an array/tuple of those, a type this block declares, `Self`, or
/// nothing.
///
/// This is what makes the kernel-side chain rule compose. `FloatMethodCheck`
/// exempts a method call whose receiver is *directly* a config `#[comptime]`
/// parameter (its host-callability is gated here instead) and checks every
/// later link of a chain normally — measured as necessary in the round-10
/// review, where `cfg.gainf().erf()` reached the `unexpanded!()` `erf` because
/// the whole chain was exempted by its root. For that split to be meaningful,
/// the value a config method hands back must be something the kernel side can
/// reason about: a primitive (whose methods the name list covers) or another
/// config type declared in this same block (whose methods are gated here).
fn check_return_types(file: &syn::File, declared: &Declared, errors: &mut Vec<syn::Error>) {
    let mut check = |sig: &syn::Signature, owner: &str| {
        let syn::ReturnType::Type(_, ty) = &sig.output else { return };
        if is_allowed_return_type(ty, declared) {
            return;
        }
        errors.push(syn::Error::new(
            ty.span(),
            format!(
                "`{owner}` returns a type vericl::config! cannot account for — a config fn must \
                 return a scalar primitive (u8..u128, i8..i128, usize/isize, bool, char, f32/f64), \
                 an array/tuple of those, `Self`, or another type declared in THIS SAME \
                 vericl::config! block. The value crosses into the kernel's body, where a method \
                 called on it is checked against the Float/Numeric reject list by name (a config \
                 parameter is exempt only for the FIRST link of a chain); a return type whose \
                 methods live outside this block would be neither gated there nor hashed here"
            ),
        ));
    };
    for item in &file.items {
        match item {
            Item::Fn(f) => check(&f.sig, &f.sig.ident.to_string()),
            Item::Impl(ii) => {
                let owner = impl_self_ty_name(&ii.self_ty).unwrap_or_else(|| "impl".to_string());
                for it in &ii.items {
                    if let syn::ImplItem::Fn(f) = it {
                        check(&f.sig, &format!("{owner}::{}", f.sig.ident));
                    }
                }
            }
            Item::Trait(t) => {
                for it in &t.items {
                    if let syn::TraitItem::Fn(f) = it {
                        check(&f.sig, &format!("{}::{}", t.ident, f.sig.ident));
                    }
                }
            }
            _ => {}
        }
    }
}

fn is_allowed_return_type(ty: &Type, declared: &Declared) -> bool {
    match ty {
        Type::Tuple(t) if t.elems.is_empty() => true,
        Type::Path(tp)
            if tp.qself.is_none()
                && tp.path.segments.len() == 1
                && tp.path.segments[0].ident == "Self" =>
        {
            true
        }
        _ => is_allowed_field_type(ty, declared),
    }
}

fn is_allowed_field_type(ty: &Type, declared: &Declared) -> bool {
    match ty {
        Type::Path(tp) if tp.qself.is_none() => {
            let Some(last) = tp.path.segments.last() else { return false };
            // No generic arguments: `Option<Foo>`/`Vec<Foo>` would need the
            // argument's own definition hashed, which is the same hole one
            // level down.
            if !matches!(last.arguments, syn::PathArguments::None) {
                return false;
            }
            let name = last.ident.to_string();
            PRIMITIVE_TYPES.contains(&name.as_str()) || declared.types.contains(&name)
        }
        Type::Array(a) => is_allowed_field_type(&a.elem, declared),
        Type::Tuple(t) => t.elems.iter().all(|e| is_allowed_field_type(e, declared)),
        Type::Paren(p) => is_allowed_field_type(&p.elem, declared),
        _ => false,
    }
}

/// G2 (design R3): reject `#[cube]` anywhere in the block.
fn check_no_cube_attr(file: &syn::File, errors: &mut Vec<syn::Error>) {
    struct CubeAttrCheck<'a> {
        errors: &'a mut Vec<syn::Error>,
    }
    impl<'ast> Visit<'ast> for CubeAttrCheck<'_> {
        fn visit_attribute(&mut self, i: &'ast syn::Attribute) {
            if i.path().segments.last().is_some_and(|s| s.ident == "cube") {
                self.errors.push(syn::Error::new(
                    i.span(),
                    "a `#[cube]` attribute inside a vericl::config! block is outside the vericl \
                     v0 subset — a comptime config's methods run in the reference twin as \
                     ordinary host Rust, so the twin would call the host body while the device \
                     gets the expanded one, and the two are only accidentally the same function; \
                     keep config methods plain (the CubeCL ecosystem's own config types are all \
                     plain Rust — 132 methods surveyed, 0 annotated `#[cube]`, \
                     docs/design-struct-comptime.md §4.2)",
                ));
            }
            syn::visit::visit_attribute(self, i);
        }
    }
    CubeAttrCheck { errors }.visit_file(file);
}

/// G3/G4/G8/G9/G10: gate every body the block declares.
fn gate_bodies(file: &syn::File, declared: &Declared, errors: &mut Vec<syn::Error>) {
    for item in &file.items {
        match item {
            Item::Fn(f) => gate_fn(&f.sig, &f.block, None, declared, errors),
            Item::Const(c) => gate_expr(&c.expr, &HashSet::new(), None, declared, errors),
            Item::Impl(ii) => {
                let self_ty = impl_self_ty_name(&ii.self_ty);
                for it in &ii.items {
                    match it {
                        syn::ImplItem::Fn(f) => {
                            gate_fn(&f.sig, &f.block, self_ty.clone(), declared, errors)
                        }
                        syn::ImplItem::Const(c) => {
                            gate_expr(&c.expr, &HashSet::new(), self_ty.clone(), declared, errors)
                        }
                        _ => {}
                    }
                }
            }
            Item::Trait(t) => {
                let self_ty = Some(t.ident.to_string());
                for it in &t.items {
                    match it {
                        syn::TraitItem::Fn(f) => {
                            if let Some(b) = &f.default {
                                gate_fn(&f.sig, b, self_ty.clone(), declared, errors);
                            }
                        }
                        syn::TraitItem::Const(c) => {
                            if let Some((_, e)) = &c.default {
                                gate_expr(e, &HashSet::new(), self_ty.clone(), declared, errors);
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}

fn gate_fn(
    sig: &syn::Signature,
    block: &syn::Block,
    self_ty: Option<String>,
    declared: &Declared,
    errors: &mut Vec<syn::Error>,
) {
    // Locals the body binds (deliberately over-inclusive: `crate::collect_locals`
    // collects every `PatIdent` in the block — `let`, `for`, closure parameters,
    // `match` arm bindings, nested `fn` parameters — with no scope tracking).
    // Over-inclusive is the safe direction for G9: it can only *fail to reject*
    // a bare read that rustc would then have to resolve itself, never accept a
    // call G4 would have caught (G4 does not consult locals except for the
    // closure-call case, where a local really is the only thing a bare
    // single-segment callee can be).
    let mut locals = crate::collect_locals(block, &[]);
    let mut local_tys: HashMap<String, RTy> = HashMap::new();
    for arg in &sig.inputs {
        match arg {
            syn::FnArg::Receiver(_) => {
                locals.insert("self".to_string());
            }
            syn::FnArg::Typed(pt) => {
                if let syn::Pat::Ident(pi) = pt.pat.as_ref() {
                    let name = pi.ident.to_string();
                    locals.insert(name.clone());
                    local_tys.insert(name, classify_ty(&pt.ty, self_ty.as_deref(), declared));
                }
            }
        }
    }
    let mut gate = BodyGate { locals: &locals, declared, errors, self_ty, local_tys };
    gate.visit_block(block);
}

fn gate_expr(
    expr: &Expr,
    locals: &HashSet<String>,
    self_ty: Option<String>,
    declared: &Declared,
    errors: &mut Vec<syn::Error>,
) {
    let mut gate =
        BodyGate { locals, declared, errors, self_ty, local_tys: HashMap::new() };
    gate.visit_expr(expr);
}

/// What a config body's expression is known to evaluate to, as far as the block
/// itself can decide it. Deliberately three-valued: `Unknown` is the honest
/// answer for anything the block does not declare, and it makes the method gate
/// fall back to the `std` name allowlist rather than guess.
#[derive(Clone, PartialEq, Eq, Debug)]
enum RTy {
    /// A scalar primitive (`u32`, `f32`, `bool`, …).
    Prim,
    /// A type this block declares.
    Decl(String),
    /// Not resolvable from the block's tokens alone.
    Unknown,
}

/// Classify a written-out type into [`RTy`], resolving `Self` to the enclosing
/// impl's type.
fn classify_ty(ty: &Type, self_ty: Option<&str>, declared: &Declared) -> RTy {
    match ty {
        Type::Path(tp) if tp.qself.is_none() => {
            let Some(last) = tp.path.segments.last() else { return RTy::Unknown };
            if !matches!(last.arguments, syn::PathArguments::None) {
                return RTy::Unknown;
            }
            let name = last.ident.to_string();
            if name == "Self" {
                return match self_ty {
                    Some(t) => RTy::Decl(t.to_string()),
                    None => RTy::Unknown,
                };
            }
            if PRIMITIVE_TYPES.contains(&name.as_str()) {
                RTy::Prim
            } else if declared.types.contains(&name) {
                RTy::Decl(name)
            } else {
                RTy::Unknown
            }
        }
        Type::Reference(r) => classify_ty(&r.elem, self_ty, declared),
        Type::Paren(p) => classify_ty(&p.elem, self_ty, declared),
        _ => RTy::Unknown,
    }
}

struct BodyGate<'a> {
    locals: &'a HashSet<String>,
    declared: &'a Declared,
    errors: &'a mut Vec<syn::Error>,
    /// The enclosing `impl`/`trait`'s type, for `self` and `Self::…`.
    self_ty: Option<String>,
    /// Types of the body's `let` bindings and parameters, filled in **source
    /// order** as the visitor walks (so `let base = self.window(); base.taps()`
    /// resolves). Over-approximate in the same direction as `locals`: a binding
    /// the block cannot type is [`RTy::Unknown`], which only ever makes the
    /// method gate stricter.
    local_tys: HashMap<String, RTy>,
}

impl BodyGate<'_> {
    /// G3 (design R4): a call to a name on the closed `FLOAT_METHOD_REJECT`
    /// list. The kernel-body walker rejects these in the kernel's own tokens;
    /// this is the same list applied to the bodies that walker cannot see.
    /// Returns `true` when it rejected, so a call that fails G3 is not also
    /// reported by G4 (both are true of `fma`; one diagnosis is enough, and the
    /// host-callability one is the more actionable).
    fn check_reject_list(&mut self, name: &Ident) -> bool {
        let s = name.to_string();
        if FLOAT_METHOD_REJECT.contains(&s.as_str()) {
            self.errors.push(syn::Error::new(
                name.span(),
                format!(
                    "config method body calls `{s}`, which is not verified host-callable — a \
                     comptime config's methods run in the reference twin as ordinary host Rust, \
                     so every call in them must be host-callable; `{s}` is `unexpanded!()` on \
                     host (it panics), so this would fail at run time as a twin panic instead of \
                     here. Use the vericl host shim (`vericl::host_shims::{s}`, where one exists) \
                     or compute the value on the host before pinning it"
                ),
            ));
            return true;
        }
        false
    }

    /// The type a `Self::X` / `T::X` path is qualified by, when this block
    /// declares it. `None` means the root is not a block-declared type (a
    /// primitive, `core`/`std`, or something G4/G9 will reject on its own).
    fn qualified_owner(&self, first: &str) -> Option<String> {
        if first == "Self" {
            return self.self_ty.clone();
        }
        if self.declared.types.contains(first) {
            return Some(first.to_string());
        }
        None
    }

    /// G9, the qualified half: `Self::K` / `TileCfg::K` names an **associated
    /// item**, and the block must declare that item, not merely the type.
    ///
    /// Measured (round-10 review, probe P7): checking only the ROOT let
    /// `self.m * Self::K` read an associated `const K` from an impl written
    /// outside the block — the same escape G4 closes for free functions,
    /// reached through a path instead of a call. Two blocks whose in-block
    /// tokens were byte-identical, with out-of-block `K = 8` and `K = 5`,
    /// computed ×24 and ×15 with identical `CONFIG_HASH`es and identical
    /// recorded kernel identities.
    ///
    /// Returns `true` if it handled (and possibly rejected) the path.
    fn check_qualified_tail(&mut self, path: &syn::Path, span: proc_macro2::Span) -> bool {
        let Some(first) = path.segments.first() else { return false };
        let first_s = first.ident.to_string();
        if first_s == "Self" && self.self_ty.is_none() {
            self.errors.push(syn::Error::new(
                span,
                "`Self::…` outside an impl block cannot be resolved by vericl::config!, so its \
                 associated item cannot be checked against what this block declares",
            ));
            return true;
        }
        let Some(owner) = self.qualified_owner(&first_s) else { return false };
        if path.segments.len() < 2 {
            return false;
        }
        let tail = path.segments[1].ident.to_string();
        if self.declared.declares_assoc(&owner, &tail) {
            return true;
        }
        self.errors.push(syn::Error::new(
            span,
            format!(
                "config method body names `{}`, but this vericl::config! block declares no \
                 associated item `{tail}` for `{owner}` — Rust lets an inherent `impl` for a local \
                 type live anywhere in the crate, so `{owner}::{tail}` would resolve to an impl \
                 whose tokens CONFIG_HASH never saw and whose body no gate ever walked (editing it \
                 would change what the kernel computes with the kernel's recorded identity \
                 unmoved). Declare `{tail}` inside this block",
                render_path(path)
            ),
        ));
        true
    }

    /// G10 (purity): the `core`/`std`/`alloc` modules a config body may not
    /// reach into. Returns `true` when it rejected.
    fn check_std_purity(&mut self, path: &syn::Path, span: proc_macro2::Span) -> bool {
        let Some(first) = path.segments.first() else { return false };
        let first_s = first.ident.to_string();
        if IMPURE_CRATE_ROOTS.contains(&first_s.as_str()) {
            self.errors.push(syn::Error::new(
                span,
                format!(
                    "config method body reaches `{}` — a config method must be a function of the \
                     tokens this block hashes and nothing else. Its value is computed separately \
                     for the reference twin, for kernel expansion and for IR extraction, so an \
                     environment-, clock- or randomness-dependent answer makes the three disagree \
                     and the recorded evidence describe a kernel that was never run. Compute the \
                     value on the host and pin it with instantiate(...)",
                    render_path(path)
                ),
            ));
            return true;
        }
        if !EXTERNAL_ROOTS.contains(&first_s.as_str()) {
            return false;
        }
        let Some(second) = path.segments.get(1) else { return false };
        let second_s = second.ident.to_string();
        if !IMPURE_STD_MODULES.contains(&second_s.as_str()) {
            return false;
        }
        let why = match second_s.as_str() {
            "mem" => "reads a target-dependent quantity (`size_of::<usize>()` is 8 on the host and \
                      4 in a kernel's addressing regime), so it would not describe the kernel it \
                      configures",
            _ => "reads state outside the block's tokens",
        };
        self.errors.push(syn::Error::new(
            span,
            format!(
                "config method body reaches `{}`, which {why} — `{first_s}::{second_s}` is on \
                 vericl::config!'s impure-module denylist. A config method's value is computed \
                 separately for the reference twin, for kernel expansion and for IR extraction, so \
                 anything that can answer differently between them makes the recorded evidence \
                 describe a kernel that was never run (measured: a `std::env::var` in a config \
                 method changed the twin's answer from 2.0 to 4.0 with the kernel's recorded \
                 identity unmoved). The pure computational surface of `core`/`std` — `core::cmp`, \
                 `core::f32::consts`, a primitive's associated functions — stays allowed. Compute \
                 the value on the host and pin it with instantiate(...)",
                render_path(path)
            ),
        ));
        true
    }

    /// G4: the callee must resolve into this block, to `Self`, to a
    /// primitive-qualified associated function, or to `core`/`std`/`alloc`.
    fn check_callee_path(&mut self, path: &syn::Path, qself: Option<&syn::QSelf>) {
        if let Some(q) = qself {
            // `<TileCfg as Trait>::f(...)` — the qualifying type carries the
            // resolution, so gate that instead of the (trait-side) path root.
            if !is_allowed_field_type(&q.ty, self.declared) {
                self.reject_callee(path.span(), &render_path(path));
            }
            return;
        }
        let Some(first) = path.segments.first() else { return };
        let first_s = first.ident.to_string();
        if path.segments.len() == 1 {
            // A bare `f(...)`: a free fn declared in this block, a
            // tuple-struct/tuple-variant constructor of a type this block
            // declares (`TileCfg(3, 8)`, `Self(3)`), or a local (a closure).
            // Nothing else can be resolved from here, and guessing is exactly
            // the class of silent hole this gate closes.
            let ok = self.declared.values.contains(&first_s)
                || self.declared.types.contains(&first_s)
                || self.locals.contains(&first_s)
                || first_s == "Self";
            if !ok {
                self.reject_callee(path.span(), &first_s);
            }
            return;
        }
        if self.check_std_purity(path, path.span()) {
            return;
        }
        // `Self::f(..)` / `TileCfg::f(..)`: the TAIL has to be declared here too
        // — the root alone says nothing about where the callee's body lives.
        if self.check_qualified_tail(path, path.span()) {
            return;
        }
        let ok = self.declared.values.contains(&first_s)
            || EXTERNAL_ROOTS.contains(&first_s.as_str())
            || PRIMITIVE_TYPES.contains(&first_s.as_str());
        if !ok {
            self.reject_callee(path.span(), &render_path(path));
        }
    }

    fn reject_callee(&mut self, span: proc_macro2::Span, name: &str) {
        self.errors.push(syn::Error::new(
            span,
            format!(
                "config method body calls `{name}`, which is not declared inside this \
                 vericl::config! block — vericl::config! hashes the block's tokens, so a function \
                 defined outside it contributes to the kernel's meaning without contributing to \
                 its CONFIG_HASH (editing that function would leave the kernel's recorded \
                 identity unmoved), and its body is never gated for host-callability. Move the \
                 function INTO this vericl::config! block, or compute the value on the host and \
                 pin it with instantiate(...). Calls to the pure part of `core`/`std`/`alloc`, to \
                 a primitive's associated functions (`u32::max`), and to anything this block \
                 declares are allowed"
            ),
        ));
    }

    /// G4, the method-call half: `recv.m(...)`.
    ///
    /// Method syntax was the escape the round-10 review found (probe P1): a
    /// config method calling `self.combine()`, with `combine` declared in an
    /// impl written *outside* the block, passed every gate — two blocks whose
    /// in-block tokens were byte-identical computed ×24 and ×11 with identical
    /// recorded kernel identities. So a method call is admitted on exactly two
    /// grounds:
    ///
    /// 1. the receiver resolves to a type **this block declares**, and the block
    ///    declares that method for it (the sound case: hashed and gated); or
    /// 2. the method name is on [`STD_METHOD_ALLOWLIST`] — the `std` inherent
    ///    surface, which resolves ahead of any user trait (see that constant's
    ///    doc for why a name check is sound *there* and nowhere else).
    ///
    /// Anything else is rejected, including a name reached through a user
    /// extension trait on a primitive (probe P5a's `impl Boost for u32`).
    fn check_method_call(&mut self, i: &syn::ExprMethodCall) {
        let name = i.method.to_string();
        let recv = self.resolve_expr(i.receiver.as_ref());
        // Resolved into the block: a config method, whose own body this same
        // gate walks. This runs BEFORE the `FLOAT_METHOD_REJECT` name check for
        // the same reason `FloatMethodCheck` is receiver-aware on the kernel
        // side (design R6): `self.window().dot()` where the block declares
        // `WindowCfg::dot` is a plain host method, not cubecl's `unexpanded!()`
        // `Float::dot`, and resolving the receiver is what lets the two be told
        // apart here.
        if let RTy::Decl(t) = &recv {
            if self.declared.declares_assoc(t, &name) {
                return;
            }
        }
        if self.check_reject_list(&i.method) {
            return;
        }
        if STD_METHOD_ALLOWLIST.contains(&name.as_str()) {
            return;
        }
        let where_ = match &recv {
            RTy::Decl(t) => format!(
                "this vericl::config! block declares no method `{name}` for `{t}` (the receiver's \
                 type), so it would resolve to an `impl {t}` written outside the block"
            ),
            RTy::Prim => format!(
                "`{name}` is not on vericl::config!'s `std` inherent-method allowlist, so on a \
                 primitive receiver it can only come from a user extension trait"
            ),
            RTy::Unknown => format!(
                "vericl::config! cannot resolve this receiver's type from the block's tokens, and \
                 `{name}` is neither declared in this block nor on the `std` inherent-method \
                 allowlist"
            ),
        };
        self.errors.push(syn::Error::new(
            i.method.span(),
            format!(
                "config method body calls `.{name}(…)`, which vericl::config! cannot account for \
                 — {where_}. A method reached that way contributes to what the kernel computes \
                 without contributing to CONFIG_HASH (editing it would leave the kernel's recorded \
                 identity unmoved) and its body is never gated for host-callability. Declare the \
                 method INSIDE this vericl::config! block, or compute the value on the host and \
                 pin it with instantiate(...)"
            ),
        ));
    }

    /// Best-effort receiver typing, from the block's tokens alone. `Unknown` is
    /// always a safe answer: it only makes [`Self::check_method_call`] fall back
    /// to the `std` name allowlist.
    fn resolve_expr(&self, e: &Expr) -> RTy {
        match crate::peel_paren(e) {
            Expr::Path(p) if p.qself.is_none() => {
                let segs = &p.path.segments;
                if segs.len() == 1 {
                    let n = segs[0].ident.to_string();
                    if n == "self" || n == "Self" {
                        return match &self.self_ty {
                            Some(t) => RTy::Decl(t.clone()),
                            None => RTy::Unknown,
                        };
                    }
                    return self.local_tys.get(&n).cloned().unwrap_or(RTy::Unknown);
                }
                // `Mode::Single`, `Self::VARIANT`, `u32::MAX` …
                let first = segs[0].ident.to_string();
                if PRIMITIVE_TYPES.contains(&first.as_str()) {
                    return RTy::Prim;
                }
                match self.qualified_owner(&first) {
                    Some(t) => RTy::Decl(t),
                    None => RTy::Unknown,
                }
            }
            Expr::Field(f) => {
                let RTy::Decl(t) = self.resolve_expr(f.base.as_ref()) else { return RTy::Unknown };
                let key = match &f.member {
                    syn::Member::Named(id) => id.to_string(),
                    syn::Member::Unnamed(idx) => idx.index.to_string(),
                };
                match self.declared.fields.get(&t).and_then(|m| m.get(&key)) {
                    Some(ty) => classify_ty(ty, self.self_ty.as_deref(), self.declared),
                    None => RTy::Unknown,
                }
            }
            Expr::MethodCall(mc) => {
                let RTy::Decl(t) = self.resolve_expr(mc.receiver.as_ref()) else {
                    return RTy::Unknown;
                };
                self.resolve_return(&t, &mc.method.to_string())
            }
            Expr::Call(c) => {
                let Expr::Path(p) = crate::peel_paren(c.func.as_ref()) else { return RTy::Unknown };
                let segs = &p.path.segments;
                if segs.len() == 1 {
                    // A tuple-struct / tuple-variant constructor.
                    let n = segs[0].ident.to_string();
                    if n == "Self" {
                        return match &self.self_ty {
                            Some(t) => RTy::Decl(t.clone()),
                            None => RTy::Unknown,
                        };
                    }
                    if self.declared.types.contains(&n) {
                        return RTy::Decl(n);
                    }
                    return RTy::Unknown;
                }
                let first = segs[0].ident.to_string();
                if PRIMITIVE_TYPES.contains(&first.as_str()) {
                    return RTy::Prim;
                }
                match self.qualified_owner(&first) {
                    Some(t) => self.resolve_return(&t, &segs[1].ident.to_string()),
                    None => RTy::Unknown,
                }
            }
            Expr::Struct(s) => {
                let n = s.path.segments.last().map(|x| x.ident.to_string()).unwrap_or_default();
                if n == "Self" {
                    return match &self.self_ty {
                        Some(t) => RTy::Decl(t.clone()),
                        None => RTy::Unknown,
                    };
                }
                if self.declared.types.contains(&n) { RTy::Decl(n) } else { RTy::Unknown }
            }
            Expr::Cast(c) => classify_ty(&c.ty, self.self_ty.as_deref(), self.declared),
            Expr::Lit(l) => match l.lit {
                syn::Lit::Str(_) | syn::Lit::ByteStr(_) | syn::Lit::CStr(_) => RTy::Unknown,
                _ => RTy::Prim,
            },
            Expr::Binary(b) => {
                // Comparison/logical operators produce `bool`; arithmetic keeps
                // the operand type when both sides are primitives.
                match b.op {
                    syn::BinOp::Eq(_)
                    | syn::BinOp::Ne(_)
                    | syn::BinOp::Lt(_)
                    | syn::BinOp::Le(_)
                    | syn::BinOp::Gt(_)
                    | syn::BinOp::Ge(_)
                    | syn::BinOp::And(_)
                    | syn::BinOp::Or(_) => RTy::Prim,
                    _ => {
                        let l = self.resolve_expr(b.left.as_ref());
                        if l == RTy::Prim && self.resolve_expr(b.right.as_ref()) == RTy::Prim {
                            RTy::Prim
                        } else {
                            RTy::Unknown
                        }
                    }
                }
            }
            Expr::Unary(u) => match u.op {
                syn::UnOp::Deref(_) => self.resolve_expr(u.expr.as_ref()),
                _ => self.resolve_expr(u.expr.as_ref()),
            },
            Expr::Reference(r) => self.resolve_expr(r.expr.as_ref()),
            _ => RTy::Unknown,
        }
    }

    fn resolve_return(&self, ty: &str, method: &str) -> RTy {
        match self.declared.returns.get(&(ty.to_string(), method.to_string())) {
            Some(Some(t)) => classify_ty(t, Some(ty), self.declared),
            // A declared `fn` with no return type: unit, which has no methods
            // worth resolving.
            Some(None) => RTy::Unknown,
            None => RTy::Unknown,
        }
    }

    /// G9: a path *expression* (a value read) must be a local, `self`/`Self`, a
    /// name this block declares, or a primitive-/std-qualified path. A bare
    /// `SOME_CONST` declared outside the block is a value the kernel's meaning
    /// depends on that CONFIG_HASH cannot see.
    fn check_value_path(&mut self, path: &syn::Path, span: proc_macro2::Span) {
        let Some(first) = path.segments.first() else { return };
        let first_s = first.ident.to_string();
        if path.segments.len() == 1 {
            let ok = self.locals.contains(&first_s)
                || first_s == "self"
                || first_s == "Self"
                || self.declared.values.contains(&first_s)
                || self.declared.types.contains(&first_s);
            if !ok {
                self.reject_value_path(path, span);
            }
            return;
        }
        if self.check_std_purity(path, span) {
            return;
        }
        if self.check_qualified_tail(path, span) {
            return;
        }
        let ok = self.declared.values.contains(&first_s)
            || EXTERNAL_ROOTS.contains(&first_s.as_str())
            || PRIMITIVE_TYPES.contains(&first_s.as_str());
        if !ok {
            self.reject_value_path(path, span);
        }
    }

    fn reject_value_path(&mut self, path: &syn::Path, span: proc_macro2::Span) {
        self.errors.push(syn::Error::new(
            span,
            format!(
                "config method body reads `{}`, which is neither a local nor declared inside \
                 this vericl::config! block — a `const`/`static` defined outside the block \
                 participates in what the kernel computes without participating in its \
                 CONFIG_HASH, so editing it would leave the kernel's recorded identity \
                 unmoved. Move the `const` INTO this block (it is hashed there), or pin the \
                 value with instantiate(...)",
                render_path(path)
            ),
        ));
    }
}

impl<'ast> Visit<'ast> for BodyGate<'_> {
    fn visit_expr_method_call(&mut self, i: &'ast syn::ExprMethodCall) {
        self.check_method_call(i);
        syn::visit::visit_expr_method_call(self, i);
    }

    /// `let` bindings feed the receiver-type environment, in source order: the
    /// initializer is gated (and typed) first, then the binding is recorded, so
    /// `let base = self.window(); base.taps()` resolves `base` to `WindowCfg`.
    fn visit_local(&mut self, i: &'ast syn::Local) {
        if let Some(init) = &i.init {
            self.visit_expr(&init.expr);
            if let Some((_, div)) = &init.diverge {
                self.visit_expr(div);
            }
        }
        let (name, declared_ty) = match &i.pat {
            syn::Pat::Ident(pi) => (Some(pi.ident.to_string()), None),
            syn::Pat::Type(pt) => match pt.pat.as_ref() {
                syn::Pat::Ident(pi) => (Some(pi.ident.to_string()), Some(pt.ty.as_ref())),
                _ => (None, None),
            },
            _ => (None, None),
        };
        if let Some(name) = name {
            let ty = match (declared_ty, &i.init) {
                (Some(t), _) => classify_ty(t, self.self_ty.as_deref(), self.declared),
                (None, Some(init)) => self.resolve_expr(&init.expr),
                (None, None) => RTy::Unknown,
            };
            self.local_tys.insert(name, ty);
        }
    }

    fn visit_expr_call(&mut self, i: &'ast syn::ExprCall) {
        if let Expr::Path(p) = crate::peel_paren(i.func.as_ref()) {
            let rejected_by_name =
                p.path.segments.last().is_some_and(|last| self.check_reject_list(&last.ident));
            if !rejected_by_name {
                self.check_callee_path(&p.path, p.qself.as_ref());
            }
        } else {
            // A computed callee (`(self.f)(x)`, `arr[i](x)`) cannot be resolved
            // to a declaration, so it cannot be gated — reject rather than
            // wave through.
            self.errors.push(syn::Error::new(
                i.func.span(),
                "a computed callee in a config method body is outside the vericl v1 \
                 struct-comptime subset — vericl::config! can only certify a call whose callee it \
                 can resolve to a declaration inside the block; call a named function instead",
            ));
        }
        // Deliberately does NOT recurse through `check_callee_path` twice: the
        // arguments still get the full treatment.
        for a in &i.args {
            self.visit_expr(a);
        }
    }

    fn visit_expr_struct(&mut self, i: &'ast syn::ExprStruct) {
        let named = i.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
        if named != "Self" && !self.declared.types.contains(&named) {
            self.errors.push(syn::Error::new(
                i.path.span(),
                format!(
                    "config method body constructs `{named}`, a type not declared inside this \
                     vericl::config! block — its definition and its methods would then determine \
                     what the kernel computes without being covered by CONFIG_HASH; declare the \
                     type in this block"
                ),
            ));
        }
        syn::visit::visit_expr_struct(self, i);
    }

    fn visit_expr_path(&mut self, i: &'ast syn::ExprPath) {
        if i.qself.is_none() {
            self.check_value_path(&i.path, i.span());
        }
        syn::visit::visit_expr_path(self, i);
    }

    fn visit_expr_macro(&mut self, i: &'ast syn::ExprMacro) {
        self.errors.push(syn::Error::new(
            i.span(),
            "a macro invocation in a config method body is outside the vericl v1 struct-comptime \
             subset — a macro's tokens are opaque to the gates above (`anything!(fma(a, b, c))` \
             would evade both the host-callability check and the \
             calls-must-be-declared-in-this-block check wholesale), so admitting one would make \
             the block's hash cover text vericl never inspected; write the expression out (a \
             `matches!` is an `if let`, an `assert!` is an `if { panic }`)",
        ));
    }

    fn visit_stmt_macro(&mut self, i: &'ast syn::StmtMacro) {
        self.errors.push(syn::Error::new(
            i.span(),
            "a macro invocation in a config method body is outside the vericl v1 struct-comptime \
             subset — a macro's tokens are opaque to the gates above, so admitting one would make \
             the block's hash cover text vericl never inspected",
        ));
    }

    fn visit_item_macro(&mut self, i: &'ast syn::ItemMacro) {
        self.errors.push(syn::Error::new(
            i.span(),
            "a macro invocation in a config method body is outside the vericl v1 struct-comptime \
             subset — a macro's tokens are opaque to the gates above",
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(src: &str) -> String {
        let ts: TokenStream2 = src.parse().expect("valid tokens");
        expand(ts).unwrap_or_else(|e| panic!("expected acceptance, got: {e}")).to_string()
    }

    fn err(src: &str) -> String {
        let ts: TokenStream2 = src.parse().expect("valid tokens");
        match expand(ts) {
            Ok(t) => panic!("expected rejection, got acceptance: {t}"),
            Err(e) => e.to_string(),
        }
    }

    fn hash_of(src: &str) -> String {
        let out = ok(src);
        let marker = "CONFIG_HASH : & 'static str = \"";
        let i = out.find(marker).unwrap_or_else(|| panic!("no CONFIG_HASH in: {out}"));
        let rest = &out[i + marker.len()..];
        rest[..rest.find('"').expect("closing quote")].to_string()
    }

    const BASE: &str = r#"
        #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
        pub struct TileCfg { pub m: u32, pub n: u32 }
        impl TileCfg { pub fn total(&self) -> u32 { self.m * self.n } }
    "#;

    /// M1(a) — the design's §5.1 A/B at the macro's own level: the config
    /// method edit that leaves a kernel's `SOURCE_HASH` bit-identical must move
    /// `CONFIG_HASH`. This is the whole point of the macro.
    #[test]
    fn method_body_edit_moves_config_hash() {
        let alt = BASE.replace("self.m * self.n", "self.m + self.n");
        assert_ne!(hash_of(BASE), hash_of(&alt), "a config method body edit must move CONFIG_HASH");
    }

    /// M1(b) — hash granularity, documented in the module doc: whitespace and
    /// ordinary comments do NOT move the hash (the input is tokenized), a doc
    /// comment DOES (it tokenizes to a `#[doc]` attribute).
    #[test]
    fn config_hash_granularity_is_token_level() {
        let spaced = BASE.replace("{ self.m * self.n }", "{\n\n    self.m * self.n\n\n}");
        assert_eq!(hash_of(BASE), hash_of(&spaced), "whitespace must not move CONFIG_HASH");
        let commented = BASE.replace("impl TileCfg", "// a comment\nimpl TileCfg");
        assert_eq!(hash_of(BASE), hash_of(&commented), "a `//` comment must not move CONFIG_HASH");
        let documented = BASE.replace("pub struct TileCfg", "/// docs\npub struct TileCfg");
        assert_ne!(hash_of(BASE), hash_of(&documented), "a doc comment must move CONFIG_HASH");
    }

    /// M1(d) — the declared items are re-emitted VERBATIM: the output starts
    /// with exactly the input tokens, with only the `impl ConfigIdentity`
    /// appended.
    #[test]
    fn items_are_re_emitted_verbatim() {
        let ts: TokenStream2 = BASE.parse().unwrap();
        let out = expand(ts.clone()).unwrap().to_string();
        assert!(out.starts_with(&ts.to_string()), "input tokens must be re-emitted verbatim: {out}");
        assert!(out.contains("impl :: vericl :: ConfigIdentity for TileCfg"), "{out}");
    }

    /// Two distinct types in one block share the block's hash — the block, not
    /// the type, is the unit of identity (design §7, nested configs).
    #[test]
    fn every_declared_type_gets_an_impl() {
        let out = ok("pub struct A { pub m: u32 } pub enum B { X, Y }");
        assert!(out.contains("impl :: vericl :: ConfigIdentity for A"), "{out}");
        assert!(out.contains("impl :: vericl :: ConfigIdentity for B"), "{out}");
    }

    /// G1.
    #[test]
    fn a_block_with_no_type_is_rejected() {
        assert!(err("pub fn f() -> u32 { 1 }").contains("at least one struct or enum"));
    }

    /// G2 / design R3 — and the negative control: an ordinary derive is fine.
    #[test]
    fn cube_attribute_is_rejected_anywhere_in_the_block() {
        let e = err("pub struct C { pub m: u32 } #[cube] impl C { pub fn f(&self) -> u32 { 1 } }");
        assert!(e.contains("`#[cube]` attribute inside a vericl::config! block"), "{e}");
        let e2 = err(
            "pub struct C { pub m: u32 } impl C { #[cube] pub fn f(&self) -> u32 { 1 } }",
        );
        assert!(e2.contains("`#[cube]`"), "{e2}");
        ok("#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)] pub struct C { pub m: u32 }");
    }

    /// G3 / design R4 — the I3 shape: `fma` in a config method body is a
    /// COMPILE error at the callee's span, not a runtime twin panic.
    #[test]
    fn reject_listed_call_in_a_config_method_is_rejected() {
        let e = err(
            "pub struct C { pub m: u32 } impl C { pub fn f(&self) -> u32 { fma(1, 2, 3) } }",
        );
        assert!(e.contains("calls `fma`"), "{e}");
        assert!(e.contains("not verified host-callable"), "{e}");
        let m = err(
            "pub struct C { pub m: u32 } impl C { pub fn f(&self) -> u32 { self.m.mul_hi(2) } }",
        );
        assert!(m.contains("calls `mul_hi`"), "{m}");
    }

    /// G3 discrimination: a config method NAMED `dot` (the design's M4 false
    /// positive) is a declaration, not a call, and compiles. Since round 10 the
    /// method gate resolves receivers, so *calling* an in-block `dot` on an
    /// in-block type compiles too — it is a plain host method, gated by this
    /// same walker, not cubecl's `unexpanded!()` `Float::dot`. Calling `dot` on
    /// anything the block does NOT declare is still rejected by name.
    #[test]
    fn a_config_method_named_dot_can_be_declared_and_called_in_block() {
        ok("pub struct C { pub m: u32 } impl C { pub fn dot(&self) -> u32 { self.m } }");
        ok("pub struct W { pub m: u32 } pub struct S { pub w: W } \
            impl W { pub fn dot(&self) -> u32 { self.m } } \
            impl S { pub fn w(&self) -> W { self.w } \
                     pub fn dot(&self) -> u32 { self.w().dot() } }");
        // …and on a primitive receiver `dot` is cubecl's intrinsic name again.
        let e = err("pub struct C { pub m: f32 } impl C { pub fn d(&self) -> f32 { \
                     self.m.dot(self.m) } }");
        assert!(e.contains("calls `dot`"), "{e}");
        assert!(e.contains("not verified host-callable"), "{e}");
    }

    /// G4 / design risk 2 — the sharpest open question in the design, closed in
    /// the sound direction: a free function defined outside the block is
    /// rejected by name; moving it into the block accepts it (and hashes it).
    #[test]
    fn free_function_outside_the_block_is_rejected_and_inside_is_accepted() {
        let e = err("pub struct C { pub m: u32 } impl C { pub fn f(&self) -> u32 { packing(self.m) } }");
        assert!(e.contains("calls `packing`"), "{e}");
        assert!(e.contains("not declared inside this vericl::config! block"), "{e}");
        ok("pub struct C { pub m: u32 } \
            pub const fn packing(v: u32) -> u32 { v << 1 } \
            impl C { pub fn f(&self) -> u32 { packing(self.m) } }");
        // …and moving it in moves the hash when its body changes — the whole
        // reason the gate is worth having.
        let a = "pub struct C { pub m: u32 } pub const fn p(v: u32) -> u32 { v << 1 } \
                 impl C { pub fn f(&self) -> u32 { p(self.m) } }";
        let b = "pub struct C { pub m: u32 } pub const fn p(v: u32) -> u32 { v << 2 } \
                 impl C { pub fn f(&self) -> u32 { p(self.m) } }";
        assert_ne!(hash_of(a), hash_of(b), "an in-block free fn's body must move CONFIG_HASH");
    }

    /// G4, the constructor forms: a tuple struct / tuple variant declared in
    /// the block is callable by name (`Wrap(3)`, `Mode::Scaled(3)`, `Self(3)`),
    /// while an undeclared constructor is not.
    #[test]
    fn tuple_constructors_of_declared_types_are_callable() {
        ok("pub struct Wrap(pub u32); pub enum M { A(u32), B } \
            impl Wrap { \
              pub fn a(&self) -> Wrap { Wrap(self.0) } \
              pub fn b(&self) -> Self { Self(self.0) } \
              pub fn c(&self) -> M { M::A(self.0) } }");
        let e = err("pub struct C { pub m: u32 } impl C { pub fn f(&self) -> u32 { Foreign(1) } }");
        assert!(e.contains("`Foreign`"), "{e}");
    }

    /// G4 allowlist: std/core, primitive associated fns, `Self`, and the
    /// block's own types resolve; an unrelated path root does not.
    #[test]
    fn callee_allowlist_admits_std_primitive_and_self() {
        ok("pub struct C { pub m: u32 } impl C { \
              pub fn f(&self) -> u32 { core::cmp::max(self.m, u32::min(1, 2)) } \
              pub fn g(&self) -> u32 { Self::h(self.m) } \
              pub fn h(v: u32) -> u32 { v } }");
        let e = err("pub struct C { pub m: u32 } impl C { pub fn f(&self) -> u32 { \
                     cubecl::prelude::something(self.m) } }");
        assert!(e.contains("cubecl::prelude::something"), "{e}");
    }

    /// G5.
    #[test]
    fn a_generic_config_type_is_rejected() {
        let e = err("pub struct C<S> { pub s: S }");
        assert!(e.contains("generic config type"), "{e}");
    }

    /// G6 / design §7 — the cross-block nested-config hole, closed: a nested
    /// config's type must be declared in the SAME block.
    #[test]
    fn a_field_of_an_undeclared_type_is_rejected() {
        let e = err("pub struct Stage { pub tile: TileCfg, pub k: u32 }");
        assert!(e.contains("cannot account for"), "{e}");
        // Declared in the same block: accepted, and one hash covers both.
        ok("pub struct TileCfg { pub m: u32 } pub struct Stage { pub tile: TileCfg, pub k: u32 }");
        // Arrays and tuples of allowed types are fine; a generic wrapper is not.
        ok("pub struct C { pub a: [u32; 4], pub b: (u32, bool) }");
        assert!(err("pub struct C { pub a: Option<u32> }").contains("cannot account for"));
    }

    /// G7.
    #[test]
    fn disallowed_item_kinds_are_rejected() {
        let e = err("pub struct C { pub m: u32 } pub static S: u32 = 1;");
        assert!(e.contains("only `struct`, `enum`, `impl`"), "{e}");
        let m = err("pub struct C { pub m: u32 } pub mod inner { pub fn f() {} }");
        assert!(m.contains("only `struct`, `enum`, `impl`"), "{m}");
    }

    /// G7, the ecosystem-shaped case: a macro-generated config type (CubeCL's
    /// own `define_3d_size_base!` family) gets its own message naming the
    /// reason and the workaround, rather than the generic item-kind text.
    #[test]
    fn a_macro_declared_config_type_is_rejected_with_a_targeted_message() {
        let e = err("define_3d_size_base!(TileSize, u32);");
        assert!(e.contains("macro invocation cannot declare a config type"), "{e}");
        assert!(e.contains("define_3d_size_base"), "the message must name the real shape: {e}");
    }

    /// G8.
    #[test]
    fn a_macro_in_a_config_body_is_rejected() {
        let e = err("pub struct C { pub m: u32 } impl C { pub fn f(&self) -> bool { \
                     matches!(self.m, 1) } }");
        assert!(e.contains("macro invocation in a config method body"), "{e}");
    }

    /// G9 — an out-of-block `const` read is rejected; an in-block one is not.
    #[test]
    fn out_of_block_const_read_is_rejected() {
        let e = err("pub struct C { pub m: u32 } impl C { pub fn f(&self) -> u32 { self.m * K } }");
        assert!(e.contains("reads `K`"), "{e}");
        ok("pub struct C { pub m: u32 } pub const K: u32 = 3; \
            impl C { pub fn f(&self) -> u32 { self.m * K } }");
        // A local, an enum variant of a declared enum, and `u32::MAX` all resolve.
        ok("pub struct C { pub m: u32 } pub enum M { A, B } \
            impl C { pub fn f(&self) -> u32 { let t = self.m; let _ = M::A; t.min(u32::MAX) } }");
    }

    /// G4, the METHOD-call half — round-10 review probe P1, closed.
    ///
    /// `self.combine()` with `combine` declared in an impl *outside* the block
    /// passed every gate before this: two blocks with byte-identical in-block
    /// tokens computed ×24 and ×11 with identical `CONFIG_HASH`es and identical
    /// recorded kernel identities. The positive control beside it is the same
    /// call with `combine` moved in.
    #[test]
    fn out_of_block_method_called_through_method_syntax_is_rejected() {
        let e = err("pub struct C { pub m: u32, pub n: u32 } \
                     impl C { pub fn total(&self) -> u32 { self.combine() } }");
        assert!(e.contains("calls `.combine(…)`"), "{e}");
        assert!(e.contains("declares no method `combine` for `C`"), "{e}");
        // …and moving it in accepts it, and its body then moves the hash.
        ok("pub struct C { pub m: u32, pub n: u32 } \
            impl C { pub fn total(&self) -> u32 { self.combine() } \
                     pub fn combine(&self) -> u32 { self.m * self.n } }");
        let a = "pub struct C { pub m: u32, pub n: u32 } \
                 impl C { pub fn t(&self) -> u32 { self.c() } pub fn c(&self) -> u32 { self.m * self.n } }";
        let b = "pub struct C { pub m: u32, pub n: u32 } \
                 impl C { pub fn t(&self) -> u32 { self.c() } pub fn c(&self) -> u32 { self.m + self.n } }";
        assert_ne!(hash_of(a), hash_of(b), "an in-block method's body must move CONFIG_HASH");
    }

    /// G4/method — round-10 review probe P5a: a user extension trait
    /// implemented on a **primitive** outside the block (`impl Boost for u32`),
    /// called in method form from an in-block config method. `self.m.boost()`
    /// turned `m` into `m * 7` through code the block neither hashed nor gated.
    #[test]
    fn user_extension_trait_method_on_a_primitive_field_is_rejected() {
        let e = err("pub struct C { pub m: u32 } \
                     impl C { pub fn total(&self) -> u32 { self.m.boost() } }");
        assert!(e.contains("calls `.boost(…)`"), "{e}");
        assert!(e.contains("user extension trait"), "{e}");
        // Positive control: the `std` inherent surface on the same receiver.
        ok("pub struct C { pub m: u32 } \
            impl C { pub fn total(&self) -> u32 { self.m.pow(2).min(9).wrapping_add(1) } }");
    }

    /// G4/method receiver resolution: a nested config's method resolves through
    /// a field, through a declared method's return type, and through a `let`.
    #[test]
    fn method_calls_resolve_through_fields_returns_and_lets() {
        ok("pub struct W { pub taps: u32 } pub struct S { pub w: W } \
            impl W { pub fn taps(&self) -> u32 { self.taps } } \
            impl S { pub fn w(&self) -> W { self.w } \
                     pub fn a(&self) -> u32 { self.w.taps() } \
                     pub fn b(&self) -> u32 { self.w().taps() } \
                     pub fn c(&self) -> u32 { let t = self.w(); t.taps() } }");
        // …and the same shapes reject when the method is NOT declared in-block.
        let e = err("pub struct W { pub taps: u32 } pub struct S { pub w: W } \
                     impl S { pub fn a(&self) -> u32 { self.w.extra() } }");
        assert!(e.contains("declares no method `extra` for `W`"), "{e}");
    }

    /// G9, the qualified half — round-10 review probe P7. `Self::K` reading an
    /// associated `const` declared in an impl OUTSIDE the block was accepted by
    /// a root-only check; the tail is now resolved against the block.
    #[test]
    fn out_of_block_associated_const_is_rejected() {
        let e = err("pub struct C { pub m: u32 } \
                     impl C { pub fn total(&self) -> u32 { self.m * Self::K } }");
        assert!(e.contains("Self :: K") || e.contains("Self::K"), "{e}");
        assert!(e.contains("declares no associated item `K` for `C`"), "{e}");
        // In-block: accepted, through both spellings.
        ok("pub struct C { pub m: u32 } \
            impl C { pub const K: u32 = 8; \
                     pub fn a(&self) -> u32 { self.m * Self::K } \
                     pub fn b(&self) -> u32 { self.m * C::K } }");
        // The same escape through a qualified CALL, not a read.
        let c = err("pub struct C { pub m: u32 } \
                     impl C { pub fn total(&self) -> u32 { Self::combine(self.m) } }");
        assert!(c.contains("declares no associated item `combine` for `C`"), "{c}");
    }

    /// G11 — a custom derive is the unhashed-impl sibling of G7's macro
    /// rejection; the `std` derives stay, and their associated items resolve.
    #[test]
    fn custom_derives_are_rejected_and_std_derives_are_not() {
        let e = err("#[derive(Clone, Copy, CubeType)] pub struct C { pub m: u32 }");
        assert!(e.contains("`#[derive(CubeType)]`"), "{e}");
        assert!(e.contains("derive macro's own definition"), "{e}");
        let s = err("#[derive(serde::Serialize)] pub struct C { pub m: u32 }");
        assert!(s.contains("serde::Serialize"), "{s}");
        ok("#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default, PartialOrd, Ord)] \
            pub struct C { pub m: u32 }");
        // A derived associated item resolves against the block.
        ok("#[derive(Clone, Copy, Default)] pub struct C { pub m: u32 } \
            impl C { pub fn d(&self) -> u32 { Self::default().m } \
                     pub fn c(&self) -> C { self.clone() } }");
    }

    /// G14 (round 11) — the classification split. `cfg_attr` is expanded by
    /// rustc *after* G2/G11 have read the attribute list, so it re-spells both
    /// gates at once: `#[cfg_attr(all(), cube)]` puts a `#[cube]` method past
    /// G2 (the measured ×24-vs-×11 divergence class) and
    /// `#[cfg_attr(all(), derive(Evil))]` puts a custom derive past G11.
    #[test]
    fn cfg_attr_anywhere_in_the_block_is_rejected() {
        for src in [
            "#[cfg_attr(all(), cube)] pub struct C { pub m: u32 }",
            "#[cfg_attr(all(), derive(serde::Serialize))] pub struct C { pub m: u32 }",
            "pub struct C { #[cfg_attr(all(), serde(skip))] pub m: u32 }",
            "pub struct C { pub m: u32 } #[cfg_attr(all(), cube)] impl C { pub fn t(&self) -> u32 { self.m } }",
            "pub struct C { pub m: u32 } impl C { #[cfg_attr(all(), cube)] pub fn t(&self) -> u32 { self.m } }",
        ] {
            let e = err(src);
            assert!(e.contains("`#[cfg_attr(…)]`"), "{src}: {e}");
            assert!(e.contains("vericl::config!"), "the message must name THIS macro: {src}: {e}");
        }
        // Negative control: an ordinary `#[derive]` and a doc comment are
        // untouched by this gate.
        ok("/// docs\n#[derive(Clone, Copy)] pub struct C { pub m: u32 }");
    }

    /// G12's derive-name half (round 11): the derive gate admits `#[derive(X)]`
    /// by comparing `X` to the `std` set BY NAME, exactly as G4/G9 resolve a
    /// path root by name — so a `use … as Hash;` is the same escape one
    /// namespace over.
    #[test]
    fn rebinding_a_std_derive_name_is_rejected() {
        for d in ["Hash", "Debug", "Clone", "Ord"] {
            let e = err(&format!(
                "use crate::evil as {d}; #[derive({d})] pub struct C {{ pub m: u32 }}"
            ));
            assert!(e.contains("rebinds a DERIVE name"), "{d}: {e}");
            assert!(e.contains("vericl::config!"), "{d}: {e}");
        }
        ok("use core::cmp as _c; pub struct C { pub m: u32 }");
    }

    /// G10 (purity) — round-10 review probe P2. `std::env::var` passed every one
    /// of G1–G9 and made the twin's answer depend on the environment.
    #[test]
    fn impure_std_modules_are_rejected_and_pure_ones_are_not() {
        let e = err("pub struct C { pub m: u32 } impl C { pub fn t(&self) -> u32 { \
                     match std::env::var(\"X\") { Ok(_) => self.m + 2, Err(_) => self.m } } }");
        assert!(e.contains("std::env::var") || e.contains("std :: env :: var"), "{e}");
        assert!(e.contains("impure-module denylist"), "{e}");
        for path in ["std::process::id()", "std::time::Instant::now()", "std::fs::metadata(\"x\")"]
        {
            let m = err(&format!(
                "pub struct C {{ pub m: u32 }} impl C {{ pub fn t(&self) -> u32 {{ \
                 let _ = {path}; self.m }} }}"
            ));
            assert!(m.contains("impure-module denylist"), "{path}: {m}");
        }
        let r = err("pub struct C { pub m: u32 } \
                     impl C { pub fn t(&self) -> u32 { rand::random::<u32>() } }");
        assert!(r.contains("randomness-dependent") || r.contains("clock-"), "{r}");
        // `mem` is denied for target-dependence, with its own diagnosis.
        let mm = err("pub struct C { pub m: u32 } \
                      impl C { pub fn t(&self) -> usize { core::mem::size_of::<usize>() } }");
        assert!(mm.contains("target-dependent"), "{mm}");
        // The pure computational surface stays.
        ok("pub struct C { pub m: u32 } impl C { \
              pub fn a(&self) -> u32 { core::cmp::max(self.m, 1) } \
              pub fn b(&self) -> f32 { core::f32::consts::PI } }");
    }

    /// G12 — round-10 review probe P5b: `use crate::evil as core;` re-pointed
    /// G4/G9's allowlisted root at user code, and `core::cmp::max(self.m, 1)`
    /// evaluated to `self.m * 100`.
    #[test]
    fn rebinding_an_allowlisted_path_root_is_rejected() {
        let e = err("use crate::evil as core; pub struct C { pub m: u32 } \
                     impl C { pub fn t(&self) -> u32 { core::cmp::max(self.m, 1) } }");
        assert!(e.contains("rebinds a path root"), "{e}");
        let g = err("use crate::evil::*; pub struct C { pub m: u32 }");
        assert!(g.contains("glob"), "{g}");
        let p = err("use crate::mine as u32; pub struct C { pub m: u32 }");
        assert!(p.contains("rebinds a path root"), "{p}");
        // An ordinary import under its own name is fine as an ITEM…
        ok("use core::cmp::max; pub struct C { pub m: u32 } \
            impl C { pub fn t(&self) -> u32 { core::cmp::max(self.m, 1) } }");
    }

    /// …but calling an imported free function by its BARE name stays rejected,
    /// and that is the sound direction: `use core::cmp::max;` and
    /// `use crate::evil::max;` are indistinguishable at the call site, so a bare
    /// single-segment callee must still resolve to something the block declares.
    /// The fix the message names — write the qualified path — is the one that
    /// keeps the root check meaningful.
    #[test]
    fn a_bare_call_to_an_imported_free_function_is_still_rejected() {
        let e = err("use core::cmp::max; pub struct C { pub m: u32 } \
                     impl C { pub fn t(&self) -> u32 { max(self.m, 1) } }");
        assert!(e.contains("calls `max`"), "{e}");
        assert!(e.contains("not declared inside this vericl::config! block"), "{e}");
    }

    /// G13 — a config fn's return type is gated exactly like a field's, so the
    /// kernel-side chain rule (`FloatMethodCheck` exempts only the FIRST link of
    /// a chain rooted at a config parameter) has something to compose with.
    #[test]
    fn config_return_types_are_gated() {
        let e = err("pub struct C { pub m: u32 } \
                     impl C { pub fn t(&self) -> String { String::new() } }");
        assert!(e.contains("returns a type vericl::config! cannot account for"), "{e}");
        let o = err("pub struct C { pub m: u32 } \
                     impl C { pub fn t(&self) -> Option<u32> { None } }");
        assert!(o.contains("returns a type"), "{o}");
        ok("pub struct T { pub m: u32 } pub struct C { pub t: T } \
            impl C { pub fn a(&self) -> u32 { self.t.m } \
                     pub fn b(&self) -> T { self.t } \
                     pub fn c(&self) -> Self { *self } \
                     pub fn d(&self) -> (u32, bool) { (self.t.m, true) } \
                     pub fn e(&self) {} }");
    }

    /// `use core::cmp::max;` then a bare `max(...)` — a `use` brings the name
    /// into the value namespace, and G4's bare-callee branch already admits it
    /// because `collect_locals`… does not. Pinned so the interaction is not
    /// silently changed: the import must be accepted and the call resolved.
    #[test]
    fn imported_free_function_is_callable_by_its_bare_name() {
        ok("use core::cmp::max; pub struct C { pub m: u32 } \
            pub const fn max2(a: u32) -> u32 { a } \
            impl C { pub fn t(&self) -> u32 { max2(self.m) } }");
    }

    /// Loop bounds, `match`, `if`, nested method chains — the real config
    /// idiom — all compile through the gates untouched.
    #[test]
    fn the_real_config_idiom_is_accepted() {
        ok("
            #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
            pub struct Tile { pub m: u32, pub n: u32 }
            #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
            pub enum Mode { Single, Double }
            #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
            pub struct Stage { pub tile: Tile, pub mode: Mode }
            impl Tile {
                pub fn total(&self) -> u32 { self.m * self.n }
                pub fn m(&self) -> u32 { self.m }
            }
            impl Stage {
                pub fn tile(&self) -> Tile { self.tile }
                pub fn taps(&self) -> u32 {
                    let base = self.tile().total();
                    match self.mode { Mode::Single => base, Mode::Double => 2 * base }
                }
                pub fn wide(&self) -> bool { if self.taps() > 4 { true } else { false } }
            }
        ");
    }
}
