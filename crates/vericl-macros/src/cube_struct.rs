//! `vericl::cube_struct! { … }` — the declaration form for a **runtime**
//! (non-`#[comptime]`) `CubeType` struct parameter's type.
//!
//! # What it is for
//!
//! CubeCL lets a `#[cube]` item take `data: MyStruct` where `MyStruct` derives
//! `CubeType`/`CubeLaunch`: the struct parameter is lowered as a **positional
//! flattening of its fields** at the struct's own parameter slot, in field
//! declaration order (`docs/design-cubetype-args.md` §2 — measured three ways:
//! bit-exact GPU output vs the flattened spelling, identical `KernelDefinition`,
//! byte-identical `kernel_ir_hash`). VeriCL accepted the helper half of that
//! shape before this macro existed, **silently and with no diagnostic at all**
//! (design §4.2, probe V3), with two measured defects:
//!
//! 1. **the identity hole, live at `e5589f3`** — the struct type's *definition*
//!    is in neither input of `SOURCE_HASH`. With a
//!    `#[cube] impl Pair { fn fold(&self) -> u32 }` edited from `self.a *
//!    self.b` to `self.a + self.b`, the reference twin went from `[3, 6, 9, 12]`
//!    to `[4, 5, 6, 7]` while the kernel's `SOURCE_HASH`, the helper's
//!    `SOURCE_HASH` **and** `identity().source_hash` all stayed bit-identical,
//!    and evidence recorded against the first build verified FRESH against the
//!    second (design §4.1, probe V4);
//! 2. **the positional-constructor hazard** — `<Name>Launch::new` fills fields
//!    by position (`generate_struct.rs:92-114`), so swapping two same-typed
//!    fields in the *declaration* changes the computed function with the kernel
//!    body and the launch-call text byte-unchanged (design §4.3, probe X2).
//!
//! Wrapping the declaration in one item macro is what makes both fixable, and —
//! decisively — what makes the feature *implementable at all*: a token-only
//! attribute macro on the kernel cannot know what `args` contains, so it could
//! not emit `UniformLaunch::new(…)`, could not build the twin's binding, and
//! could not resolve `gen(args.lower_bound in …)`. The identity requirement and
//! the implementability requirement are the same requirement, which is why this
//! is one mechanism rather than two (design §5.2).
//!
//! # What it does
//!
//! - **Re-emits every declared item**, prefixed with the derives the macro owns:
//!   `Clone`, `Copy`, `::cubecl::prelude::CubeType` and
//!   `::cubecl::prelude::CubeLaunch`. Owning the derives closes the derive-set
//!   escape — dropping `CubeLaunch` by hand would change the type from
//!   launchable to device-local with the kernel's tokens unchanged — and owning
//!   `Clone`/`Copy` is what lets the generated twin bind the struct by value the
//!   way the design's §5.4 twin mapping requires.
//! - **Hashes the whole block** into
//!   `impl ::vericl::StructIdentity for T { const STRUCT_HASH }`, which the
//!   kernel/helper folds into its recorded identity via
//!   `::vericl::combine_source_hash`. Where the type can also serve `#[comptime]`
//!   position it emits `ConfigIdentity` with the same hash (design §6, "one type,
//!   both positions" — **conditional**, see below); the converse never holds, and
//!   a `vericl::config!` type used as a runtime parameter lands on
//!   `StructIdentity`'s `#[diagnostic::on_unimplemented]` note.
//!
//! # "One type, both positions", precisely (round-11 correction)
//!
//! The design's §6 claim was shipped unconditionally and is **false in general**.
//! `ConfigIdentity` was emitted for every declared struct, but the trait is only
//! half of what `#[comptime]` position needs: CubeCL `Debug`-formats a comptime
//! parameter and derives `Hash`/`Eq` over it. Measured — a two-`u32` declared
//! struct as `#[comptime] c: IntCfg` failed with `no method named 'hash'`, `no
//! method named 'eq'` and "doesn't implement `Debug`", **even though the
//! `ConfigIdentity` impl was there**; and for a float-field struct no set of
//! derives can fix it, because `f32` is neither `Hash` nor `Eq`.
//!
//! So the honest rule, and what this macro now implements:
//!
//! - a declared type is comptime-usable **iff** every field in its transitive
//!   shape is an integer/`bool`/`char`, a declared unit enum, or another
//!   comptime-usable declared struct — one `f32`/`f64` anywhere disqualifies it;
//! - for those types the macro emits `Debug`/`PartialEq`/`Eq`/`Hash` **itself**
//!   (skipping any the author already wrote) together with `ConfigIdentity`, so
//!   the capability needs no recipe from the author at all;
//! - for the rest it emits neither, so naming one in `#[comptime]` position is a
//!   single `ConfigIdentity` `on_unimplemented` note that says *why*, instead of
//!   three raw trait errors pointing at `#[cube(launch)]`.
//!
//! The same four derives are why a declared **enum** is no longer re-emitted
//! untouched: without them the documented "unit enum `#[cube(comptime)]` field"
//! shape did not compile at all (`E0204` on the owning struct's `Copy`, then
//! `hash`/`clone` not found on the enum). It still gets no `CubeType`.
//!
//! The same fix is what makes a **declared struct** usable as a
//! `#[cube(comptime)]` field, pinned whole by `instantiate(p.win = Win { … })`
//! (`strided_window_sum` in `vericl-examples`). It is pinned *whole*: there is
//! no `gen(p.win.taps in …)` surface, and the nested spec alias for such a path
//! points at a marker type whose NAME is the diagnosis, so the mismatch is
//! rustc's `E0560` naming the reason rather than a raw `E0308` between two
//! generated type names (round 11, LOW 5).
//!
//! One honesty note on the enum, since CS2 advertises it: a comptime enum FIELD
//! enters the `CompilationArg` (two pins are two compiled kernels), but the v1
//! subset gives a kernel *body* no way to branch on it — a `match` inside
//! `#[cube]` lowers as a DEVICE match and needs `CubeEnum`, and a `comptime!`
//! block may reference only `#[comptime]` PARAMETERS, not a field of a runtime
//! struct. So its role today is the declaration/launch/identity path, not a
//! body-level switch. Stated rather than advertised.
//! - **Emits the launch/generation plumbing from the field order it hashed**:
//!   a hidden `<T>__VericlSpec` type carrying one `gen(...)` range per runtime
//!   field and one `instantiate(...)` pinned value per `#[cube(comptime)]`
//!   field, with `__vericl_draw` (the twin/GPU input), `__vericl_launch_arg`
//!   (the positional `<T>Launch::new`) and `__vericl_compilation_arg` (the
//!   IR-extraction `CompilationArg`, hand-built rather than obtained by
//!   registering — design risk 8).
//!
//! **Why a spec type rather than a token-level field list.** The kernel macro
//! never learns `T`'s fields; it only learns the *names the author wrote* in
//! `gen(p.f in …)` / `instantiate(p.c = …)`. It emits those as a struct literal
//! of `<T>__VericlSpec`, so **rustc's own struct-literal exhaustiveness check is
//! the field-coverage gate**: a field with no range is `E0063: missing field`
//! naming it, a misspelled one is `E0560: no field named`, and both fire at the
//! kernel's own span. That is why this works across crate boundaries, where a
//! macro-time registry could not.
//!
//! # The gates
//!
//! | # | Gate | Why |
//! |---|---|---|
//! | CS1 | the block must declare at least one `struct` | otherwise nothing gets a `StructIdentity` and the macro is a no-op that looks like a declaration (config G1) |
//! | CS2 | every field type must be a **single-segment** path naming a launch scalar (`f32`/`f64`/`u32`/`i32`/`u64`/`i64`) or a struct declared in **this** block; a `#[cube(comptime)]` field admits integer/`bool`/`char` and a unit-only enum declared in this block | a type declared in a *different* block would contribute meaning without contributing to the hash (config G6). The single-segment requirement is round 11: this gate resolves by the FINAL segment, so `sm::u32` passed as a `u32` while being an `f32`, and CS8 could not see it (the rebound name was `sm`). `Array`/`Slice`/`View`/`Sequence`/`SharedMemory`/`Tensor` fields are the design §10.5 deferral |
//! | CS3 | no generics on a declared type | `impl<T> StructIdentity for P<T>` gives every instantiation one hash (config G5) |
//! | CS4 | no `impl` block, and no `#[cube]` anywhere except the `#[cube(comptime)]` field attribute | a `#[cube]` method's body runs as host Rust in the twin and as expanded device code in the kernel, and nothing reconciles them — measured as the very edit that moves the twin without moving any hash (design §4.1) |
//! | CS5 | only `std` derives; `CubeType`/`CubeLaunch`/`Clone`/`Copy` are emitted by the macro and rejected if written | a custom derive's *definition* decides the type's impls and the hash covers only the invocation (config G11) |
//! | CS6 | no macro invocation inside the block | a macro's tokens are opaque to `syn`'s visitors, so CS2–CS5 would be evaded wholesale (config G8) |
//! | CS7 | only `struct`/`enum`/`use` items | v1 has no method surface, so any other item is either dead or an unhashed escape (config G7) |
//! | CS8 | a `use` may not rebind `core`/`std`/`alloc`/`Self`/a primitive/a `std` **derive** name, and may not be a glob | CS2 resolves field types and CS5 resolves derives **by name** (config G12, from round 10; the derive-name half is round 11; shared implementation in [`crate::decl_block`]). It closes only the *single-segment* rebinding — the qualified `sm::u32` spelling is closed by CS2's single-segment requirement instead |
//! | CS9 | a declared struct must have at least one **named** field | `gen(p.field in …)` and the launch constructor are field-name-driven; a unit or tuple struct has no such surface, and a unit struct additionally has no definition an edit could move |
//! | CS10 | the declared nesting graph must be acyclic, and bounded in depth/size | a recursive declared struct has no finite value (rustc's E0072) and no finite spec type; caught here so the diagnosis names the cycle. The nested-alias enumeration is a *graph-path* walk, so a legal DAG can still be exponential — capped rather than hung (round 11) |
//! | CS11 | no `#[cfg_attr(…)]` anywhere, and a FIELD may carry only the bare `#[cube(comptime)]` marker and doc comments | every gate here classifies attributes AS WRITTEN and rustc expands `cfg_attr` afterwards: `#[cfg_attr(all(), cube(comptime))]` is classified RUNTIME by vericl and COMPTIME by CubeCL, so the extracted IR is built with the field's `Default` while the launched kernel gets the drawn value — a false `Proved`. Round 11; shared `cfg_attr` implementation in [`crate::decl_block`], and the whitelist mirrors `classify_param`'s |
//!
//! # The residual, precisely
//!
//! Identical in shape to `vericl::config!`'s, and **worse in consequence**:
//! Rust permits an inherent `impl` for a local type anywhere in the crate, so a
//! `#[cube] impl P { … }` written *outside* the block is invisible to both the
//! hash and CS4. For a config the failure is a twin panic; for a runtime struct
//! it is a **numeric divergence**, because the device gets an expanded method
//! body the twin's host method may not match. There is no macro-scope fix (a
//! `#[proc_macro]` sees only the tokens it is handed). It is accepted with the
//! same loud backstops config's is, pinned as *passing tests asserting the hole
//! exists* in `crates/vericl-examples/tests/cube_struct_out_of_block_backstop.rs`:
//! the differential lane catches any divergence that reaches an output, and
//! `ir_hash` moves whenever the value reaches the device.
//!
//! ## The forged-identity bypass (round 11)
//!
//! Recorded here rather than only in the trait's own doc, because its scope is
//! different in kind from the residual above. `::vericl::StructIdentity` is a
//! **public, unsealed** trait, so nothing stops an author writing
//!
//! ```ignore
//! impl vericl::StructIdentity for MyOwnType {
//!     const STRUCT_HASH: &'static str = "sha256:0000…";
//! }
//! ```
//!
//! for a type this macro never saw. That is not a narrow gap in one gate — it is
//! a **complete bypass of the mechanism**: the type is never declared, so CS1–CS11
//! never run on it, its `#[cube] impl` methods are unrestricted, its field order
//! is not the one any constructor was emitted from, and its recorded identity is
//! a constant the author chose, which by construction never goes stale. It is
//! exactly the hole [`::vericl::StructIdentity`] exists to close, re-opened by
//! hand, and it is the same shape as `ConfigIdentity`'s ("a hand-written impl can
//! claim any hash it likes, including a constant one").
//!
//! It is **not** closable here and is not treated as an attack surface. Sealing
//! the trait would forbid the impl, but a `#[proc_macro]` cannot seal anything —
//! and more to the point, VeriCL's guarantee has never been "an author cannot lie
//! to their own evidence file"; it is "an author who does not lie gets an
//! identity that moves when the meaning does". Every gate in this module is aimed
//! at *accidental* drift, which is the failure that actually happens. The
//! bypass is stated so that it is never mistaken for something the gates cover,
//! and it is pinned as a compiling acknowledgment test
//! (`forged_struct_identity_is_a_complete_bypass`) that will stop compiling if
//! the trait is ever sealed — at which point this section is wrong and must be
//! rewritten.

use std::collections::HashSet;

use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, format_ident, quote};
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Ident, Item, Type};

use crate::NumKind;
use crate::decl_block::{block_hash, check_derives, render_path};

/// The suffix appended to a declared struct's name to form its hidden
/// generation/launch **spec** type — the single point of custody for a runtime
/// struct parameter's field order, shared between `vericl::cube_struct!` (which
/// emits it) and `#[vericl::kernel]`/`#[vericl::helper]` (which name it from a
/// parameter's written type). See [`spec_type_path`].
pub(crate) const SPEC_SUFFIX: &str = "__VericlSpec";

/// Scalar field types a **runtime** field may have.
///
/// Deliberately CubeCL's launch-scalar set intersected with the set
/// `build_gen_field` can draw (`NumKind`): a runtime field is generated exactly
/// as a loose scalar parameter of that type is generated today, so admitting a
/// type VeriCL cannot draw would produce a struct parameter with no differential
/// input. `usize`/`bool` are therefore runtime-rejected and comptime-admitted —
/// the honest v1 line, and a narrowing of `docs/design-cubetype-args.md` §10.2
/// recorded in that section's correction note.
const RUNTIME_FIELD_TYPES: &[&str] = &["f32", "f64", "u32", "i32", "u64", "i64"];

/// Scalar field types a `#[cube(comptime)]` field may have.
///
/// No float: `<Name>CompilationArg` derives `Hash`/`Eq` over every comptime
/// field (`generate_struct.rs:196-209`), and `f32` is neither — measured (design
/// §1.2, probe X3), which is why every comptime field in the surveyed ecosystem
/// is a `u32`, a `bool` or a unit enum.
const COMPTIME_FIELD_TYPES: &[&str] = &[
    "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize", "bool",
    "char",
];

/// One field of a declared struct.
struct DeclField {
    ident: Ident,
    ty: Type,
    /// `true` for a `#[cube(comptime)]` field: it keeps its positional slot in
    /// `<Name>Launch::new` but takes the plain host type and never reaches the
    /// device (design §1.2).
    comptime: bool,
}

/// One declared struct.
struct DeclStruct {
    ident: Ident,
    vis: syn::Visibility,
    fields: Vec<DeclField>,
}

/// Everything the block declares, by name.
#[derive(Default)]
struct Declared {
    structs: Vec<DeclStruct>,
    struct_names: HashSet<String>,
    /// Unit-only `enum`s — admissible as a `#[cube(comptime)]` field type. They
    /// get `ConfigIdentity` but **not** `StructIdentity`: see the trait's
    /// "Enums" section.
    enum_names: HashSet<String>,
}

pub(crate) fn expand(ts: TokenStream2) -> syn::Result<TokenStream2> {
    let file: syn::File = syn::parse2(ts.clone()).map_err(|e| {
        syn::Error::new(
            e.span(),
            format!(
                "vericl::cube_struct! takes a block of ordinary Rust items — the runtime struct \
                 type(s) it declares, and any unit enum a `#[cube(comptime)]` field of theirs \
                 uses: {e}"
            ),
        )
    })?;

    let mut errors: Vec<syn::Error> = Vec::new();
    let declared = collect_declared(&file, &mut errors);

    // CS1.
    if declared.structs.is_empty() && errors.is_empty() {
        return Err(syn::Error::new(
            ts.span(),
            "vericl::cube_struct! { … } must declare at least one struct — it exists to give a \
             runtime CubeType parameter's type a `StructIdentity` (the hash of its whole \
             definition) and to emit its CubeType/CubeLaunch derives and its positional launch \
             constructor from the field order it hashed; a block with no struct declaration does \
             none of those, so writing one would record a guarantee that was never made",
        ));
    }

    check_field_types(&declared, &mut errors);
    check_no_cube_attr(&file, &mut errors);
    check_block_derives(&file, &mut errors);
    check_no_macros(&file, &mut errors);
    // CS8 — the shared round-10 root-rebinding gate (probe P5b), one
    // implementation with `vericl::config!`'s G12.
    crate::decl_block::check_use_items(
        &file,
        "vericl::cube_struct!",
        "CS2 resolves a field type",
        &mut errors,
    );
    // CS11 — the round-11 classification-split gate, in two halves: `cfg_attr`
    // anywhere (shared with `vericl::config!`'s G14) and the field-attribute
    // whitelist.
    crate::decl_block::check_no_cfg_attr(&file, "vericl::cube_struct!", &mut errors);
    check_field_attrs(&file, &mut errors);

    if let Some(combined) = errors.clone().into_iter().reduce(|mut a, b| {
        a.combine(b);
        a
    }) {
        return Err(combined);
    }

    // CS10 — the nested-path enumeration below is a graph walk, so the cycle
    // check must run before it (and after CS2, which is what makes the graph
    // finite in the first place).
    let nested_paths = enumerate_nested_paths(&declared)?;

    // The hash covers the WHOLE block, at exactly the granularity a kernel's own
    // `SOURCE_HASH` uses. A field NAME, TYPE, ORDER or attribute edit all move
    // it — the field-order case being the §4.3 launch-side hazard, which this
    // macro turns from a silent miscomputation into a stale-evidence report.
    let hash = block_hash(&ts);

    // Which declared structs can serve COMPTIME position — measured, not
    // assumed (round-11 review, MODERATE 2). See [`comptime_usable_structs`].
    let comptime_usable = comptime_usable_structs(&declared);

    let mut out = TokenStream2::new();

    for item in &file.items {
        match item {
            Item::Struct(s) => {
                // Re-emitted with the derives the macro owns prefixed (design
                // §5.2 point 3). `Clone`/`Copy` are ours too: the generated twin
                // binds a struct parameter BY VALUE (design §5.4), and the
                // harness both hands it to `check_assumes` and launches with it,
                // so a non-`Copy` declared struct would fail to compile in
                // generated code for a reason the author never wrote.
                //
                // `Debug`/`PartialEq`/`Eq`/`Hash` are added exactly when every
                // field admits them: those are the four CubeCL needs to accept
                // the type in `#[comptime]` position, and emitting them for a
                // type that cannot have them would replace one honest error
                // with four confusing ones.
                let extra = comptime_derives(&s.attrs, comptime_usable.contains(&s.ident.to_string()));
                out.extend(quote! {
                    #[derive(
                        ::core::clone::Clone,
                        ::core::marker::Copy,
                        ::cubecl::prelude::CubeType,
                        ::cubecl::prelude::CubeLaunch,
                    )]
                    #extra
                    #s
                });
            }
            // A declared enum's place in the v1 subset is as a
            // `#[cube(comptime)]` field type or a `#[comptime]` parameter,
            // which keeps the plain host type and never reaches the device — so
            // it gets no `CubeType`/`CubeLaunch` derive (that would claim a
            // device representation vericl has no twin model for), but it DOES
            // get the host derives both of those positions require. Measured
            // (round-11 review): without them, the documented "unit enum
            // comptime FIELD" shape did not compile at all — the owning
            // struct's `Copy` failed on `E0204` and CubeCL's generated
            // `CompilationArg` could not `hash`/`clone` it.
            Item::Enum(e) => {
                let extra = comptime_derives(&e.attrs, true);
                out.extend(quote! {
                    #[derive(::core::clone::Clone, ::core::marker::Copy)]
                    #extra
                    #e
                });
            }
            other => out.extend(other.to_token_stream()),
        }
    }

    for s in &declared.structs {
        let name = &s.ident;
        out.extend(quote! {
            impl ::vericl::StructIdentity for #name {
                const STRUCT_HASH: &'static str = #hash;
            }
        });
        // `ConfigIdentity` — i.e. "this type may ALSO be a `#[comptime]`
        // parameter" — only where that is true. A float-field struct cannot be
        // one at any price (`f32` is neither `Hash` nor `Eq`), and claiming
        // otherwise is what round 11 caught: the impl was emitted
        // unconditionally while the type failed to compile in that position.
        // Withholding it lands the author on `ConfigIdentity`'s
        // `#[diagnostic::on_unimplemented]` note, which names this case.
        if comptime_usable.contains(&name.to_string()) {
            out.extend(quote! {
                impl ::vericl::ConfigIdentity for #name {
                    const CONFIG_HASH: &'static str = #hash;
                }
            });
        }
        out.extend(spec_items(s, &declared)?);
    }
    // A declared enum carries `ConfigIdentity` only (see `StructIdentity`'s
    // "Enums" section): it may be a `#[comptime]` parameter or a comptime field,
    // never a runtime parameter. A unit-only enum is always hashable, so this is
    // unconditional.
    for e in &file.items {
        let Item::Enum(e) = e else { continue };
        let name = &e.ident;
        out.extend(quote! {
            impl ::vericl::ConfigIdentity for #name {
                const CONFIG_HASH: &'static str = #hash;
            }
        });
    }

    // One `type Root__VericlSpec__a__b = Inner__VericlSpec;` per nested path, so
    // `#[vericl::kernel]` can name a nested field's spec type from the dotted
    // `gen(p.a.b in …)` clause alone — the only thing it knows.
    for a in nested_paths {
        let NestedAlias { alias, target, vis, comptime_whole } = a;
        if comptime_whole {
            // The field at this path is `#[cube(comptime)]`, so it is pinned
            // WHOLE (`instantiate(p.f = T { … })`) and has no per-sub-field
            // spec. Pointing the alias at `T__VericlSpec` — which is what this
            // loop used to do — made `gen(p.f.k in …)` a raw `E0308` between two
            // generated type names. A marker struct puts the reason in the type
            // NAME, so rustc's own `E0560: no field named` says it: "struct
            // `…__is_a_comptime_field_pinned_whole_by_instantiate` has no field
            // named `k`".
            let marker = format_ident!("{}__is_a_comptime_field_pinned_whole_by_instantiate", alias);
            out.extend(quote! {
                #[doc(hidden)]
                #[allow(non_camel_case_types)]
                #[derive(::core::clone::Clone, ::core::marker::Copy)]
                #vis struct #marker;
                #[doc(hidden)]
                #[allow(non_camel_case_types)]
                #vis type #alias = #marker;
            });
            continue;
        }
        out.extend(quote! {
            #[doc(hidden)]
            #[allow(non_camel_case_types)]
            #vis type #alias = #target;
        });
    }

    Ok(out)
}

/// The four `std` derives a type needs, on top of the `Clone`/`Copy` this macro
/// already emits, before CubeCL will accept it in `#[comptime]` position.
///
/// Measured (round-11 review): a `cube_struct!` type with two `u32` fields used
/// as `#[comptime] c: IntCfg` failed to compile with `no method named 'hash'`,
/// `no method named 'eq'` and "doesn't implement `Debug`"; adding exactly these
/// four made it compile and run. That is the whole recipe — and it is also why
/// a float-field struct can never be one, since `f32` is neither `Hash` nor
/// `Eq`.
const COMPTIME_POSITION_DERIVES: &[&str] = &["Debug", "PartialEq", "Eq", "Hash"];

/// The subset of [`COMPTIME_POSITION_DERIVES`] to emit for an item, as a
/// `#[derive(...)]` attribute (empty when `usable` is false, or when the author
/// wrote them all).
///
/// Author-written entries are skipped rather than rejected: unlike
/// `CubeType`/`CubeLaunch`, these are `std` derives whose expansion is fixed, so
/// "the macro owns them" buys nothing and a duplicate would be a confusing
/// `E0119` for code the author was right to write.
fn comptime_derives(attrs: &[syn::Attribute], usable: bool) -> TokenStream2 {
    if !usable {
        return TokenStream2::new();
    }
    let written: HashSet<String> =
        crate::decl_block::derive_paths(attrs).iter().map(render_path).collect();
    let paths: Vec<syn::Path> = COMPTIME_POSITION_DERIVES
        .iter()
        .filter(|d| !written.contains(**d))
        .map(|d| match *d {
            "Debug" => syn::parse_quote!(::core::fmt::Debug),
            "PartialEq" => syn::parse_quote!(::core::cmp::PartialEq),
            "Eq" => syn::parse_quote!(::core::cmp::Eq),
            _ => syn::parse_quote!(::core::hash::Hash),
        })
        .collect();
    if paths.is_empty() {
        return TokenStream2::new();
    }
    quote!(#[derive(#(#paths),*)])
}

/// The declared structs that can serve **both** parameter positions — every
/// field of which admits `Hash`/`Eq` (design §6, corrected in round 11).
///
/// A struct qualifies iff every field's type is an integer/`bool`/`char`, a
/// declared unit enum, or another qualifying declared struct. A single `f32` or
/// `f64` field anywhere in its transitive shape disqualifies it, runtime or
/// comptime: CubeCL's generated `<Name>CompilationArg` derives `Hash`/`Eq` and
/// a `#[comptime]` parameter is `Debug`-formatted, and no float type is any of
/// those.
///
/// Computed as a fixpoint rather than one pass because a struct's answer depends
/// on its nested structs' answers, and `enumerate_nested_paths` (which is what
/// proves the graph acyclic) has not run yet. The iteration is monotone —
/// entries are only ever removed — so it terminates on any graph, cyclic or not.
fn comptime_usable_structs(declared: &Declared) -> HashSet<String> {
    let mut usable: HashSet<String> =
        declared.structs.iter().map(|s| s.ident.to_string()).collect();
    loop {
        let mut changed = false;
        for s in &declared.structs {
            let name = s.ident.to_string();
            if !usable.contains(&name) {
                continue;
            }
            let ok = s.fields.iter().all(|f| {
                let Type::Path(tp) = &f.ty else { return false };
                let Some(last) = tp.path.segments.last() else { return false };
                let t = last.ident.to_string();
                COMPTIME_FIELD_TYPES.contains(&t.as_str())
                    || declared.enum_names.contains(&t)
                    || usable.contains(&t)
            });
            if !ok {
                usable.remove(&name);
                changed = true;
            }
        }
        if !changed {
            return usable;
        }
    }
}

/// CS3/CS7/CS9 + the declared-name tables.
fn collect_declared(file: &syn::File, errors: &mut Vec<syn::Error>) -> Declared {
    let mut d = Declared::default();
    for item in &file.items {
        match item {
            Item::Struct(s) => {
                reject_generics(&s.generics, &s.ident, errors);
                let syn::Fields::Named(named) = &s.fields else {
                    // CS9.
                    errors.push(syn::Error::new(
                        s.fields.span(),
                        format!(
                            "`{}` must have named fields inside a vericl::cube_struct! block — a \
                             runtime struct parameter's whole contract surface is field-name-driven \
                             (`gen({}_param.field in lo..=hi)`, `instantiate({}_param.field = …)`, \
                             and the field-by-name reads the reference twin performs), and a tuple \
                             or unit struct has none of it. A unit struct additionally has no \
                             definition an edit could move, so declaring one here would record an \
                             identity that can never go stale",
                            s.ident, s.ident, s.ident
                        ),
                    ));
                    continue;
                };
                if named.named.is_empty() {
                    errors.push(syn::Error::new(
                        s.fields.span(),
                        format!(
                            "`{}` declares no fields — a vericl::cube_struct! type with no fields \
                             contributes nothing to a kernel's inputs and has no definition an \
                             edit could move, so its STRUCT_HASH would certify nothing",
                            s.ident
                        ),
                    ));
                }
                let mut fields = Vec::new();
                for f in &named.named {
                    let ident = f.ident.clone().expect("named fields have idents");
                    fields.push(DeclField {
                        ident,
                        ty: f.ty.clone(),
                        comptime: is_cube_comptime_field(&f.attrs),
                    });
                }
                d.struct_names.insert(s.ident.to_string());
                d.structs.push(DeclStruct {
                    ident: s.ident.clone(),
                    vis: s.vis.clone(),
                    fields,
                });
            }
            Item::Enum(e) => {
                reject_generics(&e.generics, &e.ident, errors);
                let mut unit_only = true;
                for v in &e.variants {
                    if !matches!(v.fields, syn::Fields::Unit) {
                        unit_only = false;
                        errors.push(syn::Error::new(
                            v.fields.span(),
                            format!(
                                "variant `{}::{}` carries a payload — an enum inside a \
                                 vericl::cube_struct! block is admissible only as a \
                                 `#[cube(comptime)]` FIELD type, where it is a plain host constant \
                                 that never reaches the device. A payload-carrying RUNTIME enum is \
                                 outside the vericl v0 subset: CubeCL lowers it to a tag plus \
                                 every variant's payload (`generate_runtime_enum.rs`) and the twin \
                                 would need a matching host discriminant model",
                                e.ident, v.ident
                            ),
                        ));
                    }
                }
                if unit_only {
                    d.enum_names.insert(e.ident.to_string());
                }
            }
            Item::Use(_) => {}
            // CS6, targeted: the ecosystem's dominant declaration spelling for
            // size/shape families is a macro (`define_3d_size_base!`), and its
            // tokens are in the block while the macro's DEFINITION is not.
            Item::Macro(m) => errors.push(syn::Error::new(
                m.mac.path.span(),
                "a macro invocation cannot declare a type inside vericl::cube_struct! — the \
                 invocation's tokens are hashed but the MACRO's definition is not, so an edit to \
                 the macro would change the struct's fields (and therefore the positional \
                 `<Name>Launch::new` this block emits) while leaving STRUCT_HASH and every \
                 kernel's recorded identity unmoved, and none of the field gates can walk an \
                 unexpanded macro. Write the declaration out inside the block",
            )),
            // CS7.
            other => errors.push(syn::Error::new(
                other.span(),
                "only `struct`, `enum` and `use` items are allowed inside vericl::cube_struct! — \
                 v1 declares FIELDS only. In particular an `impl` block is rejected: a `#[cube]` \
                 method's body runs as ordinary host Rust in the reference twin and as expanded \
                 device code in the kernel, and nothing reconciles the two (measured: editing such \
                 a method changed the twin from [3,6,9,12] to [4,5,6,7] with every recorded hash \
                 bit-identical, docs/design-cubetype-args.md §4.1). Write the operation as a \
                 `#[vericl::helper]` free function taking the struct — a helper's twin is \
                 generated from the same tokens the device gets, and its body is gated",
            )),
        }
    }
    d
}

/// `true` for a **bare** `#[cube(comptime)]` field attribute — the one `#[cube]`
/// spelling CS4 admits, because it is CubeCL's own field-level marker rather
/// than a body the twin would have to reproduce.
///
/// "Bare" is load-bearing and is checked structurally rather than by scanning
/// for the word: the path must be exactly `cube` (one segment, so a `x::cube`
/// that could resolve anywhere is not it) and its argument list must be exactly
/// the single path `comptime`. Anything else — `#[cube(comptime, launch)]`, a
/// key/value form, an unparseable argument list — is not this attribute and is
/// rejected by [`check_field_attrs`] rather than silently treated as it.
fn is_bare_cube_comptime(a: &syn::Attribute) -> bool {
    if !a.path().is_ident("cube") {
        return false;
    }
    let syn::Meta::List(list) = &a.meta else { return false };
    let Ok(nested) = list.parse_args_with(
        syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
    ) else {
        return false;
    };
    nested.len() == 1 && matches!(&nested[0], syn::Meta::Path(p) if p.is_ident("comptime"))
}

/// `true` if any of `attrs` is the bare `#[cube(comptime)]` field marker.
fn is_cube_comptime_field(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(is_bare_cube_comptime)
}

/// CS11's field half: a declared struct's FIELD may carry only the attributes
/// this macro classifies against — the bare `#[cube(comptime)]` marker and doc
/// comments. Every other attribute is rejected **by name**.
///
/// This mirrors the discipline `classify_param` already holds a kernel's own
/// parameters to (`#[comptime]` or nothing — any other parameter attribute is a
/// rejection, not a shrug), and for the same reason. Every downstream decision
/// about a field is made by reading this list by name: whether the field is
/// drawn per case or pinned by `instantiate(...)`, which positional slot it
/// takes in `<Name>Launch::new`, and — the sharp one — whether its *value* or
/// its `Default` enters the `CompilationArg` the extracted IR is built from. An
/// attribute the macro does not recognize is therefore either inert (and
/// misleading, since the block's hash records it as if it mattered) or a
/// classification rustc and VeriCL disagree about.
///
/// An allowlist rather than a denylist is the whole point: the round-11 escape
/// was a *spelling* (`#[cfg_attr(all(), cube(comptime))]`) that no denylist
/// naming `cube` could have caught, and the next one would be another spelling.
fn check_field_attrs(file: &syn::File, errors: &mut Vec<syn::Error>) {
    for item in &file.items {
        let Item::Struct(s) = item else { continue };
        for f in s.fields.iter() {
            let fname = f
                .ident
                .as_ref()
                .map(|i| i.to_string())
                .unwrap_or_else(|| "<unnamed>".to_string());
            for a in &f.attrs {
                if a.path().is_ident("doc") || is_bare_cube_comptime(a) {
                    continue;
                }
                errors.push(syn::Error::new(
                    a.span(),
                    format!(
                        "`#[{}]` on `{}.{fname}` is outside the vericl v1 subset — a \
                         vericl::cube_struct! field may carry only the bare `#[cube(comptime)]` \
                         marker and doc comments, and every other attribute is rejected BY NAME. \
                         The field's attribute list is what decides whether the field is drawn per \
                         case or pinned by instantiate(...), which positional slot it takes in \
                         `{}Launch::new`, and whether its VALUE or its `Default` enters the \
                         `CompilationArg` the extracted IR (and therefore every proved obligation) \
                         is built from — so an attribute vericl does not recognize is either inert \
                         and misleading, or a classification vericl and rustc would disagree \
                         about. This is the same whitelist a `#[comptime]` KERNEL parameter is \
                         held to",
                        render_path(a.path()),
                        s.ident,
                        s.ident,
                    ),
                ));
            }
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
            "a generic vericl cube struct (`{name}<…>`) is outside the vericl v1 subset — one \
             `vericl::cube_struct!` block hashes to one STRUCT_HASH, so every instantiation of \
             `{name}<…>` would carry the SAME identity and a change reachable only through one \
             type argument would be invisible to kernel identity. There is a second, independent \
             reason measured in CubeCL itself: `CubeLaunch`'s generated `CompilationArg` type is \
             emitted with the struct's plain generics and WITHOUT the launch `where` clause, so a \
             generic scalar field does not compile under the natural bound \
             (docs/design-cubetype-args.md §1.1). Declare the concrete shapes you launch (the \
             KERNEL's own generics are unaffected)"
        ),
    ));
}

/// CS2: a field's type must be a launch scalar, or a struct declared in **this**
/// block; a `#[cube(comptime)]` field admits the integer/bool/char set and a
/// unit-only enum declared in this block.
fn check_field_types(declared: &Declared, errors: &mut Vec<syn::Error>) {
    for s in &declared.structs {
        for f in &s.fields {
            if field_type_ok(&f.ty, f.comptime, declared) {
                continue;
            }
            let owner = &s.ident;
            let fname = &f.ident;
            let ty_txt = f.ty.to_token_stream().to_string();
            // The buffer-valued deferral keeps its own diagnosis even when
            // written qualified (`cubecl::prelude::Array<f32>`) — that author
            // wants to know about the deferral, not about path resolution.
            let msg = if is_buffer_valued_type(&f.ty) {
                format!(
                    "a vericl cube struct field must be a scalar (f32/f64/u32/i32/u64/i64), a \
                     struct declared in this same block, or a `#[cube(comptime)]` \
                     integer/bool/char or unit enum — `{owner}.{fname}: {ty_txt}` is a \
                     buffer-valued field, which is DEFERRED. It lowers to its own kernel binding \
                     (measured: one buffer per array field, flattened in place at the struct's \
                     parameter slot, `&mut` making every field ReadWrite, an unread field still \
                     bound — docs/design-cubetype-args.md §2.4, X6–X8), so supporting it needs \
                     four things that do not exist yet: a generated twin mirror type holding \
                     `&[T]`, a per-field entry in the compared-buffer set, per-field compare-tier \
                     selection, and a `gen(len(p.{fname} = N))` form. Pass the buffer as its own \
                     `&Array<T>` / `&mut Array<T>` kernel parameter instead"
                )
            } else if is_qualified_path_type(&f.ty) {
                format!(
                    "`{owner}.{fname}: {ty_txt}` names a QUALIFIED path, which is outside the \
                     vericl v1 subset — CS2 resolves a field type by the name of its FINAL \
                     SEGMENT, and a qualified path's final segment says nothing about what the \
                     path resolves to. Measured (round-11 review): with \
                     `use crate::shady as sm;` in scope and `mod shady {{ pub type u32 = f32; }}`, \
                     `{fname}: sm::u32` passes the scalar check as a `u32` while being an `f32` — \
                     and the same tail trick makes `other::Inner` pass as a struct \"declared in \
                     this block\" when it is a different type entirely. CS8 cannot catch this: it \
                     rejects a `use … as u32;` that rebinds the NAME the gate reads, and here the \
                     rebound name is the module (`sm`), not the tail. Write the unqualified name \
                     (`f32`/`f64`/`u32`/`i32`/`u64`/`i64`, or a struct declared in THIS block)"
                )
            } else if f.comptime {
                format!(
                    "`{owner}.{fname}: {ty_txt}` is not a type a `#[cube(comptime)]` vericl cube \
                     struct field may have — it must be an integer (u8..u128, i8..i128, \
                     usize/isize), `bool`, `char`, or a unit-only enum declared in THIS SAME \
                     vericl::cube_struct! block. No float: CubeCL's generated \
                     `<Name>CompilationArg` derives `Hash`/`Eq` over every comptime field and \
                     `f32` is neither, so a float comptime field does not compile (measured, \
                     docs/design-cubetype-args.md §1.2). A type declared in a DIFFERENT block \
                     would contribute meaning to the kernel without contributing to this block's \
                     STRUCT_HASH, which is the identity hole this macro exists to close"
                )
            } else {
                format!(
                    "`{owner}.{fname}: {ty_txt}` is not a type a runtime vericl cube struct field \
                     may have — it must be a launch scalar (f32/f64/u32/i32/u64/i64) or a struct \
                     declared in THIS SAME vericl::cube_struct! block. `usize`/`bool` are \
                     admissible as `#[cube(comptime)]` fields but not as runtime ones (vericl \
                     generates a runtime field exactly as it generates a loose scalar parameter of \
                     that type, and it draws those six types); a struct declared in a DIFFERENT \
                     block would contribute meaning to the kernel without contributing to this \
                     block's STRUCT_HASH, which is the identity hole this macro exists to close"
                )
            };
            errors.push(syn::Error::new(f.ty.span(), msg));
        }
    }
}

/// `true` for the CubeCL container types whose rejection deserves the deferral
/// diagnosis rather than the generic one (design §10.5).
fn is_buffer_valued_type(ty: &Type) -> bool {
    let Type::Path(tp) = ty else { return false };
    let Some(last) = tp.path.segments.last() else { return false };
    matches!(
        last.ident.to_string().as_str(),
        "Array" | "Tensor" | "Slice" | "SliceMut" | "View" | "Sequence" | "SharedMemory" | "Line"
            | "Vector" | "VirtualLayout"
    )
}

/// `true` for a path type written with more than one segment (or a leading
/// `::`) — the shape CS2 must refuse because it resolves by the FINAL segment.
///
/// Round-11 review, MODERATE 4. `field_type_ok` and [`crate::NumKind::of`] both
/// key on `path.segments.last()`, so `sm::u32` is "a `u32`" to every gate and
/// every generated draw while resolving to whatever `sm` actually exports —
/// and CS8, which rejects a `use … as u32;`, sees only a `use … as sm;` here.
/// A single-segment requirement closes the whole family at the point of
/// resolution rather than chasing spellings at the point of import.
fn is_qualified_path_type(ty: &Type) -> bool {
    match ty {
        Type::Path(tp) => {
            tp.qself.is_some() || tp.path.leading_colon.is_some() || tp.path.segments.len() > 1
        }
        _ => false,
    }
}

fn field_type_ok(ty: &Type, comptime: bool, declared: &Declared) -> bool {
    let Type::Path(tp) = ty else { return false };
    if is_qualified_path_type(ty) {
        return false;
    }
    let Some(last) = tp.path.segments.last() else { return false };
    // No generic arguments: `Option<Foo>` / `Array<f32>` would need the
    // argument's own definition accounted for, which is the same hole one level
    // down (or the §10.5 deferral).
    if !matches!(last.arguments, syn::PathArguments::None) {
        return false;
    }
    let name = last.ident.to_string();
    if comptime {
        COMPTIME_FIELD_TYPES.contains(&name.as_str())
            || declared.enum_names.contains(&name)
            || declared.struct_names.contains(&name)
    } else {
        RUNTIME_FIELD_TYPES.contains(&name.as_str()) || declared.struct_names.contains(&name)
    }
}

/// CS4: reject `#[cube]` anywhere in the block except the `#[cube(comptime)]`
/// FIELD attribute, which is CubeCL's own marker and carries no body.
fn check_no_cube_attr(file: &syn::File, errors: &mut Vec<syn::Error>) {
    struct CubeAttrCheck<'a> {
        errors: &'a mut Vec<syn::Error>,
    }
    impl CubeAttrCheck<'_> {
        fn check(&mut self, attrs: &[syn::Attribute], on_field: bool) {
            for a in attrs {
                if a.path().segments.last().is_none_or(|s| s.ident != "cube") {
                    continue;
                }
                if on_field && is_cube_comptime_field(std::slice::from_ref(a)) {
                    continue;
                }
                self.errors.push(syn::Error::new(
                    a.span(),
                    "a `#[cube]` attribute inside a vericl::cube_struct! block is outside the \
                     vericl v0 subset — the only admitted spelling is `#[cube(comptime)]` on a \
                     FIELD. A `#[cube]` method's body runs as ordinary host Rust in the reference \
                     twin and as expanded device code in the kernel, and nothing reconciles the \
                     two (measured: editing such a method changed the twin from [3,6,9,12] to \
                     [4,5,6,7] with every recorded hash bit-identical, \
                     docs/design-cubetype-args.md §4.1). Write the operation as a \
                     `#[vericl::helper]` free function taking the struct — a helper's twin is \
                     generated from the same tokens the device gets, and its body is gated",
                ));
            }
        }
    }
    let mut c = CubeAttrCheck { errors };
    for item in &file.items {
        match item {
            Item::Struct(s) => {
                c.check(&s.attrs, false);
                for f in s.fields.iter() {
                    c.check(&f.attrs, true);
                }
            }
            Item::Enum(e) => {
                c.check(&e.attrs, false);
                for v in &e.variants {
                    c.check(&v.attrs, false);
                    for f in v.fields.iter() {
                        c.check(&f.attrs, false);
                    }
                }
            }
            Item::Use(u) => c.check(&u.attrs, false),
            _ => {}
        }
    }
}

/// CS5.
fn check_block_derives(file: &syn::File, errors: &mut Vec<syn::Error>) {
    const MACRO_OWNED: &[(&str, &str)] = &[
        (
            "CubeType",
            "`#[derive(CubeType)]` is emitted by vericl::cube_struct! itself and must not be \
             written here — owning the derive set is what closes the derive-set escape: dropping \
             `CubeLaunch` (or adding a third derive) would change the type from launchable to \
             device-local with every kernel's tokens, and every kernel's SOURCE_HASH, unchanged. \
             Remove the derive",
        ),
        (
            "CubeLaunch",
            "`#[derive(CubeLaunch)]` is emitted by vericl::cube_struct! itself and must not be \
             written here — owning the derive set is what closes the derive-set escape. Remove the \
             derive",
        ),
        (
            "Clone",
            "`#[derive(Clone)]` is emitted by vericl::cube_struct! itself and must not be written \
             here — the generated reference twin binds a runtime struct parameter BY VALUE and the \
             harness both predicates on it and launches with it, so vericl derives `Clone`/`Copy` \
             rather than leaving generated code to fail on a missing one. Remove the derive",
        ),
        (
            "Copy",
            "`#[derive(Copy)]` is emitted by vericl::cube_struct! itself and must not be written \
             here — see the `Clone` diagnosis. Remove the derive",
        ),
    ];
    for item in &file.items {
        let attrs = match item {
            Item::Struct(s) => &s.attrs,
            Item::Enum(e) => &e.attrs,
            _ => continue,
        };
        check_derives(attrs, "vericl::cube_struct!", MACRO_OWNED, errors);
    }
}

/// CS6: a macro invocation anywhere in the block (including inside an
/// attribute-position expression) is opaque to CS2–CS5.
fn check_no_macros(file: &syn::File, errors: &mut Vec<syn::Error>) {
    struct MacroCheck<'a> {
        errors: &'a mut Vec<syn::Error>,
    }
    impl<'ast> Visit<'ast> for MacroCheck<'_> {
        fn visit_macro(&mut self, i: &'ast syn::Macro) {
            self.errors.push(syn::Error::new(
                i.span(),
                format!(
                    "a macro invocation (`{}!`) inside a vericl::cube_struct! block is outside the \
                     vericl v1 subset — a macro's tokens are opaque to the field gates above, so \
                     admitting one would make STRUCT_HASH cover text vericl never inspected. Write \
                     the declaration out",
                    render_path(&i.path)
                ),
            ));
        }
    }
    // `Item::Macro` is already diagnosed with a targeted message in
    // `collect_declared`; this catches every other position (a `macro!()` in a
    // const-generic argument, an attribute's tokens, an enum discriminant).
    for item in &file.items {
        if matches!(item, Item::Macro(_)) {
            continue;
        }
        MacroCheck { errors }.visit_item(item);
    }
}

/// The spec type name for a declared struct — `Uniform` -> `Uniform__VericlSpec`.
fn spec_ident(name: &Ident) -> Ident {
    format_ident!("{}{}", name, SPEC_SUFFIX)
}

/// The spec type path for a parameter's written type — the mirror image of
/// [`spec_ident`] on the kernel side, where only the parameter's type path is
/// known. `my_mod::Uniform` -> `my_mod::Uniform__VericlSpec`.
pub(crate) fn spec_type_path(ty: &Type) -> Option<syn::Path> {
    let Type::Path(tp) = ty else { return None };
    if tp.qself.is_some() {
        return None;
    }
    let mut p = tp.path.clone();
    let last = p.segments.last_mut()?;
    if !matches!(last.arguments, syn::PathArguments::None) {
        return None;
    }
    last.ident = spec_ident(&last.ident);
    Some(p)
}

/// The largest number of nested spec aliases one `vericl::cube_struct!` block
/// may expand to — CS10's *size* half (round-11 review, LOW 7).
///
/// [`enumerate_nested_paths`] walks graph PATHS, not types, because that is what
/// a dotted `gen(p.a.b in …)` clause names. CS10 already refuses a cyclic graph
/// (infinite), but an acyclic one is still exponential in its depth: eight
/// nested fields repeated eight deep is 16.7M aliases, each an emitted item.
/// 4096 is far beyond any declaration a person writes (the shipped examples
/// reach 2) and turns a hung compile into a diagnosis.
const MAX_NESTED_ALIASES: usize = 4096;

/// One emitted nested spec alias.
struct NestedAlias {
    /// `Root__VericlSpec__a__b` — the name `#[vericl::kernel]` forms from a
    /// dotted clause.
    alias: Ident,
    /// The nested struct's own spec type.
    target: Ident,
    vis: syn::Visibility,
    /// `true` when the field at this path is `#[cube(comptime)]`, i.e. pinned
    /// WHOLE by `instantiate(p.f = T { … })` rather than described field by
    /// field. Such a path has no per-sub-field spec, so the alias points at a
    /// marker type whose NAME is the diagnosis (see the emission site).
    comptime_whole: bool,
}

/// CS10 + the nested spec aliases: every path `Root.a.b…` through the declared
/// nesting graph.
#[allow(clippy::type_complexity)]
fn enumerate_nested_paths(declared: &Declared) -> syn::Result<Vec<NestedAlias>> {
    let mut out: Vec<NestedAlias> = Vec::new();
    for s in &declared.structs {
        let root = spec_ident(&s.ident);
        let mut stack: Vec<(Ident, String, Vec<String>)> =
            vec![(root.clone(), s.ident.to_string(), Vec::new())];
        while let Some((alias_so_far, ty_name, path)) = stack.pop() {
            let Some(cur) = declared.structs.iter().find(|d| d.ident == ty_name) else {
                continue;
            };
            for f in &cur.fields {
                let Type::Path(tp) = &f.ty else { continue };
                let Some(last) = tp.path.segments.last() else { continue };
                let fty = last.ident.to_string();
                if !declared.struct_names.contains(&fty) {
                    continue;
                }
                // CS10: a declared struct reachable from itself has no finite
                // value and no finite spec type.
                if path.contains(&fty) || s.ident == fty {
                    let mut cycle = vec![s.ident.to_string()];
                    cycle.extend(path.iter().cloned());
                    cycle.push(fty.clone());
                    return Err(syn::Error::new(
                        f.ty.span(),
                        format!(
                            "the declared nesting graph is cyclic ({}) — a vericl cube struct that \
                             contains itself (directly or through another declared struct) has no \
                             finite value, no finite launch constructor, and no finite generation \
                             spec. Break the cycle",
                            cycle.join(" -> ")
                        ),
                    ));
                }
                let alias = format_ident!("{}__{}", alias_so_far, f.ident);
                // CS10's size half: this is a GRAPH-path walk, so a perfectly
                // legal DAG can be exponential in its depth (`k` fields of the
                // same nested type at depth `d` is `k^d` aliases). Refuse with
                // a diagnosis rather than swallowing the compiler.
                if out.len() >= MAX_NESTED_ALIASES {
                    return Err(syn::Error::new(
                        f.ty.span(),
                        format!(
                            "this vericl::cube_struct! block's declared nesting graph expands to \
                             more than {MAX_NESTED_ALIASES} distinct field PATHS — one hidden spec \
                             alias is emitted per path (that is how `#[vericl::kernel]` resolves a \
                             dotted `gen(p.a.b in …)` clause from the parameter's type alone), and \
                             the count is the number of paths through the graph, not the number of \
                             types: `k` nested fields repeated at depth `d` is `k^d`. Flatten the \
                             declaration, or split it across blocks so each kernel names the shape \
                             it actually uses"
                        ),
                    ));
                }
                if f.comptime {
                    // A comptime struct field is pinned WHOLE — `instantiate(p.f
                    // = T { … })` — so there is no per-sub-field spec beneath
                    // it and no path to enumerate. Emitting the ordinary alias
                    // here was a live type mismatch (round 11, LOW 5): the alias
                    // named `T__VericlSpec` while the spec entry's type is `T`,
                    // so `gen(p.f.k in …)` produced a raw `E0308` about two
                    // generated type names. The marker alias below turns that
                    // into `E0560` naming the reason.
                    out.push(NestedAlias {
                        alias,
                        target: spec_ident(&last.ident),
                        vis: cur.vis.clone(),
                        comptime_whole: true,
                    });
                    continue;
                }
                out.push(NestedAlias {
                    alias: alias.clone(),
                    target: spec_ident(&last.ident),
                    vis: cur.vis.clone(),
                    comptime_whole: false,
                });
                let mut next_path = path.clone();
                next_path.push(fty.clone());
                stack.push((alias, last.ident.to_string(), next_path));
            }
        }
    }
    Ok(out)
}

/// The hidden spec type and its three generated members, for one declared
/// struct.
fn spec_items(s: &DeclStruct, declared: &Declared) -> syn::Result<TokenStream2> {
    let name = &s.ident;
    let vis = &s.vis;
    let spec = spec_ident(name);
    let launch = format_ident!("{}Launch", name);
    let comp = format_ident!("{}CompilationArg", name);

    let mut spec_fields: Vec<TokenStream2> = Vec::new();
    let mut draw_fields: Vec<TokenStream2> = Vec::new();
    let mut comp_fields: Vec<TokenStream2> = Vec::new();
    let mut launch_args: Vec<TokenStream2> = Vec::new();

    for f in &s.fields {
        let fname = &f.ident;
        let fty = &f.ty;
        if f.comptime {
            // A comptime field keeps its positional slot in `<Name>Launch::new`
            // but takes the PLAIN host type (design §1.2, probe X3). Its spec
            // entry is the `instantiate(p.f = …)` pinned value itself.
            spec_fields.push(quote!(#vis #fname: #fty));
            draw_fields.push(quote!(#fname: self.#fname));
            comp_fields.push(quote!(#fname: self.#fname));
            launch_args.push(quote!(__vericl_v.#fname));
            continue;
        }
        let fty_name =
            match fty {
                Type::Path(tp) => {
                    tp.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default()
                }
                _ => String::new(),
            };
        if declared.struct_names.contains(&fty_name) {
            let nested_spec = spec_ident(&format_ident!("{}", fty_name));
            spec_fields.push(quote!(#vis #fname: #nested_spec));
            draw_fields.push(quote!(#fname: self.#fname.__vericl_draw(__vericl_rng)));
            comp_fields.push(quote!(#fname: self.#fname.__vericl_compilation_arg()));
            launch_args
                .push(quote!(#nested_spec::__vericl_launch_arg::<R>(&__vericl_v.#fname)));
            continue;
        }
        // A runtime scalar field: one inclusive `gen(...)` range, drawn by
        // exactly the expression a loose scalar parameter of this type draws
        // with (`integer_draw_expr` / `next_f{32,64}_range`) — one point of
        // custody, so a struct field and a flat parameter are generated
        // identically.
        let Some(kind) = NumKind::of(fty) else {
            return Err(syn::Error::new(
                fty.span(),
                "internal error: CS2 admitted a runtime field type vericl cannot draw",
            ));
        };
        spec_fields.push(quote!(#vis #fname: (#fty, #fty)));
        let draw = if kind.is_float() {
            if kind == NumKind::F64 {
                quote!(__vericl_rng.next_f64_range(self.#fname.0 as f64, self.#fname.1 as f64) as #fty)
            } else {
                quote!(__vericl_rng.next_f32_range(self.#fname.0 as f32, self.#fname.1 as f32) as #fty)
            }
        } else {
            let lo: syn::Expr = syn::parse_quote!(self.#fname.0);
            let hi: syn::Expr = syn::parse_quote!(self.#fname.1);
            crate::integer_draw_expr(&quote!(#fty), kind, Some(&(lo, hi)))
        };
        draw_fields.push(quote!(#fname: #draw));
        comp_fields.push(quote!(#fname: ::core::default::Default::default()));
        launch_args.push(quote!(__vericl_v.#fname));
    }

    let spec_doc = format!(
        "VeriCL generation/launch spec for the runtime cube struct `{name}` — one inclusive \
         `gen(...)` range per runtime field and one `instantiate(...)` pinned value per \
         `#[cube(comptime)]` field, in DECLARED FIELD ORDER.\n\n\
         `#[vericl::kernel]` emits a literal of this type built from the dotted `gen(p.f in …)` / \
         `instantiate(p.c = …)` clauses the author wrote, so rustc's own struct-literal \
         exhaustiveness check is what enforces that every field has one: a missing range is \
         `E0063: missing field` naming it, a misspelled field is `E0560`. Generated code — do not \
         name this type by hand."
    );

    Ok(quote! {
        #[doc = #spec_doc]
        #[derive(::core::clone::Clone, ::core::marker::Copy)]
        #[allow(non_camel_case_types, non_snake_case)]
        #vis struct #spec {
            #(#spec_fields,)*
        }

        #[allow(non_camel_case_types, non_snake_case, clippy::all)]
        impl #spec {
            /// Draw one value of the declared struct from `__vericl_rng`, field
            /// by field in declaration order — the twin's input and the GPU's,
            /// from one draw.
            #[doc(hidden)]
            #vis fn __vericl_draw(&self, __vericl_rng: &mut ::vericl::SplitMix64) -> #name {
                #name { #(#draw_fields,)* }
            }

            /// The positional `<Name>Launch::new(…)`, emitted from the field
            /// order this block hashed (`docs/design-cubetype-args.md` §4.3):
            /// CubeCL fills a launch struct by position, so the constructor and
            /// the declaration must be written by the same reader. They are.
            #[doc(hidden)]
            #vis fn __vericl_launch_arg<R: ::cubecl::prelude::Runtime>(
                __vericl_v: &#name,
            ) -> #launch<R> {
                #launch::<R>::new( #(#launch_args,)* )
            }

            /// The `CompilationArg` for client-free IR extraction, HAND-BUILT
            /// field-wise rather than obtained by registering a launcher.
            /// `KernelLauncher::with_info`/`with_scope` route through
            /// `std::thread_local!` statics under `cubecl/std` and only
            /// `into_bindings` drains them, so a launcher fed via
            /// `LaunchArg::register` but never launched leaks its scalars into
            /// the next real launch (measured — design risk 8).
            #[doc(hidden)]
            #vis fn __vericl_compilation_arg(&self) -> #comp {
                #comp { #(#comp_fields,)* }
            }
        }
    })
}

/// Field-name keys of a declared struct's spec, exposed for unit tests.
#[cfg(test)]
fn spec_field_names(src: &str) -> Vec<String> {
    let file: syn::File = syn::parse_str(src).expect("valid items");
    let mut errors = Vec::new();
    let d = collect_declared(&file, &mut errors);
    d.structs[0].fields.iter().map(|f| f.ident.to_string()).collect()
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
        let marker = "STRUCT_HASH : & 'static str = \"";
        let i = out.find(marker).unwrap_or_else(|| panic!("no STRUCT_HASH in: {out}"));
        let rest = &out[i + marker.len()..];
        rest[..rest.find('"').expect("closing quote")].to_string()
    }

    const BASE: &str = r#"
        pub struct Uniform {
            pub lower_bound: f32,
            pub upper_bound: f32,
            #[cube(comptime)]
            pub inclusive: bool,
        }
    "#;

    /// M1 — the block's whole definition is the hash's input, and the FIELD
    /// ORDER is part of it. The reorder case is the design's §4.3 launch-side
    /// hazard: `<Name>Launch::new` is positional, so a definition-only swap of
    /// two same-typed fields changed the computed function with the kernel body
    /// and the launch-call text byte-unchanged (probe X2). Under this macro the
    /// constructor is re-emitted from the new order — so the computation stays
    /// correct — and the hash MUST move so the stored evidence goes stale.
    #[test]
    fn field_name_type_and_order_edits_all_move_struct_hash() {
        let base = hash_of(BASE);
        let renamed = hash_of(&BASE.replace("lower_bound", "lo_bound"));
        let retyped = hash_of(&BASE.replace("pub lower_bound: f32", "pub lower_bound: f64"));
        let reordered = hash_of(
            r#"
            pub struct Uniform {
                pub upper_bound: f32,
                pub lower_bound: f32,
                #[cube(comptime)]
                pub inclusive: bool,
            }
        "#,
        );
        let comptimed = hash_of(
            r#"
            pub struct Uniform {
                pub lower_bound: f32,
                #[cube(comptime)]
                pub upper_bound: u32,
                #[cube(comptime)]
                pub inclusive: bool,
            }
        "#,
        );
        assert_ne!(base, renamed, "a field RENAME must move STRUCT_HASH");
        assert_ne!(base, retyped, "a field TYPE change must move STRUCT_HASH");
        assert_ne!(base, reordered, "a field REORDER must move STRUCT_HASH (design §4.3)");
        assert_ne!(base, comptimed, "a comptime/runtime flip must move STRUCT_HASH");
    }

    /// Hash granularity, identical to a kernel's own `SOURCE_HASH`: whitespace
    /// and `//` comments do not move it, a doc comment does.
    #[test]
    fn struct_hash_granularity_is_token_level() {
        let spaced = BASE.replace("pub lower_bound: f32,", "pub lower_bound:\n\n    f32,\n");
        assert_eq!(hash_of(BASE), hash_of(&spaced), "whitespace must not move STRUCT_HASH");
        let commented = BASE.replace("pub struct Uniform", "// note\npub struct Uniform");
        assert_eq!(hash_of(BASE), hash_of(&commented), "a `//` comment must not move STRUCT_HASH");
        let documented = BASE.replace("pub struct Uniform", "/// docs\npub struct Uniform");
        assert_ne!(hash_of(BASE), hash_of(&documented), "a doc comment must move STRUCT_HASH");
    }

    /// "One type, both positions" (design §6) — as **corrected in round 11**.
    ///
    /// `STRUCT_HASH` and `CONFIG_HASH` are the same hash of the same block, but
    /// `ConfigIdentity` is emitted only where the type can genuinely occupy
    /// `#[comptime]` position: CubeCL `Debug`-formats a comptime parameter and
    /// derives `Hash`/`Eq` over it, and `f32` is none of those. The old code
    /// emitted the impl unconditionally, so `BASE` (two `f32` fields) advertised
    /// a capability that failed to compile — measured, three raw trait errors.
    #[test]
    fn config_identity_and_the_comptime_derives_track_field_hashability() {
        // Float fields: StructIdentity yes, ConfigIdentity NO, and none of the
        // four comptime-position derives.
        let float = ok(BASE);
        assert!(float.contains("impl :: vericl :: StructIdentity for Uniform"), "{float}");
        assert!(
            !float.contains("impl :: vericl :: ConfigIdentity for Uniform"),
            "a float-field struct cannot be a #[comptime] parameter at any price: {float}"
        );
        assert!(!float.contains("core :: hash :: Hash"), "{float}");

        // All-integer fields: both impls, and the four derives that make the
        // second one true rather than aspirational.
        let int_src = "pub struct IntCfg { pub m: u32, #[cube(comptime)] pub n: u32 }";
        let int = ok(int_src);
        assert!(int.contains("impl :: vericl :: StructIdentity for IntCfg"), "{int}");
        assert!(int.contains("impl :: vericl :: ConfigIdentity for IntCfg"), "{int}");
        for d in ["core :: fmt :: Debug", "core :: cmp :: PartialEq", "core :: cmp :: Eq", "core :: hash :: Hash"] {
            assert!(int.contains(d), "{d} missing: {int}");
        }
        // …and the two hashes are the same hash of the same block.
        let marker = "CONFIG_HASH : & 'static str = \"";
        let i = int.find(marker).expect("CONFIG_HASH");
        let rest = &int[i + marker.len()..];
        assert_eq!(hash_of(int_src), rest[..rest.find('"').unwrap()].to_string());

        // Transitivity, both ways: a float NESTED inside an otherwise-integer
        // struct disqualifies the outer one too.
        let nested_ok = ok("pub struct In { pub k: u32 } pub struct Out { pub i: In, pub j: i64 }");
        assert!(nested_ok.contains("impl :: vericl :: ConfigIdentity for Out"), "{nested_ok}");
        let nested_no = ok("pub struct In { pub k: f32 } pub struct Out { pub i: In, pub j: i64 }");
        assert!(!nested_no.contains("ConfigIdentity for Out"), "{nested_no}");
        assert!(!nested_no.contains("ConfigIdentity for In"), "{nested_no}");

        // An author-written std derive is not duplicated (that would be E0119).
        let written = ok("#[derive(Debug, Hash)] pub struct P { pub a: u32 }");
        assert_eq!(written.matches("Debug").count(), 1, "{written}");
        assert_eq!(written.matches("Hash").count(), 1, "{written}");
    }

    /// The macro owns the derive set (design §5.2 point 3) and emits the
    /// positional launch constructor in DECLARED order.
    #[test]
    fn macro_owns_the_derives_and_emits_the_positional_constructor() {
        let out = ok(BASE);
        assert!(out.contains("CubeType"), "{out}");
        assert!(out.contains("CubeLaunch"), "{out}");
        assert!(
            out.contains("UniformLaunch :: < R > :: new (__vericl_v . lower_bound , __vericl_v . upper_bound , __vericl_v . inclusive ,)"),
            "the launch constructor must be positional in declared field order: {out}"
        );
        // …and swapping the declaration swaps the constructor, which is why the
        // hazard becomes internal (and why the hash must move — tested above).
        let swapped = ok(&BASE.replace(
            "pub lower_bound: f32,\n            pub upper_bound: f32,",
            "pub upper_bound: f32,\n            pub lower_bound: f32,",
        ));
        assert!(
            swapped.contains("new (__vericl_v . upper_bound , __vericl_v . lower_bound"),
            "{swapped}"
        );
    }

    /// CS1.
    #[test]
    fn a_block_with_no_struct_is_rejected() {
        assert!(err("pub enum M { A, B }").contains("at least one struct"));
    }

    /// CS2 — the buffer-valued deferral gets its own diagnosis naming all four
    /// missing pieces (design R3 / risk 7: growing it requires deleting a list,
    /// not relaxing a check).
    #[test]
    fn buffer_valued_fields_are_rejected_with_the_deferral_diagnosis() {
        let e = err("pub struct B { pub a: Array<f32> }");
        assert!(e.contains("buffer-valued field"), "{e}");
        assert!(e.contains("twin mirror type"), "{e}");
        assert!(e.contains("compare-tier"), "{e}");
        assert!(e.contains("gen(len(p.a = N))"), "{e}");
        for t in ["Slice<f32>", "View<f32>", "Sequence<u32>", "SharedMemory<f32>", "Tensor<f32>"] {
            let m = err(&format!("pub struct B {{ pub a: {t} }}"));
            assert!(m.contains("buffer-valued field"), "{t}: {m}");
        }
    }

    /// CS2 — the runtime/comptime split, both directions, with the negative
    /// controls that make it non-vacuous.
    #[test]
    fn field_type_split_between_runtime_and_comptime_is_enforced() {
        ok("pub struct P { pub a: f32, pub b: f64, pub c: u32, pub d: i32, pub e: u64, pub f: i64 }");
        let u = err("pub struct P { pub a: usize }");
        assert!(u.contains("admissible as `#[cube(comptime)]` fields but not as runtime"), "{u}");
        ok("pub struct P { pub a: u32, #[cube(comptime)] pub b: usize, #[cube(comptime)] pub c: bool }");
        let f = err("pub struct P { pub a: u32, #[cube(comptime)] pub b: f32 }");
        assert!(f.contains("`Hash`/`Eq`"), "{f}");
        // A struct declared in a DIFFERENT block is not resolvable here.
        let o = err("pub struct P { pub inner: Other }");
        assert!(o.contains("declared in THIS SAME"), "{o}");
        ok("pub struct Inner { pub k: u32 } pub struct P { pub inner: Inner, pub s: f32 }");
    }

    /// CS3.
    #[test]
    fn a_generic_declared_struct_is_rejected() {
        let e = err("pub struct P<T> { pub a: T }");
        assert!(e.contains("generic vericl cube struct"), "{e}");
        assert!(e.contains("CompilationArg"), "the second, measured reason must be stated: {e}");
    }

    /// CS4 — an `impl` block and a `#[cube]` method, the two spellings of the
    /// measured divergence; with the `#[cube(comptime)]` FIELD negative control.
    #[test]
    fn impl_blocks_and_cube_attributes_are_rejected() {
        let e = err("pub struct P { pub a: u32 } impl P { pub fn f(&self) -> u32 { self.a } }");
        assert!(e.contains("only `struct`, `enum` and `use` items"), "{e}");
        assert!(e.contains("[3,6,9,12]"), "the measured divergence must be cited: {e}");
        let c = err("#[cube] pub struct P { pub a: u32 }");
        assert!(c.contains("`#[cube]` attribute inside a vericl::cube_struct! block"), "{c}");
        // The one admitted spelling.
        ok("pub struct P { pub a: u32, #[cube(comptime)] pub b: u32 }");
    }

    /// CS5 — the macro-owned derives are rejected when written, other custom
    /// derives get the generic diagnosis, std derives pass.
    #[test]
    fn macro_owned_and_custom_derives_are_rejected_std_derives_are_not() {
        for d in ["CubeType", "CubeLaunch", "Clone", "Copy"] {
            let e = err(&format!("#[derive({d})] pub struct P {{ pub a: u32 }}"));
            assert!(e.contains("emitted by vericl::cube_struct! itself"), "{d}: {e}");
        }
        let s = err("#[derive(serde::Serialize)] pub struct P { pub a: u32 }");
        assert!(s.contains("derive macro's own definition"), "{s}");
        ok("#[derive(Debug, PartialEq)] pub struct P { pub a: u32 }");
    }

    /// CS6.
    #[test]
    fn macro_invocations_are_rejected() {
        let e = err("define_sizes!(P); pub struct Q { pub a: u32 }");
        assert!(e.contains("macro invocation cannot declare a type"), "{e}");
    }

    /// CS7.
    #[test]
    fn disallowed_item_kinds_are_rejected() {
        for src in [
            "pub struct P { pub a: u32 } pub fn f() -> u32 { 1 }",
            "pub struct P { pub a: u32 } pub const K: u32 = 1;",
            "pub struct P { pub a: u32 } pub mod m { }",
            "pub struct P { pub a: u32 } pub trait T { }",
        ] {
            let e = err(src);
            assert!(e.contains("only `struct`, `enum` and `use` items"), "{src}: {e}");
        }
        ok("use core::u32 as _u32alias; pub struct P { pub a: u32 }");
    }

    /// CS8 — the shared round-10 root-rebinding gate (probe P5b), reached
    /// through `cube_struct!` rather than `config!`.
    #[test]
    fn rebinding_an_allowlisted_root_is_rejected() {
        let e = err("use crate::evil as u32; pub struct P { pub a: u32 }");
        assert!(e.contains("rebinds a path root"), "{e}");
        assert!(e.contains("vericl::cube_struct!"), "the message must name THIS macro: {e}");
        let g = err("use crate::evil::*; pub struct P { pub a: u32 }");
        assert!(g.contains("glob"), "{g}");
    }

    /// CS11, half one — the round-11 CRITICAL: the **classification split**.
    ///
    /// `#[cfg_attr(all(), cube(comptime))]` was accepted, and the field it
    /// carried was classified RUNTIME by vericl (a `(u32, u32)` spec entry,
    /// drawn per case, `Default::default()` in the `CompilationArg` the
    /// extracted IR is built from) and COMPTIME by CubeCL. The prover then
    /// discharges obligations against an IR where the field is `0` while the
    /// launched kernel runs with the drawn value — a false `Proved`. The
    /// reviewer's exploit declaration is pinned verbatim as the first case.
    #[test]
    fn cfg_attr_anywhere_in_the_block_is_rejected() {
        // The false-`Proved` exploit's own declaration block.
        let split = err(
            "pub struct Sneaky { pub a: f32, #[cfg_attr(all(), cube(comptime))] pub taps: u32 }",
        );
        assert!(split.contains("`#[cfg_attr(…)]`"), "{split}");
        assert!(split.contains("false `Proved`"), "the measured consequence must be named: {split}");
        assert!(split.contains("vericl::cube_struct!"), "the message must name THIS macro: {split}");

        // The CS4 bypass spelling…
        let cube = err("#[cfg_attr(all(), cube)] pub struct P { pub a: u32 }");
        assert!(cube.contains("`#[cfg_attr(…)]`"), "{cube}");
        // …and the CS5 one.
        let der = err("#[cfg_attr(all(), derive(serde::Serialize))] pub struct P { pub a: u32 }");
        assert!(der.contains("`#[cfg_attr(…)]`"), "{der}");
        // A `cfg_attr` on an enum, a variant and a `use` is caught too — the
        // gate is a whole-file walk, not an item-kind list.
        for src in [
            "#[cfg_attr(all(), derive(Evil))] pub enum M { A } pub struct P { pub a: u32 }",
            "pub enum M { #[cfg_attr(all(), doc = \"x\")] A } pub struct P { pub a: u32, #[cube(comptime)] pub m: M }",
            "#[cfg_attr(all(), allow(unused))] use core::u32 as _a; pub struct P { pub a: u32 }",
        ] {
            let e = err(src);
            assert!(e.contains("`#[cfg_attr(…)]`"), "{src}: {e}");
        }
    }

    /// CS11, half two — the FIELD-attribute whitelist, mirroring
    /// `classify_param`'s discipline: the bare `#[cube(comptime)]` and doc
    /// comments pass, everything else is rejected by name.
    ///
    /// The `#[cube(comptime, launch)]` case is the reason "bare" is checked
    /// structurally: the old recognizer scanned the argument list for the word
    /// `comptime` and would have classified that field comptime while handing
    /// CubeCL an attribute it means something else by.
    #[test]
    fn field_attributes_are_whitelisted_not_denylisted() {
        // Accepted: the marker, doc comments, and both together.
        ok("pub struct P { /// docs\n pub a: u32, #[cube(comptime)] /// more\n pub b: u32 }");
        for (attr, needle) in [
            ("#[serde(skip)]", "serde"),
            ("#[allow(dead_code)]", "allow"),
            ("#[cfg(feature = \"x\")]", "cfg"),
            ("#[repr(C)]", "repr"),
        ] {
            let e = err(&format!("pub struct P {{ pub a: u32, {attr} pub b: u32 }}"));
            assert!(e.contains("outside the vericl v1 subset"), "{attr}: {e}");
            assert!(e.contains(needle), "the rejection must name the attribute: {attr}: {e}");
            assert!(e.contains("P.b"), "the rejection must name the field: {attr}: {e}");
        }
        // A NON-BARE `#[cube(...)]` is not the marker: it falls through to CS4,
        // whose diagnosis is the more actionable one for a `#[cube]` spelling.
        // That it is not silently classified as the comptime marker is the
        // load-bearing half — `#[cube(comptime, launch)] pub b: f32` would
        // otherwise be admitted as a comptime field of a type CubeCL cannot
        // `Hash`.
        for attr in ["#[cube(comptime, launch)]", "#[cube]", "#[cube(comptime = true)]"] {
            let e = err(&format!("pub struct P {{ pub a: u32, {attr} pub b: f32 }}"));
            assert!(
                e.contains("the only admitted spelling is `#[cube(comptime)]` on a FIELD"),
                "{attr}: {e}"
            );
        }
        let non_bare: syn::ItemStruct =
            syn::parse_quote!(pub struct P { #[cube(comptime, launch)] pub b: f32 });
        assert!(
            !is_cube_comptime_field(&non_bare.fields.iter().next().expect("one field").attrs),
            "a non-bare `#[cube(...)]` must not be read as the comptime marker"
        );
    }

    /// CS8's derive-name half (round 11, co-reviewer): the derive gate admits a
    /// `#[derive(X)]` by comparing `X` to the `std` set BY NAME, exactly as the
    /// root gate compares a path root by name — so rebinding a `std` derive name
    /// is the same escape one namespace over.
    #[test]
    fn rebinding_a_std_derive_name_is_rejected() {
        for d in ["Hash", "Debug", "Default", "PartialEq"] {
            let e = err(&format!(
                "use crate::evil as {d}; #[derive({d})] pub struct P {{ pub a: u32 }}"
            ));
            assert!(e.contains("rebinds a DERIVE name"), "{d}: {e}");
            assert!(e.contains("vericl::cube_struct!"), "{d}: {e}");
        }
        // Negative control: a name that is neither a root, a primitive nor a
        // std derive is still importable.
        ok("use core::u32 as _u32alias; pub struct P { pub a: u32 }");
    }

    /// CS2's single-segment requirement (round 11, MODERATE 4) — the
    /// qualified-path aliasing escape CS8 structurally could not see.
    ///
    /// CS2 and `NumKind::of` both resolve a field type by its FINAL segment, so
    /// `sm::u32` was "a `u32`" to every gate and every generated draw while
    /// resolving to whatever `sm` exports; CS8 saw only `use crate::shady as
    /// sm;`, which binds no name any gate reads. Closed at the resolution point.
    #[test]
    fn qualified_field_type_paths_are_rejected() {
        for ty in ["sm::u32", "shady::f32", "crate::x::u32", "::core::primitive::u32"] {
            let e = err(&format!("pub struct P {{ pub a: {ty} }}"));
            assert!(e.contains("names a QUALIFIED path"), "{ty}: {e}");
            assert!(e.contains("FINAL SEGMENT"), "{ty}: {e}");
        }
        // The whole exploit as written, `use` and all.
        let full = err("use crate::shady as sm; pub struct P { pub a: sm::u32 }");
        assert!(full.contains("names a QUALIFIED path"), "{full}");
        // The cross-block struct spelling, which used to reach rustc as a raw
        // E0308 in generated code (the spec/draw types are built from THIS
        // block's `Inner`, the field is someone else's).
        let x = err("pub struct Inner { pub k: u32 } pub struct P { pub i: other::Inner }");
        assert!(x.contains("names a QUALIFIED path"), "{x}");
        assert!(x.contains("declared in this block"), "{x}");
        // A comptime field is covered by the same rule.
        let c = err("pub struct P { pub a: u32, #[cube(comptime)] pub b: sm::u32 }");
        assert!(c.contains("names a QUALIFIED path"), "{c}");
        // NEGATIVE CONTROL: the buffer-valued deferral keeps its own diagnosis
        // even when the author writes it qualified.
        let b = err("pub struct P { pub a: cubecl::prelude::Array<f32> }");
        assert!(b.contains("buffer-valued field"), "{b}");
        // …and unqualified scalars / block-declared structs still pass.
        ok("pub struct Inner { pub k: u32 } pub struct P { pub i: Inner, pub s: f32 }");
    }

    /// CS9.
    #[test]
    fn unit_and_tuple_structs_are_rejected() {
        let u = err("pub struct P;");
        assert!(u.contains("must have named fields"), "{u}");
        let t = err("pub struct P(pub u32);");
        assert!(t.contains("must have named fields"), "{t}");
        let e = err("pub struct P { }");
        assert!(e.contains("declares no fields"), "{e}");
    }

    /// CS10.
    #[test]
    fn a_cyclic_nesting_graph_is_rejected() {
        let e = err("pub struct A { pub b: B } pub struct B { pub a: A }");
        assert!(e.contains("cyclic"), "{e}");
        let s = err("pub struct A { pub a: A }");
        assert!(s.contains("cyclic"), "{s}");
    }

    /// An enum is re-emitted untouched, carries `ConfigIdentity` but NOT
    /// `StructIdentity` (so naming it in runtime parameter position lands on the
    /// trait's enum note), and a payload-carrying one is rejected outright.
    #[test]
    fn enums_are_comptime_only() {
        let out = ok("pub enum Mode { A, B } pub struct P { pub a: u32, #[cube(comptime)] pub m: Mode }");
        assert!(out.contains("impl :: vericl :: ConfigIdentity for Mode"), "{out}");
        assert!(!out.contains("StructIdentity for Mode"), "{out}");
        assert!(!out.contains("CubeType) ] pub enum"), "an enum must not get the derives: {out}");
        let e = err("pub enum Mode { A(u32) } pub struct P { pub a: u32 }");
        assert!(e.contains("carries a payload"), "{e}");
    }

    /// The nested spec aliases the kernel names a dotted `gen(p.a.b in …)`
    /// through — one per nested path, at every depth.
    #[test]
    fn nested_spec_aliases_are_emitted_per_path() {
        let out = ok(
            "pub struct Deep { pub k: u32 } \
             pub struct Inner { pub d: Deep, pub s: f32 } \
             pub struct Outer { pub i: Inner, pub t: f32 }",
        );
        assert!(out.contains("type Outer__VericlSpec__i = Inner__VericlSpec"), "{out}");
        assert!(out.contains("type Outer__VericlSpec__i__d = Deep__VericlSpec"), "{out}");
        assert!(out.contains("type Inner__VericlSpec__d = Deep__VericlSpec"), "{out}");
    }

    /// LOW 5 (round 11) — a `#[cube(comptime)]` field of a DECLARED struct type
    /// is pinned **whole** (`instantiate(p.win = Win { … })`), so there is no
    /// per-sub-field spec beneath it.
    ///
    /// The alias walk used to emit the ordinary `… = Win__VericlSpec` alias for
    /// that path anyway, while the spec entry's own type is `Win` — so a
    /// `gen(p.win.taps in …)` clause compiled into an assignment between two
    /// generated type names and surfaced as a raw `E0308`. The marker type puts
    /// the reason in the name, so rustc's own `E0560` states it.
    #[test]
    fn a_comptime_struct_field_is_pinned_whole_not_per_subfield() {
        let out = ok(
            "pub struct Win { pub taps: u32, pub stride: u32 } \
             pub struct Cfg { pub gain: f32, #[cube(comptime)] pub win: Win }",
        );
        // The spec entry is the VALUE type — that is what `instantiate` pins.
        assert!(out.contains("pub win : Win ,"), "{out}");
        // …so the path alias must not claim a nested SPEC.
        assert!(
            !out.contains("type Cfg__VericlSpec__win = Win__VericlSpec"),
            "the comptime path must not alias the nested spec: {out}"
        );
        assert!(
            out.contains(
                "type Cfg__VericlSpec__win = Cfg__VericlSpec__win__is_a_comptime_field_pinned_whole_by_instantiate"
            ),
            "{out}"
        );
        // NEGATIVE CONTROL: the same field as a RUNTIME field still gets the
        // ordinary nested alias, so the marker is about comptime-ness and not
        // about nesting.
        let runtime = ok(
            "pub struct Win { pub taps: u32, pub stride: u32 } \
             pub struct Cfg { pub gain: f32, pub win: Win }",
        );
        assert!(runtime.contains("type Cfg__VericlSpec__win = Win__VericlSpec"), "{runtime}");
    }

    /// CS10's size half (round 11, LOW 7): the alias walk enumerates graph
    /// PATHS, so an acyclic — therefore CS10-legal — declaration can still be
    /// exponential. Six levels of four nested fields is 5460 paths.
    #[test]
    fn the_nested_alias_walk_is_capped_with_a_diagnosis() {
        let mut src = String::from("pub struct L6 { pub k: u32 }");
        for lvl in (0..6).rev() {
            src.push_str(&format!(
                " pub struct L{lvl} {{ pub a: L{n}, pub b: L{n}, pub c: L{n}, pub d: L{n} }}",
                n = lvl + 1
            ));
        }
        let e = err(&src);
        assert!(e.contains("distinct field PATHS"), "{e}");
        assert!(e.contains("k^d"), "the growth law must be stated: {e}");
        // NEGATIVE CONTROL: the shipped shapes are nowhere near the cap.
        ok("pub struct D { pub k: u32 } pub struct I { pub d: D, pub s: f32 } \
            pub struct O { pub i: I, pub t: f32 }");
    }

    /// The spec type carries one entry per field, in declaration order.
    #[test]
    fn spec_fields_track_the_declaration() {
        assert_eq!(spec_field_names(BASE), ["lower_bound", "upper_bound", "inclusive"]);
        let out = ok(BASE);
        assert!(out.contains("pub lower_bound : (f32 , f32)"), "{out}");
        assert!(out.contains("pub inclusive : bool"), "a comptime field's spec entry is the pinned value: {out}");
    }
}

