//! Gate primitives shared by VeriCL's two **declaration-block** item macros,
//! `vericl::config! { … }` and `vericl::cube_struct! { … }`.
//!
//! Both macros exist for the same reason and are subject to the same class of
//! escape: they hash a block of tokens and then claim that the tokens they
//! hashed are the tokens that determine what the kernel computes. Every gate in
//! this module was measured as a **live escape** during the round-10 adversarial
//! review of `vericl::config!` (probes P5a/P5b/P7 and the derive audit), and the
//! escapes are not specific to configs — they are properties of "a proc macro
//! resolves names lexically, rustc resolves them globally". Factoring them here
//! rather than re-typing them in `cube_struct.rs` is deliberate: a gate that
//! exists in two copies is a gate that gets hardened in one copy.
//!
//! What is shared is exactly the *macro-agnostic* half:
//!
//! | item | gate | config | cube_struct |
//! |---|---|---|---|
//! | [`check_use_items`] | `use` may not rebind an allowlisted root, and may not glob | G12 | CS8 |
//! | [`derive_paths`] / [`STD_DERIVES`] | only `std` derives (a custom derive's *definition* is unhashed) | G11 | CS5 |
//! | [`PRIMITIVE_TYPES`] | the scalar names both blocks resolve field types against | G6 | CS2 |
//! | [`EXTERNAL_ROOTS`] | the path roots that are not user code | G4/G9 | CS8 |
//! | [`render_path`] | diagnostics | — | — |
//!
//! What is **not** shared is each macro's own subset decision: `config!`'s body
//! gates (G3/G4/G9/G10/G13) have no counterpart here because `cube_struct!`
//! declares fields only (CS4 rejects `impl` blocks outright), and
//! `cube_struct!`'s field-type rule is CubeCL's launch-scalar set rather than
//! config's host-primitive set. Sharing those would have meant one gate serving
//! two different subsets, which is how a gate stops meaning anything.

use proc_macro2::TokenStream as TokenStream2;
use quote::ToTokens;
use syn::spanned::Spanned;
use syn::{Ident, Item};

/// Scalar primitive names a declaration block resolves field types against, and
/// the path roots it treats as "not user code" when qualifying a call or read.
pub(crate) const PRIMITIVE_TYPES: &[&str] = &[
    "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize", "bool",
    "char", "f32", "f64",
];

/// Path roots a declaration block may name besides its own declarations: the
/// standard library, whose bodies are not user code and therefore not an
/// identity concern.
pub(crate) const EXTERNAL_ROOTS: &[&str] = &["core", "std", "alloc"];

/// The `std` derives a VeriCL declaration block admits, and the associated
/// items each one contributes (so `TileCfg::default()` / `self.clone()` resolve
/// against the block just like a hand-written item would).
///
/// A derive outside this set is rejected by both macros for the same reason: a
/// derive macro is a `proc_macro_derive`, so the *invocation* (`#[derive(Foo)]`)
/// is in the hashed tokens but `Foo`'s **definition** — which decides what
/// impls, methods and associated consts the type actually has — is not.
pub(crate) const STD_DERIVES: &[(&str, &[&str])] = &[
    ("Clone", &["clone", "clone_from"]),
    ("Copy", &[]),
    ("Debug", &["fmt"]),
    ("PartialEq", &["eq", "ne"]),
    ("Eq", &[]),
    ("Hash", &["hash"]),
    ("Default", &["default"]),
    ("PartialOrd", &["partial_cmp", "lt", "le", "gt", "ge"]),
    ("Ord", &["cmp", "min", "max", "clamp"]),
];

/// `true` if `name` is one of the [`STD_DERIVES`].
pub(crate) fn is_std_derive(name: &str) -> bool {
    STD_DERIVES.iter().any(|(n, _)| *n == name)
}

/// A comma-joined list of the admitted derive names, for diagnostics.
pub(crate) fn std_derive_list() -> String {
    STD_DERIVES.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
}

/// Every path named inside a `#[derive(...)]` attribute on `attrs`.
pub(crate) fn derive_paths(attrs: &[syn::Attribute]) -> Vec<syn::Path> {
    let mut out = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("derive") {
            continue;
        }
        let _ = attr.parse_nested_meta(|meta| {
            out.push(meta.path.clone());
            Ok(())
        });
    }
    out
}

/// `a::b::c` for a path, for diagnostics.
pub(crate) fn render_path(path: &syn::Path) -> String {
    path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>().join("::")
}

/// The shared root-rebinding gate — `vericl::config!`'s G12 and
/// `vericl::cube_struct!`'s CS8, one implementation.
///
/// A `use` inside a declaration block may not introduce a name the block's own
/// gates resolve as an allowlisted root, and may not be a glob (which can
/// introduce one invisibly).
///
/// Measured (round-10 review, probe P5b): `use crate::evil as core;` inside a
/// config block re-pointed G4/G9's `core` root at user code, and
/// `core::cmp::max(self.m, 1)` then evaluated to `self.m * 100` — a call into
/// unhashed, ungated code that every gate waved through because it *spelled*
/// `core`. The identical attack applies to `cube_struct!`, whose CS2 resolves a
/// field type by the name of its final segment: `use crate::evil as u32;` would
/// make an arbitrary type pass the scalar check.
///
/// `macro_name` names the macro in the diagnostic; `resolved_by` names *what*
/// resolves by name in that macro ("G4/G9 resolve a call/read", "CS2 resolves a
/// field type"), so the message stays specific in both.
pub(crate) fn check_use_items(
    file: &syn::File,
    macro_name: &str,
    resolved_by: &str,
    errors: &mut Vec<syn::Error>,
) {
    fn walk(
        tree: &syn::UseTree,
        macro_name: &str,
        resolved_by: &str,
        errors: &mut Vec<syn::Error>,
    ) {
        match tree {
            syn::UseTree::Path(p) => walk(&p.tree, macro_name, resolved_by, errors),
            syn::UseTree::Group(g) => {
                for t in &g.items {
                    walk(t, macro_name, resolved_by, errors);
                }
            }
            syn::UseTree::Name(n) => reject_bound_root(&n.ident, macro_name, resolved_by, errors),
            syn::UseTree::Rename(r) => {
                reject_bound_root(&r.rename, macro_name, resolved_by, errors)
            }
            syn::UseTree::Glob(g) => errors.push(syn::Error::new(
                g.star_token.span,
                format!(
                    "a glob `use …::*;` inside a {macro_name} block is outside the vericl v1 \
                     subset — {resolved_by} by the NAME of its path root (`core`, `std`, `alloc`, \
                     a primitive), and a glob can bind any of those names to user code without the \
                     block's tokens saying so. Import the items you need by name"
                ),
            )),
        }
    }
    fn reject_bound_root(
        name: &Ident,
        macro_name: &str,
        resolved_by: &str,
        errors: &mut Vec<syn::Error>,
    ) {
        let s = name.to_string();
        if EXTERNAL_ROOTS.contains(&s.as_str()) || PRIMITIVE_TYPES.contains(&s.as_str()) || s == "Self"
        {
            errors.push(syn::Error::new(
                name.span(),
                format!(
                    "a `use … as {s};` inside a {macro_name} block rebinds a path root that \
                     {resolved_by} BY NAME — after it, `{s}` in the block would reach code the \
                     block neither hashes nor gates while still spelling like the standard \
                     library. Import it under a different name"
                ),
            ));
        }
    }
    for item in &file.items {
        if let Item::Use(u) = item {
            walk(&u.tree, macro_name, resolved_by, errors);
        }
    }
}

/// Reject a custom `#[derive(...)]` on `attrs`, allowing [`STD_DERIVES`] plus
/// the macro's own `extra_allowed` names (which each macro emits itself and
/// therefore rejects when hand-written — see `cube_struct!`'s CS5).
///
/// `reject_written` names derives the macro emits on the author's behalf: those
/// get their own, more actionable message than "unknown derive".
pub(crate) fn check_derives(
    attrs: &[syn::Attribute],
    macro_name: &str,
    reject_written: &[(&str, &str)],
    errors: &mut Vec<syn::Error>,
) {
    for path in derive_paths(attrs) {
        let name = render_path(&path);
        let short = path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
        if let Some((_, why)) = reject_written.iter().find(|(n, _)| *n == short || *n == name) {
            errors.push(syn::Error::new(path.span(), (*why).to_string()));
            continue;
        }
        if is_std_derive(&name) {
            continue;
        }
        errors.push(syn::Error::new(
            path.span(),
            format!(
                "`#[derive({name})]` inside a {macro_name} block is outside the vericl v1 subset \
                 — {macro_name} hashes the block's tokens, and a custom derive's tokens are only \
                 its INVOCATION: the derive macro's own definition decides what impls and \
                 associated items the type has, so an edit there would change what the kernel \
                 computes while leaving the block's hash (and every kernel's recorded identity) \
                 unmoved, and none of the block's gates can walk code that does not exist until \
                 rustc expands it. This is the same reason a macro invocation cannot declare a \
                 type in the block. Allowed derives: {}",
                std_derive_list()
            ),
        ));
    }
}

/// A declaration block's SHA-256, over `TokenStream::to_string()`.
///
/// **Hash granularity**, identical for both macros and identical to a kernel's
/// own `SOURCE_HASH` (which hashes `ItemFn::to_token_stream().to_string()`):
/// whitespace, line breaks and ordinary `//` comments do **not** move it, while
/// any token change — including a doc comment, which tokenizes to a `#[doc]`
/// attribute — does. The two halves of a kernel's identity are therefore
/// sensitive to exactly the same class of edit.
pub(crate) fn block_hash(ts: &TokenStream2) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(ts.to_token_stream().to_string().as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}
