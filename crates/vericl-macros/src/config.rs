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
//! | G4 | every call must resolve into the block, to `Self`, to a primitive-qualified path, or to `core`/`std`/`alloc` (design risk 2) | a free function defined *outside* the block is neither hashed nor gated — the `uses(...)` problem one level down |
//! | G5 | a declared config type may not be generic | `impl<S> ConfigIdentity for Cfg<S>` would give every instantiation the same hash, so a change in `S`'s own block would be invisible |
//! | G6 | every field/const type must be a scalar primitive, an array/tuple of those, or a type declared in **this** block (design §7) | a nested config declared in a *different* block would contribute its methods to the kernel's meaning without contributing to its hash |
//! | G7 | only `struct`/`enum`/`impl`/`trait`/`fn`/`const`/`use` items | a `static` (interior mutability), a `mod` (unhashed contents), or a `macro_rules!` re-opens the escape G4 closes |
//! | G8 | no macro invocation in a body | a macro's tokens are opaque to `syn`'s visitors, so `anything!(fma(a, b, c))` would evade G3 and G4 wholesale |
//! | G9 | every path *expression* must be a local, `self`/`Self`, a name declared in the block, or a primitive-/`core`-/`std`-qualified path | a bare `SOME_CONST` declared outside the block is a value the kernel's meaning depends on and the hash cannot see |
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
//! - and G4/G9 mean the *in-block* half cannot silently call into it: a config
//!   method may only call what the block declares, so reaching an out-of-block
//!   impl requires the author to write the call on the kernel side, in tokens
//!   `SOURCE_HASH` already covers.
//!
//! Two narrower residuals of the same family, stated rather than papered over:
//! a trait `impl` for a declared type written outside the block (including an
//! operator-trait impl, so `self.a + self.b` on config-typed fields), and the
//! bodies of `core`/`std` items G4/G9 allow (not user code, so not an identity
//! concern). See `docs/design-struct-comptime.md` §13 risk 3.

use std::collections::HashSet;

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use sha2::{Digest, Sha256};
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Expr, Ident, Item, Type};

use crate::FLOAT_METHOD_REJECT;

/// Scalar primitives a config field may have, and the path roots a config body
/// may qualify a call/read with. Deliberately closed: a config's fields are
/// integer/bool/enum-valued by CubeCL's own construction (a `#[cube(launch)]`
/// comptime type must be `Hash + Eq`, which `f32` is not — design §1.1), and
/// anything outside this set is either a type whose definition the hash cannot
/// see or a type whose values cannot be pinned.
const PRIMITIVE_TYPES: &[&str] = &[
    "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize", "bool",
    "char", "f32", "f64",
];

/// Path roots a config body may call through or read from besides the block's
/// own declarations: the standard library (whose bodies are not user code, so
/// not an identity concern) and `Self`.
const EXTERNAL_ROOTS: &[&str] = &["core", "std", "alloc", "Self"];

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
    check_no_cube_attr(&file, &mut errors);
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
    let mut hasher = Sha256::new();
    hasher.update(ts.to_string().as_bytes());
    let hash = format!("sha256:{:x}", hasher.finalize());

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
fn collect_declared(file: &syn::File, errors: &mut Vec<syn::Error>) -> Declared {
    let mut d = Declared::default();
    for item in &file.items {
        match item {
            Item::Struct(s) => {
                reject_generics(&s.generics, &s.ident, errors);
                d.types.insert(s.ident.to_string());
                d.config_types.push(s.ident.clone());
            }
            Item::Enum(e) => {
                reject_generics(&e.generics, &e.ident, errors);
                d.types.insert(e.ident.to_string());
                d.config_types.push(e.ident.clone());
            }
            Item::Trait(t) => {
                d.types.insert(t.ident.to_string());
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

/// G3/G4/G8/G9: gate every body the block declares.
fn gate_bodies(file: &syn::File, declared: &Declared, errors: &mut Vec<syn::Error>) {
    for item in &file.items {
        match item {
            Item::Fn(f) => gate_fn(&f.sig, &f.block, declared, errors),
            Item::Const(c) => gate_expr(&c.expr, &HashSet::new(), declared, errors),
            Item::Impl(ii) => {
                for it in &ii.items {
                    match it {
                        syn::ImplItem::Fn(f) => gate_fn(&f.sig, &f.block, declared, errors),
                        syn::ImplItem::Const(c) => {
                            gate_expr(&c.expr, &HashSet::new(), declared, errors)
                        }
                        _ => {}
                    }
                }
            }
            Item::Trait(t) => {
                for it in &t.items {
                    match it {
                        syn::TraitItem::Fn(f) => {
                            if let Some(b) = &f.default {
                                gate_fn(&f.sig, b, declared, errors);
                            }
                        }
                        syn::TraitItem::Const(c) => {
                            if let Some((_, e)) = &c.default {
                                gate_expr(e, &HashSet::new(), declared, errors);
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
    for arg in &sig.inputs {
        match arg {
            syn::FnArg::Receiver(_) => {
                locals.insert("self".to_string());
            }
            syn::FnArg::Typed(pt) => {
                if let syn::Pat::Ident(pi) = pt.pat.as_ref() {
                    locals.insert(pi.ident.to_string());
                }
            }
        }
    }
    let mut gate = BodyGate { locals: &locals, declared, errors };
    gate.visit_block(block);
}

fn gate_expr(
    expr: &Expr,
    locals: &HashSet<String>,
    declared: &Declared,
    errors: &mut Vec<syn::Error>,
) {
    let mut gate = BodyGate { locals, declared, errors };
    gate.visit_expr(expr);
}

struct BodyGate<'a> {
    locals: &'a HashSet<String>,
    declared: &'a Declared,
    errors: &'a mut Vec<syn::Error>,
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
        let ok = self.declared.types.contains(&first_s)
            || self.declared.values.contains(&first_s)
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
                 pin it with instantiate(...). Calls to `core`/`std`/`alloc`, to a primitive's \
                 associated functions (`u32::max`), to `Self`, and to anything this block \
                 declares are allowed"
            ),
        ));
    }

    /// G9: a path *expression* (a value read) must be a local, `self`/`Self`, a
    /// name this block declares, or a primitive-/std-qualified path. A bare
    /// `SOME_CONST` declared outside the block is a value the kernel's meaning
    /// depends on that CONFIG_HASH cannot see.
    fn check_value_path(&mut self, path: &syn::Path, span: proc_macro2::Span) {
        let Some(first) = path.segments.first() else { return };
        let first_s = first.ident.to_string();
        let ok = if path.segments.len() == 1 {
            self.locals.contains(&first_s)
                || first_s == "self"
                || first_s == "Self"
                || self.declared.values.contains(&first_s)
                || self.declared.types.contains(&first_s)
        } else {
            self.declared.types.contains(&first_s)
                || self.declared.values.contains(&first_s)
                || EXTERNAL_ROOTS.contains(&first_s.as_str())
                || PRIMITIVE_TYPES.contains(&first_s.as_str())
        };
        if !ok {
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
}

impl<'ast> Visit<'ast> for BodyGate<'_> {
    fn visit_expr_method_call(&mut self, i: &'ast syn::ExprMethodCall) {
        let _ = self.check_reject_list(&i.method);
        syn::visit::visit_expr_method_call(self, i);
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

fn render_path(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
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
    /// positive) is a declaration, not a call, and compiles. Calling one is
    /// still rejected — the gate is about calls, and `dot` is on the closed
    /// list because its host-callability is unverified wherever it resolves.
    #[test]
    fn a_config_method_named_dot_can_be_declared() {
        ok("pub struct C { pub m: u32 } impl C { pub fn dot(&self) -> u32 { self.m } }");
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
