//! `#[vericl::kernel(...)]` — the contract attribute for CubeCL kernels.
//!
//! Placed *above* `#[cube(launch)]`, it re-emits the kernel untouched and
//! generates a sibling module `<name>_vericl` containing:
//!
//! - `SOURCE_HASH`: identity of the exact definition evidence binds to
//!   (kernel source tokens + contract + vericl version);
//! - `contract()`: static contract metadata;
//! - `check_assumes(...) -> bool`: the `assumes(...)` clauses as an
//!   executable predicate over the kernel's (read-only) parameters;
//! - `reference(..., num_threads)`: a sequential scalar twin derived from the
//!   same source tokens — `ABSOLUTE_POS` becomes a loop variable, `&Array<T>`
//!   becomes `&[T]`. This is the independent comparison: it shares only the
//!   source text with the CubeCL pipeline, not its IR or codegen.
//!
//! Kernels outside the supported v0 subset are rejected at compile time
//! rather than silently approximated.

use proc_macro::TokenStream;
use proc_macro2::{Group, Ident, TokenStream as TokenStream2, TokenTree};
use quote::{ToTokens, quote};
use sha2::{Digest, Sha256};
use syn::fold::Fold;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{
    BinOp, Expr, ExprBinary, FnArg, ItemFn, Meta, Pat, ReturnType, Token, Type, parse_macro_input,
};

/// Constructs outside the v0 subset. Encountering any of these idents in a
/// kernel body is a compile error: the reference twin cannot model them yet.
const BANNED_IDENTS: &[&str] = &[
    // topology other than ABSOLUTE_POS
    "ABSOLUTE_POS_X",
    "ABSOLUTE_POS_Y",
    "ABSOLUTE_POS_Z",
    "UNIT_POS",
    "UNIT_POS_X",
    "UNIT_POS_Y",
    "UNIT_POS_Z",
    "CUBE_POS",
    "CUBE_POS_X",
    "CUBE_POS_Y",
    "CUBE_POS_Z",
    "CUBE_DIM",
    "CUBE_DIM_X",
    "CUBE_DIM_Y",
    "CUBE_DIM_Z",
    "CUBE_COUNT",
    "CUBE_COUNT_X",
    "CUBE_COUNT_Y",
    "CUBE_COUNT_Z",
    // parallel / memory constructs the sequential twin cannot model
    "SharedMemory",
    "sync_cube",
    "sync_units",
    "sync_storage",
    // comptime and vectorization
    "comptime",
    "Vector",
    "Line",
    "Slice",
    // early exit changes meaning between per-thread and sequential execution
    "return",
];

const BANNED_PREFIXES: &[&str] = &["plane_", "Atomic"];

struct ContractSpec {
    assumes: Vec<Expr>,
    compare: TokenStream2,
    compare_desc: String,
    /// Whether the `wrapping` clause is declared, and the span to blame if
    /// the kernel turns out to be outside the subset it requires.
    wrapping: Option<proc_macro2::Span>,
}

fn parse_contract(attr: TokenStream2) -> syn::Result<ContractSpec> {
    let metas: Punctuated<Meta, Token![,]> =
        syn::parse::Parser::parse2(Punctuated::parse_terminated, attr)?;

    let mut assumes = Vec::new();
    let mut compare = quote!(::vericl::Compare::Exact);
    let mut compare_desc = "exact".to_string();
    let mut wrapping = None;

    for meta in metas {
        match &meta {
            Meta::List(list) if list.path.is_ident("assumes") => {
                let exprs: Punctuated<Expr, Token![,]> = list
                    .parse_args_with(Punctuated::parse_terminated)
                    .map_err(|e| {
                        syn::Error::new(list.span(), format!("assumes(...) expects comma-separated boolean expressions: {e}"))
                    })?;
                assumes.extend(exprs);
            }
            Meta::List(list) if list.path.is_ident("compare") => {
                let inner: Punctuated<Meta, Token![,]> = list
                    .parse_args_with(Punctuated::parse_terminated)
                    .map_err(|e| {
                        syn::Error::new(
                            list.span(),
                            format!(
                                "compare(...) expects `exact`, `max_ulp = N`, or \
                                 `abs = X[, rel = Y]`: {e}"
                            ),
                        )
                    })?;
                let mut abs: Option<Expr> = None;
                let mut rel: Option<Expr> = None;
                for m in &inner {
                    match m {
                        Meta::Path(p) if p.is_ident("exact") => {
                            compare = quote!(::vericl::Compare::Exact);
                            compare_desc = "exact".into();
                        }
                        Meta::NameValue(nv) if nv.path.is_ident("max_ulp") => {
                            let val = &nv.value;
                            compare = quote!(::vericl::Compare::MaxUlpF32(#val));
                            compare_desc = format!("f32 max_ulp={}", val.to_token_stream());
                        }
                        Meta::NameValue(nv) if nv.path.is_ident("abs") => {
                            abs = Some(nv.value.clone());
                        }
                        Meta::NameValue(nv) if nv.path.is_ident("rel") => {
                            rel = Some(nv.value.clone());
                        }
                        other => {
                            return Err(syn::Error::new(
                                other.span(),
                                "compare(...) expects `exact`, `max_ulp = N`, or `abs = X[, rel = Y]`",
                            ));
                        }
                    }
                }
                if abs.is_some() || rel.is_some() {
                    let a = abs.map_or(quote!(0.0f32), |e| e.to_token_stream());
                    let r = rel.map_or(quote!(0.0f32), |e| e.to_token_stream());
                    compare = quote!(::vericl::Compare::AbsRelF32 { abs: #a, rel: #r });
                    compare_desc = format!("f32 |e-a| <= {a} + {r}*|e|");
                }
            }
            Meta::Path(p) if p.is_ident("wrapping") => {
                wrapping = Some(p.span());
            }
            other => {
                return Err(syn::Error::new(
                    other.span(),
                    "expected `assumes(...)`, `compare(...)`, or `wrapping`",
                ));
            }
        }
    }

    Ok(ContractSpec {
        assumes,
        compare,
        compare_desc,
        wrapping,
    })
}

/// Walk a token stream: rewrite `ABSOLUTE_POS` to the sequential loop
/// variable and reject out-of-subset constructs.
fn transform_body(ts: TokenStream2, errors: &mut Vec<syn::Error>) -> TokenStream2 {
    ts.into_iter()
        .map(|tt| match tt {
            TokenTree::Ident(id) => {
                let s = id.to_string();
                if s == "ABSOLUTE_POS" {
                    return TokenTree::Ident(Ident::new("__vericl_abs_pos", id.span()));
                }
                if BANNED_IDENTS.contains(&s.as_str())
                    || BANNED_PREFIXES.iter().any(|p| s.starts_with(p))
                {
                    errors.push(syn::Error::new(
                        id.span(),
                        format!(
                            "`{s}` is outside the vericl v0 kernel subset; unsupported constructs \
                             are rejected rather than silently approximated (see README \
                             \"First release\")"
                        ),
                    ));
                }
                TokenTree::Ident(id)
            }
            TokenTree::Group(g) => {
                let inner = transform_body(g.stream(), errors);
                let mut ng = Group::new(g.delimiter(), inner);
                ng.set_span(g.span());
                TokenTree::Group(ng)
            }
            other => other,
        })
        .collect()
}

/// `true` for the integer scalar types the `wrapping` clause accepts.
/// Matched by trailing path segment so `u32`, `std::primitive::u32`, etc.
/// all count, mirroring how `elem_of_array` matches `Array` by last segment.
fn is_wrapping_integer_type(ty: &Type) -> bool {
    let Type::Path(tp) = ty else { return false };
    let Some(last) = tp.path.segments.last() else { return false };
    matches!(last.ident.to_string().as_str(), "u32" | "i32" | "u64" | "i64")
}

/// Rewrites the reference twin's checked (panic-on-overflow-in-debug)
/// integer arithmetic to its wrapping equivalents, matching WGSL's
/// wrap-on-overflow semantics. Applied only when the `wrapping` contract
/// clause is declared, and only to the derived twin — the `#[cube]` kernel
/// itself is re-emitted untouched, so its WGSL codegen is unaffected.
///
/// This fold is untyped (syn has no type information at macro-expansion
/// time), which is exactly why `wrapping` is rejected at compile time for
/// any kernel with a non-integer parameter (see the subset check in
/// `expand`): folding `+`/`-`/`*` to `wrapping_*` calls on a `f32` operand
/// would silently change floating-point kernel semantics rather than
/// approximate wrap-on-overflow, which is not a trade this macro makes
/// silently. There is no trybuild/compile-fail harness yet, so that
/// rejection is exercised by ordinary `#[should_panic]`-free `syn::Error`
/// plumbing and covered by hand rather than by a `compile_fail` doctest.
struct WrappingFold;

impl Fold for WrappingFold {
    fn fold_expr(&mut self, expr: Expr) -> Expr {
        // Fold children first (post-order) so nested binary ops are already
        // rewritten by the time we inspect the (possibly reconstructed) node.
        let expr = syn::fold::fold_expr(self, expr);
        let Expr::Binary(ExprBinary { left, op, right, .. }) = expr else {
            return expr;
        };
        match op {
            BinOp::Add(_) => syn::parse_quote!((#left).wrapping_add(#right)),
            BinOp::Sub(_) => syn::parse_quote!((#left).wrapping_sub(#right)),
            BinOp::Mul(_) => syn::parse_quote!((#left).wrapping_mul(#right)),
            BinOp::Shl(_) => syn::parse_quote!((#left).wrapping_shl((#right) as u32)),
            BinOp::Shr(_) => syn::parse_quote!((#left).wrapping_shr((#right) as u32)),
            BinOp::AddAssign(_) => syn::parse_quote!(#left = (#left).wrapping_add(#right)),
            BinOp::SubAssign(_) => syn::parse_quote!(#left = (#left).wrapping_sub(#right)),
            BinOp::MulAssign(_) => syn::parse_quote!(#left = (#left).wrapping_mul(#right)),
            BinOp::ShlAssign(_) => {
                syn::parse_quote!(#left = (#left).wrapping_shl((#right) as u32))
            }
            BinOp::ShrAssign(_) => {
                syn::parse_quote!(#left = (#left).wrapping_shr((#right) as u32))
            }
            other => Expr::Binary(ExprBinary {
                attrs: Vec::new(),
                left,
                op: other,
                right,
            }),
        }
    }
}

enum ParamKind {
    /// Plain scalar passed by value (f32, u32, i32, ...).
    Scalar(Type),
    /// `&Array<T>` — read-only buffer.
    ArrayRef(Type),
    /// `&mut Array<T>` — mutable buffer.
    ArrayMut(Type),
}

struct Param {
    name: Ident,
    kind: ParamKind,
}

fn classify_param(arg: &FnArg) -> syn::Result<Param> {
    let FnArg::Typed(pt) = arg else {
        return Err(syn::Error::new(arg.span(), "self parameters are not supported"));
    };
    if !pt.attrs.is_empty() {
        return Err(syn::Error::new(
            pt.span(),
            "parameter attributes (e.g. #[comptime]) are outside the vericl v0 subset",
        ));
    }
    let Pat::Ident(pi) = pt.pat.as_ref() else {
        return Err(syn::Error::new(pt.pat.span(), "expected a plain parameter name"));
    };
    let name = pi.ident.clone();

    match pt.ty.as_ref() {
        Type::Reference(r) => {
            let elem = elem_of_array(&r.elem).ok_or_else(|| {
                syn::Error::new(
                    r.span(),
                    "reference parameters must be &Array<T> or &mut Array<T> in the vericl v0 subset",
                )
            })?;
            if r.mutability.is_some() {
                Ok(Param { name, kind: ParamKind::ArrayMut(elem) })
            } else {
                Ok(Param { name, kind: ParamKind::ArrayRef(elem) })
            }
        }
        Type::Path(_) => Ok(Param {
            name,
            kind: ParamKind::Scalar(pt.ty.as_ref().clone()),
        }),
        other => Err(syn::Error::new(
            other.span(),
            "unsupported parameter type in the vericl v0 subset",
        )),
    }
}

/// If `ty` is `Array<T>` (with any path prefix), return `T`.
fn elem_of_array(ty: &Type) -> Option<Type> {
    let Type::Path(tp) = ty else { return None };
    let last = tp.path.segments.last()?;
    if last.ident != "Array" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(ab) = &last.arguments else {
        return None;
    };
    if ab.args.len() != 1 {
        return None;
    }
    match ab.args.first()? {
        syn::GenericArgument::Type(t) => Some(t.clone()),
        _ => None,
    }
}

/// Tidy `quote`'s token spacing for human-readable contract strings.
fn pretty(ts: &impl ToTokens) -> String {
    ts.to_token_stream()
        .to_string()
        .replace(" . ", ".")
        .replace(" (", "(")
        .replace("( ", "(")
        .replace(" )", ")")
        .replace(" ,", ",")
}

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr2: TokenStream2 = attr.into();
    let func = parse_macro_input!(item as ItemFn);

    match expand(attr2, &func) {
        Ok(generated) => {
            let mut out = func.to_token_stream();
            out.extend(generated);
            out.into()
        }
        Err(e) => {
            // Emit the original item so downstream code still sees the kernel,
            // plus the error.
            let mut out = func.to_token_stream();
            out.extend(e.to_compile_error());
            out.into()
        }
    }
}

fn expand(attr: TokenStream2, func: &ItemFn) -> syn::Result<TokenStream2> {
    let spec = parse_contract(attr.clone())?;

    // --- subset gates on the signature ---
    if !func.sig.generics.params.is_empty() || func.sig.generics.where_clause.is_some() {
        return Err(syn::Error::new(
            func.sig.generics.span(),
            "generic kernels are outside the vericl v0 subset",
        ));
    }
    if !matches!(func.sig.output, ReturnType::Default) {
        return Err(syn::Error::new(
            func.sig.output.span(),
            "kernels must not return a value",
        ));
    }

    let params: Vec<Param> = func
        .sig
        .inputs
        .iter()
        .map(classify_param)
        .collect::<syn::Result<_>>()?;

    // `wrapping` rewrites `+`/`-`/`*`/`<<`/`>>` untyped — syn has no type
    // information at macro-expansion time — so it must not be allowed to
    // touch float math. Every parameter must be an integer scalar or
    // integer Array.
    if spec.wrapping.is_some() {
        for p in &params {
            let (ok, ty_span) = match &p.kind {
                ParamKind::Scalar(ty) => (is_wrapping_integer_type(ty), ty.span()),
                ParamKind::ArrayRef(elem) | ParamKind::ArrayMut(elem) => {
                    (is_wrapping_integer_type(elem), elem.span())
                }
            };
            if !ok {
                return Err(syn::Error::new(
                    ty_span,
                    "`wrapping` is outside the vericl v0 subset for this kernel: every parameter \
                     must be an integer scalar or integer Array (u32/i32/u64/i64) when \
                     `wrapping` is declared — the fold is untyped and must not silently touch \
                     float math",
                ));
            }
        }
    }

    // --- derive the reference twin body ---
    let mut errors = Vec::new();
    let mut ref_body = transform_body(func.block.to_token_stream(), &mut errors);
    if let Some(combined) = errors.into_iter().reduce(|mut a, b| {
        a.combine(b);
        a
    }) {
        return Err(combined);
    }

    // `wrapping`: fold the already-ABSOLUTE_POS-rewritten twin body, and
    // ONLY the twin — the `#[cube]` kernel re-emitted above is untouched.
    if spec.wrapping.is_some() {
        let block: syn::Block = syn::parse2(ref_body.clone()).map_err(|e| {
            syn::Error::new(
                e.span(),
                format!(
                    "internal error deriving the `wrapping` reference twin: the rewritten body \
                     did not parse as a block ({e})"
                ),
            )
        })?;
        ref_body = WrappingFold.fold_block(block).to_token_stream();
    }

    // --- identity hash: source tokens + contract + vericl version ---
    let mut hasher = Sha256::new();
    hasher.update(func.to_token_stream().to_string().as_bytes());
    hasher.update(b"||contract:");
    hasher.update(attr.to_string().as_bytes());
    hasher.update(b"||vericl:");
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
    let hash = format!("sha256:{:x}", hasher.finalize());

    // --- generated signatures ---
    let fn_name = &func.sig.ident;
    let fn_name_str = fn_name.to_string();
    let mod_name = Ident::new(&format!("{fn_name}_vericl"), fn_name.span());
    let vis = &func.vis;

    let ref_params: Vec<TokenStream2> = params
        .iter()
        .map(|p| {
            let name = &p.name;
            match &p.kind {
                ParamKind::Scalar(ty) => quote!(#name: #ty),
                ParamKind::ArrayRef(elem) => quote!(#name: &[#elem]),
                ParamKind::ArrayMut(elem) => quote!(#name: &mut [#elem]),
            }
        })
        .collect();

    // assumes predicate sees every buffer read-only
    let pred_params: Vec<TokenStream2> = params
        .iter()
        .map(|p| {
            let name = &p.name;
            match &p.kind {
                ParamKind::Scalar(ty) => quote!(#name: #ty),
                ParamKind::ArrayRef(elem) | ParamKind::ArrayMut(elem) => quote!(#name: &[#elem]),
            }
        })
        .collect();

    let assume_exprs = &spec.assumes;
    let assume_strs: Vec<String> = spec.assumes.iter().map(pretty).collect();
    let compare = &spec.compare;
    let compare_desc = &spec.compare_desc;
    let wrapping = spec.wrapping.is_some();

    // --- structured assumes: recognized `assumes(...)` clause shapes,
    // exposed as data for the SMT bounds prover (vericl-ir) to bind buffer
    // Length variables from. Only `A.len() == B.len()` and
    // `A.len() == <int literal>` (either operand order) are recognized;
    // anything else stays string-only in `contract().assumes` and is simply
    // unavailable to the prover. That's sound by construction: fewer
    // constraints can only make an obligation harder to prove (Refuted or
    // OutOfSubset where a recognized clause would have proved), never cause
    // a false Proved.
    let array_param_names: Vec<String> = params
        .iter()
        .filter(|p| matches!(p.kind, ParamKind::ArrayRef(_) | ParamKind::ArrayMut(_)))
        .map(|p| p.name.to_string())
        .collect();
    let structured_assumes: Vec<TokenStream2> = spec
        .assumes
        .iter()
        .filter_map(|e| structured_assume_tokens(e, &array_param_names))
        .collect();

    // --- kernel_definition(): builds the CubeCL IR `KernelDefinition` with
    // zero client/runtime/device, per docs/prototypes/ir_extraction.rs and
    // docs/ir-research.md §1. `BUFFER_PARAMS` records, in the same
    // registration order, each array parameter's name and whether it's an
    // output — vericl-ir has no way to recover parameter names from the IR
    // alone (buffers are just `input(id)`/`output(id)`), and this is the
    // macro's single point of custody for that mapping (not hand-maintained
    // per kernel in the harness).
    let mut kd_stmts: Vec<TokenStream2> = Vec::new();
    let mut kd_call_args: Vec<TokenStream2> = Vec::new();
    let mut buffer_params: Vec<TokenStream2> = Vec::new();
    for p in &params {
        let name = &p.name;
        let name_str = name.to_string();
        match &p.kind {
            ParamKind::Scalar(ty) => {
                kd_stmts.push(quote! {
                    let #name = <#ty as ::cubecl::prelude::LaunchArg>::expand(
                        &::core::default::Default::default(),
                        &mut __vericl_builder,
                    );
                });
            }
            ParamKind::ArrayRef(elem) => {
                kd_stmts.push(quote! {
                    let #name = <::cubecl::prelude::Array<#elem> as ::cubecl::prelude::LaunchArg>::expand(
                        &::cubecl::prelude::ArrayCompilationArg { inplace: None },
                        &mut __vericl_builder,
                    );
                });
                buffer_params.push(quote!((#name_str, false)));
            }
            ParamKind::ArrayMut(elem) => {
                kd_stmts.push(quote! {
                    let #name = <::cubecl::prelude::Array<#elem> as ::cubecl::prelude::LaunchArg>::expand_output(
                        &::cubecl::prelude::ArrayCompilationArg { inplace: None },
                        &mut __vericl_builder,
                    );
                });
                buffer_params.push(quote!((#name_str, true)));
            }
        }
        kd_call_args.push(quote!(#name));
    }

    let doc = if wrapping {
        format!(
            "VeriCL-generated artifacts for kernel `{fn_name_str}` (compare: {compare_desc}, \
             wrapping).\n\n\
             The `reference` function is a sequential scalar twin derived from the same\n\
             source tokens as the CubeCL kernel; it shares no CubeCL machinery. Its integer\n\
             `+`/`-`/`*`/`<<`/`>>` (and compound-assign forms) are folded to `wrapping_*`\n\
             equivalents, matching WGSL's wrap-on-overflow semantics instead of Rust's\n\
             default checked/panicking behavior — declared via the `wrapping` contract clause."
        )
    } else {
        format!(
            "VeriCL-generated artifacts for kernel `{fn_name_str}` (compare: {compare_desc}).\n\n\
             The `reference` function is a sequential scalar twin derived from the same\n\
             source tokens as the CubeCL kernel; it shares no CubeCL machinery."
        )
    };

    Ok(quote! {
        #[doc = #doc]
        #[allow(non_snake_case, unused_variables, clippy::all)]
        #vis mod #mod_name {
            use super::*;

            /// Identity of the exact kernel definition + contract this
            /// module's artifacts belong to.
            pub const SOURCE_HASH: &str = #hash;

            pub fn contract() -> ::vericl::Contract {
                ::vericl::Contract {
                    kernel: #fn_name_str,
                    source_hash: SOURCE_HASH,
                    assumes: &[#(#assume_strs),*],
                    structured_assumes: &[#(#structured_assumes),*],
                    compare: #compare,
                    wrapping: #wrapping,
                }
            }

            /// The `assumes(...)` clauses as an executable predicate.
            pub fn check_assumes(#(#pred_params),*) -> bool {
                true #(&& (#assume_exprs))*
            }

            /// Sequential scalar reference execution over
            /// `ABSOLUTE_POS in 0..num_threads` — the same iteration space as
            /// the GPU dispatch, in deterministic ascending order.
            pub fn reference(#(#ref_params,)* num_threads: usize) {
                for __vericl_abs_pos in 0..num_threads #ref_body
            }

            /// Each array parameter's name and whether it's an output, in
            /// buffer-registration order — see `kernel_definition` below.
            pub const BUFFER_PARAMS: &[(&str, bool)] = &[#(#buffer_params),*];

            /// Build this kernel's CubeCL `KernelDefinition` (the IR) with no
            /// client/runtime/device involved — see
            /// docs/prototypes/ir_extraction.rs and docs/ir-research.md §1.
            pub fn kernel_definition() -> ::cubecl::prelude::KernelDefinition {
                let mut __vericl_builder = ::cubecl::prelude::KernelBuilder::default();
                __vericl_builder.runtime_properties(::core::default::Default::default());
                // Required: registers how usize/isize (ABSOLUTE_POS, .len(),
                // indices) map to concrete storage types; panics without it.
                ::cubecl::prelude::AddressType::U32.register(&mut __vericl_builder.scope);
                #(#kd_stmts)*
                #fn_name::expand(&mut __vericl_builder.scope, #(#kd_call_args),*);
                __vericl_builder.build(::cubecl::prelude::KernelSettings::default())
            }
        }
    })
}

/// Recognize `A.len() == B.len()` and `A.len() == <int literal>` (either
/// operand order) among the array parameters named in `array_params`, and
/// emit the matching `::vericl::StructuredAssume` constructor tokens.
/// Anything else returns `None` — see the call site's doc comment on why
/// that's sound rather than a silent loss of soundness.
fn structured_assume_tokens(expr: &Expr, array_params: &[String]) -> Option<TokenStream2> {
    let Expr::Binary(ExprBinary { left, op: BinOp::Eq(_), right, .. }) = expr else {
        return None;
    };
    let l_len = len_call_target(left, array_params);
    let r_len = len_call_target(right, array_params);
    match (l_len, r_len) {
        (Some(a), Some(b)) => Some(quote!(::vericl::StructuredAssume::LenEq { a: #a, b: #b })),
        (Some(a), None) => {
            let value = int_literal(right)?;
            Some(quote!(::vericl::StructuredAssume::LenEqConst { a: #a, value: #value }))
        }
        (None, Some(b)) => {
            let value = int_literal(left)?;
            Some(quote!(::vericl::StructuredAssume::LenEqConst { a: #b, value: #value }))
        }
        (None, None) => None,
    }
}

/// If `expr` is `<name>.len()` for a `name` in `array_params`, the name.
fn len_call_target(expr: &Expr, array_params: &[String]) -> Option<String> {
    let Expr::MethodCall(mc) = expr else { return None };
    if mc.method != "len" || !mc.args.is_empty() {
        return None;
    }
    let Expr::Path(p) = mc.receiver.as_ref() else { return None };
    let ident = p.path.get_ident()?.to_string();
    array_params.iter().find(|n| **n == ident).cloned()
}

/// If `expr` is an integer literal, its value.
fn int_literal(expr: &Expr) -> Option<u64> {
    let Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(li), .. }) = expr else { return None };
    li.base10_parse::<u64>().ok()
}
