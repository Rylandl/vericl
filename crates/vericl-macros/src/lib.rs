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

use std::collections::HashMap;

use proc_macro::TokenStream;
use proc_macro2::{Group, Ident, TokenStream as TokenStream2, TokenTree};
use quote::{ToTokens, format_ident, quote};
use sha2::{Digest, Sha256};
use syn::fold::Fold;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{
    BinOp, Expr, ExprBinary, ExprForLoop, ExprLoop, ExprRange, ExprWhile, FnArg, GenericParam,
    ItemFn, Meta, Pat, PatType, RangeLimits, ReturnType, Token, Type, parse_macro_input,
};

mod suite;

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
    // terminate!() is a per-lane early exit; outside #[cube] it expands to an
    // empty block, so a derived twin would silently fall through the guard
    // instead of ending the lane (latent soundness gap found in dogfooding).
    "terminate",
];

const BANNED_PREFIXES: &[&str] = &["plane_", "Atomic"];

/// `cubecl::prelude::Float`/`Numeric` trait method names empirically
/// verified as host-callable on `f32` — safe to appear in a reference
/// twin's body (after `instantiate(...)` substitutes the generic type
/// param to a concrete float). Each either has a real per-type host
/// implementation (e.g. `Float::new`) or shares a name with a std `f32`
/// inherent method, which — because Rust's method resolution always tries
/// inherent methods before trait methods, regardless of which traits are
/// `use`-imported — wins over the trait's `unexpanded!()`-panicking default
/// body. Verified in
/// `crates/vericl-examples/tests/float_method_whitelist.rs`, which calls
/// every entry on host and cross-checks it against `std`; see that file for
/// the verification method.
const FLOAT_METHOD_WHITELIST: &[&str] = &[
    "new",
    "from_int",
    "min_value",
    "max_value",
    "abs",
    "min",
    "max",
    "clamp",
    "floor",
    "ceil",
    "round",
    "trunc",
    "sqrt",
    "recip",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "ln",
    "powf",
    "powi",
    "hypot",
    "is_nan",
    "to_degrees",
    "to_radians",
];

/// `Float`/`Numeric` trait method names that exist in cubecl 0.10 but are
/// *not* on [`FLOAT_METHOD_WHITELIST`] above: `log1p`, `inverse_sqrt`,
/// `erf`, and `is_inf` are empirically confirmed to panic on a host call
/// (`Unexpanded Cube functions should not be called.` — no std-inherent
/// shadow and no manual per-type override exist for them); the rest
/// (`rhypot`, `magnitude`, `normalize`, `dot`, `mul_hi`, `saturating_add`,
/// `saturating_sub`, `from_int_128`, `from_vec`) are simply unverified.
///
/// `cast_from`/`reinterpret` are a related but distinct mechanism
/// (`cubecl::prelude::Cast`/`Reinterpret`, not `Float`/`Numeric`) found by
/// dogfooding against real kernels (`F::cast_from(index)` is a common
/// generic-index-to-float conversion idiom) — `Cast::cast_from`'s only impl
/// is a blanket `impl<P: CubePrimitive> Cast for P { fn cast_from(...) {
/// unexpanded!() } }`, so it panics on *every* type, unconditionally, with
/// no possible per-type override; same empirically for `reinterpret`. Both
/// rejected here for the same reason as the Float/Numeric names above.
///
/// Calling any of these in a reference twin body is rejected at macro time
/// rather than silently trusted — see `FloatMethodCheck`.
const FLOAT_METHOD_REJECT: &[&str] = &[
    "log1p",
    "inverse_sqrt",
    "erf",
    "is_inf",
    "rhypot",
    "magnitude",
    "normalize",
    "dot",
    "mul_hi",
    "saturating_add",
    "saturating_sub",
    "from_int_128",
    "from_vec",
    "cast_from",
    "reinterpret",
];

/// One entry inside a `gen(...)` contract clause: either a per-parameter
/// generation range (`name in lo..=hi`, applied elementwise for arrays) or a
/// length pin (`len(name = N)`) that fixes an array parameter's generated
/// length to a constant instead of the differential case size `n`.
enum GenEntry {
    Range { name: Ident, lo: Expr, hi: Expr },
    Len { name: Ident, value: Expr },
}

impl Parse for GenEntry {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        if name == "len" {
            let content;
            syn::parenthesized!(content in input);
            let target: Ident = content.parse().map_err(|e| {
                syn::Error::new(e.span(), format!("expected `len(name = N)`: {e}"))
            })?;
            content.parse::<Token![=]>()?;
            let value: Expr = content.parse()?;
            if !content.is_empty() {
                return Err(content.error("len(name = N) expects exactly one `name = N` entry"));
            }
            return Ok(GenEntry::Len { name: target, value });
        }
        input.parse::<Token![in]>().map_err(|e| {
            syn::Error::new(
                e.span(),
                format!("expected `{name} in lo..=hi` or `len({name} = N)`: {e}"),
            )
        })?;
        let range: Expr = input.parse()?;
        match range {
            Expr::Range(ExprRange {
                start: Some(lo),
                end: Some(hi),
                limits: RangeLimits::Closed(_),
                ..
            }) => Ok(GenEntry::Range { name, lo: *lo, hi: *hi }),
            other => Err(syn::Error::new(
                other.span(),
                "gen(...) ranges must be inclusive with both ends given: `name in lo..=hi`",
            )),
        }
    }
}

/// One `name = value` entry inside an `instantiate(...)` contract clause.
/// `value` is parsed as an `Expr` regardless of whether `name` turns out to
/// name a generic type parameter (where the expression must be a bare type
/// path, e.g. `f32`) or a `#[comptime]` parameter (where it's the concrete
/// literal/const value) — which one it is can only be resolved once the
/// kernel signature is known, in `expand`, not here.
struct InstantiateEntry {
    name: Ident,
    value: Expr,
}

impl Parse for InstantiateEntry {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![=]>().map_err(|e| {
            syn::Error::new(e.span(), format!("expected `{name} = <value>`: {e}"))
        })?;
        let value: Expr = input.parse()?;
        Ok(InstantiateEntry { name, value })
    }
}

struct ContractSpec {
    assumes: Vec<Expr>,
    compare: TokenStream2,
    compare_desc: String,
    /// Whether the `wrapping` clause is declared, and the span to blame if
    /// the kernel turns out to be outside the subset it requires.
    wrapping: Option<proc_macro2::Span>,
    /// `gen(...)` clause entries, in declared order (not necessarily
    /// parameter order — see `resolve_gen_plan`).
    gen_entries: Vec<GenEntry>,
    /// The `instantiate(...)` clause, if declared — v0 supports at most one
    /// per kernel (see `expand`'s gating: a kernel with generics/comptime
    /// params and no clause is an error, as is a clause on a kernel with
    /// neither).
    instantiate: Option<(proc_macro2::Span, Vec<InstantiateEntry>)>,
}

fn parse_contract(attr: TokenStream2) -> syn::Result<ContractSpec> {
    let metas: Punctuated<Meta, Token![,]> =
        syn::parse::Parser::parse2(Punctuated::parse_terminated, attr)?;

    let mut assumes = Vec::new();
    let mut compare = quote!(::vericl::Compare::Exact);
    let mut compare_desc = "exact".to_string();
    let mut wrapping = None;
    let mut gen_entries: Vec<GenEntry> = Vec::new();
    let mut instantiate: Option<(proc_macro2::Span, Vec<InstantiateEntry>)> = None;

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
            Meta::List(list) if list.path.is_ident("gen") => {
                let entries: Punctuated<GenEntry, Token![,]> = list
                    .parse_args_with(Punctuated::parse_terminated)
                    .map_err(|e| {
                        syn::Error::new(
                            list.span(),
                            format!(
                                "gen(...) expects `name in lo..=hi` range entries and/or \
                                 `len(name = N)` length pins: {e}"
                            ),
                        )
                    })?;
                gen_entries.extend(entries);
            }
            Meta::List(list) if list.path.is_ident("instantiate") => {
                if instantiate.is_some() {
                    return Err(syn::Error::new(
                        list.span(),
                        "duplicate instantiate(...) clause; vericl v0 supports exactly one \
                         instantiate(...) clause per kernel",
                    ));
                }
                let entries: Punctuated<InstantiateEntry, Token![,]> = list
                    .parse_args_with(Punctuated::parse_terminated)
                    .map_err(|e| {
                        syn::Error::new(
                            list.span(),
                            format!(
                                "instantiate(...) expects comma-separated `name = value` \
                                 entries, one per generic type parameter or #[comptime] \
                                 parameter: {e}"
                            ),
                        )
                    })?;
                instantiate = Some((list.span(), entries.into_iter().collect()));
            }
            other => {
                return Err(syn::Error::new(
                    other.span(),
                    "expected `assumes(...)`, `compare(...)`, `gen(...)`, `instantiate(...)`, or \
                     `wrapping`",
                ));
            }
        }
    }

    Ok(ContractSpec {
        assumes,
        compare,
        compare_desc,
        wrapping,
        gen_entries,
        instantiate,
    })
}

/// Token-wise substitution map from a generic type parameter's ident string
/// (e.g. `"F"`) to the concrete type's tokens (e.g. `f32`) declared by an
/// `instantiate(...)` clause. Empty for kernels with no clause — every
/// caller below is then a no-op pass-through, so non-generic kernels are
/// unaffected.
type GenericSubst = HashMap<String, TokenStream2>;

/// Walk a token stream, rewriting `ABSOLUTE_POS` to the sequential loop
/// variable, substituting any generic type parameter ident per `subst`
/// (`F` -> `f32`, token-wise — see the `instantiate(...)` design), and
/// rejecting out-of-subset constructs. Used for both the reference twin's
/// body and (via [`subst_type_tokens`]) its signature's parameter types, so
/// an instantiated kernel's twin never mentions the original generic ident.
fn transform_body(ts: TokenStream2, subst: &GenericSubst, errors: &mut Vec<syn::Error>) -> TokenStream2 {
    let mut out = TokenStream2::new();
    for tt in ts {
        match tt {
            TokenTree::Ident(id) => {
                let s = id.to_string();
                if s == "ABSOLUTE_POS" {
                    out.extend(std::iter::once(TokenTree::Ident(Ident::new(
                        "__vericl_abs_pos",
                        id.span(),
                    ))));
                    continue;
                }
                if let Some(replacement) = subst.get(&s) {
                    out.extend(replacement.clone());
                    continue;
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
                out.extend(std::iter::once(TokenTree::Ident(id)));
            }
            TokenTree::Group(g) => {
                let inner = transform_body(g.stream(), subst, errors);
                let mut ng = Group::new(g.delimiter(), inner);
                ng.set_span(g.span());
                out.extend(std::iter::once(TokenTree::Group(ng)));
            }
            other => out.extend(std::iter::once(other)),
        }
    }
    out
}

/// Token-wise generic substitution only (no ABSOLUTE_POS rewrite, no
/// banned-construct check) — used on parameter *types*, which never contain
/// either. Reparsed as a [`Type`] by the caller.
fn subst_type_tokens(ts: TokenStream2, subst: &GenericSubst) -> TokenStream2 {
    let mut out = TokenStream2::new();
    for tt in ts {
        match tt {
            TokenTree::Ident(id) => {
                let s = id.to_string();
                if let Some(replacement) = subst.get(&s) {
                    out.extend(replacement.clone());
                } else {
                    out.extend(std::iter::once(TokenTree::Ident(id)));
                }
            }
            TokenTree::Group(g) => {
                let inner = subst_type_tokens(g.stream(), subst);
                let mut ng = Group::new(g.delimiter(), inner);
                ng.set_span(g.span());
                out.extend(std::iter::once(TokenTree::Group(ng)));
            }
            other => out.extend(std::iter::once(other)),
        }
    }
    out
}

/// `true` for the integer scalar types the `wrapping` clause accepts.
/// Matched by trailing path segment so `u32`, `std::primitive::u32`, etc.
/// all count, mirroring how `elem_of_array` matches `Array` by last segment.
fn is_wrapping_integer_type(ty: &Type) -> bool {
    let Type::Path(tp) = ty else { return false };
    let Some(last) = tp.path.segments.last() else { return false };
    matches!(last.ident.to_string().as_str(), "u32" | "i32" | "u64" | "i64")
}

/// The scalar kinds `gen(...)` knows how to generate. v0 supports exactly
/// the float type used by every example (`f32`) and the integer types the
/// `wrapping` clause already recognizes (`u32`/`i32`/`u64`/`i64`) — matching
/// this project's convention of rejecting an unsupported subset explicitly
/// rather than silently approximating it. `f64` and other numeric types are
/// out of scope for v0 because `vericl::rng::SplitMix64` has no `f64`
/// generator to reuse honestly.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NumKind {
    F32,
    U32,
    I32,
    U64,
    I64,
}

impl NumKind {
    fn of(ty: &Type) -> Option<Self> {
        let Type::Path(tp) = ty else { return None };
        let last = tp.path.segments.last()?;
        match last.ident.to_string().as_str() {
            "f32" => Some(NumKind::F32),
            "u32" => Some(NumKind::U32),
            "i32" => Some(NumKind::I32),
            "u64" => Some(NumKind::U64),
            "i64" => Some(NumKind::I64),
            _ => None,
        }
    }
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

/// Strips the perf-only `#[unroll]` / `#[unroll(n)]` statement attribute from
/// `for`/`while`/`loop` expressions in the reference twin's body — it's a
/// CubeCL codegen hint, invalid in plain Rust, and the only statement
/// attribute the twin derivation understands. Any *other* attribute found in
/// the same position is left in place and reported as a compile error
/// (`errors`) rather than silently dropped, per the "reject explicitly"
/// principle — an unrecognized attribute is exactly as unsupported as an
/// unrecognized construct.
#[derive(Default)]
struct StripUnrollFold {
    errors: Vec<syn::Error>,
}

impl StripUnrollFold {
    fn is_unroll(attr: &syn::Attribute) -> bool {
        attr.path().is_ident("unroll")
    }

    /// Remove recognized `#[unroll]`/`#[unroll(n)]` entries from `attrs`,
    /// recording an error for anything else found alongside them.
    fn scrub(&mut self, attrs: Vec<syn::Attribute>) -> Vec<syn::Attribute> {
        let mut kept = Vec::new();
        for attr in attrs {
            if Self::is_unroll(&attr) {
                continue;
            }
            self.errors.push(syn::Error::new(
                attr.span(),
                format!(
                    "attribute `{}` on a reference-twin loop is outside the vericl v0 subset; \
                     only #[unroll]/#[unroll(n)] is recognized there (perf-only, stripped from \
                     the twin, since plain Rust does not have it)",
                    attr.path().to_token_stream()
                ),
            ));
            kept.push(attr);
        }
        kept
    }
}

impl Fold for StripUnrollFold {
    fn fold_expr_for_loop(&mut self, mut i: ExprForLoop) -> ExprForLoop {
        i.attrs = self.scrub(std::mem::take(&mut i.attrs));
        syn::fold::fold_expr_for_loop(self, i)
    }

    fn fold_expr_while(&mut self, mut i: ExprWhile) -> ExprWhile {
        i.attrs = self.scrub(std::mem::take(&mut i.attrs));
        syn::fold::fold_expr_while(self, i)
    }

    fn fold_expr_loop(&mut self, mut i: ExprLoop) -> ExprLoop {
        i.attrs = self.scrub(std::mem::take(&mut i.attrs));
        syn::fold::fold_expr_loop(self, i)
    }
}

/// Rejects calls (`.method(...)` or `Type::method(...)`) to a
/// [`FLOAT_METHOD_REJECT`] name anywhere in the reference twin's body — see
/// that constant's doc for why. A pure check, not a rewrite: unlike
/// `StripUnrollFold`/`WrappingFold` it never changes the tree, only
/// collects `errors`.
#[derive(Default)]
struct FloatMethodCheck {
    errors: Vec<syn::Error>,
}

impl FloatMethodCheck {
    fn check(&mut self, name: &Ident) {
        let s = name.to_string();
        if FLOAT_METHOD_REJECT.contains(&s.as_str()) {
            self.errors.push(syn::Error::new(
                name.span(),
                format!(
                    "host-callability of `F::{s}` in the reference twin is unverified — outside \
                     the vericl v0 subset; verified host-callable Float/Numeric methods are: \
                     {}",
                    FLOAT_METHOD_WHITELIST.join(", ")
                ),
            ));
        }
    }
}

impl Fold for FloatMethodCheck {
    fn fold_expr_method_call(&mut self, i: syn::ExprMethodCall) -> syn::ExprMethodCall {
        self.check(&i.method);
        syn::fold::fold_expr_method_call(self, i)
    }

    fn fold_expr_call(&mut self, i: syn::ExprCall) -> syn::ExprCall {
        if let Expr::Path(p) = i.func.as_ref() {
            if let Some(last) = p.path.segments.last() {
                self.check(&last.ident);
            }
        }
        syn::fold::fold_expr_call(self, i)
    }
}

enum ParamKind {
    /// Plain scalar passed by value (f32, u32, i32, ...).
    Scalar(Type),
    /// `&Array<T>` — read-only buffer.
    ArrayRef(Type),
    /// `&mut Array<T>` — mutable buffer.
    ArrayMut(Type),
    /// `#[comptime] name: T` — a compile-time-pinned value. Removed from the
    /// reference twin's signature entirely; bound as a `let name: T =
    /// value;` const at the top of `reference`/`check_assumes` instead (see
    /// `instantiate(...)`). `T` is the parameter's declared type (already
    /// generic-substituted, in the unlikely event it mentions the type
    /// param).
    Comptime(Type),
}

struct Param {
    name: Ident,
    kind: ParamKind,
}

/// Extract a plain parameter name from a typed function argument's pattern —
/// shared by `classify_param` and the pre-pass that discovers `#[comptime]`
/// parameter names/types before the full param list is classified.
fn fn_arg_ident(pt: &PatType) -> syn::Result<Ident> {
    let Pat::Ident(pi) = pt.pat.as_ref() else {
        return Err(syn::Error::new(pt.pat.span(), "expected a plain parameter name"));
    };
    Ok(pi.ident.clone())
}

/// `true` if `arg` carries a (the only recognized) `#[comptime]` parameter
/// attribute.
fn is_comptime_param(arg: &FnArg) -> bool {
    let FnArg::Typed(pt) = arg else { return false };
    pt.attrs.iter().any(|a| a.path().is_ident("comptime"))
}

fn classify_param(arg: &FnArg) -> syn::Result<Param> {
    let FnArg::Typed(pt) = arg else {
        return Err(syn::Error::new(arg.span(), "self parameters are not supported"));
    };
    let comptime = pt.attrs.iter().any(|a| a.path().is_ident("comptime"));
    if (comptime && pt.attrs.len() > 1) || (!comptime && !pt.attrs.is_empty()) {
        return Err(syn::Error::new(
            pt.span(),
            "parameter attributes other than #[comptime] are outside the vericl v0 subset",
        ));
    }
    let name = fn_arg_ident(pt)?;

    if comptime {
        return match pt.ty.as_ref() {
            Type::Reference(r) => Err(syn::Error::new(
                r.span(),
                "#[comptime] parameters must be plain scalar types in the vericl v0 subset \
                 (Array is not supported as a comptime parameter)",
            )),
            _ => Ok(Param { name, kind: ParamKind::Comptime(pt.ty.as_ref().clone()) }),
        };
    }

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

/// Apply `subst` (generic type ident -> concrete type tokens) to one
/// function argument's declared type, reparsing the result as a [`Type`].
/// A no-op clone when `subst` is empty (the overwhelmingly common case: a
/// non-generic kernel), so this costs nothing for existing kernels.
fn substitute_fn_arg(arg: &FnArg, subst: &GenericSubst) -> syn::Result<FnArg> {
    let FnArg::Typed(pt) = arg else { return Ok(arg.clone()) };
    if subst.is_empty() {
        return Ok(arg.clone());
    }
    let mut pt = pt.clone();
    let ty_tokens = subst_type_tokens(pt.ty.to_token_stream(), subst);
    let new_ty: Type = syn::parse2(ty_tokens).map_err(|e| {
        syn::Error::new(
            pt.ty.span(),
            format!("internal error substituting instantiate(...) types into `{}`: {e}", pt.ty.to_token_stream()),
        )
    })?;
    pt.ty = Box::new(new_ty);
    Ok(FnArg::Typed(pt))
}

/// The result of cross-referencing a kernel's `instantiate(...)` clause (if
/// any) against its actual generic type parameters and `#[comptime]`
/// parameters: a token-wise substitution map for the former, and a
/// name -> concrete-value-tokens map for the latter. Both empty for a
/// kernel with neither generics nor comptime params and no clause — the
/// ordinary, unaffected v0 path.
struct InstantiatePlan {
    generic_subst: GenericSubst,
    comptime_values: HashMap<String, TokenStream2>,
    /// Concrete types for the kernel's generic type parameters, in the
    /// *original declared generic order* — what `expand::<...>`/
    /// `launch::<..., R>` turbofish need.
    generic_types_in_order: Vec<Type>,
    /// `"F = f32"`, `"taps = 3"`, ... — pretty-printed, in clause order, for
    /// `Contract::instantiate`.
    pretty: Vec<String>,
}

/// Gate and resolve the `instantiate(...)` clause against the kernel's
/// signature. See the module doc / README for the exact rules:
/// - generics/comptime params present, no clause -> targeted error to add one
/// - clause present, no generics/comptime params -> error (unused
///   instantiation is a contract lie)
/// - clause present and needed -> every generic type param and every
///   comptime param must get exactly one value, and no clause entry may name
///   anything else.
fn resolve_instantiate(
    spec: &ContractSpec,
    sig_span: proc_macro2::Span,
    fn_name_str: &str,
    generic_params: &[Ident],
    comptime_params: &[Ident],
) -> syn::Result<InstantiatePlan> {
    let needs_instantiate = !generic_params.is_empty() || !comptime_params.is_empty();

    let Some((clause_span, entries)) = &spec.instantiate else {
        if needs_instantiate {
            return Err(syn::Error::new(
                sig_span,
                format!(
                    "kernel `{fn_name_str}` has generic type parameters and/or #[comptime] \
                     parameters but no instantiate(...) contract clause; add one naming a \
                     concrete value for each, e.g. `instantiate(F = f32, N = 8)`, so vericl can \
                     monomorphize the reference twin, launch, and IR at those values"
                ),
            ));
        }
        return Ok(InstantiatePlan {
            generic_subst: HashMap::new(),
            comptime_values: HashMap::new(),
            generic_types_in_order: Vec::new(),
            pretty: Vec::new(),
        });
    };

    if !needs_instantiate {
        return Err(syn::Error::new(
            *clause_span,
            format!(
                "kernel `{fn_name_str}` declares instantiate(...) but has no generic type \
                 parameters or #[comptime] parameters to instantiate — remove the clause (an \
                 unused instantiation is a contract lie)"
            ),
        ));
    }

    let mut generic_subst: GenericSubst = HashMap::new();
    let mut comptime_values: HashMap<String, TokenStream2> = HashMap::new();
    let mut seen: HashMap<String, proc_macro2::Span> = HashMap::new();
    let mut pretty_entries: Vec<String> = Vec::new();
    let mut errors: Vec<syn::Error> = Vec::new();

    for entry in entries {
        let key = entry.name.to_string();
        if let Some(prev) = seen.get(&key) {
            let mut e = syn::Error::new(
                entry.name.span(),
                format!("instantiate(...) declares `{key}` more than once"),
            );
            e.combine(syn::Error::new(*prev, "previously declared here"));
            errors.push(e);
            continue;
        }
        seen.insert(key.clone(), entry.name.span());
        pretty_entries.push(format!("{} = {}", key, pretty(&entry.value)));

        if generic_params.contains(&entry.name) {
            match &entry.value {
                Expr::Path(_) => {
                    let ty: Type = match syn::parse2(entry.value.to_token_stream()) {
                        Ok(t) => t,
                        Err(e) => {
                            errors.push(syn::Error::new(
                                entry.value.span(),
                                format!(
                                    "instantiate(...) value for type parameter `{key}` must be a \
                                     concrete type: {e}"
                                ),
                            ));
                            continue;
                        }
                    };
                    generic_subst.insert(key, ty.to_token_stream());
                }
                other => {
                    errors.push(syn::Error::new(
                        other.span(),
                        format!(
                            "instantiate(...) value for type parameter `{key}` must be a \
                             concrete type (e.g. `{key} = f32`), not an expression"
                        ),
                    ));
                }
            }
        } else if comptime_params.contains(&entry.name) {
            comptime_values.insert(key, entry.value.to_token_stream());
        } else {
            errors.push(syn::Error::new(
                entry.name.span(),
                format!(
                    "instantiate(...) names `{key}`, which is not a generic type parameter or \
                     #[comptime] parameter of kernel `{fn_name_str}`"
                ),
            ));
        }
    }

    for g in generic_params {
        if !seen.contains_key(&g.to_string()) {
            errors.push(syn::Error::new(
                *clause_span,
                format!(
                    "instantiate(...) is missing a value for generic type parameter `{g}` of \
                     kernel `{fn_name_str}`"
                ),
            ));
        }
    }
    for n in comptime_params {
        if !seen.contains_key(&n.to_string()) {
            errors.push(syn::Error::new(
                *clause_span,
                format!(
                    "instantiate(...) is missing a value for #[comptime] parameter `{n}` of \
                     kernel `{fn_name_str}`"
                ),
            ));
        }
    }

    if let Some(combined) = errors.into_iter().reduce(|mut a, b| {
        a.combine(b);
        a
    }) {
        return Err(combined);
    }

    let generic_types_in_order: Vec<Type> = generic_params
        .iter()
        .map(|g| syn::parse2(generic_subst[&g.to_string()].clone()).expect("validated above"))
        .collect();

    Ok(InstantiatePlan {
        generic_subst,
        comptime_values,
        generic_types_in_order,
        pretty: pretty_entries,
    })
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

/// `vericl::suite! { runtime: ..., kernels: [...], evidence: "..." }` — see
/// `suite::expand` for the full grammar and design rationale.
#[proc_macro]
pub fn suite(input: TokenStream) -> TokenStream {
    match suite::expand(input.into()) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
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
    // Lifetime/const generics and where-clauses are never supported —
    // `instantiate(...)` only knows how to pin plain type parameters.
    if let Some(wc) = &func.sig.generics.where_clause {
        return Err(syn::Error::new(
            wc.span(),
            "where-clauses on kernel generics are outside the vericl v0 subset",
        ));
    }
    for gp in &func.sig.generics.params {
        if !matches!(gp, GenericParam::Type(_)) {
            return Err(syn::Error::new(
                gp.span(),
                "only type generic parameters (e.g. `F: Float`) are supported via \
                 instantiate(...); lifetime and const generic parameters are outside the vericl \
                 v0 subset",
            ));
        }
    }
    if !matches!(func.sig.output, ReturnType::Default) {
        return Err(syn::Error::new(
            func.sig.output.span(),
            "kernels must not return a value",
        ));
    }

    let fn_name = &func.sig.ident;
    let fn_name_str = fn_name.to_string();

    let generic_param_names: Vec<Ident> = func
        .sig
        .generics
        .params
        .iter()
        .map(|gp| {
            let GenericParam::Type(tp) = gp else {
                unreachable!("non-type generics rejected above")
            };
            tp.ident.clone()
        })
        .collect();
    let comptime_param_names: Vec<Ident> = func
        .sig
        .inputs
        .iter()
        .filter(|arg| is_comptime_param(arg))
        .map(|arg| {
            let FnArg::Typed(pt) = arg else { unreachable!("is_comptime_param checked this") };
            fn_arg_ident(pt)
        })
        .collect::<syn::Result<_>>()?;

    let plan = resolve_instantiate(
        &spec,
        func.sig.span(),
        &fn_name_str,
        &generic_param_names,
        &comptime_param_names,
    )?;

    // --- derive the substituted (F -> f32, etc.) parameter list, then
    // classify it exactly as before — every downstream site (gen(...),
    // NumKind matching, structured assumes, wrapping's integer check) sees
    // concrete types and never has to know instantiate(...) exists.
    let subst_args: Vec<FnArg> = func
        .sig
        .inputs
        .iter()
        .map(|arg| substitute_fn_arg(arg, &plan.generic_subst))
        .collect::<syn::Result<_>>()?;
    let params: Vec<Param> = subst_args.iter().map(classify_param).collect::<syn::Result<_>>()?;

    // `wrapping` rewrites `+`/`-`/`*`/`<<`/`>>` untyped — syn has no type
    // information at macro-expansion time — so it must not be allowed to
    // touch float math. Every parameter (including a #[comptime] const,
    // which participates in the twin body like any other value) must be an
    // integer scalar or integer Array.
    if spec.wrapping.is_some() {
        for p in &params {
            let (ok, ty_span) = match &p.kind {
                ParamKind::Scalar(ty) | ParamKind::Comptime(ty) => {
                    (is_wrapping_integer_type(ty), ty.span())
                }
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

    // --- derive the reference twin body: ABSOLUTE_POS rewrite + F -> f32
    // token substitution + banned-construct rejection, then always parse as
    // a `syn::Block` so the twin-only Fold passes below (unroll-attribute
    // stripping, and optionally `wrapping`) can run.
    let mut errors = Vec::new();
    let ref_body_tokens = transform_body(func.block.to_token_stream(), &plan.generic_subst, &mut errors);
    if let Some(combined) = errors.into_iter().reduce(|mut a, b| {
        a.combine(b);
        a
    }) {
        return Err(combined);
    }

    let mut ref_block: syn::Block = syn::parse2(ref_body_tokens).map_err(|e| {
        syn::Error::new(
            e.span(),
            format!(
                "internal error deriving the reference twin: the rewritten body did not parse \
                 as a block ({e})"
            ),
        )
    })?;

    // Strip the perf-only #[unroll]/#[unroll(n)] statement attribute from
    // twin loops (invalid in plain Rust); any other statement attribute is a
    // compile error rather than a silent drop.
    let mut strip_unroll = StripUnrollFold::default();
    ref_block = strip_unroll.fold_block(ref_block);
    if let Some(combined) = strip_unroll.errors.into_iter().reduce(|mut a, b| {
        a.combine(b);
        a
    }) {
        return Err(combined);
    }

    // Reject any call to a Float/Numeric method whose host-callability isn't
    // verified (see FLOAT_METHOD_REJECT) — a silently panicking or
    // miscomputing twin is exactly what vericl must never ship.
    let mut float_check = FloatMethodCheck::default();
    ref_block = float_check.fold_block(ref_block);
    if let Some(combined) = float_check.errors.into_iter().reduce(|mut a, b| {
        a.combine(b);
        a
    }) {
        return Err(combined);
    }

    // `wrapping`: fold the already-ABSOLUTE_POS-rewritten twin body, and
    // ONLY the twin — the `#[cube]` kernel re-emitted above is untouched.
    if spec.wrapping.is_some() {
        ref_block = WrappingFold.fold_block(ref_block);
    }
    let ref_body = ref_block.to_token_stream();

    // --- identity hash: source tokens + contract + vericl version ---
    let mut hasher = Sha256::new();
    hasher.update(func.to_token_stream().to_string().as_bytes());
    hasher.update(b"||contract:");
    hasher.update(attr.to_string().as_bytes());
    hasher.update(b"||vericl:");
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
    let hash = format!("sha256:{:x}", hasher.finalize());

    // --- generated signatures ---
    let mod_name = Ident::new(&format!("{fn_name}_vericl"), fn_name.span());
    let vis = &func.vis;

    // #[comptime] params are removed from both signatures below and instead
    // bound as `let name: ty = value;` consts — loop-invariant by
    // construction, so they're injected once at the top of each generated
    // function that needs them (see `comptime_bindings` below).
    let ref_params: Vec<TokenStream2> = params
        .iter()
        .filter_map(|p| {
            let name = &p.name;
            match &p.kind {
                ParamKind::Scalar(ty) => Some(quote!(#name: #ty)),
                ParamKind::ArrayRef(elem) => Some(quote!(#name: &[#elem])),
                ParamKind::ArrayMut(elem) => Some(quote!(#name: &mut [#elem])),
                ParamKind::Comptime(_) => None,
            }
        })
        .collect();

    // assumes predicate sees every buffer read-only
    let pred_params: Vec<TokenStream2> = params
        .iter()
        .filter_map(|p| {
            let name = &p.name;
            match &p.kind {
                ParamKind::Scalar(ty) => Some(quote!(#name: #ty)),
                ParamKind::ArrayRef(elem) | ParamKind::ArrayMut(elem) => {
                    Some(quote!(#name: &[#elem]))
                }
                ParamKind::Comptime(_) => None,
            }
        })
        .collect();

    let comptime_bindings: Vec<TokenStream2> = params
        .iter()
        .filter_map(|p| {
            let ParamKind::Comptime(ty) = &p.kind else { return None };
            let name = &p.name;
            let value = &plan.comptime_values[&name.to_string()];
            Some(quote!(let #name: #ty = #value;))
        })
        .collect();

    let assume_exprs = &spec.assumes;
    let assume_strs: Vec<String> = spec.assumes.iter().map(pretty).collect();
    let compare = &spec.compare;
    let compare_desc = &spec.compare_desc;
    let wrapping = spec.wrapping.is_some();
    let instantiate_strs: &[String] = &plan.pretty;

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
                kd_call_args.push(quote!(#name));
            }
            ParamKind::ArrayRef(elem) => {
                kd_stmts.push(quote! {
                    let #name = <::cubecl::prelude::Array<#elem> as ::cubecl::prelude::LaunchArg>::expand(
                        &::cubecl::prelude::ArrayCompilationArg { inplace: None },
                        &mut __vericl_builder,
                    );
                });
                buffer_params.push(quote!((#name_str, false)));
                kd_call_args.push(quote!(#name));
            }
            ParamKind::ArrayMut(elem) => {
                kd_stmts.push(quote! {
                    let #name = <::cubecl::prelude::Array<#elem> as ::cubecl::prelude::LaunchArg>::expand_output(
                        &::cubecl::prelude::ArrayCompilationArg { inplace: None },
                        &mut __vericl_builder,
                    );
                });
                buffer_params.push(quote!((#name_str, true)));
                kd_call_args.push(quote!(#name));
            }
            ParamKind::Comptime(_ty) => {
                // No LaunchArg registration — `expand()` takes #[comptime]
                // params as plain values at their declared position (see
                // cubecl-macros' `normalize_kernel_ty`, which leaves a
                // `is_const` param's type untouched instead of wrapping it
                // in `LaunchArg::ExpandType`).
                let value = &plan.comptime_values[&name_str];
                kd_call_args.push(value.clone());
            }
        }
    }

    // --- generic type turbofish for the twin's own `expand`/`launch` calls
    // (kernel_definition() and conformance_case() respectively) — the
    // concrete types instantiate(...) pinned, in the kernel's declared
    // generic order.
    let generic_types = &plan.generic_types_in_order;
    let expand_turbofish = if generic_types.is_empty() {
        TokenStream2::new()
    } else {
        quote!(::<#(#generic_types),*>)
    };

    // --- conformance_case(): the macro-generated GPU launch/input-gen glue
    // (README ergonomics milestone) — `generate_case` per the `gen(...)`
    // clause, then run reference vs. GPU and compare every `&mut Array`.
    let conformance_items = build_conformance_items(
        &params,
        &spec.gen_entries,
        fn_name,
        &fn_name_str,
        &plan.comptime_values,
        generic_types,
    )?;

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
                    instantiate: &[#(#instantiate_strs),*],
                }
            }

            /// The `assumes(...)` clauses as an executable predicate.
            pub fn check_assumes(#(#pred_params),*) -> bool {
                #(#comptime_bindings)*
                true #(&& (#assume_exprs))*
            }

            /// Sequential scalar reference execution over
            /// `ABSOLUTE_POS in 0..num_threads` — the same iteration space as
            /// the GPU dispatch, in deterministic ascending order.
            pub fn reference(#(#ref_params,)* num_threads: usize) {
                #(#comptime_bindings)*
                for __vericl_abs_pos in 0..num_threads #ref_body
            }

            /// Each array parameter's name and whether it's an output, in
            /// buffer-registration order — see `kernel_definition` below.
            pub const BUFFER_PARAMS: &[(&str, bool)] = &[#(#buffer_params),*];

            /// Build this kernel's CubeCL `KernelDefinition` (the IR) with no
            /// client/runtime/device involved — see
            /// docs/prototypes/ir_extraction.rs and docs/ir-research.md §1.
            /// For a generic kernel, `expand` is called with the
            /// instantiate(...)-pinned concrete type(s) via turbofish —
            /// exactly like a real call site would (see `docs/ir-research.md`
            /// / cubecl-macros' `generate/launch.rs`, which reuses the
            /// original generics on `expand` unchanged).
            pub fn kernel_definition() -> ::cubecl::prelude::KernelDefinition {
                let mut __vericl_builder = ::cubecl::prelude::KernelBuilder::default();
                __vericl_builder.runtime_properties(::core::default::Default::default());
                // Required: registers how usize/isize (ABSOLUTE_POS, .len(),
                // indices) map to concrete storage types; panics without it.
                ::cubecl::prelude::AddressType::U32.register(&mut __vericl_builder.scope);
                #(#kd_stmts)*
                #fn_name::expand #expand_turbofish(&mut __vericl_builder.scope, #(#kd_call_args),*);
                __vericl_builder.build(::cubecl::prelude::KernelSettings::default())
            }

            #conformance_items
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

// ---------------------------------------------------------------------------
// gen(...) / conformance_case(): macro-generated GPU launch and input-gen
// glue, so per-kernel harnesses stop hand-writing it (README ergonomics
// milestone). See `GenEntry` above for the `gen(...)` clause grammar.
// ---------------------------------------------------------------------------

/// One kernel parameter's resolved generation plan, built once from
/// `params` + the `gen(...)` clause and reused for both `generate_case` and
/// `conformance_case`.
enum FieldRole {
    Scalar,
    ArrayRef,
    ArrayMut,
}

struct GenField {
    name: Ident,
    /// The type `generate_case` returns this field as: `T` for a scalar,
    /// `Vec<T>` for an array.
    owned_ty: TokenStream2,
    /// The `let #name = ...;` statement that draws this field from the RNG.
    stmt: TokenStream2,
    role: FieldRole,
    /// Element type, for arrays only.
    elem_ty: Option<Type>,
    /// Element numeric kind, for arrays only (drives comparison dispatch).
    elem_kind: Option<NumKind>,
}

/// `gen(...)` range entries by parameter name.
type GenRanges = std::collections::HashMap<String, (Expr, Expr)>;
/// `gen(...)` `len(...)` pins by parameter name.
type GenLens = std::collections::HashMap<String, Expr>;

/// Resolve the `gen(...)` clause into a lookup by parameter name, validating
/// that every entry names a real parameter (and that `len(...)` only targets
/// array parameters) before any codegen happens.
fn resolve_gen_entries(
    params: &[Param],
    gen_entries: &[GenEntry],
    fn_name_str: &str,
) -> syn::Result<(GenRanges, GenLens)> {
    let mut ranges: GenRanges = std::collections::HashMap::new();
    let mut lens: GenLens = std::collections::HashMap::new();
    for entry in gen_entries {
        match entry {
            GenEntry::Range { name, lo, hi } => {
                let key = name.to_string();
                match params.iter().find(|p| p.name == *name) {
                    None => {
                        return Err(syn::Error::new(
                            name.span(),
                            format!(
                                "gen(...) declares a range for `{key}`, but `{key}` is not a \
                                 parameter of `{fn_name_str}`"
                            ),
                        ));
                    }
                    Some(p) if matches!(p.kind, ParamKind::Comptime(_)) => {
                        return Err(syn::Error::new(
                            name.span(),
                            format!(
                                "gen(...) declares a range for `{key}`, but `{key}` is a \
                                 #[comptime] parameter pinned by instantiate(...), not a runtime \
                                 value to generate"
                            ),
                        ));
                    }
                    Some(_) => {}
                }
                if ranges.insert(key.clone(), (lo.clone(), hi.clone())).is_some() {
                    return Err(syn::Error::new(
                        name.span(),
                        format!("gen(...) declares more than one range for `{key}`"),
                    ));
                }
            }
            GenEntry::Len { name, value } => {
                let key = name.to_string();
                match params.iter().find(|p| p.name == *name) {
                    None => {
                        return Err(syn::Error::new(
                            name.span(),
                            format!(
                                "gen(...) declares len(...) for `{key}`, but `{key}` is not a \
                                 parameter of `{fn_name_str}`"
                            ),
                        ));
                    }
                    Some(p) if matches!(p.kind, ParamKind::Comptime(_)) => {
                        return Err(syn::Error::new(
                            name.span(),
                            format!(
                                "gen(...) declares len(...) for `{key}`, but `{key}` is a \
                                 #[comptime] parameter pinned by instantiate(...), not a runtime \
                                 value to generate"
                            ),
                        ));
                    }
                    Some(p) if !matches!(p.kind, ParamKind::ArrayRef(_) | ParamKind::ArrayMut(_)) => {
                        return Err(syn::Error::new(
                            name.span(),
                            format!(
                                "gen(...) len(...) only applies to Array parameters; `{key}` is \
                                 a scalar"
                            ),
                        ));
                    }
                    _ => {}
                }
                if lens.insert(key.clone(), value.clone()).is_some() {
                    return Err(syn::Error::new(
                        name.span(),
                        format!("gen(...) declares len(...) more than once for `{key}`"),
                    ));
                }
            }
        }
    }
    Ok((ranges, lens))
}

/// The `let #name = ...;` statement drawing one integer field (scalar or,
/// via `len`, an array element closure body) from `__vericl_rng`, either
/// uniformly over `range` or full-range when no range is declared —
/// integers need no explicit `gen(...)` range (unlike floats — see
/// `build_conformance_items`).
fn integer_draw_expr(ty: &TokenStream2, kind: NumKind, range: Option<&(Expr, Expr)>) -> TokenStream2 {
    if let Some((lo, hi)) = range {
        quote! {
            {
                let __lo: #ty = #lo;
                let __hi: #ty = #hi;
                let __span: i128 = (__hi as i128) - (__lo as i128) + 1;
                (__lo as i128 + (__vericl_rng.next_u64() as i128).rem_euclid(__span)) as #ty
            }
        }
    } else {
        match kind {
            NumKind::U32 => quote!(__vericl_rng.next_u32() as #ty),
            NumKind::I32 => quote!(__vericl_rng.next_u32() as #ty),
            NumKind::U64 => quote!(__vericl_rng.next_u64() as #ty),
            NumKind::I64 => quote!(__vericl_rng.next_u64() as #ty),
            NumKind::F32 => unreachable!("floats never reach integer_draw_expr"),
        }
    }
}

/// Build one parameter's [`GenField`]: its owned type, its `generate_case`
/// draw statement, and (for arrays) its element type/kind for later
/// `conformance_case` codegen.
fn build_gen_field(
    p: &Param,
    ranges: &GenRanges,
    lens: &GenLens,
    fn_name_str: &str,
) -> syn::Result<GenField> {
    let name = p.name.clone();
    let name_str = name.to_string();
    match &p.kind {
        ParamKind::Scalar(ty) => {
            let Some(kind) = NumKind::of(ty) else {
                return Err(syn::Error::new(
                    ty.span(),
                    format!(
                        "gen(...) v0 only supports f32/u32/i32/u64/i64 scalar parameters; \
                         `{name_str}: {}` is outside that set",
                        ty.to_token_stream()
                    ),
                ));
            };
            let range = ranges.get(&name_str);
            let stmt = if kind == NumKind::F32 {
                let (lo, hi) = range.ok_or_else(|| {
                    syn::Error::new(
                        ty.span(),
                        format!(
                            "kernel `{fn_name_str}`: parameter `{name_str}` is a float with no \
                             declared gen(...) range — declare `gen({name_str} in lo..=hi)`; \
                             unbounded float generation produces NaN/inf-adjacent garbage and \
                             un-provable tolerances"
                        ),
                    )
                })?;
                quote!(let #name: #ty = __vericl_rng.next_f32_range((#lo) as f32, (#hi) as f32);)
            } else {
                let draw = integer_draw_expr(&quote!(#ty), kind, range);
                quote!(let #name: #ty = #draw;)
            };
            Ok(GenField { name, owned_ty: quote!(#ty), stmt, role: FieldRole::Scalar, elem_ty: None, elem_kind: None })
        }
        ParamKind::ArrayRef(elem) | ParamKind::ArrayMut(elem) => {
            let Some(kind) = NumKind::of(elem) else {
                return Err(syn::Error::new(
                    elem.span(),
                    format!(
                        "gen(...) v0 only supports f32/u32/i32/u64/i64 array elements; \
                         `{name_str}: Array<{}>` is outside that set",
                        elem.to_token_stream()
                    ),
                ));
            };
            let range = ranges.get(&name_str);
            let len_tokens = match lens.get(&name_str) {
                Some(e) => quote!((#e) as usize),
                None => quote!(n),
            };
            let stmt = if kind == NumKind::F32 {
                let (lo, hi) = range.ok_or_else(|| {
                    syn::Error::new(
                        elem.span(),
                        format!(
                            "kernel `{fn_name_str}`: parameter `{name_str}` is a float array \
                             with no declared gen(...) range — declare `gen({name_str} in \
                             lo..=hi)`; unbounded float generation produces NaN/inf-adjacent \
                             garbage and un-provable tolerances"
                        ),
                    )
                })?;
                quote! {
                    let #name: ::std::vec::Vec<#elem> =
                        __vericl_rng.fill_f32(#len_tokens, (#lo) as f32, (#hi) as f32);
                }
            } else {
                let draw = integer_draw_expr(&quote!(#elem), kind, range);
                quote! {
                    let #name: ::std::vec::Vec<#elem> =
                        (0..#len_tokens).map(|_| #draw).collect();
                }
            };
            let role = if matches!(p.kind, ParamKind::ArrayMut(_)) { FieldRole::ArrayMut } else { FieldRole::ArrayRef };
            Ok(GenField { name, owned_ty: quote!(::std::vec::Vec<#elem>), stmt, role, elem_ty: Some(elem.clone()), elem_kind: Some(kind) })
        }
        ParamKind::Comptime(_) => {
            unreachable!("callers filter out #[comptime] params before calling build_gen_field")
        }
    }
}

/// Build the macro-generated `generate_case` and `conformance_case` items
/// for one kernel's `<name>_vericl` module (README ergonomics milestone:
/// `#[vericl::kernel]` already knows every signature, so the harness no
/// longer hand-writes GPU launch/input-gen glue per kernel).
fn build_conformance_items(
    params: &[Param],
    gen_entries: &[GenEntry],
    fn_name: &Ident,
    fn_name_str: &str,
    comptime_values: &HashMap<String, TokenStream2>,
    generic_types: &[Type],
) -> syn::Result<TokenStream2> {
    let (ranges, lens) = resolve_gen_entries(params, gen_entries, fn_name_str)?;
    // #[comptime] params are pinned by instantiate(...), not generated —
    // excluded here, so `generate_case`/`reference`'s call site never
    // mentions them. They still get a `conformance_case`/`kernel_definition`
    // argument at their declared position — see the full-`params` pass
    // below (`launch_args`), separate from this Gen-only `fields` list.
    let fields: Vec<GenField> = params
        .iter()
        .filter(|p| !matches!(p.kind, ParamKind::Comptime(_)))
        .map(|p| build_gen_field(p, &ranges, &lens, fn_name_str))
        .collect::<syn::Result<_>>()?;

    let gen_stmts: Vec<&TokenStream2> = fields.iter().map(|f| &f.stmt).collect();
    let owned_tys: Vec<&TokenStream2> = fields.iter().map(|f| &f.owned_ty).collect();
    let field_names: Vec<&Ident> = fields.iter().map(|f| &f.name).collect();
    let check_args: Vec<TokenStream2> = fields
        .iter()
        .map(|f| {
            let name = &f.name;
            match f.role {
                FieldRole::Scalar => quote!(#name),
                FieldRole::ArrayRef | FieldRole::ArrayMut => quote!(&#name),
            }
        })
        .collect();

    let generate_case_fn = quote! {
        /// Generate one differential case's inputs deterministically from
        /// `(n, seed)`, in kernel-parameter declaration order (not
        /// `gen(...)` clause order) — see the `gen(...)` contract clause.
        /// Resamples up to 64 times if `check_assumes` rejects the draw,
        /// then panics naming the kernel: a persistent rejection means the
        /// declared `gen(...)` ranges are inconsistent with the kernel's
        /// own `assumes(...)` clauses, an authoring bug to fix rather than
        /// a runtime condition to recover from.
        fn generate_case(n: usize, seed: u64) -> ( #(#owned_tys,)* ) {
            let mut __vericl_rng = ::vericl::SplitMix64::new(seed);
            for _vericl_attempt in 0..64u32 {
                #(#gen_stmts)*
                if check_assumes(#(#check_args),*) {
                    return ( #(#field_names,)* );
                }
            }
            panic!(
                "kernel `{}`: gen(...) could not produce inputs satisfying assumes(...) after \
                 64 resample attempts — the declared gen(...) ranges are inconsistent with this \
                 kernel's assumes(...) clauses",
                #fn_name_str,
            );
        }
    };

    // --- conformance_case(): reference vs. GPU, per array parameter.
    let mut ref_clone_stmts: Vec<TokenStream2> = Vec::new();
    let mut reference_args: Vec<TokenStream2> = Vec::new();
    let mut gpu_upload_stmts: Vec<TokenStream2> = Vec::new();
    let mut gpu_readback_stmts: Vec<TokenStream2> = Vec::new();
    let mut compare_stmts: Vec<TokenStream2> = Vec::new();

    for f in &fields {
        let name = &f.name;
        match f.role {
            FieldRole::Scalar => {
                reference_args.push(quote!(#name));
            }
            FieldRole::ArrayRef => {
                let elem = f.elem_ty.as_ref().expect("array field has elem_ty");
                let handle = format_ident!("__vericl_{}_handle", name);
                reference_args.push(quote!(&#name));
                gpu_upload_stmts.push(quote! {
                    let #handle = client.create_from_slice(
                        <#elem as ::cubecl::prelude::CubeElement>::as_bytes(&#name),
                    );
                });
            }
            FieldRole::ArrayMut => {
                let elem = f.elem_ty.as_ref().expect("array field has elem_ty");
                let elem_kind = f.elem_kind.expect("array field has elem_kind");
                let ref_name = format_ident!("__vericl_{}_ref", name);
                let handle = format_ident!("__vericl_{}_handle", name);
                let gpu_name = format_ident!("__vericl_{}_gpu", name);

                ref_clone_stmts.push(quote! {
                    let mut #ref_name: ::std::vec::Vec<#elem> = #name.clone();
                });
                reference_args.push(quote!(&mut #ref_name));
                gpu_upload_stmts.push(quote! {
                    let #handle = client.create_from_slice(
                        <#elem as ::cubecl::prelude::CubeElement>::as_bytes(&#name),
                    );
                });
                gpu_readback_stmts.push(quote! {
                    let #gpu_name: ::std::vec::Vec<#elem> =
                        <#elem as ::cubecl::prelude::CubeElement>::from_bytes(
                            &client.read_one(#handle).unwrap(),
                        )
                        .to_vec();
                });

                let compare_call = match elem_kind {
                    NumKind::F32 => quote!(::vericl::compare_f32_with(contract().compare, &#ref_name, &#gpu_name)),
                    NumKind::U32 => quote!(::vericl::compare_u32_with(contract().compare, &#ref_name, &#gpu_name)),
                    NumKind::I32 | NumKind::U64 | NumKind::I64 => {
                        return Err(syn::Error::new(
                            elem.span(),
                            format!(
                                "conformance_case v0 only supports comparing f32 or u32 `&mut \
                                 Array` elements; `{name}: &mut Array<{}>` is outside that set",
                                elem.to_token_stream()
                            ),
                        ));
                    }
                };
                let name_str = name.to_string();
                compare_stmts.push(quote! {
                    __vericl_reports.push((#name_str.to_string(), #compare_call));
                });
            }
        }
    }

    // `launch_args`: unlike `reference_args` above, this covers *every*
    // parameter in original declaration order, including #[comptime] ones —
    // cubecl's generated `launch` keeps a comptime param in its declared
    // position with its plain (unwrapped) type (see cubecl-macros'
    // `Launch::launch_args`), so the real kernel call needs the pinned
    // instantiate(...) value spliced in right there, not appended at the end.
    let launch_args: Vec<TokenStream2> = params
        .iter()
        .map(|p| {
            let name = &p.name;
            match &p.kind {
                ParamKind::Scalar(_) => quote!(#name),
                ParamKind::ArrayRef(_) => {
                    let handle = format_ident!("__vericl_{}_handle", name);
                    quote! { unsafe { ::cubecl::prelude::ArrayArg::from_raw_parts(#handle, #name.len()) } }
                }
                ParamKind::ArrayMut(_) => {
                    let handle = format_ident!("__vericl_{}_handle", name);
                    quote! {
                        unsafe {
                            ::cubecl::prelude::ArrayArg::from_raw_parts(#handle.clone(), #name.len())
                        }
                    }
                }
                ParamKind::Comptime(_) => comptime_values[&name.to_string()].clone(),
            }
        })
        .collect();

    let launch_turbofish = if generic_types.is_empty() {
        quote!(<R>)
    } else {
        quote!(<#(#generic_types,)* R>)
    };

    let conformance_case_fn = quote! {
        /// Run one differential case: generate inputs via `gen(...)`, run
        /// the sequential reference (catching a panic as a finding, not a
        /// harness crash), launch the real kernel with standard 1D dispatch
        /// (`CubeCount = ceil(n/cube_dim)`, `num_threads = count*cube_dim`),
        /// and compare every `&mut Array` parameter's final contents
        /// against the reference's — the single point of custody for the
        /// GPU launch/input-gen glue every kernel previously hand-wrote.
        pub fn conformance_case<R: ::cubecl::prelude::Runtime>(
            client: &::cubecl::prelude::ComputeClient<R>,
            n: usize,
            seed: u64,
            cube_dim: u32,
        ) -> ::vericl::CaseOutcome {
            let ( #(#field_names,)* ) = generate_case(n, seed);

            #(#ref_clone_stmts)*

            let __vericl_count = (n as u32).div_ceil(cube_dim).max(1);
            let __vericl_cube_count = ::cubecl::prelude::CubeCount::Static(__vericl_count, 1, 1);
            let __vericl_cube_dim = ::cubecl::prelude::CubeDim::new_1d(cube_dim);
            let __vericl_num_threads = (__vericl_count * cube_dim) as usize;

            let __vericl_ref_outcome = ::vericl::catch_reference_panic(|| {
                reference(#(#reference_args,)* __vericl_num_threads);
            });

            match __vericl_ref_outcome {
                Err(__vericl_panic_msg) => ::vericl::CaseOutcome {
                    case: format!("n={n}"),
                    reports: ::std::vec::Vec::new(),
                    reference_panic: Some(__vericl_panic_msg),
                },
                Ok(()) => {
                    #(#gpu_upload_stmts)*
                    #fn_name::launch::#launch_turbofish(
                        client,
                        __vericl_cube_count,
                        __vericl_cube_dim,
                        #(#launch_args,)*
                    );
                    #(#gpu_readback_stmts)*

                    let mut __vericl_reports: ::std::vec::Vec<(::std::string::String, ::vericl::CompareReport)> =
                        ::std::vec::Vec::new();
                    #(#compare_stmts)*

                    ::vericl::CaseOutcome {
                        case: format!("n={n}"),
                        reports: __vericl_reports,
                        reference_panic: None,
                    }
                }
            }
        }
    };

    Ok(quote! {
        #generate_case_fn
        #conformance_case_fn
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The verified-safe and unverified/unsafe Float method name lists must
    /// never overlap — that would mean a name is simultaneously trusted and
    /// rejected, an internal contradiction. Also pins that both lists are
    /// actually populated (a silently emptied `FLOAT_METHOD_REJECT` would
    /// turn `FloatMethodCheck` into a no-op without any test noticing).
    #[test]
    fn float_method_lists_are_disjoint_and_nonempty() {
        assert!(!FLOAT_METHOD_WHITELIST.is_empty());
        assert!(!FLOAT_METHOD_REJECT.is_empty());
        for name in FLOAT_METHOD_WHITELIST {
            assert!(
                !FLOAT_METHOD_REJECT.contains(name),
                "`{name}` is both whitelisted and rejected"
            );
        }
    }
}
