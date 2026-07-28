//! `vericl::suite!` — the macro-generated conformance test.
//!
//! Expands one `suite!{ ... }` invocation into a `#[test] fn
//! vericl_conformance()` that runs every listed kernel's macro-generated
//! `conformance_case` across the declared sizes, optionally discharges the
//! SMT bounds proof via `vericl-ir`, assembles the evidence manifest in the
//! existing schema, and either writes it (`VERICL_UPDATE` set) or verifies
//! it against what's on disk (`cargo test`'s default path).
//!
//! A proc-macro rather than `macro_rules!` (both were open per the design
//! doc): the DSL has several optional, order-independent, defaulted fields
//! (`sizes`, `seed`, `cube_dim`, `prove`, `extra_lane`) — `syn`'s
//! `Meta`-like parsing (the same style `parse_contract` in `lib.rs` already
//! uses) handles that directly with real error spans, where `macro_rules!`
//! would need a hand-rolled arg-muncher. Keeping it in `vericl-macros`
//! (rather than a `macro_rules!` in `vericl` core) also matches the existing
//! division of labor: this crate never depends on `cubecl` itself, it only
//! emits tokens that reference `::cubecl::`/`::vericl_ir::` paths at the
//! call site in the user's crate — the same pattern `kernel_definition()`
//! already uses in `lib.rs`.
//!
//! Multi-lane runtimes (e.g. `--features cpu` adding a `CpuRuntime` lane on
//! top of the default `wgpu` one): `runtime:` stays single per the design
//! doc's decision, and an optional `extra_lane: (cfg(...), RuntimePath)`
//! field covers the rest. This was chosen over "a second hand-written
//! `#[test]` that calls generated helper functions" because two `#[test]`s
//! sharing one evidence file race (`cargo test` does not order or
//! serialize independent tests) and would in any case try to write two
//! different claim shapes to the same manifest. Folding the extra lane into
//! the *same* test via `#[cfg(...)]` on a block keeps one test, one
//! manifest write, and reuses `entries` before it's finalized — exactly
//! `conform.rs`'s old `add_cpu_lane(&mut entries)` shape, just generated
//! instead of hand-written.
//!
//! **Missing-annotation accessor error (rustc-mediated, by design).** Each name
//! in `kernels: [...]` is resolved to that kernel's generated `<name>_vericl`
//! module and its accessor functions (`conformance_case`, `kernel_definition`,
//! `contract`, `BUFFER_PARAMS`, …). If a listed kernel is missing its
//! `#[vericl::kernel]` attribute — the annotation that *generates* that module —
//! there is nothing for `suite!` to reference, and rustc reports a plain
//! resolution error at the `suite!` call site ("failed to resolve: use of
//! undeclared crate or module `<name>_vericl`", or "cannot find function
//! `conformance_case`"). vericl-macros cannot pre-empt this with a friendlier
//! message: a `#[proc_macro]` invocation has no whole-crate visibility, so it
//! cannot tell whether `<name>` names an annotated kernel or an ordinary `fn`.
//! The fix is always the same — add `#[vericl::kernel(...)]` (and `#[cube(launch)]`)
//! to the kernel, or remove the name from `kernels:`. The guide's "Reading
//! rejections" section documents this so a user hitting the resolution error
//! recognizes the cause.

use proc_macro2::{Ident, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{Expr, LitStr, Path, Token};

enum SuiteField {
    Runtime(Path),
    Kernels(Vec<Ident>),
    Evidence(LitStr),
    Sizes(Vec<Expr>),
    Seed(Expr),
    CubeDim(Expr),
    Prove(Expr),
    FrontendIndependent(Expr),
    ExtraLane { cfg_predicate: TokenStream2, path: Path },
}

impl Parse for SuiteField {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        match name.to_string().as_str() {
            "runtime" => Ok(SuiteField::Runtime(input.parse()?)),
            "kernels" => {
                let content;
                syn::bracketed!(content in input);
                let idents: Punctuated<Ident, Token![,]> = Punctuated::parse_terminated(&content)?;
                Ok(SuiteField::Kernels(idents.into_iter().collect()))
            }
            "evidence" => Ok(SuiteField::Evidence(input.parse()?)),
            "sizes" => {
                let content;
                syn::bracketed!(content in input);
                let exprs: Punctuated<Expr, Token![,]> = Punctuated::parse_terminated(&content)?;
                Ok(SuiteField::Sizes(exprs.into_iter().collect()))
            }
            "seed" => Ok(SuiteField::Seed(input.parse()?)),
            "cube_dim" => Ok(SuiteField::CubeDim(input.parse()?)),
            "prove" => Ok(SuiteField::Prove(input.parse()?)),
            "frontend_independent" => Ok(SuiteField::FrontendIndependent(input.parse()?)),
            "extra_lane" => {
                let content;
                syn::parenthesized!(content in input);
                let cfg_kw: Ident = content.parse().map_err(|e| {
                    syn::Error::new(e.span(), format!("expected `extra_lane: (cfg(...), RuntimePath)`: {e}"))
                })?;
                if cfg_kw != "cfg" {
                    return Err(syn::Error::new(
                        cfg_kw.span(),
                        "extra_lane: (...) expects a `cfg(...)` predicate first, e.g. \
                         `extra_lane: (cfg(feature = \"cpu\"), cubecl::cpu::CpuRuntime)`",
                    ));
                }
                let cfg_inner;
                syn::parenthesized!(cfg_inner in content);
                let cfg_predicate: TokenStream2 = cfg_inner.parse()?;
                content.parse::<Token![,]>()?;
                let path: Path = content.parse()?;
                if !content.is_empty() {
                    return Err(content.error(
                        "extra_lane: (cfg(...), RuntimePath) expects exactly these two entries",
                    ));
                }
                Ok(SuiteField::ExtraLane { cfg_predicate, path })
            }
            other => Err(syn::Error::new(
                name.span(),
                format!(
                    "unknown `suite!` field `{other}`; expected one of: runtime, kernels, \
                     evidence, sizes, seed, cube_dim, prove, frontend_independent, extra_lane"
                ),
            )),
        }
    }
}

struct SuiteInput(Punctuated<SuiteField, Token![,]>);

impl Parse for SuiteInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(SuiteInput(Punctuated::parse_terminated(input)?))
    }
}

struct SuiteSpec {
    runtime: Path,
    kernels: Vec<Ident>,
    evidence: LitStr,
    sizes: Vec<Expr>,
    seed: Expr,
    cube_dim: Expr,
    /// The `cube_dim:` field's own span, when the author wrote one — R7 blames
    /// it (docs/design-2d-dispatch.md §10.3). `None` when the default applies.
    cube_dim_span: Option<proc_macro2::Span>,
    /// `Some(rank)` when every `sizes:` entry is a 2- or 3-tuple: this is a
    /// **multi-axis dispatch suite** and its sizes are per-axis EXTENTS, not
    /// thread counts (docs/design-2d-dispatch.md §4.8). `None` for the ordinary
    /// 1-D suite.
    sizes_rank: Option<u8>,
    prove: Expr,
    /// Whether this suite's primary runtime is a front-end-independent
    /// execution lane relative to the macro-derived twin. `true` (default) for
    /// a GPU backend like wgpu — a genuinely different codegen path — where the
    /// entry's trusted list records `GPU_HARDWARE_TRUST`. `false` for a lane
    /// that shares CubeCL's front end AND is the only execution lane (the f64
    /// case: WGSL has no f64, so cubecl-cpu is the sole honest backend); then
    /// the trusted list swaps in `HOST_HARDWARE_TRUST` + the explicit
    /// `shared_frontend_lane_trust` caveat, so evidence never implies an
    /// independent execution lane exists where there is none — only the twin is
    /// independent.
    frontend_independent: Expr,
    extra_lane: Option<(TokenStream2, Path)>,
}

fn default_sizes() -> Vec<Expr> {
    ["1usize", "7usize", "256usize", "1000usize", "1027usize", "4096usize", "65536usize"]
        .iter()
        .map(|s| syn::parse_str(s).expect("literal default size parses"))
        .collect()
}

fn build_spec(fields: Punctuated<SuiteField, Token![,]>) -> syn::Result<SuiteSpec> {
    let mut runtime: Option<Path> = None;
    let mut kernels: Option<Vec<Ident>> = None;
    let mut evidence: Option<LitStr> = None;
    let mut sizes: Option<Vec<Expr>> = None;
    let mut seed: Option<Expr> = None;
    let mut cube_dim: Option<Expr> = None;
    let mut prove: Option<Expr> = None;
    let mut frontend_independent: Option<Expr> = None;
    let mut extra_lane: Option<(TokenStream2, Path)> = None;

    // Underline the offending (duplicate) field's own tokens, not the whole
    // `suite!` invocation.
    let dup = |field: &str, span: proc_macro2::Span| -> syn::Error {
        syn::Error::new(span, format!("suite!: duplicate `{field}` field"))
    };
    let first_span = |exprs: &[Expr]| -> proc_macro2::Span {
        exprs.first().map(|e| e.span()).unwrap_or_else(proc_macro2::Span::call_site)
    };

    for f in fields {
        match f {
            SuiteField::Runtime(p) => {
                if runtime.is_some() {
                    return Err(dup("runtime", p.span()));
                }
                runtime = Some(p);
            }
            SuiteField::Kernels(k) => {
                if kernels.is_some() {
                    let span =
                        k.first().map(|i| i.span()).unwrap_or_else(proc_macro2::Span::call_site);
                    return Err(dup("kernels", span));
                }
                kernels = Some(k);
            }
            SuiteField::Evidence(e) => {
                if evidence.is_some() {
                    return Err(dup("evidence", e.span()));
                }
                evidence = Some(e);
            }
            SuiteField::Sizes(s) => {
                if sizes.is_some() {
                    return Err(dup("sizes", first_span(&s)));
                }
                sizes = Some(s);
            }
            SuiteField::Seed(s) => {
                if seed.is_some() {
                    return Err(dup("seed", s.span()));
                }
                seed = Some(s);
            }
            SuiteField::CubeDim(c) => {
                if cube_dim.is_some() {
                    return Err(dup("cube_dim", c.span()));
                }
                cube_dim = Some(c);
            }
            SuiteField::Prove(p) => {
                if prove.is_some() {
                    return Err(dup("prove", p.span()));
                }
                prove = Some(p);
            }
            SuiteField::FrontendIndependent(p) => {
                if frontend_independent.is_some() {
                    return Err(dup("frontend_independent", p.span()));
                }
                frontend_independent = Some(p);
            }
            SuiteField::ExtraLane { cfg_predicate, path } => {
                if extra_lane.is_some() {
                    return Err(dup("extra_lane", path.span()));
                }
                extra_lane = Some((cfg_predicate, path));
            }
        }
    }

    let call_site = proc_macro2::Span::call_site();
    let runtime = runtime.ok_or_else(|| {
        syn::Error::new(call_site, "suite! requires a `runtime: <RuntimePath>` field")
    })?;
    let kernels = kernels.ok_or_else(|| {
        syn::Error::new(call_site, "suite! requires a `kernels: [k1, k2, ...]` field")
    })?;
    let evidence = evidence.ok_or_else(|| {
        syn::Error::new(call_site, "suite! requires an `evidence: \"path/to/vericl.json\"` field")
    })?;

    // --- multi-axis suite detection (docs/design-2d-dispatch.md §4.8). A
    // 2-D/3-D suite is spelled by its SIZES: `sizes: [(37, 19), (64, 64)]`.
    // Mixing tuple and scalar entries is rejected rather than guessed — the two
    // are different units (extents vs. thread counts), and round 8's units
    // discipline says decide it, not paper over it.
    let sizes_rank = match sizes.as_deref() {
        Some(list) if !list.is_empty() => {
            let arity = |e: &Expr| match e {
                Expr::Tuple(t) => Some(t.elems.len()),
                _ => None,
            };
            let first = arity(&list[0]);
            for e in list {
                if arity(e) != first {
                    return Err(syn::Error::new(
                        e.span(),
                        "suite!: every `sizes:` entry must have the same shape — either all \
                         scalars (a 1-D suite, sizes are thread counts) or all 2-/3-tuples (a \
                         `dispatch(...)` suite, sizes are per-axis extents). The two are \
                         different units and mixing them in one evidence config is not defined \
                         (docs/design-2d-dispatch.md §4.8)",
                    ));
                }
            }
            match first {
                None => None,
                Some(n @ (2 | 3)) => Some(n as u8),
                Some(n) => {
                    return Err(syn::Error::new(
                        list[0].span(),
                        format!(
                            "suite!: a tuple `sizes:` entry declares a multi-axis dispatch \
                             suite's per-axis extents and must have 2 or 3 elements; this one \
                             has {n}"
                        ),
                    ));
                }
            }
        }
        _ => None,
    };

    Ok(SuiteSpec {
        runtime,
        kernels,
        evidence,
        sizes: sizes.unwrap_or_else(default_sizes),
        seed: seed.unwrap_or_else(|| syn::parse_quote!(0xE901u64)),
        cube_dim_span: cube_dim.as_ref().map(|c| c.span()),
        sizes_rank,
        cube_dim: cube_dim.unwrap_or_else(|| syn::parse_quote!(256u32)),
        prove: prove.unwrap_or_else(|| syn::parse_quote!(true)),
        frontend_independent: frontend_independent.unwrap_or_else(|| syn::parse_quote!(true)),
        extra_lane,
    })
}

/// Deterministic FNV-1a 64-bit hash of a kernel name, used only to decorrelate
/// different kernels' RNG streams within one suite run (two kernels sharing a
/// seed would otherwise draw from the same underlying bit stream — harmless
/// since their parameter shapes differ, but needlessly suspicious). Computed
/// at macro-expansion time so it's a fixed, reproducible per-kernel constant,
/// not a hand-maintained salt list.
fn kernel_salt(name: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// One kernel's block in the primary runtime lane: run every size, print,
/// build the `Tested` (and, when `prove`, `Proved`) claims, and push a fresh
/// `Entry`. A cooperative kernel (`COOPERATIVE_CUBE_DIM.is_some()`) runs the
/// two-prover pipeline and the differential↔race-freedom coupling of
/// docs/design-shared-memory.md §6; a non-cooperative kernel keeps the ordinary
/// bounds-only pipeline. The branch is on a per-kernel const the kernel macro
/// emits, since `suite!` (a separate macro invocation) cannot see the clauses.
/// The per-case `conformance_case` fan-out, in whichever unit this suite's
/// `sizes:` declares — a scalar thread count (`n: usize`, the 1-D suite) or a
/// per-axis extents triple (`[usize; 3]`, a `dispatch(...)` suite). The pinned
/// cube dims mean a dispatch case takes no `cube_dim` argument at all: there is
/// exactly one source of truth for it and it is the clause.
fn case_call_tokens(
    kmod: &Ident,
    salt: u64,
    sizes_rank: Option<u8>,
    sizes: TokenStream2,
    runtime: &Ident,
    client: TokenStream2,
) -> TokenStream2 {
    if sizes_rank.is_some() {
        quote! {
            #sizes
                .iter()
                .map(|&__vericl_e| {
                    // Decorrelate the per-case RNG stream by all three extents,
                    // not by their product: `(64, 64)` and `(4096, 1)` are
                    // different cases and should not share a draw.
                    let __vericl_case_salt = (__vericl_e[0] as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ (__vericl_e[1] as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
                        ^ (__vericl_e[2] as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
                    #kmod::conformance_case::<#runtime>(
                        #client,
                        __vericl_e,
                        __vericl_seed ^ #salt ^ __vericl_case_salt,
                    )
                })
                .collect()
        }
    } else {
        quote! {
            #sizes
                .iter()
                .map(|&n| {
                    #kmod::conformance_case::<#runtime>(
                        #client,
                        n,
                        __vericl_seed ^ #salt ^ (n as u64),
                        __vericl_cube_dim,
                    )
                })
                .collect()
        }
    }
}

/// The compile-time agreement check between this suite's declared `sizes:`
/// shape and each listed kernel's `dispatch(...)` clause. A rank mismatch is
/// caught here — with the reason — rather than as a raw type error on
/// `conformance_case`'s size argument.
fn dispatch_rank_check(kmod: &Ident, kernel: &Ident, sizes_rank: Option<u8>) -> TokenStream2 {
    let kname = kernel.to_string();
    match sizes_rank {
        Some(rank) => {
            let msg = format!(
                "suite!: this suite declares tuple `sizes:` (a {rank}-D dispatch suite), but \
                 kernel `{kname}`'s `dispatch(...)` clause is absent or of a different rank. The \
                 suite's size arity and the clause's `cube_dim` arity must agree — they are the \
                 same dispatch rank seen from two places."
            );
            quote! {
                const _: () = assert!(
                    match #kmod::DISPATCH_RANK {
                        ::core::option::Option::Some(__r) => __r == #rank,
                        ::core::option::Option::None => false,
                    },
                    #msg
                );
            }
        }
        None => {
            let msg = format!(
                "suite!: kernel `{kname}` declares a `dispatch(...)` clause, so its cases are \
                 per-axis EXTENTS — but this suite's `sizes:` are scalar thread counts. Declare \
                 the sizes as tuples, e.g. `sizes: [(37, 19), (64, 64)]`, in a suite of its own \
                 (docs/design-2d-dispatch.md §4.8)."
            );
            quote! {
                const _: () = assert!(#kmod::DISPATCH_RANK.is_none(), #msg);
            }
        }
    }
}

/// The `Tested` claim's `config` builder for this suite's unit — selected at
/// macro-expansion time, not at run time, because the three builders take
/// different `sizes` types (`&[usize]` vs `&[[usize; 3]]`).
fn differential_config_tokens(
    kmod: &Ident,
    sizes_rank: Option<u8>,
    sizes: TokenStream2,
) -> TokenStream2 {
    if sizes_rank.is_some() {
        // A multi-axis suite's `sizes` are per-axis EXTENTS, and the recorded
        // launch shape is the full pinned triple + rank
        // (docs/design-2d-dispatch.md §4.8, §10.4 correction 2).
        quote! {
            ::vericl::differential_dispatch_config(
                #sizes,
                __vericl_seed,
                #kmod::DISPATCH_CUBE_DIM.expect("a dispatch suite's kernels pin their cube dims"),
                #kmod::DISPATCH_RANK.expect("a dispatch kernel has a rank"),
            )
        }
    } else {
        quote! {
            if let ::core::option::Option::Some(__w) = #kmod::VECTOR_WIDTH {
                ::vericl::differential_vector_config(#sizes, __vericl_seed, __vericl_cube_dim, __w)
            } else {
                ::vericl::differential_config(#sizes, __vericl_seed, __vericl_cube_dim)
            }
        }
    }
}

/// The bounds-prover entry point for this suite's unit — likewise selected at
/// macro-expansion time so a 1-D suite never mentions the dispatch entry point.
fn prove_call_tokens(kmod: &Ident, sizes_rank: Option<u8>) -> TokenStream2 {
    if sizes_rank.is_some() {
        quote! {
            ::vericl_ir::prove_bounds_freedom_dispatch(
                &__def,
                &__buffers,
                &__assumes,
                #kmod::DISPATCH_CUBE_DIM.expect("a dispatch suite's kernels pin their cube dims"),
            )
        }
    } else {
        quote!(::vericl_ir::prove_bounds_freedom(&__def, &__buffers, &__assumes))
    }
}

/// The claim pipeline for one kernel, selected at macro-expansion time.
///
/// A **dispatch suite** emits only the ordinary (non-cooperative) pipeline:
/// `dispatch(...)` and `cooperative(...)` are mutually exclusive in v1 (D6/R4),
/// so the cooperative branch is not merely dead there — it would not type-check,
/// because `cooperative_differential_config` takes `&[usize]` thread counts
/// while a dispatch suite's sizes are `[usize; 3]` extents. Emitting the branch
/// and relying on a runtime `if` would silently couple the two units.
fn pipeline_tokens(
    kmod: &Ident,
    sizes_rank: Option<u8>,
    differential_config: &TokenStream2,
    prove_call: &TokenStream2,
) -> TokenStream2 {
    let ordinary = quote! {
                // ---- Ordinary (non-cooperative) pipeline ----
                // A vector kernel records its pinned lane width in the config
                // (design-line-vector.md §9) so the `sizes` read as line counts
                // and a re-run at a different width is a visibly different claim.
                let __config = #differential_config;
                __claims.push(::vericl::Claim {
                    kind: ::vericl::ClaimKind::Tested,
                    check: "differential".to_string(),
                    backend: Some(__vericl_backend.clone()),
                    config: __config,
                    result: __result,
                });

                if __vericl_prove {
                    let __def = #kmod::kernel_definition();
                    let __ir_hash = ::vericl_ir::kernel_ir_hash(&__def);
                    let __buffers: ::std::vec::Vec<::vericl_ir::BufferParam> = #kmod::BUFFER_PARAMS
                        .iter()
                        .map(|(name, is_output)| ::vericl_ir::BufferParam { name, is_output: *is_output })
                        .collect();
                    let __assumes: ::std::vec::Vec<::vericl_ir::Assume> = #kmod::contract()
                        .structured_assumes
                        .iter()
                        .map(|a| match *a {
                            ::vericl::StructuredAssume::LenEq { a, b } => ::vericl_ir::Assume::LenEq { a, b },
                            ::vericl::StructuredAssume::LenEqConst { a, value } => {
                                ::vericl_ir::Assume::LenEqConst { a, value }
                            }
                            ::vericl::StructuredAssume::ElemsBelowLen { arr, len_of } => {
                                ::vericl_ir::Assume::ElemsBelowLen { arr, len_of }
                            }
                            ::vericl::StructuredAssume::ElemsBelowConst { arr, bound } => {
                                ::vericl_ir::Assume::ElemsBelowConst { arr, bound }
                            }
                            ::vericl::StructuredAssume::LenPlusConstLe { a, k, b } => {
                                ::vericl_ir::Assume::LenPlusConstLe { a, k, b }
                            }
                            ::vericl::StructuredAssume::LenEqProduct {
                                a, x: _, y: _, x_scalar, y_scalar,
                            } => ::vericl_ir::Assume::LenEqProduct { a, x_scalar, y_scalar },
                        })
                        .collect();
                    let __prove_result = #prove_call;
                    let (__obligations, __claim_result) = match &__prove_result {
                        ::vericl_ir::ProveResult::Proved { obligations } => {
                            (*obligations, ::vericl::ClaimResult::Pass)
                        }
                        ::vericl_ir::ProveResult::Refuted { obligation, counterexample } => (
                            0,
                            ::vericl::ClaimResult::Fail {
                                detail: format!("REFUTED: {obligation} — counterexample: {counterexample}"),
                            },
                        ),
                        ::vericl_ir::ProveResult::OutOfSubset { reason } => (
                            0,
                            ::vericl::ClaimResult::Fail { detail: format!("outside the vericl v0 subset: {reason}") },
                        ),
                        ::vericl_ir::ProveResult::SolverError { detail } => {
                            (0, ::vericl::ClaimResult::Fail { detail: format!("solver error: {detail}") })
                        }
                    };
                    __identity.ir_hash = Some(__ir_hash);
                    __claims.push(::vericl::Claim {
                        kind: ::vericl::ClaimKind::Proved,
                        check: ::vericl_ir::SMT_OOB_FREEDOM_CHECK.to_string(),
                        backend: None,
                        config: ::vericl::proved_config_with_logic(
                            __vericl_solver.as_deref().expect("prove checked z3 above"),
                            __obligations,
                            // §10.4 correction 3: the logic actually in force.
                            // A `LenEqProduct` assume asserts a nonlinear
                            // `len = x*y` into the global context, so `QF_LIA`
                            // would be wrong for that kernel.
                            if #kmod::contract().structured_assumes.iter().any(|__a| {
                                matches!(__a, ::vericl::StructuredAssume::LenEqProduct { .. })
                            }) {
                                "QF_NIA"
                            } else {
                                "QF_LIA"
                            },
                        ),
                        result: __claim_result,
                    });
                    __trusted.extend(::vericl::proved_bounds_trust(
                        __vericl_solver.as_deref().expect("prove checked z3 above"),
                    ));
                }
    };
    if sizes_rank.is_some() {
        return quote! { { #ordinary } };
    }
    quote! {
            if let ::core::option::Option::Some(__coop_cd) = #kmod::COOPERATIVE_CUBE_DIM {
                // ---- Cooperative pipeline (docs/design-shared-memory.md §6) ----
                // The phase-split twin is a faithful reference only under
                // intra-phase race freedom + barrier non-divergence, so the
                // tested claim ALWAYS records that dependency — discharged by
                // the `smt-race-freedom` proof (strong tier), or as an explicit
                // injected assumption (honest fallback). It is never assumed
                // silently, and a racy kernel's failing race proof sinks the
                // entry rather than recording a green-by-luck tested pass.
                let __ref_desc = if #kmod::DECLARED_REFERENCE {
                    "author-supplied declared reference (not derived from kernel source)"
                } else {
                    "vericl-macros phase-split cooperative twin (derived from kernel source)"
                };
                let __tested_check = if #kmod::DECLARED_REFERENCE {
                    "differential-declared-reference"
                } else {
                    "differential"
                };

                let mut __dependency = ::vericl::RaceDependency::Assumed;
                let mut __assumption: ::core::option::Option<::vericl::Claim> = None;

                if __vericl_prove {
                    let __solver = __vericl_solver.as_deref().expect("prove checked z3 above");
                    let __def = #kmod::kernel_definition();
                    __identity.ir_hash = Some(::vericl_ir::kernel_ir_hash(&__def));
                    let __buffers: ::std::vec::Vec<::vericl_ir::BufferParam> = #kmod::BUFFER_PARAMS
                        .iter()
                        .map(|(name, is_output)| ::vericl_ir::BufferParam { name, is_output: *is_output })
                        .collect();
                    let __assumes: ::std::vec::Vec<::vericl_ir::Assume> = #kmod::contract()
                        .structured_assumes
                        .iter()
                        .map(|a| match *a {
                            ::vericl::StructuredAssume::LenEq { a, b } => ::vericl_ir::Assume::LenEq { a, b },
                            ::vericl::StructuredAssume::LenEqConst { a, value } => {
                                ::vericl_ir::Assume::LenEqConst { a, value }
                            }
                            ::vericl::StructuredAssume::ElemsBelowLen { arr, len_of } => {
                                ::vericl_ir::Assume::ElemsBelowLen { arr, len_of }
                            }
                            ::vericl::StructuredAssume::ElemsBelowConst { arr, bound } => {
                                ::vericl_ir::Assume::ElemsBelowConst { arr, bound }
                            }
                            ::vericl::StructuredAssume::LenPlusConstLe { a, k, b } => {
                                ::vericl_ir::Assume::LenPlusConstLe { a, k, b }
                            }
                            ::vericl::StructuredAssume::LenEqProduct {
                                a, x: _, y: _, x_scalar, y_scalar,
                            } => ::vericl_ir::Assume::LenEqProduct { a, x_scalar, y_scalar },
                        })
                        .collect();
                    match ::vericl_ir::prove_cooperative(
                        &__def,
                        &__buffers,
                        &__assumes,
                        __coop_cd,
                        #kmod::COOP_BARRIER_COUNT,
                    ) {
                        ::vericl_ir::CooperativeProof::Proved(__o) => {
                            // Strong tier: one sound two-thread walk discharges
                            // BOTH bounds and races; split into two claims.
                            __claims.push(::vericl::Claim {
                                kind: ::vericl::ClaimKind::Proved,
                                check: ::vericl_ir::SMT_OOB_FREEDOM_CHECK.to_string(),
                                backend: None,
                                config: ::vericl::proved_bounds_cooperative_config(__solver, __o.bounds),
                                result: ::vericl::ClaimResult::Pass,
                            });
                            __claims.push(::vericl::Claim {
                                kind: ::vericl::ClaimKind::Proved,
                                check: ::vericl_ir::SMT_RACE_FREEDOM_CHECK.to_string(),
                                backend: None,
                                config: ::vericl::proved_race_config(
                                    __solver,
                                    __o.race(),
                                    __o.phases,
                                    __o.write_write,
                                    __o.read_write,
                                    __o.intercube,
                                    __o.uniformity,
                                ),
                                result: ::vericl::ClaimResult::Pass,
                            });
                            __trusted.extend(::vericl::proved_bounds_trust(__solver));
                            __trusted.extend(::vericl::proved_race_freedom_trust(__solver));
                            __dependency = ::vericl::RaceDependency::Discharged;
                        }
                        ::vericl_ir::CooperativeProof::Refuted { obligation, counterexample } => {
                            // A genuine two-thread race: emit a FAILING race
                            // claim (the entry fails — a racy kernel belongs in
                            // demo-defects, not the honest suite) and fall back
                            // to the explicit assumption for the tested claim.
                            __claims.push(::vericl::Claim {
                                kind: ::vericl::ClaimKind::Proved,
                                check: ::vericl_ir::SMT_RACE_FREEDOM_CHECK.to_string(),
                                backend: None,
                                config: ::vericl::proved_race_config(__solver, 0, 0, 0, 0, 0, 0),
                                result: ::vericl::ClaimResult::Fail {
                                    detail: format!(
                                        "REFUTED: {obligation} — two-thread counterexample: {counterexample}"
                                    ),
                                },
                            });
                            __assumption = Some(::vericl::race_freedom_assumption_claim());
                        }
                        ::vericl_ir::CooperativeProof::OutOfSubset { reason } => {
                            // Honest fallback: race freedom is not provable for
                            // this kernel's shape. No proved claim (there is
                            // nothing discharged), inject the explicit assumption
                            // the tested claim depends on. NB the smt-oob-freedom
                            // claim is also absent here — the same walk discharges
                            // both, so if it cannot run neither property is proved.
                            println!(
                                "      note: {} cooperative proofs OutOfSubset ({reason}) — \
                                 tested claim carries the explicit race-freedom assumption",
                                #kmod::contract().kernel
                            );
                            __assumption = Some(::vericl::race_freedom_assumption_claim());
                        }
                        ::vericl_ir::CooperativeProof::SolverError { detail } => {
                            panic!(
                                "z3 solver error proving cooperative kernel `{}`: {detail}",
                                #kmod::contract().kernel
                            );
                        }
                    }
                } else {
                    // prove disabled: honest fallback — no proofs, explicit
                    // assumption (exactly as the ordinary lane omits the bounds
                    // proof under prove: false, rather than faking one).
                    __assumption = Some(::vericl::race_freedom_assumption_claim());
                }

                // Tested claim built AFTER the provers (its config cites the
                // dependency), inserted first so it heads the entry.
                __claims.insert(0, ::vericl::Claim {
                    kind: ::vericl::ClaimKind::Tested,
                    check: __tested_check.to_string(),
                    backend: Some(__vericl_backend.clone()),
                    config: ::vericl::cooperative_differential_config(
                        __vericl_sizes, __vericl_seed, __coop_cd, __ref_desc, __dependency,
                    ),
                    result: __result,
                });
                if let Some(__a) = __assumption {
                    __claims.push(__a);
                }
        } else {
            #ordinary
        }
    }
}

fn kernel_block(kernel: &Ident, sizes_rank: Option<u8>) -> TokenStream2 {
    let kmod = format_ident!("{}_vericl", kernel);
    let salt = kernel_salt(&kernel.to_string());
    let case_call = case_call_tokens(
        &kmod,
        salt,
        sizes_rank,
        quote!(__vericl_sizes),
        &format_ident!("__VericlR"),
        quote!(&__vericl_client),
    );
    let dispatch_rank_check = dispatch_rank_check(&kmod, kernel, sizes_rank);
    let differential_config =
        differential_config_tokens(&kmod, sizes_rank, quote!(__vericl_sizes));
    let prove_call = prove_call_tokens(&kmod, sizes_rank);
    let pipeline = pipeline_tokens(&kmod, sizes_rank, &differential_config, &prove_call);
    quote! {
        {
            #dispatch_rank_check
            let __outcomes: ::std::vec::Vec<::vericl::CaseOutcome> = #case_call;

            let __pass = __outcomes.iter().all(::vericl::CaseOutcome::pass);
            println!(
                "  [{}] {} ({})",
                if __pass { "PASS" } else { "FAIL" },
                #kmod::contract().kernel,
                #kmod::contract().compare.describe(),
            );
            for o in &__outcomes {
                println!("      {}", ::vericl::describe_case_outcome(o));
            }

            let __detail = __outcomes
                .iter()
                .filter(|o| !o.pass())
                .map(::vericl::describe_case_outcome)
                .collect::<::std::vec::Vec<_>>()
                .join("; ");
            let __result = if __pass {
                ::vericl::ClaimResult::Pass
            } else {
                ::vericl::ClaimResult::Fail { detail: __detail }
            };

            let mut __trusted = ::vericl::reference_twin_trust();
            __trusted.push(::vericl::backend_buffer_trust(&__vericl_backend));
            if __vericl_frontend_independent {
                __trusted.push(::vericl::GPU_HARDWARE_TRUST.to_string());
            } else {
                // Non-independent primary lane (the f64 / cubecl-cpu case): the
                // only execution backend shares CubeCL's front end with the
                // kernel under test, so evidence must NOT imply an independent
                // execution lane exists. "GPU hardware" is also a misnomer here.
                __trusted.push(::vericl::HOST_HARDWARE_TRUST.to_string());
                __trusted.push(::vericl::shared_frontend_lane_trust(&__vericl_backend));
            }
            let mut __identity = #kmod::identity();
            let mut __claims: ::std::vec::Vec<::vericl::Claim> = ::std::vec::Vec::new();

            #pipeline

            entries.push(::vericl::Entry {
                kernel: #kmod::contract().kernel.to_string(),
                identity: __identity,
                contract: #kmod::contract().record(),
                claims: __claims,
                trusted: __trusted,
            });
        }
    }
}

/// One kernel's block in an `extra_lane`: run every size on the extra
/// runtime and fold a `Tested` claim + shared-front-end trust wording onto
/// the matching entry already built by [`kernel_block`] — mirrors
/// `conform.rs`'s old `add_cpu_lane`.
fn extra_lane_kernel_block(kernel: &Ident, sizes_rank: Option<u8>) -> TokenStream2 {
    let kmod = format_ident!("{}_vericl", kernel);
    let salt = kernel_salt(&kernel.to_string());
    let case_call = case_call_tokens(
        &kmod,
        salt,
        sizes_rank,
        quote!((&__extra_sizes[..])),
        &format_ident!("__VericlExtraR"),
        quote!(&__vericl_extra_client),
    );
    let extra_size_ty: TokenStream2 =
        if sizes_rank.is_some() { quote!([usize; 3]) } else { quote!(usize) };
    let differential_config =
        differential_config_tokens(&kmod, sizes_rank, quote!(&__extra_sizes));
    quote! {
        {
            // Extra-lane sizes. For a COOPERATIVE kernel, cap to single-cube
            // cases (`n <= cube_dim`): a CPU runtime (e.g. cubecl-cpu) executes
            // a workgroup-cooperative kernel per-cube with heavy barrier-
            // simulation overhead (seconds per cube — measured ~6s/cube), so the
            // primary lane's large multi-cube sizes (65536 → 256 cubes) would
            // turn the extra lane into a many-minute run. A few single-cube
            // cases confirm the shared front end agrees; the INDEPENDENT primary
            // lane still covers every declared size (docs/design-shared-memory.md
            // — cubecl-cpu cooperative-execution performance finding). Non-
            // cooperative kernels are unaffected (all sizes).
            let __extra_sizes: ::std::vec::Vec<#extra_size_ty> =
                if let ::core::option::Option::Some(__ccd) = #kmod::COOPERATIVE_CUBE_DIM {
                    let mut __v: ::std::vec::Vec<usize> =
                        __vericl_sizes.iter().copied().filter(|&n| n <= __ccd as usize).collect();
                    if __v.is_empty() {
                        __v.push(__ccd as usize);
                    }
                    __v
                } else {
                    __vericl_sizes.to_vec()
                };
            let __outcomes: ::std::vec::Vec<::vericl::CaseOutcome> = #case_call;
            for o in &__outcomes {
                println!("      {}", ::vericl::describe_case_outcome(o));
            }
            let __pass = __outcomes.iter().all(::vericl::CaseOutcome::pass);
            let __detail = __outcomes
                .iter()
                .filter(|o| !o.pass())
                .map(::vericl::describe_case_outcome)
                .collect::<::std::vec::Vec<_>>()
                .join("; ");
            let __result = if __pass {
                ::vericl::ClaimResult::Pass
            } else {
                ::vericl::ClaimResult::Fail { detail: __detail }
            };
            if let Some(entry) = entries.iter_mut().find(|e| e.kernel == #kmod::contract().kernel) {
                let __claim = if let ::core::option::Option::Some(__coop_cd) = #kmod::COOPERATIVE_CUBE_DIM {
                    // Cooperative extra lane: mirror the main lane's coupling.
                    // The dependency is read off the entry the main lane already
                    // built — a discharged proof is present iff its passing
                    // `smt-race-freedom` claim is (no need to re-run the prover).
                    let __dependency = if entry.claims.iter().any(|c| {
                        c.kind == ::vericl::ClaimKind::Proved
                            && c.check == ::vericl::SMT_RACE_FREEDOM_CHECK
                            && matches!(c.result, ::vericl::ClaimResult::Pass)
                    }) {
                        ::vericl::RaceDependency::Discharged
                    } else {
                        ::vericl::RaceDependency::Assumed
                    };
                    let __ref_desc = if #kmod::DECLARED_REFERENCE {
                        "author-supplied declared reference (not derived from kernel source)"
                    } else {
                        "vericl-macros phase-split cooperative twin (derived from kernel source)"
                    };
                    let __tested_check = if #kmod::DECLARED_REFERENCE {
                        "differential-declared-reference"
                    } else {
                        "differential"
                    };
                    ::vericl::Claim {
                        kind: ::vericl::ClaimKind::Tested,
                        check: __tested_check.to_string(),
                        backend: Some(__vericl_extra_backend.clone()),
                        config: ::vericl::cooperative_differential_config(
                            &__extra_sizes, __vericl_seed, __coop_cd, __ref_desc, __dependency,
                        ),
                        result: __result,
                    }
                } else {
                    let __config = #differential_config;
                    ::vericl::Claim {
                        kind: ::vericl::ClaimKind::Tested,
                        check: "differential".to_string(),
                        backend: Some(__vericl_extra_backend.clone()),
                        config: __config,
                        result: __result,
                    }
                };
                entry.claims.push(__claim);
                entry.trusted.push(::vericl::shared_frontend_lane_trust(&__vericl_extra_backend));
            }
        }
    }
}

pub fn expand(input: TokenStream2) -> syn::Result<TokenStream2> {
    let SuiteInput(fields) = syn::parse2(input)?;
    let spec = build_spec(fields)?;

    let runtime_path = &spec.runtime;
    let evidence_lit = &spec.evidence;
    let sizes_exprs = &spec.sizes;

    // R7 (docs/design-2d-dispatch.md §10.3): a `dispatch(...)` kernel's block
    // size comes from its own clause, so the suite's `cube_dim:` field has
    // nothing to set. Two sources of truth for one launch parameter is how a
    // proof gets bound to a block size the launch does not use — the hazard
    // `cooperative(...)` already avoids by asserting the two equal.
    if let (Some(span), Some(_)) = (spec.cube_dim_span, spec.sizes_rank) {
        let names = spec
            .kernels
            .iter()
            .map(|k| format!("`{k}`"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(syn::Error::new(
            span,
            format!(
                "this suite declares tuple `sizes:`, so every kernel it lists ({names}) declares \
                 its own `dispatch(cube_dim = (...))` — and the suite's `cube_dim:` field has \
                 nothing to set. Two sources of truth for one launch parameter is how a proof \
                 gets bound to a block size the launch does not use. Remove `cube_dim:` from \
                 this suite, or remove the `dispatch(...)` clause from every kernel it lists"
            ),
        ));
    }

    // The declared cases, in this suite's own unit. A dispatch suite's tuples
    // are normalized to `[usize; 3]` with the unused axis at 1, which is exactly
    // the shape `conformance_case` derives its per-axis cube count from.
    let sizes_decl = match spec.sizes_rank {
        None => quote! { let __vericl_sizes: &[usize] = &[ #(#sizes_exprs),* ]; },
        Some(rank) => {
            let rows: Vec<TokenStream2> = spec
                .sizes
                .iter()
                .map(|e| {
                    let Expr::Tuple(t) = e else { unreachable!("checked in build_spec") };
                    let mut parts: Vec<TokenStream2> =
                        t.elems.iter().map(|x| quote!((#x) as usize)).collect();
                    while parts.len() < 3 {
                        parts.push(quote!(1usize));
                    }
                    quote!([ #(#parts),* ])
                })
                .collect();
            let _ = rank;
            quote! { let __vericl_sizes: &[[usize; 3]] = &[ #(#rows),* ]; }
        }
    };
    let seed_expr = &spec.seed;
    let cube_dim_expr = &spec.cube_dim;
    let prove_expr = &spec.prove;
    let frontend_independent_expr = &spec.frontend_independent;

    let kernel_blocks: Vec<TokenStream2> = spec.kernels.iter().map(|k| kernel_block(k, spec.sizes_rank)).collect();

    let extra_lane_block = match &spec.extra_lane {
        None => TokenStream2::new(),
        Some((cfg_predicate, path)) => {
            let extra_kernel_blocks: Vec<TokenStream2> =
                spec.kernels.iter().map(|k| extra_lane_kernel_block(k, spec.sizes_rank)).collect();
            quote! {
                #[cfg(#cfg_predicate)]
                {
                    type __VericlExtraR = #path;
                    let __vericl_extra_device = ::core::default::Default::default();
                    let __vericl_extra_client =
                        <__VericlExtraR as ::cubecl::prelude::Runtime>::client(&__vericl_extra_device);
                    let __vericl_extra_backend = format!(
                        "{:?}",
                        <__VericlExtraR as ::cubecl::prelude::Runtime>::name(&__vericl_extra_client),
                    );
                    println!("vericl conformance — additional lane, backend {}", __vericl_extra_backend);
                    #(#extra_kernel_blocks)*
                }
            }
        }
    };

    Ok(quote! {
        #[test]
        fn vericl_conformance() {
            type __VericlR = #runtime_path;
            let __vericl_device = ::core::default::Default::default();
            let __vericl_client = <__VericlR as ::cubecl::prelude::Runtime>::client(&__vericl_device);
            let __vericl_backend = format!(
                "{:?}",
                <__VericlR as ::cubecl::prelude::Runtime>::name(&__vericl_client),
            );
            println!("vericl conformance — backend {}", __vericl_backend);

            let __vericl_prove: bool = #prove_expr;
            if __vericl_prove && ::vericl_ir::z3_version().is_none() {
                panic!(
                    "proved claims require z3 on PATH (macOS: brew install z3; Debian/Ubuntu: \
                     apt install z3) — or set prove: false to omit proved claims from evidence"
                );
            }
            let __vericl_solver: Option<String> = if __vericl_prove {
                Some(::vericl_ir::z3_version().map(|v| format!("z3 {v}")).expect("checked above"))
            } else {
                None
            };

            let __vericl_frontend_independent: bool = #frontend_independent_expr;
            let __vericl_seed: u64 = #seed_expr;
            let __vericl_cube_dim: u32 = #cube_dim_expr;
            #sizes_decl

            let mut entries: ::std::vec::Vec<::vericl::Entry> = ::std::vec::Vec::new();

            #(#kernel_blocks)*

            #extra_lane_block

            let current = ::vericl::Manifest::new(entries);
            let __vericl_evidence_path =
                ::std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(#evidence_lit);

            if ::std::env::var("VERICL_UPDATE").is_ok() {
                if let Some(bad) = current.entries.iter().find(|e| {
                    e.claims.iter().any(|c| matches!(c.result, ::vericl::ClaimResult::Fail { .. }))
                }) {
                    panic!(
                        "refusing to store failing evidence for kernel `{}` — fix the kernel or \
                         its contract first",
                        bad.kernel
                    );
                }
                // Proof-SCOPE changes, surfaced before the file is overwritten
                // (round-11 review, risk-8 residual). `VERICL_UPDATE` refuses
                // nothing by construction, so a change that keeps every claim
                // present and passing while shrinking what it proves — a bounds
                // walk that started bailing out early, say — would otherwise be
                // absorbed into the committed manifest with nothing on screen.
                // A warning rather than a refusal: an obligation count moves
                // legitimately whenever the kernel body does.
                if let Ok(__vericl_prev) = ::vericl::Manifest::load(&__vericl_evidence_path) {
                    let __vericl_scope = ::vericl::obligation_count_changes(&__vericl_prev, &current);
                    if !__vericl_scope.is_empty() {
                        println!(
                            "vericl WARNING — proof scope changed in this update ({} kernel(s)); \
                             confirm each is intended before committing the manifest:",
                            __vericl_scope.len()
                        );
                        for __vericl_line in &__vericl_scope {
                            println!("  {}", __vericl_line);
                        }
                    }
                }
                current.save(&__vericl_evidence_path).unwrap_or_else(|e| {
                    panic!(
                        "vericl: could not write the evidence manifest to {} ({e}) — check the \
                         `evidence:` path is writable and its parent directory exists",
                        __vericl_evidence_path.display()
                    )
                });
                println!("vericl evidence written to {}", __vericl_evidence_path.display());
            } else {
                let stored = ::vericl::Manifest::load(&__vericl_evidence_path).unwrap_or_else(|e| {
                    panic!(
                        "no stored vericl evidence at {} ({e}); run with VERICL_UPDATE=1 set to \
                         seed it",
                        __vericl_evidence_path.display()
                    )
                });
                let problems = ::vericl::verify(&stored, &current);
                assert!(problems.is_empty(), "vericl evidence problems:\n{}", problems.join("\n"));
                println!("vericl evidence OK: identities match, all claims pass");
            }
        }
    })
}
