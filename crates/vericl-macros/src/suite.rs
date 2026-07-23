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

use proc_macro2::{Ident, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Expr, LitStr, Path, Token};

enum SuiteField {
    Runtime(Path),
    Kernels(Vec<Ident>),
    Evidence(LitStr),
    Sizes(Vec<Expr>),
    Seed(Expr),
    CubeDim(Expr),
    Prove(Expr),
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
                     evidence, sizes, seed, cube_dim, prove, extra_lane"
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
    prove: Expr,
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
    let mut extra_lane: Option<(TokenStream2, Path)> = None;

    let dup = |field: &str| -> syn::Error {
        syn::Error::new(proc_macro2::Span::call_site(), format!("suite!: duplicate `{field}` field"))
    };

    for f in fields {
        match f {
            SuiteField::Runtime(p) => {
                if runtime.is_some() {
                    return Err(dup("runtime"));
                }
                runtime = Some(p);
            }
            SuiteField::Kernels(k) => {
                if kernels.is_some() {
                    return Err(dup("kernels"));
                }
                kernels = Some(k);
            }
            SuiteField::Evidence(e) => {
                if evidence.is_some() {
                    return Err(dup("evidence"));
                }
                evidence = Some(e);
            }
            SuiteField::Sizes(s) => {
                if sizes.is_some() {
                    return Err(dup("sizes"));
                }
                sizes = Some(s);
            }
            SuiteField::Seed(s) => {
                if seed.is_some() {
                    return Err(dup("seed"));
                }
                seed = Some(s);
            }
            SuiteField::CubeDim(c) => {
                if cube_dim.is_some() {
                    return Err(dup("cube_dim"));
                }
                cube_dim = Some(c);
            }
            SuiteField::Prove(p) => {
                if prove.is_some() {
                    return Err(dup("prove"));
                }
                prove = Some(p);
            }
            SuiteField::ExtraLane { cfg_predicate, path } => {
                if extra_lane.is_some() {
                    return Err(dup("extra_lane"));
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

    Ok(SuiteSpec {
        runtime,
        kernels,
        evidence,
        sizes: sizes.unwrap_or_else(default_sizes),
        seed: seed.unwrap_or_else(|| syn::parse_quote!(0xE901u64)),
        cube_dim: cube_dim.unwrap_or_else(|| syn::parse_quote!(256u32)),
        prove: prove.unwrap_or_else(|| syn::parse_quote!(true)),
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
/// `Entry`.
fn kernel_block(kernel: &Ident) -> TokenStream2 {
    let kmod = format_ident!("{}_vericl", kernel);
    let salt = kernel_salt(&kernel.to_string());
    quote! {
        {
            let __outcomes: ::std::vec::Vec<::vericl::CaseOutcome> = __vericl_sizes
                .iter()
                .map(|&n| {
                    #kmod::conformance_case::<__VericlR>(
                        &__vericl_client,
                        n,
                        __vericl_seed ^ #salt ^ (n as u64),
                        __vericl_cube_dim,
                    )
                })
                .collect();

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

            let mut __claims = ::std::vec![::vericl::Claim {
                kind: ::vericl::ClaimKind::Tested,
                check: "differential".to_string(),
                backend: Some(__vericl_backend.clone()),
                config: ::vericl::differential_config(__vericl_sizes, __vericl_seed, __vericl_cube_dim),
                result: __result,
            }];

            let mut __trusted = ::vericl::reference_twin_trust();
            __trusted.push(::vericl::backend_buffer_trust(&__vericl_backend));
            __trusted.push(::vericl::GPU_HARDWARE_TRUST.to_string());

            let mut __identity = #kmod::identity();

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
                    })
                    .collect();
                let __prove_result = ::vericl_ir::prove_bounds_freedom(&__def, &__buffers, &__assumes);
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
                    check: "smt-oob-freedom".to_string(),
                    backend: None,
                    config: ::vericl::proved_config(
                        __vericl_solver.as_deref().expect("prove checked z3 above"),
                        __obligations,
                    ),
                    result: __claim_result,
                });
                __trusted.extend(::vericl::proved_bounds_trust(
                    __vericl_solver.as_deref().expect("prove checked z3 above"),
                ));
            }

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
fn extra_lane_kernel_block(kernel: &Ident) -> TokenStream2 {
    let kmod = format_ident!("{}_vericl", kernel);
    let salt = kernel_salt(&kernel.to_string());
    quote! {
        {
            let __outcomes: ::std::vec::Vec<::vericl::CaseOutcome> = __vericl_sizes
                .iter()
                .map(|&n| {
                    #kmod::conformance_case::<__VericlExtraR>(
                        &__vericl_extra_client,
                        n,
                        __vericl_seed ^ #salt ^ (n as u64),
                        __vericl_cube_dim,
                    )
                })
                .collect();
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
            let __claim = ::vericl::Claim {
                kind: ::vericl::ClaimKind::Tested,
                check: "differential".to_string(),
                backend: Some(__vericl_extra_backend.clone()),
                config: ::vericl::differential_config(__vericl_sizes, __vericl_seed, __vericl_cube_dim),
                result: __result,
            };
            if let Some(entry) = entries.iter_mut().find(|e| e.kernel == #kmod::contract().kernel) {
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
    let seed_expr = &spec.seed;
    let cube_dim_expr = &spec.cube_dim;
    let prove_expr = &spec.prove;

    let kernel_blocks: Vec<TokenStream2> = spec.kernels.iter().map(kernel_block).collect();

    let extra_lane_block = match &spec.extra_lane {
        None => TokenStream2::new(),
        Some((cfg_predicate, path)) => {
            let extra_kernel_blocks: Vec<TokenStream2> =
                spec.kernels.iter().map(extra_lane_kernel_block).collect();
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

            let __vericl_seed: u64 = #seed_expr;
            let __vericl_cube_dim: u32 = #cube_dim_expr;
            let __vericl_sizes: &[usize] = &[ #(#sizes_exprs),* ];

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
                current.save(&__vericl_evidence_path).expect("write vericl evidence manifest");
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
