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
    /// The bracket group's own span travels with the list so an EMPTY
    /// `kernels: []` can be blamed on the brackets — an empty `Vec<Ident>` has
    /// no span of its own.
    Kernels { idents: Vec<Ident>, span: proc_macro2::Span },
    Evidence(LitStr),
    /// Same, for `sizes: []`.
    Sizes { exprs: Vec<Expr>, span: proc_macro2::Span },
    Seed(Expr),
    CubeDim(Expr),
    Prove(Expr),
    /// A literal `true`/`false`, never an arbitrary expression — see
    /// [`SuiteSpec::frontend_independent`].
    FrontendIndependent(syn::LitBool),
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
                let bracket = syn::bracketed!(content in input);
                let idents: Punctuated<Ident, Token![,]> = Punctuated::parse_terminated(&content)?;
                Ok(SuiteField::Kernels {
                    idents: idents.into_iter().collect(),
                    span: bracket.span.join(),
                })
            }
            "evidence" => Ok(SuiteField::Evidence(input.parse()?)),
            "sizes" => {
                let content;
                let bracket = syn::bracketed!(content in input);
                let exprs: Punctuated<Expr, Token![,]> = Punctuated::parse_terminated(&content)?;
                Ok(SuiteField::Sizes {
                    exprs: exprs.into_iter().collect(),
                    span: bracket.span.join(),
                })
            }
            "seed" => Ok(SuiteField::Seed(input.parse()?)),
            "cube_dim" => Ok(SuiteField::CubeDim(input.parse()?)),
            "prove" => Ok(SuiteField::Prove(input.parse()?)),
            "frontend_independent" => {
                let lit: syn::LitBool = input.parse().map_err(|e| {
                    syn::Error::new(
                        e.span(),
                        "suite!: `frontend_independent:` takes a literal `true` or `false`, not an \
                         expression. It selects which CLAIM this suite's evidence records — an \
                         independent execution lane, or one sharing CubeCL's front end with the \
                         kernel under test — and a claim that depends on a runtime value cannot be \
                         checked when it is made. In most cases you should omit the field entirely \
                         and let it be derived from `runtime:`",
                    )
                })?;
                Ok(SuiteField::FrontendIndependent(lit))
            }
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
    /// execution lane relative to the macro-derived twin. `true` for a GPU
    /// backend like wgpu — a genuinely different codegen path — where the
    /// entry's trusted list records `GPU_HARDWARE_TRUST`. `false` for a lane
    /// that shares CubeCL's front end AND is the only execution lane (the f64
    /// case: WGSL has no f64, so cubecl-cpu is the sole honest backend); then
    /// the trusted list swaps in `HOST_HARDWARE_TRUST` + the explicit
    /// `shared_frontend_lane_trust` caveat, so evidence never implies an
    /// independent execution lane exists where there is none — only the twin is
    /// independent.
    ///
    /// # Not a free bool, and not a strong default (external consumer review)
    ///
    /// This used to be an arbitrary `Expr` defaulting to `true`, which meant
    /// the STRONG claim — "a genuinely independent execution lane corroborated
    /// the twin" — was what a caller got by saying nothing at all, on any
    /// runtime. That is exactly backwards: the strong claim is the one that
    /// must be earned.
    ///
    /// It is now [`derive_frontend_independence`]'s answer:
    ///
    /// * a **recognized** runtime decides it (wgpu-family → independent;
    ///   cubecl-cpu → shared front end), and an explicit `true` on a runtime
    ///   known to share the front end is a compile error rather than a bool
    ///   somebody typed;
    /// * an **unrecognized** runtime requires an explicit declaration. Neither
    ///   default is safe there: `true` is the accidental strong claim being
    ///   closed, and `false` would record `HOST_HARDWARE_TRUST` ("host CPU
    ///   execution hardware") for what may well be a discrete GPU. So the macro
    ///   asks, naming both answers and what each records.
    ///
    /// An explicit `false` on a recognized-independent runtime is always
    /// allowed: downgrading to the weak claim cannot overstate anything.
    frontend_independent: bool,
    extra_lane: Option<(TokenStream2, Path)>,
}

/// Whether a suite's primary execution lane is independent of the CubeCL front
/// end the kernel under test goes through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LaneIndependence {
    /// A genuinely different codegen path from the macro-derived twin — the
    /// differential's second leg is worth something on its own.
    Independent,
    /// Shares CubeCL's macro expansion and IR with the kernel under test. Only
    /// the derived sequential twin is an independent reference.
    SharedFrontend,
}

/// What the runtime *type* says about lane independence, by its final path
/// segment.
///
/// Syntactic recognition rather than a trait, and deliberately so: a trait
/// would have to live in a crate that depends on `cubecl` with the wgpu/cpu
/// features enabled, which is a heavier dependency than this decision is worth
/// — `vericl` core is cubecl-free by design and `vericl-ir` takes cubecl with
/// `default-features = false`. The failure mode of getting it wrong is bounded
/// by only recognizing runtimes this repository actually exercises and
/// measured: anything else lands on the explicit-declaration error rather than
/// on a guess.
fn recognize_runtime(path: &Path) -> Option<LaneIndependence> {
    match path.segments.last()?.ident.to_string().as_str() {
        // wgpu compiles the CubeCL IR to WGSL/SPIR-V and runs it on a real
        // device driver: a different code generator from rustc, which is what
        // makes the differential's two legs independent.
        "WgpuRuntime" => Some(LaneIndependence::Independent),
        // cubecl-cpu goes through the same CubeCL macro expansion and IR as the
        // kernel under test, so agreement with it does not corroborate that
        // pipeline — it exercises it twice.
        "CpuRuntime" => Some(LaneIndependence::SharedFrontend),
        _ => None,
    }
}

/// Resolve [`SuiteSpec::frontend_independent`] from the runtime type and the
/// (optional) explicit declaration. See that field's doc for the rules.
fn derive_frontend_independence(
    runtime: &Path,
    declared: Option<&syn::LitBool>,
) -> syn::Result<bool> {
    let recognized = recognize_runtime(runtime);
    match (recognized, declared) {
        (Some(LaneIndependence::SharedFrontend), Some(lit)) if lit.value => Err(syn::Error::new(
            lit.span(),
            "suite!: this suite's `runtime:` is a CubeCL runtime that shares CubeCL's front end \
             (macro expansion + IR) with the kernel under test, so declaring \
             `frontend_independent: true` would record a claim that is not true: the entry's \
             trusted list would say `GPU hardware` and omit the shared-front-end caveat, implying \
             an independent execution lane corroborated the derived twin when none did. Remove the \
             field (it is derived correctly from `runtime:`), or write `false`",
        )),
        // An explicit declaration otherwise stands. `false` is always safe —
        // it can only record the WEAKER claim.
        (_, Some(lit)) => Ok(lit.value),
        (Some(LaneIndependence::Independent), None) => Ok(true),
        (Some(LaneIndependence::SharedFrontend), None) => Ok(false),
        (None, None) => Err(syn::Error::new(
            runtime.span(),
            "suite!: vericl does not recognize this runtime, so it cannot tell whether this \
             execution lane is independent of the CubeCL front end the kernel under test goes \
             through — and neither default is safe. Add `frontend_independent: true` if this \
             runtime compiles the CubeCL IR with its own code generator and runs it on its own \
             device (the entry then records `GPU hardware` as trusted and the differential's two \
             legs are independent); add `frontend_independent: false` if it shares CubeCL's macro \
             expansion and IR with the kernel (the entry then records `host CPU execution \
             hardware` plus an explicit caveat that only the derived twin is independent). \
             Recognized without a declaration: `WgpuRuntime` (independent), `CpuRuntime` (shared \
             front end)",
        )),
    }
}

/// A `sizes:` entry that is the literal `0` makes its case compare **zero
/// elements** — `CompareReport { checked: 0, mismatches: 0, pass: true }`,
/// i.e. agreement over nothing — for every kernel whose body is guarded by
/// `pos < y.len()`. Refused for the same reason `sizes: []` is.
///
/// Only *literal* zeroes are catchable here (a size behind a `const` is an
/// opaque `Expr` to a proc macro); `vericl::CaseOutcome::pass` carries the
/// runtime backstop for the rest.
fn reject_zero_sizes(sizes: &[Expr]) -> syn::Result<()> {
    fn is_literal_zero(e: &Expr) -> bool {
        matches!(e, Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(i), .. })
            if i.base10_digits() == "0")
    }
    for e in sizes {
        let offending = match e {
            Expr::Tuple(t) => t.elems.iter().find(|x| is_literal_zero(x)),
            other if is_literal_zero(other) => Some(other),
            _ => None,
        };
        if let Some(bad) = offending {
            return Err(syn::Error::new(
                bad.span(),
                "suite!: a `sizes:` entry of 0 runs a case that compares ZERO elements — every \
                 report comes back `checked: 0, mismatches: 0, pass: true`, which is agreement \
                 over nothing, not agreement. Use a positive size (`1` is the honest degenerate \
                 case and is already in the default list)",
            ));
        }
    }
    Ok(())
}

fn default_sizes() -> Vec<Expr> {
    ["1usize", "7usize", "256usize", "1000usize", "1027usize", "4096usize", "65536usize"]
        .iter()
        .map(|s| syn::parse_str(s).expect("literal default size parses"))
        .collect()
}

fn build_spec(fields: Punctuated<SuiteField, Token![,]>) -> syn::Result<SuiteSpec> {
    let mut runtime: Option<Path> = None;
    let mut kernels: Option<(Vec<Ident>, proc_macro2::Span)> = None;
    let mut evidence: Option<LitStr> = None;
    let mut sizes: Option<(Vec<Expr>, proc_macro2::Span)> = None;
    let mut seed: Option<Expr> = None;
    let mut cube_dim: Option<Expr> = None;
    let mut prove: Option<Expr> = None;
    let mut frontend_independent: Option<syn::LitBool> = None;
    let mut extra_lane: Option<(TokenStream2, Path)> = None;

    // Underline the offending (duplicate) field's own tokens, not the whole
    // `suite!` invocation.
    let dup = |field: &str, span: proc_macro2::Span| -> syn::Error {
        syn::Error::new(span, format!("suite!: duplicate `{field}` field"))
    };

    for f in fields {
        match f {
            SuiteField::Runtime(p) => {
                if runtime.is_some() {
                    return Err(dup("runtime", p.span()));
                }
                runtime = Some(p);
            }
            SuiteField::Kernels { idents, span } => {
                if kernels.is_some() {
                    return Err(dup("kernels", span));
                }
                kernels = Some((idents, span));
            }
            SuiteField::Evidence(e) => {
                if evidence.is_some() {
                    return Err(dup("evidence", e.span()));
                }
                evidence = Some(e);
            }
            SuiteField::Sizes { exprs, span } => {
                if sizes.is_some() {
                    return Err(dup("sizes", span));
                }
                sizes = Some((exprs, span));
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
    let (kernels, kernels_span) = kernels.ok_or_else(|| {
        syn::Error::new(call_site, "suite! requires a `kernels: [k1, k2, ...]` field")
    })?;
    let evidence = evidence.ok_or_else(|| {
        syn::Error::new(call_site, "suite! requires an `evidence: \"path/to/vericl.json\"` field")
    })?;

    // --- VACUOUS-SUITE REJECTIONS (external consumer review) ---
    //
    // Both of these expand to a test that runs zero checks and reports
    // success, because every gate downstream quantifies over a set that is
    // empty. `outcomes.iter().all(CaseOutcome::pass)` is `true` over no
    // outcomes; `verify` over no entries finds no problems; the run prints
    // "vericl evidence OK: identities match, all claims pass". A green
    // conformance suite that checked nothing is the worst possible output for
    // an evidence tool, so both are refused where they are written.
    if kernels.is_empty() {
        return Err(syn::Error::new(
            kernels_span,
            "suite!: `kernels: []` declares a conformance suite over no kernels. Every check the \
             suite performs quantifies over this list, so the generated test would run nothing, \
             write an empty evidence manifest, and print `vericl evidence OK` — a green result \
             that establishes nothing. List at least one `#[vericl::kernel]`-annotated kernel, or \
             delete the `suite!` invocation",
        ));
    }
    if let Some((list, span)) = &sizes {
        if list.is_empty() {
            return Err(syn::Error::new(
                *span,
                "suite!: `sizes: []` declares a differential over no cases. `all()` over zero \
                 outcomes is `true`, so every kernel would record a PASSING `tested` claim having \
                 executed nothing. Declare at least one size, or omit the field to take the \
                 default list",
            ));
        }
        reject_zero_sizes(list)?;
    }

    let frontend_independent =
        derive_frontend_independence(&runtime, frontend_independent.as_ref())?;

    // --- multi-axis suite detection (docs/design-2d-dispatch.md §4.8). A
    // 2-D/3-D suite is spelled by its SIZES: `sizes: [(37, 19), (64, 64)]`.
    // Mixing tuple and scalar entries is rejected rather than guessed — the two
    // are different units (extents vs. thread counts), and round 8's units
    // discipline says decide it, not paper over it.
    let sizes_rank = match sizes.as_ref().map(|(l, _)| l.as_slice()) {
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
        sizes: sizes.map(|(l, _)| l).unwrap_or_else(default_sizes),
        seed: seed.unwrap_or_else(|| syn::parse_quote!(0xE901u64)),
        cube_dim_span: cube_dim.as_ref().map(|c| c.span()),
        sizes_rank,
        cube_dim: cube_dim.unwrap_or_else(|| syn::parse_quote!(256u32)),
        prove: prove.unwrap_or_else(|| syn::parse_quote!(true)),
        frontend_independent,
        extra_lane,
    })
}

/// The RNG salt-scheme tag, recorded verbatim in `Provenance::salt_scheme` on
/// every manifest this macro produces (round-13A fix 7). A differential claim's
/// `config.seed` records only the suite's *base* seed; each case actually draws
/// at `seed ^ kernel_salt(name) ^ case_salt(shape)`. Change either derivation —
/// the FNV constants in [`kernel_salt`], or the SplitMix multipliers in
/// [`case_call_tokens`]'s per-case decorrelation — and every kernel is retested
/// against a different input distribution while `config.seed` and the identity
/// stay byte-identical, so the evidence would look fresh while describing inputs
/// it was never produced under. Bumping this tag in the same edit makes
/// `verify` treat the old evidence as stale. `the_salt_scheme_pins_its_exact_
/// outputs` fails the moment a salt derivation changes without a bump, so the
/// discipline is enforced rather than trusted.
pub(crate) const SALT_SCHEME: &str = "fnv1a-name^splitmix-case/v1";

/// Deterministic FNV-1a 64-bit hash of a kernel name, used only to decorrelate
/// different kernels' RNG streams within one suite run (two kernels sharing a
/// seed would otherwise draw from the same underlying bit stream — harmless
/// since their parameter shapes differ, but needlessly suspicious). Computed
/// at macro-expansion time so it's a fixed, reproducible per-kernel constant,
/// not a hand-maintained salt list.
///
/// The [`SALT_SCHEME`] tag must be bumped whenever this derivation changes; the
/// `the_salt_scheme_pins_its_exact_outputs` test enforces that coupling.
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
            // IR-level identity is recorded UNCONDITIONALLY (external consumer
            // review, fix 3). It used to be set only inside the `if
            // __vericl_prove` branches, which tied an IDENTITY fact to whether
            // a PROOF ran — two unrelated things. Extracting the expanded IR
            // needs no solver, so `prove: false` evidence recorded `ir_hash:
            // null` and silently lost its only defence against IR-level drift
            // that leaves the source untouched (a CubeCL upgrade changing
            // codegen, say). `Identity::ir_hash`'s own doc already claimed the
            // harness fills it in; now it does.
            let mut __identity = #kmod::identity();
            __identity.ir_hash =
                Some(::vericl_ir::kernel_ir_hash(&#kmod::kernel_definition()));
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
    let frontend_independent_lit = spec.frontend_independent;
    // vericl-macros' own version, resolved when THIS crate was compiled — a
    // proc-macro crate cannot export a constant, so the literal is emitted.
    let macros_version = env!("CARGO_PKG_VERSION");
    // The salt-scheme tag, emitted the same way (a proc-macro crate cannot
    // export a const for the runner to read). Bumped whenever the salt
    // derivation changes; recorded in `Provenance::salt_scheme` (fix 7).
    let salt_scheme_lit = SALT_SCHEME;

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
                    // Record the lane this run executed. The committed manifest
                    // is the superset of lanes, so under `--features cpu` this
                    // lane matches the file and is VERIFIED; a file that does
                    // NOT record a lane this run produced is a `verify` problem
                    // (round-13A fixes 1+2), which is what catches stripping the
                    // cpu lane out of the committed evidence.
                    __vericl_lanes.push(__vericl_extra_backend.clone());
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

            let __vericl_frontend_independent: bool = #frontend_independent_lit;
            let __vericl_seed: u64 = #seed_expr;
            let __vericl_cube_dim: u32 = #cube_dim_expr;
            #sizes_decl

            // Execution lanes actually run, primary first. `mut` is only used
            // when an `extra_lane` is cfg-enabled.
            #[allow(unused_mut)]
            let mut __vericl_lanes: ::std::vec::Vec<String> = ::std::vec![__vericl_backend.clone()];

            let mut entries: ::std::vec::Vec<::vericl::Entry> = ::std::vec::Vec::new();

            #(#kernel_blocks)*

            #extra_lane_block

            // The verification-environment fingerprint (external consumer
            // review, fix 4). `Provenance::current()` supplies what vericl core
            // can see on its own (rustc, target triple, its own version, the
            // cubecl pin); the rest is only visible from here.
            let mut __vericl_provenance = ::vericl::Provenance::current();
            __vericl_provenance.vericl_ir = ::vericl_ir::VERSION.to_string();
            __vericl_provenance.vericl_macros = #macros_version.to_string();
            __vericl_provenance.z3 = __vericl_solver.clone();
            __vericl_provenance.lanes = __vericl_lanes;
            // The RNG salt-scheme tag (round-13A fix 7). `config.seed` records
            // only the base seed; the per-kernel/per-case salt fold is what a
            // change here would silently alter, so the scheme is recorded and
            // `verify` compares it.
            __vericl_provenance.salt_scheme = #salt_scheme_lit.to_string();
            // The graphics-API / backend CLASS the runtime exposes cheaply
            // (`ComputeClient::info()`), not device identity. For wgpu this is
            // the SELECTED backend (Metal / Vulkan / Dx12) — two materially
            // different code generators that both report the runtime name
            // `wgpu<wgsl>`, so recording only the name would let evidence
            // measured on one verify against the other. It does NOT distinguish
            // two Metal GPUs or a driver update (both report `"Metal"`); real
            // device identity would need `wgpu::AdapterInfo`, which this
            // runtime-generic call does not surface. See `Provenance::device`.
            __vericl_provenance.device = Some(format!("{:?}", __vericl_client.info()));

            let current = ::vericl::Manifest::with_provenance(entries, __vericl_provenance);
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
                        // Direct stderr, NOT println!: libtest captures the
                        // print macros on a passing test, and `VERICL_UPDATE`
                        // passes (it writes and returns). A proof-scope shrink
                        // the author did not intend must be visible in that
                        // passing run, so it goes to real fd 2, which the
                        // capture does not intercept (round-13A fix 1).
                        use ::std::io::Write as _;
                        let __vericl_err = ::std::io::stderr();
                        let mut __vericl_err = __vericl_err.lock();
                        let _ = writeln!(
                            __vericl_err,
                            "vericl WARNING — proof scope changed in this update ({} kernel(s)); \
                             confirm each is intended before committing the manifest:",
                            __vericl_scope.len()
                        );
                        for __vericl_line in &__vericl_scope {
                            let _ = writeln!(__vericl_err, "  {}", __vericl_line);
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
                // Evidence the file records on an execution lane this run did
                // not exercise (a `cfg` feature that is off — e.g. the cpu lane
                // under a default `cargo test`). Not a mismatch: that lane is
                // strictly verified under its own feature. Written to real
                // stderr, NOT println!, because libtest captures the print
                // macros on a passing test — so this note is genuinely on
                // screen on every run rather than swallowed (round-13A fix 1).
                let __vericl_extra = ::vericl::unrecorded_evidence(&stored, &current);
                if !__vericl_extra.is_empty() {
                    use ::std::io::Write as _;
                    let __vericl_err = ::std::io::stderr();
                    let mut __vericl_err = __vericl_err.lock();
                    let _ = writeln!(
                        __vericl_err,
                        "vericl note — {} recorded item(s) in {} were NOT re-verified in this run \
                         (an execution lane this configuration did not exercise); run under the \
                         lane's feature (e.g. `--features cpu`) to verify them:",
                        __vericl_extra.len(),
                        __vericl_evidence_path.display(),
                    );
                    for __vericl_line in &__vericl_extra {
                        let _ = writeln!(__vericl_err, "  {}", __vericl_line);
                    }
                }
                let problems = ::vericl::verify(&stored, &current);
                assert!(problems.is_empty(), "vericl evidence problems:\n{}", problems.join("\n"));
                println!(
                    "vericl evidence OK: {} kernel entr{} verified complete — identity, contract, \
                     claim set, configs, results and trust list all match, in a matching \
                     verification environment",
                    current.entries.len(),
                    if current.entries.len() == 1 { "y" } else { "ies" },
                );
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Direct tests of the `suite!` macro itself (external consumer review, fix 6).
//
// Every one of these runs `expand()` on a token stream and inspects the result
// — no GPU, no cubecl, no z3, no evidence file. The macro's field parsing, its
// rejections, and the SHAPE of what it generates (which mode-selection branch
// exists, whether `ir_hash` is computed unconditionally, what the provenance
// record is populated with) were previously testable only by compiling and
// running a real suite against real hardware, which meant they were not tested
// at all.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    /// Expand, expecting success; returns the generated tokens as text.
    fn ok(src: TokenStream2) -> String {
        match expand(src) {
            Ok(ts) => ts.to_string(),
            Err(e) => panic!("expected a successful expansion, got error: {e}"),
        }
    }

    /// Expand, expecting a compile error; returns its message.
    fn err(src: TokenStream2) -> String {
        match expand(src) {
            Ok(_) => panic!("expected a compile error, got a successful expansion"),
            Err(e) => e.to_string(),
        }
    }

    /// A minimal well-formed suite, as a builder the tests vary one field of.
    fn minimal() -> TokenStream2 {
        quote! {
            runtime: cubecl::wgpu::WgpuRuntime,
            kernels: [axpy],
            evidence: "evidence/vericl.json",
        }
    }

    // ---- field parsing ----

    #[test]
    fn every_field_parses_and_order_does_not_matter() {
        // All nine fields, in declaration order…
        let forward = ok(quote! {
            runtime: cubecl::wgpu::WgpuRuntime,
            kernels: [axpy, fir3],
            evidence: "evidence/vericl.json",
            sizes: [1usize, 256usize],
            seed: 7u64,
            cube_dim: 64u32,
            prove: false,
            frontend_independent: true,
            extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime),
        });
        // …and shuffled. `suite!`'s fields are order-independent by design.
        let shuffled = ok(quote! {
            extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime),
            frontend_independent: true,
            prove: false,
            cube_dim: 64u32,
            seed: 7u64,
            sizes: [1usize, 256usize],
            evidence: "evidence/vericl.json",
            kernels: [axpy, fir3],
            runtime: cubecl::wgpu::WgpuRuntime,
        });
        assert_eq!(forward, shuffled, "field order must not change the expansion");
    }

    #[test]
    fn a_trailing_comma_is_optional() {
        let with = ok(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [axpy], evidence: "e.json",
        });
        let without = ok(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [axpy], evidence: "e.json"
        });
        assert_eq!(with, without);
    }

    #[test]
    fn an_unknown_field_is_named_and_the_valid_set_listed() {
        let m = err(quote! {
            runtime: cubecl::wgpu::WgpuRuntime,
            kernels: [axpy],
            evidence: "e.json",
            paralellism: 4,
        });
        assert!(m.contains("unknown `suite!` field `paralellism`"), "{m}");
        assert!(m.contains("frontend_independent"), "{m}");
    }

    #[test]
    fn each_required_field_is_named_when_missing() {
        assert!(err(quote! { kernels: [axpy], evidence: "e.json" }).contains("`runtime:"), );
        assert!(err(quote! { runtime: cubecl::wgpu::WgpuRuntime, evidence: "e.json" })
            .contains("`kernels: [k1, k2, ...]`"));
        assert!(err(quote! { runtime: cubecl::wgpu::WgpuRuntime, kernels: [axpy] })
            .contains("`evidence:"));
    }

    #[test]
    fn every_field_rejects_a_duplicate() {
        for (name, dup) in [
            ("runtime", quote!(runtime: cubecl::cpu::CpuRuntime)),
            ("kernels", quote!(kernels: [other])),
            ("evidence", quote!(evidence: "other.json")),
            ("sizes", quote!(sizes: [4usize])),
            ("seed", quote!(seed: 1u64)),
            ("cube_dim", quote!(cube_dim: 32u32)),
            ("prove", quote!(prove: false)),
            ("frontend_independent", quote!(frontend_independent: false)),
            (
                "extra_lane",
                quote!(extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime)),
            ),
        ] {
            let base = quote! {
                runtime: cubecl::wgpu::WgpuRuntime,
                kernels: [axpy],
                evidence: "e.json",
                sizes: [4usize],
                seed: 0u64,
                cube_dim: 256u32,
                prove: true,
                frontend_independent: true,
                extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime),
            };
            let m = err(quote! { #base #dup, });
            assert!(m.contains(&format!("duplicate `{name}` field")), "{name}: {m}");
        }
    }

    #[test]
    fn extra_lane_shape_is_enforced() {
        let base = quote! { runtime: cubecl::wgpu::WgpuRuntime, kernels: [axpy], evidence: "e.json", };
        let m = err(quote! { #base extra_lane: (cubecl::cpu::CpuRuntime, cfg(feature = "cpu")), });
        assert!(m.contains("expects a `cfg(...)` predicate first"), "{m}");
        let m = err(quote! {
            #base extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime, 3),
        });
        assert!(m.contains("exactly these two entries"), "{m}");
    }

    // ---- VACUOUS-SUITE REJECTIONS (fix 2) ----

    /// The review's own shape: a suite over no kernels expands to a test that
    /// checks nothing and prints success.
    #[test]
    fn an_empty_kernel_list_is_rejected() {
        let m = err(quote! {
            runtime: cubecl::wgpu::WgpuRuntime,
            kernels: [],
            evidence: "e.json",
        });
        assert!(m.contains("`kernels: []`"), "{m}");
        assert!(m.contains("establishes nothing"), "{m}");
        // NEGATIVE CONTROL: one kernel and the same suite expands fine.
        let _ = ok(minimal());
    }

    /// `all()` over zero outcomes is `true`, so a suite with no sizes records a
    /// PASSING tested claim per kernel having executed nothing.
    #[test]
    fn an_empty_size_list_is_rejected() {
        let m = err(quote! {
            runtime: cubecl::wgpu::WgpuRuntime,
            kernels: [axpy],
            evidence: "e.json",
            sizes: [],
        });
        assert!(m.contains("`sizes: []`"), "{m}");
        assert!(m.contains("all()"), "{m}");
        // NEGATIVE CONTROL: one size is accepted, and so is omitting the field.
        let _ = ok(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [axpy], evidence: "e.json",
            sizes: [1usize],
        });
        let _ = ok(minimal());
    }

    /// A zero SIZE compares zero elements — the same vacuity one level down,
    /// in both the 1-D and the multi-axis spelling.
    #[test]
    fn a_zero_size_is_rejected_in_both_units() {
        let m = err(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [axpy], evidence: "e.json",
            sizes: [1usize, 0usize],
        });
        assert!(m.contains("compares ZERO elements"), "{m}");

        let m = err(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [e2d], evidence: "e.json",
            sizes: [(37, 19), (0, 4)],
        });
        assert!(m.contains("compares ZERO elements"), "{m}");

        // NEGATIVE CONTROL: 1 is the honest degenerate case and is accepted in
        // both units (the 2-D suite's own `(1, 1)` ground-truth shape).
        let _ = ok(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [axpy], evidence: "e.json",
            sizes: [1usize],
        });
        let _ = ok(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [e2d], evidence: "e.json",
            sizes: [(1, 1)],
        });
    }

    // ---- units / rank (pre-existing gates, now covered directly) ----

    #[test]
    fn mixed_scalar_and_tuple_sizes_are_rejected() {
        let m = err(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [k], evidence: "e.json",
            sizes: [(37, 19), 64usize],
        });
        assert!(m.contains("must have the same shape"), "{m}");
    }

    #[test]
    fn a_tuple_size_of_the_wrong_arity_is_rejected() {
        let m = err(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [k], evidence: "e.json",
            sizes: [(1, 2, 3, 4)],
        });
        assert!(m.contains("must have 2 or 3 elements"), "{m}");
        assert!(m.contains("has 4"), "{m}");
    }

    #[test]
    fn cube_dim_alongside_tuple_sizes_is_rejected_naming_the_kernels() {
        let m = err(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [transpose2d], evidence: "e.json",
            sizes: [(37, 19)], cube_dim: 256u32,
        });
        assert!(m.contains("Two sources of truth"), "{m}");
        assert!(m.contains("`transpose2d`"), "{m}");
    }

    // ---- frontend_independent (fix 5) ----

    #[test]
    fn lane_independence_is_recognized_from_the_runtime_type() {
        assert_eq!(
            recognize_runtime(&syn::parse_quote!(cubecl::wgpu::WgpuRuntime)),
            Some(LaneIndependence::Independent)
        );
        assert_eq!(
            recognize_runtime(&syn::parse_quote!(cubecl::cpu::CpuRuntime)),
            Some(LaneIndependence::SharedFrontend)
        );
        // Path length does not matter — only the final segment.
        assert_eq!(
            recognize_runtime(&syn::parse_quote!(WgpuRuntime)),
            Some(LaneIndependence::Independent)
        );
        assert_eq!(recognize_runtime(&syn::parse_quote!(cubecl::cuda::CudaRuntime)), None);
    }

    /// The whole point of fix 5: the STRONG claim is derived, never a default.
    #[test]
    fn the_strong_claim_is_derived_not_defaulted() {
        let wgpu: Path = syn::parse_quote!(cubecl::wgpu::WgpuRuntime);
        let cpu: Path = syn::parse_quote!(cubecl::cpu::CpuRuntime);
        let unknown: Path = syn::parse_quote!(some::vendor::FpgaRuntime);

        assert!(derive_frontend_independence(&wgpu, None).unwrap());
        assert!(!derive_frontend_independence(&cpu, None).unwrap());

        // An unrecognized runtime must not silently get EITHER claim.
        let e = derive_frontend_independence(&unknown, None).unwrap_err().to_string();
        assert!(e.contains("does not recognize this runtime"), "{e}");
        assert!(e.contains("frontend_independent: true"), "{e}");
        assert!(e.contains("frontend_independent: false"), "{e}");
        // …but an explicit declaration resolves it, either way.
        let yes: syn::LitBool = syn::parse_quote!(true);
        let no: syn::LitBool = syn::parse_quote!(false);
        assert!(derive_frontend_independence(&unknown, Some(&yes)).unwrap());
        assert!(!derive_frontend_independence(&unknown, Some(&no)).unwrap());

        // Downgrading a recognized-independent lane is always allowed.
        assert!(!derive_frontend_independence(&wgpu, Some(&no)).unwrap());
    }

    /// The accidental-strong-claim shape is a compile error, not a bool
    /// somebody typed.
    #[test]
    fn declaring_the_strong_claim_on_a_shared_frontend_runtime_is_a_compile_error() {
        let m = err(quote! {
            runtime: cubecl::cpu::CpuRuntime,
            kernels: [axpy_f64],
            evidence: "e.json",
            frontend_independent: true,
        });
        assert!(m.contains("shares CubeCL's front end"), "{m}");
        assert!(m.contains("would record a claim that is not true"), "{m}");

        // NEGATIVE CONTROL: the same suite without the field, and with `false`,
        // both expand — and to the SAME tokens, because `false` is what the
        // runtime derives.
        let derived = ok(quote! {
            runtime: cubecl::cpu::CpuRuntime, kernels: [axpy_f64], evidence: "e.json",
        });
        let explicit = ok(quote! {
            runtime: cubecl::cpu::CpuRuntime, kernels: [axpy_f64], evidence: "e.json",
            frontend_independent: false,
        });
        assert_eq!(derived, explicit);
    }

    /// An unrecognized runtime in a whole `suite!` invocation lands on the
    /// declaration requirement rather than on a guess.
    #[test]
    fn an_unrecognized_runtime_must_declare_its_lane_independence() {
        let m = err(quote! {
            runtime: some::vendor::FpgaRuntime, kernels: [axpy], evidence: "e.json",
        });
        assert!(m.contains("does not recognize this runtime"), "{m}");
        let _ = ok(quote! {
            runtime: some::vendor::FpgaRuntime, kernels: [axpy], evidence: "e.json",
            frontend_independent: true,
        });
    }

    /// It is a literal, not an expression: a runtime-valued claim cannot be
    /// checked when it is made.
    #[test]
    fn frontend_independent_refuses_a_non_literal() {
        let m = err(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [axpy], evidence: "e.json",
            frontend_independent: cfg!(feature = "cpu"),
        });
        assert!(m.contains("literal `true` or `false`"), "{m}");
    }

    /// The derived value reaches the generated code, and it is what selects the
    /// trust wording.
    #[test]
    fn the_derived_value_is_what_the_generated_code_branches_on() {
        let wgpu = ok(minimal());
        assert!(wgpu.contains("let __vericl_frontend_independent : bool = true"), "{wgpu}");
        let cpu = ok(quote! {
            runtime: cubecl::cpu::CpuRuntime, kernels: [axpy_f64], evidence: "e.json",
        });
        assert!(cpu.contains("let __vericl_frontend_independent : bool = false"), "{cpu}");
        // Both trust wordings are present in the expansion (the branch is at
        // run time); the bool above is what picks one.
        assert!(wgpu.contains("GPU_HARDWARE_TRUST"));
        assert!(wgpu.contains("HOST_HARDWARE_TRUST"));
    }

    // ---- generated shape: mode selection, ir_hash, provenance ----

    /// Both modes exist in one expansion and are selected by the environment
    /// variable at run time — the update path writes, the check path verifies.
    #[test]
    fn both_update_and_check_modes_are_generated() {
        let g = ok(minimal());
        assert!(g.contains("VERICL_UPDATE"), "{g}");
        // update path
        assert!(g.contains("refusing to store failing evidence"), "{g}");
        assert!(g.contains("obligation_count_changes"), "{g}");
        assert!(g.contains("current . save"), "{g}");
        // check path
        assert!(g.contains(":: vericl :: verify"), "{g}");
        assert!(g.contains("unrecorded_evidence"), "{g}");
        assert!(g.contains("no stored vericl evidence at"), "{g}");
    }

    /// FIX 3, structurally: the IR hash is assigned on the line after the
    /// identity is built, before any `prove` branch is entered — so it cannot
    /// be conditional on a solver being available. It is also assigned exactly
    /// once per kernel.
    #[test]
    fn the_ir_hash_is_computed_unconditionally() {
        for prove in [quote!(true), quote!(false)] {
            let g = ok(quote! {
                runtime: cubecl::wgpu::WgpuRuntime, kernels: [axpy], evidence: "e.json",
                prove: #prove,
            });
            assert!(
                g.contains(
                    "let mut __identity = axpy_vericl :: identity () ; \
                     __identity . ir_hash = Some (:: vericl_ir :: kernel_ir_hash"
                ),
                "prove: {prove} — ir_hash must be set immediately after identity(), \
                 unconditionally: {g}"
            );
            assert_eq!(
                g.matches("__identity . ir_hash = Some").count(),
                1,
                "exactly one ir_hash assignment per kernel"
            );
        }
    }

    /// FIX 4: the provenance record is populated with every field only the call
    /// site can see, and it is what the manifest is built from.
    #[test]
    fn the_provenance_record_is_populated_and_carried_into_the_manifest() {
        let g = ok(minimal());
        assert!(g.contains(":: vericl :: Provenance :: current ()"), "{g}");
        for field in ["vericl_ir", "vericl_macros", "z3", "lanes", "salt_scheme", "device"] {
            assert!(
                g.contains(&format!("__vericl_provenance . {field} =")),
                "provenance field `{field}` is not populated: {g}"
            );
        }
        assert!(g.contains(":: vericl_ir :: VERSION"), "{g}");
        // vericl-macros' own version, as a literal (a proc-macro crate cannot
        // export a constant).
        assert!(g.contains(&format!("{:?}", env!("CARGO_PKG_VERSION"))), "{g}");
        assert!(g.contains("Manifest :: with_provenance (entries , __vericl_provenance)"), "{g}");
        // …and never the provenance-less constructor.
        assert!(!g.contains("Manifest :: new"), "{g}");
    }

    /// The primary lane is always recorded; an `extra_lane` appends itself
    /// inside its own `cfg` block, so the recorded lane list tracks the feature
    /// set the evidence was actually produced under.
    #[test]
    fn the_lane_list_tracks_the_cfg_enabled_lanes() {
        let plain = ok(minimal());
        assert!(plain.contains("__vericl_lanes : :: std :: vec :: Vec < String > = :: std :: vec ! [__vericl_backend . clone ()]"), "{plain}");
        assert!(!plain.contains("__vericl_lanes . push"), "{plain}");

        let extra = ok(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [axpy], evidence: "e.json",
            extra_lane: (cfg(feature = "cpu"), cubecl::cpu::CpuRuntime),
        });
        assert!(extra.contains("__vericl_lanes . push (__vericl_extra_backend . clone ())"), "{extra}");
        assert!(extra.contains("# [cfg (feature = \"cpu\")]"), "{extra}");
    }

    /// Per-kernel expansion scales with the list, and each kernel's generated
    /// block references its own `<name>_vericl` module.
    #[test]
    fn each_listed_kernel_gets_its_own_block() {
        let g = ok(quote! {
            runtime: cubecl::wgpu::WgpuRuntime,
            kernels: [axpy, fir3, mix_u32],
            evidence: "e.json",
        });
        for k in ["axpy_vericl", "fir3_vericl", "mix_u32_vericl"] {
            assert!(g.contains(&format!("{k} :: contract ()")), "{k} missing from expansion");
        }
        assert_eq!(g.matches("__identity . ir_hash = Some").count(), 3);
    }

    /// The per-kernel RNG salt is a pure function of the name, so two kernels
    /// in one suite never share a draw and the value is reproducible.
    #[test]
    fn the_kernel_salt_is_deterministic_and_name_dependent() {
        assert_eq!(kernel_salt("axpy"), kernel_salt("axpy"));
        assert_ne!(kernel_salt("axpy"), kernel_salt("fir3"));
        assert_ne!(kernel_salt("axpy"), kernel_salt("axpz"));
    }

    /// FIX 7 (round-13A) — the salt scheme is recorded in `Provenance::
    /// salt_scheme` so a salt-derivation change invalidates evidence, but that
    /// only works if a change is *noticed*. These exact-value pins fail the
    /// instant `kernel_salt`'s FNV constants change; the failure message is the
    /// reminder to bump `SALT_SCHEME` (and regenerate every manifest) in the
    /// same edit. `config.seed` alone would not have moved.
    #[test]
    fn the_salt_scheme_pins_its_exact_outputs() {
        // FNV-1a/64 of the kernel names in the shipped suites. If these change,
        // the input distribution every case is tested against changed with them.
        assert_eq!(
            kernel_salt("axpy"),
            0x326a_9c84_0d77_2967,
            "kernel_salt changed — bump SALT_SCHEME and regenerate all evidence"
        );
        assert_eq!(
            kernel_salt("block_sum_reduce"),
            0x6c87_edd9_4564_c209,
            "kernel_salt changed — bump SALT_SCHEME and regenerate all evidence"
        );
        assert_eq!(
            kernel_salt(""),
            0xcbf2_9ce4_8422_2325,
            "the FNV offset basis changed — bump SALT_SCHEME and regenerate all evidence"
        );
        // The tag itself is stable across this arc (the derivation did not
        // change), so committed evidence is byte-stable in this field.
        assert_eq!(SALT_SCHEME, "fnv1a-name^splitmix-case/v1");
    }

    /// A dispatch (tuple-`sizes`) suite and a 1-D suite generate different
    /// claim-config builders and different prover entry points — the choice is
    /// made at expansion time, not by a run-time `if`, because the two take
    /// different size types.
    #[test]
    fn the_units_choice_is_made_at_expansion_time() {
        let one_d = ok(minimal());
        assert!(one_d.contains("differential_config"), "{one_d}");
        assert!(!one_d.contains("differential_dispatch_config"), "{one_d}");
        assert!(one_d.contains("prove_bounds_freedom (&"), "{one_d}");

        let two_d = ok(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [transpose2d], evidence: "e.json",
            sizes: [(37, 19)],
        });
        assert!(two_d.contains("differential_dispatch_config"), "{two_d}");
        assert!(two_d.contains("prove_bounds_freedom_dispatch"), "{two_d}");
        // A dispatch suite never emits the cooperative branch (the two clauses
        // are mutually exclusive, and the branch would not type-check there).
        assert!(!two_d.contains("COOPERATIVE_CUBE_DIM"), "{two_d}");
        assert!(one_d.contains("COOPERATIVE_CUBE_DIM"), "{one_d}");
    }

    /// The rank-agreement assertion between the suite's `sizes:` unit and each
    /// kernel's `dispatch(...)` clause is emitted for every kernel, in both
    /// directions.
    #[test]
    fn the_dispatch_rank_check_is_emitted_per_kernel_in_both_directions() {
        let one_d = ok(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [axpy, fir3], evidence: "e.json",
        });
        assert_eq!(one_d.matches("DISPATCH_RANK . is_none ()").count(), 2);

        let two_d = ok(quote! {
            runtime: cubecl::wgpu::WgpuRuntime, kernels: [transpose2d, box_blur3x3],
            evidence: "e.json", sizes: [(37, 19)],
        });
        assert_eq!(two_d.matches("__r == 2u8").count(), 2, "{two_d}");
    }
}
