//! Cooperative (workgroup-shared-memory) twin derivation — the phase-split
//! reference model of docs/design-shared-memory.md §4, gated by the
//! `cooperative(cube_dim = N)` contract clause (§7.1).
//!
//! This is a *second* twin-derivation mode beside the ordinary
//! loop-over-`ABSOLUTE_POS` twin in `lib.rs` (§1: that model is meaningless
//! under shared memory). Instead of one flat sequential loop, a cooperative
//! kernel's body is split at `sync_cube()` barriers into **segments**, and the
//! twin runs, per cube, per segment, per `unit_pos`, that segment (§4.1):
//!
//! ```text
//! for cube in 0..cube_count:
//!     <fresh per-cube shared tiles (poison-initialised, §4.5)>
//!     for seg in phases:
//!         for unit_pos in 0..cube_dim: run segment seg
//!         // implicit barrier between segments
//! ```
//!
//! The cooperative *tree loop* `while half > 0 { …; sync_cube(); half /= c }`
//! (a loop that *contains* a barrier) is handled by thread-loop inversion
//! (§4.2): its cube-uniform control (`half`) is hoisted to a single scalar, the
//! before-barrier body becomes a per-`unit_pos` segment, and the after-barrier
//! uniform update runs once per level.
//!
//! ## Accepted subset and its agreement with the prover
//!
//! The recognised shapes are exactly the ones the race walker
//! (`crates/vericl-ir/src/prover.rs`) accepts (§4.3): `sync_cube()` at the top
//! level of the body, and inside the one uniform-trip-count halving loop; one
//! non-cooperative accumulation loop before the first barrier (the grid-stride
//! `while k < …` shape); a single `SharedMemory` tile; a single-writer guarded
//! global store. Where the two lanes cannot be identical the **twin is the more
//! restrictive one** (never the less) so the differential and proof lanes never
//! cover a kernel the other rejects: in particular a per-thread local that is
//! genuinely *stateful* and lives across a barrier is rejected here as
//! `OutOfSubset` (only pure *topology-alias* locals — `let tid = UNIT_POS as
//! usize`, recomputed per segment — may cross a barrier), and any non-uniform
//! `sync_cube()`/tree-loop update is rejected as barrier divergence, mirroring
//! the prover's own rejections rather than silently mis-modelling them.

use std::collections::HashSet;

use proc_macro2::{Ident, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Expr, ExprWhile, Local, Pat, Stmt, Type};

use crate::{GenEntry, NumKind, Param, ParamKind, build_gen_field, resolve_gen_entries};

/// The 1-D cooperative topology builtins that leave `BANNED_IDENTS` under the
/// `cooperative(...)` clause (§7.2). `ABSOLUTE_POS` is included because in a
/// cooperative twin it is bound per-segment (`CUBE_POS*cube_dim + unit_pos`),
/// not rewritten to the flat `ABSOLUTE_POS` loop variable of the ordinary twin.
pub(crate) const COOP_ALLOWED: &[&str] = &[
    "ABSOLUTE_POS",
    "UNIT_POS",
    "CUBE_POS",
    "CUBE_DIM",
    "CUBE_COUNT",
    "SharedMemory",
    "sync_cube",
    // v1.1: a workgroup-uniform `terminate!()` (the "skip the whole cube"
    // padding-guard pattern) leaves `BANNED_IDENTS` under `cooperative(...)`.
    // It stays banned for ordinary kernels (the round-1 finding: outside #[cube]
    // it expands to an empty block, so a sequential twin would fall through the
    // guard). The phase-split twin recognises `if <uniform> { terminate!() }` at
    // the top level before any barrier and models it as a cube-level skip
    // (`analyse` / `build_reference_body`); any other `terminate` position is
    // rejected (see `reject_stray_terminate`), so it never survives to the twin
    // body as a plain-Rust `terminate!()` (which would not compile).
    "terminate",
];

/// The cooperative topology constructs whose presence makes a kernel
/// "cooperative" — the clause is required iff at least one of these appears
/// (see the gate in `expand`).
const COOP_MARKERS: &[&str] =
    &["UNIT_POS", "CUBE_POS", "CUBE_DIM", "CUBE_COUNT", "sync_cube", "SharedMemory"];

/// The per-thread topology leaves. A value that (transitively) reads one of
/// these is *thread-varying*; anything provably built only from cube-uniform
/// leaves is uniform — exactly the split the barrier-uniformity check needs.
const THREAD_VARYING_BUILTINS: &[&str] = &["UNIT_POS", "ABSOLUTE_POS"];

/// Count the `sync_cube()` barrier calls in a kernel's source body (recursively,
/// including the one inside the tree loop). This is the twin's declared
/// **top-level barrier count** — every one is a phase boundary the phase-split
/// twin can see. The prover compares it against the `SyncCube` count in the
/// (helper-inlined) IR: a `uses(...)` helper that hid a barrier inflates the IR
/// count, and the mismatch is rejected (cooperative-composition soundness crux,
/// docs/design-shared-memory.md §7.4). Counting the SOURCE body (not the twin,
/// which drops them as segment delimiters) is what makes it the *twin-declared*
/// count: the helper's own body is not part of the kernel's source tokens, so a
/// helper barrier is never counted here — exactly the asymmetry the check exploits.
pub(crate) fn count_sync_cube(body: &syn::Block) -> usize {
    #[derive(Default)]
    struct SyncCubeCounter {
        count: usize,
    }
    impl<'ast> Visit<'ast> for SyncCubeCounter {
        fn visit_expr_call(&mut self, i: &'ast syn::ExprCall) {
            if let Expr::Path(p) = i.func.as_ref() {
                if p.path.segments.last().map(|s| s.ident == "sync_cube").unwrap_or(false) {
                    self.count += 1;
                }
            }
            syn::visit::visit_expr_call(self, i);
        }
    }
    let mut c = SyncCubeCounter::default();
    c.visit_block(body);
    c.count
}

/// Whether a kernel body uses any cooperative topology construct (so the
/// `cooperative(...)` clause is required, and vice versa).
pub(crate) fn kernel_uses_cooperative(body: &syn::Block) -> bool {
    let mut c = IdentRefCollector::default();
    c.visit_block(body);
    // `SharedMemory` shows up as a path segment, not an `ExprPath` single
    // ident, so scan tokens for it too.
    let toks = quote!(#body).to_string();
    COOP_MARKERS.iter().any(|m| c.names.contains(*m)) || toks.contains("SharedMemory")
}

// ---------------------------------------------------------------------------
// Small syn visitors.
// ---------------------------------------------------------------------------

/// Collects single-segment identifier *references* (`Expr::Path` of length 1).
#[derive(Default)]
struct IdentRefCollector {
    names: HashSet<String>,
}

impl<'ast> Visit<'ast> for IdentRefCollector {
    fn visit_expr_path(&mut self, i: &'ast syn::ExprPath) {
        if i.path.leading_colon.is_none() && i.path.segments.len() == 1 {
            self.names.insert(i.path.segments[0].ident.to_string());
        }
        syn::visit::visit_expr_path(self, i);
    }
}

/// Every identifier bound by a `let`/`for`/closure pattern anywhere in a set of
/// statements (reuses the same over-inclusive posture as `LocalCollector`).
#[derive(Default)]
struct BoundCollector {
    names: HashSet<String>,
}

impl<'ast> Visit<'ast> for BoundCollector {
    fn visit_pat_ident(&mut self, i: &'ast syn::PatIdent) {
        self.names.insert(i.ident.to_string());
        syn::visit::visit_pat_ident(self, i);
    }
}

/// Peel `Expr::Paren`/`Expr::Group` wrappers to reach the underlying
/// expression, mirroring the Paren/Group recursion in [`expr_is_pure_alias`].
/// Every place that classifies an expression by its *shape* must peel first,
/// or a parenthesis silently changes the classification: `(tile)[tid]` is the
/// same place-expression as `tile[tid]`, and `(tile[tid]) OP= …` the same
/// compound assignment as `tile[tid] OP= …`. `Expr::Group` is the invisible
/// delimiter syn/proc-macro insert around interpolated fragments — same
/// treatment for the same reason.
fn unwrap_paren_group(e: &Expr) -> &Expr {
    match e {
        Expr::Paren(pe) => unwrap_paren_group(&pe.expr),
        Expr::Group(g) => unwrap_paren_group(&g.expr),
        other => other,
    }
}

/// Detects a compound assignment (`x[..] += y`, etc.) whose target is an index
/// into a shared tile — the read-modify-write that would bypass poison
/// checking. syn 2.0 lowers `a += b` to `Expr::Binary` with a compound-assign
/// `BinOp` (same shape `WrappingFold` keys on).
///
/// The LHS is classified through [`unwrap_paren_group`] at *both* levels — the
/// whole target (`(tile[tid]) += …`) and the index base (`(tile)[tid] += …`) —
/// so a parenthesis cannot smuggle the read-modify-write past this ban (round-3
/// adversarial review F1: `(tile)[tid] += 1.0` evaded the pre-fix
/// `Expr::Index{expr: Expr::Path}`-only match, and the poison twin then read
/// the unwritten cell as `0.0`).
struct SharedCompoundAssignCheck<'a> {
    shared: &'a HashSet<String>,
    hit: Option<(String, proc_macro2::Span)>,
}

impl<'a, 'ast> Visit<'ast> for SharedCompoundAssignCheck<'a> {
    fn visit_expr_binary(&mut self, i: &'ast syn::ExprBinary) {
        use syn::BinOp::*;
        let is_compound = matches!(
            i.op,
            AddAssign(_)
                | SubAssign(_)
                | MulAssign(_)
                | DivAssign(_)
                | RemAssign(_)
                | BitXorAssign(_)
                | BitAndAssign(_)
                | BitOrAssign(_)
                | ShlAssign(_)
                | ShrAssign(_)
        );
        if is_compound {
            if let Expr::Index(idx) = unwrap_paren_group(i.left.as_ref()) {
                if let Expr::Path(p) = unwrap_paren_group(idx.expr.as_ref()) {
                    if let Some(id) = p.path.get_ident() {
                        if self.shared.contains(&id.to_string()) && self.hit.is_none() {
                            self.hit = Some((id.to_string(), i.span()));
                        }
                    }
                }
            }
        }
        syn::visit::visit_expr_binary(self, i);
    }
}

fn referenced_idents_stmts(stmts: &[Stmt]) -> HashSet<String> {
    let mut c = IdentRefCollector::default();
    for s in stmts {
        c.visit_stmt(s);
    }
    c.names
}

fn referenced_idents_expr(e: &Expr) -> HashSet<String> {
    let mut c = IdentRefCollector::default();
    c.visit_expr(e);
    c.names
}

fn bound_idents_stmts(stmts: &[Stmt]) -> HashSet<String> {
    let mut c = BoundCollector::default();
    for s in stmts {
        c.visit_stmt(s);
    }
    c.names
}

// ---------------------------------------------------------------------------
// Statement classification.
// ---------------------------------------------------------------------------

/// `let [mut] name = SharedMemory::<T>::new(N);` -> (name, elem type, length).
fn parse_shared_decl(local: &Local) -> Option<(Ident, Type, Expr)> {
    let Pat::Ident(pi) = &local.pat else { return None };
    let init = local.init.as_ref()?;
    let Expr::Call(call) = init.expr.as_ref() else { return None };
    let Expr::Path(p) = call.func.as_ref() else { return None };
    let segs = &p.path.segments;
    if segs.len() < 2 {
        return None;
    }
    let last = segs.last()?;
    if last.ident != "new" {
        return None;
    }
    let owner = &segs[segs.len() - 2];
    if owner.ident != "SharedMemory" {
        return None;
    }
    // element type from `SharedMemory::<T>`
    let syn::PathArguments::AngleBracketed(ab) = &owner.arguments else {
        return None;
    };
    let elem = ab.args.iter().find_map(|a| match a {
        syn::GenericArgument::Type(t) => Some(t.clone()),
        _ => None,
    })?;
    let len = call.args.first()?.clone();
    Some((pi.ident.clone(), elem, len))
}

/// Whether a statement is a `terminate!()` macro call (either
/// `Stmt::Macro` for a trailing `terminate!()` or `Stmt::Expr(Expr::Macro)`
/// for `terminate!();`).
fn is_terminate_call(stmt: &Stmt) -> bool {
    let path = match stmt {
        Stmt::Macro(m) => &m.mac.path,
        Stmt::Expr(Expr::Macro(m), _) => &m.mac.path,
        _ => return false,
    };
    path.segments.last().map(|s| s.ident == "terminate").unwrap_or(false)
}

/// Recognise the workgroup-uniform terminate pattern `if <cond> { terminate!() }`
/// (no `else`, then-branch exactly one `terminate!()` call), returning the
/// condition. This is the only accepted `terminate` shape (docs/design-shared-
/// memory.md §4.3/§7.4): a top-level, before-any-barrier, cube-uniform "skip the
/// whole cube" guard. Any other `terminate` position/shape is rejected by
/// `reject_stray_terminate`.
fn as_terminate_if(stmt: &Stmt) -> Option<&Expr> {
    let Stmt::Expr(Expr::If(if_expr), _) = stmt else { return None };
    if if_expr.else_branch.is_some() {
        return None;
    }
    let stmts = &if_expr.then_branch.stmts;
    if stmts.len() != 1 || !is_terminate_call(&stmts[0]) {
        return None;
    }
    Some(if_expr.cond.as_ref())
}

/// Whether a statement is exactly `sync_cube();`.
fn is_sync_cube(stmt: &Stmt) -> bool {
    let Stmt::Expr(Expr::Call(call), _) = stmt else { return false };
    if !call.args.is_empty() {
        return false;
    }
    let Expr::Path(p) = call.func.as_ref() else { return false };
    p.path.segments.last().map(|s| s.ident == "sync_cube").unwrap_or(false)
}

/// The `while` expression of a *cooperative* loop (one whose body contains a
/// top-level `sync_cube()`), if this statement is one.
fn as_cooperative_while(stmt: &Stmt) -> Option<&ExprWhile> {
    let w = match stmt {
        Stmt::Expr(Expr::While(w), _) => w,
        _ => return None,
    };
    if w.body.stmts.iter().any(is_sync_cube) {
        Some(w)
    } else {
        None
    }
}

/// Whether a statement's body (recursively) contains a `sync_cube()` — used to
/// reject a barrier nested somewhere the recogniser does not model.
fn stmt_contains_sync_cube(stmt: &Stmt) -> bool {
    let toks = quote!(#stmt).to_string();
    toks.contains("sync_cube")
}

// ---------------------------------------------------------------------------
// Purity / uniformity classification.
// ---------------------------------------------------------------------------

/// Whether an expression is a *pure topology alias* body: it may reference only
/// `known` identifiers (builtins, params, earlier aliases), numeric literals,
/// casts, unary/binary arithmetic and parentheses — never an array index, a
/// call, a method call, or a macro (those are stateful or unmodelled). A
/// `let name = <pure>` immutable binding can therefore be recomputed in every
/// segment (each thread's own copy — the promotion §4.1 prescribes, realised by
/// recomputation since the value is a pure function of the thread's builtins).
fn expr_is_pure_alias(e: &Expr, known: &HashSet<String>) -> bool {
    match e {
        Expr::Path(p) => {
            if p.path.leading_colon.is_some() || p.path.segments.len() != 1 {
                // multi-segment path (e.g. `f32::MAX`) — an associated const,
                // treated as pure.
                return true;
            }
            known.contains(&p.path.segments[0].ident.to_string())
        }
        Expr::Lit(_) => true,
        Expr::Paren(pe) => expr_is_pure_alias(&pe.expr, known),
        Expr::Group(g) => expr_is_pure_alias(&g.expr, known),
        Expr::Cast(c) => expr_is_pure_alias(&c.expr, known),
        Expr::Unary(u) => expr_is_pure_alias(&u.expr, known),
        Expr::Binary(b) => {
            expr_is_pure_alias(&b.left, known) && expr_is_pure_alias(&b.right, known)
        }
        _ => false,
    }
}

/// `let name = <pure alias expr>;` (immutable only) among `known` idents.
fn parse_alias_decl(local: &Local, known: &HashSet<String>) -> Option<(Ident, Expr)> {
    let Pat::Ident(pi) = &local.pat else { return None };
    if pi.mutability.is_some() {
        return None; // a mutable local is never a pure alias
    }
    let init = local.init.as_ref()?;
    if init.diverge.is_some() {
        return None;
    }
    if expr_is_pure_alias(&init.expr, known) {
        Some((pi.ident.clone(), init.expr.as_ref().clone()))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Phase items.
// ---------------------------------------------------------------------------

enum PhaseItem {
    /// A barrier-free run of statements, executed per `unit_pos`.
    Segment(Vec<Stmt>),
    /// A cooperative tree loop: cube-uniform control hoisted, the pre-barrier
    /// body per `unit_pos`, the post-barrier uniform update run once. Boxed so
    /// this variant does not dwarf `Segment` (clippy `large_enum_variant`).
    CoopLoop(Box<CoopLoop>),
}

struct CoopLoop {
    control_name: Ident,
    control_init: Expr,
    cond: Expr,
    before: Vec<Stmt>,
    after: Vec<Stmt>,
}

/// The full analysed cooperative body.
struct Analysis {
    shared: Vec<(Ident, Type, Expr)>,
    aliases: Vec<(Ident, Expr)>,
    /// Aliases that are cube-uniform (do not transitively read
    /// `UNIT_POS`/`ABSOLUTE_POS`). Recomputed at the cube level (before the
    /// per-`unit_pos` loops) when there are terminate guards, so a uniform alias
    /// a terminate condition references (`row = CUBE_POS`) is in scope there.
    uniform_alias_names: HashSet<String>,
    /// Workgroup-uniform `terminate!()` conditions (§4.3/§7.4): each is a
    /// top-level, before-any-barrier, cube-uniform "skip the whole cube" guard,
    /// emitted as `if <cond> { continue; }` at the top of the per-cube loop.
    terminates: Vec<Expr>,
    phases: Vec<PhaseItem>,
}

fn err<T: Spanned>(node: &T, msg: impl Into<String>) -> syn::Error {
    syn::Error::new(node.span(), msg.into())
}

/// Analyse a cooperative kernel body into shared tiles, replicated topology
/// aliases, and an ordered list of phase items — rejecting anything outside the
/// §4.3 subset with a targeted `OutOfSubset`-style error (§7 wording).
fn analyse(
    body: &syn::Block,
    params: &[Param],
    fn_name_str: &str,
) -> syn::Result<Analysis> {
    let builtins: HashSet<String> =
        COOP_ALLOWED.iter().map(|s| s.to_string()).collect();
    let param_names: HashSet<String> = params.iter().map(|p| p.name.to_string()).collect();

    let mut shared: Vec<(Ident, Type, Expr)> = Vec::new();
    let mut shared_names: HashSet<String> = HashSet::new();
    let mut aliases: Vec<(Ident, Expr)> = Vec::new();
    let mut alias_names: HashSet<String> = HashSet::new();
    let mut thread_varying_aliases: HashSet<String> = HashSet::new();
    let mut terminates: Vec<Expr> = Vec::new();
    let mut cleaned: Vec<Stmt> = Vec::new();
    // Whether any shared-tile declaration or segment/barrier statement has been
    // seen. A `terminate!()` guard must precede all of them (top level, before
    // any barrier — §4.3/§7.4), so a terminate after content is rejected.
    let mut saw_content = false;

    // Pass 1: pull out shared-tile declarations, pure topology aliases, and the
    // leading workgroup-uniform `terminate!()` guards.
    for stmt in &body.stmts {
        if let Stmt::Local(local) = stmt {
            if let Some((name, elem, len)) = parse_shared_decl(local) {
                if !shared.is_empty() {
                    return Err(err(
                        local,
                        "more than one `SharedMemory` tile in a cooperative kernel is outside \
                         the vericl v1 subset (single tile per kernel — docs/design-shared-\
                         memory.md §7.3); relax later",
                    ));
                }
                shared_names.insert(name.to_string());
                shared.push((name, elem, len));
                saw_content = true;
                continue;
            }
            let mut known = builtins.clone();
            known.extend(param_names.iter().cloned());
            known.extend(alias_names.iter().cloned());
            if let Some((name, expr)) = parse_alias_decl(local, &known) {
                // classify thread-varying (depends on UNIT_POS/ABSOLUTE_POS or
                // an earlier thread-varying alias).
                let refs = referenced_idents_expr(&expr);
                let varying = refs.iter().any(|r| {
                    THREAD_VARYING_BUILTINS.contains(&r.as_str())
                        || thread_varying_aliases.contains(r)
                });
                if varying {
                    thread_varying_aliases.insert(name.to_string());
                }
                alias_names.insert(name.to_string());
                aliases.push((name, expr));
                continue;
            }
        }
        // Workgroup-uniform `terminate!()` (v1.1): `if <uniform> { terminate!() }`
        // at the top level, before any barrier — modeled as a cube-level skip.
        if let Some(cond) = as_terminate_if(stmt) {
            if saw_content {
                return Err(err(
                    stmt,
                    format!(
                        "kernel `{fn_name_str}`: `terminate!()` must appear at the top level \
                         before any barrier or shared-memory access (the workgroup-uniform \
                         \"skip the whole cube\" guard); a post-barrier or mid-body terminate is \
                         outside the vericl v1.1 subset (docs/design-shared-memory.md §4.3/§7.4)"
                    ),
                ));
            }
            // Uniformity (§5.4): a thread-varying terminate condition is barrier
            // divergence (some threads skip, others reach the barrier) — rejected,
            // exactly like a thread-varying barrier guard.
            let refs = referenced_idents_expr(cond);
            let varying = refs.iter().any(|r| {
                THREAD_VARYING_BUILTINS.contains(&r.as_str())
                    || thread_varying_aliases.contains(r)
                    || shared_names.contains(r)
            });
            if varying {
                return Err(err(
                    cond,
                    format!(
                        "kernel `{fn_name_str}`: `terminate!()` condition is thread-varying \
                         (it reads UNIT_POS/ABSOLUTE_POS or a per-thread value) — a non-uniform \
                         terminate is barrier divergence (some threads skip the cube, others \
                         reach the barrier) and is outside the vericl v1.1 subset (only a \
                         workgroup-uniform terminate is accepted — docs/design-shared-memory.md \
                         §4.3/§7.4)"
                    ),
                ));
            }
            terminates.push(cond.clone());
            continue;
        }
        saw_content = true;
        cleaned.push(stmt.clone());
    }

    // A stray `terminate!()` anywhere that was NOT recognised as the accepted
    // top-level uniform guard (a bare `terminate!()`, one nested in a loop or a
    // non-uniform/`else`-bearing `if`, …) would survive into the twin body as a
    // plain-Rust `terminate!()` macro that does not exist — reject it with the
    // targeted shape error rather than emit an uncompilable twin.
    reject_stray_terminate(&cleaned, fn_name_str)?;

    let uniform_alias_names: HashSet<String> =
        alias_names.difference(&thread_varying_aliases).cloned().collect();

    // Compound-assignment into a shared tile (`tile[i] += x`) would go through
    // the poison model's `IndexMut` (which marks-written and hands back the
    // default), silently un-poisoning the read half of the read-modify-write —
    // a shared-memory definedness-masking hole (§4.5 / §9 risk 3). The
    // reduction shape never does this (it uses `tile[i] = a + b` pure writes);
    // reject it loudly rather than mis-model it.
    {
        let mut chk = SharedCompoundAssignCheck { shared: &shared_names, hit: None };
        chk.visit_block(body);
        if let Some((name, span)) = chk.hit {
            return Err(syn::Error::new(
                span,
                format!(
                    "kernel `{fn_name_str}`: compound assignment (`{name}[…] OP= …`) into a \
                     shared tile is outside the vericl v1 subset — it would bypass the twin's \
                     poison read-before-write check (docs/design-shared-memory.md §4.5); write \
                     `{name}[i] = <expr>` with an explicit read instead"
                ),
            ));
        }
    }

    // Pass 2: segment the cleaned statements at barriers and cooperative loops.
    let mut phases: Vec<PhaseItem> = Vec::new();
    let mut cur: Vec<Stmt> = Vec::new();
    let uniform_ctx = UniformCtx {
        shared_names: &shared_names,
        thread_varying_aliases: &thread_varying_aliases,
    };

    for stmt in &cleaned {
        if is_sync_cube(stmt) {
            if !cur.is_empty() {
                phases.push(PhaseItem::Segment(std::mem::take(&mut cur)));
            }
            continue;
        }
        if as_cooperative_while(stmt).is_some() {
            // The control declaration is the last statement accumulated so far.
            let control_decl = cur.pop();
            if !cur.is_empty() {
                phases.push(PhaseItem::Segment(std::mem::take(&mut cur)));
            }
            let coop = build_coop_loop(stmt, control_decl, &uniform_ctx, fn_name_str)?;
            phases.push(coop);
            continue;
        }
        // A barrier hidden somewhere the recogniser does not structurally
        // handle (e.g. inside an `if`, a `for`, or a non-cooperative loop) is
        // rejected rather than silently dropped (§4.3 / §7.3).
        if stmt_contains_sync_cube(stmt) {
            return Err(err(
                stmt,
                "sync_cube() in an unsupported position (only a top-level barrier or one inside \
                 the uniform `while <ctrl> > 0 { …; sync_cube(); … }` tree loop is accepted) is \
                 outside the vericl v1 subset (docs/design-shared-memory.md §4.3)",
            ));
        }
        cur.push(stmt.clone());
    }
    if !cur.is_empty() {
        phases.push(PhaseItem::Segment(cur));
    }

    // Pass 3: cross-phase soundness — a per-thread local declared in one phase
    // must not be read in another. Only aliases (recomputed), shared tiles
    // (per-cube), hoisted controls, builtins and params may cross a barrier.
    let body_locals: HashSet<String> = bound_idents_stmts(&cleaned);
    let control_all: HashSet<String> = phases
        .iter()
        .filter_map(|p| match p {
            PhaseItem::CoopLoop(cl) => Some(cl.control_name.to_string()),
            _ => None,
        })
        .collect();
    for phase in &phases {
        let (refs, bound_here): (HashSet<String>, HashSet<String>) = match phase {
            PhaseItem::Segment(stmts) => {
                (referenced_idents_stmts(stmts), bound_idents_stmts(stmts))
            }
            PhaseItem::CoopLoop(cl) => {
                let mut r = referenced_idents_stmts(&cl.before);
                r.extend(referenced_idents_stmts(&cl.after));
                r.extend(referenced_idents_expr(&cl.cond));
                let mut b = bound_idents_stmts(&cl.before);
                b.extend(bound_idents_stmts(&cl.after));
                (r, b)
            }
        };
        for r in &refs {
            let ok = !body_locals.contains(r)
                || bound_here.contains(r)
                || alias_names.contains(r)
                || shared_names.contains(r)
                || control_all.contains(r);
            if !ok {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!(
                        "kernel `{fn_name_str}`: per-thread local `{r}` is live across a barrier \
                         but is not a pure topology alias — promotion of stateful cross-barrier \
                         per-thread state is outside the vericl v1 twin subset (docs/design-\
                         shared-memory.md §4.1); confine it to a single segment or express it as \
                         `let {r} = <pure function of UNIT_POS/CUBE_*>`"
                    ),
                ));
            }
        }
    }

    Ok(Analysis { shared, aliases, uniform_alias_names, terminates, phases })
}

/// Reject any `terminate` token surviving in the segment statements — a
/// `terminate!()` that was NOT recognised as the accepted top-level uniform
/// guard (`if <uniform> { terminate!() }` before any barrier). Such a terminate
/// (bare, nested in a loop/`if`-with-`else`, or after content) would be emitted
/// verbatim into the twin, which does not compile (`terminate!()` is a cubecl
/// macro with no host definition). Rejecting here gives the targeted shape error
/// instead. Token scan (recursive via `quote`) — cheap and catches every nesting.
fn reject_stray_terminate(cleaned: &[Stmt], fn_name_str: &str) -> syn::Result<()> {
    for stmt in cleaned {
        if quote!(#stmt).to_string().contains("terminate") {
            return Err(err(
                stmt,
                format!(
                    "kernel `{fn_name_str}`: `terminate!()` is only accepted as a top-level \
                     `if <workgroup-uniform condition> {{ terminate!() }}` guard before any \
                     barrier (the \"skip the whole cube\" pattern); a bare, nested, `else`-bearing, \
                     or post-barrier terminate is outside the vericl v1.1 subset (docs/design-\
                     shared-memory.md §4.3/§7.4)"
                ),
            ));
        }
    }
    Ok(())
}

struct UniformCtx<'a> {
    shared_names: &'a HashSet<String>,
    thread_varying_aliases: &'a HashSet<String>,
}

impl UniformCtx<'_> {
    /// Whether a set of referenced idents is cube-uniform: it must touch no
    /// per-thread leaf (`UNIT_POS`/`ABSOLUTE_POS`), no thread-varying alias, and
    /// no shared tile (a shared read/write is per-thread data). A bare param
    /// array name (e.g. `data` in `data.len()`) is cube-uniform, so it is
    /// allowed; a *thread-varying index* into an array taints through the
    /// `UNIT_POS`/`ABSOLUTE_POS`/alias idents this already rejects.
    fn is_uniform(&self, refs: &HashSet<String>) -> bool {
        refs.iter().all(|r| {
            !THREAD_VARYING_BUILTINS.contains(&r.as_str())
                && !self.thread_varying_aliases.contains(r)
                && !self.shared_names.contains(r)
        })
    }
}

/// Recognise and split a cooperative tree loop (§4.2). Requires the control
/// declaration `let mut <ctrl> = <uniform init>` immediately preceding it, a
/// single top-level `sync_cube()` in the body, and a cube-uniform guard and
/// post-barrier update (else barrier divergence, §7.3).
fn build_coop_loop(
    stmt: &Stmt,
    control_decl: Option<Stmt>,
    ctx: &UniformCtx,
    fn_name_str: &str,
) -> syn::Result<PhaseItem> {
    let w = as_cooperative_while(stmt).expect("caller checked");

    // Split the loop body at its single top-level barrier.
    let barrier_positions: Vec<usize> =
        w.body.stmts.iter().enumerate().filter(|(_, s)| is_sync_cube(s)).map(|(i, _)| i).collect();
    if barrier_positions.len() != 1 {
        return Err(err(
            stmt,
            format!(
                "kernel `{fn_name_str}`: a cooperative tree loop must contain exactly one \
                 top-level `sync_cube()` (found {}); more than one barrier per tree level is \
                 outside the vericl v1 subset (docs/design-shared-memory.md §4.2)",
                barrier_positions.len()
            ),
        ));
    }
    let bpos = barrier_positions[0];
    let before: Vec<Stmt> = w.body.stmts[..bpos].to_vec();
    let after: Vec<Stmt> = w.body.stmts[bpos + 1..].to_vec();

    // A nested barrier inside the pre/post body (not at top level) is unmodelled.
    if before.iter().chain(after.iter()).any(stmt_contains_sync_cube) {
        return Err(err(
            stmt,
            format!(
                "kernel `{fn_name_str}`: a `sync_cube()` nested below the top level of a tree \
                 loop is outside the vericl v1 subset (docs/design-shared-memory.md §4.3)"
            ),
        ));
    }

    // The control declaration.
    let Some(Stmt::Local(local)) = control_decl.as_ref() else {
        return Err(err(
            stmt,
            format!(
                "kernel `{fn_name_str}`: a cooperative tree loop must be immediately preceded by \
                 its control declaration `let mut <ctrl> = <cube-uniform init>;` (the tree level, \
                 e.g. `let mut half = CUBE_DIM / 2;`) — outside the vericl v1 subset"
            ),
        ));
    };
    let Pat::Ident(pi) = &local.pat else {
        return Err(err(local, "cooperative tree-loop control must be a plain `let mut <name>`"));
    };
    let control_name = pi.ident.clone();
    let Some(init) = local.init.as_ref() else {
        return Err(err(local, "cooperative tree-loop control must have an initialiser"));
    };
    let control_init = init.expr.as_ref().clone();

    // Uniformity of control init, guard, and the post-barrier update (§5.4 /
    // §7.3): a thread-varying trip count or a per-thread `after` would make the
    // hoist-and-run-once model unsound (barrier divergence).
    if !ctx.is_uniform(&referenced_idents_expr(&control_init)) {
        return Err(err(
            local,
            format!(
                "kernel `{fn_name_str}`: cooperative tree-loop control `{control_name}`'s initial \
                 value is thread-varying (barrier divergence) — outside the vericl v1 subset \
                 (docs/design-shared-memory.md §7.3)"
            ),
        ));
    }
    if !ctx.is_uniform(&referenced_idents_expr(&w.cond)) {
        return Err(err(
            &w.cond,
            format!(
                "kernel `{fn_name_str}`: cooperative tree-loop guard is thread-varying (a \
                 sync_cube() inside a loop with a thread-varying trip count is barrier divergence) \
                 — outside the vericl v1 subset (docs/design-shared-memory.md §7.3)"
            ),
        ));
    }
    if !ctx.is_uniform(&referenced_idents_stmts(&after)) {
        return Err(err(
            stmt,
            format!(
                "kernel `{fn_name_str}`: the post-barrier update of a cooperative tree loop must \
                 be cube-uniform (it is run once per level in the twin); a thread-varying update \
                 is outside the vericl v1 subset (docs/design-shared-memory.md §4.2)"
            ),
        ));
    }
    // The control variable must actually be updated in `after` (a well-formed
    // downward tree recurrence, mirroring the prover's `verify_halving_update`).
    if !bound_or_assigned(&after, &control_name) {
        return Err(err(
            stmt,
            format!(
                "kernel `{fn_name_str}`: cooperative tree-loop control `{control_name}` is not \
                 updated after the barrier — the recognised tree recurrence is `{control_name} \
                 /= <constant>` (docs/design-shared-memory.md §4.2)"
            ),
        ));
    }

    Ok(PhaseItem::CoopLoop(Box::new(CoopLoop {
        control_name,
        control_init,
        cond: w.cond.as_ref().clone(),
        before,
        after,
    })))
}

/// Whether `name` is assigned (`name OP= …` / `name = …`) anywhere in `stmts`.
fn bound_or_assigned(stmts: &[Stmt], name: &Ident) -> bool {
    let target = name.to_string();
    let toks = quote!(#(#stmts)*).to_string();
    // Cheap structural check: the control name appears on an assignment LHS.
    // (`half /= 2` / `half = half / 2`.) A false positive only ever *accepts* a
    // loop the uniformity checks already vetted, never changes the emitted twin.
    toks.contains(&target)
}

// ---------------------------------------------------------------------------
// Reference body codegen.
// ---------------------------------------------------------------------------

/// Build the body of the cooperative `reference(...)` function (the phase-split
/// twin). `params` is the classified, generic-substituted parameter list.
pub(crate) fn build_reference_body(
    body: &syn::Block,
    params: &[Param],
    fn_name_str: &str,
) -> syn::Result<TokenStream2> {
    let analysis = analyse(body, params, fn_name_str)?;

    // Cube-scope shared tiles (poison-initialised, §4.5). Rewrite the length
    // expression too (it may read `CUBE_DIM`, though the reduction shape uses a
    // literal `256`).
    let shared_decls: Vec<TokenStream2> = analysis
        .shared
        .iter()
        .map(|(name, elem, len)| {
            let len = rewrite_coop_builtins(quote!(#len));
            quote! { let mut #name = ::vericl::SharedTile::<#elem>::new_poison(#len); }
        })
        .collect();

    // The replicated topology-alias declarations, injected at the top of every
    // per-`unit_pos` loop (each thread recomputes its own copy — §4.1).
    let alias_decls: Vec<TokenStream2> = analysis
        .aliases
        .iter()
        .map(|(name, expr)| {
            let expr = rewrite_coop_builtins(quote!(#expr));
            quote! { let #name = #expr; }
        })
        .collect();

    // Cube-level workgroup-uniform `terminate!()` guards (§4.3/§7.4): each is
    // emitted as `if <cond> { continue; }` at the top of the per-cube loop —
    // "skip the whole cube". Because the condition is cube-uniform, the whole
    // cube skips together, faithful to the GPU (all threads terminate, so the
    // cube produces no output). Only emitted when there are terminates; when
    // there are none, this and the cube-level uniform aliases below are empty, so
    // a kernel without `terminate!()` is byte-identical to before this feature.
    let terminate_guards: Vec<TokenStream2> = analysis
        .terminates
        .iter()
        .map(|cond| {
            let cond = rewrite_coop_builtins(quote!(#cond));
            quote! { if #cond { continue; } }
        })
        .collect();

    // The uniform aliases a terminate condition may reference (`row = CUBE_POS`)
    // must be in scope at the cube level (before the per-`unit_pos` loops), where
    // the guards run. They are cube-uniform, so recomputing them here (from
    // `CUBE_*`/params, no `UNIT_POS`) gives the same value every thread sees; the
    // per-`unit_pos` loops re-declare (shadow) them as usual. Empty when there
    // are no terminates, keeping non-terminate kernels unchanged.
    let cube_alias_decls: Vec<TokenStream2> = if analysis.terminates.is_empty() {
        Vec::new()
    } else {
        analysis
            .aliases
            .iter()
            .filter(|(name, _)| analysis.uniform_alias_names.contains(&name.to_string()))
            .map(|(name, expr)| {
                let expr = rewrite_coop_builtins(quote!(#expr));
                quote! { let #name = #expr; }
            })
            .collect()
    };

    // Per-`unit_pos` prelude: bind the compound `ABSOLUTE_POS` alias (a
    // non-colliding internal name — `UNIT_POS`/`CUBE_*` rewrite directly to the
    // loop variables/params, `ABSOLUTE_POS` to this), then the recomputed
    // topology aliases. NOTE: the topology builtins are *rewritten to internal
    // names* rather than bound as `let UNIT_POS = …`, because
    // `UNIT_POS`/`CUBE_DIM`/… are cubecl-prelude constants in scope (via the
    // generated module's `use super::*;`), so `let CUBE_DIM = x;` would be a
    // refutable const-*pattern* match, not a fresh binding (confirmed against
    // cubecl 0.10). See `rewrite_coop_builtins`.
    let per_thread_prelude = quote! {
        let __vericl_abs_pos = __vericl_cube * cube_dim + __vericl_unit_pos;
        #(#alias_decls)*
    };

    // Emit each phase, rewriting cooperative builtins in the segment bodies.
    let mut phase_toks: Vec<TokenStream2> = Vec::new();
    for phase in &analysis.phases {
        match phase {
            PhaseItem::Segment(stmts) => {
                let body = rewrite_coop_builtins(quote!(#(#stmts)*));
                phase_toks.push(quote! {
                    for __vericl_unit_pos in 0..cube_dim {
                        #per_thread_prelude
                        #body
                    }
                });
            }
            PhaseItem::CoopLoop(cl) => {
                let CoopLoop { control_name, control_init, cond, before, after } = cl.as_ref();
                let control_init = rewrite_coop_builtins(quote!(#control_init));
                let cond = rewrite_coop_builtins(quote!(#cond));
                let before_body = rewrite_coop_builtins(quote!(#(#before)*));
                let after_body = rewrite_coop_builtins(quote!(#(#after)*));
                phase_toks.push(quote! {
                    let mut #control_name = #control_init;
                    while #cond {
                        for __vericl_unit_pos in 0..cube_dim {
                            #per_thread_prelude
                            #before_body
                        }
                        #after_body
                    }
                });
            }
        }
    }

    Ok(quote! {
        for __vericl_cube in 0..cube_count {
            #(#cube_alias_decls)*
            #(#terminate_guards)*
            #(#shared_decls)*
            #(#phase_toks)*
        }
    })
}

/// Rewrite the cooperative topology builtins to the reference's in-scope
/// loop variables / parameters (a token-wise substitution, like the ordinary
/// twin's `ABSOLUTE_POS` rewrite):
///
/// | builtin        | replacement                    |
/// |----------------|--------------------------------|
/// | `UNIT_POS`     | `__vericl_unit_pos` (loop var)  |
/// | `ABSOLUTE_POS` | `__vericl_abs_pos` (bound above)|
/// | `CUBE_POS`     | `__vericl_cube` (loop var)       |
/// | `CUBE_DIM`     | `cube_dim` (param)               |
/// | `CUBE_COUNT`   | `cube_count` (param)             |
///
/// This is required (not merely tidy): those names are cubecl-prelude
/// constants in the generated module's scope, so binding same-named locals
/// would be const-pattern matches, not bindings.
fn rewrite_coop_builtins(ts: TokenStream2) -> TokenStream2 {
    use proc_macro2::{Group, TokenTree};
    let mut out = TokenStream2::new();
    for tt in ts {
        match tt {
            TokenTree::Ident(id) => {
                let repl = match id.to_string().as_str() {
                    "UNIT_POS" => Some("__vericl_unit_pos"),
                    "ABSOLUTE_POS" => Some("__vericl_abs_pos"),
                    "CUBE_POS" => Some("__vericl_cube"),
                    "CUBE_DIM" => Some("cube_dim"),
                    "CUBE_COUNT" => Some("cube_count"),
                    _ => None,
                };
                match repl {
                    Some(name) => out.extend(std::iter::once(TokenTree::Ident(Ident::new(
                        name,
                        id.span(),
                    )))),
                    None => out.extend(std::iter::once(TokenTree::Ident(id))),
                }
            }
            TokenTree::Group(g) => {
                let inner = rewrite_coop_builtins(g.stream());
                let mut ng = Group::new(g.delimiter(), inner);
                ng.set_span(g.span());
                out.extend(std::iter::once(TokenTree::Group(ng)));
            }
            other => out.extend(std::iter::once(other)),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Cooperative conformance case codegen.
// ---------------------------------------------------------------------------

/// Build the cooperative `generate_case` + `conformance_case` items. Unlike the
/// ordinary case (§7.1 divergence): `cube_count = ceil(n / cube_dim)`, output
/// (`&mut Array`) params are sized to `cube_count` (the per-cube partials) and
/// zero-initialised rather than drawn by `gen(...)`, and the launch uses the
/// pinned `(cube_count, cube_dim)` (asserting the passed `cube_dim` equals the
/// clause's pinned value — the single source of truth, §9 risk 5).
pub(crate) fn build_conformance_items(
    params: &[Param],
    gen_entries: &[GenEntry],
    fn_name: &Ident,
    fn_name_str: &str,
    cube_dim_expr: &Expr,
    comptime_values: &std::collections::HashMap<String, TokenStream2>,
    generic_types: &[Type],
) -> syn::Result<TokenStream2> {
    // v1.1: #[comptime] params are supported. They are NOT drawn by `gen(...)`
    // and never appear in the `reference`/`check_assumes`/`generate_case` call
    // sites (the twin bakes them as `let` consts — cube-uniform, so the same
    // value in every segment); their pinned value is spliced straight into the
    // `launch` args at their declared position, exactly as the ordinary
    // (non-cooperative) `build_conformance_items` does (see `launch_args` below).
    let (ranges, lens) = resolve_gen_entries(params, gen_entries, fn_name_str)?;

    // Draw statements: input arrays + scalars via the ordinary gen machinery
    // (sized `n`); output arrays sized `cube_count`, zero-initialised.
    let mut draw_stmts: Vec<TokenStream2> = Vec::new();
    let mut field_names: Vec<Ident> = Vec::new();
    let mut owned_tys: Vec<TokenStream2> = Vec::new();
    let mut check_args: Vec<TokenStream2> = Vec::new();

    for p in params {
        let name = &p.name;
        match &p.kind {
            ParamKind::Scalar(ty) => {
                field_names.push(name.clone());
                let field = build_gen_field(p, &ranges, &lens, fn_name_str, None)?;
                let stmt = &field.stmt;
                draw_stmts.push(quote! { #stmt });
                owned_tys.push(quote!(#ty));
                check_args.push(quote!(#name));
            }
            ParamKind::ArrayRef(elem) => {
                field_names.push(name.clone());
                let field = build_gen_field(p, &ranges, &lens, fn_name_str, None)?;
                let stmt = &field.stmt;
                draw_stmts.push(quote! { #stmt });
                owned_tys.push(quote!(::std::vec::Vec<#elem>));
                check_args.push(quote!(&#name));
            }
            ParamKind::ArrayMut(elem) => {
                field_names.push(name.clone());
                // Output partials: sized `cube_count`, zero-initialised (the
                // kernel fills them). NOT drawn by gen(...).
                draw_stmts.push(quote! {
                    let #name: ::std::vec::Vec<#elem> =
                        ::std::vec![::core::default::Default::default(); __vericl_cube_count];
                });
                owned_tys.push(quote!(::std::vec::Vec<#elem>));
                check_args.push(quote!(&#name));
            }
            // #[comptime] params are baked into the twin as `let` consts and
            // spliced into `launch` as their pinned value — never drawn, never a
            // `generate_case`/`check_assumes` argument (mirrors the ordinary
            // `build_conformance_items`).
            ParamKind::Comptime(_) => {}
        }
    }

    let generate_case_fn = quote! {
        /// Generate one cooperative differential case: input arrays sized `n`
        /// and scalars via `gen(...)`, output `&mut Array` partials sized
        /// `cube_count` (per-cube outputs, §7.1), zero-initialised. Resamples
        /// against `check_assumes` up to 64 times, then panics naming the
        /// kernel (the `gen(...)`/`assumes(...)` machinery is unchanged for the
        /// inputs — the only divergence is the output sizing).
        fn generate_case(n: usize, seed: u64, __vericl_cube_count: usize) -> ( #(#owned_tys,)* ) {
            let mut __vericl_rng = ::vericl::SplitMix64::new(seed);
            for _vericl_attempt in 0..64u32 {
                #(#draw_stmts)*
                if check_assumes(#(#check_args),*) {
                    return ( #(#field_names,)* );
                }
            }
            panic!(
                "kernel `{}`: gen(...) could not produce inputs satisfying assumes(...) after 64 \
                 resample attempts — the declared gen(...) ranges are inconsistent with this \
                 kernel's assumes(...) clauses",
                #fn_name_str,
            );
        }
    };

    // conformance_case: reference vs. GPU on the per-cube partials.
    let mut ref_clone_stmts: Vec<TokenStream2> = Vec::new();
    let mut reference_args: Vec<TokenStream2> = Vec::new();
    let mut gpu_upload_stmts: Vec<TokenStream2> = Vec::new();
    let mut gpu_readback_stmts: Vec<TokenStream2> = Vec::new();
    let mut compare_stmts: Vec<TokenStream2> = Vec::new();
    let mut launch_args: Vec<TokenStream2> = Vec::new();

    for p in params {
        let name = &p.name;
        match &p.kind {
            ParamKind::Comptime(_) => {
                // cubecl keeps a comptime param in its declared launch position
                // with its plain (unwrapped) type, so splice the pinned value in
                // there directly — no reference arg (the twin bakes it as a const).
                launch_args.push(comptime_values[&name.to_string()].clone());
            }
            ParamKind::Scalar(_) => {
                reference_args.push(quote!(#name));
                launch_args.push(quote!(#name));
            }
            ParamKind::ArrayRef(elem) => {
                let handle = format_ident!("__vericl_{}_handle", name);
                reference_args.push(quote!(&#name));
                gpu_upload_stmts.push(quote! {
                    let #handle = client.create_from_slice(
                        <#elem as ::cubecl::prelude::CubeElement>::as_bytes(&#name),
                    );
                });
                launch_args.push(quote! {
                    unsafe { ::cubecl::prelude::ArrayArg::from_raw_parts(#handle, #name.len()) }
                });
            }
            ParamKind::ArrayMut(elem) => {
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
                launch_args.push(quote! {
                    unsafe {
                        ::cubecl::prelude::ArrayArg::from_raw_parts(#handle.clone(), #name.len())
                    }
                });
                gpu_readback_stmts.push(quote! {
                    let #gpu_name: ::std::vec::Vec<#elem> =
                        <#elem as ::cubecl::prelude::CubeElement>::from_bytes(
                            &client.read_one(#handle).unwrap(),
                        )
                        .to_vec();
                });
                let compare_call = match NumKind::of(elem) {
                    Some(NumKind::F32) => {
                        quote!(::vericl::compare_f32_with(contract().compare, &#ref_name, &#gpu_name))
                    }
                    Some(NumKind::F64) => {
                        quote!(::vericl::compare_f64_with(contract().compare, &#ref_name, &#gpu_name))
                    }
                    Some(NumKind::U32) => {
                        quote!(::vericl::compare_u32_with(contract().compare, &#ref_name, &#gpu_name))
                    }
                    _ => {
                        return Err(syn::Error::new(
                            elem.span(),
                            format!(
                                "cooperative conformance_case v0 only compares f32, f64, or u32 \
                                 `&mut Array` partials; `{name}: &mut Array<{}>` is outside that \
                                 set",
                                quote!(#elem)
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

    let launch_turbofish = if generic_types.is_empty() {
        quote!(<R>)
    } else {
        quote!(<#(#generic_types,)* R>)
    };

    let conformance_case_fn = quote! {
        /// Run one cooperative differential case (docs/design-shared-memory.md
        /// §7.1): dispatch `(cube_count, cube_dim)` with `cube_count = ceil(n /
        /// cube_dim)`, size each `&mut Array` output to `cube_count`, run the
        /// phase-split reference, and compare per-cube partials. The passed
        /// `cube_dim` MUST equal the `cooperative(cube_dim = …)` pinned value
        /// (the single source of truth binding `CUBE_DIM` — §9 risk 5); a
        /// mismatch is a harness bug and panics.
        pub fn conformance_case<R: ::cubecl::prelude::Runtime>(
            client: &::cubecl::prelude::ComputeClient<R>,
            n: usize,
            seed: u64,
            cube_dim: u32,
        ) -> ::vericl::CaseOutcome {
            let __vericl_pinned: u32 = #cube_dim_expr;
            assert_eq!(
                cube_dim, __vericl_pinned,
                "cooperative kernel `{}` is pinned to cube_dim = {} by its cooperative(...) \
                 clause, but the suite launched it with cube_dim = {} — binding CUBE_DIM to a \
                 block size the launch does not use is unsound (docs/design-shared-memory.md §9 \
                 risk 5)",
                #fn_name_str, __vericl_pinned, cube_dim,
            );

            let __vericl_count = (n as u32).div_ceil(cube_dim).max(1);
            let __vericl_cube_count = __vericl_count as usize;

            let ( #(#field_names,)* ) = generate_case(n, seed, __vericl_cube_count);
            #(#ref_clone_stmts)*

            let __vericl_ref_outcome = ::vericl::catch_reference_panic(|| {
                reference(#(#reference_args,)* __vericl_cube_count, cube_dim as usize);
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
                        ::cubecl::prelude::CubeCount::Static(__vericl_count, 1, 1),
                        ::cubecl::prelude::CubeDim::new_1d(cube_dim),
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

    /// Run `analyse` over a minimal cooperative body and assert it is rejected
    /// with the shared-tile compound-assignment poison-ban wording (§4.5).
    fn assert_compound_assign_rejected(src: &str) {
        let block: syn::Block = syn::parse_str(src).expect("valid block");
        let Err(err) = analyse(&block, &[], "paren_demo") else {
            panic!("compound-assign into a shared tile must be rejected: {src}");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("compound assignment") && msg.contains("poison"),
            "expected the poison-ban wording, got: {msg}"
        );
    }

    /// The bare `tile[i] += …` read-modify-write into a shared tile is banned
    /// (positive control for the check itself — pre-existing behaviour).
    #[test]
    fn bare_compound_assign_into_shared_tile_is_rejected() {
        assert_compound_assign_rejected(
            "{ let mut tile = SharedMemory::<f32>::new(256usize); \
               tile[0usize] += 1.0f32; sync_cube(); }",
        );
    }

    /// Round-3 adversarial review F1, the reviewer's exact evasion: a
    /// *parenthesised* index base — `(tile)[tid] += …` — must be rejected with
    /// the identical poison-ban wording. Pre-fix it slipped past the
    /// `Expr::Index{expr: Expr::Path}`-only match and the poison twin then read
    /// the never-written cell as `0.0` (green-but-UB on a non-zeroing backend).
    #[test]
    fn paren_compound_assign_into_shared_tile_is_rejected() {
        assert_compound_assign_rejected(
            "{ let mut tile = SharedMemory::<f32>::new(256usize); \
               (tile)[0usize] += 1.0f32; sync_cube(); }",
        );
    }

    /// The same evasion nested deeper (`((tile))[i] += …`) and with the paren
    /// around the whole target (`(tile[i]) += …`) — both peeled at both levels.
    #[test]
    fn nested_paren_compound_assign_into_shared_tile_is_rejected() {
        assert_compound_assign_rejected(
            "{ let mut tile = SharedMemory::<f32>::new(256usize); \
               ((tile))[0usize] += 1.0f32; sync_cube(); }",
        );
        assert_compound_assign_rejected(
            "{ let mut tile = SharedMemory::<f32>::new(256usize); \
               (tile[0usize]) += 1.0f32; sync_cube(); }",
        );
    }

    // -----------------------------------------------------------------
    // Workgroup-uniform `terminate!()` (v1.1, docs/design-shared-memory.md
    // §4.3/§7.4) — twin-lane recognition and rejection.
    // -----------------------------------------------------------------

    fn analyse_terminate(src: &str) -> syn::Result<Analysis> {
        let block: syn::Block = syn::parse_str(src).expect("valid block");
        analyse(&block, &[], "term_demo")
    }

    /// A top-level, before-any-barrier, cube-uniform `if CUBE_POS >= 4 {
    /// terminate!() }` is accepted and recorded as one terminate guard.
    #[test]
    fn uniform_top_level_terminate_is_accepted() {
        let a = analyse_terminate(
            "{ if CUBE_POS >= 4usize { terminate!() } \
               let mut tile = SharedMemory::<f32>::new(256usize); \
               tile[0usize] = 1.0f32; sync_cube(); }",
        )
        .expect("a uniform top-level terminate must be accepted");
        assert_eq!(a.terminates.len(), 1, "expected exactly one terminate guard");
    }

    /// A **thread-varying** terminate condition (`if UNIT_POS < 128 {
    /// terminate!() }`) is barrier divergence — rejected. Matches the prover.
    #[test]
    fn thread_varying_terminate_is_rejected() {
        let Err(err) = analyse_terminate(
            "{ if UNIT_POS < 128u32 { terminate!() } \
               let mut tile = SharedMemory::<f32>::new(256usize); sync_cube(); }",
        ) else {
            panic!("a thread-varying terminate must be rejected");
        };
        assert!(
            err.to_string().contains("thread-varying"),
            "expected the thread-varying rejection, got: {err}"
        );
    }

    /// A `terminate!()` after content (here a shared-tile declaration) is not the
    /// "skip the whole cube" guard — rejected as it must precede any barrier.
    #[test]
    fn post_content_terminate_is_rejected() {
        let Err(err) = analyse_terminate(
            "{ let mut tile = SharedMemory::<f32>::new(256usize); \
               if CUBE_POS >= 4usize { terminate!() } sync_cube(); }",
        ) else {
            panic!("a post-content terminate must be rejected");
        };
        assert!(
            err.to_string().contains("before any barrier"),
            "expected the before-any-barrier rejection, got: {err}"
        );
    }

    /// A **bare** `terminate!()` (not the recognised `if <uniform> { terminate!()
    /// }` guard) would survive into the twin as an uncompilable macro — rejected
    /// by `reject_stray_terminate` with the targeted shape error.
    #[test]
    fn bare_terminate_is_rejected() {
        let Err(err) = analyse_terminate(
            "{ terminate!(); let mut tile = SharedMemory::<f32>::new(256usize); sync_cube(); }",
        ) else {
            panic!("a bare terminate must be rejected");
        };
        assert!(
            err.to_string().contains("only accepted as a top-level"),
            "expected the stray-terminate rejection, got: {err}"
        );
    }
}
