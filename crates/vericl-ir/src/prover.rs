//! SMT-checked out-of-bounds freedom over the CubeCL IR.
//!
//! Recursive walker over `Scope.instructions`, encoding a supported subset
//! of the IR into QF_LIA and discharging one obligation per
//! `Index`/`IndexAssign` (and `Unchecked*` variant) via subprocess z3
//! (docs/ir-research.md §4). See `ProveResult::OutOfSubset` sites for the
//! exact supported subset; anything else is rejected explicitly rather than
//! silently approximated, per the vericl claim model (README "Claims and
//! trust boundaries").
//!
//! ## Soundness notes (read before touching the walker)
//!
//! - Values are modeled as *terms*, not fresh symbols: `value_of` builds a
//!   substituted expression tree rather than declaring an SMT constant per
//!   IR variable. Only genuine leaves get a declared constant: `AbsolutePos`,
//!   integer `GlobalScalar`s, per-buffer `Length`s, and `RangeLoop` induction
//!   variables (which range over a set and cannot be a deterministic
//!   function of anything else).
//! - Unsupported operations are not immediately fatal: an instruction whose
//!   `out` we cannot model (float arithmetic, `Bitwise`, `Atomic`, ...) is
//!   left unbound ("tainted") rather than aborting the whole kernel. This
//!   matters in practice: `xorshift_step` and `mix_u32` compute their output
//!   *value* with bitwise/wrapping-integer ops that never feed an index
//!   expression (every index is a bare `ABSOLUTE_POS`), so they stay fully
//!   provable even though those ops are outside the modeled subset. If a
//!   tainted value is later needed for an obligation or a branch/loop
//!   condition, resolution fails there with an explicit `OutOfSubset` at
//!   that use site — unsupported constructs are never silently dropped from
//!   a position that would affect the proof, only from positions that
//!   provably can't (array contents, which this checker never reasons
//!   about).
//! - **Branch-scoped write taint (If/IfElse):** `self.smt.push()/pop()`
//!   scopes *path conditions* around an `If`/`IfElse` arm, but `self.memo`
//!   (the `VariableKind` -> symbolic-value map) is a completely separate
//!   piece of state with no SMT-level equivalent — a naive walk that just
//!   mutates it in place while walking an arm would let a variable
//!   reassignment made *inside* one arm leak into the other arm, and into
//!   code after the branch closes, unconditionally (REGRESSION, adversarial
//!   review round 2: confirmed false `Proved` on a real OOB write — a
//!   variable clamped to a safe value only on a near-impossible path made an
//!   unrelated, unguarded, genuinely-unbounded use of that same variable
//!   look safe). `process_branch` fixes this with snapshot/restore +
//!   write-taint: before walking an arm, `self.memo` is fully cloned
//!   (`SExpr` is `Copy` and the map is small, so this is cheap); the arm is
//!   walked against that snapshot; for `IfElse`, the snapshot is restored
//!   *before* walking the else arm (so the if-arm's writes are invisible to
//!   it); after the construct, the snapshot is restored once more and then
//!   every `VariableKind` written *anywhere* in either arm (both arms, for
//!   `IfElse`; the one arm, for `If`) is set back to tainted (`None`) —
//!   deliberately conservative, per the same taint discipline as everywhere
//!   else in this file: v0 does not attempt if/else value merging (a
//!   variable set to the *same* value in both arms still taints), and a
//!   later use that actually needs the value fails explicitly at that site
//!   as `OutOfSubset` rather than silently, or worse, `Proved`ing on a
//!   leaked value. "Written anywhere in either arm" is tracked by
//!   `write_log_stack`, a stack of `VariableKind` sets with one frame per
//!   currently-open arm, pushed before and popped after that arm's walk;
//!   every genuine variable write (`bind_out`/`taint_out`, and the loop-carry
//!   pre/post taint below — anything that goes through the shared `set_var`
//!   helper) records into whatever frame is currently on *top* of the
//!   stack, never `value_of`'s read-only resolution caching. This composes
//!   correctly for nested branches with no special-casing: an inner
//!   `If`/`IfElse` finishes (and pops its own frame) strictly before its
//!   enclosing arm's walk completes, and its own merge step re-applies its
//!   taints via the same `set_var` helper — which, by then, logs into
//!   whatever frame is now on top (the *enclosing* arm's), so a write two
//!   levels deep still reaches the outermost merge without needing every
//!   frame to observe every write directly. Obligations checked *inside* an
//!   arm are unaffected by any of this — they still resolve against
//!   whatever `self.memo` holds live at that point in the walk, under that
//!   arm's own pushed path condition, exactly as before.
//! - `Branch::RangeLoop` is modeled as "fresh var `i` with `start <= i (<)= end`,
//!   walk the body once" (no unrolling) per the architecture doc. This is
//!   sound for per-iteration obligations (proving in-bounds for an arbitrary
//!   `i` in range covers every concrete iteration) but would be *unsound*
//!   for a loop-carried accumulator whose index expressions depend on values
//!   threaded across iterations, since a single symbolic pass does not
//!   represent the accumulated value at iteration `k`. **Loop-carry
//!   refinement:** rather than rejecting the whole loop, `process_range_loop`
//!   statically finds every variable the loop body (recursively, through
//!   nested branches) reassigns that was already bound outside the loop
//!   (`scope_reassigned_vars`) and taints exactly those — via the ordinary
//!   `memo`/taint machinery, same as any other unsupported construct — for
//!   the duration of the loop body walk (pushed onto `carried_stack`, which
//!   `bind_out`/`taint_out` consult so *every* write to a carried variable
//!   inside the loop stays tainted, not just its first) and, defensively,
//!   again immediately after the loop returns. This is deliberately
//!   conservative: a carried variable is never un-tainted mid-loop even by a
//!   write whose own expression doesn't depend on the carried value (e.g.
//!   `idx = i * 2`), because such a binding would only be valid for uses
//!   within that same single symbolic body-walk, and nothing tracks that
//!   scoping precisely enough to bound its reuse. Two things follow: (1) a
//!   read of a carried variable *before* its own first write in program
//!   order (relative to loop entry) correctly resolves to tainted rather
//!   than the pre-loop value, since the pre-taint runs before the body walk
//!   starts; (2) everything in the loop that doesn't touch carried state —
//!   including every other loop in the kernel — is still modeled exactly as
//!   before. Net effect: an accumulator kernel whose index/branch
//!   expressions never depend on the accumulator (e.g. a sum reduced into a
//!   local, then written to an index that's a plain function of
//!   `ABSOLUTE_POS`) now proves; one whose index *does* depend on carried
//!   state fails explicitly, as `OutOfSubset`, at that exact use site —
//!   never silently, never `Proved`.
//! - The ascending-bounds model above assumes unit stride. `RangeLoop.step`
//!   (`Some(_)` for `range_stepped`, e.g. a descending loop where
//!   `start > end` numerically) is never modeled: asserting `start <= i <
//!   end` for a genuinely descending range makes the SMT context infeasible,
//!   which would make every obligation inside the loop vacuously "provable"
//!   (UNSAT-under-contradiction, not UNSAT-because-safe). `process_range_loop`
//!   therefore rejects any loop with `step.is_some()` outright, before
//!   asserting bounds, rather than silently mismodeling it. This guard is
//!   independent of, and unaffected by, the loop-carry refinement above —
//!   it runs first, before any carried-variable analysis.
//! - **Boolean condition composition:** CubeCL 0.10 lowers `&&`/`||`/`!` to
//!   *eager* `Operator::And`/`Or`/`Not` (over already-evaluated bool
//!   sub-expressions, each its own preceding instruction) rather than to
//!   nested branches — confirmed empirically by extracting IR for guards
//!   shaped like `a && b`/`a || b`/`!a` (see docs/ir-research.md §3): both
//!   sides are always evaluated as ordinary `Comparison`/`Operator`
//!   instructions first, then combined by one more instruction, then fed as
//!   a single `Variable` to `Branch::If`/`IfElse`. This is exactly the shape
//!   `value_of`'s memoized-term model already handles: `And`/`Or`/`Not` are
//!   modeled as SMT `and`/`or`/`not` over their (recursively resolved)
//!   operands, so `if a && b` composes the same way `if a { if b { ... } }`
//!   already did — a tainted sub-condition taints the whole composed
//!   condition, resolution failing, explicitly, only at the branch that
//!   actually needs it (same discipline as everything else in this file).
//! - **Div/mod-derived indices:** `Arithmetic::Div`/`Arithmetic::Modulo` are
//!   modeled with SMT-LIB `div`/`mod` (Euclidean division), but only when an
//!   internal side-obligation — the divisor is nonzero and both operands are
//!   nonnegative, under the *live* path conditions + assumes — actually
//!   discharges (`Prover::try_discharge`, checked fresh for every div/mod
//!   site, not inferred from the operands' IR types: an intermediate
//!   expression like `a - b` over two `u32` leaves is modeled as plain
//!   integer subtraction and is not otherwise clamped nonnegative, so the
//!   nonnegativity half of the side-obligation is a real proof, not a
//!   type-driven assumption). Euclidean div/mod coincide with Rust's/WGSL's
//!   truncated-toward-zero semantics exactly when both operands are
//!   nonnegative, which is why that check is required rather than optional.
//!   If the side-obligation does not discharge (SAT, or an inconclusive
//!   `unknown`), the result is left tainted — never hard-errored, since the
//!   value may never feed an obligation — per the same taint discipline as
//!   everything else here. This side-obligation is deliberately *not*
//!   counted in `Prover::obligations` (which counts only the public
//!   `Index`/`IndexAssign` bounds obligations `ProveResult::Proved` reports):
//!   it's an internal precondition for soundly *modeling* div/mod, not a
//!   bounds check the caller asked for.
//! - **Cooperative mode (shared-memory milestone M1):** a second entry point
//!   `prove_bounds_freedom_cooperative(def, buffers, assumes, cube_dim)` opts
//!   the walk into *workgroup-cooperative* modeling by pinning `CUBE_DIM` to a
//!   concrete `cube_dim` constant (the `cooperative(cube_dim = …)` contract
//!   clause, docs/design-shared-memory.md §7.1). This flips on modeling for
//!   the 1-D topology builtins that the plain (`coop == None`) walk leaves
//!   tainted: `UnitPos` is a fresh `[0, cube_dim)` symbol (the per-thread
//!   leaf); `CubePos`/`CubeCount` are fresh nonnegative symbols (cube-uniform
//!   leaves); `CubeDim` is the concrete numeral `cube_dim`; and `AbsolutePos`
//!   is *recomputed* as `CubePos*cube_dim + UnitPos` (the 1-D identity)
//!   instead of the plain walk's opaque fresh leaf, so a `tile[UnitPos]`
//!   access and an `input[AbsolutePos]` access in the same kernel share one
//!   `UnitPos` symbol. All of these are memoized on `VariableKind`, so every
//!   occurrence of a given builtin resolves to the same symbol. **The three
//!   leaf symbols are pre-declared at the outermost SMT scope** (see
//!   `predeclare_coop_leaves`) rather than lazily on first use: SMT-LIB `pop`
//!   discards *declarations* made since the matching `push`, so a leaf first
//!   resolved inside a branch arm would have its `declare-const` and its range
//!   assertion (`0 <= unit_pos < cube_dim`, `cube_pos >= 0`) scoped to that
//!   arm and silently dropped for a later use after the branch closes.
//!   Pre-declaring keeps the leaves and their range facts in force for the
//!   whole walk. When `coop == None` every one of these builtins stays
//!   tainted, byte-for-byte as before this milestone (only `AbsolutePos` was,
//!   and still is, modeled — as a plain fresh leaf).
//! - **Shared arrays (M1):** `Index`/`IndexAssign` on a `VariableKind::
//!   SharedArray { id, length, .. }` list are modeled the same way as a global
//!   array access, except the bound is the **compile-time `length` carried in
//!   the `VariableKind`** (a `SharedMemory::<T>::new(N)` literal, §2.2), not a
//!   runtime `Length` symbol: the obligation is `0 <= index < length` against
//!   that concrete numeral. So `tile[UnitPos]` with `cube_dim <= length`
//!   discharges, and an undersized tile (`cube_dim > length`) is a genuine
//!   `Refuted`. A shared array is not a kernel parameter, so it carries no
//!   `BufferParam`; its name in obligations/counterexamples is
//!   `shared_array(id)`. This modeling is independent of `coop`: a shared
//!   access resolved with `coop == None` still checks the constant bound, but
//!   its index (`UnitPos`) is tainted there, so it fails as `OutOfSubset` at
//!   the index rather than proving — only cooperative mode makes `UnitPos`
//!   modeled enough to discharge.
//! - **`Branch::Loop` recognition (M2).** A `Branch::Loop` is CubeCL's
//!   desugaring of a `while`/`loop`, not the range-`for` that becomes
//!   `RangeLoop`. Two shapes are recognized, both keyed on the canonical
//!   `while cond { … }` desugaring — a **leading break-guard**: the first three
//!   body instructions are `c = <cond>`, `nc = Not c`, `if nc { break }`
//!   (§2.4), validated against the probe IR dumps. `recognize_break_guard`
//!   matches exactly that prefix (anything else — e.g. a `loop { body; if c {
//!   break } }` with a *trailing* break — is not recognized and stays the
//!   pre-existing `OutOfSubset`, so a bare unbounded loop is never modeled).
//!   * A loop whose body (recursively) contains a `SyncCube` is a
//!     **cooperative** loop (a barrier-carrying tree reduction). It cannot be
//!     modeled by a single-thread bounds pass without the two-thread race
//!     walker (deliverable B, milestone M3), so it is rejected `OutOfSubset`
//!     with a targeted "race walker not yet implemented (milestone M3)" reason
//!     — deferred, never silently mismodeled. This check runs *first*, so any
//!     barrier-carrying loop defers regardless of its guard shape.
//!   * A **non-cooperative** loop (no `SyncCube` inside) is modeled
//!     RangeLoop-style. The loop guard `c` is asserted as a path condition for
//!     the body (the body only runs while `c` holds), and the body is walked
//!     once symbolically. Carried variables (reused from `scope_reassigned_
//!     vars`, exactly as `process_range_loop`) split two ways: a carried,
//!     integer-typed variable that the guard comparison **upper-bounds** (the
//!     `v` in the ascending `while v < n` / `while n > v` shape) is the loop's
//!     *induction variable* — it gets a fresh symbol (nonnegative if unsigned)
//!     whose upper bound comes from the asserted guard, sound for the same
//!     reason `RangeLoop`'s `i` is (proving an obligation for an arbitrary
//!     in-range value covers every concrete iteration, and the fresh symbol
//!     *over*-approximates the actual arithmetic progression of induction
//!     values). A guard operand the guard does *not* upper-bound (a
//!     lower-bound `v > 0`, an `==`/`!=`) is **not** promoted — it stays
//!     tainted, so a descending or non-monotone loop resolves `OutOfSubset`
//!     rather than a fresh symbol bounded only from below manufacturing a
//!     spurious `Refuted` on a safe loop. Every *other* carried
//!     variable is an accumulator whose per-iteration value a single symbolic
//!     pass cannot represent, so it is **tainted**, identically to
//!     `process_range_loop`. As there, a write to any carried variable
//!     (induction included) inside the body re-taints it via `carried_stack`,
//!     so an induction value is fresh only for reads *before* its own in-body
//!     update (e.g. `data[k]` before `k += stride`); a read after the update
//!     resolves to taint, never a bogus post-update bound. If the guard itself
//!     depends on unmodeled state (does not resolve), the loop is rejected
//!     `OutOfSubset` rather than walked with an unconstrained induction symbol
//!     (which could manufacture a false `Refuted`).

use std::collections::{HashMap, HashSet};

use cubecl::ir::{
    Arithmetic, Branch, Builtin, Comparison, ConstantValue, Id, Instruction, Loop, Metadata,
    Operation, Operator, Scope, Synchronization, Type, Variable, VariableKind,
};
use cubecl::prelude::KernelDefinition;
use easy_smt::{Context, ContextBuilder, Response, SExpr};

/// One array parameter, in buffer-registration order (index == buffer id —
/// see `crates/vericl-macros`' generated `BUFFER_PARAMS`: buffer ids are
/// assigned by a single counter shared across inputs and outputs, in the
/// order each array parameter is registered while building the
/// `KernelDefinition`, so position in this slice doubles as the id).
#[derive(Debug, Clone, Copy)]
pub struct BufferParam<'a> {
    pub name: &'a str,
    pub is_output: bool,
}

/// A structured `assumes(...)` clause the macro recognized, in terms of
/// buffer parameter names. Mirrors (but does not depend on) the contract
/// layer's `vericl::StructuredAssume` — this crate has no dependency on
/// `vericl` core (see module docs), so the harness translates between the
/// two. Fewer/unrecognized assumes are sound (may cause `Refuted` or
/// `OutOfSubset` where a recognized one would have proved) since they only
/// ever narrow the search for a counterexample, never rule one out.
#[derive(Debug, Clone, Copy)]
pub enum Assume<'a> {
    LenEq { a: &'a str, b: &'a str },
    LenEqConst { a: &'a str, value: u64 },
}

#[derive(Debug, Clone)]
pub enum ProveResult {
    /// Every `Index`/`IndexAssign` obligation encountered was discharged
    /// UNSAT (i.e. no in-bounds violation is reachable).
    Proved { obligations: usize },
    /// One obligation was satisfiable — a counterexample exists.
    Refuted {
        obligation: String,
        counterexample: String,
    },
    /// The kernel (or a specific instruction) uses a construct outside the
    /// vericl v0 subset.
    OutOfSubset { reason: String },
    /// The solver process itself failed (spawn, I/O, or an `unknown`
    /// response).
    SolverError { detail: String },
}

/// `z3 --version`, or `None` if the `z3` binary isn't on `PATH`. Recorded in
/// evidence as part of the trusted solver component (docs/ir-research.md
/// §4: the subprocess solver is an external, independently versioned
/// trusted component, same posture as backend codegen).
pub fn z3_version() -> Option<String> {
    let out = std::process::Command::new("z3").arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok().map(|s| s.trim().to_string())
}

/// Prove out-of-bounds freedom for `def` over its supported IR subset.
///
/// `buffers` must be in buffer-registration order (see `BufferParam`);
/// `assumes` are the contract's recognized structured assumptions, used to
/// constrain buffer lengths before checking each obligation.
pub fn prove_bounds_freedom(
    def: &KernelDefinition,
    buffers: &[BufferParam],
    assumes: &[Assume],
) -> ProveResult {
    prove_bounds_freedom_impl(def, buffers, assumes, None)
}

/// Prove out-of-bounds freedom for a **workgroup-cooperative** `def`, pinning
/// `CUBE_DIM` to `cube_dim` (the `cooperative(cube_dim = …)` contract clause,
/// docs/design-shared-memory.md §7.1).
///
/// This is the shared-memory milestone entry point: relative to
/// `prove_bounds_freedom` it additionally models the 1-D topology builtins
/// (`UnitPos`/`CubePos`/`CubeDim`/`CubeCount`, with `AbsolutePos` recomputed as
/// `CubePos*cube_dim + UnitPos`) and accepts `SharedMemory` (`SharedArray`)
/// indexing bounded by the array's compile-time length (see the module docs'
/// "Cooperative mode" / "Shared arrays" bullets). `cube_dim` must be the block
/// size the kernel is actually launched with — binding it to a value the launch
/// does not use would be unsound (§9 risk 5), which the harness prevents by
/// sourcing both from the single `cooperative(...)` clause.
pub fn prove_bounds_freedom_cooperative(
    def: &KernelDefinition,
    buffers: &[BufferParam],
    assumes: &[Assume],
    cube_dim: u32,
) -> ProveResult {
    prove_bounds_freedom_impl(def, buffers, assumes, Some(cube_dim))
}

fn prove_bounds_freedom_impl(
    def: &KernelDefinition,
    buffers: &[BufferParam],
    assumes: &[Assume],
    coop: Option<u32>,
) -> ProveResult {
    let mut smt = match ContextBuilder::new().solver("z3").solver_args(["-smt2", "-in"]).build() {
        Ok(ctx) => ctx,
        Err(e) => {
            return ProveResult::SolverError {
                detail: format!("failed to start z3: {e}"),
            };
        }
    };

    let mut prover = Prover {
        smt: &mut smt,
        buffers,
        memo: HashMap::new(),
        buffer_len: HashMap::new(),
        declared: Vec::new(),
        fresh: 0,
        obligations: 0,
        carried_stack: Vec::new(),
        write_log_stack: Vec::new(),
        coop,
    };

    if let Err(e) = prover.assert_structured_assumes(assumes) {
        return e.into_result();
    }
    // Pre-declare the cooperative leaves (if any) at the outermost SMT scope,
    // before `process_scope` opens any branch push — see the module docs'
    // "Cooperative mode" bullet for why lazy declaration would be unsound.
    if let Err(e) = prover.predeclare_coop_leaves() {
        return e.into_result();
    }

    match prover.process_scope(&def.body) {
        Ok(()) => ProveResult::Proved {
            obligations: prover.obligations,
        },
        Err(stop) => stop.into_result(),
    }
}

enum Stop {
    OutOfSubset(String),
    Refuted { obligation: String, counterexample: String },
    SolverError(String),
}

impl Stop {
    fn into_result(self) -> ProveResult {
        match self {
            Stop::OutOfSubset(reason) => ProveResult::OutOfSubset { reason },
            Stop::Refuted { obligation, counterexample } => {
                ProveResult::Refuted { obligation, counterexample }
            }
            Stop::SolverError(detail) => ProveResult::SolverError { detail },
        }
    }
}

fn smt_err(e: std::io::Error) -> Stop {
    Stop::SolverError(format!("z3 I/O error: {e}"))
}

/// An indexable array operand: a global input/output buffer (bounded by a
/// runtime `Length`), or a `SharedMemory` tile (bounded by its compile-time
/// `length`). See the module docs' "Shared arrays" bullet.
enum ArrayRef {
    Global { id: Id },
    Shared { id: Id, length: usize },
}

struct Prover<'a, 'b> {
    smt: &'a mut Context,
    buffers: &'a [BufferParam<'b>],
    /// Memoized symbolic value per IR variable. `None` means "resolved, but
    /// to an unsupported/untracked value" (taint) — distinct from "not yet
    /// looked up", which is simply absent from the map.
    memo: HashMap<VariableKind, Option<SExpr>>,
    buffer_len: HashMap<Id, SExpr>,
    /// Every declared free constant, for rendering counterexamples.
    declared: Vec<(String, SExpr)>,
    fresh: u32,
    obligations: usize,
    /// Stack of "carried" variable-kind sets, one entry per currently-open
    /// `RangeLoop` whose body reassigns a variable bound outside it (see
    /// `process_range_loop` and the module docs' "Loop-carry refinement").
    /// Consulted by `bind_out`/`taint_out`: a write to a variable in *any*
    /// set on this stack is forced back to tainted regardless of what it
    /// would otherwise resolve to, for as long as the corresponding loop is
    /// being walked. Empty outside of (nested) carried loops, so this costs
    /// nothing for every kernel that doesn't have one.
    carried_stack: Vec<HashSet<VariableKind>>,
    /// Stack of "written variable" sets, one frame per currently-open
    /// `If`/`IfElse` arm being walked (see the module docs' "Branch-scoped
    /// write taint"). Every genuine variable write goes through `set_var`,
    /// which records `kind` into whichever frame is on *top* of this stack
    /// (if any) — `process_branch` pushes a fresh frame before walking an
    /// arm and pops it after, using the popped set to know exactly which
    /// variables to re-taint once the arm's private memo state is
    /// discarded. Empty outside of (nested) branches, so this costs
    /// nothing for a kernel with no `If`/`IfElse` at all.
    write_log_stack: Vec<HashSet<VariableKind>>,
    /// `Some(cube_dim)` in cooperative mode (the pinned `CUBE_DIM` constant),
    /// `None` for the plain single-thread bounds walk. Gates all the
    /// shared-memory-milestone leaf modeling (module docs' "Cooperative
    /// mode"); when `None`, `UnitPos`/`CubePos`/`CubeDim`/`CubeCount` stay
    /// tainted exactly as before this milestone.
    coop: Option<u32>,
}

impl<'a, 'b> Prover<'a, 'b> {
    fn buffer_name(&self, id: Id) -> String {
        self.buffers
            .get(id as usize)
            .map(|b| b.name.to_string())
            .unwrap_or_else(|| format!("<buffer {id}>"))
    }

    fn declare_int(&mut self, hint: &str, non_negative: bool) -> Result<SExpr, Stop> {
        self.fresh += 1;
        let name = format!("{hint}{}", self.fresh);
        let sort = self.smt.int_sort();
        let e = self.smt.declare_const(&name, sort).map_err(smt_err)?;
        if non_negative {
            let zero = self.smt.numeral(0);
            let ge0 = self.smt.gte(e, zero);
            self.smt.assert(ge0).map_err(smt_err)?;
        }
        self.declared.push((name, e));
        Ok(e)
    }

    fn length_of(&mut self, id: Id) -> Result<SExpr, Stop> {
        if let Some(e) = self.buffer_len.get(&id) {
            return Ok(*e);
        }
        let hint = format!("len_{}_", self.buffer_name(id));
        let e = self.declare_int(&hint, true)?;
        self.buffer_len.insert(id, e);
        Ok(e)
    }

    fn assert_structured_assumes(&mut self, assumes: &[Assume]) -> Result<(), Stop> {
        for assume in assumes {
            match *assume {
                Assume::LenEq { a, b } => {
                    let ida = self.buffer_id_by_name(a)?;
                    let idb = self.buffer_id_by_name(b)?;
                    let la = self.length_of(ida)?;
                    let lb = self.length_of(idb)?;
                    let eq = self.smt.eq(la, lb);
                    self.smt.assert(eq).map_err(smt_err)?;
                }
                Assume::LenEqConst { a, value } => {
                    let ida = self.buffer_id_by_name(a)?;
                    let la = self.length_of(ida)?;
                    let v = self.smt.numeral(value);
                    let eq = self.smt.eq(la, v);
                    self.smt.assert(eq).map_err(smt_err)?;
                }
            }
        }
        Ok(())
    }

    fn buffer_id_by_name(&self, name: &str) -> Result<Id, Stop> {
        self.buffers
            .iter()
            .position(|b| b.name == name)
            .map(|i| i as Id)
            .ok_or_else(|| {
                Stop::OutOfSubset(format!(
                    "structured assume refers to unknown buffer parameter `{name}`"
                ))
            })
    }

    // -- cooperative leaves (shared-memory milestone M1) ----------------

    /// Declare the cooperative leaf symbols at the outermost SMT scope so
    /// their `declare-const`s and range assertions outlive every branch
    /// push/pop (module docs' "Cooperative mode"). No-op when `coop` is
    /// `None`. Declaring an unused leaf (e.g. `CubeCount` in a kernel that
    /// never reads it) is harmless — a free nonnegative constant no
    /// obligation references.
    fn predeclare_coop_leaves(&mut self) -> Result<(), Stop> {
        if self.coop.is_none() {
            return Ok(());
        }
        self.unit_pos_sym()?;
        self.cube_pos_sym()?;
        self.cube_count_sym()?;
        Ok(())
    }

    /// The `UnitPos` leaf: a fresh symbol constrained to `[0, cube_dim)`.
    /// Memoized on `VariableKind::Builtin(UnitPos)` so `AbsolutePos`'s
    /// recomputation and every direct `tile[UnitPos]` share one symbol.
    fn unit_pos_sym(&mut self) -> Result<SExpr, Stop> {
        let kind = VariableKind::Builtin(Builtin::UnitPos);
        if let Some(Some(e)) = self.memo.get(&kind) {
            return Ok(*e);
        }
        let cube_dim = self.coop.expect("unit_pos_sym only reachable in cooperative mode");
        let sym = self.declare_int("unit_pos", true)?;
        let bound = self.smt.numeral(cube_dim as u64);
        let lt = self.smt.lt(sym, bound);
        self.smt.assert(lt).map_err(smt_err)?;
        self.memo.insert(kind, Some(sym));
        Ok(sym)
    }

    /// The `CubePos` leaf: a fresh nonnegative (cube-uniform) symbol.
    fn cube_pos_sym(&mut self) -> Result<SExpr, Stop> {
        let kind = VariableKind::Builtin(Builtin::CubePos);
        if let Some(Some(e)) = self.memo.get(&kind) {
            return Ok(*e);
        }
        let sym = self.declare_int("cube_pos", true)?;
        self.memo.insert(kind, Some(sym));
        Ok(sym)
    }

    /// The `CubeCount` leaf: a fresh nonnegative (cube-uniform) symbol.
    fn cube_count_sym(&mut self) -> Result<SExpr, Stop> {
        let kind = VariableKind::Builtin(Builtin::CubeCount);
        if let Some(Some(e)) = self.memo.get(&kind) {
            return Ok(*e);
        }
        let sym = self.declare_int("cube_count", true)?;
        self.memo.insert(kind, Some(sym));
        Ok(sym)
    }

    /// Resolve a topology builtin. In cooperative mode the 1-D leaves are
    /// modeled (module docs' "Cooperative mode"); otherwise only
    /// `AbsolutePos` is (a plain fresh leaf), everything else tainted —
    /// byte-for-byte the pre-milestone behavior.
    fn builtin_value(&mut self, b: Builtin) -> Option<SExpr> {
        let Some(cube_dim) = self.coop else {
            return match b {
                Builtin::AbsolutePos => self.declare_int("abs_pos", true).ok(),
                _ => None,
            };
        };
        match b {
            Builtin::UnitPos => self.unit_pos_sym().ok(),
            Builtin::CubePos => self.cube_pos_sym().ok(),
            Builtin::CubeCount => self.cube_count_sym().ok(),
            Builtin::CubeDim => Some(self.smt.numeral(cube_dim as u64)),
            // AbsolutePos = CubePos*cube_dim + UnitPos (the 1-D identity).
            Builtin::AbsolutePos => {
                let unit = self.unit_pos_sym().ok()?;
                let cube = self.cube_pos_sym().ok()?;
                let cd = self.smt.numeral(cube_dim as u64);
                let scaled = self.smt.times(cube, cd);
                Some(self.smt.plus(scaled, unit))
            }
            // X/Y/Z, plane, cluster builtins: out of the 1-D subset.
            _ => None,
        }
    }

    // -- control-flow walk ---------------------------------------------

    fn process_scope(&mut self, scope: &Scope) -> Result<(), Stop> {
        for inst in &scope.instructions {
            self.process_instruction(inst)?;
        }
        Ok(())
    }

    fn process_instruction(&mut self, inst: &Instruction) -> Result<(), Stop> {
        match &inst.operation {
            Operation::Copy(v) => {
                let val = self.value_of(v);
                self.bind_out(inst, val);
            }
            Operation::Arithmetic(a) => self.process_arithmetic(inst, a)?,
            Operation::Comparison(c) => self.process_comparison(inst, c)?,
            Operation::Operator(op) => self.process_operator(inst, op)?,
            Operation::Metadata(m) => self.process_metadata(inst, m)?,
            Operation::Branch(b) => self.process_branch(b)?,
            // Everything else (Bitwise, Atomic, Plane, CoopMma, Synchronization,
            // Barrier, Tma, NonSemantic, Marker, ...) is outside the modeled
            // subset. It is not fatal on its own: leave its `out` (if any)
            // unbound so any later obligation that actually depends on it
            // fails explicitly at that use site instead of here, where it
            // may be entirely irrelevant to array bounds (see module docs).
            _ => self.taint_out(inst),
        }
        Ok(())
    }

    /// Write `val` to `kind`'s memo slot, and — if a branch arm is
    /// currently being walked — record the write into the top frame of
    /// `write_log_stack` (module docs' "Branch-scoped write taint"). This is
    /// the single point every *genuine* variable write goes through
    /// (`bind_out`, `taint_out`, and the loop-carry pre/post taint in
    /// `process_range_loop`), as opposed to `value_of`'s read-only
    /// resolution caching, which must NOT be logged here — see the module
    /// docs for why logging a read's cache-fill would be actively wrong
    /// (it would spuriously re-taint e.g. `ABSOLUTE_POS` the first time a
    /// branch happens to be where it's lazily resolved).
    fn set_var(&mut self, kind: VariableKind, val: Option<SExpr>) {
        self.memo.insert(kind, val);
        if let Some(frame) = self.write_log_stack.last_mut() {
            frame.insert(kind);
        }
    }

    fn taint_out(&mut self, inst: &Instruction) {
        if let Some(out) = inst.out {
            self.set_var(out.kind, None);
        }
    }

    fn bind_out(&mut self, inst: &Instruction, val: Option<SExpr>) {
        if let Some(out) = inst.out {
            // Loop-carry refinement (module docs): a write to a currently-
            // carried variable stays tainted no matter what `val` resolves
            // to — never un-tainted mid-loop, since a binding computed
            // partway through the body walk would only be valid for later
            // uses within that same single symbolic iteration, and nothing
            // here tracks that scoping precisely enough to bound its reuse.
            let val = if self.is_carried(out.kind) { None } else { val };
            self.set_var(out.kind, val);
        }
    }

    /// Is `kind` in any currently-open loop's carried-variable set (see
    /// `carried_stack`)?
    fn is_carried(&self, kind: VariableKind) -> bool {
        self.carried_stack.iter().any(|carried| carried.contains(&kind))
    }

    fn process_arithmetic(&mut self, inst: &Instruction, a: &Arithmetic) -> Result<(), Stop> {
        let Some(out) = inst.out else { return Ok(()) };
        if !is_modeled_int(&out.ty) {
            self.taint_out(inst);
            return Ok(());
        }
        let val = match a {
            Arithmetic::Add(b) => self.binary_int(b, |s, l, r| s.plus(l, r)),
            Arithmetic::Sub(b) => self.binary_int(b, |s, l, r| s.sub(l, r)),
            Arithmetic::Mul(b) => self.binary_int(b, |s, l, r| s.times(l, r)),
            Arithmetic::Div(b) => self.divmod_int(b, |s, l, r| s.div(l, r))?,
            Arithmetic::Modulo(b) => self.divmod_int(b, |s, l, r| s.modulo(l, r))?,
            _ => None,
        };
        self.bind_out(inst, val);
        Ok(())
    }

    fn process_comparison(&mut self, inst: &Instruction, c: &Comparison) -> Result<(), Stop> {
        let val = match c {
            Comparison::Lower(b) => self.binary_int(b, |s, l, r| s.lt(l, r)),
            Comparison::LowerEqual(b) => self.binary_int(b, |s, l, r| s.lte(l, r)),
            Comparison::Equal(b) => self.binary_int(b, |s, l, r| s.eq(l, r)),
            Comparison::NotEqual(b) => self.binary_int(b, |s, l, r| {
                let eq = s.eq(l, r);
                s.not(eq)
            }),
            Comparison::GreaterEqual(b) => self.binary_int(b, |s, l, r| s.gte(l, r)),
            Comparison::Greater(b) => self.binary_int(b, |s, l, r| s.gt(l, r)),
            // Float-only predicates; not meaningful in the int-only encoding.
            Comparison::IsNan(_) | Comparison::IsInf(_) => None,
        };
        self.bind_out(inst, val);
        Ok(())
    }

    /// Resolve both operands of a `BinaryOperator` and apply `f`, but only
    /// when both operands are modeled integer types — a comparison or
    /// arithmetic op over floats (or bools) is left untainted-but-unmodeled.
    fn binary_int(
        &mut self,
        b: &cubecl::ir::BinaryOperator,
        f: impl FnOnce(&Context, SExpr, SExpr) -> SExpr,
    ) -> Option<SExpr> {
        if !is_modeled_int(&b.lhs.ty) || !is_modeled_int(&b.rhs.ty) {
            return None;
        }
        let l = self.value_of(&b.lhs)?;
        let r = self.value_of(&b.rhs)?;
        Some(f(self.smt, l, r))
    }

    /// Model `Arithmetic::Div`/`Arithmetic::Modulo` (see module docs
    /// "Div/mod-derived indices"): resolves both operands, then tries to
    /// discharge the internal side-obligation "divisor nonzero and both
    /// operands nonnegative" under the *current* path conditions + assumes.
    /// Only when that discharges do we bind `f` (SMT-LIB `div`/`mod`,
    /// Euclidean); otherwise the result is left tainted (`Ok(None)`) rather
    /// than erroring — the value may never feed an obligation. Propagates
    /// `Err` only for a genuine solver I/O failure.
    fn divmod_int(
        &mut self,
        b: &cubecl::ir::BinaryOperator,
        f: impl FnOnce(&Context, SExpr, SExpr) -> SExpr,
    ) -> Result<Option<SExpr>, Stop> {
        if !is_modeled_int(&b.lhs.ty) || !is_modeled_int(&b.rhs.ty) {
            return Ok(None);
        }
        let (Some(l), Some(r)) = (self.value_of(&b.lhs), self.value_of(&b.rhs)) else {
            return Ok(None);
        };

        let zero = self.smt.numeral(0);
        let eq_zero = self.smt.eq(r, zero);
        let rhs_nonzero = self.smt.not(eq_zero);
        let lhs_nonneg = self.smt.gte(l, zero);
        let rhs_nonneg = self.smt.gte(r, zero);
        let nonneg = self.smt.and(lhs_nonneg, rhs_nonneg);
        let side_obligation = self.smt.and(rhs_nonzero, nonneg);

        if !self.try_discharge(side_obligation)? {
            return Ok(None);
        }
        Ok(Some(f(self.smt, l, r)))
    }

    /// Push/assert-negated/check/pop `obligation`, returning whether it
    /// discharged (UNSAT under negation) — unlike `check_obligation`, a
    /// failure to discharge (SAT, or an inconclusive `unknown`) is *not*
    /// itself a proof failure here: callers (currently only `divmod_int`)
    /// use this to decide whether it's sound to *model* something, falling
    /// back to tainting when it isn't. A solver I/O error still propagates
    /// as a genuine `SolverError` — that's an implementation failure, not a
    /// soundness question.
    fn try_discharge(&mut self, obligation: SExpr) -> Result<bool, Stop> {
        self.smt.push().map_err(smt_err)?;
        let negated = self.smt.not(obligation);
        self.smt.assert(negated).map_err(smt_err)?;
        let response = self.smt.check();
        self.smt.pop().map_err(smt_err)?;
        match response {
            Ok(Response::Unsat) => Ok(true),
            Ok(Response::Sat) | Ok(Response::Unknown) => Ok(false),
            Err(e) => Err(smt_err(e)),
        }
    }

    /// Resolve both operands of a `BinaryOperator` whose operands are
    /// modeled `Bool`s and apply `f` — the boolean-logic counterpart of
    /// `binary_int`, used for `Operator::And`/`Or` (module docs "Boolean
    /// condition composition"). A tainted sub-condition taints the whole
    /// composed condition: resolution fails, explicitly, only at the
    /// branch/obligation site that actually needs the value.
    fn bool_binary(
        &mut self,
        b: &cubecl::ir::BinaryOperator,
        f: impl FnOnce(&Context, SExpr, SExpr) -> SExpr,
    ) -> Option<SExpr> {
        if !b.lhs.ty.is_bool() || !b.rhs.ty.is_bool() {
            return None;
        }
        let l = self.value_of(&b.lhs)?;
        let r = self.value_of(&b.rhs)?;
        Some(f(self.smt, l, r))
    }

    /// `bool_binary`'s unary counterpart, used for `Operator::Not`.
    fn bool_unary(
        &mut self,
        u: &cubecl::ir::UnaryOperator,
        f: impl FnOnce(&Context, SExpr) -> SExpr,
    ) -> Option<SExpr> {
        if !u.input.ty.is_bool() {
            return None;
        }
        let v = self.value_of(&u.input)?;
        Some(f(self.smt, v))
    }

    fn process_operator(&mut self, inst: &Instruction, op: &Operator) -> Result<(), Stop> {
        match op {
            Operator::Index(io) => self.process_index(inst, io, io.list),
            Operator::UncheckedIndex(io) => self.process_index(inst, io, io.list),
            Operator::IndexAssign(io) => {
                let list = inst.out();
                self.process_index_assign(inst, io, list)
            }
            Operator::UncheckedIndexAssign(io) => {
                let list = inst.out();
                self.process_index_assign(inst, io, list)
            }
            Operator::Cast(u) => {
                let Some(out) = inst.out else { return Ok(()) };
                let val = if is_modeled_int(&out.ty) && is_modeled_int(&u.input.ty) {
                    self.value_of(&u.input)
                } else {
                    None
                };
                self.bind_out(inst, val);
                Ok(())
            }
            // Boolean condition composition (module docs): CubeCL lowers
            // `&&`/`||`/`!` to these eagerly-evaluated operators.
            Operator::And(b) => {
                let val = self.bool_binary(b, |s, l, r| s.and(l, r));
                self.bind_out(inst, val);
                Ok(())
            }
            Operator::Or(b) => {
                let val = self.bool_binary(b, |s, l, r| s.or(l, r));
                self.bind_out(inst, val);
                Ok(())
            }
            Operator::Not(u) => {
                let val = self.bool_unary(u, |s, v| s.not(v));
                self.bind_out(inst, val);
                Ok(())
            }
            // Select/InitVector/CopyMemory* etc: not needed by the v0
            // subset; leave tainted.
            _ => {
                self.taint_out(inst);
                Ok(())
            }
        }
    }

    fn check_trivial_vectorization(
        &self,
        vector_size: cubecl::ir::VectorSize,
        unroll_factor: usize,
    ) -> Result<(), Stop> {
        if !(vector_size == 0 || vector_size == 1) || unroll_factor != 1 {
            return Err(Stop::OutOfSubset(format!(
                "vectorized/unrolled indexing (vector_size={vector_size}, \
                 unroll_factor={unroll_factor}) is outside the vericl v0 subset"
            )));
        }
        Ok(())
    }

    /// Classify an index *list* operand. Globals key their bound off a runtime
    /// `Length` symbol; a `SharedArray` keys it off the compile-time `length`
    /// carried in its `VariableKind` (module docs' "Shared arrays"). Anything
    /// else is outside the subset.
    fn array_ref(&self, list: &Variable) -> Result<ArrayRef, Stop> {
        match list.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => {
                Ok(ArrayRef::Global { id })
            }
            VariableKind::SharedArray { id, length, .. } => Ok(ArrayRef::Shared { id, length }),
            other => Err(Stop::OutOfSubset(format!(
                "indexing into `{other:?}` (not a global input/output or shared array) is outside \
                 the vericl v0 subset"
            ))),
        }
    }

    /// The bound SExpr and display name for an array reference. Global length
    /// is a (declared) runtime symbol; shared length is a concrete numeral.
    fn array_len_and_name(&mut self, aref: &ArrayRef) -> Result<(SExpr, String), Stop> {
        match *aref {
            ArrayRef::Global { id } => Ok((self.length_of(id)?, self.buffer_name(id))),
            ArrayRef::Shared { id, length } => {
                Ok((self.smt.numeral(length as u64), format!("shared_array({id})")))
            }
        }
    }

    fn process_index(
        &mut self,
        inst: &Instruction,
        io: &cubecl::ir::IndexOperator,
        list: Variable,
    ) -> Result<(), Stop> {
        self.check_trivial_vectorization(io.vector_size, io.unroll_factor)?;
        let aref = self.array_ref(&list)?;
        let (len, name) = self.array_len_and_name(&aref)?;
        let idx = self.value_of(&io.index).ok_or_else(|| {
            Stop::OutOfSubset(format!(
                "read index for `{name}[...]` depends on a construct outside the vericl v0 subset"
            ))
        })?;
        self.emit_obligation(len, &name, idx, "read")?;
        // The value *read* from the array is unknown (this checker has no
        // model of array contents) — taint, don't bind.
        self.taint_out(inst);
        Ok(())
    }

    fn process_index_assign(
        &mut self,
        inst: &Instruction,
        io: &cubecl::ir::IndexAssignOperator,
        list: Variable,
    ) -> Result<(), Stop> {
        self.check_trivial_vectorization(io.vector_size, io.unroll_factor)?;
        let aref = self.array_ref(&list)?;
        let (len, name) = self.array_len_and_name(&aref)?;
        let idx = self.value_of(&io.index).ok_or_else(|| {
            Stop::OutOfSubset(format!(
                "write index for `{name}[...] = ...` depends on a construct outside the vericl v0 \
                 subset"
            ))
        })?;
        self.emit_obligation(len, &name, idx, "write")?;
        self.taint_out(inst);
        Ok(())
    }

    fn emit_obligation(
        &mut self,
        len: SExpr,
        name: &str,
        idx: SExpr,
        kind: &str,
    ) -> Result<(), Stop> {
        let zero = self.smt.numeral(0);
        let ge0 = self.smt.gte(idx, zero);
        let lt_len = self.smt.lt(idx, len);
        let in_bounds = self.smt.and(ge0, lt_len);
        let description = format!("0 <= index < {name}.len() ({kind} access to `{name}`)");
        self.check_obligation(description, in_bounds)
    }

    fn process_metadata(&mut self, inst: &Instruction, m: &Metadata) -> Result<(), Stop> {
        let val = match m {
            Metadata::Length { var } => match var.kind {
                VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => {
                    Some(self.length_of(id)?)
                }
                _ => None,
            },
            // Metadata::BufferLength is deliberately never modeled: it is
            // the physical allocation length, not the caller-declared
            // logical length — conflating them would make the checker
            // unsound once inplace/aliasing exists (docs/ir-research.md §3).
            _ => None,
        };
        self.bind_out(inst, val);
        Ok(())
    }

    fn process_branch(&mut self, b: &Branch) -> Result<(), Stop> {
        match b {
            // Branch-scoped write taint (module docs): `self.memo` is
            // snapshotted before the arm, walked against the snapshot, and
            // — for `IfElse` — restored again before the other arm, so
            // neither arm ever sees the other's writes. After the
            // construct, the snapshot is restored once more and every
            // variable written anywhere in the arm(s) (tracked via
            // `write_log_stack`) is explicitly re-tainted, rather than
            // trusting whichever arm happened to run last.
            Branch::If(if_) => {
                let cond = self.cond_of(&if_.cond, "if")?;
                let snapshot = self.memo.clone();
                self.write_log_stack.push(HashSet::new());
                self.smt.push().map_err(smt_err)?;
                self.smt.assert(cond).map_err(smt_err)?;
                let r = self.process_scope(&if_.scope);
                self.smt.pop().map_err(smt_err)?;
                let written =
                    self.write_log_stack.pop().expect("just pushed, push/pop are balanced");
                r?;
                self.memo = snapshot;
                // Routed through `set_var` (not a raw `memo.insert`) so a
                // write two levels deep still reaches an *enclosing* arm's
                // own write-log frame, if there is one — see the module
                // docs' "composes correctly for nested branches".
                for k in written {
                    self.set_var(k, None);
                }
                Ok(())
            }
            Branch::IfElse(ie) => {
                let cond = self.cond_of(&ie.cond, "if/else")?;
                let snapshot = self.memo.clone();

                self.write_log_stack.push(HashSet::new());
                self.smt.push().map_err(smt_err)?;
                self.smt.assert(cond).map_err(smt_err)?;
                let r1 = self.process_scope(&ie.scope_if);
                self.smt.pop().map_err(smt_err)?;
                let written_if =
                    self.write_log_stack.pop().expect("just pushed, push/pop are balanced");
                r1?;

                // Restore the pre-branch snapshot before walking the else
                // arm: without this, the else arm would see the if arm's
                // writes (the confirmed round-2 manifestation).
                self.memo = snapshot.clone();

                self.write_log_stack.push(HashSet::new());
                let not_cond = self.smt.not(cond);
                self.smt.push().map_err(smt_err)?;
                self.smt.assert(not_cond).map_err(smt_err)?;
                let r2 = self.process_scope(&ie.scope_else);
                self.smt.pop().map_err(smt_err)?;
                let written_else =
                    self.write_log_stack.pop().expect("just pushed, push/pop are balanced");
                r2?;

                self.memo = snapshot;
                // Same `set_var` routing as the `If` arm above, for the
                // same nested-composition reason.
                for k in written_if.into_iter().chain(written_else) {
                    self.set_var(k, None);
                }
                Ok(())
            }
            Branch::RangeLoop(rl) => self.process_range_loop(rl),
            Branch::Loop(l) => self.process_loop(l),
            Branch::Switch(_) => {
                Err(Stop::OutOfSubset("`Branch::Switch` is outside the vericl v0 subset".into()))
            }
            Branch::Return | Branch::Break | Branch::Unreachable => Ok(()),
        }
    }

    fn cond_of(&mut self, cond: &Variable, site: &str) -> Result<SExpr, Stop> {
        self.value_of(cond).ok_or_else(|| {
            Stop::OutOfSubset(format!(
                "`{site}` condition depends on a construct outside the vericl v0 subset"
            ))
        })
    }

    fn process_range_loop(&mut self, rl: &cubecl::ir::RangeLoop) -> Result<(), Stop> {
        // Soundness guard (see module docs), MUST run before the bounds
        // assertions below: `start <= i (<)= end` only models a unit-stride
        // *ascending* range. `range_stepped` (CubeCL's stepped-range
        // constructor) can produce a descending loop where `start > end`
        // numerically, in which case those assertions are unsatisfiable —
        // the SMT context becomes infeasible and every obligation inside the
        // loop discharges vacuously (UNSAT because the context contradicts
        // itself, not because the access is safe), i.e. a false `Proved`.
        // Rejecting here, before any bounds assertion is pushed, closes that
        // gap outright rather than attempting to model the step.
        if rl.step.is_some() {
            return Err(Stop::OutOfSubset(
                "stepped range loop (range_stepped) is outside the vericl v0 subset: only \
                 unit-stride ascending ranges are modeled; stepped/descending loops are \
                 rejected rather than approximated"
                    .into(),
            ));
        }

        // Loop-carry refinement (see module docs): find every variable the
        // body (recursively, through nested branches) reassigns that was
        // already bound outside the loop -- loop-carried state (e.g. an
        // accumulator), which a single symbolic pass over the body cannot
        // soundly represent as "the value at an arbitrary iteration". Rather
        // than rejecting the whole loop, taint exactly those variables, both
        // before the walk (so a read-before-write inside the body doesn't
        // see the stale pre-loop value) and for the walk's whole duration
        // (`carried_stack`, consulted by `bind_out`/`taint_out`) -- so every
        // other index/branch in this loop, and every other loop in the
        // kernel, is still modeled exactly as before.
        let outer: HashSet<VariableKind> = self.memo.keys().copied().collect();
        let carried = scope_reassigned_vars(&rl.scope, &outer);
        for &k in &carried {
            self.set_var(k, None);
        }
        self.carried_stack.push(carried.clone());

        let r = self.process_range_loop_body(rl);

        self.carried_stack.pop();
        // Defensive: `bind_out`/`taint_out` already guarantee every carried
        // key is `None` by now (any write to it during the walk was forced
        // tainted), but re-asserting it here makes "and after the loop" an
        // explicit invariant rather than one that merely happens to hold.
        // Routed through `set_var` (not a raw `memo.insert`) so this also
        // registers as a write for an enclosing branch arm, if this loop is
        // itself nested inside one — see module docs' "Branch-scoped write
        // taint".
        for &k in &carried {
            self.set_var(k, None);
        }
        r
    }

    /// The bounds-assertion + body-walk portion of `process_range_loop`,
    /// factored out so the caller can unconditionally pop `carried_stack`
    /// (and re-taint) regardless of how this returns.
    fn process_range_loop_body(&mut self, rl: &cubecl::ir::RangeLoop) -> Result<(), Stop> {
        let start = self.value_of(&rl.start).ok_or_else(|| {
            Stop::OutOfSubset(
                "range-loop start bound depends on a construct outside the vericl v0 subset"
                    .into(),
            )
        })?;
        let end = self.value_of(&rl.end).ok_or_else(|| {
            Stop::OutOfSubset(
                "range-loop end bound depends on a construct outside the vericl v0 subset".into(),
            )
        })?;

        let i_sym = self.declare_int("loop_i", is_unsigned(&rl.i.ty))?;
        self.memo.insert(rl.i.kind, Some(i_sym));

        self.smt.push().map_err(smt_err)?;
        let ge_start = self.smt.gte(i_sym, start);
        self.smt.assert(ge_start).map_err(smt_err)?;
        let hi = if rl.inclusive { self.smt.lte(i_sym, end) } else { self.smt.lt(i_sym, end) };
        self.smt.assert(hi).map_err(smt_err)?;
        let r = self.process_scope(&rl.scope);
        self.smt.pop().map_err(smt_err)?;
        r
    }

    /// A `Branch::Loop` (CubeCL's `while`/`loop` desugaring). See the module
    /// docs' "`Branch::Loop` recognition (M2)" bullet.
    fn process_loop(&mut self, l: &Loop) -> Result<(), Stop> {
        // Cooperative loop (barrier inside the body): defer to the two-thread
        // race walker (milestone M3), which does not exist yet. Rejected
        // rather than modeled by a single-thread pass — checked FIRST, so any
        // barrier-carrying loop defers regardless of its guard shape.
        if scope_contains_sync_cube(&l.scope) {
            return Err(Stop::OutOfSubset(
                "cooperative loop (a `Branch::Loop` containing `sync_cube()`) — race walker not \
                 yet implemented (milestone M3); rejected rather than modeled without race \
                 analysis"
                    .into(),
            ));
        }
        // Non-cooperative: recognize the canonical `while` desugaring (leading
        // break-guard). Anything else (a trailing-break `loop`, an unbounded
        // loop) is not modeled — the pre-milestone rejection, unchanged.
        let Some(bg) = recognize_break_guard(&l.scope) else {
            return Err(Stop::OutOfSubset(
                "`Branch::Loop` (unbounded/break-terminated loop) is outside the vericl v0 subset"
                    .into(),
            ));
        };
        self.process_noncoop_loop(l, &bg)
    }

    /// Model a recognized non-cooperative `while` loop RangeLoop-style: the
    /// induction variable (a carried, integer guard operand) gets a fresh
    /// symbol bounded by the asserted guard; every other carried variable is
    /// tainted (module docs). Structured like `process_range_loop` so the
    /// `carried_stack` push/pop and defensive re-taint are unconditional.
    fn process_noncoop_loop(&mut self, l: &Loop, bg: &BreakGuard) -> Result<(), Stop> {
        let outer: HashSet<VariableKind> = self.memo.keys().copied().collect();
        let carried = scope_reassigned_vars(&l.scope, &outer);

        // Induction variables: carried, integer-typed operands the guard
        // *upper-bounds* (the ascending shape). Everything else carried is an
        // accumulator (tainted).
        let mut induction: HashMap<VariableKind, bool> = HashMap::new();
        for v in &bg.induction_candidates {
            if carried.contains(&v.kind) && is_modeled_int(&v.ty) {
                induction.insert(v.kind, is_unsigned(&v.ty));
            }
        }

        // Pre-bind before the body walk (like `process_range_loop`): induction
        // vars to a fresh symbol, every other carried var to taint — so a
        // read-before-write of an accumulator sees taint, not the stale
        // pre-loop value.
        for &k in &carried {
            if let Some(&unsigned) = induction.get(&k) {
                let sym = self.declare_int("loop_iv_", unsigned)?;
                self.set_var(k, Some(sym));
            } else {
                self.set_var(k, None);
            }
        }
        self.carried_stack.push(carried.clone());

        let r = self.process_noncoop_loop_body(l, bg);

        self.carried_stack.pop();
        // Defensive re-taint after the loop (same rationale as
        // `process_range_loop`): every carried variable — induction included —
        // is unknown once the loop is left.
        for &k in &carried {
            self.set_var(k, None);
        }
        r
    }

    /// The guard-assert + body-walk portion of `process_noncoop_loop`,
    /// factored out so the caller unconditionally pops `carried_stack`.
    fn process_noncoop_loop_body(&mut self, l: &Loop, bg: &BreakGuard) -> Result<(), Stop> {
        let insts = &l.scope.instructions;
        // Bind the guard comparison (before opening any SMT scope), then
        // resolve it. A guard that depends on unmodeled state is rejected
        // rather than walked with an unconstrained induction symbol (which
        // could manufacture a false `Refuted`).
        self.process_instruction(&insts[bg.guard_idx])?;
        let Some(guard) = self.value_of(&bg.guard_var) else {
            return Err(Stop::OutOfSubset(
                "loop guard condition depends on a construct outside the vericl v0 subset".into(),
            ));
        };

        self.smt.push().map_err(smt_err)?;
        self.smt.assert(guard).map_err(smt_err)?;
        // Walk the real body under the asserted guard. The `nc = Not c` and
        // the `if nc { break }` scaffolding are skipped: `nc` feeds only the
        // break, and the break arm carries no obligation.
        let mut result = Ok(());
        for inst in &insts[bg.body_start..] {
            if let Err(e) = self.process_instruction(inst) {
                result = Err(e);
                break;
            }
        }
        self.smt.pop().map_err(smt_err)?;
        result
    }

    fn check_obligation(&mut self, description: String, obligation: SExpr) -> Result<(), Stop> {
        self.smt.push().map_err(smt_err)?;
        let negated = self.smt.not(obligation);
        self.smt.assert(negated).map_err(smt_err)?;
        let response = self.smt.check();
        let outcome = match response {
            Ok(Response::Unsat) => {
                self.obligations += 1;
                Ok(())
            }
            Ok(Response::Sat) => {
                let counterexample = self.render_counterexample();
                Err(Stop::Refuted { obligation: description, counterexample })
            }
            Ok(Response::Unknown) => {
                Err(Stop::SolverError(format!("z3 returned `unknown` for obligation: {description}")))
            }
            Err(e) => Err(smt_err(e)),
        };
        self.smt.pop().map_err(smt_err)?;
        outcome
    }

    fn render_counterexample(&mut self) -> String {
        let vars: Vec<SExpr> = self.declared.iter().map(|(_, e)| *e).collect();
        match self.smt.get_value(vars) {
            Ok(vals) => self
                .declared
                .iter()
                .zip(vals.iter())
                .map(|((name, _), (_, val))| format!("{name}={}", self.smt.display(*val)))
                .collect::<Vec<_>>()
                .join(", "),
            Err(e) => format!("<failed to read counterexample model: {e}>"),
        }
    }

    // -- variable resolution ---------------------------------------------

    /// Resolve a `Variable` to its symbolic value, or `None` if it depends
    /// on something outside the modeled subset. See module docs for why
    /// this is not itself an error — callers that actually need the value
    /// (obligations, branch/loop conditions) turn `None` into an
    /// `OutOfSubset` at their own use site, with a specific description.
    fn value_of(&mut self, var: &Variable) -> Option<SExpr> {
        if let Some(cached) = self.memo.get(&var.kind) {
            return *cached;
        }
        let resolved = match var.kind {
            VariableKind::Constant(cv) => self.constant_expr(cv, &var.ty),
            VariableKind::Builtin(b) => self.builtin_value(b),
            VariableKind::GlobalScalar(id) => {
                if is_modeled_int(&var.ty) {
                    self.declare_int(&format!("scalar{id}_"), is_unsigned(&var.ty)).ok()
                } else {
                    None
                }
            }
            // Locals not yet bound by a modeled instruction, and arrays used as
            // scalar values: unsupported here.
            _ => None,
        };
        self.memo.insert(var.kind, resolved);
        resolved
    }

    fn constant_expr(&mut self, cv: ConstantValue, ty: &Type) -> Option<SExpr> {
        // Bool constants (e.g. a literal `true`/`false` folded into a
        // composed `&&`/`||`/`!` condition) are modeled directly as SMT
        // Bools — a natural companion to boolean condition composition
        // (module docs), and, like every other constant here, strictly
        // sound: it's a faithful term for the actual constant value.
        if ty.is_bool() {
            return match cv {
                ConstantValue::Bool(b) => Some(if b { self.smt.true_() } else { self.smt.false_() }),
                _ => None,
            };
        }
        if !is_modeled_int(ty) {
            return None;
        }
        match cv {
            ConstantValue::Int(v) if v < 0 => {
                let mag = self.smt.numeral((-v) as u64);
                Some(self.smt.negate(mag))
            }
            ConstantValue::Int(v) => Some(self.smt.numeral(v as u64)),
            ConstantValue::UInt(v) => Some(self.smt.numeral(v)),
            ConstantValue::Bool(_) | ConstantValue::Float(_) => None,
        }
    }
}

/// Whether `ty` is a plain (non-vector, non-atomic) integer type this
/// checker models as an SMT `Int` — explicitly excludes `Bool` even though
/// `ElemType::is_int()` counts it (booleans are built directly from
/// `Comparison`, never arithmetic).
fn is_modeled_int(ty: &Type) -> bool {
    ty.is_int() && !ty.is_bool()
}

fn is_unsigned(ty: &Type) -> bool {
    ty.is_unsigned_int() && !ty.is_bool()
}

/// Every `VariableKind` that `scope` (recursively, through nested branches)
/// reassigns and that is already in `outer` — i.e. every carried
/// (loop-accumulator-shaped) variable a `RangeLoop` body writes to. Used by
/// `process_range_loop`'s loop-carry refinement (module docs) to taint
/// exactly the carried variables rather than rejecting the whole loop.
/// Collects every match (not just the first) since the caller needs the
/// complete set to taint.
fn scope_reassigned_vars(scope: &Scope, outer: &HashSet<VariableKind>) -> HashSet<VariableKind> {
    let mut found = HashSet::new();
    collect_reassigned_vars(scope, outer, &mut found);
    found
}

fn collect_reassigned_vars(
    scope: &Scope,
    outer: &HashSet<VariableKind>,
    found: &mut HashSet<VariableKind>,
) {
    for inst in &scope.instructions {
        if let Some(out) = inst.out {
            if outer.contains(&out.kind) {
                found.insert(out.kind);
            }
        }
        if let Operation::Branch(b) = &inst.operation {
            match b {
                Branch::If(if_) => collect_reassigned_vars(&if_.scope, outer, found),
                Branch::IfElse(ie) => {
                    collect_reassigned_vars(&ie.scope_if, outer, found);
                    collect_reassigned_vars(&ie.scope_else, outer, found);
                }
                Branch::Switch(sw) => {
                    collect_reassigned_vars(&sw.scope_default, outer, found);
                    for (_, s) in &sw.cases {
                        collect_reassigned_vars(s, outer, found);
                    }
                }
                Branch::RangeLoop(rl) => collect_reassigned_vars(&rl.scope, outer, found),
                Branch::Loop(l) => collect_reassigned_vars(&l.scope, outer, found),
                Branch::Return | Branch::Break | Branch::Unreachable => {}
            }
        }
    }
}

/// The recognized leading break-guard of a canonical `while` desugaring (§2.4,
/// module docs' "`Branch::Loop` recognition"). See `recognize_break_guard`.
struct BreakGuard {
    /// Index of the guard comparison in the loop body (the `c = <cond>`
    /// instruction, always `0`).
    guard_idx: usize,
    /// The comparison's `out` variable `c` — the condition that holds
    /// throughout the body (the loop continues while `c`).
    guard_var: Variable,
    /// Guard operands the guard *upper-bounds* (the `v` in `v < n` / `n > v`)
    /// — the only induction-variable candidates. A carried, integer one of
    /// these gets a fresh symbol bounded above by the asserted guard; a
    /// lower-bound guard (`v > 0`) or a `!=`/`==` guard yields no candidate,
    /// so its operand stays tainted (→ `OutOfSubset`) rather than a symbol
    /// bounded only from below — which could manufacture a spurious
    /// `Refuted` on a safe descending loop.
    induction_candidates: Vec<Variable>,
    /// Index at which the real body begins (just past the `if nc { break }`).
    body_start: usize,
}

/// Match the canonical `while cond { … }` desugaring's leading break-guard:
/// `[0] c = <cmp>`, `[1] nc = Not c`, `[2] if nc { break }`, then the body.
/// Returns `None` for anything else (a trailing-break `loop`, an unbounded
/// loop, a non-canonical shape) — such a `Branch::Loop` is not modeled.
fn recognize_break_guard(scope: &Scope) -> Option<BreakGuard> {
    let insts = &scope.instructions;
    if insts.len() < 3 {
        return None;
    }
    // [0] c = <comparison>
    let Operation::Comparison(cmp) = &insts[0].operation else {
        return None;
    };
    let guard_var = insts[0].out?;
    let induction_candidates = guard_upper_bounded_operands(cmp);
    // [1] nc = Not c
    let Operation::Operator(Operator::Not(u)) = &insts[1].operation else {
        return None;
    };
    if u.input.kind != guard_var.kind {
        return None;
    }
    let nc = insts[1].out?;
    // [2] if nc { break }
    let Operation::Branch(Branch::If(if_)) = &insts[2].operation else {
        return None;
    };
    if if_.cond.kind != nc.kind || !scope_is_single_break(&if_.scope) {
        return None;
    }
    Some(BreakGuard { guard_idx: 0, guard_var, induction_candidates, body_start: 3 })
}

/// The operand(s) a guard comparison, asserted true in the loop body, bounds
/// from *above* — the ascending `while v < n` / `while n > v` shape. Only such
/// a variable can be a sound induction variable: a fresh symbol bounded above
/// by the asserted guard over-approximates the actual induction values (module
/// docs). A lower-bound guard (`v > 0`), or an `==`/`!=`/`IsNan`/`IsInf`
/// guard, yields no candidate — its operand stays tainted.
fn guard_upper_bounded_operands(cmp: &Comparison) -> Vec<Variable> {
    match cmp {
        // v < n  /  v <= n  →  n upper-bounds the lhs.
        Comparison::Lower(b) | Comparison::LowerEqual(b) => vec![b.lhs],
        // n > v  /  n >= v  →  n upper-bounds the rhs.
        Comparison::Greater(b) | Comparison::GreaterEqual(b) => vec![b.rhs],
        Comparison::Equal(_)
        | Comparison::NotEqual(_)
        | Comparison::IsNan(_)
        | Comparison::IsInf(_) => vec![],
    }
}

/// Whether `scope` is exactly a single `break`.
fn scope_is_single_break(scope: &Scope) -> bool {
    scope.instructions.len() == 1
        && matches!(scope.instructions[0].operation, Operation::Branch(Branch::Break))
}

/// Whether `scope` (recursively, through nested branches and loops) contains
/// a `SyncCube` barrier — i.e. whether a loop is *cooperative* (module docs'
/// "`Branch::Loop` recognition"). Recursive so a barrier nested inside an
/// `if` within the loop still marks it cooperative.
fn scope_contains_sync_cube(scope: &Scope) -> bool {
    scope.instructions.iter().any(|inst| match &inst.operation {
        Operation::Synchronization(Synchronization::SyncCube) => true,
        Operation::Branch(b) => match b {
            Branch::If(if_) => scope_contains_sync_cube(&if_.scope),
            Branch::IfElse(ie) => {
                scope_contains_sync_cube(&ie.scope_if) || scope_contains_sync_cube(&ie.scope_else)
            }
            Branch::Switch(sw) => {
                scope_contains_sync_cube(&sw.scope_default)
                    || sw.cases.iter().any(|(_, s)| scope_contains_sync_cube(s))
            }
            Branch::RangeLoop(rl) => scope_contains_sync_cube(&rl.scope),
            Branch::Loop(l) => scope_contains_sync_cube(&l.scope),
            Branch::Return | Branch::Break | Branch::Unreachable => false,
        },
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::prelude::*;

    #[cube(launch)]
    fn prover_test_axpy(alpha: f32, x: &Array<f32>, y: &mut Array<f32>) {
        if ABSOLUTE_POS < y.len() {
            y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
        }
    }

    #[cube(launch)]
    fn prover_test_axpy_off_by_one(alpha: f32, x: &Array<f32>, y: &mut Array<f32>) {
        if ABSOLUTE_POS <= y.len() {
            y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
        }
    }

    fn build_axpy() -> KernelDefinition {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let alpha = <f32 as LaunchArg>::expand(&Default::default(), &mut builder);
        let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_axpy::expand(&mut builder.scope, alpha, x, y);
        builder.build(KernelSettings::default())
    }

    fn build_axpy_off_by_one() -> KernelDefinition {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let alpha = <f32 as LaunchArg>::expand(&Default::default(), &mut builder);
        let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_axpy_off_by_one::expand(&mut builder.scope, alpha, x, y);
        builder.build(KernelSettings::default())
    }

    const AXPY_BUFFERS: &[BufferParam] =
        &[BufferParam { name: "x", is_output: false }, BufferParam { name: "y", is_output: true }];

    /// Positive control: a properly guarded access (`ABSOLUTE_POS <
    /// y.len()`) proves, given the `x.len() == y.len()` assume that makes
    /// the `x` read provable too (docs/ir-research.md §4: without it, the
    /// same obligation is SAT — asserted directly below as well).
    #[test]
    fn guarded_access_proves() {
        let def = build_axpy();
        let assumes = [Assume::LenEq { a: "x", b: "y" }];
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &assumes) {
            ProveResult::Proved { obligations } => {
                // x[pos] read, y[pos] read, y[pos] write.
                assert_eq!(obligations, 3);
            }
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// The `x.len() == y.len()` assume is load-bearing: without it, z3 can
    /// pick `x.len() = 0` with `pos = 0 < y.len()`, refuting the `x` read.
    #[test]
    fn guarded_access_without_len_assume_refutes() {
        let def = build_axpy();
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[]) {
            ProveResult::Refuted { .. } => {}
            other => panic!("expected Refuted, got {other:?}"),
        }
    }

    /// Negative control: `axpy_off_by_one`'s guard is `ABSOLUTE_POS <=
    /// y.len()`, so `ABSOLUTE_POS == y.len()` satisfies the guard but is
    /// out of bounds — the checker must refute with a counterexample that
    /// exhibits exactly that (`abs_pos` == the buffer length).
    #[test]
    fn off_by_one_guard_refutes_with_counterexample() {
        let def = build_axpy_off_by_one();
        let assumes = [Assume::LenEq { a: "x", b: "y" }];
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &assumes) {
            ProveResult::Refuted { obligation, counterexample } => {
                println!("refuted: {obligation}\ncounterexample: {counterexample}");
                assert!(!counterexample.is_empty());
                assert!(counterexample.contains("abs_pos"));
                assert!(counterexample.contains("len_y"));
            }
            other => panic!("expected Refuted, got {other:?}"),
        }
    }

    /// `z3_version` reports something when the binary is on PATH (it is on
    /// this machine and in CI, per the task setup) rather than panicking or
    /// silently returning garbage.
    #[test]
    fn z3_version_reports_a_version_string() {
        let v = z3_version().expect("z3 should be on PATH");
        assert!(v.to_lowercase().contains("z3"), "unexpected version string: {v}");
    }

    // -----------------------------------------------------------------
    // Branch-scoped write taint (If/IfElse) — REGRESSION, adversarial
    // soundness review round 2. See the module docs' "Branch-scoped write
    // taint" bullet for the fix; each test here pins one of the three
    // confirmed false-`Proved` manifestations, plus a nested-branch
    // composition check.
    // -----------------------------------------------------------------

    /// Manifestation 1 (reviewer's `if_merge_bug` shape): a variable
    /// clamped to a safe value inside an `If` with no `else` must not leak
    /// that clamp past the branch — `idx` is really `ABSOLUTE_POS`
    /// (unbounded) on every thread that doesn't take the (near-impossible)
    /// guard, but pre-fix the prover treated `idx == 0` as unconditional
    /// after the `if` closed, `Proved`ing an access that's genuinely
    /// unbounded. Post-fix: `idx` is tainted (written inside the arm), so
    /// the write index resolution fails explicitly, right here, rather
    /// than silently (or worse, `Proved`).
    #[cube(launch)]
    fn prover_test_if_write_leak(y: &mut Array<f32>) {
        let mut idx: usize = ABSOLUTE_POS;
        if ABSOLUTE_POS >= 1000000usize {
            idx = 0usize;
        }
        y[idx] = 1.0f32;
    }

    #[test]
    fn branch_write_does_not_leak_past_if() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let y = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_if_write_leak::expand(&mut builder.scope, y);
        let def = builder.build(KernelSettings::default());

        let buffers = [BufferParam { name: "y", is_output: true }];
        // Pin y.len() == 1: if the merge bug were still present, the
        // prover would see `idx` as unconditionally 0 here (0 < 1, safe)
        // even though the real value is ABSOLUTE_POS on almost every
        // thread of any dispatch with more than one thread.
        match prove_bounds_freedom(&def, &buffers, &[Assume::LenEqConst { a: "y", value: 1 }]) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("write index") && reason.contains("y"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (idx tainted, never Proved), got {other:?}"),
        }
    }

    /// Manifestation 2 (reviewer's `if_else_merge_bug` shape): the else
    /// arm must not see the if arm's writes. `idx` is untouched on the
    /// else path, so its real value there is `ABSOLUTE_POS` — genuinely
    /// unbounded against `y.len() == 1`. Pre-fix, walking both arms
    /// against the same unscoped `self.memo` meant the else arm inherited
    /// whatever the if arm (walked first) had already written, which
    /// happened to still be unsafe here (so this exact kernel refuted even
    /// before the fix) — but for the wrong reason (a leaked value, not a
    /// correctly-scoped one). Fixed: the else arm resolves `idx` from the
    /// restored pre-branch snapshot (`ABSOLUTE_POS`), and the obligation
    /// is refuted on genuine grounds.
    // `idx = 0usize` below is a dead write by design (mirrors the
    // reviewer's exact repro kernel) — the whole point of this shape is
    // that it must not leak, not that it's ever read.
    #[cube(launch)]
    #[allow(unused_assignments)]
    fn prover_test_if_arm_write_leaks_into_else(y: &mut Array<f32>) {
        let mut idx: usize = ABSOLUTE_POS;
        if ABSOLUTE_POS >= 1000000usize {
            idx = 0usize;
        } else {
            y[idx] = 2.0f32;
        }
    }

    #[test]
    fn if_arm_write_does_not_leak_into_else_arm() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let y = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_if_arm_write_leaks_into_else::expand(&mut builder.scope, y);
        let def = builder.build(KernelSettings::default());

        let buffers = [BufferParam { name: "y", is_output: true }];
        match prove_bounds_freedom(&def, &buffers, &[Assume::LenEqConst { a: "y", value: 1 }]) {
            ProveResult::Refuted { obligation, counterexample } => {
                assert!(obligation.contains('y'), "unexpected obligation: {obligation}");
                assert!(!counterexample.is_empty());
            }
            other => panic!("expected Refuted (else arm's real, unbounded idx), got {other:?}"),
        }
    }

    /// Manifestation 3 (reviewer's `post_ifelse_false_proved` shape): a
    /// post-`IfElse` read must not silently resolve to whichever arm was
    /// walked last. Both arms write `idx` (the if arm writes the real,
    /// unbounded `ABSOLUTE_POS`; the else arm clamps to a safe `0`) — pre-
    /// fix, the else arm's write (processed last, per the original
    /// sequential walk with no restore) always won at the merge point,
    /// making the post-branch `y[idx]` look like `y[0]`, always safe, even
    /// though `idx == ABSOLUTE_POS` on the overwhelmingly common dispatch
    /// (the if arm's own condition, `ABSOLUTE_POS < 1_000_000`). Fixed: the
    /// merge taints `idx` (written in both arms) rather than merging to
    /// either arm's value, so the post-branch use fails explicitly.
    // The initial `idx = ABSOLUTE_POS` below is unconditionally overwritten
    // on both arms (that's the point — both arms write `idx`, feeding the
    // post-merge taint this test pins).
    #[cube(launch)]
    #[allow(unused_assignments)]
    fn prover_test_post_ifelse_merge_taints(y: &mut Array<f32>) {
        let mut idx: usize = ABSOLUTE_POS;
        if ABSOLUTE_POS < 1000000usize {
            idx = ABSOLUTE_POS;
        } else {
            idx = 0usize;
        }
        y[idx] = 5.0f32;
    }

    #[test]
    fn post_ifelse_merge_taints_branch_written_vars() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let y = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_post_ifelse_merge_taints::expand(&mut builder.scope, y);
        let def = builder.build(KernelSettings::default());

        let buffers = [BufferParam { name: "y", is_output: true }];
        match prove_bounds_freedom(&def, &buffers, &[Assume::LenEqConst { a: "y", value: 1 }]) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("write index") && reason.contains("y"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (idx tainted post-merge), got {other:?}"),
        }
    }

    /// Nested branches: a write two levels deep (inside an `IfElse` nested
    /// inside an outer `If`'s only arm) must still reach the *outer*
    /// merge's taint set — proving the write-log stack composes
    /// recursively, not just for a single level of nesting — AND must not
    /// leak into the outer `If`'s sibling had there been one (there isn't
    /// one here; `if_arm_write_does_not_leak_into_else_arm` above already
    /// covers the single-level sibling-leak case). Real semantics: `idx`
    /// is written by the inner `IfElse` (to `0` or `1`, depending on which
    /// inner arm), so by the time the outer `If` closes, `idx`'s value on
    /// the taken-outer-arm path is genuinely path-dependent — the outer
    /// merge must taint it, not leave it at its restored pre-outer-if
    /// value (`ABSOLUTE_POS`, which would itself still correctly refute —
    /// the interesting failure mode this test guards against is the
    /// *opposite*: the inner merge's taint failing to propagate up at all,
    /// silently leaving `idx` resolved to whatever `self.memo` last held
    /// for it before the inner branch, which is exactly the bug this test
    /// would need to exist to catch if write-log composition were broken).
    #[cube(launch)]
    fn prover_test_nested_branches(y: &mut Array<f32>) {
        let mut idx: usize = ABSOLUTE_POS;
        if ABSOLUTE_POS < 1000000usize {
            if ABSOLUTE_POS < 500000usize {
                idx = 0usize;
            } else {
                idx = 1usize;
            }
        }
        y[idx] = 5.0f32;
    }

    #[test]
    fn nested_branches_restore_correctly() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let y = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_nested_branches::expand(&mut builder.scope, y);
        let def = builder.build(KernelSettings::default());

        let buffers = [BufferParam { name: "y", is_output: true }];
        match prove_bounds_freedom(&def, &buffers, &[Assume::LenEqConst { a: "y", value: 1 }]) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("write index") && reason.contains("y"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!(
                "expected OutOfSubset (idx tainted by the outer If's merge, via the nested \
                 IfElse's writes propagating up through write_log_stack), got {other:?}"
            ),
        }
    }

    /// The other half of "nested branches restore correctly": a write two
    /// levels deep (inside an `IfElse` nested inside the OUTER `IfElse`'s
    /// `if` arm) must not leak into the OUTER construct's own `else` arm
    /// (a true sibling, unlike the no-else case above). `idx` is untouched
    /// on the outer else path, so its real value there is `ABSOLUTE_POS` —
    /// genuinely unbounded against `y.len() == 1`. If the inner branch's
    /// writes ever escaped their own snapshot/restore *before* the outer
    /// snapshot is restored for the outer else arm, they'd corrupt what the
    /// outer else arm sees; since the outer restore is a full `self.memo`
    /// clone taken before the outer `if` arm (nested branch included) ever
    /// runs, this composes automatically — this test exists to pin that
    /// down explicitly, at two levels of nesting, rather than trust it by
    /// construction.
    // Both nested-arm writes below are dead by design (mirrors the other
    // sibling-leak shape above) — the point is that they must not leak,
    // not that they're read.
    #[cube(launch)]
    #[allow(unused_assignments)]
    fn prover_test_nested_branch_write_does_not_leak_into_outer_sibling(y: &mut Array<f32>) {
        let mut idx: usize = ABSOLUTE_POS;
        if ABSOLUTE_POS >= 1000000usize {
            if ABSOLUTE_POS >= 2000000usize {
                idx = 0usize;
            } else {
                idx = 1usize;
            }
        } else {
            y[idx] = 8.0f32;
        }
    }

    #[test]
    fn nested_branch_write_does_not_leak_into_outer_sibling() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let y = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_nested_branch_write_does_not_leak_into_outer_sibling::expand(
            &mut builder.scope,
            y,
        );
        let def = builder.build(KernelSettings::default());

        let buffers = [BufferParam { name: "y", is_output: true }];
        match prove_bounds_freedom(&def, &buffers, &[Assume::LenEqConst { a: "y", value: 1 }]) {
            ProveResult::Refuted { obligation, counterexample } => {
                assert!(obligation.contains('y'), "unexpected obligation: {obligation}");
                assert!(!counterexample.is_empty());
            }
            other => panic!(
                "expected Refuted (outer else arm's real, unbounded idx — untouched by the \
                 nested IfElse in the outer if arm), got {other:?}"
            ),
        }
    }

    #[cube(launch)]
    fn prover_test_ranged_copy(x: &Array<u32>, y: &mut Array<u32>) {
        for i in 0..y.len() {
            y[i] = x[i];
        }
    }

    #[cube(launch)]
    fn prover_test_ranged_accumulate(x: &Array<u32>, y: &mut Array<u32>) {
        let mut idx = 0u32;
        for i in 0..x.len() {
            idx += x[i];
        }
        y[idx as usize] = 1u32;
    }

    /// Loop-carry refinement positive control (module docs): `acc` is
    /// carried (accumulated across iterations), but it only ever feeds the
    /// *value* written to `y`, never an index or branch condition — the
    /// write index is a plain `ABSOLUTE_POS` guard, identical in shape to
    /// `prover_test_axpy`'s. Before the refinement, the whole loop was
    /// rejected wholesale (`loop_carried_accumulator_is_out_of_subset`
    /// below) regardless of whether the carried state ever reached an
    /// index; after it, this kernel proves.
    #[cube(launch)]
    fn prover_test_ranged_sum_then_guarded_write(x: &Array<u32>, y: &mut Array<u32>) {
        let mut acc = 0u32;
        for i in 0..x.len() {
            acc += x[i];
        }
        if ABSOLUTE_POS < y.len() {
            y[ABSOLUTE_POS] = acc;
        }
    }

    /// `Branch::RangeLoop` modeled as a fresh var in `[start, end)`, no
    /// unrolling: every index inside the loop body is checked for
    /// arbitrary `i` in range, which is sound for (and covers) every
    /// concrete iteration.
    #[test]
    fn bounded_range_loop_proves() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<u32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_ranged_copy::expand(&mut builder.scope, x, y);
        let def = builder.build(KernelSettings::default());

        let assumes = [Assume::LenEq { a: "x", b: "y" }];
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &assumes) {
            ProveResult::Proved { obligations } => assert_eq!(obligations, 2), // x[i] read, y[i] write
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// Loop-carry refinement negative control (module docs) — updated for
    /// the refinement: `idx` (carried) is used directly as the write
    /// *index*, so the taint that the carried refinement applies to `idx`
    /// must still surface, just at the specific site that actually needs
    /// the value (`y[idx as usize] = ...`) rather than as a wholesale
    /// rejection of the whole loop shape. Before the refinement this was
    /// `OutOfSubset` with a reason naming "loop-carried" directly (the loop
    /// itself was rejected); after it, the loop is walked (and, e.g., the
    /// `x[i]` read inside it still discharges), and it's specifically the
    /// `y[idx as usize]` write index resolution that fails, since `idx` is
    /// tainted by the time it's read there. Either way: never `Proved`.
    #[test]
    fn loop_carried_accumulator_used_as_index_is_out_of_subset() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<u32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_ranged_accumulate::expand(&mut builder.scope, x, y);
        let def = builder.build(KernelSettings::default());

        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[]) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("write index") && reason.contains("y"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset, got {other:?}"),
        }
    }

    /// Loop-carry refinement positive control (module docs, "Result:
    /// accumulator kernels whose indices don't depend on carried state
    /// become provable"): `acc` is carried, but never feeds an index — the
    /// write is guarded by a plain `ABSOLUTE_POS < y.len()`, so bounds
    /// obligations for both the in-loop `x[i]` read and the post-loop
    /// `y[ABSOLUTE_POS]` write discharge even though the kernel has
    /// loop-carried state. This is the exact regression the refinement
    /// exists to fix: before it, this kernel was wholesale `OutOfSubset`
    /// (same as the negative control above) despite being genuinely safe.
    #[test]
    fn loop_carried_accumulator_unused_as_index_proves() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<u32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_ranged_sum_then_guarded_write::expand(&mut builder.scope, x, y);
        let def = builder.build(KernelSettings::default());

        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[]) {
            // x[i] read (inside the loop), y[ABSOLUTE_POS] write (guarded).
            ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    #[cube(launch)]
    fn prover_test_stepped_descending_copy(x: &Array<u32>, y: &mut Array<u32>) {
        let n = y.len() as i32;
        for i in cubecl::prelude::range_stepped(n - 1, -1, -1) {
            let idx = i as usize;
            y[idx] = x[idx];
        }
    }

    /// REGRESSION (adversarial soundness review): `RangeLoop.step` is never
    /// read by the ascending-bounds model (`start <= i < end`). CubeCL's
    /// `range_stepped` can produce a descending loop (`start > end`
    /// numerically), for which those assertions are unsatisfiable — an
    /// infeasible SMT context vacuously "proves" every obligation inside,
    /// regardless of whether the body is actually safe. `process_range_loop`
    /// must reject any `step.is_some()` outright rather than approximate it.
    #[test]
    fn stepped_range_loop_is_out_of_subset() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<u32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_stepped_descending_copy::expand(&mut builder.scope, x, y);
        let def = builder.build(KernelSettings::default());

        let assumes = [Assume::LenEq { a: "x", b: "y" }];
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &assumes) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("stepped") || reason.contains("range_stepped"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Boolean condition composition (&&/||/!).
    // -----------------------------------------------------------------

    /// Regression pin for the shape `fir3` (vericl-examples) used to need a
    /// workaround for: a `pos >= 1 && pos < len`-style conjoined guard
    /// protecting a shifted read. Before boolean composition was modeled,
    /// this was `OutOfSubset` ("`if` condition depends on a construct
    /// outside the vericl v0 subset") since `Operator::And`'s output was
    /// tainted; now it proves.
    #[cube(launch)]
    fn prover_test_and_guard(x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS >= 1usize && ABSOLUTE_POS < y.len() {
            y[ABSOLUTE_POS] = x[ABSOLUTE_POS - 1];
        }
    }

    #[test]
    fn and_guard_proves() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<u32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_and_guard::expand(&mut builder.scope, x, y);
        let def = builder.build(KernelSettings::default());

        let assumes = [Assume::LenEq { a: "x", b: "y" }];
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &assumes) {
            // y[pos] write, x[pos-1] read.
            ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// Negative control (module docs / task spec): an `&&` guard whose arms
    /// don't actually protect the access must still `Refuted`, not
    /// `Proved` — composing `&&` correctly must never *widen* what's
    /// provable. Shaped like `axpy_off_by_one` (an off-by-one `<=` bound)
    /// with a second, genuinely non-trivial but insufficient arm ANDed in,
    /// so neither arm alone nor their conjunction actually excludes
    /// `ABSOLUTE_POS == y.len()`.
    #[cube(launch)]
    fn prover_test_and_guard_insufficient(x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS <= y.len() && ABSOLUTE_POS < 1_000_000usize {
            y[ABSOLUTE_POS] = x[ABSOLUTE_POS];
        }
    }

    #[test]
    fn and_guard_insufficient_refutes() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<u32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_and_guard_insufficient::expand(&mut builder.scope, x, y);
        let def = builder.build(KernelSettings::default());

        let assumes = [Assume::LenEq { a: "x", b: "y" }];
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &assumes) {
            ProveResult::Refuted { .. } => {}
            other => panic!("expected Refuted, got {other:?}"),
        }
    }

    /// Regression: pins that nested `if`s (the prover's *other* condition-
    /// composition shape, driven by the SMT push/pop path-condition stack
    /// rather than an `Operator::And` term) still prove exactly as before —
    /// kept as a prover unit test rather than a public example now that
    /// `fir3` (vericl-examples) has moved to the more idiomatic `&&` form
    /// (see that crate's doc comments).
    #[cube(launch)]
    fn prover_test_nested_if_guard(x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS < y.len() {
            let mut acc = x[ABSOLUTE_POS];
            if ABSOLUTE_POS >= 1usize {
                acc += x[ABSOLUTE_POS - 1];
            }
            y[ABSOLUTE_POS] = acc;
        }
    }

    #[test]
    fn nested_if_guard_still_proves() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<u32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_nested_if_guard::expand(&mut builder.scope, x, y);
        let def = builder.build(KernelSettings::default());

        let assumes = [Assume::LenEq { a: "x", b: "y" }];
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &assumes) {
            // x[pos] read, y[pos] write, guarded x[pos-1] read.
            ProveResult::Proved { obligations } => assert_eq!(obligations, 3),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// `||` positive control (De Morgan's over the negated condition for the
    /// `else` branch exercises `Operator::Or` too — see `process_branch`'s
    /// `IfElse` handling).
    #[cube(launch)]
    fn prover_test_or_guard_proves(x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS < y.len() || ABSOLUTE_POS < y.len() {
            y[ABSOLUTE_POS] = x[ABSOLUTE_POS];
        }
    }

    #[test]
    fn or_guard_proves() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<u32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_or_guard_proves::expand(&mut builder.scope, x, y);
        let def = builder.build(KernelSettings::default());

        let assumes = [Assume::LenEq { a: "x", b: "y" }];
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &assumes) {
            ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// `||` negative control: `pos < 1 || pos < y.len()` is *not* equivalent
    /// to `pos < y.len()` — when `y.len() == 0`, `pos == 0` satisfies the
    /// first arm and slips through, but `y[0]` is out of bounds. Correctly
    /// modeling `Or` must catch this, not silently widen what's provable.
    #[cube(launch)]
    fn prover_test_or_guard_refutes(x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS < 1usize || ABSOLUTE_POS < y.len() {
            y[ABSOLUTE_POS] = x[ABSOLUTE_POS];
        }
    }

    #[test]
    fn or_guard_insufficient_refutes() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<u32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_or_guard_refutes::expand(&mut builder.scope, x, y);
        let def = builder.build(KernelSettings::default());

        let assumes = [Assume::LenEq { a: "x", b: "y" }];
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &assumes) {
            ProveResult::Refuted { .. } => {}
            other => panic!("expected Refuted, got {other:?}"),
        }
    }

    /// `!` positive control (De Morgan: `!(pos >= len) == pos < len`).
    /// Deliberately not simplified to `pos < len` — the whole point is to
    /// exercise `Operator::Not`.
    #[cube(launch)]
    fn prover_test_not_guard_proves(x: &Array<u32>, y: &mut Array<u32>) {
        #[allow(clippy::nonminimal_bool)]
        if !(ABSOLUTE_POS >= y.len()) {
            y[ABSOLUTE_POS] = x[ABSOLUTE_POS];
        }
    }

    #[test]
    fn not_guard_proves() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<u32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_not_guard_proves::expand(&mut builder.scope, x, y);
        let def = builder.build(KernelSettings::default());

        let assumes = [Assume::LenEq { a: "x", b: "y" }];
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &assumes) {
            ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Div/mod-derived indices.
    // -----------------------------------------------------------------

    /// Positive control: `stride >= 1` (path condition) discharges the
    /// div side-obligation's "divisor nonzero" half (the "both operands
    /// nonnegative" half is automatic here — `ABSOLUTE_POS`/`stride` are
    /// both unsigned leaves, asserted nonnegative at declaration), so
    /// `ABSOLUTE_POS / stride` models as genuine SMT `div`; `ABSOLUTE_POS <
    /// x.len()` guards the `x` read directly, and `idx < y.len()` guards
    /// the `y` write.
    #[cube(launch)]
    fn prover_test_div_guarded(x: &Array<u32>, y: &mut Array<u32>, stride: usize) {
        if ABSOLUTE_POS < x.len() && stride >= 1usize {
            let idx = ABSOLUTE_POS / stride;
            if idx < y.len() {
                y[idx] = x[ABSOLUTE_POS];
            }
        }
    }

    /// Builds a `KernelDefinition` for one of the div/mod test kernels
    /// below, all of which share the same signature shape (two `u32`
    /// arrays plus one `usize` scalar named `stride`/`width`).
    macro_rules! build_div_mod_kernel {
        ($kernel:path) => {{
            let mut builder = KernelBuilder::default();
            builder.runtime_properties(Default::default());
            cubecl::ir::AddressType::U32.register(&mut builder.scope);
            let x =
                <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
            let y = <Array<u32> as LaunchArg>::expand_output(
                &ArrayCompilationArg { inplace: None },
                &mut builder,
            );
            let stride = <usize as LaunchArg>::expand(&Default::default(), &mut builder);
            $kernel(&mut builder.scope, x, y, stride);
            builder.build(KernelSettings::default())
        }};
    }

    #[test]
    fn div_guarded_proves() {
        let def = build_div_mod_kernel!(prover_test_div_guarded::expand);
        // No length assume needed: each buffer's obligation is guarded
        // directly against its own `.len()`.
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[]) {
            // guarded x[pos]/y[idx] read+write.
            ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// Negative (taint) control: with no guard establishing `stride != 0`,
    /// the div side-obligation cannot discharge (`stride == 0` is
    /// SAT-reachable), so `idx` is left tainted per the taint discipline —
    /// and the `if idx < y.len()` branch that then depends on it fails
    /// explicitly as `OutOfSubset`, not `Proved`.
    #[cube(launch)]
    fn prover_test_div_unguarded(x: &Array<u32>, y: &mut Array<u32>, stride: usize) {
        let idx = ABSOLUTE_POS / stride;
        if idx < y.len() {
            y[idx] = x[ABSOLUTE_POS];
        }
    }

    #[test]
    fn div_unguarded_divisor_is_out_of_subset() {
        let def = build_div_mod_kernel!(prover_test_div_unguarded::expand);
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[]) {
            ProveResult::OutOfSubset { reason } => {
                assert!(reason.contains("if"), "unexpected reason: {reason}");
            }
            other => panic!("expected OutOfSubset, got {other:?}"),
        }
    }

    /// Negative (refute) control — the task's "genuinely-unsafe decode"
    /// shape: `stride >= 1` discharges the div side-obligation (so `idx`
    /// *does* get modeled, unlike the taint control above) and the `x` read
    /// is separately guarded (`ABSOLUTE_POS < x.len()`) so it isn't what
    /// refutes — but nothing relates `x.len()`/`y.len()`, so `idx` (bounded
    /// only by `< x.len()`) can still exceed `y.len()`. The checker must
    /// find that real counterexample, not vacuously pass because the
    /// divisor guard "looks like" a bounds guard.
    #[cube(launch)]
    fn prover_test_div_index_unbounded(x: &Array<u32>, y: &mut Array<u32>, stride: usize) {
        if ABSOLUTE_POS < x.len() && stride >= 1usize {
            let idx = ABSOLUTE_POS / stride;
            y[idx] = x[ABSOLUTE_POS];
        }
    }

    #[test]
    fn div_index_unbounded_refutes() {
        let def = build_div_mod_kernel!(prover_test_div_index_unbounded::expand);
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[]) {
            // Specifically the `y[idx]` write, not the (separately guarded)
            // `x[ABSOLUTE_POS]` read — confirms the refutation is about the
            // div-derived index exceeding `y.len()`, not an unrelated bug.
            ProveResult::Refuted { obligation, .. } => {
                assert!(obligation.contains('y'), "unexpected obligation: {obligation}");
            }
            other => panic!("expected Refuted, got {other:?}"),
        }
    }

    /// `%` positive control: `ABSOLUTE_POS < x.len()` guards the `x` read
    /// directly; `width <= y.len()` plus the div/mod theory's own `0 <=
    /// mod(a,b) < b` (for `b > 0`) fact together prove `ABSOLUTE_POS %
    /// width < y.len()` for the `y` write, without any further guard.
    #[cube(launch)]
    fn prover_test_mod_guarded(x: &Array<u32>, y: &mut Array<u32>, width: usize) {
        if ABSOLUTE_POS < x.len() && width >= 1usize && width <= y.len() {
            let idx = ABSOLUTE_POS % width;
            y[idx] = x[ABSOLUTE_POS];
        }
    }

    #[test]
    fn mod_guarded_proves() {
        let def = build_div_mod_kernel!(prover_test_mod_guarded::expand);
        // No length assume needed: the `x` read is guarded directly, and
        // the `y` write is bounded by `width <= y.len()` plus the mod
        // theory's own range fact.
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[]) {
            ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Shared-memory milestone M1 — cooperative leaves + shared arrays.
    // (docs/design-shared-memory.md §8 M1.)
    // -----------------------------------------------------------------

    /// The loop-free portion of `block_sum_reduce`: the guarded shared load
    /// plus the single-writer partial store, with the tree-reduction *loop*
    /// omitted (that loop is cooperative — it carries a `sync_cube` — so it
    /// defers to the M3 race walker; see `block_sum_reduce_defers_to_m3`).
    /// This is the M1 positive control: it exercises every M1 mechanism —
    /// `UnitPos`/`CubePos`/`AbsolutePos` cooperative leaves and a
    /// `SharedMemory` tile bounded by its compile-time length — and proves.
    #[cube(launch)]
    fn prover_test_shared_load_guarded(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        } else {
            tile[tid] = 0.0f32;
        }
        sync_cube();
        if CUBE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    fn build_shared_load_guarded() -> KernelDefinition {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let input =
            <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let output = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_shared_load_guarded::expand(&mut builder.scope, input, output);
        builder.build(KernelSettings::default())
    }

    const SHARED_BUFFERS: &[BufferParam] = &[
        BufferParam { name: "input", is_output: false },
        BufferParam { name: "output", is_output: true },
    ];

    /// M1 positive control (§8): a single-thread symbolic pass over the
    /// loop-free reduction shape `Proved` with `CUBE_DIM = 256`. Obligations:
    /// `input[ABSOLUTE_POS]` read (guarded), `tile[UnitPos]` write (if arm),
    /// `tile[UnitPos]` write (else arm), `tile[0]` read, `output[CUBE_POS]`
    /// write (guarded) — five, all in bounds: `UnitPos < 256 == tile length`,
    /// `ABSOLUTE_POS`/`CUBE_POS` guarded against their buffers.
    #[test]
    fn cooperative_shared_load_proves() {
        let def = build_shared_load_guarded();
        match prove_bounds_freedom_cooperative(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::Proved { obligations } => assert_eq!(obligations, 5),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// Cooperative gating control: the *same* kernel run through the plain
    /// (non-cooperative) entry point is `OutOfSubset`, because without a
    /// pinned `CUBE_DIM` the `tile[UnitPos]` write index is unmodeled — the
    /// shared-array bound machinery is active either way, but only cooperative
    /// mode makes `UnitPos` modeled enough to discharge it. Confirms the M1
    /// leaf modeling is genuinely gated on `coop`, not leaking into the plain
    /// walk (whose behavior must stay byte-identical).
    #[test]
    fn shared_load_without_cooperative_is_out_of_subset() {
        let def = build_shared_load_guarded();
        match prove_bounds_freedom(&def, SHARED_BUFFERS, &[]) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("write index") && reason.contains("shared_array"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (UnitPos unmodeled without cube_dim), got {other:?}"),
        }
    }

    /// M1 negative control (§8): an undersized tile — `SharedMemory::new(128)`
    /// launched at `CUBE_DIM = 256` — is a genuine OOB shared store, since
    /// `tile[UnitPos]` with `UnitPos` up to `255` exceeds the 128-element tile
    /// (§7.1's `cube_dim <= SharedMemory::new(N)` check, violated). The
    /// checker must `Refuted` with a counterexample exhibiting `unit_pos >=
    /// 128`, not vacuously prove.
    #[cube(launch)]
    fn prover_test_shared_undersized_tile(output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(128usize);
        tile[tid] = 1.0f32;
        if tid == 0usize {
            output[CUBE_POS] = tile[0usize];
        }
    }

    #[test]
    fn cooperative_undersized_tile_refutes() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let output = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_shared_undersized_tile::expand(&mut builder.scope, output);
        let def = builder.build(KernelSettings::default());

        let buffers = [BufferParam { name: "output", is_output: true }];
        match prove_bounds_freedom_cooperative(&def, &buffers, &[], 256) {
            ProveResult::Refuted { obligation, counterexample } => {
                assert!(obligation.contains("shared_array"), "unexpected obligation: {obligation}");
                assert!(
                    counterexample.contains("unit_pos"),
                    "counterexample should exhibit the offending unit_pos: {counterexample}"
                );
            }
            other => panic!("expected Refuted (undersized tile), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Shared-memory milestone M2 — `Branch::Loop` recognition.
    // (docs/design-shared-memory.md §8 M2.)
    // -----------------------------------------------------------------

    /// The grid-stride accumulation phase in isolation: one non-cooperative
    /// `while` loop (no barrier inside) whose induction variable `k` starts at
    /// `ABSOLUTE_POS`, strides by `CUBE_DIM * num_cubes`, and is bounded by the
    /// guard `k < n`. `n` is a comptime constant (folded to `4096`), so
    /// `data[k]` reads discharge under `n <= data.len()`. The float
    /// accumulator `local` is carried but never indexes anything.
    #[cube(launch)]
    fn prover_test_grid_stride_accumulate(
        data: &Array<f32>,
        out: &mut Array<f32>,
        num_cubes: u32,
        #[comptime] n: usize,
    ) {
        let stride = CUBE_DIM as usize * num_cubes as usize;
        let mut k = ABSOLUTE_POS;
        let mut local = 0.0f32;
        while k < n {
            local += data[k] * data[k];
            k += stride;
        }
        if ABSOLUTE_POS < out.len() {
            out[ABSOLUTE_POS] = local;
        }
    }

    fn build_grid_stride_accumulate() -> KernelDefinition {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let data =
            <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let out = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        let num_cubes = <u32 as LaunchArg>::expand(&Default::default(), &mut builder);
        prover_test_grid_stride_accumulate::expand(&mut builder.scope, data, out, num_cubes, 4096);
        builder.build(KernelSettings::default())
    }

    /// M2 positive control (§8): the phase-0 accumulation loop's `data[k]`
    /// reads `Proved` under `n <= data.len()` — i.e. the non-cooperative
    /// `Branch::Loop` is modeled RangeLoop-style, with `k` a fresh symbol
    /// bounded by the asserted guard `k < 4096`. Obligations: `data[k]` read
    /// twice (`data[k] * data[k]`), `out[ABSOLUTE_POS]` write (guarded) —
    /// three. `data.len() == 4096` supplies `n <= data.len()`.
    #[test]
    fn noncooperative_loop_data_read_proves() {
        let def = build_grid_stride_accumulate();
        let buffers = [
            BufferParam { name: "data", is_output: false },
            BufferParam { name: "out", is_output: true },
        ];
        match prove_bounds_freedom(&def, &buffers, &[Assume::LenEqConst { a: "data", value: 4096 }]) {
            ProveResult::Proved { obligations } => assert_eq!(obligations, 3),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// Without `n <= data.len()`, the same loop's `data[k]` read `Refuted`:
    /// `k` (bounded only by `k < 4096`) can exceed an unconstrained
    /// `data.len()`. Confirms the guard-derived bound is the *only* thing
    /// making the positive control prove — the loop model doesn't vacuously
    /// pass.
    #[test]
    fn noncooperative_loop_without_len_assume_refutes() {
        let def = build_grid_stride_accumulate();
        let buffers = [
            BufferParam { name: "data", is_output: false },
            BufferParam { name: "out", is_output: true },
        ];
        match prove_bounds_freedom(&def, &buffers, &[]) {
            ProveResult::Refuted { obligation, .. } => {
                assert!(obligation.contains("data"), "unexpected obligation: {obligation}");
            }
            other => panic!("expected Refuted, got {other:?}"),
        }
    }

    /// M2 negative control (§8): a `Branch::Loop` with a *trailing* break
    /// (`loop { body; if c { break } }`) has no leading break-guard, so it is
    /// not the recognized `while` shape — it stays `OutOfSubset` with the
    /// pre-milestone "unbounded/break-terminated loop" reason, unchanged. A
    /// bare unbounded loop is never modeled.
    #[cube(launch)]
    fn prover_test_bare_loop_trailing_break(x: &Array<u32>, y: &mut Array<u32>) {
        loop {
            y[ABSOLUTE_POS] = x[ABSOLUTE_POS];
            if ABSOLUTE_POS == 0usize {
                break;
            }
        }
    }

    #[test]
    fn bare_loop_trailing_break_is_out_of_subset() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<u32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_bare_loop_trailing_break::expand(&mut builder.scope, x, y);
        let def = builder.build(KernelSettings::default());

        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[Assume::LenEq { a: "x", b: "y" }]) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("unbounded/break-terminated"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (no leading break-guard), got {other:?}"),
        }
    }

    /// Honesty control for the induction-variable restriction: a *descending*
    /// loop (`while k > 0 { …k…; k -= 1 }`) has a leading break-guard and no
    /// barrier, so it is recognized and non-cooperative — but its guard `k >
    /// 0` bounds `k` from *below*, not above. `k` is therefore NOT promoted to
    /// an induction symbol (which, bounded only from below, could manufacture
    /// a spurious `Refuted` on a safe loop); it stays tainted, and the `x[k]`
    /// read fails cleanly as `OutOfSubset` at the use site. Never a spurious
    /// `Refuted`, never a `Proved`.
    #[cube(launch)]
    fn prover_test_descending_loop(x: &Array<u32>, y: &mut Array<u32>) {
        let mut k = ABSOLUTE_POS;
        while k > 0usize {
            y[k] = x[k];
            k -= 1usize;
        }
    }

    #[test]
    fn descending_loop_is_out_of_subset_not_refuted() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<u32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_descending_loop::expand(&mut builder.scope, x, y);
        let def = builder.build(KernelSettings::default());

        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[Assume::LenEq { a: "x", b: "y" }]) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("depends on a construct outside"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (descending induction not promoted), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // The two probe reduction kernels, whole. Under M1+M2 both reach their
    // cooperative tree-reduction loop (a `Branch::Loop` carrying `sync_cube`)
    // and defer to the M3 race walker — nothing past a barrier-carrying loop
    // is modeled by a single-thread bounds pass. (These are the exact
    // clean-room probe kernels from the design's scratchpad.)
    // -----------------------------------------------------------------

    #[cube(launch)]
    fn prover_test_block_sum_reduce(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        } else {
            tile[tid] = 0.0f32;
        }
        sync_cube();

        let mut half = CUBE_DIM as usize / 2;
        while half > 0usize {
            if tid < half {
                let a = tile[tid];
                let b = tile[tid + half];
                tile[tid] = a + b;
            }
            sync_cube();
            half /= 2usize;
        }

        if tid == 0usize {
            output[CUBE_POS] = tile[0usize];
        }
    }

    /// `block_sum_reduce`, whole, in cooperative mode: the guarded load and
    /// its shared writes are modeled (M1), the top-level barrier is a no-op,
    /// but the tree-reduction `while` loop carries a `sync_cube` — cooperative
    /// — so the walk defers to M3 rather than mismodeling it. This is why the
    /// §8 M1 phrase "block_sum_reduce Proved" is, under the M2 decision to
    /// defer cooperative loops, satisfiable only for the loop-free portion
    /// (`cooperative_shared_load_proves`); the whole kernel correctly stops at
    /// the barrier-carrying loop.
    #[test]
    fn block_sum_reduce_defers_to_m3() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let input =
            <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let output = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_block_sum_reduce::expand(&mut builder.scope, input, output);
        let def = builder.build(KernelSettings::default());

        match prove_bounds_freedom_cooperative(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("milestone M3") && reason.contains("cooperative loop"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (cooperative loop deferred to M3), got {other:?}"),
        }
    }

    #[cube(launch)]
    fn prover_test_grid_stride_reduce(
        data: &Array<f32>,
        partials: &mut Array<f32>,
        num_cubes: u32,
        #[comptime] n: usize,
    ) {
        let tid = UNIT_POS as usize;
        let stride = CUBE_DIM as usize * num_cubes as usize;
        let mut k = ABSOLUTE_POS;
        let mut local = 0.0f32;
        while k < n {
            local += data[k] * data[k];
            k += stride;
        }

        let mut tile = SharedMemory::<f32>::new(256usize);
        tile[tid] = local;
        sync_cube();

        let mut half = CUBE_DIM as usize / 2;
        while half > 0usize {
            if tid < half {
                let a = tile[tid];
                let b = tile[tid + half];
                tile[tid] = a + b;
            }
            sync_cube();
            half /= 2usize;
        }

        if tid == 0usize {
            partials[CUBE_POS] = tile[0usize];
        }
    }

    /// `grid_stride_reduce`, whole, in cooperative mode with `data.len() ==
    /// 4096`: the non-cooperative accumulation loop is modeled and its
    /// `data[k]` reads discharge (M2), the `tile[UnitPos]` store proves (M1),
    /// but the subsequent tree-reduction `while` loop carries `sync_cube` —
    /// cooperative — so the walk defers to M3. Demonstrates the two loop
    /// shapes side by side in one kernel: the first modeled, the second
    /// deferred.
    #[test]
    fn grid_stride_reduce_defers_to_m3() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let data =
            <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let partials = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        let num_cubes = <u32 as LaunchArg>::expand(&Default::default(), &mut builder);
        prover_test_grid_stride_reduce::expand(&mut builder.scope, data, partials, num_cubes, 4096);
        let def = builder.build(KernelSettings::default());

        let buffers = [
            BufferParam { name: "data", is_output: false },
            BufferParam { name: "partials", is_output: true },
        ];
        match prove_bounds_freedom_cooperative(
            &def,
            &buffers,
            &[Assume::LenEqConst { a: "data", value: 4096 }],
            256,
        ) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("milestone M3") && reason.contains("cooperative loop"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (tree loop deferred to M3), got {other:?}"),
        }
    }
}
