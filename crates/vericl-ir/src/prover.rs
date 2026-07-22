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

use std::collections::{HashMap, HashSet};

use cubecl::ir::{
    Arithmetic, Branch, Comparison, ConstantValue, Id, Instruction, Metadata, Operation, Operator,
    Scope, Type, Variable, VariableKind,
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
    };

    if let Err(e) = prover.assert_structured_assumes(assumes) {
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

    fn taint_out(&mut self, inst: &Instruction) {
        if let Some(out) = inst.out {
            self.memo.insert(out.kind, None);
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
            self.memo.insert(out.kind, val);
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

    fn buffer_of(&self, list: &Variable) -> Result<(Id, bool), Stop> {
        match list.kind {
            VariableKind::GlobalInputArray(id) => Ok((id, false)),
            VariableKind::GlobalOutputArray(id) => Ok((id, true)),
            other => Err(Stop::OutOfSubset(format!(
                "indexing into `{other:?}` (not a global input/output array) is outside the \
                 vericl v0 subset"
            ))),
        }
    }

    fn process_index(
        &mut self,
        inst: &Instruction,
        io: &cubecl::ir::IndexOperator,
        list: Variable,
    ) -> Result<(), Stop> {
        self.check_trivial_vectorization(io.vector_size, io.unroll_factor)?;
        let (buf_id, _is_output) = self.buffer_of(&list)?;
        let idx = self.value_of(&io.index).ok_or_else(|| {
            Stop::OutOfSubset(format!(
                "read index for `{}[...]` depends on a construct outside the vericl v0 subset",
                self.buffer_name(buf_id)
            ))
        })?;
        self.emit_obligation(buf_id, idx, "read")?;
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
        let (buf_id, _is_output) = self.buffer_of(&list)?;
        let idx = self.value_of(&io.index).ok_or_else(|| {
            Stop::OutOfSubset(format!(
                "write index for `{}[...] = ...` depends on a construct outside the vericl v0 \
                 subset",
                self.buffer_name(buf_id)
            ))
        })?;
        self.emit_obligation(buf_id, idx, "write")?;
        self.taint_out(inst);
        Ok(())
    }

    fn emit_obligation(&mut self, buf_id: Id, idx: SExpr, kind: &str) -> Result<(), Stop> {
        let len = self.length_of(buf_id)?;
        let zero = self.smt.numeral(0);
        let ge0 = self.smt.gte(idx, zero);
        let lt_len = self.smt.lt(idx, len);
        let in_bounds = self.smt.and(ge0, lt_len);
        let description = format!(
            "0 <= index < {}.len() ({kind} access to `{}`)",
            self.buffer_name(buf_id),
            self.buffer_name(buf_id)
        );
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
            Branch::If(if_) => {
                let cond = self.cond_of(&if_.cond, "if")?;
                self.smt.push().map_err(smt_err)?;
                self.smt.assert(cond).map_err(smt_err)?;
                let r = self.process_scope(&if_.scope);
                self.smt.pop().map_err(smt_err)?;
                r
            }
            Branch::IfElse(ie) => {
                let cond = self.cond_of(&ie.cond, "if/else")?;
                self.smt.push().map_err(smt_err)?;
                self.smt.assert(cond).map_err(smt_err)?;
                let r1 = self.process_scope(&ie.scope_if);
                self.smt.pop().map_err(smt_err)?;
                r1?;

                let not_cond = self.smt.not(cond);
                self.smt.push().map_err(smt_err)?;
                self.smt.assert(not_cond).map_err(smt_err)?;
                let r2 = self.process_scope(&ie.scope_else);
                self.smt.pop().map_err(smt_err)?;
                r2
            }
            Branch::RangeLoop(rl) => self.process_range_loop(rl),
            Branch::Loop(_) => Err(Stop::OutOfSubset(
                "`Branch::Loop` (unbounded/break-terminated loop) is outside the vericl v0 subset"
                    .into(),
            )),
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
            self.memo.insert(k, None);
        }
        self.carried_stack.push(carried.clone());

        let r = self.process_range_loop_body(rl);

        self.carried_stack.pop();
        // Defensive: `bind_out`/`taint_out` already guarantee every carried
        // key is `None` by now (any write to it during the walk was forced
        // tainted), but re-asserting it here makes "and after the loop" an
        // explicit invariant rather than one that merely happens to hold.
        for &k in &carried {
            self.memo.insert(k, None);
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
            VariableKind::Builtin(cubecl::ir::Builtin::AbsolutePos) => {
                self.declare_int("abs_pos", true).ok()
            }
            VariableKind::GlobalScalar(id) => {
                if is_modeled_int(&var.ty) {
                    self.declare_int(&format!("scalar{id}_"), is_unsigned(&var.ty)).ok()
                } else {
                    None
                }
            }
            // Locals not yet bound by a modeled instruction, arrays used as
            // scalar values, and every other builtin: unsupported here.
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
}
