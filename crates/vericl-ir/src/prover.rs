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
//!   represent the accumulated value at iteration `k`. `check_no_loop_carry`
//!   rejects any loop that reassigns a variable bound outside the loop body,
//!   closing that gap rather than silently mismodeling it.
//! - The ascending-bounds model above assumes unit stride. `RangeLoop.step`
//!   (`Some(_)` for `range_stepped`, e.g. a descending loop where
//!   `start > end` numerically) is never modeled: asserting `start <= i <
//!   end` for a genuinely descending range makes the SMT context infeasible,
//!   which would make every obligation inside the loop vacuously "provable"
//!   (UNSAT-under-contradiction, not UNSAT-because-safe). `process_range_loop`
//!   therefore rejects any loop with `step.is_some()` outright, before
//!   asserting bounds, rather than silently mismodeling it.

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
            self.memo.insert(out.kind, val);
        }
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
            // And/Or/Not/Select/InitVector/CopyMemory* etc: not needed by
            // the v0 subset (booleans built purely from Comparison suffice
            // for every if-guard in the supported examples); leave tainted.
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

        // Soundness guard (see module docs): a loop that reassigns a
        // variable already bound outside it is loop-carried state
        // (an accumulator), which a single symbolic pass over the body
        // cannot soundly represent.
        let outer: HashSet<VariableKind> = self.memo.keys().copied().collect();
        if let Some(carried) = scope_reassigns_any(&rl.scope, &outer) {
            return Err(Stop::OutOfSubset(format!(
                "loop body reassigns `{carried:?}`, which is defined outside the loop — \
                 loop-carried state (e.g. accumulators) is outside the vericl v0 subset"
            )));
        }

        let start = self.value_of(&rl.start).ok_or_else(|| {
            Stop::OutOfSubset("range-loop start bound depends on a construct outside the vericl v0 subset".into())
        })?;
        let end = self.value_of(&rl.end).ok_or_else(|| {
            Stop::OutOfSubset("range-loop end bound depends on a construct outside the vericl v0 subset".into())
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

/// Does `scope` (recursively, through nested branches) reassign any variable
/// whose `VariableKind` is in `outer`? Used to reject loop-carried mutation
/// (see `process_range_loop`).
fn scope_reassigns_any(scope: &Scope, outer: &HashSet<VariableKind>) -> Option<VariableKind> {
    for inst in &scope.instructions {
        if let Some(out) = inst.out {
            if outer.contains(&out.kind) {
                return Some(out.kind);
            }
        }
        if let Operation::Branch(b) = &inst.operation {
            let found = match b {
                Branch::If(if_) => scope_reassigns_any(&if_.scope, outer),
                Branch::IfElse(ie) => scope_reassigns_any(&ie.scope_if, outer)
                    .or_else(|| scope_reassigns_any(&ie.scope_else, outer)),
                Branch::Switch(sw) => scope_reassigns_any(&sw.scope_default, outer)
                    .or_else(|| sw.cases.iter().find_map(|(_, s)| scope_reassigns_any(s, outer))),
                Branch::RangeLoop(rl) => scope_reassigns_any(&rl.scope, outer),
                Branch::Loop(l) => scope_reassigns_any(&l.scope, outer),
                Branch::Return | Branch::Break | Branch::Unreachable => None,
            };
            if found.is_some() {
                return found;
            }
        }
    }
    None
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

    /// Soundness guard: a loop that accumulates into a variable defined
    /// outside the loop is loop-carried state a single symbolic pass cannot
    /// soundly represent (see module docs) — must be rejected, not silently
    /// (mis-)modeled as if the loop ran once.
    #[test]
    fn loop_carried_accumulator_is_out_of_subset() {
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
                assert!(reason.contains("loop-carried"), "unexpected reason: {reason}");
            }
            other => panic!("expected OutOfSubset, got {other:?}"),
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
}
