//! Seeded fuzz cross-check: random in-subset kernels, three independent
//! semantics implementations checked against each other.
//!
//! For each seed the generator produces a random kernel from a small grammar
//! over the modeled constructs (guarded indexing, arithmetic/bitwise chains,
//! div/mod by a nonzero divisor, `if`/`if-else`, `match`/switch, bounded loops,
//! gathers through a valid offset table). Each kernel is realized two ways from
//! the *same* generated AST:
//!
//! * **lowered to CubeCL IR** (`lower_*`), hand-built with the same instruction
//!   shapes `#[cube]` emits (`Index`/`IndexAssign`, `Arithmetic`, `Bitwise`,
//!   `Comparison`, `Operator::And/Or`, `Branch::{If,IfElse,Switch,RangeLoop}`,
//!   `Metadata::Length`), then run through the concrete [`interpret_dispatch`]
//!   interpreter and the symbolic [`prove_bounds_freedom`] prover;
//! * **evaluated directly** (`eval_kernel`) by an independent tree-walking
//!   reference over the AST that never touches the IR.
//!
//! Two cross-checks per kernel:
//!
//! * **(a) reference ≡ interpreter** on many random inputs (valid and
//!   adversarial): the AST reference and the IR interpreter must produce the
//!   same output buffer, or report the same out-of-bounds access. A mismatch is
//!   a [`Finding`] (interpreter or lowering defect) — never silently
//!   reconciled.
//! * **(b) prover ⇄ interpreter**: if the prover says `Proved`, no
//!   assume-satisfying input may drive the interpreter out of bounds (an OOB
//!   here is a **critical** model-fidelity finding — the prover certified
//!   something the concrete semantics violates). If the prover says `Refuted`,
//!   its counterexample, replayed through the interpreter, must exhibit the OOB.
//!
//! The generated IR is *hand-built to mirror* CubeCL's shapes rather than
//! emitted by `#[cube]` (random structure cannot be macro-expanded at runtime);
//! fidelity to *real* CubeCL IR is anchored separately by the `interp.rs` unit
//! tests and the public-example cross-check, both of which run the interpreter
//! over genuine `#[cube]`-expanded `KernelDefinition`s. See `docs/interpreter.md`.

use cubecl::ir::{
    AddressType, Arithmetic, BinaryOperator, Bitwise, Branch, Builtin, Comparison, ConstantValue,
    ElemType, If, IndexAssignOperator, IndexOperator, Instruction, Metadata, Operation, Operator,
    RangeLoop, StorageType, Switch, Type, UIntKind, Variable,
};
use cubecl::prelude::{KernelBuilder, KernelDefinition, KernelSettings};

use crate::interp::{Buffer, Inputs, Outcome, ScalarBinding, interpret_dispatch};
use crate::prover::{Assume, BufferParam, ProveResult, prove_bounds_freedom};

// =====================================================================
// A tiny dependency-free SplitMix64 (vericl-ir has no rng dependency).
// =====================================================================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x1234_5678_9abc_def0)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
    fn u32(&mut self) -> u32 {
        self.next() as u32
    }
    fn choice(&mut self, n: u32) -> u32 {
        self.below(n)
    }
}

// =====================================================================
// Grammar
// =====================================================================

#[derive(Clone, Copy, Debug)]
enum BinOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

/// A pure value expression over `{ABSOLUTE_POS, loop var, scalars, constants}`
/// — never contains an array access, so evaluating it can only produce a value
/// (division/modulo use a baked nonzero constant divisor).
#[derive(Clone, Debug)]
enum Expr {
    Pos,
    LoopVar,
    Scalar(u8),
    Const(u32),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    /// `lhs / d` or `lhs % d` with `d != 0`.
    DivMod { is_div: bool, lhs: Box<Expr>, d: u32 },
}

/// The kernel body shape. Every shape guards its accesses with
/// `ABSOLUTE_POS < y.len()` (or a deliberately-broken variant), keeping the
/// prover's verdict predictable per shape.
#[derive(Clone, Debug)]
enum Shape {
    /// `if pos < y.len() { y[pos] = value }` — Proved.
    SafeDirect { value: Expr },
    /// flatten/decode: `if pos < y.len() && s0 >= 1 { y[(pos/s0)*s0 + pos%s0] = value }`
    /// (the recombined index equals `pos`) — Proved.
    SafeDivMod { value: Expr },
    /// `if pos < y.len() { y[pos] = x[idxs[pos]] }` — Proved with the length +
    /// element-range assumes.
    SafeGather,
    /// `if pos < y.len() { acc = 0; for j in 0..count { acc = acc + body }; y[pos] = acc }`
    /// — Proved (the carried accumulator never feeds an index).
    SafeLoop { count: u32, body: Expr },
    /// `if pos < y.len() { match s0 { 0 => y[pos]=v0, 1 => y[pos]=v1, _ => y[pos]=v2 } }`
    /// — Proved (every arm guarded).
    SafeSwitch { v0: Expr, v1: Expr, v2: Expr },
    /// DEFECT: `if pos <= y.len() { y[pos] = value }` — Refuted (`pos == len`).
    UnsafeOffByOne { value: Expr },
    /// DEFECT: `if pos < y.len() { y[pos] = x[pos + k] }` with no length
    /// relationship — Refuted (`x[pos + k]` overruns `x`).
    UnsafeForwardRead { k: u32 },
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Verdict {
    Proved,
    Refuted,
}

/// A fully generated kernel: its AST shape, the lowered IR, and everything the
/// prover/interpreter need (buffer order, assumes, expected verdict).
struct Kernel {
    shape: Shape,
    def: KernelDefinition,
    has_x: bool,
    has_idxs: bool,
    buffers: Vec<BufferParam<'static>>,
    assumes: Vec<Assume<'static>>,
    /// Buffer id (== registration index) of the output `y`, for reading results.
    y_id: usize,
    expected: Verdict,
}

const U32_STORAGE: StorageType = StorageType::Scalar(ElemType::UInt(UIntKind::U32));

fn u32ty() -> Type {
    Type::new(U32_STORAGE)
}
fn boolty() -> Type {
    Type::scalar(ElemType::Bool)
}
fn konst(v: u32) -> Variable {
    Variable::constant(ConstantValue::UInt(v as u64), u32ty())
}

// =====================================================================
// Generation
// =====================================================================

fn gen_expr(rng: &mut Rng, depth: u32, in_loop: bool) -> Expr {
    if depth == 0 || rng.choice(3) == 0 {
        // leaf
        let n = if in_loop { 5 } else { 4 };
        match rng.choice(n) {
            0 => Expr::Pos,
            1 => Expr::Const(rng.u32() % 64),
            2 => Expr::Scalar(0),
            3 => Expr::Scalar(1),
            _ => Expr::LoopVar,
        }
    } else if rng.choice(4) == 0 {
        // div/mod by a nonzero constant
        let d = 1 + rng.u32() % 63;
        Expr::DivMod {
            is_div: rng.choice(2) == 0,
            lhs: Box::new(gen_expr(rng, depth - 1, in_loop)),
            d,
        }
    } else {
        let op = match rng.choice(8) {
            0 => BinOp::Add,
            1 => BinOp::Sub,
            2 => BinOp::Mul,
            3 => BinOp::And,
            4 => BinOp::Or,
            5 => BinOp::Xor,
            6 => BinOp::Shl,
            _ => BinOp::Shr,
        };
        Expr::Bin(
            op,
            Box::new(gen_expr(rng, depth - 1, in_loop)),
            Box::new(gen_expr(rng, depth - 1, in_loop)),
        )
    }
}

/// Build a random kernel for `seed`.
fn gen_kernel(seed: u64) -> Kernel {
    let mut rng = Rng::new(seed);
    let shape = match rng.choice(7) {
        0 => Shape::SafeDirect { value: gen_expr(&mut rng, 3, false) },
        1 => Shape::SafeDivMod { value: gen_expr(&mut rng, 2, false) },
        2 => Shape::SafeGather,
        3 => Shape::SafeLoop { count: 1 + rng.below(6), body: gen_expr(&mut rng, 2, true) },
        4 => Shape::SafeSwitch {
            v0: gen_expr(&mut rng, 2, false),
            v1: gen_expr(&mut rng, 2, false),
            v2: gen_expr(&mut rng, 2, false),
        },
        5 => Shape::UnsafeOffByOne { value: gen_expr(&mut rng, 3, false) },
        _ => Shape::UnsafeForwardRead { k: 1 + rng.below(8) },
    };
    lower(shape)
}

/// Lower a shape to a `KernelDefinition` plus prover metadata.
fn lower(shape: Shape) -> Kernel {
    let has_x = matches!(shape, Shape::SafeGather | Shape::UnsafeForwardRead { .. });
    let has_idxs = matches!(shape, Shape::SafeGather);

    let mut b = KernelBuilder::default();
    b.runtime_properties(Default::default());
    AddressType::U32.register(&mut b.scope);

    // Buffer registration order (== id): [x?, idxs?, y].
    let mut next_id = 0usize;
    let x_var = if has_x {
        let v = *b.input_array(u32ty());
        next_id += 1;
        Some(v)
    } else {
        None
    };
    let idxs_var = if has_idxs {
        let v = *b.input_array(u32ty());
        next_id += 1;
        Some(v)
    } else {
        None
    };
    let y_var = *b.output_array(u32ty());
    let y_id = next_id;

    // Two u32 scalars, always registered (unused ones are harmless).
    let s0 = *b.scalar(U32_STORAGE);
    let s1 = *b.scalar(U32_STORAGE);
    let scalars = [s0, s1];

    let pos = Variable::builtin(Builtin::AbsolutePos, U32_STORAGE);

    let mut buffers: Vec<BufferParam<'static>> = Vec::new();
    if has_x {
        buffers.push(BufferParam { name: "x", is_output: false });
    }
    if has_idxs {
        buffers.push(BufferParam { name: "idxs", is_output: false });
    }
    buffers.push(BufferParam { name: "y", is_output: true });

    let mut assumes: Vec<Assume<'static>> = Vec::new();
    let expected;

    match &shape {
        Shape::SafeDirect { value } => {
            expected = Verdict::Proved;
            let mut body = b.scope.child();
            let val = lower_expr(&mut body, value, pos, None, &scalars);
            index_assign(&mut body, y_var, pos, val);
            emit_guarded_if(&mut b.scope, pos, y_var, body, false);
        }
        Shape::SafeDivMod { value } => {
            expected = Verdict::Proved;
            let mut body = b.scope.child();
            // idx = (pos / s0) * s0 + (pos % s0)  == pos, in bounds.
            let row = arith(&mut body, Arithmetic::Div(bin(pos, s0)));
            let col = arith(&mut body, Arithmetic::Modulo(bin(pos, s0)));
            let rw = arith(&mut body, Arithmetic::Mul(bin(row, s0)));
            let idx = arith(&mut body, Arithmetic::Add(bin(rw, col)));
            let val = lower_expr(&mut body, value, pos, None, &scalars);
            index_assign(&mut body, y_var, idx, val);
            // Guard: pos < y.len() && s0 >= 1.
            emit_guarded_if_extra(&mut b.scope, pos, y_var, body, s0);
        }
        Shape::SafeGather => {
            expected = Verdict::Proved;
            assumes.push(Assume::LenEq { a: "idxs", b: "y" });
            assumes.push(Assume::ElemsBelowLen { arr: "idxs", len_of: "x" });
            let mut body = b.scope.child();
            let off = index_read(&mut body, idxs_var.unwrap(), pos);
            let val = index_read(&mut body, x_var.unwrap(), off);
            index_assign(&mut body, y_var, pos, val);
            emit_guarded_if(&mut b.scope, pos, y_var, body, false);
        }
        Shape::SafeLoop { count, body: loop_body } => {
            expected = Verdict::Proved;
            let mut body = b.scope.child();
            // acc = 0
            let acc = *body.create_local_restricted(u32ty());
            body.register(Instruction::new(Operation::Copy(konst(0)), acc));
            // for j in 0..count { acc = acc + loop_body }
            let j = *body.create_local_restricted(u32ty());
            let mut lb = body.child();
            let contrib = lower_expr(&mut lb, loop_body, pos, Some(j), &scalars);
            let sum = arith(&mut lb, Arithmetic::Add(bin(acc, contrib)));
            lb.register(Instruction::new(Operation::Copy(sum), acc));
            body.register(Instruction::from(Branch::RangeLoop(Box::new(RangeLoop {
                i: j,
                start: konst(0),
                end: konst(*count),
                step: None,
                inclusive: false,
                scope: lb,
            }))));
            index_assign(&mut body, y_var, pos, acc);
            emit_guarded_if(&mut b.scope, pos, y_var, body, false);
        }
        Shape::SafeSwitch { v0, v1, v2 } => {
            expected = Verdict::Proved;
            let mut body = b.scope.child();
            let arm = |val: &Expr, parent: &mut cubecl::ir::Scope| {
                let mut s = parent.child();
                let v = lower_expr(&mut s, val, pos, None, &scalars);
                index_assign(&mut s, y_var, pos, v);
                s
            };
            let s0scope = arm(v0, &mut body);
            let s1scope = arm(v1, &mut body);
            let default = arm(v2, &mut body);
            body.register(Instruction::from(Branch::Switch(Box::new(Switch {
                value: s0,
                scope_default: default,
                cases: vec![(konst(0), s0scope), (konst(1), s1scope)],
            }))));
            emit_guarded_if(&mut b.scope, pos, y_var, body, false);
        }
        Shape::UnsafeOffByOne { value } => {
            expected = Verdict::Refuted;
            let mut body = b.scope.child();
            let val = lower_expr(&mut body, value, pos, None, &scalars);
            index_assign(&mut body, y_var, pos, val);
            // Guard: pos <= y.len()  (the off-by-one).
            emit_guarded_if(&mut b.scope, pos, y_var, body, true);
        }
        Shape::UnsafeForwardRead { k } => {
            expected = Verdict::Refuted;
            // LenEq lets x[pos] prove, isolating the refutation to x[pos + k].
            assumes.push(Assume::LenEq { a: "x", b: "y" });
            let mut body = b.scope.child();
            let idx = arith(&mut body, Arithmetic::Add(bin(pos, konst(*k))));
            let val = index_read(&mut body, x_var.unwrap(), idx);
            index_assign(&mut body, y_var, pos, val);
            emit_guarded_if(&mut b.scope, pos, y_var, body, false);
        }
    }

    let def = b.build(KernelSettings::default());
    Kernel { shape, def, has_x, has_idxs, buffers, assumes, y_id, expected }
}

fn bin(lhs: Variable, rhs: Variable) -> BinaryOperator {
    BinaryOperator { lhs, rhs }
}

/// Register an arithmetic op into `scope`, returning a fresh u32 local holding
/// the result.
fn arith(scope: &mut cubecl::ir::Scope, a: Arithmetic) -> Variable {
    let out = *scope.create_local_restricted(u32ty());
    scope.register(Instruction::new(Operation::Arithmetic(a), out));
    out
}

fn lower_expr(
    scope: &mut cubecl::ir::Scope,
    e: &Expr,
    pos: Variable,
    loopvar: Option<Variable>,
    scalars: &[Variable; 2],
) -> Variable {
    match e {
        Expr::Pos => pos,
        Expr::LoopVar => loopvar.expect("loop var outside a loop"),
        Expr::Scalar(i) => scalars[*i as usize],
        Expr::Const(v) => konst(*v),
        Expr::Bin(op, a, b) => {
            let av = lower_expr(scope, a, pos, loopvar, scalars);
            let bv = lower_expr(scope, b, pos, loopvar, scalars);
            let out = *scope.create_local_restricted(u32ty());
            let operation: Operation = match op {
                BinOp::Add => Arithmetic::Add(bin(av, bv)).into(),
                BinOp::Sub => Arithmetic::Sub(bin(av, bv)).into(),
                BinOp::Mul => Arithmetic::Mul(bin(av, bv)).into(),
                BinOp::And => Bitwise::BitwiseAnd(bin(av, bv)).into(),
                BinOp::Or => Bitwise::BitwiseOr(bin(av, bv)).into(),
                BinOp::Xor => Bitwise::BitwiseXor(bin(av, bv)).into(),
                BinOp::Shl => Bitwise::ShiftLeft(bin(av, bv)).into(),
                BinOp::Shr => Bitwise::ShiftRight(bin(av, bv)).into(),
            };
            scope.register(Instruction::new(operation, out));
            out
        }
        Expr::DivMod { is_div, lhs, d } => {
            let lv = lower_expr(scope, lhs, pos, loopvar, scalars);
            let dv = konst(*d);
            let a = if *is_div {
                Arithmetic::Div(bin(lv, dv))
            } else {
                Arithmetic::Modulo(bin(lv, dv))
            };
            arith(scope, a)
        }
    }
}

fn index_read(scope: &mut cubecl::ir::Scope, list: Variable, index: Variable) -> Variable {
    let out = *scope.create_local_restricted(u32ty());
    scope.register(Instruction::new(
        Operation::Operator(Operator::Index(IndexOperator {
            list,
            index,
            vector_size: 1,
            unroll_factor: 1,
        })),
        out,
    ));
    out
}

fn index_assign(scope: &mut cubecl::ir::Scope, array: Variable, index: Variable, value: Variable) {
    scope.register(Instruction::new(
        Operation::Operator(Operator::IndexAssign(IndexAssignOperator {
            index,
            value,
            vector_size: 1,
            unroll_factor: 1,
        })),
        array,
    ));
}

/// Emit `if pos {<|<=} y.len() { body }` into `parent`.
fn emit_guarded_if(
    parent: &mut cubecl::ir::Scope,
    pos: Variable,
    y: Variable,
    body: cubecl::ir::Scope,
    inclusive: bool,
) {
    let len = *parent.create_local_restricted(u32ty());
    parent.register(Instruction::new(Operation::Metadata(Metadata::Length { var: y }), len));
    let cond = *parent.create_local_restricted(boolty());
    let cmp = if inclusive {
        Comparison::LowerEqual(bin(pos, len))
    } else {
        Comparison::Lower(bin(pos, len))
    };
    parent.register(Instruction::new(Operation::Comparison(cmp), cond));
    parent.register(Instruction::from(Branch::If(Box::new(If { cond, scope: body }))));
}

/// Emit `if pos < y.len() && s0 >= 1 { body }` into `parent`.
fn emit_guarded_if_extra(
    parent: &mut cubecl::ir::Scope,
    pos: Variable,
    y: Variable,
    body: cubecl::ir::Scope,
    s0: Variable,
) {
    let len = *parent.create_local_restricted(u32ty());
    parent.register(Instruction::new(Operation::Metadata(Metadata::Length { var: y }), len));
    let c1 = *parent.create_local_restricted(boolty());
    parent.register(Instruction::new(Operation::Comparison(Comparison::Lower(bin(pos, len))), c1));
    let c2 = *parent.create_local_restricted(boolty());
    parent.register(Instruction::new(
        Operation::Comparison(Comparison::GreaterEqual(bin(s0, konst(1)))),
        c2,
    ));
    let cond = *parent.create_local_restricted(boolty());
    parent.register(Instruction::new(Operation::Operator(Operator::And(bin(c1, c2))), cond));
    parent.register(Instruction::from(Branch::If(Box::new(If { cond, scope: body }))));
}

// =====================================================================
// Independent AST reference evaluator (never touches the IR)
// =====================================================================

/// Outcome of a reference run, in the same vocabulary as [`Outcome`] so the two
/// can be compared directly.
#[derive(Clone, Debug, PartialEq)]
enum RefOutcome {
    Completed(Vec<u32>),
    Oob { array: &'static str, index: i128, thread: u32, write: bool },
}

struct RefInputs {
    x: Vec<u32>,
    idxs: Vec<u32>,
    y: Vec<u32>,
    s: [u32; 2],
    num_threads: u32,
}

fn eval_expr(e: &Expr, pos: u32, loopvar: u32, s: &[u32; 2]) -> u32 {
    match e {
        Expr::Pos => pos,
        Expr::LoopVar => loopvar,
        Expr::Scalar(i) => s[*i as usize],
        Expr::Const(v) => *v,
        Expr::Bin(op, a, b) => {
            let x = eval_expr(a, pos, loopvar, s);
            let y = eval_expr(b, pos, loopvar, s);
            match op {
                BinOp::Add => x.wrapping_add(y),
                BinOp::Sub => x.wrapping_sub(y),
                BinOp::Mul => x.wrapping_mul(y),
                BinOp::And => x & y,
                BinOp::Or => x | y,
                BinOp::Xor => x ^ y,
                BinOp::Shl => x.wrapping_shl(y),
                BinOp::Shr => x.wrapping_shr(y),
            }
        }
        Expr::DivMod { is_div, lhs, d } => {
            let x = eval_expr(lhs, pos, loopvar, s);
            if *is_div {
                x / *d
            } else {
                x % *d
            }
        }
    }
}

/// Run the kernel's shape directly over concrete inputs, mirroring the
/// interpreter's thread order and access order so out-of-bounds outcomes line
/// up exactly.
fn eval_kernel(shape: &Shape, inp: &mut RefInputs) -> RefOutcome {
    for pos in 0..inp.num_threads {
        if let Some(o) = eval_thread(shape, pos, inp) {
            return o;
        }
    }
    RefOutcome::Completed(inp.y.clone())
}

fn eval_thread(shape: &Shape, pos: u32, inp: &mut RefInputs) -> Option<RefOutcome> {
    let ylen = inp.y.len() as u32;
    let write = |inp: &mut RefInputs, idx: u32, v: u32| -> Option<RefOutcome> {
        if (idx as usize) < inp.y.len() {
            inp.y[idx as usize] = v;
            None
        } else {
            Some(RefOutcome::Oob { array: "y", index: idx as i128, thread: pos, write: true })
        }
    };
    match shape {
        Shape::SafeDirect { value } => {
            if pos < ylen {
                let v = eval_expr(value, pos, 0, &inp.s);
                return write(inp, pos, v);
            }
        }
        Shape::SafeDivMod { value } => {
            if pos < ylen && inp.s[0] >= 1 {
                let s0 = inp.s[0];
                let idx = (pos / s0) * s0 + (pos % s0);
                let v = eval_expr(value, pos, 0, &inp.s);
                return write(inp, idx, v);
            }
        }
        Shape::SafeGather => {
            if pos < ylen {
                // read idxs[pos]
                let off = match inp.idxs.get(pos as usize) {
                    Some(o) => *o,
                    None => {
                        return Some(RefOutcome::Oob {
                            array: "idxs",
                            index: pos as i128,
                            thread: pos,
                            write: false,
                        });
                    }
                };
                // read x[off]
                let v = match inp.x.get(off as usize) {
                    Some(v) => *v,
                    None => {
                        return Some(RefOutcome::Oob {
                            array: "x",
                            index: off as i128,
                            thread: pos,
                            write: false,
                        });
                    }
                };
                return write(inp, pos, v);
            }
        }
        Shape::SafeLoop { count, body } => {
            if pos < ylen {
                let mut acc = 0u32;
                for j in 0..*count {
                    acc = acc.wrapping_add(eval_expr(body, pos, j, &inp.s));
                }
                return write(inp, pos, acc);
            }
        }
        Shape::SafeSwitch { v0, v1, v2 } => {
            if pos < ylen {
                let v = match inp.s[0] {
                    0 => eval_expr(v0, pos, 0, &inp.s),
                    1 => eval_expr(v1, pos, 0, &inp.s),
                    _ => eval_expr(v2, pos, 0, &inp.s),
                };
                return write(inp, pos, v);
            }
        }
        Shape::UnsafeOffByOne { value } => {
            if pos <= ylen {
                let v = eval_expr(value, pos, 0, &inp.s);
                return write(inp, pos, v);
            }
        }
        Shape::UnsafeForwardRead { k } => {
            if pos < ylen {
                let idx = pos.wrapping_add(*k);
                let v = match inp.x.get(idx as usize) {
                    Some(v) => *v,
                    None => {
                        return Some(RefOutcome::Oob {
                            array: "x",
                            index: idx as i128,
                            thread: pos,
                            write: false,
                        });
                    }
                };
                return write(inp, pos, v);
            }
        }
    }
    None
}

// =====================================================================
// Interpreter ⇄ reference plumbing
// =====================================================================

fn build_interp_inputs(k: &Kernel, inp: &RefInputs) -> Inputs {
    let mut buffers = Vec::new();
    if k.has_x {
        buffers.push(Buffer::u32("x", &inp.x, false));
    }
    if k.has_idxs {
        buffers.push(Buffer::u32("idxs", &inp.idxs, false));
    }
    buffers.push(Buffer::u32("y", &inp.y, true));
    Inputs {
        buffers,
        scalars: vec![ScalarBinding::u32(0, inp.s[0]), ScalarBinding::u32(1, inp.s[1])],
        cube_dim: 64,
        num_threads: inp.num_threads,
    }
}

/// Compare an interpreter [`Outcome`] against a [`RefOutcome`]; `Ok(())` if they
/// agree, `Err(detail)` describing the disagreement otherwise.
fn compare(k: &Kernel, outcome: &Outcome, refr: &RefOutcome) -> Result<(), String> {
    match (outcome, refr) {
        (Outcome::Completed { buffers }, RefOutcome::Completed(y)) => {
            let got = buffers[k.y_id].as_u32();
            if &got == y {
                Ok(())
            } else {
                Err(format!("output mismatch: interp {got:?} vs reference {y:?}"))
            }
        }
        (Outcome::OutOfBounds(o), RefOutcome::Oob { array, index, thread, write }) => {
            if o.array == *array && o.index == *index && o.thread == *thread && o.write == *write {
                Ok(())
            } else {
                Err(format!(
                    "OOB mismatch: interp {o} vs reference {} {}[{}] thread {}",
                    if *write { "write" } else { "read" },
                    array,
                    index,
                    thread
                ))
            }
        }
        (Outcome::Unsupported { reason }, _) => {
            Err(format!("interpreter rejected an in-subset kernel as Unsupported: {reason}"))
        }
        (o, r) => Err(format!("outcome kind mismatch: interp {o:?} vs reference {r:?}")),
    }
}

// =====================================================================
// Input generators
// =====================================================================

/// Inputs that satisfy the kernel's declared assumes (used for the safe-probe:
/// on these, a `Proved` kernel must never go out of bounds).
fn gen_valid_inputs(k: &Kernel, rng: &mut Rng) -> RefInputs {
    let ylen = 1 + rng.below(8);
    let s0 = match &k.shape {
        Shape::SafeDivMod { .. } => 1 + rng.below(8), // divisor >= 1
        Shape::SafeSwitch { .. } => rng.below(4),     // exercise both cases + default
        _ => rng.u32(),
    };
    let s1 = rng.u32();
    let xlen = ylen; // LenEq for gather/forward-read
    let idxs: Vec<u32> = (0..ylen).map(|_| rng.below(xlen.max(1))).collect(); // < x.len()
    let x: Vec<u32> = (0..xlen).map(|_| rng.u32()).collect();
    let y = vec![0u32; ylen as usize];
    // Dispatch a few extra threads to exercise the guard boundary from below.
    let num_threads = ylen + rng.below(3);
    RefInputs { x, idxs, y, s: [s0, s1], num_threads }
}

/// Inputs that may violate assumes — used only for the reference≡interpreter
/// agreement check (both must agree, OOB or not), never for the safe-probe.
fn gen_adversarial_inputs(_k: &Kernel, rng: &mut Rng) -> RefInputs {
    let ylen = 1 + rng.below(8);
    let s0 = rng.below(4); // includes 0 (divmod guard off; switch arms)
    let s1 = rng.u32();
    // x/idxs lengths may be shorter than y, and offsets may be out of range.
    let xlen = rng.below(ylen + 2);
    let ilen = rng.below(ylen + 2);
    let idxs: Vec<u32> = (0..ilen).map(|_| rng.below(ylen + 4)).collect();
    let x: Vec<u32> = (0..xlen).map(|_| rng.u32()).collect();
    let y = vec![0u32; ylen as usize];
    // Include threads past the guard (to trip the off-by-one at pos == len).
    let num_threads = ylen + 1 + rng.below(3);
    RefInputs { x, idxs, y, s: [s0, s1], num_threads }
}

// =====================================================================
// Counterexample replay (Refuted ⟹ interpreter exhibits the OOB)
// =====================================================================

/// Parse `key=value, ...` integer pairs from a rendered counterexample.
fn parse_cex(cex: &str) -> std::collections::HashMap<String, i128> {
    let mut m = std::collections::HashMap::new();
    for part in cex.split(',') {
        let part = part.trim();
        if let Some((k, v)) = part.split_once('=') {
            if let Ok(n) = v.trim().parse::<i128>() {
                m.insert(k.trim().to_string(), n);
            }
        }
    }
    m
}

/// Look up a length-style key (`len_y`, `len_x`, …) tolerating a trailing
/// underscore in the rendered name.
fn cex_get(m: &std::collections::HashMap<String, i128>, prefix: &str) -> Option<i128> {
    m.iter()
        .find(|(k, _)| k.as_str() == prefix || k.starts_with(&format!("{prefix}_")) || k.starts_with(&format!("{prefix}=")))
        .map(|(_, v)| *v)
}

/// A minimal input, derived purely from the shape, that is guaranteed to drive
/// the interpreter out of bounds for a Refuted (defective) shape — the smallest
/// concrete witness of the same defect the prover refuted. Used both as a
/// robust fallback (z3 may assign a `len` near `2^32`, which cannot be
/// allocated) and as an independent confirmation that the refutation is real.
fn canonical_oob_witness(shape: &Shape) -> Option<RefInputs> {
    match shape {
        // Guard `pos <= y.len()`: thread `len` writes `y[len]` out of bounds.
        Shape::UnsafeOffByOne { .. } => Some(RefInputs {
            x: vec![],
            idxs: vec![],
            y: vec![0u32; 4],
            s: [1, 1],
            num_threads: 5,
        }),
        // `y[pos] = x[pos + k]`, `k >= 1`, `x.len() == y.len()`: the last thread
        // reads `x[len - 1 + k]`, out of bounds.
        Shape::UnsafeForwardRead { .. } => Some(RefInputs {
            x: vec![0u32; 4],
            idxs: vec![],
            y: vec![0u32; 4],
            s: [1, 1],
            num_threads: 4,
        }),
        _ => None,
    }
}

/// Given a Refuted verdict + counterexample, confirm the interpreter exhibits
/// an OOB. Tries an exact replay of the solver's model first (when its
/// magnitudes are small enough to allocate), then falls back to the shape's
/// minimal witness — either way the refutation must correspond to a concrete,
/// interpreter-observable OOB.
fn replay_refutation(k: &Kernel, cex: &str) -> Result<(), String> {
    // Cap the lengths we materialize: z3 may pick a `len` anywhere in the u32
    // range, and `vec![0; 2^32]` is not allocatable. A small clamp preserves
    // the defect for the pos/length-based refutations the corpus generates.
    const CAP: u32 = 4096;
    let m = parse_cex(cex);
    let abs_pos = cex_get(&m, "abs_pos").unwrap_or(0).clamp(0, CAP as i128) as u32;
    let len_y = cex_get(&m, "len_y").unwrap_or((abs_pos + 1) as i128).clamp(0, CAP as i128) as u32;
    let len_x = cex_get(&m, "len_x").unwrap_or(len_y as i128).clamp(0, CAP as i128) as u32;

    let exact = RefInputs {
        x: vec![0u32; len_x as usize],
        idxs: vec![0u32; len_y as usize],
        y: vec![0u32; len_y as usize],
        s: [1, 1],
        num_threads: (abs_pos + 1).max(len_y).min(CAP),
    };
    if let Outcome::OutOfBounds(_) =
        interpret_dispatch(&k.def, &build_interp_inputs(k, &exact))
    {
        return Ok(());
    }

    // Fallback: the shape's minimal witness of the same defect.
    if let Some(inp) = canonical_oob_witness(&k.shape) {
        if let Outcome::OutOfBounds(_) =
            interpret_dispatch(&k.def, &build_interp_inputs(k, &inp))
        {
            return Ok(());
        }
    }

    Err(format!(
        "prover Refuted (cex `{cex}`) but neither the replayed model nor the shape's minimal \
         witness produced an interpreter OOB (len_x={len_x}, len_y={len_y}, abs_pos={abs_pos})"
    ))
}

// =====================================================================
// Corpus driver
// =====================================================================

/// A cross-check disagreement (never silently reconciled).
#[derive(Clone, Debug)]
pub struct Finding {
    pub seed: u64,
    pub shape: String,
    pub kind: FindingKind,
    pub detail: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FindingKind {
    /// The interpreter and the AST reference disagreed on a concrete input.
    InterpVsReference,
    /// The prover said `Proved` but the interpreter went out of bounds on an
    /// assume-satisfying input — a model-fidelity defect (**critical**).
    ProvedButOob,
    /// The prover said `Refuted` but no interpreter OOB could be exhibited.
    RefutedButNoOob,
    /// The prover's verdict differed from the shape's expectation (a note about
    /// prover precision, not a soundness defect on its own).
    UnexpectedVerdict,
    /// The prover errored (e.g. z3 not available / `unknown`).
    SolverError,
}

#[derive(Clone, Debug, Default)]
pub struct CorpusReport {
    pub kernels: usize,
    pub inputs_checked: usize,
    pub proved: usize,
    pub refuted: usize,
    pub out_of_subset: usize,
    pub findings: Vec<Finding>,
}

impl CorpusReport {
    /// Findings that indicate a genuine soundness/fidelity defect (everything
    /// except the `UnexpectedVerdict`/`SolverError` precision notes).
    pub fn critical(&self) -> impl Iterator<Item = &Finding> {
        self.findings.iter().filter(|f| {
            matches!(
                f.kind,
                FindingKind::InterpVsReference
                    | FindingKind::ProvedButOob
                    | FindingKind::RefutedButNoOob
            )
        })
    }
}

/// Run the full cross-check over `seeds`, doing `inputs_per_kernel` random
/// agreement checks per kernel. `run_prover` gates the (b) prover cross-check
/// (skip when z3 is unavailable). Returns every disagreement as a [`Finding`].
pub fn run_corpus(
    seeds: impl Iterator<Item = u64>,
    inputs_per_kernel: usize,
    run_prover: bool,
) -> CorpusReport {
    let mut report = CorpusReport::default();

    for seed in seeds {
        let k = gen_kernel(seed);
        report.kernels += 1;
        let shape_name = shape_name(&k.shape);

        // ---- (a) reference ≡ interpreter on valid + adversarial inputs -----
        let mut irng = Rng::new(seed ^ 0xA5A5_A5A5_5A5A_5A5A);
        for i in 0..inputs_per_kernel {
            let mut ref_inp = if i % 2 == 0 {
                gen_valid_inputs(&k, &mut irng)
            } else {
                gen_adversarial_inputs(&k, &mut irng)
            };
            let inputs = build_interp_inputs(&k, &ref_inp);
            let outcome = interpret_dispatch(&k.def, &inputs);
            let refr = eval_kernel(&k.shape, &mut ref_inp);
            report.inputs_checked += 1;
            if let Err(detail) = compare(&k, &outcome, &refr) {
                report.findings.push(Finding {
                    seed,
                    shape: shape_name.clone(),
                    kind: FindingKind::InterpVsReference,
                    detail,
                });
            }
        }

        if !run_prover {
            continue;
        }

        // ---- (b) prover ⇄ interpreter --------------------------------------
        let verdict = prove_bounds_freedom(&k.def, &k.buffers, &k.assumes);
        match verdict {
            ProveResult::Proved { .. } => {
                report.proved += 1;
                if k.expected != Verdict::Proved {
                    report.findings.push(Finding {
                        seed,
                        shape: shape_name.clone(),
                        kind: FindingKind::UnexpectedVerdict,
                        detail: "prover Proved a shape expected to Refute".into(),
                    });
                }
                // Exhaustively probe assume-satisfying inputs: a Proved kernel
                // must never drive the interpreter out of bounds.
                let mut prng = Rng::new(seed ^ 0xF0F0_0F0F_0F0F_F0F0);
                for _ in 0..(inputs_per_kernel * 2) {
                    let mut ref_inp = gen_valid_inputs(&k, &mut prng);
                    let inputs = build_interp_inputs(&k, &ref_inp);
                    match interpret_dispatch(&k.def, &inputs) {
                        Outcome::OutOfBounds(o) => {
                            report.findings.push(Finding {
                                seed,
                                shape: shape_name.clone(),
                                kind: FindingKind::ProvedButOob,
                                detail: format!(
                                    "prover Proved OOB-freedom but interpreter hit {o} on an \
                                     assume-satisfying input"
                                ),
                            });
                        }
                        // A DivByZero here would also contradict a bounds
                        // proof's premises for divmod; but our safe divmod
                        // guards s0>=1, so it cannot occur. Report if it does.
                        Outcome::DivByZero { detail, thread } => {
                            report.findings.push(Finding {
                                seed,
                                shape: shape_name.clone(),
                                kind: FindingKind::ProvedButOob,
                                detail: format!("div-by-zero on thread {thread}: {detail}"),
                            });
                        }
                        _ => {}
                    }
                    let _ = eval_kernel(&k.shape, &mut ref_inp);
                }
            }
            ProveResult::Refuted { counterexample, .. } => {
                report.refuted += 1;
                if k.expected != Verdict::Refuted {
                    report.findings.push(Finding {
                        seed,
                        shape: shape_name.clone(),
                        kind: FindingKind::UnexpectedVerdict,
                        detail: "prover Refuted a shape expected to Prove".into(),
                    });
                }
                if let Err(detail) = replay_refutation(&k, &counterexample) {
                    report.findings.push(Finding {
                        seed,
                        shape: shape_name.clone(),
                        kind: FindingKind::RefutedButNoOob,
                        detail,
                    });
                }
            }
            ProveResult::OutOfSubset { reason } => {
                report.out_of_subset += 1;
                // Conservative (no claim): only a note if we expected a proof.
                if k.expected == Verdict::Proved {
                    report.findings.push(Finding {
                        seed,
                        shape: shape_name.clone(),
                        kind: FindingKind::UnexpectedVerdict,
                        detail: format!("expected Proved but prover was OutOfSubset: {reason}"),
                    });
                }
            }
            ProveResult::SolverError { detail } => {
                report.findings.push(Finding {
                    seed,
                    shape: shape_name.clone(),
                    kind: FindingKind::SolverError,
                    detail,
                });
            }
        }
    }

    report
}

fn shape_name(s: &Shape) -> String {
    match s {
        Shape::SafeDirect { .. } => "SafeDirect",
        Shape::SafeDivMod { .. } => "SafeDivMod",
        Shape::SafeGather => "SafeGather",
        Shape::SafeLoop { .. } => "SafeLoop",
        Shape::SafeSwitch { .. } => "SafeSwitch",
        Shape::UnsafeOffByOne { .. } => "UnsafeOffByOne",
        Shape::UnsafeForwardRead { .. } => "UnsafeForwardRead",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;


    /// Deterministic wired-into-`cargo test` subset: a modest seeded corpus,
    /// prover included, must produce ZERO critical findings (interpreter vs
    /// reference, Proved-but-OOB, Refuted-but-no-OOB). This is the fuzz lane's
    /// standing regression against a model-fidelity discrepancy.
    #[test]
    fn deterministic_corpus_has_no_critical_findings() {
        let report = run_corpus(0..400, 12, true);
        let critical: Vec<_> = report.critical().cloned().collect();
        assert!(
            critical.is_empty(),
            "critical findings:\n{}",
            critical
                .iter()
                .map(|f| format!("  seed {} [{}] {:?}: {}", f.seed, f.shape, f.kind, f.detail))
                .collect::<Vec<_>>()
                .join("\n")
        );
        // Sanity: the corpus actually exercised both prover verdicts.
        assert!(report.proved > 0, "no Proved kernels in the corpus");
        assert!(report.refuted > 0, "no Refuted kernels in the corpus");
        eprintln!(
            "fuzz corpus: {} kernels, {} inputs, proved {}, refuted {}, out_of_subset {}, \
             {} non-critical notes",
            report.kernels,
            report.inputs_checked,
            report.proved,
            report.refuted,
            report.out_of_subset,
            report.findings.len()
        );
    }

    /// The agreement leg alone (no prover) over a larger seed range — cheap,
    /// runs without z3.
    #[test]
    fn agreement_only_larger_corpus() {
        let report = run_corpus(1000..3000, 8, false);
        let critical: Vec<_> = report.critical().cloned().collect();
        assert!(
            critical.is_empty(),
            "critical findings:\n{}",
            critical
                .iter()
                .map(|f| format!("  seed {} [{}] {:?}: {}", f.seed, f.shape, f.kind, f.detail))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// The full corpus, behind `VERICL_FUZZ=1` (heavy: spawns z3 per kernel).
    /// Run with:
    ///   VERICL_FUZZ=1 cargo test -p vericl-ir --release full_corpus -- --ignored --nocapture
    #[test]
    #[ignore = "heavy; enable with VERICL_FUZZ=1"]
    fn full_corpus() {
        let n: u64 = std::env::var("VERICL_FUZZ_KERNELS").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000);
        let run_prover = crate::z3_version().is_some();
        let start = std::time::Instant::now();
        let report = run_corpus(0..n, 16, run_prover);
        let elapsed = start.elapsed();
        eprintln!(
            "FULL fuzz corpus: {} kernels, {} agreement inputs, proved {}, refuted {}, \
             out_of_subset {}, findings {} (critical {}) in {:.1}s (prover={})",
            report.kernels,
            report.inputs_checked,
            report.proved,
            report.refuted,
            report.out_of_subset,
            report.findings.len(),
            report.critical().count(),
            elapsed.as_secs_f64(),
            run_prover,
        );
        for f in report.critical() {
            eprintln!("  CRITICAL seed {} [{}] {:?}: {}", f.seed, f.shape, f.kind, f.detail);
        }
        assert_eq!(report.critical().count(), 0, "critical findings in full corpus");
    }
}
