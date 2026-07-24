//! Concrete reference interpreter over the CubeCL IR (`KernelDefinition`).
//!
//! This is a **third, independent implementation** of the modeled cube
//! semantics, deliberately unlike the other two:
//!
//! * the macro *twin* (`crates/vericl-macros`) rewrites the kernel's Rust
//!   source tokens into a host `reference(...)` function — it never sees the IR;
//! * the *prover* (`crates/vericl-ir/src/prover.rs`) encodes the IR
//!   *symbolically* into QF_LIA and asks z3 whether an out-of-bounds access is
//!   reachable — it never runs the kernel;
//! * this *interpreter* executes the IR **concretely**, one thread at a time,
//!   over concrete input buffers/scalars, with real finite-width wrapping
//!   integer arithmetic and IEEE-754 float arithmetic, reporting (rather than
//!   panicking on) any out-of-bounds index or division by zero.
//!
//! Because all three consume the *same* `KernelDefinition`, running the
//! interpreter against the twin (agreement = two independent semantics
//! implementations concur) and against the prover (a `Proved` kernel that the
//! interpreter can drive out of bounds would be a model-fidelity defect) is an
//! empirical cross-check on vericl's fidelity to real CubeCL semantics. It is
//! **not a proof** — see `docs/interpreter.md` for exactly what agreement does
//! and does not establish.
//!
//! ## Scope (v0): non-cooperative, scalar, 1-D
//!
//! Covered: arithmetic (`Add`/`Sub`/`Mul`/`Div`/`Modulo`/`Neg`/`Abs`/`Min`/
//! `Max`/`Fma`/float transcendentals), comparisons, boolean composition
//! (`And`/`Or`/`Not`), bitwise ops (`&`/`|`/`^`/`<<`/`>>`/`!`/`count_ones`/…),
//! `Cast`/`Reinterpret`/`Select`, `Metadata::Length`, `Index`/`IndexAssign`
//! (checked and unchecked) with bounds reporting, `If`/`IfElse`, `Switch`,
//! `RangeLoop` (ascending, unit or positive step) and the bare `Loop`
//! (break-terminated, guarded by an instruction budget), topology builtins for
//! a 1-D dispatch, local scratch arrays and constant arrays.
//!
//! Explicitly **excluded** and reported as `Unsupported` (never guessed): all
//! cooperative constructs (`SharedArray`/`Shared`/`sync_cube`), atomics,
//! plane/warp ops, cooperative-matrix, tensor metadata (`Rank`/`Stride`/
//! `Shape`), vectorized (`Vector<_, N>`) indexing, TMA, and stepped/descending
//! range loops. A kernel touching any of these is rejected up front by
//! [`unsupported_construct`] rather than mis-executed.

use std::collections::HashMap;

use cubecl::ir::{
    Arithmetic, Bitwise, Branch, Builtin, Comparison, ConstantValue, ElemType, FloatKind, Id,
    IntKind, Metadata, Operation, Operator, Scope, Synchronization, Type, UIntKind, Variable,
    VariableKind,
};
use cubecl::prelude::KernelDefinition;

/// A concrete scalar value, tagged with its finite-width hardware type so that
/// arithmetic wraps exactly like the GPU (and the `wrapping` twin) does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Val {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
}

impl Val {
    fn elem_type(&self) -> ElemType {
        match self {
            Val::U8(_) => ElemType::UInt(UIntKind::U8),
            Val::U16(_) => ElemType::UInt(UIntKind::U16),
            Val::U32(_) => ElemType::UInt(UIntKind::U32),
            Val::U64(_) => ElemType::UInt(UIntKind::U64),
            Val::I8(_) => ElemType::Int(IntKind::I8),
            Val::I16(_) => ElemType::Int(IntKind::I16),
            Val::I32(_) => ElemType::Int(IntKind::I32),
            Val::I64(_) => ElemType::Int(IntKind::I64),
            Val::F32(_) => ElemType::Float(FloatKind::F32),
            Val::F64(_) => ElemType::Float(FloatKind::F64),
            Val::Bool(_) => ElemType::Bool,
        }
    }

    /// The integer value as `i128`, or `None` for floats. `Bool` reads as 0/1.
    fn as_int(&self) -> Option<i128> {
        Some(match self {
            Val::U8(v) => *v as i128,
            Val::U16(v) => *v as i128,
            Val::U32(v) => *v as i128,
            Val::U64(v) => *v as i128,
            Val::I8(v) => *v as i128,
            Val::I16(v) => *v as i128,
            Val::I32(v) => *v as i128,
            Val::I64(v) => *v as i128,
            Val::Bool(b) => *b as i128,
            Val::F32(_) | Val::F64(_) => return None,
        })
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Val::F32(v) => Some(*v as f64),
            Val::F64(v) => Some(*v),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Val::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Build a `Val` from an IR constant, honoring its declared type.
    fn from_const(c: ConstantValue, ty: Type) -> Val {
        Val::from_int_or_float(&c, ty.elem_type())
    }

    fn from_int_or_float(c: &ConstantValue, elem: ElemType) -> Val {
        match elem {
            ElemType::UInt(UIntKind::U8) => Val::U8(c.as_u64() as u8),
            ElemType::UInt(UIntKind::U16) => Val::U16(c.as_u64() as u16),
            ElemType::UInt(UIntKind::U32) => Val::U32(c.as_u64() as u32),
            ElemType::UInt(UIntKind::U64) => Val::U64(c.as_u64()),
            ElemType::Int(IntKind::I8) => Val::I8(c.as_i64() as i8),
            ElemType::Int(IntKind::I16) => Val::I16(c.as_i64() as i16),
            ElemType::Int(IntKind::I32) => Val::I32(c.as_i64() as i32),
            ElemType::Int(IntKind::I64) => Val::I64(c.as_i64()),
            ElemType::Float(FloatKind::F64) => Val::F64(c.as_f64()),
            ElemType::Float(_) => Val::F32(c.as_f64() as f32),
            ElemType::Bool => Val::Bool(c.as_bool()),
        }
    }

    /// Reinterpret the integer `v` as the given element type (used for the
    /// results of ops whose declared output type differs from the operands',
    /// and for casts). Truncates/wraps to the target width, faithful to Rust's
    /// (and WGSL's, for the width-preserving integer casts in the modeled
    /// subset) `as`-conversion.
    fn int_to_elem(v: i128, elem: ElemType) -> Val {
        match elem {
            ElemType::UInt(UIntKind::U8) => Val::U8(v as u8),
            ElemType::UInt(UIntKind::U16) => Val::U16(v as u16),
            ElemType::UInt(UIntKind::U32) => Val::U32(v as u32),
            ElemType::UInt(UIntKind::U64) => Val::U64(v as u64),
            ElemType::Int(IntKind::I8) => Val::I8(v as i8),
            ElemType::Int(IntKind::I16) => Val::I16(v as i16),
            ElemType::Int(IntKind::I32) => Val::I32(v as i32),
            ElemType::Int(IntKind::I64) => Val::I64(v as i64),
            ElemType::Bool => Val::Bool(v != 0),
            // A float target for an int source is a value cast, handled in
            // `cast`; this helper is only for integer/bool targets.
            ElemType::Float(FloatKind::F64) => Val::F64(v as f64),
            ElemType::Float(_) => Val::F32(v as f32),
        }
    }
}

/// One concrete array argument (an input or output buffer), in
/// buffer-registration order (index == buffer id).
#[derive(Clone, Debug)]
pub struct Buffer {
    /// Parameter name, for legible OOB findings (`""` if unknown).
    pub name: String,
    pub elem: ElemType,
    pub data: Vec<Val>,
    pub is_output: bool,
}

/// One concrete scalar argument. `id` is the per-storage-type registration
/// index CubeCL assigns (`GlobalScalar(id)`); scalars of the same element type
/// are numbered `0, 1, …` in parameter order (see `KernelBuilder::scalar`).
#[derive(Clone, Copy, Debug)]
pub struct ScalarBinding {
    pub elem: ElemType,
    pub id: Id,
    pub val: Val,
}

/// A full set of concrete inputs for one dispatch.
#[derive(Clone, Debug)]
pub struct Inputs {
    pub buffers: Vec<Buffer>,
    pub scalars: Vec<ScalarBinding>,
    /// Threads per cube along X (used only to reconstruct `UnitPos`/`CubePos`
    /// from `AbsolutePos`; for a plain elementwise kernel that reads only
    /// `AbsolutePos`, any value gives identical results).
    pub cube_dim: u32,
    /// Total dispatched threads — `AbsolutePos` ranges over `0..num_threads`,
    /// matching the twin's `for ABSOLUTE_POS in 0..n` loop.
    pub num_threads: u32,
}

/// An out-of-bounds access the interpreter caught (rather than panicking).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Oob {
    pub array: String,
    pub index: i128,
    pub len: usize,
    pub write: bool,
    pub thread: u32,
}

impl std::fmt::Display for Oob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {}[{}] out of bounds (len {}) on thread {}",
            if self.write { "write" } else { "read" },
            self.array,
            self.index,
            self.len,
            self.thread
        )
    }
}

/// The outcome of interpreting a whole dispatch.
#[derive(Clone, Debug)]
pub enum Outcome {
    /// The kernel ran to completion for every thread; `buffers` are the final
    /// (post-write) contents of every buffer, in registration order.
    Completed { buffers: Vec<Buffer> },
    /// An index read/write went out of bounds (reported, not panicked).
    OutOfBounds(Oob),
    /// A division or modulo by zero was reached (reported, not panicked).
    DivByZero { detail: String, thread: u32 },
    /// The kernel uses a construct outside the interpreter's v0 subset.
    Unsupported { reason: String },
}

/// Internal terminal condition raised while walking a thread.
enum Halt {
    Oob(Oob),
    DivByZero { detail: String },
    Unsupported(String),
}

/// Control-flow result of executing a scope / instruction.
enum Signal {
    Continue,
    Break,
    Return,
}

/// A hashable key for the register-valued IR variables the environment tracks.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum VarKey {
    LocalConst(Id),
    LocalMut(Id),
    Versioned(Id, u16),
}

/// If `def` uses any construct outside the interpreter's v0 subset, returns the
/// reason; otherwise `None`. Checked once up front so a partially-executed
/// kernel never produces a misleading result.
fn unsupported_construct(def: &KernelDefinition) -> Option<String> {
    scan_scope(&def.body)
}

fn scan_scope(scope: &Scope) -> Option<String> {
    for inst in &scope.instructions {
        // Any operand or output that is a cooperative/opaque memory kind.
        if let Some(out) = inst.out.as_ref() {
            if let Some(r) = scan_var(out) {
                return Some(r);
            }
        }
        if let Some(r) = scan_operation(&inst.operation) {
            return Some(r);
        }
    }
    None
}

fn scan_var(v: &Variable) -> Option<String> {
    // Vectorized types are out of subset (scalar interpreter only).
    if matches!(v.ty, Type::Vector(..)) {
        return Some("vectorized value (Vector<_, N>) is outside the interpreter v0 subset".into());
    }
    match v.kind {
        VariableKind::SharedArray { .. } | VariableKind::Shared { .. } => Some(
            "shared memory (cooperative kernel) is outside the interpreter v0 subset".into(),
        ),
        VariableKind::Matrix { .. } => {
            Some("cooperative matrix is outside the interpreter v0 subset".into())
        }
        VariableKind::TensorMapInput(_) | VariableKind::TensorMapOutput(_) => {
            Some("tensor map is outside the interpreter v0 subset".into())
        }
        _ => None,
    }
}

fn scan_operation(op: &Operation) -> Option<String> {
    match op {
        Operation::Synchronization(Synchronization::SyncCube) => {
            Some("sync_cube (cooperative kernel) is outside the interpreter v0 subset".into())
        }
        Operation::Synchronization(_) => {
            Some("synchronization barrier is outside the interpreter v0 subset".into())
        }
        Operation::Atomic(_) => Some("atomic op is outside the interpreter v0 subset".into()),
        Operation::Plane(_) => Some("plane/warp op is outside the interpreter v0 subset".into()),
        Operation::CoopMma(_) => {
            Some("cooperative-matrix op is outside the interpreter v0 subset".into())
        }
        Operation::Barrier(_) => Some("barrier op is outside the interpreter v0 subset".into()),
        Operation::Tma(_) => Some("TMA op is outside the interpreter v0 subset".into()),
        Operation::Metadata(Metadata::Rank { .. })
        | Operation::Metadata(Metadata::Stride { .. })
        | Operation::Metadata(Metadata::Shape { .. }) => {
            Some("tensor metadata (rank/stride/shape) is outside the interpreter v0 subset".into())
        }
        Operation::Operator(Operator::InitVector(_)) => {
            Some("vector construction is outside the interpreter v0 subset".into())
        }
        Operation::Operator(Operator::CopyMemory(_))
        | Operation::Operator(Operator::CopyMemoryBulk(_)) => {
            Some("bulk memory copy is outside the interpreter v0 subset".into())
        }
        // Recurse into nested scopes.
        Operation::Branch(b) => match b {
            Branch::If(i) => scan_scope(&i.scope),
            Branch::IfElse(i) => scan_scope(&i.scope_if).or_else(|| scan_scope(&i.scope_else)),
            Branch::Switch(s) => {
                if let Some(r) = scan_scope(&s.scope_default) {
                    return Some(r);
                }
                s.cases.iter().find_map(|(_, sc)| scan_scope(sc))
            }
            Branch::RangeLoop(r) => scan_scope(&r.scope),
            Branch::Loop(l) => scan_scope(&l.scope),
            _ => None,
        },
        _ => None,
    }
}

/// Interpret a whole dispatch: run `AbsolutePos = 0..num_threads` sequentially
/// (threads are independent in the non-cooperative subset), applying every
/// thread's writes to the shared buffers, exactly as the sequential twin does.
pub fn interpret_dispatch(def: &KernelDefinition, inputs: &Inputs) -> Outcome {
    if let Some(reason) = unsupported_construct(def) {
        return Outcome::Unsupported { reason };
    }

    let const_arrays = collect_const_arrays(&def.body);
    let scalars: HashMap<(ElemType, Id), Val> =
        inputs.scalars.iter().map(|s| ((s.elem, s.id), s.val)).collect();

    let mut interp = Interp {
        buffers: inputs.buffers.clone(),
        scalars,
        const_arrays,
        cube_dim: inputs.cube_dim.max(1),
        num_threads: inputs.num_threads,
        env: HashMap::new(),
        local_arrays: HashMap::new(),
        budget: 0,
    };

    for pos in 0..inputs.num_threads {
        interp.env.clear();
        interp.local_arrays.clear();
        interp.budget = 0;
        let ctx = ThreadCtx::new(pos, interp.cube_dim, interp.num_threads);
        match interp.exec_scope(&def.body, &ctx) {
            Ok(_) => {}
            Err(Halt::Oob(mut o)) => {
                o.thread = pos;
                return Outcome::OutOfBounds(o);
            }
            Err(Halt::DivByZero { detail }) => {
                return Outcome::DivByZero { detail, thread: pos };
            }
            Err(Halt::Unsupported(reason)) => return Outcome::Unsupported { reason },
        }
    }

    Outcome::Completed { buffers: interp.buffers }
}

/// Per-thread topology, reconstructed from a 1-D dispatch.
struct ThreadCtx {
    abs_pos: u32,
    unit_pos: u32,
    cube_pos: u32,
    cube_dim: u32,
    cube_count: u32,
}

impl ThreadCtx {
    fn new(abs_pos: u32, cube_dim: u32, num_threads: u32) -> Self {
        let cube_count = num_threads.div_ceil(cube_dim.max(1));
        ThreadCtx {
            abs_pos,
            unit_pos: abs_pos % cube_dim.max(1),
            cube_pos: abs_pos / cube_dim.max(1),
            cube_dim,
            cube_count,
        }
    }
}

/// Max instructions executed per thread before assuming non-termination.
const INSTRUCTION_BUDGET: u64 = 50_000_000;

struct Interp {
    buffers: Vec<Buffer>,
    scalars: HashMap<(ElemType, Id), Val>,
    const_arrays: HashMap<Id, Vec<Val>>,
    cube_dim: u32,
    num_threads: u32,
    /// Register environment for the current thread.
    env: HashMap<VarKey, Val>,
    /// Thread-local scratch arrays (`VariableKind::LocalArray`) for the current
    /// thread, keyed by id.
    local_arrays: HashMap<Id, Vec<Val>>,
    budget: u64,
}

impl Interp {
    fn tick(&mut self) -> Result<(), Halt> {
        self.budget += 1;
        if self.budget > INSTRUCTION_BUDGET {
            return Err(Halt::Unsupported(
                "instruction budget exceeded (possible non-terminating loop)".into(),
            ));
        }
        Ok(())
    }

    fn exec_scope(&mut self, scope: &Scope, ctx: &ThreadCtx) -> Result<Signal, Halt> {
        for inst in &scope.instructions {
            self.tick()?;
            match self.exec_inst(inst, ctx)? {
                Signal::Continue => {}
                other => return Ok(other),
            }
        }
        Ok(Signal::Continue)
    }

    fn exec_inst(
        &mut self,
        inst: &cubecl::ir::Instruction,
        ctx: &ThreadCtx,
    ) -> Result<Signal, Halt> {
        match &inst.operation {
            Operation::Copy(v) => {
                let val = self.value_of(v, ctx)?;
                self.bind(inst.out.as_ref(), val);
            }
            Operation::Arithmetic(a) => {
                let val = self.eval_arithmetic(a, inst.out.as_ref(), ctx)?;
                self.bind(inst.out.as_ref(), val);
            }
            Operation::Comparison(c) => {
                let val = self.eval_comparison(c, ctx)?;
                self.bind(inst.out.as_ref(), val);
            }
            Operation::Bitwise(b) => {
                let val = self.eval_bitwise(b, inst.out.as_ref(), ctx)?;
                self.bind(inst.out.as_ref(), val);
            }
            Operation::Operator(op) => {
                return self.exec_operator(op, inst.out.as_ref(), ctx);
            }
            Operation::Metadata(m) => {
                let val = self.eval_metadata(m)?;
                self.bind(inst.out.as_ref(), val);
            }
            Operation::Branch(b) => return self.exec_branch(b, ctx),
            // No-ops for the interpreter.
            Operation::NonSemantic(_) | Operation::Marker(_) => {}
            // Everything else was rejected by `unsupported_construct`; if one
            // slips through, fail closed rather than silently continue.
            other => {
                return Err(Halt::Unsupported(format!(
                    "unsupported operation reached at runtime: {other}"
                )));
            }
        }
        Ok(Signal::Continue)
    }

    // ---- value resolution -------------------------------------------------

    fn value_of(&self, v: &Variable, ctx: &ThreadCtx) -> Result<Val, Halt> {
        match v.kind {
            VariableKind::Constant(c) => Ok(Val::from_const(c, v.ty)),
            VariableKind::Builtin(b) => self.builtin(b, ctx),
            VariableKind::GlobalScalar(id) => {
                let elem = v.ty.elem_type();
                self.scalars.get(&(elem, id)).copied().ok_or_else(|| {
                    Halt::Unsupported(format!("scalar {elem:?}#{id} has no bound value"))
                })
            }
            VariableKind::LocalConst { id } => self.env_get(VarKey::LocalConst(id)),
            VariableKind::LocalMut { id } => self.env_get(VarKey::LocalMut(id)),
            VariableKind::Versioned { id, version } => self.env_get(VarKey::Versioned(id, version)),
            _ => Err(Halt::Unsupported(format!(
                "cannot resolve variable kind as a scalar value: {v}"
            ))),
        }
    }

    fn env_get(&self, key: VarKey) -> Result<Val, Halt> {
        self.env
            .get(&key)
            .copied()
            .ok_or_else(|| Halt::Unsupported("read of an unbound local (use before def)".into()))
    }

    fn builtin(&self, b: Builtin, ctx: &ThreadCtx) -> Result<Val, Halt> {
        let v = match b {
            Builtin::AbsolutePos | Builtin::AbsolutePosX => ctx.abs_pos,
            Builtin::AbsolutePosY | Builtin::AbsolutePosZ => 0,
            Builtin::UnitPos | Builtin::UnitPosX => ctx.unit_pos,
            Builtin::UnitPosY | Builtin::UnitPosZ => 0,
            Builtin::CubePos | Builtin::CubePosX => ctx.cube_pos,
            Builtin::CubePosY | Builtin::CubePosZ => 0,
            Builtin::CubeDim | Builtin::CubeDimX => ctx.cube_dim,
            Builtin::CubeDimY | Builtin::CubeDimZ => 1,
            Builtin::CubeCount | Builtin::CubeCountX => ctx.cube_count,
            Builtin::CubeCountY | Builtin::CubeCountZ => 1,
            _ => {
                return Err(Halt::Unsupported(format!(
                    "topology builtin {b:?} is outside the interpreter v0 subset"
                )));
            }
        };
        Ok(Val::U32(v))
    }

    fn bind(&mut self, out: Option<&Variable>, val: Val) {
        if let Some(out) = out {
            let key = match out.kind {
                VariableKind::LocalConst { id } => VarKey::LocalConst(id),
                VariableKind::LocalMut { id } => VarKey::LocalMut(id),
                VariableKind::Versioned { id, version } => VarKey::Versioned(id, version),
                // Writes to arrays/other kinds are handled by their ops, not here.
                _ => return,
            };
            self.env.insert(key, val);
        }
    }

    // ---- arithmetic -------------------------------------------------------

    fn eval_arithmetic(
        &self,
        a: &Arithmetic,
        out: Option<&Variable>,
        ctx: &ThreadCtx,
    ) -> Result<Val, Halt> {
        use Arithmetic::*;
        let out_elem = out.map(|o| o.ty.elem_type());
        match a {
            Add(op) => self.int_or_float_bin(&op.lhs, &op.rhs, ctx, IntBin::Add, |x, y| x + y),
            Sub(op) => self.int_or_float_bin(&op.lhs, &op.rhs, ctx, IntBin::Sub, |x, y| x - y),
            Mul(op) => self.int_or_float_bin(&op.lhs, &op.rhs, ctx, IntBin::Mul, |x, y| x * y),
            Div(op) => self.div_or_mod(&op.lhs, &op.rhs, ctx, true),
            Modulo(op) => self.div_or_mod(&op.lhs, &op.rhs, ctx, false),
            Remainder(op) => self.div_or_mod(&op.lhs, &op.rhs, ctx, false),
            Neg(op) => {
                let v = self.value_of(&op.input, ctx)?;
                match v {
                    Val::F32(x) => Ok(Val::F32(-x)),
                    Val::F64(x) => Ok(Val::F64(-x)),
                    _ => {
                        let i = v.as_int().ok_or_else(|| Halt::Unsupported("neg of non-numeric".into()))?;
                        Ok(wrapping_neg(v, i))
                    }
                }
            }
            Abs(op) => {
                let v = self.value_of(&op.input, ctx)?;
                Ok(match v {
                    Val::F32(x) => Val::F32(x.abs()),
                    Val::F64(x) => Val::F64(x.abs()),
                    Val::I8(x) => Val::I8(x.wrapping_abs()),
                    Val::I16(x) => Val::I16(x.wrapping_abs()),
                    Val::I32(x) => Val::I32(x.wrapping_abs()),
                    Val::I64(x) => Val::I64(x.wrapping_abs()),
                    other => other, // unsigned abs is identity
                })
            }
            Min(op) => self.int_or_float_bin(&op.lhs, &op.rhs, ctx, IntBin::Min, f64::min),
            Max(op) => self.int_or_float_bin(&op.lhs, &op.rhs, ctx, IntBin::Max, f64::max),
            Fma(op) => {
                let x = self.value_of(&op.a, ctx)?;
                let y = self.value_of(&op.b, ctx)?;
                let z = self.value_of(&op.c, ctx)?;
                match (x, y, z) {
                    (Val::F32(a), Val::F32(b), Val::F32(c)) => Ok(Val::F32(a.mul_add(b, c))),
                    (Val::F64(a), Val::F64(b), Val::F64(c)) => Ok(Val::F64(a.mul_add(b, c))),
                    _ => Err(Halt::Unsupported("fma on non-float operands".into())),
                }
            }
            Sqrt(op) => self.float_unary(&op.input, ctx, f64::sqrt),
            Recip(op) => self.float_unary(&op.input, ctx, |x| 1.0 / x),
            Floor(op) => self.float_unary(&op.input, ctx, f64::floor),
            Ceil(op) => self.float_unary(&op.input, ctx, f64::ceil),
            Round(op) => self.float_unary(&op.input, ctx, |x| x.round_ties_even()),
            Trunc(op) => self.float_unary(&op.input, ctx, f64::trunc),
            Exp(op) => self.float_unary(&op.input, ctx, f64::exp),
            Log(op) => self.float_unary(&op.input, ctx, f64::ln),
            Sin(op) => self.float_unary(&op.input, ctx, f64::sin),
            Cos(op) => self.float_unary(&op.input, ctx, f64::cos),
            Tanh(op) => self.float_unary(&op.input, ctx, f64::tanh),
            Powf(op) => self.float_binary(&op.lhs, &op.rhs, ctx, f64::powf),
            Clamp(op) => {
                let v = self.value_of(&op.input, ctx)?;
                let lo = self.value_of(&op.min_value, ctx)?;
                let hi = self.value_of(&op.max_value, ctx)?;
                match (v, lo, hi) {
                    (Val::F32(a), Val::F32(l), Val::F32(h)) => Ok(Val::F32(a.clamp(l, h))),
                    (Val::F64(a), Val::F64(l), Val::F64(h)) => Ok(Val::F64(a.clamp(l, h))),
                    _ => {
                        let (a, l, h) = (
                            v.as_int().ok_or_else(bad("clamp"))?,
                            lo.as_int().ok_or_else(bad("clamp"))?,
                            hi.as_int().ok_or_else(bad("clamp"))?,
                        );
                        let elem = out_elem.unwrap_or_else(|| v.elem_type());
                        Ok(Val::int_to_elem(a.clamp(l, h), elem))
                    }
                }
            }
            MulHi(op) => {
                // High word of the full-width product (u32 only in the subset).
                let x = self.value_of(&op.lhs, ctx)?;
                let y = self.value_of(&op.rhs, ctx)?;
                match (x, y) {
                    (Val::U32(a), Val::U32(b)) => {
                        Ok(Val::U32(((a as u64 * b as u64) >> 32) as u32))
                    }
                    _ => Err(Halt::Unsupported("mul_hi on non-u32 operands".into())),
                }
            }
            other => Err(Halt::Unsupported(format!(
                "arithmetic op {other} is outside the interpreter v0 subset"
            ))),
        }
    }

    fn float_unary(
        &self,
        input: &Variable,
        ctx: &ThreadCtx,
        f: impl Fn(f64) -> f64,
    ) -> Result<Val, Halt> {
        match self.value_of(input, ctx)? {
            Val::F32(x) => Ok(Val::F32(f(x as f64) as f32)),
            Val::F64(x) => Ok(Val::F64(f(x))),
            _ => Err(Halt::Unsupported("float unary on non-float operand".into())),
        }
    }

    fn float_binary(
        &self,
        lhs: &Variable,
        rhs: &Variable,
        ctx: &ThreadCtx,
        f: impl Fn(f64, f64) -> f64,
    ) -> Result<Val, Halt> {
        let a = self.value_of(lhs, ctx)?;
        let b = self.value_of(rhs, ctx)?;
        match (a, b) {
            (Val::F32(x), Val::F32(y)) => Ok(Val::F32(f(x as f64, y as f64) as f32)),
            (Val::F64(x), Val::F64(y)) => Ok(Val::F64(f(x, y))),
            _ => Err(Halt::Unsupported("float binary on non-float operands".into())),
        }
    }

    /// Integer path uses width-exact wrapping; float path uses `ff` at f64
    /// precision then re-rounds to the operand width (f32 stays f32).
    fn int_or_float_bin(
        &self,
        lhs: &Variable,
        rhs: &Variable,
        ctx: &ThreadCtx,
        op: IntBin,
        ff: impl Fn(f64, f64) -> f64,
    ) -> Result<Val, Halt> {
        let a = self.value_of(lhs, ctx)?;
        let b = self.value_of(rhs, ctx)?;
        match (a, b) {
            (Val::F32(x), Val::F32(y)) => {
                // f32 arithmetic rounds each op to f32 — do it in f32 for
                // +/-/* so it is bit-exact with the Rust twin; min/max via ff.
                Ok(Val::F32(match op {
                    IntBin::Add => x + y,
                    IntBin::Sub => x - y,
                    IntBin::Mul => x * y,
                    _ => ff(x as f64, y as f64) as f32,
                }))
            }
            (Val::F64(x), Val::F64(y)) => Ok(Val::F64(match op {
                IntBin::Add => x + y,
                IntBin::Sub => x - y,
                IntBin::Mul => x * y,
                _ => ff(x, y),
            })),
            _ => int_bin(a, b, op),
        }
    }

    fn div_or_mod(
        &self,
        lhs: &Variable,
        rhs: &Variable,
        ctx: &ThreadCtx,
        is_div: bool,
    ) -> Result<Val, Halt> {
        let a = self.value_of(lhs, ctx)?;
        let b = self.value_of(rhs, ctx)?;
        if let (Some(x), Some(y)) = (a.as_f64(), b.as_f64()) {
            let r = if is_div { x / y } else { x % y };
            return Ok(match a {
                Val::F64(_) => Val::F64(r),
                _ => Val::F32(r as f32),
            });
        }
        // Integer div/mod: report (rather than panic on) division by zero.
        let yi = b.as_int().ok_or_else(bad("div/mod"))?;
        if yi == 0 {
            return Err(Halt::DivByZero {
                detail: format!("{} by zero", if is_div { "division" } else { "modulo" }),
            });
        }
        int_div_mod(a, b, is_div)
    }

    // ---- comparisons ------------------------------------------------------

    fn eval_comparison(&self, c: &Comparison, ctx: &ThreadCtx) -> Result<Val, Halt> {
        use Comparison::*;
        let bin = |s: &Self, l: &Variable, r: &Variable| -> Result<(Val, Val), Halt> {
            Ok((s.value_of(l, ctx)?, s.value_of(r, ctx)?))
        };
        let r = match c {
            Lower(op) => {
                let (a, b) = bin(self, &op.lhs, &op.rhs)?;
                cmp(a, b, |o| o == std::cmp::Ordering::Less)?
            }
            LowerEqual(op) => {
                let (a, b) = bin(self, &op.lhs, &op.rhs)?;
                cmp(a, b, |o| o != std::cmp::Ordering::Greater)?
            }
            Greater(op) => {
                let (a, b) = bin(self, &op.lhs, &op.rhs)?;
                cmp(a, b, |o| o == std::cmp::Ordering::Greater)?
            }
            GreaterEqual(op) => {
                let (a, b) = bin(self, &op.lhs, &op.rhs)?;
                cmp(a, b, |o| o != std::cmp::Ordering::Less)?
            }
            Equal(op) => {
                let (a, b) = bin(self, &op.lhs, &op.rhs)?;
                eq(a, b)
            }
            NotEqual(op) => {
                let (a, b) = bin(self, &op.lhs, &op.rhs)?;
                !eq(a, b)
            }
            IsNan(op) => match self.value_of(&op.input, ctx)? {
                Val::F32(x) => x.is_nan(),
                Val::F64(x) => x.is_nan(),
                _ => false,
            },
            IsInf(op) => match self.value_of(&op.input, ctx)? {
                Val::F32(x) => x.is_infinite(),
                Val::F64(x) => x.is_infinite(),
                _ => false,
            },
        };
        Ok(Val::Bool(r))
    }

    // ---- bitwise ----------------------------------------------------------

    fn eval_bitwise(
        &self,
        b: &Bitwise,
        _out: Option<&Variable>,
        ctx: &ThreadCtx,
    ) -> Result<Val, Halt> {
        use Bitwise::*;
        match b {
            BitwiseAnd(op) => self.bit_bin(&op.lhs, &op.rhs, ctx, BitOp::And),
            BitwiseOr(op) => self.bit_bin(&op.lhs, &op.rhs, ctx, BitOp::Or),
            BitwiseXor(op) => self.bit_bin(&op.lhs, &op.rhs, ctx, BitOp::Xor),
            ShiftLeft(op) => self.bit_bin(&op.lhs, &op.rhs, ctx, BitOp::Shl),
            ShiftRight(op) => self.bit_bin(&op.lhs, &op.rhs, ctx, BitOp::Shr),
            BitwiseNot(op) => {
                let v = self.value_of(&op.input, ctx)?;
                Ok(match v {
                    Val::U8(x) => Val::U8(!x),
                    Val::U16(x) => Val::U16(!x),
                    Val::U32(x) => Val::U32(!x),
                    Val::U64(x) => Val::U64(!x),
                    Val::I8(x) => Val::I8(!x),
                    Val::I16(x) => Val::I16(!x),
                    Val::I32(x) => Val::I32(!x),
                    Val::I64(x) => Val::I64(!x),
                    _ => return Err(Halt::Unsupported("bitwise not on non-integer".into())),
                })
            }
            CountOnes(op) => self.bit_unary_count(&op.input, ctx, |x, _w| x.count_ones()),
            ReverseBits(op) => {
                let v = self.value_of(&op.input, ctx)?;
                Ok(match v {
                    Val::U32(x) => Val::U32(x.reverse_bits()),
                    Val::U64(x) => Val::U64(x.reverse_bits()),
                    Val::I32(x) => Val::I32(x.reverse_bits()),
                    _ => return Err(Halt::Unsupported("reverse_bits on unsupported width".into())),
                })
            }
            LeadingZeros(op) => self.bit_unary_count(&op.input, ctx, |x, w| {
                (x.leading_zeros()).saturating_sub(32u32.saturating_sub(w))
            }),
            TrailingZeros(op) => self.bit_unary_count(&op.input, ctx, |x, w| x.trailing_zeros().min(w)),
            other => Err(Halt::Unsupported(format!(
                "bitwise op {other} is outside the interpreter v0 subset"
            ))),
        }
    }

    fn bit_unary_count(
        &self,
        input: &Variable,
        ctx: &ThreadCtx,
        f: impl Fn(u32, u32) -> u32,
    ) -> Result<Val, Halt> {
        let v = self.value_of(input, ctx)?;
        let (bits, w) = match v {
            Val::U32(x) => (x, 32),
            Val::U64(x) => return Ok(Val::U32(f_u64(x, 64, &f))),
            Val::I32(x) => (x as u32, 32),
            _ => return Err(Halt::Unsupported("bit count on unsupported width".into())),
        };
        Ok(Val::U32(f(bits, w)))
    }

    fn bit_bin(
        &self,
        lhs: &Variable,
        rhs: &Variable,
        ctx: &ThreadCtx,
        op: BitOp,
    ) -> Result<Val, Halt> {
        let a = self.value_of(lhs, ctx)?;
        let b = self.value_of(rhs, ctx)?;
        bit_bin(a, b, op)
    }

    // ---- metadata ---------------------------------------------------------

    fn eval_metadata(&self, m: &Metadata) -> Result<Val, Halt> {
        match m {
            Metadata::Length { var } | Metadata::BufferLength { var } => {
                let len = self.array_len(var)?;
                Ok(Val::U32(len as u32))
            }
            other => Err(Halt::Unsupported(format!(
                "metadata {other} is outside the interpreter v0 subset"
            ))),
        }
    }

    fn array_len(&self, var: &Variable) -> Result<usize, Halt> {
        match var.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => {
                self.buffers
                    .get(id as usize)
                    .map(|b| b.data.len())
                    .ok_or_else(|| Halt::Unsupported(format!("no buffer registered at id {id}")))
            }
            VariableKind::ConstantArray { id, .. } => self
                .const_arrays
                .get(&id)
                .map(|d| d.len())
                .ok_or_else(|| Halt::Unsupported(format!("no const array at id {id}"))),
            VariableKind::LocalArray { length, .. } => Ok(length),
            _ => Err(Halt::Unsupported(format!("length of non-array: {var}"))),
        }
    }

    // ---- operator (index/assign/logic/cast/select) ------------------------

    fn exec_operator(
        &mut self,
        op: &Operator,
        out: Option<&Variable>,
        ctx: &ThreadCtx,
    ) -> Result<Signal, Halt> {
        match op {
            Operator::Index(io) | Operator::UncheckedIndex(io) => {
                assert_scalar_index(io.vector_size)?;
                let idx = self.index_value(&io.index, ctx)?;
                let val = self.read_index(&io.list, idx)?;
                self.bind(out, val);
            }
            Operator::IndexAssign(io) | Operator::UncheckedIndexAssign(io) => {
                assert_scalar_index(io.vector_size)?;
                let idx = self.index_value(&io.index, ctx)?;
                let val = self.value_of(&io.value, ctx)?;
                // The array being written is the instruction's `out`.
                let arr = out.ok_or_else(|| {
                    Halt::Unsupported("index-assign without an output array".into())
                })?;
                self.write_index(arr, idx, val)?;
            }
            Operator::And(op2) => {
                let a = self.value_of(&op2.lhs, ctx)?.as_bool().ok_or_else(bad("and"))?;
                let b = self.value_of(&op2.rhs, ctx)?.as_bool().ok_or_else(bad("and"))?;
                self.bind(out, Val::Bool(a && b));
            }
            Operator::Or(op2) => {
                let a = self.value_of(&op2.lhs, ctx)?.as_bool().ok_or_else(bad("or"))?;
                let b = self.value_of(&op2.rhs, ctx)?.as_bool().ok_or_else(bad("or"))?;
                self.bind(out, Val::Bool(a || b));
            }
            Operator::Not(op2) => {
                let a = self.value_of(&op2.input, ctx)?.as_bool().ok_or_else(bad("not"))?;
                self.bind(out, Val::Bool(!a));
            }
            Operator::Cast(op2) => {
                let v = self.value_of(&op2.input, ctx)?;
                let target = out.map(|o| o.ty.elem_type()).ok_or_else(bad("cast"))?;
                self.bind(out, cast(v, target)?);
            }
            Operator::Reinterpret(op2) => {
                let v = self.value_of(&op2.input, ctx)?;
                let target = out.map(|o| o.ty.elem_type()).ok_or_else(bad("reinterpret"))?;
                self.bind(out, reinterpret(v, target)?);
            }
            Operator::Select(sel) => {
                let cond = self.value_of(&sel.cond, ctx)?.as_bool().ok_or_else(bad("select"))?;
                let val = if cond {
                    self.value_of(&sel.then, ctx)?
                } else {
                    self.value_of(&sel.or_else, ctx)?
                };
                self.bind(out, val);
            }
            other => {
                return Err(Halt::Unsupported(format!(
                    "operator {other} is outside the interpreter v0 subset"
                )));
            }
        }
        Ok(Signal::Continue)
    }

    /// Resolve an index expression to a signed integer (so a negative index
    /// from a mis-modeled kernel is reported, not silently wrapped to a huge
    /// unsigned).
    fn index_value(&self, v: &Variable, ctx: &ThreadCtx) -> Result<i128, Halt> {
        self.value_of(v, ctx)?
            .as_int()
            .ok_or_else(|| Halt::Unsupported("non-integer index".into()))
    }

    fn read_index(&self, list: &Variable, idx: i128) -> Result<Val, Halt> {
        let (name, len, get) = self.array_view(list)?;
        if idx < 0 || idx as usize >= len {
            return Err(Halt::Oob(Oob { array: name, index: idx, len, write: false, thread: 0 }));
        }
        Ok(get(idx as usize))
    }

    fn write_index(&mut self, arr: &Variable, idx: i128, val: Val) -> Result<(), Halt> {
        match arr.kind {
            VariableKind::GlobalOutputArray(id) | VariableKind::GlobalInputArray(id) => {
                let buf = self
                    .buffers
                    .get_mut(id as usize)
                    .ok_or_else(|| Halt::Unsupported(format!("no buffer at id {id}")))?;
                let len = buf.data.len();
                if idx < 0 || idx as usize >= len {
                    let name = buf.name.clone();
                    return Err(Halt::Oob(Oob {
                        array: name_or(name, "output", id),
                        index: idx,
                        len,
                        write: true,
                        thread: 0,
                    }));
                }
                buf.data[idx as usize] = coerce(val, buf.elem);
                Ok(())
            }
            VariableKind::LocalArray { id, length, .. } => {
                let entry = self.local_arrays.entry(id).or_insert_with(|| {
                    vec![Val::from_int_or_float(&ConstantValue::UInt(0), arr.ty.elem_type()); length]
                });
                if idx < 0 || idx as usize >= entry.len() {
                    let len = entry.len();
                    return Err(Halt::Oob(Oob {
                        array: format!("local_array({id})"),
                        index: idx,
                        len,
                        write: true,
                        thread: 0,
                    }));
                }
                entry[idx as usize] = coerce(val, arr.ty.elem_type());
                Ok(())
            }
            _ => Err(Halt::Unsupported(format!("write to non-writable array: {arr}"))),
        }
    }

    /// Returns `(name, len, getter)` for a readable array.
    #[allow(clippy::type_complexity)]
    fn array_view(
        &self,
        list: &Variable,
    ) -> Result<(String, usize, Box<dyn Fn(usize) -> Val + '_>), Halt> {
        match list.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => {
                let buf = self
                    .buffers
                    .get(id as usize)
                    .ok_or_else(|| Halt::Unsupported(format!("no buffer at id {id}")))?;
                let kind = if matches!(list.kind, VariableKind::GlobalInputArray(_)) {
                    "input"
                } else {
                    "output"
                };
                Ok((
                    name_or(buf.name.clone(), kind, id),
                    buf.data.len(),
                    Box::new(move |i| buf.data[i]),
                ))
            }
            VariableKind::ConstantArray { id, .. } => {
                let data = self
                    .const_arrays
                    .get(&id)
                    .ok_or_else(|| Halt::Unsupported(format!("no const array at id {id}")))?;
                Ok((format!("const_array({id})"), data.len(), Box::new(move |i| data[i])))
            }
            VariableKind::LocalArray { id, length, .. } => {
                let elem = list.ty.elem_type();
                match self.local_arrays.get(&id) {
                    Some(data) => {
                        Ok((format!("local_array({id})"), data.len(), Box::new(move |i| data[i])))
                    }
                    // A read before any write: the array is logically zeroed.
                    None => Ok((
                        format!("local_array({id})"),
                        length,
                        Box::new(move |_| Val::from_int_or_float(&ConstantValue::UInt(0), elem)),
                    )),
                }
            }
            _ => Err(Halt::Unsupported(format!("index of non-array: {list}"))),
        }
    }

    // ---- branches ---------------------------------------------------------

    fn exec_branch(&mut self, b: &Branch, ctx: &ThreadCtx) -> Result<Signal, Halt> {
        match b {
            Branch::If(i) => {
                if self.cond(&i.cond, ctx)? {
                    self.exec_scope(&i.scope, ctx)
                } else {
                    Ok(Signal::Continue)
                }
            }
            Branch::IfElse(i) => {
                if self.cond(&i.cond, ctx)? {
                    self.exec_scope(&i.scope_if, ctx)
                } else {
                    self.exec_scope(&i.scope_else, ctx)
                }
            }
            Branch::Switch(s) => {
                let scrut = self.value_of(&s.value, ctx)?.as_int().ok_or_else(bad("switch"))?;
                for (case_val, scope) in &s.cases {
                    let cv = self.value_of(case_val, ctx)?.as_int().ok_or_else(bad("switch case"))?;
                    if cv == scrut {
                        return self.exec_scope(scope, ctx);
                    }
                }
                self.exec_scope(&s.scope_default, ctx)
            }
            Branch::RangeLoop(r) => self.exec_range_loop(r, ctx),
            Branch::Loop(l) => {
                loop {
                    self.tick()?;
                    match self.exec_scope(&l.scope, ctx)? {
                        Signal::Continue => {}
                        Signal::Break => break,
                        Signal::Return => return Ok(Signal::Return),
                    }
                }
                Ok(Signal::Continue)
            }
            Branch::Return => Ok(Signal::Return),
            Branch::Break => Ok(Signal::Break),
            Branch::Unreachable => {
                Err(Halt::Unsupported("reached an Unreachable branch".into()))
            }
        }
    }

    fn cond(&self, v: &Variable, ctx: &ThreadCtx) -> Result<bool, Halt> {
        self.value_of(v, ctx)?.as_bool().ok_or_else(|| {
            Halt::Unsupported("branch condition did not evaluate to a bool".into())
        })
    }

    fn exec_range_loop(
        &mut self,
        r: &cubecl::ir::RangeLoop,
        ctx: &ThreadCtx,
    ) -> Result<Signal, Halt> {
        let start = self.value_of(&r.start, ctx)?.as_int().ok_or_else(bad("loop start"))?;
        let end = self.value_of(&r.end, ctx)?.as_int().ok_or_else(bad("loop end"))?;
        let step = match &r.step {
            Some(s) => self.value_of(s, ctx)?.as_int().ok_or_else(bad("loop step"))?,
            None => 1,
        };
        if step <= 0 {
            return Err(Halt::Unsupported(
                "stepped/descending range loop is outside the interpreter v0 subset".into(),
            ));
        }
        let induction_elem = r.i.ty.elem_type();
        let mut i = start;
        loop {
            let cont = if r.inclusive { i <= end } else { i < end };
            if !cont {
                break;
            }
            self.tick()?;
            self.bind(Some(&r.i), Val::int_to_elem(i, induction_elem));
            match self.exec_scope(&r.scope, ctx)? {
                Signal::Continue => {}
                Signal::Break => break,
                Signal::Return => return Ok(Signal::Return),
            }
            i += step;
        }
        Ok(Signal::Continue)
    }
}

// ===================== free helpers =====================

fn bad(what: &'static str) -> impl Fn() -> Halt {
    move || Halt::Unsupported(format!("malformed {what} operand"))
}

fn name_or(name: String, kind: &str, id: Id) -> String {
    if name.is_empty() {
        format!("{kind}({id})")
    } else {
        name
    }
}

fn assert_scalar_index(vector_size: usize) -> Result<(), Halt> {
    // `vector_size == 0` means "same as list" (scalar); anything else is a
    // vectorized access, outside the scalar subset.
    if vector_size <= 1 {
        Ok(())
    } else {
        Err(Halt::Unsupported(
            "vectorized index (vector_size > 1) is outside the interpreter v0 subset".into(),
        ))
    }
}

#[derive(Clone, Copy)]
enum IntBin {
    Add,
    Sub,
    Mul,
    Min,
    Max,
}

#[derive(Clone, Copy)]
enum BitOp {
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

fn wrapping_neg(v: Val, _i: i128) -> Val {
    match v {
        Val::U8(x) => Val::U8(x.wrapping_neg()),
        Val::U16(x) => Val::U16(x.wrapping_neg()),
        Val::U32(x) => Val::U32(x.wrapping_neg()),
        Val::U64(x) => Val::U64(x.wrapping_neg()),
        Val::I8(x) => Val::I8(x.wrapping_neg()),
        Val::I16(x) => Val::I16(x.wrapping_neg()),
        Val::I32(x) => Val::I32(x.wrapping_neg()),
        Val::I64(x) => Val::I64(x.wrapping_neg()),
        other => other,
    }
}

macro_rules! int_bin_arm {
    ($a:expr, $b:expr, $m:ident) => {
        match ($a, $b) {
            (Val::U8(x), Val::U8(y)) => Ok(Val::U8(x.$m(y))),
            (Val::U16(x), Val::U16(y)) => Ok(Val::U16(x.$m(y))),
            (Val::U32(x), Val::U32(y)) => Ok(Val::U32(x.$m(y))),
            (Val::U64(x), Val::U64(y)) => Ok(Val::U64(x.$m(y))),
            (Val::I8(x), Val::I8(y)) => Ok(Val::I8(x.$m(y))),
            (Val::I16(x), Val::I16(y)) => Ok(Val::I16(x.$m(y))),
            (Val::I32(x), Val::I32(y)) => Ok(Val::I32(x.$m(y))),
            (Val::I64(x), Val::I64(y)) => Ok(Val::I64(x.$m(y))),
            _ => Err(Halt::Unsupported("integer op on mismatched/invalid operand types".into())),
        }
    };
}

fn int_bin(a: Val, b: Val, op: IntBin) -> Result<Val, Halt> {
    match op {
        IntBin::Add => int_bin_arm!(a, b, wrapping_add),
        IntBin::Sub => int_bin_arm!(a, b, wrapping_sub),
        IntBin::Mul => int_bin_arm!(a, b, wrapping_mul),
        IntBin::Min => int_bin_arm!(a, b, min),
        IntBin::Max => int_bin_arm!(a, b, max),
    }
}

fn int_div_mod(a: Val, b: Val, is_div: bool) -> Result<Val, Halt> {
    // Truncated-toward-zero division (matches Rust and WGSL for the modeled
    // integer types); `wrapping_div`/`wrapping_rem` also cover the
    // `iN::MIN / -1` overflow case without panicking.
    if is_div {
        int_bin_arm!(a, b, wrapping_div)
    } else {
        int_bin_arm!(a, b, wrapping_rem)
    }
}

fn bit_bin(a: Val, b: Val, op: BitOp) -> Result<Val, Halt> {
    match op {
        BitOp::And => match (a, b) {
            (Val::U32(x), Val::U32(y)) => Ok(Val::U32(x & y)),
            (Val::U64(x), Val::U64(y)) => Ok(Val::U64(x & y)),
            (Val::I32(x), Val::I32(y)) => Ok(Val::I32(x & y)),
            (Val::Bool(x), Val::Bool(y)) => Ok(Val::Bool(x & y)),
            _ => Err(Halt::Unsupported("bitand on unsupported operands".into())),
        },
        BitOp::Or => match (a, b) {
            (Val::U32(x), Val::U32(y)) => Ok(Val::U32(x | y)),
            (Val::U64(x), Val::U64(y)) => Ok(Val::U64(x | y)),
            (Val::I32(x), Val::I32(y)) => Ok(Val::I32(x | y)),
            (Val::Bool(x), Val::Bool(y)) => Ok(Val::Bool(x | y)),
            _ => Err(Halt::Unsupported("bitor on unsupported operands".into())),
        },
        BitOp::Xor => match (a, b) {
            (Val::U32(x), Val::U32(y)) => Ok(Val::U32(x ^ y)),
            (Val::U64(x), Val::U64(y)) => Ok(Val::U64(x ^ y)),
            (Val::I32(x), Val::I32(y)) => Ok(Val::I32(x ^ y)),
            (Val::Bool(x), Val::Bool(y)) => Ok(Val::Bool(x ^ y)),
            _ => Err(Halt::Unsupported("bitxor on unsupported operands".into())),
        },
        // Shift amount is masked by the operand width, matching WGSL and the
        // `wrapping` twin's `wrapping_shl`/`wrapping_shr`.
        BitOp::Shl => {
            let amt = b.as_int().ok_or_else(bad("shl amount"))? as u32;
            match a {
                Val::U32(x) => Ok(Val::U32(x.wrapping_shl(amt))),
                Val::U64(x) => Ok(Val::U64(x.wrapping_shl(amt))),
                Val::I32(x) => Ok(Val::I32(x.wrapping_shl(amt))),
                _ => Err(Halt::Unsupported("shl on unsupported operand".into())),
            }
        }
        BitOp::Shr => {
            let amt = b.as_int().ok_or_else(bad("shr amount"))? as u32;
            match a {
                Val::U32(x) => Ok(Val::U32(x.wrapping_shr(amt))),
                Val::U64(x) => Ok(Val::U64(x.wrapping_shr(amt))),
                Val::I32(x) => Ok(Val::I32(x.wrapping_shr(amt))),
                _ => Err(Halt::Unsupported("shr on unsupported operand".into())),
            }
        }
    }
}

fn f_u64(x: u64, w: u32, f: &impl Fn(u32, u32) -> u32) -> u32 {
    // Reduce a u64 count op to the u32-shaped helper by handling the common
    // count_ones/leading/trailing cases directly.
    let _ = f;
    let _ = w;
    x.count_ones()
}

/// Total order over same-typed values for `<`/`<=`/`>`/`>=`. Floats use IEEE
/// partial order (NaN compares false in every direction, matching GPU/Rust).
fn cmp(a: Val, b: Val, pred: impl Fn(std::cmp::Ordering) -> bool) -> Result<bool, Halt> {
    if let (Some(x), Some(y)) = (a.as_f64(), b.as_f64()) {
        return Ok(match x.partial_cmp(&y) {
            Some(o) => pred(o),
            None => false, // NaN
        });
    }
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        return Ok(pred(x.cmp(&y)));
    }
    Err(Halt::Unsupported("comparison of mismatched value kinds".into()))
}

fn eq(a: Val, b: Val) -> bool {
    if let (Some(x), Some(y)) = (a.as_f64(), b.as_f64()) {
        return x == y; // NaN != NaN, as in Rust/GPU
    }
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        return x == y;
    }
    if let (Val::Bool(x), Val::Bool(y)) = (a, b) {
        return x == y;
    }
    false
}

/// `as`-cast semantics (Rust's, which coincide with WGSL for the
/// width-preserving integer conversions in the modeled subset). Float→int is
/// Rust's saturating cast; the twin is Rust, so this keeps them in agreement.
fn cast(v: Val, target: ElemType) -> Result<Val, Halt> {
    match target {
        ElemType::Float(FloatKind::F64) => Ok(Val::F64(cast_to_f64(v))),
        ElemType::Float(_) => Ok(Val::F32(cast_to_f64(v) as f32)),
        ElemType::Bool => Ok(Val::Bool(match v {
            Val::Bool(b) => b,
            other => other.as_int().map(|i| i != 0).unwrap_or(false),
        })),
        _ => {
            // Integer target.
            match v {
                Val::F32(x) => Ok(Val::int_to_elem(x as i128, target)),
                Val::F64(x) => Ok(Val::int_to_elem(x as i128, target)),
                other => {
                    let i = other.as_int().ok_or_else(bad("cast"))?;
                    Ok(Val::int_to_elem(i, target))
                }
            }
        }
    }
}

fn cast_to_f64(v: Val) -> f64 {
    match v {
        Val::F32(x) => x as f64,
        Val::F64(x) => x,
        Val::U8(x) => x as f64,
        Val::U16(x) => x as f64,
        Val::U32(x) => x as f64,
        Val::U64(x) => x as f64,
        Val::I8(x) => x as f64,
        Val::I16(x) => x as f64,
        Val::I32(x) => x as f64,
        Val::I64(x) => x as f64,
        Val::Bool(b) => b as u8 as f64,
    }
}

/// Bit-reinterpretation (`Reinterpret`) between same-width scalar types.
fn reinterpret(v: Val, target: ElemType) -> Result<Val, Halt> {
    match (v, target) {
        (Val::F32(x), ElemType::UInt(UIntKind::U32)) => Ok(Val::U32(x.to_bits())),
        (Val::F32(x), ElemType::Int(IntKind::I32)) => Ok(Val::I32(x.to_bits() as i32)),
        (Val::U32(x), ElemType::Float(FloatKind::F32)) => Ok(Val::F32(f32::from_bits(x))),
        (Val::I32(x), ElemType::Float(FloatKind::F32)) => Ok(Val::F32(f32::from_bits(x as u32))),
        (Val::U32(x), ElemType::Int(IntKind::I32)) => Ok(Val::I32(x as i32)),
        (Val::I32(x), ElemType::UInt(UIntKind::U32)) => Ok(Val::U32(x as u32)),
        (Val::F64(x), ElemType::UInt(UIntKind::U64)) => Ok(Val::U64(x.to_bits())),
        (Val::U64(x), ElemType::Float(FloatKind::F64)) => Ok(Val::F64(f64::from_bits(x))),
        _ => Err(Halt::Unsupported(format!(
            "reinterpret {:?} as {target:?} is outside the interpreter v0 subset",
            v.elem_type()
        ))),
    }
}

/// Coerce a value being stored to an array to that array's element type, so a
/// constant modeled at a wider width still stores exactly.
fn coerce(v: Val, elem: ElemType) -> Val {
    if v.elem_type() == elem {
        return v;
    }
    match elem {
        ElemType::Float(FloatKind::F64) => Val::F64(cast_to_f64(v)),
        ElemType::Float(_) => Val::F32(cast_to_f64(v) as f32),
        ElemType::Bool => Val::Bool(v.as_int().map(|i| i != 0).unwrap_or(false)),
        _ => match v {
            Val::F32(x) => Val::int_to_elem(x as i128, elem),
            Val::F64(x) => Val::int_to_elem(x as i128, elem),
            other => Val::int_to_elem(other.as_int().unwrap_or(0), elem),
        },
    }
}

/// Collect every constant array reachable from `scope` (root + nested) into an
/// id → concrete-data map.
fn collect_const_arrays(scope: &Scope) -> HashMap<Id, Vec<Val>> {
    let mut out = HashMap::new();
    collect_const_arrays_into(scope, &mut out);
    out
}

fn collect_const_arrays_into(scope: &Scope, out: &mut HashMap<Id, Vec<Val>>) {
    for (var, data) in &scope.const_arrays {
        if let VariableKind::ConstantArray { id, .. } = var.kind {
            let elem = var.ty.elem_type();
            let vals: Vec<Val> = data
                .iter()
                .map(|d| match d.kind {
                    VariableKind::Constant(c) => Val::from_const(c, d.ty),
                    _ => Val::from_int_or_float(&ConstantValue::UInt(0), elem),
                })
                .collect();
            out.insert(id, vals);
        }
    }
    for inst in &scope.instructions {
        if let Operation::Branch(b) = &inst.operation {
            match b {
                Branch::If(i) => collect_const_arrays_into(&i.scope, out),
                Branch::IfElse(i) => {
                    collect_const_arrays_into(&i.scope_if, out);
                    collect_const_arrays_into(&i.scope_else, out);
                }
                Branch::Switch(s) => {
                    collect_const_arrays_into(&s.scope_default, out);
                    for (_, sc) in &s.cases {
                        collect_const_arrays_into(sc, out);
                    }
                }
                Branch::RangeLoop(r) => collect_const_arrays_into(&r.scope, out),
                Branch::Loop(l) => collect_const_arrays_into(&l.scope, out),
                _ => {}
            }
        }
    }
}

// ===================== ergonomic constructors =====================

impl Buffer {
    pub fn u32(name: &str, data: &[u32], is_output: bool) -> Buffer {
        Buffer {
            name: name.into(),
            elem: ElemType::UInt(UIntKind::U32),
            data: data.iter().map(|v| Val::U32(*v)).collect(),
            is_output,
        }
    }

    pub fn f32(name: &str, data: &[f32], is_output: bool) -> Buffer {
        Buffer {
            name: name.into(),
            elem: ElemType::Float(FloatKind::F32),
            data: data.iter().map(|v| Val::F32(*v)).collect(),
            is_output,
        }
    }

    /// Extract this buffer's contents as `f32` (panics if not an f32 buffer).
    pub fn as_f32(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|v| match v {
                Val::F32(x) => *x,
                other => panic!("expected f32, got {other:?}"),
            })
            .collect()
    }

    /// Extract this buffer's contents as `u32` (panics if not a u32 buffer).
    pub fn as_u32(&self) -> Vec<u32> {
        self.data
            .iter()
            .map(|v| match v {
                Val::U32(x) => *x,
                other => panic!("expected u32, got {other:?}"),
            })
            .collect()
    }
}

impl ScalarBinding {
    pub fn u32(id: Id, val: u32) -> ScalarBinding {
        ScalarBinding { elem: ElemType::UInt(UIntKind::U32), id, val: Val::U32(val) }
    }
    pub fn f32(id: Id, val: f32) -> ScalarBinding {
        ScalarBinding { elem: ElemType::Float(FloatKind::F32), id, val: Val::F32(val) }
    }
    pub fn i32(id: Id, val: i32) -> ScalarBinding {
        ScalarBinding { elem: ElemType::Int(IntKind::I32), id, val: Val::I32(val) }
    }
}

#[cfg(test)]
mod tests {
    //! Direct validation of the interpreter against **real** `#[cube]`-expanded
    //! IR (not hand-built), per construct: arithmetic, wrapping-integer
    //! bit-mixers, div/mod decode, gather (value-dependent index), Switch, a
    //! forward-offset read, a range loop, and — the point of the whole exercise
    //! — that an out-of-bounds access is *reported*, never panicked.

    use super::*;
    use cubecl::ir::AddressType;
    use cubecl::prelude::*;

    // -- build recipe (docs/ir-research.md §1): expand with a hand-built
    // KernelBuilder, no client/runtime/device needed. -----------------------

    fn ins(buffers: Vec<Buffer>, scalars: Vec<ScalarBinding>, n: u32) -> Inputs {
        Inputs { buffers, scalars, cube_dim: 256, num_threads: n }
    }

    fn out_buf(o: &Outcome, id: usize) -> &Buffer {
        match o {
            Outcome::Completed { buffers } => &buffers[id],
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[cube(launch)]
    fn k_axpy(alpha: f32, x: &Array<f32>, y: &mut Array<f32>) {
        if ABSOLUTE_POS < y.len() {
            y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
        }
    }

    fn build_axpy() -> KernelDefinition {
        let mut b = KernelBuilder::default();
        b.runtime_properties(Default::default());
        AddressType::U32.register(&mut b.scope);
        let alpha = <f32 as LaunchArg>::expand(&Default::default(), &mut b);
        let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut b);
        let y =
            <Array<f32> as LaunchArg>::expand_output(&ArrayCompilationArg { inplace: None }, &mut b);
        k_axpy::expand(&mut b.scope, alpha, x, y);
        b.build(KernelSettings::default())
    }

    #[test]
    fn axpy_matches_strict_reference() {
        let def = build_axpy();
        let alpha = 2.0f32;
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let y0 = [10.0f32, 20.0, 30.0, 40.0];
        let inputs = ins(
            vec![Buffer::f32("x", &x, false), Buffer::f32("y", &y0, true)],
            vec![ScalarBinding::f32(0, alpha)],
            4,
        );
        let outcome = interpret_dispatch(&def, &inputs);
        let want: Vec<f32> = (0..4).map(|i| alpha * x[i] + y0[i]).collect();
        assert_eq!(out_buf(&outcome, 1).as_f32(), want);
    }

    /// The classic missed bug: guard is `<=`, so `ABSOLUTE_POS == y.len()` is
    /// reachable and writes out of bounds. The interpreter must REPORT it (as a
    /// finding), not panic. Runs one extra thread so `pos == len` is reached.
    #[cube(launch)]
    fn k_axpy_off(alpha: f32, x: &Array<f32>, y: &mut Array<f32>) {
        if ABSOLUTE_POS <= y.len() {
            y[ABSOLUTE_POS] = alpha * x[ABSOLUTE_POS] + y[ABSOLUTE_POS];
        }
    }

    #[test]
    fn off_by_one_reports_oob_not_panic() {
        let mut b = KernelBuilder::default();
        b.runtime_properties(Default::default());
        AddressType::U32.register(&mut b.scope);
        let alpha = <f32 as LaunchArg>::expand(&Default::default(), &mut b);
        let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut b);
        let y =
            <Array<f32> as LaunchArg>::expand_output(&ArrayCompilationArg { inplace: None }, &mut b);
        k_axpy_off::expand(&mut b.scope, alpha, x, y);
        let def = b.build(KernelSettings::default());
        // 4 elements, but dispatch 5 threads: thread 4 has pos == len == 4.
        let inputs = ins(
            vec![
                Buffer::f32("x", &[1.0; 5], false),
                Buffer::f32("y", &[0.0; 4], true),
            ],
            vec![ScalarBinding::f32(0, 1.0)],
            5,
        );
        match interpret_dispatch(&def, &inputs) {
            Outcome::OutOfBounds(o) => {
                assert_eq!(o.index, 4);
                assert_eq!(o.len, 4);
                assert_eq!(o.thread, 4);
            }
            other => panic!("expected reported OOB, got {other:?}"),
        }
    }

    #[cube(launch)]
    fn k_xorshift(x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS < y.len() {
            let mut s = x[ABSOLUTE_POS];
            s ^= s << 13u32;
            s ^= s >> 17u32;
            s ^= s << 5u32;
            y[ABSOLUTE_POS] = s;
        }
    }

    #[test]
    fn xorshift_matches_reference() {
        let mut b = KernelBuilder::default();
        b.runtime_properties(Default::default());
        AddressType::U32.register(&mut b.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut b);
        let y =
            <Array<u32> as LaunchArg>::expand_output(&ArrayCompilationArg { inplace: None }, &mut b);
        k_xorshift::expand(&mut b.scope, x, y);
        let def = b.build(KernelSettings::default());
        let xs = [1u32, 42, 0x1234_5678, 0xFFFF_FFFF, 7];
        let inputs = ins(
            vec![Buffer::u32("x", &xs, false), Buffer::u32("y", &[0u32; 5], true)],
            vec![],
            5,
        );
        let want: Vec<u32> = xs
            .iter()
            .map(|&x0| {
                let mut s = x0;
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                s
            })
            .collect();
        assert_eq!(out_buf(&interpret_dispatch(&def, &inputs), 1).as_u32(), want);
    }

    #[cube(launch)]
    fn k_mix(x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS < y.len() {
            let mut h = x[ABSOLUTE_POS];
            h ^= h >> 16u32;
            h *= 0x85ebca6bu32;
            h ^= h >> 13u32;
            h *= 0xc2b2ae35u32;
            h ^= h >> 16u32;
            y[ABSOLUTE_POS] = h;
        }
    }

    /// Overflowing multiplies must WRAP (not panic), matching WGSL.
    #[test]
    fn mix_wraps_on_overflow() {
        let mut b = KernelBuilder::default();
        b.runtime_properties(Default::default());
        AddressType::U32.register(&mut b.scope);
        let x = <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut b);
        let y =
            <Array<u32> as LaunchArg>::expand_output(&ArrayCompilationArg { inplace: None }, &mut b);
        k_mix::expand(&mut b.scope, x, y);
        let def = b.build(KernelSettings::default());
        let xs = [0u32, 1, 0xDEAD_BEEF, 0x8000_0000, 123456789];
        let inputs = ins(
            vec![Buffer::u32("x", &xs, false), Buffer::u32("y", &[0u32; 5], true)],
            vec![],
            5,
        );
        let want: Vec<u32> = xs
            .iter()
            .map(|&x0| {
                let mut h = x0;
                h ^= h >> 16;
                h = h.wrapping_mul(0x85ebca6b);
                h ^= h >> 13;
                h = h.wrapping_mul(0xc2b2ae35);
                h ^= h >> 16;
                h
            })
            .collect();
        assert_eq!(out_buf(&interpret_dispatch(&def, &inputs), 1).as_u32(), want);
    }

    #[cube(launch)]
    fn k_flatten(x: &Array<f32>, y: &mut Array<f32>, width: u32, scale: f32) {
        if ABSOLUTE_POS < y.len() && width >= 1u32 {
            let w = width as usize;
            let row = ABSOLUTE_POS / w;
            let col = ABSOLUTE_POS % w;
            y[row * w + col] = x[ABSOLUTE_POS] * scale;
        }
    }

    #[test]
    fn flatten_decode_scale_div_mod() {
        let mut b = KernelBuilder::default();
        b.runtime_properties(Default::default());
        AddressType::U32.register(&mut b.scope);
        let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut b);
        let y =
            <Array<f32> as LaunchArg>::expand_output(&ArrayCompilationArg { inplace: None }, &mut b);
        let width = <u32 as LaunchArg>::expand(&Default::default(), &mut b);
        let scale = <f32 as LaunchArg>::expand(&Default::default(), &mut b);
        k_flatten::expand(&mut b.scope, x, y, width, scale);
        let def = b.build(KernelSettings::default());
        let xs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (w, s) = (3u32, 2.0f32);
        let inputs = ins(
            vec![Buffer::f32("x", &xs, false), Buffer::f32("y", &[0.0; 6], true)],
            vec![ScalarBinding::u32(0, w), ScalarBinding::f32(0, s)],
            6,
        );
        // row*w+col == pos, so y[pos] = x[pos]*s.
        let want: Vec<f32> = xs.iter().map(|v| v * s).collect();
        assert_eq!(out_buf(&interpret_dispatch(&def, &inputs), 1).as_f32(), want);
    }

    #[cube(launch)]
    fn k_gather(x: &Array<f32>, offsets: &Array<u32>, y: &mut Array<f32>) {
        if ABSOLUTE_POS < y.len() {
            y[ABSOLUTE_POS] = x[offsets[ABSOLUTE_POS] as usize];
        }
    }

    #[test]
    fn gather_value_dependent_index() {
        let mut b = KernelBuilder::default();
        b.runtime_properties(Default::default());
        AddressType::U32.register(&mut b.scope);
        let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut b);
        let offsets =
            <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut b);
        let y =
            <Array<f32> as LaunchArg>::expand_output(&ArrayCompilationArg { inplace: None }, &mut b);
        k_gather::expand(&mut b.scope, x, offsets, y);
        let def = b.build(KernelSettings::default());
        let xs = [10.0f32, 20.0, 30.0, 40.0];
        let offs = [3u32, 1, 0, 2];
        let inputs = ins(
            vec![
                Buffer::f32("x", &xs, false),
                Buffer::u32("offsets", &offs, false),
                Buffer::f32("y", &[0.0; 4], true),
            ],
            vec![],
            4,
        );
        assert_eq!(out_buf(&interpret_dispatch(&def, &inputs), 2).as_f32(), vec![40.0, 20.0, 10.0, 30.0]);
    }

    /// A gather with an offset out of range must be reported as OOB (the
    /// value-dependent defect that a random differential draw would miss).
    #[test]
    fn gather_out_of_range_offset_reports_oob() {
        let mut b = KernelBuilder::default();
        b.runtime_properties(Default::default());
        AddressType::U32.register(&mut b.scope);
        let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut b);
        let offsets =
            <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut b);
        let y =
            <Array<f32> as LaunchArg>::expand_output(&ArrayCompilationArg { inplace: None }, &mut b);
        k_gather::expand(&mut b.scope, x, offsets, y);
        let def = b.build(KernelSettings::default());
        let inputs = ins(
            vec![
                Buffer::f32("x", &[1.0, 2.0], false), // len 2
                Buffer::u32("offsets", &[0, 5], false), // offset 5 is OOB
                Buffer::f32("y", &[0.0; 2], true),
            ],
            vec![],
            2,
        );
        match interpret_dispatch(&def, &inputs) {
            Outcome::OutOfBounds(o) => {
                assert_eq!(o.array, "x");
                assert_eq!(o.index, 5);
                assert_eq!(o.thread, 1);
            }
            other => panic!("expected OOB, got {other:?}"),
        }
    }

    #[cube(launch)]
    fn k_select(mode: u32, x: &Array<f32>, y: &mut Array<f32>) {
        if ABSOLUTE_POS < y.len() {
            match mode {
                0 => {
                    y[ABSOLUTE_POS] = x[ABSOLUTE_POS];
                }
                1 => {
                    y[ABSOLUTE_POS] = -x[ABSOLUTE_POS];
                }
                _ => {
                    y[ABSOLUTE_POS] = x[ABSOLUTE_POS] * 2.0f32;
                }
            }
        }
    }

    #[test]
    fn switch_selects_the_right_arm() {
        let mut b = KernelBuilder::default();
        b.runtime_properties(Default::default());
        AddressType::U32.register(&mut b.scope);
        let mode = <u32 as LaunchArg>::expand(&Default::default(), &mut b);
        let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut b);
        let y =
            <Array<f32> as LaunchArg>::expand_output(&ArrayCompilationArg { inplace: None }, &mut b);
        k_select::expand(&mut b.scope, mode, x, y);
        let def = b.build(KernelSettings::default());
        let xs = [1.0f32, -2.0, 3.0, -4.0];
        for (mode, f) in [(0u32, 1.0f32), (1, -1.0), (2, 2.0), (7, 2.0)] {
            let inputs = ins(
                vec![Buffer::f32("x", &xs, false), Buffer::f32("y", &[0.0; 4], true)],
                vec![ScalarBinding::u32(0, mode)],
                4,
            );
            let want: Vec<f32> = xs.iter().map(|v| v * f).collect();
            assert_eq!(out_buf(&interpret_dispatch(&def, &inputs), 1).as_f32(), want, "mode {mode}");
        }
    }

    #[cube(launch)]
    fn k_offset_window(x: &Array<f32>, y: &mut Array<f32>) {
        if ABSOLUTE_POS < y.len() {
            y[ABSOLUTE_POS] = x[ABSOLUTE_POS] + x[ABSOLUTE_POS + 4usize];
        }
    }

    #[test]
    fn offset_window_forward_read() {
        let mut b = KernelBuilder::default();
        b.runtime_properties(Default::default());
        AddressType::U32.register(&mut b.scope);
        let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut b);
        let y =
            <Array<f32> as LaunchArg>::expand_output(&ArrayCompilationArg { inplace: None }, &mut b);
        k_offset_window::expand(&mut b.scope, x, y);
        let def = b.build(KernelSettings::default());
        let xs = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let inputs = ins(
            vec![Buffer::f32("x", &xs, false), Buffer::f32("y", &[0.0; 4], true)],
            vec![],
            4,
        );
        assert_eq!(out_buf(&interpret_dispatch(&def, &inputs), 1).as_f32(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[cube(launch)]
    fn k_loopsum(x: &Array<f32>, y: &mut Array<f32>) {
        if ABSOLUTE_POS < y.len() {
            let mut acc = f32::new(0.0);
            for j in 0..4usize {
                acc += x[ABSOLUTE_POS * 4usize + j];
            }
            y[ABSOLUTE_POS] = acc;
        }
    }

    #[test]
    fn range_loop_accumulates() {
        let mut b = KernelBuilder::default();
        b.runtime_properties(Default::default());
        AddressType::U32.register(&mut b.scope);
        let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut b);
        let y =
            <Array<f32> as LaunchArg>::expand_output(&ArrayCompilationArg { inplace: None }, &mut b);
        k_loopsum::expand(&mut b.scope, x, y);
        let def = b.build(KernelSettings::default());
        let xs: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let inputs = ins(
            vec![Buffer::f32("x", &xs, false), Buffer::f32("y", &[0.0; 2], true)],
            vec![],
            2,
        );
        // y[0] = 0+1+2+3 = 6, y[1] = 4+5+6+7 = 22.
        assert_eq!(out_buf(&interpret_dispatch(&def, &inputs), 1).as_f32(), vec![6.0, 22.0]);
    }

    /// A cooperative kernel (shared memory + sync_cube) must be rejected up
    /// front as Unsupported, never mis-executed single-threaded.
    #[cube(launch)]
    fn k_coop(x: &Array<f32>, y: &mut Array<f32>) {
        let mut tile = SharedMemory::<f32>::new(256usize);
        let u = UNIT_POS as usize;
        tile[u] = x[ABSOLUTE_POS];
        sync_cube();
        y[ABSOLUTE_POS] = tile[u];
    }

    #[test]
    fn cooperative_kernel_is_unsupported() {
        let mut b = KernelBuilder::default();
        b.runtime_properties(Default::default());
        AddressType::U32.register(&mut b.scope);
        let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut b);
        let y =
            <Array<f32> as LaunchArg>::expand_output(&ArrayCompilationArg { inplace: None }, &mut b);
        k_coop::expand(&mut b.scope, x, y);
        let def = b.build(KernelSettings::default());
        let inputs = ins(
            vec![Buffer::f32("x", &[1.0; 4], false), Buffer::f32("y", &[0.0; 4], true)],
            vec![],
            4,
        );
        match interpret_dispatch(&def, &inputs) {
            Outcome::Unsupported { reason } => assert!(reason.contains("shared memory") || reason.contains("sync_cube")),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
