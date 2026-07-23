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
//! - **Switch modeling (match on integers).** CubeCL 0.10 lowers a Rust
//!   `match` on an integer scrutinee to a `Branch::Switch { value,
//!   scope_default, cases: Vec<(Variable, Scope)> }` — verified by extracting
//!   IR for a guarded `match mode { 0 => …, 1 | 2 => …, _ => … }` kernel: the
//!   scrutinee is the modeled `value`, each case value is an integer-literal
//!   `Constant` (an `Or`-pattern such as `1 | 2` becomes two separate cases
//!   sharing a cloned body), and the `_` arm is `scope_default` (cubecl's
//!   `numeric_match` requires a wildcard default, so `scope_default` is always
//!   present and covers every un-listed scrutinee value). `process_switch`
//!   models it as an **exhaustive if-chain**: each case arm is walked under the
//!   path condition `value == case_i`, and the default under the conjunction of
//!   all `value != case_i`. This is exact — the case conditions are mutually
//!   exclusive constants and the default's condition is precisely their joint
//!   negation, so a case set that covers the guarded range makes the default's
//!   path condition unsatisfiable under the live facts (its obligations then
//!   discharge vacuously, correctly, because that arm is genuinely unreachable
//!   for those inputs). Branch-scoped write taint is the **same** machinery as
//!   `If`/`IfElse`, only generalized from 2 arms to N+1: `self.memo` is
//!   snapshotted before each arm, restored between arms (so no arm sees
//!   another's writes), and after the construct every variable written in *any*
//!   arm (tracked through the shared `write_log_stack`/`set_var`) is tainted
//!   — v0 does no cross-arm value merge, so a per-arm write can never leak past
//!   the switch's close (the round-2 `IfElse` manifestations, replayed through
//!   Switch, refute rather than false-`Proved`). A tainted scrutinee, or a case
//!   value that is not a modeled constant, takes the whole switch
//!   `OutOfSubset` (same rule as a tainted `if` condition), never mismodeled.
//!   Race walk: each arm's condition is pushed via `race_push_guard` keyed on
//!   the scrutinee `sw.value`, so a `sync_cube()` inside a switch arm sits under
//!   a non-loop guard and is rejected by the *same* barrier-uniformity gate as
//!   any other conditional barrier — a thread-varying scrutinee is barrier
//!   divergence, a uniform one is a (v1.1-deferred) conditional barrier; both
//!   `OutOfSubset`, never a silent `Proved`.
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
//! - **Bounded-integer overflow model (the soundness foundation).** Integer
//!   values are modeled in QF_LIA (unbounded `Int`), but the encoding is made
//!   *faithful to finite-width hardware wraparound*: the invariant is that
//!   **every non-tainted modeled integer term equals the real (wrapping)
//!   hardware value at every input.** This is what makes every downstream
//!   consumer sound without a per-consumer case analysis — an index/bounds
//!   obligation, a div/mod divisor, a branch or loop guard, a loop bound, a
//!   race index, an element-assume bound all read the *true* value. A term
//!   that could diverge from hardware is instead tainted, and the existing
//!   taint discipline already fails, explicitly, at whichever consumer needs
//!   it. Concretely:
//!   * **Leaves** (`ABSOLUTE_POS`/`UNIT_POS`/`CUBE_POS`/`CUBE_COUNT`, buffer
//!     `Length`s, integer `GlobalScalar`s, loop induction vars, element-read
//!     symbols) are declared *in their type's range* `[type_min, type_max]`
//!     (`declare_leaf`) — a sound type fact (a `u32` value really is in
//!     `[0, 2^32)`; positions/lengths are `u32` per `AddressType::U32`). This
//!     is load-bearing twice over: it lets the no-overflow side-obligations
//!     below discharge for genuinely-safe arithmetic, and it keeps the
//!     single-wrap `ite` below exact. In *cooperative* mode `ABSOLUTE_POS` is
//!     not a free leaf but a `u32`-range leaf additionally pinned to the exact
//!     modular recomposition `(CubePos*cube_dim + UnitPos) mod 2^32`
//!     (`abs_pos_sym`, "Cooperative mode" bullet) — still the real hardware
//!     value, so the invariant holds; a raw unwrapped `CubePos*cube_dim +
//!     UnitPos` would break it (round 5).
//!   * **`Add`/`Sub`** are modeled *exactly* under wraparound
//!     (`wrap_to_range`): the mathematical result is folded back into range
//!     via `ite(raw > max, raw − 2^W, ite(raw < min, raw + 2^W, raw))`. Since
//!     both operands are in range, the result is at most one modulus outside
//!     it in either direction, so a single correction is exact. So `pos + 1`,
//!     `pos − 1`, `row*w + col` are the true wrapped values — e.g. an
//!     unguarded `x[pos − 1]` refutes with the real wrapped index `2^32 − 1`
//!     at `pos == 0`, not a spurious `−1`.
//!   * **`Mul`** can wrap up to `2^W` times, and `(a*b) mod 2^W` is nonlinear
//!     (QF_NIA-hard), so instead of a faithful term the product carries a
//!     no-overflow *side-obligation* `type_min ≤ a*b ≤ type_max` under the
//!     live path conditions (`checked_mul`, same discipline as div/mod).
//!     Discharged ⟹ the plain `*` term provably does not wrap and equals the
//!     real value — bind it. Not discharged ⟹ the product may wrap — taint.
//!   * **`Div`/`Modulo`** need no output overflow check (`a/b ≤ a`, `a%b < b`
//!     stay in range for nonnegative operands); their existing nonzero +
//!     nonnegativity side-obligation is now *sound under wraparound* because
//!     the divisor term equals its real value.
//!   * **`Cast` (int→int)** passes a value-preserving cast (widening / equal
//!     width without a sign reinterpretation, `cast_is_value_preserving`)
//!     through unchanged; a narrowing or same-width sign-flip cast can
//!     truncate/reinterpret, so it passes through only when a "fits the
//!     destination range" side-obligation discharges (`cast_int`) — else
//!     taint. (Every cast in the current example suite is `u32 as usize`,
//!     i.e. `u32 → u32`, value-preserving.)
//!
//!   The side-obligations (`checked_mul`, `cast_int`) are internal modeling
//!   preconditions, deliberately *not* counted in `Prover::obligations`.
//!
//!   *Why this over full QF_BV (approach (a)).* QF_BV would model wraparound
//!   for free but rewrite every existing encoding — bounds obligations, the
//!   length/element/gather assumes, the two-thread race walk, the cooperative
//!   leaves, div/mod — in a file hardened across four adversarial-review
//!   rounds, and thread signed-vs-unsigned bitvector-comparison discipline
//!   through every comparison. Approach (b) here preserves every existing
//!   encoding: the change is confined to leaf declaration and the three
//!   arithmetic handlers (plus `Cast`), composes with all the machinery above
//!   unchanged, and closes exactly the round-2 construction — a `u32` product
//!   provably nonzero in unbounded LIA (`a,b ≥ 1 ⟹ a*b ≥ 1`) that wraps to
//!   `65536 * 65536 == 2^32 ≡ 0` on hardware: `checked_mul`'s side-obligation
//!   fails, the divisor taints, and the dependent access is `OutOfSubset`,
//!   never `Proved`. The `a + b == 2^32` variant is caught too — faithful
//!   `Add` gives the divisor term `0`, so the div/mod nonzero check fails.
//!
//!   *`wrapping`-clause kernels.* A `wrapping` contract clause declares wrap
//!   intent for a kernel's **values**; it does not reach the prover (which
//!   receives only `def`/`buffers`/`assumes`), and it must not, because a
//!   wrapped **index** is still out of bounds — wrapping an index does not make
//!   it a valid index. So a `wrapping` kernel is treated identically: its value
//!   arithmetic is already tainted (array-loaded / bitwise-derived), its
//!   indices are non-wrapping (or become `OutOfSubset`/`Refuted` if they could
//!   wrap), exactly like any other kernel.
//! - **Div/mod-derived indices:** `Arithmetic::Div`/`Arithmetic::Modulo` are
//!   modeled with SMT-LIB `div`/`mod` (Euclidean division), but only when an
//!   internal side-obligation — the divisor is nonzero and both operands are
//!   nonnegative, under the *live* path conditions + assumes — actually
//!   discharges (`Prover::try_discharge`, checked fresh for every div/mod
//!   site, not inferred from the operands' IR types: a *signed* intermediate
//!   like an `i32` `a - b` can be genuinely negative, so the nonnegativity
//!   half of the side-obligation is a real proof, not a type-driven
//!   assumption). Euclidean div/mod coincide with Rust's/WGSL's
//!   truncated-toward-zero semantics exactly when both operands are
//!   nonnegative, which is why that check is required rather than optional.
//!   (An *unsigned* operand is always nonnegative — a real hardware fact the
//!   leaf bounds + faithful wraparound below encode, not an assumption — so
//!   for unsigned div/mod the nonnegativity half discharges trivially and
//!   correctly; the divisor-nonzero half is the load-bearing check, and it is
//!   now sound under wraparound because the divisor term equals its real
//!   hardware value — see the "Bounded-integer overflow model" bullet.)
//!   If the side-obligation does not discharge (SAT, or an inconclusive
//!   `unknown`), the result is left tainted — never hard-errored, since the
//!   value may never feed an obligation — per the same taint discipline as
//!   everything else here. This side-obligation is deliberately *not*
//!   counted in `Prover::obligations` (which counts only the public
//!   `Index`/`IndexAssign` bounds obligations `ProveResult::Proved` reports):
//!   it's an internal precondition for soundly *modeling* div/mod, not a
//!   bounds check the caller asked for.
//! - **Element-range assumptions (array-value-dependent indices).** This is
//!   the ONLY case array element *contents* get a model; every other read stays
//!   tainted, exactly as before. An `Assume::ElemsBelowLen { arr, len_of }` /
//!   `ElemsBelowConst { arr, bound }` (the `arr.iter().all(|v| (*v as usize) <
//!   len_of.len())` / `… < N` contract shapes) records a bound for the global
//!   array `arr`. A read `arr[i]` — whose OWN index obligation `0 <= i <
//!   arr.len()` is still emitted and must discharge exactly as today —
//!   additionally binds its result to a *fresh* symbol `v` constrained
//!   `v < bound` (and `0 <= v` when `arr`'s element type is unsigned, a sound
//!   type fact; a signed element array models `v < bound` alone, so the
//!   `0 <= index` half of a later obligation is a real proof, never assumed).
//!   This lets a gather `x[offsets[i]]` discharge the inner `offsets[i] <
//!   x.len()` obligation, and nested gathers `a[b[i]]` compose automatically:
//!   `b[i]`'s fresh symbol (bounded `< a.len()` by `b`'s own assume) is exactly
//!   what the `a[...]` obligation needs, and the fresh symbol flows through any
//!   modeled arithmetic in between unchanged. **Soundness:** the assumption is
//!   an *assumed* claim recorded in evidence (the proof is conditional on it,
//!   precisely like a length assume), and the executable `check_assumes`
//!   predicate tests it at generation time, so the differential lane only ever
//!   runs inputs satisfying it. A *wrong* bound (looser than the indexed
//!   array's length) does not hide a bug: z3 picks a `v` in `[?, bound)` that
//!   the indexed array's length does not cover and `Refuted`s, with the fresh
//!   `elem…` symbol in the counterexample at the offending boundary.
//! - **Write invalidation.** A write `arr[j] = …` (`IndexAssign`) invalidates
//!   `arr`'s element assumption for every *subsequent* read of `arr` in the
//!   walk — the written value need not satisfy the assume, so a later `arr[k]`
//!   read is tainted rather than modeled (`elem_invalidated`, monotonic). This
//!   covers an assume array the kernel also mutates (an in-place scatter) and
//!   an assume array that is itself an output. A read that *precedes* the write
//!   in program order keeps its model (the write hasn't happened yet), so both
//!   directions are honored. For a **loop**, a body that writes `arr` anywhere
//!   invalidates `arr` for the whole body *before* it is walked
//!   (`invalidate_loop_element_writes`), because a later iteration's write
//!   happens-before an earlier iteration's read — the in-order rule alone would
//!   unsoundly model a read that textually precedes the write. Every case is
//!   conservative (invalidation only ever *removes* a model), hence sound.
//! - **Length relationships (`A.len() + K <= B.len()`).** A third structured
//!   assume shape (`Assume::LenPlusConstLe`, integer literal `K`; the `K = 0`
//!   case `A.len() <= B.len()`) asserts `len_a + K <= len_b` directly over the
//!   two buffers' u32-range length leaves. Combined with a guard `i < A.len()`
//!   it discharges a forward/offset read `B[i + K]` in bounds. The `<=` is
//!   asserted verbatim (the recognized clause *is* the constraint — no
//!   index-validity reinterpretation like the element-range proxy, so `<=`
//!   is exactly correct here where only `<` was sound for elements).
//! - **Infeasible-assumption guard.** After all structured assumes are
//!   asserted (and *only* the global length facts are in scope — element-range
//!   assumes populate `elem_bounds` and assert nothing global), a `check-sat`
//!   verifies the assumption context is satisfiable. An UNSAT context (mutually
//!   contradictory assumes — the single-clause `A.len() + K <= A.len()` with
//!   `K > 0`, or a contradictory pair like `y.len() == 1` ∧ `y.len() == 2`)
//!   would discharge *every* obligation vacuously — a false `Proved`. It is
//!   rejected as `OutOfSubset` instead (the round-1 "infeasible context
//!   vacuously proves" trap, generalized to the whole assume set). Real
//!   kernels' assumes are satisfiable, so this never fires for them, and a
//!   solver `Unknown` is treated as "not provably contradictory" (allowed
//!   through — conservative).
//! - **Cooperative mode (shared-memory milestone M1):** a second entry point
//!   `prove_bounds_freedom_cooperative(def, buffers, assumes, cube_dim)` opts
//!   the walk into *workgroup-cooperative* modeling by pinning `CUBE_DIM` to a
//!   concrete `cube_dim` constant (the `cooperative(cube_dim = …)` contract
//!   clause, docs/design-shared-memory.md §7.1). This flips on modeling for
//!   the 1-D topology builtins that the plain (`coop == None`) walk leaves
//!   tainted: `UnitPos` is a fresh `[0, cube_dim)` symbol (the per-thread
//!   leaf); `CubePos`/`CubeCount` are fresh nonnegative symbols (cube-uniform
//!   leaves); `CubeDim` is the concrete numeral `cube_dim`; and `AbsolutePos`
//!   is *recomputed* from the 1-D identity `CubePos*cube_dim + UnitPos`
//!   instead of the plain walk's opaque fresh leaf, so a `tile[UnitPos]`
//!   access and an `input[AbsolutePos]` access in the same kernel share one
//!   `UnitPos` symbol. **That recomputation is the exact *modular* one**
//!   (`abs_pos_sym`), not the raw sum: `AbsolutePos` is a fresh in-range `u32`
//!   leaf `abs_pos` asserted equal to `CubePos*cube_dim + UnitPos − k*2^32`
//!   for a fresh `k ≥ 0` — i.e. `abs_pos ≡ (CubePos*cube_dim + UnitPos) mod
//!   2^32`, its unique residue and so the true hardware value. This is
//!   soundness-critical, not cosmetic: `CubePos` is a *full*-`u32` leaf, so the
//!   raw sum `CubePos*cube_dim + UnitPos` overflows `2^32` on hardware (e.g.
//!   `CubePos = 2^24`, `cube_dim = 256` ⟹ `2^32 ≡ 0`), and a raw
//!   `smt.times`/`smt.plus` term (the pre-round-5 encoding) would be that
//!   *unwrapped* over-value — which a guard `ABSOLUTE_POS < len` would then
//!   unsoundly transfer onto `CubePos` (`CubePos*256 < len`), falsely proving a
//!   bare `output[CUBE_POS]` access whose real index is unbounded (adversarial
//!   review round 5, `abs_pos_sym` docs). Both products are variable×constant,
//!   hence LINEAR (QF_LIA). All of these are memoized on `VariableKind`, so every
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
//! - **Two-thread race walk (M3) + barrier uniformity (M4).** A second entry
//!   point `prove_race_freedom(def, buffers, assumes, cube_dim)` proves
//!   data-race freedom via the GPUVerify-style two-thread reduction
//!   (docs/design-shared-memory.md §5). It reuses this whole walker — branch
//!   write-taint, loop-carry taint, div/mod modeling, bounds obligations — with
//!   a `race: Some(RaceState)` layer on top; a bounds walk (`race == None`) is
//!   byte-for-byte unchanged.
//!   * **Two arbitrary distinct threads `t1 ≠ t2` of one cube.** The body is
//!     walked *twice*, once with `UnitPos → t1`, once with `→ t2`, both sharing
//!     one SMT context and the cube-uniform leaves (`CubePos`/`CubeCount`,
//!     integer `GlobalScalar`s, buffer `Length`s — declared once, kept across
//!     the two walks; only the per-thread locals are reset). `t1 ≠ t2` is
//!     asserted *per race obligation* (in `check_race`), never globally, so it
//!     cannot make the context vacuously infeasible for a degenerate
//!     `cube_dim == 1` and thereby fake-discharge the *bounds* obligations (the
//!     round-1 "infeasible context vacuously proves" trap).
//!   * **Phase segmentation at `sync_cube()` (§5.3).** Each thread's walk
//!     records its shared/global accesses (index term + a snapshot of all live
//!     path facts as its `guard`) into the current phase; a `SyncCube` closes
//!     the phase. After both walks, for each phase and each array, every
//!     write-write and read-write cross-thread pair is checked
//!     `guard1 ∧ guard2 ∧ idx1 = idx2` — SAT ⟹ a race (`Refuted` with a
//!     two-thread counterexample), UNSAT ⟹ race-free. No cross-*phase*
//!     obligations: the barrier orders phase-`p` writes before phase-`p+1`
//!     reads (sound only because the barrier is uniform — M4). The walk also
//!     discharges the ordinary *bounds* obligation of every access it resolves
//!     (once, on the `t1` pass), which is how the tree-reduction
//!     `tile[tid+half] < len` obligation the single-thread bounds walk defers
//!     finally discharges here. Recorded guards conjoin *all* live facts
//!     (including the cooperative loop's scoped `1 ≤ half ≤ init` bounds), so a
//!     deferred obligation is self-contained and a scoped fact never leaks into
//!     an unrelated phase's query.
//!   * **Cooperative loop, race walk (§5.5 interpretation).** §5.5 pins the
//!     per-obligation SMT encodings but does not spell out the loop-phase
//!     treatment; the conservative reading taken here (documented per the task):
//!     the recognized `while half > 0 { …; sync_cube(); half /= c }` tree loop's
//!     single carried control variable `half` is modeled as *one shared*
//!     symbolic value `H` with `1 ≤ H ≤ init` (`init` = its resolved pre-loop
//!     value), the loop body is walked *once*, and the internal `SyncCube`
//!     segments the generic per-iteration phase. This is sound because: `H` is
//!     shared between the two threads (a uniform trip count means both share the
//!     same `half` at every level — exactly what makes the reduction race-free);
//!     `H ≤ init` over-approximates every level because the halving recurrence
//!     (structurally required — `half /= constant`) is non-increasing; and the
//!     barrier between iterations means iteration-`i` writes never race
//!     iteration-`i+1` reads, so one symbolic per-iteration phase covers every
//!     iteration. A differently-shaped tree loop (a `RangeLoop`, a decrement, a
//!     manual recurrence) is *not* recognized and yields `OutOfSubset`, never a
//!     wrong `Proved` (§9 risk 1).
//!   * **Barrier uniformity (M4, §5.4).** A static thread-varying taint pass
//!     (`collect_thread_varying`, a fixpoint forward dataflow: `UnitPos`/
//!     `AbsolutePos` and array-loaded values are varying, an unmodeled op's out
//!     is conservatively varying, uniform-preserving ops over all-uniform
//!     operands stay uniform) classifies every barrier-enclosing condition and
//!     cooperative-loop trip count. A `sync_cube()` under a thread-varying `if`
//!     (or inside a loop with a thread-varying trip count) is barrier divergence
//!     — `OutOfSubset` with the §7.3 wording, never a silent `Proved`. Uniform
//!     conditional barriers (the `if CUBE_POS < n` case) are also rejected —
//!     deferred to v1.1 (§4.3) — so the only accepted barrier positions are
//!     top-level and the top level of the uniform halving loop. This is the
//!     conservative direction: the taint only ever marks a value *uniform* when
//!     it is provably built from cube-uniform leaves, so a divergent barrier is
//!     never accepted (the round-2 branch-scoping analog the design flags for
//!     adversarial probing).
//!   * **Inter-cube global writes (§5.3).** Two threads in different cubes are
//!     never barrier-separated, so a global-output write must be disjoint across
//!     cubes by construction. v1 recognizes exactly the two provable cases
//!     (`out[ABSOLUTE_POS]`, globally unique; single-writer `out[CUBE_POS]` when
//!     the write's guard implies `unit_pos == 0`, checked by SMT); any other
//!     global-write index, or a global-output array that is also *read*
//!     (inter-cube read-write), is `OutOfSubset` (§7.4), never a silent
//!     `Proved`.

use std::collections::{HashMap, HashSet};

use cubecl::ir::{
    Arithmetic, Branch, Builtin, Comparison, ConstantValue, ElemType, Id, Instruction, Loop,
    Metadata, Operation, Operator, Scope, Switch, Synchronization, Type, UIntKind, Variable,
    VariableKind,
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
    /// `arr.iter().all(|v| (*v as usize) < len_of.len())` — every element of
    /// the integer array `arr` is a valid index into `len_of` (a *content*
    /// assumption, unlike the length assumptions above). Lets a read
    /// `arr[i]` (still itself in-bounds) produce a value modeled as a fresh
    /// symbol bounded `< len_of.len()` instead of tainted — the only case
    /// array element contents get a model (see the module docs'
    /// "Element-range assumptions" bullet). Invalidated for `arr` by any
    /// write to `arr`'s elements.
    ElemsBelowLen { arr: &'a str, len_of: &'a str },
    /// `arr.iter().all(|v| *v < bound)` — every element of the integer array
    /// `arr` is below the constant `bound` (the constant-bound sibling of
    /// `ElemsBelowLen`).
    ElemsBelowConst { arr: &'a str, bound: u64 },
    /// `A.len() + K <= B.len()` (integer literal `K`; the `K = 0` case is
    /// `A.len() <= B.len()`) — a length *relationship* asserted directly as
    /// `len_a + K <= len_b` over the two buffers' (u32-range) length leaves.
    /// Combined with a guard `i < A.len()` it discharges an offset read
    /// `B[i + K]` in bounds. Adding this constraint only narrows the
    /// counterexample search, so it can never mint a false `Proved`.
    LenPlusConstLe { a: &'a str, k: u64, b: &'a str },
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

/// The obligation breakdown of a `Proved` cooperative walk, split so the two
/// evidence claims one walk backs (`smt-oob-freedom` and `smt-race-freedom`)
/// each carry an honest, non-overlapping count (docs/design-shared-memory.md
/// §5.6, §6).
#[derive(Debug, Clone, Copy)]
pub struct CooperativeObligations {
    /// Out-of-bounds obligations discharged — every shared/global index proved
    /// `0 <= idx < len`, resolved once on the thread-1 pass. This is the
    /// `smt-oob-freedom` count for a cooperative kernel: it includes exactly
    /// the tree-reduction `tile[tid+half] < len` obligations that the
    /// single-thread `prove_bounds_freedom_cooperative` defers to the race walk
    /// (see `block_sum_reduce_defers_to_m3`).
    pub bounds: usize,
    /// Write-write race obligations discharged (§5.3).
    pub write_write: usize,
    /// Read-write race obligations discharged (§5.3).
    pub read_write: usize,
    /// Inter-cube single-writer global-output checks discharged (§5.3).
    pub intercube: usize,
    /// Barrier-uniformity checks passed (§5.4) — dataflow taint, no SMT query;
    /// counted for legibility in the race claim's config.
    pub uniformity: usize,
    /// Phase count (barrier intervals the body was partitioned into, §5.3).
    pub phases: usize,
}

impl CooperativeObligations {
    /// Total data-race obligations (the `smt-race-freedom` count): write-write
    /// + read-write + inter-cube single-writer.
    pub fn race(&self) -> usize {
        self.write_write + self.read_write + self.intercube
    }
}

/// The detailed outcome of the two-thread cooperative walk
/// ([`prove_cooperative`]). The single sound walk (docs/design-shared-memory.md
/// §5) discharges BOTH the out-of-bounds obligations of every access it
/// resolves AND the data-race obligations, so one walk backs two distinct
/// evidence claims — `smt-oob-freedom` and `smt-race-freedom` — each with its
/// own honest count. This is why the milestone's cooperative kernels earn a
/// `Proved` `smt-oob-freedom` claim even though `prove_bounds_freedom_
/// cooperative` alone returns `OutOfSubset` for them (the single-thread bounds
/// walk defers a barrier-carrying tree loop to this walker).
#[derive(Debug, Clone)]
pub enum CooperativeProof {
    Proved(CooperativeObligations),
    Refuted { obligation: String, counterexample: String },
    OutOfSubset { reason: String },
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
/// the exact modular `(CubePos*cube_dim + UnitPos) mod 2^32`, `abs_pos_sym`) and
/// accepts `SharedMemory` (`SharedArray`)
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

/// The `check` string of the out-of-bounds-freedom `Proved` claim. A single
/// source of truth shared by the bounds prover and the M6 evidence wiring.
pub const SMT_OOB_FREEDOM_CHECK: &str = "smt-oob-freedom";

/// The `check` string of the race-freedom `Proved` claim (sibling to
/// `"smt-oob-freedom"`), per docs/design-shared-memory.md §5.6. Defined here
/// so the (milestone M6) evidence wiring has a single source of truth.
pub const SMT_RACE_FREEDOM_CHECK: &str = "smt-race-freedom";

/// Prove **data-race freedom** for a workgroup-cooperative `def`, via the
/// two-thread symbolic reduction (docs/design-shared-memory.md §5, milestones
/// M3+M4). `cube_dim` pins `CUBE_DIM` exactly as `prove_bounds_freedom_
/// cooperative` does (§7.1 / §9 risk 5).
///
/// Two arbitrary distinct symbolic threads `t1 ≠ t2` of one cube are walked
/// (§5.1): the body is partitioned into phases at every `sync_cube()` (§5.3),
/// and within each phase every shared/global write is proved not to collide
/// (same index) with another thread's write (write-write) or read (read-write)
/// — the negation checked SAT, UNSAT meaning race-free. The walk also
/// discharges the ordinary bounds obligations of every access it resolves —
/// including the tree-reduction `tile[tid+half] < len` obligation that the
/// single-thread bounds walk defers to here (it cannot model the barrier-
/// carrying loop). Barrier uniformity (§5.4, M4) is enforced by thread-varying
/// dataflow taint: a `sync_cube()` under a thread-varying condition, or inside
/// a loop with a thread-varying trip count, is `OutOfSubset` (§7.3), never a
/// silent `Proved`.
///
/// `Proved { obligations }` counts every discharged SMT query (bounds +
/// write-write + read-write + inter-cube single-writer). A `Refuted` carries a
/// two-thread counterexample (values of `t1`/`t2` that collide). Anything
/// outside the recognized reduction subset is `OutOfSubset` with a targeted
/// reason rather than an unsound verdict.
pub fn prove_race_freedom(
    def: &KernelDefinition,
    buffers: &[BufferParam],
    assumes: &[Assume],
    cube_dim: u32,
) -> ProveResult {
    // `None` expected-barrier count: `prove_race_freedom` is the standalone
    // race verdict used by the prover's own unit tests, which build IR directly
    // and do not carry a twin-declared count. The composition barrier check
    // (§7.4) runs on the suite path (`prove_cooperative`), which threads the
    // macro's `COOP_BARRIER_COUNT`.
    match prove_race_freedom_detailed(def, buffers, assumes, cube_dim, None) {
        CooperativeProof::Proved(o) => ProveResult::Proved { obligations: o.bounds + o.race() },
        CooperativeProof::Refuted { obligation, counterexample } => {
            ProveResult::Refuted { obligation, counterexample }
        }
        CooperativeProof::OutOfSubset { reason } => ProveResult::OutOfSubset { reason },
        CooperativeProof::SolverError { detail } => ProveResult::SolverError { detail },
    }
}

/// Prove BOTH out-of-bounds freedom and data-race freedom for a
/// workgroup-cooperative `def` in one two-thread walk, keeping the obligation
/// breakdown split (see [`CooperativeProof`] / [`CooperativeObligations`]).
///
/// This is the conformance suite's cooperative entry point (docs/design-shared-
/// memory.md §6): the SAME sound walk `prove_race_freedom` performs, but
/// returning the bounds/race split so the evidence can carry two distinct
/// `Proved` claims — `smt-oob-freedom` (bounds) and `smt-race-freedom` (races)
/// — from one walk. `prove_race_freedom` is this collapsed to a single combined
/// count (for callers that only need the race verdict).
///
/// `expected_barriers` is the twin's declared top-level `sync_cube()` count
/// (the macro's `COOP_BARRIER_COUNT`). The walk first checks that the inlined IR
/// contains exactly that many `SyncCube` instructions — a `uses(...)` helper
/// that hid a barrier would inflate the IR count and is rejected `OutOfSubset`
/// (cooperative-composition soundness crux, docs/design-shared-memory.md §7.4).
pub fn prove_cooperative(
    def: &KernelDefinition,
    buffers: &[BufferParam],
    assumes: &[Assume],
    cube_dim: u32,
    expected_barriers: usize,
) -> CooperativeProof {
    prove_race_freedom_detailed(def, buffers, assumes, cube_dim, Some(expected_barriers))
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
        race: None,
        elem_bounds: HashMap::new(),
        elem_invalidated: HashSet::new(),
        coop_barrier_seen: false,
        // A cooperative bounds walk may hit a `terminate!()` before deferring at
        // the tree loop; compute the thread-varying set so its uniformity can be
        // checked. Empty for a non-cooperative walk (no terminate reachable).
        coop_varying: if coop.is_some() {
            collect_thread_varying(&def.body)
        } else {
            HashSet::new()
        },
        walk_depth: 0,
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

fn prove_race_freedom_detailed(
    def: &KernelDefinition,
    buffers: &[BufferParam],
    assumes: &[Assume],
    cube_dim: u32,
    expected_barriers: Option<usize>,
) -> CooperativeProof {
    // Cooperative-composition barrier check (§7.4): the phase-split twin only
    // segments at the kernel's *top-level* `sync_cube()` calls, and cube inlines
    // a `uses(...)` helper's IR into this scope — so if the inlined IR carries
    // more `SyncCube` instructions than the twin declared, a helper hid a barrier
    // and the twin's phase structure silently disagrees with the real execution.
    // Reject before any SMT work (the twin lane also rejects a helper containing
    // `sync_cube`, so the two lanes agree; this is the independent proof-lane
    // half). A `None` count (the `prove_race_freedom` unit-test path) skips it.
    if let Some(expected) = expected_barriers {
        let actual = count_sync_cube_ir(&def.body);
        if actual != expected {
            return CooperativeProof::OutOfSubset {
                reason: format!(
                    "cooperative kernel's IR contains {actual} `sync_cube()` barrier(s) but the \
                     phase-split twin declares {expected} top-level barrier(s): a `uses(...)` \
                     helper hid a barrier the twin cannot see — barriers must be visible at the \
                     kernel's top level (docs/design-shared-memory.md §7.4)"
                ),
            };
        }
    }
    let mut smt = match ContextBuilder::new().solver("z3").solver_args(["-smt2", "-in"]).build() {
        Ok(ctx) => ctx,
        Err(e) => {
            return CooperativeProof::SolverError { detail: format!("failed to start z3: {e}") };
        }
    };

    // Static thread-varying taint (§5.4): a pure pass over the IR, used to
    // classify barrier-enclosing conditions and cooperative-loop trip counts.
    let varying = collect_thread_varying(&def.body);

    // A valid-but-never-read placeholder for the two thread-id slots before
    // `race_setup` declares them (SExpr has no `Default` and private fields).
    let placeholder = smt.true_();

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
        // Race mode is a cooperative walk: the leaf modeling (`UnitPos` etc.)
        // and shared-array bounds are all needed.
        coop: Some(cube_dim),
        race: Some(RaceState {
            cube_dim,
            thread: Thread::T1,
            // placeholders, overwritten by `race_setup`
            t1: placeholder,
            t2: placeholder,
            fact_stack: Vec::new(),
            guard_stack: Vec::new(),
            current_phase: Vec::new(),
            phases_t1: Vec::new(),
            phases_t2: Vec::new(),
            uniform_loop: HashMap::new(),
            varying: varying.clone(),
            ww: 0,
            rw: 0,
            global_checks: 0,
            uniformity_checks: 0,
        }),
        elem_bounds: HashMap::new(),
        elem_invalidated: HashSet::new(),
        coop_barrier_seen: false,
        coop_varying: varying,
        walk_depth: 0,
    };

    if let Err(e) = prover.assert_structured_assumes(assumes) {
        return e.into_coop();
    }
    if let Err(e) = prover.race_setup() {
        return e.into_coop();
    }
    if let Err(e) = prover.race_walk(&def.body, Thread::T1) {
        return e.into_coop();
    }
    prover.race_reset_for_t2();
    if let Err(e) = prover.race_walk(&def.body, Thread::T2) {
        return e.into_coop();
    }
    // Capture the phase count before `emit_race_obligations` drains `phases_t1`.
    let phases = prover.race.as_ref().expect("race mode").phases_t1.len();
    if let Err(e) = prover.emit_race_obligations() {
        return e.into_coop();
    }
    let r = prover.race.as_ref().expect("race mode");
    CooperativeProof::Proved(CooperativeObligations {
        bounds: prover.obligations,
        write_write: r.ww,
        read_write: r.rw,
        intercube: r.global_checks,
        uniformity: r.uniformity_checks,
        phases,
    })
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

    fn into_coop(self) -> CooperativeProof {
        match self {
            Stop::OutOfSubset(reason) => CooperativeProof::OutOfSubset { reason },
            Stop::Refuted { obligation, counterexample } => {
                CooperativeProof::Refuted { obligation, counterexample }
            }
            Stop::SolverError(detail) => CooperativeProof::SolverError { detail },
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
    /// `Some(..)` only for the two-thread race walk (`prove_race_freedom`,
    /// milestones M3+M4); `None` for every bounds walk. When set, `UnitPos`
    /// resolves to whichever of the two thread symbols is currently active,
    /// shared/global accesses are recorded per phase for cross-thread
    /// obligation emission, and `sync_cube()` is a phase boundary rather than
    /// the no-op it is for a bounds walk. All race logic is gated on this being
    /// `Some`, so the bounds walk stays byte-for-byte unchanged.
    race: Option<RaceState>,
    /// Per-global-array element-range bounds, from `Assume::ElemsBelowLen`/
    /// `ElemsBelowConst` (module docs' "Element-range assumptions"). A read
    /// `arr[i]` of a global array whose id is a key here — and NOT in
    /// `elem_invalidated` — produces a value modeled as a fresh symbol
    /// `0 <= v < b` for every recorded bound `b` (multiple assumes on one
    /// array conjoin), instead of the usual taint. Empty unless an
    /// element-range assume is declared, so a kernel with none is
    /// byte-for-byte unchanged.
    elem_bounds: HashMap<Id, Vec<SExpr>>,
    /// Global-array ids whose element assumption has been invalidated by a
    /// write to that array's elements (an `IndexAssign`), for the remainder of
    /// the walk (module docs' "Write invalidation"). Monotonic: once a write
    /// to `arr` is seen — in program order, or anywhere in an enclosing loop
    /// body via the loop pre-scan — every subsequent read of `arr`'s elements
    /// is tainted rather than modeled, regardless of the assume. Conservative
    /// (only ever removes modeling), hence sound.
    elem_invalidated: HashSet<Id>,
    /// Whether a `sync_cube()` barrier (or a barrier-carrying cooperative loop)
    /// has been reached yet in this walk. A workgroup-uniform `terminate!()`
    /// (§4.3/§7.4) is only accepted *before* any barrier — the "skip the whole
    /// cube" guard — so a terminate encountered with this `true` is rejected.
    /// Set by the `Synchronization` arm of `process_instruction` and at the top
    /// of `process_cooperative_loop`. Always `false` for a non-cooperative walk
    /// (which has no `terminate!()`, so it is never read).
    coop_barrier_seen: bool,
    /// Static thread-varying taint set (`collect_thread_varying`) for a
    /// cooperative walk — the same set the race walk uses to gate barrier
    /// uniformity, additionally consulted to verify a `terminate!()` condition is
    /// cube-uniform (a thread-varying terminate is barrier divergence). Empty for
    /// a non-cooperative walk.
    coop_varying: HashSet<VariableKind>,
    /// Recursion depth of `process_scope`. `0` is the outermost (body) scope —
    /// the only one that recognises a `terminate!()` (§4.3/§7.4: the guard is
    /// top-level). Incremented on entry, decremented on exit.
    walk_depth: u32,
}

/// Which of the two symbolic threads a race walk is currently resolving
/// `UnitPos` to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Thread {
    T1,
    T2,
}

/// The array a race obligation reasons about. A `SharedMemory` tile and a
/// global buffer never alias, and two shared tiles are distinct iff their ids
/// differ (§2.2), so `(variant, id)` is a sound identity for "same array".
#[derive(Clone, Copy, PartialEq, Eq)]
enum RaceArray {
    Shared(Id),
    Global(Id),
}

/// One shared/global memory access recorded during a thread's phase walk.
/// `index`/`guard` are already thread-instantiated (built with the active
/// thread's `UnitPos` symbol); the cross-thread obligation combines a
/// thread-1 access with a thread-2 access (§5.3).
#[derive(Clone)]
struct Access {
    array: RaceArray,
    is_write: bool,
    /// The symbolic index term at this access.
    index: SExpr,
    /// Conjunction of every fact live at the access site (branch conditions,
    /// loop guards, and cooperative-loop `1 ≤ half ≤ init` bounds) — recorded
    /// so the deferred cross-thread obligation is self-contained and does not
    /// depend on any SMT scope still being open, and so a scoped fact (e.g. the
    /// tree loop's `half` bounds) never leaks into an unrelated phase's
    /// obligation (the round-1 "infeasible context vacuously proves" trap).
    guard: SExpr,
    /// Display name (`shared_array(id)` / a buffer name) for descriptions.
    name: String,
    /// The IR index variable's kind, used by the inter-cube global-write gate
    /// to recognize the two provable-by-construction disjoint patterns (§5.3).
    index_kind: VariableKind,
}

/// One enclosing branch/loop guard, tracked while it is open so a `sync_cube()`
/// reached under it can be classified for barrier uniformity (§5.4, M4).
#[derive(Clone, Copy)]
struct GuardEntry {
    /// Whether the guard condition is thread-varying (depends on
    /// `UnitPos`/`AbsolutePos` or array contents). A varying guard enclosing a
    /// barrier is barrier divergence.
    varying: bool,
    /// `true` for a loop guard (trip-count divergence wording), `false` for an
    /// `if`/`else` guard (condition-divergence wording).
    is_loop: bool,
}

/// Per-thread + phase bookkeeping for the two-thread race walk.
struct RaceState {
    cube_dim: u32,
    thread: Thread,
    /// The two distinct symbolic thread ids (`0 ≤ t < cube_dim`, `t1 ≠ t2`).
    t1: SExpr,
    t2: SExpr,
    /// Facts live at the current point, conjoined into each recorded access's
    /// `guard`. Mirrors — but is never popped by z3's `pop`, so it survives
    /// into the deferred obligation phase — the path conditions asserted on the
    /// SMT stack.
    fact_stack: Vec<SExpr>,
    /// Enclosing guards, for the barrier-uniformity check at each `sync_cube`.
    guard_stack: Vec<GuardEntry>,
    /// Accesses recorded in the currently-open phase for the active thread.
    current_phase: Vec<Access>,
    /// Completed phases, index-aligned between the two threads (both walks
    /// traverse identical control flow, so phase `i` denotes the same barrier
    /// interval for both).
    phases_t1: Vec<Vec<Access>>,
    phases_t2: Vec<Vec<Access>>,
    /// The shared symbolic `half` per cooperative-loop control variable —
    /// created in the thread-1 walk and reused in the thread-2 walk, because a
    /// uniform trip count means both threads share the *same* `half` at each
    /// tree level (that shared value is exactly what makes the reduction
    /// race-free).
    uniform_loop: HashMap<VariableKind, SExpr>,
    /// Thread-varying variable kinds (a static pre-pass over the IR), consulted
    /// for barrier uniformity.
    varying: HashSet<VariableKind>,
    /// Discharged write-write / read-write / inter-cube-single-writer counts
    /// (bounds obligations are counted by the shared `Prover::obligations`).
    ww: usize,
    rw: usize,
    global_checks: usize,
    /// Barriers verified uniform (counted once per barrier, on the thread-1
    /// walk), for the report.
    uniformity_checks: usize,
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

    /// Declare a fresh integer leaf constrained to its type's hardware range
    /// `[type_min, type_max]` (module docs' "Bounded-integer overflow model").
    /// This is a *sound type fact* — a `u32` value really is in `[0, 2^32)` —
    /// and it is load-bearing for the overflow model: it is what lets the
    /// no-wrap side-obligations of `checked_mul`/`cast_int` discharge for
    /// genuinely-safe arithmetic (e.g. `flatten_decode_scale`'s `row*w <= pos
    /// <= u32::MAX`), and what keeps the single-wrap `wrap_to_range` `ite`
    /// exact (operands in range ⟹ at most one wrap each direction).
    fn declare_leaf(&mut self, hint: &str, ty: &Type) -> Result<SExpr, Stop> {
        let e = self.declare_int(hint, is_unsigned(ty))?;
        let max = self.smt.numeral(type_max(ty));
        let le = self.smt.lte(e, max);
        self.smt.assert(le).map_err(smt_err)?;
        if !is_unsigned(ty) {
            let min = int_const(self.smt, type_min(ty));
            let ge = self.smt.gte(e, min);
            self.smt.assert(ge).map_err(smt_err)?;
        }
        Ok(e)
    }

    /// A synthetic leaf with no IR `Type` of its own (a position/length
    /// builtin) — modeled at the address-type width (`u32`, see
    /// `address_type`).
    fn declare_u32_leaf(&mut self, hint: &str) -> Result<SExpr, Stop> {
        self.declare_leaf(hint, &address_type())
    }

    fn length_of(&mut self, id: Id) -> Result<SExpr, Stop> {
        if let Some(e) = self.buffer_len.get(&id) {
            return Ok(*e);
        }
        let hint = format!("len_{}_", self.buffer_name(id));
        let e = self.declare_u32_leaf(&hint)?;
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
                // Element-range assumes assert NO global fact (they constrain
                // array *contents*, not lengths); instead they record a bound
                // consulted by `model_element_read` at each read of `arr`
                // (module docs' "Element-range assumptions"). Multiple assumes
                // on one array conjoin (each pushes its own bound).
                Assume::ElemsBelowLen { arr, len_of } => {
                    let arr_id = self.buffer_id_by_name(arr)?;
                    let len_id = self.buffer_id_by_name(len_of)?;
                    let bound = self.length_of(len_id)?;
                    self.elem_bounds.entry(arr_id).or_default().push(bound);
                }
                Assume::ElemsBelowConst { arr, bound } => {
                    let arr_id = self.buffer_id_by_name(arr)?;
                    let b = self.smt.numeral(bound);
                    self.elem_bounds.entry(arr_id).or_default().push(b);
                }
                // Length relationship `len_a + K <= len_b`, asserted verbatim.
                // Both lengths are u32-range leaves (`length_of`); `k` is a
                // small literal. A raw `plus`/`lte` is exactly right here (no
                // faithful-wrap handling needed): the constraint IS `len_a + K
                // <= len_b`, and any state where the mathematical `len_a + K`
                // exceeds `len_b`'s u32 range is genuinely infeasible (it would
                // require `len_b > 2^32 - 1`), which the assertion correctly
                // rules out rather than mismodeling.
                Assume::LenPlusConstLe { a, k, b } => {
                    let ida = self.buffer_id_by_name(a)?;
                    let idb = self.buffer_id_by_name(b)?;
                    let la = self.length_of(ida)?;
                    let lb = self.length_of(idb)?;
                    let kk = self.smt.numeral(k);
                    let sum = self.smt.plus(la, kk);
                    let le = self.smt.lte(sum, lb);
                    self.smt.assert(le).map_err(smt_err)?;
                }
            }
        }
        // Infeasible-assumption guard (round-1 "infeasible context vacuously
        // proves" trap, generalized to the assume context): if the declared
        // assumptions are mutually contradictory — their conjunction, over the
        // buffer-length range facts asserted above, is UNSAT — then EVERY
        // bounds obligation would discharge vacuously, minting a false `Proved`.
        // The single-clause form `A.len() + K <= A.len()` (K > 0) is the most
        // insidious (one clause), but a contradictory *pair* (`y.len() == 1` and
        // `y.len() == 2`, or two `LenPlusConstLe` clauses that transitively
        // contradict) is the same class. Reject any unsatisfiable assumption set
        // as `OutOfSubset` rather than yield a vacuous certificate. This runs at
        // the outermost scope, before any obligation is emitted, and only the
        // *global* length facts are in scope here (element-range assumes populate
        // `elem_bounds` and assert nothing global), so it tests exactly the
        // assumption feasibility. Real kernels' assumes are satisfiable, so it
        // never fires for them (`check-sat` returns SAT); `Unknown` is treated as
        // "not provably contradictory" and allowed through (conservative).
        if self.smt.check().map_err(smt_err)? == Response::Unsat {
            return Err(Stop::OutOfSubset(
                "the declared assumptions are mutually contradictory (their conjunction is \
                 unsatisfiable) — a contradictory contract would vacuously discharge every bounds \
                 obligation, so it is rejected rather than yielding a false `Proved`"
                    .into(),
            ));
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
        // `AbsolutePos` carries a `declare-const` + a recomposition assertion
        // (unlike the pre-round-5 raw `times`/`plus` term, which was pure), so
        // it must be predeclared at the outermost scope for the same reason as
        // the other leaves: a lazy first resolution inside a branch arm would
        // scope its declaration+assertion to that arm and drop them on the
        // matching `pop`, leaving a later use referencing an undeclared symbol.
        self.abs_pos_sym()?;
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

    /// The `CubePos` leaf: a fresh (cube-uniform) `u32`-range symbol.
    fn cube_pos_sym(&mut self) -> Result<SExpr, Stop> {
        let kind = VariableKind::Builtin(Builtin::CubePos);
        if let Some(Some(e)) = self.memo.get(&kind) {
            return Ok(*e);
        }
        let sym = self.declare_u32_leaf("cube_pos")?;
        self.memo.insert(kind, Some(sym));
        Ok(sym)
    }

    /// The `CubeCount` leaf: a fresh (cube-uniform) `u32`-range symbol.
    fn cube_count_sym(&mut self) -> Result<SExpr, Stop> {
        let kind = VariableKind::Builtin(Builtin::CubeCount);
        if let Some(Some(e)) = self.memo.get(&kind) {
            return Ok(*e);
        }
        let sym = self.declare_u32_leaf("cube_count")?;
        self.memo.insert(kind, Some(sym));
        Ok(sym)
    }

    /// The `AbsolutePos` leaf in cooperative mode: a fresh in-range `u32` symbol
    /// `abs_pos` tied to the **exact modular recomposition**
    /// `abs_pos = cube_pos*cube_dim + unit_pos − k*2^32` for a fresh wrap count
    /// `k ≥ 0` (module docs' "Cooperative mode"). This is the soundness-critical
    /// alternative to building `cube_pos*cube_dim + unit_pos` with raw
    /// `smt.times`/`smt.plus`: because `cube_pos` is a full-`u32` leaf, that raw
    /// sum can exceed `2^32`, which real hardware wraps, so the raw (unwrapped)
    /// term is *not* the hardware value and a guard `ABSOLUTE_POS < len` would
    /// unsoundly transfer a bound onto `cube_pos` that hardware never honors
    /// (adversarial review round 5). Declaring `abs_pos` as its own leaf in
    /// `[0, 2^32)` **congruent to the raw sum mod 2^32** models the wrap exactly:
    /// a value in `[0, 2^32)` congruent to `X` mod `2^32` is unique, so `abs_pos`
    /// is pinned to `X mod 2^32` = the true hardware value (multiple wraps
    /// included, since `k` is unconstrained above by soundness). This keeps the
    /// "every non-tainted modeled integer term equals the real hardware value"
    /// invariant (module docs' "Bounded-integer overflow model"). Both products
    /// are variable×constant (`cube_dim` and `2^32` are constants), hence LINEAR
    /// — QF_LIA, no QF_NIA. Memoized on `VariableKind::Builtin(AbsolutePos)`; in
    /// the race walk it is re-derived per thread (`race_walk` clears it so it
    /// picks up that thread's `UnitPos`).
    fn abs_pos_sym(&mut self) -> Result<SExpr, Stop> {
        let kind = VariableKind::Builtin(Builtin::AbsolutePos);
        if let Some(Some(e)) = self.memo.get(&kind) {
            return Ok(*e);
        }
        let cube_dim = self.coop.expect("abs_pos_sym only reachable in cooperative mode");
        let unit = self.unit_pos_sym()?;
        let cube = self.cube_pos_sym()?;
        // Fresh in-range leaf: `0 <= abs_pos <= u32::MAX` (address-type width).
        let abs = self.declare_u32_leaf("abs_pos")?;
        // Fresh wrap count `k >= 0`, additionally bounded above by the constant
        // `cube_dim - 1`: with `cube_pos <= 2^32 - 1` and `unit_pos <= cube_dim
        // - 1`, the raw sum is at most `2^32*cube_dim - 1`, so `k = floor(raw /
        // 2^32) <= cube_dim - 1`. The upper bound is *not* needed for soundness
        // (the `[0, 2^32)` range plus the congruence already pin `abs_pos` to the
        // unique residue) but tightens the model at no cost.
        let k = self.declare_int("abs_wrap", true)?;
        let k_max = self.smt.numeral((cube_dim as u64).saturating_sub(1));
        let k_le = self.smt.lte(k, k_max);
        self.smt.assert(k_le).map_err(smt_err)?;
        // abs_pos = cube_pos*cube_dim + unit_pos - k*2^32  (both products linear).
        let cd = self.smt.numeral(cube_dim as u64);
        let scaled = self.smt.times(cube, cd);
        let sum = self.smt.plus(scaled, unit);
        let modulus = self.smt.numeral(wrap_modulus(&address_type()));
        let wraps = self.smt.times(k, modulus);
        let rhs = self.smt.sub(sum, wraps);
        let eq = self.smt.eq(abs, rhs);
        self.smt.assert(eq).map_err(smt_err)?;
        self.memo.insert(kind, Some(abs));
        Ok(abs)
    }

    /// Resolve a topology builtin. In cooperative mode the 1-D leaves are
    /// modeled (module docs' "Cooperative mode"); otherwise only
    /// `AbsolutePos` is (a plain fresh leaf), everything else tainted —
    /// byte-for-byte the pre-milestone behavior.
    fn builtin_value(&mut self, b: Builtin) -> Option<SExpr> {
        let Some(cube_dim) = self.coop else {
            return match b {
                Builtin::AbsolutePos => self.declare_u32_leaf("abs_pos").ok(),
                _ => None,
            };
        };
        match b {
            Builtin::UnitPos => self.unit_pos_sym().ok(),
            Builtin::CubePos => self.cube_pos_sym().ok(),
            Builtin::CubeCount => self.cube_count_sym().ok(),
            Builtin::CubeDim => Some(self.smt.numeral(cube_dim as u64)),
            // AbsolutePos = (CubePos*cube_dim + UnitPos) mod 2^32 — the 1-D
            // identity under finite-width wraparound, encoded exactly by
            // `abs_pos_sym` (a raw `times`/`plus` here would be the *unwrapped*
            // over-value and unsound; see `abs_pos_sym`'s docs, round 5).
            Builtin::AbsolutePos => self.abs_pos_sym().ok(),
            // X/Y/Z, plane, cluster builtins: out of the 1-D subset.
            _ => None,
        }
    }

    // -- control-flow walk ---------------------------------------------

    fn process_scope(&mut self, scope: &Scope) -> Result<(), Stop> {
        // Only the outermost (body) scope handles `terminate!()` (§4.3/§7.4: the
        // guard is top-level). `walk_depth` is 0 exactly for that call.
        let top_level = self.walk_depth == 0;
        self.walk_depth += 1;
        let result = self.process_scope_inner(scope, top_level);
        self.walk_depth -= 1;
        result
    }

    fn process_scope_inner(&mut self, scope: &Scope, top_level: bool) -> Result<(), Stop> {
        for inst in &scope.instructions {
            // Recognise a cooperative `terminate!()` — `if <cond> { return }` at
            // the top level — and model it as a `!cond` path condition for the
            // rest of the walk (the threads that continue are exactly those where
            // the terminate condition was false). See `enter_coop_terminate`.
            if top_level {
                if let Some(cond_var) = self.as_coop_terminate(inst) {
                    self.enter_coop_terminate(&cond_var)?;
                    continue;
                }
            }
            self.process_instruction(inst)?;
        }
        Ok(())
    }

    /// Recognise a cooperative `terminate!()`: an `if <cond> { return }` whose
    /// then-scope is exactly a single `Branch::Return` (the IR §2.5 lowers
    /// `terminate!()` to a `Branch::Return` nested in a structured `if`).
    /// Returns the condition `Variable`. Only recognised in cooperative mode
    /// (`terminate` is banned outside a cooperative kernel, so no such shape
    /// reaches a non-cooperative walk from vericl); a `None` return leaves the
    /// `if` to ordinary `process_branch` handling (where `Branch::Return` is a
    /// no-op), which is sound but adds no `!cond`.
    fn as_coop_terminate(&self, inst: &Instruction) -> Option<Variable> {
        self.coop?; // cooperative mode only
        let Operation::Branch(Branch::If(if_)) = &inst.operation else { return None };
        let insts = &if_.scope.instructions;
        if insts.len() != 1 {
            return None;
        }
        match &insts[0].operation {
            Operation::Branch(Branch::Return) => Some(if_.cond),
            _ => None,
        }
    }

    /// Enter a recognised cooperative `terminate!()`: verify it is workgroup-
    /// uniform and before any barrier (§4.3/§7.4), then assert `!cond` as a path
    /// condition for the rest of the walk. It is asserted at the **outermost SMT
    /// scope without a `push`** (never popped) — soundness-critical: the body
    /// walk must stay at the outermost scope so a cooperative tree loop's control
    /// leaf (`half`) declared during it survives to `emit_race_obligations`; a
    /// scoping `push` here would drop that declaration on its matching `pop`,
    /// leaving a deferred cross-thread obligation referencing an undeclared
    /// symbol (the same hazard `predeclare_coop_leaves` avoids). Because the
    /// terminate is top-level and applies to the entire remaining body, never
    /// popping is exactly right — there is no later code where `!cond` should not
    /// hold. A non-uniform or post-barrier terminate is `OutOfSubset` — the same
    /// shape the twin lane rejects, so the two lanes agree.
    fn enter_coop_terminate(&mut self, cond_var: &Variable) -> Result<(), Stop> {
        // Before any barrier: a terminate after a `sync_cube()` would leave the
        // twin's "skip the whole cube" model (which the twin only recognises at
        // the top level before any barrier) — rejected on both lanes.
        if self.coop_barrier_seen {
            return Err(Stop::OutOfSubset(
                "terminate!() after a barrier is outside the vericl v1.1 subset — a \
                 workgroup-uniform terminate must precede every sync_cube() (the \"skip the whole \
                 cube\" guard, docs/design-shared-memory.md §4.3/§7.4)"
                    .into(),
            ));
        }
        // Uniformity (§5.4): a thread-varying terminate condition is barrier
        // divergence (some threads skip the cube, others reach the barrier) —
        // rejected exactly like a thread-varying barrier guard, never a silent
        // `Proved`. Uses the same static thread-varying taint set.
        if var_is_thread_varying(cond_var, &self.coop_varying) {
            return Err(Stop::OutOfSubset(
                "terminate!() under a thread-varying condition (barrier divergence) is outside the \
                 vericl v1.1 subset — only a workgroup-uniform terminate is accepted \
                 (docs/design-shared-memory.md §4.3/§7.4)"
                    .into(),
            ));
        }
        let Some(cond) = self.value_of(cond_var) else {
            return Err(Stop::OutOfSubset(
                "terminate!() condition depends on a construct outside the vericl v0 subset".into(),
            ));
        };
        let not_cond = self.smt.not(cond);
        self.smt.assert(not_cond).map_err(smt_err)?;
        // In a race walk, `!cond` is also a live fact conjoined into every
        // subsequent access's recorded guard (so the deferred cross-thread
        // obligations see it), exactly like an `if`/loop guard. `race_reset_for_
        // t2` clears the fact stack, and the thread-2 walk re-encounters the same
        // top-level terminate and re-adds it — so both threads' recorded guards
        // carry `!cond`, and both walks re-assert it (a harmless duplicate on the
        // shared, cube-uniform `CubePos`).
        if let Some(r) = self.race.as_mut() {
            r.fact_stack.push(not_cond);
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
            // A barrier is a phase boundary in the race walk (§5.3); in a
            // bounds walk it is a no-op (it has no `out`, so this matches the
            // pre-existing `_ => taint_out` behavior exactly).
            Operation::Synchronization(s) => {
                // A barrier: no `terminate!()` may follow (§4.3/§7.4). Recorded
                // in both walks (the bounds walk may see a top-level barrier
                // before deferring at the tree loop).
                self.coop_barrier_seen = true;
                if self.race.is_some() {
                    self.process_sync(s)?;
                }
            }
            // Everything else (Bitwise, Atomic, Plane, CoopMma, Barrier, Tma,
            // NonSemantic, Marker, ...) is outside the modeled subset. It is
            // not fatal on its own: leave its `out` (if any) unbound so any
            // later obligation that actually depends on it fails explicitly at
            // that use site instead of here, where it may be entirely
            // irrelevant to array bounds (see module docs).
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
            // Add/Sub are modeled *faithfully* under finite-width wraparound
            // (`wrap_to_range`): the SMT term equals the real hardware value at
            // every input, wrap included (module docs' "Bounded-integer
            // overflow model"). Operands are in range (leaf bounds), so at most
            // one wrap can occur in either direction.
            Arithmetic::Add(b) => self.wrapping_binary(b, &out.ty, |s, l, r| s.plus(l, r)),
            Arithmetic::Sub(b) => self.wrapping_binary(b, &out.ty, |s, l, r| s.sub(l, r)),
            // Mul can wrap up to `2^W` times; `(a*b) mod 2^W` is nonlinear, so
            // instead of a faithful term we discharge a no-overflow
            // side-obligation and bind the plain product only when it provably
            // cannot wrap — else taint. This is the case the round-2 adversarial
            // review's `a*b == 2^32` divisor construction lands in.
            Arithmetic::Mul(b) => self.checked_mul(b, &out.ty)?,
            Arithmetic::Div(b) => self.divmod_int(b, |s, l, r| s.div(l, r))?,
            Arithmetic::Modulo(b) => self.divmod_int(b, |s, l, r| s.modulo(l, r))?,
            _ => None,
        };
        self.bind_out(inst, val);
        Ok(())
    }

    /// Faithful finite-width Add/Sub: resolve both modeled-integer operands,
    /// apply `f` (`plus`/`sub`) to get the *mathematical* result, then fold it
    /// back into `[type_min, type_max]` with `wrap_to_range` so the SMT term is
    /// the exact hardware value at every input. Returns `None` (taint) if an
    /// operand is not a modeled integer or does not resolve — the same
    /// discipline as `binary_int`.
    fn wrapping_binary(
        &mut self,
        b: &cubecl::ir::BinaryOperator,
        out_ty: &Type,
        f: impl FnOnce(&Context, SExpr, SExpr) -> SExpr,
    ) -> Option<SExpr> {
        if !is_modeled_int(&b.lhs.ty) || !is_modeled_int(&b.rhs.ty) {
            return None;
        }
        let l = self.value_of(&b.lhs)?;
        let r = self.value_of(&b.rhs)?;
        let raw = f(self.smt, l, r);
        Some(self.wrap_to_range(raw, out_ty))
    }

    /// Fold the mathematical result `raw` back into `ty`'s hardware range via
    /// `ite(raw > max, raw - 2^W, ite(raw < min, raw + 2^W, raw))`. Sound
    /// (exact) only when `raw` is at most one modulus outside the range in
    /// either direction, which holds because every operand feeding an Add/Sub
    /// is itself in range (leaf bounds / prior faithful/tainted results). The
    /// two `ite` conditions are mutually exclusive, so at most one correction
    /// applies. See the module docs' "Bounded-integer overflow model".
    fn wrap_to_range(&mut self, raw: SExpr, ty: &Type) -> SExpr {
        let max = self.smt.numeral(type_max(ty));
        let modulus = self.smt.numeral(wrap_modulus(ty));
        let min = int_const(self.smt, type_min(ty));
        let over = self.smt.gt(raw, max);
        let raw_minus_mod = self.smt.sub(raw, modulus);
        let under = self.smt.lt(raw, min);
        let raw_plus_mod = self.smt.plus(raw, modulus);
        let inner = self.smt.ite(under, raw_plus_mod, raw);
        self.smt.ite(over, raw_minus_mod, inner)
    }

    /// Model `Arithmetic::Mul` with a no-overflow side-obligation: resolve both
    /// modeled-integer operands, then try to discharge `type_min <= a*b <=
    /// type_max` under the *live* path conditions + leaf bounds. Discharged ⟹
    /// the product provably does not wrap, so the plain SMT `*` term equals the
    /// real hardware value — bind it. Not discharged (SAT / `unknown`) ⟹ the
    /// product may wrap, so it is left tainted (`Ok(None)`), exactly like the
    /// div/mod side-obligation. `Err` only on a genuine solver I/O failure. The
    /// side-obligation is deliberately *not* counted in `Prover::obligations`
    /// (it is an internal modeling precondition, not a public bounds check).
    fn checked_mul(
        &mut self,
        b: &cubecl::ir::BinaryOperator,
        out_ty: &Type,
    ) -> Result<Option<SExpr>, Stop> {
        if !is_modeled_int(&b.lhs.ty) || !is_modeled_int(&b.rhs.ty) {
            return Ok(None);
        }
        let (Some(l), Some(r)) = (self.value_of(&b.lhs), self.value_of(&b.rhs)) else {
            return Ok(None);
        };
        let product = self.smt.times(l, r);
        let max = self.smt.numeral(type_max(out_ty));
        let le = self.smt.lte(product, max);
        let min = int_const(self.smt, type_min(out_ty));
        let ge = self.smt.gte(product, min);
        let in_range = self.smt.and(ge, le);
        if !self.try_discharge(in_range)? {
            return Ok(None);
        }
        Ok(Some(product))
    }

    /// Model an integer→integer `Cast`. A value-preserving cast (widening /
    /// equal width without a sign reinterpretation, `cast_is_value_preserving`)
    /// passes the operand term through unchanged. A narrowing or same-width
    /// signedness-flip cast can change the value, so it passes through only when
    /// a "fits the destination range" side-obligation discharges (so the value
    /// is unchanged by the cast); otherwise it taints. Keeps the invariant that
    /// every modeled term equals the real hardware value or is tainted (module
    /// docs' "Bounded-integer overflow model", Cast paragraph).
    fn cast_int(&mut self, input: &Variable, dst_ty: &Type) -> Result<Option<SExpr>, Stop> {
        if !is_modeled_int(dst_ty) || !is_modeled_int(&input.ty) {
            return Ok(None);
        }
        let Some(v) = self.value_of(input) else { return Ok(None) };
        if cast_is_value_preserving(&input.ty, dst_ty) {
            return Ok(Some(v));
        }
        let max = self.smt.numeral(type_max(dst_ty));
        let le = self.smt.lte(v, max);
        let min = int_const(self.smt, type_min(dst_ty));
        let ge = self.smt.gte(v, min);
        let fits = self.smt.and(ge, le);
        if !self.try_discharge(fits)? {
            return Ok(None);
        }
        Ok(Some(v))
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
                let val = self.cast_int(&u.input, &out.ty)?;
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
        self.access(&aref, false, idx, &name, io.index.kind, len)?;
        // The value *read* from the array is normally unknown (this checker has
        // no model of array contents) — taint, don't bind. The one exception:
        // a global array covered by an in-force element-range assumption yields
        // a value modeled as a fresh symbol bounded by the assumption, so a
        // gather `x[offsets[i]]` (or a nested `a[b[i]]`) can discharge its inner
        // index obligation (module docs' "Element-range assumptions").
        if !self.model_element_read(inst, &aref)? {
            self.taint_out(inst);
        }
        Ok(())
    }

    /// If the array read by `inst` is a global array with an in-force
    /// element-range assumption (a key in `elem_bounds`, not in
    /// `elem_invalidated`) and an integer element type, bind `inst`'s output to
    /// a fresh symbol constrained `0 <= v` (for an unsigned element type — a
    /// sound type fact) and `v < b` for every recorded bound `b`, and return
    /// `true`. Otherwise bind nothing and return `false` (the caller taints).
    ///
    /// Soundness: the assumption is an *assumed* claim recorded in evidence
    /// (like a length assume), and the executable `check_assumes` predicate
    /// tests it at generation time, so the differential lane only ever runs
    /// inputs that satisfy it. Non-negativity is asserted only from the element
    /// *type* (an unsigned value is non-negative unconditionally); a signed
    /// element array therefore models `v < b` alone, leaving the `0 <= index`
    /// half of a later index obligation to fail honestly rather than be assumed.
    fn model_element_read(
        &mut self,
        inst: &Instruction,
        aref: &ArrayRef,
    ) -> Result<bool, Stop> {
        let ArrayRef::Global { id } = *aref else { return Ok(false) };
        if self.elem_invalidated.contains(&id) {
            return Ok(false);
        }
        let Some(out) = inst.out else { return Ok(false) };
        if !is_modeled_int(&out.ty) {
            return Ok(false);
        }
        let Some(bounds) = self.elem_bounds.get(&id).cloned() else { return Ok(false) };
        let v = self.declare_leaf("elem", &out.ty)?;
        for b in bounds {
            let lt = self.smt.lt(v, b);
            self.smt.assert(lt).map_err(smt_err)?;
        }
        self.bind_out(inst, Some(v));
        Ok(true)
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
        self.access(&aref, true, idx, &name, io.index.kind, len)?;
        // Write invalidation (module docs): a write to a global array's
        // elements invalidates that array's element-range assumption for every
        // subsequent read, since the written value need not satisfy it. Covers
        // the assume array being an output the kernel also reads, and the same
        // array mutated by the kernel itself. Monotonic and conservative (only
        // ever removes modeling), hence sound.
        if !self.elem_bounds.is_empty() {
            if let ArrayRef::Global { id } = aref {
                self.elem_invalidated.insert(id);
            }
        }
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

    // -- two-thread race walk (shared-memory milestones M3 + M4) ---------
    //
    // See the module docs' "Two-thread race walk" bullet. Everything here is
    // gated on `self.race.is_some()`; a bounds walk never enters it.

    /// One shared/global access. In a bounds walk this is exactly the old
    /// `emit_obligation`; in a race walk it discharges the bounds obligation
    /// once (on the thread-1 pass — the thread-2 index is symmetric) and
    /// records the access into the current phase for the cross-thread
    /// obligations emitted after both walks (§5.3).
    fn access(
        &mut self,
        aref: &ArrayRef,
        is_write: bool,
        idx: SExpr,
        name: &str,
        index_kind: VariableKind,
        len: SExpr,
    ) -> Result<(), Stop> {
        let kind_str = if is_write { "write" } else { "read" };
        match self.race.as_ref().map(|r| r.thread) {
            None => self.emit_obligation(len, name, idx, kind_str),
            Some(thread) => {
                if thread == Thread::T1 {
                    self.emit_obligation(len, name, idx, kind_str)?;
                }
                let array = match *aref {
                    ArrayRef::Shared { id, .. } => RaceArray::Shared(id),
                    ArrayRef::Global { id } => RaceArray::Global(id),
                };
                let facts = self.race.as_ref().expect("race mode").fact_stack.clone();
                let guard = self.and_fold(&facts);
                let access = Access {
                    array,
                    is_write,
                    index: idx,
                    guard,
                    name: name.to_string(),
                    index_kind,
                };
                self.race.as_mut().expect("race mode").current_phase.push(access);
                Ok(())
            }
        }
    }

    /// Conjunction of `facts` (`true` when empty). Used to snapshot the live
    /// path condition into each recorded access's `guard`.
    fn and_fold(&mut self, facts: &[SExpr]) -> SExpr {
        match facts.split_first() {
            None => self.smt.true_(),
            Some((first, rest)) => {
                let mut acc = *first;
                for f in rest {
                    acc = self.smt.and(acc, *f);
                }
                acc
            }
        }
    }

    /// Declare the two distinct thread ids and the shared cube-uniform leaves.
    /// `t1`/`t2` are `[0, cube_dim)` and asserted distinct (§5.1–5.2); the
    /// cube-uniform leaves (`CubePos`, `CubeCount`) are declared once and shared
    /// by both threads.
    fn race_setup(&mut self) -> Result<(), Stop> {
        let cube_dim = self.race.as_ref().expect("race mode").cube_dim;
        let bound = self.smt.numeral(cube_dim as u64);
        let t1 = self.declare_int("t", true)?;
        let lt1 = self.smt.lt(t1, bound);
        self.smt.assert(lt1).map_err(smt_err)?;
        let t2 = self.declare_int("t", true)?;
        let lt2 = self.smt.lt(t2, bound);
        self.smt.assert(lt2).map_err(smt_err)?;
        // NB: `t1 != t2` is deliberately NOT asserted globally — it is part of
        // each *race* obligation (asserted per-query in `check_race`), not a
        // fact the *bounds* obligations may lean on. Asserting it globally would
        // make the whole context infeasible for a degenerate `cube_dim == 1`
        // (no two distinct threads exist), which would vacuously discharge every
        // bounds obligation — the round-1 "infeasible context vacuously proves"
        // trap. Kept local, the bounds obligations stay a genuine per-thread
        // proof at any `cube_dim`.
        // Shared cube-uniform leaves (predeclared so their range facts are in
        // force for the whole walk — same reasoning as `predeclare_coop_leaves`,
        // but NOT `unit_pos_sym`, which would clash with the per-thread `t`).
        self.cube_pos_sym()?;
        self.cube_count_sym()?;
        // Predeclare every buffer length at the outermost scope. A length can
        // appear in a recorded access guard (e.g. `CUBE_POS < output.len()`)
        // that a *deferred* cross-thread obligation re-asserts long after the
        // branch it was first resolved under has been popped — and SMT-LIB
        // `pop` discards declarations, so a lazily-declared length would be an
        // "unknown constant" there. Declaring them all up front (like the coop
        // leaves) keeps every recorded guard/index self-contained. An unused
        // length is a harmless free nonnegative symbol.
        for id in 0..self.buffers.len() {
            self.length_of(id as Id)?;
        }
        let r = self.race.as_mut().expect("race mode");
        r.t1 = t1;
        r.t2 = t2;
        Ok(())
    }

    /// Walk `body` once for `thread`, binding `UnitPos` to that thread's id and
    /// closing the final (post-last-barrier) phase.
    fn race_walk(&mut self, body: &Scope, thread: Thread) -> Result<(), Stop> {
        let t = {
            let r = self.race.as_mut().expect("race mode");
            r.thread = thread;
            match thread {
                Thread::T1 => r.t1,
                Thread::T2 => r.t2,
            }
        };
        // `UnitPos` -> this thread; force `AbsolutePos` to recompute (exact
        // modular recomposition, `abs_pos_sym`) with the new `UnitPos`.
        self.memo.insert(VariableKind::Builtin(Builtin::UnitPos), Some(t));
        self.memo.remove(&VariableKind::Builtin(Builtin::AbsolutePos));
        // Predeclare this thread's `abs_pos` leaf + recomposition assertion at
        // the outermost scope, *before* `process_scope` opens any branch push —
        // its declaration must outlive every pop so a deferred cross-thread race
        // obligation whose recorded guard/index mentions `ABSOLUTE_POS` can
        // still reference it (identical reasoning to `race_setup`'s length
        // predeclaration). Reads the `UnitPos = t` just set above.
        self.abs_pos_sym()?;
        self.process_scope(body)?;
        let phase = std::mem::take(&mut self.race.as_mut().expect("race mode").current_phase);
        let r = self.race.as_mut().expect("race mode");
        match thread {
            Thread::T1 => r.phases_t1.push(phase),
            Thread::T2 => r.phases_t2.push(phase),
        }
        Ok(())
    }

    /// Reset the per-thread value state between the two walks, keeping the
    /// cube-uniform leaves (`CubePos`/`CubeCount`/integer `GlobalScalar`s) and
    /// buffer lengths (all thread-invariant), plus the `uniform_loop`/thread-id
    /// state on `RaceState`. Per-thread locals (`UnitPos`, `AbsolutePos`, every
    /// `LocalConst`/`LocalMut`) are dropped so the thread-2 walk recomputes them
    /// against `t2`.
    fn race_reset_for_t2(&mut self) {
        self.memo.retain(|k, _| {
            matches!(
                k,
                VariableKind::Builtin(Builtin::CubePos)
                    | VariableKind::Builtin(Builtin::CubeCount)
                    | VariableKind::GlobalScalar(_)
            )
        });
        // These stacks are balanced (empty) after a successful walk; assert-ish
        // clear for defensiveness.
        self.carried_stack.clear();
        self.write_log_stack.clear();
        // Reset the barrier-seen flag so the thread-2 walk re-evaluates the
        // terminate/barrier ordering from scratch (else it would think a barrier
        // was already seen from the thread-1 walk and reject a valid terminate).
        self.coop_barrier_seen = false;
        let r = self.race.as_mut().expect("race mode");
        r.fact_stack.clear();
        r.guard_stack.clear();
        r.current_phase.clear();
    }

    /// A `sync_cube()` in the race walk: verify barrier uniformity (§5.4) and
    /// close the current phase for the active thread.
    fn process_sync(&mut self, s: &Synchronization) -> Result<(), Stop> {
        match s {
            Synchronization::SyncCube => {
                self.check_barrier_uniformity()?;
                let r = self.race.as_mut().expect("race mode");
                let phase = std::mem::take(&mut r.current_phase);
                match r.thread {
                    Thread::T1 => r.phases_t1.push(phase),
                    Thread::T2 => r.phases_t2.push(phase),
                }
                Ok(())
            }
            // SyncPlane/SyncStorage/SyncAsyncProxyShared are out of the v1
            // subset (§2.3) — rejected, never silently treated as a barrier.
            other => Err(Stop::OutOfSubset(format!(
                "`{other}` is outside the vericl v0 subset (only `sync_cube()` is modeled)"
            ))),
        }
    }

    /// Barrier-uniformity gate (§5.4, M4): every guard enclosing a `sync_cube()`
    /// must be thread-invariant, and the barrier must not sit under an `if`
    /// (even a uniform one — deferred to v1.1, §4.3). A thread-varying enclosing
    /// guard is barrier divergence (§7.3): rejected, never a silent `Proved`.
    fn check_barrier_uniformity(&mut self) -> Result<(), Stop> {
        let r = self.race.as_ref().expect("race mode");
        for g in &r.guard_stack {
            if g.is_loop {
                if g.varying {
                    return Err(Stop::OutOfSubset(
                        "sync_cube() inside a loop with a thread-varying trip count (barrier \
                         divergence) is outside the vericl v0 subset"
                            .into(),
                    ));
                }
            } else if g.varying {
                return Err(Stop::OutOfSubset(
                    "sync_cube() under a non-uniform condition (barrier divergence) is outside the \
                     vericl v0 subset"
                        .into(),
                ));
            } else {
                return Err(Stop::OutOfSubset(
                    "sync_cube() inside a conditional (cube-uniform conditional barriers are \
                     deferred to vericl v1.1) is outside the vericl v0 subset"
                        .into(),
                ));
            }
        }
        // Count each barrier once (thread-1 walk) for the report.
        if r.thread == Thread::T1 {
            self.race.as_mut().expect("race mode").uniformity_checks += 1;
        }
        Ok(())
    }

    /// Push an enclosing guard onto the race stacks: `cond` becomes a live fact
    /// (conjoined into recorded access guards) and a `GuardEntry` (for the
    /// barrier-uniformity check). No-op in a bounds walk.
    fn race_push_guard(&mut self, cond_var: &Variable, cond: SExpr, is_loop: bool) {
        if let Some(r) = self.race.as_mut() {
            let varying = var_is_thread_varying(cond_var, &r.varying);
            r.fact_stack.push(cond);
            r.guard_stack.push(GuardEntry { varying, is_loop });
        }
    }

    fn race_pop_guard(&mut self) {
        if let Some(r) = self.race.as_mut() {
            r.fact_stack.pop();
            r.guard_stack.pop();
        }
    }

    /// The cooperative tree-reduction loop (a `Branch::Loop` carrying a
    /// `sync_cube`), modeled for the race walk (§5.5 interpretation — see the
    /// module docs' "Cooperative loop, race walk"). Recognizes the canonical
    /// `while half > 0 { …; sync_cube(); half /= c }` shape, requires a
    /// cube-uniform trip count (M4), models `half` as one shared symbol
    /// `1 ≤ H ≤ init` (`init` = its resolved pre-loop value, sound because the
    /// halving recurrence is non-increasing), and walks the body once so the
    /// internal barrier segments the per-iteration phase.
    fn process_cooperative_loop(&mut self, l: &Loop) -> Result<(), Stop> {
        // A barrier-carrying loop is a barrier region: no `terminate!()` may
        // follow (§4.3/§7.4).
        self.coop_barrier_seen = true;
        let bg = recognize_break_guard(&l.scope).ok_or_else(|| {
            Stop::OutOfSubset(
                "cooperative loop is not the recognized `while <uniform> { …; sync_cube(); … }` \
                 tree shape (no leading break-guard) — outside the vericl v1 subset"
                    .into(),
            )
        })?;
        // Control variable: the operand of a downward-counter guard `half > 0`.
        let Operation::Comparison(cmp) = &l.scope.instructions[bg.guard_idx].operation else {
            return Err(Stop::OutOfSubset(
                "cooperative loop guard is not a comparison — outside the vericl v1 subset".into(),
            ));
        };
        let ctrl = downcounter_ctrl(cmp).ok_or_else(|| {
            Stop::OutOfSubset(
                "cooperative loop guard is not the recognized `half > 0` downward-counter shape \
                 (a thread-uniform tree level) — outside the vericl v1 subset"
                    .into(),
            )
        })?;
        // Trip-count uniformity (M4): the control variable must be
        // cube-uniform, else the barrier inside diverges (§7.3).
        if var_is_thread_varying(&ctrl, &self.race.as_ref().expect("race mode").varying) {
            return Err(Stop::OutOfSubset(
                "sync_cube() inside a loop with a thread-varying trip count (barrier divergence) \
                 is outside the vericl v0 subset"
                    .into(),
            ));
        }
        // Recurrence must be a uniform halving (`half /= constant`, constant
        // >= 1) so the fresh symbol's `H <= init` upper bound is a sound
        // non-increasing over-approximation. A differently-shaped tree loop
        // (a decrement, a manual recurrence, a RangeLoop) is *not* recognized
        // and yields `OutOfSubset`, never a wrong `Proved` (§9 risk 1).
        verify_halving_update(&l.scope, ctrl.kind)?;
        // Pre-loop value of the control variable (the `init` upper bound).
        let init = self.value_of(&ctrl).ok_or_else(|| {
            Stop::OutOfSubset(
                "cooperative loop control variable's initial value depends on a construct outside \
                 the vericl v0 subset"
                    .into(),
            )
        })?;
        // Shared symbolic `half`: created on the thread-1 walk, reused on the
        // thread-2 walk (uniform trip count => both threads share this value).
        let h = match self.race.as_ref().expect("race mode").uniform_loop.get(&ctrl.kind) {
            Some(h) => *h,
            None => {
                let h = self.declare_int("half", true)?;
                self.race.as_mut().expect("race mode").uniform_loop.insert(ctrl.kind, h);
                h
            }
        };

        // Carried variables: taint every accumulator, bind the control var to
        // the shared `H` (mirrors `process_noncoop_loop`).
        let outer: HashSet<VariableKind> = self.memo.keys().copied().collect();
        let carried = scope_reassigned_vars(&l.scope, &outer);
        for &k in &carried {
            if k == ctrl.kind {
                self.set_var(k, Some(h));
            } else {
                self.set_var(k, None);
            }
        }
        self.carried_stack.push(carried.clone());

        let r = self.process_cooperative_loop_body(l, &bg, h, init);

        self.carried_stack.pop();
        for &k in &carried {
            self.set_var(k, None);
        }
        r
    }

    /// Body-walk portion of `process_cooperative_loop`, factored so the caller
    /// unconditionally pops `carried_stack`. Asserts `1 <= H <= init` in a
    /// *scoped* push (never global — a scoped-only fact keeps the round-1
    /// "infeasible context vacuously proves" trap out of unrelated phases) and
    /// on the `fact_stack` (so the deferred race obligations carry it), pushes
    /// the uniform loop guard, then walks the body past the break-guard.
    fn process_cooperative_loop_body(
        &mut self,
        l: &Loop,
        bg: &BreakGuard,
        h: SExpr,
        init: SExpr,
    ) -> Result<(), Stop> {
        self.smt.push().map_err(smt_err)?;
        let one = self.smt.numeral(1);
        let ge1 = self.smt.gte(h, one);
        self.smt.assert(ge1).map_err(smt_err)?;
        let le_init = self.smt.lte(h, init);
        self.smt.assert(le_init).map_err(smt_err)?;
        {
            let r = self.race.as_mut().expect("race mode");
            r.fact_stack.push(ge1);
            r.fact_stack.push(le_init);
            // The loop guard (`half > 0`) is uniform (checked above).
            r.guard_stack.push(GuardEntry { varying: false, is_loop: true });
        }

        let mut result = Ok(());
        for inst in &l.scope.instructions[bg.body_start..] {
            if let Err(e) = self.process_instruction(inst) {
                result = Err(e);
                break;
            }
        }

        {
            let r = self.race.as_mut().expect("race mode");
            r.guard_stack.pop();
            r.fact_stack.pop();
            r.fact_stack.pop();
        }
        self.smt.pop().map_err(smt_err)?;
        result
    }

    /// After both thread walks, emit the cross-thread obligations (§5.3): within
    /// each barrier interval, no write collides with another thread's write
    /// (write-write) or read (read-write) on the same array; plus the inter-cube
    /// single-writer gate for global-output writes (§5.3, the two
    /// provable-by-construction cases). Reads never race with reads.
    fn emit_race_obligations(&mut self) -> Result<(), Stop> {
        let (phases_t1, phases_t2) = {
            let r = self.race.as_mut().expect("race mode");
            (std::mem::take(&mut r.phases_t1), std::mem::take(&mut r.phases_t2))
        };
        if phases_t1.len() != phases_t2.len() {
            return Err(Stop::SolverError(format!(
                "race walk produced mismatched phase counts ({} vs {}) — non-deterministic control \
                 flow between the two thread walks",
                phases_t1.len(),
                phases_t2.len()
            )));
        }

        for (p, (a1, a2)) in phases_t1.iter().zip(phases_t2.iter()).enumerate() {
            // Distinct arrays touched in this phase (both threads see the same
            // set, since they walk identical control flow).
            let mut arrays: Vec<RaceArray> = Vec::new();
            for acc in a1.iter().chain(a2.iter()) {
                if !arrays.contains(&acc.array) {
                    arrays.push(acc.array);
                }
            }
            for array in arrays {
                let w1: Vec<&Access> =
                    a1.iter().filter(|x| x.array == array && x.is_write).collect();
                let r1: Vec<&Access> =
                    a1.iter().filter(|x| x.array == array && !x.is_write).collect();
                let w2: Vec<&Access> =
                    a2.iter().filter(|x| x.array == array && x.is_write).collect();
                let r2: Vec<&Access> =
                    a2.iter().filter(|x| x.array == array && !x.is_write).collect();
                // write-write
                for x in &w1 {
                    for y in &w2 {
                        self.check_race(x, y, "write-write", p)?;
                        self.race.as_mut().expect("race mode").ww += 1;
                    }
                }
                // read-write (t1 writes vs t2 reads, and t2 writes vs t1 reads)
                for x in &w1 {
                    for y in &r2 {
                        self.check_race(x, y, "read-write", p)?;
                        self.race.as_mut().expect("race mode").rw += 1;
                    }
                }
                for x in &w2 {
                    for y in &r1 {
                        self.check_race(x, y, "read-write", p)?;
                        self.race.as_mut().expect("race mode").rw += 1;
                    }
                }
            }
        }

        // Inter-cube global-output disjointness (§5.3): every global-output
        // write across the whole kernel must be one of the two
        // provable-by-construction disjoint patterns.
        self.check_intercube_global(&phases_t1)?;
        Ok(())
    }

    /// A single cross-thread conflict query: `guard1 ∧ guard2 ∧ index1 = index2`
    /// (with `t1 ≠ t2` and the thread ranges already global). UNSAT ⟹ the pair
    /// cannot collide (race-free); SAT ⟹ a real two-thread race, `Refuted` with
    /// the offending model.
    fn check_race(
        &mut self,
        a: &Access,
        b: &Access,
        kind: &str,
        phase: usize,
    ) -> Result<(), Stop> {
        self.smt.push().map_err(smt_err)?;
        // The two threads are distinct — asserted here, per race obligation,
        // rather than globally (see `race_setup`).
        let (t1, t2) = {
            let r = self.race.as_ref().expect("race mode");
            (r.t1, r.t2)
        };
        let eq_threads = self.smt.eq(t1, t2);
        let distinct = self.smt.not(eq_threads);
        self.smt.assert(distinct).map_err(smt_err)?;
        self.smt.assert(a.guard).map_err(smt_err)?;
        self.smt.assert(b.guard).map_err(smt_err)?;
        let same = self.smt.eq(a.index, b.index);
        self.smt.assert(same).map_err(smt_err)?;
        let response = self.smt.check();
        let outcome = match response {
            Ok(Response::Unsat) => Ok(()),
            Ok(Response::Sat) => {
                let counterexample = self.render_counterexample();
                Err(Stop::Refuted {
                    obligation: format!(
                        "no {kind} race on `{}` between two threads (phase {phase})",
                        a.name
                    ),
                    counterexample,
                })
            }
            Ok(Response::Unknown) => Err(Stop::SolverError(format!(
                "z3 returned `unknown` for a {kind} race obligation on `{}`",
                a.name
            ))),
            Err(e) => Err(smt_err(e)),
        };
        self.smt.pop().map_err(smt_err)?;
        outcome
    }

    /// The inter-cube global-output gate (§5.3). Two threads in *different*
    /// cubes are never separated by a barrier, so a global-output write must be
    /// disjoint across cubes by construction. v1 recognizes exactly the two
    /// provable cases: an `out[ABSOLUTE_POS]` write (globally unique id) and a
    /// single-writer `out[CUBE_POS]` write guarded by `tid == 0` (distinct
    /// cubes ⟹ distinct `CUBE_POS`). Anything else — and any global-output
    /// array that is *also read* (inter-cube read-write, unproved in v1) — is
    /// `OutOfSubset` (§7.4), never a silent `Proved`.
    fn check_intercube_global(&mut self, phases: &[Vec<Access>]) -> Result<(), Stop> {
        let t1 = self.race.as_ref().expect("race mode").t1;
        let mut read_globals: HashSet<Id> = HashSet::new();
        let mut written_globals: HashSet<Id> = HashSet::new();
        for phase in phases {
            for acc in phase {
                if let RaceArray::Global(id) = acc.array {
                    if acc.is_write {
                        written_globals.insert(id);
                    } else {
                        read_globals.insert(id);
                    }
                }
            }
        }
        for phase in phases {
            for acc in phase {
                let RaceArray::Global(id) = acc.array else { continue };
                if !acc.is_write {
                    continue;
                }
                if read_globals.contains(&id) {
                    return Err(Stop::OutOfSubset(format!(
                        "global array `{}` is both read and written — inter-cube read-write \
                         disjointness is deferred to vericl v1.1 (outside the v0 subset)",
                        acc.name
                    )));
                }
                match acc.index_kind {
                    // out[ABSOLUTE_POS]: globally unique across all threads.
                    VariableKind::Builtin(Builtin::AbsolutePos) => {}
                    // out[CUBE_POS]: single-writer iff guarded by `tid == 0`
                    // (distinct cubes => distinct CUBE_POS).
                    VariableKind::Builtin(Builtin::CubePos) => {
                        // guard ∧ t1 != 0 must be UNSAT (guard implies tid == 0).
                        let zero = self.smt.numeral(0);
                        let t1_ne_0 = {
                            let eq = self.smt.eq(t1, zero);
                            self.smt.not(eq)
                        };
                        let implies_tid0 = self.smt.and(acc.guard, t1_ne_0);
                        self.smt.push().map_err(smt_err)?;
                        self.smt.assert(implies_tid0).map_err(smt_err)?;
                        let response = self.smt.check();
                        self.smt.pop().map_err(smt_err)?;
                        match response {
                            Ok(Response::Unsat) => {
                                self.race.as_mut().expect("race mode").global_checks += 1;
                            }
                            Ok(Response::Sat) => {
                                return Err(Stop::OutOfSubset(format!(
                                    "global write `{}[CUBE_POS]` is not provably a single-writer \
                                     (not guarded by `unit_pos == 0`) — inter-cube disjointness \
                                     unproved (outside the vericl v0 subset)",
                                    acc.name
                                )));
                            }
                            Ok(Response::Unknown) => {
                                return Err(Stop::SolverError(
                                    "z3 returned `unknown` for the single-writer gate".into(),
                                ));
                            }
                            Err(e) => return Err(smt_err(e)),
                        }
                    }
                    _ => {
                        return Err(Stop::OutOfSubset(format!(
                            "global write `{}[...]` index is not a provable inter-cube-disjoint \
                             pattern (only `out[ABSOLUTE_POS]` and single-writer `out[CUBE_POS]` \
                             are recognized in v1) — outside the vericl v0 subset",
                            acc.name
                        )));
                    }
                }
            }
        }
        Ok(())
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
                self.race_push_guard(&if_.cond, cond, false);
                self.smt.push().map_err(smt_err)?;
                self.smt.assert(cond).map_err(smt_err)?;
                let r = self.process_scope(&if_.scope);
                self.smt.pop().map_err(smt_err)?;
                self.race_pop_guard();
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
                self.race_push_guard(&ie.cond, cond, false);
                self.smt.push().map_err(smt_err)?;
                self.smt.assert(cond).map_err(smt_err)?;
                let r1 = self.process_scope(&ie.scope_if);
                self.smt.pop().map_err(smt_err)?;
                self.race_pop_guard();
                let written_if =
                    self.write_log_stack.pop().expect("just pushed, push/pop are balanced");
                r1?;

                // Restore the pre-branch snapshot before walking the else
                // arm: without this, the else arm would see the if arm's
                // writes (the confirmed round-2 manifestation).
                self.memo = snapshot.clone();

                self.write_log_stack.push(HashSet::new());
                let not_cond = self.smt.not(cond);
                self.race_push_guard(&ie.cond, not_cond, false);
                self.smt.push().map_err(smt_err)?;
                self.smt.assert(not_cond).map_err(smt_err)?;
                let r2 = self.process_scope(&ie.scope_else);
                self.smt.pop().map_err(smt_err)?;
                self.race_pop_guard();
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
            Branch::Switch(sw) => self.process_switch(sw),
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

    /// Model a `Branch::Switch` (Rust `match` on an integer, module docs'
    /// "Switch modeling (match on integers)") as an exhaustive if-chain: each
    /// case arm walked under the path condition `value == case_i`, the default
    /// arm walked under the conjunction of all `value != case_i`. Reuses the
    /// exact same branch-scoped write-taint machinery as `If`/`IfElse`
    /// (snapshot `self.memo` before each arm, restore between arms, then taint
    /// every variable written in *any* arm after the construct via `set_var`),
    /// generalized from 2 arms to N+1 — a per-arm write leaking past the merge
    /// is impossible for exactly the same reason as `IfElse`. The race walk
    /// pushes each arm's condition as a `race_push_guard` keyed on the scrutinee
    /// `sw.value`, so a `sync_cube()` inside a switch arm is classified (and, in
    /// v0/v1, always rejected) by the same barrier-uniformity gate as any other
    /// conditional barrier.
    fn process_switch(&mut self, sw: &Switch) -> Result<(), Stop> {
        // A tainted scrutinee is out of subset at the switch, exactly like a
        // tainted `if` condition (`cond_of`) — never silently modeled.
        let value = self.value_of(&sw.value).ok_or_else(|| {
            Stop::OutOfSubset(
                "`match`/switch scrutinee depends on a construct outside the vericl v0 subset"
                    .into(),
            )
        })?;

        // Resolve every case's value up front. cubecl only ever emits
        // integer-literal case constants for a numeric `match` (each an
        // `Or`-pattern literal is its own case, sharing a cloned body), so these
        // always resolve via `constant_expr`; a value that does not resolve
        // (not a modeled constant) takes the whole switch out of subset rather
        // than being mismodeled.
        let mut case_exprs: Vec<SExpr> = Vec::with_capacity(sw.cases.len());
        for (case_var, _) in &sw.cases {
            let c = self.value_of(case_var).ok_or_else(|| {
                Stop::OutOfSubset(
                    "`match`/switch case value is not a modeled constant — outside the vericl v0 \
                     subset"
                        .into(),
                )
            })?;
            case_exprs.push(c);
        }

        let snapshot = self.memo.clone();
        let mut all_written: HashSet<VariableKind> = HashSet::new();

        // Each case arm under `value == case_i`. Same snapshot/restore +
        // write-log discipline as one `IfElse` arm.
        for ((_, scope), case_expr) in sw.cases.iter().zip(&case_exprs) {
            let cond = self.smt.eq(value, *case_expr);
            self.write_log_stack.push(HashSet::new());
            self.race_push_guard(&sw.value, cond, false);
            self.smt.push().map_err(smt_err)?;
            self.smt.assert(cond).map_err(smt_err)?;
            let r = self.process_scope(scope);
            self.smt.pop().map_err(smt_err)?;
            self.race_pop_guard();
            let written = self.write_log_stack.pop().expect("just pushed, push/pop are balanced");
            r?;
            self.memo = snapshot.clone();
            all_written.extend(written);
        }

        // Default arm under the conjunction of `value != case_i` for every case.
        // This is exactly the negation of "some case matched", so the default
        // path condition is sound: a case set that covers the guard's whole
        // range makes the conjunction unsatisfiable under the live facts, and
        // the default arm's obligations then discharge vacuously — correct,
        // because that arm is genuinely unreachable for those inputs.
        let mut negations: Vec<SExpr> = Vec::with_capacity(case_exprs.len());
        for c in &case_exprs {
            let eq = self.smt.eq(value, *c);
            negations.push(self.smt.not(eq));
        }
        let default_cond = self.and_fold(&negations);
        self.write_log_stack.push(HashSet::new());
        self.race_push_guard(&sw.value, default_cond, false);
        self.smt.push().map_err(smt_err)?;
        self.smt.assert(default_cond).map_err(smt_err)?;
        let r = self.process_scope(&sw.scope_default);
        self.smt.pop().map_err(smt_err)?;
        self.race_pop_guard();
        let written = self.write_log_stack.pop().expect("just pushed, push/pop are balanced");
        r?;
        self.memo = snapshot;
        all_written.extend(written);

        // No if/else value merging in v0: taint every variable written in any
        // arm. Routed through `set_var` (not a raw `memo.insert`) for the same
        // nested-composition correctness as `If`/`IfElse` (a write two levels
        // deep still reaches an enclosing arm's write-log frame).
        for k in all_written {
            self.set_var(k, None);
        }
        Ok(())
    }

    /// Pre-scan a loop body for element-assumption invalidation (module docs'
    /// "Write invalidation"): any global array whose elements the body writes
    /// (recursively) is invalidated for the whole rest of the walk *before* the
    /// body is walked. A later iteration's write happens-before an earlier
    /// iteration's read at runtime, so a read that precedes the write in body
    /// order still cannot be soundly modeled — the in-program-order invalidation
    /// alone would miss it. Conservative (over-invalidates a loop that runs zero
    /// times), hence sound.
    fn invalidate_loop_element_writes(&mut self, scope: &Scope) {
        if self.elem_bounds.is_empty() {
            return; // nothing modelable to invalidate — free for every kernel
        }
        let mut writes = HashSet::new();
        collect_index_assigned_globals(scope, &mut writes);
        self.elem_invalidated.extend(writes);
    }

    fn process_range_loop(&mut self, rl: &cubecl::ir::RangeLoop) -> Result<(), Stop> {
        self.invalidate_loop_element_writes(&rl.scope);
        // Race walk: a barrier inside a range-`for` is a cooperative loop shape
        // v1's structural recognizer does not cover (it keys on the `while`-
        // halving `Branch::Loop`). Rejected `OutOfSubset` rather than
        // mismodeled — the honest answer for an unrecognized-but-valid tree
        // loop (§9 risk 1), never a wrong `Proved`.
        if self.race.is_some() && scope_contains_sync_cube(&rl.scope) {
            return Err(Stop::OutOfSubset(
                "cooperative `Branch::RangeLoop` (a `sync_cube()` inside a range-`for`) is not the \
                 recognized `while`-halving tree loop — outside the vericl v1 subset (rejected \
                 rather than mismodeled)"
                    .into(),
            ));
        }
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

        let i_sym = self.declare_leaf("loop_i", &rl.i.ty)?;
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
        self.invalidate_loop_element_writes(&l.scope);
        // Cooperative loop (barrier inside the body). In the two-thread race
        // walk it routes into the phase walker (milestone M3); in a plain
        // single-thread bounds walk it cannot be modeled without race analysis
        // and stays `OutOfSubset` (unchanged — the `..._defers_to_m3` tests).
        // Checked FIRST, so any barrier-carrying loop takes this path
        // regardless of its guard shape.
        if scope_contains_sync_cube(&l.scope) {
            if self.race.is_some() {
                return self.process_cooperative_loop(l);
            }
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
        let mut induction: HashMap<VariableKind, Type> = HashMap::new();
        for v in &bg.induction_candidates {
            if carried.contains(&v.kind) && is_modeled_int(&v.ty) {
                induction.insert(v.kind, v.ty);
            }
        }

        // Pre-bind before the body walk (like `process_range_loop`): induction
        // vars to a fresh symbol, every other carried var to taint — so a
        // read-before-write of an accumulator sees taint, not the stale
        // pre-loop value.
        for &k in &carried {
            if let Some(ty) = induction.get(&k).copied() {
                let sym = self.declare_leaf("loop_iv_", &ty)?;
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
        // In a race walk, the loop guard is a live fact (recorded into accesses
        // inside the loop) and an enclosing loop guard (barrier uniformity — a
        // non-cooperative loop has no barrier, so a thread-varying guard here is
        // harmless, but a barrier that somehow appeared would be checked).
        self.race_push_guard(&bg.guard_var, guard, true);
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
        self.race_pop_guard();
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
                    self.declare_leaf(&format!("scalar{id}_"), &var.ty).ok()
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

// ---------------------------------------------------------------------------
// Bounded-integer (finite-width) modeling for the overflow-soundness model.
// See the module docs' "Bounded-integer overflow model" bullet. Every modeled
// integer term is kept faithful to real (wrapping) hardware semantics or left
// tainted; these helpers supply the type-width facts that make that possible.
// ---------------------------------------------------------------------------

/// The address-type width vericl models integer *positions*/*lengths* at.
/// `usize`/`isize` lower to `U32`/`I32` in CubeCL 0.10 (see cubecl-ir's
/// `impl_into_variable!`), and the kernels register `AddressType::U32`, so a
/// position/length is a 32-bit value on the hardware this checker reasons
/// about. This is the width of the synthetic leaves (`ABSOLUTE_POS`,
/// `CUBE_POS`, `CUBE_COUNT`, buffer `Length`) that carry no IR `Type` of their
/// own.
fn address_type() -> Type {
    Type::scalar(ElemType::UInt(UIntKind::U32))
}

/// Bit width of a modeled integer type (8/16/32/64) — the element width, so a
/// (rejected-elsewhere) vector type would still report its lane width.
fn int_bits(ty: &Type) -> u32 {
    ty.elem_type().size_bits() as u32
}

/// The inclusive maximum a value of `ty` can hold on hardware: `2^W - 1`
/// (unsigned) or `2^(W-1) - 1` (signed). Returned as `u128` so every width up
/// to 64 bits (`u64::MAX`) is exactly representable.
fn type_max(ty: &Type) -> u128 {
    let w = int_bits(ty);
    if is_unsigned(ty) {
        (1u128 << w) - 1
    } else {
        (1u128 << (w - 1)) - 1
    }
}

/// The inclusive minimum a value of `ty` can hold: `0` (unsigned) or
/// `-2^(W-1)` (signed). Returned as `i128` (negative for signed).
fn type_min(ty: &Type) -> i128 {
    if is_unsigned(ty) {
        0
    } else {
        -(1i128 << (int_bits(ty) - 1))
    }
}

/// The wrap modulus `2^W` of `ty` (the amount added/subtracted to fold an
/// out-of-range mathematical result back into `[type_min, type_max]`).
fn wrap_modulus(ty: &Type) -> u128 {
    1u128 << int_bits(ty)
}

/// Build an SMT integer literal for a possibly-negative `i128` (SMT-LIB spells
/// a negative as `(- n)`, not `-n`, so a bare `numeral` would be malformed —
/// mirrors `constant_expr`'s existing handling of negative `Int` constants).
fn int_const(smt: &Context, v: i128) -> SExpr {
    if v < 0 {
        let mag = smt.numeral(v.unsigned_abs());
        smt.negate(mag)
    } else {
        smt.numeral(v as u128)
    }
}

/// Whether an integer→integer `Cast` from `src` to `dst` is value-preserving
/// for *every* in-range source value (so the SMT term can pass through
/// unchanged): a widening or equal-width cast that does not reinterpret sign.
/// A narrowing cast (truncation) or a same-width signedness flip
/// (reinterpretation) can change the value, so it is not accepted here — the
/// caller instead discharges a "fits the destination range" side-obligation
/// (module docs' "Bounded-integer overflow model", Cast paragraph).
fn cast_is_value_preserving(src: &Type, dst: &Type) -> bool {
    let (sw, dw) = (int_bits(src), int_bits(dst));
    let (su, du) = (is_unsigned(src), is_unsigned(dst));
    if su == du {
        // same signedness: widening or equal width preserves the value.
        dw >= sw
    } else if su {
        // unsigned -> signed: fits only if strictly wider (the top source bit
        // must not land on the destination's sign bit).
        dw > sw
    } else {
        // signed -> unsigned: a negative source reinterprets to a large
        // unsigned; never value-preserving by type alone.
        false
    }
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

/// Every global-array id whose elements `scope` (recursively, through nested
/// branches and loops) writes via an `IndexAssign`/`UncheckedIndexAssign` —
/// used by the loop element-assumption pre-scan
/// (`invalidate_loop_element_writes`). Matches both global output and input
/// arrays by id (element-range assumes are keyed by id regardless of the
/// array's input/output role).
fn collect_index_assigned_globals(scope: &Scope, found: &mut HashSet<Id>) {
    for inst in &scope.instructions {
        if let Operation::Operator(
            Operator::IndexAssign(_) | Operator::UncheckedIndexAssign(_),
        ) = &inst.operation
        {
            if let Some(out) = inst.out {
                match out.kind {
                    VariableKind::GlobalOutputArray(id) | VariableKind::GlobalInputArray(id) => {
                        found.insert(id);
                    }
                    _ => {}
                }
            }
        }
        if let Operation::Branch(b) = &inst.operation {
            match b {
                Branch::If(if_) => collect_index_assigned_globals(&if_.scope, found),
                Branch::IfElse(ie) => {
                    collect_index_assigned_globals(&ie.scope_if, found);
                    collect_index_assigned_globals(&ie.scope_else, found);
                }
                Branch::Switch(sw) => {
                    collect_index_assigned_globals(&sw.scope_default, found);
                    for (_, s) in &sw.cases {
                        collect_index_assigned_globals(s, found);
                    }
                }
                Branch::RangeLoop(rl) => collect_index_assigned_globals(&rl.scope, found),
                Branch::Loop(l) => collect_index_assigned_globals(&l.scope, found),
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

/// Count every `SyncCube` instruction in a scope, recursively (the IR analog of
/// the macro's source-level `sync_cube()` count). Used by the cooperative
/// composition barrier check: cube **inlines** a `uses(...)` helper's IR into the
/// composing kernel's own scope, so if a helper contained a barrier it would
/// show up here as an *extra* `SyncCube` beyond the top-level ones the phase-
/// split twin declared — a silent phase-structure disagreement between the twin
/// and proof lanes. Comparing this count to the twin's declared count
/// (`expected_barriers`, the macro's `COOP_BARRIER_COUNT`) catches exactly that,
/// independently of the macro's own helper gate (docs/design-shared-memory.md
/// §7.4). See `prove_race_freedom_detailed`.
fn count_sync_cube_ir(scope: &Scope) -> usize {
    scope
        .instructions
        .iter()
        .map(|inst| match &inst.operation {
            Operation::Synchronization(Synchronization::SyncCube) => 1,
            Operation::Branch(b) => match b {
                Branch::If(if_) => count_sync_cube_ir(&if_.scope),
                Branch::IfElse(ie) => {
                    count_sync_cube_ir(&ie.scope_if) + count_sync_cube_ir(&ie.scope_else)
                }
                Branch::Switch(sw) => {
                    count_sync_cube_ir(&sw.scope_default)
                        + sw.cases.iter().map(|(_, s)| count_sync_cube_ir(s)).sum::<usize>()
                }
                Branch::RangeLoop(rl) => count_sync_cube_ir(&rl.scope),
                Branch::Loop(l) => count_sync_cube_ir(&l.scope),
                Branch::Return | Branch::Break | Branch::Unreachable => 0,
            },
            _ => 0,
        })
        .sum()
}

// -- thread-varying taint + cooperative-loop recognition (M3 + M4) -------

/// Static thread-varying taint (§5.4): the set of `VariableKind`s whose value
/// depends (transitively) on `UnitPos`/`AbsolutePos` or on array contents. A
/// kind NOT in this set (and not a `UnitPos`/`AbsolutePos` leaf) is provably
/// cube-uniform — identical on every thread — which is exactly what a barrier's
/// enclosing conditions and a cooperative loop's trip count must be. Computed
/// to a fixpoint (a loop-carried variable can turn varying via a later in-body
/// update), forward over the IR; conservative — an unmodeled op's `out` is
/// treated as varying, so a barrier gated by it is rejected, never wrongly
/// accepted.
fn collect_thread_varying(scope: &Scope) -> HashSet<VariableKind> {
    let mut varying = HashSet::new();
    loop {
        let before = varying.len();
        propagate_thread_varying(scope, &mut varying);
        if varying.len() == before {
            return varying;
        }
    }
}

fn propagate_thread_varying(scope: &Scope, varying: &mut HashSet<VariableKind>) {
    for inst in &scope.instructions {
        if let Some(out) = inst.out {
            if inst_out_thread_varying(inst, varying) {
                varying.insert(out.kind);
            }
        }
        if let Operation::Branch(b) = &inst.operation {
            match b {
                Branch::If(if_) => propagate_thread_varying(&if_.scope, varying),
                Branch::IfElse(ie) => {
                    propagate_thread_varying(&ie.scope_if, varying);
                    propagate_thread_varying(&ie.scope_else, varying);
                }
                Branch::Switch(sw) => {
                    propagate_thread_varying(&sw.scope_default, varying);
                    for (_, s) in &sw.cases {
                        propagate_thread_varying(s, varying);
                    }
                }
                Branch::RangeLoop(rl) => propagate_thread_varying(&rl.scope, varying),
                Branch::Loop(l) => propagate_thread_varying(&l.scope, varying),
                Branch::Return | Branch::Break | Branch::Unreachable => {}
            }
        }
    }
}

/// Whether `var` is thread-varying: `UnitPos`/`AbsolutePos` (and the 1-D-subset-
/// external `UnitPosX/Y/Z` variants, seeded for safety) are the leaves;
/// everything else is looked up in `varying`.
fn var_is_thread_varying(var: &Variable, varying: &HashSet<VariableKind>) -> bool {
    matches!(
        var.kind,
        VariableKind::Builtin(
            Builtin::UnitPos
                | Builtin::UnitPosX
                | Builtin::UnitPosY
                | Builtin::UnitPosZ
                | Builtin::AbsolutePos
        )
    ) || varying.contains(&var.kind)
}

/// Whether an instruction's `out` is thread-varying given the current set.
/// Uniform-preserving ops with all-uniform operands stay uniform; an array load
/// is always varying (data-dependent); an unmodeled op is conservatively
/// varying.
fn inst_out_thread_varying(inst: &Instruction, varying: &HashSet<VariableKind>) -> bool {
    match &inst.operation {
        Operation::Copy(v) => var_is_thread_varying(v, varying),
        Operation::Arithmetic(a) => arith_any_varying(a, varying),
        Operation::Comparison(c) => cmp_any_varying(c, varying),
        Operation::Operator(op) => match op {
            // array load -> data-dependent value
            Operator::Index(_) | Operator::UncheckedIndex(_) => true,
            Operator::Cast(u) => var_is_thread_varying(&u.input, varying),
            Operator::And(b) | Operator::Or(b) => {
                var_is_thread_varying(&b.lhs, varying) || var_is_thread_varying(&b.rhs, varying)
            }
            Operator::Not(u) => var_is_thread_varying(&u.input, varying),
            // an IndexAssign's `out` is the array, not a scalar value.
            Operator::IndexAssign(_) | Operator::UncheckedIndexAssign(_) => false,
            _ => true,
        },
        // Buffer metadata (`Length`, ...) is cube-uniform.
        Operation::Metadata(_) => false,
        // Branch/Sync have no scalar `out`; everything else is unmodeled ->
        // conservatively varying.
        _ => true,
    }
}

fn arith_any_varying(a: &Arithmetic, v: &HashSet<VariableKind>) -> bool {
    use Arithmetic::*;
    match a {
        Add(b) | Sub(b) | Mul(b) | Div(b) | Modulo(b) | Max(b) | Min(b) | Remainder(b)
        | Powf(b) | Powi(b) => {
            var_is_thread_varying(&b.lhs, v) || var_is_thread_varying(&b.rhs, v)
        }
        Abs(u) | Neg(u) => var_is_thread_varying(&u.input, v),
        // Fma/Clamp/other shapes: conservatively varying.
        _ => true,
    }
}

fn cmp_any_varying(c: &Comparison, v: &HashSet<VariableKind>) -> bool {
    use Comparison::*;
    match c {
        Lower(b) | LowerEqual(b) | Equal(b) | NotEqual(b) | GreaterEqual(b) | Greater(b) => {
            var_is_thread_varying(&b.lhs, v) || var_is_thread_varying(&b.rhs, v)
        }
        // IsNan/IsInf: float predicate, conservatively varying.
        _ => true,
    }
}

/// The control variable of a downward-counter loop guard `half > 0` (or the
/// symmetric `0 < half`): the non-constant operand the guard lower-bounds. Only
/// the constant-zero-bounded shape is recognized, so `H >= 1` is a sound loop
/// invariant; any other comparison yields `None` (→ `OutOfSubset`).
fn downcounter_ctrl(cmp: &Comparison) -> Option<Variable> {
    match cmp {
        Comparison::Greater(b) if is_zero_const(&b.rhs) => Some(b.lhs),
        Comparison::Lower(b) if is_zero_const(&b.lhs) => Some(b.rhs),
        _ => None,
    }
}

fn is_zero_const(v: &Variable) -> bool {
    matches!(
        v.kind,
        VariableKind::Constant(ConstantValue::UInt(0)) | VariableKind::Constant(ConstantValue::Int(0))
    )
}

/// Verify the cooperative loop's control variable is updated only by a uniform
/// halving `ctrl = ctrl / c` with a constant `c >= 1`, so a fresh symbol
/// bounded `H <= init` soundly over-approximates every tree level (the
/// recurrence is non-increasing). Any other update — a decrement, a manual
/// recurrence, a data-dependent step — is rejected (§9 risk 1: honest
/// `OutOfSubset`, never a wrong `Proved`).
fn verify_halving_update(scope: &Scope, ctrl: VariableKind) -> Result<(), Stop> {
    let mut found = false;
    let all_halving = check_halving_writes(scope, ctrl, &mut found);
    if !all_halving || !found {
        return Err(Stop::OutOfSubset(
            "cooperative loop control variable is not updated by a uniform halving \
             (`half /= <constant>`), the recognized tree-reduction recurrence — outside the \
             vericl v1 subset (rejected rather than mismodeled)"
                .into(),
        ));
    }
    Ok(())
}

/// Returns whether every write to `ctrl` (recursively) is a halving; sets
/// `found` when at least one halving write exists.
fn check_halving_writes(scope: &Scope, ctrl: VariableKind, found: &mut bool) -> bool {
    for inst in &scope.instructions {
        if inst.out.map(|o| o.kind) == Some(ctrl) {
            match &inst.operation {
                Operation::Arithmetic(Arithmetic::Div(b))
                    if b.lhs.kind == ctrl && is_positive_const(&b.rhs) =>
                {
                    *found = true;
                }
                // any other write to the control variable breaks the
                // non-increasing guarantee.
                _ => return false,
            }
        }
        if let Operation::Branch(br) = &inst.operation {
            let sub_ok = match br {
                Branch::If(if_) => check_halving_writes(&if_.scope, ctrl, found),
                Branch::IfElse(ie) => {
                    check_halving_writes(&ie.scope_if, ctrl, found)
                        & check_halving_writes(&ie.scope_else, ctrl, found)
                }
                Branch::RangeLoop(rl) => check_halving_writes(&rl.scope, ctrl, found),
                Branch::Loop(l) => check_halving_writes(&l.scope, ctrl, found),
                Branch::Switch(sw) => {
                    let mut ok = check_halving_writes(&sw.scope_default, ctrl, found);
                    for (_, s) in &sw.cases {
                        ok &= check_halving_writes(s, ctrl, found);
                    }
                    ok
                }
                _ => true,
            };
            if !sub_ok {
                return false;
            }
        }
    }
    true
}

fn is_positive_const(v: &Variable) -> bool {
    matches!(v.kind, VariableKind::Constant(ConstantValue::UInt(n)) if n >= 1)
        || matches!(v.kind, VariableKind::Constant(ConstantValue::Int(n)) if n >= 1)
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

    // =================================================================
    // Bounded-integer overflow model. See the module docs'
    // "Bounded-integer overflow model" bullet. These pin the round-2
    // adversarial-review construction (a `u32` multiply provably nonzero in
    // unbounded LIA but wrapping to exactly 0) and one negative control per
    // consumer of a modeled integer (divisor, guard, index, loop bound),
    // plus positive controls that a genuinely non-wrapping (guard-bounded)
    // arithmetic chain still proves.
    // =================================================================

    /// Two `u32` arrays plus two `u32` scalars `a`, `b` — the overflow tests'
    /// shared signature. Buffers are `[x, y]` (AXPY_BUFFERS); `a`/`b` are
    /// `GlobalScalar`s, not buffers.
    macro_rules! build_ab_kernel {
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
            let a = <u32 as LaunchArg>::expand(&Default::default(), &mut builder);
            let b = <u32 as LaunchArg>::expand(&Default::default(), &mut builder);
            $kernel(&mut builder.scope, x, y, a, b);
            builder.build(KernelSettings::default())
        }};
    }

    /// THE round-2 adversarial-review construction (task headline): `a * b`
    /// guarded by `a >= 1 && b >= 1` is provably nonzero in unbounded QF_LIA,
    /// so the old model modeled `ABSOLUTE_POS % (a*b)` and `Proved` the guarded
    /// write. But real `u32` multiplication wraps `65536 * 65536` to exactly
    /// `0`, so the divisor can be zero on hardware. Under the overflow model the
    /// `Mul` no-overflow side-obligation fails (`a == b == 65536`), `a*b` is
    /// tainted, the modulo divisor is tainted, and the dependent guard is
    /// `OutOfSubset` — never `Proved`. This is the exact verdict flip the
    /// milestone exists to produce.
    #[cube(launch)]
    fn prover_test_mul_overflow_divisor(x: &Array<u32>, y: &mut Array<u32>, a: u32, b: u32) {
        if a >= 1u32 && b >= 1u32 {
            let d = (a * b) as usize;
            let idx = ABSOLUTE_POS % d;
            if idx < y.len() {
                y[idx] = x[0usize];
            }
        }
    }

    #[test]
    fn mul_overflow_divisor_is_out_of_subset() {
        let def = build_ab_kernel!(prover_test_mul_overflow_divisor::expand);
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[]) {
            ProveResult::OutOfSubset { reason } => {
                // The wrapped divisor taints the modulo, so the `if idx <
                // y.len()` guard depends on a construct outside the subset.
                assert!(reason.contains("if"), "unexpected reason: {reason}");
            }
            other => panic!(
                "expected OutOfSubset (wrapped a*b divisor tainted), got {other:?} — \
                 this was `Proved` before the overflow model"
            ),
        }
    }

    /// The `Add` sibling of the divisor construction: `a + b` also wraps
    /// (`a == b == 2^31 ⟹ a + b == 2^32 ⟹ 0`). Unlike `Mul`, `Add` is modeled
    /// *faithfully* (the SMT term is the real wrapped value), so `a + b` is not
    /// tainted — instead the div/mod nonzero side-obligation itself fails,
    /// because the faithful divisor term can be `0`. Either way: the guard that
    /// depends on the modulo is `OutOfSubset`, never `Proved`.
    #[cube(launch)]
    fn prover_test_add_overflow_divisor(x: &Array<u32>, y: &mut Array<u32>, a: u32, b: u32) {
        if a >= 1u32 && b >= 1u32 {
            let d = (a + b) as usize;
            let idx = ABSOLUTE_POS % d;
            if idx < y.len() {
                y[idx] = x[0usize];
            }
        }
    }

    #[test]
    fn add_overflow_divisor_is_out_of_subset() {
        let def = build_ab_kernel!(prover_test_add_overflow_divisor::expand);
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[]) {
            ProveResult::OutOfSubset { reason } => {
                assert!(reason.contains("if"), "unexpected reason: {reason}");
            }
            other => panic!(
                "expected OutOfSubset (faithful a+b divisor can be 0), got {other:?} — \
                 this was `Proved` before the overflow model"
            ),
        }
    }

    /// Wrapped *index* consumer: `a >= 1 && b >= 1 && (a*b) < y.len()` bounds the
    /// product in the OLD model (`a <= a*b < y.len()`), so `y[(a*b) as usize]`
    /// used to `Prove`. Under the overflow model the `Mul` taints `a*b`, so both
    /// the guard sub-condition and the index depend on a construct outside the
    /// subset — `OutOfSubset`.
    #[cube(launch)]
    fn prover_test_mul_overflow_index(_x: &Array<u32>, y: &mut Array<u32>, a: u32, b: u32) {
        if a >= 1u32 && b >= 1u32 && (a * b) < y.len() as u32 {
            y[(a * b) as usize] = 1u32;
        }
    }

    #[test]
    fn mul_overflow_index_is_out_of_subset() {
        let def = build_ab_kernel!(prover_test_mul_overflow_index::expand);
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[]) {
            ProveResult::OutOfSubset { .. } => {}
            other => panic!(
                "expected OutOfSubset (wrapped a*b index tainted), got {other:?} — \
                 this was `Proved` before the overflow model"
            ),
        }
    }

    /// Wrapped *guard* consumer, the "guard bounds a DIFFERENT index" hazard the
    /// task flags as exactly as dangerous as a wrapped divisor: `if (a + b) <
    /// y.len() { y[a] }`. In the OLD unbounded model `a + b < y.len()` implies
    /// `a < y.len()` (since `a <= a + b`), so `y[a]` `Proved`. Under the
    /// faithful overflow model `a + b` wraps (`a == b == 2^31 ⟹ 0 < y.len()`),
    /// the guard passes, and `y[a] == y[2^31]` is out of bounds — the checker
    /// `Refuted`s with that real two-scalar counterexample, catching the danger
    /// rather than being fooled by the monotonicity the unbounded model assumed.
    #[cube(launch)]
    fn prover_test_add_overflow_guard(_x: &Array<u32>, y: &mut Array<u32>, a: u32, b: u32) {
        if (a + b) < y.len() as u32 {
            y[a as usize] = 1u32;
        }
    }

    #[test]
    fn add_overflow_guard_refutes() {
        let def = build_ab_kernel!(prover_test_add_overflow_guard::expand);
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[]) {
            ProveResult::Refuted { obligation, counterexample } => {
                assert!(obligation.contains('y'), "unexpected obligation: {obligation}");
                assert!(!counterexample.is_empty());
            }
            other => panic!(
                "expected Refuted (wrapped guard admits an OOB index), got {other:?} — \
                 this was `Proved` before the overflow model"
            ),
        }
    }

    /// Wrapped *loop bound* consumer: `for i in 0..(a*b) { if i < y.len() {...} }`.
    /// The OLD model modeled the end bound `a*b` and (with the per-iteration
    /// guard) `Proved`. Under the overflow model the `Mul` taints `a*b`, so the
    /// range-loop's end bound depends on a construct outside the subset —
    /// `OutOfSubset`, reported at the loop bound.
    #[cube(launch)]
    fn prover_test_wrapped_loop_bound(x: &Array<u32>, y: &mut Array<u32>, a: u32, b: u32) {
        for i in 0..(a * b) as usize {
            if i < y.len() {
                y[i] = x[i % x.len()];
            }
        }
    }

    #[test]
    fn wrapped_loop_bound_is_out_of_subset() {
        let def = build_ab_kernel!(prover_test_wrapped_loop_bound::expand);
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[Assume::LenEq { a: "x", b: "y" }]) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("end bound") || reason.contains("range-loop"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!(
                "expected OutOfSubset (wrapped loop bound tainted), got {other:?} — \
                 this was `Proved` before the overflow model"
            ),
        }
    }

    /// Positive control (task: "assume-bounded arithmetic proves"): a multiply
    /// whose operand is guard-bounded so the product provably cannot wrap. `a <
    /// 1000` bounds `a`, so `a * 7 < 7000 <= u32::MAX` discharges the `Mul`
    /// no-overflow side-obligation, `d = a * 7` is modeled faithfully, and the
    /// inner `d < y.len()` guard proves the `y[d]`/`x[d]` accesses. Genuinely
    /// non-wrapping arithmetic still proves.
    #[cube(launch)]
    fn prover_test_guard_bounded_mul(x: &Array<u32>, y: &mut Array<u32>, a: u32, _b: u32) {
        if (a as usize) < 1000usize {
            let d = a as usize * 7usize;
            if d < y.len() {
                y[d] = x[d];
            }
        }
    }

    #[test]
    fn guard_bounded_mul_proves() {
        let def = build_ab_kernel!(prover_test_guard_bounded_mul::expand);
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[Assume::LenEq { a: "x", b: "y" }]) {
            // y[d] write, x[d] read.
            ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved (guard-bounded product cannot wrap), got {other:?}"),
        }
    }

    /// Two `u32` arrays only — for the shift/underflow controls below.
    macro_rules! build_u32_xy {
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
            $kernel(&mut builder.scope, x, y);
            builder.build(KernelSettings::default())
        }};
    }

    /// The `fir_pair_kernel`/`tap_pair_guarded_kernel` finding, at the prover
    /// level: a lone `ABSOLUTE_POS + 1 < x.len()` guard does NOT cover the
    /// `x[ABSOLUTE_POS]` read under faithful `u32` wraparound. At `pos ==
    /// u32::MAX`, `pos + 1` wraps to `0 < x.len()`, the guard passes, and
    /// `x[pos]` is out of bounds — `Refuted`. This is why those two example
    /// kernels were strengthened to state `pos < x.len()` explicitly; see their
    /// doc comments and the README "Overflow soundness" note.
    #[cube(launch)]
    fn prover_test_shifted_selfguard(x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS + 1 < x.len() {
            y[ABSOLUTE_POS] = x[ABSOLUTE_POS + 1];
        }
    }

    #[test]
    fn shifted_read_selfguard_refutes_at_type_max() {
        let def = build_u32_xy!(prover_test_shifted_selfguard::expand);
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[Assume::LenEq { a: "x", b: "y" }]) {
            ProveResult::Refuted { obligation, counterexample } => {
                // The write `y[ABSOLUTE_POS]` (== x[ABSOLUTE_POS] by len-eq) is
                // the OOB access at pos == u32::MAX.
                assert!(obligation.contains('y') || obligation.contains('x'));
                assert!(counterexample.contains("abs_pos"), "want abs_pos: {counterexample}");
            }
            other => panic!("expected Refuted (pos+1 guard does not cover pos), got {other:?}"),
        }
    }

    /// The strengthened form (what the examples now ship): stating `pos <
    /// x.len()` AND `pos + 1 < x.len()` proves both accesses and excludes the
    /// wrap point — genuinely-safe shifted-read arithmetic still proves.
    #[cube(launch)]
    fn prover_test_shifted_selfguard_strengthened(x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS < x.len() && ABSOLUTE_POS + 1 < x.len() {
            y[ABSOLUTE_POS] = x[ABSOLUTE_POS + 1];
        }
    }

    #[test]
    fn shifted_read_selfguard_strengthened_proves() {
        let def = build_u32_xy!(prover_test_shifted_selfguard_strengthened::expand);
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[Assume::LenEq { a: "x", b: "y" }]) {
            // y[pos] write, x[pos+1] read.
            ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved (both guards stated), got {other:?}"),
        }
    }

    /// Unguarded unsigned underflow: `x[ABSOLUTE_POS - 1]` with no `pos >= 1`
    /// guard. Faithful `Sub` gives the true wrapped value `2^32 - 1` at `pos ==
    /// 0`, so the read is genuinely out of bounds — `Refuted` (the counterexample
    /// exhibits `pos == 0`). Confirms the faithful underflow model surfaces the
    /// real wrapped index rather than a spurious negative one.
    #[cube(launch)]
    fn prover_test_underflow_unguarded(x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS < y.len() {
            y[ABSOLUTE_POS] = x[ABSOLUTE_POS - 1];
        }
    }

    #[test]
    fn sub_underflow_unguarded_refutes() {
        let def = build_u32_xy!(prover_test_underflow_unguarded::expand);
        match prove_bounds_freedom(&def, AXPY_BUFFERS, &[Assume::LenEq { a: "x", b: "y" }]) {
            ProveResult::Refuted { obligation, .. } => {
                assert!(obligation.contains('x'), "unexpected obligation: {obligation}");
            }
            other => panic!("expected Refuted (pos-1 underflows at pos==0), got {other:?}"),
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

    /// Adversarial review round 5 — permanent regression for the `AbsolutePos`
    /// modular-recomposition soundness fix. A cooperative kernel that guards on
    /// `ABSOLUTE_POS` but indexes with the *unguarded* `CUBE_POS`. The store's
    /// bound obligation is `0 <= cube_pos < output.len()`; the only fact in
    /// scope is the path condition `ABSOLUTE_POS < output.len()`. Under the
    /// pre-fix raw `cube_pos*256 + unit_pos` (unwrapped) model, that guard
    /// implied `cube_pos*256 < output.len()` and so `cube_pos < output.len()`,
    /// a **false `Proved{3}`** — because on real hardware `ABSOLUTE_POS` wraps
    /// mod `2^32`, so `cube_pos = 2^24, unit_pos = 0` gives `ABSOLUTE_POS = 0 <
    /// len` while `output[cube_pos]` is wildly OOB for any `len <= 2^24`. With
    /// the exact modular recomposition (`abs_pos_sym`), `abs_pos = (cube_pos*256
    /// + unit_pos) mod 2^32` is a fresh in-range leaf, so z3 picks exactly that
    /// wrapping witness and **`Refuted`s** with a counterexample exhibiting the
    /// large `cube_pos`. The genuinely-reachable OOB makes `Refuted` (not
    /// `OutOfSubset`) the honest verdict.
    #[cube(launch)]
    fn prover_test_coop_abspos_guard_cubepos_index(output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        tile[tid] = 0.0f32;
        sync_cube();
        if ABSOLUTE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    #[test]
    fn cooperative_abspos_guard_cubepos_index_refutes() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let output = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_coop_abspos_guard_cubepos_index::expand(&mut builder.scope, output);
        let def = builder.build(KernelSettings::default());

        let buffers = [BufferParam { name: "output", is_output: true }];
        match prove_bounds_freedom_cooperative(&def, &buffers, &[], 256) {
            ProveResult::Refuted { obligation, counterexample } => {
                assert!(
                    obligation.contains("output"),
                    "expected the `output[CUBE_POS]` write to refute: {obligation}"
                );
                // The witness is a large `cube_pos` whose recomposition wraps to
                // a small in-range `abs_pos` — the exact hardware behavior the
                // guard cannot see. Both symbols appear in the model.
                assert!(
                    counterexample.contains("cube_pos"),
                    "counterexample should exhibit the offending cube_pos: {counterexample}"
                );
            }
            ProveResult::Proved { obligations } => panic!(
                "FALSE PROOF (unsound): coop abs_pos-guard / cube_pos-index Proved with \
                 {obligations} obligations — hardware wraps abs_pos, so cube_pos can be 2^24 \
                 with output[cube_pos] OOB"
            ),
            other => panic!("expected Refuted (wrapped abs_pos leaves cube_pos unbounded), got {other:?}"),
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

    // =================================================================
    // Shared-memory milestones M3 (two-thread race walker) + M4 (barrier
    // uniformity). docs/design-shared-memory.md §5, §8 M3/M4.
    //
    // These kernels use a store guarded by `CUBE_POS < output.len()` (the M1
    // `shared_load_guarded` posture): the phase walker discharges the *bounds*
    // of every access it resolves — including the store — so the store must be
    // bounds-safe on its own, exactly like the M1 positive control. The whole-
    // kernel `..._defers_to_m3` kernels above leave the store unguarded because
    // the single-thread bounds walk stops at the barrier-carrying loop before
    // ever reaching it.
    // =================================================================

    /// `block_sum_reduce` (guarded store) — the v1 reduction shape.
    #[cube(launch)]
    fn prover_test_race_block_sum_reduce(input: &Array<f32>, output: &mut Array<f32>) {
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

        if tid == 0usize && CUBE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    /// Builds a `KernelDefinition` for a race-test kernel with the standard
    /// `(input: &Array<f32>, output: &mut Array<f32>)` signature.
    macro_rules! build_shared {
        ($kernel:path) => {{
            let mut builder = KernelBuilder::default();
            builder.runtime_properties(Default::default());
            cubecl::ir::AddressType::U32.register(&mut builder.scope);
            let input = <Array<f32> as LaunchArg>::expand(
                &ArrayCompilationArg { inplace: None },
                &mut builder,
            );
            let output = <Array<f32> as LaunchArg>::expand_output(
                &ArrayCompilationArg { inplace: None },
                &mut builder,
            );
            $kernel(&mut builder.scope, input, output);
            builder.build(KernelSettings::default())
        }};
    }

    /// §8 M3 positive control: `block_sum_reduce` `Proved` data-race free by the
    /// two-thread walk, with the previously-deferred bounds obligations (the
    /// tree-reduction `tile[tid+half] < 256`, §8 M1) now discharging through the
    /// phase walker. Reproduces the `tree_ww`/`tree_rw`/`tree_bounds`/`uniform_
    /// ok` probe verdicts through the real walker. Obligations (19):
    ///   bounds (8): phase 0 — `tile[tid]` write (if-arm), `input[ABS]` read
    ///     (if-arm), `tile[tid]` write (else-arm); phase 1 — `tile[tid]` read,
    ///     `tile[tid+half]` read, `tile[tid]` write; phase 2 — `tile[0]` read,
    ///     `output[CUBE_POS]` write.
    ///   write-write (6): phase 0 tile 2×2=4 (both if/else writes, all UNSAT via
    ///     `t1≠t2`); phase 1 tile 1×1=1; phase 2 output 1×1=1 (single-writer via
    ///     `tid==0`).
    ///   read-write (4): phase 1 tile `w1×r2` + `w2×r1` = 1×2 + 1×2.
    ///   inter-cube single-writer gate (1): `output[CUBE_POS]` under `tid==0`.
    #[test]
    fn block_sum_reduce_is_race_free() {
        let def = build_shared!(prover_test_race_block_sum_reduce::expand);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::Proved { obligations } => assert_eq!(obligations, 19),
            other => panic!("expected Proved (race-free), got {other:?}"),
        }
    }

    /// Cooperative-composition barrier check (§7.4, v1.1): `prove_cooperative`
    /// compares the (helper-inlined) IR's `SyncCube` count against the twin's
    /// declared `COOP_BARRIER_COUNT`. `prover_test_race_block_sum_reduce` has
    /// exactly 2 `sync_cube()` barriers.
    ///
    /// - Correct count (2) proves, matching the un-checked `prove_race_freedom`.
    /// - A count of 1 — simulating a `uses(...)` helper that hid a barrier the
    ///   phase-split twin could not see (the twin declared 1, the inlined IR
    ///   carries 2) — is rejected `OutOfSubset` with the barrier-visibility
    ///   reason, BEFORE any SMT work. This is the independent proof-lane half of
    ///   the crux; the twin lane rejects a helper containing `sync_cube` directly.
    #[test]
    fn cooperative_barrier_count_mismatch_is_rejected() {
        let def = build_shared!(prover_test_race_block_sum_reduce::expand);
        assert_eq!(count_sync_cube_ir(&def.body), 2, "the probe kernel has 2 barriers");

        // Correct count: proves (same 19 obligations as the un-checked path).
        match prove_cooperative(&def, SHARED_BUFFERS, &[], 256, 2) {
            CooperativeProof::Proved(o) => assert_eq!(o.bounds + o.race(), 19),
            other => panic!("expected Proved with the correct barrier count, got {other:?}"),
        }

        // Under-declared count: a hidden helper barrier — rejected before SMT.
        match prove_cooperative(&def, SHARED_BUFFERS, &[], 256, 1) {
            CooperativeProof::OutOfSubset { reason } => {
                assert!(
                    reason.contains("sync_cube")
                        && reason.contains("visible at the kernel's top level"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (barrier-count mismatch), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Workgroup-uniform `terminate!()` (v1.1, docs/design-shared-memory.md
    // §4.3/§7.4). The prover models `if <uniform cond> { terminate!() }` at
    // the top level before any barrier as a `!cond` path condition; a
    // non-uniform or post-barrier terminate is rejected on both lanes.
    // -----------------------------------------------------------------

    /// A cooperative kernel whose single-writer store `output[CUBE_POS]` has NO
    /// explicit `CUBE_POS < output.len()` guard — it is bounded ONLY by the
    /// workgroup-uniform `terminate!()` padding guard (`CUBE_POS >= 4`) plus the
    /// contract assume `output.len() == 4`.
    #[cube(launch)]
    fn prover_test_terminate_store(input: &Array<f32>, output: &mut Array<f32>) {
        if CUBE_POS >= 4usize {
            terminate!();
        }
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        } else {
            tile[tid] = 0.0f32;
        }
        sync_cube();
        if tid == 0usize {
            output[CUBE_POS] = tile[0usize];
        }
    }

    /// The same kernel WITHOUT the terminate guard — the store is then genuinely
    /// unbounded (`CUBE_POS` is a free `u32` leaf), so it must `Refute`.
    #[cube(launch)]
    fn prover_test_no_terminate_store(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        } else {
            tile[tid] = 0.0f32;
        }
        sync_cube();
        if tid == 0usize {
            output[CUBE_POS] = tile[0usize];
        }
    }

    /// The terminate is **load-bearing**: with the `terminate!()` guard the store
    /// `output[CUBE_POS]` proves in bounds (the modeled `!(CUBE_POS >= 4)` path
    /// condition plus `output.len() == 4` bound it), and the SAME kernel WITHOUT
    /// the terminate `Refutes` on that store (the store index is then unbounded).
    /// Pins that the prover genuinely models the terminate as a path condition,
    /// not merely accepts it. `output.len() == 4` matches `CUBE_POS < 4`.
    #[test]
    fn terminate_is_load_bearing_for_the_store_bound() {
        let assume = &[Assume::LenEqConst { a: "output", value: 4 }];

        let def = build_shared!(prover_test_terminate_store::expand);
        // 1 barrier (the pre-tree sync); the terminate adds none.
        assert_eq!(count_sync_cube_ir(&def.body), 1);
        match prove_cooperative(&def, SHARED_BUFFERS, assume, 256, 1) {
            CooperativeProof::Proved(_) => {}
            other => panic!("expected Proved with the terminate guard, got {other:?}"),
        }

        let def_no = build_shared!(prover_test_no_terminate_store::expand);
        match prove_race_freedom(&def_no, SHARED_BUFFERS, assume, 256) {
            ProveResult::Refuted { obligation, .. } => {
                assert!(obligation.contains("output"), "unexpected obligation: {obligation}");
            }
            other => panic!("expected Refuted without the terminate guard, got {other:?}"),
        }
    }

    /// A **thread-varying** terminate condition (`if UNIT_POS < 128 { terminate!()
    /// }`) is barrier divergence — some threads skip the cube, others reach the
    /// barrier. Rejected `OutOfSubset`, never a silent `Proved` (§4.3/§7.4). The
    /// twin lane rejects the identical shape.
    #[cube(launch)]
    fn prover_test_terminate_nonuniform(input: &Array<f32>, output: &mut Array<f32>) {
        if UNIT_POS < 128u32 {
            terminate!();
        }
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        } else {
            tile[tid] = 0.0f32;
        }
        sync_cube();
        if tid == 0usize && CUBE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    #[test]
    fn nonuniform_terminate_is_out_of_subset() {
        let def = build_shared!(prover_test_terminate_nonuniform::expand);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::OutOfSubset { reason } => assert!(
                reason.contains("thread-varying") && reason.contains("terminate"),
                "unexpected reason: {reason}"
            ),
            other => panic!("expected OutOfSubset (non-uniform terminate), got {other:?}"),
        }
    }

    /// A **post-barrier** terminate (`terminate!()` after a `sync_cube()`) leaves
    /// the twin's "skip the whole cube" model — rejected `OutOfSubset` (§4.3/§7.4).
    #[cube(launch)]
    fn prover_test_terminate_post_barrier(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        } else {
            tile[tid] = 0.0f32;
        }
        sync_cube();
        if CUBE_POS >= 4usize {
            terminate!();
        }
        if tid == 0usize && CUBE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    #[test]
    fn post_barrier_terminate_is_out_of_subset() {
        let def = build_shared!(prover_test_terminate_post_barrier::expand);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::OutOfSubset { reason } => assert!(
                reason.contains("after a barrier") && reason.contains("terminate"),
                "unexpected reason: {reason}"
            ),
            other => panic!("expected OutOfSubset (post-barrier terminate), got {other:?}"),
        }
    }

    /// A device fn that indexes a `SharedMemory` tile passed as a parameter.
    /// vericl **rejects** this shape at the twin lane (a #[vericl::helper] taking
    /// a `SharedMemory` param is a compile error — `SharedTile` is twin-local, so
    /// it cannot cross a helper boundary in v1.1), so this shape never reaches the
    /// prover through the vericl pipeline. This kernel exists only to *verify what
    /// the inlined IR looks like* and confirm the prover handles it **soundly**
    /// (docs/design-shared-memory.md §7.4): cube inlines the helper body into the
    /// caller's scope, so the shared read appears as an ordinary `SharedMemory`
    /// access the two-thread walk models against the tile's compile-time length —
    /// no false `Proved`.
    #[cube]
    fn prover_test_helper_reads_tile(tile: &SharedMemory<f32>, i: usize) -> f32 {
        tile[i]
    }

    #[cube(launch)]
    fn prover_test_shared_into_helper(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        } else {
            tile[tid] = 0.0f32;
        }
        sync_cube();
        if tid == 0usize && CUBE_POS < output.len() {
            output[CUBE_POS] = prover_test_helper_reads_tile(&tile, tid);
        }
    }

    /// Verifies the prover soundly handles an inlined shared-tile-into-helper
    /// access. cube inlines `prover_test_helper_reads_tile`, so the shared read
    /// `tile[tid]` (with `tid < cube_dim == 256` and the tile length 256) is
    /// modeled as an ordinary shared access and **proves** — no false `Proved`,
    /// the shared length bound still applies through inlining. (In the vericl
    /// pipeline the shape is unreachable — the twin lane rejects the helper — so
    /// this is a defense-in-depth verification, not a supported path.)
    #[test]
    fn shared_tile_into_helper_is_soundly_modeled() {
        let def = build_shared!(prover_test_shared_into_helper::expand);
        // 1 barrier in this kernel; the inlined helper adds none.
        assert_eq!(count_sync_cube_ir(&def.body), 1);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::Proved { .. } => {}
            other => panic!("expected the inlined shared read to be soundly Proved, got {other:?}"),
        }
    }

    /// `grid_stride_reduce` (guarded store) — the reduce_rssi-shaped reduction:
    /// a non-cooperative accumulation loop feeding the same tree reduction.
    #[cube(launch)]
    fn prover_test_race_grid_stride_reduce(
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

        if tid == 0usize && CUBE_POS < partials.len() {
            partials[CUBE_POS] = tile[0usize];
        }
    }

    fn build_grid_stride_race() -> KernelDefinition {
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
        prover_test_race_grid_stride_reduce::expand(&mut builder.scope, data, partials, num_cubes, 4096);
        builder.build(KernelSettings::default())
    }

    const GRID_BUFFERS: &[BufferParam] = &[
        BufferParam { name: "data", is_output: false },
        BufferParam { name: "partials", is_output: true },
    ];

    /// §8 M3 positive control: `grid_stride_reduce` `Proved` race-free. The
    /// non-cooperative accumulation loop's `data[k]` reads (read-only, no race)
    /// discharge their bounds under `data.len() == 4096` (M2), the tree
    /// reduction proves race-free exactly as `block_sum_reduce`. Obligations
    /// (16): bounds (8) = phase 0 `data[k]` read ×2 + `tile[tid]` write; phase 1
    /// tree ×3; phase 2 `tile[0]` read + `partials[CUBE_POS]` write. write-write
    /// (3) = phase 0 tile 1×1 (unconditional store) + phase 1 tile 1 + phase 2
    /// partials 1. read-write (4) = phase 1. inter-cube gate (1).
    #[test]
    fn grid_stride_reduce_is_race_free() {
        let def = build_grid_stride_race();
        match prove_race_freedom(
            &def,
            GRID_BUFFERS,
            &[Assume::LenEqConst { a: "data", value: 4096 }],
            256,
        ) {
            ProveResult::Proved { obligations } => assert_eq!(obligations, 16),
            other => panic!("expected Proved (race-free), got {other:?}"),
        }
    }

    /// §8 M3 negative control (the `racy_rw` probe, through the real walker): a
    /// neighbor combine `tile[tid] += tile[tid+1]` with **no barrier** before it
    /// — a read-write race between adjacent threads (thread `t1`'s write of
    /// `tile[t1]` collides with thread `t2`'s read of `tile[t2+1]` when
    /// `t1 == t2+1`). All accesses are bounds-safe (the `tid < 255` guard keeps
    /// the neighbor read in range), so the walker refutes on the **race**, not
    /// on bounds, with a two-thread counterexample.
    #[cube(launch)]
    fn prover_test_racy_neighbor(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        } else {
            tile[tid] = 0.0f32;
        }
        // BUG: the correct kernel has a `sync_cube()` here.
        if tid < 255usize {
            let n = tile[tid + 1usize];
            tile[tid] = tile[tid] + n;
        }
        if tid == 0usize && CUBE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    #[test]
    fn racy_neighbor_read_write_refutes() {
        let def = build_shared!(prover_test_racy_neighbor::expand);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::Refuted { obligation, counterexample } => {
                assert!(
                    obligation.contains("read-write race") && obligation.contains("shared_array"),
                    "unexpected obligation: {obligation}"
                );
                assert!(
                    counterexample.contains("t1") && counterexample.contains("t2"),
                    "expected a two-thread counterexample: {counterexample}"
                );
            }
            other => panic!("expected Refuted (neighbor RW race), got {other:?}"),
        }
    }

    /// Write-write race: every thread writes `tile[0]` (a fixed index), so any
    /// two threads collide there — `Refuted` with a two-thread counterexample.
    #[cube(launch)]
    fn prover_test_racy_ww(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[0usize] = input[ABSOLUTE_POS];
        }
        if tid == 0usize && CUBE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    #[test]
    fn racy_write_write_refutes() {
        let def = build_shared!(prover_test_racy_ww::expand);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::Refuted { obligation, counterexample } => {
                assert!(
                    obligation.contains("write-write race"),
                    "unexpected obligation: {obligation}"
                );
                assert!(!counterexample.is_empty());
            }
            other => panic!("expected Refuted (WW race on tile[0]), got {other:?}"),
        }
    }

    /// §8 M4 negative control (the `uniform_bad` probe, through the real
    /// walker): a `sync_cube()` under the thread-varying condition `tid < half`
    /// is barrier divergence — `OutOfSubset` with the §7.3 reason, never a
    /// silent `Proved`. This is the analog of the round-2 branch-scoping bug the
    /// design flags as adversarially probed (§9 risk 2).
    #[cube(launch)]
    fn prover_test_barrier_divergence(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        } else {
            tile[tid] = 0.0f32;
        }
        let half = CUBE_DIM as usize / 2;
        if tid < half {
            sync_cube();
        }
        if tid == 0usize && CUBE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    #[test]
    fn barrier_under_thread_varying_condition_is_out_of_subset() {
        let def = build_shared!(prover_test_barrier_divergence::expand);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("non-uniform condition")
                        && reason.contains("barrier divergence"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (barrier divergence), got {other:?}"),
        }
    }

    /// §8 M4 positive control (the `uniform_ok` probe, through the real walker):
    /// a `sync_cube()` at the top level of a `while half > 0` loop — a uniform
    /// (`half`-halving) trip count — is accepted. The barrier body is otherwise
    /// empty, isolating the uniformity check from any race obligation.
    #[cube(launch)]
    fn prover_test_uniform_barrier_loop(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        } else {
            tile[tid] = 0.0f32;
        }
        let mut half = CUBE_DIM as usize / 2;
        while half > 0usize {
            sync_cube();
            half /= 2usize;
        }
        if tid == 0usize && CUBE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    #[test]
    fn uniform_tree_guard_barrier_is_accepted() {
        let def = build_shared!(prover_test_uniform_barrier_loop::expand);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::Proved { .. } => {}
            other => panic!("expected Proved (uniform barrier loop accepted), got {other:?}"),
        }
    }

    /// §9 risk 1 (cooperative-loop recognition brittleness), demonstrated: a
    /// barrier inside a range-`for` is a valid-but-unrecognized tree-loop shape.
    /// The honest answer is `OutOfSubset`, never a wrong `Proved`.
    #[cube(launch)]
    fn prover_test_barrier_in_range_loop(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        } else {
            tile[tid] = 0.0f32;
        }
        for _i in 0..8usize {
            sync_cube();
        }
        if tid == 0usize && CUBE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    #[test]
    fn barrier_in_range_loop_is_out_of_subset() {
        let def = build_shared!(prover_test_barrier_in_range_loop::expand);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("RangeLoop") || reason.contains("range-`for`"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (barrier in RangeLoop), got {other:?}"),
        }
    }

    /// §9 risk 1, second demonstration: the tree loop written with a *decrement*
    /// (`half -= 1`) rather than a halving. Semantically it is still a uniform,
    /// race-free, in-bounds barrier loop, but v1's structural recognizer keys on
    /// the halving recurrence — so the honest answer is `OutOfSubset`, never a
    /// wrong `Proved`.
    #[cube(launch)]
    fn prover_test_decrement_tree(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        } else {
            tile[tid] = 0.0f32;
        }
        let mut half = CUBE_DIM as usize / 2;
        while half > 0usize {
            if tid < half {
                let a = tile[tid];
                let b = tile[tid + half];
                tile[tid] = a + b;
            }
            sync_cube();
            half -= 1usize;
        }
        if tid == 0usize && CUBE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    #[test]
    fn non_halving_tree_update_is_out_of_subset() {
        let def = build_shared!(prover_test_decrement_tree::expand);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::OutOfSubset { reason } => {
                assert!(reason.contains("halving"), "unexpected reason: {reason}");
            }
            other => panic!("expected OutOfSubset (non-halving update), got {other:?}"),
        }
    }

    /// §8 M1 bounds discharge through the phase walker, negative: an undersized
    /// tile (`SharedMemory::new(128)` at `CUBE_DIM = 256`) makes the very first
    /// `tile[tid]` store out of bounds — the race walker's bounds obligation
    /// refutes with a counterexample exhibiting the offending `unit_pos`, rather
    /// than vacuously proving race freedom over an OOB kernel.
    #[cube(launch)]
    fn prover_test_race_undersized(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(128usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        }
        sync_cube();
        if tid == 0usize && CUBE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    #[test]
    fn race_walker_undersized_tile_refutes_on_bounds() {
        let def = build_shared!(prover_test_race_undersized::expand);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::Refuted { obligation, counterexample } => {
                assert!(
                    obligation.contains("shared_array") && obligation.contains("index"),
                    "unexpected obligation: {obligation}"
                );
                assert!(
                    counterexample.contains("unit_pos") || counterexample.contains('t'),
                    "unexpected counterexample: {counterexample}"
                );
            }
            other => panic!("expected Refuted (undersized tile bounds), got {other:?}"),
        }
    }

    /// §9 risk 2 (barrier-uniformity taint looseness), the subtle case: a
    /// `sync_cube()` under the divergent inner `if tid < half` *inside* the
    /// otherwise-recognized halving tree loop. The loop trip count is uniform,
    /// so the loop itself is fine — but the barrier now sits under a
    /// thread-varying `if`, which is divergence. The walker must catch the
    /// *inner* guard (not just the outer loop guard) and reject. This is the
    /// analog of the round-2 branch-scoping bug the design promises will be
    /// adversarially probed.
    #[cube(launch)]
    fn prover_test_barrier_under_inner_if(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        } else {
            tile[tid] = 0.0f32;
        }
        let mut half = CUBE_DIM as usize / 2;
        while half > 0usize {
            if tid < half {
                let a = tile[tid];
                let b = tile[tid + half];
                tile[tid] = a + b;
                sync_cube();
            }
            half /= 2usize;
        }
        if tid == 0usize && CUBE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    #[test]
    fn barrier_under_inner_if_in_coop_loop_is_out_of_subset() {
        let def = build_shared!(prover_test_barrier_under_inner_if::expand);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("non-uniform condition")
                        && reason.contains("barrier divergence"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (barrier under inner varying if), got {other:?}"),
        }
    }

    /// Inter-cube gate (§5.3): a global store `output[tid]` collides across
    /// cubes (cube A's `tid==5` and cube B's `tid==5` both write index 5). The
    /// same-cube pair sees no intra-cube race (distinct `tid`), so the walker
    /// must NOT silently `Proved` — the inter-cube gate rejects the pattern
    /// (`OutOfSubset`), since it is neither `out[ABSOLUTE_POS]` nor a
    /// single-writer `out[CUBE_POS]`. The store is `tid`-guarded against
    /// `output.len()` so it is bounds-safe and the gate (not bounds) is what
    /// fires.
    #[cube(launch)]
    fn prover_test_global_write_bad_pattern(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        if ABSOLUTE_POS < input.len() && tid < output.len() {
            output[tid] = input[ABSOLUTE_POS];
        }
    }

    #[test]
    fn intercube_global_write_bad_pattern_is_out_of_subset() {
        let def = build_shared!(prover_test_global_write_bad_pattern::expand);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("inter-cube") && reason.contains("global write"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (inter-cube global pattern), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Element-range assumptions (array-value-dependent indices / gather).
    // See the module docs' "Element-range assumptions" and "Write
    // invalidation" bullets.
    // -----------------------------------------------------------------

    /// A gather: `y[i] = x[offsets[i]]`. The inner `x[offsets[i]]` read is only
    /// provable if the value `offsets[i]` is modeled `< x.len()` — the whole
    /// point of an element-range assume.
    #[cube(launch)]
    fn prover_test_gather(x: &Array<f32>, offsets: &Array<u32>, y: &mut Array<f32>) {
        if ABSOLUTE_POS < y.len() {
            y[ABSOLUTE_POS] = x[offsets[ABSOLUTE_POS] as usize];
        }
    }

    fn build_gather() -> KernelDefinition {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let x = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let offsets =
            <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_gather::expand(&mut builder.scope, x, offsets, y);
        builder.build(KernelSettings::default())
    }

    const GATHER_BUFFERS: &[BufferParam] = &[
        BufferParam { name: "x", is_output: false },
        BufferParam { name: "offsets", is_output: false },
        BufferParam { name: "y", is_output: true },
    ];

    /// Positive: with `offsets.len() == y.len()` (the inner-read guard cover)
    /// and `offsets[·] < x.len()`, all three obligations discharge — the
    /// value-dependent index the checker never used to model.
    #[test]
    fn gather_with_element_assume_proves() {
        let def = build_gather();
        let assumes = [
            Assume::LenEq { a: "offsets", b: "y" },
            Assume::ElemsBelowLen { arr: "offsets", len_of: "x" },
        ];
        match prove_bounds_freedom(&def, GATHER_BUFFERS, &assumes) {
            // offsets[pos] read, x[elem] read, y[pos] write.
            ProveResult::Proved { obligations } => assert_eq!(obligations, 3),
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    /// Without the element assume, the loaded `offsets[i]` is tainted (contents
    /// are opaque), so `x[offsets[i]]`'s index depends on a construct outside
    /// the subset — honest `OutOfSubset` at that read, never a guess.
    #[test]
    fn gather_without_element_assume_is_out_of_subset() {
        let def = build_gather();
        let assumes = [Assume::LenEq { a: "offsets", b: "y" }];
        match prove_bounds_freedom(&def, GATHER_BUFFERS, &assumes) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("read index") && reason.contains('x'),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (offsets[i] tainted), got {other:?}"),
        }
    }

    /// A *wrong* (too-loose) element bound does not hide the bug: nothing ties
    /// the bound to `x.len()`, so z3 picks an in-assume `elem` that overruns `x`
    /// and refutes, with the fresh element symbol in the counterexample.
    #[test]
    fn gather_with_wrong_bound_refutes_with_element_symbol() {
        let def = build_gather();
        let assumes = [
            Assume::LenEq { a: "offsets", b: "y" },
            Assume::ElemsBelowConst { arr: "offsets", bound: 1_000_000 },
        ];
        match prove_bounds_freedom(&def, GATHER_BUFFERS, &assumes) {
            ProveResult::Refuted { obligation, counterexample } => {
                assert!(obligation.contains('x'), "unexpected obligation: {obligation}");
                assert!(
                    counterexample.contains("elem") && counterexample.contains("len_x"),
                    "counterexample should exhibit the element symbol at the x boundary: \
                     {counterexample}"
                );
            }
            other => panic!("expected Refuted (element overruns x), got {other:?}"),
        }
    }

    /// Nested gather `y[i] = data[inner[outer[i]]]` with an element assume on
    /// each index layer: the fresh symbol from `outer[i]` (bounded `< inner`)
    /// is exactly the index `inner[·]` needs, whose own fresh symbol (bounded
    /// `< data`) is what `data[·]` needs — the layers compose with no special
    /// casing.
    #[cube(launch)]
    fn prover_test_nested_gather(
        data: &Array<f32>,
        inner: &Array<u32>,
        outer: &Array<u32>,
        y: &mut Array<f32>,
    ) {
        if ABSOLUTE_POS < y.len() {
            y[ABSOLUTE_POS] = data[inner[outer[ABSOLUTE_POS] as usize] as usize];
        }
    }

    #[test]
    fn nested_gather_composes_and_proves() {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(Default::default());
        cubecl::ir::AddressType::U32.register(&mut builder.scope);
        let data = <Array<f32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let inner =
            <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let outer =
            <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
        let y = <Array<f32> as LaunchArg>::expand_output(
            &ArrayCompilationArg { inplace: None },
            &mut builder,
        );
        prover_test_nested_gather::expand(&mut builder.scope, data, inner, outer, y);
        let def = builder.build(KernelSettings::default());

        let buffers = [
            BufferParam { name: "data", is_output: false },
            BufferParam { name: "inner", is_output: false },
            BufferParam { name: "outer", is_output: false },
            BufferParam { name: "y", is_output: true },
        ];
        let assumes = [
            Assume::LenEq { a: "outer", b: "y" },
            Assume::ElemsBelowLen { arr: "outer", len_of: "inner" },
            Assume::ElemsBelowLen { arr: "inner", len_of: "data" },
        ];
        match prove_bounds_freedom(&def, &buffers, &assumes) {
            // outer[pos], inner[elem], data[elem] reads + y[pos] write.
            ProveResult::Proved { obligations } => assert_eq!(obligations, 4),
            other => panic!("expected Proved (nested gather composes), got {other:?}"),
        }
    }

    // ---- write invalidation, both directions ------------------------

    /// Read BEFORE write: the modeled `offsets[i]` (used as the `x` index) is
    /// resolved before the later `offsets[i] = 7` write, so the assumption is
    /// still in force there and the gather proves — invalidation must not act
    /// retroactively.
    #[cube(launch)]
    fn prover_test_read_before_write(offsets: &mut Array<u32>, x: &Array<f32>, y: &mut Array<f32>) {
        if ABSOLUTE_POS < y.len() {
            let t = offsets[ABSOLUTE_POS];
            y[ABSOLUTE_POS] = x[t as usize];
            offsets[ABSOLUTE_POS] = 7u32;
        }
    }

    /// Write BEFORE read: the `offsets[i] = 7` write invalidates the assumption,
    /// so the subsequent `offsets[i]` load is tainted and `x[offsets[i]]` is
    /// `OutOfSubset` — the write-invalidation rule firing.
    #[cube(launch)]
    #[allow(unused_assignments)]
    fn prover_test_write_before_read(offsets: &mut Array<u32>, x: &Array<f32>, y: &mut Array<f32>) {
        if ABSOLUTE_POS < y.len() {
            offsets[ABSOLUTE_POS] = 7u32;
            let t = offsets[ABSOLUTE_POS];
            y[ABSOLUTE_POS] = x[t as usize];
        }
    }

    /// A loop that both reads and writes the assume array: even though the read
    /// textually precedes the write, a later iteration's write happens-before
    /// this iteration's read, so the loop pre-scan invalidates `offsets` for the
    /// whole body and the gather is `OutOfSubset` — never a wrong `Proved`.
    #[cube(launch)]
    fn prover_test_loop_self_mutation(offsets: &mut Array<u32>, x: &Array<f32>, y: &mut Array<f32>) {
        for i in 0..y.len() {
            let t = offsets[i];
            y[i] = x[t as usize];
            offsets[i] = 3u32;
        }
    }

    macro_rules! build_offsets_mut {
        ($kernel:path) => {{
            let mut builder = KernelBuilder::default();
            builder.runtime_properties(Default::default());
            cubecl::ir::AddressType::U32.register(&mut builder.scope);
            let offsets = <Array<u32> as LaunchArg>::expand_output(
                &ArrayCompilationArg { inplace: None },
                &mut builder,
            );
            let x = <Array<f32> as LaunchArg>::expand(
                &ArrayCompilationArg { inplace: None },
                &mut builder,
            );
            let y = <Array<f32> as LaunchArg>::expand_output(
                &ArrayCompilationArg { inplace: None },
                &mut builder,
            );
            $kernel(&mut builder.scope, offsets, x, y);
            builder.build(KernelSettings::default())
        }};
    }

    /// Buffers for the write-invalidation kernels (offsets is `&mut`, so it
    /// registers first as an output; then x input, y output).
    const OFFSETS_MUT_BUFFERS: &[BufferParam] = &[
        BufferParam { name: "offsets", is_output: true },
        BufferParam { name: "x", is_output: false },
        BufferParam { name: "y", is_output: true },
    ];

    #[test]
    fn element_read_before_write_stays_modeled_and_proves() {
        let def = build_offsets_mut!(prover_test_read_before_write::expand);
        let assumes = [
            Assume::LenEq { a: "offsets", b: "y" },
            Assume::ElemsBelowLen { arr: "offsets", len_of: "x" },
        ];
        match prove_bounds_freedom(&def, OFFSETS_MUT_BUFFERS, &assumes) {
            // offsets[pos] read, x[elem] read, y[pos] write, offsets[pos] write.
            ProveResult::Proved { obligations } => assert_eq!(obligations, 4),
            other => panic!("expected Proved (read precedes write), got {other:?}"),
        }
    }

    #[test]
    fn element_write_invalidates_subsequent_read() {
        let def = build_offsets_mut!(prover_test_write_before_read::expand);
        let assumes = [
            Assume::LenEq { a: "offsets", b: "y" },
            Assume::ElemsBelowLen { arr: "offsets", len_of: "x" },
        ];
        match prove_bounds_freedom(&def, OFFSETS_MUT_BUFFERS, &assumes) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("read index") && reason.contains('x'),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (write invalidated the assume), got {other:?}"),
        }
    }

    #[test]
    fn loop_self_mutation_invalidates_element_reads() {
        let def = build_offsets_mut!(prover_test_loop_self_mutation::expand);
        let assumes = [
            Assume::LenEq { a: "offsets", b: "y" },
            Assume::ElemsBelowLen { arr: "offsets", len_of: "x" },
        ];
        match prove_bounds_freedom(&def, OFFSETS_MUT_BUFFERS, &assumes) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("read index") && reason.contains('x'),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!(
                "expected OutOfSubset (loop pre-scan invalidates the self-mutated array), \
                 got {other:?}"
            ),
        }
    }

    // =================================================================
    // Switch modeling (Rust `match` on an integer) — module docs'
    // "Switch modeling (match on integers)" bullet. Positive/negative
    // pairs for: guarded access, per-arm write leak past the merge (the
    // round-2 If/IfElse manifestations replayed through Switch), default-
    // arm reachability (does the default receive the case negations?), and
    // race + switch (a barrier inside a switch arm is barrier divergence).
    // =================================================================

    macro_rules! build_mode_u32_xy {
        ($kernel:path) => {{
            let mut builder = KernelBuilder::default();
            builder.runtime_properties(Default::default());
            cubecl::ir::AddressType::U32.register(&mut builder.scope);
            let mode = <u32 as LaunchArg>::expand(&Default::default(), &mut builder);
            let x =
                <Array<u32> as LaunchArg>::expand(&ArrayCompilationArg { inplace: None }, &mut builder);
            let y = <Array<u32> as LaunchArg>::expand_output(
                &ArrayCompilationArg { inplace: None },
                &mut builder,
            );
            $kernel(&mut builder.scope, mode, x, y);
            builder.build(KernelSettings::default())
        }};
    }

    macro_rules! build_mode_f32_y {
        ($kernel:path) => {{
            let mut builder = KernelBuilder::default();
            builder.runtime_properties(Default::default());
            cubecl::ir::AddressType::U32.register(&mut builder.scope);
            let mode = <u32 as LaunchArg>::expand(&Default::default(), &mut builder);
            let y = <Array<f32> as LaunchArg>::expand_output(
                &ArrayCompilationArg { inplace: None },
                &mut builder,
            );
            $kernel(&mut builder.scope, mode, y);
            builder.build(KernelSettings::default())
        }};
    }

    const MODE_XY_BUFFERS: &[BufferParam] =
        &[BufferParam { name: "x", is_output: false }, BufferParam { name: "y", is_output: true }];

    /// Positive: a guarded `match mode` where every arm writes `y[pos]` and
    /// reads `x[pos]` under `if ABSOLUTE_POS < y.len()` proves. Each arm
    /// (case 0, case 1, default) re-emits its own obligations: 3 arms × (x
    /// read + y write) = 6.
    #[cube(launch)]
    fn prover_test_switch_guarded(mode: u32, x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS < y.len() {
            match mode {
                0 => {
                    y[ABSOLUTE_POS] = x[ABSOLUTE_POS] + 1u32;
                }
                1 => {
                    y[ABSOLUTE_POS] = x[ABSOLUTE_POS] + 2u32;
                }
                _ => {
                    y[ABSOLUTE_POS] = x[ABSOLUTE_POS];
                }
            }
        }
    }

    #[test]
    fn switch_guarded_access_proves() {
        let def = build_mode_u32_xy!(prover_test_switch_guarded::expand);
        match prove_bounds_freedom(&def, MODE_XY_BUFFERS, &[Assume::LenEq { a: "x", b: "y" }]) {
            ProveResult::Proved { obligations } => assert_eq!(obligations, 6),
            other => panic!("expected Proved (guarded match), got {other:?}"),
        }
    }

    /// Positive: an `Or`-pattern arm (`1 | 2`) lowers to two separate cases
    /// sharing a cloned body — the guarded access still proves. The cloned
    /// body re-emits its obligations, so there are 4 arms (0, 1, 2, default)
    /// × 2 = 8.
    #[cube(launch)]
    fn prover_test_switch_or_pattern(mode: u32, x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS < y.len() {
            match mode {
                0 => {
                    y[ABSOLUTE_POS] = x[ABSOLUTE_POS] + 1u32;
                }
                1 | 2 => {
                    y[ABSOLUTE_POS] = x[ABSOLUTE_POS] + 3u32;
                }
                _ => {
                    y[ABSOLUTE_POS] = x[ABSOLUTE_POS];
                }
            }
        }
    }

    #[test]
    fn switch_or_pattern_arm_proves() {
        let def = build_mode_u32_xy!(prover_test_switch_or_pattern::expand);
        match prove_bounds_freedom(&def, MODE_XY_BUFFERS, &[Assume::LenEq { a: "x", b: "y" }]) {
            ProveResult::Proved { obligations } => assert_eq!(obligations, 8),
            other => panic!("expected Proved (or-pattern match), got {other:?}"),
        }
    }

    /// Negative control for `switch_guarded_access_proves`: the SAME arms with
    /// NO `if ABSOLUTE_POS < y.len()` guard — the switch does not magically
    /// make an unguarded access safe, so it refutes.
    #[cube(launch)]
    fn prover_test_switch_unguarded(mode: u32, x: &Array<u32>, y: &mut Array<u32>) {
        match mode {
            0 => {
                y[ABSOLUTE_POS] = x[ABSOLUTE_POS] + 1u32;
            }
            _ => {
                y[ABSOLUTE_POS] = x[ABSOLUTE_POS];
            }
        }
    }

    #[test]
    fn switch_unguarded_access_refutes() {
        let def = build_mode_u32_xy!(prover_test_switch_unguarded::expand);
        match prove_bounds_freedom(&def, MODE_XY_BUFFERS, &[Assume::LenEq { a: "x", b: "y" }]) {
            ProveResult::Refuted { obligation, counterexample } => {
                assert!(obligation.contains('y') || obligation.contains('x'));
                assert!(!counterexample.is_empty());
            }
            other => panic!("expected Refuted (unguarded match access), got {other:?}"),
        }
    }

    // -- Per-arm write leak past the merge (round-2 manifestations, Switch) --

    /// Manifestation 1 through Switch: a variable clamped to a safe value
    /// inside a case arm must not leak that clamp past the switch. `idx` is
    /// really `ABSOLUTE_POS` (unbounded vs `y.len() == 1`) on every thread
    /// where `mode != 0`; if the case-0 clamp leaked, the post-switch
    /// `y[idx]` would look like `y[0]` (safe) unconditionally. Post-fix: `idx`
    /// is tainted (written in an arm) → `OutOfSubset` at the write, never
    /// `Proved`.
    // The single-case-plus-default `match` is deliberate: it must lower to a
    // `Branch::Switch` (not an `if`) to exercise the switch write-taint path.
    #[allow(clippy::single_match)]
    #[cube(launch)]
    fn prover_test_switch_write_leak(mode: u32, y: &mut Array<f32>) {
        let mut idx: usize = ABSOLUTE_POS;
        match mode {
            0 => {
                idx = 0usize;
            }
            _ => {}
        }
        y[idx] = 1.0f32;
    }

    #[test]
    fn switch_arm_write_does_not_leak_past_merge() {
        let def = build_mode_f32_y!(prover_test_switch_write_leak::expand);
        let buffers = [BufferParam { name: "y", is_output: true }];
        match prove_bounds_freedom(&def, &buffers, &[Assume::LenEqConst { a: "y", value: 1 }]) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("write index") && reason.contains('y'),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (idx tainted post-switch), got {other:?}"),
        }
    }

    /// Manifestation 2 through Switch: a case arm's write must not leak into
    /// the default arm. The case-0 arm clamps `idx = 0`; the default arm reads
    /// `y[idx]` where `idx` is genuinely the pre-switch `ABSOLUTE_POS` (the
    /// case-0 write is invisible to the default). Refuted on genuine grounds.
    #[cube(launch)]
    #[allow(unused_assignments)]
    fn prover_test_switch_arm_leaks_into_default(mode: u32, y: &mut Array<f32>) {
        let mut idx: usize = ABSOLUTE_POS;
        match mode {
            0 => {
                idx = 0usize;
            }
            _ => {
                y[idx] = 2.0f32;
            }
        }
    }

    #[test]
    fn switch_case_write_does_not_leak_into_default() {
        let def = build_mode_f32_y!(prover_test_switch_arm_leaks_into_default::expand);
        let buffers = [BufferParam { name: "y", is_output: true }];
        match prove_bounds_freedom(&def, &buffers, &[Assume::LenEqConst { a: "y", value: 1 }]) {
            ProveResult::Refuted { obligation, counterexample } => {
                assert!(obligation.contains('y'), "unexpected obligation: {obligation}");
                assert!(!counterexample.is_empty());
            }
            other => panic!("expected Refuted (default arm's real, unbounded idx), got {other:?}"),
        }
    }

    /// Manifestation 3 through Switch: a post-switch read must not silently
    /// resolve to whichever arm was walked last. Both the case-0 arm (real,
    /// unbounded `ABSOLUTE_POS`) and the default (`0`) write `idx`; the merge
    /// must taint it, not merge to either arm's value.
    #[cube(launch)]
    #[allow(unused_assignments)]
    fn prover_test_switch_post_merge(mode: u32, y: &mut Array<f32>) {
        let mut idx: usize = ABSOLUTE_POS;
        match mode {
            0 => {
                idx = ABSOLUTE_POS;
            }
            _ => {
                idx = 0usize;
            }
        }
        y[idx] = 5.0f32;
    }

    #[test]
    fn switch_post_merge_taints_arm_written_vars() {
        let def = build_mode_f32_y!(prover_test_switch_post_merge::expand);
        let buffers = [BufferParam { name: "y", is_output: true }];
        match prove_bounds_freedom(&def, &buffers, &[Assume::LenEqConst { a: "y", value: 1 }]) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("write index") && reason.contains('y'),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (idx tainted post-merge), got {other:?}"),
        }
    }

    /// Nested switch (the Switch analog of `nested_branches_restore_correctly`):
    /// a clamp `idx = 0` written *two levels deep* — inside an inner switch,
    /// inside an outer switch arm — must still reach the OUTERMOST merge's taint
    /// set, so the post-switch `y[idx]` sees `idx` tainted (never the leaked
    /// clamp). Confirms the write-log stack composes recursively through
    /// nested switches exactly as it does through nested `if`s (shared
    /// `set_var`/`write_log_stack`). Pre-hypothetical-bug: if the deep clamp
    /// leaked, `idx == 0` unconditionally → false `Proved`.
    #[allow(clippy::single_match)]
    #[cube(launch)]
    fn prover_test_nested_switch_clamp(mode: u32, y: &mut Array<f32>) {
        let mut idx: usize = ABSOLUTE_POS;
        match mode {
            0 => {
                match ABSOLUTE_POS % 2usize {
                    0 => {
                        idx = 0usize;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        y[idx] = 1.0f32;
    }

    #[test]
    fn nested_switch_write_does_not_leak_past_merges() {
        let def = build_mode_f32_y!(prover_test_nested_switch_clamp::expand);
        let buffers = [BufferParam { name: "y", is_output: true }];
        match prove_bounds_freedom(&def, &buffers, &[Assume::LenEqConst { a: "y", value: 1 }]) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("write index") && reason.contains('y'),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!(
                "expected OutOfSubset (deep clamp tainted through both switch merges), got {other:?}"
            ),
        }
    }

    // -- Default-arm reachability: does the default receive the negations? --

    /// Positive: a scrutinee bounded to `[0, 3)` (`ABSOLUTE_POS % 3`) whose
    /// cases 0, 1, 2 cover the whole range makes the default **unreachable** —
    /// the default's path condition (`mode != 0 && mode != 1 && mode != 2`,
    /// combined with the modeled `0 <= mode < 3`) is UNSAT, so a deliberately
    /// out-of-bounds write in the default discharges vacuously and the kernel
    /// PROVES. This pins that the default arm genuinely receives the
    /// conjunction of case negations (without them, or with the case set not
    /// covering the range, the default's OOB write would refute — see the
    /// negative control below).
    #[cube(launch)]
    fn prover_test_switch_default_covered(x: &Array<u32>, y: &mut Array<u32>) {
        let mode = ABSOLUTE_POS % 3usize;
        if ABSOLUTE_POS < y.len() {
            match mode {
                0 => {
                    y[ABSOLUTE_POS] = x[ABSOLUTE_POS];
                }
                1 => {
                    y[ABSOLUTE_POS] = x[ABSOLUTE_POS] + 1u32;
                }
                2 => {
                    y[ABSOLUTE_POS] = x[ABSOLUTE_POS] + 2u32;
                }
                _ => {
                    y[999999999usize] = 7u32;
                }
            }
        }
    }

    #[test]
    fn switch_default_covered_by_cases_proves() {
        let def = build_u32_xy!(prover_test_switch_default_covered::expand);
        match prove_bounds_freedom(&def, MODE_XY_BUFFERS, &[Assume::LenEq { a: "x", b: "y" }]) {
            ProveResult::Proved { .. } => {}
            other => panic!(
                "expected Proved (default unreachable: cases cover mode%3's range), got {other:?}"
            ),
        }
    }

    /// Negative control for `switch_default_covered_by_cases_proves`: the SAME
    /// arms, but the scrutinee is the FREE `mode` scalar (any `u32`), so cases
    /// 0, 1, 2 do NOT cover its range — the default IS reachable (e.g.
    /// `mode == 3`), and its unguarded out-of-bounds write refutes. Confirms
    /// the default's vacuous discharge above was genuinely due to the negations
    /// making it unreachable, not a mis-modeled default.
    #[cube(launch)]
    fn prover_test_switch_default_uncovered(mode: u32, x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS < y.len() {
            match mode {
                0 => {
                    y[ABSOLUTE_POS] = x[ABSOLUTE_POS];
                }
                1 => {
                    y[ABSOLUTE_POS] = x[ABSOLUTE_POS] + 1u32;
                }
                2 => {
                    y[ABSOLUTE_POS] = x[ABSOLUTE_POS] + 2u32;
                }
                _ => {
                    y[999999999usize] = 7u32;
                }
            }
        }
    }

    #[test]
    fn switch_default_not_covered_refutes() {
        let def = build_mode_u32_xy!(prover_test_switch_default_uncovered::expand);
        match prove_bounds_freedom(&def, MODE_XY_BUFFERS, &[Assume::LenEq { a: "x", b: "y" }]) {
            ProveResult::Refuted { obligation, counterexample } => {
                assert!(obligation.contains('y'), "unexpected obligation: {obligation}");
                assert!(!counterexample.is_empty());
            }
            other => panic!("expected Refuted (default reachable, OOB write), got {other:?}"),
        }
    }

    // -- Race + switch: barrier inside a switch arm is barrier divergence --

    /// Positive (the false-`Proved`-preventing one): a `sync_cube()` inside a
    /// switch arm whose scrutinee (`tid`, thread-varying) makes the arm a
    /// non-uniform guard is barrier divergence — `OutOfSubset`, never silently
    /// treated as a top-level phase boundary. The switch arm's condition is
    /// classified by the exact same barrier-uniformity gate as any `if`.
    // Single-case-plus-default `match` on purpose — must be a `Branch::Switch`,
    // with the `sync_cube()` inside a switch arm (the barrier-divergence shape).
    #[allow(clippy::single_match)]
    #[cube(launch)]
    fn prover_test_switch_barrier_divergence(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        if ABSOLUTE_POS < input.len() {
            tile[tid] = input[ABSOLUTE_POS];
        } else {
            tile[tid] = 0.0f32;
        }
        match tid {
            0 => {
                sync_cube();
            }
            _ => {}
        }
        if tid == 0usize && CUBE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    #[test]
    fn switch_arm_barrier_is_barrier_divergence() {
        let def = build_shared!(prover_test_switch_barrier_divergence::expand);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::OutOfSubset { reason } => {
                assert!(
                    reason.contains("barrier divergence")
                        || reason.contains("conditional"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected OutOfSubset (barrier inside switch arm), got {other:?}"),
        }
    }

    /// Negative control for `switch_arm_barrier_is_barrier_divergence`: a
    /// switch with NO barrier inside any arm, sitting before a genuine
    /// top-level `sync_cube()`, still proves race-free — the switch's own
    /// guard push/pop balances correctly, so the subsequent barrier is still
    /// recognized as top-level/uniform. Each thread writes its own `tile[tid]`
    /// in both arms (no write-write race under `t1 != t2`).
    #[cube(launch)]
    fn prover_test_switch_then_barrier(input: &Array<f32>, output: &mut Array<f32>) {
        let tid = UNIT_POS as usize;
        let mut tile = SharedMemory::<f32>::new(256usize);
        match CUBE_POS % 2usize {
            0 => {
                tile[tid] = 1.0f32;
            }
            _ => {
                tile[tid] = 0.0f32;
            }
        }
        let _ = input;
        sync_cube();
        if tid == 0usize && CUBE_POS < output.len() {
            output[CUBE_POS] = tile[0usize];
        }
    }

    #[test]
    fn switch_without_inner_barrier_proves_race_free() {
        let def = build_shared!(prover_test_switch_then_barrier::expand);
        match prove_race_freedom(&def, SHARED_BUFFERS, &[], 256) {
            ProveResult::Proved { .. } => {}
            other => panic!(
                "expected Proved (benign switch, top-level barrier still uniform), got {other:?}"
            ),
        }
    }

    // =================================================================
    // Length-relationship assume (`A.len() + K <= B.len()`) — the
    // "additive anchor" host-side buffer-sizing invariant. An offset read
    // `x[pos + K]` guarded only by `pos < y.len()` is in bounds exactly
    // when `y.len() + K <= x.len()`; the assume supplies that relationship.
    // =================================================================

    /// Offset-window read: `y[i] = x[i] + x[i + 4]` guarded by `i < y.len()`.
    /// The `x[i + 4]` read needs `i + 4 < x.len()`, which follows from
    /// `i < y.len()` and `y.len() + 4 <= x.len()`.
    #[cube(launch)]
    fn prover_test_offset_window(x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS < y.len() {
            y[ABSOLUTE_POS] = x[ABSOLUTE_POS] + x[ABSOLUTE_POS + 4usize];
        }
    }

    /// Positive: with `y.len() + 4 <= x.len()` declared, the offset read
    /// proves (3 obligations: `x[i]`, `x[i + 4]`, `y[i]`).
    #[test]
    fn lenrel_offset_window_proves() {
        let def = build_u32_xy!(prover_test_offset_window::expand);
        let assumes = [Assume::LenPlusConstLe { a: "y", k: 4, b: "x" }];
        match prove_bounds_freedom(&def, MODE_XY_BUFFERS, &assumes) {
            ProveResult::Proved { obligations } => assert_eq!(obligations, 3),
            other => panic!("expected Proved (offset window with len relationship), got {other:?}"),
        }
    }

    /// Negative control: `x.len() == y.len()` alone is NOT enough — it makes
    /// `x[i]` provable (`i < y.len() == x.len()`) but leaves `x[i + 4]`
    /// unbounded (`i + 4` can reach `x.len()` even when `i < x.len()`). The
    /// `+ 4` offset read is refuted, pinning that the length *relationship* is
    /// what's load-bearing, not merely equal lengths.
    #[test]
    fn lenrel_offset_window_without_relationship_refutes() {
        let def = build_u32_xy!(prover_test_offset_window::expand);
        let assumes = [Assume::LenEq { a: "x", b: "y" }];
        match prove_bounds_freedom(&def, MODE_XY_BUFFERS, &assumes) {
            ProveResult::Refuted { obligation, counterexample } => {
                assert!(obligation.contains('x'), "unexpected obligation: {obligation}");
                assert!(!counterexample.is_empty());
            }
            other => panic!("expected Refuted (offset read unbounded), got {other:?}"),
        }
    }

    /// The `K = 0` form (`A.len() <= B.len()`): a plain guarded `x[i]` read
    /// under `i < y.len()` proves when `y.len() <= x.len()` — the length
    /// relationship subsuming the `LenEq` case for this one-sided need.
    #[cube(launch)]
    fn prover_test_len_le_read(x: &Array<u32>, y: &mut Array<u32>) {
        if ABSOLUTE_POS < y.len() {
            y[ABSOLUTE_POS] = x[ABSOLUTE_POS];
        }
    }

    #[test]
    fn lenrel_k_zero_proves() {
        let def = build_u32_xy!(prover_test_len_le_read::expand);
        let assumes = [Assume::LenPlusConstLe { a: "y", k: 0, b: "x" }];
        match prove_bounds_freedom(&def, MODE_XY_BUFFERS, &assumes) {
            ProveResult::Proved { obligations } => assert_eq!(obligations, 2),
            other => panic!("expected Proved (K=0 length relationship), got {other:?}"),
        }
    }

    /// Infeasible-assumption guard: a SELF length-relationship `len_y + 4 <=
    /// len_y` (K > 0) is a contradiction. Without the guard, its unsatisfiable
    /// context vacuously discharges the offset read `x[i + 4]` — a false
    /// `Proved`. The guard rejects the contradictory assume set as
    /// `OutOfSubset` instead (the round-1 "infeasible context vacuously proves"
    /// trap, at the assume layer).
    #[test]
    fn lenrel_self_contradiction_is_rejected_not_vacuously_proved() {
        let def = build_u32_xy!(prover_test_offset_window::expand);
        let assumes = [Assume::LenPlusConstLe { a: "y", k: 4, b: "y" }];
        match prove_bounds_freedom(&def, MODE_XY_BUFFERS, &assumes) {
            ProveResult::OutOfSubset { reason } => {
                assert!(reason.contains("contradictory"), "unexpected reason: {reason}");
            }
            ProveResult::Proved { .. } => {
                panic!("VACUOUS PROOF: a contradictory assume set must never yield Proved")
            }
            other => panic!("expected OutOfSubset (contradictory assumptions), got {other:?}"),
        }
    }

    /// The same guard closes the pre-existing multi-clause form: two
    /// contradictory constant-length assumes (`y.len() == 1` and `y.len() == 2`)
    /// are unsatisfiable together and must not vacuously prove.
    #[test]
    fn contradictory_len_const_pair_is_rejected() {
        let def = build_u32_xy!(prover_test_offset_window::expand);
        let assumes =
            [Assume::LenEqConst { a: "y", value: 1 }, Assume::LenEqConst { a: "y", value: 2 }];
        match prove_bounds_freedom(&def, MODE_XY_BUFFERS, &assumes) {
            ProveResult::OutOfSubset { reason } => assert!(reason.contains("contradictory")),
            ProveResult::Proved { .. } => {
                panic!("VACUOUS PROOF: contradictory length assumes must never yield Proved")
            }
            other => panic!("expected OutOfSubset (contradictory assumptions), got {other:?}"),
        }
    }
}
