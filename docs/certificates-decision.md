# Assurance ladder Rung A — decision record (2026-07-24)

Rung A of the agreed assurance ladder (tasks/todo.md "Agreed sequencing") has two
halves that both attack the single largest opaque component in the trusted base —
the SMT solver binary (subprocess z3, docs/ir-research.md §4):

1. **Counterexample validation** (the cheap dual): independently re-check every
   `sat`-derived refutation in plain Rust before reporting it.
2. **Proof certificates for `unsat`**: have the solver emit an independently
   checkable proof for every discharged obligation, so a `Proved` claim's trust
   moves from "the solver said unsat" to "a small, auditable checker verified the
   solver's proof".

**Outcome: (1) is shipped and unconditional; (2) is blocked on tooling
availability at current versions and is deferred, documented here rather than
shipped half-checked.** This is the sanctioned outcome from the task framing: an
honest "certificates blocked on X" with the dual shipped, never a fake or
untested certification layer.

## 1. Counterexample validation — SHIPPED

Every `Refuted`-producing `check-sat` (`check_obligation` for out-of-bounds
obligations, `check_race` for two-thread data-race obligations) now re-checks the
solver's model before the verdict is reported (`crates/vericl-ir/src/prover.rs`,
`Prover::validate_counterexample`):

- The live assertion set is mirrored vericl-side in `Prover::asserts`, kept
  exactly parallel to z3's assertion stack by routing every `push`/`pop`/`assert`
  through the `s_push`/`s_pop`/`s_assert` wrappers. Flattened at the `sat` point it
  *is* the set of formulas the model must satisfy: the negated obligation, the
  live path conditions, the contract assumes, and the leaf type-range facts.
- The model values are read back with `get_value`, then each live assertion is
  evaluated in plain Rust by a small **total interpreter** over the exact SMT-LIB
  subset this file emits (`eval_sexpr`): integer `+`/`-`/`*`/`div`/`mod` (Euclidean,
  matching SMT-LIB, via `i128::{checked_div_euclid, checked_rem_euclid}`), `ite`,
  the comparisons `<`/`<=`/`>`/`>=`/`=`, boolean `and`/`or`/`not`, the literals
  `true`/`false` and non-negative numerals, and the declared free constants (bound
  by the model). The vocabulary is closed — it is precisely the set of
  `self.smt.<op>` builders used to construct assertions in the file — so the
  interpreter is total over real assertions; anything outside it is an `Err`.
- If any live assertion fails to hold under the model — a solver bug or a vericl
  encoding/parse error — the verdict fails **closed** to `SolverError`, never a
  silent (and possibly spurious) `Refuted` (`ProveResult::Refuted`'s documented
  invariant).

What this buys: for a **refutation**, the solver's `sat` verdict leaves the
trusted base. What remains trusted for a `Refuted` is the ~120-line Rust
interpreter (auditable, and unit-tested directly against a synthetic
invalid-model negative), plus vericl's own encoding — not the solver.

Cost (measured, this machine, z3 4.16.0): a `prove_bounds_freedom` call is
~7–8 ms, dominated by the z3 subprocess spawn. Validation adds **zero** solver
work on the common `Proved` path (it runs only on `sat`); on a `Refuted` it adds
one `get_value` round-trip plus microsecond-scale Rust evaluation, immeasurable
against the spawn baseline.

## 2. Proof certificates — BLOCKED at current versions

### Design prior (what would be built)

cvc5 emits [Alethe](https://verit.gitlabpages.uni.lu/alethe/) proofs for `unsat`
in QF_LIA, checkable by [Carcara](https://github.com/ufmg-smite/carcara), a Rust
proof checker. The certified path would be an **optional** layer behind a
suite-level `certify: true` flag (it requires cvc5 on `PATH`, unlike the default
z3-only lane): every discharged obligation re-solved in a fresh **non-incremental**
context with proof production on (a second pass purely for certification, since
incremental push/pop and proof production interact badly in several solvers), the
Alethe certificate checked by Carcara, and — on any check failure — the verdict
failing closed to `SolverError`. When on, each `Proved` claim's config would
record `certified: true` + the checker version, and the trusted-list wording for
that claim would shift from "the solver binary" to "the certificate checker
(small, auditable Rust)" + honest notes on what still remains trusted (vericl's
own IR encoding, and Carcara itself).

### Why it is blocked here (verified 2026-07-24)

- **cvc5 is not available.** Not on `PATH`; not a Homebrew formula (`brew info
  cvc5` → "No available formula"; `brew search cvc5` → only `cc65`). Installing it
  means fetching a prebuilt binary from GitHub releases or a from-source C++ build
  — an unrequested, heavyweight system change, and it would still only put cvc5 on
  *this* machine, whereas the certified layer requires cvc5 on every user's `PATH`.
- **Carcara is not a crates.io dependency.** The crates.io index has no `carcara`
  crate (`https://index.crates.io/ca/rc/carcara` → `NoSuchKey`). It exists only as
  the `ufmg-smite/carcara` git repository and is not published/versioned as a
  stable library API. Vendoring it would copy a large, unaudited third-party
  codebase into an otherwise small pinned-dependency tree — and, critically, with
  neither cvc5 nor Carcara present, the mandatory tests (a certificate-checking
  round-trip on real obligations; a corrupted-certificate negative) **cannot be
  written or exercised**. Shipping the plumbing anyway would be exactly the
  half-checked lane the task forbids.
- **z3's own proof format is not a substitute.** z3 can emit proofs, but its
  format is ad-hoc and lacks an independent, maintained external checker of the
  QF_LIA fragment — so it fails the independent-checker requirement that is the
  entire point of Rung A. Using z3-checks-z3 would narrow nothing.

The incremental-vs-proof interaction (cvc5 producing proofs under push/pop) was
*not* reached as a blocker because availability blocks earlier; the design already
routes around it with the fresh-context second pass, which stays acceptable only
if per-obligation solve times remain ms-scale (to be measured when unblocked).

### Path forward (when a maintainer opts in)

1. Install cvc5 (pin a release that emits Alethe for QF_LIA) and confirm it is on
   `PATH`; gate the whole lane on `vericl_ir::cvc5_version().is_some()`, mirroring
   the existing `z3_version()` trusted-component capture.
2. Add Carcara — preferably once it is published to crates.io with a stable
   library API; otherwise a reviewed git-pin vendored deliberately, not silently.
3. Implement the second-pass certifier: for each `unsat` obligation, re-solve in a
   fresh non-incremental cvc5 context with proof production, feed the Alethe proof
   to Carcara, fail closed on any check failure. Measure the added solve time.
4. Wire the `certify: true` suite flag, the `certified: true` + checker-version
   claim-config fields, and the trusted-list wording shift. Write the four required
   tests (round-trip pass; corrupted-certificate negative; and — already done here
   — model-validation positive + synthetic invalid-model negative).

Until then, the solver binary remains trusted for `Proved` claims (recorded
honestly as such in evidence and the README), while `Refuted` claims are now
independently validated per §1.
