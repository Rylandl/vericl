# Equik

> One kernel contract. Equivalent implementations.

*Archived historical document from the project's early, backend-neutral ideation phase (working name
"Equik", since renamed VeriCL). Project names from the private ideation phase have been genericized
for publication; the content is otherwise left as originally written.*

## Status

Equik is an exploratory project. This document defines the problem, intended outcomes, boundaries,
and criteria for a useful first release. It deliberately leaves implementation choices open.

## Overview

Equik is a tool for creating compute kernels whose intended behavior, executable implementations,
and supporting evidence remain connected.

Today, a kernel's specification, accelerated implementation, reference implementation, tests, and
formal reasoning often live in different forms and evolve independently. That separation makes it
easy for them to disagree while each artifact still appears reasonable in isolation. Equik aims to
give developers one place to express the intended computation and the properties that matter, then
produce or check the artifacts needed to support clearly bounded claims about the resulting kernel.

Equik is not specific to a private production codebase, RF simulation, CubeCL, a proof assistant, a
programming language, or a hardware vendor. That private production RF/signal-processing codebase is a
motivating consumer and a source of realistic use cases, not part of Equik's core domain.

## Vision

A developer should be able to define a kernel and its important behavioral constraints once, ask
Equik to realize that kernel for a chosen execution environment, and receive an inspectable body of
evidence explaining what was checked, under which assumptions, and against which exact kernel
definition.

Equik succeeds when generated code is not merely accompanied by a claim of correctness, but by
evidence whose scope and limitations are understandable to both users and automation.

## Problem statement

Accelerated compute kernels are difficult to trust for reasons that extend beyond the arithmetic in
their bodies:

- indexing and layout conventions can differ between implementations;
- boundary behavior and overflow rules can be implicit;
- parallel execution can introduce collisions or ordering differences;
- optimizations can change numerical behavior;
- reference implementations can drift away from accelerated implementations;
- tests demonstrate selected cases but do not explain the full scope of a guarantee;
- formal results can prove a model without establishing that deployed code implements that model;
- the compiler, runtime, driver, and hardware introduce additional trust boundaries.

Equik should make these relationships visible and help prevent silent disagreement between intent,
implementation, and evidence.

## Intended users

Equik is intended for people who build or depend on performance-sensitive compute kernels and need
more confidence than ordinary example-based tests provide. Potential users include:

- library authors maintaining portable CPU and accelerator implementations;
- scientific and engineering software teams;
- simulation and signal-processing developers;
- compiler and formal-methods researchers;
- teams qualifying numerical software for high-consequence workflows;
- developers who want generated kernels without treating generation as an opaque step.

Users should not need to be formal-methods experts to understand what Equik has and has not
established.

## Core user experience

At a conceptual level, a user should be able to:

1. Describe a kernel's intended inputs, outputs, computation, and relevant assumptions.
2. State the behavioral properties or equivalence claims that matter for that kernel.
3. Select one or more execution targets supported by their Equik installation.
4. Produce executable artifacts and the evidence available for those targets.
5. Run conformance checks in development and automation.
6. Inspect a report that identifies the kernel, assumptions, artifacts, checks, results, and trust
   boundaries.
7. Detect when any relevant artifact or assumption has changed without its evidence being renewed.

This workflow is a product requirement, not a decision about authoring syntax, internal
representation, commands, project layout, or implementation language.

## Project principles

### One semantic point of custody

The intended behavior of a kernel should have one authoritative point of custody. Other artifacts
should be derived from it, checked against it, or explicitly identified as external assumptions.
The implementation may determine how this authority is represented.

### Claims must be precise

Equik must say exactly what a result establishes. A proof about an abstract computation, a check of
generated source, a differential test on a device, and verification of an entire execution stack
are different claims and must not be presented as interchangeable.

### Assumptions are part of the result

Input constraints, numeric behavior, environmental requirements, unsupported cases, and trusted
components must remain attached to the evidence that depends on them.

### Evidence should survive scrutiny

Results should be inspectable, reproducible where practical, and suitable for automated rejection
when stale or incomplete. A user should be able to determine which kernel and artifacts a result
belongs to without relying on filenames or convention.

### Useful assurance is incremental

Equik should provide value before the strongest possible verification is available. It should be
possible to strengthen the evidence for a kernel over time without changing the meaning of weaker
claims or presenting them as stronger than they are.

### General core, demand-driven integrations

The core concepts must not depend on one application domain or execution technology. Integrations
should be added in response to concrete users and examples rather than an attempt to support every
language, backend, or proof technique from the outset.

### Generated artifacts remain understandable

Users should be able to inspect, diagnose, and, when appropriate, retain the artifacts Equik
produces. Failures should identify the unmet requirement or broken relationship rather than only
reporting that verification failed.

## Required capabilities

An implementation of Equik should provide the following capabilities, without this document
prescribing how they are built.

### Kernel definition

- Express the intended computation and its externally meaningful interface.
- Capture assumptions and behavioral constraints relevant to correctness.
- Make otherwise ambiguous behavior explicit when it affects a claim.
- Assign a stable identity to the exact definition used to produce evidence.

### Executable realization

- Produce or validate at least one executable realization of a kernel.
- Preserve the connection between that realization and its authoritative definition.
- Identify target-specific assumptions and unsupported behavior.
- Permit additional execution targets without changing the domain meaning of the kernel.

### Independent comparison

- Provide a way to evaluate behavior independently of the accelerated realization being checked.
- Compare realizations using criteria appropriate to the declared semantics.
- Report counterexamples or useful diagnostic context when comparison fails.

The implementation should decide whether independence is achieved through interpretation,
generation, separately authored artifacts, or another mechanism.

### Machine-checked evidence

- Support machine-checkable evidence for at least one meaningful property.
- Associate every result with its kernel definition, assumptions, and relevant artifacts.
- Distinguish proved statements from tested observations and unchecked assertions.
- Reject evidence that no longer corresponds to the inputs from which it was produced.

### Conformance and automation

- Make checks repeatable in local development and continuous integration.
- Produce both human-readable and machine-readable outcomes.
- Fail clearly when required evidence cannot be established or reproduced.
- Allow projects to declare which claims are required for acceptance.

### Trust accounting

- Identify components and transformations that are trusted rather than checked.
- Describe where each guarantee begins and ends.
- Avoid implying that source-level evidence verifies compilers, runtimes, drivers, or hardware when
  those components are outside the checked boundary.

## Candidate use cases

Initial examples should be selected for their ability to expose important correctness questions,
not because they permanently define Equik's supported domain. Useful candidates may include:

- coordinate and layout transformations;
- deterministic counters or random-number primitives;
- bounded element-wise transformations;
- reductions with observable ordering behavior;
- small signal-processing operations;
- kernels with boundary-sensitive reads or writes;
- kernels with exact integer behavior alongside kernels with numerical tolerances.

At least one example should come from outside the private codebase before the project claims to be generally
useful.

## Relationship to the private codebase

The private codebase can act as an early adopter and provide kernels with real demands around determinism,
indexing, replay, portability, and numerical comparison. An Equik integration may eventually be
used by that codebase, but:

- Equik must not contain RF-specific concepts;
- Equik must not require the private codebase to function;
- Equik's public vocabulary must make sense to unrelated compute projects;
- domain-specific policy should remain in the private codebase or an integration layer;
- success must be demonstrated with at least one unrelated use case.

## Non-goals

Unless the scope is deliberately revised, Equik is not intended to:

- verify arbitrary application programs;
- replace existing compilers, runtimes, or accelerator frameworks;
- claim verification of hardware or an entire execution stack by default;
- prove that a kernel is fast or that a chosen algorithm is appropriate;
- guarantee identical floating-point results across environments without explicit support and
  evidence for that claim;
- automatically recover complete intent from arbitrary existing source code;
- support every execution target, programming language, or verification method;
- hide assumptions in order to present a simpler correctness badge;
- become coupled to the private codebase's release process or application model.

## First-release outcomes

The first useful release should demonstrate one complete, understandable path from kernel intent to
an executable artifact and its supporting evidence. It is successful when:

- a new user can understand the purpose and boundaries of the project from its documentation;
- a small kernel can be defined without embedding application-specific concepts in Equik;
- at least one executable realization can be produced or validated;
- behavior can be checked against an independent comparison mechanism;
- at least one non-trivial behavioral property is machine-checked;
- stale or mismatched evidence is detected;
- both successful and intentionally defective examples produce useful reports;
- the report distinguishes assumptions, proofs, tests, and trusted components;
- the workflow can be enforced automatically;
- one example motivated by the private codebase and one unrelated example use the same core concepts.

The first release does not need broad language, backend, numeric, or proof coverage. A narrow path
with honest claims is sufficient.

## Evaluation criteria

Candidate implementations should be evaluated against the following questions:

- Does the design keep kernel meaning independent of any one backend?
- Can users understand the exact claim without reading Equik's implementation?
- Can evidence be matched unambiguously to the definition and executable artifact it covers?
- Does a deliberately introduced semantic defect cause an appropriate check to fail?
- Are unsupported cases rejected or disclosed rather than silently approximated?
- Can another integration be added without rewriting the core concepts?
- Is the trusted boundary smaller and clearer than it would be in an ad hoc pipeline?
- Is the workflow practical enough that a project would keep it enabled in automation?

## Open decisions for the implementing agent

The implementing agent should investigate and choose the smallest coherent design that satisfies
the outcomes above. This document intentionally does not decide:

- the authoring experience or whether Equik introduces a dedicated notation;
- the internal representation of a kernel;
- the initial implementation language or repository structure;
- the first execution target or integration mechanism;
- the first independent comparison strategy;
- the verification tools, solvers, or proof assistants, if any;
- the supported kernel subset and numerical models;
- how transformations are validated or certified;
- whether generated artifacts are checked in or created on demand;
- the terminology or hierarchy used to communicate assurance levels;
- the command-line, library, editor, or build-system interfaces;
- the report and evidence formats;
- caching, reproducibility, and artifact-distribution mechanisms;
- extension, plugin, and version-compatibility models;
- licensing, governance, release, and compatibility policies.

Decisions should be justified by the first-release outcomes rather than by hypothetical future
breadth. Material choices should be recorded with their alternatives, consequences, and the claim
boundary they create.

## Questions to answer during implementation

Before committing to a design, the implementing agent should answer:

1. What is the smallest kernel class that can demonstrate the complete Equik value proposition?
2. Which correctness failure will the prototype detect that ordinary tests could plausibly miss?
3. What is authoritative, and how is every other artifact tied back to it?
4. Which parts of the path are checked, tested, or trusted?
5. How will users see assumptions and unsupported behavior?
6. How will the prototype demonstrate that its evidence becomes invalid when an artifact drifts?
7. What would be required to add a second execution target without changing kernel meaning?
8. What example outside the private codebase will challenge the same core abstractions?
9. Which apparent guarantees must the project explicitly decline to make?
10. What evidence would justify proceeding beyond the exploratory release?

## Naming

**Equik** is a project name, not a required acronym. It evokes equivalence and compute kernels
without binding the project to a particular backend or verification technique.

The working tagline is:

> One kernel contract. Equivalent implementations.

Both name and tagline may be revisited before public release.
