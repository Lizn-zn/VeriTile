---
title: Overview
description: What VeriTile proves, how the pieces fit together, and where to start.
---

VeriTile embeds a typed Triton-style kernel language in Lean 4. It connects
kernel implementations to mathematical specifications, with proofs that
Lean can check and readers can inspect in the repository.

## Two questions a proof can answer

- **Correctness:** does this kernel implement its mathematical contract?
  The [vector addition example](https://github.com/Lizn-zn/VeriTile/blob/main/bench/examples/VectorAdd.lean)
  proves a pointwise sum, termination, and preservation of memory outside
  the output window under its stated preconditions.
- **Equivalence:** do two implementations agree on observable outputs?
  [Stable softmax](https://github.com/Lizn-zn/VeriTile/blob/main/bench/examples/SoftmaxStableEquiv.lean)
  and [fused SwiGLU](https://github.com/Lizn-zn/VeriTile/blob/main/bench/examples/FusedSwigluEquiv.lean)
  demonstrate this under their explicit rounding models and memory contracts.

The showcase contracts use `KernelIO` notation: `io ⊨ spec` for an
implementation contract, and `io₁ ≡[R] io₂` for equivalence under a rounding
model. Lower-level `ComputeCorrect` and `ComputeRefine` surfaces remain
available. The source of each theorem records its full assumptions.

## From code to proof

Follow the [Python-to-contract walkthrough](/VeriTile/cookbook/vector-add-walkthrough/)
for an executable vector-add example, its assumptions, and a rejected mutation.

1. Describe a supported kernel using the typed `triton { ... }` DSL.
2. State its input/output contract or its relationship to another kernel.
3. Prove the contract against VeriTile's operational semantics in Lean 4.
4. Check the proof and its dependencies with the repository's build and trust audits.

The DSL supports tile operations, loads and stores, masks, reductions, and
control flow within the documented [Triton subset](/VeriTile/architecture/triton-subset/).
The [translation cookbook](/VeriTile/cookbook/) explains how to work with it.

## Proof automation

The [lemma library](https://github.com/Lizn-zn/VeriTile/tree/main/VeriTile/Triton/KernelLemmas)
provides reusable facts for loops, mathematical operators, and memory.
The [proving wrapper](https://github.com/Lizn-zn/VeriTile/blob/main/scripts/prove.sh)
runs an agent proof loop with a cycle limit and saved logs. The official
comparator judges the theorems selected with `--theorem` against a snapshot of
the original task and dependencies, checks permitted axioms, and replays the
proofs in Lean's kernel. See the
[setup and usage guide](https://github.com/Lizn-zn/VeriTile/blob/main/scripts/README.md).
Artifact and bench audits also require official comparator proof replay, using
a frozen snapshot of the current repository sources. The trust gates retain
their additional checks of theorem statements and circular specifications.

## What the proof covers

Algorithm-level proofs use mathematical semantics. Abstract rounding models
support additional contracts, but do not establish full IEEE-754 behavior.
PTX code generation, TMA, detailed hardware concurrency, and Python wrapper
execution have separate boundaries. Read the
[semantic scope](/VeriTile/architecture/triton-subset/) and each theorem's
assumptions before interpreting a result.

## Start exploring

- [Read a complete vector-add proof](https://github.com/Lizn-zn/VeriTile/blob/main/bench/examples/VectorAdd.lean).
- [Inspect corpus counts and audit evidence](/VeriTile/status/).
- [Follow the repository quick start](https://github.com/Lizn-zn/VeriTile#quick-start).
- [Browse the source and build instructions](https://github.com/Lizn-zn/VeriTile).
