---
title: Bench translation cookbook
description: How to translate a Triton .py kernel into Lean and prove it correct — meta-rules, review criteria, conventions, proof templates, and known tactical traps.
---

The cookbook collects the working knowledge for translating a Triton `.py`
kernel into a `triton { ... }` block, writing its spec, and closing its
`ComputeCorrect.Realizes` proof. It distills lessons learned while building
out the bench corpus rather than the theoretical reference (for that see
[Architecture & semantics](/VeriTile/architecture/) and [Proofs & surfaces](/VeriTile/proofs/)).

## In this section

1. [Translating a kernel](/VeriTile/cookbook/translating/) — what's a faithful 1:1
   transcription, what's a tolerated gap, what counts as a deviation that
   must be fixed before merge.
2. [Conventions](/VeriTile/cookbook/conventions/) — outermost-first shapes, fully ND
   ops, naming patterns, and the rewriting cost stance.
3. [Proof templates](/VeriTile/cookbook/proof-templates/) — standard 1D scatter,
   dual-channel store, single-step loops, and which helper to reach for.
4. [`forLoop_inv` pitfalls](/VeriTile/cookbook/forloop-pitfalls/) — seven concrete
   tactical traps when proving kernels that wrap a loop.

## The four meta-rules

Before touching code, the project commits to four rules that override
tactical convenience:

### 1. Triton semantic fidelity

VeriTile is a verification project *for Triton*. When a design choice trades
cleanliness for fidelity to real Triton semantics, **fidelity wins**.
Concrete example: `shape ()` and `shape (1)` are distinct in Triton, so they
stay distinct here even though packing them together would simplify the type
machinery. Algorithm-channel divergences (e.g. modelling `ℝ` instead of
IEEE-754) are allowed but must be documented as known gaps, not silently
adopted as simplifications.

### 2. Triton users are first-class

A developer with a working `triton.jit`-decorated `.py` kernel should be able
to paste it into a `triton { ... }` block with minimal modification. The
parsed surface should mirror Triton's `tl.*` API as closely as Lean syntax
permits — `tl.load(x_ptr + offsets, mask=m, other=0.0)`, `tl.program_id(axis=0)`,
Python-style comparisons in masks, kwargs, etc. New DSL constructs should
not add antiquote requirements (`$(...)`, `$ℝ(...)`) unless unavoidable.

### 3. Outermost-first, fully ND

`TileShape := List Nat` is **outermost-first** (`shape (M, N) = [M, N]`,
list index = user-visible axis index), matching Triton / NumPy / PyTorch.
Do not flip to innermost-first even when it simplifies head-cons recursion
— take the extra Lean-side pain instead (`init ++ [last]` patterns, `_pos`
helpers to avoid `Option`).

New axis-aware operators (reduce, transpose, expand_dims, broadcast
extension, …) go in **fully ND from the start**, parameterized over
`Fin shape.length` axes. Don't introduce 2D-only or last-axis-only
variants "just for this kernel" — they become a maintenance graveyard the
moment another shape pattern arrives.

### 4. Rewriting cost is a tiebreaker, not a thesis

"We'd have to rewrite N existing proofs" is not, by itself, sufficient
justification for a worse design. Pick the design that's cleaner / more
Triton-faithful / more general; pay the rewrite cost. The project is
intended to iterate over years, so a permanently worse design saves a few
weeks at the cost of years of friction. Tactical objections about ship
timing are legitimate and may delay a refactor, but they should not change
the recommendation.

---

If you only read one page next, read
[Translating a kernel](/VeriTile/cookbook/translating/) — it's the most-applied page
when adding a new bench entry.
