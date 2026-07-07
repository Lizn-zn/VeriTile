---
title: Conventions
description: TileShape orientation, ND-general operators, naming patterns, and where to push back on shortcuts.
---

The conventions below are not stylistic preferences — each has bitten us
hard enough to be a project rule.

## Outermost-first shapes

`TileShape := List Nat` is **outermost-first**:
`shape (M, N) = [M, N]`. The list index *is* the user-visible axis index,
matching Triton, NumPy, PyTorch.

When implementing a new axis-aware operator, the natural Lean move is to
align the head of the list with the "active" axis so structural recursion
on the head is easy. For broadcasting and head-axis reduction,
innermost-first internal storage feels cleaner.

Don't do it.

**Why:** users paste real `.py` kernels and write `axis=K` referring to
the user-visible outer-first shape. Internal flipping is a permanent
ergonomic tax — every DSL macro and every doc example would need
translation in the user's head. Triton semantic fidelity wins.

**How to apply:** when a new ND operator (reduce, transpose, expand_dims,
ND-broadcast extension, …) feels easier with innermost-first, eat the
Lean-side pain on the outermost-first side:

- `init ++ [last]` recursion patterns.
- Structural induction on `init`.
- Helper `_pos` (positivity-asserting) variants to avoid `Option`.

The reduce-axis implementation in `VeriTile/Semantics.lean` —
`Tile.reduceSumDropLast`, `reduceMaxDropLast_pos` — is the canonical
template.

## Always go fully ND

Before writing `match shape with | [_, _] => ...`, or `(rest ++ [axisDim])`,
or `Input2DAt`: ask whether a future kernel could want a different rank or
axis here. The answer is almost always yes.

The discipline:

1. **Design with `Fin shape.length` axis indices.** New axis-aware
   typed-AST nodes take an explicit axis parameter, not a hard-coded
   position.
2. **Parameterize ND infrastructure by offset functions.** Predicates,
   helpers, and theorems take `TileIndex shape → Nat` rather than baking
   in 1D / 2D patterns.
3. **Provide named offset families alongside.** `Offset.linear1D`,
   `Offset.rowMajor2D`, `Offset.contig` are ergonomic helpers; the core
   API stays generic.

The reduce-axis API in `VeriTile/Core.lean` — `axisDim`,
`eraseAxis`, `setAxisOne`, `reduceShape`, `insertAxisIndex`,
`replaceAxisIndex` — is the canonical template. Mirror that pattern for
any new axis-aware op.

**What this costs:** real dependent-type work, but bounded. You pay it
once. The alternative (rank-/axis-specific helpers) costs a future rewrite
each time the bench picks up a new shape pattern.

## Rewriting cost is a tiebreaker, not a thesis

When evaluating design tradeoffs:

- Treat "but this would mean rewriting N existing proofs" as a tactical
  concern (when to ship, how to slice the work), **not** a strategic
  vote.
- If two designs are otherwise close, the one that doesn't require
  rewriting may win by a hair. If one is meaningfully cleaner, more
  correct, or more Triton-faithful, recommend it even when it implies
  a multi-hundred-line proof rewrite.
- Don't recommend a design whose **primary** justification is "preserves
  existing proofs". That's a tiebreaker, not a thesis.
- Tactical ship-timing overrides are legitimate ("let's do D-b now and
  revisit D-a after the release"). But the recommendation should not
  arrive at D-b by default.

## Naming patterns

A handful of patterns recur across the bench corpus; staying consistent
makes proofs cheaper.

- **Per-kernel files** live at `bench/tritonbench_g/<kernel>/<Kernel>.lean`
  for ports, `VeriTile/Examples/<Kernel>.lean` for examples.
- **Theorem names** end in `_correct` (kernel ↔ spec) or `_refine`
  (kernel ↔ kernel). Compute-layer projections get a parallel
  `_compute_correct` / `_compute_refine`.
- **Spec definitions** typically named `<kernel>_spec` or `<feature>Spec`.
- **Offset / index helpers** name the indexing shape:
  `inOffsetDef`, `outOffsetDef`, `outOffsetFn`, etc.
- **`_pos` suffix** marks the positivity-asserting variant of a helper —
  returns the underlying value directly instead of `Option`.

## Antiquotes are paste-in-hostile — minimize them

`$(REGION)` in `tl.load($(xReg) + offs)` and `$ℝ(t)` for ℝ literals are
both VeriTile inventions that Triton's `.py` doesn't have. They're a
friction point for the paste-in goal.

When adding a new DSL construct: design without antiquotes if at all
possible. If unavoidable, document the construct's antiquote requirement
explicitly in `documents/TritonSubset.md` and prefer context-aware
resolution over user-visible bridging.

Known paste-in conflicts that are already tracked:

- `tl.toReal(_)` — a VeriTile invention; Triton uses implicit type
  promotion.
- Strict `ℝ` vs `Nat` channel separation — rejects mixed arithmetic that
  Triton accepts.
- "Error on unknown kwargs" — rejects Triton kernels with
  `cache_modifier=` / `eviction_policy=`.

These are recorded tradeoffs, not surprises. New constructs should not
add to the list.
