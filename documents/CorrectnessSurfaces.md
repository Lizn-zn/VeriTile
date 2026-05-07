# Correctness Surfaces

This document explains which public theorem surface to use when proving
properties of `ComputeKernel`s. Surfaces live in
[`VeriTile.Triton.Float.Correctness`](../VeriTile/Triton/Float/Correctness.lean).

## Quick-Pick Table

| Goal | Use |
|---|---|
| One kernel matches a 1D `Fin n → ℝ` math spec | `ComputeCorrect.OutputArray` |
| One kernel matches an ND tensor spec | `ComputeCorrect.OutputTile` |
| One kernel matches a Real scalar reduction | `ComputeCorrect.OutputScalar` |
| One kernel matches a `Nat` scalar reduction | `ComputeCorrect.OutputNatScalar` |
| Value/index pair on every lane | `ComputeCorrect.OutputPair` |
| Value/index pair on active lanes only | `ComputeCorrect.OutputPairWhere` |
| Two kernels produce equal 1D outputs | `ComputeRefine.OutputArrayEq` |
| Two kernels produce equal ND tensor outputs | `ComputeRefine.OutputTileEq` |
| Two kernels produce equal Real scalar outputs | `ComputeRefine.OutputScalarEq` |
| Two kernels produce equal `Nat` scalar outputs | `ComputeRefine.OutputNatScalarEq` |
| Two kernels produce equal value/index pairs | `ComputeRefine.OutputPairEq` / `OutputPairEqWhere` |
| Custom postcondition over the final state | `ComputeCorrect.Post` / `ComputeRefine.Post` |
| Relation over arbitrary initial states (rare) | `ComputeCorrect.General` / `ComputeRefine.General` |

The lower-level `ComputeKernel.ComputeCorrect` and
`ComputeKernel.ComputeRefine` definitions remain the implementation layer.
New example theorem statements should normally not expose those names directly.

## Single-Kernel Correctness

Single-kernel correctness checks one kernel against a mathematical or
algorithmic specification.

### Scalar outputs

```lean
ComputeCorrect.OutputScalar k s out offset expected
-- post: s'.readMem out offset = expected

ComputeCorrect.OutputNatScalar k s out offset expected
-- post: s'.readMemValue .nat out offset = expected
```

Use these for scalar reductions such as a final `max`, `sum`, or
`argmax`-style integer index.

### Tensor outputs

```lean
ComputeCorrect.OutputTile k s view expected
-- post: ∀ idx, TensorView.observe (some s') view idx = some (expected idx)

ComputeCorrect.OutputArray k s view expected   -- 1D specialization
-- post: same, with `expected : Fin n → ℝ`
```

`OutputArray` is the 1D `n`-shape specialization of `OutputTile` and is the
preferred surface when the spec is naturally a `Fin n → ℝ` function.

### Value/index pair outputs

```lean
ComputeCorrect.OutputPair k s valueRegion indexRegion offset
  expectedValue expectedIndex
-- post: every lane satisfies the value spec and the typed Nat index spec

ComputeCorrect.OutputPairWhere k s valueRegion indexRegion offset
  active expectedValue expectedIndex
-- post: same, but only on lanes where `active i` holds
```

Use these for kernels with paired outputs such as
`tl.max(..., return_indices=True)`. `OutputPairWhere` is the right choice
when masking restricts which lanes participate.

## Kernel Refinement

Kernel refinement runs two kernels from the same initial state and
relates their observations. For ordinary optimization proofs prefer an
equality helper.

### Scalar equality

```lean
ComputeRefine.OutputScalarEq lhs rhs s lhsOut lhsOffset rhsOut rhsOffset
-- post: lhs'.readMem lhsOut lhsOffset = rhs'.readMem rhsOut rhsOffset

ComputeRefine.OutputNatScalarEq lhs rhs s lhsOut lhsOffset rhsOut rhsOffset
-- post: lhs'.readMemValue .nat ... = rhs'.readMemValue .nat ...
```

### Tensor equality

```lean
ComputeRefine.OutputTileEq lhs rhs s lhsView rhsView
-- post: ∀ idx,
--   TensorView.observe (some lhs') lhsView idx
--     = TensorView.observe (some rhs') rhsView idx

ComputeRefine.OutputArrayEq lhs rhs s lhsView rhsView   -- 1D specialization
```

### Value/index pair equality

```lean
ComputeRefine.OutputPairEq lhs rhs s
  lhsValueRegion lhsIndexRegion rhsValueRegion rhsIndexRegion offset

ComputeRefine.OutputPairEqWhere lhs rhs s ... offset active
```

Use the `Where` variant when the equality only needs to hold on active lanes.

## Escape Hatches

`Post` fixes the initial state but accepts any postcondition over the
final state:

```lean
ComputeCorrect.Post k s (fun s' => ...)
ComputeRefine.Post lhs rhs s (fun lhs' rhs' => ...)
```

`General` quantifies over the initial state itself, taking a relation
over `(s0, s')`:

```lean
ComputeCorrect.General k (fun s0 s' => ...)
ComputeRefine.General lhs rhs (fun s0 lhs' rhs' => ...)
```

`General` should be reserved for theorems that genuinely need to be
parametric in the initial state (rare). Most proofs fix `s` and use
`Post`, or pick an `Output*` helper.

## Gap Policy

The `ComputeCorrect.*` / `ComputeRefine.*` helpers use the default
`GapPolicy.ignore`, matching current Real-first algorithm proofs. If a
theorem needs to record an externally checked compute-to-algorithm gap,
state the underlying `ComputeKernel.ComputeCorrect` /
`ComputeKernel.ComputeRefine` directly with:

```lean
gap := .require contract
```

or add a dedicated helper for that theorem family. See
[`EraseDType.md`](./EraseDType.md) for the compute-to-algorithm projection
policy.

## Naming

Recommended theorem names:

- `<kernel>_compute_correct` for ordinary single-kernel correctness.
- `<kernel>_correct_view` for public manifest-compatible view surfaces.
- `<rewrite>_refinement_view` for public two-kernel refinement surfaces.

Execution-only helper lemmas may use `_exec_view` and direct `exec`
equalities. The public theorem should wrap those helpers in
`ComputeCorrect.*` or `ComputeRefine.*`. Naming details:
[`TheoremSurfaces.md`](./TheoremSurfaces.md).
