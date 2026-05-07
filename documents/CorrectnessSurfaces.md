# Correctness Surfaces

This document explains which public theorem surface to use when proving
properties of `ComputeKernel`s.

The short rule:

- Use `ComputeCorrect.Output*` for one kernel against a mathematical spec.
- Use `ComputeRefine.Output*Eq` for two kernels with equal observable outputs.
- Use `ComputeCorrect.Post` / `ComputeRefine.Post` when the observation is too
  custom for an output helper.
- Use `ComputeCorrect.General` / `ComputeRefine.General` only when the theorem
  genuinely needs a relation over arbitrary initial states.

The lower-level `ComputeKernel.ComputeCorrect` and
`ComputeKernel.ComputeRefine` definitions remain the implementation layer. New
example theorem statements should normally not expose those names directly.

## Single-Kernel Correctness

Single-kernel correctness means one kernel is checked against a mathematical or
algorithmic specification:

```lean
ComputeCorrect.OutputScalar k s out offset expected
```

The final state produced by `k` from initial state `s` must satisfy:

```lean
s'.readMem out offset = expected
```

Use this for scalar reductions such as a final `max` or `sum`.

For tensor outputs, use:

```lean
ComputeCorrect.OutputTile k s view expected
```

The final state must satisfy:

```lean
∀ idx, TensorView.observe (some s') view idx = some (expected idx)
```

For 1D tensors, `OutputArray` is the same idea with a `Fin n → ℝ` spec.

For value/index outputs such as `tl.max(..., return_indices=True)`, use:

```lean
ComputeCorrect.OutputPairWhere
  k s valueRegion indexRegion offset active expectedValue expectedIndex
```

The final state must satisfy the value and typed Nat index specs on every
active lane.

## Kernel Refinement

Kernel refinement means two kernels are run from the same initial state and
their observations are related. Most ordinary optimization proofs should use an
equality helper:

```lean
ComputeRefine.OutputTileEq lhs rhs s lhsView rhsView
```

The final states must satisfy:

```lean
∀ idx,
  TensorView.observe (some lhs') lhsView idx =
    TensorView.observe (some rhs') rhsView idx
```

For scalar observations use `OutputScalarEq`; for 1D tensors use
`OutputArrayEq`; for value/index pairs use `OutputPairEqWhere`.

## Escape Hatches

Use `ComputeCorrect.Post` when a theorem fixes one initial state but the output
condition is not just a scalar/tile/pair:

```lean
ComputeCorrect.Post k s (fun s' => ...)
```

Use `ComputeCorrect.General` only when the theorem should quantify over the
initial state itself:

```lean
ComputeCorrect.General k (fun s0 s' => ...)
```

`ComputeRefine.Post` and `ComputeRefine.General` are the corresponding
two-kernel forms.

## Gap Policy

The `ComputeCorrect.*` / `ComputeRefine.*` helpers use the default
`GapPolicy.ignore`, matching current Real-first algorithm proofs. If a theorem
needs to record an externally checked compute-to-algorithm gap, state the
underlying `ComputeKernel.ComputeCorrect` / `ComputeKernel.ComputeRefine`
directly with:

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

Execution-only helper lemmas may use `_exec_view` and direct `exec` equalities.
The public theorem should wrap those helpers in `ComputeCorrect.*` or
`ComputeRefine.*`.
