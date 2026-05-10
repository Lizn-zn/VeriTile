/-
VeriTile.Triton.Math.Reduction

Reusable reduction-shaped mathematical specifications: sum and max over a
non-empty tile, two-pass arithmetic mean and population variance, layer
normalization.

This is the pure-math layer. Definitions here only depend on Mathlib's
big-operator infrastructure and `Real.sqrt`; they do not touch `BlockState`,
`RegionName`, or memory layout. The kernel-internal running Welford recurrence
lives in `VeriTile.Examples.WelfordKernels` because it interleaves with kernel
step semantics.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Finset.Lattice.Fold

namespace VeriTile.Triton

namespace TiledReduction

/-! ## Sum and max -/

/-- Sum over a tile of length `N`. -/
noncomputable def tileSum {N : Nat} (xs : Fin N → ℝ) : ℝ := ∑ i, xs i

/-- Maximum over a non-empty tile. The non-emptiness hypothesis mirrors
`Tile.reduceMax`, which is only defined on positive-length axes since
`Finset.sup'` requires a non-empty index set. -/
noncomputable def tileMax {N : Nat} (h : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  match N, h, xs with
  | _ + 1, _, xs => Finset.univ.sup' Finset.univ_nonempty xs

/-! ## Two-pass / Welford mean and variance -/

/-- Two-pass arithmetic mean: `μ = (∑ xᵢ) / N`. Coerces `N : Nat` to `ℝ`
through Lean's natural numeric coercion. -/
noncomputable def welfordMean {N : Nat} (xs : Fin N → ℝ) : ℝ := tileSum xs / N

/-- Sum of squared deviations from the mean: `S = ∑ (xᵢ − μ)²`. -/
noncomputable def welfordSumSq {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  ∑ i, (xs i - welfordMean xs) ^ 2

/-- Population variance: `σ² = S / N`. -/
noncomputable def welfordVar {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  welfordSumSq xs / N

/-! ## LayerNorm -/

/-- LayerNorm: `(xᵢ − μ) / √(var + ε) · γᵢ + βᵢ`, where `μ`, `var` are the
two-pass mean and population variance over the input tile. -/
noncomputable def layerNorm {N : Nat}
    (xs γs βs : Fin N → ℝ) (ε : ℝ) (i : Fin N) : ℝ :=
  (xs i - welfordMean xs) / Real.sqrt (welfordVar xs + ε) * γs i + βs i

end TiledReduction

end VeriTile.Triton
