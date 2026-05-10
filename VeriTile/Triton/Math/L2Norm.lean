/-
VeriTile.Triton.Math.L2Norm

Reusable mathematical specifications for row-wise L2 normalization (RMS-norm
without an affine scale): forward output, reciprocal standard deviation, and
the gradient w.r.t. the input.

This is the pure-math layer. Definitions here only depend on Mathlib's
analytic primitives and big-operators; they do not touch `BlockState`,
`RegionName`, or memory layout. Kernel transcriptions in
`bench/l2_norm_*` should bridge their `Tile.reduceSum`-based carriers to
these operators rather than re-deriving the same formula.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Triton

namespace TiledL2Norm

/-- Squared 2-norm: `‖xs‖² = ∑ xᵢ²`. -/
noncomputable def l2NormSqSum {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  ∑ i, xs i * xs i

/-- Reciprocal standard deviation with epsilon stabilizer:
    `rstd(xs, ε) = 1 / √(‖xs‖² + ε)`. -/
noncomputable def l2NormRstd {N : Nat} (xs : Fin N → ℝ) (ε : ℝ) : ℝ :=
  1 / Real.sqrt (l2NormSqSum xs + ε)

/-- L2-normalized output at lane `i`: `xᵢ · rstd(xs, ε)`. -/
noncomputable def l2Norm {N : Nat} (xs : Fin N → ℝ) (ε : ℝ) (i : Fin N) : ℝ :=
  xs i * l2NormRstd xs ε

/-- Cross dot product `⟨dy, x⟩ = ∑ dyᵢ · xᵢ`, used by the L2-norm backward. -/
noncomputable def l2NormDot {N : Nat} (xs ys : Fin N → ℝ) : ℝ :=
  ∑ i, xs i * ys i

/-- Gradient of L2-normalize w.r.t. input at lane `i`:
    `dy · rstd  −  ⟨dy, x⟩ / (‖xs‖² + ε) · rstd · xᵢ`. -/
noncomputable def l2NormBwd {N : Nat}
    (xs dys : Fin N → ℝ) (ε : ℝ) (i : Fin N) : ℝ :=
  let rstd := l2NormRstd xs ε
  let dot := l2NormDot dys xs
  dys i * rstd - dot * (1 / (l2NormSqSum xs + ε)) * rstd * xs i

end TiledL2Norm

end VeriTile.Triton
