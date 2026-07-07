/-
VeriTile.Math.Softmax

Reusable mathematical specifications for softmax-style kernels.
-/

import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic.FieldSimp
import VeriTile.Math.LogSumExp

namespace VeriTile

namespace TiledSoftmax

/-- Naive softmax over a tile of length `n` in ℝ. -/
noncomputable def naiveSoftmaxMath {n : Nat} (x : Fin n → ℝ) : Fin n → ℝ :=
  fun i => Real.exp (x i) / ∑ j, Real.exp (x j)

/-- Max-subtracted softmax over a tile of length `n` in ℝ. -/
noncomputable def stableSoftmaxMath {n : Nat} (x : Fin n → ℝ) (m : ℝ) :
    Fin n → ℝ :=
  fun i => Real.exp (x i - m) / ∑ j, Real.exp (x j - m)

/-- Naive and max-subtracted softmax produce the same output for any shift `m`.
The load-bearing mathematical fact behind the numerical-stability trick. -/
theorem naive_eq_stable {n : Nat} (x : Fin n → ℝ) (m : ℝ) :
    naiveSoftmaxMath x = stableSoftmaxMath x m := by
  funext i
  unfold naiveSoftmaxMath stableSoftmaxMath
  simp only [Real.exp_sub]
  rw [← Finset.sum_div]
  have hm : Real.exp m ≠ 0 := Real.exp_ne_zero m
  field_simp

/-- What a naive softmax kernel writes at lane `i`. -/
noncomputable def naiveSpec {blockSize : Nat}
    (xs : Fin blockSize → ℝ) (i : Fin blockSize) : ℝ :=
  Real.exp (xs i) / ∑ j, Real.exp (xs j)

/-- What a max-shifted stable softmax kernel writes at lane `i`. -/
noncomputable def stableSpec {blockSize : Nat}
    (xs : Fin blockSize → ℝ) (m : ℝ) (i : Fin blockSize) : ℝ :=
  Real.exp (xs i - m) / ∑ j, Real.exp (xs j - m)

/-- Max of a non-empty tile, in `Tile.reduceMax`'s form. -/
noncomputable abbrev tileMax {blockSize : Nat}
    (h : 0 < blockSize) (xs : Fin blockSize → ℝ) : ℝ :=
  TiledLogSumExp.blockMax h xs

end TiledSoftmax

end VeriTile
