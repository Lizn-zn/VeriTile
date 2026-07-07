/-
VeriTile.Math.LogSumExp

Pure mathematical log-sum-exp operators and identities.

This module intentionally does not mention `Tile`, `WithBot`, masks, memory,
or program ids.  Kernel/tile semantics bridge to these operators from
`VeriTile.Semantics.TiledIndexing` and
`VeriTile.Semantics.MaskedReduction`.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Finset.Lattice.Fold

namespace VeriTile

namespace TiledLogSumExp

/-! ## Pure operators -/

/-- Maximum of a nonempty fixed-size block. -/
noncomputable def blockMax {blockSize : Nat} (h : 0 < blockSize)
    (xs : Fin blockSize → ℝ) : ℝ :=
  Finset.univ.sup' (⟨⟨0, h⟩, Finset.mem_univ _⟩ :
    (Finset.univ : Finset (Fin blockSize)).Nonempty) xs

/-- Standard complete-row log-sum-exp. -/
noncomputable def LSE {D : Nat} (xs : Fin D → ℝ)
    (HAS_SCALE : Bool) (scale : ℝ) : ℝ :=
  let scaled := fun i : Fin D => if HAS_SCALE then xs i * scale else xs i
  Real.log (∑ i, Real.exp (scaled i))

/-- Stable log-sum-exp with an arbitrary shift `m`. -/
noncomputable def stableLSEWithShift {N : Nat} (xs : Fin N → ℝ) (m : ℝ) : ℝ :=
  m + Real.log (∑ j, Real.exp (xs j - m))

/-- Stable-form complete-row log-sum-exp using the block maximum as shift. -/
noncomputable def stableLSE {D : Nat} (xs : Fin D → ℝ) (hD : 0 < D)
    (HAS_SCALE : Bool) (scale : ℝ) : ℝ :=
  let scaled := fun i : Fin D => if HAS_SCALE then xs i * scale else xs i
  stableLSEWithShift scaled (blockMax hD scaled)

/-! ## Pure identities -/

/-- The load-bearing math identity behind stable log-sum-exp. -/
theorem log_sum_exp_shift_invariant {n : Nat} (hn : 0 < n) (x : Fin n → ℝ) (m : ℝ) :
    Real.log (∑ i, Real.exp (x i)) = stableLSEWithShift x m := by
  have h_factor : ∀ i : Fin n, Real.exp (x i) = Real.exp m * Real.exp (x i - m) := by
    intro i
    rw [← Real.exp_add]
    ring_nf
  rw [show (∑ i, Real.exp (x i)) = ∑ i, Real.exp m * Real.exp (x i - m) from
        Finset.sum_congr rfl (fun i _ => h_factor i)]
  rw [← Finset.mul_sum]
  have h_em_pos : 0 < Real.exp m := Real.exp_pos m
  have h_sum_pos : 0 < ∑ i, Real.exp (x i - m) := by
    apply Finset.sum_pos
    · intro i _; exact Real.exp_pos _
    · exact ⟨⟨0, hn⟩, Finset.mem_univ _⟩
  rw [Real.log_mul (ne_of_gt h_em_pos) (ne_of_gt h_sum_pos)]
  rw [Real.log_exp]
  rfl

/-- The stable max-shifted row LSE is equal to the standard mathematical LSE. -/
theorem stableLSE_eq_LSE {D : Nat} (xs : Fin D → ℝ) (hD : 0 < D)
    (HAS_SCALE : Bool) (scale : ℝ) :
    stableLSE xs hD HAS_SCALE scale = LSE xs HAS_SCALE scale := by
  unfold stableLSE LSE
  exact (log_sum_exp_shift_invariant hD
    (fun i : Fin D => if HAS_SCALE then xs i * scale else xs i)
    (blockMax hD (fun i : Fin D => if HAS_SCALE then xs i * scale else xs i))).symm

end TiledLogSumExp

end VeriTile
