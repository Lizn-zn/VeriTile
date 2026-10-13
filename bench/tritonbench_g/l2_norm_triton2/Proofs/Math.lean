import VeriTile.Triton.Math.L2Norm
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.L2NormTriton2
open scoped BigOperators

/-- Reindex the active portion of a padded tile onto the entire logical row. -/
theorem sum_active_eq_row {N B : Nat} (hNB : N ≤ B) (f : Fin B → ℝ) :
    (∑ j : Fin B, if j.val < N then f j else 0) =
      ∑ j : Fin N, f (Fin.castLE hNB j) := by
  rw [← Finset.sum_filter]
  symm
  refine Finset.sum_bij (i := fun j _ ↦ Fin.castLE hNB j) ?_ ?_ ?_ ?_
  · intro j _
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Fin.val_castLE]
    exact j.isLt
  · intro a _ b _ hab
    exact Fin.ext (congrArg (fun j : Fin B ↦ j.val) hab)
  · intro j hj
    exact ⟨⟨j.val, (Finset.mem_filter.mp hj).2⟩, Finset.mem_univ _, by rfl⟩
  · intro j _
    rfl

/-- Independent forward formula over the logical, unpadded row. -/
noncomputable def rowForward {N : Nat} (eps : ℝ) (xs : Fin N → ℝ) (i : Fin N) : ℝ :=
  xs i * (1 / Real.sqrt ((∑ j, xs j * xs j) + eps))

/-- Independent backward formula; the two finite sums are the squared norm
and the input-gradient dot product, with the source multiplication order. -/
noncomputable def rowBackward {N : Nat} (eps : ℝ)
    (xs dys : Fin N → ℝ) (i : Fin N) : ℝ :=
  let variance := ∑ j, xs j * xs j
  let rstd := 1 / Real.sqrt (variance + eps)
  dys i * rstd - (∑ j, dys j * xs j) * (1 / (variance + eps)) * rstd * xs i

open VeriTile.Triton.TiledL2Norm

/-- Zero padding contributes no square mass outside the logical row. -/
theorem padded_square_sum {N B : Nat} (hNB : N ≤ B) (xs : Fin B → ℝ) :
    l2NormSqSum (fun j : Fin B ↦ if j.val < N then xs j else 0) =
      ∑ j : Fin N, xs (Fin.castLE hNB j) * xs (Fin.castLE hNB j) := by
  unfold l2NormSqSum
  rw [← sum_active_eq_row hNB (fun j ↦ xs j * xs j)]
  apply Finset.sum_congr rfl
  intro j _
  split <;> simp_all

/-- The backward dot also ranges over precisely the logical row. -/
theorem padded_dot_sum {N B : Nat} (hNB : N ≤ B) (xs dys : Fin B → ℝ) :
    l2NormDot (fun j : Fin B ↦ if j.val < N then dys j else 0)
      (fun j : Fin B ↦ if j.val < N then xs j else 0) =
      ∑ j : Fin N, dys (Fin.castLE hNB j) * xs (Fin.castLE hNB j) := by
  unfold l2NormDot
  rw [← sum_active_eq_row hNB (fun j ↦ dys j * xs j)]
  apply Finset.sum_congr rfl
  intro j _
  split <;> simp_all

/-- The padded specification equals normalization of the entire unpadded row. -/
theorem padded_forward_eq_row {N B : Nat} (hNB : N ≤ B) (eps : ℝ)
    (xs : Fin B → ℝ) (i : Fin N) :
    l2Norm (fun j : Fin B ↦ if j.val < N then xs j else 0) eps (Fin.castLE hNB i) =
      rowForward eps (fun j ↦ xs (Fin.castLE hNB j)) i := by
  simp only [l2Norm, l2NormRstd, Fin.val_castLE, if_pos i.isLt,
    padded_square_sum hNB, rowForward]

/-- The complete padded backward formula equals the full logical-row formula. -/
theorem padded_backward_eq_row {N B : Nat} (hNB : N ≤ B) (eps : ℝ)
    (xs dys : Fin B → ℝ) (i : Fin N) :
    l2NormBwd (fun j : Fin B ↦ if j.val < N then xs j else 0)
      (fun j : Fin B ↦ if j.val < N then dys j else 0) eps (Fin.castLE hNB i) =
      rowBackward eps (fun j ↦ xs (Fin.castLE hNB j))
        (fun j ↦ dys (Fin.castLE hNB j)) i := by
  simp only [l2NormBwd, l2NormRstd, Fin.val_castLE, if_pos i.isLt,
    padded_square_sum hNB, padded_dot_sum hNB, rowBackward]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormTriton2
