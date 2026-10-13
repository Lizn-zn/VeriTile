import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.Floor.Div
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullMath
open scoped BigOperators
noncomputable section

/-- Independent pointwise formulas, with the source's epsilon clamp only in the default branch. -/
def loss (logTarget : Bool) (eps prediction target : ℝ) : ℝ :=
  if logTarget then Real.exp target * (target - prediction)
  else target * (Real.log (max target eps) - prediction)

/-- Mathematical padding vanishes in both branches; this is not an IEEE claim. -/
theorem loss_zero (logTarget : Bool) (eps : ℝ) : loss logTarget eps 0 0 = 0 := by
  cases logTarget <;> simp [loss]

/-- The two masked loads use their explicit zero values before arithmetic. -/
def paddedLoss (logTarget : Bool) (eps : ℝ) (prediction target : Nat → ℝ)
    (N k : Nat) : ℝ :=
  loss logTarget eps (if k < N then prediction k else 0)
    (if k < N then target k else 0)

theorem paddedLoss_eq (logTarget : Bool) (eps : ℝ) (prediction target : Nat → ℝ)
    (N k : Nat) :
    paddedLoss logTarget eps prediction target N k =
      if k < N then loss logTarget eps (prediction k) (target k) else 0 := by
  by_cases h : k < N <;> simp [paddedLoss, h, loss_zero]

/-- One complete block reduction, including all padded lanes. -/
def blockLoss (logTarget : Bool) (eps : ℝ) (prediction target : Nat → ℝ)
    (N B chunk : Nat) : ℝ :=
  ∑ lane : Fin B, paddedLoss logTarget eps prediction target N (chunk * B + lane.val)

/-- Independent logical-row specification for the accumulated reduction. -/
def rowLoss (logTarget : Bool) (eps : ℝ) (prediction target : Nat → ℝ)
    (N : Nat) : ℝ :=
  ∑ k : Fin N, loss logTarget eps (prediction k.val) (target k.val)

/-- Number of positive-stride blocks needed to visit the logical row. -/
def rounds (N B : Nat) : Nat := (N + B - 1) / B

theorem rounds_cover (N B : Nat) (hB : 0 < B) : N ≤ rounds N B * B := by
  have h := (ceilDiv_le_iff_le_mul (a := B) (b := N) hB).mp
    (show N ⌈/⌉ B ≤ N ⌈/⌉ B from le_rfl)
  simpa only [rounds, Nat.ceilDiv_eq_add_pred_div, Nat.mul_comm] using h

/-- Reduction prefix after a specified number of complete blocks. -/
def prefixLoss (logTarget : Bool) (eps : ℝ) (prediction target : Nat → ℝ)
    (N B chunks : Nat) : ℝ :=
  ∑ chunk ∈ Finset.range chunks, blockLoss logTarget eps prediction target N B chunk

theorem prefixLoss_zero (logTarget : Bool) (eps : ℝ) (prediction target : Nat → ℝ)
    (N B : Nat) : prefixLoss logTarget eps prediction target N B 0 = 0 := by
  simp [prefixLoss]

theorem prefixLoss_succ (logTarget : Bool) (eps : ℝ) (prediction target : Nat → ℝ)
    (N B chunks : Nat) :
    prefixLoss logTarget eps prediction target N B (chunks + 1) =
      prefixLoss logTarget eps prediction target N B chunks +
        blockLoss logTarget eps prediction target N B chunks := by
  simp only [prefixLoss, Finset.sum_range_succ]

/-- Block reduction and sequential accumulation cover the actual padded prefix. -/
theorem prefixLoss_eq_paddedRange (logTarget : Bool) (eps : ℝ)
    (prediction target : Nat → ℝ) (N B chunks : Nat) :
    prefixLoss logTarget eps prediction target N B chunks =
      ∑ k ∈ Finset.range (chunks * B), paddedLoss logTarget eps prediction target N k := by
  induction chunks with
  | zero => simp [prefixLoss]
  | succ chunks ih =>
    rw [prefixLoss_succ, ih, Nat.succ_mul, Finset.sum_range_add]
    congr 1
    exact Fin.sum_univ_eq_sum_range
      (fun k ↦ paddedLoss logTarget eps prediction target N (chunks * B + k)) B

/-- Once all logical columns are covered, padded real reductions equal the row formula. -/
theorem prefixLoss_eq_rowLoss (logTarget : Bool) (eps : ℝ)
    (prediction target : Nat → ℝ) (N B chunks : Nat) (hCover : N ≤ chunks * B) :
    prefixLoss logTarget eps prediction target N B chunks =
      rowLoss logTarget eps prediction target N := by
  rw [prefixLoss_eq_paddedRange]
  simp_rw [paddedLoss_eq]
  let f : Nat → ℝ := fun k ↦ loss logTarget eps (prediction k) (target k)
  change (∑ k ∈ Finset.range (chunks * B), if k < N then f k else 0) = _
  calc
    _ = ∑ k ∈ Finset.range N, if k < N then f k else 0 := by
      symm
      apply Finset.sum_subset (Finset.range_mono hCover)
      intro k _ hk
      have h : ¬ k < N := by simpa only [Finset.mem_range] using hk
      simp [h]
    _ = ∑ k ∈ Finset.range N, f k := by
      apply Finset.sum_congr rfl
      intro k hk
      exact if_pos (Finset.mem_range.mp hk)
    _ = rowLoss logTarget eps prediction target N :=
      (Fin.sum_univ_eq_sum_range f N).symm

theorem complete_prefix (logTarget : Bool) (eps : ℝ)
    (prediction target : Nat → ℝ) (N B : Nat) (hB : 0 < B) :
    prefixLoss logTarget eps prediction target N B (rounds N B) =
      rowLoss logTarget eps prediction target N :=
  prefixLoss_eq_rowLoss logTarget eps prediction target N B (rounds N B)
    (rounds_cover N B hB)

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivOps.FullMath
