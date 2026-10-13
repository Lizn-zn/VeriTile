import VeriTile.Triton.Math.L2Norm
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.L2NormTriton1
open VeriTile.Triton.TiledL2Norm
/-- The independent row formula; inactive lanes contribute zero to the sum.
At the public boundary N≤B ensures that this sum covers the complete row. -/
noncomputable def rowFormula (N B : Nat) (eps : ℝ)
    (xs : Fin B → ℝ) (i : Fin B) : ℝ :=
  xs i * (1 / Real.sqrt ((∑ j : Fin B, if j.val < N then xs j * xs j else 0) + eps))

theorem padded_formula (N B : Nat) (eps : ℝ) (xs : Fin B → ℝ)
    (i : Fin B) (hi : i.val < N) :
    l2Norm (fun j : Fin B => if j.val < N then xs j else 0) eps i =
      rowFormula N B eps xs i := by
  have hs : (∑ j : Fin B, (if j.val < N then xs j else 0) *
      (if j.val < N then xs j else 0)) =
      ∑ j : Fin B, if j.val < N then xs j * xs j else 0 := by
    apply Finset.sum_congr rfl
    intro j _
    by_cases hj : j.val < N <;> simp [hj]
  simp only [l2Norm,l2NormRstd,l2NormSqSum,rowFormula,if_pos hi]
  rw [hs]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormTriton1
