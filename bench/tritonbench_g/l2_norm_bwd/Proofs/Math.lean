import VeriTile.Triton.Math.L2Norm
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.L2NormBwd
open VeriTile.Triton.TiledL2Norm

/-- Independent finite-sum gradient formula; padding contributes zero. -/
noncomputable def rowFormula (N B : Nat) (eps : ℝ) (xs dys : Fin B → ℝ) (i : Fin B) : ℝ :=
  let variance := ∑ j : Fin B, if j.val < N then xs j * xs j else 0
  let rstd := 1 / Real.sqrt (variance + eps)
  dys i * rstd - (∑ j : Fin B, if j.val < N then dys j * xs j else 0) *
    (1 / (variance + eps)) * rstd * xs i

theorem padded_formula (N B : Nat) (eps : ℝ) (xs dys : Fin B → ℝ)
    (i : Fin B) (hi : i.val < N) :
    l2NormBwd (fun j : Fin B => if j.val < N then xs j else 0)
      (fun j : Fin B => if j.val < N then dys j else 0) eps i = rowFormula N B eps xs dys i := by
  have hs (f g : Fin B → ℝ) :
      (∑ j : Fin B, (if j.val < N then f j else 0) * (if j.val < N then g j else 0)) =
      ∑ j : Fin B, if j.val < N then f j * g j else 0 := by
    apply Finset.sum_congr rfl
    intro j _
    by_cases hj : j.val < N <;> simp [hj]
  simp only [l2NormBwd,l2NormRstd,l2NormSqSum,l2NormDot,rowFormula,if_pos hi,hs]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormBwd
