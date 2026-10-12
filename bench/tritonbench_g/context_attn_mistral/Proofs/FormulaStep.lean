import bench.tritonbench_g.context_attn_mistral.Proofs.Formula
import bench.tritonbench_g.context_attn_mistral.Proofs.NormalizedStep
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.FormulaStep
open VeriTile.Triton Formula NormalizedStep
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000

/-- Every finite center (including the source's all-sentinel replacement) gives
the same new raw-sum denominator and normalized weighted output. -/
theorem update (p : Layout) (launch : Launch) (a : Inputs) (scale : ℝ) (i d c : Nat)
    (m : WithBot ℝ) (center next l acc : ℝ)
    (hBN : 0 < p.BN)
    (hl : l = factor m*Formula.denominator p launch a scale i c)
    (hbot : m = ⊥ → c = 0)
    (hacc : acc = Formula.numerator p launch a scale i d c / Formula.denominator p launch a scale i c) :
    let scores := fun j : Fin p.BN => Formula.score p launch a scale i (c*p.BN+j.val)
    let values := fun j : Fin p.BN => Formula.value p launch a (c*p.BN+j.val) d
    let raw := alpha m next*l+Real.exp (center-next)*blockSum scores center
    let den := if raw=0 then (1:ℝ)/1000000000 else raw
    den = Real.exp (-next)*Formula.denominator p launch a scale i (c+1) ∧
    acc*(l/den*alpha m next) +
      (∑ j, (Real.exp (scores j-center)*(Real.exp (center-next)/den))*values j) =
      Formula.numerator p launch a scale i d (c+1) / Formula.denominator p launch a scale i (c+1) := by
  dsimp only
  let scores := fun j : Fin p.BN => Formula.score p launch a scale i (c*p.BN+j.val)
  let values := fun j : Fin p.BN => Formula.value p launch a (c*p.BN+j.val) d
  have hz : m = ⊥ → Formula.denominator p launch a scale i c = 0 := by
    intro h
    rw [hbot h,Formula.denominator_zero]
  have hd := denominator_update scores m center next l (Formula.denominator p launch a scale i c) hl hz
  have hg := denominator_guard hBN scores m center next l (Formula.denominator p launch a scale i c)
    hl hz (Formula.denominator_nonnegative p launch a scale i c)
  rw [hg,hd,← Formula.denominator_succ]
  constructor
  · rfl
  · have ht : acc*Formula.denominator p launch a scale i c = Formula.numerator p launch a scale i d c := by
      by_cases hc : c=0
      · subst c
        simp only [Formula.denominator_zero,Formula.numerator_zero,mul_zero]
      · rw [hacc]
        exact div_mul_cancel₀ _ (ne_of_gt (Formula.denominator_positive p launch a scale i c hBN (Nat.pos_of_ne_zero hc)))
    have h := accumulator_update scores values m center next l (Formula.denominator p launch a scale i c)
      (Formula.numerator p launch a scale i d c) acc hl hz ht
    simpa only [scores,values,← Formula.denominator_succ,← Formula.numerator_succ] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.FormulaStep
