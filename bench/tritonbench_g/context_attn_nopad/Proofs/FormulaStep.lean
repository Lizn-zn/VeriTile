import bench.tritonbench_g.context_attn_nopad.Proofs.Formula
import bench.tritonbench_g.context_attn_nopad.Proofs.NormalizedStep
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.FormulaStep
open VeriTile.Triton Formula NormalizedStep
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000

def maskedScore (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : WithBot ℝ :=
  if j ≤ queryRow p l i then some (Formula.score p l a scale i j) else ⊥

theorem weight_matches (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) :
    NormalizedStep.weight (maskedScore p l a scale i j) = Formula.weight p l a scale i j := by
  unfold maskedScore Formula.weight
  split_ifs <;> rfl

/-- Bottom block centers are allowed. Their lanes are all masked, and contribute
zero to the raw sums. The updated running center itself is finite. -/
theorem update (p : Layout) (launch : Launch) (a : Inputs) (scale : ℝ) (i d c : Nat)
    (m center : WithBot ℝ) (next l acc : ℝ) (hBN : 0 < p.BN)
    (hl : l = factor m*Formula.denominator p launch a scale i c)
    (hbot : m = ⊥ → c = 0)
    (hcenter : ∀ j : Fin p.BN, center=⊥ → maskedScore p launch a scale i (c*p.BN+j.val)=⊥)
    (hacc : acc = Formula.numerator p launch a scale i d c / Formula.denominator p launch a scale i c) :
    let scores := fun j : Fin p.BN => maskedScore p launch a scale i (c*p.BN+j.val)
    let values := fun j : Fin p.BN => Formula.value p launch a (c*p.BN+j.val) d
    let den := alpha m next*l+alpha center next*(∑ j, shifted (scores j) center)
    den = Real.exp (-next)*Formula.denominator p launch a scale i (c+1) ∧
    acc*(l/den*alpha m next)+
      (∑ j, (shifted (scores j) center*(alpha center next/den))*values j) =
      Formula.numerator p launch a scale i d (c+1)/Formula.denominator p launch a scale i (c+1) := by
  dsimp only
  let scores := fun j : Fin p.BN => maskedScore p launch a scale i (c*p.BN+j.val)
  let values := fun j : Fin p.BN => Formula.value p launch a (c*p.BN+j.val) d
  have hz : m=⊥ → Formula.denominator p launch a scale i c=0 := by
    intro h
    rw [hbot h,Formula.denominator_zero]
  have hd := denominator_update scores m center next l (Formula.denominator p launch a scale i c) hl hz hcenter
  simp only [scores,weight_matches,← Formula.denominator_succ] at hd
  rw [hd]
  constructor
  · rfl
  · have ht : acc*Formula.denominator p launch a scale i c = Formula.numerator p launch a scale i d c := by
      by_cases hc : c=0
      · subst c
        simp only [Formula.denominator_zero,Formula.numerator_zero,mul_zero]
      · rw [hacc]
        exact div_mul_cancel₀ _ (ne_of_gt (Formula.denominator_positive p launch a scale i c hBN (Nat.pos_of_ne_zero hc)))
    have h := accumulator_update scores values m center next l (Formula.denominator p launch a scale i c)
      (Formula.numerator p launch a scale i d c) acc hl hz hcenter ht
    simpa only [scores,values,weight_matches,← Formula.denominator_succ,← Formula.numerator_succ] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.FormulaStep
