import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Natural-exponential normalization with an arbitrary finite block center.
The source may replace an all-sentinel maximum by zero. The identities below
therefore do not require the center to equal the maximum. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.NormalizedStep
open VeriTile.Triton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000

/-- Bottom denotes the initial empty prefix. -/
def factor (m : WithBot ℝ) : ℝ := m.elim 0 (fun r => Real.exp (-r))
def alpha (m : WithBot ℝ) (next : ℝ) : ℝ :=
  (WithBot.realExp (WithBot.realSub m (↑next))).unbotD 0

theorem shifted_exp (x center : ℝ) :
    Real.exp (x-center) = Real.exp (-center)*Real.exp x := by
  rw [← Real.exp_add]
  congr 1
  ring

theorem rescale_prefix (m : WithBot ℝ) (next l total : ℝ)
    (hl : l = factor m * total) (hbot : m = ⊥ → total = 0) :
    alpha m next*l = Real.exp (-next)*total := by
  cases m with
  | bot =>
    change l = 0*total at hl
    rw [hl,zero_mul,mul_zero,hbot rfl,mul_zero]
  | coe m =>
    rw [hl]
    change alpha (↑m) next * (Real.exp (-m)*total) = _
    simp only [alpha,WithBot.realSub_coe_coe,WithBot.realExp_coe,WithBot.unbotD_coe]
    rw [← mul_assoc,← Real.exp_add]
    congr 2
    ring

def blockSum {N : Nat} (scores : Fin N → ℝ) (center : ℝ) : ℝ :=
  ∑ j, Real.exp (scores j-center)

theorem blockSum_eq {N : Nat} (scores : Fin N → ℝ) (center : ℝ) :
    blockSum scores center = Real.exp (-center)*(∑ j, Real.exp (scores j)) := by
  simp only [blockSum,shifted_exp,Finset.mul_sum]

theorem blockSum_positive {N : Nat} (hN : 0 < N) (scores : Fin N → ℝ) (center : ℝ) :
    0 < blockSum scores center := by
  apply Finset.sum_pos'
  · intro j _; exact le_of_lt (Real.exp_pos _)
  · exact ⟨⟨0,hN⟩,Finset.mem_univ _,Real.exp_pos _⟩

/-- Raw prefix sums determine the updated denominator, even when the selected
block center differs from the block maximum. -/
theorem denominator_update {N : Nat} (scores : Fin N → ℝ)
    (m : WithBot ℝ) (center next l total : ℝ)
    (hl : l = factor m*total) (hbot : m = ⊥ → total = 0) :
    alpha m next*l + Real.exp (center-next)*blockSum scores center =
      Real.exp (-next)*(total+∑ j,Real.exp (scores j)) := by
  rw [rescale_prefix m next l total hl hbot,blockSum_eq,← mul_assoc,← Real.exp_add]
  have h : center-next+-center = -next := by ring
  rw [h,mul_add]

/-- Positive block width and a nonnegative prefix sum make the source's
zero-denominator guard unreachable in real arithmetic. -/
theorem denominator_positive {N : Nat} (hN : 0 < N) (scores : Fin N → ℝ)
    (m : WithBot ℝ) (center next l total : ℝ)
    (hl : l = factor m*total) (hbot : m = ⊥ → total = 0) (ht : 0 ≤ total) :
    0 < alpha m next*l + Real.exp (center-next)*blockSum scores center := by
  rw [denominator_update scores m center next l total hl hbot]
  apply mul_pos (Real.exp_pos _)
  have hsum : 0 < ∑ j, Real.exp (scores j) := by
    apply Finset.sum_pos'
    · intro j _; exact le_of_lt (Real.exp_pos _)
    · exact ⟨⟨0,hN⟩,Finset.mem_univ _,Real.exp_pos _⟩
  linarith

theorem denominator_guard {N : Nat} (hN : 0 < N) (scores : Fin N → ℝ)
    (m : WithBot ℝ) (center next l total : ℝ)
    (hl : l = factor m*total) (hbot : m = ⊥ → total = 0) (ht : 0 ≤ total) :
    let v := alpha m next*l + Real.exp (center-next)*blockSum scores center
    (if v=0 then (1:ℝ)/1000000000 else v) = v := by
  dsimp only
  rw [if_neg (ne_of_gt (denominator_positive hN scores m center next l total hl hbot ht))]

/-- One normalized accumulator update expressed solely by raw weighted sums. -/
theorem accumulator_update {N : Nat} (scores values : Fin N → ℝ)
    (m : WithBot ℝ) (center next l total numerator acc : ℝ)
    (hl : l = factor m*total) (hbot : m = ⊥ → total = 0)
    (hacc : acc*total = numerator) :
    let after := total+∑ j,Real.exp (scores j)
    let denominator := Real.exp (-next)*after
    acc*(l/denominator*alpha m next) +
      (∑ j, (Real.exp (scores j-center)*(Real.exp (center-next)/denominator))*values j) =
      (numerator+∑ j, Real.exp (scores j)*values j)/after := by
  dsimp only
  have hfac : Real.exp (-next) ≠ 0 := ne_of_gt (Real.exp_pos _)
  have hprefix := rescale_prefix m next l total hl hbot
  have hp : l/(Real.exp (-next)*(total+∑ j,Real.exp (scores j)))*alpha m next =
      total/(total+∑ j,Real.exp (scores j)) := by
    rw [div_mul_eq_mul_div,mul_comm l,hprefix]
    exact mul_div_mul_left _ _ hfac
  rw [hp,← mul_div_assoc,hacc]
  have each (j : Fin N) :
      (Real.exp (scores j-center)*(Real.exp (center-next)/
        (Real.exp (-next)*(total+∑ k,Real.exp (scores k)))))*values j =
        (Real.exp (scores j)*values j)/(total+∑ k,Real.exp (scores k)) := by
    rw [← mul_div_assoc,← Real.exp_add]
    have h : scores j-center+(center-next) = -next+scores j := by ring
    rw [h,Real.exp_add,mul_div_mul_left _ _ hfac,div_mul_eq_mul_div]
  simp_rw [each]
  rw [← Finset.sum_div,← add_div]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.NormalizedStep
