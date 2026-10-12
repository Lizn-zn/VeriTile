import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.NormalizedStep
open VeriTile.Triton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 800000

/-- Bottom is the source's negative-infinity score or empty prefix. -/
def weight (s : WithBot ℝ) : ℝ := s.elim 0 Real.exp
def factor (m : WithBot ℝ) : ℝ := m.elim 0 (fun r => Real.exp (-r))
def shifted (s m : WithBot ℝ) : ℝ := (WithBot.realExp (WithBot.realSub s m)).unbotD 0
def alpha (m : WithBot ℝ) (next : ℝ) : ℝ := shifted m (↑next)

theorem rescale_prefix (m : WithBot ℝ) (next l total : ℝ)
    (hl : l = factor m*total) (hbot : m=⊥ → total=0) :
    alpha m next*l = Real.exp (-next)*total := by
  cases m with
  | bot =>
    change l = 0*total at hl
    rw [hl,zero_mul,mul_zero,hbot rfl,mul_zero]
  | coe m =>
    rw [hl]
    change alpha (↑m) next*(Real.exp (-m)*total) = _
    simp only [alpha,shifted,WithBot.realSub_coe_coe,WithBot.realExp_coe,WithBot.unbotD_coe]
    rw [← mul_assoc,← Real.exp_add]
    congr 2
    ring

/-- A finite score cannot have a bottom block maximum. This implication is
sufficient to rescale every lane, including entirely masked blocks. -/
theorem rescale_lane (s center : WithBot ℝ) (next : ℝ)
    (hbot : center=⊥ → s=⊥) :
    shifted s center*alpha center next = Real.exp (-next)*weight s := by
  cases center with
  | bot =>
    rw [hbot rfl]
    change (0 : ℝ)*alpha ⊥ next = Real.exp (-next)*0
    simp
  | coe c =>
    cases s with
    | bot =>
      change (0 : ℝ)*alpha (↑c) next = Real.exp (-next)*0
      simp
    | coe s =>
      change Real.exp (s-c)*Real.exp (c-next) = Real.exp (-next)*Real.exp s
      rw [← Real.exp_add,← Real.exp_add]
      congr 1
      ring

theorem denominator_update {N : Nat} (scores : Fin N → WithBot ℝ)
    (m center : WithBot ℝ) (next l total : ℝ)
    (hl : l=factor m*total) (hbot : m=⊥ → total=0)
    (hc : ∀ j, center=⊥ → scores j=⊥) :
    alpha m next*l+alpha center next*(∑ j,shifted (scores j) center) =
      Real.exp (-next)*(total+∑ j,weight (scores j)) := by
  rw [rescale_prefix m next l total hl hbot,Finset.mul_sum]
  simp_rw [mul_comm (alpha center next),rescale_lane _ center next (hc _)]
  rw [← Finset.mul_sum,mul_add]

theorem accumulator_update {N : Nat} (scores : Fin N → WithBot ℝ) (values : Fin N → ℝ)
    (m center : WithBot ℝ) (next l total numerator acc : ℝ)
    (hl : l=factor m*total) (hbot : m=⊥ → total=0)
    (hc : ∀ j, center=⊥ → scores j=⊥) (ha : acc*total=numerator) :
    let after := total+∑ j,weight (scores j)
    let den := Real.exp (-next)*after
    acc*(l/den*alpha m next)+
      (∑ j, (shifted (scores j) center*(alpha center next/den))*values j) =
      (numerator+∑ j,weight (scores j)*values j)/after := by
  dsimp only
  have hn : Real.exp (-next) ≠ 0 := Real.exp_ne_zero _
  have hp : l/(Real.exp (-next)*(total+∑ j,weight (scores j)))*alpha m next =
      total/(total+∑ j,weight (scores j)) := by
    rw [div_mul_eq_mul_div,mul_comm l,rescale_prefix m next l total hl hbot]
    exact mul_div_mul_left _ _ hn
  rw [hp,← mul_div_assoc,ha]
  have each (j : Fin N) :
      (shifted (scores j) center*(alpha center next/
        (Real.exp (-next)*(total+∑ k,weight (scores k)))))*values j =
      (weight (scores j)*values j)/(total+∑ k,weight (scores k)) := by
    rw [← mul_div_assoc,rescale_lane _ center next (hc j),mul_div_mul_left _ _ hn,
      div_mul_eq_mul_div]
  simp_rw [each]
  rw [← Finset.sum_div,← add_div]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.NormalizedStep
