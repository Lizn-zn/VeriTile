import bench.tritonbench_g.context_attn_bloom.Proofs.Math
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.NormalizedFormula
open VeriTile.Triton Math
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def weightSum (S hi : Nat) (g : Fin S → ℝ × ℝ) : ℝ :=
  ((gKeysUpto S hi g).map (fun p => pow2 p.1)).sum

def weightedValueSum (S hi : Nat) (g : Fin S → ℝ × ℝ) : ℝ :=
  ((gKeysUpto S hi g).map (fun p => pow2 p.1 * p.2)).sum

/-- Moving a scaled prefix sum to a finite new maximum has a common factor.
The empty prefix is represented by bottom and has mathematical sum zero. -/
theorem rescale_sum (m : WithBot ℝ) (next value raw : ℝ)
    (h : value = m.elim 0 (fun r => pow2 (-r)) * raw)
    (hbot : m = ⊥ → raw = 0) :
    value * (WithBot.realExp2 (WithBot.realSub m (↑next))).unbotD 0 = pow2 (-next) * raw := by
  cases m with
  | bot => simp_all [WithBot.realSub_bot_left,WithBot.realExp2_bot]
  | coe m =>
    change value = pow2 (-m) * raw at h
    rw [h,WithBot.realSub_coe_coe,WithBot.realExp2_coe,WithBot.unbotD_coe,
      show Real.exp ((m-next)*Real.log 2) = pow2 (m-next) by simp [pow2,mul_comm]]
    rw [mul_right_comm,← pow2_add]
    ring_nf

/-- A specification using raw exponential sums, with no running maxima,
shifted exponentials, or implementation state. The causal-block guard is
retained because the source masks future scores with a finite sentinel. -/
def accumulator (S BN qpos : Nat) (g : Fin S → ℝ × ℝ) : Nat → ℝ
  | 0 => 0
  | c+1 =>
    let before := weightSum S (c*BN) g
    let after := weightSum S ((c+1)*BN) g
    accumulator S BN qpos g c * (if c*BN ≤ qpos then before/after else 1) +
      (weightedValueSum S ((c+1)*BN) g-weightedValueSum S (c*BN) g)/after

/-- Eliminate the kernel's maximum-dependent factors from one guarded step. -/
theorem step (S BN qpos c : Nat) (g : Fin S → ℝ × ℝ)
    (hBN : 0 < BN) (hwin : (c+1)*BN ≤ S) :
    gAccN S BN qpos g (c+1) =
      gAccN S BN qpos g c *
        (if c*BN ≤ qpos then weightSum S (c*BN) g/weightSum S ((c+1)*BN) g else 1) +
      (weightedValueSum S ((c+1)*BN) g-weightedValueSum S (c*BN) g)/weightSum S ((c+1)*BN) g := by
  have hn : (gStateBot S ((c+1)*BN) g).1 ≠ ⊥ := by
    rw [gStateBot_fst_eq_runningMax]
    exact gRunningMax_succ_ne_bot S BN c g hBN hwin
  obtain ⟨next,hm⟩ := WithBot.ne_bot_iff_exists.mp hn
  have hscale : pow2 (-next) ≠ 0 := ne_of_gt (by simp [pow2,Real.exp_pos])
  obtain ⟨hln,htn⟩ := gStateBot_consistent S ((c+1)*BN) g
  rw [← hm] at hln htn
  change (gStateBot S ((c+1)*BN) g).2.1 = pow2 (-next) * weightSum S ((c+1)*BN) g at hln
  change (gStateBot S ((c+1)*BN) g).2.2 = pow2 (-next) * weightedValueSum S ((c+1)*BN) g at htn
  have hl := rescale_sum (gStateBot S (c*BN) g).1 next (gStateBot S (c*BN) g).2.1
    (weightSum S (c*BN) g) (gStateBot_consistent S (c*BN) g).1
    (fun h => gKeysUpto_map_sum_eq_zero_of_bot S (c*BN) g h _)
  have ht := rescale_sum (gStateBot S (c*BN) g).1 next (gStateBot S (c*BN) g).2.2
    (weightedValueSum S (c*BN) g) (gStateBot_consistent S (c*BN) g).2
    (fun h => gKeysUpto_map_sum_eq_zero_of_bot S (c*BN) g h _)
  rw [gAccN_succ,← hm,hln,htn]
  rw [ht,← mul_sub]
  have hlratio :
      ((gStateBot S (c*BN) g).2.1 / (pow2 (-next)*weightSum S ((c+1)*BN) g)) *
        (WithBot.realExp2 (WithBot.realSub (gStateBot S (c*BN) g).1 (↑next))).unbotD 0 =
      weightSum S (c*BN) g / weightSum S ((c+1)*BN) g := by
    rw [div_mul_eq_mul_div,hl]
    exact mul_div_mul_left _ _ hscale
  rw [hlratio,mul_div_mul_left _ _ hscale]

theorem agrees (S BN qpos c : Nat) (g : Fin S → ℝ × ℝ)
    (hBN : 0 < BN) (hwin : c*BN ≤ S) :
    gAccN S BN qpos g c = accumulator S BN qpos g c := by
  induction c with
  | zero => rfl
  | succ c ih =>
    rw [step S BN qpos c g hBN hwin,accumulator,ih (by nlinarith)]

theorem natural_exp_score (x : ℝ) : pow2 (x/Real.log 2) = Real.exp x := by
  have hl : Real.log 2 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  unfold pow2
  congr 1
  field_simp

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.NormalizedFormula
