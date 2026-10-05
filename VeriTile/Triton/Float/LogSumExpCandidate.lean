import VeriTile.Triton.Float.LogExp

/-!
Compose the two admitted log rewrites for the suffix of shifted log-sum-exp.
This is a derivation from scalar atoms, not a new whole-kernel assumption.
The source implementing this expression is candidateLSEKernel in
bench/examples/StableLogSumExp/Kernels.lean.
-/

noncomputable section
namespace VeriTile.Triton.FP.LogSumExpCandidate
open Structural Guarded

/-- Product-dependent splitting followed by conditional log-exp elimination.
The outer inactive split masks the center to zero. In the active split arm,
the inner fallback retains tl.log(libdevice.exp(center)). -/
def finish {α : Type} (M : Algebra α) (lt le : α → α → Bool) (sum center : α) : α :=
  let p := M.binary (some .fp32) .real .mul sum
    (M.unary (some .fp32) .libdeviceExp center)
  let keep := le (M.literal (some .fp32) .real (1 / 2)) p &&
    le p (M.literal (some .fp32) .real 2)
  let direct := M.unary (some .fp32) .log
    (if keep then p else M.literal (some .fp32) .real 1)
  let splitSum := if keep then M.literal (some .fp32) .real 1 else sum
  let splitCenter := if keep then M.literal (some .fp32) .real 0 else center
  let split := M.binary (some .fp32) .real .add
    (M.unary (some .fp32) .log splitSum)
    (LogExp.value .tl M lt le splitCenter)
  if keep then direct else split

/-- The branch thresholds select implementations, not an extra input domain.
Only the finite/positive operands required by the two scalar atoms are used.
In particular, no unconditional log-product or log-exp identity is assumed. -/
theorem finish_eq {α : Type} [Inhabited α] (R : LogExp.Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (lt le : α → α → Bool)
    (hlt : M.compareLt (some .fp32) .real = some lt)
    (hle : M.compareLe (some .fp32) .real = some le)
    (sum center : α) (hs : D .finite sum) (hps : D .positive sum)
    (hm : D .finite center)
    (he : D .finite (M.unary (some .fp32) .libdeviceExp center))
    (hpe : D .positive (M.unary (some .fp32) .libdeviceExp center)) :
    finish M lt le sum center = M.unary (some .fp32) .log
      (M.binary (some .fp32) .real .mul sum
        (M.unary (some .fp32) .libdeviceExp center)) := by
  have hsplit := LogExp.apply_log_mul_split .tl R (by decide) M D hM s le hle sum _ hs he hps hpe
  simp only [LogExp.Backend.logOp] at hsplit
  rw [hsplit]
  have hlog := LogExp.apply_log_exp_cancel .tl R (by decide) M D hM s lt le hlt hle center hm
  unfold finish LogExp.splitProductValue
  dsimp only
  split <;> simp_all [LogExp.referenceValue, LogExp.Backend.logOp]

end VeriTile.Triton.FP.LogSumExpCandidate
