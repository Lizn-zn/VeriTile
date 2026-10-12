import bench.examples.StableLogSumExp.Proofs.UnconditionalModel
import VeriTile.Triton.Float.Disprove

/-! Disproof of the historical FP specification without fallbacks.

The shared sources in ../Kernels.lean compute:
  directLSEKernel: bf16(tl.log(sum(libdevice.exp(x))))
  stableLSEKernel: bf16(m + tl.log(sum(libdevice.exp(x - m)))), m = max(x).
The historical source remains correct over the reals (LegacyCorrect.lean).

UnconditionalModel.lean constructs a model satisfying every currently selected
arithmetic/exp/log rule and the complete numerical domain. On a singleton row,
the actual kernels successfully store different bf16 values. Thus the old
specification is false under this abstract theory, even after adding the
admitted conditional log rules. This is a logical countermodel, not a GPU
measurement, an IEEE claim, or a rejection of statistical two-gates admission.

The supported optimizedLSEKernel retains the fallbacks; its specifications
remain in ../Correct.lean and ../FPEquiv.lean. -/
namespace VeriTile.Bench.Examples.StableLogSumExpUnconditional
open VeriTile Triton StableLogSumExpFPContract
open scoped VeriTile.Spec

def equivalenceGoal (B : Nat) (R : FP.Exponential.Rules) : Prop :=
  0 < B → stable "x" "y" B ≡[R] direct "x" "y" B

/-- A positive singleton row already refutes the historical target. -/
refutation unconditional_disproved (R : FP.Exponential.Rules) : equivalenceGoal 1 R := by
  disprove [Model.exponential_models R, Model.domain_holds, Model.stored_outputs]

/-- The same counterexample satisfies the additional admitted log rules.
Adding those rules does not justify removing the fallback branches. -/
refutation unconditional_disproved_with_log_rules (R : FP.LogSumExp.Rules) :
    stable "x" "y" 1 ≡[R] direct "x" "y" 1 := by
  disprove [Model.models R, Model.domain_holds, Model.stored_outputs]

#axiomsClean unconditional_disproved
#axiomsClean unconditional_disproved_with_log_rules

end VeriTile.Bench.Examples.StableLogSumExpUnconditional
