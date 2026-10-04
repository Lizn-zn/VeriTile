import bench.examples.SoftmaxStable.Kernels
/- Stable versus naive softmax under admitted scalar FP assumptions.
Both implementations use libdevice.exp: the measured tl.exp exp-sub relation
failed the configured bias gate (0.1608954387 ULP > 0.05). -/
import bench.examples.SoftmaxStable.Contract
import VeriTile.Triton.Float.ExponentialLaws
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.SoftmaxStableFPEquiv
open VeriTile.Bench.Examples.SoftmaxStable.Kernels
open VeriTile Triton FP.Structural FP.Exponential
open SoftmaxStableFPContract SoftmaxStableFPExecution
open scoped VeriTile.Spec

/-- Row length is symbolic. The shared contract retains the intermediate
finite/nonzero domain; the experiment selects scalar assumptions only. -/
specification softmax_stable_equiv (B : Nat) (hB : 0 < B) (R : Rules) :
    stable "x" "y" B ≡[R] naive "x" "y" B := by
  apply Spec.FloatingPoint.ofNumerical (lhs := stable "x" "y" B) (rhs := naive "x" "y" B)
    (structural := fun _ _ => False) rfl rfl
  refine ⟨by simp [stable, IO₁PrivateScratch, stableIO, naiveIO],
    by simp [naive, IO₁PrivateScratch, naiveIO], ?_⟩
  intro α _ M D hM plans s hd
  exact (original_runs_under_exp R.arithmetic M D (arithmetic_models R M D hM) s plans
    "x" "y" B hB (exp_sub R M D hM s) hd).2.2

#print_fp_assumptions softmax_stable_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Examples.SoftmaxStableFPEquiv
