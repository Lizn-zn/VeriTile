/- Batch softmax output versus the normalized value from the original online
m/l registers. Both use libdevice.exp: the measured tl.exp exp-sub relation
failed the configured bias gate (0.1608954387 ULP > 0.05). -/
import bench.examples.support.OnlineSoftmaxContract
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.OnlineSoftmaxFPEquiv
open VeriTile Triton FP.Exponential OnlineSoftmaxFPContract
open scoped VeriTile.Spec

/-- Preserve the existing Correct observation scope. The online source has no
output store; its readback computes exp(x - m) / l from the actual final m/l.
The row length remains symbolic and positive. -/
specification online_softmax_equiv (N : Nat) (hN : 0 < N) (R : Rules) :
    batchOutput "x" "y" N ≡[R] normalizedOnline "x" "y" N := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := batchOutput "x" "y" N) (rhs := normalizedOnline "x" "y" N)
    (structural := fun _ _ => False) rfl rfl
  exact observed_equivalent R "x" "y" N hN

#print_fp_assumptions online_softmax_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Examples.OnlineSoftmaxFPEquiv
