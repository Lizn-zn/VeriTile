import bench.examples.OnlineSoftmax.Kernels
/- Complete two-pass online softmax versus the batch output. Both use libdevice.exp: the measured tl.exp exp-sub relation
failed the configured bias gate (0.1608954387 ULP > 0.05). -/
import bench.examples.OnlineSoftmax.Proofs.FP
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.OnlineSoftmaxFPEquiv
open VeriTile.Bench.Examples.OnlineSoftmax.Kernels
open VeriTile Triton FP.Exponential OnlineSoftmaxFPContract
open scoped VeriTile.Spec

/-- Both kernels terminate, store the same normalized output row, and
preserve every other memory cell. The row length remains symbolic and positive. -/
specification online_softmax_equiv (N : Nat) (hN : 0 < N) (R : Rules) :
    batch "x" "y" N ≡[R] online "x" "y" N := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := batch "x" "y" N) (rhs := online "x" "y" N)
    (structural := fun _ _ => False) rfl rfl
  exact output_equivalent R "x" "y" N hN

#print_fp_assumptions online_softmax_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Examples.OnlineSoftmaxFPEquiv
