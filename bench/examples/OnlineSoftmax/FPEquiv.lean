import bench.examples.OnlineSoftmax.Kernels
/- Complete two-pass online softmax versus the batch output. Both use libdevice.exp: the measured tl.exp exp-sub relation
failed the configured bias gate (0.1608954387 ULP > 0.05). -/
import bench.examples.OnlineSoftmax.Proofs.FP
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.OnlineSoftmaxFPEquiv
open VeriTile.Bench.Examples.OnlineSoftmax.Kernels
open VeriTile Triton FP.Exponential OnlineSoftmaxFPContract
open scoped VeriTile.Spec
open _root_.VeriTile.Triton.FP.Equational (Schedules)
open FP.GuardExpression
open SoftmaxStableFPContract (rowPlan center)
open OnlineSoftmaxFPComparison (rowExpressions)

/-- Check every online update and both normalization paths, including finite
sum intermediates and nonzero divisors. Also check the intermediate operands
needed to relate the sequential sum to the batch reduction tree. -/
def requirements (x : RegionName) (N : Nat) (plans : Schedules) : Condition :=
  let M := FP.GuardExpression.algebra MemoryInput
  let xs := rowExpressions x N
  let row := fun i : Fin N => xs i.val
  .all [FP.OnlineSoftmaxConditions.iterations M xs N,
    FP.SoftmaxShift.checks M row (FP.OnlineSoftmax.state M xs N).1 (prefixPlan N).tree,
    .guard .finite (FP.OnlineSoftmax.state M xs N).2,
    FP.SoftmaxShift.checks M row (center M row) (rowPlan plans N).tree,
    FP.WelfordConditions.schedule M (FP.SoftmaxShift.exponentials M row) (prefixPlan N) (rowPlan plans N)]

/-- Both public programs now write the same output row. -/
def batch (x y : RegionName) (N : Nat) : FP.Kernel₁ where
  io := {
    kernel := stableSoftmaxKernel x y N
    inp := x
    out := y
    Bin := N
    Bout := N
    read := fun pid => pid * N
    write := fun pid => pid * N }
  profile := FP.fp32
  domain := requirements x N
  comparisons := []

def online (x y : RegionName) (N : Nat) : FP.Kernel₁ :=
  { batch x y N with io := (batch x y N).io.withKernel (onlineSoftmaxKernel x y N) }

/-- Both kernels terminate, store the same normalized output row, and
preserve every other memory cell. The row length remains symbolic and positive. -/
specification online_softmax_equiv (N : Nat) (hN : 0 < N) (R : Rules) :
    batch "x" "y" N ≡[R] online "x" "y" N := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := batch "x" "y" N) (rhs := online "x" "y" N)
    (structural := fun _ _ => False) rfl rfl
  apply FP.Kernel₁.ofExecution
  exact output_equivalent R "x" "y" N hN

#print_fp_assumptions online_softmax_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Examples.OnlineSoftmaxFPEquiv
