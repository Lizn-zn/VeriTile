import bench.examples.SoftmaxStable.Kernels
/- Stable versus naive softmax under admitted scalar FP assumptions.
Both implementations use libdevice.exp: the measured tl.exp exp-sub relation
failed the configured bias gate (0.1608954387 ULP > 0.05). -/
import bench.examples.SoftmaxStable.Proofs.FP
import VeriTile.Triton.Float.ExponentialLaws
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.SoftmaxStableFPEquiv
open VeriTile.Bench.Examples.SoftmaxStable.Kernels
open VeriTile Triton FP.Structural FP.Exponential
open SoftmaxStableFPContract SoftmaxStableFPExecution
open scoped VeriTile.Spec
open SoftmaxStableFPContract (rowExpressions center rowPlan)
open _root_.VeriTile.Triton.FP.Equational (Schedules)
open FP.GuardExpression

def naiveIO (x y : RegionName) (B : Nat) : KernelIO₁ where
  kernel := naiveSoftmaxKernel x y B
  inp := x
  out := y
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

def stableIO (x y : RegionName) (B : Nat) : KernelIO₁ :=
  (naiveIO x y B).withKernel (stableSoftmaxKernel x y B)

/-- Check finite inputs, the row maximum, exponentials and sum intermediates,
plus the nonzero denominators needed by the scalar derivation. The same
addition tree is used by both kernels; inputs need not be positive. -/
def requirements (x : RegionName) (B : Nat) (plans : Schedules) : Condition :=
  let M := FP.GuardExpression.algebra MemoryInput
  let xs := rowExpressions x B
  FP.SoftmaxShift.checks M xs (center M xs) (rowPlan plans B).tree

def naive (x y : RegionName) (B : Nat) : FP.Kernel₁ where
  io := naiveIO x y B
  profile := FP.fp32
  domain := requirements x B
  comparisons := []

def stable (x y : RegionName) (B : Nat) : FP.Kernel₁ :=
  { naive x y B with io := stableIO x y B }

/-- Row length is symbolic. The shared contract retains the intermediate
finite/nonzero domain; the experiment selects scalar assumptions only. -/
specification softmax_stable_equiv (B : Nat) (hB : 0 < B) (R : Rules) :
    stable "x" "y" B ≡[R] naive "x" "y" B := by
  apply Spec.FloatingPoint.ofNumerical (lhs := stable "x" "y" B) (rhs := naive "x" "y" B)
    (structural := fun _ _ => False) rfl rfl
  apply FP.Kernel₁.ofExecution
  exact SoftmaxStableFPContract.equivalent B hB R

#print_fp_assumptions softmax_stable_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Examples.SoftmaxStableFPEquiv
