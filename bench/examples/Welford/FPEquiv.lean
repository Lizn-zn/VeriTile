import bench.examples.Welford.Kernels
/- Original online versus two-pass Welford, under admitted scalar FP atoms.
The count bound is the integer conversion rule's domain, not the GPU probe's
shape. Empty Welford rows divide by zero and are outside this FP contract. -/
import bench.examples.Welford.Proofs.FP
import VeriTile.Triton.Float.CountConversionLaws
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.WelfordFPEquiv
open VeriTile.Bench.Examples.Welford.Kernels
open VeriTile Triton FP.Structural FP.CountConversion
open WelfordFPContract WelfordFPExecution
open _root_.VeriTile.Triton.FP.Equational (ReductionPlan)
open scoped VeriTile.Spec
open WelfordFPComparison (rowPlan)
open _root_.VeriTile.Triton.FP.Equational (Schedules)
open FP.GuardExpression

/-- The original implementations expose both statistics, with symbolic row
length and stride. These contracts add no numerical relation between them. -/
def onlineIO (x mean variance : RegionName) (N stride : Nat) : KernelIO₁ₓ₂ where
  kernel := onlineWelfordKernel x mean variance N stride
  inp := x
  out1 := mean
  out2 := variance
  Bin := N
  Bout1 := 1
  Bout2 := 1
  read := fun pid => pid * stride
  write1 := fun _ => 0
  write2 := fun _ => 0

def twopassIO (x mean variance : RegionName) (N stride : Nat) : KernelIO₁ₓ₂ :=
  (onlineIO x mean variance N stride).withKernel (twopassWelfordKernel x mean variance N stride)

/-- The numerical proof domain contains initialization, each iteration, and
both normalization paths for all three statistics. Dimensions stay symbolic. -/
def requirements (x : RegionName) (N stride : Nat) (empty : ReductionPlan 0)
    (plans : Schedules) : Condition :=
  FP.WelfordConditions.complete (FP.GuardExpression.algebra MemoryInput)
    (rowExpressions x stride) empty (rowPlan plans N)

def online (x mean variance : RegionName) (N stride : Nat) (empty : ReductionPlan 0) :
    FP.Kernel₁ₓ₂ where
  io := onlineIO x mean variance N stride
  profile := FP.fp32
  domain := requirements x N stride empty

def twopass (x mean variance : RegionName) (N stride : Nat) (empty : ReductionPlan 0) :
    FP.Kernel₁ₓ₂ :=
  { online x mean variance N stride empty with io := twopassIO x mean variance N stride }

specification welford_equiv (N stride : Nat) (hN : 0 < N) (hBound : N ≤ limit)
    (empty : ReductionPlan 0) (R : Rules) :
    online "x" "mean" "variance" N stride empty ≡[R]
      twopass "x" "mean" "variance" N stride empty := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := online "x" "mean" "variance" N stride empty)
    (rhs := twopass "x" "mean" "variance" N stride empty)
    (structural := fun _ _ => False) rfl rfl
  exact WelfordFPContract.equivalent N stride hN hBound empty R

#print_fp_assumptions welford_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Examples.WelfordFPEquiv
