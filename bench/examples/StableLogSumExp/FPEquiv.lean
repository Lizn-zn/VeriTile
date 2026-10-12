import bench.examples.StableLogSumExp.Kernels
import bench.examples.StableLogSumExp.Proofs.FP
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

/-!
FP equivalence of optimizedLSEKernel and the unchanged directLSEKernel in
Kernels.lean. Both use tl.log and libdevice.exp and store a scalar bf16 result.
The conditional optimized preserves the product-log fallback near one and
the log-exp fallback near zero and outside the cancellation interval.

The contract records the finite/positive/nonzero intermediate operands used
by the scalar derivation, an arbitrary shared reduction schedule, and support
for fp32 comparisons. Branch outcomes are unrestricted. The row length is
symbolic. The specification includes successful execution and a memory frame.

The older stableLSEKernel omits both fallbacks. Its unconditional log_mul and
log_exp candidates are not admitted. Proofs/Unconditional.lean refutes the old
general FP specification with a logical countermodel of the selected atoms.
-/

noncomputable section
namespace VeriTile.Bench.Examples.StableLogSumExpFPEquiv
open VeriTile.Bench.Examples.StableLogSumExp.Kernels
open VeriTile Triton
open StableLogSumExpFPContract
open scoped VeriTile.Spec
open StableLogSumExpFPExecution
open SoftmaxStableFPContract (rowExpressions center rowPlan)
open _root_.VeriTile.Triton.FP.Equational (Schedules)
open FP.GuardExpression

def directIO (x y : RegionName) (B : Nat) : KernelIO₁ where
  kernel := directLSEKernel x y B
  inp := x
  out := y
  Bin := B
  Bout := 1
  read := fun pid => pid * B
  write := fun pid => pid

def optimizedIO (x y : RegionName) (B : Nat) : KernelIO₁ :=
  (directIO x y B).withKernel (optimizedLSEKernel x y B)

/-- Finite inputs, row maximum, exponentials and sum-tree intermediates;
exp(max) is positive and nonzero, its reciprocal is finite, and the shifted
sum is finite and positive with a finite logarithm. Checks also cover the
zero intermediates used by the scalar derivation. `plans` chooses the shared
addition tree; input values themselves need not be positive. -/
def requirements (x : RegionName) (B : Nat) (plans : Schedules) : Condition :=
  let M := FP.GuardExpression.algebra MemoryInput
  let xs := rowExpressions x B
  FP.LogSumExpShift.checks M xs (center M xs) (rowPlan plans B).tree

/-- Reference kernel under the common FP contract. Both programs use these
numerical-domain checks and comparison requirements; only the optimized
source uses the comparisons, and either Boolean result is allowed. -/
def original (x y : RegionName) (B : Nat) : FP.Kernel₁ where
  io := directIO x y B
  profile := FP.fp32
  domain := requirements x B
  comparisons := [.lt .fp32 .real, .le .fp32 .real]

/-- Same IO layout, precision and preconditions, with the optimized source. -/
def optimized (x y : RegionName) (B : Nat) : FP.Kernel₁ :=
  { original x y B with io := optimizedIO x y B }

/-- Actual source-to-source equivalence from admitted scalar atoms. -/
specification logsumexp_equiv (B : Nat) (hB : 0 < B) (R : FP.LogSumExp.Rules) :
    optimized "x" "y" B ≡[R] original "x" "y" B := by
  apply Spec.FloatingPoint.ofNumerical (lhs := optimized "x" "y" B)
    (rhs := original "x" "y" B) (structural := fun _ _ => False) rfl rfl
  exact candidate_equivalent B hB R

#print_fp_assumptions logsumexp_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.StableLogSumExpFPEquiv
