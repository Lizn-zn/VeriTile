import bench.examples.TritonBenchVectorAddition.Kernels
import VeriTile.Triton.Float.GuardedRewrite
import VeriTile.Meta.StatementAudit

/-! TritonBenchVectorAddition: contextual FP equivalence from the admitted scalar add_commute.
The loaded x/y operands must be finite, including masked undefined lanes
whose arithmetic is evaluated; the store mask is preserved.
The domain is evaluated at the actual rewrite site; experimental dimensions
are not restrictions on these symbolic kernels. -/

namespace VeriTile.Bench.Examples.TritonBenchVectorAdditionFPEquiv
open VeriTile.Bench.Examples.TritonBenchVectorAddition.Kernels
open VeriTile Triton FP.Structural FP.GuardedRewrite
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Finite operands after the original common prefix. -/
def additionSite (nElements blockSize : Nat) : Site where
  before := (originalKernel nElements blockSize).surfaceBody.take 6
  shape := [blockSize]
  left := .ref .real [blockSize] "x"
  right := .ref .real [blockSize] "y"

def original (nElements blockSize : Nat) : Program :=
  ⟨originalKernel nElements blockSize, [additionSite nElements blockSize]⟩

def optimized (nElements blockSize : Nat) : Program :=
  ⟨optimizedKernel nElements blockSize, [additionSite nElements blockSize]⟩

/-- Under the finite-operand contract, every successful run is preserved,
including all registers and memory; failed surrounding executions also agree. -/
specification vector_addition_equiv (nElements blockSize : Nat) (R : Rules) :
    original nElements blockSize ≡[R] optimized nElements blockSize := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := original nElements blockSize) (rhs := optimized nElements blockSize)
    (structural := fun _ _ => False) rfl rfl
  intro α _ M D hM s hd
  exact add_commute R M D hM s (additionSite nElements blockSize) "output"
    ((originalKernel nElements blockSize).surfaceBody.drop 7)
    (hd _ (by simp [original]))

#print_fp_assumptions vector_addition_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.TritonBenchVectorAdditionFPEquiv
