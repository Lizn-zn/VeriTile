import bench.examples.HyperConnectionsDepth.Kernels
import VeriTile.Triton.Float.GuardedRewrite
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit

/- The matrix specifications below cover symbolic dimensions and every
normalization count. They preserve the entire execution under their pointwise
operand domains; matrix operations keep the same opaque backend. -/

/-! HyperConnectionsDepth: contextual FP equivalence from the admitted scalar add_commute.
The scalar, zero-iteration source is unchanged. The actual residual
and exp-weighted branch product must be finite at the final addition.
The domain is evaluated at the actual rewrite site; experimental dimensions
are not restrictions on these symbolic kernels. -/

namespace VeriTile.Bench.Examples.HyperConnectionsDepthFPEquiv
open VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels
open VeriTile Triton FP.Structural FP.GuardedRewrite
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Finite operands after the original common prefix. -/
def additionSite (tau : ℝ) : Site where
  before := (originalKernel tau).surfaceBody.take 5
  shape := []
  left := .ref .real [] "res_mix"
  right := .ref .real [] "branch_mix"

def original (tau : ℝ) : Program :=
  ⟨originalKernel tau, [additionSite tau]⟩

def optimized (tau : ℝ) : Program :=
  ⟨optimizedKernel tau, [additionSite tau]⟩

/-- Under the finite-operand contract, every successful run is preserved,
including all registers and memory; failed surrounding executions also agree. -/
specification mhc_depth_equiv (tau : ℝ) (R : Rules) :
    original tau ≡[R] optimized tau := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := original tau) (rhs := optimized tau)
    (structural := fun _ _ => False) rfl rfl
  intro α _ M D hM s hd
  exact add_commute R M D hM s (additionSite tau) "out"
    ((originalKernel tau).surfaceBody.drop 6)
    (hd _ (by simp [original]))

#print_fp_assumptions mhc_depth_equiv
#guard_msgs (drop info) in
#auditModuleAxioms


/-- The final two matrix operands must be finite entrywise, after every
Sinkhorn iteration and the unchanged matrix multiplication have executed. -/
def matrixAdditionSite (S T D numIters : Nat) (tau : ℝ) : FP.ProfiledRewrite.Site :=
  .add ((matrixOriginal S T D numIters tau).surfaceBody.take 17) [S, D]
    (.ref .real [S, D] "res_mix") (.ref .real [S, D] "branch_mix")

def matrixOriginalProgram (S T D numIters : Nat) (tau : ℝ) : FP.ProfiledRewrite.Program :=
  ⟨matrixOriginal S T D numIters tau, [matrixAdditionSite S T D numIters tau]⟩

def matrixOptimizedProgram (S T D numIters : Nat) (tau : ℝ) : FP.ProfiledRewrite.Program :=
  ⟨matrixOptimized "res_mix" "branch_out" "h_post" "out" S T D numIters tau,
    [matrixAdditionSite S T D numIters tau]⟩

/-- Contextual equivalence for every matrix dimension and iteration count. -/
specification mhc_depth_matrix_equiv (S T D numIters : Nat) (tau : ℝ) (R : Rules) :
    matrixOriginalProgram S T D numIters tau ≡[R] matrixOptimizedProgram S T D numIters tau := by
  apply Spec.FloatingPoint.ofNumerical (lhs := matrixOriginalProgram S T D numIters tau)
    (rhs := matrixOptimizedProgram S T D numIters tau) (structural := fun _ _ => False) rfl rfl
  intro α _ M domain hM s hd
  exact FP.ProfiledRewrite.add_commute R M domain hM s
    ((matrixOriginal S T D numIters tau).surfaceBody.take 17)
    ((matrixOriginal S T D numIters tau).surfaceBody.drop 18) "out" _ _
    (hd _ (by simp [matrixOriginalProgram, matrixAdditionSite]))

#print_fp_assumptions mhc_depth_matrix_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.HyperConnectionsDepthFPEquiv
