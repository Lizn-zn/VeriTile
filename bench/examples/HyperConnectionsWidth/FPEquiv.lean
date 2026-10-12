import bench.examples.HyperConnectionsWidth.Kernels
import VeriTile.Triton.Float.GuardedRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/- The matrix specifications below cover symbolic dimensions and every
normalization count. They preserve the entire execution under their pointwise
operand domains; matrix operations keep the same opaque backend. -/

/-! Scalar, zero-iteration mHC width: commute the two products independently.
The actual tl.exp(h / tau) operands and residual must be finite at each site.
No exponential identity is used; tl.exp remains an opaque call. -/

namespace VeriTile.Bench.Examples.HyperConnectionsWidthFPEquiv
open VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels
open VeriTile Triton FP.Structural FP.GuardedRewrite
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Check the computed weight, including the division and exponential. -/
def residualSite (tau : ℝ) : Site where
  before := Source.prefixBefore (originalKernel tau).surfaceBody "res_mix"
  shape := []
  left := .exp (.div .real .nil (.ref .real [] "h_res") (.const tau))
  right := .ref .real [] "residual"

/-- The second rewrite follows the first one in the intermediate program. -/
def branchSite (tau : ℝ) : Site where
  before := Source.prefixBefore (middleKernel tau).surfaceBody "branch_in"
  shape := []
  left := .exp (.div .real .nil (.ref .real [] "h_pre") (.const tau))
  right := .ref .real [] "residual"

def original (tau : ℝ) : Program where
  kernel := originalKernel tau
  domain := [residualSite tau, branchSite tau]

def optimized (tau : ℝ) : Program :=
  { original tau with kernel := optimizedKernel tau }

specification mhc_width_equiv (tau : ℝ) (R : Rules) :
    original tau ≡[R] optimized tau := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions mhc_width_equiv
#guard_msgs (drop info) in
#auditModuleAxioms


/-- The division checks are on the loaded logit tiles and the actual fp32
literal tau. Its value must be finite and nonzero. -/
def matrixResidualSite (S T D numIters : Nat) (tau : ℝ) : FP.ProfiledRewrite.Site :=
  .div (FP.ProfiledRewrite.prefixBefore (matrixOriginal S T D numIters tau).surfaceBody "z") S [S]
    (.ref .real [S, S] "h_res_logits") tau

def matrixBranchSite (S T D numIters : Nat) (tau : ℝ) : FP.ProfiledRewrite.Site :=
  .div (FP.ProfiledRewrite.prefixBefore
    (matrixMiddle "res" "h_res" "h_pre" "res_mix" "branch_in" S T D numIters tau).surfaceBody "z" 1)
    S [T] (.ref .real [S, T] "h_pre_logits") tau

def matrixOriginalProgram (S T D numIters : Nat) (tau : ℝ) : FP.ProfiledRewrite.Program where
  kernel := matrixOriginal S T D numIters tau
  domain := [matrixResidualSite S T D numIters tau, matrixBranchSite S T D numIters tau]

def matrixOptimizedProgram (S T D numIters : Nat) (tau : ℝ) : FP.ProfiledRewrite.Program :=
  { matrixOriginalProgram S T D numIters tau with
    kernel := matrixOptimized "res" "h_res" "h_pre" "res_mix" "branch_in" S T D numIters tau }

/-- All dimensions and both full Sinkhorn loops remain in the source.
The two matrix products retain their original operand order and backend. -/
specification mhc_width_matrix_equiv (S T D numIters : Nat) (tau : ℝ) (R : Rules) :
    matrixOriginalProgram S T D numIters tau ≡[R] matrixOptimizedProgram S T D numIters tau := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions mhc_width_matrix_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.HyperConnectionsWidthFPEquiv
