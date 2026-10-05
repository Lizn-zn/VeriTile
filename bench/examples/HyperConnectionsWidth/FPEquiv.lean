import bench.examples.HyperConnectionsWidth.Kernels
import VeriTile.Triton.Float.GuardedRewrite
import VeriTile.Meta.StatementAudit

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
  before := (originalKernel tau).surfaceBody.take 3
  shape := []
  left := .exp (.div .real .nil (.ref .real [] "h_res") (.const tau))
  right := .ref .real [] "residual"

/-- The second rewrite follows the first one in the intermediate program. -/
def branchSite (tau : ℝ) : Site where
  before := (middleKernel tau).surfaceBody.take 5
  shape := []
  left := .exp (.div .real .nil (.ref .real [] "h_pre") (.const tau))
  right := .ref .real [] "residual"

def original (tau : ℝ) : Program :=
  ⟨originalKernel tau, [residualSite tau, branchSite tau]⟩

def optimized (tau : ℝ) : Program :=
  ⟨optimizedKernel tau, [residualSite tau, branchSite tau]⟩

specification mhc_width_equiv (tau : ℝ) (R : Rules) :
    original tau ≡[R] optimized tau := by
  apply Spec.FloatingPoint.ofNumerical (lhs := original tau) (rhs := optimized tau)
    (structural := fun _ _ => False) rfl rfl
  intro α _ M D hM s hd
  have first : FP.Structural.exec M (originalKernel tau) s =
      FP.Structural.exec M (middleKernel tau) s :=
    mul_commute R M D hM s (residualSite tau) "res_mix"
      ((originalKernel tau).surfaceBody.drop 4) (hd _ (by simp [original]))
  have second : FP.Structural.exec M (middleKernel tau) s =
      FP.Structural.exec M (optimizedKernel tau) s :=
    mul_commute R M D hM s (branchSite tau) "branch_in"
      ((middleKernel tau).surfaceBody.drop 6) (hd _ (by simp [original]))
  exact first.trans second

#print_fp_assumptions mhc_width_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.HyperConnectionsWidthFPEquiv
