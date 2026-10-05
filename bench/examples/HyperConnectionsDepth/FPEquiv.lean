import bench.examples.HyperConnectionsDepth.Kernels
import VeriTile.Triton.Float.GuardedRewrite
import VeriTile.Meta.StatementAudit

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

end VeriTile.Bench.Examples.HyperConnectionsDepthFPEquiv
