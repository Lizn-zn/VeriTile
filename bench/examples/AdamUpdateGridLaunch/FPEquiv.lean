import bench.examples.AdamUpdateGridLaunch.Kernels
import VeriTile.Triton.Float.GuardedRewrite
import VeriTile.Meta.StatementAudit

/-! AdamUpdateGridLaunch: contextual FP equivalence from the admitted scalar add_commute.
Only the final momentum addition is commuted. Its operands are the
computed diff * beta2 and grad, not merely the original inputs.
This is a per-program contextual rewrite, retaining both in-place stores.
The domain is evaluated at the actual rewrite site; experimental dimensions
are not restrictions on these symbolic kernels. -/

namespace VeriTile.Bench.Examples.AdamUpdateGridLaunchFPEquiv
open VeriTile.Bench.Examples.AdamUpdateGridLaunch.Kernels
open VeriTile Triton FP.Structural FP.GuardedRewrite
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Finite operands after the original common prefix. -/
def additionSite (lr wd beta1 beta2 : ℝ) (n B : Nat) : Site where
  before := (originalKernel lr wd beta1 beta2 n B).surfaceBody.take 16
  shape := [B]
  left := .mul .real .scalarR (.ref .real [B] "diff") (.const beta2)
  right := .ref .real [B] "grad"

def original (lr wd beta1 beta2 : ℝ) (n B : Nat) : Program :=
  ⟨originalKernel lr wd beta1 beta2 n B, [additionSite lr wd beta1 beta2 n B]⟩

def optimized (lr wd beta1 beta2 : ℝ) (n B : Nat) : Program :=
  ⟨optimizedKernel lr wd beta1 beta2 n B, [additionSite lr wd beta1 beta2 n B]⟩

/-- Under the finite-operand contract, every successful run is preserved,
including all registers and memory; failed surrounding executions also agree. -/
specification adam_update_equiv (lr wd beta1 beta2 : ℝ) (n B : Nat) (R : Rules) :
    original lr wd beta1 beta2 n B ≡[R] optimized lr wd beta1 beta2 n B := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := original lr wd beta1 beta2 n B) (rhs := optimized lr wd beta1 beta2 n B)
    (structural := fun _ _ => False) rfl rfl
  intro α _ M D hM s hd
  exact add_commute R M D hM s (additionSite lr wd beta1 beta2 n B) "exp_avg"
    ((originalKernel lr wd beta1 beta2 n B).surfaceBody.drop 17)
    (hd _ (by simp [original]))

#print_fp_assumptions adam_update_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.AdamUpdateGridLaunchFPEquiv
