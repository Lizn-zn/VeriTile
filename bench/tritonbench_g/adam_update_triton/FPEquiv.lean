import bench.tritonbench_g.adam_update_triton.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Complete fp32 program equivalence of the two sources in Kernels.lean.
The Program surface fixes default precision to fp32 and preserves the complete
state, both in-place stores and execution failure. Only the final momentum
addition changes. No sign, comparison, multiplication or subtraction identity
is assumed, and either outcome of each unchanged comparison is allowed. -/

namespace VeriTile.Bench.TritonBenchG.AdamUpdateTriton.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.AdamUpdateTriton
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Check the actual computed diff * beta2 and loaded grad for finiteness,
including evaluated masked lanes. The first exp_avg assignment loads momentum;
the second is this update. Positivity and nonzero are not required. -/
def additionSite (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat) : Site :=
  .add (Source.prefixBefore
    (update_fn_kernel p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B).surfaceBody
    "exp_avg" 1) [B]
    (.mul .real .scalarR (.ref .real [B] "diff") (.const beta2))
    (.ref .real [B] "grad")

def original (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat) : Program where
  kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B
  domain := [additionSite p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B]

def commuted (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat) : Program :=
  { original p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B with
    kernel := update_commuted_kernel p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B }

/-- Admitted add_commute preserves both outputs and all surrounding execution.
Unlike the independent output formulas in Correct.lean, this contextual result
also allows output-region aliasing: both programs perform the same two stores. -/
specification update_commuted_equiv (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat) (R : Rules) :
    original p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B ≡[R]
      commuted p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions update_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.AdamUpdateTriton.FPEquiv
