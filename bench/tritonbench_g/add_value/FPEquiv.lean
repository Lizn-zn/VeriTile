import bench.tritonbench_g.add_value.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Complete fp32 program equivalence of the two sources in Kernels.lean.
Swapping the operands also swaps scalar-right with scalar-left broadcasting.
The actual tile and scalar are checked after the original load prefix. -/

namespace VeriTile.Bench.TritonBenchG.AddValue.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.AddValue
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

/-- The loaded tile and the scalar interpreted at fp32 must be finite.
Masked lanes still execute arithmetic and are included. Neither positivity nor
nonzero is required; negative and zero values remain in the domain. -/
def additionSite (x_ptr output_ptr : RegionName) (N B : Nat) (value : ℝ) : Site :=
  .addScalar (Source.prefixBefore (puzzle1_kernel x_ptr output_ptr N B value).surfaceBody
    "output") B [] (.ref .real [B] "x") (.const value)

def original (x_ptr output_ptr : RegionName) (N B : Nat) (value : ℝ) : Program where
  kernel := puzzle1_kernel x_ptr output_ptr N B value
  domain := [additionSite x_ptr output_ptr N B value]

def commuted (x_ptr output_ptr : RegionName) (N B : Nat) (value : ℝ) : Program :=
  { original x_ptr output_ptr N B value with
    kernel := puzzle1_commuted_kernel x_ptr output_ptr N B value }

/-- Finite operands and admitted add_commute preserve all registers, memory and
execution failure for arbitrary symbolic sizes and scalar value. -/
specification add_value_commuted_equiv (x_ptr output_ptr : RegionName)
    (N B : Nat) (value : ℝ) (R : Rules) :
    original x_ptr output_ptr N B value ≡[R] commuted x_ptr output_ptr N B value := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions add_value_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.AddValue.FPEquiv
