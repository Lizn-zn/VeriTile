import bench.tritonbench_g.softmax_triton2.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! FP equivalence of the original and reciprocal sources in Kernels.lean.
The fp32 profile preserves opaque reductions and `tl.exp`. The numerical rule
applies to the computed numerator and row sum, after the original prefix.
Masks, strides, stores and surrounding execution failures are preserved. -/

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton2.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.SoftmaxTriton2
open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

/-- All computed numerator lanes must be finite; the scalar denominator must
be finite and nonzero. Masked lanes still participate in the arithmetic.
No positivity, experimental shape or input distribution is added. -/
def divisionSite (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) : Site :=
  .divScalar (Source.prefixBefore
    (softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
      n_cols BLOCK_SIZE).surfaceBody "softmax_output") BLOCK_SIZE []
    (.ref .real [BLOCK_SIZE] "numerator") (.ref .real [] "denominator")

def original (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) : Program where
  kernel := softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
    n_cols BLOCK_SIZE
  domain := [divisionSite output_ptr input_ptr input_row_stride output_row_stride
    n_cols BLOCK_SIZE]

def reciprocal (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) : Program :=
  { original output_ptr input_ptr input_row_stride output_row_stride n_cols BLOCK_SIZE with
    kernel := softmax_reciprocal_kernel output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE }

specification softmax_reciprocal_equiv (R : Rules)
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    original output_ptr input_ptr input_row_stride output_row_stride n_cols BLOCK_SIZE ≡[R]
      reciprocal output_ptr input_ptr input_row_stride output_row_stride n_cols BLOCK_SIZE := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions softmax_reciprocal_equiv

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.SoftmaxTriton2.FPEquiv
