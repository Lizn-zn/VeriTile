import bench.tritonbench_g.bgmv_expand_slice.Kernels
import bench.tritonbench_g.bgmv_expand_slice.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Whole-program FP equivalence of the original and output-first sources in
Kernels. The only rewrite commutes the actual fp32 reduction result and loaded
old output inside ADD_INPUTS. Input/weight dtypes, CAST_TYPE and output dtype
remain explicit; the source keeps all casts, reduction, pointers and stores.
The domain checks finite actual addition operands along the taken sentinel
branch and each reached output-loop iteration. It is not an output equation,
a fixed experiment shape or a claim about concrete IEEE equality. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.Equivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

section
variable (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (cast : Bool) (out_dtype : FloatDType)
    (x w out : RegionName) (N K : Nat) (indices : Region .int)
    (sx sk sw0 swn swk sy syn slice BN BK splits : Nat) (even add : Bool)

def original : Program where
  kernel := TypedSource.original input weight cast out_dtype x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add
  domain := FPEquivProofs.requirementsFor (TypedSource.original input weight cast out_dtype x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add)

def outputFirst : Program :=
  { original input weight cast out_dtype x w out N K indices sx sk sw0 swn swk sy syn slice BN BK splits even add with
    kernel := TypedSource.outputFirst input weight cast out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add }

specification bgmv_output_first_equiv (R : FP.ScalarArithmetic.Rules) :
    original input weight cast out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add ≡[R]
    outputFirst input weight cast out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add := by
  exact FPEquivProofs.output_first_equiv input weight cast out_dtype x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add R

end
#print_fp_assumptions bgmv_output_first_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.Equivalence
