import bench.tritonbench_g.bgmv_expand_slice.Kernels
import bench.tritonbench_g.bgmv_expand_slice.Proofs.FP32
import VeriTile.Meta.StatementAudit

/-! Whole-program equivalence of the exact original and output-first sources
when the output storage is fp32. Input, weight, cast, masking and accumulation
choices remain symbolic. Products and sums retain their original dtypes; only
promotion against the loaded fp32 output makes the addition fp32. This theorem
does not discharge the arbitrary-output contract pending in FPEquiv.lean. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.FP32Equivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

section
variable (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (cast : Bool)
    (x w out : RegionName) (N K : Nat) (indices : Region .int)
    (sx sk sw0 swn swk sy syn slice BN BK splits : Nat) (even add : Bool)

def original : Program where
  kernel := TypedSource.original input weight cast .fp32 x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add
  domain := FP32Proofs.requirementsFor (TypedSource.original input weight cast .fp32 x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add)

def outputFirst : Program :=
  { original input weight cast x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add with
    kernel := TypedSource.outputFirst input weight cast .fp32 x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add }

specification bgmv_output_first_fp32_equiv (R : FP.ScalarArithmetic.Rules) :
    original input weight cast x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add ≡[R]
    outputFirst input weight cast x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add := by
  exact FP32Proofs.output_first_equiv input weight cast x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add R

end
#print_fp_assumptions bgmv_output_first_fp32_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.FP32Equivalence
