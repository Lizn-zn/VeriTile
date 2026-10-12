import bench.tritonbench_g.bgmv_shrink_kernel.Kernels
import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Whole-program FP equivalence of the original and scalar-left sources in
Kernels. Only the final fp32 multiplication is commuted. The domain checks
finite actual accumulator/scaling operands after the complete reached signed
branch and K loop. Input/weight and output precision remain explicit; casts,
products, reductions, masks, pointer arithmetic and both output tails stay in
context. No atomic law or reduction reassociation is assumed. This is admitted
FP equivalence, not concrete IEEE equality or a numerical-error guarantee. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.Equivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

section
variable (storage : TypedSource.Storage) (out_dtype : FloatDType)
    (x w out : RegionName) (N K : Nat) (indices : Region .int) (scaling : ℝ)
    (sx sk sw0 swn swk sy syn BN BK splits : Nat)

def original : Program where
  kernel := TypedSource.original storage out_dtype x w out N K indices scaling
    sx sk sw0 swn swk sy syn BN BK splits
  domain := FPEquivProofs.requirementsFor (TypedSource.original storage out_dtype x w out N K indices scaling
    sx sk sw0 swn swk sy syn BN BK splits) BN scaling

def scalarLeft : Program :=
  { original storage out_dtype x w out N K indices scaling sx sk sw0 swn swk sy syn BN BK splits with
    kernel := TypedSource.scalarLeft storage out_dtype x w out N K indices scaling
      sx sk sw0 swn swk sy syn BN BK splits }

specification bgmv_scaling_equiv (R : FP.ScalarArithmetic.Rules) :
    original storage out_dtype x w out N K indices scaling sx sk sw0 swn swk sy syn BN BK splits ≡[R]
    scalarLeft storage out_dtype x w out N K indices scaling sx sk sw0 swn swk sy syn BN BK splits := by
  exact FPEquivProofs.scaling_equiv storage out_dtype x w out N K indices scaling
    sx sk sw0 swn swk sy syn BN BK splits R

end
#print_fp_assumptions bgmv_scaling_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.Equivalence
