import bench.tritonbench_g.bgmv_shrink_kernel.Kernels
import VeriTile.Triton.Float.Erasure

/-! Precision erasure of the complete source, including signed metadata and
both output branches. This is a source connection for the real proof, not a
GEMV correctness or memory-safety theorem. FP equivalence does not import it. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSource
open VeriTile.Triton
set_option maxHeartbeats 2200000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

section
variable (storage : TypedSource.Storage) (swapped : Bool) (out_dtype : FloatDType)
    (x w out : RegionName) (N K : Nat) (indices : Region .int) (scaling : ℝ)
    (sx sk sw0 swn swk sy syn BN BK splits : Nat)

def source := TypedSource.specialize storage swapped out_dtype x w out N K indices scaling
  sx sk sw0 swn swk sy syn BN BK splits

theorem source_projection :
    (source storage swapped out_dtype x w out N K indices scaling
      sx sk sw0 swn swk sy syn BN BK splits).toAlgorithm? =
    .ok (source storage swapped out_dtype x w out N K indices scaling
      sx sk sw0 swn swk sy syn BN BK splits).toAlgKernel := by
  cases storage <;> cases swapped <;> rfl

/-- Precision erasure preserves the complete signed-sentinel and split-K
program structure. This does not assert FP equivalence across storage types. -/
theorem storage_erasure :
    (source storage swapped out_dtype x w out N K indices scaling
      sx sk sw0 swn swk sy syn BN BK splits).eraseDType =
    (source .fp16 swapped .real x w out N K indices scaling
      sx sk sw0 swn swk sy syn BN BK splits).eraseDType := by
  cases storage <;> cases swapped <;> cases out_dtype
  all_goals
    simp only [source, TypedSource.specialize, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    try rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSource
