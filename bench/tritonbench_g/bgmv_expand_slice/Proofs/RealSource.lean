import bench.tritonbench_g.bgmv_expand_slice.Kernels
import VeriTile.Triton.Float.Erasure

/-! The complete typed sources share one real source for each compile-time cast
choice and candidate direction. This eliminates dtype cases from the future
real IO proof; it does not assert the mathematical GEMV formula or memory safety.
FP equivalence imports no theorem from this real-projection layer. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealSource
open VeriTile.Triton
set_option maxHeartbeats 2200000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

section
variable (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (CAST_TYPE swapped : Bool) (out_dtype : FloatDType)
    (x w out : RegionName) (N K : Nat) (indices : Region .int)
    (sx sk sw0 swn swk sy syn slice BN BK splits : Nat) (EVEN_K ADD_INPUTS : Bool)

def source := TypedSource.specialize input weight CAST_TYPE swapped out_dtype x w out N K indices
  sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS

theorem source_projection :
    (source input weight CAST_TYPE swapped out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS).toAlgorithm? =
    .ok (source input weight CAST_TYPE swapped out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS).toAlgKernel := by
  cases input <;> cases weight <;> cases CAST_TYPE <;> cases swapped <;> cases out_dtype <;> rfl

/-- Erasing precision preserves all source flags, masks, casts' register writes,
loops and addresses. Dtype-specialized multiplication/casts give the same real
body; no FP relation between different dtype configurations is asserted. -/
theorem storage_erasure :
    (source input weight CAST_TYPE swapped out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS).eraseDType =
    (source .fp16 .fp16 CAST_TYPE swapped .real x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits EVEN_K ADD_INPUTS).eraseDType := by
  cases input <;> cases weight <;> cases CAST_TYPE <;> cases swapped <;> cases out_dtype
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
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealSource
