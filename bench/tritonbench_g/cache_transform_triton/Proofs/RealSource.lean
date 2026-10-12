import bench.tritonbench_g.cache_transform_triton.Kernels
import VeriTile.Triton.Float.Erasure

/-! The complete real source is shared across floating storage choices.
Signed metadata, integer maximum, original masks and both stores survive
precision erasure. This source connection is not a correctness specification. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealSource
open VeriTile.Triton
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

section
variable (cos sin : RegionName) (metadata : Region .int) (cosOut sinOut : RegionName)
    (rowStride hiddenStride total H N B : Nat)

def prefill (commuted : Bool) : Kernel :=
  (TypedSource.prefill .fp32 .fp32 commuted cos sin metadata cosOut sinOut
    rowStride hiddenStride total H N B).toAlgKernel

def decode (commuted : Bool) : Kernel :=
  (TypedSource.decode .fp32 .fp32 commuted cos sin metadata cosOut sinOut
    rowStride hiddenStride H N B).toAlgKernel

theorem prefill_projection (cosType sinType : TypedSource.Storage) (commuted : Bool) :
    (TypedSource.prefill cosType sinType commuted cos sin metadata cosOut sinOut
      rowStride hiddenStride total H N B).eraseDType.toAlgorithm? =
      .ok (prefill cos sin metadata cosOut sinOut rowStride hiddenStride total H N B commuted) := by
  cases cosType <;> cases sinType <;> cases commuted
  all_goals
    simp only [prefill, TypedSource.prefill, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    try rfl

theorem decode_projection (cosType sinType : TypedSource.Storage) (commuted : Bool) :
    (TypedSource.decode cosType sinType commuted cos sin metadata cosOut sinOut
      rowStride hiddenStride H N B).eraseDType.toAlgorithm? =
      .ok (decode cos sin metadata cosOut sinOut rowStride hiddenStride H N B commuted) := by
  cases cosType <;> cases sinType <;> cases commuted
  all_goals
    simp only [decode, TypedSource.decode, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
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
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealSource
