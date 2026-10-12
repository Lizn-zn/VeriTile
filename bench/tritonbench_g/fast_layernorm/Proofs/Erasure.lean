import bench.tritonbench_g.fast_layernorm.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.Erasure
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

theorem forward_body (input weight bias : Storage) (swapped : Bool)
    (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) :
    (forward input weight bias swapped Y X W b r mu YR XR N B eps).eraseDType.toAlgKernel =
    (forward .fp32 .fp32 .fp32 swapped Y X W b r mu YR XR N B eps).eraseDType.toAlgKernel := by
  cases input <;> cases weight <;> cases bias <;> cases swapped
  all_goals simp only [forward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
  all_goals rfl

theorem backward_body (input weight bias : Storage) (swapped : Bool)
    (Y X W b r mu : RegionName) (YR XR N B : Nat) (eps : ℝ) :
    (backward input weight bias swapped Y X W b r mu YR XR N B eps).eraseDType.toAlgKernel =
    (backward .fp32 .fp32 .fp32 swapped Y X W b r mu YR XR N B eps).eraseDType.toAlgKernel := by
  cases input <;> cases weight <;> cases bias <;> cases swapped
  all_goals simp only [backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastLayernorm.Erasure
