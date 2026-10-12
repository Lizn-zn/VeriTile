import bench.tritonbench_g.fast_ce_loss.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.Erasure
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

/-- The whole typed forward source has the same real projection for every
supported storage choice; no FP equality follows from this erasure. -/
theorem forward_body (storage : Storage) (cap scale swapped : Bool)
    (X Y Z : RegionName) (L : Region .int) (V R B : Nat) (SC SL : ℝ) :
    (forward storage cap scale swapped X Y Z L V R B SC SL).eraseDType.toAlgKernel =
    (forward .fp32 cap scale swapped X Y Z L V R B SC SL).eraseDType.toAlgKernel := by
  cases storage <;> cases cap <;> cases scale <;> cases swapped
  all_goals simp only [forward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
  all_goals rfl

/-- The whole typed chunked source has the same real projection for every
supported storage choice; no FP equality follows from this erasure. -/
theorem chunked_body (storage : Storage) (cap scale swapped : Bool)
    (X Y Z : RegionName) (L : Region .int) (V C R B : Nat) (SC SL : ℝ) :
    (chunked storage cap scale swapped X Y Z L V C R B SC SL).eraseDType.toAlgKernel =
    (chunked .fp32 cap scale swapped X Y Z L V C R B SC SL).eraseDType.toAlgKernel := by
  cases storage <;> cases cap <;> cases scale <;> cases swapped
  all_goals simp only [chunked,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
  all_goals rfl

/-- The whole typed backward source has the same real projection for every
supported storage choice; no FP equality follows from this erasure. -/
theorem backward_body (storage : Storage) (cap scale swapped : Bool)
    (X Y Z : RegionName) (L : Region .int) (V R DR B : Nat) (SC SL : ℝ) :
    (backward storage cap scale swapped X Y Z L V R DR B SC SL).eraseDType.toAlgKernel =
    (backward .fp32 cap scale swapped X Y Z L V R DR B SC SL).eraseDType.toAlgKernel := by
  cases storage <;> cases cap <;> cases scale <;> cases swapped
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
end VeriTile.Bench.TritonBenchG.FastCeLoss.Erasure
