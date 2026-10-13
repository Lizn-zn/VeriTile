import bench.tritonbench_g.softmax_optimize.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FullSource
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- The complete signed source has a mathematical algorithm projection. -/
theorem projects (input : Storage) (output : Accumulation) (v : Bool)
    (Y X : RegionName) (M N B : Nat) :
    (kernel input output v Y X M N B).toAlgorithm? =
      .ok (kernel input output v Y X M N B).toAlgKernel := by
  cases input <;> cases output <;> cases v <;> rfl

/-- Mathematical dtype erasure retains the signed source control and all stores. -/
def mathematical (v : Bool) (Y X : RegionName) (M N B : Nat) : Kernel :=
  (kernel .fp32 .fp32 v Y X M N B).eraseDType.toAlgKernel

theorem projection (input : Storage) (output : Accumulation) (v : Bool)
    (Y X : RegionName) (M N B : Nat) :
    (kernel input output v Y X M N B).eraseDType.toAlgKernel =
      mathematical v Y X M N B := by
  unfold mathematical
  simp only [ComputeKernel.eraseDType, projects, ComputeKernel.toAlgKernel_fromKernelBody]
  cases input <;> cases output <;> cases v
  all_goals simp only [kernel, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, FloatDType.toTileDType,
    ComputeDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.FullSource
