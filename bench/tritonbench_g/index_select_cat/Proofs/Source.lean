import bench.tritonbench_g.index_select_cat.Kernels
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.IndexSelectCat
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def mathematicalSource (v : Bool) : Source :=
  fun out idx inp n c s0 s1 BI BC =>
    (source .fp32 .fp32 .int64 v out idx inp n c s0 s1 BI BC).eraseDType

theorem source_projects (input output : Storage) (index : IndexStorage) (v : Bool)
    (out idx inp : RegionName) (n c s0 s1 BI BC : Nat) :
    (source input output index v out idx inp n c s0 s1 BI BC).toAlgorithm? =
      .ok (source input output index v out idx inp n c s0 s1 BI BC).toAlgKernel := by
  cases input <;> cases output <;> cases index <;> cases v <;> rfl

theorem source_erasure (input output : Storage) (index : IndexStorage) (v : Bool)
    (out idx inp : RegionName) (n c s0 s1 BI BC : Nat) :
    (source input output index v out idx inp n c s0 s1 BI BC).eraseDType.toAlgKernel =
      (mathematicalSource v out idx inp n c s0 s1 BI BC).toAlgKernel := by
  cases input <;> cases output <;> cases index <;> cases v
  all_goals simp only [source, mathematicalSource, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.IndexSelectCat
