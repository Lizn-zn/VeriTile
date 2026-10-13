import bench.tritonbench_g.var_len_copy.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.TypedSource
open VeriTile Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

 theorem projects (input output : Storage) (v : Bool)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    (kernel input output v os ol ns src dst B).toAlgorithm? =
      .ok (kernel input output v os ol ns src dst B).toAlgKernel := by
  cases input <;> cases output <;> cases v <;> rfl

/-- Storage erasure leaves the complete signed loop and both memory statements. -/
 theorem projection (input output : Storage) (v : Bool)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    (kernel input output v os ol ns src dst B).eraseDType.toAlgKernel =
      (kernel .fp32 .fp32 v os ol ns src dst B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, projects, ComputeKernel.toAlgKernel_fromKernelBody]
  cases input <;> cases output <;> cases v
  all_goals simp only [kernel, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, FloatDType.toTileDType,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  all_goals rfl

end VeriTile.Bench.TritonBenchG.VarLenCopy.TypedSource

#guard_msgs (drop info) in
#auditModuleAxioms
