import bench.tritonbench_g.fast_rope_embedding.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Erasure
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

theorem body (input cosine sine : Storage) (swapped backwards : Bool) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) :
    (rope input cosine sine swapped Q C S QR CR SR seq H heads B backwards).eraseDType.toAlgKernel=
    (rope .fp32 .fp32 .fp32 swapped Q C S QR CR SR seq H heads B backwards).eraseDType.toAlgKernel := by
  cases input <;> cases cosine <;> cases sine <;> cases swapped <;> cases backwards
  all_goals simp only [rope,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Erasure
