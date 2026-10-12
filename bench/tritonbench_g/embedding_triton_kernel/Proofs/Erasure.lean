import bench.tritonbench_g.embedding_triton_kernel.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Erasure
open VeriTile.Triton TypedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

def source (swapped : Bool) : sourceType :=
  match swapped with
  | Bool.false => embeddingSource%[tl.float64, original]
  | Bool.true => embeddingSource%[tl.float64, indexCommuted]

theorem body (storage : Storage) (swapped : Bool)
    (weight ids out : RegionName) (first last : Int) (sw so n h D N NN : Nat) :
    (TypedSource.source storage swapped weight ids out first last sw so n h D N NN).eraseDType.toAlgKernel.body =
      (source swapped weight ids out first last sw so n h D N NN).toAlgKernel.body := by
  cases storage <;> cases swapped
  all_goals simp only [TypedSource.source,source,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    Except.bind,bind]
  all_goals simp only [Op.eraseDType.eq_def,VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Erasure
