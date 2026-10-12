import bench.tritonbench_g.attention_forward_triton.Kernels
import bench.tritonbench_g.attention_forward_triton.Proofs.LegacyRounding
import VeriTile.Meta.StatementAudit

/-! The typed implementation and the historical transcription have the same
real projection. This is a source connection only: correctness of the erased
program still needs its execution, formula and memory-contract proof. -/
namespace VeriTile.Bench.TritonBenchG.AttentionForwardTriton.RealProjection
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 2400000
theorem original_matches_transcription
    (q k v qs ks out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage : Nat) :
    (TypedSource.original q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType =
    (attention_forward_triton_surface q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType := by
  simp only [TypedSource.original, attention_forward_triton_surface,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind]
  rfl
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionForwardTriton.RealProjection
