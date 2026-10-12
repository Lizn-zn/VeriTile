import bench.tritonbench_g.attention_fwd_triton2.Kernels
import bench.tritonbench_g.attention_forward_triton.Proofs.RealNormalization

/-! The full sources differ from attention_forward_triton in an explicit V
cast and bf16 output. Only the real projection erases those precision details.
The equalities below transfer complete real execution and memory contracts;
they make no FP equivalence claim between these two benchmark sources. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton2.RealProjection
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 2400000
section
variable (q k v qs ks out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage : Nat)

theorem original_real_source :
    (TypedSource.original q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType =
    (AttentionForwardTriton.TypedSource.original q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType := by
  simp only [TypedSource.original, AttentionForwardTriton.TypedSource.original,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind]
  rfl

theorem original_projection_supported :
    (TypedSource.original q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgorithm? =
      .ok (TypedSource.original q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgKernel := by
  rw [original_real_source]
  exact AttentionForwardTriton.RealNormalization.original_projection_supported ..

theorem reciprocal_real_source :
    (TypedSource.reciprocal q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType =
    (AttentionForwardTriton.TypedSource.reciprocal q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType := by
  simp only [TypedSource.reciprocal, AttentionForwardTriton.TypedSource.reciprocal,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind]
  rfl

theorem reciprocal_projection_supported :
    (TypedSource.reciprocal q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgorithm? =
      .ok (TypedSource.reciprocal q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgKernel := by
  rw [reciprocal_real_source]
  exact AttentionForwardTriton.RealNormalization.reciprocal_projection_supported ..

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton2.RealProjection
