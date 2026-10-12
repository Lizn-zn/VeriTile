import bench.tritonbench_g.attention_kernel.Kernels
import bench.tritonbench_g.attention_kernel.Proofs.Legacy
import VeriTile.Meta.StatementAudit

/-! Checked correspondence of the typed reference's mathematical projection
with the historical source after erasure. This relates full ASTs, including bias
addressing and arithmetic. It does not transfer the historical IO theorem by
itself: that theorem executes a cast-bearing AST, whereas the new contract must
execute the erased AST. Real execution/IO transport is a separate obligation. -/
namespace VeriTile.Bench.TritonBenchG.AttentionKernel.LegacyProjection
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 300000
set_option linter.unusedSimpArgs false
private theorem erased_projection_supported (k : ComputeKernel)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgorithm? = .ok k.eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, h]
  simp

section
variable (storage : TypedSource.Storage)
    (q k v bias out : RegionName) (sm_scale : ℝ)
    (sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm
      Z H N P BIAS B0_SIZE BD BM BN : Nat)

theorem original_real_source :
    (TypedSource.original storage q k v bias out sm_scale
      sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm
      Z H N P BIAS B0_SIZE BD BM BN).eraseDType =
    (attention_kernel_fwd_kernel_aligned_surface q k v bias out sm_scale
      sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm
      Z H N P BIAS B0_SIZE BD BM BN .fp16).eraseDType := by
  have ht : (TypedSource.original storage q k v bias out sm_scale sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm Z H N P BIAS B0_SIZE BD BM BN).toAlgorithm? =
      .ok (TypedSource.original storage q k v bias out sm_scale sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm Z H N P BIAS B0_SIZE BD BM BN).toAlgKernel := by
    cases storage <;> rfl
  have ho : (attention_kernel_fwd_kernel_aligned_surface q k v bias out sm_scale sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm Z H N P BIAS B0_SIZE BD BM BN .fp16).toAlgorithm? =
      .ok (attention_kernel_fwd_kernel_aligned_surface q k v bias out sm_scale sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm Z H N P BIAS B0_SIZE BD BM BN .fp16).toAlgKernel := rfl
  simp only [ComputeKernel.eraseDType, ht, ho, Kernel.eraseDType]
  congr 1
  all_goals cases storage
  all_goals first | rfl | skip
  all_goals
    simp only [TypedSource.original, TypedSource.FP16.original, TypedSource.BF16.original,
      attention_kernel_fwd_kernel_aligned_surface, ComputeKernel.toAlgKernel,
      ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
      MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
      Except.bind, bind, List.map_cons, List.map_nil]
    simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil]
    rfl

theorem original_projection_supported :
    (TypedSource.original storage q k v bias out sm_scale
      sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm
      Z H N P BIAS B0_SIZE BD BM BN).eraseDType.toAlgorithm? =
      .ok (TypedSource.original storage q k v bias out sm_scale
      sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm
      Z H N P BIAS B0_SIZE BD BM BN).eraseDType.toAlgKernel := by
  exact erased_projection_supported _ (by cases storage <;> rfl)

theorem reciprocal_projection_supported :
    (TypedSource.reciprocal storage q k v bias out sm_scale
      sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm
      Z H N P BIAS B0_SIZE BD BM BN).eraseDType.toAlgorithm? =
      .ok (TypedSource.reciprocal storage q k v bias out sm_scale
      sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm
      Z H N P BIAS B0_SIZE BD BM BN).eraseDType.toAlgKernel := by
  exact erased_projection_supported _ (by cases storage <;> rfl)

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionKernel.LegacyProjection
