import bench.tritonbench_g.chunk_delta_fwd.Kernels
import bench.tritonbench_g.chunk_delta_fwd.Proofs.LegacyReal
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Precision erasure preserves the full original source. The historical real
surface already contains both loops and all stores; its old correctness proof
has a narrower regime. This bridge alone does not extend that theorem's scope. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealSource
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

section
variable (kd value initial final : TypedSource.Storage)
    (k v d v_new h initial_state final_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)

theorem original_body :
    (TypedSource.original kd value initial final k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE).eraseDType.toAlgKernel.body =
    (chunk_delta_rule_fwd_h_surface k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE).eraseDType.toAlgKernel.body := by
  cases value <;> cases final
  all_goals
    simp only [TypedSource.original, TypedSource.specialize, chunk_delta_rule_fwd_h_surface,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
      ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
      Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  all_goals simp only [VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float]
  all_goals cases kd <;> cases initial <;>
    simp [TypedSource.Storage.dtype, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealSource
