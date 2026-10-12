import bench.tritonbench_g.chunk_bwd_dqkg.Kernels
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.LegacyReal
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! The complete typed source projects to the real body used by the loop and
formula proofs. This bridge erases precision only for real correctness. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealSource
open VeriTile.Triton
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

section
variable (qk value hist dhist : TypedSource.Storage) (swapped : Bool)
    (q k v h g do_ dh dq dk dg : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat)

def source := TypedSource.specialize qk value hist dhist swapped
  q k v h g do_ dh dq dk dg s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT

theorem projects :
    (source qk value hist dhist swapped q k v h g do_ dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT).toAlgorithm? =
    .ok (source qk value hist dhist swapped q k v h g do_ dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT).toAlgKernel := by
  cases hist <;> cases dhist <;> cases swapped <;> rfl

set_option linter.unusedSimpArgs false in
/-- All four independent storage choices retain the complete real body,
including both flattened whole-tile sums and the actual history mask. -/
theorem real_body_matches :
    (source qk value hist dhist Bool.false q k v h g do_ dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT).eraseDType.toAlgKernel.body =
    (chunk_bwd_dqkg_surface q k v h g do_ dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT).toAlgKernel.body := by
  cases hist <;> cases dhist
  all_goals
    simp only [source, TypedSource.specialize, TypedSource.Storage.dtype,
        VeriTile.Triton.eraseDType, Region.cast, ComputeKernel.surfaceBody,
        ComputeKernel.eraseDType, chunk_bwd_dqkg_surface,
        ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
        ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?, ComputeDType.eraseDType, List.map_cons, List.map_nil,
        Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
        Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
        NumericDType.eraseDType, ComparableDType.eraseDType,
        IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    simp only [List.cons.injEq, and_true]
    repeat' apply And.intro
    all_goals simp [FloatDType.toTileDType, Op.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
      VeriTile.Triton.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, List.map_cons, List.map_nil]
    all_goals cases qk <;> cases value
    all_goals simp [TypedSource.Storage.dtype, Op.eraseDType.eq_def,
      FloatDType.toTileDType, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType]
end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealSource
