import bench.tritonbench_g.chunk_gla_fwd.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Both complete output sources project successfully. Erasing only numerical
precision gives a common real program for all three query/value/gate storage
choices and arbitrary history/output dtypes. The candidate order, unspecified
padding and each dot's distinct options are preserved. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputSource
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

variable (swapped : Bool) (query value gate : TypedOutput.Storage)
    (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat)

theorem projects :
    (TypedOutput.source query value gate swapped dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).toAlgorithm? =
    .ok (TypedOutput.source query value gate swapped dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).toAlgKernel := by
  cases swapped <;> cases query <;> cases value <;> cases gate <;> rfl

theorem storage_erasure :
    (TypedOutput.source query value gate swapped dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel.body =
    (TypedOutput.source .fp32 .fp32 .fp32 swapped .real .real q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases query <;> cases value <;> cases gate
  all_goals simp only [TypedOutput.source, Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  all_goals simp only [VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float]
  all_goals cases dtype <;> cases out_dtype <;>
    simp [VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputSource
