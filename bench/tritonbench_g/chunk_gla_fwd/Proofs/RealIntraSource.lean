import bench.tritonbench_g.chunk_gla_fwd.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Both complete intra-subblock sources project successfully. Numerical
storage erasure gives a common real program, while preserving the early return,
column loop, separate matrix/flat layouts and unspecified matrix padding.
This bridge is not itself a proof of the independent real output formula. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraSource
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

variable (swapped : Bool) (query key gate : TypedIntra.Storage)
    (q k g A : RegionName)
    (s_k_h s_k_t : Nat)
    (scale : ℝ) (T K BT BC BK : Nat)

theorem projects :
    (TypedIntra.source query key gate swapped q k g A
      s_k_h s_k_t scale T K BT BC BK).toAlgorithm? =
    .ok (TypedIntra.source query key gate swapped q k g A
      s_k_h s_k_t scale T K BT BC BK).toAlgKernel := by
  cases swapped <;> cases query <;> cases key <;> cases gate <;> rfl

theorem storage_erasure :
    (TypedIntra.source query key gate swapped q k g A
      s_k_h s_k_t scale T K BT BC BK).eraseDType.toAlgKernel.body =
    (TypedIntra.source .fp32 .fp32 .fp32 swapped q k g A
      s_k_h s_k_t scale T K BT BC BK).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases query <;> cases key <;> cases gate
  all_goals simp only [TypedIntra.source, Bool.false_eq_true, ↓reduceIte,
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

  all_goals simp [VeriTile.Triton.eraseDType, NumericDType.eraseDType]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraSource
