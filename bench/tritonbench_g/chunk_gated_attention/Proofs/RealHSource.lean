import bench.tritonbench_g.chunk_gated_attention.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Erasure of the complete H source, including optional seed addition,
pre-update history, both gate branches and final store. Only precision changes;
unspecified padding and full-chunk endpoint addresses are preserved. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHSource
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def source (swapped : Bool) := if swapped then TypedSource.hSumLeft else TypedSource.hOriginal
variable (swapped : Bool) (dot gate history : TypedSource.Storage)
    (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT : Nat)
    (gateK hasSeed storeFinal : Bool)

theorem projects :
    (source swapped dot gate history dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal).toAlgorithm? =
    .ok (source swapped dot gate history dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal).toAlgKernel := by
  cases swapped <;> cases dot <;> cases gate <;> cases history <;> rfl

theorem storage_erasure :
    (source swapped dot gate history dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal).eraseDType.toAlgKernel.body =
    (source swapped .fp32 .fp32 .fp32 .real .real k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases dot <;> cases gate <;> cases history
  all_goals simp only [source, Bool.false_eq_true, ↓reduceIte, TypedSource.hOriginal,
    TypedSource.hSumLeft, TypedSource.hSource, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
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
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHSource
