import bench.tritonbench_g.chunk_cumsum_vector.Kernels
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.LegacyReal
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Erasure for real correctness preserves the complete loop, triangular dot, carry, branch and stores. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealSource
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def source (swapped : Bool) := if swapped then TypedSource.sumLeft else TypedSource.original

section
variable (swapped : Bool) (dtype out_dtype : FloatDType) (s o : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)

theorem projects : (source swapped dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS).toAlgorithm? =
    .ok (source swapped dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS).toAlgKernel := by
  cases swapped <;> rfl

theorem storage_erasure :
    (source swapped dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS).eraseDType =
    (source swapped .real .real s o s_s_h s_s_t s_s_d T S BT BS).eraseDType := by
  cases swapped
  all_goals simp only [source, Bool.false_eq_true, ↓reduceIte, TypedSource.original,
    TypedSource.sumLeft, ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  all_goals simp only [VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float]
  all_goals cases dtype <;> cases out_dtype
  all_goals exact ⟨rfl, rfl, HEq.rfl, HEq.rfl⟩

theorem original_body :
    (source Bool.false dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS).eraseDType.toAlgKernel.body =
    (chunk_cumsum_vector_surface s o s_s_h s_s_t s_s_d T S BT BS).toAlgKernel.body := by
  rw [storage_erasure]
  simp only [source, Bool.false_eq_true, ↓reduceIte, TypedSource.original,
    chunk_cumsum_vector_surface, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, List.map_cons, List.map_nil,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    Except.bind, bind, eraseDType_float]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  rfl
end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealSource
