import bench.tritonbench_g.chunk_retention.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Precision erasure of the complete forward state source; no padding,
store order or loop bound is changed. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHSource
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def source (swapped : Bool) := if swapped then forwardHDotLeft else forwardH
variable (swapped : Bool) (dot : Storage)
    (dtype out_dtype : FloatDType) (k v h h0 ht : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t H T K V BT BK BV NT : Nat)
    (hasSeed storeFinal : Bool)

theorem projects :
    (source swapped dot dtype out_dtype k v h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t H T K V BT BK BV NT hasSeed storeFinal).toAlgorithm? =
    .ok (source swapped dot dtype out_dtype k v h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t H T K V BT BK BV NT hasSeed storeFinal).toAlgKernel := by
  cases swapped <;> cases dot <;> rfl

theorem storage_erasure :
    (source swapped dot dtype out_dtype k v h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t H T K V BT BK BV NT hasSeed storeFinal).eraseDType.toAlgKernel.body =
    (source swapped .fp32 .real .real k v h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t H T K V BT BK BV NT hasSeed storeFinal).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases dot
  all_goals simp only [source, Bool.false_eq_true, ↓reduceIte, forwardH,
    forwardHDotLeft, forwardHSource, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
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

end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHSource

/-! Precision erasure of the complete backward state source; no padding,
store order or loop bound is changed. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhSource
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def source (swapped : Bool) := if swapped then backwardHDotLeft else backwardH
variable (swapped : Bool) (dot : Storage)
    (out_dtype : FloatDType) (q do_ dh v : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t H T K V BT NT : Nat)
    (scale : ℝ)

theorem projects :
    (source swapped dot out_dtype q do_ dh v
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT NT).toAlgorithm? =
    .ok (source swapped dot out_dtype q do_ dh v
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT NT).toAlgKernel := by
  cases swapped <;> cases dot <;> rfl

theorem storage_erasure :
    (source swapped dot out_dtype q do_ dh v
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT NT).eraseDType.toAlgKernel.body =
    (source swapped .fp32 .real q do_ dh v
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT NT).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases dot
  all_goals simp only [source, Bool.false_eq_true, ↓reduceIte, backwardH,
    backwardHDotLeft, backwardHSource, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  all_goals simp only [VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float]
  all_goals cases out_dtype <;>
    simp [VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType]

end VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhSource

/-! Precision erasure of the complete backward three-output source; no padding,
store order or loop bound is changed. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVSource
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def source (swapped : Bool) := if swapped then backwardQKVDotLeft else backwardQKV
variable (swapped : Bool) (dot : Storage)
    (q k v h do_ dh dq dk dv : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t H T K V BT BK BV NT : Nat)
    (scale : ℝ)

theorem projects :
    (source swapped dot q k v h do_ dh dq dk dv
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT BK BV NT).toAlgorithm? =
    .ok (source swapped dot q k v h do_ dh dq dk dv
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT BK BV NT).toAlgKernel := by
  cases swapped <;> cases dot <;> rfl

theorem storage_erasure :
    (source swapped dot q k v h do_ dh dq dk dv
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT BK BV NT).eraseDType.toAlgKernel.body =
    (source swapped .fp32 q k v h do_ dh dq dk dv
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT BK BV NT).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases dot
  all_goals simp only [source, Bool.false_eq_true, ↓reduceIte, backwardQKV,
    backwardQKVDotLeft, backwardQKVSource, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  all_goals simp only [VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float]
  all_goals simp [VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType]

end VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVSource

namespace VeriTile.Bench.TritonBenchG.ChunkRetention.OutputSource
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

variable (swapped : Bool) (query value : Storage)
    (out_dtype : FloatDType) (q k v h o : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t : Nat)
    (scale : ℝ) (H T K V BT BK BV : Nat)

theorem projects :
    (forwardOSource query value swapped out_dtype q k v h o
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT BK BV).toAlgorithm? =
    .ok (forwardOSource query value swapped out_dtype q k v h o
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT BK BV).toAlgKernel := by
  cases swapped <;> cases query <;> cases value <;> rfl

theorem storage_erasure :
    (forwardOSource query value swapped out_dtype q k v h o
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT BK BV).eraseDType.toAlgKernel.body =
    (forwardOSource .fp32 .fp32 swapped .real q k v h o
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t scale H T K V BT BK BV).eraseDType.toAlgKernel.body := by
  cases swapped <;> cases query <;> cases value
  all_goals simp only [forwardOSource, Bool.false_eq_true, ↓reduceIte,
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
  all_goals cases out_dtype <;>
    simp [VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType]

end VeriTile.Bench.TritonBenchG.ChunkRetention.OutputSource

#guard_msgs (drop info) in
#auditModuleAxioms
