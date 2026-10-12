import bench.tritonbench_g.chunk_gate_recurrence.Kernels
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.LegacyReal
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Both complete typed originals project to the historical real source bodies.
This only connects source representations: the historical step/store proofs
still assume materialized carries and do not establish the complete loops. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealSource
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

section
variable (source gate seed : TypedSource.Storage) (S D O last_kv : RegionName)
    (H N K V BK BV : Nat) (hasSeed : Bool)

theorem forward_body :
    (TypedSource.forward source gate seed S D O last_kv H N K V BK BV hasSeed).eraseDType.toAlgKernel.body =
    (chunk_gate_recurrence_fwd_surface S D O last_kv H N K V BK BV hasSeed).eraseDType.toAlgKernel.body := by
  cases source <;> cases gate
  all_goals
    simp only [TypedSource.forward, TypedSource.specializeForward, chunk_gate_recurrence_fwd_surface,
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
  all_goals cases seed <;>
    simp [TypedSource.Storage.dtype, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
end

section
variable (source gate incoming : TypedSource.Storage) (S D DI DG DL DS : RegionName)
    (H N K V BK BV : Nat)

theorem backward_body :
    (TypedSource.backward source gate incoming S D DI DG DL DS H N K V BK BV).eraseDType.toAlgKernel.body =
    (chunk_gate_recurrence_bwd_surface S D DI DG DL DS H N K V BK BV).eraseDType.toAlgKernel.body := by
  cases gate <;> cases incoming
  all_goals
    simp only [TypedSource.backward, TypedSource.specializeBackward, chunk_gate_recurrence_bwd_surface,
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
  all_goals cases source <;>
    simp [TypedSource.Storage.dtype, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealSource
