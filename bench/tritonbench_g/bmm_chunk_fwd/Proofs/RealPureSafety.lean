import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealFinish
import VeriTile.Triton.Memory.TraceProofs

/-! Setup and pointer/offset preparation do not access memory. These lemmas
inspect the actual source fragments and cover only their register operations;
loads and stores require separate reached-address bounds. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealPureSafety
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 18000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem setup_safe (BM BN BK : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealSetup.code BM BN BK p) s := by
  apply TraceProofs.all_safe
  simp [RealSetup.code, RealProgram.realPart, RealProgram.parts, RealProgram.branch, RealProgram.source,
    TypedSource.specialize, Source.conditional?, Source.dynamicRange?, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind, eraseDType_float,
    List.forall_mem_cons, Stmt.TraceSafe, Op.SafeAt.eq_def]
  intro t
  simp only [evalOp.eq_def, Tile.scalar]
  intro _
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem prologue_safe (BM BN BK : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealSetup.prologue BM BN BK p) s := by
  apply TraceProofs.all_safe
  simp [RealSetup.prologue, RealProgram.realPart, RealProgram.branch, RealProgram.source,
    TypedSource.specialize, Source.conditional?, Source.dynamicRange?, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind, eraseDType_float,
    List.forall_mem_cons, Stmt.TraceSafe, Op.SafeAt.eq_def]
  intro t
  simp only [evalOp.eq_def, Tile.scalar]
  intro _
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem refresh_safe (BM BN : Nat) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealPostlude.refreshCode BM BN) s := by
  apply TraceProofs.all_safe
  simp [RealPostlude.refreshCode, BmmChunkBwd.RealFinish.refreshCode,
    BmmChunkBwd.RealResidual.rowOffsets, BmmChunkBwd.RealResidual.columnOffsets,
    Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem output_preparation_safe (BM BN : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealPostlude.outputCode BM BN p) s := by
  apply TraceProofs.all_safe
  simp [RealPostlude.outputCode, RealStore.basePointer, RealStore.pointers, Stmt.TraceSafe, Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealPureSafety
