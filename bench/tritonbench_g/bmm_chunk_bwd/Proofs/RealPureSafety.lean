import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealFinish
import VeriTile.Triton.Memory.TraceProofs

/-! Setup and pointer/offset preparation do not access memory. These lemmas
inspect the actual source fragments and cover only their register operations;
loads and stores require separate reached-address bounds. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealPureSafety
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 18000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem setup_safe (BM BN BCS : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealSetup.code BM BN BCS p) s := by
  apply TraceProofs.all_safe
  simp [RealSetup.code, RealProgram.realPart, RealProgram.parts, RealProgram.source,
    TypedSource.specialize, Source.dynamicRange?, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind, eraseDType_float,
    List.forall_mem_cons, Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem refresh_safe (BM BN : Nat) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealFinish.refreshCode BM BN) s := by
  apply TraceProofs.all_safe
  simp [RealFinish.refreshCode, RealResidual.rowOffsets, RealResidual.columnOffsets,
    Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem output_preparation_safe (BM BN : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealFinish.outputCode BM BN p) s := by
  apply TraceProofs.all_safe
  simp [RealFinish.outputCode, RealResidual.basePointer, RealResidual.pointers, Stmt.TraceSafe, Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealPureSafety
