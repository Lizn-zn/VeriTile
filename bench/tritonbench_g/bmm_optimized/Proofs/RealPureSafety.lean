import bench.tritonbench_g.bmm_optimized.Proofs.RealStore
import VeriTile.Triton.Memory.TraceProofs

/-! Register-only fragments of the exact source never access memory. Loads
and stores are handled separately with their actual masks and address bounds. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealPureSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

theorem masks_safe (BM BN BK : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealMasks.code BM BN BK p) s := by
  apply TraceProofs.all_safe
  cases hm : p.divM <;> cases hn : p.divN <;> cases hk : p.divK <;>
    simp [RealMasks.code, RealMasks.aCode, RealMasks.bCode, hm, hn, hk,
      Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem tail_safe (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealLoads.tailCode swapped BM BN BK p) s := by
  apply TraceProofs.all_safe
  cases swapped <;> simp [RealLoads.tailCode, RealLoads.update, Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem store_masks_safe (BM BN : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealStore.maskCode BM BN p.divM p.divN) s := by
  apply TraceProofs.all_safe
  cases hm : p.divM <;> cases hn : p.divN <;>
    simp [RealStore.maskCode, hm, hn, Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem setup_safe (BM BN BK : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealSetup.code BM BN BK p) s := by
  apply TraceProofs.all_safe
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk
  all_goals
    simp [RealSetup.code, RealProgram.realPart, RealProgram.parts, RealProgram.source,
      TypedSource.specialize, Source.dynamicRange?, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float,
      Stmt.TraceSafe, Op.SafeAt.eq_def]
    repeat' first
      | intro _
      | constructor
      | (apply TraceProofs.all_safe; simp [Stmt.TraceSafe, Op.SafeAt.eq_def])
      | split

/-- The complete projected source belongs to the shared flattening bridge's fragment. -/
theorem flatten_ok (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    (RealProgram.kernel swapped BM BN BK p).toAlgKernel.FlattenOk := by
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk <;> cases swapped
  all_goals
    simp [RealProgram.kernel, RealProgram.source, TypedSource.specialize,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
      ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?_map_alg,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float,
      Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealPureSafety
