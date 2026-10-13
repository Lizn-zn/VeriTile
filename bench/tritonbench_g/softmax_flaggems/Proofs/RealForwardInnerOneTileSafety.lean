import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerOneTileIO
import bench.tritonbench_g.softmax_flaggems.Proofs.RealTrace

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem source_safe (bounds : RegionBounds) (s : BlockState) (Y X : RegionName)
    (M N B : Nat) (hB : 0<B) (hr : s.pids 0<M)
    (hX : M*N≤bounds X) (hY : M*N≤bounds Y) :
    (softmax_kernel_inner .fp32 .fp32 Y X M N B Bool.true).eraseDType.toAlgKernel.TraceSafe bounds s := by
  unfold Kernel.TraceSafe
  simp [softmax_kernel_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    MaskOpt.Active, MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt, evalOp.eq_def,
    Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
    TileShape.axisDim, hB, NumericDType.add, NumericDType.mul, NumericDType.sub,
    NumericDType.div, ComparableDType.lt]
  constructor
  · intro i hi
    exact Nat.lt_of_lt_of_le (by simpa using RealAddress.non_inner_bound hr (show 0<1 by decide) hi) hX
  · intro i hi
    exact Nat.lt_of_lt_of_le (by simpa using RealAddress.non_inner_bound hr (show 0<1 by decide) hi) hY

theorem original_safe (input output : Storage) (bounds : RegionBounds) (s : BlockState)
    (Y X : RegionName) (M N B : Nat) (hB : 0<B) (hr : s.pids 0<M)
    (hX : M*N≤bounds X) (hY : M*N≤bounds Y) :
    (softmax_kernel_inner input output Y X M N B Bool.true).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [RealProjection.softmax_kernel_inner_erased]
  exact source_safe bounds s Y X M N B hB hr hX hY

theorem candidate_safe (input output : Storage) (bounds : RegionBounds) (s : BlockState)
    (Y X : RegionName) (M N B : Nat) (hB : 0<B) (hr : s.pids 0<M)
    (hX : M*N≤bounds X) (hY : M*N≤bounds Y) :
    (softmax_kernel_inner_candidate input output Y X M N B Bool.true).eraseDType.toAlgKernel.TraceSafe bounds s := by
  have h := original_safe input output bounds s Y X M N B hB hr hX hY
  unfold Kernel.TraceSafe at h ⊢
  rw [RealCandidate.forward_inner_original_body] at h
  rw [RealCandidate.forward_inner_body]
  simpa [Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,stepStmts,stepStmt,evalOp.eq_def] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile
