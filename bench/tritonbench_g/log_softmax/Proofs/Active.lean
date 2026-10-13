import bench.tritonbench_g.log_softmax.Proofs.Address

namespace VeriTile.Bench.TritonBenchG.LogSoftmax.SourceReadback
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Active-lane readback needs no injectivity hypothesis on masked-off columns. -/
theorem forward_active (output_ptr input_ptr : RegionName)
    (M N K R C : Nat) (hK : 0 < K) (s s' : BlockState)
    (hExec : exec (log_softmax_kernel output_ptr input_ptr M N K R C) s = some s')
    (idx : TileIndex [R, C]) (hActive : active (s.pids 0) M N R idx) :
    s'.readMem output_ptr (outOffset (s.pids 0) (s.pids 1) N K R idx) =
      logSoftmaxSpec s input_ptr (s.pids 0) (s.pids 1) M N K R C idx := by
  have hBN : 0 < C := Nat.lt_of_le_of_lt (Nat.zero_le _) idx.2.1.isLt
  simp [exec, log_softmax_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
        Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceMaxKeep,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, hBN] at hExec
  rw [← hExec]
  simp only [outOffset]
  have hNoCollision : ∀ j : TileIndex [R, C],
      (s.pids 0 * R + j.1.val < M ∧ j.2.1.val < N) →
      (s.pids 0 * R + j.1.val) * N * K + j.2.1.val * K + s.pids 1 =
        (s.pids 0 * R + idx.1.val) * N * K + idx.2.1.val * K + s.pids 1 → j = idx := by
    intro j hj heq
    exact SourceAddress.active_injective hK hj hActive heq
  rw [BlockState.scatter_readback_prop_masked_nd_of_true (shape := [R, C]) _
    (fun i ↦ (s.pids 0 * R + i.1.val) * N * K + i.2.1.val * K + s.pids 1) _
    (fun i ↦ s.pids 0 * R + i.1.val < M ∧ i.2.1.val < N) idx hActive hNoCollision]
  rcases hActive with ⟨hM, hN⟩
  simp [active, hM, hN, logSoftmaxSpec, logSoftmaxOf,
        logSoftmaxInputTile, outOffset, Tile.reduceMax, Tile.reduceMaxDrop,
        Tile.reduceMaxKeep, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceSumKeep, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.replaceAxisIndex,
        TileShape.dropInsertedIndex, NumericDType.sub, hBN]
  congr

/-- Backward active readback retains the full unspecified-padding reduction. -/
theorem backward_active (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K R C : Nat) (hK : 0 < K) (s s' : BlockState)
    (hExec : exec (log_softmax_backward_kernel out_ptr out_grad_ptr in_grad_ptr
      M N K R C) s = some s')
    (idx : TileIndex [R, C]) (hActive : active (s.pids 0) M N R idx) :
    s'.readMem in_grad_ptr (outOffset (s.pids 0) (s.pids 1) N K R idx) =
      logSoftmaxBackwardSpec s out_ptr out_grad_ptr M N K R C idx := by
  have hBN : 0 < C := Nat.lt_of_le_of_lt (Nat.zero_le _) idx.2.1.isLt
  simp [exec, log_softmax_backward_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, hBN] at hExec
  rw [← hExec]
  simp only [outOffset]
  have hNoCollision : ∀ j : TileIndex [R, C],
      (s.pids 0 * R + j.1.val < M ∧ j.2.1.val < N) →
      (s.pids 0 * R + j.1.val) * N * K + j.2.1.val * K + s.pids 1 =
        (s.pids 0 * R + idx.1.val) * N * K + idx.2.1.val * K + s.pids 1 → j = idx := by
    intro j hj heq
    exact SourceAddress.active_injective hK hj hActive heq
  rw [BlockState.scatter_readback_prop_masked_nd_of_true (shape := [R, C]) _
    (fun i ↦ (s.pids 0 * R + i.1.val) * N * K + i.2.1.val * K + s.pids 1) _
    (fun i ↦ s.pids 0 * R + i.1.val < M ∧ i.2.1.val < N) idx hActive hNoCollision]
  rcases hActive with ⟨hM, hN⟩
  simp [active, hM, hN, logSoftmaxBackwardSpec, logSoftmaxBackwardOf,
        logSoftmaxBackwardOutTile, logSoftmaxBackwardGradTile, outOffset,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
        TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, TileShape.replaceAxisIndex,
        TileShape.dropInsertedIndex,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        NumericDType.sub, NumericDType.mul, hBN]
  congr

end VeriTile.Bench.TritonBenchG.LogSoftmax.SourceReadback
