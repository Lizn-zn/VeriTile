import bench.tritonbench_g.max_reduction.Proofs.DimensionReal
import bench.tritonbench_g.max_reduction.Proofs.DimensionStore

namespace VeriTile.Bench.TritonBenchG.MaxReduction
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

/-- The full mixed real/int execution changes only its two active output windows. -/
theorem dimension_frame (inp ov oi : RegionName) (M N K BM BN : Nat)
    (hN : 0 < BN) (s t : BlockState)
    (he : exec (mathDimension Bool.false inp ov oi M N K BM BN) s = some t)
    (r : RegionName) (o : Nat)
    (hv : ∀ i : Fin BM, s.pids 0 * BM + i.val < M →
      r ≠ ov ∨ o ≠ maxKernelOutOffset s K BM i)
    (hi : ∀ i : Fin BM, s.pids 0 * BM + i.val < M →
      r ≠ oi ∨ o ≠ maxKernelOutOffset s K BM i) :
    t.mem r o = s.mem r o := by
  simp [exec, mathDimension, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, Option.bind,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.argMaxDrop, Tile.argBestDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, hN] at he
  rw [← he]
  rw [DimensionStore.int_frame _ _ _ _ _ _ r o (fun j _ ha => hi j.1 ha)]
  rw [DimensionStore.real_frame _ _ _ _ _ _ r o (fun j _ ha => hv j.1 ha)]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MaxReduction
