import bench.tritonbench_g.cache_transform_triton.Proofs.RealSafety
import VeriTile.Meta.Disprove

/-! The original prefill metadata load is unmasked. An allocation shorter
than N cannot satisfy actual trace safety, even for an inactive output token.
In particular, the host's unpadded three-entry metadata array cannot support
its next_power_of_2(3) = 4 load. This refutes memory safety, not the mathematical
cache-row formula under the explicit allocation bounds. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.HostSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

section
variable (cos sin : RegionName) (metadata : Region .int) (cosOut sinOut : RegionName)
    (rowStride hiddenStride total H N B : Nat) (bounds : RegionBounds)
    (commuted : Bool) (s : BlockState)

theorem metadata_bound (hN : 0 < N)
    (hs : (RealSource.prefill cos sin metadata cosOut sinOut rowStride hiddenStride total H N B commuted).TraceSafe bounds s) :
    ∀ i : Fin N, i.val < bounds metadata.cast := by
  cases commuted
  all_goals
    unfold Kernel.TraceSafe at hs
    simp [RealSource.prefill, TypedSource.prefill, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, Except.bind, bind,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.TraceSafeList, Stmt.TraceSafe,
      Op.SafeAt.eq_def, MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MaskOpt.Active,
      Op.PointerAddressesSafeOn, Op.MemorySafe, Option.bind, Option.map, tile_elementwise,
      Bool.and_eq_true, Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.select,
      Tile.reduceMaxInt, TileShape.discreteMaximum, TileShape.axisDim,
      TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul, NumericDType.sub,
      ComparableDType.lt, ComparableDType.le, hN, _root_.mul_comm] at hs
    exact hs.1

refutation prefill_unpadded_metadata_safe (hShort : bounds metadata.cast < N) :
    (RealSource.prefill cos sin metadata cosOut sinOut rowStride hiddenStride total H N B commuted).TraceSafe bounds s := by
  intro hs
  have h := metadata_bound cos sin metadata cosOut sinOut rowStride hiddenStride total H N B
    bounds commuted s (by omega) hs ⟨bounds metadata.cast, hShort⟩
  exact Nat.lt_irrefl _ h
end

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.HostSafety
