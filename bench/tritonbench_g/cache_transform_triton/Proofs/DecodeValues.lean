import bench.tritonbench_g.cache_transform_triton.Proofs.RealExecution

/-! Active values of the complete BLOCK_SIZE × HIDDEN_DIM decoding kernel.
The source row comes from initial signed metadata. Address conversion occurs
after the entire signed offset; its nonnegative domain belongs to access safety.
No whole-kernel execution or output value is assumed. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.DecodeValues
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

theorem cos_output (cos sin : RegionName) (metadata : Region .int)
    (cosOut sinOut : RegionName) (rowStride hiddenStride H N B : Nat)
    (s : BlockState) (hRegion : cosOut ≠ sinOut)
    (hInj : Function.Injective (fun i : TileIndex [B,H] =>
      (s.pids 0 * B + i.1.val) * rowStride + i.2.1.val * hiddenStride))
    (i : TileIndex [B,H]) (hAct : s.pids 0 * B + i.1.val < N) :
    (RealExecution.decodeFinish cos sin metadata cosOut sinOut rowStride hiddenStride H N B
      Bool.false s).val.readMem cosOut ((s.pids 0 * B + i.1.val) * rowStride + i.2.1.val * hiddenStride) =
      s.readMem cos ((s.readMemValue .int metadata.cast (s.pids 0 * B + i.1.val) *
        (rowStride : Int) + (i.2.1.val * hiddenStride : Nat)).toNat) := by
  dsimp [RealExecution.decodeFinish]
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
    sinOut _ _ _ _ _ cosOut _ hRegion]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj i]
  simp [hAct, Tile.remap, Broadcast.leftIndex]

theorem sin_output (cos sin : RegionName) (metadata : Region .int)
    (cosOut sinOut : RegionName) (rowStride hiddenStride H N B : Nat)
    (s : BlockState)
    (hInj : Function.Injective (fun i : TileIndex [B,H] =>
      (s.pids 0 * B + i.1.val) * rowStride + i.2.1.val * hiddenStride))
    (i : TileIndex [B,H]) (hAct : s.pids 0 * B + i.1.val < N) :
    (RealExecution.decodeFinish cos sin metadata cosOut sinOut rowStride hiddenStride H N B
      Bool.false s).val.readMem sinOut ((s.pids 0 * B + i.1.val) * rowStride + i.2.1.val * hiddenStride) =
      s.readMem sin ((s.readMemValue .int metadata.cast (s.pids 0 * B + i.1.val) *
        (rowStride : Int) + (i.2.1.val * hiddenStride : Nat)).toNat) := by
  dsimp [RealExecution.decodeFinish]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj i]
  simp [hAct, Tile.remap, Broadcast.leftIndex]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.DecodeValues
