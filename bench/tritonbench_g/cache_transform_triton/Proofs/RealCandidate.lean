import bench.tritonbench_g.cache_transform_triton.Proofs.RealExecution

/-! Transport real contracts across the candidate's exact integer-index change.
The complete resulting state is preserved, including both stores and all typed
cells; no floating equivalence rule is imported into the real proof. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealCandidate
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

section
variable (cos sin : RegionName) (metadata : Region .int) (cosOut sinOut : RegionName)
    (rowStride hiddenStride total H N B : Nat)

theorem prefill_same_result (commuted : Bool) (s : BlockState) (hN : 0 < N) :
    (RealExecution.prefillFinish cos sin metadata cosOut sinOut rowStride hiddenStride total H N B commuted s hN).val =
      (RealExecution.prefillFinish cos sin metadata cosOut sinOut rowStride hiddenStride total H N B Bool.false s hN).val := by
  cases commuted <;> simp [RealExecution.prefillFinish, _root_.mul_comm, -Tile.scalar_eta]

theorem decode_same_result (commuted : Bool) (s : BlockState) :
    (RealExecution.decodeFinish cos sin metadata cosOut sinOut rowStride hiddenStride H N B commuted s).val =
      (RealExecution.decodeFinish cos sin metadata cosOut sinOut rowStride hiddenStride H N B Bool.false s).val := by
  cases commuted <;> simp [RealExecution.decodeFinish, _root_.mul_comm, Tile.remap, Broadcast.leftIndex, -Tile.scalar_eta]

theorem prefill_executes (commuted : Bool) (s : BlockState) (hN : 0 < N) :
    exec (RealSource.prefill cos sin metadata cosOut sinOut rowStride hiddenStride total H N B commuted) s =
      some (RealExecution.prefillFinish cos sin metadata cosOut sinOut rowStride hiddenStride total H N B Bool.false s hN).val := by
  rw [RealExecution.prefill_executes _ _ _ _ _ _ _ _ _ _ _ _ _ hN,
    prefill_same_result]

theorem decode_executes (commuted : Bool) (s : BlockState) :
    exec (RealSource.decode cos sin metadata cosOut sinOut rowStride hiddenStride H N B commuted) s =
      some (RealExecution.decodeFinish cos sin metadata cosOut sinOut rowStride hiddenStride H N B Bool.false s).val := by
  rw [RealExecution.decode_executes, decode_same_result]
end

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealCandidate
