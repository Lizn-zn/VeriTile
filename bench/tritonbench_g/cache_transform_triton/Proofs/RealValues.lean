import bench.tritonbench_g.cache_transform_triton.Proofs.RealExecution
/-! Active-value readback for the complete original prefill kernel. Inputs
are the initial memory values and row selection is the independent signed
formula. Successful execution comes from RealExecution; flat memory, access
bounds, candidate transport and decoding readbacks are composed in RealFlat. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealValues
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

theorem prefill_cos (cos sin : RegionName) (metadata : Region .int)
    (cosOut sinOut : RegionName) (rowStride hiddenStride total H N B : Nat)
    (s : BlockState) (hN : 0 < N) (hRegion : cosOut ≠ sinOut)
    (hInj : Function.Injective (fun i : Fin H =>
      (s.pids 0 * B + s.pids 1) * rowStride + i.val * hiddenStride))
    (i : Fin H) (hAct : s.pids 0 * B + s.pids 1 < total) :
    (RealExecution.prefillFinish cos sin metadata cosOut sinOut rowStride hiddenStride total H N B
      Bool.false s hN).val.readMem cosOut ((s.pids 0 * B + s.pids 1) * rowStride + i.val * hiddenStride) =
      s.readMem cos ((RealMath.relativeRow hN (s.pids 0 * B + s.pids 1)
        (fun j => s.readMemValue .int metadata.cast j.val) * (rowStride : Int) +
          (i.val * hiddenStride : Nat)).toNat) := by
  have hRaw : Function.Injective (fun j : TileIndex [H] =>
      (s.pids 0 * B + s.pids 1) * rowStride + j.1.val * hiddenStride) := by
    intro a b h
    have hab := hInj h
    cases a; cases b
    simp only at hab
    cases hab
    rfl
  dsimp [RealExecution.prefillFinish]
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
    sinOut _ _ _ _ _ cosOut _ hRegion]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRaw (i, PUnit.unit)]
  simp [hAct, RealMath.relativeRow, RealMath.selectedEnd, TileShape.insertAxisIndex]
  rfl

theorem prefill_sin (cos sin : RegionName) (metadata : Region .int)
    (cosOut sinOut : RegionName) (rowStride hiddenStride total H N B : Nat)
    (s : BlockState) (hN : 0 < N)
    (hInj : Function.Injective (fun i : Fin H =>
      (s.pids 0 * B + s.pids 1) * rowStride + i.val * hiddenStride))
    (i : Fin H) (hAct : s.pids 0 * B + s.pids 1 < total) :
    (RealExecution.prefillFinish cos sin metadata cosOut sinOut rowStride hiddenStride total H N B
      Bool.false s hN).val.readMem sinOut ((s.pids 0 * B + s.pids 1) * rowStride + i.val * hiddenStride) =
      s.readMem sin ((RealMath.relativeRow hN (s.pids 0 * B + s.pids 1)
        (fun j => s.readMemValue .int metadata.cast j.val) * (rowStride : Int) +
          (i.val * hiddenStride : Nat)).toNat) := by
  have hRaw : Function.Injective (fun j : TileIndex [H] =>
      (s.pids 0 * B + s.pids 1) * rowStride + j.1.val * hiddenStride) := by
    intro a b h
    have hab := hInj h
    cases a; cases b
    simp only at hab
    cases hab
    rfl
  dsimp [RealExecution.prefillFinish]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRaw (i, PUnit.unit)]
  simp [hAct, RealMath.relativeRow, RealMath.selectedEnd, TileShape.insertAxisIndex]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealValues
