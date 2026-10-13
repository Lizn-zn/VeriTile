import bench.tritonbench_g.rotary_emb.Proofs.KPhase
import bench.tritonbench_g.rotary_emb.Proofs.Memory

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton

/-- The concrete intermediate-state construction and the independent memory
specification agree without assuming collision freedom. Collision freedom is
needed separately for the four lane readback conclusions. -/
theorem final_state_memory (p : Parameters) (s : BlockState)
    (Q K Cos Sin : RegionName) (hsep : RegionsSeparate Q K Cos Sin) :
    (kStoreState p s Q K Cos Sin).mem = (memoryResult p s Q K Cos Sin).mem := by
  have hv : p.expectedK (qStoreState p s Q Cos Sin) K Cos Sin =
      p.expectedK s K Cos Sin := by
    funext odd i
    exact k_expected_initial p s Q K Cos Sin hsep odd i
  unfold kStoreState memoryResult
  rw [hv]
  apply Scatter.pair_mem_congr
  exact q_store_state_mem p s Q Cos Sin

theorem final_state_q (p : Parameters) (s : BlockState)
    (Q K Cos Sin : RegionName) (hsep : RegionsSeparate Q K Cos Sin)
    (hGeometry : p.Geometry s) (odd : Bool) (i : p.Lane) (hi : p.activeQ s i) :
    (kStoreState p s Q K Cos Sin).readMem Q (p.qAddress s odd i) =
      p.expectedQ s Q Cos Sin odd i := by
  have hm := final_state_memory p s Q K Cos Sin hsep
  have hr := memoryResult_q p s Q K Cos Sin hsep.1 hGeometry odd i hi
  simpa only [BlockState.readMem, hm] using hr

theorem final_state_k (p : Parameters) (s : BlockState)
    (Q K Cos Sin : RegionName) (hsep : RegionsSeparate Q K Cos Sin)
    (hGeometry : p.Geometry s) (odd : Bool) (i : p.Lane) (hi : p.activeK s i) :
    (kStoreState p s Q K Cos Sin).readMem K (p.kAddress s odd i) =
      p.expectedK s K Cos Sin odd i := by
  have hm := final_state_memory p s Q K Cos Sin hsep
  have hr := memoryResult_k p s Q K Cos Sin hGeometry odd i hi
  simpa only [BlockState.readMem, hm] using hr

theorem final_state_frame (p : Parameters) (s : BlockState)
    (Q K Cos Sin : RegionName) (hsep : RegionsSeparate Q K Cos Sin) :
    BlockState.WriteWithin (p.footprint s Q K) s (kStoreState p s Q K Cos Sin) := by
  have hm := final_state_memory p s Q K Cos Sin hsep
  simpa only [BlockState.WriteWithin, BlockState.AgreesOutside, hm] using
    memoryResult_frame p s Q K Cos Sin

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
