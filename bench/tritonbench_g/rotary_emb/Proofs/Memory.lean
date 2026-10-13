import bench.tritonbench_g.rotary_emb.Proofs.Parameters

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton

/-- Independent memory transformation. All four values are functions of initial
memory; source correspondence separately justifies this under region separation. -/
noncomputable def memoryResult (p : Parameters) (s : BlockState)
    (Q K Cos Sin : RegionName) : BlockState :=
  Scatter.pair K (p.kAddress s) (p.expectedK s K Cos Sin) (p.activeK s)
    (Scatter.pair Q (p.qAddress s) (p.expectedQ s Q Cos Sin) (p.activeQ s) s)

theorem memoryResult_q (p : Parameters) (s : BlockState) (Q K Cos Sin : RegionName)
    (hQK : Q≠K) (hg : p.Geometry s) (odd : Bool) (i : p.Lane)
    (hi : p.activeQ s i) :
    (memoryResult p s Q K Cos Sin).readMem Q (p.qAddress s odd i) =
      p.expectedQ s Q Cos Sin odd i := by
  unfold memoryResult
  rw [Scatter.pair_read_other K Q hQK]
  exact Scatter.pair_readback Q _ _ _ s hg.1 odd i hi

theorem memoryResult_k (p : Parameters) (s : BlockState) (Q K Cos Sin : RegionName)
    (hg : p.Geometry s) (odd : Bool) (i : p.Lane) (hi : p.activeK s i) :
    (memoryResult p s Q K Cos Sin).readMem K (p.kAddress s odd i) =
      p.expectedK s K Cos Sin odd i := by
  exact Scatter.pair_readback K _ _ _ _ hg.2 odd i hi

theorem memoryResult_frame (p : Parameters) (s : BlockState)
    (Q K Cos Sin : RegionName) :
    BlockState.WriteWithin (p.footprint s Q K) s (memoryResult p s Q K Cos Sin) := by
  apply BlockState.writeWithin_trans
    (Scatter.pair_frame Q (p.qAddress s) (p.expectedQ s Q Cos Sin) (p.activeQ s) s ?_)
    (Scatter.pair_frame K (p.kAddress s) (p.expectedK s K Cos Sin) (p.activeK s) _ ?_)
  · intro odd i hi
    exact Or.inl ⟨rfl,i,odd,hi,rfl⟩
  · intro odd i hi
    exact Or.inr ⟨rfl,i,odd,hi,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
