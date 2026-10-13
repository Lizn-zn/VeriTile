import bench.tritonbench_g.kldiv_ops.Proofs.BackwardStep

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullBackward
open VeriTile.Triton

/-- The masked scatter changes no registers or launch metadata. -/
theorem scatter_state {B : Nat} (s : BlockState) (O : RegionName)
    (base N : Nat) (current : Int) (values : TileIndex [B] → ℝ) :
    (scatter s O base N current values).regs = s.regs ∧
    (scatter s O base N current values).pids = s.pids ∧
    (scatter s O base N current values).numPids = s.numPids ∧
    (scatter s O base N current values).undef = s.undef := by
  unfold scatter
  generalize TileShape.allIndices [B] = lanes
  induction lanes generalizing s with
  | nil => exact ⟨rfl, rfl, rfl, rfl⟩
  | cons i tail ih =>
    simp only [List.foldl_cons]
    by_cases hi : active N current i = Bool.true
    · simpa only [if_pos hi, BlockState.writeMem_regs, BlockState.writeMem_pids,
        BlockState.writeMem_numPids, BlockState.writeMem_undef] using
        ih (s.writeMem O (base + (offset current i).toNat) (values i))
    · simpa only [if_neg hi] using ih s

end VeriTile.Bench.TritonBenchG.KldivOps.FullBackward

#guard_msgs (drop info) in
#auditModuleAxioms
