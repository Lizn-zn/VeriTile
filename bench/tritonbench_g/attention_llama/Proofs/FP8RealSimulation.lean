import bench.tritonbench_g.attention_llama.Proofs.FP8RealSource
import bench.tritonbench_g.attention_llama.Proofs.FP8RealMemory
import bench.tritonbench_g.attention_llama.Proofs.RealSafetyPrologue

/-! Execution correspondence between raw-byte inputs and their decoded real
view. The complete source prologue does not read K/V; its Q load therefore
commutes with decoding those two buffers. Loop load correspondence is supplied
by `FP8RealMemory.masked_decode`. The full loop induction remains separate. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSimulation
open VeriTile.Triton RealKernel FP8RealMemory
set_option maxHeartbeats 500000
set_option maxRecDepth 16000

/-- The actual eighteen-statement prologue preserves the input view for all
shapes and strides. Only the separation of Q from the byte buffers is needed;
the local bound used in the proof is derived from the address expression. -/
theorem preamble (Q K V : RegionName) (s : BlockState)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      NH H N BM BN BD : Nat) (hQK : Q ≠ K) (hQV : Q ≠ V) :
    (stepStmts (alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      NH H N BM BN BD) s).map (decoded K V) =
      stepStmts (alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
        NH H N BM BN BD) (decoded K V s) := by
  let limit := alBase s NH sqz sqh + (s.pids 0 * BM + BM) * sqm + BD * sqk + 1
  let bounds : RegionBounds := fun r => if r = Q then limit else 0
  apply assignments_outside K V bounds
    (by simp [bounds, Ne.symm hQK]) (by simp [bounds, Ne.symm hQV])
  · intro st h
    simp only [alPreLoopG, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals exact ⟨_, _, _, _, rfl⟩
  · apply RealSafetyPrologue.preamble_safe
    intro i _
    have hm : (s.pids 0 * BM + i.1.val) * sqm ≤ (s.pids 0 * BM + BM) * sqm :=
      Nat.mul_le_mul_right sqm (Nat.add_le_add_left (Nat.le_of_lt i.1.isLt) _)
    have hd : i.2.1.val * sqk ≤ BD * sqk := Nat.mul_le_mul_right sqk (Nat.le_of_lt i.2.1.isLt)
    simp only [bounds, ↓reduceIte, limit]
    omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSimulation
