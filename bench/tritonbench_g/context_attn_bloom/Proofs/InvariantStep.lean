import bench.tritonbench_g.context_attn_bloom.Proofs.IterationExecution
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.InvariantStep
open VeriTile.Triton Math Program RealSetup
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 3000

theorem step (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial u : BlockState) (c : Nat) (hBN : 0 < p.BN)
    (hwin : (c+1)*p.BN ≤ LoopState.size p r initial)
    (hp : 0 ≤ (inputs initial r).prompt (initial.pids 0))
    (hs : Invariant.Static p r initial u) (hd : Invariant.Dynamic p r initial scale c u)
    (hi : u.regs .int [] "start_n" = some (Tile.scalar ((c*p.BN : Nat) : Int))) :
    ∃ next, stepStmts (LoopProgram.body p r swapped scale) u = some next ∧
      Invariant.Static p r initial next ∧ Invariant.Dynamic p r initial scale (c+1) next := by
  obtain ⟨next,maximum,he,hrm,hstatic,hregisters⟩ :=
    IterationExecution.executes p r swapped scale initial u c hBN hs hd hi
  have hn := LoopState.numeric_correct p r initial scale c hBN hwin hp maximum hrm
  dsimp only at hregisters
  rw [hn] at hregisters
  exact ⟨next,he,hstatic,⟨hregisters.1,hregisters.2.1,hregisters.2.2⟩⟩

theorem static_index (p : Layout) (r : Regions) (initial u : BlockState) (sn : Int)
    (h : Invariant.Static p r initial u) :
    Invariant.Static p r initial (u.setReg "start_n" .int [] (Tile.scalar sn)) := by
  have hm := h.memory
  cases h
  constructor
  · funext rg off
    simpa only [BlockState.setReg_mem] using congrFun (congrFun hm rg) off
  all_goals simp_all

theorem dynamic_index (p : Layout) (r : Regions) (initial u : BlockState) (scale : ℝ) (c : Nat) (sn : Int)
    (h : Invariant.Dynamic p r initial scale c u) :
    Invariant.Dynamic p r initial scale c (u.setReg "start_n" .int [] (Tile.scalar sn)) := by
  cases h
  constructor <;> simp_all

theorem indexed_step (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial u : BlockState) (c : Nat) (hBN : 0 < p.BN)
    (hwin : (c+1)*p.BN ≤ LoopState.size p r initial)
    (hp : 0 ≤ (inputs initial r).prompt (initial.pids 0))
    (hs : Invariant.Static p r initial u) (hd : Invariant.Dynamic p r initial scale c u) :
    ∃ next,
      stepStmts (LoopProgram.body p r swapped scale)
        (u.setReg "start_n" .int [] (Tile.scalar ((c*p.BN : Nat) : Int))) = some next ∧
      Invariant.Static p r initial next ∧ Invariant.Dynamic p r initial scale (c+1) next := by
  apply step p r swapped scale initial _ c hBN hwin hp
  · exact static_index p r initial u _ hs
  · exact dynamic_index p r initial u scale c _ hd
  · simp

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.InvariantStep
