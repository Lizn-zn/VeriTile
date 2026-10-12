import bench.tritonbench_g.context_attn_nopad.Proofs.IterationExecution
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.InvariantStep
open VeriTile.Triton Formula Program
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 3000

theorem step (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial u : BlockState) (c : Nat) (hBN : 0 < p.BN)
    (hshift : 0 ≤ (inputs initial r).start (launch initial).batch+(c*p.BN : Nat))
    (hs : Invariant.Static p r initial u) (hd : Invariant.Dynamic p r initial scale c u)
    (hi : u.regs .int [] "start_n" = some (Tile.scalar ((c*p.BN : Nat) : Int))) :
    ∃ next, stepStmts (LoopProgram.body p r swapped scale) u = some next ∧
      Invariant.Static p r initial next ∧ Invariant.Dynamic p r initial scale (c+1) next := by
  obtain ⟨m,hm,hbot,hl⟩ := hd.centers
  obtain ⟨next,maximum,he,hrm,hstatic,hregisters⟩ :=
    IterationExecution.executes p r swapped scale initial u c hBN hs hshift
      (Tile.vec m) (Invariant.denominatorTile p r initial scale c m)
      (Invariant.accumulatorTile p r initial scale c) hm hl hd.accumulator hi
  obtain ⟨mn,hn⟩ := NumericUpdate.correct p r initial scale c hBN m hbot maximum hrm
  dsimp only at hregisters
  rw [hn] at hregisters
  refine ⟨next,he,hstatic,?_,hregisters.2.2⟩
  refine ⟨fun i => some (mn i),hregisters.1,?_,hregisters.2.1⟩
  intro i hi
  cases hi

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
  obtain ⟨⟨m,hm,hbot,hl⟩,ha⟩ := h
  constructor
  · exact ⟨m,by simpa using hm,hbot,by simpa using hl⟩
  · simpa using ha

theorem indexed_step (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial u : BlockState) (c : Nat) (hBN : 0 < p.BN)
    (hshift : 0 ≤ (inputs initial r).start (launch initial).batch+(c*p.BN : Nat))
    (hs : Invariant.Static p r initial u) (hd : Invariant.Dynamic p r initial scale c u) :
    ∃ next,
      stepStmts (LoopProgram.body p r swapped scale)
        (u.setReg "start_n" .int [] (Tile.scalar ((c*p.BN : Nat) : Int))) = some next ∧
      Invariant.Static p r initial next ∧ Invariant.Dynamic p r initial scale (c+1) next := by
  apply step p r swapped scale initial _ c hBN hshift
  · exact static_index p r initial u _ hs
  · exact dynamic_index p r initial u scale c _ hd
  · simp

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.InvariantStep
