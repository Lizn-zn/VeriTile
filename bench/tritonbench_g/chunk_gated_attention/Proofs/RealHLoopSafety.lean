import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHGateSafety

/-! Complete iteration and loop access safety. Register addresses follow the
actual execution, including the history store before K/V loads. The previously
proved loop invariant supplies every reached state; no whole-program execution
or output equation is assumed as a precondition. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHLoopSafety
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles RealHAccessSafety
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

structure StepBounds (p : Layout) (l : Launch) (r : Regions) (c : Nat) (gateK : Bool)
    (bounds : RegionBounds) : Prop where
  key : KeyBounds p l r.key c bounds
  value : ValueBounds p l r.value c bounds
  history : HistoryBounds p l r.history c bounds
  gate : RealHGateSafety.Bounds p l r c gateK bounds

def Bounds (p : Layout) (l : Launch) (r : Regions) (gateK : Bool) (bounds : RegionBounds) : Prop :=
  ∀ c, c < p.NT → StepBounds p l r c gateK bounds

theorem prelude_safe (p : Layout) (l : Launch) (r : Regions) (c : Nat) (gateK : Bool)
    (f : StateValue p) (bounds : RegionBounds) (u : BlockState) (hb : StepBounds p l r c gateK bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f)) :
    Stmt.TraceSafeList bounds (RealHStep.prelude p r) u := by
  unfold RealHStep.prelude
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_eval p l r.key c u hih hik hit))]
  refine ⟨by simp [Stmt.TraceSafe,keyPtr,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (value_eval p l r.value c _
    (by simp [hih]) (by simp [hiv]) (by simp [hit])))]
  refine ⟨by simp [Stmt.TraceSafe,valuePtr,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (history_eval p l r.history c _
    (by simp [hih]) (by simp [hik]) (by simp [hiv]) (by simp [hit])))]
  refine ⟨by simp [Stmt.TraceSafe,historyPtr,times,Op.SafeAt],?_⟩
  change Stmt.TraceSafeList bounds (storeCarry p :: _) (RealHStep.pointers p l r c u)
  erw [TraceProofs.cons_of_run_iff (RealHStore.store_eval p l r.history (RealHStep.historyBase p l c)
    p.historyKey p.historyValue f "p_h" _ (by simp [RealHStep.pointers,hh]) (by simp [RealHStep.pointers]; rfl))]
  refine ⟨?_,?_⟩
  · exact store2_safe r.history (l.head*p.historyHead+c*p.K*p.V) p.K p.V p.BK p.BV
      p.historyKey p.historyValue (l.key*p.BK) (l.value*p.BV) "p_h" "b_h" bounds _ hb.history
      (by simp [RealHStep.pointers,historyPointer])
  apply TraceProofs.assign_cons (load2_safe r.key (l.head*p.keyHead) p.K p.T p.BK p.BT
    p.keyFeature p.keyTime (l.key*p.BK) (c*p.BT) "p_k" bounds _ hb.key
    (by simp [RealHStore.scatter,RealHStep.pointers,keyPointer]))
  intro key
  apply TraceProofs.assign_cons (load2_safe r.value (l.head*p.valueHead) p.T p.V p.BT p.BV
    p.valueTime p.valueFeature (c*p.BT) (l.value*p.BV) "p_v" bounds _ hb.value
    (by simp [RealHStore.scatter,RealHStep.pointers,valuePointer]))
  intro value
  exact Stmt.TraceSafeList.nil_intro

theorem iteration_safe (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (c : Nat) (gateK swapped : Bool) (f : StateValue p) (u : BlockState)
    (hu : u.undef = fun _ _ => 0) (hs : RealHStep.Separate r)
    (bounds : RegionBounds) (hb : StepBounds p l r c gateK bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f)) :
    Stmt.TraceSafeList bounds (iteration p r gateK swapped) u := by
  have he := RealHStep.prefix_executes p l r a (fun _ _ => 0) c f u hs hu hik hiv hih hit hk hv hh
  have hp := prelude_safe p l r c gateK f bounds u hb hik hiv hih hit hh
  rw [RealHStep.iteration_parts]
  apply TraceProofs.append_safe _ u hp
  intro v hv
  rw [he] at hv
  cases hv
  apply Stmt.TraceSafeList.cons_intro (RealHGateSafety.branch_safe p l r c gateK bounds _ hb.gate
    (by simp [RealHStep.prepared,RealHStep.stored,RealHStore.scatter,RealHStep.pointers,hik])
    (by simp [RealHStep.prepared,RealHStep.stored,RealHStore.scatter,RealHStep.pointers,hiv])
    (by simp [RealHStep.prepared,RealHStep.stored,RealHStore.scatter,RealHStep.pointers,hih])
    (by simp [RealHStep.prepared,RealHStep.stored,RealHStore.scatter,RealHStep.pointers,hit]))
  intro _ _
  apply TraceProofs.all_safe
  cases swapped <;> simp [Stmt.TraceSafe,RealHAlgebra.updateOp,carry,dot,Op.SafeAt,Op.SafeAt.eq_def]

theorem loop_safe (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (gateK hasSeed swapped : Bool) (hu : initial.undef = fun _ _ => 0)
    (hs : RealHStep.Separate r) (hinj : RealHLoop.HistoryNoCollision p l)
    (bounds : RegionBounds) (hb : Bounds p l r gateK bounds) (u : BlockState)
    (h : RealHLoop.Holds initial p l r a gateK hasSeed 0 u) :
    Stmt.TraceSafe bounds (RealHLoop.statement p r gateK swapped) u := by
  rw [RealHLoop.statement,Stmt.TraceSafe]
  apply TraceProofs.range_safe (P := RealHLoop.Holds initial p l r a gateK hasSeed)
  · intro c v hc hv
    apply iteration_safe p l r a c gateK swapped (expected p l a gateK hasSeed c)
      (v.setReg "i_t" .nat [] (Tile.scalar c)) (hv.undef.trans hu) hs bounds (hb c hc)
    · simpa using hv.key
    · simpa using hv.value
    · simpa using hv.head
    · simp
    · simpa using hv.inputKey
    · simpa using hv.inputValue
    · simpa using hv.carry
  · intro c v w hc hv he
    obtain ⟨w',he',hw⟩ := RealHLoop.iteration initial p l r a gateK hasSeed swapped hu hs hinj c v hc hv
    rw [he] at he'
    cases he'
    exact hw
  · exact h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHLoopSafety
