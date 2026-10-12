import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealStateSafety
import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealInnerSafety

/-! Address safety of the complete outer loop. Each iteration checks its
pre-update history store and all inner-loop loads/stores. The source's actual
inner span and overlap are retained; BC need not divide BT. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealOuterSafety
open VeriTile.Triton RealProgram
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (r : Regions) (bounds : RegionBounds) : Prop where
  inner : RealInnerSafety.Bounds p l r bounds
  history : ∀ c, c < p.NT → RealStateSafety.Bounds p l r.history
    (RealOuterStep.historyBase p l c) p.historyTime bounds

theorem prefix_safe (p : Layout) (l : Launch) (r : Regions) (c : Nat)
    (bounds : RegionBounds) (u : BlockState)
    (hb : RealStateSafety.Bounds p l r.history (RealOuterStep.historyBase p l c) p.historyTime bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    Stmt.TraceSafeList bounds (RealOuterStep.prelude p r) u := by
  unfold RealOuterStep.prelude
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (RealOuterStep.pointer_eval p l r.history c u hik hiv hih hit))]
  refine ⟨by simp [Stmt.TraceSafe, RealOuterStep.pointerOp, Op.SafeAt], ?_⟩
  apply Stmt.TraceSafeList.cons_intro
    (RealStateSafety.store_safe p l r.history (RealOuterStep.historyBase p l c)
      p.historyTime "p_h" bounds _ hb (by simp))
  intro _ _
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe, Op.SafeAt]

theorem iteration_safe (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (c : Nat)
    (frozen : StateValue p) (u : BlockState) (hu : u.undef = fun _ _ => 0)
    (hs : RealOuterStep.Separate r) (hvi : RealInnerLoop.NoCollision p l)
    (bounds : RegionBounds) (hb : Bounds p l r bounds) (hc : c < p.NT)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hkey : ∀ off, u.readMem r.key off = a.key off)
    (hdelta : ∀ off, u.readMem r.delta off = a.delta off)
    (hvalue : ∀ off, u.readMem r.value off = a.value off)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p frozen)) :
    Stmt.TraceSafeList bounds (RealOuterStep.body p r) u := by
  let pre := RealOuterStep.prepared p l r c frozen u
  have he := RealOuterStep.prefix_executes p l r c frozen u hik hiv hih hit hh
  have hi := RealOuterStep.prepared_invariant p l r a c frozen u hs hik hiv hih hit hkey hdelta hvalue hh
  have hp := prefix_safe p l r c bounds u (hb.history c hc) hik hiv hih hit
  have hz : pre.undef = fun _ _ => 0 := by
    simpa only [pre, RealOuterStep.prepared, RealStateStore.scatter, BlockState.setReg, cdfWrites_undef] using hu
  have hloop := RealInnerSafety.loop_safe pre p l r a c frozen (fun _ _ => 0) hz
    hs.keyCorrected hs.deltaCorrected hs.valueCorrected hvi bounds hb.inner pre hi
  rw [RealOuterStep.body_parts]
  apply TraceProofs.append_safe _ u hp
  intro v hv
  rw [he] at hv
  cases hv
  apply Stmt.TraceSafeList.cons_intro hloop
  intro _ _
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe, RealOuterStep.carryOp, Op.SafeAt]

theorem loop_safe (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (seed : Nat → Nat → ℝ) (useSeed : Bool) (hu : initial.undef = fun _ _ => 0)
    (hk : l.key = 0) (hK : p.K ≤ p.BK) (hBC : 0 < p.BC)
    (hs : RealOuterStep.Separate r) (hvi : RealInnerLoop.NoCollision p l)
    (hhi : RealOuterLoop.HistoryNoCollision p l)
    (bounds : RegionBounds) (hb : Bounds p l r bounds) (u : BlockState)
    (h : RealOuterLoop.Holds initial p l r a seed useSeed 0 u) :
    Stmt.TraceSafe bounds (RealOuterLoop.statement p r) u := by
  rw [RealOuterLoop.statement, Stmt.TraceSafe]
  apply TraceProofs.range_safe (P := RealOuterLoop.Holds initial p l r a seed useSeed)
  · intro c v hc hv
    obtain ⟨frozen,hf,_⟩ := hv.state
    apply iteration_safe p l r a c frozen (v.setReg "i_t" .nat [] (Tile.scalar c))
      (by simpa only [BlockState.setReg] using hv.undef.trans hu) hs hvi bounds hb hc
    · simpa using hv.key
    · simpa using hv.value
    · simpa using hv.head
    · simp
    · intro off; simpa using hv.inputKey off
    · intro off; simpa using hv.inputDelta off
    · intro off; simpa using hv.inputValue off
    · simpa using hf
  · intro c v w hc hv he
    obtain ⟨w',he',hw⟩ := RealOuterLoop.iteration initial p l r a seed useSeed hu
      hk hK hBC hs hvi hhi c v hc hv
    rw [he] at he'
    cases he'
    exact hw
  · exact h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealOuterSafety
