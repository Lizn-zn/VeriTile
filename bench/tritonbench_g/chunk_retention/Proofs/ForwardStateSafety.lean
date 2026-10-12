import bench.tritonbench_g.chunk_retention.Proofs.ForwardStateProgram
import VeriTile.Triton.Memory.TraceProofs
import bench.tritonbench_g.chunk_retention.Proofs.ForwardStateExecution

namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHDecaySafety
open VeriTile.Triton RealHMath RealHProgram
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

theorem setup_safe (p : Layout) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (setupDecay p) s := by
  apply TraceProofs.all_safe
  simp [setupDecay,Stmt.TraceSafe,baseOp,decayLane,Op.SafeAt,Op.SafeAt.eq_def]

theorem tail_safe (p : Layout) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafe bounds (tailGate p) s := by
  simp only [tailGate,Stmt.TraceSafe,Op.SafeAt,true_and]
  split
  · split
    · apply TraceProofs.all_safe
      simp [Stmt.TraceSafe,decayLane,Op.SafeAt,Op.SafeAt.eq_def]
    · trivial
  · trivial

end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHDecaySafety

/-! Actual active matrix-access bounds, with independent strides and checked tails. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHAccessSafety
open VeriTile.Triton RealHMath RealHProgram RealHPointers
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def MatrixBounds (r : RegionName) (base rows cols nr nc sr sc offR offC : Nat)
    (bounds : RegionBounds) : Prop :=
  ∀ (i : Fin nr) (j : Fin nc), offR+i.val < rows ∧ offC+j.val < cols →
    base+(offR+i.val)*sr+(offC+j.val)*sc < bounds r

def KeyBounds (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  MatrixBounds r (l.head*p.keyHead) p.K p.T p.BK p.BT p.keyFeature p.keyTime (l.key*p.BK) (c*p.BT)
def ValueBounds (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  MatrixBounds r (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueFeature (c*p.BT) (l.value*p.BV)
def StateBounds (p : Layout) (l : Launch) (r : RegionName) :=
  MatrixBounds r (l.head*p.K*p.V) p.K p.V p.BK p.BV p.V 1 (l.key*p.BK) (l.value*p.BV)
def HistoryBounds (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  MatrixBounds r (l.head*p.historyHead+c*p.K*p.V) p.K p.V p.BK p.BV p.historyKey 1 (l.key*p.BK) (l.value*p.BV)
theorem load2_safe (r : RegionName) (base rows cols nr nc sr sc offR offC : Nat)
    (name : RegName) (bounds : RegionBounds) (u : BlockState)
    (hb : MatrixBounds r base rows cols nr nc sr sc offR offC bounds)
    (hp : u.regs .blockPtr [nr,nc] name = some (matrixPointer r base rows cols nr nc sr sc offR offC)) :
    (load2 nr nc name).SafeAt bounds u := by
  rw [load2,Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he idx _
  rw [evalOp_ref,hp] at he
  cases he
  rcases idx with ⟨i,j,⟨⟩⟩
  simpa only [matrixPointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,decide_eq_true_eq] using hb i j

theorem store2_safe (r : RegionName) (base rows cols nr nc sr sc offR offC : Nat)
    (name value : RegName) (bounds : RegionBounds) (u : BlockState)
    (hb : MatrixBounds r base rows cols nr nc sr sc offR offC bounds)
    (hp : u.regs .blockPtr [nr,nc] name = some (matrixPointer r base rows cols nr nc sr sc offR offC)) :
    Stmt.TraceSafe bounds (.store .real [nr,nc] (.blockPtr (.ref .blockPtr [nr,nc] name) [0,1])
      (.ref .real [nr,nc] value) .none) u := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt,Op.SafeAt],by simp [Op.SafeAt],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he idx _
  rw [evalOp_ref,hp] at he
  cases he
  rcases idx with ⟨i,j,⟨⟩⟩
  simpa only [matrixPointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,decide_eq_true_eq] using hb i j

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHAccessSafety

/-! Complete iteration and loop access safety. Register addresses follow the
actual execution, including the history store before K/V loads. The previously
proved loop invariant supplies every reached state; no whole-program execution
or output equation is assumed as a precondition. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHLoopSafety
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles RealHAccessSafety
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

structure StepBounds (p : Layout) (l : Launch) (r : Regions) (c : Nat)
    (bounds : RegionBounds) : Prop where
  key : KeyBounds p l r.key c bounds
  value : ValueBounds p l r.value c bounds
  history : HistoryBounds p l r.history c bounds

def Bounds (p : Layout) (l : Launch) (r : Regions) (bounds : RegionBounds) : Prop :=
  ∀ c, c < p.NT → StepBounds p l r c bounds

theorem prelude_safe (p : Layout) (l : Launch) (r : Regions) (c : Nat)
    (f : StateValue p) (bounds : RegionBounds) (u : BlockState) (hb : StepBounds p l r c bounds)
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
    p.historyKey 1 f "p_h" _ (by simp [RealHStep.pointers,hh]) (by simp [RealHStep.pointers]; rfl))]
  refine ⟨?_,?_⟩
  · exact store2_safe r.history (l.head*p.historyHead+c*p.K*p.V) p.K p.V p.BK p.BV
      p.historyKey 1 (l.key*p.BK) (l.value*p.BV) "p_h" "b_h" bounds _ hb.history
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
    (c : Nat) (swapped : Bool) (f : StateValue p) (u : BlockState)
    (hu : u.undef = fun _ _ => 0) (hs : RealHStep.Separate r)
    (bounds : RegionBounds) (hb : StepBounds p l r c bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f)) :
    Stmt.TraceSafeList bounds (iteration p r swapped) u := by
  have he := RealHStep.prefix_executes p l r a (fun _ _ => 0) c f u hs hu hik hiv hih hit hk hv hh
  have hp := prelude_safe p l r c f bounds u hb hik hiv hih hit hh
  rw [RealHStep.iteration_parts]
  apply TraceProofs.append_safe _ u hp
  intro v hv
  rw [he] at hv
  cases hv
  apply TraceProofs.all_safe
  intro st h t
  simp only [List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with rfl | rfl
  · exact RealHDecaySafety.tail_safe p bounds t
  · cases swapped <;> simp [Stmt.TraceSafe,RealHAlgebra.updateOp,scaledCarry,carry,dot,Op.SafeAt,Op.SafeAt.eq_def]

theorem loop_safe (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (hasSeed swapped : Bool) (hu : initial.undef = fun _ _ => 0)
    (hs : RealHStep.Separate r) (hinj : RealHLoop.HistoryNoCollision p l)
    (bounds : RegionBounds) (hb : Bounds p l r bounds) (u : BlockState)
    (h : RealHLoop.Holds initial p l r a hasSeed 0 u) :
    Stmt.TraceSafe bounds (RealHLoop.statement p r swapped) u := by
  rw [RealHLoop.statement,Stmt.TraceSafe]
  simp only [Op.SafeAt,evalOp,Tile.scalar,true_and]
  apply TraceProofs.range_safe (P := RealHLoop.Holds initial p l r a hasSeed)
  · intro c v hc hv
    apply iteration_safe p l r a c swapped (expected p l a hasSeed c)
      (v.setReg "i_t" .nat [] (Tile.scalar c)) (hv.undef.trans hu) hs bounds (hb c hc)
    · simpa using hv.key
    · simpa using hv.value
    · simpa using hv.head
    · simp
    · simpa using hv.inputKey
    · simpa using hv.inputValue
    · simpa using hv.carry
  · intro c v w hc hv he
    obtain ⟨w',he',hw⟩ := RealHLoop.iteration initial p l r a hasSeed swapped hu hs hinj c v hc hv
    rw [he] at he'
    cases he'
    exact hw
  · exact h

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHLoopSafety

/-! Safety of the optional seed and final-state accesses. The seed is loaded
by the original assignment, and final output uses its exact checked layout.
Neither numerical values nor an output equality are safety assumptions. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHStateSafety
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHAccessSafety RealDecayState RealHDecaySafety
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem init_safe (p : Layout) (r : Regions) (hasSeed : Bool)
    (bounds : RegionBounds) (u : BlockState)
    (hb : hasSeed = Bool.true → StateBounds p (launch u) r.seed bounds) :
    Stmt.TraceSafeList bounds (initialization p r hasSeed) u := by
  unfold initialization
  simp only [List.cons_append,List.nil_append]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 0 u))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  apply TraceProofs.append_safe _ _ (setup_safe p bounds _)
  intro pre he
  rw [standard_executes p (launch u) _ (by simp [launch,standardState])] at he
  cases he
  apply TraceProofs.assign_cons (by simp [Op.SafeAt])
  intro zero
  apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
  cases hasSeed
  · rw [Stmt.TraceSafe]
    exact ⟨by simp [Op.SafeAt],by simp only [evalOp,Tile.scalar,Bool.false_eq_true,ite_false]⟩
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,↓reduceIte,true_and]
    rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (state_eval p (launch u) r.seed _
      (by simp [launch,standardState]) (by simp [launch,standardState]) (by simp [launch,standardState])))]
    refine ⟨by simp [Stmt.TraceSafe,statePtr,stateBase,times,Op.SafeAt],?_⟩
    apply TraceProofs.assign_cons
    · exact load2_safe r.seed ((launch u).head*p.K*p.V) p.K p.V p.BK p.BV p.V 1
        ((launch u).key*p.BK) ((launch u).value*p.BV) "p_h0" bounds _ (hb rfl) (by simp [statePointer])
    · intro _
      exact Stmt.TraceSafeList.nil_intro

theorem final_safe (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (bounds : RegionBounds) (u : BlockState)
    (hb : storeFinal = Bool.true → StateBounds p l r.final bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)) :
    Stmt.TraceSafeList bounds (finalization p r storeFinal) u := by
  unfold finalization
  apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
  cases storeFinal
  · rw [Stmt.TraceSafe]
    exact ⟨by simp [Op.SafeAt],by simp only [evalOp,Tile.scalar,Bool.false_eq_true,ite_false]⟩
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,↓reduceIte,true_and]
    rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (state_eval p l r.final u hih hik hiv))]
    refine ⟨by simp [Stmt.TraceSafe,statePtr,stateBase,times,Op.SafeAt],?_⟩
    apply Stmt.TraceSafeList.cons_intro
      (store2_safe r.final (l.head*p.K*p.V) p.K p.V p.BK p.BV p.V 1 (l.key*p.BK) (l.value*p.BV)
        "p_ht" "b_h" bounds _ (hb rfl) (by simp [statePointer]))
    intro _ _
    exact Stmt.TraceSafeList.nil_intro

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHStateSafety

/-! Full-source allocation safety: optional state accesses and every reached
K/V/history access, retaining independent source strides. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHSafety
open VeriTile.Triton RealHMath RealHProgram RealHAccessSafety
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (r : Regions) (hasSeed storeFinal : Bool)
    (bounds : RegionBounds) : Prop where
  initial : hasSeed = Bool.true → StateBounds p l r.seed bounds
  loop : RealHLoopSafety.Bounds p l r bounds
  final : storeFinal = Bool.true → StateBounds p l r.final bounds

theorem source_safe (p : Layout) (r : Regions) (dot : Storage)
    (seed final : FloatDType) (hasSeed storeFinal swapped : Bool)
    (u : BlockState) (hu : u.undef = fun _ _ => 0) (hs : RealHStep.Separate r)
    (hinj : RealHLoop.HistoryNoCollision p (launch u))
    (bounds : RegionBounds) (hb : Bounds p (launch u) r hasSeed storeFinal bounds) :
    (source p r dot seed final hasSeed storeFinal swapped).eraseDType.toAlgKernel.TraceSafe bounds u := by
  let l := launch u
  let a : Inputs := ⟨u.readMem r.key,u.readMem r.value,u.readMem r.seed⟩
  let start := RealHInit.initial p r a.seed (fun _ => 0) hasSeed u
  have hi := RealHInit.invariant p r a hasSeed u (fun _ => rfl) (fun _ => rfl)
  have hei := RealHInit.executes p r a.seed (fun _ => 0) hasSeed u (congrFun hu r.seed) (fun _ => rfl)
  have hloop := RealHLoopSafety.loop_safe u p l r a hasSeed swapped hu hs hinj bounds hb.loop start hi
  obtain ⟨mid,hel,hm⟩ := RealHLoop.executes u p l r a hasSeed swapped hu hs hinj start hi
  rw [Kernel.TraceSafe,source_body,body,List.append_assoc]
  apply TraceProofs.append_safe _ u (RealHStateSafety.init_safe p r hasSeed bounds u hb.initial)
  intro v hv
  rw [hei] at hv
  cases hv
  change Stmt.TraceSafeList bounds (RealHLoop.statement p r swapped :: finalization p r storeFinal) start
  rw [TraceProofs.cons_of_run_iff hel]
  exact ⟨hloop,RealHStateSafety.final_safe p l r storeFinal bounds mid hb.final hm.key hm.value hm.head⟩

theorem source_flatten_ok (p : Layout) (r : Regions) (dot : Storage)
    (seed final : FloatDType) (hasSeed storeFinal swapped : Bool) :
    (source p r dot seed final hasSeed storeFinal swapped).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,source_body]
  cases swapped <;> simp [StmtList.FlattenOk,Stmt.FlattenOk,body,initialization,finalization,
    iteration,storeCarry,carry,RealHProgram.dot,setupDecay,baseOp,decayLane,tailGate,scaledCarry,
    keyPtr,valuePtr,historyPtr,statePtr,stateBase,times,load2,Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHSafety

#guard_msgs (drop info) in
#auditModuleAxioms
