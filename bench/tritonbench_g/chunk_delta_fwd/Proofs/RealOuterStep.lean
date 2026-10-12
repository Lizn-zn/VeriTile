import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealStateStore

/-! One complete outer iteration: store the old H, reset the accumulator,
execute every inner block, then update H. Its output and frame properties refer
to the actual source row span, including overlapping windows. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealOuterStep
open VeriTile.Triton RealProgram
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def historyBase (p : Layout) (l : Launch) (outer : Nat) : Nat :=
  l.head * p.historyHead + outer * p.K * p.V

def historyAddress (p : Layout) (l : Launch) (outer e j : Nat) : Nat :=
  RealStateStore.address p l (historyBase p l outer) p.historyTime e j

def historyFootprint (p : Layout) (l : Launch) (out : RegionName) (outer : Nat) : WriteFootprint :=
  RealStateStore.footprint p l out (historyBase p l outer) p.historyTime

def footprint (p : Layout) (l : Launch) (r : Regions) (outer : Nat) : WriteFootprint :=
  WriteFootprint.union (historyFootprint p l r.history outer) (RealInnerLoop.footprint p l r.corrected outer)

def pointerOp (p : Layout) (out : RegionName) : Op .blockPtr [p.BK,p.BV] :=
  .makeBlockPtrDynOffsets out
    (.add .nat .nil (.mul .nat .nil (.ref .nat [] "i_bh") (.constNat p.historyHead))
      (.mul .nat .nil (.mul .nat .nil (.ref .nat [] "i_t") (.constNat p.K)) (.constNat p.V)))
    [p.K,p.V] [p.BK,p.BV] [p.historyTime,1]
    [.mul .nat .nil (.ref .nat [] "i_k") (.constNat p.BK),
     .mul .nat .nil (.ref .nat [] "i_v") (.constNat p.BV)]

def prelude (p : Layout) (r : Regions) : List Stmt :=
  [.assign .blockPtr [p.BK,p.BV] "p_h" (pointerOp p r.history),
   .store .real [p.BK,p.BV] (.blockPtr (.ref .blockPtr [p.BK,p.BV] "p_h") [0,1])
     (.ref .real [p.BK,p.BV] "b_h") .none,
   .assign .real [p.BK,p.BV] "b_h_cumsum" (.full [p.BK,p.BV] (.const 0))]

def carryOp (p : Layout) : Op .real [p.BK,p.BV] :=
  .add .real Broadcast.nil.consSame.consSame
    (.ref .real [p.BK,p.BV] "b_h") (.ref .real [p.BK,p.BV] "b_h_cumsum")

def body (p : Layout) (r : Regions) : List Stmt :=
  cdfOuterBody r.key r.value r.delta r.corrected r.history
    p.keyHead p.keyTime p.keyDim p.valueHead p.valueTime p.valueDim p.historyHead p.historyTime
    p.T p.K p.V p.BT p.BC p.BK p.BV

theorem body_parts (p : Layout) (r : Regions) :
    body p r = prelude p r ++ [RealInnerLoop.statement p r,
      .assign .real [p.BK,p.BV] "b_h" (carryOp p)] := rfl

def prepared (p : Layout) (l : Launch) (r : Regions) (outer : Nat)
    (frozen : StateValue p) (u : BlockState) : BlockState :=
  (RealStateStore.scatter p l r.history (historyBase p l outer) p.historyTime frozen
    (u.setReg "p_h" .blockPtr [p.BK,p.BV]
      (RealStateStore.pointer p l r.history (historyBase p l outer) p.historyTime))).setReg
    "b_h_cumsum" .real [p.BK,p.BV] (stateTile p (fun _ _ => 0))

theorem pointer_eval (p : Layout) (l : Launch) (out : RegionName) (outer : Nat) (u : BlockState)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar outer)) :
    evalOp (pointerOp p out) u =
      some (RealStateStore.pointer p l out (historyBase p l outer) p.historyTime) := by
  exact makeBlockPtr_2d_eval out u _ _ _ [p.K,p.V] [p.BK,p.BV] [p.historyTime,1]
    (historyBase p l outer) (l.key * p.BK) (l.value * p.BV)
    (addMulMulMul_eval u "i_bh" "i_t" l.head p.historyHead outer p.K p.V hih hit)
    (mulConst_eval u "i_k" l.key p.BK hik) (mulConst_eval u "i_v" l.value p.BV hiv)

theorem prefix_executes (p : Layout) (l : Launch) (r : Regions) (outer : Nat)
    (frozen : StateValue p) (u : BlockState)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar outer))
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p frozen)) :
    stepStmts (prelude p r) u = some (prepared p l r outer frozen u) := by
  unfold prelude
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p l r.history outer u hik hiv hih hit))]
  rw [stepStmts.cons_some (RealStateStore.store_eval p l r.history (historyBase p l outer) p.historyTime
    frozen "p_h" _ (by simp [hh]) (by simp))]
  have hz (v : BlockState) : evalOp (.full [p.BK,p.BV] (.const 0)) v =
      some (stateTile p (fun _ _ => 0)) := by
    simp only [evalOp_full, evalOp_const, Option.map_some]
    rfl
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (hz _)),stepStmts.nil]
  rfl

theorem carry_eval (p : Layout) (frozen acc : StateValue p) (u : BlockState)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p frozen))
    (ha : u.regs .real [p.BK,p.BV] "b_h_cumsum" = some (stateTile p acc)) :
    evalOp (carryOp p) u = some (stateTile p (fun e j => frozen e j + acc e j)) := by
  rw [carryOp, evalOp_add]
  simp only [evalOp_ref, hh, ha, Option.bind_some, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  rfl

structure Separate (r : Regions) : Prop where
  keyHistory : r.key ≠ r.history
  deltaHistory : r.delta ≠ r.history
  valueHistory : r.value ≠ r.history
  keyCorrected : r.key ≠ r.corrected
  deltaCorrected : r.delta ≠ r.corrected
  valueCorrected : r.value ≠ r.corrected
  outputs : r.history ≠ r.corrected

structure Result (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (outer : Nat)
    (frozen : StateValue p) (u v : BlockState) : Prop where
  key : v.regs .nat [] "i_k" = some (Tile.scalar l.key)
  value : v.regs .nat [] "i_v" = some (Tile.scalar l.value)
  head : v.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  time : v.regs .nat [] "i_t" = some (Tile.scalar outer)
  state : v.regs .real [p.BK,p.BV] "b_h" = some (stateTile p (RealWindow.advance p l a outer frozen))
  inputKey : ∀ off, v.readMem r.key off = a.key off
  inputDelta : ∀ off, v.readMem r.delta off = a.delta off
  inputValue : ∀ off, v.readMem r.value off = a.value off
  history : ∀ (e : Fin p.BK) (j : Fin p.BV), RealStateStore.active p l e.val j.val →
    v.readMem r.history (historyAddress p l outer e.val j.val) = frozen e j
  corrected : ∀ (t : Nat) (j : Fin p.BV), activeValue p l t j.val →
    outer * p.BT ≤ t → t < outer * p.BT + RealWindow.span p →
    v.readMem r.corrected (valueAddress p l t j.val) = correctedValue p l a frozen t j
  frame : BlockState.WriteWithin (footprint p l r outer) u v
  pids : v.pids = u.pids
  undef : v.undef = u.undef

theorem prepared_invariant (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (outer : Nat)
    (frozen : StateValue p) (u : BlockState) (hs : Separate r)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar outer))
    (hkey : ∀ off, u.readMem r.key off = a.key off)
    (hdelta : ∀ off, u.readMem r.delta off = a.delta off)
    (hvalue : ∀ off, u.readMem r.value off = a.value off)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p frozen)) :
    RealInnerLoop.Holds (prepared p l r outer frozen u) p l r a outer frozen (fun _ _ => 0) 0
      (prepared p l r outer frozen u) := by
  let pre := prepared p l r outer frozen u
  change RealInnerLoop.Holds pre p l r a outer frozen (fun _ _ => 0) 0 pre
  apply RealInnerLoop.entry
  · simp [pre, prepared, RealStateStore.scatter, hik]
  · simp [pre, prepared, RealStateStore.scatter, hiv]
  · simp [pre, prepared, RealStateStore.scatter, hih]
  · simp [pre, prepared, RealStateStore.scatter, hit]
  · intro off
    dsimp [pre, prepared]
    rw [RealStateStore.scatter,
      BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hs.keyHistory]
    simpa using hkey off
  · intro off
    dsimp [pre, prepared]
    rw [RealStateStore.scatter,
      BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hs.deltaHistory]
    simpa using hdelta off
  · intro off
    dsimp [pre, prepared]
    rw [RealStateStore.scatter,
      BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hs.valueHistory]
    simpa using hvalue off
  · simp [pre, prepared, RealStateStore.scatter, hh]
  · simp [pre, prepared]

theorem executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (outer : Nat)
    (frozen : StateValue p) (u : BlockState) (hu : u.undef = fun _ _ => 0) (hs : Separate r)
    (hvi : RealInnerLoop.NoCollision p l)
    (hhi : RealStateStore.NoCollision p l (historyBase p l outer) p.historyTime)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar outer))
    (hkey : ∀ off, u.readMem r.key off = a.key off)
    (hdelta : ∀ off, u.readMem r.delta off = a.delta off)
    (hvalue : ∀ off, u.readMem r.value off = a.value off)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p frozen)) :
    ∃ v, stepStmts (body p r) u = some v ∧ Result p l r a outer frozen u v := by
  let pre := prepared p l r outer frozen u
  have hepre := prefix_executes p l r outer frozen u hik hiv hih hit hh
  have hpre := prepared_invariant p l r a outer frozen u hs hik hiv hih hit hkey hdelta hvalue hh
  obtain ⟨mid,heinner,hm⟩ := RealInnerLoop.executes pre p l r a outer frozen (fun _ _ => 0)
    (by simpa only [pre, prepared, RealStateStore.scatter, BlockState.setReg, cdfWrites_undef] using hu)
    hs.keyCorrected hs.deltaCorrected hs.valueCorrected hvi pre hpre
  let v := mid.setReg "b_h" .real [p.BK,p.BV] (stateTile p (RealWindow.advance p l a outer frozen))
  have hcarry := carry_eval p frozen
    (RealInnerLoop.accumulated p l a outer (RealInnerLoop.count p) frozen (fun _ _ => 0)) mid hm.state hm.acc
  refine ⟨v,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · rw [body_parts, stepStmts.append_some hepre]
    rw [stepStmts.cons_some heinner,stepStmts.cons_some (stepStmt_assign_eq_some hcarry),stepStmts.nil]
    rfl
  · simpa [v] using hm.key
  · simpa [v] using hm.value
  · simpa [v] using hm.head
  · simpa [v] using hm.time
  · simp [v]
  · intro off; simpa [v] using hm.inputKey off
  · intro off; simpa [v] using hm.inputDelta off
  · intro off; simpa [v] using hm.inputValue off
  · intro e j ha
    have hf := hm.frame r.history (historyAddress p l outer e.val j.val) (by
      intro h; exact hs.outputs h.1)
    change mid.readMem r.history (historyAddress p l outer e.val j.val) = _
    rw [BlockState.readMem, ← hf]
    change pre.readMem r.history (historyAddress p l outer e.val j.val) = _
    exact RealStateStore.readback p l r.history (historyBase p l outer) p.historyTime frozen
      (u.setReg "p_h" .blockPtr [p.BK,p.BV]
        (RealStateStore.pointer p l r.history (historyBase p l outer) p.historyTime)) hhi e j ha
  · intro t j ha hb ht
    exact hm.output t j ha hb ht
  · have hp := RealStateStore.frame p l r.history (historyBase p l outer) p.historyTime frozen
      (u.setReg "p_h" .blockPtr [p.BK,p.BV]
        (RealStateStore.pointer p l r.history (historyBase p l outer) p.historyTime))
    intro rg off hn
    exact (hp rg off (fun hh => hn (Or.inl hh))).trans (hm.frame rg off (fun hh => hn (Or.inr hh)))
  · simpa [v, pre, prepared, RealStateStore.scatter, BlockState.foldl_writeMem_prop_masked_pids] using hm.pids
  · simpa only [v, pre, prepared, RealStateStore.scatter, BlockState.setReg, cdfWrites_undef] using hm.undef

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealOuterStep
