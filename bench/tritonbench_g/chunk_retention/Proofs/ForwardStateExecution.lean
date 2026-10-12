import bench.tritonbench_g.chunk_retention.Proofs.ForwardStateProgram

/-! One complete H iteration: construct pointers, store the old H, load K/V,
and update H. The execution and output
properties retain arbitrary unspecified lane values. Input separation is
required only against the history store performed before these loads. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHStep
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles RealHLoads RealHAlgebra RealDecayExecution RealDecayState
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def historyBase (p : Layout) (l : Launch) (c : Nat) : Nat := l.head*p.historyHead+c*p.K*p.V

def prelude (p : Layout) (r : Regions) : List Stmt :=
  [.assign .blockPtr [p.BK,p.BT] "p_k" (keyPtr p r.key),
   .assign .blockPtr [p.BT,p.BV] "p_v" (valuePtr p r.value),
   .assign .blockPtr [p.BK,p.BV] "p_h" (historyPtr p r.history),
   storeCarry p,
   .assign .real [p.BK,p.BT] "b_k" (load2 p.BK p.BT "p_k"),
   .assign .real [p.BT,p.BV] "b_v" (load2 p.BT p.BV "p_v")]

theorem iteration_parts (p : Layout) (r : Regions) (swapped : Bool) :
    iteration p r swapped = prelude p r ++
      [tailGate p, .assign .real [p.BK,p.BV] "b_h" (updateOp p swapped)] := rfl

def pointers (p : Layout) (l : Launch) (r : Regions) (c : Nat) (u : BlockState) : BlockState :=
  ((u.setReg "p_k" .blockPtr [p.BK,p.BT] (keyPointer p l r.key c)).setReg
    "p_v" .blockPtr [p.BT,p.BV] (valuePointer p l r.value c)).setReg
    "p_h" .blockPtr [p.BK,p.BV] (historyPointer p l r.history c)

def stored (p : Layout) (l : Launch) (r : Regions) (c : Nat) (f : StateValue p) (u : BlockState) : BlockState :=
  RealHStore.scatter p l r.history (historyBase p l c) p.historyKey 1 f (pointers p l r c u)

def prepared (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (f : StateValue p) (u : BlockState) : BlockState :=
  let x := loaded p l r a padding c
  ((stored p l r c f u).setReg "b_k" .real [p.BK,p.BT] (Tile.mat (fun e t => some (x.key e t)))).setReg
    "b_v" .real [p.BT,p.BV] (Tile.mat (fun t j => some (x.value t j)))

structure Separate (r : Regions) : Prop where
  key : r.key ≠ r.history
  value : r.value ≠ r.history

theorem stored_other (p : Layout) (l : Launch) (r : Regions) (c : Nat)
    (f : StateValue p) (u : BlockState) (region : RegionName) (h : region ≠ r.history) :
    ∀ off, (stored p l r c f u).readMem region off = u.readMem region off := by
  intro off
  unfold stored RealHStore.scatter
  rw [BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ h]
  simp [pointers]

theorem prefix_executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (f : StateValue p) (u : BlockState)
    (hs : Separate r) (hu : u.undef = padding)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f)) :
    stepStmts (prelude p r) u = some (prepared p l r a padding c f u) := by
  unfold prelude
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_eval p l r.key c u hih hik hit))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_eval p l r.value c _
    (by simp [hih]) (by simp [hiv]) (by simp [hit])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (history_eval p l r.history c _
    (by simp [hih]) (by simp [hik]) (by simp [hiv]) (by simp [hit])))]
  change stepStmts (storeCarry p :: _) (pointers p l r c u) = _
  erw [stepStmts.cons_some (RealHStore.store_eval p l r.history (historyBase p l c)
    p.historyKey 1 f "p_h" _ (by simp [pointers,hh]) (by simp [pointers]; rfl))]
  change stepStmts (.assign .real [p.BK,p.BT] "b_k" _ :: _) (stored p l r c f u) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_load p l r.key a.key (padding r.key) c "p_k" _
    (fun off => (stored_other p l r c f u r.key hs.key off).trans (hk off))
    (by rw [stored,RealHStore.scatter_undef]; exact congrFun hu r.key)
    (by simp [stored,RealHStore.scatter,pointers])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_load p l r.value a.value (padding r.value) c "p_v" _
    (by simpa using (fun off => (stored_other p l r c f u r.value hs.value off).trans (hv off)))
    (by change (stored p l r c f u).undef r.value = _; rw [stored,RealHStore.scatter_undef]; exact congrFun hu r.value)
    (by simp [stored,RealHStore.scatter,pointers]))),stepStmts.nil]
  rfl

structure Result (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat)
    (f : StateValue p) (u v : BlockState) : Prop where
  keyIndex : v.regs .nat [] "i_k" = some (Tile.scalar l.key)
  valueIndex : v.regs .nat [] "i_v" = some (Tile.scalar l.value)
  headIndex : v.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  timeIndex : v.regs .nat [] "i_t" = some (Tile.scalar c)
  carry : v.regs .real [p.BK,p.BV] "b_h" = some (stateTile p (advance (loaded p l r a padding c) f))
  inputKey : ∀ off, v.readMem r.key off = a.key off
  inputValue : ∀ off, v.readMem r.value off = a.value off
  history : ∀ (e : Fin p.BK) (j : Fin p.BV), RealHStore.active p l e.val j.val →
    v.readMem r.history (RealHStore.address p l (historyBase p l c) p.historyKey 1 e.val j.val) = f e j
  frame : BlockState.WriteWithin (RealHStore.footprint p l r.history (historyBase p l c) p.historyKey 1) u v
  pids : v.pids = u.pids
  undef : v.undef = u.undef
  decayBase : v.regs .real [] "b_b" = some (Tile.scalar (base p l))
  offsets : v.regs .int [p.BT] "o_i" = some (indices p.BT)
  decayScalar : v.regs .real [] "d_b" = some (Tile.scalar (carryFactor p l c))
  decayLanes : v.regs .real [p.BT] "d_i" = some (weights p.BT (length p c) (base p l))

theorem executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (swapped : Bool)
    (f : StateValue p) (u : BlockState) (hs : Separate r) (hu : u.undef = padding)
    (hinj : RealHStore.NoCollision p l (historyBase p l c) p.historyKey 1)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hbase : u.regs .real [] "b_b" = some (Tile.scalar (base p l)))
    (hindices : u.regs .int [p.BT] "o_i" = some (indices p.BT))
    (hfactor : u.regs .real [] "d_b" = some (Tile.scalar (Decay.factor (base p l) p.BT)))
    (hweights : u.regs .real [p.BT] "d_i" = some (weights p.BT p.BT (base p l)))
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f)) :
    ∃ v, stepStmts (iteration p r swapped) u = some v ∧ Result p l r a padding c f u v := by
  let pre := prepared p l r a padding c f u
  let x := loaded p l r a padding c
  let gated := gateState p l c pre
  let v := gated.setReg "b_h" .real [p.BK,p.BV] (stateTile p (advance x f))
  have hp := prefix_executes p l r a padding c f u hs hu hik hiv hih hit hk hv hh
  have hegate := gate_executes p l c pre
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hit])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hindices])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hbase])
  have hg := gate_factors p l c pre
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hfactor])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hweights])
  have headd := update_eval p x swapped f gated
    (by simp [gated,gate_regs,pre,prepared,stored,RealHStore.scatter,pointers,hh])
    (by simp [gated,gate_regs,pre,prepared,x])
    (by simp [gated,gate_regs,pre,prepared,x])
    (by simpa only [gated,x,loaded] using hg.1)
    (by simpa only [gated,x,loaded,weights] using hg.2)
  have hmem : v.mem = (stored p l r c f u).mem := by
    change gated.mem = _
    simp only [gated,gate_memory]
    rfl
  refine ⟨v,?_,⟨?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩⟩
  · rw [iteration_parts,stepStmts.append_some hp,stepStmts.cons_some hegate,
      stepStmts.cons_some (stepStmt_assign_eq_some headd),stepStmts.nil]
  · simp [v,gated,gate_regs,pre,prepared,stored,RealHStore.scatter,pointers,hik]
  · simp [v,gated,gate_regs,pre,prepared,stored,RealHStore.scatter,pointers,hiv]
  · simp [v,gated,gate_regs,pre,prepared,stored,RealHStore.scatter,pointers,hih]
  · simp [v,gated,gate_regs,pre,prepared,stored,RealHStore.scatter,pointers,hit]
  · simp [v,x]
  · intro off
    unfold BlockState.readMem
    rw [hmem]
    exact (stored_other p l r c f u r.key hs.key off).trans (hk off)
  · intro off
    unfold BlockState.readMem
    rw [hmem]
    exact (stored_other p l r c f u r.value hs.value off).trans (hv off)
  · intro e j ha
    unfold BlockState.readMem
    rw [hmem]
    exact RealHStore.readback p l r.history (historyBase p l c) p.historyKey 1 f _ hinj e j ha
  · intro region off hn
    rw [hmem]
    exact RealHStore.frame p l r.history (historyBase p l c) p.historyKey 1 f (pointers p l r c u) region off hn
  · simp [v,gated,pre,prepared,stored,RealHStore.scatter,BlockState.foldl_writeMem_prop_masked_pids,pointers]
  · change gated.undef = _
    dsimp only [gated]
    rw [gate_padding]
    change (stored p l r c f u).undef = _
    rw [stored,RealHStore.scatter_undef]
    rfl
  · simp [v,gated,gate_regs,pre,prepared,stored,RealHStore.scatter,pointers,hbase]
  · simp [v,gated,gate_regs,pre,prepared,stored,RealHStore.scatter,pointers,hindices]
  · simpa [v,gated] using hg.1
  · simpa [v,gated] using hg.2

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHStep

/-! Every iteration of the complete H loop. The carry is the independent
closed form, history retains every earlier pre-update state, and unrelated
complete memory cells are preserved. Zero unspecified values are an explicit
premise for this zero-extended formula; the step theorem itself is unrestricted. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHLoop
open VeriTile.Triton RealHMath RealHProgram RealHTiles RealDecayExecution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def address (p : Layout) (l : Launch) (c e j : Nat) : Nat :=
  RealHStore.address p l (RealHStep.historyBase p l c) p.historyKey 1 e j

def HistoryNoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ (c d : Nat) (e f : Fin p.BK) (i j : Fin p.BV), c < p.NT → d < p.NT →
    RealHStore.active p l e.val i.val → RealHStore.active p l f.val j.val →
    address p l c e.val i.val = address p l d f.val j.val → c = d ∧ e = f ∧ i = j

def footprint (p : Layout) (l : Launch) (r : Regions) : WriteFootprint := fun cell =>
  cell.1 = r.history ∧ ∃ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT ∧
    RealHStore.active p l e.val j.val ∧ cell.2 = address p l c e.val j.val

structure Holds (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (hasSeed : Bool) (c : Nat) (u : BlockState) : Prop where
  count_le : c ≤ p.NT
  key : u.regs .nat [] "i_k" = some (Tile.scalar l.key)
  value : u.regs .nat [] "i_v" = some (Tile.scalar l.value)
  head : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  carry : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p (expected p l a hasSeed c))
  inputKey : ∀ off, u.readMem r.key off = a.key off
  inputValue : ∀ off, u.readMem r.value off = a.value off
  history : ∀ (q : Nat) (e : Fin p.BK) (j : Fin p.BV), q < c → RealHStore.active p l e.val j.val →
    u.readMem r.history (address p l q e.val j.val) = expected p l a hasSeed q e j
  frame : BlockState.WriteWithin (footprint p l r) initial u
  pids : u.pids = initial.pids
  undef : u.undef = initial.undef
  decayBase : u.regs .real [] "b_b" = some (Tile.scalar (base p l))
  offsets : u.regs .int [p.BT] "o_i" = some (indices p.BT)
  standard : c < p.NT →
    u.regs .real [] "d_b" = some (Tile.scalar (Decay.factor (base p l) p.BT)) ∧
    u.regs .real [p.BT] "d_i" = some (weights p.BT p.BT (base p l))

theorem iteration (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (hasSeed swapped : Bool) (hu : initial.undef = fun _ _ => 0)
    (hs : RealHStep.Separate r) (hinj : HistoryNoCollision p l)
    (c : Nat) (u : BlockState) (hc : c < p.NT) (h : Holds initial p l r a hasSeed c u) :
    ∃ v, stepStmts (RealHProgram.iteration p r swapped) (u.setReg "i_t" .nat [] (Tile.scalar c)) = some v ∧
      Holds initial p l r a hasSeed (c+1) v := by
  let entry := u.setReg "i_t" .nat [] (Tile.scalar c)
  have hci : RealHStore.NoCollision p l (RealHStep.historyBase p l c) p.historyKey 1 := by
    intro e f i j he hf heq
    exact (hinj c c e f i j hc hc he hf heq).2
  obtain ⟨v,he,hv⟩ := RealHStep.executes p l r a (fun _ _ => 0) c swapped
    (expected p l a hasSeed c) entry hs (h.undef.trans hu) hci
    (by simp [entry,h.key]) (by simp [entry,h.value]) (by simp [entry,h.head]) (by simp [entry])
    (by simpa [entry] using h.inputKey) (by simpa [entry] using h.inputValue)
    (by simpa [entry] using h.decayBase) (by simpa [entry] using h.offsets)
    (by simpa [entry] using (h.standard hc).1) (by simpa [entry] using (h.standard hc).2)
    (by simpa [entry] using h.carry)
  refine ⟨v,he,Nat.succ_le_of_lt hc,hv.keyIndex,hv.valueIndex,hv.headIndex,?_,
    hv.inputKey,hv.inputValue,?_,?_,?_,?_,hv.decayBase,hv.offsets,?_⟩
  · simpa only [advance_expected] using hv.carry
  · intro q e j hq ha
    by_cases hqc : q = c
    · subst q
      exact hv.history e j ha
    · have hq' : q < c := by omega
      have hf := hv.frame r.history (address p l q e.val j.val) (by
        rintro ⟨_,f,i,hi,hoff⟩
        exact hqc (hinj q c e f j i (Nat.lt_trans hq' hc) hc ha hi hoff).1)
      change v.readMem r.history (address p l q e.val j.val) = _
      unfold BlockState.readMem
      rw [← hf]
      exact h.history q e j hq' ha
  · intro region off hn
    apply (h.frame region off hn).trans
    apply hv.frame region off
    rintro ⟨hr,e,j,ha,hoff⟩
    exact hn ⟨hr,c,e,j,hc,ha,hoff⟩
  · exact hv.pids.trans h.pids
  · exact hv.undef.trans h.undef
  · intro hn
    have hne : c ≠ p.NT - 1 := by omega
    have hlen : length p c = p.BT := by simp [length,Decay.chunkLength,hne]
    simpa only [carryFactor,hlen] using And.intro hv.decayScalar hv.decayLanes

def statement (p : Layout) (r : Regions) (swapped : Bool) : Stmt :=
  .forRangeDyn "i_t" (.constNat 0) (.constNat p.NT) (.constNat 1) (RealHProgram.iteration p r swapped)

theorem executes (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (hasSeed swapped : Bool) (hu : initial.undef = fun _ _ => 0)
    (hs : RealHStep.Separate r) (hinj : HistoryNoCollision p l)
    (u : BlockState) (h : Holds initial p l r a hasSeed 0 u) :
    ∃ v, stepStmt (statement p r swapped) u = some v ∧ Holds initial p l r a hasSeed p.NT v := by
  obtain ⟨last,v,he,hend,hv⟩ := forRangeDyn_inv (idx := "i_t") (start := 0) (stop := p.NT) (step := 1)
    (P := Holds initial p l r a hasSeed) (body := RealHProgram.iteration p r swapped)
    (evalOp_constNat 0 u) (evalOp_constNat p.NT u) (evalOp_constNat 1 u) (by decide : 1 ≠ 0) h
    (fun c u hc h => iteration initial p l r a hasSeed swapped hu hs hinj c u hc h)
  have hlast : last = p.NT := Nat.le_antisymm hv.count_le hend
  subst last
  exact ⟨v,he,hv⟩

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHLoop

/-! Original zero initialization and optional seed assignment, before any output store.
The seed can alias outputs; launch order is key, value, head. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHInit
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles RealHLoads RealDecayExecution RealDecayState
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def registers (p : Layout) (u : BlockState) : BlockState :=
  (standardState p (launch u) (((u.setReg "i_k" .nat [] (Tile.scalar (u.pids 0))).setReg
    "i_v" .nat [] (Tile.scalar (u.pids 1))).setReg
    "i_bh" .nat [] (Tile.scalar (u.pids 2)))).setReg
    "b_h" .real [p.BK,p.BV] (stateTile p (fun _ _ => 0))

def starting (p : Layout) (l : Launch) (a padding : Nat → ℝ) (hasSeed : Bool) : StateValue p :=
  if hasSeed then seedData p l a padding else fun _ _ => 0

def initial (p : Layout) (r : Regions) (a padding : Nat → ℝ)
    (hasSeed : Bool) (u : BlockState) : BlockState :=
  if hasSeed then
    ((registers p u).setReg "p_h0" .blockPtr [p.BK,p.BV] (statePointer p (launch u) r.seed)).setReg
      "b_h" .real [p.BK,p.BV] (stateTile p (seedData p (launch u) a padding))
  else registers p u

theorem if_true (body : List Stmt) (u : BlockState) :
    stepStmt (.ifThen (.constBool Bool.true) body) u = stepStmts body u := by
  simp only [stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some,Tile.scalar,↓reduceIte]

theorem if_false (body : List Stmt) (u : BlockState) :
    stepStmt (.ifThen (.constBool Bool.false) body) u = some u := by
  simp only [stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some,Tile.scalar,Bool.false_eq_true,↓reduceIte]

theorem executes (p : Layout) (r : Regions) (a padding : Nat → ℝ)
    (hasSeed : Bool) (u : BlockState) (hu : u.undef r.seed = padding)
    (hm : ∀ off, u.readMem r.seed off = a off) :
    stepStmts (initialization p r hasSeed) u = some (initial p r a padding hasSeed u) := by
  unfold initialization
  simp only [List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 u))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.append_some (standard_executes p (launch u) _ (by simp [launch,standardState]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BK,p.BV] (.const 0)) _ =
    some (stateTile p (fun _ _ => 0)) from by simp [evalOp_full,evalOp_const]; rfl))]
  cases hasSeed
  · rw [stepStmts.cons_some (if_false _ _),stepStmts.nil]
    rfl
  · rw [stepStmts.cons_some (st := .ifThen (.constBool Bool.true) _) (by
      rw [if_true]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some (state_eval p (launch u) r.seed _
        (by simp [launch,standardState]) (by simp [launch,standardState]) (by simp [launch,standardState])))]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some (seed_load p (launch u) r.seed a padding "p_h0" _
        (by simpa using hm) (by simpa only [BlockState.setReg] using hu) (by simp))),stepStmts.nil])]
    rw [stepStmts.nil]
    rfl

theorem starting_expected (p : Layout) (l : Launch) (a : Inputs) (hasSeed : Bool) :
    starting p l a.seed (fun _ => 0) hasSeed = expected p l a hasSeed 0 := by
  funext e j
  cases hasSeed <;> simp [starting,expected,seedData,RealHMath.seed]

theorem invariant (p : Layout) (r : Regions) (a : Inputs) (hasSeed : Bool) (u : BlockState)
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off) :
    RealHLoop.Holds u p (launch u) r a hasSeed 0 (initial p r a.seed (fun _ => 0) hasSeed u) := by
  refine ⟨Nat.zero_le _,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · cases hasSeed <;> simp [initial,registers,standardState,launch]
  · cases hasSeed <;> simp [initial,registers,standardState,launch]
  · cases hasSeed <;> simp [initial,registers,standardState,launch]
  · rw [← starting_expected]
    cases hasSeed <;> simp [initial,registers,standardState,starting]
  · intro off; cases hasSeed <;> simpa [initial,registers,standardState] using hk off
  · intro off; cases hasSeed <;> simpa [initial,registers,standardState] using hv off
  · intro q _ _ hq; omega
  · intro _ _ _; cases hasSeed <;> rfl
  · cases hasSeed <;> rfl
  · cases hasSeed <;> rfl

  · cases hasSeed <;> simp [initial,registers,standardState]
  · cases hasSeed <;> simp [initial,registers,standardState]
  · intro _; cases hasSeed <;> simp [initial,registers,standardState]

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHInit

/-! Optional final-state store, with its source mask, contiguous state layout
and full-cell frame. Disabling it leaves memory unchanged. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHFinal
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def base (p : Layout) (l : Launch) : Nat := l.head*p.K*p.V

def output (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (f : StateValue p) (u : BlockState) : BlockState :=
  if storeFinal then RealHStore.scatter p l r.final (base p l) p.V 1 f
    (u.setReg "p_ht" .blockPtr [p.BK,p.BV] (statePointer p l r.final))
  else u

def footprint (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool) : WriteFootprint :=
  fun cell => storeFinal = Bool.true ∧ RealHStore.footprint p l r.final (base p l) p.V 1 cell

theorem executes (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (f : StateValue p) (u : BlockState)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f)) :
    stepStmts (finalization p r storeFinal) u = some (output p l r storeFinal f u) := by
  unfold finalization
  cases storeFinal
  · rw [stepStmts.cons_some (RealHInit.if_false _ _),stepStmts.nil]
    rfl
  · rw [stepStmts.cons_some (st := .ifThen (.constBool Bool.true) _) (by
      rw [RealHInit.if_true]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some (state_eval p l r.final u hih hik hiv))]
      erw [stepStmts.cons_some (RealHStore.store_eval p l r.final (base p l) p.V 1 f "p_ht" _
        (by simp [hh]) (by simp; rfl)),stepStmts.nil])]
    rw [stepStmts.nil]
    rfl

theorem readback (p : Layout) (l : Launch) (r : Regions) (f : StateValue p) (u : BlockState)
    (hinj : RealHStore.NoCollision p l (base p l) p.V 1)
    (e : Fin p.BK) (j : Fin p.BV) (ha : RealHStore.active p l e.val j.val) :
    (output p l r Bool.true f u).readMem r.final
      (stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val)) = f e j := by
  simpa [output,RealHStore.address,base,stateAddress] using
    RealHStore.readback p l r.final (base p l) p.V 1 f _ hinj e j ha

theorem frame (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (f : StateValue p) (u : BlockState) :
    BlockState.WriteWithin (footprint p l r storeFinal) u (output p l r storeFinal f u) := by
  cases storeFinal
  · intro _ _ _; rfl
  · intro region off hn
    apply RealHStore.frame p l r.final (base p l) p.V 1 f
      (u.setReg "p_ht" .blockPtr [p.BK,p.BV] (statePointer p l r.final)) region off
    intro h
    exact hn ⟨rfl,h⟩

theorem unchanged (p : Layout) (l : Launch) (r : Regions) (f : StateValue p) (u : BlockState) :
    (output p l r Bool.false f u).mem = u.mem := rfl

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHFinal

/-! Complete real execution of both typed H sources, at arbitrary symbolic
sizes and strides. All seed/final switches and all iterations are covered.
The independent formula requires zero unspecified lanes; the source is not
silently changed to request zero padding. Finite-allocation safety, flat execution and public
IO packaging are proved in RealHSafety, RealHFlat and RealHIO. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHExecution
open VeriTile.Triton RealHMath RealHProgram RealHTiles
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def footprint (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool) : WriteFootprint :=
  WriteFootprint.union (RealHLoop.footprint p l r) (RealHFinal.footprint p l r storeFinal)

structure Outputs (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (hasSeed storeFinal : Bool) (v : BlockState) : Prop where
  history : ∀ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT → RealHStore.active p l e.val j.val →
    v.readMem r.history (historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val)) =
      closed p l a hasSeed c (keyIndex p l e.val) (valueIndex p l j.val)
  final : storeFinal = Bool.true → ∀ (e : Fin p.BK) (j : Fin p.BV), RealHStore.active p l e.val j.val →
    v.readMem r.final (stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val)) =
      closed p l a hasSeed p.NT (keyIndex p l e.val) (valueIndex p l j.val)

theorem executes (p : Layout) (r : Regions) (dot : Storage)
    (seed final : FloatDType) (hasSeed storeFinal swapped : Bool) (a : Inputs) (u : BlockState)
    (hu : u.undef = fun _ _ => 0) (hs : RealHStep.Separate r)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final)
    (hhi : RealHLoop.HistoryNoCollision p (launch u))
    (hfi : storeFinal = Bool.true → RealHStore.NoCollision p (launch u) (RealHFinal.base p (launch u)) p.V 1)
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hseed : ∀ off, u.readMem r.seed off = a.seed off) :
    ∃ v, exec (source p r dot seed final hasSeed storeFinal swapped).eraseDType.toAlgKernel u = some v ∧
      Outputs p (launch u) r a hasSeed storeFinal v ∧
      BlockState.WriteWithin (footprint p (launch u) r storeFinal) u v := by
  let l := launch u
  let start := RealHInit.initial p r a.seed (fun _ => 0) hasSeed u
  have hei := RealHInit.executes p r a.seed (fun _ => 0) hasSeed u (congrFun hu r.seed) hseed
  obtain ⟨mid,hel,hm⟩ := RealHLoop.executes u p l r a hasSeed swapped hu hs hhi start
    (RealHInit.invariant p r a hasSeed u hk hv)
  let f := expected p l a hasSeed p.NT
  let v := RealHFinal.output p l r storeFinal f mid
  have hef := RealHFinal.executes p l r storeFinal f mid hm.key hm.value hm.head hm.carry
  have hf := RealHFinal.frame p l r storeFinal f mid
  refine ⟨v,?_,⟨?_,?_⟩,?_⟩
  · rw [exec,source_body,body,List.append_assoc,stepStmts.append_some hei]
    change stepStmts (RealHLoop.statement p r swapped :: finalization p r storeFinal) start = some v
    rw [stepStmts.cons_some hel]
    exact hef
  · intro c e j hc ha
    have hframe := hf r.history (historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val))
      (fun h => hfsep h.1 h.2.1)
    unfold BlockState.readMem
    rw [← hframe]
    simpa [RealHLoop.address,RealHStore.address,RealHStep.historyBase,historyAddress,Nat.mul_one] using hm.history c e j hc ha
  · intro hon e j ha
    subst storeFinal
    exact RealHFinal.readback p l r f mid (hfi rfl) e j ha
  · intro region off hn
    exact (hm.frame region off (fun h => hn (Or.inl h))).trans
      (hf region off (fun h => hn (Or.inr h)))

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHExecution

#guard_msgs (drop info) in
#auditModuleAxioms
