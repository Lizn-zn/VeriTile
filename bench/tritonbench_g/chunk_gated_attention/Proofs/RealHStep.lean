import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHGate
import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHStore

/-! One complete H iteration: construct pointers, store the old H, load K/V,
execute the selected gate branch, and update H. The execution and output
properties retain arbitrary unspecified lane values. Input separation is
required only against the history store performed before these loads. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHStep
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles RealHLoads RealHAlgebra
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

theorem iteration_parts (p : Layout) (r : Regions) (gateK swapped : Bool) :
    iteration p r gateK swapped = prelude p r ++
      [.ifThenElse (.constBool gateK) (gateKey p r) (RealHProgram.gateValue p r),
       .assign .real [p.BK,p.BV] "b_h" (updateOp p swapped)] := rfl

def pointers (p : Layout) (l : Launch) (r : Regions) (c : Nat) (u : BlockState) : BlockState :=
  ((u.setReg "p_k" .blockPtr [p.BK,p.BT] (keyPointer p l r.key c)).setReg
    "p_v" .blockPtr [p.BT,p.BV] (valuePointer p l r.value c)).setReg
    "p_h" .blockPtr [p.BK,p.BV] (historyPointer p l r.history c)

def stored (p : Layout) (l : Launch) (r : Regions) (c : Nat) (f : StateValue p) (u : BlockState) : BlockState :=
  RealHStore.scatter p l r.history (historyBase p l c) p.historyKey p.historyValue f (pointers p l r c u)

def prepared (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (f : StateValue p) (u : BlockState) : BlockState :=
  let x := loaded p l r a padding c
  ((stored p l r c f u).setReg "b_k" .real [p.BK,p.BT] (Tile.mat (fun e t => some (x.key e t)))).setReg
    "b_v" .real [p.BT,p.BV] (Tile.mat (fun t j => some (x.value t j)))

structure Separate (r : Regions) : Prop where
  key : r.key ≠ r.history
  value : r.value ≠ r.history
  gate : r.gate ≠ r.history

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
    p.historyKey p.historyValue f "p_h" _ (by simp [pointers,hh]) (by simp [pointers]; rfl))]
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
    (padding : RegionName → Nat → ℝ) (c : Nat) (gateK : Bool)
    (f : StateValue p) (u v : BlockState) : Prop where
  keyIndex : v.regs .nat [] "i_k" = some (Tile.scalar l.key)
  valueIndex : v.regs .nat [] "i_v" = some (Tile.scalar l.value)
  headIndex : v.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  timeIndex : v.regs .nat [] "i_t" = some (Tile.scalar c)
  carry : v.regs .real [p.BK,p.BV] "b_h" = some (stateTile p (advance (loaded p l r a padding c) gateK f))
  inputKey : ∀ off, v.readMem r.key off = a.key off
  inputValue : ∀ off, v.readMem r.value off = a.value off
  inputGate : ∀ off, v.readMem r.gate off = a.gate off
  history : ∀ (e : Fin p.BK) (j : Fin p.BV), RealHStore.active p l e.val j.val →
    v.readMem r.history (RealHStore.address p l (historyBase p l c) p.historyKey p.historyValue e.val j.val) = f e j
  frame : BlockState.WriteWithin (RealHStore.footprint p l r.history (historyBase p l c) p.historyKey p.historyValue) u v
  pids : v.pids = u.pids
  undef : v.undef = u.undef

theorem executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (gateK swapped : Bool)
    (f : StateValue p) (u : BlockState) (hs : Separate r) (hu : u.undef = padding)
    (hinj : RealHStore.NoCollision p l (historyBase p l c) p.historyKey p.historyValue)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hg : ∀ off, u.readMem r.gate off = a.gate off)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f)) :
    ∃ v, stepStmts (iteration p r gateK swapped) u = some v ∧ Result p l r a padding c gateK f u v := by
  let pre := prepared p l r a padding c f u
  let x := loaded p l r a padding c
  let mid := RealHGate.output p l r a padding c gateK f pre
  let v := mid.setReg "b_h" .real [p.BK,p.BV] (stateTile p (advance x gateK f))
  have hp := prefix_executes p l r a padding c f u hs hu hik hiv hih hit hk hv hh
  have heg := RealHGate.executes p l r a padding c gateK f pre
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hik])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hiv])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hih])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hit])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hh])
    (by simp [pre,prepared]) (by simp [pre,prepared])
    (by simpa [pre,prepared] using (fun off => (stored_other p l r c f u r.gate hs.gate off).trans (hg off)))
    (by change (stored p l r c f u).undef = _; rw [stored,RealHStore.scatter_undef]; exact hu)
  have hm := RealHGate.output_result p l r a padding c gateK f pre
    (by simp [pre,prepared]) (by simp [pre,prepared])
  have headd := update_eval p x gateK swapped f mid hm.carry hm.key hm.value
  have hmem : v.mem = (stored p l r c f u).mem := by
    change mid.mem = (stored p l r c f u).mem
    exact hm.mem
  refine ⟨v,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · rw [iteration_parts,stepStmts.append_some hp,stepStmts.cons_some heg,
      stepStmts.cons_some (stepStmt_assign_eq_some headd),stepStmts.nil]
  · simpa [v,pre,prepared,stored,RealHStore.scatter,pointers,hik] using hm.keyIndex
  · simpa [v,pre,prepared,stored,RealHStore.scatter,pointers,hiv] using hm.valueIndex
  · simpa [v,pre,prepared,stored,RealHStore.scatter,pointers,hih] using hm.headIndex
  · simpa [v,pre,prepared,stored,RealHStore.scatter,pointers,hit] using hm.timeIndex
  · simp [v,x]
  · intro off
    unfold BlockState.readMem
    rw [hmem]
    exact (stored_other p l r c f u r.key hs.key off).trans (hk off)
  · intro off
    unfold BlockState.readMem
    rw [hmem]
    exact (stored_other p l r c f u r.value hs.value off).trans (hv off)
  · intro off
    unfold BlockState.readMem
    rw [hmem]
    exact (stored_other p l r c f u r.gate hs.gate off).trans (hg off)
  · intro e j ha
    unfold BlockState.readMem
    rw [hmem]
    exact RealHStore.readback p l r.history (historyBase p l c) p.historyKey p.historyValue f _ hinj e j ha
  · intro region off hn
    rw [hmem]
    exact RealHStore.frame p l r.history (historyBase p l c) p.historyKey p.historyValue f (pointers p l r c u) region off hn
  · change mid.pids = u.pids
    rw [hm.pids]
    simp [pre,prepared,stored,RealHStore.scatter,BlockState.foldl_writeMem_prop_masked_pids,pointers]
  · change mid.undef = u.undef
    rw [hm.undef]
    change (stored p l r c f u).undef = _
    rw [stored,RealHStore.scatter_undef]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHStep
