import bench.tritonbench_g.chunk_linear_attn.Proofs.BackwardStateProgram

/-! One reverse-state chunk: store the prior carry, load and scale Q, load dO,
then add the contraction. The step retains arbitrary unspecified lane values;
only the closed-form theorem later requires a zero padding oracle. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhStep
open VeriTile.Triton
open RealHMath (Layout Launch)
open RealHProgram (keyPtr valuePtr historyPtr load2)
open RealHTiles (StateValue stateTile)
open RealHPointers RealHLoads RealDhMath RealDhProgram RealDhTiles RealDhAlgebra
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def historyBase (p : Layout) (l : Launch) (c : Nat) : Nat := l.head*p.historyHead+c*p.K*p.V

def prelude (p : Layout) (r : Regions) (scale : ℝ) : List Stmt :=
  [.assign .blockPtr [p.BK,p.BT] "p_q" (keyPtr p r.query),
   .assign .blockPtr [p.BT,p.BV] "p_do" (valuePtr p r.incoming),
   .assign .blockPtr [p.BK,p.BV] "p_dh" (historyPtr p r.history),
   storeCarry p,
   .assign .real [p.BK,p.BT] "b_q" (load2 p.BK p.BT "p_q"),
   .assign .real [p.BK,p.BT] "b_q" (scaledQuery p scale),
   .assign .real [p.BT,p.BV] "b_do" (load2 p.BT p.BV "p_do")]

theorem iteration_parts (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) :
    iterationBody p r scale swapped = prelude p r scale ++
      [.assign .real [p.BK,p.BV] "b_dh" (update p swapped)] := rfl

def pointers (p : Layout) (l : Launch) (r : Regions) (c : Nat) (u : BlockState) : BlockState :=
  ((u.setReg "p_q" .blockPtr [p.BK,p.BT] (keyPointer p l r.query c)).setReg
    "p_do" .blockPtr [p.BT,p.BV] (valuePointer p l r.incoming c)).setReg
    "p_dh" .blockPtr [p.BK,p.BV] (historyPointer p l r.history c)

def stored (p : Layout) (l : Launch) (r : Regions) (c : Nat) (f : StateValue p) (u : BlockState) : BlockState :=
  RealDhStore.scatter p l r.history (historyBase p l c) p.historyKey 1 f (pointers p l r c u)

def prepared (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (scale : ℝ) (f : StateValue p) (u : BlockState) : BlockState :=
  let x := loaded p l r a padding c
  (((stored p l r c f u).setReg "b_q" .real [p.BK,p.BT] (Tile.mat (fun e t => some (x.query e t)))).setReg "b_q" .real [p.BK,p.BT] (Tile.mat (fun e t => some (x.query e t * scale)))).setReg
    "b_do" .real [p.BT,p.BV] (Tile.mat (fun t j => some (x.incoming t j)))

structure Separate (r : Regions) : Prop where
  key : r.query ≠ r.history
  value : r.incoming ≠ r.history

theorem stored_other (p : Layout) (l : Launch) (r : Regions) (c : Nat)
    (f : StateValue p) (u : BlockState) (region : RegionName) (h : region ≠ r.history) :
    ∀ off, (stored p l r c f u).readMem region off = u.readMem region off := by
  intro off
  unfold stored RealDhStore.scatter
  rw [BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ h]
  simp [pointers]

theorem prefix_executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (scale : ℝ) (f : StateValue p) (u : BlockState)
    (hs : Separate r) (hu : u.undef = padding)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hk : ∀ off, u.readMem r.query off = a.query off)
    (hv : ∀ off, u.readMem r.incoming off = a.incoming off)
    (hh : u.regs .real [p.BK,p.BV] "b_dh" = some (stateTile p f)) :
    stepStmts (prelude p r scale) u = some (prepared p l r a padding c scale f u) := by
  unfold prelude
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_eval p l r.query c u hih hik hit))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_eval p l r.incoming c _
    (by simp [hih]) (by simp [hiv]) (by simp [hit])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (history_eval p l r.history c _
    (by simp [hih]) (by simp [hik]) (by simp [hiv]) (by simp [hit])))]
  change stepStmts (storeCarry p :: _) (pointers p l r c u) = _
  erw [stepStmts.cons_some (RealDhStore.store_eval p l r.history (historyBase p l c)
    p.historyKey 1 f "p_dh" _ (by simp [pointers,hh]) (by simp [pointers]; rfl))]
  change stepStmts (.assign .real [p.BK,p.BT] "b_q" _ :: _) (stored p l r c f u) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_load p l r.query a.query (padding r.query) c "p_q" _
    (fun off => (stored_other p l r c f u r.query hs.key off).trans (hk off))
    (by rw [stored,RealDhStore.scatter_undef]; exact congrFun hu r.query)
    (by simp [stored,RealDhStore.scatter,pointers])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (scale_eval p (loaded p l r a padding c) scale _ (by
    simp [loaded,RealHTiles.keyData])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_load p l r.incoming a.incoming (padding r.incoming) c "p_do" _
    (by simpa using (fun off => (stored_other p l r c f u r.incoming hs.value off).trans (hv off)))
    (by change (stored p l r c f u).undef r.incoming = _; rw [stored,RealDhStore.scatter_undef]; exact congrFun hu r.incoming)
    (by simp [stored,RealDhStore.scatter,pointers]))),stepStmts.nil]
  rfl

structure Result (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (scale : ℝ)
    (f : StateValue p) (u v : BlockState) : Prop where
  keyIndex : v.regs .nat [] "i_k" = some (Tile.scalar l.key)
  valueIndex : v.regs .nat [] "i_v" = some (Tile.scalar l.value)
  headIndex : v.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  timeIndex : v.regs .nat [] "i_t" = some (Tile.scalar c)
  carry : v.regs .real [p.BK,p.BV] "b_dh" = some (stateTile p (advance (loaded p l r a padding c) scale f))
  inputKey : ∀ off, v.readMem r.query off = a.query off
  inputValue : ∀ off, v.readMem r.incoming off = a.incoming off
  history : ∀ (e : Fin p.BK) (j : Fin p.BV), RealDhStore.active p l e.val j.val →
    v.readMem r.history (RealDhStore.address p l (historyBase p l c) p.historyKey 1 e.val j.val) = f e j
  frame : BlockState.WriteWithin (RealDhStore.footprint p l r.history (historyBase p l c) p.historyKey 1) u v
  pids : v.pids = u.pids
  undef : v.undef = u.undef

theorem executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (padding : RegionName → Nat → ℝ) (c : Nat) (scale : ℝ) (swapped : Bool)
    (f : StateValue p) (u : BlockState) (hs : Separate r) (hu : u.undef = padding)
    (hinj : RealDhStore.NoCollision p l (historyBase p l c) p.historyKey 1)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hk : ∀ off, u.readMem r.query off = a.query off)
    (hv : ∀ off, u.readMem r.incoming off = a.incoming off)
    (hh : u.regs .real [p.BK,p.BV] "b_dh" = some (stateTile p f)) :
    ∃ v, stepStmts (iterationBody p r scale swapped) u = some v ∧ Result p l r a padding c scale f u v := by
  let pre := prepared p l r a padding c scale f u
  let x := loaded p l r a padding c
  let v := pre.setReg "b_dh" .real [p.BK,p.BV] (stateTile p (advance x scale f))
  have hp := prefix_executes p l r a padding c scale f u hs hu hik hiv hih hit hk hv hh
  have headd := update_eval p x scale swapped f pre
    (by simp [pre,prepared,stored,RealDhStore.scatter,pointers,hh])
    (by simp [pre,prepared,x]) (by simp [pre,prepared,x])
  have hmem : v.mem = (stored p l r c f u).mem := rfl
  refine ⟨v,?_,⟨?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩⟩
  · rw [iteration_parts,stepStmts.append_some hp,
      stepStmts.cons_some (stepStmt_assign_eq_some headd),stepStmts.nil]
  · simp [v,pre,prepared,stored,RealDhStore.scatter,pointers,hik]
  · simp [v,pre,prepared,stored,RealDhStore.scatter,pointers,hiv]
  · simp [v,pre,prepared,stored,RealDhStore.scatter,pointers,hih]
  · simp [v,pre,prepared,stored,RealDhStore.scatter,pointers,hit]
  · simp [v,x]
  · intro off
    unfold BlockState.readMem
    rw [hmem]
    exact (stored_other p l r c f u r.query hs.key off).trans (hk off)
  · intro off
    unfold BlockState.readMem
    rw [hmem]
    exact (stored_other p l r c f u r.incoming hs.value off).trans (hv off)
  · intro e j ha
    unfold BlockState.readMem
    rw [hmem]
    exact RealDhStore.readback p l r.history (historyBase p l c) p.historyKey 1 f _ hinj e j ha
  · intro region off hn
    rw [hmem]
    exact RealDhStore.frame p l r.history (historyBase p l c) p.historyKey 1 f (pointers p l r c u) region off hn
  · simp [v,pre,prepared,stored,RealDhStore.scatter,BlockState.foldl_writeMem_prop_masked_pids,pointers]
  · change (stored p l r c f u).undef = _
    rw [stored,RealDhStore.scatter_undef]
    rfl

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhStep

/-! Complete reverse loop: n counts finished iterations, while NT-1-n is the
physical chunk. History stores the carry before the current contribution.
The closed-form interpretation explicitly uses a zero unspecified-value oracle. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhLoop
open VeriTile.Triton
open RealHMath (Layout Launch)
open RealHTiles (StateValue stateTile)
open RealDhMath RealDhProgram RealDhTiles
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def address (p : Layout) (l : Launch) (c e j : Nat) : Nat :=
  RealDhStore.address p l (RealDhStep.historyBase p l c) p.historyKey 1 e j

def HistoryNoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ (c d : Nat) (e f : Fin p.BK) (i j : Fin p.BV), c < p.NT → d < p.NT →
    RealDhStore.active p l e.val i.val → RealDhStore.active p l f.val j.val →
    address p l c e.val i.val = address p l d f.val j.val → c = d ∧ e = f ∧ i = j

def footprint (p : Layout) (l : Launch) (r : Regions) : WriteFootprint := fun cell =>
  cell.1 = r.history ∧ ∃ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT ∧
    RealDhStore.active p l e.val j.val ∧ cell.2 = address p l c e.val j.val

structure Holds (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (scale : ℝ) (n : Nat) (u : BlockState) : Prop where
  count_le : n ≤ p.NT
  key : u.regs .nat [] "i_k" = some (Tile.scalar l.key)
  value : u.regs .nat [] "i_v" = some (Tile.scalar l.value)
  head : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  carry : u.regs .real [p.BK,p.BV] "b_dh" = some (stateTile p (expected p l a scale n))
  inputQuery : ∀ off, u.readMem r.query off = a.query off
  inputGradient : ∀ off, u.readMem r.incoming off = a.incoming off
  history : ∀ (q : Nat) (e : Fin p.BK) (j : Fin p.BV), q < n → RealDhStore.active p l e.val j.val →
    u.readMem r.history (address p l (p.NT-1-q) e.val j.val) = expected p l a scale q e j
  frame : BlockState.WriteWithin (footprint p l r) initial u
  pids : u.pids = initial.pids
  undef : u.undef = initial.undef

theorem reverseIndex_eval (p : Layout) (n : Nat) (u : BlockState)
    (hj : u.regs .nat [] "j" = some (Tile.scalar n)) :
    evalOp (reverseIndex p) u = some (Tile.scalar (p.NT-1-n)) := by
  simp only [reverseIndex,evalOp_sub,evalOp_constNat,evalOp_ref,hj,
    Option.bind_eq_bind,Option.bind_some]
  rfl

theorem iteration (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (scale : ℝ) (swapped : Bool) (hu : initial.undef = fun _ _ => 0)
    (hs : RealDhStep.Separate r) (hinj : HistoryNoCollision p l)
    (n : Nat) (u : BlockState) (hn : n < p.NT) (h : Holds initial p l r a scale n u) :
    ∃ v, stepStmts (RealDhProgram.iteration p r scale swapped) (u.setReg "j" .nat [] (Tile.scalar n)) = some v ∧
      Holds initial p l r a scale (n+1) v := by
  let c := p.NT-1-n
  have hc : c < p.NT := by dsimp [c]; omega
  let entry := (u.setReg "j" .nat [] (Tile.scalar n)).setReg "i_t" .nat [] (Tile.scalar c)
  have hci : RealDhStore.NoCollision p l (RealDhStep.historyBase p l c) p.historyKey 1 := by
    intro e f i j he hf heq
    exact (hinj c c e f i j hc hc he hf heq).2
  obtain ⟨v,he,hv⟩ := RealDhStep.executes p l r a (fun _ _ => 0) c scale swapped
    (expected p l a scale n) entry hs (h.undef.trans hu) hci
    (by simp [entry,h.key]) (by simp [entry,h.value]) (by simp [entry,h.head]) (by simp [entry])
    (by simpa [entry] using h.inputQuery) (by simpa [entry] using h.inputGradient)
    (by simpa [entry] using h.carry)
  refine ⟨v,?_,Nat.succ_le_of_lt hn,hv.keyIndex,hv.valueIndex,hv.headIndex,?_,
    hv.inputKey,hv.inputValue,?_,?_,?_,?_⟩
  · rw [RealDhProgram.iteration,stepStmts.cons_some (stepStmt_assign_eq_some
      (reverseIndex_eval p n _ (by simp)))]
    exact he
  · simpa only [c,advance_expected p l r a scale n hn] using hv.carry
  · intro q e j hq ha
    by_cases hqn : q = n
    · subst q
      exact hv.history e j ha
    · have hq' : q < n := by omega
      have hqr : p.NT-1-q < p.NT := by omega
      have hf := hv.frame r.history (address p l (p.NT-1-q) e.val j.val) (by
        rintro ⟨_,f,i,hi,hoff⟩
        have heq := (hinj (p.NT-1-q) c e f j i hqr hc ha hi hoff).1
        dsimp [c] at heq
        omega)
      unfold BlockState.readMem
      rw [← hf]
      exact h.history q e j hq' ha
  · intro region off hf
    apply (h.frame region off hf).trans
    apply hv.frame region off
    rintro ⟨hr,e,j,ha,hoff⟩
    exact hf ⟨hr,c,e,j,hc,ha,hoff⟩
  · exact hv.pids.trans h.pids
  · exact hv.undef.trans h.undef

def statement (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : Stmt :=
  .forRange "j" 0 p.NT 1 (RealDhProgram.iteration p r scale swapped)

theorem executes (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (scale : ℝ) (swapped : Bool) (hu : initial.undef = fun _ _ => 0)
    (hs : RealDhStep.Separate r) (hinj : HistoryNoCollision p l)
    (u : BlockState) (h : Holds initial p l r a scale 0 u) :
    ∃ v, stepStmt (statement p r scale swapped) u = some v ∧ Holds initial p l r a scale p.NT v := by
  obtain ⟨last,v,he,hend,hv⟩ := forRange_inv (idx := "j") (start := 0) (stop := p.NT) (step := 1)
    (P := Holds initial p l r a scale) (body := RealDhProgram.iteration p r scale swapped)
    (by decide : 1 ≠ 0) h
    (fun n u hn h => iteration initial p l r a scale swapped hu hs hinj n u hn h)
  have hlast : last = p.NT := Nat.le_antisymm hv.count_le hend
  subst last
  exact ⟨v,he,hv⟩

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhLoop

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhInit
open VeriTile.Triton
open RealHMath (Layout Launch)
open RealHTiles (stateTile)
open RealDhMath RealDhProgram RealDhTiles
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def registers (p : Layout) (u : BlockState) : BlockState :=
  (((u.setReg "i_k" .nat [] (Tile.scalar (u.pids 0))).setReg
    "i_v" .nat [] (Tile.scalar (u.pids 1))).setReg
    "i_bh" .nat [] (Tile.scalar (u.pids 2))).setReg
    "b_dh" .real [p.BK,p.BV] (stateTile p (fun _ _ => 0))

theorem executes (p : Layout) (u : BlockState) :
    stepStmts (initialization p) u = some (registers p u) := by
  unfold initialization
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 u))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BK,p.BV] (.const 0)) _ =
    some (stateTile p (fun _ _ => 0)) from by simp [evalOp_full,evalOp_const]; rfl)),stepStmts.nil]
  rfl

theorem invariant (p : Layout) (r : Regions) (a : Inputs) (scale : ℝ) (u : BlockState)
    (hq : ∀ off, u.readMem r.query off = a.query off)
    (hd : ∀ off, u.readMem r.incoming off = a.incoming off) :
    RealDhLoop.Holds u p (launch u) r a scale 0 (registers p u) := by
  refine ⟨Nat.zero_le _,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · simp [registers,launch]
  · simp [registers,launch]
  · simp [registers,launch]
  · simp only [registers,BlockState.setReg_same,Option.some.injEq]
    apply congrArg (stateTile p)
    funext e j
    simp [RealDhTiles.expected]
  · exact hq
  · exact hd
  · intro q _ _ hq; omega
  · intro _ _ _; rfl
  · rfl
  · rfl

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhInit

/-! Complete real execution of both typed reverse-state sources. Each stored
chunk is the contraction sum over strictly later chunks of the initial arrays.
Dimensions and strides stay symbolic, including the empty NT=0 loop. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhExecution
open VeriTile.Triton
open RealHMath (Layout Launch historyAddress keyIndex valueIndex)
open RealDhMath RealDhProgram RealDhTiles
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

def Outputs (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (v : BlockState) : Prop :=
  ∀ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT → RealDhStore.active p l e.val j.val →
    v.readMem r.history (historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val)) =
      closed p l a scale c (keyIndex p l e.val) (valueIndex p l j.val)

theorem executes (p : Layout) (r : Regions) (query incoming : Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool) (a : Inputs) (u : BlockState)
    (hu : u.undef = fun _ _ => 0) (hs : RealDhStep.Separate r)
    (hinj : RealDhLoop.HistoryNoCollision p (launch u))
    (hq : ∀ off, u.readMem r.query off = a.query off)
    (hd : ∀ off, u.readMem r.incoming off = a.incoming off) :
    ∃ v, exec (source p r query incoming out scale swapped).eraseDType.toAlgKernel u = some v ∧
      Outputs p (launch u) r a scale v ∧
      BlockState.WriteWithin (RealDhLoop.footprint p (launch u) r) u v := by
  let l := launch u
  let start := RealDhInit.registers p u
  have hei := RealDhInit.executes p u
  obtain ⟨v,hel,hv⟩ := RealDhLoop.executes u p l r a scale swapped hu hs hinj start
    (RealDhInit.invariant p r a scale u hq hd)
  refine ⟨v,?_,?_,hv.frame⟩
  · rw [exec,source_body,body,stepStmts.append_some hei]
    change stepStmts [RealDhLoop.statement p r scale swapped] start = some v
    rw [stepStmts.cons_some hel,stepStmts.nil]
  · intro c e j hc ha
    have hr := reverse_index p.NT c hc
    have he := hv.history (p.NT-1-c) e j hr.1 ha
    have hf := closed_at_reverse p l a scale (p.NT-1-c)
      (keyIndex p l e.val) (valueIndex p l j.val) hr.1
    rw [hr.2] at he hf
    rw [hf]
    simpa only [RealDhLoop.address,RealDhStore.address,RealDhStep.historyBase,historyAddress,
      Nat.mul_one,expected] using he

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhExecution

#guard_msgs (drop info) in
#auditModuleAxioms
