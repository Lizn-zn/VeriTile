import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealOuterStep

/-! Induction over every outer chunk. History stores the pre-update matrix.
For corrected V, rows below the next window start already have their final
value; rows reached early by an overlapping window may be overwritten later.
The invariant therefore retains the source's full overlap behavior. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealOuterLoop
open VeriTile.Triton RealProgram
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def HistoryNoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ (c d : Nat) (e f : Fin p.BK) (i j : Fin p.BV), c < p.NT → d < p.NT →
    RealStateStore.active p l e.val i.val → RealStateStore.active p l f.val j.val →
    RealOuterStep.historyAddress p l c e.val i.val = RealOuterStep.historyAddress p l d f.val j.val →
      c = d ∧ e = f ∧ i = j

def footprint (p : Layout) (l : Launch) (r : Regions) : WriteFootprint := fun cell =>
  (cell.1 = r.history ∧ ∃ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT ∧
    RealStateStore.active p l e.val j.val ∧ cell.2 = RealOuterStep.historyAddress p l c e.val j.val) ∨
  (cell.1 = r.corrected ∧ ∃ (t : Nat) (j : Fin p.BV), activeValue p l t j.val ∧
    cell.2 = valueAddress p l t j.val)

structure Holds (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (seed : Nat → Nat → ℝ) (useSeed : Bool) (c : Nat) (u : BlockState) : Prop where
  count_le : c ≤ p.NT
  key : u.regs .nat [] "i_k" = some (Tile.scalar l.key)
  value : u.regs .nat [] "i_v" = some (Tile.scalar l.value)
  head : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  state : ∃ f : StateValue p, u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f) ∧
    RealValues.Represents p l f (RealWindow.expected p l a seed useSeed c)
  inputKey : ∀ off, u.readMem r.key off = a.key off
  inputDelta : ∀ off, u.readMem r.delta off = a.delta off
  inputValue : ∀ off, u.readMem r.value off = a.value off
  history : ∀ (q : Nat) (e : Fin p.BK) (j : Fin p.BV), q < c →
    RealStateStore.active p l e.val j.val →
    u.readMem r.history (RealOuterStep.historyAddress p l q e.val j.val) =
      RealWindow.expected p l a seed useSeed q e.val (valueIndex p l j.val)
  corrected : ∀ (q t : Nat) (j : Fin p.BV), q < c → activeValue p l t j.val →
    q * p.BT ≤ t → t < (q + 1) * p.BT →
    u.readMem r.corrected (valueAddress p l t j.val) =
      RealMath.corrected (RealValues.arrays p l a seed) p.K
        (RealWindow.expected p l a seed useSeed q) t (valueIndex p l j.val)
  frame : BlockState.WriteWithin (footprint p l r) initial u
  pids : u.pids = initial.pids
  undef : u.undef = initial.undef

theorem read_of_mem_eq (u v : BlockState) (r : RegionName) (off : Nat)
    (h : u.mem r off = v.mem r off) : u.readMem r off = v.readMem r off := by
  simp only [BlockState.readMem,h]

theorem iteration (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (seed : Nat → Nat → ℝ) (useSeed : Bool) (hu : initial.undef = fun _ _ => 0) (hk : l.key = 0) (hK : p.K ≤ p.BK)
    (hBC : 0 < p.BC) (hs : RealOuterStep.Separate r)
    (hvi : RealInnerLoop.NoCollision p l) (hhi : HistoryNoCollision p l)
    (c : Nat) (u : BlockState) (hc : c < p.NT) (h : Holds initial p l r a seed useSeed c u) :
    ∃ v, stepStmts (RealOuterStep.body p r) (u.setReg "i_t" .nat [] (Tile.scalar c)) = some v ∧
      Holds initial p l r a seed useSeed (c + 1) v := by
  obtain ⟨frozen,hfreg,hf⟩ := h.state
  let entry := u.setReg "i_t" .nat [] (Tile.scalar c)
  have hinj : RealStateStore.NoCollision p l (RealOuterStep.historyBase p l c) p.historyTime := by
    intro e f i j he hf heq
    exact (hhi c c e f i j hc hc he hf heq).2
  obtain ⟨v,he,hv⟩ := RealOuterStep.executes p l r a c frozen entry (h.undef.trans hu) hs hvi hinj
    (by simp [entry,h.key]) (by simp [entry,h.value]) (by simp [entry,h.head]) (by simp [entry])
    (by intro off; simpa [entry] using h.inputKey off)
    (by intro off; simpa [entry] using h.inputDelta off)
    (by intro off; simpa [entry] using h.inputValue off)
    (by simpa [entry] using hfreg)
  refine ⟨v,he,Nat.succ_le_of_lt hc,hv.key,hv.value,hv.head,⟨_,hv.state,?_⟩,
    hv.inputKey,hv.inputDelta,hv.inputValue,?_,?_,?_,?_,?_⟩
  · exact RealWindow.advance_represents p l a seed useSeed c frozen hk hK hf
  · intro q e j hq ha
    by_cases hqc : q = c
    · subst q
      rw [hv.history e j ha]
      apply hf e j
      · simpa [RealStateStore.active,keyIndex,hk] using ha.1
      · exact ha.2
    · have hq' : q < c := by omega
      have hframe := hv.frame r.history (RealOuterStep.historyAddress p l q e.val j.val) (by
        intro hm
        rcases hm with hh | hv'
        · rcases hh with ⟨_,f,i,hi,hoff⟩
          have heq := (hhi q c e f j i (Nat.lt_trans hq' hc) hc ha hi hoff).1
          exact hqc heq
        · exact hs.outputs hv'.1)
      rw [← read_of_mem_eq entry v _ _ hframe]
      exact h.history q e j hq' ha
  · intro q t j hq ha hlo hupper
    by_cases hqc : q = c
    · subst q
      have ht : t < c * p.BT + RealWindow.span p := by
        have hw := RealWindow.span_ge p hBC
        simp only [Nat.add_mul,Nat.one_mul] at hupper
        omega
      rw [hv.corrected t j ha hlo ht]
      exact RealValues.corrected_eq p l a seed _ frozen hk hK hf t j ha
    · have hq' : q < c := by omega
      have hbefore : t < c * p.BT := by
        have hle := Nat.mul_le_mul_right p.BT (Nat.succ_le_of_lt hq')
        exact Nat.lt_of_lt_of_le hupper hle
      have hframe := hv.frame r.corrected (valueAddress p l t j.val) (by
        intro hm
        rcases hm with hh | hv'
        · exact hs.outputs.symm hh.1
        · rcases hv' with ⟨_,t',j',ha',hb',_,hoff⟩
          have heq := (hvi t t' j j' ha ha' hoff).1
          omega)
      rw [← read_of_mem_eq entry v _ _ hframe]
      exact h.corrected q t j hq' ha hlo hupper
  · intro rg off hn
    apply (h.frame rg off hn).trans
    apply hv.frame rg off
    intro hm
    apply hn
    rcases hm with hh | hv'
    · rcases hh with ⟨hr,e,j,ha,hoff⟩
      exact Or.inl ⟨hr,c,e,j,hc,ha,hoff⟩
    · rcases hv' with ⟨hr,t,j,ha,_,_,hoff⟩
      exact Or.inr ⟨hr,t,j,ha,hoff⟩
  · exact hv.pids.trans h.pids
  · exact hv.undef.trans h.undef

def statement (p : Layout) (r : Regions) : Stmt :=
  .forRange "i_t" 0 p.NT 1 (RealOuterStep.body p r)

theorem executes (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (seed : Nat → Nat → ℝ) (useSeed : Bool) (hu : initial.undef = fun _ _ => 0) (hk : l.key = 0) (hK : p.K ≤ p.BK)
    (hBC : 0 < p.BC) (hs : RealOuterStep.Separate r)
    (hvi : RealInnerLoop.NoCollision p l) (hhi : HistoryNoCollision p l)
    (u : BlockState) (h : Holds initial p l r a seed useSeed 0 u) :
    ∃ v, stepStmt (statement p r) u = some v ∧ Holds initial p l r a seed useSeed p.NT v := by
  obtain ⟨final,v,he,hf,hv⟩ := forRange_inv (P := Holds initial p l r a seed useSeed)
    (by decide : 1 ≠ 0) h
    (fun c u hc h => iteration initial p l r a seed useSeed hu hk hK hBC hs hvi hhi c u hc h)
  have hfinal : final = p.NT := Nat.le_antisymm hv.count_le hf
  subst final
  exact ⟨v,he,hv⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealOuterLoop
