import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHStep

/-! Every iteration of the complete H loop. The carry is the independent
closed form, history retains every earlier pre-update state, and unrelated
complete memory cells are preserved. Zero unspecified values are an explicit
premise for this zero-extended formula; the step theorem itself is unrestricted. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHLoop
open VeriTile.Triton RealHMath RealHProgram RealHTiles
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def address (p : Layout) (l : Launch) (c e j : Nat) : Nat :=
  RealHStore.address p l (RealHStep.historyBase p l c) p.historyKey p.historyValue e j

def HistoryNoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ (c d : Nat) (e f : Fin p.BK) (i j : Fin p.BV), c < p.NT → d < p.NT →
    RealHStore.active p l e.val i.val → RealHStore.active p l f.val j.val →
    address p l c e.val i.val = address p l d f.val j.val → c = d ∧ e = f ∧ i = j

def footprint (p : Layout) (l : Launch) (r : Regions) : WriteFootprint := fun cell =>
  cell.1 = r.history ∧ ∃ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT ∧
    RealHStore.active p l e.val j.val ∧ cell.2 = address p l c e.val j.val

structure Holds (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (gateK hasSeed : Bool) (c : Nat) (u : BlockState) : Prop where
  count_le : c ≤ p.NT
  key : u.regs .nat [] "i_k" = some (Tile.scalar l.key)
  value : u.regs .nat [] "i_v" = some (Tile.scalar l.value)
  head : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  carry : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p (expected p l a gateK hasSeed c))
  inputKey : ∀ off, u.readMem r.key off = a.key off
  inputValue : ∀ off, u.readMem r.value off = a.value off
  inputGate : ∀ off, u.readMem r.gate off = a.gate off
  history : ∀ (q : Nat) (e : Fin p.BK) (j : Fin p.BV), q < c → RealHStore.active p l e.val j.val →
    u.readMem r.history (address p l q e.val j.val) = expected p l a gateK hasSeed q e j
  frame : BlockState.WriteWithin (footprint p l r) initial u
  pids : u.pids = initial.pids
  undef : u.undef = initial.undef

theorem iteration (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (gateK hasSeed swapped : Bool) (hu : initial.undef = fun _ _ => 0)
    (hs : RealHStep.Separate r) (hinj : HistoryNoCollision p l)
    (c : Nat) (u : BlockState) (hc : c < p.NT) (h : Holds initial p l r a gateK hasSeed c u) :
    ∃ v, stepStmts (RealHProgram.iteration p r gateK swapped) (u.setReg "i_t" .nat [] (Tile.scalar c)) = some v ∧
      Holds initial p l r a gateK hasSeed (c+1) v := by
  let entry := u.setReg "i_t" .nat [] (Tile.scalar c)
  have hci : RealHStore.NoCollision p l (RealHStep.historyBase p l c) p.historyKey p.historyValue := by
    intro e f i j he hf heq
    exact (hinj c c e f i j hc hc he hf heq).2
  obtain ⟨v,he,hv⟩ := RealHStep.executes p l r a (fun _ _ => 0) c gateK swapped
    (expected p l a gateK hasSeed c) entry hs (h.undef.trans hu) hci
    (by simp [entry,h.key]) (by simp [entry,h.value]) (by simp [entry,h.head]) (by simp [entry])
    (by simpa [entry] using h.inputKey) (by simpa [entry] using h.inputValue)
    (by simpa [entry] using h.inputGate) (by simpa [entry] using h.carry)
  refine ⟨v,he,Nat.succ_le_of_lt hc,hv.keyIndex,hv.valueIndex,hv.headIndex,?_,
    hv.inputKey,hv.inputValue,hv.inputGate,?_,?_,?_,?_⟩
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

def statement (p : Layout) (r : Regions) (gateK swapped : Bool) : Stmt :=
  .forRange "i_t" 0 p.NT 1 (RealHProgram.iteration p r gateK swapped)

theorem executes (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (gateK hasSeed swapped : Bool) (hu : initial.undef = fun _ _ => 0)
    (hs : RealHStep.Separate r) (hinj : HistoryNoCollision p l)
    (u : BlockState) (h : Holds initial p l r a gateK hasSeed 0 u) :
    ∃ v, stepStmt (statement p r gateK swapped) u = some v ∧ Holds initial p l r a gateK hasSeed p.NT v := by
  obtain ⟨last,v,he,hend,hv⟩ := forRange_inv (P := Holds initial p l r a gateK hasSeed)
    (by decide : 1 ≠ 0) h
    (fun c u hc h => iteration initial p l r a gateK hasSeed swapped hu hs hinj c u hc h)
  have hlast : last = p.NT := Nat.le_antisymm hv.count_le hend
  subst last
  exact ⟨v,he,hv⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHLoop
