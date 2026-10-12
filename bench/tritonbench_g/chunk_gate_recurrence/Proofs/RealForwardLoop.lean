import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardStep

/-! The full forward loop derives its carry from independent input arrays.
Each store extends the proved output prefix; inputs and every other tagged
memory cell are preserved. No materialized previous-state buffer is assumed. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardLoop
open VeriTile.Triton RealProgram RealForwardProgram RealForwardStep
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def footprint (p : Layout) (l : Launch) (O : RegionName) : WriteFootprint :=
  fun cell => cell.1 = O ∧ ∃ (t : Nat) (e : Fin p.BK) (j : Fin p.BV),
    t < p.N ∧ cell.2 = address p l t e.val j.val

theorem scatter_frame (p : Layout) (l : Launch) (O : RegionName) (t : Nat) (ht : t < p.N)
    (f : Values p) (u : BlockState) :
    BlockState.WriteWithin (footprint p l O) u (scatter p l O t f u) := by
  have h := BlockState.foldl_writeMem_prop_writeWithin (P := footprint p l O) O
    (fun i : TileIndex [p.BK,p.BV] => address p l t i.1.val i.2.1.val)
    (fun i => f i.1 i.2.1) (fun _ => True) (TileShape.allIndices [p.BK,p.BV]) u
    (fun i _ _ => ⟨rfl,t,i.1,i.2.1,ht,rfl⟩)
  simpa only [scatter, ite_true] using h

structure Holds (initial : BlockState) (p : Layout) (l : Launch) (S D O : RegionName)
    (a : Inputs p) (hasSeed : Bool) (c : Nat) (u : BlockState) : Prop where
  count_le : c ≤ p.N - 1
  source : u.regs .ptr [p.BK,p.BV] "S" = some (pointer p l S c)
  gate : u.regs .ptr [] "D" = some (gatePointer p l D c)
  target : u.regs .ptr [p.BK,p.BV] "O" = some (pointer p l O (c+1))
  carry : u.regs .real [p.BK,p.BV] "acc" = some (tile p (RealMath.forward a hasSeed c))
  inputSource : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
    u.readMem S (address p l t e.val j.val) = a.source t e j
  inputGate : ∀ t, t < p.N → u.readMem D (gateAddress p l t) = a.gate t
  output : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t ≤ c →
    u.readMem O (address p l t e.val j.val) = RealMath.forward a hasSeed t e j
  frame : BlockState.WriteWithin (footprint p l O) initial u

theorem next_carry (p : Layout) (a : Inputs p) (hasSeed : Bool) (c : Nat) :
    next p a c (RealMath.forward a hasSeed c) = RealMath.forward a hasSeed (c+1) := by
  funext e j
  exact (RealMath.forward_step a hasSeed c e j).symm

theorem iteration (initial : BlockState) (p : Layout) (l : Launch) (S D O : RegionName)
    (a : Inputs p) (hasSeed : Bool) (hSO : S ≠ O) (hDO : D ≠ O) (hi : NoCollision p l)
    (c : Nat) (u : BlockState) (hc : c < p.N-1)
    (h : Holds initial p l S D O a hasSeed c u) :
    ∃ v, stepStmts (loopBody p) (u.setReg "_i" .nat [] (Tile.scalar c)) = some v ∧
      Holds initial p l S D O a hasSeed (c+1) v := by
  let entry := u.setReg "_i" .nat [] (Tile.scalar c)
  let acc := RealMath.forward a hasSeed c
  let pre := beforeStore p a c acc entry
  let v := result p l S D O a c acc entry
  have he := RealForwardStep.executes p l S D O a c acc entry
    (by simp [entry,h.source]) (by simp [entry,h.gate]) (by simp [entry,h.target])
    (by simpa [entry,acc] using h.carry)
    (by intro e j; simpa [entry] using h.inputSource c e j (by omega))
    (by simpa [entry] using h.inputGate c (by omega))
  refine ⟨v,he,Nat.succ_le_of_lt hc,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · simp [v,result]
  · simp [v,result]
  · simp [v,result,Nat.add_assoc]
  · simp [v,result,scatter,beforeStore,acc,next_carry]
  · intro t e j ht
    change (scatter p l O (c+1) (next p a c acc) pre).readMem S _ = _
    rw [scatter, BlockState.scatter_preserves_other_region _ _ _ _ hSO]
    simpa [pre,beforeStore,entry] using h.inputSource t e j ht
  · intro t ht
    change (scatter p l O (c+1) (next p a c acc) pre).readMem D _ = _
    rw [scatter, BlockState.scatter_preserves_other_region _ _ _ _ hDO]
    simpa [pre,beforeStore,entry] using h.inputGate t ht
  · intro t e j ht
    change (scatter p l O (c+1) (next p a c acc) pre).readMem O _ = _
    by_cases heq : t = c+1
    · subst t
      rw [scatter_read p l O (c+1) _ pre hi e j]
      exact congrFun (congrFun (next_carry p a hasSeed c) e) j
    · rw [scatter_other_chunk p l O (c+1) t _ pre hi (Ne.symm heq) e j]
      simpa [pre,beforeStore,entry] using h.output t e j (by omega)
  · have hs := scatter_frame p l O (c+1) (by omega) (next p a c acc) pre
    intro rg off hn
    exact (h.frame rg off hn).trans (hs rg off hn)

theorem stop_eval (p : Layout) (u : BlockState) :
    evalOp (stop p) u = some (Tile.scalar (p.N-1)) := by
  simp only [stop, evalOp_sub, evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
  rfl

theorem executes (initial : BlockState) (p : Layout) (l : Launch) (S D O : RegionName)
    (a : Inputs p) (hasSeed : Bool) (hSO : S ≠ O) (hDO : D ≠ O) (hi : NoCollision p l)
    (u : BlockState) (h : Holds initial p l S D O a hasSeed 0 u) :
    ∃ v, stepStmt (statement p) u = some v ∧
      Holds initial p l S D O a hasSeed (p.N-1) v := by
  obtain ⟨final,v,he,hf,hv⟩ := forRangeDyn_inv
    (P := Holds initial p l S D O a hasSeed)
    (evalOp_constNat 0 u) (stop_eval p u) (evalOp_constNat 1 u) (by decide) h
    (fun c u hc h => iteration initial p l S D O a hasSeed hSO hDO hi c u hc h)
  have hfinal : final = p.N-1 := Nat.le_antisymm hv.count_le hf
  subst final
  exact ⟨v,he,hv⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardLoop
