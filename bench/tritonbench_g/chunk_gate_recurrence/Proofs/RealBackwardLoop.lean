import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardStep
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardLoop

/-! Complete reverse-loop invariant: after k iterations the carry is suffix
N-k, the already-written DI/DG suffix has its independent formula, and all
input cells and other tagged memory cells are preserved. Registers needed by
the later DL postlude remain available with their original values. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardLoop
open VeriTile.Triton RealProgram RealBackwardProgram RealBackwardPointers RealBackwardStep
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

structure Fixed (p : Layout) (l : Launch) (u : BlockState) : Prop where
  head : u.regs .nat [] "offset_bh" = some (Tile.scalar l.head)
  key : u.regs .nat [] "offset_d" = some (Tile.scalar l.key)
  value : u.regs .nat [] "offset_s" = some (Tile.scalar l.value)
  numK : u.regs .nat [] "NUM_K" = some (Tile.scalar (numK p))
  numV : u.regs .nat [] "NUM_V" = some (Tile.scalar (numV p))
  offsK : u.regs .nat [p.BK] "offs_k" = some (Tile.vec fun i => i.val)
  offsV : u.regs .nat [p.BV] "offs_v" = some (Tile.vec fun i => i.val)

structure Separation (S D DI DG DS : RegionName) : Prop where
  sourceDI : S ≠ DI
  sourceDG : S ≠ DG
  gateDI : D ≠ DI
  gateDG : D ≠ DG
  incomingDI : DS ≠ DI
  incomingDG : DS ≠ DG
  outputs : DI ≠ DG

def footprint (p : Layout) (l : Launch) (DI DG : RegionName) : WriteFootprint := fun cell =>
  (cell.1 = DI ∧ ∃ (t : Nat) (e : Fin p.BK) (j : Fin p.BV),
    t < p.N-1 ∧ cell.2 = address p l t e.val j.val) ∨
  (cell.1 = DG ∧ ∃ t, t < p.N-1 ∧ cell.2 = gradientAddress p l t)

structure Holds (initial : BlockState) (p : Layout) (l : Launch) (S D DI DG DS : RegionName)
    (a : Inputs p) (k : Nat) (u : BlockState) : Prop where
  count_le : k ≤ p.N-1
  fixed : Fixed p l u
  source : u.regs .ptr [p.BK,p.BV] "S" = some (backwardPointer p l S (p.N-2) k)
  target : u.regs .ptr [p.BK,p.BV] "DI" = some (backwardPointer p l DI (p.N-2) k)
  incoming : u.regs .ptr [p.BK,p.BV] "DS" = some (backwardPointer p l DS (p.N-1) k)
  gradient : u.regs .ptr [] "DG" = some (backwardGradient p l DG k)
  gate : u.regs .ptr [] "D" = some (backwardGate p l D k)
  carry : u.regs .real [p.BK,p.BV] "Dacc" = some (tile p (RealMath.reverse a p.N (p.N-k)))
  inputSource : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N-1 →
    u.readMem S (address p l t e.val j.val) = a.source t e j
  inputIncoming : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
    u.readMem DS (address p l t e.val j.val) = a.incoming t e j
  inputGate : ∀ t, t < p.N → u.readMem D (gateAddress p l t) = a.gate t
  outputDI : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), p.N-1-k ≤ t → t < p.N-1 →
    u.readMem DI (address p l t e.val j.val) = RealMath.reverse a p.N (t+1) e j
  outputDG : ∀ t, p.N-1-k ≤ t → t < p.N-1 →
    u.readMem DG (gradientAddress p l t) = RealMath.gateGradient a p.N t
  frame : BlockState.WriteWithin (footprint p l DI DG) initial u

theorem next_carry (p : Layout) (a : Inputs p) (k : Nat) (hk : k < p.N) :
    next p a (p.N-1-k) (RealMath.reverse a p.N (p.N-k)) =
      RealMath.reverse a p.N (p.N-1-k) := by
  have hsucc : p.N-k = (p.N-1-k)+1 := by omega
  funext e j
  rw [hsucc]
  exact (RealMath.reverse_step a p.N (p.N-1-k) (by omega) e j).symm

theorem stored_other (p : Layout) (l : Launch) (DI DG r : RegionName) (a : Inputs p)
    (k : Nat) (acc : Values p) (u : BlockState) (off : Nat) (hi : r ≠ DI) (hg : r ≠ DG) :
    (stored p l DI DG a k acc u).readMem r off = u.readMem r off := by
  unfold stored scatter
  rw [BlockState.scatter_preserves_other_region _ _ _ _ hi,
    BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hg]
  rfl

theorem stored_frame (p : Layout) (l : Launch) (DI DG : RegionName) (a : Inputs p)
    (k : Nat) (acc : Values p) (u : BlockState) (hk : k < p.N-1) :
    BlockState.WriteWithin (footprint p l DI DG) u (stored p l DI DG a k acc u) := by
  let pre := beforeStore p a k acc u
  let f := next p a (p.N-1-k) acc
  have hg := BlockState.writeMem_writeWithin (P := footprint p l DI DG) (s := pre)
    DG (gradientAddress p l (p.N-2-k)) (contraction p a (p.N-2-k) f)
    (Or.inr ⟨rfl,p.N-2-k,by omega,rfl⟩)
  have hi := BlockState.foldl_writeMem_prop_writeWithin (P := footprint p l DI DG) DI
    (fun i : TileIndex [p.BK,p.BV] => address p l (p.N-2-k) i.1.val i.2.1.val)
    (fun i => f i.1 i.2.1) (fun _ => True) (TileShape.allIndices [p.BK,p.BV])
    (pre.writeMem DG (gradientAddress p l (p.N-2-k)) (contraction p a (p.N-2-k) f))
    (fun i _ _ => Or.inl ⟨rfl,p.N-2-k,i.1,i.2.1,by omega,rfl⟩)
  intro r off hn
  exact (hg r off hn).trans (hi r off hn)

theorem iteration (initial : BlockState) (p : Layout) (l : Launch) (S D DI DG DS : RegionName)
    (a : Inputs p) (sep : Separation S D DI DG DS) (hi : NoCollision p l)
    (hgi : Function.Injective (gradientAddress p l))
    (k : Nat) (u : BlockState) (hk : k < p.N-1)
    (h : Holds initial p l S D DI DG DS a k u) :
    ∃ v, stepStmts (loopBody p) (u.setReg "_i" .nat [] (Tile.scalar k)) = some v ∧
      Holds initial p l S D DI DG DS a (k+1) v := by
  let entry := u.setReg "_i" .nat [] (Tile.scalar k)
  let acc := RealMath.reverse a p.N (p.N-k)
  let pre := beforeStore p a k acc entry
  let f := next p a (p.N-1-k) acc
  let v := result p l S D DI DG DS a k acc entry
  have hnext : f = RealMath.reverse a p.N (p.N-1-k) := next_carry p a k (by omega)
  have hidx : p.N-1-k = (p.N-2-k)+1 := by omega
  have he := RealBackwardStep.executes p l S D DI DG DS a k acc entry hk
    (by simp [entry,h.source]) (by simp [entry,h.target]) (by simp [entry,h.incoming])
    (by simp [entry,h.gradient]) (by simp [entry,h.gate]) (by simpa [entry,acc] using h.carry)
    (by simp [entry,h.fixed.numK]) (by simp [entry,h.fixed.numV])
    (by intro e j; simpa [entry] using h.inputSource (p.N-2-k) e j (by omega))
    (by intro e j; simpa [entry] using h.inputIncoming (p.N-1-k) e j (by omega))
    (by simpa [entry] using h.inputGate (p.N-1-k) (by omega))
  refine ⟨v,he,Nat.succ_le_of_lt hk,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · rcases h.fixed with ⟨hh,hkey,hval,hnk,hnv,hek,hev⟩
    constructor <;> simp [v,result,stored,scatter,beforeStore,entry,hh,hkey,hval,hnk,hnv,hek,hev]
  · simp [v,result]
  · simp [v,result]
  · simp [v,result]
  · simp [v,result]
  · simp [v,result]
  · have hn : p.N-(k+1) = p.N-1-k := by omega
    simp [v,result,stored,scatter,beforeStore,acc,next_carry p a k (by omega),hn]
  · intro t e j ht
    change (stored p l DI DG a k acc entry).readMem S _ = _
    rw [stored_other p l DI DG S a k acc entry _ sep.sourceDI sep.sourceDG]
    exact h.inputSource t e j ht
  · intro t e j ht
    change (stored p l DI DG a k acc entry).readMem DS _ = _
    rw [stored_other p l DI DG DS a k acc entry _ sep.incomingDI sep.incomingDG]
    exact h.inputIncoming t e j ht
  · intro t ht
    change (stored p l DI DG a k acc entry).readMem D _ = _
    rw [stored_other p l DI DG D a k acc entry _ sep.gateDI sep.gateDG]
    exact h.inputGate t ht
  · intro t e j hlo hhi
    change (scatter p l DI (p.N-2-k) f
      (pre.writeMem DG (gradientAddress p l (p.N-2-k)) (contraction p a (p.N-2-k) f))).readMem DI _ = _
    by_cases ht : t = p.N-2-k
    · subst t
      rw [scatter_read p l DI _ _ _ hi,hnext,hidx]
    · rw [scatter_other_chunk p l DI _ t _ _ hi (Ne.symm ht),
        BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ sep.outputs]
      exact h.outputDI t e j (by omega) hhi
  · intro t hlo hhi
    change (scatter p l DI (p.N-2-k) f
      (pre.writeMem DG (gradientAddress p l (p.N-2-k)) (contraction p a (p.N-2-k) f))).readMem DG _ = _
    rw [scatter,BlockState.scatter_preserves_other_region _ _ _ _ (Ne.symm sep.outputs)]
    by_cases ht : t = p.N-2-k
    · subst t
      simp only [BlockState.writeMem_readMem,↓reduceIte]
      rw [hnext,hidx]
      rfl
    · rw [BlockState.writeMem_readMem_of_ne_offset _ _ _ _ _ _ (fun heq => ht (hgi heq))]
      exact h.outputDG t (by omega) hhi
  · have hs := stored_frame p l DI DG a k acc entry hk
    intro r off hn
    exact (h.frame r off hn).trans (hs r off hn)

theorem executes (initial : BlockState) (p : Layout) (l : Launch) (S D DI DG DS : RegionName)
    (a : Inputs p) (sep : Separation S D DI DG DS) (hi : NoCollision p l)
    (hgi : Function.Injective (gradientAddress p l))
    (u : BlockState) (h : Holds initial p l S D DI DG DS a 0 u) :
    ∃ v, stepStmt (statement p) u = some v ∧
      Holds initial p l S D DI DG DS a (p.N-1) v := by
  obtain ⟨final,v,he,hf,hv⟩ := forRangeDyn_inv
    (P := Holds initial p l S D DI DG DS a)
    (evalOp_constNat 0 u) (RealForwardLoop.stop_eval p u) (evalOp_constNat 1 u) (by decide) h
    (fun k u hk h => iteration initial p l S D DI DG DS a sep hi hgi k u hk h)
  have hfinal : final = p.N-1 := Nat.le_antisymm hv.count_le hf
  subst final
  exact ⟨v,he,hv⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardLoop
