import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardFlatCore
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardFlatState

/-! Direct execution of the five pointer decrements in flat memory. DS and D
remain live through the DL postlude, so their values commute exactly even in
the final iteration. S/DI/DG commute exactly before that iteration; afterwards
only their unused register values may differ. No head-index restriction or
no-underflow premise for dead pointers is introduced. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardFlatTail
open VeriTile.Triton RealProgram RealBackwardProgram RealBackwardPointers RealBackwardFlatState
open RealForwardProgram (stride)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 300000
set_option linter.unusedSimpArgs false

def regionResult (p : Layout) (l : Launch) (S D DI DG DS : RegionName) (k : Nat) (u : BlockState) : BlockState :=
  let u1 := u.setReg "S" .ptr [p.BK,p.BV] (backwardPointer p l S (p.N-2) (k+1))
  let u2 := u1.setReg "DI" .ptr [p.BK,p.BV] (backwardPointer p l DI (p.N-2) (k+1))
  let u3 := u2.setReg "DS" .ptr [p.BK,p.BV] (backwardPointer p l DS (p.N-1) (k+1))
  let u4 := u3.setReg "DG" .ptr [] (backwardGradient p l DG (k+1))
  u4.setReg "D" .ptr [] (backwardGate p l D (k+1))

def result (A : FlatAlloc) (p : Layout) (l : Launch) (S D DI DG DS : RegionName) (k : Nat) (u : BlockState) : BlockState :=
  let u1 := (A.flattenState u).setReg "S" .ptr [p.BK,p.BV]
    (Tile.ptrSub .scalarR (A.trTile (backwardPointer p l S (p.N-2) k)) (Tile.scalar (p.K*p.V)))
  let u2 := u1.setReg "DI" .ptr [p.BK,p.BV]
    (Tile.ptrSub .scalarR (A.trTile (backwardPointer p l DI (p.N-2) k)) (Tile.scalar (p.K*p.V)))
  let u3 := u2.setReg "DS" .ptr [p.BK,p.BV] (A.trTile (backwardPointer p l DS (p.N-1) (k+1)))
  let u4 := u3.setReg "DG" .ptr []
    (Tile.ptrSub .nil (A.trTile (backwardGradient p l DG k)) (Tile.scalar (numK p*numV p)))
  u4.setReg "D" .ptr [] (A.trTile (backwardGate p l D (k+1)))

theorem translated_decrement (A : FlatAlloc) {n : Nat} {sh : TileShape} (ptr : Tile .ptr (n :: sh)) (amount : Nat)
    (h : ∀ i, amount ≤ (ptr.data i).2) :
    Tile.ptrSub .scalarR (A.trTile ptr) (Tile.scalar amount) =
      A.trTile (Tile.ptrSub .scalarR ptr (Tile.scalar amount)) := by
  apply Tile.ext
  intro i
  simp [Tile.ptrSub,Broadcast.leftIndex,Broadcast.rightIndex,FlatAlloc.trTile,
    FlatAlloc.trCarrier,FlatAlloc.addr,Nat.add_sub_assoc (h i)]

theorem translated_scalar_decrement (A : FlatAlloc) (ptr : Tile .ptr []) (amount : Nat)
    (h : ∀ i, amount ≤ (ptr.data i).2) :
    Tile.ptrSub .nil (A.trTile ptr) (Tile.scalar amount) =
      A.trTile (Tile.ptrSub .nil ptr (Tile.scalar amount)) := by
  apply Tile.ext
  intro i
  simp [Tile.ptrSub,Broadcast.leftIndex,Broadcast.rightIndex,FlatAlloc.trTile,
    FlatAlloc.trCarrier,FlatAlloc.addr,Nat.add_sub_assoc (h i)]

theorem pointer_room (p : Layout) (l : Launch) (r : RegionName) (start k : Nat) (hk : k+1 ≤ start) :
    ∀ i, p.K*p.V ≤ ((backwardPointer p l r start k).data i).2 := by
  rw [backwardPointer_live p l r start k (by omega)]
  rintro ⟨e,j,⟨⟩⟩
  change p.K*p.V ≤ address p l (start-k) e.val j.val
  have hc : start-k = (start-k-1)+1 := by omega
  rw [hc,address_succ]
  omega

theorem gradient_room (p : Layout) (l : Launch) (r : RegionName) (k : Nat) (hk : k+1 ≤ p.N-2) :
    ∀ i, numK p*numV p ≤ ((backwardGradient p l r k).data i).2 := by
  rw [backwardGradient_live p l r k (by omega)]
  intro i
  change numK p*numV p ≤ gradientAddress p l (p.N-2-k)
  have hc : 1 ≤ p.N-2-k := by omega
  have hm := Nat.mul_le_mul_right (numK p*numV p) hc
  dsimp [gradientAddress]
  simp only [Nat.mul_assoc,Nat.one_mul] at hm ⊢
  omega

theorem gate_room (p : Layout) (l : Launch) (r : RegionName) (k : Nat) (hk : k+1 ≤ p.N-1) :
    ∀ i, 1 ≤ ((backwardGate p l r k).data i).2 := by
  rw [backwardGate_live p l r k (by omega)]
  intro i
  change 1 ≤ gateAddress p l (p.N-1-k)
  dsimp [gateAddress]
  omega

theorem pointer_exact (A : FlatAlloc) (p : Layout) (l : Launch) (r : RegionName)
    (start k : Nat) (hk : k+1 ≤ start) :
    Tile.ptrSub .scalarR (A.trTile (backwardPointer p l r start k)) (Tile.scalar (p.K*p.V)) =
      A.trTile (backwardPointer p l r start (k+1)) := by
  rw [translated_decrement A _ _ (pointer_room p l r start k hk),pointer_decrement]

theorem gradient_exact (A : FlatAlloc) (p : Layout) (l : Launch) (r : RegionName)
    (k : Nat) (hk : k+1 ≤ p.N-2) :
    Tile.ptrSub .nil (A.trTile (backwardGradient p l r k)) (Tile.scalar (numK p*numV p)) =
      A.trTile (backwardGradient p l r (k+1)) := by
  rw [translated_scalar_decrement A _ _ (gradient_room p l r k hk),gradient_decrement]

theorem gate_exact (A : FlatAlloc) (p : Layout) (l : Launch) (r : RegionName)
    (k : Nat) (hk : k+1 ≤ p.N-1) :
    Tile.ptrSub .nil (A.trTile (backwardGate p l r k)) (Tile.scalar 1) =
      A.trTile (backwardGate p l r (k+1)) := by
  rw [translated_scalar_decrement A _ _ (gate_room p l r k hk),gate_decrement]

theorem agree (A : FlatAlloc) (p : Layout) (l : Launch) (S D DI DG DS : RegionName) (k : Nat) (u : BlockState) :
    Agree (result A p l S D DI DG DS k u) (A.flattenState (regionResult p l S D DI DG DS k u)) := by
  simp only [result,regionResult,FlatAlloc.flattenState_setReg]
  apply Agree.setReg
  apply Agree.setDead (name := "DG") (hn := by simp [Live])
  apply Agree.setReg
  apply Agree.setDead (name := "DI") (hn := by simp [Live])
  apply Agree.setDead (name := "S") (hn := by simp [Live])
  exact Agree.refl _

theorem exact_before_last (A : FlatAlloc) (p : Layout) (l : Launch) (S D DI DG DS : RegionName)
    (k : Nat) (u : BlockState) (hk : k+1 < p.N-1) :
    result A p l S D DI DG DS k u = A.flattenState (regionResult p l S D DI DG DS k u) := by
  simp only [result,regionResult,FlatAlloc.flattenState_setReg,
    pointer_exact A p l S (p.N-2) k (by omega),pointer_exact A p l DI (p.N-2) k (by omega),
    gradient_exact A p l DG k (by omega)]

theorem decrement_eval {sh : TileShape} (bc : Broadcast sh [] sh) (name : RegName) (ptr : Tile .ptr sh) (amount : Op .nat [])
    (n : Nat) (u : BlockState) (hp : u.regs .ptr sh name = some ptr)
    (hn : evalOp amount u = some (Tile.scalar n)) :
    evalOp (.ptrSub bc (.ref .ptr sh name) amount) u = some (Tile.ptrSub bc ptr (Tile.scalar n)) := by
  rw [evalOp.eq_def]
  simp only [evalOp_ref,hp,hn,Option.bind_some,Option.bind_eq_bind]

theorem executes (A : FlatAlloc) (p : Layout) (l : Launch) (S D DI DG DS : RegionName)
    (k : Nat) (u : BlockState) (hk : k < p.N-1)
    (hs : u.regs .ptr [p.BK,p.BV] "S" = some (backwardPointer p l S (p.N-2) k))
    (hdi : u.regs .ptr [p.BK,p.BV] "DI" = some (backwardPointer p l DI (p.N-2) k))
    (hds : u.regs .ptr [p.BK,p.BV] "DS" = some (backwardPointer p l DS (p.N-1) k))
    (hdg : u.regs .ptr [] "DG" = some (backwardGradient p l DG k))
    (hd : u.regs .ptr [] "D" = some (backwardGate p l D k))
    (hnk : u.regs .nat [] "NUM_K" = some (Tile.scalar (numK p)))
    (hnv : u.regs .nat [] "NUM_V" = some (Tile.scalar (numV p))) :
    stepStmts (A.flattenStmts (RealBackwardFlatCore.tail p)) (A.flattenState u) =
      some (result A p l S D DI DG DS k u) := by
  have htail : A.flattenStmts (RealBackwardFlatCore.tail p) = RealBackwardFlatCore.tail p := by
    simp only [RealBackwardFlatCore.tail,stride,gradientStride,FlatAlloc.flattenStmts,
      FlatAlloc.flattenStmt,FlatAlloc.flattenOp]
  rw [htail]
  unfold RealBackwardFlatCore.tail
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (decrement_eval .scalarR "S" (A.trTile (backwardPointer p l S (p.N-2) k)) (stride p) (p.K*p.V) _
    (by simp [FlatAlloc.flattenState,FlatAlloc.trRegs,hs])
    (by simp only [stride,evalOp.eq_def,Option.bind,bind]; rfl)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (decrement_eval .scalarR "DI" (A.trTile (backwardPointer p l DI (p.N-2) k)) (stride p) (p.K*p.V) _
    (by simp [FlatAlloc.flattenState,FlatAlloc.trRegs,hdi])
    (by simp only [stride,evalOp.eq_def,Option.bind,bind]; rfl)))]
  have hdsNext := pointer_exact A p l DS (p.N-1) k (by omega)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some ((decrement_eval .scalarR "DS" _ (stride p) (p.K*p.V) _
    (by simp [FlatAlloc.flattenState,FlatAlloc.trRegs,hds])
    (by simp only [stride,evalOp.eq_def,Option.bind,bind]; rfl)).trans (congrArg some hdsNext)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (decrement_eval .nil "DG" (A.trTile (backwardGradient p l DG k)) gradientStride (numK p*numV p) _
    (by simp [FlatAlloc.flattenState,FlatAlloc.trRegs,hdg])
    (by simp [gradientStride,evalOp.eq_def,FlatAlloc.flattenState,FlatAlloc.trRegs,hnk,hnv,
      FlatAlloc.trTile,FlatAlloc.trCarrier]; rfl)))]
  have hdNext := gate_exact A p l D k (by omega)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some ((decrement_eval .nil "D" _ (.constNat 1) 1 _
    (by simp [FlatAlloc.flattenState,FlatAlloc.trRegs,hd]) (evalOp_constNat 1 _)).trans (congrArg some hdNext)))]
  rw [stepStmts.nil]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardFlatTail
