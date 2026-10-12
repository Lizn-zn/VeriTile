import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHTiles

/-! Masked state-matrix stores used for both pre-update history and final state.
The base and both strides distinguish their layouts. Readback requires only
active lanes to have distinct addresses; the frame preserves complete cells. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHStore
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHTiles
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def active (p : Layout) (l : Launch) (e j : Nat) : Prop :=
  keyIndex p l e < p.K ∧ valueIndex p l j < p.V
instance (p : Layout) (l : Launch) (e j : Nat) : Decidable (active p l e j) :=
  inferInstanceAs (Decidable (_ ∧ _))

def address (p : Layout) (l : Launch) (base rowStride colStride e j : Nat) : Nat :=
  base + keyIndex p l e * rowStride + valueIndex p l j * colStride

def pointer (p : Layout) (l : Launch) (out : RegionName) (base rowStride colStride : Nat) :
    Tile .blockPtr [p.BK,p.BV] :=
  ⟨fun _ => BlockPtr.mk out base [p.K,p.V] [p.BK,p.BV] [rowStride,colStride]
    [l.key * p.BK,l.value * p.BV]⟩

def scatter (p : Layout) (l : Launch) (out : RegionName) (base rowStride colStride : Nat)
    (frozen : StateValue p) (u : BlockState) : BlockState :=
  (TileShape.allIndices [p.BK,p.BV]).foldl (fun st idx =>
    if active p l idx.1.val idx.2.1.val then
      st.writeMem out (address p l base rowStride colStride idx.1.val idx.2.1.val) (frozen idx.1 idx.2.1)
    else st) u

def footprint (p : Layout) (l : Launch) (out : RegionName) (base rowStride colStride : Nat) : WriteFootprint :=
  fun cell => cell.1 = out ∧ ∃ (e : Fin p.BK) (j : Fin p.BV), active p l e.val j.val ∧
    cell.2 = address p l base rowStride colStride e.val j.val

def NoCollision (p : Layout) (l : Launch) (base rowStride colStride : Nat) : Prop :=
  ∀ (e f : Fin p.BK) (i j : Fin p.BV), active p l e.val i.val → active p l f.val j.val →
    address p l base rowStride colStride e.val i.val = address p l base rowStride colStride f.val j.val → e = f ∧ i = j

theorem store_eval (p : Layout) (l : Launch) (out : RegionName) (base rowStride colStride : Nat)
    (frozen : StateValue p) (nm : RegName) (u : BlockState)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p frozen))
    (hp : u.regs .blockPtr [p.BK,p.BV] nm = some (pointer p l out base rowStride colStride)) :
    stepStmt (.store .real [p.BK,p.BV] (.blockPtr (.ref .blockPtr [p.BK,p.BV] nm) [0,1])
      (.ref .real [p.BK,p.BV] "b_h") .none) u = some (scatter p l out base rowStride colStride frozen u) := by
  unfold stepStmt scatter
  simp only [evalOp_ref, hh, hp, Option.bind, Option.map]
  apply congrArg some
  apply congrArg (fun f => List.foldl f u (TileShape.allIndices [p.BK,p.BV]))
  funext st idx
  rcases idx with ⟨e,j,⟨⟩⟩
  simp only [pointer, TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, Bool.true_and, active, keyIndex, valueIndex, address]
  by_cases h : l.key * p.BK + e.val < p.K ∧ l.value * p.BV + j.val < p.V
  · simp only [h, decide_true, if_true, BlockState.writeMemTyped_real]; rfl
  · simp only [h, decide_false, Bool.false_eq_true, if_false]

theorem readback (p : Layout) (l : Launch) (out : RegionName) (base rowStride colStride : Nat)
    (frozen : StateValue p) (u : BlockState) (hinj : NoCollision p l base rowStride colStride)
    (e : Fin p.BK) (j : Fin p.BV) (ha : active p l e.val j.val) :
    (scatter p l out base rowStride colStride frozen u).readMem out (address p l base rowStride colStride e.val j.val) = frozen e j := by
  exact BlockState.scatter_readback_prop_masked_nd_of_true (region := out) u
    (fun idx : TileIndex [p.BK,p.BV] => address p l base rowStride colStride idx.1.val idx.2.1.val)
    (fun idx => frozen idx.1 idx.2.1) (fun idx => active p l idx.1.val idx.2.1.val)
    (e,j,PUnit.unit) ha (by
      rintro ⟨f,i,⟨⟩⟩ hi heq
      obtain ⟨hf,hj⟩ := hinj f e i j hi ha heq
      subst f; subst i; rfl)

theorem frame (p : Layout) (l : Launch) (out : RegionName) (base rowStride colStride : Nat)
    (frozen : StateValue p) (u : BlockState) :
    BlockState.WriteWithin (footprint p l out base rowStride colStride) u (scatter p l out base rowStride colStride frozen u) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro idx _ ha
  exact ⟨rfl,idx.1,idx.2.1,ha,rfl⟩

@[simp] theorem scatter_undef (p : Layout) (l : Launch) (out : RegionName)
    (base rowStride colStride : Nat) (frozen : StateValue p) (u : BlockState) :
    (scatter p l out base rowStride colStride frozen u).undef = u.undef := by
  unfold scatter
  generalize TileShape.allIndices [p.BK,p.BV] = lanes
  induction lanes generalizing u with
  | nil => rfl
  | cons i is ih =>
    simp only [List.foldl_cons,ih]
    split <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHStore
