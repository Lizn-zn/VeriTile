import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterAlgebra
import VeriTile.Triton.Memory.TraceProofs

/-! Bounds for the source's actual active loads. Matrix and flat-reference
addresses are specified separately; no equality of timeStride and K is assumed.
Inactive lanes need no allocation even when their unspecified values are used. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterAccessSafety
open VeriTile.Triton RealInterMath RealInterProgram RealInterPointers
open ChunkGatedAttention.RealHPointers (matrixPointer)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def MatrixBounds (r : RegionName) (base rows cols nr nc sr sc offR offC : Nat)
    (bounds : RegionBounds) : Prop :=
  ∀ (i : Fin nr) (j : Fin nc), offR+i.val < rows ∧ offC+j.val < cols →
    base+(offR+i.val)*sr+(offC+j.val)*sc < bounds r

def QueryBounds (p : Layout) (l : Launch) (r : RegionName) (kb : Nat) :=
  MatrixBounds r (l.head*p.headStride) p.T p.K p.BC p.BK p.timeStride 1 (rowStart p l) (kb*p.BK)
def KeyBounds (p : Layout) (l : Launch) (r : RegionName) (kb : Nat) :=
  MatrixBounds r (l.head*p.headStride) p.K p.T p.BK p.BC 1 p.timeStride (kb*p.BK) (columnStart p l)
def ReferenceBounds (p : Layout) (l : Launch) (r : RegionName) (kb : Nat) (bounds : RegionBounds) :=
  ∀ e : Fin p.BK, kb*p.BK+e.val < p.K → referenceAddress p l (kb*p.BK+e.val) < bounds r

theorem load2_safe (r : RegionName) (base rows cols nr nc sr sc offR offC : Nat)
    (name : RegName) (bounds : RegionBounds) (u : BlockState)
    (hb : MatrixBounds r base rows cols nr nc sr sc offR offC bounds)
    (hp : u.regs .blockPtr [nr,nc] name = some (matrixPointer r base rows cols nr nc sr sc offR offC)) :
    (load2 nr nc name).SafeAt bounds u := by
  rw [load2,Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he idx _
  rw [evalOp_ref,hp] at he
  cases he
  rcases idx with ⟨i,j,⟨⟩⟩
  simpa only [matrixPointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,decide_eq_true_eq] using hb i j

theorem reference_safe (p : Layout) (l : Launch) (r : RegionName) (kb : Nat)
    (bounds : RegionBounds) (u : BlockState)
    (hb : ReferenceBounds p l r kb bounds)
    (hp : u.regs .ptr [p.BK] "p_gn" = some (referencePointer p l r kb))
    (hm : u.regs .bool [p.BK] "m_k" = some (keyMask p kb)) :
    (referenceLoad p).SafeAt bounds u := by
  rw [referenceLoad,Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt],by simp [Op.SafeAt],?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he idx ha
  rw [evalOp_ref,hp] at he
  cases he
  rcases idx with ⟨e,⟨⟩⟩
  have hk : kb*p.BK+e.val < p.K := by
    simpa [MaskOpt.Active,evalOp_ref,hm,keyMask,Tile.vec] using ha
  exact hb e hk

theorem query_pointer_safe (p : Layout) (r : RegionName) (bounds : RegionBounds) (s : BlockState) :
    (queryPtr p r).SafeAt bounds s := by
  simp [queryPtr,timeOp,times,Op.SafeAt]

theorem key_pointer_safe (p : Layout) (r : RegionName) (bounds : RegionBounds) (s : BlockState) :
    (keyPtr p r).SafeAt bounds s := by
  simp [keyPtr,timeOp,times,Op.SafeAt]

theorem reference_pointer_safe (p : Layout) (r : RegionName) (bounds : RegionBounds) (s : BlockState) :
    (referencePtr p r).SafeAt bounds s := by
  simp [referencePtr,timeOp,times,Op.SafeAt]

theorem query_gate_safe (p : Layout) (scale : ℝ) (bounds : RegionBounds) (s : BlockState) :
    (queryGate p scale).SafeAt bounds s := by
  simp [queryGate,Op.SafeAt,Op.SafeAt.eq_def]

theorem key_gate_safe (p : Layout)  (bounds : RegionBounds) (s : BlockState) :
    (keyGate p).SafeAt bounds s := by
  simp [keyGate,Op.SafeAt,Op.SafeAt.eq_def]

theorem update_safe (p : Layout) (swapped : Bool) (bounds : RegionBounds) (s : BlockState) :
    (RealInterAlgebra.updateOp p swapped).SafeAt bounds s := by
  cases swapped <;> simp [RealInterAlgebra.updateOp,carry,dot,Op.SafeAt,Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterAccessSafety
