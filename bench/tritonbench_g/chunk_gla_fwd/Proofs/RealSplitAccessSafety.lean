import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealSplitMemory
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterAccessSafety

/-! Safety of the source-active memory accesses. Matrix and flat layouts
have separate bounds; masked-out lanes need no allocated input cell. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitAccessSafety
open VeriTile.Triton RealSplitMath RealSplitProgram RealSplitMemory
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def MatrixBounds (p : Layout) (l : Launch) (r : RegionName) (bounds : RegionBounds) : Prop :=
  ∀ (i : Fin p.BC) (e : Fin p.BK), rowActive p l i.val → featureIndex p l e.val < p.K →
    matrixAddress p l i.val e.val < bounds r
def KeyBounds (p : Layout) (l : Launch) (r : RegionName) (j : Nat) (bounds : RegionBounds) : Prop :=
  ∀ e : Fin p.BK, featureIndex p l e.val < p.K → keyAddress p l j e.val < bounds r
def OutputBounds (p : Layout) (l : Launch) (r : RegionName) (j : Nat) (bounds : RegionBounds) : Prop :=
  ∀ i : Fin p.BC, rowActive p l i.val → outputAddress p l i.val j < bounds r

theorem matrix_safe (p : Layout) (l : Launch) (r : RegionName) (name : RegName)
    (bounds : RegionBounds) (s : BlockState) (hb : MatrixBounds p l r bounds)
    (hp : s.regs .blockPtr [p.BC,p.BK] name = some (queryPointer p l r)) :
    (load2 p name).SafeAt bounds s := by
  apply RealInterAccessSafety.load2_safe r (l.head*p.headStride) p.T p.K p.BC p.BK p.timeStride 1 (rowStart p l) (l.feature*p.BK) name bounds s
  · intro i e he
    simpa [matrixAddress,featureIndex,Nat.zero_add,Nat.mul_one] using hb i e he.1 (by simpa using he.2)
  · exact hp

theorem key_safe (p : Layout) (l : Launch) (r : RegionName) (j : Nat) (name : RegName)
    (bounds : RegionBounds) (s : BlockState) (hb : KeyBounds p l r j bounds)
    (hp : s.regs .ptr [p.BK] name = some (keyPointer p l r j))
    (hm : s.regs .bool [p.BK] "m_k" = some (keyMask p l)) :
    (keyLoad p name).SafeAt bounds s := by
  rw [keyLoad,Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt],by simp [Op.SafeAt],?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he idx ha
  rw [evalOp_ref,hp] at he
  cases he
  rcases idx with ⟨e,⟨⟩⟩
  have hk : featureIndex p l e.val < p.K := by simpa [MaskOpt.Active,evalOp_ref,hm,keyMask,Tile.vec] using ha
  exact hb e hk

theorem output_safe (p : Layout) (l : Launch) (out : RegionName) (j : Nat)
    (bounds : RegionBounds) (s : BlockState) (hb : OutputBounds p l out j bounds)
    (ho : s.regs .nat [p.BC] "o_A" = some (offsets p l))
    (hj : s.regs .nat [] "j" = some (Tile.scalar j))
    (hm : s.regs .bool [p.BC] "m_A" = some (rowMask p l)) :
    Stmt.TraceSafe bounds (.store .real [p.BC] (.region out (outputOffsets p))
      (.ref .real [p.BC] "b_A") (.mask (.ref .bool [p.BC] "m_A"))) s := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt,outputOffsets,Op.SafeAt],by simp [Op.SafeAt],by simp [MaskOpt.SafeAt,Op.SafeAt],?_⟩
  intro os he idx ha
  simp only [outputOffsets,evalOp_add,evalOp_ref,ho,hj,Option.bind_some,Option.bind_eq_bind] at he
  cases he
  rcases idx with ⟨i,⟨⟩⟩
  have hi : rowActive p l i.val := by simpa [MaskOpt.Active,evalOp_ref,hm,rowMask,Tile.vec] using ha
  simpa [offsets,outputAddress,Tile.vec,-Tile.scalar_eta,Tile.scalar,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add] using hb i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitAccessSafety
