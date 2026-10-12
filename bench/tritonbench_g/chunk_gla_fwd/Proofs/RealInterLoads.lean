import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterPointers

/-! Actual inter-subblock loads, including the differently strided flat gate
reference. Zero extension is conditional on a zero unspecified-value oracle;
the source still has omitted block-pointer padding. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterLoads
open VeriTile.Triton RealInterMath RealInterProgram RealInterPointers
open ChunkGatedAttention.RealHPointers (load2_eval)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def queryTile (p : Layout) (l : Launch) (a : Nat → ℝ) (kb : Nat) : Tile .real [p.BC,p.BK] :=
  Tile.mat (fun i e => some (matrixValue p l a (rowStart p l+i.val) (kb*p.BK+e.val)))
def keyTile (p : Layout) (l : Launch) (a : Nat → ℝ) (kb : Nat) : Tile .real [p.BK,p.BC] :=
  Tile.mat (fun e j => some (matrixValue p l a (columnStart p l+j.val) (kb*p.BK+e.val)))
def referenceTile (p : Layout) (l : Launch) (a : Inputs) (kb : Nat) : Tile .real [p.BK] :=
  Tile.vec (fun e => some (reference p l a (kb*p.BK+e.val)))

theorem query_load (p : Layout) (l : Launch) (r : RegionName) (a : Nat → ℝ)
    (kb : Nat) (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a off) (hu : ∀ off, u.undef r off = 0)
    (hp : u.regs .blockPtr [p.BC,p.BK] name = some (queryPointer p l r kb)) :
    evalOp (load2 p.BC p.BK name) u = some (queryTile p l a kb) := by
  erw [load2_eval r u name (l.head*p.headStride) p.T p.K p.BC p.BK p.timeStride 1 (rowStart p l) (kb*p.BK) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,e,⟨⟩⟩
  simp only [queryTile,Tile.mat,matrixValue,matrixAddress,hm,hu,Nat.mul_one]
  split <;> rfl

theorem key_load (p : Layout) (l : Launch) (r : RegionName) (a : Nat → ℝ)
    (kb : Nat) (name : RegName) (u : BlockState)
    (hm : ∀ off, u.readMem r off = a off) (hu : ∀ off, u.undef r off = 0)
    (hp : u.regs .blockPtr [p.BK,p.BC] name = some (keyPointer p l r kb)) :
    evalOp (load2 p.BK p.BC name) u = some (keyTile p l a kb) := by
  erw [load2_eval r u name (l.head*p.headStride) p.K p.T p.BK p.BC 1 p.timeStride (kb*p.BK) (columnStart p l) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp only [keyTile,Tile.mat,matrixValue,matrixAddress,hm,hu,Nat.mul_one]
  by_cases h : columnStart p l+j.val < p.T ∧ kb*p.BK+e.val < p.K
  · simp only [h,h.1,h.2,and_self,if_true]
    congr 2 <;> omega
  · simp only [show ¬ (kb*p.BK+e.val < p.K ∧ columnStart p l+j.val < p.T) by tauto,
      h,if_false]
    rfl

theorem reference_load (p : Layout) (l : Launch) (r : RegionName) (a : Inputs)
    (kb : Nat) (u : BlockState) (hm : ∀ off, u.readMem r off = a.gate off)
    (hp : u.regs .ptr [p.BK] "p_gn" = some (referencePointer p l r kb))
    (hk : u.regs .bool [p.BK] "m_k" = some (keyMask p kb)) :
    evalOp (referenceLoad p) u = some (referenceTile p l a kb) := by
  simp only [referenceLoad,evalOp,hp,hk,Option.bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,⟨⟩⟩
  simp only [referenceTile,reference,referencePointer,keyMask,Tile.vec,
    BlockState.readMemValue_real,hm,decide_eq_true_eq]
  split <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterLoads
