import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealSplitProgram
import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHPointers

/-! Actual pointers, masks and loads. The cached matrix tiles retain arbitrary
padding. The streamed flat loads use explicit zero fill and advance by K. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitMemory
open VeriTile.Triton RealSplitMath RealSplitProgram
open ChunkGatedAttention.RealHPointers (matrixPointer make2 times_eval load2_eval)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false
structure Grid (l : Launch) (s : BlockState) : Prop where
  chunk : s.regs .nat [] "i_t" = some (Tile.scalar l.chunk)
  head : s.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  row : s.regs .nat [] "i_i" = some (Tile.scalar l.block)
  column : s.regs .nat [] "i_j" = some (Tile.scalar l.block)
  feature : s.regs .nat [] "i_k" = some (Tile.scalar l.feature)
  heads : s.regs .nat [] "n_bh" = some (Tile.scalar l.heads)

def queryPointer (p : Layout) (l : Launch) (r : RegionName) :=
  matrixPointer r (l.head*p.headStride) p.T p.K p.BC p.BK p.timeStride 1 (rowStart p l) (l.feature*p.BK)
def keyPointer (p : Layout) (l : Launch) (r : RegionName) (j : Nat) : Tile .ptr [p.BK] :=
  Tile.vec (fun e => (r,keyAddress p l j e.val))
def offsets (p : Layout) (l : Launch) : Tile .nat [p.BC] :=
  Tile.vec (fun i => outputAddress p l i.val 0)
def indices (n : Nat) : Tile .nat [n] := Tile.vec (fun e => e.val)
def keyIndices (p : Layout) (l : Launch) : Tile .nat [p.BK] := Tile.vec (fun e => featureIndex p l e.val)
def keyMask (p : Layout) (l : Launch) : Tile .bool [p.BK] := Tile.vec (fun e => decide (featureIndex p l e.val < p.K))
def rowMask (p : Layout) (l : Launch) : Tile .bool [p.BC] := Tile.vec (fun i => decide (rowActive p l i.val))
def matrixTile (p : Layout) (l : Launch) (a padding : Nat → ℝ) : Tile .real [p.BC,p.BK] :=
  Tile.mat (fun i e => some (matrixValue p l a padding i.val e.val))
def keyTile (p : Layout) (l : Launch) (a : Nat → ℝ) (j : Nat) : Tile .real [p.BK] :=
  Tile.vec (fun e => some (keyValue p l a j e.val))

theorem time_eval (p : Layout) (s : BlockState) (name : RegName) (chunk block : Nat)
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar chunk))
    (hb : s.regs .nat [] name = some (Tile.scalar block)) :
    evalOp (timeOp p name) s = some (Tile.scalar (chunk*p.BT+block*p.BC)) := by
  simp only [timeOp,times,evalOp_add,evalOp_mul,evalOp_ref,evalOp_constNat,ht,hb,Option.bind_some,Option.bind_eq_bind]
  rfl

theorem query_eval (p : Layout) (l : Launch) (r : RegionName) (s : BlockState) (hg : Grid l s) :
    evalOp (queryPtr p r) s = some (queryPointer p l r) :=
  make2 s r _ _ _ _ _ _ _ _ _ _ _ _ (times_eval s _ _ _ hg.head)
    (time_eval p s "i_i" _ _ hg.chunk hg.row) (times_eval s _ _ _ hg.feature)

theorem key_eval (p : Layout) (l : Launch) (r : RegionName) (s : BlockState)
    (hg : Grid l s) (hk : s.regs .nat [p.BK] "o_k" = some (keyIndices p l)) :
    evalOp (keyPtr p r) s = some (keyPointer p l r 0) := by
  simp only [keyPtr,timeOp,times,evalOp,evalOp_ref,hg.chunk,hg.head,hg.column,hk,
    Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,⟨⟩⟩
  simp only [keyPointer,keyAddress,featureIndex,keyIndices,rowStart,Tile.vec,Tile.scalar,indices,
    Tile.ptrAdd,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,
    Nat.zero_add,Nat.add_zero,Region.cast_self]

theorem offsets_eval (p : Layout) (l : Launch) (s : BlockState) (hg : Grid l s) :
    evalOp (offsetsOp p) s = some (offsets p l) := by
  simp only [offsetsOp,rowsOp,timeOp,times,evalOp,evalOp_ref,hg.chunk,hg.head,hg.row,hg.column,hg.feature,hg.heads,
    Option.bind_some,Option.bind_eq_bind]
  rfl

theorem indices_eval (p : Layout) (l : Launch) (s : BlockState)
    (hk : s.regs .nat [] "i_k" = some (Tile.scalar l.feature)) :
    evalOp (.add .nat .scalarL (times "i_k" p.BK) (.arange p.BK)) s = some (keyIndices p l) := by
  simp only [times,evalOp_add,evalOp_mul,evalOp_ref,evalOp_constNat,evalOp_arange,hk,
    Option.bind_some,Option.bind_eq_bind]
  rfl

theorem key_mask_eval (p : Layout) (l : Launch) (s : BlockState)
    (hk : s.regs .nat [p.BK] "o_k" = some (keyIndices p l)) :
    evalOp (.lt .nat .scalarR (.ref .nat [p.BK] "o_k") (.constNat p.K)) s = some (keyMask p l) := by
  simp only [evalOp,evalOp_ref,hk,Option.bind_some,Option.bind_eq_bind]
  rfl

theorem row_mask_eval (p : Layout) (l : Launch) (s : BlockState) (hg : Grid l s) :
    evalOp (.lt .nat .scalarR (rowsOp p) (.constNat p.T)) s = some (rowMask p l) := by
  simp only [rowsOp,timeOp,times,evalOp,evalOp_ref,hg.chunk,hg.row,Option.bind_some,Option.bind_eq_bind]
  rfl

theorem matrix_load (p : Layout) (l : Launch) (r : RegionName) (a padding : Nat → ℝ)
    (name : RegName) (s : BlockState)
    (hm : ∀ off, s.readMem r off = a off) (hu : ∀ off, s.undef r off = padding off)
    (hp : s.regs .blockPtr [p.BC,p.BK] name = some (queryPointer p l r)) :
    evalOp (load2 p name) s = some (matrixTile p l a padding) := by
  erw [load2_eval r s name (l.head*p.headStride) p.T p.K p.BC p.BK p.timeStride 1 (rowStart p l) (l.feature*p.BK) hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,e,⟨⟩⟩
  simp only [matrixTile,Tile.mat,matrixValue,matrixAddress,featureIndex,rowActive,hm,hu,Nat.mul_one,Nat.zero_add]
  split <;> rfl

theorem key_load (p : Layout) (l : Launch) (r : RegionName) (a : Nat → ℝ)
    (j : Nat) (name : RegName) (s : BlockState) (hm : ∀ off, s.readMem r off = a off)
    (hp : s.regs .ptr [p.BK] name = some (keyPointer p l r j))
    (hk : s.regs .bool [p.BK] "m_k" = some (keyMask p l)) :
    evalOp (keyLoad p name) s = some (keyTile p l a j) := by
  simp only [keyLoad,evalOp,hp,hk,Option.bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,⟨⟩⟩
  simp only [keyTile,keyValue,keyPointer,keyMask,Tile.vec,BlockState.readMemValue_real,hm,decide_eq_true_eq]
  split <;> rfl

theorem advance_eval (p : Layout) (l : Launch) (r : RegionName) (j : Nat) (name : RegName) (s : BlockState)
    (hp : s.regs .ptr [p.BK] name = some (keyPointer p l r j)) :
    evalOp (advance p name) s = some (keyPointer p l r (j+1)) := by
  simp only [advance,evalOp,hp,Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,⟨⟩⟩
  simp only [keyPointer,keyAddress,Tile.vec,Tile.ptrAdd,Tile.scalar,Broadcast.leftIndex,Broadcast.rightIndex]
  congr 1
  ring

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitMemory
