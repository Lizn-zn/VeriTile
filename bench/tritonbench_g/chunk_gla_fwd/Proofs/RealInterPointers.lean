import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterProgram
import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHPointers

/-! Evaluation of the inter-subblock source pointers and K-lane mask. The
flat gate reference retains its K stride, independently of the matrix time
stride. No allocation or padding value is inferred from the pointer shape. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterPointers
open VeriTile.Triton RealInterMath RealInterProgram
open ChunkGatedAttention.RealHPointers (matrixPointer make2 times_eval)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

structure Grid (p : Layout) (l : Launch) (s : BlockState) : Prop where
  chunk : s.regs .nat [] "i_t" = some (Tile.scalar l.chunk)
  head : s.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  row : s.regs .nat [] "i_i" = some (Tile.scalar (rowBlock p l))
  column : s.regs .nat [] "i_j" = some (Tile.scalar (columnBlock p l))

def queryPointer (p : Layout) (l : Launch) (r : RegionName) (kb : Nat) :=
  matrixPointer r (l.head*p.headStride) p.T p.K p.BC p.BK p.timeStride 1 (rowStart p l) (kb*p.BK)
def keyPointer (p : Layout) (l : Launch) (r : RegionName) (kb : Nat) :=
  matrixPointer r (l.head*p.headStride) p.K p.T p.BK p.BC 1 p.timeStride (kb*p.BK) (columnStart p l)
def outputPointer (p : Layout) (l : Launch) (r : RegionName) :=
  matrixPointer r (l.head*p.T*p.BT) p.T p.BT p.BC p.BC p.BT 1 (rowStart p l) (columnBlock p l*p.BC)
def referencePointer (p : Layout) (l : Launch) (r : RegionName) (kb : Nat) : Tile .ptr [p.BK] :=
  Tile.vec (fun e => (r,referenceAddress p l (kb*p.BK+e.val)))
def keyIndices (p : Layout) (kb : Nat) : Tile .nat [p.BK] := Tile.vec (fun e => kb*p.BK+e.val)
def keyMask (p : Layout) (kb : Nat) : Tile .bool [p.BK] := Tile.vec (fun e => decide (kb*p.BK+e.val < p.K))

theorem time_eval (p : Layout) (s : BlockState) (name : RegName) (chunk block : Nat)
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar chunk))
    (hb : s.regs .nat [] name = some (Tile.scalar block)) :
    evalOp (timeOp p name) s = some (Tile.scalar (chunk*p.BT+block*p.BC)) := by
  simp only [timeOp,times,evalOp_add,evalOp_mul,evalOp_ref,evalOp_constNat,ht,hb,Option.bind_some,Option.bind_eq_bind]
  rfl

theorem query_eval (p : Layout) (l : Launch) (r : RegionName) (kb : Nat) (s : BlockState)
    (hg : Grid p l s) (hk : s.regs .nat [] "i_k" = some (Tile.scalar kb)) :
    evalOp (queryPtr p r) s = some (queryPointer p l r kb) :=
  make2 s r _ _ _ _ _ _ _ _ _ _ _ _ (times_eval s _ _ _ hg.head)
    (time_eval p s "i_i" _ _ hg.chunk hg.row) (times_eval s _ _ _ hk)

theorem key_eval (p : Layout) (l : Launch) (r : RegionName) (kb : Nat) (s : BlockState)
    (hg : Grid p l s) (hk : s.regs .nat [] "i_k" = some (Tile.scalar kb)) :
    evalOp (keyPtr p r) s = some (keyPointer p l r kb) :=
  make2 s r _ _ _ _ _ _ _ _ _ _ _ _ (times_eval s _ _ _ hg.head)
    (times_eval s _ _ _ hk) (time_eval p s "i_j" _ _ hg.chunk hg.column)

theorem output_eval (p : Layout) (l : Launch) (r : RegionName) (s : BlockState) (hg : Grid p l s) :
    evalOp (outputPtr p r) s = some (outputPointer p l r) := by
  apply make2
  · simp only [evalOp_mul,times,evalOp_ref,evalOp_constNat,hg.head,Option.bind_some,Option.bind_eq_bind]
    rfl
  · exact time_eval p s "i_i" _ _ hg.chunk hg.row
  · exact times_eval s _ _ _ hg.column

theorem indices_eval (p : Layout) (kb : Nat) (s : BlockState)
    (hk : s.regs .nat [] "i_k" = some (Tile.scalar kb)) :
    evalOp (.add .nat .scalarL (times "i_k" p.BK) (.arange p.BK)) s = some (keyIndices p kb) := by
  simp only [times,evalOp_add,evalOp_mul,evalOp_ref,evalOp_constNat,evalOp_arange,hk,
    Option.bind_some,Option.bind_eq_bind]
  rfl

theorem mask_eval (p : Layout) (kb : Nat) (s : BlockState)
    (hk : s.regs .nat [p.BK] "o_k" = some (keyIndices p kb)) :
    evalOp (.lt .nat .scalarR (.ref .nat [p.BK] "o_k") (.constNat p.K)) s = some (keyMask p kb) := by
  simp only [evalOp,evalOp_ref,hk,Option.bind_some,Option.bind_eq_bind]
  rfl

theorem reference_eval (p : Layout) (l : Launch) (r : RegionName) (kb : Nat) (s : BlockState)
    (hg : Grid p l s) (hk : s.regs .nat [p.BK] "o_k" = some (keyIndices p kb)) :
    evalOp (referencePtr p r) s = some (referencePointer p l r kb) := by
  simp only [referencePtr,timeOp,times,evalOp,evalOp_ref,hg.chunk,hg.head,hg.row,hk,
    Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,⟨⟩⟩
  simp only [referencePointer,referenceAddress,rowStart,keyIndices,Tile.vec,Tile.scalar,
    Tile.ptrAdd,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,Nat.zero_add,Region.cast_self]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterPointers
