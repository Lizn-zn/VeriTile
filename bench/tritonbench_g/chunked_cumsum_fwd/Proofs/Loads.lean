import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Initialization
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Tiles

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Loads
open VeriTile.Triton Math Program Initialization Tiles
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

def inputValues (p : Layout) (l : Launch) (a : Inputs) (i : Fin p.BH) (j : Fin p.BC) : ℝ :=
  if activeInput p l i.val j.val then a.dt (dtAddress p l i.val j.val) else 0
def biasValues (p : Layout) (l : Launch) (a : Inputs) (i : Fin p.BH) : ℝ :=
  if head p l i.val < p.heads then a.bias (head p l i.val*p.biasHead) else 0

theorem headMask_eval (p : Layout) (l : Launch) (u : BlockState)
    (hh : u.regs .nat [p.BH] "offs_h" = some (heads p l)) :
    evalOp (headMask p) u = some (Tile.vec (fun i => decide (head p l i.val < p.heads))) := by
  simp only [headMask,headOp,evalOp.eq_def,hh,Option.bind_eq_bind,Option.bind_some,Option.pure_def]
  rfl

theorem mask_eval (p : Layout) (l : Launch) (u : BlockState) (lim : Op .nat []) (n : Nat)
    (hh : u.regs .nat [p.BH] "offs_h" = some (heads p l))
    (hc : u.regs .nat [p.BC] "offs_c" = some (columns p))
    (hn : evalOp lim u = some (Tile.scalar n)) :
    evalOp (mask p lim) u = some (Tile.mat (fun i j => decide (head p l i.val < p.heads ∧ j.val < n))) := by
  unfold mask
  rw [evalOp]
  simp only [evalOp_lt,evalOp_constNat,headOp,columnOp]
  erw [evalOp_expandDim,evalOp_ref,hh,evalOp_expandDim,evalOp_ref,hc]
  simp only [hn,Option.bind_eq_bind,Option.bind_some,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [heads,columns,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex]

theorem inputMask_eval (p : Layout) (l : Launch) (u : BlockState)
    (hh : u.regs .nat [p.BH] "offs_h" = some (heads p l))
    (hc : u.regs .nat [p.BC] "offs_c" = some (columns p))
    (hl : u.regs .nat [] "chunk_size_limit" = some (Tile.scalar (min p.chunk (p.sequence-l.chunk*p.chunk)))) :
    evalOp (inputMask p) u = some (Tile.mat (fun i j => decide (activeInput p l i.val j.val))) :=
  mask_eval p l u _ _ hh hc (by simpa only [evalOp_ref] using hl)

theorem outputMask_eval (p : Layout) (l : Launch) (u : BlockState)
    (hh : u.regs .nat [p.BH] "offs_h" = some (heads p l))
    (hc : u.regs .nat [p.BC] "offs_c" = some (columns p)) :
    evalOp (outputMask p) u = some (Tile.mat (fun i j => decide (activeOutput p l i.val j.val))) :=
  mask_eval p l u _ _ hh hc (evalOp_constNat _ _)

theorem loadDt_eval (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (u : BlockState)
    (hp : u.regs .ptr [p.BH,p.BC] "dt_ptrs" = some (dtPointers p l r))
    (hm : evalOp (inputMask p) u = some (Tile.mat (fun i j => decide (activeInput p l i.val j.val))))
    (ha : ∀ off, u.readMem r.dt off = a.dt off) :
    evalOp (loadDt p) u = some (matrix p (inputValues p l a)) := by
  unfold loadDt
  rw [evalOp]
  simp only [evalOp_ref,hp,hm,Option.bind_eq_bind,Option.bind_some]
  simp only [zero,evalOp.eq_def,Option.bind_eq_bind,Option.bind_some,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [matrix,inputValues,dtPointers,BlockState.readMemValue,ha]
  split_ifs <;> rfl

theorem loadRate_eval (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (u : BlockState)
    (hp : u.regs .ptr [p.BH] "A_ptrs" = some (ratePointers p l r))
    (hm : evalOp (headMask p) u = some (Tile.vec (fun i => decide (head p l i.val < p.heads))))
    (ha : ∀ off, u.readMem r.rate off = a.rate off) :
    evalOp (loadRate p) u = some (row p (fun i => rate p l a i.val)) := by
  unfold loadRate
  rw [evalOp]
  simp only [evalOp_ref,hp,hm,Option.bind_eq_bind,Option.bind_some]
  simp only [evalOp.eq_def,Option.bind_eq_bind,Option.bind_some,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp [row,rate,ratePointers,BlockState.readMemValue,ha]
  split_ifs <;> rfl

theorem loadBias_eval (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (u : BlockState)
    (hh : u.regs .nat [p.BH] "offs_h" = some (heads p l))
    (ha : ∀ off, u.readMem r.bias off = a.bias off) :
    evalOp (loadBias p r) u = some (row p (biasValues p l a)) := by
  simp only [loadBias,evalOp.eq_def,headOp,hh,headMask,Option.bind_eq_bind,Option.bind_some,Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp [row,biasValues,heads,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,BlockState.readMemValue,ha]
  split_ifs <;> rfl

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Loads
