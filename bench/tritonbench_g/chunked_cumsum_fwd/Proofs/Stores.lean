import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Preparation
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Scan

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Stores
open VeriTile.Triton Math Program Initialization Tiles Loads Preparation
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

abbrev Address (p : Layout) := Fin p.BH → Fin p.BC → Nat
abbrev Value (p : Layout) := Fin p.BH → Fin p.BC → ℝ

def scatter (p : Layout) (l : Launch) (out : RegionName) (address : Address p)
    (value : Value p) (u : BlockState) : BlockState :=
  (TileShape.allIndices [p.BH,p.BC]).foldl (fun st idx =>
    if activeOutput p l idx.1.val idx.2.1.val then
      st.writeMem out (address idx.1 idx.2.1) (value idx.1 idx.2.1) else st) u

def footprint (p : Layout) (l : Launch) (out : RegionName) (address : Address p) : WriteFootprint :=
  fun cell => cell.1 = out ∧ ∃ (i : Fin p.BH) (j : Fin p.BC),
    activeOutput p l i.val j.val ∧ cell.2 = address i j

def NoCollision (p : Layout) (l : Launch) (address : Address p) : Prop :=
  ∀ (i k : Fin p.BH) (j m : Fin p.BC), activeOutput p l i.val j.val → activeOutput p l k.val m.val →
    address i j = address k m → i = k ∧ j = m

theorem store_eval (p : Layout) (l : Launch) (out : RegionName) (address : Address p)
    (value : Value p) (ptrName valName : RegName) (u : BlockState)
    (hp : u.regs .ptr [p.BH,p.BC] ptrName = some (Tile.mat (fun i j => (out,address i j))))
    (hv : u.regs .real [p.BH,p.BC] valName = some (matrix p value))
    (hm : evalOp (outputMask p) u = some (Tile.mat (fun i j => decide (activeOutput p l i.val j.val)))) :
    stepStmt (.store .real [p.BH,p.BC] (.ptr (.ref .ptr [p.BH,p.BC] ptrName))
      (.ref .real [p.BH,p.BC] valName) (.mask (outputMask p))) u = some (scatter p l out address value u) := by
  unfold stepStmt scatter
  simp only [evalOp_ref,hp,hv,hm,Option.bind,Option.map]
  apply congrArg some
  apply congrArg (fun f => List.foldl f u (TileShape.allIndices [p.BH,p.BC]))
  funext st idx
  rcases idx with ⟨i,j,⟨⟩⟩
  simp [matrix,BlockState.writeMemTyped_real]

theorem readback (p : Layout) (l : Launch) (out : RegionName) (address : Address p)
    (value : Value p) (u : BlockState) (hinj : NoCollision p l address)
    (i : Fin p.BH) (j : Fin p.BC) (ha : activeOutput p l i.val j.val) :
    (scatter p l out address value u).readMem out (address i j) = value i j := by
  exact BlockState.scatter_readback_prop_masked_nd_of_true (region := out) u
    (fun idx : TileIndex [p.BH,p.BC] => address idx.1 idx.2.1)
    (fun idx => value idx.1 idx.2.1) (fun idx => activeOutput p l idx.1.val idx.2.1.val)
    (i,j,PUnit.unit) ha (by
      rintro ⟨k,m,⟨⟩⟩ hk heq
      obtain ⟨hi,hj⟩ := hinj k i m j hk ha heq
      subst k; subst m; rfl)

theorem frame (p : Layout) (l : Launch) (out : RegionName) (address : Address p)
    (value : Value p) (u : BlockState) :
    BlockState.WriteWithin (footprint p l out address) u (scatter p l out address value u) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro idx _ ha
  exact ⟨rfl,idx.1,idx.2.1,ha,rfl⟩

theorem other (p : Layout) (l : Launch) (out : RegionName) (address : Address p)
    (value : Value p) (u : BlockState) (r : RegionName) (hr : r ≠ out) (off : Nat) :
    (scatter p l out address value u).readMem r off = u.readMem r off :=
  BlockState.scatter_prop_masked_preserves_other_region out _ _ _ r hr off _ u

theorem ready (h : Ready p l r u) (out : RegionName) (address : Address p) (value : Value p) :
    Ready p l r (scatter p l out address value u) := by
  rcases h with ⟨hh,hc,hl,hd,ha,ho,hs⟩
  constructor <;> simpa only [scatter,BlockState.foldl_writeMem_prop_masked_regs]

theorem registers (p : Layout) (l : Launch) (out : RegionName) (address : Address p)
    (value : Value p) (u : BlockState) :
    (scatter p l out address value u).regs = u.regs := by
  funext d shape name
  simp only [scatter,BlockState.foldl_writeMem_prop_masked_regs]

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Stores
