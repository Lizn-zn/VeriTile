import bench.tritonbench_g.context_attn_bloom.Proofs.AddressBounds
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.SafeOps
open VeriTile.Triton Math Program RealSetup PreludeProgram
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

theorem maskedLoad {dtype : TileDType} {shape : TileShape} (r : Region dtype)
    (off : Op .nat shape) (mask : Op .bool shape) (other : Op dtype shape)
    (bounds : RegionBounds) (u : BlockState) (offsets : Tile .nat shape) (m : Tile .bool shape)
    (ho : evalOp off u = some offsets) (hm : evalOp mask u = some m)
    (hso : off.SafeAt bounds u) (hsm : mask.SafeAt bounds u) (hsv : other.SafeAt bounds u)
    (hb : ∀ i, m.data i → offsets.data i < bounds (Region.cast r)) :
    (Op.load dtype (.region r off) (.maskOther mask other)).SafeAt bounds u := by
  rw [Op.SafeAt.eq_def]
  refine ⟨hso,⟨hsm,hsv⟩,?_⟩
  intro offsets' he i hi
  rw [ho] at he
  cases he
  obtain ⟨m',he,hi⟩ := hi
  rw [hm] at he
  cases he
  exact hb i hi

theorem metadata (r : RegionName) (batch : Nat) (u : BlockState) (bounds : RegionBounds)
    (hr : u.regs .nat [] "cur_batch" = some (Tile.scalar batch)) (hb : batch < bounds r) :
    (loadMeta r).SafeAt bounds u := by
  simp only [loadMeta,Op.SafeAt.eq_def,MemAccess.SafeAt,MaskOpt.SafeAt,n,true_and]
  intro offsets he i _
  rw [evalOp_ref,hr] at he
  cases he
  exact hb

theorem queryMask_eval (p : Layout) (rows : Fin p.BM → Nat) (seq : Int) (u : BlockState)
    (hr : u.regs .nat [p.BM] "offs_m" = some (Tile.vec rows))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val)))
    (hs : u.regs .int [] "cur_batch_seq_len" = some (Tile.scalar seq)) :
    evalOp (queryMask p) u = some (Tile.mat (dtype := .bool)
      (fun i d => decide ((rows i : Int) < seq ∧ d.val < p.headDim))) := by
  simp [queryMask,rowView,dimView,PreludeProgram.rows,dimensions,z,evalOp.eq_def,hr,hd,hs,
    tile_elementwise,Tile.scalar,Tile.vec,Tile.mat,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,-Tile.scalar_eta]
  rfl

end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.SafeOps
