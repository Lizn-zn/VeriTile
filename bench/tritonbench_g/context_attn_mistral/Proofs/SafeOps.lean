import bench.tritonbench_g.context_attn_mistral.Proofs.AddressBounds
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.SafeOps
open VeriTile.Triton Formula Program PreludeProgram
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

theorem maskedPointerLoad {dtype : TileDType} {shape : TileShape}
    (ptr : Op .ptr shape) (mask : Op .bool shape) (other : Op dtype shape)
    (bounds : RegionBounds) (u : BlockState) (pointers : Tile .ptr shape) (m : Tile .bool shape)
    (hmem : ptr.MemorySafe bounds)
    (hp : evalOp ptr u = some pointers) (hm : evalOp mask u = some m)
    (hsp : ptr.SafeAt bounds u) (hsm : mask.SafeAt bounds u) (hsv : other.SafeAt bounds u)
    (hb : ∀ i, m.data i → (pointers.data i).2 < bounds (pointers.data i).1) :
    (Op.load dtype (.ptr ptr) (.maskOther mask other)).SafeAt bounds u := by
  rw [Op.SafeAt.eq_def]
  refine ⟨hsp,⟨hsm,hsv⟩,?_⟩
  unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe Op.PointerAddressesSafeOn
  refine ⟨hmem,?_⟩
  intro pointers' he i hi
  rw [hp] at he
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
    (hs : u.regs .int [] "cur_batch_seq_len" = some (Tile.scalar seq)) :
    evalOp (queryMask p) u = some (Tile.mat (dtype := .bool)
      (fun i _ => decide ((rows i : Int) < seq))) := by
  simp [queryMask,rowView,dimView,PreludeProgram.rows,dimensions,z,evalOp.eq_def,hr,hs,
    tile_elementwise,Tile.remap,Tile.scalar,Tile.vec,Tile.mat,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,-Tile.scalar_eta]
  all_goals rfl

end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.SafeOps
