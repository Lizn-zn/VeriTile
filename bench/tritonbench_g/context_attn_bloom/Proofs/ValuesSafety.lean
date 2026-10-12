import bench.tritonbench_g.context_attn_bloom.Proofs.SafeOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.ValuesSafety
open VeriTile.Triton Math Program RealSetup PreludeProgram ValuesProgram ValuesExecution
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

theorem safe (p : Layout) (r : Regions) (sn last : Int) (head : Nat) (pages : Fin p.BN → Int)
    (u : BlockState) (bounds : RegionBounds)
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hl : u.regs .int [] "block_end_loc" = some (Tile.scalar last))
    (hh : u.regs .nat [] "cur_kv_head" = some (Tile.scalar head))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val)))
    (hp : u.regs .int [p.BN] "kv_loc" = some (Tile.vec pages))
    (hb : ∀ (j : Fin p.BN) (d : Fin p.D), sn+j.val < last → d.val < p.headDim →
      (address p head (pages j) d.val).toNat < bounds r.value) :
    Stmt.TraceSafeList bounds (values p r) u := by
  let off : Tile .int [p.BN,p.D] := Tile.mat (fun j d => address p head (pages j) d.val)
  let s1 := u.setReg "off_v" .int [p.BN,p.D] off
  have h1 : evalOp (offsets p) u = some off := by
    simp [offsets,tokenView,channelView,n,off,address,evalOp.eq_def,hh,hd,hp,
      Tile.mat,Tile.vec,Tile.scalar,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
      Option.bind,Nat.cast_mul,-Tile.scalar_eta]
  rw [ValuesProgram.source]
  unfold code
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some h1)]
  refine ⟨by simp [Stmt.TraceSafe,offsets,tokenView,channelView,n,Op.SafeAt.eq_def],?_⟩
  change Stmt.TraceSafeList bounds _ s1
  apply TraceProofs.assign_cons
  · apply SafeOps.maskedLoad r.value _ _ _ bounds s1
      (Tile.mat (fun j d => (address p head (pages j) d.val).toNat))
      (Tile.mat (dtype := .bool) (fun j d => decide (sn+j.val < last ∧ d.val < p.headDim)))
    · simp [evalOp.eq_def,s1,off,Tile.mat]
    · simp [mask,channelView,z,s1,evalOp.eq_def,hs,hl,hn,hd,
        Tile.mat,Tile.vec,Tile.scalar,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
        Option.bind,-Tile.scalar_eta]
      rfl
    · simp [Op.SafeAt.eq_def]
    · simp [mask,channelView,z,Op.SafeAt.eq_def]
    · simp [Op.SafeAt.eq_def]
    · intro i hi
      have hi' : sn+i.1.val < last ∧ i.2.1.val < p.headDim := by simpa only [Tile.mat,decide_eq_true_eq] using hi
      exact hb i.1 i.2.1 hi'.1 hi'.2
  intro v
  exact Stmt.TraceSafeList.nil_intro
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.ValuesSafety
