import bench.tritonbench_g.context_attn_bloom.Proofs.SafeOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.KeysSafety
open VeriTile.Triton Math Program RealSetup PreludeProgram LoopKeysProgram LoopKeysExecution
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem gather_safe (p : Layout) (r : Regions) (sn req last : Int) (u : BlockState) (bounds : RegionBounds)
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hr : u.regs .int [] "cur_batch_req_idx" = some (Tile.scalar req))
    (hl : u.regs .int [] "block_end_loc" = some (Tile.scalar last))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hb : ∀ j : Fin p.BN, sn+j.val < last → (p.requestRow*req+p.requestToken*(sn+j.val)).toNat < bounds r.tokens) :
    (gather p r).SafeAt bounds u := by
  unfold gather
  apply SafeOps.maskedLoad (dtype := .int) (Region.cast r.tokens) _ _ _ bounds u
    (Tile.vec (fun j => (p.requestRow*req+p.requestToken*(sn+j.val)).toNat))
    (Tile.vec (dtype := .bool) (fun j => decide (sn+j.val < last)))
  · simp [tokenIndices,lanes,z,evalOp.eq_def,hs,hr,hn,Tile.vec,Tile.scalar,
      tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,-Tile.scalar_eta]
  · simp [tokenIndices,lanes,z,evalOp.eq_def,hs,hl,hn,Tile.vec,Tile.scalar,
      tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,-Tile.scalar_eta]
    rfl
  · simp [tokenIndices,lanes,z,Op.SafeAt.eq_def]
  · simp [tokenIndices,lanes,z,Op.SafeAt.eq_def]
  · simp [Op.SafeAt.eq_def]
  · intro i hi
    exact hb i.1 (by simpa only [Tile.vec,decide_eq_true_eq] using hi)

theorem keys_safe (p : Layout) (r : Regions) (sn req last : Int) (head : Nat)
    (u : BlockState) (bounds : RegionBounds)
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hr : u.regs .int [] "cur_batch_req_idx" = some (Tile.scalar req))
    (hl : u.regs .int [] "block_end_loc" = some (Tile.scalar last))
    (hh : u.regs .nat [] "cur_kv_head" = some (Tile.scalar head))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val)))
    (ht : ∀ j : Fin p.BN, sn+j.val < last → (p.requestRow*req+p.requestToken*(sn+j.val)).toNat < bounds r.tokens)
    (hk : ∀ (d : Fin p.D) (j : Fin p.BN), sn+j.val < last → d.val < p.headDim →
      (keyAddress p head (page p r sn req last u j.val) d.val).toNat < bounds r.key) :
    Stmt.TraceSafeList bounds (keys p r) u := by
  let s1 := u.setReg "start_n" .int [] (Tile.scalar sn)
  let pages : Fin p.BN → Int := fun j => page p r sn req last u j.val
  let s2 := s1.setReg "kv_loc" .int [p.BN] (Tile.vec pages)
  let off : Tile .int [p.D,p.BN] := Tile.mat (fun d j => keyAddress p head (pages j) d.val)
  let s3 := s2.setReg "off_k" .int [p.D,p.BN] off
  have h1 : evalOp (z "start_n") u = some (Tile.scalar sn) := by simpa only [z,evalOp_ref] using hs
  have h2 : evalOp (gather p r) s1 = some (Tile.vec pages) := by
    simp [gather,tokenIndices,lanes,z,s1,pages,page,evalOp.eq_def,hs,hr,hl,hn,
      Tile.vec,Tile.scalar,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,-Tile.scalar_eta]
  have h3 : evalOp (keyOffsets p) s2 = some off := by
    simp [keyOffsets,tokenView,channelView,n,s2,s1,off,pages,keyAddress,evalOp.eq_def,hh,hd,
      Tile.mat,Tile.vec,Tile.scalar,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
      Option.bind,Nat.cast_mul,-Tile.scalar_eta]
  rw [LoopKeysProgram.source]
  unfold code
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some h1)]
  refine ⟨by simp [Stmt.TraceSafe,z,Op.SafeAt.eq_def],?_⟩
  change Stmt.TraceSafeList bounds _ s1
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some h2)]
  refine ⟨?_,?_⟩
  · simpa only [Stmt.TraceSafe] using gather_safe p r sn req last s1 bounds (by simp [s1]) (by simpa [s1] using hr)
      (by simpa [s1] using hl) (by simpa [s1] using hn) ht
  change Stmt.TraceSafeList bounds _ s2
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some h3)]
  refine ⟨by simp [Stmt.TraceSafe,keyOffsets,tokenView,channelView,n,Op.SafeAt.eq_def],?_⟩
  change Stmt.TraceSafeList bounds _ s3
  apply TraceProofs.assign_cons
  · apply SafeOps.maskedLoad r.key _ _ _ bounds s3
      (Tile.mat (fun d j => (keyAddress p head (pages j) d.val).toNat))
      (Tile.mat (dtype := .bool) (fun d j => decide (sn+j.val < last ∧ d.val < p.headDim)))
    · simp [evalOp.eq_def,s3,off,Tile.mat]
    · simp [keyMask,lanes,channelView,z,s3,s2,s1,evalOp.eq_def,hl,hn,hd,
        Tile.mat,Tile.vec,Tile.scalar,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
        Option.bind,-Tile.scalar_eta]
      rfl
    · simp [Op.SafeAt.eq_def]
    · simp [keyMask,lanes,channelView,z,Op.SafeAt.eq_def]
    · simp [Op.SafeAt.eq_def]
    · intro i hi
      have hi' : sn+i.2.1.val < last ∧ i.1.val < p.headDim := by simpa only [Tile.mat,decide_eq_true_eq] using hi
      exact hk i.1 i.2.1 hi'.1 hi'.2
  intro v
  exact Stmt.TraceSafeList.nil_intro

end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.KeysSafety
