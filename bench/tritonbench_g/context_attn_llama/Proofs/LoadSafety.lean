import bench.tritonbench_g.context_attn_llama.Proofs.SafeOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.LoadSafety
open VeriTile.Triton Formula Program PreludeProgram LoopProgram
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

theorem gather_safe (p : Layout) (r : Regions) (sn req last : Int) (u : BlockState) (bounds : RegionBounds)
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hr : u.regs .int [] "cur_batch_req_idx" = some (Tile.scalar req))
    (hl : u.regs .int [] "block_end_loc" = some (Tile.scalar last))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hb : ∀ j : Fin p.BN, sn+j.val < last → (p.reqRow*req+p.reqToken*(sn+j.val)).toNat < bounds r.pages) :
    ((Op.load .int (.region r.pages (pageIndex p).castIntToNat) (.maskOther (pageMask p) (.broadcast (.constInt 0) [p.BN])))).SafeAt bounds u := by
  apply SafeOps.maskedLoad (dtype := .int) (Region.cast r.pages) _ _ _ bounds u
    (Tile.vec (fun j => (p.reqRow*req+p.reqToken*(sn+j.val)).toNat))
    (Tile.vec (dtype := .bool) (fun j => decide (sn+j.val < last)))
  · simp [pageIndex,pageMask,lanes,z,evalOp.eq_def,hs,hr,hn,Tile.vec,Tile.scalar,
      tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,-Tile.scalar_eta]
  · simp [pageIndex,pageMask,lanes,z,evalOp.eq_def,hs,hl,hn,Tile.vec,Tile.scalar,
      tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,-Tile.scalar_eta]
    rfl
  · simp [pageIndex,pageMask,lanes,z,Op.SafeAt.eq_def]
  · simp [pageIndex,pageMask,lanes,z,Op.SafeAt.eq_def]
  · simp [Op.SafeAt.eq_def]
  · intro i hi
    exact hb i.1 (by simpa only [Tile.vec,decide_eq_true_eq] using hi)

theorem key_safe (p : Layout) (r : Regions) (sn req last : Int) (head : Nat)
    (u : BlockState) (bounds : RegionBounds)
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hr : u.regs .int [] "cur_batch_req_idx" = some (Tile.scalar req))
    (hl : u.regs .int [] "block_end_loc" = some (Tile.scalar last))
    (hh : u.regs .nat [] "cur_kv_head" = some (Tile.scalar head))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val)))
    (ht : ∀ j : Fin p.BN, sn+j.val < last → (p.reqRow*req+p.reqToken*(sn+j.val)).toNat < bounds r.pages)
    (hk : ∀ (d : Fin p.D) (j : Fin p.BN), sn+j.val < last →
      (KeysExecution.address p head (KeysExecution.pages p r req sn last u j.val) d.val).toNat < bounds r.key) :
    Stmt.TraceSafeList bounds (keys p r) u := by
  let s1 := u.setReg "start_n" .int [] (Tile.scalar sn)
  let pages : Fin p.BN → Int := fun j => KeysExecution.pages p r req sn last u j.val
  let s2 := s1.setReg "kv_loc" .int [p.BN] (Tile.vec pages)
  let off : Tile .int [p.D,p.BN] := Tile.mat (fun d j => KeysExecution.address p head (pages j) d.val)
  let s3 := s2.setReg "off_k" .int [p.D,p.BN] off
  have h1 : evalOp (z "start_n") u = some (Tile.scalar sn) := by simpa only [z,evalOp_ref] using hs
  have h2 : evalOp ((Op.load .int (.region r.pages (pageIndex p).castIntToNat) (.maskOther (pageMask p) (.broadcast (.constInt 0) [p.BN])))) s1 = some (Tile.vec pages) := by
    simp [pageIndex,pageMask,lanes,z,s1,pages,KeysExecution.pages,evalOp.eq_def,hs,hr,hl,hn,
      Tile.vec,Tile.scalar,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,-Tile.scalar_eta]
  have h3 : evalOp (keyOffsets p) s2 = some off := by
    simp [keyOffsets,pageKeyView,pageLanes,keyChannel,dimensions,n,s2,s1,off,pages,KeysExecution.address,evalOp.eq_def,hh,hd,
      Tile.mat,Tile.vec,Tile.scalar,Tile.remap,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
      Option.bind,Nat.cast_mul,-Tile.scalar_eta]
  unfold keys
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some h1)]
  refine ⟨by simp [Stmt.TraceSafe,z,Op.SafeAt.eq_def],?_⟩
  change Stmt.TraceSafeList bounds _ s1
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some h2)]
  refine ⟨?_,?_⟩
  · simpa only [Stmt.TraceSafe] using gather_safe p r sn req last s1 bounds (by simp [s1]) (by simpa [s1] using hr)
      (by simpa [s1] using hl) (by simpa [s1] using hn) ht
  change Stmt.TraceSafeList bounds _ s2
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some h3)]
  refine ⟨by simp [Stmt.TraceSafe,keyOffsets,pageKeyView,pageLanes,keyChannel,dimensions,n,Op.SafeAt.eq_def],?_⟩
  change Stmt.TraceSafeList bounds _ s3
  apply TraceProofs.assign_cons
  · apply SafeOps.maskedLoad r.key _ _ _ bounds s3
      (Tile.mat (fun d j => (KeysExecution.address p head (pages j) d.val).toNat))
      (Tile.mat (dtype := .bool) (fun d j => decide (sn+j.val < last)))
    · simp [evalOp.eq_def,s3,off,Tile.mat]
    · simp [keyMask,keyView,lanes,keyChannel,dimensions,z,s3,s2,s1,evalOp.eq_def,hl,hn,hd,
        Tile.mat,Tile.vec,Tile.scalar,Tile.remap,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
        Option.bind,-Tile.scalar_eta]
      rfl
    · simp [Op.SafeAt.eq_def]
    · simp [keyMask,keyView,lanes,keyChannel,dimensions,z,Op.SafeAt.eq_def]
    · simp [Op.SafeAt.eq_def]
    · intro i hi
      have hi' : sn+i.2.1.val < last := by simpa only [Tile.mat,decide_eq_true_eq] using hi
      exact hk i.1 i.2.1 hi'
  intro v
  exact Stmt.TraceSafeList.nil_intro

theorem value_safe (p : Layout) (r : Regions) (head : Nat) (pages : Fin p.BN → Int) (sn limit : Int)
    (u : BlockState) (bounds : RegionBounds)
    (hpages : u.regs .int [p.BN] "kv_loc" = some (Tile.vec pages))
    (hh : u.regs .nat [] "cur_kv_head" = some (Tile.scalar head))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hl : u.regs .int [] "block_end_loc" = some (Tile.scalar limit))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val)))
    (hbound : ∀ (j : Fin p.BN) (d : Fin p.D), sn+j.val < limit →
      (ValuesExecution.address p head pages j d).toNat < bounds r.value) :
    Stmt.TraceSafeList bounds (values p r) u := by
  let start := u
  let off : Tile .int [p.BN,p.D] := Tile.mat (fun j d => ValuesExecution.address p head pages j d)
  let s1 := start.setReg "off_v" .int [p.BN,p.D] off
  have h1 : evalOp (valueOffsets p) start = some off := by
    simp [valueOffsets,pageValueView,pageLanes,valueView,valueChannel,dimView,lanes,dimensions,n,z,start,off,ValuesExecution.address,evalOp.eq_def,hpages,hh,hs,hn,hd,
      Tile.mat,Tile.vec,Tile.scalar,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
      Option.bind,Nat.cast_mul,-Tile.scalar_eta]
  unfold values
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some h1)]
  refine ⟨by simp [Stmt.TraceSafe,valueOffsets,pageValueView,pageLanes,valueView,valueChannel,dimView,lanes,dimensions,n,z,Op.SafeAt.eq_def],?_⟩
  change Stmt.TraceSafeList bounds _ s1
  apply TraceProofs.assign_cons
  · apply SafeOps.maskedLoad r.value _ _ _ bounds s1
      (Tile.mat (fun j d => (ValuesExecution.address p head pages j d).toNat))
      (Tile.mat (dtype := .bool) (fun j d => decide (sn+j.val < limit)))
    · simp [evalOp.eq_def,s1,off,Tile.mat]
    · simp [valueMask,valueView,valueChannel,dimView,lanes,dimensions,n,z,s1,start,evalOp.eq_def,hs,hl,hn,hd,
        Tile.mat,Tile.vec,Tile.scalar,Tile.remap,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
        Option.bind,-Tile.scalar_eta]
      all_goals rfl
    · simp [Op.SafeAt.eq_def]
    · simp [valueMask,valueView,valueChannel,dimView,lanes,dimensions,z,Op.SafeAt.eq_def]
    · simp [Op.SafeAt.eq_def]
    · intro i hi
      have hi' : sn+i.1.val < limit := by simpa only [Tile.mat,decide_eq_true_eq] using hi
      exact hbound i.1 i.2.1 hi'
  intro v
  apply TraceProofs.all_safe
  simp [vec,mat,Stmt.TraceSafe,Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.LoadSafety
