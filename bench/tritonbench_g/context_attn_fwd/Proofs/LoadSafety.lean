import bench.tritonbench_g.context_attn_fwd.Proofs.SafeOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.LoadSafety
open VeriTile.Triton Formula Program PreludeProgram LoopProgram
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

theorem key_safe (p : Layout) (r : Regions) (batch head : Nat) (sn limit : Int)
    (u : BlockState) (bounds : RegionBounds)
    (hb : u.regs .nat [] "cur_batch" = some (Tile.scalar batch))
    (hh : u.regs .nat [] "cur_kv_head" = some (Tile.scalar head))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hl : u.regs .int [] "block_end_loc" = some (Tile.scalar limit))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val)))
    (hbound : ∀ (j : Fin p.BN) (d : Fin p.D), sn+j.val < limit →
      (KeysExecution.address p batch head sn d j).toNat < bounds r.key) :
    Stmt.TraceSafeList bounds (keys p r) u := by
  let start := u.setReg "start_n" .int [] (Tile.scalar sn)
  have hstart : evalOp (z "start_n") u = some (Tile.scalar sn) := by simpa only [z,evalOp_ref] using hs
  let off : Tile .int [p.D,p.BN] := Tile.mat (fun d j => KeysExecution.address p batch head sn d j)
  let s1 := start.setReg "off_k" .int [p.D,p.BN] off
  have h1 : evalOp (keyOffsets p) start = some off := by
    simp [keyOffsets,keyView,keyChannel,lanes,dimensions,n,z,start,off,KeysExecution.address,evalOp.eq_def,hb,hh,hs,hn,hd,
      Tile.mat,Tile.vec,Tile.scalar,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
      Option.bind,Nat.cast_mul,-Tile.scalar_eta]
  unfold keys
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some hstart)]
  refine ⟨by simp [Stmt.TraceSafe,z,Op.SafeAt.eq_def],?_⟩
  change Stmt.TraceSafeList bounds _ start
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some h1)]
  refine ⟨by simp [Stmt.TraceSafe,keyOffsets,keyView,keyChannel,lanes,dimensions,n,z,Op.SafeAt.eq_def],?_⟩
  change Stmt.TraceSafeList bounds _ s1
  apply TraceProofs.assign_cons
  · apply SafeOps.maskedLoad r.key _ _ _ bounds s1
      (Tile.mat (fun d j => (KeysExecution.address p batch head sn d j).toNat))
      (Tile.mat (dtype := .bool) (fun d j => decide (sn+j.val < limit)))
    · simp [evalOp.eq_def,s1,off,Tile.mat]
    · simp [keyMask,keyView,keyChannel,lanes,dimensions,n,z,s1,start,evalOp.eq_def,hs,hl,hn,hd,
        Tile.mat,Tile.vec,Tile.scalar,Tile.remap,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
        Option.bind,-Tile.scalar_eta]
      all_goals rfl
    · simp [Op.SafeAt.eq_def]
    · simp [keyMask,keyView,keyChannel,lanes,dimensions,z,Op.SafeAt.eq_def]
    · simp [Op.SafeAt.eq_def]
    · intro i hi
      have hi' : sn+i.2.1.val < limit := by simpa only [Tile.mat,decide_eq_true_eq] using hi
      exact hbound i.2.1 i.1 hi'
  intro v
  exact Stmt.TraceSafeList.nil_intro

theorem value_safe (p : Layout) (r : Regions) (batch head : Nat) (sn limit : Int)
    (u : BlockState) (bounds : RegionBounds)
    (hb : u.regs .nat [] "cur_batch" = some (Tile.scalar batch))
    (hh : u.regs .nat [] "cur_kv_head" = some (Tile.scalar head))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hl : u.regs .int [] "block_end_loc" = some (Tile.scalar limit))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val)))
    (hbound : ∀ (j : Fin p.BN) (d : Fin p.D), sn+j.val < limit →
      (ValuesExecution.address p batch head sn j d).toNat < bounds r.value) :
    Stmt.TraceSafeList bounds (values p r) u := by
  let start := u
  let off : Tile .int [p.BN,p.D] := Tile.mat (fun j d => ValuesExecution.address p batch head sn j d)
  let s1 := start.setReg "off_v" .int [p.BN,p.D] off
  have h1 : evalOp (valueOffsets p) start = some off := by
    simp [valueOffsets,valueView,valueChannel,dimView,lanes,dimensions,n,z,start,off,ValuesExecution.address,evalOp.eq_def,hb,hh,hs,hn,hd,
      Tile.mat,Tile.vec,Tile.scalar,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
      Option.bind,Nat.cast_mul,-Tile.scalar_eta]
  unfold values
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some h1)]
  refine ⟨by simp [Stmt.TraceSafe,valueOffsets,valueView,valueChannel,dimView,lanes,dimensions,n,z,Op.SafeAt.eq_def],?_⟩
  change Stmt.TraceSafeList bounds _ s1
  apply TraceProofs.assign_cons
  · apply SafeOps.maskedLoad r.value _ _ _ bounds s1
      (Tile.mat (fun j d => (ValuesExecution.address p batch head sn j d).toNat))
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
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.LoadSafety
