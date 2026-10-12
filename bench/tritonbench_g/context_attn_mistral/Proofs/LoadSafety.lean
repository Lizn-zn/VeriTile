import bench.tritonbench_g.context_attn_mistral.Proofs.SafeOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.LoadSafety
open VeriTile.Triton Formula Program PreludeProgram LoopProgram
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

theorem key_safe (p : Layout) (r : Regions) (head : Nat) (start sn limit : Int)
    (u : BlockState) (bounds : RegionBounds)
    (hp : u.regs .ptr [p.D,p.BN] "k_ptrs" = some (Tile.mat (fun d j =>
      (Region.cast r.key,j.val*p.kRow+head*p.kHead+d.val*p.kDim))))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hb : u.regs .int [] "cur_batch_in_all_start_index" = some (Tile.scalar start))
    (hl : u.regs .int [] "cur_batch_seq_len" = some (Tile.scalar limit))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hbound : ∀ (j : Fin p.BN) (d : Fin p.D), sn+j.val < limit →
      KeysExecution.address p head start sn d j < bounds r.key) :
    Stmt.TraceSafeList bounds (keys p r) u := by
  let state := u.setReg "start_n" .int [] (Tile.scalar sn)
  have hstart : evalOp (z "start_n") u = some (Tile.scalar sn) := by simpa only [z,evalOp_ref] using hs
  unfold keys
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some hstart)]
  refine ⟨by simp [Stmt.TraceSafe,z,Op.SafeAt.eq_def],?_⟩
  change Stmt.TraceSafeList bounds _ state
  apply TraceProofs.assign_cons
  · apply SafeOps.maskedPointerLoad _ _ _ bounds state
      (Tile.mat (fun d j => (Region.cast r.key,KeysExecution.address p head start sn d j)))
      (Tile.mat (dtype := .bool) (fun d j => decide (sn+j.val < limit)))
    · simp [keyAdvance,z,Op.MemorySafe]
    · simp [evalOp.eq_def,keyAdvance,z,state,hp,hb,hs,KeysExecution.address,
        Tile.mat,Tile.scalar,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
        Option.bind,-Tile.scalar_eta]
    · simp [keyMask,keyView,lanes,z,state,evalOp.eq_def,hs,hl,hn,
        Tile.mat,Tile.vec,Tile.scalar,Tile.remap,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
        Option.bind,-Tile.scalar_eta]
      all_goals rfl
    · simp [keyAdvance,z,Op.SafeAt.eq_def]
    · simp [keyMask,keyView,lanes,z,Op.SafeAt.eq_def]
    · simp [Op.SafeAt.eq_def]
    · intro i hi
      have hi' : sn+i.2.1.val < limit := by simpa only [Tile.mat,decide_eq_true_eq] using hi
      exact hbound i.2.1 i.1 hi'
  intro v
  exact Stmt.TraceSafeList.nil_intro

theorem value_safe (p : Layout) (r : Regions) (head : Nat) (start sn limit : Int)
    (u : BlockState) (bounds : RegionBounds)
    (hp : u.regs .ptr [p.BN,p.D] "v_ptrs" = some (Tile.mat (fun j d =>
      (Region.cast r.value,j.val*p.vRow+head*p.vHead+d.val*p.vDim))))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hb : u.regs .int [] "cur_batch_in_all_start_index" = some (Tile.scalar start))
    (hl : u.regs .int [] "cur_batch_seq_len" = some (Tile.scalar limit))
    (hn : u.regs .nat [p.BN] "offs_n" = some (Tile.vec (fun j => j.val)))
    (hbound : ∀ (j : Fin p.BN) (d : Fin p.D), sn+j.val < limit →
      ValuesExecution.address p head start sn d j < bounds r.value) :
    Stmt.TraceSafeList bounds (values p r) u := by
  let state := u
  unfold values
  apply TraceProofs.assign_cons
  · apply SafeOps.maskedPointerLoad _ _ _ bounds state
      (Tile.mat (fun j d => (Region.cast r.value,ValuesExecution.address p head start sn d j)))
      (Tile.mat (dtype := .bool) (fun j d => decide (sn+j.val < limit)))
    · simp [valueAdvance,z,Op.MemorySafe]
    · simp [evalOp.eq_def,valueAdvance,z,state,hp,hb,hs,ValuesExecution.address,
        Tile.mat,Tile.scalar,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
        Option.bind,-Tile.scalar_eta]
    · simp [valueMask,valueView,lanes,z,state,evalOp.eq_def,hs,hl,hn,
        Tile.mat,Tile.vec,Tile.scalar,Tile.remap,tile_elementwise,Broadcast.leftIndex,Broadcast.rightIndex,
        Option.bind,-Tile.scalar_eta]
      all_goals rfl
    · simp [valueAdvance,z,Op.SafeAt.eq_def]
    · simp [valueMask,valueView,lanes,z,Op.SafeAt.eq_def]
    · simp [Op.SafeAt.eq_def]
    · intro i hi
      have hi' : sn+i.1.val < limit := by simpa only [Tile.mat,decide_eq_true_eq] using hi
      exact hbound i.1 i.2.1 hi'
  intro v
  exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.LoadSafety
