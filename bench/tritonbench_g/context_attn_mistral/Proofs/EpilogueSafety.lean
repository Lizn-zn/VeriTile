import bench.tritonbench_g.context_attn_mistral.Proofs.SafeOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.EpilogueSafety
open VeriTile.Triton Formula Program PreludeProgram EpilogueExecution
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

theorem stores_safe (p : Layout) (r : Regions) (head : Nat) (start seq : Int) (rows : Fin p.BM → Nat)
    (u : BlockState) (bounds : RegionBounds)
    (hh : u.regs .nat [] "cur_head" = some (Tile.scalar head))
    (hs : u.regs .int [] "cur_batch_in_all_start_index" = some (Tile.scalar start))
    (hl : u.regs .int [] "cur_batch_seq_len" = some (Tile.scalar seq))
    (hr : u.regs .nat [p.BM] "offs_m" = some (Tile.vec rows))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val)))
    (hb : ∀ i : TileIndex [p.BM,p.D], (rows i.1 : Int) < seq →
      address p head start rows i < bounds r.out) :
    Stmt.TraceSafeList bounds (EpilogueProgram.code p r) u := by
  let off : Tile .int [p.BM,p.D] := Tile.mat (fun i d =>
    (start+(rows i : Int))*p.outRow+head*p.outHead+d.val*p.outDim)
  let s1 := u.setReg "off_o" .int [p.BM,p.D] off
  let ptr : Tile .ptr [p.BM,p.D] := { data := fun i => (r.out,address p head start rows i) }
  let s2 := s1.setReg "out_ptrs" .ptr [p.BM,p.D] ptr
  have h1 : evalOp (EpilogueProgram.offsets p) u = some off := by
    simp [EpilogueProgram.offsets,rowView,dimView,PreludeProgram.rows,dimensions,n,z,off,
      evalOp.eq_def,hh,hs,hr,hd,Tile.mat,Tile.vec,Tile.scalar,tile_elementwise,
      Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,Nat.cast_mul,-Tile.scalar_eta]
  have h2 : evalOp (.ptrAdd .scalarL (.ptrBase r.out) (Op.ref .int [p.BM,p.D] "off_o").castIntToNat) s1 =
      some ptr := by
    simp [evalOp.eq_def,s1,off,ptr,address,Tile.mat,Tile.scalar,tile_elementwise,
      Broadcast.leftIndex,Broadcast.rightIndex,Option.bind,-Tile.scalar_eta]
  simp only [EpilogueProgram.code,List.drop_succ_cons,List.drop_zero]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some h1)]
  refine ⟨by simp [Stmt.TraceSafe,EpilogueProgram.offsets,rowView,dimView,PreludeProgram.rows,dimensions,n,z,Op.SafeAt.eq_def],?_⟩
  change Stmt.TraceSafeList bounds _ s1
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some h2)]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt.eq_def],?_⟩
  change Stmt.TraceSafeList bounds _ s2
  apply Stmt.TraceSafeList.cons_intro
  · rw [Stmt.TraceSafe]
    refine ⟨by simp [MemAccess.SafeAt,Op.SafeAt.eq_def],by simp [Op.SafeAt.eq_def],
      by simp [MaskOpt.SafeAt,queryMask,rowView,dimView,PreludeProgram.rows,dimensions,z,Op.SafeAt.eq_def],?_⟩
    unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe Op.PointerAddressesSafeOn
    refine ⟨by simp [Op.MemorySafe],?_⟩
    intro ptr' he i hi
    have hptr : evalOp (.ref .ptr [p.BM,p.D] "out_ptrs") s2 = some ptr := by simp [evalOp_ref,s2]
    rw [hptr] at he
    cases he
    obtain ⟨m,he,hi⟩ := hi
    rw [SafeOps.queryMask_eval p rows seq s2 (by simpa [s2,s1] using hr)
      (by simpa [s2,s1] using hl)] at he
    cases he
    have hi' : (rows i.1 : Int) < seq := by simpa only [Tile.mat,decide_eq_true_eq] using hi
    exact hb i hi'
  · intro _ _
    exact Stmt.TraceSafeList.nil_intro

theorem safe (p : Layout) (r : Regions) (head : Nat) (start seq : Int) (rows : Fin p.BM → Nat)
    (u : BlockState) (bounds : RegionBounds)
    (hh : u.regs .nat [] "cur_head" = some (Tile.scalar head))
    (hs : u.regs .int [] "cur_batch_in_all_start_index" = some (Tile.scalar start))
    (hl : u.regs .int [] "cur_batch_seq_len" = some (Tile.scalar seq))
    (hr : u.regs .nat [p.BM] "offs_m" = some (Tile.vec rows))
    (hd : u.regs .nat [p.D] "offs_d" = some (Tile.vec (fun d => d.val)))
    (hb : ∀ i : TileIndex [p.BM,p.D], (rows i.1 : Int) < seq → address p head start rows i < bounds r.out) :
    Stmt.TraceSafeList bounds (EpilogueProgram.post p r) u := by
  rw [EpilogueProgram.source]
  exact stores_safe p r head start seq rows u bounds hh hs hl hr hd hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.EpilogueSafety
