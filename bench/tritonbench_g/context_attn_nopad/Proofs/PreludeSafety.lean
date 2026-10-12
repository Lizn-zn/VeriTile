import bench.tritonbench_g.context_attn_nopad.Proofs.SafeOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.PreludeSafety
open VeriTile.Triton Formula Program PreludeProgram
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem launch_safe (p : Layout) (u : BlockState) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds (launchCode p) u := by
  apply TraceProofs.all_safe
  simp [launchCode,n,Stmt.TraceSafe,Op.SafeAt.eq_def]

theorem metadata_safe (r : Regions) (batch : Nat) (u : BlockState) (bounds : RegionBounds)
    (hr : u.regs .nat [] "cur_batch" = some (Tile.scalar batch))
    (h1 : batch < bounds r.start) (h2 : batch < bounds r.sequence) :
    Stmt.TraceSafeList bounds (metadataCode r) u := by
  unfold metadataCode
  apply TraceProofs.assign_cons (SafeOps.metadata r.sequence batch u bounds hr h2)
  intro v1
  apply TraceProofs.assign_cons (SafeOps.metadata r.start batch _ bounds (by simpa using hr) h1)
  intro v2
  exact Stmt.TraceSafeList.nil_intro

theorem offsets_safe (p : Layout) (u : BlockState) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds (offsetCode p) u := by
  apply TraceProofs.all_safe
  simp [offsetCode,offset,keyOffset,valueOffset,keyView,valueView,keyChannel,lanes,n,z,rowView,dimView,rows,dimensions,Stmt.TraceSafe,Op.SafeAt.eq_def]

theorem query_safe (p : Layout) (r : Regions) (rows : Fin p.BM → Nat) (seq : Int)
    (offsets : Fin p.BM → Fin p.D → Int) (u : BlockState) (bounds : RegionBounds)
    (hr : u.regs .nat [p.BM] "offs_m" = some (Tile.vec rows))
    (hs : u.regs .int [] "cur_batch_seq_len" = some (Tile.scalar seq))
    (ho : u.regs .int [p.BM,p.D] "off_q" = some (Tile.mat offsets))
    (hb : ∀ i d, (rows i : Int) < seq → (offsets i d).toNat < bounds r.query) :
    Stmt.TraceSafeList bounds (queryCode p r) u := by
  unfold queryCode
  apply TraceProofs.assign_cons
  · unfold loadQuery
    apply SafeOps.maskedLoad r.query _ _ _ bounds u
      (Tile.mat (fun i d => (offsets i d).toNat))
      (Tile.mat (dtype := .bool) (fun i _ => decide ((rows i : Int) < seq)))
    · simp [evalOp.eq_def,ho,Tile.mat]
    · exact SafeOps.queryMask_eval p rows seq u hr hs
    · simp [Op.SafeAt.eq_def]
    · simp [queryMask,rowView,dimView,PreludeProgram.rows,dimensions,z,Op.SafeAt.eq_def]
    · simp [Op.SafeAt.eq_def]
    · intro i hi
      have hi' : (rows i.1 : Int) < seq := by simpa only [Tile.mat,decide_eq_true_eq] using hi
      exact hb i.1 i.2.1 hi'
  intro q
  apply TraceProofs.all_safe
  simp [n,z,Stmt.TraceSafe,Op.SafeAt.eq_def]

theorem complete (p : Layout) (r : Regions) (u : BlockState) (bounds : RegionBounds)
    (hb : AddressBounds.Valid p r (launch u) (inputs u r) bounds) :
    Stmt.TraceSafeList bounds (prelude p r) u := by
  rw [PreludeProgram.source]
  simp only [List.append_assoc]
  apply TraceProofs.append_safe
  · exact launch_safe p u bounds
  intro s he
  rw [PreludeLaunch.executes] at he
  cases he
  apply TraceProofs.append_safe
  · exact metadata_safe r (u.pids 0) _ bounds (PreludeLaunch.batch p u) hb.start hb.sequence
  intro s he
  change stepStmts (metadataCode r) (PreludeComplete.launchState p u) = some s at he
  rw [PreludeComplete.metadata_executes p r u] at he
  cases he
  apply TraceProofs.append_safe
  · exact offsets_safe p _ bounds
  intro s he
  rw [PreludeComplete.offsets_execute p r u] at he
  cases he
  apply query_safe p r (fun i => queryRow p (launch u) i.val) (Formula.sequence p (launch u) (inputs u r))
    (fun i d => AddressBounds.queryAddress p (launch u) (inputs u r) i.val d.val)
  · exact PreludeOffsets.rows _ _ _ _ _
  · simp [PreludeComplete.offsetState,PreludeOffsets.prepared,PreludeComplete.metadataState,
      PreludeMetadata.prepared,PreludeComplete.launchState,PreludeLaunch.launched,Formula.sequence,launch,inputs]
  · exact PreludeOffsets.off_q _ _ _ _ _
  · intro i d hi
    exact AddressBounds.inside_toNat (hb.query i d hi)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.PreludeSafety
