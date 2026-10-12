import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealSetupSafety
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealPostSafety

/-! Whole-source address safety from the three explicit access-window contracts.
The proof executes the actual prologue and uses the full value-loop invariant
to reach the postlude. No successful whole-program run is a caller premise. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealSafety
open VeriTile.Triton RealMath
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

structure Bounds (s : BlockState) (p : Layout)
    (q k v h g dout dh dq dk dg : RegionName) (bounds : RegionBounds) : Prop where
  setup : RealSetupSafety.Bounds s p.T p.BT g bounds
  loop : RealLoopSafety.Bounds p (RealLoop.launch s) ⟨v,h,dout,dh⟩ bounds
  post : RealPostSafety.Bounds p (RealLoop.launch s) q k dq dk dg bounds

theorem source_safe (s : BlockState) (p : Layout)
    (qk valueStorage hist dhist : TypedSource.Storage)
    (q k v h g dout dh dqOut dkOut dgOut : RegionName)
    (bounds : RegionBounds) (hb : Bounds s p q k v h g dout dh dqOut dkOut dgOut bounds)
    (hzero : s.undef = fun _ _ => 0) :
    (RealSource.source qk valueStorage hist dhist Bool.false q k v h g dout dh dqOut dkOut dgOut
      p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
      p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType.toAlgKernel.TraceSafe bounds s := by
  let regions : RealStep.Regions := ⟨v,h,dout,dh⟩
  let a := RealPost.inputs s q k v h g dout dh
  obtain ⟨setup, t, hsetup, hprefix, hmem, hi⟩ :=
    RealLoop.prologue_and_loop s p regions g a hzero
      (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
  obtain ⟨setup', hsetup', _, _, hm, hik, hit, hib, hnb, _, _, _, hq, hk, hs, hdg, _⟩ :=
    cbdPreLoop_run g p.T p.BT p.BK s (congrFun hzero g)
  have heq : setup' = setup := Option.some.inj (hsetup'.symm.trans hsetup)
  subst setup'
  have hloop := RealLoopSafety.loop_safe setup p (RealLoop.launch s) regions a bounds hb.loop
    ((cbdPreLoop_undef g p.T p.BT p.BK s setup hsetup).trans hzero)
    (fun off => hm v off) (fun off => hm h off) (fun off => hm dout off) (fun off => hm dh off)
    hik hit hib (RealLoop.initial setup p (RealLoop.launch s) a hq hk hs hdg)
  have hpre : Stmt.TraceSafeList bounds (cbdPreLoop g p.T p.BT p.BK ++ [RealLoop.statement p regions]) s := by
    apply TraceProofs.append_safe _ s (RealSetupSafety.safe s g p.T p.BT p.BK bounds hb.setup)
    intro u hu
    rw [hsetup] at hu
    cases hu
    apply Stmt.TraceSafeList.cons_intro hloop
    intro _ _
    exact Stmt.TraceSafeList.nil_intro
  have grid : RealFrame.Grid t (RealLoop.launch s) := ⟨
    ((hi.fixed .nat [] "i_k" (by decide)).trans hik),
    ((hi.fixed .nat [] "i_t" (by decide)).trans hit),
    ((hi.fixed .nat [] "i_bh" (by decide)).trans hib),
    ((hi.fixed .nat [] "n_bh" (by decide)).trans hnb)⟩
  rw [Kernel.TraceSafe, RealSource.real_body_matches, cbd_body_eq]
  change Stmt.TraceSafeList bounds
    ((cbdPreLoop g p.T p.BT p.BK ++ [RealLoop.statement p regions]) ++
      cbdPostLoop q k dqOut dkOut dgOut p.s_k_h p.s_k_t p.scale p.T p.K p.BT p.BK) s
  apply TraceProofs.append_safe _ s hpre
  intro u hu
  rw [hprefix] at hu
  cases hu
  exact RealPostSafety.safe t p (RealLoop.launch s) q k dqOut dkOut dgOut bounds hb.post grid

theorem source_flatten_ok (p : Layout)
    (qk valueStorage hist dhist : TypedSource.Storage)
    (q k v h g dout dh dqOut dkOut dgOut : RegionName) :
    (RealSource.source qk valueStorage hist dhist Bool.false q k v h g dout dh dqOut dkOut dgOut
      p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
      p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk, RealSource.real_body_matches, cbd_body_eq]
  simp [StmtList.FlattenOk, Stmt.FlattenOk, cbdPreLoop, cbdLoopBody, cbdPostLoop,
    cbdStopOp, Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealSafety
