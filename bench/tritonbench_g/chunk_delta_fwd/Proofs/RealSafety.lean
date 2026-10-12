import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealOuterSafety
import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealExecution

/-! Whole-source address safety, including both optional-state branches. Each
bound is on an active source address. The real loop invariant supplies the
reached register states; no whole-program execution is assumed by the caller. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealSafety
open VeriTile.Triton RealProgram
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (r : Regions) (useSeed storeFinal : Bool)
    (bounds : RegionBounds) : Prop where
  initial : useSeed = Bool.true →
    RealStateSafety.Bounds p l r.initial (RealInit.base p l) p.V bounds
  loop : RealOuterSafety.Bounds p l r bounds
  final : storeFinal = Bool.true →
    RealStateSafety.Bounds p l r.final (RealInit.base p l) p.V bounds

theorem source_safe (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (useSeed storeFinal : Bool) (u : BlockState) (hu : u.undef = fun _ _ => 0)
    (hk : u.pids 0 = 0) (hK : p.K ≤ p.BK) (hBC : 0 < p.BC)
    (hs : RealOuterStep.Separate r)
    (hvi : RealInnerLoop.NoCollision p (RealInit.launch u))
    (hhi : RealOuterLoop.HistoryNoCollision p (RealInit.launch u))
    (bounds : RegionBounds) (hb : Bounds p (RealInit.launch u) r useSeed storeFinal bounds) :
    (RealExecution.source p r kd value initial final useSeed storeFinal).eraseDType.toAlgKernel.TraceSafe bounds u := by
  let l := RealInit.launch u
  let a : Inputs := ⟨u.readMem r.key, u.readMem r.delta, u.readMem r.value⟩
  let seed := u.readMem r.initial
  let start := RealInit.initial p r seed useSeed u
  have hi := RealInit.invariant p r a seed useSeed u hk (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
  have heinit := RealInit.executes p r seed useSeed u hu (fun _ => rfl)
  have hloop := RealOuterSafety.loop_safe u p l r a (RealInit.seed p l seed) useSeed hu
    hk hK hBC hs hvi hhi bounds hb.loop start hi
  obtain ⟨mid,heloop,hm⟩ := RealOuterLoop.executes u p l r a (RealInit.seed p l seed) useSeed hu
    hk hK hBC hs hvi hhi start hi
  rw [Kernel.TraceSafe, RealExecution.body_parts, List.append_assoc]
  apply TraceProofs.append_safe _ u (RealStateSafety.init_safe p r useSeed bounds u hb.initial)
  intro v hv
  rw [heinit] at hv
  cases hv
  change Stmt.TraceSafeList bounds
    (RealOuterLoop.statement p r :: cdfEpilogue r.final p.K p.V p.BK p.BV storeFinal) start
  rw [TraceProofs.cons_of_run_iff heloop]
  exact ⟨hloop, RealStateSafety.final_safe p l r storeFinal bounds mid hb.final hm.key hm.value hm.head⟩

theorem source_flatten_ok (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (useSeed storeFinal : Bool) :
    (RealExecution.source p r kd value initial final useSeed storeFinal).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk, RealExecution.body_parts]
  simp [StmtList.FlattenOk, Stmt.FlattenOk, cdfPrologue, cdfEpilogue,
    RealOuterLoop.statement, RealOuterStep.body, cdfOuterBody, chunkDeltaInnerBody, cdfStopOp,
    Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealSafety
