import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHLoopSafety
import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHStateSafety

/-! Whole-source allocation safety for original and candidate H. Bounds cover
only the enabled seed/final stores and the selected gate branch, at each actual
chunk. The source's full flattened endpoints and all symbolic strides remain. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHSafety
open VeriTile.Triton RealHMath RealHProgram RealHAccessSafety
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (r : Regions) (gateK hasSeed storeFinal : Bool)
    (bounds : RegionBounds) : Prop where
  initial : hasSeed = Bool.true → StateBounds p l r.seed bounds
  loop : RealHLoopSafety.Bounds p l r gateK bounds
  final : storeFinal = Bool.true → StateBounds p l r.final bounds

theorem source_safe (p : Layout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal swapped : Bool)
    (u : BlockState) (hu : u.undef = fun _ _ => 0) (hs : RealHStep.Separate r)
    (hinj : RealHLoop.HistoryNoCollision p (launch u))
    (bounds : RegionBounds) (hb : Bounds p (launch u) r gateK hasSeed storeFinal bounds) :
    (source p r dot gate history seed final gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel.TraceSafe bounds u := by
  let l := launch u
  let a : Inputs := ⟨u.readMem r.key,u.readMem r.value,u.readMem r.gate,u.readMem r.seed⟩
  let start := RealHInit.initial p r a.seed (fun _ => 0) hasSeed u
  have hi := RealHInit.invariant p r a gateK hasSeed u (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
  have hei := RealHInit.executes p r a.seed (fun _ => 0) hasSeed u (congrFun hu r.seed) (fun _ => rfl)
  have hloop := RealHLoopSafety.loop_safe u p l r a gateK hasSeed swapped hu hs hinj bounds hb.loop start hi
  obtain ⟨mid,hel,hm⟩ := RealHLoop.executes u p l r a gateK hasSeed swapped hu hs hinj start hi
  rw [Kernel.TraceSafe,source_body,body,List.append_assoc]
  apply TraceProofs.append_safe _ u (RealHStateSafety.init_safe p r hasSeed bounds u hb.initial)
  intro v hv
  rw [hei] at hv
  cases hv
  change Stmt.TraceSafeList bounds (RealHLoop.statement p r gateK swapped :: finalization p r storeFinal) start
  rw [TraceProofs.cons_of_run_iff hel]
  exact ⟨hloop,RealHStateSafety.final_safe p l r storeFinal bounds mid hb.final hm.key hm.value hm.head⟩

theorem source_flatten_ok (p : Layout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal swapped : Bool) :
    (source p r dot gate history seed final gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,source_body]
  cases swapped <;> simp [StmtList.FlattenOk,Stmt.FlattenOk,body,initialization,finalization,
    iteration,gateKey,RealHProgram.gateValue,storeCarry,carry,RealHProgram.dot,
    keyPtr,valuePtr,historyPtr,statePtr,stateBase,endpointPtr,endpointOp,times,load1,load2,Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHSafety
