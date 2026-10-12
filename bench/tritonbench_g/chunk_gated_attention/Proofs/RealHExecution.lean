import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHFinal

/-! Complete real execution of both typed H sources, at arbitrary symbolic
sizes and strides. All gate/seed/final switches and all iterations are covered.
The independent formula requires zero unspecified lanes; the source is not
silently changed to request zero padding. Finite-allocation safety, flat execution and public
IO packaging are proved in RealHSafety, RealHFlat and RealHIO. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHExecution
open VeriTile.Triton RealHMath RealHProgram RealHTiles
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def footprint (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool) : WriteFootprint :=
  WriteFootprint.union (RealHLoop.footprint p l r) (RealHFinal.footprint p l r storeFinal)

structure Outputs (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (gateK hasSeed storeFinal : Bool) (v : BlockState) : Prop where
  history : ∀ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT → RealHStore.active p l e.val j.val →
    v.readMem r.history (historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val)) =
      closed p l a gateK hasSeed c (keyIndex p l e.val) (valueIndex p l j.val)
  final : storeFinal = Bool.true → ∀ (e : Fin p.BK) (j : Fin p.BV), RealHStore.active p l e.val j.val →
    v.readMem r.final (stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val)) =
      closed p l a gateK hasSeed p.NT (keyIndex p l e.val) (valueIndex p l j.val)

theorem executes (p : Layout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal swapped : Bool) (a : Inputs) (u : BlockState)
    (hu : u.undef = fun _ _ => 0) (hs : RealHStep.Separate r)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final)
    (hhi : RealHLoop.HistoryNoCollision p (launch u))
    (hfi : storeFinal = Bool.true → RealHStore.NoCollision p (launch u) (RealHFinal.base p (launch u)) p.V 1)
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hg : ∀ off, u.readMem r.gate off = a.gate off)
    (hseed : ∀ off, u.readMem r.seed off = a.seed off) :
    ∃ v, exec (source p r dot gate history seed final gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel u = some v ∧
      Outputs p (launch u) r a gateK hasSeed storeFinal v ∧
      BlockState.WriteWithin (footprint p (launch u) r storeFinal) u v := by
  let l := launch u
  let start := RealHInit.initial p r a.seed (fun _ => 0) hasSeed u
  have hei := RealHInit.executes p r a.seed (fun _ => 0) hasSeed u (congrFun hu r.seed) hseed
  obtain ⟨mid,hel,hm⟩ := RealHLoop.executes u p l r a gateK hasSeed swapped hu hs hhi start
    (RealHInit.invariant p r a gateK hasSeed u hk hv hg)
  let f := expected p l a gateK hasSeed p.NT
  let v := RealHFinal.output p l r storeFinal f mid
  have hef := RealHFinal.executes p l r storeFinal f mid hm.key hm.value hm.head hm.carry
  have hf := RealHFinal.frame p l r storeFinal f mid
  refine ⟨v,?_,⟨?_,?_⟩,?_⟩
  · rw [exec,source_body,body,List.append_assoc,stepStmts.append_some hei]
    change stepStmts (RealHLoop.statement p r gateK swapped :: finalization p r storeFinal) start = some v
    rw [stepStmts.cons_some hel]
    exact hef
  · intro c e j hc ha
    have hframe := hf r.history (historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val))
      (fun h => hfsep h.1 h.2.1)
    unfold BlockState.readMem
    rw [← hframe]
    exact hm.history c e j hc ha
  · intro hon e j ha
    subst storeFinal
    exact RealHFinal.readback p l r f mid (hfi rfl) e j ha
  · intro region off hn
    exact (hm.frame region off (fun h => hn (Or.inl h))).trans
      (hf region off (fun h => hn (Or.inr h)))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHExecution
