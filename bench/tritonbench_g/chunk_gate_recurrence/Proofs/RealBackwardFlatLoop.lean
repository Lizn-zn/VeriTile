import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardFlatTail

/-! Strict flat correspondence at every non-final reverse iteration. In the
last iteration only the three dead pointer registers may diverge. The result
still retains all live registers for the real DL postlude. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardFlatLoop
open VeriTile.Triton RealProgram RealBackwardProgram RealBackwardPointers RealBackwardLoop RealBackwardFlatState
open RealBackwardStep (stored beforeStore)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem body_bridge (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (initial : BlockState) (p : Layout) (l : Launch) (S D DI DG DL DS : RegionName)
    (a : Inputs p) (hb : RealBackwardSafety.Bounds p l S D DI DG DL DS A.extent)
    (k : Nat) (u v : BlockState) (hk : k < p.N-1)
    (h : Holds initial p l S D DI DG DS a k u) (hu : u.undef = fun _ _ => 0)
    (he : stepStmts (loopBody p) (u.setReg "_i" .nat [] (Tile.scalar k)) = some v) :
    ∃ t, stepStmts (A.flattenStmts (loopBody p))
        ((A.flattenState u).setReg "_i" .nat [] (Tile.scalar k)) = some t ∧
      Agree t (A.flattenState v) ∧
      (k+1 < p.N-1 → t = A.flattenState v) ∧ v.undef = (fun _ _ => 0) := by
  let entry := u.setReg "_i" .nat [] (Tile.scalar k)
  let acc := RealMath.reverse a p.N (p.N-k)
  let mid := stored p l DI DG a k acc entry
  have hc := RealBackwardFlatCore.executes p l S D DI DG DS a k acc entry hk
    (by simp [entry,h.source]) (by simp [entry,h.target]) (by simp [entry,h.incoming])
    (by simp [entry,h.gradient]) (by simp [entry,h.gate]) (by simpa [entry,acc] using h.carry)
    (by intro e j; simpa [entry] using h.inputSource (p.N-2-k) e j (by omega))
    (by intro e j; simpa [entry] using h.inputIncoming (p.N-1-k) e j (by omega))
    (by simpa [entry] using h.inputGate (p.N-1-k) (by omega))
  have hs := RealBackwardFlatCore.safe p l S D DI DG DL DS A.extent hb k hk entry
    (by simp [entry,h.source]) (by simp [entry,h.target]) (by simp [entry,h.incoming])
    (by simp [entry,h.gradient]) (by simp [entry,h.gate])
  have hcf := A.stepStmts_flatten hd hcov (RealBackwardFlatCore.core p) entry hs
    (RealBackwardFlatCore.flatten_ok p) hu
  rw [hc,Option.map_some] at hcf
  have ht := RealBackwardFlatTail.executes A p l S D DI DG DS k mid hk
    (by simp [mid,stored,scatter,beforeStore,entry,h.source])
    (by simp [mid,stored,scatter,beforeStore,entry,h.target])
    (by simp [mid,stored,scatter,beforeStore,entry,h.incoming])
    (by simp [mid,stored,scatter,beforeStore,entry,h.gradient])
    (by simp [mid,stored,scatter,beforeStore,entry,h.gate])
    (by simp [mid,stored,scatter,beforeStore,entry,h.fixed.numK])
    (by simp [mid,stored,scatter,beforeStore,entry,h.fixed.numV])
  have he' := RealBackwardStep.executes p l S D DI DG DS a k acc entry hk
    (by simp [entry,h.source]) (by simp [entry,h.target]) (by simp [entry,h.incoming])
    (by simp [entry,h.gradient]) (by simp [entry,h.gate]) (by simpa [entry,acc] using h.carry)
    (by simp [entry,h.fixed.numK]) (by simp [entry,h.fixed.numV])
    (by intro e j; simpa [entry] using h.inputSource (p.N-2-k) e j (by omega))
    (by intro e j; simpa [entry] using h.inputIncoming (p.N-1-k) e j (by omega))
    (by simpa [entry] using h.inputGate (p.N-1-k) (by omega))
  have hv : RealBackwardFlatTail.regionResult p l S D DI DG DS k mid = v :=
    Option.some.inj (he'.symm.trans he)
  refine ⟨RealBackwardFlatTail.result A p l S D DI DG DS k mid,?_,?_,?_,?_⟩
  · have hset : (A.flattenState u).setReg "_i" .nat [] (Tile.scalar k) = A.flattenState entry :=
      (A.flattenState_setReg u "_i" .nat [] (Tile.scalar k)).symm
    rw [hset,RealBackwardFlatCore.split,FlatAlloc.flattenStmts_append,
      ProgramRewrite.append,hcf]
    exact ht
  · rw [← hv]
    exact RealBackwardFlatTail.agree A p l S D DI DG DS k mid
  · intro hnext
    rw [RealBackwardFlatTail.exact_before_last A p l S D DI DG DS k mid hnext,hv]
  · rw [← hv]
    change mid.undef = _
    exact (stepStmts_undef (RealBackwardFlatCore.core p) entry mid
      (RealBackwardFlatCore.flatten_ok p) hc).trans hu

theorem range_bridge (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (initial : BlockState) (p : Layout) (l : Launch) (S D DI DG DL DS : RegionName)
    (a : Inputs p) (sep : Separation S D DI DG DS) (hi : NoCollision p l)
    (hgi : Function.Injective (gradientAddress p l))
    (hb : RealBackwardSafety.Bounds p l S D DI DG DL DS A.extent) :
    ∀ (k : Nat) (u : BlockState), Holds initial p l S D DI DG DS a k u →
      u.undef = (fun _ _ => 0) →
      ∃ v t, stepForRangeAux "_i" k (p.N-1) 1 (loopBody p) u = some v ∧
        stepForRangeAux "_i" k (p.N-1) 1 (A.flattenStmts (loopBody p)) (A.flattenState u) = some t ∧
        Holds initial p l S D DI DG DS a (p.N-1) v ∧ Agree t (A.flattenState v) ∧ v.undef = (fun _ _ => 0)
  | k,u,h,hu => by
    by_cases hk : k < p.N-1
    · obtain ⟨v,he,hv⟩ := iteration initial p l S D DI DG DS a sep hi hgi k u hk h
      obtain ⟨t,hflat,hagree,hstrict,hundef⟩ := body_bridge A hd hcov initial p l S D DI DG DL DS a hb k u v hk h hu he
      by_cases hnext : k+1 < p.N-1
      · obtain ⟨v',t',he',ht',hv',ha',hu'⟩ := range_bridge A hd hcov initial p l S D DI DG DL DS a sep hi hgi hb (k+1) v hv hundef
        refine ⟨v',t',?_,?_,hv',ha',hu'⟩
        · rw [stepForRangeAux.step_lt (by decide) hk,he]
          exact he'
        · rw [stepForRangeAux.step_lt (by decide) hk,hflat,hstrict hnext]
          exact ht'
      · have hend : k+1 = p.N-1 := by have := hv.count_le; omega
        refine ⟨v,t,?_,?_,hend ▸ hv,hagree,hundef⟩
        · rw [stepForRangeAux.step_lt (by decide) hk,he]
          exact stepForRangeAux.step_ge (by decide) (by omega)
        · rw [stepForRangeAux.step_lt (by decide) hk,hflat]
          exact stepForRangeAux.step_ge (by decide) (by omega)
    · have hend : k = p.N-1 := by have := h.count_le; omega
      refine ⟨u,A.flattenState u,?_,?_,hend ▸ h,Agree.refl _,hu⟩
      · exact stepForRangeAux.step_ge (by decide) (by omega)
      · exact stepForRangeAux.step_ge (by decide) (by omega)
termination_by k _ _ _ => p.N-1-k
decreasing_by omega

theorem executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (initial : BlockState) (p : Layout) (l : Launch) (S D DI DG DL DS : RegionName)
    (a : Inputs p) (sep : Separation S D DI DG DS) (hi : NoCollision p l)
    (hgi : Function.Injective (gradientAddress p l))
    (hb : RealBackwardSafety.Bounds p l S D DI DG DL DS A.extent)
    (u : BlockState) (h : Holds initial p l S D DI DG DS a 0 u) (hu : u.undef = fun _ _ => 0) :
    ∃ v t, stepStmt (statement p) u = some v ∧
      stepStmt (A.flattenStmt (statement p)) (A.flattenState u) = some t ∧
      Holds initial p l S D DI DG DS a (p.N-1) v ∧ Agree t (A.flattenState v) ∧ v.undef = (fun _ _ => 0) := by
  obtain ⟨v,t,he,ht,hv,ha,hu'⟩ := range_bridge A hd hcov initial p l S D DI DG DL DS a sep hi hgi hb 0 u h hu
  refine ⟨v,t,?_,?_,hv,ha,hu'⟩
  · simpa only [statement,stepForRangeAux.forRangeDyn_unfold,
      RealForwardLoop.stop_eval,evalOp_constNat,Option.bind_some] using he
  · simpa only [statement,FlatAlloc.flattenStmt,RealForwardProgram.stop,
      FlatAlloc.flattenOp,stepForRangeAux.forRangeDyn_unfold,evalOp.eq_def,
      Option.bind_some,Option.bind_eq_bind,Tile.bop,Tile.scalar,NumericDType.sub] using ht

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardFlatLoop
