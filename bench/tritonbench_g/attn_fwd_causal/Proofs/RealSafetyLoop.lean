import bench.tritonbench_g.attn_fwd_causal.Proofs.RealSafetyBody

/-! Safety follows the actual helper range and its pointer invariant.
Every visited K/V lane is bounded only when its source mask is active;
each visited key scale is bounded unconditionally. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyLoop
open VeriTile.Triton RealLoads RealMemory RealLoop
open AttentionFwdTriton3.RealLoopFormula (Carry)
noncomputable section
set_option maxHeartbeats 1000000

structure Bounds (bounds : RegionBounds) (c : Config) (BN BD lower upper : Nat) : Prop where
  keys : ∀ i, i < RealFormula.tripCount lower upper BN → ∀ (j : Fin BN) (d : Fin BD),
    cursor lower BN i+j.val < c.context → d.val < c.activeHead →
    c.base+d.val+j.val*c.strideKN+c.headDim*cursor lower BN i < bounds c.K
  scales : ∀ i, i < RealFormula.tripCount lower upper BN →
    c.keyScaleBase+cursor lower BN i/BN < bounds c.KS
  values : ∀ i, i < RealFormula.tripCount lower upper BN → ∀ (j : Fin BN) (d : Fin BD),
    cursor lower BN i+j.val < c.context → d.val < c.activeHead →
    c.base+j.val*c.strideM+d.val*c.strideD+c.headDim*cursor lower BN i < bounds c.V

theorem loop_safe_from {BM BN BD : Nat} (bounds : RegionBounds) (mem s : BlockState) (c : Config)
    (lower upper : Nat) (second diagonal : Bool) (seed : Carry BM BD) (hN : 0 < BN)
    (hb : Bounds bounds c BN BD lower upper)
    (i : Nat) (hi : i ≤ RealFormula.tripCount lower upper BN) (hmem : s.mem = mem.mem)
    (hr : Registers (inputs mem c BM BN BD) c
      (after (inputs mem c BM BN BD) c lower second diagonal seed i) (cursor lower BN i) second s) :
    Stmt.forRangeTraceSafe bounds "start_n" (cursor lower BN i) upper BN
      (sourceBody BM BN BD c.context c.activeHead c.headDim second diagonal) s := by
  rw [Stmt.forRangeTraceSafe, if_neg (by omega)]
  by_cases hlt : i < RealFormula.tripCount lower upper BN
  · rw [if_pos ((cursor_lt lower upper BN i hN).mpr hlt)]
    let si := s.setReg "start_n" .nat [] (Tile.scalar (cursor lower BN i))
    have hp : RealIteration.Pointers si c BN BD (cursor lower BN i) second := by
      refine ⟨?_, ?_, ?_⟩
      · cases second <;> simpa [si, keyName] using hr.pointers.key
      · cases second <;> simpa [si, scaleName] using hr.pointers.scale
      · cases second <;> simpa [si, valueName] using hr.pointers.value
    refine ⟨RealSafetyBody.safe bounds si c BM BN BD (cursor lower BN i) second diagonal hp
      (by simpa [si] using hr.keys) (by simp [si]) (by simpa [si] using hr.undef)
      (hb.keys i hlt) (hb.scales i hlt) (hb.values i hlt), ?_⟩
    obtain ⟨t, et, ht, hf⟩ := invariant_step mem s c lower second diagonal seed hN i hmem hr
    rw [et]
    simpa only [cursor_succ] using loop_safe_from bounds mem t c lower upper second diagonal seed hN hb
      (i+1) (by omega) (hf.mem.trans hmem) ht
  · rw [if_neg (by have ht := cursor_lt lower upper BN i hN; omega)]
    trivial
termination_by RealFormula.tripCount lower upper BN-i

theorem loop_safe {BM BN BD : Nat} (bounds : RegionBounds) (s : BlockState) (c : Config)
    (lower upper : Nat) (second diagonal : Bool) (seed : Carry BM BD) (hN : 0 < BN)
    (hb : Bounds bounds c BN BD lower upper)
    (hr : Registers (inputs s c BM BN BD) c seed lower second s) :
    Stmt.forRangeTraceSafe bounds "start_n" lower upper BN
      (sourceBody BM BN BD c.context c.activeHead c.headDim second diagonal) s := by
  simpa only [cursor, Nat.zero_mul, Nat.add_zero] using
    loop_safe_from bounds s s c lower upper second diagonal seed hN hb 0 (Nat.zero_le _) rfl
      (by simpa only [after, AttentionFwdTriton3.RealLoopFormula.closed_zero, cursor,
        Nat.zero_mul, Nat.add_zero] using hr)

theorem statement_safe {BM BN BD : Nat} (bounds : RegionBounds) (s : BlockState) (c : Config)
    (lower upper : Nat) (second diagonal : Bool) (seed : Carry BM BD) (hN : 0 < BN)
    (hb : Bounds bounds c BN BD lower upper)
    (hl : s.regs .nat [] (RealSetup.lowerName second) = some (Tile.scalar lower))
    (hu : s.regs .nat [] (RealSetup.upperName second) = some (Tile.scalar upper))
    (hr : Registers (inputs s c BM BN BD) c seed lower second s) :
    Stmt.TraceSafe bounds (RealSetupSource.statement BM BN BD c.context c.activeHead c.headDim second diagonal) s := by
  rw [RealSetupSource.statement, Stmt.TraceSafe]
  refine ⟨by simp [Op.SafeAt.eq_def], by simp [Op.SafeAt.eq_def], by simp [Op.SafeAt.eq_def], ?_⟩
  simp only [evalOp_ref, hl, hu, evalOp_constNat, Tile.scalar]
  exact loop_safe bounds s c lower upper second diagonal seed hN hb hr

theorem visit_mem (lower upper BN i : Nat) (diagonal : Bool)
    (hi : i < RealFormula.tripCount lower upper BN) :
    (⟨cursor lower BN i,diagonal⟩ : RealFormula.Visit) ∈ RealFormula.visits lower upper BN diagonal := by
  exact List.mem_map.mpr ⟨i, List.mem_range.mpr hi, rfl⟩

theorem bounds_of_visits (bounds : RegionBounds) (c : Config) (Out : RegionName)
    (BM BN BD stage lower upper : Nat) (diagonal : Bool)
    (h : RealAccessSafety.Windows c Out BM BN BD stage bounds)
    (hsub : RealFormula.visits lower upper BN diagonal ⊆ RealFormula.allVisits c.startM BM BN stage) :
    Bounds bounds c BN BD lower upper := by
  exact ⟨fun i hi => h.key _ (hsub (visit_mem lower upper BN i diagonal hi)),
    fun i hi => h.keyScale _ (hsub (visit_mem lower upper BN i diagonal hi)),
    fun i hi => h.value _ (hsub (visit_mem lower upper BN i diagonal hi))⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyLoop
