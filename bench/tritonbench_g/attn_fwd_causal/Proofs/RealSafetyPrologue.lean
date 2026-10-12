import bench.tritonbench_g.attn_fwd_causal.Proofs.RealSafetyParts

/-! The source prologue derives its pointers from program IDs and strides.
Only the masked query load and the unmasked query-scale load access memory.
Their bounds are ordinary input-window inequalities. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyPrologue
open VeriTile.Triton RealLoads RealAccessSafety
open AttentionFwdTriton3.RealAccess AttentionFwdTriton3.RealRegisterSafety
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem coordinates_safe (H N BM BN sz sh sm : Nat) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealCoordinates.code H N BM BN sz sh sm) s := by
  apply list_safe
  simp [RealCoordinates.code, RealCoordinates.ceilExpr, Op.SafeAt.eq_def]

theorem pointers_safe (c : Config) (Out : RegionName) (BM BN BD : Nat)
    (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealPointers.code c Out BM BN BD) s := by
  apply list_safe
  simp [RealPointers.code, RealPointers.rowExpr, RealPointers.rowPointerExpr,
    RealPointers.queryScaleExpr, RealPointers.keyExpr, RealPointers.keyScaleExpr, Op.SafeAt.eq_def]

theorem init_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (Out : RegionName) (BM BN BD : Nat)
    (hp : RealPointers.Registers c Out BM BN BD s)
    (hbQ : ∀ (r : Fin BM) (d : Fin BD), c.startM*BM+r.val < c.context → d.val < c.activeHead →
      c.base+(c.startM*BM+r.val)*c.strideM+d.val*c.strideD < bounds c.Q)
    (hbS : c.queryScaleBase+c.startM < bounds c.QS) :
    Stmt.TraceSafeList bounds (RealInit.code BM BD c.context c.activeHead) s := by
  unfold RealInit.code
  apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
  intro m
  apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
  intro l
  apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
  intro acc
  apply TraceProofs.assign_cons
  · apply query_safe
    · simpa using hp.query
    · simpa using hp.rows
    · exact hbQ
  · intro q
    apply Stmt.TraceSafeList.cons_intro
    · rw [Stmt.TraceSafe]
      apply query_scale_safe
      · simpa using hp.queryScale
      · exact hbS
    · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem safe (bounds : RegionBounds) (s : BlockState) (Q K V QS KS Out : RegionName)
    (sz sh sm sd sn H N HD BM BN BD HA : Nat)
    (hbQ : ∀ (r : Fin BM) (d : Fin BD), s.pids 0*BM+r.val < N → d.val < HA →
      (s.pids 1/H*sz+s.pids 1%H*sh)+(s.pids 0*BM+r.val)*sm+d.val*sd < bounds Q)
    (hbS : s.pids 1*RealCoordinates.ceil N BM+s.pids 0 < bounds QS) :
    Stmt.TraceSafeList bounds (RealPrologue.code Q K V QS KS Out sz sh sm sd sn H N HD BM BN BD HA) s := by
  let c := RealPrologue.configuration Q K V QS KS sm sd sn N HD HA
    (RealCoordinates.values s H N BM BN sz sh sm)
  obtain ⟨sc, ec, hc, _⟩ := RealCoordinates.executes s H N BM BN sz sh sm
  obtain ⟨sp, ep, hp, _⟩ := RealPointers.executes sc c Out BM BN BD
    hc.row hc.base hc.queryScaleBase hc.keyScaleBase
  have hinit := init_safe bounds sp c Out BM BN BD hp hbQ hbS
  have hptr := append_of_run (pointers_safe c Out BM BN BD bounds sc) ep hinit
  have hall := append_of_run (coordinates_safe H N BM BN sz sh sm bounds s) ec hptr
  simpa only [RealPrologue.code, List.append_assoc] using hall

theorem setup_safe (BM BN BD HD : Nat) (second : Bool) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealSetup.code BM BN BD HD second) s := by
  apply list_safe
  simp [RealSetup.code, RealSetup.lowerExpr, RealSetup.upperExpr, Op.SafeAt.eq_def]

theorem first_setup_safe (BM BN BD HD : Nat) (diagonal : Bool) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealFirstSetup.code BM BN BD HD diagonal) s := by
  apply list_safe
  simp [RealFirstSetup.code, RealFirstSetup.copies, RealSetup.code,
    RealSetup.lowerExpr, RealSetup.upperExpr, Op.SafeAt.eq_def]

theorem epilogue_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (Out : RegionName)
    (BM BD : Nat) (candidate : Bool)
    (hp : s.regs .ptr [BM,BD] "O_block_ptr" = some (RealPointers.outputPointer c Out BM BD))
    (hr : s.regs .nat [BM] "offs_m" = some (RealPointers.rows BM (c.startM*BM)))
    (hb : ∀ i, RealEpilogue.active c BM BD i = Bool.true → RealEpilogue.address c BM BD i < bounds Out) :
    Stmt.TraceSafeList bounds (RealSource.epilogue BM BD c.context c.activeHead candidate) s := by
  unfold RealSource.epilogue RealSource.normalization
  apply TraceProofs.assign_cons (by cases candidate <;> simp [Op.SafeAt.eq_def])
  intro a
  apply Stmt.TraceSafeList.cons_intro
  · apply output_safe bounds _ c Out BM BD
    · simpa using hp
    · simpa using hr
    · exact hb
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyPrologue
