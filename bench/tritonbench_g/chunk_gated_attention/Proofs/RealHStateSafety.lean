import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHAccessSafety

/-! Safety of the optional seed and final-state accesses. The seed is loaded
inside the original addition, and final output uses its exact checked layout.
Neither numerical values nor an output equality are safety assumptions. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHStateSafety
open VeriTile.Triton RealHMath RealHProgram RealHPointers RealHAccessSafety
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem init_safe (p : Layout) (r : Regions) (hasSeed : Bool)
    (bounds : RegionBounds) (u : BlockState)
    (hb : hasSeed = Bool.true → StateBounds p (launch u) r.seed bounds) :
    Stmt.TraceSafeList bounds (initialization p r hasSeed) u := by
  unfold initialization
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 0 u))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  apply TraceProofs.assign_cons (by simp [Op.SafeAt])
  intro zero
  apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
  cases hasSeed
  · rw [Stmt.TraceSafe]
    exact ⟨by simp [Op.SafeAt],by simp only [evalOp,Tile.scalar,Bool.false_eq_true,ite_false]⟩
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,↓reduceIte,true_and]
    rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (state_eval p (launch u) r.seed _
      (by simp [launch]) (by simp [launch]) (by simp [launch])))]
    refine ⟨by simp [Stmt.TraceSafe,statePtr,stateBase,times,Op.SafeAt],?_⟩
    apply TraceProofs.assign_cons
    · rw [Op.SafeAt.eq_def]
      refine ⟨by simp [carry,Op.SafeAt],?_⟩
      exact load2_safe r.seed ((launch u).head*p.K*p.V) p.K p.V p.BK p.BV p.V 1
        ((launch u).key*p.BK) ((launch u).value*p.BV) "p_h" bounds _ (hb rfl) (by simp [statePointer])
    · intro _
      exact Stmt.TraceSafeList.nil_intro

theorem final_safe (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool)
    (bounds : RegionBounds) (u : BlockState)
    (hb : storeFinal = Bool.true → StateBounds p l r.final bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)) :
    Stmt.TraceSafeList bounds (finalization p r storeFinal) u := by
  unfold finalization
  apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
  cases storeFinal
  · rw [Stmt.TraceSafe]
    exact ⟨by simp [Op.SafeAt],by simp only [evalOp,Tile.scalar,Bool.false_eq_true,ite_false]⟩
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt,evalOp,Tile.scalar,↓reduceIte,true_and]
    rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (state_eval p l r.final u hih hik hiv))]
    refine ⟨by simp [Stmt.TraceSafe,statePtr,stateBase,times,Op.SafeAt],?_⟩
    apply Stmt.TraceSafeList.cons_intro
      (store2_safe r.final (l.head*p.K*p.V) p.K p.V p.BK p.BV p.V 1 (l.key*p.BK) (l.value*p.BV)
        "p_h" "b_h" bounds _ (hb rfl) (by simp [statePointer]))
    intro _ _
    exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHStateSafety
