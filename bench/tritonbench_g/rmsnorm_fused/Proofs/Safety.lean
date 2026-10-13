import bench.tritonbench_g.rmsnorm_fused.Proofs.Candidate
import bench.tritonbench_g.rmsnorm_fused.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.RmsnormFused
open VeriTile.Triton
set_option maxHeartbeats 2000000

namespace Safety

theorem append_iff (bounds : RegionBounds) (before after : List Stmt) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds (before ++ after) s ↔
      Stmt.TraceSafeListR .triv bounds before s ∧
        ∀ t, stepStmts before s = some t → Stmt.TraceSafeListR .triv bounds after t := by
  induction before generalizing s with
  | nil => simp [Stmt.TraceSafeListR, stepStmts]
  | cons head tail ih =>
    simp only [List.cons_append, Stmt.TraceSafeListR, stepStmtR_triv, stepStmts]
    cases he : stepStmt head s with
    | none => simp
    | some t => simp only [Option.bind_some, ih, and_assoc]

theorem prefix_exec (Y : RegionName) (stride : Nat) (s : BlockState) :
    stepStmts (pointerPrefix Bool.true Y stride) s = stepStmts (pointerPrefix Bool.false Y stride) s := by
  simp [pointerPrefix, stepStmts, stepStmt, evalOp.eq_def, Tile.bop, NumericDType.mul, Nat.mul_comm]

theorem typed_safe_iff (bounds : RegionBounds) (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) (s : BlockState) :
    (typedKernel input weight output v X Y W stride N B eps).eraseDType.toAlgKernel.TraceSafeR .triv bounds s ↔
      (rms_norm_fwd_fused X Y W stride N B eps).toAlgKernel.TraceSafeR .triv bounds s := by
  rw [typed_projection]
  cases v
  · rfl
  · change (mathKernel Bool.true X Y W stride N B eps).TraceSafeR .triv bounds s ↔
      (mathKernel Bool.false X Y W stride N B eps).TraceSafeR .triv bounds s
    unfold Kernel.TraceSafeR
    rw [← List.take_append_drop 2 (mathKernel Bool.true X Y W stride N B eps).body,
      ← List.take_append_drop 2 (mathKernel Bool.false X Y W stride N B eps).body]
    have ht : (mathKernel Bool.true X Y W stride N B eps).body.drop 2 =
        (mathKernel Bool.false X Y W stride N B eps).body.drop 2 := rfl
    rw [prefix_source, prefix_source, ht, append_iff, append_iff, prefix_exec]
    have hp (v : Bool) (t : BlockState) : Stmt.TraceSafeListR .triv bounds (pointerPrefix v Y stride) t := by
      cases v <;> simp [pointerPrefix, Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
        stepStmtR_triv, stepStmt, evalOp.eq_def]
    simp only [hp, true_and]

end Safety

/-- Full source trace safety on the wrapper's launch domain, with no value or
execution premises. Pointer-prefix safety and execution are transported together. -/
theorem typed_trace_safe (input weight output : Storage) (v : Bool)
    (bounds : RegionBounds) (X Y W : RegionName) (stride N B : Nat) (eps : ℝ)
    (hNpos : 0 < N) (hNle : N ≤ B) (s : BlockState)
    (hbx : ∀ j : Fin B, j.val < N → s.pids 0 * stride + j.val < bounds X)
    (hbw : ∀ j : Fin B, j.val < N → j.val < bounds W)
    (hbo : ∀ j : Fin B, j.val < N → s.pids 0 * stride + j.val < bounds Y) :
    (typedKernel input weight output v X Y W stride N B eps).eraseDType.toAlgKernel.TraceSafeR .triv bounds s :=
  (Safety.typed_safe_iff bounds input weight output v X Y W stride N B eps s).mpr
    (legacy_trace_safe .triv bounds X Y W stride N B eps hNpos hNle s hbx hbw hbo)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormFused
