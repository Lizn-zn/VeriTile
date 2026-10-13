import bench.tritonbench_g.kldiv_triton.Proofs.Candidate

namespace VeriTile.Bench.TritonBenchG.KldivTriton.Candidate
open VeriTile.Triton TypedSource Source
set_option maxHeartbeats 3000000

theorem trace_append_iff (bounds : RegionBounds) (left right : List Stmt) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds (left ++ right) s ↔
      Stmt.TraceSafeListR .triv bounds left s ∧
        ∀ t, stepStmts left s = some t → Stmt.TraceSafeListR .triv bounds right t := by
  induction left generalizing s with
  | nil => simp [Stmt.TraceSafeListR, stepStmts]
  | cons head tail ih =>
    simp only [List.cons_append, Stmt.TraceSafeListR, stepStmtR_triv, stepStmts]
    cases he : stepStmt head s with
    | none => simp
    | some t => simp only [Option.bind_some, ih, and_assoc]

theorem forward_prefix_exec (Y T O : RegionName) (sy st so : Nat) (s : BlockState) :
    stepStmts (forwardPrefix Bool.true Y T O sy st so) s =
      stepStmts (forwardPrefix Bool.false Y T O sy st so) s := by
  simp [forwardPrefix, rowPointer, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.mul, Int.mul_comm]

theorem backward_prefix_exec (O : RegionName) (so : Nat) (s : BlockState) :
    stepStmts (backwardPrefix Bool.true O so) s =
      stepStmts (backwardPrefix Bool.false O so) s := by
  simp [backwardPrefix, rowPointer, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.mul, Int.mul_comm]

theorem forward_prefix_safe (bounds : RegionBounds) (v : Bool) (Y T O : RegionName)
    (sy st so : Nat) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds (forwardPrefix v Y T O sy st so) s := by
  apply Stmt.TraceSafeListR.of_forall
  intro stmt hm state
  simp only [forwardPrefix, List.mem_cons, List.not_mem_nil, or_false] at hm
  rcases hm with rfl | rfl | rfl | rfl <;>
    cases v <;> simp [rowPointer, Stmt.TraceSafeR, Op.SafeAtR.eq_def]

theorem backward_prefix_safe (bounds : RegionBounds) (v : Bool) (O : RegionName)
    (so : Nat) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds (backwardPrefix v O so) s := by
  apply Stmt.TraceSafeListR.of_forall
  intro stmt hm state
  simp only [backwardPrefix, List.mem_cons, List.not_mem_nil, or_false] at hm
  rcases hm with rfl | rfl <;>
    cases v <;> simp [rowPointer, Stmt.TraceSafeR, Op.SafeAtR.eq_def]

/-- The candidate preserves safety through equal actual prefix states, not merely equal final memory. -/
theorem forward_safe_iff (bounds : RegionBounds) (v : Bool) (Y T O : RegionName)
    (sy st so N B : Nat) (flag : Bool) (reduction : Nat) (s : BlockState) :
    (forwardMath v Y T O sy st so N B flag reduction).TraceSafeR .triv bounds s ↔
      (FullAlgorithm.forward Y T O sy st so N B flag reduction).TraceSafeR .triv bounds s := by
  cases v
  · rfl
  · change (forwardMath Bool.true Y T O sy st so N B flag reduction).TraceSafeR .triv bounds s ↔
      (forwardMath Bool.false Y T O sy st so N B flag reduction).TraceSafeR .triv bounds s
    unfold Kernel.TraceSafeR
    rw [← List.take_append_drop 4 (forwardMath Bool.true Y T O sy st so N B flag reduction).body,
      ← List.take_append_drop 4 (forwardMath Bool.false Y T O sy st so N B flag reduction).body]
    have ht : (forwardMath Bool.true Y T O sy st so N B flag reduction).body.drop 4 =
        (forwardMath Bool.false Y T O sy st so N B flag reduction).body.drop 4 := rfl
    rw [forward_prefix_source, forward_prefix_source]
    simp only [trace_append_iff, forward_prefix_safe, true_and, forward_prefix_exec, ht]

theorem backward_safe_iff (bounds : RegionBounds) (v : Bool) (T O : RegionName)
    (st so N B : Nat) (flag : Bool) (s : BlockState) :
    (backwardMath v T O st so N B flag).TraceSafeR .triv bounds s ↔
      (FullAlgorithm.backward T O st so N B flag).TraceSafeR .triv bounds s := by
  cases v
  · rfl
  · change (backwardMath Bool.true T O st so N B flag).TraceSafeR .triv bounds s ↔
      (backwardMath Bool.false T O st so N B flag).TraceSafeR .triv bounds s
    unfold Kernel.TraceSafeR
    rw [← List.take_append_drop 2 (backwardMath Bool.true T O st so N B flag).body,
      ← List.take_append_drop 2 (backwardMath Bool.false T O st so N B flag).body]
    have ht : (backwardMath Bool.true T O st so N B flag).body.drop 2 =
        (backwardMath Bool.false T O st so N B flag).body.drop 2 := by cases flag <;> rfl
    rw [backward_prefix_source, backward_prefix_source]
    simp only [trace_append_iff, backward_prefix_safe, true_and, backward_prefix_exec, ht]

end VeriTile.Bench.TritonBenchG.KldivTriton.Candidate
#guard_msgs (drop info) in
#auditModuleAxioms
