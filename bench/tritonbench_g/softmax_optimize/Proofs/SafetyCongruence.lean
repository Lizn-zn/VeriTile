import VeriTile.Triton.Memory.TraceProofs
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton

theorem trace_append_iff (bounds : RegionBounds) (before after : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (before ++ after) s ↔
      Stmt.TraceSafeList bounds before s ∧
        ∀ t, stepStmts before s = some t → Stmt.TraceSafeList bounds after t := by
  induction before generalizing s with
  | nil => simp [Stmt.TraceSafeList, stepStmts]
  | cons head tail ih =>
    simp only [List.cons_append, Stmt.TraceSafeList, stepStmts]
    cases heq : stepStmt head s with
    | none => simp
    | some next => simp only [Option.bind_some, ih, and_assoc]

/-- Equally safe prefixes with identical optional successor states can be
replaced before any common suffix, including when either prefix fails. -/
theorem trace_prefix_congr (bounds : RegionBounds) (left right suffix : List Stmt)
    (hsafe : ∀ s, Stmt.TraceSafeList bounds left s ↔ Stmt.TraceSafeList bounds right s)
    (hrun : ∀ s, stepStmts left s = stepStmts right s) (s : BlockState) :
    Stmt.TraceSafeList bounds (left ++ suffix) s ↔
      Stmt.TraceSafeList bounds (right ++ suffix) s := by
  rw [trace_append_iff, trace_append_iff, hsafe, hrun]

/-- Signed-range safety is extensional in the body's safety and execution.
The initial counter and stop may be negative; zero stride remains total. -/
theorem signed_trace_congr (bounds : RegionBounds) (index : RegName)
    (stop : Int) (stride : Nat) (left right : List Stmt)
    (hsafe : ∀ s, Stmt.TraceSafeList bounds left s ↔ Stmt.TraceSafeList bounds right s)
    (hrun : ∀ s, stepStmts left s = stepStmts right s) (cur : Int) (s : BlockState) :
    Stmt.forRangeIntTraceSafe bounds index cur stop stride left s ↔
      Stmt.forRangeIntTraceSafe bounds index cur stop stride right s := by
  generalize hn : (stop - cur).toNat = n
  induction n using Nat.strong_induction_on generalizing cur s with
  | h n ih =>
    conv_lhs => rw [Stmt.forRangeIntTraceSafe]
    conv_rhs => rw [Stmt.forRangeIntTraceSafe]
    by_cases hz : stride = 0
    · simp only [hz, ↓reduceIte]
    simp only [hz, ↓reduceIte]
    by_cases hc : cur < stop
    · simp only [hc, ↓reduceIte, hsafe, hrun]
      cases he : stepStmts right (s.setReg index .int [] (Tile.scalar cur)) with
      | none => rfl
      | some t =>
        apply and_congr_right
        intro _
        exact ih (stop - (cur + stride)).toNat (by omega) (cur + stride) t rfl
    · simp only [hc, ↓reduceIte]

theorem signed_statement_trace_congr (bounds : RegionBounds) (index : RegName)
    (start stop : Op .int []) (stride : Op .nat []) (left right : List Stmt)
    (hsafe : ∀ s, Stmt.TraceSafeList bounds left s ↔ Stmt.TraceSafeList bounds right s)
    (hrun : ∀ s, stepStmts left s = stepStmts right s) (s : BlockState) :
    Stmt.TraceSafe bounds (.forRangeInt index start stop stride left) s ↔
      Stmt.TraceSafe bounds (.forRangeInt index start stop stride right) s := by
  simp only [Stmt.TraceSafe]
  cases evalOp start s <;> cases evalOp stop s <;> cases evalOp stride s <;>
    simp only [signed_trace_congr bounds index _ _ left right hsafe hrun]

theorem trace_cons_congr (bounds : RegionBounds) (a b : Stmt) (left right : List Stmt)
    (hsafe : ∀ s, Stmt.TraceSafe bounds a s ↔ Stmt.TraceSafe bounds b s)
    (hrun : ∀ s, stepStmt a s = stepStmt b s)
    (htail : ∀ s, Stmt.TraceSafeList bounds left s ↔ Stmt.TraceSafeList bounds right s)
    (s : BlockState) :
    Stmt.TraceSafeList bounds (a :: left) s ↔ Stmt.TraceSafeList bounds (b :: right) s := by
  simp only [Stmt.TraceSafeList, hsafe, hrun]
  cases stepStmt b s <;> simp only [htail]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
