import VeriTile.Triton.Memory.TraceProofs
import VeriTile.Triton.Semantics.ProgramRewrite

/-! Safety transport for program rewrites. Execution equality is accompanied by
an explicit safety equivalence; equal final states alone do not imply equal
memory-access safety. -/
namespace VeriTile.Triton.TraceRewrite

theorem append_iff (bounds : RegionBounds) (before after : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (before ++ after) s ↔
      Stmt.TraceSafeList bounds before s ∧
        ∀ t, stepStmts before s = some t → Stmt.TraceSafeList bounds after t := by
  induction before generalizing s with
  | nil => simp [Stmt.TraceSafeList]
  | cons st rest ih =>
    cases he : stepStmt st s <;> simp [Stmt.TraceSafeList, stepStmts, he, ih, and_assoc]

/-- A constant branch preserves safety for the selected body, including a
body that fails before the following code is reached. -/
theorem constant_branch (bounds : RegionBounds) (condition : Bool)
    (body after : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (.ifThen (.constBool condition) body :: after) s ↔
      Stmt.TraceSafeList bounds ((if condition then body else []) ++ after) s := by
  cases condition
  · simp [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp, stepStmt]
  · cases he : stepStmts body s <;>
      simp [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp, stepStmt,
        append_iff, he]

/-- Lift both execution and safety preservation through a dynamic range. -/
theorem range_congr (bounds : RegionBounds) (index : RegName) (cur stop stride : Nat)
    (lhs rhs : List Stmt) (hexec : ∀ s, stepStmts lhs s = stepStmts rhs s)
    (hsafe : ∀ s, Stmt.TraceSafeList bounds lhs s ↔ Stmt.TraceSafeList bounds rhs s)
    (s : BlockState) :
    Stmt.forRangeTraceSafe bounds index cur stop stride lhs s ↔
      Stmt.forRangeTraceSafe bounds index cur stop stride rhs s := by
  rw [Stmt.forRangeTraceSafe, Stmt.forRangeTraceSafe]
  by_cases hz : stride = 0
  · simp only [hz, ↓reduceIte]
  · simp only [hz, ↓reduceIte]
    by_cases hc : cur < stop
    · simp only [hc, ↓reduceIte, hsafe, hexec]
      cases stepStmts rhs (s.setReg index .nat [] (Tile.scalar cur)) with
      | none => rfl
      | some t => exact and_congr_right (fun _ => range_congr bounds index
          (cur + stride) stop stride lhs rhs hexec hsafe t)
    · simp only [hc, ↓reduceIte]
termination_by stop - cur
decreasing_by omega

theorem dynamic_range_congr (bounds : RegionBounds) (index : RegName)
    (start stop stride : Op .nat []) (lhs rhs : List Stmt)
    (hexec : ∀ s, stepStmts lhs s = stepStmts rhs s)
    (hsafe : ∀ s, Stmt.TraceSafeList bounds lhs s ↔ Stmt.TraceSafeList bounds rhs s)
    (s : BlockState) :
    Stmt.TraceSafe bounds (.forRangeDyn index start stop stride lhs) s ↔
      Stmt.TraceSafe bounds (.forRangeDyn index start stop stride rhs) s := by
  simp only [Stmt.TraceSafe]
  cases evalOp start s <;> cases evalOp stop s <;> cases evalOp stride s <;>
    simp only [range_congr bounds index _ _ _ lhs rhs hexec hsafe]

end VeriTile.Triton.TraceRewrite
