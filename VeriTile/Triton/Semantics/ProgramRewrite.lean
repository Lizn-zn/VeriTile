import VeriTile.Triton.Semantics.Step

/-! Execution congruence for program fragments. These lemmas preserve the
complete state and failure result; they assume no numerical rewrite law. -/
namespace VeriTile.Triton.ProgramRewrite

theorem append (before after : List Stmt) (s : BlockState) :
    stepStmts (before ++ after) s = (stepStmts before s).bind (stepStmts after) := by
  induction before generalizing s with
  | nil => simp only [List.nil_append, stepStmts.nil, Option.bind_some]
  | cons st tail ih =>
    simp only [List.cons_append, stepStmts]
    cases stepStmt st s with
    | none => rfl
    | some t => exact ih t

theorem context (before after lhs rhs : List Stmt)
    (h : ∀ s, stepStmts lhs s = stepStmts rhs s) (s : BlockState) :
    stepStmts (before ++ lhs ++ after) s = stepStmts (before ++ rhs ++ after) s := by
  simp only [append, h]

/-- Inline a constant branch without assuming that its body succeeds. -/
theorem constant_branch (condition : Bool) (body after : List Stmt) (s : BlockState) :
    stepStmts (.ifThen (.constBool condition) body :: after) s =
      stepStmts ((if condition then body else []) ++ after) s := by
  cases condition <;> simp only [stepStmts, stepStmt, evalOp, Tile.scalar_data,
    bind, Option.bind, Bool.false_eq_true, ↓reduceIte, List.nil_append, append]
  cases stepStmts body s <;> rfl

/-- Lift a checked body rewrite through every reached range iteration,
including zero stride and bounds that produce no iterations. -/
theorem range_congr (index : RegName) (cur stop stride : Nat)
    (lhs rhs : List Stmt) (hbody : ∀ s, stepStmts lhs s = stepStmts rhs s)
    (s : BlockState) :
    stepForRangeAux index cur stop stride lhs s =
      stepForRangeAux index cur stop stride rhs s := by
  by_cases hz : stride = 0
  · subst stride
    simp only [stepForRangeAux.step_zero_step]
  · have hs : 0 < stride := Nat.pos_of_ne_zero hz
    by_cases hc : cur < stop
    · rw [stepForRangeAux.step_lt hz hc, stepForRangeAux.step_lt hz hc, hbody]
      cases stepStmts rhs (s.setReg index .nat [] (Tile.scalar cur)) with
      | none => rfl
      | some st => exact range_congr index (cur + stride) stop stride lhs rhs hbody st
    · rw [stepForRangeAux.step_ge hz (by omega), stepForRangeAux.step_ge hz (by omega)]
termination_by stop - cur
decreasing_by omega

theorem dynamic_range_congr (index : RegName) (start stop stride : Op .nat [])
    (lhs rhs : List Stmt) (hbody : ∀ s, stepStmts lhs s = stepStmts rhs s)
    (s : BlockState) :
    stepStmt (.forRangeDyn index start stop stride lhs) s =
      stepStmt (.forRangeDyn index start stop stride rhs) s := by
  simp only [stepForRangeAux.forRangeDyn_unfold]
  congr 1
  funext a
  congr 1
  funext b
  congr 1
  funext c
  exact range_congr _ _ _ _ _ _ hbody s

end VeriTile.Triton.ProgramRewrite
