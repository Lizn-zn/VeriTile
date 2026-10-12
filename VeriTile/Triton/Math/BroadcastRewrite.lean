import VeriTile.Triton.Memory.TraceProofs

/-! Real division/reciprocal rewriting through arbitrary broadcasting. These
lemmas concern the mathematical interpreter only; FP specifications use the
separate admitted relation in Float/BroadcastRewrite. -/
namespace VeriTile.Triton.RealBroadcastRewrite

private theorem reciprocal_value (a b : WithBot ℝ) :
    WithBot.realDiv a b = WithBot.realMul a (WithBot.realDiv (some 1) b) := by
  change Option.map₂ (· / ·) a b =
    Option.map₂ (· * ·) a (Option.map₂ (· / ·) (some 1) b)
  cases a using Option.rec <;> cases b using Option.rec <;>
    simp [Option.map₂, div_eq_mul_inv]

theorem eval_div_mul_rcp (bc : Broadcast sa (n :: rest) shape)
    (a : Op .real sa) (b : Op .real (n :: rest)) (one : ℝ) (hone : one = 1)
    (s : BlockState) :
    evalOp (.div .real bc a b) s =
      evalOp (.mul .real bc a (.div .real .scalarL (.const one) b)) s := by
  subst one
  simp only [evalOp_div, evalOp_mul, evalOp_const]
  cases ha : evalOp a s <;> cases hb : evalOp b s
  all_goals simp only [Option.bind_eq_bind, Option.bind_none, Option.bind_some]
  rename_i va vb
  congr 1
  ext i
  exact reciprocal_value _ _

theorem assign_div_mul_rcp (bc : Broadcast sa (n :: rest) shape)
    (out : RegName) (a : Op .real sa) (b : Op .real (n :: rest))
    (one : ℝ) (hone : one = 1) (s : BlockState) :
    stepStmt (.assign .real shape out (.div .real bc a b)) s =
      stepStmt (.assign .real shape out
        (.mul .real bc a (.div .real .scalarL (.const one) b))) s := by
  simp only [stepStmt, eval_div_mul_rcp bc a b one hone]

/-- The unchanged prefix and suffix retain all their memory effects and failures. -/
theorem contextual_div_mul_rcp (before after : List Stmt)
    (bc : Broadcast sa (n :: rest) shape) (out : RegName)
    (a : Op .real sa) (b : Op .real (n :: rest)) (one : ℝ) (hone : one = 1)
    (s : BlockState) :
    stepStmts (before ++ .assign .real shape out (.div .real bc a b) :: after) s =
      stepStmts (before ++ .assign .real shape out
        (.mul .real bc a (.div .real .scalarL (.const one) b)) :: after) s := by
  induction before generalizing s with
  | nil => simp only [List.nil_append, stepStmts, assign_div_mul_rcp bc out a b one hone]
  | cons st tail ih =>
      simp only [List.cons_append, stepStmts]
      cases stepStmt st s with
      | none => rfl
      | some t => exact ih t

/-- Rewriting the numerical expression introduces no new memory accesses. -/
theorem contextual_traceSafe (before after : List Stmt)
    (bc : Broadcast sa (n :: rest) shape) (out : RegName)
    (a : Op .real sa) (b : Op .real (n :: rest)) (one : ℝ) (hone : one = 1)
    (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds
        (before ++ .assign .real shape out (.div .real bc a b) :: after) s ↔
      Stmt.TraceSafeList bounds (before ++ .assign .real shape out
        (.mul .real bc a (.div .real .scalarL (.const one) b)) :: after) s := by
  induction before generalizing s with
  | nil =>
      have hs : Stmt.TraceSafe bounds (.assign .real shape out (.div .real bc a b)) s ↔
          Stmt.TraceSafe bounds (.assign .real shape out
            (.mul .real bc a (.div .real .scalarL (.const one) b))) s := by
        conv_lhs =>
          rw [Stmt.TraceSafe, Op.SafeAt.eq_def]
          dsimp only
        conv_rhs =>
          rw [Stmt.TraceSafe, Op.SafeAt.eq_def]
          dsimp only
          arg 2
          rw [Op.SafeAt.eq_def]
          dsimp only
          arg 1
          rw [Op.SafeAt.eq_def]
          dsimp only
        simp only [true_and]
      simp only [List.nil_append, Stmt.TraceSafeList, hs,
        assign_div_mul_rcp bc out a b one hone]
  | cons st tail ih =>
      simp only [List.cons_append, Stmt.TraceSafeList]
      cases stepStmt st s with
      | none => rfl
      | some t => exact and_congr_right (fun _ => ih t)

end VeriTile.Triton.RealBroadcastRewrite
