import VeriTile.Triton.Semantics

/-! Real pointwise rewrites. These use real arithmetic only and are never
imported by the FP equivalence specifications. -/
namespace VeriTile.Triton.MatrixRewrite
set_option maxHeartbeats 1600000

private theorem real_add_comm (a b : WithBot ℝ) : WithBot.realAdd a b = WithBot.realAdd b a := by
  cases a <;> cases b
  · rfl
  · rfl
  · rfl
  · simp [WithBot.realAdd, add_comm]

private theorem transfer (before after : List Stmt) (lhs rhs : Stmt) (s t : BlockState)
    (hs : ∀ u, stepStmt lhs u = stepStmt rhs u)
    (h : stepStmts (before ++ lhs :: after) s = some t) :
    stepStmts (before ++ rhs :: after) s = some t := by
  obtain ⟨u, hu, ht⟩ := stepStmts.append_some_iff.mp h
  apply stepStmts.append_some_iff.mpr
  refine ⟨u, hu, ?_⟩
  simpa only [stepStmts, hs u] using ht

private theorem eval_add (a b : Op .real [S, D]) (s : BlockState) :
    evalOp (.add .real (.consSame (.consSame .nil)) a b) s =
      ((evalOp a s).bind fun va => (evalOp b s).bind fun vb =>
        some (Tile.bop WithBot.realAdd (.consSame (.consSame .nil)) va vb)) := by
  simp only [evalOp_add]
  rfl

private theorem eval_div (a : Op .real [S, D]) (tau : ℝ) (s : BlockState) :
    evalOp (.div .real .scalarR a (.const tau)) s =
      ((evalOp a s).bind fun va =>
        some (Tile.bop WithBot.realDiv .scalarR va (Tile.scalar (tau : WithBot ℝ)))) := by
  simp only [evalOp_div, evalOp_const]
  rfl

private theorem eval_mul_rcp (a : Op .real [S, D]) (tau : ℝ) (s : BlockState) :
    evalOp (.mul .real .scalarR a (.div .real .nil (.const 1) (.const tau))) s =
      ((evalOp a s).bind fun va =>
        some (Tile.bop WithBot.realMul .scalarR va (Tile.scalar ((1 / tau : ℝ) : WithBot ℝ)))) := by
  simp only [evalOp_mul, evalOp_div, evalOp_const]
  rfl

theorem add_commute (before after : List Stmt) (out : String)
    (a b : Op .real [S, D]) (s t : BlockState)
    (h : stepStmts (before ++ .assign .real [S, D] out
      (.add .real (.consSame (.consSame .nil)) a b) :: after) s = some t) :
    stepStmts (before ++ .assign .real [S, D] out
      (.add .real (.consSame (.consSame .nil)) b a) :: after) s = some t := by
  apply transfer before after _ _ s t (fun u => ?_) h
  simp only [stepStmt, eval_add]
  cases ha : evalOp a u <;> cases hb : evalOp b u
  all_goals simp only [Option.bind_none, Option.bind_some]
  rename_i va vb
  congr 2
  ext i
  exact real_add_comm _ _

theorem div_mul_rcp (before after : List Stmt) (out : String)
    (a : Op .real [S, D]) (tau : ℝ) (s t : BlockState)
    (h : stepStmts (before ++ .assign .real [S, D] out
      (.div .real .scalarR a (.const tau)) :: after) s = some t) :
    stepStmts (before ++ .assign .real [S, D] out
      (.mul .real .scalarR a (.div .real .nil (.const 1) (.const tau))) :: after) s = some t := by
  apply transfer before after _ _ s t (fun u => ?_) h
  simp only [stepStmt, eval_div, eval_mul_rcp]
  cases ha : evalOp a u
  · rfl
  rename_i va
  simp only [Option.bind_some]
  congr 2
  ext i
  cases hv : va.data i <;>
    simp [Tile.bop, WithBot.realDiv, WithBot.realMul, hv, div_eq_mul_inv]

end VeriTile.Triton.MatrixRewrite
