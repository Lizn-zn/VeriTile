import VeriTile.Triton.Core.Integer
import VeriTile.Triton.Semantics.EvalOp
import Mathlib.Data.Int.DivMod

/-! Evaluation of the runtime integer constructors, including operand failure.
The formulas are integer identities, not floating rewrite assumptions. -/
namespace VeriTile.Triton

theorem IntegerExpr.tdiv_formula (x y : Int) :
    (if x < 0 then -((-x) / y) else x / y) = x.tdiv y := by
  by_cases hx : x < 0
  · rw [if_pos hx, ← Int.tdiv_eq_ediv_of_nonneg (by omega), Int.neg_tdiv, Int.neg_neg]
  · rw [if_neg hx, Int.tdiv_eq_ediv_of_nonneg (by omega)]

theorem eval_truncDiv_of_eq {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) (s : BlockState)
    (x : Tile .int a) (y : Tile .int b)
    (hx : evalOp lhs s = some x) (hy : evalOp rhs s = some y) :
    evalOp (Op.truncDiv bc lhs rhs) s = some (Tile.bop Int.tdiv bc x y) := by
  simp only [Op.truncDiv, evalOp, hx, hy, bind, pure, Option.bind_some, Tile.scalar, Tile.select, Tile.cop,
    Tile.bop, NumericDType.sub, NumericDType.add, IntegralDType.floorDiv, ComparableDType.lt,
    IntegerExpr.same_left, IntegerExpr.same_right]
  apply congrArg some
  apply Tile.ext
  intro i
  simp only [decide_eq_true_eq, zero_sub, zero_add, add_zero]
  exact IntegerExpr.tdiv_formula _ _

theorem eval_truncDiv {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) (s : BlockState) :
    evalOp (Op.truncDiv bc lhs rhs) s = (do
      let x ← evalOp lhs s
      let y ← evalOp rhs s
      pure (Tile.bop Int.tdiv bc x y)) := by
  cases hx : evalOp lhs s with
  | none => simp [Op.truncDiv, evalOp, hx]
  | some x =>
      cases hy : evalOp rhs s with
      | none => simp [Op.truncDiv, evalOp, hx, hy]
      | some y => exact eval_truncDiv_of_eq bc lhs rhs s x y hx hy

theorem eval_truncMod_of_eq {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) (s : BlockState)
    (x : Tile .int a) (y : Tile .int b)
    (hx : evalOp lhs s = some x) (hy : evalOp rhs s = some y) :
    evalOp (Op.truncMod bc lhs rhs) s = some (Tile.bop Int.tmod bc x y) := by
  simp only [Op.truncMod, evalOp, eval_truncDiv_of_eq bc lhs rhs s x y hx hy,
    hx, hy, bind, Option.bind_some, Tile.scalar, Tile.bop,
    NumericDType.sub, NumericDType.mul, NumericDType.add,
    IntegerExpr.same_left, IntegerExpr.same_right]
  apply congrArg some
  apply Tile.ext
  intro i
  simp only [zero_add, add_zero, Int.tmod_def]

theorem eval_truncMod {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) (s : BlockState) :
    evalOp (Op.truncMod bc lhs rhs) s = (do
      let x ← evalOp lhs s
      let y ← evalOp rhs s
      pure (Tile.bop Int.tmod bc x y)) := by
  cases hx : evalOp lhs s with
  | none => simp [Op.truncMod, evalOp, hx]
  | some x =>
      cases hy : evalOp rhs s with
      | none => simp [Op.truncMod, evalOp, hx, hy]
      | some y => exact eval_truncMod_of_eq bc lhs rhs s x y hx hy

end VeriTile.Triton
