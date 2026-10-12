import VeriTile.Triton.Semantics.Integer
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.EvalOpR

/-! Signed integer division is exact in both FP execution interfaces. The
opaque floating backend supplies no law for these control operations. -/
namespace VeriTile.Triton

theorem evalOpR_truncDiv_of_eq (R : RoundingModel) {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) (s : BlockState)
    (x : Tile .int a) (y : Tile .int b)
    (hx : evalOpR R lhs s = some x) (hy : evalOpR R rhs s = some y) :
    evalOpR R (Op.truncDiv bc lhs rhs) s = some (Tile.bop Int.tdiv bc x y) := by
  simp only [Op.truncDiv, evalOpR, hx, hy, bind, pure, Option.bind_some, Tile.scalar, Tile.select, Tile.cop,
    Tile.bop, NumericDType.sub, NumericDType.add, IntegralDType.floorDiv, ComparableDType.lt,
    IntegerExpr.same_left, IntegerExpr.same_right]
  apply congrArg some
  apply Tile.ext
  intro i
  simp only [decide_eq_true_eq, zero_sub, zero_add, add_zero]
  exact IntegerExpr.tdiv_formula _ _

theorem FP.Structural.eval_truncDiv_of_eq {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (p : Option ComputeDType) {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) (s : FP.Structural.State α)
    (x : FP.Structural.Values α .int a) (y : FP.Structural.Values α .int b)
    (hx : FP.Structural.evalOp M p lhs s = some x)
    (hy : FP.Structural.evalOp M p rhs s = some y) :
    FP.Structural.evalOp M p (Op.truncDiv bc lhs rhs) s =
      some (FP.Structural.bop Int.tdiv bc x y) := by
  simp only [FP.Structural.evalOp, hx, hy, bind, pure, Option.bind_some,
    FP.Structural.bop, FP.Structural.numeric, FP.Structural.numericLt,
    FP.Structural.integralFloorDiv, IntegerExpr.same_left, IntegerExpr.same_right]
  apply congrArg some
  funext i
  simp only [decide_eq_true_eq, zero_sub, zero_add, add_zero]
  exact IntegerExpr.tdiv_formula _ _

theorem evalOpR_truncDiv (R : RoundingModel) {a b out : TileShape}
    (bc : Broadcast a b out) (lhs : Op .int a) (rhs : Op .int b) (s : BlockState) :
    evalOpR R (Op.truncDiv bc lhs rhs) s = (do
      let x ← evalOpR R lhs s
      let y ← evalOpR R rhs s
      pure (Tile.bop Int.tdiv bc x y)) := by
  cases hx : evalOpR R lhs s with
  | none => simp [Op.truncDiv, evalOpR, hx]
  | some x =>
      cases hy : evalOpR R rhs s with
      | none => simp [Op.truncDiv, evalOpR, hx, hy]
      | some y => exact evalOpR_truncDiv_of_eq R bc lhs rhs s x y hx hy

theorem evalOpR_truncMod_of_eq (R : RoundingModel) {a b out : TileShape}
    (bc : Broadcast a b out) (lhs : Op .int a) (rhs : Op .int b) (s : BlockState)
    (x : Tile .int a) (y : Tile .int b)
    (hx : evalOpR R lhs s = some x) (hy : evalOpR R rhs s = some y) :
    evalOpR R (Op.truncMod bc lhs rhs) s = some (Tile.bop Int.tmod bc x y) := by
  simp only [Op.truncMod, evalOpR, evalOpR_truncDiv_of_eq R bc lhs rhs s x y hx hy,
    hx, hy, bind, Option.bind_some, Tile.scalar, Tile.bop,
    NumericDType.sub, NumericDType.mul, NumericDType.add,
    IntegerExpr.same_left, IntegerExpr.same_right]
  apply congrArg some
  apply Tile.ext
  intro i
  simp only [zero_add, add_zero, Int.tmod_def]

theorem evalOpR_truncMod (R : RoundingModel) {a b out : TileShape}
    (bc : Broadcast a b out) (lhs : Op .int a) (rhs : Op .int b) (s : BlockState) :
    evalOpR R (Op.truncMod bc lhs rhs) s = (do
      let x ← evalOpR R lhs s
      let y ← evalOpR R rhs s
      pure (Tile.bop Int.tmod bc x y)) := by
  cases hx : evalOpR R lhs s with
  | none => simp [Op.truncMod, evalOpR, evalOpR_truncDiv, hx]
  | some x =>
      cases hy : evalOpR R rhs s with
      | none => simp [Op.truncMod, evalOpR, evalOpR_truncDiv, hx, hy]
      | some y => exact evalOpR_truncMod_of_eq R bc lhs rhs s x y hx hy

theorem FP.Structural.eval_truncDiv {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (p : Option ComputeDType) {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p (Op.truncDiv bc lhs rhs) s = (do
      let x ← FP.Structural.evalOp M p lhs s
      let y ← FP.Structural.evalOp M p rhs s
      pure (FP.Structural.bop Int.tdiv bc x y)) := by
  cases hx : FP.Structural.evalOp M p lhs s with
  | none => simp [FP.Structural.evalOp, hx]
  | some x =>
      cases hy : FP.Structural.evalOp M p rhs s with
      | none => simp [FP.Structural.evalOp, hx, hy]
      | some y => exact FP.Structural.eval_truncDiv_of_eq M p bc lhs rhs s x y hx hy

theorem FP.Structural.eval_truncMod_of_eq {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) {a b out : TileShape}
    (bc : Broadcast a b out) (lhs : Op .int a) (rhs : Op .int b) (s : FP.Structural.State α)
    (x : FP.Structural.Values α .int a) (y : FP.Structural.Values α .int b)
    (hx : FP.Structural.evalOp M p lhs s = some x)
    (hy : FP.Structural.evalOp M p rhs s = some y) :
    FP.Structural.evalOp M p (Op.truncMod bc lhs rhs) s =
      some (FP.Structural.bop Int.tmod bc x y) := by
  simp only [FP.Structural.evalOp,
    hx, hy, bind, pure, Option.bind_some, FP.Structural.numeric]
  apply congrArg some
  funext i
  simp only [FP.Structural.bop, IntegerExpr.same_left, IntegerExpr.same_right,
    zero_add, add_zero, Int.tmod_def, FP.Structural.integralFloorDiv,
    decide_eq_true_eq, zero_sub, IntegerExpr.tdiv_formula]

theorem FP.Structural.eval_truncMod {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (p : Option ComputeDType) {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p (Op.truncMod bc lhs rhs) s = (do
      let x ← FP.Structural.evalOp M p lhs s
      let y ← FP.Structural.evalOp M p rhs s
      pure (FP.Structural.bop Int.tmod bc x y)) := by
  cases hx : FP.Structural.evalOp M p lhs s with
  | none => simp [FP.Structural.evalOp, hx]
  | some x =>
      cases hy : FP.Structural.evalOp M p rhs s with
      | none => simp [FP.Structural.evalOp, hx, hy]
      | some y => exact FP.Structural.eval_truncMod_of_eq M p bc lhs rhs s x y hx hy

end VeriTile.Triton
