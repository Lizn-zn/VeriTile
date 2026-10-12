import VeriTile.Triton.Core.Ast

/-! Runtime signed integer division and remainder. These are expression
constructors made from exact integer primitives; floating algebra laws are
neither used nor assumed. Raw `Op.floorDiv` and `Op.mod` keep their existing
Euclidean convention. Python constexpr calculations belong in Lean
antiquotations (using `Int.fdiv`/`Int.fmod` when signs can differ).
The integer carrier is mathematical Int; these definitions do not model
fixed-width overflow or make division by zero a valid GPU operation.
The constructors are transparent abbreviations so existing evaluator proofs
can still inspect the AST directly, including the unchanged Nat cases. -/
namespace VeriTile.Triton

namespace IntegerExpr
def sameShape : (sh : TileShape) → Broadcast sh sh sh
  | [] => .nil
  | _ :: sh => .consSame (sameShape sh)

@[simp] theorem same_left (sh : TileShape) (i : TileIndex sh) :
    (sameShape sh).leftIndex i = i := by
  induction sh with
  | nil => cases i; rfl
  | cons n sh ih =>
      cases i with
      | mk i is => simp [sameShape, ih]

@[simp] theorem same_right (sh : TileShape) (i : TileIndex sh) :
    (sameShape sh).rightIndex i = i := by
  induction sh with
  | nil => cases i; rfl
  | cons n sh ih =>
      cases i with
      | mk i is => simp [sameShape, ih]

end IntegerExpr
open IntegerExpr

/-- Runtime signed division truncates toward zero. The existing Euclidean
primitive is reused only after making the dividend nonnegative. -/
abbrev Op.truncDiv {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) : Op .int out :=
  let x := Op.add .int bc lhs (Op.full b (.constInt 0))
  let y := Op.add .int bc (Op.full a (.constInt 0)) rhs
  let z := Op.full out (.constInt 0)
  let same := sameShape out
  .where (.lt .int same x z)
    (.sub .int same z (.floorDiv .int same (.sub .int same z x) y))
    (.floorDiv .int same x y)

/-- Runtime remainder has the dividend's sign, corresponding to C division. -/
abbrev Op.truncMod {a b out : TileShape} (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) : Op .int out :=
  let x := Op.add .int bc lhs (Op.full b (.constInt 0))
  let y := Op.add .int bc (Op.full a (.constInt 0)) rhs
  .sub .int (sameShape out) x
    (.mul .int (sameShape out) y (Op.truncDiv bc lhs rhs))

/-- Runtime integer `//`; the Nat branch retains the original AST. -/
abbrev Op.tritonDiv {dtype : TileDType} (h : IntegralDType dtype)
    (bc : Broadcast a b out) (lhs : Op dtype a) (rhs : Op dtype b) : Op dtype out :=
  match h with
  | .nat => .floorDiv .nat bc lhs rhs
  | .int => .truncDiv bc lhs rhs

/-- Runtime integer `%`, paired with `tritonDiv`. -/
abbrev Op.tritonMod {dtype : TileDType} (h : IntegralDType dtype)
    (bc : Broadcast a b out) (lhs : Op dtype a) (rhs : Op dtype b) : Op dtype out :=
  match h with
  | .nat => .mod .nat bc lhs rhs
  | .int => .truncMod bc lhs rhs

end VeriTile.Triton
