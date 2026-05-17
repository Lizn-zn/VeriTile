/-
VeriTile.Triton.Semantics.Scalar

Scalar carrier operations and typed dtype witness semantics.
-/

import Mathlib.Data.Real.Basic
import VeriTile.Triton.Core

namespace VeriTile.Triton

namespace Tile

@[simp] theorem scalar_data {dtype : TileDType} (x : TileCarrier dtype) :
    (Tile.scalar x).data PUnit.unit = x := rfl

@[simp] theorem scalar_data_index {dtype : TileDType}
    (x : TileCarrier dtype) (i : TileIndex []) :
    (Tile.scalar x).data i = x := by
  cases i
  rfl

@[simp] theorem scalar_eta {dtype : TileDType} (x : TileCarrier dtype) :
    ({ data := fun _ : TileIndex [] => x } : Tile dtype []) =
      Tile.scalar x := rfl

@[simp] theorem vec_data {dtype : TileDType} {n : Nat}
    (f : Fin n → TileCarrier dtype) (i : TileIndex [n]) :
    (Tile.vec f).data i = f i.1 := rfl

@[simp] theorem vec_data_fin {dtype : TileDType} {n : Nat}
    (f : Fin n → TileCarrier dtype) (i : Fin n) :
    (Tile.vec f).data (i, PUnit.unit) = f i := rfl

end Tile

/-- `⊥`-propagating arithmetic on `WithBot ℝ`.

`Option.map₂ f` returns `none` if either argument is `none`, and `some (f a b)`
on two `some` values. This gives uniform ⊥-propagation for sub/mul/div without
having to chase Mathlib's `WithBot` typeclass instances (which don't always
align with our intended IEEE-style propagation — e.g. Mathlib's `WithBot.Mul`
defines `0 * ⊥ = 0`, but we want `0 * ⊥ = ⊥`). -/
@[simp] def WithBot.realAdd : WithBot ℝ → WithBot ℝ → WithBot ℝ := Option.map₂ (· + ·)
@[simp] def WithBot.realSub : WithBot ℝ → WithBot ℝ → WithBot ℝ := Option.map₂ (· - ·)
@[simp] def WithBot.realMul : WithBot ℝ → WithBot ℝ → WithBot ℝ := Option.map₂ (· * ·)
@[simp] noncomputable def WithBot.realDiv : WithBot ℝ → WithBot ℝ → WithBot ℝ := Option.map₂ (· / ·)

namespace NumericDType

def add : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => WithBot.realAdd x y
  | .fp32, x, y => WithBot.realAdd x y
  | .fp16, x, y => WithBot.realAdd x y
  | .bf16, x, y => WithBot.realAdd x y
  | .int, x, y => x + y
  | .nat, x, y => x + y

def sub : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => WithBot.realSub x y
  | .fp32, x, y => WithBot.realSub x y
  | .fp16, x, y => WithBot.realSub x y
  | .bf16, x, y => WithBot.realSub x y
  | .int, x, y => x - y
  | .nat, x, y => x - y

def mul : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => WithBot.realMul x y
  | .fp32, x, y => WithBot.realMul x y
  | .fp16, x, y => WithBot.realMul x y
  | .bf16, x, y => WithBot.realMul x y
  | .int, x, y => x * y
  | .nat, x, y => x * y

noncomputable def div : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => WithBot.realDiv x y
  | .fp32, x, y => WithBot.realDiv x y
  | .fp16, x, y => WithBot.realDiv x y
  | .bf16, x, y => WithBot.realDiv x y
  | .int, x, y => x / y
  | .nat, x, y => x / y

/-- `.nat` channel addition unfolds to `Nat.+`. -/
theorem nat_add (x y : Nat) : NumericDType.add .nat x y = x + y := rfl

/-- `.int` channel addition unfolds to `Int.+`. -/
theorem int_add (x y : Int) : NumericDType.add .int x y = x + y := rfl

/-- `.nat` channel subtraction unfolds to `Nat.-` (truncated). -/
theorem nat_sub (x y : Nat) : NumericDType.sub .nat x y = x - y := rfl

/-- `.int` channel subtraction unfolds to `Int.-`. -/
theorem int_sub (x y : Int) : NumericDType.sub .int x y = x - y := rfl

/-- `.nat` channel multiplication unfolds to `Nat.*`. -/
theorem nat_mul (x y : Nat) : NumericDType.mul .nat x y = x * y := rfl

/-- `.int` channel multiplication unfolds to `Int.*`. -/
theorem int_mul (x y : Int) : NumericDType.mul .int x y = x * y := rfl

end NumericDType

namespace IntegralDType

def floorDiv : IntegralDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .int, x, y => x / y
  | .nat, x, y => x / y

def mod : IntegralDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .int, x, y => x % y
  | .nat, x, y => x % y

/-- `IntegralDType.nat.floorDiv` unfolds to natural-number `/`. -/
@[simp] theorem nat_floorDiv (x y : Nat) :
    IntegralDType.floorDiv .nat x y = x / y := rfl

/-- `IntegralDType.int.floorDiv` unfolds to integer `/`. -/
@[simp] theorem int_floorDiv (x y : Int) :
    IntegralDType.floorDiv .int x y = x / y := rfl

/-- `IntegralDType.nat.mod` unfolds to natural-number `%`. -/
@[simp] theorem nat_mod (x y : Nat) :
    IntegralDType.mod .nat x y = x % y := rfl

/-- `IntegralDType.int.mod` unfolds to integer `%`. -/
@[simp] theorem int_mod (x y : Int) :
    IntegralDType.mod .int x y = x % y := rfl

end IntegralDType

namespace ComparableDType

noncomputable def lt :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x < y)
  | .fp32, x, y => decide (x < y)
  | .fp16, x, y => decide (x < y)
  | .bf16, x, y => decide (x < y)
  | .int, x, y => decide (x < y)
  | .nat, x, y => decide (x < y)

noncomputable def le :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x ≤ y)
  | .fp32, x, y => decide (x ≤ y)
  | .fp16, x, y => decide (x ≤ y)
  | .bf16, x, y => decide (x ≤ y)
  | .int, x, y => decide (x ≤ y)
  | .nat, x, y => decide (x ≤ y)

noncomputable def eq :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x = y)
  | .fp32, x, y => decide (x = y)
  | .fp16, x, y => decide (x = y)
  | .bf16, x, y => decide (x = y)
  | .int, x, y => decide (x = y)
  | .nat, x, y => decide (x = y)

noncomputable def gt :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x > y)
  | .fp32, x, y => decide (x > y)
  | .fp16, x, y => decide (x > y)
  | .bf16, x, y => decide (x > y)
  | .int, x, y => decide (x > y)
  | .nat, x, y => decide (x > y)

noncomputable def ge :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x ≥ y)
  | .fp32, x, y => decide (x ≥ y)
  | .fp16, x, y => decide (x ≥ y)
  | .bf16, x, y => decide (x ≥ y)
  | .int, x, y => decide (x ≥ y)
  | .nat, x, y => decide (x ≥ y)

@[simp] theorem nat_ge_eq_true (x y : Nat) :
    (ComparableDType.nat.ge x y = Bool.true) = (y ≤ x) := by
  simp [ge]

noncomputable def ne :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x ≠ y)
  | .fp32, x, y => decide (x ≠ y)
  | .fp16, x, y => decide (x ≠ y)
  | .bf16, x, y => decide (x ≠ y)
  | .int, x, y => decide (x ≠ y)
  | .nat, x, y => decide (x ≠ y)

/-! ### Bridge lemmas: `ComparableDType.{gt,lt,ge,le,eq,ne} = Bool.true ↔ _`

These let `simp` unfold the Bool-valued comparison forms produced by
`tl.maximum` / `tl.minimum` / `tl.where(... > ...)` / `tl.where(... != ...)`
into ordinary propositional inequalities on the underlying carrier. Without
these, bench proofs end up stuck on `decide ... = Bool.true` subgoals. -/

@[simp] theorem real_gt_eq_true (x y : TileCarrier .real) :
    (ComparableDType.real.gt x y = Bool.true) ↔ x > y := by
  simp [gt]

@[simp] theorem real_lt_eq_true (x y : TileCarrier .real) :
    (ComparableDType.real.lt x y = Bool.true) ↔ x < y := by
  simp [lt]

@[simp] theorem real_ge_eq_true (x y : TileCarrier .real) :
    (ComparableDType.real.ge x y = Bool.true) ↔ x ≥ y := by
  simp [ge]

@[simp] theorem real_le_eq_true (x y : TileCarrier .real) :
    (ComparableDType.real.le x y = Bool.true) ↔ x ≤ y := by
  simp [le]

@[simp] theorem real_eq_eq_true (x y : TileCarrier .real) :
    (ComparableDType.real.eq x y = Bool.true) ↔ x = y := by
  simp [eq]

@[simp] theorem real_ne_eq_true (x y : TileCarrier .real) :
    (ComparableDType.real.ne x y = Bool.true) ↔ x ≠ y := by
  simp [ne]

@[simp] theorem int_gt_eq_true (x y : Int) :
    (ComparableDType.int.gt x y = Bool.true) ↔ x > y := by
  simp [gt]

@[simp] theorem int_lt_eq_true (x y : Int) :
    (ComparableDType.int.lt x y = Bool.true) ↔ x < y := by
  simp [lt]

@[simp] theorem nat_gt_eq_true (x y : Nat) :
    (ComparableDType.nat.gt x y = Bool.true) ↔ x > y := by
  simp [gt]

@[simp] theorem nat_lt_eq_true (x y : Nat) :
    (ComparableDType.nat.lt x y = Bool.true) ↔ x < y := by
  simp [lt]

@[simp] theorem nat_le_eq_true (x y : Nat) :
    (ComparableDType.nat.le x y = Bool.true) ↔ x ≤ y := by
  simp [le]

@[simp] theorem nat_eq_eq_true (x y : Nat) :
    (ComparableDType.nat.eq x y = Bool.true) ↔ x = y := by
  simp [eq]

@[simp] theorem nat_ne_eq_true (x y : Nat) :
    (ComparableDType.nat.ne x y = Bool.true) ↔ x ≠ y := by
  simp [ne]

@[simp] theorem int_ge_eq_true (x y : Int) :
    (ComparableDType.int.ge x y = Bool.true) ↔ x ≥ y := by
  simp [ge]

@[simp] theorem int_le_eq_true (x y : Int) :
    (ComparableDType.int.le x y = Bool.true) ↔ x ≤ y := by
  simp [le]

@[simp] theorem int_eq_eq_true (x y : Int) :
    (ComparableDType.int.eq x y = Bool.true) ↔ x = y := by
  simp [eq]

@[simp] theorem int_ne_eq_true (x y : Int) :
    (ComparableDType.int.ne x y = Bool.true) ↔ x ≠ y := by
  simp [ne]

end ComparableDType

namespace FloatDType

def ofWithBot : (dtype : FloatDType) → WithBot ℝ → TileCarrier dtype.toTileDType
  | .real, x => x
  | .fp32, x => x
  | .fp16, x => x
  | .bf16, x => x

def toWithBot : (dtype : FloatDType) → TileCarrier dtype.toTileDType → WithBot ℝ
  | .real, x => x
  | .fp32, x => x
  | .fp16, x => x
  | .bf16, x => x

def ofReal (h : FloatDType) (x : ℝ) : TileCarrier h.toTileDType :=
  h.ofWithBot (some x)

def cast (src dst : FloatDType)
    (x : TileCarrier src.toTileDType) : TileCarrier dst.toTileDType :=
  dst.ofWithBot (src.toWithBot x)

def storeValue (h : FloatDType) (x : TileCarrier h.toTileDType) : ℝ :=
  (h.toWithBot x).unbotD 0

@[simp] theorem real_ofWithBot (x : WithBot ℝ) :
    FloatDType.real.ofWithBot x = x := rfl

@[simp] theorem real_toWithBot (x : TileCarrier FloatDType.real.toTileDType) :
    FloatDType.real.toWithBot x = x := rfl

@[simp] theorem real_ofReal (x : ℝ) :
    FloatDType.real.ofReal x = (some x : WithBot ℝ) := rfl

@[simp] theorem real_storeValue (x : TileCarrier FloatDType.real.toTileDType) :
    FloatDType.real.storeValue x = x.unbotD 0 := rfl

/-- `FloatDType.fp32.ofWithBot` is the identity in the current placeholder
implementation. Documents that fp32→bytes rounding is not yet modeled; once
rounding is added (#129/#137), this lemma will need an explicit rounding
adjustment. -/
@[simp] theorem fp32_ofWithBot (x : WithBot ℝ) :
    FloatDType.fp32.ofWithBot x = x := rfl

@[simp] theorem fp32_toWithBot (x : TileCarrier FloatDType.fp32.toTileDType) :
    FloatDType.fp32.toWithBot x = x := rfl

@[simp] theorem fp16_ofWithBot (x : WithBot ℝ) :
    FloatDType.fp16.ofWithBot x = x := rfl

@[simp] theorem fp16_toWithBot (x : TileCarrier FloatDType.fp16.toTileDType) :
    FloatDType.fp16.toWithBot x = x := rfl

@[simp] theorem bf16_ofWithBot (x : WithBot ℝ) :
    FloatDType.bf16.ofWithBot x = x := rfl

@[simp] theorem bf16_toWithBot (x : TileCarrier FloatDType.bf16.toTileDType) :
    FloatDType.bf16.toWithBot x = x := rfl

end FloatDType

/-- Cast a typed tile across definitional dtype/shape equalities. -/
def castTile {dtype shape dtype' shape'}
    (hd : dtype = dtype') (hs : shape = shape')
    (v : Tile dtype' shape') : Tile dtype shape := by
  subst dtype'
  subst shape'
  exact v

@[simp] theorem castTile_self {dtype shape}
    (hd : dtype = dtype) (hs : shape = shape) (v : Tile dtype shape) :
    castTile hd hs v = v := by
  cases hd
  cases hs
  rfl

end VeriTile.Triton
