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
  | .int32, x, y => x + y
  | .nat, x, y => x + y

def sub : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => WithBot.realSub x y
  | .fp32, x, y => WithBot.realSub x y
  | .fp16, x, y => WithBot.realSub x y
  | .bf16, x, y => WithBot.realSub x y
  | .int32, x, y => x - y
  | .nat, x, y => x - y

def mul : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => WithBot.realMul x y
  | .fp32, x, y => WithBot.realMul x y
  | .fp16, x, y => WithBot.realMul x y
  | .bf16, x, y => WithBot.realMul x y
  | .int32, x, y => x * y
  | .nat, x, y => x * y

noncomputable def div : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => WithBot.realDiv x y
  | .fp32, x, y => WithBot.realDiv x y
  | .fp16, x, y => WithBot.realDiv x y
  | .bf16, x, y => WithBot.realDiv x y
  | .int32, x, y => x / y
  | .nat, x, y => x / y

end NumericDType

namespace IntegralDType

def floorDiv : IntegralDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .int32, x, y => x / y
  | .nat, x, y => x / y

def mod : IntegralDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .int32, x, y => x % y
  | .nat, x, y => x % y

end IntegralDType

namespace ComparableDType

noncomputable def lt :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x < y)
  | .fp32, x, y => decide (x < y)
  | .fp16, x, y => decide (x < y)
  | .bf16, x, y => decide (x < y)
  | .int32, x, y => decide (x < y)
  | .nat, x, y => decide (x < y)

noncomputable def le :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x ≤ y)
  | .fp32, x, y => decide (x ≤ y)
  | .fp16, x, y => decide (x ≤ y)
  | .bf16, x, y => decide (x ≤ y)
  | .int32, x, y => decide (x ≤ y)
  | .nat, x, y => decide (x ≤ y)

noncomputable def eq :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x = y)
  | .fp32, x, y => decide (x = y)
  | .fp16, x, y => decide (x = y)
  | .bf16, x, y => decide (x = y)
  | .int32, x, y => decide (x = y)
  | .nat, x, y => decide (x = y)

noncomputable def gt :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x > y)
  | .fp32, x, y => decide (x > y)
  | .fp16, x, y => decide (x > y)
  | .bf16, x, y => decide (x > y)
  | .int32, x, y => decide (x > y)
  | .nat, x, y => decide (x > y)

noncomputable def ge :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x ≥ y)
  | .fp32, x, y => decide (x ≥ y)
  | .fp16, x, y => decide (x ≥ y)
  | .bf16, x, y => decide (x ≥ y)
  | .int32, x, y => decide (x ≥ y)
  | .nat, x, y => decide (x ≥ y)

noncomputable def ne :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x ≠ y)
  | .fp32, x, y => decide (x ≠ y)
  | .fp16, x, y => decide (x ≠ y)
  | .bf16, x, y => decide (x ≠ y)
  | .int32, x, y => decide (x ≠ y)
  | .nat, x, y => decide (x ≠ y)

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
