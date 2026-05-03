/-
VeriTile.Triton.Core.Types

Basic names, dtype tags, carriers, and dtype witnesses for the Triton core AST.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Order.WithBot

namespace VeriTile.Triton

/-- Symbolic name for a memory region (input/output buffer). -/
abbrev RegionName := String

/-- Symbolic name for a register (scalar or tile variable). -/
abbrev RegName := String

/-- Padding behavior for out-of-bounds block-pointer loads. -/
inductive PaddingOption where
  | zero
  deriving DecidableEq, Repr

/--
Runtime value carried by Triton block pointers.

The model is intentionally layout-explicit rather than hardware-specific:
`parentShape`, `blockShape`, `strides`, and `offsets` are outermost-first lists.
For a block-local index `idx`, the parent tensor coordinate is
`offsets + idx`; the flat memory address is
`baseOffset + Σ axis, (offsets[axis] + idx[axis]) * strides[axis]`.
-/
structure BlockPtr where
  region : RegionName
  baseOffset : Nat
  parentShape : List Nat
  blockShape : List Nat
  strides : List Nat
  offsets : List Nat
  deriving DecidableEq, Repr

namespace BlockPtr

private def nthD (xs : List Nat) (i : Nat) : Nat :=
  xs.getD i 0

private def checkedInBounds (parentShape offsets idx : List Nat) (axis : Nat) : Bool :=
  nthD offsets axis + nthD idx axis < nthD parentShape axis

@[simp] private theorem checkedInBounds_empty_parent_axis0 (idx : List Nat) :
    checkedInBounds [0] [0] idx 0 = false := by
  simp [checkedInBounds, nthD]

def inBounds (ptr : BlockPtr) (idx checkedAxes : List Nat) : Bool :=
  checkedAxes.all (checkedInBounds ptr.parentShape ptr.offsets idx)

def address (ptr : BlockPtr) (idx : List Nat) : Nat :=
  let axisOffsets :=
    (List.range ptr.strides.length).map fun axis =>
      (nthD ptr.offsets axis + nthD idx axis) * nthD ptr.strides axis
  ptr.baseOffset + axisOffsets.sum

def advance (ptr : BlockPtr) (deltas : List Nat) : BlockPtr :=
  let offsets :=
    (List.range (max ptr.offsets.length deltas.length)).map
      (fun axis => nthD ptr.offsets axis + nthD deltas axis)
  { ptr with offsets := offsets }

@[simp] theorem inBounds_empty_parent_axis0 (ptr : BlockPtr) (idx : List Nat) :
    BlockPtr.inBounds { ptr with parentShape := [0], offsets := [0] } idx [0] = false := by
  simp [BlockPtr.inBounds, checkedInBounds, nthD]

@[simp] theorem inBounds_mk_empty_parent_axis0
    (region : RegionName) (baseOffset : Nat) (blockShape strides idx : List Nat) :
    BlockPtr.inBounds
      { region := region, baseOffset := baseOffset, parentShape := [0],
        blockShape := blockShape, strides := strides, offsets := [0] }
      idx [0] = false := by
  simp [BlockPtr.inBounds, checkedInBounds, nthD]

end BlockPtr

/--
Triton block-local value type.

VeriTile uses "tile" rather than "tensor" for Triton program values: a scalar
is a rank-0 tile, `tl.arange` produces a 1D tile, and future block-pointer /
FlashAttention work will introduce 2D tiles. This avoids conflating Triton
block-local values with framework-level tensors.
-/
inductive TileDType where
  | real
  | fp32
  | fp16
  | bf16
  | int32
  | nat
  | bool
  | ptr
  | blockPtr
  deriving DecidableEq, Repr

/-- Lean carrier for each VeriTile tile dtype.

`.real` is `WithBot ℝ` rather than `ℝ` so we can faithfully model `tl.full((),
-float('inf'))` as the bottom element. Triton attention kernels rely on `-inf`
as the seed of the running-max accumulator; using a finite stand-in like
`-1e38` would either be wrong (when input data goes below `-1e38`) or require
range preconditions on every theorem. With `WithBot ℝ`:

* `Op.negInf` evaluates to `⊥ : WithBot ℝ` — true `-∞`
* `max ⊥ x = x` for all `x` (Mathlib LinearOrder), so the seed is consumed by
  the first iteration and `⊥` does not escape into normal arithmetic
* arithmetic (`+`, `-`, `*`, `/`) propagates `⊥` (Mathlib WithBot instances)
* `exp ⊥ := 0` and `sigmoid ⊥ := 0` mirror IEEE behavior of `exp(-∞)`/
  `sigmoid(-∞)`; see `WithBot.realExp` / `WithBot.realSigmoid` in Semantics

`tl.load` reads typed `MemCell`s through the operational `readMemAs` view.
`tl.store` writes typed cells; floating stores demote via `unbot' 0` —
well-formed kernels never store `⊥`.

Hardware-shaped floating channels (`.fp32`, `.fp16`, `.bf16`) currently share
the same `WithBot ℝ` carrier as `.real`. They are a type-layer hook: kernels
can express dtype choices and casts, while this branch still proves over the
existing mathematical real model rather than IEEE rounding semantics. -/
abbrev TileCarrier : TileDType → Type
  | .real => WithBot ℝ
  | .fp32 => WithBot ℝ
  | .fp16 => WithBot ℝ
  | .bf16 => WithBot ℝ
  | .int32 => Int
  | .nat  => Nat
  | .bool => Bool
  | .ptr  => RegionName × Nat
  | .blockPtr => BlockPtr

/-- Floating data channels currently backed by the mathematical real model. -/
inductive FloatDType where
  | real
  | fp32
  | fp16
  | bf16
  deriving DecidableEq, Repr

namespace FloatDType

/-- The concrete AST dtype represented by a floating dtype tag. -/
def toTileDType : FloatDType → TileDType
  | .real => .real
  | .fp32 => .fp32
  | .fp16 => .fp16
  | .bf16 => .bf16

@[simp] theorem toTileDType_real : FloatDType.real.toTileDType = .real := rfl
@[simp] theorem toTileDType_fp32 : FloatDType.fp32.toTileDType = .fp32 := rfl
@[simp] theorem toTileDType_fp16 : FloatDType.fp16.toTileDType = .fp16 := rfl
@[simp] theorem toTileDType_bf16 : FloatDType.bf16.toTileDType = .bf16 := rfl

end FloatDType


/-- DTypes that support Triton's arithmetic operators in the current model. -/
inductive NumericDType : TileDType → Type where
  | real : NumericDType .real
  | fp32 : NumericDType .fp32
  | fp16 : NumericDType .fp16
  | bf16 : NumericDType .bf16
  | int32 : NumericDType .int32
  | nat  : NumericDType .nat

/-- Integer-like dtypes that support floor division and remainder. -/
inductive IntegralDType : TileDType → Type where
  | int32 : IntegralDType .int32
  | nat  : IntegralDType .nat

/-- DTypes that support Triton's comparison operators in the current model. -/
inductive ComparableDType : TileDType → Type where
  | real : ComparableDType .real
  | fp32 : ComparableDType .fp32
  | fp16 : ComparableDType .fp16
  | bf16 : ComparableDType .bf16
  | int32 : ComparableDType .int32
  | nat  : ComparableDType .nat

end VeriTile.Triton
