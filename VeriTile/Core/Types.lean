/-
VeriTile.Core.Types

Basic names, dtype tags, carriers, and dtype witnesses for the Triton core AST.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Order.WithBot

namespace VeriTile

/-- Symbolic name for a register (scalar or tile variable). -/
abbrev RegName := String

/-- Padding behavior for out-of-bounds block-pointer loads. -/
inductive PaddingOption where
  | zero
  deriving DecidableEq, Repr

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
  | int
  | nat
  | bool
  | ptr
  | blockPtr
  deriving DecidableEq, Repr

/-- Symbolic name for a memory region (input/output buffer), indexed by the
buffer's element dtype.

The phantom `d` lets the DSL recover `tl.load` / `tl.store` return dtypes
from the region argument instead of needing a separate macro-side `dtype=`
kwarg. Backwards compatibility: `RegionName := Region .real` keeps every
real-typed kernel signature unchanged. Non-real kernels (e.g.
`triton_argmax`'s `mid_index : torch.int64`) declare `Region .nat` directly.
-/
structure Region (d : TileDType) where
  name : String
  deriving DecidableEq, Repr, Hashable

/-- String literals at Region call sites elaborate as `Region .real` by
default, matching the legacy `RegionName := String` behaviour for kernels
that don't opt into typed regions. Non-real callers construct the region
explicitly: `(⟨"R"⟩ : Region .nat)`. -/
instance : Coe String (Region .real) where
  coe s := ⟨s⟩

namespace Region

/-- Cast a region's phantom dtype tag without changing its underlying name.
Useful when bridging between layers that drop or restore element-dtype
tracking (e.g., dynamic pointer arithmetic that erases the source region's
dtype). -/
@[inline] def cast {d d' : TileDType} (r : Region d) : Region d' :=
  ⟨r.name⟩

@[simp] theorem cast_name {d d' : TileDType} (r : Region d) :
    (Region.cast (d := d) (d' := d') r).name = r.name := rfl

@[simp] theorem cast_id {d : TileDType} (r : Region d) :
    Region.cast (d := d) (d' := d) r = r := rfl

@[simp] theorem cast_cast {d d' d'' : TileDType} (r : Region d) :
    Region.cast (d := d') (d' := d'') (Region.cast (d := d) (d' := d') r) =
      Region.cast (d := d) (d' := d'') r := rfl

end Region

/-- Legacy alias. Equals `Region .real` so every existing
`(X : RegionName)` parameter remains a real-typed region. New non-real
kernels write `(X : Region .nat)` etc. directly. -/
abbrev RegionName := Region .real

instance {d : TileDType} : Coe RegionName (Region d) where
  coe r := Region.cast r

instance {d : TileDType} : CoeOut (Region d) RegionName where
  coe r := Region.cast r

/--
Runtime value carried by Triton block pointers.

The model is intentionally layout-explicit rather than hardware-specific:
`parentShape`, `blockShape`, `strides`, and `offsets` are outermost-first lists.
For a block-local index `idx`, the parent tensor coordinate is
`offsets + idx`; the flat memory address is
`baseOffset + Σ axis, (offsets[axis] + idx[axis]) * strides[axis]`.

`region` carries `Region .real` for now; typed block pointers (real / nat /
int element channels) are deferred to a later migration.
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

@[simp] theorem address_1d
    (region : RegionName) (base len BT stride off : Nat) (i : Nat) :
    BlockPtr.address
      { region := region, baseOffset := base, parentShape := [len],
        blockShape := [BT], strides := [stride], offsets := [off] }
      [i] =
        base + (off + i) * stride := by
  simp [BlockPtr.address, nthD, List.range, List.range.loop]

@[simp] theorem inBounds_1d
    (region : RegionName) (base len BT stride off : Nat) (i : Nat) :
    BlockPtr.inBounds
      { region := region, baseOffset := base, parentShape := [len],
        blockShape := [BT], strides := [stride], offsets := [off] }
      [i] [0] =
        decide (off + i < len) := by
  simp [BlockPtr.inBounds, checkedInBounds, nthD]

@[simp] theorem address_2d_zero_offsets
    (region : RegionName) (base rows cols BT BS strideT strideS : Nat)
    (i j : Nat) :
    BlockPtr.address
      { region := region, baseOffset := base, parentShape := [rows, cols],
        blockShape := [BT, BS], strides := [strideT, strideS], offsets := [0, 0] }
      [i, j] =
        base + i * strideT + j * strideS := by
  simp [BlockPtr.address, nthD, List.range, List.range.loop]
  omega

@[simp] theorem inBounds_2d_zero_offsets
    (region : RegionName) (base rows cols BT BS strideT strideS : Nat)
    (i j : Nat) :
    BlockPtr.inBounds
      { region := region, baseOffset := base, parentShape := [rows, cols],
        blockShape := [BT, BS], strides := [strideT, strideS], offsets := [0, 0] }
      [i, j] [0, 1] =
        decide (i < rows ∧ j < cols) := by
  simp [BlockPtr.inBounds, checkedInBounds, nthD]

@[simp] theorem address_2d_zero_row_offset
    (region : RegionName) (base rows cols BT BS strideT strideS colOff : Nat)
    (i j : Nat) :
    BlockPtr.address
      { region := region, baseOffset := base, parentShape := [rows, cols],
        blockShape := [BT, BS], strides := [strideT, strideS],
        offsets := [0, colOff] }
      [i, j] =
        base + i * strideT + (colOff + j) * strideS := by
  simp [BlockPtr.address, nthD, List.range, List.range.loop]
  omega

@[simp] theorem inBounds_2d_zero_row_offset
    (region : RegionName) (base rows cols BT BS strideT strideS colOff : Nat)
    (i j : Nat) :
    BlockPtr.inBounds
      { region := region, baseOffset := base, parentShape := [rows, cols],
        blockShape := [BT, BS], strides := [strideT, strideS],
        offsets := [0, colOff] }
      [i, j] [0, 1] =
        decide (i < rows ∧ colOff + j < cols) := by
  simp [BlockPtr.inBounds, checkedInBounds, nthD]

/-- General 2D address with arbitrary per-axis offsets `[rowOff, colOff]`:
each axis contributes `(offset + index) · stride`. -/
@[simp] theorem address_2d_offsets
    (region : RegionName) (base rows cols BT BS strideT strideS rowOff colOff : Nat)
    (i j : Nat) :
    BlockPtr.address
      { region := region, baseOffset := base, parentShape := [rows, cols],
        blockShape := [BT, BS], strides := [strideT, strideS],
        offsets := [rowOff, colOff] }
      [i, j] =
        base + (rowOff + i) * strideT + (colOff + j) * strideS := by
  simp [BlockPtr.address, nthD, List.range, List.range.loop]
  omega

/-- General 2D `inBounds` with arbitrary offsets `[rowOff, colOff]`. -/
@[simp] theorem inBounds_2d_offsets
    (region : RegionName) (base rows cols BT BS strideT strideS rowOff colOff : Nat)
    (i j : Nat) :
    BlockPtr.inBounds
      { region := region, baseOffset := base, parentShape := [rows, cols],
        blockShape := [BT, BS], strides := [strideT, strideS],
        offsets := [rowOff, colOff] }
      [i, j] [0, 1] =
        decide (rowOff + i < rows ∧ colOff + j < cols) := by
  simp [BlockPtr.inBounds, checkedInBounds, nthD]

/-- 2D address with a row offset only (`[rowOff, 0]`). -/
@[simp] theorem address_2d_row_offset
    (region : RegionName) (base rows cols BT BS strideT strideS rowOff : Nat)
    (i j : Nat) :
    BlockPtr.address
      { region := region, baseOffset := base, parentShape := [rows, cols],
        blockShape := [BT, BS], strides := [strideT, strideS],
        offsets := [rowOff, 0] }
      [i, j] =
        base + (rowOff + i) * strideT + j * strideS := by
  simp [BlockPtr.address, nthD, List.range, List.range.loop]
  omega

/-- 2D `inBounds` with a row offset only (`[rowOff, 0]`). -/
@[simp] theorem inBounds_2d_row_offset
    (region : RegionName) (base rows cols BT BS strideT strideS rowOff : Nat)
    (i j : Nat) :
    BlockPtr.inBounds
      { region := region, baseOffset := base, parentShape := [rows, cols],
        blockShape := [BT, BS], strides := [strideT, strideS],
        offsets := [rowOff, 0] }
      [i, j] [0, 1] =
        decide (rowOff + i < rows ∧ j < cols) := by
  simp [BlockPtr.inBounds, checkedInBounds, nthD]

/-- An empty boundary-check list checks no axes, so `inBounds` is unconditionally
`true` (the all-`[]` quantifier). This is the block-ptr load's "no boundary
check" case (`tl.load` without `boundary_check`). -/
@[simp] theorem inBounds_nil_checkedAxes (ptr : BlockPtr) (idx : List Nat) :
    BlockPtr.inBounds ptr idx [] = true := by
  simp [BlockPtr.inBounds]

private def nthDInt (xs : List Int) (i : Nat) : Int :=
  xs.getD i 0

/-- Advance a block pointer by per-axis **signed** offset deltas (Triton's
`tl.advance`). Pointer offsets are non-negative addresses, so each advanced axis
is `((offset : Int) + delta).toNat`; for the common non-negative delta this is
just `offset + delta`, and signed deltas model a backward pointer rewind (used by
e.g. multi-block backward attention). -/
def advance (ptr : BlockPtr) (deltas : List Int) : BlockPtr :=
  let offsets :=
    (List.range (max ptr.offsets.length deltas.length)).map
      (fun axis => ((nthD ptr.offsets axis : Int) + nthDInt deltas axis).toNat)
  { ptr with offsets := offsets }

/-- `advance` on a well-formed 2D block-pointer by **non-negative** (`Nat`-coerced)
per-axis deltas adds them to the per-axis offsets (`[rowOff, colOff]` advanced by
`[↑dRow, ↑dCol]`). The `region`, `baseOffset`, `parentShape`, `blockShape`,
`strides` are unchanged. This is the normal form for every `tl.advance` whose
deltas are non-negative (the vast majority); it preserves the pre-`Int` behavior. -/
@[simp] theorem advance_2d_offsets
    (region : RegionName) (base rows cols BT BS strideT strideS rowOff colOff
      dRow dCol : Nat) :
    BlockPtr.advance
      { region := region, baseOffset := base, parentShape := [rows, cols],
        blockShape := [BT, BS], strides := [strideT, strideS],
        offsets := [rowOff, colOff] }
      [(dRow : Int), (dCol : Int)] =
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [rowOff + dRow, colOff + dCol] } := by
  simp only [BlockPtr.advance, nthD, nthDInt, List.range, List.range.loop,
    List.map_cons, List.map_nil, List.getD_cons_zero, List.getD_cons_succ,
    List.getD_nil, List.length_cons, List.length_nil, BlockPtr.mk.injEq,
    List.cons.injEq, and_true, true_and, Nat.max_self]
  omega

/-- `Int.toNat` of a sum of two `Nat`-casts collapses to `Nat` addition. Terminal
`@[simp]` normalizer so advanced block-ptr offsets (which go through `Int` + `toNat`)
reduce back to plain `Nat` arithmetic regardless of the delta's coercion form. -/
@[simp] theorem toNat_natCast_add_natCast (a b : Nat) :
    (((a : Int)) + (b : Int)).toNat = a + b := by omega

@[simp] theorem toNat_zero_add_natCast (a : Nat) :
    (((0 : Int)) + (a : Int)).toNat = a := by omega

@[simp] theorem toNat_natCast_add_zero (a : Nat) :
    (((a : Int)) + (0 : Int)).toNat = a := by omega

/-- General signed-delta form of `advance` on a 2D block pointer: each axis is
`((off : Int) + d).toNat`. `@[simp]` so it fires for any delta form; the `toNat`
normalizers above then collapse the non-negative cases to `Nat` addition. -/
@[simp] theorem advance_2d_offsets_int
    (region : RegionName) (base rows cols BT BS strideT strideS rowOff colOff : Nat)
    (dRow dCol : Int) :
    BlockPtr.advance
      { region := region, baseOffset := base, parentShape := [rows, cols],
        blockShape := [BT, BS], strides := [strideT, strideS],
        offsets := [rowOff, colOff] }
      [dRow, dCol] =
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [((rowOff : Int) + dRow).toNat, ((colOff : Int) + dCol).toNat] } := by
  simp only [BlockPtr.advance, nthD, nthDInt, List.range, List.range.loop,
    List.map_cons, List.map_nil, List.getD_cons_zero, List.getD_cons_succ,
    List.getD_nil, List.length_cons, List.length_nil, Nat.max_self]

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

/-- Lean carrier for each VeriTile tile dtype.

`.real` is `WithBot ℝ` rather than `ℝ` so we can faithfully model `tl.full((),
-float("inf"))` as the bottom element. Triton attention kernels rely on `-inf`
as the seed of the running-max accumulator; using a finite stand-in like
`-1e38` would either be wrong (when input data goes below `-1e38`) or require
range preconditions on every theorem. With `WithBot ℝ`:

* `Op.negInf` evaluates to `⊥ : WithBot ℝ` — true `-∞`
* `max ⊥ x = x` for all `x` (Mathlib LinearOrder), so the seed is consumed by
  the first iteration and `⊥` does not escape into normal arithmetic
* arithmetic (`+`, `-`, `*`, `/`) propagates `⊥` (Mathlib WithBot instances)
* `exp ⊥ := 0` and `sigmoid ⊥ := 0` mirror IEEE behavior of `exp(-∞)`/
  `sigmoid(-∞)`; see `WithBot.realExp` / `WithBot.realSigmoid` in Semantics

`tl.load` reads typed `MemCell`s through the operational `readMemValue` view.
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
  | .int => Int
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
  | int : NumericDType .int
  | nat  : NumericDType .nat

/-- Integer-like dtypes that support floor division and remainder. -/
inductive IntegralDType : TileDType → Type where
  | int : IntegralDType .int
  | nat  : IntegralDType .nat

/-- DTypes that support Triton's comparison operators in the current model. -/
inductive ComparableDType : TileDType → Type where
  | real : ComparableDType .real
  | fp32 : ComparableDType .fp32
  | fp16 : ComparableDType .fp16
  | bf16 : ComparableDType .bf16
  | int : ComparableDType .int
  | nat  : ComparableDType .nat

namespace FloatDType

def numericDType : (f : FloatDType) → NumericDType f.toTileDType
  | .real => NumericDType.real
  | .fp32 => NumericDType.fp32
  | .fp16 => NumericDType.fp16
  | .bf16 => NumericDType.bf16

def comparableDType : (f : FloatDType) → ComparableDType f.toTileDType
  | .real => ComparableDType.real
  | .fp32 => ComparableDType.fp32
  | .fp16 => ComparableDType.fp16
  | .bf16 => ComparableDType.bf16

end FloatDType

end VeriTile
