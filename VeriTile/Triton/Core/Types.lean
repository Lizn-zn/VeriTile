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

`tl.load` lifts `mem : RegionName → Nat → ℝ` to `some _ : WithBot ℝ`.
`tl.store` demotes via `unbot' 0` — well-formed kernels never store `⊥`.

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

/-- Floating data channels currently backed by the mathematical real model. -/
inductive FloatDType : TileDType → Type where
  | real : FloatDType .real
  | fp32 : FloatDType .fp32
  | fp16 : FloatDType .fp16
  | bf16 : FloatDType .bf16


/-- DTypes that support Triton's arithmetic operators in the current model. -/
inductive NumericDType : TileDType → Type where
  | real : NumericDType .real
  | fp32 : NumericDType .fp32
  | fp16 : NumericDType .fp16
  | bf16 : NumericDType .bf16
  | int32 : NumericDType .int32
  | nat  : NumericDType .nat

/-- DTypes that support Triton's comparison operators in the current model. -/
inductive ComparableDType : TileDType → Type where
  | real : ComparableDType .real
  | fp32 : ComparableDType .fp32
  | fp16 : ComparableDType .fp16
  | bf16 : ComparableDType .bf16
  | int32 : ComparableDType .int32
  | nat  : ComparableDType .nat

end VeriTile.Triton
