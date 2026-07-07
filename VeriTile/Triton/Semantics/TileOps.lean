/-
VeriTile.Triton.Semantics.TileOps

Tile-level semantic operators used by expression evaluation.
-/

import Mathlib.Data.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import VeriTile.Triton.Math.Erf
import VeriTile.Triton.Semantics.State

namespace VeriTile.Triton

def Tile.bop {dtype a b out}
    (op : TileCarrier dtype → TileCarrier dtype → TileCarrier dtype)
    (bc : Broadcast a b out) (x : Tile dtype a) (y : Tile dtype b) :
    Tile dtype out :=
  ⟨fun i => op (x.data (bc.leftIndex i)) (y.data (bc.rightIndex i))⟩

@[simp] theorem Tile.bop_data {dtype a b out}
    (op : TileCarrier dtype → TileCarrier dtype → TileCarrier dtype)
    (bc : Broadcast a b out) (x : Tile dtype a) (y : Tile dtype b)
    (i : TileIndex out) :
    (Tile.bop op bc x y).data i =
      op (x.data (bc.leftIndex i)) (y.data (bc.rightIndex i)) := rfl

def Tile.cop {dtype a b out}
    (op : TileCarrier dtype → TileCarrier dtype → Bool)
    (bc : Broadcast a b out) (x : Tile dtype a) (y : Tile dtype b) :
    Tile .bool out :=
  ⟨fun i => op (x.data (bc.leftIndex i)) (y.data (bc.rightIndex i))⟩

@[simp] theorem Tile.cop_data {dtype a b out}
    (op : TileCarrier dtype → TileCarrier dtype → Bool)
    (bc : Broadcast a b out) (x : Tile dtype a) (y : Tile dtype b)
    (i : TileIndex out) :
    (Tile.cop op bc x y).data i =
      op (x.data (bc.leftIndex i)) (y.data (bc.rightIndex i)) := rfl

def Tile.ptrAdd {a b out}
    (bc : Broadcast a b out) (ptrs : Tile .ptr a) (offs : Tile .nat b) :
    Tile .ptr out :=
  ⟨fun i =>
    let p := ptrs.data (bc.leftIndex i)
    let o := offs.data (bc.rightIndex i)
    (p.1, p.2 + o)⟩

@[simp] theorem Tile.ptrAdd_data {a b out}
    (bc : Broadcast a b out) (ptrs : Tile .ptr a) (offs : Tile .nat b)
    (i : TileIndex out) :
    (Tile.ptrAdd bc ptrs offs).data i =
      ((ptrs.data (bc.leftIndex i)).1,
       (ptrs.data (bc.leftIndex i)).2 + offs.data (bc.rightIndex i)) := rfl

def Tile.ptrSub {a b out}
    (bc : Broadcast a b out) (ptrs : Tile .ptr a) (offs : Tile .nat b) :
    Tile .ptr out :=
  ⟨fun i =>
    let p := ptrs.data (bc.leftIndex i)
    let o := offs.data (bc.rightIndex i)
    (p.1, p.2 - o)⟩

@[simp] theorem Tile.ptrSub_data {a b out}
    (bc : Broadcast a b out) (ptrs : Tile .ptr a) (offs : Tile .nat b)
    (i : TileIndex out) :
    (Tile.ptrSub bc ptrs offs).data i =
      ((ptrs.data (bc.leftIndex i)).1,
       (ptrs.data (bc.leftIndex i)).2 - offs.data (bc.rightIndex i)) := rfl

def Tile.uop {dtype shape} (op : TileCarrier dtype → TileCarrier dtype)
    (x : Tile dtype shape) : Tile dtype shape :=
  ⟨fun i => op (x.data i)⟩

@[simp] theorem Tile.uop_data {dtype shape}
    (op : TileCarrier dtype → TileCarrier dtype)
    (x : Tile dtype shape) (i : TileIndex shape) :
    (Tile.uop op x).data i = op (x.data i) := rfl

def Tile.remap {dtype inShape outShape}
    (map : TileIndex outShape → TileIndex inShape)
    (x : Tile dtype inShape) : Tile dtype outShape :=
  ⟨fun i => x.data (map i)⟩

/-- Generic masked row load carrier: valid lanes read `load`, inactive lanes
produce `other`. This is Triton carrier plumbing, not an operator-specific
math bridge. -/
def Tile.maskedRowLoad {α : Type*} {n : Nat}
    (load : Fin n → α)
    (active : Fin n → Prop) [DecidablePred active]
    (other : α) (i : Fin n) : α :=
  if active i then load i else other

/-- Generic one-dimensional masked tile with `other` for inactive lanes. -/
def Tile.maskedRowTile {dtype : TileDType} {n : Nat}
    (load : Fin n → TileCarrier dtype)
    (active : Fin n → Prop) [DecidablePred active]
    (other : TileCarrier dtype) : Tile dtype [n] :=
  ⟨fun i => Tile.maskedRowLoad load active other i.1⟩

@[simp] theorem Tile.maskedRowLoad_active {α : Type*} {n : Nat}
    (load : Fin n → α)
    (active : Fin n → Prop) [DecidablePred active]
    (other : α) {i : Fin n} (h : active i) :
    Tile.maskedRowLoad load active other i = load i := by
  simp [Tile.maskedRowLoad, h]

@[simp] theorem Tile.maskedRowLoad_inactive {α : Type*} {n : Nat}
    (load : Fin n → α)
    (active : Fin n → Prop) [DecidablePred active]
    (other : α) {i : Fin n} (h : ¬ active i) :
    Tile.maskedRowLoad load active other i = other := by
  simp [Tile.maskedRowLoad, h]

@[simp] theorem Tile.maskedRowTile_data {dtype : TileDType} {n : Nat}
    (load : Fin n → TileCarrier dtype)
    (active : Fin n → Prop) [DecidablePred active]
    (other : TileCarrier dtype) (i : TileIndex [n]) :
    (Tile.maskedRowTile load active other).data i =
      Tile.maskedRowLoad load active other i.1 := rfl

def Tile.reshape {dtype inShape outShape}
    (x : Tile dtype inShape) : Tile dtype outShape :=
  let inputs := TileShape.allIndices inShape
  ⟨fun i =>
    match inputs[TileShape.linearIndex outShape i]? with
    | some j => x.data j
    | none => BlockState.defaultCarrier dtype⟩

def Tile.join {dtype shape}
    (a b : Tile dtype shape) : Tile dtype (shape ++ [2]) :=
  ⟨fun i =>
    let parts := TileShape.splitLastIndex shape i
    if parts.2.val = 0 then a.data parts.1 else b.data parts.1⟩

def Tile.split {dtype shape}
    (side : Fin 2) (x : Tile dtype (shape ++ [2])) : Tile dtype shape :=
  ⟨fun i => x.data (TileShape.appendAxisIndex shape i side)⟩

/-! ### Lifted unary functions on `WithBot ℝ`

`exp(-∞) = 0` and `sigmoid(-∞) = 0` are the IEEE-faithful values; this is what
makes the OnlineSoftmax `m`-seed flow correctly: with `m_0 = ⊥`, the first
iteration's `exp(m_0 - m_1) * l_0 = exp(⊥) * 0 = 0 * 0 = 0`, recovering the
correct base-case `l_1 = exp(x_0 - x_0) = 1`. -/

noncomputable def WithBot.realExp : WithBot ℝ → WithBot ℝ
  | none   => some 0           -- exp(-∞) = 0
  | some r => some (Real.exp r)

noncomputable def WithBot.realLog : WithBot ℝ → WithBot ℝ
  | none   => none
  | some r => some (Real.log r)

noncomputable def WithBot.realLog2 : WithBot ℝ → WithBot ℝ
  | none   => none
  | some r => some (Real.log r / Real.log 2)

noncomputable def WithBot.realExp2 : WithBot ℝ → WithBot ℝ
  | none   => some 0
  | some r => some (Real.exp (r * Real.log 2))

noncomputable def WithBot.realSigmoid : WithBot ℝ → WithBot ℝ
  | none   => some 0           -- sigmoid(-∞) = 0
  | some r => some (Real.sigmoid r)

noncomputable def WithBot.realSqrt : WithBot ℝ → WithBot ℝ
  | none   => none             -- sqrt(-∞) undefined
  | some r => some (Real.sqrt r)

noncomputable def WithBot.realRsqrt : WithBot ℝ → WithBot ℝ
  | none   => none             -- rsqrt(-∞) undefined
  | some r => some (1 / Real.sqrt r)

noncomputable def WithBot.realTanh : WithBot ℝ → WithBot ℝ
  | none   => some (-1)        -- tanh(-∞) = -1
  | some r => some (Real.tanh r)

noncomputable def WithBot.realSin : WithBot ℝ → WithBot ℝ
  | none   => none
  | some r => some (Real.sin r)

noncomputable def WithBot.realCos : WithBot ℝ → WithBot ℝ
  | none   => none
  | some r => some (Real.cos r)

noncomputable def WithBot.realTan : WithBot ℝ → WithBot ℝ
  | none   => none
  | some r => some (Real.tan r)

noncomputable def WithBot.realAtan : WithBot ℝ → WithBot ℝ
  | none   => some (-(Real.pi / 2))
  | some r => some (Real.arctan r)

noncomputable def WithBot.realCosh : WithBot ℝ → WithBot ℝ
  | none   => none
  | some r => some (Real.cosh r)

noncomputable def WithBot.realSinh : WithBot ℝ → WithBot ℝ
  | none   => none
  | some r => some (Real.sinh r)

noncomputable def WithBot.realErf : WithBot ℝ → WithBot ℝ
  | none   => some (-1)        -- erf(-∞) = -1
  | some r => some (VeriTile.Math.realErf r)

/-- `⊥`-propagating element-wise power on `WithBot ℝ` (`Op.pow` carrier
semantics): `Real.rpow` on two finite lanes, `⊥` if either lane is `⊥`,
matching the uniform `Option.map₂` propagation of `WithBot.realSub` /
`WithBot.realMul` / `WithBot.realDiv`. -/
noncomputable def WithBot.realPow : WithBot ℝ → WithBot ℝ → WithBot ℝ :=
  Option.map₂ Real.rpow

noncomputable def ScanOp.eval : ScanOp → List (WithBot ℝ) → WithBot ℝ
  | .sum, xs => xs.foldl WithBot.realAdd ((0 : ℝ) : WithBot ℝ)
  | .prod, xs => xs.foldl WithBot.realMul ((1 : ℝ) : WithBot ℝ)
  | .max, xs => xs.foldl (fun acc x => if acc ≤ x then x else acc) (⊥ : WithBot ℝ)
  | .min, [] => ⊥
  | .min, x :: xs => xs.foldl (fun acc x => if acc ≤ x then acc else x) x

/-- `ScanOp.sum.eval` unfolds to the standard `foldl + 0` pattern over
`WithBot.realAdd`. Useful for cumsum kernel proofs. -/
@[simp] theorem ScanOp.eval_sum (xs : List (WithBot ℝ)) :
    ScanOp.eval .sum xs = xs.foldl WithBot.realAdd ((0 : ℝ) : WithBot ℝ) := rfl

/-- `ScanOp.prod.eval` unfolds to the standard `foldl * 1` pattern over
`WithBot.realMul`. -/
@[simp] theorem ScanOp.eval_prod (xs : List (WithBot ℝ)) :
    ScanOp.eval .prod xs = xs.foldl WithBot.realMul ((1 : ℝ) : WithBot ℝ) := rfl

/-- `ScanOp.max.eval` unfolds to `foldl` with the strict-less-than
comparator over `WithBot ℝ`. -/
@[simp] theorem ScanOp.eval_max (xs : List (WithBot ℝ)) :
    ScanOp.eval .max xs =
      xs.foldl (fun acc x => if acc ≤ x then x else acc) (⊥ : WithBot ℝ) := rfl

/-- `ScanOp.min.eval` on the empty list yields `⊥`. -/
@[simp] theorem ScanOp.eval_min_nil :
    ScanOp.eval .min ([] : List (WithBot ℝ)) = (⊥ : WithBot ℝ) := rfl

/-- `ScanOp.min.eval` on `x :: xs` folds with the strict-greater-than
comparator starting from `x`. -/
@[simp] theorem ScanOp.eval_min_cons (x : WithBot ℝ) (xs : List (WithBot ℝ)) :
    ScanOp.eval .min (x :: xs) =
      xs.foldl (fun acc x => if acc ≤ x then acc else x) x := rfl

@[simp] theorem WithBot.realExp_some (r : ℝ) :
    WithBot.realExp (some r) = some (Real.exp r) := rfl
@[simp] theorem WithBot.realLog_some (r : ℝ) :
    WithBot.realLog (some r) = some (Real.log r) := rfl
@[simp] theorem WithBot.realLog2_some (r : ℝ) :
    WithBot.realLog2 (some r) = some (Real.log r / Real.log 2) := rfl
@[simp] theorem WithBot.realExp2_some (r : ℝ) :
    WithBot.realExp2 (some r) = some (Real.exp (r * Real.log 2)) := rfl
@[simp] theorem WithBot.realSigmoid_some (r : ℝ) :
    WithBot.realSigmoid (some r) = some (Real.sigmoid r) := rfl
@[simp] theorem WithBot.realSqrt_some (r : ℝ) :
    WithBot.realSqrt (some r) = some (Real.sqrt r) := rfl
@[simp] theorem WithBot.realRsqrt_some (r : ℝ) :
    WithBot.realRsqrt (some r) = some (1 / Real.sqrt r) := rfl
@[simp] theorem WithBot.realTanh_some (r : ℝ) :
    WithBot.realTanh (some r) = some (Real.tanh r) := rfl
@[simp] theorem WithBot.realSin_some (r : ℝ) :
    WithBot.realSin (some r) = some (Real.sin r) := rfl
@[simp] theorem WithBot.realCos_some (r : ℝ) :
    WithBot.realCos (some r) = some (Real.cos r) := rfl
@[simp] theorem WithBot.realTan_some (r : ℝ) :
    WithBot.realTan (some r) = some (Real.tan r) := rfl
@[simp] theorem WithBot.realAtan_some (r : ℝ) :
    WithBot.realAtan (some r) = some (Real.arctan r) := rfl
@[simp] theorem WithBot.realCosh_some (r : ℝ) :
    WithBot.realCosh (some r) = some (Real.cosh r) := rfl
@[simp] theorem WithBot.realSinh_some (r : ℝ) :
    WithBot.realSinh (some r) = some (Real.sinh r) := rfl
@[simp] theorem WithBot.realErf_some (r : ℝ) :
    WithBot.realErf (some r) = some (VeriTile.Math.realErf r) := rfl
@[simp] theorem WithBot.realPow_some (a b : ℝ) :
    WithBot.realPow (some a) (some b) = some (Real.rpow a b) := rfl

@[simp] theorem WithBot.realExp_bot :
    WithBot.realExp (⊥ : WithBot ℝ) = ((0 : ℝ) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realExp2_bot :
    WithBot.realExp2 (⊥ : WithBot ℝ) = ((0 : ℝ) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realSigmoid_bot :
    WithBot.realSigmoid (⊥ : WithBot ℝ) = ((0 : ℝ) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realTanh_bot :
    WithBot.realTanh (⊥ : WithBot ℝ) = ((-1 : ℝ) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realAtan_bot :
    WithBot.realAtan (⊥ : WithBot ℝ) = ((-(Real.pi / 2) : ℝ) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realErf_bot :
    WithBot.realErf (⊥ : WithBot ℝ) = ((-1 : ℝ) : WithBot ℝ) := rfl

/-! ### Bot-preserving math primitives

The following math functions are undefined at -∞ in the kernel semantics
(they return `none`/`⊥` on `⊥` input rather than a finite value). These
`_bot` lemmas let `simp` propagate `⊥` cleanly through chained tile
expressions like `tl.log(tl.exp(x) - max)`. -/

@[simp] theorem WithBot.realLog_bot :
    WithBot.realLog (⊥ : WithBot ℝ) = (⊥ : WithBot ℝ) := rfl

@[simp] theorem WithBot.realLog2_bot :
    WithBot.realLog2 (⊥ : WithBot ℝ) = (⊥ : WithBot ℝ) := rfl

@[simp] theorem WithBot.realSqrt_bot :
    WithBot.realSqrt (⊥ : WithBot ℝ) = (⊥ : WithBot ℝ) := rfl

@[simp] theorem WithBot.realRsqrt_bot :
    WithBot.realRsqrt (⊥ : WithBot ℝ) = (⊥ : WithBot ℝ) := rfl

@[simp] theorem WithBot.realSin_bot :
    WithBot.realSin (⊥ : WithBot ℝ) = (⊥ : WithBot ℝ) := rfl

@[simp] theorem WithBot.realCos_bot :
    WithBot.realCos (⊥ : WithBot ℝ) = (⊥ : WithBot ℝ) := rfl

@[simp] theorem WithBot.realTan_bot :
    WithBot.realTan (⊥ : WithBot ℝ) = (⊥ : WithBot ℝ) := rfl

@[simp] theorem WithBot.realCosh_bot :
    WithBot.realCosh (⊥ : WithBot ℝ) = (⊥ : WithBot ℝ) := rfl

@[simp] theorem WithBot.realSinh_bot :
    WithBot.realSinh (⊥ : WithBot ℝ) = (⊥ : WithBot ℝ) := rfl

@[simp] theorem WithBot.realExp_coe (r : ℝ) :
    WithBot.realExp ((r : ℝ) : WithBot ℝ) = (((Real.exp r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realLog_coe (r : ℝ) :
    WithBot.realLog ((r : ℝ) : WithBot ℝ) = (((Real.log r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realLog2_coe (r : ℝ) :
    WithBot.realLog2 ((r : ℝ) : WithBot ℝ) =
      (((Real.log r / Real.log 2 : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realExp2_coe (r : ℝ) :
    WithBot.realExp2 ((r : ℝ) : WithBot ℝ) =
      (((Real.exp (r * Real.log 2) : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realSigmoid_coe (r : ℝ) :
    WithBot.realSigmoid ((r : ℝ) : WithBot ℝ) = (((Real.sigmoid r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realSqrt_coe (r : ℝ) :
    WithBot.realSqrt ((r : ℝ) : WithBot ℝ) = (((Real.sqrt r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realRsqrt_coe (r : ℝ) :
    WithBot.realRsqrt ((r : ℝ) : WithBot ℝ) = (((1 / Real.sqrt r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realTanh_coe (r : ℝ) :
    WithBot.realTanh ((r : ℝ) : WithBot ℝ) = (((Real.tanh r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realSin_coe (r : ℝ) :
    WithBot.realSin ((r : ℝ) : WithBot ℝ) = (((Real.sin r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realCos_coe (r : ℝ) :
    WithBot.realCos ((r : ℝ) : WithBot ℝ) = (((Real.cos r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realTan_coe (r : ℝ) :
    WithBot.realTan ((r : ℝ) : WithBot ℝ) = (((Real.tan r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realAtan_coe (r : ℝ) :
    WithBot.realAtan ((r : ℝ) : WithBot ℝ) = (((Real.arctan r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realCosh_coe (r : ℝ) :
    WithBot.realCosh ((r : ℝ) : WithBot ℝ) = (((Real.cosh r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realSinh_coe (r : ℝ) :
    WithBot.realSinh ((r : ℝ) : WithBot ℝ) = (((Real.sinh r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realErf_coe (r : ℝ) :
    WithBot.realErf ((r : ℝ) : WithBot ℝ) =
      (((VeriTile.Math.realErf r : ℝ)) : WithBot ℝ) := rfl

/-! ### Algebraic simp lemmas on `WithBot ℝ` arithmetic helpers -/

@[simp] theorem WithBot.realAdd_coe_coe (a b : ℝ) :
    WithBot.realAdd ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (((a + b : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realSub_coe_coe (a b : ℝ) :
    WithBot.realSub ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (((a - b : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realMul_coe_coe (a b : ℝ) :
    WithBot.realMul ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (((a * b : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realDiv_coe_coe (a b : ℝ) :
    WithBot.realDiv ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (((a / b : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realPow_coe_coe (a b : ℝ) :
    WithBot.realPow ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (((Real.rpow a b : ℝ)) : WithBot ℝ) := rfl

/-- `realSub ⊥ x = ⊥` (and symmetrically). The corresponding `realAdd` and
`realMul` propagate `⊥` for the same reason — but only `Sub` is needed for the
OnlineSoftmax L proof at iter 0 (`exp(M_0 - M_1) = exp(⊥ - ↑x₀) = exp(⊥) = 0`). -/
@[simp] theorem WithBot.realSub_bot_left (x : WithBot ℝ) :
    WithBot.realSub (⊥ : WithBot ℝ) x = (⊥ : WithBot ℝ) := by
  cases x <;> rfl
@[simp] theorem WithBot.realSub_bot_right (x : WithBot ℝ) :
    WithBot.realSub x (⊥ : WithBot ℝ) = (⊥ : WithBot ℝ) := by
  cases x <;> rfl

@[simp] theorem WithBot.realMul_coe_zero (a : ℝ) :
    WithBot.realMul ((a : ℝ) : WithBot ℝ) (((0 : ℝ)) : WithBot ℝ)
      = (((0 : ℝ)) : WithBot ℝ) := by
  show ((a * 0 : ℝ) : WithBot ℝ) = _
  simp

/-- `↑a + ↑b = ↑(a + b)` for `WithBot ℝ`. Reverse of Mathlib's `WithBot.coe_add`.
NOT marked `@[simp]` to avoid loop with the forward direction; use as `←` rewrite
where consolidating coe outward is desired. -/
theorem WithBot.coe_add_coe_eq (a b : ℝ) :
    ((a : ℝ) : WithBot ℝ) + ((b : ℝ) : WithBot ℝ) = (((a + b : ℝ)) : WithBot ℝ) :=
  (WithBot.coe_add a b).symm

/-- `Option.map f ↑x = ↑(f x)` for `WithBot ℝ`. -/
@[simp] theorem WithBot.coe_map_coe_eq {α} (f : ℝ → α) (a : ℝ) :
    Option.map f ((a : ℝ) : WithBot ℝ) = some (f a) := rfl

/-- Reduce-sum with `keep_dims = false` along an arbitrary Triton axis. -/
noncomputable def Tile.reduceSumDrop {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile .real (TileShape.eraseAxis shape axis) :=
  ⟨fun outIdx =>
    @Finset.sum (Fin (TileShape.axisDim shape axis)) (WithBot ℝ) _ Finset.univ
      (fun k => x.data (TileShape.insertAxisIndex shape axis outIdx k))⟩

/-- Reduce-sum with `keep_dims = true` along an arbitrary Triton axis. -/
noncomputable def Tile.reduceSumKeep {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile .real (TileShape.setAxisOne shape axis) :=
  ⟨fun outIdx =>
    @Finset.sum (Fin (TileShape.axisDim shape axis)) (WithBot ℝ) _ Finset.univ
      (fun k => x.data (TileShape.replaceAxisIndex shape axis outIdx k))⟩

/-- Reduce-sum along an arbitrary Triton axis. -/
noncomputable def Tile.reduceSum {shape : TileShape}
    (axis : Fin shape.length) (keepDims : Bool) (x : Tile .real shape) :
    Tile .real (TileShape.reduceShape shape axis keepDims) := by
  cases keepDims
  · exact Tile.reduceSumDrop axis x
  · exact Tile.reduceSumKeep axis x

/-! ### `simp` lemmas for the structural cases of reduceSum -/

@[simp] theorem Tile.reduceSum_false {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile.reduceSum axis false x = Tile.reduceSumDrop axis x := rfl

@[simp] theorem Tile.reduceSum_true {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile.reduceSum axis true x = Tile.reduceSumKeep axis x := rfl

/-- Body of `reduceSumDrop` at an output index. Lets `simp` push past the
opaque `Tile` constructor so kernel proofs can reach the `Finset.sum` form. -/
@[simp] theorem Tile.reduceSumDrop_data {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape)
    (outIdx : TileIndex (TileShape.eraseAxis shape axis)) :
    (Tile.reduceSumDrop axis x).data outIdx =
      @Finset.sum (Fin (TileShape.axisDim shape axis)) (WithBot ℝ) _ Finset.univ
        (fun k => x.data (TileShape.insertAxisIndex shape axis outIdx k)) := rfl

@[simp] theorem Tile.reduceSumKeep_data {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape)
    (outIdx : TileIndex (TileShape.setAxisOne shape axis)) :
    (Tile.reduceSumKeep axis x).data outIdx =
      @Finset.sum (Fin (TileShape.axisDim shape axis)) (WithBot ℝ) _ Finset.univ
        (fun k => x.data (TileShape.replaceAxisIndex shape axis outIdx k)) := rfl

/-! ### `simp` helpers for the `WithBot ℝ` boundary

A typical kernel proof reduces `tl.sum(x)` to `∑ i, (some (xs i) : WithBot ℝ)`,
then `tl.store` demotes via `Option.getD … 0`. Mathlib's `WithBot.coe_sum`
gives us `↑(∑ f i) = ∑ ↑(f i)` (for `WithBot.coe = some`); read backwards, it
collapses the sum-of-`some`s to a `some` of the sum, which then evaluates the
`getD`. -/

/-- `WithBot.unbotD` on a `some _` projects out the value. Same as
`WithBot.unbotD_coe` but stated in the `some`-form `evalOp` produces. -/
@[simp] theorem WithBot.unbotD_some {α} (d a : α) :
    WithBot.unbotD d (some a : WithBot α) = a := rfl

/-- `Option.map₂` on two `some`-lifted values produces a `some`-lifted result.
Required because `Tile.bop` for `.real` arithmetic uses `Option.map₂`. -/
@[simp] theorem Option.map₂_some_some (f : ℝ → ℝ → ℝ) (a b : ℝ) :
    Option.map₂ f (some a : WithBot ℝ) (some b : WithBot ℝ)
      = (some (f a b) : WithBot ℝ) := rfl

@[simp] theorem Option.map_some_real (f : ℝ → ℝ) (a : ℝ) :
    Option.map f (some a : WithBot ℝ) = (some (f a) : WithBot ℝ) := rfl

/-- `↑`-form: when proofs land on the coe view of `WithBot ℝ`, the same
reductions apply. Simp matches syntactically, so we need both forms. -/
@[simp] theorem Option.map₂_coe_coe (f : ℝ → ℝ → ℝ) (a b : ℝ) :
    Option.map₂ f ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = ((f a b : ℝ) : WithBot ℝ) := rfl

@[simp] theorem Option.map_coe_real (f : ℝ → ℝ) (a : ℝ) :
    Option.map f ((a : ℝ) : WithBot ℝ) = ((f a : ℝ) : WithBot ℝ) := rfl

/-- Mixed-form: `some` on left, `↑` on right (and vice versa). Both forms appear
in goals after partial simp normalization. -/
@[simp] theorem Option.map₂_some_coe (f : ℝ → ℝ → ℝ) (a b : ℝ) :
    Option.map₂ f (some a : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (some (f a b) : WithBot ℝ) := rfl

@[simp] theorem Option.map₂_coe_some (f : ℝ → ℝ → ℝ) (a b : ℝ) :
    Option.map₂ f ((a : ℝ) : WithBot ℝ) (some b : WithBot ℝ)
      = (some (f a b) : WithBot ℝ) := rfl


@[simp] theorem WithBot.sum_some_eq_some {ι} (s : Finset ι) (f : ι → ℝ) :
    (∑ i ∈ s, ((f i : ℝ) : WithBot ℝ)) = (((∑ i ∈ s, f i) : ℝ) : WithBot ℝ) :=
  (WithBot.coe_sum s f).symm

/-- `some`-form companion to `WithBot.sum_some_eq_some`. The explicit
`AddCommMonoid (WithBot ℝ)` ascription via `@Finset.sum` is what makes the
typeclass resolution fire — bare `some _ : WithBot ℝ` ascription elaborates
to `Option ℝ` for typeclass purposes. -/
@[simp] theorem WithBot.sum_someTerm_eq_some {ι} (s : Finset ι) (f : ι → ℝ) :
    @Finset.sum ι (WithBot ℝ) _ s (fun i => (some (f i) : WithBot ℝ))
      = (some (∑ i ∈ s, f i) : WithBot ℝ) := by
  show @Finset.sum ι (WithBot ℝ) _ s (fun i => ((f i : ℝ) : WithBot ℝ)) = _
  rw [WithBot.sum_some_eq_some]
  rfl


/-- After a `tl.sum` reduce on a `some`-lifted tile, demoting via `unbotD 0`
(used by `tl.store`) recovers the underlying ℝ-valued sum.

Stated in the `some`-form (rather than `↑`) because Real `tl.load` values
arrive in the `some` carrier and `tl.sum` on a tile of `some`-values gives
`∑ some (xs i)` literally. -/
@[simp] theorem WithBot.unbotD_sum_some {ι} (s : Finset ι) (f : ι → ℝ) :
    WithBot.unbotD (0 : ℝ) (∑ i ∈ s, (some (f i) : WithBot ℝ)) = ∑ i ∈ s, f i := by
  show WithBot.unbotD (0 : ℝ) (∑ i ∈ s, ((f i : ℝ) : WithBot ℝ)) = ∑ i ∈ s, f i
  rw [WithBot.sum_some_eq_some]
  rfl

/-- `sup'` over `↑`-lifted values equals `↑` of the `sup'`. -/
theorem WithBot.sup'_coe {ι} [LinearOrder ι] (s : Finset ι) (h : s.Nonempty)
    (f : ι → ℝ) :
    s.sup' h (fun i => ((f i : ℝ) : WithBot ℝ)) = ((s.sup' h f : ℝ) : WithBot ℝ) := by
  refine h.cons_induction ?_ ?_
  · intro a; rfl
  · intro a s' ha hne ih
    rw [Finset.sup'_cons hne, Finset.sup'_cons hne, ih]
    rfl

/-- `some`-form of `WithBot.sup'_coe`. Required because `evalOp` produces
literal `some _` calls, not `↑`. The `@Finset.sup' (WithBot ℝ) ι` ascription
forces typeclass resolution to use `SemilatticeSup (WithBot ℝ)` rather than
the elaborated `Option ℝ`. -/
@[simp] theorem WithBot.sup'_someTerm_eq_some {ι} [LinearOrder ι]
    (s : Finset ι) (h : s.Nonempty) (f : ι → ℝ) :
    @Finset.sup' (WithBot ℝ) ι _ s h (fun i => (some (f i) : WithBot ℝ))
      = (some (s.sup' h f) : WithBot ℝ) := by
  show @Finset.sup' (WithBot ℝ) ι _ s h (fun i => ((f i : ℝ) : WithBot ℝ)) = _
  rw [WithBot.sup'_coe]; rfl

/-- Companion to `WithBot.sup'_coe`: store-side demote on a max reduce. -/
@[simp] theorem WithBot.unbotD_sup'_some {ι} [LinearOrder ι] (s : Finset ι)
    (h : s.Nonempty) (f : ι → ℝ) :
    WithBot.unbotD (0 : ℝ) (s.sup' h (fun i => (some (f i) : WithBot ℝ)))
      = s.sup' h f := by
  show WithBot.unbotD (0 : ℝ) (s.sup' h (fun i => ((f i : ℝ) : WithBot ℝ))) = _
  rw [WithBot.sup'_coe]; rfl

/-! ### Reduce-max along an arbitrary axis

Returns `none` only when the reduced dimension is `0` (empty `sup'`
undefined). -/

noncomputable def Tile.reduceMaxDrop {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Option (Tile .real (TileShape.eraseAxis shape axis)) :=
  if h : 0 < TileShape.axisDim shape axis then
    some ⟨fun outIdx =>
      (Finset.univ : Finset (Fin (TileShape.axisDim shape axis))).sup'
        (by exact ⟨⟨0, h⟩, Finset.mem_univ _⟩)
        (fun k => x.data (TileShape.insertAxisIndex shape axis outIdx k))⟩
  else
    none

/-- Tile.reduceMaxDrop returns `none` when the reduction axis has zero
length. Useful as a simp lemma when proving correctness of kernels that
guard `tl.max(..., return_indices=False)` against empty axes. -/
@[simp] theorem Tile.reduceMaxDrop_eq_none {shape : TileShape}
    {axis : Fin shape.length} {x : Tile .real shape}
    (h : TileShape.axisDim shape axis = 0) :
    Tile.reduceMaxDrop axis x = none := by
  unfold Tile.reduceMaxDrop
  rw [dif_neg (by omega)]

noncomputable def Tile.reduceMaxKeep {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Option (Tile .real (TileShape.setAxisOne shape axis)) :=
  if h : 0 < TileShape.axisDim shape axis then
    some ⟨fun outIdx =>
      (Finset.univ : Finset (Fin (TileShape.axisDim shape axis))).sup'
        (by exact ⟨⟨0, h⟩, Finset.mem_univ _⟩)
        (fun k => x.data (TileShape.replaceAxisIndex shape axis outIdx k))⟩
  else
    none

noncomputable def Tile.reduceMax {shape : TileShape}
    (axis : Fin shape.length) (keepDims : Bool) (x : Tile .real shape) :
    Option (Tile .real (TileShape.reduceShape shape axis keepDims)) := by
  cases keepDims
  · exact Tile.reduceMaxDrop axis x
  · exact Tile.reduceMaxKeep axis x

/-- Nat-valued reduce-max along an arbitrary axis. -/
noncomputable def Tile.reduceMaxNatDrop {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .nat shape) :
    Option (Tile .nat (TileShape.eraseAxis shape axis)) :=
  if h : 0 < TileShape.axisDim shape axis then
    some ⟨fun outIdx =>
      (Finset.univ : Finset (Fin (TileShape.axisDim shape axis))).sup'
        (by exact ⟨⟨0, h⟩, Finset.mem_univ _⟩)
        (fun k => x.data (TileShape.insertAxisIndex shape axis outIdx k))⟩
  else
    none

noncomputable def Tile.reduceMaxNatKeep {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .nat shape) :
    Option (Tile .nat (TileShape.setAxisOne shape axis)) :=
  if h : 0 < TileShape.axisDim shape axis then
    some ⟨fun outIdx =>
      (Finset.univ : Finset (Fin (TileShape.axisDim shape axis))).sup'
        (by exact ⟨⟨0, h⟩, Finset.mem_univ _⟩)
        (fun k => x.data (TileShape.replaceAxisIndex shape axis outIdx k))⟩
  else
    none

noncomputable def Tile.reduceMaxNat {shape : TileShape}
    (axis : Fin shape.length) (keepDims : Bool) (x : Tile .nat shape) :
    Option (Tile .nat (TileShape.reduceShape shape axis keepDims)) := by
  cases keepDims
  · exact Tile.reduceMaxNatDrop axis x
  · exact Tile.reduceMaxNatKeep axis x

/-- Directed scan along an axis. At each output index, folds all input lanes
whose coordinate on `axis` is `≤` the output coordinate (`.forward`, a prefix
scan) or `≥` it (`.reverse`, a suffix scan — `reverse=True`). Every `ScanOp`
is commutative and associative, so the fold order within the kept lane set is
immaterial. -/
noncomputable def Tile.scan {shape : TileShape}
    (op : ScanOp) (axis : Fin shape.length) (dir : ScanDirection)
    (x : Tile .real shape) : Tile .real shape :=
  match dir with
  | .forward =>
    ⟨fun idx =>
      let coord := TileShape.axisCoord shape axis idx
      let vals :=
        ((Finset.univ : Finset (Fin (TileShape.axisDim shape axis))).filter
          (fun k => k.val ≤ coord.val)).toList.map
          (fun k => x.data (TileShape.replaceAxisCoord shape axis idx k))
      op.eval vals⟩
  | .reverse =>
    ⟨fun idx =>
      let coord := TileShape.axisCoord shape axis idx
      let vals :=
        ((Finset.univ : Finset (Fin (TileShape.axisDim shape axis))).filter
          (fun k => coord.val ≤ k.val)).toList.map
          (fun k => x.data (TileShape.replaceAxisCoord shape axis idx k))
      op.eval vals⟩

/-- Direct unfolding of forward `Tile.scan.data` to a list-folded `op.eval`
over the prefix of lanes at most the output coordinate on `axis`. Useful as a
simp lemma when proving cumsum/recurrent kernel correctness. -/
@[simp] theorem Tile.scan_data {shape : TileShape}
    (op : ScanOp) (axis : Fin shape.length) (x : Tile .real shape)
    (idx : TileIndex shape) :
    (Tile.scan op axis .forward x).data idx =
      op.eval
        (((Finset.univ : Finset (Fin (TileShape.axisDim shape axis))).filter
          (fun k => k.val ≤ (TileShape.axisCoord shape axis idx).val)).toList.map
          (fun k => x.data (TileShape.replaceAxisCoord shape axis idx k))) := rfl

/-- Direct unfolding of reverse `Tile.scan.data` to a list-folded `op.eval`
over the suffix of lanes at least the output coordinate on `axis`. -/
@[simp] theorem Tile.scan_reverse_data {shape : TileShape}
    (op : ScanOp) (axis : Fin shape.length) (x : Tile .real shape)
    (idx : TileIndex shape) :
    (Tile.scan op axis .reverse x).data idx =
      op.eval
        (((Finset.univ : Finset (Fin (TileShape.axisDim shape axis))).filter
          (fun k => (TileShape.axisCoord shape axis idx).val ≤ k.val)).toList.map
          (fun k => x.data (TileShape.replaceAxisCoord shape axis idx k))) := rfl

noncomputable def Tile.argBestDrop {shape : TileShape}
    (better : WithBot ℝ → WithBot ℝ → Bool)
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile .nat (TileShape.eraseAxis shape axis) :=
  ⟨fun outIdx =>
    if h : 0 < TileShape.axisDim shape axis then
      let first : Fin (TileShape.axisDim shape axis) := ⟨0, h⟩
      let best :=
        (List.finRange (TileShape.axisDim shape axis)).foldl
          (fun best k =>
            if better
                (x.data (TileShape.insertAxisIndex shape axis outIdx k))
                (x.data (TileShape.insertAxisIndex shape axis outIdx best)) then
              k
            else
              best)
          first
      best.val
    else
      0⟩

/-- Axis index of the maximum value. Ties keep the earlier/smaller axis
coordinate because replacement only occurs on strict improvement. -/
noncomputable def Tile.argMaxDrop {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile .nat (TileShape.eraseAxis shape axis) :=
  Tile.argBestDrop (fun candidate current => decide (current < candidate)) axis x

/-- Axis index of the minimum value. Ties keep the earlier/smaller axis
coordinate because replacement only occurs on strict improvement. -/
noncomputable def Tile.argMinDrop {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile .nat (TileShape.eraseAxis shape axis) :=
  Tile.argBestDrop (fun candidate current => decide (candidate < current)) axis x

/-- Definitional unfolding of `Tile.argMaxDrop` to `Tile.argBestDrop` over
strict-less-than. Useful as a simp lemma for proving argmax kernel
correctness. -/
@[simp] theorem Tile.argMaxDrop_unfold {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile.argMaxDrop axis x =
      Tile.argBestDrop (fun candidate current => decide (current < candidate)) axis x :=
  rfl

/-- Definitional unfolding of `Tile.argMinDrop` to `Tile.argBestDrop` over
strict-less-than (with arguments swapped). -/
@[simp] theorem Tile.argMinDrop_unfold {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile.argMinDrop axis x =
      Tile.argBestDrop (fun candidate current => decide (candidate < current)) axis x :=
  rfl

/-- Sort values along `axis` in ascending order, preserving all other
coordinates. -/
noncomputable def Tile.sortAxis {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile .real shape :=
  ⟨fun idx =>
    let coord := TileShape.axisCoord shape axis idx
    let vals :=
      (List.finRange (TileShape.axisDim shape axis)).map
        (fun k => x.data (TileShape.replaceAxisCoord shape axis idx k))
    (vals.mergeSort (fun a b => decide (a ≤ b))).getD coord.val ⊥⟩

/-! ### `simp` lemmas for the structural cases of reduceMax -/

@[simp] theorem Tile.reduceMax_false {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile.reduceMax axis false x = Tile.reduceMaxDrop axis x := rfl

@[simp] theorem Tile.reduceMax_true {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile.reduceMax axis true x = Tile.reduceMaxKeep axis x := rfl

def Tile.natToReal {shape} (x : Tile .nat shape) : Tile .real shape :=
  ⟨fun i => some ((x.data i : ℝ))⟩

/-- `Tile.natToReal` lifts each `Nat` lane to a `some` real value. Useful
when bridging Nat-channel indices into Real-channel arithmetic (e.g. when
adding `tl.arange(0, BLOCK_SIZE)` to a pointer that's then multiplied by a
real value). -/
@[simp] theorem Tile.natToReal_data {shape : TileShape}
    (x : Tile .nat shape) (idx : TileIndex shape) :
    (Tile.natToReal x).data idx = some ((x.data idx : ℝ)) := rfl

/-- Block-level (possibly batched) matrix multiply (Triton's `tl.dot`):
`c[…, m, n] = ∑_k a[…, m, k] * b[…, k, n]`.

Operates on the trailing two dims; any batch prefix `batch : TileShape` is
broadcast pointwise. Structural recursion on the batch fixes one outer
index per level and recurses on the remaining batch.

`⊥`-propagating: if any factor in any pair is `⊥` the corresponding
summand is `⊥`, and `Finset.sum` over `WithBot ℝ` propagates `⊥` through
the whole entry. Well-formed kernels load real (non-`⊥`) values into the
operands, so the dot product collapses to a regular `↑(∑ a*b)` term and
the `tl.store` demotion via `unbotD 0` recovers the underlying ℝ value. -/
noncomputable def Tile.dot : (batch : TileShape) → {M K N : Nat} →
    Tile .real (batch ++ [M, K]) → Tile .real (batch ++ [K, N]) →
    Tile .real (batch ++ [M, N])
  | [], _, K, _, a, b =>
      ⟨fun (m, n, _) =>
        @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
          (fun k => Option.map₂ (· * ·)
                      (a.data (m, k, PUnit.unit))
                      (b.data (k, n, PUnit.unit)))⟩
  | _ :: rest, M, K, N, a, b =>
      ⟨fun (i, restIdx) =>
        (Tile.dot rest (M := M) (K := K) (N := N)
            ⟨fun rIdx => a.data (i, rIdx)⟩
            ⟨fun rIdx => b.data (i, rIdx)⟩).data restIdx⟩

/-- Body of `Tile.dot` at the rank-2 base case (`batch = []`); lets `simp`
expose the `Finset.sum` form for kernel proofs. -/
@[simp] theorem Tile.dot_nil_data {M K N : Nat}
    (a : Tile .real [M, K]) (b : Tile .real [K, N])
    (m : Fin M) (n : Fin N) (rest : TileIndex []) :
    (Tile.dot [] a b).data (m, n, rest) =
      @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·)
                    (a.data (m, k, PUnit.unit))
                    (b.data (k, n, PUnit.unit))) := rfl

/-- Recursive step: `(Tile.dot (b :: rest) a b').data (i, restIdx)` slices
each operand at outer index `i` and recurses on the remaining batch. -/
@[simp] theorem Tile.dot_cons_data {b : Nat} {rest : TileShape} {M K N : Nat}
    (a : Tile .real ((b :: rest) ++ [M, K]))
    (b' : Tile .real ((b :: rest) ++ [K, N]))
    (i : Fin b) (restIdx : TileIndex (rest ++ [M, N])) :
    (Tile.dot (b :: rest) a b').data (i, restIdx) =
      (Tile.dot rest (M := M) (K := K) (N := N)
          ⟨fun rIdx => a.data (i, rIdx)⟩
          ⟨fun rIdx => b'.data (i, rIdx)⟩).data restIdx := rfl

/-- Trailing-two-axes transpose (`Tile.dot`-style framework: matrix dims
are the trailing two, leading dims are an unchanged `batch` prefix). -/
def Tile.transpose : {dtype : TileDType} → (batch : TileShape) →
    {M N : Nat} →
    Tile dtype (batch ++ [M, N]) → Tile dtype (batch ++ [N, M])
  | _, [],         _, _, x =>
      ⟨fun (n, m, _) => x.data (m, n, PUnit.unit)⟩
  | _, _ :: rest,  _, _, x =>
      ⟨fun (i, restIdx) =>
        (Tile.transpose rest
            ⟨fun rIdx => x.data (i, rIdx)⟩).data restIdx⟩

/-- Body of `Tile.transpose` at the rank-2 base case: swap `(m, n) ↔ (n, m)`. -/
@[simp] theorem Tile.transpose_nil_data {dtype : TileDType} {M N : Nat}
    (x : Tile dtype [M, N]) (n : Fin N) (m : Fin M) (rest : TileIndex []) :
    (Tile.transpose [] x).data (n, m, rest) =
      x.data (m, n, PUnit.unit) := rfl

/-- Recursive step: `(Tile.transpose (b :: rest) x).data (i, restIdx)`
slices at outer index `i` and recurses on the remaining batch. -/
@[simp] theorem Tile.transpose_cons_data {dtype : TileDType}
    {b : Nat} {rest : TileShape} {M N : Nat}
    (x : Tile dtype ((b :: rest) ++ [M, N]))
    (i : Fin b) (restIdx : TileIndex (rest ++ [N, M])) :
    (Tile.transpose (b :: rest) x).data (i, restIdx) =
      (Tile.transpose rest (M := M) (N := N)
          ⟨fun rIdx => x.data (i, rIdx)⟩).data restIdx := rfl

/-- Insert a unit-size axis at position `axis`. The output index is
projected back to the input by `dropInsertedIndex`, which throws away
the inserted slot's `Fin 1` coordinate. -/
def Tile.expandDim {dtype : TileDType} {shape : TileShape}
    (axis : Fin (shape.length + 1)) (x : Tile dtype shape) :
    Tile dtype (TileShape.insertAxis shape axis 1) :=
  ⟨fun idx => x.data (TileShape.dropInsertedIndex shape axis 1 idx)⟩

@[simp] theorem Tile.expandDim_data {dtype : TileDType} {shape : TileShape}
    (axis : Fin (shape.length + 1)) (x : Tile dtype shape)
    (idx : TileIndex (TileShape.insertAxis shape axis 1)) :
    (Tile.expandDim axis x).data idx =
      x.data (TileShape.dropInsertedIndex shape axis 1 idx) := rfl

/-- Lift a plain ℝ-valued tile-shaped function into a `Tile .real`. Useful
at the spec / boundary layer: `BlockState.readMem` reads ℝ, never `⊥`, so a
`tl.load`-fed kernel input is naturally a `TileIndex shape → ℝ`. The
`⊥` sentinel of `WithBot ℝ` is reserved for `-inf` / masked-off /
`tl.full(_, -inf)` values introduced *inside* a kernel; spec-level
inputs and outputs should round-trip through `ofReal` / `unbotD 0`. -/
def Tile.ofReal {shape : TileShape} (x : TileIndex shape → ℝ) :
    Tile .real shape :=
  ⟨fun i => some (x i)⟩

@[simp] theorem Tile.ofReal_data {shape : TileShape}
    (x : TileIndex shape → ℝ) (i : TileIndex shape) :
    (Tile.ofReal x).data i = some (x i) := rfl

/-- Element-wise select (`tl.where(cond, a, b)`): per-cell, pick from `a`
when `cond` is `true`, else from `b`. Same-shape; broadcast lifting is
done at the DSL layer. Named `select` to avoid the Lean `where` keyword
in definition position; the AST constructor `Op.where` and DSL surface
`tl.where(...)` keep the user-facing Triton spelling. -/
def Tile.select {dtype : TileDType} {shape : TileShape}
    (c : Tile .bool shape) (a b : Tile dtype shape) :
    Tile dtype shape :=
  ⟨fun idx => if c.data idx then a.data idx else b.data idx⟩

@[simp] theorem Tile.select_data {dtype : TileDType} {shape : TileShape}
    (c : Tile .bool shape) (a b : Tile dtype shape) (idx : TileIndex shape) :
    (Tile.select c a b).data idx = if c.data idx then a.data idx else b.data idx := rfl

end VeriTile.Triton
