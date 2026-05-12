/-
VeriTile.Triton.Core.Shape

Tile shapes, indices, tiles, and broadcasting for the Triton core AST.
-/

import Mathlib.Data.List.Nodup
import VeriTile.Triton.Core.Types

namespace VeriTile.Triton

/--
Shape of a Triton block-local tile.

Shapes are represented **outermost-first**, matching Triton / NumPy / PyTorch:
a user shape `(M, N)` (M rows, N cols) is stored as `[M, N]`. The head of
the list is the outermost dim, the last element is the innermost dim.
User axis `K` corresponds directly to list index `K`.

Reduction axes use Triton / NumPy convention directly: user axis `K` is list
index `K`.
-/
abbrev TileShape := List Nat

/-- Index type for a tile shape. -/
abbrev TileIndex : TileShape → Type
  | [] => PUnit
  | n :: rest => Fin n × TileIndex rest

namespace TileShape

/-- Convert a dependent tile index to an outermost-first list of coordinates. -/
def indexToList : (shape : TileShape) → TileIndex shape → List Nat
  | [], _ => []
  | _ :: rest, idx => idx.1.val :: indexToList rest idx.2

/-- Dimension at an axis. -/
def axisDim : (shape : TileShape) → Fin shape.length → Nat
  | [], axis => nomatch axis
  | d :: _, ⟨0, _⟩ => d
  | _ :: rest, ⟨k + 1, h⟩ => axisDim rest ⟨k, Nat.succ_lt_succ_iff.mp h⟩

/-- Remove an axis from a shape (`keep_dims = false` reduction shape). -/
def eraseAxis : (shape : TileShape) → Fin shape.length → TileShape
  | [], axis => nomatch axis
  | _ :: rest, ⟨0, _⟩ => rest
  | d :: rest, ⟨k + 1, h⟩ => d :: eraseAxis rest ⟨k, Nat.succ_lt_succ_iff.mp h⟩

/-- Replace an axis by dimension `1` (`keep_dims = true` reduction shape). -/
def setAxisOne : (shape : TileShape) → Fin shape.length → TileShape
  | [], axis => nomatch axis
  | _ :: rest, ⟨0, _⟩ => 1 :: rest
  | d :: rest, ⟨k + 1, h⟩ => d :: setAxisOne rest ⟨k, Nat.succ_lt_succ_iff.mp h⟩

/-- Output shape of a reduction over `axis`. -/
def reduceShape (shape : TileShape) (axis : Fin shape.length) (keepDims : Bool) :
    TileShape :=
  if keepDims then setAxisOne shape axis else eraseAxis shape axis

/-- Insert the reduced-axis coordinate into a `keep_dims = false` output index. -/
def insertAxisIndex :
    (shape : TileShape) → (axis : Fin shape.length) →
    TileIndex (eraseAxis shape axis) → Fin (axisDim shape axis) → TileIndex shape
  | [], axis, _, _ => nomatch axis
  | _ :: _, ⟨0, _⟩, outIdx, k => (k, outIdx)
  | _ :: rest, ⟨k + 1, h⟩, outIdx, r =>
      (outIdx.1,
       insertAxisIndex rest ⟨k, Nat.succ_lt_succ_iff.mp h⟩ outIdx.2 r)

/-- Replace the unit reduced-axis coordinate of a `keep_dims = true` output
index with the concrete reduced-axis coordinate. -/
def replaceAxisIndex :
    (shape : TileShape) → (axis : Fin shape.length) →
    TileIndex (setAxisOne shape axis) → Fin (axisDim shape axis) → TileIndex shape
  | [], axis, _, _ => nomatch axis
  | _ :: _, ⟨0, _⟩, outIdx, k => (k, outIdx.2)
  | _ :: rest, ⟨k + 1, h⟩, outIdx, r =>
      (outIdx.1,
       replaceAxisIndex rest ⟨k, Nat.succ_lt_succ_iff.mp h⟩ outIdx.2 r)

/-- Read the coordinate at `axis` from a same-shape index. -/
def axisCoord :
    (shape : TileShape) → (axis : Fin shape.length) →
    TileIndex shape → Fin (axisDim shape axis)
  | [], axis, _ => nomatch axis
  | _ :: _, ⟨0, _⟩, idx => idx.1
  | _ :: rest, ⟨k + 1, h⟩, idx =>
      axisCoord rest ⟨k, Nat.succ_lt_succ_iff.mp h⟩ idx.2

/-- Replace the coordinate at `axis` in a same-shape index. -/
def replaceAxisCoord :
    (shape : TileShape) → (axis : Fin shape.length) →
    TileIndex shape → Fin (axisDim shape axis) → TileIndex shape
  | [], axis, _, _ => nomatch axis
  | _ :: _, ⟨0, _⟩, idx, r => (r, idx.2)
  | _ :: rest, ⟨k + 1, h⟩, idx, r =>
      (idx.1,
       replaceAxisCoord rest ⟨k, Nat.succ_lt_succ_iff.mp h⟩ idx.2 r)

/-- Flip the coordinate at `axis`, preserving all other coordinates. -/
def flipIndex (shape : TileShape) (axis : Fin shape.length)
    (idx : TileIndex shape) : TileIndex shape :=
  replaceAxisCoord shape axis idx (Fin.rev (axisCoord shape axis idx))

/-- Product of all dimensions. Scalars have one element. -/
def product : TileShape → Nat
  | [] => 1
  | d :: rest => d * product rest

/-- Row-major / outer-to-inner linear index. -/
def linearIndex : (shape : TileShape) → TileIndex shape → Nat
  | [], _ => 0
  | _ :: rest, idx => idx.1.val * product rest + linearIndex rest idx.2

/-- Append a final axis coordinate to an index. -/
def appendAxisIndex :
    (shape : TileShape) → TileIndex shape → Fin n → TileIndex (shape ++ [n])
  | [], _, k => (k, PUnit.unit)
  | _ :: rest, idx, k => (idx.1, appendAxisIndex rest idx.2 k)

/-- Split the final coordinate from an index of `shape ++ [n]`. -/
def splitLastIndex :
    (shape : TileShape) → TileIndex (shape ++ [n]) → TileIndex shape × Fin n
  | [], idx => (PUnit.unit, idx.1)
  | _ :: rest, idx =>
      let tail := splitLastIndex rest idx.2
      ((idx.1, tail.1), tail.2)

@[simp] theorem axisDim_head (d : Nat) (rest : TileShape) :
    axisDim (d :: rest) ⟨0, Nat.succ_pos _⟩ = d := rfl

@[simp] theorem eraseAxis_head (d : Nat) (rest : TileShape) :
    eraseAxis (d :: rest) ⟨0, Nat.succ_pos _⟩ = rest := rfl

@[simp] theorem eraseAxis_succ (d : Nat) (rest : TileShape) (k : Nat)
    (h : k + 1 < (d :: rest).length) :
    eraseAxis (d :: rest) ⟨k + 1, h⟩ =
      d :: eraseAxis rest ⟨k, Nat.succ_lt_succ_iff.mp h⟩ := rfl

@[simp] theorem setAxisOne_head (d : Nat) (rest : TileShape) :
    setAxisOne (d :: rest) ⟨0, Nat.succ_pos _⟩ = 1 :: rest := rfl

@[simp] theorem setAxisOne_succ (d : Nat) (rest : TileShape) (k : Nat)
    (h : k + 1 < (d :: rest).length) :
    setAxisOne (d :: rest) ⟨k + 1, h⟩ =
      d :: setAxisOne rest ⟨k, Nat.succ_lt_succ_iff.mp h⟩ := rfl

@[simp] theorem reduceShape_false (shape : TileShape) (axis : Fin shape.length) :
    reduceShape shape axis false = eraseAxis shape axis := rfl

@[simp] theorem reduceShape_true (shape : TileShape) (axis : Fin shape.length) :
    reduceShape shape axis true = setAxisOne shape axis := rfl

@[simp] theorem insertAxisIndex_head {d : Nat} {rest : TileShape}
    (outIdx : TileIndex rest) (k : Fin d) :
    insertAxisIndex (d :: rest) ⟨0, Nat.succ_pos _⟩ outIdx k = (k, outIdx) := rfl

@[simp] theorem replaceAxisIndex_head {d : Nat} {rest : TileShape}
    (outIdx : TileIndex (1 :: rest)) (k : Fin d) :
    replaceAxisIndex (d :: rest) ⟨0, Nat.succ_pos _⟩ outIdx k = (k, outIdx.2) := rfl

@[simp] theorem axisCoord_head {d : Nat} {rest : TileShape}
    (idx : TileIndex (d :: rest)) :
    axisCoord (d :: rest) ⟨0, Nat.succ_pos _⟩ idx = idx.1 := rfl

@[simp] theorem replaceAxisCoord_head {d : Nat} {rest : TileShape}
    (idx : TileIndex (d :: rest)) (k : Fin d) :
    replaceAxisCoord (d :: rest) ⟨0, Nat.succ_pos _⟩ idx k = (k, idx.2) := rfl

@[simp] theorem product_nil :
    product [] = 1 := rfl

@[simp] theorem product_cons (d : Nat) (rest : TileShape) :
    product (d :: rest) = d * product rest := rfl

@[simp] theorem appendAxisIndex_nil (k : Fin n) :
    appendAxisIndex [] PUnit.unit k = (k, PUnit.unit) := rfl

@[simp] theorem appendAxisIndex_cons {d n : Nat} {rest : TileShape}
    (idx : TileIndex (d :: rest)) (k : Fin n) :
    appendAxisIndex (d :: rest) idx k =
      (idx.1, appendAxisIndex rest idx.2 k) := rfl

@[simp] theorem splitLastIndex_nil (idx : TileIndex [n]) :
    splitLastIndex [] idx = (PUnit.unit, idx.1) := rfl

@[simp] theorem splitLastIndex_cons {d n : Nat} {rest : TileShape}
    (idx : TileIndex ((d :: rest) ++ [n])) :
    splitLastIndex (d :: rest) idx =
      let tail := splitLastIndex rest idx.2
      ((idx.1, tail.1), tail.2) := rfl

/-- Insert a new axis of size `n` at position `axis` of `shape`. The axis
position ranges over `Fin (shape.length + 1)` (one slot more than the
original rank — past the existing axes). Used by `Op.expandDim` to encode
Triton's `e[:, None]` / `e[None, :]` / `tl.expand_dims` shape change.

Matches on `shape` first (then `axis`) so the empty-shape case picks up
the only `Fin 1` value at the function level — this keeps each constructor
clause `rfl`-reducible to a concrete `List Nat` when `shape` is a literal
in user code. -/
def insertAxis : (shape : TileShape) → Fin (shape.length + 1) → Nat → TileShape
  | [],         _,          n => [n]
  | d :: rest,  ⟨0, _⟩,     n => n :: d :: rest
  | d :: rest,  ⟨k + 1, h⟩, n =>
      d :: insertAxis rest ⟨k, Nat.succ_lt_succ_iff.mp h⟩ n

/-- Project an output index of `insertAxis shape axis n` back to an input
index of `shape` by dropping the inserted-axis component. The dropped
component lives in `Fin n`; semantics that need to use it (e.g. broadcast
arguments other than `n = 1`) read it before calling.

Matches `shape` then `axis` to mirror `insertAxis`. -/
def dropInsertedIndex :
    (shape : TileShape) → (axis : Fin (shape.length + 1)) → (n : Nat) →
    TileIndex (insertAxis shape axis n) → TileIndex shape
  | [],         _,         _, _   => PUnit.unit
  | _ :: _,     ⟨0, _⟩,    _, idx => idx.2
  | _ :: rest,  ⟨k + 1, h⟩, n, idx =>
      (idx.1, dropInsertedIndex rest ⟨k, Nat.succ_lt_succ_iff.mp h⟩ n idx.2)

@[simp] theorem insertAxis_zero_nil (n : Nat) :
    insertAxis [] ⟨0, Nat.succ_pos _⟩ n = [n] := rfl

@[simp] theorem insertAxis_zero_cons (d : Nat) (rest : TileShape) (n : Nat) :
    insertAxis (d :: rest) ⟨0, Nat.succ_pos _⟩ n = n :: d :: rest := rfl

@[simp] theorem insertAxis_succ {d : Nat} {rest : TileShape}
    {k : Nat} {h : k + 1 < rest.length + 2} {n : Nat} :
    insertAxis (d :: rest) ⟨k + 1, h⟩ n =
      d :: insertAxis rest ⟨k, Nat.succ_lt_succ_iff.mp h⟩ n := rfl

@[simp] theorem length_insertAxis
    (shape : TileShape) (axis : Fin (shape.length + 1)) (n : Nat) :
    (insertAxis shape axis n).length = shape.length + 1 := by
  induction shape with
  | nil =>
      cases axis
      rfl
  | cons d rest ih =>
      cases axis with
      | mk k hk =>
          cases k with
          | zero => rfl
          | succ k =>
              simp [insertAxis, ih]

/-- Generic head-case rewrite, useful when `shape` is abstract (the
per-cons simp lemmas above already fire when `shape` is a concrete
list). -/
theorem insertAxis_head (shape : TileShape) (n : Nat) :
    insertAxis shape ⟨0, Nat.succ_pos _⟩ n = n :: shape := by
  cases shape <;> rfl

@[simp] theorem dropInsertedIndex_nil (n : Nat) (idx : TileIndex [n]) :
    dropInsertedIndex [] ⟨0, Nat.succ_pos _⟩ n idx = PUnit.unit := rfl

@[simp] theorem dropInsertedIndex_zero_cons {d : Nat} {rest : TileShape} {n : Nat}
    (idx : TileIndex (n :: d :: rest)) :
    dropInsertedIndex (d :: rest) ⟨0, Nat.succ_pos _⟩ n idx = idx.2 := rfl

@[simp] theorem dropInsertedIndex_succ {d : Nat} {rest : TileShape}
    {k : Nat} {h : k + 1 < rest.length + 2} {n : Nat}
    (idx : TileIndex (insertAxis (d :: rest) ⟨k + 1, h⟩ n)) :
    dropInsertedIndex (d :: rest) ⟨k + 1, h⟩ n idx =
      (idx.1, dropInsertedIndex rest ⟨k, Nat.succ_lt_succ_iff.mp h⟩ n idx.2) := rfl

/-- Enumerate all indices of a shape, in row-major / outer-to-inner order. -/
def allIndices : (shape : TileShape) → List (TileIndex shape)
  | [] => [PUnit.unit]
  | n :: rest =>
      (List.finRange n).flatMap fun i =>
        (allIndices rest).map fun is => (i, is)

@[simp] theorem allIndices_nil :
    TileShape.allIndices ([] : TileShape) = [PUnit.unit] := rfl

/-- `allIndices_cons` is NOT marked `@[simp]` on purpose: keeping
`(TileShape.allIndices (n :: rest)).foldl ...` opaque under `simp` lets
`scatter_readback_nd` fire by pattern-matching on the head form. Use
`unfold TileShape.allIndices` (or `simp [TileShape.allIndices]`) explicitly
when you need the inductive step. -/
theorem allIndices_cons (n : Nat) (rest : TileShape) :
    TileShape.allIndices (n :: rest) =
      (List.finRange n).flatMap fun i =>
        (TileShape.allIndices rest).map fun is => (i, is) := rfl

@[simp] theorem mem_allIndices : ∀ (shape : TileShape) (i : TileIndex shape),
    i ∈ allIndices shape
  | [], i => by
      cases i
      simp [allIndices]
  | _ :: rest, i => by
      rcases i with ⟨hd, tl⟩
      simp [allIndices, mem_allIndices rest tl]

@[simp] theorem allIndices_nodup : ∀ (shape : TileShape), (allIndices shape).Nodup
  | [] => by simp [allIndices]
  | n :: rest => by
      rw [allIndices, List.nodup_flatMap]
      constructor
      · intro _ _
        exact (allIndices_nodup rest).map (fun _ _ h => by cases h; rfl)
      · exact (List.nodup_finRange n).imp (fun hdis => by
          unfold Function.onFun
          rw [List.disjoint_left]
          intro z hz1 hz2
          rcases List.mem_map.1 hz1 with ⟨_, _, rfl⟩
          rcases List.mem_map.1 hz2 with ⟨_, _, hzb⟩
          injection hzb with hxy _
          exact hdis hxy.symm)

end TileShape

/-- A shaped, typed Triton tile. -/
structure Tile (dtype : TileDType) (shape : TileShape) where
  data : TileIndex shape → TileCarrier dtype

namespace Tile

/-- Extensionality for tiles: tiles are equal when their data functions agree
pointwise. This keeps ND kernel proofs from expanding the `Tile` structure
manually every time they compare loaded/computed tiles. -/
@[ext] theorem ext {dtype : TileDType} {shape : TileShape}
    {x y : Tile dtype shape} (h : ∀ i, x.data i = y.data i) : x = y := by
  cases x
  cases y
  simp only at h
  congr
  exact funext h

/-- Rank-0 tile constructor. -/
def scalar {dtype : TileDType} (x : TileCarrier dtype) : Tile dtype [] :=
  ⟨fun _ => x⟩

/-- 1D tile constructor. -/
def vec {dtype : TileDType} {n : Nat}
    (f : Fin n → TileCarrier dtype) : Tile dtype [n] :=
  ⟨fun i => f i.1⟩

/-- 2D tile constructor (outermost-first storage: a `(rows × cols)` matrix
has shape `[rows, cols]`, indexed as `(row, col, _)`). -/
def mat {dtype : TileDType} {rows cols : Nat}
    (f : Fin rows → Fin cols → TileCarrier dtype) : Tile dtype [rows, cols] :=
  ⟨fun (row, col, _) => f row col⟩

end Tile

/-- Binary broadcasting cases supported by Triton elementwise operators. -/
inductive Broadcast : TileShape → TileShape → TileShape → Type where
  | nil      : Broadcast [] [] []
  | scalarL  : Broadcast [] (n :: r) (n :: r)
  | scalarR  : Broadcast (n :: r) [] (n :: r)
  | consSame : Broadcast a b c → Broadcast (n :: a) (n :: b) (n :: c)
  | consL    : Broadcast a b c → Broadcast (1 :: a) (n :: b) (n :: c)
  | consR    : Broadcast a b c → Broadcast (n :: a) (1 :: b) (n :: c)

namespace Broadcast

/-- Evaluate the output index back into each input index for a broadcast. -/
def leftIndex {a b out : TileShape} :
    Broadcast a b out → TileIndex out → TileIndex a
  | .nil, _ => PUnit.unit
  | .scalarL, _ => PUnit.unit
  | .scalarR, i => i
  | .consSame bc, i => (i.1, leftIndex bc i.2)
  | .consL bc, i => (⟨0, Nat.succ_pos 0⟩, leftIndex bc i.2)
  | .consR bc, i => (i.1, leftIndex bc i.2)

/-- Evaluate the output index back into each input index for a broadcast. -/
def rightIndex {a b out : TileShape} :
    Broadcast a b out → TileIndex out → TileIndex b
  | .nil, _ => PUnit.unit
  | .scalarL, i => i
  | .scalarR, _ => PUnit.unit
  | .consSame bc, i => (i.1, rightIndex bc i.2)
  | .consL bc, i => (i.1, rightIndex bc i.2)
  | .consR bc, i => (⟨0, Nat.succ_pos 0⟩, rightIndex bc i.2)

/-! ### `simp` equation lemmas for `leftIndex` / `rightIndex`

`def`s with pattern matching on dependent inductives sometimes don't unfold under
`simp [Broadcast.leftIndex]` directly (Lean's simp normaliser leaves the head
applied). These per-constructor equation lemmas plug that gap so 1D and 2D
kernel proofs uniformly reduce broadcast indexing one cons-case at a time. -/

@[simp] theorem leftIndex_nil (i : TileIndex []) :
    leftIndex (.nil : Broadcast [] [] []) i = PUnit.unit := rfl

@[simp] theorem leftIndex_scalarL (i : TileIndex (n :: r)) :
    leftIndex (.scalarL : Broadcast [] (n :: r) (n :: r)) i = PUnit.unit := rfl

@[simp] theorem leftIndex_scalarR (i : TileIndex (n :: r)) :
    leftIndex (.scalarR : Broadcast (n :: r) [] (n :: r)) i = i := rfl

@[simp] theorem leftIndex_consSame {a b c : TileShape}
    (bc : Broadcast a b c) (i : TileIndex (n :: c)) :
    leftIndex (.consSame bc : Broadcast (n :: a) (n :: b) (n :: c)) i =
      (i.1, leftIndex bc i.2) := rfl

@[simp] theorem leftIndex_consL {a b c : TileShape}
    (bc : Broadcast a b c) (i : TileIndex (n :: c)) :
    leftIndex (.consL bc : Broadcast (1 :: a) (n :: b) (n :: c)) i =
      (⟨0, Nat.succ_pos 0⟩, leftIndex bc i.2) := rfl

@[simp] theorem leftIndex_consR {a b c : TileShape}
    (bc : Broadcast a b c) (i : TileIndex (n :: c)) :
    leftIndex (.consR bc : Broadcast (n :: a) (1 :: b) (n :: c)) i =
      (i.1, leftIndex bc i.2) := rfl

@[simp] theorem rightIndex_nil (i : TileIndex []) :
    rightIndex (.nil : Broadcast [] [] []) i = PUnit.unit := rfl

@[simp] theorem rightIndex_scalarL (i : TileIndex (n :: r)) :
    rightIndex (.scalarL : Broadcast [] (n :: r) (n :: r)) i = i := rfl

@[simp] theorem rightIndex_scalarR (i : TileIndex (n :: r)) :
    rightIndex (.scalarR : Broadcast (n :: r) [] (n :: r)) i = PUnit.unit := rfl

@[simp] theorem rightIndex_consSame {a b c : TileShape}
    (bc : Broadcast a b c) (i : TileIndex (n :: c)) :
    rightIndex (.consSame bc : Broadcast (n :: a) (n :: b) (n :: c)) i =
      (i.1, rightIndex bc i.2) := rfl

@[simp] theorem rightIndex_consL {a b c : TileShape}
    (bc : Broadcast a b c) (i : TileIndex (n :: c)) :
    rightIndex (.consL bc : Broadcast (1 :: a) (n :: b) (n :: c)) i =
      (i.1, rightIndex bc i.2) := rfl

@[simp] theorem rightIndex_consR {a b c : TileShape}
    (bc : Broadcast a b c) (i : TileIndex (n :: c)) :
    rightIndex (.consR bc : Broadcast (n :: a) (1 :: b) (n :: c)) i =
      (⟨0, Nat.succ_pos 0⟩, rightIndex bc i.2) := rfl

end Broadcast

end VeriTile.Triton
