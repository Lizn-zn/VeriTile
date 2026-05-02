/-
VeriTile.Triton.Core

Data types for the embedded Triton subset (Phase 1 scope).

Scope decisions for P1:
* Single-block, single program_id, deterministic execution.
* Rank-polymorphic tile shapes in the core; individual operators may still
  expose only the ranks currently modeled by the DSL / semantics.
* Floating-point arithmetic modelled in `ℝ` (Mathlib `Real`).
* Excluded: `tl.atomic_*`, async copy, multi-block coordination,
  Hopper/Blackwell-specific ops (TMA, WGMMA).

Operational semantics live in `VeriTile.Triton.Semantics`.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.List.Nodup
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
`tl.store` demotes via `unbot' 0` — well-formed kernels never store `⊥`. -/
abbrev TileCarrier : TileDType → Type
  | .real => WithBot ℝ
  | .nat  => Nat
  | .bool => Bool
  | .ptr  => RegionName × Nat

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

@[simp] theorem axisDim_head (d : Nat) (rest : TileShape) :
    axisDim (d :: rest) ⟨0, Nat.succ_pos _⟩ = d := rfl

@[simp] theorem eraseAxis_head (d : Nat) (rest : TileShape) :
    eraseAxis (d :: rest) ⟨0, Nat.succ_pos _⟩ = rest := rfl

@[simp] theorem setAxisOne_head (d : Nat) (rest : TileShape) :
    setAxisOne (d :: rest) ⟨0, Nat.succ_pos _⟩ = 1 :: rest := rfl

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

/-- DTypes that support Triton's arithmetic operators in the current model. -/
inductive NumericDType : TileDType → Type where
  | real : NumericDType .real
  | nat  : NumericDType .nat

/-- DTypes that support Triton's comparison operators in the current model. -/
inductive ComparableDType : TileDType → Type where
  | real : ComparableDType .real
  | nat  : ComparableDType .nat

/--
P1 Triton expressions.

Each constructor models one Triton expression or block-level reduction.
Statement-level constructs (assignment, control flow, memory writes) live
in `Stmt`.

Notes on individual constructors:
* `const c`     produces an `ℝ`-valued data scalar.
* `constNat n`  produces a `Nat`-valued address/size scalar (used in
                offset arithmetic, tile lengths, and program-id-derived
                indices). See RP2 for the rationale behind separating the
                `ℝ` and `Nat` channels.
* `negInf` is a sentinel for `tl.full((), -inf)` used in `tl.full(... -float('inf'))`.
* `programId axis` returns the current `tl.program_id(axis)` as a `Nat`
                    scalar. Out-of-range axes evaluate to `0` per the
                    `BlockState.pids` total-function model.
* `arange n` produces a length-`n` `Nat`-valued tile `[0, 1, ..., n-1]`.
* `broadcast e n` lifts a scalar to a length-`n` tile.
* `full n e` fills a length-`n` tile with the scalar value of `e`.
* `reduceMax`/`reduceSum` are block-level reductions along an arbitrary
  Triton axis. The `keepDims` flag controls whether the reduced rank dim is
  stripped (`false`) or collapsed to `1` (`true`), matching Triton's
  `tl.sum / tl.max` `keep_dims=` kwarg.
* `load region offset` evaluates `offset` (a `Nat`-valued scalar or tile,
  i.e. produced from `constNat` / `programId` / `arange` / `Nat`-arithmetic)
  and reads from `region`. Scalar offset = single-cell read; tile offset
  = gather.
* `natToReal` lifts a `Nat`-channel value (`scalarNat` / `tileNat`) into
  the `ℝ` channel. Used by kernels that mix loop counters / sizes with
  ℝ data (e.g. Welford's `delta / (i + 1)`, division by block size).
* `sqrt` applies `Real.sqrt` pointwise on the `ℝ` channel. Used by
  LayerNorm's `1 / √(var + ε)` and similar normalization kernels.
* `lt`/`le`/`eq`/`gt`/`ge`/`ne` are pointwise comparison operators
  producing values in the **Bool channel** (`scalarBool` / `tileBool`).
  Both `Nat × Nat → Bool` and `ℝ × ℝ → Bool` carriers are supported via
  typed constructors; mixed-channel comparison rejects at elaboration time.
  Shape semantics follow Triton: `()×()→()`, `(n)×()→(n)`, `()×(n)→(n)`,
  `(n)×(n)→(n)` (length match required for tile×tile).
* `load region offset opts` reads from `region`. Per RP1, region is
  a kernel-level name. Per the mask extension (Issue #16/#17):
  - `opts.mask = none, opts.other = none`: classic unmasked load. Result shape
    follows `offset` shape (scalarNat → scalar; tileNat → tile gather).
  - `opts.mask = some m, opts.other = some o`: masked load. `m` evaluates to a
    `scalarBool` (broadcasts) or `tileBool` (per-lane, length-matching).
    For each lane where `m` is `true`, read from memory; where `false`,
    use `o`. Result still follows `offset` shape.
  - `opts.mask = some m, opts.other = none`: Triton leaves masked-off values
    undefined. The operational semantics models this with the state's
    `undef` oracle.
  - `opts.mask = none, opts.other = some o`: `none` (semantic error).
-/
inductive Op : TileDType → TileShape → Type where
  | const     : ℝ → Op .real []
  | constNat  : Nat → Op .nat []
  | constBool : Bool → Op .bool []
  | negInf    : Op .real []
  | programId : (axis : Nat) → Op .nat []
  | ref       : (dtype : TileDType) → (shape : TileShape) → RegName → Op dtype shape
  | arange    : (n : Nat) → Op .nat [n]
  | broadcast : Op dtype [] → (shape : TileShape) → Op dtype shape
  | full      : (shape : TileShape) → Op dtype [] → Op dtype shape
  | add       : NumericDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op dtype out
  | sub       : NumericDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op dtype out
  | mul       : NumericDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op dtype out
  | div       : NumericDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op dtype out
  | exp       : Op .real shape → Op .real shape
  | log       : Op .real shape → Op .real shape
  | sigmoid   : Op .real shape → Op .real shape
  | sqrt      : Op .real shape → Op .real shape
  | tanh      : Op .real shape → Op .real shape
  | lt        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | le        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | eq        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | gt        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | ge        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | ne        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | boolAnd   : Broadcast a b out → Op .bool a → Op .bool b → Op .bool out
  | max2      : Broadcast a b out → Op .real a → Op .real b → Op .real out
  /--
  Element-wise select (`tl.where(cond, a, b)`): pick `a` where the bool
  tile `cond` is `true`, else pick `b`. All three arguments share the
  same shape; broadcasting (e.g., scalar `-inf`) is handled at the DSL
  layer by lifting through `Op.broadcast` / `Op.full` before this node
  is constructed. -/
  | where     : {dtype : TileDType} → {shape : TileShape} →
                Op .bool shape → Op dtype shape → Op dtype shape →
                Op dtype shape
  /-- Block-level reduce-max along an arbitrary axis. -/
  | reduceMax : (axis : Fin shape.length) → (keepDims : Bool) →
                Op .real shape →
                Op .real (TileShape.reduceShape shape axis keepDims)
  /-- Block-level reduce-sum along an arbitrary axis. -/
  | reduceSum : (axis : Fin shape.length) → (keepDims : Bool) →
                Op .real shape →
                Op .real (TileShape.reduceShape shape axis keepDims)
  /--
  Block-level (possibly batched) matrix multiply (`tl.dot` in Triton):
  `c[…, m, n] = ∑_k a[…, m, k] * b[…, k, n]`.

  Operates on the **last two dims** of each operand; any leading batch
  prefix (`batch : TileShape`) is broadcast pointwise. Two-arg
  `tl.dot(a, b)` is the rank-2 case `batch = []`; FA-2 / grouped-GEMM
  kernels use a non-empty batch prefix.

  Both operands must agree on the batch prefix and on the shared inner
  dim `K`. The accumulator form `tl.dot(a, b, acc)` desugars at the DSL
  level to `acc + tl.dot(a, b)`, so the AST has only the binary node. -/
  | dot       : {batch : TileShape} → {M K N : Nat} →
                Op .real (batch ++ [M, K]) → Op .real (batch ++ [K, N]) →
                Op .real (batch ++ [M, N])
  /--
  Trailing-two-axes transpose (`x.T` in Triton, mirrors how `Op.dot`
  treats the trailing two dims as the matrix and any leading dims as a
  passthrough `batch` prefix). For 2D this is the standard `.T`; for
  rank ≥ 3 it transposes the inner matrix at every batch coordinate.

  Arbitrary axis permutations (e.g. swapping leading axes or cyclic
  reorderings) are *not* expressible by this constructor alone. They
  can be composed from multiple `transpose`s in some — but not all —
  cases; a fully general `Op.permuteAxes` is left as a future
  follow-up issue. -/
  | transpose : {dtype : TileDType} → {batch : TileShape} → {M N : Nat} →
                Op dtype (batch ++ [M, N]) →
                Op dtype (batch ++ [N, M])
  /--
  Insert a unit-size axis at position `axis` of `shape`. Models Triton's
  `tl.expand_dims(e, axis = K)` and the slicer surface forms `e[:, None]`
  / `e[None, :]`. Fully ND: `axis` ranges over `Fin (shape.length + 1)`,
  one slot more than the input rank. -/
  | expandDim : {dtype : TileDType} → {shape : TileShape} →
                (axis : Fin (shape.length + 1)) →
                Op dtype shape →
                Op dtype (TileShape.insertAxis shape axis 1)
  | ptrBase   : (region : RegionName) → Op .ptr []
  | ptrAdd    : Broadcast a b out → Op .ptr a → Op .nat b → Op .ptr out
  | load      : (region : RegionName) → (offset : Op .nat shape) → Op .real shape
  | loadMask  : (region : RegionName) → (offset : Op .nat shape) →
                (mask : Op .bool shape) → Op .real shape
  | loadMaskOther : (region : RegionName) → (offset : Op .nat shape) →
                (mask : Op .bool shape) → (other : Op .real shape) → Op .real shape
  | loadPtr   : Op .ptr shape → Op .real shape
  | loadPtrMask : Op .ptr shape → Op .bool shape → Op .real shape
  | loadPtrMaskOther : Op .ptr shape → Op .bool shape → Op .real shape → Op .real shape
  | natToReal : Op .nat shape → Op .real shape

/--
P1 Triton statements (mutating constructs).

* `assign name e` defines or updates the register `name` to the value of `e`.
* `store region offset value mask` writes `value` to `region` at `offset`. If
  `value` is a tile and `offset` is a scalar, the tile is written contiguously
  starting at `offset`. With `mask = none`, every lane is written. With
  `mask = some m`, lanes where `m` evaluates `false` are **left untouched**
  (Triton `tl.store` semantics — no `other` parameter on store side).
* `forLoop i n body` runs `body` `n` times, with the scalar register `i` bound
  to the iteration index.
* `ifThen cond body` runs `body` when the scalar `cond` evaluates `true`, and
  is a no-op when `false`. This is a VeriTile DSL-level scalar conditional
  used to model Triton block-level control-flow patterns. It has no `else`;
  the FA-2 block-skipping pattern `if start_n + BLOCK_N <= start_m: continue`
  rewrites to `if not skippable { ...work... }`.
-/
inductive Stmt : Type where
  | assign  : (dtype : TileDType) → (shape : TileShape) → RegName → Op dtype shape → Stmt
  | store   : (region : RegionName) → (shape : TileShape) →
              (offset : Op .nat shape) → (value : Op .real shape) → Stmt
  | storeMask : (region : RegionName) → (shape : TileShape) →
              (offset : Op .nat shape) → (value : Op .real shape) →
              (mask : Op .bool shape) → Stmt
  | storePtr : (shape : TileShape) →
              (ptr : Op .ptr shape) → (value : Op .real shape) → Stmt
  | storePtrMask : (shape : TileShape) →
              (ptr : Op .ptr shape) → (value : Op .real shape) →
              (mask : Op .bool shape) → Stmt
  | forLoop : (idx : RegName) → (n : Nat) → (body : List Stmt) → Stmt
  | ifThen  : (cond : Op .bool []) → (body : List Stmt) → Stmt

instance : Inhabited Stmt :=
  ⟨.assign .real [] "" (.const 0)⟩

/--
A complete Triton kernel.

* `inputs` / `outputs` are the names of memory regions referenced by the kernel.
  These are metadata only; the operational semantics treats memory as a single
  global region map.
* `body` is the sequence of statements executed for each `program_id`.
-/
structure Kernel where
  inputs  : List RegionName
  outputs : List RegionName
  body    : List Stmt
  deriving Inhabited

end VeriTile.Triton
