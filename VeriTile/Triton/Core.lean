/-
VeriTile.Triton.Core

Data types for the embedded Triton subset (Phase 1 scope).

Scope decisions for P1:
* Single-block, single program_id, deterministic execution.
* 1-D tiles only (vectors).
* Floating-point arithmetic modelled in `ℝ` (Mathlib `Real`).
* Excluded: `tl.atomic_*`, `tl.dot`, async copy, multi-block coordination,
  Hopper/Blackwell-specific ops (TMA, WGMMA).

Operational semantics live in `VeriTile.Triton.Semantics`.
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
  | nat
  | bool
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

/--
Shape of a Triton block-local tile.

`scalar` is shape `()`, distinct from `vec 1`; this matches Triton's
broadcasting behavior. `mat` is the planned 2D extension for block-pointer and
attention kernels. A fully variadic shape can replace this once 2D support has
settled.
-/
inductive TileShape where
  | scalar
  | vec : Nat → TileShape
  | mat : Nat → Nat → TileShape
  deriving DecidableEq, Repr

/-- Index type for a tile shape. -/
abbrev TileIndex : TileShape → Type
  | .scalar  => PUnit
  | .vec n   => Fin n
  | .mat m n => Fin m × Fin n

/-- A shaped, typed Triton tile. -/
structure Tile (dtype : TileDType) (shape : TileShape) where
  data : TileIndex shape → TileCarrier dtype

namespace Tile

/-- Rank-0 tile constructor. -/
def scalar {dtype : TileDType} (x : TileCarrier dtype) : Tile dtype .scalar :=
  ⟨fun _ => x⟩

/-- 1D tile constructor. -/
def vec {dtype : TileDType} {n : Nat}
    (f : Fin n → TileCarrier dtype) : Tile dtype (.vec n) :=
  ⟨f⟩

/-- 2D tile constructor. -/
def mat {dtype : TileDType} {m n : Nat}
    (f : Fin m → Fin n → TileCarrier dtype) : Tile dtype (.mat m n) :=
  ⟨fun i => f i.1 i.2⟩

end Tile

/-- Binary broadcasting cases supported by Triton elementwise operators. -/
inductive Broadcast : TileShape → TileShape → TileShape → Type where
  | same   : Broadcast shape shape shape
  | left   : Broadcast .scalar shape shape
  | right  : Broadcast shape .scalar shape

/-- Evaluate the output index back into each input index for a broadcast. -/
def Broadcast.leftIndex {a b out : TileShape} :
    Broadcast a b out → TileIndex out → TileIndex a
  | .same, i => i
  | .left, _ => PUnit.unit
  | .right, i => i

/-- Evaluate the output index back into each input index for a broadcast. -/
def Broadcast.rightIndex {a b out : TileShape} :
    Broadcast a b out → TileIndex out → TileIndex b
  | .same, i => i
  | .left, i => i
  | .right, _ => PUnit.unit

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
* `programId` returns the current `tl.program_id(axis=0)` as a `Nat` scalar.
* `arange n` produces a length-`n` `Nat`-valued tile `[0, 1, ..., n-1]`.
* `broadcast e n` lifts a scalar to a length-`n` tile.
* `full n e` fills a length-`n` tile with the scalar value of `e`.
* `reduceMax`/`reduceSum` are block-level `axis=0` reductions on a tile.
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
  | const     : ℝ → Op .real .scalar
  | constNat  : Nat → Op .nat .scalar
  | constBool : Bool → Op .bool .scalar
  | negInf    : Op .real .scalar
  | programId : Op .nat .scalar
  | ref       : (dtype : TileDType) → (shape : TileShape) → RegName → Op dtype shape
  | arange    : (n : Nat) → Op .nat (.vec n)
  | broadcast : Op dtype .scalar → (shape : TileShape) → Op dtype shape
  | full      : (shape : TileShape) → Op dtype .scalar → Op dtype shape
  | add       : NumericDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op dtype out
  | sub       : NumericDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op dtype out
  | mul       : NumericDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op dtype out
  | div       : NumericDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op dtype out
  | exp       : Op .real shape → Op .real shape
  | log       : Op .real shape → Op .real shape
  | sigmoid   : Op .real shape → Op .real shape
  | sqrt      : Op .real shape → Op .real shape
  | lt        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | le        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | eq        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | gt        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | ge        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | ne        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | max2      : Broadcast a b out → Op .real a → Op .real b → Op .real out
  | reduceMax : Op .real (.vec n) → Op .real .scalar
  | reduceSum : Op .real (.vec n) → Op .real .scalar
  | load      : (region : RegionName) → (offset : Op .nat shape) → Op .real shape
  | loadMask  : (region : RegionName) → (offset : Op .nat shape) →
                (mask : Op .bool shape) → Op .real shape
  | loadMaskOther : (region : RegionName) → (offset : Op .nat shape) →
                (mask : Op .bool shape) → (other : Op .real shape) → Op .real shape
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
-/
inductive Stmt : Type where
  | assign  : (dtype : TileDType) → (shape : TileShape) → RegName → Op dtype shape → Stmt
  | store   : (region : RegionName) → (shape : TileShape) →
              (offset : Op .nat shape) → (value : Op .real shape) → Stmt
  | storeMask : (region : RegionName) → (shape : TileShape) →
              (offset : Op .nat shape) → (value : Op .real shape) →
              (mask : Op .bool shape) → Stmt
  | forLoop : (idx : RegName) → (n : Nat) → (body : List Stmt) → Stmt

instance : Inhabited Stmt :=
  ⟨.assign .real .scalar "" (.const 0)⟩

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
