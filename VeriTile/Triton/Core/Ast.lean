/-
VeriTile.Triton.Core.Ast

Typed expression, statement, and kernel AST definitions for the Triton core.
-/

import VeriTile.Triton.Core.Shape

namespace VeriTile.Triton

/-
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
* `load dtype mem mask` reads from a `MemAccess`: named-region plus `.nat`
  offsets, first-class pointer values, or block pointers with checked axes.
  Scalar offset = single-cell read; tile offset = gather. With `MaskOpt.mask`,
  masked-off lanes use the state's `undef` oracle; with `MaskOpt.maskOther`,
  masked-off lanes use the supplied `other` value.
-/
inductive ScanOp where
  | sum
  | prod
  | max
  | min
  deriving Repr, BEq

mutual

inductive Op : TileDType → TileShape → Type where
  | const     : ℝ → Op .real []
  | constFloat : (dtype : FloatDType) → ℝ → Op dtype.toTileDType []
  | constNat  : Nat → Op .nat []
  | constBool : Bool → Op .bool []
  | negInf    : Op .real []
  | programId : (axis : Nat) → Op .nat []
  | ref       : (dtype : TileDType) → (shape : TileShape) → RegName → Op dtype shape
  | arange    : (n : Nat) → Op .nat [n]
  | broadcast : Op dtype [] → (shape : TileShape) → Op dtype shape
  | full      : (shape : TileShape) → Op dtype [] → Op dtype shape
  | castFloat : (src dst : FloatDType) → Op src.toTileDType shape → Op dst.toTileDType shape
  | add       : NumericDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op dtype out
  | sub       : NumericDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op dtype out
  | mul       : NumericDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op dtype out
  | div       : NumericDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op dtype out
  | floorDiv  : IntegralDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op dtype out
  | mod       : IntegralDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op dtype out
  | bitAnd    : Broadcast a b out → Op .nat a → Op .nat b → Op .nat out
  | bitOr     : Broadcast a b out → Op .nat a → Op .nat b → Op .nat out
  | bitXor    : Broadcast a b out → Op .nat a → Op .nat b → Op .nat out
  | shiftLeft : Broadcast a b out → Op .nat a → Op .nat b → Op .nat out
  | shiftRight : Broadcast a b out → Op .nat a → Op .nat b → Op .nat out
  | exp       : Op .real shape → Op .real shape
  | exp2      : Op .real shape → Op .real shape
  | log       : Op .real shape → Op .real shape
  | log2      : Op .real shape → Op .real shape
  | sigmoid   : Op .real shape → Op .real shape
  | sqrt      : Op .real shape → Op .real shape
  | tanh      : Op .real shape → Op .real shape
  | sin       : Op .real shape → Op .real shape
  | cos       : Op .real shape → Op .real shape
  | tan       : Op .real shape → Op .real shape
  | atan      : Op .real shape → Op .real shape
  | cosh      : Op .real shape → Op .real shape
  | sinh      : Op .real shape → Op .real shape
  | lt        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | le        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | eq        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | gt        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | ge        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | ne        : ComparableDType dtype → Broadcast a b out → Op dtype a → Op dtype b → Op .bool out
  | boolAnd   : Broadcast a b out → Op .bool a → Op .bool b → Op .bool out
  | boolOr    : Broadcast a b out → Op .bool a → Op .bool b → Op .bool out
  | boolNot   : Op .bool shape → Op .bool shape
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
  /-- Prefix scan along an axis. `tl.cumsum` and `tl.cumprod` lower to this
  node with `.sum` / `.prod`; `tl.associative_scan` accepts the closed
  `ScanOp` enum rather than arbitrary functions. -/
  | scan      : (op : ScanOp) → (axis : Fin shape.length) →
                Op .real shape → Op .real shape
  /-- Axis index of the maximum value, with ties resolved toward the smallest
  axis coordinate. -/
  | argMax    : (axis : Fin shape.length) → Op .real shape →
                Op .nat (TileShape.eraseAxis shape axis)
  /-- Axis index of the minimum value, with ties resolved toward the smallest
  axis coordinate. -/
  | argMin    : (axis : Fin shape.length) → Op .real shape →
                Op .nat (TileShape.eraseAxis shape axis)
  /-- Sort values along `axis` in ascending order. -/
  | sort      : (axis : Fin shape.length) → Op .real shape → Op .real shape
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
  /-- Generic row-major reshape/view. The DSL checks obvious literal element
  count mismatches; the total semantics defaults out-of-range remaps to the
  dtype default carrier so this node remains total on symbolic shapes. -/
  | reshape  : {dtype : TileDType} → (outShape : TileShape) →
                Op dtype shape → Op dtype outShape
  /-- Generic pure index-remap view. Used by surface `tl.flip` and
  `tl.permute`, where the macro can generate a typed `TileIndex` map. -/
  | remap    : {dtype : TileDType} → (outShape : TileShape) →
                (TileIndex outShape → TileIndex shape) →
                Op dtype shape → Op dtype outShape
  /-- Join two same-shaped tensors by adding a final minor axis of size 2. -/
  | join     : {dtype : TileDType} → {shape : TileShape} →
                Op dtype shape → Op dtype shape → Op dtype (shape ++ [2])
  /-- Project one side of a final size-2 minor axis. This is the expression
  form used by the DSL for `tl.split(x, 0)` / `tl.split(x, 1)`. -/
  | split    : {dtype : TileDType} → {shape : TileShape} →
                (side : Fin 2) → Op dtype (shape ++ [2]) → Op dtype shape
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
  | makeBlockPtr : (region : RegionName) → (baseOffset : Nat) →
                (parentShape : List Nat) → (blockShape : TileShape) →
                (strides offsets : List Nat) →
                Op .blockPtr blockShape
  | advanceBlockPtr : Op .blockPtr shape → (offsetDeltas : List Nat) → Op .blockPtr shape
  | load      : (dtype : TileDType) → MemAccess shape →
                MaskOpt dtype shape → Op dtype shape
  | natToReal : Op .nat shape → Op .real shape

/-- Memory address form shared by load and store nodes. -/
inductive MemAccess : TileShape → Type where
  | region : (region : RegionName) → (offset : Op .nat shape) → MemAccess shape
  | ptr : Op .ptr shape → MemAccess shape
  | blockPtr : Op .blockPtr shape → (boundaryCheck : List Nat) → MemAccess shape

/-- Optional mask/other clause for Triton memory operations. -/
inductive MaskOpt : TileDType → TileShape → Type where
  | none : MaskOpt dtype shape
  | mask : Op .bool shape → MaskOpt dtype shape
  | maskOther : Op .bool shape → Op dtype shape → MaskOpt dtype shape

end

namespace Op

/-- Ergonomic constructor for the common unmasked Real region load. -/
def load' (region : RegionName) (off : Op .nat shape) : Op .real shape :=
  .load .real (.region region off) .none

end Op

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
  | store   : (dtype : TileDType) → (shape : TileShape) →
              MemAccess shape → (value : Op dtype shape) →
              (mask : MaskOpt dtype shape) → Stmt
  | forLoop : (idx : RegName) → (n : Nat) → (body : List Stmt) → Stmt
  | ifThen  : (cond : Op .bool []) → (body : List Stmt) → Stmt

instance : Inhabited Stmt :=
  ⟨.assign .real [] "" (.const 0)⟩

namespace Stmt

/-- Ergonomic constructor for the common unmasked Real region store. -/
def store' (region : RegionName) (shape : TileShape)
    (off : Op .nat shape) (value : Op .real shape) : Stmt :=
  .store .real shape (.region region off) value .none

end Stmt

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

/-! ## Compute-facing kernel wrapper -/

/-- Long-term name for the algorithm-layer dtype universe. -/
abbrev AlgDType := TileDType

/-- Long-term name for the algorithm-layer expression AST. -/
abbrev AlgOp := Op

/-- Long-term name for the algorithm-layer kernel AST. -/
abbrev AlgKernel := Kernel

/--
Compute-facing kernel surface.

For now every compute kernel is just an algorithm kernel wrapper. Future
compute-only nodes such as bitcast will extend this type, while algorithm
correctness continues to project through `ComputeKernel.toAlgorithm?`.
-/
inductive ComputeKernel where
  | fromAlg : AlgKernel → ComputeKernel
  deriving Inhabited

namespace ComputeKernel

/-- Project the current algorithm-only compute kernel subset back to `Kernel`. -/
def toAlgKernel : ComputeKernel → AlgKernel
  | .fromAlg k => k

instance : Coe ComputeKernel AlgKernel where
  coe := toAlgKernel

@[simp] theorem toAlgKernel_fromAlg (k : AlgKernel) :
    (ComputeKernel.fromAlg k).toAlgKernel = k := rfl

@[simp] theorem coe_fromAlg (k : AlgKernel) :
    ((ComputeKernel.fromAlg k : ComputeKernel) : AlgKernel) = k := rfl

end ComputeKernel

end VeriTile.Triton
