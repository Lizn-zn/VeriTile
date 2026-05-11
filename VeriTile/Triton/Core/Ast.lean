/-
VeriTile.Triton.Core.Ast

Typed expression, statement, and kernel AST definitions for the Triton core.
-/

import Mathlib.Data.Rat.Defs
import VeriTile.Triton.Core.Shape

namespace VeriTile.Triton

set_option linter.unusedVariables false

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
* `negInf` is a sentinel for `tl.full((), -inf)` used in
  `tl.full(... -float("inf"))`.
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

/-- Read-modify-write operation tag shared by algorithm AST and traces. -/
inductive RMWOp where
  | add
  | max
  | min
  | xchg
  | cas
  deriving DecidableEq, BEq, Repr

mutual

inductive Op : TileDType → TileShape → Type where
  | const     : ℝ → Op .real []
  | constFloat : (dtype : FloatDType) → ℝ → Op dtype.toTileDType []
  | constNat  : Nat → Op .nat []
  | constInt  : Int → Op .int []
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
  | erf       : Op .real shape → Op .real shape
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
  | ptrBase   : {d : TileDType} → (region : Region d) → Op .ptr []
  | ptrAdd    : {d : TileDType} → Broadcast a b out → Op .ptr a → Op .nat b → Op .ptr out
  | makeBlockPtr : {d : TileDType} → (region : Region d) → (baseOffset : Nat) →
                (parentShape : List Nat) → (blockShape : TileShape) →
                (strides offsets : List Nat) →
                Op .blockPtr blockShape
  | advanceBlockPtr : {d : TileDType} → Op .blockPtr shape → (offsetDeltas : List Nat) → Op .blockPtr shape
  | load      : (dtype : TileDType) → MemAccess dtype shape →
                MaskOpt dtype shape → Op dtype shape
  | natToReal : Op .nat shape → Op .real shape

/-- Memory address form shared by load and store nodes.

The `d` index records the access's element dtype (the channel `Op.load`
returns / `Stmt.store` writes). Direct regions carry the same phantom dtype.
Pointer and block-pointer carriers keep their flat runtime dtype tags, but
their constructors carry the element dtype as an implicit phantom argument. -/
inductive MemAccess : (d : TileDType) → TileShape → Type where
  | region : (region : Region d) → (offset : Op .nat shape) → MemAccess d shape
  | ptr : {d : TileDType} → Op .ptr shape → MemAccess d shape
  | blockPtr : {d : TileDType} → Op .blockPtr shape → (boundaryCheck : List Nat) → MemAccess d shape

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
* `atomicAdd h mem value mask` is a proof-facing read-modify-write marker for
  `tl.atomic_add`. It is restricted by `NumericDType` so non-additive channels
  such as Bool and pointers cannot be represented as atomic additions.
* `atomicRMW op dtype mem input extra mask dest` is the return-valued
  order-sensitive RMW marker for operations such as `tl.atomic_xchg` and
  `tl.atomic_cas`. `dest = some name` records the returned old value; `none`
  is the discarded-return form. Execution semantics are added by the
  concurrency trace layer, not by pretending these operations are commutative.
* `forLoop i n body` runs `body` `n` times, with the scalar register `i` bound
  to the iteration index.
* `forRange i start stop step body` runs `body` for `i = start, start+step,
  start+2*step, …` while `i < stop`. Models Python `for i in range(start, stop, step)`.
* `ifThen cond body` runs `body` when the scalar `cond` evaluates `true`, and
  is a no-op when `false`.
* `ifThenElse cond thenBody elseBody` runs one of two scalar-gated statement
  blocks. This models Triton block-level `if ... else ...` control flow without
  early-exit effects such as `break` or `continue`.
-/
inductive Stmt : Type where
  | assign  : (dtype : TileDType) → (shape : TileShape) → RegName → Op dtype shape → Stmt
  | store   : (dtype : TileDType) → (shape : TileShape) →
              MemAccess dtype shape → (value : Op dtype shape) →
              (mask : MaskOpt dtype shape) → Stmt
  | atomicAdd : NumericDType dtype → (shape : TileShape) →
              MemAccess dtype shape → (value : Op dtype shape) →
              (mask : MaskOpt dtype shape) → Stmt
  | atomicRMW : RMWOp → (dtype : TileDType) → (shape : TileShape) →
              MemAccess dtype shape → (input : Op dtype shape) →
              (extraInput : Option (Op dtype shape)) →
              (mask : MaskOpt dtype shape) → (dest : Option RegName) → Stmt
  | forLoop : (idx : RegName) → (n : Nat) → (body : List Stmt) → Stmt
  | forRange : (idx : RegName) → (start stop step : Nat) → (body : List Stmt) → Stmt
  | ifThen  : (cond : Op .bool []) → (body : List Stmt) → Stmt
  | ifThenElse : (cond : Op .bool []) → (thenBody elseBody : List Stmt) → Stmt

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

/-- Error type for compute-kernel features that cannot be projected into the
algorithm layer. -/
inductive EraseDTypeError where
  | requiresComputeSemantics (op : String)
  | requiresEffectProjection (op : String)
  | unsupportedBitcast (reason : String)
  deriving Repr, BEq

/-! ## Compute-facing dtype/op skeleton -/

structure UInt32Bits where
  bits : BitVec 32
  deriving Repr, BEq, DecidableEq

structure Int32Bits where
  bits : BitVec 32
  deriving Repr, BEq, DecidableEq

structure Float32Bits where
  bits : BitVec 32
  deriving Repr, BEq, DecidableEq

namespace UInt32Bits

def toNat (x : UInt32Bits) : Nat :=
  x.bits.toNat

end UInt32Bits

namespace Int32Bits

def toInt (x : Int32Bits) : Int :=
  let n := x.bits.toNat
  if n < 2^31 then
    Int.ofNat n
  else
    Int.ofNat n - Int.ofNat (2^32)

end Int32Bits

namespace Float32Bits

private def pow2 (n : Nat) : Rat :=
  (2 : Rat) ^ n

private def scalePow2 (q : Rat) (e : Int) : Rat :=
  if _h : 0 ≤ e then
    q * pow2 e.toNat
  else
    q / pow2 (-e).toNat

/--
Computable finite-normal IEEE-754 binary32 decode to `Rat`.

This initial algorithm bridge intentionally supports only finite normal values.
Zeros, subnormals, infinities, and NaNs return `none` until the compute-correct
IEEE model is widened.
-/
def decodeRat (x : Float32Bits) : Option Rat :=
  let n := x.bits.toNat
  let sign := n / 2^31
  let exp := (n / 2^23) % 256
  let frac := n % 2^23
  if exp = 0 ∨ exp = 255 then
    none
  else
    let significand : Rat := (2^23 + frac : Nat)
    let signed := if sign = 0 then significand else -significand
    some (scalePow2 signed (Int.ofNat exp - 150))

noncomputable def decodeReal (x : Float32Bits) : WithBot ℝ :=
  match decodeRat x with
  | some q => some (q : ℝ)
  | none => none

end Float32Bits

inductive ComputeDType where
  | uint32
  | int32
  | fp32
  deriving Repr, BEq, DecidableEq

namespace ComputeDType

def eraseDType : ComputeDType → AlgDType
  | .uint32 => .nat
  | .int32 => .int
  | .fp32 => .real

def width : ComputeDType → Nat
  | .uint32 => 32
  | .int32 => 32
  | .fp32 => 32

end ComputeDType

def ComputeCarrier : ComputeDType → Type
  | .uint32 => UInt32Bits
  | .int32 => Int32Bits
  | .fp32 => Float32Bits

inductive ComputeOp : ComputeDType → TileShape → Type where
  | alg : (dtype : ComputeDType) →
      Op dtype.eraseDType shape →
      ComputeOp dtype shape
  | const : ComputeCarrier dtype → ComputeOp dtype []
  | bitcast :
      (src dst : ComputeDType) →
      src.width = dst.width →
      ComputeOp src shape →
      ComputeOp dst shape

namespace ComputeOp

def bitcastPayload (src dst : ComputeDType) :
    ComputeCarrier src → Option (ComputeCarrier dst) :=
  match src, dst with
  | .uint32, .uint32 => fun x => some x
  | .uint32, .int32 => fun x => some ({ bits := x.bits } : Int32Bits)
  | .uint32, .fp32 => fun x => some ({ bits := x.bits } : Float32Bits)
  | .int32, .uint32 => fun x => some ({ bits := x.bits } : UInt32Bits)
  | .int32, .int32 => fun x => some x
  | .int32, .fp32 => fun x => some ({ bits := x.bits } : Float32Bits)
  | .fp32, .uint32 => fun x => some ({ bits := x.bits } : UInt32Bits)
  | .fp32, .int32 => fun x => some ({ bits := x.bits } : Int32Bits)
  | .fp32, .fp32 => fun x => some x

def constPayload? : ComputeOp dtype [] → Option (ComputeCarrier dtype)
  | .alg _ _ => none
  | .const value => some value
  | .bitcast src dst _ e =>
      match constPayload? e with
      | some value => bitcastPayload src dst value
      | none => none

def constToAlgorithm? :
    (dtype : ComputeDType) → ComputeCarrier dtype →
      Except EraseDTypeError (Op dtype.eraseDType [])
  | .uint32, value => Except.ok (Op.constNat value.toNat)
  | .int32, value => Except.ok (Op.constInt value.toInt)
  | .fp32, value =>
      match Float32Bits.decodeRat value with
      | some q => Except.ok (Op.const (q : ℝ))
      | none => Except.error (.unsupportedBitcast "unsupported fp32 decode")

def constOpToAlgorithm? (op : ComputeOp dtype []) :
    Except EraseDTypeError (Op dtype.eraseDType []) :=
  match constPayload? op with
  | some value => constToAlgorithm? dtype value
  | none => Except.error (.requiresComputeSemantics "runtime bitcast")

def toAlgorithm? : (op : ComputeOp dtype shape) →
    Except EraseDTypeError (Op dtype.eraseDType shape)
  | .alg _ e => Except.ok e
  | .const value => constToAlgorithm? _ value
  | .bitcast src dst h e =>
      match shape with
      | [] => constOpToAlgorithm? (.bitcast src dst h e)
      | _ => Except.error (.requiresComputeSemantics "non-scalar runtime bitcast")

end ComputeOp

inductive ComputeExpr : AlgDType → TileShape → Type where
  | alg : Op dtype shape → ComputeExpr dtype shape
  | compute : ComputeOp cdtype shape → ComputeExpr cdtype.eraseDType shape

namespace ComputeExpr

def toAlgorithm? : ComputeExpr dtype shape → Except EraseDTypeError (Op dtype shape)
  | .alg e => Except.ok e
  | .compute e => ComputeOp.toAlgorithm? e

@[simp] theorem toAlgorithm?_alg (e : Op dtype shape) :
    (ComputeExpr.alg e).toAlgorithm? = Except.ok e := rfl

end ComputeExpr

inductive ComputeStmt : Type where
  | alg : Stmt → ComputeStmt
  | assign : (dtype : AlgDType) → (shape : TileShape) → RegName →
      ComputeExpr dtype shape → ComputeStmt
  | store : (dtype : AlgDType) → (shape : TileShape) →
      MemAccess dtype shape → ComputeExpr dtype shape → MaskOpt dtype shape → ComputeStmt
  | atomicAdd : NumericDType dtype → (shape : TileShape) →
      MemAccess dtype shape → ComputeExpr dtype shape → MaskOpt dtype shape → ComputeStmt
  | atomicRMW : RMWOp → (dtype : AlgDType) → (shape : TileShape) →
      MemAccess dtype shape → ComputeExpr dtype shape →
      Option (ComputeExpr dtype shape) → MaskOpt dtype shape →
      Option RegName → ComputeStmt
  | effectMarker : (op : String) → ComputeStmt
  | forLoop : (idx : RegName) → (n : Nat) → (body : List ComputeStmt) → ComputeStmt
  | forRange : (idx : RegName) → (start stop step : Nat) → (body : List ComputeStmt) → ComputeStmt
  | ifThen : (cond : Op .bool []) → (body : List ComputeStmt) → ComputeStmt
  | ifThenElse : (cond : Op .bool []) →
      (thenBody elseBody : List ComputeStmt) → ComputeStmt

namespace ComputeStmt

mutual

def toAlgorithm? : ComputeStmt → Except EraseDTypeError Stmt
  | .alg st => Except.ok st
  | .assign dtype shape name e => do
      Except.ok (.assign dtype shape name (← e.toAlgorithm?))
  | .store dtype shape mem value mask => do
      Except.ok (.store dtype shape mem (← value.toAlgorithm?) mask)
  | .atomicAdd h shape mem value mask => do
      Except.ok (.atomicAdd h shape mem (← value.toAlgorithm?) mask)
  | .atomicRMW op dtype shape mem input extraInput mask dest => do
      let input' ← input.toAlgorithm?
      let extraInput' ←
        match extraInput with
        | none => Except.ok none
        | some extra => Except.ok (some (← extra.toAlgorithm?))
      Except.ok (.atomicRMW op dtype shape mem input' extraInput' mask dest)
  | .effectMarker op =>
      Except.error (.requiresEffectProjection op)
  | .forLoop idx n body => do
      Except.ok (.forLoop idx n (← listToAlgorithm? body))
  | .forRange idx start stop step body => do
      Except.ok (.forRange idx start stop step (← listToAlgorithm? body))
  | .ifThen cond body => do
      Except.ok (.ifThen cond (← listToAlgorithm? body))
  | .ifThenElse cond thenBody elseBody => do
      Except.ok (.ifThenElse cond (← listToAlgorithm? thenBody) (← listToAlgorithm? elseBody))

def listToAlgorithm? : List ComputeStmt → Except EraseDTypeError (List Stmt)
  | [] => Except.ok []
  | st :: rest => do
      let st' ← toAlgorithm? st
      let rest' ← listToAlgorithm? rest
      Except.ok (st' :: rest')

end

@[simp] theorem toAlgorithm?_alg (st : Stmt) :
    ComputeStmt.toAlgorithm? (ComputeStmt.alg st) = Except.ok st := rfl

@[simp] theorem toAlgorithm?_assign_alg
    (dtype : AlgDType) (shape : TileShape) (name : RegName)
    (e : Op dtype shape) :
    ComputeStmt.toAlgorithm?
        (ComputeStmt.assign dtype shape name (ComputeExpr.alg e)) =
      Except.ok (Stmt.assign dtype shape name e) := rfl

@[simp] theorem toAlgorithm?_store_alg
    (dtype : AlgDType) (shape : TileShape)
    (mem : MemAccess dtype shape) (value : Op dtype shape) (mask : MaskOpt dtype shape) :
    ComputeStmt.toAlgorithm?
        (ComputeStmt.store dtype shape mem (ComputeExpr.alg value) mask) =
      Except.ok (Stmt.store dtype shape mem value mask) := rfl

@[simp] theorem toAlgorithm?_atomicAdd_alg
    {dtype : AlgDType} (h : NumericDType dtype) (shape : TileShape)
    (mem : MemAccess dtype shape) (value : Op dtype shape) (mask : MaskOpt dtype shape) :
    ComputeStmt.toAlgorithm?
        (ComputeStmt.atomicAdd h shape mem (ComputeExpr.alg value) mask) =
      Except.ok (Stmt.atomicAdd h shape mem value mask) := rfl

@[simp] theorem toAlgorithm?_atomicRMW_alg
    (op : RMWOp) (dtype : AlgDType) (shape : TileShape)
    (mem : MemAccess dtype shape) (input : Op dtype shape)
    (extraInput : Option (Op dtype shape)) (mask : MaskOpt dtype shape)
    (dest : Option RegName) :
    ComputeStmt.toAlgorithm?
        (ComputeStmt.atomicRMW op dtype shape mem (ComputeExpr.alg input)
          (extraInput.map ComputeExpr.alg) mask dest) =
      Except.ok (Stmt.atomicRMW op dtype shape mem input extraInput mask dest) := by
  cases extraInput <;> rfl

@[simp] theorem listToAlgorithm?_nil :
    ComputeStmt.listToAlgorithm? [] = Except.ok [] := rfl

@[simp] theorem listToAlgorithm?_cons_alg (st : Stmt) (rest : List ComputeStmt) :
    ComputeStmt.listToAlgorithm? (ComputeStmt.alg st :: rest) =
      match ComputeStmt.listToAlgorithm? rest with
      | Except.ok rest' => Except.ok (st :: rest')
      | Except.error e => Except.error e := by
  change Except.bind (Except.ok st)
      (fun st' => Except.bind (ComputeStmt.listToAlgorithm? rest)
        (fun rest' => Except.ok (st' :: rest'))) =
    match ComputeStmt.listToAlgorithm? rest with
    | Except.ok rest' => Except.ok (st :: rest')
    | Except.error e => Except.error e
  cases ComputeStmt.listToAlgorithm? rest <;> rfl

@[simp] theorem listToAlgorithm?_cons_assign_alg
    (dtype : AlgDType) (shape : TileShape) (name : RegName)
    (e : Op dtype shape) (rest : List ComputeStmt) :
    ComputeStmt.listToAlgorithm?
        (ComputeStmt.assign dtype shape name (ComputeExpr.alg e) :: rest) =
      match ComputeStmt.listToAlgorithm? rest with
      | Except.ok rest' => Except.ok (Stmt.assign dtype shape name e :: rest')
      | Except.error e => Except.error e := by
  change Except.bind (Except.ok (Stmt.assign dtype shape name e))
      (fun st' => Except.bind (ComputeStmt.listToAlgorithm? rest)
        (fun rest' => Except.ok (st' :: rest'))) =
    match ComputeStmt.listToAlgorithm? rest with
    | Except.ok rest' => Except.ok (Stmt.assign dtype shape name e :: rest')
    | Except.error e => Except.error e
  cases ComputeStmt.listToAlgorithm? rest <;> rfl

@[simp] theorem listToAlgorithm?_cons_store_alg
    (dtype : AlgDType) (shape : TileShape)
    (mem : MemAccess dtype shape) (value : Op dtype shape) (mask : MaskOpt dtype shape)
    (rest : List ComputeStmt) :
    ComputeStmt.listToAlgorithm?
        (ComputeStmt.store dtype shape mem (ComputeExpr.alg value) mask :: rest) =
      match ComputeStmt.listToAlgorithm? rest with
      | Except.ok rest' => Except.ok (Stmt.store dtype shape mem value mask :: rest')
      | Except.error e => Except.error e := by
  change Except.bind (Except.ok (Stmt.store dtype shape mem value mask))
      (fun st' => Except.bind (ComputeStmt.listToAlgorithm? rest)
        (fun rest' => Except.ok (st' :: rest'))) =
    match ComputeStmt.listToAlgorithm? rest with
    | Except.ok rest' => Except.ok (Stmt.store dtype shape mem value mask :: rest')
    | Except.error e => Except.error e
  cases ComputeStmt.listToAlgorithm? rest <;> rfl

@[simp] theorem listToAlgorithm?_cons_atomicAdd_alg
    {dtype : AlgDType} (h : NumericDType dtype) (shape : TileShape)
    (mem : MemAccess dtype shape) (value : Op dtype shape) (mask : MaskOpt dtype shape)
    (rest : List ComputeStmt) :
    ComputeStmt.listToAlgorithm?
        (ComputeStmt.atomicAdd h shape mem (ComputeExpr.alg value) mask :: rest) =
      match ComputeStmt.listToAlgorithm? rest with
      | Except.ok rest' => Except.ok (Stmt.atomicAdd h shape mem value mask :: rest')
      | Except.error e => Except.error e := by
  change Except.bind (Except.ok (Stmt.atomicAdd h shape mem value mask))
      (fun st' => Except.bind (ComputeStmt.listToAlgorithm? rest)
        (fun rest' => Except.ok (st' :: rest'))) =
    match ComputeStmt.listToAlgorithm? rest with
    | Except.ok rest' => Except.ok (Stmt.atomicAdd h shape mem value mask :: rest')
    | Except.error e => Except.error e
  cases ComputeStmt.listToAlgorithm? rest <;> rfl

@[simp] theorem listToAlgorithm?_cons_atomicRMW_alg
    (op : RMWOp) (dtype : AlgDType) (shape : TileShape)
    (mem : MemAccess dtype shape) (input : Op dtype shape)
    (extraInput : Option (Op dtype shape)) (mask : MaskOpt dtype shape)
    (dest : Option RegName) (rest : List ComputeStmt) :
    ComputeStmt.listToAlgorithm?
        (ComputeStmt.atomicRMW op dtype shape mem (ComputeExpr.alg input)
          (extraInput.map ComputeExpr.alg) mask dest :: rest) =
      match ComputeStmt.listToAlgorithm? rest with
      | Except.ok rest' =>
          Except.ok (Stmt.atomicRMW op dtype shape mem input extraInput mask dest :: rest')
      | Except.error e => Except.error e := by
  cases extraInput <;>
    change Except.bind
      (Except.ok (Stmt.atomicRMW op dtype shape mem input _ mask dest))
      (fun st' => Except.bind (ComputeStmt.listToAlgorithm? rest)
        (fun rest' => Except.ok (st' :: rest'))) =
        match ComputeStmt.listToAlgorithm? rest with
        | Except.ok rest' =>
            Except.ok (Stmt.atomicRMW op dtype shape mem input _ mask dest :: rest')
        | Except.error e => Except.error e
    <;> cases ComputeStmt.listToAlgorithm? rest <;> rfl

@[simp] theorem listToAlgorithm?_cons_ifThen
    (cond : Op .bool []) (body : List ComputeStmt) (rest : List ComputeStmt) :
    ComputeStmt.listToAlgorithm? (ComputeStmt.ifThen cond body :: rest) =
      match ComputeStmt.listToAlgorithm? body, ComputeStmt.listToAlgorithm? rest with
      | Except.ok body', Except.ok rest' => Except.ok (Stmt.ifThen cond body' :: rest')
      | Except.error e, _ => Except.error e
      | _, Except.error e => Except.error e := by
  simp only [ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?]
  cases ComputeStmt.listToAlgorithm? body <;>
    cases ComputeStmt.listToAlgorithm? rest <;> rfl

@[simp] theorem listToAlgorithm?_cons_ifThenElse
    (cond : Op .bool []) (thenBody elseBody : List ComputeStmt)
    (rest : List ComputeStmt) :
    ComputeStmt.listToAlgorithm? (ComputeStmt.ifThenElse cond thenBody elseBody :: rest) =
      match ComputeStmt.listToAlgorithm? thenBody,
            ComputeStmt.listToAlgorithm? elseBody,
            ComputeStmt.listToAlgorithm? rest with
      | Except.ok thenBody', Except.ok elseBody', Except.ok rest' =>
          Except.ok (Stmt.ifThenElse cond thenBody' elseBody' :: rest')
      | Except.error e, _, _ => Except.error e
      | _, Except.error e, _ => Except.error e
      | _, _, Except.error e => Except.error e := by
  simp only [ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?]
  cases ComputeStmt.listToAlgorithm? thenBody <;>
      cases ComputeStmt.listToAlgorithm? elseBody <;>
      cases ComputeStmt.listToAlgorithm? rest <;> rfl

@[simp] theorem listToAlgorithm?_cons_forLoop
    (idx : RegName) (n : Nat) (body : List ComputeStmt) (rest : List ComputeStmt) :
    ComputeStmt.listToAlgorithm? (ComputeStmt.forLoop idx n body :: rest) =
      match ComputeStmt.listToAlgorithm? body, ComputeStmt.listToAlgorithm? rest with
      | Except.ok body', Except.ok rest' => Except.ok (Stmt.forLoop idx n body' :: rest')
      | Except.error e, _ => Except.error e
      | _, Except.error e => Except.error e := by
  simp only [ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?]
  cases ComputeStmt.listToAlgorithm? body <;>
    cases ComputeStmt.listToAlgorithm? rest <;> rfl

@[simp] theorem listToAlgorithm?_cons_forRange
    (idx : RegName) (start stop step : Nat)
    (body : List ComputeStmt) (rest : List ComputeStmt) :
    ComputeStmt.listToAlgorithm?
        (ComputeStmt.forRange idx start stop step body :: rest) =
      match ComputeStmt.listToAlgorithm? body, ComputeStmt.listToAlgorithm? rest with
      | Except.ok body', Except.ok rest' =>
          Except.ok (Stmt.forRange idx start stop step body' :: rest')
      | Except.error e, _ => Except.error e
      | _, Except.error e => Except.error e := by
  simp only [ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?]
  cases ComputeStmt.listToAlgorithm? body <;>
    cases ComputeStmt.listToAlgorithm? rest <;> rfl

@[simp] theorem listToAlgorithm?_append :
    ∀ xs ys : List ComputeStmt,
      ComputeStmt.listToAlgorithm? (xs ++ ys) =
        match ComputeStmt.listToAlgorithm? xs, ComputeStmt.listToAlgorithm? ys with
        | Except.ok xs', Except.ok ys' => Except.ok (xs' ++ ys')
        | Except.error e, _ => Except.error e
        | _, Except.error e => Except.error e
  | [], ys => by
      cases hys : ComputeStmt.listToAlgorithm? ys <;> simp [hys]
  | x :: xs, ys => by
      cases hx : ComputeStmt.toAlgorithm? x <;> simp [ComputeStmt.listToAlgorithm?, hx]
      · rfl
      · rw [listToAlgorithm?_append xs ys]
        cases ComputeStmt.listToAlgorithm? xs <;>
          cases ComputeStmt.listToAlgorithm? ys <;> rfl

@[simp] theorem listToAlgorithm?_map_alg :
    ∀ body : List Stmt,
      ComputeStmt.listToAlgorithm? (body.map ComputeStmt.alg) = Except.ok body
  | [] => rfl
  | st :: rest => by
      change (do
          let st' ← ComputeStmt.toAlgorithm? (ComputeStmt.alg st)
          let rest' ← ComputeStmt.listToAlgorithm? (rest.map ComputeStmt.alg)
          Except.ok (st' :: rest')) = Except.ok (st :: rest)
      rw [toAlgorithm?_alg, listToAlgorithm?_map_alg rest]
      rfl

end ComputeStmt

/--
Compute-facing kernel surface.

Pure algorithm statements are represented as `ComputeStmt.alg`; compute-only
statements use the other `ComputeStmt` constructors. Every kernel projects
through the same `ComputeStmt.listToAlgorithm?` traversal.
-/
inductive ComputeKernel where
  | mk : (inputs outputs : List RegionName) → (body : List ComputeStmt) → ComputeKernel
  deriving Inhabited

namespace ComputeKernel

/-- Build a compute-facing kernel from an already-constructed algorithm body. -/
def fromAlgBody (inputs outputs : List RegionName) (body : List Stmt) : ComputeKernel :=
  ComputeKernel.mk inputs outputs (body.map ComputeStmt.alg)

/-- Fallible projection from the compute-facing kernel surface to the algorithm
kernel surface. -/
def toAlgorithm? : ComputeKernel → Except EraseDTypeError AlgKernel
  | .mk inputs outputs body => do
      Except.ok (Kernel.mk inputs outputs (← ComputeStmt.listToAlgorithm? body))

@[simp] theorem toAlgorithm?_mk
    (inputs outputs : List RegionName) (body : List ComputeStmt) :
    (ComputeKernel.mk inputs outputs body).toAlgorithm? =
      match ComputeStmt.listToAlgorithm? body with
      | Except.ok body' => Except.ok (Kernel.mk inputs outputs body')
      | Except.error e => Except.error e := by
  change Except.bind (ComputeStmt.listToAlgorithm? body)
      (fun body' => Except.ok (Kernel.mk inputs outputs body')) =
    match ComputeStmt.listToAlgorithm? body with
    | Except.ok body' => Except.ok (Kernel.mk inputs outputs body')
    | Except.error e => Except.error e
  cases ComputeStmt.listToAlgorithm? body <;> rfl

@[simp] theorem toAlgorithm?_mk_map_alg
    (inputs outputs : List RegionName) (body : List Stmt) :
    (ComputeKernel.mk inputs outputs (body.map ComputeStmt.alg)).toAlgorithm? =
      Except.ok (Kernel.mk inputs outputs body) := by
  change (do
      let body' ← ComputeStmt.listToAlgorithm? (body.map ComputeStmt.alg)
      Except.ok (Kernel.mk inputs outputs body')) =
    Except.ok (Kernel.mk inputs outputs body)
  rw [ComputeStmt.listToAlgorithm?_map_alg]
  rfl

@[simp] theorem toAlgorithm?_fromAlgBody
    (inputs outputs : List RegionName) (body : List Stmt) :
    (ComputeKernel.fromAlgBody inputs outputs body).toAlgorithm? =
      Except.ok (Kernel.mk inputs outputs body) := by
  simp [ComputeKernel.fromAlgBody]

/--
Legacy coercion used by existing examples whose definitions are still annotated
as `Kernel`. New proof-facing code should use `ComputeKernel.toAlgorithm?` or
`ComputeKernel.ComputeCorrect` so failed compute projection is explicit.
-/
def toAlgKernel (ck : ComputeKernel) : AlgKernel :=
  match ck.toAlgorithm? with
  | Except.ok k => k
  | Except.error _ => default

@[simp] theorem toAlgKernel_mk
    (inputs outputs : List RegionName) (body : List ComputeStmt) :
    (ComputeKernel.mk inputs outputs body).toAlgKernel =
      match ComputeStmt.listToAlgorithm? body with
      | Except.ok body' => Kernel.mk inputs outputs body'
      | Except.error _ => default := by
  change
    (match Except.bind (ComputeStmt.listToAlgorithm? body)
        (fun body' => Except.ok (Kernel.mk inputs outputs body')) with
      | Except.ok k => k
      | Except.error _ => default) =
    match ComputeStmt.listToAlgorithm? body with
    | Except.ok body' => Kernel.mk inputs outputs body'
    | Except.error _ => default
  cases ComputeStmt.listToAlgorithm? body <;> rfl

@[simp] theorem toAlgKernel_mk_map_alg
    (inputs outputs : List RegionName) (body : List Stmt) :
    (ComputeKernel.mk inputs outputs (body.map ComputeStmt.alg)).toAlgKernel =
      Kernel.mk inputs outputs body := by
  simp [toAlgKernel]

@[simp] theorem toAlgKernel_fromAlgBody
    (inputs outputs : List RegionName) (body : List Stmt) :
    (ComputeKernel.fromAlgBody inputs outputs body).toAlgKernel =
      Kernel.mk inputs outputs body := by
  simp [ComputeKernel.fromAlgBody]

instance : Coe ComputeKernel AlgKernel where
  coe := toAlgKernel

/-- Projected algorithm inputs for compatibility with existing proof scripts. -/
abbrev inputs (ck : ComputeKernel) : List RegionName :=
  ck.toAlgKernel.inputs

/-- Projected algorithm outputs for compatibility with existing proof scripts. -/
abbrev outputs (ck : ComputeKernel) : List RegionName :=
  ck.toAlgKernel.outputs

/-- Projected algorithm body for compatibility with existing proof scripts. -/
abbrev body (ck : ComputeKernel) : List Stmt :=
  ck.toAlgKernel.body

@[simp] theorem body_mk
    (inputs outputs : List RegionName) (body : List ComputeStmt) :
    (ComputeKernel.mk inputs outputs body).body =
      match ComputeStmt.listToAlgorithm? body with
      | Except.ok body' => body'
      | Except.error _ => (default : AlgKernel).body := by
  cases h : ComputeStmt.listToAlgorithm? body <;>
    simp [ComputeKernel.body, h]

@[simp] theorem body_mk_map_alg
    (inputs outputs : List RegionName) (body : List Stmt) :
    (ComputeKernel.mk inputs outputs (body.map ComputeStmt.alg)).body = body := by
  simp [ComputeKernel.body]

@[simp] theorem body_fromAlgBody
    (inputs outputs : List RegionName) (body : List Stmt) :
    (ComputeKernel.fromAlgBody inputs outputs body).body = body := by
  simp [ComputeKernel.fromAlgBody]

@[simp] theorem toAlgKernel_body_mk_map_alg
    (inputs outputs : List RegionName) (body : List Stmt) :
    (ComputeKernel.mk inputs outputs (body.map ComputeStmt.alg)).toAlgKernel.body =
      body := by
  exact body_mk_map_alg inputs outputs body

end ComputeKernel

end VeriTile.Triton
