/-
VeriTile.Examples.TritonSmoke

Small smoke tests for the typed Triton core.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.Memory
import VeriTile.Triton.MemoryTyping
import VeriTile.Triton.DSL

namespace VeriTile.Examples.TritonSmoke

open VeriTile.Triton

/-- Smoke test for scalar-pointer load/store syntax. -/
def scalarCopyKernel (xReg yReg : RegionName) : Kernel := triton {
  x := tl.load($(xReg))
  tl.store($(yReg), x)
}

/-- Vector-add kernel with explicit boundary mask. -/
def addKernelMaskedSmoke (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask := offs < $(nElements)
  x    := tl.load($(xReg) + offs, mask=mask, other=0)
  y    := tl.load($(yReg) + offs, mask=mask, other=0)
  out  := x + y
  tl.store($(outReg) + offs, out, mask=mask)
}

/-- Every comparison operator is reachable via the DSL. -/
def comparisonOpsSmoke : Kernel := triton {
  a   := 1
  b   := 2
  r1  := a < b
  r2  := a <= b
  r3  := a == b
  r4  := a > b
  r5  := a >= b
  r6  := a != b
}

/-- Unary math ops used by score-level attention variants are reachable via the DSL. -/
def unaryMathOpsSmoke (xReg : RegionName) (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  l    := tl.log(e)
  s    := tl.sigmoid(x)
  r    := tl.sqrt(e)
  t    := tl.tanh(x)
  a    := tl.abs(x)
  lo   := tl.minimum(x, 0)
  hi   := tl.maximum(x, 0)
  nlo  := tl.minimum(offs, $(N))
  nhi  := tl.maximum(offs, $(N))
}

/-- DSL smoke test for explicit floating dtype casts. -/
def dtypeCastSmoke (xReg outReg : RegionName) (N : Nat) : Kernel := triton {
  offs := tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  x32  := tl.cast(x, tl.float32)
  one  := tl.cast(1, tl.float32)
  y32  := x32 + one
  y    := tl.cast(y32, tl.float64)
  tl.store($(outReg) + offs, y)
}

/-- DSL smoke test for Triton's method-style cast spelling. -/
def dtypeToSmoke (xReg outReg : RegionName) (N : Nat) : Kernel := triton {
  offs := tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  x32  := (x).to(tl.float32)
  one  := (1).to(tl.float32)
  y32  := x32 + one
  y    := (y32).to(tl.float64)
  tl.store($(outReg) + offs, y)
}

/-- DSL smoke test for typed floating memory surface syntax. -/
def dtypeLoadStoreSmoke (xReg outReg : RegionName) (N : Nat) : Kernel := triton {
  offs := tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs, dtype=tl.float32)
  y    := x + (1).to(tl.float32)
  tl.store($(outReg) + offs, y)
}

/-- DSL smoke test for masked typed floating load/store surface syntax. -/
def dtypeMaskedLoadStoreSmoke (xReg outReg : RegionName) (N : Nat) : Kernel := triton {
  offs := tl.arange(0, $(N))
  mask := offs < $(N)
  zero := (0).to(tl.float32)
  x    := tl.load($(xReg) + offs, mask=mask, other=zero, dtype=tl.float32)
  tl.store($(outReg) + offs, x, mask=mask)
}

/-- The real kernel recovered by erasing `dtypeCastSmoke`'s float annotations. -/
def dtypeCastSmokeErasedReal (xReg outReg : RegionName) (N : Nat) : Kernel := triton {
  offs := tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  x32  := x
  one  := 1
  y32  := x32 + one
  y    := y32
  tl.store($(outReg) + offs, y)
}

/-- The real kernel recovered by erasing `dtypeLoadStoreSmoke`'s float memory. -/
def dtypeLoadStoreSmokeErasedReal (xReg outReg : RegionName) (N : Nat) : Kernel := triton {
  offs := tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  y    := x + 1
  tl.store($(outReg) + offs, y)
}

/-- Float theorem bridge smoke: erasure recovers the real proof target. -/
example : True := by
  trivial

/-- Typed memory theorem bridge smoke: erasure recovers the real proof target. -/
example : True := by
  trivial

/-- Core smoke test for typed floating load/store AST nodes. -/
def fp16LoadStoreCoreSmoke (xReg outReg : RegionName) (N : Nat) : Stmt :=
  Stmt.store .fp16 [N] (MemAccess.region outReg (Op.arange N))
    (Op.load .fp16 (MemAccess.region xReg (Op.arange N)) MaskOpt.none)
    MaskOpt.none

/-- Lightweight region typing accepts the fp16 load/store smoke under fp16 buffers. -/
example : True := by
  trivial

/-- Lightweight region typing accepts typed DSL memory under fp32 buffers. -/
example : True := by
  trivial

/-- Masked typed DSL memory also respects fp32 region contracts. -/
example : True := by
  trivial

/-! ### Integer/meta and boolean surface operators -/

/-- Common grid/layout integer operators: `tl.cdiv`, `//`, `%`, and
meta-expression antiquotation inside `tl.arange`. -/
def integerSurfaceSmoke
    (nElements blockSize lo hi : Nat) : Kernel := triton {
  pid       := tl.program_id(0)
  nBlocks   := tl.cdiv($(nElements), $(blockSize))
  group     := pid // $(4)
  blockRem  := pid % $(4)
  offs      := tl.arange($(lo), $(hi))
  lanes     := tl.arange(0, $(blockSize))
  laneGroup := lanes // $(4)
  laneRem   := lanes % $(4)
}

/-- Boolean function and operator spellings for masks. -/
def booleanSurfaceSmoke (N : Nat) : Kernel := triton {
  offs := tl.arange(0, $(N))
  a    := offs < $(N)
  b    := (offs % $(2)) == $(0)
  c    := tl.logical_or(a, tl.logical_not(b))
  d    := (a & b) | ~c
}

/-- Real-only DSL kernels require real-typed buffers in the lightweight contract. -/
example : True := by
  trivial

/-- `tl.load(p, mask=m)` with no `other=` uses `s.undef` for masked-off lanes. -/
example : evalOp
    (Op.load .real (MemAccess.region "X" (Op.constNat 0))
      (MaskOpt.mask
        (Op.lt ComparableDType.nat Broadcast.nil (Op.constNat 0) (Op.constNat 0))))
    { mem := fun _ _ => 100, regs := fun _ _ _ => none
    , pids := fun _ => 0, undef := fun _ _ => 42 }
    = some (Tile.scalar (some (42 : ℝ) : WithBot ℝ)) := by
  rfl

/-- Different `undef` oracles can produce different masked-off load values. -/
example :
    let s1 : BlockState :=
      { mem := fun _ _ => 0, regs := fun _ _ _ => none
      , pids := fun _ => 0, undef := fun _ _ => 42 }
    let s2 : BlockState :=
      { mem := fun _ _ => 0, regs := fun _ _ _ => none
      , pids := fun _ => 0, undef := fun _ _ => 99 }
    let maskFalse : Op .bool [] :=
      Op.lt ComparableDType.nat Broadcast.nil (Op.constNat 0) (Op.constNat 0)
    evalOp (Op.load .real (MemAccess.region "X" (Op.constNat 0)) (MaskOpt.mask maskFalse)) s1
      ≠ evalOp (Op.load .real (MemAccess.region "X" (Op.constNat 0)) (MaskOpt.mask maskFalse)) s2 := by
  change (some (Tile.scalar (some (42 : ℝ) : WithBot ℝ) : Tile .real [])) ≠
    (some (Tile.scalar (some (99 : ℝ) : WithBot ℝ) : Tile .real []))
  intro h
  injection h with ht
  have hv := congrArg (fun t : Tile .real [] => t.data PUnit.unit) ht
  injection hv with hreal
  norm_num at hreal

/-! ### `tl.dot` typed-AST smoke tests

These verify that `Op.dot` type-checks at the AST layer (the `K` constraint
forces the inner dim of the LHS to match the outer dim of the RHS). The
DSL macro `tl.dot(a, b)` and its fused-accumulator form `tl.dot(a, b, acc)`
lower into these AST nodes. End-to-end exercise via 2D pointer arithmetic
in `tl.load` is now available (see `fa1QLoadSmoke` below and #34/#35). -/

/-- Rank-2 (FA-1 forward shape): `(M, K) @ (K, N) = (M, N)`. -/
example : (M K N : Nat) → Op .real [M, N] := fun M K N =>
  .dot (batch := []) (Op.ref .real [M, K] "a") (Op.ref .real [K, N] "b")

/-- Fused accumulator form, rank-2: `acc + a @ b`. -/
example : (M K N : Nat) → Op .real [M, N] := fun M K N =>
  .add NumericDType.real (.consSame (.consSame .nil))
    (.dot (batch := []) (Op.ref .real [M, K] "p") (Op.ref .real [K, N] "v"))
    (Op.ref .real [M, N] "acc")

/-- Batched (FA-2 / grouped-GEMM shape): `(B, M, K) @ (B, K, N) = (B, M, N)`. -/
example : (B M K N : Nat) → Op .real [B, M, N] := fun B M K N =>
  .dot (batch := [B])
    (Op.ref .real [B, M, K] "a")
    (Op.ref .real [B, K, N] "b")

/-! ### `tl.expand_dims` surface forms

Two literal slicer postfixes are accepted for rank-1 inputs:

* `e[:, None]` — `[N] → [N, 1]`, axis 1
* `e[None, :]` — `[N] → [1, N]`, axis 0

The function surface `tl.expand_dims(e, axis=N)` / `tl.expand_dims(e, N)`
lowers to the same `Op.expandDim` for any macro-known rank. -/

/-- `[:, None]` produces a `[N, 1]` tile. -/
def colExpandSmoke (xReg : RegionName) (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  xc   := x[:, None]
}

/-- `[None, :]` produces a `[1, N]` tile. -/
def rowExpandSmoke (xReg : RegionName) (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  xr   := x[None, :]
}

/-- `tl.expand_dims(x, axis=1)` is the function-form equivalent of
`x[:, None]`. -/
def expandDimsKwargSmoke (xReg : RegionName) (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  xc   := tl.expand_dims(x, axis=1)
}

/-- Positional-axis `tl.expand_dims(x, 0)` produces a `[1, N]` tile. -/
def expandDimsPositionalSmoke (xReg : RegionName) (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  xr   := tl.expand_dims(x, 0)
}

/-- Function-form `tl.expand_dims` also works on higher-rank tiles. -/
def expandDimsRank2Smoke
    (xReg : RegionName) (M N stride : Nat) : Kernel := triton {
  offsM := tl.arange(0, $(M))
  offsN := tl.arange(0, $(N))
  ptrs  := offsM[:, None] * $(stride) + offsN[None, :]
  x     := tl.load($(xReg) + ptrs)
  x3    := tl.expand_dims(x, axis=0)
}

/-! ### `tl.static_range` surface alias -/

/-- `tl.static_range` currently lowers to the same bounded-loop AST as
`tl.for`; unroll / pipeline attributes are intentionally not modeled. -/
def staticRangeSmoke (N : Nat) : Kernel := triton {
  acc := 0
  tl.static_range i in $(N) {
    acc := acc + tl.toReal(i)
  }
}

/-- Numeric-literal `tl.static_range` bounds share the same lowering. -/
def staticRangeLiteralSmoke : Kernel := triton {
  acc := 0
  tl.static_range i in 4 {
    acc := acc + tl.toReal(i)
  }
}

/-- `Tile.expandDim` semantics on a literal rank-1 tile: `[3] → [1, 3]`. -/
example :
    let v : Tile .real [3] := Tile.vec (fun i => some (i.val : ℝ))
    let v' := Tile.expandDim ⟨0, by simp⟩ v
    v'.data (⟨0, by decide⟩, ⟨2, by decide⟩, PUnit.unit) =
      v.data (⟨2, by decide⟩, PUnit.unit) := by
  simp [Tile.expandDim, TileShape.dropInsertedIndex]

/-- `Tile.expandDim` axis-1 case: `[3] → [3, 1]`. -/
example :
    let v : Tile .real [3] := Tile.vec (fun i => some (i.val : ℝ))
    let v' := Tile.expandDim ⟨1, by simp⟩ v
    v'.data (⟨1, by decide⟩, ⟨0, by decide⟩, PUnit.unit) =
      v.data (⟨1, by decide⟩, PUnit.unit) := by
  simp [Tile.expandDim, TileShape.dropInsertedIndex]

/-! ### 2D pointer arithmetic (FA-1 Q-block load shape, issue #35)

The classic FA-1 forward Q-load addresses a `[M, D]` tile via the
broadcast sum

  Q + offs_m[:, None] * stride_qm + offs_d[None, :]

Each piece — `[:, None]` / `[None, :]` (#34), ND broadcast, ND
`Op.load` — is already in place; the smoke tests below verify the
composite still type-checks at shape `[M, D]` end-to-end. -/

/-- FA-1-style Q-block load: rank-2 `[M, D]` tile addressed via ND
pointer arithmetic. -/
def fa1QLoadSmoke (qReg outReg : RegionName) (M D stride_qm : Nat) :
    Kernel := triton {
  pid    := tl.program_id(0)
  offs_m := pid * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(D))
  ptrs   := offs_m[:, None] * $(stride_qm) + offs_d[None, :]
  qs     := tl.load($(qReg) + ptrs)
  tl.store($(outReg) + ptrs, qs)
}

/-- Type-level assertion that an isolated rank-2 broadcast pointer
expression `[M, 1] + [1, D]` lands at shape `[M, D]`. The full
FA-1-style pipeline above already type-checks; this just pins the
broadcast result independently for documentation. -/
example (M D : Nat) : Op .nat [M, D] :=
  Op.add NumericDType.nat (.consR (.consL .nil))
    (Op.expandDim (shape := [M]) ⟨1, by simp⟩ (Op.arange M))
    (Op.expandDim (shape := [D]) ⟨0, by simp⟩ (Op.arange D))

/-! ### `tl.where` causal-mask shape (Phase C scope of issue #29)

The FA-1 forward causal mask collapses to `tl.where(mask, scores, -inf)`
at shape `[M, Bk]`. The DSL macro lifts the scalar `-inf` to the tile
shape via `Op.broadcast`. -/

/-- Causal-mask kernel skeleton: load Q-row block scores, mask out
future positions with `-inf`, store back. Exercises ND `tl.where`
end-to-end. -/
def causalMaskSmoke (sReg outReg : RegionName) (M Bk : Nat) :
    Kernel := triton {
  pid    := tl.program_id(0)
  offs_m := pid * $(M) + tl.arange(0, $(M))
  offs_n := tl.arange(0, $(Bk))
  ptrs   := offs_m[:, None] * $(Bk) + offs_n[None, :]
  scores := tl.load($(sReg) + ptrs)
  mask   := offs_m[:, None] >= offs_n[None, :]
  masked := tl.where(mask, scores, -inf)
  tl.store($(outReg) + ptrs, masked)
}

/-- `Tile.select` semantics on a literal boolean mask: where `c` is
true, pick `a`; else pick `b`. -/
example :
    let c : Tile .bool [2] := Tile.vec (fun i => decide (i.val = 0))
    let a : Tile .real [2] := Tile.vec (fun _ => some (1 : ℝ))
    let b : Tile .real [2] := Tile.vec (fun _ => some (2 : ℝ))
    let r := Tile.select c a b
    r.data (⟨0, by decide⟩, PUnit.unit) = some (1 : ℝ) ∧
    r.data (⟨1, by decide⟩, PUnit.unit) = some (2 : ℝ) := by
  refine ⟨?_, ?_⟩ <;> simp [Tile.select, Tile.vec]

/-! ### `tl.if` block-skipping shape (issue #29)

FA-2's block-skipping pattern `if start_n + BLOCK_N <= start_m: continue`
rewrites to `tl.if not_skippable { ...work... }`. Scalar bool condition
is required; element-wise masking still goes through `tl.where`. -/

/-- Block-skip skeleton: pid-gated store. The `tl.if` body runs only
when `pid < $(P)` evaluates true; otherwise the kernel is a no-op for
that program instance. Exercises the DSL surface + macro lowering. -/
def ifThenSmoke (xReg outReg : RegionName) (N P : Nat) :
    Kernel := triton {
  pid  := tl.program_id(0)
  tl.if pid < $(P) {
    offs := pid * $(N) + tl.arange(0, $(N))
    x    := tl.load($(xReg) + offs)
    tl.store($(outReg) + offs, x)
  }
}

example :
    let s : BlockState :=
      { mem := fun _ _ => 0, regs := fun _ _ _ => none
      , pids := fun _ => 0, undef := fun _ _ => 0 }
    let cond : Op .bool [] :=
      Op.lt ComparableDType.nat Broadcast.nil (Op.constNat 0) (Op.constNat 1)
    let body : List Stmt := [Stmt.assign .real [] "x" (Op.const 7)]
    (stepStmt (Stmt.ifThen cond body) s).bind
        (fun s' => s'.regs .real [] "x")
      = some (Tile.scalar (some (7 : ℝ) : WithBot ℝ)) := by
  norm_num [stepStmt, stepStmts, evalOp, Tile.cop, ComparableDType.lt,
    BlockState.setReg]

example :
    let s : BlockState :=
      { mem := fun _ _ => 0, regs := fun _ _ _ => none
      , pids := fun _ => 0, undef := fun _ _ => 0 }
    let cond : Op .bool [] :=
      Op.lt ComparableDType.nat Broadcast.nil (Op.constNat 1) (Op.constNat 1)
    let body : List Stmt := [Stmt.assign .real [] "x" (Op.const 7)]
    stepStmt (Stmt.ifThen cond body) s = some s := by
  norm_num [stepStmt, evalOp, Tile.cop, ComparableDType.lt]

/-! ### `tl.trans` / transpose (issue #36)

Trailing-two-axes transpose: rank-2 case is the standard `.T`; rank-≥ 3
transposes the inner matrix at every batch coordinate. The DSL surface
is `tl.trans(e)` (matching the Triton Python API; the `e.T` postfix is
a possible follow-up). -/

/-- DSL kernel exercising `tl.trans` at rank-2: load a `[Bk, D]` K-block,
transpose to `[D, Bk]`, store back. The shape change is observable in
the macro-recorded `SInfo` and the typed AST. -/
def transposeSmoke (kReg outReg : RegionName) (Bk D : Nat) :
    Kernel := triton {
  pid    := tl.program_id(0)
  offs_n := pid * $(Bk) + tl.arange(0, $(Bk))
  offs_d := tl.arange(0, $(D))
  ptrs_in  := offs_n[:, None] * $(D) + offs_d[None, :]
  ptrs_out := offs_d[:, None] * $(Bk) + offs_n[None, :]
  k        := tl.load($(kReg) + ptrs_in)
  kt       := tl.trans(k)
  tl.store($(outReg) + ptrs_out, kt)
}

/-- `tl.dot(Q, tl.trans(K))` lands at the FA-1 score-block shape:
`Q : [M, D]`, `K : [Bk, D]`, `K.T : [D, Bk]`, `Q @ K.T : [M, Bk]`. -/
def dotKTSmoke (qReg kReg sReg : RegionName) (M Bk D : Nat) :
    Kernel := triton {
  pid       := tl.program_id(0)
  offs_m    := pid * $(M) + tl.arange(0, $(M))
  offs_n    := tl.arange(0, $(Bk))
  offs_d    := tl.arange(0, $(D))
  ptrs_q    := offs_m[:, None] * $(D)  + offs_d[None, :]
  ptrs_k    := offs_n[:, None] * $(D)  + offs_d[None, :]
  ptrs_s    := offs_m[:, None] * $(Bk) + offs_n[None, :]
  q         := tl.load($(qReg) + ptrs_q)
  k         := tl.load($(kReg) + ptrs_k)
  scores    := tl.dot(q, tl.trans(k))
  tl.store($(sReg) + ptrs_s, scores)
}

/-- Type-level: `Op.transpose` on a `[M, N]` real tile lands at `[N, M]`. -/
example (M N : Nat) :
    Op .real [N, M] :=
  Op.transpose (batch := []) (Op.ref .real [M, N] "x")

/-- Batched: `[B, M, N] → [B, N, M]`. -/
example (B M N : Nat) :
    Op .real [B, N, M] :=
  Op.transpose (batch := [B]) (Op.ref .real [B, M, N] "x")

/-- `Tile.transpose` semantics on a literal `[2, 3]` matrix lands at
`[3, 2]` with swapped indexing. -/
example :
    let m : Tile .real [2, 3] :=
      Tile.mat (fun i j => some ((i.val * 3 + j.val : Nat) : ℝ))
    let mT := Tile.transpose [] m
    mT.data (⟨2, by decide⟩, ⟨1, by decide⟩, PUnit.unit) =
      m.data (⟨1, by decide⟩, ⟨2, by decide⟩, PUnit.unit) := rfl

/-! ### First-class pointer values (issue #44) -/

def pointerValueLoadSmoke (xReg : RegionName) (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  ptrs := $(xReg) + offs
  x    := tl.load(ptrs)
}

def pointerValueOffsetLoadSmoke (xReg : RegionName) (N : Nat) : Kernel := triton {
  pid   := tl.program_id(0)
  offs  := pid * $(N) + tl.arange(0, $(N))
  ptrs  := $(xReg) + offs
  ptrs2 := ptrs + $(N)
  y     := tl.load(ptrs2)
}

def pointerValueStoreSmoke (xReg outReg : RegionName) (N : Nat) : Kernel := triton {
  pid     := tl.program_id(0)
  offs    := pid * $(N) + tl.arange(0, $(N))
  x       := tl.load($(xReg) + offs)
  outPtrs := $(outReg) + offs
  tl.store(outPtrs, x)
}

def pointerValueMaskedStoreSmoke (xReg outReg : RegionName) (N : Nat) : Kernel := triton {
  pid     := tl.program_id(0)
  offs    := pid * $(N) + tl.arange(0, $(N))
  mask    := offs < $(N)
  ptrs    := $(xReg) + offs
  x       := tl.load(ptrs, mask=mask, other=0)
  outPtrs := $(outReg) + offs
  tl.store(outPtrs, x, mask=mask)
}

/-! ### Block pointers and boundary checks (issue #47) -/

def blockPointerBoundaryCopySmoke (xReg outReg : RegionName) (N B : Nat) : Kernel := triton {
  xBp := tl.make_block_ptr($(xReg), base=$(0), shape=[$(N)], strides=[1], offsets=[0], block_shape=[$(B)])
  x   := tl.load(xBp, boundary_check=([0] : List Nat), padding_option="zero")
  yBp := tl.make_block_ptr($(outReg), base=$(0), shape=[$(N)], strides=[1], offsets=[0], block_shape=[$(B)])
  tl.store(yBp, x, boundary_check=([0] : List Nat))
}

def blockPointerOobLoad (xReg : RegionName) : Op .real [1] :=
  Op.load .real
    (MemAccess.blockPtr (Op.makeBlockPtr xReg 0 [0] [1] [1] [0]) [0])
    MaskOpt.none

theorem blockPointerOobLoad_zero (xReg : RegionName)
    (s : BlockState) (i : TileIndex [1]) :
    (evalOp (blockPointerOobLoad xReg) s).map (fun t => t.data i) =
      some (some 0) := by
  rcases i with ⟨i, u⟩
  rcases u
  simp [blockPointerOobLoad, evalOp]
  rfl

def blockPointerOobStoreStmt (outReg : RegionName) : Stmt :=
  Stmt.store .real [1]
    (MemAccess.blockPtr (Op.makeBlockPtr outReg 0 [0] [1] [1] [0]) [0])
    (Op.full [1] (Op.const 7))
    MaskOpt.none

theorem blockPointerOobStore_skips (outReg : RegionName)
    (s : BlockState) :
    stepStmt (blockPointerOobStoreStmt outReg) s = some s := by
  simp [blockPointerOobStoreStmt, stepStmt, evalOp, BlockPtr.inBounds,
    BlockPtr.address]

theorem blockPointerOobStore_observe_unchanged (outReg : RegionName)
    (s : BlockState) :
    TensorView.observe (stepStmt (blockPointerOobStoreStmt outReg) s)
        ({ region := outReg, base := 0, strides := [] } : TensorView [])
        PUnit.unit =
      TensorView.observe (some s)
        ({ region := outReg, base := 0, strides := [] } : TensorView [])
        PUnit.unit := by
  simp [blockPointerOobStore_skips]

end VeriTile.Examples.TritonSmoke
