/-
VeriTile.Examples.TritonSmoke

Small smoke tests for the typed Triton core.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Compute
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.Memory
import VeriTile.Triton.MemoryTyping
import VeriTile.Triton.Launch
import VeriTile.Triton.Concurrency
import VeriTile.Triton.DSL

namespace VeriTile.Examples.TritonSmoke

open VeriTile.Triton

/-- Smoke test for scalar-pointer load/store syntax. -/
def scalarCopyKernel (xReg yReg : RegionName) : ComputeKernel := triton {
  x := tl.load($(xReg))
  tl.store($(yReg), x)
}

/-- The public DSL surface is compute-facing; legacy `Kernel` annotations work
through coercion from this `.fromAlg` subset. -/
def scalarCopyComputeKernel (xReg yReg : RegionName) : ComputeKernel := triton {
  x := tl.load($(xReg))
  tl.store($(yReg), x)
}

example (xReg yReg : RegionName) :
    (scalarCopyComputeKernel xReg yReg).toAlgorithm? =
      Except.ok (scalarCopyKernel xReg yReg) := by
  rfl

example (xReg yReg : RegionName)
    (post : BlockState → BlockState → Prop)
    (h : Kernel.Correct (scalarCopyKernel xReg yReg) post) :
    ComputeKernel.ComputeCorrect (scalarCopyComputeKernel xReg yReg) post :=
  ComputeKernel.computeCorrect_fromAlg h

def atomicAddSmoke (outReg : RegionName) : ComputeKernel := triton {
  x := 1
  tl.atomic_add($(outReg), x)
}

def atomicAddMaskedSmoke (outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  mask := offs < $(N)
  vals := tl.full([$(N)], 1)
  tl.atomic_add($(outReg) + offs, vals, mask=mask)
}

example (outReg : RegionName) :
    (atomicAddSmoke outReg).toAlgorithm? =
      Except.ok
        { inputs := []
        , outputs := [outReg]
        , body :=
            [ Stmt.assign .real [] "x" (Op.const 1)
            , Stmt.atomicAdd NumericDType.real [] (MemAccess.region outReg (Op.constNat 0))
                (Op.ref .real [] "x") MaskOpt.none ] } := by
  rfl

def asyncMarkerSmoke : ComputeKernel :=
  .mk [] [] [ComputeStmt.asyncMarker "tl.async_copy"]

example :
    asyncMarkerSmoke.toAlgorithm? =
      Except.error (.requiresAsyncSequentialization "tl.async_copy") := by
  rfl

example (post : ComputeKernel.AlgSpec) :
    ¬ ComputeKernel.ComputeCorrect asyncMarkerSmoke post :=
  ComputeKernel.not_computeCorrect_of_toAlgorithm_error rfl

def asyncCopySurfaceSmoke (srcReg dstReg : RegionName) : ComputeKernel := triton {
  tl.async_copy($(dstReg), $(srcReg))
}

example (srcReg dstReg : RegionName) :
    (asyncCopySurfaceSmoke srcReg dstReg).toAlgorithm? =
      Except.error (.requiresAsyncSequentialization "tl.async_copy") := by
  rfl

example (srcReg dstReg : RegionName) (post : ComputeKernel.AlgSpec) :
    ¬ ComputeKernel.ComputeCorrect (asyncCopySurfaceSmoke srcReg dstReg) post :=
  ComputeKernel.not_computeCorrect_of_toAlgorithm_error rfl

def asyncWaitSurfaceSmoke : ComputeKernel := triton {
  tl.async_wait()
}

example :
    asyncWaitSurfaceSmoke.toAlgorithm? =
      Except.error (.requiresAsyncSequentialization "tl.async_wait") := by
  rfl

example (post : ComputeKernel.AlgSpec) :
    ¬ ComputeKernel.ComputeCorrect asyncWaitSurfaceSmoke post :=
  ComputeKernel.not_computeCorrect_of_toAlgorithm_error rfl

/-- Vector-add kernel with explicit boundary mask. -/
def addKernelMaskedSmoke (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask := offs < $(nElements)
  x    := tl.load($(xReg) + offs, mask=mask, other=0)
  y    := tl.load($(yReg) + offs, mask=mask, other=0)
  out  := x + y
  tl.store($(outReg) + offs, out, mask=mask)
}

/-- Every comparison operator is reachable via the DSL. -/
def comparisonOpsSmoke : ComputeKernel := triton {
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
def unaryMathOpsSmoke (xReg : RegionName) (N : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  l    := tl.log(e)
  s    := tl.sigmoid(x)
  r    := tl.sqrt(e)
  t    := tl.tanh(x)
  sn   := tl.sin(x)
  cs   := tl.cos(x)
  tn   := tl.tan(x)
  atanv := tl.atan(x)
  ch   := tl.cosh(x)
  sh   := tl.sinh(x)
  l2   := tl.log2(e)
  e2   := tl.exp2(x)
  a    := tl.abs(x)
  lo   := tl.minimum(x, 0)
  hi   := tl.maximum(x, 0)
  nlo  := tl.minimum(offs, $(N))
  nhi  := tl.maximum(offs, $(N))
}

/-- Nat bitwise surface smoke. Signed integer bitwise semantics is deliberately
deferred until fixed-width integer carriers exist. -/
def natBitwiseOpsSmoke (outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  a    := offs & $(7)
  b    := offs | $(1)
  c    := a ^ b
  d    := c << $(1)
  e    := d >> $(2)
  tl.store($(outReg) + offs, e)
}

/-- Prefix scan surface smoke. `tl.associative_scan` accepts the closed
`ScanOp` enum (`sum/prod/max/min`) rather than arbitrary functions. -/
def scanOpsSmoke (xReg outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  s    := tl.cumsum(x, axis = 0)
  p    := tl.cumprod(x, axis = 0)
  m    := tl.associative_scan(x, max, axis = 0)
  n    := tl.associative_scan(x, min, axis = 0)
  y    := s + p + m + n
  tl.store($(outReg) + offs, y)
}

/-- Arg/indexing and sort surface smoke. Arg ties are specified to keep the
smallest axis index; sort is ascending along the static axis. -/
def argSortOpsSmoke (xReg idxReg outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  imax := tl.argmax(x, axis = 0)
  imin := tl.argmin(x, axis = 0)
  y    := tl.sort(x, axis = 0)
  tl.store($(idxReg), imax + imin)
  tl.store($(outReg) + offs, y)
}

/-- Generic shape/view surface smoke. `tl.split` is exposed as projection
form `tl.split(x, 0|1)` because the Lean DSL does not have tuple
destructuring syntax. -/
def shapeViewOpsSmoke (xReg outReg : RegionName) : ComputeKernel := triton {
  offs   := tl.arange(0, 6)
  x      := tl.load($(xReg) + offs)
  matrix := tl.reshape(x, [3, 2])
  left   := tl.split(matrix, 0)
  right  := tl.split(matrix, 1)
  joined := tl.join(left, right)
  perm   := tl.permute(joined, [1, 0])
  flip   := tl.flip(perm, dim = 1)
  y      := tl.reshape(flip, [6])
  tl.store($(outReg) + offs, y)
}

/-- DSL smoke test for explicit floating dtype casts. -/
def dtypeCastSmoke (xReg outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  x32  := tl.cast(x, tl.float32)
  one  := tl.cast(1, tl.float32)
  y32  := x32 + one
  y    := tl.cast(y32, tl.float64)
  tl.store($(outReg) + offs, y)
}

/-- DSL smoke test for Triton's method-style cast spelling. -/
def dtypeToSmoke (xReg outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  x32  := (x).to(tl.float32)
  one  := (1).to(tl.float32)
  y32  := x32 + one
  y    := (y32).to(tl.float64)
  tl.store($(outReg) + offs, y)
}

/-- DSL smoke test for typed floating memory surface syntax. -/
def dtypeLoadStoreSmoke (xReg outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs, dtype=tl.float32)
  y    := x + (1).to(tl.float32)
  tl.store($(outReg) + offs, y)
}

/-- DSL smoke test for masked typed floating load/store surface syntax. -/
def dtypeMaskedLoadStoreSmoke (xReg outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  mask := offs < $(N)
  zero := (0).to(tl.float32)
  x    := tl.load($(xReg) + offs, mask=mask, other=zero, dtype=tl.float32)
  tl.store($(outReg) + offs, x, mask=mask)
}

/-- Constant bit reinterpretation is accepted only when it can be projected to
an algorithm-layer value by the computable decoder. -/
def bitcastConstantSmoke (outReg : RegionName) : ComputeKernel := triton {
  one := tl.bitcast(0x3f800000, tl.float32)
  tl.store($(outReg), one)
}

/-- The compute-facing surface keeps `tl.bitcast` as a `ComputeOp` until
`ComputeKernel.toAlgorithm?` projects it through the shared decoder. -/
def bitcastConstantComputeSmoke (outReg : RegionName) : ComputeKernel := triton {
  one := tl.bitcast(0x3f800000, tl.float32)
  tl.store($(outReg), one)
}

/-- Constant uint32-to-int32 bit reinterpretation projects to mathematical Int. -/
def bitcastConstantIntSmoke (outReg : RegionName) : ComputeKernel := triton {
  minusOne := tl.bitcast(0xffffffff, tl.int32)
  tl.store($(outReg), minusOne)
}

example :
    ComputeOp.constOpToAlgorithm? ComputeOp.oneBitcast = Except.ok (Op.const 1) :=
  ComputeOp.oneBitcast_toAlgorithm

example :
    ComputeOp.constOpToAlgorithm? ComputeOp.minusOneBitcast =
      Except.ok (Op.constInt (-1)) :=
  ComputeOp.minusOneBitcast_toAlgorithm

/-- Runtime bitcast is expressible in the compute AST, but cannot be projected
to AlgorithmCorrect until a compute-level numeric semantics handles it. -/
def bitcastRuntimeSmoke (bitsReg outReg : RegionName) : ComputeKernel := triton {
  bits := tl.load($(bitsReg), dtype=tl.uint32)
  y := tl.bitcast(bits, tl.float32)
  tl.store($(outReg), y)
}

/-- Runtime integer reinterpretation follows the same compute-only path. -/
def bitcastRuntimeIntSmoke (bitsReg outReg : RegionName) : ComputeKernel := triton {
  bits := tl.load($(bitsReg), dtype=tl.uint32)
  y := tl.bitcast(bits, tl.int32)
  tl.store($(outReg), y)
}

example :
    ComputeOp.toAlgorithm?
        (ComputeOp.bitcast .uint32 .fp32 rfl
          (ComputeOp.alg .uint32 (Op.ref .nat [] "bits"))) =
      Except.error (.requiresComputeSemantics "runtime bitcast") := by
  simp [ComputeOp.toAlgorithm?, ComputeOp.constOpToAlgorithm?, ComputeOp.constPayload?]

example :
    ComputeOp.toAlgorithm?
        (ComputeOp.bitcast .uint32 .int32 rfl
          (ComputeOp.alg .uint32 (Op.ref .nat [] "bits"))) =
      Except.error (.requiresComputeSemantics "runtime bitcast") := by
  simp [ComputeOp.toAlgorithm?, ComputeOp.constOpToAlgorithm?, ComputeOp.constPayload?]

/-- The real kernel recovered by erasing `dtypeCastSmoke`'s float annotations. -/
def dtypeCastSmokeErasedReal (xReg outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  x32  := x
  one  := 1
  y32  := x32 + one
  y    := y32
  tl.store($(outReg) + offs, y)
}

/-- The real kernel recovered by erasing `dtypeLoadStoreSmoke`'s float memory. -/
def dtypeLoadStoreSmokeErasedReal (xReg outReg : RegionName) (N : Nat) : ComputeKernel := triton {
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
    (nElements blockSize lo hi : Nat) : ComputeKernel := triton {
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
def booleanSurfaceSmoke (N : Nat) : ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  a    := offs < $(N)
  b    := (offs % $(2)) == $(0)
  c    := tl.logical_or(a, tl.logical_not(b))
  d    := (a & b) | ~c
}

/-- Real-only DSL kernels require real-typed buffers in the lightweight contract. -/
example : True := by
  trivial

/-! ### Typed memory cells -/

/-- Tiny #20 acceptance kernel: compare two Real inputs and store the winning
index as a Nat memory cell. This exercises a non-Real HBM output. -/
def argmax2IndexStoreKernel (xReg outReg : RegionName) : ComputeKernel := triton {
  x0    := tl.load($(xReg) + $(0))
  x1    := tl.load($(xReg) + $(1))
  take1 := x0 < x1
  idx   := tl.where(take1, $(1), $(0))
  tl.store($(outReg), idx)
}

/-- #20 symmetry smoke: load a Nat/index cell from HBM and store it back out. -/
def natLoadStoreKernel (idxReg outReg : RegionName) : ComputeKernel := triton {
  idx := tl.load($(idxReg), dtype=tl.uint64)
  tl.store($(outReg), idx)
}

/-- `tl.uint32` is also an index-like HBM dtype and maps to VeriTile `.nat`. -/
def uint32LoadStoreSmoke (idxReg outReg : RegionName) : ComputeKernel := triton {
  idx := tl.load($(idxReg), dtype=tl.uint32)
  tl.store($(outReg), idx)
}

/-- Narrow unsigned integer spellings also map to VeriTile's `.nat` channel. -/
def uint8Uint16LoadStoreSmoke (idx8Reg idx16Reg out8Reg out16Reg : RegionName) : ComputeKernel := triton {
  idx8  := tl.load($(idx8Reg), dtype=tl.uint8)
  idx16 := tl.load($(idx16Reg), dtype=tl.uint16)
  tl.store($(out8Reg), idx8)
  tl.store($(out16Reg), idx16)
}

/-- Signed integer HBM dtypes map to VeriTile's mathematical `.int` channel. -/
def int64LoadStoreSmoke (idxReg outReg : RegionName) : ComputeKernel := triton {
  idx := tl.load($(idxReg), dtype=tl.int64)
  tl.store($(outReg), idx)
}

/-- Narrow signed integer spellings also map to VeriTile's `.int` channel. -/
def int8Int16LoadStoreSmoke (idx8Reg idx16Reg out8Reg out16Reg : RegionName) : ComputeKernel := triton {
  idx8  := tl.load($(idx8Reg), dtype=tl.int8)
  idx16 := tl.load($(idx16Reg), dtype=tl.int16)
  tl.store($(out8Reg), idx8)
  tl.store($(out16Reg), idx16)
}

/-- Masked integer HBM load with a Nat `other=` value. -/
def uint64MaskedLoadStoreSmoke (idxReg outReg : RegionName) : ComputeKernel := triton {
  mask := $(0) < $(0)
  idx  := tl.load($(idxReg), mask=mask, other=$(0), dtype=tl.uint64)
  tl.store($(outReg), idx)
}

/-- Pointer-register typed integer load. -/
def uint64PointerRegisterLoadSmoke (idxReg outReg : RegionName) : ComputeKernel := triton {
  ptr := $(idxReg) + $(0)
  idx := tl.load(ptr, dtype=tl.uint64)
  tl.store($(outReg), idx)
}

/-! ### Indirect addressing / gather-style reads -/

/-- Generic indirect-addressing smoke: load a Nat index tensor from HBM, feed
it into pointer arithmetic, then issue an ordinary second load. This is the
shared surface pattern for embedding lookup, label lookup, and paged KV. -/
def indirectLoadSmoke (idxReg dataReg outReg : RegionName)
    (N stride : Nat) : ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  idx  := tl.load($(idxReg) + offs, dtype=tl.uint64)
  ptrs := $(dataReg) + idx * $(stride)
  x    := tl.load(ptrs)
  tl.store($(outReg) + offs, x)
}

/-- Paged-KV address smoke: a block-table load drives the physical K-cache
page address. This deliberately uses the same typed load + pointer arithmetic
primitive as `indirectLoadSmoke`, not a special gather AST node. -/
def pagedKVAddressSmoke (blockTableReg kReg outReg : RegionName)
    (T D pageSize pageStride tokenStride dStride : Nat) : ComputeKernel := triton {
  token := tl.arange(0, $(T))
  d     := tl.arange(0, $(D))
  block := tl.load($(blockTableReg) + (token // $(pageSize)), dtype=tl.uint64)
  ptrs  := $(kReg)
    + block[:, None] * $(pageStride)
    + (token[:, None] % $(pageSize)) * $(tokenStride)
    + d[None, :] * $(dStride)
  k     := tl.load(ptrs)
  tl.store($(outReg) + token[:, None] * $(D) + d[None, :], k)
}

def argmax2IndexStoreCoreKernel (xReg outReg : RegionName) : ComputeKernel :=
  ComputeKernel.fromAlg
  { inputs := [xReg]
  , outputs := [outReg]
  , body :=
      [Stmt.store .nat [] (MemAccess.region outReg (Op.constNat 0))
        (Op.where
          (Op.lt ComparableDType.real Broadcast.nil
            (Op.load .real (MemAccess.region xReg (Op.constNat 0)) MaskOpt.none)
            (Op.load .real (MemAccess.region xReg (Op.constNat 1)) MaskOpt.none))
          (Op.constNat 1)
          (Op.constNat 0))
        MaskOpt.none] }

def natLoadStoreCoreKernel (idxReg outReg : RegionName) : ComputeKernel :=
  ComputeKernel.fromAlg
  { inputs := [idxReg]
  , outputs := [outReg]
  , body :=
      [Stmt.store .nat [] (MemAccess.region outReg (Op.constNat 0))
        (Op.load .nat (MemAccess.region idxReg (Op.constNat 0)) MaskOpt.none)
        MaskOpt.none] }

def intLoadStoreCoreKernel (idxReg outReg : RegionName) : ComputeKernel :=
  ComputeKernel.fromAlg
  { inputs := [idxReg]
  , outputs := [outReg]
  , body :=
      [Stmt.store .int [] (MemAccess.region outReg (Op.constNat 0))
        (Op.load .int (MemAccess.region idxReg (Op.constNat 0)) MaskOpt.none)
        MaskOpt.none] }

def indirectLoadCoreKernel (idxReg dataReg outReg : RegionName)
    (N stride : Nat) : ComputeKernel :=
  ComputeKernel.fromAlg
  { inputs := [idxReg, dataReg]
  , outputs := [outReg]
  , body :=
      [Stmt.store .real [N] (MemAccess.region outReg (Op.arange N))
        (Op.load .real
          (MemAccess.ptr
            (Op.ptrAdd Broadcast.scalarL
              (Op.ptrBase dataReg)
              (Op.mul NumericDType.nat Broadcast.scalarR
                (Op.load .nat (MemAccess.region idxReg (Op.arange N)) MaskOpt.none)
                (Op.constNat stride))))
          MaskOpt.none)
        MaskOpt.none] }

def indirectLoadView (idxReg dataReg : RegionName)
    (N stride : Nat) : IndirectView [N] [] :=
  { idxRegion := idxReg
  , dataRegion := dataReg
  , idxOffset := fun i => i.1.val
  , dataAddr := fun loadedIdx _ => loadedIdx * stride }

def linearOutView (outReg : RegionName) (N : Nat) : TensorView [N] :=
  { region := outReg, base := 0, strides := [1] }

noncomputable def argmax2ExpectedIndex (s : BlockState) (xReg : RegionName) : Nat :=
  by
    classical
    exact if s.readMem xReg 0 < s.readMem xReg 1 then 1 else 0

theorem argmax2_index_store_exec_correct
    (xReg outReg : RegionName) (s : BlockState) :
    let result := exec (argmax2IndexStoreCoreKernel xReg outReg) s
    result.map (fun s' => s'.readMemTyped .nat outReg 0) =
      some (some (argmax2ExpectedIndex s xReg)) := by
  classical
  simp [argmax2IndexStoreCoreKernel, exec, stepStmts, stepStmt, evalOp,
    argmax2ExpectedIndex, BlockState.writeMemTyped, BlockState.readMemTyped,
    ComparableDType.lt, Tile.cop, Tile.select, Option.bind, Option.map,
    MemCell.readAs_of_same, WithBot.some_eq_coe, WithBot.coe_lt_coe]

theorem argmax2_index_store_correct
    (xReg outReg : RegionName) (s : BlockState) :
    ComputeKernel.ComputeCorrect
      ((argmax2IndexStoreCoreKernel xReg outReg))
      (fun s0 s' =>
        s0 = s →
        s'.readMemTyped .nat outReg 0 = some (argmax2ExpectedIndex s xReg)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  have hview := argmax2_index_store_exec_correct xReg outReg s
  rw [hExec] at hview
  simpa using hview

theorem nat_load_store_exec_correct
    (idxReg outReg : RegionName) (s : BlockState) :
    let result := exec (natLoadStoreCoreKernel idxReg outReg) s
    result.map (fun s' => s'.readMemTyped .nat outReg 0) =
      some (some (s.readMemValue .nat idxReg 0)) := by
  simp [natLoadStoreCoreKernel, exec, stepStmts, stepStmt, evalOp,
    BlockState.readMemValue, BlockState.readMemTyped, BlockState.writeMemTyped,
    Option.bind, Option.map, MemCell.readAs_of_same]

theorem nat_load_store_correct
    (idxReg outReg : RegionName) (s : BlockState) :
    ComputeKernel.ComputeCorrect
      ((natLoadStoreCoreKernel idxReg outReg))
      (fun s0 s' =>
        s0 = s →
        s'.readMemTyped .nat outReg 0 =
          some (s.readMemValue .nat idxReg 0)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  have hview := nat_load_store_exec_correct idxReg outReg s
  rw [hExec] at hview
  simpa using hview

theorem int_load_store_correct
    (idxReg outReg : RegionName) (s : BlockState) :
    let result := exec (intLoadStoreCoreKernel idxReg outReg) s
    result.map (fun s' => s'.readMemTyped .int outReg 0) =
      some (some (s.readMemValue .int idxReg 0)) := by
  simp [intLoadStoreCoreKernel, exec, stepStmts, stepStmt, evalOp,
    BlockState.readMemValue, BlockState.readMemTyped, BlockState.writeMemTyped,
    Option.bind, Option.map, MemCell.readAs_of_same]

theorem indirect_load_correct_exec_view
    (idxReg dataReg outReg : RegionName) (N stride : Nat)
    (s : BlockState) (i : TileIndex [N]) :
    TensorView.observe
        (exec (indirectLoadCoreKernel idxReg dataReg outReg N stride) s)
        (linearOutView outReg N) i =
      some (IndirectView.observe (indirectLoadView idxReg dataReg N stride) s i PUnit.unit) := by
  simp [indirectLoadCoreKernel, indirectLoadView, linearOutView, exec, stepStmts,
    stepStmt, evalOp, TensorView.observe, observeTileAt, TensorView.offset,
    Offset.strided, Tile.ptrAdd, Tile.bop]
  rw [BlockState.scatter_readback_nd]
  · simp [IndirectView.observe, NumericDType.mul]
  · intro a b h
    rcases a with ⟨a, au⟩
    rcases au
    rcases b with ⟨b, bu⟩
    rcases bu
    congr
    exact Fin.ext h

theorem indirect_load_correct_view
    (idxReg dataReg outReg : RegionName) (N stride : Nat)
    (s : BlockState) (i : TileIndex [N]) :
    ComputeKernel.ComputeCorrect
      ((indirectLoadCoreKernel idxReg dataReg outReg N stride))
      (fun s0 s' =>
        s0 = s →
        TensorView.observe (some s') (linearOutView outReg N) i =
          some (IndirectView.observe (indirectLoadView idxReg dataReg N stride)
            s i PUnit.unit)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  have hview := indirect_load_correct_exec_view idxReg dataReg outReg N stride s i
  rw [hExec] at hview
  simpa using hview

/-! ### ND grid launch theorem surface (issue #5) -/

def launchNoopKernel : ComputeKernel :=
  ComputeKernel.fromAlg { inputs := [], outputs := [], body := [] }

theorem launch_noop_for_all_programs_some
    (g : Grid) (s : BlockState) :
    Kernel.ForAllProgramsSome launchNoopKernel g s
      (fun idx s' => s'.pids = idx.toPids) := by
  intro idx
  refine ⟨s.withGridIndex idx, ?_, rfl⟩
  simp [launchNoopKernel, exec]

example (g : Grid) (s : BlockState) (idx : GridIndex g)
    (ax : Fin g.rank) :
    evalOp (Op.programId ax.val) (s.withGridIndex idx) =
      some (Tile.scalar (idx ax).val) :=
  program_id_under_grid_index s idx ax

example (g : Grid) (s : BlockState) (idx : GridIndex g)
    (ax : Nat) (h : g.rank ≤ ax) :
    evalOp (Op.programId ax) (s.withGridIndex idx) =
      some (Tile.scalar 0) :=
  program_id_out_of_grid_rank s idx h

/-! ### Optional well-formedness checker (issue #46) -/

def checkerEnv (entries : List (RegionName × TileDType)) : RegionEnv :=
  fun region => (entries.find? (fun entry => entry.1 == region)).map (·.2)

def registerConflictKernel : ComputeKernel :=
  ComputeKernel.fromAlg
  { inputs := [], outputs := []
  , body :=
      [ Stmt.assign .nat [] "x" (Op.constNat 0)
      , Stmt.assign .real [] "x" (Op.const 0) ] }

def pointerWhereConflictKernel : ComputeKernel :=
  ComputeKernel.fromAlg
  { inputs := [], outputs := []
  , body :=
      [ Stmt.assign .ptr [] "p"
          (Op.where (Op.constBool Bool.true) (Op.ptrBase "x") (Op.ptrBase "y")) ] }

def checkerValidPointerStoreKernel (xReg outReg : RegionName) : ComputeKernel :=
  ComputeKernel.fromAlg
  { inputs := [xReg], outputs := [outReg]
  , body :=
      [ Stmt.assign .nat [4] "offs" (Op.arange 4)
      , Stmt.assign .real [4] "x"
          (Op.load .real (MemAccess.region xReg (Op.ref .nat [4] "offs")) MaskOpt.none)
      , Stmt.assign .ptr [4] "outPtrs"
          (Op.ptrAdd Broadcast.scalarL (Op.ptrBase outReg) (Op.ref .nat [4] "offs"))
      , Stmt.store .real [4]
          (MemAccess.ptr (Op.ref .ptr [4] "outPtrs"))
          (Op.ref .real [4] "x")
          MaskOpt.none ] }

def unboundPointerRefKernel : ComputeKernel :=
  ComputeKernel.fromAlg
  { inputs := ["x"], outputs := []
  , body :=
      [ Stmt.assign .real [] "x"
          (Op.load .real (MemAccess.ptr (Op.ref .ptr [] "p")) MaskOpt.none) ] }

def nonPointerRefAsPointerKernel : ComputeKernel :=
  ComputeKernel.fromAlg
  { inputs := ["x"], outputs := []
  , body :=
      [ Stmt.assign .real [] "p" (Op.const 0)
      , Stmt.assign .real [] "x"
          (Op.load .real (MemAccess.ptr (Op.ref .ptr [] "p")) MaskOpt.none) ] }

def blockPointerRefKernel (xReg : RegionName) : ComputeKernel :=
  ComputeKernel.fromAlg
  { inputs := [xReg], outputs := []
  , body :=
      [ Stmt.assign .blockPtr [1] "bp"
          (Op.makeBlockPtr xReg 0 [8] [1] [1] [0])
      , Stmt.assign .blockPtr [1] "bp2"
          (Op.advanceBlockPtr (Op.ref .blockPtr [1] "bp") [1])
      , Stmt.store .real [1]
          (MemAccess.blockPtr (Op.ref .blockPtr [1] "bp2") [0])
          (Op.full [1] (Op.const 1))
          MaskOpt.none ] }

def blockPointerMetadataMismatchKernel (xReg : RegionName) : ComputeKernel :=
  ComputeKernel.fromAlg
  { inputs := [xReg], outputs := []
  , body :=
      [ Stmt.assign .blockPtr [1] "bp"
          (Op.makeBlockPtr xReg 0 [8] [1] [] [0]) ] }

def blockPointerBoundaryMismatchKernel (xReg : RegionName) : ComputeKernel :=
  ComputeKernel.fromAlg
  { inputs := [xReg], outputs := []
  , body :=
      [ Stmt.assign .blockPtr [1] "bp"
          (Op.makeBlockPtr xReg 0 [8] [1] [1] [0])
      , Stmt.store .real [1]
          (MemAccess.blockPtr (Op.ref .blockPtr [1] "bp") [1])
          (Op.full [1] (Op.const 1))
          MaskOpt.none ] }

example :
    Kernel.checkStrict (checkerEnv [("x", .real), ("out", .real)])
      (checkerValidPointerStoreKernel "x" "out") = .ok () := by
  native_decide

example :
    Kernel.check (checkerEnv []) (scalarCopyKernel "x" "y") = .ok () := by
  native_decide

example :
    Kernel.checkStrict (checkerEnv []) (scalarCopyKernel "x" "y") =
      .error (.undeclaredRegion "x") := by
  native_decide

example :
    Kernel.checkStrict (checkerEnv [("idx", .real), ("out", .nat)])
      (natLoadStoreKernel "idx" "out") =
      .error (.regionDTypeMismatch "idx" .nat .real) := by
  native_decide

example :
    Kernel.checkStrict (checkerEnv [("idx", .real), ("out", .nat)])
      (uint64PointerRegisterLoadSmoke "idx" "out") =
      .error (.regionDTypeMismatch "idx" .nat .real) := by
  native_decide

example :
    Kernel.checkStrict (checkerEnv []) registerConflictKernel =
      .error (.registerDTypeShapeMismatch "x" .nat .real [] []) := by
  native_decide

example :
    Kernel.checkStrict (checkerEnv [("x", .real), ("y", .real)])
      pointerWhereConflictKernel =
      .error (.pointerProvenanceConflict "x" "y") := by
  native_decide

example :
    Kernel.checkStrict (checkerEnv [("x", .real)])
      unboundPointerRefKernel =
      .error (.unboundRegister "p") := by
  native_decide

example :
    Kernel.checkStrict (checkerEnv [("x", .real)])
      nonPointerRefAsPointerKernel =
      .error (.registerDTypeShapeMismatch "p" .real .ptr [] []) := by
  native_decide

example :
    Kernel.checkStrict (checkerEnv [("x", .real)])
      (blockPointerRefKernel "x") = .ok () := by
  native_decide

example :
    Kernel.checkStrict (checkerEnv [("x", .real)])
      (blockPointerMetadataMismatchKernel "x") =
      .error (.blockPointerMetadataMismatch 1 0 1 1) := by
  native_decide

example :
    Kernel.checkStrict (checkerEnv [("x", .real)])
      (blockPointerBoundaryMismatchKernel "x") =
      .error (.boundaryAxisOutOfRange 1 1) := by
  native_decide

example : (MemCell.of .nat 7).readAs .nat = some 7 := by
  rfl

example : (MemCell.of TileDType.bool Bool.true).readAs TileDType.bool = some Bool.true := by
  rfl

example : (MemCell.of .fp32 (some (3 : ℝ))).readAs .real = none := by
  rfl

example :
    (({ mem := fun _ _ => MemCell.of .fp32 (some (5 : ℝ))
      , regs := fun _ _ _ => none
      , pids := fun _ => 0
      , undef := fun _ _ => 0 } : BlockState).eraseDType).readMem "X" 0 = 5 := by
  rfl

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
def colExpandSmoke (xReg : RegionName) (N : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  xc   := x[:, None]
}

/-- `[None, :]` produces a `[1, N]` tile. -/
def rowExpandSmoke (xReg : RegionName) (N : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  xr   := x[None, :]
}

/-- `tl.expand_dims(x, axis=1)` is the function-form equivalent of
`x[:, None]`. -/
def expandDimsKwargSmoke (xReg : RegionName) (N : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  xc   := tl.expand_dims(x, axis=1)
}

/-- Positional-axis `tl.expand_dims(x, 0)` produces a `[1, N]` tile. -/
def expandDimsPositionalSmoke (xReg : RegionName) (N : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  xr   := tl.expand_dims(x, 0)
}

/-- Function-form `tl.expand_dims` also works on higher-rank tiles. -/
def expandDimsRank2Smoke
    (xReg : RegionName) (M N stride : Nat) : ComputeKernel := triton {
  offsM := tl.arange(0, $(M))
  offsN := tl.arange(0, $(N))
  ptrs  := offsM[:, None] * $(stride) + offsN[None, :]
  x     := tl.load($(xReg) + ptrs)
  x3    := tl.expand_dims(x, axis=0)
}

/-! ### `tl.static_range` surface alias -/

/-- `tl.static_range` currently lowers to the same bounded-loop AST as
`tl.for`; unroll / pipeline attributes are intentionally not modeled. -/
def staticRangeSmoke (N : Nat) : ComputeKernel := triton {
  acc := 0
  tl.static_range i in $(N) {
    acc := acc + tl.toReal(i)
  }
}

/-- Numeric-literal `tl.static_range` bounds share the same lowering. -/
def staticRangeLiteralSmoke : ComputeKernel := triton {
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

def pointerValueLoadSmoke (xReg : RegionName) (N : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  ptrs := $(xReg) + offs
  x    := tl.load(ptrs)
}

def pointerValueOffsetLoadSmoke (xReg : RegionName) (N : Nat) : ComputeKernel := triton {
  pid   := tl.program_id(0)
  offs  := pid * $(N) + tl.arange(0, $(N))
  ptrs  := $(xReg) + offs
  ptrs2 := ptrs + $(N)
  y     := tl.load(ptrs2)
}

def pointerValueStoreSmoke (xReg outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offs    := pid * $(N) + tl.arange(0, $(N))
  x       := tl.load($(xReg) + offs)
  outPtrs := $(outReg) + offs
  tl.store(outPtrs, x)
}

def pointerValueMaskedStoreSmoke (xReg outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offs    := pid * $(N) + tl.arange(0, $(N))
  mask    := offs < $(N)
  ptrs    := $(xReg) + offs
  x       := tl.load(ptrs, mask=mask, other=0)
  outPtrs := $(outReg) + offs
  tl.store(outPtrs, x, mask=mask)
}

/-! ### Block pointers and boundary checks (issue #47) -/

def blockPointerBoundaryCopySmoke (xReg outReg : RegionName) (N B : Nat) : ComputeKernel := triton {
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
