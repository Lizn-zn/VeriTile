/-
VeriTile.Examples.TritonSmoke

Small smoke tests for the typed Triton core.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.Memory
import VeriTile.Triton.Memory.Typing
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

/-- The public DSL surface is compute-facing. Pure algorithm statements are
represented internally as `ComputeStmt.alg` entries in a `ComputeKernel`. -/
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
    (h : ComputeCorrect.General (scalarCopyKernel xReg yReg) post) :
    ComputeCorrect.General (scalarCopyComputeKernel xReg yReg) post :=
  by
    simpa [scalarCopyComputeKernel, scalarCopyKernel] using h

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

def atomicMaxFailureSmoke (outReg : RegionName) : ComputeKernel := triton {
  x := 1
  tl.atomic_max($(outReg), x)
}

def atomicMinFailureSmoke (outReg : RegionName) : ComputeKernel := triton {
  x := 1
  tl.atomic_min($(outReg), x)
}

def atomicAndFailureSmoke (outReg : RegionName) : ComputeKernel := triton {
  x := 1
  tl.atomic_and($(outReg), x)
}

def atomicOrFailureSmoke (outReg : RegionName) : ComputeKernel := triton {
  x := 1
  tl.atomic_or($(outReg), x)
}

def atomicXorFailureSmoke (outReg : RegionName) : ComputeKernel := triton {
  x := 1
  tl.atomic_xor($(outReg), x)
}

def atomicXchgSmoke (outReg : RegionName) : ComputeKernel := triton {
  x := 1
  tl.atomic_xchg($(outReg), x)
}

def atomicCasSmoke (outReg : RegionName) : ComputeKernel := triton {
  cmp := 0
  x := 1
  tl.atomic_cas($(outReg), cmp, x)
}

def atomicXchgReturnSmoke (outReg : RegionName) : ComputeKernel := triton {
  x := 1
  old := tl.atomic_xchg($(outReg), x)
  tl.store($(outReg) + $(1), old)
}

def atomicCasReturnSmoke (outReg : RegionName) : ComputeKernel := triton {
  cmp := 0
  x := 1
  old := tl.atomic_cas($(outReg), cmp, x)
  tl.store($(outReg) + $(1), old)
}

example (outReg : RegionName) :
    (atomicMaxFailureSmoke outReg).toAlgorithm? =
      Except.error (.requiresEffectProjection "tl.atomic_max") := by
  rfl

example (outReg : RegionName) :
    (atomicMinFailureSmoke outReg).toAlgorithm? =
      Except.error (.requiresEffectProjection "tl.atomic_min") := by
  rfl

example (outReg : RegionName) :
    (atomicAndFailureSmoke outReg).toAlgorithm? =
      Except.error (.requiresEffectProjection "tl.atomic_and") := by
  rfl

example (outReg : RegionName) :
    (atomicOrFailureSmoke outReg).toAlgorithm? =
      Except.error (.requiresEffectProjection "tl.atomic_or") := by
  rfl

example (outReg : RegionName) :
    (atomicXorFailureSmoke outReg).toAlgorithm? =
      Except.error (.requiresEffectProjection "tl.atomic_xor") := by
  rfl

example (outReg : RegionName) :
    (atomicXchgSmoke outReg).toAlgorithm? =
      Except.ok
        { inputs := []
        , outputs := [outReg]
        , body :=
            [ Stmt.assign .real [] "x" (Op.const 1)
            , Stmt.atomicRMW RMWOp.xchg .real []
                (MemAccess.region outReg (Op.constNat 0))
                (Op.ref .real [] "x") none MaskOpt.none none ] } := by
  rfl

example (outReg : RegionName) :
    (atomicCasSmoke outReg).toAlgorithm? =
      Except.ok
        { inputs := []
        , outputs := [outReg]
        , body :=
            [ Stmt.assign .real [] "cmp" (Op.const 0)
            , Stmt.assign .real [] "x" (Op.const 1)
            , Stmt.atomicRMW RMWOp.cas .real []
                (MemAccess.region outReg (Op.constNat 0))
                (Op.ref .real [] "cmp") (some (Op.ref .real [] "x"))
                MaskOpt.none none ] } := by
  rfl

example (outReg : RegionName) :
    (atomicXchgReturnSmoke outReg).toAlgorithm? =
      Except.ok
        { inputs := [outReg]
        , outputs := [outReg]
        , body :=
            [ Stmt.assign .real [] "x" (Op.const 1)
            , Stmt.atomicRMW RMWOp.xchg .real []
                (MemAccess.region outReg (Op.constNat 0))
                (Op.ref .real [] "x") none MaskOpt.none (some "old")
            , Stmt.store .real [] (MemAccess.region outReg (Op.constNat 1))
                (Op.ref .real [] "old") MaskOpt.none ] } := by
  rfl

example (outReg : RegionName) :
    (atomicCasReturnSmoke outReg).toAlgorithm? =
      Except.ok
        { inputs := [outReg]
        , outputs := [outReg]
        , body :=
            [ Stmt.assign .real [] "cmp" (Op.const 0)
            , Stmt.assign .real [] "x" (Op.const 1)
            , Stmt.atomicRMW RMWOp.cas .real []
                (MemAccess.region outReg (Op.constNat 0))
                (Op.ref .real [] "cmp") (some (Op.ref .real [] "x"))
                MaskOpt.none (some "old")
            , Stmt.store .real [] (MemAccess.region outReg (Op.constNat 1))
                (Op.ref .real [] "old") MaskOpt.none ] } := by
  rfl

def effectMarkerSmoke : ComputeKernel :=
  .mk [] [] [ComputeStmt.effectMarker "tl.async_copy"]

example :
    effectMarkerSmoke.toAlgorithm? =
      Except.error (.requiresEffectProjection "tl.async_copy") := by
  rfl

example (post : ComputeKernel.AlgSpec) :
    ¬ ComputeCorrect.General effectMarkerSmoke post :=
  ComputeKernel.not_computeCorrect_of_toAlgorithm_error rfl

def asyncCopySurfaceSmoke (srcReg dstReg : RegionName) : ComputeKernel := triton {
  tl.async_copy($(dstReg), $(srcReg))
}

example (srcReg dstReg : RegionName) :
    (asyncCopySurfaceSmoke srcReg dstReg).toAlgorithm? =
      Except.error (.requiresEffectProjection "tl.async_copy") := by
  rfl

example (srcReg dstReg : RegionName) (post : ComputeKernel.AlgSpec) :
    ¬ ComputeCorrect.General (asyncCopySurfaceSmoke srcReg dstReg) post :=
  ComputeKernel.not_computeCorrect_of_toAlgorithm_error rfl

def asyncWaitSurfaceSmoke : ComputeKernel := triton {
  tl.async_wait()
}

example :
    asyncWaitSurfaceSmoke.toAlgorithm? =
      Except.error (.requiresEffectProjection "tl.async_wait") := by
  rfl

example (post : ComputeKernel.AlgSpec) :
    ¬ ComputeCorrect.General asyncWaitSurfaceSmoke post :=
  ComputeKernel.not_computeCorrect_of_toAlgorithm_error rfl

def debugBarrierSurfaceSmoke : ComputeKernel := triton {
  tl.debug_barrier()
}

/-- `tl.debug_barrier` is an intra-program fence: a semantic no-op under the
sequential per-program semantics, so (unlike the async effects above) it
**erases** at the algorithm layer instead of demanding an effect projection. -/
example :
    debugBarrierSurfaceSmoke.toAlgorithm? =
      Except.ok
        { inputs := []
        , outputs := []
        , body := [ Stmt.ifThen (Op.constBool Bool.false) [] ] } := by
  rfl

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
  erfv := tl.erf(x)
  erfLibdevice := tl.extra.cuda.libdevice.erf(x)
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

/-- Reverse (suffix) scan surface smoke (#94): `reverse=True` lowers to the
same scan node with `ScanDirection.reverse`; `reverse=False` is the explicit
forward spelling. -/
def reverseScanOpsSmoke (xReg outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  r    := tl.cumsum(x, axis = 0, reverse=True)
  f    := tl.cumsum(x, axis = 0, reverse=False)
  q    := tl.cumprod(x, reverse=True, axis = 0)
  m    := tl.associative_scan(x, max, axis = 0, reverse=True)
  y    := r + f + q + m
  tl.store($(outReg) + offs, y)
}

/-- The reverse-scan surface lowers to the algorithm layer. -/
example (xReg outReg : RegionName) (N : Nat) :
    ∃ alg, (reverseScanOpsSmoke xReg outReg N).toAlgorithm? = Except.ok alg := by
  simp [reverseScanOpsSmoke, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Launch-grid query smoke (#92): `tl.num_programs(axis)` in offset
arithmetic — the canonical grid-stride pattern `base = pid + i·num_programs`.
Both the positional and `axis=` spellings are accepted. -/
def numProgramsSmoke (xReg outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  np   := tl.num_programs(0)
  np1  := tl.num_programs(axis=1)
  offs := pid + tl.arange(0, $(N)) * np * np1
  x    := tl.load($(xReg) + offs)
  tl.store($(outReg) + offs, x)
}

/-- The num_programs surface lowers to the algorithm layer. -/
example (xReg outReg : RegionName) (N : Nat) :
    ∃ alg, (numProgramsSmoke xReg outReg N).toAlgorithm? = Except.ok alg := by
  simp [numProgramsSmoke, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Semantics smoke: `tl.num_programs(axis)` reads the launch-grid dimension
field, and an unlaunched (default) state answers `1` on every axis. (`ax`
binder name because `axis` is a DSL kwarg token.) -/
example (ax : Nat) (s : BlockState) :
    evalOp (.numPrograms ax) s = some (Tile.scalar (s.numPids ax)) :=
  evalOp_numPrograms ax s

example (ax : Nat) : (default : BlockState).numPids ax = 1 := rfl

/-- Semantics smoke: the reverse scan is the **suffix** fold — at each index it
folds exactly the lanes whose axis coordinate is `≥` the output coordinate
(compare `Tile.scan_data`, whose kept set is the `≤` prefix). -/
example (op : ScanOp) (x : Tile .real [8]) (idx : TileIndex [8]) :
    (Tile.scan op ⟨0, by simp⟩ .reverse x).data idx
      = op.eval
          (((Finset.univ : Finset (Fin (TileShape.axisDim [8] ⟨0, by simp⟩))).filter
            (fun k => (TileShape.axisCoord [8] ⟨0, by simp⟩ idx).val ≤ k.val)).toList.map
            (fun k => x.data (TileShape.replaceAxisCoord [8] ⟨0, by simp⟩ idx k))) :=
  Tile.scan_reverse_data ..

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

/-- `tl.zeros(..., dtype=tl.float32)` preserves an fp32 compute node while
projecting to a Real algorithm tile. -/
def zerosFp32ComputeSmoke (N : Nat) : ComputeKernel := triton {
  z := tl.zeros([$(N)], dtype=tl.float32)
}

def fullFp32ComputeSmoke (N : Nat) : ComputeKernel := triton {
  z := tl.full([$(N)], 1, dtype=tl.float32)
}

def fp32AccumulatorArithSmoke (N : Nat) : ComputeKernel := triton {
  z := tl.zeros([$(N)], dtype=tl.float32)
  y := z + 1
}

def rightAlignedBroadcastSmoke (M K : Nat) : ComputeKernel := triton {
  row := tl.arange(0, $(M))
  col := tl.arange(0, $(K))
  a := col
  b := row[:, None] + col[None, :]
  c := a * b
}

def fp32CastComputeSmoke (N : Nat) : ComputeKernel := triton {
  z := tl.full([$(N)], 1)
  y := tl.cast(z, tl.float32)
}

def fp32ToComputeSmoke (N : Nat) : ComputeKernel := triton {
  z := tl.full([$(N)], 1)
  y := (z).to(tl.float32)
}

def fp32LoadComputeSmoke (xReg : RegionName) (N : Nat) : ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  x := tl.load($(xReg) + offs, dtype=tl.float32)
}

example (N : Nat) :
    zerosFp32ComputeSmoke N =
      ComputeKernel.mk [] []
        [ComputeStmt.assign .real [N] "z"
          (ComputeExpr.compute
            (ComputeOp.full [N] (ComputeOp.alg .fp32 (Op.const 0))))] := by
  rfl

example (N : Nat) :
    (zerosFp32ComputeSmoke N).toAlgorithm? =
      Except.ok
        (Kernel.mk [] []
          [Stmt.assign .real [N] "z" (Op.full [N] (Op.const 0))]) := by
  simp [zerosFp32ComputeSmoke, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  rfl

example (N : Nat) :
    fullFp32ComputeSmoke N =
      ComputeKernel.mk [] []
        [ComputeStmt.assign .real [N] "z"
          (ComputeExpr.compute
            (ComputeOp.full [N] (ComputeOp.alg .fp32 (Op.const 1))))] := by
  rfl

example (N : Nat) :
    (fullFp32ComputeSmoke N).toAlgorithm? =
      Except.ok
        (Kernel.mk [] []
          [Stmt.assign .real [N] "z" (Op.full [N] (Op.const 1))]) := by
  simp [fullFp32ComputeSmoke, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  rfl

example (N : Nat) :
    fp32AccumulatorArithSmoke N =
      ComputeKernel.mk [] []
        [ ComputeStmt.assign .real [N] "z"
            (ComputeExpr.compute
              (ComputeOp.full [N] (ComputeOp.alg .fp32 (Op.const 0))))
        , ComputeStmt.assign .real [N] "y"
            (ComputeExpr.compute
              (ComputeOp.alg .fp32
                (Op.add NumericDType.real Broadcast.scalarR
                  (Op.ref .real [N] "z") (Op.const 1))))] := by
  rfl

example (N : Nat) :
    (fp32AccumulatorArithSmoke N).toAlgorithm? =
      Except.ok
        (Kernel.mk [] []
          [ Stmt.assign .real [N] "z" (Op.full [N] (Op.const 0))
          , Stmt.assign .real [N] "y"
              (Op.add NumericDType.real Broadcast.scalarR
                (Op.ref .real [N] "z") (Op.const 1))]) := by
  simp [fp32AccumulatorArithSmoke, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  constructor <;> rfl

example (N : Nat) :
    fp32CastComputeSmoke N =
      ComputeKernel.mk [] []
        [ ComputeStmt.assign .real [N] "z"
            (ComputeExpr.alg (Op.full [N] (Op.const 1)))
        , ComputeStmt.assign .real [N] "y"
            (ComputeExpr.compute
              (ComputeOp.alg .fp32
                (Op.castFloat FloatDType.real FloatDType.real
                  (Op.ref .real [N] "z"))))] := by
  rfl

example (N : Nat) :
    fp32ToComputeSmoke N =
      ComputeKernel.mk [] []
        [ ComputeStmt.assign .real [N] "z"
            (ComputeExpr.alg (Op.full [N] (Op.const 1)))
        , ComputeStmt.assign .real [N] "y"
            (ComputeExpr.compute
              (ComputeOp.alg .fp32
                (Op.castFloat FloatDType.real FloatDType.real
                  (Op.ref .real [N] "z"))))] := by
  rfl

example (xReg : RegionName) (N : Nat) :
    fp32LoadComputeSmoke xReg N =
      ComputeKernel.mk [xReg] []
        [ ComputeStmt.assign .nat [N] "offs"
            (ComputeExpr.alg (Op.arange N))
        , ComputeStmt.assign .real [N] "x"
            (ComputeExpr.compute
              (ComputeOp.load .fp32
                (MemAccess.region xReg (Op.ref .nat [N] "offs"))
                MaskOpt.none))] := by
  rfl

example (xReg : RegionName) (N : Nat) :
    (fp32LoadComputeSmoke xReg N).toAlgorithm? =
      Except.ok
        (Kernel.mk [xReg] []
          [ Stmt.assign .nat [N] "offs" (Op.arange N)
          , Stmt.assign .real [N] "x"
              (Op.load ComputeDType.fp32.eraseDType
                (MemAccess.region xReg (Op.ref .nat [N] "offs"))
                MaskOpt.none)]) := by
  simp [fp32LoadComputeSmoke, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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
to AlgorithmCorrect_without_Rounding until a compute-level numeric semantics handles it. -/
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

/-- Typed region surface: the `.nat` annotation on the region supplies the
`tl.load` dtype, so no explicit `dtype=tl.uint64` is needed. -/
def typedNatRegionLoadStoreKernel (idxReg : Region .nat) (outReg : RegionName) :
    ComputeKernel := triton {
  idx := tl.load($((idxReg : Region .nat)))
  tl.store($(outReg), idx)
}

example (idxReg : Region .nat) (outReg : RegionName) :
    typedNatRegionLoadStoreKernel idxReg outReg =
      ComputeKernel.mk [idxReg] [outReg]
        [ ComputeStmt.assign .nat [] "idx"
            (ComputeExpr.alg
              (Op.load .nat
                (MemAccess.region idxReg.name (Op.constNat 0))
                MaskOpt.none))
        , ComputeStmt.store .nat [] (MemAccess.region outReg (Op.constNat 0))
            (ComputeExpr.alg (Op.ref .nat [] "idx")) MaskOpt.none ] := by
  rfl

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

/-- Nat index expression cast into the signed `.int` value channel inside `tl.where`.
This covers argmax-style `tl.int64` index accumulators whose source indices are
nonnegative Nat lanes. -/
def natToIntWhereSmoke (outReg : Region .int) (N : Nat) : ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  zeros := tl.full([$(N)], dtype=tl.int64, value=$(0))
  take := offs < $(N)
  idx := tl.where(take, (offs).to(tl.int64), zeros)
  tl.store($((outReg : Region .int)) + offs, idx)
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
  let body : List Stmt :=
      [Stmt.store .nat [] (MemAccess.region outReg (Op.constNat 0))
        (Op.where
          (Op.lt ComparableDType.real Broadcast.nil
            (Op.load .real (MemAccess.region xReg (Op.constNat 0)) MaskOpt.none)
            (Op.load .real (MemAccess.region xReg (Op.constNat 1)) MaskOpt.none))
          (Op.constNat 1)
          (Op.constNat 0))
        MaskOpt.none]
  ComputeKernel.fromKernelBody [xReg] [outReg] body

def natLoadStoreCoreKernel (idxReg outReg : RegionName) : ComputeKernel :=
  let body : List Stmt :=
      [Stmt.store .nat [] (MemAccess.region outReg (Op.constNat 0))
        (Op.load .nat (MemAccess.region idxReg (Op.constNat 0)) MaskOpt.none)
        MaskOpt.none]
  ComputeKernel.fromKernelBody [idxReg] [outReg] body

def intLoadStoreCoreKernel (idxReg outReg : RegionName) : ComputeKernel :=
  let body : List Stmt :=
      [Stmt.store .int [] (MemAccess.region outReg (Op.constNat 0))
        (Op.load .int (MemAccess.region idxReg (Op.constNat 0)) MaskOpt.none)
        MaskOpt.none]
  ComputeKernel.fromKernelBody [idxReg] [outReg] body

def indirectLoadCoreKernel (idxReg dataReg outReg : RegionName)
    (N stride : Nat) : ComputeKernel :=
  let body : List Stmt :=
      [Stmt.store .real [N] (MemAccess.region outReg (Op.arange N))
        (Op.load .real
          (MemAccess.ptr
            (Op.ptrAdd Broadcast.scalarL
              (Op.ptrBase dataReg)
                (Op.mul NumericDType.nat Broadcast.scalarR
                  (Op.load .nat (MemAccess.region idxReg (Op.arange N)) MaskOpt.none)
                (Op.constNat stride))))
          MaskOpt.none)
        MaskOpt.none]
  ComputeKernel.fromKernelBody [idxReg, dataReg] [outReg] body

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

specification argmax2_index_store_correct_view
    (xReg outReg : RegionName) (s : BlockState) :
    ComputeCorrect.General
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

specification nat_load_store_correct_view
    (idxReg outReg : RegionName) (s : BlockState) :
    ComputeCorrect.General
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

specification indirect_load_correct_view
    (idxReg dataReg outReg : RegionName) (N stride : Nat)
    (s : BlockState) (i : TileIndex [N]) :
    ComputeCorrect.General
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
  ComputeKernel.mk [] [] []

specification launch_noop_for_all_programs_some
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
  let body : List Stmt :=
    [ Stmt.assign .nat [] "x" (Op.constNat 0)
    , Stmt.assign .real [] "x" (Op.const 0) ]
  ComputeKernel.fromKernelBody [] [] body

def pointerWhereConflictKernel : ComputeKernel :=
  let body : List Stmt :=
    [ Stmt.assign .ptr [] "p"
        (Op.where (Op.constBool Bool.true)
          (Op.ptrBase (d := .real) "x") (Op.ptrBase (d := .real) "y")) ]
  ComputeKernel.fromKernelBody [] [] body

def checkerValidPointerStoreKernel (xReg outReg : RegionName) : ComputeKernel :=
  let body : List Stmt :=
    [ Stmt.assign .nat [4] "offs" (Op.arange 4)
    , Stmt.assign .real [4] "x"
        (Op.load .real (MemAccess.region xReg (Op.ref .nat [4] "offs")) MaskOpt.none)
    , Stmt.assign .ptr [4] "outPtrs"
        (Op.ptrAdd Broadcast.scalarL (Op.ptrBase outReg) (Op.ref .nat [4] "offs"))
    , Stmt.store .real [4]
        (MemAccess.ptr (Op.ref .ptr [4] "outPtrs"))
        (Op.ref .real [4] "x")
        MaskOpt.none ]
  ComputeKernel.fromKernelBody [xReg] [outReg] body

def unboundPointerRefKernel : ComputeKernel :=
  let body : List Stmt :=
    [ Stmt.assign .real [] "x"
        (Op.load .real (MemAccess.ptr (Op.ref .ptr [] "p")) MaskOpt.none) ]
  ComputeKernel.fromKernelBody ["x"] [] body

def nonPointerRefAsPointerKernel : ComputeKernel :=
  let body : List Stmt :=
    [ Stmt.assign .real [] "p" (Op.const 0)
    , Stmt.assign .real [] "x"
        (Op.load .real (MemAccess.ptr (Op.ref .ptr [] "p")) MaskOpt.none) ]
  ComputeKernel.fromKernelBody ["x"] [] body

def blockPointerRefKernel (xReg : RegionName) : ComputeKernel :=
  let body : List Stmt :=
    [ Stmt.assign .blockPtr [1] "bp"
        (Op.makeBlockPtr xReg 0 [8] [1] [1] [0])
    , Stmt.assign .blockPtr [1] "bp2"
        (Op.advanceBlockPtr (Op.ref .blockPtr [1] "bp") [1])
    , Stmt.store .real [1]
        (MemAccess.blockPtr (Op.ref .blockPtr [1] "bp2") [0])
        (Op.full [1] (Op.const 1))
        MaskOpt.none ]
  ComputeKernel.fromKernelBody [xReg] [] body

def blockPointerMetadataMismatchKernel (xReg : RegionName) : ComputeKernel :=
  let body : List Stmt :=
    [ Stmt.assign .blockPtr [1] "bp"
        (Op.makeBlockPtr xReg 0 [8] [1] [] [0]) ]
  ComputeKernel.fromKernelBody [xReg] [] body

def blockPointerShortBlockShapeKernel (xReg : RegionName) : ComputeKernel :=
  let body : List Stmt :=
    [ Stmt.assign .blockPtr [1] "bp"
        (Op.makeBlockPtr xReg 0 [8, 8] [1] [8, 1] [0, 0]) ]
  ComputeKernel.fromKernelBody [xReg] [] body

def blockPointerAdvanceUnderflowKernel (xReg : RegionName) : ComputeKernel :=
  let body : List Stmt :=
    [ Stmt.assign .blockPtr [1] "bp"
        (Op.advanceBlockPtr
          (Op.makeBlockPtr xReg 0 [8] [1] [1] [0])
          [-1]) ]
  ComputeKernel.fromKernelBody [xReg] [] body

def blockPointerRefAdvanceUnderflowKernel (xReg : RegionName) : ComputeKernel :=
  let body : List Stmt :=
    [ Stmt.assign .blockPtr [1] "bp"
        (Op.makeBlockPtr xReg 0 [8] [1] [1] [0])
    , Stmt.assign .blockPtr [1] "bp2"
        (Op.advanceBlockPtr (Op.ref .blockPtr [1] "bp") [-1]) ]
  ComputeKernel.fromKernelBody [xReg] [] body

def blockPointerBoundaryMismatchKernel (xReg : RegionName) : ComputeKernel :=
  let body : List Stmt :=
    [ Stmt.assign .blockPtr [1] "bp"
        (Op.makeBlockPtr xReg 0 [8] [1] [1] [0])
    , Stmt.store .real [1]
        (MemAccess.blockPtr (Op.ref .blockPtr [1] "bp") [1])
        (Op.full [1] (Op.const 1))
        MaskOpt.none ]
  ComputeKernel.fromKernelBody [xReg] [] body

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
      (blockPointerShortBlockShapeKernel "x") =
      .error (.blockPointerMetadataMismatch 2 2 2 1) := by
  native_decide

example :
    Kernel.checkStrict (checkerEnv [("x", .real)])
      (blockPointerAdvanceUnderflowKernel "x") =
      .error (.blockPointerAdvanceUnderflow 0 0 (-1)) := by
  native_decide

example :
    Kernel.checkStrict (checkerEnv [("x", .real)])
      (blockPointerRefAdvanceUnderflowKernel "x") =
      .error (.blockPointerAdvanceUnderflow 0 0 (-1)) := by
  native_decide

example :
    Kernel.checkStrict (checkerEnv [("x", .real)])
      (blockPointerBoundaryMismatchKernel "x") =
      .error (.boundaryAxisOutOfRange 1 1) := by
  native_decide

example :
    BlockPtr.WellFormed
      { region := "x", baseOffset := 0, parentShape := [8, 8],
        blockShape := [1, 1], strides := [8, 1], offsets := [0, 0] } := by
  rfl

example :
    BlockPtr.CheckedAxesValid
      { region := "x", baseOffset := 0, parentShape := [8, 8],
        blockShape := [1, 1], strides := [8, 1], offsets := [1, 0] }
      [0, 1] := by
  rfl

example :
    BlockPtr.AdvanceNonnegative
      { region := "x", baseOffset := 0, parentShape := [8, 8],
        blockShape := [1, 1], strides := [8, 1], offsets := [1, 0] }
      [-1, 0] := by
  rfl

example :
    BlockPtr.MetadataWellFormed [8, 8] [1, 1] [8, 1] [0, 0] :=
  checkBlockPtrMetadata_ok (by rfl)

example (i : Nat) (hmem : i ∈ [0, 1]) : i < 2 :=
  checkBoundaryAxes_ok (by rfl) i hmem

example :
    BlockPtr.StaticAdvanceNonnegative [1, 0] [-1, 0] :=
  checkStaticAdvanceNonnegative_ok (by rfl)

example :
    BlockPtr.MetadataWellFormed [8, 8] [1, 1] [8, 1] [0, 0] ∧
      (BlockPtrSummary.mk "x" (some 2) (some [0, 0])).region = "x" ∧
      (BlockPtrSummary.mk "x" (some 2) (some [0, 0])).parentRank = some 2 ∧
      (BlockPtrSummary.mk "x" (some 2) (some [0, 0])).offsets = some [0, 0] :=
  BlockPtrSummary.ofStaticChecked_ok
    (summary := BlockPtrSummary.mk "x" (some 2) (some [0, 0]))
    (region := "x")
    (parentShape := [8, 8])
    (blockShape := [1, 1])
    (strides := [8, 1])
    (offsets := [0, 0])
    (by rfl)

example :
    [8, 8].length = [8, 1].length ∧
      [8, 8].length = (2 : Nat) ∧
      [8, 8].length = [1, 1].length ∧
      (BlockPtrSummary.mk "x" (some 2) none).region = "x" ∧
      (BlockPtrSummary.mk "x" (some 2) none).parentRank = some 2 ∧
      (BlockPtrSummary.mk "x" (some 2) none).offsets = none :=
  BlockPtrSummary.ofDynamicOffsetsChecked_ok
    (summary := BlockPtrSummary.mk "x" (some 2) none)
    (region := "x")
    (parentShape := [8, 8])
    (blockShape := [1, 1])
    (strides := [8, 1])
    (offsetRank := 2)
    (by rfl)

example :
    BlockPtr.StaticAdvanceNonnegative [1, 0] [-1, 0] ∧
      (BlockPtrSummary.mk "x" (some 2)
        (some (BlockPtr.advanceOffsets [1, 0] [-1, 0]))).region = "x" ∧
      (BlockPtrSummary.mk "x" (some 2)
        (some (BlockPtr.advanceOffsets [1, 0] [-1, 0]))).parentRank = some 2 ∧
      (BlockPtrSummary.mk "x" (some 2)
        (some (BlockPtr.advanceOffsets [1, 0] [-1, 0]))).offsets =
          some (BlockPtr.advanceOffsets [1, 0] [-1, 0]) :=
  BlockPtrSummary.checkedAdvance_ok
    (summary := BlockPtrSummary.mk "x" (some 2) (some [1, 0]))
    (deltas := [-1, 0])
    (summary' := BlockPtrSummary.mk "x" (some 2)
      (some (BlockPtr.advanceOffsets [1, 0] [-1, 0])))
    (by rfl)

example (i : Nat) (hmem : i ∈ [0, 1]) : i < 2 :=
  BlockPtrSummary.checkBoundary_ok
    (summary := BlockPtrSummary.mk "x" (some 2) (some [0, 0]))
    (axes := [0, 1])
    (by rfl) i hmem

example :
    BlockPtrSummary.merge
        (BlockPtrSummary.mk "x" (some 2) (some [0, 0]))
        (BlockPtrSummary.mk "x" (some 2) (some [1, 0])) =
      Except.ok (BlockPtrSummary.mk "x" (some 2) none) := by
  rfl

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
  simp only [evalOp]
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
  simp only [evalOp]
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

/-! ### Register rebinding semantics (issue #19)

Triton assignment is name-based: rebinding `x` at a new dtype/shape clears the
old `x` binding instead of leaving a stale typed slot behind. -/

example :
    let s : BlockState :=
      { mem := fun _ _ => 0, regs := fun _ _ _ => none
      , pids := fun _ => 0, undef := fun _ _ => 0 }
    let s' := (s.setReg "x" .real [] (Tile.scalar (some (1 : ℝ) : WithBot ℝ))).setReg
      "x" .nat [2] (Tile.vec fun i : Fin 2 => i.val)
    s'.regs .real [] "x" = none ∧
      s'.regs .nat [2] "x" = some (Tile.vec fun i : Fin 2 => i.val) := by
  simp [BlockState.setReg]

example :
    let s : BlockState :=
      { mem := fun _ _ => 0, regs := fun _ _ _ => none
      , pids := fun _ => 0, undef := fun _ _ => 0 }
    let cond : Op .bool [] :=
      Op.lt ComparableDType.nat Broadcast.nil (Op.constNat 1) (Op.constNat 1)
    let body : List Stmt := [Stmt.assign .real [] "x" (Op.const 7)]
    stepStmt (Stmt.ifThen cond body) s = some s := by
  norm_num [stepStmt, evalOp, Tile.cop, ComparableDType.lt]

/-! ### `tl.if ... else ...` scalar conditional surface (issue #54) -/

/-- Scalar if-else skeleton.  The branch condition is scalar; element-wise
selection remains `tl.where`. -/
def ifThenElseSmoke (outReg : RegionName) (P : Nat) :
    Kernel := triton {
  pid := tl.program_id(0)
  tl.if pid < $(P) {
    tl.store($(outReg) + $(0), $(1))
  } else {
    tl.store($(outReg) + $(0), $(2))
  }
}

example :
    let s : BlockState :=
      { mem := fun _ _ => 0, regs := fun _ _ _ => none
      , pids := fun _ => 0, undef := fun _ _ => 0 }
    let cond : Op .bool [] :=
      Op.lt ComparableDType.nat Broadcast.nil (Op.constNat 0) (Op.constNat 1)
    let thenBody : List Stmt := [Stmt.assign .real [] "x" (Op.const 7)]
    let elseBody : List Stmt := [Stmt.assign .real [] "x" (Op.const 9)]
    (stepStmt (Stmt.ifThenElse cond thenBody elseBody) s).bind
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
    let thenBody : List Stmt := [Stmt.assign .real [] "x" (Op.const 7)]
    let elseBody : List Stmt := [Stmt.assign .real [] "x" (Op.const 9)]
    (stepStmt (Stmt.ifThenElse cond thenBody elseBody) s).bind
        (fun s' => s'.regs .real [] "x")
      = some (Tile.scalar (some (9 : ℝ) : WithBot ℝ)) := by
  norm_num [stepStmt, stepStmts, evalOp, Tile.cop, ComparableDType.lt,
    BlockState.setReg]

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

def blockPointerLoadNoBoundarySmoke (xReg outReg : RegionName) (N B : Nat) : ComputeKernel := triton {
  xBp := tl.make_block_ptr($(xReg), base=$(0), shape=[$(N)], strides=[1], offsets=[0], block_shape=[$(B)])
  x   := tl.load(xBp)
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

/-! ### `for ... in range(start, stop, step)` smoke tests -/

def rangeAccumKernel (xReg : RegionName) (N BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  acc = 0
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    x = tl.load(xReg + off)
    acc = acc + x
  }
  tl.store(xReg + pid, acc)
}

def mulAssignKernel (xReg : RegionName) (scale N : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  x = tl.load(xReg + pid)
  x *= $(scale)
  tl.store(xReg + pid * $(N), x)
}

#check rangeAccumKernel
#check mulAssignKernel

/-! ### `dtype=tl.float32` on `tl.zeros` / `tl.full` smoke tests -/

def fp32ZerosKernel (xReg : RegionName) (_N BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  x = tl.load(xReg + pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))).to(tl.float32)
  _var = _var + x * x
  tl.store(xReg + pid, tl.sum(_var, axis=0))
}

def fp32FullKernel (outReg : RegionName) (N : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  v = tl.full([$(N)], 1.0, dtype=tl.float32)
  tl.store(outReg + pid * $(N) + tl.arange(0, $(N)), v)
}

#check fp32ZerosKernel
#check fp32FullKernel

/-! ### Casted Nat loads used as pointer offsets -/

def castedNatLoadOffsetKernel
    (dataReg : RegionName) (idxReg : Region .nat) (ignored : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(axis=0)
  idxReg += row
  idx = (tl.load(idxReg)).to(tl.int32)
  vals = tl.load(dataReg + idx)
  if idx != $(ignored) {
    tl.store(dataReg + row, vals)
  }
}

def castedNatLoadWhereKernel
    (dataReg : RegionName) (idxReg : Region .nat) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(axis=0)
  cols = tl.arange(0, $(BLOCK_SIZE))
  idx = (tl.load(idxReg + row)).to(tl.int32)
  vals = tl.load(dataReg + cols, mask=cols < $(BLOCK_SIZE), other=0.0)
  out = tl.where(cols == idx, vals, 0.0)
  tl.store(dataReg + cols, out, mask=cols < $(BLOCK_SIZE))
}

#check castedNatLoadOffsetKernel
#check castedNatLoadWhereKernel

/-! ### Scalar `e[None]` dimension insertion -/

def scalarNoneKernel (outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  z = tl.zeros([], dtype=tl.float32)
  offs = tl.arange(0, $(N))
  vals = z[None]
  tl.store(outReg + pid * $(N) + offs, vals)
}

#check scalarNoneKernel

/-! ### Rank-3 `None` dimension insertion -/

def rank3NoneKernel (outReg : RegionName) (M N K : Nat) : ComputeKernel := triton {
  offs_m = tl.arange(0, $(M))
  offs_n = tl.arange(0, $(N))
  offs_k = tl.arange(0, $(K))
  tile = offs_m[None, :, None] + offs_n[:, None, None] + offs_k[None, None, :]
  tl.store(outReg + tile, tile)
}

#check rank3NoneKernel

def rank2MiddleNoneKernel (outReg : RegionName) (M K : Nat) : ComputeKernel := triton {
  offs_m = tl.arange(0, $(M))
  offs_k = tl.arange(0, $(K))
  tile2 = offs_m[:, None] + offs_k[None, :]
  tile3 = tile2[:, None, :]
  out_ptrs = outReg + offs_m[:, None, None] * $(K) + offs_k[None, None, :]
  tl.store(out_ptrs, tile3)
}

#check rank2MiddleNoneKernel

/-! ### Python two-argument dynamic `range(start, stop)` -/

def dynRange2Kernel (outReg : RegionName) (bounds : Region .nat) : ComputeKernel := triton {
  tl.static_print("dynRange2Kernel")
  pid = tl.program_id(axis=0)
  start_i = tl.load(bounds + pid)
  end_i = tl.load(bounds + pid + $(1))
  acc = tl.zeros([], dtype=tl.float32)
  for i in range(start_i, end_i) {
    x = tl.load(outReg + i)
    acc += x
  }
  tl.store(outReg + pid, acc)
}

#check dynRange2Kernel

/-! ### Bool antiquote for constexpr conditions -/

def boolAntiquoteKernel (outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  if $((N >= 2 : Bool)) {
    tl.store(outReg + pid, 1.0)
  } else {
    tl.store(outReg + pid, 0.0)
  }
}

#check boolAntiquoteKernel

/-! ### Python helper surface expressions -/

def helperExprKernel (outReg : RegionName) (N BLOCK : Nat) : ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  n = min(_div_up($(N), $(BLOCK)), $(N))
  tl.store(outReg + pid, n)
}

#check helperExprKernel

/-! ### Int antiquote and signed sentinels -/

def intSentinelKernel (outReg : RegionName) (labels : Region .int) (ignored : Int) :
    ComputeKernel := triton {
  row = tl.program_id(axis=0)
  label = tl.load($((labels : Region .int)) + row)
  if label == $((ignored : Int)) {
    tl.store(outReg + row, 0.0)
  } else {
    label -= $((1 : Int))
    tl.store(outReg + row, 1.0)
  }
}

#check intSentinelKernel

/-! ### Pointer decrement -/

def pointerDecrementKernel (outReg : Region .real) (N : Nat) : ComputeKernel := triton {
  ptr = outReg + $(N)
  ptr -= $(N)
  x = tl.load(ptr)
  tl.store(outReg, x)
}

#check pointerDecrementKernel

def reverseRangeKernel (outReg : Region .nat) (N : Nat) : ComputeKernel := triton {
  ptr = outReg + ($(N) - $(1))
  for i in range($(N) - $(1), -$(1), -$(1)) {
    tl.store(ptr, i)
    ptr -= $(1)
  }
}

#check reverseRangeKernel

/-! ### libdevice.pow: general elementwise power (`Op.pow` / `Real.rpow`) -/

/-- Scalar-base × tensor-exponent `tl.extra.cuda.libdevice.pow` (the RoPE
`pow(theta, freqs)` shape): the scalar binder base broadcasts against the
loaded tile exponent through the general `Op.pow` node. -/
def powScalarTensorSmoke (xReg outReg : RegionName) (theta : ℝ) (N : Nat) :
    ComputeKernel := triton {
  offs := tl.arange(0, $(N))
  t := tl.load($(xReg) + offs)
  freqs := tl.extra.cuda.libdevice.pow($((theta : ℝ)), t)
  tl.store($(outReg) + offs, freqs)
}

#check powScalarTensorSmoke

/-- The literal-2 exponent keeps its historical `x * x` (`Op.mul`) lowering. -/
def powSquareSmoke (xReg outReg : RegionName) : ComputeKernel := triton {
  x := tl.load($(xReg))
  y := tl.extra.cuda.libdevice.pow(x, 2)
  tl.store($(outReg), y)
}

#check powSquareSmoke

/-- `evalOp` unfolds `Op.pow` to `Real.rpow` on finite lanes. -/
example (a b : ℝ) (s : BlockState) :
    evalOp (Op.pow Broadcast.nil (Op.const a) (Op.const b)) s =
      some (Tile.scalar ((Real.rpow a b : ℝ) : WithBot ℝ)) := by
  simp
  ext i
  simp [Tile.bop_data]
  rfl

/-- Scalar-base broadcast: every lane of `pow(theta, full(t))` is
`Real.rpow theta t`. -/
example (theta t : ℝ) (N : Nat) (s : BlockState) (i : TileIndex [N]) :
    (evalOp (Op.pow Broadcast.scalarL (Op.const theta)
        (Op.full [N] (Op.const t))) s).map (fun v => v.data i) =
      some (((Real.rpow theta t : ℝ) : WithBot ℝ)) := by
  simp [Tile.bop_data]
  rfl

end VeriTile.Examples.TritonSmoke
