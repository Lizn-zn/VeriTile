/-
VeriTile.Examples.Common

Shared example-level helpers for the parameterized-region kernel correctness
pattern. The reusable memory-contract layer (`InputAt`, `Offset`,
`TensorView`) lives in `VeriTile.Triton.Memory`; this file keeps only the
ergonomic shorthands used by worked examples.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Memory
import VeriTile.Triton.Correctness
import VeriTile.Triton.Float.StepR

namespace VeriTile.Examples

open VeriTile.Triton

/-- Cast `Fin k` to `Fin n` when `k ≤ n`.

Used by prefix-style proofs that compare a first-`k` recurrence against a
length-`n` input vector. -/
def castFin {n k : Nat} (h : k ≤ n) (i : Fin k) : Fin n :=
  ⟨i.val, lt_of_lt_of_le i.isLt h⟩

/-- Offset of lane `i` in the current program's contiguous 1-D tile:
`s.pid * N + i.val` — the address underlying the `InputLoadedAt` /
`outWritesTo` / `InputCellsLoadedAt` contracts, exposed as a shared name for
worked examples to use in place of a local address def. `@[reducible]` so it
is defeq to the raw expression the contracts inline. -/
@[reducible] def programLaneOffset (s : BlockState) (N : Nat) (i : Fin N) : Nat :=
  s.pid * N + i.val

/-- Region `region` holds tile `xs` at offsets `[pid*N, pid*N + N - 1]`. -/
def InputLoadedAt (s : BlockState) (region : RegionName)
    (N : Nat) (xs : Fin N → ℝ) : Prop :=
  ∀ i : Fin N, s.readMem region (s.pid * N + i.val) = xs i

/-- All listed inputs are loaded (the list form of `InputLoadedAt`):
`InputsLoaded s N [(xReg, xs), (yReg, ys)]` says region `xReg` holds tile
`xs` and region `yReg` holds tile `ys`, each at the current program's
offsets. Specifications take one `InputsLoaded` hypothesis instead of one
`InputLoadedAt` line per input. -/
def InputsLoaded (s : BlockState) (N : Nat) :
    List (RegionName × (Fin N → ℝ)) → Prop
  | [] => True
  | p :: ps => InputLoadedAt s p.1 N p.2 ∧ InputsLoaded s N ps

@[simp] theorem inputsLoaded_nil (s : BlockState) (N : Nat) :
    InputsLoaded s N [] := trivial

@[simp] theorem inputsLoaded_cons (s : BlockState) (N : Nat)
    (p : RegionName × (Fin N → ℝ)) (ps : List (RegionName × (Fin N → ℝ))) :
    InputsLoaded s N (p :: ps) ↔
      InputLoadedAt s p.1 N p.2 ∧ InputsLoaded s N ps := Iff.rfl

/-- Write map for the current program's masked 1-D output tile: lane `i`
writes `region` at `s.pid * N + i.val`, active exactly when the column bound
admits it (`s.pid * N + i.val < cols`). The output-side counterpart of
`InputLoadedAt`. -/
def outWritesTo (s : BlockState) (region : RegionName)
    (cols N : Nat) : ComputeCorrect.WriteMap (Fin N) :=
  ComputeCorrect.WriteMap.writeIf
    (fun i : Fin N => s.pid * N + i.val < cols)
    (fun i => (region, s.pid * N + i.val))

/-- A memory cell carrying the real value `v` at the narrow-float dtype `dt`.
Narrow-float stores are only observable as their typed `MemCell` (the real
decode is tag-exact), so `MemCell`-layer contracts speak of these cells rather
than raw reals. Cased on `dt` so the `TileCarrier` (all `WithBot ℝ` for float
channels) reduces at each dtype. -/
def floatCell : FloatDType → ℝ → MemCell
  | .real, v => MemCell.of .real (some v)
  | .fp32, v => MemCell.of .fp32 (some v)
  | .fp16, v => MemCell.of .fp16 (some v)
  | .bf16, v => MemCell.of .bf16 (some v)

/-- `MemCell`-layer counterpart of `InputLoadedAt`: the current program's active
lanes (`s.pid * N + i.val < cols`) of `region` hold the narrow-float cells
`floatCell dt (vs i)`. This is the contract a downstream kernel places on a
`dt`-typed intermediate tensor materialized by an upstream kernel (e.g. the
bf16 scratch between two fused-vs-unfused pipeline stages). -/
def InputCellsLoadedAt (s : BlockState) (region : RegionName)
    (cols N : Nat) (dt : FloatDType) (vs : Fin N → ℝ) : Prop :=
  ∀ i : Fin N, s.pid * N + i.val < cols →
    s.mem region (s.pid * N + i.val) = floatCell dt (vs i)

/-- Transfer a one-dimensional loaded tile across states when the consumer
state has the same `pid` and agrees with the producer on the loaded region. -/
theorem InputLoadedAt.transfer
    {sSrc sDst : BlockState} {region : RegionName}
    {N : Nat} {xs : Fin N → ℝ}
    (hPid : sDst.pid = sSrc.pid)
    (hMem : ∀ offset, sDst.readMem region offset = sSrc.readMem region offset)
    (hLoaded : InputLoadedAt sSrc region N xs) :
    InputLoadedAt sDst region N xs := by
  intro i
  calc
    sDst.readMem region (sDst.pid * N + i.val)
        = sSrc.readMem region (sDst.pid * N + i.val) := hMem _
    _ = sSrc.readMem region (sSrc.pid * N + i.val) := by rw [hPid]
    _ = xs i := hLoaded i

/-- Canonical 1D tensor view for the tile owned by the current
`program_id(0)`: offsets `[s.pid * N, s.pid * N + N)`.

This is the view-level counterpart of `InputLoadedAt`. It is intentionally an
example helper; general kernels should expose their own `TensorView`s or
layout bundles. -/
def programTileView (s : BlockState) (region : RegionName) (N : Nat) :
    TensorView [N] :=
  { region := region, base := s.pid * N, strides := [1] }

/-- Feature-vector view at offsets `[0, N)`. -/
def featureView (region : RegionName) (N : Nat) : TensorView [N] :=
  { region := region, base := 0, strides := [1] }

/-- Row tile view at offsets `[s.pid * rowStride, s.pid * rowStride + N)`. -/
def rowTileView (s : BlockState) (region : RegionName)
    (rowStride N : Nat) : TensorView [N] :=
  { region := region, base := s.pid * rowStride, strides := [1] }

/-- Scalar cell view at a single memory offset. -/
def scalarCellView (region : RegionName) (base : Nat) : TensorView [] :=
  { region := region, base := base, strides := [] }

theorem inputLoadedAt_of_programTileView_loaded
    {s : BlockState} {region : RegionName} {N : Nat} {xs : Fin N → ℝ}
    (h : TensorView.loaded s (programTileView s region N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    InputLoadedAt s region N xs := by
  intro i
  have hi := h (i, PUnit.unit)
  simpa [TensorView.loaded, InputAt, TensorView.offset,
         programTileView, Offset.strided] using hi

/-- Region `region` holds a feature vector at offsets `[0, N)`.
    Used for per-column parameters such as LayerNorm `γ` and `β`, which are
    shared across rows and are loaded with `tl.arange(0, N)`, not
    `pid * N + tl.arange(0, N)`. -/
def InputFeatureLoadedAt (s : BlockState) (region : RegionName)
    (N : Nat) (xs : Fin N → ℝ) : Prop :=
  ∀ i : Fin N, s.readMem region i.val = xs i

theorem inputFeatureLoadedAt_of_featureView_loaded
    {s : BlockState} {region : RegionName} {N : Nat} {xs : Fin N → ℝ}
    (h : TensorView.loaded s (featureView region N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    InputFeatureLoadedAt s region N xs := by
  intro i
  have hi := h (i, PUnit.unit)
  simpa [TensorView.loaded, InputAt, TensorView.offset,
         featureView, Offset.strided] using hi

/-- Read region `region` at the cell `pid*N + i.val` from the optional
    final `BlockState` of an `exec` call. -/
noncomputable def observeAt
    (sf : Option BlockState) (region : RegionName)
    (N : Nat) (basePid : Nat) (i : Fin N) : Option ℝ :=
  sf.map (·.readMem region (basePid * N + i.val))

/-! ## Row-wise (2D-style) variants -/

/-- Region `region` holds the `blockSize`-cell tile `xs` at offsets
    `[s.pid * rowStride, s.pid * rowStride + blockSize)`. -/
def InputRowLoadedAt (s : BlockState) (region : RegionName)
    (rowStride blockSize : Nat) (xs : Fin blockSize → ℝ) : Prop :=
  ∀ i : Fin blockSize, s.readMem region (s.pid * rowStride + i.val) = xs i

theorem inputRowLoadedAt_of_rowTileView_loaded
    {s : BlockState} {region : RegionName}
    {rowStride blockSize : Nat} {xs : Fin blockSize → ℝ}
    (h : TensorView.loaded s (rowTileView s region rowStride blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1)) :
    InputRowLoadedAt s region rowStride blockSize xs := by
  intro i
  have hi := h (i, PUnit.unit)
  simpa [TensorView.loaded, InputAt, TensorView.offset,
         rowTileView, Offset.strided] using hi

/-- Read region `region` at the single cell `basePid` from the optional
    final `BlockState` of an `exec` call. Models the
    *single-scalar-per-block* output pattern (`tl.store(tl.ptr($(yReg)) + row, _)`)
    distinct from the tile-scatter pattern observed by `observeAt`. -/
noncomputable def observeRowAt
    (sf : Option BlockState) (region : RegionName) (basePid : Nat) : Option ℝ :=
  sf.map (·.readMem region basePid)

/-! ## Backwards-compatible 1D injectivity helper -/

/-- Stride injectivity for the canonical 1D scatter offset
`fun idx : TileIndex [n] => base + idx.1.val`.

Used by every 1D kernel proof feeding `scatter_readback_nd` /
`scatter_readback_prop_masked_nd` after the ND switch. Equivalent to
`Offset.linear1D_inj`; kept as a separate name for the existing call sites. -/
theorem injective_offset_singleton {n : Nat} (base : Nat) :
    Function.Injective (fun idx : TileIndex [n] => base + idx.1.val) := by
  rintro ⟨a, _⟩ ⟨b, _⟩ hab
  obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
  rfl

/-! ## Shared online-Welford loop body

The raw-AST loop body of the online-Welford `tl.for` loop, shared by the
Welford and LayerNorm showcases (`bench/examples/`) and
`VeriTile.Examples.WelfordKernels`. Its per-iteration effect is characterized
per consumer (against that file's loop invariant); the body itself and its
cast-free degeneration are consumer-independent. -/

/-- Online Welford loop body: `xi := load x[pid*blockSize + i]`,
`delta := xi − M`, `M += delta/(i+1)`, `delta2 := xi − M`, `S += delta·delta2`. -/
def onlineWelfordLoopBody (xReg : RegionName) (blockSize : Nat) : List Stmt :=
  [Stmt.assign .real [] "xi"
      (Op.load .real (MemAccess.region xReg
        (Op.add .nat .nil
          (Op.mul .nat .nil (Op.ref .nat [] "pid")
            (Op.constNat blockSize))
          (Op.ref .nat [] "i"))) MaskOpt.none),
    Stmt.assign .real [] "delta"
      (Op.sub .real .nil (Op.ref .real [] "xi")
        (Op.ref .real [] "M")),
    Stmt.assign .real [] "M"
      (Op.add .real .nil (Op.ref .real [] "M")
        (Op.div .real .nil (Op.ref .real [] "delta")
          (Op.add .real .nil (Op.ref .nat [] "i").natToReal
            (Op.const 1)))),
    Stmt.assign .real [] "delta2"
      (Op.sub .real .nil (Op.ref .real [] "xi")
        (Op.ref .real [] "M")),
    Stmt.assign .real [] "S"
      (Op.add .real .nil (Op.ref .real [] "S")
        (Op.mul .real .nil (Op.ref .real [] "delta")
          (Op.ref .real [] "delta2")))]

/-- The online Welford loop body is cast-free: it steps identically under
`execR R` and `exec`. Feed to `stepForLoopAuxR_castFree` to degenerate the
whole loop. -/
theorem onlineWelfordLoopBody_castFree (R : RoundingModel)
    (xReg : RegionName) (blockSize : Nat) (t : BlockState) :
    stepStmtsR R (onlineWelfordLoopBody xReg blockSize) t
      = stepStmts (onlineWelfordLoopBody xReg blockSize) t := by
  simp only [onlineWelfordLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp]
  rfl

end VeriTile.Examples
