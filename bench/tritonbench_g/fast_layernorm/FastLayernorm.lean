import VeriTile.Triton

/-!
# `fast_layernorm` — strict per-kernel correctness

`layernorm_forward` is the Unsloth fast LayerNorm forward: each program `row_idx`
normalizes one row of `X` by its mean and variance, applies affine scale `W` and
bias `b`, and writes the normalized row `Y = ((X - mean) * inv_var) * W + b`
together with the per-row reciprocal std `r = inv_var` and mean `mu`. A companion
`layernorm_backward` kernel is also transcribed and its `dX` output verified.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies, for one program (one row). The host launch (grid over rows,
the host-side `BLOCK_SIZE` choice, scheduling, and how the runtime composes
per-row writes into the buffers) is the *trusted boundary*, not a proof
obligation here. Because `row_idx = tl.program_id(0)` is universally quantified,
the per-program statement covers every row of the grid.

## Proof architecture

```
layernorm_forward_correctness                 ← TOP THEOREM (layernormForwardIO ⊨ forward triple)
  ├─ layernorm_forward_flattenOk              bridge fragment membership
  ├─ layernorm_forward_traceSafe              per-execution lane-wise safety walk
  └─ layernorm_forward_region_run             region-model masked Hoare triple
       ├─ layernorm_forward_exec_isSome       termination
       ├─ layernorm_forward_y_correct         ← masked Y store per lane
       ├─ layernorm_forward_inv_var_correct   ← scalar rstd store into r
       ├─ layernorm_forward_mean_correct      ← scalar mean store into mu
       ├─ layernormYSpec_congr /
       │  invVarFullSpec_congr /
       │  meanFullSpec_congr                  only active lanes feed the specs
       └─ layernorm_forward_frame             masked-scatter + scalar-store frame
layernorm_backward_dx_compute_correct         ← backward dX (separate kernel)
  └─ layernorm_backward_dx_correct
```

The headline is the two-axis masked Hoare-triple combinator
`layernormForwardIO … ⊨ f` (`Masked2DKernelIO₃ₓ₃.Implements`, three inputs /
three outputs): for every disjoint flat placement of the six buffers, every
program id all of whose active read lanes and write cells are in bounds, and
every launch state whose active input lanes hold `xs` (the `X` row), `ws` (the
weights) and `bs` (the bias), the translated pointer kernel terminates, every
active `Y` lane `j` holds `layernormYSpec n_cols BLOCK_SIZE eps xs ws bs j`,
the scalar cells `r[pid]` / `mu[pid]` hold `invVarFullSpec` / `meanFullSpec`,
and every other memory cell is unchanged.

There are additional proof-oriented store-slice theorems
(`layernorm_forward_inv_var_store_slice_*`, `layernorm_forward_mean_store_slice_*`)
that isolate individual scalar stores; the layernorm row math is defined inline
in this file rather than reusing `VeriTile.Triton.Math.RMSNorm`.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `.to(tl.float32)`
load casts reduce to identity at the algorithm layer (post-erasure all dtypes
unify to `ℝ`). The reductions `tl.sum(X_row) / n_cols` and `tl.sum(XX*XX) /
n_cols` sum over the *padded* `BLOCK_SIZE` block, but out-of-range lanes are
masked to `0` (load `other=0`), so each sum equals the logical row length
`n_cols`. The reciprocal std is `inv_var = rsqrt(row_var + eps) =
1 / sqrt(var + eps)`; the affine step is `(XX * inv_var) * W + b`. The scalar
stores `r` and `mu` are characterized via `WithBot.unbotD 0` wrappers
(`invVarFullSpec`, `meanFullSpec`). This is a single-block kernel: `BLOCK_SIZE`
covers the whole row in one pass, with the `col_offsets < n_cols` mask handling
the padded tail. The headline takes the three output buffers pairwise distinct
(`Y ≠ r`, `Y ≠ mu`, `r ≠ mu`; symmetric forms are derived) so each output's
readback sees through the other stores, and `0 < BLOCK_SIZE`: the two scalar
stores are **unconditional**, and the `⊨` interface carries their in-bounds and
frame obligations on the write-active lane `0`, which must exist. The grid is
1-D; the signature's second program-id axis is unused (windows and masks are
constant in `pid₁`).
-/

namespace VeriTile.Bench.TritonBenchG.FastLayernorm

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₃ₓ₃

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful transcription of `fast_layernorm.py`'s `layernorm_forward`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
def layernorm_forward
    (Y X W b r mu : RegionName)
    (Y_row_stride X_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)

  Y += row_idx * $(Y_row_stride)
  X += row_idx * $(X_row_stride)
  r += row_idx
  mu += row_idx

  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)

  mean_X = tl.sum(X_row, axis=0) / $(n_cols)
  XX = X_row - mean_X
  row_var = tl.sum(XX * XX, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + $(eps))
  tl.store(r, inv_var)
  tl.store(mu, mean_X)
  output = (XX * inv_var) * W_row + b_row
  tl.store(Y + col_offsets, output, mask=mask)
}

/-- Faithful transcription of `fast_layernorm.py`'s `layernorm_backward`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
def layernorm_backward
    (dY X W b r mu : RegionName)
    (dY_row_stride X_row_stride n_cols : Nat)
    (_eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)

  dY += row_idx * $(dY_row_stride)
  X += row_idx * $(X_row_stride)
  r += row_idx
  mu += row_idx

  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)

  inv_var = tl.load(r).to(tl.float32)
  mean = tl.load(mu).to(tl.float32)
  normed = (X_row - mean) * inv_var
  dY_W = dY_row * W_row
  dX_row = dY_W - tl.sum(dY_W, axis=0) / $(n_cols) -
    normed * tl.sum(dY_W * normed, axis=0) / $(n_cols)
  dX_row = dX_row * inv_var
  tl.store(dY + col_offsets, dX_row, mask=mask)
}

/-- Element `j` of **this program's row** of a row-major matrix region `R`
(row = `pid`, row stride `row_stride`): `R[pid·row_stride + j]`. The `X` and
`dY` row loads all use this layout. -/
noncomputable def rowElem (s : BlockState) (R : RegionName)
    (row_stride j : Nat) : ℝ :=
  s.readMem R (s.pid * row_stride + j)

/-- Masked input row tile: lane `j < n_cols` holds `xs j`, masked lanes are
`0`, matching `mask=…, other=0`. Pure in the row values `xs`. -/
noncomputable def layernormInputTile
    (n_cols BLOCK_SIZE : Nat) (xs : Fin BLOCK_SIZE → ℝ) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < n_cols then some (xs idx.1)
      else some (0 : ℝ) }

/-- `mean_X = tl.sum(X_row) / n_cols` over the masked row: masked lanes enter
as `0`, neutral for the sum, so the padded-block sum equals the sum over the
active prefix. -/
noncomputable def layernormMeanCarrier
    (n_cols BLOCK_SIZE : Nat) (xs : Fin BLOCK_SIZE → ℝ) :
    WithBot ℝ :=
  Option.map (fun a => a / (n_cols : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (layernormInputTile n_cols BLOCK_SIZE xs)).data PUnit.unit)

/-- `XX = X_row - mean_X`, lane-wise over the masked row tile. -/
noncomputable def layernormCenteredTile
    (n_cols BLOCK_SIZE : Nat) (xs : Fin BLOCK_SIZE → ℝ) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      Option.map₂ (fun x mean => x - mean)
        ((layernormInputTile n_cols BLOCK_SIZE xs).data idx)
        (layernormMeanCarrier n_cols BLOCK_SIZE xs) }

/-- `row_var = tl.sum(XX * XX) / n_cols`. -/
noncomputable def layernormVarCarrier
    (n_cols BLOCK_SIZE : Nat) (xs : Fin BLOCK_SIZE → ℝ) :
    WithBot ℝ :=
  Option.map (fun a => a / (n_cols : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (layernormCenteredTile n_cols BLOCK_SIZE xs)
        (layernormCenteredTile n_cols BLOCK_SIZE xs))).data PUnit.unit)

/-- `inv_var = tl.math.rsqrt(row_var + eps)`. -/
noncomputable def layernormInvVarCarrier
    (n_cols BLOCK_SIZE : Nat) (eps : ℝ) (xs : Fin BLOCK_SIZE → ℝ) :
    WithBot ℝ :=
  WithBot.realRsqrt
    (Option.map (fun a => a + eps)
      (layernormVarCarrier n_cols BLOCK_SIZE xs))

/-- Exact affine LayerNorm value computed by the kernel at lane `idx`, as a
pure function of the row `xs`, weights `ws` and bias `bs`:
`((x - mean) * inv_var) * w + b` with the row statistics threaded through
`layernormMeanCarrier` / `layernormInvVarCarrier`. -/
noncomputable def layernormYSpec
    (n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs ws bs : Fin BLOCK_SIZE → ℝ) (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun affine bias => affine + bias)
      (Option.map₂ (fun scaled w => scaled * w)
        (Option.map₂ (fun centered inv => centered * inv)
          (Option.map₂ (fun x mean => x - mean)
            (some (xs idx))
            (layernormMeanCarrier n_cols BLOCK_SIZE xs))
          (layernormInvVarCarrier n_cols BLOCK_SIZE eps xs))
        (some (ws idx)))
      (some (bs idx)))

def yOutOffset (s : BlockState) (Y_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * Y_row_stride + i.val

/-- Executed-state correctness for the `Y` output of `layernorm_forward`:
active lanes hold `layernormYSpec` of the loaded row/weights/bias, inactive
lanes are preserved. -/
theorem layernorm_forward_y_correct
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYr : Y ≠ r) (hYmu : Y ≠ mu)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i))
    (hExec : exec (layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
          n_cols eps BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Y (yOutOffset s Y_row_stride i) =
        if i.val < n_cols then
          layernormYSpec n_cols BLOCK_SIZE eps
            (fun j => rowElem s X X_row_stride j.val)
            (fun j => s.readMem W j.val)
            (fun j => s.readMem bias j.val) i
        else s.readMem Y (yOutOffset s Y_row_stride i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, layernorm_forward, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.sub, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, layernormYSpec, layernormInvVarCarrier, layernormVarCarrier,
            layernormCenteredTile, layernormMeanCarrier, layernormInputTile, rowElem,
            Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realRsqrt, NumericDType.mul]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYr, hYmu]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-! ### Backward (`layernorm_backward`) -/

/-- Per-lane gradient input tile from `dY`. -/
noncomputable def layernormBackwardDYTile
    (s : BlockState) (dY : RegionName)
    (dY_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * dY_row_stride + idx.1.val
      if idx.1.val < n_cols then some (s.readMem dY off) else some (0 : ℝ) }

/-- Per-lane scale (`W`) tile (broadcast row). -/
noncomputable def layernormBackwardWTile
    (s : BlockState) (W : RegionName) (n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < n_cols then some (s.readMem W idx.1.val) else some (0 : ℝ) }

/-- The forward kernel's already-stored `inv_var` / `mean` are reread by
backward; the spec treats them as values currently in memory. -/
noncomputable def layernormBackwardInvVar
    (s : BlockState) (r : RegionName) : ℝ :=
  s.readMem r s.pid

noncomputable def layernormBackwardMean
    (s : BlockState) (mu : RegionName) : ℝ :=
  s.readMem mu s.pid

/-- `(X - mean) * inv_var`, mirroring the kernel's masked-load → centered →
scaled chain. The false branch carries `(0 - mean) * inv_var` because the
kernel computes `(other=0 - mean) * inv_var` before any masking is reapplied,
so the spec must match for elementwise tile equality. -/
noncomputable def layernormBackwardNormedTile
    (s : BlockState) (X r mu : RegionName)
    (X_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * X_row_stride + idx.1.val
      Option.map
        ((fun a => a * layernormBackwardInvVar s r) ∘
          fun a => a - layernormBackwardMean s mu)
        (if idx.1.val < n_cols then some (s.readMem X off) else some (0 : ℝ)) }

/-- `dY_W = dY_row * W_row` per lane. -/
noncomputable def layernormBackwardDYWTile
    (s : BlockState) (dY W : RegionName)
    (dY_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
    (layernormBackwardDYTile s dY dY_row_stride n_cols BLOCK_SIZE)
    (layernormBackwardWTile s W n_cols BLOCK_SIZE)

/-- Sum-mean of `dY_W` across the row. -/
noncomputable def layernormBackwardSumDYWMean
    (s : BlockState) (dY W : RegionName)
    (dY_row_stride n_cols BLOCK_SIZE : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (n_cols : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (layernormBackwardDYWTile s dY W dY_row_stride n_cols BLOCK_SIZE)).data
        PUnit.unit)

/-- Sum-mean of `dY_W * normed`. -/
noncomputable def layernormBackwardSumDYWNormedMean
    (s : BlockState) (dY X W r mu : RegionName)
    (dY_row_stride X_row_stride n_cols BLOCK_SIZE : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (n_cols : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (layernormBackwardDYWTile s dY W dY_row_stride n_cols BLOCK_SIZE)
        (layernormBackwardNormedTile s X r mu X_row_stride n_cols
          BLOCK_SIZE))).data PUnit.unit)

/-- Per-lane `dX` value matching the Python backward kernel formula:
`((dY*W - mean(dY*W) - normed * mean(dY*W*normed)) * inv_var)`.

Structured to mirror the kernel's elementwise broadcast evaluation: the
sum-by-n_cols and scaling by `normed_i` are folded into the `Option.map`s
to match the kernel-produced term. -/
noncomputable def layernormDXSpec
    (s : BlockState) (dY X W r mu : RegionName)
    (dY_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  let dyw_i := rowElem s dY dY_row_stride i.val *
    s.readMem W i.val
  let normed_i_inv := (rowElem s X X_row_stride i.val -
    s.readMem mu s.pid) * s.readMem r s.pid
  WithBot.unbotD 0
    (Option.map (fun v => v * s.readMem r s.pid)
      (Option.map₂ (fun a b => a - b)
        (Option.map
          ((fun b => dyw_i - b) ∘ fun a => a / (n_cols : ℝ))
          ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
            (layernormBackwardDYWTile s dY W dY_row_stride n_cols
              BLOCK_SIZE)).data PUnit.unit))
        (Option.map
          ((fun a => a / (n_cols : ℝ)) ∘ fun b => normed_i_inv * b)
          ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
            (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
              (layernormBackwardDYWTile s dY W dY_row_stride n_cols BLOCK_SIZE)
              (layernormBackwardNormedTile s X r mu X_row_stride n_cols
                BLOCK_SIZE))).data PUnit.unit))))

def dXOutOffset (s : BlockState) (dY_row_stride : Nat) (i : Fin BLOCK_SIZE) :
    Nat :=
  s.pid * dY_row_stride + i.val

/-- Executed-state correctness for `layernorm_backward`'s in-place `dX` store.

Mirrors `layernorm_forward_y_correct`: `BLOCK_SIZE`-many lanes, masked write
into `dY + col_offsets`. The kernel reads `dY_row` from `dY` *before* writing
back, so `s.readMem dY ...` references the original `dY` values throughout
the spec. -/
theorem layernorm_backward_dx_correct
    (dY X W bias r mu : RegionName)
    (dY_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => dXOutOffset s dY_row_stride i))
    (hExec : exec (layernorm_backward dY X W bias r mu dY_row_stride
          X_row_stride n_cols eps BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem dY (dXOutOffset s dY_row_stride i) =
        if i.val < n_cols then
          layernormDXSpec s dY X W r mu dY_row_stride X_row_stride n_cols
            BLOCK_SIZE i
        else s.readMem dY (dXOutOffset s dY_row_stride i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, layernorm_backward, stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.sub, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [dXOutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, layernormDXSpec, rowElem, layernormBackwardDYWTile,
            layernormBackwardDYTile, layernormBackwardWTile,
            layernormBackwardNormedTile, layernormBackwardSumDYWMean,
            layernormBackwardSumDYWNormedMean, layernormBackwardInvVar,
            layernormBackwardMean,
            Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.mul]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for `layernorm_backward`'s in-place `dX` store. -/
theorem layernorm_backward_dx_compute_correct
    (dY X W bias r mu : RegionName)
    (dY_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => dXOutOffset s dY_row_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layernorm_backward dY X W bias r mu dY_row_stride
        X_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (dY, dXOutOffset s dY_row_stride i)))
      (expected := fun i =>
        layernormDXSpec s dY X W r mu dY_row_stride X_row_stride n_cols
          BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layernorm_backward, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layernorm_backward_dx_correct dY X W bias r mu dY_row_stride
    X_row_stride n_cols BLOCK_SIZE eps s s' hOutInj hExec i
  simpa [hActive] using h

/-- Full-kernel spec for the `inv_var` (rstd) store of `layernorm_forward`.

Wraps `layernormInvVarCarrier` with `WithBot.unbotD 0` so that the readback
of the kernel's scalar write into `r` matches the carrier's value as a
plain `ℝ`. -/
noncomputable def invVarFullSpec
    (n_cols BLOCK_SIZE : Nat) (eps : ℝ) (xs : Fin BLOCK_SIZE → ℝ) : ℝ :=
  WithBot.unbotD 0
    (layernormInvVarCarrier n_cols BLOCK_SIZE eps xs)

/-- Full-kernel spec for the `mean` (mu) store of `layernorm_forward`.

Wraps `layernormMeanCarrier` with `WithBot.unbotD 0` so that the readback
of the kernel's scalar write into `mu` matches the carrier's value as a
plain `ℝ`. -/
noncomputable def meanFullSpec
    (n_cols BLOCK_SIZE : Nat) (xs : Fin BLOCK_SIZE → ℝ) : ℝ :=
  WithBot.unbotD 0
    (layernormMeanCarrier n_cols BLOCK_SIZE xs)

/-- Executed-state correctness for the `inv_var` (rstd) scalar store of
`layernorm_forward`. The kernel writes a single element at offset `s.pid`
into `r`; this theorem strips the trailing `mu` writeMem and the `Y` foldl
to expose that store. -/
theorem layernorm_forward_inv_var_correct
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hRY : r ≠ Y) (hRmu : r ≠ mu)
    (hExec : exec (layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
          n_cols eps BLOCK_SIZE) s = some s') :
    s'.readMem r s.pid =
      invVarFullSpec n_cols BLOCK_SIZE eps
        (fun j => rowElem s X X_row_stride j.val) := by
  simp [exec, layernorm_forward, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRY]
  simp only [BlockState.setReg_readMem]
  rw [BlockState.writeMem_readMem_of_ne_region _ mu _ _ _ _ hRmu]
  rw [BlockState.writeMem_readMem]
  simp only [BlockState.pid_eq, and_self, if_true]
  simp [invVarFullSpec, layernormInvVarCarrier, layernormVarCarrier,
        layernormCenteredTile, layernormMeanCarrier, layernormInputTile, rowElem,
        Tile.bop, Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.mul,
        WithBot.realRsqrt]
  rfl

/-- Executed-state correctness for the `mean` (mu) scalar store of
`layernorm_forward`. The kernel writes a single element at offset `s.pid`
into `mu`; this theorem strips the trailing `Y` foldl to expose that store.
The `r` writeMem is beneath the `mu` writeMem at a different region, so it
is irrelevant once the readback resolves at `mu`. -/
theorem layernorm_forward_mean_correct
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hMuY : mu ≠ Y)
    (hExec : exec (layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
          n_cols eps BLOCK_SIZE) s = some s') :
    s'.readMem mu s.pid =
      meanFullSpec n_cols BLOCK_SIZE
        (fun j => rowElem s X X_row_stride j.val) := by
  simp [exec, layernorm_forward, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hMuY]
  simp only [BlockState.setReg_readMem]
  rw [BlockState.writeMem_readMem]
  simp only [BlockState.pid_eq, and_self, if_true]
  simp only [meanFullSpec, layernormMeanCarrier, layernormInputTile, rowElem,
        Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex]
  rfl

/-- Proof-oriented inv_var (rstd) store slice of `fast_layernorm.py`'s
`layernorm_forward`. Takes a precomputed `InvVarPre` scalar (per row) and
proves the scalar writeback into `r`. -/
def layernorm_forward_inv_var_store_slice
    (InvVarPre r : RegionName) : ComputeKernel := triton {
  row_idx = tl.program_id(0)
  inv_var = tl.load(InvVarPre + row_idx)
  tl.store(r + row_idx, inv_var)
}

def rOutOffset (s : BlockState) : Nat := s.pid

noncomputable def invVarStoreSpec (s : BlockState) (InvVarPre : RegionName) : ℝ :=
  s.readMem InvVarPre s.pid

theorem layernorm_forward_inv_var_store_slice_correct
    (InvVarPre r : RegionName) (s s' : BlockState)
    (hExec : exec (layernorm_forward_inv_var_store_slice InvVarPre r) s = some s') :
    s'.readMem r (rOutOffset s) = invVarStoreSpec s InvVarPre := by
  simp [exec, layernorm_forward_inv_var_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add] at hExec
  subst s'
  simp [rOutOffset, invVarStoreSpec]

theorem layernorm_forward_inv_var_store_slice_compute_correct
    (InvVarPre r : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layernorm_forward_inv_var_store_slice InvVarPre r)
      (initialState := s)
      (write := fun _ : PUnit => some (r, rOutOffset s))
      (expected := fun _ => invVarStoreSpec s InvVarPre) := by
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layernorm_forward_inv_var_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layernorm_forward_inv_var_store_slice_correct InvVarPre r s s' hExec

/-- Proof-oriented mean (mu) store slice of `fast_layernorm.py`'s
`layernorm_forward`. Same scalar-copy pattern. -/
def layernorm_forward_mean_store_slice
    (MeanPre mu : RegionName) : ComputeKernel := triton {
  row_idx = tl.program_id(0)
  mean_X = tl.load(MeanPre + row_idx)
  tl.store(mu + row_idx, mean_X)
}

noncomputable def meanStoreSpec (s : BlockState) (MeanPre : RegionName) : ℝ :=
  s.readMem MeanPre s.pid

theorem layernorm_forward_mean_store_slice_correct
    (MeanPre mu : RegionName) (s s' : BlockState)
    (hExec : exec (layernorm_forward_mean_store_slice MeanPre mu) s = some s') :
    s'.readMem mu (rOutOffset s) = meanStoreSpec s MeanPre := by
  simp [exec, layernorm_forward_mean_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add] at hExec
  subst s'
  simp [rOutOffset, meanStoreSpec]

theorem layernorm_forward_mean_store_slice_compute_correct
    (MeanPre mu : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layernorm_forward_mean_store_slice MeanPre mu)
      (initialState := s)
      (write := fun _ : PUnit => some (mu, rOutOffset s))
      (expected := fun _ => meanStoreSpec s MeanPre) := by
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layernorm_forward_mean_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layernorm_forward_mean_store_slice_correct MeanPre mu s s' hExec

/-! ### The `⊨` specification (forward) -/

/-- The masked input tile only reads the active lanes: rows agreeing below
`n_cols` yield the same tile (masked lanes are `0` either way). -/
theorem layernormInputTile_congr (n_cols BLOCK_SIZE : Nat)
    (xs xs' : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → xs j = xs' j) :
    layernormInputTile n_cols BLOCK_SIZE xs
      = layernormInputTile n_cols BLOCK_SIZE xs' := by
  unfold layernormInputTile
  congr 1
  funext idx
  by_cases hj : idx.1.val < n_cols
  · simp only [if_pos hj, hx idx.1 hj]
  · simp only [if_neg hj]

/-- `layernormYSpec` at an **active** lane only reads the active lanes of its
inputs: rows/weights/biases agreeing below `n_cols` yield the same value. -/
theorem layernormYSpec_congr (n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs xs' ws ws' bs bs' : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → xs j = xs' j)
    (hw : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → ws j = ws' j)
    (hb : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → bs j = bs' j)
    (i : Fin BLOCK_SIZE) (hi : i.val < n_cols) :
    layernormYSpec n_cols BLOCK_SIZE eps xs ws bs i
      = layernormYSpec n_cols BLOCK_SIZE eps xs' ws' bs' i := by
  unfold layernormYSpec layernormInvVarCarrier layernormVarCarrier
    layernormCenteredTile layernormMeanCarrier
  rw [layernormInputTile_congr n_cols BLOCK_SIZE xs xs' hx,
      hx i hi, hw i hi, hb i hi]

/-- `invVarFullSpec` only reads the active lanes of the row. -/
theorem invVarFullSpec_congr (n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs xs' : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → xs j = xs' j) :
    invVarFullSpec n_cols BLOCK_SIZE eps xs
      = invVarFullSpec n_cols BLOCK_SIZE eps xs' := by
  unfold invVarFullSpec layernormInvVarCarrier layernormVarCarrier
    layernormCenteredTile layernormMeanCarrier
  rw [layernormInputTile_congr n_cols BLOCK_SIZE xs xs' hx]

/-- `meanFullSpec` only reads the active lanes of the row. -/
theorem meanFullSpec_congr (n_cols BLOCK_SIZE : Nat)
    (xs xs' : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → xs j = xs' j) :
    meanFullSpec n_cols BLOCK_SIZE xs = meanFullSpec n_cols BLOCK_SIZE xs' := by
  unfold meanFullSpec layernormMeanCarrier
  rw [layernormInputTile_congr n_cols BLOCK_SIZE xs xs' hx]

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked loads with `other`, dtype casts, sum reductions, `rsqrt`,
scalar stores, masked store). -/
theorem layernorm_forward_flattenOk
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols : Nat) (eps : ℝ) (BLOCK_SIZE : Nat) :
    ((layernorm_forward Y X W bias r mu Y_row_stride X_row_stride n_cols eps
        BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [layernorm_forward, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Termination: the kernel executes to completion from any state — including
`BLOCK_SIZE = 0`, since the only reductions are `sum`s (total on empty axes). -/
private theorem layernorm_forward_exec_isSome
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState) :
    ∃ s1, exec ((layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
        n_cols eps BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, layernorm_forward, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked `Y` store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-- Frame half: every memory cell not actively written by the three stores —
the masked `Y` row scatter and the two scalar cells `r[pid]` / `mu[pid]` — is
preserved by the run. -/
private theorem layernorm_forward_frame
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s s1 : BlockState)
    (hExec : exec ((layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
        n_cols eps BLOCK_SIZE).toAlgKernel) s = some s1)
    (r' : RegionName) (o' : Nat)
    (hmissY : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(Y = r' ∧ s.pids 0 * Y_row_stride + i.val = o'))
    (hmissR : ¬(r = r' ∧ s.pids 0 = o'))
    (hmissMu : ¬(mu = r' ∧ s.pids 0 = o')) :
    s1.mem r' o' = s.mem r' o' := by
  simp [exec, layernorm_forward, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r' o' _ _ ?_) ?_
  · intro k _ hmk hc
    exact hmissY k.1 (by simpa using hmk) hc
  · show (BlockState.writeMem _ mu _ _).mem r' o' = s.mem r' o'
    rw [BlockState.writeMem_mem,
        if_neg (fun hc => hmissMu ⟨hc.1.symm, hc.2.symm⟩)]
    show (BlockState.writeMem _ r _ _).mem r' o' = s.mem r' o'
    rw [BlockState.writeMem_mem,
        if_neg (fun hc => hmissR ⟨hc.1.symm, hc.2.symm⟩)]
    rfl

/-- Per-execution safety walk: one computational unfold walks all the
statements — the pointer/mask/index staging, the reductions and register
arithmetic are memory-silent — and reduces the six memory accesses (masked
row load of `X`, masked loads of `W` and `b`, unconditional scalar stores to
`r` and `mu`, masked row store to `Y`) to the **lane-wise** bounds
hypotheses. -/
theorem layernorm_forward_traceSafe
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (bounds : RegionBounds) (s : BlockState)
    (hin1 : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pids 0 * X_row_stride + j.val < bounds X)
    (hin2 : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → j.val < bounds W)
    (hin3 : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → j.val < bounds bias)
    (hout1 : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pids 0 * Y_row_stride + j.val < bounds Y)
    (hout2 : s.pids 0 < bounds r)
    (hout3 : s.pids 0 < bounds mu) :
    Kernel.TraceSafe bounds
      ((layernorm_forward Y X W bias r mu Y_row_stride X_row_stride n_cols eps
        BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [layernorm_forward, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
    FloatDType.toWithBot,
    Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin1 a ha, fun a ha => hin2 a ha, fun a ha => hin3 a ha,
    hout2, hout3, fun a ha => hout1 a ha⟩

/-- **The region-model masked Hoare triple** — termination, active-lane `Y`
values, the two scalar output cells, and frame off the written cells, from any
launch state whose active input lanes hold `xs` (the `X` row), `ws` (the
weights) and `bs` (the bias). This is the `hrun` obligation of the `⊨`
headline. The output-distinctness hypotheses let each output's readback see
through the other stores. -/
theorem layernorm_forward_region_run
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (hYr : Y ≠ r) (hYmu : Y ≠ mu) (hRmu : r ≠ mu)
    (s₀ : BlockState) (xs ws bs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem X (s₀.pids 0 * X_row_stride + j.val) = xs j)
    (hw : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → s₀.readMem W j.val = ws j)
    (hb : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → s₀.readMem bias j.val = bs j) :
    ∃ s1, exec ((layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
          n_cols eps BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem Y (s₀.pids 0 * Y_row_stride + j.val)
            = layernormYSpec n_cols BLOCK_SIZE eps xs ws bs j)
      ∧ s1.readMem r (s₀.pids 0) = invVarFullSpec n_cols BLOCK_SIZE eps xs
      ∧ s1.readMem mu (s₀.pids 0) = meanFullSpec n_cols BLOCK_SIZE xs
      ∧ (∀ r' o',
          (r' ≠ Y ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o' ≠ s₀.pids 0 * Y_row_stride + j.val) →
          (r' ≠ r ∨ o' ≠ s₀.pids 0) →
          (r' ≠ mu ∨ o' ≠ s₀.pids 0) →
          s1.mem r' o' = s₀.mem r' o') := by
  have hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s₀ Y_row_stride i) := by
    intro a b h
    simp only [yOutOffset] at h
    exact Fin.ext (Nat.add_left_cancel h)
  obtain ⟨s1, hs1⟩ := layernorm_forward_exec_isSome Y X W bias r mu
    Y_row_stride X_row_stride n_cols BLOCK_SIZE eps s₀
  refine ⟨s1, hs1, fun j hj => ?_, ?_, ?_, fun r' o' hY hR hMu => ?_⟩
  · have h := layernorm_forward_y_correct Y X W bias r mu Y_row_stride
      X_row_stride n_cols BLOCK_SIZE eps s₀ s1 hYr hYmu hOutInj hs1 j
    rw [if_pos hj] at h
    exact h.trans (layernormYSpec_congr n_cols BLOCK_SIZE eps _ xs _ ws _ bs
      (fun k hk => hx k hk) (fun k hk => hw k hk) (fun k hk => hb k hk) j hj)
  · exact (layernorm_forward_inv_var_correct Y X W bias r mu Y_row_stride
      X_row_stride n_cols BLOCK_SIZE eps s₀ s1 (Ne.symm hYr) hRmu hs1).trans
      (invVarFullSpec_congr n_cols BLOCK_SIZE eps _ xs (fun k hk => hx k hk))
  · exact (layernorm_forward_mean_correct Y X W bias r mu Y_row_stride
      X_row_stride n_cols BLOCK_SIZE eps s₀ s1 (Ne.symm hYmu) hs1).trans
      (meanFullSpec_congr n_cols BLOCK_SIZE _ xs (fun k hk => hx k hk))
  · refine layernorm_forward_frame Y X W bias r mu Y_row_stride X_row_stride
      n_cols BLOCK_SIZE eps s₀ s1 hs1 r' o' ?_ ?_ ?_
    · rintro i hi ⟨hr, ho⟩
      rcases hY with hne | hno
      · exact hne hr.symm
      · exact hno i hi ho.symm
    · rintro ⟨hr, ho⟩
      rcases hR with hne | hno
      · exact hne hr.symm
      · exact hno ho.symm
    · rintro ⟨hr, ho⟩
      rcases hMu with hne | hno
      · exact hne hr.symm
      · exact hno ho.symm

/-- `layernorm_forward`'s masked three-input / three-output **IO signature** —
the whole kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`in3` — the input matrix `X`, the per-column weights `W`, the
  per-column bias `b`;
* `out1`/`out2`/`out3` — the output matrix `Y`, the per-row reciprocal-std
  vector `r`, the per-row mean vector `mu`;
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`write1` — **per-lane row windows**: program `pid` reads its `X` row
  at `pid * X_row_stride + j` and writes its `Y` row at
  `pid * Y_row_stride + j` (the host-side one-program-per-row launch
  convention);
* `read2`/`read3` — the weight and bias windows are **absolute and
  pid-independent**: every program reads `W[j]` / `b[j]`;
* `write2`/`write3` — the **scalar** store cells `r[pid]` / `mu[pid]`, the
  same for every lane;
* `mask` — the active lanes `j < n_cols`, the same for every program: the row
  prefix that actually exists in the matrix. All three loads and the `Y` store
  share it (`read2Mask`/`read3Mask`/`writeMask1` keep their defaults);
* `writeMask2`/`writeMask3` — lane `0` carries each scalar; the other lanes
  are write-inactive and carry no obligations on either side.

The grid is 1-D, so the second program-id axis is an unused parameter: windows
and masks are constant in `pid₁` (the headline's `∀ pid₁` quantification is
vacuous but honest). The windows and masks are declared, not parsed from the
kernel; the headline **proves** the kernel's actual addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes. -/
def layernormForwardIO (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : Masked2DKernelIO₃ₓ₃ where
  kernel := layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
    n_cols eps BLOCK_SIZE
  in1 := X
  in2 := W
  in3 := bias
  out1 := Y
  out2 := r
  out3 := mu
  B := BLOCK_SIZE
  read1 := fun pid _ j => pid * X_row_stride + j.val
  read2 := fun _ _ j => j.val
  read3 := fun _ _ j => j.val
  write1 := fun pid _ j => pid * Y_row_stride + j.val
  write2 := fun pid _ _ => pid
  write3 := fun pid _ _ => pid
  mask := fun _ _ j => j.val < n_cols
  writeMask2 := fun _ _ j => j.val = 0
  writeMask3 := fun _ _ j => j.val = 0

/-- **The headline**: `layernorm_forward` implements the exact affine LayerNorm
triple over the active row prefix on its masked three-input / three-output IO
signature — for every disjoint flat placement of the six buffers, every
program id whose active lanes and scalar store cells are in bounds, and every
launch state whose active input lanes hold `xs` (the `X` row), `ws` (the
weights) and `bs` (the bias), the translated pointer kernel terminates, every
active `Y` lane `j` holds
`layernormYSpec n_cols BLOCK_SIZE eps xs ws bs j = ((x - mean)·inv_var)·w + b`,
the scalar cells `r[pid]` / `mu[pid]` hold
`invVarFullSpec n_cols BLOCK_SIZE eps xs = rsqrt(var + eps)` and
`meanFullSpec n_cols BLOCK_SIZE xs = sum(x)/n_cols`, and every other memory
cell is unchanged. `0 < BLOCK_SIZE` is required: the two scalar stores are
unconditional, and the interface carries their in-bounds/frame obligations on
the write-active lane `0`, which must exist. The output buffers must be
pairwise distinct (`Y ≠ r`, `Y ≠ mu`, `r ≠ mu`) so each readback sees through
the other stores. Proof: `Masked2DKernelIO₃ₓ₃.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
specification layernorm_forward_correctness
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (hB : 0 < BLOCK_SIZE)
    (hYr : Y ≠ r) (hYmu : Y ≠ mu) (hRmu : r ≠ mu) :
    layernormForwardIO Y X W bias r mu Y_row_stride X_row_stride n_cols eps
        BLOCK_SIZE ⊨
      fun _ _ xs ws bs =>
        (fun i => layernormYSpec n_cols BLOCK_SIZE eps xs ws bs i,
         fun _ => invVarFullSpec n_cols BLOCK_SIZE eps xs,
         fun _ => meanFullSpec n_cols BLOCK_SIZE xs) := by
  refine Masked2DKernelIO₃ₓ₃.Implements.intro _ ?_ ?_ ?_
  · exact layernorm_forward_flattenOk Y X W bias r mu Y_row_stride
      X_row_stride n_cols eps BLOCK_SIZE
  · intro bounds s h1 h2 h3 h4 h5 h6
    exact layernorm_forward_traceSafe Y X W bias r mu Y_row_stride
      X_row_stride n_cols BLOCK_SIZE eps bounds s h1 h2 h3 h4
      (h5 ⟨0, hB⟩ rfl) (h6 ⟨0, hB⟩ rfl)
  · intro s₀ xs ws bs hx hw hb
    obtain ⟨s1, hexec, hvalY, hvalR, hvalMu, hframe⟩ :=
      layernorm_forward_region_run Y X W bias r mu Y_row_stride X_row_stride
        n_cols BLOCK_SIZE eps hYr hYmu hRmu s₀ xs ws bs hx hw hb
    refine ⟨s1, hexec, hvalY, fun j _ => hvalR, fun j _ => hvalMu,
      fun r' o' h1 h2 h3 => ?_⟩
    refine hframe r' o' h1 ?_ ?_
    · rcases h2 with hne | hno
      · exact Or.inl hne
      · exact Or.inr (hno ⟨0, hB⟩ rfl)
    · rcases h3 with hne | hno
      · exact Or.inl hne
      · exact Or.inr (hno ⟨0, hB⟩ rfl)

end VeriTile.Bench.TritonBenchG.FastLayernorm
