import VeriTile.Triton

/-!
# `fast_rms_layernorm` — strict per-kernel correctness

`_rms_layernorm_forward` is the Unsloth fast RMSNorm forward: each program
`row_idx` normalizes one row of `X` by its root-mean-square, scales by per-column
weights `W`, and writes `Y = (X * inv_var) * W` together with the per-row
reciprocal std `r = inv_var`. A `_gemma_rms_layernorm_forward` variant uses the
`(W + 1)` Gemma scaling, and a `_rms_layernorm_backward` kernel (plain and Gemma)
is also transcribed and its `dY` gradient verified.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies, for one program (one row). The host launch (grid over rows
`(n_rows,)`, the host-side `BLOCK_SIZE` choice via `calculate_settings`, the
`GEMMA` heuristic dispatch, scheduling, and how the runtime composes per-row
writes into the buffers) is the *trusted boundary*, not a proof obligation here.
Because `row_idx = tl.program_id(0)` is universally quantified, the per-program
statement covers every row of the grid.

## Proof architecture

```
rms_layernorm_forward_correctness             ← TOP THEOREM (rmsLayernormFwdIO ⊨ (Y-spec, rstd-spec))
  ├─ rms_layernorm_forward_flattenOk          bridge fragment membership
  ├─ rms_layernorm_forward_traceSafe          per-execution lane-wise safety walk
  └─ rms_layernorm_forward_region_run         region-model masked Hoare triple
       ├─ rms_layernorm_forward_exec_isSome   termination
       ├─ rms_layernorm_forward_y_correct     ← masked Y store readback
       ├─ rms_layernorm_forward_inv_var_correct  ← scalar rstd store into r
       ├─ rmsLayernormYSpec_eq_of_loaded / rmsInvVarSpec_eq_of_loaded  pure-spec bridges
       └─ rms_layernorm_forward_frame         masked-scatter + scalar-store frame
gemma_rms_layernorm_forward_correctness       ← TOP THEOREM (Gemma forward, analogous)
rms_layernorm_backward_dy_compute_correct     ← backward dY (plain)
gemma_rms_layernorm_backward_dy_compute_correct  ← backward dY (Gemma)
```

The two forward headlines are the masked two-output Hoare-triple combinator
`rmsLayernormFwdIO … ⊨ f` / `gemmaRmsLayernormFwdIO … ⊨ f`
(`Masked2DKernelIO₂ₓ₂.Implements`): for every disjoint flat placement of the
four buffers (`X`, `W`, `Y`, `r`), every program id whose active lanes and
scalar rstd cell are in bounds, and every launch state whose active input lanes
hold `xs` (the `X` row) and `ws` (the weights), the translated pointer kernel
terminates, every active `Y` lane holds the pure spec `rmsFwdYSpec` /
`gemmaRmsFwdYSpec`, the scalar cell `r[row_idx·r_row_stride]` holds
`rmsFwdInvVarSpec`, and every other memory cell is unchanged. The RMS row math
(`rmsInputTile`, `rmsSumCarrier`, `rmsInvVarCarrier`, and their pure `rmsFwd*`
reparametrizations) is defined inline in this file rather than reusing
`VeriTile.Triton.Math.RMSNorm`.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `.to(tl.float32)` load
casts and the `normed.to(W_row.dtype)` cast reduce to identity at the algorithm
layer (post-erasure all dtypes unify to `ℝ`). The reduction
`tl.sum(X_row * X_row) / n_cols` sums over the *padded* `BLOCK_SIZE` block, but
out-of-range lanes are masked to `0` (load `other=0`), so the sum equals the
logical row length `n_cols`. The reciprocal std is
`inv_var = rsqrt(meanSq + eps) = 1 / sqrt(meanSq + eps)` (`rmsInvVarCarrier`);
the affine step is `(X * inv_var) * W` (plain) or `(X * inv_var) * (W + 1)`
(Gemma). The scalar `r` store is characterized via `rmsInvVarSpec`
(`WithBot.unbotD 0` of the carrier). This is a single-block kernel: `BLOCK_SIZE`
covers the whole row in one pass, with the `col_offsets < n_cols` mask handling
the padded tail. Output/output disjointness (`Y ≠ r`) is assumed: the scalar
rstd store must not alias the row store. `0 < BLOCK_SIZE` is genuinely forced:
the `r` store is **unmasked** in the kernel, so its safety bound and single-cell
frame exclusion are carried by the lane-0 write gate (`writeMask2`), which needs
at least one lane. The grid is 1-D; the headline signature's second program-id
axis is unused (windows and masks are constant in `pid₁`). The
`@triton.heuristics` GEMMA dispatch is modeled by the two separate forward
kernels rather than a runtime branch. `@triton.autotune` is not modeled.
-/

namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂ₓ₂

set_option linter.unusedSimpArgs false

set_option maxHeartbeats 5000000

/-- Faithful transcription of `fast_rms_layernorm.py`'s
`_rms_layernorm_forward`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
def rms_layernorm_forward
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)

  Y += row_idx * $(Y_row_stride)
  X += row_idx * $(X_row_stride)
  r += row_idx * $(r_row_stride)

  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets * $(W_row_stride), mask=mask, other=0)

  row_var = tl.sum(X_row * X_row, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + $(eps))
  tl.store(r, inv_var)
  normed = X_row * inv_var
  normed = (normed).to(W_row.dtype)
  output = normed * W_row
  tl.store(Y + col_offsets, output, mask=mask)
}

/-- Faithful transcription of `fast_rms_layernorm.py`'s
`_gemma_rms_layernorm_forward`.

The Python kernel accepts `W_row_stride` but loads `W + col_offsets`, so this
surface preserves that stride-free weight access. -/
def gemma_rms_layernorm_forward
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)

  Y += row_idx * $(Y_row_stride)
  X += row_idx * $(X_row_stride)
  r += row_idx * $(r_row_stride)

  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

  row_var = tl.sum(X_row * X_row, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + $(eps))
  tl.store(r, inv_var)
  normed = X_row * inv_var
  output = normed * (W_row + 1.0)

  tl.store(Y + col_offsets, output, mask=mask)
}

/-- Faithful transcription of `fast_rms_layernorm.py`'s
`_rms_layernorm_backward` for `GEMMA = false`.

Documented constexpr branch specialization: the Python kernel branches on
`GEMMA`; this surface covers the `GEMMA = false` branch.

The Python kernel accepts `dW`/`dW_row_stride` but does not write `dW`; this
surface preserves the in-place `dY` writeback. -/
def rms_layernorm_backward
    (dY X W r _dW : RegionName)
    (dY_row_stride X_row_stride _W_row_stride r_row_stride _dW_row_stride
      n_cols : Nat)
    (_eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)

  dY += row_idx * $(dY_row_stride)
  X += row_idx * $(X_row_stride)
  r += row_idx * $(r_row_stride)

  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

  inv_var = tl.load(r).to(tl.float32)
  normed = X_row * inv_var
  dY_W = dY_row * W_row

  rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
  output = inv_var / $(n_cols) * ($(n_cols) * dY_W - normed * rowsum_dY_normed)
  tl.store(dY + col_offsets, output, mask=mask)
}

/-- Faithful transcription of `fast_rms_layernorm.py`'s
`_rms_layernorm_backward` for `GEMMA = true`.

Documented constexpr branch specialization: the Python kernel branches on
`GEMMA`; this surface covers the `GEMMA = true` branch. -/
def gemma_rms_layernorm_backward
    (dY X W r _dW : RegionName)
    (dY_row_stride X_row_stride _W_row_stride r_row_stride _dW_row_stride
      n_cols : Nat)
    (_eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)

  dY += row_idx * $(dY_row_stride)
  X += row_idx * $(X_row_stride)
  r += row_idx * $(r_row_stride)

  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

  inv_var = tl.load(r).to(tl.float32)
  normed = X_row * inv_var
  dY_W = dY_row * (W_row + 1.0)

  rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
  output = inv_var / $(n_cols) * ($(n_cols) * dY_W - normed * rowsum_dY_normed)
  tl.store(dY + col_offsets, output, mask=mask)
}

/-- Element `j` of **this program's row** of a row-major matrix region `R`
(row = `pid`, row stride `row_stride`): `R[pid·row_stride + j]`. The `X` and
`dY` row loads all use this layout. -/
noncomputable def rowElem (s : BlockState) (R : RegionName)
    (row_stride j : Nat) : ℝ :=
  s.readMem R (s.pid * row_stride + j)

noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < n_cols then some (rowElem s X X_row_stride idx.1.val)
      else some (0 : ℝ) }

noncomputable def rmsSumCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (rmsInputTile s X X_row_stride n_cols BLOCK_SIZE)
      (rmsInputTile s X X_row_stride n_cols BLOCK_SIZE))).data PUnit.unit

noncomputable def rmsInvVarCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  WithBot.realRsqrt
    (Option.map ((fun a => a + eps) ∘ fun a => a / (n_cols : ℝ))
      (rmsSumCarrier s X X_row_stride n_cols BLOCK_SIZE))

noncomputable def rmsInvVarSpec
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) : ℝ :=
  WithBot.unbotD 0 (rmsInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps)

noncomputable def rmsLayernormYSpec
    (s : BlockState) (X W : RegionName)
    (X_row_stride W_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun x w => x * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (rowElem s X X_row_stride idx.val))
        (rmsInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps))
      (some (s.readMem W (idx.val * W_row_stride))))

noncomputable def gemmaRmsLayernormYSpec
    (s : BlockState) (X W : RegionName)
    (X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * (w + 1.0))
      (Option.map₂ (fun x inv => x * inv)
        (some (rowElem s X X_row_stride idx.val))
        (rmsInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps))
      (some (s.readMem W idx.val)))

noncomputable def rmsBackwardDYTilePlain
    (s : BlockState) (dY W : RegionName)
    (dY_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      Option.map₂ (fun dy w => dy * w)
        (if idx.1.val < n_cols then
          some (rowElem s dY dY_row_stride idx.1.val)
        else some (0 : ℝ))
        (if idx.1.val < n_cols then some (s.readMem W idx.1.val)
        else some (0 : ℝ)) }

noncomputable def rmsBackwardDYTileGemma
    (s : BlockState) (dY W : RegionName)
    (dY_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      Option.map₂ (fun dy w => dy * w)
        (if idx.1.val < n_cols then
          some (rowElem s dY dY_row_stride idx.1.val)
        else some (0 : ℝ))
        (Option.map (fun w => w + 1.0)
          (if idx.1.val < n_cols then some (s.readMem W idx.1.val)
          else some (0 : ℝ))) }

noncomputable def rmsBackwardNormedTile
    (s : BlockState) (X r : RegionName)
    (X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      Option.map (fun x => x * s.readMem r (s.pid * r_row_stride))
        (if idx.1.val < n_cols then
          some (rowElem s X X_row_stride idx.1.val)
        else some (0 : ℝ)) }

noncomputable def rmsBackwardRowSumCarrierPlain
    (s : BlockState) (dY X W r : RegionName)
    (dY_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (rmsBackwardDYTilePlain s dY W dY_row_stride n_cols BLOCK_SIZE)
      (rmsBackwardNormedTile s X r X_row_stride r_row_stride
        n_cols BLOCK_SIZE))).data PUnit.unit

noncomputable def rmsBackwardRowSumCarrierGemma
    (s : BlockState) (dY X W r : RegionName)
    (dY_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (rmsBackwardDYTileGemma s dY W dY_row_stride n_cols BLOCK_SIZE)
      (rmsBackwardNormedTile s X r X_row_stride r_row_stride
        n_cols BLOCK_SIZE))).data PUnit.unit

noncomputable def rmsBackwardDYSpecPlain
    (s : BlockState) (dY X W r : RegionName)
    (dY_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map
      ((fun b => s.readMem r (s.pid * r_row_stride) / (n_cols : ℝ) * b) ∘
        (fun b => (n_cols : ℝ) *
          (s.readMem dY (s.pid * dY_row_stride + idx.val) *
            s.readMem W idx.val) - b) ∘
        (fun b =>
          s.readMem X (s.pid * X_row_stride + idx.val) *
          s.readMem r (s.pid * r_row_stride) * b))
      (rmsBackwardRowSumCarrierPlain s dY X W r dY_row_stride X_row_stride
        r_row_stride n_cols BLOCK_SIZE))

noncomputable def rmsBackwardDYSpecGemma
    (s : BlockState) (dY X W r : RegionName)
    (dY_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map
      ((fun b => s.readMem r (s.pid * r_row_stride) / (n_cols : ℝ) * b) ∘
        (fun b => (n_cols : ℝ) *
          (s.readMem dY (s.pid * dY_row_stride + idx.val) *
            (s.readMem W idx.val + 1.0)) - b) ∘
        (fun b =>
          s.readMem X (s.pid * X_row_stride + idx.val) *
          s.readMem r (s.pid * r_row_stride) * b))
      (rmsBackwardRowSumCarrierGemma s dY X W r dY_row_stride X_row_stride
        r_row_stride n_cols BLOCK_SIZE))

def dyOutOffset (s : BlockState) (dY_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * dY_row_stride + i.val

def yOutOffset (s : BlockState) (Y_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * Y_row_stride + i.val

/-- Executed-state correctness for the `Y` output of `_rms_layernorm_forward`. -/
theorem rms_layernorm_forward_y_correct
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hRegions : Y ≠ r)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i))
    (hExec : exec (rms_layernorm_forward Y X W r Y_row_stride X_row_stride
          W_row_stride r_row_stride n_cols eps BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Y (yOutOffset s Y_row_stride i) =
        if i.val < n_cols then
          rmsLayernormYSpec s X W X_row_stride W_row_stride n_cols BLOCK_SIZE eps i
        else s.readMem Y (yOutOffset s Y_row_stride i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, rms_layernorm_forward, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, rmsLayernormYSpec, rmsInvVarCarrier, rmsSumCarrier,
            rmsInputTile, rowElem, Tile.reduceSum, Tile.reduceSumDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realRsqrt, NumericDType.mul]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hRegions]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Executed-state correctness for the `Y` output of
`_gemma_rms_layernorm_forward`. -/
theorem gemma_rms_layernorm_forward_y_correct
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hRegions : Y ≠ r)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i))
    (hExec : exec (gemma_rms_layernorm_forward Y X W r Y_row_stride X_row_stride
          r_row_stride n_cols eps BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Y (yOutOffset s Y_row_stride i) =
        if i.val < n_cols then
          gemmaRmsLayernormYSpec s X W X_row_stride n_cols BLOCK_SIZE eps i
        else s.readMem Y (yOutOffset s Y_row_stride i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, gemma_rms_layernorm_forward, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, gemmaRmsLayernormYSpec, rmsInvVarCarrier, rmsSumCarrier,
            rmsInputTile, rowElem, Tile.reduceSum, Tile.reduceSumDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realRsqrt, NumericDType.mul]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hRegions]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Executed-state correctness for the in-place `dY` output of
`_rms_layernorm_backward` with `GEMMA = false`. -/
theorem rms_layernorm_backward_dy_correct
    (dY X W r _dW : RegionName)
    (dY_row_stride X_row_stride _W_row_stride r_row_stride _dW_row_stride
      n_cols BLOCK_SIZE : Nat)
    (_eps : ℝ) (s s' : BlockState)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => dyOutOffset s dY_row_stride i))
    (hExec : exec (rms_layernorm_backward dY X W r _dW dY_row_stride X_row_stride
          _W_row_stride r_row_stride _dW_row_stride n_cols _eps BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem dY (dyOutOffset s dY_row_stride i) =
        if i.val < n_cols then
          rmsBackwardDYSpecPlain s dY X W r dY_row_stride X_row_stride
            r_row_stride n_cols BLOCK_SIZE i
        else s.readMem dY (dyOutOffset s dY_row_stride i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, rms_layernorm_backward, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.mul,
          NumericDType.sub, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [dyOutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, rmsBackwardDYSpecPlain, rmsBackwardRowSumCarrierPlain,
            rmsBackwardDYTilePlain, rmsBackwardNormedTile, rowElem, Tile.reduceSum,
            Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex, NumericDType.mul]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the in-place `dY` output of
`_rms_layernorm_backward` with `GEMMA = false`. -/
theorem rms_layernorm_backward_dy_compute_correct
    (dY X W r _dW : RegionName)
    (dY_row_stride X_row_stride _W_row_stride r_row_stride _dW_row_stride
      n_cols BLOCK_SIZE : Nat)
    (_eps : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => dyOutOffset s dY_row_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_layernorm_backward dY X W r _dW dY_row_stride X_row_stride
        _W_row_stride r_row_stride _dW_row_stride n_cols _eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (dY, dyOutOffset s dY_row_stride i)))
      (expected := fun i =>
        rmsBackwardDYSpecPlain s dY X W r dY_row_stride X_row_stride
          r_row_stride n_cols BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rms_layernorm_backward, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rms_layernorm_backward_dy_correct dY X W r _dW dY_row_stride X_row_stride
    _W_row_stride r_row_stride _dW_row_stride n_cols BLOCK_SIZE _eps s s'
    hOutInj hExec i
  simpa [hActive] using h

/-- Executed-state correctness for the in-place `dY` output of
`_rms_layernorm_backward` with `GEMMA = true`. -/
theorem gemma_rms_layernorm_backward_dy_correct
    (dY X W r _dW : RegionName)
    (dY_row_stride X_row_stride _W_row_stride r_row_stride _dW_row_stride
      n_cols BLOCK_SIZE : Nat)
    (_eps : ℝ) (s s' : BlockState)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => dyOutOffset s dY_row_stride i))
    (hExec : exec (gemma_rms_layernorm_backward dY X W r _dW dY_row_stride X_row_stride
          _W_row_stride r_row_stride _dW_row_stride n_cols _eps BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem dY (dyOutOffset s dY_row_stride i) =
        if i.val < n_cols then
          rmsBackwardDYSpecGemma s dY X W r dY_row_stride X_row_stride
            r_row_stride n_cols BLOCK_SIZE i
        else s.readMem dY (dyOutOffset s dY_row_stride i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, gemma_rms_layernorm_backward, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.sub, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [dyOutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, rmsBackwardDYSpecGemma, rmsBackwardRowSumCarrierGemma,
            rmsBackwardDYTileGemma, rmsBackwardNormedTile, rowElem, Tile.reduceSum,
            Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex, NumericDType.mul]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the in-place `dY` output of
`_rms_layernorm_backward` with `GEMMA = true`. -/
theorem gemma_rms_layernorm_backward_dy_compute_correct
    (dY X W r _dW : RegionName)
    (dY_row_stride X_row_stride _W_row_stride r_row_stride _dW_row_stride
      n_cols BLOCK_SIZE : Nat)
    (_eps : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => dyOutOffset s dY_row_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := gemma_rms_layernorm_backward dY X W r _dW dY_row_stride X_row_stride
        _W_row_stride r_row_stride _dW_row_stride n_cols _eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (dY, dyOutOffset s dY_row_stride i)))
      (expected := fun i =>
        rmsBackwardDYSpecGemma s dY X W r dY_row_stride X_row_stride
          r_row_stride n_cols BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [gemma_rms_layernorm_backward, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := gemma_rms_layernorm_backward_dy_correct dY X W r _dW dY_row_stride X_row_stride
    _W_row_stride r_row_stride _dW_row_stride n_cols BLOCK_SIZE _eps s s'
    hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented inv_var (rstd) store slice of `fast_rms_layernorm.py`'s
`_rms_layernorm_forward`. Takes a precomputed `InvVarPre` scalar and proves
the strided scalar writeback into `r` at offset `row_idx * r_row_stride`. -/
def rms_layernorm_forward_inv_var_store_slice
    (InvVarPre r : RegionName) (r_row_stride : Nat) : ComputeKernel := triton {
  row_idx = tl.program_id(0)
  inv_var = tl.load(InvVarPre + row_idx * $(r_row_stride))
  tl.store(r + row_idx * $(r_row_stride), inv_var)
}

def rOutOffset (s : BlockState) (r_row_stride : Nat) : Nat :=
  s.pid * r_row_stride

noncomputable def invVarStoreSpec (s : BlockState) (InvVarPre : RegionName)
    (r_row_stride : Nat) : ℝ :=
  s.readMem InvVarPre (rOutOffset s r_row_stride)

theorem rms_layernorm_forward_inv_var_store_slice_correct
    (InvVarPre r : RegionName) (r_row_stride : Nat)
    (s s' : BlockState)
    (hExec : exec (rms_layernorm_forward_inv_var_store_slice InvVarPre r r_row_stride)
      s = some s') :
    s'.readMem r (rOutOffset s r_row_stride) =
      invVarStoreSpec s InvVarPre r_row_stride := by
  simp [exec, rms_layernorm_forward_inv_var_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul] at hExec
  subst s'
  simp [rOutOffset, invVarStoreSpec]

theorem rms_layernorm_forward_inv_var_store_slice_compute_correct
    (InvVarPre r : RegionName) (r_row_stride : Nat) (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_layernorm_forward_inv_var_store_slice InvVarPre r r_row_stride)
      (initialState := s)
      (write := fun _ : PUnit => some (r, rOutOffset s r_row_stride))
      (expected := fun _ => invVarStoreSpec s InvVarPre r_row_stride) := by
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rms_layernorm_forward_inv_var_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact rms_layernorm_forward_inv_var_store_slice_correct InvVarPre r r_row_stride s s' hExec

/-! ### Full-kernel rstd correctness

Per #139, the slice proofs above rely on a precomputed `InvVarPre` region and
are not sufficient as full-kernel guarantees. The following theorems close the
`r` (rstd / inv_var) store for the full `_rms_layernorm_forward` and
`_gemma_rms_layernorm_forward` kernels by stripping the trailing `Y` write
foldl with `BlockState.foldl_writeMem_const_region_prop_masked_readMem_other`
(requiring `r ≠ Y`) and reducing the scalar `r` writeMem with `simp`. -/

/-- Executed-state correctness for the `r` (rstd / inv_var) output of
`_rms_layernorm_forward`. -/
theorem rms_layernorm_forward_inv_var_correct
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hRegions : r ≠ Y)
    (hExec : exec (rms_layernorm_forward Y X W r Y_row_stride X_row_stride
          W_row_stride r_row_stride n_cols eps BLOCK_SIZE) s = some s') :
    s'.readMem r (rOutOffset s r_row_stride) =
      rmsInvVarSpec s X X_row_stride n_cols BLOCK_SIZE eps := by
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, rms_layernorm_forward, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
          Y _ _ _ _ _ _ _ hRegions]
    simp [rOutOffset, rmsInvVarSpec, rmsInvVarCarrier, rmsSumCarrier,
          rmsInputTile, rowElem, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          WithBot.realRsqrt, NumericDType.mul]
    rfl
  · simp [exec, rms_layernorm_forward, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    have hB' : BLOCK_SIZE = 0 := Nat.eq_zero_of_not_pos hB
    subst hB'
    simp [TileShape.allIndices, List.finRange,
          rOutOffset, rmsInvVarSpec, rmsInvVarCarrier, rmsSumCarrier,
          rmsInputTile, rowElem, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          WithBot.realRsqrt, NumericDType.mul]

/-- Executed-state correctness for the `r` (rstd / inv_var) output of
`_gemma_rms_layernorm_forward`. -/
theorem gemma_rms_layernorm_forward_inv_var_correct
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hRegions : r ≠ Y)
    (hExec : exec (gemma_rms_layernorm_forward Y X W r Y_row_stride X_row_stride
          r_row_stride n_cols eps BLOCK_SIZE) s = some s') :
    s'.readMem r (rOutOffset s r_row_stride) =
      rmsInvVarSpec s X X_row_stride n_cols BLOCK_SIZE eps := by
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, gemma_rms_layernorm_forward, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
          Y _ _ _ _ _ _ _ hRegions]
    simp [rOutOffset, rmsInvVarSpec, rmsInvVarCarrier, rmsSumCarrier,
          rmsInputTile, rowElem, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          WithBot.realRsqrt, NumericDType.mul]
    rfl
  · simp [exec, gemma_rms_layernorm_forward, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    have hB' : BLOCK_SIZE = 0 := Nat.eq_zero_of_not_pos hB
    subst hB'
    simp [TileShape.allIndices, List.finRange,
          rOutOffset, rmsInvVarSpec, rmsInvVarCarrier, rmsSumCarrier,
          rmsInputTile, rowElem, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          WithBot.realRsqrt, NumericDType.mul]

/-! ### The `⊨` specifications

The headlines below restate the two forward kernels' correctness on the
flat-memory `Masked2DKernelIO₂ₓ₂` surface. The specs are **pure** functions of
the loaded row `xs` and weights `ws` (`rmsFwd*` below); the bridges
`rms*Spec_eq_of_loaded` connect them to the `BlockState`-reading carriers the
readback lemmas above are stated with. -/

/-- Pure masked input row tile: lane `j < n_cols` holds `xs j`, masked lanes
are `0` (matching `mask=…, other=0`). The `xs`-reparametrized form of
`rmsInputTile`. -/
noncomputable def rmsFwdInputTile (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < n_cols then some (xs idx.1) else some (0 : ℝ) }

/-- Pure `sum(X_row * X_row)` over the masked row (masked lanes enter as `0`,
neutral for the sum). -/
noncomputable def rmsFwdSumCarrier (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) : WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (rmsFwdInputTile n_cols BLOCK_SIZE xs)
      (rmsFwdInputTile n_cols BLOCK_SIZE xs))).data PUnit.unit

/-- Pure `inv_var = rsqrt(sum(x*x)/n_cols + eps)` carrier. -/
noncomputable def rmsFwdInvVarCarrier (n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs : Fin BLOCK_SIZE → ℝ) : WithBot ℝ :=
  WithBot.realRsqrt
    (Option.map ((fun a => a + eps) ∘ fun a => a / (n_cols : ℝ))
      (rmsFwdSumCarrier n_cols BLOCK_SIZE xs))

/-- Pure per-row rstd value stored to `r`: `WithBot.unbotD 0` of the
`rsqrt` carrier. -/
noncomputable def rmsFwdInvVarSpec (n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs : Fin BLOCK_SIZE → ℝ) : ℝ :=
  WithBot.unbotD 0 (rmsFwdInvVarCarrier n_cols BLOCK_SIZE eps xs)

/-- Pure per-lane `Y` value of the plain forward: `(x · inv_var) · w`. -/
noncomputable def rmsFwdYSpec (n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs ws : Fin BLOCK_SIZE → ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun x w => x * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (xs i))
        (rmsFwdInvVarCarrier n_cols BLOCK_SIZE eps xs))
      (some (ws i)))

/-- Pure per-lane `Y` value of the Gemma forward: `(x · inv_var) · (w + 1)`. -/
noncomputable def gemmaRmsFwdYSpec (n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs ws : Fin BLOCK_SIZE → ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * (w + 1.0))
      (Option.map₂ (fun x inv => x * inv)
        (some (xs i))
        (rmsFwdInvVarCarrier n_cols BLOCK_SIZE eps xs))
      (some (ws i)))

/-- The masked input tile only reads the **active** lanes of the `X` row: if
those lanes hold `xs`, the `BlockState` tile is the pure tile of `xs` (masked
lanes are `0` on both sides). -/
theorem rmsInputTile_eq_of_loaded
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.readMem X (s.pid * X_row_stride + j.val) = xs j) :
    rmsInputTile s X X_row_stride n_cols BLOCK_SIZE
      = rmsFwdInputTile n_cols BLOCK_SIZE xs := by
  unfold rmsInputTile rmsFwdInputTile rowElem
  congr 1
  funext idx
  by_cases hj : idx.1.val < n_cols
  · simp only [if_pos hj, hx idx.1 hj]
  · simp only [if_neg hj]

/-- Pure-spec bridge for the rstd store: a launch state whose active `X` lanes
hold `xs` computes the pure `rmsFwdInvVarSpec xs`. -/
theorem rmsInvVarSpec_eq_of_loaded
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.readMem X (s.pid * X_row_stride + j.val) = xs j) :
    rmsInvVarSpec s X X_row_stride n_cols BLOCK_SIZE eps
      = rmsFwdInvVarSpec n_cols BLOCK_SIZE eps xs := by
  unfold rmsInvVarSpec rmsFwdInvVarSpec rmsInvVarCarrier rmsFwdInvVarCarrier
    rmsSumCarrier rmsFwdSumCarrier
  rw [rmsInputTile_eq_of_loaded s X X_row_stride n_cols BLOCK_SIZE xs hx]

/-- Pure-spec bridge for the plain `Y` store at an **active** lane. -/
theorem rmsLayernormYSpec_eq_of_loaded
    (s : BlockState) (X W : RegionName)
    (X_row_stride W_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs ws : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.readMem X (s.pid * X_row_stride + j.val) = xs j)
    (hw : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.readMem W (j.val * W_row_stride) = ws j)
    (i : Fin BLOCK_SIZE) (hi : i.val < n_cols) :
    rmsLayernormYSpec s X W X_row_stride W_row_stride n_cols BLOCK_SIZE eps i
      = rmsFwdYSpec n_cols BLOCK_SIZE eps xs ws i := by
  unfold rmsLayernormYSpec rmsFwdYSpec rowElem rmsInvVarCarrier
    rmsFwdInvVarCarrier rmsSumCarrier rmsFwdSumCarrier
  rw [rmsInputTile_eq_of_loaded s X X_row_stride n_cols BLOCK_SIZE xs hx,
      hx i hi, hw i hi]

/-- Pure-spec bridge for the Gemma `Y` store at an **active** lane. -/
theorem gemmaRmsLayernormYSpec_eq_of_loaded
    (s : BlockState) (X W : RegionName)
    (X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs ws : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.readMem X (s.pid * X_row_stride + j.val) = xs j)
    (hw : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.readMem W j.val = ws j)
    (i : Fin BLOCK_SIZE) (hi : i.val < n_cols) :
    gemmaRmsLayernormYSpec s X W X_row_stride n_cols BLOCK_SIZE eps i
      = gemmaRmsFwdYSpec n_cols BLOCK_SIZE eps xs ws i := by
  unfold gemmaRmsLayernormYSpec gemmaRmsFwdYSpec rowElem rmsInvVarCarrier
    rmsFwdInvVarCarrier rmsSumCarrier rmsFwdSumCarrier
  rw [rmsInputTile_eq_of_loaded s X X_row_stride n_cols BLOCK_SIZE xs hx,
      hx i hi, hw i hi]

/-- Termination: the plain forward executes to completion from any state —
including `BLOCK_SIZE = 0`, since its reductions (`sum`, `rsqrt`) are total, so
no `0 < BLOCK_SIZE` side condition is needed here. -/
private theorem rms_layernorm_forward_exec_isSome
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState) :
    ∃ s1, exec ((rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        W_row_stride r_row_stride n_cols eps BLOCK_SIZE).toAlgKernel) s
      = some s1 := by
  simp [exec, rms_layernorm_forward, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Termination for the Gemma forward (same shape as the plain one). -/
private theorem gemma_rms_layernorm_forward_exec_isSome
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState) :
    ∃ s1, exec ((gemma_rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        r_row_stride n_cols eps BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, gemma_rms_layernorm_forward, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked `Y` store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (ρ : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = ρ ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem ρ o = s.mem ρ o := by
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

/-- Frame half for the plain forward: every memory cell not actively written —
every cell outside the active `Y` lanes and off the scalar `r` store cell — is
preserved by the run. -/
private theorem rms_layernorm_forward_frame
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s s1 : BlockState)
    (hExec : exec ((rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        W_row_stride r_row_stride n_cols eps BLOCK_SIZE).toAlgKernel) s
      = some s1)
    (ρ : RegionName) (o : Nat)
    (hmissY : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(Y = ρ ∧ s.pid * Y_row_stride + i.val = o))
    (hmissR : ¬(r = ρ ∧ s.pid * r_row_stride = o)) :
    s1.mem ρ o = s.mem ρ o := by
  simp [exec, rms_layernorm_forward, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ ρ o _ _ ?_) ?_
  · intro k _ hmk hc
    exact hmissY k.1 (by simpa using hmk) hc
  · simp only [BlockState.setReg_mem, BlockState.writeMem_mem]
    exact if_neg fun hc => hmissR ⟨hc.1.symm, hc.2.symm⟩

/-- Frame half for the Gemma forward (same footprint as the plain one). -/
private theorem gemma_rms_layernorm_forward_frame
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s s1 : BlockState)
    (hExec : exec ((gemma_rms_layernorm_forward Y X W r Y_row_stride
        X_row_stride r_row_stride n_cols eps BLOCK_SIZE).toAlgKernel) s
      = some s1)
    (ρ : RegionName) (o : Nat)
    (hmissY : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(Y = ρ ∧ s.pid * Y_row_stride + i.val = o))
    (hmissR : ¬(r = ρ ∧ s.pid * r_row_stride = o)) :
    s1.mem ρ o = s.mem ρ o := by
  simp [exec, gemma_rms_layernorm_forward, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ ρ o _ _ ?_) ?_
  · intro k _ hmk hc
    exact hmissY k.1 (by simpa using hmk) hc
  · simp only [BlockState.setReg_mem, BlockState.writeMem_mem]
    exact if_neg fun hc => hmissR ⟨hc.1.symm, hc.2.symm⟩

/-- The plain forward sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked loads with `other`, dtype casts, sum reduction,
`rsqrt`, scalar store, masked store). -/
theorem rms_layernorm_forward_flattenOk
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) :
    ((rms_layernorm_forward Y X W r Y_row_stride X_row_stride W_row_stride
        r_row_stride n_cols eps BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rms_layernorm_forward, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- The Gemma forward sits inside the flat-memory bridge's covered fragment. -/
theorem gemma_rms_layernorm_forward_flattenOk
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) :
    ((gemma_rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        r_row_stride n_cols eps BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [gemma_rms_layernorm_forward, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Per-execution safety walk for the plain forward: the masked `X`/`W` loads
and the masked `Y` store reduce to **lane-wise** bounds at the active lanes;
the **unmasked** scalar `r` store reduces to the single-cell bound
`row_idx · r_row_stride < bounds r`. -/
theorem rms_layernorm_forward_traceSafe
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (bounds : RegionBounds) (s : BlockState)
    (hin1 : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * X_row_stride + j.val < bounds X)
    (hin2 : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      j.val * W_row_stride < bounds W)
    (hout1 : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * Y_row_stride + j.val < bounds Y)
    (hout2 : s.pid * r_row_stride < bounds r) :
    Kernel.TraceSafe bounds
      ((rms_layernorm_forward Y X W r Y_row_stride X_row_stride W_row_stride
        r_row_stride n_cols eps BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [rms_layernorm_forward, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.div,
    ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
    FloatDType.toWithBot,
    Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin1 a ha, fun a ha => hin2 a ha, hout2,
    fun a ha => hout1 a ha⟩

/-- Per-execution safety walk for the Gemma forward (contiguous `W` window). -/
theorem gemma_rms_layernorm_forward_traceSafe
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (bounds : RegionBounds) (s : BlockState)
    (hin1 : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * X_row_stride + j.val < bounds X)
    (hin2 : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → j.val < bounds W)
    (hout1 : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * Y_row_stride + j.val < bounds Y)
    (hout2 : s.pid * r_row_stride < bounds r) :
    Kernel.TraceSafe bounds
      ((gemma_rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        r_row_stride n_cols eps BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [gemma_rms_layernorm_forward, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.div,
    ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
    FloatDType.toWithBot,
    Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin1 a ha, fun a ha => hin2 a ha, hout2,
    fun a ha => hout1 a ha⟩

/-- **The region-model masked Hoare triple** for the plain forward —
termination, active-lane `Y` values, the scalar `r` (rstd) value, and frame off
the active `Y` lanes and the `r` cell, from any launch state whose active input
lanes hold `xs` (the `X` row) and `ws` (the weights). This is the `hrun`
obligation of the `⊨` headline. `Y ≠ r` is required: the unconditional scalar
rstd store must not alias the masked row store. -/
theorem rms_layernorm_forward_region_run
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (hYr : Y ≠ r)
    (s₀ : BlockState) (xs ws : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem X (s₀.pid * X_row_stride + j.val) = xs j)
    (hw : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem W (j.val * W_row_stride) = ws j) :
    ∃ s1, exec ((rms_layernorm_forward Y X W r Y_row_stride X_row_stride
          W_row_stride r_row_stride n_cols eps BLOCK_SIZE).toAlgKernel) s₀
        = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem Y (s₀.pid * Y_row_stride + j.val)
            = rmsFwdYSpec n_cols BLOCK_SIZE eps xs ws j)
      ∧ s1.readMem r (s₀.pid * r_row_stride)
          = rmsFwdInvVarSpec n_cols BLOCK_SIZE eps xs
      ∧ (∀ ρ o,
          (ρ ≠ Y ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * Y_row_stride + j.val) →
          (ρ ≠ r ∨ o ≠ s₀.pid * r_row_stride) →
          s1.mem ρ o = s₀.mem ρ o) := by
  have hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s₀ Y_row_stride i) := by
    intro a b h
    simp only [yOutOffset] at h
    exact Fin.ext (Nat.add_left_cancel h)
  obtain ⟨s1, hs1⟩ := rms_layernorm_forward_exec_isSome Y X W r Y_row_stride
    X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE eps s₀
  refine ⟨s1, hs1, fun j hj => ?_, ?_, fun ρ o h1 h2 => ?_⟩
  · have h := rms_layernorm_forward_y_correct Y X W r Y_row_stride X_row_stride
      W_row_stride r_row_stride n_cols BLOCK_SIZE eps s₀ s1 hYr hOutInj hs1 j
    simp only [yOutOffset] at h
    rw [h, if_pos hj]
    exact rmsLayernormYSpec_eq_of_loaded s₀ X W X_row_stride W_row_stride
      n_cols BLOCK_SIZE eps xs ws hx hw j hj
  · have h := rms_layernorm_forward_inv_var_correct Y X W r Y_row_stride
      X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE eps s₀ s1
      (Ne.symm hYr) hs1
    simp only [rOutOffset] at h
    rw [h]
    exact rmsInvVarSpec_eq_of_loaded s₀ X X_row_stride n_cols BLOCK_SIZE eps
      xs hx
  · refine rms_layernorm_forward_frame Y X W r Y_row_stride X_row_stride
      W_row_stride r_row_stride n_cols BLOCK_SIZE eps s₀ s1 hs1 ρ o
      (fun i hi hc => ?_) (fun hc => ?_)
    · rcases h1 with hne | hno
      · exact hne hc.1.symm
      · exact hno i hi hc.2.symm
    · rcases h2 with hne | hno
      · exact hne hc.1.symm
      · exact hno hc.2.symm

/-- **The region-model masked Hoare triple** for the Gemma forward (contiguous
`W` window, `(w + 1)` scaling; otherwise as the plain one). -/
theorem gemma_rms_layernorm_forward_region_run
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (hYr : Y ≠ r)
    (s₀ : BlockState) (xs ws : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem X (s₀.pid * X_row_stride + j.val) = xs j)
    (hw : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem W j.val = ws j) :
    ∃ s1, exec ((gemma_rms_layernorm_forward Y X W r Y_row_stride X_row_stride
          r_row_stride n_cols eps BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem Y (s₀.pid * Y_row_stride + j.val)
            = gemmaRmsFwdYSpec n_cols BLOCK_SIZE eps xs ws j)
      ∧ s1.readMem r (s₀.pid * r_row_stride)
          = rmsFwdInvVarSpec n_cols BLOCK_SIZE eps xs
      ∧ (∀ ρ o,
          (ρ ≠ Y ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * Y_row_stride + j.val) →
          (ρ ≠ r ∨ o ≠ s₀.pid * r_row_stride) →
          s1.mem ρ o = s₀.mem ρ o) := by
  have hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s₀ Y_row_stride i) := by
    intro a b h
    simp only [yOutOffset] at h
    exact Fin.ext (Nat.add_left_cancel h)
  obtain ⟨s1, hs1⟩ := gemma_rms_layernorm_forward_exec_isSome Y X W r
    Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE eps s₀
  refine ⟨s1, hs1, fun j hj => ?_, ?_, fun ρ o h1 h2 => ?_⟩
  · have h := gemma_rms_layernorm_forward_y_correct Y X W r Y_row_stride
      X_row_stride r_row_stride n_cols BLOCK_SIZE eps s₀ s1 hYr hOutInj hs1 j
    simp only [yOutOffset] at h
    rw [h, if_pos hj]
    exact gemmaRmsLayernormYSpec_eq_of_loaded s₀ X W X_row_stride
      n_cols BLOCK_SIZE eps xs ws hx hw j hj
  · have h := gemma_rms_layernorm_forward_inv_var_correct Y X W r Y_row_stride
      X_row_stride r_row_stride n_cols BLOCK_SIZE eps s₀ s1 (Ne.symm hYr) hs1
    simp only [rOutOffset] at h
    rw [h]
    exact rmsInvVarSpec_eq_of_loaded s₀ X X_row_stride n_cols BLOCK_SIZE eps
      xs hx
  · refine gemma_rms_layernorm_forward_frame Y X W r Y_row_stride X_row_stride
      r_row_stride n_cols BLOCK_SIZE eps s₀ s1 hs1 ρ o
      (fun i hi hc => ?_) (fun hc => ?_)
    · rcases h1 with hne | hno
      · exact hne hc.1.symm
      · exact hno i hi hc.2.symm
    · rcases h2 with hne | hno
      · exact hne hc.1.symm
      · exact hno hc.2.symm

/-- `_rms_layernorm_forward`'s masked two-output **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`out1`/`out2` — which buffer is which argument (the wiring): the
  input matrix `X`, the per-column weights `W`, the output matrix `Y`, the
  per-row rstd vector `r`;
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`write1` — **strided row windows**: program `row_idx` reads its `X`
  row at `row_idx · X_row_stride + j` and writes its `Y` row at
  `row_idx · Y_row_stride + j` (the host-side one-program-per-row launch);
* `read2` — the weight window is **pid-independent and column-strided**: every
  program reads `W[j · W_row_stride]` (the Python kernel scales the offsets by
  `W_row_stride`);
* `write2` — the **scalar** rstd cell `r[row_idx · r_row_stride]`, the same
  for every lane;
* `mask` — the active lanes `j < n_cols`, the same for every program; the
  load masks and the `Y` store mask coincide, so `read2Mask`/`writeMask1` keep
  their `mask` default;
* `writeMask2` — lane `0` carries the scalar rstd; the other lanes are
  write-inactive and carry no obligations on either side.

The grid is 1-D, so the second program-id axis is an unused parameter: windows
and masks are constant in `pid₁` (the headline's `∀ pid₁` quantification is
vacuous but honest). The windows and masks are declared, not parsed from the
kernel; the headline **proves** the kernel's actual addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes. -/
def rmsLayernormFwdIO (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : Masked2DKernelIO₂ₓ₂ where
  kernel := rms_layernorm_forward Y X W r Y_row_stride X_row_stride
    W_row_stride r_row_stride n_cols eps BLOCK_SIZE
  in1 := X
  in2 := W
  out1 := Y
  out2 := r
  B := BLOCK_SIZE
  read1 := fun row_idx _ j => row_idx * X_row_stride + j.val
  read2 := fun _ _ j => j.val * W_row_stride
  write1 := fun row_idx _ j => row_idx * Y_row_stride + j.val
  write2 := fun row_idx _ _ => row_idx * r_row_stride
  mask := fun _ _ j => j.val < n_cols
  writeMask2 := fun _ _ j => j.val = 0

/-- `_gemma_rms_layernorm_forward`'s masked two-output **IO signature** — as
`rmsLayernormFwdIO`, except the weight window is **contiguous**: the Python
kernel accepts `W_row_stride` but loads `W + col_offsets`, and this signature
preserves that stride-free weight access (`read2 = j`). -/
def gemmaRmsLayernormFwdIO (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : Masked2DKernelIO₂ₓ₂ where
  kernel := gemma_rms_layernorm_forward Y X W r Y_row_stride X_row_stride
    r_row_stride n_cols eps BLOCK_SIZE
  in1 := X
  in2 := W
  out1 := Y
  out2 := r
  B := BLOCK_SIZE
  read1 := fun row_idx _ j => row_idx * X_row_stride + j.val
  read2 := fun _ _ j => j.val
  write1 := fun row_idx _ j => row_idx * Y_row_stride + j.val
  write2 := fun row_idx _ _ => row_idx * r_row_stride
  mask := fun _ _ j => j.val < n_cols
  writeMask2 := fun _ _ j => j.val = 0

/-- **The headline (plain forward)**: `_rms_layernorm_forward` implements the
exact RMS normalization pair on its masked two-output IO signature — for every
disjoint flat placement of the four buffers, every program id whose active
lanes and scalar rstd cell are in bounds, and every launch state whose active
input lanes hold `xs` (the `X` row) and `ws` (the weights), the translated
pointer kernel terminates, every active `Y` lane `j` holds
`rmsFwdYSpec … = (xs j · inv_var) · ws j`, the rstd cell holds
`rmsFwdInvVarSpec … = rsqrt(sum(x²)/n_cols + eps)`, and every other memory cell
is unchanged. Side conditions, both genuinely forced: `Y ≠ r` (the
unconditional scalar rstd store must not alias the masked row store) and
`0 < BLOCK_SIZE` (the `r` store is unmasked in the kernel, so its safety bound
and frame exclusion are carried by the lane-0 gate `writeMask2`, which needs a
lane). Proof: `Masked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model
masked triple with the flat-memory bridge side conditions. -/
specification rms_layernorm_forward_correctness
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (hYr : Y ≠ r) (hB : 0 < BLOCK_SIZE) :
    rmsLayernormFwdIO Y X W r Y_row_stride X_row_stride W_row_stride
        r_row_stride n_cols eps BLOCK_SIZE ⊨
      fun _ _ xs ws =>
        (fun i => rmsFwdYSpec n_cols BLOCK_SIZE eps xs ws i,
         fun _ => rmsFwdInvVarSpec n_cols BLOCK_SIZE eps xs) := by
  refine Masked2DKernelIO₂ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact rms_layernorm_forward_flattenOk Y X W r Y_row_stride X_row_stride
      W_row_stride r_row_stride n_cols BLOCK_SIZE eps
  · intro bounds s h1 h2 h3 h4
    exact rms_layernorm_forward_traceSafe Y X W r Y_row_stride X_row_stride
      W_row_stride r_row_stride n_cols BLOCK_SIZE eps bounds s h1 h2
      h3 (h4 ⟨0, hB⟩ rfl)
  · intro s₀ xs ws hx hw
    obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
      rms_layernorm_forward_region_run Y X W r Y_row_stride X_row_stride
        W_row_stride r_row_stride n_cols BLOCK_SIZE eps hYr s₀ xs ws hx hw
    refine ⟨s1, hexec, hval1, fun j _ => hval2, fun ρ o h1 h2 => ?_⟩
    refine hframe ρ o h1 ?_
    rcases h2 with hne | hno
    · exact Or.inl hne
    · exact Or.inr (hno ⟨0, hB⟩ rfl)

/-- **The headline (Gemma forward)**: `_gemma_rms_layernorm_forward` implements
the Gemma-scaled RMS normalization pair on its masked two-output IO signature —
as the plain headline, with every active `Y` lane `j` holding
`gemmaRmsFwdYSpec … = (xs j · inv_var) · (ws j + 1)` over the contiguous weight
window. Same genuinely-forced side conditions (`Y ≠ r`, `0 < BLOCK_SIZE`). -/
specification gemma_rms_layernorm_forward_correctness
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (hYr : Y ≠ r) (hB : 0 < BLOCK_SIZE) :
    gemmaRmsLayernormFwdIO Y X W r Y_row_stride X_row_stride r_row_stride
        n_cols eps BLOCK_SIZE ⊨
      fun _ _ xs ws =>
        (fun i => gemmaRmsFwdYSpec n_cols BLOCK_SIZE eps xs ws i,
         fun _ => rmsFwdInvVarSpec n_cols BLOCK_SIZE eps xs) := by
  refine Masked2DKernelIO₂ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact gemma_rms_layernorm_forward_flattenOk Y X W r Y_row_stride
      X_row_stride r_row_stride n_cols BLOCK_SIZE eps
  · intro bounds s h1 h2 h3 h4
    exact gemma_rms_layernorm_forward_traceSafe Y X W r Y_row_stride
      X_row_stride r_row_stride n_cols BLOCK_SIZE eps bounds s h1 h2
      h3 (h4 ⟨0, hB⟩ rfl)
  · intro s₀ xs ws hx hw
    obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
      gemma_rms_layernorm_forward_region_run Y X W r Y_row_stride X_row_stride
        r_row_stride n_cols BLOCK_SIZE eps hYr s₀ xs ws hx hw
    refine ⟨s1, hexec, hval1, fun j _ => hval2, fun ρ o h1 h2 => ?_⟩
    refine hframe ρ o h1 ?_
    rcases h2 with hne | hno
    · exact Or.inl hne
    · exact Or.inr (hno ⟨0, hB⟩ rfl)

end VeriTile.Bench.TritonBenchG.FastRmsLayernorm
