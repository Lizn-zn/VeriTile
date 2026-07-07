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
Because `row_idx = tl.program_id(0)` is universally quantified (via `s.pid`), the
per-program statement covers every row of the grid.

## Proof architecture

```
rms_layernorm_forward_output_summary          ← TOP THEOREM (plain forward)
  ├─ rms_layernorm_forward_surface_toAlgorithm_supported   surface lowers
  └─ rms_layernorm_forward_all_outputs_compute_correct
       ├─ rms_layernorm_forward_y_compute_correct   ← masked Y store
       │    └─ rms_layernorm_forward_y_correct
       └─ rms_layernorm_forward_inv_var_compute_correct  ← scalar rstd store into r
            └─ rms_layernorm_forward_inv_var_correct
gemma_rms_layernorm_forward_output_summary    ← TOP THEOREM (Gemma forward)
  └─ gemma_rms_layernorm_forward_all_outputs_compute_correct  (analogous)
rms_layernorm_backward_dy_compute_correct     ← backward dY (plain)
gemma_rms_layernorm_backward_dy_compute_correct  ← backward dY (Gemma)
```

The two forward summaries are the public top theorems. The RMS row math
(`rmsInputTile`, `rmsSumCarrier`, `rmsInvVarCarrier`, `rmsLayernormYSpec`,
`gemmaRmsLayernormYSpec`) is defined inline in this file rather than reusing
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
the padded tail. Output/input disjointness (`Y ≠ r`, `r ≠ Y`) is assumed. The
`@triton.heuristics` GEMMA dispatch is modeled by the two separate forward
kernels rather than a runtime branch. `@triton.autotune` is not modeled.
-/

namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm

open VeriTile.Triton

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

/-- The regular RMS layernorm forward surface lowers to the algorithm layer,
including the row inverse-variance store and masked `Y` writeback. -/
theorem rms_layernorm_forward_surface_toAlgorithm_supported
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ∃ alg, (rms_layernorm_forward Y X W r Y_row_stride X_row_stride
      W_row_stride r_row_stride n_cols eps BLOCK_SIZE).toAlgorithm? =
        Except.ok alg := by
  simp [rms_layernorm_forward, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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

/-- The Gemma RMS layernorm forward surface lowers to the algorithm layer,
including the row inverse-variance store and masked `Y` writeback. -/
theorem gemma_rms_layernorm_forward_surface_toAlgorithm_supported
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ∃ alg, (gemma_rms_layernorm_forward Y X W r Y_row_stride
      X_row_stride r_row_stride n_cols eps BLOCK_SIZE).toAlgorithm? =
        Except.ok alg := by
  simp [gemma_rms_layernorm_forward, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

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
    (hOutInj : Function.Injective
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

/-- Compute-facing correctness for the `Y` output of `_rms_layernorm_forward`. -/
theorem rms_layernorm_forward_y_compute_correct
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hRegions : Y ≠ r)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i)) :
    ComputeCorrect.Realizes
      (kernel := rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        W_row_stride r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (Y, yOutOffset s Y_row_stride i)))
      (expected := fun i =>
        rmsLayernormYSpec s X W X_row_stride W_row_stride n_cols BLOCK_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rms_layernorm_forward, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rms_layernorm_forward_y_correct Y X W r Y_row_stride X_row_stride
    W_row_stride r_row_stride n_cols BLOCK_SIZE eps s s' hRegions hOutInj hExec i
  simpa [hActive] using h

/-- Executed-state correctness for the `Y` output of
`_gemma_rms_layernorm_forward`. -/
theorem gemma_rms_layernorm_forward_y_correct
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hRegions : Y ≠ r)
    (hOutInj : Function.Injective
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

/-- Compute-facing correctness for the `Y` output of
`_gemma_rms_layernorm_forward`. -/
theorem gemma_rms_layernorm_forward_y_compute_correct
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hRegions : Y ≠ r)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i)) :
    ComputeCorrect.Realizes
      (kernel := gemma_rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (Y, yOutOffset s Y_row_stride i)))
      (expected := fun i =>
        gemmaRmsLayernormYSpec s X W X_row_stride n_cols BLOCK_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [gemma_rms_layernorm_forward, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := gemma_rms_layernorm_forward_y_correct Y X W r Y_row_stride X_row_stride
    r_row_stride n_cols BLOCK_SIZE eps s s' hRegions hOutInj hExec i
  simpa [hActive] using h

/-- Executed-state correctness for the in-place `dY` output of
`_rms_layernorm_backward` with `GEMMA = false`. -/
theorem rms_layernorm_backward_dy_correct
    (dY X W r _dW : RegionName)
    (dY_row_stride X_row_stride _W_row_stride r_row_stride _dW_row_stride
      n_cols BLOCK_SIZE : Nat)
    (_eps : ℝ) (s s' : BlockState)
    (hOutInj : Function.Injective
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
    ComputeCorrect.Realizes
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
    (hOutInj : Function.Injective
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
      (kernel := rms_layernorm_forward_inv_var_store_slice InvVarPre r r_row_stride)
      (initialState := s)
      (write := fun _ : PUnit => some (r, rOutOffset s r_row_stride))
      (expected := fun _ => invVarStoreSpec s InvVarPre r_row_stride) := by
  unfold ComputeCorrect.Realizes
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

/-- Compute-facing correctness for the `r` (rstd / inv_var) output of
`_rms_layernorm_forward`. -/
theorem rms_layernorm_forward_inv_var_compute_correct
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hRegions : r ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        W_row_stride r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := fun _ : PUnit => some (r, rOutOffset s r_row_stride))
      (expected := fun _ =>
        rmsInvVarSpec s X X_row_stride n_cols BLOCK_SIZE eps) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rms_layernorm_forward, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact rms_layernorm_forward_inv_var_correct Y X W r Y_row_stride X_row_stride
    W_row_stride r_row_stride n_cols BLOCK_SIZE eps s s' hRegions hExec

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

/-- Compute-facing correctness for the `r` (rstd / inv_var) output of
`_gemma_rms_layernorm_forward`. -/
theorem gemma_rms_layernorm_forward_inv_var_compute_correct
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hRegions : r ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := gemma_rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := fun _ : PUnit => some (r, rOutOffset s r_row_stride))
      (expected := fun _ =>
        rmsInvVarSpec s X X_row_stride n_cols BLOCK_SIZE eps) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [gemma_rms_layernorm_forward, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact gemma_rms_layernorm_forward_inv_var_correct Y X W r Y_row_stride X_row_stride
    r_row_stride n_cols BLOCK_SIZE eps s s' hRegions hExec

/-- Full forward output coverage for `_rms_layernorm_forward`: vector `Y` and
row rstd/inv-var `r` are both characterized against the Python formula. -/
theorem rms_layernorm_forward_all_outputs_compute_correct
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hYr : Y ≠ r) (hrY : r ≠ Y)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i)) :
    (ComputeCorrect.Realizes
      (kernel := rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        W_row_stride r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (Y, yOutOffset s Y_row_stride i)))
      (expected := fun i =>
        rmsLayernormYSpec s X W X_row_stride W_row_stride n_cols
          BLOCK_SIZE eps i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        W_row_stride r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := fun _ : PUnit => some (r, rOutOffset s r_row_stride))
      (expected := fun _ =>
        rmsInvVarSpec s X X_row_stride n_cols BLOCK_SIZE eps)) := by
  constructor
  · exact rms_layernorm_forward_y_compute_correct Y X W r Y_row_stride
      X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE eps s hYr
      hOutInj
  · exact rms_layernorm_forward_inv_var_compute_correct Y X W r Y_row_stride
      X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE eps s hrY

/-- Full forward output coverage for `_gemma_rms_layernorm_forward`: vector `Y`
and row rstd/inv-var `r` are both characterized against the Python formula. -/
theorem gemma_rms_layernorm_forward_all_outputs_compute_correct
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hYr : Y ≠ r) (hrY : r ≠ Y)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i)) :
    (ComputeCorrect.Realizes
      (kernel := gemma_rms_layernorm_forward Y X W r Y_row_stride
        X_row_stride r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (Y, yOutOffset s Y_row_stride i)))
      (expected := fun i =>
        gemmaRmsLayernormYSpec s X W X_row_stride n_cols BLOCK_SIZE eps i)) ∧
    (ComputeCorrect.Realizes
      (kernel := gemma_rms_layernorm_forward Y X W r Y_row_stride
        X_row_stride r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := fun _ : PUnit => some (r, rOutOffset s r_row_stride))
      (expected := fun _ =>
        rmsInvVarSpec s X X_row_stride n_cols BLOCK_SIZE eps)) := by
  constructor
  · exact gemma_rms_layernorm_forward_y_compute_correct Y X W r Y_row_stride
      X_row_stride r_row_stride n_cols BLOCK_SIZE eps s hYr hOutInj
  · exact gemma_rms_layernorm_forward_inv_var_compute_correct Y X W r
      Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE eps s hrY

/-- Public forward summary for regular RMS layernorm: the full Python forward
surface lowers, and the checked forward kernel characterizes both
Python-observable outputs `Y` and `r`. -/
theorem rms_layernorm_forward_output_summary
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hYr : Y ≠ r) (hrY : r ≠ Y)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i)) :
    (∃ alg, (rms_layernorm_forward Y X W r Y_row_stride X_row_stride
      W_row_stride r_row_stride n_cols eps BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        W_row_stride r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (Y, yOutOffset s Y_row_stride i)))
      (expected := fun i =>
        rmsLayernormYSpec s X W X_row_stride W_row_stride n_cols
          BLOCK_SIZE eps i)) ∧
    (ComputeCorrect.Realizes
      (kernel := rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        W_row_stride r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := fun _ : PUnit => some (r, rOutOffset s r_row_stride))
      (expected := fun _ =>
        rmsInvVarSpec s X X_row_stride n_cols BLOCK_SIZE eps))) := by
  constructor
  · exact rms_layernorm_forward_surface_toAlgorithm_supported Y X W r
      Y_row_stride X_row_stride W_row_stride r_row_stride n_cols eps BLOCK_SIZE
  · exact rms_layernorm_forward_all_outputs_compute_correct Y X W r
      Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE
      eps s hYr hrY hOutInj

/-- Public forward summary for Gemma RMS layernorm: the full Python forward
surface lowers, and the checked forward kernel characterizes both
Python-observable outputs `Y` and `r`. -/
theorem gemma_rms_layernorm_forward_output_summary
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hYr : Y ≠ r) (hrY : r ≠ Y)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i)) :
    (∃ alg, (gemma_rms_layernorm_forward Y X W r Y_row_stride
      X_row_stride r_row_stride n_cols eps BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := gemma_rms_layernorm_forward Y X W r Y_row_stride
        X_row_stride r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (Y, yOutOffset s Y_row_stride i)))
      (expected := fun i =>
        gemmaRmsLayernormYSpec s X W X_row_stride n_cols BLOCK_SIZE eps i)) ∧
    (ComputeCorrect.Realizes
      (kernel := gemma_rms_layernorm_forward Y X W r Y_row_stride
        X_row_stride r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := fun _ : PUnit => some (r, rOutOffset s r_row_stride))
      (expected := fun _ =>
        rmsInvVarSpec s X X_row_stride n_cols BLOCK_SIZE eps))) := by
  constructor
  · exact gemma_rms_layernorm_forward_surface_toAlgorithm_supported Y X W r
      Y_row_stride X_row_stride r_row_stride n_cols eps BLOCK_SIZE
  · exact gemma_rms_layernorm_forward_all_outputs_compute_correct Y X W r
      Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE eps s hYr hrY
      hOutInj

end VeriTile.Bench.TritonBenchG.FastRmsLayernorm
