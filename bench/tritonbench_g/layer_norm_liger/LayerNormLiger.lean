import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `layer_norm_liger` — strict per-kernel correctness

`_layer_norm_forward_kernel` is the Liger LayerNorm forward: each program
`row_idx` normalizes one row of `X` by its mean and variance, applies affine
scale `W` and bias `B`, and writes the normalized row
`Y = (X - mean) * rstd * W + B` together with the per-row mean `Mean` and
reciprocal std `RSTD`, where `rstd = 1 / sqrt(var + eps)`. The Liger backward
surface is also transcribed (lowering checked).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body, for one program (one row). The host launch (grid over rows `(n_rows,)`,
the host-side `BLOCK_SIZE` choice via `calculate_settings`, scheduling, and how
the runtime composes per-row writes into the buffers) is the *trusted boundary*,
not a proof obligation here. Because `row_idx = tl.program_id(0)` is universally
quantified (via `s.pid`), the per-program statement covers every row of the grid.

## Proof architecture

```
layer_norm_liger_forward_output_summary       ← TOP THEOREM
  ├─ layer_norm_liger_forward_surface_toAlgorithm_supported   surface lowers
  └─ layer_norm_liger_forward_all_outputs_compute_correct
       ├─ layer_norm_liger_forward_y_compute_correct      ← masked Y store
       │    └─ layer_norm_liger_forward_y_correct
       ├─ layer_norm_liger_forward_mean_compute_correct   ← scalar mean store
       │    └─ layer_norm_liger_forward_mean_correct
       └─ layer_norm_liger_forward_inv_var_compute_correct ← scalar rstd store
            └─ layer_norm_liger_forward_inv_var_correct
```

The summary characterizes all three Python-observable forward outputs `Y`,
`Mean`, and `RSTD`. There are additional proof-oriented store-slice theorems
(`layer_norm_liger_forward_mean_store_slice_*`,
`layer_norm_liger_forward_rstd_store_slice_*`) that isolate the scalar stores;
the layernorm row math is defined inline in this file rather than reusing
`VeriTile.Triton.Math.RMSNorm`.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The reductions
`tl.sum(X_row) / n_cols` and `tl.sum((X_row - mean)²) / n_cols` sum over the
*padded* `BLOCK_SIZE` block, but out-of-range lanes are masked to `0` (load
`other=0`), so each sum equals the logical row length `n_cols`. The reciprocal
std is `rstd = rsqrt(var + eps) = 1 / sqrt(var + eps)`
(`layernormInvVarCarrier`); the affine step is `(X - mean) * rstd * W + B`. The
scalar `Mean` / `RSTD` stores are characterized via `WithBot.unbotD 0` of their
carriers. This is a single-block kernel: `BLOCK_SIZE` covers the whole row in one
pass, with the `col_offsets < n_cols` mask handling the padded tail.
Output/output disjointness among `Y`, `Mean`, `RSTD` is assumed (`Y ≠ Mean`,
`Y ≠ RSTD`, `Mean ≠ Y`, `Mean ≠ RSTD`, `RSTD ≠ Y`).
-/

namespace VeriTile.Bench.TritonBenchG.LayerNormLiger

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `layer_norm_liger.py`'s
`_layer_norm_forward_kernel`.

The source accepts `W_row_stride` and `B_row_stride` but does not use them in
pointer arithmetic; they are preserved in the surface signature. -/
def layer_norm_liger_forward_surface
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  Y_ptr += row_idx * $(Y_row_stride)
  X_ptr += row_idx * $(X_row_stride)
  Mean_ptr += row_idx * $(Mean_row_stride)
  RSTD_ptr += row_idx * $(RSTD_row_stride)
  X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
  W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
  B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)
  mean = tl.sum(X_row, axis=0) / $(n_cols)
  var = tl.sum((X_row - mean) * (X_row - mean), axis=0) / $(n_cols)
  rstd = tl.rsqrt(var + $(eps))
  tl.store(Mean_ptr, mean)
  tl.store(RSTD_ptr, rstd)
  Y_row = (X_row - mean) * rstd * W_row + B_row
  tl.store(Y_ptr + col_offsets, Y_row, mask=mask)
}

/-- The full Liger layer-norm forward surface lowers to the algorithm layer,
including row `Mean`/`RSTD` stores and masked `Y` writeback. -/
theorem layer_norm_liger_forward_surface_toAlgorithm_supported
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ∃ alg, (layer_norm_liger_forward_surface Y_ptr Y_row_stride X_ptr
      X_row_stride W_ptr W_row_stride B_ptr B_row_stride Mean_ptr
      Mean_row_stride RSTD_ptr RSTD_row_stride n_cols eps BLOCK_SIZE).toAlgorithm? =
        Except.ok alg := by
  simp [layer_norm_liger_forward_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription of `layer_norm_liger.py`'s
`_layer_norm_backward_kernel`.

This preserves the row-block loop, `DX` writeback, accumulated `DW`/`DB`
partials, pointer increments, and target dtype casts. -/
def layer_norm_liger_backward_surface
    (X W Mean RSTD DX DW DB DY : RegionName)
    (stride_x stride_dx stride_dw stride_db stride_dy
      n_rows n_cols rows_per_program BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_block_id = tl.program_id(0)
  row_start = row_block_id * $(rows_per_program)
  row_end = min((row_block_id + $(1)) * $(rows_per_program), $(n_rows))
  cols = tl.arange(0, $(BLOCK_SIZE))
  mask = cols < $(n_cols)
  dw_row = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  db_row = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  X += row_start * $(stride_x)
  Mean += row_start
  RSTD += row_start
  DX += row_start * $(stride_dx)
  DY += row_start * $(stride_dy)
  for _row in range(row_start, row_end, $(1)) {
    x = tl.load(X + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    dy = tl.load(DY + cols, mask=mask, other=0.0)
    mean = tl.load(Mean)
    rstd = tl.load(RSTD)
    x_hat = (x - mean) * rstd
    wdy = w * dy
    c1 = tl.sum(x_hat * wdy, axis=0) / $(n_cols)
    c2 = tl.sum(wdy, axis=0) / $(n_cols)
    dx = (wdy - (x_hat * c1 + c2)) * rstd
    tl.store(DX + cols, (dx).to(DX.dtype.element_ty), mask=mask)
    dw_row += dy * x_hat
    db_row += dy
    X += $(stride_x)
    Mean += $(1)
    RSTD += $(1)
    DX += $(stride_dx)
    DY += $(stride_dy)
  }
  tl.store(DW + row_block_id * $(stride_dw) + cols,
    (dw_row).to(DW.dtype.element_ty), mask=mask)
  tl.store(DB + row_block_id * $(stride_db) + cols,
    (db_row).to(DB.dtype.element_ty), mask=mask)
}

/-- Proof-oriented forward slice of `layer_norm_liger.py`'s
`_layer_norm_forward_kernel`.

This covers the forward `Y` output for one row block, including masked loads,
mean/variance/rstd computation, `Mean`/`RSTD` side stores, affine transform, and
masked `Y` store. The source forward kernel's `W_row_stride` and `B_row_stride`
parameters are not used in pointer arithmetic, so this slice omits them. -/
def layer_norm_liger_forward
    (Y X W B Mean RSTD : RegionName)
    (Y_row_stride X_row_stride Mean_row_stride RSTD_row_stride n_cols
      BLOCK_SIZE : Nat)
    (eps : ℝ) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)

  Y += row_idx * $(Y_row_stride)
  X += row_idx * $(X_row_stride)
  Mean += row_idx * $(Mean_row_stride)
  RSTD += row_idx * $(RSTD_row_stride)

  X_row = tl.load(X + col_offsets, mask=mask, other=0)
  W_row = tl.load(W + col_offsets, mask=mask, other=0)
  B_row = tl.load(B + col_offsets, mask=mask, other=0)

  mean = tl.sum(X_row, axis=0) / $(n_cols)
  var = tl.sum((X_row - mean) * (X_row - mean), axis=0) / $(n_cols)
  rstd = tl.rsqrt(var + $(eps))
  tl.store(Mean, mean)
  tl.store(RSTD, rstd)
  Y_row = (X_row - mean) * rstd * W_row + B_row
  tl.store(Y + col_offsets, Y_row, mask=mask)
}

def xOffset (s : BlockState) (X_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * X_row_stride + i.val

def yOffset (s : BlockState) (Y_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * Y_row_stride + i.val

noncomputable def layernormInputTile
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < n_cols then
        some (s.readMem X (xOffset s X_row_stride idx.1))
      else some (0 : ℝ) }

noncomputable def layernormMeanCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (n_cols : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (layernormInputTile s X X_row_stride n_cols BLOCK_SIZE)).data PUnit.unit)

noncomputable def layernormCenteredTile
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      Option.map₂ (fun x mean => x - mean)
        ((layernormInputTile s X X_row_stride n_cols BLOCK_SIZE).data idx)
        (layernormMeanCarrier s X X_row_stride n_cols BLOCK_SIZE) }

noncomputable def layernormVarCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (n_cols : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (layernormCenteredTile s X X_row_stride n_cols BLOCK_SIZE)
        (layernormCenteredTile s X X_row_stride n_cols BLOCK_SIZE))).data
        PUnit.unit)

noncomputable def layernormInvVarCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  WithBot.realRsqrt
    (Option.map (fun a => a + eps)
      (layernormVarCarrier s X X_row_stride n_cols BLOCK_SIZE))

noncomputable def layernormYSpec
    (s : BlockState) (X W B : RegionName)
    (X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun affine bias => affine + bias)
      (Option.map₂ (fun scaled w => scaled * w)
        (Option.map₂ (fun centered inv => centered * inv)
          (Option.map₂ (fun x mean => x - mean)
            (some (s.readMem X (xOffset s X_row_stride idx)))
            (layernormMeanCarrier s X X_row_stride n_cols BLOCK_SIZE))
          (layernormInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps))
        (some (s.readMem W idx.val)))
      (some (s.readMem B idx.val)))

/-- Executed-state correctness for the `Y` output of the Liger forward slice. -/
theorem layer_norm_liger_forward_y_correct
    (Y X W B Mean RSTD : RegionName)
    (Y_row_stride X_row_stride Mean_row_stride RSTD_row_stride n_cols
      BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYMean : Y ≠ Mean) (hYRSTD : Y ≠ RSTD)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s Y_row_stride i))
    (hExec : exec (layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps) s =
        some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Y (yOffset s Y_row_stride i) =
        if i.val < n_cols then
          layernormYSpec s X W B X_row_stride n_cols BLOCK_SIZE eps i
        else s.readMem Y (yOffset s Y_row_stride i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, layer_norm_liger_forward, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.sub, NumericDType.div, ComparableDType.lt,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, layernormYSpec, layernormInvVarCarrier, layernormVarCarrier,
            layernormCenteredTile, layernormMeanCarrier, layernormInputTile,
            xOffset, Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realRsqrt, NumericDType.mul]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYMean, hYRSTD]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the `Y` output of the Liger forward slice. -/
theorem layer_norm_liger_forward_y_compute_correct
    (Y X W B Mean RSTD : RegionName)
    (Y_row_stride X_row_stride Mean_row_stride RSTD_row_stride n_cols
      BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hYMean : Y ≠ Mean) (hYRSTD : Y ≠ RSTD)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s Y_row_stride i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (Y, yOffset s Y_row_stride i)))
      (expected := fun i =>
        layernormYSpec s X W B X_row_stride n_cols BLOCK_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_liger_forward, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_liger_forward_y_correct Y X W B Mean RSTD Y_row_stride
    X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps s s'
    hYMean hYRSTD hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented Mean store slice of `layer_norm_liger.py`'s
`_layer_norm_forward_kernel`. Takes a precomputed `MeanPre` scalar (per row)
and proves the scalar writeback into Mean. -/
def layer_norm_liger_forward_mean_store_slice
    (MeanPre Mean : RegionName) (Mean_row_stride : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  mean = tl.load(MeanPre + row * $(Mean_row_stride))
  tl.store(Mean + row * $(Mean_row_stride), mean)
}

def meanRowOffset (s : BlockState) (Mean_row_stride : Nat) : Nat :=
  s.pid * Mean_row_stride

noncomputable def meanStoreSpec (s : BlockState) (MeanPre : RegionName)
    (Mean_row_stride : Nat) : ℝ :=
  s.readMem MeanPre (meanRowOffset s Mean_row_stride)

theorem layer_norm_liger_forward_mean_store_slice_correct
    (MeanPre Mean : RegionName) (Mean_row_stride : Nat) (s s' : BlockState)
    (hExec : exec (layer_norm_liger_forward_mean_store_slice MeanPre Mean Mean_row_stride)
      s = some s') :
    s'.readMem Mean (meanRowOffset s Mean_row_stride) =
      meanStoreSpec s MeanPre Mean_row_stride := by
  simp [exec, layer_norm_liger_forward_mean_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul] at hExec
  subst s'
  simp [meanRowOffset, meanStoreSpec]

theorem layer_norm_liger_forward_mean_store_slice_compute_correct
    (MeanPre Mean : RegionName) (Mean_row_stride : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_liger_forward_mean_store_slice MeanPre Mean Mean_row_stride)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, meanRowOffset s Mean_row_stride))
      (expected := fun _ => meanStoreSpec s MeanPre Mean_row_stride) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_liger_forward_mean_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_liger_forward_mean_store_slice_correct MeanPre Mean Mean_row_stride
    s s' hExec

/-- Proof-oriented RSTD store slice of `layer_norm_liger.py`'s
`_layer_norm_forward_kernel`. Same scalar-copy pattern. -/
def layer_norm_liger_forward_rstd_store_slice
    (RSTDPre RSTD : RegionName) (RSTD_row_stride : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  rstd = tl.load(RSTDPre + row * $(RSTD_row_stride))
  tl.store(RSTD + row * $(RSTD_row_stride), rstd)
}

noncomputable def rstdStoreSpec (s : BlockState) (RSTDPre : RegionName)
    (RSTD_row_stride : Nat) : ℝ :=
  s.readMem RSTDPre (meanRowOffset s RSTD_row_stride)

theorem layer_norm_liger_forward_rstd_store_slice_correct
    (RSTDPre RSTD : RegionName) (RSTD_row_stride : Nat) (s s' : BlockState)
    (hExec : exec (layer_norm_liger_forward_rstd_store_slice RSTDPre RSTD RSTD_row_stride)
      s = some s') :
    s'.readMem RSTD (meanRowOffset s RSTD_row_stride) =
      rstdStoreSpec s RSTDPre RSTD_row_stride := by
  simp [exec, layer_norm_liger_forward_rstd_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul] at hExec
  subst s'
  simp [meanRowOffset, rstdStoreSpec]

theorem layer_norm_liger_forward_rstd_store_slice_compute_correct
    (RSTDPre RSTD : RegionName) (RSTD_row_stride : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_liger_forward_rstd_store_slice RSTDPre RSTD RSTD_row_stride)
      (initialState := s)
      (write := fun _ : PUnit => some (RSTD, meanRowOffset s RSTD_row_stride))
      (expected := fun _ => rstdStoreSpec s RSTDPre RSTD_row_stride) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_liger_forward_rstd_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_liger_forward_rstd_store_slice_correct RSTDPre RSTD RSTD_row_stride
    s s' hExec

/-! ## Full-kernel scalar-store correctness for `Mean` and `inv_var` (rstd).

Per #139, slice proofs over precomputed inputs are insufficient. These
theorems characterize the per-row `Mean` and `RSTD` writebacks for the full
`layer_norm_liger_forward` kernel by stripping the trailing `Y` foldl with
the cross-region helper and (for `Mean`) peeling the intervening `RSTD`
scalar store. -/

/-- Algorithm-layer correctness for the `inv_var` (rstd) scalar store of the
full `layer_norm_liger_forward` kernel. -/
theorem layer_norm_liger_forward_inv_var_correct
    (Y X W B Mean RSTD : RegionName)
    (Y_row_stride X_row_stride Mean_row_stride RSTD_row_stride n_cols
      BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hRSTDY : RSTD ≠ Y)
    (hExec : exec (layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps) s =
        some s') :
    s'.readMem RSTD (s.pid * RSTD_row_stride) =
      WithBot.unbotD 0
        (layernormInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps) := by
  simp [exec, layer_norm_liger_forward, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRSTDY]
  simp [BlockState.writeMem_readMem, layernormInvVarCarrier,
        layernormVarCarrier, layernormCenteredTile, layernormMeanCarrier,
        layernormInputTile, xOffset, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realRsqrt, NumericDType.mul]
  rfl

/-- Compute-facing correctness for the `inv_var` (rstd) scalar store of the
full `layer_norm_liger_forward` kernel. -/
theorem layer_norm_liger_forward_inv_var_compute_correct
    (Y X W B Mean RSTD : RegionName)
    (Y_row_stride X_row_stride Mean_row_stride RSTD_row_stride n_cols
      BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hRSTDY : RSTD ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := fun _ : PUnit => some (RSTD, s.pid * RSTD_row_stride))
      (expected := fun _ =>
        WithBot.unbotD 0
          (layernormInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps)) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_liger_forward, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_liger_forward_inv_var_correct Y X W B Mean RSTD Y_row_stride
    X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps s s'
    hRSTDY hExec

/-- Algorithm-layer correctness for the `Mean` scalar store of the full
`layer_norm_liger_forward` kernel. -/
theorem layer_norm_liger_forward_mean_correct
    (Y X W B Mean RSTD : RegionName)
    (Y_row_stride X_row_stride Mean_row_stride RSTD_row_stride n_cols
      BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hMeanY : Mean ≠ Y) (hMeanRSTD : Mean ≠ RSTD)
    (hExec : exec (layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps) s =
        some s') :
    s'.readMem Mean (s.pid * Mean_row_stride) =
      WithBot.unbotD 0
        (layernormMeanCarrier s X X_row_stride n_cols BLOCK_SIZE) := by
  simp [exec, layer_norm_liger_forward, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hMeanY]
  simp only [BlockState.setReg_readMem]
  rw [BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hMeanRSTD]
  simp [BlockState.writeMem_readMem, layernormMeanCarrier,
        layernormInputTile, xOffset, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.mul]
  rfl

/-- Compute-facing correctness for the `Mean` scalar store of the full
`layer_norm_liger_forward` kernel. -/
theorem layer_norm_liger_forward_mean_compute_correct
    (Y X W B Mean RSTD : RegionName)
    (Y_row_stride X_row_stride Mean_row_stride RSTD_row_stride n_cols
      BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hMeanY : Mean ≠ Y) (hMeanRSTD : Mean ≠ RSTD) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, s.pid * Mean_row_stride))
      (expected := fun _ =>
        WithBot.unbotD 0
          (layernormMeanCarrier s X X_row_stride n_cols BLOCK_SIZE)) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_liger_forward, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_liger_forward_mean_correct Y X W B Mean RSTD Y_row_stride
    X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps s s'
    hMeanY hMeanRSTD hExec

/-- Full forward output coverage for the Liger layer-norm kernel: vector `Y`,
row `Mean`, and row `RSTD` are all characterized against the Python formula. -/
theorem layer_norm_liger_forward_all_outputs_compute_correct
    (Y X W B Mean RSTD : RegionName)
    (Y_row_stride X_row_stride Mean_row_stride RSTD_row_stride n_cols
      BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hYMean : Y ≠ Mean) (hYRSTD : Y ≠ RSTD)
    (hMeanY : Mean ≠ Y) (hMeanRSTD : Mean ≠ RSTD)
    (hRSTDY : RSTD ≠ Y)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s Y_row_stride i)) :
    (ComputeCorrect.Realizes
      (kernel := layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (Y, yOffset s Y_row_stride i)))
      (expected := fun i =>
        layernormYSpec s X W B X_row_stride n_cols BLOCK_SIZE eps i)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, s.pid * Mean_row_stride))
      (expected := fun _ =>
        WithBot.unbotD 0
          (layernormMeanCarrier s X X_row_stride n_cols BLOCK_SIZE))) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := fun _ : PUnit => some (RSTD, s.pid * RSTD_row_stride))
      (expected := fun _ =>
        WithBot.unbotD 0
          (layernormInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps))) := by
  constructor
  · exact layer_norm_liger_forward_y_compute_correct Y X W B Mean RSTD
      Y_row_stride X_row_stride Mean_row_stride RSTD_row_stride n_cols
      BLOCK_SIZE eps s hYMean hYRSTD hOutInj
  · constructor
    · exact layer_norm_liger_forward_mean_compute_correct Y X W B Mean RSTD
        Y_row_stride X_row_stride Mean_row_stride RSTD_row_stride n_cols
        BLOCK_SIZE eps s hMeanY hMeanRSTD
    · exact layer_norm_liger_forward_inv_var_compute_correct Y X W B Mean RSTD
        Y_row_stride X_row_stride Mean_row_stride RSTD_row_stride n_cols
        BLOCK_SIZE eps s hRSTDY

/-- Public forward summary for Liger layer norm: the full Python forward surface
lowers, and the checked forward kernel characterizes all Python-observable
forward outputs `Y`, `Mean`, and `RSTD`. -/
theorem layer_norm_liger_forward_output_summary
    (Y X W B Mean RSTD : RegionName)
    (Y_row_stride X_row_stride W_row_stride B_row_stride
      Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hYMean : Y ≠ Mean) (hYRSTD : Y ≠ RSTD)
    (hMeanY : Mean ≠ Y) (hMeanRSTD : Mean ≠ RSTD)
    (hRSTDY : RSTD ≠ Y)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s Y_row_stride i)) :
    (∃ alg, (layer_norm_liger_forward_surface Y Y_row_stride X
      X_row_stride W W_row_stride B B_row_stride Mean Mean_row_stride RSTD
      RSTD_row_stride n_cols eps BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (Y, yOffset s Y_row_stride i)))
      (expected := fun i =>
        layernormYSpec s X W B X_row_stride n_cols BLOCK_SIZE eps i)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, s.pid * Mean_row_stride))
      (expected := fun _ =>
        WithBot.unbotD 0
          (layernormMeanCarrier s X X_row_stride n_cols BLOCK_SIZE))) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := fun _ : PUnit => some (RSTD, s.pid * RSTD_row_stride))
      (expected := fun _ =>
        WithBot.unbotD 0
          (layernormInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps)))) := by
  constructor
  · exact layer_norm_liger_forward_surface_toAlgorithm_supported Y
      Y_row_stride X X_row_stride W W_row_stride B B_row_stride Mean
      Mean_row_stride RSTD RSTD_row_stride n_cols eps BLOCK_SIZE
  · exact layer_norm_liger_forward_all_outputs_compute_correct Y X W B
      Mean RSTD Y_row_stride X_row_stride Mean_row_stride RSTD_row_stride
      n_cols BLOCK_SIZE eps s hYMean hYRSTD hMeanY hMeanRSTD hRSTDY hOutInj

end VeriTile.Bench.TritonBenchG.LayerNormLiger
