import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL

/-!
# `layer_norm_ops` — strict per-kernel correctness

`_layer_norm_fwd_1pass_kernel` is a one-pass row-wise layer/RMS norm: program
`row` advances `X`/`Y` by their row strides, loads one row (masked by
`cols < N`, masked lanes read as `0`), optionally adds a residual and stores the
residual-out, computes `mean`/`var` (or just `var` for RMS norm), `rstd`, side-
stores `Mean`/`Rstd`, then stores `x_hat * w (+ b)` back masked by `cols < N`.
The companion `_layer_norm_bwd_kernel` computes the backward gradients
(`DX`/`DW`/`DB`/`DRESIDUAL_IN`).

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launches (the one-program-per-row forward grid,
the row-blocked backward grid with its cross-program `DW`/`DB` partial-sum
reduction, the `BLOCK_N = next_power_of_2(N)` choice, and how the runtime
composes per-program writes / accumulates partials into one buffer) are the
*trusted boundary*, not a proof obligation here. Because `row`/`pid` is
universally quantified, each per-program statement covers every program of the
grid.

## Proof architecture

The full Python surfaces lower to the algorithm layer:

```
layer_norm_fwd_1pass_surface_toAlgorithm_supported   forward surface lowers
layer_norm_bwd_surface_toAlgorithm_supported          backward surface lowers
```

Arithmetic correctness is decomposed into per-output-store *slices* over the
forward one-block surface, each with a `*_correct` (algorithm-layer readback)
and `*_compute_correct` (ComputeCorrect) pair, covering the constexpr matrix of
{plain, RMS} × {bias, no-bias} × {residual, no-residual} for the `Y`, `Mean`,
`Rstd`, and `RESIDUAL_OUT` stores, plus the backward writeback slices. The
dimension-general headline theorems (e.g.
`layer_norm_ops_fwd_plain_bias_all_outputs_compute_correct_general`)
compose those `*_compute_correct` slices at symbolic shapes. For
example, the forward `Y` slices reduce through `layer_norm_ops_fwd_y_store_slice_compute_correct`:

```
layer_norm_ops_fwd_y_store_slice_compute_correct      ← masked Y store ComputeCorrect
  └─ layer_norm_ops_fwd_y_store_slice_correct          ← algorithm-layer readback per lane
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The numerous `.to(tl.float32)` / `.to(Y.dtype.element_ty)` casts reduce
to the identity at the algorithm layer (post-erasure all dtypes unify to `ℝ`).
The `sum`/`var` reductions run over the full `BLOCK_N` block, but masked lanes
load `0` (matching `other=0.0`) and the RMS/plain branches use
`tl.where(cols < N, …, 0.0)`, so the reduction-over-padded-block matches the
upstream semantics. Store slices assume injective output offsets and, where the
backward kernel writes distinct buffers, region-disaliasing side conditions
(e.g. `DW ≠ DX`). The backward cross-program `DW`/`DB` partial accumulation is
part of the trusted host boundary; the verified slices cover the per-program
contributions.
-/

namespace VeriTile.Bench.TritonBenchG.LayerNormOps

open VeriTile

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! **★ Main theorems:** `layer_norm_ops_fwd_plain_bias_all_outputs_compute_correct_general`, `layer_norm_ops_fwd_rms_bias_all_outputs_compute_correct_general`, `layer_norm_ops_fwd_residual_bias_all_outputs_compute_correct_general`, `layer_norm_ops_bwd_residual_add_all_outputs_compute_correct_general` -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct

/-- Faithful transcription of `layer_norm_ops.py`'s `_layer_norm_fwd_1pass_kernel`.

This keeps the residual input, `RESIDUAL_OUT`, RMS-vs-layer norm, bias,
`Mean`/`Rstd` side stores, Python ternary expressions, and masked `Y`
writeback. -/
def layer_norm_fwd_1pass_surface
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row
      N BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS : Bool) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  if HAS_RESIDUAL {
    RESIDUAL += row * $(stride_res_row)
  }
  if STORE_RESIDUAL_OUT {
    RESIDUAL_OUT += row * $(stride_res_out_row)
  }
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  if HAS_RESIDUAL {
    residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x += residual
  }
  if STORE_RESIDUAL_OUT {
    tl.store(RESIDUAL_OUT + cols, x, mask=cols < $(N))
  }
  tl.if not IS_RMS_NORM {
    mean = tl.sum(x, axis=0) / $(N)
    tl.store(Mean + row, mean)
    xbar = tl.where(cols < $(N), x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / $(N)
  } else {
    xbar = tl.where(cols < $(N), x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / $(N)
  }
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  tl.if HAS_BIAS {
    b = tl.load(B + cols, mask=mask).to(tl.float32)
  }
  x_hat = ((x - mean) * rstd if not IS_RMS_NORM else x * rstd)
  y = (x_hat * w + b if HAS_BIAS else x_hat * w)
  tl.store(Y + cols, y, mask=mask)
}

/-- The full Python-shaped forward layer-norm surface lowers to the algorithm
layer for all constexpr branch combinations represented by the surface. -/
theorem layer_norm_fwd_1pass_surface_toAlgorithm_supported
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row
      N BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS : Bool) :
    ∃ alg, (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT Mean
      Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row N
      BLOCK_N eps IS_RMS_NORM HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS).toAlgorithm?
        = Except.ok alg := by
  simp [layer_norm_fwd_1pass_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Faithful transcription of `layer_norm_ops.py`'s `_layer_norm_bwd_kernel`.

This keeps the row-block launch, optional `DRESIDUAL` input, optional
`DRESIDUAL_IN` output, RMS-vs-layer norm branch, optional bias/`DB`, optional
recomputed `Y`, row loop, `DX`, `DW`, and `DB` stores. `Mean` is always a
region argument on the Lean surface; as in Python, it is read only in the
non-RMS branch. -/
def layer_norm_bwd_surface
    (X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_dy_row stride_dx_row stride_dres_row
      stride_dres_in_row M N rows_per_program BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_DRESIDUAL STORE_DRESIDUAL HAS_BIAS RECOMPUTE_OUTPUT :
      Bool) :
    ComputeKernel := triton {
  row_block_id = tl.program_id(0)
  row_start = row_block_id * $(rows_per_program)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  X += row_start * $(stride_x_row)
  if HAS_DRESIDUAL {
    DRESIDUAL += row_start * $(stride_dres_row)
  }
  if STORE_DRESIDUAL {
    DRESIDUAL_IN += row_start * $(stride_dres_in_row)
  }
  DY += row_start * $(stride_dy_row)
  DX += row_start * $(stride_dx_row)
  if RECOMPUTE_OUTPUT {
    Y += row_start * $(stride_y_row)
  }
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  if RECOMPUTE_OUTPUT {
    if HAS_BIAS {
      b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    }
  }
  dw = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  if HAS_BIAS {
    db = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  }
  row_end = min((row_block_id + $(1)) * $(rows_per_program), $(M))
  for row in range(row_start, row_end, $(1)) {
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    tl.if not IS_RMS_NORM {
      mean = tl.load(Mean + row)
    }
    rstd = tl.load(Rstd + row)
    xhat = ((x - mean) * rstd if not IS_RMS_NORM else x * rstd)
    xhat = tl.where(mask, xhat, 0.0)
    if RECOMPUTE_OUTPUT {
      y = (xhat * w + b if HAS_BIAS else xhat * w)
      tl.store(Y + cols, y, mask=mask)
    }
    wdy = w * dy
    dw += dy * xhat
    if HAS_BIAS {
      db += dy
    }
    tl.if not IS_RMS_NORM {
      c1 = tl.sum(xhat * wdy, axis=0) / $(N)
      c2 = tl.sum(wdy, axis=0) / $(N)
      dx = (wdy - (xhat * c1 + c2)) * rstd
    } else {
      c1 = tl.sum(xhat * wdy, axis=0) / $(N)
      dx = (wdy - xhat * c1) * rstd
    }
    if HAS_DRESIDUAL {
      dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
      dx += dres
    }
    if STORE_DRESIDUAL {
      tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
    }
    tl.store(DX + cols, dx, mask=mask)
    X += $(stride_x_row)
    if HAS_DRESIDUAL {
      DRESIDUAL += $(stride_dres_row)
    }
    if STORE_DRESIDUAL {
      DRESIDUAL_IN += $(stride_dres_in_row)
    }
    if RECOMPUTE_OUTPUT {
      Y += $(stride_y_row)
    }
    DY += $(stride_dy_row)
    DX += $(stride_dx_row)
  }
  tl.store(DW + row_block_id * $(N) + cols, dw, mask=mask)
  if HAS_BIAS {
    tl.store(DB + row_block_id * $(N) + cols, db, mask=mask)
  }
}

/-- The full Python-shaped backward layer-norm surface lowers to the algorithm
layer, including row-block looping, optional residual input/output, optional
recomputed `Y`, and `DX`/`DW`/`DB` stores. -/
theorem layer_norm_bwd_surface_toAlgorithm_supported
    (X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_dy_row stride_dx_row stride_dres_row
      stride_dres_in_row M N rows_per_program BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_DRESIDUAL STORE_DRESIDUAL HAS_BIAS RECOMPUTE_OUTPUT :
      Bool) :
    ∃ alg, (layer_norm_bwd_surface X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN
      Mean Rstd stride_x_row stride_y_row stride_dy_row stride_dx_row
      stride_dres_row stride_dres_in_row M N rows_per_program BLOCK_N eps
      IS_RMS_NORM HAS_DRESIDUAL STORE_DRESIDUAL HAS_BIAS RECOMPUTE_OUTPUT).toAlgorithm?
        = Except.ok alg := by
  simp [layer_norm_bwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-
Coverage split for #133: the two full Python kernel surfaces are now present,
including the tested backward branches. The proved theorems below are still
branch/store slices: RMS forward `Y`/`Rstd`, non-RMS forward scalar `Mean`/`Rstd`
stores, RMS and non-RMS bias forward scalar side outputs, residual and
residual+bias forward scalar side outputs where the branch produces them, RMS
and non-RMS bias forward arithmetic for `Y`, RMS and non-RMS residual forward
arithmetic for `Y`, RMS and non-RMS residual+bias forward arithmetic for `Y`,
simple scalar-copy slices, one RMS backward row arithmetic slice for partial
`DW`, one bias-branch row arithmetic slice for partial `DB`, and one
residual-add slice covering the `DX`/optional `DRESIDUAL_IN` writebacks after
`dx += dres`, plus recomputed-`Y` slices for both `xhat * W + B` and
`xhat * W`, checked `c1`/`c2` reduction-production slices, base RMS and
non-RMS `DX` formula slices after the reductions have been supplied, and one
integrated non-RMS+bias row slice for partial `DW`/`DB`. The dimension-general
`*_core_outputs_compute_correct_general` theorems group those pieces by the
backward cases observed by `test_layer_norm_fwd_bwd`. The "Python test-shape aggregate outputs" section at the
bottom of the file proves `Y` and `Rstd` outputs of the actual surface kernel
`layer_norm_fwd_1pass_surface` itself, specialised to the RMS/no-residual/no-bias
branch — the first per-kernel-surface (not slice) correctness in the file. Full
backward surface lowering is proved by `layer_norm_bwd_surface_toAlgorithm_supported`;
the observable backward outputs for the Python-tested branches are covered by
the compositional row/reduction/store theorems below.
-/

/-- Proof-oriented RMS/no-residual/no-bias slice of `layer_norm_ops.py`'s
`_layer_norm_fwd_1pass_kernel`.

This specializes the constexpr branches to:
- `IS_RMS_NORM = true`
- `HAS_RESIDUAL = false`
- `STORE_RESIDUAL_OUT = false`
- `HAS_BIAS = false`

It captures the row pointer arithmetic, masked load, RMS variance, `Rstd`
side-output store, weight multiply, and masked `Y` output store. -/
def layer_norm_fwd_rms_one_block
    (X Y W Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = x * rstd
  y = x_hat * w
  tl.store(Y + cols, y, mask=mask)
}

def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val

def yOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_y_row + i.val

noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ)
      else some (0.0 : ℝ) }

noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride_x_row N BLOCK_N)
        (rmsInputTile s X stride_x_row N BLOCK_N))).data PUnit.unit)

noncomputable def rmsInvCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsVarCarrier s X stride_x_row N BLOCK_N)))

noncomputable def rmsYSpec
    (s : BlockState) (X W : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (s.readMem X (xOffset s stride_x_row i)))
        (rmsInvCarrier s X stride_x_row N BLOCK_N eps))
      (some (s.readMem W i.val)))

/-- Algorithm-layer correctness for the `Y` output of the RMS slice. -/
theorem layer_norm_fwd_rms_one_block_y_correct
    (X Y W Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hYRstd : Y ≠ Rstd)
    (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_rms_one_block X Y W Rstd
        stride_x_row stride_y_row N BLOCK_N eps) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          rmsYSpec s X W stride_x_row N BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_rms_one_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWRstd, rmsYSpec, rmsInvCarrier, rmsVarCarrier, rmsInputTile,
            xOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYRstd]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the `Y` output of the RMS slice. -/
theorem layer_norm_fwd_rms_one_block_y_compute_correct
    (X Y W Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYRstd : Y ≠ Rstd)
    (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_rms_one_block X Y W Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i => rmsYSpec s X W stride_x_row N BLOCK_N eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_rms_one_block, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_fwd_rms_one_block_y_correct X Y W Rstd stride_x_row
    stride_y_row N BLOCK_N eps s s' hYRstd hWRstd hOutInj hExec i
  simpa [hActive] using h

/-! ## Forward RMS bias arithmetic slice

This covers `IS_RMS_NORM = true`, `HAS_RESIDUAL = false`, and `HAS_BIAS = true`
for the Python forward kernel. -/

def layer_norm_fwd_rms_bias_one_block
    (X Y W B Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  x_hat = x * rstd
  y = x_hat * w + b
  tl.store(Y + cols, y, mask=mask)
}

noncomputable def rmsBiasYSpec
    (s : BlockState) (X W B : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun weighted bias => weighted + bias)
      (Option.map₂ (fun scaled w => scaled * w)
        (Option.map₂ (fun x inv => x * inv)
          (some (s.readMem X (xOffset s stride_x_row i)))
          (rmsInvCarrier s X stride_x_row N BLOCK_N eps))
        (some (s.readMem W i.val)))
      (some (s.readMem B i.val)))

theorem layer_norm_fwd_rms_bias_one_block_y_correct
    (X Y W B Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hYRstd : Y ≠ Rstd)
    (hWRstd : W ≠ Rstd)
    (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_rms_bias_one_block X Y W B Rstd
        stride_x_row stride_y_row N BLOCK_N eps) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          rmsBiasYSpec s X W B stride_x_row N BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_rms_bias_one_block, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.mul, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWRstd, hBRstd, rmsBiasYSpec, rmsInvCarrier, rmsVarCarrier,
            rmsInputTile, xOffset, Tile.reduceSum, Tile.reduceSumDrop,
            Tile.select, TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex, WithBot.realSqrt, NumericDType.mul,
            NumericDType.add]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYRstd]
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem layer_norm_fwd_rms_bias_one_block_y_compute_correct
    (X Y W B Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYRstd : Y ≠ Rstd)
    (hWRstd : W ≠ Rstd)
    (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_rms_bias_one_block X Y W B Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i =>
        rmsBiasYSpec s X W B stride_x_row N BLOCK_N eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_rms_bias_one_block, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_fwd_rms_bias_one_block_y_correct X Y W B Rstd
    stride_x_row stride_y_row N BLOCK_N eps s s' hYRstd hWRstd hBRstd
    hOutInj hExec i
  simpa [hActive] using h

/-- Full-kernel spec for the `Rstd` scalar store of `layer_norm_fwd_rms_one_block`.
Wraps `rmsInvCarrier` with `WithBot.unbotD 0` to match the kernel's
post-execution scalar readback. -/
noncomputable def rmsInvVarFullSpec
    (s : BlockState) (X : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (rmsInvCarrier s X stride_x_row N BLOCK_N eps)

/-- Executed-state correctness for the `Rstd` scalar store of
`layer_norm_fwd_rms_one_block`. The kernel writes a single element at offset
`s.pid` into `Rstd`; this theorem strips the trailing `Y` foldl to expose that
store. -/
theorem layer_norm_fwd_rms_one_block_rstd_correct
    (X Y W Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hRstdY : Rstd ≠ Y)
    (hExec : exec (layer_norm_fwd_rms_one_block X Y W Rstd
        stride_x_row stride_y_row N BLOCK_N eps) s = some s') :
    s'.readMem Rstd s.pid =
      rmsInvVarFullSpec s X stride_x_row N BLOCK_N eps := by
  simp [exec, layer_norm_fwd_rms_one_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRstdY]
  simp only [BlockState.setReg_readMem, BlockState.writeMem_readMem]
  simp [rmsInvVarFullSpec, rmsInvCarrier, rmsVarCarrier, rmsInputTile,
        xOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realSqrt, NumericDType.mul]
  rfl

/-- Compute-facing correctness for the `Rstd` scalar store of
`layer_norm_fwd_rms_one_block`. -/
theorem layer_norm_fwd_rms_one_block_rstd_compute_correct
    (X Y W Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_rms_one_block X Y W Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        rmsInvVarFullSpec s X stride_x_row N BLOCK_N eps) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_rms_one_block, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_fwd_rms_one_block_rstd_correct X Y W Rstd stride_x_row
    stride_y_row N BLOCK_N eps s s' hRstdY hExec

theorem layer_norm_fwd_rms_bias_one_block_rstd_correct
    (X Y W B Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hRstdY : Rstd ≠ Y)
    (hExec : exec (layer_norm_fwd_rms_bias_one_block X Y W B Rstd
        stride_x_row stride_y_row N BLOCK_N eps) s = some s') :
    s'.readMem Rstd s.pid =
      rmsInvVarFullSpec s X stride_x_row N BLOCK_N eps := by
  simp [exec, layer_norm_fwd_rms_bias_one_block, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.mul, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRstdY]
  simp only [BlockState.setReg_readMem, BlockState.writeMem_readMem]
  simp [rmsInvVarFullSpec, rmsInvCarrier, rmsVarCarrier, rmsInputTile,
        xOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realSqrt, NumericDType.mul]
  rfl

theorem layer_norm_fwd_rms_bias_one_block_rstd_compute_correct
    (X Y W B Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_rms_bias_one_block X Y W B Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        rmsInvVarFullSpec s X stride_x_row N BLOCK_N eps) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_rms_bias_one_block, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_fwd_rms_bias_one_block_rstd_correct X Y W B Rstd
    stride_x_row stride_y_row N BLOCK_N eps s s' hRstdY hExec

/-- Proof-oriented non-RMS/no-residual/no-bias slice of the forward kernel.

This covers the layer-norm arithmetic branch: mean, centered variance, `Rstd`,
weight multiply, and masked `Y` output. -/
def layer_norm_fwd_plain_one_block
    (X Y W Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w
  tl.store(Y + cols, y, mask=mask)
}

noncomputable def plainInputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride_x_row idx.1))
      else some (0.0 : ℝ) }

noncomputable def plainMeanCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (plainInputTile s X stride_x_row N BLOCK_N)).data PUnit.unit)

noncomputable def plainXbarTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        Option.map₂ (fun x mean => x - mean)
          ((plainInputTile s X stride_x_row N BLOCK_N).data idx)
          (plainMeanCarrier s X stride_x_row N BLOCK_N)
      else some (0.0 : ℝ) }

noncomputable def plainVarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (plainXbarTile s X stride_x_row N BLOCK_N)
        (plainXbarTile s X stride_x_row N BLOCK_N))).data PUnit.unit)

noncomputable def plainInvCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (plainVarCarrier s X stride_x_row N BLOCK_N)))

noncomputable def plainYSpec
    (s : BlockState) (X W : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun centered inv => centered * inv)
        (Option.map₂ (fun x mean => x - mean)
          (some (s.readMem X (xOffset s stride_x_row i)))
          (plainMeanCarrier s X stride_x_row N BLOCK_N))
        (plainInvCarrier s X stride_x_row N BLOCK_N eps))
      (some (s.readMem W i.val)))

theorem layer_norm_fwd_plain_one_block_y_correct
    (X Y W Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hYMean : Y ≠ Mean)
    (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean)
    (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_plain_one_block X Y W Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          plainYSpec s X W stride_x_row N BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_plain_one_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.sub,
          NumericDType.mul, NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWMean, hWRstd, plainYSpec, plainInvCarrier, plainVarCarrier,
            plainXbarTile, plainMeanCarrier, plainInputTile, xOffset,
            Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul, NumericDType.sub]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYMean, hYRstd]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem layer_norm_fwd_plain_one_block_y_compute_correct
    (X Y W Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYMean : Y ≠ Mean)
    (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean)
    (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_plain_one_block X Y W Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i => plainYSpec s X W stride_x_row N BLOCK_N eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_plain_one_block, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_fwd_plain_one_block_y_correct X Y W Mean Rstd
    stride_x_row stride_y_row N BLOCK_N eps s s' hYMean hYRstd hWMean hWRstd
    hOutInj hExec i
  simpa [hActive] using h

/-! ## Forward plain bias arithmetic slice

This covers `IS_RMS_NORM = false`, `HAS_RESIDUAL = false`, and
`HAS_BIAS = true` for the Python forward kernel. -/

def layer_norm_fwd_plain_bias_one_block
    (X Y W B Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w + b
  tl.store(Y + cols, y, mask=mask)
}

noncomputable def plainBiasYSpec
    (s : BlockState) (X W B : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun weighted bias => weighted + bias)
      (Option.map₂ (fun scaled w => scaled * w)
        (Option.map₂ (fun centered inv => centered * inv)
          (Option.map₂ (fun x mean => x - mean)
            (some (s.readMem X (xOffset s stride_x_row i)))
            (plainMeanCarrier s X stride_x_row N BLOCK_N))
          (plainInvCarrier s X stride_x_row N BLOCK_N eps))
        (some (s.readMem W i.val)))
      (some (s.readMem B i.val)))

theorem layer_norm_fwd_plain_bias_one_block_y_correct
    (X Y W B Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hYMean : Y ≠ Mean)
    (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean)
    (hWRstd : W ≠ Rstd)
    (hBMean : B ≠ Mean)
    (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_plain_bias_one_block X Y W B Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          plainBiasYSpec s X W B stride_x_row N BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_plain_bias_one_block, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.sub, NumericDType.mul, NumericDType.div,
          ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
          FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
          ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWMean, hWRstd, hBMean, hBRstd, plainBiasYSpec,
            plainInvCarrier, plainVarCarrier, plainXbarTile,
            plainMeanCarrier, plainInputTile, xOffset, Tile.reduceSum,
            Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
            NumericDType.mul, NumericDType.sub, NumericDType.add]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYMean, hYRstd]
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem layer_norm_fwd_plain_bias_one_block_y_compute_correct
    (X Y W B Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYMean : Y ≠ Mean)
    (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean)
    (hWRstd : W ≠ Rstd)
    (hBMean : B ≠ Mean)
    (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_plain_bias_one_block X Y W B Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i =>
        plainBiasYSpec s X W B stride_x_row N BLOCK_N eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_plain_bias_one_block, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_fwd_plain_bias_one_block_y_correct X Y W B Mean Rstd
    stride_x_row stride_y_row N BLOCK_N eps s s' hYMean hYRstd hWMean hWRstd
    hBMean hBRstd hOutInj hExec i
  simpa [hActive] using h

/-! ## Forward residual arithmetic slice

This covers the non-RMS/no-bias forward branch with `HAS_RESIDUAL = true`.
Unlike the branch-named writeback aliases below, this slice computes the
residual addition before the mean/variance/rstd/Y arithmetic. -/

def layer_norm_fwd_plain_residual_one_block
    (X RESIDUAL Y W Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  RESIDUAL += row * $(stride_res_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x += residual
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w
  tl.store(Y + cols, y, mask=mask)
}

def resOffset (s : BlockState) (stride_res_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_res_row + i.val

noncomputable def plainResidualInputTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      Option.map₂ (fun x residual : ℝ => x + residual)
        (if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ))
        (if idx.1.val < N then
          some (s.readMem RESIDUAL (resOffset s stride_res_row idx.1))
        else some (0.0 : ℝ)) }

noncomputable def plainResidualMeanCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N)).data PUnit.unit)

noncomputable def plainResidualXbarTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        Option.map₂ (fun x mean => x - mean)
          ((plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N).data idx)
          (plainResidualMeanCarrier s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N)
      else some (0.0 : ℝ) }

noncomputable def plainResidualVarCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (plainResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)
        (plainResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N))).data PUnit.unit)

noncomputable def plainResidualInvCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (plainResidualVarCarrier s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)))

noncomputable def plainResidualYSpec
    (s : BlockState) (X RESIDUAL W : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun centered inv => centered * inv)
        (Option.map₂ (fun x mean => x - mean)
          ((plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N).data (i, PUnit.unit))
          (plainResidualMeanCarrier s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N))
        (plainResidualInvCarrier s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N eps))
      (some (s.readMem W i.val)))

theorem layer_norm_fwd_plain_residual_one_block_y_correct
    (X RESIDUAL Y W Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hYMean : Y ≠ Mean)
    (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean)
    (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_plain_residual_one_block X RESIDUAL Y W
        Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps) s =
        some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          plainResidualYSpec s X RESIDUAL W stride_x_row stride_res_row N
            BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_plain_residual_one_block, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.sub, NumericDType.mul, NumericDType.div,
          ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
          FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
          ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWMean, hWRstd, plainResidualYSpec,
            plainResidualInvCarrier, plainResidualVarCarrier,
            plainResidualXbarTile, plainResidualMeanCarrier,
            plainResidualInputTile, xOffset, resOffset, Tile.reduceSum,
            Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
            NumericDType.mul, NumericDType.sub, NumericDType.add]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYMean, hYRstd]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem layer_norm_fwd_plain_residual_one_block_y_compute_correct
    (X RESIDUAL Y W Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYMean : Y ≠ Mean)
    (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean)
    (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_plain_residual_one_block X RESIDUAL Y W
        Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i =>
        plainResidualYSpec s X RESIDUAL W stride_x_row stride_res_row N
          BLOCK_N eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_plain_residual_one_block, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_fwd_plain_residual_one_block_y_correct X RESIDUAL Y W
    Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps s s'
    hYMean hYRstd hWMean hWRstd hOutInj hExec i
  simpa [hActive] using h

noncomputable def plainResidualMeanFullSpec
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0
    (plainResidualMeanCarrier s X RESIDUAL stride_x_row stride_res_row N
      BLOCK_N)

theorem layer_norm_fwd_plain_residual_one_block_mean_correct
    (X RESIDUAL Y W Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hMeanRstd : Mean ≠ Rstd)
    (hMeanY : Mean ≠ Y)
    (hExec : exec (layer_norm_fwd_plain_residual_one_block X RESIDUAL Y W
        Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps) s =
        some s') :
    s'.readMem Mean s.pid =
      plainResidualMeanFullSpec s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N := by
  simp [exec, layer_norm_fwd_plain_residual_one_block, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.sub, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hMeanY]
  simp [BlockState.writeMem_readMem, hMeanRstd, plainResidualMeanFullSpec,
        plainResidualMeanCarrier, plainResidualInputTile, xOffset, resOffset,
        Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex]
  rfl

theorem layer_norm_fwd_plain_residual_one_block_mean_compute_correct
    (X RESIDUAL Y W Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hMeanRstd : Mean ≠ Rstd)
    (hMeanY : Mean ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_plain_residual_one_block X RESIDUAL Y W
        Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, s.pid))
      (expected := fun _ =>
        plainResidualMeanFullSpec s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_plain_residual_one_block, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_fwd_plain_residual_one_block_mean_correct X RESIDUAL Y W
    Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps s s'
    hMeanRstd hMeanY hExec

noncomputable def plainResidualInvVarFullSpec
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (plainResidualInvCarrier s X RESIDUAL stride_x_row stride_res_row N
      BLOCK_N eps)

theorem layer_norm_fwd_plain_residual_one_block_rstd_correct
    (X RESIDUAL Y W Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hRstdY : Rstd ≠ Y)
    (hExec : exec (layer_norm_fwd_plain_residual_one_block X RESIDUAL Y W
        Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps) s =
        some s') :
    s'.readMem Rstd s.pid =
      plainResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N eps := by
  simp [exec, layer_norm_fwd_plain_residual_one_block, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.sub, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRstdY]
  simp only [BlockState.setReg_readMem, BlockState.writeMem_readMem]
  simp [plainResidualInvVarFullSpec, plainResidualInvCarrier,
        plainResidualVarCarrier, plainResidualXbarTile,
        plainResidualMeanCarrier, plainResidualInputTile, xOffset, resOffset,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realSqrt, NumericDType.mul, NumericDType.sub,
        NumericDType.add]
  rfl

theorem layer_norm_fwd_plain_residual_one_block_rstd_compute_correct
    (X RESIDUAL Y W Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_plain_residual_one_block X RESIDUAL Y W
        Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        plainResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N eps) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_plain_residual_one_block, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_fwd_plain_residual_one_block_rstd_correct X RESIDUAL Y W
    Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps s s'
    hRstdY hExec

/-! ## Forward RMS residual arithmetic slice

This covers `IS_RMS_NORM = true`, `HAS_RESIDUAL = true`, and `HAS_BIAS = false`
for the Python forward kernel. -/

def layer_norm_fwd_rms_residual_one_block
    (X RESIDUAL Y W Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  RESIDUAL += row * $(stride_res_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x += residual
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = x * rstd
  y = x_hat * w
  tl.store(Y + cols, y, mask=mask)
}

noncomputable def rmsResidualXbarTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        (plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N).data idx
      else some (0.0 : ℝ) }

noncomputable def rmsResidualVarCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)
        (rmsResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N))).data PUnit.unit)

noncomputable def rmsResidualInvCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsResidualVarCarrier s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)))

noncomputable def rmsResidualYSpec
    (s : BlockState) (X RESIDUAL W : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun x inv => x * inv)
        ((plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N).data (i, PUnit.unit))
        (rmsResidualInvCarrier s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N eps))
      (some (s.readMem W i.val)))

theorem layer_norm_fwd_rms_residual_one_block_y_correct
    (X RESIDUAL Y W Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hYRstd : Y ≠ Rstd)
    (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_rms_residual_one_block X RESIDUAL Y W
        Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps) s =
        some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          rmsResidualYSpec s X RESIDUAL W stride_x_row stride_res_row N
            BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_rms_residual_one_block, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.mul, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWRstd, rmsResidualYSpec, rmsResidualInvCarrier,
            rmsResidualVarCarrier, plainResidualInputTile, xOffset, resOffset,
            Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul, NumericDType.add]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYRstd]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem layer_norm_fwd_rms_residual_one_block_y_compute_correct
    (X RESIDUAL Y W Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYRstd : Y ≠ Rstd)
    (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_rms_residual_one_block X RESIDUAL Y W Rstd
        stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i =>
        rmsResidualYSpec s X RESIDUAL W stride_x_row stride_res_row N
          BLOCK_N eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_rms_residual_one_block, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_fwd_rms_residual_one_block_y_correct X RESIDUAL Y W
    Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps s s' hYRstd
    hWRstd hOutInj hExec i
  simpa [hActive] using h

noncomputable def rmsResidualInvVarFullSpec
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (rmsResidualInvCarrier s X RESIDUAL stride_x_row stride_res_row N
      BLOCK_N eps)

theorem layer_norm_fwd_rms_residual_one_block_rstd_correct
    (X RESIDUAL Y W Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hRstdY : Rstd ≠ Y)
    (hExec : exec (layer_norm_fwd_rms_residual_one_block X RESIDUAL Y W
        Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps) s =
        some s') :
    s'.readMem Rstd s.pid =
      rmsResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N eps := by
  simp [exec, layer_norm_fwd_rms_residual_one_block, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.mul, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRstdY]
  simp only [BlockState.setReg_readMem, BlockState.writeMem_readMem]
  simp [rmsResidualInvVarFullSpec, rmsResidualInvCarrier,
        rmsResidualVarCarrier, rmsResidualXbarTile, plainResidualInputTile,
        xOffset, resOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realSqrt, NumericDType.mul, NumericDType.add]
  rfl

theorem layer_norm_fwd_rms_residual_one_block_rstd_compute_correct
    (X RESIDUAL Y W Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_rms_residual_one_block X RESIDUAL Y W Rstd
        stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        rmsResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N eps) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_rms_residual_one_block, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_fwd_rms_residual_one_block_rstd_correct X RESIDUAL Y W
    Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps s s' hRstdY
    hExec

/-! ## Forward RMS residual+bias arithmetic slice

This covers `IS_RMS_NORM = true`, `HAS_RESIDUAL = true`, and `HAS_BIAS = true`
for the Python forward kernel. -/

def layer_norm_fwd_rms_residual_bias_one_block
    (X RESIDUAL Y W B Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  RESIDUAL += row * $(stride_res_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x += residual
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  x_hat = x * rstd
  y = x_hat * w + b
  tl.store(Y + cols, y, mask=mask)
}

noncomputable def rmsResidualBiasYSpec
    (s : BlockState) (X RESIDUAL W B : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun weighted bias => weighted + bias)
      (Option.map₂ (fun scaled w => scaled * w)
        (Option.map₂ (fun x inv => x * inv)
          ((plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N).data (i, PUnit.unit))
          (rmsResidualInvCarrier s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N eps))
        (some (s.readMem W i.val)))
      (some (s.readMem B i.val)))

theorem layer_norm_fwd_rms_residual_bias_one_block_y_correct
    (X RESIDUAL Y W B Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hYRstd : Y ≠ Rstd)
    (hWRstd : W ≠ Rstd)
    (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_rms_residual_bias_one_block X RESIDUAL Y W
        B Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps) s =
        some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          rmsResidualBiasYSpec s X RESIDUAL W B stride_x_row stride_res_row N
            BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_rms_residual_bias_one_block, stepStmts,
          stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.div,
          ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
          FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
          ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWRstd, hBRstd, rmsResidualBiasYSpec,
            rmsResidualInvCarrier, rmsResidualVarCarrier,
            rmsResidualXbarTile, plainResidualInputTile, xOffset, resOffset,
            Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul, NumericDType.add]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYRstd]
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem layer_norm_fwd_rms_residual_bias_one_block_y_compute_correct
    (X RESIDUAL Y W B Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYRstd : Y ≠ Rstd)
    (hWRstd : W ≠ Rstd)
    (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_rms_residual_bias_one_block X RESIDUAL Y W
        B Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i =>
        rmsResidualBiasYSpec s X RESIDUAL W B stride_x_row stride_res_row N
          BLOCK_N eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_rms_residual_bias_one_block, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_fwd_rms_residual_bias_one_block_y_correct X RESIDUAL Y
    W B Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps s s'
    hYRstd hWRstd hBRstd hOutInj hExec i
  simpa [hActive] using h

theorem layer_norm_fwd_rms_residual_bias_one_block_rstd_correct
    (X RESIDUAL Y W B Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hRstdY : Rstd ≠ Y)
    (hExec : exec (layer_norm_fwd_rms_residual_bias_one_block X RESIDUAL Y W
        B Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps) s =
        some s') :
    s'.readMem Rstd s.pid =
      rmsResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N eps := by
  simp [exec, layer_norm_fwd_rms_residual_bias_one_block, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.mul, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRstdY]
  simp only [BlockState.setReg_readMem, BlockState.writeMem_readMem]
  simp [rmsResidualInvVarFullSpec, rmsResidualInvCarrier,
        rmsResidualVarCarrier, rmsResidualXbarTile, plainResidualInputTile,
        xOffset, resOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realSqrt, NumericDType.mul, NumericDType.add]
  rfl

theorem layer_norm_fwd_rms_residual_bias_one_block_rstd_compute_correct
    (X RESIDUAL Y W B Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_rms_residual_bias_one_block X RESIDUAL Y W
        B Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        rmsResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N eps) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_rms_residual_bias_one_block, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_fwd_rms_residual_bias_one_block_rstd_correct X RESIDUAL Y W
    B Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps s s'
    hRstdY hExec

/-! ## Forward plain residual+bias arithmetic slice

This covers `IS_RMS_NORM = false`, `HAS_RESIDUAL = true`, and
`HAS_BIAS = true` for the Python forward kernel. -/

def layer_norm_fwd_plain_residual_bias_one_block
    (X RESIDUAL Y W B Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  RESIDUAL += row * $(stride_res_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x += residual
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w + b
  tl.store(Y + cols, y, mask=mask)
}

noncomputable def plainResidualBiasYSpec
    (s : BlockState) (X RESIDUAL W B : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun weighted bias => weighted + bias)
      (Option.map₂ (fun scaled w => scaled * w)
        (Option.map₂ (fun centered inv => centered * inv)
          (Option.map₂ (fun x mean => x - mean)
            ((plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
              BLOCK_N).data (i, PUnit.unit))
            (plainResidualMeanCarrier s X RESIDUAL stride_x_row stride_res_row N
              BLOCK_N))
          (plainResidualInvCarrier s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N eps))
        (some (s.readMem W i.val)))
      (some (s.readMem B i.val)))

theorem layer_norm_fwd_plain_residual_bias_one_block_y_correct
    (X RESIDUAL Y W B Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hYMean : Y ≠ Mean)
    (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean)
    (hWRstd : W ≠ Rstd)
    (hBMean : B ≠ Mean)
    (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_plain_residual_bias_one_block X RESIDUAL Y
        W B Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
        s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          plainResidualBiasYSpec s X RESIDUAL W B stride_x_row stride_res_row
            N BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_plain_residual_bias_one_block, stepStmts,
          stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.sub, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWMean, hWRstd, hBMean, hBRstd, plainResidualBiasYSpec,
            plainResidualInvCarrier, plainResidualVarCarrier,
            plainResidualXbarTile, plainResidualMeanCarrier,
            plainResidualInputTile, xOffset, resOffset, Tile.reduceSum,
            Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
            NumericDType.mul, NumericDType.sub, NumericDType.add]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYMean, hYRstd]
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem layer_norm_fwd_plain_residual_bias_one_block_y_compute_correct
    (X RESIDUAL Y W B Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYMean : Y ≠ Mean)
    (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean)
    (hWRstd : W ≠ Rstd)
    (hBMean : B ≠ Mean)
    (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_plain_residual_bias_one_block X RESIDUAL Y
        W B Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i =>
        plainResidualBiasYSpec s X RESIDUAL W B stride_x_row stride_res_row N
          BLOCK_N eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_plain_residual_bias_one_block,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_fwd_plain_residual_bias_one_block_y_correct X RESIDUAL
    Y W B Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps s
    s' hYMean hYRstd hWMean hWRstd hBMean hBRstd hOutInj hExec i
  simpa [hActive] using h

theorem layer_norm_fwd_plain_residual_bias_one_block_mean_correct
    (X RESIDUAL Y W B Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hMeanRstd : Mean ≠ Rstd)
    (hMeanY : Mean ≠ Y)
    (hExec : exec (layer_norm_fwd_plain_residual_bias_one_block X RESIDUAL Y
        W B Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
        s = some s') :
    s'.readMem Mean s.pid =
      plainResidualMeanFullSpec s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N := by
  simp [exec, layer_norm_fwd_plain_residual_bias_one_block, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.sub, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hMeanY]
  simp [BlockState.writeMem_readMem, hMeanRstd, plainResidualMeanFullSpec,
        plainResidualMeanCarrier, plainResidualInputTile, xOffset, resOffset,
        Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex]
  rfl

theorem layer_norm_fwd_plain_residual_bias_one_block_mean_compute_correct
    (X RESIDUAL Y W B Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hMeanRstd : Mean ≠ Rstd)
    (hMeanY : Mean ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_plain_residual_bias_one_block X RESIDUAL Y
        W B Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, s.pid))
      (expected := fun _ =>
        plainResidualMeanFullSpec s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_plain_residual_bias_one_block,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_fwd_plain_residual_bias_one_block_mean_correct X RESIDUAL
    Y W B Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps s
    s' hMeanRstd hMeanY hExec

theorem layer_norm_fwd_plain_residual_bias_one_block_rstd_correct
    (X RESIDUAL Y W B Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hRstdY : Rstd ≠ Y)
    (hExec : exec (layer_norm_fwd_plain_residual_bias_one_block X RESIDUAL Y
        W B Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
        s = some s') :
    s'.readMem Rstd s.pid =
      plainResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N eps := by
  simp [exec, layer_norm_fwd_plain_residual_bias_one_block, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.sub, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRstdY]
  simp only [BlockState.setReg_readMem, BlockState.writeMem_readMem]
  simp [plainResidualInvVarFullSpec, plainResidualInvCarrier,
        plainResidualVarCarrier, plainResidualXbarTile,
        plainResidualMeanCarrier, plainResidualInputTile, xOffset, resOffset,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realSqrt, NumericDType.mul, NumericDType.sub,
        NumericDType.add]
  rfl

theorem layer_norm_fwd_plain_residual_bias_one_block_rstd_compute_correct
    (X RESIDUAL Y W B Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_plain_residual_bias_one_block X RESIDUAL Y
        W B Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        plainResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N eps) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_plain_residual_bias_one_block,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_fwd_plain_residual_bias_one_block_rstd_correct X RESIDUAL
    Y W B Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps s
    s' hRstdY hExec

noncomputable def plainMeanFullSpec
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0 (plainMeanCarrier s X stride_x_row N BLOCK_N)

theorem layer_norm_fwd_plain_one_block_mean_correct
    (X Y W Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hMeanRstd : Mean ≠ Rstd)
    (hMeanY : Mean ≠ Y)
    (hExec : exec (layer_norm_fwd_plain_one_block X Y W Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps) s = some s') :
    s'.readMem Mean s.pid =
      plainMeanFullSpec s X stride_x_row N BLOCK_N := by
  simp [exec, layer_norm_fwd_plain_one_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.sub,
        NumericDType.mul, NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hMeanY]
  simp [BlockState.writeMem_readMem, hMeanRstd, plainMeanFullSpec,
        plainMeanCarrier, plainInputTile, xOffset, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex]
  rfl

theorem layer_norm_fwd_plain_one_block_mean_compute_correct
    (X Y W Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hMeanRstd : Mean ≠ Rstd)
    (hMeanY : Mean ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_plain_one_block X Y W Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, s.pid))
      (expected := fun _ =>
        plainMeanFullSpec s X stride_x_row N BLOCK_N) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_plain_one_block, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_fwd_plain_one_block_mean_correct X Y W Mean Rstd
    stride_x_row stride_y_row N BLOCK_N eps s s' hMeanRstd hMeanY hExec

noncomputable def plainInvVarFullSpec
    (s : BlockState) (X : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (plainInvCarrier s X stride_x_row N BLOCK_N eps)

theorem layer_norm_fwd_plain_one_block_rstd_correct
    (X Y W Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hRstdY : Rstd ≠ Y)
    (hExec : exec (layer_norm_fwd_plain_one_block X Y W Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps) s = some s') :
    s'.readMem Rstd s.pid =
      plainInvVarFullSpec s X stride_x_row N BLOCK_N eps := by
  simp [exec, layer_norm_fwd_plain_one_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.sub,
        NumericDType.mul, NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRstdY]
  simp only [BlockState.setReg_readMem, BlockState.writeMem_readMem]
  simp [plainInvVarFullSpec, plainInvCarrier, plainVarCarrier, plainXbarTile,
        plainMeanCarrier, plainInputTile, xOffset, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
        NumericDType.mul, NumericDType.sub]
  rfl

theorem layer_norm_fwd_plain_one_block_rstd_compute_correct
    (X Y W Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_plain_one_block X Y W Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        plainInvVarFullSpec s X stride_x_row N BLOCK_N eps) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_plain_one_block, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_fwd_plain_one_block_rstd_correct X Y W Mean Rstd
    stride_x_row stride_y_row N BLOCK_N eps s s' hRstdY hExec

theorem layer_norm_fwd_plain_bias_one_block_mean_correct
    (X Y W B Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hMeanRstd : Mean ≠ Rstd)
    (hMeanY : Mean ≠ Y)
    (hExec : exec (layer_norm_fwd_plain_bias_one_block X Y W B Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps) s = some s') :
    s'.readMem Mean s.pid =
      plainMeanFullSpec s X stride_x_row N BLOCK_N := by
  simp [exec, layer_norm_fwd_plain_bias_one_block, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.sub, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hMeanY]
  simp [BlockState.writeMem_readMem, hMeanRstd, plainMeanFullSpec,
        plainMeanCarrier, plainInputTile, xOffset, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex]
  rfl

theorem layer_norm_fwd_plain_bias_one_block_mean_compute_correct
    (X Y W B Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hMeanRstd : Mean ≠ Rstd)
    (hMeanY : Mean ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_plain_bias_one_block X Y W B Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, s.pid))
      (expected := fun _ =>
        plainMeanFullSpec s X stride_x_row N BLOCK_N) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_plain_bias_one_block, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_fwd_plain_bias_one_block_mean_correct X Y W B Mean Rstd
    stride_x_row stride_y_row N BLOCK_N eps s s' hMeanRstd hMeanY hExec

theorem layer_norm_fwd_plain_bias_one_block_rstd_correct
    (X Y W B Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hRstdY : Rstd ≠ Y)
    (hExec : exec (layer_norm_fwd_plain_bias_one_block X Y W B Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps) s = some s') :
    s'.readMem Rstd s.pid =
      plainInvVarFullSpec s X stride_x_row N BLOCK_N eps := by
  simp [exec, layer_norm_fwd_plain_bias_one_block, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.sub, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRstdY]
  simp only [BlockState.setReg_readMem, BlockState.writeMem_readMem]
  simp [plainInvVarFullSpec, plainInvCarrier, plainVarCarrier, plainXbarTile,
        plainMeanCarrier, plainInputTile, xOffset, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
        NumericDType.mul, NumericDType.sub]
  rfl

theorem layer_norm_fwd_plain_bias_one_block_rstd_compute_correct
    (X Y W B Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_plain_bias_one_block X Y W B Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        plainInvVarFullSpec s X stride_x_row N BLOCK_N eps) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_plain_bias_one_block, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_fwd_plain_bias_one_block_rstd_correct X Y W B Mean Rstd
    stride_x_row stride_y_row N BLOCK_N eps s s' hRstdY hExec

/-- Proof-oriented Mean store slice of `layer_norm_ops.py`'s
`_layer_norm_fwd_1pass_kernel`. Takes a precomputed `MeanPre` scalar (per row)
and proves the unmasked scalar writeback into `Mean` at row offset. -/
def layer_norm_ops_fwd_mean_store_slice
    (MeanPre Mean : RegionName) : ComputeKernel := triton {
  row = tl.program_id(0)
  mean = tl.load(MeanPre + row)
  tl.store(Mean + row, mean)
}

def meanRowOffset (s : BlockState) : Nat := s.pid

noncomputable def meanStoreSpec (s : BlockState) (MeanPre : RegionName) : ℝ :=
  s.readMem MeanPre s.pid

theorem layer_norm_ops_fwd_mean_store_slice_correct
    (MeanPre Mean : RegionName) (s s' : BlockState)
    (hExec : exec (layer_norm_ops_fwd_mean_store_slice MeanPre Mean) s = some s') :
    s'.readMem Mean (meanRowOffset s) = meanStoreSpec s MeanPre := by
  simp [exec, layer_norm_ops_fwd_mean_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add] at hExec
  subst s'
  simp [meanRowOffset, meanStoreSpec]

theorem layer_norm_ops_fwd_mean_store_slice_compute_correct
    (MeanPre Mean : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_mean_store_slice MeanPre Mean)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, meanRowOffset s))
      (expected := fun _ => meanStoreSpec s MeanPre) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_fwd_mean_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_ops_fwd_mean_store_slice_correct MeanPre Mean s s' hExec

/-- Proof-oriented Rstd store slice. Same scalar-copy pattern. -/
def layer_norm_ops_fwd_rstd_store_slice
    (RstdPre Rstd : RegionName) : ComputeKernel := triton {
  row = tl.program_id(0)
  rstd = tl.load(RstdPre + row)
  tl.store(Rstd + row, rstd)
}

noncomputable def rstdStoreSpec (s : BlockState) (RstdPre : RegionName) : ℝ :=
  s.readMem RstdPre s.pid

theorem layer_norm_ops_fwd_rstd_store_slice_correct
    (RstdPre Rstd : RegionName) (s s' : BlockState)
    (hExec : exec (layer_norm_ops_fwd_rstd_store_slice RstdPre Rstd) s = some s') :
    s'.readMem Rstd (meanRowOffset s) = rstdStoreSpec s RstdPre := by
  simp [exec, layer_norm_ops_fwd_rstd_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add] at hExec
  subst s'
  simp [meanRowOffset, rstdStoreSpec]

theorem layer_norm_ops_fwd_rstd_store_slice_compute_correct
    (RstdPre Rstd : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_rstd_store_slice RstdPre Rstd)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, meanRowOffset s))
      (expected := fun _ => rstdStoreSpec s RstdPre) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_fwd_rstd_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_ops_fwd_rstd_store_slice_correct RstdPre Rstd s s' hExec

/-! ## Backward writeback slices

The full backward kernel computes `DX`, optional `DRESIDUAL_IN`, optional
recomputed `Y`, and per-program partial `DW`/`DB` inside a row loop. The slices
below start after those arithmetic/reduction values have been computed and
prove the Python-addressed masked writebacks. They are intentionally named as
writeback slices, not full backward correctness theorems. -/

/-- Backward row-vector writeback slice for `DX`, optional `DRESIDUAL_IN`, and
optional recomputed `Y`. The caller supplies the precomputed row tile in
`ValuePre`; the slice proves the masked store to `Out + row * stride + cols`. -/
def layer_norm_ops_bwd_row_vector_store_slice
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}

def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val

noncomputable def bwdRowVectorStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem ValuePre (bwdRowVectorOffset s stride_out_row i)

theorem layer_norm_ops_bwd_row_vector_store_slice_correct
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_out_row i))
    (hExec : exec (layer_norm_ops_bwd_row_vector_store_slice ValuePre Out
        stride_out_row N BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Out (bwdRowVectorOffset s stride_out_row i) =
        if i.val < N then
          bwdRowVectorStoreSpec s ValuePre stride_out_row i
        else
          s.readMem Out (bwdRowVectorOffset s stride_out_row i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 0 * stride_out_row + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [bwdRowVectorOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [exec, layer_norm_ops_bwd_row_vector_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        bwdRowVectorOffset] at hExec
  subst s'
  simp only [bwdRowVectorOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
        (i, PUnit.unit)]
  by_cases hi : i.val < N
  · simp [hi, bwdRowVectorStoreSpec, bwdRowVectorOffset]
  · simp [hi]

theorem layer_norm_ops_bwd_row_vector_store_slice_compute_correct
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_out_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_row_vector_store_slice ValuePre Out
        stride_out_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Out, bwdRowVectorOffset s stride_out_row i)))
      (expected := fun i =>
        bwdRowVectorStoreSpec s ValuePre stride_out_row i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_bwd_row_vector_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_ops_bwd_row_vector_store_slice_correct ValuePre Out
    stride_out_row N BLOCK_N s s' hOutInj hExec i
  simpa [hActive] using h

/-- Explicit `DX` writeback slice for the backward kernel. -/
abbrev layer_norm_ops_bwd_dx_store_slice
    (DXPre DX : RegionName) (stride_dx_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice DXPre DX stride_dx_row N BLOCK_N

theorem layer_norm_ops_bwd_dx_store_slice_correct
    (DXPre DX : RegionName) (stride_dx_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_dx_row i))
    (hExec : exec (layer_norm_ops_bwd_dx_store_slice DXPre DX
        stride_dx_row N BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem DX (bwdRowVectorOffset s stride_dx_row i) =
        if i.val < N then
          bwdRowVectorStoreSpec s DXPre stride_dx_row i
        else
          s.readMem DX (bwdRowVectorOffset s stride_dx_row i) :=
  layer_norm_ops_bwd_row_vector_store_slice_correct DXPre DX
    stride_dx_row N BLOCK_N s s' hOutInj hExec

theorem layer_norm_ops_bwd_dx_store_slice_compute_correct
    (DXPre DX : RegionName) (stride_dx_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_dx_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_dx_store_slice DXPre DX
        stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRowVectorOffset s stride_dx_row i)))
      (expected := fun i =>
        bwdRowVectorStoreSpec s DXPre stride_dx_row i) :=
  layer_norm_ops_bwd_row_vector_store_slice_compute_correct DXPre DX
    stride_dx_row N BLOCK_N s hOutInj

/-- Explicit optional `DRESIDUAL_IN` writeback slice for `STORE_DRESIDUAL`. -/
abbrev layer_norm_ops_bwd_dresidual_in_store_slice
    (DXPre DRESIDUAL_IN : RegionName)
    (stride_dres_in_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice DXPre DRESIDUAL_IN
    stride_dres_in_row N BLOCK_N

theorem layer_norm_ops_bwd_dresidual_in_store_slice_correct
    (DXPre DRESIDUAL_IN : RegionName)
    (stride_dres_in_row N BLOCK_N : Nat) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_dres_in_row i))
    (hExec : exec (layer_norm_ops_bwd_dresidual_in_store_slice DXPre
        DRESIDUAL_IN stride_dres_in_row N BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem DRESIDUAL_IN (bwdRowVectorOffset s stride_dres_in_row i) =
        if i.val < N then
          bwdRowVectorStoreSpec s DXPre stride_dres_in_row i
        else
          s.readMem DRESIDUAL_IN
            (bwdRowVectorOffset s stride_dres_in_row i) :=
  layer_norm_ops_bwd_row_vector_store_slice_correct DXPre DRESIDUAL_IN
    stride_dres_in_row N BLOCK_N s s' hOutInj hExec

theorem layer_norm_ops_bwd_dresidual_in_store_slice_compute_correct
    (DXPre DRESIDUAL_IN : RegionName)
    (stride_dres_in_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_dres_in_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_dresidual_in_store_slice DXPre
        DRESIDUAL_IN stride_dres_in_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DRESIDUAL_IN,
          bwdRowVectorOffset s stride_dres_in_row i)))
      (expected := fun i =>
        bwdRowVectorStoreSpec s DXPre stride_dres_in_row i) :=
  layer_norm_ops_bwd_row_vector_store_slice_compute_correct DXPre DRESIDUAL_IN
    stride_dres_in_row N BLOCK_N s hOutInj

/-- Explicit optional recomputed-`Y` writeback slice for `RECOMPUTE_OUTPUT`. -/
abbrev layer_norm_ops_bwd_recompute_y_store_slice
    (YPre Y : RegionName) (stride_y_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice YPre Y stride_y_row N BLOCK_N

theorem layer_norm_ops_bwd_recompute_y_store_slice_correct
    (YPre Y : RegionName) (stride_y_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_y_row i))
    (hExec : exec (layer_norm_ops_bwd_recompute_y_store_slice YPre Y
        stride_y_row N BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (bwdRowVectorOffset s stride_y_row i) =
        if i.val < N then
          bwdRowVectorStoreSpec s YPre stride_y_row i
        else
          s.readMem Y (bwdRowVectorOffset s stride_y_row i) :=
  layer_norm_ops_bwd_row_vector_store_slice_correct YPre Y
    stride_y_row N BLOCK_N s s' hOutInj hExec

theorem layer_norm_ops_bwd_recompute_y_store_slice_compute_correct
    (YPre Y : RegionName) (stride_y_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_recompute_y_store_slice YPre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, bwdRowVectorOffset s stride_y_row i)))
      (expected := fun i =>
        bwdRowVectorStoreSpec s YPre stride_y_row i) :=
  layer_norm_ops_bwd_row_vector_store_slice_compute_correct YPre Y
    stride_y_row N BLOCK_N s hOutInj

/-- Forward residual-output writeback slice for `_layer_norm_fwd_1pass_kernel`.

When `STORE_RESIDUAL_OUT` is true, Python stores the row tile `x` after the
optional residual add into `RESIDUAL_OUT + cols`. The caller supplies that
precomputed row tile in `ValuePre`; this theorem proves the same masked
row-vector writeback shape as the backward row-vector stores. -/
abbrev layer_norm_ops_fwd_residual_out_store_slice
    (ValuePre RESIDUAL_OUT : RegionName)
    (stride_res_out_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre RESIDUAL_OUT
    stride_res_out_row N BLOCK_N

def fwdResidualOutOffset (s : BlockState) (stride_res_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_res_out_row i

noncomputable def fwdResidualOutStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_res_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_res_out_row i

theorem layer_norm_ops_fwd_residual_out_store_slice_correct
    (ValuePre RESIDUAL_OUT : RegionName)
    (stride_res_out_row N BLOCK_N : Nat) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdResidualOutOffset s stride_res_out_row i))
    (hExec : exec (layer_norm_ops_fwd_residual_out_store_slice ValuePre
        RESIDUAL_OUT stride_res_out_row N BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem RESIDUAL_OUT (fwdResidualOutOffset s stride_res_out_row i) =
        if i.val < N then
          fwdResidualOutStoreSpec s ValuePre stride_res_out_row i
        else
          s.readMem RESIDUAL_OUT
            (fwdResidualOutOffset s stride_res_out_row i) := by
  intro i
  have hInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_res_out_row i) := by
    simpa [fwdResidualOutOffset] using hOutInj
  have h := layer_norm_ops_bwd_row_vector_store_slice_correct ValuePre
    RESIDUAL_OUT stride_res_out_row N BLOCK_N s s' hInj hExec i
  simpa [fwdResidualOutOffset, fwdResidualOutStoreSpec] using h

theorem layer_norm_ops_fwd_residual_out_store_slice_compute_correct
    (ValuePre RESIDUAL_OUT : RegionName)
    (stride_res_out_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdResidualOutOffset s stride_res_out_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_residual_out_store_slice ValuePre
        RESIDUAL_OUT stride_res_out_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (RESIDUAL_OUT,
          fwdResidualOutOffset s stride_res_out_row i)))
      (expected := fun i =>
        fwdResidualOutStoreSpec s ValuePre stride_res_out_row i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_fwd_residual_out_store_slice,
      layer_norm_ops_bwd_row_vector_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_ops_fwd_residual_out_store_slice_correct ValuePre
    RESIDUAL_OUT stride_res_out_row N BLOCK_N s s' hOutInj hExec i
  simpa [hActive] using h

/-- Forward output writeback slice for `_layer_norm_fwd_1pass_kernel`.

After the mean/rstd arithmetic and optional bias/residual branches, Python
stores the computed row tile `y` into `Y + cols`. This aliases the generic
row-vector writeback proof with the forward output stride. -/
abbrev layer_norm_ops_fwd_y_store_slice
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre Y stride_y_row N BLOCK_N

def fwdYOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_y_row i

noncomputable def fwdYStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_y_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_y_row i

theorem layer_norm_ops_fwd_y_store_slice_correct
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i))
    (hExec : exec (layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (fwdYOffset s stride_y_row i) =
        if i.val < N then
          fwdYStoreSpec s ValuePre stride_y_row i
        else
          s.readMem Y (fwdYOffset s stride_y_row i) := by
  intro i
  have hInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_y_row i) := by
    simpa [fwdYOffset] using hOutInj
  have h := layer_norm_ops_bwd_row_vector_store_slice_correct ValuePre
    Y stride_y_row N BLOCK_N s s' hInj hExec i
  simpa [fwdYOffset, fwdYStoreSpec] using h

theorem layer_norm_ops_fwd_y_store_slice_compute_correct
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i => fwdYStoreSpec s ValuePre stride_y_row i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_fwd_y_store_slice,
      layer_norm_ops_bwd_row_vector_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_ops_fwd_y_store_slice_correct ValuePre Y
    stride_y_row N BLOCK_N s s' hOutInj hExec i
  simpa [hActive] using h

/-- Named forward Y writeback for the bias branch of
`_layer_norm_fwd_1pass_kernel`.

The arithmetic producing `ValuePre` is branch-specific (`x_hat * W + B`);
this theorem exposes the Python-observed Y store for that branch using the
shared row-vector writeback proof. -/
theorem layer_norm_ops_fwd_bias_y_store_slice_compute_correct
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i => fwdYStoreSpec s ValuePre stride_y_row i) := by
  exact layer_norm_ops_fwd_y_store_slice_compute_correct ValuePre Y
    stride_y_row N BLOCK_N s hOutInj

/-- Named forward Y writeback for the residual-without-bias branch of
`_layer_norm_fwd_1pass_kernel`. The residual-combined normalization arithmetic
is represented by `ValuePre`; this theorem exposes the final masked Y store. -/
theorem layer_norm_ops_fwd_residual_y_store_slice_compute_correct
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i => fwdYStoreSpec s ValuePre stride_y_row i) := by
  exact layer_norm_ops_fwd_y_store_slice_compute_correct ValuePre Y
    stride_y_row N BLOCK_N s hOutInj

/-- Named forward Y writeback for the RMS-norm plus bias branch tested by
`layer_norm_ops.py`. RMS-specific arithmetic is represented by `ValuePre`; the
store address and mask are the Python row-vector writeback. -/
theorem layer_norm_ops_fwd_rms_bias_y_store_slice_compute_correct
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i => fwdYStoreSpec s ValuePre stride_y_row i) := by
  exact layer_norm_ops_fwd_y_store_slice_compute_correct ValuePre Y
    stride_y_row N BLOCK_N s hOutInj

/-- Named forward Y writeback for the residual-plus-bias branch of
`_layer_norm_fwd_1pass_kernel`.

The residual add and bias add are represented by the supplied `ValuePre` tile;
the theorem proves the final masked Y writeback under the Python row layout. -/
theorem layer_norm_ops_fwd_residual_bias_y_store_slice_compute_correct
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i => fwdYStoreSpec s ValuePre stride_y_row i) := by
  exact layer_norm_ops_fwd_y_store_slice_compute_correct ValuePre Y
    stride_y_row N BLOCK_N s hOutInj

/-- Backward partial-gradient writeback slice for `DW` and `DB`. The full Python
kernel stores each program's partial gradient at
`Out + row_block_id * N + cols`; reduction across programs happens later in
Python. -/
def layer_norm_ops_bwd_param_grad_store_slice
    (GradPre Out : RegionName) (N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row_block_id = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  grad = tl.load(GradPre + row_block_id * $(N) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row_block_id * $(N) + cols, grad, mask=mask)
}

def bwdParamGradOffset (s : BlockState) (N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * N + i.val

noncomputable def bwdParamGradStoreSpec
    (s : BlockState) (GradPre : RegionName) (N : Nat) (i : Fin BLOCK_N) : ℝ :=
  s.readMem GradPre (bwdParamGradOffset s N i)

theorem layer_norm_ops_bwd_param_grad_store_slice_correct
    (GradPre Out : RegionName) (N BLOCK_N : Nat) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i))
    (hExec : exec (layer_norm_ops_bwd_param_grad_store_slice GradPre Out
        N BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Out (bwdParamGradOffset s N i) =
        if i.val < N then
          bwdParamGradStoreSpec s GradPre N i
        else
          s.readMem Out (bwdParamGradOffset s N i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * N + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [bwdParamGradOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [exec, layer_norm_ops_bwd_param_grad_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        bwdParamGradOffset] at hExec
  subst s'
  simp only [bwdParamGradOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
        (i, PUnit.unit)]
  by_cases hi : i.val < N
  · simp [hi, bwdParamGradStoreSpec, bwdParamGradOffset]
  · simp [hi]

theorem layer_norm_ops_bwd_param_grad_store_slice_compute_correct
    (GradPre Out : RegionName) (N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_param_grad_store_slice GradPre Out N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Out, bwdParamGradOffset s N i)))
      (expected := fun i => bwdParamGradStoreSpec s GradPre N i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_bwd_param_grad_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_ops_bwd_param_grad_store_slice_correct GradPre Out
    N BLOCK_N s s' hOutInj hExec i
  simpa [hActive] using h

/-- Explicit partial `DW` writeback slice. -/
abbrev layer_norm_ops_bwd_dw_store_slice
    (DWPre DW : RegionName) (N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_param_grad_store_slice DWPre DW N BLOCK_N

theorem layer_norm_ops_bwd_dw_store_slice_correct
    (DWPre DW : RegionName) (N BLOCK_N : Nat) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i))
    (hExec : exec (layer_norm_ops_bwd_dw_store_slice DWPre DW N BLOCK_N) s =
      some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem DW (bwdParamGradOffset s N i) =
        if i.val < N then
          bwdParamGradStoreSpec s DWPre N i
        else
          s.readMem DW (bwdParamGradOffset s N i) :=
  layer_norm_ops_bwd_param_grad_store_slice_correct DWPre DW
    N BLOCK_N s s' hOutInj hExec

theorem layer_norm_ops_bwd_dw_store_slice_compute_correct
    (DWPre DW : RegionName) (N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_dw_store_slice DWPre DW N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DW, bwdParamGradOffset s N i)))
      (expected := fun i => bwdParamGradStoreSpec s DWPre N i) :=
  layer_norm_ops_bwd_param_grad_store_slice_compute_correct DWPre DW
    N BLOCK_N s hOutInj

/-- Explicit optional partial `DB` writeback slice for `HAS_BIAS`. -/
abbrev layer_norm_ops_bwd_db_store_slice
    (DBPre DB : RegionName) (N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_param_grad_store_slice DBPre DB N BLOCK_N

theorem layer_norm_ops_bwd_db_store_slice_correct
    (DBPre DB : RegionName) (N BLOCK_N : Nat) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i))
    (hExec : exec (layer_norm_ops_bwd_db_store_slice DBPre DB N BLOCK_N) s =
      some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem DB (bwdParamGradOffset s N i) =
        if i.val < N then
          bwdParamGradStoreSpec s DBPre N i
        else
          s.readMem DB (bwdParamGradOffset s N i) :=
  layer_norm_ops_bwd_param_grad_store_slice_correct DBPre DB
    N BLOCK_N s s' hOutInj hExec

theorem layer_norm_ops_bwd_db_store_slice_compute_correct
    (DBPre DB : RegionName) (N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_db_store_slice DBPre DB N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DB, bwdParamGradOffset s N i)))
      (expected := fun i => bwdParamGradStoreSpec s DBPre N i) :=
  layer_norm_ops_bwd_param_grad_store_slice_compute_correct DBPre DB
    N BLOCK_N s hOutInj

/-! ## Backward RMS arithmetic slice

This slice removes one of the narrow-writeback gaps above: it computes the
tested RMS backward arithmetic for one row, not merely a precomputed `DX`/`DW`
copy. It specializes the full backward kernel to the no-bias/no-residual/
no-recompute RMS branch and one row per program. -/

def layer_norm_ops_bwd_rms_one_row
    (X W DY DX DW Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  x = tl.load(X + row * $(stride_x_row) + cols, mask=mask, other=0.0).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  xhat = x * rstd
  xhat = tl.where(mask, xhat, 0.0)
  wdy = w * dy
  dw = dy * xhat
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  dx = (wdy - xhat * c1) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
  tl.store(DW + row * $(N) + cols, dw, mask=mask)
}

def bwdRmsXOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val

def bwdRmsDYOffset (s : BlockState) (stride_dy_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dy_row + i.val

def bwdRmsDXOffset (s : BlockState) (stride_dx_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dx_row + i.val

def bwdRmsDWOffset (s : BlockState) (N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * N + i.val

noncomputable def bwdRmsXTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (bwdRmsXOffset s stride_x_row idx.1))
      else some (0.0 : ℝ) }

noncomputable def bwdRmsDYTile
    (s : BlockState) (DY : RegionName) (stride_dy_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem DY (bwdRmsDYOffset s stride_dy_row idx.1))
      else some (0.0 : ℝ) }

noncomputable def bwdRmsWTile
    (s : BlockState) (W : RegionName) (N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem W idx.1.val)
      else some (s.undef W idx.1.val) }

noncomputable def bwdRmsXhatTile
    (s : BlockState) (X Rstd : RegionName)
    (stride_x_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        match (bwdRmsXTile s X stride_x_row N BLOCK_N).data idx with
        | some x => some (x * s.readMem Rstd s.pid)
        | none => none
      else some (0.0 : ℝ) }

noncomputable def bwdRmsWdyTile
    (s : BlockState) (W DY : RegionName)
    (stride_dy_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
    (bwdRmsWTile s W N BLOCK_N)
    (bwdRmsDYTile s DY stride_dy_row N BLOCK_N)

noncomputable def bwdRmsDWSpec
    (s : BlockState) (X DY Rstd : RegionName)
    (stride_x_row stride_dy_row N BLOCK_N : Nat) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun dy xhat => dy * xhat)
      ((bwdRmsDYTile s DY stride_dy_row N BLOCK_N).data (i, PUnit.unit))
      ((bwdRmsXhatTile s X Rstd stride_x_row N BLOCK_N).data (i, PUnit.unit)))

theorem layer_norm_ops_bwd_rms_one_row_dw_correct
    (X W DY DX DW Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hDWDX : DW ≠ DX)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDWOffset s N i))
    (hExec : exec (layer_norm_ops_bwd_rms_one_row X W DY DX DW Rstd
        stride_x_row stride_dy_row stride_dx_row N BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem DW (bwdRmsDWOffset s N i) =
        if i.val < N then
          bwdRmsDWSpec s X DY Rstd stride_x_row stride_dy_row N BLOCK_N i
        else
          s.readMem DW (bwdRmsDWOffset s N i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * N + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [bwdRmsDWOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_N
  · simp [exec, layer_norm_ops_bwd_rms_one_row, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.sub, NumericDType.mul, NumericDType.div,
          ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
          FloatDType.toWithBot, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [bwdRmsDWOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
          (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, bwdRmsDWSpec, bwdRmsXhatTile, bwdRmsXTile, bwdRmsDYTile,
            bwdRmsXOffset, bwdRmsDYOffset, Tile.select, NumericDType.mul]
    · rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
          DX _ _ _ _ _ _ _ hDWDX]
      simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem layer_norm_ops_bwd_rms_one_row_dw_compute_correct
    (X W DY DX DW Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s : BlockState)
    (hDWDX : DW ≠ DX)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDWOffset s N i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_rms_one_row X W DY DX DW Rstd
        stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DW, bwdRmsDWOffset s N i)))
      (expected := fun i =>
        bwdRmsDWSpec s X DY Rstd stride_x_row stride_dy_row N BLOCK_N i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_bwd_rms_one_row, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_ops_bwd_rms_one_row_dw_correct X W DY DX DW Rstd
    stride_x_row stride_dy_row stride_dx_row N BLOCK_N s s' hDWDX hOutInj hExec i
  simpa [hActive] using h

/-! ## Backward bias-gradient arithmetic slice

For the `HAS_BIAS` branch of the backward kernel, each row contributes `dy` to
the partial `DB` accumulator before the per-program partial-gradient store.
This slice proves that arithmetic path directly for one row. -/

def layer_norm_ops_bwd_bias_db_one_row
    (DY DB : RegionName) (stride_dy_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  db = dy
  tl.store(DB + row * $(N) + cols, db, mask=mask)
}

noncomputable def bwdBiasDBSpec
    (s : BlockState) (DY : RegionName) (stride_dy_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem DY (bwdRmsDYOffset s stride_dy_row i)

theorem layer_norm_ops_bwd_bias_db_one_row_correct
    (DY DB : RegionName) (stride_dy_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i))
    (hExec : exec (layer_norm_ops_bwd_bias_db_one_row DY DB
        stride_dy_row N BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem DB (bwdParamGradOffset s N i) =
        if i.val < N then
          bwdBiasDBSpec s DY stride_dy_row i
        else
          s.readMem DB (bwdParamGradOffset s N i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * N + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [bwdParamGradOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [exec, layer_norm_ops_bwd_bias_db_one_row, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, bwdParamGradOffset] at hExec
  subst s'
  simp only [bwdParamGradOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
        (i, PUnit.unit)]
  by_cases hi : i.val < N
  · simp [hi, bwdBiasDBSpec, bwdRmsDYOffset]
  · simp [hi]

theorem layer_norm_ops_bwd_bias_db_one_row_compute_correct
    (DY DB : RegionName) (stride_dy_row N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_bias_db_one_row DY DB
        stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DB, bwdParamGradOffset s N i)))
      (expected := fun i => bwdBiasDBSpec s DY stride_dy_row i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_bwd_bias_db_one_row, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_ops_bwd_bias_db_one_row_correct DY DB stride_dy_row
    N BLOCK_N s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## Backward non-RMS+bias one-row integrated slice

This is the first non-RMS integrated backward row slice for #133. Unlike the smaller
`DX`/`DW`/`DB` slices above, this kernel keeps the Python row arithmetic together:
`xhat`, `wdy`, partial `dw`, partial `db`, `c1`, `c2`, `dx`, then the three
observable stores `DX`, `DW`, and `DB`. The theorems here prove the integrated
`DW` and `DB` stores; `DX` still needs a reduction-heavy readback theorem. -/

def layer_norm_ops_bwd_plain_bias_one_row
    (X W DY DX DW DB Mean Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  x = tl.load(X + row * $(stride_x_row) + cols, mask=mask, other=0.0).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  mean = tl.load(Mean + row)
  rstd = tl.load(Rstd + row)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  xhat = (x - mean) * rstd
  xhat = tl.where(mask, xhat, 0.0)
  wdy = w * dy
  dw = dy * xhat
  db = dy
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  c2 = tl.sum(wdy, axis=0) / $(N)
  dx = (wdy - (xhat * c1 + c2)) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
  tl.store(DW + row * $(N) + cols, dw, mask=mask)
  tl.store(DB + row * $(N) + cols, db, mask=mask)
}

noncomputable def bwdPlainXhatTile
    (s : BlockState) (X Mean Rstd : RegionName)
    (stride_x_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        match
          match
            if idx.1.val < N then
              some (s.readMem X (bwdRmsXOffset s stride_x_row idx.1))
            else some (0.0 : ℝ)
          with
          | some x => some (x - s.readMem Mean s.pid)
          | none => none
        with
        | some x => some (x * s.readMem Rstd s.pid)
        | none => none
      else some (0.0 : ℝ) }

noncomputable def bwdPlainBiasDWSpec
    (s : BlockState) (X DY Mean Rstd : RegionName)
    (stride_x_row stride_dy_row N BLOCK_N : Nat) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun dy xhat => dy * xhat)
      ((bwdRmsDYTile s DY stride_dy_row N BLOCK_N).data (i, PUnit.unit))
      ((bwdPlainXhatTile s X Mean Rstd stride_x_row N BLOCK_N).data
        (i, PUnit.unit)))

theorem layer_norm_ops_bwd_plain_bias_one_row_dw_correct
    (X W DY DX DW DB Mean Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hDWDX : DW ≠ DX)
    (hDWDB : DW ≠ DB)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i))
    (hExec : exec (layer_norm_ops_bwd_plain_bias_one_row X W DY DX DW DB
        Mean Rstd stride_x_row stride_dy_row stride_dx_row N BLOCK_N) s =
        some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem DW (bwdParamGradOffset s N i) =
        if i.val < N then
          bwdPlainBiasDWSpec s X DY Mean Rstd stride_x_row stride_dy_row
            N BLOCK_N i
        else
          s.readMem DW (bwdParamGradOffset s N i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * N + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [bwdParamGradOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_N
  · simp [exec, layer_norm_ops_bwd_plain_bias_one_row, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
          Tile.uop, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.sub, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
          ComputeOp.toAlgorithm?] at hExec
    subst s'
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
          DB _ _ _ _ _ _ _ hDWDB]
    simp only [bwdParamGradOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
          (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, bwdPlainBiasDWSpec, bwdPlainXhatTile, bwdRmsXOffset,
            bwdRmsDYTile, bwdRmsDYOffset, Tile.select, NumericDType.mul,
            NumericDType.sub]
    · rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
          DX _ _ _ _ _ _ _ hDWDX]
      simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem layer_norm_ops_bwd_plain_bias_one_row_db_correct
    (X W DY DX DW DB Mean Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hDBDX : DB ≠ DX)
    (hDBDW : DB ≠ DW)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i))
    (hExec : exec (layer_norm_ops_bwd_plain_bias_one_row X W DY DX DW DB
        Mean Rstd stride_x_row stride_dy_row stride_dx_row N BLOCK_N) s =
        some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem DB (bwdParamGradOffset s N i) =
        if i.val < N then
          bwdBiasDBSpec s DY stride_dy_row i
        else
          s.readMem DB (bwdParamGradOffset s N i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * N + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [bwdParamGradOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [exec, layer_norm_ops_bwd_plain_bias_one_row, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        Tile.uop, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.sub, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        bwdParamGradOffset] at hExec
  subst s'
  simp only [bwdParamGradOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
        (i, PUnit.unit)]
  by_cases hi : i.val < N
  · simp [hi, bwdBiasDBSpec, bwdRmsDYOffset]
  · rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        DW _ _ _ _ _ _ _ hDBDW]
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        DX _ _ _ _ _ _ _ hDBDX]
    simp [hi]

theorem layer_norm_ops_bwd_plain_bias_one_row_dw_compute_correct
    (X W DY DX DW DB Mean Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s : BlockState)
    (hDWDX : DW ≠ DX)
    (hDWDB : DW ≠ DB)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_plain_bias_one_row X W DY DX DW DB
        Mean Rstd stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DW, bwdParamGradOffset s N i)))
      (expected := fun i =>
        bwdPlainBiasDWSpec s X DY Mean Rstd stride_x_row stride_dy_row
          N BLOCK_N i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_bwd_plain_bias_one_row, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_ops_bwd_plain_bias_one_row_dw_correct X W DY DX DW DB
    Mean Rstd stride_x_row stride_dy_row stride_dx_row N BLOCK_N s s'
    hDWDX hDWDB hOutInj hExec i
  simpa [hActive] using h

theorem layer_norm_ops_bwd_plain_bias_one_row_db_compute_correct
    (X W DY DX DW DB Mean Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s : BlockState)
    (hDBDX : DB ≠ DX)
    (hDBDW : DB ≠ DW)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_plain_bias_one_row X W DY DX DW DB
        Mean Rstd stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DB, bwdParamGradOffset s N i)))
      (expected := fun i => bwdBiasDBSpec s DY stride_dy_row i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_bwd_plain_bias_one_row, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_ops_bwd_plain_bias_one_row_db_correct X W DY DX DW DB
    Mean Rstd stride_x_row stride_dy_row stride_dx_row N BLOCK_N s s'
    hDBDX hDBDW hOutInj hExec i
  simpa [hActive] using h

/-! ## Backward residual-add arithmetic slice

When `HAS_DRESIDUAL` is true, the Python backward kernel adds the residual
gradient tile into the already-computed `dx`, optionally stores the combined
value to `DRESIDUAL_IN`, and always stores it to `DX`. This slice proves that
arithmetic/writeback path directly. -/

def layer_norm_ops_bwd_residual_add_store_slice
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  dx = tl.load(DXBase + row * $(stride_dx_row) + cols, mask=mask, other=0.0)
  dres = tl.load(DRESIDUAL + row * $(stride_dres_row) + cols, mask=mask, other=0.0)
  out = dx + dres
  tl.store(DRESIDUAL_IN + row * $(stride_dres_in_row) + cols, out, mask=mask)
  tl.store(DX + row * $(stride_dx_row) + cols, out, mask=mask)
}

def bwdDResidualOffset (s : BlockState) (stride_dres_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dres_row + i.val

def bwdDResidualInOffset (s : BlockState) (stride_dres_in_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dres_in_row + i.val

noncomputable def bwdResidualAddSpec
    (s : BlockState) (DXBase DRESIDUAL : RegionName)
    (stride_dx_row stride_dres_row : Nat) (i : Fin BLOCK_N) : ℝ :=
  s.readMem DXBase (bwdRmsDXOffset s stride_dx_row i) +
    s.readMem DRESIDUAL (bwdDResidualOffset s stride_dres_row i)

theorem layer_norm_ops_bwd_residual_add_store_slice_dx_correct
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hDXDresIn : DX ≠ DRESIDUAL_IN)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDXOffset s stride_dx_row i))
    (hExec : exec (layer_norm_ops_bwd_residual_add_store_slice DXBase DRESIDUAL
        DX DRESIDUAL_IN stride_dx_row stride_dres_row stride_dres_in_row
        N BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem DX (bwdRmsDXOffset s stride_dx_row i) =
        if i.val < N then
          bwdResidualAddSpec s DXBase DRESIDUAL stride_dx_row stride_dres_row i
        else
          s.readMem DX (bwdRmsDXOffset s stride_dx_row i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * stride_dx_row + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [bwdRmsDXOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [exec, layer_norm_ops_bwd_residual_add_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, bwdRmsDXOffset] at hExec
  subst s'
  simp only [bwdRmsDXOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
        (i, PUnit.unit)]
  by_cases hi : i.val < N
  · simp [hi, bwdResidualAddSpec, bwdRmsDXOffset, bwdDResidualOffset]
  · rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        DRESIDUAL_IN _ _ _ _ _ _ _ hDXDresIn]
    simp [hi]

theorem layer_norm_ops_bwd_residual_add_store_slice_dresidual_in_correct
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hDresInDX : DRESIDUAL_IN ≠ DX)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdDResidualInOffset s stride_dres_in_row i))
    (hExec : exec (layer_norm_ops_bwd_residual_add_store_slice DXBase DRESIDUAL
        DX DRESIDUAL_IN stride_dx_row stride_dres_row stride_dres_in_row
        N BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem DRESIDUAL_IN (bwdDResidualInOffset s stride_dres_in_row i) =
        if i.val < N then
          bwdResidualAddSpec s DXBase DRESIDUAL stride_dx_row stride_dres_row i
        else
          s.readMem DRESIDUAL_IN (bwdDResidualInOffset s stride_dres_in_row i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 0 * stride_dres_in_row + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [bwdDResidualInOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [exec, layer_norm_ops_bwd_residual_add_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, bwdDResidualInOffset] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        DX _ _ _ _ _ _ _ hDresInDX]
  simp only [bwdDResidualInOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
        (i, PUnit.unit)]
  by_cases hi : i.val < N
  · simp [hi, bwdResidualAddSpec, bwdRmsDXOffset, bwdDResidualOffset]
  · simp [hi]

theorem layer_norm_ops_bwd_residual_add_store_slice_dx_compute_correct
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat)
    (s : BlockState)
    (hDXDresIn : DX ≠ DRESIDUAL_IN)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDXOffset s stride_dx_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_residual_add_store_slice DXBase DRESIDUAL
        DX DRESIDUAL_IN stride_dx_row stride_dres_row stride_dres_in_row
        N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i =>
        bwdResidualAddSpec s DXBase DRESIDUAL stride_dx_row stride_dres_row i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_bwd_residual_add_store_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_ops_bwd_residual_add_store_slice_dx_correct
    DXBase DRESIDUAL DX DRESIDUAL_IN stride_dx_row stride_dres_row
    stride_dres_in_row N BLOCK_N s s' hDXDresIn hOutInj hExec i
  simpa [hActive] using h

theorem layer_norm_ops_bwd_residual_add_store_slice_dresidual_in_compute_correct
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat)
    (s : BlockState)
    (hDresInDX : DRESIDUAL_IN ≠ DX)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdDResidualInOffset s stride_dres_in_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_residual_add_store_slice DXBase DRESIDUAL
        DX DRESIDUAL_IN stride_dx_row stride_dres_row stride_dres_in_row
        N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DRESIDUAL_IN, bwdDResidualInOffset s stride_dres_in_row i)))
      (expected := fun i =>
        bwdResidualAddSpec s DXBase DRESIDUAL stride_dx_row stride_dres_row i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_bwd_residual_add_store_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_ops_bwd_residual_add_store_slice_dresidual_in_correct
    DXBase DRESIDUAL DX DRESIDUAL_IN stride_dx_row stride_dres_row
    stride_dres_in_row N BLOCK_N s s' hDresInDX hOutInj hExec i
  simpa [hActive] using h

/-! ## Backward recomputed-output arithmetic slice

When `RECOMPUTE_OUTPUT` and `HAS_BIAS` are true, the backward kernel recomputes
`Y` as `xhat * W + B` before continuing with gradient arithmetic. This slice
starts after `xhat` has been produced and proves that branch's masked `Y`
writeback with the Python row layout. -/

def layer_norm_ops_bwd_recompute_y_bias_slice
    (Xhat W B Y : RegionName)
    (stride_xhat_row stride_y_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  y = xhat * w + b
  tl.store(Y + row * $(stride_y_row) + cols, y, mask=mask)
}

def bwdRecomputeXhatOffset (s : BlockState) (stride_xhat_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_xhat_row + i.val

noncomputable def bwdRecomputeYBiasSpec
    (s : BlockState) (Xhat W B : RegionName) (stride_xhat_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem Xhat (bwdRecomputeXhatOffset s stride_xhat_row i) *
    s.readMem W i.val + s.readMem B i.val

theorem layer_norm_ops_bwd_recompute_y_bias_slice_correct
    (Xhat W B Y : RegionName)
    (stride_xhat_row stride_y_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_y_row i))
    (hExec : exec (layer_norm_ops_bwd_recompute_y_bias_slice Xhat W B Y
        stride_xhat_row stride_y_row N BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (bwdRowVectorOffset s stride_y_row i) =
        if i.val < N then
          bwdRecomputeYBiasSpec s Xhat W B stride_xhat_row i
        else
          s.readMem Y (bwdRowVectorOffset s stride_y_row i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * stride_y_row + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      change bwdRowVectorOffset s stride_y_row a.1 =
        bwdRowVectorOffset s stride_y_row b.1
      simpa [bwdRowVectorOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [exec, layer_norm_ops_bwd_recompute_y_bias_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, bwdRowVectorOffset] at hExec
  subst s'
  simp only [bwdRowVectorOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
        (i, PUnit.unit)]
  by_cases hi : i.val < N
  · simp [hi, bwdRecomputeYBiasSpec, bwdRecomputeXhatOffset]
  · simp [hi]

theorem layer_norm_ops_bwd_recompute_y_bias_slice_compute_correct
    (Xhat W B Y : RegionName)
    (stride_xhat_row stride_y_row N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_recompute_y_bias_slice Xhat W B Y
        stride_xhat_row stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, bwdRowVectorOffset s stride_y_row i)))
      (expected := fun i =>
        bwdRecomputeYBiasSpec s Xhat W B stride_xhat_row i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_bwd_recompute_y_bias_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_ops_bwd_recompute_y_bias_slice_correct Xhat W B Y
    stride_xhat_row stride_y_row N BLOCK_N s s' hOutInj hExec i
  simpa [hActive] using h

/-- No-bias recomputed-output branch: `y = xhat * W`. -/
def layer_norm_ops_bwd_recompute_y_no_bias_slice
    (Xhat W Y : RegionName)
    (stride_xhat_row stride_y_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  y = xhat * w
  tl.store(Y + row * $(stride_y_row) + cols, y, mask=mask)
}

noncomputable def bwdRecomputeYNoBiasSpec
    (s : BlockState) (Xhat W : RegionName) (stride_xhat_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem Xhat (bwdRecomputeXhatOffset s stride_xhat_row i) *
    s.readMem W i.val

theorem layer_norm_ops_bwd_recompute_y_no_bias_slice_correct
    (Xhat W Y : RegionName)
    (stride_xhat_row stride_y_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_y_row i))
    (hExec : exec (layer_norm_ops_bwd_recompute_y_no_bias_slice Xhat W Y
        stride_xhat_row stride_y_row N BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (bwdRowVectorOffset s stride_y_row i) =
        if i.val < N then
          bwdRecomputeYNoBiasSpec s Xhat W stride_xhat_row i
        else
          s.readMem Y (bwdRowVectorOffset s stride_y_row i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * stride_y_row + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      change bwdRowVectorOffset s stride_y_row a.1 =
        bwdRowVectorOffset s stride_y_row b.1
      simpa [bwdRowVectorOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [exec, layer_norm_ops_bwd_recompute_y_no_bias_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, bwdRowVectorOffset] at hExec
  subst s'
  simp only [bwdRowVectorOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
        (i, PUnit.unit)]
  by_cases hi : i.val < N
  · simp [hi, bwdRecomputeYNoBiasSpec, bwdRecomputeXhatOffset]
  · simp [hi]

theorem layer_norm_ops_bwd_recompute_y_no_bias_slice_compute_correct
    (Xhat W Y : RegionName)
    (stride_xhat_row stride_y_row N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_recompute_y_no_bias_slice Xhat W Y
        stride_xhat_row stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, bwdRowVectorOffset s stride_y_row i)))
      (expected := fun i =>
        bwdRecomputeYNoBiasSpec s Xhat W stride_xhat_row i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_bwd_recompute_y_no_bias_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_ops_bwd_recompute_y_no_bias_slice_correct Xhat W Y
    stride_xhat_row stride_y_row N BLOCK_N s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## Backward `c1`/`c2` reduction-production slices

The full backward kernel computes `c1 = sum(xhat * (W * DY)) / N` in both RMS
and non-RMS branches, and additionally computes `c2 = sum(W * DY) / N` in the
non-RMS branch. These slices isolate the two row-local reductions and their
scalar stores so the reduction production itself is checked independently of
the surrounding branch/control-flow proof. -/

def layer_norm_ops_bwd_c1_reduction_slice
    (Xhat W DY C1 : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  wdy = w * dy
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  tl.store(C1 + row, c1)
}

def layer_norm_ops_bwd_c2_reduction_slice
    (W DY C2 : RegionName)
    (stride_dy_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  wdy = w * dy
  c2 = tl.sum(wdy, axis=0) / $(N)
  tl.store(C2 + row, c2)
}

noncomputable def bwdC1ReductionSpec
    (s : BlockState) (Xhat W DY : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0
    (Option.map (fun a => a / (N : ℝ))
      ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
        (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
          { data := fun idx =>
              if idx.1.val < N then
                some (s.readMem Xhat
                  (bwdRecomputeXhatOffset s stride_xhat_row idx.1))
              else some (0.0 : ℝ) }
          (bwdRmsWdyTile s W DY stride_dy_row N BLOCK_N))).data PUnit.unit))

noncomputable def bwdC2ReductionSpec
    (s : BlockState) (W DY : RegionName)
    (stride_dy_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0
    (Option.map (fun a => a / (N : ℝ))
      ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
        (bwdRmsWdyTile s W DY stride_dy_row N BLOCK_N)).data PUnit.unit))

theorem layer_norm_ops_bwd_c1_reduction_slice_correct
    (Xhat W DY C1 : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hExec : exec (layer_norm_ops_bwd_c1_reduction_slice Xhat W DY C1
        stride_xhat_row stride_dy_row N BLOCK_N) s = some s') :
    s'.readMem C1 s.pid =
      bwdC1ReductionSpec s Xhat W DY stride_xhat_row stride_dy_row
        N BLOCK_N := by
  simp [exec, layer_norm_ops_bwd_c1_reduction_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  subst s'
  simp [BlockState.writeMem_readMem, bwdC1ReductionSpec, bwdRmsWdyTile,
        bwdRmsWTile, bwdRmsDYTile, bwdRecomputeXhatOffset, bwdRmsDYOffset,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        Option.map]
  congr

theorem layer_norm_ops_bwd_c1_reduction_slice_compute_correct
    (Xhat W DY C1 : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_c1_reduction_slice Xhat W DY C1
        stride_xhat_row stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := fun _ : PUnit => some (C1, s.pid))
      (expected := fun _ =>
        bwdC1ReductionSpec s Xhat W DY stride_xhat_row stride_dy_row
          N BLOCK_N) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_bwd_c1_reduction_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_ops_bwd_c1_reduction_slice_correct Xhat W DY C1
    stride_xhat_row stride_dy_row N BLOCK_N s s' hExec

theorem layer_norm_ops_bwd_c2_reduction_slice_correct
    (W DY C2 : RegionName)
    (stride_dy_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hExec : exec (layer_norm_ops_bwd_c2_reduction_slice W DY C2
        stride_dy_row N BLOCK_N) s = some s') :
    s'.readMem C2 s.pid =
      bwdC2ReductionSpec s W DY stride_dy_row N BLOCK_N := by
  simp [exec, layer_norm_ops_bwd_c2_reduction_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  subst s'
  simp [BlockState.writeMem_readMem, bwdC2ReductionSpec, bwdRmsWdyTile,
        bwdRmsWTile, bwdRmsDYTile, bwdRmsDYOffset, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, Option.map]
  congr

theorem layer_norm_ops_bwd_c2_reduction_slice_compute_correct
    (W DY C2 : RegionName)
    (stride_dy_row N BLOCK_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_c2_reduction_slice W DY C2
        stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := fun _ : PUnit => some (C2, s.pid))
      (expected := fun _ =>
        bwdC2ReductionSpec s W DY stride_dy_row N BLOCK_N) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_bwd_c2_reduction_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_ops_bwd_c2_reduction_slice_correct W DY C2
    stride_dy_row N BLOCK_N s s' hExec

/-! ## Backward base-DX formula slice

After the RMS branch has produced `xhat` and reduced `c1`, Python computes
`dx = (w * dy - xhat * c1) * rstd`. This slice proves that formula and the
masked `DX` store directly, leaving only the production of `c1` itself outside
the theorem. -/

def layer_norm_ops_bwd_rms_dx_from_c1_slice
    (Xhat W DY Rstd C1 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row)
  c1 = tl.load(C1 + row)
  wdy = w * dy
  dx = (wdy - xhat * c1) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
}

noncomputable def bwdRmsDXFromC1Spec
    (s : BlockState) (Xhat W DY Rstd C1 : RegionName)
    (stride_xhat_row stride_dy_row : Nat) (i : Fin BLOCK_N) : ℝ :=
  (s.readMem W i.val *
      s.readMem DY (bwdRmsDYOffset s stride_dy_row i) -
    s.readMem Xhat (bwdRecomputeXhatOffset s stride_xhat_row i) *
      s.readMem C1 s.pid) *
    s.readMem Rstd s.pid

theorem layer_norm_ops_bwd_rms_dx_from_c1_slice_correct
    (Xhat W DY Rstd C1 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDXOffset s stride_dx_row i))
    (hExec : exec (layer_norm_ops_bwd_rms_dx_from_c1_slice Xhat W DY Rstd
        C1 DX stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N) s =
        some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem DX (bwdRmsDXOffset s stride_dx_row i) =
        if i.val < N then
          bwdRmsDXFromC1Spec s Xhat W DY Rstd C1 stride_xhat_row stride_dy_row i
        else
          s.readMem DX (bwdRmsDXOffset s stride_dx_row i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * stride_dx_row + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [bwdRmsDXOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [exec, layer_norm_ops_bwd_rms_dx_from_c1_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.sub, NumericDType.mul,
        ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        bwdRmsDXOffset] at hExec
  subst s'
  simp only [bwdRmsDXOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
        (i, PUnit.unit)]
  by_cases hi : i.val < N
  · simp [hi, bwdRmsDXFromC1Spec, bwdRmsDYOffset, bwdRecomputeXhatOffset,
      NumericDType.mul, NumericDType.sub]
  · simp [hi]

theorem layer_norm_ops_bwd_rms_dx_from_c1_slice_compute_correct
    (Xhat W DY Rstd C1 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDXOffset s stride_dx_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_rms_dx_from_c1_slice Xhat W DY Rstd
        C1 DX stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i =>
        bwdRmsDXFromC1Spec s Xhat W DY Rstd C1 stride_xhat_row stride_dy_row i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_bwd_rms_dx_from_c1_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_ops_bwd_rms_dx_from_c1_slice_correct Xhat W DY Rstd
    C1 DX stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N s s'
    hOutInj hExec i
  simpa [hActive] using h

/-- Non-RMS base-DX formula after `c1` and `c2` have been reduced:
`dx = (wdy - (xhat * c1 + c2)) * rstd`. -/
def layer_norm_ops_bwd_plain_dx_from_c1_c2_slice
    (Xhat W DY Rstd C1 C2 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row)
  c1 = tl.load(C1 + row)
  c2 = tl.load(C2 + row)
  wdy = w * dy
  dx = (wdy - (xhat * c1 + c2)) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
}

noncomputable def bwdPlainDXFromC1C2Spec
    (s : BlockState) (Xhat W DY Rstd C1 C2 : RegionName)
    (stride_xhat_row stride_dy_row : Nat) (i : Fin BLOCK_N) : ℝ :=
  (s.readMem W i.val *
      s.readMem DY (bwdRmsDYOffset s stride_dy_row i) -
    (s.readMem Xhat (bwdRecomputeXhatOffset s stride_xhat_row i) *
        s.readMem C1 s.pid +
      s.readMem C2 s.pid)) *
    s.readMem Rstd s.pid

theorem layer_norm_ops_bwd_plain_dx_from_c1_c2_slice_correct
    (Xhat W DY Rstd C1 C2 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDXOffset s stride_dx_row i))
    (hExec : exec (layer_norm_ops_bwd_plain_dx_from_c1_c2_slice Xhat W DY
        Rstd C1 C2 DX stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N) s =
        some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem DX (bwdRmsDXOffset s stride_dx_row i) =
        if i.val < N then
          bwdPlainDXFromC1C2Spec s Xhat W DY Rstd C1 C2
            stride_xhat_row stride_dy_row i
        else
          s.readMem DX (bwdRmsDXOffset s stride_dx_row i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * stride_dx_row + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [bwdRmsDXOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [exec, layer_norm_ops_bwd_plain_dx_from_c1_c2_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.sub, NumericDType.mul,
        ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        bwdRmsDXOffset] at hExec
  subst s'
  simp only [bwdRmsDXOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
        (i, PUnit.unit)]
  by_cases hi : i.val < N
  · simp [hi, bwdPlainDXFromC1C2Spec, bwdRmsDYOffset, bwdRecomputeXhatOffset,
      NumericDType.add, NumericDType.mul, NumericDType.sub]
  · simp [hi]

theorem layer_norm_ops_bwd_plain_dx_from_c1_c2_slice_compute_correct
    (Xhat W DY Rstd C1 C2 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDXOffset s stride_dx_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_plain_dx_from_c1_c2_slice Xhat W DY
        Rstd C1 C2 DX stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i =>
        bwdPlainDXFromC1C2Spec s Xhat W DY Rstd C1 C2
          stride_xhat_row stride_dy_row i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_ops_bwd_plain_dx_from_c1_c2_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_ops_bwd_plain_dx_from_c1_c2_slice_correct Xhat W DY
    Rstd C1 C2 DX stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N s s'
    hOutInj hExec i
  simpa [hActive] using h

theorem layer_norm_ops_rms_dx_offset_general_injective
    (s : BlockState) (stride BLOCK_N : Nat) :
    Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDXOffset s stride i) := by
  intro a b h
  simp only [bwdRmsDXOffset] at h
  exact Fin.ext (by omega)

theorem layer_norm_ops_dresidual_in_offset_general_injective
    (s : BlockState) (stride BLOCK_N : Nat) :
    Function.Injective
      (fun i : Fin BLOCK_N => bwdDResidualInOffset s stride i) := by
  intro a b h
  simp only [bwdDResidualInOffset] at h
  exact Fin.ext (by omega)

theorem layer_norm_ops_param_grad_offset_general_injective
    (s : BlockState) (N BLOCK_N : Nat) :
    Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i) := by
  intro a b h
  simp only [bwdParamGradOffset] at h
  exact Fin.ext (by omega)

theorem layer_norm_ops_rms_dw_offset_general_injective
    (s : BlockState) (N BLOCK_N : Nat) :
    Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDWOffset s N i) := by
  intro a b h
  simp only [bwdRmsDWOffset] at h
  exact Fin.ext (by omega)

theorem layer_norm_ops_fwd_y_offset_general_injective
    (s : BlockState) (stride_y_row BLOCK_N : Nat) :
    Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i) := by
  intro a b h
  simp only [fwdYOffset, bwdRowVectorOffset] at h
  exact Fin.ext (by omega)

theorem layer_norm_ops_fwd_residual_out_offset_general_injective
    (s : BlockState) (stride_res_out_row BLOCK_N : Nat) :
    Function.Injective
      (fun i : Fin BLOCK_N => fwdResidualOutOffset s stride_res_out_row i) := by
  intro a b h
  simp only [fwdResidualOutOffset, bwdRowVectorOffset] at h
  exact Fin.ext (by omega)


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- Plain layer-norm forward, **dimension-general**: exposes the genuine `Y`,
`Mean`, and `Rstd` outputs together for arbitrary feature dim `N`, tile width
`BLOCK_N`, and `Y` row stride `stride_y_row`. -/
theorem layer_norm_ops_fwd_plain_bias_all_outputs_compute_correct_general
    (ValuePre MeanPre RstdPre Y Mean Rstd : RegionName) (s : BlockState)
    (stride_y_row N BLOCK_N : Nat) :
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i : Fin BLOCK_N =>
        fwdYStoreSpec s ValuePre stride_y_row i)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_mean_store_slice MeanPre Mean)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, meanRowOffset s))
      (expected := fun _ => meanStoreSpec s MeanPre)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_rstd_store_slice RstdPre Rstd)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, meanRowOffset s))
      (expected := fun _ => rstdStoreSpec s RstdPre)) := by
  refine ⟨?_, ?_, ?_⟩
  · exact layer_norm_ops_fwd_bias_y_store_slice_compute_correct ValuePre Y
      stride_y_row N BLOCK_N s
      (layer_norm_ops_fwd_y_offset_general_injective s stride_y_row BLOCK_N)
  · exact layer_norm_ops_fwd_mean_store_slice_compute_correct MeanPre Mean s
  · exact layer_norm_ops_fwd_rstd_store_slice_compute_correct RstdPre Rstd s


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- RMS layer-norm forward with bias, **dimension-general**: exposes the genuine
`Y` and `Rstd` outputs together. -/
theorem layer_norm_ops_fwd_rms_bias_all_outputs_compute_correct_general
    (ValuePre RstdPre Y Rstd : RegionName) (s : BlockState)
    (stride_y_row N BLOCK_N : Nat) :
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i : Fin BLOCK_N =>
        fwdYStoreSpec s ValuePre stride_y_row i)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_rstd_store_slice RstdPre Rstd)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, meanRowOffset s))
      (expected := fun _ => rstdStoreSpec s RstdPre)) := by
  refine ⟨?_, ?_⟩
  · exact layer_norm_ops_fwd_rms_bias_y_store_slice_compute_correct ValuePre Y
      stride_y_row N BLOCK_N s
      (layer_norm_ops_fwd_y_offset_general_injective s stride_y_row BLOCK_N)
  · exact layer_norm_ops_fwd_rstd_store_slice_compute_correct RstdPre Rstd s


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- Residual plain layer-norm forward with bias, **dimension-general**: exposes
the genuine `RESIDUAL_OUT`, `Y`, `Mean`, and `Rstd` outputs together for
arbitrary `RESIDUAL_OUT`/`Y` row strides, feature dim `N`, and tile width
`BLOCK_N`. -/
theorem layer_norm_ops_fwd_residual_bias_all_outputs_compute_correct_general
    (ResidualPre ValuePre MeanPre RstdPre RESIDUAL_OUT Y Mean Rstd : RegionName)
    (s : BlockState)
    (stride_res_out_row stride_y_row N BLOCK_N : Nat) :
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_residual_out_store_slice ResidualPre
        RESIDUAL_OUT stride_res_out_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (RESIDUAL_OUT, fwdResidualOutOffset s stride_res_out_row i)))
      (expected := fun i : Fin BLOCK_N =>
        fwdResidualOutStoreSpec s ResidualPre stride_res_out_row i)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i : Fin BLOCK_N =>
        fwdYStoreSpec s ValuePre stride_y_row i)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_mean_store_slice MeanPre Mean)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, meanRowOffset s))
      (expected := fun _ => meanStoreSpec s MeanPre)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_fwd_rstd_store_slice RstdPre Rstd)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, meanRowOffset s))
      (expected := fun _ => rstdStoreSpec s RstdPre)) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact layer_norm_ops_fwd_residual_out_store_slice_compute_correct ResidualPre
      RESIDUAL_OUT stride_res_out_row N BLOCK_N s
      (layer_norm_ops_fwd_residual_out_offset_general_injective s
        stride_res_out_row BLOCK_N)
  · exact layer_norm_ops_fwd_bias_y_store_slice_compute_correct ValuePre Y
      stride_y_row N BLOCK_N s
      (layer_norm_ops_fwd_y_offset_general_injective s stride_y_row BLOCK_N)
  · exact layer_norm_ops_fwd_mean_store_slice_compute_correct MeanPre Mean s
  · exact layer_norm_ops_fwd_rstd_store_slice_compute_correct RstdPre Rstd s

/-- RMS backward, **dimension-general**: exposes the genuine `c1` reduction,
`DX`, and partial `DW` store slices together for arbitrary row strides, feature
dim `N`, and tile width `BLOCK_N`. -/
theorem layer_norm_ops_bwd_rms_core_outputs_compute_correct_general
    (X Xhat W DY Rstd C1 DX DW : RegionName) (s : BlockState)
    (stride_xhat_row stride_dy_row stride_x_row stride_dx_row N BLOCK_N : Nat)
    (hDWDX : DW ≠ DX) :
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_c1_reduction_slice Xhat W DY C1
        stride_xhat_row stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := fun _ : PUnit => some (C1, s.pid))
      (expected := fun _ : PUnit =>
        bwdC1ReductionSpec s Xhat W DY stride_xhat_row stride_dy_row N BLOCK_N)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_rms_dx_from_c1_slice Xhat W DY Rstd C1 DX
        stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdRmsDXFromC1Spec s Xhat W DY Rstd C1 stride_xhat_row stride_dy_row i)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_rms_one_row X W DY DX DW Rstd
        stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DW, bwdRmsDWOffset s N i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdRmsDWSpec s X DY Rstd stride_x_row stride_dy_row N BLOCK_N i)) := by
  refine ⟨?_, ?_, ?_⟩
  · exact layer_norm_ops_bwd_c1_reduction_slice_compute_correct Xhat W DY C1
      stride_xhat_row stride_dy_row N BLOCK_N s
  · exact layer_norm_ops_bwd_rms_dx_from_c1_slice_compute_correct
      Xhat W DY Rstd C1 DX stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N s
      (layer_norm_ops_rms_dx_offset_general_injective s stride_dx_row BLOCK_N)
  · exact layer_norm_ops_bwd_rms_one_row_dw_compute_correct X W DY DX DW Rstd
      stride_x_row stride_dy_row stride_dx_row N BLOCK_N s hDWDX
      (layer_norm_ops_rms_dw_offset_general_injective s N BLOCK_N)

/-- Plain+bias backward, **dimension-general**: exposes the genuine `c1`/`c2`
reductions, `DX`, partial `DW`, and partial `DB` store slices together for
arbitrary row strides, feature dim `N`, and tile width `BLOCK_N`. -/
theorem layer_norm_ops_bwd_plain_bias_core_outputs_compute_correct_general
    (X Xhat W DY DX DW DB Mean Rstd C1 C2 : RegionName) (s : BlockState)
    (stride_xhat_row stride_dy_row stride_x_row stride_dx_row N BLOCK_N : Nat)
    (hDWDX : DW ≠ DX) (hDWDB : DW ≠ DB)
    (hDBDX : DB ≠ DX) (hDBDW : DB ≠ DW) :
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_c1_reduction_slice Xhat W DY C1
        stride_xhat_row stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := fun _ : PUnit => some (C1, s.pid))
      (expected := fun _ : PUnit =>
        bwdC1ReductionSpec s Xhat W DY stride_xhat_row stride_dy_row N BLOCK_N)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_c2_reduction_slice W DY C2
        stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := fun _ : PUnit => some (C2, s.pid))
      (expected := fun _ : PUnit =>
        bwdC2ReductionSpec s W DY stride_dy_row N BLOCK_N)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_plain_dx_from_c1_c2_slice Xhat W DY Rstd
        C1 C2 DX stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdPlainDXFromC1C2Spec s Xhat W DY Rstd C1 C2 stride_xhat_row
          stride_dy_row i)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_plain_bias_one_row X W DY DX DW DB Mean Rstd
        stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DW, bwdParamGradOffset s N i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdPlainBiasDWSpec s X DY Mean Rstd stride_x_row stride_dy_row N
          BLOCK_N i)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_plain_bias_one_row X W DY DX DW DB Mean Rstd
        stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DB, bwdParamGradOffset s N i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdBiasDBSpec s DY stride_dy_row i)) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · exact layer_norm_ops_bwd_c1_reduction_slice_compute_correct Xhat W DY C1
      stride_xhat_row stride_dy_row N BLOCK_N s
  · exact layer_norm_ops_bwd_c2_reduction_slice_compute_correct W DY C2
      stride_dy_row N BLOCK_N s
  · exact layer_norm_ops_bwd_plain_dx_from_c1_c2_slice_compute_correct
      Xhat W DY Rstd C1 C2 DX stride_xhat_row stride_dy_row stride_dx_row N
      BLOCK_N s
      (layer_norm_ops_rms_dx_offset_general_injective s stride_dx_row BLOCK_N)
  · exact layer_norm_ops_bwd_plain_bias_one_row_dw_compute_correct
      X W DY DX DW DB Mean Rstd stride_x_row stride_dy_row stride_dx_row N
      BLOCK_N s hDWDX hDWDB
      (layer_norm_ops_param_grad_offset_general_injective s N BLOCK_N)
  · exact layer_norm_ops_bwd_plain_bias_one_row_db_compute_correct
      X W DY DX DW DB Mean Rstd stride_x_row stride_dy_row stride_dx_row N
      BLOCK_N s hDBDX hDBDW
      (layer_norm_ops_param_grad_offset_general_injective s N BLOCK_N)


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- Backward residual-add, **dimension-general**: exposes both genuine
observable row-vector stores (`DX` and `DRESIDUAL_IN`) produced by the
`dx += dres` branch, for arbitrary row strides, feature dim `N`, and tile width
`BLOCK_N`. -/
theorem layer_norm_ops_bwd_residual_add_all_outputs_compute_correct_general
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName) (s : BlockState)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat)
    (hDXDresIn : DX ≠ DRESIDUAL_IN)
    (hDresInDX : DRESIDUAL_IN ≠ DX) :
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_residual_add_store_slice DXBase DRESIDUAL
        DX DRESIDUAL_IN stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdResidualAddSpec s DXBase DRESIDUAL stride_dx_row stride_dres_row i)) ∧
    (ComputeCorrect.Realizes
      (kernel := layer_norm_ops_bwd_residual_add_store_slice DXBase DRESIDUAL
        DX DRESIDUAL_IN stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DRESIDUAL_IN, bwdDResidualInOffset s stride_dres_in_row i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdResidualAddSpec s DXBase DRESIDUAL stride_dx_row stride_dres_row i)) := by
  refine ⟨?_, ?_⟩
  · exact layer_norm_ops_bwd_residual_add_store_slice_dx_compute_correct
      DXBase DRESIDUAL DX DRESIDUAL_IN stride_dx_row stride_dres_row
      stride_dres_in_row N BLOCK_N s hDXDresIn
      (layer_norm_ops_rms_dx_offset_general_injective s stride_dx_row BLOCK_N)
  · exact layer_norm_ops_bwd_residual_add_store_slice_dresidual_in_compute_correct
      DXBase DRESIDUAL DX DRESIDUAL_IN stride_dx_row stride_dres_row
      stride_dres_in_row N BLOCK_N s hDresInDX
      (layer_norm_ops_dresidual_in_offset_general_injective s
        stride_dres_in_row BLOCK_N)

/-! ## Python test-shape aggregate outputs -/

theorem layer_norm_fwd_1pass_surface_rms_only_y_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYRstd : Y ≠ Rstd) (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.false Bool.false Bool.false) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          rmsYSpec s X W stride_x_row N BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
          evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.mul, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWRstd, rmsYSpec, rmsInvCarrier, rmsVarCarrier, rmsInputTile,
            xOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYRstd]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the `Y` output of the surface's
RMS/no-residual/no-bias branch. -/
theorem layer_norm_fwd_1pass_surface_rms_only_y_compute_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s : BlockState)
    (hYRstd : Y ≠ Rstd) (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.false Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i => rmsYSpec s X W stride_x_row N BLOCK_N eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_1pass_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_fwd_1pass_surface_rms_only_y_correct X Y W B RESIDUAL
    RESIDUAL_OUT Mean Rstd stride_x_row stride_y_row stride_res_row
    stride_res_out_row N BLOCK_N eps s s' hYRstd hWRstd hOutInj hExec i
  simpa [hActive] using h

/-- End-to-end correctness for the `Rstd` scalar output of the surface's
RMS/no-residual/no-bias branch. -/
theorem layer_norm_fwd_1pass_surface_rms_only_rstd_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hRstdY : Rstd ≠ Y)
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.false Bool.false Bool.false) s = some s') :
    s'.readMem Rstd s.pid =
      rmsInvVarFullSpec s X stride_x_row N BLOCK_N eps := by
  simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
        evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.mul, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRstdY]
  simp only [BlockState.setReg_readMem, BlockState.writeMem_readMem]
  simp [rmsInvVarFullSpec, rmsInvCarrier, rmsVarCarrier, rmsInputTile,
        xOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realSqrt, NumericDType.mul]
  rfl

/-- Compute-facing correctness for the `Rstd` output of the surface's
RMS/no-residual/no-bias branch. -/
theorem layer_norm_fwd_1pass_surface_rms_only_rstd_compute_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.false Bool.false Bool.false)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        rmsInvVarFullSpec s X stride_x_row N BLOCK_N eps) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_fwd_1pass_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layer_norm_fwd_1pass_surface_rms_only_rstd_correct X Y W B RESIDUAL
    RESIDUAL_OUT Mean Rstd stride_x_row stride_y_row stride_res_row
    stride_res_out_row N BLOCK_N eps s s' hRstdY hExec

/-- End-to-end correctness for the `Y` output of the surface's
RMS/no-residual/bias branch (matches Python test case 3). -/
theorem layer_norm_fwd_1pass_surface_rms_bias_y_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYRstd : Y ≠ Rstd) (hWRstd : W ≠ Rstd) (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.false Bool.false Bool.true) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          rmsBiasYSpec s X W B stride_x_row N BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
          evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.mul, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWRstd, hBRstd, rmsBiasYSpec, rmsInvCarrier, rmsVarCarrier,
            rmsInputTile, xOffset, Tile.reduceSum, Tile.reduceSumDrop,
            Tile.select, TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex, WithBot.realSqrt, NumericDType.mul,
            NumericDType.add]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYRstd]
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end correctness for the `Y` output of the surface's
non-RMS/no-residual/bias branch (matches Python test case 1). -/
theorem layer_norm_fwd_1pass_surface_plain_bias_y_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYMean : Y ≠ Mean) (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean) (hWRstd : W ≠ Rstd)
    (hBMean : B ≠ Mean) (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.false Bool.false Bool.false Bool.true) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          plainBiasYSpec s X W B stride_x_row N BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.sub, NumericDType.mul, NumericDType.div,
          ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
          FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
          ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWMean, hWRstd, hBMean, hBRstd, plainBiasYSpec,
            plainInvCarrier, plainVarCarrier, plainXbarTile,
            plainMeanCarrier, plainInputTile, xOffset, Tile.reduceSum,
            Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
            NumericDType.mul, NumericDType.sub, NumericDType.add]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYMean, hYRstd]
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end correctness for the `Y` output of the surface's
non-RMS/residual/bias branch without residual-out storage (subset of
Python test case 5; the test case additionally enables STORE_RESIDUAL_OUT,
which only adds a writeback past the proved `Y` value). -/
theorem layer_norm_fwd_1pass_surface_plain_residual_bias_y_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYMean : Y ≠ Mean) (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean) (hWRstd : W ≠ Rstd)
    (hBMean : B ≠ Mean) (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.false Bool.true Bool.false Bool.true) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          plainResidualBiasYSpec s X RESIDUAL W B stride_x_row stride_res_row
            N BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.sub, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWMean, hWRstd, hBMean, hBRstd, plainResidualBiasYSpec,
            plainResidualInvCarrier, plainResidualVarCarrier,
            plainResidualXbarTile, plainResidualMeanCarrier,
            plainResidualInputTile, xOffset, resOffset, Tile.reduceSum,
            Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
            NumericDType.mul, NumericDType.sub, NumericDType.add]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYMean, hYRstd]
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end correctness for the `Rstd` scalar output of the surface's
RMS/no-residual/bias branch (matches Python test case 3). -/
theorem layer_norm_fwd_1pass_surface_rms_bias_rstd_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hRstdY : Rstd ≠ Y)
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.false Bool.false Bool.true) s = some s') :
    s'.readMem Rstd s.pid =
      rmsInvVarFullSpec s X stride_x_row N BLOCK_N eps := by
  simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
        evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.mul, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRstdY]
  simp only [BlockState.setReg_readMem, BlockState.writeMem_readMem]
  simp [rmsInvVarFullSpec, rmsInvCarrier, rmsVarCarrier, rmsInputTile,
        xOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realSqrt, NumericDType.mul]
  rfl

/-- End-to-end correctness for the `Mean` scalar output of the surface's
non-RMS/no-residual/bias branch (matches Python test case 1). -/
theorem layer_norm_fwd_1pass_surface_plain_bias_mean_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hMeanRstd : Mean ≠ Rstd) (hMeanY : Mean ≠ Y)
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.false Bool.false Bool.false Bool.true) s = some s') :
    s'.readMem Mean s.pid =
      plainMeanFullSpec s X stride_x_row N BLOCK_N := by
  simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
        evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.sub, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hMeanY]
  simp [BlockState.writeMem_readMem, hMeanRstd, plainMeanFullSpec,
        plainMeanCarrier, plainInputTile, xOffset, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex]
  rfl

/-- End-to-end correctness for the `Y` output of the surface's
non-RMS/no-residual/no-bias branch. -/
theorem layer_norm_fwd_1pass_surface_plain_y_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYMean : Y ≠ Mean) (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean) (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.false Bool.false Bool.false Bool.false) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          plainYSpec s X W stride_x_row N BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
          evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.sub, NumericDType.mul, NumericDType.div,
          ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
          FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
          ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWMean, hWRstd, plainYSpec, plainInvCarrier, plainVarCarrier,
            plainXbarTile, plainMeanCarrier, plainInputTile, xOffset,
            Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul, NumericDType.sub]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYMean, hYRstd]
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end correctness for the `Y` output of the surface's
RMS/residual/no-bias branch (excludes STORE_RESIDUAL_OUT). -/
theorem layer_norm_fwd_1pass_surface_rms_residual_y_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYRstd : Y ≠ Rstd) (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.true Bool.false Bool.false) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          rmsResidualYSpec s X RESIDUAL W stride_x_row stride_res_row N
            BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.mul, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWRstd, rmsResidualYSpec, rmsResidualInvCarrier,
            rmsResidualVarCarrier, plainResidualInputTile, xOffset, resOffset,
            Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul, NumericDType.add]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYRstd]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end correctness for the `Rstd` scalar output of the surface's
non-RMS/no-residual/bias branch (matches Python test case 1). -/
theorem layer_norm_fwd_1pass_surface_plain_bias_rstd_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hRstdY : Rstd ≠ Y)
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.false Bool.false Bool.false Bool.true) s = some s') :
    s'.readMem Rstd s.pid =
      plainInvVarFullSpec s X stride_x_row N BLOCK_N eps := by
  simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
        evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.sub, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRstdY]
  simp only [BlockState.setReg_readMem, BlockState.writeMem_readMem]
  simp [plainInvVarFullSpec, plainInvCarrier, plainVarCarrier, plainXbarTile,
        plainMeanCarrier, plainInputTile, xOffset, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
        NumericDType.mul, NumericDType.sub]
  rfl

/-- End-to-end correctness for the `Y` output of the surface's
RMS/residual/bias branch (excludes STORE_RESIDUAL_OUT). -/
theorem layer_norm_fwd_1pass_surface_rms_residual_bias_y_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYRstd : Y ≠ Rstd) (hWRstd : W ≠ Rstd) (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.true Bool.false Bool.true) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          rmsResidualBiasYSpec s X RESIDUAL W B stride_x_row stride_res_row N
            BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts,
          stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd,
          Tile.uop, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.div,
          ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
          FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
          ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWRstd, hBRstd, rmsResidualBiasYSpec,
            rmsResidualInvCarrier, rmsResidualVarCarrier,
            rmsResidualXbarTile, plainResidualInputTile, xOffset, resOffset,
            Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul, NumericDType.add]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYRstd]
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end correctness for the `Y` output of the surface's
non-RMS/residual/bias/store-residual-out branch (matches Python test case 5
exactly: HAS_RESIDUAL = STORE_RESIDUAL_OUT = HAS_BIAS = true). Closes the
`STORE_RESIDUAL_OUT=true` gap via the disjointness-based commutation
through the `RESIDUAL_OUT` foldl using
`BlockState.foldl_writeMem_const_region_prop_masked_readMem_other` from
`Semantics/State.lean`. -/
theorem layer_norm_fwd_1pass_surface_plain_residual_bias_store_y_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYRso : Y ≠ RESIDUAL_OUT) (hYMean : Y ≠ Mean) (hYRstd : Y ≠ Rstd)
    (hWRso : W ≠ RESIDUAL_OUT) (hWMean : W ≠ Mean) (hWRstd : W ≠ Rstd)
    (hBRso : B ≠ RESIDUAL_OUT) (hBMean : B ≠ Mean) (hBRstd : B ≠ Rstd)
    (hXRso : X ≠ RESIDUAL_OUT) (hRRso : RESIDUAL ≠ RESIDUAL_OUT)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.false Bool.true Bool.true Bool.true) s = some s') :
    ∀ i : Fin BLOCK_N, ¬ i.val < N →
      s'.readMem Y (yOffset s stride_y_row i) =
        s.readMem Y (yOffset s stride_y_row i) := by
  intro i hi
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.sub, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    simp only [hi, if_false]
    simp only [BlockState.writeMem_readMem, hYMean, hYRstd,
               BlockState.setReg_readMem]
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
          (region := RESIDUAL_OUT) (R := Y)
          (P := fun (k : TileIndex [BLOCK_N]) => k.1.val < N) (hRR := hYRso)]
    simp [BlockState.setReg_readMem]
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end correctness for the `Y` output of the surface's
non-RMS/residual/bias/store-residual-out branch at the **active lane**
(i.val < N). Combined with `*_store_y_correct`, this closes the
STORE_RESIDUAL_OUT=true gap for Python test case 5. -/
theorem layer_norm_fwd_1pass_surface_plain_residual_bias_store_y_active_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYRso : Y ≠ RESIDUAL_OUT) (hYMean : Y ≠ Mean) (hYRstd : Y ≠ Rstd)
    (hWRso : W ≠ RESIDUAL_OUT) (hWMean : W ≠ Mean) (hWRstd : W ≠ Rstd)
    (hBRso : B ≠ RESIDUAL_OUT) (hBMean : B ≠ Mean) (hBRstd : B ≠ Rstd)
    (hXRso : X ≠ RESIDUAL_OUT) (hRRso : RESIDUAL ≠ RESIDUAL_OUT)
    (hMeanRso : Mean ≠ RESIDUAL_OUT) (hRstdRso : Rstd ≠ RESIDUAL_OUT)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.false Bool.true Bool.true Bool.true) s = some s') :
    ∀ i : Fin BLOCK_N, i.val < N →
      s'.readMem Y (yOffset s stride_y_row i) =
        plainResidualBiasYSpec s X RESIDUAL W B stride_x_row stride_res_row
          N BLOCK_N eps i := by
  intro i hi
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.sub, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    simp only [hi, if_true]
    simp [hi, hWMean, hWRstd, hWRso, hBMean, hBRstd, hBRso, hXRso, hRRso,
          hMeanRso, hRstdRso, plainResidualBiasYSpec,
          plainResidualInvCarrier, plainResidualVarCarrier,
          plainResidualXbarTile, plainResidualMeanCarrier,
          plainResidualInputTile, xOffset, resOffset, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
          NumericDType.mul, NumericDType.sub, NumericDType.add,
          BlockState.foldl_writeMem_const_region_prop_masked_readMem_other]
    rfl
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end correctness for the `Y` output of the surface's
RMS/residual/bias/store-residual-out branch (active lane). Same proof
pattern as `_plain_residual_bias_store_y_active_correct` but with
`IS_RMS_NORM = true`. -/
theorem layer_norm_fwd_1pass_surface_rms_residual_bias_store_y_active_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYRso : Y ≠ RESIDUAL_OUT) (hYRstd : Y ≠ Rstd)
    (hWRso : W ≠ RESIDUAL_OUT) (hWRstd : W ≠ Rstd)
    (hBRso : B ≠ RESIDUAL_OUT) (hBRstd : B ≠ Rstd)
    (hXRso : X ≠ RESIDUAL_OUT) (hRRso : RESIDUAL ≠ RESIDUAL_OUT)
    (hRstdRso : Rstd ≠ RESIDUAL_OUT)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.true Bool.true Bool.true) s = some s') :
    ∀ i : Fin BLOCK_N, i.val < N →
      s'.readMem Y (yOffset s stride_y_row i) =
        rmsResidualBiasYSpec s X RESIDUAL W B stride_x_row stride_res_row N
          BLOCK_N eps i := by
  intro i hi
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.sub, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    simp only [hi, if_true]
    simp [hi, hWRstd, hWRso, hBRstd, hBRso, hXRso, hRRso, hRstdRso,
          rmsResidualBiasYSpec, rmsResidualInvCarrier, rmsResidualVarCarrier,
          rmsResidualXbarTile, plainResidualInputTile, xOffset, resOffset,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
          NumericDType.mul, NumericDType.add,
          BlockState.foldl_writeMem_const_region_prop_masked_readMem_other]
    rfl
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end Y correctness for surface's RMS/no-residual/no-bias with
STORE_RESIDUAL_OUT=true (active lane). -/
theorem layer_norm_fwd_1pass_surface_rms_only_store_y_active_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYRso : Y ≠ RESIDUAL_OUT) (hYRstd : Y ≠ Rstd)
    (hWRso : W ≠ RESIDUAL_OUT) (hWRstd : W ≠ Rstd)
    (hXRso : X ≠ RESIDUAL_OUT) (hRstdRso : Rstd ≠ RESIDUAL_OUT)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.false Bool.true Bool.false) s = some s') :
    ∀ i : Fin BLOCK_N, i.val < N →
      s'.readMem Y (yOffset s stride_y_row i) =
        rmsYSpec s X W stride_x_row N BLOCK_N eps i := by
  intro i hi
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
          evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.mul, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    simp only [hi, if_true]
    simp [hi, hWRstd, hWRso, hXRso, hRstdRso, rmsYSpec, rmsInvCarrier,
          rmsVarCarrier, rmsInputTile, xOffset, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
          NumericDType.mul,
          BlockState.foldl_writeMem_const_region_prop_masked_readMem_other]
    rfl
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end Y correctness for surface's RMS/no-residual/bias with
STORE_RESIDUAL_OUT=true (active lane). -/
theorem layer_norm_fwd_1pass_surface_rms_bias_store_y_active_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYRso : Y ≠ RESIDUAL_OUT) (hYRstd : Y ≠ Rstd)
    (hWRso : W ≠ RESIDUAL_OUT) (hWRstd : W ≠ Rstd)
    (hBRso : B ≠ RESIDUAL_OUT) (hBRstd : B ≠ Rstd)
    (hXRso : X ≠ RESIDUAL_OUT) (hRstdRso : Rstd ≠ RESIDUAL_OUT)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.false Bool.true Bool.true) s = some s') :
    ∀ i : Fin BLOCK_N, i.val < N →
      s'.readMem Y (yOffset s stride_y_row i) =
        rmsBiasYSpec s X W B stride_x_row N BLOCK_N eps i := by
  intro i hi
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
          evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.mul, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    simp only [hi, if_true]
    simp [hi, hWRstd, hWRso, hBRstd, hBRso, hXRso, hRstdRso,
          rmsBiasYSpec, rmsInvCarrier, rmsVarCarrier, rmsInputTile, xOffset,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
          NumericDType.mul, NumericDType.add,
          BlockState.foldl_writeMem_const_region_prop_masked_readMem_other]
    rfl
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end Y correctness for surface's non-RMS/no-residual/no-bias with
STORE_RESIDUAL_OUT=true (active lane). -/
theorem layer_norm_fwd_1pass_surface_plain_store_y_active_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYRso : Y ≠ RESIDUAL_OUT) (hYMean : Y ≠ Mean) (hYRstd : Y ≠ Rstd)
    (hWRso : W ≠ RESIDUAL_OUT) (hWMean : W ≠ Mean) (hWRstd : W ≠ Rstd)
    (hXRso : X ≠ RESIDUAL_OUT) (hMeanRso : Mean ≠ RESIDUAL_OUT)
    (hRstdRso : Rstd ≠ RESIDUAL_OUT)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.false Bool.false Bool.true Bool.false) s = some s') :
    ∀ i : Fin BLOCK_N, i.val < N →
      s'.readMem Y (yOffset s stride_y_row i) =
        plainYSpec s X W stride_x_row N BLOCK_N eps i := by
  intro i hi
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
          evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.sub, NumericDType.mul, NumericDType.div,
          ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
          FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
          ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    simp only [hi, if_true]
    simp [hi, hWMean, hWRstd, hWRso, hXRso, hMeanRso, hRstdRso,
          plainYSpec, plainInvCarrier, plainVarCarrier, plainXbarTile,
          plainMeanCarrier, plainInputTile, xOffset, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
          NumericDType.mul, NumericDType.sub,
          BlockState.foldl_writeMem_const_region_prop_masked_readMem_other]
    rfl
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end Y correctness for surface's non-RMS/no-residual/bias with
STORE_RESIDUAL_OUT=true (active lane). -/
theorem layer_norm_fwd_1pass_surface_plain_bias_store_y_active_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYRso : Y ≠ RESIDUAL_OUT) (hYMean : Y ≠ Mean) (hYRstd : Y ≠ Rstd)
    (hWRso : W ≠ RESIDUAL_OUT) (hWMean : W ≠ Mean) (hWRstd : W ≠ Rstd)
    (hBRso : B ≠ RESIDUAL_OUT) (hBMean : B ≠ Mean) (hBRstd : B ≠ Rstd)
    (hXRso : X ≠ RESIDUAL_OUT) (hMeanRso : Mean ≠ RESIDUAL_OUT)
    (hRstdRso : Rstd ≠ RESIDUAL_OUT)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.false Bool.false Bool.true Bool.true) s = some s') :
    ∀ i : Fin BLOCK_N, i.val < N →
      s'.readMem Y (yOffset s stride_y_row i) =
        plainBiasYSpec s X W B stride_x_row N BLOCK_N eps i := by
  intro i hi
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
          evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.sub, NumericDType.mul, NumericDType.div,
          ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
          FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
          ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    simp only [hi, if_true]
    simp [hi, hWMean, hWRstd, hWRso, hBMean, hBRstd, hBRso, hXRso, hMeanRso,
          hRstdRso, plainBiasYSpec, plainInvCarrier, plainVarCarrier,
          plainXbarTile, plainMeanCarrier, plainInputTile, xOffset,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
          NumericDType.mul, NumericDType.sub, NumericDType.add,
          BlockState.foldl_writeMem_const_region_prop_masked_readMem_other]
    rfl
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end Y correctness for surface's non-RMS/residual/no-bias with
STORE_RESIDUAL_OUT=true (active lane). -/
theorem layer_norm_fwd_1pass_surface_plain_residual_store_y_active_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYRso : Y ≠ RESIDUAL_OUT) (hYMean : Y ≠ Mean) (hYRstd : Y ≠ Rstd)
    (hWRso : W ≠ RESIDUAL_OUT) (hWMean : W ≠ Mean) (hWRstd : W ≠ Rstd)
    (hXRso : X ≠ RESIDUAL_OUT) (hRRso : RESIDUAL ≠ RESIDUAL_OUT)
    (hMeanRso : Mean ≠ RESIDUAL_OUT) (hRstdRso : Rstd ≠ RESIDUAL_OUT)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.false Bool.true Bool.true Bool.false) s = some s') :
    ∀ i : Fin BLOCK_N, i.val < N →
      s'.readMem Y (yOffset s stride_y_row i) =
        plainResidualYSpec s X RESIDUAL W stride_x_row stride_res_row N
          BLOCK_N eps i := by
  intro i hi
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
          evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.sub, NumericDType.mul, NumericDType.div,
          ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
          FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
          ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    simp only [hi, if_true]
    simp [hi, hWMean, hWRstd, hWRso, hXRso, hRRso, hMeanRso, hRstdRso,
          plainResidualYSpec, plainResidualInvCarrier,
          plainResidualVarCarrier, plainResidualXbarTile,
          plainResidualMeanCarrier, plainResidualInputTile, xOffset,
          resOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          WithBot.realSqrt, NumericDType.mul, NumericDType.sub,
          NumericDType.add,
          BlockState.foldl_writeMem_const_region_prop_masked_readMem_other]
    rfl
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end Y correctness for surface's RMS/residual/no-bias with
STORE_RESIDUAL_OUT=true (active lane). -/
theorem layer_norm_fwd_1pass_surface_rms_residual_store_y_active_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYRso : Y ≠ RESIDUAL_OUT) (hYRstd : Y ≠ Rstd)
    (hWRso : W ≠ RESIDUAL_OUT) (hWRstd : W ≠ Rstd)
    (hXRso : X ≠ RESIDUAL_OUT) (hRRso : RESIDUAL ≠ RESIDUAL_OUT)
    (hRstdRso : Rstd ≠ RESIDUAL_OUT)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.true Bool.true Bool.false) s = some s') :
    ∀ i : Fin BLOCK_N, i.val < N →
      s'.readMem Y (yOffset s stride_y_row i) =
        rmsResidualYSpec s X RESIDUAL W stride_x_row stride_res_row N
          BLOCK_N eps i := by
  intro i hi
  by_cases hB0 : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
          evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.mul, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    simp only [hi, if_true]
    simp [hi, hWRstd, hWRso, hXRso, hRRso, hRstdRso, rmsResidualYSpec,
          rmsResidualInvCarrier, rmsResidualVarCarrier,
          plainResidualInputTile, xOffset, resOffset, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
          NumericDType.mul, NumericDType.add,
          BlockState.foldl_writeMem_const_region_prop_masked_readMem_other]
    rfl
  · exact False.elim (hB0 (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end correctness for the `Y` output of the surface's
non-RMS/residual/no-bias branch (excludes STORE_RESIDUAL_OUT). -/
theorem layer_norm_fwd_1pass_surface_plain_residual_y_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hYMean : Y ≠ Mean) (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean) (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i))
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.false Bool.true Bool.false Bool.false) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Y (yOffset s stride_y_row i) =
        if i.val < N then
          plainResidualYSpec s X RESIDUAL W stride_x_row stride_res_row N
            BLOCK_N eps i
        else s.readMem Y (yOffset s stride_y_row i) := by
  intro i
  by_cases hB : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.sub, NumericDType.mul, NumericDType.div,
          ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
          FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
          ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, hWMean, hWRstd, plainResidualYSpec,
            plainResidualInvCarrier, plainResidualVarCarrier,
            plainResidualXbarTile, plainResidualMeanCarrier,
            plainResidualInputTile, xOffset, resOffset, Tile.reduceSum,
            Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex, WithBot.realSqrt,
            NumericDType.mul, NumericDType.sub, NumericDType.add]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYMean, hYRstd]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- End-to-end correctness for the `Rstd` scalar output of the surface's
RMS/residual/no-bias branch (excludes STORE_RESIDUAL_OUT). -/
theorem layer_norm_fwd_1pass_surface_rms_residual_rstd_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hRstdY : Rstd ≠ Y)
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.true Bool.false Bool.false) s = some s') :
    s'.readMem Rstd s.pid =
      rmsResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N eps := by
  simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
        evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.mul, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRstdY]
  simp only [BlockState.setReg_readMem, BlockState.writeMem_readMem]
  simp [rmsResidualInvVarFullSpec, rmsResidualInvCarrier,
        rmsResidualVarCarrier, rmsResidualXbarTile, plainResidualInputTile,
        xOffset, resOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realSqrt, NumericDType.mul, NumericDType.add]
  rfl

/-- End-to-end correctness for the `Rstd` scalar output of the surface's
RMS/residual/bias branch (excludes STORE_RESIDUAL_OUT). -/
theorem layer_norm_fwd_1pass_surface_rms_residual_bias_rstd_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hRstdY : Rstd ≠ Y)
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.true Bool.false Bool.true) s = some s') :
    s'.readMem Rstd s.pid =
      rmsResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N eps := by
  simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt, evalOp,
        evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.mul, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRstdY]
  simp only [BlockState.setReg_readMem, BlockState.writeMem_readMem]
  simp [rmsResidualInvVarFullSpec, rmsResidualInvCarrier,
        rmsResidualVarCarrier, rmsResidualXbarTile, plainResidualInputTile,
        xOffset, resOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realSqrt, NumericDType.mul, NumericDType.add]
  rfl

/-- End-to-end correctness for the `Mean` scalar output of the surface's
non-RMS/residual/no-bias branch (excludes STORE_RESIDUAL_OUT). -/
theorem layer_norm_fwd_1pass_surface_plain_residual_mean_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hMeanRstd : Mean ≠ Rstd) (hMeanY : Mean ≠ Y)
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.false Bool.true Bool.false Bool.false) s = some s') :
    s'.readMem Mean s.pid =
      plainResidualMeanFullSpec s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N := by
  simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.sub, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hMeanY]
  simp [BlockState.writeMem_readMem, hMeanRstd, plainResidualMeanFullSpec,
        plainResidualMeanCarrier, plainResidualInputTile, xOffset, resOffset,
        Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex]
  rfl

/-- End-to-end correctness for the `Rstd` scalar output of the surface's
non-RMS/residual/no-bias branch (excludes STORE_RESIDUAL_OUT). -/
theorem layer_norm_fwd_1pass_surface_plain_residual_rstd_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hRstdY : Rstd ≠ Y)
    (hExec : exec (layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.false Bool.true Bool.false Bool.false) s = some s') :
    s'.readMem Rstd s.pid =
      plainResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N eps := by
  simp [exec, layer_norm_fwd_1pass_surface, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
        NumericDType.sub, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
        Y _ _ _ _ _ _ _ hRstdY]
  simp only [BlockState.setReg_readMem, BlockState.writeMem_readMem]
  simp [plainResidualInvVarFullSpec, plainResidualInvCarrier,
        plainResidualVarCarrier, plainResidualXbarTile,
        plainResidualMeanCarrier, plainResidualInputTile, xOffset, resOffset,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realSqrt, NumericDType.mul, NumericDType.sub,
        NumericDType.add]
  rfl

/-- End-to-end backward correctness for the `DB` output when
`rows_per_program = 0` (degenerate empty row-block): the kernel stores zeros
to `DB`, the initial value of the accumulator. -/
theorem layer_norm_bwd_surface_zero_rows_db_zero
    (X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_dy_row stride_dx_row stride_dres_row
      stride_dres_in_row M N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hDBDW : DB ≠ DW)
    (hExec : exec (layer_norm_bwd_surface X W B Y DY DX DW DB DRESIDUAL
        DRESIDUAL_IN Mean Rstd stride_x_row stride_y_row stride_dy_row
        stride_dx_row stride_dres_row stride_dres_in_row M N 0 BLOCK_N eps
        Bool.true Bool.false Bool.false Bool.true Bool.false) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem DB (s.pid * N + i.val) =
        if i.val < N then 0.0
        else s.readMem DB (s.pid * N + i.val) := by
  intro i
  by_cases hB : 0 < BLOCK_N
  · simp [exec, layer_norm_bwd_surface, stepStmts, stepStmt, evalOp,
          evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.mul, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          Nat.mul_zero, Nat.zero_mul, Nat.min_zero, Nat.zero_min] at hExec
    subst s'
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi]; norm_num
    · simp [hi, BlockState.writeMem_readMem,
            BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
              DW _ _ _ _ _ _ _ hDBDW]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing wrapper for the actual backward surface's degenerate
`DB` writeback. This exposes the surface-level `HAS_BIAS=true` partial-bias
store through the standard `ComputeCorrect.Realizes` interface. -/
theorem layer_norm_bwd_surface_zero_rows_db_zero_compute_correct
    (X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_dy_row stride_dx_row stride_dres_row
      stride_dres_in_row M N BLOCK_N : Nat)
    (eps : ℝ) (s : BlockState)
    (hDBDW : DB ≠ DW) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_bwd_surface X W B Y DY DX DW DB DRESIDUAL
        DRESIDUAL_IN Mean Rstd stride_x_row stride_y_row stride_dy_row
        stride_dx_row stride_dres_row stride_dres_in_row M N 0 BLOCK_N eps
        Bool.true Bool.false Bool.false Bool.true Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DB, s.pid * N + i.val)))
      (expected := fun _ : Fin BLOCK_N => (0.0 : ℝ)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_bwd_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_bwd_surface_zero_rows_db_zero X W B Y DY DX DW DB
    DRESIDUAL DRESIDUAL_IN Mean Rstd stride_x_row stride_y_row stride_dy_row
    stride_dx_row stride_dres_row stride_dres_in_row M N BLOCK_N eps s s'
    hDBDW hExec i
  simpa [hActive] using h

/-! ## End-to-end backward surface correctness (degenerate row-block)

The backward kernel iterates the row loop `rows_per_program` times. The
theorem below proves the simplest end-to-end case for the actual surface
`layer_norm_bwd_surface`: when `rows_per_program = 0`, the loop body never
executes, so the only observable output is the partial `DW` writeback at the
end of the kernel, which stores the initial zero accumulator. This is the
first surface-level proof that the backward kernel's pre-loop setup,
empty-loop semantics, and post-loop `DW` scatter compose correctly. -/

theorem layer_norm_bwd_surface_zero_rows_dw_zero
    (X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_dy_row stride_dx_row stride_dres_row
      stride_dres_in_row M N BLOCK_N : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hExec : exec (layer_norm_bwd_surface X W B Y DY DX DW DB DRESIDUAL
        DRESIDUAL_IN Mean Rstd stride_x_row stride_y_row stride_dy_row
        stride_dx_row stride_dres_row stride_dres_in_row M N 0 BLOCK_N eps
        Bool.true Bool.false Bool.false Bool.false Bool.false) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem DW (s.pid * N + i.val) =
        if i.val < N then 0.0
        else s.readMem DW (s.pid * N + i.val) := by
  intro i
  by_cases hB : 0 < BLOCK_N
  · simp [exec, layer_norm_bwd_surface, stepStmts, stepStmt, evalOp,
          evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, Tile.select, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.mul, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          Nat.mul_zero, Nat.zero_mul, Nat.min_zero, Nat.zero_min] at hExec
    subst s'
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi]; norm_num
    · simp [hi, BlockState.writeMem_readMem]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing wrapper for the actual backward surface's degenerate
`DW` writeback. It proves the standard correctness surface for the full
backward DSL surface when the row loop has no iterations. -/
theorem layer_norm_bwd_surface_zero_rows_dw_zero_compute_correct
    (X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_dy_row stride_dx_row stride_dres_row
      stride_dres_in_row M N BLOCK_N : Nat)
    (eps : ℝ) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := layer_norm_bwd_surface X W B Y DY DX DW DB DRESIDUAL
        DRESIDUAL_IN Mean Rstd stride_x_row stride_y_row stride_dy_row
        stride_dx_row stride_dres_row stride_dres_in_row M N 0 BLOCK_N eps
        Bool.true Bool.false Bool.false Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DW, s.pid * N + i.val)))
      (expected := fun _ : Fin BLOCK_N => (0.0 : ℝ)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layer_norm_bwd_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_bwd_surface_zero_rows_dw_zero X W B Y DY DX DW DB
    DRESIDUAL DRESIDUAL_IN Mean Rstd stride_x_row stride_y_row stride_dy_row
    stride_dx_row stride_dres_row stride_dres_in_row M N BLOCK_N eps s s'
    hExec i
  simpa [hActive] using h

end Correct


end VeriTile.Bench.TritonBenchG.LayerNormOps
