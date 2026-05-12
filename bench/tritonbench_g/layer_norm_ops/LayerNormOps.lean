import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.LayerNormOps

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Surface transcription of `layer_norm_ops.py`'s `_layer_norm_fwd_1pass_kernel`.

This preserves the forward constexpr branches for residual input,
`RESIDUAL_OUT`, RMS-vs-layer norm, bias, `Mean`/`Rstd` side stores, and masked
`Y` writeback. -/
def layer_norm_fwd_1pass_surface
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row
      N BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS : Bool) :
    ComputeKernel := triton {
  row = tl.program_id(axis=0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  if HAS_RESIDUAL {
    RESIDUAL += row * $(stride_res_row)
  }
  if STORE_RESIDUAL_OUT {
    RESIDUAL_OUT += row * $(stride_res_out_row)
  }
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
  if HAS_RESIDUAL {
    residual = tl.load(RESIDUAL + cols, mask=mask, other=0.0).to(tl.float32)
    x += residual
  }
  if STORE_RESIDUAL_OUT {
    tl.store(RESIDUAL_OUT + cols, x, mask=mask)
  }
  mean = 0.0
  x_centered = x
  var = 0.0
  if IS_RMS_NORM {
    xbar = tl.where(mask, x_centered, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / $(N)
  } else {
    mean = tl.sum(x, axis=0) / $(N)
    tl.store(Mean + row, mean)
    x_centered = x - mean
    xbar = tl.where(mask, x_centered, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / $(N)
  }
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = x_centered * rstd
  if HAS_BIAS {
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    y = x_hat * w + b
  } else {
    y = x_hat * w
  }
  tl.store(Y + cols, y, mask=mask)
}

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
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * stride_y_row + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [yOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_N
  · simp [exec, layer_norm_fwd_rms_one_block, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
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

end VeriTile.Bench.TritonBenchG.LayerNormOps
