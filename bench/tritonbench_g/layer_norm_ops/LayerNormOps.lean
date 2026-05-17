import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.LayerNormOps

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

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
  · simp [exec, layer_norm_fwd_rms_one_block, stepStmts, stepStmt, evalOp,
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
  simp [exec, layer_norm_fwd_rms_one_block, stepStmts, stepStmt, evalOp,
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
  simp [exec, layer_norm_ops_fwd_mean_store_slice, stepStmts, stepStmt, evalOp,
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
  simp [exec, layer_norm_ops_fwd_rstd_store_slice, stepStmts, stepStmt, evalOp,
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

end VeriTile.Bench.TritonBenchG.LayerNormOps
