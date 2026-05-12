import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm

open VeriTile.Triton

set_option maxHeartbeats 5000000

/-- Faithful transcription of `fast_rms_layernorm.py`'s
`_rms_layernorm_forward`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter.
- Python `normed.to(W_row.dtype)` is represented with the source-region
  element dtype `W.dtype.element_ty`, which is the dtype of `W_row`. -/
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
  normed = (normed).to(W.dtype.element_ty)
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

The Python kernel accepts `dW`/`dW_row_stride` but does not write `dW`; this
surface preserves the in-place `dY` writeback. -/
def rms_layernorm_backward
    (dY X W r _dW : RegionName)
    (dY_row_stride X_row_stride r_row_stride _dW_row_stride n_cols : Nat)
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
`_rms_layernorm_backward` for `GEMMA = true`. -/
def gemma_rms_layernorm_backward
    (dY X W r _dW : RegionName)
    (dY_row_stride X_row_stride r_row_stride _dW_row_stride n_cols : Nat)
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

noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * X_row_stride + idx.1.val
      if idx.1.val < n_cols then some (s.readMem X off) else some (0 : ℝ) }

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
        (some (s.readMem X (s.pid * X_row_stride + idx.val)))
        (rmsInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps))
      (some (s.readMem W (idx.val * W_row_stride))))

noncomputable def gemmaRmsLayernormYSpec
    (s : BlockState) (X W : RegionName)
    (X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * (w + 1.0))
      (Option.map₂ (fun x inv => x * inv)
        (some (s.readMem X (s.pid * X_row_stride + idx.val)))
        (rmsInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps))
      (some (s.readMem W idx.val)))

noncomputable def rmsBackwardDYTile
    (s : BlockState) (dY W : RegionName)
    (dY_row_stride n_cols BLOCK_SIZE : Nat) (GEMMA : Bool) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      Option.map₂ (fun dy w => dy * (w + if GEMMA then 1.0 else 0.0))
        (if idx.1.val < n_cols then
          some (s.readMem dY (s.pid * dY_row_stride + idx.1.val))
        else some (0 : ℝ))
        (if idx.1.val < n_cols then some (s.readMem W idx.1.val)
        else some (0 : ℝ)) }

noncomputable def rmsBackwardDYTilePlain
    (s : BlockState) (dY W : RegionName)
    (dY_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      Option.map₂ (fun dy w => dy * w)
        (if idx.1.val < n_cols then
          some (s.readMem dY (s.pid * dY_row_stride + idx.1.val))
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
          some (s.readMem dY (s.pid * dY_row_stride + idx.1.val))
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
          some (s.readMem X (s.pid * X_row_stride + idx.1.val))
        else some (0 : ℝ)) }

noncomputable def rmsBackwardRowSumCarrier
    (s : BlockState) (dY X W r : RegionName)
    (dY_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) (GEMMA : Bool) :
    WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (rmsBackwardDYTile s dY W dY_row_stride n_cols BLOCK_SIZE GEMMA)
      (rmsBackwardNormedTile s X r X_row_stride r_row_stride
        n_cols BLOCK_SIZE))).data PUnit.unit

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

noncomputable def rmsBackwardDYSpec
    (s : BlockState) (dY X W r : RegionName)
    (dY_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (GEMMA : Bool) (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun dYW rowsum =>
      s.readMem r (s.pid * r_row_stride) / (n_cols : ℝ) *
        ((n_cols : ℝ) * dYW -
          (s.readMem X (s.pid * X_row_stride + idx.val) *
            s.readMem r (s.pid * r_row_stride)) * rowsum))
      ((rmsBackwardDYTile s dY W dY_row_stride n_cols BLOCK_SIZE GEMMA).data
        (idx, PUnit.unit))
      (rmsBackwardRowSumCarrier s dY X W r dY_row_stride X_row_stride
        r_row_stride n_cols BLOCK_SIZE GEMMA))

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
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * Y_row_stride + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [yOutOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, rms_layernorm_forward, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, rmsLayernormYSpec, rmsInvVarCarrier, rmsSumCarrier,
            rmsInputTile, Tile.reduceSum, Tile.reduceSumDrop,
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
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * Y_row_stride + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [yOutOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, gemma_rms_layernorm_forward, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [yOutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, gemmaRmsLayernormYSpec, rmsInvVarCarrier, rmsSumCarrier,
            rmsInputTile, Tile.reduceSum, Tile.reduceSumDrop,
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
    (dY_row_stride X_row_stride r_row_stride _dW_row_stride n_cols BLOCK_SIZE : Nat)
    (_eps : ℝ) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => dyOutOffset s dY_row_stride i))
    (hExec : exec (rms_layernorm_backward dY X W r _dW dY_row_stride X_row_stride
          r_row_stride _dW_row_stride n_cols _eps BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem dY (dyOutOffset s dY_row_stride i) =
        if i.val < n_cols then
          rmsBackwardDYSpecPlain s dY X W r dY_row_stride X_row_stride
            r_row_stride n_cols BLOCK_SIZE i
        else s.readMem dY (dyOutOffset s dY_row_stride i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * dY_row_stride + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [dyOutOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, rms_layernorm_backward, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.mul,
          NumericDType.sub, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [dyOutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, rmsBackwardDYSpecPlain, rmsBackwardRowSumCarrierPlain,
            rmsBackwardDYTilePlain, rmsBackwardNormedTile, Tile.reduceSum,
            Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex, NumericDType.mul]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the in-place `dY` output of
`_rms_layernorm_backward` with `GEMMA = false`. -/
theorem rms_layernorm_backward_dy_compute_correct
    (dY X W r _dW : RegionName)
    (dY_row_stride X_row_stride r_row_stride _dW_row_stride n_cols BLOCK_SIZE : Nat)
    (_eps : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => dyOutOffset s dY_row_stride i)) :
    ComputeCorrect.Realizes
      (kernel := rms_layernorm_backward dY X W r _dW dY_row_stride X_row_stride
        r_row_stride _dW_row_stride n_cols _eps BLOCK_SIZE)
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
    r_row_stride _dW_row_stride n_cols BLOCK_SIZE _eps s s' hOutInj hExec i
  simpa [hActive] using h

/-- Executed-state correctness for the in-place `dY` output of
`_rms_layernorm_backward` with `GEMMA = true`. -/
theorem gemma_rms_layernorm_backward_dy_correct
    (dY X W r _dW : RegionName)
    (dY_row_stride X_row_stride r_row_stride _dW_row_stride n_cols BLOCK_SIZE : Nat)
    (_eps : ℝ) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => dyOutOffset s dY_row_stride i))
    (hExec : exec (gemma_rms_layernorm_backward dY X W r _dW dY_row_stride X_row_stride
          r_row_stride _dW_row_stride n_cols _eps BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem dY (dyOutOffset s dY_row_stride i) =
        if i.val < n_cols then
          rmsBackwardDYSpecGemma s dY X W r dY_row_stride X_row_stride
            r_row_stride n_cols BLOCK_SIZE i
        else s.readMem dY (dyOutOffset s dY_row_stride i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * dY_row_stride + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [dyOutOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, gemma_rms_layernorm_backward, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.sub, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    subst s'
    simp only [dyOutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, rmsBackwardDYSpecGemma, rmsBackwardRowSumCarrierGemma,
            rmsBackwardDYTileGemma, rmsBackwardNormedTile, Tile.reduceSum,
            Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex, NumericDType.mul]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the in-place `dY` output of
`_rms_layernorm_backward` with `GEMMA = true`. -/
theorem gemma_rms_layernorm_backward_dy_compute_correct
    (dY X W r _dW : RegionName)
    (dY_row_stride X_row_stride r_row_stride _dW_row_stride n_cols BLOCK_SIZE : Nat)
    (_eps : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => dyOutOffset s dY_row_stride i)) :
    ComputeCorrect.Realizes
      (kernel := gemma_rms_layernorm_backward dY X W r _dW dY_row_stride X_row_stride
        r_row_stride _dW_row_stride n_cols _eps BLOCK_SIZE)
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
    r_row_stride _dW_row_stride n_cols BLOCK_SIZE _eps s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.FastRmsLayernorm
