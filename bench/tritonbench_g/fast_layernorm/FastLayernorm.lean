import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FastLayernorm

open VeriTile.Triton

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

noncomputable def layernormInputTile
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * X_row_stride + idx.1.val
      if idx.1.val < n_cols then some (s.readMem X off) else some (0 : ℝ) }

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
        (layernormCenteredTile s X X_row_stride n_cols BLOCK_SIZE))).data PUnit.unit)

noncomputable def layernormInvVarCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  WithBot.realRsqrt
    (Option.map (fun a => a + eps)
      (layernormVarCarrier s X X_row_stride n_cols BLOCK_SIZE))

noncomputable def layernormYSpec
    (s : BlockState) (X W bias : RegionName)
    (X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun affine bias => affine + bias)
      (Option.map₂ (fun scaled w => scaled * w)
        (Option.map₂ (fun centered inv => centered * inv)
          (Option.map₂ (fun x mean => x - mean)
            (some (s.readMem X (s.pid * X_row_stride + idx.val)))
            (layernormMeanCarrier s X X_row_stride n_cols BLOCK_SIZE))
          (layernormInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps))
        (some (s.readMem W idx.val)))
      (some (s.readMem bias idx.val)))

def yOutOffset (s : BlockState) (Y_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * Y_row_stride + i.val

/-- Executed-state correctness for the `Y` output of `layernorm_forward`. -/
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
          layernormYSpec s X W bias X_row_stride n_cols BLOCK_SIZE eps i
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
            layernormCenteredTile, layernormMeanCarrier, layernormInputTile,
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
  let dyw_i := s.readMem dY (s.pid * dY_row_stride + i.val) *
    s.readMem W i.val
  let normed_i_inv := (s.readMem X (s.pid * X_row_stride + i.val) -
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
    · simp [hi, layernormDXSpec, layernormBackwardDYWTile,
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
    ComputeCorrect.Realizes
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

/-- Compute-facing correctness for the `Y` output of `layernorm_forward`. -/
theorem layernorm_forward_y_compute_correct
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hYr : Y ≠ r) (hYmu : Y ≠ mu)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i)) :
    ComputeCorrect.Realizes
      (kernel := layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
        n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (Y, yOutOffset s Y_row_stride i)))
      (expected := fun i =>
        layernormYSpec s X W bias X_row_stride n_cols BLOCK_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layernorm_forward, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layernorm_forward_y_correct Y X W bias r mu Y_row_stride X_row_stride
    n_cols BLOCK_SIZE eps s s' hYr hYmu hOutInj hExec i
  simpa [hActive] using h

/-- Full-kernel spec for the `inv_var` (rstd) store of `layernorm_forward`.

Wraps `layernormInvVarCarrier` with `WithBot.unbotD 0` so that the readback
of the kernel's scalar write into `r` matches the carrier's value as a
plain `ℝ`. -/
noncomputable def invVarFullSpec
    (s : BlockState) (X : RegionName)
    (X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (layernormInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps)

/-- Full-kernel spec for the `mean` (mu) store of `layernorm_forward`.

Wraps `layernormMeanCarrier` with `WithBot.unbotD 0` so that the readback
of the kernel's scalar write into `mu` matches the carrier's value as a
plain `ℝ`. -/
noncomputable def meanFullSpec
    (s : BlockState) (X : RegionName)
    (X_row_stride n_cols BLOCK_SIZE : Nat) : ℝ :=
  WithBot.unbotD 0
    (layernormMeanCarrier s X X_row_stride n_cols BLOCK_SIZE)

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
      invVarFullSpec s X X_row_stride n_cols BLOCK_SIZE eps := by
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
        layernormCenteredTile, layernormMeanCarrier, layernormInputTile,
        Tile.bop, Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.mul,
        WithBot.realRsqrt]
  rfl

/-- Compute-facing correctness for the `inv_var` (rstd) scalar store of
`layernorm_forward`. -/
theorem layernorm_forward_inv_var_compute_correct
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hRY : r ≠ Y) (hRmu : r ≠ mu) :
    ComputeCorrect.Realizes
      (kernel := layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
        n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := fun _ : PUnit => some (r, s.pid))
      (expected := fun _ =>
        invVarFullSpec s X X_row_stride n_cols BLOCK_SIZE eps) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layernorm_forward, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layernorm_forward_inv_var_correct Y X W bias r mu Y_row_stride
    X_row_stride n_cols BLOCK_SIZE eps s s' hRY hRmu hExec

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
      meanFullSpec s X X_row_stride n_cols BLOCK_SIZE := by
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
  simp only [meanFullSpec, layernormMeanCarrier, layernormInputTile,
        Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex]
  rfl

/-- Compute-facing correctness for the `mean` (mu) scalar store of
`layernorm_forward`. -/
theorem layernorm_forward_mean_compute_correct
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hMuY : mu ≠ Y) :
    ComputeCorrect.Realizes
      (kernel := layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
        n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := fun _ : PUnit => some (mu, s.pid))
      (expected := fun _ =>
        meanFullSpec s X X_row_stride n_cols BLOCK_SIZE) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layernorm_forward, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layernorm_forward_mean_correct Y X W bias r mu Y_row_stride
    X_row_stride n_cols BLOCK_SIZE eps s s' hMuY hExec

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
    ComputeCorrect.Realizes
      (kernel := layernorm_forward_inv_var_store_slice InvVarPre r)
      (initialState := s)
      (write := fun _ : PUnit => some (r, rOutOffset s))
      (expected := fun _ => invVarStoreSpec s InvVarPre) := by
  unfold ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
      (kernel := layernorm_forward_mean_store_slice MeanPre mu)
      (initialState := s)
      (write := fun _ : PUnit => some (mu, rOutOffset s))
      (expected := fun _ => meanStoreSpec s MeanPre) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layernorm_forward_mean_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact layernorm_forward_mean_store_slice_correct MeanPre mu s s' hExec

end VeriTile.Bench.TritonBenchG.FastLayernorm
