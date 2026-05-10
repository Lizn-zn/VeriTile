import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.LayerNormLiger

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

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

  Y_base = Y + row_idx * $(Y_row_stride)
  X_base = X + row_idx * $(X_row_stride)
  Mean_base = Mean + row_idx * $(Mean_row_stride)
  RSTD_base = RSTD + row_idx * $(RSTD_row_stride)

  X_row = tl.load(X_base + col_offsets, mask=mask, other=0.0)
  W_row = tl.load(W + col_offsets, mask=mask, other=0.0)
  B_row = tl.load(B + col_offsets, mask=mask, other=0.0)

  mean = tl.sum(X_row, axis=0) / tl.toReal($(n_cols))
  XX = X_row - mean
  row_var = tl.sum(XX * XX, axis=0) / tl.toReal($(n_cols))
  inv_var = 1 / tl.sqrt(row_var + $(eps))
  tl.store(Mean_base, mean)
  tl.store(RSTD_base, inv_var)
  output = (XX * inv_var) * W_row + B_row
  tl.store(Y_base + col_offsets, output, mask=mask)
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
      else some (0.0 : ℝ) }

noncomputable def layernormMeanCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map₂ (fun a n => a / n)
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (layernormInputTile s X X_row_stride n_cols BLOCK_SIZE)).data PUnit.unit)
    ((Tile.scalar n_cols).natToReal.data PUnit.unit)

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
  Option.map₂ (fun a n => a / n)
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (layernormCenteredTile s X X_row_stride n_cols BLOCK_SIZE)
        (layernormCenteredTile s X X_row_stride n_cols BLOCK_SIZE))).data
        PUnit.unit)
    ((Tile.scalar n_cols).natToReal.data PUnit.unit)

noncomputable def layernormInvVarCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (layernormVarCarrier s X X_row_stride n_cols BLOCK_SIZE)))

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
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * Y_row_stride + idx.1.val) := by
    intro a bidx h
    have hab : a.1 = bidx.1 := by
      apply hOutInj
      simpa [yOffset] using h
    cases a
    cases bidx
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, layer_norm_liger_forward, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.sub, NumericDType.div, ComparableDType.lt] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, layernormYSpec, layernormInvVarCarrier, layernormVarCarrier,
            layernormCenteredTile, layernormMeanCarrier, layernormInputTile,
            xOffset, Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul]
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
  · simp [layer_norm_liger_forward]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layer_norm_liger_forward_y_correct Y X W B Mean RSTD Y_row_stride
    X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps s s'
    hYMean hYRSTD hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.LayerNormLiger
