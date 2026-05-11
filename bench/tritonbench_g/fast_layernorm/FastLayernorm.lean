import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FastLayernorm

open VeriTile.Triton

set_option maxHeartbeats 5000000

/-- Faithful transcription of `fast_layernorm.py`'s `layernorm_forward`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter.
- Python pointer mutation `Y += ...` / `X += ...` / `r += ...` / `mu += ...`
  -> explicit base pointer registers.
- Python `.to(...)` casts are omitted at the algorithm layer.
- Python scalar division by `n_cols` is written as division by
  `tl.toReal($(n_cols))` to make the algorithm-layer dtype explicit. -/
def layernorm_forward
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)

  Y_base = Y + row_idx * $(Y_row_stride)
  X_base = X + row_idx * $(X_row_stride)
  r_base = r + row_idx
  mu_base = mu + row_idx

  X_row = tl.load(X_base + col_offsets, mask=mask, other=0.0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0.0)
  b_row = tl.load(bias + col_offsets, mask=mask, other=0.0)

  mean_X = tl.sum(X_row, axis=0) / tl.toReal($(n_cols))
  XX = X_row - mean_X
  row_var = tl.sum(XX * XX, axis=0) / tl.toReal($(n_cols))
  inv_var = tl.math.rsqrt(row_var + $(eps))
  tl.store(r_base, inv_var)
  tl.store(mu_base, mean_X)
  output = (XX * inv_var) * W_row + b_row
  tl.store(Y_base + col_offsets, output, mask=mask)
}

noncomputable def layernormInputTile
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * X_row_stride + idx.1.val
      if idx.1.val < n_cols then some (s.readMem X off) else some (0.0 : ℝ) }

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
        (layernormCenteredTile s X X_row_stride n_cols BLOCK_SIZE))).data PUnit.unit)
    ((Tile.scalar n_cols).natToReal.data PUnit.unit)

noncomputable def layernormInvVarCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (layernormVarCarrier s X X_row_stride n_cols BLOCK_SIZE)))

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
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * Y_row_stride + idx.1.val) := by
    intro a bidx h
    have hab : a.1 = bidx.1 := by
      apply hOutInj
      simpa [yOutOffset] using h
    cases a
    cases bidx
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, layernorm_forward, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.sub, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot] at hExec
    subst s'
    simp only [yOutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, layernormYSpec, layernormInvVarCarrier, layernormVarCarrier,
            layernormCenteredTile, layernormMeanCarrier, layernormInputTile,
            Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul]
      rfl
    · simp [hi, BlockState.writeMem_readMem, hYr, hYmu]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

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
  · simp [layernorm_forward]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layernorm_forward_y_correct Y X W bias r mu Y_row_stride X_row_stride
    n_cols BLOCK_SIZE eps s s' hYr hYmu hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.FastLayernorm
