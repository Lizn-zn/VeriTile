import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Semantics.MaskedReduction

namespace VeriTile.Bench.TritonBenchG.L2NormTriton1

open VeriTile.Triton
open VeriTile.Triton.TiledL2Norm

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `l2_norm_triton1.py`'s `_l2_norm_fwd_1pass_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` → Lean `Nat` parameter. -/
def l2_norm_fwd_1pass_kernel
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_x_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  y = x * rstd
  tl.store(Y + cols, y, mask=mask)
}

noncomputable def l2InputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      let off := s.pid * stride_x_row + idx.1.val
      if idx.1.val < N then
        if idx.1.val < N then some (s.readMem X off) else some (0.0 : ℝ)
      else
        some (0.0 : ℝ) }

noncomputable def l2Load
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (idx : Fin BLOCK_N) : ℝ :=
  if idx.val < N then
    s.readMem X (s.pid * stride_x_row + idx.val)
  else
    0

noncomputable def l2VarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) : WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (l2InputTile s X stride_x_row N BLOCK_N)
      (l2InputTile s X stride_x_row N BLOCK_N))).data PUnit.unit

theorem l2VarCarrier_eq_l2NormSqSum
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    l2VarCarrier s X stride_x_row N BLOCK_N =
      some (l2NormSqSum (l2Load s X stride_x_row N BLOCK_N)) := by
  simp [l2VarCarrier, l2InputTile, Tile.reduceSum,
    Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, Tile.bop, NumericDType.mul,
    BlockState.pid_eq]
  refine (reduceSum_masked_sq_eq_some_sum
    (fun k : Fin BLOCK_N => s.readMem X (s.pids 0 * stride_x_row + k.val))
    (fun k : Fin BLOCK_N => k.val < N)).trans ?_
  congr 1
  unfold l2NormSqSum l2Load
  apply Finset.sum_congr rfl
  intro k _hk
  by_cases h : k.val < N <;> simp [h, BlockState.pid_eq]

noncomputable def l2RstdCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) (eps : ℝ) :
    WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt (Option.map (fun a => a + eps)
      (l2VarCarrier s X stride_x_row N BLOCK_N)))

noncomputable def l2Spec
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) (eps : ℝ)
    (idx : Fin BLOCK_N) : ℝ :=
  l2Norm (l2Load s X stride_x_row N BLOCK_N) eps idx

theorem l2_norm_fwd_1pass_kernel_correct
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s s' : BlockState)
    (hExec : exec (l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      let outAddr := s.pid * stride_x_row + i.val
      s'.readMem Y outAddr =
        if i.val < N then
          l2Spec s X stride_x_row N BLOCK_N eps i
        else s.readMem Y outAddr := by
  intro i
  by_cases hB : 0 < BLOCK_N
  · simp [exec, l2_norm_fwd_1pass_kernel, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.div,
          ComparableDType.lt, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    rw [← hExec]
    simp [BlockState.pid_eq]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · have hvar := l2VarCarrier_eq_l2NormSqSum s X stride_x_row N BLOCK_N
      simp [l2VarCarrier, l2InputTile, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.bop, NumericDType.mul, BlockState.pid_eq] at hvar
      simp [hi, l2Spec, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, WithBot.realSqrt, l2Load, l2Norm, l2NormRstd]
      erw [hvar]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem l2_norm_fwd_1pass_kernel_compute_correct
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_N => i.val < N)
          (fun i => (Y, s.pid * stride_x_row + i.val)))
      (expected := fun i => l2Spec s X stride_x_row N BLOCK_N eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [l2_norm_fwd_1pass_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := l2_norm_fwd_1pass_kernel_correct X Y stride_x_row N eps BLOCK_N s s' hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.L2NormTriton1
