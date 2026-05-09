import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.L2NormTriton1

open VeriTile.Triton

/-- Faithful transcription of `l2_norm_triton1.py`'s `_l2_norm_fwd_1pass_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` → Lean `Nat` parameter.
- Python pointer `X += row * stride_x_row` / `Y += row * stride_x_row` →
  explicit `X_base` / `Y_base` registers; this is the same address expression
  without mutating the pointer argument name. -/
def l2_norm_fwd_1pass_kernel
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X_base = X + row * $(stride_x_row)
  Y_base = Y + row * $(stride_x_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = (tl.load(X_base + cols, mask=cols < $(N), other=0.0)).to(tl.float32)
  xbar = tl.where(cols < $(N), x, (0.0).to(tl.float32))
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  y = x * rstd
  tl.store(Y_base + cols, y, mask=mask)
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

noncomputable def l2VarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) : WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (l2InputTile s X stride_x_row N BLOCK_N)
      (l2InputTile s X stride_x_row N BLOCK_N))).data PUnit.unit

noncomputable def l2Var
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0 (l2VarCarrier s X stride_x_row N BLOCK_N)

noncomputable def l2RstdCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) (eps : ℝ) :
    WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt (Option.map (fun a => a + eps)
      (l2VarCarrier s X stride_x_row N BLOCK_N)))

noncomputable def l2Spec
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) (eps : ℝ)
    (idx : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun x rstd => x * rstd)
      (some (s.readMem X (s.pid * stride_x_row + idx.val)))
      (l2RstdCarrier s X stride_x_row N BLOCK_N eps))

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
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * stride_x_row + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  by_cases hB : 0 < BLOCK_N
  · simp [exec, l2_norm_fwd_1pass_kernel, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.div,
          ComparableDType.lt] at hExec
    subst s'
    simp [BlockState.pid_eq]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, l2Spec, l2RstdCarrier, l2VarCarrier, l2InputTile,
            Tile.reduceSum, Tile.reduceSumDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
            WithBot.realSqrt, NumericDType.mul]
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
  · simp [l2_norm_fwd_1pass_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := l2_norm_fwd_1pass_kernel_correct X Y stride_x_row N eps BLOCK_N s s' hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.L2NormTriton1
