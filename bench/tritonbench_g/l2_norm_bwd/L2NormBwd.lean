import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.L2NormBwd

open VeriTile.Triton

/-- Faithful transcription of `l2_norm_bwd.py`'s `_l2_norm_bwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` -> Lean `Nat` parameter.
- Python pointer mutation `X += ...` / `DX += ...` / `DY += ...` -> explicit
  base pointer registers. -/
def l2_norm_bwd_kernel
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X_base = X + row * $(stride_x_row)
  DX_base = DX + row * $(stride_x_row)
  DY_base = DY + row * $(stride_x_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X_base + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x = tl.where(cols < $(N), x, (0.0).to(tl.float32))
  var = tl.sum(x * x)
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  dy = tl.load(DY_base + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  dy = tl.where(cols < $(N), dy, (0.0).to(tl.float32))
  dx = dy * rstd - tl.sum(dy * x) * (1 / (var + $(eps))) * rstd * x
  tl.store(DX_base + cols, dx, mask=mask)
}

noncomputable def l2BwdInputTile
    (s : BlockState) (R : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      let off := s.pids 0 * stride_x_row + idx.1.val
      if idx.1.val < N then
        if idx.1.val < N then some (s.readMem R off) else some (0.0 : ℝ)
      else
        some (0.0 : ℝ) }

noncomputable def l2BwdVarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) : WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (l2BwdInputTile s X stride_x_row N BLOCK_N)
      (l2BwdInputTile s X stride_x_row N BLOCK_N))).data PUnit.unit

noncomputable def l2BwdDotCarrier
    (s : BlockState) (X DY : RegionName) (stride_x_row N BLOCK_N : Nat) : WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (l2BwdInputTile s DY stride_x_row N BLOCK_N)
      (l2BwdInputTile s X stride_x_row N BLOCK_N))).data PUnit.unit

noncomputable def l2BwdRstdCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) (eps : ℝ) :
    WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt (Option.map (fun a => a + eps)
      (l2BwdVarCarrier s X stride_x_row N BLOCK_N)))

noncomputable def l2BwdSpec
    (s : BlockState) (X DY : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) (idx : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun lhs rhs => lhs - rhs)
      (Option.map₂ (fun dy rstd => dy * rstd)
        (some (s.readMem DY (s.pids 0 * stride_x_row + idx.val)))
        (l2BwdRstdCarrier s X stride_x_row N BLOCK_N eps))
      (Option.map₂ (fun coeff x => coeff * x)
        (Option.map₂ (fun lhs rstd => lhs * rstd)
          (Option.map₂ (fun dot invVar => dot * invVar)
            (l2BwdDotCarrier s X DY stride_x_row N BLOCK_N)
            (Option.map ((fun b => b⁻¹) ∘ fun a => a + eps)
              (l2BwdVarCarrier s X stride_x_row N BLOCK_N)))
          (l2BwdRstdCarrier s X stride_x_row N BLOCK_N eps))
        (some (s.readMem X (s.pids 0 * stride_x_row + idx.val)))))

def l2BwdOutOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * stride_x_row + i.val

/-- Algorithm-layer correctness for `_l2_norm_bwd_kernel`. -/
theorem l2_norm_bwd_kernel_correct
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s s' : BlockState)
    (hExec : exec (l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      let outAddr := l2BwdOutOffset s stride_x_row i
      s'.readMem DX outAddr =
        if i.val < N then
          l2BwdSpec s X DY stride_x_row N BLOCK_N eps i
        else s.readMem DX outAddr := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * stride_x_row + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  by_cases hB : 0 < BLOCK_N
  · simp [exec, l2_norm_bwd_kernel, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.sub, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot] at hExec
    subst s'
    simp only [l2BwdOutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, l2BwdSpec, l2BwdRstdCarrier, l2BwdDotCarrier, l2BwdVarCarrier,
            l2BwdInputTile, Tile.reduceSum, Tile.reduceSumDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for `_l2_norm_bwd_kernel`. -/
theorem l2_norm_bwd_kernel_compute_correct
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, l2BwdOutOffset s stride_x_row i)))
      (expected := fun i => l2BwdSpec s X DY stride_x_row N BLOCK_N eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [l2_norm_bwd_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := l2_norm_bwd_kernel_correct X DY DX stride_x_row N eps BLOCK_N s s' hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.L2NormBwd
