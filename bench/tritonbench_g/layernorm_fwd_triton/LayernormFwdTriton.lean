import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.LayernormFwdTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Proof-oriented one-block slice of `layernorm_fwd_triton.py`'s
`_layer_norm_fwd_kernel`.

The upstream kernel loops over `range(0, N, BLOCK_SIZE)` for mean, variance,
and output. This slice specializes to the single-block case while keeping the
two-dimensional `Seq`/`H` program ids, base pointer arithmetic, masked loads,
mean/variance/rstd computation, W multiply, and masked Y store.

Allowed mechanical Lean-syntax-only changes:
- Python pointer mutation is written using explicit base pointer registers.
- Python `.to(...)` casts are omitted at the algorithm layer.
- Python `N: tl.constexpr` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat`
  parameters.
- The Python `stride_x_hd`, `stride_y_hd`, and `stride_w_hd` parameters are not
  used by the source kernel body, so this proof slice omits them. -/
def layernorm_fwd_triton_one_block
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn : Nat)
    (N BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  Seq = tl.program_id(0)
  H = tl.program_id(1)
  X_base = X + Seq * $(stride_x_N) + H * $(stride_x_hn)
  Y_base = Y + Seq * $(stride_y_N) + H * $(stride_y_hn)
  W_base = W + H * $(stride_w_hn)
  cols = tl.arange(0, $(BLOCK_SIZE))
  a = tl.load(X_base + cols, mask=cols < $(N), other=0.0)
  mean = tl.sum(a, axis=0) / tl.toReal($(N))
  x_for_var = tl.load(X_base + cols, mask=cols < $(N), other=0.0)
  centered_for_var = tl.where(cols < $(N), x_for_var - mean, 0.0)
  var = tl.sum(centered_for_var * centered_for_var, axis=0) / tl.toReal($(N))
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  w = tl.load(W_base + cols, mask=mask)
  x = tl.load(X_base + cols, mask=mask, other=0.0)
  x_hat = (x - mean) * rstd
  y = x_hat * w
  tl.store(Y_base + cols, y, mask=mask)
}

def xOffset
    (s : BlockState) (stride_x_N stride_x_hn : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride_x_N + s.pids 1 * stride_x_hn + i.val

def yOffset
    (s : BlockState) (stride_y_N stride_y_hn : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride_y_N + s.pids 1 * stride_y_hn + i.val

def wOffset (s : BlockState) (stride_w_hn : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * stride_w_hn + i.val

noncomputable def layernormInputTile
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride_x_N stride_x_hn idx.1))
      else some (0.0 : ℝ) }

noncomputable def layernormMeanCarrier
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map₂ (fun a n => a / n)
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (layernormInputTile s X stride_x_N stride_x_hn N BLOCK_SIZE)).data
        PUnit.unit)
    ((Tile.scalar N).natToReal.data PUnit.unit)

noncomputable def layernormCenteredTile
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N then
        Option.map₂ (fun x mean => x - mean)
          (if idx.1.val < N then
            some (s.readMem X (xOffset s stride_x_N stride_x_hn idx.1))
          else some (0.0 : ℝ))
          (layernormMeanCarrier s X stride_x_N stride_x_hn N BLOCK_SIZE)
      else some (0.0 : ℝ) }

noncomputable def layernormVarCarrier
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map₂ (fun a n => a / n)
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (layernormCenteredTile s X stride_x_N stride_x_hn N BLOCK_SIZE)
        (layernormCenteredTile s X stride_x_N stride_x_hn N BLOCK_SIZE))).data
        PUnit.unit)
    ((Tile.scalar N).natToReal.data PUnit.unit)

noncomputable def layernormInvVarCarrier
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) (eps : ℝ) :
    WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (layernormVarCarrier s X stride_x_N stride_x_hn N BLOCK_SIZE)))

noncomputable def layernormYSpec
    (s : BlockState) (X W : RegionName)
    (stride_x_N stride_x_hn stride_w_hn N BLOCK_SIZE : Nat) (eps : ℝ)
    (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun centered inv => centered * inv)
        (Option.map₂ (fun x mean => x - mean)
          (some (s.readMem X (xOffset s stride_x_N stride_x_hn i)))
          (layernormMeanCarrier s X stride_x_N stride_x_hn N BLOCK_SIZE))
        (layernormInvVarCarrier s X stride_x_N stride_x_hn N BLOCK_SIZE eps))
      (some (s.readMem W (wOffset s stride_w_hn i))))

/-- Algorithm-layer correctness for the one-block layernorm forward slice. -/
theorem layernorm_fwd_triton_one_block_correct
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride_y_N stride_y_hn i))
    (hExec : exec (layernorm_fwd_triton_one_block X W Y
        stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
        N BLOCK_SIZE eps) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Y (yOffset s stride_y_N stride_y_hn i) =
        if i.val < N then
          layernormYSpec s X W stride_x_N stride_x_hn stride_w_hn
            N BLOCK_SIZE eps i
        else s.readMem Y (yOffset s stride_y_N stride_y_hn i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        s.pids 0 * stride_y_N + s.pids 1 * stride_y_hn + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [yOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, layernorm_fwd_triton_one_block, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.sub, NumericDType.div, ComparableDType.lt] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, layernormYSpec, layernormInvVarCarrier, layernormVarCarrier,
            layernormCenteredTile, layernormMeanCarrier, layernormInputTile,
            xOffset, wOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-block layernorm forward slice. -/
theorem layernorm_fwd_triton_one_block_compute_correct
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride_y_N stride_y_hn i)) :
    ComputeCorrect.Realizes
      (kernel := layernorm_fwd_triton_one_block X W Y
        stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
        N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride_y_N stride_y_hn i)))
      (expected := fun i =>
        layernormYSpec s X W stride_x_N stride_x_hn stride_w_hn
          N BLOCK_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layernorm_fwd_triton_one_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layernorm_fwd_triton_one_block_correct X W Y
    stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps
    s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.LayernormFwdTriton
