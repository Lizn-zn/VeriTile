import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.LayernormFwdTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `layernorm_fwd_triton.py`'s
`_layer_norm_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameters.
- The Python `stride_x_hd`, `stride_y_hd`, and `stride_w_hd` parameters are not
  used by the source kernel body, so this translation omits them. -/
def layernorm_fwd_triton
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn : Nat)
    (N BLOCK_SIZE : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  Seq = tl.program_id(0)
  H = tl.program_id(1)
  X += Seq * $(stride_x_N) + H * $(stride_x_hn)
  Y += Seq * $(stride_y_N) + H * $(stride_y_hn)
  W += H * $(stride_w_hn)
  _mean = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    a = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=0) / $(N)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x - mean, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = (x - mean) * rstd
    y = x_hat * w
    tl.store(Y + cols, (y).to(X.dtype.element_ty), mask=mask)
  }
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
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (layernormInputTile s X stride_x_N stride_x_hn N BLOCK_SIZE)).data
        PUnit.unit)

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
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (layernormCenteredTile s X stride_x_N stride_x_hn N BLOCK_SIZE)
        (layernormCenteredTile s X stride_x_N stride_x_hn N BLOCK_SIZE))).data
        PUnit.unit)

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
theorem layernorm_fwd_triton_correct
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride_y_N stride_y_hn i))
    (hExec : exec (layernorm_fwd_triton X W Y
        stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
        N BLOCK_SIZE eps) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Y (yOffset s stride_y_N stride_y_hn i) =
        if i.val < N then
          layernormYSpec s X W stride_x_N stride_x_hn stride_w_hn
            N BLOCK_SIZE eps i
        else s.readMem Y (yOffset s stride_y_N stride_y_hn i) := by
  sorry
/-- Compute-facing correctness for the one-block layernorm forward slice. -/
theorem layernorm_fwd_triton_compute_correct
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride_y_N stride_y_hn i)) :
    ComputeCorrect.Realizes
      (kernel := layernorm_fwd_triton X W Y
        stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
        N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride_y_N stride_y_hn i)))
      (expected := fun i =>
        layernormYSpec s X W stride_x_N stride_x_hn stride_w_hn
          N BLOCK_SIZE eps i) := by
  sorry
end VeriTile.Bench.TritonBenchG.LayernormFwdTriton
