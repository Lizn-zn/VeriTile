import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `rmsnorm_fused_llama.py`'s `_rms_norm_fwd_fused`.

Allowed mechanical Lean-syntax-only changes:
- Python `Y += row * stride` / `X += row * stride` pointer mutation →
  explicit `Y_base` / `X_base` base-pointer registers.
- Python `.to(tl.float32)` / `.to(tl.float16)` casts are omitted at the
  algorithm layer.
- Python `N` / `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameters. -/
def rms_norm_fwd_fused_llama
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  Y_base = Y + row * $(stride)
  X_base = X + row * $(stride)
  _var = tl.zeros([$(BLOCK_SIZE)])
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X_base + cols, mask=cols < $(N), other=0.0)
    _var = _var + x * x
  }
  var = tl.sum(_var, axis=0) / tl.toReal($(N))
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask)
    x = tl.load(X_base + cols, mask=mask, other=0.0)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y_base + cols, y, mask=mask)
  }
}

def xOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val

def yOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val

noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride idx.1))
      else some (0.0 : ℝ) }

noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map₂ (fun a n => a / n)
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride N BLOCK_SIZE)
        (rmsInputTile s X stride N BLOCK_SIZE))).data PUnit.unit)
    ((Tile.scalar N).natToReal.data PUnit.unit)

noncomputable def rmsInvCarrier
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsVarCarrier s X stride N BLOCK_SIZE)))

noncomputable def rmsnormSpec
    (s : BlockState) (X W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (s.readMem X (xOffset s stride i)))
        (rmsInvCarrier s X stride N BLOCK_SIZE eps))
      (some (s.readMem W i.val)))

/-- Algorithm-layer correctness for the Llama RMSNorm kernel. -/
theorem rms_norm_fwd_fused_llama_correct
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i))
    (hExec : exec (rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps)
        s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Y (yOffset s stride i) =
        if i.val < N then
          rmsnormSpec s X W stride N BLOCK_SIZE eps i
        else s.readMem Y (yOffset s stride i) := by
  sorry

/-- Compute-facing correctness for the Llama RMSNorm kernel. -/
theorem rms_norm_fwd_fused_llama_compute_correct
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    ComputeCorrect.Realizes
      (kernel := rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i => rmsnormSpec s X W stride N BLOCK_SIZE eps i) := by
  sorry

end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama
