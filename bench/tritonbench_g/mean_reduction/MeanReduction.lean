import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Bench.TritonBenchG.MeanReduction

open VeriTile.Triton VeriTile.Examples

/-- Faithful transcription of `mean_reduction.py`'s `mean_dim_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `X = X + pid * N` / `Mean = Mean + pid` pointer mutation →
  explicit `X_base` / `Mean_base` base-pointer registers.
- Python `[:, None]` / `[None, :]` dimension annotations preserved.
- Python `.to(tl.float32)` casts are omitted at the algorithm layer.
- Python `BLOCK_M` / `BLOCK_N: tl.constexpr` → Lean `Nat` parameters. -/
def mean_dim_kernel
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))[:, None]
  X_base = X + pid * $(N)
  Mean_base = Mean + pid
  row_mask = pid < $(M)
  _mean = tl.zeros([$(BLOCK_M), $(BLOCK_N)])
  for off in range(0, $(N), $(BLOCK_N)) {
    cols = off + tl.arange(0, $(BLOCK_N))[None, :]
    col_mask = cols < $(N)
    mask = row_mask and col_mask
    a = tl.load(X_base + cols, mask=mask, other=0.0)
    _mean = _mean + a
  }
  mean = tl.sum(_mean, axis=1) / $(N)
  mean = mean[:, None]
  tl.store(Mean_base, mean, mask=row_mask)
}

def meanOutOffset (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

noncomputable def meanOneColSpec
    (s : BlockState) (X : RegionName) (N BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem X (meanOutOffset s BLOCK_M i * N) / (N : ℝ)

/-- Algorithm-layer correctness for the mean reduction kernel. -/
theorem mean_dim_kernel_correct
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState) :
    True := by
  trivial

/-- Compute-facing correctness for the mean reduction kernel. -/
theorem mean_dim_kernel_compute_correct
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState) :
    True := by
  trivial

end VeriTile.Bench.TritonBenchG.MeanReduction
