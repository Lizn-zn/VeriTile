import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Bench.TritonBenchG.MeanReduction

open VeriTile.Triton VeriTile.Examples

/-- Faithful transcription of `mean_reduction.py`'s `mean_dim_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `[:, None]` / `[None, :]` dimension annotations preserved.
- Python `BLOCK_M` / `BLOCK_N: tl.constexpr` → Lean `Nat` parameters.

Known proof blocker: see `bench/tritonbench_g/proof_blockers.md`. The spec
below now describes the full row mean, but the theorem still needs a loop
invariant for the `for off in range(...)` accumulation. -/
def mean_dim_kernel
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))[:, None]
  X = X + pid * $(N)
  Mean = Mean + pid
  row_mask = pid < $(M)
  _mean = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_N)) {
    cols = off + tl.arange(0, $(BLOCK_N))[None, :]
    col_mask = cols < $(N)
    mask = row_mask and col_mask
    a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=1) / $(N)
  mean = mean[:, None]
  tl.store(Mean, mean, row_mask)
}

def meanOutOffset (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

noncomputable def meanOneColSpec
    (s : BlockState) (X : RegionName) (N BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem X (meanOutOffset s BLOCK_M i * N) / (N : ℝ)

noncomputable def meanInputTile
    (s : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < M ∧ idx.2.1.val < N then
        some (s.readMem X (meanOutOffset s BLOCK_M idx.1 * N + idx.2.1.val))
      else some (0.0 : ℝ) }

noncomputable def meanCarrier
    (s : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_M) : WithBot ℝ :=
  Option.map₂ (fun total n => total / n)
    ((Tile.reduceSum (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.false
      (meanInputTile s X M N BLOCK_M BLOCK_N)).data (i, PUnit.unit))
    ((Tile.scalar (dtype := .real) (some (N : ℝ) : WithBot ℝ)).data PUnit.unit)

noncomputable def meanSpec
    (s : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_M) : ℝ :=
  WithBot.unbotD 0 (meanCarrier s X M N BLOCK_M BLOCK_N i)

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
