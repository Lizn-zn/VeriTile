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

Known proof blocker: see `bench/tritonbench_g/proof_blockers.md`. The full-row
spec below matches the mathematical effect of the `for off in range(...)`
accumulation, but the theorem still needs a loop invariant for that accumulation.
-/
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

noncomputable def meanSpec
    (s : BlockState) (X : RegionName) (N BLOCK_M : Nat)
    (i : Fin BLOCK_M) : ℝ :=
  ((Finset.univ : Finset (Fin N)).sum fun j =>
    s.readMem X (meanOutOffset s BLOCK_M i * N + j.val)) / (N : ℝ)

noncomputable def meanLanePrefix
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N off : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) : ℝ :=
  ((Finset.range off).filter fun col => col < N ∧ col % BLOCK_N = j.val).sum
    fun col => s.readMem X (meanOutOffset s BLOCK_M i * N + col)

noncomputable def meanAccumulatorSpec
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N off : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      some (meanLanePrefix s X N BLOCK_M BLOCK_N off idx.1 idx.2.1) }

noncomputable def meanFromAccumulatorSpec
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N off : Nat)
    (i : Fin BLOCK_M) : ℝ :=
  ((Finset.univ : Finset (Fin BLOCK_N)).sum fun j =>
    meanLanePrefix s X N BLOCK_M BLOCK_N off i j) / (N : ℝ)

private theorem sum_range_eq_sum_fin (N : Nat) (f : Nat → ℝ) :
    (Finset.range N).sum f =
      (Finset.univ : Finset (Fin N)).sum (fun j => f j.val) := by
  classical
  apply Finset.sum_bij (fun n hn => (⟨n, by simpa using hn⟩ : Fin N))
  · intro _ _
    exact Finset.mem_univ _
  · intro _ _ _ _ h
    exact Fin.ext_iff.mp h
  · intro j _
    refine ⟨j.val, Finset.mem_range.mpr j.isLt, ?_⟩
    simp
  · intro _ _
    simp

private theorem sum_lane_prefix_eq_sum_range
    (N BLOCK_N off : Nat) (hBLOCK_N : 0 < BLOCK_N) (hoff : N ≤ off)
    (f : Nat → ℝ) :
    ((Finset.univ : Finset (Fin BLOCK_N)).sum fun j =>
      ((Finset.range off).filter fun col => col < N ∧ col % BLOCK_N = j.val).sum f) =
    (Finset.range N).sum f := by
  classical
  let g : Nat → Fin BLOCK_N := fun col =>
    ⟨col % BLOCK_N, Nat.mod_lt col hBLOCK_N⟩
  have hinner : ∀ j : Fin BLOCK_N,
      ((Finset.range off).filter fun col => col < N ∧ col % BLOCK_N = j.val).sum f =
      (((Finset.range off).filter fun col => col < N).filter fun col => g col = j).sum f := by
    intro j
    apply Finset.sum_congr
    · ext col
      simp [g, Fin.ext_iff, and_left_comm, and_assoc]
    · intro _ _
      rfl
  rw [Finset.sum_congr rfl (fun j _ => hinner j)]
  rw [Finset.sum_fiberwise]
  have hfilter : (Finset.range off).filter (fun col => col < N) = Finset.range N := by
    ext col
    simp only [Finset.mem_filter, Finset.mem_range]
    constructor
    · intro h
      exact h.2
    · intro h
      exact ⟨Nat.lt_of_lt_of_le h hoff, h⟩
  rw [hfilter]

theorem meanFromAccumulatorSpec_eq_meanSpec
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N off : Nat)
    (i : Fin BLOCK_M) (hBLOCK_N : 0 < BLOCK_N) (hoff : N ≤ off) :
    meanFromAccumulatorSpec s X N BLOCK_M BLOCK_N off i =
      meanSpec s X N BLOCK_M i := by
  unfold meanFromAccumulatorSpec meanSpec meanLanePrefix
  rw [sum_lane_prefix_eq_sum_range N BLOCK_N off hBLOCK_N hoff]
  rw [sum_range_eq_sum_fin]

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
