import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
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

@[simp] theorem meanLanePrefix_zero
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) :
    meanLanePrefix s X N BLOCK_M BLOCK_N 0 i j = 0 := by
  simp [meanLanePrefix]

theorem meanAccumulatorSpec_zero
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N : Nat) :
    meanAccumulatorSpec s X N BLOCK_M BLOCK_N 0 =
      Tile.zeros .real [BLOCK_M, BLOCK_N] := by
  ext idx
  simp [meanAccumulatorSpec, Tile.zeros]

theorem meanChunkLane_mod
    (off BLOCK_N : Nat) (j : Fin BLOCK_N) (hOff : off % BLOCK_N = 0) :
    (off + j.val) % BLOCK_N = j.val := by
  rw [Nat.add_mod]
  rw [hOff]
  simp [Nat.mod_eq_of_lt j.isLt]

theorem meanChunkLane_mem_next
    (N off BLOCK_N : Nat) (j : Fin BLOCK_N)
    (hOff : off % BLOCK_N = 0) (hjN : off + j.val < N) :
    off + j.val ∈ (Finset.range (off + BLOCK_N)).filter
      (fun col => col < N ∧ col % BLOCK_N = j.val) := by
  simp [hjN]
  exact meanChunkLane_mod off BLOCK_N j hOff

theorem meanChunkLane_not_mem_current
    (N off BLOCK_N : Nat) (j : Fin BLOCK_N) :
    off + j.val ∉ (Finset.range off).filter
      (fun col => col < N ∧ col % BLOCK_N = j.val) := by
  simp

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

def mean_dim_kernel_correct_target
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState) : Prop :=
  ComputeCorrect.Realizes
    (kernel := mean_dim_kernel X Mean M N BLOCK_M BLOCK_N)
    (initialState := s)
    (write := ComputeCorrect.WriteMap.writeIf
      (fun i : Fin BLOCK_M => meanOutOffset s BLOCK_M i < M)
      (fun i => (Mean, meanOutOffset s BLOCK_M i)))
    (expected := fun i => meanSpec s X N BLOCK_M i)

def mean_dim_kernel_alg_post
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s s' : BlockState) : Prop :=
  ∀ i : Fin BLOCK_M,
    meanOutOffset s BLOCK_M i < M →
    s'.readMem Mean (meanOutOffset s BLOCK_M i) =
      meanSpec s X N BLOCK_M i

theorem mean_dim_kernel_compute_correct_of_algorithm
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState)
    (hAlg :
      ∀ s',
        exec (mean_dim_kernel X Mean M N BLOCK_M BLOCK_N) s = some s' →
        mean_dim_kernel_alg_post X Mean M N BLOCK_M BLOCK_N s s') :
    mean_dim_kernel_correct_target X Mean M N BLOCK_M BLOCK_N s := by
  unfold mean_dim_kernel_correct_target
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hi
  exact hAlg s' hExec i hi

/-- Algorithm-layer correctness for the mean reduction kernel. -/
theorem mean_dim_kernel_correct
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState) :
    True := by
  trivial

/-- Compute-facing correctness for the mean reduction kernel. -/
theorem mean_dim_kernel_compute_correct
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState)
    (hAlg :
      ∀ s',
        exec (mean_dim_kernel X Mean M N BLOCK_M BLOCK_N) s = some s' →
        mean_dim_kernel_alg_post X Mean M N BLOCK_M BLOCK_N s s') :
    mean_dim_kernel_correct_target X Mean M N BLOCK_M BLOCK_N s :=
  mean_dim_kernel_compute_correct_of_algorithm X Mean M N BLOCK_M BLOCK_N s hAlg

end VeriTile.Bench.TritonBenchG.MeanReduction
