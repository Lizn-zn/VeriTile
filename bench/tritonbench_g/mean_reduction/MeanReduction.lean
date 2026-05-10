import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Bench.TritonBenchG.MeanReduction

open VeriTile.Triton VeriTile.Examples

/-- Proof-oriented `BLOCK_N = 1` slice of `mean_reduction.py`'s
`mean_dim_kernel`.

The upstream kernel computes a row mean by reducing a `[BLOCK_M, BLOCK_N]`
tile. This slice covers the degenerate one-column reduction used to establish
the row mask, pointer arithmetic, division by `N`, and output write contract
without introducing a loop invariant for the full `for off in range(0, N,
BLOCK_N)` body. -/
def mean_dim_kernel_one_col
    (X Mean : RegionName)
    (M N BLOCK_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  mask = pid < $(M)
  x = tl.load(X + pid * $(N), mask=mask, other=0.0)
  mean = x / $(N)
  tl.store(Mean + pid, mean, mask=mask)
}

def meanOutOffset (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

noncomputable def meanOneColSpec
    (s : BlockState) (X : RegionName) (N BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem X (meanOutOffset s BLOCK_M i * N) / (N : ℝ)

/-- Algorithm-layer correctness for the one-column mean slice. -/
theorem mean_dim_kernel_one_col_correct
    (X Mean : RegionName)
    (M N BLOCK_M : Nat) (_hBlockM : 0 < BLOCK_M)
    (s : BlockState) :
    ∀ i : Fin BLOCK_M,
      let addr := meanOutOffset s BLOCK_M i
      observeAt (exec (mean_dim_kernel_one_col X Mean M N BLOCK_M) s)
          Mean BLOCK_M (s.pids 0) i =
        some (if addr < M then meanOneColSpec s X N BLOCK_M i
              else s.readMem Mean addr) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] => s.pids 0 * BLOCK_M + idx.1.val) := by
    intro a b h
    rcases a with ⟨a, _⟩
    rcases b with ⟨b, _⟩
    simp at h ⊢
    exact Fin.ext h
  simp [observeAt, exec, mean_dim_kernel_one_col, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop,
        NumericDType.add, NumericDType.mul, NumericDType.div, ComparableDType.lt]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BLOCK_M + i.val < M
  · simp [hi, meanOneColSpec, meanOutOffset]
  · simp [hi, meanOutOffset]

/-- Compute-facing correctness for the one-column mean slice. -/
theorem mean_dim_kernel_one_col_compute_correct
    (X Mean : RegionName)
    (M N BLOCK_M : Nat) (hBlockM : 0 < BLOCK_M)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := mean_dim_kernel_one_col X Mean M N BLOCK_M)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_M => meanOutOffset s BLOCK_M i < M)
        (fun i => (Mean, meanOutOffset s BLOCK_M i)))
      (expected := fun i => meanOneColSpec s X N BLOCK_M i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := mean_dim_kernel_one_col_correct X Mean M N BLOCK_M hBlockM s i
  rw [hExec] at h
  have hActive' : meanOutOffset s BLOCK_M i < M := by
    simpa using hActive
  simpa [observeAt, hActive'] using h

end VeriTile.Bench.TritonBenchG.MeanReduction
