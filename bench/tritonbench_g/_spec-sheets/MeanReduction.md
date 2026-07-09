# Spec sheet — `bench/tritonbench_g/mean_reduction/MeanReduction.lean`

**Python source:** `bench/tritonbench_g/mean_reduction/mean_reduction.py`

## Public theorem: `mean_dim_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `mean_dim_kernel`: the DSL surface lowers to
the algorithm layer, and the masked store to `Mean` is compute-correct — every
active row lane holds the row mean `meanSpec`, out-of-bounds rows are preserved.
The only side condition is `BLOCK_N ≠ 0`. -/
```
</details>

**Statement:**
```lean
theorem mean_dim_kernel_output_summary
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState)
    (hStepNe : BLOCK_N ≠ 0) :
    (∃ alg, (mean_dim_kernel X Mean M N BLOCK_M BLOCK_N).toAlgorithm? =
        Except.ok alg) ∧
    mean_dim_kernel_correct_target X Mean M N BLOCK_M BLOCK_N s
```

**Assumptions / layout contracts:**
- `hStepNe : BLOCK_N ≠ 0`

**Closed-form spec defs (transitive):** `mean_dim_kernel`, `mean_dim_kernel_correct_target`, `meanOutOffset`, `meanSpec`, `meanInpElem`

<details><summary><code>mean_dim_kernel</code></summary>

```
/-- Faithful transcription of `mean_reduction.py`'s `mean_dim_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `[:, None]` / `[None, :]` dimension annotations preserved.
- Python `BLOCK_M` / `BLOCK_N: tl.constexpr` → Lean `Nat` parameters.

The proof below connects the full-row spec to a loop invariant for the
`for off in range(...)` accumulation.
-/
```
```lean
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
```
</details>

<details><summary><code>mean_dim_kernel_correct_target</code></summary>

```lean
def mean_dim_kernel_correct_target
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState) : Prop :=
  ComputeCorrect.Realizes_without_Rounding
    (kernel := mean_dim_kernel X Mean M N BLOCK_M BLOCK_N)
    (initialState := s)
    (write := ComputeCorrect.WriteMap.writeIf
      (fun i : Fin BLOCK_M => meanOutOffset s BLOCK_M i < M)
      (fun i => (Mean, meanOutOffset s BLOCK_M i)))
    (expected := fun i => meanSpec s X N BLOCK_M i)
```
</details>

<details><summary><code>meanOutOffset</code></summary>

```lean
def meanOutOffset (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>meanSpec</code></summary>

```lean
noncomputable def meanSpec
    (s : BlockState) (X : RegionName) (N BLOCK_M : Nat)
    (i : Fin BLOCK_M) : ℝ :=
  ((Finset.univ : Finset (Fin N)).sum fun j =>
    meanInpElem s X N BLOCK_M i j.val) / (N : ℝ)
```
</details>

<details><summary><code>meanInpElem</code></summary>

```
/-- Input element `X[row, col]` of the row-major `[M, N]` input, at global row
`meanOutOffset s BLOCK_M i = pid0·BLOCK_M + i` (row stride `N`, unit column
stride). -/
```
```lean
noncomputable def meanInpElem
    (s : BlockState) (X : RegionName) (N BLOCK_M : Nat)
    (i : Fin BLOCK_M) (col : Nat) : ℝ :=
  s.readMem X (meanOutOffset s BLOCK_M i * N + col)
```
</details>

## Also present (pinned special-case summaries)
- `mean_dim_kernel_compute_correct`
