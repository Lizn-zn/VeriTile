# Spec sheet — `bench/tritonbench_g/max_reduction/MaxReduction.lean`

**Python source:** `bench/tritonbench_g/max_reduction/max_reduction.py`

## Public theorem: `max_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for the 2D value/index `max_kernel`: the DSL
surface lowers to the algorithm layer, and the masked value/index stores are
compute-correct — every active row lane `i` holds the row-wise maximum
(`maxKernelValueSpec`) in `out_value` and its argmax (`maxKernelIndexSpec`) in
`out_index`. Carries the existing side conditions of
`max_kernel_compute_correct`: `hOutInj` (injective output offsets) and
`hOutRegions` (`out_value ≠ out_index`). -/
```
</details>

**Statement:**
```lean
theorem max_kernel_output_summary
    (inp out_value out_index : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => maxKernelOutOffset s K BLOCK_M i))
    (hOutRegions : out_value ≠ out_index) :
    (∃ alg, (max_kernel inp out_value out_index M N K BLOCK_M BLOCK_N).toAlgorithm?
        = Except.ok alg) ∧
    ComputeCorrect.OutputPairWhere
      (max_kernel inp out_value out_index M N K BLOCK_M BLOCK_N)
      s out_value out_index
      (maxKernelOutOffset s K BLOCK_M)
      (fun i : Fin BLOCK_M => s.pids 0 * BLOCK_M + i.val < M)
      (maxKernelValueSpec s inp M N K BLOCK_M BLOCK_N)
      (maxKernelIndexSpec s inp M N K BLOCK_M BLOCK_N)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => maxKernelOutOffset s K BLOCK_M i)`
- `hOutRegions : out_value ≠ out_index`
- `fun i : Fin BLOCK_M => s.pids 0 * BLOCK_M + i.val < M`

**Closed-form spec defs (transitive):** `maxKernelOutOffset`, `max_kernel`, `maxKernelValueSpec`, `maxKernelIndexSpec`, `maxKernelInputTile`

<details><summary><code>maxKernelOutOffset</code></summary>

```
/-- Output offset for the 2D value/index max kernel at row lane `i`. -/
```
```lean
def maxKernelOutOffset (s : BlockState) (K BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  (s.pids 0 * BLOCK_M + i.val) * K + s.pids 1
```
</details>

<details><summary><code>max_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel`
(autotuned, returns value + index via `tl.max(..., return_indices=True)`).

Allowed mechanical Lean-syntax-only changes apply. The `@triton.autotune`
and `@triton.heuristics` decorators that wrap `max_kernel` are not DSL-side
constructs (they configure the launch grid, not the kernel body); they have
no in-kernel transcription. -/
```
```lean
def max_kernel
    (inp out_value out_index : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  offset_index = m_offset * $(K) + pid_k
  mask1 = m_offset < $(M)
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  inp_ptrs = inp + offset
  inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  result_value, result_index := tl.max(inp_vals, axis=1, return_indices=True)
  out_value_ptrs = out_value + offset_index
  out_index_ptrs = out_index + offset_index
  tl.store(out_value_ptrs, result_value, mask=mask1)
  tl.store(out_index_ptrs, result_index, mask=mask1)
}
```
</details>

<details><summary><code>maxKernelValueSpec</code></summary>

```
/-- Exact max value written by `max_kernel` at row lane `i`. -/
```
```lean
noncomputable def maxKernelValueSpec
    (s : BlockState) (inp : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (i : Fin BLOCK_M) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.false
      (maxKernelInputTile s inp M N K BLOCK_M BLOCK_N) with
  | some out => WithBot.unbotD 0 (out.data (i, PUnit.unit))
  | none => 0
```
</details>

<details><summary><code>maxKernelIndexSpec</code></summary>

```
/-- Exact argmax index written by `max_kernel` at row lane `i`. -/
```
```lean
noncomputable def maxKernelIndexSpec
    (s : BlockState) (inp : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (i : Fin BLOCK_M) : Nat :=
  (Tile.argMaxDrop (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩
    (maxKernelInputTile s inp M N K BLOCK_M BLOCK_N)).data (i, PUnit.unit)
```
</details>

<details><summary><code>maxKernelInputTile</code></summary>

```
/-- Input tile for the 2D value/index max kernel. Masked lanes are `⊥`,
matching `other=-float("inf")`. -/
```
```lean
noncomputable def maxKernelInputTile
    (s : BlockState) (inp : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      let m := s.pids 0 * BLOCK_M + idx.1.val
      let n := idx.2.1.val
      if m < M ∧ n < N then
        some (s.readMem inp (m * N * K + n * K + s.pids 1))
      else none }
```
</details>

## Also present (pinned special-case summaries)
- `max_kernel_1_compute_correct`
- `max_kernel_2_compute_correct`
- `max_kernel_compute_correct`
