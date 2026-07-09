# Spec sheet — `bench/tritonbench_g/dequantize_rowwise/DequantizeRowwise.lean`

**Python source:** `bench/tritonbench_g/dequantize_rowwise/dequantize_rowwise.py`

## Public theorem: `dequantize_rowwise_output_summary_general`

<details><summary>docstring</summary>

```
/-- **General public summary for `dequantize_rowwise`.** Dimension-general over
`n_elements BLOCK_SIZE P2` and the scale `inv_127`: the kernel lowers, and every
active lane `i < BLOCK_SIZE` of the output realizes the genuine per-row
dequantized value `dequantizeRowwiseSpec` (`x[i] * state_x[row] * inv_127`),
NOT a self-referential read-back. The Python cases (`(2,4)`, `(N,16)`, `(N,8)`,
`(N,32)`) are special cases. -/
```
</details>

**Statement:**
```lean
theorem dequantize_rowwise_output_summary_general
    (x_ptr state_x output_ptr : RegionName) (inv_127 : ℝ)
    (n_elements BLOCK_SIZE P2 : Nat) (s : BlockState) :
    (∃ alg, (dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
      n_elements BLOCK_SIZE P2).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Closed-form spec defs (transitive):** `dequantize_rowwise_kernel`

<details><summary><code>dequantize_rowwise_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `dequantize_rowwise.py`'s `_dequantize_rowwise`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` / `P2: tl.constexpr` → Lean `Nat`
  parameters.
- `n_elements` is kept as `_n_elements`: the upstream Triton kernel accepts it
  but does not use it in the body. -/
```
```lean
def dequantize_rowwise_kernel
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (_n_elements BLOCK_SIZE P2 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(state_x + pid)
  output = max_val * x * $(inv_127)
  tl.store(output_ptr + offsets, output, mask=row_mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `dequantize_rowwise_kernel_compute_correct`
