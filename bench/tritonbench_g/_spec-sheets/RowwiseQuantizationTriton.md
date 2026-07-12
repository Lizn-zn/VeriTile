# Spec sheet — `bench/tritonbench_g/rowwise_quantization_triton/RowwiseQuantizationTriton.lean`

**Python source:** `bench/tritonbench_g/rowwise_quantization_triton/rowwise_quantization_triton.py`

## Public theorem: `quantize_rowwise_blocked_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general blocked output summary.** For arbitrary `_n_elements`,
`BLOCK_SIZE`, padded width `P2`, and real scale `scale127` (and any program id in
`s`), the faithful full surface is recorded as **blocked** at algorithm erasure
(it stores CUDA `llrint`/int8 results, not the real-valued pre-rounding
expression), while the checked proof slices realize the genuine pre-rounding
scaled row output `scale127 * (x / max_val)` (`quantizeRowwiseScaledSpec`) at
every in-range lane and the per-row `output_maxs` writeback
(`quantizeRowwiseMaxSpec`). This holds over arbitrary (symbolic) dimensions and
`scale127`. The `max(|x|)` reduction is
taken as the precomputed `MaxVals` input; the `llrint` rounding / int8 cast
remain the honest, unmodeled blocker. -/
```
</details>

**Statement:**
```lean
specification quantize_rowwise_blocked_output_summary_general
    (x_ptr output_ptr MaxVals output_maxs : RegionName)
    (_n_elements BLOCK_SIZE P2 : Nat) (scale127 : ℝ) (s : BlockState) :
    (∃ err,
      (quantize_rowwise_real_surface x_ptr output_ptr output_maxs
        _n_elements BLOCK_SIZE P2).toAlgorithm? = Except.error err) ∧
    ((ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Closed-form spec defs (transitive):** `quantize_rowwise_real_surface`

<details><summary><code>quantize_rowwise_real_surface</code></summary>

```
/-- Real-valued surface of `rowwise_quantization_triton.py`'s
`_quantize_rowwise`.

This preserves row addressing, `tl.abs`, masked max reduction, the `output_maxs`
store, CUDA `llrint` surface operation, and the scaled output expression. The
algorithm carrier records the pre-cast real value. -/
```
```lean
def quantize_rowwise_real_surface
    (x_ptr output_ptr output_maxs : RegionName)
    (_n_elements BLOCK_SIZE P2 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  abs_x = tl.abs(x)
  max_val = tl.max(tl.where(row_mask, abs_x, 0.0), axis=0)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x / max_val))
  tl.store(output_ptr + offsets, output, mask=row_mask)
  tl.store(output_maxs + pid, max_val)
}
```
</details>

## Also present (pinned special-case summaries)
- `quantize_rowwise_scaled_store_slice_compute_correct`
- `quantize_rowwise_max_store_slice_compute_correct`
