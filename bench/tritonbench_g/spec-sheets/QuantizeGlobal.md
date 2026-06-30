# Spec sheet — `bench/tritonbench_g/quantize_global/QuantizeGlobal.lean`

**Python source:** `bench/tritonbench_g/quantize_global/quantize_global.py`

## Public theorem: `quantize_global_blocked_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general blocked output summary.** For arbitrary `n_elements`,
`BLOCK_SIZE`, and real scale `scale127` (and any program id in `s`), the faithful
full surface is recorded as **blocked** at algorithm projection (it stores CUDA
`llrint`/int8 results, not the real-valued pre-rounding expression), while the
checked store slice realizes the genuine pre-rounding quantity
`scale127 * (x * absmax_inv)` (`quantizeGlobalScaledSpec`) at every in-range lane,
leaving out-of-range lanes unchanged. Concrete Python benchmark shapes are
instantiations of this (with `scale127 = 127.0`). No injectivity hypothesis is
needed: the 1-D block offset map is injective by construction. The `llrint`
rounding / int8 cast remain the honest, unmodeled blocker. -/
```
</details>

**Statement:**
```lean
theorem quantize_global_blocked_output_summary_general
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) (s : BlockState) :
    (∃ err, (quantize_global_surface x_ptr absmax_inv_ptr output_ptr
      n_elements BLOCK_SIZE).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes
      (kernel
```

**Closed-form spec defs (transitive):** `quantize_global_surface`

<details><summary><code>quantize_global_surface</code></summary>

```
/-- Faithful transcription of `quantize_global.py`'s `_quantize_global`.

The CUDA `llrint` operation is preserved as a surface operation; the algorithm
carrier records the pre-cast real value. -/
```
```lean
def quantize_global_surface
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `quantize_global_scaled_store_slice_compute_correct`
