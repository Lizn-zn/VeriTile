# Spec sheet — `bench/tritonbench_g/quant_transpose_kernel/QuantTransposeKernel.lean`

**Python source:** `bench/tritonbench_g/quant_transpose_kernel/quant_transpose_kernel.py`

## Public theorem: `quantize_global_transpose_blocked_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general blocked output summary.** For arbitrary strides, sizes
`M`/`N`, block sizes `BLOCK_M`/`BLOCK_N`, group factor `GROUP_M`, and real scale
`scale127` (and any program coordinates in `s`), the faithful full surface is
recorded as **blocked** at algorithm erasure (it stores CUDA `llrint`/int8
results, not the real-valued pre-rounding expression), while the checked
scaled-store slice realizes the genuine pre-rounding quantity
`scale127 * (a * absmax_inv)` (`quantTransposeScaledSpec`) at every in-range tile
lane, leaving out-of-range lanes unchanged. This holds over arbitrary (symbolic)
dimensions. Output-address injectivity for the transposed writeback is taken as a
hypothesis (`hOutInj`). The `llrint` rounding / int8 cast remain the honest,
unmodeled blocker. -/
```
</details>

**Statement:**
```lean
theorem quantize_global_transpose_blocked_output_summary_general
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M : Nat)
    (scale127 : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx)) :
    (∃ err, (quantize_global_transpose_real_surface A AbsmaxInv B
      stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N
      GROUP_M).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes
      (kernel
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx)`

**Closed-form spec defs (transitive):** `bOffset`, `quantize_global_transpose_real_surface`, `rowIndex`, `colIndex`

<details><summary><code>bOffset</code></summary>

```lean
def bOffset
    (s : BlockState) (stride_bm stride_bn BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  rowIndex s BLOCK_M idx.1 * stride_bm + colIndex s BLOCK_N idx.2.1 * stride_bn
```
</details>

<details><summary><code>quantize_global_transpose_real_surface</code></summary>

```
/-- Real-valued surface of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

This preserves the grouped one-dimensional program-id schedule, masked load,
global scale, CUDA `llrint` surface operation, transposed store addressing, and
masked writeback. The algorithm carrier records the pre-cast real value. -/
```
```lean
def quantize_global_transpose_real_surface
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  grid_m = ($((M : Nat)) + $((BLOCK_M : Nat)) - $((1 : Nat))) // $((BLOCK_M : Nat))
  grid_n = ($((N : Nat)) + $((BLOCK_N : Nat)) - $((1 : Nat))) // $((BLOCK_N : Nat))
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // group_size
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  A = A + (rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  a = tl.load(A, mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  B = B + (rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))
  tl.store(B, output, mask=mask)
}
```
</details>

<details><summary><code>rowIndex</code></summary>

```lean
def rowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (s : BlockState) (BLOCK_N : Nat) (j : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + j.val
```
</details>

## Also present (pinned special-case summaries)
- `quantize_global_transpose_scaled_store_slice_compute_correct`
