# Spec sheet — `bench/tritonbench_g/dequantize_matmul/DequantizeMatmul.lean`

**Python source:** `bench/tritonbench_g/dequantize_matmul/dequantize_matmul.py`

## Public theorem: `dequantize_matmul_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general output summary for `dequantize_matmul.py`.**

For arbitrary matrix dims `K`/`N`, strides `stride_bk`/`stride_bn`/`stride_fpbk`/
`stride_fpbn`, and block sizes `BLOCK_SIZE_N`/`BLOCK_SIZE_K` (and any program ids
in `s`), under the honest row-major output-offset injectivity side condition the
`dequantize_kernel` surface lowers to the algorithm layer and the masked store
into `fp_b` realizes the genuine elementwise dequantize `int_b * scale_b`
(`dequantizeSpec`) on every active lane (`k·BSK + i < K ∧ n·BSN + j < N`),
unchanged otherwise. The four pinned
`dequantize_matmul_python_{128x128,64x256,32x256,256x64}_output_summary` (mirroring
the `@triton.autotune` configs) are concrete instantiations of this one theorem. -/
```
</details>

**Statement:**
```lean
specification dequantize_matmul_output_summary_general
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K)) :
    (∃ alg, (dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
      stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K)`

**Closed-form spec defs (transitive):** `fpbOffset`, `dequantize_kernel`

<details><summary><code>fpbOffset</code></summary>

```lean
def fpbOffset (s : BlockState) (stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_fpbk +
    (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_fpbn
```
</details>

<details><summary><code>dequantize_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `dequantize_matmul.py`'s `dequantize_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_N: tl.constexpr` / `BLOCK_SIZE_K: tl.constexpr` → Lean
  `Nat` parameters. -/
```
```lean
def dequantize_kernel
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = tl.arange(0, $(BLOCK_SIZE_N))
  b_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_bk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_bn)
  fpb_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_fpbk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_fpbn)
  bs_offs = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]
  n_mask = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :] < $(N)
  mask = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None] < $(K)) & n_mask
  int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
  tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `dequantize_kernel_compute_correct`
