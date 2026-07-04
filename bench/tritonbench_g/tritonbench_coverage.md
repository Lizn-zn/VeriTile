# TritonBench-G v1 Coverage Analysis

This document records what fraction of [TritonBench-G v1][tb] (the 184
GitHub-scraped real Triton kernels released as the headline channel of
[TritonBench, ACL 2025 Findings][paper]) sits inside, near, or outside the
current VeriTile DSL semantic boundary.

[tb]: https://github.com/thunlp/TritonBench/tree/main/data/TritonBench_G_v1
[paper]: https://arxiv.org/abs/2502.14752

> Performed 2026-05-05 against TritonBench commit on `main` at clone time.
> The classifier is **static primitive matching only** — it answers "does this
> kernel use Triton constructs that VeriTile's DSL contract covers?", not
> "is the proof feasible". Many `OK`-verdict kernels still need fresh proof
> engineering. See §Caveats.

## Headline

| Verdict | Count | Share |
|---|---:|---:|
| **OK** — primitives covered by current DSL contract (after stripping pure performance hints) | **141** | 77% |
| **Soft** — small extension to current DSL/proof layer (surface adapter or one new primitive class) | **15** | 8% |
| **Hard** — new semantic layer required (concurrency, RNG, low-precision dtype, etc.) | **28** | 15% |

The 77% OK figure is **after** assuming two trivially-buildable adapters land:

1. `tl.math.*` surface adapter — every observed call (`exp2`, `log2`, `rsqrt`,
   `sin`) already has Real semantics in VeriTile under different spellings
   (`tl.exp2`, `tl.log2`, `1/tl.sqrt(x)`, `tl.sin`).
2. `tl.extra.cuda.libdevice.*` surface adapter — covers `pow` (rewrite to
   `exp(b·log(a))`), `tanh` (already modeled), `llrint` (round + cast to
   `.int`). The remaining `erf` (1 file) is a new transcendental primitive.

Without those adapters: 121 OK / 31 Soft / 32 Hard.

## Per-family verdict matrix

| Family | Total | OK | Soft | Hard |
|---|---:|---:|---:|---:|
| Softmax | 12 | 12 | 0 | 0 |
| Indirect (KV cache, gather, embedding) | 12 | 12 | 0 | 0 |
| Loss (cross-entropy, KL) | 7 | 7 | 0 | 0 |
| RoPE | 10 | 9 | 0 | 1 |
| Attention | 29 | 25 | 1 | 3 |
| Norm | 17 | 13 | 0 | 4 |
| Quant | 10 | 8 | 0 | 2 |
| Activation | 8 | 6 | 1 | 1 |
| Recurrent / SSM / RWKV / chunked | 18 | 11 | 5 | 2 |
| Elementwise | 20 | 17 | 2 | 1 |
| MatMul | 31 | 16 | 6 | 9 |
| Conv | 2 | 1 | 0 | 1 |
| RNG / Sampling | 5 | 2 | 0 | 3 |
| Other | 3 | 2 | 0 | 1 |
| **Total** | **184** | **141** | **15** | **28** |

## Hard-gap files (28) — semantics required, kernel cannot be soundly modeled today

Sorted by gap reason, then file:

### `tl.debug_barrier()` used as cross-tile synchronization (8) — issue #12

Forces ordering between tiles within a program; a real semantic atom in this
corpus, not just a profiling marker. Hard until VeriTile's concurrency
boundary expands.

- `parallel_attention.py` (Attention)
- `parallel_retention_attention.py` (Attention)
- `fused_layernorm_triton.py` (Norm)
- `layer_norm_welfold.py` (Norm)
- `fused_recurrent_delta.py` (Recurrent)
- `fused_recurrent_retention.py` (Recurrent)
- `rms_rbe_matmul.py` (MatMul; also fp8)
- `rbe_triton_transform.py` (RoPE)

### FP8 dtype (7) — needs FP8 dtype channel + dot semantics

Every observed use is `.to(tl.float8e5, bitcast=True)` or
`tl.float8e4nv` accumulator dtype — i.e., the kernel relies on FP8's
exponent/mantissa layout, not just narrower storage.

- `attention_llama.py` (Attention)
- `f8_conversion_utils.py` (Conv) — purely an FP16↔FP8 conversion utility
- `llama_ff_triton.py` (Elementwise)
- `matmul_persistent_triton.py` (MatMul)
- `rms_matmul_rbe.py` (MatMul)
- `rms_rbe_matmul.py` (MatMul; also debug_barrier)
- `triton_matmul.py` (MatMul)

### RNG primitives (4) — issue #41

`tl.rand`, `tl.philox`, `tl.uniform_to_normal`, `tl.uint_to_uniform_float`.

- `multinomial_sampling.py` (RNG/Sampling) — uses `tl.rand` + `tl.cumsum`
- `seeded_dropout.py` (RNG/Sampling) — uses `tl.rand`
- `uniform_sampling.py` (RNG/Sampling) — uses `tl.philox`, `tl.uint_to_uniform_float`
- `layer_norm_fwd.py` (Norm) — fused layernorm + `tl.rand` dropout

### `tl.atomic_cas` / `tl.atomic_xchg` spinlock loops (3) — issue #12

Lock-based reductions and cross-tile coordination. There is no
`tl.atomic_max` / `tl.atomic_min` / `tl.atomic_or` / `tl.atomic_and` /
`tl.atomic_xor` in TritonBench-G v1 — the atomics surface used in this corpus
is exactly `{ add, cas, xchg }`.

- `streamk_matmul.py` (MatMul; also `tl.atomic_add`)
- `layer_norm_triton.py` (Norm)
- `spinning_lock_reduction.py` (Other)

### Int4 packed unpack semantics (4)

Pattern: load int32, then `(x >> shift) & 0xF` to extract a 4-bit lane,
multiply by per-group scale. VeriTile has 32-bit `tl.bitcast` and `.nat`
bitwise ops, but does not yet model "packed int4 → fp" dequantization as a
typed primitive.

- `int4_matmul.py` (MatMul; also `tl.atomic_add`)
- `matmul_dequant_int4.py` (MatMul)
- `matmul_dequantize.py` (MatMul; also `tl.atomic_add`)
- `matmul_dequantize_int4.py` (MatMul)

### FP4 packed unpack via `tl.interleave` (2)

`tl.interleave` is not modeled, and the FP4 (`e2m1`) unpack rule is
specific to the format.

- `fp4_to_bf16.py` (Quant)
- `fp4_to_bf16_conversion.py` (Quant)

### `tl.extra.cuda.libdevice.erf` (1) — needs new primitive

Only one observed `tl.extra` use that is not covered by the adapter. Implements
exact GeLU. VeriTile has `ApproxGeLU` (tanh-style); exact GeLU needs `erf` as
a new modeled unary math op.

- `triton_linear_activation.py` (Activation)

## Soft-gap files (15) — small DSL/proof extension

### `tl.num_programs(axis)` not modeled (9) — DSL extension

Triton primitive returning the grid size along an axis. VeriTile has
`tl.program_id(axis)` but not `tl.num_programs`. Mostly used in persistent /
chunked kernels to compute per-program work share.

- `relu_strided_buffer.py` (Activation)
- `chunk_linear_attn.py` (Attention)
- `isfinite_kernel.py` (Elementwise)
- `pow_scalar_tensor.py` (Elementwise)
- `bmm_optimized.py` (MatMul)
- `chunk_bwd_dqkg.py` (Recurrent)
- `chunk_gla_fwd.py` (Recurrent)
- `chunk_retention.py` (Recurrent)
- `chunk_retention_ops.py` (Recurrent)

### `tl.atomic_add` for split-K accumulation (6) — proof shape

VeriTile has the `Stmt.atomicAdd` marker, sequential semantics, and a Real
grid-merge sum theorem. These kernels use `atomic_add` to merge partial
products from multiple programs — needs the merge proof packaged in the
kernel's correctness theorem.

- `bgmv_shrink_kernel.py` (MatMul)
- `int8_dequant_matmul.py` (MatMul; also int_dot)
- `int8_matmul_quantization.py` (MatMul; also int_dot)
- `streamk_matmul.py` (MatMul; also Hard via `atomic_cas/xchg`)
- `int4_matmul.py` (MatMul; also Hard via int4_packed)
- `matmul_dequantize.py` (MatMul; also Hard via int4_packed)

### `tl.dot` over int channels (3) — extend dot semantics

VeriTile's `tl.dot` is currently over `.real` only. Int8 × int8 → int32 dot
is well-defined and would need a typed lift.

- `int8_matmul_kernel.py` (MatMul)
- `int8_matmul_quantization.py` (MatMul; also atomic_add)
- `int_scaled_matmul.py` (MatMul)

### Reverse-direction scan (1)

RESOLVED (#94): `tl.cumsum` / `tl.cumprod` / `tl.associative_scan` now accept
`reverse=True/False`, lowering to the directed scan node (`ScanDirection`,
suffix fold). The kernel below is unblocked but not yet ported.

- `reversed_cumsum_scalar.py` (Recurrent)

## Hint-only port-strip tally (informational)

These are not gaps — they are pure performance/codegen hints that disappear at
hand-port time and do not affect functional semantics. The classifier records
them per file because they are very common (≈half the corpus) and matter for
the porting workflow:

- `num_stages=…` keyword: 69 files
- `@triton.autotune`: 46 files
- `tl.max_contiguous` / `tl.multiple_of`: 24 files
- `@triton.heuristics`: 19 files
- `allow_tf32` / `input_precision`: 18 files
- `tl.static_print` / `tl.static_assert`: 4 files

## Remediation roadmap (by # of files unlocked)

Sorted by yield. Numbers count files moving up at least one verdict tier
(Hard → Soft/OK or Soft → OK).

| Action | Files unlocked | Effort | Tracking |
|---|---:|---|---|
| `tl.math.*` surface adapter (`exp2/log2/rsqrt/sin`) | 24 | trivial | doc-only, no issue yet |
| `tl.extra.cuda.libdevice.{pow,tanh,llrint}` surface adapter | 9 | small | doc-only, no issue yet |
| `tl.num_programs(axis)` AST + semantics | 9 | small | new DSL extension |
| `tl.atomic_add` proof shape for split-K | 6 | medium | extends #12 atomic_add theorem |
| Concurrency boundary: `tl.debug_barrier` + `tl.atomic_cas/xchg` | 8 + 3 = 11 | large | #12 |
| FP8 dtype channel + `tl.dot` lift | 7 | large | needs new issue |
| Int4 packed unpack semantics | 4 | medium | needs new issue |
| RNG model (`tl.rand`, philox, uniform-to-float) | 4 | large | #41 |
| FP4 packed unpack + `tl.interleave` | 2 | medium | needs new issue |
| `tl.dot` over int8/int32 | 3 | medium | extends typed dot |
| `tl.extra.cuda.libdevice.erf` primitive | 1 | small | new unary math op |
| Reverse-direction scan | 1 | small | extends scan AST |

If only the two "trivial" adapters land, OK rises 121 → 141 (77%). If RNG +
concurrency + FP8 + int4 + FP4 all land, the remaining 28 Hard go to ≤ 5.

## Caveats — what this analysis does NOT say

1. **Static-only.** The verdict is based on grep of Triton primitives. No
   kernel has actually been ported through VeriTile's DSL, so `OK` does not
   guarantee the kernel will lower without surface-syntax friction (e.g.,
   shape-derived broadcasting that needs syntactic alignment, or 6D index
   formulas requiring new TensorView shapes).
2. **`OK` ≠ trivial proof.** Many `OK` kernels — FA-1 backward, RWKV6, Mamba
   SSM, chunked GLA — need fresh proof engineering even though the DSL covers
   their primitives. The `OK` verdict only attests to *expressibility*,
   not to *proof feasibility*.
3. **Decorator hints.** The "Hints to strip" column is informational. The
   classifier treats them as no-ops; in reality you must still pick one
   `BLOCK_*` configuration during the port.
4. **Family classification is by filename.** `chunk_gated_attention.py` for
   instance is filed under Attention; the gating logic is a Recurrent-flavored
   recurrence. Cross-family kernels are placed by their dominant operation.
5. **One commit-time snapshot.** TritonBench's `main` may add or move kernels.
   Re-run the classifier (script in `bench/scripts/` if added later) to
   refresh.

## Per-file verdict appendix

Each section lists every file in the family. `Reasons` lists Hard then Soft
labels; `Hints to strip` lists pure performance/codegen hints that disappear
at hand-port time.
### Activation (8)

| File | Verdict | Reasons | Hints to strip |
|---|---|---|---|
| `triton_linear_activation.py` | **Hard** | tl.extra | autotune, heuristics, num_stages, contig_hint |
| `relu_strided_buffer.py` | Soft | num_programs | — |
| `fused_activation.py` | OK | — | — |
| `geglu_tanh_triton.py` | OK | — | — |
| `relu_triton_kernel.py` | OK | — | — |
| `swiglu_backward.py` | OK | — | autotune, heuristics |
| `swiglu_fwd.py` | OK | — | autotune |
| `swiglu_triton.py` | OK | — | — |

### Attention (29)

| File | Verdict | Reasons | Hints to strip |
|---|---|---|---|
| `attention_llama.py` | **Hard** | fp8 | num_stages |
| `parallel_attention.py` | **Hard** | debug_barrier | num_stages, tf32_hint |
| `parallel_retention_attention.py` | **Hard** | debug_barrier | num_stages, tf32_hint |
| `chunk_linear_attn.py` | Soft | num_programs | num_stages, tf32_hint |
| `attention_forward_triton.py` | OK | — | num_stages, contig_hint |
| `attention_fwd_triton1.py` | OK | — | num_stages, tf32_hint |
| `attention_fwd_triton2.py` | OK | — | num_stages, contig_hint |
| `attention_fwd_triton3.py` | OK | — | heuristics, num_stages, contig_hint |
| `attention_kernel.py` | OK | — | num_stages |
| `attention_kernel_aligned.py` | OK | — | num_stages |
| `attention_score.py` | OK | — | heuristics, contig_hint |
| `attn_fwd_causal.py` | OK | — | num_stages, contig_hint |
| `attn_fwd_triton.py` | OK | — | num_stages, contig_hint |
| `block_sparse_attn.py` | OK | — | static_print |
| `chunk_gated_attention.py` | OK | — | autotune, num_stages, tf32_hint |
| `context_attn_bloom.py` | OK | — | num_stages, contig_hint |
| `context_attn_fwd.py` | OK | — | num_stages, contig_hint |
| `context_attn_llama.py` | OK | — | num_stages, contig_hint |
| `context_attn_mistral.py` | OK | — | num_stages, contig_hint |
| `context_attn_nopad.py` | OK | — | num_stages, contig_hint |
| `flash_attn.py` | OK | — | num_stages |
| `flash_decode2_llama.py` | OK | — | num_stages |
| `flash_decode2_phi.py` | OK | — | num_stages |
| `lightning_attention.py` | OK | — | — |
| `mixed_sparse_attention.py` | OK | — | num_stages |
| `token_attn_llama2.py` | OK | — | num_stages |
| `token_attn_mistral.py` | OK | — | num_stages, contig_hint |
| `token_attn_reduceV.py` | OK | — | num_stages, contig_hint |
| `triton_attention.py` | OK | — | num_stages |

### Conv (2)

| File | Verdict | Reasons | Hints to strip |
|---|---|---|---|
| `f8_conversion_utils.py` | **Hard** | fp8 | — |
| `triton_conv2d_fwd.py` | OK | — | tf32_hint |

### Elementwise (20)

| File | Verdict | Reasons | Hints to strip |
|---|---|---|---|
| `llama_ff_triton.py` | **Hard** | fp8 | num_stages |
| `isfinite_kernel.py` | Soft | num_programs | — |
| `pow_scalar_tensor.py` | Soft | num_programs | — |
| `add_example.py` | OK | — | — |
| `add_value.py` | OK | — | — |
| `cosine_compute.py` | OK | — | — |
| `fifth_order_sph_harmonics.py` | OK | — | — |
| `masked_add_cuda.py` | OK | — | — |
| `matrix_reduction.py` | OK | — | — |
| `matrix_transpose.py` | OK | — | — |
| `max_reduction.py` | OK | — | autotune, heuristics |
| `mean_reduction.py` | OK | — | — |
| `mul_exponent_compensator.py` | OK | — | — |
| `nested_loops_processing.py` | OK | — | — |
| `sin_computation.py` | OK | — | — |
| `sin_kernel.py` | OK | — | — |
| `square_matrix.py` | OK | — | — |
| `triton_mul2.py` | OK | — | — |
| `vector_addition.py` | OK | — | — |
| `vector_addition_custom.py` | OK | — | — |

### Indirect (12)

| File | Verdict | Reasons | Hints to strip |
|---|---|---|---|
| `cache_transform_triton.py` | OK | — | — |
| `destindex_copy.py` | OK | — | num_stages |
| `destindex_copy_kv1.py` | OK | — | num_stages |
| `destindex_copy_kv2.py` | OK | — | num_stages |
| `embedding_triton_kernel.py` | OK | — | num_stages, contig_hint |
| `index_select_bwd.py` | OK | — | — |
| `index_select_cat.py` | OK | — | — |
| `kcache_copy_triton.py` | OK | — | — |
| `kv_cache_copy.py` | OK | — | — |
| `kv_cache_filling.py` | OK | — | num_stages |
| `masked_select.py` | OK | — | autotune |
| `var_len_copy.py` | OK | — | — |

### Loss (7)

| File | Verdict | Reasons | Hints to strip |
|---|---|---|---|
| `cross_entropy1.py` | OK | — | heuristics |
| `cross_entropy2.py` | OK | — | — |
| `cross_entropy_ops.py` | OK | — | heuristics |
| `fast_ce_loss.py` | OK | — | heuristics |
| `kldiv_compute.py` | OK | — | — |
| `kldiv_ops.py` | OK | — | — |
| `kldiv_triton.py` | OK | — | — |

### MatMul (31)

| File | Verdict | Reasons | Hints to strip |
|---|---|---|---|
| `int4_matmul.py` | **Hard** | int4_packed, atomic_add | autotune, num_stages, static_print |
| `matmul_dequant_int4.py` | **Hard** | int4_packed | autotune, num_stages |
| `matmul_dequantize.py` | **Hard** | int4_packed, atomic_add | autotune, num_stages |
| `matmul_dequantize_int4.py` | **Hard** | int4_packed | autotune, num_stages |
| `matmul_persistent_triton.py` | **Hard** | fp8 | num_stages, contig_hint |
| `rms_matmul_rbe.py` | **Hard** | fp8 | num_stages |
| `rms_rbe_matmul.py` | **Hard** | debug_barrier, fp8 | num_stages |
| `streamk_matmul.py` | **Hard** | atomic(cas,xchg), atomic_add | num_stages |
| `triton_matmul.py` | **Hard** | fp8 | num_stages, contig_hint |
| `bgmv_shrink_kernel.py` | Soft | atomic_add | contig_hint |
| `bmm_optimized.py` | Soft | num_programs | autotune, heuristics, num_stages, tf32_hint |
| `int8_dequant_matmul.py` | Soft | atomic_add | autotune, heuristics, num_stages, contig_hint |
| `int8_matmul_kernel.py` | Soft | int_dot | autotune, num_stages, static_print |
| `int8_matmul_quantization.py` | Soft | atomic_add, int_dot | autotune, num_stages |
| `int_scaled_matmul.py` | Soft | int_dot | num_stages, tf32_hint, contig_hint |
| `batched_vecmat_mult.py` | Soft | surface_blocker: tl.broadcast/rank-3 insertion unsupported; one-row proof slice only | num_stages |
| `bgmv_expand_slice.py` | OK | — | — |
| `bmm_chunk_bwd.py` | OK | — | autotune, num_stages |
| `bmm_chunk_fwd.py` | OK | — | autotune, num_stages |
| `dequantize_matmul.py` | OK | — | autotune, num_stages |
| `iv_dependent_matmul.py` | OK | — | num_stages |
| `lora_expand_gemv.py` | OK | — | contig_hint |
| `matmul_kernel.py` | OK | — | — |
| `matmul_leakyrelu.py` | OK | — | — |
| `matmul_leakyrelu_fp8.py` | OK | — | autotune, num_stages |
| `matmul_tma.py` | OK | — | — |
| `matmul_triton1.py` | OK | — | tf32_hint |
| `matmul_triton2.py` | OK | — | autotune, num_stages |
| `matmul_triton_autotune.py` | OK | — | autotune, num_stages |
| `matrix_vector_multip.py` | OK | — | autotune, num_stages |
| `sgmv_expand_slice.py` | OK | — | contig_hint |

### Norm (17)

| File | Verdict | Reasons | Hints to strip |
|---|---|---|---|
| `fused_layernorm_triton.py` | **Hard** | debug_barrier | autotune, num_stages |
| `layer_norm_fwd.py` | **Hard** | RNG | autotune, heuristics |
| `layer_norm_triton.py` | **Hard** | atomic(cas,xchg) | — |
| `layer_norm_welfold.py` | **Hard** | debug_barrier | autotune, num_stages |
| `fast_layernorm.py` | OK | — | — |
| `fast_rms_layernorm.py` | OK | — | heuristics |
| `l2_norm_bwd.py` | OK | — | — |
| `l2_norm_triton1.py` | OK | — | — |
| `l2_norm_triton2.py` | OK | — | — |
| `layer_norm_liger.py` | OK | — | — |
| `layer_norm_ops.py` | OK | — | autotune, heuristics |
| `layernorm_fwd_triton.py` | OK | — | — |
| `rms_norm_triton.py` | OK | — | — |
| `rmsnorm_fused.py` | OK | — | — |
| `rmsnorm_fused_llama.py` | OK | — | — |
| `rmsnorm_implementation.py` | OK | — | — |
| `rmsnorm_triton.py` | OK | — | — |

### Other (3)

| File | Verdict | Reasons | Hints to strip |
|---|---|---|---|
| `spinning_lock_reduction.py` | **Hard** | atomic(cas,xchg) | — |
| `adam_update_triton.py` | OK | — | autotune |
| `triton_argmax.py` | OK | — | — |

### Quant (10)

| File | Verdict | Reasons | Hints to strip |
|---|---|---|---|
| `fp4_to_bf16.py` | **Hard** | fp4 | — |
| `fp4_to_bf16_conversion.py` | **Hard** | fp4 | autotune |
| `dequantize_rowwise.py` | OK | — | — |
| `int8_quantization.py` | OK | — | — |
| `quant_transpose_kernel.py` | OK | — | autotune |
| `quantize_copy_kv.py` | OK | — | num_stages |
| `quantize_global.py` | OK | — | autotune, num_stages |
| `quantize_kv_copy.py` | OK | — | num_stages |
| `quantize_kv_transform.py` | OK | — | num_stages |
| `rowwise_quantization_triton.py` | OK | — | autotune, num_stages |

### RNG/Sampling (5)

| File | Verdict | Reasons | Hints to strip |
|---|---|---|---|
| `multinomial_sampling.py` | **Hard** | RNG | — |
| `seeded_dropout.py` | **Hard** | RNG | — |
| `uniform_sampling.py` | **Hard** | RNG | heuristics, static_print |
| `apply_penalty.py` | OK | — | — |
| `dropout_triton.py` | OK | — | — |

### Recurrent (18)

| File | Verdict | Reasons | Hints to strip |
|---|---|---|---|
| `fused_recurrent_delta.py` | **Hard** | debug_barrier | num_stages |
| `fused_recurrent_retention.py` | **Hard** | debug_barrier | num_stages |
| `chunk_bwd_dqkg.py` | Soft | num_programs | autotune, tf32_hint |
| `chunk_gla_fwd.py` | Soft | num_programs | autotune, tf32_hint, contig_hint |
| `chunk_retention.py` | Soft | num_programs | num_stages, tf32_hint |
| `chunk_retention_ops.py` | Soft | num_programs | autotune, tf32_hint |
| `reversed_cumsum_scalar.py` | Soft | reverse_scan | autotune |
| `chunk_cumsum_kernel.py` | OK | — | autotune |
| `chunk_cumsum_vector.py` | OK | — | autotune, tf32_hint |
| `chunk_delta_fwd.py` | OK | — | autotune, tf32_hint |
| `chunk_gate_recurrence.py` | OK | — | tf32_hint |
| `chunk_gla_simple.py` | OK | — | autotune, tf32_hint |
| `chunked_cumsum_fwd.py` | OK | — | autotune |
| `decay_cumsum.py` | OK | — | — |
| `diag_ssm_triton.py` | OK | — | autotune |
| `fused_recurrent_hgrn.py` | OK | — | autotune |
| `fused_rwkv6_kernel.py` | OK | — | num_stages |
| `reversed_cumsum.py` | OK | — | autotune, tf32_hint |

### RoPE (10)

| File | Verdict | Reasons | Hints to strip |
|---|---|---|---|
| `rbe_triton_transform.py` | **Hard** | debug_barrier | — |
| `fast_rope_embedding.py` | OK | — | — |
| `fused_rotary_embedding.py` | OK | — | — |
| `rope_backward_transform.py` | OK | — | — |
| `rope_embedding.py` | OK | — | heuristics |
| `rope_transform.py` | OK | — | — |
| `rotary_emb.py` | OK | — | num_stages |
| `rotary_emb_nopad.py` | OK | — | — |
| `rotary_transform.py` | OK | — | — |
| `rotary_transform_ops.py` | OK | — | — |

### Softmax (12)

| File | Verdict | Reasons | Hints to strip |
|---|---|---|---|
| `ksoftmax_triton.py` | OK | — | autotune, heuristics |
| `log_softmax.py` | OK | — | autotune, heuristics |
| `logsumexp_fwd.py` | OK | — | autotune, heuristics |
| `softmax_flaggems.py` | OK | — | autotune, heuristics |
| `softmax_optimize.py` | OK | — | — |
| `softmax_reducev.py` | OK | — | num_stages, contig_hint |
| `softmax_triton1.py` | OK | — | — |
| `softmax_triton2.py` | OK | — | — |
| `softmax_triton3.py` | OK | — | — |
| `token_softmax_bloom.py` | OK | — | — |
| `token_softmax_llama.py` | OK | — | — |
| `triton_softmax.py` | OK | — | — |
