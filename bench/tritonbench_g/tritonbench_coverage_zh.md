# TritonBench-G v1 覆盖分析

本文档记录 [TritonBench-G v1][tb](184 个 GitHub-scraped 真实 Triton
kernel,作为 [TritonBench, ACL 2025 Findings][paper] 的 headline channel
发布)中,有多少比例位于、靠近或位于当前 VeriTile DSL 语义边界之外。

[tb]: https://github.com/thunlp/TritonBench/tree/main/data/TritonBench_G_v1
[paper]: https://arxiv.org/abs/2502.14752

> 2026-05-05 对 clone 时 `main` 上的 TritonBench commit 执行。
> 分类器是 **静态 primitive 匹配** —— 它回答 "这个 kernel 用的 Triton
> 构造是否被 VeriTile 的 DSL 契约覆盖",不是 "proof 可行吗"。许多
> `OK` 判定的 kernel 仍需新 proof 工程。见 §Caveats。

## Headline

| 判定 | 数量 | 占比 |
|---|---:|---:|
| **OK** —— primitive 被当前 DSL 契约覆盖(剥离纯 performance hint 后)| **141** | 77% |
| **Soft** —— DSL/proof 层小扩展(surface adapter 或一类新 primitive)| **15** | 8% |
| **Hard** —— 需要新语义层(concurrency、RNG、低精度 dtype 等)| **28** | 15% |

77% OK 数字假设两个 trivially-buildable 的 adapter 已 land:

1. `tl.math.*` surface adapter —— 所有观察到的调用(`exp2`、`log2`、
   `rsqrt`、`sin`)在 VeriTile 中以不同 spelling 已经有 Real 语义
   (`tl.exp2`、`tl.log2`、`1/tl.sqrt(x)`、`tl.sin`)。
2. `tl.extra.cuda.libdevice.*` surface adapter —— 覆盖 `pow`(重写为
   `exp(b·log(a))`)、`tanh`(已建模)、`llrint`(round + cast 到
   `.int`)。剩下的 `erf`(1 文件)是新的超越 primitive。

不带这些 adapter:121 OK / 31 Soft / 32 Hard。

## 状态更新(2026-07-05)—— 过时 blocker 清扫

下方表格是 2026-05-05 的静态快照存档。其中引用的若干能力缺口此后已在
DSL 落地,第一批曾被 blocker 卡住的 kernel 现已移植**并证明**(逐 kernel
权威状态见 `proof_gap_manifest.tsv`):

- `tl.num_programs`(#92)、`tl.atomic_add` 值级证明、`tl.debug_barrier`
  (在算法层擦除为 no-op)、`tl.math.*` / `tl.extra.cuda.libdevice.*`
  适配器(含精确 `erf`)、通用 `tl.extra.cuda.libdevice.pow`(`Op.pow` /
  `Real.rpow`,标量底 × 张量指数)均已实现 —— 下表中 `num_programs`、
  `atomic_add`、`debug_barrier`、`tl.extra` 的 blocker 标注已过时。
- 2026-07-05 批次移植 + 证明完成(9 个):`relu_strided_buffer`、
  `pow_scalar_tensor`、`fused_layernorm_triton`、`layer_norm_welfold`、
  `bgmv_shrink_kernel`、`rbe_triton_transform`、`triton_linear_activation`、
  `fused_recurrent_delta`、`fused_recurrent_retention`。
- **重新分类**:`isfinite_kernel` 下表列为 Soft/`num_programs`,但其真实
  blocker 是 `isfinited`/`finitef` libdevice 浮点分类内建函数——在精确 ℝ
  语义下退化(ℝ 中一切有限),归属 #447 浮点建模家族,而非任何 surface
  adapter 缺口。
- 截至本次更新仍开放:`int_dot`(#93)、`int4_packed`(#95)、`fp8` 与
  `fp4`(#447/#52 家族)、`RNG`(#41)、`atomic(cas,xchg)` 并发
  (#12/#84)。

## Per-family 判定矩阵

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

## Hard-gap 文件 (28) —— 需要语义,kernel 今天无法可靠建模

按 gap 原因再按文件排序:

### `tl.debug_barrier()` 用作 cross-tile 同步 (8) —— issue #12

强制 program 内 tile 之间的 ordering;在这个语料里是真实的语义原子,
不只是 profiling 标记。在 VeriTile concurrency 边界扩展前都是 Hard。

- `parallel_attention.py` (Attention)
- `parallel_retention_attention.py` (Attention)
- `fused_layernorm_triton.py` (Norm)
- `layer_norm_welfold.py` (Norm)
- `fused_recurrent_delta.py` (Recurrent)
- `fused_recurrent_retention.py` (Recurrent)
- `rms_rbe_matmul.py` (MatMul; also fp8)
- `rbe_triton_transform.py` (RoPE)

### FP8 dtype (7) —— 需要 FP8 dtype channel + dot 语义

观察到的每次使用都是 `.to(tl.float8e5, bitcast=True)` 或
`tl.float8e4nv` accumulator dtype —— 即 kernel 依赖 FP8 的 exponent/mantissa
布局,不只是更窄的存储。

- `attention_llama.py` (Attention)
- `f8_conversion_utils.py` (Conv) —— 纯 FP16↔FP8 转换 utility
- `llama_ff_triton.py` (Elementwise)
- `matmul_persistent_triton.py` (MatMul)
- `rms_matmul_rbe.py` (MatMul)
- `rms_rbe_matmul.py` (MatMul; also debug_barrier)
- `triton_matmul.py` (MatMul)

### RNG primitive (4) —— issue #41

`tl.rand`、`tl.philox`、`tl.uniform_to_normal`、`tl.uint_to_uniform_float`。

- `multinomial_sampling.py` (RNG/Sampling) —— 用 `tl.rand` + `tl.cumsum`
- `seeded_dropout.py` (RNG/Sampling) —— 用 `tl.rand`
- `uniform_sampling.py` (RNG/Sampling) —— 用 `tl.philox`、`tl.uint_to_uniform_float`
- `layer_norm_fwd.py` (Norm) —— fused layernorm + `tl.rand` dropout

### `tl.atomic_cas` / `tl.atomic_xchg` 自旋锁循环 (3) —— issue #12

基于锁的 reduction 与 cross-tile 协调。TritonBench-G v1 中没有
`tl.atomic_max` / `tl.atomic_min` / `tl.atomic_or` / `tl.atomic_and` /
`tl.atomic_xor` —— 这个语料用到的 atomic surface 恰好是
`{ add, cas, xchg }`。

- `streamk_matmul.py` (MatMul; also `tl.atomic_add`)
- `layer_norm_triton.py` (Norm)
- `spinning_lock_reduction.py` (Other)

### Int4 packed unpack 语义 (4)

模式:load int32,然后 `(x >> shift) & 0xF` 提取 4-bit lane,乘以 per-group
scale。VeriTile 有 32-bit `tl.bitcast` 与 `.nat` bitwise op,但还没把
"packed int4 → fp" dequantization 建模为 typed primitive。

- `int4_matmul.py` (MatMul; also `tl.atomic_add`)
- `matmul_dequant_int4.py` (MatMul)
- `matmul_dequantize.py` (MatMul; also `tl.atomic_add`)
- `matmul_dequantize_int4.py` (MatMul)

### FP4 packed unpack 经 `tl.interleave` (2)

`tl.interleave` 未建模,FP4(`e2m1`)的 unpack 规则是格式特定的。

- `fp4_to_bf16.py` (Quant)
- `fp4_to_bf16_conversion.py` (Quant)

### `tl.extra.cuda.libdevice.erf` (1) —— 需要新 primitive

唯一一处 `tl.extra` 使用未被 adapter 覆盖。实现的是精确 GeLU。VeriTile
有 `ApproxGeLU`(tanh 风格);精确 GeLU 需要 `erf` 作为新建模的一元数学 op。

- `triton_linear_activation.py` (Activation)

## Soft-gap 文件 (15) —— DSL/proof 小扩展

### `tl.num_programs(axis)` 未建模 (9) —— DSL 扩展

返回某轴上 grid size 的 Triton primitive。VeriTile 有 `tl.program_id(axis)`
但没有 `tl.num_programs`。多用于 persistent / chunked kernel 计算
per-program 工作份额。

- `relu_strided_buffer.py` (Activation)
- `chunk_linear_attn.py` (Attention)
- `isfinite_kernel.py` (Elementwise)
- `pow_scalar_tensor.py` (Elementwise)
- `bmm_optimized.py` (MatMul)
- `chunk_bwd_dqkg.py` (Recurrent)
- `chunk_gla_fwd.py` (Recurrent)
- `chunk_retention.py` (Recurrent)
- `chunk_retention_ops.py` (Recurrent)

### `tl.atomic_add` 用于 split-K 累加 (6) —— proof shape

VeriTile 有 `Stmt.atomicAdd` 标记、顺序语义与 Real grid-merge sum theorem。
这些 kernel 用 `atomic_add` 合并多个 program 的 partial product —— 需要
把 merge proof 打包进 kernel 的 correctness theorem。

- `bgmv_shrink_kernel.py` (MatMul)
- `int8_dequant_matmul.py` (MatMul; also int_dot)
- `int8_matmul_quantization.py` (MatMul; also int_dot)
- `streamk_matmul.py` (MatMul; also Hard via `atomic_cas/xchg`)
- `int4_matmul.py` (MatMul; also Hard via int4_packed)
- `matmul_dequantize.py` (MatMul; also Hard via int4_packed)

### `tl.dot` 在 int channel (3) —— 扩展 dot 语义

VeriTile 的 `tl.dot` 当前只在 `.real` 上。Int8 × int8 → int32 dot
是良定义的,需要 typed lift。

- `int8_matmul_kernel.py` (MatMul)
- `int8_matmul_quantization.py` (MatMul; also atomic_add)
- `int_scaled_matmul.py` (MatMul)

### Reverse-direction scan (1)

VeriTile 的 `tl.cumsum` / `tl.associative_scan` 只走 forward-axis。

- `reversed_cumsum_scalar.py` (Recurrent)

## Hint-only port-strip 统计(供参考)

这些不是 gap —— 它们是纯 performance/codegen hint,在手工 port 时消失,
不影响功能语义。分类器按文件记录它们,因为它们非常常见(约一半语料),
对 porting workflow 重要:

- `num_stages=…` 关键字:69 文件
- `@triton.autotune`:46 文件
- `tl.max_contiguous` / `tl.multiple_of`:24 文件
- `@triton.heuristics`:19 文件
- `allow_tf32` / `input_precision`:18 文件
- `tl.static_print` / `tl.static_assert`:4 文件

## 补救路线图(按解锁文件数排序)

按产出排序。数字统计至少向上一档判定层(Hard → Soft/OK 或 Soft → OK)
的文件数。

| 行动 | 解锁文件 | 工作量 | Tracking |
|---|---:|---|---|
| `tl.math.*` surface adapter (`exp2/log2/rsqrt/sin`) | 24 | trivial | doc-only,暂无 issue |
| `tl.extra.cuda.libdevice.{pow,tanh,llrint}` surface adapter | 9 | small | doc-only,暂无 issue |
| `tl.num_programs(axis)` AST + 语义 | 9 | small | 新 DSL 扩展 |
| split-K 的 `tl.atomic_add` proof shape | 6 | medium | 扩展 #12 atomic_add theorem |
| Concurrency 边界:`tl.debug_barrier` + `tl.atomic_cas/xchg` | 8 + 3 = 11 | large | #12 |
| FP8 dtype channel + `tl.dot` lift | 7 | large | 需要新 issue |
| Int4 packed unpack 语义 | 4 | medium | 需要新 issue |
| RNG model(`tl.rand`、philox、uniform-to-float)| 4 | large | #41 |
| FP4 packed unpack + `tl.interleave` | 2 | medium | 需要新 issue |
| `tl.dot` 在 int8/int32 | 3 | medium | 扩展 typed dot |
| `tl.extra.cuda.libdevice.erf` primitive | 1 | small | 新一元数学 op |
| Reverse-direction scan | 1 | small | 扩展 scan AST |

如果只是两个 "trivial" adapter 落地,OK 从 121 升到 141(77%)。如果
RNG + concurrency + FP8 + int4 + FP4 全部落地,剩下 28 个 Hard 降到 ≤ 5。

## Caveat —— 本分析 *不* 说什么

1. **静态分析。** 判定基于对 Triton primitive 的 grep。没有 kernel
   实际通过 VeriTile DSL port 过,所以 `OK` 不保证 kernel 能无 surface-syntax
   摩擦地 lower(例如 shape-derived broadcasting 需要句法对齐,或 6D 索引
   公式需要新 TensorView shape)。
2. **`OK` ≠ trivial proof。** 许多 `OK` kernel —— FA-1 backward、RWKV6、
   Mamba SSM、chunked GLA —— 即便 DSL 覆盖其 primitive,仍需新 proof
   工程。`OK` 判定只证明 *可表达性*,不证明 *proof 可行性*。
3. **Decorator hint。** "Hints to strip" 列是参考性的。分类器把它们当
   no-op;实际上 port 时仍必须挑一个 `BLOCK_*` 配置。
4. **Family 分类按文件名。** 例如 `chunk_gated_attention.py` 归在
   Attention 下;gating 逻辑是 Recurrent 风味的递推。跨 family 的 kernel
   按主导操作归类。
5. **一次 commit-time 快照。** TritonBench 的 `main` 可能添加或移动
   kernel。重新跑分类器(脚本未来若加,在 `bench/scripts/`)以刷新。

## Per-file 判定附录

每节列出 family 中每个文件。`Reasons` 列先 Hard 后 Soft 标签;
`Hints to strip` 列纯 performance/codegen hint,在手工 port 时消失。
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
| `batched_vecmat_mult.py` | Soft | surface_blocker: 不支持 tl.broadcast/rank-3 insertion；仅有单行证明切片 | num_stages |
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
