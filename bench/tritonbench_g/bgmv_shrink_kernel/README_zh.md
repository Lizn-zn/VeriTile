# bgmv_shrink_kernel

- 源文件:`bgmv_shrink_kernel.py`
- Corpus:TritonBench-G v1
- 状态:DONE —— DSL 移植(`BgmvShrinkKernel.lean`)、genuine 的缩放 rank-slice
  收缩 spec,以及 `ComputeCorrect.Realizes` 总结
  (`bgmv_shrink_kernel_output_summary_general`)已 sorry-free 证明。

`_bgmv_shrink_kernel` 是 batched LoRA *shrink* GEMV:program
`(pid_sk, cur_batch)` 通过 `lora_indices[cur_batch]` 选择 LoRA-A 矩阵
(有符号 `-1` 哨兵 = 早退),沿 rank 维 `K` 的本 program 切片
(步长 `BLOCK_K·SPLIT_K`,带 mask 的尾块加载)累加
`tl.sum(tiled_a * tiled_b, 1)`,乘以 `scaling`,然后按 `SPLIT_K == 1`
分支做带 mask 的 store,否则 `tl.atomic_add` 写入 `out[cur_batch, 0:N]`。

这是首个走通 `tl.atomic_add` 执行路径的 bench 移植:`SPLIT_K > 1` 分支按
per-program 的读-改-写义务证明
`out-after = out-before + scaling·Σ_slice input·loraA`(跨 program 对
`pid_sk` 求和是 host 侧受信组合),依赖本文件新增的 masked atomic-scatter
读回引理(`foldl_atomicAdd_at` / `foldl_atomicAdd_preserve`)。constexpr 的
`SPLIT_K == 1` store-vs-atomic 尾分支拆成两个证明 surface(`matmul_tma`
先例);哨兵早退在忠实 surface 中表示为 guard,且 `-1` 路径已证不写内存
(`bgmv_shrink_sentinel_skip_no_write`)。主定理对 `N`、`K`(带 mask 尾块,
无整除假设)、`BLOCK_N`、`BLOCK_K`、`SPLIT_K`、全部 stride、`scaling`、
program id 及数据相关的 `lora_index` 全部一般化;side condition 为
`0 < BLOCK_K`、`0 < SPLIT_K`、`0 < cn_stride`。
