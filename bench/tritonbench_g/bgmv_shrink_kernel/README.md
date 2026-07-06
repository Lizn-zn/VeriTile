# bgmv_shrink_kernel

- Source file: `bgmv_shrink_kernel.py`
- Corpus: TritonBench-G v1
- Status: DONE — DSL port (`BgmvShrinkKernel.lean`), genuine scaled rank-slice
  contraction spec, and `ComputeCorrect.Realizes` summary
  (`bgmv_shrink_kernel_output_summary_general`) proven sorry-free.

`_bgmv_shrink_kernel` is the batched LoRA *shrink* GEMV: program
`(pid_sk, cur_batch)` selects the LoRA-A matrix via `lora_indices[cur_batch]`
(signed `-1` sentinel = early return), accumulates
`tl.sum(tiled_a * tiled_b, 1)` over its `pid_sk`-slice of the rank dimension
`K` (stride `BLOCK_K·SPLIT_K`, masked tail loads), scales by `scaling`, and
either masked-stores (`SPLIT_K == 1`) or `tl.atomic_add`s (`SPLIT_K > 1`) the
`BLOCK_N`-vector into `out[cur_batch, 0:N]`.

First bench port to exercise the `tl.atomic_add` exec path: the `SPLIT_K > 1`
branch is proven as the per-program read-modify-write obligation
`out-after = out-before + scaling·Σ_slice input·loraA` (the cross-program sum
over `pid_sk` is the host's trusted composition), via new in-file masked
atomic-scatter readback lemmas (`foldl_atomicAdd_at` /
`foldl_atomicAdd_preserve`). The constexpr `SPLIT_K == 1` store-vs-atomic tail
is split into two proof surfaces (`matmul_tma` precedent); the sentinel early
return is a guard in the faithful surface, with the `-1` path proven
write-free (`bgmv_shrink_sentinel_skip_no_write`). The headline is general
over `N`, `K` (masked tail — no divisibility assumption), `BLOCK_N`,
`BLOCK_K`, `SPLIT_K`, all strides, `scaling`, program ids, and the
data-dependent `lora_index`; side conditions are `0 < BLOCK_K`, `0 < SPLIT_K`,
`0 < cn_stride`.
