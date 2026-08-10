# chunk_gla_fwd

- 源文件:`chunk_gla_fwd.py`(上游 `data/TritonBench_G_v1/chunk_gla_fwd.py`)
- Corpus:TritonBench-G v1
- 规模:368 行,5 个 `@triton.jit` kernel
- 状态:**已移植**(输出 kernel)—— `ChunkGlaFwd.lean`,主定理
  `chunk_gla_fwd_o_exec_genuine`(`exec` 级、维度一般、0 `sorry`)。

五个 kernel 四比一:四个构建块内注意力矩阵 `A`,`chunk_gla_fwd_kernel_o` 消费它
产出输出。**本港覆盖输出 kernel**;四个 `A` 构建者是可信边界(多 kernel 文件的
子集覆盖是既有形态 —— `triton_linear_activation` 和 `kv_cache_filling` 同款)。

`chunk_gla_fwd_kernel_o` 是已移植 `chunk_gla_simple` 的门控版兄弟:`q`/`h`/`v`/`o`
的 block-pointer 布局相同,但门是每个 K 块加载的二维 `[T, K]` 张量,且 `A` 从内存
读入。定理的逐 lane 值:

```
O[i, p] = Σ_{kb < cdiv(K, BK)} Σ_e (scale·q ⊙ exp g)[i, kb·BK+e] · h[kb·BK+e, p]
            + Σ_j tril(A)[i, j] · v[j, p]
```

与兄弟港的头条不同,**没有 `K = BK` 假设** —— K 循环在完整多块一般性下验证。这
之所以负担得起,是因为 kernel 每轮从 `i_k` 重算 block pointer 而不推进它们。全部
维度、stride 和 `scale` 保持符号化;唯一侧条件是标准的行主序输出单射 `hInj`。
