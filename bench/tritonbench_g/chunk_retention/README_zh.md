# chunk_retention

- 源文件:`chunk_retention.py`(上游 `data/TritonBench_G_v1/chunk_retention.py`)
- Corpus:TritonBench-G v1
- 规模:451 行,4 个 `@triton.jit` kernel
- 状态:**已移植**(两个状态递推 kernel)—— `ChunkRetention.lean`,主定理
  `crh_fwd_h_exec_genuine` 与 `crh_bwd_dh_exec_genuine`(`exec` 级、0 `sorry`)。

已移植 `chunk_linear_attn` 对的 retention(带衰减)兄弟:四个 kernel 二比二,
**本港覆盖状态递推对**(文件首核 `fwd_kernel_h` + `bwd_kernel_dh`);
`fwd_kernel_o`/`bwd_kernel_dqkv` 是可信边界。

`fwd_kernel_h` 跑衰减递推

```
h[·,·,t] = H_t,   H_0 = (h0 或 0),   H_{t+1} = d_b(t)·H_t + k_tᵀ·(v_t ⊙ d_i(t))
```

每头衰减 `b_b = log2(1 - 2^(-5 - i_h))`,每 chunk 因子 `2^(len(t)·b_b)`,其中
`len(t)` 在**毛边末 chunk** 上是 `T % BT` —— kernel 在循环内该 chunk 处重绑
`d_b`/`d_i`,证明的循环不变量把寄存器条款写成条件式(`c < NT →` 标准值),
一条定理同时覆盖毛边与整除、以及全部四种门配置。规约 `crhState` 是**递归的**
(不是幂闭式):毛边 chunk 让最后一步有自己的衰减长度,递归使逐 chunk 推进
在定义上成立。全部符号化(维度、stride、`H`、`NT`);布局副条件与
`chunk_linear_attn` 相同的三条。

三条忠实性披露(全在 `.lean` 序言):

1. 衰减前奏的隐式 int→float 提升拼作 `tl.toReal(...)` —— 已注册的
   `Translation-surface blocker:`(`rbe_triton_transform` 先例)。
2. 毛边 chunk 上游的 `(T % BT) - o_i - 1` 在整数瓦片上取负;`.nat` 通道把这些
   lane 截断为 `0`。该分歧对每个店不可观察 —— 这些 lane 乘的是超出 `T` 的
   `v` 行,块指针边界检查已将其归零。
3. **`bwd_kernel_dh` 只在方形可编译**(`BK = BT = BV`,由其 `tl.dot` 形状强制),
   循环无店且含死 `dh` 载入,末尾单店复用循环后的 `i_t`(Python 留 `0`;
   surface 预初始化 `i_t = 0`,因 DSL 对循环体名作用域化)。头条证明的正是
   这个 kernel 实际的计算 —— `dh` 块 = 打乱收缩下的
   `d_b · Σ_{全部 chunk} (do ⊙ d_i)·v` —— 怪相如实披露,不抹平。

`bwd_kernel_dh` 降序(`range(NT-1, -1, -1)`);surface 用升序换元拼写
(`chunk_linear_attn` 拼法,零库改动)。算术在 `ℝ` 上;宿主启动是可信边界。
