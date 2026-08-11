# chunk_linear_attn

- 源文件:`chunk_linear_attn.py`(上游 `data/TritonBench_G_v1/chunk_linear_attn.py`)
- Corpus:TritonBench-G v1
- 规模:309 行,4 个 `@triton.jit` kernel
- 状态:**已移植**(两个状态递推 kernel)—— `ChunkLinearAttn.lean`,主定理
  `cla_fwd_h_exec_genuine` 与 `cla_bwd_dh_exec_genuine`(`exec` 级、维度一般、
  0 `sorry`)。

四个 kernel 二比二:状态递推对(文件首核 `chunk_linear_attn_fwd_kernel_h` 及其
降序镜像 `bwd_kernel_dh`)与融合输出/梯度对(`fwd_kernel_o`、`bwd_kernel_dqkv`)。
**本港覆盖状态递推对**;另两个是可信边界(多 kernel 文件的子集覆盖是既有形态 ——
`chunk_gla_fwd`、`triton_linear_activation` 同款)。

两个已移植 kernel 共享一套骨架:`[BK, BV]` 运行状态**每 chunk 存一次内存**,再被
一次 `tl.dot` 推进 ——

```
fwd_h:  h[·,·,t] = H_t,   H_0 = (h0 或 0),  H_{t+1} = H_t + k_tᵀ · v_t
        (STORE_FINAL_STATE 时另有 ht = H_NT)
bwd_dh: dh[·,·,t] = D_t,  D_NT = 0,         D_{t-1} = D_t + (scale·q_t)ᵀ · do_t
```

因此定理的逐 lane 值是**逐 chunk 的内存读回**:chunk `t` 的块持有该 chunk 自身
贡献*之前*的状态(正向 = 种子 + 更早的全部 chunk;反向 = 严格更晚的全部 chunk)。
循环不变量携带运行状态、逐 chunk 存储历史条款、以及未触碰区域的 frame;不同
chunk 的存储互不相扰,因为每个 chunk 独占 `h`/`dh` 的一个 `K·V` 块。

`bwd_kernel_dh` 迭代 `for i_t in range(NT - 1, -1, -1)`。surface 用升序拼写同一
迭代序列 —— `for j in range(0, NT)`,循环体第一句 `i_t = NT - 1 - j` —— 即降序
循环的既有拼法(`triton_linear_activation`),换元在 surface 里可观察而非藏进
规约。这是降序 `range` 杠杆的第一个已移植消费者,且**零库改动**:普通
`forRange_inv` 驱动升序计数器即可。

正向 kernel 的一条定理覆盖 `USE_INITIAL_STATE` / `STORE_FINAL_STATE` 全部四种
配置。所有维度、stride、`scale`、chunk 数 `NT` 全部保持符号;副条件为区域互异
加三条布局事实,在启动器的连续状态张量(`h.stride(2) = V`)下均为等式/显然:
`BV ≤ s_h_t`、`(K-1)·s_h_t + V ≤ K·V`、`BV ≤ V`。

算术在 `ℝ` 上(算法层);宿主启动(3-D 网格 `(NK, NV, B·H)` 与宿主计算的
`NT = cdiv(T, BT)`)是可信边界。
