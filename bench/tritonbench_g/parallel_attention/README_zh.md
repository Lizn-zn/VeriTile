# parallel_attention

- 源文件:`parallel_attention.py`(上游 `data/TritonBench_G_v1/parallel_attention.py`)
- Corpus:TritonBench-G v1
- 规模:481 行,4 个 `@triton.jit` kernel
- 状态:**已移植**(前向 + 反向 dk/dv helper)—— `ParallelAttention.lean`,主定理
  `pa_fwd_o_exec_genuine` 与 `pa_bwd_dkv_exec_genuine`(`exec` 级、维度一般、
  0 `sorry`)。

上游实现 *rebased*(二次核)attention:分数是 `(q·k)²` 而非 `exp(q·k)`,因此
因果 attention 分解为未归一化的值和 `o` 加单独存储的归一化子 `z`(宿主在核外做
`o / (z + eps)`)。

- `parallel_rebased_fwd_kernel`(被启动的前向 JIT)分两段流式消费 K/V ——
  严格对角线以下的块不加掩码、用 `tl.advance` 推进 block pointer,对角块因果
  掩码 —— 末尾同时存 `o`(block pointer,boundary-check)与 `z`(平指针,掩码)。
- `_parallel_rebased_bwd_dkv`(反向 dk/dv helper JIT)以**相反**方向流式消费
  Q/dO:严格对角线以上的块走**降序 `-BTS` 循环**(降序 range 杠杆家族的最后
  成员),然后对角块加掩码,循环内有 `i_v == 0` 运行时门把 `dz` 归一化子梯度
  折进来。它作为标准独立 kernel 验证,标量实参 `i_bh, i_c, i_k, i_v, i_h` 全称
  量化 —— 即外壳 `parallel_rebased_bwd_kernel` 会传入的值。

头条:`o[a, p]` = `Σ_{t ≤ i_c·BTL + a} score(a,t)² · v[t,p]`,`z[a]` 为其归一
化子(逐 program 的因果 rebased attention,毛边尾部精确 —— boundary-check 窗口
直接烧进 guarded 值函数,**无需**任何对 `T` 的整除假设);`dk`/`dv` = 全键扫
`Σ_t keep·(2·ds·s·q)` / `Σ_t keep·s²·do`,门项 `[i_v = 0]·dz[t]` 在 `ds` 里。
唯一圈数假设是宿主自己的 `assert BTL % BTS == 0`。

helper-JIT 结构(DSL 无跨 `@triton.jit` 调用面)、被丢弃的末尾裸 `return`、
以及降序循环换元(`for j in range(0, cdiv(cdiv(T,BTS)·BTS − (i_c+1)·BTL,
BTS))`,体首 `i = cdiv(T,BTS)·BTS − BTS − j·BTS`;python 的 `start − stop` 里
`±BTS` 在 ℤ 精确相消,ℕ 截断减法复现空循环)是登记在 `proof_blockers.md` 的
`Translation-surface blocker:`。反向外壳 `parallel_rebased_bwd_kernel` 与
`_parallel_rebased_bwd_dq` 是可信边界。
