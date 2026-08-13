# parallel_retention_attention

- 源文件:`parallel_retention_attention.py`(上游 `data/TritonBench_G_v1/parallel_retention_attention.py`)
- Corpus:TritonBench-G v1
- 规模:398 行,4 个 `@triton.jit` kernel
- 状态:**已移植**(前向 + 反向 dk/dv helper)——
  `ParallelRetentionAttention.lean`,主定理 `pra_fwd_o_exec_genuine` 与
  `pra_bwd_dkv_exec_genuine`(`exec` 级、维度一般、0 `sorry`)。

上游实现 chunk-parallel *retention* attention:分数是普通点积 `q·k` 乘以
按头指数衰减 `γ^(pos_q − pos_k)`,其中 `γ = 2^b_b`、
`b_b = log2(1 − 2^(−5 − i_h))`、`i_h = i_bh % H`。

- `parallel_retention_fwd_kernel`(被启动的前向 JIT)分两段流式消费 K/V ——
  严格对角线以下的块不加掩码、用 `tl.advance` 推进 block pointer,配衰减
  累加递推 `b_o = b_o · 2^(b_b·BTS) + (Q·K_j ⊙ d_h)·V_j`;然后对角块因果
  掩码、配块内衰减瓦片 `d_s` —— 中间用 `d_q[:, None] = 2^(a·b_b)` 重缩放,
  末尾存 `o`(block pointer,boundary-check)。
- `_parallel_retention_bwd_dkv`(反向 dk/dv helper JIT)以**相反**方向流式
  消费 Q/dO:严格对角线以上的块走**降序 `-BTS` 循环**(降序 range 杠杆
  家族的最后成员 —— 该杠杆就此收官),配镜像衰减递推(`b_dk *= d_b`、
  `b_dv *= d_b`、`b_do` 先乘 `d_q = 2^(c·b_b)`、
  `b_kd = b_k ⊙ 2^((BTL−r)·b_b)`),循环后重缩放
  `b_dk *= d_h[:, None]·scale` / `b_dv *= scale`,然后对角块加掩码。它作为
  标准独立 kernel 验证,标量实参 `i_bh, i_c, i_k, i_v, i_h` 全称量化 ——
  即外壳 `parallel_retention_bwd_kernel` 会传入的值。

头条:`o[a, p]` =
`Σ_{t ≤ i_c·BTL + a} 2^((i_c·BTL + a − t)·b_b) · score(a,t) · v[t,p]`,
其中 `score(a,t) = (scale·q[a]) · k[t]` 取 `BK` 头窗口(逐 program 的因果
retention attention,毛边尾部精确 —— boundary-check 窗口直接烧进 guarded
值函数,**无需**任何对 `T` 的整除假设);`dk`/`dv` = 全查询扫
`Σ_{t ≥ i_c·BTL + r} 2^((t − i_c·BTL − r)·b_b) · scale · (v[r]·do[t]) · q[t]`
/ `… · (k[r]·q[t]) · do[t]`,扫描范围是 kernel 的精确迭代域
`max(cdiv(T,BTS)·BTS, (i_c+1)·BTL)`。唯一圈数假设是宿主自己的
`assert BTL % BTS == 0`。

helper-JIT 结构(DSL 无跨 `@triton.jit` 调用面)、被丢弃的末尾裸
`return`、降序循环换元(`for j in range(0, cdiv(cdiv(T,BTS)·BTS −
(i_c+1)·BTL, BTS))`,体首 `i = cdiv(T,BTS)·BTS − BTS − j·BTS`;python 的
`start − stop` 里 `±BTS` 在 ℤ 精确相消,ℕ 截断减法复现空循环 ——
`parallel_attention` 换元原样照搬)、对角衰减的一元负号下标
`-o_k[:, None] + o_q[None, :]` 改拼为减法 `o_q[None, :] - o_k[:, None]`
(在因果 `tl.where` 掩码保留的每一条 lane 上完全一致)、以及显式
`tl.toReal(...)` int→float 转换(`chunk_retention` 先例),都是登记在
`proof_blockers.md` 的 `Translation-surface blocker:`。反向外壳
`parallel_retention_bwd_kernel` 与 `_parallel_retention_bwd_dq` 是可信边界。
