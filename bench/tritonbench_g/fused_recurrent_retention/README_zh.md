# fused_recurrent_retention

- 源文件:`fused_recurrent_retention.py`
- Corpus:TritonBench-G v1
- 状态:DONE —— DSL 移植(`FusedRecurrentRetention.lean`)、genuine 衰减外积
  闭式 spec、`ComputeCorrect.Realizes` 汇总定理
  (`fused_recurrent_retention_output_summary_general`)全部证明完成,无
  sorry,覆盖**两个**被启动的 kernel(前向 + autograd 反向)。

`fused_recurrent_retention_fwd_kernel` 是 flash-linear-attention 的 fused
recurrent retention 扫描:每个 program `(i_v, i_k, i_bh)` 在 `0..T` 循环中
携带 `[BV, BK]` 状态 `h`,以标量 per-head 衰减 `b_b = 1 − 2^(−5 − i_h)`
(`tl.math.exp2`)更新 `h = b_b·h + k_t ⊗ v_t`,并从**更新后**状态输出
`o_t = Σ_k h·(scale·q_t)`;支持可选的 `initial_state` 种子与 `final_state`
存储。反向 kernel(经 `loss.backward()` 触发)先做前向时间的 `dq` 扫描,
在逐字转写的 `tl.debug_barrier()` 之后,再做反向时间的 `dk`/`dv` 扫描,
携带梯度状态 `d_h += scale·q_t ⊗ do_t; d_h *= b_b`,指针以 `-=` 递减。

两个完整 surface 均可下降到 algorithm 层。genuine 闭式为
`stateClosed(m) = seed·b_b^m + Σ_{t<m} k_t⊗v_t·b_b^(m−1−t)`(其 `outClosed`
归约即经典 retention 形式 `o_t = Σ_{s≤t} b_b^(t−s)·(scale·q_t·k_s)·v_s`,见
`outClosed_as_decayed_dots`),以及
`dStateClosed(t) = Σ_{t≤u<T} scale·q_u⊗do_u·b_b^(u−t)`,`dq`/`dk`/`dv` 为其
归约 —— 全部只引用输入 region。跨步 `range(0, T)` 折叠是受信边界:每个循环
体 face 在自传播的进位不变式(`HPrev = stateClosed(m)` /
`DHPrev = b_b·dStateClosed(m+1)`)下对闭式实现,打包方式沿用
`fused_rwkv6_kernel` / `fused_recurrent_hgrn`。step slice 以非掩码方式读取
每个时间行(in-bounds 情形;partial tail-block 的 `other=0` 车道与宿主端
`o.sum(0)`/`dq.sum(0)`/`dk.sum(0)`/`dv.sum(0)` 跨 block 归约为文档化受信
边界)。
