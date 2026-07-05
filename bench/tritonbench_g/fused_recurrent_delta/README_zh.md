# fused_recurrent_delta

- 源文件:`fused_recurrent_delta.py`
- Corpus:TritonBench-G v1
- 状态:DONE —— DSL 移植(`FusedRecurrentDelta.lean`)覆盖**两个** kernel
  (`fused_recurrent_fwd_kernel` + `fused_recurrent_bwd_kernel`),genuine
  独立 delta-rule 递推 spec(`deltaState`),以及 `ComputeCorrect.Realizes`
  汇总定理(`fused_recurrent_delta_output_summary_general`)全部证明完成,
  无 sorry。

前向 kernel 是 flash-linear-attention **delta rule** 递推扫描:每个
`(i_v, i_k, i_bh)` program 每个时间步先读出状态(`v_minus = Σ h·k`),把
delta `v_new = v − v_minus` **原地写回 `v`**,再做 rank-1 状态更新
`h += k ⊗ (β ⊙ v_new)`(`β` 为 per-`(b,h,t)` 标量或 headwise 行,
`IS_HEADWISE_BETA`),并从更新后状态输出 `o = Σ h·q·scale`;另有可选的
`h0` 初始状态与 `ht` 终态存储。反向 kernel 先做逆时间 `d_h` 扫描
(`dk`/`dv`/`dbeta`、可选 `dh0`),经过 `tl.debug_barrier()`,再正向重算
状态完成 `dk` 的原地修正与 `dq`。

delta-rule 转移 `(I − β k kᵀ)` 没有几何衰减闭式,因此 spec 就是按输入
region 递归定义的显式递推 `deltaState`(`reversed_cumsum_scalar`
carry-fold 先例)。两个 kernel 的每个存储输出都是维度通用主定理的一个
clause —— 前向各 face 对应 `deltaState` / `vNewClosed` / `outputClosed`,
反向各 face 对应基于物化 carried-state 缓冲的 genuine 单步梯度公式。
跨步折叠(`h` / `d_h` 的携带)是文档化的受信循环边界
(`fused_recurrent_hgrn` / `fused_rwkv6_kernel` 先例),`v` 原地覆写的
影响在模块 docstring 中明确说明。诚实侧条件:`BK ≤ K`、`BV ≤ V`。
