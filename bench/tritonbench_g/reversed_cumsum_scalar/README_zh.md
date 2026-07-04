# reversed_cumsum_scalar

- 源文件:`reversed_cumsum_scalar.py`(pinned 上游 `603e28a`)
- Corpus:TritonBench-G v1
- 状态:DONE —— DSL 移植(`ReversedCumsumScalar.lean`)、genuine 后缀和闭式
  spec、`ComputeCorrect.Realizes` 汇总定理
  (`reversed_cumsum_scalar_output_summary_general`)全部证明完成,无 sorry。

`chunk_global_reversed_cumsum_scalar_kernel` 按 chunk **从最后一个 chunk 向前**
遍历每个 batch·head 行(`range(cdiv(T,BT)−1,−1,−1)`),每个 chunk 输出
`b_s − tl.cumsum(b_s) + b_z` —— chunk 内后缀和加上进位的后缀总和 `b_z`
(store 之前先用当前 chunk 总和更新)。注意 kernel 通过减法恒等式配合
**前向** `tl.cumsum` 实现反向 cumsum;本移植等的 DSL 能力其实是倒序 range
循环(而非 `reverse=True`)。

证明架构镜像前向孪生 `chunk_cumsum_kernel`(chunk 内 scan 恒等式 + carry-fold)
与同族 `reversed_cumsum` 的打包方式:单 chunk 路径(`T ≤ BT`,覆盖 Python
基准 `T = 4`、`BT = 16`)端到端 genuine;逐 chunk face 在显式 carry 缓冲假设下
陈述,跨 chunk 的 `b_z` 调度是文档化的受信运行时边界(#290 式进位寄存器)。
按 `chunk_cumsum_kernel` 的 `surface_loop_correct` 风格补全 loop 不变式证明是
自然的后续升级。
