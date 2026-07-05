# layer_norm_welfold

- 源文件:`layer_norm_welfold.py`
- Corpus:TritonBench-G v1
- 状态:DONE —— DSL 移植(`LayerNormWelfold.lean`)、genuine 闭式
  mean / rstd / 仿射输出 spec、`ComputeCorrect.Realizes` 汇总定理
  (`layer_norm_welfold_output_summary_general`)全部证明完成,无 sorry。

尽管目录名如此,`triton_red_fused_native_layer_norm_no_welford` 其实是
**两趟(non-Welford)** torch-inductor LayerNorm 前向:第一个 tiled 循环
累加行和(mean → `in_out_ptr0`,前置一次 `tl.debug_barrier()`),第二个
tiled 循环累加平方偏差(`libdevice.rsqrt(var + 1e-05)` → `in_out_ptr1`,
前置第二次 barrier),第三个 tiled 循环把归一化仿射输出
`((x − mean)·rstd)·w + b` 写入 `out_ptr0`。

证明架构镜像 Welford 姊妹 kernel `fused_layernorm_triton`(spec 相同,
归约代数更简单 —— 不需要矩合并恒等式):完整 faithful surface(三个循环、
两个 barrier、`tl.full` 累加器)可降到 algorithm 层;`XBLOCK = 1` 的两个
归约阶段在单归约 block(`rnumel ≤ RBLOCK`)下从 `in_ptr0` 端到端 genuine,
第二趟消费的是**寄存器**里的 mean;normalize 循环在诚实的 mean/rstd 单元
假设下覆盖**所有** chunk 下标(恰为归约阶段所存的值)。多迭代累加递推是
纯步进 face `sum_accumulate_step_closed`;跨迭代 `_tmp3`/`_tmp12` 寄存器
调度是文档化的受信运行时边界(#290 式进位寄存器)。
