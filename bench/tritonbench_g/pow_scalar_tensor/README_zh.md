# pow_scalar_tensor

- 源文件:`pow_scalar_tensor.py`
- Corpus:TritonBench-G v1
- 状态:DONE —— DSL 移植(`PowScalarTensor.lean`)、genuine 标量底数幂闭式
  spec(`powSpec = Real.rpow val0 (in0[t·in0_stride0])`)、
  `ComputeCorrect.Realizes` 汇总定理
  (`pow_scalar_tensor_output_summary_general`)全部证明完成,无 sorry。

`pow_func_scalar_tensor_kernel_rank_1` 是 FlagGems pointwise-codegen 的
rank-1 elementwise kernel(与 `relu_strided_buffer` 同一骨架):带
`boundary_check` 的 block-ptr load、内联的 `pow_func_scalar_tensor` helper
(`_pow(val0.to(tl.float32), in0)` —— 运行时标量是底数 BASE,加载的张量是
指数 EXPONENT)、block-ptr store。两个 `one_tile_per_cta` constexpr 分支均
已验证:单 tile 分支逐 lane 验证,grid-stride 循环通过循环不变式
`gs_loop_readback`(跨迭代 disjointness + 输入区保持)端到端验证。headline
完全 dimension-general(`val0`、`s0`、strides、`tile_size0`、
`tiles_per_cta` 全部全称量化),honest 侧条件为 `0 < out0_stride0`,
grid-stride 分支另需 `in0_ptr ≠ out0_ptr` 与 `0 < numPids`。

建模边界:`_pow` 建模为 `Op.pow` / Mathlib `Real.rpow` —— 对 `val0 > 0`
精确;负底数配非整数指数时 `rpow` 返回 junk-value 约定而 CUDA `pow` 返回
NaN(见 `VeriTile/Triton/Core/Ast.lean` 中 `Op.pow` 的 doc comment)。
`.to(tl.float32)` / `.to(*_ptr.type.element_ty)` cast 在 ℝ 通道上消解为恒等。

本目录是 TritonBench-G 全形式化 roadmap 的 per-kernel 工作目录。
