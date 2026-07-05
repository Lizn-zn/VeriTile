# triton_linear_activation

- 源文件:`triton_linear_activation.py`
- Corpus:TritonBench-G v1
- 状态:DONE —— DSL 移植(`TritonLinearActivation.lean`),真闭式 spec
  `activation(bias + Σ_k A·B)`,打包的 `ComputeCorrect.Realizes` 总结
  (`triton_linear_activation_output_summary_general`)已 sorry-free 证明。

`kernel_fma` 是 xformers 风格的融合线性层 `Out = activation(A × Wᵀ + bias)`:
L2 分组的线性 pid 调度、`%M`/`%N` 下标回绕(kernel 注释里的
"trick to avoid masking on M and N")、可选的带 mask bias 行种子(`HAS_BIAS`)、
`acc += tl.dot(a, b)` 的 K 循环、可选的激活前值溢出到 `ACT_INPUTS`
(`SHOULD_SAVE_ACT_INPUTS`,为 backward 保存),以及字符串 constexpr 的激活门
(`tanh` / `gelu` / `fast_gelu` / `relu` / 恒等;`@triton.jit` helper 内联 ——
GELU 用精确实数误差函数 `VeriTile.Math.realErf`,fast-GELU 用 `Real.tanh`)。

主定理是一条打包的、维度通用的定理,覆盖任意形状、stride、block 尺寸、
group 大小、两个 `Bool` constexpr 开关以及**任意** `ACTIVATION` 字符串:
合取 (2) 证明每个 `C` 单元等于真·激活后的线性形式(对输入内存的 `gemmSum`
`Finset.sum`);合取 (3) 证明每个 `ACT_INPUTS` 单元等于未激活的激活前值。
诚实边界:`K_LOAD_MASK_NEEDED = True` 的 heuristics 分支(`K = BLOCK_K ·
numKBlocks` 整除),以及 per-program 的 tile 适配假设 `hFitM`/`hFitN`
(当 `BLOCK_M ∣ M ∧ BLOCK_N ∣ N` 时恒成立)—— kernel 的 store mask 检查的是
**已回绕**的偏移,越界 tile 会真实地回绕覆盖而不是被 mask 掉。
