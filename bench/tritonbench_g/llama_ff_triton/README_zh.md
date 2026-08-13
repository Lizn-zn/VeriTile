# llama_ff_triton

- 源文件:`llama_ff_triton.py`(上游 `data/TritonBench_G_v1/llama_ff_triton.py`)
- Corpus:TritonBench-G v1
- 规模:152 行,1 个 `@triton.jit` kernel
- 状态:**已移植** —— `LlamaFfTriton.lean`,主定理
  `ff_llama_closed_form_correct`(`exec` 级闭式、维度一般、0 `sorry`)。

上游是 llama FFN 前半——**RMSNorm 融合 SwiGLU 双 GEMM**:每个输出瓦片流式
走 K 块,同时累加 RMS 矩(`a_sum += pow(a, 2)`)与两路 GEMM
(`acc1 += dot(a·rms_w, w1)`、`acc2 += dot(a·rms_w, w3)`),尾声用
`rsqrt(sum(a_sum)/K + EPS)` 归一化并做 `silu(acc1) * acc2` 门控,掩码店到
fp16 输出。

头条:每条活跃输出 cell 持 `silu(n_i · S1[i,j]) · (n_i · S2[i,j])` 的
fp16 转换,其中 `S1/S2` 是对 kernel `% M`/`% N` 回绕下标的真双重和 GEMM
参照 `Σ_t Σ_e A·RMS·W{1,3}`,`n_i = 1/√((Σ A²)/K + EPS)`——全部独立于
kernel 导出。侧条件:`K = BLOCK_SIZE_K · numKBlocks` 表述(载入无掩码)、
`soutn = 1` + `BN ≤ soutm`(店 lane 单射)、干净输入 `hundef`。

Translation-surface blocker(登记于 `proof_blockers.md`):`USE_FP8`
constexpr 臂整体掉臂(参数 + 分支;该路径把 int8 权重字节经
`bitcast=True` 位重解释为 `tl.float8e5`——ℝ 模型位级限制,不存在值层面
忠实转写;`sgmv_expand_slice` 掉臂先例);`tl.cdiv(K, BLOCK_SIZE_K)` 以反引
`numKBlocks` 提供;店的隐式 fp16 转换显式拼作
`(accumulator).to(tl.float16)`(`f8_conversion_utils` 先例);循环计数器
`_i` 对应 Python 的 `_`。宿主启动与逐 dtype 分派是可信边界。

`⊨[R]` io 面在 `StreamMasked3DKernelIO₄` 上结构可行(模块 docstring 已记
路径)但本轮未出——仅 exec 头条,`bmm_optimized` 先例。
