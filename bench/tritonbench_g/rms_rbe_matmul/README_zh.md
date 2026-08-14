# rms_rbe_matmul

- 源文件:`rms_rbe_matmul.py`(上游 `data/TritonBench_G_v1/rms_rbe_matmul.py`)
- Corpus:TritonBench-G v1
- 规模:190 行,2 个 `@triton.jit` kernel(其一为文件内死代码,见下)
- 状态:**已移植** —— `RmsRbeMatmul.lean`,主定理
  `rms_matmul_rbe_closed_form_correct`(`exec` 级闭式、维度一般、0 `sorry`)。

目标 JIT:**`rms_matmul_rbe`**(文件第二核)——批量 RMSNorm 融合 GEMM:
`out[b] = (rms(x[b]) · rms_w) @ w`,即 `llama_ff_triton` 的核减 SwiGLU 门
加 batch 网格轴。文件第一核 `rbe_triton`(旋转位置编码)是**文件内死代码**:
没有任何宿主函数启动它,且它调用文件里未定义的 `get_freq_multi_tokens`
——上游文件自己都跑不了它。不建模。

头条:每条活跃输出 cell 持 `n_i · S[i,j]` 的 fp16 转换,
`S = Σ_t Σ_e X[b, r(i), ·]·RMS[·]·W[·, c(j)]`(每次 X 读带批偏移
`pid_batch·stride_x_batch`,`% M`/`% N` 回绕下标),
`n_i = 1/√((Σ X²)/K + EPS)`——独立于 kernel 导出。侧条件:
`K = BLOCK_SIZE_K · numKBlocks`(载入无掩码)、`son = 1` + `BN ≤ som`
(店 lane 单射)、干净输入 `hundef`。

Translation-surface blocker(登记于 `proof_blockers.md`):目标 JIT 为文件
第二核(`rbe_triton` 死代码/不建模,如上);`USE_FP8` constexpr 臂掉臂
(`bitcast=True` 位重解释,ℝ 模型限制;`llama_ff_triton` 先例);未用参数
`start_token_position`/`RBE_EPILOGUE`/`THETA` 删除;`tl.cdiv(K, BLOCK_SIZE_K)`
以反引 `numKBlocks` 提供;店的隐式 fp16 转换显式拼作
`(accumulator).to(tl.float16)`;循环计数器 `_i` 对应 Python 的 `_`。宿主
启动与逐 dtype 分派是可信边界。
