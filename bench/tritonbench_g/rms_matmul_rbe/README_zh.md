# rms_matmul_rbe

- 源文件:`rms_matmul_rbe.py`(上游 `data/TritonBench_G_v1/rms_matmul_rbe.py`)
- Corpus:TritonBench-G v1
- 规模:279 行,2 个 `@triton.jit` kernel
- 状态:**已移植** —— `RmsMatmulRbe.lean`,主定理
  `rms_matmul_rbe_closed_form_correct`(GEMM)与
  `rms_matmul_rbe_qkv_closed_form_correct`(QKV 调用者——单条捆绑
  specification,三个 `Realizes` 面的合取),均为 `exec` 级、维度一般、
  0 `sorry`。

文件首核 `rms_matmul_rbe` 是批量 RMSNorm 融合 GEMM(与 `rms_rbe_matmul`
港的目标 JIT 逐字节相同),此处为孪生镜像。第二核——宿主唯一启动的
——是 **`rms_matmul_rbe_qkv` 跨 JIT 调用者**:体内调 `rms_matmul_rbe(...)`
三次(Q/K/V 权重与输出,共享 `x`/`rms_w`)。DSL 无 jit 间调用面,故被调核
体内联三份(`attn_fwd` 内联先例);证明把单 pass 机器按 region/stride
参数化一次,三次应用 + pass 间帧传输(10 条区域互异假设——恰为传输所需
之集;宿主传的是互异缓冲区)。

QKV 头条:全程跑完后 Q/K/V 各自的活跃 cell 持 `n_i · S_w[i,j]` 的 fp16
转换——共享 RMS 归一化因子对各自权重矩阵。侧条件:
`K = BLOCK_SIZE_K · numKBlocks`(载入无掩码)、每路输出的行主序单射对、
互异事实、干净输入 `hundef`。

Translation-surface blocker(登记于 `proof_blockers.md`):跨 JIT 调用
内联;`USE_FP8` constexpr 臂掉臂(`bitcast=True` 位重解释,ℝ 模型限制);
被调核未用参数(`start_token_position`/`RBE_EPILOGUE`/`THETA`)在内联下
消没;`tl.cdiv(K, BLOCK_SIZE_K)` 以反引 `numKBlocks` 提供;店的隐式 fp16
转换显式拼作 `(accumulator).to(tl.float16)`;循环计数器 `_i`。宿主启动与
逐 dtype 分派是可信边界。
