# int4_matmul

- 源文件:`int4_matmul.py`(上游 `data/TritonBench_G_v1/int4_matmul.py`)
- Corpus:TritonBench-G v1
- 规模:251 行,1 个 `@triton.jit` kernel
- 状态:**已移植** —— `Int4Matmul.lean`,主定理
  `int4_matmul_exec_genuine`(`exec` 级、维度一般、0 `sorry`)。

目标 JIT = `matmul_kernel`,文件中唯一的 `@triton.jit` kernel(启动器为
`matmul_dequantize_int4_s2`)。GPTQ 式 int4 反量化 GEMM:
`C = A · dequant(B)`。`qweight` 沿 K 轴把八个 4-bit 权重打包进一个 32-bit
字,`qzeros` 沿 N 轴把八个 4-bit 零点打包进一个字,每 `group_size` 个 K 行
共享一组 `(scale, zero-point)`;标准 group-swizzle pid 分解;行/列偏移带
`% M` / `% N` **回绕**且载入无掩码;fp32 累加器。与孪生港
`matmul_dequantize_int4`(先乘 scale 再减)不同,本 kernel **先减零点
nibble** —— 差值是有符号的。

三处披露的 surface 决策(见模块的 `Translation-surface blocker:` 标记):

- **`SPLIT_K` 固定为 `1`**(`tl.store` 臂)。autotune 表扫 `SPLIT_K ∈ {1,
  2}`;`tl.atomic_add` 臂(向 `reset_to_zero=['c_ptr']` 的宿主清零 `C`
  累加)随 constexpr 一并掉臂。头条携带对应的启动事实 `s.pids 1 = 0`
  (grid 第 1 轴宽度即 `SPLIT_K`)。
- **`numKBlocks` 循环界**:`tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)` 写成反引
  号绑定子 `numKBlocks`,侧条件 `K = BLOCK_SIZE_K · numKBlocks`(kernel
  自己 docstring 里的 assert;载入无掩码,迭代数必须精确)。
- **显式 cast 的有符号反量化**:打包容器建模为 `Region .nat`(DSL 的位运
  算 `>>` / `&` 定义在该通道),但 `int_b - int_bzp` 会取负而 ℕ 减法截断,
  故减法拼写为 `tl.cast(·, tl.int32) - tl.cast(·, tl.int32)`,经
  `Op.intToReal` 提升到 ℝ —— 本港是该算子的**首个消费者**。源码另一条
  assert `BLOCK_SIZE_K % 8 == 0` 即头条的 `hBK8`,用来把 kernel 的
  `(BLOCK_SIZE_K * stride_bk) // 8` 指针步进与打包字地址等同。

全部维度、stride、block size 与 `group_size` 保持符号化。store 掩码作用在
**未回绕**的输出坐标上,因此凡掩码放行的 lane 都有 `row < M` / `col < N`,
偏移回绕恒等消去 —— 头条闭式读的是平直的 `A[row, ·]` 行与 `B`/`BS`/`BZP`
列,组行按 lane 取 `(e + k·BK) / group_size`。输出回读需要"不同 lane 落在
不同 `C` 地址"(`hInj`);行主序的 `C` 由 `cAddr_injective` 直接给出。
