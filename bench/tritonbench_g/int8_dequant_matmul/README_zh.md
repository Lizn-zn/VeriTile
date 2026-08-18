# int8_dequant_matmul

- 源文件:`int8_dequant_matmul.py`(上游 `data/TritonBench_G_v1/int8_dequant_matmul.py`)
- Corpus:TritonBench-G v1
- 规模:211 行,1 个 `@triton.jit` kernel
- 状态:**已移植** —— `Int8DequantMatmul.lean`,主定理
  `int8_dequant_matmul_exec_genuine`(`exec` 级、维度一般、0 `sorry`)。

目标 JIT = `_int8_matmul_rowwise_dequantize`,文件中唯一的 `@triton.jit`
kernel(启动器为 `int8_matmul_rowwise_dequantize`)。int8×int8→int32 GEMM
加逐行反量化尾声:`A`、`B` 是有符号 `torch.int8` 张量(建模为
`Region .int`),K 循环经整数 `tl.dot` 累加精确 ℤ 矩阵乘 —— 本港是
**`Op.dotInt` 的首个消费者**;尾声按 kernel 原括号结构
`w_factor * (x_factor * (acc * divfactor))` 逐格重标定(逐行 `state_x`、
逐列 `state_w`、宿主 `divfactor = 1/(127·127)`),降为 fp16,`has_bias`
时再加逐列 fp16 偏置,最后掩码写入 `torch.float16` 输出 `C`。标准
group-swizzle pid 分解;行/列偏移带 `% M` / `% N` **回绕**且 K 循环载入
无掩码。

五处披露的 surface 决策(见模块的 `Translation-surface blocker:` 标记):

- **`SPLIT_K` 固定为 `1`**(`tl.store` 臂)。autotune 表扫 `SPLIT_K ∈ {1,
  2, 4, 8, 16}`;`tl.atomic_add` 臂(向 `pre_hook=init_to_zero("C")` 的
  宿主清零 `C` 累加)随 constexpr 一并掉臂。所有 `* SPLIT_K` 因子折为
  `SPLIT_K = 1` 值;头条携带启动事实 `s.pids 1 = 0`(grid 第 1 轴宽度即
  `SPLIT_K`),而 `pid_z` 与 `rk = pid_z * BLOCK_K + tl.arange(0,
  BLOCK_K)` 原样保留。
- **`EVEN_K` 固定为 `True`**(无掩码载入臂)。`EVEN_K` 是
  `@triton.heuristics` constexpr `K % (BLOCK_K * SPLIT_K) == 0`,掩码
  `else` 臂随 constexpr 掉臂;循环界 `tl.cdiv(K, BLOCK_K * SPLIT_K)` 写成
  反引号绑定子 `numKBlocks`,侧条件 `K = BLOCK_K · numKBlocks`(载入无
  掩码,迭代数必须精确)。
- **`has_bias` 保留为真 `Bool` 参数,两臂全建模**(`matmul_dequantize`
  的 `NO_GROUPS` 先例):宿主传 `0`/`1`;头条用守卫偏置项
  `if has_bias then bias(col) else 0` 把两臂收进一条陈述。
- **`.to(C.dtype.element_ty)` 拼写为 `.to(tl.float16)`**(宿主把 `C`
  分配为 `torch.float16`):累加器降型编译为真 `Op.castFloat real →
  fp16`,偏置载入直接落 `.fp16` 通道,终端 store 因而是 **`.fp16` 型**,
  头条在 `MemCell` 层回读输出(`MemCell.of .fp16 (fp16(i8Spec))` ——
  `matmul_dequantize` `matmul_kernel` 先例;占位 cast 是恒等)。
- **`$(n)` 字面量反引号**:下标算术里的整数字面量写 `$(n)`(裸字面量会
  被 DSL 推断为 `.real`)。

全部维度、stride、block size(含 `GROUP_M`)保持符号化。store 掩码作用在
**未回绕**的输出坐标上,因此凡掩码放行的 lane 都有 `row < M` / `col <
N`,偏移回绕恒等消去 —— 头条闭式以有符号 ℤ 值读平直的 `A[row, ·]` 行与
`B[·, col]` 列,平直的 `state_x[row]` / `state_w[col]` 标定,偏置本就按
未回绕 `rn` 读、天然落在平直 `col`。输出回读需要"不同 lane 落在不同 `C`
地址"(`hInj`);行主序的 `C` 由 `cAddr_injective` 直接给出。
