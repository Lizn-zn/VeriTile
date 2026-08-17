# matmul_dequant_int4

- 源文件:`matmul_dequant_int4.py`(上游 `data/TritonBench_G_v1/matmul_dequant_int4.py`)
- Corpus:TritonBench-G v1
- 规模:302 行,2 个 `@triton.jit` kernel
- 状态:**已移植** —— `MatmulDequantInt4.lean`,主定理
  `matmul_dequant_int4_exec_genuine`(`exec` 级、维度一般、0 `sorry`)。

目标 JIT = `dequantize_kernel`,文件中唯一被宿主启动的 kernel
(`matmul_dequantize_int4_s1` → `dequantize_int4` →
`dequantize_kernel[grid]`,随后宿主侧 `torch.mm(a, fp_b)`;测试只调
`matmul_dequantize_int4_s1`)。直线型 int4 反量化瓦片:程序
`(k_block_idx, n_block_idx)` 载入 `[BLOCK_SIZE_K, BLOCK_SIZE_N]` 一片打包
权重字(`b`,沿 K 轴每个 32-bit 字打包八个 4-bit 权重)、对应的打包零点字
(`zp`,沿 N 轴每字打包八个 4-bit 零点,每 `group_size` 个 K 行一组行)与
按组 scale,移位加掩码解出两枚 nibble,在共享掩码
`(offs_n < N) & (offs_k < K)` 下存回
`fp_weight = (nib(b) − nib(zp)) · scale`。

文件的第一个 kernel `matmul4_kernel` 在本文件中是死代码 —— 没有任何宿主
引用它,且其函数体与 `matmul_dequantize_int4` 港已移植的 kernel 逐字节相同
—— 按 `rms_rbe_matmul` 死核先例不建模。

两处披露的 surface 决策(见模块的 `Translation-surface blocker:` 标记):

- **显式 cast 的有符号反量化** —— 打包容器建模为 `Region .nat`(DSL 的位
  运算 `>>` / `&` 定义在该通道),但 nibble 差会取负而 ℕ 减法截断,故减法
  拼写为 `tl.cast((int32_b >> b_shift) & 0xF, tl.int32) -
  tl.cast((zp_b >> bzp_shift) & 0xF, tl.int32)`,乘积经 `Op.intToReal`
  提升到 ℝ —— 即 `int4_matmul` 的有符号反量化改拼,此处内联应用。
- **反引号整型字面量** —— 源码索引算术里的 `8` / `4` / `0xF` /
  `group_size` 写成 `$(8)` / `$(4)` / `$(15)` / `$(group_size)`(裸字面量
  会被推断成 `.real`;`0xF` 同值改十进制拼写 —— `int4_matmul` 先例)。

三个掩码载入忠实保留 `other=0.0`(在 `.nat` 通道上 expander 把 `0.0` 读作
nat 常量 `0`,值恒等)。全部维度、stride、block size 与 `group_size` 保持
符号化,两个 program id 全自由,故头条一次覆盖所有 autotune 配置与全部网格
程序。Lean 的自然数除法全定义,无需 `0 < group_size` 假设,也没有整除侧条
件(载入与 store 共用同一掩码)。输出回读需要"不同 lane 落在不同 `fp_b`
地址"(`hInj`);宿主行主序的 `fp_b` 由 `fpbAddr_injective` 直接给出。
