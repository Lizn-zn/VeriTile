# matmul_dequantize_int4

- 源文件:`matmul_dequantize_int4.py`(上游 `data/TritonBench_G_v1/matmul_dequantize_int4.py`)
- Corpus:TritonBench-G v1
- 规模:268 行,1 个 `@triton.jit` kernel
- 状态:**已移植** —— `MatmulDequantizeInt4.lean`,主定理
  `matmul_dequantize_int4_exec_genuine`(`exec` 级、维度一般、0 `sorry`)。

`matmul4_kernel` 是 GPTQ 式的反量化 GEMM:`C = A · dequant(qweight)`。`qweight`
沿 K 轴把八个 4-bit 权重打包进一个 32-bit 字,`qzeros` 沿 N 轴把八个 4-bit 零点
打包进一个字。每个 K 步用移位加半字节掩码解包权重,乘 `scales`,再减去该组的零点。

`NO_GROUPS` 两种配置由同一条定理覆盖:差别只在**组行**,而规约在
`groupRow NO_GROUPS` 处读 `scales`/`zeros` —— 开旗时每步都是 `0`,否则是
`k // (group_size // BLOCK_SIZE_K)`。全部维度、stride、block size 和 `group_size`
保持符号化。

输出回读需要"不同 lane 落在不同 `C` 地址"上,这就是定理的 `hInj`;行主序的 `C`
由 `cAddr_injective` 直接给出。
