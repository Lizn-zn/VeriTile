# int8_matmul_quantization

- 源文件:`int8_matmul_quantization.py`(上游 `data/TritonBench_G_v1/int8_matmul_quantization.py`)
- Corpus:TritonBench-G v1
- 规模:267 行,2 个 `@triton.jit` kernel —— **两个都被启动、两个都建模**
- 状态:**已移植** —— `Int8MatmulQuantization.lean`,主定理
  `int8_matmul_quantization_quantize_exec_genuine` 与
  `int8_matmul_quantization_matmul_exec_genuine`(`exec` 级、维度一般、
  0 `sorry`)。

Kernel 1 = `quantize_int8_perrow_kernel`(audit anchor;启动器
`quantize_int8_perrow`),kernel 2 = `matmul_kernel`(启动器
`matmul_int8`);测试经 `matmul_quantize_int8` 同时驱动两者。二者构成动态
量化流水线:kernel 1 对 `fpa` 的每行沿 K 分块**流式扫两遍** —— 第一遍在
K 掩码(`other=0.0`)瓦片上累计逐行 running abs-max,然后
`a_scale = a_max / 127.`;第二遍重载同样的块,量化
`(fpa / a_scale[:, None]).to(tl.int8)` 并按 K 掩码存 `.int` 瓦片,最后把
逐行 scale 存进 fp16 向量。Kernel 2 是消费端 int8×int8→int32 GEMM:
group-swizzle pid、`% M` / `% N` 回绕偏移、余数掩码的 `.int` K 载入喂给
精确 ℤ `tl.dot`(`Op.dotInt`),尾声
`(acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(tl.float16)`
经两轴掩码 fp16 store 写回。

披露的 surface 决策(模块内两条 `Translation-surface blocker:` 标记,
每 kernel 一条):

- **ceil 形 `numKBlocks`**(两个 kernel):`tl.cdiv` 迭代数写成反引号绑定
  子,侧条件只取*覆盖半边* `K ≤ numKBlocks · BLOCK_SIZE_K`:K 载入/存储
  带余数掩码加 `other=0.0`,多出的全掩码块贡献 0 且不写内存 —— 对 ragged
  `K` 诚实(不同于无掩码循环 matmul 港的精确整除表述)。
- **`SPLIT_K` 固定为 `1`**(kernel 2,`tl.store` 臂):autotune 表扫
  `SPLIT_K ∈ {1, 2}`;`tl.atomic_add` 臂(向 `reset_to_zero=['c_ptr']`
  的宿主清零 `C` 累加)随 constexpr 掉臂;头条携带 `s.pids 1 = 0`。
- **`accumulator.to(tl.float32)` 走隐式提升**(kernel 2):显式拼写在 DSL
  中被 int 型标识符钉死,尾声乘积裸拼,降级项在 Python cast 的原位携带
  `Op.intToReal`。
- **隐式 fp16 scale 存储 cast**(kernel 1):宿主把 `a_scale` 分配为
  `torch.float16`,裸 `tl.store(as_ptr + as_offs, a_scale)` 拼作
  `(a_scale).to(tl.float16)`,scale 单元按 fp16 型 `MemCell` 回读。
  怪僻的**无 stride** `as_offs` arange
  (`pid_m * BLOCK_SIZE_M * stride_asm + tl.arange(0, BLOCK_SIZE_M)`)
  照原样拼写。
- **`.to(tl.int8)`**(kernel 1)降级为 `Op.castRealToInt8` —— 向零截断
  进无界 `.int` 载体;`±127` 硬件饱和是未建模的 `#154` 族边界。

Kernel 1 的行窗口回绕(`% M`)但 store 无行掩码;头条携带宿主的精确平铺
启动事实 `hFit : pid·BM + BM ≤ M`(grid = `M // BLOCK_SIZE_M` 个程序,
`BLOCK_SIZE_M = 1` 行),回绕在其下恒等。它的单条 `exec` 陈述同时携带
**两个**存储通道:每个 `(row, kg)` int8 单元持有按 kernel 自算的流式
abs-max scale 截断的商,每个 scale 单元持有该 scale 的 fp16 像。
Kernel 2 的头条读
`((Σ_{j<K} A[row,j]·B[j,col] : ℤ) : ℝ) · a_scale[row] · b_scale[col]` ——
掩码双重和在 ceil 形假设下收敛为干净的 `∑ j : Fin K`。两个头条都取
store 映射单射假设(`hInj`,kernel 2 附行主序判别引理);kernel 1 另需
其"写后读"循环与双区域尾声所迫使的区域互异事实。

本港随港带一条机械性库 rider:DSL 的 `other=` 展开表为字面量 `0.0` 增加
了 `.int` 臂(`Op.constInt 0`),镜像既有 `.nat` 臂 —— kernel 2 的掩码
`.int` 载入是首个消费者。
