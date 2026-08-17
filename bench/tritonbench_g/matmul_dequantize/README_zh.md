# matmul_dequantize

- 源文件:`matmul_dequantize.py`(上游 `data/TritonBench_G_v1/matmul_dequantize.py`)
- Corpus:TritonBench-G v1
- 规模:358 行,3 个 `@triton.jit` kernel —— **三个全部被启动,全部建模**
- 状态:**已移植** —— `MatmulDequantize.lean`,主定理
  `matmul_dequantize_matmul4_exec_genuine` /
  `matmul_dequantize_matmul_exec_genuine` /
  `matmul_dequantize_dequantize_exec_genuine`(`exec` 级、维度一般、0
  `sorry`)。

上游文件走三条独立的 int4 反量化路径,每条对应一个 test case,且每个 JIT
都是本 bench 已移植 kernel 的轻文本变体 —— 本港按源码顺序把三套证明栈镜像
进同一个 namespace(`decay_cumsum` 多 kernel 单文件先例):

| JIT(首个 = audit 锚) | 启动器 | 模板港 | 相对模板的差异 |
|---|---|---|---|
| `matmul4_kernel` | `matmul_dequantize_int4_gptq` | `matmul_dequantize_int4` | 去掉孪生源码 `tl.dot` 操作数上的 `.to(a.dtype)`;死绑定 `c` 的 cast 直接拼 `tl.float16` |
| `matmul_kernel` | `matmul_dequantize_int4_s2` | `int4_matmul` | 三处工作 cast 直接拼 `tl.float16`(孪生的 `element_ty` 拼法会被擦除;这里不会) |
| `dequantize_kernel` | `matmul_dequantize_int4_s1` → `dequantize_int4` | `matmul_dequant_int4` | kernel 体**逐字节相同** —— 纯改名镜像 |

## `matmul4_kernel` —— GPTQ 反量化 GEMM,`NO_GROUPS` 双臂

`C = A · dequant(qweight)`:K 轴每 32-bit 字打包八个 4-bit 权重,N 轴每字
打包八个 4-bit 零点,先乘 scale 再减预乘的零点,标准 group-swizzle pid 分
解。`NO_GROUPS : Bool` 保持真参数 —— 一条头条经 `groupRow` 统一子同时覆盖
双臂(循环前组行 0 预载 vs 每步在 `k // (groupsize // BLOCK_SIZE_K)` 重
载)。`c = accumulator.to(tl.float16)` 编译成真 `Op.castFloat`,但仍是
**死绑定**:store 写的是 `accumulator`,与源码一致,因此头条陈述的是
float32 累加器。披露:下标算术中的整数字面量反引号化 `$(n)`(`0xF` 写
`$(15)`);`bits` / `infearure_per_bits` 按普通语句转写。

## `matmul_kernel` —— 有符号反量化 GEMM,`SPLIT_K = 1` 臂,fp16 店

`int4_matmul` 的三条披露原样沿用:**(1)** `SPLIT_K` 固定为 `1`(`tl.store`
臂;`tl.atomic_add` 臂随 constexpr 掉臂;头条携带 `s.pids 1 = 0`);
**(2)** 循环界 `tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)` 写成反引号绑定子
`numKBlocks`,侧条件 `K = BLOCK_SIZE_K · numKBlocks`;**(3)** 有符号
nibble 差在 `.int` 通道上以 `tl.cast(·, tl.int32) - tl.cast(·, tl.int32)`
拼写并经 `Op.intToReal` 提升;`hBK8 : BK % 8 = 0` 把
`(BLOCK_SIZE_K * SPLIT_K * stride_bk // 8)` 指针步进与打包字地址等同。与
孪生不同,三处 cast **忠实存活**:`b = ((int_b - int_bzp) * bs).to(
tl.float16)` 把 `b` 放上 `.fp16` 寄存器通道,dot 的 `a.to(tl.float16)` /
`b.to(tl.float16)` 被 DSL 的实值 `tl.dot` 回宽 `fp16 → real`(cast 恒等使
往返消去),被 store 的 `c = accumulator.to(tl.float16)` 让终端店变为
`.fp16` 类型 —— 头条因此在 `MemCell` 层读输出:
`C[row, col] = MemCell.of .fp16 (fp16(accSpec))`(`matmul_tma` f16 臂先
例,这里落在带掩码的 `.ptr` 店上)。

## `dequantize_kernel` —— 独立瓦片反量化器

`fp_b[k, n] = (nib(b) − nib(zp)) · scale`,所有载入与 store 共用同一
`(offs_n < N) & (offs_k < K)` 门;`other=0.0` kwarg 忠实保留(打包 `.nat`
通道上字面量读作 nat `0`;被掩掉的 lane 永不被消费)。披露 = 
`matmul_dequant_int4` 的两条:反量化语句的内联 `.int` 通道改拼,以及
`$(n)` 字面量反引号化。头条在维度变量之外唯一的假设是 `hInj`;宿主行主序
的 `fp_b` 由 `fpbAddr_injective` 直接给出。

三条头条中全部维度、stride、block size 与 `group_size` 保持符号化;任何
输出区都不会被读回进规约。两个 matmul kernel 共享 pid-swizzle 定义
(`numPidM` … `pidN`)与 `C` 地址映射(`cAddr`、`cAddr_injective`)——
它们在源码里逐字相同。
