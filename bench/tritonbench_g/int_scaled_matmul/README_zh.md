# int_scaled_matmul

- 源文件:`int_scaled_matmul.py`(上游 `data/TritonBench_G_v1/int_scaled_matmul.py`)
- Corpus:TritonBench-G v1
- 规模:303 行,2 个 `@triton.jit` kernel —— **两个都被启动、两个都建模**
- 状态:**已移植** —— `IntScaledMatmul.lean`,主定理
  `int_scaled_matmul_matmul_exec_genuine`(kernel 1)与
  `int_scaled_matmul_scaled_exec_genuine`(kernel 2)
  (`exec` 级、维度一般、0 `sorry`)。
- 库层 rider:本港落地了两项:(a)`tl.broadcast_to(e, [dims*])` 表达式
  语法(降低到既有的 `Op.broadcast` / `Op.remap` 广播机制;
  `bench/tests/TritonSmoke.lean` 有 smoke 门禁);(b)block-pointer 元素
  dtype 继承 —— DSL 指针元素 dtype 传播新增 `tl.make_block_ptr` 臂,经
  block pointer 的 `tl.load(bp, …)` 默认取 base 区域的元素 dtype(未标注
  base 保持历史 `.real` 默认;既有 block-ptr 各港降低结果不变)。

audit 锚定 JIT = `matmul_kernel_with_block_pointers`(文件第一个 kernel,
启动器 `int_matmul_kernel`):**block-pointer** int8×int8→int32 GEMM。三个
带运行时 per-`pid` 偏移的 `tl.make_block_ptr` 视图
(`Op.makeBlockPtrDynOffsets`)、`.int` 通道上的 `boundary_check=(0, 1)`
**零填充**载入、步长形循环 `for k in range(0, K, BLOCK_K)` 上的
`Op.dotInt` 累加(直接拼写 —— `Stmt.forRange` 自带步长,本 kernel 不引入
任何 trip-count binder)、循环内对两个输入指针的 `tl.advance`、以及带边界
检查的 block-pointer int32 store。`GROUP_M` constexpr 被**重绑为寄存器**
(`GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)`),与 Python 重绑名字
的方式完全一致 —— 已探针验证:名为 `GROUP_M` 的寄存器与 Lean binder 共存
无冲突(binder 只以反引形式出现)。

第二个 kernel `scaled_matmul_kernel_with_block_pointers`(启动器
`int_scaled_matmul_kernel`)是经典指针 GEMM —— 名字里有 block pointer 但
实际没有 —— `triton.ops.matmul` swizzle、值层擦除的
`tl.max_contiguous(tl.multiple_of(…))` 背后的 `% M`/`% N` 回绕行/列、
**降序**循环 `for k in range(K, 0, -BLOCK_K)` 带真 `EVEN_K` Bool constexpr
(**两臂全建模**:无掩码载入 vs `rk < k` 剩余数掩码 + `other=0.0`),以及
inductor 生成的后缀:`rm`/`rn` 重物化、`xindex = idx_n + (N * idx_m)` 平
铺寻址、`tmp0 = tl.load(s1_ptr + tl.broadcast_to(idx_m, …), mask,
eviction_policy=…)`(位置 mask 原生可解析;eviction 提示擦除)、以及
`acc * tmp0` 的掩码 store —— `.int × .real` 乘积经 `Op.intToReal` 自动
提升。

对每个在界输出格(`row < M`、`col < N`),两条头条给出:

```
kernel 1:  C[row·s_cm + col·s_cn]  =  MemCell.of .int  (∑ kk<K, A[row,kk]·B[kk,col])          (精确 ℤ)
kernel 2:  C[col + N·row]          =  MemCell.of .real ((∑ kk<K, A[row,kk]·B[kk,col] : ℤ→ℝ) · s1[row])
```

其中 `s1` 在**无步长**地址 `row` 处读取(`stride_s1m`/`stride_s1n` 由启动
器传入但在 body 中死亡 —— `fused_recurrent_retention` 死步长先例)。

披露的 surface 决策(模块内两条 `Translation-surface blocker:` 标记):

- **Kernel 1** ——(1)block-ptr 载入忠实拼写
  (`tl.load(bp, boundary_check=(0, 1))`,无附加 kwarg):`.int` 通道由
  `tl.make_block_ptr` 的类型化 base 区域继承
  (`base=$((a_ptr : Region .int))`,rider (b));披露的只有整数宽度擦除
  (`int8_quantization` 定宽家族)。(2)`tl.advance` 增量与 `tl.zeros`
  形状的元组→方括号改写。(3)`GROUP_M` 寄存器重绑拼写(左侧寄存器、
  `min` 内反引参数)。(4)TMA `order=(1, 0)` 元组是调度元数据,DSL 擦除
  (`matmul_tma` 先例)。
- **Kernel 2** ——(1)降序循环呈现为升序换元
  `for j in range(0, numKBlocks, 1)`,剩余数在循环体首句重物化
  `k = K - j*BLOCK_K`(ℕ 截断减法;`parallel_retention_attention` 降序
  杠杆先例)。(2)`ACC_TYPE: tl.constexpr = tl.int32` 按其唯一实例化固定
  (`bmm_optimized` 定臂家族)。(3)`tl.broadcast_to(…, mask.shape)` 的
  属性实参改写为字面维度 `[BLOCK_M, BLOCK_N]`。(4)
  `eviction_policy="evict_last"` 保留并作为纯缓存提示擦除。(5)宽度擦
  除;`.int` 载入上的 `other=0.0` 忠实落为 `Op.constInt 0`。存进 int32
  分配的 `C` 的 `.real` 型 store 是诚实建模:cell 携带**精确实数**乘积;
  硬件 float→int32 容器截断在模型之外(#154 定宽家族)。

假设设计:kernel 1 只需 `0 < BLOCK_K`(range 步长为正)加 store 地址映射
的 `hInj`(行主序 `C` 由 `mmCAddr_injective_rowMajor` 卸掉)—— 边界检查
让每个轴都参差安全,**不要求 `K` 可整除**。kernel 2 取
`hCeil : K ≤ numKBlocks·BLOCK_K`(ceil 形;多余的全掩码迭代贡献为零)与
`hEven : EVEN_K = true → K = numKBlocks·BLOCK_K`(精确形,只对无掩码臂
要求);其平铺 store 地址**无需注入性假设** —— 掩码通过的 lane 上
`col < N`,故 `col + N·row` 直接可证注入(`smCAddr_inj_active`)。所有
维度、步长、块尺寸保持符号化;网格是 1-D,无附加 program-id 假设。
