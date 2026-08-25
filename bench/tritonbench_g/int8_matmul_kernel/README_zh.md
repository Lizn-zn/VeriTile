# int8_matmul_kernel

- 源文件:`int8_matmul_kernel.py`(上游 `data/TritonBench_G_v1/int8_matmul_kernel.py`)
- Corpus:TritonBench-G v1
- 规模:271 行,1 个 `@triton.jit` kernel
- 状态:**已移植** —— `Int8MatmulKernel.lean`,主定理
  `int8_matmul_kernel_exec_genuine`(`exec` 级、维度一般、0 `sorry`)。

目标 JIT = `matmul_kernel`,文件中唯一的 `@triton.jit` kernel(启动器为
`matmul`,1-D group-swizzle 网格)。**纯整数** 2-bit 打包权重 GEMM:`A` 是
`torch.int32`(`Region .int`),`B` 是 `torch.uint8`(`Region .nat`),沿 K
每字节打包四个 2-bit 权重(`b.shape = [K//4, N]`),`C` 是 `torch.int32`
—— 终端 store 为 **`.int` 型**,全程无浮点。嵌套循环外层是 2-bit **字段
下标** `i ∈ {0..3}`、内层是打包 K 块 `j`:每个内层步抽取字段 `i`
(`mask = 3 << (2*i)`;`(b_uint8 & mask) >> (2*i)`),减去全一
`tensor_full` 移入有符号区间 `{−1,0,1,2}`,再累加精确整数
`tl.dot(a, b − 1, out_dtype=tl.int32)`(`Op.dotInt`)。`a_ptrs`
**跨两层循环连续推进**(从不重置),`A` 的列按序扫过 `0..K−1`;而
`b_ptrs` 在每个外层迭代顶部重绑回 `B` 起点。group-swizzle pid 分解带本
kernel 特有的 `pid_m` 行**多一层 `% num_pid_in_group`**(忠实拼写);
行/列偏移带 `% M` / `% N` 回绕。

对每个在界输出格,头条给出的 int32 存储值(ℤ 上):

```
C[row, col] = ∑ i<4, ∑ kk<K/4, A[row, i·(K/4) + kk] · (bits_i(B[kk, col]) − 1)
bits_i(w)   = (w >>> (2·i)) &&& 3          (K/4 = numKBlocks · BLOCK_SIZE_K)
```

四处披露的 surface 决策(见模块的 `Translation-surface blocker:` 标记):

- **`tl.cdiv(K // 4, BLOCK_SIZE_K)` 写成反引 binder `numKBlocks`**(内层
  循环界与 `k = i * tl.cdiv(K // 4, BLOCK_SIZE_K) + j` 记账),诚实侧条件
  `hK : K = 4 · (BLOCK_SIZE_K · numKBlocks)` —— 正是源码自己的
  `tl.static_assert(K % (4 * BLOCK_SIZE_K) == 0)`(留在 surface 中,值层
  擦除为 no-op 语句)。在 `hK` 下两个载入掩码**退化全真**,K 向不可能
  参差,载入化归为无掩码读。
- **元组形状参数改写为方括号**:`tl.zeros((BM, BN), …)` →
  `tl.zeros([$(BM), $(BN)], …)`、`tl.full((1,), 1, …)` →
  `tl.full([1], 1, …)`(audit 已知的 `(1,)` vs `[1]` 改写)。
- **整数位宽被擦除**(`int8_quantization` 定宽家族):`A` 载入上的
  `.to(tl.int8)` 在 `.int` 通道是 no-op —— 模型保留完整启动态 int32 值,
  而 int8 硬件会对 ≥ 128 的值回绕(宿主测试喂 `A` 值 `0..255`);2-bit
  抽取上的 `.to(tl.int8)` 降为真有符号跳变 `Op.castNatToInt`(抽取值仅
  `{0..3}`,不可能回绕);`tl.full([1], 1, dtype=tl.int8)` 降为擦宽的
  `.int` 字面量 `Op.constInt 1`。
- **`$(n)` 字面量反引**(裸字面量会被 DSL 表达式类型推断成 `.real`)。

已探测且**忠实拼写**(非披露项):`tl.dot` 的 `out_dtype=tl.int32` kwarg
原样保留、被宏擦除(`Op.dotInt` 本就是精确 ℤ,该 kwarg 语义冗余);
`b - tensor_full`(`[BK, BN] − [1]`)经升秩广播 `Broadcast.leadR` 编译、
无需改写;`tl.program_id(axis=0)` 是原生语法;循环内名为 `mask` 的寄存器
不遮蔽任何东西(store 掩码是独立的 `c_mask`)。

所有维度、步长、块尺寸保持符号参数。store 掩码在**未回绕**输出坐标上,
放行的每条 lane 上 `row < M` / `col < N` 使回绕成为恒等 —— 头条直接读
普通 `A[row, ·]` 行(有符号 ℤ)与普通打包字节 `B[kk, col]`(ℕ)。输出
读回在 `MemCell` 层(`MemCell.of .int`),需要不同 lane 落在不同 `C`
地址(`hInj`);`imCAddr_injective` 对行主序 `C` 卸掉它。网格是 1-D,
无需额外 program-id 假设。
