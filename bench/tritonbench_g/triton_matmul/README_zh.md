# triton_matmul

- 源文件:`triton_matmul.py`(上游 `data/TritonBench_G_v1/triton_matmul.py`)
- Corpus:TritonBench-G v1
- 规模:133 行,1 个 `@triton.jit` kernel
- 状态:**已移植** —— `TritonMatmul.lean`,主定理
  `triton_matmul_f16_closed_form_correct` / `triton_matmul_f8_closed_form_correct`
  (exec 闭式)与 `triton_matmul_f16_io_correctness` /
  `triton_matmul_f8_io_correctness`(`⊨[R]` 流式 io 面),
  全部维度一般、0 `sorry`。

上游是 Triton 教程的 L2 分组瓦片 GEMM `C = A × B`,与孪生
`matmul_triton_autotune` 有两处不同:越界 A 行 / B 列下标用
`tl.where(offs < M, offs, 0)` **钳制**到 0(带
`tl.max_contiguous`/`tl.multiple_of` 提示)而非 `% M` 回绕;尾声降精度按
**输出缓冲区的编译期 dtype** 分支——宿主分配 fp8 则 `float8e4nv`,否则
`float16`。

两条尾声臂经一套 od 泛型证明栈全部证毕:每条活跃输出 cell 持
`od(Σ_{k<K} A[i,k]·B[k,j])`(exec 闭式);流式 `⊨[R]` 面上输出窗口读回
`od` 类型 cell,值为 `R.round od (Σ A·B)`,对每个舍入模型 R——fp8 臂是
**语料库首个 fp8 matmul 面**。侧条件:`K = BLOCK_SIZE_K · numKBlocks`
圈数表述(模板先例)、`scn = 1` + `BN ≤ scm`(店 lane 单射)、精确面的
干净输入 `hundef`。

Translation-surface blocker(登记于 `proof_blockers.md`):constexpr 尾声
拆成两个 Lean surface(`matmul_tma` 先例);`tl.cdiv(K, BLOCK_SIZE_K)` 以
反引 `numKBlocks` 提供;K 循环计数器拼作 `kk`。
`triton.jit(launch_metadata=...)` 钩子与宿主逐 dtype 配置表是可信边界。
