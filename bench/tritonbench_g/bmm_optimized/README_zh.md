# bmm_optimized

- 源文件:`bmm_optimized.py`(上游 `data/TritonBench_G_v1/bmm_optimized.py`)
- Corpus:TritonBench-G v1
- 规模:232 行,1 个 `@triton.jit` kernel
- 状态:**已移植** —— `BmmOptimized.lean`,主定理 `bmm_o_exec_genuine`
  (`exec` 级、维度一般、0 `sorry`)。

上游实现批量 GEMM(`O[b] = A[b] · B[b]`,3-D 网格
`(cdiv(M,TILE_M), cdiv(N,TILE_N), batch)`),带 `GROUP_M` CTA 重排
(`tl.num_programs` 驱动,毛边末组有运行时 `GROUP_SIZE` 门)与
`DIVISIBLE_M/N/K` 启发式 constexpr(宿主报告维度整除时省掉掩码)。

移植固定 constexpr 赋值
`DIVISIBLE_M = DIVISIBLE_N = DIVISIBLE_K = False` —— 全掩码臂,唯一对
**任意** `M, N, K` 全域正确的臂(其余臂是各自整除域上的省掩码优化)——
并**完整**转写两条 `GROUP_M` CTA 重排臂,运行时 `GROUP_SIZE` 边界门作
嵌套 `Stmt.ifThenElse`。

头条:任意启动网格上的每个 program,掩码 `o` 店在每条窗口内 lane 上等于
`Σ_{t < K} A[pid_b, m, t] · B[pid_b, t, n]`,其中 `(m, n)` 是 CTA 重排后
瓦片的全局行/列(`bmmPidM`/`bmmPidN` 闭式 —— `GROUP_M = 1` 恒等映射,
否则分组重排,两臂在同一定理内证毕)。侧条件:`TILE_N ≤ N`(店 lane
单射)、`0 < TILE_K`(kernel 自己的 `tl.cdiv` 圈数)、干净输入 `hundef`
(掩码载入无 `other`,罩外 lane 读 `undef` 通道 —— `bmm_chunk_fwd`
惯例)。对 `M`、`N`、`K` **无任何**整除假设。

constexpr 掩码特化、三条批偏移参数重赋值(`A += pid_b*M*K` 等)折进指针
瓦片构造、元组赋值 `pid_m, pid_n = pidx, pidy` 拆分、以及
`range(num_iters)` 拼作 `range($(0), num_iters, $(1))`,都是登记在
`proof_blockers.md` 的 `Translation-surface blocker:`。`triton.autotune`
配置扫描与宿主启动是可信边界(`TILE_M/TILE_N/TILE_K/GROUP_M` 保持符号
绑定子)。
