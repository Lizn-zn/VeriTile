# Triton 与 TileLang：语言设计及 VeriTile 验证建模研究

> 研究快照与访问日期：**2026-07-14**。外部源码快照：Triton
> [`daa721bdb145fedc3dee3f5be03d2237f63fe7ae`](https://github.com/triton-lang/triton/tree/daa721bdb145fedc3dee3f5be03d2237f63fe7ae)，TileLang
> [`70548a170747e214ada118bc02a1dc098de4bd81`](https://github.com/tile-ai/tilelang/tree/70548a170747e214ada118bc02a1dc098de4bd81)。仓库快照：本地
> `main@a373519dc250aed3b8b64f9f0af991195d4d2766`，当前分支
> `dny@d2e42586dd64d20f75f012fc3811d4f23ca2f851`。
>
> 证据标签：**[来源事实]** 表示官方文档、官方源码/IR，或本仓库固定分支/提交直接显示的事实；**[源码推断]** 表示由这些定义推出、但上游未明文承诺的含义；**[VeriTile 建议]** 表示本文的建模或工程决策，不冒充上游事实。

## 1. 摘要

1. **两种语言不是同一抽象的不同拼写。** Triton 的核心单位是 1--3D grid 中的 blocked program instance；程序对 shaped tensor/value 和 pointer tensor 操作，编译器再做线程分工、布局、共享内存和异步调度。TileLang 的核心是 TIR `PrimFunc` 内的显式 `T.Kernel` launch、serial/parallel/pipelined loop、global/shared/fragment region 和 tile operator；用户可以直接约束线程、布局、流水线 stage/order 及硬件原语。[来源事实：Triton blocked-program 模型与编译器职责](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/docs/programming-guide/chapter-1/introduction.rst#L15-L56)；[来源事实：TileLang launch、loop、memory scope](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/language_basics.md#L82-L157)。
2. **双 frontend 的 neutral core 应取语义并集，不应取表面交集。** 纯索引映射、广播、同步 `copy` 和同步 GEMM 可 canonicalize 为较小的 load/compute/store；但显式 memory-space tag、线程/并行域、barrier、atomic scope/order、async token、pipeline stage、cluster/peer 可见性若会影响值或可执行性，就必须在 foundation 的 memory/concurrency/launch 契约中保留。[VeriTile 建议]
3. **布局要拆成“逻辑索引语义”和“物理线程布局”。** Triton TTIR 的 ranked tensor/pointer shape 与 TileLang region/view 的逻辑映射决定读写哪个 cell，不能擦除；TritonGPU encoding、TileLang `Fragment` 的 thread/replication mapping、coalescing/swizzle 若只影响实现，应从 algorithm layer 擦除，作为 lowering 证书或外部 test-backed contract。TileLang `LayoutNode` 本身把输入逻辑坐标映射到输出索引，而 `FragmentNode` 另含 thread/replicate 映射，说明两类信息确有分层空间。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/src/layout/layout.h#L46-L100)；[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/src/layout/layout.h#L135-L187)。
4. **`main` 的 deep embedding 已有强 typed shape/dtype 骨架，但其语法、命名和状态仍深度 Triton 化。** `Op : TileDType → TileShape → Type`、`MemAccess`、`Stmt`、`Kernel` 在类型中编码不少不变量；同时 `RegName := String`、Triton 专用 block pointer、flat `RegionName → Nat → MemCell`、顺序 `Option` executor 和仅 program-id 的 launch 状态，使 TileLang 的 lexical buffers、memory scopes、parallel loops、break/continue、async pipeline 很难在不扭曲语义的情况下直接塞入。[来源事实：AST](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Core/Ast.lean#L74-L342)；[来源事实：状态](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Semantics/State.lean#L220-L240)；[来源事实：顺序执行](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Semantics/Step.lean#L106-L214)。
5. **不推荐继续把每个上游 surface intrinsic 手写成 Lean constructor。** 这会让可信转录边界、版本漂移和证明重写成本随两个语言的 API 并集增长；上游已经在变化，例如 Triton 当前把 `make_block_ptr` 标成 deprecated 并引入 tensor descriptor/TMA，而 TileLang 同时有同步/显式异步 copy、WGMMA、TCGEN05 和 cluster copy。[来源事实：Triton deprecation 与 descriptor](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/language/core.py#L2631-L2686)；[来源事实：TileLang copy variants](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/instructions.md#L36-L78)。
6. **推荐“固定上游 IR 快照 + 小型 typed neutral core + 可检查 lowering certificate”的混合架构。** Triton 从前端已产生 TTIR，TileLang 从 parser/builder 已产生 TIR/TileOp；importer 应读取稳定、版本固定的结构化 IR，而不是重新解释任意 Python。Lean 内只信任一个小 checker：验证 imported term 的 typing、shape、region、effect、launch 和 lowering certificate，再运行 neutral semantics。官方编译器可做不可信 producer，differential/round-trip tests 负责发现 importer 漂移。[VeriTile 建议]
7. **algorithm layer、rounding model 与 compute gap 必须继续分工。** 数学矩阵乘、索引和 declared outputs 上的算法关系进入 Lean；cast/accumulation/TF32 等舍入位置进入 rounding model；NaN、subnormal、out-of-range cast、具体 tensor-core approximation、cache/coalescing、硬件 barrier/TMA fidelity 进入 compute gap 或 test-backed contract。Triton 官方明确 `tl.dot` 的 TF32 路径可能截断输入，且 float-to-int 越界行为不具可移植定义。[来源事实：dot precision](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/language/core.py#L2369-L2407)；[来源事实：越界 cast](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/docs/python-api/triton-semantics.rst#L40-L49)。

## 2. 范围与方法

### 2.1 范围

本文比较 Triton 与 TileLang 的：编程抽象和执行模型、shape/type、索引和布局、控制流、内存层次、同步/异步/流水线、矩阵乘和硬件专用原语、编译/IR pipeline、host/launch specialization、扩展与版本演进。目标不是列出完整 API，而是判断差异会落到 VeriTile 的哪一层。

本文使用仓库约定术语：**algorithm layer** 是 Lean 证明的 ℝ/ℤ/ℕ erased view；**rounding model** 只记录舍入事件结构；具体浮点和硬件行为与 algorithm layer 的差异属于 **compute gap**；双语言共用 **foundation**，其中唯一 DSL-agnostic typed IR 是 **neutral core**；Triton/TileLang 只提供 **frontend**；内存以 **region** 和 logical cell offset 表示，scope 以 **memory-space tag** 表示；多 rank peer state 位于 **world layer**。[来源事实：仓库术语契约 `CONTEXT.md`](../../CONTEXT.md#L9-L91)，快照 `dny@d2e42586`。

### 2.2 一手资料口径

- Triton：官方 `triton-lang/triton` 文档、Python language/runtime API、TTIR/TTGIR TableGen 和 backend pass pipeline，固定到 `daa721bd`。
- TileLang：官方 `tile-ai/tilelang` 文档、Python DSL、TIR/TileOp C++ 定义、layout 与 CUDA pass pipeline，固定到 `70548a17`。
- VeriTile：只读检查本地 `main@a373519d` 的 git object（`git ls-tree -r main`、`git show main:<path>`、`git grep ... main`）；另读当前 `dny@d2e42586`，只将其称为“当前分支已有方向/实现”，不称为 `main` 事实。
- 未使用博客、论文转述、搜索摘要。官方文档自身标记 draft 或 API 尚未完整实现处，不提升为稳定承诺。例如 TileLang overview 明示 beginner hardware-unaware interface “not yet fully implemented”。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/get_started/overview.md#L15-L30)。

### 2.3 推断规则

- 上游文档/API 声明 observable semantics 时按来源事实记录。
- pass 名称只证明“编译器执行该转换”，不证明转换的形式正确性；有关语义保持均标为源码推断或 VeriTile 建议。
- performance hint 只有在不改变 algorithm-layer result、终止性、可见性或失败条件时才可 erase；无法证明时默认保留或拒绝，不以“编译器通常会处理”替代契约。

## 3. 两语言对比表

| 维度 | Triton | TileLang | 对验证模型的落点 |
|---|---|---|---|
| 基本执行单位 | 1--3D grid 中的 blocked program instance；`program_id(axis)`/`num_programs(axis)` 暴露 instance 坐标和 grid 大小。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/language/core.py#L1953-L1985) | `T.Kernel(*blocks, threads=...)` 同时形成 block 和 thread bindings；结构化 loop 位于其内。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/language/kernel.py#L277-L340) | core launch contract 至少需 grid/block index；TileLang 若显式 thread index 影响地址，则还需 thread domain 或在 frontend 展开成并行 lane 域。不能只保留 Triton PID。 |
| 值与 shape | scalar 和 ranked tensor；TTIR 类型含 f8/f16/bf16/f32/f64、i1/i4/i8/i16/i32/i64、scalar/tensor pointer。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/include/triton/Dialect/Triton/IR/TritonTypes.td#L18-L92) | `PrimFunc` 参数是 shaped `Tensor`/`Buffer`，shape 可 concrete 或 symbolic；buffer、region 和 scalar `PrimExpr` 并存。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/language_basics.md#L15-L80) | neutral typing 要支持 symbolic dimension constraints、fixed-width integer/low precision tags、scalar vs tile/buffer views。算法 carrier 可 erase，但 dtype/shape witness 不能过早丢失。 |
| promotion/broadcast | 有明确 kind/width/signedness promotion；广播左补 1，再按 equal-or-one 扩展；tensor-scalar 有单独规则。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/docs/python-api/triton-semantics.rst#L6-L37) | 算术基于 TIR dtype；`T.copy` 的 region extent 推导不是通用 tensor broadcast，源码还警告不同 extent/size-1 可能产生意外代码。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/language/copy_op.py#L92-L108) | frontend 必须各自完成 promotion 和 shape elaboration，再输出显式 cast/broadcast/index map；core 不应复刻两套隐式规则。 |
| 索引表示 | pointer arithmetic、tensor of pointers、block pointer/新 tensor descriptor；load/store mask 与 block-pointer boundary check 是不同模式。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/language/core.py#L2497-L2539) | buffer multidimensional indexing、negative index、slice→`BufferRegion`；`T.copy` 把范围编码为 `tl.region` 再 lowering。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/python_compatibility.md#L30-L36)；[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/language/copy_op.py#L92-L108) | canonical form 应是 `region + logical index map + predicate + OOB policy`；descriptor/block pointer 是 frontend 可保留诊断 provenance 的 sugar，除非其异步语义可观察。 |
| 布局 | TTIR ranked tensor 尚无 thread layout；TTIR→TTGIR 时加入 blocked/GPU encoding，再做 coalesce、layout conversion elimination、matmul acceleration、pipeline。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/third_party/nvidia/backend/compiler.py#L245-L337) | `Parallel` 可带 `Fragment` layout，缺省由 LayoutInference 推断；layout validator 要求覆盖 parallel nest 且维度匹配。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/language/loop.py#L13-L71) | 逻辑 view/index map 必须保留；thread ownership、replication、swizzle 若只服务 lowering 则 erase，但需 checker 验证它没有改变覆盖、重叠和 race contract。 |
| 控制流 | 变量离开 `if`/`for` 后必须在所有路径定义，不能像 Python 那样按运行路径动态决定变量是否存在。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/docs/python-api/triton-semantics.rst#L49-L70) | 支持 `if/elif/else`、ternary、serial/unroll/parallel/pipelined、`while`、`break`、`continue`。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/control_flow.md#L7-L12) | core 需 lexical binding/SSA 或 well-scoped local IDs、general loop/while 与 control effects；仅 `String` register + counted loop 不足。compile-time branch 应由 frontend 消除。 |
| 内存层次 | surface 主要是 global pointer/descriptor；共享内存分配与同步多由 TTGIR lowering 决定。[来源事实：编译器自动 shared allocation/sync](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/docs/programming-guide/chapter-1/introduction.rst#L52-L56) | global argument、`alloc_shared`、`alloc_local`、`alloc_fragment`、`alloc_var` 是显式 scoped buffers。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/language/allocate.py#L1-L15) | flat semantic memory可保留，但 region metadata 必须有 memory-space tag、owner/visibility/lifetime；局部 allocation 不能全伪装成 global input/output。 |
| 同步 copy | 普通 load/store 是 pointer operation；cache/eviction/volatile 在 IR/API 中显式，但大多是物理 hint。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/include/triton/Dialect/Triton/IR/TritonOps.td#L214-L314) | `T.copy` observable semantics 是同步，即使 lowering 到 TMA/cp.async 也自动 wait/sync。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/instructions.md#L36-L46) | `T.copy` 可 canonicalize 为读取源 snapshot 后写目标 region；cache/eviction erase。其 lowering 是否真的保持同步是外部 test-backed contract。 |
| 显式 async | Triton 高层普通 kernel 多依赖 compiler pipeline；TTIR 另有 descriptor/TMA abstraction，目标后端插入 async lowering/fence。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/include/triton/Dialect/Triton/IR/TritonTypes.td#L101-L115)；[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/third_party/nvidia/backend/compiler.py#L315-L327) | `T.async_copy` 发出 cp.async+commit 而不自动 wait，消费前必须 wait；跨线程还需 barrier。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/language/copy_op.py#L190-L231) | 显式 async 不能擦成同步 copy 后仍声称验证原程序。core concurrency 需 token/in-flight transfer、wait、barrier 和 visibility；否则 frontend 应明确 unsupported。 |
| 软件流水线 | backend options 包含 `num_stages`，TTGIR pipeline/schedule/warp-specialize passes 按 capability 运行。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/third_party/nvidia/backend/compiler.py#L282-L307) | `T.Pipelined` 可编译器推断或显式 `stage/order/sync/group`；replayable scalar bind 不占 schedule slot。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/language/loop.py#L112-L173) | 若流水线只是顺序程序的 certified schedule，可在 P1 erase 为 loop；若用户显式 async/warp specialization 形成可观察 overlap，则 P3 保留 event/order。Bind replay 需要 lexical SSA，不应复制 effectful load。 |
| GEMM | `tl.dot` 是 2D/3D（高 rank reshape 后处理）matrix product，可带 accumulator、input precision 和 imprecise accumulation 参数。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/language/core.py#L2369-L2441) | `T.gemm(A,B,C)` 直接更新 fragment C；默认接口同步，另有无隐式 wait 的 WGMMA/TCGEN05 variants。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/language/gemm_op.py#L149-L218) | core 应有数学 `matmulAcc`，显式 transpose/shape/accumulator；同步 variant canonicalize，异步 issue/wait 留 concurrency；input precision/cast placement进入 rounding model/compute gap。 |
| atomics | TTIR atomic 带 RMW op、mask、memory semantic（relaxed/acquire/release/acq_rel）和 scope（CTA/GPU/system）。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/include/triton/Dialect/Triton/IR/TritonOps.td#L350-L385)；[枚举](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/include/triton/Dialect/Triton/IR/TritonAttrDefs.td#L21-L30) | TileLang API 有 atomic add/max/min/load/store、memory order，以及 block/grid/warp sync。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/instructions.md#L190-L207)；[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/instructions.md#L221-L226) | RMW value semantics、order、scope 必须保留；只在已证明 commutative/disjoint 的 slice 中 sequentialize。 |
| host specialization | JIT cache key由参数 specialization/options 形成；`constexpr` 值进入 constants；grid callable 在 launch 前求值并规范成 3D。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/runtime/jit.py#L702-L780) | outer JIT/factory 参数把 tile sizes/dtype bake 入 TIR；支持 lazy PrimFunc-return 与 eager builder 两种模式。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/jit/__init__.py#L257-L326) | verified artifact 必须绑定 source revision、specialization constants、target-independent launch expression和选中 config；autotune 搜索过程不进 algorithm semantics。 |
| IR pipeline | Python AST→TTIR→TTGIR→LLVM IR→PTX/CUBIN（NVIDIA）；AMD 后端对应 TTIR→TTGIR→LLVM IR→AMDGCN/HSACO。[来源事实：NVIDIA stages](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/third_party/nvidia/backend/compiler.py#L583-L589)；[AMD stages](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/third_party/amd/backend/compiler.py#L586-L592) | Python parser/builder→TIR+TileOp→pipeline planning/layout inference/LowerTileOp→flatten/vectorize/storage rewrite→host/device split→target source/runtime。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/cuda/pipeline.py#L70-L138)；[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/cuda/pipeline.py#L141-L254) | 最佳 import cut 是语义仍清楚、但 Python metaprogramming 已消失的 TTIR/TIR+TileOp；不要从 PTX 反推 algorithm layer，也不要信任任意 Python 执行。 |
| 扩展/演进 | backend 抽象要求 `supports_target`、options、dialect、stages；TTIR 同时持续增加 descriptor、dot-scaled 等节点。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/backends/compiler.py#L23-L50) | Tile operator 通过 `Lower`、`InferLayout`、`Clone` 与注册宏扩展；同一语言混合 tile library 和 thread primitives。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/src/op/operator.h#L148-L202)；[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/get_started/overview.md#L3-L50) | neutral core 仍应是 closed type；新上游 op 经分类（canonicalize/core node/external/reject）进入 union policy，不能把 frontend extension object塞进 proof surface。 |

## 4. 语义差异如何落入验证模型

### 4.1 可以 erase 或 canonicalize

以下结论是 **[VeriTile 建议]**；“可擦除”均以 frontend 生成并通过 checker 的 side condition 为前提。

| 上游构造 | neutral form | 必需 side condition |
|---|---|---|
| Triton cache modifier/eviction policy；TileLang coalesced width、rasterization、swizzle hint | 删除，只留 provenance | 不改变 logical address、value、termination、memory order；性能差异交给外部测试。 |
| Triton tensor of pointers；TileLang BufferRegion/slice | `region + indexMap + lane predicate` | element dtype、rank、extent、OOB policy 已检查；alias 显式。 |
| Triton `make_block_ptr`/`advance`；TileLang view/reshape | logical view descriptor 或归一化 index map | element count/byte reinterpret side condition；纯 view 不制造 copy。 |
| TileLang 同步 `T.copy` | source snapshot 的 pointwise load/store | source/destination region 和 extents明确；overlap 时定义 copy order或拒绝。官方只承诺 statement 后可消费目标，不替 VeriTile 决定 overlap 语义。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/instructions.md#L40-L46) |
| `T.serial`/Triton ordinary loop | general core loop | bounds/step/termination已 elaborated；unroll annotation删除。 |
| 编译时 `constexpr`/outer Python branch | 选中分支和 concrete constants | artifact 记录 specialization environment，重放可得到相同 IR。 |
| 同步 `tl.dot` / `T.gemm` | `matmulAcc` | shape、transpose、accumulator dtype 和 mathematical accumulation order已声明。 |
| 自动插入的布局、pipeline、barrier | P1 中擦为顺序 algorithm program | 上游 lowering 是 external contract；若验证目标是 async correctness，则不能擦除。 |

### 4.2 必须保留到 neutral core 或 foundation contract

以下是 **[VeriTile 建议]**。

1. **Typed value universe：** bool、fixed-width signed/unsigned integer provenance、real-valued floating channels、pointer/view、shape expression和scalar/tile distinction。algorithm carrier 可映射到 ℤ/ℕ/ℝ，但溢出、bitcast 和 packing 不能静默当普通整数/实数。
2. **Shape/index evidence：** symbolic dimension environment、rank、broadcast witness、reshape element-count witness、axis bounds、index map、predicate、OOB result（undefined/zero/NaN/skip/fail）。
3. **Region metadata：** element dtype、logical extent、memory-space tag（global/shared/fragment/local/tmem 等按支持集扩展）、owner domain、lifetime、read/write permission。semantic memory 仍可 flat；tag 供 visibility/safety rule 使用。
4. **Binding/control：** locally nameless/de Bruijn 或 typed SSA IDs、block arguments/phi、lexical allocation scope、if/loop/while、break/continue/return effect。frontend 不应把 effectful expression靠复制“简化”。
5. **Launch：** grid rank/dims、program/block index、thread-domain size、cluster dims、specialization constants；需要区分 “program instance” 与 “thread/lane”，否则 TileLang thread-derived address无法解释。
6. **Concurrency：** atomic RMW op、memory order/scope、barrier participant scope、async issue/token/wait、pipeline stage/iteration、happens-before；只有有证明的 disciplined slice 才能映射回 sequential execution。
7. **World layer：** cluster/peer/multi-rank address 和 signal 不进入单-rank `BlockState`；TileLang `copy_cluster` 已有 destination CTA、cluster mask 和 remote barrier 参数，说明未来至少有 cluster-level peer state需求。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/language/copy_op.py#L137-L187)。
8. **Rounding events：** explicit cast、store narrowing、GEMM input precision/accumulator、fast/IEEE math selection；只记录位置和结构，不在 Lean 内承诺硬件误差幅度。

### 4.3 应放在 test-backed contract / compute gap

以下边界是 **[VeriTile 建议]**，外部事实只用于说明为何不能在 algorithm layer 中偷渡。

- IEEE rounding、NaN、subnormal、overflow、out-of-range float→int、bit-level reinterpret；Triton 官方明确后者跨 compiler/interpreter/backend/toolkit 可不同。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/docs/python-api/triton-semantics.rst#L43-L47)。
- TF32/BF16x3/FP8/FP4 等具体 tensor-core 路径和误差；TTIR `DotOp` 的 input precision 决定硬件路径，硬件不存在或输入非 f32 时可被忽略。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/include/triton/Dialect/Triton/IR/TritonOps.td#L665-L700)。
- cache、coalescing、bank conflict、register pressure、shared-memory allocation大小、occupancy、性能和 autotune winner。
- 上游 pass 是否正确保持同步 `T.copy`、自动 OOB guard、layout inference和自动 barrier；这些 compiler correctness 主张不能由 VeriTile 的 kernel theorem反向推出。
- CUDA/HIP/PTX 对具体 async instruction、mbarrier、warpgroup、cluster visibility 的硬件 fidelity；Lean concurrency model最多证明抽象 event contract。
- importer 对官方 IR 的 parser/serializer compatibility、source→IR transcription fidelity；以固定 revision corpus、round-trip 和 differential execution检查。

## 5. 双 frontend 验证适配清单

### 5.1 Neutral core 的最小正确边界

| 面 | 最小新增/调整 | 不应进入 core |
|---|---|---|
| `Type` | scalar kind、algorithm carrier、source dtype tag、shape expr；pointer/view type；低精度 packing标记 | `tl.float16`/`T.float16` 两套构造名；backend class。 |
| `Expr` | literal/ref/cast、arith/compare/select、shape/index map、load、reduction/scan、matmulAcc | cache hint、warp policy、Python syntax provenance。 |
| `Stmt` | let/assign/store/copy、allocate/free scope、if/general loop/while/control effect、atomic、async issue/wait/barrier | 每个 Triton/TileLang API 一个 constructor。 |
| `Kernel` | typed parameters、declared inputs/outputs、local regions、launch contract、specialization environment | frontend 名称作为 semantic discriminant。provenance 只放 metadata/diagnostics。 |
| State | region memory、typed locals/SSA env、launch indices；P3 扩展 event/token/scheduler state | compiler cache、autotuner状态、PTX寄存器分配。 |
| Memory | logical cells + region metadata + alias/view relation + footprints/frame | byte-level具体地址，除非处理 bitcast/packed dtype 的独立 compute projection。 |
| Concurrency | scope/order、HB、barrier phase、in-flight copy、disciplined interleaving | 通用 GPU ISA semantics；未支持 op 明确拒绝。 |
| Correctness | 所有 frontend 共用同一 execution/refinement/fusion surface | `TritonCorrect`、`TileLangCorrect` 两套 theorem。 |

此边界遵守 union policy：任何一个支持 frontend 中具有干净 algorithm-layer semantics 的 construct都可进入一个 closed neutral type；不是只保留两语言共同构造，也不是让 frontend 注入任意 semantic node。[来源事实：仓库 union policy](../../CONTEXT.md#L54-L69)，`dny@d2e42586`。

### 5.2 Triton frontend lowering 职责

1. 固定 source/compiler revision、JIT signature、constexpr、target-neutral options和 launch grid；记录选中 autotune config，但不把搜索过程放入语义。
2. 从 TTIR 读取 ranked type、pointer address space、SSA region/control flow、load/store/atomic/reduce/scan/dot；不从 Python 文本猜 shape。
3. 实现 Triton promotion、broadcast、scalar-vs-tensor division规则；官方明确 mixed-sign integer tensor division趋零，而纯 scalar 又遵循 Python，frontend 需在 IR 已定型后导入明确 op。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/docs/python-api/triton-semantics.rst#L40-L45)。
4. 把 pointer/block pointer/tensor descriptor降为 checked view/index map；保留 OOB padding、undefined lane和mask store semantics。
5. 把 `DotOp`/`DotScaledOp` 拆为 algorithm matmul、rounding event metadata和 compute-gap obligation；不把 TF32 当精确 ℝ 事实。
6. 识别 inline asm、extern elementwise、bitcast、unsupported descriptor/atomic/concurrency op并结构化失败；TTIR 允许 external elementwise region，不能默认视作纯实函数。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/include/triton/Dialect/Triton/IR/TritonOps.td#L823-L866)。

### 5.3 TileLang frontend lowering 职责

1. 固定 TileLang/TVM revision、target-independent TIR/TileOp、outer factory constants、shape bindings、`T.Kernel` grid/thread/cluster attrs和 pass config。
2. 从 TIR lexical buffers导出 region declaration、memory-space tag、extent、view/alias和 allocation lifetime；global argument与local shared/fragment不得混同。
3. 把 serial/unroll/parallel/pipelined/while和 control effects降到 neutral control；`Parallel` 必须给出覆盖/独占或允许 race 的 contract。
4. 将同步 `T.copy` canonicalize；将 `async_copy`、TMA、mbarrier、warpgroup、cluster copy降为 concurrency event，或在 P1 明确 unsupported，而不是悄悄同步化。
5. 对 `T.gemm` 输出 `matmulAcc`，保留 transpose、accumulator和同步 completion；对 WGMMA/TCGEN05 issue形式保留 token/wait requirement。
6. 校验 layout logical mapping与 parallel domain覆盖；物理 `Fragment` thread/replicate mapping默认不进 algorithm layer，但作为 race/disjointness certificate 的输入。
7. 对自动 OOB legalization生成 predicate/guard certificate；官方说 pass 会在无法证明安全时加 guard，但该 pass 本身不在 Lean 信任链内。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/control_flow.md#L41-L45)。
8. Tile operator registry可扩展，unknown `tl.tileop.*` 必须 fail closed；不能借 C++ virtual `Lower` 结果来假定 high-level op语义。[来源事实：operator interface](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/src/op/operator.h#L148-L202)。

### 5.4 Lowering checker 的共同输出

**[VeriTile 建议]** 每个 imported kernel 至少携带：

- `SourceId = {dsl, repository, commit, path, symbol, sourceHash}`；
- `Specialization = {constants, symbolicDims, selectedConfig, targetClass}`；
- typed neutral kernel；
- source-location provenance map（只用于诊断）；
- region table（dtype、extent、memory-space tag、owner/lifetime）；
- launch contract；
- unsupported/erased/external obligation清单；
- lowering certificate：type/shape/index、effect、allocation scope、parallel coverage、async discipline；
- rounding-event catalog和 compute-gap policy；
- importer schema version和上游 IR schema fingerprint。

## 6. `main` deep embedding 专项检查

### 6.1 检查对象与现状

本节只陈述本地 `main@a373519d`。`main` 的核心文件是
`VeriTile/Triton/Core/{Types,Shape,Ast}.lean`、`VeriTile/Triton/DSL/*`、
`VeriTile/Triton/Semantics/*`、`VeriTile/Triton/Memory/*` 和
`VeriTile/Triton/Launch/*`。`git show` 统计显示 `Core/Ast.lean` 1,013 行、
`DSL/Syntax.lean` 317 行、`DSL/Typing.lean` 250 行、`Semantics/EvalOp.lean`
657 行、`Semantics/Step.lean` 916 行；这些数字只描述该 commit，不代表质量或完整度。[来源事实：`main@a373519d` 路径与符号](https://github.com/Lizn-zn/VeriTile/tree/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton)。

`Op : TileDType → TileShape → Type` 将 dtype/shape放进索引，`Broadcast` witness、`Fin shape.length` axis和 typed `MemAccess` 排除了多类 ill-typed term；`TileShape := List Nat`，`TileIndex` 是 shape-indexed product，tile carrier是索引到值的函数。[来源事实：typed AST](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Core/Ast.lean#L74-L291)；[来源事实：shape/index](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Core/Shape.lean#L14-L30)。

另一方面，surface macro 自行维护 `DInfo`/`SInfo`/string-keyed `Env`，shape相等依赖 syntax term string，broadcast elaboration在 macro-time重新实现；这不是从官方 TTIR 导入。[来源事实：macro typing](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/DSL/Typing.lean#L12-L56)；[来源事实：broadcast elaboration](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/DSL/Typing.lean#L176-L250)。因此当前 proof证明的是手工/宏生成的 embedded term，而不是自动建立真实 Python/TTIR 与该 term 的等价。[源码推断]

### 6.2 双 frontend 下的具体困难

#### AST 规模与版本漂移

- `Op` 已将 Triton arithmetic、math、reduce/scan、dot、view、pointer、block pointer和load逐个封闭列举；`Stmt` 又逐个列举 store/atomic/loop/if。[来源事实](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Core/Ast.lean#L74-L342)。TileLang 若照此加入 allocation、copy、parallel、pipeline、barrier、TMA/WGMMA/TCGEN05，会使每个 evaluator、frame theorem、rounding evaluator和macro traversal同步扩张。[源码推断]
- 上游并非静态：Triton当前已有 `dot_scaled`、tensor descriptor且 deprecate block pointer；TileLang operator通过registry持续扩展。因此“surface API constructor集合”不是稳定 proof boundary。[来源事实：Triton](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/include/triton/Dialect/Triton/IR/TritonOps.td#L665-L755)；[来源事实：TileLang](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/src/op/operator.h#L187-L202)。

#### Type/shape invariants

- GADT很好地保证构造后 invariant，但 macro前端先用 `DInfo/SInfo` 重做一套弱类型 elaboration；symbolic dimension相等是 `toString` syntax key，而不是证明等式。[来源事实](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/DSL/Typing.lean#L12-L56)。
- TileLang shape是 TIR `PrimExpr`，可来自 buffer shape、dynamic var和算术表达式；固定 `List Nat` 无法直接表示运行时 symbolic shape constraint。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/language_basics.md#L36-L80)。若每个 dynamic size变成 Lean theorem参数仍可工作，但 importer需要显式 constraint context，而非 syntax string equality。[VeriTile 建议]
- `TileDType` 将所有 signed width折到 `.int`、unsigned width折到 `.nat` 的 frontend做法会丢失溢出、packing和合法 instruction选择；这可用于 algorithm carrier，但不能充当完整 source typing。[来源事实：main dtype expansion](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/DSL/Typing.lean#L103-L139)。

#### 绑定与控制流

- `RegName := String`，`ref`/`assign`按名字访问；`BlockState.setReg` 在新 dtype/shape赋值时清除同名其他 typed slot。[来源事实](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Core/Types.lean#L10-L14)；[来源事实](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Semantics/State.lean#L264-L283)。这不直接表达 lexical shadowing、SSA block args或TileLang buffer allocation lifetime。[源码推断]
- `Stmt` 只有 counted/range loop和scalar condition，没有 while/break/continue；TileLang官方支持这些 control effects。[来源事实：main](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Core/Ast.lean#L299-L326)；[来源事实：TileLang](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/control_flow.md#L112-L130)。靠宏展开所有路径会放大 term并使 loop invariant难复用。[源码推断]
- TileLang pipeline对 replayable `Bind` 与依赖 pipeline-produced buffer 的 `Bind` 区别处理；简单 substitution若复制后者会复制内存读并改变schedule。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/software_pipeline.md#L116-L208)。

#### Layout/index semantics

- `main` 的 `Op.remap` 直接存 Lean函数 `TileIndex outShape → TileIndex shape`，proof方便但不可序列化、难以由外部 IR稳定导入，也难审计其来自哪段source index expression。[来源事实](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Core/Ast.lean#L211-L236)。
- `BlockPtr` 自带 parentShape/blockShape/strides/offsets，但注释承认其 region仍固定 real、typed block pointer后移；这不足以统一 TileLang任意 dtype scoped buffer/view。[来源事实](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Core/Types.lean#L75-L94)。
- Triton逻辑 tensor通常不把thread ownership放进TTIR，而TileLang可显式给`Parallel`整个nest附 `Fragment` mapping；将二者都压成row-major tile会失去 parallel coverage/race证据，将physical mapping都放进value semantics又会污染算法证明。[源码推断；上游证据见对比表布局行]

#### Memory spaces

- `Kernel.inputs/outputs` 只是 metadata，executor 的 memory是单个 `RegionName → Nat → MemCell`；没有 allocation、memory-space tag、owner或lifetime。[来源事实：Kernel](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Core/Ast.lean#L328-L342)；[来源事实：BlockState](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Semantics/State.lean#L220-L240)。
- TileLang显式区分 shared/local/fragment，且 shared是block-visible、fragment是per-thread/register-layout对象。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/language_basics.md#L119-L133)。若完全擦除tag，无法陈述跨线程可见性、barrier必要性和local region framing。[源码推断]

#### Async/concurrency/launch

- `exec` 是 `List Stmt` 上的 deterministic sequential `Option`；atomic也在同一状态直接fold，`BlockState`无token、phase、thread、scheduler。[来源事实](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Semantics/Step.lean#L106-L214)；[来源事实](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Semantics/Step.lean#L398-L399)。它适合P1顺序algorithm execution，不足以解释TileLang显式async copy或barrier。[源码推断]
- `main` launch的 `Grid`/`GridIndex` 只向state注入program IDs；没有threads/block dims/cluster dims。[来源事实](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Launch/Grid.lean#L12-L80)。TileLang `T.Kernel` 同时暴露block/thread bindings，必须扩展launch contract或先做有证据的parallel-loop lowering。[源码推断]
- `main` TTIR-like atomic node没有Triton官方IR已有的memory semantic/scope字段；双frontend下若继续省略，会把acquire/release/CTA/GPU/system程序合并成同一term。[来源事实：main node](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Core/Ast.lean#L299-L309)；[来源事实：official TTIR](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/include/triton/Dialect/Triton/IR/TritonOps.td#L350-L385)。

#### Frontend provenance 泄漏与 proof maintenance

- AST注释、namespace、constructor语义和DSL语法直接写 `tl.*`；当前 `dny` 虽把机械基础迁到 `VeriTile/Core`、`Semantics` 等，并把Triton宏放到 `VeriTile/Frontend/Triton`，但 `Core/Ast.lean` 的注释和节点仍明显沿用Triton，且没有 TileLang frontend。前者是当前分支已有迁移方向，后者是未完成状态，不能称双frontend已经落地。[来源事实：当前分支 `VeriTile/Core/Ast.lean`](../../VeriTile/Core/Ast.lean#L1-L19)，`dny@d2e42586`；[来源事实：当前分支 frontend 目录](../../VeriTile/Frontend/Triton/DSL.lean)，`dny@d2e42586`。
- 每加一个constructor至少会触及 `evalOp`/`stepStmt`、rounding evaluator、frame/footprint、macro inference和大量simp theorem；deep embedding索引越强，非法term越少，但dependent pattern matching与proof rewrite维护成本越高。[源码推断：受影响路径可见 `main@a373519d` 的 `VeriTile/Triton/{Semantics,Memory,Float,DSL}`。]
- provenance若成为core sum type（`.tritonDot` vs `.tilelangGemm`）会产生两套等价semantics和proof；按union policy，应统一成 `.matmulAcc`，source location留metadata。[VeriTile 建议]

#### 转录可信边界

- `main` 的 Lean macro解析“像 Triton 的语法”，但不执行官方 Triton parser/JIT/semantic lowering；因此无法自行证明真实 Python source、specialization和TTIR与Lean term一致。[源码推断：`main` surface是Lean `declare_syntax_cat`/macro，而非官方IR reader；见](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/DSL/Syntax.lean#L1-L40)。
- TileLang还允许outer Python factory、lazy/eager JIT、TIR parser/builder和registered tile operators；再手写一套Lean surface会增加第二个独立transcription boundary。[来源事实](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/jit/__init__.py#L257-L326)。

## 7. Modeling 方案比较

### 7.1 方案 A：继续手写全量 deep embedding

**形状：** 扩展现有 GADT；为 TileLang 新写 Lean syntax/macro；每个两语言 intrinsic手工lower到共享constructor或新增constructor。

**优点：** Lean kernel term可读；类型索引早期排除错误；现有proof和simp资产复用最多；不需要外部IR parser即可做小例子。

**缺点：** 真实source→Lean term完全是人工/宏可信边界；surface drift直接扩张AST与proof；TileLang symbolic TIR、allocation、parallel/pipeline很难自然表达；两套macro inference容易与官方promotion/typing漂移。

**适用：** 保留为worked example、golden fixture和importer oracle，不作为规模化 ingestion 主路径。[VeriTile 建议]

### 7.2 方案 B：typed intrinsic / HOAS 或 shallow-ish elaborator

**形状：** Lean API以typed buffer/tile combinator和HOAS binder表示kernel；elaborator将Triton/TileLang surface直接执行成Lean函数或小语句IR，尽量少存syntax tree。

**优点：** binding/shape在Lean elaboration中自然；手写kernel ergonomics好；对数学纯操作proof短；比constructor-per-API更容易canonicalize。

**缺点：** HOAS函数难序列化、比较、位置映射和独立检查；effect/control/async的shallow interpretation容易把source evaluation order藏进meta-level；自动import官方IR仍需reification；trusted elaborator变大。现有`Op.remap`内嵌Lean函数已显示类似审计问题。[源码推断；main证据](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Core/Ast.lean#L227-L236)。

**适用：** 作为neutral core的builder/notation层，不作为唯一存储格式或可信交换格式。[VeriTile 建议]

### 7.3 方案 C：从官方 IR/AST 自动导入小型 neutral core

**形状：** Triton importer消费固定revision TTIR/MLIR；TileLang importer消费固定revision TIR+TileOp JSON/text/FFI dump；二者输出同一first-order typed neutral core和obligation manifest。

**优点：** Python metaprogramming、constexpr和官方typing已在cut point前完成；可保存SSA、source loc、shape、memory effect；双frontend共享semantics/proof；版本变化由schema adapter和fixtures隔离；批量自动化最佳。

**缺点：** parser/importer和上游compiler成为transcription TCB或需额外checker；TTIR/TIR并非稳定标准ABI；过低cut point会吸入physical layout，过高cut point会保留unknown tile op；需要fail-closed coverage catalog。

**适用：** 推荐的生产ingestion主路径，但必须配方案D的certificate checker，而非直接信任importer。[VeriTile 建议]

### 7.4 方案 D：IR importer + untrusted canonicalizer + Lean checked certificate（推荐）

**形状：** C方案的producer同时输出typed neutral term与局部lowering certificate；Lean内小checker重算/验证 type、shape、index map、effect、region、launch和支持集。复杂canonicalization（pointer→view、copy→loop、parallel coverage）由外部工具提出，Lean checker只验关系。

**优点：** 缩小可信计算基；保留自动化和source provenance；允许frontend快速随版本演进，同时core保持closed；proof针对canonical neutral node；失败可定位为unsupported/import/certificate/proof/compute-gap。

**缺点：** certificate language和checker前期成本最高；需要为control/alias/parallel设计可检查witness；并发certificate不能一步到位。

**适用：** 中长期foundation；P1先覆盖纯顺序slice，P3再加event certificate。[VeriTile 建议]

### 7.5 决策矩阵

评分：5最佳（TCB越小、保真/复用/自动化越高、漂移越易控制、成本越低则分越高）。评分是 **[VeriTile 建议]**，不是来源事实。

| 方案 | TCB 小 | 语义保真 | Proof ergonomics | 双语言复用 | 版本漂移 | 实现成本 | 自动化 | 总体 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A 手写 deep embedding | 3 | 2 | 4（小例）/2（全量） | 2 | 1 | 3（起步）/1（长期） | 1 | 16/35 |
| B typed intrinsic/HOAS | 2 | 3 | 4 | 3 | 2 | 3 | 2 | 19/35 |
| C 官方 IR 自动导入 | 2 | 4 | 4 | 5 | 3 | 2 | 5 | 25/35 |
| D IR + checked certificate | 5 | 5 | 4 | 5 | 4 | 1 | 5 | **29/35** |

关键取舍：方案A的“Lean里看起来最可信”不等于真实source最可信；如果source transcription靠人，GADT只证明转录后的term well-typed。方案C/D把最危险的边界显式化并可持续测试/检查。[VeriTile 建议]

## 8. 推荐架构

### 8.1 四层而非两套语义

1. **Source artifact layer（非可信）：** Triton Python+JIT specialization或TileLang factory+PrimFunc，固定revision/hash/config。
2. **Frontend import layer（非可信 producer）：** TTIR adapter与TIR/TileOp adapter；输出neutral term、provenance、obligation和certificate。frontend不拥有semantic node/theorem。
3. **Foundation（Lean可信核心）：** closed typed neutral core、checker、deterministic P1 semantics、region/frame、rounding model、launch；另有P3 concurrency/world layer，不污染普通P1 executor。
4. **External contracts：** official compiler execution differential tests、compute gap、hardware/target support、performance和import round-trip。

### 8.2 建议的 core 形状

**[VeriTile 建议]** 采用 first-order typed SSA/block IR，而不是继续以`String` mutable register为唯一binding：

```text
DType       := Bool | Int(width,signed) | Float(sourceTag) | Ptr(region,dtype)
ShapeExpr   := Const n | Sym id | Add/Mul/CeilDiv ...
ValueType   := Scalar DType | Tile DType Shape
IndexExpr   := affine-ish arithmetic + checked fallback expression
View        := {region, logicalShape, indexMap, predicate, oobPolicy}
RegionDecl  := {dtype, extent, memorySpace, owner, lifetime}
Expr        := pure first-order ops | load View | reduce | scan | matmulAcc
Stmt        := let | store | allocate | if | loop | while | control
             | atomic(order,scope) | asyncIssue | wait | barrier
Kernel      := params + regions + launch + blocks + outputs + metadata
```

不要把 index language 限成 affine：真实 pointer gather、indirect index和dynamic bounds需要一般整数表达式；可以把“可证明 affine”的子类作为优化/自动化 witness，而非core语法边界。[VeriTile 建议]

### 8.3 两种执行语义

- **P1 sequential semantics：** ordinary loads/stores、synchronous copy、pure matmul、structured control；保持现有`Option`/`Except` fail-closed风格，但错误应区分 ill-typed（checker拒绝）、undefined source behavior、unsupported和runtime precondition failure。
- **P3 event semantics：** per-agent control state、in-flight token、barrier phase、atomic/HB、world layer；通过discipline theorem将成功interleaving refinement到P1 sequential projection。不能把所有async节点塞进`stepStmt`后仍称普通block execution。[VeriTile 建议]

### 8.4 Proof surface

- kernel correctness、fusion correctness、frame/locality和rounding theorem只接受checked neutral kernel；frontend provenance不出现在命题主类型。
- 每个import artifact有定理或checker soundness结论：“certificate通过 ⇒ neutral term满足well-typed/well-scoped/well-regioned contract”；不需要在每个kernel proof重复shape plumbing。
- source fidelity不伪装成Lean theorem：`sourceHash + officialIRHash + importerVersion + differentialTests` 是外部evidence；报告verdict分别呈现algorithm proof、rounding warnings和compute-gap status。[VeriTile 建议]

## 9. 分阶段路线

### 阶段 0：冻结边界与 corpus（1个垂直切片）

1. 选同一计算的真实 Triton、TileLang kernel，限制为pointwise或简单matmul，无显式async/while/bitcast。
2. 固定上游commit、source path、specialization、target-neutral IR dump格式。
3. 写construct coverage manifest：`canonicalized / core / external / unsupported`，未知op fail closed。
4. 验收：两个frontend输出alpha-equivalent或可证明equivalent的neutral kernel，共用一个headline theorem。

### 阶段 1：P1 typed import

1. 引入symbolic shape context、typed SSA ID、serial control、view/index map和region table。
2. TTIR importer覆盖 arithmetic、broadcast、load/store/mask、reduce、dot、program IDs。
3. TIR/TileOp importer覆盖BufferRegion、serial/parallel pointwise、同步copy、shared/fragment allocation、同步gemm。
4. checker验证type/shape/index/allocation/launch；保留现有deep embedding例子作为cross-check，不批量重写所有proof。

### 阶段 2：Memory/layout/launch

1. 加memory-space tag、owner/lifetime和view alias；flat memory与region theorem保持transport关系。
2. TileLang `Parallel`输出coverage/disjointness certificate；Triton PID与TileLang block/thread launch统一到launch contract。
3. 擦除physical layout前验证logical index preservation；增加layout/index differential fixtures。

### 阶段 3：Rounding/compute gap

1. importer生成cast/store/GEMM/fast-math rounding event。
2. `tl.dot` precision、TileLang accum dtype和explicit IEEE/fast op进入catalog。
3. 外部checker按source/backend/config执行；不把数值magnitude theorem塞入algorithm layer。

### 阶段 4：P3 async/concurrency

1. 先支持TileLang `async_copy + wait + shared barrier`最小slice；token和finite-prefix safety，不承诺liveness。
2. 再支持atomic order/scope、pipeline stage/iteration；证明disciplined schedule refinement到sequential result。
3. 最后加cluster/peer operation和world layer；单GPU P1 kernel类型不携带world state。

### 阶段 5：迁移与收敛

1. 新kernel默认经IR importer；手写Lean DSL仅用于fixture/最小example。
2. 当import覆盖和checker稳定后，逐步把现有Triton frontend macro降为同一builder，不一次性删除已有proof资产。
3. 每次上游revision升级运行schema diff、unsupported diff、IR round-trip、semantic differential和proof regression；只有manifest审阅后更新固定revision。

## 10. 风险与待验证问题

1. **TTIR/TIR序列化稳定性未知。** 官方源码显示内部IR和pass pipeline，但未在所读一手资料中找到跨版本稳定交换格式承诺。需原型比较MLIR bytecode/text、TIR JSON和FFI结构，选择可版本化schema；当前结论标为未知。
2. **TileLang同步 `T.copy` 的overlapping src/dst顺序未明确。** 官方资料承诺statement后目标可用，但未明确任意overlap的memmove/memcpy/并行lane行为。首版应要求disjoint或相同pointwise mapping，否则unsupported。
3. **TileLang `Parallel` 的算法级race规则需确认。** 文档说明layout validator和可选race check pass，但未找到完整language memory model；在确认前，frontend只能接受可证明disjoint writes、atomic或只读共享。[来源事实：可选 VerifyParallelLoop pass](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/cuda/pipeline.py#L78-L80)。
4. **Triton generic combine region覆盖不足。** 官方TTIR reduce/scan带自定义combine region；现有main只用closed `ScanOp`及少数reduce。需要决定导入受限pure scalar lambda还是先白名单combiner。[来源事实](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/include/triton/Dialect/Triton/IR/TritonOps.td#L760-L818)。
5. **Undefined lane与OOB policy。** Triton masked load `other=None`产生undefined值；TileLang自动guard可能跳过访问。需区分nondeterministic value、poison-like不可观察值、skip和zero padding，不能统一成0。[来源事实：Triton undefined masked lane](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/language/core.py#L2522-L2531)。
6. **Low-precision packing/byte alias。** TileLang支持float8/6/4和vector pack，layout reshape还处理不同element bit width共享physical slot；当前logical-cell region不足以证明bitcast/packed alias。可先作为compute-only op拒绝，后续单独加byte-view bridge。[来源事实：dtype families](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/type_system.md#L13-L39)；[来源事实：layout rescale](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/src/layout/layout.h#L84-L100)。
7. **Host specialization identity。** Triton JIT和TileLang factory/cache key包含不同信息；需定义artifact identity，避免证明一个config却运行另一个config。autotune应返回“选中config + hash”，不是抽象地证明autotuner。[来源事实：Triton cache specialization](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/runtime/jit.py#L738-L757)；[来源事实：TileLang cache内容](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/autotuning.md#L198-L208)。
8. **Compiler-generated barrier信任。** TileLang CUDA pipeline在LowerTileOp后执行ThreadSync和fence injection；Triton backend也插pipeline/fence。VeriTile可验证source-level sequential semantics，但不能据此宣称backend barrier正确，除非外部合同覆盖或另做compiler verification。[来源事实：TileLang passes](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/cuda/pipeline.py#L220-L239)；[来源事实：Triton fence insertion](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/third_party/nvidia/backend/compiler.py#L315-L327)。
9. **当前分支中立化尚未完成语义去品牌化。** `dny` 已机械迁移路径并有Triton frontend目录，但core注释/节点仍有Triton专用内容，也没有TileLang frontend或memory-space tag。应把它作为迁移起点，不作为已验收architecture。[来源事实：当前分支](../../VeriTile/Core/Ast.lean#L1-L57)，`dny@d2e42586`；本地搜索未发现memory-space定义。
10. **`main` 与当前分支基线不同。** 本报告按用户要求检查本地`main@a373519d`；工作树当前是`dny@d2e42586`，且远端tracking ref可能前进。实施前必须重新记录branch heads和merge-base，不把本报告SHA称永久current。

## 11. 来源清单

### 11.1 Triton 官方

- [Programming Guide: blocked program / compiler responsibilities](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/docs/programming-guide/chapter-1/introduction.rst#L15-L56)
- [Triton Semantics: promotion, broadcasting, division, cast, scope](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/docs/python-api/triton-semantics.rst#L4-L70)
- [Language core: program IDs, arange, dot, load/store, descriptors](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/language/core.py#L1953-L2039)
- [Language core: dot and precision](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/language/core.py#L2363-L2488)
- [Language core: memory and descriptors](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/language/core.py#L2491-L2686)
- [TTIR types](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/include/triton/Dialect/Triton/IR/TritonTypes.td#L18-L115)
- [TTIR load/store/atomic](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/include/triton/Dialect/Triton/IR/TritonOps.td#L196-L415)
- [TTIR program ID/dot/reduce/scan](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/include/triton/Dialect/Triton/IR/TritonOps.td#L623-L818)
- [TTIR memory/order/precision enums](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/include/triton/Dialect/Triton/IR/TritonAttrDefs.td#L6-L137)
- [JIT specialization and launch](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/runtime/jit.py#L628-L780)
- [Autotuner](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/runtime/autotuner.py#L19-L100)
- [NVIDIA compiler pipeline](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/third_party/nvidia/backend/compiler.py#L245-L384)
- [Backend extension interface](https://github.com/triton-lang/triton/blob/daa721bdb145fedc3dee3f5be03d2237f63fe7ae/python/triton/backends/compiler.py#L23-L50)

### 11.2 TileLang 官方

- [Overview: interfaces, flow, tile/memory model](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/get_started/overview.md#L3-L84)
- [Language basics](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/language_basics.md#L15-L223)
- [Control flow](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/control_flow.md#L7-L143)
- [Type system](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/type_system.md#L7-L41)
- [Instructions: copy/async/GEMM/sync/atomics](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/instructions.md#L14-L109)
- [Software pipeline semantics](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/software_pipeline.md#L19-L86)
- [Python subset](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/python_compatibility.md#L17-L59)
- [`T.Kernel` launch API](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/language/kernel.py#L149-L340)
- [`Parallel` / `Pipelined`](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/language/loop.py#L13-L173)
- [Memory allocation scopes](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/language/allocate.py#L1-L77)
- [`copy` / `async_copy`](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/language/copy_op.py#L54-L231)
- [GEMM variants](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/language/gemm_op.py#L149-L278)
- [Layout/Fragment IR](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/src/layout/layout.h#L43-L194)
- [Tile operator interface](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/src/op/operator.h#L148-L202)
- [CUDA lowering pipeline](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/cuda/pipeline.py#L70-L254)
- [JIT modes](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/tilelang/jit/__init__.py#L257-L326)
- [Autotuning](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/programming_guides/autotuning.md#L1-L12)
- [Targets](https://github.com/tile-ai/tilelang/blob/70548a170747e214ada118bc02a1dc098de4bd81/docs/get_started/targets.md#L1-L49)

### 11.3 VeriTile 仓库

- [`CONTEXT.md`](../../CONTEXT.md)，`dny@d2e42586`：algorithm layer、compute gap、rounding model、foundation、neutral core、union policy、frontend、region、memory-space tag、world layer。
- [`main: VeriTile/Triton/Core/Ast.lean`](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Core/Ast.lean)，`main@a373519d`：deep embedding AST。
- [`main: VeriTile/Triton/Core/Types.lean`](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Core/Types.lean)，`main@a373519d`：dtype、region、block pointer、carrier。
- [`main: VeriTile/Triton/Core/Shape.lean`](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Core/Shape.lean)，`main@a373519d`：shape/index/broadcast。
- [`main: VeriTile/Triton/DSL/Syntax.lean`](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/DSL/Syntax.lean)，`main@a373519d`：Lean-embedded Triton surface。
- [`main: VeriTile/Triton/DSL/Typing.lean`](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/DSL/Typing.lean)，`main@a373519d`：macro-time dtype/shape inference。
- [`main: VeriTile/Triton/Semantics/State.lean`](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Semantics/State.lean)，`main@a373519d`：flat region state。
- [`main: VeriTile/Triton/Semantics/Step.lean`](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Semantics/Step.lean)，`main@a373519d`：deterministic sequential executor。
- [`main: VeriTile/Triton/Launch/Grid.lean`](https://github.com/Lizn-zn/VeriTile/blob/a373519dc250aed3b8b64f9f0af991195d4d2766/VeriTile/Triton/Launch/Grid.lean)，`main@a373519d`：program-grid surface。
- 当前分支 [`VeriTile/Core/Ast.lean`](../../VeriTile/Core/Ast.lean) 与 [`VeriTile/Frontend/Triton/DSL.lean`](../../VeriTile/Frontend/Triton/DSL.lean)，`dny@d2e42586`：中立物理目录/Triton frontend迁移的已有方向；不代表TileLang frontend、memory-space或P3已经实现。

## 12. 结论

最小正确的foundation不是“把Triton AST改名后继续加TileLang constructor”，也不是只保留两语言共有的pointwise子集。正确边界是：以union policy维护一个closed、first-order、typed neutral core；两frontend在官方typed IR之后完成各自promotion、shape、index、allocation和launch elaboration；logical address/value/effect保留，纯physical schedule/layout擦除；显式async、barrier、atomic order和peer visibility进入独立concurrency/world layer；rounding structure留在Lean，具体数值和hardware fidelity留在compute gap。

现有deep embedding的typed shape/dtype、region/frame、sequential semantics和proof库应当复用，但应降格为foundation实现与importer oracle，而不是继续承担两套真实Python DSL的完整parser和可信转录。生产路线应从一个双frontend P1垂直切片开始，用固定TTIR/TIR快照、fail-closed importer和Lean checked certificate闭合真实source到neutral theorem的边界；在这一链条稳定之前，不应提前把TileLang的async/TMA/cluster原语“同步化”后宣称支持。
