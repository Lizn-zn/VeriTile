---
title: "并发语义边界"
---

本文档记录 VeriTile 对非顺序 GPU 效应的边界划分。它是 issue #12 的设计入口。

## 当前的确定性边界

面向证明的算法语义目前仍是确定性的:

```lean
Kernel.exec : Kernel -> BlockState -> Option BlockState
```

`Kernel.exec` 按顺序逐条 step statement 来执行一个符号 program instance。
它没有 scheduler、没有 interleaving trace、没有 in-flight operation,
也没有独立的 shared-memory 或 barrier state。

确定性内存栈由以下部分实现:

- #48 active-lane bounds safety;
- #60 predicate frame contract;
- #49 disjoint whole-grid merge;
- #61 footprint extraction helper;
- #62 unrelated-frame helper。

这一栈支持确定性、disjoint、顺序的内存推理。它不建模 overlapping write、
atomic、barrier、shared memory、async/TMA、WGMMA dispatch/wait、
warp specialization,也不建模 scheduling/interleaving。

## 双层架构

VeriTile 使用两层 kernel:

```text
ComputeKernel
  -- erase dtype / erase hardware payload -->
AlgKernel (= Kernel)
```

`ComputeKernel` 是贴近真实 Triton 的那一层。它持有那些不直接属于
数学证明层的效应:目前是 dtype 和 bit payload,未来如果加入,还有
atomic、async/TMA、barrier、shared memory、WGMMA 以及 scheduling 注解。

`AlgKernel` 是证明层。它就是已有的 `Kernel` 类型,带数学 Real/Nat/Int
语义,后续可能扩展面向证明的抽象 effect marker,例如代数化的
`atomic_add`。`Kernel.Correct_without_Rounding` / `Kernel.Refine` 在这一层证明;面向
compute 的公开 surface 通过 `ComputeKernel.ComputeCorrect` /
`ComputeKernel.ComputeRefine` 把这些证明暴露出去。

近期架构里没有独立的 `ConcurrentKernel` 层。如果未来某个特性确实需要
第三层,那应该开一个独立的架构 issue,并配上明确的 consumer。

## 投影语义

形式上的 bridge 是:

```lean
ComputeKernel.toAlgorithm? : ComputeKernel -> Except _ AlgKernel
```

目前这个 bridge 覆盖那些可投影到算法层的 compute-facing 构造。随着 #12
的特性逐步落地,bridge 主要还是一步表示擦除:

- 擦除 dtype 或 bit payload;
- 折叠已支持的常量 bitcast;
- 当存在 algorithm-level 解释时,把硬件味的 `atomic_add` 投影成面向证明
  的 AlgKernel 抽象 atomic/reduction marker;
- 当 discipline 显式时,把规范的 async/TMA 投影成面向证明的 sequentialization
  marker,或普通的 algorithmic load/store;
- 拒绝不支持或非确定性的效应。

这个 bridge 故意允许失败。投影失败意味着该 kernel 在当前证明层没有
`ProjectedCorrect` / `ProjectedRefine` statement,因此公开的
`ComputeCorrect` / `ComputeRefine` surface 也无法对其 discharge。

bridge 不需要消除每一个并发构造。特别地,未来的 atomic 支持可以投影成:

```text
ComputeKernel.atomic_add fp32
  --> AlgKernel.atomic_add real
```

并把以下 theorem:

```text
AlgKernel.atomic_add real == mathematical sum/fold
```

留到 AlgKernel 证明层。这样可以把 associativity / commutativity 论证
放到数学律真正成立的地方。

## 正确性轨道

| 特性类别 | 投影后的算法证明 / ComputeCorrect | 运行时 / 差分测试 |
| --- | --- | --- |
| 确定性且可投影到算法层 | 是,经由 `toAlgorithm?` 和 `Kernel.Correct_without_Rounding` | 可选 |
| 带代数抽象的 atomic 或 reduction | 是,前提是 `toAlgorithm?` 投影到抽象的 AlgKernel reduction marker,并由 theorem discharge | 推荐 |
| 具有有效 sequentialization discipline 的 async/TMA | 是,前提是 `toAlgorithm?` 投影到 sequential 的 AlgKernel 形式或抽象 sequentialization marker,并配 theorem | 推荐 |
| 非顺序且无算法投影 | 否 | 在更强语义出现之前只走 testing/runtime |

只有当 `ComputeKernel.toAlgorithm?` 成功且对应的 `AlgKernel` theorem 已经
证完时,`ComputeCorrect` / `ComputeRefine` 才可用。可选的 `GapPolicy`
contract(#58/#59)记录那些在语法上可表达、但 bit-level compute 语义
未在内部证明的 compute-to-algorithm gap,由外部检查覆盖。

## 失败模式

未来投影失败应该在文档中、必要时也在 error type 中可区分:

- dtype 或 bit payload 无法被擦除到数学语义;
- atomic 操作没有被接受的 AlgKernel marker 或代数 theorem;
- async/TMA 缺少有效的 sequentialization marker 或 discipline theorem;
- barrier 或 shared-memory 行为需要一个尚不存在的 trace model;
- overlapping write 既不被 disjoint merge 也不被 atomic/reduce theorem 覆盖。

本文档不要求重构当前 error type。它记录的是未来 #12 实现切片应当保留的
分类。

## Follow-Up 顺序

预期的实现顺序是:

1. Trace/interleaving 词汇,只覆盖 atomic 和 barrier 所需。
2. Atomic-only 语义或算法抽象。这是 atomic 正确性工作的直接前置,
   包括 #43 FA-1 backward dQ。
3. Async/TMA sequentialization theorem 家族。
4. WGMMA、warp specialization 以及完整的 Hopper 形 kernel,推迟到
   更强的并发基础设施出现之后。

## 分层路线图

并发路线图刻意分层。每一层有不同的语义对象,所以后续层应当扩展早期层,
而不是把它们的 machinery 强行塞进更简单的证明里。

| Layer | 语义对象 | 覆盖范围 | 跟踪 |
| --- | --- | --- | --- |
| L1: single-cell linearized RMW | `MemCell -> RMWEvent -> Option (MemCell × RMWEvent)` 加 per-cell 有序 event list | `atomic_add`、`atomic_xchg`、`atomic_cas`、single-cell max/min/and/or/xor | #66 处理可交换 add;#82 处理 order-sensitive xchg/cas |
| L2: multi-cell atomic transaction | 跨多 cell 的 state transformer | DCAS / MCAS / 事务内存式 primitive | 暂无活跃 issue;待真实 consumer 出现时再开 |
| L3: cross-cell ordering + async | happens-before / visibility 图加 fence、barrier、async completion | memory ordering、async copy、TMA/WGMMA visibility、producer-consumer warp specialization | #12 长周期 |
| L4: 无锁数据结构 invariant | L1/L3 加上把抽象状态关联到内存的 data-structure invariant | queue / stack / set / lock-free protocol | 暂无活跃 issue;由 consumer 驱动 |

#82 只实现 L1。它必须与之后的 L2/L3/L4 工作兼容,但它不能依赖
multi-cell transaction、global scheduler、async visibility 或 data-structure
invariant。

## Trace 词汇

第一份词汇切片放在:

```text
VeriTile/Triton/Concurrency/Trace.lean
```

它定义了 `ThreadId`、`RMWOp`、`MemoryEvent`、`TraceEvent` 和 `Trace`。
`MemoryEvent.rmw` 携带共享的 `RMWEvent` payload:

```lean
structure RMWEvent where
  cell : MemCellAddr
  op : RMWOp
  input : MemCell
  extraInput : Option MemCell := none
  observed : Option MemCell := none
  result : Option MemCell := none

inductive MemoryEvent where
  | read (region : RegionName) (offset : Nat) (value : MemCell)
  | write (region : RegionName) (offset : Nat) (value : MemCell)
  | rmw (event : RMWEvent)
```

可交换 atomic 把 `input` 作为自己的 contribution,可选字段留空。
order-sensitive atomic 例如 xchg/cas 可以填 `extraInput`、`observed`
和 `result`,而不必引入第二种 event type。trace 模块不是 scheduler,
也不会改 `Kernel.exec`。

## #82 PR0 审计结果

#82 的第一步实现是在加入 `atomic_xchg` / `atomic_cas` 语义之前先做
ownership/API 审计。当前结果是:

```text
API ready; proceed to PR1.
```

审计细节:

- `PermissionModel` / `OwnershipMap` 对 permission type 是抽象的,
  公开 discipline predicate 没有写死 `WarpId` ownership。
- `RMWEvent` 已经是单一可扩展的 RMW payload。trace 层不需要并行的
  CAS/XCHG event type。
- `MemoryEvent.rmw` 已经携带 `RMWEvent`,所以加入 `.xchg` / `.cas`
  语义不需要重构 constructor 形状。
- `Stmt.atomicAdd` 仍是 #66 可交换 theorem 的独立公开 surface。#82
  应当增加一个 return-valued 的 RMW constructor,而不是重载
  `Stmt.atomicAdd`。
- `Trace.LinearizesAt` 已经提供 per-cell 的 trace hook。#82 可以加一个
  通用化的 event-list linearization predicate,但不需要 global scheduler
  或 timestamp map。

## Grid Launcher

两个关系式 launcher 把 per-program 的 kernel correctness 提升到
whole-grid correctness:

- **`Kernel.GridLaunchedOrdinary k g s sFinal`** —— 用于 ordinary
  (无 atomic 的)kernel。说明:从 `s` 出发,在 `g` 的每个 grid index 上
  跑 `k`,得到 `sFinal`,且 per-program write 两两 disjoint。
  per-program frame write 的 composition 是关键事实;`mergeFrames`
  完成 merge。
- **`Kernel.GridLaunchedRMW k g s sFinal linearization`** —— 用于
  在单个 cell 上含 atomic RMW 操作的 kernel。把 ordinary frame write
  与显式的 per-cell linearization witness 合起来。下面的 atomic 切片
  会用到。

把 per-program correctness 提到 `ForAllProgramsSome k g s post` 的形式,
等价于 "对每个 grid index `idx`,exec 成功且 postcondition 在
`s.withGridIndex idx` 上成立"。grid 级 theorem 通常通过这个
`ForAllProgramsSome` 形状来包装一个 per-program ComputeCorrect。

## Atomic Add 切片

第一片具体的 atomic 切片刻意做窄:

- `tl.atomic_add` 降为面向证明的 `Stmt.atomicAdd` marker;
- 单 program 的 `stepStmt` 顺序执行 read-add-write 更新;
- `Stmt.atomicTraceEvents` 把 active lane 记录为 `MemoryEvent.rmw ... .add value`;
- `Kernel.mergeFramesWithAtomic` 把 #49 的 ordinary frame write 与
  选中的 grid-level atomic trace 合起来;
- `Kernel.mergeFramesWithAtomic_atomicAdd_eq_finsetSum` 把 Real
  最终 cell 值的 theorem 表述为初始值加 trace payload 的 `Finset.sum`。

这是 `Limited` atomic 支持,不是完整的并发 executor。
`tl.atomic_xchg` 和 `tl.atomic_cas` 现在投影到 return-valued 的
`Stmt.atomicRMW` 算法 marker,single-cell RMW 语义放在
`RMWOp.apply` / `RMWTrace.applyLinearized`。它们也有可执行的
single-program statement 语义、通过
`Stmt.atomicTraceEvents` / `Kernel.AtomicTraceStateful` 实现的 stateful
trace emission,以及把 ordinary frame write 与显式 per-cell linearization
witness 合起来的 single-cell whole-grid launcher 关系
(`Kernel.GridLaunchedRMW`)。其他 Triton atomic surface
(`tl.atomic_max`、`tl.atomic_min`、`tl.atomic_and`、`tl.atomic_or`、
`tl.atomic_xor`)目前降到 `ComputeStmt.effectMarker`,并以
`requiresEffectProjection` 投影失败。

## Async / TMA Contract 切片

async/TMA 切片目前只是文档化 contract 加 compute-facing failure marker,
不是实现。

设计方式镜像 `atomic_add` 的 marker 模式。未来 compute-facing 的
`tl.async_*` 或 TMA 语法只在已识别 discipline 可用时才能投影:

```text
ComputeKernel async/TMA surface
  -> AlgKernel async/TMA sequentialization marker
  -> theorem 把 marker 消除到普通数学行为
```

Contract 给出未来必需 discipline 的命名:

- 每个 async issue 都有匹配的 wait;
- 在匹配 wait 之前,任何 read 都不能观察到 destination;
- destination ownership 在 program slice 内部是无歧义的;
- overlapping destination 要么显式排序,要么被拒绝。

投影失败用具名原因 `requiresEffectProjection`。
`ComputeStmt.effectMarker` 是这条路径的显式 AST hook:它让 async/TMA
形态的语法在 compute 层可表达,同时保留 "尚无 `ProjectedCorrect` 投影"
这一事实。DSL surface `tl.async_copy(dst, src)`、`tl.async_wait()` 和
`tl.debug_barrier()` 目前降到这个 failure marker;它们没有可执行语义,
也不蕴含 shared-memory、barrier 或 TMA 建模。

显式 shared-memory state、TMA destination state、WGMMA operand layout
以及 scope-tagged footprint 仍然是 #65 的触发项。

## Non-Goals

本边界文档不实现:

- 超出 limited `tl.atomic_add` 面向证明切片的可执行 `tl.atomic_*`;
- async copy 或 TMA;
- WGMMA 或 warp specialization;
- shared memory 或 barrier;
- Iris 风格或 separation-logic 基础设施;
- scheduler 或 interleaving 语义;
- 任何对 `Kernel.exec` 的修改。

目的是在定义未来非顺序效应进入位置的同时,保持当前确定性的
`ComputeCorrect` / `ProjectedCorrect` 故事稳定。
