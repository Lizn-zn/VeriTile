---
title: "正确性 surface"
---

本文档说明在证明 `ComputeKernel` 的性质时,应当选用哪个公开 theorem surface。
所有 surface 定义在
[`VeriTile.Triton.Correctness`](https://github.com/Lizn-zn/VeriTile/blob/main/VeriTile.Triton.Correctness.lean)。

## 速查表

| 目标 | 用 |
|---|---|
| 一个 kernel realize 某个输出规范 | `ComputeCorrect.Realizes` |
| 两个 kernel realize 某个输出关系 | `ComputeRefine.Realizes` |
| Whole-grid 启动(每个 program 都正确写入)| `Kernel.ForAllProgramsSome`(临时;见 Grid 章节)|
| 每个 lane 都输出 value/index 对 | `ComputeCorrect.OutputPair` |
| 仅在 active lane 上输出 value/index 对 | `ComputeCorrect.OutputPairWhere` |
| 两个 kernel 产生相同的 value/index 对 | `ComputeRefine.OutputPairEq` / `OutputPairEqWhere` |
| 自定义 final-state 后置条件 | `ComputeCorrect.Post` / `ComputeRefine.Post` |
| 关系跨任意初始状态(罕见) | `ComputeCorrect.General` / `ComputeRefine.General` |

底层的 `ComputeKernel.ComputeCorrect` 和 `ComputeKernel.ComputeRefine` 定义仍是
实现层。新示例 theorem 一般不应直接暴露这两个名字。

## 输出 write map

最通用的输出 surface 把实际写入 map 和期望值 map 分开:

```lean
abbrev ComputeCorrect.WriteMap (ι : Type) := ι → Option MemCellAddr

ComputeCorrect.Realizes
  (kernel := k)
  (initialState := s)
  (write := write)
  (expected := expected)
-- post: ∀ i, match write i with
--   | some addr => read final addr = expected i
--   | none => True
```

`Realizes` 会根据 expected value 的类型自动选择 readback:`ℝ` 使用
`BlockState.readMem`,`Nat` 使用 `readMemValue .nat`,`MemCell` 使用
algorithm-layer 精确 cell 等式。常见标量/tensor readback 仍可使用更顺手的
wrapper,例如 `OutputScalar`、`OutputArray`、`OutputNatScalar`。

masked store 通常用 `WriteMap.writeIf` 表示:

```lean
write := ComputeCorrect.WriteMap.writeIf
  (fun i : Fin BLOCK_SIZE => base + i.val < N)
  (fun i => (out, base + i.val))
```

这样 theorem 只说明 active lane 写了哪些地址以及写成什么值。未写地址的 preserve
性质交给 frame/preserve theorem。

标准证明步骤是:

```lean
rw [ComputeCorrect.realizes_writeIf_iff]
```

它会把输出义务变成 `∀ i, mask i → read final (addr i) = expected i`。

whole-grid launch theorem 应先给出 final-state execution surface;当它能表示成
产生 final state 的 `ComputeKernel` surface 后,仍然使用同一个 `Realizes`
形式描述输出。VeriTile 不保留单独的 `GridOutputAt` 用户 surface;grid
execution 是执行模型问题,不是新的输出形状。

### Grid 启动的当前状态

VeriTile 暂未提供 whole-grid `launchExec : ComputeKernel → Grid → BlockState
→ Option BlockState`。在它落地之前,grid 定理使用 per-program-local 形态
`Kernel.ForAllProgramsSome`,即:对每个 typed grid index,从
`s.withGridIndex idx` 出发跑 kernel,得到的 state 满足 per-`idx` 的后置
条件。规范例子见 `logsumexp_fwd_kernel_grid_blockLSE_correct`。等
launcher 落地后,grid 定理会迁移到 `ComputeCorrect.Realizes`(以 launch
作为额外参数),用户面向的定理形态保持一致。

## 单 kernel 正确性

单 kernel 正确性把一个 kernel 对照数学规范或算法规范来检查。

### 标量输出

```lean
ComputeCorrect.OutputScalar k s out offset expected
-- post: s'.readMem out offset = expected

ComputeCorrect.OutputNatScalar k s out offset expected
-- post: s'.readMemValue .nat out offset = expected
```

适用于标量 reduction,例如最终的 `max`、`sum`,或 `argmax` 风格的整数 index。

### Tensor 输出

```lean
ComputeCorrect.OutputTile k s view expected
-- post: ∀ idx, TensorView.observe (some s') view idx = some (expected idx)

ComputeCorrect.OutputArray k s view expected   -- 1D specialization
-- post: same, with `expected : Fin n → ℝ`
```

`OutputArray` 是 `OutputTile` 在一维 `n`-shape 下的特化,当规范天然是
`Fin n → ℝ` 函数时优先使用它。两者都是通过 `WriteMap.ofTensorView`
封装出来的 `Realizes` wrapper,目的是让 tensor-view theorem statement 更可读。

### Value/index 对输出

```lean
ComputeCorrect.OutputPair k s valueRegion indexRegion offset
  expectedValue expectedIndex
-- post: every lane satisfies the value spec and the typed Nat index spec

ComputeCorrect.OutputPairWhere k s valueRegion indexRegion offset
  active expectedValue expectedIndex
-- post: same, but only on lanes where `active i` holds
```

适用于具有成对输出的 kernel,例如 `tl.max(..., return_indices=True)`。
当 mask 限制了哪些 lane 参与时,选 `OutputPairWhere`。

`OutputPair*` 仍是独立定义,不是 `Realizes` 的 wrapper —— 因为 value 通道是
`ℝ`、index 通道是 `Nat`,两个通道 readback 类型不同,而 `Realizes` 的
typeclass 每次只能 dispatch 一个 carrier。将来可以加一个 per-lane carrier
typeclass 把它们一并归入,但在出现第二种 heterogeneous 输出 kernel 之前
不值得做。

## Kernel refinement

Kernel refinement 从同一初始状态出发跑两个 kernel,并关联它们的 observation。
做常规优化证明时优先选 equality helper。

### 标量等式

```lean
ComputeRefine.OutputScalarEq lhs rhs s lhsOut lhsOffset rhsOut rhsOffset
-- post: lhs'.readMem lhsOut lhsOffset = rhs'.readMem rhsOut rhsOffset

ComputeRefine.OutputNatScalarEq lhs rhs s lhsOut lhsOffset rhsOut rhsOffset
-- post: lhs'.readMemValue .nat ... = rhs'.readMemValue .nat ...
```

### Tensor 等式

```lean
ComputeRefine.OutputTileEq lhs rhs s lhsView rhsView
-- post: ∀ idx,
--   TensorView.observe (some lhs') lhsView idx
--     = TensorView.observe (some rhs') rhsView idx

ComputeRefine.OutputArrayEq lhs rhs s lhsView rhsView   -- 1D specialization
```

### Value/index 对等式

```lean
ComputeRefine.OutputPairEq lhs rhs s
  lhsValueRegion lhsIndexRegion rhsValueRegion rhsIndexRegion offset

ComputeRefine.OutputPairEqWhere lhs rhs s ... offset active
```

当等式只需在 active lane 上成立时,使用 `Where` 变体。

## Escape hatch

`Post` 固定初始状态,但接受任意 final-state 后置条件:

```lean
ComputeCorrect.Post k s (fun s' => ...)
ComputeRefine.Post lhs rhs s (fun lhs' rhs' => ...)
```

`General` 把初始状态本身也量化,接受 `(s0, s')` 上的关系:

```lean
ComputeCorrect.General k (fun s0 s' => ...)
ComputeRefine.General lhs rhs (fun s0 lhs' rhs' => ...)
```

`General` 应保留给确实需要在初始状态上 parametric 的 theorem(罕见)。
绝大多数证明固定 `s` 后用 `Post`,或选一个 `Output*` helper。

## Gap policy

`ComputeCorrect.*` / `ComputeRefine.*` helper 默认使用 `GapPolicy.ignore`,
对应当前以 Real 为主的算法证明。如果某 theorem 需要记录已通过外部检查的
compute-to-algorithm gap,可直接陈述底层 `ComputeKernel.ComputeCorrect` /
`ComputeKernel.ComputeRefine`,并写:

```lean
gap := .require contract
```

或为该 theorem 家族新增专用 helper。compute-to-algorithm 投影策略详见
[`EraseDType_zh.md`](/VeriTile/zh-cn/architecture/erase-dtype/)。

## 命名

推荐 theorem 命名:

- `<kernel>_compute_correct`:常规单 kernel 正确性。
- `<kernel>_correct_view`:对外的 manifest 兼容 view surface。
- `<rewrite>_refinement_view`:对外的双 kernel refinement surface。

仅用于执行的 helper lemma 可用 `_exec_view` 命名,以及直接的 `exec` 等式。
对外的 theorem 应把这些 helper 包到 `ComputeCorrect.*` 或 `ComputeRefine.*`
里。命名细则:[`TheoremSurfaces.md`](/VeriTile/zh-cn/proofs/theorem-surfaces/)。
