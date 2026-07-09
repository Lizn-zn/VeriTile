# 正确性 surface

本文档说明在证明 `ComputeKernel` 的性质时,应当选用哪个公开 theorem surface。
所有 surface 定义在
[`VeriTile.Triton.Correctness`](../VeriTile.Triton.Correctness.lean)。

## 舍入默认

先读这一节 —— 它固定了下面每个 surface 名字的含义。**不带限定词**的 surface
名字(`Realizes`、`Refines`、`RefinesAt`、`Correct`)是**舍入模型** surface:
它们接受一个 `RoundingModel R`,在 R-threaded 语义 `execR` 下执行。**精确实数
理想化** —— 也就是旧的"一切都是实数算术、没有量化"的读法 —— 是显式带限定词的
`*_without_Rounding` surface(`Realizes_without_Rounding`、
`Refines_without_Rounding`、`RefinesAt_without_Rounding`、
`Correct_without_Rounding`)。

经验法则:不带限定词的名字是 kernel 在硬件上*实际*做的(带舍入);
`*_without_Rounding` 是当舍入与结论无关时可退回的数学家理想化。精确 surface
在 trivial model `.triv` 处(此时 `execR` 退化为 `exec`)**从**舍入 surface
**退化出来**,反方向从不成立。141 个 ported kernel 大多在 `*_without_Rounding`
上证明(它们的输出是精确实数 spec);`bench/examples/` 下带 bf16 边界 store 的
showcase pair 落在舍入 surface `Refines R` 上。

## 速查表

| 目标 | 用 |
|---|---|
| 一个 kernel realize 某个输出规范(带舍入)| `ComputeCorrect.Realizes` |
| …… 该规范的精确实数理想化 | `ComputeCorrect.Realizes_without_Rounding` |
| 一个 kernel refine 另一个,writes-equality(带舍入)| `ComputeRefine.Refines` |
| …… 精确实数理想化 | `ComputeRefine.Refines_without_Rounding` |
| 两个 kernel 逐地址 pointwise 相符 | `ComputeRefine.RefinesAt`(精确:`RefinesAt_without_Rounding`)|
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

### `Realizes` vs `Refines` —— 命名方案

VeriTile 用命名区分两个验证问题(每个名字都是舍入 surface;加
`_without_Rounding` 后缀得到精确实数理想化):

- **`Realizes`** —— *一个 kernel realize 某个 spec*(单 kernel 对照期望输出)。
  `ComputeCorrect.Realizes` 是舍入形式;`ComputeCorrect.Realizes_without_Rounding`
  是大多数 ported kernel 使用的精确单 kernel 主力。
- **`Refines`** —— *一个 kernel refine 另一个 kernel*(两个 kernel 互相对照)。
  `ComputeRefine.Refines`(writes-equality)、`ComputeRefine.RefinesAt`
  (逐地址 pointwise 关系),以及它们的精确镜像 `Refines_without_Rounding` /
  `RefinesAt_without_Rounding`。

### Writes-equality:`ComputeRefine.Refines`

标准的双 kernel surface 是 `ComputeRefine.Refines`。在舍入模型 `R` 下,它从
同一初始状态经 `execR R` 跑两个 kernel,断言它们做了**相同的写入** —— 两个终态
memory 在所有非 `scratch` region cell 上逐格相等:

```lean
ComputeRefine.Refines R lhs rhs s scratch
-- := ExecRefineR R lhs rhs s (fun l r =>
--      ∀ region ∉ scratch, ∀ offset, l.mem region offset = r.mem region offset)
```

相同写入位置、相同写入值,一条等式 —— 对给定的 `R`。`scratch` 列出两个 kernel
*允许*不一致的 region —— pipeline 的中间 tensor(例如 fused-vs-unfused SwiGLU
pair 的 `[S]`,或 `FusedSiLU` 里的 `zReg`/`siluReg` 临时区)。当两个 kernel 必须
在整个 memory 上一致时传 `[]`。`bench/examples/` 里每个 `*_refinement_view` 定理
都落在这个 surface 上;精确实数变体 `ComputeRefine.Refines_without_Rounding`
(无 `R`,在 `exec` 下执行)是 `bench/examples/FusedSiLU.lean` 用的理想化。

### Pointwise:`ComputeRefine.RefinesAt`

当两个 kernel 写到**不同**目标 cell,或对照关系不是等式时,用 pointwise 形式。
它通过两张独立的 `WriteMap` 关联两个 kernel 的输出:

```lean
ComputeRefine.RefinesAt lhs rhs s lhsWrite rhsWrite relation
-- post: ∀ i, match lhsWrite i, rhsWrite i with
--   | some la, some ra => relation i (read lhs' la) (read rhs' ra)
--   | _, _ => True
```

两边 read 的 carrier 类型各自推断,所以 `RefinesAt` 覆盖异构 layout 对照和证明
中间件。同一 buffer 的普通等价用两边相同的 write map。首选 whole-memory 的
`Refines`;确实需要 per-side 值时才用 `RefinesAt` 这个 escape hatch。

### 舍入模型 surface(窄浮点,#447)

不带限定词的 surface 就是舍入 surface:每个都对一个 `RoundingModel R`
(`round : FloatDType → ℝ → ℝ`,幂等性是它的一个 defining field)parametric,
并在 R-threaded 语义 `execR` 下执行。它们定义在
[`VeriTile.Triton.Float.Refine`](../VeriTile/Triton/Float/Refine.lean):

- `ComputeRefine.Realizes kernel s write expected` —— 单 kernel 对照
  `R`-annotated spec `expected : RoundingModel → ι → α`。spec 的形状**就是**
  rounding-event 账本:没有 `R.round` 表示观察路径无 rounding;一个 `R.round`
  表示一次最终量化;嵌套的 `R.round` 每个算一次事件。
- `ComputeRefine.Refines R lhs rhs s scratch` —— `R` 下的 writes-equality
  pair surface。
- `ComputeRefine.RefinesAt R lhs rhs s lhsWrite rhsWrite relation` —— `R` 下
  的 pointwise pair surface。

精确实数理想化是 `*_without_Rounding` 镜像,它们在 trivial model `.triv` 处
(此时 `execR` 退化为 `exec`)**从**舍入 surface **退化出来**:桥
`ComputeRefine.Realizes.toRealizes`("舍入断言蕴含理想 correctness")把任何
`Realizes` 变成关于 `expected .triv` 的普通
`ComputeCorrect.Realizes_without_Rounding`。退化引理 `refines_triv_iff` /
`refinesAt_triv_iff` 以同样方式恢复 `Refines_without_Rounding` /
`RefinesAt_without_Rounding`。∀R 组合模式的范例见
[`bench/examples/FusedSwiglu.lean`](../bench/examples/FusedSwiglu.lean);边界舍入的
showcase(LogSumExp、softmax、Welford、LayerNorm、FusedSiLU)对固定的 `R` 落在
`Refines R` 上。

### Equality helper

两边读回同一 buffer 的常规优化证明,下面的 equality helper 是 `ExecRefine` 的
薄封装,仍是最可读的选择。

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
[`EraseDType_zh.md`](./EraseDType_zh.md)。

## 命名

推荐 theorem 命名:

- `<kernel>_compute_correct`:常规单 kernel 正确性。
- `<kernel>_correct_view`:对外的 manifest 兼容 view surface。
- `<rewrite>_refinement_view`:对外的双 kernel refinement surface。

仅用于执行的 helper lemma 可用 `_exec_view` 命名,以及直接的 `exec` 等式。
对外的 theorem 应把这些 helper 包到 `ComputeCorrect.*` 或 `ComputeRefine.*`
里。命名细则:[`TheoremSurfaces.md`](./TheoremSurfaces.md)。
