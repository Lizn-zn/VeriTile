---
title: 翻译一个 kernel
description: .py → .lean 翻译什么算忠实、哪些 gap 容忍、哪些偏离必须在合入前修。
---

Gold standard 是
[`bench/tritonbench_g/add_example/AddExample.lean`](https://github.com/Lizn-zn/VeriTile/blob/main/bench/tritonbench_g/add_example/AddExample.lean):
`def kernel ... : ComputeKernel := triton { ... }` 的 body 就是 `.py`
kernel 函数体**一行不多一行不少**的转写,不加 statement、不合并、不偷偷
"修正"。

之所以这么严格,是因为项目的心智模型是:用户带着"我把 `.py` 粘进来看
证明"的预期来。任何隐式简化或合并都会让证明从"这个 kernel"漂移到
"VeriTile 对这个 kernel 的代数变形"。

审一份新翻译(或写一份新翻译时)用下表分类,它告诉你该 push back 哪些
诱惑。

## ✓ 忠实

`.py` 函数体和 `triton { ... }` body 行对行对应。不加 statement、不合并、
不重排。`tl.*` 调用矩阵一模一样,只有 Lean 语法包装(词法上)有差别。

## ⚠️ Minor(可容忍)

两类计 MINOR —— 不 block 合入,但要标注。

### 1. Lean-syntax forced changes —— 白名单内

只有以下两条算 Lean-syntax 强制变化:

- **`$(x)` 插值** 包 Lean 参数 / 编译期常量(包含
  `BLOCK_SIZE: tl.constexpr` 模型为 `Nat` 参数 + `$(BLOCK_SIZE)` 使用)。
- **`if HAS_X { ... }` DSL gate** 替代 Python 的 `if HAS_X:` constexpr
  分支。

白名单之外的转换**都不接受**,具体包括:

- `.py` 没标 dtype 时给 `tl.load` 加显式 `dtype=...`。
- `keep_dims=true` 替代 `[:, None]`。
- `1 / tl.sqrt(x)` 替代 `tl.math.rsqrt(x)`。
- `(0.0).to(tl.float32)` 显式 cast。
- 指针 mutation `X += offset` 改成每次显式 `base + offset`。
- `x += rhs` 改成 `x = x + rhs`。
- "Real-first policy" 偷偷擦掉 `.to(tl.float32)`。

DSL 表达不出 `.py` 写法 = DSL gap,**开 issue 修 DSL**,不是改 kernel
body 的绿灯。

### 2. 已文档化的 specialization / partial slice

允许做有文档的部分特化,但 preamble doc 注释必须**明确写出来**:

- "proof-oriented one-block slice" / "single-tile" / "one-iteration" ——
  upstream 是循环,Lean 只证一次迭代。
- 单 constexpr 分支特化(`IS_RMS_NORM=true`、`HAS_BIAS=false`、
  `LOG=false` 等)。
- 单输出通道 slice(如 `FifthOrderSphHarmonics` 只取 `Y00`)。
- 未使用参数省略(1D path 里的 `*_row_stride`)。

preamble 必须说"这是 slice / specialization",不是"这是完整 kernel
的等价形式"。

## ✗ Deviates —— 必须修

下面这些不是 gap,是翻译 bug。

### 加了 `.py` 没有的 statement

任何多出来的 `tl.where` / cast / mask 处理。例(真实):`RmsnormFused.lean`
曾多了一条 `x_masked = tl.where(cols < N, x_for_var, 0.0)`,而 `.py` 的
对应 padding 是通过 `tl.load(..., other=0.0)` 完成的。纸面上等价,但证明
谈的是**另一个 kernel**。

### 加了 `.py` 没有的算术结构

地址或值上凭空加 stride 系数、broadcast 维度、scale 因子。例:把
`tl.load(W + col_offsets, ...)` 改成 `tl.load(W + col_offsets * W_row_stride, ...)`,
当 `W` 有非平凡 stride 时**结果就不一样**了 —— 即使"看起来更对"。

### 把 reduction 循环折成单个 `tl.sum`

upstream 是 `for off in range(0, N, BLOCK_N): var += xf*xf`,翻译写成
`var = tl.sum(x*x, axis=0)`。即使数学等价,body 结构变了,证明"证的不是
这个 kernel"。两种可接受修法:

- 把 `for` 循环加回去(忠实 —— 但证明也得改)。
- preamble 明写"将 N-block 累加循环折叠成单 tile `tl.sum`,要求
  `BLOCK ≥ N`"(承认特化)。

### 重排 compute 语句

即使数据流等价,1:1 行对应丢了就不算忠实。

### 合并 / 省略 `tl.*` 调用,且无文档

连锁 cast 省了一个、masked load 偷偷退化成 bare load,都属这一类。

### 改控制流

Python 多分支 `if/elif/else` 折成 `tl.where`、`for` 展开、`while` 改
`for`,都要明文写出意图。

## 审计输出格式

审一个文件分三档:

| 结论 | 含义 |
|---|---|
| `✓ FAITHFUL` | 逐行 1:1。 |
| `⚠️ MINOR` | 落入上面的"可容忍 gap"。第 2 类要引用 preamble 注释。 |
| `❌ DEVIATES` | 落入"必须修"任一条。给出 `.py` 行号 + `.lean` 行号 + 一行偏差描述。 |

审计可并发跑 —— ~15 个 kernel 一批 agent,合适。

## 写哪种 surface

翻译稳定后,选定理 surface。chooser 在
[Correctness surfaces](/VeriTile/zh-cn/proofs/correctness-surfaces/);一眼速查:

| 目标 | Surface |
|---|---|
| 一个 kernel realize 某输出 spec | `ComputeCorrect.Realizes_without_Rounding` |
| 两个 kernel realize 某输出关系 | `ComputeRefine.Refines_without_Rounding` |
| 值 + 索引同时输出(如 `return_indices=True`) | `ComputeCorrect.OutputPairWhere` |
| 自定义 final-state 后置条件 | `ComputeCorrect.Post` / `ComputeRefine.Post` |
| 关系跨任意初始状态 | `ComputeCorrect.General` / `ComputeRefine.General` |
