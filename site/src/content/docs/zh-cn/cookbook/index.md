---
title: Bench 翻译手册
description: 如何把一个 Triton .py kernel 翻译进 Lean 并完成正确性证明 —— 元规则、审查标准、约定、证明模板、已知 tactical 陷阱。
---

本节汇总把一个 Triton `.py` kernel 翻译进 `triton { ... }` block、写出
spec、完成 `ComputeCorrect.Realizes` 证明所需的工作经验。它把 bench
语料堆出来的教训沉淀下来,不是理论参考(那个看
[架构与语义](/VeriTile/zh-cn/architecture/) 和 [证明与表面](/VeriTile/zh-cn/proofs/))。

## 本节包含

1. [翻译一个 kernel](/VeriTile/zh-cn/cookbook/translating/) —— 什么算 1:1 忠实转写、
   哪些 gap 容忍、哪些必须在合入前修。
2. [约定](/VeriTile/zh-cn/cookbook/conventions/) —— 外侧优先 shape、完全 ND 算子、
   命名模式、重写成本的态度。
3. [证明模板](/VeriTile/zh-cn/cookbook/proof-templates/) —— 标准 1D scatter、双通道
   写、单步循环,以及该用哪个 helper。
4. [`forLoop_inv` 陷阱](/VeriTile/zh-cn/cookbook/forloop-pitfalls/) —— 证带 loop
   kernel 时的七个 tactical 坑。

## 四条元规则

动代码之前,项目承诺四条规则,优先级高于"现在写得顺手":

### 1. Triton 语义保真度

VeriTile 是面向 Triton 的验证项目。当一个设计选择在"代码整洁"和"对真实
Triton 语义保真"之间二选一时,**保真度赢**。具体例子:Triton 自己就区分
`shape ()` 和 `shape (1)`,所以我们也区分 —— 即便把它们打包在一起会让类型
机制更简单。算法通道层的偏离(比如建模 `ℝ` 而不是 IEEE-754)是允许的,
但必须明确记录为已知 gap,不能当作 silent simplification。

### 2. Triton 用户是一等公民

一个拿着可用的 `triton.jit` 装饰过的 `.py` kernel 的开发者,应该可以把它
粘到一个 `triton { ... }` 块里,**只做最少修改** 就能 parse 并 lower。
parse 后的 surface 要在 Lean 语法允许的范围内尽量贴 Triton `tl.*` API ——
`tl.load(x_ptr + offsets, mask=m, other=0.0)`、`tl.program_id(axis=0)`、
Python 风格的 mask 比较、kwargs 等等。新的 DSL 构造**不要**引入新的
antiquote 需求(`$(...)`、`$ℝ(...)`),除非真的没法绕。

### 3. 外侧优先,完全 ND

`TileShape := List Nat` **外侧优先**(`shape (M, N) = [M, N]`,list 索引 =
用户可见的轴索引),对齐 Triton / NumPy / PyTorch。即使内部存储翻成
"内侧优先"会让 head-cons 递归更容易写,**也不要翻**。在 Lean 侧吃下额外
痛苦(`init ++ [last]` 模式、`_pos` 版本免去 `Option`)。

新的轴感知算子(reduce、transpose、expand_dims、broadcast 扩展……)**从
一开始就要完全 ND**,以 `Fin shape.length` 作为轴参数。不要为了"这个
kernel 暂时只用 2D"加 2D-only 或 last-axis-only 变体 —— 它会在下一个
shape pattern 进来时立刻变成维护坟场。

### 4. 重写代价是 tiebreaker,不是 thesis

"那我们就要重写 N 个证明" 单独是**不足以**作为选择更差设计的理由的。
挑更整洁 / 更忠于 Triton / 更通用的设计,把重写代价吃下去。项目目标是
跨数年的迭代,一个永久更差的设计省下几周,代价是数年的摩擦。Ship 时机
的 tactical 反对是合理的,可以推迟某次重构,但不应该改变推荐方向。

---

只读一页的话,接下来读
[翻译一个 kernel](/VeriTile/zh-cn/cookbook/translating/) —— 添加新 bench 时最常
查的一页。
