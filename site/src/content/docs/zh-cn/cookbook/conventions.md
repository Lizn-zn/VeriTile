---
title: 约定
description: TileShape 朝向、ND-general 算子、命名模式、何处该 push back。
---

下面的约定不是风格偏好 —— 每一条都被狠咬过,所以是项目级规则。

## 外侧优先 shape

`TileShape := List Nat` **外侧优先**:`shape (M, N) = [M, N]`。list 索引
*就是*用户可见轴索引,对齐 Triton、NumPy、PyTorch。

实现新的轴感知算子时,Lean 侧的自然冲动是把"active 轴"放在 list 头部,
让头部递归更好写。对 broadcast 和头轴 reduce 来说,**内侧优先**内部
存储"看起来更整洁"。

**不要这么做。**

**为什么:**用户粘 `.py` kernel 时写 `axis=K`,K 指的是用户可见的外侧
优先 shape。内部翻转就是永久的人体工学税 —— 每个 DSL 宏、每个文档示例
都要在用户脑里翻译一次。Triton 语义保真胜出。

**怎么用:**新 ND 算子(reduce、transpose、expand_dims、ND-broadcast
扩展……)觉得用内侧优先更好写时,把痛苦吃在 Lean 外侧优先侧:

- `init ++ [last]` 递归模式。
- 对 `init` 做结构归纳。
- 提 `_pos`(positivity-asserting)版本免掉 `Option`。

`VeriTile/Triton/Semantics.lean` 里的 reduce 实现 ——
`Tile.reduceSumDropLast`、`reduceMaxDropLast_pos` —— 是范本。

## 一定走完全 ND

写 `match shape with | [_, _] => ...`、`(rest ++ [axisDim])`、
`Input2DAt` 之前问自己:未来的 kernel 会不会想换 rank 或换轴?答案几乎
总是"会"。

纪律:

1. **设计时用 `Fin shape.length` 轴索引。**新轴感知 typed-AST 节点收
   显式轴参数,不要硬编码位置。
2. **ND 基础设施用 offset 函数参数化。**predicate、helper、定理收
   `TileIndex shape → Nat`,而不是把 1D / 2D pattern 烤进去。
3. **同时提供命名 offset 族。**`Offset.linear1D`、`Offset.rowMajor2D`、
   `Offset.contig` 是人体工学 helper;核心 API 保持通用。

`VeriTile/Triton/Core.lean` 的 reduce-axis API —— `axisDim`、
`eraseAxis`、`setAxisOne`、`reduceShape`、`insertAxisIndex`、
`replaceAxisIndex` —— 是范本。新轴感知算子照这个 pattern 来。

**代价:**真实的 dependent-type 工作,但有界。一次付清。相反,
rank/axis 特化 helper 每次 bench 进新 shape pattern 都要重写一次。

## 重写代价是 tiebreaker,不是 thesis

评估设计 tradeoff 时:

- 把"那要重写 N 个已有证明"当成 tactical 顾虑(什么时候 ship、怎么切片),
  **不是**战略票。
- 两个设计势均力敌时,不需要重写的那个可以险胜。但如果一个**明显更
  整洁 / 更正确 / 更忠于 Triton**,即使要写几百行重写也推荐它。
- 不要把"保留现有证明"当作设计的**主要**论据。它是 tiebreaker,不是
  thesis。
- Ship 时机的 tactical override 是合法的("先用 D-b,release 后再改
  D-a")。但默认推荐不应该走到 D-b。

## 命名模式

bench 语料里反复出现的几个 pattern。一致一点,proof 更便宜。

- **Per-kernel 文件**位置:port 在
  `bench/tritonbench_g/<kernel>/<Kernel>.lean`,example 在
  `VeriTile/Examples/<Kernel>.lean`。
- **定理名**结尾 `_correct`(kernel ↔ spec)或 `_refine`
  (kernel ↔ kernel)。Compute-layer 投影对应 `_compute_correct` /
  `_compute_refine`。
- **Spec 定义**通常叫 `<kernel>_spec` 或 `<feature>Spec`。
- **Offset / index helper** 用 shape 命名:`inOffsetDef`、`outOffsetDef`、
  `outOffsetFn`。
- **`_pos` 后缀**表示 positivity-asserting 变体 —— 直接返回值而不是
  `Option`。

## Antiquote 对 paste-in 不友好 —— 尽量少用

`$(REGION)` 包在 `tl.load($(xReg) + offs)` 里、`$ℝ(t)` 包 ℝ 字面量,这
两个都是 VeriTile 发明,Triton `.py` 里没有。对 paste-in 目标是摩擦。

新 DSL 构造:能不用 antiquote 就不用。不可避免时,把 antiquote 需求明确
记入 `documents/TritonSubset.md`,优先走 context-aware resolution 而不是
让用户显式桥接。

已记录的 paste-in 冲突:

- `tl.toReal(_)` —— VeriTile 发明;Triton 用隐式类型提升。
- 严格 `ℝ` / `Nat` 通道分离 —— 拒绝 Triton 接受的混合算术。
- "未知 kwargs 一律报错" —— 拒绝带 `cache_modifier=` /
  `eviction_policy=` 的 Triton kernel。

这些是已记录的 tradeoff,不是意外。新构造不要把这个列表越加越长。
