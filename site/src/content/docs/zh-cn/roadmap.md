---
title: 路线图
description: VeriTile 下一步 —— substrate 工作、bench 扩展、待解设计问题。
---

路线图按**解锁 bench 扩展的 substrate 工作**组织,而不是 kernel-by-kernel
清单。一次 substrate 解锁(比如 `tl.dot` 算法模型)往往一次性闭掉 10+ 个
bench 条目,所以杠杆在 substrate。

:::caution[草稿]
本页是 working draft。优先级排序和时间反映当下对开放 issue 集群的最佳
猜测,**不是承诺计划**。做项目级决策前,核对
[实时 issue 列表](https://github.com/Lizn-zn/VeriTile/issues)。
:::

## 顶层 substrate 缺口

[状态页](/VeriTile/zh-cn/status/) 的 bench 覆盖普查列出八个 substrate 缺口,
合在一起占了未证 bench 大头。按预期杠杆粗略排序:

### 1. `tl.dot` 算法模型

**解锁:**全部 10 个 matmul、14 个 flash / attention 系列、
`attn_fwd_causal`、所有用 `tl.dot` 的 chunk kernel。可能是项目里单点
杠杆最高的 substrate item。

**工作形状:**typed `tl.dot` 在 Compute 层对 `Real` 矩阵乘法的模型,
加上 `tl.dot` 位于 `forLoop_inv` body 内时需要的循环归纳刻画。issue
#93 跟踪,#128 / #135 等引用。

### 2. Streaming softmax invariant chain

**解锁:**attention forward kernel、`token_softmax_*`、`token_attn_*`。
基于 #1,因为这些 kernel 同时含 `tl.dot` 和 streaming softmax。

**工作形状:**把 `(m, l, O)` accumulator 提升为 `Semantics/StreamingAccumulator.lean`
里的可复用不变量(issue #112)。

### 3. forLoop 包裹的 store 证明

**解锁:**`rope_embedding`、`fast_rope_embedding`、`kv_cache_filling`、
`kv_cache_copy`。现有 slice 证明覆盖一次迭代;完整 loop 证明需要带交错
store 的 per-iteration invariant 配 `forLoop_inv`。

**工作形状:**类比现有跨区域 strip,但面向**同区域**交错 store 的
offset-set 不重合 helper。helper 落地后,每 kernel 估计 100–300 行。

### 4. 多 block layer norm / RMS norm

**解锁:**`rmsnorm_implementation`(#122)、`layernorm_fwd_triton`
(#123)、`layer_norm_ops`(#133)。**不是** substrate 卡 ——
`forRange_inv` / `forLoop_inv` 等 helper 都在;工作量在跨循环 scalar
穿线。每 kernel 多周。先例在
`bench/examples/FusedLayerNorm.lean` 和 `OnlineSoftmax.lean`。

### 5. `tl.cumsum` 方向感知语义

**解锁:**带 backward / reverse 方向的 recurrent / cumsum kernel
(issue #94)。以及把 cumsum 跟 streaming 组合的 kernel。

### 6. Int 舍入 + packed int4 / int8

**解锁:**`int8_quantization`(issue #129)、量化 KV per-block scales
(#137)。需要 Compute 层 `llrint` / int8-cast 构建,加上 int4 打包
语义(issue #95)。

### 7. 有符号指针运算

**解锁:**`conv2d` padded(issue #130)。需要 typed `Int` 指针模型。

### 8. constexpr 分支 case-split

**解锁:**`rotary_transform`(4 分支 `Stmt.ifThenElse`)、
`layer_norm_ops` 完整 forward(4 Bool flag → 16 组合)。工作在 tactic
层,不在 substrate 层。

## 近期 bench 扩展

substrate 之外:`bench/tritonbench_g/` 里 **43 个 README-only 脚手架**
还没有 `.py` / `.lean`。一部分卡在 substrate 后面(attention 系列等
#93 / #135),另一部分可以独立推进。开放问题:哪些脚手架应该插到
substrate 队列前面?

## DSL / surface 演进

列出来是因为每个 bench port 都受影响,不是因为迫在眉睫:

- **减少 / 移除 `$(REGION)` antiquote** 服务 paste-in 目标 —— 能用
  context-aware resolution 就用。
- **`tl.toReal(_)` 移除** —— Triton 用隐式类型提升;VeriTile 的显式形式
  对 paste-in 不友好。
- **`ℝ` / `Nat` 混合算术** —— Triton 接受;VeriTile 当前拒绝。与通道
  分离的 tradeoff。
- **未知 kwarg 策略** —— 当前对 `cache_modifier=` / `eviction_policy=`
  hard-fail。可能需要放宽到 "warn + ignore"。

## 待解设计问题

更整洁的答案尚未承诺的几条:

1. **Refinement 在 bench 的覆盖** —— surface 在,但 `bench/` 里
   `ComputeRefine.Refines_without_Rounding` 零调用。语料里有适合 "kernel ↔ kernel"
   的对吗?还是这个 surface 只在 `VeriTile/Examples/` 里值回票价?
2. **Manifest 作 source of truth** —— `KernelManifest.md` 记录"追踪
   什么";bench 状态(slice / full / substrate-blocked)是否应该编码
   *进* manifest + CI 强制,而不是从 `grep` 重算?
3. **Verso 架构 deck** —— `verso/` 的 slide deck 独立维护。要不要并入
   本站当 `/overview/` 页,还是因为它独立 toolchain(`v4.30.0-rc2`)
   保持独立?
4. **i18n 对等** —— 本站每页今天都 EN/ZH 各一份。长期值得吗?还是
   某些页(changelog、raw API 参考)只做英文?

## 优先级是怎么定的

这一页是**描述性**的 —— 什么开着、什么卡着。优先级在站外定(issue
triage、项目讨论),这页跟着调整。如果发现这页落后于实际方向,正确做法
是更新这页,而不是悄悄改方向。
