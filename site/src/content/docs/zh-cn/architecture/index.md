---
title: 架构与语义
description: VeriTile 各部分如何拼起来 —— 分层、支持的 Triton 子集、证明所依赖的语义模型。
---

本节是 VeriTile 的**规范侧**:代码如何分层、DSL 支持哪些 Triton 特性、
证明承诺哪些语义模型。

## 本节包含

- [代码组织](/VeriTile/zh-cn/architecture/code-organization/) —— 三层架构、什么在哪一层、
  加一个 operator / bridge lemma / 新 kernel 翻译时一般要改哪里。
- [Triton 子集](/VeriTile/zh-cn/architecture/triton-subset/) —— 实际嵌入的 Triton-like
  surface syntax,哪些建模 / 哪些故意不在范围内。
- [Erase + dtype](/VeriTile/zh-cn/architecture/erase-dtype/) —— typed `Op` 项如何通过
  `toAlgorithm?` 投影到数学(`ℝ` / `ℤ`)通道供算法证明使用。
- [GPU 内存模型](/VeriTile/zh-cn/architecture/gpu-memory-model/) —— kernel 语义对
  region、指针、读 / 写做了什么假设。
- [并发语义](/VeriTile/zh-cn/architecture/concurrency-semantics/) —— 串行化投影模型,
  以及刻意*不*作的承诺。
- [内存安全](/VeriTile/zh-cn/architecture/memory-safety/) —— `tl.load` / `tl.store`、
  mask、边界的安全性。
