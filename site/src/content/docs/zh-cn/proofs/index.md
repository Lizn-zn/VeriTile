---
title: 证明与表面
description: VeriTile 对外暴露的定理 surface、证明遵循的约定、以及记录覆盖情况的 manifest。
---

本节是 VeriTile 的**证明侧**:用户面对的定理是什么形状、写证明的约定、
以及让整套 artifact 保持诚实的簿记。

## 本节包含

- [定理 surface](/VeriTile/zh-cn/proofs/theorem-surfaces/) —— 顶层形状
  (`ComputeCorrect.Realizes`、`ComputeRefine.Realizes` 等),kernel 与 spec
  用这些 surface 表达。
- [Correctness surface 指南](/VeriTile/zh-cn/proofs/correctness-surfaces/) —— 逐个
  surface 看,给定 kernel/spec 关系应该挑哪个。
- [证明约定](/VeriTile/zh-cn/proofs/proof-conventions/) —— 命名、结构、不变量
  放置、bench 语料里反复出现的模式。
- [Approx-GELU φ 策略](/VeriTile/zh-cn/proofs/approx-gelu-phi-strategy/) —— GELU /
  `φ` 系列特有的数值近似证明策略。
- [Kernel manifest](/VeriTile/zh-cn/proofs/kernel-manifest/) —— manifest 追踪什么、
  schema、CI 如何用它检测漂移。
