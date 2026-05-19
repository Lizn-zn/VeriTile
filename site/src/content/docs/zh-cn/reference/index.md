---
title: 参考
description: 构建、脚本与处理 VeriTile 时常用的外部资源。
---

操作层面的参考。大部分链接指回仓库本身 —— 这里描述的是构建与工具,
不是设计。

## 构建与 toolchain

- **Lean toolchain**:`v4.29.0`。Pinned 在
  [`lean-toolchain`](https://github.com/Lizn-zn/VeriTile/blob/main/lean-toolchain)。
- **构建**:仓库根目录 `lake build`。
- **Manifest + sorry 检查**:`scripts/check-artifact.sh`。
- **Bench port 检查**:`bench/check_ports.sh`。

## 子项目 README

这些 README 留在仓库里,是各自领域的权威来源:

- [顶层 README](https://github.com/Lizn-zn/VeriTile/blob/main/README_zh.md)
  —— quick-start + 定理 surface 选择器。
- [`bench/tritonbench_g/` README](https://github.com/Lizn-zn/VeriTile/blob/main/bench/tritonbench_g/README_zh.md)
  —— bench 布局、port checklist、kernel 清单。
- [`scripts/` README](https://github.com/Lizn-zn/VeriTile/blob/main/scripts/README_zh.md)
  —— 各脚本作用以及 CI 如何使用它们。
- [`verso/` README](https://github.com/Lizn-zn/VeriTile/blob/main/verso/README_zh.md)
  —— 架构概览 slide deck(独立子项目)。
- [`documents/` 索引](https://github.com/Lizn-zn/VeriTile/blob/main/documents/README_zh.md)
  —— 原始 markdown 设计笔记(本站把它们重新渲染在
  [架构](/VeriTile/zh-cn/architecture/) 与 [证明](/VeriTile/zh-cn/proofs/) 下)。
