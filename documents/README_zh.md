# VeriTile 参考文档

[English](README.md) | **中文**

VeriTile 嵌入的 Triton 子集、语义模型,以及 trusted-boundary 策略的长篇参考。
每个 active 文档都是一份 *contract*:建模了什么、未建模什么、边界在哪里。

已闭合阶段的设计笔记和已解决 research-problem 记录放在
[`archive/`](./archive/) —— 作为历史 artifact 保留;不再为当前工作负责。

## Active 参考文档

任务导向索引(对应顶层 [`README.md`](../README.md) 的 Documentation Map):

| 问题 | 文档 |
|---|---|
| 支持哪些 Triton 构造? | [TritonSubset.md](./TritonSubset.md) ([中文](./TritonSubset_zh.md)) |
| 用哪个定理 surface? | [CorrectnessSurfaces.md](./CorrectnessSurfaces.md) |
| 定理 surface 命名规约 | [TheoremSurfaces_zh.md](./TheoremSurfaces_zh.md) |
| Kernel manifest 怎么用? | [KernelManifest_zh.md](./KernelManifest_zh.md) |
| dtype erasure 怎么工作? | [EraseDType.md](./EraseDType.md) |
| 内存安全 / framing 怎么工作? | [MemorySafety_zh.md](./MemorySafety_zh.md) |
| GPU 内存模型是什么? | [GpuMemoryModel.md](./GpuMemoryModel.md) |
| 原子操作 / async copy 怎么建模? | [ConcurrencySemantics.md](./ConcurrencySemantics.md) |
| ApproxGeLU 中段 certified-error 策略 | [ApproxGeluPhiStrategy.md](./ApproxGeluPhiStrategy.md) |

## 架构 / 路线图

端到端项目计划与 roadmap 放在仓库根目录:

- [`../PLAN.md`](../PLAN.md) —— 架构、决策日志、phase 状态
- 实时 roadmap:GitHub issue
  [`#91`](https://github.com/Lizn-zn/VeriTile/issues/91)

## Archive

已闭合阶段的笔记保留在 [`archive/`](./archive/):

- `DslMacroOptions.md` —— macro 设计探索,决策为 option C
  (`triton { ... }` block macro)。已闭合。
- `ForLoopInvDesign.md` —— Phase B `forLoop_inv` 接口决策。
  已闭合;当前实现与之一致。
- `ResearchProblemPointerRegion.md` —— RP1:pointer vs named region。
  结论:全程使用 named region。
- `ResearchProblemAddressTyping.md` —— RP2:ℝ-uniform vs Nat-bifurcated
  `Value`。结论:bifurcated `Value`。
- `Proposal.md` / `Proposal_zh.md` —— 初始项目 proposal。技术内容已经
  超出它的范围;保留作为历史背景。
