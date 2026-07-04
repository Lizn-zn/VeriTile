# VeriTile

[English](README.md) | **中文**

📖 **文档站:** [lizn-zn.github.io/VeriTile/](https://lizn-zn.github.io/VeriTile/)(bench 翻译手册、项目状态、架构)。本地起:`./site/scripts/dev.sh`。

VeriTile 把一个 typed Triton 风格 kernel DSL 嵌入到 Lean 4,然后证明这些 kernel
对数学规范的正确性 (correctness),或者两个 kernel 之间的等价性 (refinement)。
实现上,DSL 通过 `triton { ... }` 语法嵌入,为支持的子集定义类型化的
操作语义(typed `Op : TileDType → TileShape → Type`),并对外暴露
`ComputeCorrect` / `ComputeRefine` 两套定理 surface。

算法层证明跑在已 erase 的 `.real`(数学)通道上;可选的 `GapPolicy`
**记录但不内部证明** compute-to-algorithm gap(IEEE-754 / PTX / TMA / 并发),
这一层始终走外部检查。详见
[Triton 子集与 gap](./documents/TritonSubset_zh.md)。

## VeriTile 做什么

- **DSL**:typed Triton 子集,带 `triton { ... }` 宏、ND tile shape、reduction、
  mask(`mask=`/`other=`)、block-pointer 操作、`if`/`for` 控制流。
- **定理 surface**:`ComputeCorrect.Realizes`(kernel ↔ 数学规范)和
  `ComputeRefine.Realizes`(kernel pair 等价)。两者都通过 `toAlgorithm?`
  投影,底层走 `Kernel.Correct` / `Kernel.Refine`。
- **示例**:15 个 TritonBench-G 端口及其证明
  (见 [`bench/tritonbench_g/`](./bench/tritonbench_g/)),加上
  FlashAttention-1 forward、online softmax、Welford、LayerNorm、log-sum-exp。
- **CI gate**:`lake build` + `scripts/check-artifact.sh`(无 `sorry`、
  公理白名单、manifest schema、文档漂移检查)。
  `bench/check_ports.sh` 是独立的本地检查(不在 CI 中运行),
  逐个构建 TritonBench-G 端口。

不在范围内:IEEE-754 浮点语义、PTX 级 codegen、详细并发(原子操作 /
async-copy 序列化,投影边界以外)、Python wrapper 执行。

## Quick Start

### 1. 写一个 `ComputeKernel`

Kernel 是 region-polymorphic 的:内存区域以 `RegionName` 形参传入;
`tl.load` / `tl.store` 用一等公民的指针表达式。

```lean
def addKernel (xReg yReg outReg : RegionName) (n : Nat) : ComputeKernel := triton {
  pid     = tl.program_id(0)
  offsets = pid * $(n) + tl.arange(0, $(n))
  x       = tl.load($(xReg) + offsets)
  y       = tl.load($(yReg) + offsets)
  tl.store($(outReg) + offsets, x + y)
}
```

### 2. 选一个定理 surface

| 目标 | 用 |
|---|---|
| 一个 kernel realize 某个输出规范 | `ComputeCorrect.Realizes` |
| 两个 kernel realize 某个输出关系 | `ComputeRefine.Realizes` |
| 值 + 索引同时输出(如 `tl.max(..., return_indices=True)`)| `ComputeCorrect.OutputPairWhere` |
| 自定义 final-state 后置条件 | `ComputeCorrect.Post` / `ComputeRefine.Post` |
| 关系跨任意初始状态(罕见) | `ComputeCorrect.General` / `ComputeRefine.General` |

完整 surface 指南:[CorrectnessSurfaces.md](./documents/CorrectnessSurfaces.md)。

### 3. 通过投影到算法 kernel 来证

```lean
theorem add_kernel_correct
    (xReg yReg outReg : RegionName) (n : Nat) (hN : 0 < n)
    (s : BlockState) (xs ys : Fin n → ℝ)
    (h_x : TensorView.loadedArray s (programTileView s xReg n) xs)
    (h_y : TensorView.loadedArray s (programTileView s yReg n) ys) :
    ComputeCorrect.Realizes
      (kernel := addKernel xReg yReg outReg n)
      (initialState := s)
      (write := fun i : Fin n => some (outReg, s.pid * n + i.val))
      (expected := fun i => xs i + ys i) := by
  -- 先桥接到投影后的算法 kernel,再在 Real 语义层关 goal
  ...
```

标准套路:`ComputeKernel.computeCorrect_of_toAlgKernel rfl` 处理投影,
然后 `simp` 把 `exec` 化简到 body 递推;代数内容由 `simp` 在 spec 上
关掉,或者引用 `Mathlib` 的数学引理。LLM 证明 wrapper `scripts/prove.sh`
自动化这个循环。

### 4. 登记到 kernel manifest

往 [`scripts/kernel-manifest.tsv`](./scripts/kernel-manifest.tsv) 加一行,
让 `scripts/check-artifact.sh` 在 CI 里识别这条定理。Schema 和命名规约:
[KernelManifest.md](./documents/KernelManifest.md)、
[TheoremSurfaces.md](./documents/TheoremSurfaces.md)。

## 最小示例

逐元素向量加法对照 `addSpec xs ys i = xs i + ys i` 数学规范——
见 [`VeriTile/Examples/VectorAdd.lean`](./VeriTile/Examples/VectorAdd.lean)。

## Refinement 示例

Naive softmax vs 数值稳定 softmax(kernel pair refinement)——
见 [`VeriTile/Examples/SoftmaxEq.lean`](./VeriTile/Examples/SoftmaxEq.lean)。

## 文档地图

任务导向:

| 问题 | 文档 |
|---|---|
| 支持哪些 Triton 构造? | [TritonSubset_zh.md](./documents/TritonSubset_zh.md) |
| 用哪个定理 surface? | [CorrectnessSurfaces.md](./documents/CorrectnessSurfaces.md) |
| dtype erasure 怎么工作? | [EraseDType.md](./documents/EraseDType.md) |
| 内存安全 / framing 怎么工作? | [MemorySafety.md](./documents/MemorySafety.md) |
| GPU 内存模型是什么? | [GpuMemoryModel.md](./documents/GpuMemoryModel.md) |
| 原子操作 / async copy 怎么建模? | [ConcurrencySemantics.md](./documents/ConcurrencySemantics.md) |
| Kernel manifest 怎么用? | [KernelManifest.md](./documents/KernelManifest.md) |
| 定理 surface 命名规约 | [TheoremSurfaces.md](./documents/TheoremSurfaces.md) |
| LLM 证明 wrapper | [scripts/README.md](./scripts/README.md) |

## 目录结构

```text
VeriTile/
  Triton/                  AST、操作语义、内存、DSL、数学
  Examples/                已证 correctness/refinement 范例
bench/tritonbench_g/       TritonBench-G v1 端口(15 个已闭合)
documents/                 设计笔记、子集规范、surface 指南
scripts/                   CI gate、kernel manifest、LLM 证明 wrapper
verso/                     幻灯片 / 概览
```

## 验证

- `lake build` —— 类型检查 + 构建完整库与示例
- `scripts/check-artifact.sh` —— `lake build` ∧ 无 `sorry` ∧ 公理白名单 ∧
  kernel-manifest schema ∧ README/文档术语漂移检查
- `bench/check_ports.sh` —— TritonBench-G 端口逐个构建
  (本地手动运行;不属于 CI gate)

## 环境

- Lean 4(`v4.29.0`)+ Mathlib
- [Claude Code CLI](https://docs.claude.com/en/docs/claude-code) +
  [`lean4-skills`](https://github.com/lean4-skills/lean4-skills)
- `jq`(`scripts/prove.sh` 用)

## 路线图

长期项目。目标:把真实 Triton kernel(forward / backward / 并发 /
生产级 layout / autograd)以最小修改纳入 Lean 的证明范围。当前 roadmap:
[#91](https://github.com/Lizn-zn/VeriTile/issues/91)。架构与决策记录:
[PLAN.md](./PLAN.md)。

## License

[MIT](./LICENSE) © 2026 Zenan Li.
