# VeriTile

[English](README.md) | **中文**

VeriTile 是一个实验性的 Lean 4 项目, 用于 Triton GPU 算子及其优化的翻译验证
(translation validation). 项目试图证明 Triton 算子的某些性质, 以及优化前后
Triton kernel pair 的等价性.

实现方法是把一个小型 Triton 风格 kernel 语言嵌入到 Lean 中, 为当前支持的子集
定义操作语义, 并证明单个 kernel 满足某些算法性质, 以及某些 kernel pair 产生相同的可观测输出.
证明过程通过 LLM Agent 完成.

## 目录

- [当前状态](#当前状态)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [其他示例](#其他示例)
- [更多文档](#更多文档)
- [目录结构](#目录结构)
- [路线图](#路线图)
- [License](#license)

## 当前状态

Phase A 已完成. 见 release
[`v0.1-tier1`](https://github.com/Lizn-zn/VeriTile/releases/tag/v0.1-tier1).

Phase A 包含:

- 3 个已闭合的 Tier 1 kernel-pair refinement 定理.
- 当前 Triton 子集的 `triton { ... }` Lean 宏.
- gather/scatter tile load/store 的操作语义.
- held-out LLM 证明评测 harness.
- FlashAttention forward 可行性研究.

## 环境配置

#### Lean 以及 LLM Agent

- Lean 4 (version `v4.29.0`) & Mathlib
- [Claude Code CLI](https://docs.claude.com/en/docs/claude-code)
- [`lean4-skills`](https://github.com/lean4-skills/lean4-skills), 提供 `/lean4:autoprove`

#### 其他

- `jq`, 用于解析 `scripts/prove.sh` 的 JSON 输出

## 快速开始

#### 1. 给定一个原始 Triton kernel. 例如 naive softmax:

```lean
def naiveSoftmaxKernel (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load(X, offs)
  e    := tl.exp(x)
  s    := tl.sum(e)
  y    := e / s
  tl.store(Y, offs, y)
}
```

#### 2. 产生一个优化后的 Triton kernel. 这个优化可以由 Claude 生成, 也可以由人类专家提出. 例如数值稳定 softmax:

```lean
def stableSoftmaxKernel (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load(X, offs)
  m    := tl.max(x)
  e    := tl.exp(x - m)
  s    := tl.sum(e)
  y    := e / s
  tl.store(Y, offs, y)
}
```

#### 3. 在 Lean 中写出两个 kernel 的等价定理, 先把证明留给 LLM Agent:

```lean
theorem softmax_kernels_refinement
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N -> Real)
    (h_x : InputLoaded s N xs) :
    forall i : Fin N,
      observeY (exec (naiveSoftmaxKernel N) s) N s.pid i =
      observeY (exec (stableSoftmaxKernel N) s) N s.pid i := by
  sorry
```

#### 4. 用 `scripts/prove.sh` 调用 LLM Agent 自动生成证明:

```bash
scripts/prove.sh path/to/your_refinement_theorem.lean --max-cycles 5
```

对 softmax refinement, 生成后的证明会把两边分别化简到各自的 specification, 再调用数学恒等式:

```lean
theorem softmax_kernels_refinement
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N -> Real)
    (h_x : InputLoaded s N xs) :
    forall i : Fin N,
      observeY (exec (naiveSoftmaxKernel N) s) N s.pid i =
      observeY (exec (stableSoftmaxKernel N) s) N s.pid i := by
  intro i
  rw [softmax_naive_correct N hN s xs h_x i,
      softmax_stable_correct N hN s xs h_x i]
  congr 1
  unfold naiveSpec stableSpec
  have h := naive_eq_stable xs (tileMax hN xs)
  simp only [] at h
  exact congrFun h i
```

完整 softmax 例子见 [`VeriTile/Examples/SoftmaxEq.lean`](./VeriTile/Examples/SoftmaxEq.lean).

## 其他示例

主要示例:

| 文件 | 说明 |
| --- | --- |
| [`VeriTile/Examples/SoftmaxEq.lean`](./VeriTile/Examples/SoftmaxEq.lean) | naive softmax vs 数值稳定 softmax |
| [`VeriTile/Examples/LogSumExpEq.lean`](./VeriTile/Examples/LogSumExpEq.lean) | 直接 log-sum-exp vs shift-trick log-sum-exp |
| [`VeriTile/Examples/SoftmaxReciprocal.lean`](./VeriTile/Examples/SoftmaxReciprocal.lean) | stable softmax 除法 vs 预计算倒数 |
| [`VeriTile/Examples/WelfordMath.lean`](./VeriTile/Examples/WelfordMath.lean) | Phase B 准备用的 Welford 恒等式 |

## 更多文档

- [支持的 Triton 子集](./documents/TritonSubset_zh.md)
- [LLM 证明 Wrapper](./scripts/README.md)
- [LLM benchmark 协议](./bench/llm_eval/README.md)

## 目录结构

```text
VeriTile/
  Triton/
    Core.lean          Kernel AST
    Semantics.lean     操作语义
    DSL.lean           `triton { ... }` 宏
    Examples.lean      直接构造子形式示例
  Examples/
    SoftmaxEq.lean
    LogSumExpEq.lean
    SoftmaxReciprocal.lean
    WelfordMath.lean

bench/llm_eval/        Held-out 证明评测 harness
scripts/               证明 wrapper 脚本
Notes/                 Proposal, 实现笔记, 可行性研究
PLAN_zh.md             项目路线图
VeriTile.lean          Lean library 顶层入口
lakefile.toml          Lake 项目定义
lean-toolchain         锁定的 Lean toolchain
```

## 路线图

- **Phase A:** Tier 1 refinement 示例, LLM proof wrapper, T3 scouting. 已完成.
- **Phase B:** `forLoop` 语义, Tier 2 kernel, differential testing.
- **Phase C:** `tl.dot`, masking, 2D tile, FlashAttention forward 证明.
- **Phase D:** multi-block 执行, FA-1 vs FA-2, 论文 artifact.

完整计划见 [`PLAN_zh.md`](./PLAN_zh.md).

## License

[MIT](./LICENSE) © 2026 Zenan Li.
