---
title: "VeriTile 代码组织"
---

VeriTile 把三件事分到三层。知道某个东西属于哪一层,以后加新 operator、新桥
引理、新 kernel 转写时不会乱搬。

## 三层

```
                    ┌─────────────────────────────────────────────┐
                    │  bench/tritonbench_g/<kernel>/X.lean        │  ← per-kernel glue
   per-kernel       │  VeriTile/Examples/X.lean                   │
                    │  - kernel 定义 (`triton { ... }`)           │
                    │  - kernel 专用 *Spec / *Load / *Offset       │
                    │  - kernel correctness theorem                │
                    └─────────────────────────────────────────────┘
                                       │ uses
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │  VeriTile/Triton/Semantics/X.lean           │  ← 桥接机制
    bridging        │  - `Tile` / `WithBot` / `Option.map₂`       │
                    │  - tiled indexing (lane / validLanes / ...) │
                    │  - masked reductions (sup' / sum / dot)     │
                    │  - kernel-agnostic 但绑 Triton              │
                    └─────────────────────────────────────────────┘
                                       │ uses
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │  VeriTile/Triton/Math/X.lean                │  ← 纯数学
   pure math        │  - `(Fin N → ℝ) → ...` 算子                 │
                    │  - 非平凡的数学等式                         │
                    │  - 只依赖 Mathlib                            │
                    └─────────────────────────────────────────────┘
```

每一层只能依赖它下面的层。

## 各层职责

### `VeriTile/Triton/Math/`

**命名**: 按数学算子。`Math/Activation.lean`, `Math/Reduction.lean`,
`Math/L2Norm.lean`, `Math/LogSumExp.lean`, `Math/Softmax.lean`,
`Math/Loss.lean`。

**应该放进来**:
- 签名是 `Fin N → ℝ`(或 `ℝ → ℝ` 等)→ ℝ 的定义,**不**出现 `BlockState`,
  `RegionName`, `WithBot`, `Tile`。
- 这些算子之间的恒等式(如 `naive_softmax = stable_softmax`,
  `welford_running = twoPass`)。

**入选规则**: ≥ 2 个 caller(或者 1 个 caller 加上即将出现的第二个),并
且定义有非平凡数学内容(不是一行 argument-binding wrapper)。

**不应该放进来**:
- Kernel layout(`s.pid * stride + i.val`)
- `WithBot` / `Tile` / `Option.map₂` —— 那是 `Semantics/` 的事
- 每个 kernel 自己的 `*Spec`,只是把数学绑到 kernel 参数元组上

### `VeriTile/Triton/Semantics/`

**命名**: 按**机制**,**不**按 operator。`Semantics/TiledIndexing.lean`,
`Semantics/MaskedReduction.lean`, `Semantics/Step.lean`,
`Semantics/State.lean`。(未来可能的:`StreamingAccumulator`,
`AtomicReduction`, `BroadcastReshape`。)

一个"机制"是"`Tile` / `WithBot` / `Option.map₂` 与某一类数学算子之间的连
接方式"。每个 `Semantics/X.lean` 拥有一种机制,保持 kernel-agnostic。

**应该放进来**:
- Lane / block 索引原语和分区引理
  (`laneIdx`, `validLanes`, `blockIndex`, `sum_exp_partition`).
- WithBot / Tile carrier 桥引理
  (`withBot_sup'_partial`, `reduceSum_masked_sq_eq_some_sum`,
  `reduceSum_masked_dot_eq_some_sum`).
- 凡是泛化在 `(load, active)` 或 `(load, mask)` 上的引理 —— 任何 L2 / softmax /
  reduction 类 kernel 都可以代入。

**不应该放进来**:
- 某个 kernel 的具体 `s.pid * stride` 代数 —— 那是 per-kernel 层。
- 非 Triton 数学 —— 那是 `Math/`。
- 只有一个 kernel 在用的机制: 先内联在那个 bench 文件里,等出现第二个 caller
  再 promote(跟 `Math/` 的 "≥ 2 callers" 规则一致)。

**为什么按机制不按 operator 命名**: 一个 operator(比如 softmax)通常用到
多种机制(tiled indexing + masked reduction)。一个机制(比如 masked
reduction)会被多种 operator 复用。按机制命名,去重才对得上 ——
`MaskedReduction.lean` 同时容纳 softmax / log-sum-exp / L2 / masked-max 的桥
引理。

### `bench/tritonbench_g/<kernel>/`, `VeriTile/Examples/`

**命名**: 按 kernel(`L2NormTriton1.lean`, `FlashAttention1.lean`)。

**应该放进来**:
- `triton { ... }` kernel 转写
- Kernel-local 辅助: `*Offset`, `*Load`, `*Carrier`, `*Spec`
- Kernel correctness theorem(algorithm-layer + compute-facing)

**模式**: glue code。`*Spec` 写成"这个 kernel 写出
`Triton.TiledX.operator (load_xs ...) ...`",通过 `Semantics/` 的桥引理
把 kernel layout 接到数学算子。

## 加新东西时的归属判断

写新引理或新定义时,问自己:

1. **碰到 `BlockState` / `RegionName` 或 kernel layout 没?**
   → per-kernel glue。放 kernel 文件里。

2. **碰到 `Tile` / `WithBot` / `Option.map₂`,但泛化在 kernel 的 load 和
   mask 上?**
   → 机制。放进现有 `Semantics/X.lean` 中主题最匹配的那个;或者等到第二个 caller
   出现,再开一个新的 `Semantics/<机制名>.lean`。

3. **是纯 `(Fin N → ℝ) → ...` 而且 ≥ 2 个 kernel 复用?**
   → 数学算子。按 operator 名字进 `Math/X.lean`。

4. **只是 Mathlib + kernel 参数名的一行组合?**
   → 留在 kernel 文件里。**不要**移到 `Math/`。

## 相关

- [`ProofConventions.md`](/VeriTile/zh-cn/proofs/proof-conventions/) —— 证明 tactic 约定,包括
  `erw` 作为 carrier-bridge 的 fallback。
- [`CorrectnessSurfaces.md`](/VeriTile/zh-cn/proofs/correctness-surfaces/) —— 用户面 theorem
  surface(`Realizes`, `WriteMap`, `OutputReadable`)。
