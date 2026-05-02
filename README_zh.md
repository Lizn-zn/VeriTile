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
- [研究问题](#研究问题)
- [架构](#架构)
- [目录结构](#目录结构)
- [路线图](#路线图)
- [License](#license)

## 当前状态

Phase B 已完成,核心 Phase C 栈也已经落到 `main`。见 release
[`v0.2-tier2`](https://github.com/Lizn-zn/VeriTile/releases/tag/v0.2-tier2)
(前一里程碑:[`v0.1-tier1`](https://github.com/Lizn-zn/VeriTile/releases/tag/v0.1-tier1))。

Phase B 截止包含:

- 6 个已闭合 kernel-pair refinement 定理(Tier 1 × 3 + Tier 2 × 3)。
- `forLoop` 操作语义 + `forLoop_inv` 归纳引理。
- Online softmax recurrence ≡ batch softmax(论文核心,无输入范围前提)。
- Welford online recurrence ≡ two-pass mean/variance。
- LayerNorm fused 单 pass ≡ two-pass。
- Typed `Op : TileDType → TileShape → Type`,全链路 typed `evalOp`/`stepStmt`
  与 typed 寄存器文件。
- `WithBot ℝ` 载体:`tl.full((), -inf)` 降为真正的 `⊥`,使用 `-inf` 作 max
  累加器种子的 kernel 不再需要输入范围假设。
- Triton 忠实的 mask 语义:`tl.load(p, mask=m, other=o)` 降为独立 AST 形式;
  `other=None` 通过 per-state `undef` oracle 非确定性建模。
- 6 个比较算子(`<`、`<=`、`==`、`>`、`>=`、`!=`)在 `.real`/`.nat` 通道上,
  产生 `.bool` 通道。

`main` 上已有的核心 Phase C 增量:

- ND tile shape 与 ND broadcast,以及 typed `Op : TileDType → TileShape → Type`。
- `tl.dot`、trailing-axis transpose、`tl.where`、`tl.sqrt`、reduction
  `axis` / `keep_dims`、multi-axis `tl.program_id`、strided-offset memory helper。
- 4D strided Q/K/V/O tensor view 上的 FA-1 v0/full-tile forward 证明:
  `fa1_forward_correct_4D_views` 与
  `fa1_forward_correct_4D_causal_views`。
  Boundary-masked FA-1 v1 kernel 和 recurrence scaffold 已有,但最终 correctness
  theorem 仍在推进中。
- CI 中的 artifact gate:`scripts/check-artifact.sh`,检查 `lake build`、无
  `sorry`、axiom whitelist、关键 theorem surface、README/example 漂移。

当前支持的 Triton 子集和已知语义 gap 见
[`documents/TritonSubset_zh.md`](./documents/TritonSubset_zh.md)。重要未建模部分包括
完整 IEEE-754 语义、block pointer、`boundary_check`、atomic、async copy 和 whole-grid
launch semantics。

## 环境配置

#### Lean 以及 LLM Agent

- Lean 4 (version `v4.29.0`) & Mathlib
- [Claude Code CLI](https://docs.claude.com/en/docs/claude-code)
- [`lean4-skills`](https://github.com/lean4-skills/lean4-skills), 提供 `/lean4:autoprove`

#### 其他

- `jq`, 用于解析 `scripts/prove.sh` 的 JSON 输出

## 快速开始

> **关于 DSL 的约定:** VeriTile 里的 Triton kernel 是 *region-多态* 的——每个
> kernel 都把 buffer region 作为 `RegionName` 参数,并使用接近 Triton 的
> pointer-plus-offset 语法: `tl.load($(xReg) + offs)` /
> `tl.store($(yReg) + offs, y)`。这意味着同一个 kernel 可以用任意 buffer
> 名实例化,正确性定理对所有实例化都成立。**不允许** `tl.load(X + offs)` 这种
> 裸标识符简写形式(会解析失败);设计动机见
> [RP1](./Notes/research_problem_pointer_vs_named_region.md)。

#### 1. 给定一个原始 Triton kernel. 例如 naive softmax:

```lean
def naiveSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e)
  y    := e / s
  tl.store($(yReg) + offs, y)
}
```

#### 2. 产生一个优化后的 Triton kernel. 这个优化可以由 Claude 生成, 也可以由人类专家提出. 例如数值稳定 softmax:

```lean
def stableSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x)
  e    := tl.exp(x - m)
  s    := tl.sum(e)
  y    := e / s
  tl.store($(yReg) + offs, y)
}
```

#### 3. 在 Lean 中写出两个 kernel 的等价定理, 先把证明留给 LLM Agent:

```lean
theorem softmax_kernels_refinement_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N -> Real)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    forall idx : TileIndex [N],
      TensorView.observe (exec (naiveSoftmaxKernel  xReg yReg N) s)
          (programTileView s yReg N) idx =
      TensorView.observe (exec (stableSoftmaxKernel xReg yReg N) s)
          (programTileView s yReg N) idx := by
  sorry
```

#### 4. 用 `scripts/prove.sh` 调用 LLM Agent 自动生成证明:

```bash
scripts/prove.sh path/to/your_refinement_theorem.lean --max-cycles 5
```

对 softmax refinement, 生成后的证明会把两边分别化简到各自的 specification, 再调用数学恒等式:

```lean
theorem softmax_kernels_refinement_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N -> Real)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    forall idx : TileIndex [N],
      TensorView.observe (exec (naiveSoftmaxKernel  xReg yReg N) s)
          (programTileView s yReg N) idx =
      TensorView.observe (exec (stableSoftmaxKernel xReg yReg N) s)
          (programTileView s yReg N) idx := by
  exact softmax_kernels_refinement_view xReg yReg N hN s xs h_x
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
| [`VeriTile/Examples/VectorAdd.lean`](./VeriTile/Examples/VectorAdd.lean) | 逐元素加法(multi-buffer kernel ↔ math 正确性) |
| [`VeriTile/Examples/FusedSiLU.lean`](./VeriTile/Examples/FusedSiLU.lean) | 融合 sigmoid MLP block vs 手动展开 `1/(1+exp(-z))`(kernel-pair refinement)|
| [`VeriTile/Examples/WelfordKernels.lean`](./VeriTile/Examples/WelfordKernels.lean) | Online Welford vs two-pass mean/variance |
| [`VeriTile/Examples/FlashAttention1/V0.lean`](./VeriTile/Examples/FlashAttention1/V0.lean) | FA-1 v0/full-tile forward correctness,4D strided layout,含 non-causal 与 causal |
| [`VeriTile/Examples/FlashAttention1/V1Boundary.lean`](./VeriTile/Examples/FlashAttention1/V1Boundary.lean) | FA-1 v1 boundary-mask 与 D-tail correctness,覆盖 4D strided layout |

## 更多文档

- [支持的 Triton 子集与语义 gap](./documents/TritonSubset_zh.md)
- [FA-1 Step 2 boundary-mask 计划](./documents/FA1_Step2_BoundaryMasks.md)
- [LLM 证明 Wrapper](./scripts/README.md)
- [LLM benchmark 协议](./bench/llm_eval/README.md)

## 研究问题

实施过程中需要权衡的设计决策记录在 research-problem notes 里。每条 note 说明
问题、列举设计空间、给出推荐方案、并标注何时重新评估。本节是索引,完整讨论在
[`Notes/`](./Notes) 下。

- **RP1:指针 vs 命名 region** —— DSL 是否应该建模 first-class 指针(CUDA
  风格 `x_ptr + offsets` 作为 value),还是用静态 region 名 + 动态 offset?
  已决议:Phases A–D 全部使用命名 region。
  → [research_problem_pointer_vs_named_region.md](./Notes/research_problem_pointer_vs_named_region.md)
  (英文)

- **RP2:地址类型 —— ℝ-统一 vs Nat-双化 `Value`** —— 运行时 `Value` 用
  统一的 ℝ 装载并在访存边界用 `realToNat` 取整,还是把 ℝ(数据)与 Nat
  (地址)分离成两个独立通道?已决议:双化;`realToNat` 删除;每个 kernel
  证明里的 `hcast` boilerplate(全库 ~40 行)随之消失。
  → [research_problem_address_typing.md](./Notes/research_problem_address_typing.md)
  (英文)

## 架构

类型系统是 VeriTile 正确性保证的骨架。从 DSL 宏到运行时求值，全链路 typed —— 没有
任何动态 tag union 或 existential wrapper。

```text
TileDType              TileShape                TileIndex
(.real/.nat/.bool)     (.scalar/.vec n/.mat m n) (PUnit / Fin n / Fin m × Fin n)
       │                       │                          │
   TileCarrier                 │                          │
   (ℝ / Nat / Bool)            │                          │
       │                       │                          │
       ▼                       ▼                          ▼
  ┌──────────────────────────────────────────────────────────┐
  │  structure Tile (dtype : TileDType) (shape : TileShape)  │
  │    data : TileIndex shape → TileCarrier dtype            │
  │  ────────────────────────────────────────────────────── │
  │  构造子: Tile.scalar x, Tile.vec f, Tile.mat f           │
  └──────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      Tile.bop        Tile.uop        Tile.cop
   (add/sub/mul/div)  (exp/log/σ/√)  (lt/le/eq/gt/ge/ne)
   NumericDType dtype                  → Tile .bool out
   Broadcast a b out

  ══════════════════ AST 层 ══════════════════

  inductive Op : TileDType → TileShape → Type    ← 以 dtype 和 shape 为索引
    .const ℝ           : Op .real .scalar
    .constNat Nat      : Op .nat  .scalar
    .arange n          : Op .nat  (.vec n)
    .add NumDType Broadcast Op Op  : Op dtype out
    .lt  CmpDType Broadcast Op Op  : Op .bool out
    .load region (Op .nat shape)   : Op .real shape
    ...

  inductive Stmt : Type                           ← 存在量化边界
    .assign (dtype) (shape) RegName (Op dtype shape)
    .store  region  (shape) (Op .nat shape) (Op .real shape)
    .forLoop idx n (List Stmt)

  ══════════════════ 运行时层 ══════════════════

  abbrev RegFile :=
    (dtype : TileDType) → (shape : TileShape) → RegName
      → Option (Tile dtype shape)           ← typed 查询，无 tag dispatch

  structure BlockState
    mem   : RegionName → Nat → ℝ            ← 扁平内存（仅 ℝ）
    regs  : RegFile                          ← typed 寄存器文件
    pid   : Nat                              ← program_id
    undef : RegionName → Nat → ℝ            ← masked-load oracle (other=None)

  evalOp    : Op dtype shape → BlockState → Option (Tile dtype shape)
  stepStmt  : Stmt → BlockState → Option BlockState     ┐
  stepStmts : List Stmt → ...                            │ mutual
  stepForLoopAux : ...                                   ┘
  exec      : Kernel → BlockState → Option BlockState

  ══════════════════ DSL 层 ══════════════════

  triton { ... }  宏
    │  将 tritonExpr / tritonStmt 语法展开
    │  为 typed Op / Stmt / Kernel 项
    │  kwargs: mask=, other=, axis=, keep_dims=
    ▼
  Kernel { inputs, outputs, body : List Stmt }
```

**数据流。** DSL 宏把 Triton 风格的 surface syntax 展开为 typed `Op dtype shape`
项。`evalOp` 在 `BlockState` 上求值 `Op`，返回 `Option (Tile dtype shape)` ——
类型索引全程穿透。`stepStmt` 对 `Stmt` 做 pattern match（`Stmt` 的每个构造子
内部存在量化了 `dtype` 和 `shape`），立即恢复具体的索引，再调用 `evalOp`。结果
存入 `RegFile`——一个以 `(dtype, shape, name)` 为索引的依赖类型函数。
**全链路没有任何 erase-and-recover 环节。**

## 目录结构

```text
VeriTile/
  Triton/
    Core.lean          Kernel AST
    Semantics.lean     操作语义
    Memory.lean        memory contract、offset、tensor view
    DSL.lean           `triton { ... }` 宏
    Examples.lean      直接构造子形式示例
  Examples/
    SoftmaxEq.lean
    LogSumExpEq.lean
    SoftmaxReciprocal.lean
    WelfordKernels.lean
    FlashAttention1.lean

bench/llm_eval/        Held-out 证明评测 harness
scripts/               证明 wrapper 脚本
documents/             项目文档
Notes/                 Proposal, 实现笔记, 可行性研究
PLAN_zh.md             项目路线图
VeriTile.lean          Lean library 顶层入口
lakefile.toml          Lake 项目定义
lean-toolchain         锁定的 Lean toolchain
```

## 路线图

- **Phase A:** Tier 1 refinement 示例, LLM proof wrapper, T3 scouting. 已完成.
- **Phase B:** `forLoop` 语义, Tier 2 kernel, differential testing. 已完成.
- **Phase C:** ND tile、masking、`tl.dot`、strided memory、FA-1 forward
  correctness。核心 non-causal / causal 4D theorem 已在 `main`。
- **Phase D:** 收敛剩余语义 gap:block pointer、`boundary_check`、IEEE-754
  保真、whole-grid launch semantics、FA-1 vs FA-2、论文 artifact packaging。

完整计划见 [`PLAN_zh.md`](./PLAN_zh.md).

## License

[MIT](./LICENSE) © 2026 Zenan Li.
