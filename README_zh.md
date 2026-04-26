# VeriTile

[English](README.md) | **中文**

基于 LLM 辅助 Lean 4 证明的 Triton GPU kernel 翻译验证 (translation validation)。

完整 proposal 见 `Notes/proposal.md`(英文)/ `Notes/proposal_zh.md`(中文)。Program plan 见 `PLAN.md` / `PLAN_zh.md`。

**当前状态:** Phase A 已完成 —— 见 release [`v0.1-tier1`](https://github.com/Lizn-zn/VeriTile/releases/tag/v0.1-tier1)。3 个 Tier 1 kernel-pair 定理闭合,T3(FlashAttention)可行性研究交付,LLM 证明起草 wrapper 上线。

## 这个仓库展示了什么

两个 Triton kernel(naive softmax 与数值稳定 softmax)通过 `triton { ... }`
宏嵌入到 Lean 中,并基于我们的操作语义证明了它们的**算法等价性** ——
形式与 arm-in-lean 的 `tnum_const_refinement` 同构:

```lean
-- 两个 kernel(Lean 内部使用 Triton 风格语法):
def naiveSoftmaxKernel (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load(X, offs)
  e    := tl.exp(x)
  s    := tl.sum(e)
  y    := e / s
  tl.store(Y, offs, y)
}

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

-- Refinement 定理(对标 `tnum_const_refinement`):
theorem softmax_kernels_refinement
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (naiveSoftmaxKernel N) s) N s.pid i =
      observeY (exec (stableSoftmaxKernel N) s) N s.pid i
```

定理含义:**对相同输入,两个 kernel 在 `Y` 区域写入的值逐位置相等。**
这是观测意义下的(算法层)等价 —— 两个 kernel 的中间寄存器
(`m`, `e`, `s`)有意不同;若主张完整 `BlockState` 相等,会是一个严格更强
(且为假)的命题。

另外两对 kernel(LSE 直算 vs shift-trick、softmax `e/s` vs 预算倒数)采用同样的模式;见 §Refinement 结构。

## 环境(prepare)

### 必备

- **[Elan](https://github.com/leanprover/elan)** —— Lean 4 工具链管理器。
  通过 `curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh` 安装。
- **Lake** —— 随 Elan 自带,Lean 构建管理。
- **Lean 4** v4.15.0(由 `lean-toolchain` 锁定;v4.29.0 上也可用)。
- **Mathlib** —— `lake update` 自动拉取。

用户 shell 不会自动 source `.elanrc`,所以在跑 lake / lean 的 shell 里手动加 `PATH`:

```bash
export PATH="$HOME/.elan/bin:$PATH"
```

或对每条命令前缀 `PATH=...`(下面 §Usage 的例子都内联了)。

### 可选(LLM 证明 wrapper 用)

- **[Claude Code CLI](https://docs.claude.com/en/docs/claude-code)** —— `claude` 在 PATH 上。
- **[lean4-skills 插件](https://github.com/lean4-skills/lean4-skills)** —— 提供 `/lean4:autoprove`。安装:`claude /plugin add lean4/lean4-skills`(或你偏好的 plugin 安装方式)。
- **`jq`** —— `scripts/prove.sh` 解析 JSON 输出用。macOS:`brew install jq`。

不安装这些也能正常构建和阅览项目,只有 `scripts/prove.sh`(及 held-out eval)需要它们。

### 可选(后续阶段做 differential testing)

- **GPU + Triton + flash-attn** —— Phase B–D 才用,differential testing 工件需要(见 PLAN.md §Differential testing)。Phase A 不需要。

## 使用(Usage)

### 构建项目

```bash
PATH="$HOME/.elan/bin:$PATH" lake update     # 约 5–15 分钟,首次拉 Mathlib
PATH="$HOME/.elan/bin:$PATH" lake build
```

预期:构建干净,无警告,无 sorry。

### 浏览一个 worked example

最快了解项目做什么的方式:

```bash
$EDITOR VeriTile/Examples/SoftmaxEq.lean       # 原型,注释最详细
$EDITOR VeriTile/Examples/LogSumExpEq.lean     # log-sum-exp 对(单 scalar 输出)
$EDITOR VeriTile/Examples/SoftmaxReciprocal.lean   # 用 1/s 省 N 次除法
$EDITOR VeriTile/Examples/WelfordMath.lean     # math-only Welford 恒等式(Phase B 准备)
```

每个文件依次给出:`triton { ... }` 宏写的 kernel 定义 → 数学恒等式 → kernel 正确性引理 → headline refinement 定理。

### 在 held-out 定理上跑 LLM 证明 wrapper

```bash
# Dry run / 参数检查
scripts/prove.sh --help

# 真跑(经 claude 调 /lean4:autoprove,会烧 API budget)
scripts/prove.sh bench/llm_eval/softmax_naive_correct_held_out.lean --max-cycles 5
```

每次 trial 的 JSON log 落在 `Logs/`;成功 / 失败由对(可能被修改的)文件重跑 `lake env lean` 判定。

### 跑 Phase A 的 5-trial eval

```bash
mkdir -p bench/llm_eval/results
> bench/llm_eval/results/summary.log
for trial in 1 2 3 4 5; do
  TRIAL_FILE="/tmp/heldout_$(date +%s)_${trial}.lean"
  cp bench/llm_eval/softmax_naive_correct_held_out.lean "${TRIAL_FILE}"
  scripts/prove.sh "${TRIAL_FILE}" --max-cycles 5 \
    >> bench/llm_eval/results/summary.log 2>&1
done
grep -c "\[SUCCESS\]" bench/llm_eval/results/summary.log
```

Phase A eval 报告在 `bench/llm_eval/results/phase_a_report.md`。

### 加一个新的 kernel-pair 例子

参照 `LogSumExpEq.lean`(最简单)的结构:

1. 用 `triton { ... }` 宏定义两个 kernel
2. 写出数学恒等式(`theorem ... : math_a = math_b`)
3. 写两个 correctness 引理(每个 kernel 通过对操作语义 `simp` 化简到闭式)
4. 把它们组合成 headline refinement 定理
5. 把文件加进 `VeriTile.lean` 的 import 列表
6. `lake build` —— 应该无 sorry 编译过

`scripts/prove.sh` 可以试着帮你证那个数学恒等式;操作语义 walk-through 通常是从已有 kernel 抄改 2–10 行。

## Triton 子集(当前范围)

**已纳入:**
* `tl.load`, `tl.store` —— 都支持 **gather/scatter**(tile 值作为 offset)。
* `tl.arange`, `tl.broadcast`, `tl.full`,标量 / 张量常量。
* `tl.exp`, `tl.log`, `tl.maximum`,基本算术(`+ − * /`)。
* `tl.max`, `tl.sum` —— block 级 reduction(axis = 0),通过 Mathlib
  `Finset.sup'` / `Finset.sum` 定义。
* `tl.program_id`, `tl.constexpr`。
* `assign`, `store`。`forLoop` AST 已存在,但操作语义返回 `none`。

**未纳入(Phase B+):**
* `tl.atomic_*`, `tl.dot`,异步 copy。
* 多 block 协调、cluster 级算子。
* Hopper / Blackwell 算子(TMA、WGMMA)。
* `forLoop` 操作语义 —— 需要与 `stepStmts` 互递归,并显式
  `termination_by (sizeOf body + 1, n − i)` 度量。Phase B 工作。

**信任假设:**浮点算术建模在 `ℝ`(Mathlib `Real`)上。IEEE-754 保真不在本项目范围内;differential testing 工件(Phase B+)在数值上抓粗 divergence,不到位级。

## `triton { ... }` 宏

参照 arm-in-lean 的 `arm64 { ... }` 模式。把 Triton 风格的代码块
lower 到我们的 `Op` / `Stmt` / `Kernel` AST。

约定:
* 裸的小写标识符 → 寄存器引用(`pid`, `x`, `e`)。
* `tl.load(...)` / `tl.store(...)` 的第一个参数 → 裸标识符,被解释为
  memory region 名(字符串字面量)。
* `$(LeanTerm)` → 反引(antiquote)一个 Lean 层值(用于 kernel 的
  `tl.constexpr` 大小参数 `N`)。
* 数字字面量 → `Op.const`。
* 语句以换行分隔(`tritonStmt*` 语法类别)。

`Notes/MacroOptions.md` 讨论了为什么选择宏语法,而非直接调用构造子 /
外部 S 表达式 / scoped notation,以及各方案的取舍。

## Refinement 结构

形态与 `tnum_const_refinement` 一致:`exec` 跑每个 kernel,观测函数
(我们这边是 `readMem`,arm-in-lean 那边是 `readReg`)抽出输出通道,
定理断言逐位置相等。数学步骤被隔离在 `naive_eq_stable` 中;操作语义
walk-through 通过 simp 在我们的 gather/scatter 语义上把
`exec kernel s` 化简为闭式。

### Tier 1(当前范围,全部闭合)

| 定理 | 文件 | 数学恒等式 | 状态 |
|---|---|---|---|
| `softmax_kernels_refinement` | `Examples/SoftmaxEq.lean` | `naive_eq_stable` | ✓ |
| `log_sum_exp_refinement` | `Examples/LogSumExpEq.lean` | `log_sum_exp_shift_invariant` | ✓ |
| `softmax_reciprocal_refinement` | `Examples/SoftmaxReciprocal.lean` | `div_eq_mul_inv_real` | ✓ |
| `welford_eq_two_pass`(数学引理,Phase B 准备) | `Examples/WelfordMath.lean` | 自身 | ✓ |

每个 kernel-pair 定理把一个数学恒等式与两个操作语义 walk-through(每个 kernel 一个)组合,最后委派给 `scatter_readback` 做 tile-write 的 readback。

### 详解:softmax(原型)

```
softmax_kernels_refinement              ✓ FULL PROOF
  ├─ softmax_naive_correct              ✓ FULL PROOF (操作语义 walk-through)
  ├─ softmax_stable_correct             ✓ FULL PROOF (操作语义 walk-through)
  └─ naive_eq_stable                    ✓ FULL PROOF (数学;6 行 tactic)
                                            使用 Real.exp_sub, Finset.sum_div,
                                            Real.exp_ne_zero, field_simp

操作语义辅助:
  scatter_readback                      ✓ FULL PROOF (List.foldl 在不同 offset
                                            上的归纳,基于 Nodup +
                                            offsetFn 的 injectivity)
```

数学内容被隔离在 `naive_eq_stable` 中;两个
`softmax_*_correct` 引理是纯操作语义 walk-through,通过对
`exec / stepStmts / stepStmt / evalOp / Value.bop / writeMem` 做 simp
把 `exec kernel s` 化简到闭式输出,最后委派给 `scatter_readback`
完成 readback 步骤。`scatter_readback` 自身是 `List.foldl` 上的归纳:
将 `List.finRange n = l₁ ++ i :: l₂` 拆分,用 `Nodup` + injectivity 证明
后续写入都不会触碰目标 cell,然后读出位置 `i` 写入的值。其他两个 Tier 1 定理结构一样(LSE 在 `pid` 单点写一个 scalar 而不是 tile,所以跳过 `scatter_readback`)。

## 文件布局

```
README.md / README_zh.md         本文件(英 / 中)
PLAN.md / PLAN_zh.md             Program plan(英 / 中)
LICENSE                          MIT
lakefile.toml, lean-toolchain    Lake 构建(Lean 4 v4.15.0;在 v4.29.0 上也可用)
VeriTile.lean                    顶层库入口

VeriTile/Triton/
  Core.lean                      Op / Stmt / Kernel 数据类型
  Semantics.lean                 BlockState、evalOp(gather)、stepStmt(scatter)、
                                   exec、scatter_readback
  DSL.lean                       `triton { ... }` 宏
  Examples.lean                  手工构造的 naiveSoftmax kernel(直接构造子形式)

VeriTile/Examples/
  SoftmaxEq.lean                 Naive ↔ stable softmax(Tier 1 #1)
  LogSumExpEq.lean               Direct LSE ↔ shift-trick LSE(Tier 1 #2)
  SoftmaxReciprocal.lean         Stable softmax ↔ 预算倒数(Tier 1 #3)
  WelfordMath.lean               welford_eq_two_pass 数学引理(Phase B 准备)

scripts/
  prove.sh                       封装 /lean4:autoprove 的 bash wrapper,benchmark eval 用
  README.md                      Wrapper 用法 + pin 的 plugin 版本

bench/llm_eval/
  softmax_naive_correct_held_out.lean   LLM eval 的 held-out 副本
  README.md                      Eval 协议
  results/                       每 trial 日志 + phase_a_report.md

Notes/
  proposal.md / proposal_zh.md   项目 proposal(英 / 中)
  T3_scouting.md                 FlashAttention forward 可行性研究
                                   (Phase A 交付物,~15 页)
  MacroOptions.md                技术调研:宏实现的几种取舍
  2026-04-26-phase-a-implementation.md   Phase A 实施计划
```

## 路线图

**Phase A —— Tier 1 + LLM 工具 + T3 scouting(已完成,[`v0.1-tier1`](https://github.com/Lizn-zn/VeriTile/releases/tag/v0.1-tier1)):**
- [x] 项目布局、lakefile、lean-toolchain
- [x] `Op` / `Stmt` / `Kernel` 类型 + `evalOp`、`stepStmt`、`exec`
- [x] `triton { ... }` 宏 + `tl.log`(`Op.log`)
- [x] `softmax_kernels_refinement`(#1)、`log_sum_exp_refinement`(#2)、
      `softmax_reciprocal_refinement`(#3)—— 全部完整证明
- [x] `welford_eq_two_pass` 数学引理(Phase B 准备)
- [x] `scripts/prove.sh` LLM 证明起草 wrapper
- [x] `bench/llm_eval/` held-out 评测 harness
- [x] T3 forward 可行性研究(`Notes/T3_scouting.md`)

**Phase B —— forLoop 语义 + Tier 2(下一阶段):**
- [ ] `forLoop` 操作语义(互递归 + `termination_by`)
- [ ] `forLoop_inv` invariant 推理引理(带 idx 寄存器绑定)
- [ ] `welford_kernels_refinement`(#4)—— `welford_eq_two_pass` 上升到 kernel 层
- [ ] `online_softmax_recurrence_eq_batch`(#5)—— *FlashAttention 算法核心*
- [ ] `layernorm_kernels_refinement`(#6)—— 单 pass 融合 vs 两 pass
- [ ] 1–2 个选定代表性 kernel 的 differential testing 工件

**Phase C —— `tl.dot` + masking + FA forward(Tier 3-A):**
- [ ] `Value.tile2D` 二维矩阵 tile 构造子 + `Op.dot` 语义
- [ ] 比较算子、`Op.where` 用于 masking、多轴广播
- [ ] `fa_forward_correct`(#7)—— FA-1 forward ≡ standard attention denotation

**Phase D —— multi-block + FA-1 ↔ FA-2 + 论文(Tier 3-B):**
- [ ] Multi-block grid 执行模型 + disjoint-writes 引理
- [ ] `fa_2_forward_correct`(#8)—— FA-2 forward ≡ standard attention denotation
- [ ] `fa1_eq_fa2` headline corollary 通过 spec 传递性
- [ ] 投 PLDI / OOPSLA-spring

**Program 之外(P3+):**
- [ ] Python lifter(真实 `.py` Triton → 嵌入式 `triton { ... }` term)
- [ ] FA-backward、更多 kernel 家族(RoPE、paged attention 等)
- [ ] IEEE-754 保真(精化 `ℝ` 模型)

## 定位

最近的邻居:

* **arm-in-lean** —— 同样的 refinement 形态,目标是 ARM64。我们把这个
  模式移植到 Triton GPU kernel。
* **Vero**(preprint, 2026)—— 用 Lean 校验过的 LLVM transfer function
  代码生成。我们做的 refinement 在 kernel 层(而非 compiler-pass 层),
  并且在算法层(而非实现层)。
* **ATL**(POPL'20)—— Coq 嵌入的张量 DSL,带等价证明;在张量程序等价
  这一点上是最近的亲戚,但它跑在自定义 DSL 上,而非生产级 kernel。

VeriTile 的交集:**算法层 + 面向 LLM + Lean 验证 + 可在生产级 Triton 上
执行。** 上述三个邻居中,没有一个同时覆盖这四个维度。

## License

[MIT](./LICENSE) © 2026 Zenan Li。
