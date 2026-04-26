# VeriTile

[English](README.md) | **中文**

基于 LLM 辅助 Lean 4 证明的 Triton GPU kernel 翻译验证 (translation validation)。

完整 proposal 见 `Notes/proposal.md`(英文)/ `Notes/proposal_zh.md`(中文)。Program plan 见 `PLAN.md` / `PLAN_zh.md`。

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

## Refinement 结构

形态与 `tnum_const_refinement` 一致:`exec` 跑每个 kernel,观测函数
(我们这边是 `readMem`,arm-in-lean 那边是 `readReg`)抽出输出通道,
定理断言逐位置相等。数学步骤被隔离在 `naive_eq_stable` 中;操作语义
walk-through 通过 simp 在我们的 gather/scatter 语义上把
`exec kernel s` 化简为闭式。

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

P1 已经无 sorry。数学内容被隔离在 `naive_eq_stable` 中;两个
`softmax_*_correct` 引理是纯操作语义 walk-through,通过对
`exec / stepStmts / stepStmt / evalOp / Value.bop / writeMem` 做 simp
把 `exec kernel s` 化简到闭式输出,最后委派给 `scatter_readback`
完成 readback 步骤。`scatter_readback` 自身是 `List.foldl` 上的归纳:
将 `List.finRange n = l₁ ++ i :: l₂` 拆分,用 `Nodup` + injectivity 证明
后续写入都不会触碰目标 cell,然后读出位置 `i` 写入的值。

## 文件布局

```
README.md / README_zh.md         本文件(英 / 中)
PLAN.md / PLAN_zh.md             Program plan(英 / 中)
lakefile.toml, lean-toolchain    Lake 构建(Lean 4 v4.15.0;在 v4.29.0 上也可用)
VeriTile.lean                    顶层库入口

VeriTile/Triton/
  Core.lean                      Op / Stmt / Kernel 数据类型(P1 子集)
  Semantics.lean                 BlockState、evalOp(gather)、stepStmt(scatter)、
                                   exec、scatter_readback
  DSL.lean                       `triton { ... }` 宏
  Examples.lean                  手工构造的 naiveSoftmax kernel(直接构造子形式)

VeriTile/Examples/
  SoftmaxEq.lean                 Naive vs stable softmax + refinement 定理

Notes/
  proposal.md / proposal_zh.md   项目 proposal(英 / 中)
  MacroOptions.md                技术调研:宏实现的几种取舍
```

## 构建

需要 Elan + Lake。用户 shell 不会自动 source `.elanrc`,所以请手动加
`PATH` 前缀:

```bash
PATH="$HOME/.elan/bin:$PATH" lake update     # 约 5–15 分钟,下载 Mathlib
PATH="$HOME/.elan/bin:$PATH" lake build
```

预期:构建干净,无警告,无 sorry。

## Triton 子集(P1 范围)

**已纳入:**
* `tl.load`, `tl.store` —— 都支持 **gather/scatter**(tile 值作为 offset)。
* `tl.arange`, `tl.broadcast`, `tl.full`,标量 / 张量常量。
* `tl.exp`, `tl.maximum`,基本算术(`+ − * /`)。
* `tl.max`, `tl.sum` —— block 级 reduction(axis = 0),通过 Mathlib
  `Finset.sup'` / `Finset.sum` 定义。
* `tl.program_id`, `tl.constexpr`。
* `assign`, `store`。`forLoop` AST 已存在,但操作语义返回 `none`。

**未纳入(P2+):**
* `tl.atomic_*`, `tl.dot`,异步 copy。
* 多 block 协调、cluster 级算子。
* Hopper / Blackwell 算子(TMA、WGMMA)。
* `forLoop` 操作语义 —— 需要与 `stepStmts` 互递归,并显式
  `termination_by (sizeOf body + 1, n − i)` 度量。

**信任假设:**浮点算术建模在 `ℝ`(Mathlib `Real`)上。IEEE-754 保真不在
本项目范围内;由对 PyTorch 做 differential testing 来覆盖(P5+)。

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

## 路线图

**P1(骨架 —— 已完成):**
- [x] 项目布局、lakefile、lean-toolchain。
- [x] `Op` / `Stmt` / `Kernel` 类型。
- [x] 操作语义(`evalOp`, `stepStmt`, `exec`),含 gather 和 scatter。
- [x] `triton { ... }` 宏。
- [x] 通过宏写出的双 kernel softmax 例子。
- [x] 数学等价(`naive_eq_stable`):完整证明。
- [x] Refinement 定理(`softmax_kernels_refinement`):完整证明。
- [x] `scatter_readback`:完整证明(`List.foldl` 归纳)。
- [x] `softmax_naive_correct`, `softmax_stable_correct`:完整证明
      (操作语义 walk-through + `scatter_readback`)。

**P2(操作语义打磨):**
- [ ] `forLoop` 操作语义(互递归 + `termination_by`)。
- [ ] 等长 tile-tile 上 `Value.bop` 的自定义 simp 引理。
- [ ] 从宏 body 自动推导 `Kernel.inputs` / `Kernel.outputs`。

**P3+(研究方向 —— 见 `proposal.md`):**
- [ ] Python lifter(真实 `.py` Triton → 嵌入式 `triton { ... }` term)。
- [ ] LLM 驱动的证明起草框架(Vero 风格)。
- [ ] 更多等价示例:Welford 方差、log-sum-exp、scan 重排。
- [ ] FlashAttention 风格的分解(stretch goal)。

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
