# VeriTile:基于 LLM 辅助 Lean 证明的 Triton Kernel 翻译验证

**研究 Proposal**
李泽南(zenan.li@apodex.ai)
2026-04-25

---

## 摘要

我们提出 **VeriTile**:一个为 **Triton GPU kernel 提供机器可检查等价证明**的系统。用户提交普通的 `.py` Triton 代码,**全程不接触 Lean**。系统内部把 Triton 子集深度嵌入到 Lean 4(`triton { ... }` macro),用 LLM agent 产生优化变体和等价证明(或对用户提交的两份 kernel 直接证等价);lifter 和 extractor 处理 `.py` Triton ↔ Lean 嵌入式 Triton 之间的往返。系统瞄准生产 ML kernel 开发中**算法到实现的鸿沟**——工程师手写优化(FlashAttention、online softmax、blocked GEMM)却没有形式化方式验证重写保持语义。

---

## 1. 问题陈述

现代 ML 系统的性能提升越来越依赖**手写 kernel 重写**:online softmax(1 遍 vs 3 遍)、FlashAttention(不显式物化 $N \times N$ attention 矩阵)、Welford 方差、blocked Cholesky、Kahan 求和。这些重写在实数算术下与朴素形式等价,但通常**引入源码中不存在的运算**——例如 FlashAttention 的 running-max 校正项。工程师重写、目测、跑端到端测试、上线。bug 经常漏掉:PyTorch / JAX backward pass 的 off-by-one 和数值病态长期存在。

当前的验证现状:

- **没有主流工具能验证两个 Triton kernel 等价。** 工程师无法机器检查自己的手写优化。
- **编译器内部验证**(TVM Z3、LLVM Alive2)处理的是**单个 pass 内的规则化简**,不是**整个 kernel 重写**。
- **数学抽象层面的算法等价验证**(ATL,POPL'20)需要把 kernel lift 到独立的代数表示,这(a)引入翻译 gap,(b)漏掉抽象之外的实现层重写(layout、swizzle、register tile 顺序)。
- **LLM 能提出 kernel 重写**但不提供正确性保证;最近的 benchmark(Vero,2026)显示 LLM 即使在编译器内部验证任务上也只通过 1/27(LLVM transfer function)。

我们的系统填这个空白:

1. **直接对 Triton 代码本身**验证 kernel 级等价(不走单独抽象)。
2. **对终端用户隐藏 Lean**:`.py` Triton 进,经过验证的 `.py` Triton 出。
3. 用 LLM 产**优化变体**和**等价证明**,Lean kernel 是信任锚。

---

## 2. 方法

三个架构承诺:

**A. Triton 在 Lean 4 中深度嵌入。** Triton 的一个子集(单 block、确定性、覆盖算法层重写所需的 op)被形式化为 Lean 归纳类型 `TritonKernel`,带有作用在 block 状态(memory region、register file、program counter)上的操作语义。Kernel 等价直接在这个语义模型上建立——**验证目标就是用户的 Triton 代码本身**。

**B. 通过 lifter / extractor 实现"零 Lean 用户体验"。** 用户像现在一样在 `.py` 文件里写 Triton。lifter 解析 AST,产出包在 `triton { ... }` Lean macro 里的 `TritonKernel` 项。LLM agent 完全在 Lean 内部工作。Extractor 把 Lean 项还原成 `.py` Triton。**用户从不打开 `.lean` 文件**。

**C. 两种工作模式:**

- **优化模式**:用户提交朴素 Triton kernel;系统(LLM 驱动)产出优化变体**和** Lean 检查通过的等价证明;用户拿到优化版 `.py` kernel。
- **验证模式**:用户提交两个 Triton kernel(他们手写的朴素版和优化版);系统产出 Lean 检查通过的等价证明;用户得到 ✓ 或反例。

### 架构

```
   用户面向的 .py 代码(写 / 读)
   ┌─────────────────────────────────────────────────┐
   │  naive_kernel.py    optimized_kernel.py         │
   │       │                     ▲                   │
   └───────┼─────────────────────┼───────────────────┘
           │ Lifter              │ Extractor
           │ (Python AST → Lean) │ (Lean → Python)
           ▼                     │
   ┌─────────────────────────────┼───────────────────┐
   │  Lean 4 层(可信)           │                   │
   │  ┌────────────────────────┐ │                   │
   │  │ triton { ... } macro   │ │                   │
   │  │ TritonKernel : Type    │ │                   │
   │  │ exec : ... → State     │ │                   │
   │  │                        │ │                   │
   │  │ LLM 产出:              │ │                   │
   │  │   - 优化 kernel         │─┘                   │
   │  │   - 等价证明            │                     │
   │  │                        │                     │
   │  │ Lean kernel:对每个     │                     │
   │  │  产物机器检查           │                     │
   │  └────────────────────────┘                     │
   └──────────────────────────────────────────────────┘
                              │
                              │ 差分测试桥接
                              ▼
                      PyTorch 参考实现
```

**信任边界:**

- *可信(机器检查)*:`TritonKernel` 的 Lean 操作语义、LLM 产出并由 Lean kernel 检查的等价证明。
- *不可信(差分测试)*:lifter、extractor、Triton 编译器、GPU。任何分歧由"两个 kernel 各自跟 PyTorch 参考对比输出"来 catch。

这是一个有意识的 scope 选择:VeriTile 保证**两个 Triton AST 在我们的形式 Triton 语义下等价**。Triton 自己是否如实编译它们是 Triton 的问题——但我们的等价 claim 在任何统一编译下都不变。

---

## 3. 完整示例:Online Softmax

### 3.1 这个变换

朴素 softmax 需要对输入 $x \in \mathbb{R}^n$ 做三遍扫描:

$$
m  = \max_i x_i \qquad
s  = \sum_i \exp(x_i - m) \qquad
y_i = \exp(x_i - m) / s
$$

Online softmax(Milakov & Gimelshein, 2018)用单遍 fold 同时维护 max 和分母:

$$
\begin{aligned}
\text{从 } (m_{\text{old}}, d_{\text{old}}) \text{ 和新元素 } x_i: \quad
& m_{\text{new}} = \max(m_{\text{old}}, x_i) \\
& d_{\text{new}} = d_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}}) + \exp(x_i - m_{\text{new}})
\end{aligned}
$$

最终 $y_i = \exp(x_i - m_{\text{final}}) / d_{\text{final}}$。校正因子 $d_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}})$ 是被发明出来的——朴素形式里不存在。这正是 VeriTile 要直接在 Triton 代码上验证的那类重写。

### 3.2 用户面向的文件

用户写普通 Triton:

```python
# naive_softmax.py
import triton
import triton.language as tl

@triton.jit
def naive_softmax(X, Y, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x = tl.load(X + offs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + offs, y)
```

```python
# online_softmax.py(用户提交,或 VeriTile 生成)
@triton.jit
def online_softmax(X, Y, N: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * N
    m = tl.full((), -float('inf'), dtype=tl.float32)
    d = tl.full((), 0.0, dtype=tl.float32)
    for i in range(N):
        x_i = tl.load(X + base + i)
        m_new = tl.maximum(m, x_i)
        d = d * tl.exp(m - m_new) + tl.exp(x_i - m_new)
        m = m_new
    offs = base + tl.arange(0, N)
    x = tl.load(X + offs)
    y = tl.exp(x - m) / d
    tl.store(Y + offs, y)
```

用户调用:

```bash
$ veritile verify naive_softmax.py online_softmax.py
[lift]    naive_softmax.py    -> Lean term (triton { ... })
[lift]    online_softmax.py   -> Lean term (triton { ... })
[prove]   构造等价证明(LLM agent,~14 分钟)
[check]   Lean kernel: PROOF ACCEPTED
[verify]  ✓ 在 VeriTile-Triton 操作语义下等价
[diff]    在 10000 个随机输入上做差分测试:max ULP diff = 3 ✓
```

### 3.3 内部 Lean 表示

在 Lean 内部(用户看不到),kernel 被 lift 成:

```lean
def naive_softmax_kernel : TritonKernel := triton {
  @T.jit
  def main(X: Ptr float32, Y: Ptr float32, N: tl.constexpr):
      pid = tl.program_id(0)
      offs = pid * N + tl.arange(0, N)
      x = tl.load(X + offs)
      m = tl.max(x, axis=0)
      e = tl.exp(x - m)
      s = tl.sum(e, axis=0)
      y = e / s
      tl.store(Y + offs, y)
}

def online_softmax_kernel : TritonKernel := triton {
  @T.jit
  def main(X: Ptr float32, Y: Ptr float32, N: tl.constexpr):
      pid = tl.program_id(0)
      base = pid * N
      m = tl.full((), Float.neg_inf, dtype=tl.float32)
      d = tl.full((), 0.0, dtype=tl.float32)
      for i in range(N):
          x_i = tl.load(X + base + i)
          m_new = tl.maximum(m, x_i)
          d = d * tl.exp(m - m_new) + tl.exp(x_i - m_new)
          m = m_new
      offs = base + tl.arange(0, N)
      x = tl.load(X + offs)
      y = tl.exp(x - m) / d
      tl.store(Y + offs, y)
}
```

### 3.4 等价定理

```lean
-- 等价陈述:对所有程序输入(memory state X、kernel 参数 N、program_id),
-- 两个 kernel 在算法等价意义下产生相同输出 memory Y(把 fp32 当作 ℝ 处理)。
theorem online_eq_naive
    (X : MemoryRegion ℝ)
    (h_X_finite : ∀ i ∈ X.range, ¬(X.read i).isNaN)
    (h_align : X.aligned 16)
    (N : ℕ) (h_N : N > 0)
    (pid : ℕ) :
    let s_naive  := exec naive_softmax_kernel
                          (initialState X N pid)
    let s_online := exec online_softmax_kernel
                          (initialState X N pid)
    ∀ i, i ∈ outputRange pid N →
         readMem s_naive.Y i = readMem s_online.Y i := by
  intro X h_X_finite h_align N h_N pid
  -- 展开两个 kernel 的执行轨迹
  simp only [exec, naive_softmax_kernel, online_softmax_kernel,
             TritonKernel.step, ...]
  -- 关键引理:操作语义下的 scan 不变量
  apply scan_invariant_corresponds_to_max_and_sum_of_exp
  · exact h_X_finite
  · exact h_align
  · -- 用 Mathlib 的 Real.exp 引理做实际代数步骤
    sorry  -- LLM 目标位置
```

`sorry` 由 LLM agent 填。证明归约到(a)证明两个 kernel 的执行轨迹在各自循环结束后的相关 memory 位置上一致,(b)代数恒等式 $d_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}}) \cdot \exp(x_i - m_{\text{old}}) = d_{\text{old}} \cdot \exp(x_i - m_{\text{new}})$,这正是让 rescaling 成立的核心。Mathlib 的 `Real.exp_add`、`Real.exp_sub`、`Finset.sum_range_succ` 是主力;操作语义部分需要 `tl.load`/`tl.store` 别名引理和 `tl.max`/`tl.sum` 的 block 级 reduce 引理,这些是 VeriTile prelude 的一部分。

---

## 4. Triton 子集与操作语义

P1 阶段我们形式化一个**有意识压小的** Triton 子集:

**P1 包含:**
- `tl.load`、`tl.store`(带显式对齐)
- `tl.arange`、`tl.broadcast`、标量 / tensor 常量
- `tl.exp`、`tl.log`、`tl.sqrt`、`tl.maximum`、基础算术
- `tl.max`、`tl.sum`、`tl.min`(block 级 reduce,axis=0)
- `tl.program_id`、`tl.constexpr`
- `for`、`if`、标量 / tensor 变量

**P1 暂不包含**(后续可加):
- `tl.atomic_*`(P3+ 如有需要)
- `tl.dot`、`tl.tensor` matmul 原语(P5+ 给 blocked GEMM)
- async copy、software pipelining
- 多 block 协调、cluster 级 op
- Hopper / Blackwell 专用 op(TMA、WGMMA)

**操作语义设计:**

```lean
-- Block 级状态
structure BlockState where
  memory   : MemoryRegion → BitVec 32 → ℝ          -- 抽象 fp32 → ℝ
  registers: RegMap                                 -- 命名标量 / tensor 寄存器
  pid      : ℕ                                      -- program_id
  pc       : Nat                                    -- program counter

-- 单步
inductive step : TritonStmt → BlockState → BlockState → Prop where
  | load   : ...
  | store  : ...
  | reduce : ...
  ...

-- 多步执行
def exec (k : TritonKernel) (s₀ : BlockState) : BlockState := ...
```

Memory 用一个把 fp32 存储抽象成 ℝ 值的偏映射建模(带有限定义域假设:输入无 NaN / 无 Inf,中间值的不变量从输入推出)。

**我们接受的根本简化**:把 fp32 算术当 ℝ 处理。这是 ATL、Halide-equivalence、和大多数算法层验证工作的**相同信任假设**。浮点 soundness 留给差分测试。这是个有意识、有充分依据的 scope 决策。

---

## 5. LLM 集成

LLM agent 在四个层面工作:

| 任务 | 输入 | 输出 | 难度 |
|---|---|---|---|
| **Lifter 辅助** | 不规则的外部 `.py` Triton | 嵌入式 `triton { ... }` 项 | 低-中(parser 兜底) |
| **优化模式** | 朴素 `TritonKernel` | 优化 `TritonKernel` + 等价证明 | 高 |
| **验证模式** | 两个 `TritonKernel` 项 | 等价证明 | 中-高 |
| **反例解释** | 失败的证明尝试 | 重写不安全的可读说明 | 低 |

我们采用 Vero 的 harness 模式:coding agent(Claude Code 或同等)在迭代循环中运行,接 Lean 4 MCP server,把证明能力(`apply`、`simp`、`omega`、`induction`、kernel 专用 tactic)作为 tool 暴露。每个 task 时间预算:对齐 Vero 的 1 小时 wall-clock,$15 成本上限。

**关键工程赌注。** 操作语义证明比代数抽象证明难(Vero 1/27 的 baseline 是在更简单模型下)。我们的 mitigation:(a)在 P1 建立 VeriTile 专用战术库,(b)为 prelude 灌入 kernel 形状识别引理(比如"如果一个 kernel 的 body 是 fold + extract 模式,就把等价归约到 fold-step 等价"),(c)给 LLM 喂工作过的证明作为 in-context 范例。

---

## 6. Benchmark 设计 — VeriTile-Bench

我们仿照 Vero 的任务结构构建 VeriTile-Bench:

| 类别 | 任务数 | 难度 |
|---|---|---|
| Reduction 重排(结合律、交换律) | 5 | 低 |
| 在线算法(softmax、layernorm、Welford) | 6 | 中 |
| 数值稳定化(log-sum-exp、max-subtraction) | 4 | 中 |
| Scan / fold 等价(顺序 ↔ 树形、prefix sum) | 4 | 中 |
| 循环融合 / 分裂(验证合并 pass) | 4 | 中 |
| Blocked / tiled 等价(blocked GEMM、blocked reduction) | 4 | 高 |
| 完整 FlashAttention 风格分解 | 1 | 极高(stretch) |

**总计约 28 个任务。** 每个任务是一对 Triton kernel(`naive.py`、`optimized.py`)加一个 read-only 的 Lean spec 文件声明等价定理。验证机制:read-only spec 的 hash 检查 + Lean kernel + axiom 白名单。

**通过率指标:**

1. **优化模式端到端**:agent 收到 `naive.py`,产出 `optimized.py` + 证明;Lean 接受。
2. **验证模式**:agent 收到 `naive.py` 和 `optimized.py` 两份;产出证明;Lean 接受。
3. **仅证明**:人写两个 kernel,LLM 产证明。

报告这三档可以把 LLM 的变换能力和证明能力分开。

---

## 7. 时间线

| 阶段 | 月份 | 里程碑 |
|---|---|---|
| **P1: Triton 子集语义** | 1-3 | Lean 归纳类型 `TritonKernel`、操作语义、约 30 条 reduction 引理(load/store 别名、reduce 恒等式、exp/sum 互动)。手工证 `(0+x)·1 = x` 类平凡等价。 |
| **P2: `triton { ... }` macro** | 3-4 | Lean meta 程序把 Triton 风格语法解析成 `TritonKernel`。在测试语料上 macro-parse → pretty-print → re-parse 是恒等。 |
| **P3: Lifter (.py → Lean)** | 4-5 | Python AST → Lean 源码。处理 80% 的标准 Triton 模式;剩余 20% 报"不支持的特性"明确错误。 |
| **P4: 第一个真实证明** | 5-6 | 在操作语义下手证 online softmax 的 `online_eq_naive`。**Gate:证明超过 ~1500 行或卡住,缩小子集或 pivot。** |
| **P5: Extractor + 端到端 demo** | 6-7 | Lean `TritonKernel` → `.py`。整条 pipeline 跑通:lift → 证 → extract → GPU 执行 + diff-test。 |
| **P6: LLM 证明草拟** | 7-9 | 给定手写 kernel 对,LLM 产等价证明。**Gate:前 10 个 benchmark task 通过率 < 5%,定位改为 "verified workflow + benchmark" 而非自动系统。** |
| **P7: LLM 优化模式** | 9-10 | 给 naive 输入,LLM 产优化 kernel + 证明。至少 1 个端到端完全自动通过。 |
| **P8: 完整 benchmark + 论文** | 10-12 | VeriTile-Bench 跑遍 Claude / GPT-5 / Gemini;arXiv + 投稿(PLDI、OOPSLA、ASPLOS、NeurIPS Datasets&Benchmarks、MLSys)。 |

每个阶段都有显式 gate 标准。失败模式会让贡献叙事换方向但不会杀项目。

---

## 8. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| Triton 子集操作语义太大无法形式化 | 中 | 高 | P1 子集激进收紧(单 block,无 atomic);按需扩张 |
| LLM 在操作语义上写不出证明 | 中-高 | 高 | 建 VeriTile 专用战术库(P1);给丰富 in-context 范例;失败则改框为"带人工辅助的验证器" |
| Lifter 处理不了真实世界 Triton 习语 | 中 | 中 | 覆盖标准模式(80%);不支持的明确报错;迭代扩面 |
| Triton 语言演化破坏子集 | 低-中 | 中 | 锁 Triton 版本;子集小到更新可控 |
| Mathlib 缺 fp ↔ ℝ 建模的关键引理 | 中 | 中 | P1 建小型 VeriTile prelude;反哺 Mathlib |
| 操作语义证明复杂度让 benchmark 跑不动 | 中 | 高 | 从平凡等价开始;复杂度爬坡推动子集 / 战术改进 |
| Triton miscompilation 干扰差分测试 | 低 | 中 | 跟 PyTorch eager 参考交叉验证;遇到 miscompile 上报上游 |
| 翻译验证工作跟 Triton 团队未来形式化重叠 | 低 | 中 | 先发优势;早期接触 Triton 团队作合作而非竞争 |
| Benchmark 被批"过窄" | 中 | 中 | 强调:首个 ML kernel 级翻译验证 benchmark;呈现从 Alive2 / Vero 演化的脉络 |

---

## 9. 预期贡献

1. **VeriTile 语义**:首个对 Triton 子集做操作语义的 Lean 4 形式化,专为 kernel 级等价推理设计。
2. **VeriTile 系统**:lifter + macro + extractor + LLM 证明 harness,作为一个 CLI 工具呈现——`.py` Triton 进,经过验证的 `.py` Triton 出。
3. **VeriTile-Bench**:首个 ML 系统中 kernel 级翻译验证 benchmark——约 28 个任务,naive/optimized Triton 配对。
4. **实证研究**:首个对 LLM 在 kernel 级操作语义证明能力的测量;与代数抽象证明(Vero LLVM)对比。
5. **可验证 kernel 目录**:经过形式化验证的 Triton kernel 开源库(online softmax、Welford 等)直接可作为生产 drop-in。

---

## 10. 与现有工作的定位

| 系统 | 验证什么 | 验证器 | LLM 在环 | 用户面向语言 |
|---|---|---|---|---|
| Alive2 | LLVM peephole 规则 sound | SMT | 否 | LLVM IR(编译器内部) |
| TVM Z3 (PR #1367) | TVM arith 规则 sound | SMT | 否 | TVM TIR(编译器内部) |
| Vero (LLVM) | LLVM transfer function sound + optimal | Lean | 是 | Lean(编译器内部) |
| ATL (POPL'20) | 张量表达式等价 | Coq | 否 | ATL DSL(独立于 C) |
| CompCert | 编译器保 C 语义 | Coq | 否 | C(生产) |
| **VeriTile** | **两个 Triton kernel 等价** | **Lean** | **是** | **Triton(生产)** |

VeriTile 是这个独特组合:**kernel 级 + LLM 驱动 + Lean 验证 + 零 Lean 用户体验 + 操作语义 scope**。最接近的类比是"Alive2 for Triton kernels"——目前不存在。CompCert 提供最接近的形式方法类比,但是单作者、十年规模,且是 C 编译而非 ML kernel 重写。

---

## 11. 第一周的具体起步

```lean
-- File: VeriTile/Core.lean
import Mathlib

namespace VeriTile

-- 起步用的最小 Triton op
inductive Op : Type where
  | const     : ℝ → Op
  | load      : (ptr : String) → (offset : Op) → Op
  | store     : (ptr : String) → (offset : Op) → (value : Op) → Op
  | arange    : (n : ℕ) → Op
  | broadcast : Op → (n : ℕ) → Op
  | add | sub | mul | div : Op → Op → Op
  | exp       : Op → Op
  | reduceSum : Op → Op
  | reduceMax : Op → Op
  -- 6-8 more

-- Block 状态和单步操作(草图)
structure BlockState where
  mem  : String → ℕ → ℝ
  pid  : ℕ
  -- ...

def step : Op → BlockState → BlockState := sorry

-- 平凡的第一个目标
example : ∀ s, step (.add (.const 1) (.const 2)) s = step (.const 3) s := by
  intro s
  rfl
```

第一周目标:文件能编译,上述平凡等价过。总计约 150 行,1-2 天。一旦这步过,后续每一步都是结构上的扩展。

---

## 附录 A:失败模式分析(预测性)

基于 Vero 公布的失败分析,我们预测 VeriTile-Bench 上 LLM agent 的三种主要失败模式,**操作语义目标让每种都比 Vero 的 bitvector 设定更严重**:

1. **Imprecise imitation**(估计约 50%):LLM 产出一个看起来像已知优化形式但有微妙语义差异的 kernel(reduce 顺序不同导致 fp 值不同;边界 case 处理不同)。应对:差分测试作为强过滤;拒绝语料反喂作为范例。
2. **Incomplete proof**(约 30%):LLM 写出正确 kernel 但证明里塞 `sorry` 或调用不合法引理。应对:checker 强制 axiom 白名单;显式 `sorry` 检测。
3. **Bypass attempts**(约 20%):LLM 修改等价定理陈述使其变得平凡可证。应对:对 read-only 定理段做加密 hash(Vero 模式)。

我们预期 VeriTile 的失败分布**比 Vero 更偏向(2)**,因为操作语义证明更长,留 `sorry` 的诱惑更高。

---

## 参考文献

- Dao, T. et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
- Milakov, M. & Gimelshein, N. *Online normalizer calculation for softmax.* arXiv:1805.02867, 2018.
- Bernstein, G. L. et al. *ATL: A Tensor Language for Verifying Tensor Programs.* POPL 2020.
- Lopes, N. P. et al. *Alive2: Bounded Translation Validation for LLVM.* PLDI 2021.
- Tillet, P. et al. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* MAPL 2019.
- TileLang authors. *TileLang: A Composable Tiled Programming Model for AI Systems.* arXiv:2504.17577.
- Vero authors. *From Specification to Kernel Commit: Verified Code Generation on Real-World Systems.*(preprint, 2026).
- Leroy, X. *A Formally Verified Compiler Back-End.* JAR, 2009(CompCert).
- Pnueli, A. et al. *Translation Validation.* TACAS 1998.

---

*Proposal 完。*
