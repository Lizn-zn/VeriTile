# `triton { ... }` 嵌入的 macro 方案

**状态:** 技术调研,P2 规划阶段。
**受众:** VeriTile 实现团队(主要是:未来的我)。
**日期:** 2026-04-25。

---

## 问题

提案(§3.3、§11)展示的 lifted Triton kernel 使用 `triton { ... }` 句法形式,
例如:

```lean
def naive_softmax_kernel : TritonKernel := triton {
  @T.jit
  def main(X: Ptr float32, Y: Ptr float32, N: tl.constexpr):
      pid = tl.program_id(0)
      ...
}
```

这只是愿景。今天提交的骨架(`Examples.naiveSoftmax`)用的是
**直接构造子调用**(`.assign "pid" .programId` 等)——没有任何句法糖。
本文整理从后者走向前者的现实可选方案,并给出推荐。

## 我们到底在优化什么

要清醒一点:**用户并不直接写嵌入式 Triton。**
用户写 `.py` Triton;lifter 产出 Lean term。所以 `triton { ... }`
句法的受众是:

1. **Lifter**(Python 输出 Lean 源码)。关心:可预测、易于发出的句法。
   *不*关心人类阅读时是否简洁。
2. **测试 / 示例文件**(VeriTile 贡献者手写)。关心:debug 时的可读性,
   以及能否以最小修改从真实 Triton 代码 copy-paste 进来。
3. **提案和论文的审稿人 / 读者**。关心:视觉上是否像真实 Triton,
   以便 formalization 看起来"对劲"。

(2) 和 (3) 真实存在但是次要;(1) 占主导。所以 verbose-but-mechanical
的形式可以接受;beautiful-but-fragile 的 parser 不可接受。

## 方案 A —— 不要 macro,直接用构造子

直接像 Lean 原生那样使用 `.const`、`.add`、`.assign`、`.store`。
这就是 `Examples.naiveSoftmax` 当前的做法。

```lean
def naiveSoftmax (N : Nat) : Kernel where
  inputs  := ["X"]
  outputs := ["Y"]
  body    := [
    .assign "pid" .programId,
    .assign "offs"
      (.add (.broadcast (.mul .programId (.const (N : ℝ))) N) (.arange N)),
    .assign "m"   (.reduceMax (.ref "x")),
    ...
  ]
```

**优点:**
* Zero meta-programming 工作量,自然就成。
* Lifter 平凡地发出这种代码——就是构造子调用。
* 错误由 Lean 的常规类型检查器在 `Op` / `Stmt` 上报告。
* 重构 `Op` / `Stmt` 自动传播。

**缺点:**
* 视觉噪声大。Triton 的 `pid * N + tl.arange(0, N)` 变成深嵌套的
  `.add (.broadcast (.mul ...) N) (.arange N)`。
* 没有省略括号的优先级;一切都全括号化。
* 没法直接把真实 Triton 复制粘贴到 Lean 测试文件。

**工作量:** 0 天。已经做完了。

## 方案 B —— 给单个 operator 加 Lean notation

加 Lean 的 `notation` / `infixl` 声明,让算术构造子看起来像中缀算符。
再加上一些 helper function 处理较啰嗦的构造子。

```lean
namespace VeriTile.Notation

scoped infixl:65 " ⊕ " => Op.add
scoped infixl:65 " ⊖ " => Op.sub
scoped infixl:70 " ⊗ " => Op.mul
scoped infixl:70 " ⊘ " => Op.div

scoped notation:max "$" name => Op.ref name
scoped notation:max "ℝ" => fun (c : ℝ) => Op.const c

scoped notation:max "load[" r ", " o "]" => Op.load r o
scoped notation:max "max⟨" e "⟩"  => Op.reduceMax e
scoped notation:max "sum⟨" e "⟩"  => Op.reduceSum e
scoped notation:max "broadcast(" e ", " n ")" => Op.broadcast e n
scoped notation:max "bcast " n " of " e => Op.broadcast e n

end VeriTile.Notation
```

示例随之变成:

```lean
open VeriTile VeriTile.Notation in
def naiveSoftmax' (N : Nat) : Kernel where
  inputs  := ["X"]
  outputs := ["Y"]
  body    := [
    .assign "pid" .programId,
    .assign "offs" (bcast N of (Op.programId ⊗ Op.const N) ⊕ Op.arange N),
    -- ...
  ]
```

**优点:**
* 可读性有适度提升。
* 零新增 meta 机制——`notation` 自 Lean 4 v4.0 起就内建。
* Lifter 平凡地发出(字符串略短)。
* Scoped,不污染全局命名空间。

**缺点:**
* 没解决更大的问题:控制流(`forLoop`、`assign`、`store`)看起来仍然像
  Lean 不像 Triton。
* 自定义的 Unicode operator(⊕ ⊗)虽然好看,但拿模糊换简短。Triton 风格
  的 `+`/`*` 会和 Lean 自己的冲突——除非 `Op` 单独 namespace,
  且只在没歧义的时候"看起来像"算术。
* 还是没法粘贴真实 Triton。

**工作量:** ~1 天设计 + 写 notation 声明。基本安全,因为 `notation`
非侵入式。

## 方案 C —— 自定义句法块(`triton { ... }`)

用 Lean 4 的 macro / elaborator 基础设施,把 Python-like 块解析成
`Stmt list` 或 `Kernel`。这是 ARM-in-Lean 例子走的路。

形态:

```lean
syntax (name := tritonStmt) "triton " "{" tritonBody "}" : term
syntax tritonBody := (tritonStmt)*
syntax tritonStmt :=
  | tritonAssign
  | tritonStore
  | tritonFor

syntax tritonAssign := ident " = " tritonExpr
syntax tritonStore  := "tl.store" "(" ident " + " tritonExpr ", " tritonExpr ")"
syntax tritonFor    := "for " ident " in range(" num "):" tritonBody

-- ... 还有 tritonExpr,有自己的语法覆盖 Op 构造子。

@[term_elab tritonStmt] def elabTritonStmt : Elab.Term.TermElab := ...
```

Elaborator 走解析后的语法树,发出 `Op` / `Stmt` / `Kernel` 的构造子应用。

**优点:**
* 真的看起来像 Triton。可以从用户 `.py` 文件直接 copy-paste 大块进
  `.lean` 测试 fixture。
* Lifter 可以在花括号内 verbatim 发出 Triton——不需要构造子映射。
* 论文图最有冲击力,审稿人视觉最佳。

**缺点:**
* **巨大的前期成本。** Lean 4 macro/elaborator 工作是一项独立技能;
  即便相对小的 ARM-in-Lean DSL 也是几百行 meta 代码。Triton 的 surface,
  即便我们的 P1 子集,也有 ~15-20 种形态。现实估计:
  **2-3 周做出一个能用的子集,~1 个月做到合理覆盖**。
* 错误信息变差。Macro 展开错误以晦涩著称;用户调试 `triton { ... }`
  里的 parse 失败会看到合成行号,而不是源码行号。
* 维护成本:每次扩展 `Op` / `Stmt` 类型(P3-P7 全程都会发生),
  macro 都需要相应的 parser case。构造子调用会自动更新;
  macro parser 不会。
* **Triton 句法有 Python-isms,我们的 macro 复制不了。** 缩进块、
  augmented assignment(`d *= ...`)、`@triton.jit` decorator 内的函数
  定义、默认参数值。Macro 必须定义它自己更简单的 grammar,
  *resemble* Triton 但不是 Triton——会在不易察觉处令人困惑。
* Lifter 仍然要把真实 `.py` Triton 转换成我们 macro 能 parse 的形式,
  所以 macro 并没有降低 lifter 的复杂度。

**工作量:** 2-4 周。有真实的 slipping 风险。

## 方案 D —— 编译期解析的外部格式(S-expression / JSON)

Lifter 输出独立的 `.triton.sexp`(或 `.json`)文件。Lean term elaborator
在编译期读取它,产生 `Kernel` 值。

```lean
def naiveSoftmaxKernel : Kernel := loadKernelFromFile "naive_softmax.triton.sexp"
```

**优点:**
* 把 lifter 的输出格式和 Lean 句法完全解耦。外部格式可以是任何我们想要的
  东西(S-expr 是 Lisp 风格 AST 的自然选择)。
* Lean 侧很小:就是文件读取器 + AST → 构造子映射。
* `.triton.sexp` 文件可以单独 inspect、diff、regenerate,不用动 Lean 文件。

**缺点:**
* 把 kernel 藏到独立文件里,损害 proposal / paper / docs 的可读性
  (没法 inline 引用 kernel)。
* Lean 4 通过 `compileTimeIO` 等方式支持编译期 file I/O,但比较 fiddly;
  可能和 Lake 的增量构建配合不佳。
* 测试 fixture 变成两文件:`.lean` 测试加 `.sexp` 数据。手写测试很烦。

**工作量:** 3-5 天。Lean 侧 reader 简单,Lake 集成是耗时大头。

## 对比表

| 标准 | A: raw | B: notation | C: macro block | D: external file |
|---|---|---|---|---|
| Lifter 复杂度(Python 侧) | trivial | trivial | medium(必须发出有效 macro)| trivial(发出 S-expr)|
| Lean meta 复杂度 | none | tiny(~1 天)| high(2-4 周)| low-medium(3-5 天)|
| 测试 fixture 可读性 | 差 | 一般 | 优 | 差(分文件)|
| 论文可读性 | 差 | 中等 | 优 | 差 |
| `Op` schema 变更后的维护 | 自动 | 自动 | 手动 | 自动(S-expr 是结构化的)|
| 错误信息 | Lean-native(好)| Lean-native(好)| macro-elaboration(差)| parse-time(中)|
| "在 surface 上 bikeshed" 风险 | 无 | 低 | 高 | 低 |
| 出第一个能用版本的时间 | 0 | 1 天 | 2-4 周 | 3-5 天 |

## 推荐

**阶段顺序:** A → B → C 或 D 作为 P5+ 的可选打磨。

* **P1-P4:留在方案 A。** 今天提交的骨架已经这样做了。构造子调用很丑
  但是 *能跑*;形态 100% 受我们控制;重构自动。把 P2 花在 macro 工作上
  会延误真正的研究(operational semantics + 第一个 proof)。当整个目标是
  "Lean 能不能 kernel-check 这个证明"时,漂亮的句法无关紧要。

* **P5(第一个端到端 demo 之后):加方案 B。** 一旦核心语义稳定、我们也
  知道哪些形态反复出现(例如我们一直写 `pid * N + arange N` 作为 offset),
  就为这些 pattern 引入 scoped notation。便宜、scoped、可移除。

* **P6+(如果审稿人抱怨,或者我们要做 public surface):考虑方案 C 或 D。**
  默认选 **D(外部文件)**——风险更低、meta 投入更小、Op schema 变化
  时更易演化。只有在测试文件里有明确 in-line Triton 需求时才做 C。

**反向推荐:** 不要在 P1-P4 尝试方案 C。Macro 的 hype-cycle 是真实且
诱人的——"让它看起来像 Triton!"——但这是支线任务。研究瓶颈是
"LLM 能不能产生 Lean 接受的证明",不是"论文里源码看起来够不够漂亮"。

## 这对 lifter(Python)意味着什么

Lifter 的输出格式由这个推荐决定:

* **P1-P5:** 用 Python 字符串发出 Lean 源码,直接构造子调用。大致:

  ```python
  def emit_op(op_ast) -> str:
      if isinstance(op_ast, Const):
          return f".const ({op_ast.value} : ℝ)"
      elif isinstance(op_ast, BinOp):
          return f"(.{op_ast.op} {emit_op(op_ast.lhs)} {emit_op(op_ast.rhs)})"
      ...
  ```

  输出进入 `VeriTile/Generated/<KernelName>.lean`。

* **P6+(如果加方案 D):** Lifter 改为发出 `.triton.sexp`。结构相同,
  只是 surface 不同。

## 现有参考 / prior art

* **arm-in-lean**(用户早期的例子):ARM64 汇编的完整方案 C。
  ~1000+ 行 meta。值得读一下 pattern;它的 scope 更小(没有 Python
  缩进规则,没有 `@decorator`)。
* **Verso**:Lean 4 文档框架,自定义句法。Heavy meta。对我们的需求
  overkill,但有不错的 `syntax` / `elab` pattern 例子。
* **Mathlib 的 `calc`**:内建的等式推理句法。中等复杂度 macro 的优秀范例。
* **Lean 4 Manual,"Macros and Elaboration"**:官方参考,
  https://lean-lang.org/lean4/doc/macros.html 。
* **`leanprover/lean4-samples`** 仓库:小型 DSL 例子。

## 开放问题

* 我们的 P1 子集里有没有 Python-ism 在方案 C 下处理不干净?
  (头号嫌犯:`tl.constexpr` 标注、`for i in range(N)` 与
  `for i in tl.serial(N)` 的区分。)
* 如果最终有两种 surface 形式(raw + notation),我们想同时支持还是迁移?
* 对 lifter 发出的 Lean 文件:每次 build 重新生成,还是 check-in 的产物?
  (推荐:重新生成,但 check in 一个已知好的 fixture 作为示范。)

---

**底线:** 从简单做起,只在有具体研究 / UX 理由时升级。方案 A 已 commit;
方案 B 在 P5;P2 什么也不做。
