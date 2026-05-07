# RP1: Pointer 与 Named Region

**状态:** 已解决。Phase A–D 默认:**named region**。
**日期:** 2026-04-27。
**Owner:** Phase A。
**重审条件:** 见 §何时重审。

---

## TL;DR

嵌入式 Triton DSL 用 **`Op.load (region : String) (offset : Op)`** 建模
内存访问 —— 在 kernel 定义时选定的静态 region 名 + 动态 offset 表达式。
我们 **不** 建模一等公民 pointer(CUDA 风格的 `x_ptr + offsets` 作为 value)。

这是有意为之的 verification-side 优化:让 region 不相交自动成立、
消除 aliasing 分析、不需要 memory-safety 不变式、保持 `evalOp` 是干净的
结构递归。代价是 DSL paste-in 的小句法 gap(`tl.load(X, offs)` 对
`tl.load(x_ptr + offs)`),论文上可接受,且必要时可由 ~20 行装饰性
macro sugar 关闭。

8 个主要 correctness theorem 没有一个需要一等公民 pointer。

---

## 1. 问题陈述

嵌入式 Triton DSL 应该如何建模内存访问:

- **(γ) 一等公民 pointer**,如 Triton/CUDA 那样 —— `x_ptr + offsets`
  是一个 pointer(或 pointer-tile)类型的运行时值,可以赋给 register、
  条件选择(`if cond then x_ptr else y_ptr`)、存到内存里、用于间接索引等;
  *或*
- **Named region + offset** —— region 是 kernel 定义时选定的静态 `String`,
  offset 是动态 `Op`-valued 表达式;用户写 `tl.load(X, offs)` 而不是
  `tl.load(X_ptr + offs)`。

这一决定贯穿 `Op.load` / `Stmt.store`、`BlockState.mem` 与项目里的
每一个 kernel proof。

## 2. 拆解 "支持 pointer"

口语说的"支持 pointer"混淆了三个独立关切。只有 **(γ)** 是真正的
semantic-layer 决策;**(α)** 已经支持,**(β)** 是另一个装饰性问题。

### (α) Pointer 算术作为 offset 算术 —— 已支持

Triton 层表达式如 `x_ptr + offs[:, None] * stride + offs[None, :]`
编码一个 offset 计算。在我们的模型里,同样的计算写成 `tl.load(X, expr)`
的 *offset 参数*:

```python
# Triton
x = tl.load(x_ptr + offs[:, None] * stride + offs[None, :])
```
```lean
-- VeriTile DSL(语义等价)
x := tl.load(X, offs[:, None] * stride + offs[None, :])
```

数学是相同的;只是 "X" 写一次,而不是携带一个 pointer 类型的值。
表达力没损失。

### (β) 装饰性句法 `tl.load(X_ptr + offs)` —— 另一个关切

DSL macro 的纯句法扩展,使字面文本 `tl.load(X_ptr + offs)` 可以解析并
lower 到 `Op.load "X" offs`。这是 ~20 行 macro 工作,**不改变语义**,
存在仅为了 paper paste-in 友好。

状态:**推迟到 P3+**,与 `CONTRIBUTING.md` 教程或论文附录准备一起做。
不紧迫。

### (γ) 一等公民 pointer 作为 value —— 真正的问题

把 pointer 作为 `Value` 构造子加入(与 `Value.scalar` 和 `Value.tile`
并列):

```lean
-- 假想扩展(NOT proposed)
inductive Value where
  | scalar  : ℝ → Value
  | tile    : (n : Nat) → (Fin n → ℝ) → Value
  | pointer : RegionName → Nat → Value          -- base region + offset
  | ptrTile : (n : Nat) → (Fin n → RegionName × Nat) → Value
```

这样用户才能写 `ptr := if cond then X else Y`(`ptr` 是 register 类型的
pointer 值),或者 `tl.load(ptr_array[i])` 做间接访问。

## 3. 为何 named region 在 verification 侧胜出

五个具体理由,每一个都有框架现成的支撑。

### 3.1 Region 不相交 `rfl`-trivial

`BlockState.mem : RegionName → Nat → ℝ` 是两层函数:第一层以 `String`
为键。按构造:

```lean
(s.writeMem "X" k v).mem "Y" o = s.mem "Y" o    -- by rfl
```

(`writeMem` 的定义是 `if r = region ∧ o = offset then v else
s.mem r o`;对 `r = "Y" ≠ "X"` 条件为 `false`,走 else 分支。)
所以对 "X" 的写不影响对 "Y" 的读,可证,零 proof effort。

(γ) 下:`ptr_x : Value.pointer "X" 0` 与 `ptr_y : Value.pointer "Y" 0`
*可能* 别名,如果它们底层 region / offset 重叠 —— 然后 operational
semantics 必须表达 "alias" 是什么、不相交情况长什么样、每次写如何 discharge。
这是 separation logic / Iris / VST 的领地。**整片研究方向**。

### 3.2 不需要 aliasing 分析

`BlockState.scatter_readback`(P1 的主力,~50 行)只要求
`Function.Injective offsetFn` *在单个 region 内*。跨 region 别名按构造
不可能,所以我们永远不必考虑 "如果这次对 Y 的 scatter 与之前对 X 的
scatter 撞了怎么办"。

(γ) 下:`scatter_readback` 引理的假设变成 "所有写命中两两不相交的内存
单元",这是 `(region, offset)` injective 而不是 offset-only injective。
仍可处理,但 case-split 工作量翻倍。

### 3.3 不在 scope 内的 memory safety

`mem` 是一个全函数 `RegionName → Nat → ℝ`。在 `"X"` 的 offset `999999`
处写,会成功并产生那个 cell 现持有新值的 `BlockState` —— 没有 segfault、
没有 OOB exception、没有 valid-pointer 不变式要维护。kernel 的
correctness theorem 只关心 spec 实际检查的 cell(`observeY` 在
`pid * N + i.val` 处读取);别处的 junk 写无害。

这是 **deliberate** 的,且按 `PLAN.md` §Out of scope 明确不在 scope
(与 IEEE-754 fidelity 并列)。Memory safety 是另一篇论文。

(γ) 下:pointer 携带 validity / liveness / range 不变式。每次 load/store
都有 memory-safety 义务(或 OOB 上的 defined behavior,后者本身成为 spec
的一部分)。kernel 的 correctness theorem 必须把它穿起来。

### 3.4 `evalOp` 保持干净结构递归

```lean
| .load region off, s => match evalOp off s with
                         | some (.scalar c) => some (.scalar (s.readMem region (realToNat c)))
                         | some (.tile n f) => some (.tile n (fun i => s.readMem region (realToNat (f i))))
                         | _ => none
```

`region` 是 macro-time `String`,烧进 AST。`evalOp` 不需要先算 "这个
pointer 指向哪个 region",再解包 `(region, offset)`。递归只在 offset
上一遍。

(γ) 下:`evalOp .load ptrExpr` 会先把 `ptrExpr` evaluate 到
`Value.pointer`、解包成 `(region, base_offset)`、再 load。多一层间接、
proof 里多 sub-goal、更多 `Option`-juggling。

### 3.5 现有示例无 proof tax

Tier 1 proof(`softmax_naive_correct`、`softmax_stable_correct`、
`log_sum_exp_*`、`softmax_reciprocal_*`)在 named-region 模型下每个
~30–50 行。主要工作是符号执行 + 调用 `scatter_readback`。

(γ) 下,带强制不相交推理估计:**每条 theorem ~100–150 行** —— 3× 税。
横跨 Tier 1+2+3 proof,全项目额外 1000+ 行。

## 4. 8 主 theorem 何时真正需要 (γ)

| 场景 | 需要 (γ)? | 在我们的 8 theorem 里? |
|---|---|---|
| `tl.load(X, complex_offset)` | 不(α 覆盖)| 全部 —— softmax、LSE、FA |
| 2D / 多维 load `X[i*sM + j*sN]` | 不(α 覆盖)| FA forward |
| 单 kernel 多 buffer(X、Y、Out)| 不(named region 覆盖)| add\_kernel、FA、Welford |
| **条件 buffer**:`ptr := if cond then X else Y; load(ptr)` | 是 | 无 |
| **间接索引**:`ptr_tile := load(P, idx); val := load(ptr_tile)` | 是 | 无 |
| **`tl.make_block_ptr` + `tl.advance`**(Triton 2.x block pointer API)| (γ) 或 (α) 等价重写 | FA-2 reference 用这种风格;(α) 重写算法等价 |
| **Pointer 作为 kernel 参数**(可变 buffer 数)| 是 | 无 |
| **Pointer 存在内存里** | 是(+ heap-typed memory)| 无 |

**结论:8 主 theorem 没有一个需要 (γ)。** FA-2 的 reference 实现
用 block pointer,但 offset-arithmetic 重写可证等价(重写正是
"显式化 block pointer 隐式追踪的内容"),也是我们在 Phase D 要
formalize 的内容。

## 5. (β) 装饰性句法决定 —— **已落地**

独立于 (γ) 问题。Triton 与 VeriTile 的句法 gap 现在在 surface 层关闭:

```python
# Triton
x = tl.load(x_ptr + offsets)
tl.store(out_ptr + offsets, value)
```
```lean
-- VeriTile(当前)
x := tl.load($(xReg) + offs)
tl.store($(outReg) + offs, value)
```

Macro 把 `tl.load($(R) + offs)` lower 成 `Op.load R offs`,
`tl.store($(R) + offs, v)` lower 成 `Stmt.store R offs v`。Scalar-pointer
sugar `tl.load($(R))` / `tl.store($(R), v)` desugar 到 offset `0`。

**重要:** `+` 是 **纯 surface-syntax cue**。`$(xReg)` 仍是
`RegionName`(`String`),`offs` 仍是 `Nat`-valued offset 表达式,
没有引入一等公民 pointer 值 —— macro 解构 surface 形式并发出现有的
region+offset AST。从 operational semantics 视角看,这与之前逗号分隔的
`tl.load($(xReg), offs)` 等同;RP1 对 (γ) 的拒绝仍然成立。

2026-04-27 解决。`expandExpr` / `expandStmt` / `exprRegions` /
`stmtRegions` 扩展约 30 行;无语义改动;8 个 correctness/refinement
theorem 全部 closed,proof body 没改。

**Refactor 2026-04-27(后)**:把 `tritonPtr` 提取为独立的
`declare_syntax_cat`,有两个 production(`$(R)` 与 `$(R) + offs`)。
`tl.load` / `tl.store` 规则各自从两条硬编码形态合并为单条
`tl.load(tritonPtr)` / `tl.store(tritonPtr, value)`。内部 AST 不变 ——
`expandPtr` helper 仍把 `tritonPtr` lower 成 `(region, offset)` 对
喂给 `Op.load` / `Stmt.store`。这是作为 GH issue \#1 跟踪的
surface-syntax 清理;动机是未来 pointer 形态(masked load 带
`mask=`/`other=`、Phase C 的 2D pointer)扩展 `tritonPtr` 而不是
向 `tritonExpr` / `tritonStmt` 加新 production。RP1 对 (γ) 的拒绝
仍然成立。

## 6. 类似工作

针对 GPU / 数组 kernel 的相似 verification 项目一致选择 named-buffer
抽象,而不是 pointer-arithmetic 语义:

- **Vero**(PLDI 2024)—— Triton lowering verification;在 proof 层
  用 named buffer 抽象。
- **ATL**(POPL 2022)—— 数组语言带结构化推理;proof 义务里没有 pointer。
- **KaTen** / TVM operator verification(CGO 2023)—— operator 层
  buffer 抽象。

PL-verification 社区的审稿人期望是:pointer / aliasing / memory-safety
推理是 **separate concern**,由 separation-logic 项目(Iris、VST)处理,
不应与算法等价 verification 捆绑。

我们的选择与已有 pattern 一致。审稿人 pushback 不太可能。

## 7. 何时重审

如果以下任何一项成真,重新评估 (γ) 决定:

- **Pivot 到 Triton 编译器优化 formalization。** 如果项目 scope 从
  "kernel 重写的算法等价" 变到 "验证 Triton 从 Triton IR 到 PTX 的
  lowering pass",pointer 在源语言里成为一等公民,(γ) 被强制。

- **条件 buffer 选择成为目标 kernel pattern。** 如果未来某个 Tier 扩展
  到带 `ptr = if cond then X else Y` 的 kernel,(γ) 被强制。当前 scope
  里没有这种 kernel。

- **Block-pointer API(`tl.make_block_ptr`/`tl.advance`)成为独立
  verified rewrite target。** 如果未来论文要验证 block-pointer API
  本身是一个保 correctness 的、相对 offset-arithmetic 的 rewrite
  (即 block-pointer 抽象 *就是* 算法内容),(γ) 成为这项工作的自然选择。

- **间接 / gather-of-gather 访问 pattern**(pointer array)。
  当前 P3+;若推进,会强制 (γ)。

如果以上都不发生,named region 仍是选定的抽象。

## 8. 具体对应(论文附录用)

最常见 Triton pattern 的并列映射:

| Triton | VeriTile DSL | 产生的 AST |
|---|---|---|
| `tl.program_id(0)` | `tl.program_id(0)` | `Op.programId` |
| `tl.arange(0, BLOCK)` | `tl.arange($(BLOCK))` 或 `tl.arange(0, $(BLOCK))` | `Op.arange BLOCK` |
| `pid * BLOCK + tl.arange(0, BLOCK)` | `pid * $(BLOCK) + tl.arange(0, $(BLOCK))` | `Op.add (Op.mul ...) (Op.arange BLOCK)` |
| `tl.load(x_ptr + offsets)` | `tl.load($(xReg) + offs)` | `Op.load xReg (Op.ref "offs")` |
| `tl.load(x_ptr)`(单标量)| `tl.load($(xReg))`(sugar)| `Op.load xReg (Op.constNat 0)` |
| `tl.store(out_ptr + offsets, value)` | `tl.store($(outReg) + offs, value)` | `Stmt.store outReg (Op.ref "offs") ...` |
| `tl.store(out_ptr, value)`(单标量)| `tl.store($(outReg), value)`(sugar)| `Stmt.store outReg (Op.constNat 0) ...` |
| `x_ptr + i*M + j`(2D 地址)| `tl.load($(xReg) + $(i)*$(M) + $(j))` | `Op.load xReg (Op.add (Op.mul ...) ...)` |

未覆盖的 pattern(需要 (γ)):

| Triton | VeriTile | 状态 |
|---|---|---|
| `ptr := if cond then x_ptr else y_ptr; tl.load(ptr + offs)` | 不可表达 | 仅 (γ);8 theorem 不需要 |
| `ptr_arr[i]` 存 pointer | 不可表达 | 仅 (γ);不需要 |
| `block_ptr := tl.make_block_ptr(...); x := tl.load(block_ptr); block_ptr := tl.advance(block_ptr, ...)` | 重写为显式 offset 算术 | FA-2 选 (α) 重写 |

## 9. 决策日志

- **2026-04-27** —— pointer 语义讨论在 add\_kernel feasibility review
  期间浮出。确认:Phase A–D 沿用 named-region 模型。(β) 装饰性句法
  原本推迟到 P3+。记录为 RP1。
- **2026-04-27(后)** —— (β) 装饰性句法早于计划落地:
  `tl.load($(R) + offs)` / `tl.store($(R) + offs, v)` 与 scalar-pointer
  sugar `tl.load($(R))` / `tl.store($(R), v)`。纯 macro-time desugar;
  无语义改动;(γ) 仍被拒绝。
- **2026-04-27(后)** —— surface-syntax 清理:把 `tritonPtr` 提取为
  独立 syntax category,把 4 条硬编码 `tl.load` / `tl.store` 规则合并
  为 2 条。解决 GH issue #1;AST 不变;为 Phase C masked / 2D pointer
  形态留出 extension point。

---

*相关 framing 决定见 `PLAN.md` §Out of scope 与 §Risk register
(IEEE-754 fidelity、atomic、multi-stream、Python lifter)。*
