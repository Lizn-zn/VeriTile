# `forLoop_inv` 接口设计

**状态:** 规范,2026-04-29 已批准。
**Phase:** B(Tier 2 streaming reduction)。
**涉及文件:** `VeriTile/Semantics.lean`、新增 `VeriTile/LoopInvariant.lean`、`VeriTile.lean`。

本规范确定 VeriTile 嵌入 Triton 子集中 `Stmt.forLoop` 的操作语义,以及
所有 Phase B(welford / online softmax / layernorm)和 Phase C
(FA-1 forward,单 program-id)将依赖的 loop-induction 引理族形态。
它取代 `PLAN.md:83–94` 里草拟的签名。

## 1. 目标

提供一个可复用的 loop-induction 引理 `forLoop_inv`,使每个 Tier 2 /
Tier 3-A kernel-pair proof 都可以这样关掉它的 loop:

1. 选一个 invariant `P : Nat → BlockState → Prop`,描述 kernel 累加器在
   `k` 次迭代后的运行时状态。
2. 证 `P 0 s_init`(entry)。
3. 证 step 义务:每次 body 迭代保持 `P` 并返回 `some`(即不抛运行时错误)。
4. 得到 `P n s_final`,其中 `s_final` 是 operational semantics 产生的。

在我们的嵌入里,Phase B 和 Phase C 跑的 kernel 都使用 **单个** `forLoop`
配 *多语句 inner body* —— Welford / online softmax / layernorm 在输入 tile 上
有一个 `forLoop`;FA-1 forward 在 KV block 上有一个 `forLoop`(Q-block
维度通过 `pid` 在 grid 层处理,而非通过内层 `forLoop`)。FA-2 也是
每 program-id 单 `forLoop`。

因此引理必须能在 (a) 多语句 body 和 (b) loop 之后顺序跟随的语句下复合。
*防御性地*,设计也要兼容未来 `forLoop` 嵌套 `forLoop` 的 kernel
(某些 FA variant 和我们论文之后加的任何 block-tiled kernel 需要这点),
所以 `mutual` 块的改动是放在 `stepStmt` / `stepStmts` 上,
而不是在结构递归之外定义 `stepForLoopAux`。

## 2. Carrier 选择(已决定)

`P : Nat → BlockState → Prop`。

Invariant 谈论的是 *整个* `BlockState`,而不是抽象的逻辑状态 `S`。
下游用户从 `BlockState` 取寄存器值通过 `s.regs name` 加上对 `Value`
构造子(`Value.scalar`、`Value.tile`、`Value.scalarNat`、`Value.tileNat`)
的模式匹配;现有的 `Value.asScalar` / `Value.asScalarNat` helper
覆盖 scalar 情况。Tile 读取通过直接模式匹配——今天没有 `Value.asTile`,
本规范也不加(预先引入它会让 `simp` 配置复杂化;只在某个具体 kernel
proof 提出需求时再加)。出现的 boilerplate 由 simp lemma 吸收,
而不是引入并行的抽象层。

理由:与现有 P1 风格一致(`BlockState.scatter_readback` 也是
`BlockState`-level + simp-driven),避免维护抽象 / 具象 round-trip 引理,
并且直接兼容嵌套循环情况(内层 `forLoop_inv` 调用继续谈论同一个
`BlockState`)。

## 3. Operational semantics

### 3.1 `mutual` 块

`stepStmt` 和 `stepStmts` 移入一个 `mutual` 块,加上新的
`stepForLoopAux`:

```lean
mutual
  noncomputable def stepStmt : Stmt → BlockState → Option BlockState
    | .assign name e, s        => -- unchanged
    | .store region off val, s => -- unchanged
    | .forLoop idx n body, s   => stepForLoopAux idx 0 n body s

  noncomputable def stepStmts : List Stmt → BlockState → Option BlockState
    | [], s         => some s
    | st :: rest, s => (stepStmt st s).bind (stepStmts rest)

  noncomputable def stepForLoopAux
      (idx : RegName) (i n : Nat) (body : List Stmt) :
      BlockState → Option BlockState
    | s =>
        if i < n then
          (stepStmts body (s.setReg idx (Value.scalarNat i))).bind
            (stepForLoopAux idx (i+1) n body)
        else some s
end
termination_by
  stepStmt        st _           => (sizeOf st, 0, 0)
  stepStmts       l  _           => (sizeOf l, 0, 0)
  stepForLoopAux  _ i n body _   => (sizeOf body + 1, n - i, 0)
decreasing_by
  -- per-case ordering: stepStmt → stepStmts (body of forLoop body) decreases
  -- on the first component when stmt is forLoop, decreases via list-shrink on
  -- non-forLoop stepStmt → stepStmt; stepForLoopAux self-recursion decreases
  -- on the lex pair (sizeOf body + 1) ↘ unchanged, (n - i) ↘ strict.
  -- Concrete tactic: `decreasing_tactic` for each goal; if it fails on the
  -- aux-self-recursion case, fall back to `simp_wf; omega`.
```

(上面的 `decreasing_by` 块是草图;实际 tactic 调用在实现时敲定。
lex pair 是 `PLAN.md` 已经规定的。)

### 3.2 Sub-decision

| 决策 | 取值 | 备注 |
|---|---|---|
| `idx` value channel | `Value.scalarNat i` | 不是 `Value.scalar (i : ℝ)`(`PLAN.md:88` 草图)。Body 的 offset 算术(`pid * BLOCK + idx * STRIDE`)在 `Nat` channel,所以 `scalarNat` 是唯一一致的选择。 |
| `idx` 循环后可见性 | 不恢复成 loop 前的值 | 寄存器 `idx` 的最后一次写存活。如果 `body` 自身 *没* 写 `idx`,存活值是 last-iteration 绑定 `n − 1`(或 `none`,若 `n = 0` 且进入 loop 前 `idx` 未设置)。如果 `body` 写 `idx`,存活值是 `body` 最后写的值。无 scope/shadow 机制;用户负责跨嵌套 loop 不要别名 register name。 |
| `n = 0` 行为 | identity(`stepForLoopAux idx 0 0 body s = some s`)| `if i < n` 守卫的自然 base case。 |
| 终止度量 | `(sizeOf body + 1, n − i)` lex | 按 `PLAN.md`。 |
| 嵌套 loop 名字冲突 | 用户保证 `RegName` 不相交 | 外层 loop `"i"`、内层 loop `"j"` 等——明确 *不是* scope 机制。 |

## 4. 引理族

放在新文件 `VeriTile/LoopInvariant.lean`。引理族有一个 master
lemma 加一小组 ergonomics corollary。Corollary 按需添加;Phase B 只需要
下面两个。

### 4.1 Master lemma `forLoop_inv`(Form 1)

```lean
theorem forLoop_inv
    {idx : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx (.scalarNat i)) = some s' ∧
          P (i+1) s') :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      P n s_final
```

证明大纲:

- 推广到 state-quantified 辅助形式:对所有 `i ≤ n` 与所有满足 `P i s` 的
  `s`,存在 `s_final` 使
  `stepForLoopAux idx i n body s = some s_final` 且 `P n s_final`。
- 对 `n − i` 强归纳(等价于从 `i = n` 开始的 `Nat.le_induction`)。
- Base `i = n`:`stepForLoopAux idx n n body s = some s` 直接成立。
- Step `i < n`:调用 `h_step i s` 得到 `s'` 使 `stepStmts body (s.setReg idx ...) = some s'` 且 `P (i+1) s';再与 `i+1` 的归纳假设组合。
- 最后 `forLoop_inv` 是 `i = 0` 的实例,根据定义有 `stepStmt (.forLoop ...) s_init = stepForLoopAux idx 0 n body s_init`。

### 4.2 读出 corollary `forLoop_readout_scalar`

```lean
theorem forLoop_readout_scalar
    {idx outReg : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    {f : Nat → ℝ}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx (.scalarNat i)) = some s' ∧
          P (i+1) s')
    (h_readout : ∀ s, P n s → s.regs outReg = some (.scalar (f n))) :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      s_final.regs outReg = some (.scalar (f n))
```

对 online softmax / layernorm:`outReg` 是 `"m"`、`"l"`、`"μ"`、`"var"` 等;
`f n` 是 kernel 在 `n` 次迭代后计算出的 closed-form 值。

### 4.3 读出 corollary `forLoop_readout_tile`

```lean
theorem forLoop_readout_tile
    {idx outReg : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    {len : Nat} {f : Nat → Fin len → ℝ}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx (.scalarNat i)) = some s' ∧
          P (i+1) s')
    (h_readout : ∀ s, P n s → s.regs outReg = some (.tile len (f n))) :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      s_final.regs outReg = some (.tile len (f n))
```

对 FA-1 forward:`outReg = "O"`、`len = D`、`f n d = O_n d`。

### 4.4 推迟的 corollary

`forLoop_readout_nat`、`forLoop_readout_tileNat`、`forLoop_readout_two`
(一次读两个寄存器,例如 online softmax 的 `(m, l)`)—— 在 Tier 2 实现
首次出现具体需求时再加。不要预写;YAGNI。

### 4.5 为什么不内置 `InitInvariant`

早期草稿在 `forLoop_readout_scalar` 里烧入一个 `InitInvariant s_init`
假设,使用户传 `h_init_state : InitInvariant s_init`,然后内部派生
`P 0 s_init`。我们拒绝了这一做法:corollary 应保持对 `P` 入口条件
的参数化。上下文里有 `InitInvariant` 谓词的 caller 用一个本地 `have`
预先 discharge `P 0 s_init`:

```lean
have h0 : P 0 s_init := h_init_state.toP0
forLoop_readout_scalar h0 h_step h_readout
```

这样 corollary 可在 setup 谓词差异很大的 kernel 之间复用。

## 5. 文件布局

| 文件 | 状态 | 责任 |
|---|---|---|
| `VeriTile/Semantics.lean` | 修改 | 把 `stepStmt` / `stepStmts` 转成 `mutual` 块;加 `stepForLoopAux`;替换 forLoop 的 `none` 占位;加 `termination_by`。 |
| `VeriTile/LoopInvariant.lean` | 新建 | `forLoop_inv` master lemma;`forLoop_readout_scalar` 与 `forLoop_readout_tile` corollary。 |
| `VeriTile.lean` | 修改 | 加 `import VeriTile.LoopInvariant`。 |

`Core.lean` 不变(`Stmt.forLoop` 构造子已存在;只有它的语义之前
是占位)。

## 6. 验证

- 每次文件改动后 `lake build` clean。
- `Examples/SoftmaxEq.lean`、`LogSumExpEq.lean` 等里现有的 P1 `simp`
  walkthrough 继续通过类型检查(`mutual` 块改变 reduction 行为不应导致
  regression)。
- `LoopInvariant.lean` 里一段小 `example` 块演练 1-stmt body forLoop,
  并用 `forLoop_inv` 关掉。具体来说:一个 *Nat-channel* 计数器
  `cnt := cnt + 1`(因为 `idx` 是 `Value.scalarNat`,通过 `Value.bop`
  混合 `ℝ` 和 `Nat` 返回 `none`——所以 example body 不能是
  `acc := acc + idx` 配 `acc : Value.scalar`)。Invariant 是
  `P k s := s.regs "cnt" = some (.scalarNat k)`,readout corollary 调用
  是(推迟的)`Nat` variant 的 `forLoop_readout` —— 或者,在
  `forLoop_readout_nat` 存在之前,用 master `forLoop_inv` 加手工
  `obtain`。这验证 master lemma + 基础的 mutual elaboration。

## 7. 风险(承自 `PLAN.md` Phase B 风险登记)

| 风险 | 缓解 |
|---|---|
| `mutual` 块 + `decreasing_by` cascade 没法干净 elaborate | fall back 到 `Nat.iterate` 风格的 `stepForLoop`,通过 `Nat.rec` 定义,与 `stepStmt` 的结构递归解耦。少量 proof-engineering 成本;语义不变。 |
| `mutual` 改动后 `simp` 在 `stepStmt` 周边变得脆 | 加 `@[simp]` 化简引理把 `forLoop`/非-`forLoop` case 暴露为分立等式;这是局部工程,不影响接口。 |
| Step 假设 ergonomics:body 用 `.bind` 链时 `∃ s'` 形态笨拙 | 如果 3+ kernel proof 出现,提供小的 simp/`Option.bind` helper;否则交给用户。 |
| 下游 proof 在 Tier 2 关闭前需要 `forLoop_readout_two`(一次读 2 个寄存器)| 在首次出现具体需求时加 corollary,沿用 `forLoop_readout_scalar` 的形态。 |

## 8. 不在范围(本规范)

- Multi-program-id grid 协调(Phase D)。
- `tl.dot` / 2D tile 语义(Phase C)。
- Mask 处理(Phase C)。
- Backward-pass forLoop(P3+)。
- 任何对 `Core.lean` 中 `Stmt.forLoop` 构造子签名的改动。
