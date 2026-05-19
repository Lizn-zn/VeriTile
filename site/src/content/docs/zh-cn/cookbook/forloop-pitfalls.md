---
title: forLoop_inv 陷阱
description: 用 forLoop_inv / forRange_inv 证 kernel 时反复栽跟头的七个 tactical 坑。
---

第一次非平凡的 `forLoop_inv` 使用(最痛苦的是 `online_welford_correct`)
时栽出来的坑。`Triton.Semantics` 里 `mutual` 块周围的 simp / unfold 机制
在特定地方很脆。subagent 不被提醒就会再栽一次。

派 kernel 证明任务(body 被 loop 包,pre/post 有 statement)时,把下面
当作"已知 hazards" 塞进 prompt。

## 1. `forLoop_unfold` 对外层 `simp only` 太热心

朴素地写

```lean
simp only [exec, kernel, stepStmts, stepStmt, evalOp, ...]
```

会把 `forLoop` arm unfold 成 `stepForLoopAux` 形式,之后 `forLoop_inv`
的结论(在 `stepStmt (.forLoop ...)` 形式)就无法 substitute 上去了。

**绕过:**用私有 helper `stmts_cons` / `stmts_nil`,证法是

```lean
conv_lhs => unfold stepStmts; rw [h]
```

逐 statement 串过 pre/post。直接写 `unfold stepStmts; rw [h]` 会留下一个
半化简的 match,因为 `some` 分支上的递归 `stepStmts` 也会 unfold。

## 2. `stepStmts [] s = some s` 不是 `rfl`

mutual block 的方程编译器**不**把它做成 definitionally 相等。空尾巴 base
case 必须 `unfold stepStmts` 才能 discharge。

## 3. `if_neg h_mv` 在 `∧`-条件里不会自动触发

`BlockState.writeMem` unfold 后,`if` 条件是

```text
region = region ∧ off = off
```

直接 `simp [if_neg h_mv]` 可能无法 rewrite 这个 conjunction。

**绕过:**作为 simp lemma 传

```lean
(fun h : meanReg = varReg => h_mv h)
```

让 simp 直接消第一个合取项。

## 4. 值通道 match cascade 让 per-iteration body 证明很脆

每个 statement 通过 `match s.regs name` 解出 register。要走过(比如)
Welford body 的五条 statement,需要在每一层都带着平行的
`have h{name}{i}` 事实,加上显式 `BlockState.setReg` unfold。

**预算:**每条 body statement 大约**五个** simp 块,不是一个全局 `simp`。
"显式 `setReg` 链 + per-step `have`" 这种分阶段写法能编译过;一个大
`simp` 要么打圈要么失败。

## 5. Bool-mask vs Prop-if 不匹配

masked load/store 的 kernel correctness,`simp` 会自动把

```text
if (decide P : Bool) then ...
```

通过 `decide_eq_true_eq` 改写成

```text
if (P : Prop) then ...
```

用 Bool mask 写的 lemma —— 比如 `BlockState.scatter_readback_masked` ——
就 match 不上 simp 后的 goal 了。

**绕过:**额外提供 Prop-flavored 推论(`scatter_readback_prop_masked`),
收 `P : Fin n → Prop` 配 `[DecidablePred P]`,内部用
`simp only [decide_eq_true_eq]` 做 Bool→Prop 转换。bench 的 masked-add
proof 最早就卡在这里。

## 6. `List.Disjoint` 是 4-arg 形式,不是 2-arg

Mathlib 当前的 `List.Disjoint l₁ l₂` unfold 成

```text
∀ a ∈ l₁, ∀ b ∈ l₂, a ≠ b
```

—— 显式 4-arg,不是更老的 `∀ a ∈ l₁, a ∉ l₂` 2-arg 形。

从 `hl1_disj : Disjoint l₁ (i :: l₂)` 和 `hk : i ∈ l₁` 提矛盾,正确调用是

```lean
(hl1_disj i hk i List.mem_cons_self) rfl
```

第二个 `i` 和 `mem_cons_self` 提供 `b` 和它的成员关系;得到的 `i ≠ i`
被 `rfl` 反证。

## 7. `subst` 方向意外

如果有 `hki : k = i`,其中 `i` 是定理参数,`k` 是 `intro` 的 local
binding,`subst hki` 把 `i` 替换成 `k` —— 消的是**参数**,不是 local。
下文对 `i` 的引用全炸。

**绕过:**用

```lean
rw [hki] at <hyp>
```

只在想替的那条 hypothesis 里把 `k → i`,其他 context 保持。

---

派 sub-agent 做 `forLoop_inv` kernel 证明时,把上面七条作为已知 hazards
塞进 prompt。Welford 证明在这些 pattern 就位后用了 +297 行;朴素 simp
驱动的尝试要么打圈要么失败。
