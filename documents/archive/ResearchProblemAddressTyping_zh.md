# RP2: 地址 typing —— ℝ-uniform 与 Nat-bifurcated `Value`

**状态:** 已解决。Phase A(打磨后)→D 默认:**bifurcated**
(`Value` 把 `scalarNat`/`tileNat` 与 `scalar`/`tile` 分开携带)。
**日期:** 2026-04-27。
**Owner:** Phase A 打磨(Tier 1 close 与 Phase B start 之间)。
**重审条件:** 见 §何时重审。

---

## TL;DR

最初,`Value` 只有单一 scalar carrier `ℝ`:

```lean
inductive Value where
  | scalar : ℝ → Value
  | tile   : (n : Nat) → (Fin n → ℝ) → Value
```

内存地址(例如 `pid * BLOCK + tl.arange(N)[i]`)在 evaluation 时按
**`ℝ` 算术** 计算,然后在 offset 到达 `BlockState.mem` 时通过
`realToNat : ℝ → Nat := ⌊·⌋₊` floor 回 `Nat`。每个 kernel proof 都带
8 行 `hcast` boilerplate,重新建立 `(pid : ℝ) * (N : ℝ) + (k.val : ℝ)`
的 round-trip 是 identity。

我们 bifurcate `Value`,把 `Nat` 与 `ℝ` 分开携带。`realToNat` cast
完全移除;地址算术从头到尾留在 `Nat`。

---

## 1. 问题陈述

`Value`(`evalOp` 产生并存进 register 的运行时值类型)应使用:

- **Uniform `ℝ`** —— 单一 scalar carrier;地址(`pid`、`arange`、offset)
  以 `ℝ`-valued 编码,在 `mem`-访问边界 floor 到 `Nat`;*或*
- **Bifurcated `ℝ` / `Nat`** —— `Nat`-valued 标量 / tile(用于地址、
  size、index)与 `ℝ`-valued 标量 / tile(用于数据、accumulator、
  归一化输出)分开的构造子。

这一决定贯穿 `Op.programId` / `Op.arange` / `Op.constNat`、`Value.bop`、
`Value.uop`、`Value.reduceSum/Max`、`evalOp.load/.store`,以及每个
kernel proof 的结构。

## 2. 为何 uniform `ℝ` 是 Phase-A 默认

bootstrap 时合理的三个理由:

1. **uniform `Value` → 更简单的 `evalOp` dispatch。** 单一 scalar carrier
   意味着 `Op.add` 总是 evaluate 成 `(· + · : ℝ → ℝ → ℝ)`;不必按操作数
   出处分情况。

2. **`Value.bop` 保持 4 个 case。** Pointwise binary lift over scalar/tile
   是 4 种组合。加 `Nat` carrier 把每个轴翻倍,变成 16 种组合
   (大多数是无意义的混合类型 case)。

3. **Triton 数据值 *是* 浮点。** `Op.exp`、`Op.div`、`Op.log` 与
   `Op.reduceSum/Max` 都对 `ℝ` 操作。把它们编码进 `ℝ`-only carrier 是
   显然选择;地址被顺手带上,是意外不是 design。

代价(只在 Tier 1 close 后才完全显现)是:

- `realToNat` 是 hack。它唯一的工作是撤销 `Op.programId` / `Op.arange`
  引入的隐式 `Nat → ℝ` cast。语义文件里明确标了
  `TODO(P1 polish)`。
- 每个 kernel-correctness proof 携带一个 `hcast` 引理 —— 8 行 ——
  重新证 `realToNat (↑pid * ↑N + ↑k.val) = pid * N + k.val`。五个 proof
  逐字共享这一段,≈ 40 行重复 boilerplate。
- 每个 `simp`-reduce 后的 kernel goal 都带 `(↑s.pid : ℝ)` cast 加上
  最外层 `realToNat` wrapper,视觉噪声升高。
- `evalOp .exp Op.programId` 在 uniform 模型下 **静默地 well-typed** ——
  `exp(pid)` evaluate 成完全合法的 `ℝ`。语义太宽松:接受了任何 Triton
  作者都不会写的程序。

## 3. 为何 bifurcation 划算

Tier 1 close 后,成本-收益对比转向:

- **`hcast` 完全消失。** 地址算术从头到尾留在 `Nat`。kernel goal 直接
  抵达 scatter-readback 形态,offset 是 `s.pid * N + k.val : Nat`,无 cast。
- **Phase B/C/D 扩展。** 每个新 kernel pattern 都带新地址算术 ——
  `forLoop` 引入 `idx`(一个 `Nat` 计数器);FA forward 加 2-D layout
  `i * stride_i + j * stride_j`;FA-2 有 multi-axis `program_id`。无
  bifurcation,每个 pattern 都需要自己的 `hcast` 引理 —— 到 Phase C
  结束 ≈ 10 个这样的引理。有 bifurcation:0 个。
- **Type-level 守护反对无意义 kernel。** Bifurcation 后,
  `Op.exp Op.programId` 是 `evalOp` 错误(`none`):`Value.uop Real.exp`
  不能接受 `scalarNat`。语义层拒绝把 float op 应用到地址值的程序。
- **`Value.bop` 8 个 case 可接受。** 16 个假想积 case 一半是 ℝ-only(4)
  或 Nat-only(4);其余是混合类型错误,塌缩到一个 `_, _ => none` arm。

## 4. Bifurcation,具体

```lean
inductive Value where
  | scalar    : ℝ   → Value             -- 数据 scalar(`e / s`、accumulator)
  | scalarNat : Nat → Value             -- 地址 scalar(`pid`、`idx`、size)
  | tile      : (n : Nat) → (Fin n → ℝ)   → Value   -- 数据 tile(`tl.exp`、…)
  | tileNat   : (n : Nat) → (Fin n → Nat) → Value   -- offset tile(`arange`、…)
```

`evalOp` 映射:

| Op | 旧 | 新 |
|---|---|---|
| `Op.const c`        | `scalar c`                     | `scalar c`(不变)|
| `Op.constNat n`     | (不存在)                       | `scalarNat n`(新构造子)|
| `Op.programId`      | `scalar (s.pid : ℝ)`           | `scalarNat s.pid` |
| `Op.arange n`       | `tile n (fun i => (i.val : ℝ))`| `tileNat n (fun i => i.val)` |
| `Op.add`/`mul`/`sub`/`div` (Nat × Nat) | 不支持 | `scalarNat`/`tileNat` 算术经 `bop` |
| `Op.exp`/`log` (on Nat)               | 静默 cast | `none`(语义错误)|
| `Op.reduceSum/Max` (on Nat tile)      | 静默求和 | `none` |
| `Op.load region offsetExpr`           | `realToNat (eval offsetExpr)` | 直接来自 `scalarNat`/`tileNat` 的 `Nat` |
| `Stmt.store region offsetExpr value`  | `realToNat ...` | 直接 `Nat` |

`realToNat` **删除**。

DSL 约定更新:

| DSL 形态 | 旧展开 | 新展开 |
|---|---|---|
| `$(t : Nat)` antiquote | `Op.const ((t : Nat) : ℝ)` | `Op.constNat t` |
| 数字字面量 `5` | `Op.const ((5 : Nat) : ℝ)` | `Op.const 5`(ℝ)|
| `tl.arange(N)`(字面 / `$(N)`)| `Op.arange N`(Nat)| `Op.arange N`(不变;产 `tileNat`)|

约定:**`$(...)` antiquote 是地址 / size channel(`Nat`);
裸数字字面量是数据 channel(`ℝ`)。**

## 5. Cost / benefit

| 维度 | Uniform `ℝ` | Bifurcated |
|---|---|---|
| `Value` 构造子           | 2          | 4 |
| `Value.bop` case         | 4          | 8 + 1 混合类型 catch-all |
| `realToNat` 存在         | 是         | **否** |
| 每 proof `hcast`         | 8 行       | 0 |
| 抓出的 type 错误:`exp(pid)` | 否     | **是**(返回 `none`)|
| Phase C 2-D 地址 proof tax | × pattern | 0 |

净:~+60 行新语义基础设施;~−40 行(加未来节省)proof boilerplate;
消除整一类弱 typing。

## 6. 实现日志

- Phase-A 脚手架(≈ 2026-04-25)有意选 uniform `ℝ`;在 `Semantics.lean`
  里标 `TODO(P1 polish): bifurcate Value into .scalarReal / .scalarNat …`。
- 2026-04-27,Tier 1 close 之后:bifurcation 一批执行。Tier 1 proof
  甩掉 `hcast` 引理(5 个文件中逐字删除);`realToNat` 移除;
  `Value.bop` 扩展;`evalOp` dispatch 添加。

## 7. 何时重审

- **如果我们曾需要 `ℝ` 中的一等公民索引算术**(例如某个非 Triton DSL 的
  连续索引),bifurcation 可能需要第三个 carrier。Triton 不在 scope。
- **如果 `Value.bop` 的 case-split 成为瓶颈**(例如 Phase C 引入许多
  跨类型语义微妙的 op variant),重审是否把 dispatch 内联到 `evalOp`。
- **如果我们决定建模 IEEE-754 fidelity**(当前不在 scope),数据 carrier
  可能从 `ℝ` 改成更丰富的类型;bifurcation 只会让那个选择更清晰 ——
  地址不论如何都属于 `Nat`。

---

*另见 `RP1`(`research_problem_pointer_vs_named_region.md`)关于 region
naming 的相关决定。两个决定都把框架推向更强的内存操作 typing:region
静态命名(不是一等公民 pointer),地址算术静态 `Nat`(不是 floor 后的
`ℝ`)。*
