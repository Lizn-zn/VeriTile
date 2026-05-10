# VeriTile 证明约定

[English](ProofConventions.md) | **中文**

从反复写 kernel 证明里总结出来的 tactic 级约定。**不**是绝对规则 —— 证
明写起来不舒服时可以偏离,但下面这些是 VeriTile 大多数 kernel 证明的默认
选择。

## Carrier 桥接和 `erw`

VeriTile 的 kernel-side 计算住在 `WithBot ℝ` 值的 carrier 里:
`Tile.reduceSum (Tile.bop mul ...)`, `Option.map₂`, `Option.map`,
`WithBot.realSqrt`, `WithBot.unbotD`。数学层是纯 `ℝ`。我们用
`VeriTile.Triton.Semantics.MaskedReduction`(以及其它同级 `Semantics/` 机
制文件)里的定理桥接两者:

```lean
theorem reduceSum_masked_sq_eq_some_sum
    (load : Fin BLOCK_N → ℝ) (active : Fin BLOCK_N → Prop) [DecidablePred active] :
    @Finset.sum (Fin BLOCK_N) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·) (if-shape) (if-shape))
      = some (∑ k, if active k then load k * load k else 0)
```

### `rw` / `simp_rw` / `simp` 经常 fail; `erw` 是 fallback

上面的桥引理看起来是干净的 rewrite target。实际上,**`rw [bridge _ _]`,
`simp_rw [bridge _ _]`,甚至 `simp [@simp bridge _ _]` 都无法 unify** 桥
引理的 `∑ k, Option.map₂ ...` pattern 和 kernel 目标里的
`∑ x, Option.map₂ ...`,虽然两者 alpha-等价。原因看起来是 Lean 的 tactic
模式匹配器无法跨越 lambda + `FloatDType.cast` / coe wrapper 的边界。

**默认 fallback**: 用 `erw`(extended rewrite,允许 up-to-defeq 匹配)。
能跑通的模式:

```lean
have hcarrier := bridge_lemma (fun k => …kernel-side load…)
                              (fun k => …kernel-side mask…)
simp [bridge_unfolds] at hcarrier
-- 现在 hcarrier 是 kernel-shape 形式
…
simp [..., FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
      WithBot.realSqrt, ...] -- 把 carrier 暴露出来
erw [hcarrier]                -- 桥到纯 ℝ 形式
rfl                           -- 两边结构相等了
```

`bench/l2_norm_triton1` 里的 `l2VarCarrier_eq_l2NormSqSum` 就是靠 `erw`
unblock 的。看那个文件里的具体写法。

### `erw` 的代价

`erw` 比 `rw` 贵,因为它每次匹配尝试都做一遍 definitional unfold。在单
个证明步骤里没问题,但**不要**把 `erw` 放进全局 simp 集合或重循环里 ——
失败的匹配尝试会累积。`erw` 一次只用一处。

如果 `rw` 能用,优先 `rw`。只有当目标里有 lambda-bound 变量 + cast/wrapper
结构挡住直接 unification 时,才换 `erw`。

## 开场 simp:优先用 `tile_elementwise`

Kernel 证明通常以一个 "kernel-exec 标准化" simp 开场。用
`VeriTile.Triton.Semantics.BroadcastReshape` 注册的 `tile_elementwise`
simp set 取代以前那串 8-15 个 lemma 名:

```lean
simp [exec, <kernel_def>, stepStmts, stepStmt, evalOp,
      tile_elementwise] at hExec
```

`tile_elementwise` 一次性展开 `Tile.bop / Tile.cop / Tile.uop /
Tile.ptrAdd`、reshape 算子(`Tile.expandDim / Tile.ofReal /
Tile.natToReal / Tile.dot / Tile.transpose / Tile.select`)、reduction
(`Tile.reduceSum* / Tile.reduceMax*`)、shape 算术(`TileShape.*`)、按
dtype 的 projection(`NumericDType.* / IntegralDType.* /
ComparableDType.* / FloatDType.cast / FloatDType.ofWithBot /
FloatDType.toWithBot`)。

要扩 set,在 `Semantics/BroadcastReshape.lean` 加 `attribute
[tile_elementwise]` 一行(attribute 本身在 `Semantics/BroadcastReshape/Attr.lean`
注册,因为 Lean 4 不允许在同一文件 declare + use 同一个 simp attribute)。

`subst s'` 之后的 simp 收尾通常带 per-kernel `*Spec`。如果 `tile_elementwise`
已经够用,收尾就是 `simp [hi, *Spec]`,不再需要额外的 dtype / tile 名字。

**已有证明**: 早于 `tile_elementwise` 的 kernel 文件保留原 simp 列表,迁
移是机械动作 —— 把 5-15 个 `Tile.*` / `NumericDType.*` /
`ComparableDType.*` / `FloatDType.*` / `TileShape.*` 名字换成单个
`tile_elementwise` token。完整示例见 `bench/swiglu_fwd`、
`bench/swiglu_triton`、`bench/swiglu_backward`、`bench/geglu_tanh_triton`、
`bench/masked_add_cuda`、`bench/fused_activation`、`bench/triton_softmax`、
`Examples/FlashAttention1/NaiveKernel.lean`。

## `BlockState.pid_eq` 标准化

Kernel 证明里 `s.pid` 和 `s.pids 0` 经常混用。它们 definitionally 相等
但 `rw` 区分它们。约定: 把 `BlockState.pid_eq` 加进 simp 列表,让所有
`s.pid` 都被改写成 `s.pids 0`(或反过来,取决于引理方向)。在调用任何
`BlockState.scatter_readback_*` 引理**之前**做这步。

## scatter readback 的 `Function.Injective` 模板

绝大多数 kernel 证明需要一个 injectivity 引理喂给
`BlockState.scatter_readback_prop_masked_nd`(或其无 mask 变体)。标准
形式:

```lean
have h_inj : Function.Injective
    (fun idx : TileIndex [BLOCK_N] => s.pids 0 * stride + idx.1.val) := by
  rintro ⟨a, _⟩ ⟨b, _⟩ hab
  obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
  rfl
```

这一段在几十个 bench 文件里都长得一样。如果某个 kernel 偏离这个形状(比
如 2D scatter, multi-pid, broadcast),在偏离处加注释说明原因,后人才能
看懂。

## Active-only spec

Kernel correctness theorem 用 `ComputeCorrect.WriteMap.writeIf` 来 gate
active mask。`*Spec` 是 **active lane 的 expected value** —— inactive
lane 走 `writeIf`-preserve 路径,根本不消费 spec。这意味着 spec 可以在所
有 `Fin BLOCK_N` 索引上有定义,即使只 active 的会被观测,把
`Math.l2Norm (load_with_masked_zero_tail)` 写成 inactive lane 也合法的形
式没问题 —— writeIf 会吃掉 inactive 结果。

**不要**在 spec 里加 `if active then ... else preserve` 来模仿 writeIf
—— 让 writeIf 做 masking,spec 保持纯净。

## 相关

- [`CodeOrganization.md`](./CodeOrganization.md) —— 三层结构(Math /
  Semantics / per-kernel glue)
- [`CorrectnessSurfaces.md`](./CorrectnessSurfaces.md) —— 用户面 theorem
  surface(`Realizes`, `WriteMap`, `OutputReadable`)
