# DType 擦除与 Compute Kernel surface

VeriTile 使用两层 kernel:

- `Kernel` / `Op` 是算法层。它带有数学 dtype:`.real`、`.fp32`、`.fp16`、
  `.bf16`、`.int`、`.nat`、`.bool`、pointer 和 block-pointer channel。
- `ComputeKernel` 是面向 compute 的 DSL surface。`triton { ... }` 宏始终产出
  `ComputeKernel`。

DSL 只有一个面向 compute 的 surface。每个 kernel 都以
`ComputeKernel.mk inputs outputs body` 形式发出,其中 `body` 是
`ComputeStmt` 列表。纯算法语句以 `ComputeStmt.alg` 项表示;面向 compute 的
语法使用其他 `ComputeStmt` 构造子。

## 算法投影

`ComputeKernel.toAlgorithm?` 是从 compute surface 到算法层的公开投影:

```lean
ComputeKernel.toAlgorithm? : ComputeKernel -> Except EraseDTypeError AlgKernel
```

投影会递归 lowering 每个 `ComputeStmt` / `ComputeExpr`;`ComputeStmt.alg`
原样透传。无法被投影的 compute-only 特性必须在这里失败,而不能伪造算法
含义。

`ComputeKernel.ComputeCorrect` 是公开的证明 surface,隐藏了 `Except`
管线。默认情况下,它只要求投影后算法层的证明:

```lean
ComputeKernel.ComputeCorrect ck post (gap := .ignore) :=
  True ∧
    match ck.toAlgorithm? with
    | Except.ok ak => Kernel.Correct_without_Rounding ak post
    | Except.error _ => False
```

当某 theorem 需要记录已通过外部验证的 compute-to-algorithm gap 时,使用
required gap contract:

```lean
ComputeKernel.ComputeCorrect ck post (gap := .require contract) :=
  ExternalChecked contract ∧
    match ck.toAlgorithm? with
    | Except.ok ak => Kernel.Correct_without_Rounding ak post
    | Except.error _ => False
```

`ComputeKernel.ComputeRefine` 是对应的双 kernel surface,带相同的可选
`GapPolicy`。内部 helper 引理在专门处理投影后算法 kernel 时,仍可使用
`Kernel.Correct_without_Rounding` / `Kernel.Refine`。

转换引理:

```lean
ComputeKernel.computeCorrect_of_toAlgorithm_eq :
  ck.toAlgorithm? = Except.ok ak ->
  Kernel.Correct_without_Rounding ak post ->
  ComputeKernel.ComputeCorrect ck post

ComputeKernel.computeRefine_of_toAlgorithm_eq :
  lhs.toAlgorithm? = Except.ok lhsAlg ->
  rhs.toAlgorithm? = Except.ok rhsAlg ->
  Kernel.Refine lhsAlg rhsAlg rel ->
  ComputeKernel.ComputeRefine lhs rhs rel
```

这样既保留了投影后算法证明的可复用性,又把投影成功显式化。

为这个投影提供了一个定义展开引理:

```lean
ComputeKernel.eval_eq_exec_of_toAlgorithm? :
  ck.toAlgorithm? = Except.ok ak ->
  ∀ s, ck.eval s = exec ak s
```

这 **不是** soundness theorem。`ComputeKernel.eval` 的*定义*就是
`match ck.toAlgorithm? with | Except.ok ak => exec ak s | Except.error _ => none`,
所以上面这条引理只是成功分支下的定义展开。它是给 `simp` 链用的便利引理,
不是关于 compute-vs-algorithm 语义的论断。

**有意为之:Lean 内部没有把 bit-level compute 执行连接到算法执行的
theorem。** `ComputeKernel.ComputeCorrect` 是关于投影后算法 kernel,加上
可选的外部 gap 义务的陈述。形式化的 compute-to-algorithm 桥是
`ComputeKernel.toAlgorithm?` / `eraseDType`:它把面向 compute 的语法映射到
Real / Int / Nat 算法层,Lean 证明在这一层进行。compute 层的数值行为——
IEEE rounding、NaN propagation、denormal、hardware-dot precision、
fast-math 等——通过外部 gap checker 进行经验验证(见 PLAN.md
"Verification architecture" 与 #59),而不是通过 Lean theorem。

阅读 `ComputeCorrect` Lean 证书的用户应这样理解:"投影后的算法结构(在
Real / Int / Nat 上)在 Lean 中已被证明正确。" 如果 theorem 使用了
`gap := .require contract`,它还记录了对应 compute-to-algorithm gap
contract 的可信外部验证 token。

## 命名

按设计存在两个 `eraseDType` 家族:

- 算法层 `Op.eraseDType` / `Kernel.eraseDType` 折叠算法 dtype tag,例如
  `.fp32 -> .real`。
- compute 层 `ComputeKernel.toAlgorithm?` 把面向 compute 的 kernel 投影到
  算法层,在出现 compute-only 操作时可能失败。

`FloatDType.eraseFloat` 有意仍叫 `eraseFloat`;它是 float-only 的 witness 操作,
不是全局的 dtype-erasure API。

## 整数宽度拼写

算法正确性把整数宽度视为仅拼写差异:

- `tl.int8`、`tl.int16`、`tl.int32`、`tl.int64` 映射到 `.int`
- `tl.uint8`、`tl.uint16`、`tl.uint32`、`tl.uint64` 映射到 `.nat`

这些拼写在算法层不引入 wraparound、overflow、signed-width 或 bit-vector
payload 语义。

## Bitcast 策略

`tl.bitcast` 是 compute-only。它不能被建模成数值 `tl.cast`,也不能变成
`AlgOp.bitcast`。

DSL 接受在 `tl.uint32`、`tl.int32` 和 `tl.float32` 之间做 32-bit payload
reinterpretation。当目标类型可在数学算法层表达时,常量 uint32 bit pattern
是可投影的:`tl.uint32` 投影到 `.nat`,`tl.int32` 投影到 `.int`,
`tl.float32` 仅在这些 bit 解码出一个 finite-normal binary32 值时投影到
`.real`。运行时 bitcast 由 `ComputeOp.bitcast` 表示,但
`ComputeKernel.toAlgorithm?` 对它返回 `Except.error`。

### 同一个 bitcast 的两种 view

对于一个被接受的常量 `tl.bitcast`,宏会发出两个并行 term:

- **算法 view**(`EOut.term`):`Op.const ((Float32Bits.decodeRat
  { bits := BitVec.ofNat 32 <bits> }).get (by decide) : ℝ)`。这是投影后的
  算法侧 term。
- **Compute view**(`EOut.computeTerm`):`ComputeExpr.compute (ComputeOp.bitcast
  ComputeDType.uint32 <dst> rfl (ComputeOp.const ⟨BitVec.ofNat 32 <bits>⟩))`。
  外围 kernel 在路由到 `ComputeKernel.mk inputs outputs body` 时使用的就是
  它。

两种 view 都把 `Float32Bits.decodeRat` 当作单一权威 decoder。系统中没有第二个
IEEE decoder;特别是宏不会在 `MacroM` 中重新实现 decoding。算法 view 的
`Option.get (by decide)` 在类型检查时,在具体的 `BitVec 32` 字面量上展开
decoder。

### 宏期 admissibility

`tl.bitcast` 在宏展开时只接受已建模的 32-bit payload 目标:

- `dst ∉ {tl.uint32, tl.int32, tl.float32}` → `Macro.throwError`。
- 数值字面量源必须能装入 `uint32`(`bits < 2^32`)。
- 字面量 `dst = tl.float32` 进一步拒绝 exponent field
  `(bits / 2^23) % 256 = 0`(zero/subnormal)或 `255`(NaN/Inf),因为这些
  值目前还不可投影到算法层。
- 非字面量源以运行时 `ComputeOp.bitcast` 形式接受;到算法层的投影会以
  `Except.error (.requiresComputeSemantics "runtime bitcast")` 失败。

任何通过上述全部检查的 `tl.float32` 字面量都是 finite-normal binary32
pattern。对于这些 bit,`Float32Bits.decodeRat` 按构造为 `some _`。因此
发出的算法侧 `Op.const` 中的 `Option.get (by decide)` 是 *defense-in-depth*
断言:如果将来宏期 admissibility 检查出现回归,elaboration 会在这里大声
失败,而不是悄悄替换为 placeholder 常量。运行时 bitcast 没有算法 fallback
常量。

### 代表性 theorem

```lean
ComputeOp.oneBitcast_toAlgorithm :
  ComputeOp.constOpToAlgorithm? ComputeOp.oneBitcast = Except.ok (Op.const 1)

ComputeOp.minusOneBitcast_toAlgorithm :
  ComputeOp.constOpToAlgorithm? ComputeOp.minusOneBitcast =
    Except.ok (Op.constInt (-1))
```

不检查 bitcast 值的算法证明无需改动即可继续工作;检查特定解码值的证明会
通过 `Float32Bits.decodeRat` 化简(例如 `decide` / `native_decide`)。

### 不在本文范围内

运行时 bitcast 现在可以表达,但仍是 compute-only:

- ComputeOp 的表示必须保持足够忠实,以便测试 pipeline 能把 `ComputeKernel`
  反提到 Triton 源(这是连接到 compute 层数值行为的桥梁——不存在覆盖运行时
  compute 行为的 Lean 内部 IEEE soundness theorem);
- 不支持的运行时情况会让 `ComputeKernel.toAlgorithm?` 返回 `Except.error`,
  因此 `ProjectedCorrect` / `ComputeCorrect` 在 Lean 中无法对其声明;
- `Except.error` 分支不能被 inline 到带 fallback 常量的算法侧 `Op` term。
