# 支持的 Triton 子集与语义 gap

本文档记录 VeriTile 当前嵌入 Lean 的 Triton-like surface syntax、背后的语义模型,
以及尚未建模的部分。它是 artifact 层面的契约:如果 kernel 使用了本文档之外的
语法,DSL 要么拒绝它,要么当前证明并不声称覆盖该 Triton 特性的真实语义。

## 范围概览

VeriTile 建模的是一个 typed Triton-style kernel language,不是 Python 执行语义,
也不是完整 Triton IR。kernel 写在 Lean 的 `triton { ... }` 块里,然后降到 typed
AST:

```lean
Op   : TileDType → TileShape → Type
Stmt : Type
```

当前 shape 模型是 ND tile,shape list 采用 outermost-first。标量 shape 是 `[]`;
矩阵 `[M, D]` 的索引类型是 `TileIndex [M, D]`。

## 当前支持的 surface syntax

### 控制流与 program id

- `tl.program_id(axis)`,其中 `axis` 是数字字面量或 `$(n)`。
  运行时状态保存 `pids : Nat → Nat`,所以任意 axis 都有定义。
- `tl.for i in $(n) { ... }` 与 `tl.for i in N { ... }`。
  loop 有操作语义,证明通过 `forLoop_inv`。
- 寄存器赋值:
  `x := expr`。

### 常量与 dtype

- 实数字面量: `0`, `1`, `3.5` 等,降到 `.real` channel。
- Nat 反引用: `$(n)`,降到 `.nat` channel。
- Real 反引用: `$ℝ(x)`,把 Lean `ℝ` 项降到 `.real` channel。
- `-inf` 降到 `Op.negInf`,内部表示为 `⊥ : WithBot ℝ`。
- `tl.toReal(x)` 把 `.nat` 标量/tile 转成 `.real`。

当前 channel:

- `.real`: 用 `WithBot ℝ` 建模,主要为了表达 `-inf`。
- `.nat`: 用于 offset、loop counter、size 和地址算术。
- `.bool`: 由比较产生,用于 mask 和 `tl.where`。

### Tile 构造与 shape 操作

- `tl.arange(n)` 与 `tl.arange(start, end)`。
  双参数形式降为 `start + tl.arange(end - start)`;
  `tl.arange(0, end)` 会折叠成 `tl.arange(end)`。
- `tl.full([dims...], value)`。
- `tl.zeros([dims...])`,是 `tl.full([dims...], 0)` 的糖。
- rank-1 输入上的 `e[:, None]` 和 `e[None, :]`。
  它们降到 `Op.expandDim`。
- `tl.trans(e)` 交换最后两个 axis,前面的 axis 作为 batch prefix 保持不变。

### 算术、比较与 broadcast

- 算术: `+`, `-`, `*`, `/`,作用于 `.real` 或 `.nat`。
  混合 channel 算术会被 DSL 拒绝。
- 点态比较: `<`, `<=`, `==`, `>`, `>=`, `!=`,作用于 `.real` 或 `.nat`,
  结果是 `.bool`。
- 双参数 `tl.max(a, b)` 作为 `.real` 上的点态 max。
- Broadcast 是 ND 的:同维度、scalar-to-tile、或维度 `1` 扩到另一边。
  DSL 通过语法构造 broadcast proof,所以语义等价但写法不同的维度表达式,
  可能仍需要写成一致形式。

### 一元数学函数

- `tl.exp`
- `tl.log`
- `tl.sigmoid`
- `tl.sqrt`

这些都作用于 `.real` channel。

### Reduction

- `tl.sum(x)`
- `tl.max(x)`
- `tl.sum(x, axis=N)`
- `tl.max(x, axis=N)`
- `keep_dims=true|false`

省略 `axis` 时遵循 Triton 的 `axis=None`:对所有维度做 reduction。
显式 `axis=N` 时只 reduce 一个 axis。当前 reduction 作用于 `.real` tile。

### 矩阵操作

- `tl.dot(a, b)`。
  作用于最后两个维度:
  `[..., M, K] × [..., K, N] → [..., M, N]`。
- `tl.dot(a, b, acc)` 支持 Triton accumulator 形式,降为
  `acc + tl.dot(a, b)`。
- `tl.trans(e)` 是 `tl.dot(Q, Kᵀ)` 需要的 trailing-two-axis transpose。

当前 `tl.dot` 是 `.real` 抽象上的数学矩阵乘法,不是 Triton 硬件 dot 指令的
bit-level / precision 语义。

### 内存操作

支持的 load:

- `tl.load($(region))`
- `tl.load($(region) + offset)`
- `tl.load(ptr, mask=mask)`
- `tl.load(ptr, mask=mask, other=other)`
- `tl.load(ptr, other=other, mask=mask)`

支持的 store:

- `tl.store($(region), value)`
- `tl.store($(region) + offset, value)`
- `tl.store(ptr, value, mask=mask)`

未知 kwarg 会报错。特别地,`tl.load(..., boundary_check=...)` 和
`tl.store(..., boundary_check=...)` 不会被静默忽略。

masked load 语义:

- `mask=true` 时读内存。
- `mask=false` 且提供 `other` 时返回 `other`。
- `mask=false` 且省略 `other` 时,Triton 语义是 undefined;VeriTile 用
  `BlockState.undef` 建模。

masked store 语义:

- `mask=true` 时写入该 lane。
- `mask=false` 时保持原内存不变。

## Memory model

VeriTile 不建模 first-class Triton/CUDA pointer。surface syntax 看起来像 pointer:

```lean
tl.load($(xReg) + offs)
tl.store($(outReg) + offs, value)
```

但 `$(xReg)` 是 Lean `RegionName`,不是 pointer value。内存模型是:

```lean
RegionName → Nat → ℝ
```

Offset 是显式 `.nat` 表达式。高维 tensor 通过用户写出的 strided offset 公式表示,
例如:

```lean
b * stride_b + h * stride_h + i * stride_s + d * stride_d
```

examples 用 `InputAt` 和 `Offset.strided` 等 helper 把这些 offset 公式连接到数学
tensor slice。aliasing 通过选择相同或不同的 `RegionName` 表示;pointer value、
pointer cast、pointer comparison、block pointer 都还没有建模。

## 浮点模型

算术目前是 `ℝ` abstraction:

- real data 建模为 `WithBot ℝ`。
- `-inf` 精确建模为 `⊥`。
- 语义里定义了 `exp ⊥ = 0` 和 `sigmoid ⊥ = 0`。
- store 会把 `⊥` 用默认值降回内存;well-formed kernel 不应该 store `⊥`。

含义是:当前 theorem 证明的是实数数学正确性,不是 IEEE-754 bit-level 等价。
rounding、NaN、signed zero、overflow、underflow、denormal、exception flag、
硬件 dot precision、fast-math rewrite 都未建模。

## 尚不支持或尚未真实建模

- 完整 IEEE-754 浮点语义。
- Triton block pointer 以及 block-pointer-only 限制。
- `boundary_check` / `padding_option`。
- first-class pointer 与 pointer-valued expression。
- named-region equality 之外的通用 pointer alias analysis。
- 完整 Python/Triton JIT 语义、decorator、meta-parameter execution,
  以及 `triton { ... }` 块外的 Python control flow。
- atomic operations。
- async copy / TMA / shared-memory staging。
- barrier、跨 program 或跨 warp synchronization。
- grid launch semantics。当前 theorem 描述一个带 `BlockState.pids` 的符号 program
  instance;整个 grid 的覆盖性需要手动对合法 program id 量化。
- cache/performance hint,例如 `cache_modifier`, `eviction_policy`, `volatile`,
  `is_volatile`。
- 任意 axis permutation。`tl.trans` 只交换最后两个 axis。
- 高 rank surface slicing。目前只支持 rank-1 的 `[:, None]` / `[None, :]`。
- integer width、overflow、signedness。`.nat` channel 是数学 `Nat`。

## 文档自动生成

目前还没有自动生成这个 subset 文档的工具。未来可以做一个脚本,从 DSL syntax /
`Op` constructors 抽取 raw table,再和本文档比对,用来抓
"实现了但没写文档" 或 "文档写了但 DSL 已不接受" 的漂移。

现阶段本文档仍应人工维护,因为 artifact 的关键声明是语义层面的,不只是语法列表;
上面的 gap 需要人来判断。
