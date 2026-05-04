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
- `tl.static_range i in $(n) { ... }` 与 `tl.static_range i in N { ... }`
  是同一个 bounded-loop AST 的 surface alias。当前不建模 unroll / pipeline
  属性。
- `tl.if cond { ... }` 支持 scalar bool 条件(`Op .bool []`)。
  当前没有 `else`、`break` 或 `continue`;Triton 的 block-skipping pattern
  应该通过取反 skip 条件并包住有效 body 来表示。逐元素条件选择仍使用 `tl.where`。
- 寄存器赋值:
  `x := expr`。

### 常量与 dtype

- 实数字面量: `0`, `1`, `3.5` 等,降到 `.real` channel。
- Nat 反引用: `$(n)`,降到 `.nat` channel。
- Real 反引用: `$ℝ(x)`,把 Lean `ℝ` 项降到 `.real` channel。
- `-inf` 降到 `Op.negInf`,内部表示为 `⊥ : WithBot ℝ`。
- `tl.toReal(x)` 把 `.nat` 标量/tile 转成 `.real`。
- `tl.cast(x, tl.float64|tl.float32|tl.float16|tl.bfloat16)` 改变浮点
  dtype index。当前语义里它保留底层 `WithBot ℝ` 值,不建模 rounding。
- `(x).to(tl.float64|tl.float32|tl.float16|tl.bfloat16)` 也支持,作为
  method-style cast spelling。裸 identifier 需要加括号,避免 Lean 把 `x.to`
  解析成一个 hierarchical identifier。
- `tl.bitcast(x, tl.uint32|tl.int32|tl.float32)` 在 compute AST 里表示
  32-bit payload reinterpretation。常量 uint32 bit pattern 可以投影到算法层
  `.nat`、`.int` 或 finite-normal `.real`;runtime bitcast 可以表达,但仍是
  compute-only,会让 `ComputeKernel.toAlgorithm?` 失败。

当前 channel:

- `.real`: 用 `WithBot ℝ` 建模,主要为了表达 `-inf`。
- `.fp32`, `.fp16`, `.bf16`: 显式浮点 dtype channel,当前和 `.real` 一样
  使用 `WithBot ℝ` 数学 carrier。
- `.int`: AST 层 signed mathematical-integer channel,已有算术/比较语义。
  `tl.int8`、`tl.int16`、`tl.int32` 和 `tl.int64` 都映射到这里;bit width、
  overflow 和 signed cast fidelity 还没建模。
- `.nat`: 用于 offset、loop counter、size 和地址算术。
- `.bool`: 由比较产生,用于 mask 和 `tl.where`。

### Tile 构造与 shape 操作

- `tl.arange(n)` 与 `tl.arange(start, end)`。
  双参数形式降为 `start + tl.arange(end - start)`;
  `tl.arange(0, end)` 会折叠成 `tl.arange(end)`。
  两个 bound 都支持数字字面量和 `$(...)` Lean `Nat` meta-expression。
- `tl.full([dims...], value)`。
- `tl.zeros([dims...])`,是 `tl.full([dims...], 0)` 的糖。
- rank-1 输入上的 `e[:, None]` 和 `e[None, :]`。
  它们降到 `Op.expandDim`。
- `tl.expand_dims(e, axis=N)` 与 `tl.expand_dims(e, N)` 在 literal axis
  位置插入 unit axis,支持任意 macro-known rank。
- `tl.trans(e)` 交换最后两个 axis,前面的 axis 作为 batch prefix 保持不变。

### 算术、比较与 broadcast

- 算术: `+`, `-`, `*`, `/`,作用于 numeric channel。
  混合 channel 算术会被 DSL 拒绝。
- 整数 floor division / remainder: `.nat` / `.int` 上的 `//` 与 `%`。
- `.nat` 上的 `tl.cdiv(x, y)`,当前降为 `(x + y - 1) / y`。
- 点态比较: `<`, `<=`, `==`, `>`, `>=`, `!=`,作用于 `.real` 或 `.nat`,
  结果是 `.bool`。
- Bool operator: `.bool` 上的 `tl.logical_and`、`tl.logical_or`、
  `tl.logical_not`,以及 mask 常用写法 `a & b`、`a | b`、`~a`。
- Nat bitwise operator: `.nat` channel 上的 `&`, `|`, `^`, `<<`, `>>`。`.nat`
  上不建模 `~`,因为数学自然数没有固定 bit width;signed/fixed-width
  bitwise 语义留给 fixed-width integer model。
- 双参数 `tl.max(a, b)` 作为 `.real` 上的点态 max。
- `tl.maximum(a, b)` 与 `tl.minimum(a, b)` 是基于 comparison + `tl.where`
  的点态选择 sugar,支持 comparable channel。分支 broadcast 当前限于
  scalar-to-tile lifting,与 `tl.where` 一致。
- Prefix scan: `.real` tile 上的 `tl.cumsum`、`tl.cumprod` 和
  `tl.associative_scan(x, op, axis=N)`。支持的 associative op 是闭合枚举
  `sum`、`prod`、`max`、`min`;不把任意用户函数塞进 AST。
- Index/order op: `.real` tile 上的 `tl.argmax`、`tl.argmin`、`tl.sort`,
  需要静态 `axis=N`。arg tie 返回最小 axis index;sort 沿所选 axis 升序。
- Shape/view op: `tl.reshape`、`tl.view`、`tl.ravel`、`tl.permute`、
  `tl.flip`、`tl.join` 和 projection 形式的 `tl.split(x, 0|1)`。View
  语义是 row-major reshape 或 typed index remap。`tl.split` 暂时暴露成两个
  projection,因为 Lean DSL 还没有 `a, b = tl.split(x)` 这种 tuple destructuring。
- Broadcast 是 ND 的:同维度、scalar-to-tile、或维度 `1` 扩到另一边。
  DSL 通过语法构造 broadcast proof,所以语义等价但写法不同的维度表达式,
  可能仍需要写成一致形式。
- 逐元素选择: `tl.where(cond, a, b)`。
  条件必须是 `.bool`,两个分支 dtype 必须一致,并支持 scalar-to-tile lifting。
  非 scalar 操作数必须已经有相同 surface shape;需要时先用当前支持的 unit-axis
  slicer 把 shape 对齐。

### 一元数学函数

- `tl.exp`
- `tl.log`
- `tl.sigmoid`
- `tl.sqrt`
- `tl.tanh`
- `tl.sin`
- `tl.cos`
- `tl.tan`
- `tl.atan`
- `tl.cosh`
- `tl.sinh`
- `tl.log2`(Real 语义为 `tl.log(x) / log(2)`)
- `tl.exp2`(Real 语义为 `tl.exp(x * log(2))`)
- `tl.abs`

这些都作用于 `.real` channel。`tl.abs(x)` 当前在实数语义里降为
`tl.where(x < 0, 0 - x, x)`。

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
- `ptrs := $(region) + offset; tl.load(ptrs)`
- `tl.load(ptr, mask=mask)`
- `tl.load(ptr, mask=mask, other=other)`
- `tl.load(ptr, other=other, mask=mask)`
- `tl.load(ptr, dtype=tl.float32|tl.float16|tl.bfloat16|tl.int8|tl.int16|tl.int32|tl.int64|tl.uint8|tl.uint16|tl.uint32|tl.uint64)`
- `tl.load(ptr, mask=mask, other=other, dtype=...)`
- `bp := tl.make_block_ptr($(region), base=$(base), shape=[...],
  strides=[...], offsets=[...], block_shape=[...])`
- `bp2 := tl.advance(bp, [deltas...])`
- `tl.load(bp, boundary_check=([axes...] : List Nat), padding_option="zero")`

支持的 store:

- `tl.store($(region), value)`
- `tl.store($(region) + offset, value)`
- `ptrs := $(region) + offset; tl.store(ptrs, value)`
- `tl.store(ptr, value, mask=mask)`
- `tl.store(ptr, value, dtype=...)`,其中 `dtype` 必须和 value dtype 一致。
  不写 `dtype=` 时,store 从 `value` 推断 dtype。
- `tl.store(bp, value, boundary_check=([axes...] : List Nat))`

未知 kwarg 会报错。对 block pointer 来说,`boundary_check` 只支持在
block-pointer `tl.load` / `tl.store` 上使用,不能和 `mask` / `other` 混用。
当前唯一建模的 `padding_option` 是 `"zero"`。

masked load 语义:

- `mask=true` 时读内存。
- `mask=false` 且提供 `other` 时返回 `other`。
- `mask=false` 且省略 `other` 时,Triton 语义是 undefined;VeriTile 用
  `BlockState.undef` 建模。

masked store 语义:

- `mask=true` 时写入该 lane。
- `mask=false` 时保持原内存不变。

## Memory model

VeriTile 用一个窄版 first-class pointer value 建模 pointer:

```lean
RegionName × Nat
```

runtime memory model 在 cell 边界是 typed:

```lean
RegionName → Nat → MemCell
```

proof-facing 的 Real 兼容 API 仍然是 `BlockState.readMem` /
`BlockState.writeMem`,所以现有数学正确性 theorem 继续暴露 Real-valued
observation surface。换句话说,旧的 theorem-facing `RegionName → Nat → ℝ`
view 仍然作为 API 层存在,但它不再是 runtime storage representation。

Region dtype contract 单独放在 `VeriTile.Triton.Memory.Typing`:

```lean
RegionTyping := RegionName → TileDType
Kernel.RespectsRegionTyping Γ k
```

这是轻量静态层。它检查 named-region load/store 是否使用 `Γ` 声明的 dtype,
执行语义则写入 typed `MemCell`。对于 pointer-valued load/store,静态可见的
`$(region)` pointer base 会按 `Γ` 检查;pointer register 在这一层先视为
dynamic/external value。

可选 executable checker 在不替代 proof-side contract 的前提下,补诊断和 pointer
provenance tracking:

```lean
RegionEnv := RegionName → Option TileDType
Kernel.check Γ k
Kernel.checkStrict Γ k
```

`Kernel.check` 是 lax mode,未声明 region 会跳过;`checkStrict` 会把未声明 region
报错。checker 跟踪 register dtype/shape 一致性、pointer / block-pointer
provenance、直接和 pointer-derived load/store dtype mismatch,以及基本 block-pointer
metadata sanity。它刻意不证明 bounds、alias、launch coverage、page ownership 或
IEEE/hardware dtype fidelity。

pointer value 可以 inline 使用、赋值和复用:

```lean
ptrs := $(xReg) + offs
x := tl.load(ptrs)
ptrs2 := ptrs + stride
```

这个模型故意比 CUDA/Triton pointer 窄:目前支持从 `RegionName` 创建 pointer base、
pointer 加 `.nat` offset、以及通过 pointer-valued register 做 load/store。它也把
block pointer 建模为 first-class `.blockPtr` tile value,携带 base region、base
offset、parent shape、block shape、strides 和 logical offsets。block-pointer
load/store 会逐 lane 计算地址;checked 维度越界的 load lane 返回 zero,checked 维度
越界的 store lane 不写内存。pointer cast、pointer comparison、硬件/TMA block-pointer
行为、typed address space 还没有建模。

Offset 是显式 `.nat` 表达式。高维 tensor 通过用户写出的 strided offset 公式表示,
例如:

```lean
b * stride_b + h * stride_h + i * stride_s + d * stride_d
```

公开 theorem surface 用 `TensorView.loaded` / `TensorView.observe` 把这些 offset
公式连接到数学 tensor slice。证明内部仍可使用更底层的 `InputAt` escape hatch
表示任意 offset map,再把结果封装成 `TensorView`。aliasing 通过选择相同或不同的
`RegionName` 表示;这些 named region 之外的通用 pointer alias analysis 还没有建模。

## 浮点模型

算术目前是 `ℝ` abstraction:

- real data 建模为 `WithBot ℝ`。
- `-inf` 精确建模为 `⊥`。
- 语义里定义了 `exp ⊥ = 0` 和 `sigmoid ⊥ = 0`。
- store 会把 `⊥` 用默认值降回内存;well-formed kernel 不应该 store `⊥`。

含义是:当前 theorem 证明的是实数数学正确性,不是 IEEE-754 bit-level 等价。
rounding、NaN、signed zero、overflow、underflow、denormal、exception flag、
硬件 dot precision、fast-math rewrite 都未建模。

core AST 现在使用统一的 dtype-indexed memory form:

```lean
Op.load    : TileDType → MemAccess shape → MaskOpt dtype shape → Op ...
Stmt.store : TileDType → MemAccess shape → Op ... → MaskOpt dtype shape → Stmt
```

公开 DSL 的 `tl.load` 默认仍是 `.real`,但支持
`dtype=tl.float32|tl.float16|tl.bfloat16|tl.int*/tl.uint*`,
用于生成 typed memory node。所有 `tl.uint*` spelling 映射到 VeriTile 的
`.nat` channel,用于非负 index/block-table value。所有 `tl.int*` spelling
映射到 VeriTile 的 `.int` channel,这是数学 signed-integer abstraction,没有
bit-width 或 overflow 语义。`tl.store` 从写入的 value 推断 dtype,也支持可选的、
必须匹配 value dtype 的 `dtype=` surface spelling。

Float theorem policy: 算法正确性 / refinement theorem 证明在擦除后的 `.real`
kernel 上。面向 DSL 的 compute surface 是 `ComputeKernel`;
`ComputeKernel.ComputeCorrect` 和 `ComputeKernel.ComputeRefine` 通过
`toAlgorithm?` 把可擦除的 compute kernel 投影到算法层。已有 float-facing theorem
仍可使用 `k.eraseDType = realK` 这类 erasure 等式,为带 dtype 标注的 algorithm
kernel 复用 Real proof。数值 compute correctness/refinement 仍由观测层 predicate
和 differential tests 单独支撑,不是 IEEE-754 proof。这些定义放在
`VeriTile.Triton.Float`;compute / algorithm split 和 bitcast policy 见
`documents/EraseDType.md`。

## Operator / syntax 覆盖 checklist

这个表是 GitHub issue #15 当前的 operator-coverage contract。`Supported`
表示已有 Lean AST 构造或 DSL lowering、操作语义,并且覆盖当前 examples 需要的证明
surface。`Limited` 表示 VeriTile 有意只支持 Triton 特性的窄子集。`Gap` 表示使用该
特性的 kernel 目前不在语义契约内。

| Area | Status | Coverage |
| --- | --- | --- |
| scalar/tile 常量 | Supported | 实数字面量、`$(n)`、`$ℝ(x)`、`-inf`、register ref |
| program id | Limited | literal 或 antiquoted `Nat` axis 的 `tl.program_id(axis)`;可通过 `GridIndex` / `Kernel.ForAllPrograms` 做 ND grid 量化,但没有 launch executor |
| loop | Supported | bounded `tl.for`;`tl.static_range` alias 降到同一个 loop AST |
| conditional | Limited | 只有 `tl.if cond { ... }`;没有 `else`、`break`、`continue` |
| arithmetic | Supported | 同 channel numeric 上的 `+`, `-`, `*`, `/`;integer `//` / `%`;`.nat` `tl.cdiv`;pointer offset 支持 `ptr + nat` |
| comparison | Supported | `.real` 或 `.nat` 上的 `<`, `<=`, `==`, `>`, `>=`, `!=` |
| bool op | Supported | `tl.logical_and`、`tl.logical_or`、`tl.logical_not`,以及 `&`、`|`、`~` mask spelling |
| Nat bitwise op | Limited | `.nat` 上的 `&`, `|`, `^`, `<<`, `>>`;暂不支持 `.nat ~` 和 signed fixed-width bitwise 语义 |
| pointwise select | Supported | `tl.where(cond, a, b)`,支持 scalar lifting,非 scalar shape 需一致 |
| unary math | Supported | `tl.exp`, `tl.exp2`, `tl.log`, `tl.log2`, `tl.sigmoid`, `tl.sqrt`, `tl.tanh`, `tl.sin`, `tl.cos`, `tl.tan`, `tl.atan`, `tl.cosh`, `tl.sinh` |
| reduction | Supported | `.real` tile 上的 `tl.sum`, `tl.max`,可带 `axis` / `keep_dims` |
| prefix scan | Limited | `.real` tile 上的 `tl.cumsum`, `tl.cumprod`, `tl.associative_scan(x, sum/prod/max/min, axis=N)`;不支持 arbitrary combine function |
| index/order op | Limited | `.real` tile 上的 `tl.argmax`, `tl.argmin`, `tl.sort`,需要静态 `axis=N`;arg tie 返回最小 axis index,sort 升序 |
| broadcast | Supported | ND same-dim、scalar-to-tile、dimension-`1` expansion |
| shape construction | Limited | `tl.arange`, `tl.full`, `tl.zeros`,rank-1 `[:, None]` / `[None, :]`,literal-axis `tl.expand_dims` |
| shape/view op | Limited | `tl.reshape`, `tl.view`, `tl.ravel`, `tl.permute`, `tl.flip`, `tl.join`,projection-form `tl.split(x, 0|1)` |
| transpose | Supported | `tl.trans(e)` 交换最后两个 axis;任意静态 permutation 用 `tl.permute(e, [axes])` |
| matrix multiply | Supported | 数学 `ℝ` 模型下的 `tl.dot(a, b)` 和 `tl.dot(a, b, acc)` |
| bitcast | Limited | 32-bit compute payload `tl.uint32` / `tl.int32` / `tl.float32`;常量 uint32 pattern 可投影到算法值,runtime bitcast 可表达但 compute-only |
| load | Limited | pointer-expression load,可带 `mask` / `other` / float/int*/uint* spelling-only integer `dtype=`;block-pointer load 支持 `boundary_check` 和 `padding_option="zero"` |
| store | Limited | pointer-expression store,可带 `mask`;dtype 从 value 推断,也可写匹配的 `dtype=`;block-pointer store 支持 `boundary_check` |
| memory bounds safety | Limited | `Kernel.MemorySafe` / `ComputeKernel.MemorySafe` 证明 direct、pointer、block-pointer memory op 的 active-lane region bounds;不包含 alias、race、frame 或 permission model |
| memory frame contract | Limited | predicate-level `WriteFootprint`、`BlockState.WriteWithin`、`Stmt.StoreAddressesWithin` 和 `Kernel.ExecFrame`,用于 single-program write-frame reasoning;`tileImage` / `activeTileImage` helper 可提取 direct、masked、checked block-pointer store footprint, unrelated-region preservation helper 覆盖常见 frame readback 目标 |
| disjoint grid composition | Limited | `Kernel.GridFrames`、`GridWritesDisjoint` 和 `mergeFrames` 能合并 pairwise-disjoint write footprint 的 explicit per-program frame;没有 overlapping write 或 scheduling semantics |
| tensor view | Supported | theorem surface 的 strided `TensorView.loaded` / `TensorView.observe` wrapper |
| integer memory | Limited | typed cell 加 typed load/store 支持 Nat/index 和数学 signed-Int HBM value;还没有更完整的 signed/unsigned width lattice |
| randomness | Gap | 还没有 `tl.rand` 或 RNG state model (#41) |
| indirection | Limited | typed index load 可以参与 pointer arithmetic,表达 gather / paged-KV 风格 data-dependent address (#42);还没有 alias/bounds/page-ownership proof layer |
| block pointer | Limited | `tl.make_block_ptr`、`tl.advance`、带 checked-axis zero padding / store skip 的 block-pointer load/store;没有硬件/TMA 行为 |
| atomic / async / barrier | Limited | `tl.atomic_add` 已有 AlgKernel `Stmt.atomicAdd` marker、单程序顺序语义、trace payload 词汇和 Real grid-merge sum theorem;其他 `tl.atomic_*`、`tl.async_copy`、`tl.async_wait`、`tl.debug_barrier` 目前只会 lowering 到 compute-facing failure marker;async/TMA discipline 仍只是文档 contract,没有实现;还没有完整 scheduler、可执行 barrier、可执行 async copy 语义、TMA AST 或 IEEE atomic 语义 (#12/#67/#68/#69/#71/#72/#76) |
| floating-point fidelity | Gap | 只有 real-valued model;没有 IEEE-754 或 mixed-precision hardware semantics (#11) |

## 表达力矩阵

这个矩阵回答的是另一个问题:如果用户从一个真实 Triton kernel 出发,当前 Lean DSL
会卡在哪一类 gap 上?

| Pattern | Status | Gap type | Practical impact |
| --- | --- | --- | --- |
| dense elementwise kernel | Mostly supported | proof/theorem surface | 点态算术、mask、cast、pointer value 和 TensorView observation 都有;更强 dtype/memory claim 仍走 real abstraction。 |
| softmax / reduction / LayerNorm / Welford | Supported for current examples | proof engineering | reduction、loop、mask、TensorView wrapper 都已有;新 kernel 主要是 invariant 和 theorem packaging。 |
| FlashAttention-style dense tiled kernel | Supported for FA-1 forward subset | proof engineering + limited semantics | dot、transpose、causal/boundary mask、D-tail、4D view 已覆盖;async/shared-memory/hardware dot fidelity 仍不在当前范围内。 |
| first-class pointer expression | Limited | surface + lightweight semantics | 支持 `ptrs := $(r) + offs`、pointer register、pointer load/store、pointer offset update;没有 pointer cast/comparison/alias analysis。 |
| block pointer / `boundary_check` | Limited | surface + sequential semantics | 支持 `tl.make_block_ptr`、`tl.advance`、zero-padded checked load、checked store-skip;没有 `order`、非 zero padding、TMA 或硬件行为。 |
| typed floating memory | Limited | semantic abstraction | `dtype=tl.float32/fp16/bf16` 会生成 typed floating node,算法证明擦除到 real;没有 IEEE rounding。 |
| integer / bool tensor memory | Limited | dtype coverage | typed cell 加 typed load/store 支持 Nat/index 和数学 signed-Int HBM value;还没有完整 Triton integer-width lattice。 |
| indirect / gather addressing | Limited | surface + view semantics (#42) | typed index tensor load 可以驱动 pointer arithmetic 和普通 masked load;alias analysis、bounds proof、page ownership、paged FA-1 等价还没建模。 |
| active-lane memory bounds | Limited | Lean proof predicate (#48) | `Kernel.MemorySafe` 按 `RegionBounds` 检查 direct region offset、dynamic pointer address、mask activeness 和 `boundary_check` block-pointer lane;没有 race freedom、frame theorem 或 permission accounting。 |
| single-program write footprint/frame | Limited | predicate-level frame contract + extraction helpers (#60/#61) | `WriteFootprint := (RegionName × Nat) → Prop` 和 `BlockState.WriteWithin` 表达一次 execution 只修改 supplied footprint 内的 cell;`tileImage` / `activeTileImage` helper 可提取 direct、masked、checked block-pointer store footprint。 |
| RNG / dropout | Gap | state/probabilistic semantics (#41) | 阻塞 faithful dropout 和随机 kernel。 |
| atomics / async / shared memory / barriers | Limited | atomic-add slice + concurrency boundary (#12) | `tl.atomic_add` 已有 proof-facing marker 和 Real trace/grid-sum theorem;不支持的 atomic family 成员和 async/barrier surface 会显式 projection failure;async/TMA、shared memory、barrier、完整 scheduling、IEEE atomic 行为仍是 gap。 |
| whole-grid launch semantics | Limited | ND grid theorem surface + disjoint/atomic merge (#5/#49/#12) | `GridIndex`、`BlockState.withGridIndex`、`Kernel.ForAllPrograms`、`ForAllProgramsSome` 支持 per-program correctness 量化;`Kernel.mergeFrames` 处理 pairwise-disjoint footprint,`Kernel.mergeFramesWithAtomic` 处理选定 Real atomic-add contribution;没有完整 race/scheduler/interleaved executor。 |
| Python/Triton source ingestion | Gap | front-end/lifter (#10) | 用户必须写 Lean `triton { ... }`;decorator、Python-side constexpr execution、一般 Python control flow 还不能解析。 |
| type checking / pointer provenance | Limited | optional checker (#46) | `Kernel.check` / `checkStrict` 跟踪 register dtype/shape、pointer 和 block-pointer provenance、dtype mismatch、基本 block-pointer metadata;不证明 bounds、alias、launch 或 page ownership。 |

近期表达力优先级应先消除 core semantic gap,再做完整 Python lifter:RNG/dropout
(#41)、atomics/async/concurrency (#12)、bounds/memory-safety assumption (#48)。
如果 kernel 的操作本身还不可表达,lifter 提早做收益不大。

## 尚不支持或尚未真实建模

- 完整 IEEE-754 浮点语义。
- block-pointer 硬件/TMA 行为,以及 `"zero"` 之外的 padding option。
- `RegionName × Nat` pointer value 之外的完整 CUDA/Triton pointer 语义。
- named-region equality 之外的通用 pointer alias analysis。
- 完整 Python/Triton JIT 语义、decorator、meta-parameter execution,
  以及 `triton { ... }` 块外的 Python control flow。
- atomic operations。
- async copy / TMA / shared-memory staging。
- barrier、跨 program 或跨 warp synchronization。
- 完整 grid launch execution。`GridIndex` / `Kernel.ForAllPrograms` 提供
  ND grid 上每个 program instance 的 theorem surface,但 VeriTile 仍不建模
  sequential/concurrent launch executor、global memory merge、overlapping
  writes、race、atomic 或 scheduling。
- cache/performance hint,例如 `cache_modifier`, `eviction_policy`, `volatile`,
  `is_volatile`。
- 高 rank bracket slicing。目前只支持 rank-1 的 `[:, None]` / `[None, :]`;
  显式插入 unit axis 请用 `tl.expand_dims(e, axis=N)`。
- integer width、overflow、signedness 和 signed bitwise 行为。`.nat`
  channel 是数学 `Nat`。

## 文档自动生成

目前还没有自动生成这个 subset 文档的工具。未来可以做一个脚本,从 DSL syntax /
`Op` constructors 抽取 raw table,再和本文档比对,用来抓
"实现了但没写文档" 或 "文档写了但 DSL 已不接受" 的漂移。

现阶段本文档仍应人工维护,因为 artifact 的关键声明是语义层面的,不只是语法列表;
上面的 gap 需要人来判断。
