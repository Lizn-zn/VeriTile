# TritonBench-G 翻译审查标准

[English](review_criteria.md) | **中文**

审 `bench/tritonbench_g/<kernel>/<Kernel>.lean` 对照上游 `<kernel>.py` 时,
什么 gap 容忍、什么必须修。

## Gold standard

`bench/tritonbench_g/add_example/AddExample.lean` 是参照基准。

合格翻译有两个组成部分:

1. `def kernel ... : ComputeKernel := triton { ... }` body 是上游 `.py`
   Triton kernel 函数体的**一行不多一行不少**的 1:1 转写 —— 不加 statement、
   不删 statement、不重排、不合并。
2. 正确性用 `ComputeCorrect.Realizes` 标准写法。

为什么这条线高:VeriTile 对用户面的承诺是"把 `.py` kernel 粘到 Lean 里,
证的是**这个** kernel"。任何隐式简化/合并/补足都让翻译变成"VeriTile 的
kernel",而不是"你的 kernel"。

## 可容忍 gap(⚠️ MINOR — 不 block)

### A. Lean 语法强制要求(白名单,穷举)

只接受以下两种 Lean-syntax 转换:

- `$(x)` 包 Lean 参数 / 编译期常量。包含结构性的
  `BLOCK_SIZE: tl.constexpr` → `Nat` 参数改写,使用处写 `$(BLOCK_SIZE)`。
- `if HAS_X { ... }` DSL gate 替代 Python `if HAS_X:`(`tl.constexpr`
  分支)。

**白名单之外的转换全部不接受**,落入下面的 DEVIATES(§1–§6)。具体地,
下列转换**不**算机械改写:

- `.py` 没标 `dtype=...` 时给 load 加显式 dtype 标注。
- 用 `tl.max(x, axis=1, keep_dims=true)` 替代 `tl.max(x, axis=1)[:, None]`。
- 用 `1 / tl.sqrt(x)` 替代 `tl.math.rsqrt(x)`。
- 用 `(0.0).to(tl.float32)` 显式 cast 解决类型推断。
- 指针 mutation `X += offset` 改写成显式 `base + offset`。
- `+=` 改写成 `= + ...`。
- 在所谓 "Real-first policy" 下擦除 `.to(tl.float32)`。

DSL 表达不出 `.py` 写法,那是 **DSL gap,要开 issue 修 DSL**,不是给翻译
重写 kernel body 的绿灯。翻译必须跟 `.py` 完全对齐;允许的机械改动只限于
上面这个白名单。

### B. 已文档化的 partial slice / specialization

只在 kernel preamble doc-comment **显式说明** scope 时才允许。preamble
不能声称跟上游完整 kernel 等价。

- "proof-oriented one-block slice" / "single-tile slice" /
  "single-iteration slice":上游有循环,Lean 只证一次迭代。
- 单 `tl.constexpr` 分支特化(例:`IS_RMS_NORM=true`、`HAS_BIAS=false`、
  `LOG=false`)。
- 单输出 channel(例:`FifthOrderSphHarmonics` 只覆盖 `Y00`)。
- 没用的参数省略(例:1D path 里的 `*_row_stride`)。

## 必须修 gap(❌ DEVIATES)

### 1. 加了 `.py` 没有的 statement

任何额外 `tl.where`、额外 cast、额外 mask 处理行,只要 `.py` 没写,就违反
"一行不多一行不少"。

例:`RmsnormFused.lean` 多了独立 `x_masked = tl.where(cols < N, x_for_var,
0.0)`,而 `.py` 是用 `tl.load(..., other=0.0)` 处理 padding。这条 `tl.where`
是多出来的 statement。

### 2. 加了 `.py` 没有的算术结构

给地址/索引加 stride 系数、broadcast 维度、scale 因子,只要 `.py` 没写,即使
"在某些配置下看起来更对"也是真语义偏差。

例:`FastRmsLayernorm.lean` 写 `tl.load(W + col_offsets * W_row_stride, ...)`,
`.py` 写 `tl.load(W + col_offsets, ...)`。`W` 有非平凡 stride 时两者计算
结果不同。

### 3. reduction 循环折叠成单条 `tl.sum` / `tl.max`

上游 `for off in range(0, N, BLOCK_N): var += xf*xf`,Lean 写
`var = tl.sum(x*x, axis=0)`。即使数学等价,kernel body 的结构变了,proof
说的不再是这个 kernel。

两条可接受的修法:
- 补回 `for` 循环(faithful;proof 也得改)。
- preamble 明确写"特化为 single-tile,把 N-block accumulation 循环折叠
  为单条 `tl.sum`,要求 `BLOCK ≥ N`"(承认特化)。

### 4. compute statement 重排

即使数据流等价,丢掉 1:1 line 对应就不算 faithful。

### 5. 未文档化的 `tl.*` 调用省略

省略链式 cast、masked-load 退化为 bare load 等,没有 preamble 说明。

### 6. 控制流改动

把 Python 多分支 `if` 折成 `tl.where`、`for` 展开、`while` 改 `for`,
没有 preamble 说明。

## 审计输出格式

每个文件给三档之一:

- `✓ FAITHFUL` —— line-by-line 1:1。
- `⚠️ MINOR` —— 落入某个可容忍类别。如果是已文档化的 slice / specialization,
  引用 preamble 行号。
- `❌ DEVIATES` —— 落入某个必须修类别。给出 `.py` 行号、`.lean` 行号、具体
  偏差描述。

## 大批量审计的工作流

横扫多个 kernel 时,每个 read-only audit agent 一批 ~15 个文件是合适的颗粒度。
每个 agent 读每对文件,套这个标准,每个文件输出一个 verdict block。
