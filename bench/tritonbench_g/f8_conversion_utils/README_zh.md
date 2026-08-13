# f8_conversion_utils

- 源文件:`f8_conversion_utils.py`(上游 `data/TritonBench_G_v1/f8_conversion_utils.py`)
- Corpus:TritonBench-G v1
- 规模:68 行,2 个 `@triton.jit` kernel
- 状态:**已移植** —— `F8ConversionUtils.lean`,主定理
  `f16_to_f8_io_correctness` + `f8_to_f16_io_correctness`
  (`⊨[R, ·]` io 面、维度一般、0 `sorry`)。

上游实现 fp8 ↔ fp16 缓冲区转换:两个 kernel 都是掩码 1-D 拷贝
(`offs = pid·BLOCK_SIZE + arange`,`mask = offs < N`),fp8 性完全住在宿主的
`triton.reinterpret(x, tl.float8e5)` 指针里——店在目标元素类型上**隐式**
转换。`kernel_f8_to_f16` 连店两次(上游原文逐字重复)。

**fp8 dtype 通道的首个消费者**(`.f8e4`/`.f8e5` 随本港加入
`TileDType`/`FloatDType`):f16→f8 头条是语料库首个 fp8 边界量化——

> `f16ToF8IO Y X N BLOCK_SIZE ⊨[R, .f8e5] fun xs i => xs i`

对**每一个**舍入模型 `R`:每条活跃输出 lane 读回 `.f8e5` 类型 cell,值为
`R.round .f8e5 (xs j)` —— 输入值在 e5m2 网格上量化恰好一次(cast 位点 +
类型化店位点经 `R.round_idem` 合并)。反方向在 `.fp16` 上陈述同一契约
(重复店幂等)。除 io 皮固有假设(窗口界 + 精确 ℝ 输入 lane)外零侧条件。

Translation-surface blocker(登记于 `proof_blockers.md`):上游店没有
cast 文本——Triton 在类型化指针处隐式转换——而 DSL 按**值**给店定型,
故隐式转换显式拼作 `(x).to(tl.float16)` / `(x).to(tl.float8e5)`。Python
`BLOCK_SIZE: tl.constexpr` 化为 Lean `Nat` 参数。宿主启动(网格、
`reinterpret`、`numel` 簿记)是可信边界;f8 侧*输入*缓冲区的值落在 f8
网格上是宿主管线事实,定理不需要(输入按精确 ℝ 消费)。
