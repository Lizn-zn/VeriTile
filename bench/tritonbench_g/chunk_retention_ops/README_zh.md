# chunk_retention_ops

- 源文件:`chunk_retention_ops.py`(上游 `data/TritonBench_G_v1/chunk_retention_ops.py`)
- Corpus:TritonBench-G v1
- 规模:363 行,4 个 `@triton.jit` kernel
- 状态:**已移植**(两个状态递推 kernel)—— `ChunkRetentionOps.lean`,主定理
  `cro_fwd_h_exec_genuine` 与 `cro_bwd_dh_exec_genuine`(`exec` 级、维度一般、
  0 `sorry`)。

上游文件是 `chunk_retention` 的语料孪生,但关键处**不是**逐字节重复:

- `fwd_kernel_h` 与已移植的 `chunk_retention` 逐语句同一(仅
  `initial_state`/`final_state` 拼作 `h0`/`ht`),整套正向证明栈 —— 衰减前奏、
  毛边末 chunk 边界门、递归 `croState`、逐 chunk 店历史 —— 原样搬运。
- `bwd_kernel_dh` 是**修好的**兄弟:`chunk_retention` 的反向只在方形可编译、
  单店、带死载入;这里是干净的维度一般店历史 kernel —— 降序载运
  `D_next = d_b·D + (scale·q)ᵀ·(do ⊙ d_i)` 逐 chunk 循环内存店,`d_b`/`d_i`
  固定(反向无毛边重绑)。其规约 `croDhCarry` 与正向同为递归 ⇒ 衰减推进
  定义上成立。

头条:`h[·,·,t]` = 衰减正向的 chunk 前状态(四种门配置、毛边与整除同覆盖);
`dh[·,·,t]` = 降序载运 `croDhCarry (NT-1-t)` 的 chunk 前值。全部符号化;
布局副条件为标准三条。

衰减前奏的隐式 int→float 提升拼作 `tl.toReal(...)` —— 已在
`proof_blockers.md` 注册的 `Translation-surface blocker:`。正向毛边 `d_i`
尾 lane 的 nat 截断在序言披露(经边界检查的 `v` 不可观察)。降序循环用升序
换元拼写(`chunk_linear_attn` 拼法,零库改动)。
