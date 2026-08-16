# attention_llama

- 源文件：`attention_llama.py`（上游 `data/TritonBench_G_v1/attention_llama.py`）
- 语料：TritonBench-G v1
- 规模：173 行，1 个 `@triton.jit` kernel
- 状态：**已移植** — `AttentionLlama.lean`，主定理
  `attention_llama_fwd_closed_form_correct`（非因果）与
  `attention_llama_fwd_causal_closed_form_correct`（因果），均为
  `exec` 级、维度一般、0 `sorry`。

目标 JIT：**`_fwd_kernel`** — 文件唯一的 `@triton.jit` kernel。
FlashAttention-1 式流式 softmax（自然 `exp`、循环内归一化
`l_rcp = 1/l_curr; p *= l_rcp; acc *= (l_prev*l_rcp)`）、平指针掩码
载入、循环内指针推进（`k_ptrs += BLOCK_N*stride_kn` 等）、结尾单个
掩码 `Out` 店。启动器 `triton_fa`：
`grid = (cdiv(m_size, BLOCK), head_size·batch)`，
`BLOCK_M = BLOCK_N = 64`，`BLOCK_DMODEL = Lk ∈ {16,32,64,128}`。

**参数别名（H 是 QUERY 长度）**：启动器传 `N_HEAD := head_size`、
`H := m_size`、`N_CTX := n_size` —— `offs_m < H` 的 q 载入/Out 店掩码
是查询行界，H 并非头数。批/头分解为 `pid₁ // N_HEAD` 与
`pid₁ % N_HEAD`。

**孪生 surface**（`IS_CAUSAL` constexpr；triton_matmul 先例）：
非因果臂忠实保留 `block_n_end = N_CTX`；因果臂保留两条
`block_n_end` 赋值加因果 `tl.where(offs_m ≥ block_n_offs +
start_position, qk, -inf)`。循环上界在两臂都是运行时寄存器
（`forRangeDyn`）。其余语句逐字节一致（含循环后
`start_m`/`offs_m`/`offs_d` 重物化）。Out 店是 `.real`：fp32 `acc`
未经 cast 直接写出（`p.to(Q.dtype.element_ty)` 在 ℝ 上擦除），两条
头条均为精确 ℝ 陈述。

**头条**（均为 `Realizes_without_Rounding`，写图 = kernel 自己的掩码
店；期望值从输入内存经 kernel 自己的偏移表达式读出；Q 瓦片带 kernel
自己的 `row < H` 载入门，在活跃行恒真）：

- 非因果：每个活跃 Out 单元 = `attentionReal`（N_CTX 个键的完整自然
  exp softmax，缩放 `sm_scale`）。边条件：
  `N_CTX = BLOCK_N · numKVBlocks`、`0 < numKVBlocks`、输出偏移单射、
  `hundef`。
- 因果：每个活跃 Out 单元 = `attentionRealCausalBlock`，
  `qStart = pid₀·BLOCK_M − start_position`，键域为
  `BLOCK_N·numCausalBlocks`；行 `pid₀·BLOCK_M + i` 的可见键恰为
  `{j < span : j + start_position ≤ 行}`。边条件：
  `hspanEq : (pid₀+1)·BLOCK_N + start_position = BLOCK_N·numCausalBlocks`
  （即 `BLOCK_N ∣ start_position` 的等式形）、`span ≤ N_CTX`、
  `hsp : start_position ≤ pid₀·BLOCK_M`（逐块全一般；`pid₀ = 0` 时
  迫使 `start_position = 0`，正是全部上游测试）、单射、`hundef`。
  证明**不需要** `BLOCK_M = BLOCK_N`（span 整除后无行程数幽灵）。

**幽灵道/整除怪癖**：`offs_n < N_CTX` 那条 where 用的是 `offs_n`
而非 `block_n_offs`，在 `BLOCK_N ≤ N_CTX` 下是恒真幌子（忠实保留、
证明消掉）；非因果不整除时尾块幽灵道载 `k=0` 贡献 `exp(0−m)` 质量
（上游怪癖，故有整除条件）；因果向：谓词是
`行 ≥ 键 + start_position`，即 `start_position` **收缩**可见域，
不是前缀缓存语义。

**位置参数 `other`**：Python 的 K/V 载入是位置参数
`tl.load(ptrs, mask, 0.)`；DSL 文法只有 `other=` 关键字形，写成
关键字会破坏 audit 的 kwarg 序列扫描。移植将其拼成纯掩码载入，
死道读 undef 载体 —— 由头条假设 `hundef` 钉为 0（与 Python 提供的
值相同）。q 载入的 `other=0.0` 在 Python 里就是关键字，保留为
`MaskOpt.maskOther`。

Translation-surface blocker（已登记 `proof_blockers.md`）：
`USE_FP8 = True` 臂（`k.to(tl.float8e5, bitcast=True)` 再
`.to(tl.float16)`）是 int8 字节 → fp8 位重解释 —— 既有 ℝ 模型极限
家族（`llama_ff_triton` 先例）。本移植只建模 `USE_FP8 = False`
（上游测例 1/2）；测例 3/4（int8 K/V）在建模面之外。`IS_CAUSAL`
constexpr 拆为孪生 surface。宿主启动/grid/num_warps 为可信边界。
