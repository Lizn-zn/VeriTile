# VeriTile — 项目计划

[English](PLAN.md) | **中文**

VeriTile 是一个长期项目:把真正的 Triton kernel 验证带进 Lean 4。不是 toy,
不是为某个 deadline 精心挑选的几个定理 —— 目标是让 Triton 工程师能把自己日常
ship 的 production `.py` kernel(forward + backward + 并发原语 + production-scale
layout / masking / autograd 路径)经过最小修改放进 VeriTile 的嵌入式 DSL,得到机器
可检查的正确性保证。

本文档替代原 2026-04-26 brainstorm 的 "PLDI 7-10 月 program plan"(2026-05-05
重定向,见 §决策日志 8-9)。

## 项目北极星

Triton verification 真正落地需要四个轴同时推进:

1. **DSL 表面接近真 Triton** —— 接受 production `.py` kernel 实际使用的 op、
   控制流、memory primitive、concurrency primitive,而不只是 attention forward
   所需的最小子集
2. **算法层证明覆盖 production kernel** —— 完整 FA-1 forward + backward,FA-2,
   fused-norm 全家族,grouped GEMM,Mamba SSM,RoPE,GQA / MQA / MLA 等,而不是
   8 个 hand-picked 定理
3. **Triton 用户友好度** —— `.py` paste-in 改动量从"重写嵌入"逐步降到"标 spec /
   标量化点";最终目标是几乎零改动
4. **横向 infra 真实** —— Algorithm/Compute 双层、Memory frame、Concurrency
   trace/refinement、ND general 框架(无 ad-hoc 维度 shortcut)全部 production-grade

这是开放路线图,**不是固定 N 个定理交付即结束**。下面的 §状态、§进行中、§路线图按
这四个轴展开。实时 issue 路线图在 GitHub issue #91;本文档记录架构、状态和决策日志。

## 验证架构(永久,自 v0.2 起锁定)

VeriTile 的验证拆成**两个独立互补的层**:

```
Algorithm 层(Lean 证明)             Compute-gap 层(外部测试)
────────────────────────             ──────────────────────
ProjectedCorrect ck spec             GapPolicy.require contract
  := ck.toAlgorithm? = ok ∧            := Python 检查 ComputeKernel
     Kernel.Correct ak spec               行为与投影后的 AlgKernel
                                          行为在 contract tolerance 内匹配。
  Real / Int / Nat 语义。              外部 pipeline,不是 Lean 定理。
  无 IEEE rounding,无 NaN,
  无 fp exception flag。
```

两层**不通过任何内部 Lean 定理连接**。compute-facing syntax 与投影后的 algorithm
syntax 之间的数值 gap 是经验性的(测试),**这是 design 决定**。一个
VeriTile-certified compute theorem 意味着:
> 投影后的算法结构由 Lean 证明正确,**并且** 当使用 `gap := .require contract`
> 时,该 contract 命名的 compute-to-algorithm gap 已由外部验证。

完整 certification 需要两层都成立。

### 永久性

IEEE-754 形式化(NaN 传播、denormal、rounding mode、硬件 dot precision、
fast-math、exception flag)**永久不在 Lean 证明范围内**。这不是延期 ——
不存在计划中的 bridge 定理把 compute 执行连到 Lean 内部的算法执行上。
需要的 compute gap 现在是、未来也是 test-backed。

### Algorithm 层细节

1. DSL 接受用户面向的 float kernel,`.fp32`/`.fp16`/`.bf16` dtype 标注照常。
   bit-level 常量(例如哨兵 `1.0` 写成 `tl.bitcast(0x3f800000, tl.float32)`)
   通过单一 computable decoder 接受;只支持 finite-normal binary32 模式。
2. 算法正确性在 `Kernel.eraseDType`(fp* → real,intN → int,uintN → nat)
   之后的算法层 kernel 上证明。
3. 形式 Lean 层使用 `ComputeKernel.ComputeCorrect` /
   `ComputeKernel.ComputeRefine`,默认 `gap := .ignore`;当需要记录外部 compute-gap
   检查时使用 `gap := .require contract`。现有 `Kernel.Correct` 证明在算法侧
   kernel 上继续有效。

### Compute 层细节(测试)

1. `ComputeKernel` 忠实保留用户写的 compute-facing AST(包括 `ComputeOp.bitcast`、
   fp/int 宽度拼写等)。
2. Python gap checker(独立 workstream)导出或镜像 `ComputeKernel` 及其投影后的
   `AlgKernel`,生成样本输入,校验 exact equality 或
   `|compute_output - algorithm_output| < epsilon` 等 contract relation。
3. epsilon 策略 per-op / per-kernel,写入 `ComputeGapContract`;Lean 只记录
   `ExternalChecked contract`。

### 用户视角

- 算法结构有 bug:`ComputeKernel.ComputeCorrect` /
  `ComputeKernel.ComputeRefine` 抓(投影到内部 `ProjectedCorrect` /
  `ProjectedRefine` 后 Lean 证明 fail)。
- IEEE-specific edge case 有 bug(NaN、overflow、denormal 处理等):
  required compute-gap checker 抓(测试 fail)。**这一层 by design 没有 Lean 证明义务**。
- "Full compute-facing certificate" 意味着投影后的 Lean 证明成立,并且当 theorem
  选择 `gap := .require contract` 时,所需 gap contract 已验证。

Real 算法正确性到 floating computation 的 trusted bridge **是 external gap checker**,
不是 Lean 定理。

## 状态(2026-05-05)

### Tier 1 — Loop-free kernel pairs ✅(`v0.1-tier1`)

- `softmax_kernels_refinement` —— naive ↔ 数值稳定 softmax
- `log_sum_exp_refinement` —— direct LSE ↔ shift-trick LSE
- `softmax_reciprocal_refinement` —— `y = e/s` 逐元素除法 ↔ `inv_s = 1/s; y = e * inv_s`
- math 引理 `welford_eq_two_pass`(为 Tier 2 Welford 准备)

### Tier 2 — Streaming reductions ✅(`v0.2-tier2`)

- `welford_kernels_refinement` —— Welford ↔ two-pass variance
- `online_softmax_recurrence_eq_batch` —— FlashAttention 算法核心
- `layernorm_kernels_refinement` —— fused single-pass ↔ two-pass LayerNorm
- 操作语义 `forLoop_inv`(所有循环 kernel 的工作母机;每轮先把循环计数器绑定到
  索引寄存器,支持嵌套循环)
- 超出原 scope 的:Mask + Bool channel(masked load/store、`tl.load(p, mask=m, other=o)`、
  6 个比较算子、`other=None` 非确定性 oracle)、Typed Tile 重构
  (`Op : TileDType → TileShape → Type` 全链路 typed `evalOp`/`stepStmt`,`RegFile` 由
  `(dtype, shape, name)` 索引)、`WithBot ℝ` 通道载体(`Op.negInf` 求值为真正的 `⊥`
  而非 `-1e38` stand-in)

### Tier 3-A — FA-1 forward 全套 ✅(待打 `v0.3-tier3a`)

- `fa1_forward_correct`(non-causal,single-block 推理)
- `fa1_forward_correct_strided`(任意 stride layout)
- `fa1_forward_correct_strided_causal`
- `fa1_forward_correct_4D` / `fa1_forward_correct_4D_causal`(`batch × heads × seq × dim`)
- `V1Boundary.lean`:boundary-mask 与 boundaryD 的 8 个变体定理(部分 KV 块、
  部分头维)
- `ScoreVariants.lean`:softcap / ALiBi / sliding-window / 三合一组合,
  4 个 forward 变体定理
- 共 ~16k 行 FA-1 forward 证明(V0 + V1Boundary + ScoreVariants + Common)

### 横向 infra ✅(超出原 PLAN scope,与 Tier 推进并行)

- **双层架构**:`ComputeKernel` / `AlgorithmKernel` 用 `eraseDType` 桥接;
  `Kernel.Correct` / `Kernel.Refine` 在算法侧 kernel 上证;`ProjectedCorrect` /
  `ProjectedRefine` 是内部投影义务;`ComputeKernel.ComputeCorrect` /
  `ComputeKernel.ComputeRefine` 是 user-facing 入口;bitcast 通过 compute
  projection 路由
- **Float dtype erasure**:`.fp32` / `.fp16` / `.bf16` → `ℝ`;bitcast 通过单一
  computable decoder;invalid bitcast bits 在 macro 展开时拒绝
- **整数 dtype**:signed Int abstraction(typed integer load,Nat store coverage,
  Nat bitwise surface)
- **Memory subsystem**:Bounds(读越界安全)、Footprint(读写区域提取)、
  Frame(disjoint frame composition、unrelated frame preservation)、
  Block pointer boundary semantics、proof memory API 通过 `readMem`
- **Concurrency framework**:Trace 词汇、Refinement、Async sequentialization 合约、
  Atomic add proof surface、Async copy / async wait / debug barrier failure markers、
  Producer-consumer discipline、projection failure correctness lemmas、
  unsupported atomic failure markers、generalized effect projection
- **ND general 路径**:ND grid launch theorem、grid composition、`Fin shape.length`
  通用 axis、`expand_dims` / `static_range` / shape view 表面
- **DSL 表面扩张**:prefix scan、arg / sort、shape view、unary math、bitcast、
  expand_dims、static_range、minimum / maximum / abs、Boolean operators 与比较
- **DSL 模块化**:Triton core / DSL / semantics / float / memory typing 拆分,
  optional Triton well-formedness checker

### 数据

- 80 个 `.lean` 文件,~42.7k 行
- 全库 0 sorry(除 `FlashAttention1/Backward.lean` 1 个,正在做)
- `lake build` 干净

## 进行中

### FA-1 backward

原 PLAN 把 backward 列为 P3+ 永远不做;重定向后进 in-scope(见 §决策日志 9)。

`VeriTile/Examples/FlashAttention1/Backward.lean`(495 行,1 个 `sorry`):

- `streamingLSE_eq_lseReal` —— forward LSE store 用 `m_i + tl.log(l_i)` 等于
  unshifted log-sum-exp(backward 重建 P 的关键桥)
- `attentionBackwardReal_eq_reverseMode` —— closed-form reverse-mode FA-1 backward
- 数学层 tile bridges:probability、`dP = dO · Vᵀ`、row correction
  `D_i = Σⱼ P_ij · dP_ij`、`dS = P * (dP - corr[:, None])`、`dV = Pᵀ · dO`、
  `dQ = dS · K · scale`、`dK = dSᵀ · Q · scale`
- `softmax_jvp_identity` —— softmax JVP 形式
- `strippedBackward_tile_bridges_complete` —— bundled 数学层 surface
- `fa1BackwardStrippedKernel_correct` —— 主定理(stripped:无 mask、无 multi-block、
  单 program-id),sub-2b 执行 wiring 是当前 sorry

下一步:闭合 sub-2b → 加 mask → 加 multi-block → causal backward → FA-2 backward。

## 路线图(优先级排序,无固定时间窗口)

### 近期 — Tier 3-A 收口与 backward 第一阶段

- 闭合 FA-1 backward stripped 主定理(剩 1 个 sorry)
- 打 `v0.3-tier3a` tag(forward 全套已就位)
- FA-1 backward 加 mask
- FA-1 backward 接入 multi-block(等中期 multi-block 语义)

### 中期 — Tier 3-B FA-2 forward + headline corollary

*语义扩展:*

- `multiBlockExec : Kernel → InitMem → Grid → FinalMem` 模型 —— 跑遍所有 program_id
  (3D grid:`batch × heads × Q-blocks`),组合各 program 的写入
- Stride / layout 模型(每 program_id 行优先连续;通用 layout 系统不做)
- Disjoint-writes 引理 —— 无 atomic 的纯前向 kernel,各 program 输出区域不相交,
  可独立分析后再组合

*Kernel 与定理:*

- FA-2 kernel 通过 `triton { ... }` 嵌入(多轴 program_id、延迟 rescale 路径、
  mask-skip 路径)
- `fa_2_forward_correct` —— 类似 FA-1 forward 的:(a) 序列长度并行化、
  (b) 延迟 rescaling、(c) 全 mask 块跳过
  - 延迟 rescale 等价:`O_final / l_final` 不依赖中间 `O` 是否每步 rescale
    (`delayed_rescale_eq` helper)
  - 块跳过正确性:全 mask 块对递推贡献为 0
  - Tier 3-A single-program invariant 工具大量复用
- **Headline corollary `fa1_eq_fa2`** —— ~30 行,通过 spec 传递性导出

### 中期 — Tier 3-C FA backward 全套

- FA-1 backward + mask + causal + multi-block(在 stripped 单块闭合后展开)
- FA-2 backward(沿用 multi-block 语义)
- Headline corollary `fa1_backward_eq_fa2_backward`(对偶于 forward)

### 中期 — Tier 4 Production kernel 第二批

按 Triton 工程师真实使用频率排序:

- **Grouped GEMM** —— production attention serving、MoE 路由、batched matmul 关键路径
- **Mamba / SSM** —— state-space model,Triton 实现是 production 候选
- **RoPE** —— rotary position embedding,attention serving 必需
- **Fused-norm 全家族** —— RMSNorm、QK-norm、LayerNorm with bias
- **多头 attention variants** —— GQA、MQA、MLA(MoE-aware attention)

每个 production kernel 都需要 forward + backward + 与对应数学 spec 的 Lean 等价。

### 远期 — 横向 infra 续

- **Concurrency 真正接进证明链** —— 当前是 failure markers + projection 边界。
  下一步:atomic add correctness 主定理、async copy 序列化等价定理、producer-consumer
  pattern 的端到端 refinement
- **`.py` paste-in 表面** —— Python lifter 雏形(原 P3+ 项推进到中远期);macro
  接受度从"嵌入式风格"逼近"真 `.py` 几乎不改"
- **ND general 框架收口** —— 消除残余的 1D / 2D-specific 路径,所有 op 都通过
  通用 `Fin shape.length` axis
- **Effect framework 完善** —— compute effect marker 化,projection failure
  correctness lemmas 升级为完整的 effect-aware 证明 stack
- **Block pointer 全套** —— 当前是 boundary semantics + 部分 layout;扩展到 stride
  manipulation、multi-D 嵌套 block pointer

## 显式保留外部(不在 Lean 证明链)

经过 v0.2 双层架构锁定,以下永久不在 Lean 证明范围:

- **IEEE-754 bit-level 行为** —— NaN 传播、denormal、rounding mode、硬件 dot
  precision、fast-math、exception flag。Compute 层用 differential testing 覆盖
- **Triton autotune 配置 / 性能 cost model** —— autotune 选择(`BLOCK_M` /
  `BLOCK_N` / `num_warps`)不影响算法等价
- **WGMMA / Hopper-specific 指令调度** —— 算法等价不依赖,纯性能层
- **PTX / SASS 后端忠实性** —— 假设 Triton 编译器忠实实现 Triton 语义

注意:**Backward**、**Python lifter**、**并发原语证明** 已从原 PLAN 的 P3+ 列表
**移出**,进入路线图(见 §决策日志 9)。

## Cross-cutting

### 风险登记表

| Risk | 触发信号 | Mitigation |
|---|---|---|
| FA-1 backward `sub-2b` 执行 wiring 长期卡住 | sorry 持续 ≥ 2 周 | `/lean4:autoprove --deep=stuck`;失败则切成更小 stripped sublemma |
| `multiBlockExec` 形式模型超预期复杂 | 起草 disjoint-writes 不变量时 | 限定为"每 program_id 行优先连续";通用 layout 系统不做 |
| FA-2 multi-block 抽象超预期复杂 | 起草 `multiBlockExec` 时暴露隐藏复杂度 | 先 stripped FA-2(单 program,无 mask-skip);完整版作为后续 |
| `tl.dot` 在 `Value.tile2D` 上 simp 表现糟糕 | FA 证明卡 simp | 自定义 simp 引理;或改 `unfold + induction` 风格 |
| 并发原语证明引入太多复杂度 | atomic add 主定理推进 ≥ 3 周无进展 | 退化成 sequential consistency 单层 spec(无 race 模型);完整 weak memory 留远期 |
| Production kernel 第二批 scope 失控 | Tier 4 列表持续扩张 | 一次只取 1-2 个 kernel 闭合 + tag,而不是同时启动多条线 |
| Python lifter 与嵌入式 macro 接受度 mismatch | lifter 输出无法 type-check | macro 接受度先扩,lifter 后跟;不反向驱动 |
| `/lean4:autoprove` 处理不了长证明(~300+ 行) | held-out close rate < 30% | 切成插件能闭合的子引理 + 人工组合;benchmark 报告记录人机分工 |
| 上游 `lean4` 插件 break / API 变更 | `/lean4:autoprove` 报错或输出格式变化 | pin 具体插件版本(commit / version 记录在 `CONTRIBUTING.md`);fall back 跑老缓存版 |
| Float abstraction bridge 对外部 reviewer 太强 | review / audit 要求 IEEE-754 proof | 双层架构永久锁定:VeriTile 证明 Real abstraction,通过 erasure 暴露 float-facing theorem,代表性 float 行为用测试验证 |
| Diff-test 数值差超容差(代表性 kernel) | GPU 输出 vs reference | 黄旗,非 gate —— 排查 IEEE-754 内禀差异、`.py` 实现 bug、对应断言不准。形式证明仍然有效 |

### 外部验证

VeriTile 把两种外部验证分开:

1. **Compute-gap contracts** —— 只在 theorem 使用 `gap := .require contract`
   时必需。这类检查比较 compute-facing `ComputeKernel` 行为与投影后的 `AlgKernel`
   行为是否满足 contract relation(例如 exact equality 或 epsilon bound)。Lean 记录
   `ExternalChecked contract`;不在内部证明 IEEE / bit-level compute 行为。
2. **Runtime credibility tests** —— 可选的 smoke / differential tests,比较代表性
   可运行 Triton/Python 实现与 PyTorch / `flash-attn` reference。它们帮助说明嵌入式
   DSL 对应真实可运行 kernel,但不是 Lean 证明链,也不替代 `GapPolicy` contract。

**Runtime credibility tests 解决不了什么** —— 对手写 `.py` 跑 diff testing
不能验证嵌入式操作语义本身,也不能关闭嵌入式 AST 与真实 `.py` 源码之间的 gap
(那需要 Python lifter,见 §远期)。

**当前 runtime 代表集**(随路线图扩展):

- 1-2 个 Tier 1+2 kernel(例如 `naiveSoftmaxKernel`、一个 streaming kernel)
- FA-1 forward(各变体抽样)
- 中期:FA-2 forward、FA-1 backward、FA-1↔FA-2 互查
- 远期:Tier 4 production kernel 抽样

**容差**:Tier 1+2 ≤ 1e-5 元素绝对差;FA-class ≤ 1e-3(`flash_attn` 参考自身 vs
PyTorch 量级)。

**Runtime 失败处理**:runtime diff-test 失败是黄旗,不是 gate —— 排查:IEEE-754
内禀差异(可接受,文档化)、`.py` 实现 bug(修)、对应断言不准(对照重检)。它不让
嵌入式语义上的 Lean theorem 失效。required compute-gap contract 失败则不同:
使用 `gap := .require contract` 的 theorem 在外部证书重新生成并通过之前,不应视为
完整 certified。

### LLM benchmark protocol

让 close rate 指标可复现:

- **Held-out 集**:每个新增 Tier 提前圈 5-10 条引理到 `bench/llm_eval/<tier>/`
  独立文件结构;主库不依赖 held-out 文件。**圈定时机:每个 Tier 的*开头*,
  在该 Tier 的 LLM 工具调优*之前*** —— 防止"在 eval 上调工具"
- **每个 held-out theorem 允许的上下文**:held-out 文件的 imports + 该文件中
  已证(在 held-out 之前)的所有 theorem + Mathlib 引理名查找 + 工具自带的
  retrieval(若有)。模型看到的就是 theorem statement、section 上下文、显式
  检索的引理。**不允许**看到:项目库其余内容(imports 之外)、held-out 引理
  本身的证明、任何针对该 held-out 引理的人工提示
- **eval 期间禁止**:人工 prompt iteration、人工 lemma name 提示、对 held-out
  引理 fine-tune、对 LLM 输出做任何编辑后再跑。eval 是 hands-off:工具端到端跑,
  成功 / 失败,就一个数据点
- **试验数**:每条 held-out 引理 N=5 次独立运行,随机 seed 不同。Close rate =
  (全部 (lemma, seed) 对中成功的次数) / (5 × |held-out 集|)
- **Cost 报告**:整个评测的 API call 数、token 数、wall-clock、美元成本 —— 每
  Tier 报告。包含失败重试
- **跨 Tier 允许的 prompt iteration**:一个 Tier 的 eval 跑完之后,prompt template
  可以改;改过的工具用于*下一个 Tier 开头*重新圈定的*新* held-out 集

### Open-source 节奏

- 主仓库:`github.com/Lizn-zn/VeriTile`(已公开)
- 主分支恒可构建
- Tier 完成时打 tag(已有 `v0.1-tier1`、`v0.2-tier2`;待打 `v0.3-tier3a`,
  之后随路线图增长)
- 每个 release 配 release notes(新定理、新语义、benchmark 数据、scope 变化)
- README 持续维护双语(英 + 中)
- `CONTRIBUTING.md`,加 kernel-pair 的 tutorial(以 Tier 1 log-sum-exp 作为
  worked example)
- `scripts/prove.sh` 留在仓库内(不单独打包);benchmark eval 结果与项目一起
  发布。底层的 `lean4` Claude Code 插件是 upstream,不 vendor —— `CONTRIBUTING.md`
  记录 pin 的版本

## 决策日志

1. **2026-04-26 brainstorm** —— Bet 选择 (b) "Verified kernel rewrites" + (e)
   "工具可用",mix α(Attention-headlined),Tier 3 框架 T3-D(T3-A + T3-B),
   sequencing approach 2(垂直切片)。详情参考原 brainstorm 记录
2. **2026-04-26** —— LLM 工具策略:wrap 现有 `lean4` Claude Code 插件
   (`/lean4:autoprove`),做薄薄 `scripts/prove.sh`,而非自建 Python 证明工具。
   理由:插件已经提供 LSP 集成、多 cycle 迭代、deep-mode 升级、tactic cascade、
   修复模式 —— 自己重新造是不必要的投资
3. **2026-04-26** —— Differential testing:非形式 validation 工件,不是 phase gate,
   不属于可信证明链。仅在选定代表性 kernel 上维护(全 program 共 ~3-4 个),
   作为 credibility 补充
4. **2026-05-04** —— 验证架构永久锁定:Algorithm 层(Lean 证明)+ Compute 层(外部
   测试),两层不通过 Lean 定理连接,bridge 是经验性的(测试)。IEEE-754 形式化
   永久不在 Lean 证明范围
5. **2026-05-04** —— Memory subsystem 落地:Bounds、Footprint、Frame、disjoint
   composition,作为 Tier 3 multi-block 的 prerequisite
6. **2026-05-04** —— Concurrency framework 落地(failure markers + projection
   边界):为远期 atomic / async 主定理打底,但当前不在主证明链上
7. **2026-05-04** —— Float dtype 双层架构落地:`ComputeKernel` / `AlgorithmKernel`
   通过 `eraseDType` 桥接,float-facing theorem 通过 erasure 暴露
8. **2026-05-05** —— **PLAN 重定向**。从"7-10 个月 PLDI program plan + 8 个
   hand-picked 主正确性定理 + headline corollary"改为"开放路线图的 Triton
   verification 项目"。

   理由:
   - 原 PLAN 的 8 主定理 scope 实际上仅是 Triton verification 的 forward 子集;
     real Triton 工程师面对的 production kernel(backward、并发、多变体 attention、
     grouped GEMM、SSM)涵盖远远更广
   - 投稿驱动的 deadline / scope 缩水规则(Gate A→B、B→C、C→D 的 fallback 投 OOPSLA
     等)与项目的真实长期方向不匹配
   - 横向 infra 实际投入(Float erasure 双层、Memory frame、Concurrency framework、
     ND launch、Score variants)已远超原 PLAN 列出的 supporting work,反映"做真
     Triton verification"的方向
   - 原 PLAN "P3+ 永远不做" 的 backward / Python lifter / 并发原语,在实际项目
     节奏中已经或即将进入 in-scope

   形态变化:
   - **删**:7-10 个月时间窗口、Phase A/B/C/D 结构、Gate A→B/B→C/C→D 决策规则、
     PLDI / OOPSLA / ICSE 投稿框架、`v1.0-pldi` 目标 tag
   - **保**:验证架构(Algorithm/Compute 双层)、Differential testing、LLM benchmark
     protocol、决策日志、open-source 节奏
   - **改**:Scope 从"8 主定理"改为"已完成 + 进行中 + 路线图"的开放分层

9. **2026-05-05** —— **Backward pass、Python lifter、并发原语证明 移出 P3+,
   进入 in-scope**。

   - **FA-1 backward** 已经在做(`Backward.lean` 495 行,1 sorry),作为近期
     收口目标
   - **Python lifter** 从 P3+ 推进到中远期 —— Triton 用户友好度需要 paste-in 表面,
     不止文档对应
   - **并发原语证明** 当前是 failure markers + projection 边界(已搭),下一步是
     atomic add correctness、async copy 序列化等价等主定理

   理由:这三项是"真 Triton verification"与"toy Triton verification"的分界。
   Triton 工程师日常 ship backward;production kernel 大量使用 atomic / async;
   不接受 paste-in 的 DSL 没有 user。

## Implementation plan

主线执行追踪在 pinned roadmap issue #91(分层)+ 各 per-task issue。跨 phase
仍有价值的长文档归到 [`documents/`](./documents) 下(例如
`documents/ForLoopInvDesign.md`、`documents/DslMacroOptions.md`)。
