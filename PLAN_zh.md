# VeriTile — Program Plan

[English](PLAN.md) | **中文**

一个 7–10 个月的研究 program plan,通过 **8 个 Triton kernel 主正确性定理**(含 FlashAttention forward 正确性)+ **headline corollary `fa1_eq_fa2`** 交付一篇 "Verified kernel rewrites" 论文。

本文档是 2026-04-26 brainstorm 中达成一致的 program-level plan。下面每个 phase 各自有独立的 implementation plan(进入该 phase 时通过 `writing-plans` 流程编写)。

## 目标

**主目标** —— 投稿 PLDI(fallback OOPSLA-spring),核心 claim:

> 8 个生产 Triton kernel 主正确性定理(包括 FA-1 forward ≡ standard attention denotation 与 FA-2 forward ≡ standard attention denotation,推出 headline corollary FA-1 ↔ FA-2)在 Lean 4 中针对嵌入式 Triton 操作语义形式化证明,证明起草由社区 `lean4` Claude Code 插件(`/lean4:autoprove`)加速,并以非形式 differential-testing 工件展示嵌入式算法对应可在 GPU 上运行的代码。

**副目标** —— 工具链(操作语义、嵌入式 `triton { ... }` DSL、对 `/lean4:autoprove` 的薄封装 `scripts/prove.sh` 用于 benchmark eval、为 differential testing 而手写的 Python 对应件)做扎实,可内部使用、可开源接收外部贡献。

## 范围:8 个主正确性定理 + helpers + corollary

| Tier | Phase | 主定理 | 类型 |
|---|---|---|---|
| 1 | A | (#1) `softmax_kernels_refinement`(已完成)·(#2) `log_sum_exp_refinement`(已完成)·(#3) `softmax_reciprocal_refinement`(已完成) | kernel ↔ kernel × 3 |
| 2 | B | (#4) `welford_kernels_refinement`(已完成)·(#5) `online_softmax_recurrence_eq_batch`(已完成)·(#6) `layernorm_kernels_refinement`(已完成) | kernel ↔ kernel × 3 |
| 3 | C, D | (#7) `fa_forward_correct`(T3-A,Phase C)·(#8) `fa_2_forward_correct`(T3-B,Phase D) | kernel ↔ math × 2 |

**外加 helpers**(不计入主定理):
- 数学引理 `welford_eq_two_pass`(Phase A,#4 的准备)
- 操作语义 helper `forLoop_inv`(Phase B,所有循环 kernel 证明的工作母机)
- Mask 处理 helper(Phase C;具体形式取决于 Phase A scouting 结果 —— 见 §Phase C)
- `delayed_rescale_eq` helper(Phase D,在 #8 中使用)

**外加 headline corollary**(派生,**不是第 9 个主定理**):
- **`fa1_eq_fa2 : Y_fa1 = Y_fa2`** —— 由 #7 与 #8 通过 `standardAttentionMath` 的 spec 传递性导出(~30 行)。**这是论文 foreground 的句子**,但内容已经被 #7 与 #8 完全包含。

## 验证架构(永久)

VeriTile 的验证拆成**两个独立互补的层**(锁定理由见 #58):

```
Algorithm 层(Lean 证明)             Compute 层(外部测试)
────────────────────────             ──────────────────────
AlgorithmCorrect ck spec             ComputeCorrect ck
  := ck.toAlgorithm? = ok ∧            := 实际 Triton kernel 在差分
     Kernel.Correct ak spec               测试下与算法 spec 比较,
                                          epsilon bound 内匹配。
  Real / Int / Nat 语义。              外部 pipeline,不是 Lean 定理。
  无 IEEE rounding,无 NaN,
  无 fp exception flag。
```

两层**不通过任何内部 Lean 定理连接**。两者之间的 bridge 是经验性的(测试),
**这是 design 决定**。一个 VeriTile-certified kernel 意味着:
> 算法结构由 Lean 证明正确(AlgorithmCorrect),**并且** fp 层实现已通过
> 代表性测试集与该算法对照验证(ComputeCorrect)。

完整 certification 需要两层都成立。

### 永久性

IEEE-754 形式化(NaN 传播、denormal、rounding mode、硬件 dot precision、
fast-math、exception flag)**永久不在 Lean 证明范围内**。这不是延期 ——
不存在计划中的 bridge 定理把 compute 执行连到 Lean 内部的算法执行上。
ComputeCorrect 现在是、未来也是 test-backed。

### Algorithm 层细节

1. DSL 接受用户面向的 float kernel,`.fp32`/`.fp16`/`.bf16` dtype 标注照常。
   bit-level 常量(例如哨兵 `1.0` 写成 `tl.bitcast(0x3f800000, tl.float32)`)
   通过单一 computable decoder 接受;只支持 finite-normal binary32 模式。
2. 算法正确性在 `Kernel.eraseDType`(fp* → real,intN → int,uintN → nat)
   之后的算法层 kernel 上证明。
3. 形式 Lean 层使用 `ComputeKernel.AlgorithmCorrect` 和 `Kernel.AlgorithmRefine`。
   现有 `Kernel.Correct` 证明在算法侧 kernel 上继续有效。

### Compute 层细节(测试)

1. `ComputeKernel` 忠实保留用户写的 compute-facing AST(包括 `ComputeOp.bitcast`、
   fp/int 宽度拼写等),让测试 pipeline 能 lift 回 Triton 源码做实机执行。
2. 差分测试 pipeline(独立 workstream,见 #58 follow-up)生成样本输入,
   执行 lift 出来的 Triton kernel,按 `|actual_output - algorithm_output_decoded|
   < epsilon` 校验。
3. epsilon 策略 per-op / per-kernel,文档化在每个 test suite 旁边;不进 Lean 证明。

### 用户视角

- 算法结构有 bug:AlgorithmCorrect 抓(Lean 证明 fail)。
- IEEE-specific edge case 有 bug(NaN、overflow、denormal 处理等):
  ComputeCorrect 抓(测试 fail)。**这一层 by design 没有 Lean 证明义务**。
- "End-to-end correct" 意味着两层都绿。

Real 算法正确性到 floating computation 的 trusted bridge **是测试 pipeline**,
不是 Lean 定理。

### Tier 1 — Loop-free(Phase A,3 对)

1. **`softmax_kernels_refinement`** —— 已完成 —— naive ↔ 数值稳定 softmax
2. **`log_sum_exp_refinement`** —— direct LSE kernel ↔ shift-trick LSE kernel(`logsumexp(x) = m + log(Σ exp(x − m))`,`m = max(x)`)
3. **`softmax_reciprocal_refinement`** —— `y = e/s` 逐元素除法 kernel ↔ `inv_s = 1/s; y = e * inv_s` 预算倒数 kernel(省 N−1 次除法;数学上轻量,展示 fused-multiply 重写模式)

外加一个 *math-only* 引理 `welford_eq_two_pass`(为 Phase B 的 Welford kernel 定理做准备;不计入 kernel 对总数)

### Tier 2 — Streaming reductions(Phase B,3 对)

4. **Welford ↔ two-pass variance**(kernel-level)—— Welford 用 `forLoop` 跨 tile 维护 `(M, S, n)`;two-pass 用两次 `tl.sum`。把 Phase A 的数学引理上升到真 kernel
5. **Online softmax recurrence ≡ batch softmax** —— *FlashAttention 算法核心,论文中心*。证明流式递推 `(m_new, l_new) = (max(m_old, max(x_block)), exp(m_old − m_new) · l_old + Σ exp(x_block − m_new))` 产生与一次性 batch 形式相同的 `(m, l)`,通过对 block 数归纳
6. **Fused single-pass LayerNorm ≡ two-pass LayerNorm** —— 用 Welford 引理 + affine `(x − μ)/√(var + ε) · γ + β` 变换。展示同一 lemma 跨 kernel 对的复用

### Tier 3 — 生产 attention(Phase C+D)

7. **`fa_forward_correct`**(T3-A,Phase C)—— FlashAttention-1 forward kernel ≡ `standardAttentionMath Q K V causal` denotation。Single-block 推理:每个 program_id 处理一块 Q × 完整 KV 循环。把 Phase B 的 `(m, l)` invariant 扩展为 `(m, l, O)` 加上输出累加器。输出是 (S × D) 矩阵的逐元素 `(i, d)`,**不是** scalar(`observeY2D` 读 `(i, d)` cell)

8. **`fa_2_forward_correct`**(T3-B,Phase D)—— FlashAttention-2 forward kernel ≡ `standardAttentionMath` denotation。形态与 #7 一致,三个工程差异:(a) 序列长度并行(FA-2 grid 是 3D:`batch × heads × Q-blocks`)、(b) `O` 延迟 rescale、(c) 全 mask 块跳过。Single program-id 部分大量复用 Phase C 的 `(m, l, O)` invariant 工具;multi-block 协调是 Phase D 新增

**Headline corollary**(派生,**不是第 9 个定理**):`fa1_eq_fa2 : Y_fa1 = Y_fa2`,由 #7 与 #8 通过 `standardAttentionMath` 的 spec 传递性导出(~30 行)

## Phase 结构(总计 ~28–42 周)

### Phase A — Tier 1 + `scripts/prove.sh` + T3 scouting(~6–8 wk)

**交付物:**
- 3 个 Tier 1 kernel-pair 定理闭合(#1 已完成,关闭 #2 与 #3)
- 1 个 math-only 引理(`welford_eq_two_pass`)
- `scripts/prove.sh`:~80 行 bash 封装,内部调用 `claude -p "/lean4:autoprove <file> --commit=never --planning=off --review-source=none --max-cycles=N"`;捕获 JSON 流输出到 `Logs/`;从 result message 解析成功 / 失败;成功 exit 0,失败 exit 1(用于 benchmark close-rate 统计)。`lean4` Claude Code 插件(已安装在 `~/.claude/plugins/cache/lean4-skills/`)提供 LSP 集成(`lean_multi_attempt`)、多 cycle 迭代 + 卡住检测、deep mode 升级、每个 sorry 2-3 个 tactic 候选的 cascade、build 错误的修复模式 —— 我们 wrap,不重新实现。
- T3 scouting 文档 `Notes/T3_scouting.md`(~10–15 页):
  - FA forward 伪代码 + invariant 草稿
  - B/C/D 语义扩展所需的 lemma 清单
  - 显式风险登记表(未预期的形式化 gap)
  - 可行性判断:T3-B 在窗口内可行,还是只做 T3-A?

**验证:**
- 在主库之外建独立的 `bench/llm_eval/` 目录,把 `softmax_naive_correct` 复制一份并把证明体替换为 `sorry`(主库的版本保持 intact 且已证)。`scripts/prove.sh`(封装 `/lean4:autoprove`)必须在这个 held-out 副本上 close rate ≥ 1/3,N=5 次独立运行(见 §LLM benchmark protocol)
- P1 + Tier 1 全部 build 干净

**Exit gate(`v0.1-tier1`):** 全部交付物达成;`lake build` 干净

### Phase B — forLoop 语义 + Tier 2(~8–12 wk)—— **已完成**(`v0.2-tier2`)

**交付物:**

*操作语义:*
- 把 `stepStmt` 与 `stepStmts` 改为 `mutual` 块,以支持嵌套循环
- `forLoop` 操作语义,显式 `termination_by (sizeOf body + 1, n − i)` lex 度量
- `forLoop_inv` 引理(Phase B/C/D 所有循环证明的工作母机);**每轮在 step body 之前先把循环计数器绑定到索引寄存器 `idx`**,这样 body 中读 `idx` 的代码看到的是当前 iteration:

  ```
  ∀ (idx : RegName) (n : Nat) (body : List Stmt) (P : Nat → BlockState → Prop),
    P 0 s_init →
    (∀ i s, P i s →
      ∃ s', stepStmts body (s.setReg idx (Value.scalar (i : ℝ))) = some s'
            ∧ P (i+1) s') →
    ∃ s_final, exec_forLoop idx n body s_init = some s_final ∧ P n s_final
  ```

  索引绑定不可省:Welford / online softmax / FA 的 body 都把 `idx` 作为 register 引用(比如 `offs := pid * BLOCK + idx * STRIDE`)。不在 `stepStmts body` 之前绑 `idx`,body 对 `i` 的数据依赖就无法在 `BlockState` 里表达

*定理:*
- 3 个 Tier 2 kernel-pair 定理(#4、#5、#6)
- Tier 2 #5(online softmax recurrence)证明预估 ~80–120 行 Lean;**这是论文核心技术贡献**

*LLM 工具:Phase A 的 `scripts/prove.sh` 之外不需要新工作。* `lean4` 插件已有的能力(LSP 搜索、tactic cascade、deep mode 升级、修复模式)覆盖了我们本来会在这个 Phase 建的东西。Phase B 的"工具工作"就是在 Tier 2 held-out 引理上跑 wrapper 并记录数字。

*Differential testing 工件(非 gating,见 §Differential testing):*
- 建立 `tools/diff_test/` 目录与 side-by-side 对应关系的约定
- 在 1–2 个 *选定的代表性* Tier 1+2 kernel 上,手写对应的 `.py` Triton 实现并一次性对 PyTorch reference 比较(逐元素绝对差最大值 ≤ 1e-5)
- 其余 Tier 1+2 kernel 不做 diff-test 工件;它们的正确性完全由 Lean 证明承担

**验证(gating):**
- ~~`scripts/prove.sh` 在 Phase B held-out 集上 close rate ≥ 50%~~ —— **gate 时缩水。** LLM 评测推迟到后续 milestone;Phase B 仅以已闭合 Lean 定理交付

**验证(工件,非 gating):**
- ~~1–2 个 Tier 1+2 kernel 的 diff-test 工件~~ —— **gate 时缩水。** Diff-test infra 推迟;Tier 1+2 正确性完全由形式 Lean 证明对 embedded 操作语义承担

**额外交付物(超出 Phase B 原定 scope):**
- Mask + Bool channel 扩展(Slices 1–4):masked load/store、`tl.load(p, mask=m, other=o)` DSL 表面、6 个比较算子、通过 `BlockState.undef` oracle 实现 `other=None` 的非确定性
- Typed Tile 重构:`Op : TileDType → TileShape → Type` 全链路 typed `evalOp`/`stepStmt`;`RegFile` 由 `(dtype, shape, name)` 索引 —— 消除动态 tagged `Value` union
- `WithBot ℝ` 作为 `.real` 通道载体:`Op.negInf` 求值为真正的 `⊥` 而非 `-1e38` stand-in,从 OnlineSoftmax 定理中删除 `h_lo` 前提,扫除 Phase C FlashAttention 的前提扩散障碍

**Exit gate(`v0.2-tier2`):** ✅ 已达成。3 个 Tier 2 kernel-pair 定理闭合;全库 0 sorry;`lake build` 干净(2729 jobs)

### Phase C — `tl.dot` + masking + Tier 3-A(~8–12 wk)

**交付物:**

*语义扩展(比初步预估更广 —— 见 Phase A scouting 结果):*
- `Value.tile2D : (m n : Nat) → (Fin m → Fin n → ℝ) → Value` 2D matrix tile 构造子
- `Op.dot a b`:`(A · B)[i,k] = Σⱼ A[i,j] · B[j,k]`,通过 Mathlib `Finset.sum` 表达
- 比较算子 `Op.lt`、`Op.le`(返回 0-1 `ℝ` tile 或布尔 tile)
- `Op.where(cond, a, b)` 元素级 mask:基于 *predicate tile* 分支,**不是** sentinel value
- 多轴广播 `Op.broadcastTo`:把 1D tile 提升到 2D 形状(例如 Q row 提升到 (Q-block × K-block) score tile)
- 因果索引辅助:"当前 Q-block 起点"、"当前 KV-block 起点"等的索引算术,与 `Fin` 算术的等价证明
- 2D 输出寻址:`observeY2D : Option BlockState → Fin S → Fin D → Option ℝ` 读 `(s.pid_to_address (i, d))`(1D `observeY` 仅适用于 Tier 1+2)
- 这里 single-block 推理已足够覆盖 *算法内容* —— multi-program-id grid 协调推到 Phase D

*Mask 处理 —— 待定的 design choice(由 Phase A scouting 拍板):*

天真主张"score = `-1e38` 在 softmax 中权重为 0"在 `ℝ` 上**为假**,因为 `Real.exp(-1e38) ≠ 0`。两种可行解决,二选一是 Phase A scouting 的交付物,实现落在 Phase C:

* **Option α —— 扩展实数。** Score 用 `WithBot ℝ`(或 `EReal`);把 `Real.exp` 扩展为 `⊥ ↦ 0`。kernel 中 mask 的位置在*语义层面*产生 `⊥`(浮点级 `-1e38` 是*concretization*,只与 differential testing 相关,与形式证明无关)。数学干净;工程代价不轻 —— `Op.dot`、`Op.where` 等都要在扩展实数上 compose
* **Option β —— Mask-predicate denotation。** Spec 直接用 `maskedSoftmax (x : Fin n → ℝ) (mask : Fin n → Bool)`:`maskedSoftmax x mask i = if mask i then 0 else exp(x i) / Σⱼ if mask j then 0 else exp(x j)`。kernel 嵌入必须在 AST 层级显式穿一个 Boolean mask register 经过 `Op.where`,**不能**用 sentinel float。绕过扩展实数;但要求 kernel 嵌入在 AST 上暴露 mask predicate

Phase A scouting 在 FA pseudocode 上各跑一次评估,推荐一个;Phase C 实施

*定理:*
- `fa_forward_correct` —— 完整 Lean 证明,无 sorry
- 命题(草稿;mask 表达取决于 Phase A scouting 的决议):

  ```
  theorem fa_forward_correct
      (Q K V : Matrix (Fin S) (Fin D) ℝ) (causal : Bool)
      (s : BlockState) (h_inputs : InputsLoaded s Q K V) :
      ∀ (i : Fin S) (d : Fin D),
        observeY2D (exec FAForwardKernel s) i d
          = some ((standardAttentionMath Q K V causal) i d)
  ```

  其中 `standardAttentionMath Q K V causal : Matrix (Fin S) (Fin D) ℝ` 是数学层 attention 输出(S × D 矩阵),`observeY2D` 读 kernel `Y` 区域的 `(i, d)` cell。定理是逐元素的;行级 / 矩阵级形式是 corollary

- 证明思路:per Q block 的 `forLoop_inv`,invariant `(m_k, l_k, O_k)` 扩展自 Phase B 的 `(m, l)` 递推,加上输出 `O_k : Fin D → ℝ`。逐元素正确性由最后一轮的 `O_k / l_k` 推出

- **预估上调**:总计 ~400–600 行 Lean(语义扩展 ~150 行,kernel 定理证明 ~300 行,mask 相关引理 ~50–150 行,取决于 mask option)。最初 200–300 行预估没考虑比较 / 广播 / 2D-layout / 因果索引语义

*LLM 工具:依然在 Phase A 的 wrapper 之外不需要新工作。* `lean4` 插件的 `/lean4:autoprove` 通过 LSP(`lean_multi_attempt`、`lean_diagnostic_messages`)已经实现 step-wise tactic 生成 —— 我们本来要建的"v0.3 proof-state interaction"已经在那。Tier 3-A 的 ~400–600 行长证明,插件的 deep mode(`--deep=stuck`)处理长子目标链;若不能闭合整个定理,就闭合各个子目标,我们手工组合。任何人机分工都记入 benchmark report,论文可以描述工具做了什么 vs 人做了什么。

*Differential testing 工件(非 gating,见 §Differential testing):*
- 把 **FA forward**(Phase C 的选定代表性 kernel)加入 diff-test 集
- 在 `tools/diff_test/python/` 手写对应的 Python Triton kernel,GPU 上跑,与 `flash-attn` 的 `flash_attn_func` 比较,因果 / 非因果两种,容差 ≤ 1e-3
- 这是 credibility 工件,不是 phase gate。失败会触发调查,但只要形式证明闭合就不阻塞 Phase C 退出

**验证(gating):**
- `scripts/prove.sh` 在 Phase C held-out 集上 close rate ≥ 30%(见 §LLM benchmark protocol;held-out 引理是 `fa_forward_correct` 的子步骤)。证明变长时 close rate 期望降低;benchmark report 记录走势

**验证(工件,非 gating):**
- FA forward 的 diff-test 工件已就位(见 §Differential testing)

**Exit gate(`v0.3-tier3a`):** 全部交付物达成;Phase D 推迟或缩水时可投 OOPSLA

### Phase D — multi-block + FA-1 ↔ FA-2 + paper(~6–10 wk)

**交付物:**

*语义扩展:*
- `multiBlockExec : Kernel → InitMem → Grid → FinalMem` 模型:跑遍所有 program_id(可多轴 grid,FA-2 是 3D:`batch × heads × Q-blocks`),组合各 program 的写入
- Stride / layout 模型:够表达 FA-2 每个 `(batch, head, Q-block)` 的输出区域(每 program_id 行优先连续;通用 layout 系统不做)
- Disjoint-writes 引理:无 atomic 的纯前向 kernel,各 program_id 输出区域不相交,可独立分析每个 program 的局部正确性后再组合
- **预估上调**:~200–300 行语义扩展。最初 80 行预估只覆盖 `multiBlockExec` 裸定义;disjoint-write 证明 + 多轴 program_id + 3D-grid 不相交不变量才是真活儿。**有意不引入并发 / 原子**

*Kernel 与定理:*
- FA-2 kernel 通过 `triton { ... }` 嵌入(多轴 program_id、延迟 rescale 路径、mask-skip 路径)
- **主定理 #8 —— `fa_2_forward_correct`** —— 类似 Phase C 的 #7,针对 FA-2 的:(a) 序列长度并行化,(b) 延迟 rescaling,(c) 全 mask 块跳过
  - 延迟 rescale 等价:`O_final / l_final` 不依赖中间 `O` 是否每步 rescale(`delayed_rescale_eq` helper)
  - 块跳过正确性:全 mask 块对递推贡献为 0(用 Phase C 的 mask 处理引理)
  - **预估上调**:~250–400 行新代码(原 150–200 行预估没考虑 multi-block stride / layout 推理);Phase C single-program invariant 工具大量复用
- **Headline corollary `fa1_eq_fa2`** 通过 spec 传递性(~30 行):#7 与 #8 都证 kernel ≡ `standardAttentionMath`,所以两 kernel 输出相等。**这是论文 headline,但不是单独证明的定理**

*Benchmark 报告(本 Phase 唯一的"工具工作"):*
- 在完整 benchmark 集(各 Phase eval 中收集的 Tier 1+2+3 全部 held-out 定理)上跑 `scripts/prove.sh`,每条 N=5 trials
- 聚合 close rate / 平均 retry 次数 / wall-clock / API 成本
- 产出 `bench/llm_eval/results/full_report.md`,论文 §LLM-assisted proof tooling 节引用此表

*Differential testing 收尾(工件,非 gating,见 §Differential testing):*
- 把 **FA-2 forward** 加入 diff-test 集;一次性 **FA-1 vs FA-2 互查**(相同输入,输出在容差内一致),作为 `fa1_eq_fa2` corollary 的非形式 sanity-check 补充
- 为选定 dtype-annotated kernel 增加 float-facing theorem smoke coverage:
  typed DSL surface、erasure equality、代表性数值一致性。
- `tools/diff_test/` 最终状态:1–2 个 Tier 1+2 代表 + FA-1 + FA-2 + FA-1↔FA-2 互查。代表集之外的 Tier 1+2 kernel 仅靠证明保证

*论文 draft:*

10 节大纲:

1. Introduction —— 动机:LLM-driven kernel 重写 + 验证缺口
2. Background —— Triton、arm-in-lean、Vero、ATL
3. 嵌入式 Triton 子集与操作语义
4. Algorithmic equivalence framework(refinement 模式、gather/scatter)
5. **Online softmax recurrence**(Phase B 中心)
6. **FA forward correctness**(Phase C)
7. **FA-1 ↔ FA-2 verified rewrite**(Phase D 核心)
8. LLM-assisted proof drafting via the `lean4` Claude Code 插件:对 `/lean4:autoprove` 在 VeriTile benchmark 上的 evaluation,报告每个 Tier 的 close rate、deep-mode 升级率、总 wall-clock / API 成本。诚实地框定为对一个现有社区工具的 evaluation,不是我们造的工具。
9. Differential testing evaluation
10. Related work + Conclusion

**验证(gating):**
- 全部定理闭合,`lake build` 干净
- 完整 benchmark report(`bench/llm_eval/results/full_report.md`)可重现:重跑 `scripts/prove.sh` 产生的 close rate 在报告数字 ±10% 以内
- 论文 draft 完成并提交

**验证(工件,非 gating):**
- Diff-test 工件集完整:1–2 个 Tier 1+2 代表 + FA-1 + FA-2 + FA-1↔FA-2 互查,各自在容差内一致

**Exit gate(`v1.0-pldi`):** 论文提交;打 tag 发布

## Cross-cutting

### 风险登记表

| Risk | Phase | 触发信号 | Mitigation |
|---|---|---|---|
| `forLoop_inv` 接口不能干净支持嵌套循环 | B | Phase A scouting 在 invariant 草拟时发现 | 退化方案:`Nat.iterate` 形式(不直观但等价语义) |
| Online softmax recurrence Lean 形式化卡死 | B | 持续 ≥ 2 周无进展 | 跑 `scripts/prove.sh --max-cycles=20 --deep=stuck` 攻这个证明;再不行简化为 stripped 版(从此证明中砍 mask,Phase C 再加回) |
| `tl.dot` 在 `Value.tile2D` 上 simp 表现糟糕 | C | `fa_forward` 证明卡 simp | 自定义 simp 引理(P1 已为 `Value.bop` 等长 tile-tile 做过先例);simp 不合作就改用 `unfold + induction` 风格 |
| Mask sentinel-vs-real-zero gap(`Real.exp(-1e38) ≠ 0`)阻塞天真的 mask 形式化 | A scouting / C | Phase A 写 pseudocode 时发现 | 选 Option α(扩展实数:`WithBot ℝ` / `EReal`)或 Option β(mask-predicate denotation);Phase A 拍板 |
| 2D stride / layout 形式模型超预期复杂 | D | 起草 `multiBlockExec` 不变量时 | 限定为"每 program_id 行优先连续";论文中作为 assumed layout 文档化 |
| FA-2 multi-block 抽象超预期复杂 | D | Phase C 末草拟 `multiBlockExec` 时暴露隐藏复杂度 | T3-D → T3-A only;论文目标降档 OOPSLA |
| `/lean4:autoprove` 处理不了长证明(Tier 3 ~200–600 行) | C | Phase B 末 held-out close rate < 30% | 诚实报告:把长证明切成插件能闭合的子引理 + 人工写的组合;benchmark report 里描述人机分工 |
| 上游 `lean4` 插件 break / API 变更 | A–D | `/lean4:autoprove` 报错或输出格式变化 | 在仓库里 pin 一个具体插件版本(commit / version 记录在 CONTRIBUTING.md);fall back 跑老缓存版 |
| Float abstraction bridge 对 reviewer 来说太强 | C–D | review / internal audit 要求 IEEE-754 proof | 明确写出 trusted-boundary policy:VeriTile 证明 Real abstraction,通过 erasure 暴露 float-facing theorem,用测试验证代表性 float 行为。IEEE-754 proof 是 out of scope,不是隐藏未完成项。 |
| Diff-test 数值差超容差(代表性 kernel) | B–D | GPU 输出 vs reference | 黄旗,非 gate —— 排查:IEEE-754 内禀差异(可接受,文档化)、`.py` 实现 bug(修)、对应断言不准(对照重检)。形式证明仍然有效;diff-test 工件更新或文档化其范围 |
| 错过 PLDI deadline(~Month 11) | D | Phase C 末时间评估 | 投 OOPSLA-spring(~Month 14)或 ICSE-empirical |

### Differential testing

Differential testing 是**非形式 validation 工件,不属于可信证明链**。在 *选定的代表性* kernel 上,我们维护手写的 Python Triton 对应件,与 PyTorch / `flash-attn` reference 比较,以展示嵌入式算法对应可在 GPU 上运行的代码。

**它是什么** —— 一个 credibility 工件,帮助读者相信嵌入式 `triton { ... }` AST 对应一个真实的、可运行的 Triton kernel。Reviewer 可能(合理地)问:"形式证明的是不是真有人在跑的 kernel?" diff-test 工件 + 论文附录的 side-by-side correspondence 表,就是答案。

**它不是什么** —— 它**不是** phase gate。可信证明链是 Lean 定理对嵌入式操作语义的命题;正确性靠这个,不靠数值一致。

**它解决不了什么** —— 对手写 `.py` 跑 diff testing **不能**验证嵌入式操作语义本身,**也不能**关闭嵌入式 AST 与真实 `.py` 之间的 gap(AST↔`.py` 对应是人工断言,论文附录支撑;只有 Python lifter 才能形式化关闭 —— 那是 P3+ 工作)

它也不证明浮点正确性。float test 支撑的是 dtype-annotated kernel 与 Real
abstraction 之间的 trusted bridge;它不替代完整 IEEE-754 形式化。

**选定代表性集合**(整个 program 的 diff-test 集,共 ~3–4 个 kernel):
- 1–2 个 Tier 1+2 kernel(如最简单的 `naiveSoftmaxKernel`;一个 Tier 2 streaming kernel)
- FA forward(Phase C)
- FA-2 forward(Phase D)
- FA-1↔FA-2 互查(Phase D)

代表集之外的 Tier 1+2 kernel 仅靠证明保证 —— 加上去工程上线性增长,但额外样本对 credibility 的边际收益很小,只要代表通过就够。

**容差**:Tier 1+2 代表 vs PyTorch reference ≤ 1e-5 元素绝对差;FA forward 与 FA-2 forward vs `flash_attn_func` ≤ 1e-3(FA 参考自身 vs PyTorch 大致就是这个量级)

**失败处理**:diff-test 失败是黄旗,不是 gate —— 排查:IEEE-754 内禀差异(可接受,文档化)、`.py` 实现 bug(修)、对应断言不准(对照重检)。它**不**让形式证明失效;证明对嵌入式语义的命题仍然成立

### LLM benchmark protocol

让 close rate 指标可复现:

- **Held-out 集**:每个 Phase 提前圈 5–10 条引理,提交到 `bench/llm_eval/<phase>/` 独立文件结构;主库不依赖 held-out 文件。**圈定时机:每个 Phase 的*开头*,在该 Phase 的 LLM 工具调优*之前*** —— 防止"在 eval 上调工具"。Phase A 的 held-out 就是上面说的 `softmax_naive_correct` 副本。Phase B 与 Phase C 的 held-out 由工程师在该 Phase 开头从 #4–#6 与 #7 的子引理中挑选
- **每个 held-out theorem 允许的上下文**:held-out 文件的 imports + 该文件中已证(在 held-out 之前)的所有 theorem + Mathlib 引理名查找 + 工具自带的 retrieval(若有)。模型看到的就是 theorem statement、section 上下文、显式检索的引理。**不允许**看到:项目库其余内容(imports 之外)、held-out 引理本身的证明、任何针对该 held-out 引理的人工提示
- **eval 期间禁止**:人工 prompt iteration、人工 lemma name 提示、对 held-out 引理 fine-tune、对 LLM 输出做任何编辑后再跑。eval 是 hands-off:工具端到端跑,成功 / 失败,就一个数据点
- **试验数**:每条 held-out 引理 N=5 次独立运行,随机 seed 不同。Close rate = (全部 (lemma, seed) 对中成功的次数) / (5 × |held-out 集|)
- **Cost 报告**:整 phase 的 API call 数、token 数、wall-clock、美元成本 —— 每 phase 报告。包含失败重试
- **跨 Phase 允许的 prompt iteration**:一个 Phase 的 eval 跑完之后,prompt template 可以改;改过的 v0.X+1 工具用于*下一个 Phase 开头*重新圈定的*新* held-out 集

### Phase 间 gate 决策规则

**Gate A → B** —— scouting 必须显示 T3-B 在窗口内可行。NO → 缩到 T3-A only(失去 #8),Tier 2 扩 1–2 对(候选:scan reordering、RoPE 重排),重校时间。主定理总数变成 8–9 个(3 + 4-或-5 + 1)

**Gate B → C** —— online softmax recurrence 证明必须达 paper 质量。NO → 冻结于 `v0.2-tier2`,以 Tier 1+2 投 CGO/CC,投稿后再启动新一轮 Phase C

**Gate C → D** —— 时间预算检查
- Phase C ≤ 10 wk → 全套 Phase D
- 11–14 wk → 简化 Phase D(stripped FA-1 ↔ stripped FA-2:砍因果 mask、砍块跳过、简化 work partitioning)
- > 14 wk → 跳 Phase D,以 T3-A only 投 OOPSLA

"Stripped" 定义:保留算法核心(online softmax、`tl.dot`、输出累加器)但省略工程细节(无因果 mask、无 block-skip 优化、单 block grid)

**Gate D → 投稿** —— 全定理闭合 + LLM benchmark 数据齐 + paper draft 完成 + 代表性 kernel 的 diff-test 工件已就位(非 gating,但论文附录引用)。投稿。打 tag `v1.0-pldi`

### Open-source 节奏

- 主仓库:`github.com/Lizn-zn/VeriTile`(已公开)
- 主分支恒可构建;每个 phase 退出时打 tag:
  - `v0.1-tier1`、`v0.2-tier2`、`v0.3-tier3a`、`v1.0-pldi`
- 每个 release 配 release notes(新定理、新语义、benchmark 数据)
- README 持续维护双语(英 + 中)
- `CONTRIBUTING.md`,加 kernel-pair 的 tutorial(以 Tier 1 log-sum-exp 作为 worked example)
- `scripts/prove.sh` 留在仓库内(不单独打包);benchmark eval 结果与项目一起发布。底层的 `lean4` Claude Code 插件是 upstream,不 vendor —— `CONTRIBUTING.md` 记录 pin 的版本

### 显式 *未纳入* 范围(P3+)

写明白防止 scope creep + reviewer 期望落空:

- **Python lifter**(`.py` Triton → 嵌入式 `triton { ... }`):本论文手工嵌入。Lifter 是另一篇(empirical 导向)
- **IEEE-754 保真**:浮点用 `ℝ` 建模。Differential testing 是近似,不是位级等价。kernel 中使用的有限 mask sentinel(如 `-1e38`)是仅与 *可运行 Python kernel* 相关的 *concretization*;形式证明根据 Phase A 选定的方案使用扩展实数或 mask-predicate denotation
- **Multi-stream / async copy**:FA-1/FA-2 forward 不需要
- **Backward pass(FA-backward)**:本论文只做 forward。FA-1 vs FA-2 backward 是后续
- **Triton autotune 配置**:不影响算法等价
- **WGMMA / Hopper-specific 算子**:FA-1/FA-2 forward 不强依赖
- **并发 / 原子操作**:forward kernel 是纯函数;atomic 语义推迟

### 时间分布(粗略 Gantt)

```
Month               1   2   3   4   5   6   7   8   9   10
Phase A           [====]
Phase B               [========]
Phase C                        [========]
Phase D                                 [======]
scripts/prove.sh  [=]
Benchmark runs       [=]    [=]    [=]    [=]
Diff-test (代表) [==] [=]   [==]   [==]
Paper writing                               [====]

PLDI deadline (主)                                     ▲ (~Month 11)
OOPSLA-spring (fallback)                                  ▲ (~Month 14)
```

`scripts/prove.sh` 是 Phase A 一次性 ~1 天交付。Benchmark runs 在每 Phase 末(~1 天)。Diff-test 是断续工作(每 phase 一个代表性 kernel,~1 周)

## 决策日志

2026-04-26 brainstorm 中达成的关键决策:

1. **Bet 选择** —— 选 (b) "Verified kernel rewrites" + (e) "工具可用" 而非 (a) Vero-for-Triton、(c) 纯语义论文、(d) bug 研究。理由:在窗口内冲一档 venue 论文的概率最高,考虑到当前 artifact 状态(P1 无 sorry、LLM 工具未启、lifter 未启)
2. **Family 选择** —— Mix α(Attention-headlined)而非 β(norm-fusion-heavy)、γ(matmul-tiling)。理由:FA 的 online recurrence 有真正的数学深度,与 ATL/KaTen/TVM 的 compiler-rewrite 验证形成清晰区隔
3. **Tier 3 框架** —— T3-D(T3-A + T3-B 都做)而非 A-only / B-only / 仅 stripped。理由:用户承诺 ambitious scope,资源弹性;T3-A 作 Phase C 出口、T3-B 作 Phase D capstone,Gate C → D 设有缩水规则
4. **Sequencing** —— Approach 2(垂直切片)而非 1(自底向上)、3(tracer bullet)。理由:每个 phase 都有可投 artifact;Phase A 的 T3 scouting 把 tracer-bullet 的优势(早接触硬问题)吸收进来,同时不放弃 Tier 1 warmup 的价值
5. **LLM 工具 —— wrap,不造轮子** —— 选择封装现有的 `lean4` Claude Code 插件(`/lean4:autoprove`),做一个薄薄的 `scripts/prove.sh`,而非自建 Python 证明工具。从 200 行 Python 项目 + 多 phase 升级路线图(v0.2 retrieval、v0.3 proof-state interaction、v0.4 benchmark harness)缩到 ~80 行 bash wrapper。理由:插件已经提供 LSP 集成(`lean_multi_attempt`)、多 cycle 迭代 + 卡住检测、deep-mode 升级、tactic cascade、修复模式 —— 自己重新造是不必要的投资。代价:论文 "LLM tooling" 一节变成对一个现有社区工具的 evaluation,不是我们造的工具。诚实、less novel,但准确。决策时间:2026-04-26 Phase A 中期,在用户已安装的 Claude Code skills 里发现该插件之后
6. **Differential testing** —— *非形式 validation 工件*,不是 phase gate,不属于可信证明链。仅在选定代表性 kernel 上维护(全 program 共 ~3–4 个),作为 credibility 补充。理由:reviewer 必问"这是真 kernel 吗";我们用论文附录的 side-by-side correspondence + GPU 代表性证据回答。从早期草稿降级,因为意识到全量 diff-testing 是过度投资 —— 它不能验证 AST↔`.py` 对应,只有 Python lifter 能(P3+)
7. **Out-of-scope 决策** —— backward pass、IEEE-754 保真、Python lifter、原子操作、autotune 全部推迟。理由:每个潜在地都是另一篇 follow-up;捆绑会让本论文贡献失焦

## 各 phase 的 implementation plan

本文档是 program-level plan。每个 phase 各自需要带具体任务与排序的 implementation plan。Implementation plan 在每个 phase 启动时通过 `writing-plans` 流程编写:

- Phase A implementation plan —— 在 Phase A 启动前编写
- Phase B implementation plan —— 在 Gate A → B 时编写
- Phase C implementation plan —— 在 Gate B → C 时编写
- Phase D implementation plan —— 在 Gate C → D 时编写

每个 implementation plan 应回引本文档相关章节作为 context
