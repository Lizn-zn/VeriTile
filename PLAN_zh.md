# VeriTile — Program Plan

[English](PLAN.md) | **中文**

一个 7–10 个月的研究 program plan,目标是通过 8 个生产相关的 Triton kernel 等价性定理(含 FlashAttention forward 正确性 + FA-1 ↔ FA-2 verified rewrite)交付一篇 "Verified kernel rewrites" 论文。

本文档是 2026-04-26 brainstorm 中达成一致的 program-level plan。下面每个 phase 各自有独立的 implementation plan(进入该 phase 时通过 `writing-plans` 流程编写)。

## 目标

**主目标** —— 投稿 PLDI(fallback OOPSLA-spring),核心 claim:

> 8 个生产相关 Triton kernel 等价定理(包括 FlashAttention forward ≡ standard attention denotation 与 FlashAttention-1 ↔ FlashAttention-2)在 Lean 4 中针对嵌入式 Triton 操作语义形式化证明,辅以 LLM 证明起草工具与 GPU 上的 differential testing 作为可信度支撑。

**副目标** —— 工具链(操作语义、嵌入式 `triton { ... }` DSL、LLM 证明 harness、differential testing)做扎实,可内部使用、可开源接收外部贡献。

## 范围:Kernel 对集合(8 个定理)

分布:Tier 1(3)+ Tier 2(3)+ Tier 3(2)。其中 7 个是 kernel ↔ kernel 等价定理,1 个(Tier 3-A)是 kernel ↔ math-spec 正确性定理。

### Tier 1 — Loop-free(Phase A,3 对)

1. **`softmax_kernels_refinement`** —— 已完成 —— naive ↔ 数值稳定 softmax
2. **`log_sum_exp_refinement`** —— direct LSE kernel ↔ shift-trick LSE kernel(`logsumexp(x) = m + log(Σ exp(x − m))`,`m = max(x)`)
3. **`softmax_reciprocal_refinement`** —— `y = e/s` 逐元素除法 kernel ↔ `inv_s = 1/s; y = e * inv_s` 预算倒数 kernel(省 N−1 次除法;数学上轻量,展示 fused-multiply 重写模式)

外加一个 *math-only* 引理 `welford_eq_two_pass`(为 Phase B 的 Welford kernel 定理做准备;不计入 kernel 对总数)

### Tier 2 — Streaming reductions(Phase B,3 对)

4. **Welford ↔ two-pass variance**(kernel-level)—— Welford 用 `forLoop` 跨 tile 维护 `(M, S, n)`;two-pass 用两次 `tl.sum`。把 Phase A 的数学引理上升到真 kernel
5. **Online softmax recurrence ≡ batch softmax** —— *FlashAttention 算法核心,论文中心*。证明流式递推 `(m_new, l_new) = (max(m_old, max(x_block)), exp(m_old − m_new) · l_old + Σ exp(x_block − m_new))` 产生与一次性 batch 形式相同的 `(m, l)`,通过对 block 数归纳
6. **Fused single-pass LayerNorm ≡ two-pass LayerNorm** —— 用 Welford 引理 + affine `(x − μ)/√(var + ε) · γ + β` 变换。展示同一 lemma 跨 kernel 对的复用

### Tier 3 — 生产 attention(Phase C+D,2 个定理)

7. **`fa_forward_correct`**(T3-A,Phase C)—— FlashAttention forward kernel ≡ `standardAttentionMath Q K V causal` denotation。Single-block 推理:每个 program_id 处理一块 Q × 完整 KV 循环。把 Phase B 的 `(m, l)` invariant 扩展为 `(m, l, O)` 加上输出累加器
8. **`fa1_eq_fa2`**(T3-B,Phase D)—— FA-1 ↔ FA-2 verified rewrite,通过 `standardAttentionMath` 的 spec 传递性。需要 multi-block grid execution 语义(FA-2 在序列长度上并行化)

## Phase 结构(总计 ~28–42 周)

### Phase A — Tier 1 + LLM tool MVP + T3 scouting(~6–8 wk)

**交付物:**
- 3 个 Tier 1 kernel-pair 定理闭合(#1 已完成,关闭 #2 与 #3)
- 1 个 math-only 引理(`welford_eq_two_pass`)
- LLM 证明工具 MVP:~200 行 Python 脚本,接口 `prove.py file_path theorem_name`,循环 prompt → API → 写回 → `lake env lean` 验证 → 失败回填重试,N=5 次
- T3 scouting 文档 `Notes/T3_scouting.md`(~10–15 页):
  - FA forward 伪代码 + invariant 草稿
  - B/C/D 语义扩展所需的 lemma 清单
  - 显式风险登记表(未预期的形式化 gap)
  - 可行性判断:T3-B 在窗口内可行,还是只做 T3-A?

**验证:**
- LLM MVP 必须能在 `softmax_naive_correct`(为测试制造 sorry)上的 close rate ≥ 1/3,可重复
- P1 + Tier 1 全部 build 干净

**Exit gate(`v0.1-tier1`):** 全部交付物达成;`lake build` 干净

### Phase B — forLoop 语义 + Tier 2(~8–12 wk)

**交付物:**

*操作语义:*
- 把 `stepStmt` 与 `stepStmts` 改为 `mutual` 块,以支持嵌套循环
- `forLoop` 操作语义,显式 `termination_by (sizeOf body + 1, n − i)` lex 度量
- `forLoop_inv` 引理(Phase B/C/D 所有循环证明的工作母机):

  ```
  ∀ (P : Nat → BlockState → Prop),
    P 0 s_init →
    (∀ i s, P i s → ∃ s', stepStmts body s = some s' ∧ P (i+1) s') →
    ∃ s_final, exec_forLoop body n s_init = some s_final ∧ P n s_final
  ```

*定理:*
- 3 个 Tier 2 kernel-pair 定理(#4、#5、#6)
- Tier 2 #5(online softmax recurrence)证明预估 ~80–120 行 Lean;**这是论文核心技术贡献**

*LLM 工具 v0.2:*
- Lemma retrieval:对 Mathlib + 项目内引理做向量 / BM25 检索,top-k 注入 prompt
- Smarter retry:维护"已失败策略"集合,prompt 时显式排除
- Structured output:模型输出 JSON `{tactic_block, reasoning}`,先做语法验证再写回文件

*Differential testing harness:*
- `tools/diff_test/` 目录
- 每个 Tier 1 + Tier 2 kernel 配 `.py` Triton 实现(从开源工程如 unsloth、vllm 抄,或参照 Triton tutorial 自写)
- 两个 `.py` kernel 的输出对 PyTorch reference 比较;两个互相比较
- 容差:逐元素绝对差最大值 ≤ 1e-5

**验证:**
- LLM v0.2 在保留的 Tier 2 引理(如 LayerNorm fusion 的某个子步骤)上 close rate ≥ 50%
- Differential testing 6 个 Tier 1+2 kernel 全部通过

**Exit gate(`v0.2-tier2`):** 全部交付物达成;Phase C 推迟时可投 CGO/CC

### Phase C — `tl.dot` + masking + Tier 3-A(~8–12 wk)

**交付物:**

*语义扩展:*
- `Value.tile2D : (m n : Nat) → (Fin m → Fin n → ℝ) → Value` 构造子
- `Op.dot a b` 语义:`(A · B)[i,k] = Σⱼ A[i,j] · B[j,k]`,通过 Mathlib `Finset.sum` 表达
- `Op.where(cond, a, b)` 用于 masking
- 引理 `softmax_neg_inf_zero`:`score = −∞`(或有限替身如 `-1e38`)的位置在 softmax 中权重为 0
- Single-block 推理已足够(FA-1 forward 与 FA-2 forward 都是每个 program_id 处理一个 (batch, head) 或 (batch, head, Q-block);program 内证明覆盖算法内容)

*定理:*
- `fa_forward_correct` —— 完整 Lean 证明,无 sorry
- 命题(草稿):

  ```
  theorem fa_forward_correct
      (Q K V : Matrix (Fin S) (Fin D) ℝ) (causal : Bool)
      (s : BlockState) (h_inputs : InputsLoaded s Q K V) :
      ∀ (i : Fin S),
        observeY (exec FAForwardKernel s) S s.pid i
          = some (standardAttentionMath Q K V causal i)
  ```

- 证明思路:per Q block 的 forLoop_inv,invariant `(m_k, l_k, O_k)` 扩展自 Phase B 的 `(m, l)` 递推。预估 ~200–300 行 Lean(含语义扩展)

*LLM 工具 v0.3:*
- Proof-state interaction:抓取 Lean 中间证明状态(经 Lean MCP server 或直接 `lean --server`),失败时把当前 goal 喂回模型,实现 step-wise tactic 生成
- 对 ~200 行长证明是必需的(v0.2 的一次性模式不够 scale)

*Differential testing 扩展:*
- 把 FA forward 嵌入 `triton { ... }` 后转译为可运行 `.py`,GPU 上对 `flash-attn` 包的 `flash_attn_func` 比较
- 因果 / 非因果两种
- 容差 ≤ 1e-3(FA 参考实现 vs PyTorch reference 大致就是这个量级)

**验证:**
- LLM v0.3 在保留的 Tier 3-A 子引理上 close rate ≥ 30%
- Differential testing FA forward ↔ `flash_attn_func` 通过

**Exit gate(`v0.3-tier3a`):** 全部交付物达成;Phase D 推迟或缩水时可投 OOPSLA

### Phase D — multi-block + FA-1 ↔ FA-2 + paper(~6–10 wk)

**交付物:**

*语义扩展:*
- `multiBlockExec : Kernel → InitMem → Grid → FinalMem` 模型:跑遍所有 program_id,组合写入
- Disjoint-writes 引理:无 atomic 的纯前向 kernel,各 program_id 输出区域不相交,可独立分析每个 program 的局部正确性后再组合
- ~80 行语义扩展;**有意不引入并发 / 原子**(本论文不做)

*Kernel 与定理:*
- FA-2 kernel 通过 `triton { ... }` 嵌入
- `fa_2_forward_correct` —— 类似 Phase C 的 `fa_forward_correct`,但针对 FA-2 的:(a) 序列长度并行化,(b) 延迟 rescaling,(c) 全 mask 块跳过
  - 延迟 rescale 等价:`O_final / l_final` 不依赖中间 `O` 是否每步 rescale,独立小引理
  - 块跳过正确性:`−∞`-mask 块贡献为 0(复用 `softmax_neg_inf_zero`)
  - 预估 ~150–200 行新代码;大量复用 Phase C invariant 工具
- `fa1_eq_fa2` 通过 spec 传递性的 corollary(~20 行)

*LLM 工具 v0.4(release-ready):*
- Benchmark suite:Tier 1+2+3 全部定理打包为 evaluation set,自动测 close rate / 平均 retry 次数 / wall-clock / API 成本
- Parallel sampling:多个候选 tactic 并行生成,第一个通过的获胜
- Cost tracking:整个论文实验聚合 API tokens / cost / 时间,在论文中报告
- CLI polish:`prove --theorem foo --max-retries 5 --strategy retrieval+stepwise` 风格接口
- 独立发布为 PyPI 包(`lean-llm-prover`,具体名字看可用性)

*Differential testing 收尾:*
- 完整 7 对表(不计入 math-only `welford_eq_two_pass`):每对 kernel 对 PyTorch reference + 互相比较
- FA-1 vs FA-2 互查:相同输入,输出在容差内一致

*论文 draft:*

10 节大纲:

1. Introduction —— 动机:LLM-driven kernel 重写 + 验证缺口
2. Background —— Triton、arm-in-lean、Vero、ATL
3. 嵌入式 Triton 子集与操作语义
4. Algorithmic equivalence framework(refinement 模式、gather/scatter)
5. **Online softmax recurrence**(Phase B 中心)
6. **FA forward correctness**(Phase C)
7. **FA-1 ↔ FA-2 verified rewrite**(Phase D 核心)
8. LLM-assisted proof tooling(Phase A→D 演化 + benchmark 数据)
9. Differential testing evaluation
10. Related work + Conclusion

**验证:**
- 全部定理闭合,`lake build` 干净
- LLM v0.4 benchmark 数据可重现
- 完整 7 对 differential testing 通过
- 论文 draft 完成并提交

**Exit gate(`v1.0-pldi`):** 论文提交;打 tag 发布

## Cross-cutting

### 风险登记表

| Risk | Phase | 触发信号 | Mitigation |
|---|---|---|---|
| `forLoop_inv` 接口不能干净支持嵌套循环 | B | Phase A scouting 在 invariant 草拟时发现 | 退化方案:`Nat.iterate` 形式(不直观但等价语义) |
| Online softmax recurrence Lean 形式化卡死 | B | 持续 ≥ 2 周无进展 | 强力调用 LLM 工具;再不行简化为 stripped 版(从此证明中砍 mask,Phase C 再加回) |
| `tl.dot` 在 `Value.tile2D` 上 simp 表现糟糕 | C | `fa_forward` 证明卡 simp | 自定义 simp 引理(P1 已为 `Value.bop` 等长 tile-tile 做过先例);simp 不合作就改用 `unfold + induction` 风格 |
| FA-2 multi-block 抽象超预期复杂 | D | Phase C 末草拟 `multiBlockExec` 时暴露隐藏复杂度 | T3-D → T3-A only;论文目标降档 OOPSLA |
| LLM 工具无法处理 ~200 行长证明 | C | Phase B 末保留引理 close rate < 30% | 诚实报告:LLM 只在短证明上有效;不强用于长证明;工具定位变为"证明片段助手"而非"完整证明生成器" |
| Differential testing 数值差 > 1e-3 即使收紧 kernel 实现 | B–D | GPU 输出 vs reference 超容差 | 三种排查:IEEE-754 内禀差异(可接受,文档标注);kernel 实现 bug(修);我方嵌入语义 bug(修并重证) |
| 错过 PLDI deadline(~Month 11) | D | Phase C 末时间评估 | 投 OOPSLA-spring(~Month 14)或 ICSE-empirical |

### Phase 间 gate 决策规则

**Gate A → B** —— scouting 必须显示 T3-B 在窗口内可行。NO → 缩到 T3-A only,Tier 2 扩 1–2 对(候选:scan reordering、RoPE 重排),重校时间。Kernel 对总数仍 7–8

**Gate B → C** —— online softmax recurrence 证明必须达 paper 质量。NO → 冻结于 `v0.2-tier2`,以 Tier 1+2 投 CGO/CC,投稿后再启动新一轮 Phase C

**Gate C → D** —— 时间预算检查
- Phase C ≤ 10 wk → 全套 Phase D
- 11–14 wk → 简化 Phase D(stripped FA-1 ↔ stripped FA-2:砍因果 mask、砍块跳过、简化 work partitioning)
- > 14 wk → 跳 Phase D,以 T3-A only 投 OOPSLA

"Stripped" 定义:保留算法核心(online softmax、`tl.dot`、输出累加器)但省略工程细节(无因果 mask、无 block-skip 优化、单 block grid)

**Gate D → 投稿** —— 全定理闭合 + LLM benchmark 数据齐 + 完整 diff-testing 通过 + paper draft 完成。投稿。打 tag `v1.0-pldi`

### Open-source 节奏

- 主仓库:`github.com/Lizn-zn/VeriTile`(已公开)
- 主分支恒可构建;每个 phase 退出时打 tag:
  - `v0.1-tier1`、`v0.2-tier2`、`v0.3-tier3a`、`v1.0-pldi`
- 每个 release 配 release notes(新定理、新语义、benchmark 数据)
- README 持续维护双语(英 + 中)
- `CONTRIBUTING.md`,加 kernel-pair 的 tutorial(以 Tier 1 log-sum-exp 作为 worked example)
- LLM 工具在 Phase D 末作为 PyPI 独立包发布

### 显式 *未纳入* 范围(P3+)

写明白防止 scope creep + reviewer 期望落空:

- **Python lifter**(`.py` Triton → 嵌入式 `triton { ... }`):本论文手工嵌入。Lifter 是另一篇(empirical 导向)
- **IEEE-754 保真**:浮点用 `ℝ` 建模。Differential testing 是近似,不是位级等价
- **Multi-stream / async copy**:FA-1/FA-2 forward 不需要
- **Backward pass(FA-backward)**:本论文只做 forward。FA-1 vs FA-2 backward 是后续
- **Triton autotune 配置**:不影响算法等价
- **WGMMA / Hopper-specific 算子**:FA-1/FA-2 forward 不强依赖
- **并发 / 原子操作**:forward kernel 是纯函数;atomic 语义推迟

### 时间分布(粗略 Gantt)

```
Month            1   2   3   4   5   6   7   8   9   10
Phase A        [====]
Phase B            [========]
Phase C                     [========]
Phase D                              [======]
LLM tool dev   [=============================]
DiffTest infra     [=========================]
Paper writing                            [====]

PLDI deadline (主)                                  ▲ (~Month 11)
OOPSLA-spring (fallback)                              ▲ (~Month 14)
```

## 决策日志

2026-04-26 brainstorm 中达成的关键决策:

1. **Bet 选择** —— 选 (b) "Verified kernel rewrites" + (e) "工具可用" 而非 (a) Vero-for-Triton、(c) 纯语义论文、(d) bug 研究。理由:在窗口内冲一档 venue 论文的概率最高,考虑到当前 artifact 状态(P1 无 sorry、LLM 工具未启、lifter 未启)
2. **Family 选择** —— Mix α(Attention-headlined)而非 β(norm-fusion-heavy)、γ(matmul-tiling)。理由:FA 的 online recurrence 有真正的数学深度,与 ATL/KaTen/TVM 的 compiler-rewrite 验证形成清晰区隔
3. **Tier 3 框架** —— T3-D(T3-A + T3-B 都做)而非 A-only / B-only / 仅 stripped。理由:用户承诺 ambitious scope,资源弹性;T3-A 作 Phase C 出口、T3-B 作 Phase D capstone,Gate C → D 设有缩水规则
4. **Sequencing** —— Approach 2(垂直切片)而非 1(自底向上)、3(tracer bullet)。理由:每个 phase 都有可投 artifact;Phase A 的 T3 scouting 把 tracer-bullet 的优势(早接触硬问题)吸收进来,同时不放弃 Tier 1 warmup 的价值
5. **LLM 工具集成** —— 显式贯穿全 phase 的交付物(MVP → harness),非可选。理由:用户明确"肯定用 LLM 写证明";工具按 phase 演化,在 Phase D 时成为论文 secondary contribution
6. **Differential testing** —— 全 phase 必备的 credibility 基础设施,与形式证明分离。理由:reviewer 必问"这是真 kernel 吗";我们用 GPU 输出对 reference 一致来回答,而形式证明留在 `ℝ` 上
7. **Out-of-scope 决策** —— backward pass、IEEE-754 保真、Python lifter、原子操作、autotune 全部推迟。理由:每个潜在地都是另一篇 follow-up;捆绑会让本论文贡献失焦

## 各 phase 的 implementation plan

本文档是 program-level plan。每个 phase 各自需要带具体任务与排序的 implementation plan。Implementation plan 在每个 phase 启动时通过 `writing-plans` 流程编写:

- Phase A implementation plan —— 在 Phase A 启动前编写
- Phase B implementation plan —— 在 Gate A → B 时编写
- Phase C implementation plan —— 在 Gate B → C 时编写
- Phase D implementation plan —— 在 Gate C → D 时编写

每个 implementation plan 应回引本文档相关章节作为 context
