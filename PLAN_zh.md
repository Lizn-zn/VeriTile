# VeriTile — Program Plan

[English](PLAN.md) | **中文**

一个 7–10 个月的研究 program plan,通过 **8 个 Triton kernel 主正确性定理**(含 FlashAttention forward 正确性)+ **headline corollary `fa1_eq_fa2`** 交付一篇 "Verified kernel rewrites" 论文。

本文档是 2026-04-26 brainstorm 中达成一致的 program-level plan。下面每个 phase 各自有独立的 implementation plan(进入该 phase 时通过 `writing-plans` 流程编写)。

## 目标

**主目标** —— 投稿 PLDI(fallback OOPSLA-spring),核心 claim:

> 8 个生产 Triton kernel 主正确性定理(包括 FA-1 forward ≡ standard attention denotation 与 FA-2 forward ≡ standard attention denotation,推出 headline corollary FA-1 ↔ FA-2)在 Lean 4 中针对嵌入式 Triton 操作语义形式化证明,辅以 LLM 证明起草工具与 GPU 上的 differential testing 作为可信度支撑。

**副目标** —— 工具链(操作语义、嵌入式 `triton { ... }` DSL、LLM 证明 harness、differential testing)做扎实,可内部使用、可开源接收外部贡献。

## 范围:8 个主正确性定理 + helpers + corollary

| Tier | Phase | 主定理 | 类型 |
|---|---|---|---|
| 1 | A | (#1) `softmax_kernels_refinement`(已完成)·(#2) `log_sum_exp_refinement`·(#3) `softmax_reciprocal_refinement` | kernel ↔ kernel × 3 |
| 2 | B | (#4) `welford_kernels_refinement`·(#5) `online_softmax_recurrence_eq_batch`·(#6) `layernorm_kernels_refinement` | kernel ↔ kernel × 3 |
| 3 | C, D | (#7) `fa_forward_correct`(T3-A,Phase C)·(#8) `fa_2_forward_correct`(T3-B,Phase D) | kernel ↔ math × 2 |

**外加 helpers**(不计入主定理):
- 数学引理 `welford_eq_two_pass`(Phase A,#4 的准备)
- 操作语义 helper `forLoop_inv`(Phase B,所有循环 kernel 证明的工作母机)
- Mask 处理 helper(Phase C;具体形式取决于 Phase A scouting 结果 —— 见 §Phase C)
- `delayed_rescale_eq` helper(Phase D,在 #8 中使用)

**外加 headline corollary**(派生,**不是第 9 个主定理**):
- **`fa1_eq_fa2 : Y_fa1 = Y_fa2`** —— 由 #7 与 #8 通过 `standardAttentionMath` 的 spec 传递性导出(~30 行)。**这是论文 foreground 的句子**,但内容已经被 #7 与 #8 完全包含。

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
- 在主库之外建独立的 `bench/llm_eval/` 目录,把 `softmax_naive_correct` 复制一份并把证明体替换为 `sorry`(主库的版本保持 intact 且已证)。LLM MVP 必须在这个 held-out 副本上 close rate ≥ 1/3,N=5 次独立运行(见 §LLM benchmark protocol)
- P1 + Tier 1 全部 build 干净

**Exit gate(`v0.1-tier1`):** 全部交付物达成;`lake build` 干净

### Phase B — forLoop 语义 + Tier 2(~8–12 wk)

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
- LLM v0.2 在 Phase B held-out 集上 close rate ≥ 50%(见 §LLM benchmark protocol)
- Differential testing 6 个 Tier 1+2 kernel 全部通过

**Exit gate(`v0.2-tier2`):** 全部交付物达成;Phase C 推迟时可投 CGO/CC

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

*LLM 工具 v0.3:*
- Proof-state interaction:抓取 Lean 中间证明状态(经 Lean MCP server 或直接 `lean --server`),失败时把当前 goal 喂回模型,实现 step-wise tactic 生成
- 对 ~200 行长证明是必需的(v0.2 的一次性模式不够 scale)

*Differential testing 扩展:*
- 每个嵌入式 `triton { ... }` kernel **配一个手写的对应 Python Triton kernel**,放 `tools/diff_test/python/`。两边**不是**自动互生 —— 对应关系由人工作者声明并审阅(Python lifter 是 P3+ 工作;见 §未纳入范围)
- 跑手写 Python kernel,GPU 上对 `flash-attn` 包的 `flash_attn_func` 比较
- 因果 / 非因果两种
- 容差 ≤ 1e-3(FA 参考实现 vs PyTorch reference 大致就是这个量级)
- **对应关系声明**(论文中必须显式表述):附录给出 Lean AST 指令到 Python 对应的 side-by-side 表 + CI 检查两边 kernel signature / shape 一致。Reviewer 会问"形式证明的是不是你跑的那个 kernel" —— 这张表 + CI 是答案

**验证:**
- LLM v0.3 在 Phase C held-out 集上 close rate ≥ 30%(见 §LLM benchmark protocol;held-out 引理是 `fa_forward_correct` 的子步骤)
- Differential testing FA forward ↔ `flash_attn_func` 通过

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

*LLM 工具 v0.4(release-ready):*
- Benchmark suite:Tier 1+2+3 全部定理打包为 evaluation set,自动测 close rate / 平均 retry 次数 / wall-clock / API 成本
- Parallel sampling:多个候选 tactic 并行生成,第一个通过的获胜
- Cost tracking:整个论文实验聚合 API tokens / cost / 时间,在论文中报告
- CLI polish:`prove --theorem foo --max-retries 5 --strategy retrieval+stepwise` 风格接口
- 独立发布为 PyPI 包(`lean-llm-prover`,具体名字看可用性)

*Differential testing 收尾:*
- 完整 differential testing 表:
  - 6 个 Tier 1+2 kernel-pair 测试,每对 kernel ↔ kernel 对 PyTorch reference + 两 kernel 互相比较
  - FA-1 forward vs `flash_attn_func` / PyTorch reference
  - FA-2 forward vs `flash_attn_func` / PyTorch reference
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
- 完整 differential testing 表通过(6 个 Tier 1+2 kernel pair + FA-1 + FA-2 + FA-1 vs FA-2)
- 论文 draft 完成并提交

**Exit gate(`v1.0-pldi`):** 论文提交;打 tag 发布

## Cross-cutting

### 风险登记表

| Risk | Phase | 触发信号 | Mitigation |
|---|---|---|---|
| `forLoop_inv` 接口不能干净支持嵌套循环 | B | Phase A scouting 在 invariant 草拟时发现 | 退化方案:`Nat.iterate` 形式(不直观但等价语义) |
| Online softmax recurrence Lean 形式化卡死 | B | 持续 ≥ 2 周无进展 | 强力调用 LLM 工具;再不行简化为 stripped 版(从此证明中砍 mask,Phase C 再加回) |
| `tl.dot` 在 `Value.tile2D` 上 simp 表现糟糕 | C | `fa_forward` 证明卡 simp | 自定义 simp 引理(P1 已为 `Value.bop` 等长 tile-tile 做过先例);simp 不合作就改用 `unfold + induction` 风格 |
| Mask sentinel-vs-real-zero gap(`Real.exp(-1e38) ≠ 0`)阻塞天真的 mask 形式化 | A scouting / C | Phase A 写 pseudocode 时发现 | 选 Option α(扩展实数:`WithBot ℝ` / `EReal`)或 Option β(mask-predicate denotation);Phase A 拍板 |
| 2D stride / layout 形式模型超预期复杂 | D | 起草 `multiBlockExec` 不变量时 | 限定为"每 program_id 行优先连续";论文中作为 assumed layout 文档化 |
| FA-2 multi-block 抽象超预期复杂 | D | Phase C 末草拟 `multiBlockExec` 时暴露隐藏复杂度 | T3-D → T3-A only;论文目标降档 OOPSLA |
| LLM 工具无法处理 ~200 行长证明 | C | Phase B 末保留引理 close rate < 30% | 诚实报告:LLM 只在短证明上有效;不强用于长证明;工具定位变为"证明片段助手"而非"完整证明生成器" |
| Differential testing 数值差 > 1e-3 即使收紧 kernel 实现 | B–D | GPU 输出 vs reference 超容差 | 三种排查:IEEE-754 内禀差异(可接受,文档标注);kernel 实现 bug(修);我方嵌入语义 bug(修并重证) |
| 错过 PLDI deadline(~Month 11) | D | Phase C 末时间评估 | 投 OOPSLA-spring(~Month 14)或 ICSE-empirical |

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
- **IEEE-754 保真**:浮点用 `ℝ` 建模。Differential testing 是近似,不是位级等价。kernel 中使用的有限 mask sentinel(如 `-1e38`)是仅与 *可运行 Python kernel* 相关的 *concretization*;形式证明根据 Phase A 选定的方案使用扩展实数或 mask-predicate denotation
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
