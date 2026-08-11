# VeriTile 验证 benchmark

本文件夹是 VeriTile kernel-verification benchmark 的所在地 ——
VeriTile 承诺要验证的 Triton kernel 集合,跟踪来源、tier、masking variant
与 proof 状态。

> 状态(2026-05-05):**TritonBench-G v1** 选作外部 anchor。
> 静态覆盖分析已 land 在
> [`tritonbench_coverage.md`](./tritonbench_coverage.md)。导入的上游
> Python kernel 放在 [`tritonbench_g/`](./tritonbench_g/);Lean port
> 放在 `VeriTile/Examples/`。Per-kernel manifest 现在是
> [`scripts/kernel-manifest.tsv`](../scripts/kernel-manifest.tsv);
> per-port proof-status 表会随 port 落地填充。

## 当前状态

目前还没有正式的 benchmark manifest。事实上的 benchmark 是
`VeriTile/Examples/` —— VectorAdd、Softmax、OnlineSoftmax、LayerNorm、
Welford、FusedSiLU、ApproxGeLU、RowWise、GridComposition、FlashAttention-1
forward 跨 dense / causal / boundary / D-tail variant,加上 smoke 与
frame 文件。它能支撑开发,但缺少 source attribution、版本化 manifest
和发布的进度表 —— 所以目前不能作为外部读者可见的 benchmark *artifact*。

之前的 `bench/llm_eval/` LLM-proof-drafting eval 在 2026-05-05 退役;
LLM-assist 评估不再是项目的 benchmark 维度。

## TritonBench-G v1 anchor

我们正在把 verification benchmark 与 [TritonBench-G v1][tb] 对齐
(184 个 GitHub-scraped 真实 Triton kernel,ACL 2025 Findings)。
**184 个里已移植 155 个。** 最初(2026-05-05)的静态 primitive scan 只估出
141 个在 DSL 契约内;它点名的那些杠杆(`tl.math.*` / `tl.extra` adapter、
concurrency 边界、`tl.num_programs`、`atomic_add` proof shape)此后都已落地,
所以那组估计已被取代 —— 它是历史,不是现状。

剩下的 **29 个是"占位目录建了、上游 `.py` 从未导入"**。按 DSL 的**实际** surface
(95 个 `tl.*` 形式,从 `VeriTile/Triton/DSL/**` 提取)重新实测:

| 判定 | 数量 |
|---|---:|
| 现在就能移植 —— 用到的一切形式全在 DSL 里 | 1 |
| 被缺失 primitive 或 ℝ 模型限制阻塞 | 28 |

那 28 个的解锁杠杆按产出排序:fp8 dtype channel(7)、**降序 `for` range**(4,
其首个消费者 `chunk_linear_attn` 已通过升序换元零库改动落地)、
RNG(4)、有符号定宽整数算术(3)、`Stmt` 里的 `while` 语句(3)、
`tl.interleave`(2)、整数通道 `tl.dot`(2)、`tl.static_assert`(2,宏层 no-op)、
`tl.broadcast_to`(1,别名)、IEEE inf/NaN + `libdevice.isfinited`(1)。逐 kernel
表、测法,以及"可移植"到底声称了什么和没声称什么,见
[`tritonbench_coverage.md`](./tritonbench_coverage.md) —— 包括 2026-08-10 那次把
`现在就能移植` 从 9 改成 2 的复核:此前的判定是按"每个文件只看一个 jit kernel"
形成的。

判定是 *可表达性*,不是 *proof 可行性*:许多可表达的 kernel
(FA-1 backward、RWKV6、Mamba SSM、chunked GLA)即便 primitive 今天能
lower,仍需要全新的 proof 工程。

[tb]: https://github.com/thunlp/TritonBench/tree/main/data/TritonBench_G_v1

## 外部 Triton kernel 语料

可作为 verification target 的现实世界 Triton 来源调研。Attribution
与规模在 2026-05-05 时按 arXiv / repo README 核对。

| 语料 | 规模 | 特征 | 与 VeriTile 的契合 |
|---|---|---|---|
| **TritonBench-G**(THUNLP / 清华,ACL 2025 Findings;arXiv 2502.14752)| 184 个 GitHub-scraped 真实 Triton kernel | 最贴近 in-the-wild "production Triton" | **2026-05-05 选作 VeriTile 的外部 anchor。** 静态扫描显示 184 个 kernel 中,在两个 trivial surface adapter 后 77% 处于当前 DSL 契约内;concurrency 边界成本真实但有界(184 中约 11)。见 `tritonbench_coverage.md`。 |
| **TritonBench-T**(同篇)| PyTorch 接口对齐的 kernel(摘要未给精确数;查 repo)| 提供 reference 实现 | 与 Compute-layer differential-testing 路线配合很好。 |
| **liger-kernel**(LinkedIn;arXiv 2410.10989)| LLM 训练 kernel 集合 | RMSNorm、RoPE、SwiGLU、CrossEntropy、FusedLinearCrossEntropy、…(已与 HF-transformers 集成)| **与 VeriTile Tier 4 路线图重叠最高。** 单文件、上游有测试、训练相关、外部维护。 |
| **FlagGems**(BAAI / FlagOpen)| ~184 op,目标 216(README)| Triton 实现的 PyTorch op 覆盖 | 许多逐元素 op 在 Tier 1 / Tier 2 自然落地;批量参差。 |
| **Unsloth kernel**(unslothai/unsloth)| 训练 fast-path kernel | RoPE、MLP、NF4 dequant、cross-entropy、RMSNorm、SwiGLU、GeGLU | liger 的训练侧补集。 |
| KernelBench(斯坦福;arXiv 2502.10517,2025 年 2 月)| 250 个 PyTorch 任务,横跨 L1(100 单 op)/ L2(100 fused)/ L3(50 architecture);+ 20 个 L4 HF aspirational | **生成** benchmark(Torch → CUDA),不是 verification | 不是 verification target。 |

## 方向

按顺序:

1. **先 land 两个 trivial surface adapter** —— `tl.math.*`
   (`exp2/log2/rsqrt/sin`)与 `tl.extra.cuda.libdevice.*`
   (`pow/tanh/llrint`)。这些是句法的;语义已存在于 VeriTile 现有
   spelling 下。两者一起把 33 个 TritonBench-G 文件从 Soft/Hard 移到 OK,
   零语义成本。

2. **port 第一个具体的 TritonBench-G kernel。** 在 `Examples/` 之外挑
   一个小型 OK 判定文件 —— 候选:`vector_addition.py`、
   `softmax_triton1.py`、`rope_embedding.py`、`matmul_kernel.py`。用这次
   port 设计 per-kernel metadata schema(source URL + commit、选定的
   BLOCK_* 配置、丢弃的 hint、public theorem symbol、判定)。

3. **版本化 manifest。** 一旦一个外部 kernel 入库,冻结 schema 并在
   `Examples/` 与新的 TritonBench-G port 上回填。manifest 是项目 README
   进度表的真理来源。

4. **按产出顺序攻克补救**,按
   [`tritonbench_coverage.md`](./tritonbench_coverage.md):
   `tl.num_programs`(9 文件)、atomic_add proof shape(6),然后是更大
   的语义投资 —— concurrency 边界(#12)、FP8 channel、int4 packed、
   RNG(#41)、FP4。

## 已记录的取舍

- **TritonBench-G anchor。** 早期草稿偏向先做 liger-kernel,因为
  TritonBench-G "第一天就会撞上 concurrency 边界"。实际扫描否定了这点:
  184 个 kernel 中只有 11 个(`tl.debug_barrier` x8 + `atomic_cas/xchg`
  x3)处于 concurrency 边界。77% OK 数字让 TritonBench-G 成为可行的
  对外可读 primary benchmark,而不仅是 stress test。
- **静态判定。** "OK" 证明 *可表达性* 在当前 DSL 契约下,而不是
  *proof 可行性*。许多 OK kernel 仍需新 proof 工程。这反映在 headline
  数字以及 `tritonbench_coverage.md` §Caveats。
- **首个 port 之后再定 manifest。** Per-kernel metadata schema 由首个
  真实外部 port 塑形,而不是预先猜测。
- **无 LLM-eval 维度。** 项目不再 benchmark LLM proof-drafting。如果
  它回归,作为独立子 benchmark,而不是 *the* benchmark。
