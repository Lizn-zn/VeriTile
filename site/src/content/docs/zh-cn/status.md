---
title: 项目状态
description: VeriTile 当下状况 —— bench 覆盖率、各块卡在哪、哪些设计文档是 current 的。
---

VeriTile 的当下快照,根据下方日期的仓库实测得出。所有数字来自
`bench/tritonbench_g/` 和 `git log` 的直接扫描,不是 manifest。

:::tip[最近核验]
下文数字针对 `main` 在 **2026-05-19** 核验过。要刷新,重跑
[`scripts/check-artifact.sh`](https://github.com/Lizn-zn/VeriTile/blob/main/scripts/check-artifact.sh)
和本页顶部的统计命令。
:::

## Bench 语料

| 指标 | 值 |
|---|---|
| `bench/tritonbench_g/<kernel>/` 目录数 | **184** |
| 已配对 `.py` + `.lean` | **141** |
| README-only 脚手架(尚未 port) | **43** |
| 全 bench `ComputeCorrect.Realizes_without_Rounding` 证明数 | **742** |
| 全 bench `ComputeRefine.Refines_without_Rounding` 证明数 | 0 |
| 全 bench `sorry` / `admit` | **0** |
| `VeriTile/` + `bench/` 定理 + 引理总数 | **2,567** |

零 `sorry` / 零 `admit` 由 CI 中的
[`scripts/check-artifact.sh`](https://github.com/Lizn-zn/VeriTile/blob/main/scripts/check-artifact.sh)
强制保证。加 `sorry` 会让 build 失败。

bench 中 `ComputeRefine.Refines_without_Rounding` 为零的原因:refinement surface 存在,
但目前每个 bench port 都是 "kernel ↔ 数学 spec" 而不是 "kernel ↔ kernel",
所以当前语料用 `ComputeCorrect.Realizes_without_Rounding` 是对的 surface。

## Bench 覆盖状态

141 个有 `.lean` port 的 kernel 分三档覆盖。

### 全闭合

Python 测试的每个输出都有完整 `ComputeCorrect.Realizes_without_Rounding`。约占已 port
bench 的一半。例子:`add_example`、`cosine_compute`、`dropout_triton`、
`swiglu_fwd`、`kldiv_compute`、`dequantize_rowwise`、`l2_norm_triton2`、
`apply_penalty`、`destindex_copy{,_kv1,_kv2}`、`max_reduction`、
`var_len_copy`、`fifth_order_sph_harmonics`(Y00–Y10)、以及 rotary
embedding 的 Q+K half store。

### Slice-only

Lean 只证 kernel 的部分 slice,preamble doc 注释也明说了。无法完整闭合
的原因分类:

- **同区域 offset 不重合** —— 多 store kernel 写同一区域但 offset 交错
  (rotary embedding 系列、fifth-order 球谐多输出)。跨区域 helper 不适用。
- **kernel 内 reduceSum 交错** —— `fast_layernorm`、`fast_rms_layernorm`、
  `layer_norm_liger`、`layer_norm_ops`:Y store 闭合;mean / rstd store
  跟 `reduceSum + setReg` 交错,`rw [BlockState.writeMem_readMem]` 不
  做交互式 goal 观察就 unify 不掉。

### Substrate-blocked

已闭合的 slice 已经达到 basic-lemma 层之上最大 scope,继续推进需要大块
新 substrate:

- **`tl.dot` 算法层模型** —— 全部 10 个 matmul、全部 14 个 flash /
  attention 系列、`attn_fwd_causal`、所有用 `tl.dot` 的 chunk kernel。
- **Streaming softmax invariant chain** —— attention forward kernel、
  `token_softmax_*`、`token_attn_*`。
- **`tl.cumsum` 方向感知语义** —— recurrent / cumsum backward(issue #94)。
- **`make_block_ptr` + 边界检查** —— cumsum、recurrent、用 block pointer
  的 attention。
- **Int 舍入 / packed int4-int8** —— `int8_quantization`、量化 KV per-block
  scales(issue #129 / #137)。
- **有符号指针运算** —— `conv2d`(issue #130)。
- **forLoop 包裹的 store 证明** —— `rope_embedding`、`fast_rope_embedding`、
  `kv_cache_filling`、`kv_cache_copy`。要走 `forLoop_inv` 证法,不是
  flat-foldl strip。
- **constexpr 分支 case-split** —— `rotary_transform`(4 个
  `Stmt.ifThenElse` 分支)、`layer_norm_ops` forward(4 个 Bool flag → 16
  组合)。

上述 substrate-blocker 加起来占了未证 bench 大头。已有的 helper 看
cookbook 的[证明模板](/VeriTile/zh-cn/cookbook/proof-templates/) 页。

## 近期批次

最近 5 次涉及 `bench/` 的提交:

| 日期 | 主题 |
|---|---|
| 2026-05-19 | post-#118 closure batch(bench + semantics + verso) |
| 2026-05-16 | fifth_order Y05–Y10、int8 scale、semantics helper |
| 2026-05-14 | triton_argmax scaffolding + softmax_flaggems refinements |
| 2026-05-13 | 清理 stalled-agent 残留 + 加 forRangeDyn_inv |
| 2026-05-12 | apply_penalty 重写 + rmsnorm Phase A carriers |

## 开放 issue 集群

GitHub bench 相关开放 issue,按 blocker 归类:

| 集群 | Issues | 卡在哪 |
|---|---|---|
| Attention / matmul | #128, #135 | `tl.dot` 算法模型 + autograd backward |
| RoPE / rotary | #134 | Q+K + KV cache + KV_GROUP_NUM gating |
| Recurrent / cumsum | #136 | 方向感知 scan、`make_block_ptr` |
| Layer norm + RMS norm 多 block | #122, #123, #133 | 跨循环 scalar 穿线(非 substrate;多周工作) |
| Quantization | #129, #137 | int 舍入、packed int4 / int8 |
| Diag SSM | #119 | 复杂数学 forward + backward,4 个 kernel |
| Conv2D | #130 | 有符号指针运算 |
| Layer-norm ops 完整 surface | #133 | constexpr 4-flag 16 组合 case split |
| RoPE 底向上基础设施 | #91 | execution roadmap tracker |
| Slice 证明后续 | #139 | ~30 个 store-slice 条目 |

上表**不全** —— 完整集合看
[GitHub 上的实时 issue 列表](https://github.com/Lizn-zn/VeriTile/issues)。

## 设计文档时效性

所有 `documents/*.md` 最近一次改动在 2026-05-04 ~ 2026-05-11 之间。
当下快照里没有任何文档被标记为 stale。各文件 `git log` 最近 touch
日期:

| 文档 | 最近 touch |
|---|---|
| EraseDType.md | 2026-05-11 |
| README.md | 2026-05-10 |
| ProofConventions.md | 2026-05-10 |
| CodeOrganization.md | 2026-05-10 |
| TritonSubset.md | 2026-05-09 |
| TheoremSurfaces.md | 2026-05-09 |
| CorrectnessSurfaces.md | 2026-05-09 |
| KernelManifest.md | 2026-05-07 |
| ConcurrencySemantics.md | 2026-05-07 |
| ApproxGeluPhiStrategy.md | 2026-05-06 |
| MemorySafety.md | 2026-05-05 |
| GpuMemoryModel.md | 2026-05-04 |

如果一份文档超过 ~30 天没动、而它描述的领域又有过实质性新工作,那就该
在依赖它之前先核一下时效性。

## 如何刷新本页

仓库根目录跑:

```bash
# bench 目录 / 已 port / README-only
find bench/tritonbench_g -mindepth 1 -maxdepth 1 -type d | wc -l
# ComputeCorrect.Realizes_without_Rounding 证明数
grep -rh 'ComputeCorrect.Realizes_without_Rounding\b' bench/tritonbench_g --include='*.lean' | wc -l
# sorry / admit 强零
grep -rE '(^|[^a-zA-Z_])(sorry|admit)( |$)' bench --include='*.lean' | wc -l
# 文档时效性
for f in documents/*.md; do printf '%s\t%s\n' "$(git log -1 --format='%cs' -- "$f")" "$(basename "$f")"; done | sort -r
```
