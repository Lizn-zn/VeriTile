---
title: Project status
description: Where VeriTile stands today — bench coverage, what's blocked on what, and which design documents are current.
---

A snapshot of where the project is, captured against the repository at
the date below. Numbers come from a direct sweep of `bench/tritonbench_g/`
and `git log`, not from a manifest.

:::tip[Last verified]
Numbers below verified against `main` on **2026-05-19**. Re-run
[`scripts/check-artifact.sh`](https://github.com/Lizn-zn/VeriTile/blob/main/scripts/check-artifact.sh)
and the counts at the top of this file to refresh.
:::

## Bench corpus

| Metric | Value |
|---|---|
| `bench/tritonbench_g/<kernel>/` directories | **184** |
| With paired `.py` + `.lean` | **141** |
| README-only scaffolds (no port yet) | **43** |
| `ComputeCorrect.Realizes_without_Rounding` proofs across bench | **742** |
| `ComputeRefine.Refines_without_Rounding` proofs across bench | 0 |
| `sorry` / `admit` across bench | **0** |
| Theorems + lemmas across `VeriTile/` + `bench/` | **2,567** |

The zero-`sorry`, zero-`admit` invariant is enforced by
[`scripts/check-artifact.sh`](https://github.com/Lizn-zn/VeriTile/blob/main/scripts/check-artifact.sh)
in CI. Adding a `sorry` breaks the build.

Why `ComputeRefine.Refines_without_Rounding` is zero across bench: the refinement surface
exists, but every bench port so far has been "kernel ↔ math spec" rather
than "kernel ↔ kernel", so `ComputeCorrect.Realizes_without_Rounding` is the right surface
for the current corpus.

## Bench coverage status

The 141 kernels with `.lean` ports fall into three coverage buckets.

### Fully closed

Full `ComputeCorrect.Realizes_without_Rounding` for every Python-tested output. Roughly half
the ported bench. Examples: `add_example`, `cosine_compute`,
`dropout_triton`, `swiglu_fwd`, `kldiv_compute`, `dequantize_rowwise`,
`l2_norm_triton2`, `apply_penalty`, `destindex_copy{,_kv1,_kv2}`,
`max_reduction`, `var_len_copy`, `fifth_order_sph_harmonics`
(Y00–Y10), and the rotary embedding Q+K half stores.

### Slice-only

Lean proves a partial slice of the kernel and the preamble doc-comment
says so. Categories blocking full closure:

- **Intra-region offset disjointness** — multi-store kernels writing to
  the same region at interleaved offsets (rotary embedding families,
  fifth-order spherical harmonics multi-output). Cross-region helpers
  don't apply.
- **Intra-kernel reduceSum interleaving** — `fast_layernorm`,
  `fast_rms_layernorm`, `layer_norm_liger`, `layer_norm_ops`: Y store
  closes; mean / rstd stores interleave with `reduceSum + setReg` in a
  way that `rw [BlockState.writeMem_readMem]` won't unify without
  interactive goal inspection.

### Substrate-blocked

Closed slice is at the maximum scope achievable without new
infrastructure beyond the basic-lemma layer. The remaining work needs
substantial new substrate:

- **`tl.dot` algorithmic model** — all 10 matmul kernels, all 14
  flash / attention families, `attn_fwd_causal`, all chunk kernels with
  `tl.dot`.
- **Streaming softmax invariant chain** — attention forward kernels,
  `token_softmax_*`, `token_attn_*`.
- **`tl.cumsum` direction-aware semantics** — recurrent / cumsum
  backward kernels (issue #94).
- **`make_block_ptr` + boundary checks** — cumsum, recurrent,
  attention with block pointers.
- **Int rounding / packed int4-int8** — `int8_quantization`,
  quantized KV per-block scales (issues #129 / #137).
- **Signed pointer arithmetic** — `conv2d` (issue #130).
- **forLoop-wrapped store proofs** — `rope_embedding`,
  `fast_rope_embedding`, `kv_cache_filling`, `kv_cache_copy`. Need
  `forLoop_inv`-based proofs, not flat-foldl strip.
- **Constexpr branch case-splitting** — `rotary_transform` (4
  `Stmt.ifThenElse` branches), `layer_norm_ops` forward (4 Bool flags
  → 16 combos).

The collected substrate-blocker categories account for the bulk of
unproven bench entries. See the cookbook's
[proof templates](/VeriTile/cookbook/proof-templates/) page for the helpers that
*do* exist.

## Recent batches

Last five commits touching `bench/`:

| Date | Subject |
|---|---|
| 2026-05-19 | post-#118 closure batch (bench + semantics + verso) |
| 2026-05-16 | fifth_order Y05–Y10, int8 scale, semantics helper |
| 2026-05-14 | triton_argmax scaffolding + softmax_flaggems refinements |
| 2026-05-13 | clean stalled-agent artifacts + add forRangeDyn_inv |
| 2026-05-12 | apply_penalty rewrite + rmsnorm Phase A carriers |

## Open issue clusters

GitHub issue clusters open against the bench, grouped by blocker:

| Cluster | Issues | What's blocked |
|---|---|---|
| Attention / matmul | #128, #135 | `tl.dot` algorithmic model + autograd backward |
| RoPE / rotary | #134 | Q+K + KV cache + KV_GROUP_NUM gating |
| Recurrent / cumsum | #136 | direction-aware scan, `make_block_ptr` |
| Layer norm + RMS norm multi-block | #122, #123, #133 | cross-loop scalar threading (not substrate; multi-week work) |
| Quantization | #129, #137 | int rounding, packed int4 / int8 |
| Diag SSM | #119 | complex math forward + backward, 4 kernels |
| Conv2D | #130 | signed pointer arithmetic |
| Layer-norm ops full surface | #133 | constexpr 4-flag, 16-combo case split |
| RoPE bottom-up infra | #91 | execution roadmap tracker |
| Slice proofs follow-up | #139 | ~30 remaining store-slice entries |

The above is **not exhaustive** — see
[the live issue list on GitHub](https://github.com/Lizn-zn/VeriTile/issues)
for the full set.

## Design documents currency

All `documents/*.md` last touched between 2026-05-04 and 2026-05-11.
None are flagged as stale at the time of this snapshot. Each file's
last-touched date from `git log`:

| Document | Last-touched |
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

If a document hasn't been touched in over ~30 days and the area it
describes has shipped material new work, that's a cue to verify currency
before relying on it.

## How to refresh this page

Re-run from repo root:

```bash
# bench dirs / paired ports / README-only
find bench/tritonbench_g -mindepth 1 -maxdepth 1 -type d | wc -l
# ComputeCorrect.Realizes_without_Rounding proof count
grep -rh 'ComputeCorrect.Realizes_without_Rounding\b' bench/tritonbench_g --include='*.lean' | wc -l
# sorry / admit hard zero
grep -rE '(^|[^a-zA-Z_])(sorry|admit)( |$)' bench --include='*.lean' | wc -l
# docs currency
for f in documents/*.md; do printf '%s\t%s\n' "$(git log -1 --format='%cs' -- "$f")" "$(basename "$f")"; done | sort -r
```
