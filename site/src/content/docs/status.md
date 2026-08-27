---
title: Project status
description: Where VeriTile stands today — bench coverage, what's blocked on what, and which design documents are current.
---

A snapshot of where the project is, captured against the repository at
the date below. Numbers come from a direct sweep of `bench/tritonbench_g/`
and `git log`, not from a manifest.

:::tip[Last verified]
Numbers below verified against `main` on **2026-08-27**. Re-run
[`bench/audit_tritonbench_g.sh`](https://github.com/Lizn-zn/VeriTile/blob/main/bench/audit_tritonbench_g.sh)
(the full bench gate) and
[`scripts/check-artifact.sh`](https://github.com/Lizn-zn/VeriTile/blob/main/scripts/check-artifact.sh)
to refresh.
:::

## Bench corpus

| Metric | Value |
|---|---|
| `bench/tritonbench_g/<kernel>/` directories | **184** |
| With paired `.py` + `.lean` | **173** |
| README-only scaffolds (no port yet) | **11** |
| Ports elaborating (`bench/check_ports.sh`) | **173 ok, 0 fail** |
| Bench files axiom-clean (`bench/audit_trust.sh`) | **196 ok, 0 fail** |
| Headline declarations classified in `proof_gap_manifest.tsv` | **345** (344 `full_value_candidate`, 1 `blocked_summary`) |
| Ports stating a `KernelIO` `⊨` face | **153** |
| … of which also carry the rounding face `⊨[R]` | **68** |
| `ComputeRefine.Refines_without_Rounding` proofs across bench | 0 |
| `sorry` / `admit` across bench | **0** |

The zero-`sorry`, zero-`admit` invariant is enforced by
[`scripts/check-artifact.sh`](https://github.com/Lizn-zn/VeriTile/blob/main/scripts/check-artifact.sh);
the stronger "no smuggled axiom in any headline" invariant is enforced by
`bench/audit_trust.sh`, which appends `#axiomsClean` to every headline in every
standalone bench file and compiles it.

Why `ComputeRefine.Refines_without_Rounding` is zero across bench: the refinement surface
exists, but every `bench/tritonbench_g` port is "kernel ↔ math spec" rather
than "kernel ↔ kernel". The kernel-vs-kernel stories live in `bench/examples/`,
where they are stated on the `⊨` refinement notation `io₁ ≡[R] io₂`.

## Bench coverage status

All 173 ported kernels compile, are dimension-general (no test-shape pins), are
non-self-referential (`scripts/spec_sheet.py` reports `self-ref-flagged: 0`),
and are axiom-clean. The old "slice-only" and "substrate-blocked" buckets are
gone — `tl.dot`, the streaming-softmax invariant chain, direction-aware
`tl.cumsum`, `make_block_ptr` boundary checks, packed int4/int8 and signed
pointer arithmetic all landed.

What is still tracked, and where:

- **1 blocked summary** — `quant_transpose_kernel`, on fixed-width int8 cast
  semantics (issue #154). Recorded in `proof_gap_manifest.tsv`.
- **16 ports not yet on a named correctness surface** — their headlines are
  inline exec-existentials that omit the frame conjunct. Each carries a
  `Correctness-surface blocker:` marker and a row under "Correctness-Surface
  Blockers" in `proof_blockers.md`; the bench gate rejects any unregistered
  addition.
- **40 registered translation-surface deviations** — deliberate, documented
  departures from a literal 1:1 transcription (inlined helper JITs, constexpr
  arm specialization, …), each with a `Translation-surface blocker:` marker
  registered in both `proof_blockers.md` and `completion_audit.md`.
- **11 unported upstream kernels** — README-only scaffolds, blocked on RNG,
  `while` loops, fp4 and IEEE-bit paths.

See the cookbook's
[proof templates](/VeriTile/cookbook/proof-templates/) page for the helpers that
exist today.

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
