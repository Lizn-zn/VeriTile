# VeriTile Verification Benchmark

This folder is the home for VeriTile's kernel-verification benchmark — the set
of Triton kernels VeriTile commits to verifying, with source, tier, masking
variant, and proof-status tracked.

> **Every per-kernel main theorem must meet the standard in
> [`MAIN_THEOREM_CONVENTIONS.md`](./MAIN_THEOREM_CONVENTIONS.md)** (dimension-general,
> non-self-referential, `ComputeCorrect.Realizes` form, axiom-clean, honest
> hypotheses; no test-shape/dead code). Read it before adding or reviewing a kernel.

> Status (2026-05-05): **TritonBench-G v1** picked as the external anchor.
> Static coverage analysis landed at
> [`tritonbench_coverage.md`](./tritonbench_coverage.md). Imported upstream
> Python kernels go under [`tritonbench_g/`](./tritonbench_g/); Lean
> ports live under `VeriTile/Examples/`. Per-kernel manifest is now
> [`scripts/kernel-manifest.tsv`](../scripts/kernel-manifest.tsv); per-port
> proof-status table will populate as ports land.

## Current state

There is no formal benchmark manifest yet. The de facto benchmark is
`VeriTile/Examples/` — VectorAdd, Softmax, OnlineSoftmax, LayerNorm, Welford,
FusedSiLU, ApproxGeLU, RowWise, GridComposition, FlashAttention-1 forward
across dense / causal / boundary / D-tail variants, plus the smoke and frame
files. It works for development but lacks source attribution, a versioned
manifest, and a published progress table — so it is not currently usable as a
benchmark *artifact* visible to outside readers.

The earlier `bench/llm_eval/` LLM-proof-drafting eval was retired on
2026-05-05; LLM-assist evaluation is no longer a benchmark axis of the
project.

## TritonBench-G v1 anchor

We are aligning the verification benchmark with [TritonBench-G v1][tb] (184
GitHub-scraped real Triton kernels, ACL 2025 Findings). A static primitive
scan of all 184 kernels against the current DSL contract gives, after two
trivial surface adapters (`tl.math.*`, `tl.extra.cuda.libdevice.*`):

| Verdict | Count | Share |
|---|---:|---:|
| OK (DSL covers primitives) | 141 | 77% |
| Soft (small DSL/proof extension) | 15 | 8% |
| Hard (new semantic layer required) | 28 | 15% |

Top remediation levers, ranked by file yield: `tl.math.*` adapter (24),
`tl.extra` adapter (9), concurrency boundary (#12, unlocks 11), `tl.num_programs`
(9), FP8 dtype channel (7), `atomic_add` proof shape (6), int4 packed unpack
(4), RNG (#41, 4). See [`tritonbench_coverage.md`](./tritonbench_coverage.md)
for the full per-file table, hard-gap reasons, and caveats.

The "OK" verdict is *expressibility*, not *proof feasibility*: many `OK`
kernels (FA-1 backward, RWKV6, Mamba SSM, chunked GLA) still need fresh proof
engineering even though their primitives lower today.

[tb]: https://github.com/thunlp/TritonBench/tree/main/data/TritonBench_G_v1

## External Triton kernel corpora

Survey of real-world Triton sources that can act as verification targets.
Attributions and sizes verified against arXiv / repo READMEs as of
2026-05-05.

| Corpus | Size | Character | Fit for VeriTile |
|---|---|---|---|
| **TritonBench-G** (THUNLP / Tsinghua, ACL 2025 Findings; arXiv 2502.14752) | 184 GitHub-scraped real Triton kernels | Closest to "production Triton" in the wild | **Picked as VeriTile's external anchor (2026-05-05).** Static scan shows 77% of the 184 kernels are within current DSL contract after two trivial surface adapters; concurrency-boundary cost is real but bounded (~11 of 184). See `tritonbench_coverage.md`. |
| **TritonBench-T** (same paper) | PyTorch-interface-aligned kernels (exact count not given in abstract; check repo) | Reference implementations available | Pairs well with the Compute-layer differential-testing track. |
| **liger-kernel** (LinkedIn; arXiv 2410.10989) | LLM-training kernel collection | RMSNorm, RoPE, SwiGLU, CrossEntropy, FusedLinearCrossEntropy, … (HF-transformers integrated) | **Highest overlap with VeriTile's Tier 4 roadmap.** Single-file, upstream tests, training-relevant, externally maintained. |
| **FlagGems** (BAAI / FlagOpen) | ~184 ops, target 216 (README) | Triton implementation of PyTorch op coverage | Many elementwise ops fall out at Tier 1 / Tier 2 for free; bulk is uneven. |
| **Unsloth kernels** (unslothai/unsloth) | Training-fast-path kernels | RoPE, MLP, NF4 dequant, cross-entropy, RMSNorm, SwiGLU, GeGLU | Training-side complement to liger. |
| KernelBench (Stanford; arXiv 2502.10517, Feb 2025) | 250 PyTorch tasks across L1 (100 single ops) / L2 (100 fused) / L3 (50 architectures); + 20 L4 HF aspirational | **Generation** benchmark (Torch → CUDA), not verification | Not a verification target. |

## Direction

In order:

1. **Land the two trivial surface adapters first** — `tl.math.*`
   (`exp2/log2/rsqrt/sin`) and `tl.extra.cuda.libdevice.*` (`pow/tanh/llrint`).
   These are syntactic; semantics already exist under VeriTile's existing
   spellings. Together they move 33 TritonBench-G files from Soft/Hard
   into OK at zero semantic cost.

2. **Port the first concrete TritonBench-G kernel.** Pick a small OK-verdict
   file outside `Examples/` — candidates: `vector_addition.py`,
   `softmax_triton1.py`, `rope_embedding.py`, `matmul_kernel.py`. Use the
   port to design the per-kernel metadata schema (source URL +
   commit, BLOCK_* config chosen, dropped hints, public theorem symbol,
   verdict).

3. **Versioned manifest.** Once one external kernel is in, freeze
   the schema and back-fill it across `Examples/` and the new TritonBench-G
   port. The manifest is the source of truth for the project README progress
   table.

4. **Tackle remediation in yield order**, per
   [`tritonbench_coverage.md`](./tritonbench_coverage.md): `tl.num_programs`
   (9 files), atomic_add proof shape (6), then the larger semantic
   investments — concurrency boundary (#12), FP8 channel, int4 packed,
   RNG (#41), FP4.

## Trade-offs recorded

- **TritonBench-G anchor.** Earlier draft favoured liger-kernel first because
  TritonBench-G "would hit the concurrency boundary on day one." The actual
  scan disproves that: only 11 of 184 kernels (`tl.debug_barrier` x8 +
  `atomic_cas/xchg` x3) sit on the concurrency boundary. The 77% OK figure
  makes TritonBench-G a viable, externally legible primary benchmark, not
  just a stress test.
- **Static-only verdict.** "OK" attests to *expressibility* under the current
  DSL contract, not to *proof feasibility*. Many OK kernels still need fresh
  proof engineering. Reflected in the headline numbers and in
  `tritonbench_coverage.md` §Caveats.
- **Manifest after first port.** The per-kernel metadata schema is shaped by
  the first real external port rather than guessed up-front.
- **No LLM-eval axis.** The project no longer benchmarks LLM proof-drafting.
  If that returns, it is a separate sub-benchmark, not the benchmark.
