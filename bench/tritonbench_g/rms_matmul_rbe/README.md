# rms_matmul_rbe

- Source file: `rms_matmul_rbe.py` (upstream `data/TritonBench_G_v1/rms_matmul_rbe.py`)
- Corpus: TritonBench-G v1
- Size: 279 lines, 2 `@triton.jit` kernels
- Status: **PORTED** — `RmsMatmulRbe.lean`, main theorems
  `rms_matmul_rbe_closed_form_correct` (the GEMM) and
  `rms_matmul_rbe_qkv_closed_form_correct` (the QKV caller — one bundled
  specification, a conjunction of three `Realizes` faces), both
  `exec`-level, dimension-general, 0 `sorry`.

The file's first kernel `rms_matmul_rbe` is the batched RMSNorm-fused GEMM
(byte-identical to the `rms_rbe_matmul` port's target JIT); its stack here
is a twin mirror. The second kernel — the only one any host function
launches — is **`rms_matmul_rbe_qkv`**, a **cross-JIT caller**: its body
invokes `rms_matmul_rbe(...)` three times (Q/K/V weights and outputs,
shared `x`/`rms_w`). The DSL has no jit-to-jit call surface, so the callee
body is inlined three times (the `attn_fwd` inline precedent), and the
proof factors one region/stride-parameterized pass bundle applied thrice
with frame transport between passes (10 region-distinctness hypotheses —
exactly the set the transport consumes; the host passes distinct buffers).

QKV headline: after the full run, each of Q/K/V's active cells holds the
fp16 cast of `n_i · S_w[i,j]` — the shared RMS normalizer against the
pass's own weight matrix. Side conditions: `K = BLOCK_SIZE_K · numKBlocks`
(unmasked loads), per-output row-major injectivity pairs, the distinctness
facts, clean-input `hundef`.

Translation-surface blocker (registered in `proof_blockers.md`): the
cross-JIT calls inlined; the `USE_FP8` constexpr arm dropped
(`bitcast=True` bit-reinterpretation, an ℝ-model limit); the callee's
unused parameters (`start_token_position`/`RBE_EPILOGUE`/`THETA`)
dropped/vanishing under inlining; `tl.cdiv(K, BLOCK_SIZE_K)` as the
antiquoted `numKBlocks`; the stores' implicit fp16 casts spelled
`(accumulator).to(tl.float16)`; loop counters `_i`. Host launch and
per-dtype dispatch are the trusted boundary.
