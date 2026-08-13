# llama_ff_triton

- Source file: `llama_ff_triton.py` (upstream `data/TritonBench_G_v1/llama_ff_triton.py`)
- Corpus: TritonBench-G v1
- Size: 152 lines, 1 `@triton.jit` kernel
- Status: **PORTED** — `LlamaFfTriton.lean`, main theorem
  `ff_llama_closed_form_correct` (`exec`-level closed form,
  dimension-general, 0 `sorry`).

The upstream kernel is the llama FFN first half — an **RMSNorm-fused SwiGLU
dual GEMM**: per output tile it streams K-blocks accumulating the RMS moment
(`a_sum += pow(a, 2)`) and two GEMMs (`acc1 += dot(a·rms_w, w1)`,
`acc2 += dot(a·rms_w, w3)`), then normalizes both by
`rsqrt(sum(a_sum)/K + EPS)` and gates `silu(acc1) * acc2` into a masked
fp16 store.

Headline: every active output cell holds the fp16 cast of
`silu(n_i · S1[i,j]) · (n_i · S2[i,j])`, where `S1/S2` are the genuine
double-sum GEMM references `Σ_t Σ_e A[r(i),·]·RMS[·]·W{1,3}[·,c(j)]` over the
kernel's `% M`/`% N`-wrapped indices and `n_i = 1/√((Σ A²)/K + EPS)` — all
derived independently of the kernel. Side conditions: the
`K = BLOCK_SIZE_K · numKBlocks` presentation (the loads are unmasked),
`soutn = 1` + `BN ≤ soutm` store-lane injectivity, clean-input `hundef`.

Translation-surface blocker (registered in `proof_blockers.md`): the
`USE_FP8` constexpr arm is dropped (parameter + branches; that path
bit-reinterprets int8 weight bytes as `tl.float8e5` via `bitcast=True` — an
ℝ-model limit with no value-level transcription; `sgmv_expand_slice`
dropped-arm precedent); `tl.cdiv(K, BLOCK_SIZE_K)` is the antiquoted
`numKBlocks`; the store's implicit fp16 cast is spelled
`(accumulator).to(tl.float16)` (`f8_conversion_utils` precedent); the loop
counter is `_i` for Python's `_`. The host launch and per-dtype dispatch are
the trusted boundary.

A `⊨[R]` io face is feasible on `StreamMasked3DKernelIO₄` (documented in the
module docstring) but not shipped — exec-headline-only, the `bmm_optimized`
precedent.
