# rms_rbe_matmul

- Source file: `rms_rbe_matmul.py` (upstream `data/TritonBench_G_v1/rms_rbe_matmul.py`)
- Corpus: TritonBench-G v1
- Size: 190 lines, 2 `@triton.jit` kernels (one dead in-file; see below)
- Status: **PORTED** — `RmsRbeMatmul.lean`, main theorem
  `rms_matmul_rbe_closed_form_correct` (`exec`-level closed form,
  dimension-general, 0 `sorry`).

Target JIT: **`rms_matmul_rbe`** (the file's second kernel) — a batched
RMSNorm-fused GEMM: `out[b] = (rms(x[b]) · rms_w) @ w`, `llama_ff_triton`'s
kernel minus the SwiGLU gate plus a batch grid axis. The file's first
kernel, `rbe_triton` (rotary embedding), is **dead code in this file**: no
host function launches it and it calls `get_freq_multi_tokens`, which is
undefined in the file — the upstream file itself cannot run it. It is not
modeled.

Headline: every active output cell holds the fp16 cast of `n_i · S[i,j]`
with `S = Σ_t Σ_e X[b, r(i), ·]·RMS[·]·W[·, c(j)]` (batch offset
`pid_batch·stride_x_batch` in every X read, `% M`/`% N`-wrapped indices) and
`n_i = 1/√((Σ X²)/K + EPS)` — derived independently of the kernel. Side
conditions: `K = BLOCK_SIZE_K · numKBlocks` (unmasked loads), `son = 1` +
`BN ≤ som` store-lane injectivity, clean-input `hundef`.

Translation-surface blocker (registered in `proof_blockers.md`): target JIT
is the file's second kernel (`rbe_triton` dead/unmodeled as above); the
`USE_FP8` constexpr arm dropped (bit-reinterprets int8 bytes as
`tl.float8e5` via `bitcast=True` — an ℝ-model limit; `llama_ff_triton`
precedent); the unused parameters `start_token_position`/`RBE_EPILOGUE`/
`THETA` dropped; `tl.cdiv(K, BLOCK_SIZE_K)` as the antiquoted `numKBlocks`;
the store's implicit fp16 cast spelled `(accumulator).to(tl.float16)`; the
loop counter `_i` for Python's `_`. Host launch and per-dtype dispatch are
the trusted boundary.
