# f8_conversion_utils

- Source file: `f8_conversion_utils.py` (upstream `data/TritonBench_G_v1/f8_conversion_utils.py`)
- Corpus: TritonBench-G v1
- Size: 68 lines, 2 `@triton.jit` kernels
- Status: **PORTED** — `F8ConversionUtils.lean`, main theorems
  `f16_to_f8_io_correctness` + `f8_to_f16_io_correctness`
  (`⊨[R, ·]` io faces, dimension-general, 0 `sorry`).

The upstream file implements fp8 ↔ fp16 buffer conversion: each kernel is a
masked 1-D copy (`offs = pid·BLOCK_SIZE + arange`, `mask = offs < N`), and
the fp8-ness lives entirely in the host's `triton.reinterpret(x,
tl.float8e5)` pointer — the stores cast **implicitly** to the destination
element type. `kernel_f8_to_f16` stores twice (verbatim duplicate in the
upstream body).

**First consumer of the fp8 dtype channel** (`.f8e4`/`.f8e5` added to
`TileDType`/`FloatDType` alongside this port): the f16→f8 headline is the
corpus's first fp8 boundary quantization —

> `f16ToF8IO Y X N BLOCK_SIZE ⊨[R, .f8e5] fun xs i => xs i`

for **every** rounding model `R`: every active output lane reads back as an
`.f8e5`-typed cell holding `R.round .f8e5 (xs j)`, the input value
quantized once onto the e5m2 grid (cast site + typed store site collapse by
`R.round_idem`). The reverse direction states the same contract at `.fp16`
(the duplicate store is idempotent). No hypotheses beyond the io skin's:
window bounds and exact-ℝ input lanes.

Translation-surface blocker (registered in `proof_blockers.md`): the
upstream stores carry no cast text — Triton casts implicitly at the typed
pointer — while the DSL types a store by its value, so the implicit casts
are spelled `(x).to(tl.float16)` / `(x).to(tl.float8e5)`. Python
`BLOCK_SIZE: tl.constexpr` becomes a Lean `Nat` parameter. The host launch
(grid, `reinterpret`, `numel` bookkeeping) is the trusted boundary; that
the f8-side *input* buffer holds f8-grid points is a host-pipeline fact the
theorems do not need (inputs are consumed as exact ℝ).
