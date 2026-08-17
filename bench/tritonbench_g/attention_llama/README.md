# attention_llama

- Source file: `attention_llama.py` (upstream `data/TritonBench_G_v1/attention_llama.py`)
- Corpus: TritonBench-G v1
- Size: 173 lines, 1 `@triton.jit` kernel
- Status: **PORTED** — `AttentionLlama.lean`, main theorems
  `attention_llama_fwd_closed_form_correct` (non-causal) and
  `attention_llama_fwd_causal_closed_form_correct` (causal), both
  `exec`-level, dimension-general, 0 `sorry`.

Target JIT: **`_fwd_kernel`** — the file's only `@triton.jit` kernel, a
FlashAttention-1-style streaming softmax (natural `exp`, in-loop
normalization `l_rcp = 1/l_curr; p *= l_rcp; acc *= (l_prev*l_rcp)`),
plain masked pointer loads, in-loop pointer advance
(`k_ptrs += BLOCK_N*stride_kn`, `v_ptrs += BLOCK_N*stride_vk`), one
masked `Out` store. Launcher `triton_fa`:
`grid = (cdiv(m_size, BLOCK), head_size·batch)`,
`BLOCK_M = BLOCK_N = BLOCK = 64`, `BLOCK_DMODEL = Lk ∈ {16,32,64,128}`.

**Parameter aliasing (H is the QUERY length).** The launcher passes
`N_HEAD := head_size`, `H := m_size`, `N_CTX := n_size` — so the
`offs_m[:, None] < H` mask on the `q` load and the `Out` store bounds
the QUERY row, not a head count, despite the name. The batch/head split
is `batch_id = pid₁ // N_HEAD`, `off_hz = pid₁ % N_HEAD`.

**Twin surfaces** (`IS_CAUSAL` constexpr; the `triton_matmul` twin
precedent): `attention_llama_fwd_surface` keeps `block_n_end = N_CTX`;
`attention_llama_fwd_causal_surface` keeps BOTH `block_n_end`
assignments plus the causal
`qk = tl.where(offs_m[:,None] >= (block_n_offs[None,:] + start_position), qk, -inf)`.
The loop bound is a runtime register in both arms (`forRangeDyn`).
Every other statement is byte-faithful and identical between the twins
(including the post-loop `start_m`/`offs_m`/`offs_d` rematerializations).
The `Out` store is `.real`: the kernel stores the fp32 `acc` WITHOUT a
cast (`p = p.to(Q.dtype.element_ty)` erases over ℝ), so both headlines
are exact-ℝ statements about the stored cells.

**Headlines** (both `ComputeCorrect.Realizes_without_Rounding` over the
kernel's own masked-store write map, expected values read from INPUT
memory through the kernel's own offset expressions; the Q tile carries
the kernel's own `row < H` load guard, true on every active row):

- *Non-causal*: every active `Out` cell equals `attentionReal` — the
  full natural-exp softmax over the `N_CTX` keys at scale `sm_scale`.
  Side conditions: `N_CTX = BLOCK_N · numKVBlocks`, `0 < numKVBlocks`,
  output-offset injectivity, `hundef` (clean undef carrier).
- *Causal*: every active `Out` cell equals `attentionRealCausalBlock`
  at `qStart = pid₀·BLOCK_M − start_position` over the streamed span
  `BLOCK_N·numCausalBlocks`; visible keys of row `pid₀·BLOCK_M + i` are
  exactly `{j < span : j + start_position ≤ pid₀·BLOCK_M + i}`. Side
  conditions: `hspanEq : (pid₀+1)·BLOCK_N + start_position =
  BLOCK_N·numCausalBlocks` (the `BLOCK_N ∣ start_position` trip-count
  divisibility in equation form), `span ≤ N_CTX`,
  `hsp : start_position ≤ pid₀·BLOCK_M` (fully general per-block; at
  `pid₀ = 0` it forces `start_position = 0`, which is every upstream
  test), offset injectivity, `hundef`. `BLOCK_M = BLOCK_N` is NOT
  needed by the proof (the divisible span leaves no trip-count ghosts).

**Ghost-lane / divisibility quirks** (why the side conditions):

- The unconditional `qk = tl.where(offs_n[None,:] < N_CTX, qk, -inf)`
  uses `offs_n` (0..BLOCK_N−1), NOT `block_n_offs` — under
  `BLOCK_N ≤ N_CTX` (implied by both headlines) it is an identity
  no-op, kept faithfully in the surface and proved away.
- Non-causal, non-divisible `N_CTX`: tail-block ghost lanes load
  `k = 0`, contribute `exp(0−m)` softmax mass, and skew the result —
  an upstream quirk; hence `N_CTX = BLOCK_N·numKVBlocks`.
- Causal: under the span divisibility there are no trip-count ghosts,
  and `span ≤ N_CTX` keeps every loaded lane in range — the causal arm
  tolerates arbitrary `N_CTX ≥ span`.
- **Causal direction quirk**: the predicate is
  `query_row ≥ key + start_position`, i.e. `start_position` SHRINKS
  visibility. It does NOT model "attend to a prefix cache" (that would
  be `j ≤ row + start_position`). The port models what the kernel
  computes.

**Positional-`other` loads.** The Python spells the K/V loads
`tl.load(ptrs, mask, 0.)` with a POSITIONAL `other`; the DSL grammar
has kwarg `other=` only, and a kwarg spelling would break the audit's
kwarg-sequence scan. The port spells them as plain masked loads whose
dead lanes read the undef carrier — pinned to `0` (the same value the
Python supplies) by the headline hypothesis `hundef`. The `q` load's
`other=0.0` is a Python kwarg and is kept as `MaskOpt.maskOther`.

Translation-surface blocker (registered in `proof_blockers.md`): the
`USE_FP8 = True` arm (`k.to(tl.float8e5, bitcast=True)` then
`.to(tl.float16)`, same for `v`) is an int8-byte → fp8
BIT-REINTERPRETATION — the established ℝ-model-limit family
(`llama_ff_triton` precedent). The port models `USE_FP8 = False` only
(upstream test cases 1/2); cases 3/4 (int8 K/V) are OUT of the modeled
surface. `IS_CAUSAL` is a constexpr split into the twin surfaces. Host
launch/grid/num_warps are the trusted boundary.
