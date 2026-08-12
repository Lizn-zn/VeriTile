# parallel_retention_attention

- Source file: `parallel_retention_attention.py` (upstream `data/TritonBench_G_v1/parallel_retention_attention.py`)
- Corpus: TritonBench-G v1
- Size: 398 lines, 4 `@triton.jit` kernels
- Status: **PORTED** (forward + the backward dk/dv helper) —
  `ParallelRetentionAttention.lean`, main theorems `pra_fwd_o_exec_genuine`
  and `pra_bwd_dkv_exec_genuine` (`exec`-level, dimension-general,
  0 `sorry`).

The upstream file implements chunk-parallel *retention* attention: the
score is the plain dot product `q·k` weighted by the per-head exponential
decay `γ^(pos_q − pos_k)` with `γ = 2^b_b`,
`b_b = log2(1 − 2^(−5 − i_h))`, `i_h = i_bh % H`.

- `parallel_retention_fwd_kernel` (the launched forward JIT) streams K/V in
  two phases — the strictly-below-diagonal blocks unmasked with
  `tl.advance`d block pointers and the decayed accumulator recurrence
  `b_o = b_o · 2^(b_b·BTS) + (Q·K_j ⊙ d_h)·V_j`, then the diagonal block
  causally masked with the intra-block decay tile `d_s` — rescales by
  `d_q[:, None] = 2^(a·b_b)` in between, and stores `o` (block pointer,
  boundary-checked).
- `_parallel_retention_bwd_dkv` (the backward dk/dv helper JIT) streams
  Q/dO in the *opposite* order: the strictly-above-diagonal blocks via a
  **descending `-BTS` loop** (the final member of the descending-range
  lever family — the lever is now closed) with the mirrored decay
  recurrence (`b_dk *= d_b`, `b_dv *= d_b`, `b_do` pre-scaled by
  `d_q = 2^(c·b_b)`, `b_kd = b_k ⊙ 2^((BTL−r)·b_b)`), the post-loop
  rescalings `b_dk *= d_h[:, None]·scale` / `b_dv *= scale`, then the
  diagonal block masked. It is verified as a standalone kernel over
  universally-quantified scalar arguments `i_bh, i_c, i_k, i_v, i_h` — the
  values the shell `parallel_retention_bwd_kernel` would pass.

Headlines: `o[a, p]` =
`Σ_{t ≤ i_c·BTL + a} 2^((i_c·BTL + a − t)·b_b) · score(a,t) · v[t,p]` with
`score(a,t) = (scale·q[a]) · k[t]` over the `BK` head window (per-program
causal retention attention, exact on ragged tails — the boundary-checked
windows are baked into the guarded value functions, so **no** divisibility
hypothesis on `T` is needed); `dk`/`dv` = the full query sweep
`Σ_{t ≥ i_c·BTL + r} 2^((t − i_c·BTL − r)·b_b) · scale · (v[r]·do[t]) · q[t]`
/ `… · (k[r]·q[t]) · do[t]` over the kernel's exact iteration range
`max(cdiv(T,BTS)·BTS, (i_c+1)·BTL)`. The only trip-count hypothesis is the
host's own `assert BTL % BTS == 0`.

The helper-JIT structure (no cross-`@triton.jit` call surface), the dropped
trailing bare `return`, the descending-loop respelling
(`for j in range(0, cdiv(cdiv(T,BTS)·BTS − (i_c+1)·BTL, BTS))` with
body-first `i = cdiv(T,BTS)·BTS − BTS − j·BTS`; the `±BTS` cancels exactly
in ℤ and ℕ-truncation reproduces Python's empty range — the
`parallel_attention` respelling verbatim), the diagonal unary-minus decay
index `-o_k[:, None] + o_q[None, :]` respelled as the subtraction
`o_q[None, :] - o_k[:, None]` (identical on every lane the causal
`tl.where` mask keeps), and the explicit `tl.toReal(...)` int→float casts
(the `chunk_retention` precedent) are a `Translation-surface blocker:`
registered in `proof_blockers.md`. The backward shell
`parallel_retention_bwd_kernel` and `_parallel_retention_bwd_dq` are the
trusted boundary.
