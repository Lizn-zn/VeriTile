# parallel_attention

- Source file: `parallel_attention.py` (upstream `data/TritonBench_G_v1/parallel_attention.py`)
- Corpus: TritonBench-G v1
- Size: 481 lines, 4 `@triton.jit` kernels
- Status: **PORTED** (forward + the backward dk/dv helper) —
  `ParallelAttention.lean`, main theorems `pa_fwd_o_exec_genuine` and
  `pa_bwd_dkv_exec_genuine` (`exec`-level, dimension-general, 0 `sorry`).

The upstream file implements *rebased* (quadratic-kernel) attention: scores
are `(q·k)²` instead of `exp(q·k)`, so causal attention factors into an
unnormalized value sum `o` plus a separately stored normalizer `z` (the host
divides `o / (z + eps)` outside the kernel).

- `parallel_rebased_fwd_kernel` (the launched forward JIT) streams K/V in
  two phases — the strictly-below-diagonal blocks unmasked with
  `tl.advance`d block pointers, then the diagonal block causally masked —
  and stores both `o` (block pointer, boundary-checked) and `z` (plain
  pointer, masked).
- `_parallel_rebased_bwd_dkv` (the backward dk/dv helper JIT) streams Q/dO
  in the *opposite* order: the strictly-above-diagonal blocks via a
  **descending `-BTS` loop** (the last member of the descending-range lever
  family), then the diagonal block masked, with an `i_v == 0` runtime gate
  folding the `dz` normalizer gradient in. It is verified as a standalone
  kernel over universally-quantified scalar arguments `i_bh, i_c, i_k, i_v,
  i_h` — the values the shell `parallel_rebased_bwd_kernel` would pass.

Headlines: `o[a, p]` = `Σ_{t ≤ i_c·BTL + a} score(a,t)² · v[t,p]` with
`z[a]` its normalizer (per-program causal rebased attention, exact on
ragged tails — the boundary-checked windows are baked into the guarded
value functions, so **no** divisibility hypothesis on `T` is needed);
`dk`/`dv` = the full key sweep `Σ_t keep·(2·ds·s·q)` / `Σ_t keep·s²·do`
with the gate term `[i_v = 0]·dz[t]` inside `ds`. The only trip-count
hypothesis is the host's own `assert BTL % BTS == 0`.

The helper-JIT structure (no cross-`@triton.jit` call surface), the dropped
trailing bare `return`, and the descending-loop respelling
(`for j in range(0, cdiv(cdiv(T,BTS)·BTS − (i_c+1)·BTL, BTS))` with
body-first `i = cdiv(T,BTS)·BTS − BTS − j·BTS`; the `±BTS` cancels exactly
in ℤ and ℕ-truncation reproduces Python's empty range) are a
`Translation-surface blocker:` registered in `proof_blockers.md`. The
backward shell `parallel_rebased_bwd_kernel` and `_parallel_rebased_bwd_dq`
are the trusted boundary.
