# chunk_retention

- Source file: `chunk_retention.py` (upstream `data/TritonBench_G_v1/chunk_retention.py`)
- Corpus: TritonBench-G v1
- Size: 451 lines, 4 `@triton.jit` kernels
- Status: **PORTED** (both state-recurrence kernels) — `ChunkRetention.lean`,
  main theorems `crh_fwd_h_exec_genuine` and `crh_bwd_dh_exec_genuine`
  (`exec`-level, 0 `sorry`).

The retention (decayed) sibling of the ported `chunk_linear_attn` pair: the
file's four kernels split two/two, and **this port covers the state-recurrence
pair** (`fwd_kernel_h`, the file's first kernel, and `bwd_kernel_dh`);
`fwd_kernel_o`/`bwd_kernel_dqkv` are the trusted boundary.

`fwd_kernel_h` runs the decayed recurrence

```
h[·,·,t] = H_t,   H_0 = (h0 or 0),   H_{t+1} = d_b(t)·H_t + k_tᵀ·(v_t ⊙ d_i(t))
```

with the per-head decay `b_b = log2(1 - 2^(-5 - i_h))` and per-chunk factors
`2^(len(t)·b_b)`, where `len(t)` is `T % BT` on a **ragged last chunk** — the
kernel rebinds `d_b`/`d_i` inside the loop on that chunk, and the proof's loop
invariant carries the register clause conditionally (`c < NT →` standard) so
one theorem covers ragged and exact division alike, plus all four
`USE_INITIAL_STATE`/`STORE_FINAL_STATE` configurations. The spec `crhState` is
*recursive* (not a power closed form): the ragged chunk gives the final step
its own decay length, and the recursion makes the per-chunk advance
definitional. Everything stays symbolic (dims, strides, `H`, `NT`); the layout
side conditions are the same three as `chunk_linear_attn`.

Three faithfulness disclosures (all in the `.lean` preamble):

1. The decay prologue's implicit int→float promotions are spelled
   `tl.toReal(...)` — a `Translation-surface blocker:` registered in
   `proof_blockers.md` (the `rbe_triton_transform` precedent).
2. On the ragged chunk, `(T % BT) - o_i - 1` goes negative on an integer tile
   upstream; the `.nat` channel truncates those lanes at `0`. The divergence
   is unobservable through every store — those lanes multiply `v` rows beyond
   `T`, which the block-pointer boundary check zeroes.
3. **`bwd_kernel_dh` is only compilable square** (`BK = BT = BV`, forced by
   its own `tl.dot` shapes), its loop is store-free with a dead `dh` load, and
   its single final store reuses the loop variable `i_t` *after* the loop
   (Python leaves `0`; the surface pre-initializes `i_t = 0` since the DSL
   scopes loop-body names). The headline proves exactly what this kernel
   computes — `dh` block = `d_b · Σ_{all chunks} (do ⊙ d_i)·v` under its
   scrambled contraction — with the oddities documented, not smoothed over.

`bwd_kernel_dh` descends (`range(NT-1, -1, -1)`); the surface spells the
ascending change of variable (`chunk_linear_attn` respelling, zero library
change). Arithmetic is over `ℝ`; the host launch is the trusted boundary.
