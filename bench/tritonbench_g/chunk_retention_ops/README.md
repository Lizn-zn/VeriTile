# chunk_retention_ops

- Source file: `chunk_retention_ops.py` (upstream `data/TritonBench_G_v1/chunk_retention_ops.py`)
- Corpus: TritonBench-G v1
- Size: 363 lines, 4 `@triton.jit` kernels
- Status: **PORTED** (both state-recurrence kernels) — `ChunkRetentionOps.lean`,
  main theorems `cro_fwd_h_exec_genuine` and `cro_bwd_dh_exec_genuine`
  (`exec`-level, dimension-general, 0 `sorry`).

The upstream file is `chunk_retention`'s corpus twin, but the two are **not**
byte-duplicates where it matters:

- `fwd_kernel_h` is statement-for-statement identical to the ported
  `chunk_retention` sibling's (only `initial_state`/`final_state` are spelled
  `h0`/`ht`), so the whole forward proof stack — decay prologue, ragged
  last-chunk boundary gate, recursive `croState`, per-chunk store history —
  carries over.
- `bwd_kernel_dh` is the **fixed** sibling: where `chunk_retention`'s backward
  is only compilable square, stores once, and carries dead loads, this one is
  the clean dimension-general store-history kernel — per-chunk in-loop stores
  of the descending carry `D_next = d_b·D + (scale·q)ᵀ·(do ⊙ d_i)`, fixed
  `d_b`/`d_i` (no ragged rebind on the backward side). Its spec `croDhCarry`
  is recursive like the forward's, so the decayed advance is definitional.

Headlines: `h[·,·,t]` = decayed pre-chunk forward state (all four
`USE_INITIAL_STATE`/`STORE_FINAL_STATE` configurations, ragged and exact
division alike); `dh[·,·,t]` = pre-chunk descending carry `croDhCarry (NT-1-t)`.
Everything symbolic; the layout side conditions are the standard three.

The decay prologue's implicit int→float promotions are spelled
`tl.toReal(...)` — a `Translation-surface blocker:` registered in
`proof_blockers.md`. The nat-truncated ragged `d_i` tail (forward) is disclosed
in the preamble and is unobservable through the boundary-checked `v`. The
descending loop is spelled as its ascending change of variable
(`chunk_linear_attn` respelling, zero library change).
