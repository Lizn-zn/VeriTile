# bmm_optimized

- Source file: `bmm_optimized.py` (upstream `data/TritonBench_G_v1/bmm_optimized.py`)
- Corpus: TritonBench-G v1
- Size: 232 lines, 1 `@triton.jit` kernel
- Status: **PORTED** — `BmmOptimized.lean`, main theorem `bmm_o_exec_genuine`
  (`exec`-level, dimension-general, 0 `sorry`).

The upstream file implements a batched GEMM (`O[b] = A[b] · B[b]` on a 3-D
grid `(cdiv(M,TILE_M), cdiv(N,TILE_N), batch)`) with a `GROUP_M` CTA-reorder
swizzle (`tl.num_programs`-driven, with a runtime `GROUP_SIZE` gate on the
ragged final group) and `DIVISIBLE_M/N/K` heuristic constexprs that elide
masks when the host reports a divisible dimension.

The port fixes the constexpr assignment
`DIVISIBLE_M = DIVISIBLE_N = DIVISIBLE_K = False` — the fully-masked arm,
the only one total for **arbitrary** `M, N, K` (the other arms are
mask-elision optimizations on their divisible domains) — and transcribes
**both** `GROUP_M` CTA-reorder arms in full, including the runtime
`GROUP_SIZE` boundary gate as a nested `Stmt.ifThenElse`.

Headline: for every program on any launch grid, the masked `o` store holds
`Σ_{t < K} A[pid_b, m, t] · B[pid_b, t, n]` at every in-window lane, where
`(m, n)` are the global row/column of the CTA-reordered tile
(`bmmPidM`/`bmmPidN` closed forms — identity for `GROUP_M = 1`, the grouped
swizzle otherwise, both arms proven inside one theorem). Side conditions:
`TILE_N ≤ N` (store-lane injectivity), `0 < TILE_K` (the kernel's own
`tl.cdiv` trip count), and the clean-input `hundef` (the masked loads carry
no `other`, so masked-off lanes read the `undef` channel — the
`bmm_chunk_fwd` convention). **No** divisibility hypotheses on `M`, `N`, or
`K`.

The constexpr mask specialization, the three batch-offset parameter
reassignments (`A += pid_b*M*K` …) folded into the pointer-tile
constructions, the split tuple assignment `pid_m, pid_n = pidx, pidy`, and
the `range(num_iters)` spelled `range($(0), num_iters, $(1))` are a
`Translation-surface blocker:` registered in `proof_blockers.md`. The
`triton.autotune` config sweep and the host launch are the trusted boundary
(`TILE_M/TILE_N/TILE_K/GROUP_M` stay symbolic binders).
