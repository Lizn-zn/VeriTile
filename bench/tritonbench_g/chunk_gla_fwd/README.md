# chunk_gla_fwd

- Source file: `chunk_gla_fwd.py` (upstream `data/TritonBench_G_v1/chunk_gla_fwd.py`)
- Corpus: TritonBench-G v1
- Size: 368 lines, 5 `@triton.jit` kernels
- Status: **PORTED** (output kernel) — `ChunkGlaFwd.lean`, main theorem
  `chunk_gla_fwd_o_exec_genuine` (`exec`-level, dimension-general, 0 `sorry`).

The file's five kernels split four/one: four build the intra-chunk attention
matrix `A`, and `chunk_gla_fwd_kernel_o` consumes it to produce the output. **The
port covers the output kernel**; the four `A`-builders are the trusted boundary
(covering a subset of a multi-kernel file is the established shape here —
`triton_linear_activation` and `kv_cache_filling` do the same).

`chunk_gla_fwd_kernel_o` is the gated sibling of the ported `chunk_gla_simple`:
same `q`/`h`/`v`/`o` block-pointer layout, but the gate is a 2-D `[T, K]` tensor
loaded per K block, and `A` is read from memory. The theorem's per-lane value is

```
O[i, p] = Σ_{kb < cdiv(K, BK)} Σ_e (scale·q ⊙ exp g)[i, kb·BK+e] · h[kb·BK+e, p]
            + Σ_j tril(A)[i, j] · v[j, p]
```

Unlike the sibling's headline, there is **no `K = BK` hypothesis** — the K loop is
verified in full multi-block generality, which is affordable because the kernel
recomputes its block pointers from `i_k` each iteration instead of advancing them.
Every dimension, stride, and the `scale` stay symbolic; the only side condition is
the standard row-major output-injectivity `hInj`.
