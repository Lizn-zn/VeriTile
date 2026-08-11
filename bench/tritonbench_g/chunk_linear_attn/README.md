# chunk_linear_attn

- Source file: `chunk_linear_attn.py` (upstream `data/TritonBench_G_v1/chunk_linear_attn.py`)
- Corpus: TritonBench-G v1
- Size: 309 lines, 4 `@triton.jit` kernels
- Status: **PORTED** (both state-recurrence kernels) — `ChunkLinearAttn.lean`,
  main theorems `cla_fwd_h_exec_genuine` and `cla_bwd_dh_exec_genuine`
  (`exec`-level, dimension-general, 0 `sorry`).

The file's four kernels split two/two: the state-recurrence pair
(`chunk_linear_attn_fwd_kernel_h`, the file's first kernel, and its descending
mirror `chunk_linear_attn_bwd_kernel_dh`) and the fused output/gradient pair
(`fwd_kernel_o`, `bwd_kernel_dqkv`). **The port covers the state-recurrence
pair**; the other two are the trusted boundary (covering a subset of a
multi-kernel file is the established shape here — `chunk_gla_fwd` and
`triton_linear_activation` do the same).

Both ported kernels share one skeleton: a `[BK, BV]` running state is **stored
to memory at every chunk** and then advanced by one `tl.dot` —

```
fwd_h:  h[·,·,t] = H_t,   H_0 = (h0 or 0),  H_{t+1} = H_t + k_tᵀ · v_t
        (and ht = H_NT when STORE_FINAL_STATE)
bwd_dh: dh[·,·,t] = D_t,  D_NT = 0,         D_{t-1} = D_t + (scale·q_t)ᵀ · do_t
```

so each theorem's per-lane value is a **per-chunk memory readback**: chunk `t`'s
block holds the state *before* that chunk's own contribution (the seed plus all
earlier chunks forward; the strictly-later chunks backward). The loop invariant
carries the running state, a per-chunk store-history clause, and an
untouched-regions frame; distinct chunks' stores stay disjoint because each
chunk owns one `K·V` block of `h`/`dh`.

`bwd_kernel_dh` iterates `for i_t in range(NT - 1, -1, -1)`. The surface spells
the identical iteration sequence ascending — `for j in range(0, NT)` with
`i_t = NT - 1 - j` as the body's first statement — the established respelling
for descending loops (`triton_linear_activation`), observable in the surface
rather than hidden in a spec. This is the first ported consumer of the
descending-`range` lever, and it needed **no library change**: plain
`forRange_inv` drives the ascending counter.

One theorem covers all four `USE_INITIAL_STATE` / `STORE_FINAL_STATE`
configurations of the forward kernel. Every dimension, stride, `scale`, and the
chunk count `NT` stay symbolic; the side conditions are region distinctness and
three layout facts that are equalities/immediate under the launcher's contiguous
state tensor (`h.stride(2) = V`): `BV ≤ s_h_t`, `(K-1)·s_h_t + V ≤ K·V`, and
`BV ≤ V`.

Arithmetic is over `ℝ` (the algorithm layer); the host launch (the 3-D grid
`(NK, NV, B·H)` and the host-computed `NT = cdiv(T, BT)`) is the trusted
boundary.
