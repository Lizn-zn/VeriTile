# chunk_bwd_dqkg

- Source file: `chunk_bwd_dqkg.py` (upstream `data/TritonBench_G_v1/chunk_bwd_dqkg.py`)
- Corpus: TritonBench-G v1
- Size: 178 lines, 1 `@triton.jit` kernel
- Status: **PORTED** — `ChunkBwdDqkg.lean`, main theorem
  `chunk_bwd_dqkg_exec_genuine` (`exec`-level, dimension-general, 0 `sorry`).

`chunk_simple_gla_bwd_kernel_dqkg` is the simple-GLA chunked backward for the
three gradients `dq`, `dk`, `dg` — the backward partner of the ported
`chunk_gla_simple` forward.

The value-axis loop is verified in the launcher's own single value-block regime
`V ≤ BV` (the launcher sets `BV = min(next_power_of_2(V), 64)`), stated as an
explicit hypothesis; every dimension, stride and the `scale` stay symbolic.
