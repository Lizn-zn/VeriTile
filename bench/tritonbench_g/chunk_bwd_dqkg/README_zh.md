# chunk_bwd_dqkg

- 源文件:`chunk_bwd_dqkg.py`(上游 `data/TritonBench_G_v1/chunk_bwd_dqkg.py`)
- Corpus:TritonBench-G v1
- 规模:178 行,1 个 `@triton.jit` kernel
- 状态:**已移植** —— `ChunkBwdDqkg.lean`,主定理
  `chunk_bwd_dqkg_exec_genuine`(`exec` 级、维度一般、0 `sorry`)。

`chunk_simple_gla_bwd_kernel_dqkg` 是 simple-GLA 分块反向,产出 `dq`/`dk`/`dg`
三个梯度,与已移植的 `chunk_gla_simple` 前向配对。

value 轴循环在 launcher 自己的单值块区制 `V ≤ BV` 下验证(launcher 设
`BV = min(next_power_of_2(V), 64)`),该区制写成显式假设;全部维度、stride 和
`scale` 保持符号化。
