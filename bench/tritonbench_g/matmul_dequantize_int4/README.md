# matmul_dequantize_int4

- Source file: `matmul_dequantize_int4.py` (upstream `data/TritonBench_G_v1/matmul_dequantize_int4.py`)
- Corpus: TritonBench-G v1
- Size: 268 lines, 1 `@triton.jit` kernel
- Status: **PORTED** — `MatmulDequantizeInt4.lean`, main theorem
  `matmul_dequantize_int4_exec_genuine` (`exec`-level, dimension-general,
  0 `sorry`).

`matmul4_kernel` is a GPTQ-style dequantizing GEMM: `C = A · dequant(qweight)`,
where `qweight` packs eight 4-bit weights per 32-bit word along K and `qzeros`
packs eight 4-bit zero-points per word along N. Every K step unpacks the weights
with a shift and a nibble mask, scales them, and subtracts the group's zero-point.

Both configurations of the `NO_GROUPS` flag are covered by the one theorem: it is
the *group row* that differs, and the spec reads `scales` / `zeros` at
`groupRow NO_GROUPS`, which is `0` at every step under the flag and
`k // (group_size // BLOCK_SIZE_K)` otherwise. Every dimension, stride, block size
and the `group_size` stays symbolic.

The output readback needs distinct lanes to have distinct `C` addresses; that is
the theorem's `hInj`, and `cAddr_injective` discharges it for a row-major `C`.
