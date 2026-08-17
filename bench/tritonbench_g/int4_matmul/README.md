# int4_matmul

- Source file: `int4_matmul.py` (upstream `data/TritonBench_G_v1/int4_matmul.py`)
- Corpus: TritonBench-G v1
- Size: 251 lines, 1 `@triton.jit` kernel
- Status: **PORTED** — `Int4Matmul.lean`, main theorem
  `int4_matmul_exec_genuine` (`exec`-level, dimension-general, 0 `sorry`).

Target JIT = `matmul_kernel`, the file's only `@triton.jit` kernel (launcher
`matmul_dequantize_int4_s2`). A GPTQ-style int4-dequantizing GEMM:
`C = A · dequant(B)`, with eight 4-bit weights packed per 32-bit word along K
(`qweight`) and eight 4-bit zero-points packed per word along N (`qzeros`),
one `(scale, zero-point)` group row per `group_size` K rows, the standard
group-swizzled pid decomposition, `% M` / `% N` **wrapped** row/col offsets
with unmasked loads, and an fp32 accumulator. Unlike the
`matmul_dequantize_int4` twin (which scales before subtracting), this kernel
subtracts the zero-point nibble **first** — a signed difference.

Three disclosed surface decisions (the module's `Translation-surface
blocker:` marker):

- **`SPLIT_K` fixed to `1`** — the `tl.store` arm. The autotune table sweeps
  `SPLIT_K ∈ {1, 2}`; the `tl.atomic_add` arm (accumulating into the
  host-zeroed `C` of `reset_to_zero=['c_ptr']`) is dropped with the
  constexpr. The headline carries the matching launch fact `s.pids 1 = 0`
  (grid axis 1 has extent `SPLIT_K`).
- **`numKBlocks` loop bound** — `tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)` is the
  antiquoted binder `numKBlocks` with side condition
  `K = BLOCK_SIZE_K · numKBlocks` (the kernel's own docstring assert; the
  loads are unmasked, so the trip count must be exact).
- **Signed dequant via explicit casts** — the packed containers are
  `Region .nat` (the DSL's bitwise `>>` / `&` live on that channel), but
  `int_b - int_bzp` goes negative and ℕ subtraction truncates, so the
  subtraction is spelled `tl.cast(·, tl.int32) - tl.cast(·, tl.int32)` and
  promotes to ℝ through `Op.intToReal` — this port is that operator's
  **first consumer**. `BLOCK_SIZE_K % 8 == 0` (the source's other assert)
  is the headline's `hBK8`, identifying the kernel's
  `(BLOCK_SIZE_K * stride_bk) // 8` pointer advance with the packed-word
  address.

Every dimension, stride, block size and `group_size` stays symbolic. The
store mask is over the *unwrapped* output coordinates, so on every lane it
lets through, `row < M` / `col < N` make the offset wraps the identity — the
headline's closed form reads the plain `A[row, ·]` row and `B`/`BS`/`BZP`
column, with per-lane group rows `(e + k·BK) / group_size`. The output
readback needs distinct lanes at distinct `C` addresses (`hInj`);
`cAddr_injective` discharges it for a row-major `C`.
