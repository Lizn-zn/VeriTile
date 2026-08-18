# int8_dequant_matmul

- Source file: `int8_dequant_matmul.py` (upstream `data/TritonBench_G_v1/int8_dequant_matmul.py`)
- Corpus: TritonBench-G v1
- Size: 211 lines, 1 `@triton.jit` kernel
- Status: **PORTED** — `Int8DequantMatmul.lean`, main theorem
  `int8_dequant_matmul_exec_genuine` (`exec`-level, dimension-general, 0 `sorry`).

Target JIT = `_int8_matmul_rowwise_dequantize`, the file's only `@triton.jit`
kernel (launcher `int8_matmul_rowwise_dequantize`). An int8×int8→int32 GEMM
with a rowwise-dequantization epilogue: `A`, `B` are signed `torch.int8`
tensors (modelled as `Region .int`), the K loop accumulates the exact ℤ
matmul through the integer `tl.dot` — this port is the **first consumer of
`Op.dotInt`** — and the epilogue rescales each output cell as
`w_factor * (x_factor * (acc * divfactor))` (per-row `state_x` scale,
per-column `state_w` scale, host `divfactor = 1/(127·127)`), downcasts to
fp16 and, under `has_bias`, adds a per-column fp16 bias before the masked
store into the `torch.float16` output `C`. Standard group-swizzled pid
decomposition, `% M` / `% N` **wrapped** row/col offsets with unmasked
K-loop loads.

Five disclosed surface decisions (the module's `Translation-surface
blocker:` marker):

- **`SPLIT_K` fixed to `1`** — the `tl.store` arm. The autotune table sweeps
  `SPLIT_K ∈ {1, 2, 4, 8, 16}`; the `tl.atomic_add` arm (accumulating into
  the host-zeroed `C` of `pre_hook=init_to_zero("C")`) is dropped with the
  constexpr. Every `* SPLIT_K` factor folds to its `SPLIT_K = 1` value; the
  headline carries the launch fact `s.pids 1 = 0` (grid axis 1 has extent
  `SPLIT_K`), and `pid_z` / `rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)`
  stay in the surface faithfully.
- **`EVEN_K` fixed to `True`** — the unmasked-load arm. `EVEN_K` is the
  `@triton.heuristics` constexpr `K % (BLOCK_K * SPLIT_K) == 0`, so the
  masked `else` arm drops with the constexpr, and the loop bound
  `tl.cdiv(K, BLOCK_K * SPLIT_K)` is the antiquoted binder `numKBlocks`
  with side condition `K = BLOCK_K · numKBlocks` (the loads are unmasked,
  so the trip count must be exact).
- **`has_bias` kept as a genuine `Bool` parameter, both arms modeled** (the
  `matmul_dequantize` `NO_GROUPS` precedent): the host passes `0`/`1`; the
  headline covers both arms in one statement through the guarded bias term
  `if has_bias then bias(col) else 0`.
- **`.to(C.dtype.element_ty)` spelled `.to(tl.float16)`** (the host
  allocates `C` as `torch.float16`): the accumulator downcast compiles to a
  genuine `Op.castFloat real → fp16` and the bias load lands directly on
  the `.fp16` channel, so the terminal store is **`.fp16`-typed** and the
  headline reads the output back at the `MemCell` level
  (`MemCell.of .fp16 (fp16(i8Spec))` — the `matmul_dequantize`
  `matmul_kernel` precedent; the placeholder cast is the identity).
- **`$(n)` literal antiquoting** for integer literals inside index
  arithmetic (a bare literal is inferred `.real` by the DSL).

Every dimension, stride and block size stays symbolic (`GROUP_M` included).
The store mask is over the *unwrapped* output coordinates, so on every lane
it lets through, `row < M` / `col < N` make the offset wraps the identity —
the headline's closed form reads the plain `A[row, ·]` row and `B[·, col]`
column as signed ℤ values, the plain `state_x[row]` / `state_w[col]` scales,
and the bias at the plain `col` (the kernel reads it at the unwrapped `rn`
to begin with). The output readback needs distinct lanes at distinct `C`
addresses (`hInj`); `cAddr_injective` discharges it for a row-major `C`.
