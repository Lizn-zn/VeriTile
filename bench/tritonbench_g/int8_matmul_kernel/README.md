# int8_matmul_kernel

- Source file: `int8_matmul_kernel.py` (upstream `data/TritonBench_G_v1/int8_matmul_kernel.py`)
- Corpus: TritonBench-G v1
- Size: 271 lines, 1 `@triton.jit` kernel
- Status: **PORTED** — `Int8MatmulKernel.lean`, main theorem
  `int8_matmul_kernel_exec_genuine` (`exec`-level, dimension-general, 0 `sorry`).

Target JIT = `matmul_kernel`, the file's only `@triton.jit` kernel (launcher
`matmul`, 1-D group-swizzled grid). A **pure-integer** GEMM over 2-bit-packed
weights: `A` is `torch.int32` (`Region .int`), `B` is `torch.uint8`
(`Region .nat`) with four 2-bit weights packed per byte along K
(`b.shape = [K//4, N]`), `C` is `torch.int32` — the terminal store is
**`.int`-typed** and no float appears anywhere. The nested loop puts the
2-bit **field index** `i ∈ {0..3}` outside and the packed K blocks `j`
inside: each inner step extracts field `i` (`mask = 3 << (2*i)`;
`(b_uint8 & mask) >> (2*i)`), shifts it into signed range `{−1,0,1,2}` by
subtracting the all-ones `tensor_full`, and accumulates the exact integer
`tl.dot(a, b − 1, out_dtype=tl.int32)` (`Op.dotInt`). `a_ptrs` advances
**continuously across both loops** (never reset) so `A`'s columns sweep
`0..K−1` in order, while `b_ptrs` is rebound to the start of `B` at the top
of each outer iteration. Group-swizzled pid decomposition with this
kernel's **extra `% num_pid_in_group`** in the `pid_m` line (spelled
faithfully); `% M` / `% N` wrapped row/col offsets.

For an in-range output cell the headline's stored int32 is, over ℤ:

```
C[row, col] = ∑ i<4, ∑ kk<K/4, A[row, i·(K/4) + kk] · (bits_i(B[kk, col]) − 1)
bits_i(w)   = (w >>> (2·i)) &&& 3          (K/4 = numKBlocks · BLOCK_SIZE_K)
```

Four disclosed surface decisions (the module's `Translation-surface
blocker:` marker):

- **`tl.cdiv(K // 4, BLOCK_SIZE_K)` as the antiquoted binder `numKBlocks`**
  (the inner-loop bound and the `k = i * tl.cdiv(K // 4, BLOCK_SIZE_K) + j`
  bookkeeping), with the honest side condition
  `hK : K = 4 · (BLOCK_SIZE_K · numKBlocks)` — exactly the source's own
  `tl.static_assert(K % (4 * BLOCK_SIZE_K) == 0)`, which is kept in the
  surface (it value-erases to a no-op statement). Under `hK` both load
  masks are **degenerately all-true**, so raggedness in K is impossible
  and the loads reduce to unmasked reads.
- **Tuple shape arguments respelled with brackets**:
  `tl.zeros((BM, BN), …)` → `tl.zeros([$(BM), $(BN)], …)` and
  `tl.full((1,), 1, …)` → `tl.full([1], 1, …)` (the audit-known `(1,)` vs
  `[1]` shape respell).
- **Integer widths are erased** (the `int8_quantization` fixed-width
  family): `.to(tl.int8)` on the `A` load is a no-op on the `.int` channel
  — the model keeps the full launch-state int32 value, where int8 hardware
  would wrap values ≥ 128 (the host test feeds `A` values in `0..255`);
  `.to(tl.int8)` on the 2-bit extract lowers to the genuine signed hop
  `Op.castNatToInt` (extract values are `{0..3}`, no wrap reachable); and
  `tl.full([1], 1, dtype=tl.int8)` lowers to the width-erased `.int`
  literal `Op.constInt 1`.
- **`$(n)` literal antiquoting** for integer literals inside index
  arithmetic (a bare literal is inferred `.real` by the DSL).

Probed and spelled **faithfully** (not deviations): the
`out_dtype=tl.int32` kwarg on `tl.dot` is kept and macro-erased (`Op.dotInt`
is exact ℤ, so the kwarg is semantically redundant); `b - tensor_full`
(`[BK, BN] − [1]`) compiles through the rank-promoting broadcast
`Broadcast.leadR` with no respell; `tl.program_id(axis=0)` is native
syntax; the in-loop register named `mask` shadows nothing (the store mask
is the separate `c_mask`).

Every dimension, stride and block size stays symbolic. The store mask is
over the *unwrapped* output coordinates, so on every lane it lets through
`row < M` / `col < N` make the offset wraps the identity — the headline
reads the plain `A[row, ·]` row (signed ℤ) and the plain packed
`B[kk, col]` bytes (ℕ). The output readback is at the `MemCell` level
(`MemCell.of .int`) and needs distinct lanes at distinct `C` addresses
(`hInj`); `imCAddr_injective` discharges it for a row-major `C`. The grid
is 1-D, so there is no auxiliary program-id hypothesis.
