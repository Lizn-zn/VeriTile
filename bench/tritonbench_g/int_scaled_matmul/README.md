# int_scaled_matmul

- Source file: `int_scaled_matmul.py` (upstream `data/TritonBench_G_v1/int_scaled_matmul.py`)
- Corpus: TritonBench-G v1
- Size: 303 lines, 2 `@triton.jit` kernels — **both launched and both modeled**
- Status: **PORTED** — `IntScaledMatmul.lean`, main theorems
  `int_scaled_matmul_matmul_exec_genuine` (kernel 1) and
  `int_scaled_matmul_scaled_exec_genuine` (kernel 2)
  (`exec`-level, dimension-general, 0 `sorry`).
- Library riders: this port landed (a) the `tl.broadcast_to(e, [dims*])`
  expression syntax (lowering to the existing `Op.broadcast` / `Op.remap`
  broadcast machinery; smoke-gated in `bench/tests/TritonSmoke.lean`), and
  (b) block-pointer element-dtype inheritance — `tl.make_block_ptr` arms in
  the DSL's pointer-element-dtype propagation, so `tl.load(bp, …)` through
  a block pointer whose base is a typed region defaults to the base's
  element dtype (untyped bases keep the historical `.real` default; all
  existing block-ptr ports lower unchanged).

Audit-anchor JIT = `matmul_kernel_with_block_pointers` (the file's first
kernel, launcher `int_matmul_kernel`): a **block-pointer** int8×int8→int32
GEMM. Three `tl.make_block_ptr` views with runtime per-`pid` offsets
(`Op.makeBlockPtrDynOffsets`), `boundary_check=(0, 1)` **zero-fill** loads
on the `.int` channel, an `Op.dotInt` accumulation over the step-form loop
`for k in range(0, K, BLOCK_K)` (spelled directly — `Stmt.forRange` carries
the step, so no trip-count binder exists for this kernel), in-loop
`tl.advance` on both input pointers, and a boundary-checked block-pointer
store of the int32 tile. The `GROUP_M` constexpr is **rebound as a
register** (`GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)`) exactly as
Python rebinds the name — probed: the register named `GROUP_M` coexists
with the Lean binder, since the binder only ever appears antiquoted.

The second kernel, `scaled_matmul_kernel_with_block_pointers` (launcher
`int_scaled_matmul_kernel`), is a classic pointer GEMM — no block pointers
despite the name — with the `triton.ops.matmul` swizzle, `% M`/`% N`
wrapped rows/cols behind value-erased
`tl.max_contiguous(tl.multiple_of(…))`, the **descending** loop
`for k in range(K, 0, -BLOCK_K)` with the genuine `EVEN_K` Bool constexpr
(**both arms modeled**: unmasked loads vs `rk < k` remaining-count masks
with `other=0.0`), and the inductor-generated suffix: `rm`/`rn`
rematerialized, `xindex = idx_n + (N * idx_m)` flat addressing,
`tmp0 = tl.load(s1_ptr + tl.broadcast_to(idx_m, …), mask, eviction_policy=…)`
(positional mask parses natively; the eviction hint erases), and the masked
store of `acc * tmp0` — an `.int × .real` product auto-promoted through
`Op.intToReal`.

For an in-range output cell (`row < M`, `col < N`) the headlines give:

```
kernel 1:  C[row·s_cm + col·s_cn]  =  MemCell.of .int  (∑ kk<K, A[row,kk]·B[kk,col])          (exact ℤ)
kernel 2:  C[col + N·row]          =  MemCell.of .real ((∑ kk<K, A[row,kk]·B[kk,col] : ℤ→ℝ) · s1[row])
```

with `s1` read at the **strideless** address `row` (`stride_s1m`/`stride_s1n`
are passed but dead in the body — the `fused_recurrent_retention`
dead-stride precedent).

Disclosed surface decisions (the module's two `Translation-surface
blocker:` markers):

- **Kernel 1** — (1) the block-ptr loads are spelled faithfully
  (`tl.load(bp, boundary_check=(0, 1))`, no added kwargs): the `.int`
  channel is inherited from the typed base region
  (`base=$((a_ptr : Region .int))`, rider (b)); what is disclosed is that
  integer widths are erased (the `int8_quantization` fixed-width family).
  (2) tuple→bracket respells for `tl.advance` deltas and `tl.zeros` shapes.
  (3) the `GROUP_M` register rebind spelling (register on the left,
  antiquoted parameter inside `min`). (4) the TMA `order=(1, 0)` tuples are
  scheduling metadata the DSL erases (the `matmul_tma` precedent).
- **Kernel 2** — (1) the descending loop is presented as the ascending
  change of variable `for j in range(0, numKBlocks, 1)` with the remaining
  count rematerialized as the first body statement `k = K - j*BLOCK_K`
  (ℕ-truncated; the `parallel_retention_attention` descending-lever
  precedent). (2) `ACC_TYPE: tl.constexpr = tl.int32` is fixed at its only
  instantiation (`bmm_optimized` fixed-arm family). (3) the
  `tl.broadcast_to(…, mask.shape)` attribute argument is respelled to the
  literal dims `[BLOCK_M, BLOCK_N]`. (4) `eviction_policy="evict_last"` is
  kept and erased as a pure cache hint. (5) widths erased; `other=0.0` on
  `.int` loads is the faithful `Op.constInt 0`. The `.real`-typed store
  into the int32-allocated `C` is honest: the cell carries the **exact
  real** product; the hardware float→int32 container truncation is outside
  the model (#154 fixed-width family).

Hypothesis design: kernel 1 needs only `0 < BLOCK_K` (positive range step)
plus `hInj` on the store address map (`mmCAddr_injective_rowMajor`
discharges it for row-major `C`) — the boundary checks make every axis
ragged-safe with **no divisibility of `K`**. Kernel 2 takes `hCeil : K ≤
numKBlocks·BLOCK_K` (the ceil form; extra masked iterations contribute
nothing) and `hEven : EVEN_K = true → K = numKBlocks·BLOCK_K` (the exact
form, demanded only of the unmasked arm); its flat store address needs **no
injectivity hypothesis** — on masked lanes `col < N`, so `col + N·row` is
injective outright (`smCAddr_inj_active`). Every dimension, stride and
block size stays symbolic; grids are 1-D, so there is no auxiliary
program-id hypothesis.
