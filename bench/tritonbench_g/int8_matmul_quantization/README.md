# int8_matmul_quantization

- Source file: `int8_matmul_quantization.py` (upstream `data/TritonBench_G_v1/int8_matmul_quantization.py`)
- Corpus: TritonBench-G v1
- Size: 267 lines, 2 `@triton.jit` kernels — **both launched, both modeled**
- Status: **PORTED** — `Int8MatmulQuantization.lean`, main theorems
  `int8_matmul_quantization_quantize_exec_genuine` and
  `int8_matmul_quantization_matmul_exec_genuine` (`exec`-level,
  dimension-general, 0 `sorry`).

Kernel 1 = `quantize_int8_perrow_kernel` (the audit anchor; launcher
`quantize_int8_perrow`), kernel 2 = `matmul_kernel` (launcher `matmul_int8`);
the test drives both through `matmul_quantize_int8`. Together they form a
dynamic-quantization pipeline: kernel 1 streams each row of the float matrix
`fpa` over K blocks **twice** — pass 1 accumulates the per-row running
abs-max over K-masked (`other=0.0`) tiles, then `a_scale = a_max / 127.`;
pass 2 reloads, quantizes `(fpa / a_scale[:, None]).to(tl.int8)` and
K-mask-stores the `.int` tile, and finally stores the per-row scale into an
fp16 vector. Kernel 2 is the consuming int8×int8→int32 GEMM: group-swizzled
pids, `% M` / `% N` wrapped offsets, remainder-masked `.int` K loads feeding
the exact ℤ `tl.dot` (`Op.dotInt`), and the epilogue
`(acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(tl.float16)`
under a two-axis-masked fp16 store.

Disclosed surface decisions (the module's two `Translation-surface
blocker:` markers, one per kernel):

- **Ceil-form `numKBlocks`** (both kernels) — the `tl.cdiv` trip counts are
  antiquoted binders with the *covering-half* side condition
  `K ≤ numKBlocks · BLOCK_SIZE_K` only: the K loads/stores are
  remainder-masked with `other=0.0`, so extra all-masked blocks contribute
  `0` and store nothing — honest at ragged `K` (unlike the exact-multiple
  presentations of the unmasked-loop matmul ports).
- **`SPLIT_K` fixed to `1`** (kernel 2, the `tl.store` arm): the autotune
  table sweeps `SPLIT_K ∈ {1, 2}`; the `tl.atomic_add` arm (into the
  host-zeroed `C` of `reset_to_zero=['c_ptr']`) drops with the constexpr;
  the headline carries `s.pids 1 = 0`.
- **`accumulator.to(tl.float32)` via the implicit promotion** (kernel 2):
  the explicit spelling is int-typed-ident-blocked in the DSL, so the
  epilogue product is spelled bare and the lowered term carries
  `Op.intToReal` at exactly the Python cast's site.
- **Implicit fp16 scale-store cast** (kernel 1): the host allocates
  `a_scale` as `torch.float16`, so the bare `tl.store(as_ptr + as_offs,
  a_scale)` is spelled `(a_scale).to(tl.float16)` and the scale cells read
  back as fp16-typed `MemCell`s. The quirky **unstrided** `as_offs` arange
  (`pid_m * BLOCK_SIZE_M * stride_asm + tl.arange(0, BLOCK_SIZE_M)`) is
  spelled faithfully.
- **`.to(tl.int8)`** (kernel 1) lowers to `Op.castRealToInt8` —
  truncation toward zero into the unbounded `.int` carrier; hardware
  saturation at `±127` is the unmodeled `#154`-family boundary.

Kernel 1's row window is wrapped (`% M`) but its stores are row-unmasked;
the headline carries the host's exact-tiling launch fact
`hFit : pid·BM + BM ≤ M` (grid = `M // BLOCK_SIZE_M` programs of
`BLOCK_SIZE_M = 1` rows), under which the wrap is the identity. Its single
`exec` statement carries **both** store channels: every `(row, kg)` int8
cell holds the truncated quotient against the kernel-computed streaming
abs-max scale, and every scale cell holds the fp16 image of that scale.
Kernel 2's headline reads
`((Σ_{j<K} A[row,j]·B[j,col] : ℤ) : ℝ) · a_scale[row] · b_scale[col]` — the
masked double sum collapses to the clean `∑ j : Fin K` under the ceil-form
hypothesis. Both headlines take store-map injectivity hypotheses (`hInj`,
row-major discharge lemma provided for kernel 2) and kernel 1 additionally
the region-distinctness facts its read-after-write loop and two-region tail
force.

One mechanical library rider shipped with this port: the DSL's `other=`
elaboration table gained the `.int` arm for the literal `0.0`
(`Op.constInt 0`), mirroring the existing `.nat` arm — kernel 2's masked
`.int` loads are its first consumer.
