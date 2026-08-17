# matmul_dequant_int4

- Source file: `matmul_dequant_int4.py` (upstream `data/TritonBench_G_v1/matmul_dequant_int4.py`)
- Corpus: TritonBench-G v1
- Size: 302 lines, 2 `@triton.jit` kernels
- Status: **PORTED** — `MatmulDequantInt4.lean`, main theorem
  `matmul_dequant_int4_exec_genuine` (`exec`-level, dimension-general,
  0 `sorry`).

Target JIT = `dequantize_kernel`, the only kernel any host in the file
launches (`matmul_dequantize_int4_s1` → `dequantize_int4` →
`dequantize_kernel[grid]`, then a host-side `torch.mm(a, fp_b)`; the test
exercises `matmul_dequantize_int4_s1` only). A straight-line int4
dequantization tile: program `(k_block_idx, n_block_idx)` loads a
`[BLOCK_SIZE_K, BLOCK_SIZE_N]` tile of packed weight words (`b`, eight 4-bit
weights per 32-bit word along K), the matching packed zero-point words
(`zp`, eight 4-bit zero-points per word along N, one group row per
`group_size` K rows) and the per-group scales, unpacks both nibbles with a
shift-and-mask, and stores `fp_weight = (nib(b) − nib(zp)) · scale` under
the shared mask `(offs_n < N) & (offs_k < K)`.

The file's first kernel, `matmul4_kernel`, is dead code in this file — no
host references it, and its body is byte-identical to the kernel already
ported in the `matmul_dequantize_int4` port — so, per the `rms_rbe_matmul`
dead-kernel precedent, it is not modeled.

Two disclosed surface decisions (the module's `Translation-surface
blocker:` marker):

- **Signed dequant via explicit casts** — the packed containers are
  `Region .nat` (the DSL's bitwise `>>` / `&` live on that channel), but the
  nibble difference goes negative and ℕ subtraction truncates, so the
  subtraction is spelled `tl.cast((int32_b >> b_shift) & 0xF, tl.int32) -
  tl.cast((zp_b >> bzp_shift) & 0xF, tl.int32)` and the product promotes to
  ℝ through `Op.intToReal` — the `int4_matmul` signed-dequant respell,
  applied inline.
- **Antiquoted integer literals** — the source's `8` / `4` / `0xF` /
  `group_size` inside index arithmetic are written `$(8)` / `$(4)` /
  `$(15)` / `$(group_size)` (a bare literal is inferred `.real`; `0xF` keeps
  its value in decimal spelling — the `int4_matmul` precedent).

The three masked loads keep `other=0.0` faithfully (on the `.nat` channels
the expander reads `0.0` as the nat constant `0` — value-identical). Every
dimension, stride, block size and `group_size` stays symbolic, and both
program ids are free, so the headline covers every autotune config and every
grid program at once. Nat division is total in Lean, so no `0 < group_size`
hypothesis is needed, and there is no divisibility side condition (loads and
store share one mask). The output readback needs distinct lanes at distinct
`fp_b` addresses (`hInj`); `fpbAddr_injective` discharges it for the host's
row-major `fp_b`.
