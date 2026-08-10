import VeriTile.Triton

/-!
# `matmul_dequantize_int4` — strict per-kernel correctness

`matmul_dequantize_int4.py`'s `matmul4_kernel` is a GPTQ-style dequantizing GEMM:
`C = A · dequant(qweight)`, where `qweight` holds **eight 4-bit weights packed per
32-bit word** along the K axis, and `qzeros` holds eight 4-bit zero-points packed
per word along the N axis. Each output block is accumulated over the K axis; on
every step the packed weights are unpacked with a shift and a nibble mask, scaled,
and shifted by the group's zero-point.

One program owns one `(pid_m, pid_n)` output block, chosen by the standard
group-swizzled `pid` decomposition (`GROUP_SIZE_M` rows of blocks at a time).

## The two dequantization channels

The packed tensors are `torch.IntTensor`, and both are modelled as `Region .nat`:
the DSL's bitwise operators (`>>`, `&`) are defined on the `.nat` channel only.
This is a statement about the *container*, not the extracted values —
`(x >> 4i) & 0xF` selects the same nibble whether the shift is logical or
arithmetic, and every extracted nibble is in `[0, 15]`, hence non-negative. The
unpacked values cross to `ℝ` immediately at `* scales`, so no integer step ever
needs to go negative.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` and the
host launch (the 1-D grid `cdiv(M,BM)·cdiv(N,BN)`, the block sizes, and
`NO_GROUPS = (group_size == K)`) are the *trusted boundary*. Every dimension,
stride and block size stays a symbolic parameter.

Two spelling notes, per `bench/MAIN_THEOREM_CONVENTIONS.md`, both surface syntax
rather than semantics:

* integer literals inside index arithmetic are written `$(n)`, since a bare
  literal is inferred `.real` by the DSL's expression typing. The source's `0xF`
  is written `$(15)` for the same reason — same value, decimal spelling.
* the source's `bits = 4` / `infearure_per_bits = 8` are ordinary assignments and
  are transcribed as such (they are `tl.constexpr`-free Python locals, so they
  lower to real statements, not to inlined constants).

## Faithfulness note worth flagging

The source computes `c = accumulator.to(c_ptr.dtype.element_ty)` and then stores
**`accumulator`**, not `c` — so `c` is a dead binding. That is transcribed as
written; the spec below is therefore about the `float32` accumulator.
-/

namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4

open VeriTile.Triton

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-! ## Kernel surface (faithful transcription) -/

def matmul_dequantize_int4_surface
    (a_ptr c_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ComputeKernel := triton {
  bits = $(4)
  infearure_per_bits = $(8)
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_k = tl.cdiv($(K), $(BLOCK_SIZE_K))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = a_ptr + (offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak))
  a_mask = (offs_am[:, None] < $(M))
  b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
  scales_ptrs = scales_ptr + offs_bn * $(stride_scales_n)
  zeros_ptrs = zeros_ptr + ((offs_bn // infearure_per_bits) * $(stride_zeros_n))
  shifter = (offs_k % infearure_per_bits) * bits
  zeros_shifter = (offs_bn % infearure_per_bits) * bits
  if NO_GROUPS {
    scales = tl.load(scales_ptrs)
    zeros = tl.load(zeros_ptrs)
    zeros = (zeros >> zeros_shifter) & $(15)
    zeros = zeros * scales
  }
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), num_pid_k, $(1)) {
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs)
    if not NO_GROUPS {
      g_id = k // ($(groupsize) // $(BLOCK_SIZE_K))
      ptr = scales_ptrs + g_id * $(stride_scales_g)
      scales = tl.load(ptr)
      ptr = zeros_ptrs + g_id * $(stride_zeros_g)
      zeros = tl.load(ptr)
      zeros = (zeros >> zeros_shifter) & $(15)
      zeros = (zeros) * scales
    }
    b = (b >> shifter[:, None]) & $(15)
    b = b * scales[None, :] - zeros[None, :]
    accumulator += tl.dot(a, (b).to(a.dtype))
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += ($(BLOCK_SIZE_K) // infearure_per_bits) * $(stride_bk)
  }
  c = (accumulator).to(c_ptr.dtype.element_ty)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = c_ptr + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, accumulator, mask=c_mask)
}

theorem matmul_dequantize_int4_surface_toAlgorithm_supported
    (a_ptr c_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ∃ alg, (matmul_dequantize_int4_surface a_ptr c_ptr scales_ptr b_ptr zeros_ptr
      M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize NO_GROUPS
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M).toAlgorithm?
        = Except.ok alg := by
  simp [matmul_dequantize_int4_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4
