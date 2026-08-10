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

/-! ## The group-swizzled block coordinates

`pid` is decomposed exactly as the source decomposes it, so the spec below is
stated in the same coordinates the kernel computes rather than in a re-derived
form. -/

/-- `num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)`. -/
def numPidM (M BM : Nat) : Nat := (M + BM - 1) / BM

/-- `num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)`. -/
def numPidN (N BN : Nat) : Nat := (N + BN - 1) / BN

/-- `num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)` — the K-loop trip count. -/
def numPidK (K BK : Nat) : Nat := (K + BK - 1) / BK

/-- `num_pid_in_group = GROUP_SIZE_M * num_pid_n`. -/
def numPidInGroup (N BN GM : Nat) : Nat := GM * numPidN N BN

/-- `group_id = pid // num_pid_in_group`. -/
def groupId (s : BlockState) (N BN GM : Nat) : Nat :=
  s.pids 0 / numPidInGroup N BN GM

/-- `first_pid_m = group_id * GROUP_SIZE_M`. -/
def firstPidM (s : BlockState) (N BN GM : Nat) : Nat := groupId s N BN GM * GM

/-- `group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)`. -/
def groupSizeM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  min (numPidM M BM - firstPidM s N BN GM) GM

/-- `pid_m = first_pid_m + (pid % group_size_m)`. -/
def pidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  firstPidM s N BN GM + s.pids 0 % groupSizeM s M N BM BN GM

/-- `pid_n = (pid % num_pid_in_group) // group_size_m`. -/
def pidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % numPidInGroup N BN GM / groupSizeM s M N BM BN GM


/-! ## Masked element accessors

Each is the kernel's own address arithmetic, guarded by the mask the matching load
carries. All read the **launch** state's memory. -/

/-- `a[r, k*BK + e]` at K step `k`. Masked on the row axis only — that is the
kernel's `a_mask = offs_am[:, None] < M` with `other=0.0`. There is deliberately
no K-axis mask: the source documents `K % BLOCK_SIZE_K == 0` as a precondition,
and transcribing a mask it does not have would be unfaithful. -/
noncomputable def aElem (s : BlockState) (a : RegionName)
    (M stride_am stride_ak BM BK pm : Nat) (r k e : Nat) : ℝ :=
  if pm * BM + r < M then
    s.readMem a ((pm * BM + r) * stride_am + (e + k * BK) * stride_ak)
  else 0

/-- The packed 32-bit word holding the weight nibble for `(k, e, c)`. Unmasked,
matching the source's bare `tl.load(b_ptrs)`. Eight weights share a word along K,
hence the `e / 8`. -/
def bWord (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BN BK pn : Nat) (k e c : Nat) : Nat :=
  s.readMemValue .nat b
    ((e / 8 + k * (BK / 8)) * stride_bk + (pn * BN + c) * stride_bn)

/-- The unpacked 4-bit weight: `(word >> (e % 8) * 4) & 0xF`. -/
def bNibble (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BN BK pn : Nat) (k e c : Nat) : Nat :=
  bWord s b stride_bk stride_bn BN BK pn k e c >>> (e % 8 * 4) &&& 15

/-- The group row read at K step `k`. Under `NO_GROUPS` the scales and zeros are
loaded once before the loop **from the base pointers**, i.e. at row `0`; otherwise
each step reloads at `k // (groupsize // BLOCK_SIZE_K)`. -/
def groupRow (NO_GROUPS : Bool) (groupsize BK k : Nat) : Nat :=
  if NO_GROUPS then 0 else k / (groupsize / BK)

/-- `scales[g, pn*BN + c]`. -/
noncomputable def scalesElem (s : BlockState) (scales : RegionName)
    (stride_scales_g stride_scales_n BN pn : Nat) (g c : Nat) : ℝ :=
  s.readMem scales ((pn * BN + c) * stride_scales_n + g * stride_scales_g)

/-- The packed word holding the zero-point nibble for column `pn*BN + c`. Eight
zero-points share a word along N, hence the `/ 8`. -/
def zerosWord (s : BlockState) (zeros : Region .nat)
    (stride_zeros_g stride_zeros_n BN pn : Nat) (g c : Nat) : Nat :=
  s.readMemValue .nat zeros
    ((pn * BN + c) / 8 * stride_zeros_n + g * stride_zeros_g)

/-- The unpacked 4-bit zero-point. -/
def zerosNibble (s : BlockState) (zeros : Region .nat)
    (stride_zeros_g stride_zeros_n BN pn : Nat) (g c : Nat) : Nat :=
  zerosWord s zeros stride_zeros_g stride_zeros_n BN pn g c
    >>> ((pn * BN + c) % 8 * 4) &&& 15

/-- `zeros` after the kernel's `zeros = zeros * scales`: the **scaled**
zero-point. Both branches apply this before the loop body uses it. -/
noncomputable def zeroScaled (s : BlockState) (scales : RegionName)
    (zeros : Region .nat)
    (stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n BN pn : Nat)
    (g c : Nat) : ℝ :=
  (zerosNibble s zeros stride_zeros_g stride_zeros_n BN pn g c : ℝ)
    * scalesElem s scales stride_scales_g stride_scales_n BN pn g c

/-- `b` after unpack, scale and zero-point shift at K step `k`:
`b * scales - zeros`, where `zeros` is already scaled. -/
noncomputable def bDequant (s : BlockState) (scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (groupsize stride_bk stride_bn stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BN BK pn : Nat) (k e c : Nat) : ℝ :=
  (bNibble s b stride_bk stride_bn BN BK pn k e c : ℝ)
      * scalesElem s scales stride_scales_g stride_scales_n BN pn
          (groupRow NO_GROUPS groupsize BK k) c
    - zeroScaled s scales zeros stride_scales_g stride_scales_n
        stride_zeros_g stride_zeros_n BN pn (groupRow NO_GROUPS groupsize BK k) c


/-! ## The accumulator

`tl.dot(a, b)` over the block's K extent, summed across the `num_pid_k` steps. -/

/-- One K step's contribution to output cell `(r, c)`. -/
noncomputable def accStep (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn : Nat) (k r c : Nat) : ℝ :=
  ∑ e : Fin BK,
    aElem s a M stride_am stride_ak BM BK pm r k e.val
      * bDequant s scales b zeros NO_GROUPS groupsize stride_bk stride_bn
          stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
          BN BK pn k e.val c

/-- **The stored value.** `accumulator` after all `num_pid_k` K steps. (The source
also computes `c = accumulator.to(...)` but stores `accumulator`, so this is what
lands in memory.) -/
noncomputable def accSpec (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M K groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn : Nat) (r c : Nat) : ℝ :=
  ∑ k : Fin (numPidK K BK),
    accStep s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
      stride_bk stride_bn stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BM BN BK pm pn k.val r c

/-- The `C` store address for output cell `(r, c)`. -/
def cAddr (stride_cm stride_cn BM BN pm pn : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  stride_cm * (pm * BM + idx.1.val) + stride_cn * (pn * BN + idx.2.1.val)

/-! ## Compiled body decomposition

The algorithm-lowered statement lists, checked against the macro output by `rfl`
rather than assumed. Lowerings worth naming because they are not guessable from
the source text:

* `//` and `%` lower to `Op.floorDiv` / `Op.mod` on `IntegralDType.nat`, **not** to
  `Op.div` — `Op.div .nat` is what `tl.cdiv` expands to;
* `min(a, b)` is `Op.where (Op.lt …) a b` (there is no `Op.min`);
* a pointer expression `R + off` is `Op.ptrAdd Broadcast.scalarL (Op.ptrBase R) off`;
* **the rank-broadcast load mask** `a_mask : [BM, 1]` used on a `[BM, BK]` load is
  wrapped by the DSL in `Op.remap … Broadcast.leftIndex`, and the `other=0.0`
  makes it a `MaskOpt.maskOther` against `Op.broadcast (Op.const 0.0) [BM, BK]`;
* `nat * real` inserts `Op.natToReal` on the nat side. -/

/-- The compiled prologue: the two packing constants, the swizzled block
coordinates, the four pointer tiles, the two shift vectors, the `NO_GROUPS`
pre-load, and the zeroed accumulator. -/
def mdqPreLoop (a_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn
      stride_scales_n stride_zeros_n : Nat) (NO_GROUPS : Bool)
    (BM BN BK GM : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "bits" (Op.constNat 4),
    Stmt.assign .nat [] "infearure_per_bits" (Op.constNat 8),
    Stmt.assign .nat [] "pid" (Op.programId 0),
    Stmt.assign .nat [] "num_pid_m"
      (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
        (Op.constNat BM)),
    Stmt.assign .nat [] "num_pid_n"
      (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
        (Op.constNat BN)),
    Stmt.assign .nat [] "num_pid_k"
      (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK)) (Op.constNat 1))
        (Op.constNat BK)),
    Stmt.assign .nat [] "num_pid_in_group"
      (Op.mul .nat Broadcast.nil (Op.constNat GM) (Op.ref .nat [] "num_pid_n")),
    Stmt.assign .nat [] "group_id"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "num_pid_in_group")),
    Stmt.assign .nat [] "first_pid_m"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM)),
    Stmt.assign .nat [] "group_size_m"
      (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
            (Op.ref .nat [] "first_pid_m"))
          (Op.constNat GM))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
          (Op.ref .nat [] "first_pid_m"))
        (Op.constNat GM)),
    Stmt.assign .nat [] "pid_m"
      (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "group_size_m"))),
    Stmt.assign .nat [] "pid_n"
      (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "num_pid_in_group"))
        (Op.ref .nat [] "group_size_m")),
    Stmt.assign .nat [BM] "offs_am"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
        (Op.arange BM)),
    Stmt.assign .nat [BN] "offs_bn"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
        (Op.arange BN)),
    Stmt.assign .nat [BK] "offs_k" (Op.arange BK),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase a_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
            (Op.constNat stride_am))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_ak)))),
    Stmt.assign .bool [BM, 1] "a_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat M)),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase b_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.floorDiv IntegralDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
              (Op.ref .nat [] "infearure_per_bits"))
            (Op.constNat stride_bk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
            (Op.constNat stride_bn)))),
    Stmt.assign .ptr [BN] "scales_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase scales_ptr)
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn")
          (Op.constNat stride_scales_n))),
    Stmt.assign .ptr [BN] "zeros_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase zeros_ptr)
        (Op.mul .nat Broadcast.scalarR
          (Op.floorDiv IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn")
            (Op.ref .nat [] "infearure_per_bits"))
          (Op.constNat stride_zeros_n))),
    Stmt.assign .nat [BK] "shifter"
      (Op.mul .nat Broadcast.scalarR
        (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BK] "offs_k")
          (Op.ref .nat [] "infearure_per_bits"))
        (Op.ref .nat [] "bits")),
    Stmt.assign .nat [BN] "zeros_shifter"
      (Op.mul .nat Broadcast.scalarR
        (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn")
          (Op.ref .nat [] "infearure_per_bits"))
        (Op.ref .nat [] "bits")),
    Stmt.ifThen (Op.constBool NO_GROUPS)
      [ Stmt.assign .real [BN] "scales"
          (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN] "scales_ptrs")) MaskOpt.none),
        Stmt.assign .nat [BN] "zeros"
          (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BN] "zeros_ptrs")) MaskOpt.none),
        Stmt.assign .nat [BN] "zeros"
          (Op.bitAnd Broadcast.scalarR
            (Op.shiftRight (Broadcast.consSame Broadcast.nil) (Op.ref .nat [BN] "zeros")
              (Op.ref .nat [BN] "zeros_shifter"))
            (Op.constNat 15)),
        Stmt.assign .real [BN] "zeros"
          (Op.mul .real (Broadcast.consSame Broadcast.nil)
            (Op.natToReal (Op.ref .nat [BN] "zeros"))
            (Op.ref .real [BN] "scales")) ],
    Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)) ]

set_option maxRecDepth 8000 in
/-- The prologue is the first 24 statements of the lowered body, by `rfl`. -/
theorem mdq_preLoop_eq (a_ptr c_ptr scales_ptr : RegionName)
    (b_ptr zeros_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool) (BM BN BK GM : Nat) :
    ((matmul_dequantize_int4_surface a_ptr c_ptr scales_ptr b_ptr zeros_ptr
        M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
        groupsize NO_GROUPS BM BN BK GM).toAlgKernel.body.take 24)
      = mdqPreLoop a_ptr scales_ptr b_ptr zeros_ptr M N K stride_am stride_ak
          stride_bk stride_bn stride_scales_n stride_zeros_n NO_GROUPS BM BN BK GM := by
  rfl

/-- The compiled K-loop body: the two loads, the `not NO_GROUPS` per-group reload,
the unpack / scale / shift of `b`, the `tl.dot` accumulation, and the two pointer
advances. -/
def mdqLoopBody
    (groupsize stride_ak stride_bk stride_scales_g stride_zeros_g : Nat)
    (NO_GROUPS : Bool) (BM BN BK : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BK] "a"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        (MaskOpt.maskOther
          (Op.remap [BM, BK] (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
            (Op.ref .bool [BM, 1] "a_mask"))
          (Op.broadcast (Op.const 0.0) [BM, BK]))),
    Stmt.assign .nat [BK, BN] "b"
      (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BK, BN] "b_ptrs")) MaskOpt.none),
    Stmt.ifThen (Op.boolNot (Op.constBool NO_GROUPS))
      [ Stmt.assign .nat [] "g_id"
          (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "k")
            (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat groupsize)
              (Op.constNat BK))),
        Stmt.assign .ptr [BN] "ptr"
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BN] "scales_ptrs")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "g_id")
              (Op.constNat stride_scales_g))),
        Stmt.assign .real [BN] "scales"
          (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN] "ptr")) MaskOpt.none),
        Stmt.assign .ptr [BN] "ptr"
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BN] "zeros_ptrs")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "g_id")
              (Op.constNat stride_zeros_g))),
        Stmt.assign .nat [BN] "zeros"
          (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BN] "ptr")) MaskOpt.none),
        Stmt.assign .nat [BN] "zeros"
          (Op.bitAnd Broadcast.scalarR
            (Op.shiftRight (Broadcast.consSame Broadcast.nil) (Op.ref .nat [BN] "zeros")
              (Op.ref .nat [BN] "zeros_shifter"))
            (Op.constNat 15)),
        Stmt.assign .real [BN] "zeros"
          (Op.mul .real (Broadcast.consSame Broadcast.nil)
            (Op.natToReal (Op.ref .nat [BN] "zeros"))
            (Op.ref .real [BN] "scales")) ],
    Stmt.assign .nat [BK, BN] "b"
      (Op.bitAnd Broadcast.scalarR
        (Op.shiftRight (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .nat [BK, BN] "b")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "shifter")))
        (Op.constNat 15)),
    Stmt.assign .real [BK, BN] "b"
      (Op.sub .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
          (Op.natToReal (Op.ref .nat [BK, BN] "b"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BN] "scales")))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BN] "zeros"))),
    Stmt.assign .real [BM, BN] "accumulator"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "accumulator")
        (Op.dot (batch := []) (Op.ref .real [BM, BK] "a") (Op.ref .real [BK, BN] "b"))),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_ak))),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "b_ptrs")
        (Op.mul .nat Broadcast.nil
          (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat BK)
            (Op.ref .nat [] "infearure_per_bits"))
          (Op.constNat stride_bk))) ]

/-- The compiled tail: the dead `c` binding, the output coordinates, the `C`
pointer tile, the two-axis store mask, and the store — which writes
`accumulator`, not `c`. -/
def mdqPostLoop (c_ptr : RegionName) (M N stride_cm stride_cn BM BN : Nat) :
    List Stmt :=
  [ Stmt.assign .real [BM, BN] "c" (Op.ref .real [BM, BN] "accumulator"),
    Stmt.assign .nat [BM] "offs_cm"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
        (Op.arange BM)),
    Stmt.assign .nat [BN] "offs_cn"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
        (Op.arange BN)),
    Stmt.assign .ptr [BM, BN] "c_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase c_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_cm)
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_cn)
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))),
    Stmt.assign .bool [BM, BN] "c_mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))),
    Stmt.store .real [BM, BN] (MemAccess.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
      (Op.ref .real [BM, BN] "accumulator")
      (MaskOpt.mask (Op.ref .bool [BM, BN] "c_mask")) ]

set_option maxRecDepth 20000 in
/-- **Full body split (by `rfl`).** The lowered surface is exactly
`mdqPreLoop ++ [forRangeDyn "k" 0 num_pid_k 1 mdqLoopBody] ++ mdqPostLoop`
— 31 top-level statements, every one checked against the macro output. -/
theorem mdq_body_eq (a_ptr c_ptr scales_ptr : RegionName)
    (b_ptr zeros_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool) (BM BN BK GM : Nat) :
    (matmul_dequantize_int4_surface a_ptr c_ptr scales_ptr b_ptr zeros_ptr
        M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
        groupsize NO_GROUPS BM BN BK GM).toAlgKernel.body
      = mdqPreLoop a_ptr scales_ptr b_ptr zeros_ptr M N K stride_am stride_ak
            stride_bk stride_bn stride_scales_n stride_zeros_n NO_GROUPS BM BN BK GM
        ++ [Stmt.forRangeDyn "k" (Op.constNat 0) (Op.ref .nat [] "num_pid_k")
              (Op.constNat 1)
              (mdqLoopBody groupsize stride_ak stride_bk
                stride_scales_g stride_zeros_g NO_GROUPS BM BN BK)]
        ++ mdqPostLoop c_ptr M N stride_cm stride_cn BM BN := by
  rfl

/-! ## Pointer tiles

A `.ptr` tile carries one `(region, offset)` pair per lane, and a `.ptr` load is
unconditionally in bounds — the mask is the only gate. `a_ptrs` and `b_ptrs` are
advanced once per K step, so each is a function of the step `k`, and the two
`mdq*Ptrs_succ` lemmas are what lets the invariant carry across an iteration. -/

/-- `a_ptrs` lane `(r, e)` at K step `k`, in the accumulated form the kernel
actually reaches: the initial offset plus `k` advances of `BK * stride_ak`. -/
def mdqAAddr (stride_am stride_ak BM BK pm k : Nat) (idx : TileIndex [BM, BK]) : Nat :=
  (pm * BM + idx.1.val) * stride_am + idx.2.1.val * stride_ak + k * (BK * stride_ak)

/-- `b_ptrs` lane `(e, c)` at K step `k`. Eight weights share a word along K, so a
step advances by `(BK / 8) * stride_bk`. -/
def mdqBAddr (stride_bk stride_bn BN BK pn k : Nat) (idx : TileIndex [BK, BN]) : Nat :=
  idx.1.val / 8 * stride_bk + (pn * BN + idx.2.1.val) * stride_bn
    + k * (BK / 8 * stride_bk)

noncomputable def mdqAPtrs (a : RegionName) (stride_am stride_ak BM BK pm k : Nat) :
    Tile .ptr [BM, BK] :=
  ⟨fun idx => (a, mdqAAddr stride_am stride_ak BM BK pm k idx)⟩

noncomputable def mdqBPtrs (b : Region .nat)
    (stride_bk stride_bn BN BK pn k : Nat) : Tile .ptr [BK, BN] :=
  ⟨fun idx => (Region.cast b, mdqBAddr stride_bk stride_bn BN BK pn k idx)⟩

/-- One `a_ptrs += BLOCK_SIZE_K * stride_ak` advance. -/
theorem mdqAPtrs_succ (a : RegionName) (stride_am stride_ak BM BK pm k : Nat) :
    Tile.ptrAdd Broadcast.scalarR (mdqAPtrs a stride_am stride_ak BM BK pm k)
        (Tile.scalar (BK * stride_ak))
      = mdqAPtrs a stride_am stride_ak BM BK pm (k + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, mdqAPtrs, mdqAAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

/-- One `b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk` advance. -/
theorem mdqBPtrs_succ (b : Region .nat) (stride_bk stride_bn BN BK pn k : Nat) :
    Tile.ptrAdd Broadcast.scalarR (mdqBPtrs b stride_bk stride_bn BN BK pn k)
        (Tile.scalar (BK / 8 * stride_bk))
      = mdqBPtrs b stride_bk stride_bn BN BK pn (k + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, mdqBPtrs, mdqBAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

/-- The `a_ptrs` address agrees with `aElem`'s: the kernel's accumulated
`e*stride_ak + k*(BK*stride_ak)` is `aElem`'s `(e + k*BK)*stride_ak`. -/
theorem mdqAAddr_eq (stride_am stride_ak BM BK pm k : Nat) (idx : TileIndex [BM, BK]) :
    mdqAAddr stride_am stride_ak BM BK pm k idx
      = (pm * BM + idx.1.val) * stride_am + (idx.2.1.val + k * BK) * stride_ak := by
  simp only [mdqAAddr]
  ring

/-- The `b_ptrs` address agrees with `bWord`'s. -/
theorem mdqBAddr_eq (stride_bk stride_bn BN BK pn k : Nat) (idx : TileIndex [BK, BN]) :
    mdqBAddr stride_bk stride_bn BN BK pn k idx
      = (idx.1.val / 8 + k * (BK / 8)) * stride_bk
        + (pn * BN + idx.2.1.val) * stride_bn := by
  simp only [mdqBAddr]
  ring

/-! ## The loop body's value tiles

Each is named in the form the eval recipes will produce, with a separate `_data`
bridge to the spec accessor — the same split that kept `chunk_bwd_dqkg`'s address
arithmetic out of its step proofs. -/

/-- `a_mask` as the prologue leaves it: a `[BM, 1]` bool tile, `true` exactly on
in-region rows. -/
def mdqAMask (M BM pm : Nat) : Tile .bool [BM, 1] :=
  ⟨fun idx => decide (pm * BM + idx.1.val < M)⟩

/-- `shifter[e] = (e % 8) * 4` — `offs_k` is `arange BK`, so lane `e` holds `e`. -/
def mdqShifter (BK : Nat) : Tile .nat [BK] := ⟨fun idx => idx.1.val % 8 * 4⟩

/-- `zeros_shifter[c] = ((pn*BN + c) % 8) * 4`. -/
def mdqZerosShifter (BN pn : Nat) : Tile .nat [BN] :=
  ⟨fun idx => (pn * BN + idx.1.val) % 8 * 4⟩

/-- `a` at K step `k`. In-region rows read memory; the rest read the load's
`other=0.0`. The mask is rank-broadcast from `[BM, 1]`, so it does not depend on
the K lane — which is exactly why there is no K-axis boundary handling. -/
noncomputable def mdqATile (s : BlockState) (a : RegionName)
    (M stride_am stride_ak BM BK pm k : Nat) : Tile .real [BM, BK] :=
  ⟨fun idx => if pm * BM + idx.1.val < M then
      some (s.readMem a (mdqAAddr stride_am stride_ak BM BK pm k idx)) else some 0⟩

/-- The loaded packed weight words at K step `k`. Unmasked. -/
noncomputable def mdqBRaw (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BN BK pn k : Nat) : Tile .nat [BK, BN] :=
  ⟨fun idx => s.readMemValue .nat b (mdqBAddr stride_bk stride_bn BN BK pn k idx)⟩

/-- `b` after the nibble extraction. -/
noncomputable def mdqBNibbleTile (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BN BK pn k : Nat) : Tile .nat [BK, BN] :=
  ⟨fun idx => bNibble s b stride_bk stride_bn BN BK pn k idx.1.val idx.2.1.val⟩

/-- `accumulator` after `i` K steps: the partial sum the invariant carries. -/
noncomputable def mdqAccTile (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn : Nat) (i : Nat) : Tile .real [BM, BN] :=
  ⟨fun idx => some (∑ j : Fin i,
      accStep s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
        stride_bk stride_bn stride_scales_g stride_scales_n
        stride_zeros_g stride_zeros_n BM BN BK pm pn j.val idx.1.val idx.2.1.val)⟩

/-- At `i = 0` the accumulator is the zero tile `tl.zeros` produces. -/
theorem mdqAccTile_zero (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn : Nat) :
    mdqAccTile s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
        stride_bk stride_bn stride_scales_g stride_scales_n
        stride_zeros_g stride_zeros_n BM BN BK pm pn 0
      = (⟨fun _ => some 0⟩ : Tile .real [BM, BN]) := by
  apply Tile.ext
  intro idx
  simp [mdqAccTile]

/-- At `i = numPidK` the accumulator is `accSpec` — the stored value. -/
theorem mdqAccTile_full (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M K groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn : Nat) (idx : TileIndex [BM, BN]) :
    (mdqAccTile s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
        stride_bk stride_bn stride_scales_g stride_scales_n
        stride_zeros_g stride_zeros_n BM BN BK pm pn (numPidK K BK)).data idx
      = some (accSpec s a scales b zeros NO_GROUPS M K groupsize stride_am
          stride_ak stride_bk stride_bn stride_scales_g stride_scales_n
          stride_zeros_g stride_zeros_n BM BN BK pm pn idx.1.val idx.2.1.val) := by
  rfl

/-- **The invariant's accumulator step.** Extending the partial sum by one K step
adds that step's `tl.dot` contribution — the algebraic content of the fold. -/
theorem mdqAccTile_succ (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn i : Nat) (idx : TileIndex [BM, BN]) :
    (mdqAccTile s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
        stride_bk stride_bn stride_scales_g stride_scales_n
        stride_zeros_g stride_zeros_n BM BN BK pm pn (i + 1)).data idx
      = some ((∑ j : Fin i,
            accStep s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
              stride_bk stride_bn stride_scales_g stride_scales_n
              stride_zeros_g stride_zeros_n BM BN BK pm pn j.val
              idx.1.val idx.2.1.val)
          + accStep s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
              stride_bk stride_bn stride_scales_g stride_scales_n
              stride_zeros_g stride_zeros_n BM BN BK pm pn i
              idx.1.val idx.2.1.val) := by
  simp only [mdqAccTile]
  exact congrArg some (Fin.sum_univ_castSucc _)


/-! ### Bridges from the tiles to the spec accessors -/

theorem mdqATile_data (s : BlockState) (a : RegionName)
    (M stride_am stride_ak BM BK pm k : Nat) (idx : TileIndex [BM, BK]) :
    (mdqATile s a M stride_am stride_ak BM BK pm k).data idx
      = some (aElem s a M stride_am stride_ak BM BK pm idx.1.val k idx.2.1.val) := by
  simp only [mdqATile, aElem, mdqAAddr_eq]
  split <;> rfl

theorem mdqBRaw_data (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BN BK pn k : Nat) (idx : TileIndex [BK, BN]) :
    (mdqBRaw s b stride_bk stride_bn BN BK pn k).data idx
      = bWord s b stride_bk stride_bn BN BK pn k idx.1.val idx.2.1.val := by
  simp only [mdqBRaw, bWord, mdqBAddr_eq]

/-- The nibble tile is `bitAnd 15` over `shiftRight`, applied to the raw words —
i.e. exactly the two compiled statements, read back through `bNibble`. -/
theorem mdqBNibbleTile_eq (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BN BK pn k : Nat) :
    mdqBNibbleTile s b stride_bk stride_bn BN BK pn k
      = Tile.bop (· &&& ·) Broadcast.scalarR
          (Tile.bop (· >>> ·) (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (mdqBRaw s b stride_bk stride_bn BN BK pn k)
            (Tile.expandDim ⟨1, by simp⟩ (mdqShifter BK)))
          (Tile.scalar 15) := by
  apply Tile.ext
  intro idx
  simp only [mdqBNibbleTile, bNibble, Tile.bop_data, Tile.scalar,
    Tile.expandDim_data, mdqShifter, TileShape.dropInsertedIndex,
    Broadcast.leftIndex, Broadcast.rightIndex, mdqBRaw_data]

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4
