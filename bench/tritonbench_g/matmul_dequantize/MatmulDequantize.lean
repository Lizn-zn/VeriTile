import VeriTile.Triton

/-!
# `matmul_dequantize` — strict per-kernel correctness, all three launched JIT kernels

This file is the DSL port of `matmul_dequantize.py`'s `matmul4_kernel` — the
audit anchor, the upstream file's first `@triton.jit` kernel, launched by
`matmul_dequantize_int4_gptq` (test case 1). The same upstream file defines and
launches two more JIT kernels, and both are modeled below, **in source order**,
following the `decay_cumsum` multi-kernel-file precedent: the second JIT is
`matmul_kernel` (launched by `matmul_dequantize_int4_s2`, test case 2) and the
third is `dequantize_kernel` (launched by `matmul_dequantize_int4_s1` →
`dequantize_int4`, test case 3, followed by a host-side `torch.mm(a, fp_b)`).
Three faithful surfaces, three disjoint proof stacks, three headlines:

* `matmul_dequantize_matmul4_exec_genuine` — `matmul4_kernel`, a GPTQ-style
  int4-dequantizing GEMM (`C = A · dequant(qweight)`; scale first, then
  subtract the pre-scaled zero-point) with a `NO_GROUPS : Bool` constexpr kept
  as a genuine parameter — **both arms** are covered by one statement through
  the `groupRow` unifier;
* `matmul_dequantize_matmul_exec_genuine` — `matmul_kernel`, the sibling
  int4-dequantizing GEMM that subtracts the zero-point **first** (a signed
  nibble difference) and then scales, with `% M` / `% N` wrapped offsets,
  unmasked loads, and an `SPLIT_K`-swept autotune table;
* `matmul_dequantize_dequantize_exec_genuine` — `dequantize_kernel`, the
  standalone tile dequantizer `fp_b[k, n] = (nib(b) − nib(zp)) · scale` with
  one shared load/store gate.

Each kernel is a light textual variant of a kernel already ported in this
bench: `matmul4_kernel` differs from the `matmul_dequantize_int4` port's twin
only in dropping that source's `.to(a.dtype)` on the `tl.dot` operand and in
spelling the dead `c`-cast as a direct `tl.float16`; `matmul_kernel` differs
from the `int4_matmul` port's twin only in its docstring, in spelling its
three working casts as direct `tl.float16` (where the twin's source wrote
`.to(a.dtype)` / `.to(c_ptr.dtype.element_ty)`, which the DSL erases); and
`dequantize_kernel` is **byte-identical** to the kernel modeled in the
`matmul_dequant_int4` port. Each section below mirrors its template's spec
accessors, invariants and proof architecture, renamed into this one namespace.

## `matmul4_kernel` (section 1)

One program owns one `(pid_m, pid_n)` output block chosen by the standard
group-swizzled `pid` decomposition; the K loop unpacks eight 4-bit weights per
32-bit word (`(b >> shifter) & 0xF`), scales, shifts by the group's pre-scaled
zero-point, and accumulates `tl.dot(a, b)`. Under `NO_GROUPS` the scales and
zeros are pre-loaded once (group row 0); otherwise the loop reloads them at
`k // (groupsize // BLOCK_SIZE_K)`.

Translation-surface blocker: (`matmul4_kernel`) one disclosed spelling family,
not semantic — integer literals inside index arithmetic are antiquoted `$(n)`
(a bare literal is inferred `.real` by the DSL's expression typing): the
source's `0xF` is written `$(15)` — same value, decimal spelling — and the
`tl.constexpr`-free Python locals `bits = 4` / `infearure_per_bits = 8` are
transcribed as ordinary statements (the `matmul_dequantize_int4` disclosure,
reused verbatim). Two faithfulness notes: **(a)** unlike the twin's source,
`accumulator += tl.dot(a, b)` carries no cast on `b` — none is transcribed;
**(b)** `c = accumulator.to(tl.float16)` spells its dtype directly, so it
compiles to a genuine `Op.castFloat real → fp16` — but it is still a **dead
binding**: the store writes `accumulator`, not `c`, exactly as the source
does, so the headline is about the float32 accumulator.

## `matmul_kernel` (section 2)

The `SPLIT_K = 1` arm of the sibling GEMM: per-K-step rebuilt scale /
zero-point pointers with per-lane group rows `(e + k·BK) // group_size`, a
signed nibble difference, wrapped `% M` / `% N` offsets with unmasked loads,
and a masked store of `c = accumulator.to(tl.float16)` — here `c` **is**
stored, and the direct `tl.float16` spelling makes the store an
`.fp16`-typed quantization event, so the headline reads the output back at
the `MemCell` level (`MemCell.of .fp16 (fp16(accSpec))`, the `matmul_tma`
f16-branch precedent; `FloatDType.cast` is the placeholder-identity, so the
carried value is `accSpec` itself). The in-loop
`b = ((int_b - int_bzp) * bs).to(tl.float16)` and the dot's
`(a).to(tl.float16)` / `(b).to(tl.float16)` survive **faithfully** as
`Op.castFloat` events (the `attention_fwd_triton2` `p.to(tl.float16)`
precedent); the DSL's `tl.dot` re-widens both fp16 operands to the ℝ carrier,
and the round-trips collapse by cast identity.

Translation-surface blocker: (`matmul_kernel`) three disclosed surface
deviations, none semantic — the `int4_matmul` disclosures (1)–(3) verbatim.
**(1)** `SPLIT_K` is fixed to `1` (the `tl.store` arm): the `@triton.autotune`
table sweeps `SPLIT_K` (the shown config pins 1; more are elided) and the
launch grid's second axis is `SPLIT_K`, so the `tl.atomic_add` arm
(accumulating into the host-zeroed `C` of `reset_to_zero=['c_ptr']`) is
dropped with the constexpr — the `bmm_optimized` fixed-arm precedent. Every
`* SPLIT_K` factor is folded to its `SPLIT_K = 1` value, and the headline
carries the matching launch fact `s.pids 1 = 0` (grid axis 1 has extent
`SPLIT_K = 1`). **(2)** the K-loop bound `tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)`
is spelled as the antiquoted binder `numKBlocks` with the honest side
condition `K = BLOCK_SIZE_K * numKBlocks` (the loads are unmasked, so the trip
count is exact — the `llama_ff_triton` `tl.cdiv` precedent). **(3)** the
dequant subtraction is spelled `(tl.cast(int_b, tl.int32) -
tl.cast(int_bzp, tl.int32)) * bs`: the two packed containers live on the
`.nat` channel (bitwise `>>` / `&` are defined there), the nibble difference
goes negative, and ℕ subtraction truncates — so the subtraction must run on
the signed `.int` channel and promote to ℝ via `Op.intToReal`
(`tl.cast(x, tl.int32)` is the explicit `Op.castNatToInt` spelling; a bare
`.to(tl.int32)` on a nat value is width-erased to a no-op). Unlike the twin,
the statement's trailing cast — here a direct `.to(tl.float16)` — is **kept**:
it does not route through the int-expected expansion, so it compiles
faithfully.

## `dequantize_kernel` (section 3)

One program owns the `[BLOCK_SIZE_K, BLOCK_SIZE_N]` tile at
`(k_block_idx, n_block_idx)` of the dequantized weight `fp_b`: it loads a tile
of packed int4 weight words (eight 4-bit weights per 32-bit word along K,
hence `offs_k // 8`), the matching packed zero-point words (eight per word
along N, one group row per `group_size` K rows), and the per-group scales,
unpacks the two nibbles, and stores `fp_weight = (nib(b) − nib(zp)) · scale` —
every load and the store masked by the same `(offs_n < N) & (offs_k < K)`
gate. The kernel body is byte-identical to the one already ported in
`matmul_dequant_int4`; this section is that port's proof stack, renamed.

Translation-surface blocker: (`dequantize_kernel`) two disclosed surface
deviations, neither semantic — the `matmul_dequant_int4` disclosures verbatim.
**(1)** the dequant statement is spelled
`fp_weight = (tl.cast((int32_b >> b_shift) & $(15), tl.int32) -
tl.cast((zp_b >> bzp_shift) & $(15), tl.int32)) * scale_b` — the same signed
`.int`-channel hop as `matmul_kernel`'s disclosure (3), applied inline.
**(2)** integer literals are antiquoted `$(n)` and the `group_size` runtime
argument is the antiquoted binder `$(group_size)`. The three masked loads keep
their `other=0.0` kwargs faithfully; on the two packed `.nat` channels the
expander reads the literal `0.0` as the nat constant `0` (value-identical, and
masked-off lanes are never consumed — the store mask equals the load mask).

## The two packed channels (all three kernels)

The packed weight / zero-point tensors are `torch.IntTensor`s and are modeled
as `Region .nat`: the DSL's bitwise operators are defined on the `.nat`
channel only. This is a statement about the *container*, not the extracted
values — `(x >> 4i) & 0xF` selects the same nibble whether the shift is
logical or arithmetic, and every extracted nibble is in `[0, 15]`, hence
non-negative. `matmul4_kernel` scales before subtracting, so its integers
never go negative; the other two kernels subtract nibbles first — hence their
explicit `.int` hop.

## Proof map

```
matmul_dequantize_matmul4_exec_genuine          headline 1  (matmul4_kernel)
├─ mdq_body_eq / mdqPreLoopScalars_run / mdqPreLoopTiles_run
├─ mdqLoop_collapse ── mdqLoopBody_run ── mdq_groupBranch_run (both NO_GROUPS paths)
└─ mdqPostLoop_run ── mdq_store_props          (cAddr_injective discharges hInj)
matmul_dequantize_matmul_exec_genuine           headline 2  (matmul_kernel)
├─ i4_body_eq / i4PreLoopScalars_run / i4PreLoopTiles_run
├─ i4Loop_collapse ── i4LoopBody_run (fp16 cast + double-cast dot recipes)
└─ i4PostLoop_run ── i4_store_props            (fp16 MemCell readback)
matmul_dequantize_dequantize_exec_genuine       headline 3  (dequantize_kernel)
├─ dq_exec_isSome                               one big simp, 18 statements
└─ dq_correct ── scatter_readback_prop_masked_nd  (fpbAddr_injective ⊢ hInj)
```

Every stored value is built bottom-up from each kernel's own accessors over
the **launch** state's memory; no output region is ever read back into a spec,
so no part of any trust path is self-referential.

## Modeling boundary

Arithmetic is over ℝ (not bit-accurate IEEE float; the fp16 cast sites are
recorded as `Op.castFloat` events whose placeholder cast is the identity);
`@triton.autotune` and the three host launches (grids, strides, `group_size`,
`NO_GROUPS = (group_size == K)`, the downstream `torch.mm`) are the *trusted
boundary*. Every dimension, stride and block size stays a symbolic parameter,
so each headline covers every autotune config of its kernel at once.
-/

namespace VeriTile.Bench.TritonBenchG.MatmulDequantize

open VeriTile.Triton

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-! # ════════ Kernel 1 of 3: `matmul4_kernel` ════════ -/

/-! ## Kernel surface (faithful transcription) -/

def matmul_dequantize_matmul4_surface
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
    accumulator += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += ($(BLOCK_SIZE_K) // infearure_per_bits) * $(stride_bk)
  }
  c = (accumulator).to(tl.float16)
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
    ∃ alg, (matmul_dequantize_matmul4_surface a_ptr c_ptr scales_ptr b_ptr zeros_ptr
      M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize NO_GROUPS
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M).toAlgorithm?
        = Except.ok alg := by
  simp [matmul_dequantize_matmul4_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-! ## The group-swizzled block coordinates

`pid` is decomposed exactly as the source decomposes it, so the spec below is
stated in the same coordinates the kernel computes rather than in a re-derived
form. `matmul_kernel` (the file's second JIT, ported further below) swizzles
its `pid` with the **same** eleven-statement decomposition and addresses `C`
with the same `stride_cm * row + stride_cn * col` map, so `numPidM` … `pidN`,
`cAddr` and `cAddr_injective` here are shared by both matmul sections. -/

/-- `num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)`. -/
def numPidM (M BM : Nat) : Nat := (M + BM - 1) / BM

/-- `num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)`. -/
def numPidN (N BN : Nat) : Nat := (N + BN - 1) / BN

/-- `num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)` — the K-loop trip count in
`matmul4_kernel`. In `matmul_kernel` below the same binding is **dead** (that
loop's bound is `numKBlocks`), but it is still transcribed and walked. -/
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

/-- The prologue's **first** twelve statements: the two packing constants, the
program id, the three `tl.cdiv` trip counts and the group swizzle down to
`(pid_m, pid_n)`. All `nat` scalars. -/
def mdqPreLoopScalars (M N K BM BN BK GM : Nat) : List Stmt :=
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
        (Op.ref .nat [] "group_size_m")) ]

/-- The four statements the prologue's `if NO_GROUPS` branch runs: `scales` and the
packed `zeros` word are loaded **once**, from the base pointers — i.e. at group row
`0` — then unpacked and scaled. Under `NO_GROUPS` the K loop never reloads them, so
this is the only place they are ever set. -/
def mdqGroupPreload (BN : Nat) : List Stmt :=
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
        (Op.ref .real [BN] "scales")) ]

/-- The prologue's **ten index statements**: the offset vectors, the four pointer
tiles, the row mask and the two shifters. Split off from the scalars, and from the
pre-load that follows, so that no walk has to read a register through a twenty-deep
`setReg` tower — and so that the opaque state the pre-load returns is only ever
consumed by two statements. -/
def mdqPreLoopIndex (a_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : Region .nat)
    (M stride_am stride_ak stride_bk stride_bn
      stride_scales_n stride_zeros_n : Nat)
    (BM BN BK : Nat) : List Stmt :=
  [ Stmt.assign .nat [BM] "offs_am"
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
        (Op.ref .nat [] "bits")) ]

/-- The prologue's second half: the ten index statements, the `NO_GROUPS` pre-load
and the zeroed accumulator. -/
def mdqPreLoopTiles (a_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : Region .nat)
    (M stride_am stride_ak stride_bk stride_bn
      stride_scales_n stride_zeros_n : Nat) (NO_GROUPS : Bool)
    (BM BN BK : Nat) : List Stmt :=
  mdqPreLoopIndex a_ptr scales_ptr b_ptr zeros_ptr M stride_am stride_ak stride_bk
      stride_bn stride_scales_n stride_zeros_n BM BN BK
    ++ [ Stmt.ifThen (Op.constBool NO_GROUPS) (mdqGroupPreload BN),
         Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)) ]

/-- The compiled prologue: `mdqPreLoopScalars` then `mdqPreLoopTiles`. -/
def mdqPreLoop (a_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn
      stride_scales_n stride_zeros_n : Nat) (NO_GROUPS : Bool)
    (BM BN BK GM : Nat) : List Stmt :=
  mdqPreLoopScalars M N K BM BN BK GM
    ++ mdqPreLoopTiles a_ptr scales_ptr b_ptr zeros_ptr M stride_am stride_ak
        stride_bk stride_bn stride_scales_n stride_zeros_n NO_GROUPS BM BN BK

set_option maxRecDepth 8000 in
/-- The prologue is the first 24 statements of the lowered body, by `rfl`. -/
theorem mdq_preLoop_eq (a_ptr c_ptr scales_ptr : RegionName)
    (b_ptr zeros_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool) (BM BN BK GM : Nat) :
    ((matmul_dequantize_matmul4_surface a_ptr c_ptr scales_ptr b_ptr zeros_ptr
        M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
        groupsize NO_GROUPS BM BN BK GM).toAlgKernel.body.take 24)
      = mdqPreLoop a_ptr scales_ptr b_ptr zeros_ptr M N K stride_am stride_ak
          stride_bk stride_bn stride_scales_n stride_zeros_n NO_GROUPS BM BN BK GM := by
  rfl

/-- The seven statements the `if not NO_GROUPS` branch runs: the group row, then a
reload of `scales` and of the packed `zeros` word, the nibble extraction, and the
scaling `zeros = zeros * scales`. Named separately because the step proof has to
reason about the branch on its own — under `NO_GROUPS` it is skipped entirely and
the prologue's pre-load at row `0` is what the body uses. -/
def mdqGroupReload (groupsize stride_scales_g stride_zeros_g BN BK : Nat) :
    List Stmt :=
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
        (Op.ref .real [BN] "scales")) ]

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
      (mdqGroupReload groupsize stride_scales_g stride_zeros_g BN BK),
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

/-- The compiled tail: the dead `c = (accumulator).to(tl.float16)` binding —
here a **genuine `Op.castFloat` fp16 downcast** (the source spells the dtype
directly, so nothing erases), but of a register the store never reads — the
output coordinates, the `C` pointer tile, the two-axis store mask, and the
store — which writes `accumulator`, not `c`. -/
def mdqPostLoop (c_ptr : RegionName) (M N stride_cm stride_cn BM BN : Nat) :
    List Stmt :=
  [ Stmt.assign .fp16 [BM, BN] "c"
      (Op.castFloat FloatDType.real FloatDType.fp16
        (Op.ref .real [BM, BN] "accumulator")),
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
    (matmul_dequantize_matmul4_surface a_ptr c_ptr scales_ptr b_ptr zeros_ptr
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

/-! ## Per-statement eval recipes

Private copies, since bench ports never import each other. The `.ptr` family is
new for this port — `chunk_bwd_dqkg` used block pointers, whose loads carry a
boundary check; a `.ptr` load is unconditionally in bounds, so the mask is the
only gate and there is no `ok` branch to discharge. -/

/-- Unmasked `.ptr` load: every lane reads its own `(region, offset)`. -/
private theorem mdq_load_ptr_none {dtype : TileDType} {sh : TileShape}
    (nm : RegName) (t : BlockState) (pt : Tile .ptr sh)
    (hp : t.regs .ptr sh nm = some pt) :
    evalOp (Op.load dtype (MemAccess.ptr (Op.ref .ptr sh nm)) MaskOpt.none) t
      = some (⟨fun i => t.readMemValue dtype (pt.data i).1 (pt.data i).2⟩ :
          Tile dtype sh) := by
  simp only [evalOp, evalOp_ref, hp]
  rfl

/-- `.ptr` load with `mask=` / `other=`: masked-out lanes take the `other` tile. -/
private theorem mdq_load_ptr_maskOther {sh : TileShape}
    (nm : RegName) (t : BlockState) (pt : Tile .ptr sh)
    (mk : Op .bool sh) (ot : Op .real sh) (mv : Tile .bool sh) (ov : Tile .real sh)
    (hp : t.regs .ptr sh nm = some pt)
    (hm : evalOp mk t = some mv) (ho : evalOp ot t = some ov) :
    evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr sh nm))
        (MaskOpt.maskOther mk ot)) t
      = some (⟨fun i => if mv.data i then
            t.readMemValue .real (pt.data i).1 (pt.data i).2 else ov.data i⟩ :
          Tile .real sh) := by
  simp only [evalOp, evalOp_ref, hp, hm, ho]
  rfl

/-- Pointer advance / offset. -/
private theorem mdq_ptrAdd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (pnm : RegName) (t : BlockState) (pt : Tile .ptr a) (off : Op .nat b)
    (ov : Tile .nat b)
    (hp : t.regs .ptr a pnm = some pt) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ref .ptr a pnm) off) t = some (Tile.ptrAdd bc pt ov) := by
  simp only [evalOp, evalOp_ref, hp, ho]
  rfl

/-- `&` on the `nat` channel — there is no `evalOp_bitAnd` in the library. -/
private theorem mdq_bitAnd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.bitAnd bc x y) t = some (Tile.bop (· &&& ·) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `>>` on the `nat` channel. -/
private theorem mdq_shiftRight_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.shiftRight bc x y) t = some (Tile.bop (· >>> ·) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `Op.remap` — how the DSL rank-broadcasts the `[BM, 1]` load mask. -/
private theorem mdq_remap_eval {dtype : TileDType} {inS outS : TileShape}
    (map : TileIndex outS → TileIndex inS) (x : Op dtype inS) (t : BlockState)
    (vx : Tile dtype inS) (hx : evalOp x t = some vx) :
    evalOp (Op.remap outS map x) t = some (Tile.remap map vx) := by
  simp only [evalOp, hx]
  rfl

/-- `tl.broadcast_to`-style scalar fill, as the load's `other=0.0` produces. -/
private theorem mdq_broadcast_eval {dtype : TileDType} (e : Op dtype [])
    (sh : TileShape) (t : BlockState) (v : Tile dtype [])
    (hv : evalOp e t = some v) :
    evalOp (Op.broadcast e sh) t
      = some (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype sh) := by
  simp only [evalOp, hv]
  rfl

/-! ## The scales / zeros channel

Both are `[BN]` tiles whose base pointer is set up once in the prologue and then
offset by `g_id * stride_*_g` inside the loop (or not offset at all, under
`NO_GROUPS`). Naming the two base address functions keeps the group row as the
only moving part. -/

/-- `scales_ptrs` lane `c`. -/
def mdqScalesAddr (stride_scales_n BN pn : Nat) (idx : TileIndex [BN]) : Nat :=
  (pn * BN + idx.1.val) * stride_scales_n

/-- `zeros_ptrs` lane `c` — eight zero-points share a word along N. -/
def mdqZerosAddr (stride_zeros_n BN pn : Nat) (idx : TileIndex [BN]) : Nat :=
  (pn * BN + idx.1.val) / 8 * stride_zeros_n

noncomputable def mdqScalesPtrs (scales : RegionName) (stride_scales_n BN pn : Nat) :
    Tile .ptr [BN] :=
  ⟨fun idx => (scales, mdqScalesAddr stride_scales_n BN pn idx)⟩

noncomputable def mdqZerosPtrs (zeros : Region .nat) (stride_zeros_n BN pn : Nat) :
    Tile .ptr [BN] :=
  ⟨fun idx => (Region.cast zeros, mdqZerosAddr stride_zeros_n BN pn idx)⟩

/-- The packed zero-point words as the load delivers them, before the nibble is
extracted — eight zero-points per word, so lanes `8c … 8c+7` share one. -/
noncomputable def mdqZerosRaw (s : BlockState) (zeros : Region .nat)
    (stride_zeros_g stride_zeros_n BN pn g : Nat) : Tile .nat [BN] :=
  ⟨fun idx => zerosWord s zeros stride_zeros_g stride_zeros_n BN pn g idx.1.val⟩

/-- `scales` as either branch leaves it, at group row `g`. -/
noncomputable def mdqScalesTile (s : BlockState) (scales : RegionName)
    (stride_scales_g stride_scales_n BN pn g : Nat) : Tile .real [BN] :=
  ⟨fun idx => some (scalesElem s scales stride_scales_g stride_scales_n BN pn g
      idx.1.val)⟩

/-- `zeros` as either branch leaves it — already **scaled**, since both branches
end with `zeros = zeros * scales`. -/
noncomputable def mdqZerosTile (s : BlockState) (scales : RegionName)
    (zeros : Region .nat)
    (stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n BN pn g : Nat) :
    Tile .real [BN] :=
  ⟨fun idx => some (zeroScaled s scales zeros stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BN pn g idx.1.val)⟩

/-- Offsetting `scales_ptrs` by `g * stride_scales_g` lands on `scalesElem`'s
address at group row `g`. -/
theorem mdqScalesPtrs_offset (scales : RegionName)
    (stride_scales_g stride_scales_n BN pn g : Nat) (idx : TileIndex [BN]) :
    (Tile.ptrAdd Broadcast.scalarR (mdqScalesPtrs scales stride_scales_n BN pn)
        (Tile.scalar (g * stride_scales_g))).data idx
      = (scales, (pn * BN + idx.1.val) * stride_scales_n + g * stride_scales_g) := by
  simp only [Tile.ptrAdd_data, mdqScalesPtrs, mdqScalesAddr, Tile.scalar,
    Broadcast.leftIndex]

/-- Likewise for `zeros_ptrs` and `zerosWord`'s address. -/
theorem mdqZerosPtrs_offset (zeros : Region .nat)
    (stride_zeros_g stride_zeros_n BN pn g : Nat) (idx : TileIndex [BN]) :
    (Tile.ptrAdd Broadcast.scalarR (mdqZerosPtrs zeros stride_zeros_n BN pn)
        (Tile.scalar (g * stride_zeros_g))).data idx
      = (Region.cast zeros,
          (pn * BN + idx.1.val) / 8 * stride_zeros_n + g * stride_zeros_g) := by
  simp only [Tile.ptrAdd_data, mdqZerosPtrs, mdqZerosAddr, Tile.scalar,
    Broadcast.leftIndex]


/-! ## The K-loop invariant

`i ≤ numPidK` is part of the predicate on purpose: `forRangeDyn_inv` concludes
only `stop ≤ final`, so carrying the upper bound is what pins `final = stop` and
lets the readout use `mdqAccTile_full`.

The `NO_GROUPS` case is the one that makes this invariant unlike a plain
carry-fold: under that flag `scales`/`zeros` are loop-invariant (loaded once,
before the loop, at group row `0`), while otherwise they are re-established by the
body at row `groupRow`. Both are covered by asking only that the registers hold
the *tiles for the row `groupRow NO_GROUPS groupsize BK i`* — which under
`NO_GROUPS` is `0` for every `i`. -/

/-- The state carried across K steps. -/
noncomputable def mdqInv (s0 : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M N K groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK GM : Nat) (i : Nat) (s : BlockState) : Prop :=
  i ≤ numPidK K BK
  ∧ s.mem = s0.mem
  ∧ s.pids = s0.pids
  ∧ s.regs .nat [] "bits" = some (Tile.scalar 4)
  ∧ s.regs .nat [] "infearure_per_bits" = some (Tile.scalar 8)
  ∧ s.regs .nat [] "pid_m" = some (Tile.scalar (pidM s0 M N BM BN GM))
  ∧ s.regs .nat [] "pid_n" = some (Tile.scalar (pidN s0 M N BM BN GM))
  ∧ s.regs .bool [BM, 1] "a_mask"
      = some (mdqAMask M BM (pidM s0 M N BM BN GM))
  ∧ s.regs .nat [BK] "shifter" = some (mdqShifter BK)
  ∧ s.regs .nat [BN] "zeros_shifter"
      = some (mdqZerosShifter BN (pidN s0 M N BM BN GM))
  ∧ s.regs .ptr [BN] "scales_ptrs"
      = some (mdqScalesPtrs scales stride_scales_n BN (pidN s0 M N BM BN GM))
  ∧ s.regs .ptr [BN] "zeros_ptrs"
      = some (mdqZerosPtrs zeros stride_zeros_n BN (pidN s0 M N BM BN GM))
  ∧ s.regs .ptr [BM, BK] "a_ptrs"
      = some (mdqAPtrs a stride_am stride_ak BM BK (pidM s0 M N BM BN GM) i)
  ∧ s.regs .ptr [BK, BN] "b_ptrs"
      = some (mdqBPtrs b stride_bk stride_bn BN BK (pidN s0 M N BM BN GM) i)
  ∧ s.regs .real [BM, BN] "accumulator"
      = some (mdqAccTile s0 a scales b zeros NO_GROUPS M groupsize stride_am
          stride_ak stride_bk stride_bn stride_scales_g stride_scales_n
          stride_zeros_g stride_zeros_n BM BN BK (pidM s0 M N BM BN GM)
          (pidN s0 M N BM BN GM) i)
  ∧ (NO_GROUPS = Bool.true →
      s.regs .real [BN] "scales"
        = some (mdqScalesTile s0 scales stride_scales_g stride_scales_n BN
            (pidN s0 M N BM BN GM) 0)
      ∧ s.regs .real [BN] "zeros"
        = some (mdqZerosTile s0 scales zeros stride_scales_g stride_scales_n
            stride_zeros_g stride_zeros_n BN (pidN s0 M N BM BN GM) 0))

/-- Under `NO_GROUPS` the group row is `0` at every step, so the loop-invariant
pre-load is exactly what every iteration needs. -/
theorem groupRow_of_noGroups (groupsize BK i : Nat) :
    groupRow Bool.true groupsize BK i = 0 := by
  simp [groupRow]

/-- `mdqInv` pins memory as a function, so every read agrees with the launch
state — including the `.nat` reads the packed channels use. -/
theorem mdqInv_readMemValue (s0 : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M N K groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK GM i : Nat) (s : BlockState)
    (h : mdqInv s0 a scales b zeros NO_GROUPS M N K groupsize stride_am stride_ak
      stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
      stride_zeros_n BM BN BK GM i s)
    (dtype : TileDType) (rg : RegionName) (off : Nat) :
    s.readMemValue dtype rg off = s0.readMemValue dtype rg off := by
  obtain ⟨_, hmem, _⟩ := h
  simp only [BlockState.readMemValue, BlockState.readMemAs, BlockState.readMemTyped,
    hmem]

/-! ### The two K-step loads, bridged to the named tiles -/

/-- The `a` load — pointer tile, rank-broadcast mask, `other=0.0` — lands exactly
on `mdqATile`, and on the *launch* state's memory. -/
private theorem mdq_aLoad_eq (s0 : BlockState) (a : RegionName) (t : BlockState)
    (M stride_am stride_ak BM BK pm i : Nat)
    (hmem : t.mem = s0.mem)
    (hap : t.regs .ptr [BM, BK] "a_ptrs"
      = some (mdqAPtrs a stride_am stride_ak BM BK pm i))
    (hmask : t.regs .bool [BM, 1] "a_mask" = some (mdqAMask M BM pm)) :
    evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        (MaskOpt.maskOther
          (Op.remap [BM, BK]
            (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
            (Op.ref .bool [BM, 1] "a_mask"))
          (Op.broadcast (Op.const 0.0) [BM, BK]))) t
      = some (mdqATile s0 a M stride_am stride_ak BM BK pm i) := by
  rw [mdq_load_ptr_maskOther "a_ptrs" t
    (mdqAPtrs a stride_am stride_ak BM BK pm i) _ _
    (Tile.remap (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
      (mdqAMask M BM pm))
    (⟨fun _ => some (0.0 : ℝ)⟩ : Tile .real [BM, BK])
    hap
    (mdq_remap_eval _ _ t _ (by rw [evalOp_ref]; exact hmask))
    (mdq_broadcast_eval _ _ t _ (evalOp_const 0.0 t))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [mdqATile, mdqAPtrs, Tile.remap, mdqAMask, Broadcast.leftIndex,
    BlockState.readMemValue_real, BlockState.readMem, hmem, decide_eq_true_eq]
  split
  · rfl
  · norm_num

/-- The packed-weight load lands on `mdqBRaw`, on the launch state's memory. -/
private theorem mdq_bLoad_eq (s0 : BlockState) (b : Region .nat) (t : BlockState)
    (stride_bk stride_bn BN BK pn i : Nat)
    (hmem : t.mem = s0.mem)
    (hbp : t.regs .ptr [BK, BN] "b_ptrs"
      = some (mdqBPtrs b stride_bk stride_bn BN BK pn i)) :
    evalOp (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BK, BN] "b_ptrs"))
        MaskOpt.none) t
      = some (mdqBRaw s0 b stride_bk stride_bn BN BK pn i) := by
  rw [mdq_load_ptr_none "b_ptrs" t (mdqBPtrs b stride_bk stride_bn BN BK pn i) hbp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [mdqBRaw, mdqBPtrs, BlockState.readMemValue, BlockState.readMemTyped,
    hmem]

/-- A `[BN]` `.real` load through an offset `scales_ptrs` lands on `mdqScalesTile`
at that group row. -/
private theorem mdq_scalesLoad_eq (s0 : BlockState) (scales : RegionName)
    (t : BlockState) (stride_scales_g stride_scales_n BN pn g : Nat)
    (hmem : t.mem = s0.mem)
    (hp : t.regs .ptr [BN] "ptr"
      = some (Tile.ptrAdd Broadcast.scalarR
          (mdqScalesPtrs scales stride_scales_n BN pn)
          (Tile.scalar (g * stride_scales_g)))) :
    evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN] "ptr")) MaskOpt.none) t
      = some (mdqScalesTile s0 scales stride_scales_g stride_scales_n BN pn g) := by
  rw [mdq_load_ptr_none "ptr" t _ hp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [mdqScalesTile, scalesElem, mdqScalesPtrs_offset,
    BlockState.readMemValue_real, BlockState.readMem, hmem]

/-- The packed zero-point load through an offset `zeros_ptrs` lands on
`zerosWord` at that group row. -/
private theorem mdq_zerosLoad_eq (s0 : BlockState) (zeros : Region .nat)
    (t : BlockState) (stride_zeros_g stride_zeros_n BN pn g : Nat)
    (hmem : t.mem = s0.mem)
    (hp : t.regs .ptr [BN] "ptr"
      = some (Tile.ptrAdd Broadcast.scalarR
          (mdqZerosPtrs zeros stride_zeros_n BN pn)
          (Tile.scalar (g * stride_zeros_g)))) :
    evalOp (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BN] "ptr")) MaskOpt.none) t
      = some (mdqZerosRaw s0 zeros stride_zeros_g stride_zeros_n BN pn g) := by
  rw [mdq_load_ptr_none "ptr" t _ hp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [mdqZerosRaw, zerosWord, mdqZerosPtrs_offset, BlockState.readMemValue,
    BlockState.readMemTyped, hmem]

/-- `//` on `nat` scalars — there is no `evalOp_floorDiv` in the library. -/
private theorem mdq_floorDiv_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.floorDiv IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.floorDiv IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `name * c` on a `nat` scalar register. -/
private theorem mdq_mulRef_eval (t : BlockState) (nm : RegName) (val c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c)) t
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `Op.natToReal`. -/
private theorem mdq_natToReal_eval {sh : TileShape} (x : Op .nat sh)
    (t : BlockState) (vx : Tile .nat sh) (hx : evalOp x t = some vx) :
    evalOp (Op.natToReal x) t = some (Tile.natToReal vx) := by
  simp only [evalOp, hx]
  rfl

/-- The `g_id` value the branch computes at K step `i`, which is exactly
`groupRow` on the non-`NO_GROUPS` side. -/
theorem groupRow_of_groups (groupsize BK i : Nat) :
    groupRow Bool.false groupsize BK i = i / (groupsize / BK) := by
  simp [groupRow]

/-- **The branch's last two statements.** Extract the zero-point nibble, then
scale it — i.e. `mdqZerosTile` is exactly that composition over the loaded word. -/
private theorem mdqZerosTile_eq (s0 : BlockState) (scales : RegionName)
    (zeros : Region .nat)
    (stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n BN pn g : Nat) :
    mdqZerosTile s0 scales zeros stride_scales_g stride_scales_n
        stride_zeros_g stride_zeros_n BN pn g
      = Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
          (Tile.natToReal
            (Tile.bop (· &&& ·) Broadcast.scalarR
              (Tile.bop (· >>> ·) (Broadcast.consSame Broadcast.nil)
                (mdqZerosRaw s0 zeros stride_zeros_g stride_zeros_n BN pn g)
                (mdqZerosShifter BN pn))
              (Tile.scalar 15)))
          (mdqScalesTile s0 scales stride_scales_g stride_scales_n BN pn g) := by
  apply Tile.ext
  intro idx
  simp only [mdqZerosTile, zeroScaled, zerosNibble, mdqScalesTile, mdqZerosRaw,
    mdqZerosShifter, Tile.bop_data, Tile.natToReal, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul, WithBot.realMul]
  rfl

/-- `//` on two `nat` **scalars**, in the shape the group row is computed in. -/
private theorem mdq_floorDivScalar_eval (x y : Op .nat []) (t : BlockState)
    (u v : Nat) (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u / v)) := by
  rw [mdq_floorDiv_eval Broadcast.nil x y t _ _ hx hy]
  rfl

/-- `*` on two tiles, given both operand values — the library's `evalOp_mul` is
stated as a `do` block, which is not a `rw` target once the operands are known. -/
private theorem mdq_mulTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mul h bc x y) t = some (Tile.bop h.mul bc vx vy) := by
  rw [evalOp_mul, hx, hy]
  rfl

/-- `Stmt.ifThen`'s step equation — `stepStmt` is defined by well-founded
recursion, so the definitional unfolding has to be named. -/
private theorem mdq_ifThen_step (cond : Op .bool []) (body : List Stmt)
    (t : BlockState) :
    stepStmt (Stmt.ifThen cond body) t
      = (evalOp cond t).bind
          (fun c => if c.data PUnit.unit then stepStmts body t else some t) := by
  unfold stepStmt
  cases evalOp cond t <;> rfl

/-- The branch guard `not NO_GROUPS` is a closed scalar. -/
private theorem mdq_notFlag_eval (flag : Bool) (t : BlockState) :
    evalOp (Op.boolNot (Op.constBool flag)) t = some (Tile.scalar (!flag)) := by
  simp only [evalOp]
  rfl

/-! ### The `NO_GROUPS` branch

The one place where the two configurations of this kernel genuinely diverge, and
the crux of the step proof. Both paths are made to land on the *same* statement —
`scales` and `zeros` hold the tiles for group row `groupRow NO_GROUPS groupsize BK
i` — so everything downstream is written once:

* `NO_GROUPS = true`: the branch is skipped, the state is untouched, and the
  registers still hold the prologue's row-`0` pre-load, which `groupRow_of_noGroups`
  identifies with the row asked for.
* `NO_GROUPS = false`: the seven statements re-establish both registers at row
  `i / (groupsize / BK)`, which `groupRow_of_groups` identifies with the same.

The register-preservation clause is stated as a side condition on *names* rather
than as a list of the individual registers the rest of the body needs — the branch
writes only `g_id`, `ptr`, `scales`, `zeros`, and nothing else has to be enumerated. -/

private theorem mdq_groupBranch_run (s0 : BlockState) (scales : RegionName)
    (zeros : Region .nat) (t : BlockState) (NO_GROUPS : Bool)
    (groupsize stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BN BK pn i : Nat)
    (hmem : t.mem = s0.mem)
    (hk : t.regs .nat [] "k" = some (Tile.scalar i))
    (hsp : t.regs .ptr [BN] "scales_ptrs"
      = some (mdqScalesPtrs scales stride_scales_n BN pn))
    (hzp : t.regs .ptr [BN] "zeros_ptrs"
      = some (mdqZerosPtrs zeros stride_zeros_n BN pn))
    (hzs : t.regs .nat [BN] "zeros_shifter" = some (mdqZerosShifter BN pn))
    (hpre : NO_GROUPS = Bool.true →
      t.regs .real [BN] "scales"
          = some (mdqScalesTile s0 scales stride_scales_g stride_scales_n BN pn 0)
        ∧ t.regs .real [BN] "zeros"
          = some (mdqZerosTile s0 scales zeros stride_scales_g stride_scales_n
              stride_zeros_g stride_zeros_n BN pn 0)) :
    ∃ t', stepStmt (Stmt.ifThen (Op.boolNot (Op.constBool NO_GROUPS))
          (mdqGroupReload groupsize stride_scales_g stride_zeros_g BN BK)) t
        = some t'
      ∧ t'.mem = t.mem
      ∧ t'.pids = t.pids
      ∧ (∀ (dtype : TileDType) (sh : TileShape) (nm : RegName),
          nm ≠ "g_id" → nm ≠ "ptr" → nm ≠ "scales" → nm ≠ "zeros" →
          t'.regs dtype sh nm = t.regs dtype sh nm)
      ∧ t'.regs .real [BN] "scales"
          = some (mdqScalesTile s0 scales stride_scales_g stride_scales_n BN pn
              (groupRow NO_GROUPS groupsize BK i))
      ∧ t'.regs .real [BN] "zeros"
          = some (mdqZerosTile s0 scales zeros stride_scales_g stride_scales_n
              stride_zeros_g stride_zeros_n BN pn
              (groupRow NO_GROUPS groupsize BK i)) := by
  -- The DSL claims `true` / `false` as expression tokens, so the two
  -- configurations are separated by `by_cases` on the flag rather than by named
  -- `cases` alternatives.
  by_cases hng : NO_GROUPS = Bool.true
  · -- `NO_GROUPS`: the guard is `false`, the state is untouched, and the
    -- prologue's row-`0` pre-load is exactly what is being asked for.
    subst hng
    obtain ⟨hs, hz⟩ := hpre rfl
    refine ⟨t, ?_, rfl, rfl, fun _ _ _ _ _ _ _ => rfl, ?_, ?_⟩
    · rw [mdq_ifThen_step, mdq_notFlag_eval]
      rfl
    · rw [groupRow_of_noGroups]; exact hs
    · rw [groupRow_of_noGroups]; exact hz
  · -- Otherwise the guard is `true`: walk the seven statements.
    have hflag : NO_GROUPS = Bool.false := by simpa using hng
    subst hflag
    have hstep : stepStmt (Stmt.ifThen (Op.boolNot (Op.constBool Bool.false))
          (mdqGroupReload groupsize stride_scales_g stride_zeros_g BN BK)) t
        = stepStmts (mdqGroupReload groupsize stride_scales_g stride_zeros_g BN BK)
            t := by
      rw [mdq_ifThen_step, mdq_notFlag_eval]
      rfl
    rw [hstep, groupRow_of_groups]
    unfold mdqGroupReload
    -- 1. `g_id = k // (groupsize // BLOCK_SIZE_K)`
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (mdq_floorDivScalar_eval _ _ t i (groupsize / BK)
        (by rw [evalOp_ref]; exact hk)
        (mdq_floorDivScalar_eval _ _ t groupsize BK (evalOp_constNat _ _)
          (evalOp_constNat _ _))))]
    -- 2. `ptr = scales_ptrs + g_id * stride_scales_g`
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (mdq_ptrAdd_eval Broadcast.scalarR "scales_ptrs" _
        (mdqScalesPtrs scales stride_scales_n BN pn) _
        (Tile.scalar (i / (groupsize / BK) * stride_scales_g))
        (by simpa using hsp)
        (mdq_mulRef_eval _ "g_id" (i / (groupsize / BK)) stride_scales_g (by simp))))]
    -- 3. `scales = tl.load(ptr)`
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (mdq_scalesLoad_eq s0 scales _ stride_scales_g stride_scales_n BN pn
        (i / (groupsize / BK)) (by simpa using hmem) (by simp)))]
    -- 4. `ptr = zeros_ptrs + g_id * stride_zeros_g`
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (mdq_ptrAdd_eval Broadcast.scalarR "zeros_ptrs" _
        (mdqZerosPtrs zeros stride_zeros_n BN pn) _
        (Tile.scalar (i / (groupsize / BK) * stride_zeros_g))
        (by simpa using hzp)
        (mdq_mulRef_eval _ "g_id" (i / (groupsize / BK)) stride_zeros_g (by simp))))]
    -- 5. `zeros = tl.load(ptr)` — the packed word
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (mdq_zerosLoad_eq s0 zeros _ stride_zeros_g stride_zeros_n BN pn
        (i / (groupsize / BK)) (by simpa using hmem) (by simp)))]
    -- 6. `zeros = (zeros >> zeros_shifter) & 0xF`.  The intermediate tiles are
    -- given explicitly: `rw` cannot leave them as metavariables here, since the
    -- register value is what the next statement reads back.
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (mdq_bitAnd_eval Broadcast.scalarR _ _ _
        (Tile.bop (· >>> ·) (Broadcast.consSame Broadcast.nil)
          (mdqZerosRaw s0 zeros stride_zeros_g stride_zeros_n BN pn
            (i / (groupsize / BK)))
          (mdqZerosShifter BN pn))
        (Tile.scalar 15)
        (mdq_shiftRight_eval (Broadcast.consSame Broadcast.nil) _ _ _
          (mdqZerosRaw s0 zeros stride_zeros_g stride_zeros_n BN pn
            (i / (groupsize / BK)))
          (mdqZerosShifter BN pn) (by rw [evalOp_ref]; simp)
          (by rw [evalOp_ref]; simpa using hzs))
        (evalOp_constNat _ _)))]
    -- 7. `zeros = zeros.to(tl.float16) * scales`
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (mdq_mulTile_eval NumericDType.real (Broadcast.consSame Broadcast.nil) _ _ _
        (Tile.natToReal
          (Tile.bop (· &&& ·) Broadcast.scalarR
            (Tile.bop (· >>> ·) (Broadcast.consSame Broadcast.nil)
              (mdqZerosRaw s0 zeros stride_zeros_g stride_zeros_n BN pn
                (i / (groupsize / BK)))
              (mdqZerosShifter BN pn))
            (Tile.scalar 15)))
        (mdqScalesTile s0 scales stride_scales_g stride_scales_n BN pn
          (i / (groupsize / BK)))
        (mdq_natToReal_eval _ _ _ (by rw [evalOp_ref]; simp))
        (by rw [evalOp_ref]; simp)))]
    rw [stepStmts.nil]
    refine ⟨_, rfl, rfl, rfl, ?_, ?_, ?_⟩
    · intro dtype sh nm h1 h2 h3 h4
      simp [h1, h2, h3, h4]
    · simp
    · rw [mdqZerosTile_eq]
      simp

/-! ### The dequantized weight tile and the accumulator step

The last three compute statements of the body. `b` is rewritten twice — once to
extract the nibble on the `nat` channel, once to scale and shift it onto the `real`
channel — and only then does `tl.dot` fire, so the dequantization is entirely
inside the K step and `bDequant` is the right unit to name. -/

/-- `b` after both dequant statements at K step `k`: the unpacked nibble scaled by
`scales`, minus the already-scaled `zeros`, both read at that step's group row. -/
noncomputable def mdqBDequantTile (s : BlockState) (scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (groupsize stride_bk stride_bn stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BN BK pn k : Nat) : Tile .real [BK, BN] :=
  ⟨fun idx => some (bDequant s scales b zeros NO_GROUPS groupsize stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n BN BK pn
      k idx.1.val idx.2.1.val)⟩

/-- The dequantized tile **is** the compiled `b * scales[None, :] - zeros[None, :]`,
with the group row supplied by `groupRow`. Both `[BN]` operands are rank-broadcast
along the K axis, which is why one group row serves the whole tile. -/
theorem mdqBDequantTile_eq (s : BlockState) (scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (groupsize stride_bk stride_bn stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BN BK pn k : Nat) :
    mdqBDequantTile s scales b zeros NO_GROUPS groupsize stride_bk stride_bn
        stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n BN BK pn k
      = Tile.bop NumericDType.real.sub
          (Broadcast.consR (Broadcast.consSame Broadcast.nil))
          (Tile.bop NumericDType.real.mul
            (Broadcast.consR (Broadcast.consSame Broadcast.nil))
            (Tile.natToReal (mdqBNibbleTile s b stride_bk stride_bn BN BK pn k))
            (Tile.expandDim ⟨0, by simp⟩
              (mdqScalesTile s scales stride_scales_g stride_scales_n BN pn
                (groupRow NO_GROUPS groupsize BK k))))
          (Tile.expandDim ⟨0, by simp⟩
            (mdqZerosTile s scales zeros stride_scales_g stride_scales_n
              stride_zeros_g stride_zeros_n BN pn
              (groupRow NO_GROUPS groupsize BK k))) := by
  apply Tile.ext
  intro idx
  simp only [mdqBDequantTile, bDequant, mdqBNibbleTile, mdqScalesTile, mdqZerosTile,
    Tile.bop_data, Tile.natToReal, Tile.expandDim_data, TileShape.dropInsertedIndex,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub, NumericDType.mul,
    WithBot.realSub, WithBot.realMul]
  rfl

/-- A `WithBot ℝ` sum of pointwise products of `some`s is the `some` of the `ℝ`
sum — the collapse `Tile.dot` needs, since every operand lane here is a loaded
(non-`⊥`) value. -/
private theorem mdq_coe_sum_mul {n : Nat} (f g : Fin n → ℝ) :
    (@Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e =>
        Option.map₂ (fun x y : ℝ => x * y) (some (f e)) (some (g e)))
      = some (∑ e : Fin n, f e * g e) := by
  rw [show (@Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e =>
        Option.map₂ (fun x y : ℝ => x * y) (some (f e)) (some (g e)))
      = (@Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e =>
          (some (f e * g e) : WithBot ℝ)) from Finset.sum_congr rfl fun e _ => by simp]
  show (Finset.univ.sum fun e => ((f e * g e : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]
  rfl

/-- **The accumulator statement.** `accumulator += tl.dot(a, b)` extends the partial
sum by exactly one `accStep`. -/
theorem mdqAccTile_dot_succ (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn i : Nat) :
    Tile.bop NumericDType.real.add
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (mdqAccTile s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
          stride_bk stride_bn stride_scales_g stride_scales_n
          stride_zeros_g stride_zeros_n BM BN BK pm pn i)
        (Tile.dot [] (mdqATile s a M stride_am stride_ak BM BK pm i)
          (mdqBDequantTile s scales b zeros NO_GROUPS groupsize stride_bk stride_bn
            stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n BN BK pn i))
      = mdqAccTile s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
          stride_bk stride_bn stride_scales_g stride_scales_n
          stride_zeros_g stride_zeros_n BM BN BK pm pn (i + 1) := by
  apply Tile.ext
  intro idx
  obtain ⟨r, c, u⟩ := idx
  rw [mdqAccTile_succ]
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, mdqAccTile,
    NumericDType.add, WithBot.realAdd]
  -- `erw`: `Tile.dot`'s operand shapes are `[] ++ [M, K]`, so `Tile.dot_nil_data`
  -- does not fire under `rw` / `simp only`.
  erw [Tile.dot_nil_data]
  simp only [mdqATile_data, mdqBDequantTile]
  rw [mdq_coe_sum_mul]
  simp [accStep]

/-! ### The remaining eval recipes -/

/-- `-` on two tiles, both operand values known. -/
private theorem mdq_subTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.sub h bc x y) t = some (Tile.bop h.sub bc vx vy) := by
  rw [evalOp_sub, hx, hy]
  rfl

/-- `+` on two tiles, both operand values known. -/
private theorem mdq_addTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.add h bc x y) t = some (Tile.bop h.add bc vx vy) := by
  rw [evalOp_add, hx, hy]
  rfl

/-- `*` on two `nat` **scalars** — the shape both pointer advances use. -/
private theorem mdq_mulScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mul .nat Broadcast.nil x y) t = some (Tile.scalar (u * v)) := by
  rw [evalOp_mul, hx, hy]
  rfl

/-- `[:, None]` / `[None, :]`.  The axis binder is spelled `ax`: the DSL claims
`axis` as a keyword-argument token, so it cannot be used as an identifier here. -/
private theorem mdq_expandDim_eval {dtype : TileDType} {sh : TileShape}
    (ax : Fin (sh.length + 1)) (x : Op dtype sh) (t : BlockState)
    (v : Tile dtype sh) (hv : evalOp x t = some v) :
    evalOp (Op.expandDim ax x) t = some (Tile.expandDim ax v) := by
  rw [evalOp_expandDim, hv]
  rfl

/-- `tl.dot` at rank 2. `erw`, not `rw`: the operand shapes are `[] ++ [M, K]`,
which does not unfold at reducible transparency, so `evalOp_dot` silently fails to
fire under `rw` / `simp only`. -/
private theorem mdq_dot_eval {M K N : Nat} (x : Op .real [M, K]) (y : Op .real [K, N])
    (t : BlockState) (vx : Tile .real [M, K]) (vy : Tile .real [K, N])
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.dot (batch := []) x y) t = some (Tile.dot [] vx vy) := by
  erw [evalOp_dot, hx, hy]
  rfl

/-! ### The K step

One iteration takes `mdqInv` at `i` to `mdqInv` at `i + 1`. The eight statements are
walked in order; the only branching is the `NO_GROUPS` reload, and
`mdq_groupBranch_run` has already collapsed both of its paths to one conclusion, so
statements 4–8 are walked once. -/

theorem mdqLoopBody_run (s0 : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M N K groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK GM i : Nat) (s : BlockState)
    (hnext : i + 1 ≤ numPidK K BK)
    (hk : s.regs .nat [] "k" = some (Tile.scalar i))
    (hinv : mdqInv s0 a scales b zeros NO_GROUPS M N K groupsize stride_am stride_ak
      stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
      stride_zeros_n BM BN BK GM i s) :
    ∃ s', stepStmts (mdqLoopBody groupsize stride_ak stride_bk stride_scales_g
          stride_zeros_g NO_GROUPS BM BN BK) s = some s'
      ∧ mdqInv s0 a scales b zeros NO_GROUPS M N K groupsize stride_am stride_ak
          stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
          stride_zeros_n BM BN BK GM (i + 1) s' := by
  obtain ⟨-, hmem, hpids, hbits, hipb, hpm, hpn, hmask, hshift, hzshift,
    hspt, hzpt, hapt, hbpt, hacc, hpreload⟩ := hinv
  set pm := pidM s0 M N BM BN GM with hpmDef
  set pn := pidN s0 M N BM BN GM with hpnDef
  unfold mdqLoopBody
  -- 1. `a = tl.load(a_ptrs, mask=a_mask[:, None], other=0.0)`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_aLoad_eq s0 a s M stride_am stride_ak BM BK pm i hmem hapt hmask))]
  -- 2. `b = tl.load(b_ptrs)` — the packed weight words
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_bLoad_eq s0 b _ stride_bk stride_bn BN BK pn i (by simpa using hmem)
      (by simpa using hbpt)))]
  -- 3. the `not NO_GROUPS` reload, both paths already on one conclusion
  obtain ⟨t, hbranch, htmem, htpids, hkeep, hscalesT, hzerosT⟩ :=
    mdq_groupBranch_run s0 scales zeros
      ((s.setReg "a" .real [BM, BK] (mdqATile s0 a M stride_am stride_ak BM BK pm i)).setReg
        "b" .nat [BK, BN] (mdqBRaw s0 b stride_bk stride_bn BN BK pn i))
      NO_GROUPS groupsize stride_scales_g
      stride_scales_n stride_zeros_g stride_zeros_n BN BK pn i
      (by simpa using hmem) (by simpa using hk) (by simpa using hspt)
      (by simpa using hzpt) (by simpa using hzshift)
      (fun hf => by simpa using hpreload hf)
  rw [stepStmts.cons_some hbranch]
  -- What the branch left untouched, transported all the way back to `s`.
  have hkeepS : ∀ (dtype : TileDType) (sh : TileShape) (nm : RegName),
      nm ≠ "g_id" → nm ≠ "ptr" → nm ≠ "scales" → nm ≠ "zeros" →
      nm ≠ "a" → nm ≠ "b" → t.regs dtype sh nm = s.regs dtype sh nm := by
    intro dtype sh nm h1 h2 h3 h4 h5 h6
    rw [hkeep dtype sh nm h1 h2 h3 h4]
    simp [h5, h6]
  have htATile : t.regs .real [BM, BK] "a"
      = some (mdqATile s0 a M stride_am stride_ak BM BK pm i) := by
    rw [hkeep .real [BM, BK] "a" (by decide) (by decide) (by decide) (by decide)]
    simp
  have htBRaw : t.regs .nat [BK, BN] "b"
      = some (mdqBRaw s0 b stride_bk stride_bn BN BK pn i) := by
    rw [hkeep .nat [BK, BN] "b" (by decide) (by decide) (by decide) (by decide)]
    simp
  have htShift : t.regs .nat [BK] "shifter" = some (mdqShifter BK) := by
    rw [hkeepS .nat [BK] "shifter" (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hshift
  have htIpb : t.regs .nat [] "infearure_per_bits" = some (Tile.scalar 8) := by
    rw [hkeepS .nat [] "infearure_per_bits" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]
    exact hipb
  have htAcc : t.regs .real [BM, BN] "accumulator"
      = some (mdqAccTile s0 a scales b zeros NO_GROUPS M groupsize stride_am
          stride_ak stride_bk stride_bn stride_scales_g stride_scales_n
          stride_zeros_g stride_zeros_n BM BN BK pm pn i) := by
    rw [hkeepS .real [BM, BN] "accumulator" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]
    exact hacc
  have htAPtrs : t.regs .ptr [BM, BK] "a_ptrs"
      = some (mdqAPtrs a stride_am stride_ak BM BK pm i) := by
    rw [hkeepS .ptr [BM, BK] "a_ptrs" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]
    exact hapt
  have htBPtrs : t.regs .ptr [BK, BN] "b_ptrs"
      = some (mdqBPtrs b stride_bk stride_bn BN BK pn i) := by
    rw [hkeepS .ptr [BK, BN] "b_ptrs" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]
    exact hbpt
  -- 4. `b = (b >> shifter[:, None]) & 0xF`
  have h4 : evalOp (Op.bitAnd Broadcast.scalarR
        (Op.shiftRight (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .nat [BK, BN] "b")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "shifter")))
        (Op.constNat 15)) t
      = some (mdqBNibbleTile s0 b stride_bk stride_bn BN BK pn i) := by
    rw [mdqBNibbleTile_eq]
    exact mdq_bitAnd_eval Broadcast.scalarR _ _ t _ (Tile.scalar 15)
      (mdq_shiftRight_eval _ _ _ t (mdqBRaw s0 b stride_bk stride_bn BN BK pn i)
        (Tile.expandDim ⟨1, by simp⟩ (mdqShifter BK))
        (by rw [evalOp_ref]; exact htBRaw)
        (mdq_expandDim_eval _ _ t (mdqShifter BK) (by rw [evalOp_ref]; exact htShift)))
      (evalOp_constNat _ _)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h4)]
  -- 5. `b = b.to(tl.float16) * scales[None, :] - zeros[None, :]`
  have h5 : evalOp (Op.sub .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
          (Op.natToReal (Op.ref .nat [BK, BN] "b"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BN] "scales")))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BN] "zeros")))
        (t.setReg "b" .nat [BK, BN] (mdqBNibbleTile s0 b stride_bk stride_bn BN BK pn i))
      = some (mdqBDequantTile s0 scales b zeros NO_GROUPS groupsize stride_bk
          stride_bn stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
          BN BK pn i) := by
    rw [mdqBDequantTile_eq]
    exact mdq_subTile_eval NumericDType.real _ _ _ _ _ _
      (mdq_mulTile_eval NumericDType.real _ _ _ _ _ _
        (mdq_natToReal_eval _ _ _ (by rw [evalOp_ref]; simp))
        (mdq_expandDim_eval _ _ _ _ (by rw [evalOp_ref]; simpa using hscalesT)))
      (mdq_expandDim_eval _ _ _ _ (by rw [evalOp_ref]; simpa using hzerosT))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h5)]
  -- 6. `accumulator += tl.dot(a, b)`
  have h6 : evalOp (Op.add .real
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "accumulator")
        (Op.dot (batch := []) (Op.ref .real [BM, BK] "a")
          (Op.ref .real [BK, BN] "b")))
        (((t.setReg "b" .nat [BK, BN]
              (mdqBNibbleTile s0 b stride_bk stride_bn BN BK pn i)).setReg
            "b" .real [BK, BN]
            (mdqBDequantTile s0 scales b zeros NO_GROUPS groupsize stride_bk
              stride_bn stride_scales_g stride_scales_n stride_zeros_g
              stride_zeros_n BN BK pn i)))
      = some (mdqAccTile s0 a scales b zeros NO_GROUPS M groupsize stride_am
          stride_ak stride_bk stride_bn stride_scales_g stride_scales_n
          stride_zeros_g stride_zeros_n BM BN BK pm pn (i + 1)) := by
    rw [← mdqAccTile_dot_succ]
    exact mdq_addTile_eval NumericDType.real _ _ _ _ _ _
      (by rw [evalOp_ref]; simpa using htAcc)
      (mdq_dot_eval _ _ _ _ _ (by rw [evalOp_ref]; simpa using htATile)
        (by rw [evalOp_ref]; simp))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h6)]
  -- 7. `a_ptrs += BLOCK_SIZE_K * stride_ak`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_ak))) _
      = some (mdqAPtrs a stride_am stride_ak BM BK pm (i + 1)) from by
      rw [← mdqAPtrs_succ]
      exact mdq_ptrAdd_eval Broadcast.scalarR "a_ptrs" _ _ _
        (Tile.scalar (BK * stride_ak)) (by simpa using htAPtrs)
        (mdq_mulScalarNat_eval _ _ _ BK stride_ak (evalOp_constNat _ _)
          (evalOp_constNat _ _))))]
  -- 8. `b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "b_ptrs")
        (Op.mul .nat Broadcast.nil
          (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat BK)
            (Op.ref .nat [] "infearure_per_bits"))
          (Op.constNat stride_bk))) _
      = some (mdqBPtrs b stride_bk stride_bn BN BK pn (i + 1)) from by
      rw [← mdqBPtrs_succ]
      exact mdq_ptrAdd_eval Broadcast.scalarR "b_ptrs" _ _ _
        (Tile.scalar (BK / 8 * stride_bk)) (by simpa using htBPtrs)
        (mdq_mulScalarNat_eval _ _ _ (BK / 8) stride_bk
          (mdq_floorDivScalar_eval _ _ _ BK 8 (evalOp_constNat _ _)
            (by rw [evalOp_ref]; simpa using htIpb))
          (evalOp_constNat _ _))))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, hnext, htmem.trans hmem, htpids.trans hpids, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hkeepS .nat [] "bits" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) |>.trans hbits
  · simpa using htIpb
  · simpa using hkeepS .nat [] "pid_m" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) |>.trans hpm
  · simpa using hkeepS .nat [] "pid_n" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) |>.trans hpn
  · simpa using hkeepS .bool [BM, 1] "a_mask" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) |>.trans hmask
  · simpa using htShift
  · simpa using hkeepS .nat [BN] "zeros_shifter" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) |>.trans hzshift
  · simpa using hkeepS .ptr [BN] "scales_ptrs" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) |>.trans hspt
  · simpa using hkeepS .ptr [BN] "zeros_ptrs" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) |>.trans hzpt
  -- `pm` / `pn` are `set`-bound above, so these three need the defining equations
  -- to line up with the raw `pidM` / `pidN` inside `mdqInv`.
  · simp [hpmDef]
  · simp [hpnDef]
  · simp [hpmDef, hpnDef]
  · -- Under `NO_GROUPS` the branch left `scales` / `zeros` at row
    -- `groupRow Bool.true groupsize BK i`, which is `0` — the row the invariant asks
    -- for, and the row the *next* iteration will read them at.
    intro hf
    refine ⟨?_, ?_⟩
    · rw [show (0 : Nat) = groupRow NO_GROUPS groupsize BK i by
        rw [hf, groupRow_of_noGroups]]
      simpa using hscalesT
    · rw [show (0 : Nat) = groupRow NO_GROUPS groupsize BK i by
        rw [hf, groupRow_of_noGroups]]
      simpa using hzerosT

/-- The loop combinator writes the induction variable before each iteration, and
`mdqInv` constrains no register named `"k"`, so the invariant survives that write. -/
theorem mdqInv_setReg_k (s0 : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M N K groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK GM i j : Nat) (s : BlockState)
    (h : mdqInv s0 a scales b zeros NO_GROUPS M N K groupsize stride_am stride_ak
      stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
      stride_zeros_n BM BN BK GM i s) :
    mdqInv s0 a scales b zeros NO_GROUPS M N K groupsize stride_am stride_ak
      stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
      stride_zeros_n BM BN BK GM i (s.setReg "k" .nat [] (Tile.scalar j)) := by
  obtain ⟨hle, hmem, hpids, hbits, hipb, hpm, hpn, hmask, hshift, hzshift,
    hspt, hzpt, hapt, hbpt, hacc, hpreload⟩ := h
  exact ⟨hle, hmem, hpids, by simpa using hbits, by simpa using hipb,
    by simpa using hpm, by simpa using hpn, by simpa using hmask,
    by simpa using hshift, by simpa using hzshift, by simpa using hspt,
    by simpa using hzpt, by simpa using hapt, by simpa using hbpt,
    by simpa using hacc, fun hf => by simpa using hpreload hf⟩

/-! ### Collapsing the K loop

`forRangeDyn_inv` concludes only `stop ≤ final`, so the readout needs the other
inequality; that is what the invariant's own `i ≤ numPidK K BK` conjunct is for —
together they pin `final = numPidK K BK` and hand the accumulator to
`mdqAccTile_full`. -/

theorem mdqLoop_collapse (s0 : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M N K groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK GM : Nat) (s : BlockState)
    (hnpk : s.regs .nat [] "num_pid_k" = some (Tile.scalar (numPidK K BK)))
    (h0 : mdqInv s0 a scales b zeros NO_GROUPS M N K groupsize stride_am stride_ak
      stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
      stride_zeros_n BM BN BK GM 0 s) :
    ∃ sF, stepStmt (Stmt.forRangeDyn "k" (Op.constNat 0)
          (Op.ref .nat [] "num_pid_k") (Op.constNat 1)
          (mdqLoopBody groupsize stride_ak stride_bk stride_scales_g
            stride_zeros_g NO_GROUPS BM BN BK)) s = some sF
      ∧ mdqInv s0 a scales b zeros NO_GROUPS M N K groupsize stride_am stride_ak
          stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
          stride_zeros_n BM BN BK GM (numPidK K BK) sF := by
  obtain ⟨final, sF, hrun, hfinal, hP⟩ :=
    forRangeDyn_inv (idx := "k") (start := 0) (stop := numPidK K BK) (step := 1)
      (P := fun i t => mdqInv s0 a scales b zeros NO_GROUPS M N K groupsize stride_am
        stride_ak stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
        stride_zeros_n BM BN BK GM i t)
      (evalOp_constNat _ _) (by rw [evalOp_ref]; exact hnpk) (evalOp_constNat _ _)
      one_ne_zero h0
      (fun i t hi hinv => by
        obtain ⟨s', hs', hinv'⟩ :=
          mdqLoopBody_run s0 a scales b zeros NO_GROUPS M N K groupsize stride_am
            stride_ak stride_bk stride_bn stride_scales_g stride_scales_n
            stride_zeros_g stride_zeros_n BM BN BK GM i _ (by omega) (by simp)
            (mdqInv_setReg_k s0 a scales b zeros NO_GROUPS M N K groupsize stride_am
              stride_ak stride_bk stride_bn stride_scales_g stride_scales_n
              stride_zeros_g stride_zeros_n BM BN BK GM i i t hinv)
        exact ⟨s', hs', hinv'⟩)
  -- `hP.1` is the invariant's `final ≤ numPidK`; `hfinal` is the reverse.
  have hEq : final = numPidK K BK := le_antisymm hP.1 hfinal
  subst hEq
  exact ⟨sF, hrun, hP⟩

/-! ## The prologue's building blocks

Every prologue statement is one of two kinds: a `nat` **scalar** step (the pid
swizzle and the three `tl.cdiv`s), or a shaped **index tile** (the offset vectors,
the four pointer tiles, the mask and the two shifters). The scalar family all has
the same shape — `Tile.bop f Broadcast.nil (Tile.scalar u) (Tile.scalar v)` is
`Tile.scalar (f u v)` by `rfl` — so those recipes are listed together, followed by
one `evalOp` lemma per index tile that lands directly on the named tile. -/

/-- `setReg` leaves memory alone, at **function** level. The library's
`setReg_mem` is pointwise, and closing a twelve-deep tower's `t.mem = s.mem` by a
single `rfl` overruns `whnf`; twelve cheap rewrites do not. -/
private theorem mdq_setReg_mem {dtype : TileDType} {sh : TileShape}
    (s : BlockState) (nm : RegName) (v : Tile dtype sh) :
    (s.setReg nm dtype sh v).mem = s.mem := rfl

/-- `Op.div` on any numeric channel (this is what `tl.cdiv` expands to; `//` is
`Op.floorDiv`). -/
private theorem mdq_divTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.div h bc x y) t = some (Tile.bop h.div bc vx vy) := by
  rw [evalOp_div, hx, hy]
  rfl

/-- `%` on the `nat` channel. -/
private theorem mdq_mod_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mod IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.mod IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `<`. -/
private theorem mdq_ltTile_eval {dtype : TileDType} (h : ComparableDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.lt h bc x y) t = some (Tile.cop h.lt bc vx vy) := by
  rw [evalOp_lt, hx, hy]
  rfl

/-- `tl.where` — how `min(a, b)` is lowered, there being no `Op.min`. -/
private theorem mdq_where_eval {dtype : TileDType} {sh : TileShape}
    (c : Op .bool sh) (x y : Op dtype sh) (t : BlockState)
    (vc : Tile .bool sh) (vx vy : Tile dtype sh)
    (hc : evalOp c t = some vc) (hx : evalOp x t = some vx)
    (hy : evalOp y t = some vy) :
    evalOp (Op.where c x y) t = some (Tile.select vc vx vy) := by
  rw [evalOp_where, hc, hx, hy]
  rfl

/-- `tl.zeros` / `tl.full`. -/
private theorem mdq_full_eval {dtype : TileDType} (sh : TileShape) (e : Op dtype [])
    (t : BlockState) (v : Tile dtype []) (hv : evalOp e t = some v) :
    evalOp (Op.full sh e) t
      = some (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype sh) := by
  rw [evalOp_full, hv]
  rfl

/-- A pointer tile built from a bare region base, as every `R + offs` in the
prologue is. -/
private theorem mdq_ptrAddBase_eval {d : TileDType} {b out : TileShape}
    (bc : Broadcast [] b out) (rg : Region d) (t : BlockState)
    (off : Op .nat b) (ov : Tile .nat b) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ptrBase rg) off) t
      = some (Tile.ptrAdd bc (Tile.scalar ((Region.cast rg : RegionName), 0)) ov) := by
  simp only [evalOp, ho]
  rfl

/-! ### `nat` scalars -/

private theorem mdq_addScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.add .nat Broadcast.nil x y) t = some (Tile.scalar (u + v)) := by
  rw [mdq_addTile_eval NumericDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem mdq_subScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.sub .nat Broadcast.nil x y) t = some (Tile.scalar (u - v)) := by
  rw [mdq_subTile_eval NumericDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem mdq_divScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.div .nat Broadcast.nil x y) t = some (Tile.scalar (u / v)) := by
  rw [mdq_divTile_eval NumericDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem mdq_modScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u % v)) := by
  rw [mdq_mod_eval Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem mdq_ltScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.lt ComparableDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (decide (u < v))) := by
  rw [mdq_ltTile_eval ComparableDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem mdq_whereScalarNat_eval (c : Op .bool []) (x y : Op .nat [])
    (t : BlockState) (cv : Bool) (u v : Nat)
    (hc : evalOp c t = some (Tile.scalar cv))
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.where c x y) t = some (Tile.scalar (if cv then u else v)) := by
  rw [mdq_where_eval c x y t _ _ _ hc hx hy]
  rfl

/-- `min` is spelled as a `tl.where` by the DSL; this is the arithmetic identity
that turns the lowered form back into `groupSizeM`'s `min`. -/
private theorem mdq_min_as_where (u v : Nat) : (if u < v then u else v) = min u v := by
  rcases Nat.lt_or_ge u v with h | h
  · rw [if_pos h]; omega
  · rw [if_neg (by omega)]; omega

/-! ### Index tiles -/

/-- `pid_* * BLOCK + tl.arange(0, BLOCK)`, and `tl.arange` on its own at `base = 0`. -/
def mdqOffsTile (base BD : Nat) : Tile .nat [BD] := ⟨fun idx => base + idx.1.val⟩

private theorem mdq_arange_eq_offs (BD : Nat) :
    (Tile.vec (fun i => (i.val : Nat)) : Tile .nat [BD]) = mdqOffsTile 0 BD := by
  apply Tile.ext
  intro idx
  simp [mdqOffsTile, Tile.vec]

private theorem mdq_offs_eval (nm : RegName) (t : BlockState) (BD base c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar base)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c))
        (Op.arange BD)) t
      = some (mdqOffsTile (base * c) BD) := by
  rw [mdq_addTile_eval NumericDType.nat Broadcast.scalarL _ _ t
    (Tile.scalar (base * c)) (Tile.vec (fun i => (i.val : Nat)))
    (mdq_mulScalarNat_eval _ _ t base c (by rw [evalOp_ref]; exact hr)
      (evalOp_constNat _ _))
    (evalOp_arange _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqOffsTile, Tile.vec, Broadcast.rightIndex, NumericDType.add]

/-- `a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak`. -/
private theorem mdq_aPtrsInit_eval (a : RegionName) (t : BlockState)
    (stride_am stride_ak BM BK pm : Nat)
    (ham : t.regs .nat [BM] "offs_am" = some (mdqOffsTile (pm * BM) BM))
    (hok : t.regs .nat [BK] "offs_k" = some (mdqOffsTile 0 BK)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase a)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
            (Op.constNat stride_am))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_ak)))) t
      = some (mdqAPtrs a stride_am stride_ak BM BK pm 0) := by
  rw [mdq_ptrAddBase_eval _ _ t _ _
    (mdq_addTile_eval NumericDType.nat _ _ _ t _ _
      (mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar stride_am)
        (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact ham))
        (evalOp_constNat _ _))
      (mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar stride_ak)
        (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hok))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqAPtrs, mdqAAddr, mdqOffsTile, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `a_mask = offs_am[:, None] < M`. -/
private theorem mdq_aMaskInit_eval (t : BlockState) (M BM pm : Nat)
    (ham : t.regs .nat [BM] "offs_am" = some (mdqOffsTile (pm * BM) BM)) :
    evalOp ((Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
        (Op.constNat M) : Op .bool [BM, 1])) t
      = some (mdqAMask M BM pm) := by
  rw [mdq_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar M)
    (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact ham))
    (evalOp_constNat _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqAMask, mdqOffsTile, Tile.cop_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, ComparableDType.lt]

/-- `b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn`. -/
private theorem mdq_bPtrsInit_eval (b : Region .nat) (t : BlockState)
    (stride_bk stride_bn BN BK pn : Nat)
    (hok : t.regs .nat [BK] "offs_k" = some (mdqOffsTile 0 BK))
    (hbn : t.regs .nat [BN] "offs_bn" = some (mdqOffsTile (pn * BN) BN))
    (hipb : t.regs .nat [] "infearure_per_bits" = some (Tile.scalar 8)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase b)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.floorDiv IntegralDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
              (Op.ref .nat [] "infearure_per_bits"))
            (Op.constNat stride_bk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
            (Op.constNat stride_bn)))) t
      = some (mdqBPtrs b stride_bk stride_bn BN BK pn 0) := by
  rw [mdq_ptrAddBase_eval _ _ t _ _
    (mdq_addTile_eval NumericDType.nat _ _ _ t _ _
      (mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar stride_bk)
        (mdq_floorDiv_eval Broadcast.scalarR _ _ t _ (Tile.scalar 8)
          (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hok))
          (by rw [evalOp_ref]; exact hipb))
        (evalOp_constNat _ _))
      (mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar stride_bn)
        (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hbn))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqBPtrs, mdqBAddr, mdqOffsTile, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `scales_ptrs = scales_ptr + offs_bn * stride_scales_n`. -/
private theorem mdq_scalesPtrsInit_eval (scales : RegionName) (t : BlockState)
    (stride_scales_n BN pn : Nat)
    (hbn : t.regs .nat [BN] "offs_bn" = some (mdqOffsTile (pn * BN) BN)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase scales)
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn")
          (Op.constNat stride_scales_n))) t
      = some (mdqScalesPtrs scales stride_scales_n BN pn) := by
  rw [mdq_ptrAddBase_eval _ _ t _ _
    (mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
      (Tile.scalar stride_scales_n) (by rw [evalOp_ref]; exact hbn)
      (evalOp_constNat _ _))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqScalesPtrs, mdqScalesAddr, mdqOffsTile, Tile.ptrAdd_data, Tile.bop_data,
    Broadcast.leftIndex, NumericDType.mul]

/-- `zeros_ptrs = zeros_ptr + (offs_bn // 8) * stride_zeros_n`. -/
private theorem mdq_zerosPtrsInit_eval (zeros : Region .nat) (t : BlockState)
    (stride_zeros_n BN pn : Nat)
    (hbn : t.regs .nat [BN] "offs_bn" = some (mdqOffsTile (pn * BN) BN))
    (hipb : t.regs .nat [] "infearure_per_bits" = some (Tile.scalar 8)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase zeros)
        (Op.mul .nat Broadcast.scalarR
          (Op.floorDiv IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn")
            (Op.ref .nat [] "infearure_per_bits"))
          (Op.constNat stride_zeros_n))) t
      = some (mdqZerosPtrs zeros stride_zeros_n BN pn) := by
  rw [mdq_ptrAddBase_eval _ _ t _ _
    (mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
      (Tile.scalar stride_zeros_n)
      (mdq_floorDiv_eval Broadcast.scalarR _ _ t _ (Tile.scalar 8)
        (by rw [evalOp_ref]; exact hbn) (by rw [evalOp_ref]; exact hipb))
      (evalOp_constNat _ _))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqZerosPtrs, mdqZerosAddr, mdqOffsTile, Tile.ptrAdd_data, Tile.bop_data,
    Broadcast.leftIndex, NumericDType.mul]

/-- `shifter = (offs_k % infearure_per_bits) * bits`. -/
private theorem mdq_shifterInit_eval (t : BlockState) (BK : Nat)
    (hok : t.regs .nat [BK] "offs_k" = some (mdqOffsTile 0 BK))
    (hipb : t.regs .nat [] "infearure_per_bits" = some (Tile.scalar 8))
    (hbits : t.regs .nat [] "bits" = some (Tile.scalar 4)) :
    evalOp (Op.mul .nat Broadcast.scalarR
        (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BK] "offs_k")
          (Op.ref .nat [] "infearure_per_bits"))
        (Op.ref .nat [] "bits")) t
      = some (mdqShifter BK) := by
  rw [mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar 4)
    (mdq_mod_eval Broadcast.scalarR _ _ t _ (Tile.scalar 8)
      (by rw [evalOp_ref]; exact hok) (by rw [evalOp_ref]; exact hipb))
    (by rw [evalOp_ref]; exact hbits)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqShifter, mdqOffsTile, Tile.bop_data, Broadcast.leftIndex,
    NumericDType.mul]

/-- `zeros_shifter = (offs_bn % infearure_per_bits) * bits`. -/
private theorem mdq_zerosShifterInit_eval (t : BlockState) (BN pn : Nat)
    (hbn : t.regs .nat [BN] "offs_bn" = some (mdqOffsTile (pn * BN) BN))
    (hipb : t.regs .nat [] "infearure_per_bits" = some (Tile.scalar 8))
    (hbits : t.regs .nat [] "bits" = some (Tile.scalar 4)) :
    evalOp (Op.mul .nat Broadcast.scalarR
        (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn")
          (Op.ref .nat [] "infearure_per_bits"))
        (Op.ref .nat [] "bits")) t
      = some (mdqZerosShifter BN pn) := by
  rw [mdq_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar 4)
    (mdq_mod_eval Broadcast.scalarR _ _ t _ (Tile.scalar 8)
      (by rw [evalOp_ref]; exact hbn) (by rw [evalOp_ref]; exact hipb))
    (by rw [evalOp_ref]; exact hbits)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqZerosShifter, mdqOffsTile, Tile.bop_data, Broadcast.leftIndex,
    NumericDType.mul]

/-! ### The prologue's `NO_GROUPS` pre-load

The mirror of `mdq_groupBranch_run`, on the positive guard and at group row `0`. Its
conclusion is deliberately weaker: nothing after it *needs* `scales` / `zeros` unless
`NO_GROUPS` holds, because otherwise the K loop reloads them every step. -/

/-- A `[BN]` `.real` load straight off `scales_ptrs` — group row `0`. -/
private theorem mdq_scalesLoad0_eq (s0 : BlockState) (scales : RegionName)
    (t : BlockState) (stride_scales_g stride_scales_n BN pn : Nat)
    (hmem : t.mem = s0.mem)
    (hp : t.regs .ptr [BN] "scales_ptrs"
      = some (mdqScalesPtrs scales stride_scales_n BN pn)) :
    evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN] "scales_ptrs"))
        MaskOpt.none) t
      = some (mdqScalesTile s0 scales stride_scales_g stride_scales_n BN pn 0) := by
  rw [mdq_load_ptr_none "scales_ptrs" t _ hp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [mdqScalesTile, scalesElem, mdqScalesPtrs, mdqScalesAddr,
    BlockState.readMemValue_real, BlockState.readMem, hmem, Nat.zero_mul, Nat.add_zero]

/-- The packed zero-point load straight off `zeros_ptrs` — group row `0`. -/
private theorem mdq_zerosLoad0_eq (s0 : BlockState) (zeros : Region .nat)
    (t : BlockState) (stride_zeros_g stride_zeros_n BN pn : Nat)
    (hmem : t.mem = s0.mem)
    (hp : t.regs .ptr [BN] "zeros_ptrs"
      = some (mdqZerosPtrs zeros stride_zeros_n BN pn)) :
    evalOp (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BN] "zeros_ptrs"))
        MaskOpt.none) t
      = some (mdqZerosRaw s0 zeros stride_zeros_g stride_zeros_n BN pn 0) := by
  rw [mdq_load_ptr_none "zeros_ptrs" t _ hp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [mdqZerosRaw, zerosWord, mdqZerosPtrs, mdqZerosAddr,
    BlockState.readMemValue, BlockState.readMemTyped, hmem, Nat.zero_mul, Nat.add_zero]

/-- `Op.constBool`. -/
private theorem mdq_constBool_eval (flag : Bool) (t : BlockState) :
    evalOp (Op.constBool flag) t = some (Tile.scalar flag) := by
  simp only [evalOp]

private theorem mdq_groupPreload_run (s0 : BlockState) (scales : RegionName)
    (zeros : Region .nat) (t : BlockState) (NO_GROUPS : Bool)
    (stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n BN pn : Nat)
    (hmem : t.mem = s0.mem)
    (hsp : t.regs .ptr [BN] "scales_ptrs"
      = some (mdqScalesPtrs scales stride_scales_n BN pn))
    (hzp : t.regs .ptr [BN] "zeros_ptrs"
      = some (mdqZerosPtrs zeros stride_zeros_n BN pn))
    (hzs : t.regs .nat [BN] "zeros_shifter" = some (mdqZerosShifter BN pn)) :
    ∃ t', stepStmt (Stmt.ifThen (Op.constBool NO_GROUPS) (mdqGroupPreload BN)) t
        = some t'
      ∧ t'.mem = t.mem
      ∧ t'.pids = t.pids
      ∧ (∀ (dtype : TileDType) (sh : TileShape) (nm : RegName),
          nm ≠ "scales" → nm ≠ "zeros" → t'.regs dtype sh nm = t.regs dtype sh nm)
      ∧ (NO_GROUPS = Bool.true →
          t'.regs .real [BN] "scales"
              = some (mdqScalesTile s0 scales stride_scales_g stride_scales_n BN pn 0)
            ∧ t'.regs .real [BN] "zeros"
              = some (mdqZerosTile s0 scales zeros stride_scales_g stride_scales_n
                  stride_zeros_g stride_zeros_n BN pn 0)) := by
  by_cases hng : NO_GROUPS = Bool.true
  · subst hng
    have hstep : stepStmt (Stmt.ifThen (Op.constBool Bool.true) (mdqGroupPreload BN)) t
        = stepStmts (mdqGroupPreload BN) t := by
      rw [mdq_ifThen_step, mdq_constBool_eval]
      rfl
    rw [hstep]
    unfold mdqGroupPreload
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (mdq_scalesLoad0_eq s0 scales t stride_scales_g stride_scales_n BN pn hmem hsp))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (mdq_zerosLoad0_eq s0 zeros _ stride_zeros_g stride_zeros_n BN pn
        (by simpa using hmem) (by simpa using hzp)))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (mdq_bitAnd_eval Broadcast.scalarR _ _ _
        (Tile.bop (· >>> ·) (Broadcast.consSame Broadcast.nil)
          (mdqZerosRaw s0 zeros stride_zeros_g stride_zeros_n BN pn 0)
          (mdqZerosShifter BN pn))
        (Tile.scalar 15)
        (mdq_shiftRight_eval (Broadcast.consSame Broadcast.nil) _ _ _
          (mdqZerosRaw s0 zeros stride_zeros_g stride_zeros_n BN pn 0)
          (mdqZerosShifter BN pn) (by rw [evalOp_ref]; simp)
          (by rw [evalOp_ref]; simpa using hzs))
        (evalOp_constNat _ _)))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (mdq_mulTile_eval NumericDType.real (Broadcast.consSame Broadcast.nil) _ _ _
        (Tile.natToReal
          (Tile.bop (· &&& ·) Broadcast.scalarR
            (Tile.bop (· >>> ·) (Broadcast.consSame Broadcast.nil)
              (mdqZerosRaw s0 zeros stride_zeros_g stride_zeros_n BN pn 0)
              (mdqZerosShifter BN pn))
            (Tile.scalar 15)))
        (mdqScalesTile s0 scales stride_scales_g stride_scales_n BN pn 0)
        (mdq_natToReal_eval _ _ _ (by rw [evalOp_ref]; simp))
        (by rw [evalOp_ref]; simp)))]
    rw [stepStmts.nil]
    refine ⟨_, rfl, rfl, rfl, ?_, ?_⟩
    · intro dtype sh nm h1 h2
      simp [h1, h2]
    · intro _
      refine ⟨by simp, ?_⟩
      rw [mdqZerosTile_eq]
      simp
  · -- The guard is `false`: nothing runs, and the conclusion's clause is vacuous.
    have hflag : NO_GROUPS = Bool.false := by simpa using hng
    subst hflag
    refine ⟨t, ?_, rfl, rfl, fun _ _ _ _ _ => rfl, fun hf => absurd hf (by simp)⟩
    rw [mdq_ifThen_step, mdq_constBool_eval]
    rfl

/-! ### The pid swizzle, statement by statement

Each recipe lands on the **named** quantity rather than on raw arithmetic, so the
walk below carries `numPidM` / `groupId` / `pidM` and never a `(M + BM - 1) / BM`.
Each is definitionally the arithmetic it names, which is why the `rw` closes. -/

private theorem mdq_cdiv_eval (t : BlockState) (X BX : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat X) (Op.constNat BX)) (Op.constNat 1))
        (Op.constNat BX)) t
      = some (Tile.scalar ((X + BX - 1) / BX)) :=
  mdq_divScalarNat_eval _ _ t (X + BX - 1) BX
    (mdq_subScalarNat_eval _ _ t (X + BX) 1
      (mdq_addScalarNat_eval _ _ t X BX (evalOp_constNat _ _) (evalOp_constNat _ _))
      (evalOp_constNat _ _))
    (evalOp_constNat _ _)

private theorem mdq_numPidM_eval (t : BlockState) (M BM : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
        (Op.constNat BM)) t
      = some (Tile.scalar (numPidM M BM)) := mdq_cdiv_eval t M BM

private theorem mdq_numPidN_eval (t : BlockState) (N BN : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
        (Op.constNat BN)) t
      = some (Tile.scalar (numPidN N BN)) := mdq_cdiv_eval t N BN

private theorem mdq_numPidK_eval (t : BlockState) (K BK : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK)) (Op.constNat 1))
        (Op.constNat BK)) t
      = some (Tile.scalar (numPidK K BK)) := mdq_cdiv_eval t K BK

private theorem mdq_numPidInGroup_eval (t : BlockState) (N BN GM : Nat)
    (hnpn : t.regs .nat [] "num_pid_n" = some (Tile.scalar (numPidN N BN))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.constNat GM)
        (Op.ref .nat [] "num_pid_n")) t
      = some (Tile.scalar (numPidInGroup N BN GM)) := by
  rw [mdq_mulScalarNat_eval _ _ t GM (numPidN N BN) (evalOp_constNat _ _)
    (by rw [evalOp_ref]; exact hnpn)]
  rfl

private theorem mdq_groupId_eval (s t : BlockState) (N BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hnig : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (numPidInGroup N BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "num_pid_in_group")) t
      = some (Tile.scalar (groupId s N BN GM)) := by
  rw [mdq_floorDivScalar_eval _ _ t (s.pids 0) (numPidInGroup N BN GM)
    (by rw [evalOp_ref]; exact hpid) (by rw [evalOp_ref]; exact hnig)]
  rfl

private theorem mdq_firstPidM_eval (s t : BlockState) (N BN GM : Nat)
    (hgid : t.regs .nat [] "group_id" = some (Tile.scalar (groupId s N BN GM))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
        (Op.constNat GM)) t
      = some (Tile.scalar (firstPidM s N BN GM)) := by
  rw [mdq_mulScalarNat_eval _ _ t (groupId s N BN GM) GM
    (by rw [evalOp_ref]; exact hgid) (evalOp_constNat _ _)]
  rfl

private theorem mdq_groupSizeM_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hnpm : t.regs .nat [] "num_pid_m" = some (Tile.scalar (numPidM M BM)))
    (hfpm : t.regs .nat [] "first_pid_m"
      = some (Tile.scalar (firstPidM s N BN GM))) :
    evalOp (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
            (Op.ref .nat [] "first_pid_m"))
          (Op.constNat GM))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
          (Op.ref .nat [] "first_pid_m"))
        (Op.constNat GM)) t
      = some (Tile.scalar (groupSizeM s M N BM BN GM)) := by
  have hsub : evalOp (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
      (Op.ref .nat [] "first_pid_m")) t
      = some (Tile.scalar (numPidM M BM - firstPidM s N BN GM)) :=
    mdq_subScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hnpm)
      (by rw [evalOp_ref]; exact hfpm)
  rw [mdq_whereScalarNat_eval _ _ _ t
    (decide (numPidM M BM - firstPidM s N BN GM < GM))
    (numPidM M BM - firstPidM s N BN GM) GM
    (mdq_ltScalarNat_eval _ _ t _ _ hsub (evalOp_constNat _ _)) hsub
    (evalOp_constNat _ _)]
  simp only [decide_eq_true_eq, mdq_min_as_where, groupSizeM]

private theorem mdq_pidM_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hfpm : t.regs .nat [] "first_pid_m" = some (Tile.scalar (firstPidM s N BN GM)))
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hgsm : t.regs .nat [] "group_size_m"
      = some (Tile.scalar (groupSizeM s M N BM BN GM))) :
    evalOp (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "group_size_m"))) t
      = some (Tile.scalar (pidM s M N BM BN GM)) := by
  rw [mdq_addScalarNat_eval _ _ t (firstPidM s N BN GM)
    (s.pids 0 % groupSizeM s M N BM BN GM) (by rw [evalOp_ref]; exact hfpm)
    (mdq_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hgsm))]
  rfl

private theorem mdq_pidN_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hnig : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (numPidInGroup N BN GM)))
    (hgsm : t.regs .nat [] "group_size_m"
      = some (Tile.scalar (groupSizeM s M N BM BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "num_pid_in_group"))
        (Op.ref .nat [] "group_size_m")) t
      = some (Tile.scalar (pidN s M N BM BN GM)) := by
  rw [mdq_floorDivScalar_eval _ _ t (s.pids 0 % numPidInGroup N BN GM)
    (groupSizeM s M N BM BN GM)
    (mdq_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hnig))
    (by rw [evalOp_ref]; exact hgsm)]
  rfl

private theorem mdq_offsK_eval (t : BlockState) (BK : Nat) :
    evalOp (Op.arange BK) t = some (mdqOffsTile 0 BK) := by
  rw [evalOp_arange, mdq_arange_eq_offs]

/-! ### The prologue walks -/

/-- The twelve scalar statements. Memory is untouched, and the only registers
anything downstream reads are the two packing constants, the K trip count and the
block coordinates. -/
theorem mdqPreLoopScalars_run (s : BlockState) (M N K BM BN BK GM : Nat) :
    ∃ t, stepStmts (mdqPreLoopScalars M N K BM BN BK GM) s = some t
      ∧ t.mem = s.mem
      ∧ t.pids = s.pids
      ∧ t.regs .nat [] "bits" = some (Tile.scalar 4)
      ∧ t.regs .nat [] "infearure_per_bits" = some (Tile.scalar 8)
      ∧ t.regs .nat [] "num_pid_k" = some (Tile.scalar (numPidK K BK))
      ∧ t.regs .nat [] "pid_m" = some (Tile.scalar (pidM s M N BM BN GM))
      ∧ t.regs .nat [] "pid_n" = some (Tile.scalar (pidN s M N BM BN GM)) := by
  unfold mdqPreLoopScalars
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat 4 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat 8 _))]
  -- `pid` carries `(state).pids 0`; every later reader reconciles that with
  -- `s.pids 0` by `simp`, since `setReg` leaves `pids` alone.
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mdq_numPidM_eval _ M BM))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mdq_numPidN_eval _ N BN))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mdq_numPidK_eval _ K BK))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_numPidInGroup_eval _ N BN GM (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_groupId_eval s _ N BN GM (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_firstPidM_eval s _ N BN GM (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_groupSizeM_eval s _ M N BM BN GM (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_pidM_eval s _ M N BM BN GM (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_pidN_eval s _ M N BM BN GM (by simp) (by simp) (by simp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp [mdq_setReg_mem]

/-- The ten index statements. Everything they read is either written here or is one
of the four registers the scalar half left behind. -/
theorem mdqPreLoopIndex_run (a scales : RegionName) (b zeros : Region .nat)
    (M stride_am stride_ak stride_bk stride_bn stride_scales_n stride_zeros_n
      BM BN BK pm pn : Nat) (t : BlockState)
    (hbits : t.regs .nat [] "bits" = some (Tile.scalar 4))
    (hipb : t.regs .nat [] "infearure_per_bits" = some (Tile.scalar 8))
    (hpm : t.regs .nat [] "pid_m" = some (Tile.scalar pm))
    (hpn : t.regs .nat [] "pid_n" = some (Tile.scalar pn)) :
    ∃ t', stepStmts (mdqPreLoopIndex a scales b zeros M stride_am stride_ak stride_bk
          stride_bn stride_scales_n stride_zeros_n BM BN BK) t = some t'
      ∧ t'.mem = t.mem
      ∧ t'.pids = t.pids
      ∧ (∀ (dtype : TileDType) (sh : TileShape) (nm : RegName),
          nm ≠ "offs_am" → nm ≠ "offs_bn" → nm ≠ "offs_k" → nm ≠ "a_ptrs" →
          nm ≠ "a_mask" → nm ≠ "b_ptrs" → nm ≠ "scales_ptrs" → nm ≠ "zeros_ptrs" →
          nm ≠ "shifter" → nm ≠ "zeros_shifter" →
          t'.regs dtype sh nm = t.regs dtype sh nm)
      ∧ t'.regs .bool [BM, 1] "a_mask" = some (mdqAMask M BM pm)
      ∧ t'.regs .nat [BK] "shifter" = some (mdqShifter BK)
      ∧ t'.regs .nat [BN] "zeros_shifter" = some (mdqZerosShifter BN pn)
      ∧ t'.regs .ptr [BN] "scales_ptrs"
          = some (mdqScalesPtrs scales stride_scales_n BN pn)
      ∧ t'.regs .ptr [BN] "zeros_ptrs"
          = some (mdqZerosPtrs zeros stride_zeros_n BN pn)
      ∧ t'.regs .ptr [BM, BK] "a_ptrs"
          = some (mdqAPtrs a stride_am stride_ak BM BK pm 0)
      ∧ t'.regs .ptr [BK, BN] "b_ptrs"
          = some (mdqBPtrs b stride_bk stride_bn BN BK pn 0) := by
  unfold mdqPreLoopIndex
  -- the three offset vectors
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_offs_eval "pid_m" t BM pm BM hpm))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_offs_eval "pid_n" _ BN pn BN (by simpa using hpn)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mdq_offsK_eval _ BK))]
  -- the four pointer tiles, the row mask and the two shifters
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_aPtrsInit_eval a _ stride_am stride_ak BM BK pm (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_aMaskInit_eval _ M BM pm (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_bPtrsInit_eval b _ stride_bk stride_bn BN BK pn (by simp) (by simp)
      (by simpa using hipb)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_scalesPtrsInit_eval scales _ stride_scales_n BN pn (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_zerosPtrsInit_eval zeros _ stride_zeros_n BN pn (by simp)
      (by simpa using hipb)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_shifterInit_eval _ BK (by simp) (by simpa using hipb)
      (by simpa using hbits)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_zerosShifterInit_eval _ BN pn (by simp) (by simpa using hipb)
      (by simpa using hbits)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [mdq_setReg_mem]
  · simp
  · intro dtype sh nm h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
    simp [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10]
  all_goals simp

/-- The prologue's second half, ending on `mdqInv` at step `0`. `num_pid_k` is
carried through untouched — the loop combinator reads it from the post-prologue
state. -/
theorem mdqPreLoopTiles_run (s0 : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M N K groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK GM : Nat) (t : BlockState)
    (hmem : t.mem = s0.mem)
    (hpids : t.pids = s0.pids)
    (hbits : t.regs .nat [] "bits" = some (Tile.scalar 4))
    (hipb : t.regs .nat [] "infearure_per_bits" = some (Tile.scalar 8))
    (hpm : t.regs .nat [] "pid_m" = some (Tile.scalar (pidM s0 M N BM BN GM)))
    (hpn : t.regs .nat [] "pid_n" = some (Tile.scalar (pidN s0 M N BM BN GM))) :
    ∃ t', stepStmts (mdqPreLoopTiles a scales b zeros M stride_am stride_ak
          stride_bk stride_bn stride_scales_n stride_zeros_n NO_GROUPS BM BN BK) t
        = some t'
      ∧ t'.regs .nat [] "num_pid_k" = t.regs .nat [] "num_pid_k"
      ∧ mdqInv s0 a scales b zeros NO_GROUPS M N K groupsize stride_am stride_ak
          stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
          stride_zeros_n BM BN BK GM 0 t' := by
  obtain ⟨t1, hrun1, h1mem, h1pids, h1keep, h1mask, h1shift, h1zshift, h1sp, h1zp,
    h1ap, h1bp⟩ :=
    mdqPreLoopIndex_run a scales b zeros M stride_am stride_ak stride_bk stride_bn
      stride_scales_n stride_zeros_n BM BN BK (pidM s0 M N BM BN GM)
      (pidN s0 M N BM BN GM) t hbits hipb hpm hpn
  unfold mdqPreLoopTiles
  rw [stepStmts.append_some hrun1]
  -- the `NO_GROUPS` pre-load, then the zeroed accumulator
  obtain ⟨t2, hpre, h2mem, h2pids, h2keep, h2load⟩ :=
    mdq_groupPreload_run s0 scales zeros t1 NO_GROUPS stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BN (pidN s0 M N BM BN GM)
      (h1mem.trans hmem) h1sp h1zp h1zshift
  rw [stepStmts.cons_some hpre]
  -- The accumulator statement is bundled into an opaque `t3` for the same reason the
  -- two branches are: every register below is then reached by a named rewrite rather
  -- than by `simp`-ing through a `setReg` applied to a state that is itself opaque.
  obtain ⟨t3, hrun3, h3mem, h3pids, h3keep, h3acc⟩ :
      ∃ t3, stepStmt (Stmt.assign .real [BM, BN] "accumulator"
              (Op.full [BM, BN] (Op.const 0))) t2 = some t3
        ∧ t3.mem = t2.mem
        ∧ t3.pids = t2.pids
        ∧ (∀ (dtype : TileDType) (sh : TileShape) (nm : RegName),
            nm ≠ "accumulator" → t3.regs dtype sh nm = t2.regs dtype sh nm)
        ∧ t3.regs .real [BM, BN] "accumulator"
            = some (mdqAccTile s0 a scales b zeros NO_GROUPS M groupsize stride_am
                stride_ak stride_bk stride_bn stride_scales_g stride_scales_n
                stride_zeros_g stride_zeros_n BM BN BK (pidM s0 M N BM BN GM)
                (pidN s0 M N BM BN GM) 0) := by
    refine ⟨_, stepStmt_assign_eq_some
      (show evalOp (Op.full [BM, BN] (Op.const 0)) t2
          = some (mdqAccTile s0 a scales b zeros NO_GROUPS M groupsize stride_am
              stride_ak stride_bk stride_bn stride_scales_g stride_scales_n
              stride_zeros_g stride_zeros_n BM BN BK (pidM s0 M N BM BN GM)
              (pidN s0 M N BM BN GM) 0) from by
        rw [mdqAccTile_zero]
        exact mdq_full_eval [BM, BN] (Op.const 0) t2 _ (evalOp_const 0 t2)),
      rfl, rfl, ?_, ?_⟩
    · intro dtype sh nm h
      simp [h]
    · simp
  rw [stepStmts.cons_some hrun3, stepStmts.nil]
  -- Every register is now reached through the three preservation clauses.
  have keep31 : ∀ (dtype : TileDType) (sh : TileShape) (nm : RegName),
      nm ≠ "accumulator" → nm ≠ "scales" → nm ≠ "zeros" →
      t3.regs dtype sh nm = t1.regs dtype sh nm := by
    intro dtype sh nm h1 h2 h3
    rw [h3keep dtype sh nm h1, h2keep dtype sh nm h2 h3]
  refine ⟨_, rfl, ?_, Nat.zero_le _, h3mem.trans (h2mem.trans (h1mem.trans hmem)),
    h3pids.trans (h2pids.trans (h1pids.trans hpids)), ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, h3acc, ?_⟩
  · rw [keep31 .nat [] "num_pid_k" (by decide) (by decide) (by decide),
      h1keep .nat [] "num_pid_k" (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
  · rw [keep31 .nat [] "bits" (by decide) (by decide) (by decide),
      h1keep .nat [] "bits" (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hbits
  · rw [keep31 .nat [] "infearure_per_bits" (by decide) (by decide) (by decide),
      h1keep .nat [] "infearure_per_bits" (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide)]
    exact hipb
  · rw [keep31 .nat [] "pid_m" (by decide) (by decide) (by decide),
      h1keep .nat [] "pid_m" (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hpm
  · rw [keep31 .nat [] "pid_n" (by decide) (by decide) (by decide),
      h1keep .nat [] "pid_n" (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hpn
  · rw [keep31 .bool [BM, 1] "a_mask" (by decide) (by decide) (by decide)]
    exact h1mask
  · rw [keep31 .nat [BK] "shifter" (by decide) (by decide) (by decide)]
    exact h1shift
  · rw [keep31 .nat [BN] "zeros_shifter" (by decide) (by decide) (by decide)]
    exact h1zshift
  · rw [keep31 .ptr [BN] "scales_ptrs" (by decide) (by decide) (by decide)]
    exact h1sp
  · rw [keep31 .ptr [BN] "zeros_ptrs" (by decide) (by decide) (by decide)]
    exact h1zp
  · rw [keep31 .ptr [BM, BK] "a_ptrs" (by decide) (by decide) (by decide)]
    exact h1ap
  · rw [keep31 .ptr [BK, BN] "b_ptrs" (by decide) (by decide) (by decide)]
    exact h1bp
  · intro hf
    obtain ⟨hs, hz⟩ := h2load hf
    exact ⟨(h3keep .real [BN] "scales" (by decide)).trans hs,
      (h3keep .real [BN] "zeros" (by decide)).trans hz⟩

/-! ## The output store

A masked `.ptr` store, so — unlike `chunk_bwd_dqkg`'s block-pointer stores — the
`c_mask` is the only gate and there is no boundary check to discharge. The lane-to-
address map has to be injective for the readback to name a unique lane; that is the
headline's `hInj`, and `cAddr_injective` discharges it from the two conditions a
row-major `C` satisfies. -/

/-- The `C` pointer tile. -/
noncomputable def mdqCPtrs (c : RegionName) (stride_cm stride_cn BM BN pm pn : Nat) :
    Tile .ptr [BM, BN] :=
  ⟨fun idx => (c, cAddr stride_cm stride_cn BM BN pm pn idx)⟩

/-- `c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)`. -/
def mdqCMask (M N BM BN pm pn : Nat) : Tile .bool [BM, BN] :=
  ⟨fun idx => decide (pm * BM + idx.1.val < M) && decide (pn * BN + idx.2.1.val < N)⟩

/-- The post-store state: one masked scatter over the `[BM, BN]` output tile. -/
noncomputable def mdqStoreState (c : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat)
    (f : TileIndex [BM, BN] → ℝ) (t : BlockState) : BlockState :=
  (TileShape.allIndices [BM, BN]).foldl
    (fun acc i => if pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N then
        acc.writeMem c (cAddr stride_cm stride_cn BM BN pm pn i) (f i)
      else acc) t

/-- `&` on the bool channel. -/
private theorem mdq_boolAnd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .bool a) (y : Op .bool b) (t : BlockState)
    (vx : Tile .bool a) (vy : Tile .bool b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.boolAnd bc x y) t
      = some (Tile.bop (fun u v : Bool => u && v) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

private theorem mdq_cPtrsInit_eval (c : RegionName) (t : BlockState)
    (stride_cm stride_cn BM BN pm pn : Nat)
    (hcm : t.regs .nat [BM] "offs_cm" = some (mdqOffsTile (pm * BM) BM))
    (hcn : t.regs .nat [BN] "offs_cn" = some (mdqOffsTile (pn * BN) BN)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase c)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_cm)
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_cn)
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) t
      = some (mdqCPtrs c stride_cm stride_cn BM BN pm pn) := by
  rw [mdq_ptrAddBase_eval _ _ t _ _
    (mdq_addTile_eval NumericDType.nat _ _ _ t _ _
      (mdq_mulTile_eval NumericDType.nat Broadcast.scalarL _ _ t
        (Tile.scalar stride_cm) _ (evalOp_constNat _ _)
        (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcm)))
      (mdq_mulTile_eval NumericDType.nat Broadcast.scalarL _ _ t
        (Tile.scalar stride_cn) _ (evalOp_constNat _ _)
        (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcn))))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqCPtrs, cAddr, mdqOffsTile, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

private theorem mdq_cMaskInit_eval (t : BlockState) (M N BM BN pm pn : Nat)
    (hcm : t.regs .nat [BM] "offs_cm" = some (mdqOffsTile (pm * BM) BM))
    (hcn : t.regs .nat [BN] "offs_cn" = some (mdqOffsTile (pn * BN) BN)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        ((Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm"))
          (Op.constNat M) : Op .bool [BM, 1]))
        ((Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))
          (Op.constNat N) : Op .bool [1, BN]))) t
      = some (mdqCMask M N BM BN pm pn) := by
  rw [mdq_boolAnd_eval _ _ _ t _ _
    (mdq_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar M)
      (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcm))
      (evalOp_constNat _ _))
    (mdq_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar N)
      (mdq_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcn))
      (evalOp_constNat _ _))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mdqCMask, mdqOffsTile, Tile.bop_data, Tile.cop_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]

private theorem mdq_store_eq (c : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat) (t : BlockState)
    (vt : Tile .real [BM, BN]) (f : TileIndex [BM, BN] → ℝ)
    (hfv : ∀ i, vt.data i = some (f i))
    (hcp : t.regs .ptr [BM, BN] "c_ptrs"
      = some (mdqCPtrs c stride_cm stride_cn BM BN pm pn))
    (hcmask : t.regs .bool [BM, BN] "c_mask" = some (mdqCMask M N BM BN pm pn))
    (hv : t.regs .real [BM, BN] "accumulator" = some vt) :
    stepStmt (Stmt.store .real [BM, BN]
        (MemAccess.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
        (Op.ref .real [BM, BN] "accumulator")
        (MaskOpt.mask (Op.ref .bool [BM, BN] "c_mask"))) t
      = some (mdqStoreState c M N stride_cm stride_cn BM BN pm pn f t) := by
  unfold stepStmt mdqStoreState
  simp only [evalOp_ref, hv, hcp, hcmask, Option.map_some]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [BM, BN])) ?_)
  funext acc i
  obtain ⟨r, cc, u⟩ := i
  by_cases hb : pm * BM + r.val < M ∧ pn * BN + cc.val < N
  · simp only [mdqCMask, if_pos hb, mdqCPtrs, BlockState.writeMemTyped_real, hfv]
    simp [hb.1, hb.2]
  · simp only [mdqCMask, if_neg hb, mdqCPtrs]
    rw [if_neg]
    intro hcon
    exact hb (by simpa using hcon)

private theorem mdq_store_props (c : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat) (t : BlockState)
    (f : TileIndex [BM, BN] → ℝ)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN pm pn i)) :
    ∀ i : TileIndex [BM, BN],
      (pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N) →
      (mdqStoreState c M N stride_cm stride_cn BM BN pm pn f t).readMem c
          (cAddr stride_cm stride_cn BM BN pm pn i) = f i := by
  classical
  intro i hi
  unfold mdqStoreState
  have h := BlockState.scatter_readback_prop_masked_nd (region := c) t
    (fun j : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN pm pn j) f
    (fun j : TileIndex [BM, BN] => pm * BM + j.1.val < M ∧ pn * BN + j.2.1.val < N)
    hInj i
  rw [h, if_pos hi]

/-- `hInj` for a row-major `C`: distinct output lanes get distinct addresses as soon
as the column stride is positive and one block row fits inside the row stride. -/
theorem cAddr_injective (stride_cm stride_cn BM BN pm pn : Nat)
    (hcn : 0 < stride_cn) (hfit : BN * stride_cn ≤ stride_cm) :
    Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN pm pn i) := by
  intro i j hij
  obtain ⟨r₁, c₁, u₁⟩ := i
  obtain ⟨r₂, c₂, u₂⟩ := j
  simp only [cAddr] at hij
  -- Every product below is opaque to `omega`, so name them first.
  have hexp : ∀ r c : Nat, stride_cm * (pm * BM + r) + stride_cn * (pn * BN + c)
      = stride_cm * (pm * BM) + stride_cm * r
        + (stride_cn * (pn * BN) + stride_cn * c) := by
    intro r c
    rw [Nat.mul_add, Nat.mul_add]
  rw [hexp, hexp] at hij
  have key : stride_cm * r₁.val + stride_cn * c₁.val
      = stride_cm * r₂.val + stride_cn * c₂.val := by omega
  -- A whole block row fits under one row stride, so the row index is forced first.
  have hrow : ∀ c : Fin BN, stride_cn * c.val < stride_cm := fun c =>
    lt_of_lt_of_le
      (by rw [Nat.mul_comm]; exact Nat.mul_lt_mul_of_lt_of_le c.isLt (le_refl _) hcn)
      hfit
  have hr : r₁.val = r₂.val := by
    rcases Nat.lt_trichotomy r₁.val r₂.val with h | h | h
    · have : stride_cm * r₁.val + stride_cm ≤ stride_cm * r₂.val := by
        rw [← Nat.mul_succ]
        exact Nat.mul_le_mul_left _ h
      have := hrow c₁
      omega
    · exact h
    · have : stride_cm * r₂.val + stride_cm ≤ stride_cm * r₁.val := by
        rw [← Nat.mul_succ]
        exact Nat.mul_le_mul_left _ h
      have := hrow c₂
      omega
  have hc : c₁.val = c₂.val := by
    have : stride_cn * c₁.val = stride_cn * c₂.val := by rw [hr] at key; omega
    exact Nat.eq_of_mul_eq_mul_left hcn this
  simp only [Prod.mk.injEq]
  exact ⟨Fin.ext hr, Fin.ext hc, trivial⟩

/-- `(accumulator).to(tl.float16)` — the dead `c` binding's fp16 downcast:
pointwise `FloatDType.cast` (the `attention_fwd_triton2` `p.to(tl.float16)`
recipe). The value is never stored, but the statement is walked faithfully. -/
private theorem mdq_castFp16_eval {sh : TileShape} (nm : RegName) (t : BlockState)
    (vt : Tile .real sh) (h : t.regs .real sh nm = some vt) :
    evalOp (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real sh nm)) t
      = some (⟨fun ix => FloatDType.real.cast FloatDType.fp16 (vt.data ix)⟩
          : Tile .fp16 sh) := by
  rw [evalOp_castFloat]
  simp [evalOp_ref, h]

/-! ### The tail

Six statements: the dead fp16-cast `c` binding, the two output coordinate
vectors, the `C` pointer tile, the two-axis mask, and the store — which writes
`accumulator`, not `c`, exactly as the source does. -/

theorem mdqPostLoop_run (s0 : BlockState) (c a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M N K groupsize stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK GM : Nat) (t : BlockState)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN
        (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM) i))
    (hinv : mdqInv s0 a scales b zeros NO_GROUPS M N K groupsize stride_am stride_ak
      stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
      stride_zeros_n BM BN BK GM (numPidK K BK) t) :
    ∃ sF, stepStmts (mdqPostLoop c M N stride_cm stride_cn BM BN) t = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (pidM s0 M N BM BN GM * BM + idx.1.val < M
            ∧ pidN s0 M N BM BN GM * BN + idx.2.1.val < N) →
          sF.readMem c (cAddr stride_cm stride_cn BM BN (pidM s0 M N BM BN GM)
              (pidN s0 M N BM BN GM) idx)
            = accSpec s0 a scales b zeros NO_GROUPS M K groupsize stride_am stride_ak
                stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
                stride_zeros_n BM BN BK (pidM s0 M N BM BN GM)
                (pidN s0 M N BM BN GM) idx.1.val idx.2.1.val := by
  obtain ⟨-, -, -, -, -, hpm, hpn, -, -, -, -, -, -, -, hacc, -⟩ := hinv
  unfold mdqPostLoop
  -- 1. the dead `c = (accumulator).to(tl.float16)` binding — a genuine fp16
  -- downcast, but of a value the store never reads (it stores `accumulator`)
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some TileDType.fp16 [BM, BN] "c"
    (Op.castFloat FloatDType.real FloatDType.fp16
      (Op.ref .real [BM, BN] "accumulator")) _
    (⟨fun ix => FloatDType.real.cast FloatDType.fp16
        ((mdqAccTile s0 a scales b zeros NO_GROUPS M groupsize stride_am
          stride_ak stride_bk stride_bn stride_scales_g stride_scales_n
          stride_zeros_g stride_zeros_n BM BN BK (pidM s0 M N BM BN GM)
          (pidN s0 M N BM BN GM) (numPidK K BK)).data ix)⟩ : Tile .fp16 [BM, BN])
    (mdq_castFp16_eval "accumulator" t _ hacc))]
  -- 2-3. the output coordinate vectors
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_offs_eval "pid_m" _ BM (pidM s0 M N BM BN GM) BM (by simpa using hpm)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_offs_eval "pid_n" _ BN (pidN s0 M N BM BN GM) BN (by simpa using hpn)))]
  -- 4-5. the `C` pointer tile and the two-axis mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_cPtrsInit_eval c _ stride_cm stride_cn BM BN (pidM s0 M N BM BN GM)
      (pidN s0 M N BM BN GM) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mdq_cMaskInit_eval _ M N BM BN (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM)
      (by simp) (by simp)))]
  -- 6. the store
  rw [stepStmts.cons_some
    (mdq_store_eq c M N stride_cm stride_cn BM BN (pidM s0 M N BM BN GM)
      (pidN s0 M N BM BN GM) _ _
      (fun idx => accSpec s0 a scales b zeros NO_GROUPS M K groupsize stride_am
        stride_ak stride_bk stride_bn stride_scales_g stride_scales_n stride_zeros_g
        stride_zeros_n BM BN BK (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM)
        idx.1.val idx.2.1.val)
      (fun idx => mdqAccTile_full s0 a scales b zeros NO_GROUPS M K groupsize
        stride_am stride_ak stride_bk stride_bn stride_scales_g stride_scales_n
        stride_zeros_g stride_zeros_n BM BN BK (pidM s0 M N BM BN GM)
        (pidN s0 M N BM BN GM) idx)
      (by simp) (by simp) (by simp [hacc]))]
  rw [stepStmts.nil]
  exact ⟨_, rfl, mdq_store_props c M N stride_cm stride_cn BM BN _ _ _ _ hInj⟩

/-! ## Main theorem -/

set_option maxHeartbeats 1000000 in
/-- **Genuine, dimension-general correctness.** For every launch state, the kernel
runs to completion and every in-range output lane of `C` holds `accSpec`: the sum
over all `num_pid_k` K steps of `tl.dot` between the row-masked `A` tile and the
dequantized `B` tile, with `scales` / `zeros` read at that step's group row.

`hInj` says distinct output lanes get distinct `C` addresses — without it the
readback could not name a lane. `cAddr_injective` discharges it for a row-major
`C`. The `NO_GROUPS` flag is free: both configurations are covered by the same
statement, since `groupRow` is what differs and `accSpec` takes it into account. -/
specification matmul_dequantize_matmul4_exec_genuine
    (a_ptr c_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool) (BM BN BK GM : Nat) (s : BlockState)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN
        (pidM s M N BM BN GM) (pidN s M N BM BN GM) i)) :
    ∃ sF, exec (matmul_dequantize_matmul4_surface a_ptr c_ptr scales_ptr b_ptr zeros_ptr
        M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
        groupsize NO_GROUPS BM BN BK GM).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (pidM s M N BM BN GM * BM + idx.1.val < M
            ∧ pidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.readMem c_ptr (cAddr stride_cm stride_cn BM BN (pidM s M N BM BN GM)
              (pidN s M N BM BN GM) idx)
            = accSpec s a_ptr scales_ptr b_ptr zeros_ptr NO_GROUPS M K groupsize
                stride_am stride_ak stride_bk stride_bn stride_scales_g
                stride_scales_n stride_zeros_g stride_zeros_n BM BN BK
                (pidM s M N BM BN GM) (pidN s M N BM BN GM)
                idx.1.val idx.2.1.val := by
  rw [exec, mdq_body_eq]
  -- prologue: the scalars, then the tiles
  obtain ⟨t1, hrun1, h1mem, h1pids, h1bits, h1ipb, h1npk, h1pm, h1pn⟩ :=
    mdqPreLoopScalars_run s M N K BM BN BK GM
  obtain ⟨t2, hrun2, h2npk, h2inv⟩ :=
    mdqPreLoopTiles_run s a_ptr scales_ptr b_ptr zeros_ptr NO_GROUPS M N K groupsize
      stride_am stride_ak stride_bk stride_bn stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BM BN BK GM t1 h1mem h1pids h1bits h1ipb h1pm h1pn
  simp only [mdqPreLoop, List.append_assoc]
  rw [stepStmts.append_some hrun1, stepStmts.append_some hrun2]
  -- the collapsed K loop
  obtain ⟨t3, hrun3, h3inv⟩ :=
    mdqLoop_collapse s a_ptr scales_ptr b_ptr zeros_ptr NO_GROUPS M N K groupsize
      stride_am stride_ak stride_bk stride_bn stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BM BN BK GM t2 (h2npk.trans h1npk) h2inv
  rw [show [Stmt.forRangeDyn "k" (Op.constNat 0) (Op.ref .nat [] "num_pid_k")
          (Op.constNat 1)
          (mdqLoopBody groupsize stride_ak stride_bk stride_scales_g
            stride_zeros_g NO_GROUPS BM BN BK)]
        ++ mdqPostLoop c_ptr M N stride_cm stride_cn BM BN
      = Stmt.forRangeDyn "k" (Op.constNat 0) (Op.ref .nat [] "num_pid_k")
          (Op.constNat 1)
          (mdqLoopBody groupsize stride_ak stride_bk stride_scales_g
            stride_zeros_g NO_GROUPS BM BN BK)
        :: mdqPostLoop c_ptr M N stride_cm stride_cn BM BN from rfl]
  rw [stepStmts.cons_some hrun3]
  -- the tail
  obtain ⟨sF, hpost, hout⟩ :=
    mdqPostLoop_run s c_ptr a_ptr scales_ptr b_ptr zeros_ptr NO_GROUPS M N K groupsize
      stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_scales_g
      stride_scales_n stride_zeros_g stride_zeros_n BM BN BK GM t3 hInj h3inv
  exact ⟨sF, hpost, hout⟩


/-! # ════════ Kernel 2 of 3: `matmul_kernel` ════════ -/

/-! ## Kernel surface (faithful transcription, `SPLIT_K = 1` arm) -/

def matmul_dequantize_matmul_surface
    (a_ptr c_ptr bs_ptr : RegionName) (b_ptr bzp_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_bsk stride_bsn stride_bzpk stride_bzpn
      group_size : Nat)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  pid_sp_k = tl.program_id(axis=1)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_k = tl.cdiv($(K), $(BLOCK_SIZE_K))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = pid_sp_k * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = a_ptr + (offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak))
  b_ptrs = b_ptr + ((offs_k[:, None] // $(8)) * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), $(numKBlocks), $(1)) {
    bs_ptrs = bs_ptr + (((offs_k[:, None] + k * $(BLOCK_SIZE_K)) // $(group_size)) * $(stride_bsk) + offs_bn[None, :] * $(stride_bsn))
    bzp_ptrs = bzp_ptr + (((offs_k[:, None] + k * $(BLOCK_SIZE_K)) // $(group_size)) * $(stride_bzpk) + (offs_bn[None, :] // $(8)) * $(stride_bzpn))
    b_shift_bits = (offs_k[:, None] % $(8)) * $(4)
    bzp_shift_bits = (offs_bn[None, :] % $(8)) * $(4)
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    bs = tl.load(bs_ptrs)
    bzp = tl.load(bzp_ptrs)
    int_b = (b >> b_shift_bits) & $(15)
    int_bzp = (bzp >> bzp_shift_bits) & $(15)
    b = ((tl.cast(int_b, tl.int32) - tl.cast(int_bzp, tl.int32)) * bs).to(tl.float16)
    accumulator += tl.dot((a).to(tl.float16), (b).to(tl.float16))
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += ($(BLOCK_SIZE_K) * $(stride_bk) // $(8))
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = c_ptr + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}

theorem int4_matmul_surface_toAlgorithm_supported
    (a_ptr c_ptr bs_ptr : RegionName) (b_ptr bzp_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_bsk stride_bsn stride_bzpk stride_bzpn
      group_size : Nat)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ∃ alg, (matmul_dequantize_matmul_surface a_ptr c_ptr bs_ptr b_ptr bzp_ptr
      M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_bsk stride_bsn stride_bzpk stride_bzpn group_size
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks).toAlgorithm?
        = Except.ok alg := by
  simp [matmul_dequantize_matmul_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Block coordinates and the `C` address — shared with `matmul4_kernel`

`matmul_kernel` decomposes `pid` with exactly the same group swizzle as
`matmul4_kernel` and addresses `C` with exactly the same
`stride_cm * row + stride_cn * col` map, so this section reuses `numPidM` …
`pidN`, `cAddr` and `cAddr_injective` from the `matmul4_kernel` section above.
(`num_pid_k` is a **dead binding** here — this kernel's loop bound is
`numKBlocks`, computed inline in the source — but it is still transcribed and
walked.) -/

/-! ## Element accessors

Each is the kernel's own address arithmetic over the **launch** state's
memory, parameterized by the *absolute* row / column number, so that the
wrapped (`% M` / `% N`) and unwrapped readings share one definition — the
invariant instantiates them at the wrapped index, the headline (whose store
mask guarantees `row < M`, `col < N`) at the plain one. -/

/-- `a[row, k*BK + e]` at K step `k`. Unmasked, matching the source's bare
`tl.load(a_ptrs)`. -/
noncomputable def i4AElem (s : BlockState) (a : RegionName)
    (stride_am stride_ak BK : Nat) (row k e : Nat) : ℝ :=
  s.readMem a (row * stride_am + (e + k * BK) * stride_ak)

/-- The packed 32-bit word holding the weight nibble for `(k, e, col)`. Eight
weights share a word along K, hence the `e / 8`. -/
def i4BWord (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BK : Nat) (k e col : Nat) : Nat :=
  s.readMemValue .nat b ((e / 8 + k * (BK / 8)) * stride_bk + col * stride_bn)

/-- The unpacked 4-bit weight: `(word >> (e % 8) * 4) & 0xF`. -/
def i4BNibble (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BK : Nat) (k e col : Nat) : Nat :=
  i4BWord s b stride_bk stride_bn BK k e col >>> (e % 8 * 4) &&& 15

/-- `bs[(e + k*BK) // group_size, col]` — the scale at lane `(k, e, col)`'s
group row. Per-lane: the group row varies **within** the `[BK, BN]` tile
whenever `group_size < BLOCK_SIZE_K`. -/
noncomputable def bsElem (s : BlockState) (bs : RegionName)
    (group_size stride_bsk stride_bsn BK : Nat) (k e col : Nat) : ℝ :=
  s.readMem bs ((e + k * BK) / group_size * stride_bsk + col * stride_bsn)

/-- The packed word holding the zero-point nibble for `(k, e, col)`. Eight
zero-points share a word along N, hence the `col / 8`. -/
def bzpWord (s : BlockState) (bzp : Region .nat)
    (group_size stride_bzpk stride_bzpn BK : Nat) (k e col : Nat) : Nat :=
  s.readMemValue .nat bzp
    ((e + k * BK) / group_size * stride_bzpk + col / 8 * stride_bzpn)

/-- The unpacked 4-bit zero-point: `(word >> (col % 8) * 4) & 0xF`. -/
def bzpNibble (s : BlockState) (bzp : Region .nat)
    (group_size stride_bzpk stride_bzpn BK : Nat) (k e col : Nat) : Nat :=
  bzpWord s bzp group_size stride_bzpk stride_bzpn BK k e col
    >>> (col % 8 * 4) &&& 15

/-- The dequantized weight at `(k, e, col)`: the **signed** nibble difference
(`ℤ`-subtraction — this is what the `.int` hop buys; ℕ subtraction would
truncate), embedded into ℝ and scaled. -/
noncomputable def i4BDequant (s : BlockState) (bs : RegionName)
    (b bzp : Region .nat)
    (group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BK : Nat) (k e col : Nat) : ℝ :=
  (((i4BNibble s b stride_bk stride_bn BK k e col : ℤ)
      - (bzpNibble s bzp group_size stride_bzpk stride_bzpn BK k e col : ℤ) : ℤ) : ℝ)
    * bsElem s bs group_size stride_bsk stride_bsn BK k e col

/-! ## The accumulator -/

/-- One K step's contribution to output cell `(row, col)`. -/
noncomputable def i4AccStep (s : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (group_size stride_am stride_ak stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BK : Nat) (row col k : Nat) : ℝ :=
  ∑ e : Fin BK,
    i4AElem s a stride_am stride_ak BK row k e.val
      * i4BDequant s bs b bzp group_size stride_bk stride_bn stride_bsk stride_bsn
          stride_bzpk stride_bzpn BK k e.val col

/-- **The stored value.** `accumulator` after all `numKBlocks` K steps (the
kernel stores `c = accumulator.to(...)`, which erases to `accumulator`). -/
noncomputable def i4AccSpec (s : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (group_size stride_am stride_ak stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BK numKBlocks : Nat) (row col : Nat) : ℝ :=
  ∑ k : Fin numKBlocks,
    i4AccStep s a bs b bzp group_size stride_am stride_ak stride_bk stride_bn
      stride_bsk stride_bsn stride_bzpk stride_bzpn BK row col k.val

/-! ## Compiled body decomposition

The algorithm-lowered statement lists, checked against the macro output by
`rfl`. Lowerings worth naming because they are not guessable from the source
text:

* `//` / `%` lower to `Op.floorDiv` / `Op.mod` on `IntegralDType.nat`;
  `tl.cdiv` expands to `Op.div .nat`;
* `min(a, b)` is `Op.where (Op.lt …) a b` (there is no `Op.min`);
* `b_shift_bits` / `bzp_shift_bits` are **`[BK, 1]` / `[1, BN]` registers**
  (the source builds them with `[:, None]` / `[None, :]` at assignment time),
  and the shifts broadcast them with `Broadcast.consR` on the unit axis;
* the dequant statement is `Op.mul .real` over `Op.intToReal` of an
  `Op.sub .int` of two `Op.castNatToInt`s — the signed-promotion chain;
* the `b_ptrs` advance offset is `(BK * stride_bk) // 8` — the **whole
  product** divided (source parenthesization), not `(BK // 8) * stride_bk`;
  the two agree exactly because `8 ∣ BK` (the headline's `hBK8`);
* every `.to(tl.float16)` lowers to a genuine `Op.castFloat` quantization
  event: the dequant statement assigns `b` on the **`.fp16` register channel**,
  the dot's `(a).to(tl.float16)` / `(b).to(tl.float16)` are wrapped by the
  DSL's `expandDot` in a re-widening `Op.castFloat fp16 → real` (algorithm-layer
  `tl.dot` is real-valued), and `c = (accumulator).to(tl.float16)` makes the
  terminal store **`.fp16`-typed** — the `matmul_tma` f16 lowering, on `.ptr`
  stores. -/

/-- The prologue's eleven scalar statements: both program ids, the three
`tl.cdiv` trip counts (`num_pid_k` dead) and the group swizzle down to
`(pid_m, pid_n)`. -/
def i4PreLoopScalars (M N K BM BN BK GM : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid" (Op.programId 0),
    Stmt.assign .nat [] "pid_sp_k" (Op.programId 1),
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
        (Op.ref .nat [] "group_size_m")) ]

/-- The prologue's six index statements: the two **wrapped** offset vectors,
`offs_k` (built from `pid_sp_k`), the two pointer tiles and the zeroed
accumulator. -/
def i4PreLoopIndex (a_ptr : RegionName) (b_ptr : Region .nat)
    (M N stride_am stride_ak stride_bk stride_bn : Nat)
    (BM BN BK : Nat) : List Stmt :=
  [ Stmt.assign .nat [BM] "offs_am"
      (Op.mod IntegralDType.nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
          (Op.arange BM))
        (Op.constNat M)),
    Stmt.assign .nat [BN] "offs_bn"
      (Op.mod IntegralDType.nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
          (Op.arange BN))
        (Op.constNat N)),
    Stmt.assign .nat [BK] "offs_k"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_sp_k") (Op.constNat BK))
        (Op.arange BK)),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase a_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
            (Op.constNat stride_am))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_ak)))),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase b_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.floorDiv IntegralDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
              (Op.constNat 8))
            (Op.constNat stride_bk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
            (Op.constNat stride_bn)))),
    Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)) ]

/-- The K-loop body's first four statements: the per-step **recomputed**
`bs_ptrs` / `bzp_ptrs` (group row `(offs_k + k*BK) // group_size`, a runtime
scalar divisor) and the two shifter tiles. Split off so the main body walk
starts from an opaque state instead of pushing a four-deeper `setReg` tower
through every later read. -/
def i4LoopPtrs (bs_ptr : RegionName) (bzp_ptr : Region .nat)
    (group_size stride_bsk stride_bsn stride_bzpk stride_bzpn BN BK : Nat) :
    List Stmt :=
  [ Stmt.assign .ptr [BK, BN] "bs_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase bs_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.floorDiv IntegralDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK)))
              (Op.constNat group_size))
            (Op.constNat stride_bsk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
            (Op.constNat stride_bsn)))),
    Stmt.assign .ptr [BK, BN] "bzp_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase bzp_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.floorDiv IntegralDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK)))
              (Op.constNat group_size))
            (Op.constNat stride_bzpk))
          (Op.mul .nat Broadcast.scalarR
            (Op.floorDiv IntegralDType.nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
              (Op.constNat 8))
            (Op.constNat stride_bzpn)))),
    Stmt.assign .nat [BK, 1] "b_shift_bits"
      (Op.mul .nat Broadcast.scalarR
        (Op.mod IntegralDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
          (Op.constNat 8))
        (Op.constNat 4)),
    Stmt.assign .nat [1, BN] "bzp_shift_bits"
      (Op.mul .nat Broadcast.scalarR
        (Op.mod IntegralDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
          (Op.constNat 8))
        (Op.constNat 4)) ]

/-- The K-loop body's four unmasked loads. -/
def i4LoopLoads (BM BN BK : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BK] "a"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        MaskOpt.none),
    Stmt.assign .nat [BK, BN] "b"
      (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BK, BN] "b_ptrs"))
        MaskOpt.none),
    Stmt.assign .real [BK, BN] "bs"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK, BN] "bs_ptrs"))
        MaskOpt.none),
    Stmt.assign .nat [BK, BN] "bzp"
      (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BK, BN] "bzp_ptrs"))
        MaskOpt.none) ]

/-- The K-loop body's compute tail: the two nibble extractions, the signed
dequant **downcast to the `.fp16` channel** (`.to(tl.float16)` compiles to a
genuine `Op.castFloat`), the `tl.dot` accumulation — whose operands carry the
source's `(a).to(tl.float16)` / `(b).to(tl.float16)` casts, each re-widened
`fp16 → real` by the DSL's real-valued `tl.dot` — and the two pointer
advances. -/
def i4LoopCompute (stride_ak stride_bk BM BN BK : Nat) : List Stmt :=
  [ Stmt.assign .nat [BK, BN] "int_b"
          (Op.bitAnd Broadcast.scalarR
            (Op.shiftRight (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (Op.ref .nat [BK, BN] "b")
              (Op.ref .nat [BK, 1] "b_shift_bits"))
            (Op.constNat 15)),
        Stmt.assign .nat [BK, BN] "int_bzp"
          (Op.bitAnd Broadcast.scalarR
            (Op.shiftRight (Broadcast.consR (Broadcast.consSame Broadcast.nil))
              (Op.ref .nat [BK, BN] "bzp")
              (Op.ref .nat [1, BN] "bzp_shift_bits"))
            (Op.constNat 15)),
        Stmt.assign .fp16 [BK, BN] "b"
          (Op.castFloat FloatDType.real FloatDType.fp16
            (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Op.intToReal
                (Op.sub .int (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                  (Op.castNatToInt (Op.ref .nat [BK, BN] "int_b"))
                  (Op.castNatToInt (Op.ref .nat [BK, BN] "int_bzp"))))
              (Op.ref .real [BK, BN] "bs"))),
        Stmt.assign .real [BM, BN] "accumulator"
          (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BN] "accumulator")
            (Op.dot (batch := [])
              (Op.castFloat FloatDType.fp16 FloatDType.real
                (Op.castFloat FloatDType.real FloatDType.fp16
                  (Op.ref .real [BM, BK] "a")))
              (Op.castFloat FloatDType.fp16 FloatDType.real
                (Op.castFloat FloatDType.fp16 FloatDType.fp16
                  (Op.ref .fp16 [BK, BN] "b"))))),
        Stmt.assign .ptr [BM, BK] "a_ptrs"
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "a_ptrs")
            (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_ak))),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "b_ptrs")
        (Op.floorDiv IntegralDType.nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_bk))
          (Op.constNat 8))) ]

/-- The full compiled K-loop body: recomputes ++ loads ++ compute tail. -/
def i4LoopBody (bs_ptr : RegionName) (bzp_ptr : Region .nat)
    (group_size stride_ak stride_bk stride_bsk stride_bsn
      stride_bzpk stride_bzpn BM BN BK : Nat) : List Stmt :=
  i4LoopPtrs bs_ptr bzp_ptr group_size stride_bsk stride_bsn stride_bzpk
      stride_bzpn BN BK
    ++ i4LoopLoads BM BN BK
    ++ i4LoopCompute stride_ak stride_bk BM BN BK

/-- The compiled tail: the (stored!) `c = (accumulator).to(tl.float16)`
binding — a genuine `Op.castFloat` fp16 downcast — the two **unwrapped**
output coordinate vectors, the `C` pointer tile, the two-axis store mask, and
the masked **`.fp16`-typed** store of `c`. -/
def i4PostLoop (c_ptr : RegionName) (M N stride_cm stride_cn BM BN : Nat) :
    List Stmt :=
  [ Stmt.assign .fp16 [BM, BN] "c"
      (Op.castFloat FloatDType.real FloatDType.fp16
        (Op.ref .real [BM, BN] "accumulator")),
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
    Stmt.store .fp16 [BM, BN] (MemAccess.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
      (Op.ref .fp16 [BM, BN] "c")
      (MaskOpt.mask (Op.ref .bool [BM, BN] "c_mask")) ]

set_option maxRecDepth 20000 in
/-- **Full body split (by `rfl`).** The lowered surface is exactly
`i4PreLoopScalars ++ i4PreLoopIndex ++ [forRange "k" 0 numKBlocks 1 i4LoopBody]
++ i4PostLoop` — 24 top-level statements, every one checked against the macro
output. -/
theorem i4_body_eq (a_ptr c_ptr bs_ptr : RegionName) (b_ptr bzp_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_bsk stride_bsn stride_bzpk stride_bzpn group_size : Nat)
    (BM BN BK GM numKBlocks : Nat) :
    (matmul_dequantize_matmul_surface a_ptr c_ptr bs_ptr b_ptr bzp_ptr
        M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        stride_bsk stride_bsn stride_bzpk stride_bzpn group_size
        BM BN BK GM numKBlocks).toAlgKernel.body
      = i4PreLoopScalars M N K BM BN BK GM
        ++ i4PreLoopIndex a_ptr b_ptr M N stride_am stride_ak stride_bk stride_bn
            BM BN BK
        ++ [Stmt.forRange "k" 0 numKBlocks 1
              (i4LoopBody bs_ptr bzp_ptr group_size stride_ak stride_bk
                stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK)]
        ++ i4PostLoop c_ptr M N stride_cm stride_cn BM BN := by
  rfl

/-! ## Offset and pointer tiles

A `.ptr` tile carries one `(region, offset)` pair per lane, and a `.ptr` load
is unconditionally in bounds — there is no mask on any load in this kernel.
`a_ptrs` / `b_ptrs` are advanced once per K step; `bs_ptrs` / `bzp_ptrs` are
**rebuilt** from `k` each iteration, so they never enter the invariant. -/

/-- `pid_* * BLOCK + tl.arange(0, BLOCK)` (and `tl.arange` alone at base 0). -/
def i4Offs (base BD : Nat) : Tile .nat [BD] := ⟨fun idx => base + idx.1.val⟩

/-- The **wrapped** offset vector `(base + e) % Mm` — `offs_am` / `offs_bn`. -/
def i4WrapOffs (base BD Mm : Nat) : Tile .nat [BD] :=
  ⟨fun idx => (base + idx.1.val) % Mm⟩

/-- `a_ptrs` lane `(r, e)` at K step `k`, in the accumulated form the kernel
reaches: the wrapped row offset plus `k` advances of `BK * stride_ak`. -/
def i4AAddr (M stride_am stride_ak BM BK pm k : Nat)
    (idx : TileIndex [BM, BK]) : Nat :=
  (pm * BM + idx.1.val) % M * stride_am + idx.2.1.val * stride_ak
    + k * (BK * stride_ak)

/-- `b_ptrs` lane `(e, c)` at K step `k`. The advance is the kernel's literal
`(BK * stride_bk) // 8` — the identification with `i4BWord`'s
`(e/8 + k*(BK/8)) * stride_bk` happens in `i4BAddr_eq`, under `8 ∣ BK`. -/
def i4BAddr (N stride_bk stride_bn BN BK pn k : Nat)
    (idx : TileIndex [BK, BN]) : Nat :=
  idx.1.val / 8 * stride_bk + (pn * BN + idx.2.1.val) % N * stride_bn
    + k * (BK * stride_bk / 8)

noncomputable def i4APtrs (a : RegionName)
    (M stride_am stride_ak BM BK pm k : Nat) : Tile .ptr [BM, BK] :=
  ⟨fun idx => (a, i4AAddr M stride_am stride_ak BM BK pm k idx)⟩

noncomputable def i4BPtrs (b : Region .nat)
    (N stride_bk stride_bn BN BK pn k : Nat) : Tile .ptr [BK, BN] :=
  ⟨fun idx => (Region.cast b, i4BAddr N stride_bk stride_bn BN BK pn k idx)⟩

/-- One `a_ptrs += BLOCK_SIZE_K * stride_ak` advance. -/
theorem i4APtrs_succ (a : RegionName) (M stride_am stride_ak BM BK pm k : Nat) :
    Tile.ptrAdd Broadcast.scalarR (i4APtrs a M stride_am stride_ak BM BK pm k)
        (Tile.scalar (BK * stride_ak))
      = i4APtrs a M stride_am stride_ak BM BK pm (k + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, i4APtrs, i4AAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

/-- One `b_ptrs += (BLOCK_SIZE_K * stride_bk // 8)` advance. Pure `ring` —
no divisibility needed, because the address carries the advance verbatim. -/
theorem i4BPtrs_succ (b : Region .nat) (N stride_bk stride_bn BN BK pn k : Nat) :
    Tile.ptrAdd Broadcast.scalarR (i4BPtrs b N stride_bk stride_bn BN BK pn k)
        (Tile.scalar (BK * stride_bk / 8))
      = i4BPtrs b N stride_bk stride_bn BN BK pn (k + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, i4BPtrs, i4BAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

/-- The `a_ptrs` address agrees with `i4AElem`'s at the wrapped row. -/
theorem i4AAddr_eq (M stride_am stride_ak BM BK pm k : Nat)
    (idx : TileIndex [BM, BK]) :
    i4AAddr M stride_am stride_ak BM BK pm k idx
      = (pm * BM + idx.1.val) % M * stride_am
        + (idx.2.1.val + k * BK) * stride_ak := by
  simp only [i4AAddr]
  ring

/-- The `b_ptrs` address agrees with `i4BWord`'s — this is where `8 ∣ BK`
(the source's `assert BLOCK_SIZE_K % 8 == 0`) earns its keep: it turns the
kernel's `k * ((BK * stride_bk) / 8)` into `k * (BK / 8) * stride_bk`. -/
theorem i4BAddr_eq (N stride_bk stride_bn BN BK pn k : Nat)
    (hBK8 : BK % 8 = 0) (idx : TileIndex [BK, BN]) :
    i4BAddr N stride_bk stride_bn BN BK pn k idx
      = (idx.1.val / 8 + k * (BK / 8)) * stride_bk
        + (pn * BN + idx.2.1.val) % N * stride_bn := by
  obtain ⟨q, hq⟩ := Nat.dvd_of_mod_eq_zero hBK8
  subst hq
  simp only [i4BAddr]
  rw [Nat.mul_assoc 8 q stride_bk, Nat.mul_div_cancel_left _ (by norm_num),
    Nat.mul_div_cancel_left _ (by norm_num : (0:Nat) < 8)]
  ring

/-- `bs_ptrs` lane `(e, c)` at K step `k`: per-lane group row
`(e + k*BK) // group_size`, wrapped column. -/
def i4BsAddr (N group_size stride_bsk stride_bsn BN BK pn k : Nat)
    (idx : TileIndex [BK, BN]) : Nat :=
  (idx.1.val + k * BK) / group_size * stride_bsk
    + (pn * BN + idx.2.1.val) % N * stride_bsn

/-- `bzp_ptrs` lane `(e, c)` at K step `k` — eight zero-points share a word
along N, hence the `/ 8` on the wrapped column. -/
def i4BzpAddr (N group_size stride_bzpk stride_bzpn BN BK pn k : Nat)
    (idx : TileIndex [BK, BN]) : Nat :=
  (idx.1.val + k * BK) / group_size * stride_bzpk
    + (pn * BN + idx.2.1.val) % N / 8 * stride_bzpn

noncomputable def i4BsPtrs (bs : RegionName)
    (N group_size stride_bsk stride_bsn BN BK pn k : Nat) : Tile .ptr [BK, BN] :=
  ⟨fun idx => (bs, i4BsAddr N group_size stride_bsk stride_bsn BN BK pn k idx)⟩

noncomputable def i4BzpPtrs (bzp : Region .nat)
    (N group_size stride_bzpk stride_bzpn BN BK pn k : Nat) : Tile .ptr [BK, BN] :=
  ⟨fun idx => (Region.cast bzp,
    i4BzpAddr N group_size stride_bzpk stride_bzpn BN BK pn k idx)⟩

/-! ## The loop body's value tiles -/

/-- `b_shift_bits = (offs_k[:, None] % 8) * 4` — a `[BK, 1]` tile. -/
def i4BShift (BK : Nat) : Tile .nat [BK, 1] := ⟨fun idx => idx.1.val % 8 * 4⟩

/-- `bzp_shift_bits = (offs_bn[None, :] % 8) * 4` — a `[1, BN]` tile over the
wrapped column. -/
def i4BzpShift (N BN pn : Nat) : Tile .nat [1, BN] :=
  ⟨fun idx => (pn * BN + idx.2.1.val) % N % 8 * 4⟩

/-- `a` at K step `k` — every lane reads memory (unmasked load). -/
noncomputable def i4ATile (s : BlockState) (a : RegionName)
    (M stride_am stride_ak BM BK pm k : Nat) : Tile .real [BM, BK] :=
  ⟨fun idx => some (s.readMem a (i4AAddr M stride_am stride_ak BM BK pm k idx))⟩

/-- The loaded packed weight words at K step `k`. -/
noncomputable def i4BRaw (s : BlockState) (b : Region .nat)
    (N stride_bk stride_bn BN BK pn k : Nat) : Tile .nat [BK, BN] :=
  ⟨fun idx => s.readMemValue .nat b (i4BAddr N stride_bk stride_bn BN BK pn k idx)⟩

/-- The loaded scales at K step `k`. -/
noncomputable def i4BsTile (s : BlockState) (bs : RegionName)
    (N group_size stride_bsk stride_bsn BN BK pn k : Nat) : Tile .real [BK, BN] :=
  ⟨fun idx => some (s.readMem bs
    (i4BsAddr N group_size stride_bsk stride_bsn BN BK pn k idx))⟩

/-- The loaded packed zero-point words at K step `k`. -/
noncomputable def i4BzpRaw (s : BlockState) (bzp : Region .nat)
    (N group_size stride_bzpk stride_bzpn BN BK pn k : Nat) : Tile .nat [BK, BN] :=
  ⟨fun idx => s.readMemValue .nat bzp
    (i4BzpAddr N group_size stride_bzpk stride_bzpn BN BK pn k idx)⟩

/-- `int_b` — the unpacked weight nibbles, at the wrapped column. -/
noncomputable def i4IntB (s : BlockState) (b : Region .nat)
    (N stride_bk stride_bn BN BK pn k : Nat) : Tile .nat [BK, BN] :=
  ⟨fun idx => i4BNibble s b stride_bk stride_bn BK k idx.1.val
    ((pn * BN + idx.2.1.val) % N)⟩

/-- `int_bzp` — the unpacked zero-point nibbles, at the wrapped column. -/
noncomputable def i4IntBzp (s : BlockState) (bzp : Region .nat)
    (N group_size stride_bzpk stride_bzpn BN BK pn k : Nat) : Tile .nat [BK, BN] :=
  ⟨fun idx => bzpNibble s bzp group_size stride_bzpk stride_bzpn BK k idx.1.val
    ((pn * BN + idx.2.1.val) % N)⟩

/-- What `Op.castNatToInt` evaluates to, as a named combinator. -/
def i4NatToInt {sh : TileShape} (t : Tile .nat sh) : Tile .int sh :=
  ⟨fun idx => Int.ofNat (t.data idx)⟩

/-- `b` after the signed dequant at K step `k`. -/
noncomputable def i4BDequantTile (s : BlockState) (bs : RegionName)
    (b bzp : Region .nat)
    (N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BN BK pn k : Nat) : Tile .real [BK, BN] :=
  ⟨fun idx => some (i4BDequant s bs b bzp group_size stride_bk stride_bn
    stride_bsk stride_bsn stride_bzpk stride_bzpn BK k idx.1.val
    ((pn * BN + idx.2.1.val) % N))⟩

/-- `accumulator` after `i` K steps: the partial sum the invariant carries,
at the **wrapped** row / column of each lane. -/
noncomputable def i4AccTile (s : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (M N group_size stride_am stride_ak stride_bk stride_bn stride_bsk
      stride_bsn stride_bzpk stride_bzpn BM BN BK pm pn : Nat) (i : Nat) :
    Tile .real [BM, BN] :=
  ⟨fun idx => some (∑ j : Fin i,
      i4AccStep s a bs b bzp group_size stride_am stride_ak stride_bk stride_bn
        stride_bsk stride_bsn stride_bzpk stride_bzpn BK
        ((pm * BM + idx.1.val) % M) ((pn * BN + idx.2.1.val) % N) j.val)⟩

/-- At `i = 0` the accumulator is the zero tile `tl.zeros` produces. -/
theorem i4AccTile_zero (s : BlockState) (a bs : RegionName) (b bzp : Region .nat)
    (M N group_size stride_am stride_ak stride_bk stride_bn stride_bsk
      stride_bsn stride_bzpk stride_bzpn BM BN BK pm pn : Nat) :
    i4AccTile s a bs b bzp M N group_size stride_am stride_ak stride_bk
        stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK pm pn 0
      = (⟨fun _ => some 0⟩ : Tile .real [BM, BN]) := by
  apply Tile.ext
  intro idx
  simp [i4AccTile]

/-- At `i = numKBlocks` the accumulator is `i4AccSpec` (at the wrapped lane). -/
theorem i4AccTile_full (s : BlockState) (a bs : RegionName) (b bzp : Region .nat)
    (M N group_size stride_am stride_ak stride_bk stride_bn stride_bsk
      stride_bsn stride_bzpk stride_bzpn BM BN BK pm pn numKBlocks : Nat)
    (idx : TileIndex [BM, BN]) :
    (i4AccTile s a bs b bzp M N group_size stride_am stride_ak stride_bk
        stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK pm pn
        numKBlocks).data idx
      = some (i4AccSpec s a bs b bzp group_size stride_am stride_ak stride_bk
          stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BK numKBlocks
          ((pm * BM + idx.1.val) % M) ((pn * BN + idx.2.1.val) % N)) := by
  rfl

/-- **The invariant's accumulator step.** -/
theorem i4AccTile_succ (s : BlockState) (a bs : RegionName) (b bzp : Region .nat)
    (M N group_size stride_am stride_ak stride_bk stride_bn stride_bsk
      stride_bsn stride_bzpk stride_bzpn BM BN BK pm pn i : Nat)
    (idx : TileIndex [BM, BN]) :
    (i4AccTile s a bs b bzp M N group_size stride_am stride_ak stride_bk
        stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK pm pn
        (i + 1)).data idx
      = some ((∑ j : Fin i,
            i4AccStep s a bs b bzp group_size stride_am stride_ak stride_bk
              stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BK
              ((pm * BM + idx.1.val) % M) ((pn * BN + idx.2.1.val) % N) j.val)
          + i4AccStep s a bs b bzp group_size stride_am stride_ak stride_bk
              stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BK
              ((pm * BM + idx.1.val) % M) ((pn * BN + idx.2.1.val) % N) i) := by
  simp only [i4AccTile]
  exact congrArg some (Fin.sum_univ_castSucc _)

/-! ### Bridges from the tiles to the spec accessors -/

theorem i4ATile_data (s : BlockState) (a : RegionName)
    (M stride_am stride_ak BM BK pm k : Nat) (idx : TileIndex [BM, BK]) :
    (i4ATile s a M stride_am stride_ak BM BK pm k).data idx
      = some (i4AElem s a stride_am stride_ak BK ((pm * BM + idx.1.val) % M) k
          idx.2.1.val) := by
  simp only [i4ATile, i4AElem, i4AAddr_eq]

/-- The raw-word tile reads `i4BWord`'s address — under `8 ∣ BK`. -/
theorem i4BRaw_data (s : BlockState) (b : Region .nat)
    (N stride_bk stride_bn BN BK pn k : Nat) (hBK8 : BK % 8 = 0)
    (idx : TileIndex [BK, BN]) :
    (i4BRaw s b N stride_bk stride_bn BN BK pn k).data idx
      = i4BWord s b stride_bk stride_bn BK k idx.1.val
          ((pn * BN + idx.2.1.val) % N) := by
  simp only [i4BRaw, i4BWord, i4BAddr_eq N stride_bk stride_bn BN BK pn k hBK8]

theorem i4BsTile_data (s : BlockState) (bs : RegionName)
    (N group_size stride_bsk stride_bsn BN BK pn k : Nat)
    (idx : TileIndex [BK, BN]) :
    (i4BsTile s bs N group_size stride_bsk stride_bsn BN BK pn k).data idx
      = some (bsElem s bs group_size stride_bsk stride_bsn BK k idx.1.val
          ((pn * BN + idx.2.1.val) % N)) := by
  rfl

theorem i4BzpRaw_data (s : BlockState) (bzp : Region .nat)
    (N group_size stride_bzpk stride_bzpn BN BK pn k : Nat)
    (idx : TileIndex [BK, BN]) :
    (i4BzpRaw s bzp N group_size stride_bzpk stride_bzpn BN BK pn k).data idx
      = bzpWord s bzp group_size stride_bzpk stride_bzpn BK k idx.1.val
          ((pn * BN + idx.2.1.val) % N) := by
  rfl

/-- The weight-nibble tile **is** the compiled `(b >> b_shift_bits) & 0xF`
applied to the raw words. The shift's `[BK, 1]` operand is rank-broadcast on
the unit axis, so lane `(e, c)` reads shifter lane `(e, 0)`. -/
theorem i4IntB_eq (s : BlockState) (b : Region .nat)
    (N stride_bk stride_bn BN BK pn k : Nat) (hBK8 : BK % 8 = 0) :
    i4IntB s b N stride_bk stride_bn BN BK pn k
      = Tile.bop (· &&& ·) Broadcast.scalarR
          (Tile.bop (· >>> ·) (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (i4BRaw s b N stride_bk stride_bn BN BK pn k)
            (i4BShift BK))
          (Tile.scalar 15) := by
  apply Tile.ext
  intro idx
  simp only [i4IntB, i4BNibble, Tile.bop_data, Tile.scalar, i4BShift,
    Broadcast.leftIndex, Broadcast.rightIndex,
    i4BRaw_data s b N stride_bk stride_bn BN BK pn k hBK8]

/-- Likewise for the zero-point nibbles: `(bzp >> bzp_shift_bits) & 0xF`,
`[1, BN]` shifter rank-broadcast on the unit axis. -/
theorem i4IntBzp_eq (s : BlockState) (bzp : Region .nat)
    (N group_size stride_bzpk stride_bzpn BN BK pn k : Nat) :
    i4IntBzp s bzp N group_size stride_bzpk stride_bzpn BN BK pn k
      = Tile.bop (· &&& ·) Broadcast.scalarR
          (Tile.bop (· >>> ·) (Broadcast.consR (Broadcast.consSame Broadcast.nil))
            (i4BzpRaw s bzp N group_size stride_bzpk stride_bzpn BN BK pn k)
            (i4BzpShift N BN pn))
          (Tile.scalar 15) := by
  apply Tile.ext
  intro idx
  simp only [i4IntBzp, bzpNibble, Tile.bop_data, Tile.scalar, i4BzpShift,
    Broadcast.leftIndex, Broadcast.rightIndex, i4BzpRaw_data]

/-- The dequant tile **is** the compiled signed-promotion chain: `intToReal`
of the `.int` difference of the two nibble tiles, times the scales tile. -/
theorem i4BDequantTile_eq (s : BlockState) (bs : RegionName) (b bzp : Region .nat)
    (N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BN BK pn k : Nat) :
    i4BDequantTile s bs b bzp N group_size stride_bk stride_bn stride_bsk
        stride_bsn stride_bzpk stride_bzpn BN BK pn k
      = Tile.bop NumericDType.real.mul
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Tile.intToReal
            (Tile.bop NumericDType.int.sub
              (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (i4NatToInt (i4IntB s b N stride_bk stride_bn BN BK pn k))
              (i4NatToInt
                (i4IntBzp s bzp N group_size stride_bzpk stride_bzpn BN BK pn k))))
          (i4BsTile s bs N group_size stride_bsk stride_bsn BN BK pn k) := by
  apply Tile.ext
  intro idx
  simp only [i4BDequantTile, i4BDequant, i4NatToInt, i4IntB, i4IntBzp, i4BsTile,
    Tile.bop_data, Tile.intToReal_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, NumericDType.sub, WithBot.realMul, Int.ofNat_eq_natCast]
  rfl

/-! ## Per-statement eval recipes

Private copies, since bench ports never import each other. The two `i4_cast*`
recipes are new to this port — `Op.castNatToInt` / `Op.intToReal` are the
signed-promotion chain this kernel is the first consumer of. -/

/-- Unmasked `.ptr` load: every lane reads its own `(region, offset)`. -/
private theorem i4_load_ptr_none {dtype : TileDType} {sh : TileShape}
    (nm : RegName) (t : BlockState) (pt : Tile .ptr sh)
    (hp : t.regs .ptr sh nm = some pt) :
    evalOp (Op.load dtype (MemAccess.ptr (Op.ref .ptr sh nm)) MaskOpt.none) t
      = some (⟨fun i => t.readMemValue dtype (pt.data i).1 (pt.data i).2⟩ :
          Tile dtype sh) := by
  simp only [evalOp, evalOp_ref, hp]
  rfl

/-- Pointer advance / offset. -/
private theorem i4_ptrAdd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (pnm : RegName) (t : BlockState) (pt : Tile .ptr a) (off : Op .nat b)
    (ov : Tile .nat b)
    (hp : t.regs .ptr a pnm = some pt) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ref .ptr a pnm) off) t = some (Tile.ptrAdd bc pt ov) := by
  simp only [evalOp, evalOp_ref, hp, ho]
  rfl

/-- `&` on the `nat` channel — there is no `evalOp_bitAnd` in the library. -/
private theorem i4_bitAnd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.bitAnd bc x y) t = some (Tile.bop (· &&& ·) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `>>` on the `nat` channel. -/
private theorem i4_shiftRight_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.shiftRight bc x y) t = some (Tile.bop (· >>> ·) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `tl.cast(x, tl.int32)` — the explicit nat→int hop. -/
private theorem i4_castNatToInt_eval {sh : TileShape} (x : Op .nat sh)
    (t : BlockState) (vx : Tile .nat sh) (hx : evalOp x t = some vx) :
    evalOp (Op.castNatToInt x) t = some (i4NatToInt vx) := by
  simp only [evalOp, hx]
  rfl

/-- `Op.intToReal` — the signed ℝ embedding. -/
private theorem i4_intToReal_eval {sh : TileShape} (x : Op .int sh)
    (t : BlockState) (vx : Tile .int sh) (hx : evalOp x t = some vx) :
    evalOp (Op.intToReal x) t = some (Tile.intToReal vx) := by
  simp only [evalOp, hx]
  rfl

/-- `(x).to(tl.float16)` — the fp16 downcast of a `.real` operand: pointwise
`FloatDType.cast` (the `attention_fwd_triton2` `p.to(tl.float16)` recipe). -/
private theorem i4_castFp16_eval {sh : TileShape} (x : Op .real sh) (t : BlockState)
    (vx : Tile .real sh) (hx : evalOp x t = some vx) :
    evalOp (Op.castFloat FloatDType.real FloatDType.fp16 x) t
      = some (⟨fun ix => FloatDType.real.cast FloatDType.fp16 (vx.data ix)⟩
          : Tile .fp16 sh) := by
  rw [evalOp_castFloat]
  simp [hx]

/-- The `expandDot` re-widening of `(a).to(tl.float16)`: the
`fp16 → real` cast of the `real → fp16` cast collapses back to the original
`.real` register value (the placeholder `FloatDType.cast` is the identity on
the shared `WithBot ℝ` carrier — the `attention_fwd_triton2` `acc2_op_eval`
round-trip). -/
private theorem i4_castRTReal_eval {sh : TileShape} (nm : RegName) (t : BlockState)
    (vt : Tile .real sh) (h : t.regs .real sh nm = some vt) :
    evalOp (Op.castFloat FloatDType.fp16 FloatDType.real
        (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real sh nm))) t
      = some vt := by
  rw [evalOp_castFloat, evalOp_castFloat]
  simp [evalOp_ref, h]
  ext ix
  simp [FloatDType.cast]

/-- The `expandDot` re-widening of `(b).to(tl.float16)` where `b` is already
the fp16 register holding the downcast dequant tile: the
`fp16 → fp16 → real` chain lands back on the underlying `.real` tile. -/
private theorem i4_castRTFp16_eval {sh : TileShape} (nm : RegName) (t : BlockState)
    (vt : Tile .real sh)
    (h : t.regs .fp16 sh nm
      = some (⟨fun ix => FloatDType.real.cast FloatDType.fp16 (vt.data ix)⟩
          : Tile .fp16 sh)) :
    evalOp (Op.castFloat FloatDType.fp16 FloatDType.real
        (Op.castFloat FloatDType.fp16 FloatDType.fp16 (Op.ref .fp16 sh nm))) t
      = some vt := by
  rw [evalOp_castFloat, evalOp_castFloat]
  simp [evalOp_ref, h]
  ext ix
  simp [FloatDType.cast]

/-- `//` on the `nat` channel — there is no `evalOp_floorDiv` in the library. -/
private theorem i4_floorDiv_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.floorDiv IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.floorDiv IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `%` on the `nat` channel. -/
private theorem i4_mod_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mod IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.mod IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `Op.div` on any numeric channel (what `tl.cdiv` expands to). -/
private theorem i4_divTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.div h bc x y) t = some (Tile.bop h.div bc vx vy) := by
  rw [evalOp_div, hx, hy]
  rfl

/-- `<`. -/
private theorem i4_ltTile_eval {dtype : TileDType} (h : ComparableDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.lt h bc x y) t = some (Tile.cop h.lt bc vx vy) := by
  rw [evalOp_lt, hx, hy]
  rfl

/-- `tl.where` — how `min(a, b)` is lowered, there being no `Op.min`. -/
private theorem i4_where_eval {dtype : TileDType} {sh : TileShape}
    (c : Op .bool sh) (x y : Op dtype sh) (t : BlockState)
    (vc : Tile .bool sh) (vx vy : Tile dtype sh)
    (hc : evalOp c t = some vc) (hx : evalOp x t = some vx)
    (hy : evalOp y t = some vy) :
    evalOp (Op.where c x y) t = some (Tile.select vc vx vy) := by
  rw [evalOp_where, hc, hx, hy]
  rfl

/-- `tl.zeros`. -/
private theorem i4_full_eval {dtype : TileDType} (sh : TileShape) (e : Op dtype [])
    (t : BlockState) (v : Tile dtype []) (hv : evalOp e t = some v) :
    evalOp (Op.full sh e) t
      = some (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype sh) := by
  rw [evalOp_full, hv]
  rfl

/-- A pointer tile built from a bare region base. -/
private theorem i4_ptrAddBase_eval {d : TileDType} {b out : TileShape}
    (bc : Broadcast [] b out) (rg : Region d) (t : BlockState)
    (off : Op .nat b) (ov : Tile .nat b) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ptrBase rg) off) t
      = some (Tile.ptrAdd bc (Tile.scalar ((Region.cast rg : RegionName), 0)) ov) := by
  simp only [evalOp, ho]
  rfl

/-- `*` on two tiles, both operand values known. -/
private theorem i4_mulTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mul h bc x y) t = some (Tile.bop h.mul bc vx vy) := by
  rw [evalOp_mul, hx, hy]
  rfl

/-- `-` on two tiles, both operand values known. -/
private theorem i4_subTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.sub h bc x y) t = some (Tile.bop h.sub bc vx vy) := by
  rw [evalOp_sub, hx, hy]
  rfl

/-- `+` on two tiles, both operand values known. -/
private theorem i4_addTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.add h bc x y) t = some (Tile.bop h.add bc vx vy) := by
  rw [evalOp_add, hx, hy]
  rfl

/-- `[:, None]` / `[None, :]` (the axis binder is spelled `ax`: `axis` is a
DSL keyword-argument token). -/
private theorem i4_expandDim_eval {dtype : TileDType} {sh : TileShape}
    (ax : Fin (sh.length + 1)) (x : Op dtype sh) (t : BlockState)
    (v : Tile dtype sh) (hv : evalOp x t = some v) :
    evalOp (Op.expandDim ax x) t = some (Tile.expandDim ax v) := by
  rw [evalOp_expandDim, hv]
  rfl

/-- `tl.dot` at rank 2. `erw`, not `rw`: the operand shapes are `[] ++ [M, K]`,
which does not unfold at reducible transparency, so `evalOp_dot` silently
fails to fire under `rw` / `simp only`. -/
private theorem i4_dot_eval {M K N : Nat} (x : Op .real [M, K]) (y : Op .real [K, N])
    (t : BlockState) (vx : Tile .real [M, K]) (vy : Tile .real [K, N])
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.dot (batch := []) x y) t = some (Tile.dot [] vx vy) := by
  erw [evalOp_dot, hx, hy]
  rfl

/-- `&` on the bool channel. -/
private theorem i4_boolAnd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .bool a) (y : Op .bool b) (t : BlockState)
    (vx : Tile .bool a) (vy : Tile .bool b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.boolAnd bc x y) t
      = some (Tile.bop (fun u v : Bool => u && v) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-! ### `nat` scalar shapes -/

private theorem i4_mulScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mul .nat Broadcast.nil x y) t = some (Tile.scalar (u * v)) := by
  rw [evalOp_mul, hx, hy]
  rfl

private theorem i4_addScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.add .nat Broadcast.nil x y) t = some (Tile.scalar (u + v)) := by
  rw [evalOp_add, hx, hy]
  rfl

private theorem i4_subScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.sub .nat Broadcast.nil x y) t = some (Tile.scalar (u - v)) := by
  rw [evalOp_sub, hx, hy]
  rfl

private theorem i4_divScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.div .nat Broadcast.nil x y) t = some (Tile.scalar (u / v)) := by
  rw [i4_divTile_eval NumericDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem i4_modScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u % v)) := by
  rw [i4_mod_eval Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem i4_floorDivScalar_eval (x y : Op .nat []) (t : BlockState)
    (u v : Nat) (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u / v)) := by
  rw [i4_floorDiv_eval Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem i4_ltScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.lt ComparableDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (decide (u < v))) := by
  rw [i4_ltTile_eval ComparableDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem i4_whereScalarNat_eval (c : Op .bool []) (x y : Op .nat [])
    (t : BlockState) (cv : Bool) (u v : Nat)
    (hc : evalOp c t = some (Tile.scalar cv))
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.where c x y) t = some (Tile.scalar (if cv then u else v)) := by
  rw [i4_where_eval c x y t _ _ _ hc hx hy]
  rfl

/-- `name * c` on a `nat` scalar register. -/
private theorem i4_mulRef_eval (t : BlockState) (nm : RegName) (val c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c)) t
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `min` is spelled as a `tl.where` by the DSL. -/
private theorem i4_min_as_where (u v : Nat) : (if u < v then u else v) = min u v := by
  rcases Nat.lt_or_ge u v with h | h
  · rw [if_pos h]; omega
  · rw [if_neg (by omega)]; omega

/-- `setReg` leaves memory alone, at **function** level (the library's
`setReg_mem` is pointwise; a deep tower's `t.mem = s.mem` by one `rfl`
overruns `whnf`). -/
private theorem i4_setReg_mem {dtype : TileDType} {sh : TileShape}
    (s : BlockState) (nm : RegName) (v : Tile dtype sh) :
    (s.setReg nm dtype sh v).mem = s.mem := rfl

/-! ### The four loads, bridged to the named tiles -/

/-- The unmasked `a` load lands on `i4ATile`, on the launch state's memory. -/
private theorem i4_aLoad_eq (s0 : BlockState) (a : RegionName) (t : BlockState)
    (M stride_am stride_ak BM BK pm i : Nat)
    (hmem : t.mem = s0.mem)
    (hap : t.regs .ptr [BM, BK] "a_ptrs"
      = some (i4APtrs a M stride_am stride_ak BM BK pm i)) :
    evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        MaskOpt.none) t
      = some (i4ATile s0 a M stride_am stride_ak BM BK pm i) := by
  rw [i4_load_ptr_none "a_ptrs" t _ hap]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [i4ATile, i4APtrs, BlockState.readMemValue_real, BlockState.readMem,
    hmem]

/-- The packed-weight load lands on `i4BRaw`. -/
private theorem i4_bLoad_eq (s0 : BlockState) (b : Region .nat) (t : BlockState)
    (N stride_bk stride_bn BN BK pn i : Nat)
    (hmem : t.mem = s0.mem)
    (hbp : t.regs .ptr [BK, BN] "b_ptrs"
      = some (i4BPtrs b N stride_bk stride_bn BN BK pn i)) :
    evalOp (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BK, BN] "b_ptrs"))
        MaskOpt.none) t
      = some (i4BRaw s0 b N stride_bk stride_bn BN BK pn i) := by
  rw [i4_load_ptr_none "b_ptrs" t _ hbp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [i4BRaw, i4BPtrs, BlockState.readMemValue, BlockState.readMemTyped,
    hmem]

/-- The scales load lands on `i4BsTile`. -/
private theorem i4_bsLoad_eq (s0 : BlockState) (bs : RegionName) (t : BlockState)
    (N group_size stride_bsk stride_bsn BN BK pn i : Nat)
    (hmem : t.mem = s0.mem)
    (hsp : t.regs .ptr [BK, BN] "bs_ptrs"
      = some (i4BsPtrs bs N group_size stride_bsk stride_bsn BN BK pn i)) :
    evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK, BN] "bs_ptrs"))
        MaskOpt.none) t
      = some (i4BsTile s0 bs N group_size stride_bsk stride_bsn BN BK pn i) := by
  rw [i4_load_ptr_none "bs_ptrs" t _ hsp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [i4BsTile, i4BsPtrs, BlockState.readMemValue_real, BlockState.readMem,
    hmem]

/-- The packed zero-point load lands on `i4BzpRaw`. -/
private theorem i4_bzpLoad_eq (s0 : BlockState) (bzp : Region .nat)
    (t : BlockState) (N group_size stride_bzpk stride_bzpn BN BK pn i : Nat)
    (hmem : t.mem = s0.mem)
    (hzp : t.regs .ptr [BK, BN] "bzp_ptrs"
      = some (i4BzpPtrs bzp N group_size stride_bzpk stride_bzpn BN BK pn i)) :
    evalOp (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BK, BN] "bzp_ptrs"))
        MaskOpt.none) t
      = some (i4BzpRaw s0 bzp N group_size stride_bzpk stride_bzpn BN BK pn i) := by
  rw [i4_load_ptr_none "bzp_ptrs" t _ hzp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [i4BzpRaw, i4BzpPtrs, BlockState.readMemValue, BlockState.readMemTyped,
    hmem]

/-! ### The accumulator step -/

/-- A `WithBot ℝ` sum of pointwise products of `some`s is the `some` of the ℝ
sum — the collapse `Tile.dot` needs, every operand lane being a loaded value. -/
private theorem i4_coe_sum_mul {n : Nat} (f g : Fin n → ℝ) :
    (@Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e =>
        Option.map₂ (fun x y : ℝ => x * y) (some (f e)) (some (g e)))
      = some (∑ e : Fin n, f e * g e) := by
  rw [show (@Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e =>
        Option.map₂ (fun x y : ℝ => x * y) (some (f e)) (some (g e)))
      = (@Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e =>
          (some (f e * g e) : WithBot ℝ)) from Finset.sum_congr rfl fun e _ => by simp]
  show (Finset.univ.sum fun e => ((f e * g e : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]
  rfl

/-- **The accumulator statement.** `accumulator += tl.dot(a, b)` extends the
partial sum by exactly one `i4AccStep` — at the wrapped lane coordinates. -/
theorem i4AccTile_dot_succ (s : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (M N group_size stride_am stride_ak stride_bk stride_bn stride_bsk
      stride_bsn stride_bzpk stride_bzpn BM BN BK pm pn i : Nat) :
    Tile.bop NumericDType.real.add
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (i4AccTile s a bs b bzp M N group_size stride_am stride_ak stride_bk
          stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK pm pn i)
        (Tile.dot [] (i4ATile s a M stride_am stride_ak BM BK pm i)
          (i4BDequantTile s bs b bzp N group_size stride_bk stride_bn stride_bsk
            stride_bsn stride_bzpk stride_bzpn BN BK pn i))
      = i4AccTile s a bs b bzp M N group_size stride_am stride_ak stride_bk
          stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK pm pn
          (i + 1) := by
  apply Tile.ext
  intro idx
  obtain ⟨r, c, u⟩ := idx
  rw [i4AccTile_succ]
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, i4AccTile,
    NumericDType.add, WithBot.realAdd]
  -- `erw`: `Tile.dot`'s operand shapes are `[] ++ [M, K]`, so `Tile.dot_nil_data`
  -- does not fire under `rw` / `simp only`.
  erw [Tile.dot_nil_data]
  simp only [i4ATile_data, i4BDequantTile]
  rw [i4_coe_sum_mul]
  simp [i4AccStep]

/-! ### The pid swizzle, statement by statement

Each recipe lands on the **named** quantity (`numPidM` / `groupId` / `pidM` …)
rather than on raw arithmetic; each is definitionally the arithmetic it names. -/

private theorem i4_cdiv_eval (t : BlockState) (X BX : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat X) (Op.constNat BX)) (Op.constNat 1))
        (Op.constNat BX)) t
      = some (Tile.scalar ((X + BX - 1) / BX)) :=
  i4_divScalarNat_eval _ _ t (X + BX - 1) BX
    (i4_subScalarNat_eval _ _ t (X + BX) 1
      (i4_addScalarNat_eval _ _ t X BX (evalOp_constNat _ _) (evalOp_constNat _ _))
      (evalOp_constNat _ _))
    (evalOp_constNat _ _)

private theorem i4_numPidM_eval (t : BlockState) (M BM : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
        (Op.constNat BM)) t
      = some (Tile.scalar (numPidM M BM)) := i4_cdiv_eval t M BM

private theorem i4_numPidN_eval (t : BlockState) (N BN : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
        (Op.constNat BN)) t
      = some (Tile.scalar (numPidN N BN)) := i4_cdiv_eval t N BN

private theorem i4_numPidK_eval (t : BlockState) (K BK : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK)) (Op.constNat 1))
        (Op.constNat BK)) t
      = some (Tile.scalar (numPidK K BK)) := i4_cdiv_eval t K BK

private theorem i4_numPidInGroup_eval (t : BlockState) (N BN GM : Nat)
    (hnpn : t.regs .nat [] "num_pid_n" = some (Tile.scalar (numPidN N BN))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.constNat GM)
        (Op.ref .nat [] "num_pid_n")) t
      = some (Tile.scalar (numPidInGroup N BN GM)) := by
  rw [i4_mulScalarNat_eval _ _ t GM (numPidN N BN) (evalOp_constNat _ _)
    (by rw [evalOp_ref]; exact hnpn)]
  rfl

private theorem i4_groupId_eval (s t : BlockState) (N BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hnig : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (numPidInGroup N BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "num_pid_in_group")) t
      = some (Tile.scalar (groupId s N BN GM)) := by
  rw [i4_floorDivScalar_eval _ _ t (s.pids 0) (numPidInGroup N BN GM)
    (by rw [evalOp_ref]; exact hpid) (by rw [evalOp_ref]; exact hnig)]
  rfl

private theorem i4_firstPidM_eval (s t : BlockState) (N BN GM : Nat)
    (hgid : t.regs .nat [] "group_id" = some (Tile.scalar (groupId s N BN GM))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
        (Op.constNat GM)) t
      = some (Tile.scalar (firstPidM s N BN GM)) := by
  rw [i4_mulScalarNat_eval _ _ t (groupId s N BN GM) GM
    (by rw [evalOp_ref]; exact hgid) (evalOp_constNat _ _)]
  rfl

private theorem i4_groupSizeM_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hnpm : t.regs .nat [] "num_pid_m" = some (Tile.scalar (numPidM M BM)))
    (hfpm : t.regs .nat [] "first_pid_m"
      = some (Tile.scalar (firstPidM s N BN GM))) :
    evalOp (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
            (Op.ref .nat [] "first_pid_m"))
          (Op.constNat GM))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
          (Op.ref .nat [] "first_pid_m"))
        (Op.constNat GM)) t
      = some (Tile.scalar (groupSizeM s M N BM BN GM)) := by
  have hsub : evalOp (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
      (Op.ref .nat [] "first_pid_m")) t
      = some (Tile.scalar (numPidM M BM - firstPidM s N BN GM)) :=
    i4_subScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hnpm)
      (by rw [evalOp_ref]; exact hfpm)
  rw [i4_whereScalarNat_eval _ _ _ t
    (decide (numPidM M BM - firstPidM s N BN GM < GM))
    (numPidM M BM - firstPidM s N BN GM) GM
    (i4_ltScalarNat_eval _ _ t _ _ hsub (evalOp_constNat _ _)) hsub
    (evalOp_constNat _ _)]
  simp only [decide_eq_true_eq, i4_min_as_where, groupSizeM]

private theorem i4_pidM_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hfpm : t.regs .nat [] "first_pid_m" = some (Tile.scalar (firstPidM s N BN GM)))
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hgsm : t.regs .nat [] "group_size_m"
      = some (Tile.scalar (groupSizeM s M N BM BN GM))) :
    evalOp (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "group_size_m"))) t
      = some (Tile.scalar (pidM s M N BM BN GM)) := by
  rw [i4_addScalarNat_eval _ _ t (firstPidM s N BN GM)
    (s.pids 0 % groupSizeM s M N BM BN GM) (by rw [evalOp_ref]; exact hfpm)
    (i4_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hgsm))]
  rfl

private theorem i4_pidN_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hnig : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (numPidInGroup N BN GM)))
    (hgsm : t.regs .nat [] "group_size_m"
      = some (Tile.scalar (groupSizeM s M N BM BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "num_pid_in_group"))
        (Op.ref .nat [] "group_size_m")) t
      = some (Tile.scalar (pidN s M N BM BN GM)) := by
  rw [i4_floorDivScalar_eval _ _ t (s.pids 0 % numPidInGroup N BN GM)
    (groupSizeM s M N BM BN GM)
    (i4_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hnig))
    (by rw [evalOp_ref]; exact hgsm)]
  rfl

/-! ### The index tiles -/

/-- `pid_* * BLOCK + tl.arange(0, BLOCK)` from a scalar register. -/
private theorem i4_offs_eval (nm : RegName) (t : BlockState) (BD base c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar base)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c))
        (Op.arange BD)) t
      = some (i4Offs (base * c) BD) := by
  rw [i4_addTile_eval NumericDType.nat Broadcast.scalarL _ _ t
    (Tile.scalar (base * c)) (Tile.vec (fun i => (i.val : Nat)))
    (i4_mulScalarNat_eval _ _ t base c (by rw [evalOp_ref]; exact hr)
      (evalOp_constNat _ _))
    (evalOp_arange _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i4Offs, Tile.vec, Broadcast.rightIndex, NumericDType.add]

/-- `offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)` under the
launch fact `pid_sp_k = 0` (grid axis 1 has extent `SPLIT_K = 1`). -/
private theorem i4_offsK_eval (t : BlockState) (BK : Nat)
    (hr : t.regs .nat [] "pid_sp_k" = some (Tile.scalar 0)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_sp_k") (Op.constNat BK))
        (Op.arange BK)) t
      = some (i4Offs 0 BK) := by
  have h := i4_offs_eval "pid_sp_k" t BK 0 BK hr
  rwa [Nat.zero_mul] at h

/-- The wrapped offsets `(pid_* * BLOCK + tl.arange(0, BLOCK)) % Mm`. -/
private theorem i4_wrapOffs_eval (nm : RegName) (t : BlockState)
    (BD base c Mm : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar base)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c))
          (Op.arange BD))
        (Op.constNat Mm)) t
      = some (i4WrapOffs (base * c) BD Mm) := by
  rw [i4_mod_eval Broadcast.scalarR _ _ t (i4Offs (base * c) BD) (Tile.scalar Mm)
    (i4_offs_eval nm t BD base c hr) (evalOp_constNat _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i4WrapOffs, i4Offs, Tile.bop_data, Broadcast.leftIndex,
    Broadcast.rightIndex]

/-- `a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak`
— at K step `0`. -/
private theorem i4_aPtrsInit_eval (a : RegionName) (t : BlockState)
    (M stride_am stride_ak BM BK pm : Nat)
    (ham : t.regs .nat [BM] "offs_am" = some (i4WrapOffs (pm * BM) BM M))
    (hok : t.regs .nat [BK] "offs_k" = some (i4Offs 0 BK)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase a)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
            (Op.constNat stride_am))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_ak)))) t
      = some (i4APtrs a M stride_am stride_ak BM BK pm 0) := by
  rw [i4_ptrAddBase_eval _ _ t _ _
    (i4_addTile_eval NumericDType.nat _ _ _ t _ _
      (i4_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_am)
        (i4_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact ham))
        (evalOp_constNat _ _))
      (i4_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_ak)
        (i4_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hok))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i4APtrs, i4AAddr, i4WrapOffs, i4Offs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] *
stride_bn` — at K step `0`. -/
private theorem i4_bPtrsInit_eval (b : Region .nat) (t : BlockState)
    (N stride_bk stride_bn BN BK pn : Nat)
    (hok : t.regs .nat [BK] "offs_k" = some (i4Offs 0 BK))
    (hbn : t.regs .nat [BN] "offs_bn" = some (i4WrapOffs (pn * BN) BN N)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase b)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.floorDiv IntegralDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
              (Op.constNat 8))
            (Op.constNat stride_bk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
            (Op.constNat stride_bn)))) t
      = some (i4BPtrs b N stride_bk stride_bn BN BK pn 0) := by
  rw [i4_ptrAddBase_eval _ _ t _ _
    (i4_addTile_eval NumericDType.nat _ _ _ t _ _
      (i4_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_bk)
        (i4_floorDiv_eval Broadcast.scalarR _ _ t _ (Tile.scalar 8)
          (i4_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hok))
          (evalOp_constNat _ _))
        (evalOp_constNat _ _))
      (i4_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_bn)
        (i4_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hbn))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i4BPtrs, i4BAddr, i4WrapOffs, i4Offs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-! ### The in-loop recomputed tiles -/

/-- `bs_ptrs`, rebuilt at K step `i` from the loop register `k`. -/
private theorem i4_bsPtrs_eval (bs : RegionName) (t : BlockState)
    (N group_size stride_bsk stride_bsn BN BK pn i : Nat)
    (hok : t.regs .nat [BK] "offs_k" = some (i4Offs 0 BK))
    (hbn : t.regs .nat [BN] "offs_bn" = some (i4WrapOffs (pn * BN) BN N))
    (hk : t.regs .nat [] "k" = some (Tile.scalar i)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase bs)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.floorDiv IntegralDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK)))
              (Op.constNat group_size))
            (Op.constNat stride_bsk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
            (Op.constNat stride_bsn)))) t
      = some (i4BsPtrs bs N group_size stride_bsk stride_bsn BN BK pn i) := by
  have hoff : evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.floorDiv IntegralDType.nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK)))
            (Op.constNat group_size))
          (Op.constNat stride_bsk))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
          (Op.constNat stride_bsn)) : Op .nat [BK, BN]) t
      = some (Tile.bop NumericDType.nat.add
          (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Tile.bop NumericDType.nat.mul Broadcast.scalarR
            (Tile.bop (IntegralDType.floorDiv IntegralDType.nat) Broadcast.scalarR
              (Tile.bop NumericDType.nat.add Broadcast.scalarR
                (Tile.expandDim ⟨1, by simp⟩ (i4Offs 0 BK))
                (Tile.scalar (i * BK)))
              (Tile.scalar group_size))
            (Tile.scalar stride_bsk))
          (Tile.bop NumericDType.nat.mul Broadcast.scalarR
            (Tile.expandDim ⟨0, by simp⟩ (i4WrapOffs (pn * BN) BN N))
            (Tile.scalar stride_bsn))) :=
    i4_addTile_eval NumericDType.nat _ _ _ t _ _
      (i4_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ _
        (i4_floorDiv_eval Broadcast.scalarR _ _ t _ _
          (i4_addTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ _
            (i4_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hok))
            (i4_mulRef_eval t "k" i BK hk))
          (evalOp_constNat _ _))
        (evalOp_constNat _ _))
      (i4_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ _
        (i4_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hbn))
        (evalOp_constNat _ _))
  rw [i4_ptrAddBase_eval _ _ t _ _ hoff]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i4BsPtrs, i4BsAddr, i4WrapOffs, i4Offs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `bzp_ptrs`, rebuilt at K step `i`. -/
private theorem i4_bzpPtrs_eval (bzp : Region .nat) (t : BlockState)
    (N group_size stride_bzpk stride_bzpn BN BK pn i : Nat)
    (hok : t.regs .nat [BK] "offs_k" = some (i4Offs 0 BK))
    (hbn : t.regs .nat [BN] "offs_bn" = some (i4WrapOffs (pn * BN) BN N))
    (hk : t.regs .nat [] "k" = some (Tile.scalar i)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase bzp)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.floorDiv IntegralDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK)))
              (Op.constNat group_size))
            (Op.constNat stride_bzpk))
          (Op.mul .nat Broadcast.scalarR
            (Op.floorDiv IntegralDType.nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
              (Op.constNat 8))
            (Op.constNat stride_bzpn)))) t
      = some (i4BzpPtrs bzp N group_size stride_bzpk stride_bzpn BN BK pn i) := by
  have hoff : evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.floorDiv IntegralDType.nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK)))
            (Op.constNat group_size))
          (Op.constNat stride_bzpk))
        (Op.mul .nat Broadcast.scalarR
          (Op.floorDiv IntegralDType.nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
            (Op.constNat 8))
          (Op.constNat stride_bzpn)) : Op .nat [BK, BN]) t
      = some (Tile.bop NumericDType.nat.add
          (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Tile.bop NumericDType.nat.mul Broadcast.scalarR
            (Tile.bop (IntegralDType.floorDiv IntegralDType.nat) Broadcast.scalarR
              (Tile.bop NumericDType.nat.add Broadcast.scalarR
                (Tile.expandDim ⟨1, by simp⟩ (i4Offs 0 BK))
                (Tile.scalar (i * BK)))
              (Tile.scalar group_size))
            (Tile.scalar stride_bzpk))
          (Tile.bop NumericDType.nat.mul Broadcast.scalarR
            (Tile.bop (IntegralDType.floorDiv IntegralDType.nat) Broadcast.scalarR
              (Tile.expandDim ⟨0, by simp⟩ (i4WrapOffs (pn * BN) BN N))
              (Tile.scalar 8))
            (Tile.scalar stride_bzpn))) :=
    i4_addTile_eval NumericDType.nat _ _ _ t _ _
      (i4_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ _
        (i4_floorDiv_eval Broadcast.scalarR _ _ t _ _
          (i4_addTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ _
            (i4_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hok))
            (i4_mulRef_eval t "k" i BK hk))
          (evalOp_constNat _ _))
        (evalOp_constNat _ _))
      (i4_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ _
        (i4_floorDiv_eval Broadcast.scalarR _ _ t _ _
          (i4_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hbn))
          (evalOp_constNat _ _))
        (evalOp_constNat _ _))
  rw [i4_ptrAddBase_eval _ _ t _ _ hoff]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i4BzpPtrs, i4BzpAddr, i4WrapOffs, i4Offs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `b_shift_bits = (offs_k[:, None] % 8) * 4` — a `[BK, 1]` tile. -/
private theorem i4_bShift_eval (t : BlockState) (BK : Nat)
    (hok : t.regs .nat [BK] "offs_k" = some (i4Offs 0 BK)) :
    evalOp ((Op.mul .nat Broadcast.scalarR
        (Op.mod IntegralDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
          (Op.constNat 8))
        (Op.constNat 4)) : Op .nat [BK, 1]) t
      = some (i4BShift BK) := by
  rw [i4_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar 4)
    (i4_mod_eval Broadcast.scalarR _ _ t _ (Tile.scalar 8)
      (i4_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hok))
      (evalOp_constNat _ _))
    (evalOp_constNat _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i4BShift, i4Offs, Tile.bop_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul]

/-- `bzp_shift_bits = (offs_bn[None, :] % 8) * 4` — a `[1, BN]` tile. -/
private theorem i4_bzpShift_eval (t : BlockState) (N BN pn : Nat)
    (hbn : t.regs .nat [BN] "offs_bn" = some (i4WrapOffs (pn * BN) BN N)) :
    evalOp ((Op.mul .nat Broadcast.scalarR
        (Op.mod IntegralDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
          (Op.constNat 8))
        (Op.constNat 4)) : Op .nat [1, BN]) t
      = some (i4BzpShift N BN pn) := by
  rw [i4_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar 4)
    (i4_mod_eval Broadcast.scalarR _ _ t _ (Tile.scalar 8)
      (i4_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hbn))
      (evalOp_constNat _ _))
    (evalOp_constNat _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i4BzpShift, i4WrapOffs, Tile.bop_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul]

/-! ## The K-loop invariant

`i ≤ numKBlocks` is part of the predicate on purpose: `forRange_inv` concludes
only `stop ≤ final`, so carrying the upper bound is what pins
`final = numKBlocks` and lets the readout use `i4AccTile_full`. `bs_ptrs` /
`bzp_ptrs` / the shifters are **not** carried — the body rebuilds them from
`k` before using them. -/

/-- The state carried across K steps. -/
noncomputable def i4Inv (s0 : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (M N group_size stride_am stride_ak stride_bk stride_bn stride_bsk
      stride_bsn stride_bzpk stride_bzpn BM BN BK GM numKBlocks : Nat)
    (i : Nat) (s : BlockState) : Prop :=
  i ≤ numKBlocks
  ∧ s.mem = s0.mem
  ∧ s.pids = s0.pids
  ∧ s.regs .nat [] "pid_m" = some (Tile.scalar (pidM s0 M N BM BN GM))
  ∧ s.regs .nat [] "pid_n" = some (Tile.scalar (pidN s0 M N BM BN GM))
  ∧ s.regs .nat [BK] "offs_k" = some (i4Offs 0 BK)
  ∧ s.regs .nat [BN] "offs_bn"
      = some (i4WrapOffs (pidN s0 M N BM BN GM * BN) BN N)
  ∧ s.regs .ptr [BM, BK] "a_ptrs"
      = some (i4APtrs a M stride_am stride_ak BM BK (pidM s0 M N BM BN GM) i)
  ∧ s.regs .ptr [BK, BN] "b_ptrs"
      = some (i4BPtrs b N stride_bk stride_bn BN BK (pidN s0 M N BM BN GM) i)
  ∧ s.regs .real [BM, BN] "accumulator"
      = some (i4AccTile s0 a bs b bzp M N group_size stride_am stride_ak
          stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
          BM BN BK (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM) i)

/-- The loop combinator writes the induction variable before each iteration,
and `i4Inv` constrains no register named `"k"`. -/
theorem i4Inv_setReg_k (s0 : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (M N group_size stride_am stride_ak stride_bk stride_bn stride_bsk
      stride_bsn stride_bzpk stride_bzpn BM BN BK GM numKBlocks i j : Nat)
    (s : BlockState)
    (h : i4Inv s0 a bs b bzp M N group_size stride_am stride_ak stride_bk
      stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK GM
      numKBlocks i s) :
    i4Inv s0 a bs b bzp M N group_size stride_am stride_ak stride_bk stride_bn
      stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK GM numKBlocks i
      (s.setReg "k" .nat [] (Tile.scalar j)) := by
  obtain ⟨hle, hmem, hpids, hpm, hpn, hok, hbn, hapt, hbpt, hacc⟩ := h
  exact ⟨hle, hmem, hpids, by simpa using hpm, by simpa using hpn,
    by simpa using hok, by simpa using hbn, by simpa using hapt,
    by simpa using hbpt, by simpa using hacc⟩

/-! ### The four recomputed tiles, as one opaque step -/

private theorem i4LoopPtrs_run (bs : RegionName) (bzp : Region .nat)
    (t : BlockState)
    (N group_size stride_bsk stride_bsn stride_bzpk stride_bzpn BN BK pn i : Nat)
    (hok : t.regs .nat [BK] "offs_k" = some (i4Offs 0 BK))
    (hbn : t.regs .nat [BN] "offs_bn" = some (i4WrapOffs (pn * BN) BN N))
    (hk : t.regs .nat [] "k" = some (Tile.scalar i)) :
    ∃ t', stepStmts (i4LoopPtrs bs bzp group_size stride_bsk stride_bsn
          stride_bzpk stride_bzpn BN BK) t = some t'
      ∧ t'.mem = t.mem
      ∧ t'.pids = t.pids
      ∧ (∀ (dtype : TileDType) (sh : TileShape) (nm : RegName),
          nm ≠ "bs_ptrs" → nm ≠ "bzp_ptrs" → nm ≠ "b_shift_bits" →
          nm ≠ "bzp_shift_bits" → t'.regs dtype sh nm = t.regs dtype sh nm)
      ∧ t'.regs .ptr [BK, BN] "bs_ptrs"
          = some (i4BsPtrs bs N group_size stride_bsk stride_bsn BN BK pn i)
      ∧ t'.regs .ptr [BK, BN] "bzp_ptrs"
          = some (i4BzpPtrs bzp N group_size stride_bzpk stride_bzpn BN BK pn i)
      ∧ t'.regs .nat [BK, 1] "b_shift_bits" = some (i4BShift BK)
      ∧ t'.regs .nat [1, BN] "bzp_shift_bits" = some (i4BzpShift N BN pn) := by
  unfold i4LoopPtrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_bsPtrs_eval bs t N group_size stride_bsk stride_bsn BN BK pn i
      hok hbn hk))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_bzpPtrs_eval bzp _ N group_size stride_bzpk stride_bzpn BN BK pn i
      (by simpa using hok) (by simpa using hbn) (by simpa using hk)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_bShift_eval _ BK (by simpa using hok)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_bzpShift_eval _ N BN pn (by simpa using hbn)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [i4_setReg_mem]
  · simp
  · intro dtype sh nm h1 h2 h3 h4
    simp [h1, h2, h3, h4]
  all_goals simp

/-! ### The four unmasked loads, as one opaque step -/

private theorem i4LoopLoads_run (s0 : BlockState) (a bs : RegionName)
    (b bzp : Region .nat) (t : BlockState)
    (M N group_size stride_am stride_ak stride_bk stride_bn stride_bsk
      stride_bsn stride_bzpk stride_bzpn BM BN BK pm pn i : Nat)
    (hmem : t.mem = s0.mem)
    (hapt : t.regs .ptr [BM, BK] "a_ptrs"
      = some (i4APtrs a M stride_am stride_ak BM BK pm i))
    (hbpt : t.regs .ptr [BK, BN] "b_ptrs"
      = some (i4BPtrs b N stride_bk stride_bn BN BK pn i))
    (hbsp : t.regs .ptr [BK, BN] "bs_ptrs"
      = some (i4BsPtrs bs N group_size stride_bsk stride_bsn BN BK pn i))
    (hbzpp : t.regs .ptr [BK, BN] "bzp_ptrs"
      = some (i4BzpPtrs bzp N group_size stride_bzpk stride_bzpn BN BK pn i)) :
    ∃ t', stepStmts (i4LoopLoads BM BN BK) t = some t'
      ∧ t'.mem = t.mem
      ∧ t'.pids = t.pids
      ∧ (∀ (dtype : TileDType) (sh : TileShape) (nm : RegName),
          nm ≠ "a" → nm ≠ "b" → nm ≠ "bs" → nm ≠ "bzp" →
          t'.regs dtype sh nm = t.regs dtype sh nm)
      ∧ t'.regs .real [BM, BK] "a"
          = some (i4ATile s0 a M stride_am stride_ak BM BK pm i)
      ∧ t'.regs .nat [BK, BN] "b"
          = some (i4BRaw s0 b N stride_bk stride_bn BN BK pn i)
      ∧ t'.regs .real [BK, BN] "bs"
          = some (i4BsTile s0 bs N group_size stride_bsk stride_bsn BN BK pn i)
      ∧ t'.regs .nat [BK, BN] "bzp"
          = some (i4BzpRaw s0 bzp N group_size stride_bzpk stride_bzpn BN BK
              pn i) := by
  unfold i4LoopLoads
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_aLoad_eq s0 a t M stride_am stride_ak BM BK pm i hmem hapt))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_bLoad_eq s0 b _ N stride_bk stride_bn BN BK pn i (by simpa using hmem)
      (by simpa using hbpt)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_bsLoad_eq s0 bs _ N group_size stride_bsk stride_bsn BN BK pn i
      (by simpa using hmem) (by simpa using hbsp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_bzpLoad_eq s0 bzp _ N group_size stride_bzpk stride_bzpn BN BK pn i
      (by simpa using hmem) (by simpa using hbzpp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [i4_setReg_mem]
  · simp
  · intro dtype sh nm h1 h2 h3 h4
    simp [h1, h2, h3, h4]
  all_goals simp

/-! ### The K step -/

theorem i4LoopBody_run (s0 : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (M N group_size stride_am stride_ak stride_bk stride_bn stride_bsk
      stride_bsn stride_bzpk stride_bzpn BM BN BK GM numKBlocks i : Nat)
    (s : BlockState)
    (hBK8 : BK % 8 = 0)
    (hnext : i + 1 ≤ numKBlocks)
    (hk : s.regs .nat [] "k" = some (Tile.scalar i))
    (hinv : i4Inv s0 a bs b bzp M N group_size stride_am stride_ak stride_bk
      stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK GM
      numKBlocks i s) :
    ∃ s', stepStmts (i4LoopBody bs bzp group_size stride_ak stride_bk
          stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK) s = some s'
      ∧ i4Inv s0 a bs b bzp M N group_size stride_am stride_ak stride_bk
          stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK GM
          numKBlocks (i + 1) s' := by
  obtain ⟨-, hmem, hpids, hpm, hpn, hok, hbn, hapt, hbpt, hacc⟩ := hinv
  set pm := pidM s0 M N BM BN GM with hpmDef
  set pn := pidN s0 M N BM BN GM with hpnDef
  unfold i4LoopBody
  rw [List.append_assoc]
  -- 1-4. the recomputed pointer / shifter tiles
  obtain ⟨t1, hrun1, h1mem, h1pids, h1keep, h1bsp, h1bzpp, h1bsh, h1bzsh⟩ :=
    i4LoopPtrs_run bs bzp s N group_size stride_bsk stride_bsn stride_bzpk
      stride_bzpn BN BK pn i hok hbn hk
  rw [stepStmts.append_some hrun1]
  -- what the recompute left untouched
  have h1apt : t1.regs .ptr [BM, BK] "a_ptrs"
      = some (i4APtrs a M stride_am stride_ak BM BK pm i) := by
    rw [h1keep .ptr [BM, BK] "a_ptrs" (by decide) (by decide) (by decide)
      (by decide)]
    exact hapt
  have h1bpt : t1.regs .ptr [BK, BN] "b_ptrs"
      = some (i4BPtrs b N stride_bk stride_bn BN BK pn i) := by
    rw [h1keep .ptr [BK, BN] "b_ptrs" (by decide) (by decide) (by decide)
      (by decide)]
    exact hbpt
  -- 5-8. the four loads
  obtain ⟨t2, hrun2, h2mem, h2pids, h2keep, h2a, h2b, h2bs, h2bzp⟩ :=
    i4LoopLoads_run s0 a bs b bzp t1 M N group_size stride_am stride_ak
      stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
      BM BN BK pm pn i (h1mem.trans hmem) h1apt h1bpt h1bsp h1bzpp
  rw [stepStmts.append_some hrun2]
  unfold i4LoopCompute
  -- registers reachable at t2
  have h2bsh : t2.regs .nat [BK, 1] "b_shift_bits" = some (i4BShift BK) := by
    rw [h2keep .nat [BK, 1] "b_shift_bits" (by decide) (by decide) (by decide)
      (by decide)]
    exact h1bsh
  have h2bzsh : t2.regs .nat [1, BN] "bzp_shift_bits"
      = some (i4BzpShift N BN pn) := by
    rw [h2keep .nat [1, BN] "bzp_shift_bits" (by decide) (by decide) (by decide)
      (by decide)]
    exact h1bzsh
  have h2acc : t2.regs .real [BM, BN] "accumulator"
      = some (i4AccTile s0 a bs b bzp M N group_size stride_am stride_ak
          stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
          BM BN BK pm pn i) := by
    rw [h2keep .real [BM, BN] "accumulator" (by decide) (by decide) (by decide)
      (by decide),
      h1keep .real [BM, BN] "accumulator" (by decide) (by decide) (by decide)
      (by decide)]
    exact hacc
  have h2apt : t2.regs .ptr [BM, BK] "a_ptrs"
      = some (i4APtrs a M stride_am stride_ak BM BK pm i) := by
    rw [h2keep .ptr [BM, BK] "a_ptrs" (by decide) (by decide) (by decide)
      (by decide)]
    exact h1apt
  have h2bpt : t2.regs .ptr [BK, BN] "b_ptrs"
      = some (i4BPtrs b N stride_bk stride_bn BN BK pn i) := by
    rw [h2keep .ptr [BK, BN] "b_ptrs" (by decide) (by decide) (by decide)
      (by decide)]
    exact h1bpt
  -- 9. `int_b = (b >> b_shift_bits) & 0xF`
  have h9 : evalOp (Op.bitAnd Broadcast.scalarR
        (Op.shiftRight (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .nat [BK, BN] "b")
          (Op.ref .nat [BK, 1] "b_shift_bits"))
        (Op.constNat 15)) t2
      = some (i4IntB s0 b N stride_bk stride_bn BN BK pn i) := by
    rw [i4IntB_eq s0 b N stride_bk stride_bn BN BK pn i hBK8]
    exact i4_bitAnd_eval Broadcast.scalarR _ _ t2 _ (Tile.scalar 15)
      (i4_shiftRight_eval _ _ _ t2
        (i4BRaw s0 b N stride_bk stride_bn BN BK pn i) (i4BShift BK)
        (by rw [evalOp_ref]; exact h2b) (by rw [evalOp_ref]; exact h2bsh))
      (evalOp_constNat _ _)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h9)]
  -- 10. `int_bzp = (bzp >> bzp_shift_bits) & 0xF`
  have h10 : evalOp (Op.bitAnd Broadcast.scalarR
        (Op.shiftRight (Broadcast.consR (Broadcast.consSame Broadcast.nil))
          (Op.ref .nat [BK, BN] "bzp")
          (Op.ref .nat [1, BN] "bzp_shift_bits"))
        (Op.constNat 15))
        (t2.setReg "int_b" .nat [BK, BN]
          (i4IntB s0 b N stride_bk stride_bn BN BK pn i))
      = some (i4IntBzp s0 bzp N group_size stride_bzpk stride_bzpn BN BK pn i) := by
    rw [i4IntBzp_eq]
    exact i4_bitAnd_eval Broadcast.scalarR _ _ _ _ (Tile.scalar 15)
      (i4_shiftRight_eval _ _ _ _
        (i4BzpRaw s0 bzp N group_size stride_bzpk stride_bzpn BN BK pn i)
        (i4BzpShift N BN pn)
        (by rw [evalOp_ref]; simpa using h2bzp)
        (by rw [evalOp_ref]; simpa using h2bzsh))
      (evalOp_constNat _ _)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h10)]
  -- 11. `b = ((tl.cast(int_b, tl.int32) - tl.cast(int_bzp, tl.int32)) * bs).to(tl.float16)`
  have h11r : evalOp (Op.mul .real
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.intToReal
          (Op.sub .int (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.castNatToInt (Op.ref .nat [BK, BN] "int_b"))
            (Op.castNatToInt (Op.ref .nat [BK, BN] "int_bzp"))))
        (Op.ref .real [BK, BN] "bs"))
        ((t2.setReg "int_b" .nat [BK, BN]
            (i4IntB s0 b N stride_bk stride_bn BN BK pn i)).setReg
          "int_bzp" .nat [BK, BN]
          (i4IntBzp s0 bzp N group_size stride_bzpk stride_bzpn BN BK pn i))
      = some (i4BDequantTile s0 bs b bzp N group_size stride_bk stride_bn
          stride_bsk stride_bsn stride_bzpk stride_bzpn BN BK pn i) := by
    rw [i4BDequantTile_eq]
    exact i4_mulTile_eval NumericDType.real _ _ _ _ _ _
      (i4_intToReal_eval _ _ _
        (i4_subTile_eval NumericDType.int _ _ _ _ _ _
          (i4_castNatToInt_eval _ _ _ (by rw [evalOp_ref]; simp))
          (i4_castNatToInt_eval _ _ _ (by rw [evalOp_ref]; simp))))
      (by rw [evalOp_ref]; simpa using h2bs)
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some TileDType.fp16 [BK, BN] "b"
    (Op.castFloat FloatDType.real FloatDType.fp16
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.intToReal
          (Op.sub .int (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.castNatToInt (Op.ref .nat [BK, BN] "int_b"))
            (Op.castNatToInt (Op.ref .nat [BK, BN] "int_bzp"))))
        (Op.ref .real [BK, BN] "bs"))) _
    (⟨fun ix => FloatDType.real.cast FloatDType.fp16
        ((i4BDequantTile s0 bs b bzp N group_size stride_bk stride_bn
          stride_bsk stride_bsn stride_bzpk stride_bzpn BN BK pn i).data ix)⟩
      : Tile .fp16 [BK, BN])
    (i4_castFp16_eval _ _ _ h11r))]
  -- 12. `accumulator += tl.dot((a).to(tl.float16), (b).to(tl.float16))` — the
  -- fp16 round-trips collapse by cast identity, so the accumulation is the
  -- plain real dot of the loaded `a` tile and the dequantized `b` tile.
  have h12 : evalOp (Op.add .real
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "accumulator")
        (Op.dot (batch := [])
          (Op.castFloat FloatDType.fp16 FloatDType.real
            (Op.castFloat FloatDType.real FloatDType.fp16
              (Op.ref .real [BM, BK] "a")))
          (Op.castFloat FloatDType.fp16 FloatDType.real
            (Op.castFloat FloatDType.fp16 FloatDType.fp16
              (Op.ref .fp16 [BK, BN] "b")))))
        (((t2.setReg "int_b" .nat [BK, BN]
              (i4IntB s0 b N stride_bk stride_bn BN BK pn i)).setReg
            "int_bzp" .nat [BK, BN]
            (i4IntBzp s0 bzp N group_size stride_bzpk stride_bzpn BN BK pn
              i)).setReg
          "b" .fp16 [BK, BN]
          (⟨fun ix => FloatDType.real.cast FloatDType.fp16
              ((i4BDequantTile s0 bs b bzp N group_size stride_bk stride_bn
                stride_bsk stride_bsn stride_bzpk stride_bzpn BN BK pn
                i).data ix)⟩ : Tile .fp16 [BK, BN]))
      = some (i4AccTile s0 a bs b bzp M N group_size stride_am stride_ak
          stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
          BM BN BK pm pn (i + 1)) := by
    rw [← i4AccTile_dot_succ]
    exact i4_addTile_eval NumericDType.real _ _ _ _ _ _
      (by rw [evalOp_ref]; simpa using h2acc)
      (i4_dot_eval _ _ _ _ _
        (i4_castRTReal_eval "a" _ _ (by simpa using h2a))
        (i4_castRTFp16_eval "b" _ _ (by simp)))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h12)]
  -- 13. `a_ptrs += BLOCK_SIZE_K * stride_ak`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_ak))) _
      = some (i4APtrs a M stride_am stride_ak BM BK pm (i + 1)) from by
      rw [← i4APtrs_succ]
      exact i4_ptrAdd_eval Broadcast.scalarR "a_ptrs" _ _ _
        (Tile.scalar (BK * stride_ak)) (by simpa using h2apt)
        (i4_mulScalarNat_eval _ _ _ BK stride_ak (evalOp_constNat _ _)
          (evalOp_constNat _ _))))]
  -- 14. `b_ptrs += (BLOCK_SIZE_K * stride_bk // 8)`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "b_ptrs")
        (Op.floorDiv IntegralDType.nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_bk))
          (Op.constNat 8))) _
      = some (i4BPtrs b N stride_bk stride_bn BN BK pn (i + 1)) from by
      rw [← i4BPtrs_succ]
      exact i4_ptrAdd_eval Broadcast.scalarR "b_ptrs" _ _ _
        (Tile.scalar (BK * stride_bk / 8)) (by simpa using h2bpt)
        (i4_floorDivScalar_eval _ _ _ (BK * stride_bk) 8
          (i4_mulScalarNat_eval _ _ _ BK stride_bk (evalOp_constNat _ _)
            (evalOp_constNat _ _))
          (evalOp_constNat _ _))))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, hnext,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [i4_setReg_mem]
    exact h2mem.trans (h1mem.trans hmem)
  · simp only [BlockState.setReg_pids]
    exact h2pids.trans (h1pids.trans hpids)
  · have := (h2keep .nat [] "pid_m" (by decide) (by decide) (by decide)
      (by decide)).trans ((h1keep .nat [] "pid_m" (by decide) (by decide)
      (by decide) (by decide)).trans hpm)
    simpa using this
  · have := (h2keep .nat [] "pid_n" (by decide) (by decide) (by decide)
      (by decide)).trans ((h1keep .nat [] "pid_n" (by decide) (by decide)
      (by decide) (by decide)).trans hpn)
    simpa using this
  · have := (h2keep .nat [BK] "offs_k" (by decide) (by decide) (by decide)
      (by decide)).trans ((h1keep .nat [BK] "offs_k" (by decide) (by decide)
      (by decide) (by decide)).trans hok)
    simpa using this
  · have := (h2keep .nat [BN] "offs_bn" (by decide) (by decide) (by decide)
      (by decide)).trans ((h1keep .nat [BN] "offs_bn" (by decide) (by decide)
      (by decide) (by decide)).trans hbn)
    simpa using this
  · simp [hpmDef]
  · simp [hpnDef]
  · simp [hpmDef, hpnDef]

/-! ### Collapsing the K loop -/

theorem i4Loop_collapse (s0 : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (M N group_size stride_am stride_ak stride_bk stride_bn stride_bsk
      stride_bsn stride_bzpk stride_bzpn BM BN BK GM numKBlocks : Nat)
    (s : BlockState)
    (hBK8 : BK % 8 = 0)
    (h0 : i4Inv s0 a bs b bzp M N group_size stride_am stride_ak stride_bk
      stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK GM
      numKBlocks 0 s) :
    ∃ sF, stepStmt (Stmt.forRange "k" 0 numKBlocks 1
          (i4LoopBody bs bzp group_size stride_ak stride_bk stride_bsk
            stride_bsn stride_bzpk stride_bzpn BM BN BK)) s = some sF
      ∧ i4Inv s0 a bs b bzp M N group_size stride_am stride_ak stride_bk
          stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK GM
          numKBlocks numKBlocks sF := by
  obtain ⟨final, sF, hrun, hfinal, hP⟩ :=
    forRange_inv (idx := "k") (start := 0) (stop := numKBlocks) (step := 1)
      (P := fun i t => i4Inv s0 a bs b bzp M N group_size stride_am stride_ak
        stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
        BM BN BK GM numKBlocks i t)
      one_ne_zero h0
      (fun i t hi hinv => by
        obtain ⟨s', hs', hinv'⟩ :=
          i4LoopBody_run s0 a bs b bzp M N group_size stride_am stride_ak
            stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
            BM BN BK GM numKBlocks i _ hBK8 (by omega) (by simp)
            (i4Inv_setReg_k s0 a bs b bzp M N group_size stride_am stride_ak
              stride_bk stride_bn stride_bsk stride_bsn stride_bzpk
              stride_bzpn BM BN BK GM numKBlocks i i t hinv)
        exact ⟨s', hs', hinv'⟩)
  have hEq : final = numKBlocks := le_antisymm hP.1 hfinal
  subst hEq
  exact ⟨sF, hrun, hP⟩

/-! ## The prologue walks -/

/-- The eleven scalar statements. Memory is untouched; the registers anything
downstream reads are the SPLIT_K program id and the block coordinates. -/
theorem i4PreLoopScalars_run (s : BlockState) (M N K BM BN BK GM : Nat) :
    ∃ t, stepStmts (i4PreLoopScalars M N K BM BN BK GM) s = some t
      ∧ t.mem = s.mem
      ∧ t.pids = s.pids
      ∧ t.regs .nat [] "pid_sp_k" = some (Tile.scalar (s.pids 1))
      ∧ t.regs .nat [] "pid_m" = some (Tile.scalar (pidM s M N BM BN GM))
      ∧ t.regs .nat [] "pid_n" = some (Tile.scalar (pidN s M N BM BN GM)) := by
  unfold i4PreLoopScalars
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (i4_numPidM_eval _ M BM))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (i4_numPidN_eval _ N BN))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (i4_numPidK_eval _ K BK))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_numPidInGroup_eval _ N BN GM (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_groupId_eval s _ N BN GM (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_firstPidM_eval s _ N BN GM (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_groupSizeM_eval s _ M N BM BN GM (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_pidM_eval s _ M N BM BN GM (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_pidN_eval s _ M N BM BN GM (by simp) (by simp) (by simp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_⟩ <;> simp [i4_setReg_mem]

/-- The six index statements, ending on `i4Inv` at step `0`. The `pid_sp_k = 0`
hypothesis is the launch fact `s.pids 1 = 0` (grid axis 1 has extent
`SPLIT_K = 1`), already rewritten by the caller. -/
theorem i4PreLoopTiles_run (s0 : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (M N group_size stride_am stride_ak stride_bk stride_bn stride_bsk
      stride_bsn stride_bzpk stride_bzpn BM BN BK GM numKBlocks : Nat)
    (t : BlockState)
    (hmem : t.mem = s0.mem)
    (hpids : t.pids = s0.pids)
    (hpsk : t.regs .nat [] "pid_sp_k" = some (Tile.scalar 0))
    (hpm : t.regs .nat [] "pid_m" = some (Tile.scalar (pidM s0 M N BM BN GM)))
    (hpn : t.regs .nat [] "pid_n" = some (Tile.scalar (pidN s0 M N BM BN GM))) :
    ∃ t', stepStmts (i4PreLoopIndex a b M N stride_am stride_ak stride_bk
          stride_bn BM BN BK) t = some t'
      ∧ i4Inv s0 a bs b bzp M N group_size stride_am stride_ak stride_bk
          stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK GM
          numKBlocks 0 t' := by
  unfold i4PreLoopIndex
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_wrapOffs_eval "pid_m" t BM (pidM s0 M N BM BN GM) BM M hpm))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_wrapOffs_eval "pid_n" _ BN (pidN s0 M N BM BN GM) BN N
      (by simpa using hpn)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_offsK_eval _ BK (by simpa using hpsk)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_aPtrsInit_eval a _ M stride_am stride_ak BM BK (pidM s0 M N BM BN GM)
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_bPtrsInit_eval b _ N stride_bk stride_bn BN BK (pidN s0 M N BM BN GM)
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM, BN] (Op.const 0)) _
        = some (i4AccTile s0 a bs b bzp M N group_size stride_am stride_ak
            stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
            BM BN BK (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM) 0) from by
      rw [i4AccTile_zero]
      exact i4_full_eval [BM, BN] (Op.const 0) _ _ (evalOp_const 0 _)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [i4_setReg_mem]
    exact hmem
  · simpa using hpids
  · simpa using hpm
  · simpa using hpn
  · simp
  · simp
  · simp
  · simp
  · simp

/-! ## The output store

A masked **`.fp16`-typed** `.ptr` store: the stored `c` register is the fp16
downcast of the accumulator, so each active lane writes a `.fp16` memory cell
(`writeMemTyped .fp16`), and the readback is stated at the `MemCell` level
via `scatter_memcell_fp16_prop_masked_nd` — the `matmul_tma` f16 precedent,
here on a masked `.ptr` store. `c_mask` is the only gate. The lane-to-address
map has to be injective for the readback to name a unique lane; that is the
headline's `hInj`, and the shared `cAddr_injective` discharges it for a
row-major `C`. -/

/-- The `C` pointer tile. -/
noncomputable def i4CPtrs (c : RegionName) (stride_cm stride_cn BM BN pm pn : Nat) :
    Tile .ptr [BM, BN] :=
  ⟨fun idx => (c, cAddr stride_cm stride_cn BM BN pm pn idx)⟩

/-- `c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)` — over the
**unwrapped** output coordinates. -/
def i4CMask (M N BM BN pm pn : Nat) : Tile .bool [BM, BN] :=
  ⟨fun idx => decide (pm * BM + idx.1.val < M) && decide (pn * BN + idx.2.1.val < N)⟩

/-- The post-store state: one masked **`.fp16`-typed** scatter over the
`[BM, BN]` output tile. -/
noncomputable def i4StoreState (c : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat)
    (f : TileIndex [BM, BN] → TileCarrier .fp16) (t : BlockState) : BlockState :=
  (TileShape.allIndices [BM, BN]).foldl
    (fun acc i => if pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N then
        acc.writeMemTyped .fp16 c (cAddr stride_cm stride_cn BM BN pm pn i) (f i)
      else acc) t

private theorem i4_cPtrsInit_eval (c : RegionName) (t : BlockState)
    (stride_cm stride_cn BM BN pm pn : Nat)
    (hcm : t.regs .nat [BM] "offs_cm" = some (i4Offs (pm * BM) BM))
    (hcn : t.regs .nat [BN] "offs_cn" = some (i4Offs (pn * BN) BN)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase c)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_cm)
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_cn)
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) t
      = some (i4CPtrs c stride_cm stride_cn BM BN pm pn) := by
  rw [i4_ptrAddBase_eval _ _ t _ _
    (i4_addTile_eval NumericDType.nat _ _ _ t _ _
      (i4_mulTile_eval NumericDType.nat Broadcast.scalarL _ _ t
        (Tile.scalar stride_cm) _ (evalOp_constNat _ _)
        (i4_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcm)))
      (i4_mulTile_eval NumericDType.nat Broadcast.scalarL _ _ t
        (Tile.scalar stride_cn) _ (evalOp_constNat _ _)
        (i4_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcn))))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i4CPtrs, cAddr, i4Offs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

private theorem i4_cMaskInit_eval (t : BlockState) (M N BM BN pm pn : Nat)
    (hcm : t.regs .nat [BM] "offs_cm" = some (i4Offs (pm * BM) BM))
    (hcn : t.regs .nat [BN] "offs_cn" = some (i4Offs (pn * BN) BN)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        ((Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm"))
          (Op.constNat M) : Op .bool [BM, 1]))
        ((Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))
          (Op.constNat N) : Op .bool [1, BN]))) t
      = some (i4CMask M N BM BN pm pn) := by
  rw [i4_boolAnd_eval _ _ _ t _ _
    (i4_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar M)
      (i4_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcm))
      (evalOp_constNat _ _))
    (i4_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar N)
      (i4_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcn))
      (evalOp_constNat _ _))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i4CMask, i4Offs, Tile.bop_data, Tile.cop_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]

/-- The masked store of the **`c` register** (this kernel stores `c`, not
`accumulator` — and `c` lives on the `.fp16` channel). -/
private theorem i4_store_eq (c : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat) (t : BlockState)
    (vt : Tile .fp16 [BM, BN]) (f : TileIndex [BM, BN] → TileCarrier .fp16)
    (hfv : ∀ i, vt.data i = f i)
    (hcp : t.regs .ptr [BM, BN] "c_ptrs"
      = some (i4CPtrs c stride_cm stride_cn BM BN pm pn))
    (hcmask : t.regs .bool [BM, BN] "c_mask" = some (i4CMask M N BM BN pm pn))
    (hv : t.regs .fp16 [BM, BN] "c" = some vt) :
    stepStmt (Stmt.store .fp16 [BM, BN]
        (MemAccess.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
        (Op.ref .fp16 [BM, BN] "c")
        (MaskOpt.mask (Op.ref .bool [BM, BN] "c_mask"))) t
      = some (i4StoreState c M N stride_cm stride_cn BM BN pm pn f t) := by
  unfold stepStmt i4StoreState
  simp only [evalOp_ref, hv, hcp, hcmask, Option.map_some]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [BM, BN])) ?_)
  funext acc i
  obtain ⟨r, cc, u⟩ := i
  by_cases hb : pm * BM + r.val < M ∧ pn * BN + cc.val < N
  · simp only [i4CMask, if_pos hb, i4CPtrs, hfv]
    simp [hb.1, hb.2]
  · simp only [i4CMask, if_neg hb, i4CPtrs]
    rw [if_neg]
    intro hcon
    exact hb (by simpa using hcon)

private theorem i4_store_props (c : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat) (t : BlockState)
    (f : TileIndex [BM, BN] → TileCarrier .fp16)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN pm pn i)) :
    ∀ i : TileIndex [BM, BN],
      (pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N) →
      (i4StoreState c M N stride_cm stride_cn BM BN pm pn f t).mem c
          (cAddr stride_cm stride_cn BM BN pm pn i)
        = MemCell.of .fp16
            (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (f i))) := by
  classical
  intro i hi
  unfold i4StoreState
  have h := scatter_memcell_fp16_prop_masked_nd (region := c) t
    (fun j : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN pm pn j) f
    (fun j : TileIndex [BM, BN] => pm * BM + j.1.val < M ∧ pn * BN + j.2.1.val < N)
    hInj i
  rw [h, if_pos hi]

/- `hInj` for a row-major `C` is discharged by the shared `cAddr_injective`
(the `matmul4_kernel` section above). -/

/-! ### The tail

Six statements: the fp16-cast `c` binding (which **is** stored), the two
unwrapped output coordinate vectors, the `C` pointer tile, the two-axis mask,
and the masked `.fp16` store of `c`. On every lane the mask lets through,
`row < M` and `col < N` turn the wrapped accumulator lane into the plain
`i4AccSpec` — that is the `Nat.mod_eq_of_lt` step — and the readback lands on
the `.fp16` memory cell holding `fp16(i4AccSpec)`. -/

theorem i4PostLoop_run (s0 : BlockState) (c a bs : RegionName)
    (b bzp : Region .nat)
    (M N group_size stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK GM numKBlocks : Nat)
    (t : BlockState)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN
        (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM) i))
    (hinv : i4Inv s0 a bs b bzp M N group_size stride_am stride_ak stride_bk
      stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK GM
      numKBlocks numKBlocks t) :
    ∃ sF, stepStmts (i4PostLoop c M N stride_cm stride_cn BM BN) t = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (pidM s0 M N BM BN GM * BM + idx.1.val < M
            ∧ pidN s0 M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem c (cAddr stride_cm stride_cn BM BN (pidM s0 M N BM BN GM)
              (pidN s0 M N BM BN GM) idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (i4AccSpec s0 a bs b bzp group_size stride_am stride_ak
                  stride_bk stride_bn stride_bsk stride_bsn stride_bzpk
                  stride_bzpn BK numKBlocks
                  (pidM s0 M N BM BN GM * BM + idx.1.val)
                  (pidN s0 M N BM BN GM * BN + idx.2.1.val)))) := by
  obtain ⟨-, -, -, hpm, hpn, -, -, -, -, hacc⟩ := hinv
  unfold i4PostLoop
  -- 1. `c = (accumulator).to(tl.float16)` — a genuine fp16 downcast, and the
  -- register the store reads
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some TileDType.fp16 [BM, BN] "c"
    (Op.castFloat FloatDType.real FloatDType.fp16
      (Op.ref .real [BM, BN] "accumulator")) _
    (⟨fun ix => FloatDType.real.cast FloatDType.fp16
        ((i4AccTile s0 a bs b bzp M N group_size stride_am stride_ak stride_bk
          stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK
          (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM) numKBlocks).data ix)⟩
      : Tile .fp16 [BM, BN])
    (i4_castFp16_eval _ _ _ (by rw [evalOp_ref]; exact hacc)))]
  -- 2-3. the unwrapped output coordinate vectors
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_offs_eval "pid_m" _ BM (pidM s0 M N BM BN GM) BM (by simpa using hpm)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_offs_eval "pid_n" _ BN (pidN s0 M N BM BN GM) BN (by simpa using hpn)))]
  -- 4-5. the `C` pointer tile and the two-axis mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_cPtrsInit_eval c _ stride_cm stride_cn BM BN (pidM s0 M N BM BN GM)
      (pidN s0 M N BM BN GM) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i4_cMaskInit_eval _ M N BM BN (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM)
      (by simp) (by simp)))]
  -- 6. the `.fp16` store of `c`
  rw [stepStmts.cons_some
    (i4_store_eq c M N stride_cm stride_cn BM BN (pidM s0 M N BM BN GM)
      (pidN s0 M N BM BN GM) _
      (⟨fun ix => FloatDType.real.cast FloatDType.fp16
          ((i4AccTile s0 a bs b bzp M N group_size stride_am stride_ak stride_bk
            stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK
            (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM) numKBlocks).data ix)⟩
        : Tile .fp16 [BM, BN])
      (fun idx => FloatDType.real.cast FloatDType.fp16
        (some (i4AccSpec s0 a bs b bzp group_size stride_am stride_ak stride_bk
          stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BK numKBlocks
          ((pidM s0 M N BM BN GM * BM + idx.1.val) % M)
          ((pidN s0 M N BM BN GM * BN + idx.2.1.val) % N))))
      (fun idx => congrArg (FloatDType.real.cast FloatDType.fp16)
        (i4AccTile_full s0 a bs b bzp M N group_size stride_am stride_ak
          stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
          BM BN BK (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM)
          numKBlocks idx))
      (by simp) (by simp) (by simp))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx hidx
  rw [i4_store_props c M N stride_cm stride_cn BM BN
      (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM) _
      (fun idx => FloatDType.real.cast FloatDType.fp16
        (some (i4AccSpec s0 a bs b bzp group_size stride_am stride_ak stride_bk
          stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BK numKBlocks
          ((pidM s0 M N BM BN GM * BM + idx.1.val) % M)
          ((pidN s0 M N BM BN GM * BN + idx.2.1.val) % N))))
      hInj idx hidx]
  simp only [Nat.mod_eq_of_lt hidx.1, Nat.mod_eq_of_lt hidx.2]
  simp only [FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue,
    FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot, FloatDType.fp16_toWithBot,
    WithBot.unbotD_some]

/-! ## Main theorem -/

set_option maxHeartbeats 1000000 in
-- `hK` is deliberately carried even though only the *loop bound* `numKBlocks`
-- appears semantically (K itself survives only in the dead `num_pid_k`
-- binding): it is the launch fact that makes the `numKBlocks` presentation of
-- `tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)` faithful.
set_option linter.unusedVariables false in
/-- **Genuine, dimension-general correctness.** For every launch state, the
`SPLIT_K = 1` arm of the kernel runs to completion, and every in-range output
lane of `C` holds `i4AccSpec`: the sum over all `numKBlocks` K steps of `tl.dot`
between the loaded `A` tile and the signed-dequantized weight tile
`(nib(B) − nib(BZP)) · BS`, with per-lane group rows `(e + k·BK) / group_size`.
The store writes `c = accumulator.to(tl.float16)` — a genuine fp16
quantization event, so every active output cell is a **`.fp16`-typed memory
cell** and the conclusion is stated at the `MemCell` level:
`C[row, col] = MemCell.of .fp16 (fp16(accSpec))` (the `matmul_tma` f16-branch
precedent; the placeholder `FloatDType.cast` is the identity, so the carried
value is `i4AccSpec` itself).

The hypotheses are the kernel's own launch facts: `hK` is the source's
`assert K % (BLOCK_SIZE_K * SPLIT_K) == 0` (the loop trip count `numKBlocks`
is exact — the loads are unmasked); `hBK8` is the source's
`assert BLOCK_SIZE_K % 8 == 0` (eight nibbles per packed word); `hpid1` is
grid axis 1 having extent `SPLIT_K = 1`; `hInj` says distinct output lanes get
distinct `C` addresses — `cAddr_injective` discharges it for a row-major `C`.
The `% M` / `% N` offset wraps disappear on exactly the lanes the store mask
lets through (`Nat.mod_eq_of_lt`). -/
specification matmul_dequantize_matmul_exec_genuine
    (a_ptr c_ptr bs_ptr : RegionName) (b_ptr bzp_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_bsk stride_bsn stride_bzpk stride_bzpn group_size : Nat)
    (BM BN BK GM numKBlocks : Nat) (s : BlockState)
    (hK : K = BK * numKBlocks)
    (hBK8 : BK % 8 = 0)
    (hpid1 : s.pids 1 = 0)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN
        (pidM s M N BM BN GM) (pidN s M N BM BN GM) i)) :
    ∃ sF, exec (matmul_dequantize_matmul_surface a_ptr c_ptr bs_ptr b_ptr bzp_ptr
        M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        stride_bsk stride_bsn stride_bzpk stride_bzpn group_size
        BM BN BK GM numKBlocks).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (pidM s M N BM BN GM * BM + idx.1.val < M
            ∧ pidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem c_ptr (cAddr stride_cm stride_cn BM BN (pidM s M N BM BN GM)
              (pidN s M N BM BN GM) idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (i4AccSpec s a_ptr bs_ptr b_ptr bzp_ptr group_size
                  stride_am stride_ak stride_bk stride_bn stride_bsk stride_bsn
                  stride_bzpk stride_bzpn BK numKBlocks
                  (pidM s M N BM BN GM * BM + idx.1.val)
                  (pidN s M N BM BN GM * BN + idx.2.1.val)))) := by
  rw [exec, i4_body_eq]
  -- prologue: the scalars, then the index tiles
  obtain ⟨t1, hrun1, h1mem, h1pids, h1psk, h1pm, h1pn⟩ :=
    i4PreLoopScalars_run s M N K BM BN BK GM
  obtain ⟨t2, hrun2, h2inv⟩ :=
    i4PreLoopTiles_run s a_ptr bs_ptr b_ptr bzp_ptr M N group_size stride_am
      stride_ak stride_bk stride_bn stride_bsk stride_bsn stride_bzpk
      stride_bzpn BM BN BK GM numKBlocks t1 h1mem h1pids (hpid1 ▸ h1psk)
      h1pm h1pn
  simp only [List.append_assoc]
  rw [stepStmts.append_some hrun1, stepStmts.append_some hrun2]
  -- the collapsed K loop
  obtain ⟨t3, hrun3, h3inv⟩ :=
    i4Loop_collapse s a_ptr bs_ptr b_ptr bzp_ptr M N group_size stride_am
      stride_ak stride_bk stride_bn stride_bsk stride_bsn stride_bzpk
      stride_bzpn BM BN BK GM numKBlocks t2 hBK8 h2inv
  rw [show [Stmt.forRange "k" 0 numKBlocks 1
          (i4LoopBody bs_ptr bzp_ptr group_size stride_ak stride_bk stride_bsk
            stride_bsn stride_bzpk stride_bzpn BM BN BK)]
        ++ i4PostLoop c_ptr M N stride_cm stride_cn BM BN
      = Stmt.forRange "k" 0 numKBlocks 1
          (i4LoopBody bs_ptr bzp_ptr group_size stride_ak stride_bk stride_bsk
            stride_bsn stride_bzpk stride_bzpn BM BN BK)
        :: i4PostLoop c_ptr M N stride_cm stride_cn BM BN from rfl]
  rw [stepStmts.cons_some hrun3]
  -- the tail
  obtain ⟨sF, hpost, hout⟩ :=
    i4PostLoop_run s c_ptr a_ptr bs_ptr b_ptr bzp_ptr M N group_size stride_am
      stride_ak stride_bk stride_bn stride_cm stride_cn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BM BN BK GM numKBlocks t3 hInj h3inv
  exact ⟨sF, hpost, hout⟩


/-! # ════════ Kernel 3 of 3: `dequantize_kernel` ════════ -/

/-! ## Kernel surface (faithful transcription)

Allowed mechanical Lean-syntax-only changes: Python `BLOCK_SIZE_K:
tl.constexpr` / `BLOCK_SIZE_N: tl.constexpr` and the runtime scalar arguments
(`K`, `N`, `group_size`, the eight strides) become Lean `Nat` binders; the
packed-word regions are typed `Region .nat`. Everything else is the two
disclosed respells in the module preamble. -/

def matmul_dequantize_dequantize_surface
    (b_ptr : Region .nat) (b_scale_ptr : RegionName) (b_zp_ptr : Region .nat)
    (fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn : Nat)
    (BLOCK_SIZE_K BLOCK_SIZE_N : Nat) : ComputeKernel := triton {
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = k_block_idx * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = n_block_idx * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  fpb_offs = offs_k[:, None] * $(stride_fpbk) + offs_n[None, :] * $(stride_fpbn)
  b_offs = (offs_k[:, None] // $(8)) * $(stride_bk) + offs_n[None, :] * $(stride_bn)
  bzp_offs = (offs_k[:, None] // $(group_size)) * $(stride_bzpk) + (offs_n[None, :] // $(8)) * $(stride_bzpn)
  bs_offs = (offs_k[:, None] // $(group_size)) * $(stride_bsk) + offs_n[None, :] * $(stride_bsn)
  n_mask = offs_n[None, :] < $(N)
  k_mask = offs_k[:, None] < $(K)
  mask = n_mask & k_mask
  int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=mask, other=0.0)
  b_shift = (offs_k[:, None] % $(8)) * $(4)
  bzp_shift = (offs_n[None, :] % $(8)) * $(4)
  fp_weight = (tl.cast((int32_b >> b_shift) & $(15), tl.int32) - tl.cast((zp_b >> bzp_shift) & $(15), tl.int32)) * scale_b
  tl.store(fpb_ptr + fpb_offs, fp_weight, mask=mask)
}

theorem matmul_dequant_int4_surface_toAlgorithm_supported
    (b_ptr b_zp_ptr : Region .nat) (b_scale_ptr fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn BK BN : Nat) :
    ∃ alg, (matmul_dequantize_dequantize_surface b_ptr b_scale_ptr b_zp_ptr fpb_ptr
      K N group_size stride_bk stride_bn stride_bsk stride_bsn stride_bzpk
      stride_bzpn stride_fpbk stride_fpbn BK BN).toAlgorithm?
        = Except.ok alg := by
  simp [matmul_dequantize_dequantize_surface, ComputeExpr.toAlgorithm?]

/-! ## Element accessors

Each accessor is the kernel's own address arithmetic over the **launch**
state's memory, parameterized by the *absolute* row `k = k_block_idx ·
BLOCK_SIZE_K + i` and column `n = n_block_idx · BLOCK_SIZE_N + j`. -/

/-- The packed 32-bit word holding the weight nibble for absolute cell
`(k, n)`: `b[k // 8, n]`. Eight weights share a word along K, hence `k / 8`. -/
def dqBWord (s : BlockState) (b_ptr : Region .nat)
    (stride_bk stride_bn k n : Nat) : Nat :=
  s.readMemValue .nat b_ptr.cast (k / 8 * stride_bk + n * stride_bn)

/-- The unpacked 4-bit weight: `(dqBWord >> (k % 8) * 4) & 0xF`. -/
def dqBNibble (s : BlockState) (b_ptr : Region .nat)
    (stride_bk stride_bn k n : Nat) : Nat :=
  dqBWord s b_ptr stride_bk stride_bn k n >>> (k % 8 * 4) &&& 15

/-- The packed word holding the zero-point nibble for absolute cell `(k, n)`:
`zp[k // group_size, n // 8]`. Eight zero-points share a word along N, hence
`n / 8`; one group row per `group_size` K rows. -/
def dqZpWord (s : BlockState) (b_zp_ptr : Region .nat)
    (group_size stride_bzpk stride_bzpn k n : Nat) : Nat :=
  s.readMemValue .nat b_zp_ptr.cast
    (k / group_size * stride_bzpk + n / 8 * stride_bzpn)

/-- The unpacked 4-bit zero-point: `(dqZpWord >> (n % 8) * 4) & 0xF`. -/
def dqZpNibble (s : BlockState) (b_zp_ptr : Region .nat)
    (group_size stride_bzpk stride_bzpn k n : Nat) : Nat :=
  dqZpWord s b_zp_ptr group_size stride_bzpk stride_bzpn k n >>> (n % 8 * 4) &&& 15

/-- The scale at absolute cell `(k, n)`: `bs[k // group_size, n]`. -/
noncomputable def dqScElem (s : BlockState) (b_scale_ptr : RegionName)
    (group_size stride_bsk stride_bsn k n : Nat) : ℝ :=
  s.readMem b_scale_ptr (k / group_size * stride_bsk + n * stride_bsn)

/-- **The stored value** at absolute cell `(k, n)`: the **signed** nibble
difference (ℤ-subtraction — this is what the `.int` hop of disclosure (1)
buys; ℕ subtraction would truncate), embedded into ℝ and scaled. -/
noncomputable def dequantSpec (s : BlockState) (b_scale_ptr : RegionName)
    (b_ptr b_zp_ptr : Region .nat)
    (group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn : Nat) (k n : Nat) : ℝ :=
  (((dqBNibble s b_ptr stride_bk stride_bn k n : ℤ)
      - (dqZpNibble s b_zp_ptr group_size stride_bzpk stride_bzpn k n : ℤ) : ℤ) : ℝ)
    * dqScElem s b_scale_ptr group_size stride_bsk stride_bsn k n

/-- The `fp_b` store address for tile cell `idx` of program `(p₀, p₁)`:
`(p₀·BK + i) · stride_fpbk + (p₁·BN + j) · stride_fpbn`. -/
def fpbAddr (stride_fpbk stride_fpbn BK BN p₀ p₁ : Nat)
    (idx : TileIndex [BK, BN]) : Nat :=
  (p₀ * BK + idx.1.val) * stride_fpbk + (p₁ * BN + idx.2.1.val) * stride_fpbn

/-- The kernel's store gate `mask = n_mask & k_mask`, in the kernel's own
conjunct order (the N test comes first in the source). -/
def dqActive (K N BK BN p₀ p₁ : Nat) (idx : TileIndex [BK, BN]) : Prop :=
  p₁ * BN + idx.2.1.val < N ∧ p₀ * BK + idx.1.val < K

/-- `fpbAddr` is injective for a row-major `fp_b` (`stride_fpbn > 0` and a
K-row stride wide enough to hold `BLOCK_SIZE_N` columns) — the discharge
lemma for the headline's `hInj`. The host's contiguous `fp_b` of
`torch.ones((K, N))` has `stride_fpbk = N ≥ BN · 1` for every launched tile. -/
theorem fpbAddr_injective (stride_fpbk stride_fpbn BK BN p₀ p₁ : Nat)
    (hn : 0 < stride_fpbn) (hfit : BN * stride_fpbn ≤ stride_fpbk) :
    Function.Injective
      (fun idx : TileIndex [BK, BN] =>
        fpbAddr stride_fpbk stride_fpbn BK BN p₀ p₁ idx) := by
  intro i j hij
  obtain ⟨r₁, c₁, u₁⟩ := i
  obtain ⟨r₂, c₂, u₂⟩ := j
  simp only [fpbAddr] at hij
  have hexp : ∀ r c : Nat,
      (p₀ * BK + r) * stride_fpbk + (p₁ * BN + c) * stride_fpbn
        = p₀ * BK * stride_fpbk + r * stride_fpbk
          + (p₁ * BN * stride_fpbn + c * stride_fpbn) := by
    intro r c
    rw [Nat.add_mul, Nat.add_mul]
  rw [hexp, hexp] at hij
  have key : r₁.val * stride_fpbk + c₁.val * stride_fpbn
      = r₂.val * stride_fpbk + c₂.val * stride_fpbn := by omega
  have hrow : ∀ c : Fin BN, c.val * stride_fpbn < stride_fpbk := fun c =>
    lt_of_lt_of_le (Nat.mul_lt_mul_of_lt_of_le c.isLt (le_refl _) hn) hfit
  have hr : r₁.val = r₂.val := by
    rcases Nat.lt_trichotomy r₁.val r₂.val with h | h | h
    · have : r₁.val * stride_fpbk + stride_fpbk ≤ r₂.val * stride_fpbk := by
        rw [← Nat.succ_mul]
        exact Nat.mul_le_mul_right _ h
      have := hrow c₁
      omega
    · exact h
    · have : r₂.val * stride_fpbk + stride_fpbk ≤ r₁.val * stride_fpbk := by
        rw [← Nat.succ_mul]
        exact Nat.mul_le_mul_right _ h
      have := hrow c₂
      omega
  have hc : c₁.val = c₂.val := by
    have : c₁.val * stride_fpbn = c₂.val * stride_fpbn := by
      rw [hr] at key; omega
    exact Nat.eq_of_mul_eq_mul_right hn this
  simp only [Prod.mk.injEq]
  exact ⟨Fin.ext hr, Fin.ext hc, trivial⟩

/-! ## Termination and per-lane readback

The kernel is an 18-statement straight line: one big `simp` over the
step/eval definitions runs it completely, normalizing `exec` to `some` of a
masked scatter `foldl` over the output store (the masked loads are total —
each carries `other=` — and so is the masked store). The readback lemma then
peels that `foldl` with `scatter_readback_prop_masked_nd`. -/

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any launch state. -/
theorem dq_exec_isSome
    (b_ptr b_zp_ptr : Region .nat) (b_scale_ptr fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn BK BN : Nat)
    (s : BlockState) :
    ∃ s1, exec ((matmul_dequantize_dequantize_surface b_ptr b_scale_ptr b_zp_ptr fpb_ptr
      K N group_size stride_bk stride_bn stride_bsk stride_bsn stride_bzpk
      stride_bzpn stride_fpbk stride_fpbn BK BN).toAlgKernel) s = some s1 := by
  simp [exec, matmul_dequantize_dequantize_surface, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
        Option.bind, Tile.bop, Tile.cop, Tile.expandDim,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, IntegralDType.floorDiv, IntegralDType.mod,
        TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- Per-lane readback: on every lane the store gate lets through, the final
memory holds `dequantSpec` at the lane's `fp_b` address. The proof normalizes
`exec` to the masked scatter `foldl`, peels it with
`scatter_readback_prop_masked_nd` (this is where `hInj` is spent), and
collapses the gated loads with the lane's own bounds facts. -/
theorem dq_correct
    (b_ptr b_zp_ptr : Region .nat) (b_scale_ptr fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn BK BN : Nat)
    (s s' : BlockState)
    (hInj : Function.Injective
      (fun idx : TileIndex [BK, BN] =>
        (s.pids 0 * BK + idx.1.val) * stride_fpbk
          + (s.pids 1 * BN + idx.2.1.val) * stride_fpbn))
    (hExec : exec ((matmul_dequantize_dequantize_surface b_ptr b_scale_ptr b_zp_ptr
      fpb_ptr K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn BK BN).toAlgKernel) s
        = some s') :
    ∀ idx : TileIndex [BK, BN],
      dqActive K N BK BN (s.pids 0) (s.pids 1) idx →
      s'.readMem fpb_ptr
          (fpbAddr stride_fpbk stride_fpbn BK BN (s.pids 0) (s.pids 1) idx)
        = dequantSpec s b_scale_ptr b_ptr b_zp_ptr group_size stride_bk
            stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
            (s.pids 0 * BK + idx.1.val) (s.pids 1 * BN + idx.2.1.val) := by
  intro idx hact
  obtain ⟨hn, hk⟩ := hact
  simp [exec, matmul_dequantize_dequantize_surface, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
        Option.bind, Tile.bop, Tile.cop, Tile.expandDim,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, IntegralDType.floorDiv, IntegralDType.mod,
        TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  simp only [fpbAddr]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj idx]
  rw [if_pos ⟨hn, hk⟩]
  simp [dequantSpec, dqBNibble, dqBWord, dqZpNibble, dqZpWord, dqScElem, hn, hk]

/-! ## Main theorem -/

set_option maxHeartbeats 1600000 in
/-- **Genuine, dimension-general correctness.** For every launch state, the
kernel runs to completion, and every in-range lane `(i, j)` of the program's
`[BK, BN]` output tile — absolute row `k = pid₀·BK + i < K`, absolute column
`n = pid₁·BN + j < N`, exactly the lanes the store mask lets through — holds
the dequantized weight

`fp_b[k, n] = ((dqBWord >> (k % 8)·4) & 0xF − (dqZpWord >> (n % 8)·4) & 0xF) · sc`

as an ℝ equation whose nibble difference is computed in ℤ, where `dqBWord` is
the packed word at `b[k / 8, n]`, `dqZpWord` the packed word at
`zp[k / group_size, n / 8]`, and `sc` the scale at `bs[k / group_size, n]` —
all read from the **launch** state's memory (`dequantSpec`).

Every dimension (`K`, `N`, `group_size`, all eight strides, `BK`, `BN`) and
both program ids are free; the only other hypothesis is `hInj` — distinct
tile lanes must land on distinct `fp_b` cells — which `fpbAddr_injective`
discharges for the host's row-major `fp_b`. No `hundef` is needed: all three
loads carry `other=`, and the store mask equals the load mask. -/
specification matmul_dequantize_dequantize_exec_genuine
    (b_ptr : Region .nat) (b_scale_ptr : RegionName) (b_zp_ptr : Region .nat)
    (fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn : Nat)
    (BK BN : Nat) (s : BlockState)
    (hInj : Function.Injective
      (fun idx : TileIndex [BK, BN] =>
        fpbAddr stride_fpbk stride_fpbn BK BN (s.pids 0) (s.pids 1) idx)) :
    ∃ sF, exec ((matmul_dequantize_dequantize_surface b_ptr b_scale_ptr b_zp_ptr fpb_ptr
        K N group_size stride_bk stride_bn stride_bsk stride_bsn stride_bzpk
        stride_bzpn stride_fpbk stride_fpbn BK BN).toAlgKernel) s = some sF
      ∧ ∀ idx : TileIndex [BK, BN],
          (s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BN + idx.2.1.val < N) →
          sF.readMem fpb_ptr
              (fpbAddr stride_fpbk stride_fpbn BK BN (s.pids 0) (s.pids 1) idx)
            = dequantSpec s b_scale_ptr b_ptr b_zp_ptr group_size stride_bk
                stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
                (s.pids 0 * BK + idx.1.val) (s.pids 1 * BN + idx.2.1.val) := by
  obtain ⟨sF, hExec⟩ := dq_exec_isSome b_ptr b_zp_ptr b_scale_ptr fpb_ptr
    K N group_size stride_bk stride_bn stride_bsk stride_bsn stride_bzpk
    stride_bzpn stride_fpbk stride_fpbn BK BN s
  refine ⟨sF, hExec, fun idx hidx => ?_⟩
  exact dq_correct b_ptr b_zp_ptr b_scale_ptr fpb_ptr K N group_size
    stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
    stride_fpbk stride_fpbn BK BN s sF
    (by simpa [fpbAddr] using hInj) hExec idx ⟨hidx.2, hidx.1⟩

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.MatmulDequantize
