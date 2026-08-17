import VeriTile.Triton

/-!
# `int4_matmul` — strict per-kernel correctness

This file is the DSL port of `int4_matmul.py`'s `matmul_kernel` — the
target JIT, the file's **only** `@triton.jit` kernel (the launcher is
`matmul_dequantize_int4_s2`). It is a
GPTQ-style int4-dequantizing GEMM: `C = A · dequant(B)`, where `B` holds
**eight 4-bit weights packed per 32-bit word along the K axis** and `bzp`
holds eight 4-bit zero-points packed per word along the N axis, with one
`(scale, zero-point)` group row per `group_size` K rows. On every K step the
packed words are unpacked with a shift and a nibble mask, the zero-point
nibble is subtracted **first** (a signed difference), and only then is the
result scaled: `b = ((int_b - int_bzp) * bs)`.

One program owns one `(pid_m, pid_n)` output block, chosen by the standard
group-swizzled `pid` decomposition (`GROUP_SIZE_M` rows of blocks at a time);
`pid_sp_k = tl.program_id(axis=1)` is the SPLIT_K lane.

Translation-surface blocker: three disclosed surface deviations, none
semantic. **(1)** `SPLIT_K` is fixed to `1` (the `tl.store` arm): the
`@triton.autotune` table sweeps `SPLIT_K ∈ {1, 2}` and the launch grid's
second axis is `SPLIT_K`, so the `tl.atomic_add` arm (accumulating into the
host-zeroed `C` of `reset_to_zero=['c_ptr']`) is dropped with the constexpr —
the `bmm_optimized` fixed-arm precedent. Every `* SPLIT_K` factor is folded
to its `SPLIT_K = 1` value, and the headline carries the matching launch fact
`s.pids 1 = 0` (grid axis 1 has extent `SPLIT_K = 1`). **(2)** the K-loop
bound `tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)` is spelled as the antiquoted
binder `numKBlocks` with the honest side condition `K = BLOCK_SIZE_K *
numKBlocks` (the kernel's own docstring asserts `K % (BLOCK_SIZE_K * SPLIT_K)
== 0`, and its loads are unmasked, so the trip count is exact — the
`llama_ff_triton` `tl.cdiv` precedent). **(3)** the dequant subtraction is
spelled `(tl.cast(int_b, tl.int32) - tl.cast(int_bzp, tl.int32)) * bs`: the
two packed channels live on the `.nat` channel (bitwise `>>`/`&` are defined
there), the nibble difference goes negative, and ℕ subtraction truncates — so
the subtraction must run on the signed `.int` channel and promote to ℝ via
`Op.intToReal`. A bare `.to(tl.int32)` on a nat value is width-erased to a
no-op by the inference layer; `tl.cast(x, tl.int32)` is the explicit
`Op.castNatToInt` spelling (the `kcache_copy_triton` `tl.cast` precedent, the
`chunk_retention` disclosed-promotion precedent). The statement's trailing
`.to(a.dtype)` — an erased-identity cast on the ℝ carrier — is dropped from
this one assignment: the DSL's `.to(<ident>.dtype)` wrapper routes any
expression whose raw text mentions `tl.int*` through the int-expected
expansion, which rejects the ℝ `bs` operand; the immediately following
`tl.dot(a, (b).to(a.dtype))` keeps its cast faithfully, so the value channel
is unchanged. This is the **first consumer of `Op.intToReal`**.

## The two packed channels

`qweight` (`b_ptr`) and `qzeros` (`bzp_ptr`) are `torch.IntTensor`s and both
are modelled as `Region .nat`: the DSL's bitwise operators are defined on the
`.nat` channel only. This is a statement about the *container*, not the
extracted values — `(x >> 4i) & 0xF` selects the same nibble whether the
shift is logical or arithmetic, and every extracted nibble is in `[0, 15]`,
hence non-negative. Unlike the `matmul_dequantize_int4` twin (which scales
before subtracting, so its integers never go negative), the *difference* of
two nibbles here is signed — hence the explicit `.int` hop above.

## The wrapped offsets

`offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M` (and the
`% N` sibling) wrap the row/col offsets, and the loads are **unmasked** — an
out-of-range lane reads a wrapped-around row instead of being masked off. The
store mask `(offs_cm < M) & (offs_cn < N)` is over the *unwrapped* offsets,
so on every lane the store actually writes, `i < M` makes the wrap the
identity — which is how the headline's conclusion gets to read the plain
`A[row, ·]` row.

## Proof map

```
int4_matmul_exec_genuine                        the headline
├─ i4_body_eq                 24 statements by `rfl`
├─ i4PreLoopScalars_run       11 statements: pids + three `tl.cdiv`s + swizzle
├─ i4PreLoopTiles_run         → `i4Inv … 0`  (offsets, 2 pointer tiles, zeros)
├─ i4Loop_collapse            `forRange_inv` over `i4Inv`
│  └─ i4LoopBody_run          14 statements: `i4Inv i → i4Inv (i+1)`
│     ├─ i4LoopPtrs_run       the 4 recomputed pointer/shifter tiles
│     ├─ i4LoopLoads_run      the 4 unmasked loads
│     ├─ i4BDequantTile_eq    the nibble / signed-difference / scale statements
│     └─ i4AccTile_dot_succ   `accumulator += tl.dot(a, b)`
└─ i4PostLoop_run             6 statements: output coords, mask, masked store
   └─ i4_store_props          scatter readback  (+ `Nat.mod_eq_of_lt` unwrap)
cAddr_injective                                 discharges the headline's `hInj`
```

The stored value is `accSpec`, built bottom-up from the kernel's own
accessors (`aElem`, `bWord`/`bNibble`, `bsElem`, `bzpWord`/`bzpNibble`) over
the **launch** state's memory. The output region `c_ptr` is never read back
into a spec, so no part of the trust path is self-referential.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` and
the host launch (the grid `(cdiv(M,BM)·cdiv(N,BN), SPLIT_K)`, the block
sizes, `group_size`) are the *trusted boundary*. Every dimension, stride,
block size and `group_size` stays a symbolic parameter.  Integer literals
inside index arithmetic are written `$(n)` (a bare literal is inferred
`.real` by the DSL's expression typing); the source's `0xF` is written
`$(15)` — same value, decimal spelling.  `c = accumulator.to(
c_ptr.dtype.element_ty)` erases to the `.real` carrier (the `attn_fwd`
`element_ty` precedent), and — unlike the twin, which stores `accumulator` —
this kernel genuinely stores `c`.
-/

namespace VeriTile.Bench.TritonBenchG.Int4Matmul

open VeriTile.Triton

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-! ## Kernel surface (faithful transcription, `SPLIT_K = 1` arm) -/

def int4_matmul_surface
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
    b = (tl.cast(int_b, tl.int32) - tl.cast(int_bzp, tl.int32)) * bs
    accumulator += tl.dot(a, (b).to(a.dtype))
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += ($(BLOCK_SIZE_K) * $(stride_bk) // $(8))
  }
  c = (accumulator).to(c_ptr.dtype.element_ty)
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
    ∃ alg, (int4_matmul_surface a_ptr c_ptr bs_ptr b_ptr bzp_ptr
      M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_bsk stride_bsn stride_bzpk stride_bzpn group_size
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks).toAlgorithm?
        = Except.ok alg := by
  simp [int4_matmul_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## The group-swizzled block coordinates

`pid` is decomposed exactly as the source decomposes it, so the spec below is
stated in the same coordinates the kernel computes rather than in a re-derived
form. -/

/-- `num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)`. -/
def numPidM (M BM : Nat) : Nat := (M + BM - 1) / BM

/-- `num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)`. -/
def numPidN (N BN : Nat) : Nat := (N + BN - 1) / BN

/-- `num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)` — a **dead binding** in this kernel
(the loop bound is `tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)`, computed inline);
still transcribed and walked. -/
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


/-! ## Element accessors

Each is the kernel's own address arithmetic over the **launch** state's
memory, parameterized by the *absolute* row / column number, so that the
wrapped (`% M` / `% N`) and unwrapped readings share one definition — the
invariant instantiates them at the wrapped index, the headline (whose store
mask guarantees `row < M`, `col < N`) at the plain one. -/

/-- `a[row, k*BK + e]` at K step `k`. Unmasked, matching the source's bare
`tl.load(a_ptrs)`. -/
noncomputable def aElem (s : BlockState) (a : RegionName)
    (stride_am stride_ak BK : Nat) (row k e : Nat) : ℝ :=
  s.readMem a (row * stride_am + (e + k * BK) * stride_ak)

/-- The packed 32-bit word holding the weight nibble for `(k, e, col)`. Eight
weights share a word along K, hence the `e / 8`. -/
def bWord (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BK : Nat) (k e col : Nat) : Nat :=
  s.readMemValue .nat b ((e / 8 + k * (BK / 8)) * stride_bk + col * stride_bn)

/-- The unpacked 4-bit weight: `(word >> (e % 8) * 4) & 0xF`. -/
def bNibble (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BK : Nat) (k e col : Nat) : Nat :=
  bWord s b stride_bk stride_bn BK k e col >>> (e % 8 * 4) &&& 15

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
noncomputable def bDequant (s : BlockState) (bs : RegionName)
    (b bzp : Region .nat)
    (group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BK : Nat) (k e col : Nat) : ℝ :=
  (((bNibble s b stride_bk stride_bn BK k e col : ℤ)
      - (bzpNibble s bzp group_size stride_bzpk stride_bzpn BK k e col : ℤ) : ℤ) : ℝ)
    * bsElem s bs group_size stride_bsk stride_bsn BK k e col

/-! ## The accumulator -/

/-- One K step's contribution to output cell `(row, col)`. -/
noncomputable def accStep (s : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (group_size stride_am stride_ak stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BK : Nat) (row col k : Nat) : ℝ :=
  ∑ e : Fin BK,
    aElem s a stride_am stride_ak BK row k e.val
      * bDequant s bs b bzp group_size stride_bk stride_bn stride_bsk stride_bsn
          stride_bzpk stride_bzpn BK k e.val col

/-- **The stored value.** `accumulator` after all `numKBlocks` K steps (the
kernel stores `c = accumulator.to(...)`, which erases to `accumulator`). -/
noncomputable def accSpec (s : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (group_size stride_am stride_ak stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BK numKBlocks : Nat) (row col : Nat) : ℝ :=
  ∑ k : Fin numKBlocks,
    accStep s a bs b bzp group_size stride_am stride_ak stride_bk stride_bn
      stride_bsk stride_bsn stride_bzpk stride_bzpn BK row col k.val

/-- The `C` store address for output cell `(r, c)`. -/
def cAddr (stride_cm stride_cn BM BN pm pn : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  stride_cm * (pm * BM + idx.1.val) + stride_cn * (pn * BN + idx.2.1.val)

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
  the two agree exactly because `8 ∣ BK` (the headline's `hBK8`). -/

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
dequant, the `tl.dot` accumulation, and the two pointer advances. -/
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
        Stmt.assign .real [BK, BN] "b"
          (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.intToReal
              (Op.sub .int (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                (Op.castNatToInt (Op.ref .nat [BK, BN] "int_b"))
                (Op.castNatToInt (Op.ref .nat [BK, BN] "int_bzp"))))
            (Op.ref .real [BK, BN] "bs")),
        Stmt.assign .real [BM, BN] "accumulator"
          (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BM, BN] "accumulator")
            (Op.dot (batch := []) (Op.ref .real [BM, BK] "a")
              (Op.ref .real [BK, BN] "b"))),
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

/-- The compiled tail: the (stored!) `c` binding, the two **unwrapped** output
coordinate vectors, the `C` pointer tile, the two-axis store mask, and the
masked store of `c`. -/
def i4PostLoop (c_ptr : RegionName) (M N stride_cm stride_cn BM BN : Nat) :
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
      (Op.ref .real [BM, BN] "c")
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
    (int4_matmul_surface a_ptr c_ptr bs_ptr b_ptr bzp_ptr
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
`(BK * stride_bk) // 8` — the identification with `bWord`'s
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

/-- The `a_ptrs` address agrees with `aElem`'s at the wrapped row. -/
theorem i4AAddr_eq (M stride_am stride_ak BM BK pm k : Nat)
    (idx : TileIndex [BM, BK]) :
    i4AAddr M stride_am stride_ak BM BK pm k idx
      = (pm * BM + idx.1.val) % M * stride_am
        + (idx.2.1.val + k * BK) * stride_ak := by
  simp only [i4AAddr]
  ring

/-- The `b_ptrs` address agrees with `bWord`'s — this is where `8 ∣ BK`
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
  ⟨fun idx => bNibble s b stride_bk stride_bn BK k idx.1.val
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
  ⟨fun idx => some (bDequant s bs b bzp group_size stride_bk stride_bn
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
      accStep s a bs b bzp group_size stride_am stride_ak stride_bk stride_bn
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

/-- At `i = numKBlocks` the accumulator is `accSpec` (at the wrapped lane). -/
theorem i4AccTile_full (s : BlockState) (a bs : RegionName) (b bzp : Region .nat)
    (M N group_size stride_am stride_ak stride_bk stride_bn stride_bsk
      stride_bsn stride_bzpk stride_bzpn BM BN BK pm pn numKBlocks : Nat)
    (idx : TileIndex [BM, BN]) :
    (i4AccTile s a bs b bzp M N group_size stride_am stride_ak stride_bk
        stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BM BN BK pm pn
        numKBlocks).data idx
      = some (accSpec s a bs b bzp group_size stride_am stride_ak stride_bk
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
            accStep s a bs b bzp group_size stride_am stride_ak stride_bk
              stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BK
              ((pm * BM + idx.1.val) % M) ((pn * BN + idx.2.1.val) % N) j.val)
          + accStep s a bs b bzp group_size stride_am stride_ak stride_bk
              stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BK
              ((pm * BM + idx.1.val) % M) ((pn * BN + idx.2.1.val) % N) i) := by
  simp only [i4AccTile]
  exact congrArg some (Fin.sum_univ_castSucc _)

/-! ### Bridges from the tiles to the spec accessors -/

theorem i4ATile_data (s : BlockState) (a : RegionName)
    (M stride_am stride_ak BM BK pm k : Nat) (idx : TileIndex [BM, BK]) :
    (i4ATile s a M stride_am stride_ak BM BK pm k).data idx
      = some (aElem s a stride_am stride_ak BK ((pm * BM + idx.1.val) % M) k
          idx.2.1.val) := by
  simp only [i4ATile, aElem, i4AAddr_eq]

/-- The raw-word tile reads `bWord`'s address — under `8 ∣ BK`. -/
theorem i4BRaw_data (s : BlockState) (b : Region .nat)
    (N stride_bk stride_bn BN BK pn k : Nat) (hBK8 : BK % 8 = 0)
    (idx : TileIndex [BK, BN]) :
    (i4BRaw s b N stride_bk stride_bn BN BK pn k).data idx
      = bWord s b stride_bk stride_bn BK k idx.1.val
          ((pn * BN + idx.2.1.val) % N) := by
  simp only [i4BRaw, bWord, i4BAddr_eq N stride_bk stride_bn BN BK pn k hBK8]

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
  simp only [i4IntB, bNibble, Tile.bop_data, Tile.scalar, i4BShift,
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
  simp only [i4BDequantTile, bDequant, i4NatToInt, i4IntB, i4IntBzp, i4BsTile,
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
partial sum by exactly one `accStep` — at the wrapped lane coordinates. -/
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
  simp [accStep]

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
  -- 11. `b = (tl.cast(int_b, tl.int32) - tl.cast(int_bzp, tl.int32)) * bs`
  have h11 : evalOp (Op.mul .real
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
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h11)]
  -- 12. `accumulator += tl.dot(a, b)`
  have h12 : evalOp (Op.add .real
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "accumulator")
        (Op.dot (batch := []) (Op.ref .real [BM, BK] "a")
          (Op.ref .real [BK, BN] "b")))
        (((t2.setReg "int_b" .nat [BK, BN]
              (i4IntB s0 b N stride_bk stride_bn BN BK pn i)).setReg
            "int_bzp" .nat [BK, BN]
            (i4IntBzp s0 bzp N group_size stride_bzpk stride_bzpn BN BK pn
              i)).setReg
          "b" .real [BK, BN]
          (i4BDequantTile s0 bs b bzp N group_size stride_bk stride_bn
            stride_bsk stride_bsn stride_bzpk stride_bzpn BN BK pn i))
      = some (i4AccTile s0 a bs b bzp M N group_size stride_am stride_ak
          stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
          BM BN BK pm pn (i + 1)) := by
    rw [← i4AccTile_dot_succ]
    exact i4_addTile_eval NumericDType.real _ _ _ _ _ _
      (by rw [evalOp_ref]; simpa using h2acc)
      (i4_dot_eval _ _ _ _ _ (by rw [evalOp_ref]; simpa using h2a)
        (by rw [evalOp_ref]; simp))
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

A masked `.ptr` store: `c_mask` is the only gate. The lane-to-address map has
to be injective for the readback to name a unique lane; that is the headline's
`hInj`, and `cAddr_injective` discharges it for a row-major `C`. -/

/-- The `C` pointer tile. -/
noncomputable def i4CPtrs (c : RegionName) (stride_cm stride_cn BM BN pm pn : Nat) :
    Tile .ptr [BM, BN] :=
  ⟨fun idx => (c, cAddr stride_cm stride_cn BM BN pm pn idx)⟩

/-- `c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)` — over the
**unwrapped** output coordinates. -/
def i4CMask (M N BM BN pm pn : Nat) : Tile .bool [BM, BN] :=
  ⟨fun idx => decide (pm * BM + idx.1.val < M) && decide (pn * BN + idx.2.1.val < N)⟩

/-- The post-store state: one masked scatter over the `[BM, BN]` output tile. -/
noncomputable def i4StoreState (c : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat)
    (f : TileIndex [BM, BN] → ℝ) (t : BlockState) : BlockState :=
  (TileShape.allIndices [BM, BN]).foldl
    (fun acc i => if pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N then
        acc.writeMem c (cAddr stride_cm stride_cn BM BN pm pn i) (f i)
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
`accumulator`). -/
private theorem i4_store_eq (c : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat) (t : BlockState)
    (vt : Tile .real [BM, BN]) (f : TileIndex [BM, BN] → ℝ)
    (hfv : ∀ i, vt.data i = some (f i))
    (hcp : t.regs .ptr [BM, BN] "c_ptrs"
      = some (i4CPtrs c stride_cm stride_cn BM BN pm pn))
    (hcmask : t.regs .bool [BM, BN] "c_mask" = some (i4CMask M N BM BN pm pn))
    (hv : t.regs .real [BM, BN] "c" = some vt) :
    stepStmt (Stmt.store .real [BM, BN]
        (MemAccess.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
        (Op.ref .real [BM, BN] "c")
        (MaskOpt.mask (Op.ref .bool [BM, BN] "c_mask"))) t
      = some (i4StoreState c M N stride_cm stride_cn BM BN pm pn f t) := by
  unfold stepStmt i4StoreState
  simp only [evalOp_ref, hv, hcp, hcmask, Option.map_some]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [BM, BN])) ?_)
  funext acc i
  obtain ⟨r, cc, u⟩ := i
  by_cases hb : pm * BM + r.val < M ∧ pn * BN + cc.val < N
  · simp only [i4CMask, if_pos hb, i4CPtrs, BlockState.writeMemTyped_real, hfv]
    simp [hb.1, hb.2]
  · simp only [i4CMask, if_neg hb, i4CPtrs]
    rw [if_neg]
    intro hcon
    exact hb (by simpa using hcon)

private theorem i4_store_props (c : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat) (t : BlockState)
    (f : TileIndex [BM, BN] → ℝ)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN pm pn i)) :
    ∀ i : TileIndex [BM, BN],
      (pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N) →
      (i4StoreState c M N stride_cm stride_cn BM BN pm pn f t).readMem c
          (cAddr stride_cm stride_cn BM BN pm pn i) = f i := by
  classical
  intro i hi
  unfold i4StoreState
  have h := BlockState.scatter_readback_prop_masked_nd (region := c) t
    (fun j : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN pm pn j) f
    (fun j : TileIndex [BM, BN] => pm * BM + j.1.val < M ∧ pn * BN + j.2.1.val < N)
    hInj i
  rw [h, if_pos hi]

/-- `hInj` for a row-major `C`: distinct output lanes get distinct addresses as
soon as the column stride is positive and one block row fits inside the row
stride. -/
theorem cAddr_injective (stride_cm stride_cn BM BN pm pn : Nat)
    (hcn : 0 < stride_cn) (hfit : BN * stride_cn ≤ stride_cm) :
    Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN pm pn i) := by
  intro i j hij
  obtain ⟨r₁, c₁, u₁⟩ := i
  obtain ⟨r₂, c₂, u₂⟩ := j
  simp only [cAddr] at hij
  have hexp : ∀ r c : Nat, stride_cm * (pm * BM + r) + stride_cn * (pn * BN + c)
      = stride_cm * (pm * BM) + stride_cm * r
        + (stride_cn * (pn * BN) + stride_cn * c) := by
    intro r c
    rw [Nat.mul_add, Nat.mul_add]
  rw [hexp, hexp] at hij
  have key : stride_cm * r₁.val + stride_cn * c₁.val
      = stride_cm * r₂.val + stride_cn * c₂.val := by omega
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

/-! ### The tail

Six statements: the `c` binding (which **is** stored), the two unwrapped
output coordinate vectors, the `C` pointer tile, the two-axis mask, and the
masked store of `c`. On every lane the mask lets through, `row < M` and
`col < N` turn the wrapped accumulator lane into the plain `accSpec` — that is
the `Nat.mod_eq_of_lt` step. -/

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
          sF.readMem c (cAddr stride_cm stride_cn BM BN (pidM s0 M N BM BN GM)
              (pidN s0 M N BM BN GM) idx)
            = accSpec s0 a bs b bzp group_size stride_am stride_ak stride_bk
                stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BK
                numKBlocks (pidM s0 M N BM BN GM * BM + idx.1.val)
                (pidN s0 M N BM BN GM * BN + idx.2.1.val) := by
  obtain ⟨-, -, -, hpm, hpn, -, -, -, -, hacc⟩ := hinv
  unfold i4PostLoop
  -- 1. `c = accumulator.to(c_ptr.dtype.element_ty)` — erased to a plain ref
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BM, BN] "accumulator") t = some _ from
      (evalOp_ref _ _ _ t).trans hacc))]
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
  -- 6. the store of `c`
  rw [stepStmts.cons_some
    (i4_store_eq c M N stride_cm stride_cn BM BN (pidM s0 M N BM BN GM)
      (pidN s0 M N BM BN GM) _ _
      (fun idx => accSpec s0 a bs b bzp group_size stride_am stride_ak stride_bk
        stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BK numKBlocks
        ((pidM s0 M N BM BN GM * BM + idx.1.val) % M)
        ((pidN s0 M N BM BN GM * BN + idx.2.1.val) % N))
      (fun idx => i4AccTile_full s0 a bs b bzp M N group_size stride_am
        stride_ak stride_bk stride_bn stride_bsk stride_bsn stride_bzpk
        stride_bzpn BM BN BK (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM)
        numKBlocks idx)
      (by simp) (by simp) (by simp))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx hidx
  rw [i4_store_props c M N stride_cm stride_cn BM BN
      (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM) _
      (fun idx => accSpec s0 a bs b bzp group_size stride_am stride_ak stride_bk
        stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn BK numKBlocks
        ((pidM s0 M N BM BN GM * BM + idx.1.val) % M)
        ((pidN s0 M N BM BN GM * BN + idx.2.1.val) % N))
      hInj idx hidx,
    Nat.mod_eq_of_lt hidx.1, Nat.mod_eq_of_lt hidx.2]

/-! ## Main theorem -/

set_option maxHeartbeats 1000000 in
-- `hK` is deliberately carried even though only the *loop bound* `numKBlocks`
-- appears semantically (K itself survives only in the dead `num_pid_k`
-- binding): it is the launch fact that makes the `numKBlocks` presentation of
-- `tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)` faithful.
set_option linter.unusedVariables false in
/-- **Genuine, dimension-general correctness.** For every launch state, the
`SPLIT_K = 1` arm of the kernel runs to completion, and every in-range output
lane of `C` holds `accSpec`: the sum over all `numKBlocks` K steps of `tl.dot`
between the loaded `A` tile and the signed-dequantized weight tile
`(nib(B) − nib(BZP)) · BS`, with per-lane group rows `(e + k·BK) / group_size`.

The hypotheses are the kernel's own launch facts: `hK` is the source's
`assert K % (BLOCK_SIZE_K * SPLIT_K) == 0` (the loop trip count `numKBlocks`
is exact — the loads are unmasked); `hBK8` is the source's
`assert BLOCK_SIZE_K % 8 == 0` (eight nibbles per packed word); `hpid1` is
grid axis 1 having extent `SPLIT_K = 1`; `hInj` says distinct output lanes get
distinct `C` addresses — `cAddr_injective` discharges it for a row-major `C`.
The `% M` / `% N` offset wraps disappear on exactly the lanes the store mask
lets through (`Nat.mod_eq_of_lt`). -/
specification int4_matmul_exec_genuine
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
    ∃ sF, exec (int4_matmul_surface a_ptr c_ptr bs_ptr b_ptr bzp_ptr
        M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        stride_bsk stride_bsn stride_bzpk stride_bzpn group_size
        BM BN BK GM numKBlocks).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (pidM s M N BM BN GM * BM + idx.1.val < M
            ∧ pidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.readMem c_ptr (cAddr stride_cm stride_cn BM BN (pidM s M N BM BN GM)
              (pidN s M N BM BN GM) idx)
            = accSpec s a_ptr bs_ptr b_ptr bzp_ptr group_size stride_am
                stride_ak stride_bk stride_bn stride_bsk stride_bsn
                stride_bzpk stride_bzpn BK numKBlocks
                (pidM s M N BM BN GM * BM + idx.1.val)
                (pidN s M N BM BN GM * BN + idx.2.1.val) := by
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

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.Int4Matmul
