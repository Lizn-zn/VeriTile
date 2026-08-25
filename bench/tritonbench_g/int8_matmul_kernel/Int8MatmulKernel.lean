import VeriTile.Triton

/-!
# `int8_matmul_kernel` — strict per-kernel correctness

This file is the DSL port of `int8_matmul_kernel.py`'s `matmul_kernel` — the
file's only JIT kernel (launcher `matmul`, 1-D group-swizzled grid). It is a
**pure-integer** GEMM over 2-bit-packed weights: `A` is `torch.int32`
(`Region .int`), `B` is `torch.uint8` (`Region .nat`) with **four 2-bit
weights packed per byte along K** (`b.shape = [K//4, N]`), and `C` is
`torch.int32` — the store is **`.int`-typed** end to end, no float anywhere.

The nested loop iterates the 2-bit **field index** `i ∈ {0,1,2,3}` outside
and the packed K blocks `j` inside: each inner step extracts field `i` from
the packed byte (`mask = 3 << (2*i)`; `b = ((b_uint8 & mask) >> (2*i))`),
shifts it into signed range by subtracting the all-ones `tensor_full`
(`{0..3} − 1 = {−1,0,1,2}`), and accumulates the exact integer
`tl.dot(a, b − 1, out_dtype=tl.int32)` (`Op.dotInt`). `a_ptrs` advances
**continuously across both loops** (never reset), so `A`'s columns sweep
`0..K−1` in order while `b_ptrs` is rebound to the start of `B` at the top
of each outer iteration (each field re-reads the same `K/4` packed rows).

Translation-surface blocker: four disclosed surface deviations, none
semantic. **(1)** the inner-loop trip count `tl.cdiv(K // 4, BLOCK_SIZE_K)`
(the loop bound and the `k = i * tl.cdiv(K // 4, BLOCK_SIZE_K) + j`
bookkeeping) is spelled as the antiquoted binder `numKBlocks`, with the
honest side condition `hK : K = 4 * (BLOCK_SIZE_K * numKBlocks)` — exactly
the source's own `tl.static_assert(K % (4 * BLOCK_SIZE_K) == 0)`, which is
kept in the surface (it value-erases to a no-op statement). Under `hK`
both load masks are **degenerately all-true** (`offs_k < K - k*BK` and
`offs_k < K`), so raggedness in K is impossible and the loads reduce to
unmasked reads. **(2)** tuple shape arguments are respelled with brackets:
`tl.zeros((BM, BN), …)` → `tl.zeros([$(BM), $(BN)], …)` and
`tl.full((1,), 1, …)` → `tl.full([1], 1, …)` (the audit-known `(1,)` vs
`[1]` shape respell). **(3)** integer **widths are erased** (the
`int8_quantization` fixed-width family): `.to(tl.int8)` on the `A` load is
a no-op on the `.int` channel — the model keeps the full launch-state
int32 value, where int8 hardware would wrap values ≥ 128 (the host test
feeds `A` values in `0..255`); `.to(tl.int8)` on the 2-bit extract lowers
to the genuine signed hop `Op.castNatToInt` (extract values are `{0..3}`,
so no wrap is reachable there); and `tl.full([1], 1, dtype=tl.int8)`
lowers to the width-erased `.int` literal `Op.constInt 1`. **(4)** integer
literals inside index arithmetic are written `$(n)` (a bare literal is
inferred `.real` by the DSL's expression typing).

Faithful spellings worth noting (probed, **not** deviations): the
`out_dtype=tl.int32` kwarg on `tl.dot` is kept and macro-erased (on the
`.int` channel `Op.dotInt` is exact ℤ arithmetic, so the kwarg is
semantically redundant); `b - tensor_full` (`[BK, BN] − [1]`) compiles
through the rank-promoting broadcast `Broadcast.leadR` with no respell;
`tl.program_id(axis=0)` is native syntax; the register named `mask`
shadows nothing (the store mask is the separate `c_mask`). This kernel's
group swizzle carries an extra `% num_pid_in_group` in the `pid_m` line
(`first_pid_m + ((pid % num_pid_in_group) % group_size_m)`) — spelled and
modelled faithfully, coordinates are *not* re-derived.

## The packed-layout spec

For an in-range output cell `(row, col)` the stored int32 is, over ℤ:

`C[row, col] = ∑ i<4, ∑ kk<K/4, A[row, i·(K/4) + kk] · (bits_i(B[kk, col]) − 1)`

where `K/4 = numKBlocks · BLOCK_SIZE_K` and
`bits_i(w) = (w >>> (2·i)) &&& 3` extracts field `i` of the packed byte
(the kernel's `((w & (3 << 2i)) >> 2i)` form is bridged by a `testBit`
identity). `A` is read on the signed `.int` channel, the packed `B` word
on the `.nat` channel, both from the **launch** state — the output `C` is
never read back into the spec, so no part of the trust path is
self-referential.

## Proof map

```
int8_matmul_kernel_exec_genuine                 the headline
├─ im_body_eq                 23 top-level statements by `rfl`
├─ imPreLoopScalars_run       assert no-op + 9 scalars: swizzle → (pid_m, pid_n)
├─ imPreLoopTiles_run         → `imOInv … 0` (wrapped offsets, ptr tiles, zeros)
├─ imLoop_collapse            `forRange_inv` over the OUTER invariant `imOInv`
│  └─ imOuterBody_run         b_ptrs rebind + the collapsed inner loop
│     └─ imInnerBody_run      9 statements (`forRange_inv` over `imIInv`)
│        └─ imAccTile_dotInt_succ   `acc += tl.dot(a, b − 1)` (exact ℤ step)
├─ accVal_final               field-block double sum → the flattened spec
└─ imPostLoop_run             c = acc, ptr/mask tiles, masked `.int` store
   └─ im_store_props          `MemCell`-level `.int` scatter readback
imCAddr_injective                               discharges the headline's `hInj`
```

## Modeling boundary

Arithmetic is exact ℤ / ℕ (no floats, no rounding anywhere in this
kernel); `@triton.autotune` and the host launch (the 1-D grid
`cdiv(M,BM)·cdiv(N,BN)`, block sizes, `GROUP_SIZE_M`) are the *trusted
boundary*. Every dimension, stride and block size stays a symbolic
parameter.
-/

namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel

open VeriTile.Triton

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-! ## Kernel surface (faithful transcription) -/

set_option linter.unusedVariables false in
def int8_matmul_kernel_surface
    (A : Region .int) (B : Region .nat) (C : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  tl.static_assert($(K) % ($(4) * $(BLOCK_SIZE_K)) == 0, "K / 4 must be divisible by BLOCK_SIZE_K => K divisible by 4*BLOCK_SIZE_K")
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = $((A : Region .int)) + (offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak))
  b_ptrs = $((B : Region .nat)) + (offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.int32)
  for i in range($(0), $(4), $(1)) {
    b_ptrs = $((B : Region .nat)) + (offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
    for j in range($(0), $(numKBlocks), $(1)) {
      k = i * $(numKBlocks) + j
      a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0).to(tl.int8)
      b_uint8 = tl.load(b_ptrs, mask=offs_k[:, None] < $(K), other=0)
      mask = $(3) << ($(2) * i)
      b = ((b_uint8 & mask) >> ($(2) * i)).to(tl.int8)
      tensor_full = tl.full([1], 1, dtype=tl.int8)
      accumulator += tl.dot(a, (b - tensor_full), out_dtype=tl.int32)
      a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
      b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
    }
  }
  c = accumulator
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}

/-- The surface lowers to a supported algorithm. -/
theorem int8_matmul_kernel_surface_toAlgorithm_supported
    (A : Region .int) (B : Region .nat) (C : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BM BN BK GM numKBlocks : Nat) :
    ∃ alg, (int8_matmul_kernel_surface A B C M N K
      stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BM BN BK GM numKBlocks).toAlgorithm? = Except.ok alg := by
  simp [int8_matmul_kernel_surface, ComputeExpr.toAlgorithm?]

/-! ## The group-swizzled block coordinates

`pid` is decomposed exactly as the source decomposes it — including this
kernel's extra `% num_pid_in_group` inside the `pid_m` line — so the spec
below is stated in the same coordinates the kernel computes rather than in
a re-derived form. -/

/-- `num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)`. -/
def numPidM (M BM : Nat) : Nat := (M + BM - 1) / BM

/-- `num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)`. -/
def numPidN (N BN : Nat) : Nat := (N + BN - 1) / BN

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

/-- `pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)` —
note the extra `% num_pid_in_group` vs the family's usual
`pid % group_size_m`; spelled faithfully. -/
def pidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  firstPidM s N BN GM
    + s.pids 0 % numPidInGroup N BN GM % groupSizeM s M N BM BN GM

/-- `pid_n = (pid % num_pid_in_group) // group_size_m`. -/
def pidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % numPidInGroup N BN GM / groupSizeM s M N BM BN GM

/-! ## Element accessors

Each is the kernel's own address arithmetic over the **launch** state's
memory, parameterized by the *absolute* row / column / K coordinate, so the
wrapped (`% M` / `% N`) and unwrapped readings share one definition — the
invariant instantiates them at the wrapped index, the headline (whose store
mask guarantees `row < M`, `col < N`) at the plain one. -/

/-- `A[row, kg]` — a signed `.int`-channel read (the host's `torch.int32`
input; widths are erased, see disclosure (3)). -/
def aElem (s : BlockState) (A : Region .int)
    (stride_am stride_ak : Nat) (row kg : Nat) : ℤ :=
  s.readMemValue .int (Region.cast A) (row * stride_am + kg * stride_ak)

/-- `B[kk, col]` — the packed `torch.uint8` byte holding four 2-bit weights,
read on the `.nat` channel (`kk` ranges over the packed rows `0..K/4-1`). -/
def bWord (s : BlockState) (B : Region .nat)
    (stride_bk stride_bn : Nat) (kk col : Nat) : ℕ :=
  s.readMemValue .nat (Region.cast B) (kk * stride_bk + col * stride_bn)

/-- `bits_i(w) = (w >>> (2·i)) &&& 3` — the 2-bit field `i` of a packed
byte, a value in `{0, 1, 2, 3}`. -/
def bBits (i w : Nat) : ℕ := w >>> (2 * i) &&& 3

/-- The kernel's own extract form `(w & (3 << 2i)) >> 2i` **is** `bits_i` —
a `testBit`-level identity. -/
private theorem im_extract_eq (w sh : Nat) :
    (w &&& 3 <<< sh) >>> sh = w >>> sh &&& 3 := by
  apply Nat.eq_of_testBit_eq
  intro n
  simp [Nat.testBit_shiftRight, Nat.testBit_and, Nat.testBit_shiftLeft]

/-! ## The accumulator and the stored value

The accumulator is indexed by the outer field `i` and the inner packed
block `j`. `fieldStep` is one inner iteration's ℤ contribution — the exact
sum-product `Op.dotInt` computes over the loaded `A` tile and the shifted
2-bit extract. The `A` column index is written in the split form
`i·(numKBlocks·BK) + (j·BK + e)` (equal to the pointer's
`(i·numKBlocks + j)·BK + e` by `ring`), which is what lets the per-field
double sum flatten to the packed-layout spec. -/

/-- One inner step's ℤ contribution to output cell `(row, col)`. -/
def fieldStep (s : BlockState) (A : Region .int) (B : Region .nat)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks : Nat)
    (row col i j : Nat) : ℤ :=
  ∑ e : Fin BK,
    aElem s A stride_am stride_ak row (i * (numKBlocks * BK) + (j * BK + e.val))
      * ((bBits i (bWord s B stride_bk stride_bn (j * BK + e.val) col) : ℤ) - 1)

/-- The running sum of field `i` after `j` inner steps. -/
def fieldPartial (s : BlockState) (A : Region .int) (B : Region .nat)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks : Nat)
    (row col i j : Nat) : ℤ :=
  ∑ j' : Fin j,
    fieldStep s A B stride_am stride_ak stride_bk stride_bn BK numKBlocks
      row col i j'.val

/-- The accumulator after completing fields `0..i-1` and `j` inner steps of
field `i`. -/
def accVal (s : BlockState) (A : Region .int) (B : Region .nat)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks : Nat)
    (row col i j : Nat) : ℤ :=
  (∑ i' : Fin i,
      fieldPartial s A B stride_am stride_ak stride_bk stride_bn BK numKBlocks
        row col i'.val numKBlocks)
    + fieldPartial s A B stride_am stride_ak stride_bk stride_bn BK numKBlocks
        row col i j

/-- **The stored value**: the packed-layout double sum
`∑ i<4, ∑ kk<K/4, A[row, i·(K/4)+kk] · (bits_i(B[kk, col]) − 1)` over ℤ,
with `K/4 = numKBlocks · BK` (the `hK` side condition). `A`'s column index
`i·(K/4) + kk` sweeps `0..K-1` in order — the continuously-advancing
`a_ptrs`; `B`'s row index `kk` re-sweeps the packed rows once per field. -/
def imSpec (s : BlockState) (A : Region .int) (B : Region .nat)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks : Nat)
    (row col : Nat) : ℤ :=
  ∑ i : Fin 4, ∑ kk : Fin (numKBlocks * BK),
    aElem s A stride_am stride_ak row (i.val * (numKBlocks * BK) + kk.val)
      * ((bBits i.val (bWord s B stride_bk stride_bn kk.val col) : ℤ) - 1)

theorem accVal_zero (s : BlockState) (A : Region .int) (B : Region .nat)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks row col : Nat) :
    accVal s A B stride_am stride_ak stride_bk stride_bn BK numKBlocks
      row col 0 0 = 0 := by
  simp [accVal, fieldPartial]

/-- One `Op.dotInt` accumulation extends the running sum by one step. -/
theorem accVal_step (s : BlockState) (A : Region .int) (B : Region .nat)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks row col i j : Nat) :
    accVal s A B stride_am stride_ak stride_bk stride_bn BK numKBlocks
        row col i j
      + fieldStep s A B stride_am stride_ak stride_bk stride_bn BK numKBlocks
          row col i j
      = accVal s A B stride_am stride_ak stride_bk stride_bn BK numKBlocks
          row col i (j + 1) := by
  unfold accVal
  have h : fieldPartial s A B stride_am stride_ak stride_bk stride_bn BK
        numKBlocks row col i (j + 1)
      = fieldPartial s A B stride_am stride_ak stride_bk stride_bn BK
          numKBlocks row col i j
        + fieldStep s A B stride_am stride_ak stride_bk stride_bn BK
            numKBlocks row col i j := by
    unfold fieldPartial
    rw [Fin.sum_univ_castSucc]
    simp
  rw [h]
  ring

/-- Rolling over to the next field: a finished inner loop is one completed
field. -/
theorem accVal_roll (s : BlockState) (A : Region .int) (B : Region .nat)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks row col i : Nat) :
    accVal s A B stride_am stride_ak stride_bk stride_bn BK numKBlocks
        row col i numKBlocks
      = accVal s A B stride_am stride_ak stride_bk stride_bn BK numKBlocks
          row col (i + 1) 0 := by
  unfold accVal
  rw [Fin.sum_univ_castSucc]
  have h0 : fieldPartial s A B stride_am stride_ak stride_bk stride_bn BK
      numKBlocks row col (i + 1) 0 = 0 := by simp [fieldPartial]
  rw [h0]
  simp

/-- Block-indexed double sum → flat range sum (the reindex
`kk = j·BK + e`). -/
private theorem im_sum_flatten (g : ℕ → ℤ) (nB BK : Nat) :
    (∑ j : Fin nB, ∑ e : Fin BK, g (j.val * BK + e.val))
      = ∑ n ∈ Finset.range (nB * BK), g n := by
  induction nB with
  | zero => simp
  | succ m ih =>
    rw [Fin.sum_univ_castSucc]
    simp only [Fin.val_castSucc, Fin.val_last]
    rw [ih, show (m + 1) * BK = m * BK + BK from by ring, Finset.sum_range_add]
    congr 1
    rw [Fin.sum_univ_eq_sum_range (fun e => g (m * BK + e)) BK]

/-- After all four fields the accumulator **is** the packed-layout spec. -/
theorem accVal_final (s : BlockState) (A : Region .int) (B : Region .nat)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks row col : Nat) :
    accVal s A B stride_am stride_ak stride_bk stride_bn BK numKBlocks
        row col 4 0
      = imSpec s A B stride_am stride_ak stride_bk stride_bn BK numKBlocks
          row col := by
  unfold accVal imSpec
  have h0 : fieldPartial s A B stride_am stride_ak stride_bk stride_bn BK
      numKBlocks row col 4 0 = 0 := by simp [fieldPartial]
  rw [h0, add_zero]
  refine Finset.sum_congr rfl fun i _ => ?_
  have hflat := im_sum_flatten
    (fun n => aElem s A stride_am stride_ak row (i.val * (numKBlocks * BK) + n)
      * ((bBits i.val (bWord s B stride_bk stride_bn n col) : ℤ) - 1))
    numKBlocks BK
  calc fieldPartial s A B stride_am stride_ak stride_bk stride_bn BK
        numKBlocks row col i.val numKBlocks
      = ∑ j' : Fin numKBlocks, ∑ e : Fin BK,
          (fun n => aElem s A stride_am stride_ak row
              (i.val * (numKBlocks * BK) + n)
            * ((bBits i.val (bWord s B stride_bk stride_bn n col) : ℤ) - 1))
            (j'.val * BK + e.val) := rfl
    _ = ∑ n ∈ Finset.range (numKBlocks * BK),
          aElem s A stride_am stride_ak row (i.val * (numKBlocks * BK) + n)
            * ((bBits i.val (bWord s B stride_bk stride_bn n col) : ℤ) - 1) :=
        hflat
    _ = ∑ kk : Fin (numKBlocks * BK),
          aElem s A stride_am stride_ak row
              (i.val * (numKBlocks * BK) + kk.val)
            * ((bBits i.val (bWord s B stride_bk stride_bn kk.val col) : ℤ)
                - 1) :=
        (Fin.sum_univ_eq_sum_range _ _).symm

/-- The `C` store address for output lane `(r, c)` — the kernel's own
const-first order `stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]`. -/
def imCAddr (stride_cm stride_cn BM BN pm pn : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  stride_cm * (pm * BM + idx.1.val) + stride_cn * (pn * BN + idx.2.1.val)

/-! ## Compiled body decomposition

The algorithm-lowered statement lists, checked against the macro output by
`rfl`. Lowerings worth naming because they are not guessable from the
source text:

* `tl.static_assert` value-erases to the no-op
  `Stmt.ifThen (Op.constBool Bool.false) []` (the condition is
  macro-checked, then discarded);
* `tl.cdiv` expands to `Op.div .nat` on `(X + BX - 1)`; `min(a, b)` is
  `Op.where (Op.lt …) a b` (there is no `Op.min`);
* the `A` load is **`.int`-typed** and its trailing `.to(tl.int8)` erases
  (`.int → .int` widths are one channel); the packed `b_uint8` load is
  **`.nat`-typed**; both are masked-with-`other`
  (`(Op.constInt 0).broadcast` / `(Op.constNat 0).broadcast`), the masks
  compiling through `Op.remap` rank lifts;
* `((b_uint8 & mask) >> (2*i)).to(tl.int8)` compiles to `Op.castNatToInt`
  over `Op.shiftRight` / `Op.bitAnd` — the genuine signed hop;
* `tl.full([1], 1, dtype=tl.int8)` is `Op.full [1] (Op.constInt 1)`, and
  `b - tensor_full` broadcasts `[BK, BN] − [1]` through
  `Broadcast.leadR (Broadcast.consR Broadcast.nil)`;
* `accumulator += tl.dot(a, …, out_dtype=tl.int32)` lowers to
  `Op.add .int` over **`Op.dotInt`** (the kwarg is erased — exact ℤ). -/

/-- The prologue's assert no-op plus nine scalar statements: the program id
and the group swizzle down to `(pid_m, pid_n)`. -/
def imPreLoopScalars (M N BM BN GM : Nat) : List Stmt :=
  [ Stmt.ifThen (Op.constBool Bool.false) [],
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
        (Op.mod IntegralDType.nat Broadcast.nil
          (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
            (Op.ref .nat [] "num_pid_in_group"))
          (Op.ref .nat [] "group_size_m"))),
    Stmt.assign .nat [] "pid_n"
      (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "num_pid_in_group"))
        (Op.ref .nat [] "group_size_m")) ]

/-- The prologue's six index/tile statements: the wrapped offset vectors
(this source wraps inline: `(pid_* * B + arange) % D` is one statement),
`offs_k`, the two typed pointer tiles and the zeroed `.int` accumulator. -/
def imPreLoopTiles (A : Region .int) (B : Region .nat)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK : Nat) : List Stmt :=
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
    Stmt.assign .nat [BK] "offs_k" (Op.arange BK),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
            (Op.constNat stride_am))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_ak)))),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_bk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
            (Op.constNat stride_bn)))),
    Stmt.assign .int [BM, BN] "accumulator" (Op.full [BM, BN] (Op.constInt 0)) ]

/-- The `a`-load mask operand: `offs_k[None, :] < K - k·BLOCK_SIZE_K`,
remapped over the row axis (`k` is the scalar bookkeeping register). -/
def imAMaskOp (K BM BK : Nat) : Op .bool [BM, BK] :=
  Op.remap [BM, BK] Broadcast.nil.consSame.consL.leftIndex
    (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
      (Op.sub .nat Broadcast.nil (Op.constNat K)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK))))

/-- The `b_uint8`-load mask operand: `offs_k[:, None] < K`, remapped over
the column axis (degenerately all-true under `hK`; spelled faithfully). -/
def imBMaskOp (K BK BN : Nat) : Op .bool [BK, BN] :=
  Op.remap [BK, BN] Broadcast.nil.consL.consSame.leftIndex
    (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat K))

/-- The inner-loop body: the `k` bookkeeping, the two masked loads, the
2-bit field extract, the shifted `Op.dotInt` accumulation, and the two
pointer advances. -/
def imInnerBody (K stride_ak stride_bk BM BN BK numKBlocks : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "k"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i") (Op.constNat numKBlocks))
        (Op.ref .nat [] "j")),
    Stmt.assign .int [BM, BK] "a"
      (Op.load .int (MemAccess.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        (MaskOpt.maskOther (imAMaskOp K BM BK)
          ((Op.constInt 0).broadcast [BM, BK]))),
    Stmt.assign .nat [BK, BN] "b_uint8"
      (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BK, BN] "b_ptrs"))
        (MaskOpt.maskOther (imBMaskOp K BK BN)
          ((Op.constNat 0).broadcast [BK, BN]))),
    Stmt.assign .nat [] "mask"
      (Op.shiftLeft Broadcast.nil (Op.constNat 3)
        (Op.mul .nat Broadcast.nil (Op.constNat 2) (Op.ref .nat [] "i"))),
    Stmt.assign .int [BK, BN] "b"
      (Op.castNatToInt
        (Op.shiftRight Broadcast.scalarR
          (Op.bitAnd Broadcast.scalarR (Op.ref .nat [BK, BN] "b_uint8")
            (Op.ref .nat [] "mask"))
          (Op.mul .nat Broadcast.nil (Op.constNat 2) (Op.ref .nat [] "i")))),
    Stmt.assign .int [1] "tensor_full" (Op.full [1] (Op.constInt 1)),
    Stmt.assign .int [BM, BN] "accumulator"
      (Op.add .int (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .int [BM, BN] "accumulator")
        (Op.dotInt (batch := []) (Op.ref .int [BM, BK] "a")
          (Op.sub .int (Broadcast.leadR (Broadcast.consR Broadcast.nil))
            (Op.ref .int [BK, BN] "b") (Op.ref .int [1] "tensor_full")))),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_ak))),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "b_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_bk))) ]

/-- The outer-loop body: the `b_ptrs` rebind to the start of `B` (each
field re-reads the same packed rows) and the collapsed inner loop. -/
def imOuterBody (B : Region .nat)
    (K stride_ak stride_bk stride_bn BM BN BK numKBlocks : Nat) : List Stmt :=
  [ Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_bk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
            (Op.constNat stride_bn)))),
    Stmt.forRange "j" 0 numKBlocks 1
      (imInnerBody K stride_ak stride_bk BM BN BK numKBlocks) ]

/-- The compiled tail: the bare `c = accumulator` re-assign, the fresh
unwrapped `offs_cm` / `offs_cn`, the const-first `c_ptrs` tile, the
two-axis mask, and the masked **`.int`-typed** store of `c`. -/
def imPostLoop (C : RegionName) (M N stride_cm stride_cn BM BN : Nat) : List Stmt :=
  [ Stmt.assign .int [BM, BN] "c" (Op.ref .int [BM, BN] "accumulator"),
    Stmt.assign .nat [BM] "offs_cm"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
        (Op.arange BM)),
    Stmt.assign .nat [BN] "offs_cn"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
        (Op.arange BN)),
    Stmt.assign .ptr [BM, BN] "c_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_cm)
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_cn)
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))),
    Stmt.assign .bool [BM, BN] "c_mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm"))
          (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))
          (Op.constNat N))),
    Stmt.store .int [BM, BN] (MemAccess.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
      (Op.ref .int [BM, BN] "c")
      (MaskOpt.mask (Op.ref .bool [BM, BN] "c_mask")) ]

set_option maxRecDepth 20000 in
set_option linter.unusedVariables false in
/-- **Full body split (by `rfl`).** The lowered surface is exactly
`imPreLoopScalars ++ imPreLoopTiles ++ [forRange "i" 0 4 1 imOuterBody]
++ imPostLoop` — 23 top-level statements, every one checked against the
macro output (the outer body nests `forRange "j" 0 numKBlocks 1
imInnerBody`). -/
theorem im_body_eq (A : Region .int) (B : Region .nat) (C : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BM BN BK GM numKBlocks : Nat) :
    (int8_matmul_kernel_surface A B C M N K
        stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        BM BN BK GM numKBlocks).toAlgKernel.body
      = imPreLoopScalars M N BM BN GM
        ++ imPreLoopTiles A B M N stride_am stride_ak stride_bk stride_bn BM BN BK
        ++ [Stmt.forRange "i" 0 4 1
              (imOuterBody B K stride_ak stride_bk stride_bn BM BN BK numKBlocks)]
        ++ imPostLoop C M N stride_cm stride_cn BM BN := by
  rfl

/-! ## Offset, pointer and value tiles -/

/-- `pid_* * BLOCK + tl.arange(0, BLOCK)` (and `tl.arange` alone at base 0). -/
def imOffs (base BD : Nat) : Tile .nat [BD] := ⟨fun idx => base + idx.1.val⟩

/-- The **wrapped** offset vector `(base + e) % Mm` — `offs_am` / `offs_bn`. -/
def imWrapOffs (base BD Mm : Nat) : Tile .nat [BD] :=
  ⟨fun idx => (base + idx.1.val) % Mm⟩

/-- `a_ptrs` lane `(r, e)` after `kTot` global advances (`kTot` counts
across **both** loops — the pointer is never reset). -/
def imAAddr (M stride_am stride_ak BM BK pm kTot : Nat)
    (idx : TileIndex [BM, BK]) : Nat :=
  (pm * BM + idx.1.val) % M * stride_am + idx.2.1.val * stride_ak
    + kTot * (BK * stride_ak)

/-- `b_ptrs` lane `(e, c)` at inner step `j` (reset per outer iteration). -/
def imBAddr (N stride_bk stride_bn BN BK pn j : Nat)
    (idx : TileIndex [BK, BN]) : Nat :=
  idx.1.val * stride_bk + (pn * BN + idx.2.1.val) % N * stride_bn
    + j * (BK * stride_bk)

noncomputable def imAPtrs (A : Region .int)
    (M stride_am stride_ak BM BK pm kTot : Nat) : Tile .ptr [BM, BK] :=
  ⟨fun idx => (Region.cast A, imAAddr M stride_am stride_ak BM BK pm kTot idx)⟩

noncomputable def imBPtrs (B : Region .nat)
    (N stride_bk stride_bn BN BK pn j : Nat) : Tile .ptr [BK, BN] :=
  ⟨fun idx => (Region.cast B, imBAddr N stride_bk stride_bn BN BK pn j idx)⟩

/-- One `a_ptrs += BLOCK_SIZE_K * stride_ak` advance. -/
theorem imAPtrs_succ (A : Region .int) (M stride_am stride_ak BM BK pm kTot : Nat) :
    Tile.ptrAdd Broadcast.scalarR (imAPtrs A M stride_am stride_ak BM BK pm kTot)
        (Tile.scalar (BK * stride_ak))
      = imAPtrs A M stride_am stride_ak BM BK pm (kTot + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, imAPtrs, imAAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

/-- One `b_ptrs += BLOCK_SIZE_K * stride_bk` advance. -/
theorem imBPtrs_succ (B : Region .nat) (N stride_bk stride_bn BN BK pn j : Nat) :
    Tile.ptrAdd Broadcast.scalarR (imBPtrs B N stride_bk stride_bn BN BK pn j)
        (Tile.scalar (BK * stride_bk))
      = imBPtrs B N stride_bk stride_bn BN BK pn (j + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, imBPtrs, imBAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

/-- The `a_ptrs` address agrees with `aElem`'s at the wrapped row. -/
theorem imAAddr_eq (M stride_am stride_ak BM BK pm kTot : Nat)
    (idx : TileIndex [BM, BK]) :
    imAAddr M stride_am stride_ak BM BK pm kTot idx
      = (pm * BM + idx.1.val) % M * stride_am
        + (kTot * BK + idx.2.1.val) * stride_ak := by
  simp only [imAAddr]
  ring

/-- The `b_ptrs` address agrees with `bWord`'s at the wrapped column. -/
theorem imBAddr_eq (N stride_bk stride_bn BN BK pn j : Nat)
    (idx : TileIndex [BK, BN]) :
    imBAddr N stride_bk stride_bn BN BK pn j idx
      = (j * BK + idx.1.val) * stride_bk
        + (pn * BN + idx.2.1.val) % N * stride_bn := by
  simp only [imBAddr]
  ring

/-- The loaded `a` tile after `kTot` advances — under `hK` the mask is
all-true, so every lane reads the `.int` channel. -/
def imATile (s : BlockState) (A : Region .int)
    (M stride_am stride_ak BM BK pm kTot : Nat) : Tile .int [BM, BK] :=
  ⟨fun idx => aElem s A stride_am stride_ak ((pm * BM + idx.1.val) % M)
    (kTot * BK + idx.2.1.val)⟩

/-- The loaded packed `b_uint8` tile at inner step `j`. -/
def imBWordTile (s : BlockState) (B : Region .nat)
    (N stride_bk stride_bn BN BK pn j : Nat) : Tile .nat [BK, BN] :=
  ⟨fun idx => bWord s B stride_bk stride_bn (j * BK + idx.1.val)
    ((pn * BN + idx.2.1.val) % N)⟩

/-- The signed 2-bit extract `b` for field `i` at inner step `j`. -/
def imBExtractTile (s : BlockState) (B : Region .nat)
    (N stride_bk stride_bn BN BK pn i j : Nat) : Tile .int [BK, BN] :=
  ⟨fun idx => (bBits i (bWord s B stride_bk stride_bn (j * BK + idx.1.val)
    ((pn * BN + idx.2.1.val) % N)) : ℤ)⟩

/-- `accumulator` at outer field `i`, inner step `j`, at the **wrapped**
row / column of each lane. -/
def imAccTile (s : BlockState) (A : Region .int) (B : Region .nat)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK numKBlocks pm pn : Nat)
    (i j : Nat) : Tile .int [BM, BN] :=
  ⟨fun idx => accVal s A B stride_am stride_ak stride_bk stride_bn BK numKBlocks
    ((pm * BM + idx.1.val) % M) ((pn * BN + idx.2.1.val) % N) i j⟩

/-- At `(0, 0)` the accumulator is the `.int` zero tile `tl.zeros` produces. -/
theorem imAccTile_zero (s : BlockState) (A : Region .int) (B : Region .nat)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK numKBlocks pm pn : Nat) :
    imAccTile s A B M N stride_am stride_ak stride_bk stride_bn BM BN BK
        numKBlocks pm pn 0 0
      = (⟨fun _ => 0⟩ : Tile .int [BM, BN]) := by
  apply Tile.ext
  intro idx
  simp [imAccTile, accVal_zero]

/-- A finished inner loop rolls into the next field's step `0`. -/
theorem imAccTile_roll (s : BlockState) (A : Region .int) (B : Region .nat)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK numKBlocks pm pn i : Nat) :
    imAccTile s A B M N stride_am stride_ak stride_bk stride_bn BM BN BK
        numKBlocks pm pn i numKBlocks
      = imAccTile s A B M N stride_am stride_ak stride_bk stride_bn BM BN BK
          numKBlocks pm pn (i + 1) 0 := by
  apply Tile.ext
  intro idx
  simp only [imAccTile]
  rw [accVal_roll]

/-- **The `Op.dotInt` accumulator step.** `accumulator += tl.dot(a, b − 1)`
extends the running sum by one `fieldStep` — at the wrapped lane
coordinates. The `[1]`-shaped `tensor_full` broadcasts through
`Broadcast.leadR`; the pointer index `(i·numKBlocks + j)·BK + e` is
re-split as `i·(numKBlocks·BK) + (j·BK + e)` by `ring`. -/
theorem imAccTile_dotInt_succ (s : BlockState) (A : Region .int) (B : Region .nat)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK numKBlocks pm pn i j : Nat) :
    Tile.bop NumericDType.int.add
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (imAccTile s A B M N stride_am stride_ak stride_bk stride_bn BM BN BK
          numKBlocks pm pn i j)
        (Tile.dotInt []
          (imATile s A M stride_am stride_ak BM BK pm (i * numKBlocks + j))
          (Tile.bop NumericDType.int.sub
            (Broadcast.leadR (Broadcast.consR Broadcast.nil))
            (imBExtractTile s B N stride_bk stride_bn BN BK pn i j)
            (⟨fun _ => 1⟩ : Tile .int [1])))
      = imAccTile s A B M N stride_am stride_ak stride_bk stride_bn BM BN BK
          numKBlocks pm pn i (j + 1) := by
  apply Tile.ext
  intro idx
  obtain ⟨r, c, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, imAccTile,
    NumericDType.int_add]
  -- `erw`: `Tile.dotInt`'s operand shapes are `[] ++ [M, K]`, so
  -- `Tile.dotInt_nil_data` does not fire under `rw` / `simp only`.
  erw [Tile.dotInt_nil_data]
  rw [← accVal_step s A B stride_am stride_ak stride_bk stride_bn BK numKBlocks
    ((pm * BM + r.val) % M) ((pn * BN + c.val) % N) i j]
  congr 1
  unfold fieldStep
  refine Finset.sum_congr rfl fun e _ => ?_
  simp only [imATile, imBExtractTile, Tile.bop_data, Broadcast.leftIndex,
    NumericDType.int_sub]
  rw [show i * (numKBlocks * BK) + (j * BK + e.val)
      = (i * numKBlocks + j) * BK + e.val from by ring]

/-! ## Per-statement eval recipes

Private copies, since bench ports never import each other. `Op.shiftLeft` /
`Op.shiftRight` / `Op.bitAnd` / `Op.castNatToInt` follow the `int4_matmul`
recipes; `Op.dotInt` needs `erw` (dependent `batch ++ [M, K]` operand
shape). -/

/-- What `Op.castNatToInt` evaluates to, as a named combinator. -/
def imNatToInt {sh : TileShape} (t : Tile .nat sh) : Tile .int sh :=
  ⟨fun idx => Int.ofNat (t.data idx)⟩

/-- `Stmt.ifThen` steps by evaluating the guard. -/
private theorem im_ifThen_step (cond : Op .bool []) (body : List Stmt)
    (t : BlockState) :
    stepStmt (Stmt.ifThen cond body) t
      = (evalOp cond t).bind
          (fun c => if c.data PUnit.unit then stepStmts body t else some t) := by
  unfold stepStmt
  cases evalOp cond t <;> rfl

/-- A closed `Bool` guard. -/
private theorem im_constBool_eval (b : Bool) (t : BlockState) :
    evalOp (Op.constBool b) t = some (Tile.scalar b) := by
  simp [evalOp]

/-- The value-erased `tl.static_assert` is a no-op step. -/
private theorem im_assert_step (t : BlockState) :
    stepStmt (Stmt.ifThen (Op.constBool Bool.false) []) t = some t := by
  rw [im_ifThen_step, im_constBool_eval]
  rfl

/-- The `.int` literal. -/
private theorem im_constInt_eval (n : Int) (t : BlockState) :
    evalOp (Op.constInt n) t = some (Tile.scalar n) := by
  simp [evalOp]

/-- `tl.zeros` / `tl.full`. -/
private theorem im_full_eval {dtype : TileDType} (sh : TileShape) (e : Op dtype [])
    (t : BlockState) (v : Tile dtype []) (hv : evalOp e t = some v) :
    evalOp (Op.full sh e) t
      = some (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype sh) := by
  rw [evalOp_full, hv]
  rfl

/-- `%` on the `nat` channel. -/
private theorem im_mod_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mod IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.mod IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `//` on the `nat` channel. -/
private theorem im_floorDiv_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.floorDiv IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.floorDiv IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `Op.div` (what `tl.cdiv` expands to). -/
private theorem im_divTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.div h bc x y) t = some (Tile.bop h.div bc vx vy) := by
  rw [evalOp_div, hx, hy]
  rfl

/-- `<`. -/
private theorem im_ltTile_eval {dtype : TileDType} (h : ComparableDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.lt h bc x y) t = some (Tile.cop h.lt bc vx vy) := by
  rw [evalOp_lt, hx, hy]
  rfl

/-- `tl.where` — how `min(a, b)` is lowered, there being no `Op.min`. -/
private theorem im_where_eval {dtype : TileDType} {sh : TileShape}
    (c : Op .bool sh) (x y : Op dtype sh) (t : BlockState)
    (vc : Tile .bool sh) (vx vy : Tile dtype sh)
    (hc : evalOp c t = some vc) (hx : evalOp x t = some vx)
    (hy : evalOp y t = some vy) :
    evalOp (Op.where c x y) t = some (Tile.select vc vx vy) := by
  rw [evalOp_where, hc, hx, hy]
  rfl

/-- Pointer advance / offset. -/
private theorem im_ptrAdd_eval {a b : TileShape} {out : TileShape}
    (bc : Broadcast a b out)
    (pnm : RegName) (t : BlockState) (pt : Tile .ptr a) (off : Op .nat b)
    (ov : Tile .nat b)
    (hp : t.regs .ptr a pnm = some pt) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ref .ptr a pnm) off) t
      = some (Tile.ptrAdd bc pt ov) := by
  simp only [evalOp, evalOp_ref, hp, ho]
  rfl

/-- A pointer tile built from a bare region base. -/
private theorem im_ptrAddBase_eval {d : TileDType} {b out : TileShape}
    (bc : Broadcast [] b out) (rg : Region d) (t : BlockState)
    (off : Op .nat b) (ov : Tile .nat b) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ptrBase rg) off) t
      = some (Tile.ptrAdd bc (Tile.scalar ((Region.cast rg : RegionName), 0)) ov) := by
  simp only [evalOp, ho]
  rfl

/-- `*` on two tiles. -/
private theorem im_mulTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mul h bc x y) t = some (Tile.bop h.mul bc vx vy) := by
  rw [evalOp_mul, hx, hy]
  rfl

/-- `+` on two tiles (the loop uses it on `.int`, the swizzle on `.nat`). -/
private theorem im_addTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.add h bc x y) t = some (Tile.bop h.add bc vx vy) := by
  rw [evalOp_add, hx, hy]
  rfl

/-- `-` on two tiles (`b - tensor_full` on `.int`; the swizzle on `.nat`). -/
private theorem im_subTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.sub h bc x y) t = some (Tile.bop h.sub bc vx vy) := by
  rw [evalOp_sub, hx, hy]
  rfl

/-- `[:, None]` / `[None, :]`. -/
private theorem im_expandDim_eval {dtype : TileDType} {sh : TileShape}
    (ax : Fin (sh.length + 1)) (x : Op dtype sh) (t : BlockState)
    (v : Tile dtype sh) (hv : evalOp x t = some v) :
    evalOp (Op.expandDim ax x) t = some (Tile.expandDim ax v) := by
  rw [evalOp_expandDim, hv]
  rfl

/-- `&` on the `bool` channel (the store mask). -/
private theorem im_boolAnd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .bool a) (y : Op .bool b) (t : BlockState)
    (vx : Tile .bool a) (vy : Tile .bool b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.boolAnd bc x y) t
      = some (Tile.bop (fun u v : Bool => u && v) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `&` on the `nat` channel — the 2-bit field select. -/
private theorem im_bitAnd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.bitAnd bc x y) t = some (Tile.bop (· &&& ·) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `>>` on the `nat` channel. -/
private theorem im_shiftRight_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.shiftRight bc x y) t = some (Tile.bop (· >>> ·) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `<<` on the `nat` channel — the `mask = 3 << (2*i)` scalar. -/
private theorem im_shiftLeft_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.shiftLeft bc x y) t = some (Tile.bop (· <<< ·) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `.to(tl.int8)` on the `.nat` extract — the explicit nat→int hop. -/
private theorem im_castNatToInt_eval {sh : TileShape} (x : Op .nat sh)
    (t : BlockState) (vx : Tile .nat sh) (hx : evalOp x t = some vx) :
    evalOp (Op.castNatToInt x) t = some (imNatToInt vx) := by
  simp only [evalOp, hx]
  rfl

/-- `Op.remap` (the rank-lift the load masks compile through). -/
private theorem im_remap_eval {dtype : TileDType} {sh : TileShape}
    (outShape : TileShape) (map : TileIndex outShape → TileIndex sh)
    (x : Op dtype sh) (t : BlockState) (v : Tile dtype sh)
    (hv : evalOp x t = some v) :
    evalOp (Op.remap outShape map x) t = some (Tile.remap map v) := by
  simp only [evalOp, hv]
  rfl

/-- A masked-with-`other` `.ptr` load, fully general. -/
private theorem im_load_ptr_maskOther {dtype : TileDType} {sh : TileShape}
    (nm : RegName) (maskOp : Op .bool sh) (otherOp : Op dtype sh)
    (t : BlockState) (pt : Tile .ptr sh) (masks : Tile .bool sh)
    (others : Tile dtype sh)
    (hp : t.regs .ptr sh nm = some pt)
    (hm : evalOp maskOp t = some masks)
    (ho : evalOp otherOp t = some others) :
    evalOp (Op.load dtype (MemAccess.ptr (Op.ref .ptr sh nm))
        (MaskOpt.maskOther maskOp otherOp)) t
      = some (⟨fun i => if masks.data i
            then t.readMemValue dtype (pt.data i).1 (pt.data i).2
            else others.data i⟩ : Tile dtype sh) := by
  simp only [evalOp, evalOp_ref, hp, hm, ho]
  rfl

/-- The broadcast `.int` `other = 0`. -/
private theorem im_broadcastInt_eval (sh : TileShape) (t : BlockState) :
    evalOp ((Op.constInt (0 : Int)).broadcast sh) t
      = some (⟨fun _ => (0 : ℤ)⟩ : Tile .int sh) := by
  simp only [evalOp]
  rfl

/-- The broadcast `.nat` `other = 0`. -/
private theorem im_broadcastNat_eval (sh : TileShape) (t : BlockState) :
    evalOp ((Op.constNat 0).broadcast sh) t
      = some (⟨fun _ => (0 : ℕ)⟩ : Tile .nat sh) := by
  simp only [evalOp]
  rfl

/-- `tl.dot` on the `.int` channel at rank 2. `erw`, not `rw`: the operand
shapes are `[] ++ [M, K]`, which does not unfold at reducible transparency. -/
private theorem im_dotInt_eval {M K N : Nat} (x : Op .int [M, K])
    (y : Op .int [K, N]) (t : BlockState)
    (vx : Tile .int [M, K]) (vy : Tile .int [K, N])
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.dotInt (batch := []) x y) t = some (Tile.dotInt [] vx vy) := by
  erw [evalOp_dotInt, hx, hy]
  rfl

/-! ### `nat` scalar shapes -/

private theorem im_mulScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mul .nat Broadcast.nil x y) t = some (Tile.scalar (u * v)) := by
  rw [evalOp_mul, hx, hy]
  rfl

private theorem im_addScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.add .nat Broadcast.nil x y) t = some (Tile.scalar (u + v)) := by
  rw [evalOp_add, hx, hy]
  rfl

private theorem im_subScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.sub .nat Broadcast.nil x y) t = some (Tile.scalar (u - v)) := by
  rw [evalOp_sub, hx, hy]
  rfl

private theorem im_divScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.div .nat Broadcast.nil x y) t = some (Tile.scalar (u / v)) := by
  rw [im_divTile_eval NumericDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem im_modScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u % v)) := by
  rw [im_mod_eval Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem im_floorDivScalar_eval (x y : Op .nat []) (t : BlockState)
    (u v : Nat) (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u / v)) := by
  rw [im_floorDiv_eval Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem im_ltScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.lt ComparableDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (decide (u < v))) := by
  rw [im_ltTile_eval ComparableDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem im_whereScalarNat_eval (c : Op .bool []) (x y : Op .nat [])
    (t : BlockState) (cv : Bool) (u v : Nat)
    (hc : evalOp c t = some (Tile.scalar cv))
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.where c x y) t = some (Tile.scalar (if cv then u else v)) := by
  rw [im_where_eval c x y t _ _ _ hc hx hy]
  rfl

/-- `name * c` on a `nat` scalar register. -/
private theorem im_mulRef_eval (t : BlockState) (nm : RegName) (val c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c)) t
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `min` is spelled as a `tl.where` by the DSL. -/
private theorem im_min_as_where (u v : Nat) :
    (if u < v then u else v) = min u v := by
  rcases Nat.lt_or_ge u v with h | h
  · rw [if_pos h]; omega
  · rw [if_neg (by omega)]; omega

/-- `setReg` leaves memory alone, at **function** level (the library's
`setReg_mem` is pointwise; a deep tower's `t.mem = s.mem` by one `rfl`
overruns `whnf`). -/
private theorem im_setReg_mem {dtype : TileDType} {sh : TileShape}
    (s : BlockState) (nm : RegName) (v : Tile dtype sh) :
    (s.setReg nm dtype sh v).mem = s.mem := rfl

/-! ### The pid swizzle, statement by statement -/

private theorem im_cdiv_eval (t : BlockState) (X BX : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat X) (Op.constNat BX)) (Op.constNat 1))
        (Op.constNat BX)) t
      = some (Tile.scalar ((X + BX - 1) / BX)) :=
  im_divScalarNat_eval _ _ t (X + BX - 1) BX
    (im_subScalarNat_eval _ _ t (X + BX) 1
      (im_addScalarNat_eval _ _ t X BX (evalOp_constNat _ _) (evalOp_constNat _ _))
      (evalOp_constNat _ _))
    (evalOp_constNat _ _)

private theorem im_width_eval (t : BlockState) (N BN GM : Nat)
    (hgn : t.regs .nat [] "num_pid_n" = some (Tile.scalar (numPidN N BN))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.constNat GM)
        (Op.ref .nat [] "num_pid_n")) t
      = some (Tile.scalar (numPidInGroup N BN GM)) := by
  rw [im_mulScalarNat_eval _ _ t GM (numPidN N BN) (evalOp_constNat _ _)
    (by rw [evalOp_ref]; exact hgn)]
  rfl

private theorem im_groupId_eval (s t : BlockState) (N BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (numPidInGroup N BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "num_pid_in_group")) t
      = some (Tile.scalar (groupId s N BN GM)) := by
  rw [im_floorDivScalar_eval _ _ t (s.pids 0) (numPidInGroup N BN GM)
    (by rw [evalOp_ref]; exact hpid) (by rw [evalOp_ref]; exact hwid)]
  rfl

private theorem im_firstPidM_eval (s t : BlockState) (N BN GM : Nat)
    (hgid : t.regs .nat [] "group_id" = some (Tile.scalar (groupId s N BN GM))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
        (Op.constNat GM)) t
      = some (Tile.scalar (firstPidM s N BN GM)) := by
  rw [im_mulRef_eval t "group_id" (groupId s N BN GM) GM hgid]
  rfl

private theorem im_groupSize_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hgm : t.regs .nat [] "num_pid_m" = some (Tile.scalar (numPidM M BM)))
    (hfp : t.regs .nat [] "first_pid_m"
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
    im_subScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hgm)
      (by rw [evalOp_ref]; exact hfp)
  rw [im_whereScalarNat_eval _ _ _ t
    (decide (numPidM M BM - firstPidM s N BN GM < GM))
    (numPidM M BM - firstPidM s N BN GM) GM
    (im_ltScalarNat_eval _ _ t _ _ hsub (evalOp_constNat _ _)) hsub
    (evalOp_constNat _ _)]
  simp only [decide_eq_true_eq, im_min_as_where, groupSizeM]

private theorem im_pidM_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hfp : t.regs .nat [] "first_pid_m"
      = some (Tile.scalar (firstPidM s N BN GM)))
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (numPidInGroup N BN GM)))
    (hgs : t.regs .nat [] "group_size_m"
      = some (Tile.scalar (groupSizeM s M N BM BN GM))) :
    evalOp (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
        (Op.mod IntegralDType.nat Broadcast.nil
          (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
            (Op.ref .nat [] "num_pid_in_group"))
          (Op.ref .nat [] "group_size_m"))) t
      = some (Tile.scalar (pidM s M N BM BN GM)) := by
  rw [im_addScalarNat_eval _ _ t (firstPidM s N BN GM)
    (s.pids 0 % numPidInGroup N BN GM % groupSizeM s M N BM BN GM)
    (by rw [evalOp_ref]; exact hfp)
    (im_modScalarNat_eval _ _ t _ _
      (im_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
        (by rw [evalOp_ref]; exact hwid))
      (by rw [evalOp_ref]; exact hgs))]
  rfl

private theorem im_pidN_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (numPidInGroup N BN GM)))
    (hgs : t.regs .nat [] "group_size_m"
      = some (Tile.scalar (groupSizeM s M N BM BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "num_pid_in_group"))
        (Op.ref .nat [] "group_size_m")) t
      = some (Tile.scalar (pidN s M N BM BN GM)) := by
  rw [im_floorDivScalar_eval _ _ t (s.pids 0 % numPidInGroup N BN GM)
    (groupSizeM s M N BM BN GM)
    (im_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hwid))
    (by rw [evalOp_ref]; exact hgs)]
  rfl

/-! ### The index tiles -/

/-- `pid_* * BLOCK + tl.arange(0, BLOCK)` from a scalar register. -/
private theorem im_offs_eval (nm : RegName) (t : BlockState) (BD base c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar base)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c))
        (Op.arange BD)) t
      = some (imOffs (base * c) BD) := by
  rw [im_addTile_eval NumericDType.nat Broadcast.scalarL _ _ t
    (Tile.scalar (base * c)) (Tile.vec (fun i => (i.val : Nat)))
    (im_mulScalarNat_eval _ _ t base c (by rw [evalOp_ref]; exact hr)
      (evalOp_constNat _ _))
    (evalOp_arange _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [imOffs, Tile.vec, Broadcast.rightIndex, NumericDType.add]

/-- Bare `tl.arange(0, BK)` — `offs_k`. -/
private theorem im_arange_eval (t : BlockState) (BD : Nat) :
    evalOp (Op.arange BD) t = some (imOffs 0 BD) := by
  rw [evalOp_arange]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [imOffs, Tile.vec]

/-- The single-statement wrapped offset `(pid_* * B + arange) % D`. -/
private theorem im_wrapOffs_eval (nm : RegName) (t : BlockState)
    (BD base c Mm : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar base)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c))
          (Op.arange BD))
        (Op.constNat Mm)) t
      = some (imWrapOffs (base * c) BD Mm) := by
  rw [im_mod_eval Broadcast.scalarR _ _ t (imOffs (base * c) BD) (Tile.scalar Mm)
    (im_offs_eval nm t BD base c hr) (evalOp_constNat _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [imWrapOffs, imOffs, Tile.bop_data, Broadcast.leftIndex,
    Broadcast.rightIndex]

/-- `a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)`
— at global step `0`. -/
private theorem im_aPtrsInit_eval (A : Region .int) (t : BlockState)
    (M stride_am stride_ak BM BK pm : Nat)
    (ham : t.regs .nat [BM] "offs_am" = some (imWrapOffs (pm * BM) BM M))
    (hrk : t.regs .nat [BK] "offs_k" = some (imOffs 0 BK)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
            (Op.constNat stride_am))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_ak)))) t
      = some (imAPtrs A M stride_am stride_ak BM BK pm 0) := by
  rw [im_ptrAddBase_eval _ _ t _ _
    (im_addTile_eval NumericDType.nat _ _ _ t _ _
      (im_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_am)
        (im_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact ham))
        (evalOp_constNat _ _))
      (im_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_ak)
        (im_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrk))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [imAPtrs, imAAddr, imWrapOffs, imOffs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)`
— inner step `0` (the prologue init **and** the per-field rebind). -/
private theorem im_bPtrsInit_eval (B : Region .nat) (t : BlockState)
    (N stride_bk stride_bn BN BK pn : Nat)
    (hrk : t.regs .nat [BK] "offs_k" = some (imOffs 0 BK))
    (hrbn : t.regs .nat [BN] "offs_bn" = some (imWrapOffs (pn * BN) BN N)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_bk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
            (Op.constNat stride_bn)))) t
      = some (imBPtrs B N stride_bk stride_bn BN BK pn 0) := by
  rw [im_ptrAddBase_eval _ _ t _ _
    (im_addTile_eval NumericDType.nat _ _ _ t _ _
      (im_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_bk)
        (im_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrk))
        (evalOp_constNat _ _))
      (im_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_bn)
        (im_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrbn))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [imBPtrs, imBAddr, imWrapOffs, imOffs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-! ### The degenerate load masks

Under the static assert's `hK : K = 4·(BK·numKBlocks)` every mask lane is
true: `k = i·numKBlocks + j` with `i < 4`, `j < numKBlocks` gives
`(k+1)·BK ≤ K` for the `a` mask, and `BK ≤ K` for the `b` mask. -/

private theorem im_aMask_true (K BK numKBlocks i j e : Nat)
    (hK : K = 4 * (BK * numKBlocks)) (hi : i < 4) (hj : j < numKBlocks)
    (he : e < BK) :
    e < K - (i * numKBlocks + j) * BK := by
  have hle : i * numKBlocks + j + 1 ≤ 4 * numKBlocks := by
    have h3 : i * numKBlocks ≤ 3 * numKBlocks :=
      Nat.mul_le_mul_right numKBlocks (by omega)
    omega
  have hbound : (i * numKBlocks + j) * BK + BK ≤ K := by
    calc (i * numKBlocks + j) * BK + BK = (i * numKBlocks + j + 1) * BK := by
          ring
      _ ≤ 4 * numKBlocks * BK := Nat.mul_le_mul_right BK hle
      _ = K := by rw [hK]; ring
  omega

private theorem im_bMask_true (K BK numKBlocks j e : Nat)
    (hK : K = 4 * (BK * numKBlocks)) (hj : j < numKBlocks) (he : e < BK) :
    e < K := by
  have h1 : BK * 1 ≤ BK * numKBlocks := Nat.mul_le_mul_left BK (by omega)
  omega

/-- The `a`-load mask evaluates to the per-lane `e < K - k·BK` guard. -/
private theorem im_aMask_eval (t : BlockState) (K BM BK c : Nat)
    (hk : t.regs .nat [BK] "offs_k" = some (imOffs 0 BK))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar c)) :
    evalOp (imAMaskOp K BM BK) t
      = some (⟨fun idx => decide (idx.2.1.val < K - c * BK)⟩
          : Tile .bool [BM, BK]) := by
  unfold imAMaskOp
  rw [im_remap_eval _ _ _ t _
    (im_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ _
      (im_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hk))
      (im_subScalarNat_eval _ _ t K (c * BK) (evalOp_constNat _ _)
        (im_mulRef_eval t "k" c BK hkk)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp [Tile.remap, Tile.cop_data, Tile.expandDim_data, imOffs,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]

/-- The `b_uint8`-load mask evaluates to the per-lane `e < K` guard. -/
private theorem im_bMask_eval (t : BlockState) (K BK BN : Nat)
    (hk : t.regs .nat [BK] "offs_k" = some (imOffs 0 BK)) :
    evalOp (imBMaskOp K BK BN) t
      = some (⟨fun idx => decide (idx.1.val < K)⟩ : Tile .bool [BK, BN]) := by
  unfold imBMaskOp
  rw [im_remap_eval _ _ _ t _
    (im_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ _
      (im_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hk))
      (evalOp_constNat _ _))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨e, cc, u⟩ := idx
  simp [Tile.remap, Tile.cop_data, Tile.expandDim_data, imOffs,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]

/-! ### The typed loads, bridged to the named tiles -/

/-- The masked `.int` `a` load lands on the unmasked `imATile` (the mask is
all-true under `hK`), on the launch state's memory. -/
private theorem im_aLoad_eq (s0 : BlockState) (A : Region .int) (t : BlockState)
    (M K stride_am stride_ak BM BK numKBlocks pm i j : Nat)
    (hK : K = 4 * (BK * numKBlocks)) (hi : i < 4) (hj : j < numKBlocks)
    (hmem : t.mem = s0.mem)
    (hap : t.regs .ptr [BM, BK] "a_ptrs"
      = some (imAPtrs A M stride_am stride_ak BM BK pm (i * numKBlocks + j)))
    (hk : t.regs .nat [BK] "offs_k" = some (imOffs 0 BK))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar (i * numKBlocks + j))) :
    evalOp (Op.load .int (MemAccess.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        (MaskOpt.maskOther (imAMaskOp K BM BK)
          ((Op.constInt 0).broadcast [BM, BK]))) t
      = some (imATile s0 A M stride_am stride_ak BM BK pm (i * numKBlocks + j)) := by
  rw [im_load_ptr_maskOther "a_ptrs" (imAMaskOp K BM BK) _ t _ _ _ hap
    (im_aMask_eval t K BM BK (i * numKBlocks + j) hk hkk)
    (im_broadcastInt_eval [BM, BK] t)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [imATile, imAPtrs, decide_eq_true_eq]
  rw [if_pos (im_aMask_true K BK numKBlocks i j e.val hK hi hj e.isLt)]
  rw [imAAddr_eq]
  simp only [aElem, BlockState.readMemValue, BlockState.readMemTyped, hmem]

/-- The masked `.nat` `b_uint8` load lands on the unmasked `imBWordTile`. -/
private theorem im_bLoad_eq (s0 : BlockState) (B : Region .nat) (t : BlockState)
    (N K stride_bk stride_bn BN BK numKBlocks pn j : Nat)
    (hK : K = 4 * (BK * numKBlocks)) (hj : j < numKBlocks)
    (hmem : t.mem = s0.mem)
    (hbp : t.regs .ptr [BK, BN] "b_ptrs"
      = some (imBPtrs B N stride_bk stride_bn BN BK pn j))
    (hk : t.regs .nat [BK] "offs_k" = some (imOffs 0 BK)) :
    evalOp (Op.load .nat (MemAccess.ptr (Op.ref .ptr [BK, BN] "b_ptrs"))
        (MaskOpt.maskOther (imBMaskOp K BK BN)
          ((Op.constNat 0).broadcast [BK, BN]))) t
      = some (imBWordTile s0 B N stride_bk stride_bn BN BK pn j) := by
  rw [im_load_ptr_maskOther "b_ptrs" (imBMaskOp K BK BN) _ t _ _ _ hbp
    (im_bMask_eval t K BK BN hk) (im_broadcastNat_eval [BK, BN] t)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨e, cc, u⟩ := idx
  simp only [imBWordTile, imBPtrs, decide_eq_true_eq]
  rw [if_pos (im_bMask_true K BK numKBlocks j e.val hK hj e.isLt)]
  rw [imBAddr_eq]
  simp only [bWord, BlockState.readMemValue, BlockState.readMemTyped, hmem]

/-! ### The in-loop scalars and the extract -/

/-- `k = i * numKBlocks + j`. -/
private theorem im_k_eval (t : BlockState) (numKBlocks i j : Nat)
    (hi : t.regs .nat [] "i" = some (Tile.scalar i))
    (hj : t.regs .nat [] "j" = some (Tile.scalar j)) :
    evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i") (Op.constNat numKBlocks))
        (Op.ref .nat [] "j")) t
      = some (Tile.scalar (i * numKBlocks + j)) := by
  rw [im_addScalarNat_eval _ _ t (i * numKBlocks) j
    (im_mulRef_eval t "i" i numKBlocks hi) (by rw [evalOp_ref]; exact hj)]

/-- `mask = 3 << (2 * i)`. -/
private theorem im_maskShift_eval (t : BlockState) (i : Nat)
    (hi : t.regs .nat [] "i" = some (Tile.scalar i)) :
    evalOp (Op.shiftLeft Broadcast.nil (Op.constNat 3)
        (Op.mul .nat Broadcast.nil (Op.constNat 2) (Op.ref .nat [] "i"))) t
      = some (Tile.scalar (3 <<< (2 * i))) := by
  rw [im_shiftLeft_eval Broadcast.nil _ _ t (Tile.scalar 3) (Tile.scalar (2 * i))
    (evalOp_constNat _ _)
    (im_mulScalarNat_eval _ _ t 2 i (evalOp_constNat _ _)
      (by rw [evalOp_ref]; exact hi))]
  rfl

/-- `b = ((b_uint8 & mask) >> (2*i)).to(tl.int8)` lands on the signed 2-bit
extract tile — the `(w & (3 << s)) >> s = bits` bridge happens per lane. -/
private theorem im_bExtract_eval (s0 : BlockState) (B : Region .nat)
    (t : BlockState) (N stride_bk stride_bn BN BK pn i j : Nat)
    (hbu : t.regs .nat [BK, BN] "b_uint8"
      = some (imBWordTile s0 B N stride_bk stride_bn BN BK pn j))
    (hmask : t.regs .nat [] "mask" = some (Tile.scalar (3 <<< (2 * i))))
    (hi : t.regs .nat [] "i" = some (Tile.scalar i)) :
    evalOp (Op.castNatToInt
        (Op.shiftRight Broadcast.scalarR
          (Op.bitAnd Broadcast.scalarR (Op.ref .nat [BK, BN] "b_uint8")
            (Op.ref .nat [] "mask"))
          (Op.mul .nat Broadcast.nil (Op.constNat 2) (Op.ref .nat [] "i")))) t
      = some (imBExtractTile s0 B N stride_bk stride_bn BN BK pn i j) := by
  rw [im_castNatToInt_eval _ t _
    (im_shiftRight_eval Broadcast.scalarR _ _ t _ (Tile.scalar (2 * i))
      (im_bitAnd_eval Broadcast.scalarR _ _ t _ (Tile.scalar (3 <<< (2 * i)))
        (by rw [evalOp_ref]; exact hbu) (by rw [evalOp_ref]; exact hmask))
      (im_mulScalarNat_eval _ _ t 2 i (evalOp_constNat _ _)
        (by rw [evalOp_ref]; exact hi)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨e, cc, u⟩ := idx
  simp only [imNatToInt, imBExtractTile, imBWordTile, Tile.bop_data,
    Broadcast.leftIndex, Tile.scalar, bBits, Int.ofNat_eq_natCast]
  rw [im_extract_eq]

/-! ## The nested loop invariants

The inner invariant (fixed field `i`) carries exactly the registers the
inner body reads or the tail still needs: the loop counter register `"i"`
(the `k` bookkeeping and the shift read it), the block coordinates, the
offset vectors (`offs_k` for the masks, `offs_bn` for the next rebind),
the two pointers and the accumulator. `j ≤ numKBlocks` pins the collapse's
final index. The outer invariant is the inner one at a field boundary
(`b_ptrs` unconstrained — the rebind reestablishes it; `"i"` unconstrained
— the combinator rewrites it). -/

/-- The state carried across inner steps of field `i`. -/
noncomputable def imIInv (s0 : BlockState) (A : Region .int) (B : Region .nat)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK GM numKBlocks i : Nat)
    (j : Nat) (s : BlockState) : Prop :=
  j ≤ numKBlocks
  ∧ s.mem = s0.mem
  ∧ s.pids = s0.pids
  ∧ s.regs .nat [] "i" = some (Tile.scalar i)
  ∧ s.regs .nat [] "pid_m" = some (Tile.scalar (pidM s0 M N BM BN GM))
  ∧ s.regs .nat [] "pid_n" = some (Tile.scalar (pidN s0 M N BM BN GM))
  ∧ s.regs .nat [BK] "offs_k" = some (imOffs 0 BK)
  ∧ s.regs .nat [BN] "offs_bn"
      = some (imWrapOffs (pidN s0 M N BM BN GM * BN) BN N)
  ∧ s.regs .ptr [BM, BK] "a_ptrs"
      = some (imAPtrs A M stride_am stride_ak BM BK (pidM s0 M N BM BN GM)
          (i * numKBlocks + j))
  ∧ s.regs .ptr [BK, BN] "b_ptrs"
      = some (imBPtrs B N stride_bk stride_bn BN BK (pidN s0 M N BM BN GM) j)
  ∧ s.regs .int [BM, BN] "accumulator"
      = some (imAccTile s0 A B M N stride_am stride_ak stride_bk stride_bn
          BM BN BK numKBlocks (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM) i j)

/-- The state carried across outer fields. -/
noncomputable def imOInv (s0 : BlockState) (A : Region .int) (B : Region .nat)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK GM numKBlocks : Nat)
    (i : Nat) (s : BlockState) : Prop :=
  i ≤ 4
  ∧ s.mem = s0.mem
  ∧ s.pids = s0.pids
  ∧ s.regs .nat [] "pid_m" = some (Tile.scalar (pidM s0 M N BM BN GM))
  ∧ s.regs .nat [] "pid_n" = some (Tile.scalar (pidN s0 M N BM BN GM))
  ∧ s.regs .nat [BK] "offs_k" = some (imOffs 0 BK)
  ∧ s.regs .nat [BN] "offs_bn"
      = some (imWrapOffs (pidN s0 M N BM BN GM * BN) BN N)
  ∧ s.regs .ptr [BM, BK] "a_ptrs"
      = some (imAPtrs A M stride_am stride_ak BM BK (pidM s0 M N BM BN GM)
          (i * numKBlocks))
  ∧ s.regs .int [BM, BN] "accumulator"
      = some (imAccTile s0 A B M N stride_am stride_ak stride_bk stride_bn
          BM BN BK numKBlocks (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM) i 0)

/-- The inner combinator writes `"j"` before each iteration; `imIInv`
constrains no register named `"j"`. -/
theorem imIInv_setReg_j (s0 : BlockState) (A : Region .int) (B : Region .nat)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK GM numKBlocks i j v : Nat)
    (s : BlockState)
    (h : imIInv s0 A B M N stride_am stride_ak stride_bk stride_bn BM BN BK GM
      numKBlocks i j s) :
    imIInv s0 A B M N stride_am stride_ak stride_bk stride_bn BM BN BK GM
      numKBlocks i j (s.setReg "j" .nat [] (Tile.scalar v)) := by
  obtain ⟨hle, hmem, hpids, hi, hpm, hpn, hk, hbn, hA, hB, hacc⟩ := h
  exact ⟨hle, hmem, hpids, by simpa using hi, by simpa using hpm,
    by simpa using hpn, by simpa using hk, by simpa using hbn,
    by simpa using hA, by simpa using hB, by simpa using hacc⟩

/-- The outer combinator writes `"i"` before each iteration; `imOInv`
constrains no register named `"i"`. -/
theorem imOInv_setReg_i (s0 : BlockState) (A : Region .int) (B : Region .nat)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK GM numKBlocks i v : Nat)
    (s : BlockState)
    (h : imOInv s0 A B M N stride_am stride_ak stride_bk stride_bn BM BN BK GM
      numKBlocks i s) :
    imOInv s0 A B M N stride_am stride_ak stride_bk stride_bn BM BN BK GM
      numKBlocks i (s.setReg "i" .nat [] (Tile.scalar v)) := by
  obtain ⟨hle, hmem, hpids, hpm, hpn, hk, hbn, hA, hacc⟩ := h
  exact ⟨hle, hmem, hpids, by simpa using hpm, by simpa using hpn,
    by simpa using hk, by simpa using hbn, by simpa using hA,
    by simpa using hacc⟩

/-! ### The inner step -/

theorem imInnerBody_run (s0 : BlockState) (A : Region .int) (B : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn BM BN BK GM numKBlocks i j : Nat)
    (s : BlockState)
    (hK : K = 4 * (BK * numKBlocks)) (hi : i < 4)
    (hnext : j + 1 ≤ numKBlocks)
    (hjreg : s.regs .nat [] "j" = some (Tile.scalar j))
    (hinv : imIInv s0 A B M N stride_am stride_ak stride_bk stride_bn BM BN BK
      GM numKBlocks i j s) :
    ∃ s', stepStmts (imInnerBody K stride_ak stride_bk BM BN BK numKBlocks) s
        = some s'
      ∧ imIInv s0 A B M N stride_am stride_ak stride_bk stride_bn BM BN BK GM
          numKBlocks i (j + 1) s' := by
  obtain ⟨-, hmem, hpids, hireg, hpm, hpn, hoffk, hoffbn, hA, hB, hacc⟩ := hinv
  set pm := pidM s0 M N BM BN GM with hpmDef
  set pn := pidN s0 M N BM BN GM with hpnDef
  unfold imInnerBody
  -- 1. `k = i * numKBlocks + j`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_k_eval s numKBlocks i j hireg hjreg))]
  -- 2. `a = tl.load(a_ptrs, mask=…, other=0).to(tl.int8)` — all-true mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_aLoad_eq s0 A _ M K stride_am stride_ak BM BK numKBlocks pm i j
      hK hi (by omega) (by simpa [im_setReg_mem] using hmem)
      (by simpa using hA) (by simpa using hoffk) (by simp)))]
  -- 3. `b_uint8 = tl.load(b_ptrs, mask=…, other=0)` — all-true mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_bLoad_eq s0 B _ N K stride_bk stride_bn BN BK numKBlocks pn j
      hK (by omega) (by simpa [im_setReg_mem] using hmem)
      (by simpa using hB) (by simpa using hoffk)))]
  -- 4. `mask = 3 << (2 * i)`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_maskShift_eval _ i (by simpa using hireg)))]
  -- 5. `b = ((b_uint8 & mask) >> (2 * i)).to(tl.int8)`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_bExtract_eval s0 B _ N stride_bk stride_bn BN BK pn i j
      (by simp) (by simp) (by simpa using hireg)))]
  -- 6. `tensor_full = tl.full([1], 1, dtype=tl.int8)`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [1] (Op.constInt 1)) _
        = some ((⟨fun _ => 1⟩ : Tile .int [1])) from by
      rw [im_full_eval [1] (Op.constInt 1) _ _ (im_constInt_eval 1 _)]
      rfl))]
  -- 7. `accumulator += tl.dot(a, (b - tensor_full), out_dtype=tl.int32)`
  have h7 : evalOp (Op.add .int
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .int [BM, BN] "accumulator")
        (Op.dotInt (batch := []) (Op.ref .int [BM, BK] "a")
          (Op.sub .int (Broadcast.leadR (Broadcast.consR Broadcast.nil))
            (Op.ref .int [BK, BN] "b") (Op.ref .int [1] "tensor_full"))))
        ((((((s.setReg "k" .nat [] (Tile.scalar (i * numKBlocks + j))).setReg
            "a" .int [BM, BK]
            (imATile s0 A M stride_am stride_ak BM BK pm
              (i * numKBlocks + j))).setReg
            "b_uint8" .nat [BK, BN]
            (imBWordTile s0 B N stride_bk stride_bn BN BK pn j)).setReg
            "mask" .nat [] (Tile.scalar (3 <<< (2 * i)))).setReg
            "b" .int [BK, BN]
            (imBExtractTile s0 B N stride_bk stride_bn BN BK pn i j)).setReg
            "tensor_full" .int [1] (⟨fun _ => 1⟩ : Tile .int [1]))
      = some (imAccTile s0 A B M N stride_am stride_ak stride_bk stride_bn
          BM BN BK numKBlocks pm pn i (j + 1)) := by
    rw [← imAccTile_dotInt_succ s0 A B M N stride_am stride_ak stride_bk
      stride_bn BM BN BK numKBlocks pm pn i j]
    exact im_addTile_eval NumericDType.int _ _ _ _ _ _
      (by rw [evalOp_ref]; simpa using hacc)
      (im_dotInt_eval _ _ _ _ _
        (by rw [evalOp_ref]; simp)
        (im_subTile_eval NumericDType.int _ _ _ _ _ _
          (by rw [evalOp_ref]; simp) (by rw [evalOp_ref]; simp)))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h7)]
  -- 8. `a_ptrs += BLOCK_SIZE_K * stride_ak`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_ak))) _
      = some (imAPtrs A M stride_am stride_ak BM BK pm
          (i * numKBlocks + j + 1)) from by
      rw [← imAPtrs_succ]
      exact im_ptrAdd_eval Broadcast.scalarR "a_ptrs" _ _ _
        (Tile.scalar (BK * stride_ak)) (by simpa using hA)
        (im_mulScalarNat_eval _ _ _ BK stride_ak (evalOp_constNat _ _)
          (evalOp_constNat _ _))))]
  -- 9. `b_ptrs += BLOCK_SIZE_K * stride_bk`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "b_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_bk))) _
      = some (imBPtrs B N stride_bk stride_bn BN BK pn (j + 1)) from by
      rw [← imBPtrs_succ]
      exact im_ptrAdd_eval Broadcast.scalarR "b_ptrs" _ _ _
        (Tile.scalar (BK * stride_bk)) (by simpa using hB)
        (im_mulScalarNat_eval _ _ _ BK stride_bk (evalOp_constNat _ _)
          (evalOp_constNat _ _))))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, hnext, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [im_setReg_mem]
    exact hmem
  · simp only [BlockState.setReg_pids]
    exact hpids
  · simpa using hireg
  · simpa using hpm
  · simpa using hpn
  · simpa using hoffk
  · simpa using hoffbn
  · rw [show i * numKBlocks + (j + 1) = i * numKBlocks + j + 1 from by omega]
    simp [hpmDef]
  · simp [hpnDef]
  · simp [hpmDef, hpnDef]

/-! ### One outer field -/

theorem imOuterBody_run (s0 : BlockState) (A : Region .int) (B : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn BM BN BK GM numKBlocks i : Nat)
    (s : BlockState)
    (hK : K = 4 * (BK * numKBlocks)) (hnext : i + 1 ≤ 4)
    (hireg : s.regs .nat [] "i" = some (Tile.scalar i))
    (hinv : imOInv s0 A B M N stride_am stride_ak stride_bk stride_bn BM BN BK
      GM numKBlocks i s) :
    ∃ s', stepStmts (imOuterBody B K stride_ak stride_bk stride_bn BM BN BK
        numKBlocks) s = some s'
      ∧ imOInv s0 A B M N stride_am stride_ak stride_bk stride_bn BM BN BK GM
          numKBlocks (i + 1) s' := by
  obtain ⟨-, hmem, hpids, hpm, hpn, hoffk, hoffbn, hA, hacc⟩ := hinv
  unfold imOuterBody
  -- 1. the `b_ptrs` rebind to the start of `B`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_bPtrsInit_eval B _ N stride_bk stride_bn BN BK (pidN s0 M N BM BN GM)
      hoffk hoffbn))]
  -- 2. the collapsed inner loop
  have h0 : imIInv s0 A B M N stride_am stride_ak stride_bk stride_bn BM BN BK
      GM numKBlocks i 0
      (s.setReg "b_ptrs" .ptr [BK, BN]
        (imBPtrs B N stride_bk stride_bn BN BK (pidN s0 M N BM BN GM) 0)) := by
    refine ⟨Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp only [im_setReg_mem]
      exact hmem
    · simp only [BlockState.setReg_pids]
      exact hpids
    · simpa using hireg
    · simpa using hpm
    · simpa using hpn
    · simpa using hoffk
    · simpa using hoffbn
    · rw [show i * numKBlocks + 0 = i * numKBlocks from by omega]
      simpa using hA
    · simp
    · simpa using hacc
  obtain ⟨final, sF, hrun, hfinal, hP⟩ :=
    forRange_inv (idx := "j") (start := 0) (stop := numKBlocks) (step := 1)
      (P := fun j t => imIInv s0 A B M N stride_am stride_ak stride_bk
        stride_bn BM BN BK GM numKBlocks i j t)
      one_ne_zero h0
      (fun j t hj hinvj => by
        obtain ⟨s', hs', hinv'⟩ :=
          imInnerBody_run s0 A B M N K stride_am stride_ak stride_bk stride_bn
            BM BN BK GM numKBlocks i j _ hK (by omega) (by omega) (by simp)
            (imIInv_setReg_j s0 A B M N stride_am stride_ak stride_bk stride_bn
              BM BN BK GM numKBlocks i j j t hinvj)
        exact ⟨s', hs', hinv'⟩)
  -- `rw … at hP`, not `subst`: `subst` would eliminate the *binder*
  -- `numKBlocks` (the RHS variable), breaking every later mention of it.
  have hEq : final = numKBlocks := le_antisymm hP.1 hfinal
  rw [hEq] at hP
  rw [stepStmts.cons_some hrun, stepStmts.nil]
  obtain ⟨-, hFmem, hFpids, hFireg, hFpm, hFpn, hFoffk, hFoffbn, hFA, -, hFacc⟩ := hP
  refine ⟨sF, rfl, hnext, hFmem, hFpids, hFpm, hFpn, hFoffk, hFoffbn, ?_, ?_⟩
  · rw [show (i + 1) * numKBlocks = i * numKBlocks + numKBlocks from by ring]
    exact hFA
  · rw [← imAccTile_roll]
    exact hFacc

/-! ### Collapsing the outer loop -/

theorem imLoop_collapse (s0 : BlockState) (A : Region .int) (B : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn BM BN BK GM numKBlocks : Nat)
    (s : BlockState)
    (hK : K = 4 * (BK * numKBlocks))
    (h0 : imOInv s0 A B M N stride_am stride_ak stride_bk stride_bn BM BN BK
      GM numKBlocks 0 s) :
    ∃ sF, stepStmt (Stmt.forRange "i" 0 4 1
          (imOuterBody B K stride_ak stride_bk stride_bn BM BN BK numKBlocks)) s
        = some sF
      ∧ imOInv s0 A B M N stride_am stride_ak stride_bk stride_bn BM BN BK GM
          numKBlocks 4 sF := by
  obtain ⟨final, sF, hrun, hfinal, hP⟩ :=
    forRange_inv (idx := "i") (start := 0) (stop := 4) (step := 1)
      (P := fun i t => imOInv s0 A B M N stride_am stride_ak stride_bk
        stride_bn BM BN BK GM numKBlocks i t)
      one_ne_zero h0
      (fun i t hi hinv => by
        obtain ⟨s', hs', hinv'⟩ :=
          imOuterBody_run s0 A B M N K stride_am stride_ak stride_bk stride_bn
            BM BN BK GM numKBlocks i _ hK (by omega) (by simp)
            (imOInv_setReg_i s0 A B M N stride_am stride_ak stride_bk stride_bn
              BM BN BK GM numKBlocks i i t hinv)
        exact ⟨s', hs', hinv'⟩)
  have hEq : final = 4 := le_antisymm hP.1 hfinal
  subst hEq
  exact ⟨sF, hrun, hP⟩

/-! ## The prologue walks -/

/-- The assert no-op plus the nine scalar statements. Memory is untouched;
the registers anything downstream reads are the block coordinates. -/
theorem imPreLoopScalars_run (s : BlockState) (M N BM BN GM : Nat) :
    ∃ t, stepStmts (imPreLoopScalars M N BM BN GM) s = some t
      ∧ t.mem = s.mem
      ∧ t.pids = s.pids
      ∧ t.regs .nat [] "pid_m" = some (Tile.scalar (pidM s M N BM BN GM))
      ∧ t.regs .nat [] "pid_n" = some (Tile.scalar (pidN s M N BM BN GM)) := by
  unfold imPreLoopScalars
  rw [stepStmts.cons_some (im_assert_step s)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (im_cdiv_eval _ M BM))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (im_cdiv_eval _ N BN))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_width_eval _ N BN GM (by simp [numPidN])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_groupId_eval s _ N BN GM (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_firstPidM_eval s _ N BN GM (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_groupSize_eval s _ M N BM BN GM (by simp [numPidM]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_pidM_eval s _ M N BM BN GM (by simp) (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_pidN_eval s _ M N BM BN GM (by simp) (by simp) (by simp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_⟩ <;> simp [im_setReg_mem]

/-- The six index/tile statements, ending on `imOInv` at field `0`. -/
theorem imPreLoopTiles_run (s0 : BlockState) (A : Region .int) (B : Region .nat)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK GM numKBlocks : Nat)
    (t : BlockState)
    (hmem : t.mem = s0.mem)
    (hpids : t.pids = s0.pids)
    (hpm : t.regs .nat [] "pid_m" = some (Tile.scalar (pidM s0 M N BM BN GM)))
    (hpn : t.regs .nat [] "pid_n" = some (Tile.scalar (pidN s0 M N BM BN GM))) :
    ∃ t', stepStmts (imPreLoopTiles A B M N stride_am stride_ak stride_bk
        stride_bn BM BN BK) t = some t'
      ∧ imOInv s0 A B M N stride_am stride_ak stride_bk stride_bn BM BN BK GM
          numKBlocks 0 t' := by
  unfold imPreLoopTiles
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_wrapOffs_eval "pid_m" t BM (pidM s0 M N BM BN GM) BM M hpm))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_wrapOffs_eval "pid_n" _ BN (pidN s0 M N BM BN GM) BN N
      (by simpa using hpn)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (im_arange_eval _ BK))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_aPtrsInit_eval A _ M stride_am stride_ak BM BK (pidM s0 M N BM BN GM)
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_bPtrsInit_eval B _ N stride_bk stride_bn BN BK (pidN s0 M N BM BN GM)
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM, BN] (Op.constInt 0)) _
        = some (imAccTile s0 A B M N stride_am stride_ak stride_bk stride_bn
            BM BN BK numKBlocks (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM)
            0 0) from by
      rw [imAccTile_zero]
      exact im_full_eval [BM, BN] (Op.constInt 0) _ _ (im_constInt_eval 0 _)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [im_setReg_mem]
    exact hmem
  · simpa using hpids
  · simpa using hpm
  · simpa using hpn
  · simp
  · simp
  · simp
  · simp

/-! ## Memory infrastructure for the `.int` store tail

Mem-cell level, dtype-generic where possible; the `.int` scatter readback
takes a mask-restricted injectivity (discharged here by the headline's full
`hInj`). -/

/-- `writeMemTyped` at any other cell. -/
private theorem im_writeMemTyped_mem_other {dtype : TileDType} (s : BlockState)
    (rg : RegionName) (a : Nat) (v : TileCarrier dtype) (ρ : RegionName) (o : Nat)
    (h : ¬(rg = ρ ∧ a = o)) :
    (s.writeMemTyped dtype rg a v).mem ρ o = s.mem ρ o := by
  cases dtype <;>
    · show (if ρ = rg ∧ o = a then _ else s.mem ρ o) = s.mem ρ o
      rw [if_neg (fun hc => h ⟨hc.1.symm, hc.2.symm⟩)]

/-- `writeMemTyped .int` at its own cell. -/
private theorem im_writeMemTyped_int_mem_self (s : BlockState)
    (rg : RegionName) (a : Nat) (v : Int) :
    (s.writeMemTyped .int rg a v).mem rg a = MemCell.of .int v := by
  show (if rg = rg ∧ a = a then MemCell.of .int v else s.mem rg a)
    = MemCell.of .int v
  rw [if_pos ⟨rfl, rfl⟩]

/-- Cell-level frame for a masked scatter foldl: every cell missed by all
active writes is unchanged. -/
private theorem im_foldl_write_mem_preserve {dtype : TileDType} {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → TileCarrier dtype)
    (P : α → Prop) [DecidablePred P] (ρ : RegionName) (o : Nat) (l : List α) :
    ∀ s : BlockState, (∀ k ∈ l, P k → ¬(region = ρ ∧ offsetFn k = o)) →
      ((l.foldl (fun acc k =>
          if P k then acc.writeMemTyped dtype region (offsetFn k) (valueFn k)
          else acc) s)).mem ρ o = s.mem ρ o := by
  induction l with
  | nil => intro s _; rfl
  | cons hd tl ih =>
      intro s h
      rw [List.foldl_cons]
      have htl : ∀ k ∈ tl, P k → ¬(region = ρ ∧ offsetFn k = o) :=
        fun k hk => h k (List.mem_cons_of_mem hd hk)
      by_cases hP : P hd
      · rw [if_pos hP, ih _ htl,
          im_writeMemTyped_mem_other _ _ _ _ _ _ (h hd List.mem_cons_self hP)]
      · rw [if_neg hP]
        exact ih _ htl

/-- `.int` scatter readback with mask-restricted injectivity: at an active
lane's cell, the foldl leaves exactly that lane's `MemCell.of .int` write. -/
private theorem im_scatter_int_mem {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → Int) (P : TileIndex shape → Prop)
    [DecidablePred P]
    (h_inj : ∀ k₁ k₂, P k₁ → P k₂ → offsetFn k₁ = offsetFn k₂ → k₁ = k₂)
    (i : TileIndex shape) (hPi : P i) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k)
         else acc) s).mem region (offsetFn i)
      = MemCell.of .int (valueFn i) := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  rw [show TileShape.allIndices shape = l₁ ++ i :: l₂ from hl, List.foldl_append,
    List.foldl_cons]
  have h_l2 : ∀ k ∈ l₂, P k → ¬(region = region ∧ offsetFn k = offsetFn i) := by
    intro k hk hPk hc
    have hki : k = i := h_inj k i hPk hPi hc.2
    subst hki
    exact hi_notin_l2 hk
  rw [im_foldl_write_mem_preserve (dtype := .int) region offsetFn valueFn P
    region (offsetFn i) l₂ _ h_l2]
  rw [if_pos hPi]
  exact im_writeMemTyped_int_mem_self _ _ _ _

/-! ## The output store -/

/-- The `c_ptrs` pointer tile. -/
noncomputable def imCPtrs (C : RegionName)
    (stride_cm stride_cn BM BN pm pn : Nat) : Tile .ptr [BM, BN] :=
  ⟨fun idx => (C, imCAddr stride_cm stride_cn BM BN pm pn idx)⟩

/-- `c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)` — over the
**unwrapped** output coordinates. -/
def imCMask (M N BM BN pm pn : Nat) : Tile .bool [BM, BN] :=
  ⟨fun idx => decide (pm * BM + idx.1.val < M) && decide (pn * BN + idx.2.1.val < N)⟩

/-- The post-store state: one masked **`.int`-typed** scatter over the
`[BM, BN]` output tile. -/
noncomputable def imStoreState (C : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat)
    (f : TileIndex [BM, BN] → ℤ) (t : BlockState) : BlockState :=
  (TileShape.allIndices [BM, BN]).foldl
    (fun acc i => if pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N then
        acc.writeMemTyped .int C (imCAddr stride_cm stride_cn BM BN pm pn i) (f i)
      else acc) t

/-- `c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]`
— strides on the **left** of the products (the kernel's own order). -/
private theorem im_cPtrsInit_eval (C : RegionName) (t : BlockState)
    (stride_cm stride_cn BM BN pm pn : Nat)
    (hcm : t.regs .nat [BM] "offs_cm" = some (imOffs (pm * BM) BM))
    (hcn : t.regs .nat [BN] "offs_cn" = some (imOffs (pn * BN) BN)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_cm)
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_cn)
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) t
      = some (imCPtrs C stride_cm stride_cn BM BN pm pn) := by
  rw [im_ptrAddBase_eval _ _ t _ _
    (im_addTile_eval NumericDType.nat _ _ _ t _ _
      (im_mulTile_eval NumericDType.nat Broadcast.scalarL _ _ t
        (Tile.scalar stride_cm) _
        (evalOp_constNat _ _)
        (im_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcm)))
      (im_mulTile_eval NumericDType.nat Broadcast.scalarL _ _ t
        (Tile.scalar stride_cn) _
        (evalOp_constNat _ _)
        (im_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcn))))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [imCPtrs, imCAddr, imOffs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `c_mask` — the comparisons happen at rank 2 (the unit axes are inserted
**inside** the `lt` operands). -/
private theorem im_cMask_eval (t : BlockState) (M N BM BN pm pn : Nat)
    (hcm : t.regs .nat [BM] "offs_cm" = some (imOffs (pm * BM) BM))
    (hcn : t.regs .nat [BN] "offs_cn" = some (imOffs (pn * BN) BN)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm"))
          (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))
          (Op.constNat N))) t
      = some (imCMask M N BM BN pm pn) := by
  rw [im_boolAnd_eval (Broadcast.consR (Broadcast.consL Broadcast.nil)) _ _ t _ _
    (im_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _
      (Tile.scalar M)
      (im_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcm))
      (evalOp_constNat _ _))
    (im_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _
      (Tile.scalar N)
      (im_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcn))
      (evalOp_constNat _ _))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [imCMask, imOffs, Tile.bop_data, Tile.cop_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]

/-- The masked `.int` store of the **`c` register** steps to the named
scatter state. -/
private theorem im_store_eq (C : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat) (t : BlockState)
    (vt : Tile .int [BM, BN]) (f : TileIndex [BM, BN] → ℤ)
    (hfv : ∀ i, vt.data i = f i)
    (hcp : t.regs .ptr [BM, BN] "c_ptrs"
      = some (imCPtrs C stride_cm stride_cn BM BN pm pn))
    (hcmask : t.regs .bool [BM, BN] "c_mask" = some (imCMask M N BM BN pm pn))
    (hv : t.regs .int [BM, BN] "c" = some vt) :
    stepStmt (Stmt.store .int [BM, BN]
        (MemAccess.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
        (Op.ref .int [BM, BN] "c")
        (MaskOpt.mask (Op.ref .bool [BM, BN] "c_mask"))) t
      = some (imStoreState C M N stride_cm stride_cn BM BN pm pn f t) := by
  unfold stepStmt imStoreState
  simp only [evalOp_ref, hv, hcp, hcmask, Option.map_some]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [BM, BN])) ?_)
  funext acc i
  obtain ⟨r, cc, u⟩ := i
  by_cases hb : pm * BM + r.val < M ∧ pn * BN + cc.val < N
  · rw [if_pos (show (imCMask M N BM BN pm pn).data (r, cc, u) = Bool.true from by
      simp only [imCMask, Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨hb.1, hb.2⟩)]
    rw [if_pos hb, hfv]
    rfl
  · rw [if_neg (show ¬((imCMask M N BM BN pm pn).data (r, cc, u) = Bool.true) from by
      simp only [imCMask, Bool.and_eq_true, decide_eq_true_eq]
      exact fun hc => hb hc)]
    rw [if_neg hb]

/-- `MemCell`-level readback of the masked `.int` scatter on every active
lane. -/
private theorem im_store_props (C : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat) (t : BlockState)
    (f : TileIndex [BM, BN] → ℤ)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => imCAddr stride_cm stride_cn BM BN pm pn i)) :
    ∀ i : TileIndex [BM, BN],
      (pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N) →
      (imStoreState C M N stride_cm stride_cn BM BN pm pn f t).mem C
          (imCAddr stride_cm stride_cn BM BN pm pn i)
        = MemCell.of .int (f i) := by
  classical
  intro i hi
  unfold imStoreState
  exact im_scatter_int_mem t
    (fun j : TileIndex [BM, BN] => imCAddr stride_cm stride_cn BM BN pm pn j) f
    (fun j : TileIndex [BM, BN] => pm * BM + j.1.val < M ∧ pn * BN + j.2.1.val < N)
    (fun k₁ k₂ _ _ h => hInj h) i hi

/-- `hInj` for a row-major `C`, in the kernel's const-first address form:
distinct output lanes get distinct addresses as soon as the column stride is
positive and one block row fits inside the row stride. -/
theorem imCAddr_injective (stride_cm stride_cn BM BN pm pn : Nat)
    (hcn : 0 < stride_cn) (hfit : BN * stride_cn ≤ stride_cm) :
    Function.Injective
      (fun i : TileIndex [BM, BN] => imCAddr stride_cm stride_cn BM BN pm pn i) := by
  intro i j hij
  obtain ⟨r₁, c₁, u₁⟩ := i
  obtain ⟨r₂, c₂, u₂⟩ := j
  simp only [imCAddr] at hij
  have hexp : ∀ r c : Nat, stride_cm * (pm * BM + r) + stride_cn * (pn * BN + c)
      = pm * BM * stride_cm + r * stride_cm
        + (pn * BN * stride_cn + c * stride_cn) := by
    intro r c
    ring
  rw [hexp, hexp] at hij
  have key : r₁.val * stride_cm + c₁.val * stride_cn
      = r₂.val * stride_cm + c₂.val * stride_cn := by omega
  have hrow : ∀ c : Fin BN, c.val * stride_cn < stride_cm := fun c =>
    lt_of_lt_of_le (Nat.mul_lt_mul_of_lt_of_le c.isLt (le_refl _) hcn) hfit
  have hr : r₁.val = r₂.val := by
    rcases Nat.lt_trichotomy r₁.val r₂.val with h | h | h
    · have : r₁.val * stride_cm + stride_cm ≤ r₂.val * stride_cm := by
        rw [← Nat.succ_mul]
        exact Nat.mul_le_mul_right _ h
      have := hrow c₁
      omega
    · exact h
    · have : r₂.val * stride_cm + stride_cm ≤ r₁.val * stride_cm := by
        rw [← Nat.succ_mul]
        exact Nat.mul_le_mul_right _ h
      have := hrow c₂
      omega
  have hc : c₁.val = c₂.val := by
    have : c₁.val * stride_cn = c₂.val * stride_cn := by
      rw [hr] at key; omega
    exact Nat.eq_of_mul_eq_mul_right hcn this
  simp only [Prod.mk.injEq]
  exact ⟨Fin.ext hr, Fin.ext hc, trivial⟩

/-! ### The tail

Six statements: the bare `c = accumulator` re-assign, the fresh unwrapped
`offs_cm` / `offs_cn`, the const-first `c_ptrs` tile, the two-axis mask,
and the masked `.int` store of `c`. On every lane the mask lets through,
`row < M` and `col < N` turn the wrapped lane into the plain coordinates
(`Nat.mod_eq_of_lt`) and `accVal_final` flattens the field-block double sum
into the packed-layout `imSpec`. -/

theorem imPostLoop_run (s0 : BlockState) (A : Region .int) (B : Region .nat)
    (C : RegionName)
    (M N stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BM BN BK GM numKBlocks : Nat) (t : BlockState)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => imCAddr stride_cm stride_cn BM BN
        (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM) i))
    (hinv : imOInv s0 A B M N stride_am stride_ak stride_bk stride_bn BM BN BK
      GM numKBlocks 4 t) :
    ∃ sF, stepStmts (imPostLoop C M N stride_cm stride_cn BM BN) t = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (pidM s0 M N BM BN GM * BM + idx.1.val < M
            ∧ pidN s0 M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem C (imCAddr stride_cm stride_cn BM BN (pidM s0 M N BM BN GM)
              (pidN s0 M N BM BN GM) idx)
            = MemCell.of .int (imSpec s0 A B stride_am stride_ak stride_bk
                stride_bn BK numKBlocks
                (pidM s0 M N BM BN GM * BM + idx.1.val)
                (pidN s0 M N BM BN GM * BN + idx.2.1.val)) := by
  obtain ⟨-, hmem, -, hpm, hpn, -, -, -, hacc⟩ := hinv
  set pm := pidM s0 M N BM BN GM with hpmDef
  set pn := pidN s0 M N BM BN GM with hpnDef
  unfold imPostLoop
  -- 1. `c = accumulator`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .int [BM, BN] "accumulator") t
        = some (imAccTile s0 A B M N stride_am stride_ak stride_bk stride_bn
            BM BN BK numKBlocks pm pn 4 0) from by
      rw [evalOp_ref]; exact hacc))]
  -- 2-3. `offs_cm` / `offs_cn` (fresh, unwrapped)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_offs_eval "pid_m" _ BM pm BM (by simpa using hpm)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_offs_eval "pid_n" _ BN pn BN (by simpa using hpn)))]
  -- 4-5. the `c_ptrs` tile and the two-axis mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_cPtrsInit_eval C _ stride_cm stride_cn BM BN pm pn (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (im_cMask_eval _ M N BM BN pm pn (by simp) (by simp)))]
  -- 6. the masked `.int` store of `c`
  rw [stepStmts.cons_some
    (im_store_eq C M N stride_cm stride_cn BM BN pm pn _
      (imAccTile s0 A B M N stride_am stride_ak stride_bk stride_bn BM BN BK
        numKBlocks pm pn 4 0)
      (fun ix => accVal s0 A B stride_am stride_ak stride_bk stride_bn BK
        numKBlocks ((pm * BM + ix.1.val) % M) ((pn * BN + ix.2.1.val) % N) 4 0)
      (fun _ => rfl)
      (by simp) (by simp) (by simp))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx hidx
  rw [im_store_props C M N stride_cm stride_cn BM BN pm pn _ _ hInj idx hidx]
  rw [Nat.mod_eq_of_lt hidx.1, Nat.mod_eq_of_lt hidx.2, accVal_final]

/-! ## Main theorem -/

set_option maxHeartbeats 1000000 in
set_option linter.unusedVariables false in
/-- **Genuine, dimension-general correctness.** For every launch state
satisfying the source's own `tl.static_assert` (`hK`: `K` is divisible by
`4·BLOCK_SIZE_K`, presented as `K = 4·(BK·numKBlocks)` with `numKBlocks`
the inner trip count `tl.cdiv(K//4, BK)`), the kernel runs to completion
and every in-range output lane of `C` holds the `.int` memory cell carrying
the exact ℤ packed-weight matmul

`imSpec = ∑ i<4, ∑ kk<K/4, A[row, i·(K/4)+kk] · (bits_i(B[kk, col]) − 1)`

— the four 2-bit fields of each packed byte of `B`, shifted into
`{−1,0,1,2}` by the `tensor_full` subtraction, dotted against the
continuously-swept columns of `A` (`Op.dotInt` over all `4·numKBlocks`
inner steps). No rounding model appears anywhere: the kernel is
pure-integer, and the store readback is stated at the `MemCell` level
(`MemCell.of .int`).

`hInj` says distinct output lanes get distinct `C` addresses —
`imCAddr_injective` discharges it for a row-major `C`. The `% M` / `% N`
offset wraps disappear on exactly the lanes the store mask lets through
(`Nat.mod_eq_of_lt`). The grid is 1-D, so there is no auxiliary
program-id hypothesis. -/
specification int8_matmul_kernel_exec_genuine
    (A : Region .int) (B : Region .nat) (C : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BM BN BK GM numKBlocks : Nat) (s : BlockState)
    (hK : K = 4 * (BK * numKBlocks))
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => imCAddr stride_cm stride_cn BM BN
        (pidM s M N BM BN GM) (pidN s M N BM BN GM) i)) :
    ∃ sF, exec (int8_matmul_kernel_surface A B C M N K
        stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        BM BN BK GM numKBlocks).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (pidM s M N BM BN GM * BM + idx.1.val < M
            ∧ pidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem C (imCAddr stride_cm stride_cn BM BN (pidM s M N BM BN GM)
              (pidN s M N BM BN GM) idx)
            = MemCell.of .int (imSpec s A B stride_am stride_ak stride_bk
                stride_bn BK numKBlocks
                (pidM s M N BM BN GM * BM + idx.1.val)
                (pidN s M N BM BN GM * BN + idx.2.1.val)) := by
  rw [exec, im_body_eq]
  -- prologue: the scalars, then the index tiles
  obtain ⟨t1, hrun1, h1mem, h1pids, h1pm, h1pn⟩ :=
    imPreLoopScalars_run s M N BM BN GM
  obtain ⟨t2, hrun2, h2inv⟩ :=
    imPreLoopTiles_run s A B M N stride_am stride_ak stride_bk stride_bn
      BM BN BK GM numKBlocks t1 h1mem h1pids h1pm h1pn
  simp only [List.append_assoc]
  rw [stepStmts.append_some hrun1, stepStmts.append_some hrun2]
  -- the collapsed nested loop
  obtain ⟨t3, hrun3, h3inv⟩ :=
    imLoop_collapse s A B M N K stride_am stride_ak stride_bk stride_bn
      BM BN BK GM numKBlocks t2 hK h2inv
  rw [show [Stmt.forRange "i" 0 4 1
          (imOuterBody B K stride_ak stride_bk stride_bn BM BN BK numKBlocks)]
        ++ imPostLoop C M N stride_cm stride_cn BM BN
      = Stmt.forRange "i" 0 4 1
          (imOuterBody B K stride_ak stride_bk stride_bn BM BN BK numKBlocks)
        :: imPostLoop C M N stride_cm stride_cn BM BN
      from rfl]
  rw [stepStmts.cons_some hrun3]
  -- the tail
  obtain ⟨sF, hpost, hout⟩ :=
    imPostLoop_run s A B C M N stride_am stride_ak stride_bk stride_bn
      stride_cm stride_cn BM BN BK GM numKBlocks t3 hInj h3inv
  exact ⟨sF, hpost, hout⟩

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.Int8MatmulKernel
