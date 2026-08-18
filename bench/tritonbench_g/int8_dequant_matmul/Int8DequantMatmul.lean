import VeriTile.Triton

/-!
# `int8_dequant_matmul` — strict per-kernel correctness

This file is the DSL port of `int8_dequant_matmul.py`'s
`_int8_matmul_rowwise_dequantize` — the file's only JIT kernel (the launcher
is `int8_matmul_rowwise_dequantize`). It is an int8×int8→int32 GEMM with a
rowwise-dequantization epilogue: `A` and `B` hold **signed 8-bit** integers,
the K loop accumulates the exact ℤ matmul through the integer `tl.dot`
(`Op.dotInt` — this port is that operator's **first consumer**), and the
epilogue rescales the integer accumulator per output cell:
`acc = w_factor · (x_factor · (acc · divfactor))` with the per-row `state_x`
scale, the per-column `state_w` scale and the host's
`divfactor = 1/(127·127)`, then downcasts to fp16 and (under `has_bias`)
adds a per-column fp16 bias before the masked store into `C`.

One program owns one `(pid_m, pid_n)` output block, chosen by the standard
group-swizzled `pid` decomposition (`GROUP_M` rows of blocks at a time);
`pid_z = tl.program_id(1)` is the SPLIT_K lane.

Translation-surface blocker: five disclosed surface deviations, none
semantic. **(1)** `SPLIT_K` is fixed to `1` (the `tl.store` arm): the
`@triton.autotune` table sweeps `SPLIT_K ∈ {1, 2, 4, 8, 16}` and the launch
grid's second axis is `SPLIT_K`, so the `tl.atomic_add` arm (accumulating
into the host-zeroed `C` of `pre_hook=init_to_zero("C")`) is dropped with
the constexpr — the `int4_matmul` disclosure-(1) precedent. Every
`* SPLIT_K` factor is folded to its `SPLIT_K = 1` value, and the headline
carries the matching launch fact `s.pids 1 = 0` (grid axis 1 has extent
`SPLIT_K = 1`); `pid_z` itself and the faithful
`rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)` stay in the surface.
**(2)** `EVEN_K` is fixed to `True` (the unmasked-load arm): `EVEN_K` is
the `@triton.heuristics` constexpr `K % (BLOCK_K * SPLIT_K) == 0`, so the
masked `else` arm drops with the constexpr — the `bmm_optimized`
heuristics fixed-arm and `sgmv_expand_slice` EVEN_K precedents — and the
loop bound `tl.cdiv(K, BLOCK_K * SPLIT_K)` is spelled as the antiquoted
binder `numKBlocks` with the honest side condition `K = BLOCK_K *
numKBlocks` (the `llama_ff_triton` `tl.cdiv` precedent). **(3)** `has_bias`
is kept as a genuine `Bool` parameter and **both arms are modeled** (the
`matmul_dequantize` `NO_GROUPS` precedent): the host passes `0`/`1`, and
the headline covers both arms in one statement through the guarded bias
term `if has_bias then bias(col) else 0`. **(4)** the two
`.to(C.dtype.element_ty)` casts are spelled `.to(tl.float16)` (the host
allocates `C` as `torch.float16`; the `llama_ff_triton` store-cast
precedent — the raw `.dtype.element_ty` wrapper is int-expected in this
DSL), so each compiles to a genuine fp16 quantization event — the bias
load lands directly on the `.fp16` channel and the terminal store is
**`.fp16`-typed**. **(5)** integer literals inside index arithmetic are
written `$(n)` (a bare literal is inferred `.real` by the DSL's expression
typing).

## The `.int` regions

`A` and `B` are `torch.int8` tensors and are modelled as `Region .int`
binders: the loads land on the signed `.int` channel (int8 values are
signed, in `[-128, 127]`), the accumulator `tl.zeros((BLOCK_M, BLOCK_N),
dtype=tl.int32)` lives on the same channel, and `acc += tl.dot(a, b)` is
the integer matmul `Op.dotInt` — exact ℤ arithmetic, no rounding. The
single promotion to ℝ happens once, in the epilogue: `acc * divfactor`
promotes the ℤ accumulator through `Op.intToReal` (the `int4_matmul`
precedent) and the two float factors multiply on the ℝ carrier.

## The wrapped offsets

`ram = rm % M` / `rbn = rn % N` wrap the row/col offsets (spelled through
the source's `tl.max_contiguous(tl.multiple_of(…, BLOCK), BLOCK)` layout
hints, both value-erasers in the DSL), and the K-loop loads are
**unmasked** — an out-of-range lane reads a wrapped-around row instead of
being masked off. `rm` / `rn` are then **rebound** to the unwrapped
offsets (the store coordinates), and the store mask
`(rm < M)[:, None] & (rn < N)[None, :]` is over those, so on every lane
the store actually writes, `row < M` makes the wrap the identity — which
is how the headline's conclusion gets to read the plain `A[row, ·]` row.
The `int4_matmul` twin handles exactly this shape.

## Proof map

```
int8_dequant_matmul_exec_genuine                the headline
├─ i8_body_eq                 28 statements by `rfl`
├─ i8PreLoopScalars_run       9 statements: pids + trip counts + swizzle
├─ i8PreLoopTiles_run         → `i8Inv … 0`  (offsets, ptr tiles, factors, zeros)
├─ i8Loop_collapse            `forRange_inv` over `i8Inv`
│  └─ i8LoopBody_run          5 statements: 2 `.int` loads, `Op.dotInt`, advances
│     └─ i8AccTile_dotInt_succ    `acc += tl.dot(a, b)` (exact ℤ step)
└─ i8PostLoop_run             epilogue + `has_bias` branch + `.fp16` store
   ├─ i8_biasBranch_run       both `has_bias` arms, one landing statement
   └─ i8_store_props          `MemCell`-level fp16 scatter readback
cAddr_injective                                 discharges the headline's `hInj`
```

The stored value is built bottom-up from the kernel's own accessors
(`aElem` / `bElem` as ℤ readers on the `.int` regions, `wFElem` / `xFElem`
/ `biasElem` as ℝ readers) over the **launch** state's memory. The output
region `C` is never read back into a spec, so no part of the trust path is
self-referential.

## Modeling boundary

Arithmetic is over ℝ / ℤ (not bit-accurate IEEE float; the fp16 cast sites
are kept as explicit quantization events whose placeholder semantics is
the identity), and `@triton.autotune` / `@triton.heuristics` and the host
launch (the grid `(cdiv(M,BM)·cdiv(N,BN), SPLIT_K)`, the block sizes,
`GROUP_M = 8`, `divfactor = 1/(127·127)`) are the *trusted boundary*.
Every dimension, stride and block size stays a symbolic parameter; the
dead `ACC_TYPE = tl.float32` constexpr (the source never reads it) has no
surface footprint.
-/

namespace VeriTile.Bench.TritonBenchG.Int8DequantMatmul

open VeriTile.Triton

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-! ## Kernel surface (faithful transcription, `SPLIT_K = 1` / `EVEN_K = True`
arm) -/

set_option linter.unusedVariables false in
def int8_dequant_matmul_surface
    (A B : Region .int) (C bias state_x_ptr state_w_ptr : RegionName)
    (M N K : Nat) (divfactor : ℝ) (has_bias : Bool)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  pid_z = tl.program_id(1)
  grid_m = tl.cdiv($(M), $(BLOCK_M))
  grid_n = tl.cdiv($(N), $(BLOCK_N))
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // (group_size)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  ram = tl.max_contiguous(tl.multiple_of(rm % $(M), $(BLOCK_M)), $(BLOCK_M))
  rbn = tl.max_contiguous(tl.multiple_of(rn % $(N), $(BLOCK_N)), $(BLOCK_N))
  rk = pid_z * $(BLOCK_K) + tl.arange(0, $(BLOCK_K))
  A = $((A : Region .int)) + (ram[:, None] * $(stride_am) + rk[None, :] * $(stride_ak))
  B = $((B : Region .int)) + (rk[:, None] * $(stride_bk) + rbn[None, :] * $(stride_bn))
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  w_factor = tl.load(state_w_ptr + rbn)[None, :]
  x_factor = tl.load(state_x_ptr + ram)[:, None]
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.int32)
  for k in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(A)
    b = tl.load(B)
    acc += tl.dot(a, b)
    A += $(BLOCK_K) * $(stride_ak)
    B += $(BLOCK_K) * $(stride_bk)
  }
  acc = w_factor * (x_factor * (acc * $((divfactor : ℝ))))
  acc = (acc).to(tl.float16)
  if has_bias {
    bias = tl.load(bias + rn).to(tl.float16)
    acc = acc + bias[None, :]
  }
  C = C + (rm[:, None] * $(stride_cm) + rn[None, :] * $(stride_cn))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  tl.store(C, acc, mask=mask)
}

theorem int8_dequant_matmul_surface_toAlgorithm_supported
    (A B : Region .int) (C bias state_x_ptr state_w_ptr : RegionName)
    (M N K : Nat) (divfactor : ℝ) (has_bias : Bool)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M numKBlocks : Nat) :
    ∃ alg, (int8_dequant_matmul_surface A B C bias state_x_ptr state_w_ptr
      M N K divfactor has_bias
      stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K GROUP_M numKBlocks).toAlgorithm?
        = Except.ok alg := by
  simp [int8_dequant_matmul_surface, ComputeExpr.toAlgorithm?]

/-! ## The group-swizzled block coordinates

`pid` is decomposed exactly as the source decomposes it (the source computes
`group_id * GROUP_M` inline instead of naming a `first_pid_m`), so the spec
below is stated in the same coordinates the kernel computes rather than in a
re-derived form. -/

/-- `grid_m = tl.cdiv(M, BLOCK_M)`. -/
def gridM (M BM : Nat) : Nat := (M + BM - 1) / BM

/-- `grid_n = tl.cdiv(N, BLOCK_N)`. -/
def gridN (N BN : Nat) : Nat := (N + BN - 1) / BN

/-- `width = GROUP_M * grid_n`. -/
def i8Width (N BN GM : Nat) : Nat := GM * gridN N BN

/-- `group_id = pid // width`. -/
def groupId (s : BlockState) (N BN GM : Nat) : Nat :=
  s.pids 0 / i8Width N BN GM

/-- `group_size = min(grid_m - group_id * GROUP_M, GROUP_M)`. -/
def groupSize (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  min (gridM M BM - groupId s N BN GM * GM) GM

/-- `pid_m = group_id * GROUP_M + (pid % group_size)`. -/
def pidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  groupId s N BN GM * GM + s.pids 0 % groupSize s M N BM BN GM

/-- `pid_n = (pid % width) // (group_size)`. -/
def pidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % i8Width N BN GM / groupSize s M N BM BN GM

/-! ## Element accessors

Each is the kernel's own address arithmetic over the **launch** state's
memory, parameterized by the *absolute* row / column number, so that the
wrapped (`% M` / `% N`) and unwrapped readings share one definition — the
invariant instantiates them at the wrapped index, the headline (whose store
mask guarantees `row < M`, `col < N`) at the plain one. -/

/-- `A[row, k*BK + e]` at K step `k` — a **signed** `.int`-channel read
(int8 values are signed). Unmasked, matching the source's bare `tl.load(A)`. -/
def aElem (s : BlockState) (A : Region .int)
    (stride_am stride_ak BK : Nat) (row k e : Nat) : ℤ :=
  s.readMemValue .int (Region.cast A) (row * stride_am + (e + k * BK) * stride_ak)

/-- `B[k*BK + e, col]` at K step `k` — the signed `.int` read of `B`. -/
def bElem (s : BlockState) (B : Region .int)
    (stride_bk stride_bn BK : Nat) (k e col : Nat) : ℤ :=
  s.readMemValue .int (Region.cast B) ((e + k * BK) * stride_bk + col * stride_bn)

/-- `state_w[col]` — the per-column dequant scale (`w_factor` lane). -/
noncomputable def wFElem (s : BlockState) (state_w : RegionName) (col : Nat) : ℝ :=
  s.readMem state_w col

/-- `state_x[row]` — the per-row dequant scale (`x_factor` lane). -/
noncomputable def xFElem (s : BlockState) (state_x : RegionName) (row : Nat) : ℝ :=
  s.readMem state_x row

/-- `bias[col]` — the fp16 bias lane, decoded exactly the way the kernel's
`.fp16`-typed load decodes it (`readMemAs .fp16`: an fp16 cell's real payload,
`0` on a dtype-mismatched cell). -/
noncomputable def biasElem (s : BlockState) (bias : RegionName) (addr : Nat) : ℝ :=
  FloatDType.fp16.storeValue (s.readMemValue .fp16 bias addr)

/-- The `.fp16` channel always decodes to a *present* real value: the load's
lane is `some (biasElem …)` — dtype mismatch degrades to `0`, never to `⊥`. -/
private theorem i8_readMemValue_fp16 (s : BlockState) (rg : RegionName) (a : Nat) :
    s.readMemValue .fp16 rg a = some (biasElem s rg a) := by
  unfold biasElem
  show s.readMemAs .fp16 rg a
    = some (FloatDType.fp16.storeValue (s.readMemAs .fp16 rg a))
  unfold BlockState.readMemAs
  cases h : (s.mem rg a).readAs FloatDType.fp16.toTileDType <;>
    simp [FloatDType.ofReal, FloatDType.storeValue]

/-- `readMemValue .fp16` reads only the memory component. -/
private theorem i8_readMemValue_fp16_mem (t s0 : BlockState) (rg : RegionName)
    (a : Nat) (h : t.mem = s0.mem) :
    t.readMemValue .fp16 rg a = s0.readMemValue .fp16 rg a := by
  show t.readMemAs .fp16 rg a = s0.readMemAs .fp16 rg a
  unfold BlockState.readMemAs
  rw [h]

/-! ## The accumulator and the stored value -/

/-- One K step's ℤ contribution to output cell `(row, col)` — the exact
integer sum-product `Op.dotInt` computes. -/
def accStep (s : BlockState) (A B : Region .int)
    (stride_am stride_ak stride_bk stride_bn BK : Nat) (row col k : Nat) : ℤ :=
  ∑ e : Fin BK,
    aElem s A stride_am stride_ak BK row k e.val
      * bElem s B stride_bk stride_bn BK k e.val col

/-- The full ℤ accumulator: `acc` after all `numKBlocks` K steps. -/
def accSpec (s : BlockState) (A B : Region .int)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks : Nat)
    (row col : Nat) : ℤ :=
  ∑ k : Fin numKBlocks,
    accStep s A B stride_am stride_ak stride_bk stride_bn BK row col k.val

/-- The rescaled epilogue value at `(row, col)` — the kernel's exact
parenthesization `w_factor * (x_factor * (acc * divfactor))`, with the ℤ
accumulator promoted to ℝ once (`Op.intToReal`) and then scaled. -/
noncomputable def i8ProdSpec (s : BlockState) (A B : Region .int)
    (state_x state_w : RegionName)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks : Nat)
    (divfactor : ℝ) (row col : Nat) : ℝ :=
  wFElem s state_w col
    * (xFElem s state_x row
        * (((accSpec s A B stride_am stride_ak stride_bk stride_bn BK
              numKBlocks row col : ℤ) : ℝ)
            * divfactor))

/-- **The stored value** (before the terminal fp16 quantization): the rescaled
product plus the `has_bias`-guarded per-column bias. -/
noncomputable def i8Spec (s : BlockState) (A B : Region .int)
    (bias state_x state_w : RegionName)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks : Nat)
    (divfactor : ℝ) (has_bias : Bool) (row col : Nat) : ℝ :=
  i8ProdSpec s A B state_x state_w stride_am stride_ak stride_bk stride_bn
      BK numKBlocks divfactor row col
    + (if has_bias then biasElem s bias col else 0)

/-- The `C` store address for output cell `(r, c)` — the kernel's
`rm[:, None] * stride_cm + rn[None, :] * stride_cn`. -/
def cAddr (stride_cm stride_cn BM BN pm pn : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  (pm * BM + idx.1.val) * stride_cm + (pn * BN + idx.2.1.val) * stride_cn

/-! ## Compiled body decomposition

The algorithm-lowered statement lists, checked against the macro output by
`rfl`. Lowerings worth naming because they are not guessable from the source
text:

* `tl.cdiv` expands to `Op.div .nat` on `(X + BX - 1)`; `min(a, b)` is
  `Op.where (Op.lt …) a b` (there is no `Op.min`);
* `tl.max_contiguous` / `tl.multiple_of` are value-erasers: `ram` / `rbn`
  compile to the bare `Op.mod` of the offset vector;
* the two `tl.load`s inside the K loop are **`.int`-typed** (the typed
  `Region .int` binders route the pointer-element dtype), and
  `acc += tl.dot(a, b)` lowers to `Op.add .int` over **`Op.dotInt`**;
* the epilogue statement is one assignment whose expression nests the three
  multiplies with `Op.intToReal` on the innermost `.int` operand;
* `(acc).to(tl.float16)` compiles to `Op.castFloat real → fp16` (register
  channel hop `.real → .fp16`), and the bias load-with-cast compiles to a
  **direct `.fp16`-typed load** (`Op.load .fp16 (MemAccess.region …)`);
* the `has_bias` branch is `Stmt.ifThen (Op.constBool has_bias)` around the
  two bias statements; the bias add is `Op.add NumericDType.fp16`. -/

/-- The prologue's nine scalar statements: both program ids, the two `tl.cdiv`
trip counts and the group swizzle down to `(pid_m, pid_n)`. -/
def i8PreLoopScalars (M N BM BN GM : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid" (Op.programId 0),
    Stmt.assign .nat [] "pid_z" (Op.programId 1),
    Stmt.assign .nat [] "grid_m"
      (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
        (Op.constNat BM)),
    Stmt.assign .nat [] "grid_n"
      (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
        (Op.constNat BN)),
    Stmt.assign .nat [] "width"
      (Op.mul .nat Broadcast.nil (Op.constNat GM) (Op.ref .nat [] "grid_n")),
    Stmt.assign .nat [] "group_id"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "width")),
    Stmt.assign .nat [] "group_size"
      (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM)))
          (Op.constNat GM))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM)))
        (Op.constNat GM)),
    Stmt.assign .nat [] "pid_m"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM))
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "group_size"))),
    Stmt.assign .nat [] "pid_n"
      (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "width"))
        (Op.ref .nat [] "group_size")) ]

/-- The prologue's twelve index/tile statements: the offset vectors (wrapped
and rebound-unwrapped), the two typed pointer tiles, the two factor loads and
the zeroed `.int` accumulator. -/
def i8PreLoopIndex (A B : Region .int) (state_x_ptr state_w_ptr : RegionName)
    (M N stride_am stride_ak stride_bk stride_bn : Nat)
    (BM BN BK : Nat) : List Stmt :=
  [ Stmt.assign .nat [BM] "rm"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
        (Op.arange BM)),
    Stmt.assign .nat [BN] "rn"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
        (Op.arange BN)),
    Stmt.assign .nat [BM] "ram"
      (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BM] "rm")
        (Op.constNat M)),
    Stmt.assign .nat [BN] "rbn"
      (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BN] "rn")
        (Op.constNat N)),
    Stmt.assign .nat [BK] "rk"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_z") (Op.constNat BK))
        (Op.arange BK)),
    Stmt.assign .ptr [BM, BK] "A"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "ram"))
            (Op.constNat stride_am))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "rk"))
            (Op.constNat stride_ak)))),
    Stmt.assign .ptr [BK, BN] "B"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "rk"))
            (Op.constNat stride_bk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "rbn"))
            (Op.constNat stride_bn)))),
    Stmt.assign .nat [BM] "rm"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
        (Op.arange BM)),
    Stmt.assign .nat [BN] "rn"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
        (Op.arange BN)),
    Stmt.assign .real [1, BN] "w_factor"
      (Op.expandDim ⟨0, by simp⟩
        (Op.load .real (MemAccess.region state_w_ptr (Op.ref .nat [BN] "rbn"))
          MaskOpt.none)),
    Stmt.assign .real [BM, 1] "x_factor"
      (Op.expandDim ⟨1, by simp⟩
        (Op.load .real (MemAccess.region state_x_ptr (Op.ref .nat [BM] "ram"))
          MaskOpt.none)),
    Stmt.assign .int [BM, BN] "acc" (Op.full [BM, BN] (Op.constInt 0)) ]

/-- The K-loop body: two unmasked `.int` loads, the `Op.dotInt` accumulation,
and the two pointer advances (`* SPLIT_K` folded to `1`). -/
def i8LoopBody (stride_ak stride_bk BM BN BK : Nat) : List Stmt :=
  [ Stmt.assign .int [BM, BK] "a"
      (Op.load .int (MemAccess.ptr (Op.ref .ptr [BM, BK] "A")) MaskOpt.none),
    Stmt.assign .int [BK, BN] "b"
      (Op.load .int (MemAccess.ptr (Op.ref .ptr [BK, BN] "B")) MaskOpt.none),
    Stmt.assign .int [BM, BN] "acc"
      (Op.add .int (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .int [BM, BN] "acc")
        (Op.dotInt (batch := []) (Op.ref .int [BM, BK] "a")
          (Op.ref .int [BK, BN] "b"))),
    Stmt.assign .ptr [BM, BK] "A"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "A")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_ak))),
    Stmt.assign .ptr [BK, BN] "B"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "B")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_bk))) ]

/-- The `has_bias` arm: the direct `.fp16`-typed bias load (at the
**unwrapped** `rn`) and the `.fp16`-channel bias add. -/
def i8BiasArm (bias : RegionName) (BM BN : Nat) : List Stmt :=
  [ Stmt.assign .fp16 [BN] "bias"
      (Op.load .fp16 (MemAccess.region (Region.cast bias) (Op.ref .nat [BN] "rn"))
        MaskOpt.none),
    Stmt.assign .fp16 [BM, BN] "acc"
      (Op.add NumericDType.fp16 (Broadcast.consR (Broadcast.consSame Broadcast.nil))
        (Op.ref .fp16 [BM, BN] "acc")
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .fp16 [BN] "bias"))) ]

/-- The compiled tail: the epilogue rescale (one nested-expression
assignment), the fp16 downcast, the `has_bias` branch, the `C` pointer tile,
the two-axis store mask, and the masked **`.fp16`-typed** store of `acc`. -/
def i8PostLoop (C bias : RegionName) (divfactor : ℝ) (has_bias : Bool)
    (M N stride_cm stride_cn BM BN : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BN] "acc"
      (Op.mul .real (Broadcast.consL (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [1, BN] "w_factor")
        (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Op.ref .real [BM, 1] "x_factor")
          (Op.mul .real Broadcast.scalarR
            (Op.intToReal (Op.ref .int [BM, BN] "acc"))
            (Op.const divfactor)))),
    Stmt.assign .fp16 [BM, BN] "acc"
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "acc")),
    Stmt.ifThen (Op.constBool has_bias) (i8BiasArm bias BM BN),
    Stmt.assign .ptr [BM, BN] "C"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "rm"))
            (Op.constNat stride_cm))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "rn"))
            (Op.constNat stride_cn)))),
    Stmt.assign .bool [BM, BN] "mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "rm")
            (Op.constNat M)))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "rn")
            (Op.constNat N)))),
    Stmt.store .fp16 [BM, BN] (MemAccess.ptr (Op.ref .ptr [BM, BN] "C"))
      (Op.ref .fp16 [BM, BN] "acc")
      (MaskOpt.mask (Op.ref .bool [BM, BN] "mask")) ]

set_option maxRecDepth 20000 in
set_option linter.unusedVariables false in
/-- **Full body split (by `rfl`).** The lowered surface is exactly
`i8PreLoopScalars ++ i8PreLoopIndex ++ [forRange "k" 0 numKBlocks 1 i8LoopBody]
++ i8PostLoop` — 28 top-level statements, every one checked against the macro
output. -/
theorem i8_body_eq (A B : Region .int) (C bias state_x_ptr state_w_ptr : RegionName)
    (M N K : Nat) (divfactor : ℝ) (has_bias : Bool)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BM BN BK GM numKBlocks : Nat) :
    (int8_dequant_matmul_surface A B C bias state_x_ptr state_w_ptr
        M N K divfactor has_bias
        stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        BM BN BK GM numKBlocks).toAlgKernel.body
      = i8PreLoopScalars M N BM BN GM
        ++ i8PreLoopIndex A B state_x_ptr state_w_ptr M N stride_am stride_ak
            stride_bk stride_bn BM BN BK
        ++ [Stmt.forRange "k" 0 numKBlocks 1
              (i8LoopBody stride_ak stride_bk BM BN BK)]
        ++ i8PostLoop C bias divfactor has_bias M N stride_cm stride_cn BM BN := by
  rfl

/-! ## Offset, pointer and value tiles -/

/-- `pid_* * BLOCK + tl.arange(0, BLOCK)` (and `tl.arange` alone at base 0). -/
def i8Offs (base BD : Nat) : Tile .nat [BD] := ⟨fun idx => base + idx.1.val⟩

/-- The **wrapped** offset vector `(base + e) % Mm` — `ram` / `rbn`. -/
def i8WrapOffs (base BD Mm : Nat) : Tile .nat [BD] :=
  ⟨fun idx => (base + idx.1.val) % Mm⟩

/-- `A` (the rebound pointer tile) lane `(r, e)` at K step `k`: the wrapped
row offset plus `k` advances of `BK * stride_ak`. -/
def i8AAddr (M stride_am stride_ak BM BK pm k : Nat)
    (idx : TileIndex [BM, BK]) : Nat :=
  (pm * BM + idx.1.val) % M * stride_am + idx.2.1.val * stride_ak
    + k * (BK * stride_ak)

/-- `B` lane `(e, c)` at K step `k`. -/
def i8BAddr (N stride_bk stride_bn BN BK pn k : Nat)
    (idx : TileIndex [BK, BN]) : Nat :=
  idx.1.val * stride_bk + (pn * BN + idx.2.1.val) % N * stride_bn
    + k * (BK * stride_bk)

noncomputable def i8APtrs (A : Region .int)
    (M stride_am stride_ak BM BK pm k : Nat) : Tile .ptr [BM, BK] :=
  ⟨fun idx => (Region.cast A, i8AAddr M stride_am stride_ak BM BK pm k idx)⟩

noncomputable def i8BPtrs (B : Region .int)
    (N stride_bk stride_bn BN BK pn k : Nat) : Tile .ptr [BK, BN] :=
  ⟨fun idx => (Region.cast B, i8BAddr N stride_bk stride_bn BN BK pn k idx)⟩

/-- One `A += BLOCK_K * stride_ak` advance. -/
theorem i8APtrs_succ (A : Region .int) (M stride_am stride_ak BM BK pm k : Nat) :
    Tile.ptrAdd Broadcast.scalarR (i8APtrs A M stride_am stride_ak BM BK pm k)
        (Tile.scalar (BK * stride_ak))
      = i8APtrs A M stride_am stride_ak BM BK pm (k + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, i8APtrs, i8AAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

/-- One `B += BLOCK_K * stride_bk` advance. -/
theorem i8BPtrs_succ (B : Region .int) (N stride_bk stride_bn BN BK pn k : Nat) :
    Tile.ptrAdd Broadcast.scalarR (i8BPtrs B N stride_bk stride_bn BN BK pn k)
        (Tile.scalar (BK * stride_bk))
      = i8BPtrs B N stride_bk stride_bn BN BK pn (k + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, i8BPtrs, i8BAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

/-- The `A` address agrees with `aElem`'s at the wrapped row. -/
theorem i8AAddr_eq (M stride_am stride_ak BM BK pm k : Nat)
    (idx : TileIndex [BM, BK]) :
    i8AAddr M stride_am stride_ak BM BK pm k idx
      = (pm * BM + idx.1.val) % M * stride_am
        + (idx.2.1.val + k * BK) * stride_ak := by
  simp only [i8AAddr]
  ring

/-- The `B` address agrees with `bElem`'s at the wrapped column. -/
theorem i8BAddr_eq (N stride_bk stride_bn BN BK pn k : Nat)
    (idx : TileIndex [BK, BN]) :
    i8BAddr N stride_bk stride_bn BN BK pn k idx
      = (idx.1.val + k * BK) * stride_bk
        + (pn * BN + idx.2.1.val) % N * stride_bn := by
  simp only [i8BAddr]
  ring

/-- The loaded `a` tile at K step `k` — every lane reads the `.int` channel
(unmasked load). -/
def i8ATile (s : BlockState) (A : Region .int)
    (M stride_am stride_ak BM BK pm k : Nat) : Tile .int [BM, BK] :=
  ⟨fun idx => s.readMemValue .int (Region.cast A)
    (i8AAddr M stride_am stride_ak BM BK pm k idx)⟩

/-- The loaded `b` tile at K step `k`. -/
def i8BTile (s : BlockState) (B : Region .int)
    (N stride_bk stride_bn BN BK pn k : Nat) : Tile .int [BK, BN] :=
  ⟨fun idx => s.readMemValue .int (Region.cast B)
    (i8BAddr N stride_bk stride_bn BN BK pn k idx)⟩

theorem i8ATile_data (s : BlockState) (A : Region .int)
    (M stride_am stride_ak BM BK pm k : Nat) (idx : TileIndex [BM, BK]) :
    (i8ATile s A M stride_am stride_ak BM BK pm k).data idx
      = aElem s A stride_am stride_ak BK ((pm * BM + idx.1.val) % M) k
          idx.2.1.val := by
  simp only [i8ATile, aElem, i8AAddr_eq]

theorem i8BTile_data (s : BlockState) (B : Region .int)
    (N stride_bk stride_bn BN BK pn k : Nat) (idx : TileIndex [BK, BN]) :
    (i8BTile s B N stride_bk stride_bn BN BK pn k).data idx
      = bElem s B stride_bk stride_bn BK k idx.1.val
          ((pn * BN + idx.2.1.val) % N) := by
  simp only [i8BTile, bElem, i8BAddr_eq]

/-- `w_factor = tl.load(state_w_ptr + rbn)[None, :]` — a `[1, BN]` tile over
the wrapped column. -/
noncomputable def i8WTile (s : BlockState) (state_w : RegionName)
    (N BN pn : Nat) : Tile .real [1, BN] :=
  ⟨fun idx => some (wFElem s state_w ((pn * BN + idx.2.1.val) % N))⟩

/-- `x_factor = tl.load(state_x_ptr + ram)[:, None]` — a `[BM, 1]` tile over
the wrapped row. -/
noncomputable def i8XTile (s : BlockState) (state_x : RegionName)
    (M BM pm : Nat) : Tile .real [BM, 1] :=
  ⟨fun idx => some (xFElem s state_x ((pm * BM + idx.1.val) % M))⟩

/-- The loaded fp16 bias vector, at the **unwrapped** `rn`. -/
noncomputable def i8BiasTile (s : BlockState) (bias : RegionName)
    (BN pn : Nat) : Tile .fp16 [BN] :=
  ⟨fun idx => some (biasElem s bias (pn * BN + idx.1.val))⟩

/-- `acc` after `i` K steps: the exact ℤ partial sum the invariant carries,
at the **wrapped** row / column of each lane. -/
def i8AccTile (s : BlockState) (A B : Region .int)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK pm pn : Nat)
    (i : Nat) : Tile .int [BM, BN] :=
  ⟨fun idx => ∑ j : Fin i,
    accStep s A B stride_am stride_ak stride_bk stride_bn BK
      ((pm * BM + idx.1.val) % M) ((pn * BN + idx.2.1.val) % N) j.val⟩

/-- At `i = 0` the accumulator is the `.int` zero tile `tl.zeros` produces. -/
theorem i8AccTile_zero (s : BlockState) (A B : Region .int)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK pm pn : Nat) :
    i8AccTile s A B M N stride_am stride_ak stride_bk stride_bn BM BN BK pm pn 0
      = (⟨fun _ => 0⟩ : Tile .int [BM, BN]) := by
  apply Tile.ext
  intro idx
  simp [i8AccTile]

/-- At `i = numKBlocks` the accumulator is `accSpec` (at the wrapped lane). -/
theorem i8AccTile_full (s : BlockState) (A B : Region .int)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK pm pn
      numKBlocks : Nat) (idx : TileIndex [BM, BN]) :
    (i8AccTile s A B M N stride_am stride_ak stride_bk stride_bn BM BN BK pm pn
        numKBlocks).data idx
      = accSpec s A B stride_am stride_ak stride_bk stride_bn BK numKBlocks
          ((pm * BM + idx.1.val) % M) ((pn * BN + idx.2.1.val) % N) := by
  rfl

/-- **The `Op.dotInt` accumulator step.** `acc += tl.dot(a, b)` extends the
exact ℤ partial sum by one `accStep` — at the wrapped lane coordinates. -/
theorem i8AccTile_dotInt_succ (s : BlockState) (A B : Region .int)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK pm pn i : Nat) :
    Tile.bop NumericDType.int.add
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (i8AccTile s A B M N stride_am stride_ak stride_bk stride_bn BM BN BK
          pm pn i)
        (Tile.dotInt [] (i8ATile s A M stride_am stride_ak BM BK pm i)
          (i8BTile s B N stride_bk stride_bn BN BK pn i))
      = i8AccTile s A B M N stride_am stride_ak stride_bk stride_bn BM BN BK
          pm pn (i + 1) := by
  apply Tile.ext
  intro idx
  obtain ⟨r, c, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, i8AccTile]
  -- `erw`: `Tile.dotInt`'s operand shapes are `[] ++ [M, K]`, so
  -- `Tile.dotInt_nil_data` does not fire under `rw` / `simp only`.
  erw [Tile.dotInt_nil_data]
  simp only [i8ATile_data, i8BTile_data, NumericDType.int_add]
  rw [Fin.sum_univ_castSucc]
  simp [accStep]

/-! ## Per-statement eval recipes

Private copies, since bench ports never import each other. `i8_dotInt_eval`
is new to this port — `Op.dotInt` is the integer matmul this kernel is the
first consumer of; like `Op.dot`, its dependent `batch ++ [M, K]` operand
shape needs `erw`. -/

/-- Unmasked `.ptr` load: every lane reads its own `(region, offset)`. -/
private theorem i8_load_ptr_none {dtype : TileDType} {sh : TileShape}
    (nm : RegName) (t : BlockState) (pt : Tile .ptr sh)
    (hp : t.regs .ptr sh nm = some pt) :
    evalOp (Op.load dtype (MemAccess.ptr (Op.ref .ptr sh nm)) MaskOpt.none) t
      = some (⟨fun i => t.readMemValue dtype (pt.data i).1 (pt.data i).2⟩ :
          Tile dtype sh) := by
  simp only [evalOp, evalOp_ref, hp]
  rfl

/-- Pointer advance / offset. -/
private theorem i8_ptrAdd_eval {a b : TileShape} {out : TileShape}
    (bc : Broadcast a b out)
    (pnm : RegName) (t : BlockState) (pt : Tile .ptr a) (off : Op .nat b)
    (ov : Tile .nat b)
    (hp : t.regs .ptr a pnm = some pt) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ref .ptr a pnm) off) t
      = some (Tile.ptrAdd bc pt ov) := by
  simp only [evalOp, evalOp_ref, hp, ho]
  rfl

/-- `%` on the `nat` channel. -/
private theorem i8_mod_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mod IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.mod IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `//` on the `nat` channel. -/
private theorem i8_floorDiv_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.floorDiv IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.floorDiv IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `Op.div` on any numeric channel (what `tl.cdiv` expands to). -/
private theorem i8_divTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.div h bc x y) t = some (Tile.bop h.div bc vx vy) := by
  rw [evalOp_div, hx, hy]
  rfl

/-- `<`. -/
private theorem i8_ltTile_eval {dtype : TileDType} (h : ComparableDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.lt h bc x y) t = some (Tile.cop h.lt bc vx vy) := by
  rw [evalOp_lt, hx, hy]
  rfl

/-- `tl.where` — how `min(a, b)` is lowered, there being no `Op.min`. -/
private theorem i8_where_eval {dtype : TileDType} {sh : TileShape}
    (c : Op .bool sh) (x y : Op dtype sh) (t : BlockState)
    (vc : Tile .bool sh) (vx vy : Tile dtype sh)
    (hc : evalOp c t = some vc) (hx : evalOp x t = some vx)
    (hy : evalOp y t = some vy) :
    evalOp (Op.where c x y) t = some (Tile.select vc vx vy) := by
  rw [evalOp_where, hc, hx, hy]
  rfl

/-- `tl.zeros`. -/
private theorem i8_full_eval {dtype : TileDType} (sh : TileShape) (e : Op dtype [])
    (t : BlockState) (v : Tile dtype []) (hv : evalOp e t = some v) :
    evalOp (Op.full sh e) t
      = some (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype sh) := by
  rw [evalOp_full, hv]
  rfl

/-- The `.int` zero literal. -/
private theorem i8_constInt_eval (n : Int) (t : BlockState) :
    evalOp (Op.constInt n) t = some (Tile.scalar n) := by
  simp [evalOp]

/-- A closed `Bool` branch guard. -/
private theorem i8_constBool_eval (b : Bool) (t : BlockState) :
    evalOp (Op.constBool b) t = some (Tile.scalar b) := by
  simp [evalOp]

/-- A pointer tile built from a bare region base. -/
private theorem i8_ptrAddBase_eval {d : TileDType} {b out : TileShape}
    (bc : Broadcast [] b out) (rg : Region d) (t : BlockState)
    (off : Op .nat b) (ov : Tile .nat b) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ptrBase rg) off) t
      = some (Tile.ptrAdd bc (Tile.scalar ((Region.cast rg : RegionName), 0)) ov) := by
  simp only [evalOp, ho]
  rfl

/-- `*` on two tiles, both operand values known. -/
private theorem i8_mulTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mul h bc x y) t = some (Tile.bop h.mul bc vx vy) := by
  rw [evalOp_mul, hx, hy]
  rfl

/-- `+` on two tiles, both operand values known (any numeric channel — the
loop uses it on `.int`, the bias add on `.fp16`). -/
private theorem i8_addTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.add h bc x y) t = some (Tile.bop h.add bc vx vy) := by
  rw [evalOp_add, hx, hy]
  rfl

/-- `-` on two tiles, both operand values known. -/
private theorem i8_subTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.sub h bc x y) t = some (Tile.bop h.sub bc vx vy) := by
  rw [evalOp_sub, hx, hy]
  rfl

/-- `[:, None]` / `[None, :]` (the axis binder is spelled `ax`: `axis` is a
DSL keyword-argument token). -/
private theorem i8_expandDim_eval {dtype : TileDType} {sh : TileShape}
    (ax : Fin (sh.length + 1)) (x : Op dtype sh) (t : BlockState)
    (v : Tile dtype sh) (hv : evalOp x t = some v) :
    evalOp (Op.expandDim ax x) t = some (Tile.expandDim ax v) := by
  rw [evalOp_expandDim, hv]
  rfl

/-- `Op.intToReal` — the signed ℝ embedding of the epilogue promotion. -/
private theorem i8_intToReal_eval {sh : TileShape} (x : Op .int sh)
    (t : BlockState) (vx : Tile .int sh) (hx : evalOp x t = some vx) :
    evalOp (Op.intToReal x) t = some (Tile.intToReal vx) := by
  simp only [evalOp, hx]
  rfl

/-- `(acc).to(tl.float16)` — the fp16 downcast of the `.real` accumulator:
pointwise `FloatDType.cast` (the `matmul_dequantize` recipe). -/
private theorem i8_castFp16_eval {sh : TileShape} (nm : RegName) (t : BlockState)
    (vt : Tile .real sh) (h : t.regs .real sh nm = some vt) :
    evalOp (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real sh nm)) t
      = some (⟨fun ix => FloatDType.real.cast FloatDType.fp16 (vt.data ix)⟩
          : Tile .fp16 sh) := by
  rw [evalOp_castFloat]
  simp [evalOp_ref, h]

/-- `tl.dot` on the `.int` channel (`Op.dotInt`) at rank 2. `erw`, not `rw`:
the operand shapes are `[] ++ [M, K]`, which does not unfold at reducible
transparency, so `evalOp_dotInt` silently fails to fire under `rw` /
`simp only` (the `TritonSmoke.dotIntSmoke` pattern). -/
private theorem i8_dotInt_eval {M K N : Nat} (x : Op .int [M, K])
    (y : Op .int [K, N]) (t : BlockState)
    (vx : Tile .int [M, K]) (vy : Tile .int [K, N])
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.dotInt (batch := []) x y) t = some (Tile.dotInt [] vx vy) := by
  erw [evalOp_dotInt, hx, hy]
  rfl

/-- `&` on the bool channel. -/
private theorem i8_boolAnd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .bool a) (y : Op .bool b) (t : BlockState)
    (vx : Tile .bool a) (vy : Tile .bool b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.boolAnd bc x y) t
      = some (Tile.bop (fun u v : Bool => u && v) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `Stmt.ifThen` steps by evaluating the guard. -/
private theorem i8_ifThen_step (cond : Op .bool []) (body : List Stmt)
    (t : BlockState) :
    stepStmt (Stmt.ifThen cond body) t
      = (evalOp cond t).bind
          (fun c => if c.data PUnit.unit then stepStmts body t else some t) := by
  unfold stepStmt
  cases evalOp cond t <;> rfl

/-! ### `nat` scalar shapes -/

private theorem i8_mulScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mul .nat Broadcast.nil x y) t = some (Tile.scalar (u * v)) := by
  rw [evalOp_mul, hx, hy]
  rfl

private theorem i8_addScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.add .nat Broadcast.nil x y) t = some (Tile.scalar (u + v)) := by
  rw [evalOp_add, hx, hy]
  rfl

private theorem i8_subScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.sub .nat Broadcast.nil x y) t = some (Tile.scalar (u - v)) := by
  rw [evalOp_sub, hx, hy]
  rfl

private theorem i8_divScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.div .nat Broadcast.nil x y) t = some (Tile.scalar (u / v)) := by
  rw [i8_divTile_eval NumericDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem i8_modScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u % v)) := by
  rw [i8_mod_eval Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem i8_floorDivScalar_eval (x y : Op .nat []) (t : BlockState)
    (u v : Nat) (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u / v)) := by
  rw [i8_floorDiv_eval Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem i8_ltScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.lt ComparableDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (decide (u < v))) := by
  rw [i8_ltTile_eval ComparableDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem i8_whereScalarNat_eval (c : Op .bool []) (x y : Op .nat [])
    (t : BlockState) (cv : Bool) (u v : Nat)
    (hc : evalOp c t = some (Tile.scalar cv))
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.where c x y) t = some (Tile.scalar (if cv then u else v)) := by
  rw [i8_where_eval c x y t _ _ _ hc hx hy]
  rfl

/-- `name * c` on a `nat` scalar register. -/
private theorem i8_mulRef_eval (t : BlockState) (nm : RegName) (val c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c)) t
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `min` is spelled as a `tl.where` by the DSL. -/
private theorem i8_min_as_where (u v : Nat) :
    (if u < v then u else v) = min u v := by
  rcases Nat.lt_or_ge u v with h | h
  · rw [if_pos h]; omega
  · rw [if_neg (by omega)]; omega

/-- `setReg` leaves memory alone, at **function** level (the library's
`setReg_mem` is pointwise; a deep tower's `t.mem = s.mem` by one `rfl`
overruns `whnf`). -/
private theorem i8_setReg_mem {dtype : TileDType} {sh : TileShape}
    (s : BlockState) (nm : RegName) (v : Tile dtype sh) :
    (s.setReg nm dtype sh v).mem = s.mem := rfl

/-! ### The typed loads, bridged to the named tiles -/

/-- The unmasked `.int` `a` load lands on `i8ATile`, on the launch state's
memory. -/
private theorem i8_aLoad_eq (s0 : BlockState) (A : Region .int) (t : BlockState)
    (M stride_am stride_ak BM BK pm i : Nat)
    (hmem : t.mem = s0.mem)
    (hap : t.regs .ptr [BM, BK] "A"
      = some (i8APtrs A M stride_am stride_ak BM BK pm i)) :
    evalOp (Op.load .int (MemAccess.ptr (Op.ref .ptr [BM, BK] "A"))
        MaskOpt.none) t
      = some (i8ATile s0 A M stride_am stride_ak BM BK pm i) := by
  rw [i8_load_ptr_none "A" t _ hap]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [i8ATile, i8APtrs, BlockState.readMemValue, BlockState.readMemTyped,
    hmem]

/-- The unmasked `.int` `b` load lands on `i8BTile`. -/
private theorem i8_bLoad_eq (s0 : BlockState) (B : Region .int) (t : BlockState)
    (N stride_bk stride_bn BN BK pn i : Nat)
    (hmem : t.mem = s0.mem)
    (hbp : t.regs .ptr [BK, BN] "B"
      = some (i8BPtrs B N stride_bk stride_bn BN BK pn i)) :
    evalOp (Op.load .int (MemAccess.ptr (Op.ref .ptr [BK, BN] "B"))
        MaskOpt.none) t
      = some (i8BTile s0 B N stride_bk stride_bn BN BK pn i) := by
  rw [i8_load_ptr_none "B" t _ hbp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [i8BTile, i8BPtrs, BlockState.readMemValue, BlockState.readMemTyped,
    hmem]

/-- The `w_factor` region load + `[None, :]` lands on `i8WTile`. -/
private theorem i8_wFactor_eval (s0 : BlockState) (state_w : RegionName)
    (t : BlockState) (N BN pn : Nat)
    (hmem : t.mem = s0.mem)
    (hrbn : t.regs .nat [BN] "rbn" = some (i8WrapOffs (pn * BN) BN N)) :
    evalOp (Op.expandDim ⟨0, by simp⟩
        (Op.load .real (MemAccess.region state_w (Op.ref .nat [BN] "rbn"))
          MaskOpt.none)) t
      = some (i8WTile s0 state_w N BN pn) := by
  have hload : evalOp (Op.load .real
        (MemAccess.region state_w (Op.ref .nat [BN] "rbn")) MaskOpt.none) t
      = some (⟨fun i => some (wFElem s0 state_w ((pn * BN + i.1.val) % N))⟩ :
          Tile .real [BN]) := by
    rw [evalOp_load_region_none]
    simp only [evalOp_ref, hrbn, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    apply Tile.ext
    intro idx
    simp only [BlockState.readMemValue_real, Region.cast_id, i8WrapOffs, wFElem]
    unfold BlockState.readMem
    rw [hmem]
  rw [i8_expandDim_eval _ _ t _ hload]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i8WTile, Tile.expandDim_data, TileShape.dropInsertedIndex]

/-- The `x_factor` region load + `[:, None]` lands on `i8XTile`. -/
private theorem i8_xFactor_eval (s0 : BlockState) (state_x : RegionName)
    (t : BlockState) (M BM pm : Nat)
    (hmem : t.mem = s0.mem)
    (hram : t.regs .nat [BM] "ram" = some (i8WrapOffs (pm * BM) BM M)) :
    evalOp (Op.expandDim ⟨1, by simp⟩
        (Op.load .real (MemAccess.region state_x (Op.ref .nat [BM] "ram"))
          MaskOpt.none)) t
      = some (i8XTile s0 state_x M BM pm) := by
  have hload : evalOp (Op.load .real
        (MemAccess.region state_x (Op.ref .nat [BM] "ram")) MaskOpt.none) t
      = some (⟨fun i => some (xFElem s0 state_x ((pm * BM + i.1.val) % M))⟩ :
          Tile .real [BM]) := by
    rw [evalOp_load_region_none]
    simp only [evalOp_ref, hram, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    apply Tile.ext
    intro idx
    simp only [BlockState.readMemValue_real, Region.cast_id, i8WrapOffs, xFElem]
    unfold BlockState.readMem
    rw [hmem]
  rw [i8_expandDim_eval _ _ t _ hload]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i8XTile, Tile.expandDim_data, TileShape.dropInsertedIndex]

/-- The `.fp16`-typed bias load lands on `i8BiasTile` — at the **unwrapped**
`rn`, every lane a present (`some`) fp16 decode. -/
private theorem i8_biasLoad_eq (s0 : BlockState) (bias : RegionName)
    (t : BlockState) (BN pn : Nat)
    (hmem : t.mem = s0.mem)
    (hrn : t.regs .nat [BN] "rn" = some (i8Offs (pn * BN) BN)) :
    evalOp (Op.load .fp16
        (MemAccess.region (Region.cast bias) (Op.ref .nat [BN] "rn"))
        MaskOpt.none) t
      = some (i8BiasTile s0 bias BN pn) := by
  rw [evalOp_load_region_none]
  simp only [evalOp_ref, hrn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [i8BiasTile, i8Offs, Region.cast_cast, Region.cast_id]
  rw [i8_readMemValue_fp16_mem t s0 _ _ hmem, i8_readMemValue_fp16]

/-! ### The pid swizzle, statement by statement -/

private theorem i8_cdiv_eval (t : BlockState) (X BX : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat X) (Op.constNat BX)) (Op.constNat 1))
        (Op.constNat BX)) t
      = some (Tile.scalar ((X + BX - 1) / BX)) :=
  i8_divScalarNat_eval _ _ t (X + BX - 1) BX
    (i8_subScalarNat_eval _ _ t (X + BX) 1
      (i8_addScalarNat_eval _ _ t X BX (evalOp_constNat _ _) (evalOp_constNat _ _))
      (evalOp_constNat _ _))
    (evalOp_constNat _ _)

private theorem i8_width_eval (t : BlockState) (N BN GM : Nat)
    (hgn : t.regs .nat [] "grid_n" = some (Tile.scalar (gridN N BN))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.constNat GM)
        (Op.ref .nat [] "grid_n")) t
      = some (Tile.scalar (i8Width N BN GM)) := by
  rw [i8_mulScalarNat_eval _ _ t GM (gridN N BN) (evalOp_constNat _ _)
    (by rw [evalOp_ref]; exact hgn)]
  rfl

private theorem i8_groupId_eval (s t : BlockState) (N BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "width" = some (Tile.scalar (i8Width N BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "width")) t
      = some (Tile.scalar (groupId s N BN GM)) := by
  rw [i8_floorDivScalar_eval _ _ t (s.pids 0) (i8Width N BN GM)
    (by rw [evalOp_ref]; exact hpid) (by rw [evalOp_ref]; exact hwid)]
  rfl

private theorem i8_groupSize_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hgm : t.regs .nat [] "grid_m" = some (Tile.scalar (gridM M BM)))
    (hgid : t.regs .nat [] "group_id" = some (Tile.scalar (groupId s N BN GM))) :
    evalOp (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
              (Op.constNat GM)))
          (Op.constNat GM))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
            (Op.constNat GM)))
        (Op.constNat GM)) t
      = some (Tile.scalar (groupSize s M N BM BN GM)) := by
  have hsub : evalOp (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM))) t
      = some (Tile.scalar (gridM M BM - groupId s N BN GM * GM)) :=
    i8_subScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hgm)
      (i8_mulRef_eval t "group_id" (groupId s N BN GM) GM hgid)
  rw [i8_whereScalarNat_eval _ _ _ t
    (decide (gridM M BM - groupId s N BN GM * GM < GM))
    (gridM M BM - groupId s N BN GM * GM) GM
    (i8_ltScalarNat_eval _ _ t _ _ hsub (evalOp_constNat _ _)) hsub
    (evalOp_constNat _ _)]
  simp only [decide_eq_true_eq, i8_min_as_where, groupSize]

private theorem i8_pidM_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hgid : t.regs .nat [] "group_id" = some (Tile.scalar (groupId s N BN GM)))
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hgs : t.regs .nat [] "group_size"
      = some (Tile.scalar (groupSize s M N BM BN GM))) :
    evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM))
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "group_size"))) t
      = some (Tile.scalar (pidM s M N BM BN GM)) := by
  rw [i8_addScalarNat_eval _ _ t (groupId s N BN GM * GM)
    (s.pids 0 % groupSize s M N BM BN GM)
    (i8_mulRef_eval t "group_id" (groupId s N BN GM) GM hgid)
    (i8_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hgs))]
  rfl

private theorem i8_pidN_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "width" = some (Tile.scalar (i8Width N BN GM)))
    (hgs : t.regs .nat [] "group_size"
      = some (Tile.scalar (groupSize s M N BM BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "width"))
        (Op.ref .nat [] "group_size")) t
      = some (Tile.scalar (pidN s M N BM BN GM)) := by
  rw [i8_floorDivScalar_eval _ _ t (s.pids 0 % i8Width N BN GM)
    (groupSize s M N BM BN GM)
    (i8_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hwid))
    (by rw [evalOp_ref]; exact hgs)]
  rfl

/-! ### The index tiles -/

/-- `pid_* * BLOCK + tl.arange(0, BLOCK)` from a scalar register. -/
private theorem i8_offs_eval (nm : RegName) (t : BlockState) (BD base c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar base)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c))
        (Op.arange BD)) t
      = some (i8Offs (base * c) BD) := by
  rw [i8_addTile_eval NumericDType.nat Broadcast.scalarL _ _ t
    (Tile.scalar (base * c)) (Tile.vec (fun i => (i.val : Nat)))
    (i8_mulScalarNat_eval _ _ t base c (by rw [evalOp_ref]; exact hr)
      (evalOp_constNat _ _))
    (evalOp_arange _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i8Offs, Tile.vec, Broadcast.rightIndex, NumericDType.add]

/-- `rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)` under the launch fact
`pid_z = 0` (grid axis 1 has extent `SPLIT_K = 1`). -/
private theorem i8_rk_eval (t : BlockState) (BK : Nat)
    (hr : t.regs .nat [] "pid_z" = some (Tile.scalar 0)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_z") (Op.constNat BK))
        (Op.arange BK)) t
      = some (i8Offs 0 BK) := by
  have h := i8_offs_eval "pid_z" t BK 0 BK hr
  rwa [Nat.zero_mul] at h

/-- The wrap statements `ram = rm % M` / `rbn = rn % N` (the
`tl.max_contiguous(tl.multiple_of(…))` wrappers are value-erased). -/
private theorem i8_wrapVec_eval (nm : RegName) (t : BlockState)
    (BD base Mm : Nat)
    (hr : t.regs .nat [BD] nm = some (i8Offs base BD)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BD] nm)
        (Op.constNat Mm)) t
      = some (i8WrapOffs base BD Mm) := by
  rw [i8_mod_eval Broadcast.scalarR _ _ t (i8Offs base BD) (Tile.scalar Mm)
    (by rw [evalOp_ref]; exact hr) (evalOp_constNat _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i8WrapOffs, i8Offs, Tile.bop_data, Broadcast.leftIndex,
    Broadcast.rightIndex]

/-- `A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)` — at K
step `0`. -/
private theorem i8_aPtrsInit_eval (A : Region .int) (t : BlockState)
    (M stride_am stride_ak BM BK pm : Nat)
    (hram : t.regs .nat [BM] "ram" = some (i8WrapOffs (pm * BM) BM M))
    (hrk : t.regs .nat [BK] "rk" = some (i8Offs 0 BK)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "ram"))
            (Op.constNat stride_am))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "rk"))
            (Op.constNat stride_ak)))) t
      = some (i8APtrs A M stride_am stride_ak BM BK pm 0) := by
  rw [i8_ptrAddBase_eval _ _ t _ _
    (i8_addTile_eval NumericDType.nat _ _ _ t _ _
      (i8_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_am)
        (i8_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hram))
        (evalOp_constNat _ _))
      (i8_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_ak)
        (i8_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrk))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i8APtrs, i8AAddr, i8WrapOffs, i8Offs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)` — at K
step `0`. -/
private theorem i8_bPtrsInit_eval (B : Region .int) (t : BlockState)
    (N stride_bk stride_bn BN BK pn : Nat)
    (hrk : t.regs .nat [BK] "rk" = some (i8Offs 0 BK))
    (hrbn : t.regs .nat [BN] "rbn" = some (i8WrapOffs (pn * BN) BN N)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "rk"))
            (Op.constNat stride_bk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "rbn"))
            (Op.constNat stride_bn)))) t
      = some (i8BPtrs B N stride_bk stride_bn BN BK pn 0) := by
  rw [i8_ptrAddBase_eval _ _ t _ _
    (i8_addTile_eval NumericDType.nat _ _ _ t _ _
      (i8_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_bk)
        (i8_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrk))
        (evalOp_constNat _ _))
      (i8_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_bn)
        (i8_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrbn))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i8BPtrs, i8BAddr, i8WrapOffs, i8Offs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-! ## The K-loop invariant

`i ≤ numKBlocks` is part of the predicate on purpose: `forRange_inv` concludes
only `stop ≤ final`, so carrying the upper bound is what pins
`final = numKBlocks` and lets the readout use `i8AccTile_full`. The carried
registers are exactly those the loop advances (`A`, `B`, `acc`) plus those the
epilogue still needs (`rm`, `rn`, `w_factor`, `x_factor`). -/

/-- The state carried across K steps. -/
noncomputable def i8Inv (s0 : BlockState) (A B : Region .int)
    (state_x state_w : RegionName)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK GM numKBlocks : Nat)
    (i : Nat) (s : BlockState) : Prop :=
  i ≤ numKBlocks
  ∧ s.mem = s0.mem
  ∧ s.pids = s0.pids
  ∧ s.regs .nat [BM] "rm" = some (i8Offs (pidM s0 M N BM BN GM * BM) BM)
  ∧ s.regs .nat [BN] "rn" = some (i8Offs (pidN s0 M N BM BN GM * BN) BN)
  ∧ s.regs .real [1, BN] "w_factor"
      = some (i8WTile s0 state_w N BN (pidN s0 M N BM BN GM))
  ∧ s.regs .real [BM, 1] "x_factor"
      = some (i8XTile s0 state_x M BM (pidM s0 M N BM BN GM))
  ∧ s.regs .ptr [BM, BK] "A"
      = some (i8APtrs A M stride_am stride_ak BM BK (pidM s0 M N BM BN GM) i)
  ∧ s.regs .ptr [BK, BN] "B"
      = some (i8BPtrs B N stride_bk stride_bn BN BK (pidN s0 M N BM BN GM) i)
  ∧ s.regs .int [BM, BN] "acc"
      = some (i8AccTile s0 A B M N stride_am stride_ak stride_bk stride_bn
          BM BN BK (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM) i)

/-- The loop combinator writes the induction variable before each iteration,
and `i8Inv` constrains no register named `"k"`. -/
theorem i8Inv_setReg_k (s0 : BlockState) (A B : Region .int)
    (state_x state_w : RegionName)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK GM numKBlocks i j : Nat)
    (s : BlockState)
    (h : i8Inv s0 A B state_x state_w M N stride_am stride_ak stride_bk
      stride_bn BM BN BK GM numKBlocks i s) :
    i8Inv s0 A B state_x state_w M N stride_am stride_ak stride_bk stride_bn
      BM BN BK GM numKBlocks i (s.setReg "k" .nat [] (Tile.scalar j)) := by
  obtain ⟨hle, hmem, hpids, hrm, hrn, hw, hx, hA, hB, hacc⟩ := h
  exact ⟨hle, hmem, hpids, by simpa using hrm, by simpa using hrn,
    by simpa using hw, by simpa using hx, by simpa using hA,
    by simpa using hB, by simpa using hacc⟩

/-! ### The K step -/

theorem i8LoopBody_run (s0 : BlockState) (A B : Region .int)
    (state_x state_w : RegionName)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK GM numKBlocks i : Nat)
    (s : BlockState)
    (hnext : i + 1 ≤ numKBlocks)
    (hinv : i8Inv s0 A B state_x state_w M N stride_am stride_ak stride_bk
      stride_bn BM BN BK GM numKBlocks i s) :
    ∃ s', stepStmts (i8LoopBody stride_ak stride_bk BM BN BK) s = some s'
      ∧ i8Inv s0 A B state_x state_w M N stride_am stride_ak stride_bk
          stride_bn BM BN BK GM numKBlocks (i + 1) s' := by
  obtain ⟨-, hmem, hpids, hrm, hrn, hw, hx, hA, hB, hacc⟩ := hinv
  set pm := pidM s0 M N BM BN GM with hpmDef
  set pn := pidN s0 M N BM BN GM with hpnDef
  unfold i8LoopBody
  -- 1. `a = tl.load(A)`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_aLoad_eq s0 A s M stride_am stride_ak BM BK pm i hmem hA))]
  -- 2. `b = tl.load(B)`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_bLoad_eq s0 B _ N stride_bk stride_bn BN BK pn i (by simpa using hmem)
      (by simpa using hB)))]
  -- 3. `acc += tl.dot(a, b)` — the `Op.dotInt` step
  have h3 : evalOp (Op.add .int
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .int [BM, BN] "acc")
        (Op.dotInt (batch := []) (Op.ref .int [BM, BK] "a")
          (Op.ref .int [BK, BN] "b")))
        ((s.setReg "a" .int [BM, BK]
            (i8ATile s0 A M stride_am stride_ak BM BK pm i)).setReg
          "b" .int [BK, BN] (i8BTile s0 B N stride_bk stride_bn BN BK pn i))
      = some (i8AccTile s0 A B M N stride_am stride_ak stride_bk stride_bn
          BM BN BK pm pn (i + 1)) := by
    rw [← i8AccTile_dotInt_succ]
    exact i8_addTile_eval NumericDType.int _ _ _ _ _ _
      (by rw [evalOp_ref]; simpa using hacc)
      (i8_dotInt_eval _ _ _ _ _ (by rw [evalOp_ref]; simp)
        (by rw [evalOp_ref]; simp))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h3)]
  -- 4. `A += BLOCK_K * stride_ak`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "A")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_ak))) _
      = some (i8APtrs A M stride_am stride_ak BM BK pm (i + 1)) from by
      rw [← i8APtrs_succ]
      exact i8_ptrAdd_eval Broadcast.scalarR "A" _ _ _
        (Tile.scalar (BK * stride_ak)) (by simpa using hA)
        (i8_mulScalarNat_eval _ _ _ BK stride_ak (evalOp_constNat _ _)
          (evalOp_constNat _ _))))]
  -- 5. `B += BLOCK_K * stride_bk`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "B")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_bk))) _
      = some (i8BPtrs B N stride_bk stride_bn BN BK pn (i + 1)) from by
      rw [← i8BPtrs_succ]
      exact i8_ptrAdd_eval Broadcast.scalarR "B" _ _ _
        (Tile.scalar (BK * stride_bk)) (by simpa using hB)
        (i8_mulScalarNat_eval _ _ _ BK stride_bk (evalOp_constNat _ _)
          (evalOp_constNat _ _))))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, hnext, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [i8_setReg_mem]
    exact hmem
  · simp only [BlockState.setReg_pids]
    exact hpids
  · simpa using hrm
  · simpa using hrn
  · simpa using hw
  · simpa using hx
  · simp [hpmDef]
  · simp [hpnDef]
  · simp [hpmDef, hpnDef]

/-! ### Collapsing the K loop -/

theorem i8Loop_collapse (s0 : BlockState) (A B : Region .int)
    (state_x state_w : RegionName)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK GM numKBlocks : Nat)
    (s : BlockState)
    (h0 : i8Inv s0 A B state_x state_w M N stride_am stride_ak stride_bk
      stride_bn BM BN BK GM numKBlocks 0 s) :
    ∃ sF, stepStmt (Stmt.forRange "k" 0 numKBlocks 1
          (i8LoopBody stride_ak stride_bk BM BN BK)) s = some sF
      ∧ i8Inv s0 A B state_x state_w M N stride_am stride_ak stride_bk
          stride_bn BM BN BK GM numKBlocks numKBlocks sF := by
  obtain ⟨final, sF, hrun, hfinal, hP⟩ :=
    forRange_inv (idx := "k") (start := 0) (stop := numKBlocks) (step := 1)
      (P := fun i t => i8Inv s0 A B state_x state_w M N stride_am stride_ak
        stride_bk stride_bn BM BN BK GM numKBlocks i t)
      one_ne_zero h0
      (fun i t hi hinv => by
        obtain ⟨s', hs', hinv'⟩ :=
          i8LoopBody_run s0 A B state_x state_w M N stride_am stride_ak
            stride_bk stride_bn BM BN BK GM numKBlocks i _ (by omega)
            (i8Inv_setReg_k s0 A B state_x state_w M N stride_am stride_ak
              stride_bk stride_bn BM BN BK GM numKBlocks i i t hinv)
        exact ⟨s', hs', hinv'⟩)
  have hEq : final = numKBlocks := le_antisymm hP.1 hfinal
  subst hEq
  exact ⟨sF, hrun, hP⟩

/-! ## The prologue walks -/

/-- The nine scalar statements. Memory is untouched; the registers anything
downstream reads are the SPLIT_K program id and the block coordinates. -/
theorem i8PreLoopScalars_run (s : BlockState) (M N BM BN GM : Nat) :
    ∃ t, stepStmts (i8PreLoopScalars M N BM BN GM) s = some t
      ∧ t.mem = s.mem
      ∧ t.pids = s.pids
      ∧ t.regs .nat [] "pid_z" = some (Tile.scalar (s.pids 1))
      ∧ t.regs .nat [] "pid_m" = some (Tile.scalar (pidM s M N BM BN GM))
      ∧ t.regs .nat [] "pid_n" = some (Tile.scalar (pidN s M N BM BN GM)) := by
  unfold i8PreLoopScalars
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (i8_cdiv_eval _ M BM))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (i8_cdiv_eval _ N BN))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_width_eval _ N BN GM (by simp [gridN])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_groupId_eval s _ N BN GM (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_groupSize_eval s _ M N BM BN GM (by simp [gridM]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_pidM_eval s _ M N BM BN GM (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_pidN_eval s _ M N BM BN GM (by simp) (by simp) (by simp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_⟩ <;> simp [i8_setReg_mem]

/-- The twelve index/tile statements, ending on `i8Inv` at step `0`. The
`pid_z = 0` hypothesis is the launch fact `s.pids 1 = 0` (grid axis 1 has
extent `SPLIT_K = 1`), already rewritten by the caller. -/
theorem i8PreLoopTiles_run (s0 : BlockState) (A B : Region .int)
    (state_x state_w : RegionName)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK GM numKBlocks : Nat)
    (t : BlockState)
    (hmem : t.mem = s0.mem)
    (hpids : t.pids = s0.pids)
    (hpz : t.regs .nat [] "pid_z" = some (Tile.scalar 0))
    (hpm : t.regs .nat [] "pid_m" = some (Tile.scalar (pidM s0 M N BM BN GM)))
    (hpn : t.regs .nat [] "pid_n" = some (Tile.scalar (pidN s0 M N BM BN GM))) :
    ∃ t', stepStmts (i8PreLoopIndex A B state_x state_w M N stride_am stride_ak
          stride_bk stride_bn BM BN BK) t = some t'
      ∧ i8Inv s0 A B state_x state_w M N stride_am stride_ak stride_bk
          stride_bn BM BN BK GM numKBlocks 0 t' := by
  unfold i8PreLoopIndex
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_offs_eval "pid_m" t BM (pidM s0 M N BM BN GM) BM hpm))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_offs_eval "pid_n" _ BN (pidN s0 M N BM BN GM) BN (by simpa using hpn)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_wrapVec_eval "rm" _ BM (pidM s0 M N BM BN GM * BM) M (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_wrapVec_eval "rn" _ BN (pidN s0 M N BM BN GM * BN) N (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_rk_eval _ BK (by simpa using hpz)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_aPtrsInit_eval A _ M stride_am stride_ak BM BK (pidM s0 M N BM BN GM)
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_bPtrsInit_eval B _ N stride_bk stride_bn BN BK (pidN s0 M N BM BN GM)
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_offs_eval "pid_m" _ BM (pidM s0 M N BM BN GM) BM (by simpa using hpm)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_offs_eval "pid_n" _ BN (pidN s0 M N BM BN GM) BN (by simpa using hpn)))]
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some TileDType.real [1, BN]
    "w_factor"
    (Op.expandDim ⟨0, by simp⟩
      (Op.load .real (MemAccess.region state_w (Op.ref .nat [BN] "rbn"))
        MaskOpt.none)) _
    (i8WTile s0 state_w N BN (pidN s0 M N BM BN GM))
    (i8_wFactor_eval s0 state_w _ N BN (pidN s0 M N BM BN GM)
      (by simpa [i8_setReg_mem] using hmem) (by simp)))]
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some TileDType.real [BM, 1]
    "x_factor"
    (Op.expandDim ⟨1, by simp⟩
      (Op.load .real (MemAccess.region state_x (Op.ref .nat [BM] "ram"))
        MaskOpt.none)) _
    (i8XTile s0 state_x M BM (pidM s0 M N BM BN GM))
    (i8_xFactor_eval s0 state_x _ M BM (pidM s0 M N BM BN GM)
      (by simpa [i8_setReg_mem] using hmem) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM, BN] (Op.constInt 0)) _
        = some (i8AccTile s0 A B M N stride_am stride_ak stride_bk stride_bn
            BM BN BK (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM) 0) from by
      rw [i8AccTile_zero]
      exact i8_full_eval [BM, BN] (Op.constInt 0) _ _ (i8_constInt_eval 0 _)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [i8_setReg_mem]
    exact hmem
  · simpa using hpids
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp

/-! ## The epilogue

The rescale statement, the fp16 downcast, and the `has_bias` branch — the two
configurations land on one statement through the guarded bias term. -/

/-- `acc` after the epilogue rescale, at the wrapped lane coordinates. -/
noncomputable def i8ProdTile (s : BlockState) (A B : Region .int)
    (state_x state_w : RegionName)
    (M N stride_am stride_ak stride_bk stride_bn BK numKBlocks : Nat)
    (divfactor : ℝ) (BM BN pm pn : Nat) : Tile .real [BM, BN] :=
  ⟨fun idx => some (i8ProdSpec s A B state_x state_w stride_am stride_ak
    stride_bk stride_bn BK numKBlocks divfactor
    ((pm * BM + idx.1.val) % M) ((pn * BN + idx.2.1.val) % N))⟩

/-- The epilogue statement `acc = w_factor * (x_factor * (acc * divfactor))`:
the ℤ accumulator is promoted through `Op.intToReal` once and the two loaded
factors multiply on the ℝ carrier — landing on `i8ProdTile`. -/
private theorem i8_prod_eval (s0 : BlockState) (A B : Region .int)
    (state_x state_w : RegionName) (t : BlockState)
    (M N stride_am stride_ak stride_bk stride_bn BM BN BK pm pn numKBlocks : Nat)
    (divfactor : ℝ)
    (hw : t.regs .real [1, BN] "w_factor" = some (i8WTile s0 state_w N BN pn))
    (hx : t.regs .real [BM, 1] "x_factor" = some (i8XTile s0 state_x M BM pm))
    (hacc : t.regs .int [BM, BN] "acc"
      = some (i8AccTile s0 A B M N stride_am stride_ak stride_bk stride_bn
          BM BN BK pm pn numKBlocks)) :
    evalOp (Op.mul .real (Broadcast.consL (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [1, BN] "w_factor")
        (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Op.ref .real [BM, 1] "x_factor")
          (Op.mul .real Broadcast.scalarR
            (Op.intToReal (Op.ref .int [BM, BN] "acc"))
            (Op.const divfactor)))) t
      = some (i8ProdTile s0 A B state_x state_w M N stride_am stride_ak
          stride_bk stride_bn BK numKBlocks divfactor BM BN pm pn) := by
  rw [i8_mulTile_eval NumericDType.real _ _ _ t _ _
    (by rw [evalOp_ref]; exact hw)
    (i8_mulTile_eval NumericDType.real _ _ _ t _ _
      (by rw [evalOp_ref]; exact hx)
      (i8_mulTile_eval NumericDType.real Broadcast.scalarR _ _ t _
        (Tile.scalar (some divfactor))
        (i8_intToReal_eval _ t _ (by rw [evalOp_ref]; exact hacc))
        (evalOp_const divfactor t)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, c, u⟩ := idx
  simp [i8ProdTile, i8ProdSpec, i8WTile, i8XTile, i8AccTile, accSpec,
    Tile.bop_data, Tile.intToReal_data, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul, WithBot.realMul]

/-- **The `has_bias` branch, both arms.** The two configurations land on the
same statement: `acc` holds the fp16 image of `prodF + (if has_bias then
bias(col) else 0)` — under `has_bias = false` the branch is skipped and the
bias term is `0`; under `has_bias = true` the two arm statements load the
fp16 bias at the unwrapped `rn` and add it on the `.fp16` channel. -/
private theorem i8_biasBranch_run (s0 : BlockState) (bias : RegionName)
    (t : BlockState) (has_bias : Bool) (BM BN pn : Nat)
    (prodF : TileIndex [BM, BN] → ℝ)
    (hmem : t.mem = s0.mem)
    (hrn : t.regs .nat [BN] "rn" = some (i8Offs (pn * BN) BN))
    (hacc : t.regs .fp16 [BM, BN] "acc"
      = some (⟨fun ix => FloatDType.real.cast FloatDType.fp16 (some (prodF ix))⟩
          : Tile .fp16 [BM, BN])) :
    ∃ t', stepStmt (Stmt.ifThen (Op.constBool has_bias) (i8BiasArm bias BM BN)) t
        = some t'
      ∧ t'.mem = t.mem
      ∧ (∀ (dtype : TileDType) (sh : TileShape) (nm : RegName),
          nm ≠ "bias" → nm ≠ "acc" → t'.regs dtype sh nm = t.regs dtype sh nm)
      ∧ t'.regs .fp16 [BM, BN] "acc"
          = some (⟨fun ix => FloatDType.real.cast FloatDType.fp16
              (some (prodF ix
                + (if has_bias then biasElem s0 bias (pn * BN + ix.2.1.val)
                   else 0)))⟩ : Tile .fp16 [BM, BN]) := by
  -- The DSL claims `true` / `false` as expression tokens, so the two
  -- configurations are separated by `by_cases` on the flag.
  by_cases hb : has_bias = Bool.true
  · -- `has_bias`: walk the two arm statements.
    subst hb
    have hstep : stepStmt (Stmt.ifThen (Op.constBool Bool.true)
          (i8BiasArm bias BM BN)) t
        = stepStmts (i8BiasArm bias BM BN) t := by
      rw [i8_ifThen_step, i8_constBool_eval]
      rfl
    rw [hstep]
    unfold i8BiasArm
    -- 1. `bias = tl.load(bias + rn).to(tl.float16)`
    rw [stepStmts.cons_some (@stepStmt_assign_eq_some TileDType.fp16 [BN] "bias"
      (Op.load .fp16
        (MemAccess.region (Region.cast bias) (Op.ref .nat [BN] "rn"))
        MaskOpt.none) _
      (i8BiasTile s0 bias BN pn)
      (i8_biasLoad_eq s0 bias t BN pn hmem hrn))]
    -- 2. `acc = acc + bias[None, :]` — the `.fp16`-channel add
    rw [stepStmts.cons_some (@stepStmt_assign_eq_some TileDType.fp16 [BM, BN] "acc"
      (Op.add NumericDType.fp16
        (Broadcast.consR (Broadcast.consSame Broadcast.nil))
        (Op.ref .fp16 [BM, BN] "acc")
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .fp16 [BN] "bias"))) _
      (Tile.bop NumericDType.fp16.add
        (Broadcast.consR (Broadcast.consSame Broadcast.nil))
        (⟨fun ix => FloatDType.real.cast FloatDType.fp16 (some (prodF ix))⟩
          : Tile .fp16 [BM, BN])
        (Tile.expandDim ⟨0, by simp⟩ (i8BiasTile s0 bias BN pn)))
      (i8_addTile_eval NumericDType.fp16 _ _ _ _ _ _
        (by rw [evalOp_ref]; simpa using hacc)
        (i8_expandDim_eval _ _ _ _ (by rw [evalOp_ref]; simp))))]
    rw [stepStmts.nil]
    refine ⟨_, rfl, ?_, ?_, ?_⟩
    · simp [i8_setReg_mem]
    · intro dtype sh nm h1 h2
      simp [h1, h2]
    · have htile : Tile.bop NumericDType.fp16.add
          (Broadcast.consR (Broadcast.consSame Broadcast.nil))
          (⟨fun ix => FloatDType.real.cast FloatDType.fp16 (some (prodF ix))⟩
            : Tile .fp16 [BM, BN])
          (Tile.expandDim ⟨0, by simp⟩ (i8BiasTile s0 bias BN pn))
          = (⟨fun ix => FloatDType.real.cast FloatDType.fp16
              (some (prodF ix
                + (if Bool.true then biasElem s0 bias (pn * BN + ix.2.1.val)
                   else 0)))⟩ : Tile .fp16 [BM, BN]) := by
        apply Tile.ext
        intro idx
        obtain ⟨r, c, u⟩ := idx
        simp [Tile.bop_data, Tile.expandDim_data, TileShape.dropInsertedIndex,
          Broadcast.leftIndex, Broadcast.rightIndex, i8BiasTile,
          NumericDType.add, WithBot.realAdd, FloatDType.cast]
      rw [← htile]
      simp
  · -- No bias: the branch is skipped, and the guarded bias term is `0`.
    have hflag : has_bias = Bool.false := by simpa using hb
    subst hflag
    refine ⟨t, ?_, rfl, fun _ _ _ _ _ => rfl, ?_⟩
    · rw [i8_ifThen_step, i8_constBool_eval]
      rfl
    · rw [hacc]
      refine congrArg some ?_
      apply Tile.ext
      intro idx
      simp

/-! ## The output store

A masked **`.fp16`-typed** `.ptr` store: the stored `acc` register is the
fp16 image of the biased rescaled value, so each active lane writes a `.fp16`
memory cell (`writeMemTyped .fp16`), and the readback is stated at the
`MemCell` level via `scatter_memcell_fp16_prop_masked_nd` — the
`matmul_dequantize` `matmul_kernel` precedent. `mask` is the only gate. The
lane-to-address map has to be injective for the readback to name a unique
lane; that is the headline's `hInj`, and `cAddr_injective` discharges it for
a row-major `C`. -/

/-- The `C` pointer tile. -/
noncomputable def i8CPtrs (C : RegionName) (stride_cm stride_cn BM BN pm pn : Nat) :
    Tile .ptr [BM, BN] :=
  ⟨fun idx => (C, cAddr stride_cm stride_cn BM BN pm pn idx)⟩

/-- `mask = (rm < M)[:, None] & (rn < N)[None, :]` — over the **unwrapped**
output coordinates. -/
def i8CMask (M N BM BN pm pn : Nat) : Tile .bool [BM, BN] :=
  ⟨fun idx => decide (pm * BM + idx.1.val < M) && decide (pn * BN + idx.2.1.val < N)⟩

/-- The post-store state: one masked **`.fp16`-typed** scatter over the
`[BM, BN]` output tile. -/
noncomputable def i8StoreState (C : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat)
    (f : TileIndex [BM, BN] → TileCarrier .fp16) (t : BlockState) : BlockState :=
  (TileShape.allIndices [BM, BN]).foldl
    (fun acc i => if pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N then
        acc.writeMemTyped .fp16 C (cAddr stride_cm stride_cn BM BN pm pn i) (f i)
      else acc) t

/-- `C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)`. -/
private theorem i8_cPtrsInit_eval (C : RegionName) (t : BlockState)
    (stride_cm stride_cn BM BN pm pn : Nat)
    (hrm : t.regs .nat [BM] "rm" = some (i8Offs (pm * BM) BM))
    (hrn : t.regs .nat [BN] "rn" = some (i8Offs (pn * BN) BN)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "rm"))
            (Op.constNat stride_cm))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "rn"))
            (Op.constNat stride_cn)))) t
      = some (i8CPtrs C stride_cm stride_cn BM BN pm pn) := by
  rw [i8_ptrAddBase_eval _ _ t _ _
    (i8_addTile_eval NumericDType.nat _ _ _ t _ _
      (i8_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_cm)
        (i8_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrm))
        (evalOp_constNat _ _))
      (i8_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_cn)
        (i8_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrn))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i8CPtrs, cAddr, i8Offs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `mask = (rm < M)[:, None] & (rn < N)[None, :]` — the comparisons happen at
rank 1 and the unit axes are inserted after (source order). -/
private theorem i8_cMask_eval (t : BlockState) (M N BM BN pm pn : Nat)
    (hrm : t.regs .nat [BM] "rm" = some (i8Offs (pm * BM) BM))
    (hrn : t.regs .nat [BN] "rn" = some (i8Offs (pn * BN) BN)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "rm")
            (Op.constNat M)))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "rn")
            (Op.constNat N)))) t
      = some (i8CMask M N BM BN pm pn) := by
  have hlt1 : evalOp ((Op.expandDim ⟨1, by simp⟩
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "rm")
          (Op.constNat M))) : Op .bool [BM, 1]) t
      = some ((Tile.expandDim ⟨1, by simp⟩
          (Tile.cop ComparableDType.nat.lt Broadcast.scalarR (i8Offs (pm * BM) BM)
            (Tile.scalar M))) : Tile .bool [BM, 1]) :=
    i8_expandDim_eval _ _ t _
      (i8_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar M) (by rw [evalOp_ref]; exact hrm) (evalOp_constNat _ _))
  have hlt2 : evalOp ((Op.expandDim ⟨0, by simp⟩
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "rn")
          (Op.constNat N))) : Op .bool [1, BN]) t
      = some ((Tile.expandDim ⟨0, by simp⟩
          (Tile.cop ComparableDType.nat.lt Broadcast.scalarR (i8Offs (pn * BN) BN)
            (Tile.scalar N))) : Tile .bool [1, BN]) :=
    i8_expandDim_eval _ _ t _
      (i8_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar N) (by rw [evalOp_ref]; exact hrn) (evalOp_constNat _ _))
  rw [i8_boolAnd_eval (Broadcast.consR (Broadcast.consL Broadcast.nil)) _ _ t _ _
    hlt1 hlt2]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [i8CMask, i8Offs, Tile.bop_data, Tile.cop_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]

/-- The masked `.fp16` store of the **`acc` register**. -/
private theorem i8_store_eq (C : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat) (t : BlockState)
    (vt : Tile .fp16 [BM, BN]) (f : TileIndex [BM, BN] → TileCarrier .fp16)
    (hfv : ∀ i, vt.data i = f i)
    (hcp : t.regs .ptr [BM, BN] "C"
      = some (i8CPtrs C stride_cm stride_cn BM BN pm pn))
    (hcmask : t.regs .bool [BM, BN] "mask" = some (i8CMask M N BM BN pm pn))
    (hv : t.regs .fp16 [BM, BN] "acc" = some vt) :
    stepStmt (Stmt.store .fp16 [BM, BN]
        (MemAccess.ptr (Op.ref .ptr [BM, BN] "C"))
        (Op.ref .fp16 [BM, BN] "acc")
        (MaskOpt.mask (Op.ref .bool [BM, BN] "mask"))) t
      = some (i8StoreState C M N stride_cm stride_cn BM BN pm pn f t) := by
  unfold stepStmt i8StoreState
  simp only [evalOp_ref, hv, hcp, hcmask, Option.map_some]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [BM, BN])) ?_)
  funext acc i
  obtain ⟨r, cc, u⟩ := i
  by_cases hb : pm * BM + r.val < M ∧ pn * BN + cc.val < N
  · simp only [i8CMask, if_pos hb, i8CPtrs, hfv]
    simp [hb.1, hb.2]
  · simp only [i8CMask, if_neg hb, i8CPtrs]
    rw [if_neg]
    intro hcon
    exact hb (by simpa using hcon)

private theorem i8_store_props (C : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat) (t : BlockState)
    (f : TileIndex [BM, BN] → TileCarrier .fp16)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN pm pn i)) :
    ∀ i : TileIndex [BM, BN],
      (pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N) →
      (i8StoreState C M N stride_cm stride_cn BM BN pm pn f t).mem C
          (cAddr stride_cm stride_cn BM BN pm pn i)
        = MemCell.of .fp16
            (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (f i))) := by
  classical
  intro i hi
  unfold i8StoreState
  have h := scatter_memcell_fp16_prop_masked_nd (region := C) t
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
  have hexp : ∀ r c : Nat, (pm * BM + r) * stride_cm + (pn * BN + c) * stride_cn
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

Six statements: the epilogue rescale, the fp16 downcast, the `has_bias`
branch, the `C` pointer tile, the two-axis mask, and the masked `.fp16` store
of `acc`. On every lane the mask lets through, `row < M` and `col < N` turn
the wrapped lane into the plain `i8Spec` — that is the `Nat.mod_eq_of_lt`
step — and the readback lands on the `.fp16` memory cell holding
`fp16(i8Spec)`. -/

theorem i8PostLoop_run (s0 : BlockState) (A B : Region .int)
    (C bias state_x state_w : RegionName) (divfactor : ℝ) (has_bias : Bool)
    (M N stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BM BN BK GM numKBlocks : Nat) (t : BlockState)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN
        (pidM s0 M N BM BN GM) (pidN s0 M N BM BN GM) i))
    (hinv : i8Inv s0 A B state_x state_w M N stride_am stride_ak stride_bk
      stride_bn BM BN BK GM numKBlocks numKBlocks t) :
    ∃ sF, stepStmts (i8PostLoop C bias divfactor has_bias M N stride_cm
        stride_cn BM BN) t = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (pidM s0 M N BM BN GM * BM + idx.1.val < M
            ∧ pidN s0 M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem C (cAddr stride_cm stride_cn BM BN (pidM s0 M N BM BN GM)
              (pidN s0 M N BM BN GM) idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (i8Spec s0 A B bias state_x state_w stride_am stride_ak
                  stride_bk stride_bn BK numKBlocks divfactor has_bias
                  (pidM s0 M N BM BN GM * BM + idx.1.val)
                  (pidN s0 M N BM BN GM * BN + idx.2.1.val)))) := by
  obtain ⟨-, hmem, -, hrm, hrn, hw, hx, -, -, hacc⟩ := hinv
  set pm := pidM s0 M N BM BN GM with hpmDef
  set pn := pidN s0 M N BM BN GM with hpnDef
  unfold i8PostLoop
  -- 1. the epilogue rescale
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_prod_eval s0 A B state_x state_w t M N stride_am stride_ak stride_bk
      stride_bn BM BN BK pm pn numKBlocks divfactor hw hx hacc))]
  -- 2. `acc = (acc).to(tl.float16)` — the genuine fp16 downcast
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some TileDType.fp16 [BM, BN] "acc"
    (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "acc")) _
    (⟨fun ix => FloatDType.real.cast FloatDType.fp16
        ((i8ProdTile s0 A B state_x state_w M N stride_am stride_ak stride_bk
          stride_bn BK numKBlocks divfactor BM BN pm pn).data ix)⟩
      : Tile .fp16 [BM, BN])
    (i8_castFp16_eval _ _ _ (by simp)))]
  -- 3. the `has_bias` branch (both arms) — the state is spelled explicitly
  -- (an `obtain` has no rewrite target to pin a `_` state against)
  obtain ⟨t3, hstep3, h3mem, h3keep, h3acc⟩ :=
    i8_biasBranch_run s0 bias
      ((t.setReg "acc" .real [BM, BN]
          (i8ProdTile s0 A B state_x state_w M N stride_am stride_ak stride_bk
            stride_bn BK numKBlocks divfactor BM BN pm pn)).setReg
        "acc" .fp16 [BM, BN]
        (⟨fun ix => FloatDType.real.cast FloatDType.fp16
            ((i8ProdTile s0 A B state_x state_w M N stride_am stride_ak
              stride_bk stride_bn BK numKBlocks divfactor BM BN pm pn).data ix)⟩
          : Tile .fp16 [BM, BN]))
      has_bias BM BN pn
      (fun ix => i8ProdSpec s0 A B state_x state_w stride_am stride_ak
        stride_bk stride_bn BK numKBlocks divfactor
        ((pm * BM + ix.1.val) % M) ((pn * BN + ix.2.1.val) % N))
      (by simpa [i8_setReg_mem] using hmem)
      (by simpa using hrn)
      (by simp; rfl)
  rw [stepStmts.cons_some hstep3]
  -- 4-5. the `C` pointer tile and the two-axis mask
  have h3rm : t3.regs .nat [BM] "rm" = some (i8Offs (pm * BM) BM) := by
    rw [h3keep .nat [BM] "rm" (by decide) (by decide)]
    simpa using hrm
  have h3rn : t3.regs .nat [BN] "rn" = some (i8Offs (pn * BN) BN) := by
    rw [h3keep .nat [BN] "rn" (by decide) (by decide)]
    simpa using hrn
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_cPtrsInit_eval C _ stride_cm stride_cn BM BN pm pn h3rm h3rn))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (i8_cMask_eval _ M N BM BN pm pn (by simpa using h3rm)
      (by simpa using h3rn)))]
  -- 6. the `.fp16` store of `acc`
  rw [stepStmts.cons_some
    (i8_store_eq C M N stride_cm stride_cn BM BN pm pn _
      (⟨fun ix => FloatDType.real.cast FloatDType.fp16
          (some (i8ProdSpec s0 A B state_x state_w stride_am stride_ak
              stride_bk stride_bn BK numKBlocks divfactor
              ((pm * BM + ix.1.val) % M) ((pn * BN + ix.2.1.val) % N)
            + (if has_bias then biasElem s0 bias (pn * BN + ix.2.1.val)
               else 0)))⟩ : Tile .fp16 [BM, BN])
      (fun ix => FloatDType.real.cast FloatDType.fp16
        (some (i8ProdSpec s0 A B state_x state_w stride_am stride_ak
            stride_bk stride_bn BK numKBlocks divfactor
            ((pm * BM + ix.1.val) % M) ((pn * BN + ix.2.1.val) % N)
          + (if has_bias then biasElem s0 bias (pn * BN + ix.2.1.val)
             else 0))))
      (fun _ => rfl)
      (by simp) (by simp) (by simpa using h3acc))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx hidx
  rw [i8_store_props C M N stride_cm stride_cn BM BN pm pn _ _ hInj idx hidx]
  simp only [Nat.mod_eq_of_lt hidx.1, Nat.mod_eq_of_lt hidx.2]
  simp only [i8Spec, FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue,
    FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot,
    FloatDType.fp16_toWithBot, WithBot.unbotD_some]

/-! ## Main theorem -/

set_option maxHeartbeats 1000000 in
-- `hK` is deliberately carried even though only the *loop bound* `numKBlocks`
-- appears semantically (`K` survives nowhere else in the `SPLIT_K = 1` /
-- `EVEN_K = True` arm): it is the launch fact — the `EVEN_K` heuristic
-- `K % (BLOCK_K * SPLIT_K) == 0` — that makes the `numKBlocks` presentation
-- of `tl.cdiv(K, BLOCK_K * SPLIT_K)` faithful.
set_option linter.unusedVariables false in
/-- **Genuine, dimension-general correctness.** For every launch state, the
`SPLIT_K = 1` / `EVEN_K = True` arm of the kernel runs to completion, and
every in-range output lane of `C` holds the `.fp16` memory cell carrying
`i8Spec`: the exact ℤ int8 matmul (`Op.dotInt` over all `numKBlocks` K
steps), rescaled per output cell in the kernel's own parenthesization
`w_factor * (x_factor * (acc * divfactor))`, plus the `has_bias`-guarded
per-column fp16 bias. The terminal store writes
`acc = ….to(tl.float16)` — a genuine fp16 quantization event, so the
conclusion is stated at the `MemCell` level:
`C[row, col] = MemCell.of .fp16 (fp16(i8Spec))` (the `matmul_dequantize`
`matmul_kernel` precedent; the placeholder `FloatDType.cast` is the
identity, so the carried value is `i8Spec` itself — the fp16 cast between
the product and the bias add likewise erases to the ℝ sum).

The hypotheses are the kernel's own launch facts: `hK` is the `EVEN_K`
heuristic `K % (BLOCK_K * SPLIT_K) == 0` (the loop trip count `numKBlocks`
is exact — the K-loop loads are unmasked); `hpid1` is grid axis 1 having
extent `SPLIT_K = 1`; `hInj` says distinct output lanes get distinct `C`
addresses — `cAddr_injective` discharges it for a row-major `C`. The
`% M` / `% N` offset wraps disappear on exactly the lanes the store mask
lets through (`Nat.mod_eq_of_lt`); the bias is read at the **unwrapped**
`rn`, so its lane is the plain `col` from the start. -/
specification int8_dequant_matmul_exec_genuine
    (A B : Region .int) (C bias state_x_ptr state_w_ptr : RegionName)
    (M N K : Nat) (divfactor : ℝ) (has_bias : Bool)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BM BN BK GM numKBlocks : Nat) (s : BlockState)
    (hK : K = BK * numKBlocks)
    (hpid1 : s.pids 1 = 0)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN
        (pidM s M N BM BN GM) (pidN s M N BM BN GM) i)) :
    ∃ sF, exec (int8_dequant_matmul_surface A B C bias state_x_ptr state_w_ptr
        M N K divfactor has_bias
        stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        BM BN BK GM numKBlocks).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (pidM s M N BM BN GM * BM + idx.1.val < M
            ∧ pidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem C (cAddr stride_cm stride_cn BM BN (pidM s M N BM BN GM)
              (pidN s M N BM BN GM) idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (i8Spec s A B bias state_x_ptr state_w_ptr
                  stride_am stride_ak stride_bk stride_bn BK numKBlocks
                  divfactor has_bias
                  (pidM s M N BM BN GM * BM + idx.1.val)
                  (pidN s M N BM BN GM * BN + idx.2.1.val)))) := by
  rw [exec, i8_body_eq]
  -- prologue: the scalars, then the index tiles
  obtain ⟨t1, hrun1, h1mem, h1pids, h1pz, h1pm, h1pn⟩ :=
    i8PreLoopScalars_run s M N BM BN GM
  obtain ⟨t2, hrun2, h2inv⟩ :=
    i8PreLoopTiles_run s A B state_x_ptr state_w_ptr M N stride_am stride_ak
      stride_bk stride_bn BM BN BK GM numKBlocks t1 h1mem h1pids
      (hpid1 ▸ h1pz) h1pm h1pn
  simp only [List.append_assoc]
  rw [stepStmts.append_some hrun1, stepStmts.append_some hrun2]
  -- the collapsed K loop
  obtain ⟨t3, hrun3, h3inv⟩ :=
    i8Loop_collapse s A B state_x_ptr state_w_ptr M N stride_am stride_ak
      stride_bk stride_bn BM BN BK GM numKBlocks t2 h2inv
  rw [show [Stmt.forRange "k" 0 numKBlocks 1
          (i8LoopBody stride_ak stride_bk BM BN BK)]
        ++ i8PostLoop C bias divfactor has_bias M N stride_cm stride_cn BM BN
      = Stmt.forRange "k" 0 numKBlocks 1
          (i8LoopBody stride_ak stride_bk BM BN BK)
        :: i8PostLoop C bias divfactor has_bias M N stride_cm stride_cn BM BN
      from rfl]
  rw [stepStmts.cons_some hrun3]
  -- the tail
  obtain ⟨sF, hpost, hout⟩ :=
    i8PostLoop_run s A B C bias state_x_ptr state_w_ptr divfactor has_bias
      M N stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BM BN BK GM numKBlocks t3 hInj h3inv
  exact ⟨sF, hpost, hout⟩

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.Int8DequantMatmul
