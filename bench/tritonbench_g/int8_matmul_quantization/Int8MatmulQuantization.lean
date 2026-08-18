import VeriTile.Triton

/-!
# `int8_matmul_quantization` — strict per-kernel correctness

This file is the DSL port of `int8_matmul_quantization.py`'s
`quantize_int8_perrow_kernel` (the audit anchor, the file's first JIT) —
launched by `quantize_int8_perrow`. The file's second JIT kernel,
`matmul_kernel` (launched by `matmul_int8`; the test drives both through
`matmul_quantize_int8`), is ported below it, in source order.

**Kernel 1** is a per-row float→int8 quantizer: program `pid_m` owns
`BLOCK_SIZE_M` rows of `fpa` and streams over the K dimension **twice**.
Pass 1 computes the running per-row abs-max
`a_max = max(a_max, tl.max(tl.abs(fpa), axis=1))` over K-masked
(`other=0.0`) blocks; then `a_scale = a_max / 127.`; pass 2 reloads the same
blocks, quantizes `inta = (fpa / a_scale[:, None]).to(tl.int8)`
(`Op.castRealToInt8` — truncation toward zero) and K-mask-stores the `.int`
tile into `a`; finally the per-row scale is stored (unmasked) into the fp16
vector `a_scale`. The row window is **wrapped** (`% M`) but the stores are
row-unmasked — the host grid is exactly `M // BLOCK_SIZE_M` programs of
`BLOCK_SIZE_M = 1` rows, so the headline carries the exact-tiling launch
fact `hFit : pid·BM + BM ≤ M` under which the wrap is the identity.

**Kernel 2** is the int8×int8→int32 GEMM consuming kernel 1's outputs:
the standard group-swizzled pid decomposition, `% M` / `% N` wrapped
row/col offsets, remainder-masked (`other=0.0`) `.int` K loads,
`accumulator += tl.dot(a, b)` on the integer channel (`Op.dotInt`), the two
1-D masked scale-factor loads, and the epilogue
`c = (accumulator.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(tl.float16)`
followed by the two-axis-masked fp16 store of `C`.

Translation-surface blocker: (`quantize_int8_perrow_kernel`) three disclosed
surface deviations, none semantic. **(1)** the loop bound
`tl.cdiv(K, BLOCK_SIZE_K)` is spelled as the antiquoted binder `numKBlocks`
— in **ceil form**: the K loads/stores are remainder-masked, so the headline
needs only the covering half `K ≤ numKBlocks · BLOCK_SIZE_K` (extra
all-masked blocks contribute `0` to the abs-max and store nothing; with
fewer blocks the statement would be false). **(2)** the terminal scale store
is spelled `tl.store(as_ptr + as_offs, (a_scale).to(tl.float16))`: the host
allocates `a_scale` as `torch.float16`, so the bare Python store carries an
implicit fp16 quantization event, made explicit as the DSL's fp16 store cast
(the `f8_conversion_utils` implicit-store-cast precedent) — the scale cells
are read back as **fp16-typed `MemCell`s**. **(3)** the scale literal `127.`
is spelled `127.0`, and integer literals inside index arithmetic are written
`$(n)`.

Translation-surface blocker: (`matmul_kernel`) four disclosed surface
deviations, none semantic. **(1)** `SPLIT_K` is fixed to `1` (the
`tl.store` arm): the `@triton.autotune` table sweeps `SPLIT_K ∈ {1, 2}` and
the launch grid's second axis is `SPLIT_K`, so the `tl.atomic_add` arm
(accumulating into the host-zeroed `C` of `reset_to_zero=['c_ptr']`) drops
with the constexpr — the `int8_dequant_matmul` disclosure-(1) precedent.
Every `* SPLIT_K` factor is folded to its `SPLIT_K = 1` value and the
headline carries the launch fact `s.pids 1 = 0`; `pid_sp_k` itself and the
faithful `offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)`
stay in the surface. **(2)** the epilogue's `accumulator.to(tl.float32)` is
carried by the implicit `Op.intToReal` promotion the DSL inserts at exactly
that site when the `.int` accumulator meets the ℝ-channel scale factors:
the explicit method-cast spelling is int-typed-ident-blocked in the DSL
(`tl.cast` rejects signed-integer casts and `tl.toReal` nat-pins its
argument), so the surface spells the product bare —
`(accumulator * a_scale[:, None] * b_scale[None, :]).to(tl.float16)` — and
the lowered term contains `Op.intToReal (Op.ref .int "accumulator")` in the
position the Python cast occupies. **(3)** the loop bound
`tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)` is the antiquoted `numKBlocks`, again
in the masked **ceil form** `K ≤ numKBlocks · BLOCK_SIZE_K` (the K loads
are remainder-masked with `other=0.0`, so the guarded terms vanish beyond
`K` — the `matmul_triton_autotune` masked-loop shape, honest at ragged
`K`). **(4)** integer literals inside index arithmetic are written `$(n)`.

## The `.int` regions and the int8 cast

Kernel 2's `A`/`B` and kernel 1's output `a` are `torch.int8` tensors,
modeled as `Region .int` binders: loads/stores are `.int`-typed (int8
values are signed) and `tl.dot` on that channel is the exact ℤ matmul
`Op.dotInt`. Kernel 1's `.to(tl.int8)` lowers to `Op.castRealToInt8`, whose
value semantics `tritonTruncTowardZero` truncates the real quotient toward
zero into the **unbounded** `.int` carrier — the faithfulness boundary of
the `#154` family (`quantize_copy_kv` / `dequantize_rowwise` precedents):
hardware saturation at `±127` is *not* modeled, so the statement is exact
precisely on rows whose quotients stay inside the int8 range (which the
per-row `a_max / 127` scale guarantees for finite inputs up to the
unmodeled fp16 rounding of the divisor in kernel 2's consumer). The `.int`
loads' `other=0.0` maps to `Op.constInt 0` (a mechanical `.int`-arm
addition to the DSL's `other=` elaboration table, mirroring the existing
`.nat` arm).

## Proof map

```
int8_matmul_quantization_quantize_exec_genuine   kernel-1 headline
├─ q_body_eq                    12 statements by `rfl`
├─ qPrologue_run                pid + offsets (wrap killed by hFit) + ptr tiles
├─ qLoop1_collapse              `forRange_inv` over qInv1 (streaming abs-max)
│  └─ qLoop1Body_run            masked load → qFpaTile; running max → qMaxTile
├─ qMid_run                     a_scale = a_max/127; fpa_ptrs rebind → qInv2 0
├─ qLoop2_collapse              `forRange_inv` over qInv2 (write history)
│  └─ qLoop2Body_run            load → int8 cast → masked `.int` scatter
└─ qTail_run                    as_offs + unmasked fp16 scale store
int8_matmul_quantization_matmul_exec_genuine     kernel-2 headline
├─ mm_body_eq                   28 statements by `rfl`
├─ mmPreLoopScalars_run         pids + trip counts + group swizzle
├─ mmPreLoopTiles_run           offsets, ptr tiles, masked scale loads, zeros
├─ mmLoop_collapse              `forRange_inv` over mmInv (masked dotInt)
│  └─ mmLoopBody_run            2 masked `.int` loads, `Op.dotInt`, advances
└─ mmPostLoop_run               epilogue + fp16 store (`MemCell` readback)
   └─ mmAccSpec_eq_finK         masked double sum = `∑ j : Fin K` under hK
```

Both headlines read every stored value bottom-up from the kernels' own
accessors over the **launch** state's memory; no output region is read back
into a spec, so no part of the trust path is self-referential.

## Modeling boundary

Arithmetic is over ℝ / ℤ (not bit-accurate IEEE float; the fp16 cast sites
are kept as explicit quantization events whose placeholder semantics is the
identity, and the int8 cast is the exact truncation described above).
`@triton.autotune` and the host launch (kernel 1's grid
`(M // BLOCK_SIZE_M,)` with `BLOCK_SIZE_M = 1`,
`BLOCK_SIZE_K = next_power_of_2(K)`; kernel 2's grid
`(cdiv(M,BM)·cdiv(N,BN), SPLIT_K)`) are the *trusted boundary*. Every
dimension, stride and block size stays a symbolic parameter.
-/

namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

section QuantizeKernel

/-! ## Kernel 1: `quantize_int8_perrow_kernel` — surface -/

set_option linter.unusedVariables false in
def int8_matmul_quantization_quantize_surface
    (fpa_ptr : RegionName) (a_ptr : Region .int) (as_ptr : RegionName)
    (M K : Nat)
    (stride_fpam stride_fpak stride_am stride_ak stride_asm : Nat)
    (BLOCK_SIZE_M BLOCK_SIZE_K numKBlocks : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  fpa_ptrs = fpa_ptr + offs_am[:, None] * $(stride_fpam) + offs_k[None, :] * $(stride_fpak)
  a_ptrs = $((a_ptr : Region .int)) + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  a_max = tl.zeros([$(BLOCK_SIZE_M)], dtype=tl.float32)
  for k in range($(0), $(numKBlocks), $(1)) {
    fpa = tl.load(fpa_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    a_max = tl.maximum(a_max, tl.max(tl.abs(fpa), axis=1))
    fpa_ptrs += $(BLOCK_SIZE_K) * $(stride_fpak)
  }
  a_scale = (a_max / 127.0)
  fpa_ptrs = fpa_ptr + offs_am[:, None] * $(stride_fpam) + offs_k[None, :] * $(stride_fpak)
  for k in range($(0), $(numKBlocks), $(1)) {
    fpa = tl.load(fpa_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    inta = (fpa / a_scale[:, None]).to(tl.int8)
    tl.store(a_ptrs, inta, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K))
    fpa_ptrs += $(BLOCK_SIZE_K) * $(stride_fpak)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
  }
  as_offs = pid_m * $(BLOCK_SIZE_M) * $(stride_asm) + tl.arange(0, $(BLOCK_SIZE_M))
  tl.store(as_ptr + as_offs, (a_scale).to(tl.float16))
}

theorem int8_matmul_quantization_quantize_surface_toAlgorithm_supported
    (fpa_ptr : RegionName) (a_ptr : Region .int) (as_ptr : RegionName)
    (M K stride_fpam stride_fpak stride_am stride_ak stride_asm : Nat)
    (BM BK numKBlocks : Nat) :
    ∃ alg, (int8_matmul_quantization_quantize_surface fpa_ptr a_ptr as_ptr M K
      stride_fpam stride_fpak stride_am stride_ak stride_asm
      BM BK numKBlocks).toAlgorithm? = Except.ok alg := by
  simp [int8_matmul_quantization_quantize_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-! ## Pure per-row specs -/

/-- `fpa[row, kg]` — the kernel's own address arithmetic over the launch
state's memory. -/
noncomputable def qFpaElem (s : BlockState) (fpa : RegionName)
    (sfm sfk : Nat) (row kg : Nat) : ℝ :=
  s.readMem fpa (row * sfm + kg * sfk)

/-- One K block's masked row abs-max: `max_e (if kb·BK+e < K then |f (kb·BK+e)| else 0)`
(the `other=0.0` fill of the masked lanes, pushed through `tl.abs`). Total
(`0` at `BK = 0`); the kernel's `tl.max` reduce forces `0 < BK` anyway. -/
noncomputable def qBlockMax (K BK : Nat) (f : Nat → ℝ) (kb : Nat) : ℝ :=
  if h : 0 < BK then
    Finset.univ.sup' ⟨⟨0, h⟩, Finset.mem_univ _⟩
      (fun e : Fin BK => if kb * BK + e.val < K then |f (kb * BK + e.val)| else 0)
  else 0

/-- **The streaming abs-max**, `0`-seeded: `a_max` after `i` K blocks. This IS
the honest spec — `a_max = tl.zeros(...)` seeds the running max at `0`. -/
noncomputable def qMaxPartial (K BK : Nat) (f : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | i + 1 => max (qMaxPartial K BK f i) (qBlockMax K BK f i)

/-- `a_scale = a_max / 127.` after the full first loop. -/
noncomputable def qScaleSpec (K BK numKBlocks : Nat) (f : Nat → ℝ) : ℝ :=
  qMaxPartial K BK f numKBlocks / 127

/-- **The stored int8 value**: `castRealToInt8`'s own truncation-toward-zero of
`fpa[row, kg] / a_scale[row]`. -/
noncomputable def qInt8Spec (K BK numKBlocks : Nat) (f : Nat → ℝ) (kg : Nat) : ℤ :=
  tritonTruncTowardZero (f kg / qScaleSpec K BK numKBlocks f)

/-! ## Offset / pointer / value tiles -/

/-- `base + tl.arange(0, BD)`. -/
def qOffs (base BD : Nat) : Tile .nat [BD] := ⟨fun idx => base + idx.1.val⟩

/-- The wrapped offset vector `(base + r) % Mm` — `offs_am` as computed. -/
def qWrapOffs (base BD Mm : Nat) : Tile .nat [BD] :=
  ⟨fun idx => (base + idx.1.val) % Mm⟩

/-- Under the exact-tiling launch fact `pm·BM + BM ≤ M` the row wrap is the
identity (the host grid is `M // BLOCK_SIZE_M` programs of `BLOCK_SIZE_M`
rows). -/
theorem qWrapOffs_eq_qOffs (pm BM M : Nat) (hFit : pm * BM + BM ≤ M) :
    qWrapOffs (pm * BM) BM M = qOffs (pm * BM) BM := by
  apply Tile.ext
  intro idx
  simp only [qWrapOffs, qOffs]
  exact Nat.mod_eq_of_lt (by have := idx.1.isLt; omega)

/-- `fpa_ptrs` lane `(r, e)` at K step `c`. -/
def qFpaAddr (sfm sfk BM BK pm c : Nat) (idx : TileIndex [BM, BK]) : Nat :=
  (pm * BM + idx.1.val) * sfm + idx.2.1.val * sfk + c * (BK * sfk)

/-- `a_ptrs` lane `(r, e)` at K step `c`. -/
def qAAddr (sam sak BM BK pm c : Nat) (idx : TileIndex [BM, BK]) : Nat :=
  (pm * BM + idx.1.val) * sam + idx.2.1.val * sak + c * (BK * sak)

/-- The absolute-`kg` form of the `a` store address. -/
def qAAddrK (sam sak BM pm : Nat) (r kg : Nat) : Nat :=
  (pm * BM + r) * sam + kg * sak

theorem qAAddr_eq_qAAddrK (sam sak BM BK pm c : Nat) (idx : TileIndex [BM, BK]) :
    qAAddr sam sak BM BK pm c idx
      = qAAddrK sam sak BM pm idx.1.val (c * BK + idx.2.1.val) := by
  simp only [qAAddr, qAAddrK]
  ring

noncomputable def qFpaPtrs (fpa : RegionName) (sfm sfk BM BK pm c : Nat) :
    Tile .ptr [BM, BK] :=
  ⟨fun idx => (fpa, qFpaAddr sfm sfk BM BK pm c idx)⟩

noncomputable def qAPtrs (Ac : RegionName) (sam sak BM BK pm c : Nat) :
    Tile .ptr [BM, BK] :=
  ⟨fun idx => (Ac, qAAddr sam sak BM BK pm c idx)⟩

theorem qFpaPtrs_succ (fpa : RegionName) (sfm sfk BM BK pm c : Nat) :
    Tile.ptrAdd Broadcast.scalarR (qFpaPtrs fpa sfm sfk BM BK pm c)
        (Tile.scalar (BK * sfk))
      = qFpaPtrs fpa sfm sfk BM BK pm (c + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, qFpaPtrs, qFpaAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

theorem qAPtrs_succ (Ac : RegionName) (sam sak BM BK pm c : Nat) :
    Tile.ptrAdd Broadcast.scalarR (qAPtrs Ac sam sak BM BK pm c)
        (Tile.scalar (BK * sak))
      = qAPtrs Ac sam sak BM BK pm (c + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, qAPtrs, qAAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

/-- The loaded `fpa` tile at K step `c`: masked lanes take the `other=0.0`
fill, active lanes read the launch state's `fpa` row. -/
noncomputable def qFpaTile (s : BlockState) (fpa : RegionName)
    (K sfm sfk BM BK pm c : Nat) : Tile .real [BM, BK] :=
  ⟨fun idx => some (if c * BK + idx.2.1.val < K then
      qFpaElem s fpa sfm sfk (pm * BM + idx.1.val) (c * BK + idx.2.1.val)
    else 0)⟩

/-- The running-max register tile after `i` K blocks. -/
noncomputable def qMaxTile (s : BlockState) (fpa : RegionName)
    (K sfm sfk BM BK pm : Nat) (i : Nat) : Tile .real [BM] :=
  ⟨fun idx => some (qMaxPartial K BK
      (qFpaElem s fpa sfm sfk (pm * BM + idx.1.val)) i)⟩

/-- The `a_scale` register tile. -/
noncomputable def qScaleTile (s : BlockState) (fpa : RegionName)
    (K sfm sfk BM BK numKBlocks pm : Nat) : Tile .real [BM] :=
  ⟨fun idx => some (qScaleSpec K BK numKBlocks
      (qFpaElem s fpa sfm sfk (pm * BM + idx.1.val)))⟩

/-- The `inta` register tile at K step `c` (its masked lanes are never
stored). -/
noncomputable def qIntaTile (s : BlockState) (fpa : RegionName)
    (K sfm sfk BM BK numKBlocks pm c : Nat) : Tile .int [BM, BK] :=
  ⟨fun idx => tritonTruncTowardZero
      ((if c * BK + idx.2.1.val < K then
          qFpaElem s fpa sfm sfk (pm * BM + idx.1.val) (c * BK + idx.2.1.val)
        else 0)
        / qScaleSpec K BK numKBlocks
            (qFpaElem s fpa sfm sfk (pm * BM + idx.1.val)))⟩

/-- The K-mask value tile at loop index `c`: lane `(r, e)` is
`decide (e < K - c·BK)` (nat truncated subtraction — the kernel's own mask
arithmetic). -/
def qKMask (K BM BK c : Nat) : Tile .bool [BM, BK] :=
  ⟨fun idx => decide (idx.2.1.val < K - c * BK)⟩

/-! ## Compiled body decomposition -/

/-- The lowered `mask=offs_k[None, :] < K - k * BLOCK_SIZE_K` operand (shared
by both loops' loads and by the store). -/
def qKMaskOp (K BM BK : Nat) : Op .bool [BM, BK] :=
  Op.remap [BM, BK] Broadcast.nil.consSame.consL.leftIndex
    (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
      (Op.sub .nat Broadcast.nil (Op.constNat K)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK))))

/-- The masked `fpa` load statement expression (both loops). -/
def qFpaLoadOp (K BM BK : Nat) : Op .real [BM, BK] :=
  Op.load .real (MemAccess.ptr (Op.ref .ptr [BM, BK] "fpa_ptrs"))
    (MaskOpt.maskOther (qKMaskOp K BM BK) ((Op.const 0.0).broadcast [BM, BK]))

/-- The lowered `tl.abs(fpa)` (a `where`/`lt`/`sub` triple — the DSL has no
`Op.abs`). -/
def qAbsOp (BM BK : Nat) : Op .real [BM, BK] :=
  Op.where
    (Op.lt ComparableDType.real Broadcast.scalarR
      (Op.ref .real [BM, BK] "fpa") (Op.const 0))
    (Op.sub .real Broadcast.scalarL (Op.const 0) (Op.ref .real [BM, BK] "fpa"))
    (Op.ref .real [BM, BK] "fpa")

/-- The lowered `tl.max(tl.abs(fpa), axis=1)`. -/
def qReduceOp (BM BK : Nat) : Op .real [BM] :=
  Op.reduceMax ⟨1, by simp⟩ Bool.false (qAbsOp BM BK)

/-- The lowered `tl.maximum(a_max, tl.max(tl.abs(fpa), axis=1))` — a
`gt`/`where` pair with the reduce operand duplicated (the DSL has no
`Op.max2` route for `tl.maximum`). -/
def qAbsMaxOp (BM BK : Nat) : Op .real [BM] :=
  Op.where
    (Op.gt ComparableDType.real (Broadcast.consSame Broadcast.nil)
      (Op.ref .real [BM] "a_max") (qReduceOp BM BK))
    (Op.ref .real [BM] "a_max") (qReduceOp BM BK)

/-- The prologue: program id, the two offset vectors, the two pointer tiles
and the zeroed running max. -/
def qPrologue (fpa_ptr : RegionName) (a_ptr : Region .int)
    (M stride_fpam stride_fpak stride_am stride_ak BM BK : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid_m" (Op.programId 0),
    Stmt.assign .nat [BK] "offs_k" (Op.arange BK),
    Stmt.assign .nat [BM] "offs_am"
      (Op.mod IntegralDType.nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
          (Op.arange BM))
        (Op.constNat M)),
    Stmt.assign .ptr [BM, BK] "fpa_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase fpa_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
            (Op.constNat stride_fpam))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_fpak)))),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase a_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
            (Op.constNat stride_am))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_ak)))),
    Stmt.assign .real [BM] "a_max" (Op.full [BM] (Op.const 0)) ]

/-- The abs-max loop body: masked load, running max, pointer advance. -/
def qLoop1Body (K stride_fpak BM BK : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BK] "fpa" (qFpaLoadOp K BM BK),
    Stmt.assign .real [BM] "a_max" (qAbsMaxOp BM BK),
    Stmt.assign .ptr [BM, BK] "fpa_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "fpa_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_fpak))) ]

/-- Between the loops: the `/127.` scale and the `fpa_ptrs` rebind to the
row start. -/
def qMid (fpa_ptr : RegionName) (stride_fpam stride_fpak BM BK : Nat) : List Stmt :=
  [ Stmt.assign .real [BM] "a_scale"
      (Op.div .real Broadcast.scalarR (Op.ref .real [BM] "a_max")
        (Op.const 127.0)),
    Stmt.assign .ptr [BM, BK] "fpa_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase fpa_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
            (Op.constNat stride_fpam))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_fpak)))) ]

/-- The quantize loop body: masked load, the int8 cast of the scaled row, the
K-masked `.int` store, both pointer advances. -/
def qLoop2Body (K stride_fpak stride_ak BM BK : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BK] "fpa" (qFpaLoadOp K BM BK),
    Stmt.assign .int [BM, BK] "inta"
      (Op.castRealToInt8 (Op.castFloat FloatDType.real FloatDType.real
        (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [BM, BK] "fpa")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "a_scale"))))),
    Stmt.store .int [BM, BK] (MemAccess.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
      (Op.ref .int [BM, BK] "inta") (MaskOpt.mask (qKMaskOp K BM BK)),
    Stmt.assign .ptr [BM, BK] "fpa_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "fpa_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_fpak))),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_ak))) ]

/-- The tail: the (unstrided-arange) `as_offs` vector and the unmasked
fp16-cast scale store. -/
def qTail (as_ptr : RegionName) (stride_asm BM : Nat) : List Stmt :=
  [ Stmt.assign .nat [BM] "as_offs"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
          (Op.constNat stride_asm))
        (Op.arange BM)),
    Stmt.store .fp16 [BM]
      (MemAccess.region (Region.cast as_ptr) (Op.ref .nat [BM] "as_offs"))
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM] "a_scale"))
      MaskOpt.none ]

set_option maxRecDepth 20000 in
/-- **Full body split (by `rfl`)** — 12 top-level statements. -/
theorem q_body_eq (fpa_ptr : RegionName) (a_ptr : Region .int) (as_ptr : RegionName)
    (M K stride_fpam stride_fpak stride_am stride_ak stride_asm : Nat)
    (BM BK numKBlocks : Nat) :
    (int8_matmul_quantization_quantize_surface fpa_ptr a_ptr as_ptr M K
        stride_fpam stride_fpak stride_am stride_ak stride_asm
        BM BK numKBlocks).toAlgKernel.body
      = qPrologue fpa_ptr a_ptr M stride_fpam stride_fpak stride_am stride_ak BM BK
        ++ [Stmt.forRange "k" 0 numKBlocks 1 (qLoop1Body K stride_fpak BM BK)]
        ++ qMid fpa_ptr stride_fpam stride_fpak BM BK
        ++ [Stmt.forRange "k" 0 numKBlocks 1 (qLoop2Body K stride_fpak stride_ak BM BK)]
        ++ qTail as_ptr stride_asm BM := by
  rfl

/-! ## Per-statement eval recipes (private copies — bench ports never import
each other) -/

/-- `setReg` leaves memory alone, at **function** level. -/
private theorem q_setReg_mem {dtype : TileDType} {sh : TileShape}
    (s : BlockState) (nm : RegName) (v : Tile dtype sh) :
    (s.setReg nm dtype sh v).mem = s.mem := rfl

/-- `Tile.scalar`'s single lane, as a loop-free rewrite (`Tile.scalar` itself
in a simp set loops against `scalar_eta`). -/
private theorem q_scalar_data {dtype : TileDType} (x : TileCarrier dtype)
    (u : TileIndex []) : (Tile.scalar x).data u = x := rfl


private theorem q_addTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.add h bc x y) t = some (Tile.bop h.add bc vx vy) := by
  rw [evalOp_add, hx, hy]
  rfl

private theorem q_subTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.sub h bc x y) t = some (Tile.bop h.sub bc vx vy) := by
  rw [evalOp_sub, hx, hy]
  rfl

private theorem q_mulTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mul h bc x y) t = some (Tile.bop h.mul bc vx vy) := by
  rw [evalOp_mul, hx, hy]
  rfl

private theorem q_divTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.div h bc x y) t = some (Tile.bop h.div bc vx vy) := by
  rw [evalOp_div, hx, hy]
  rfl

private theorem q_ltTile_eval {dtype : TileDType} (h : ComparableDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.lt h bc x y) t = some (Tile.cop h.lt bc vx vy) := by
  rw [evalOp_lt, hx, hy]
  rfl

private theorem q_gtTile_eval {dtype : TileDType} (h : ComparableDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.gt h bc x y) t = some (Tile.cop h.gt bc vx vy) := by
  rw [evalOp_gt, hx, hy]
  rfl

private theorem q_where_eval {dtype : TileDType} {sh : TileShape}
    (c : Op .bool sh) (x y : Op dtype sh) (t : BlockState)
    (vc : Tile .bool sh) (vx vy : Tile dtype sh)
    (hc : evalOp c t = some vc) (hx : evalOp x t = some vx)
    (hy : evalOp y t = some vy) :
    evalOp (Op.where c x y) t = some (Tile.select vc vx vy) := by
  rw [evalOp_where, hc, hx, hy]
  rfl

private theorem q_expandDim_eval {dtype : TileDType} {sh : TileShape}
    (ax : Fin (sh.length + 1)) (x : Op dtype sh) (t : BlockState)
    (v : Tile dtype sh) (hv : evalOp x t = some v) :
    evalOp (Op.expandDim ax x) t = some (Tile.expandDim ax v) := by
  rw [evalOp_expandDim, hv]
  rfl

private theorem q_full_eval {dtype : TileDType} (sh : TileShape) (e : Op dtype [])
    (t : BlockState) (v : Tile dtype []) (hv : evalOp e t = some v) :
    evalOp (Op.full sh e) t
      = some (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype sh) := by
  rw [evalOp_full, hv]
  rfl

private theorem q_mod_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mod IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.mod IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

private theorem q_ptrAdd_eval {a b : TileShape} {out : TileShape}
    (bc : Broadcast a b out)
    (pnm : RegName) (t : BlockState) (pt : Tile .ptr a) (off : Op .nat b)
    (ov : Tile .nat b)
    (hp : t.regs .ptr a pnm = some pt) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ref .ptr a pnm) off) t
      = some (Tile.ptrAdd bc pt ov) := by
  simp only [evalOp, evalOp_ref, hp, ho]
  rfl

private theorem q_ptrAddBase_eval {d : TileDType} {b out : TileShape}
    (bc : Broadcast [] b out) (rg : Region d) (t : BlockState)
    (off : Op .nat b) (ov : Tile .nat b) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ptrBase rg) off) t
      = some (Tile.ptrAdd bc (Tile.scalar ((Region.cast rg : RegionName), 0)) ov) := by
  simp only [evalOp, ho]
  rfl

private theorem q_mulScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mul .nat Broadcast.nil x y) t = some (Tile.scalar (u * v)) := by
  rw [evalOp_mul, hx, hy]
  rfl

/-- `name * c` on a `nat` scalar register. -/
private theorem q_mulRef_eval (t : BlockState) (nm : RegName) (val c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c)) t
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `pid_m * BLOCK + tl.arange(0, BLOCK)` from a scalar register. -/
private theorem q_offs_eval (nm : RegName) (t : BlockState) (BD base c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar base)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c))
        (Op.arange BD)) t
      = some (qOffs (base * c) BD) := by
  rw [q_addTile_eval NumericDType.nat Broadcast.scalarL _ _ t
    (Tile.scalar (base * c)) (Tile.vec (fun i => (i.val : Nat)))
    (q_mulRef_eval t nm base c hr)
    (evalOp_arange _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [qOffs, Tile.vec, Broadcast.rightIndex, NumericDType.add]

/-- The wrap statement `offs_am = (pid_m * BM + arange BM) % M`. -/
private theorem q_offsAm_eval (t : BlockState) (BM base M : Nat)
    (hr : t.regs .nat [] "pid_m" = some (Tile.scalar base)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
          (Op.arange BM))
        (Op.constNat M)) t
      = some (qWrapOffs (base * BM) BM M) := by
  rw [q_mod_eval Broadcast.scalarR _ _ t (qOffs (base * BM) BM) (Tile.scalar M)
    (q_offs_eval "pid_m" t BM base BM hr) (evalOp_constNat _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [qWrapOffs, qOffs, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]

/-- `fpa_ptrs = fpa_ptr + offs_am[:, None] * sfm + offs_k[None, :] * sfk` — at
K step `0`, under the unwrapped `offs_am` (the launch-fact rewrite). Also the
`fpa_ptrs` rebind between the loops. -/
private theorem q_fpaPtrsInit_eval (fpa_ptr : RegionName) (t : BlockState)
    (sfm sfk BM BK pm : Nat)
    (ham : t.regs .nat [BM] "offs_am" = some (qOffs (pm * BM) BM))
    (hk : t.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase fpa_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
            (Op.constNat sfm))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat sfk)))) t
      = some (qFpaPtrs fpa_ptr sfm sfk BM BK pm 0) := by
  rw [q_ptrAddBase_eval _ _ t _ _
    (q_addTile_eval NumericDType.nat _ _ _ t _ _
      (q_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar sfm)
        (q_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact ham))
        (evalOp_constNat _ _))
      (q_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar sfk)
        (q_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hk))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [qFpaPtrs, qFpaAddr, qOffs, Tile.vec, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- Same statement shape for the typed `a_ptrs` tile. -/
private theorem q_aPtrsInit_eval (a_ptr : Region .int) (t : BlockState)
    (sam sak BM BK pm : Nat)
    (ham : t.regs .nat [BM] "offs_am" = some (qOffs (pm * BM) BM))
    (hk : t.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase a_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
            (Op.constNat sam))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat sak)))) t
      = some (qAPtrs (Region.cast a_ptr) sam sak BM BK pm 0) := by
  rw [q_ptrAddBase_eval _ _ t _ _
    (q_addTile_eval NumericDType.nat _ _ _ t _ _
      (q_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar sam)
        (q_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact ham))
        (evalOp_constNat _ _))
      (q_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar sak)
        (q_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hk))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [qAPtrs, qAAddr, qOffs, Tile.vec, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `t.readMem` reads only the region's memory. -/
private theorem q_readMem_congr (t s0 : BlockState) (rg : RegionName)
    (hmem : ∀ o, t.mem rg o = s0.mem rg o) (a : Nat) :
    t.readMem rg a = s0.readMem rg a := by
  unfold BlockState.readMem
  rw [hmem]

/-- `expandDim ⟨0,_⟩` on a rank-1 nat register (private autotune-port clone). -/
private theorem q_evalOp_expandDim_zero_nat {D : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] name)) s =
      (s.regs .nat [D] name).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] => v.data (i.2.1, PUnit.unit) }
          : Tile .nat [1, D])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

/-- `Op.remap` eval (private clone — the autotune port keeps its copy local
too). -/
private theorem q_evalOp_remap {dtype shape} (outShape : TileShape)
    (map : TileIndex outShape → TileIndex shape) (a : Op dtype shape)
    (s : BlockState) :
    evalOp (.remap outShape map a) s = (do
      let v ← evalOp a s
      some (Tile.remap map v)) := by
  simp [evalOp]

/-- The K-mask operand evaluates to the honest per-lane tile
`decide (e < K - c·BK)` at loop index `c`. -/
private theorem qKMaskOp_eval (st : BlockState) (K BM BK c : Nat)
    (hk : st.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkk : st.regs .nat [] "k" = some (Tile.scalar c)) :
    evalOp (qKMaskOp K BM BK) st = some (qKMask K BM BK c) := by
  have heval : evalOp (qKMaskOp K BM BK) st = some
      (Tile.remap (dtype := .bool) Broadcast.nil.consSame.consL.leftIndex
        (Tile.cop ComparableDType.nat.lt Broadcast.scalarR
          (⟨fun i : TileIndex [1, BK] =>
            (Tile.vec (fun e : Fin BK => (e.val : Nat))).data (i.2.1, PUnit.unit)⟩
            : Tile .nat [1, BK])
          (Tile.bop NumericDType.nat.sub Broadcast.nil (Tile.scalar K)
            (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar c)
              (Tile.scalar BK))))) := by
    unfold qKMaskOp
    rw [q_evalOp_remap]
    conv_lhs => arg 1; rw [evalOp_lt]; arg 1; rw [q_evalOp_expandDim_zero_nat, hk]
    simp only [Option.bind_eq_bind, Option.bind_some, evalOp_sub, evalOp_mul,
      evalOp_constNat, evalOp_ref, hkk]
  rw [heval]
  refine congrArg some ?_
  apply Tile.ext
  intro i
  simp only [qKMask, Tile.remap, Tile.cop_data, Tile.vec, Tile.bop_data,
    Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt,
    NumericDType.sub, NumericDType.mul]
  rfl

/-- The broadcast `other=0.0` operand. -/
private theorem q_constBroadcast_eval (sh : TileShape) (t : BlockState) :
    evalOp ((Op.const 0.0).broadcast sh) t
      = some (⟨fun _ => some (0.0 : ℝ)⟩ : Tile .real sh) := by
  simp only [evalOp, evalOp_const, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- General masked-with-`other` `.ptr` load on any channel: active lanes read
memory, masked-off lanes take the `other` value. -/
private theorem q_load_maskOther_eval {dtype : TileDType} {sh : TileShape}
    (pnm : RegName) (maskOp : Op .bool sh) (otherOp : Op dtype sh) (t : BlockState)
    (pt : Tile .ptr sh) (mt : Tile .bool sh) (ot : Tile dtype sh)
    (hp : t.regs .ptr sh pnm = some pt)
    (hm : evalOp maskOp t = some mt) (ho : evalOp otherOp t = some ot) :
    evalOp (Op.load dtype (MemAccess.ptr (Op.ref .ptr sh pnm))
        (MaskOpt.maskOther maskOp otherOp)) t
      = some (⟨fun i => if mt.data i
            then t.readMemValue dtype (pt.data i).1 (pt.data i).2
            else ot.data i⟩ : Tile dtype sh) := by
  simp only [evalOp, evalOp_ref, hp, hm, ho, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- The masked `fpa` load lands on `qFpaTile` — over the *launch* state's
`fpa` cells (the loop-2 stores never touch the `fpa` region). -/
private theorem q_fpaLoad_eq (s0 : BlockState) (fpa : RegionName) (t : BlockState)
    (K sfm sfk BM BK pm c : Nat)
    (hmem : ∀ o, t.mem fpa o = s0.mem fpa o)
    (hp : t.regs .ptr [BM, BK] "fpa_ptrs" = some (qFpaPtrs fpa sfm sfk BM BK pm c))
    (hk : t.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar c)) :
    evalOp (qFpaLoadOp K BM BK) t = some (qFpaTile s0 fpa K sfm sfk BM BK pm c) := by
  unfold qFpaLoadOp
  rw [q_load_maskOther_eval "fpa_ptrs" _ _ t _ _ _ hp
    (qKMaskOp_eval t K BM BK c hk hkk) (q_constBroadcast_eval [BM, BK] t)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [qKMask, qFpaPtrs, qFpaTile, BlockState.readMemValue_real]
  by_cases hcond : c * BK + e.val < K
  · rw [if_pos (by simp only [decide_eq_true_eq]; omega), if_pos hcond]
    rw [q_readMem_congr t s0 fpa hmem]
    refine congrArg some (congrArg _ ?_)
    simp only [qFpaAddr]
    ring
  · rw [if_neg (by simp only [decide_eq_true_eq]; omega), if_neg hcond]
    norm_num

/-! ## The abs-max loop (loop 1) -/

/-- `tl.abs(fpa)` at K step `c`: active lanes hold `|fpa[row, kg]|`, masked
lanes `|0| = 0`. -/
noncomputable def qAbsTile (s : BlockState) (fpa : RegionName)
    (K sfm sfk BM BK pm c : Nat) : Tile .real [BM, BK] :=
  ⟨fun idx => some (if c * BK + idx.2.1.val < K then
      |qFpaElem s fpa sfm sfk (pm * BM + idx.1.val) (c * BK + idx.2.1.val)|
    else 0)⟩

/-- `tl.max(tl.abs(fpa), axis=1)` at K step `c` — the per-row block max. -/
noncomputable def qBlockMaxTile (s : BlockState) (fpa : RegionName)
    (K sfm sfk BM BK pm c : Nat) : Tile .real [BM] :=
  ⟨fun idx => some (qBlockMax K BK
      (qFpaElem s fpa sfm sfk (pm * BM + idx.1.val)) c)⟩

/-- The lowered `tl.abs` (`where`/`lt`/`sub`) lands on `qAbsTile`. -/
private theorem q_abs_eval (s0 : BlockState) (fpa : RegionName) (t : BlockState)
    (K sfm sfk BM BK pm c : Nat)
    (hfpa : t.regs .real [BM, BK] "fpa"
      = some (qFpaTile s0 fpa K sfm sfk BM BK pm c)) :
    evalOp (qAbsOp BM BK) t = some (qAbsTile s0 fpa K sfm sfk BM BK pm c) := by
  unfold qAbsOp
  rw [q_where_eval _ _ _ t _ _ _
    (q_ltTile_eval ComparableDType.real Broadcast.scalarR _ _ t _ _
      (by rw [evalOp_ref]; exact hfpa) (evalOp_const 0 t))
    (q_subTile_eval NumericDType.real Broadcast.scalarL _ _ t _ _
      (evalOp_const 0 t) (by rw [evalOp_ref]; exact hfpa))
    (by rw [evalOp_ref]; exact hfpa)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [qFpaTile, qAbsTile, Tile.select, Tile.cop_data, Tile.bop_data,
    Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt,
    NumericDType.sub, WithBot.realSub, q_scalar_data, decide_eq_true_eq]
  set v : ℝ := if c * BK + e.val < K then
      qFpaElem s0 fpa sfm sfk (pm * BM + r.val) (c * BK + e.val)
    else 0 with hv
  have habs : (if c * BK + e.val < K then
      |qFpaElem s0 fpa sfm sfk (pm * BM + r.val) (c * BK + e.val)|
    else 0) = |v| := by
    rw [hv]
    by_cases hc : c * BK + e.val < K
    · rw [if_pos hc, if_pos hc]
    · rw [if_neg hc, if_neg hc, abs_zero]
  rw [habs]
  split_ifs with hneg
  · -- the `Option ℝ` order on two `some` lanes unfolds to the real order
    have hv0 : v < 0 := WithBot.coe_lt_coe.mp hneg
    show (some ((0 : ℝ) - v) : Option ℝ) = some |v|
    rw [abs_of_neg hv0]
    norm_num
  · have hv0 : ¬v < 0 := fun h => hneg (WithBot.coe_lt_coe.mpr h)
    rw [abs_of_nonneg (not_lt.mp hv0)]

/-- The `tl.max(..., axis=1)` reduce lands on `qBlockMaxTile` (needs
`0 < BLOCK_SIZE_K`: the reduce is a `sup'` over the K lanes). -/
private theorem q_reduce_eval (s0 : BlockState) (fpa : RegionName) (t : BlockState)
    (K sfm sfk BM BK pm c : Nat) (hBK : 0 < BK)
    (hfpa : t.regs .real [BM, BK] "fpa"
      = some (qFpaTile s0 fpa K sfm sfk BM BK pm c)) :
    evalOp (qReduceOp BM BK) t
      = some (qBlockMaxTile s0 fpa K sfm sfk BM BK pm c) := by
  unfold qReduceOp
  erw [evalOp_reduceMax]
  rw [q_abs_eval s0 fpa t K sfm sfk BM BK pm c hfpa]
  simp only [Option.bind_eq_bind, Option.bind_some, Tile.reduceMax_false]
  unfold Tile.reduceMaxDrop
  rw [dif_pos (show (0 : Nat) < TileShape.axisDim [BM, BK] ⟨1, by simp⟩ from hBK)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, u⟩ := idx
  show Finset.univ.sup' ⟨⟨0, hBK⟩, Finset.mem_univ _⟩
      (fun k : Fin BK =>
        (qAbsTile s0 fpa K sfm sfk BM BK pm c).data
          (TileShape.insertAxisIndex [BM, BK] ⟨1, by simp⟩ ((r, u) : TileIndex [BM]) k))
    = (qBlockMaxTile s0 fpa K sfm sfk BM BK pm c).data (r, u)
  simp only [qAbsTile, qBlockMaxTile]
  rw [WithBot.sup'_someTerm_eq_some]
  refine congrArg some ?_
  unfold qBlockMax
  rw [dif_pos hBK]
  rfl

/-- The running-max statement: `tl.maximum(a_max, tl.max(tl.abs(fpa), axis=1))`
extends the streaming max by one block (`qMaxPartial (i+1)` unfolds to exactly
`max prev blockMax`). -/
private theorem q_absMax_eval (s0 : BlockState) (fpa : RegionName) (t : BlockState)
    (K sfm sfk BM BK pm i : Nat) (hBK : 0 < BK)
    (hfpa : t.regs .real [BM, BK] "fpa"
      = some (qFpaTile s0 fpa K sfm sfk BM BK pm i))
    (hamax : t.regs .real [BM] "a_max"
      = some (qMaxTile s0 fpa K sfm sfk BM BK pm i)) :
    evalOp (qAbsMaxOp BM BK) t
      = some (qMaxTile s0 fpa K sfm sfk BM BK pm (i + 1)) := by
  unfold qAbsMaxOp
  rw [q_where_eval _ _ _ t _ _ _
    (q_gtTile_eval ComparableDType.real (Broadcast.consSame Broadcast.nil) _ _ t _ _
      (by rw [evalOp_ref]; exact hamax)
      (q_reduce_eval s0 fpa t K sfm sfk BM BK pm i hBK hfpa))
    (by rw [evalOp_ref]; exact hamax)
    (q_reduce_eval s0 fpa t K sfm sfk BM BK pm i hBK hfpa)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, u⟩ := idx
  simp only [qMaxTile, qBlockMaxTile, Tile.select, Tile.cop_data,
    Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt,
    decide_eq_true_eq, gt_iff_lt, qMaxPartial]
  set p : ℝ := qMaxPartial K BK (qFpaElem s0 fpa sfm sfk (pm * BM + r.val)) i
  set b : ℝ := qBlockMax K BK (qFpaElem s0 fpa sfm sfk (pm * BM + r.val)) i
  split_ifs with hgt
  · have hbp : b < p := WithBot.coe_lt_coe.mp hgt
    rw [max_eq_left (le_of_lt hbp)]
  · have hbp : ¬b < p := fun h => hgt (WithBot.coe_lt_coe.mpr h)
    rw [max_eq_right (not_lt.mp hbp)]

/-! ### The loop-1 invariant -/

set_option linter.unusedVariables false in
/-- The state carried across the abs-max loop. Memory is untouched; `a_ptrs`
sits at block `0` throughout (loop 1 never advances it), and `offs_am` is the
(unwrapped, by the launch fact) row window the mid-section rebind reads. -/
noncomputable def qInv1 (s0 : BlockState) (fpa : RegionName) (a_ptr : Region .int)
    (M K sfm sfk sam sak BM BK numKBlocks : Nat) (i : Nat) (s : BlockState) : Prop :=
  i ≤ numKBlocks
  ∧ s.mem = s0.mem
  ∧ s.pids = s0.pids
  ∧ s.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))
  ∧ s.regs .nat [BM] "offs_am" = some (qOffs (s0.pids 0 * BM) BM)
  ∧ s.regs .ptr [BM, BK] "fpa_ptrs" = some (qFpaPtrs fpa sfm sfk BM BK (s0.pids 0) i)
  ∧ s.regs .ptr [BM, BK] "a_ptrs"
      = some (qAPtrs (Region.cast a_ptr) sam sak BM BK (s0.pids 0) 0)
  ∧ s.regs .real [BM] "a_max" = some (qMaxTile s0 fpa K sfm sfk BM BK (s0.pids 0) i)

/-- The loop combinator writes the induction variable before each iteration;
`qInv1` constrains no register named `"k"`. -/
theorem qInv1_setReg_k (s0 : BlockState) (fpa : RegionName) (a_ptr : Region .int)
    (M K sfm sfk sam sak BM BK numKBlocks i j : Nat) (s : BlockState)
    (h : qInv1 s0 fpa a_ptr M K sfm sfk sam sak BM BK numKBlocks i s) :
    qInv1 s0 fpa a_ptr M K sfm sfk sam sak BM BK numKBlocks i
      (s.setReg "k" .nat [] (Tile.scalar j)) := by
  obtain ⟨hle, hmem, hpids, hpm, hk, ham, hfp, hap, hmax⟩ := h
  exact ⟨hle, hmem, hpids, by simpa using hpm, by simpa using hk,
    by simpa using ham, by simpa using hfp, by simpa using hap,
    by simpa using hmax⟩

set_option linter.unusedVariables false in
/-- One abs-max loop iteration. -/
theorem qLoop1Body_run (s0 : BlockState) (fpa : RegionName) (a_ptr : Region .int)
    (M K sfm sfk sam sak BM BK numKBlocks i : Nat) (hBK : 0 < BK)
    (s : BlockState)
    (hnext : i + 1 ≤ numKBlocks)
    (hkk : s.regs .nat [] "k" = some (Tile.scalar i))
    (hinv : qInv1 s0 fpa a_ptr M K sfm sfk sam sak BM BK numKBlocks i s) :
    ∃ s', stepStmts (qLoop1Body K sfk BM BK) s = some s'
      ∧ qInv1 s0 fpa a_ptr M K sfm sfk sam sak BM BK numKBlocks (i + 1) s' := by
  obtain ⟨-, hmem, hpids, hpm, hk, ham, hfp, hap, hmax⟩ := hinv
  unfold qLoop1Body
  -- 1. `fpa = tl.load(fpa_ptrs, mask=…, other=0.0)`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (q_fpaLoad_eq s0 fpa s K sfm sfk BM BK (s0.pids 0) i
      (fun o => by rw [hmem]) hfp hk hkk))]
  -- 2. `a_max = tl.maximum(a_max, tl.max(tl.abs(fpa), axis=1))`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (q_absMax_eval s0 fpa _ K sfm sfk BM BK (s0.pids 0) i hBK
      (by simp) (by simpa using hmax)))]
  -- 3. `fpa_ptrs += BLOCK_SIZE_K * stride_fpak`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "fpa_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sfk))) _
      = some (qFpaPtrs fpa sfm sfk BM BK (s0.pids 0) (i + 1)) from by
      rw [← qFpaPtrs_succ]
      exact q_ptrAdd_eval Broadcast.scalarR "fpa_ptrs" _ _ _
        (Tile.scalar (BK * sfk)) (by simpa using hfp)
        (q_mulScalarNat_eval _ _ _ BK sfk (evalOp_constNat _ _)
          (evalOp_constNat _ _))))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, hnext, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [q_setReg_mem]
    exact hmem
  · simp only [BlockState.setReg_pids]
    exact hpids
  · simpa using hpm
  · simpa using hk
  · simpa using ham
  · simp
  · simpa using hap
  · simp

/-- Collapsing the abs-max loop. -/
theorem qLoop1_collapse (s0 : BlockState) (fpa : RegionName) (a_ptr : Region .int)
    (M K sfm sfk sam sak BM BK numKBlocks : Nat) (hBK : 0 < BK) (s : BlockState)
    (h0 : qInv1 s0 fpa a_ptr M K sfm sfk sam sak BM BK numKBlocks 0 s) :
    ∃ sF, stepStmt (Stmt.forRange "k" 0 numKBlocks 1 (qLoop1Body K sfk BM BK)) s
        = some sF
      ∧ qInv1 s0 fpa a_ptr M K sfm sfk sam sak BM BK numKBlocks numKBlocks sF := by
  obtain ⟨final, sF, hrun, hfinal, hP⟩ :=
    forRange_inv (idx := "k") (start := 0) (stop := numKBlocks) (step := 1)
      (P := fun i t => qInv1 s0 fpa a_ptr M K sfm sfk sam sak BM BK numKBlocks i t)
      one_ne_zero h0
      (fun i t hi hinv => by
        obtain ⟨s', hs', hinv'⟩ :=
          qLoop1Body_run s0 fpa a_ptr M K sfm sfk sam sak BM BK numKBlocks i hBK
            _ (by omega) (by simp)
            (qInv1_setReg_k s0 fpa a_ptr M K sfm sfk sam sak BM BK numKBlocks
              i i t hinv)
        exact ⟨s', hs', hinv'⟩)
  have hEq : final = numKBlocks := le_antisymm hP.1 hfinal
  subst hEq
  exact ⟨sF, hrun, hP⟩

/-- The prologue walk: from any launch state satisfying the exact-tiling fact,
the six prologue statements land on `qInv1 … 0`. -/
theorem qPrologue_run (s0 : BlockState) (fpa : RegionName) (a_ptr : Region .int)
    (M K sfm sfk sam sak BM BK numKBlocks : Nat)
    (hFit : s0.pids 0 * BM + BM ≤ M) :
    ∃ t, stepStmts (qPrologue fpa a_ptr M sfm sfk sam sak BM BK) s0 = some t
      ∧ qInv1 s0 fpa a_ptr M K sfm sfk sam sak BM BK numKBlocks 0 t := by
  unfold qPrologue
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s0))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BK _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod IntegralDType.nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
          (Op.arange BM))
        (Op.constNat M)) _
      = some (qOffs (s0.pids 0 * BM) BM) from by
      rw [q_offsAm_eval _ BM (s0.pids 0) M (by simp),
        qWrapOffs_eq_qOffs (s0.pids 0) BM M hFit]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (q_fpaPtrsInit_eval fpa _ sfm sfk BM BK (s0.pids 0) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (q_aPtrsInit_eval a_ptr _ sam sak BM BK (s0.pids 0) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM] (Op.const 0)) _
        = some (qMaxTile s0 fpa K sfm sfk BM BK (s0.pids 0) 0) from by
      rw [q_full_eval [BM] (Op.const 0) _ _ (evalOp_const 0 _)]
      refine congrArg some ?_
      apply Tile.ext
      intro idx
      simp [qMaxTile, qMaxPartial]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [q_setReg_mem]
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp

/-! ## Memory infrastructure for the two-store tail

Mem-cell level, dtype-generic where possible; the `.int` scatter readback
carries a **mask-restricted** injectivity hypothesis (global tile injectivity
is false for the host's row-major strides once the masked K lanes run past
`K`). -/

/-- `writeMemTyped` at any other cell. -/
private theorem q_writeMemTyped_mem_other {dtype : TileDType} (s : BlockState)
    (rg : RegionName) (a : Nat) (v : TileCarrier dtype) (ρ : RegionName) (o : Nat)
    (h : ¬(rg = ρ ∧ a = o)) :
    (s.writeMemTyped dtype rg a v).mem ρ o = s.mem ρ o := by
  cases dtype <;>
    · show (if ρ = rg ∧ o = a then _ else s.mem ρ o) = s.mem ρ o
      rw [if_neg (fun hc => h ⟨hc.1.symm, hc.2.symm⟩)]

/-- `writeMemTyped .int` at its own cell. -/
private theorem q_writeMemTyped_int_mem_self (s : BlockState)
    (rg : RegionName) (a : Nat) (v : Int) :
    (s.writeMemTyped .int rg a v).mem rg a = MemCell.of .int v := by
  show (if rg = rg ∧ a = a then MemCell.of .int v else s.mem rg a)
    = MemCell.of .int v
  rw [if_pos ⟨rfl, rfl⟩]

/-- `writeMemTyped` keeps registers. -/
private theorem q_writeMemTyped_regs {dtype : TileDType} (s : BlockState)
    (rg : RegionName) (a : Nat) (v : TileCarrier dtype) :
    (s.writeMemTyped dtype rg a v).regs = s.regs := by
  cases dtype <;> rfl

/-- `writeMemTyped` keeps program ids. -/
private theorem q_writeMemTyped_pids {dtype : TileDType} (s : BlockState)
    (rg : RegionName) (a : Nat) (v : TileCarrier dtype) :
    (s.writeMemTyped dtype rg a v).pids = s.pids := by
  cases dtype <;> rfl

/-- A masked scatter foldl keeps registers. -/
private theorem q_foldl_write_regs {dtype : TileDType} {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → TileCarrier dtype)
    (P : α → Prop) [DecidablePred P] (l : List α) :
    ∀ s : BlockState,
      ((l.foldl (fun acc k =>
          if P k then acc.writeMemTyped dtype region (offsetFn k) (valueFn k)
          else acc) s)).regs = s.regs := by
  induction l with
  | nil => intro s; rfl
  | cons hd tl ih =>
      intro s
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP, ih, q_writeMemTyped_regs]
      · rw [if_neg hP, ih]

/-- A masked scatter foldl keeps program ids. -/
private theorem q_foldl_write_pids {dtype : TileDType} {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → TileCarrier dtype)
    (P : α → Prop) [DecidablePred P] (l : List α) :
    ∀ s : BlockState,
      ((l.foldl (fun acc k =>
          if P k then acc.writeMemTyped dtype region (offsetFn k) (valueFn k)
          else acc) s)).pids = s.pids := by
  induction l with
  | nil => intro s; rfl
  | cons hd tl ih =>
      intro s
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP, ih, q_writeMemTyped_pids]
      · rw [if_neg hP, ih]

/-- Cell-level frame for a masked scatter foldl: every cell whose (region,
offset) is missed by all active writes is unchanged. -/
private theorem q_foldl_write_mem_preserve {dtype : TileDType} {α : Type}
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
          q_writeMemTyped_mem_other _ _ _ _ _ _ (h hd List.mem_cons_self hP)]
      · rw [if_neg hP]
        exact ih _ htl

/-- `.int` scatter readback with **mask-restricted** injectivity: at an active
lane's cell, the foldl leaves exactly that lane's `MemCell.of .int` write. -/
private theorem q_scatter_int_mem {region : RegionName} {shape : TileShape}
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
  rw [q_foldl_write_mem_preserve (dtype := .int) region offsetFn valueFn P region
    (offsetFn i) l₂ _ h_l2]
  rw [if_pos hPi]
  exact q_writeMemTyped_int_mem_self _ _ _ _

/-! ## The quantize loop (loop 2) -/

/-- The `inta = (fpa / a_scale[:, None]).to(tl.int8)` statement lands on
`qIntaTile`. -/
private theorem q_inta_eval (s0 : BlockState) (fpa : RegionName) (t : BlockState)
    (K sfm sfk BM BK numKBlocks pm c : Nat)
    (hfpa : t.regs .real [BM, BK] "fpa"
      = some (qFpaTile s0 fpa K sfm sfk BM BK pm c))
    (hscale : t.regs .real [BM] "a_scale"
      = some (qScaleTile s0 fpa K sfm sfk BM BK numKBlocks pm)) :
    evalOp (Op.castRealToInt8 (Op.castFloat FloatDType.real FloatDType.real
        (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [BM, BK] "fpa")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "a_scale"))))) t
      = some (qIntaTile s0 fpa K sfm sfk BM BK numKBlocks pm c) := by
  have hexp : evalOp ((Op.expandDim ⟨1, by simp⟩
        (Op.ref .real [BM] "a_scale")) : Op .real [BM, 1]) t
      = some (Tile.expandDim ⟨1, by simp⟩
          (qScaleTile s0 fpa K sfm sfk BM BK numKBlocks pm)) :=
    q_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hscale)
  have hdiv := q_divTile_eval NumericDType.real
    (Broadcast.consSame (Broadcast.consR Broadcast.nil)) _ _ t _ _
    (show evalOp (Op.ref .real [BM, BK] "fpa") t
        = some (qFpaTile s0 fpa K sfm sfk BM BK pm c) from by
      rw [evalOp_ref]; exact hfpa)
    hexp
  have hcast : evalOp (Op.castFloat FloatDType.real FloatDType.real
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BK] "fpa")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "a_scale")))) t
      = some (Tile.bop NumericDType.real.div
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (qFpaTile s0 fpa K sfm sfk BM BK pm c)
          (Tile.expandDim ⟨1, by simp⟩
            (qScaleTile s0 fpa K sfm sfk BM BK numKBlocks pm))) := by
    rw [evalOp_castFloat]
    erw [hdiv]
    simp only [Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    apply Tile.ext
    intro idx
    simp [FloatDType.cast]
  simp only [evalOp]
  erw [hcast]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [qIntaTile, qFpaTile, qScaleTile, Tile.bop_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.div, WithBot.realDiv, Option.map₂, Option.map_some,
    Option.bind_some, Option.bind_eq_bind, WithBot.realToInt8_some]

/-- The masked `.int` store of `inta` steps to the named scatter state. -/
private theorem q_store_eq (Ac : RegionName) (t : BlockState)
    (K sam sak BM BK pm c : Nat) (vt : Tile .int [BM, BK])
    (hap : t.regs .ptr [BM, BK] "a_ptrs" = some (qAPtrs Ac sam sak BM BK pm c))
    (hv : t.regs .int [BM, BK] "inta" = some vt)
    (hk : t.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar c)) :
    stepStmt (Stmt.store .int [BM, BK] (MemAccess.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        (Op.ref .int [BM, BK] "inta") (MaskOpt.mask (qKMaskOp K BM BK))) t
      = some ((TileShape.allIndices [BM, BK]).foldl
          (fun acc i => if c * BK + i.2.1.val < K then
              acc.writeMemTyped .int Ac (qAAddr sam sak BM BK pm c i) (vt.data i)
            else acc) t) := by
  unfold stepStmt
  simp only [evalOp_ref, hv, hap, qKMaskOp_eval t K BM BK c hk hkk,
    Option.map_some, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [BM, BK])) ?_)
  funext acc i
  obtain ⟨r, e, u⟩ := i
  by_cases hb : c * BK + e.val < K
  · rw [if_pos (show (qKMask K BM BK c).data (r, e, u) = Bool.true from by
      simp only [qKMask, decide_eq_true_eq]; omega)]
    rw [if_pos hb]
    rfl
  · rw [if_neg (show ¬((qKMask K BM BK c).data (r, e, u) = Bool.true) from by
      simp only [qKMask, decide_eq_true_eq]; omega)]
    rw [if_neg hb]

/-- The state carried across the quantize loop: the advancing pointer pair,
the pinned scale register, and the **write history** — every already-covered
`(row, kg)` cell holds its quantized `MemCell`, and the `fpa` region is
untouched (which is what keeps the loads reading launch-state rows). -/
noncomputable def qInv2 (s0 : BlockState) (fpa : RegionName) (a_ptr : Region .int)
    (K sfm sfk sam sak BM BK numKBlocks : Nat) (i : Nat) (s : BlockState) : Prop :=
  i ≤ numKBlocks
  ∧ s.pids = s0.pids
  ∧ (∀ o, s.mem fpa o = s0.mem fpa o)
  ∧ s.regs .nat [] "pid_m" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))
  ∧ s.regs .real [BM] "a_scale"
      = some (qScaleTile s0 fpa K sfm sfk BM BK numKBlocks (s0.pids 0))
  ∧ s.regs .ptr [BM, BK] "fpa_ptrs"
      = some (qFpaPtrs fpa sfm sfk BM BK (s0.pids 0) i)
  ∧ s.regs .ptr [BM, BK] "a_ptrs"
      = some (qAPtrs (Region.cast a_ptr) sam sak BM BK (s0.pids 0) i)
  ∧ (∀ (r : Fin BM) (kg : Nat), kg < K → kg < i * BK →
      s.mem (Region.cast a_ptr) (qAAddrK sam sak BM (s0.pids 0) r.val kg)
        = MemCell.of .int (qInt8Spec K BK numKBlocks
            (qFpaElem s0 fpa sfm sfk (s0.pids 0 * BM + r.val)) kg))

theorem qInv2_setReg_k (s0 : BlockState) (fpa : RegionName) (a_ptr : Region .int)
    (K sfm sfk sam sak BM BK numKBlocks i j : Nat) (s : BlockState)
    (h : qInv2 s0 fpa a_ptr K sfm sfk sam sak BM BK numKBlocks i s) :
    qInv2 s0 fpa a_ptr K sfm sfk sam sak BM BK numKBlocks i
      (s.setReg "k" .nat [] (Tile.scalar j)) := by
  obtain ⟨hle, hpids, hfmem, hpm, hk, hsc, hfp, hap, hcontent⟩ := h
  exact ⟨hle, hpids, hfmem, by simpa using hpm, by simpa using hk,
    by simpa using hsc, by simpa using hfp, by simpa using hap, hcontent⟩

set_option maxHeartbeats 1000000 in
/-- One quantize-loop iteration: the masked load reads the untouched `fpa`
rows, the store lays down block `i`'s quantized cells (mask-restricted
injectivity from the headline's `hInj`), and the previously-written cells
survive (later blocks live at strictly larger `kg`). -/
theorem qLoop2Body_run (s0 : BlockState) (fpa : RegionName) (a_ptr : Region .int)
    (K sfm sfk sam sak BM BK numKBlocks i : Nat) (s : BlockState)
    (hnext : i + 1 ≤ numKBlocks)
    (hFpaNe : fpa ≠ (Region.cast a_ptr : RegionName))
    (hInj : Function.Injective (fun p : Fin BM × Fin K =>
      qAAddrK sam sak BM (s0.pids 0) p.1.val p.2.val))
    (hkk : s.regs .nat [] "k" = some (Tile.scalar i))
    (hinv : qInv2 s0 fpa a_ptr K sfm sfk sam sak BM BK numKBlocks i s) :
    ∃ s', stepStmts (qLoop2Body K sfk sak BM BK) s = some s'
      ∧ qInv2 s0 fpa a_ptr K sfm sfk sam sak BM BK numKBlocks (i + 1) s' := by
  classical
  obtain ⟨-, hpids, hfmem, hpm, hk, hsc, hfp, hap, hcontent⟩ := hinv
  set pm := s0.pids 0 with hpmDef
  -- the two register states before / after the store
  set t2 := (s.setReg "fpa" .real [BM, BK]
      (qFpaTile s0 fpa K sfm sfk BM BK pm i)).setReg
    "inta" .int [BM, BK] (qIntaTile s0 fpa K sfm sfk BM BK numKBlocks pm i)
    with ht2Def
  set S := (TileShape.allIndices [BM, BK]).foldl
      (fun acc j => if i * BK + j.2.1.val < K then
          acc.writeMemTyped .int (Region.cast a_ptr)
            (qAAddr sam sak BM BK pm i j)
            ((qIntaTile s0 fpa K sfm sfk BM BK numKBlocks pm i).data j)
        else acc) t2
    with hSdef
  have hSregs : S.regs = t2.regs := by
    rw [hSdef]
    exact q_foldl_write_regs (dtype := .int) _ _ _ _ _ t2
  have hSpids : S.pids = t2.pids := by
    rw [hSdef]
    exact q_foldl_write_pids (dtype := .int) _ _ _ _ _ t2
  unfold qLoop2Body
  -- 1. the masked load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (q_fpaLoad_eq s0 fpa s K sfm sfk BM BK pm i hfmem hfp hk hkk))]
  -- 2. the quantized tile
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (q_inta_eval s0 fpa _ K sfm sfk BM BK numKBlocks pm i (by simp)
      (by simpa using hsc)))]
  -- 3. the masked `.int` store
  rw [stepStmts.cons_some (q_store_eq (Region.cast a_ptr) t2 K sam sak BM BK pm i
    (qIntaTile s0 fpa K sfm sfk BM BK numKBlocks pm i)
    (by rw [ht2Def]; simpa using hap) (by rw [ht2Def]; simp)
    (by rw [ht2Def]; simpa using hk) (by rw [ht2Def]; simpa using hkk))]
  rw [← hSdef]
  -- 4. `fpa_ptrs` advance
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "fpa_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sfk))) S
      = some (qFpaPtrs fpa sfm sfk BM BK pm (i + 1)) from by
      rw [← qFpaPtrs_succ]
      refine q_ptrAdd_eval Broadcast.scalarR "fpa_ptrs" _ _ _
        (Tile.scalar (BK * sfk)) ?_
        (q_mulScalarNat_eval _ _ _ BK sfk (evalOp_constNat _ _)
          (evalOp_constNat _ _))
      rw [hSregs, ht2Def]
      simpa using hfp))]
  -- 5. `a_ptrs` advance
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sak)))
        (S.setReg "fpa_ptrs" .ptr [BM, BK] (qFpaPtrs fpa sfm sfk BM BK pm (i + 1)))
      = some (qAPtrs (Region.cast a_ptr) sam sak BM BK pm (i + 1)) from by
      rw [← qAPtrs_succ]
      refine q_ptrAdd_eval Broadcast.scalarR "a_ptrs" _ _ _
        (Tile.scalar (BK * sak)) ?_
        (q_mulScalarNat_eval _ _ _ BK sak (evalOp_constNat _ _)
          (evalOp_constNat _ _))
      rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs, ht2Def]
      simpa using hap))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, hnext, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]
    rw [hSpids, ht2Def]
    simpa using hpids
  · -- the `fpa` region survives the `.int` scatter (distinct region name)
    intro o
    simp only [q_setReg_mem]
    rw [hSdef, q_foldl_write_mem_preserve (dtype := .int) _ _ _ _ fpa o _ _
      (fun j _ _ hc => hFpaNe hc.1.symm)]
    rw [ht2Def]
    simp only [q_setReg_mem]
    exact hfmem o
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs, ht2Def]
    simpa using hpm
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs, ht2Def]
    simpa using hk
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs, ht2Def]
    simpa using hsc
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    simp
    rfl
  · simp
    rfl
  · -- the write history, one block further
    intro r kg hkgK hkgBlk
    simp only [q_setReg_mem]
    by_cases hOld : kg < i * BK
    · -- an earlier block's cell: missed by every one of block `i`'s writes
      have hmiss : ∀ j ∈ TileShape.allIndices [BM, BK],
          i * BK + (j : TileIndex [BM, BK]).2.1.val < K →
          ¬((Region.cast a_ptr : RegionName) = Region.cast a_ptr
            ∧ qAAddr sam sak BM BK pm i j
              = qAAddrK sam sak BM pm r.val kg) := by
        rintro ⟨r', e', u'⟩ - hb hc
        have haddr : qAAddrK sam sak BM pm r'.val (i * BK + e'.val)
            = qAAddrK sam sak BM pm r.val kg := by
          have h0 := hc.2
          rw [qAAddr_eq_qAAddrK] at h0
          exact h0
        have heq : (⟨r', ⟨i * BK + e'.val, hb⟩⟩ : Fin BM × Fin K)
            = ⟨r, ⟨kg, hkgK⟩⟩ := hInj haddr
        have hkgEq : i * BK + e'.val = kg := by
          have := congrArg (fun p : Fin BM × Fin K => p.2.val) heq
          simpa using this
        omega
      rw [hSdef, q_foldl_write_mem_preserve (dtype := .int) _ _ _ _
        (Region.cast a_ptr) (qAAddrK sam sak BM pm r.val kg) _ _ hmiss]
      rw [ht2Def]
      simp only [q_setReg_mem]
      exact hcontent r kg hkgK hOld
    · -- block `i`'s own cell: the scatter readback at lane `(r, kg - i·BK)`
      have hkgLo : i * BK ≤ kg := not_lt.mp hOld
      have hkgHi : kg < i * BK + BK := by
        have h0 := hkgBlk
        rw [Nat.succ_mul] at h0
        exact h0
      have hlaneK : i * BK + (kg - i * BK) < K := by omega
      have hInjLanes : ∀ k₁ k₂ : TileIndex [BM, BK],
          i * BK + k₁.2.1.val < K → i * BK + k₂.2.1.val < K →
          qAAddr sam sak BM BK pm i k₁ = qAAddr sam sak BM BK pm i k₂ →
          k₁ = k₂ := by
        rintro ⟨r₁, e₁, u₁⟩ ⟨r₂, e₂, u₂⟩ hb₁ hb₂ hc
        have h1 : qAAddrK sam sak BM pm r₁.val (i * BK + e₁.val)
            = qAAddrK sam sak BM pm r₂.val (i * BK + e₂.val) := by
          have h0 := hc
          rw [qAAddr_eq_qAAddrK, qAAddr_eq_qAAddrK] at h0
          exact h0
        have heq := hInj (a₁ := ⟨r₁, ⟨i * BK + e₁.val, hb₁⟩⟩)
          (a₂ := ⟨r₂, ⟨i * BK + e₂.val, hb₂⟩⟩) h1
        have hr : r₁ = r₂ := congrArg Prod.fst heq
        have he : e₁.val = e₂.val := by
          have := congrArg (fun p : Fin BM × Fin K => p.2.val) heq
          simp only at this
          omega
        exact Prod.ext hr (Prod.ext (Fin.ext he) rfl)
      have hread := q_scatter_int_mem (region := Region.cast a_ptr)
        (shape := [BM, BK]) t2
        (fun j : TileIndex [BM, BK] => qAAddr sam sak BM BK pm i j)
        (fun j => (qIntaTile s0 fpa K sfm sfk BM BK numKBlocks pm i).data j)
        (fun j : TileIndex [BM, BK] => i * BK + j.2.1.val < K)
        hInjLanes (r, ⟨kg - i * BK, by omega⟩, PUnit.unit) hlaneK
      have haddr : qAAddr sam sak BM BK pm i (r, ⟨kg - i * BK, by omega⟩, PUnit.unit)
          = qAAddrK sam sak BM pm r.val kg := by
        rw [qAAddr_eq_qAAddrK]
        show qAAddrK sam sak BM pm r.val (i * BK + (kg - i * BK))
          = qAAddrK sam sak BM pm r.val kg
        congr 1
        omega
      rw [hSdef]
      show ((TileShape.allIndices [BM, BK]).foldl _ t2).mem (Region.cast a_ptr)
          (qAAddrK sam sak BM pm r.val kg)
        = MemCell.of .int (qInt8Spec K BK numKBlocks
            (qFpaElem s0 fpa sfm sfk (pm * BM + r.val)) kg)
      have hlane : i * BK + (kg - i * BK) = kg := by omega
      rw [← haddr, hread]
      refine congrArg _ ?_
      simp only [qIntaTile, qInt8Spec]
      rw [hlane, if_pos hkgK]

theorem qLoop2_collapse (s0 : BlockState) (fpa : RegionName) (a_ptr : Region .int)
    (K sfm sfk sam sak BM BK numKBlocks : Nat) (s : BlockState)
    (hFpaNe : fpa ≠ (Region.cast a_ptr : RegionName))
    (hInj : Function.Injective (fun p : Fin BM × Fin K =>
      qAAddrK sam sak BM (s0.pids 0) p.1.val p.2.val))
    (h0 : qInv2 s0 fpa a_ptr K sfm sfk sam sak BM BK numKBlocks 0 s) :
    ∃ sF, stepStmt (Stmt.forRange "k" 0 numKBlocks 1
          (qLoop2Body K sfk sak BM BK)) s = some sF
      ∧ qInv2 s0 fpa a_ptr K sfm sfk sam sak BM BK numKBlocks numKBlocks sF := by
  obtain ⟨final, sF, hrun, hfinal, hP⟩ :=
    forRange_inv (idx := "k") (start := 0) (stop := numKBlocks) (step := 1)
      (P := fun i t => qInv2 s0 fpa a_ptr K sfm sfk sam sak BM BK numKBlocks i t)
      one_ne_zero h0
      (fun i t hi hinv => by
        obtain ⟨s', hs', hinv'⟩ :=
          qLoop2Body_run s0 fpa a_ptr K sfm sfk sam sak BM BK numKBlocks i _
            (by omega) hFpaNe hInj (by simp)
            (qInv2_setReg_k s0 fpa a_ptr K sfm sfk sam sak BM BK numKBlocks
              i i t hinv)
        exact ⟨s', hs', hinv'⟩)
  have hEq : final = numKBlocks := le_antisymm hP.1 hfinal
  subst hEq
  exact ⟨sF, hrun, hP⟩

/-! ## Between the loops -/

/-- `a_scale = (a_max / 127.)` lands on `qScaleTile` (the `127.0` literal is
the real number `127`). -/
private theorem q_scale_eval (s0 : BlockState) (fpa : RegionName) (t : BlockState)
    (K sfm sfk BM BK numKBlocks pm : Nat)
    (hamax : t.regs .real [BM] "a_max"
      = some (qMaxTile s0 fpa K sfm sfk BM BK pm numKBlocks)) :
    evalOp (Op.div .real Broadcast.scalarR (Op.ref .real [BM] "a_max")
        (Op.const 127.0)) t
      = some (qScaleTile s0 fpa K sfm sfk BM BK numKBlocks pm) := by
  rw [q_divTile_eval NumericDType.real Broadcast.scalarR _ _ t _ _
    (by rw [evalOp_ref]; exact hamax) (evalOp_const 127.0 t)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [qScaleTile, qScaleSpec, qMaxTile, Tile.bop_data, Broadcast.leftIndex,
    Broadcast.rightIndex, q_scalar_data, NumericDType.div, WithBot.realDiv,
    Option.map₂, Option.map_some, Option.bind_some, Option.bind_eq_bind]
  norm_num

/-- The mid-section: the scale statement and the `fpa_ptrs` rebind land on
`qInv2 … 0` (block-`0` content is vacuous; memory is still the launch
memory). -/
theorem qMid_run (s0 : BlockState) (fpa : RegionName) (a_ptr : Region .int)
    (M K sfm sfk sam sak BM BK numKBlocks : Nat) (t : BlockState)
    (hinv : qInv1 s0 fpa a_ptr M K sfm sfk sam sak BM BK numKBlocks numKBlocks t) :
    ∃ t', stepStmts (qMid fpa sfm sfk BM BK) t = some t'
      ∧ qInv2 s0 fpa a_ptr K sfm sfk sam sak BM BK numKBlocks 0 t' := by
  obtain ⟨-, hmem, hpids, hpm, hk, ham, hfp, hap, hmax⟩ := hinv
  unfold qMid
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (q_scale_eval s0 fpa t K sfm sfk BM BK numKBlocks (s0.pids 0) hmax))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (q_fpaPtrsInit_eval fpa _ sfm sfk BM BK (s0.pids 0)
      (by simpa using ham) (by simpa using hk)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]
    exact hpids
  · intro o
    simp only [q_setReg_mem]
    rw [hmem]
  · simpa using hpm
  · simpa using hk
  · simp
  · simp
  · simpa using hap
  · intro r kg _ hkg0
    rw [Nat.zero_mul] at hkg0
    exact absurd hkg0 (Nat.not_lt_zero _)

/-! ## The tail: the unstrided `as_offs` vector and the fp16 scale store -/

/-- `as_offs = pid_m * BLOCK_SIZE_M * stride_asm + tl.arange(0, BLOCK_SIZE_M)`
(the arange is **unstrided** — the source's own quirk, spelled faithfully). -/
private theorem q_asOffs_eval (t : BlockState) (BM sasm base : Nat)
    (hr : t.regs .nat [] "pid_m" = some (Tile.scalar base)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
          (Op.constNat sasm))
        (Op.arange BM)) t
      = some (qOffs (base * BM * sasm) BM) := by
  rw [q_addTile_eval NumericDType.nat Broadcast.scalarL _ _ t
    (Tile.scalar (base * BM * sasm)) (Tile.vec (fun i => (i.val : Nat)))
    (q_mulScalarNat_eval _ _ t (base * BM) sasm
      (q_mulRef_eval t "pid_m" base BM hr) (evalOp_constNat _ _))
    (evalOp_arange _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [qOffs, Tile.vec, Broadcast.rightIndex, NumericDType.add]

/-- Cell-level frame for an **unconditional** scatter foldl. -/
private theorem q_foldl_write_mem_preserve' {dtype : TileDType} {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → TileCarrier dtype)
    (ρ : RegionName) (o : Nat) (l : List α) :
    ∀ s : BlockState, (∀ k ∈ l, ¬(region = ρ ∧ offsetFn k = o)) →
      ((l.foldl (fun acc k =>
          acc.writeMemTyped dtype region (offsetFn k) (valueFn k)) s)).mem ρ o
        = s.mem ρ o := by
  induction l with
  | nil => intro s _; rfl
  | cons hd tl ih =>
      intro s h
      rw [List.foldl_cons, ih _ (fun k hk => h k (List.mem_cons_of_mem hd hk)),
        q_writeMemTyped_mem_other _ _ _ _ _ _ (h hd List.mem_cons_self)]

/-- The unmasked fp16 scale store steps to the unconditional scatter state. -/
private theorem q_scaleStore_eq (as_ptr : RegionName) (t : BlockState)
    (BM base : Nat) (vt : Tile .real [BM])
    (hoffs : t.regs .nat [BM] "as_offs" = some (qOffs base BM))
    (hsc : t.regs .real [BM] "a_scale" = some vt) :
    stepStmt (Stmt.store .fp16 [BM]
        (MemAccess.region (Region.cast as_ptr) (Op.ref .nat [BM] "as_offs"))
        (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM] "a_scale"))
        MaskOpt.none) t
      = some ((TileShape.allIndices [BM]).foldl
          (fun acc i => acc.writeMemTyped .fp16 as_ptr (base + i.1.val)
            (FloatDType.real.cast FloatDType.fp16 (vt.data i))) t) := by
  have hval : @evalOp TileDType.fp16 [BM]
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM] "a_scale")) t
      = some (⟨fun i => FloatDType.real.cast FloatDType.fp16 (vt.data i)⟩
          : Tile .fp16 [BM]) := by
    show evalOp (Op.castFloat FloatDType.real FloatDType.fp16
        (Op.ref .real [BM] "a_scale")) t
      = some ⟨fun i => FloatDType.real.cast FloatDType.fp16 (vt.data i)⟩
    rw [evalOp_castFloat]
    simp [evalOp_ref, hsc]
  unfold stepStmt
  rw [hval]
  simp only [evalOp_ref, hoffs, Option.bind_eq_bind, Option.bind_some,
    Option.map_some]
  rfl

/-- The scale-store address map is injective. -/
private theorem q_asAddr_injective (base BM : Nat) :
    Function.Injective (fun i : TileIndex [BM] => base + i.1.val) := by
  intro a b h
  obtain ⟨a1, u1⟩ := a
  obtain ⟨b1, u2⟩ := b
  have h' : base + a1.val = base + b1.val := h
  exact Prod.ext (Fin.ext (Nat.add_left_cancel h')) rfl

/-- **The tail walk**: from the loop-2 invariant at `numKBlocks`, the two tail
statements terminate, the fp16 scale cells land, and the int8 cells survive
(the scale store writes a different region). -/
theorem qTail_run (s0 : BlockState) (fpa : RegionName) (a_ptr : Region .int)
    (as_ptr : RegionName)
    (K sfm sfk sam sak sasm BM BK numKBlocks : Nat) (t : BlockState)
    (hAsNe : as_ptr ≠ (Region.cast a_ptr : RegionName))
    (hinv : qInv2 s0 fpa a_ptr K sfm sfk sam sak BM BK numKBlocks numKBlocks t) :
    ∃ sF, stepStmts (qTail as_ptr sasm BM) t = some sF
      ∧ (∀ (r : Fin BM) (kg : Nat), kg < K → kg < numKBlocks * BK →
          sF.mem (Region.cast a_ptr) (qAAddrK sam sak BM (s0.pids 0) r.val kg)
            = MemCell.of .int (qInt8Spec K BK numKBlocks
                (qFpaElem s0 fpa sfm sfk (s0.pids 0 * BM + r.val)) kg))
      ∧ (∀ r : Fin BM,
          sF.mem as_ptr (s0.pids 0 * BM * sasm + r.val)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (qScaleSpec K BK numKBlocks
                  (qFpaElem s0 fpa sfm sfk (s0.pids 0 * BM + r.val)))))) := by
  obtain ⟨-, -, -, hpm, -, hsc, -, -, hcontent⟩ := hinv
  set pm := s0.pids 0 with hpmDef
  unfold qTail
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (q_asOffs_eval t BM sasm pm hpm))]
  rw [stepStmts.cons_some (q_scaleStore_eq as_ptr _ BM (pm * BM * sasm)
    (qScaleTile s0 fpa K sfm sfk BM BK numKBlocks pm)
    (by simp) (by simpa using hsc))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · -- the int8 cells survive the fp16 scale store (different region)
    intro r kg hkgK hkgBlk
    rw [q_foldl_write_mem_preserve' as_ptr _ _ (Region.cast a_ptr)
      (qAAddrK sam sak BM pm r.val kg) _ _
      (fun j _ hc => hAsNe hc.1)]
    simp only [q_setReg_mem]
    exact hcontent r kg hkgK hkgBlk
  · -- the fp16 scale cells
    intro r
    have h := scatter_memcell_fp16_nd
      (region := as_ptr) (shape := [BM])
      ((t.setReg "as_offs" .nat [BM] (qOffs (pm * BM * sasm) BM)))
      (fun i : TileIndex [BM] => pm * BM * sasm + i.1.val)
      (fun i => FloatDType.real.cast FloatDType.fp16
        ((qScaleTile s0 fpa K sfm sfk BM BK numKBlocks pm).data i))
      (q_asAddr_injective (pm * BM * sasm) BM) (r, PUnit.unit)
    rw [h]
    simp only [qScaleTile, FloatDType.cast, FloatDType.ofReal,
      FloatDType.storeValue, FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot,
      FloatDType.fp16_toWithBot, WithBot.unbotD_some]

set_option maxHeartbeats 1000000 in
set_option linter.unusedVariables false in
/-- **Genuine, dimension-general correctness of `quantize_int8_perrow_kernel`
(the audit anchor).** From any launch state satisfying the host's own launch
facts, the kernel runs to completion and

* every `(row, kg)` cell of the int8 output `a` holds the `MemCell.of .int`
  carrying `castRealToInt8`'s truncation-toward-zero of
  `fpa[row, kg] / a_scale[row]`, where `a_scale[row] = a_max[row] / 127` and
  `a_max[row]` is the **kernel-computed streaming abs-max**
  `qMaxPartial` — the `0`-seeded running max over all `numKBlocks` masked
  K blocks (`max(prev, max_e (if kb·BK+e < K then |fpa[row, kb·BK+e]| else 0))`
  — the `other=0.0` fill is absorbed by the `0` seed);
* every `a_scale` cell of the fp16 scale vector holds the fp16 cell of the
  same per-row scale (the store's implicit fp16 cast — the host allocates
  `a_scale` as `torch.float16`).

The hypotheses are the launch facts: `hBK` (the `tl.max` reduce needs a
nonempty K tile — `BLOCK_SIZE_K = next_power_of_2(K)` on the host), `hK`
(`numKBlocks = cdiv(K, BLOCK_SIZE_K)` covers `K` — the ceil half that is
semantically forced; extra all-masked blocks contribute `0` to the max and
store nothing), `hFit` (the grid is `M // BLOCK_SIZE_M` programs of
`BLOCK_SIZE_M` rows — exact tiling, so the row-unmasked stores stay in their
window and the `% M` wrap is the identity), the two region-distinctness facts
(the quantize loop reads `fpa` after writing `a`; the scale store must not
clobber `a`), and the store-map injectivity `hInj` (row-major `a`). -/
specification int8_matmul_quantization_quantize_exec_genuine
    (fpa_ptr : RegionName) (a_ptr : Region .int) (as_ptr : RegionName)
    (M K : Nat)
    (stride_fpam stride_fpak stride_am stride_ak stride_asm : Nat)
    (BM BK numKBlocks : Nat) (s : BlockState)
    (hBK : 0 < BK)
    (hK : K ≤ numKBlocks * BK)
    (hFit : s.pids 0 * BM + BM ≤ M)
    (hFpaNe : fpa_ptr ≠ (Region.cast a_ptr : RegionName))
    (hAsNe : as_ptr ≠ (Region.cast a_ptr : RegionName))
    (hInj : Function.Injective (fun p : Fin BM × Fin K =>
      (s.pids 0 * BM + p.1.val) * stride_am + p.2.val * stride_ak)) :
    ∃ sF, exec (int8_matmul_quantization_quantize_surface fpa_ptr a_ptr as_ptr
        M K stride_fpam stride_fpak stride_am stride_ak stride_asm
        BM BK numKBlocks).toAlgKernel s = some sF
      ∧ (∀ (r : Fin BM) (kg : Fin K),
          sF.mem (Region.cast a_ptr)
              ((s.pids 0 * BM + r.val) * stride_am + kg.val * stride_ak)
            = MemCell.of .int (qInt8Spec K BK numKBlocks
                (qFpaElem s fpa_ptr stride_fpam stride_fpak
                  (s.pids 0 * BM + r.val)) kg.val))
      ∧ (∀ r : Fin BM,
          sF.mem as_ptr (s.pids 0 * BM * stride_asm + r.val)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (qScaleSpec K BK numKBlocks
                  (qFpaElem s fpa_ptr stride_fpam stride_fpak
                    (s.pids 0 * BM + r.val)))))) := by
  rw [exec, q_body_eq]
  obtain ⟨t1, hrun1, hinv10⟩ :=
    qPrologue_run s fpa_ptr a_ptr M K stride_fpam stride_fpak stride_am
      stride_ak BM BK numKBlocks hFit
  simp only [List.append_assoc]
  rw [stepStmts.append_some hrun1]
  obtain ⟨t2, hrun2, hinv1F⟩ :=
    qLoop1_collapse s fpa_ptr a_ptr M K stride_fpam stride_fpak stride_am
      stride_ak BM BK numKBlocks hBK t1 hinv10
  rw [show [Stmt.forRange "k" 0 numKBlocks 1
        (qLoop1Body K stride_fpak BM BK)]
      ++ (qMid fpa_ptr stride_fpam stride_fpak BM BK
        ++ ([Stmt.forRange "k" 0 numKBlocks 1
              (qLoop2Body K stride_fpak stride_ak BM BK)]
          ++ qTail as_ptr stride_asm BM))
    = Stmt.forRange "k" 0 numKBlocks 1 (qLoop1Body K stride_fpak BM BK)
      :: (qMid fpa_ptr stride_fpam stride_fpak BM BK
        ++ ([Stmt.forRange "k" 0 numKBlocks 1
              (qLoop2Body K stride_fpak stride_ak BM BK)]
          ++ qTail as_ptr stride_asm BM)) from rfl]
  rw [stepStmts.cons_some hrun2]
  obtain ⟨t3, hrun3, hinv20⟩ :=
    qMid_run s fpa_ptr a_ptr M K stride_fpam stride_fpak stride_am stride_ak
      BM BK numKBlocks t2 hinv1F
  rw [stepStmts.append_some hrun3]
  obtain ⟨t4, hrun4, hinv2F⟩ :=
    qLoop2_collapse s fpa_ptr a_ptr K stride_fpam stride_fpak stride_am
      stride_ak BM BK numKBlocks t3 hFpaNe hInj hinv20
  rw [show [Stmt.forRange "k" 0 numKBlocks 1
        (qLoop2Body K stride_fpak stride_ak BM BK)]
      ++ qTail as_ptr stride_asm BM
    = Stmt.forRange "k" 0 numKBlocks 1 (qLoop2Body K stride_fpak stride_ak BM BK)
      :: qTail as_ptr stride_asm BM from rfl]
  rw [stepStmts.cons_some hrun4]
  obtain ⟨sF, hrun5, hint, hscale⟩ :=
    qTail_run s fpa_ptr a_ptr as_ptr K stride_fpam stride_fpak stride_am
      stride_ak stride_asm BM BK numKBlocks t4 hAsNe hinv2F
  refine ⟨sF, hrun5, ?_, hscale⟩
  intro r kg
  exact hint r kg.val kg.isLt (lt_of_lt_of_le kg.isLt hK)

end QuantizeKernel

section MatmulKernel

/-! # `matmul_kernel` — the int8×int8→int32 GEMM with per-row/per-column
dequantization scales

The file's second JIT kernel: an `SPLIT_K = 1` grouped-swizzle GEMM whose `A`
and `B` operands live on the signed `.int` channel, whose K-loop loads are
**remainder-masked** (`mask = offs_k < K - k·BLOCK_K`, `other = 0`) so the
trip count `numKBlocks` only needs to cover `K` (`K ≤ numKBlocks·BLOCK_K`),
and whose epilogue rescales the exact ℤ accumulator by the two masked-loaded
scale vectors `a_scale[row] · b_scale[col]` before the fp16 downcast and the
masked store into `C`. -/

set_option linter.unusedVariables false in
/-- The `matmul_kernel` surface (`SPLIT_K = 1` arm: the `tl.atomic_add` else
branch drops with the constexpr; every `* SPLIT_K` factor is folded to its
`SPLIT_K = 1` value; `pid_sp_k` stays faithful). The loop bound
`tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)` is spelled as the antiquoted binder
`numKBlocks`; the K loads carry the source's genuine remainder masks with
`other=0.0`, and `c = (accumulator.to(tl.float32) * a_scale[:, None] *
b_scale[None, :]).to(tl.float16)` is spelled through the DSL's implicit
`Op.intToReal` promotion (the explicit `.to(tl.float32)` is ident-blocked —
disclosed in the preamble). -/
def int8_matmul_quantization_matmul_surface
    (A : Region .int) (as_ptr : RegionName) (B : Region .int) (bs_ptr C : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn stride_cm stride_cn : Nat)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  pid_sp_k = tl.program_id(1)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = pid_sp_k * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = $((A : Region .int)) + (offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak))
  b_ptrs = $((B : Region .int)) + (offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
  as_ptrs = as_ptr + offs_am * $(stride_asm)
  bs_ptrs = bs_ptr + offs_bn * $(stride_bsn)
  a_scale = tl.load(as_ptrs, mask=offs_am < $(M), other=0.0)
  b_scale = tl.load(bs_ptrs, mask=offs_bn < $(N), other=0.0)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.int32)
  for k in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    accumulator += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c = (accumulator * a_scale[:, None] * b_scale[None, :]).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}

/-- The surface lowers to a supported algorithm. -/
theorem int8_matmul_quantization_matmul_surface_toAlgorithm_supported
    (A : Region .int) (as_ptr : RegionName) (B : Region .int) (bs_ptr C : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn stride_cm stride_cn : Nat)
    (BM BN BK GM numKBlocks : Nat) :
    ∃ alg, (int8_matmul_quantization_matmul_surface A as_ptr B bs_ptr C M N K
      stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn stride_cm stride_cn
      BM BN BK GM numKBlocks).toAlgorithm? = Except.ok alg := by
  simp [int8_matmul_quantization_matmul_surface, ComputeExpr.toAlgorithm?]

/-! ## The group-swizzled block coordinates

`pid` is decomposed exactly as the source decomposes it. Unlike the
`int8_dequant_matmul` twin (which inlines `group_id * GROUP_SIZE_M`), this
source **names** the product as the `first_pid_m` register — the formulas
below are nevertheless identical to the twin's; only the eval-lemma keying
changes. -/

/-- `num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)`. -/
def mmGridM (M BM : Nat) : Nat := (M + BM - 1) / BM

/-- `num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)`. -/
def mmGridN (N BN : Nat) : Nat := (N + BN - 1) / BN

/-- `num_pid_in_group = GROUP_SIZE_M * num_pid_n`. -/
def mmWidth (N BN GM : Nat) : Nat := GM * mmGridN N BN

/-- `group_id = pid // num_pid_in_group`. -/
def mmGroupId (s : BlockState) (N BN GM : Nat) : Nat :=
  s.pids 0 / mmWidth N BN GM

/-- `group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)` (with
`first_pid_m = group_id * GROUP_SIZE_M`). -/
def mmGroupSize (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  min (mmGridM M BM - mmGroupId s N BN GM * GM) GM

/-- `pid_m = first_pid_m + (pid % group_size_m)`. -/
def mmPidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  mmGroupId s N BN GM * GM + s.pids 0 % mmGroupSize s M N BM BN GM

/-- `pid_n = (pid % num_pid_in_group) // group_size_m`. -/
def mmPidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % mmWidth N BN GM / mmGroupSize s M N BM BN GM

/-! ## Element accessors

Each is the kernel's own address arithmetic over the **launch** state's
memory, parameterized by the *absolute* row / column / K coordinate, so the
wrapped (`% M` / `% N`) and unwrapped readings share one definition. -/

/-- `A[row, kg]` — a signed `.int`-channel read (int8 values are signed). -/
def mmAElem (s : BlockState) (A : Region .int)
    (stride_am stride_ak : Nat) (row kg : Nat) : ℤ :=
  s.readMemValue .int (Region.cast A) (row * stride_am + kg * stride_ak)

/-- `B[kg, col]` — the signed `.int` read of `B`. -/
def mmBElem (s : BlockState) (B : Region .int)
    (stride_bk stride_bn : Nat) (kg col : Nat) : ℤ :=
  s.readMemValue .int (Region.cast B) (kg * stride_bk + col * stride_bn)

/-- `a_scale[row]` — the per-row dequant scale lane. -/
noncomputable def mmAScaleElem (s : BlockState) (as_ptr : RegionName)
    (stride_asm : Nat) (row : Nat) : ℝ :=
  s.readMem as_ptr (row * stride_asm)

/-- `b_scale[col]` — the per-column dequant scale lane. -/
noncomputable def mmBScaleElem (s : BlockState) (bs_ptr : RegionName)
    (stride_bsn : Nat) (col : Nat) : ℝ :=
  s.readMem bs_ptr (col * stride_bsn)

/-! ## The masked accumulator

The `if kb·BK + e < K` guard is the load masks' honest content: a masked-off
K lane loads the `other = 0` value on **both** operands, so its product
contributes `0` to `tl.dot` — the guarded summand is exactly what the kernel
accumulates, with no divisibility assumption on `K`. -/

/-- One K step's ℤ contribution to output cell `(row, col)`: the exact
integer sum-product `Op.dotInt` computes over the **masked** `a` / `b` tiles
(out-of-range K lanes contribute the `other = 0` product). -/
def mmAccStep (s : BlockState) (A B : Region .int)
    (stride_am stride_ak stride_bk stride_bn K BK : Nat) (row col kb : Nat) : ℤ :=
  ∑ e : Fin BK, if kb * BK + e.val < K
    then mmAElem s A stride_am stride_ak row (kb * BK + e.val)
      * mmBElem s B stride_bk stride_bn (kb * BK + e.val) col
    else 0

/-- The full ℤ accumulator: `accumulator` after all `numKBlocks` K steps. -/
def mmAccSpec (s : BlockState) (A B : Region .int)
    (stride_am stride_ak stride_bk stride_bn K BK numKBlocks : Nat)
    (row col : Nat) : ℤ :=
  ∑ kb : Fin numKBlocks,
    mmAccStep s A B stride_am stride_ak stride_bk stride_bn K BK row col kb.val

/-- The block-indexed masked double sum, flattened to a single
`Finset.range (numKBlocks * BK)` sum of the guarded summand. -/
private theorem mmAccSpec_range (s : BlockState) (A B : Region .int)
    (stride_am stride_ak stride_bk stride_bn K BK numKBlocks : Nat)
    (row col : Nat) :
    mmAccSpec s A B stride_am stride_ak stride_bk stride_bn K BK numKBlocks row col
      = ∑ n ∈ Finset.range (numKBlocks * BK),
          (if n < K
           then mmAElem s A stride_am stride_ak row n
             * mmBElem s B stride_bk stride_bn n col
           else 0) := by
  unfold mmAccSpec
  induction numKBlocks with
  | zero => simp
  | succ m ih =>
    rw [Fin.sum_univ_castSucc]
    simp only [Fin.val_castSucc, Fin.val_last]
    rw [ih, show (m + 1) * BK = m * BK + BK from by ring, Finset.sum_range_add]
    congr 1
    unfold mmAccStep
    rw [Fin.sum_univ_eq_sum_range (fun e => if m * BK + e < K
      then mmAElem s A stride_am stride_ak row (m * BK + e)
        * mmBElem s B stride_bk stride_bn (m * BK + e) col
      else 0)]

/-- **The reindex bridge.** Under the launch fact `K ≤ numKBlocks · BK`
(the loop covers `K`), the masked block-indexed accumulator equals the plain
`Fin K` matrix-product sum: the guarded terms vanish outside `range K`
(`Finset.sum_subset`) and the guard is true inside it. At larger `K` the
double sum would undercount and the identity would be false — which is why
the headline carries exactly this hypothesis. -/
theorem mmAccSpec_eq_finK (s : BlockState) (A B : Region .int)
    (stride_am stride_ak stride_bk stride_bn K BK numKBlocks : Nat)
    (row col : Nat) (hK : K ≤ numKBlocks * BK) :
    mmAccSpec s A B stride_am stride_ak stride_bk stride_bn K BK numKBlocks row col
      = ∑ j : Fin K, mmAElem s A stride_am stride_ak row j.val
          * mmBElem s B stride_bk stride_bn j.val col := by
  have hsub : ∑ n ∈ Finset.range K,
        (if n < K
         then mmAElem s A stride_am stride_ak row n
           * mmBElem s B stride_bk stride_bn n col
         else 0)
      = ∑ n ∈ Finset.range (numKBlocks * BK),
        (if n < K
         then mmAElem s A stride_am stride_ak row n
           * mmBElem s B stride_bk stride_bn n col
         else 0) :=
    Finset.sum_subset (Finset.range_subset_range.mpr hK)
      (fun x _ hx => if_neg (by simpa [Finset.mem_range] using hx))
  rw [mmAccSpec_range, ← hsub,
    Fin.sum_univ_eq_sum_range (fun j => mmAElem s A stride_am stride_ak row j
      * mmBElem s B stride_bk stride_bn j col)]
  exact Finset.sum_congr rfl (fun x hx => if_pos (Finset.mem_range.mp hx))

/-- The `C` store address for output lane `(r, c)` — the kernel's own
const-first order `stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]`. -/
def mmCAddr (stride_cm stride_cn BM BN pm pn : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  stride_cm * (pm * BM + idx.1.val) + stride_cn * (pn * BN + idx.2.1.val)

/-! ## Compiled body decomposition

The algorithm-lowered statement lists, checked against the macro output by
`rfl`. Lowerings worth naming: `tl.cdiv` expands to `Op.div .nat` on
`(X + BX - 1)`; `min(a, b)` is an `Op.where`; the two K-loop loads are
**masked `.int`-typed `maskOther` loads** whose mask is the remapped rank-1
comparison `offs_k < K - k·BLOCK_K` (nat truncated subtraction) and whose
`other` is the broadcast `.int` zero; the scale loads are 1-D `.real`
`maskOther` loads over the **wrapped** offsets; the epilogue is one
`castFloat`-headed assignment nesting the two scale multiplies over the
`Op.intToReal`-promoted accumulator. -/

/-- The prologue's ten scalar statements: both program ids, the two `tl.cdiv`
trip counts and the group swizzle down to `(pid_m, pid_n)` — with the
source's named `first_pid_m` register. -/
def mmPreLoopScalars (M N BM BN GM : Nat) : List Stmt :=
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

/-- The prologue's ten index/tile statements: the wrapped offset vectors, the
SPLIT_K-lane `offs_k`, the two typed GEMM pointer tiles, the two 1-D scale
pointer tiles, the two masked `.real` scale loads and the zeroed `.int`
accumulator. -/
def mmPreLoopTiles (A : Region .int) (as_ptr : RegionName) (B : Region .int)
    (bs_ptr : RegionName)
    (M N stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn : Nat)
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
    Stmt.assign .ptr [BM] "as_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase as_ptr)
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BM] "offs_am")
          (Op.constNat stride_asm))),
    Stmt.assign .ptr [BN] "bs_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase bs_ptr)
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn")
          (Op.constNat stride_bsn))),
    Stmt.assign .real [BM] "a_scale"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BM] "as_ptrs"))
        (MaskOpt.maskOther
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "offs_am")
            (Op.constNat M))
          ((Op.const 0.0).broadcast [BM]))),
    Stmt.assign .real [BN] "b_scale"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN] "bs_ptrs"))
        (MaskOpt.maskOther
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn")
            (Op.constNat N))
          ((Op.const 0.0).broadcast [BN]))),
    Stmt.assign .int [BM, BN] "accumulator" (Op.full [BM, BN] (Op.constInt 0)) ]

/-- The `a`-load mask operand: `offs_k[None, :] < K - k·BLOCK_K`, remapped
over the row axis. `K - k·BLOCK_K` is nat truncated subtraction — at a fully
out-of-range block the difference is `0` and every lane is masked off. -/
def mmAMaskOp (K BM BK : Nat) : Op .bool [BM, BK] :=
  Op.remap [BM, BK] Broadcast.nil.consSame.consL.leftIndex
    (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
      (Op.sub .nat Broadcast.nil (Op.constNat K)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK))))

/-- The `b`-load mask operand: `offs_k[:, None] < K - k·BLOCK_K`, remapped
over the column axis. -/
def mmBMaskOp (K BK BN : Nat) : Op .bool [BK, BN] :=
  Op.remap [BK, BN] Broadcast.nil.consL.consSame.leftIndex
    (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
      (Op.sub .nat Broadcast.nil (Op.constNat K)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "k") (Op.constNat BK))))

/-- The K-loop body: two remainder-masked `.int` loads (`other = 0`), the
`Op.dotInt` accumulation, and the two pointer advances (`* SPLIT_K` folded
to `1`). -/
def mmLoopBody (K stride_ak stride_bk BM BN BK : Nat) : List Stmt :=
  [ Stmt.assign .int [BM, BK] "a"
      (Op.load .int (MemAccess.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        (MaskOpt.maskOther (mmAMaskOp K BM BK)
          ((Op.constInt 0).broadcast [BM, BK]))),
    Stmt.assign .int [BK, BN] "b"
      (Op.load .int (MemAccess.ptr (Op.ref .ptr [BK, BN] "b_ptrs"))
        (MaskOpt.maskOther (mmBMaskOp K BK BN)
          ((Op.constInt 0).broadcast [BK, BN]))),
    Stmt.assign .int [BM, BN] "accumulator"
      (Op.add .int (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .int [BM, BN] "accumulator")
        (Op.dotInt (batch := []) (Op.ref .int [BM, BK] "a")
          (Op.ref .int [BK, BN] "b"))),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_ak))),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "b_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_bk))) ]

/-- The compiled tail: the fp16-cast epilogue (one nested-expression
assignment over `Op.intToReal`), the fresh unwrapped `offs_cm` / `offs_cn`
vectors, the const-first `c_ptrs` tile, the two-axis rank-2 store mask, and
the masked **`.fp16`-typed** store of `c`. -/
def mmPostLoop (C : RegionName) (M N stride_cm stride_cn BM BN : Nat) : List Stmt :=
  [ Stmt.assign .fp16 [BM, BN] "c"
      (Op.castFloat FloatDType.real FloatDType.fp16
        (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
          (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.intToReal (Op.ref .int [BM, BN] "accumulator"))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "a_scale")))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BN] "b_scale")))),
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
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))),
    Stmt.store .fp16 [BM, BN] (MemAccess.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
      (Op.ref .fp16 [BM, BN] "c")
      (MaskOpt.mask (Op.ref .bool [BM, BN] "c_mask")) ]

set_option maxRecDepth 20000 in
set_option linter.unusedVariables false in
/-- **Full body split (by `rfl`).** The lowered surface is exactly
`mmPreLoopScalars ++ mmPreLoopTiles ++ [forRange "k" 0 numKBlocks 1 mmLoopBody]
++ mmPostLoop`, every statement checked against the macro output. -/
theorem mm_body_eq (A : Region .int) (as_ptr : RegionName) (B : Region .int)
    (bs_ptr C : RegionName) (M N K : Nat)
    (stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn
      stride_cm stride_cn : Nat)
    (BM BN BK GM numKBlocks : Nat) :
    (int8_matmul_quantization_matmul_surface A as_ptr B bs_ptr C M N K
        stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn
        stride_cm stride_cn BM BN BK GM numKBlocks).toAlgKernel.body
      = mmPreLoopScalars M N BM BN GM
        ++ mmPreLoopTiles A as_ptr B bs_ptr M N stride_am stride_ak stride_asm
            stride_bk stride_bn stride_bsn BM BN BK
        ++ [Stmt.forRange "k" 0 numKBlocks 1
              (mmLoopBody K stride_ak stride_bk BM BN BK)]
        ++ mmPostLoop C M N stride_cm stride_cn BM BN := by
  rfl

/-! ## Offset, pointer and value tiles -/

/-- `pid_* * BLOCK + tl.arange(0, BLOCK)` (and `tl.arange` alone at base 0). -/
def mmOffs (base BD : Nat) : Tile .nat [BD] := ⟨fun idx => base + idx.1.val⟩

/-- The **wrapped** offset vector `(base + e) % Mm` — `offs_am` / `offs_bn`. -/
def mmWrapOffs (base BD Mm : Nat) : Tile .nat [BD] :=
  ⟨fun idx => (base + idx.1.val) % Mm⟩

/-- `a_ptrs` lane `(r, e)` at K step `k`: the wrapped row offset plus `k`
advances of `BK * stride_ak`. -/
def mmAAddr (M stride_am stride_ak BM BK pm k : Nat)
    (idx : TileIndex [BM, BK]) : Nat :=
  (pm * BM + idx.1.val) % M * stride_am + idx.2.1.val * stride_ak
    + k * (BK * stride_ak)

/-- `b_ptrs` lane `(e, c)` at K step `k`. -/
def mmBAddr (N stride_bk stride_bn BN BK pn k : Nat)
    (idx : TileIndex [BK, BN]) : Nat :=
  idx.1.val * stride_bk + (pn * BN + idx.2.1.val) % N * stride_bn
    + k * (BK * stride_bk)

/-- The `a_ptrs` pointer tile at K step `k`. -/
noncomputable def mmAPtrs (A : Region .int)
    (M stride_am stride_ak BM BK pm k : Nat) : Tile .ptr [BM, BK] :=
  ⟨fun idx => (Region.cast A, mmAAddr M stride_am stride_ak BM BK pm k idx)⟩

/-- The `b_ptrs` pointer tile at K step `k`. -/
noncomputable def mmBPtrs (B : Region .int)
    (N stride_bk stride_bn BN BK pn k : Nat) : Tile .ptr [BK, BN] :=
  ⟨fun idx => (Region.cast B, mmBAddr N stride_bk stride_bn BN BK pn k idx)⟩

/-- One `a_ptrs += BLOCK_K * stride_ak` advance. -/
theorem mmAPtrs_succ (A : Region .int) (M stride_am stride_ak BM BK pm k : Nat) :
    Tile.ptrAdd Broadcast.scalarR (mmAPtrs A M stride_am stride_ak BM BK pm k)
        (Tile.scalar (BK * stride_ak))
      = mmAPtrs A M stride_am stride_ak BM BK pm (k + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, mmAPtrs, mmAAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

/-- One `b_ptrs += BLOCK_K * stride_bk` advance. -/
theorem mmBPtrs_succ (B : Region .int) (N stride_bk stride_bn BN BK pn k : Nat) :
    Tile.ptrAdd Broadcast.scalarR (mmBPtrs B N stride_bk stride_bn BN BK pn k)
        (Tile.scalar (BK * stride_bk))
      = mmBPtrs B N stride_bk stride_bn BN BK pn (k + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, mmBPtrs, mmBAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

/-- The `a_ptrs` address agrees with `mmAElem`'s at the wrapped row and the
absolute K coordinate `k·BK + e`. -/
theorem mmAAddr_eq (M stride_am stride_ak BM BK pm k : Nat)
    (idx : TileIndex [BM, BK]) :
    mmAAddr M stride_am stride_ak BM BK pm k idx
      = (pm * BM + idx.1.val) % M * stride_am
        + (k * BK + idx.2.1.val) * stride_ak := by
  simp only [mmAAddr]
  ring

/-- The `b_ptrs` address agrees with `mmBElem`'s at the wrapped column. -/
theorem mmBAddr_eq (N stride_bk stride_bn BN BK pn k : Nat)
    (idx : TileIndex [BK, BN]) :
    mmBAddr N stride_bk stride_bn BN BK pn k idx
      = (k * BK + idx.1.val) * stride_bk
        + (pn * BN + idx.2.1.val) % N * stride_bn := by
  simp only [mmBAddr]
  ring

/-- The loaded `a` tile at K step `c` — the **honestly masked** load: an
in-range K lane reads the `.int` channel at the wrapped row, a masked-off
lane takes the `other = 0` value. -/
def mmATile (s : BlockState) (A : Region .int)
    (M K stride_am stride_ak BM BK pm c : Nat) : Tile .int [BM, BK] :=
  ⟨fun idx => if c * BK + idx.2.1.val < K
    then mmAElem s A stride_am stride_ak ((pm * BM + idx.1.val) % M)
      (c * BK + idx.2.1.val)
    else 0⟩

/-- The loaded `b` tile at K step `c` (masked, wrapped column). -/
def mmBTile (s : BlockState) (B : Region .int)
    (N K stride_bk stride_bn BN BK pn c : Nat) : Tile .int [BK, BN] :=
  ⟨fun idx => if c * BK + idx.1.val < K
    then mmBElem s B stride_bk stride_bn (c * BK + idx.1.val)
      ((pn * BN + idx.2.1.val) % N)
    else 0⟩

theorem mmATile_data (s : BlockState) (A : Region .int)
    (M K stride_am stride_ak BM BK pm c : Nat) (idx : TileIndex [BM, BK]) :
    (mmATile s A M K stride_am stride_ak BM BK pm c).data idx
      = if c * BK + idx.2.1.val < K
        then mmAElem s A stride_am stride_ak ((pm * BM + idx.1.val) % M)
          (c * BK + idx.2.1.val)
        else 0 := rfl

theorem mmBTile_data (s : BlockState) (B : Region .int)
    (N K stride_bk stride_bn BN BK pn c : Nat) (idx : TileIndex [BK, BN]) :
    (mmBTile s B N K stride_bk stride_bn BN BK pn c).data idx
      = if c * BK + idx.1.val < K
        then mmBElem s B stride_bk stride_bn (c * BK + idx.1.val)
          ((pn * BN + idx.2.1.val) % N)
        else 0 := rfl

/-- `a_scale`: the masked 1-D `.real` load over the wrapped row — a masked
lane takes `other = 0`; the `if` lives **inside** the `some` (the `.real`
channel decodes present values). -/
noncomputable def mmAScaleTile (s : BlockState) (as_ptr : RegionName)
    (M stride_asm BM pm : Nat) : Tile .real [BM] :=
  ⟨fun idx => some (if (pm * BM + idx.1.val) % M < M
    then mmAScaleElem s as_ptr stride_asm ((pm * BM + idx.1.val) % M)
    else 0)⟩

/-- `b_scale`: the masked 1-D `.real` load over the wrapped column. -/
noncomputable def mmBScaleTile (s : BlockState) (bs_ptr : RegionName)
    (N stride_bsn BN pn : Nat) : Tile .real [BN] :=
  ⟨fun idx => some (if (pn * BN + idx.1.val) % N < N
    then mmBScaleElem s bs_ptr stride_bsn ((pn * BN + idx.1.val) % N)
    else 0)⟩

/-- The `as_ptrs` pointer tile (wrapped row × `stride_asm`). -/
noncomputable def mmASPtrs (as_ptr : RegionName)
    (M stride_asm BM pm : Nat) : Tile .ptr [BM] :=
  ⟨fun idx => (as_ptr, (pm * BM + idx.1.val) % M * stride_asm)⟩

/-- The `bs_ptrs` pointer tile (wrapped column × `stride_bsn`). -/
noncomputable def mmBSPtrs (bs_ptr : RegionName)
    (N stride_bsn BN pn : Nat) : Tile .ptr [BN] :=
  ⟨fun idx => (bs_ptr, (pn * BN + idx.1.val) % N * stride_bsn)⟩

/-- `accumulator` after `i` K steps: the masked exact-ℤ partial sum the
invariant carries, at the **wrapped** row / column of each lane. -/
def mmAccTile (s : BlockState) (A B : Region .int)
    (M N K stride_am stride_ak stride_bk stride_bn BM BN BK pm pn : Nat)
    (i : Nat) : Tile .int [BM, BN] :=
  ⟨fun idx => ∑ j : Fin i,
    mmAccStep s A B stride_am stride_ak stride_bk stride_bn K BK
      ((pm * BM + idx.1.val) % M) ((pn * BN + idx.2.1.val) % N) j.val⟩

/-- At `i = 0` the accumulator is the `.int` zero tile `tl.zeros` produces. -/
theorem mmAccTile_zero (s : BlockState) (A B : Region .int)
    (M N K stride_am stride_ak stride_bk stride_bn BM BN BK pm pn : Nat) :
    mmAccTile s A B M N K stride_am stride_ak stride_bk stride_bn BM BN BK pm pn 0
      = (⟨fun _ => 0⟩ : Tile .int [BM, BN]) := by
  apply Tile.ext
  intro idx
  simp [mmAccTile]

/-- Per-lane masking distributes over the product: a masked-off K lane
contributes the `other = 0` product on both operands. -/
private theorem mm_ite_mul_ite (p : Prop) [Decidable p] (a b : ℤ) :
    (if p then a else 0) * (if p then b else 0) = if p then a * b else 0 := by
  by_cases h : p <;> simp [h]

/-- **The `Op.dotInt` accumulator step.** `accumulator += tl.dot(a, b)`
extends the masked exact-ℤ partial sum by one `mmAccStep` — the two loads'
per-lane guards coincide, so the guarded product is exactly the step
summand. -/
theorem mmAccTile_dotInt_succ (s : BlockState) (A B : Region .int)
    (M N K stride_am stride_ak stride_bk stride_bn BM BN BK pm pn i : Nat) :
    Tile.bop NumericDType.int.add
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (mmAccTile s A B M N K stride_am stride_ak stride_bk stride_bn BM BN BK
          pm pn i)
        (Tile.dotInt [] (mmATile s A M K stride_am stride_ak BM BK pm i)
          (mmBTile s B N K stride_bk stride_bn BN BK pn i))
      = mmAccTile s A B M N K stride_am stride_ak stride_bk stride_bn BM BN BK
          pm pn (i + 1) := by
  apply Tile.ext
  intro idx
  obtain ⟨r, c, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, mmAccTile]
  -- `erw`: `Tile.dotInt`'s operand shapes are `[] ++ [M, K]`, so
  -- `Tile.dotInt_nil_data` does not fire under `rw` / `simp only`.
  erw [Tile.dotInt_nil_data]
  simp only [mmATile_data, mmBTile_data, NumericDType.int_add, mm_ite_mul_ite]
  rw [Fin.sum_univ_castSucc]
  simp [mmAccStep]

/-! ## Per-statement eval recipes

Private copies, since bench ports never import each other. New to this port
(beyond the `int8_dequant_matmul` twin's set): the **general partial-mask
`maskOther` pointer-load recipe** `mm_load_ptr_maskOther` (the
`matmul_triton_autotune` precedent proves its masks all-true under exact
divisibility; here the masks are genuinely partial, so the per-lane `if` is
kept), the `Op.remap` recipe, and the two broadcast-`other` recipes. -/

/-- A masked-with-`other` `.ptr` load, fully general: a lane whose mask is
true reads its own `(region, offset)` cell; a masked-off lane takes the
`other` value. -/
private theorem mm_load_ptr_maskOther {dtype : TileDType} {sh : TileShape}
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

/-- `Op.remap` (the rank-lift the loop masks compile through). -/
private theorem mm_remap_eval {dtype : TileDType} {sh : TileShape}
    (outShape : TileShape) (map : TileIndex outShape → TileIndex sh)
    (x : Op dtype sh) (t : BlockState) (v : Tile dtype sh)
    (hv : evalOp x t = some v) :
    evalOp (Op.remap outShape map x) t = some (Tile.remap map v) := by
  simp only [evalOp, hv]
  rfl

/-- The broadcast `.real` `other = 0.0` operand. -/
private theorem mm_broadcastReal_eval (sh : TileShape) (t : BlockState) :
    evalOp ((Op.const (0.0 : ℝ)).broadcast sh) t
      = some (⟨fun _ => some (0.0 : ℝ)⟩ : Tile .real sh) := by
  simp only [evalOp, evalOp_const]
  rfl

/-- The broadcast `.int` `other = 0` operand. -/
private theorem mm_broadcastInt_eval (sh : TileShape) (t : BlockState) :
    evalOp ((Op.constInt (0 : Int)).broadcast sh) t
      = some (⟨fun _ => (0 : ℤ)⟩ : Tile .int sh) := by
  simp only [evalOp]
  rfl

/-- Pointer advance / offset. -/
private theorem mm_ptrAdd_eval {a b : TileShape} {out : TileShape}
    (bc : Broadcast a b out)
    (pnm : RegName) (t : BlockState) (pt : Tile .ptr a) (off : Op .nat b)
    (ov : Tile .nat b)
    (hp : t.regs .ptr a pnm = some pt) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ref .ptr a pnm) off) t
      = some (Tile.ptrAdd bc pt ov) := by
  simp only [evalOp, evalOp_ref, hp, ho]
  rfl

/-- `%` on the `nat` channel. -/
private theorem mm_mod_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mod IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.mod IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `//` on the `nat` channel. -/
private theorem mm_floorDiv_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.floorDiv IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.floorDiv IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-- `Op.div` on any numeric channel (what `tl.cdiv` expands to). -/
private theorem mm_divTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.div h bc x y) t = some (Tile.bop h.div bc vx vy) := by
  rw [evalOp_div, hx, hy]
  rfl

/-- `<`. -/
private theorem mm_ltTile_eval {dtype : TileDType} (h : ComparableDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.lt h bc x y) t = some (Tile.cop h.lt bc vx vy) := by
  rw [evalOp_lt, hx, hy]
  rfl

/-- `tl.where` — how `min(a, b)` is lowered, there being no `Op.min`. -/
private theorem mm_where_eval {dtype : TileDType} {sh : TileShape}
    (c : Op .bool sh) (x y : Op dtype sh) (t : BlockState)
    (vc : Tile .bool sh) (vx vy : Tile dtype sh)
    (hc : evalOp c t = some vc) (hx : evalOp x t = some vx)
    (hy : evalOp y t = some vy) :
    evalOp (Op.where c x y) t = some (Tile.select vc vx vy) := by
  rw [evalOp_where, hc, hx, hy]
  rfl

/-- `tl.zeros`. -/
private theorem mm_full_eval {dtype : TileDType} (sh : TileShape) (e : Op dtype [])
    (t : BlockState) (v : Tile dtype []) (hv : evalOp e t = some v) :
    evalOp (Op.full sh e) t
      = some (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype sh) := by
  rw [evalOp_full, hv]
  rfl

/-- The `.int` zero literal. -/
private theorem mm_constInt_eval (n : Int) (t : BlockState) :
    evalOp (Op.constInt n) t = some (Tile.scalar n) := by
  simp [evalOp]

/-- A pointer tile built from a bare region base. -/
private theorem mm_ptrAddBase_eval {d : TileDType} {b out : TileShape}
    (bc : Broadcast [] b out) (rg : Region d) (t : BlockState)
    (off : Op .nat b) (ov : Tile .nat b) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ptrBase rg) off) t
      = some (Tile.ptrAdd bc (Tile.scalar ((Region.cast rg : RegionName), 0)) ov) := by
  simp only [evalOp, ho]
  rfl

/-- `*` on two tiles, both operand values known. -/
private theorem mm_mulTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mul h bc x y) t = some (Tile.bop h.mul bc vx vy) := by
  rw [evalOp_mul, hx, hy]
  rfl

/-- `+` on two tiles, both operand values known. -/
private theorem mm_addTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.add h bc x y) t = some (Tile.bop h.add bc vx vy) := by
  rw [evalOp_add, hx, hy]
  rfl

/-- `-` on two tiles, both operand values known. -/
private theorem mm_subTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.sub h bc x y) t = some (Tile.bop h.sub bc vx vy) := by
  rw [evalOp_sub, hx, hy]
  rfl

/-- `[:, None]` / `[None, :]`. -/
private theorem mm_expandDim_eval {dtype : TileDType} {sh : TileShape}
    (ax : Fin (sh.length + 1)) (x : Op dtype sh) (t : BlockState)
    (v : Tile dtype sh) (hv : evalOp x t = some v) :
    evalOp (Op.expandDim ax x) t = some (Tile.expandDim ax v) := by
  rw [evalOp_expandDim, hv]
  rfl

/-- `Op.intToReal` — the signed ℝ embedding carrying the source's
`accumulator.to(tl.float32)` (the epilogue's single ℤ→ℝ promotion). -/
private theorem mm_intToReal_eval {sh : TileShape} (x : Op .int sh)
    (t : BlockState) (vx : Tile .int sh) (hx : evalOp x t = some vx) :
    evalOp (Op.intToReal x) t = some (Tile.intToReal vx) := by
  simp only [evalOp, hx]
  rfl

/-- `(…).to(tl.float16)` — the fp16 downcast, with a **general** `.real`
operand (the epilogue's cast heads a nested multiply, not a register ref). -/
private theorem mm_castFp16_eval {sh : TileShape} (x : Op .real sh)
    (t : BlockState) (v : Tile .real sh) (hx : evalOp x t = some v) :
    evalOp (Op.castFloat FloatDType.real FloatDType.fp16 x) t
      = some (⟨fun ix => FloatDType.real.cast FloatDType.fp16 (v.data ix)⟩
          : Tile .fp16 sh) := by
  rw [evalOp_castFloat]
  simp [hx]

/-- `tl.dot` on the `.int` channel (`Op.dotInt`) at rank 2. `erw`, not `rw`:
the operand shapes are `[] ++ [M, K]`, which does not unfold at reducible
transparency. -/
private theorem mm_dotInt_eval {M K N : Nat} (x : Op .int [M, K])
    (y : Op .int [K, N]) (t : BlockState)
    (vx : Tile .int [M, K]) (vy : Tile .int [K, N])
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.dotInt (batch := []) x y) t = some (Tile.dotInt [] vx vy) := by
  erw [evalOp_dotInt, hx, hy]
  rfl

/-- `&` on the bool channel. -/
private theorem mm_boolAnd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .bool a) (y : Op .bool b) (t : BlockState)
    (vx : Tile .bool a) (vy : Tile .bool b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.boolAnd bc x y) t
      = some (Tile.bop (fun u v : Bool => u && v) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

/-! ### `nat` scalar shapes -/

private theorem mm_mulScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mul .nat Broadcast.nil x y) t = some (Tile.scalar (u * v)) := by
  rw [evalOp_mul, hx, hy]
  rfl

private theorem mm_addScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.add .nat Broadcast.nil x y) t = some (Tile.scalar (u + v)) := by
  rw [evalOp_add, hx, hy]
  rfl

private theorem mm_subScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.sub .nat Broadcast.nil x y) t = some (Tile.scalar (u - v)) := by
  rw [evalOp_sub, hx, hy]
  rfl

private theorem mm_divScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.div .nat Broadcast.nil x y) t = some (Tile.scalar (u / v)) := by
  rw [mm_divTile_eval NumericDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem mm_modScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u % v)) := by
  rw [mm_mod_eval Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem mm_floorDivScalar_eval (x y : Op .nat []) (t : BlockState)
    (u v : Nat) (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u / v)) := by
  rw [mm_floorDiv_eval Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem mm_ltScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.lt ComparableDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (decide (u < v))) := by
  rw [mm_ltTile_eval ComparableDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem mm_whereScalarNat_eval (c : Op .bool []) (x y : Op .nat [])
    (t : BlockState) (cv : Bool) (u v : Nat)
    (hc : evalOp c t = some (Tile.scalar cv))
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.where c x y) t = some (Tile.scalar (if cv then u else v)) := by
  rw [mm_where_eval c x y t _ _ _ hc hx hy]
  rfl

/-- `name * c` on a `nat` scalar register. -/
private theorem mm_mulRef_eval (t : BlockState) (nm : RegName) (val c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c)) t
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `min` is spelled as a `tl.where` by the DSL. -/
private theorem mm_min_as_where (u v : Nat) :
    (if u < v then u else v) = min u v := by
  rcases Nat.lt_or_ge u v with h | h
  · rw [if_pos h]; omega
  · rw [if_neg (by omega)]; omega

/-- `setReg` leaves memory alone, at **function** level (a deep tower's
`t.mem = s.mem` by one `rfl` overruns `whnf`). -/
private theorem mm_setReg_mem {dtype : TileDType} {sh : TileShape}
    (s : BlockState) (nm : RegName) (v : Tile dtype sh) :
    (s.setReg nm dtype sh v).mem = s.mem := rfl

/-! ### The loop masks and the typed loads, bridged to the named tiles -/

/-- The `a`-load mask evaluates to the **honest lane tile**
`decide (e < K - k·BK)` — genuinely partial at a ragged `K` (nat truncated
subtraction; a fully out-of-range block masks every lane off). -/
private theorem mm_aMask_eval (t : BlockState) (K BM BK c : Nat)
    (hk : t.regs .nat [BK] "offs_k" = some (mmOffs 0 BK))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar c)) :
    evalOp (mmAMaskOp K BM BK) t
      = some (⟨fun idx => decide (idx.2.1.val < K - c * BK)⟩
          : Tile .bool [BM, BK]) := by
  unfold mmAMaskOp
  rw [mm_remap_eval _ _ _ t _
    (mm_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ _
      (mm_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hk))
      (mm_subScalarNat_eval _ _ t K (c * BK) (evalOp_constNat _ _)
        (mm_mulRef_eval t "k" c BK hkk)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp [Tile.remap, Tile.cop_data, Tile.expandDim_data, mmOffs,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]

/-- The `b`-load mask lane tile (`decide (e < K - k·BK)` over the row axis). -/
private theorem mm_bMask_eval (t : BlockState) (K BK BN c : Nat)
    (hk : t.regs .nat [BK] "offs_k" = some (mmOffs 0 BK))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar c)) :
    evalOp (mmBMaskOp K BK BN) t
      = some (⟨fun idx => decide (idx.1.val < K - c * BK)⟩
          : Tile .bool [BK, BN]) := by
  unfold mmBMaskOp
  rw [mm_remap_eval _ _ _ t _
    (mm_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ _
      (mm_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hk))
      (mm_subScalarNat_eval _ _ t K (c * BK) (evalOp_constNat _ _)
        (mm_mulRef_eval t "k" c BK hkk)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨e, cc, u⟩ := idx
  simp [Tile.remap, Tile.cop_data, Tile.expandDim_data, mmOffs,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]

/-- The masked `.int` `a` load lands on `mmATile`, on the launch state's
memory: the mask lane `e < K - k·BK` is normalized to the tile's guard
`k·BK + e < K` (omega on nat truncated subtraction). -/
private theorem mm_aLoad_eq (s0 : BlockState) (A : Region .int) (t : BlockState)
    (M K stride_am stride_ak BM BK pm c : Nat)
    (hmem : t.mem = s0.mem)
    (hap : t.regs .ptr [BM, BK] "a_ptrs"
      = some (mmAPtrs A M stride_am stride_ak BM BK pm c))
    (hk : t.regs .nat [BK] "offs_k" = some (mmOffs 0 BK))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar c)) :
    evalOp (Op.load .int (MemAccess.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        (MaskOpt.maskOther (mmAMaskOp K BM BK)
          ((Op.constInt 0).broadcast [BM, BK]))) t
      = some (mmATile s0 A M K stride_am stride_ak BM BK pm c) := by
  rw [mm_load_ptr_maskOther "a_ptrs" (mmAMaskOp K BM BK) _ t _ _ _ hap
    (mm_aMask_eval t K BM BK c hk hkk) (mm_broadcastInt_eval [BM, BK] t)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [mmATile, mmAPtrs, decide_eq_true_eq]
  have hiff : ((e.val : Nat) < K - c * BK) ↔ (c * BK + e.val < K) := by omega
  have hval : t.readMemValue .int (Region.cast A)
      (mmAAddr M stride_am stride_ak BM BK pm c (r, e, u))
      = mmAElem s0 A stride_am stride_ak ((pm * BM + r.val) % M)
          (c * BK + e.val) := by
    rw [mmAAddr_eq]
    simp only [mmAElem, BlockState.readMemValue, BlockState.readMemTyped, hmem]
  rw [if_congr hiff hval rfl]

/-- The masked `.int` `b` load lands on `mmBTile`. -/
private theorem mm_bLoad_eq (s0 : BlockState) (B : Region .int) (t : BlockState)
    (N K stride_bk stride_bn BN BK pn c : Nat)
    (hmem : t.mem = s0.mem)
    (hbp : t.regs .ptr [BK, BN] "b_ptrs"
      = some (mmBPtrs B N stride_bk stride_bn BN BK pn c))
    (hk : t.regs .nat [BK] "offs_k" = some (mmOffs 0 BK))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar c)) :
    evalOp (Op.load .int (MemAccess.ptr (Op.ref .ptr [BK, BN] "b_ptrs"))
        (MaskOpt.maskOther (mmBMaskOp K BK BN)
          ((Op.constInt 0).broadcast [BK, BN]))) t
      = some (mmBTile s0 B N K stride_bk stride_bn BN BK pn c) := by
  rw [mm_load_ptr_maskOther "b_ptrs" (mmBMaskOp K BK BN) _ t _ _ _ hbp
    (mm_bMask_eval t K BK BN c hk hkk) (mm_broadcastInt_eval [BK, BN] t)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨e, cc, u⟩ := idx
  simp only [mmBTile, mmBPtrs, decide_eq_true_eq]
  have hiff : ((e.val : Nat) < K - c * BK) ↔ (c * BK + e.val < K) := by omega
  have hval : t.readMemValue .int (Region.cast B)
      (mmBAddr N stride_bk stride_bn BN BK pn c (e, cc, u))
      = mmBElem s0 B stride_bk stride_bn (c * BK + e.val)
          ((pn * BN + cc.val) % N) := by
    rw [mmBAddr_eq]
    simp only [mmBElem, BlockState.readMemValue, BlockState.readMemTyped, hmem]
  rw [if_congr hiff hval rfl]

/-- The masked 1-D `a_scale` load lands on `mmAScaleTile`: a true-mask lane
decodes the present `.real` cell (`readMemValue_real`), a masked-off lane
takes `other = 0` — the `if` commutes through the `some`. -/
private theorem mm_aScale_eval (s0 : BlockState) (as_ptr : RegionName)
    (t : BlockState) (M stride_asm BM pm : Nat)
    (hmem : t.mem = s0.mem)
    (hasp : t.regs .ptr [BM] "as_ptrs" = some (mmASPtrs as_ptr M stride_asm BM pm))
    (ham : t.regs .nat [BM] "offs_am" = some (mmWrapOffs (pm * BM) BM M)) :
    evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BM] "as_ptrs"))
        (MaskOpt.maskOther
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "offs_am")
            (Op.constNat M))
          ((Op.const 0.0).broadcast [BM]))) t
      = some (mmAScaleTile s0 as_ptr M stride_asm BM pm) := by
  rw [mm_load_ptr_maskOther "as_ptrs" _ _ t _ _ _ hasp
    (mm_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar M)
      (by rw [evalOp_ref]; exact ham) (evalOp_constNat _ _))
    (mm_broadcastReal_eval [BM] t)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, u⟩ := idx
  simp only [mmASPtrs, mmAScaleTile, mmWrapOffs, Tile.cop_data,
    Broadcast.leftIndex, ComparableDType.lt, Tile.scalar, decide_eq_true_eq]
  by_cases hg : (pm * BM + r.val) % M < M
  · rw [if_pos hg, if_pos hg]
    simp only [mmAScaleElem, BlockState.readMemValue_real]
    unfold BlockState.readMem
    rw [hmem]
  · rw [if_neg hg, if_neg hg]
    norm_num

/-- The masked 1-D `b_scale` load lands on `mmBScaleTile`. -/
private theorem mm_bScale_eval (s0 : BlockState) (bs_ptr : RegionName)
    (t : BlockState) (N stride_bsn BN pn : Nat)
    (hmem : t.mem = s0.mem)
    (hbsp : t.regs .ptr [BN] "bs_ptrs" = some (mmBSPtrs bs_ptr N stride_bsn BN pn))
    (hbn : t.regs .nat [BN] "offs_bn" = some (mmWrapOffs (pn * BN) BN N)) :
    evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN] "bs_ptrs"))
        (MaskOpt.maskOther
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn")
            (Op.constNat N))
          ((Op.const 0.0).broadcast [BN]))) t
      = some (mmBScaleTile s0 bs_ptr N stride_bsn BN pn) := by
  rw [mm_load_ptr_maskOther "bs_ptrs" _ _ t _ _ _ hbsp
    (mm_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ (Tile.scalar N)
      (by rw [evalOp_ref]; exact hbn) (evalOp_constNat _ _))
    (mm_broadcastReal_eval [BN] t)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, u⟩ := idx
  simp only [mmBSPtrs, mmBScaleTile, mmWrapOffs, Tile.cop_data,
    Broadcast.leftIndex, ComparableDType.lt, Tile.scalar, decide_eq_true_eq]
  by_cases hg : (pn * BN + r.val) % N < N
  · rw [if_pos hg, if_pos hg]
    simp only [mmBScaleElem, BlockState.readMemValue_real]
    unfold BlockState.readMem
    rw [hmem]
  · rw [if_neg hg, if_neg hg]
    norm_num

/-! ### The pid swizzle, statement by statement -/

private theorem mm_cdiv_eval (t : BlockState) (X BX : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat X) (Op.constNat BX)) (Op.constNat 1))
        (Op.constNat BX)) t
      = some (Tile.scalar ((X + BX - 1) / BX)) :=
  mm_divScalarNat_eval _ _ t (X + BX - 1) BX
    (mm_subScalarNat_eval _ _ t (X + BX) 1
      (mm_addScalarNat_eval _ _ t X BX (evalOp_constNat _ _) (evalOp_constNat _ _))
      (evalOp_constNat _ _))
    (evalOp_constNat _ _)

private theorem mm_width_eval (t : BlockState) (N BN GM : Nat)
    (hgn : t.regs .nat [] "num_pid_n" = some (Tile.scalar (mmGridN N BN))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.constNat GM)
        (Op.ref .nat [] "num_pid_n")) t
      = some (Tile.scalar (mmWidth N BN GM)) := by
  rw [mm_mulScalarNat_eval _ _ t GM (mmGridN N BN) (evalOp_constNat _ _)
    (by rw [evalOp_ref]; exact hgn)]
  rfl

private theorem mm_groupId_eval (s t : BlockState) (N BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (mmWidth N BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "num_pid_in_group")) t
      = some (Tile.scalar (mmGroupId s N BN GM)) := by
  rw [mm_floorDivScalar_eval _ _ t (s.pids 0) (mmWidth N BN GM)
    (by rw [evalOp_ref]; exact hpid) (by rw [evalOp_ref]; exact hwid)]
  rfl

/-- The named `first_pid_m = group_id * GROUP_SIZE_M` register (the twin
computes this product inline; this source names it). -/
private theorem mm_firstPidM_eval (s t : BlockState) (N BN GM : Nat)
    (hgid : t.regs .nat [] "group_id" = some (Tile.scalar (mmGroupId s N BN GM))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
        (Op.constNat GM)) t
      = some (Tile.scalar (mmGroupId s N BN GM * GM)) :=
  mm_mulRef_eval t "group_id" (mmGroupId s N BN GM) GM hgid

/-- `group_size_m` — the `min` spelled as `Op.where`, with both compare/sub
operands reading the **named** `first_pid_m` register. -/
private theorem mm_groupSize_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hgm : t.regs .nat [] "num_pid_m" = some (Tile.scalar (mmGridM M BM)))
    (hfpm : t.regs .nat [] "first_pid_m"
      = some (Tile.scalar (mmGroupId s N BN GM * GM))) :
    evalOp (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
            (Op.ref .nat [] "first_pid_m"))
          (Op.constNat GM))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
          (Op.ref .nat [] "first_pid_m"))
        (Op.constNat GM)) t
      = some (Tile.scalar (mmGroupSize s M N BM BN GM)) := by
  have hsub : evalOp (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
      (Op.ref .nat [] "first_pid_m")) t
      = some (Tile.scalar (mmGridM M BM - mmGroupId s N BN GM * GM)) :=
    mm_subScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hgm)
      (by rw [evalOp_ref]; exact hfpm)
  rw [mm_whereScalarNat_eval _ _ _ t
    (decide (mmGridM M BM - mmGroupId s N BN GM * GM < GM))
    (mmGridM M BM - mmGroupId s N BN GM * GM) GM
    (mm_ltScalarNat_eval _ _ t _ _ hsub (evalOp_constNat _ _)) hsub
    (evalOp_constNat _ _)]
  simp only [decide_eq_true_eq, mm_min_as_where, mmGroupSize]

/-- `pid_m = first_pid_m + (pid % group_size_m)`. -/
private theorem mm_pidM_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hfpm : t.regs .nat [] "first_pid_m"
      = some (Tile.scalar (mmGroupId s N BN GM * GM)))
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hgs : t.regs .nat [] "group_size_m"
      = some (Tile.scalar (mmGroupSize s M N BM BN GM))) :
    evalOp (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "group_size_m"))) t
      = some (Tile.scalar (mmPidM s M N BM BN GM)) := by
  rw [mm_addScalarNat_eval _ _ t (mmGroupId s N BN GM * GM)
    (s.pids 0 % mmGroupSize s M N BM BN GM)
    (by rw [evalOp_ref]; exact hfpm)
    (mm_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hgs))]
  rfl

private theorem mm_pidN_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (mmWidth N BN GM)))
    (hgs : t.regs .nat [] "group_size_m"
      = some (Tile.scalar (mmGroupSize s M N BM BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "num_pid_in_group"))
        (Op.ref .nat [] "group_size_m")) t
      = some (Tile.scalar (mmPidN s M N BM BN GM)) := by
  rw [mm_floorDivScalar_eval _ _ t (s.pids 0 % mmWidth N BN GM)
    (mmGroupSize s M N BM BN GM)
    (mm_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hwid))
    (by rw [evalOp_ref]; exact hgs)]
  rfl

/-! ### The index tiles -/

/-- `pid_* * BLOCK + tl.arange(0, BLOCK)` from a scalar register. -/
private theorem mm_offs_eval (nm : RegName) (t : BlockState) (BD base c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar base)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c))
        (Op.arange BD)) t
      = some (mmOffs (base * c) BD) := by
  rw [mm_addTile_eval NumericDType.nat Broadcast.scalarL _ _ t
    (Tile.scalar (base * c)) (Tile.vec (fun i => (i.val : Nat)))
    (mm_mulScalarNat_eval _ _ t base c (by rw [evalOp_ref]; exact hr)
      (evalOp_constNat _ _))
    (evalOp_arange _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mmOffs, Tile.vec, Broadcast.rightIndex, NumericDType.add]

/-- The fused wrapped offset statement
`offs_a* = (pid_* * BLOCK + tl.arange(0, BLOCK)) % D`. -/
private theorem mm_wrapOffs_eval (nm : RegName) (t : BlockState) (BD base Mm : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar base)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat BD))
          (Op.arange BD))
        (Op.constNat Mm)) t
      = some (mmWrapOffs (base * BD) BD Mm) := by
  rw [mm_mod_eval Broadcast.scalarR _ _ t (mmOffs (base * BD) BD) (Tile.scalar Mm)
    (mm_offs_eval nm t BD base BD hr) (evalOp_constNat _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mmWrapOffs, mmOffs, Tile.bop_data, Broadcast.leftIndex,
    Broadcast.rightIndex]

/-- `offs_k = pid_sp_k * BLOCK_K + tl.arange(0, BLOCK_K)` under the launch
fact `pid_sp_k = 0` (grid axis 1 has extent `SPLIT_K = 1`). -/
private theorem mm_offsK_eval (t : BlockState) (BK : Nat)
    (hr : t.regs .nat [] "pid_sp_k" = some (Tile.scalar 0)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_sp_k") (Op.constNat BK))
        (Op.arange BK)) t
      = some (mmOffs 0 BK) := by
  have h := mm_offs_eval "pid_sp_k" t BK 0 BK hr
  rwa [Nat.zero_mul] at h

/-- `a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)`
— at K step `0`. -/
private theorem mm_aPtrsInit_eval (A : Region .int) (t : BlockState)
    (M stride_am stride_ak BM BK pm : Nat)
    (ham : t.regs .nat [BM] "offs_am" = some (mmWrapOffs (pm * BM) BM M))
    (hrk : t.regs .nat [BK] "offs_k" = some (mmOffs 0 BK)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am"))
            (Op.constNat stride_am))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_ak)))) t
      = some (mmAPtrs A M stride_am stride_ak BM BK pm 0) := by
  rw [mm_ptrAddBase_eval _ _ t _ _
    (mm_addTile_eval NumericDType.nat _ _ _ t _ _
      (mm_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_am)
        (mm_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact ham))
        (evalOp_constNat _ _))
      (mm_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_ak)
        (mm_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrk))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mmAPtrs, mmAAddr, mmWrapOffs, mmOffs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)`
— at K step `0`. -/
private theorem mm_bPtrsInit_eval (B : Region .int) (t : BlockState)
    (N stride_bk stride_bn BN BK pn : Nat)
    (hrk : t.regs .nat [BK] "offs_k" = some (mmOffs 0 BK))
    (hbn : t.regs .nat [BN] "offs_bn" = some (mmWrapOffs (pn * BN) BN N)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
            (Op.constNat stride_bk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn"))
            (Op.constNat stride_bn)))) t
      = some (mmBPtrs B N stride_bk stride_bn BN BK pn 0) := by
  rw [mm_ptrAddBase_eval _ _ t _ _
    (mm_addTile_eval NumericDType.nat _ _ _ t _ _
      (mm_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_bk)
        (mm_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrk))
        (evalOp_constNat _ _))
      (mm_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar stride_bn)
        (mm_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hbn))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mmBPtrs, mmBAddr, mmWrapOffs, mmOffs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `as_ptrs = as_ptr + offs_am * stride_asm` — a 1-D pointer tile over the
wrapped row. -/
private theorem mm_asPtrsInit_eval (as_ptr : RegionName) (t : BlockState)
    (M stride_asm BM pm : Nat)
    (ham : t.regs .nat [BM] "offs_am" = some (mmWrapOffs (pm * BM) BM M)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase as_ptr)
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BM] "offs_am")
          (Op.constNat stride_asm))) t
      = some (mmASPtrs as_ptr M stride_asm BM pm) := by
  rw [mm_ptrAddBase_eval _ _ t _ _
    (mm_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
      (Tile.scalar stride_asm)
      (by rw [evalOp_ref]; exact ham) (evalOp_constNat _ _))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mmASPtrs, mmWrapOffs, Tile.ptrAdd_data, Tile.bop_data,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul]

/-- `bs_ptrs = bs_ptr + offs_bn * stride_bsn`. -/
private theorem mm_bsPtrsInit_eval (bs_ptr : RegionName) (t : BlockState)
    (N stride_bsn BN pn : Nat)
    (hbn : t.regs .nat [BN] "offs_bn" = some (mmWrapOffs (pn * BN) BN N)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase bs_ptr)
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn")
          (Op.constNat stride_bsn))) t
      = some (mmBSPtrs bs_ptr N stride_bsn BN pn) := by
  rw [mm_ptrAddBase_eval _ _ t _ _
    (mm_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
      (Tile.scalar stride_bsn)
      (by rw [evalOp_ref]; exact hbn) (evalOp_constNat _ _))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mmBSPtrs, mmWrapOffs, Tile.ptrAdd_data, Tile.bop_data,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul]

/-! ## The K-loop invariant

`i ≤ numKBlocks` is part of the predicate on purpose: `forRange_inv`
concludes only `stop ≤ final`, so carrying the upper bound is what pins
`final = numKBlocks`. The carried registers are those the loop advances
(`a_ptrs`, `b_ptrs`, `accumulator`), the mask input `offs_k`, and those the
epilogue still needs (`pid_m`, `pid_n`, `a_scale`, `b_scale`). -/

/-- The state carried across K steps. -/
noncomputable def mmInv (s0 : BlockState) (A B : Region .int)
    (as_ptr bs_ptr : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn
      BM BN BK GM numKBlocks : Nat)
    (i : Nat) (s : BlockState) : Prop :=
  i ≤ numKBlocks
  ∧ s.mem = s0.mem
  ∧ s.pids = s0.pids
  ∧ s.regs .nat [] "pid_m" = some (Tile.scalar (mmPidM s0 M N BM BN GM))
  ∧ s.regs .nat [] "pid_n" = some (Tile.scalar (mmPidN s0 M N BM BN GM))
  ∧ s.regs .nat [BK] "offs_k" = some (mmOffs 0 BK)
  ∧ s.regs .real [BM] "a_scale"
      = some (mmAScaleTile s0 as_ptr M stride_asm BM (mmPidM s0 M N BM BN GM))
  ∧ s.regs .real [BN] "b_scale"
      = some (mmBScaleTile s0 bs_ptr N stride_bsn BN (mmPidN s0 M N BM BN GM))
  ∧ s.regs .ptr [BM, BK] "a_ptrs"
      = some (mmAPtrs A M stride_am stride_ak BM BK (mmPidM s0 M N BM BN GM) i)
  ∧ s.regs .ptr [BK, BN] "b_ptrs"
      = some (mmBPtrs B N stride_bk stride_bn BN BK (mmPidN s0 M N BM BN GM) i)
  ∧ s.regs .int [BM, BN] "accumulator"
      = some (mmAccTile s0 A B M N K stride_am stride_ak stride_bk stride_bn
          BM BN BK (mmPidM s0 M N BM BN GM) (mmPidN s0 M N BM BN GM) i)

/-- The loop combinator writes the induction variable before each iteration,
and `mmInv` constrains no register named `"k"`. -/
theorem mmInv_setReg_k (s0 : BlockState) (A B : Region .int)
    (as_ptr bs_ptr : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn
      BM BN BK GM numKBlocks i j : Nat) (s : BlockState)
    (h : mmInv s0 A B as_ptr bs_ptr M N K stride_am stride_ak stride_asm
      stride_bk stride_bn stride_bsn BM BN BK GM numKBlocks i s) :
    mmInv s0 A B as_ptr bs_ptr M N K stride_am stride_ak stride_asm
      stride_bk stride_bn stride_bsn BM BN BK GM numKBlocks i
      (s.setReg "k" .nat [] (Tile.scalar j)) := by
  obtain ⟨hle, hmem, hpids, hpm, hpn, hk, ha, hb, hA, hB, hacc⟩ := h
  exact ⟨hle, hmem, hpids, by simpa using hpm, by simpa using hpn,
    by simpa using hk, by simpa using ha, by simpa using hb,
    by simpa using hA, by simpa using hB, by simpa using hacc⟩

/-! ### The K step -/

set_option linter.unnecessarySimpa false in
/-- One loop-body iteration, run from the combinator's `setReg "k" i` state
(the masks read the `"k"` register): the two masked loads land on the
guarded tiles, `Op.dotInt` extends the masked partial sum, the pointers
advance one block. -/
theorem mmLoopBody_run (s0 : BlockState) (A B : Region .int)
    (as_ptr bs_ptr : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn
      BM BN BK GM numKBlocks i : Nat) (s : BlockState)
    (hnext : i + 1 ≤ numKBlocks)
    (hinv : mmInv s0 A B as_ptr bs_ptr M N K stride_am stride_ak stride_asm
      stride_bk stride_bn stride_bsn BM BN BK GM numKBlocks i s) :
    ∃ s', stepStmts (mmLoopBody K stride_ak stride_bk BM BN BK)
        (s.setReg "k" .nat [] (Tile.scalar i)) = some s'
      ∧ mmInv s0 A B as_ptr bs_ptr M N K stride_am stride_ak stride_asm
          stride_bk stride_bn stride_bsn BM BN BK GM numKBlocks (i + 1) s' := by
  have hinvk := mmInv_setReg_k s0 A B as_ptr bs_ptr M N K stride_am stride_ak
    stride_asm stride_bk stride_bn stride_bsn BM BN BK GM numKBlocks i i s hinv
  obtain ⟨-, hmem, hpids, hpm, hpn, hk, ha, hb, hA, hB, hacc⟩ := hinvk
  have hkk : (s.setReg "k" .nat [] (Tile.scalar i)).regs .nat [] "k"
      = some (Tile.scalar i) := by simp
  set pm := mmPidM s0 M N BM BN GM with hpmDef
  set pn := mmPidN s0 M N BM BN GM with hpnDef
  unfold mmLoopBody
  -- 1. `a = tl.load(a_ptrs, mask=…, other=0.0)`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_aLoad_eq s0 A _ M K stride_am stride_ak BM BK pm i hmem hA hk hkk))]
  -- 2. `b = tl.load(b_ptrs, mask=…, other=0.0)`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_bLoad_eq s0 B _ N K stride_bk stride_bn BN BK pn i
      (by simp only [mm_setReg_mem]; exact hmem)
      (by simpa using hB) (by simpa using hk) (by simpa using hkk)))]
  -- 3. `accumulator += tl.dot(a, b)` — the masked `Op.dotInt` step
  have h3 : evalOp (Op.add .int
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .int [BM, BN] "accumulator")
        (Op.dotInt (batch := []) (Op.ref .int [BM, BK] "a")
          (Op.ref .int [BK, BN] "b")))
        (((s.setReg "k" .nat [] (Tile.scalar i)).setReg "a" .int [BM, BK]
            (mmATile s0 A M K stride_am stride_ak BM BK pm i)).setReg
          "b" .int [BK, BN] (mmBTile s0 B N K stride_bk stride_bn BN BK pn i))
      = some (mmAccTile s0 A B M N K stride_am stride_ak stride_bk stride_bn
          BM BN BK pm pn (i + 1)) := by
    rw [← mmAccTile_dotInt_succ]
    exact mm_addTile_eval NumericDType.int _ _ _ _ _ _
      (by rw [evalOp_ref]; simpa using hacc)
      (mm_dotInt_eval _ _ _ _ _ (by rw [evalOp_ref]; simp)
        (by rw [evalOp_ref]; simp))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h3)]
  -- 4. `a_ptrs += BLOCK_K * stride_ak`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_ak))) _
      = some (mmAPtrs A M stride_am stride_ak BM BK pm (i + 1)) from by
      rw [← mmAPtrs_succ]
      exact mm_ptrAdd_eval Broadcast.scalarR "a_ptrs" _ _ _
        (Tile.scalar (BK * stride_ak)) (by simpa using hA)
        (mm_mulScalarNat_eval _ _ _ BK stride_ak (evalOp_constNat _ _)
          (evalOp_constNat _ _))))]
  -- 5. `b_ptrs += BLOCK_K * stride_bk`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "b_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat stride_bk))) _
      = some (mmBPtrs B N stride_bk stride_bn BN BK pn (i + 1)) from by
      rw [← mmBPtrs_succ]
      exact mm_ptrAdd_eval Broadcast.scalarR "b_ptrs" _ _ _
        (Tile.scalar (BK * stride_bk)) (by simpa using hB)
        (mm_mulScalarNat_eval _ _ _ BK stride_bk (evalOp_constNat _ _)
          (evalOp_constNat _ _))))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, hnext, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [mm_setReg_mem]
    exact hmem
  · simp only [BlockState.setReg_pids]
    exact hpids
  · simpa using hpm
  · simpa using hpn
  · simpa using hk
  · simpa using ha
  · simpa using hb
  · simp [hpmDef]
  · simp [hpnDef]
  · simp [hpmDef, hpnDef]

/-! ### Collapsing the K loop -/

theorem mmLoop_collapse (s0 : BlockState) (A B : Region .int)
    (as_ptr bs_ptr : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn
      BM BN BK GM numKBlocks : Nat) (s : BlockState)
    (h0 : mmInv s0 A B as_ptr bs_ptr M N K stride_am stride_ak stride_asm
      stride_bk stride_bn stride_bsn BM BN BK GM numKBlocks 0 s) :
    ∃ sF, stepStmt (Stmt.forRange "k" 0 numKBlocks 1
          (mmLoopBody K stride_ak stride_bk BM BN BK)) s = some sF
      ∧ mmInv s0 A B as_ptr bs_ptr M N K stride_am stride_ak stride_asm
          stride_bk stride_bn stride_bsn BM BN BK GM numKBlocks numKBlocks sF := by
  obtain ⟨final, sF, hrun, hfinal, hP⟩ :=
    forRange_inv (idx := "k") (start := 0) (stop := numKBlocks) (step := 1)
      (P := fun i t => mmInv s0 A B as_ptr bs_ptr M N K stride_am stride_ak
        stride_asm stride_bk stride_bn stride_bsn BM BN BK GM numKBlocks i t)
      one_ne_zero h0
      (fun i t hi hinv => by
        obtain ⟨s', hs', hinv'⟩ :=
          mmLoopBody_run s0 A B as_ptr bs_ptr M N K stride_am stride_ak
            stride_asm stride_bk stride_bn stride_bsn BM BN BK GM numKBlocks i t
            (by omega) hinv
        exact ⟨s', hs', hinv'⟩)
  have hEq : final = numKBlocks := le_antisymm hP.1 hfinal
  subst hEq
  exact ⟨sF, hrun, hP⟩

/-! ## The prologue walks -/

/-- The ten scalar statements. Memory is untouched; the registers anything
downstream reads are the SPLIT_K program id and the block coordinates. -/
theorem mmPreLoopScalars_run (s : BlockState) (M N BM BN GM : Nat) :
    ∃ t, stepStmts (mmPreLoopScalars M N BM BN GM) s = some t
      ∧ t.mem = s.mem
      ∧ t.pids = s.pids
      ∧ t.regs .nat [] "pid_sp_k" = some (Tile.scalar (s.pids 1))
      ∧ t.regs .nat [] "pid_m" = some (Tile.scalar (mmPidM s M N BM BN GM))
      ∧ t.regs .nat [] "pid_n" = some (Tile.scalar (mmPidN s M N BM BN GM)) := by
  unfold mmPreLoopScalars
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mm_cdiv_eval _ M BM))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mm_cdiv_eval _ N BN))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_width_eval _ N BN GM (by simp [mmGridN])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_groupId_eval s _ N BN GM (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_firstPidM_eval s _ N BN GM (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_groupSize_eval s _ M N BM BN GM (by simp [mmGridM]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_pidM_eval s _ M N BM BN GM (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_pidN_eval s _ M N BM BN GM (by simp) (by simp) (by simp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_⟩ <;> simp [mm_setReg_mem]

/-- The ten index/tile statements, ending on `mmInv` at step `0`. The
`pid_sp_k = 0` hypothesis is the launch fact `s.pids 1 = 0` (grid axis 1
has extent `SPLIT_K = 1`), already rewritten by the caller. -/
theorem mmPreLoopTiles_run (s0 : BlockState) (A B : Region .int)
    (as_ptr bs_ptr : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn
      BM BN BK GM numKBlocks : Nat) (t : BlockState)
    (hmem : t.mem = s0.mem)
    (hpids : t.pids = s0.pids)
    (hpz : t.regs .nat [] "pid_sp_k" = some (Tile.scalar 0))
    (hpm : t.regs .nat [] "pid_m" = some (Tile.scalar (mmPidM s0 M N BM BN GM)))
    (hpn : t.regs .nat [] "pid_n" = some (Tile.scalar (mmPidN s0 M N BM BN GM))) :
    ∃ t', stepStmts (mmPreLoopTiles A as_ptr B bs_ptr M N stride_am stride_ak
          stride_asm stride_bk stride_bn stride_bsn BM BN BK) t = some t'
      ∧ mmInv s0 A B as_ptr bs_ptr M N K stride_am stride_ak stride_asm
          stride_bk stride_bn stride_bsn BM BN BK GM numKBlocks 0 t' := by
  unfold mmPreLoopTiles
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_wrapOffs_eval "pid_m" t BM (mmPidM s0 M N BM BN GM) M hpm))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_wrapOffs_eval "pid_n" _ BN (mmPidN s0 M N BM BN GM) N
      (by simpa using hpn)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_offsK_eval _ BK (by simpa using hpz)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_aPtrsInit_eval A _ M stride_am stride_ak BM BK (mmPidM s0 M N BM BN GM)
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_bPtrsInit_eval B _ N stride_bk stride_bn BN BK (mmPidN s0 M N BM BN GM)
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_asPtrsInit_eval as_ptr _ M stride_asm BM (mmPidM s0 M N BM BN GM)
      (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_bsPtrsInit_eval bs_ptr _ N stride_bsn BN (mmPidN s0 M N BM BN GM)
      (by simp)))]
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some TileDType.real [BM] "a_scale"
    (Op.load .real (MemAccess.ptr (Op.ref .ptr [BM] "as_ptrs"))
      (MaskOpt.maskOther
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "offs_am")
          (Op.constNat M))
        ((Op.const 0.0).broadcast [BM]))) _
    (mmAScaleTile s0 as_ptr M stride_asm BM (mmPidM s0 M N BM BN GM))
    (mm_aScale_eval s0 as_ptr _ M stride_asm BM (mmPidM s0 M N BM BN GM)
      (by simpa [mm_setReg_mem] using hmem) (by simp) (by simp)))]
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some TileDType.real [BN] "b_scale"
    (Op.load .real (MemAccess.ptr (Op.ref .ptr [BN] "bs_ptrs"))
      (MaskOpt.maskOther
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn")
          (Op.constNat N))
        ((Op.const 0.0).broadcast [BN]))) _
    (mmBScaleTile s0 bs_ptr N stride_bsn BN (mmPidN s0 M N BM BN GM))
    (mm_bScale_eval s0 bs_ptr _ N stride_bsn BN (mmPidN s0 M N BM BN GM)
      (by simpa [mm_setReg_mem] using hmem) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM, BN] (Op.constInt 0)) _
        = some (mmAccTile s0 A B M N K stride_am stride_ak stride_bk stride_bn
            BM BN BK (mmPidM s0 M N BM BN GM) (mmPidN s0 M N BM BN GM) 0) from by
      rw [mmAccTile_zero]
      exact mm_full_eval [BM, BN] (Op.constInt 0) _ _ (mm_constInt_eval 0 _)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [mm_setReg_mem]
    exact hmem
  · simpa using hpids
  · simpa using hpm
  · simpa using hpn
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp

/-! ## The epilogue and the output store

The epilogue is one `castFloat`-headed assignment: the ℤ accumulator is
promoted once through `Op.intToReal` (carrying the source's
`accumulator.to(tl.float32)`) and multiplied by the two masked-loaded scale
vectors in the kernel's own left-associated order
`(acc * a_scale[:, None]) * b_scale[None, :]`, then downcast to fp16. -/

/-- The stored value at lane `ix` (before the fp16 quantization): the masked
ℤ accumulator at the wrapped lane, times the two guarded scale lanes —
the kernel's exact parenthesization `x * a * b`. -/
noncomputable def mmCVal (s : BlockState) (A B : Region .int)
    (as_ptr bs_ptr : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn
      BM BN BK numKBlocks pm pn : Nat) (ix : TileIndex [BM, BN]) : ℝ :=
  ((mmAccSpec s A B stride_am stride_ak stride_bk stride_bn K BK numKBlocks
      ((pm * BM + ix.1.val) % M) ((pn * BN + ix.2.1.val) % N) : ℤ) : ℝ)
    * (if (pm * BM + ix.1.val) % M < M
       then mmAScaleElem s as_ptr stride_asm ((pm * BM + ix.1.val) % M) else 0)
    * (if (pn * BN + ix.2.1.val) % N < N
       then mmBScaleElem s bs_ptr stride_bsn ((pn * BN + ix.2.1.val) % N) else 0)

/-- The epilogue statement
`c = (accumulator * a_scale[:, None] * b_scale[None, :]).to(tl.float16)`
lands on the fp16 image of `mmCVal`. -/
private theorem mm_c_eval (s0 : BlockState) (A B : Region .int)
    (as_ptr bs_ptr : RegionName) (t : BlockState)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn
      BM BN BK numKBlocks pm pn : Nat)
    (ha : t.regs .real [BM] "a_scale"
      = some (mmAScaleTile s0 as_ptr M stride_asm BM pm))
    (hb : t.regs .real [BN] "b_scale"
      = some (mmBScaleTile s0 bs_ptr N stride_bsn BN pn))
    (hacc : t.regs .int [BM, BN] "accumulator"
      = some (mmAccTile s0 A B M N K stride_am stride_ak stride_bk stride_bn
          BM BN BK pm pn numKBlocks)) :
    evalOp (Op.castFloat FloatDType.real FloatDType.fp16
        (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
          (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.intToReal (Op.ref .int [BM, BN] "accumulator"))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "a_scale")))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BN] "b_scale")))) t
      = some (⟨fun ix => FloatDType.real.cast FloatDType.fp16
          (some (mmCVal s0 A B as_ptr bs_ptr M N K stride_am stride_ak
            stride_asm stride_bk stride_bn stride_bsn BM BN BK numKBlocks
            pm pn ix))⟩ : Tile .fp16 [BM, BN]) := by
  have haccP : evalOp (Op.intToReal (Op.ref .int [BM, BN] "accumulator")) t
      = some (Tile.intToReal (mmAccTile s0 A B M N K stride_am stride_ak
          stride_bk stride_bn BM BN BK pm pn numKBlocks)) :=
    mm_intToReal_eval _ t _ (by rw [evalOp_ref]; exact hacc)
  have haP : evalOp
        ((Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "a_scale")) : Op .real [BM, 1]) t
      = some (Tile.expandDim ⟨1, by simp⟩
          (mmAScaleTile s0 as_ptr M stride_asm BM pm)) :=
    mm_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact ha)
  have hbP : evalOp
        ((Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BN] "b_scale")) : Op .real [1, BN]) t
      = some (Tile.expandDim ⟨0, by simp⟩
          (mmBScaleTile s0 bs_ptr N stride_bsn BN pn)) :=
    mm_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hb)
  have hmul1 : evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.intToReal (Op.ref .int [BM, BN] "accumulator"))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "a_scale"))) t
      = some (Tile.bop NumericDType.real.mul
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Tile.intToReal (mmAccTile s0 A B M N K stride_am stride_ak
            stride_bk stride_bn BM BN BK pm pn numKBlocks))
          (Tile.expandDim ⟨1, by simp⟩
            (mmAScaleTile s0 as_ptr M stride_asm BM pm))) :=
    mm_mulTile_eval NumericDType.real _ _ _ t _ _ haccP haP
  have hmul2 : evalOp (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.intToReal (Op.ref .int [BM, BN] "accumulator"))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "a_scale")))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BN] "b_scale"))) t
      = some (Tile.bop NumericDType.real.mul
          (Broadcast.consR (Broadcast.consSame Broadcast.nil))
          (Tile.bop NumericDType.real.mul
            (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Tile.intToReal (mmAccTile s0 A B M N K stride_am stride_ak
              stride_bk stride_bn BM BN BK pm pn numKBlocks))
            (Tile.expandDim ⟨1, by simp⟩
              (mmAScaleTile s0 as_ptr M stride_asm BM pm)))
          (Tile.expandDim ⟨0, by simp⟩
            (mmBScaleTile s0 bs_ptr N stride_bsn BN pn))) :=
    mm_mulTile_eval NumericDType.real _ _ _ t _ _ hmul1 hbP
  -- `Eq.trans`, not `rw`: the dependent expandDim result shapes make the
  -- kabstract pattern fail to key even on a syntactically identical target.
  refine Eq.trans (mm_castFp16_eval _ t _ hmul2) (congrArg some ?_)
  apply Tile.ext
  intro idx
  obtain ⟨r, c, u⟩ := idx
  simp [mmCVal, mmAccTile, mmAScaleTile, mmBScaleTile, mmAccSpec,
    Tile.bop_data, Tile.intToReal_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, WithBot.realMul]

/-- The `c_ptrs` pointer tile. -/
noncomputable def mmCPtrs (C : RegionName)
    (stride_cm stride_cn BM BN pm pn : Nat) : Tile .ptr [BM, BN] :=
  ⟨fun idx => (C, mmCAddr stride_cm stride_cn BM BN pm pn idx)⟩

/-- `c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)` — over the
**unwrapped** output coordinates. -/
def mmCMask (M N BM BN pm pn : Nat) : Tile .bool [BM, BN] :=
  ⟨fun idx => decide (pm * BM + idx.1.val < M) && decide (pn * BN + idx.2.1.val < N)⟩

/-- The post-store state: one masked **`.fp16`-typed** scatter over the
`[BM, BN]` output tile. -/
noncomputable def mmStoreState (C : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat)
    (f : TileIndex [BM, BN] → TileCarrier .fp16) (t : BlockState) : BlockState :=
  (TileShape.allIndices [BM, BN]).foldl
    (fun acc i => if pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N then
        acc.writeMemTyped .fp16 C (mmCAddr stride_cm stride_cn BM BN pm pn i) (f i)
      else acc) t

/-- `c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]`
— strides on the **left** of the products (the kernel's own order). -/
private theorem mm_cPtrsInit_eval (C : RegionName) (t : BlockState)
    (stride_cm stride_cn BM BN pm pn : Nat)
    (hcm : t.regs .nat [BM] "offs_cm" = some (mmOffs (pm * BM) BM))
    (hcn : t.regs .nat [BN] "offs_cn" = some (mmOffs (pn * BN) BN)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_cm)
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat stride_cn)
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) t
      = some (mmCPtrs C stride_cm stride_cn BM BN pm pn) := by
  rw [mm_ptrAddBase_eval _ _ t _ _
    (mm_addTile_eval NumericDType.nat _ _ _ t _ _
      (mm_mulTile_eval NumericDType.nat Broadcast.scalarL _ _ t
        (Tile.scalar stride_cm) _
        (evalOp_constNat _ _)
        (mm_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcm)))
      (mm_mulTile_eval NumericDType.nat Broadcast.scalarL _ _ t
        (Tile.scalar stride_cn) _
        (evalOp_constNat _ _)
        (mm_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcn))))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mmCPtrs, mmCAddr, mmOffs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `c_mask` — the comparisons happen at rank 2 (the unit axes are inserted
**inside** the `lt` operands, unlike the twin). -/
private theorem mm_cMask_eval (t : BlockState) (M N BM BN pm pn : Nat)
    (hcm : t.regs .nat [BM] "offs_cm" = some (mmOffs (pm * BM) BM))
    (hcn : t.regs .nat [BN] "offs_cn" = some (mmOffs (pn * BN) BN)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm"))
          (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))
          (Op.constNat N))) t
      = some (mmCMask M N BM BN pm pn) := by
  rw [mm_boolAnd_eval (Broadcast.consR (Broadcast.consL Broadcast.nil)) _ _ t _ _
    (mm_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _
      (Tile.scalar M)
      (mm_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcm))
      (evalOp_constNat _ _))
    (mm_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _
      (Tile.scalar N)
      (mm_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hcn))
      (evalOp_constNat _ _))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [mmCMask, mmOffs, Tile.bop_data, Tile.cop_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]

/-- The masked `.fp16` store of the **`c` register**. -/
private theorem mm_store_eq (C : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat) (t : BlockState)
    (vt : Tile .fp16 [BM, BN]) (f : TileIndex [BM, BN] → TileCarrier .fp16)
    (hfv : ∀ i, vt.data i = f i)
    (hcp : t.regs .ptr [BM, BN] "c_ptrs"
      = some (mmCPtrs C stride_cm stride_cn BM BN pm pn))
    (hcmask : t.regs .bool [BM, BN] "c_mask" = some (mmCMask M N BM BN pm pn))
    (hv : t.regs .fp16 [BM, BN] "c" = some vt) :
    stepStmt (Stmt.store .fp16 [BM, BN]
        (MemAccess.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
        (Op.ref .fp16 [BM, BN] "c")
        (MaskOpt.mask (Op.ref .bool [BM, BN] "c_mask"))) t
      = some (mmStoreState C M N stride_cm stride_cn BM BN pm pn f t) := by
  unfold stepStmt mmStoreState
  simp only [evalOp_ref, hv, hcp, hcmask, Option.map_some]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [BM, BN])) ?_)
  funext acc i
  obtain ⟨r, cc, u⟩ := i
  by_cases hb : pm * BM + r.val < M ∧ pn * BN + cc.val < N
  · simp only [mmCMask, if_pos hb, mmCPtrs, hfv]
    simp [hb.1, hb.2]
  · simp only [mmCMask, if_neg hb, mmCPtrs]
    rw [if_neg]
    intro hcon
    exact hb (by simpa using hcon)

/-- `MemCell`-level readback of the masked `.fp16` scatter on every active
lane (`scatter_memcell_fp16_prop_masked_nd`). -/
private theorem mm_store_props (C : RegionName)
    (M N stride_cm stride_cn BM BN pm pn : Nat) (t : BlockState)
    (f : TileIndex [BM, BN] → TileCarrier .fp16)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => mmCAddr stride_cm stride_cn BM BN pm pn i)) :
    ∀ i : TileIndex [BM, BN],
      (pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N) →
      (mmStoreState C M N stride_cm stride_cn BM BN pm pn f t).mem C
          (mmCAddr stride_cm stride_cn BM BN pm pn i)
        = MemCell.of .fp16
            (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (f i))) := by
  classical
  intro i hi
  unfold mmStoreState
  have h := scatter_memcell_fp16_prop_masked_nd (region := C) t
    (fun j : TileIndex [BM, BN] => mmCAddr stride_cm stride_cn BM BN pm pn j) f
    (fun j : TileIndex [BM, BN] => pm * BM + j.1.val < M ∧ pn * BN + j.2.1.val < N)
    hInj i
  rw [h, if_pos hi]

/-- `hInj` for a row-major `C`, in the kernel's const-first address form:
distinct output lanes get distinct addresses as soon as the column stride is
positive and one block row fits inside the row stride. -/
theorem mmCAddr_injective (stride_cm stride_cn BM BN pm pn : Nat)
    (hcn : 0 < stride_cn) (hfit : BN * stride_cn ≤ stride_cm) :
    Function.Injective
      (fun i : TileIndex [BM, BN] => mmCAddr stride_cm stride_cn BM BN pm pn i) := by
  intro i j hij
  obtain ⟨r₁, c₁, u₁⟩ := i
  obtain ⟨r₂, c₂, u₂⟩ := j
  simp only [mmCAddr] at hij
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

Six statements: the fp16-cast epilogue, the fresh unwrapped `offs_cm` /
`offs_cn`, the const-first `c_ptrs` tile, the two-axis mask, and the masked
`.fp16` store of `c`. On every lane the mask lets through, `row < M` and
`col < N` turn the wrapped lane into the plain coordinates
(`Nat.mod_eq_of_lt`), force the scale guards true (`row < M` gives `M > 0`),
and `mmAccSpec_eq_finK` (under `hK`) turns the masked block sum into the
plain `Fin K` matrix product. -/

theorem mmPostLoop_run (s0 : BlockState) (A B : Region .int)
    (as_ptr bs_ptr C : RegionName)
    (M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn
      stride_cm stride_cn BM BN BK GM numKBlocks : Nat) (t : BlockState)
    (hK : K ≤ numKBlocks * BK)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => mmCAddr stride_cm stride_cn BM BN
        (mmPidM s0 M N BM BN GM) (mmPidN s0 M N BM BN GM) i))
    (hinv : mmInv s0 A B as_ptr bs_ptr M N K stride_am stride_ak stride_asm
      stride_bk stride_bn stride_bsn BM BN BK GM numKBlocks numKBlocks t) :
    ∃ sF, stepStmts (mmPostLoop C M N stride_cm stride_cn BM BN) t = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (mmPidM s0 M N BM BN GM * BM + idx.1.val < M
            ∧ mmPidN s0 M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem C (mmCAddr stride_cm stride_cn BM BN (mmPidM s0 M N BM BN GM)
              (mmPidN s0 M N BM BN GM) idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some ((((∑ j : Fin K, mmAElem s0 A stride_am stride_ak
                          (mmPidM s0 M N BM BN GM * BM + idx.1.val) j.val
                          * mmBElem s0 B stride_bk stride_bn j.val
                            (mmPidN s0 M N BM BN GM * BN + idx.2.1.val)) : ℤ) : ℝ)
                  * mmAScaleElem s0 as_ptr stride_asm
                      (mmPidM s0 M N BM BN GM * BM + idx.1.val)
                  * mmBScaleElem s0 bs_ptr stride_bsn
                      (mmPidN s0 M N BM BN GM * BN + idx.2.1.val)))) := by
  obtain ⟨-, hmem, -, hpm, hpn, -, ha, hb, -, -, hacc⟩ := hinv
  set pm := mmPidM s0 M N BM BN GM with hpmDef
  set pn := mmPidN s0 M N BM BN GM with hpnDef
  unfold mmPostLoop
  -- 1. the epilogue: `c = (accumulator * a_scale[:, None] * b_scale[None, :]).to(tl.float16)`
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some TileDType.fp16 [BM, BN] "c"
    (Op.castFloat FloatDType.real FloatDType.fp16
      (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.intToReal (Op.ref .int [BM, BN] "accumulator"))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "a_scale")))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BN] "b_scale")))) _
    (⟨fun ix => FloatDType.real.cast FloatDType.fp16
        (some (mmCVal s0 A B as_ptr bs_ptr M N K stride_am stride_ak stride_asm
          stride_bk stride_bn stride_bsn BM BN BK numKBlocks pm pn ix))⟩
      : Tile .fp16 [BM, BN])
    (mm_c_eval s0 A B as_ptr bs_ptr t M N K stride_am stride_ak stride_asm
      stride_bk stride_bn stride_bsn BM BN BK numKBlocks pm pn ha hb hacc))]
  -- 2-3. the fresh unwrapped `offs_cm` / `offs_cn`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_offs_eval "pid_m" _ BM pm BM (by simpa using hpm)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_offs_eval "pid_n" _ BN pn BN (by simpa using hpn)))]
  -- 4-5. the `c_ptrs` tile and the two-axis mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_cPtrsInit_eval C _ stride_cm stride_cn BM BN pm pn (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_cMask_eval _ M N BM BN pm pn (by simp) (by simp)))]
  -- 6. the `.fp16` store of `c`
  rw [stepStmts.cons_some
    (mm_store_eq C M N stride_cm stride_cn BM BN pm pn _
      (⟨fun ix => FloatDType.real.cast FloatDType.fp16
          (some (mmCVal s0 A B as_ptr bs_ptr M N K stride_am stride_ak stride_asm
            stride_bk stride_bn stride_bsn BM BN BK numKBlocks pm pn ix))⟩
        : Tile .fp16 [BM, BN])
      (fun ix => FloatDType.real.cast FloatDType.fp16
        (some (mmCVal s0 A B as_ptr bs_ptr M N K stride_am stride_ak stride_asm
          stride_bk stride_bn stride_bsn BM BN BK numKBlocks pm pn ix)))
      (fun _ => rfl)
      (by simp) (by simp) (by simp))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx hidx
  rw [mm_store_props C M N stride_cm stride_cn BM BN pm pn _ _ hInj idx hidx]
  have hM : (pm * BM + idx.1.val) % M = pm * BM + idx.1.val :=
    Nat.mod_eq_of_lt hidx.1
  have hN : (pn * BN + idx.2.1.val) % N = pn * BN + idx.2.1.val :=
    Nat.mod_eq_of_lt hidx.2
  simp only [mmCVal, hM, hN]
  rw [if_pos hidx.1, if_pos hidx.2,
    mmAccSpec_eq_finK s0 A B stride_am stride_ak stride_bk stride_bn K BK
      numKBlocks _ _ hK]
  simp only [FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue,
    FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot, FloatDType.fp16_toWithBot,
    WithBot.unbotD_some]

/-! ## Main theorem -/

set_option maxHeartbeats 1000000 in
set_option linter.unusedVariables false in
/-- **Genuine, dimension-general correctness** (`SPLIT_K = 1` arm — the
autotune table's `reset_to_zero=['c_ptr']` host convention zeroes `C`, and
the grid's second axis has extent `SPLIT_K`, so `hpid1 : s.pids 1 = 0` is
the launch fact and the `tl.atomic_add` else-arm drops with the constexpr).
For every launch state, the kernel runs to completion and every in-range
output lane of `C` holds the `.fp16` memory cell carrying

`fp16( (Σ_{j<K} A[row,j]·B[j,col] : ℤ) · a_scale[row] · b_scale[col] )`

— the exact ℤ int8 matmul (`Op.dotInt`), rescaled in the kernel's own
left-associated order, and downcast once (the placeholder `FloatDType.cast`
is the identity, so the carried value is the product itself). The source's
`accumulator.to(tl.float32)` is carried by the DSL's implicit `Op.intToReal`
promotion (the explicit spelling is ident-blocked; disclosed in the
preamble).

The K-loop loads are **remainder-masked with `other = 0`**, and the guarded
spec terms are exactly those masks' content: a masked-off K lane contributes
the `0·0` product, so extra all-masked trailing blocks add `0` and the only
launch fact forced is `hK : K ≤ numKBlocks · BLOCK_SIZE_K` (the host's
`numKBlocks = tl.cdiv(K, BLOCK_SIZE_K)` satisfies it with no divisibility
assumption — at larger `K` the loop would undercount the sum and this
statement would be false). `hInj` says distinct output lanes get distinct
`C` addresses — `mmCAddr_injective` discharges it for a row-major `C`. The
`% M` / `% N` offset wraps disappear on exactly the lanes the store mask
lets through (`Nat.mod_eq_of_lt`), which also forces the scale-load masks
true, so the stored value reads the plain `A`/`B` rows and scale lanes. -/
specification int8_matmul_quantization_matmul_exec_genuine
    (A : Region .int) (as_ptr : RegionName) (B : Region .int) (bs_ptr C : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn
      stride_cm stride_cn : Nat)
    (BM BN BK GM numKBlocks : Nat) (s : BlockState)
    (hK : K ≤ numKBlocks * BK)
    (hpid1 : s.pids 1 = 0)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => mmCAddr stride_cm stride_cn BM BN
        (mmPidM s M N BM BN GM) (mmPidN s M N BM BN GM) i)) :
    ∃ sF, exec (int8_matmul_quantization_matmul_surface A as_ptr B bs_ptr C
        M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn
        stride_cm stride_cn BM BN BK GM numKBlocks).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (mmPidM s M N BM BN GM * BM + idx.1.val < M
            ∧ mmPidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem C (mmCAddr stride_cm stride_cn BM BN (mmPidM s M N BM BN GM)
              (mmPidN s M N BM BN GM) idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some ((((∑ j : Fin K, mmAElem s A stride_am stride_ak
                          (mmPidM s M N BM BN GM * BM + idx.1.val) j.val
                          * mmBElem s B stride_bk stride_bn j.val
                            (mmPidN s M N BM BN GM * BN + idx.2.1.val)) : ℤ) : ℝ)
                  * mmAScaleElem s as_ptr stride_asm
                      (mmPidM s M N BM BN GM * BM + idx.1.val)
                  * mmBScaleElem s bs_ptr stride_bsn
                      (mmPidN s M N BM BN GM * BN + idx.2.1.val)))) := by
  rw [exec, mm_body_eq]
  -- prologue: the scalars, then the index tiles
  obtain ⟨t1, hrun1, h1mem, h1pids, h1pz, h1pm, h1pn⟩ :=
    mmPreLoopScalars_run s M N BM BN GM
  obtain ⟨t2, hrun2, h2inv⟩ :=
    mmPreLoopTiles_run s A B as_ptr bs_ptr M N K stride_am stride_ak stride_asm
      stride_bk stride_bn stride_bsn BM BN BK GM numKBlocks t1 h1mem h1pids
      (hpid1 ▸ h1pz) h1pm h1pn
  simp only [List.append_assoc]
  rw [stepStmts.append_some hrun1, stepStmts.append_some hrun2]
  -- the collapsed K loop
  obtain ⟨t3, hrun3, h3inv⟩ :=
    mmLoop_collapse s A B as_ptr bs_ptr M N K stride_am stride_ak stride_asm
      stride_bk stride_bn stride_bsn BM BN BK GM numKBlocks t2 h2inv
  rw [show [Stmt.forRange "k" 0 numKBlocks 1
          (mmLoopBody K stride_ak stride_bk BM BN BK)]
        ++ mmPostLoop C M N stride_cm stride_cn BM BN
      = Stmt.forRange "k" 0 numKBlocks 1
          (mmLoopBody K stride_ak stride_bk BM BN BK)
        :: mmPostLoop C M N stride_cm stride_cn BM BN
      from rfl]
  rw [stepStmts.cons_some hrun3]
  -- the tail
  obtain ⟨sF, hpost, hout⟩ :=
    mmPostLoop_run s A B as_ptr bs_ptr C M N K stride_am stride_ak stride_asm
      stride_bk stride_bn stride_bsn stride_cm stride_cn BM BN BK GM numKBlocks
      t3 hK hInj h3inv
  exact ⟨sF, hpost, hout⟩

end MatmulKernel

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization
