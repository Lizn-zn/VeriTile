import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `triton_matmul` — closed-form L2-grouped GEMM correctness (fp16 + fp8 epilogues)

`triton_matmul.py`'s `matmul_kernel` is the Triton-tutorial L2-grouped tiled
GEMM `C = A × B`: a single linear `pid` is split into an `(pid_m, pid_n)` tile
coordinate via the L2-grouping schedule, the A-row / B-column indices are
**clamped** to `0` outside the matrix (`tl.where(offs < M, offs, 0)`, unlike
the `% M` wrap of the `matmul_triton_autotune` twin), a
`BLOCK_SIZE_M × BLOCK_SIZE_N` output tile is accumulated by the fused
`accumulator = tl.dot(a, b, accumulator)` reduction over the K dimension (with
the per-block `offs_k < K - k·BLOCK_K` load masks), the accumulator is downcast
by the constexpr output-dtype epilogue — `float8e4nv` when the output buffer is
fp8, `float16` otherwise — and the tile is stored to `C` under the
`(row<M) & (col<N)` boundary mask.

This file proves the **full K-loop + epilogue** correct against a genuine
mathematical matrix product, once per epilogue arm: every active output cell
`C[i,j]` of the computed tile equals `od( Σ_{k < K} A[i,k] · B[k,j] )` over `ℝ`,
where `K = BLOCK_SIZE_K · numKBlocks` and `od` is the arm's output grid
(`fp16` / `f8e4`). This is NOT the kernel's own emitted value — the real-valued
`Σ_k A·B` GEMM reference is derived independently of the kernel from the loaded
`A`/`B` tiles.

## Proof architecture

```
triton_matmul_f16_closed_form_correct \                  ← EXACT TOP THEOREMS (ComputeCorrect.Realizes_without_Rounding,
triton_matmul_f8_closed_form_correct  /                    one per epilogue arm; both instantiate the od-generic layer)
  └─ matmul_exec_closed_form              ← od-generic exec closed form (every active cell)
       ├─ preLoop        (P 0: accumulator = 0, pointers seeded, schedule + clamps derived)
       ├─ matmul_step    (one K-block: masked dot advances the partial sum)
       ├─ matmul_postLoop (od cast + masked store = closed form)
       └─ forRange_inv   (loop-invariant principle, drives the K-loop)

triton_matmul_f16_io_correctness \                       ← STREAMING `⊨[R]` HEADLINES (StreamMasked2DKernelIO₂,
triton_matmul_f8_io_correctness  /                         outDType = .fp16 / .f8e4; both instantiate the od-generic layer)
  ├─ matmul_flattenOk_body / matmul_traceSafeR_body   (flat bridge + looping safety walk, od-generic)
  └─ matmul_runR = preLoop + matmul_step (reused verbatim; prologue/loop are cast-free)
       └─ matmul_postLoopR  (R store tail: R.cast + masked writeMemAsR, one boundary round)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the dot reduction is
accumulated over `ℝ` and the modeled `.to(tl.float16)` / `.to(tl.float8e4nv)`
casts are the placeholders `FloatDType.real.cast .fp16` / `.f8e4`. The
contracted dimension is presented as `K = BLOCK_SIZE_K · numKBlocks` so the loop
trip count `cdiv(K, BLOCK_SIZE_K) = numKBlocks` is exact and the per-block load
masks `offs_k < K - k·BLOCK_K` are uniformly satisfied (the kernel's
tail-masking is genuinely modeled but vacuous at exact-multiple K).
`num_warps`/`num_stages` and the host's per-dtype config table are NOT modeled:
the surface is a single symbolic config, the trusted boundary. The host launch
(grid, linear-pid scheduling) is trusted; the per-program statement is
universally quantified over `s`, covering every program of the grid. The
L2-grouping index math (`pid → (pid_m, pid_n)`) and the `tl.where` index clamps
are transcribed exactly as the kernel computes them and the spec's layout
references the same derived `offs_am`/`offs_bn`/`offs_cm`/`offs_cn`, so they are
not separate proof obligations. `tl.max_contiguous`/`tl.multiple_of` are
compiler scheduling hints; the DSL erases them to their value argument.
Output-offset injectivity (distinct lanes hit distinct addresses) is the only
assumed disjointness.

## Translation-surface blocker

Translation-surface blocker: (a) the kernel's constexpr epilogue branch
`if (c_ptr.dtype.element_ty == tl.float8e4nv):` — a compile-time dispatch on
the OUTPUT buffer's element dtype — is split into two Lean surfaces
(`triton_matmul_f16_surface` with `c = (accumulator).to(tl.float16)` and
`triton_matmul_f8_surface` with `c = (accumulator).to(tl.float8e4nv)`), the
`matmul_tma` twin-surface precedent, so the branch statement itself does not
appear in either surface; (b) the loop trip count `tl.cdiv(K, BLOCK_SIZE_K)` is
supplied as the antiquoted `numKBlocks` binder (so that `tl.cdiv` call does not
appear as a surface statement; the `K = BLOCK_SIZE_K · numKBlocks` presentation,
template precedent); (c) the K-loop counter is spelled `kk` where Python spells
it `k` (avoiding a clash with the antiquoted dimension binder `K`). The textual
py↔lean scans in `bench/audit_tritonbench_g.sh` exempt this port on this marker
(registered in `proof_blockers.md`).
-/

namespace VeriTile.Bench.TritonBenchG.TritonMatmul

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! ## Surfaces -/

/-- Faithful transcription of `triton_matmul.py`'s `matmul_kernel`, **fp16
epilogue arm** (`c_ptr.dtype.element_ty ≠ tl.float8e4nv`).

The contracted dimension is presented as `K = BLOCK_SIZE_K · numKBlocks` so the
loop bound `tl.cdiv(K, BLOCK_SIZE_K) = numKBlocks` is exact; it is supplied as
the antiquoted `numKBlocks`. All other surface structure — the L2-grouping
schedule, the `tl.where` index clamps, the `tl.max_contiguous`/`tl.multiple_of`
hints (DSL-erased to their value argument), the per-block
`offs_k < K - kk·BLOCK_K` load masks, the fused `tl.dot(a, b, accumulator)`,
the `float16` cast, and the `(row<M)&(col<N)`-masked store — is transcribed
verbatim. -/
def triton_matmul_f16_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  start_m = pid_m * $(BLOCK_SIZE_M)
  start_n = pid_n * $(BLOCK_SIZE_N)
  offs_am = start_m + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = start_n + tl.arange(0, $(BLOCK_SIZE_N))
  offs_am = tl.where(offs_am < $(M), offs_am, $(0))
  offs_bn = tl.where(offs_bn < $(N), offs_bn, $(0))
  offs_am = tl.max_contiguous(tl.multiple_of(offs_am, $(BLOCK_SIZE_M)), $(BLOCK_SIZE_M))
  offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, $(BLOCK_SIZE_N)), $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for kk in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}

/-- Faithful transcription of `triton_matmul.py`'s `matmul_kernel`, **fp8
epilogue arm** (`c_ptr.dtype.element_ty == tl.float8e4nv`): identical to
`triton_matmul_f16_surface` except the epilogue cast
`c = (accumulator).to(tl.float8e4nv)`. -/
def triton_matmul_f8_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  start_m = pid_m * $(BLOCK_SIZE_M)
  start_n = pid_n * $(BLOCK_SIZE_N)
  offs_am = start_m + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = start_n + tl.arange(0, $(BLOCK_SIZE_N))
  offs_am = tl.where(offs_am < $(M), offs_am, $(0))
  offs_bn = tl.where(offs_bn < $(N), offs_bn, $(0))
  offs_am = tl.max_contiguous(tl.multiple_of(offs_am, $(BLOCK_SIZE_M)), $(BLOCK_SIZE_M))
  offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, $(BLOCK_SIZE_N)), $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for kk in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c = (accumulator).to(tl.float8e4nv)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}

/-- The fp16-arm surface lowers to the algorithm layer. -/
theorem triton_matmul_f16_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ∃ alg, (triton_matmul_f16_surface A B C M N K stride_am stride_ak
      stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N
      BLOCK_SIZE_K GROUP_SIZE_M numKBlocks).toAlgorithm? = Except.ok alg := by
  simp [triton_matmul_f16_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- The fp8-arm surface lowers to the algorithm layer. -/
theorem triton_matmul_f8_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ∃ alg, (triton_matmul_f8_surface A B C M N K stride_am stride_ak
      stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N
      BLOCK_SIZE_K GROUP_SIZE_M numKBlocks).toAlgorithm? = Except.ok alg := by
  simp [triton_matmul_f8_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## exec-stepping helpers -/

@[simp] theorem evalOp_expandDim_zero_nat {D : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] name)) s =
      (s.regs .nat [D] name).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] => v.data (i.2.1, PUnit.unit) } : Tile .nat [1, D])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

@[simp] theorem evalOp_expandDim_one_nat {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] name)) s =
      (s.regs .nat [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .nat [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

theorem evalOp_remap {dtype shape} (outShape : TileShape)
    (map : TileIndex outShape → TileIndex shape) (a : Op dtype shape) (s : BlockState) :
    evalOp (.remap outShape map a) s = (do
      let v ← evalOp a s; some (Tile.remap map v)) := by simp [evalOp]

/-- `Op.broadcast` of a scalar operand evaluates to the constant tile (the
`tl.where` clamp's `other = 0` operand). -/
@[simp] theorem evalOp_broadcast_scalar {dtype : TileDType} (e : Op dtype [])
    (shape : TileShape) (s : BlockState) :
    evalOp (e.broadcast shape) s = (do
      let v ← evalOp e s
      some (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype shape)) := by
  simp [evalOp]

/-- A `.ptr` load with `mask=…, other=…` whose mask is **uniformly true** reduces
to the clean `readMem` load. (The kernel's `offs_k < K - k·BLOCK_K` mask is
uniformly true at exact-multiple `K`.) -/
theorem load_ptr_maskOther_true_real {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (otherOp : Op .real shape)
    (s : BlockState) (ptrs : Tile .ptr shape) (masks : Tile .bool shape)
    (hp : evalOp ptrOp s = some ptrs) (hm : evalOp maskOp s = some masks)
    (others : Tile .real shape) (ho : evalOp otherOp s = some others)
    (hmask : ∀ i, masks.data i = Bool.true) :
    evalOp (.load .real (.ptr ptrOp) (.maskOther maskOp otherOp)) s
      = some ⟨fun i => some (s.readMem (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, hp, hm, ho, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [hmask i, if_true, BlockState.readMemValue_real]

/-- `a_ptrs` eval: cell `(i,e) = (A, offs_am i · sam + offs_k e · sak)`. -/
theorem aptrs_eval (s : BlockState) (A : RegionName) (M K sam sak : Nat) (gm : Fin M → Nat)
    (hm : s.regs .nat [M] "offs_am" = some (Tile.vec gm))
    (hk : s.regs .nat [K] "offs_k" = some (Tile.vec (fun e : Fin K => e.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_am")) (Op.constNat sam))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [K] "offs_k")) (Op.constNat sak)))) s
      = some (⟨fun idx : TileIndex [M, K] => (A.cast, gm idx.1 * sam + idx.2.1.val * sak)⟩ : Tile .ptr [M, K]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hm, hk, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `b_ptrs` eval: cell `(e,j) = (B, offs_k e · sbk + offs_bn j · sbn)`. -/
theorem bptrs_eval (s : BlockState) (B : RegionName) (K N sbk sbn : Nat) (gn : Fin N → Nat)
    (hk : s.regs .nat [K] "offs_k" = some (Tile.vec (fun e : Fin K => e.val)))
    (hn : s.regs .nat [N] "offs_bn" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [K] "offs_k")) (Op.constNat sbk))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_bn")) (Op.constNat sbn)))) s
      = some (⟨fun idx : TileIndex [K, N] => (B.cast, idx.1.val * sbk + gn idx.2.1 * sbn)⟩ : Tile .ptr [K, N]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hk, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `c_ptrs` eval: cell `(i,j) = (C, scm · offs_cm i + scn · offs_cn j)`
(strides on the **left** of the products). -/
theorem cptrs_eval (s : BlockState) (C : RegionName) (M N scm scn : Nat) (gm : Fin M → Nat) (gn : Fin N → Nat)
    (hm : s.regs .nat [M] "offs_cm" = some (Tile.vec gm))
    (hn : s.regs .nat [N] "offs_cn" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarL (Op.constNat scm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_cm")))
        (Op.mul .nat Broadcast.scalarL (Op.constNat scn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_cn"))))) s
      = some (⟨fun idx : TileIndex [M, N] => (C.cast, scm * gm idx.1 + scn * gn idx.2.1)⟩ : Tile .ptr [M, N]) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hm, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `accumulator` init eval: `tl.zeros` → the all-`0` tile. -/
theorem acc_init_eval (s : BlockState) (M N : Nat) :
    evalOp (Op.full [M, N] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [M, N] => some (0 : ℝ)⟩ : Tile .real [M, N]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

/-- `a_ptrs += BLOCK_K · sak` eval. -/
theorem aptr_adv_eval (s : BlockState) (M K BK sak : Nat) (ap : Tile .ptr [M, K])
    (ha : s.regs .ptr [M, K] "a_ptrs" = some ap) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [M, K] "a_ptrs")
      (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sak))) s
      = some (Tile.ptrAdd Broadcast.scalarR ap (Tile.scalar (BK * sak))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, ha, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- `b_ptrs += BLOCK_K · sbk` eval. -/
theorem bptr_adv_eval (s : BlockState) (K N BK sbk : Nat) (bp : Tile .ptr [K, N])
    (hb : s.regs .ptr [K, N] "b_ptrs" = some bp) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [K, N] "b_ptrs")
      (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sbk))) s
      = some (Tile.ptrAdd Broadcast.scalarR bp (Tile.scalar (BK * sbk))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, hb, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- **`accumulator = tl.dot(a, b, accumulator)` statement eval** (fused form
`dot a b + accumulator`). -/
theorem accdot_op_eval (M K N : Nat) (st : BlockState)
    (zt : Tile .real [M, N]) (xt : Tile .real [M, K]) (yt : Tile .real [K, N])
    (hz : st.regs .real [M, N] "accumulator" = some zt)
    (hx : st.regs .real [M, K] "a" = some xt)
    (hy : st.regs .real [K, N] "b" = some yt) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.dot (batch := []) (Op.ref .real [M, K] "a") (Op.ref .real [K, N] "b"))
        (Op.ref .real [M, N] "accumulator")) st
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Tile.dot [] xt yt) zt) := by
  have hd : evalOp (Op.dot (batch := []) (Op.ref .real [M, K] "a")
        (Op.ref .real [K, N] "b")) st = some (Tile.dot [] xt yt) := by
    rw [evalOp_dot]; simp [hx, hy]
  rw [evalOp_add]
  simp only [evalOp_ref, hz, bind, Option.bind_some]
  erw [hd]
  rfl

/-- `dot + accumulator` lane `(i,j)`: `some (dv + zv)`. -/
theorem dotadd_eval (M N : Nat) (dt zt : Tile .real [M, N]) (i : Fin M) (j : Fin N) (dv zv : ℝ)
    (hd : dt.data (i, j, PUnit.unit) = some dv) (hz : zt.data (i, j, PUnit.unit) = some zv) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) dt zt).data
        (i, j, PUnit.unit) = some (dv + zv) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hd, hz, NumericDType.add,
    WithBot.realAdd, Option.map₂, Option.bind, Option.map]

/-! ## Scheduling + GEMM closed-form spec -/

/-- `min` as the kernel's `tl.where(a < b, a, b)` spells it. -/
def kernelMin (a b : Nat) : Nat := if a < b then a else b

/-- The kernel's L2-grouping derivation of `pid_m` from the linear `pid` —
**this** kernel's spelling `first_pid_m + (pid % group_size_m)` (the twin
`matmul_triton_autotune` reduces `pid % num_pid_in_group` first). -/
def pidM (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  first_pid_m + (pid % group_size_m)

/-- The kernel's L2-grouping derivation of `pid_n` from the linear `pid`. -/
def pidN (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  (pid % num_pid_in_group) / group_size_m

/-- Global output row of tile lane `i`: `pid_m · BLOCK_M + i`, **before** the
`tl.where` clamp (the kernel's `start_m + arange`, and its `offs_cm`). -/
def rowGlobal (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  pidM (s.pids 0) M N BM BN GM * BM + i.val

/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`, before clamp. -/
def colGlobal (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  pidN (s.pids 0) M N BM BN GM * BN + j.val

/-- The kernel's `tl.where(v < bound, v, 0)` index clamp (this kernel's
out-of-range guard, where the `matmul_triton_autotune` twin wraps with `%`). -/
def clampIdx (v bound : Nat) : Nat := if v < bound then v else 0

/-- The clamped A-row index of tile lane `i` (the kernel's `offs_am` after the
`tl.where(offs_am < M, offs_am, 0)` reassignment). -/
def rowIndex (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  clampIdx (rowGlobal s M N BM BN GM i) M

/-- The clamped B-column index of tile lane `j` (the kernel's `offs_bn` after
the `tl.where(offs_bn < N, offs_bn, 0)` reassignment). -/
def colIndex (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  clampIdx (colGlobal s M N BM BN GM j) N

/-- `A[i, k] = readMem A (offs_am i · stride_am + k · stride_ak)`. -/
noncomputable def aElem (s : BlockState) (A : RegionName) (M N BM BN GM sam sak : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex s M N BM BN GM i * sam + k * sak)

/-- `B[k, j] = readMem B (k · stride_bk + offs_bn j · stride_bn)`. -/
noncomputable def bElem (s : BlockState) (B : RegionName) (M N BM BN GM sbk sbn : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (k * sbk + colIndex s M N BM BN GM j * sbn)

/-- **Genuine matmul spec** (over ℝ):
`C[i,j] = Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j]`. -/
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks : Nat)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  gemmSum (aElem s A M N BM BN GM sam sak i) (bElem s B M N BM BN GM sbk sbn j)
    (BLOCK_K * numKBlocks)

/-- Partial GEMM accumulator after `c` K-blocks: `Σ_{k < c·BLOCK_K} A·B`. -/
noncomputable def accPartial (s : BlockState) (A B : RegionName)
    (M N BM BN GM sam sak sbk sbn BLOCK_K : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) : ℝ :=
  gemmSum (aElem s A M N BM BN GM sam sak i) (bElem s B M N BM BN GM sbk sbn j) (c * BLOCK_K)

/-- One-block step of the partial accumulator (the shared `gemmSum_blockSucc`). -/
theorem accPartial_succ (s : BlockState) (A B : RegionName)
    (M N BM BN GM sam sak sbk sbn BLOCK_K : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) :
    accPartial s A B M N BM BN GM sam sak sbk sbn BLOCK_K i j (c + 1)
      = accPartial s A B M N BM BN GM sam sak sbk sbn BLOCK_K i j c
        + (Finset.univ.sum fun e : Fin BLOCK_K =>
            aElem s A M N BM BN GM sam sak i (c * BLOCK_K + e.val)
              * bElem s B M N BM BN GM sbk sbn j (c * BLOCK_K + e.val)) :=
  gemmSum_blockSucc (aElem s A M N BM BN GM sam sak i) (bElem s B M N BM BN GM sbk sbn j) BLOCK_K c

/-! ## Body decomposition -/

/-- The masked-load `mask=` operand for the `a` load (`offs_k[None,:] < K - kk·BK`,
remapped over the row axis). -/
def aMaskOp (K BM BK : Nat) : Op .bool [BM, BK] :=
  Op.remap [BM, BK] Broadcast.nil.consSame.consL.leftIndex
    (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k"))
      (Op.sub .nat Broadcast.nil (Op.constNat K)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "kk") (Op.constNat BK))))

/-- The masked-load `mask=` operand for the `b` load (`offs_k[:,None] < K - kk·BK`,
remapped over the column axis). -/
def bMaskOp (K BK BN : Nat) : Op .bool [BK, BN] :=
  Op.remap [BK, BN] Broadcast.nil.consL.consSame.leftIndex
    (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k"))
      (Op.sub .nat Broadcast.nil (Op.constNat K)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "kk") (Op.constNat BK))))

/-- The 5-statement K-loop body, transcribed (masked loads + fused dot + advances). -/
def matmulLoopBody (K BM BN BK sak sbk : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BK] "a"
      (Op.load .real (.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        (.maskOther (aMaskOp K BM BK) ((Op.const 0.0).broadcast [BM, BK]))),
    Stmt.assign .real [BK, BN] "b"
      (Op.load .real (.ptr (Op.ref .ptr [BK, BN] "b_ptrs"))
        (.maskOther (bMaskOp K BK BN) ((Op.const 0.0).broadcast [BK, BN]))),
    Stmt.assign .real [BM, BN] "accumulator"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.dot (batch := []) (Op.ref .real [BM, BK] "a") (Op.ref .real [BK, BN] "b"))
        (Op.ref .real [BM, BN] "accumulator")),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sak))),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "b_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sbk))) ]

/-- The 6-statement post-loop tail, parameterized by the epilogue's output
dtype `od` (`.fp16` / `.f8e4` — the two constexpr arms): `od` cast, the two
`offs_c*` vectors, the `c_ptrs` chunk, the `c_mask`, and the masked store. -/
def matmulStoreTail (od : FloatDType) (C : RegionName) (M N scm scn BM BN : Nat) : List Stmt :=
  [ Stmt.assign od.toTileDType [BM, BN] "c"
      (Op.castFloat FloatDType.real od (Op.ref .real [BM, BN] "accumulator")),
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
          (Op.mul .nat Broadcast.scalarL (Op.constNat scm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat scn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))),
    Stmt.assign .bool [BM, BN] "c_mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))),
    Stmt.store od.toTileDType [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
      (Op.ref od.toTileDType [BM, BN] "c") (.mask (Op.ref .bool [BM, BN] "c_mask")) ]

/-- The 21-statement prologue (statements 0–20) as an explicit list: the 9
L2-grouping schedule scalars, the `start_m`/`start_n` scalars, the raw index
vectors, the two `tl.where` clamps, the two DSL-erased
`tl.max_contiguous(tl.multiple_of(…))` identity reassignments, `offs_k`, the
two pointer seeds, and the zero accumulator. -/
def matmulPrologue (A B : RegionName)
    (M N BM BN BK GM sam sak sbk sbn : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid" (Op.programId 0),
    Stmt.assign .nat [] "num_pid_m"
      (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
        (Op.constNat BM)),
    Stmt.assign .nat [] "num_pid_n"
      (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
        (Op.constNat BN)),
    Stmt.assign .nat [] "num_pid_in_group"
      (Op.mul .nat Broadcast.nil (Op.constNat GM) (Op.ref .nat [] "num_pid_n")),
    Stmt.assign .nat [] "group_id"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group")),
    Stmt.assign .nat [] "first_pid_m"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM)),
    Stmt.assign .nat [] "group_size_m"
      ((Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
          (Op.constNat GM)).where
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
        (Op.constNat GM)),
    Stmt.assign .nat [] "pid_m"
      (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
        (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "group_size_m"))),
    Stmt.assign .nat [] "pid_n"
      (Op.floorDiv .nat Broadcast.nil
        (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group"))
        (Op.ref .nat [] "group_size_m")),
    Stmt.assign .nat [] "start_m"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)),
    Stmt.assign .nat [] "start_n"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)),
    Stmt.assign .nat [BM] "offs_am"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_m") (Op.arange BM)),
    Stmt.assign .nat [BN] "offs_bn"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.arange BN)),
    Stmt.assign .nat [BM] "offs_am"
      ((Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "offs_am") (Op.constNat M)).where
        (Op.ref .nat [BM] "offs_am") ((Op.constNat 0).broadcast [BM])),
    Stmt.assign .nat [BN] "offs_bn"
      ((Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn") (Op.constNat N)).where
        (Op.ref .nat [BN] "offs_bn") ((Op.constNat 0).broadcast [BN])),
    Stmt.assign .nat [BM] "offs_am" (Op.ref .nat [BM] "offs_am"),
    Stmt.assign .nat [BN] "offs_bn" (Op.ref .nat [BN] "offs_bn"),
    Stmt.assign .nat [BK] "offs_k" (Op.arange BK),
    Stmt.assign .ptr [BM, BK] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat sam))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sak)))),
    Stmt.assign .ptr [BK, BN] "b_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sbk))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat sbn)))),
    Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)) ]

/-- The whole lowered body, shared by both epilogue arms up to `od`:
prologue (21) ++ [K-loop] ++ store-tail (6). -/
def matmulBody (od : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) : List Stmt :=
  matmulPrologue A B M N BM BN BK GM sam sak sbk sbn
    ++ (Stmt.forRange "kk" 0 numKBlocks 1 (matmulLoopBody K BM BN BK sak sbk)
        :: matmulStoreTail od C M N scm scn BM BN)

/-- Body decomposition of the fp16 arm. By `rfl`. -/
theorem triton_matmul_f16_body_eq (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) :
    (triton_matmul_f16_surface A B C M N K sam sak sbk sbn scm scn BM BN BK GM
        numKBlocks).toAlgKernel.body
      = matmulBody FloatDType.fp16 A B C M N K sam sak sbk sbn scm scn BM BN BK GM
          numKBlocks := rfl

/-- Body decomposition of the fp8 arm. By `rfl`. -/
theorem triton_matmul_f8_body_eq (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) :
    (triton_matmul_f8_surface A B C M N K sam sak sbk sbn scm scn BM BN BK GM
        numKBlocks).toAlgKernel.body
      = matmulBody FloatDType.f8e4 A B C M N K sam sak sbk sbn scm scn BM BN BK GM
          numKBlocks := rfl

/-! ## Loop invariant -/

/-- **Loop invariant** (counter `c = block index`, step `1`).

After `c` K-blocks: program ids and `mem`/`undef` fixed; the `pid_m`/`pid_n`/
`offs_am`/`offs_bn`/`offs_k` registers seeded (the index vectors through the
`tl.where` clamps); `accumulator` equals the partial GEMM accumulator
`accPartial … c`; and the `a_ptrs`/`b_ptrs` advanced by `c` blocks. -/
noncomputable def matmulInvariant
    (A B : RegionName) (s0 : BlockState) (M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks : Nat)
    (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ c ≤ numKBlocks ∧
  (s.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s0.pids 0) M N BM BN GM))) ∧
  (s.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s0.pids 0) M N BM BN GM))) ∧
  (s.regs .real [BM, BN] "accumulator" = some ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 c)⟩) ∧
  (s.regs .nat [BM] "offs_am" = some (Tile.vec (fun r : Fin BM => rowIndex s0 M N BM BN GM r))) ∧
  (s.regs .nat [BN] "offs_bn" = some (Tile.vec (fun j : Fin BN => colIndex s0 M N BM BN GM j))) ∧
  (s.regs .nat [BLOCK_K] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_K => e.val))) ∧
  (s.regs .ptr [BM, BLOCK_K] "a_ptrs" = some ⟨fun idx : TileIndex [BM, BLOCK_K] =>
      (A.cast, rowIndex s0 M N BM BN GM idx.1 * sam + idx.2.1.val * sak + c * BLOCK_K * sak)⟩) ∧
  (s.regs .ptr [BLOCK_K, BN] "b_ptrs" = some ⟨fun idx : TileIndex [BLOCK_K, BN] =>
      (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BLOCK_K * sbk)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

set_option maxHeartbeats 4000000 in
/-- **preLoop scalars** (statements 0–17): the 9 L2-schedule scalars, the
`start_m`/`start_n` scalars, and the clamped index vectors
`offs_am`/`offs_bn`/`offs_k` (through the `tl.where` clamps and the DSL-erased
`tl.max_contiguous` identity reassignments). -/
theorem preLoop_scalars (s : BlockState) (M N BM BN BK GM : Nat) :
    ∃ s18, stepStmts
      [ Stmt.assign .nat [] "pid" (Op.programId 0),
        Stmt.assign .nat [] "num_pid_m"
          (Op.div .nat Broadcast.nil
            (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
            (Op.constNat BM)),
        Stmt.assign .nat [] "num_pid_n"
          (Op.div .nat Broadcast.nil
            (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
            (Op.constNat BN)),
        Stmt.assign .nat [] "num_pid_in_group"
          (Op.mul .nat Broadcast.nil (Op.constNat GM) (Op.ref .nat [] "num_pid_n")),
        Stmt.assign .nat [] "group_id"
          (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group")),
        Stmt.assign .nat [] "first_pid_m"
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM)),
        Stmt.assign .nat [] "group_size_m"
          ((Op.lt ComparableDType.nat Broadcast.nil
              (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
              (Op.constNat GM)).where
            (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
            (Op.constNat GM)),
        Stmt.assign .nat [] "pid_m"
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
            (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "group_size_m"))),
        Stmt.assign .nat [] "pid_n"
          (Op.floorDiv .nat Broadcast.nil
            (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group"))
            (Op.ref .nat [] "group_size_m")),
        Stmt.assign .nat [] "start_m"
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)),
        Stmt.assign .nat [] "start_n"
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)),
        Stmt.assign .nat [BM] "offs_am"
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_m") (Op.arange BM)),
        Stmt.assign .nat [BN] "offs_bn"
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.arange BN)),
        Stmt.assign .nat [BM] "offs_am"
          ((Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "offs_am") (Op.constNat M)).where
            (Op.ref .nat [BM] "offs_am") ((Op.constNat 0).broadcast [BM])),
        Stmt.assign .nat [BN] "offs_bn"
          ((Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn") (Op.constNat N)).where
            (Op.ref .nat [BN] "offs_bn") ((Op.constNat 0).broadcast [BN])),
        Stmt.assign .nat [BM] "offs_am" (Op.ref .nat [BM] "offs_am"),
        Stmt.assign .nat [BN] "offs_bn" (Op.ref .nat [BN] "offs_bn"),
        Stmt.assign .nat [BK] "offs_k" (Op.arange BK) ] s = some s18
      ∧ s18.pids = s.pids
      ∧ s18.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s.pids 0) M N BM BN GM))
      ∧ s18.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s.pids 0) M N BM BN GM))
      ∧ s18.regs .nat [BM] "offs_am" = some (Tile.vec (fun i : Fin BM => rowIndex s M N BM BN GM i))
      ∧ s18.regs .nat [BN] "offs_bn" = some (Tile.vec (fun j : Fin BN => colIndex s M N BM BN GM j))
      ∧ s18.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))
      ∧ s18.undef = s.undef
      ∧ s18.mem = s.mem := by
  simp only [pidM, pidN, rowIndex, colIndex, rowGlobal, colGlobal, cdiv, kernelMin, clampIdx]
  simp [stepStmts, stepStmt, evalOp_floorDiv, evalOp_mod, Option.bind, BlockState.setReg,
    Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.div,
    NumericDType.sub, IntegralDType.floorDiv, IntegralDType.mod, Tile.select, Tile.cop,
    ComparableDType.lt]

set_option maxHeartbeats 1000000 in
/-- **preLoop** (the full 21-statement prologue): from a clean input state
(`undef = 0`), the prologue steps to a state satisfying `matmulInvariant … 0` —
the base case (`accumulator = 0`, pointers seeded, schedule + clamps derived). -/
theorem preLoop (A B : RegionName) (s : BlockState)
    (M N BM BN BK sam sak sbk sbn GM numKBlocks : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts (matmulPrologue A B M N BM BN BK GM sam sak sbk sbn) s = some s'
      ∧ matmulInvariant A B s M N BM BN GM sam sak sbk sbn BK numKBlocks 0 s' := by
  obtain ⟨s18, h18, hpids, hpm, hpn, hm, hn, hk, huf, hmem⟩ := preLoop_scalars s M N BM BN BK GM
  rw [show matmulPrologue A B M N BM BN BK GM sam sak sbk sbn
      = [ Stmt.assign .nat [] "pid" (Op.programId 0),
          Stmt.assign .nat [] "num_pid_m"
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
              (Op.constNat BM)),
          Stmt.assign .nat [] "num_pid_n"
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
              (Op.constNat BN)),
          Stmt.assign .nat [] "num_pid_in_group"
            (Op.mul .nat Broadcast.nil (Op.constNat GM) (Op.ref .nat [] "num_pid_n")),
          Stmt.assign .nat [] "group_id"
            (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group")),
          Stmt.assign .nat [] "first_pid_m"
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM)),
          Stmt.assign .nat [] "group_size_m"
            ((Op.lt ComparableDType.nat Broadcast.nil
                (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
                (Op.constNat GM)).where
              (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
              (Op.constNat GM)),
          Stmt.assign .nat [] "pid_m"
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
              (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "group_size_m"))),
          Stmt.assign .nat [] "pid_n"
            (Op.floorDiv .nat Broadcast.nil
              (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group"))
              (Op.ref .nat [] "group_size_m")),
          Stmt.assign .nat [] "start_m"
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)),
          Stmt.assign .nat [] "start_n"
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)),
          Stmt.assign .nat [BM] "offs_am"
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_m") (Op.arange BM)),
          Stmt.assign .nat [BN] "offs_bn"
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.arange BN)),
          Stmt.assign .nat [BM] "offs_am"
            ((Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "offs_am") (Op.constNat M)).where
              (Op.ref .nat [BM] "offs_am") ((Op.constNat 0).broadcast [BM])),
          Stmt.assign .nat [BN] "offs_bn"
            ((Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn") (Op.constNat N)).where
              (Op.ref .nat [BN] "offs_bn") ((Op.constNat 0).broadcast [BN])),
          Stmt.assign .nat [BM] "offs_am" (Op.ref .nat [BM] "offs_am"),
          Stmt.assign .nat [BN] "offs_bn" (Op.ref .nat [BN] "offs_bn"),
          Stmt.assign .nat [BK] "offs_k" (Op.arange BK) ]
      ++ [ Stmt.assign .ptr [BM, BK] "a_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat sam))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sak)))),
          Stmt.assign .ptr [BK, BN] "b_ptrs"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
              (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sbk))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat sbn)))),
          Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)) ] from rfl,
    stepStmts.append_some h18,
    stepStmts.cons_some (stepStmt_assign_eq_some
      (aptrs_eval s18 A BM BK sam sak (fun i => rowIndex s M N BM BN GM i) (by simpa using hm) (by simpa using hk))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (bptrs_eval _ B BK BN sbk sbn (fun j => colIndex s M N BM BN GM j) (by simp [hk]) (by simp [hn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval _ BM BN)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp [hpids], by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hpm]
  · simp [hpn]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    simp only [accPartial, Nat.zero_mul, gemmSum_zero]
  · simp [hm]
  · simp [hn]
  · simp [hk]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · intro rg o; simp [huf, hundef]
  · exact hmem

/-- The `a` load mask tile evaluates to **all-true** when the K-block index `c`
satisfies `c < numKBlocks` and `K = BK · numKBlocks` (so `offs_k[e] < K - c·BK`
holds for every K-lane). -/
theorem aMaskOp_eval (st : BlockState) (K BM BK numKBlocks c : Nat)
    (hK : K = BK * numKBlocks) (hc : c < numKBlocks)
    (hk : st.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkk : st.regs .nat [] "kk" = some (Tile.scalar c)) :
    ∃ masks : Tile .bool [BM, BK], evalOp (aMaskOp K BM BK) st = some masks
      ∧ ∀ i, masks.data i = Bool.true := by
  have heval : evalOp (aMaskOp K BM BK) st = some
      (Tile.remap (dtype := .bool) Broadcast.nil.consSame.consL.leftIndex
        (Tile.cop ComparableDType.nat.lt Broadcast.scalarR
          (⟨fun i : TileIndex [1, BK] => (Tile.vec (fun e : Fin BK => (e.val : Nat))).data (i.2.1, PUnit.unit)⟩
            : Tile .nat [1, BK])
          (Tile.bop NumericDType.nat.sub Broadcast.nil (Tile.scalar K)
            (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar c) (Tile.scalar BK))))) := by
    unfold aMaskOp
    rw [evalOp_remap]
    conv_lhs => arg 1; rw [evalOp_lt]; arg 1; rw [evalOp_expandDim_zero_nat, hk]
    simp only [Option.bind_eq_bind, Option.bind_some, evalOp_sub, evalOp_mul, evalOp_constNat,
      evalOp_ref, hkk]
  refine ⟨_, heval, ?_⟩
  intro i
  simp only [Tile.remap, Tile.cop_data, Tile.vec, Tile.scalar, Tile.bop_data,
    Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt, NumericDType.sub, NumericDType.mul,
    decide_eq_true_eq]
  have : (i.2.1.val : Nat) < K - c * BK := by
    have hlt : (i.2.1.val : Nat) < BK := i.2.1.isLt
    have : c * BK + BK ≤ K := by
      rw [hK]; calc c * BK + BK = (c + 1) * BK := by ring
        _ ≤ numKBlocks * BK := Nat.mul_le_mul_right _ hc
        _ = BK * numKBlocks := Nat.mul_comm _ _
    omega
  simpa using this

/-- The `b` load mask tile evaluates to all-true under the same condition. -/
theorem bMaskOp_eval (st : BlockState) (K BK BN numKBlocks c : Nat)
    (hK : K = BK * numKBlocks) (hc : c < numKBlocks)
    (hk : st.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkk : st.regs .nat [] "kk" = some (Tile.scalar c)) :
    ∃ masks : Tile .bool [BK, BN], evalOp (bMaskOp K BK BN) st = some masks
      ∧ ∀ i, masks.data i = Bool.true := by
  have heval : evalOp (bMaskOp K BK BN) st = some
      (Tile.remap (dtype := .bool) Broadcast.nil.consL.consSame.leftIndex
        (Tile.cop ComparableDType.nat.lt Broadcast.scalarR
          (⟨fun i : TileIndex [BK, 1] => (Tile.vec (fun e : Fin BK => (e.val : Nat))).data (i.1, PUnit.unit)⟩
            : Tile .nat [BK, 1])
          (Tile.bop NumericDType.nat.sub Broadcast.nil (Tile.scalar K)
            (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar c) (Tile.scalar BK))))) := by
    unfold bMaskOp
    rw [evalOp_remap]
    conv_lhs => arg 1; rw [evalOp_lt]; arg 1; rw [evalOp_expandDim_one_nat, hk]
    simp only [Option.bind_eq_bind, Option.bind_some, evalOp_sub, evalOp_mul, evalOp_constNat,
      evalOp_ref, hkk]
  refine ⟨_, heval, ?_⟩
  intro i
  simp only [Tile.remap, Tile.cop_data, Tile.vec, Tile.scalar, Tile.bop_data,
    Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt, NumericDType.sub, NumericDType.mul,
    decide_eq_true_eq]
  have : (i.1.val : Nat) < K - c * BK := by
    have hlt : (i.1.val : Nat) < BK := i.1.isLt
    have : c * BK + BK ≤ K := by
      rw [hK]; calc c * BK + BK = (c + 1) * BK := by ring
        _ ≤ numKBlocks * BK := Nat.mul_le_mul_right _ hc
        _ = BK * numKBlocks := Nat.mul_comm _ _
    omega
  simpa using this

/-- `(Op.const 0.0).broadcast [M,N]` eval succeeds (the `other` operand; its value
is irrelevant since the load mask is uniformly true). -/
theorem const_broadcast_eval (st : BlockState) (M N : Nat) :
    ∃ t : Tile .real [M, N], evalOp ((Op.const (0.0 : ℝ)).broadcast [M, N]) st = some t := by
  simp only [evalOp, evalOp_const, Option.bind_eq_bind, Option.bind_some]
  exact ⟨_, rfl⟩

/-- The masked `a` load, keyed on register readbacks: at exact-multiple `K` and
block index `c < numKBlocks`, it reads the genuine `readMem` cells. -/
theorem load_a_eval (st : BlockState) (K BM BK numKBlocks c : Nat) (ap : Tile .ptr [BM, BK])
    (hK : K = BK * numKBlocks) (hc : c < numKBlocks)
    (hap : st.regs .ptr [BM, BK] "a_ptrs" = some ap)
    (hkof : st.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkk : st.regs .nat [] "kk" = some (Tile.scalar c)) :
    evalOp (Op.load .real (.ptr (Op.ref .ptr [BM, BK] "a_ptrs"))
        (.maskOther (aMaskOp K BM BK) ((Op.const 0.0).broadcast [BM, BK]))) st
      = some ⟨fun i => some (st.readMem (ap.data i).1 (ap.data i).2)⟩ := by
  obtain ⟨amasks, hamasks, htrue⟩ := aMaskOp_eval st K BM BK numKBlocks c hK hc hkof hkk
  obtain ⟨aother, haother⟩ := const_broadcast_eval st BM BK
  exact load_ptr_maskOther_true_real (Op.ref .ptr [BM, BK] "a_ptrs") (aMaskOp K BM BK)
    ((Op.const 0.0).broadcast [BM, BK]) st ap amasks (by rw [evalOp_ref, hap]) hamasks
    aother haother htrue

/-- The masked `b` load, keyed on register readbacks. -/
theorem load_b_eval (st : BlockState) (K BK BN numKBlocks c : Nat) (bp : Tile .ptr [BK, BN])
    (hK : K = BK * numKBlocks) (hc : c < numKBlocks)
    (hbp : st.regs .ptr [BK, BN] "b_ptrs" = some bp)
    (hkof : st.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hkk : st.regs .nat [] "kk" = some (Tile.scalar c)) :
    evalOp (Op.load .real (.ptr (Op.ref .ptr [BK, BN] "b_ptrs"))
        (.maskOther (bMaskOp K BK BN) ((Op.const 0.0).broadcast [BK, BN]))) st
      = some ⟨fun i => some (st.readMem (bp.data i).1 (bp.data i).2)⟩ := by
  obtain ⟨bmasks, hbmasks, htrue⟩ := bMaskOp_eval st K BK BN numKBlocks c hK hc hkof hkk
  obtain ⟨bother, hbother⟩ := const_broadcast_eval st BK BN
  exact load_ptr_maskOther_true_real (Op.ref .ptr [BK, BN] "b_ptrs") (bMaskOp K BK BN)
    ((Op.const 0.0).broadcast [BK, BN]) st bp bmasks (by rw [evalOp_ref, hbp]) hbmasks
    bother hbother htrue

set_option maxHeartbeats 2000000 in
/-- **Step lemma**: one K-loop body iteration advances the invariant by one block.
The masked loads (uniformly-true masks at exact-multiple `K`) read the genuine
`A`/`B` cells, the fused `tl.dot(a, b, accumulator)` adds the `c`-th block's dot to
the partial GEMM accumulator, and the `a`/`b` pointers advance one step. -/
theorem matmul_step (A B : RegionName) (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks : Nat) (K : Nat) (hK : K = BLOCK_K * numKBlocks)
    (c : Nat) (s : BlockState) (hclt : c < numKBlocks)
    (hinv : matmulInvariant A B s0 M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks c s) :
    ∃ s', stepStmts (matmulLoopBody K BM BN BLOCK_K sak sbk)
        (s.setReg "kk" .nat [] (Tile.scalar c)) = some s'
      ∧ matmulInvariant A B s0 M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks (c + 1) s' := by
  simp only [matmulInvariant] at hinv
  obtain ⟨hpids, hcle, hpm, hpn, hz, hm, hn, hk, hap, hbp, hundef, hmem⟩ := hinv
  set apT : Tile .ptr [BM, BLOCK_K] :=
    ⟨fun idx : TileIndex [BM, BLOCK_K] =>
      (A.cast, rowIndex s0 M N BM BN GM idx.1 * sam + idx.2.1.val * sak + c * BLOCK_K * sak)⟩ with hapT
  set bpT : Tile .ptr [BLOCK_K, BN] :=
    ⟨fun idx : TileIndex [BLOCK_K, BN] =>
      (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BLOCK_K * sbk)⟩ with hbpT
  set zT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 c)⟩ with hzT
  set sk := s.setReg "kk" .nat [] (Tile.scalar c) with hsk
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  have hapk : sk.regs .ptr [BM, BLOCK_K] "a_ptrs" = some apT := by simp [hsk, hap, hapT]
  have hbpk : sk.regs .ptr [BLOCK_K, BN] "b_ptrs" = some bpT := by simp [hsk, hbp, hbpT]
  have hzk : sk.regs .real [BM, BN] "accumulator" = some zT := by simp [hsk, hz, hzT]
  have hkk : sk.regs .nat [] "kk" = some (Tile.scalar c) := by simp [hsk]
  have hkofk : sk.regs .nat [BLOCK_K] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_K => e.val)) := by
    simp [hsk, hk]
  set asub : Tile .real [BM, BLOCK_K] :=
    ⟨fun idx => some (sk.readMem (apT.data idx).1 (apT.data idx).2)⟩ with hasub
  set bsub : Tile .real [BLOCK_K, BN] :=
    ⟨fun idx => some (sk.readMem (bpT.data idx).1 (bpT.data idx).2)⟩ with hbsub
  unfold matmulLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_a_eval sk K BM BLOCK_K numKBlocks c apT hK hclt hapk hkofk hkk))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_b_eval _ K BLOCK_K BN numKBlocks c bpT hK hclt
          (by simp [hbpk]) (by simp [hkofk]) (by simp [hkk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (accdot_op_eval BM BLOCK_K BN _ zT asub bsub
          (by simp [hzk, hasub, hbsub, BlockState.setReg_readMem])
          (by simp [hasub, BlockState.setReg_readMem])
          (by simp [hbsub, BlockState.setReg_readMem])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (aptr_adv_eval _ BM BLOCK_K BLOCK_K sak apT (by simp [hapk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (bptr_adv_eval _ BLOCK_K BN BLOCK_K sbk bpT (by simp [hbpk])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [matmulInvariant]
  refine ⟨by simp [hsk, hpids], by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hsk, hpm]
  · simp [hsk, hpn]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    have has : ∀ e : Fin BLOCK_K, asub.data (idx.1, e, PUnit.unit)
        = some (aElem s0 A M N BM BN GM sam sak idx.1 (c * BLOCK_K + e.val)) := by
      intro e
      simp only [hasub, hapT, hrmem, aElem, Region.cast_id]
      rw [show rowIndex s0 M N BM BN GM idx.1 * sam + e.val * sak + c * BLOCK_K * sak
            = rowIndex s0 M N BM BN GM idx.1 * sam + (c * BLOCK_K + e.val) * sak from by ring]
    have hbs : ∀ e : Fin BLOCK_K, bsub.data (e, idx.2.1, PUnit.unit)
        = some (bElem s0 B M N BM BN GM sbk sbn idx.2.1 (c * BLOCK_K + e.val)) := by
      intro e
      simp only [hbsub, hbpT, hrmem, bElem, Region.cast_id]
      rw [show e.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BLOCK_K * sbk
            = (c * BLOCK_K + e.val) * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn from by ring]
    rw [dotadd_eval BM BN (Tile.dot [] asub bsub) zT idx.1 idx.2.1
        (Finset.univ.sum fun e : Fin BLOCK_K =>
          aElem s0 A M N BM BN GM sam sak idx.1 (c * BLOCK_K + e.val)
            * bElem s0 B M N BM BN GM sbk sbn idx.2.1 (c * BLOCK_K + e.val))
        (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 c)
        (tile_dot_data BM BLOCK_K BN asub bsub idx.1 idx.2.1 _ _ has hbs)
        (by rw [hzT])]
    show some _ = some (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 (c + 1))
    rw [accPartial_succ, add_comm]
  · simp [hsk, hm]
  · simp [hsk, hn]
  · simp [hsk, hk]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hapT, NumericDType.add]
    ring
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hbpT, NumericDType.add]
    ring
  · intro rg o; simp [hsk, hundef]
  · show _ = s0.mem
    rw [← hmem, hsk]; rfl

/-! ## Post-loop: od cast + masked store -/

/-- The output store address for tile lane `(i,j)`: `scm · offs_cm i + scn · offs_cn j`
(the kernel's `c_ptrs`, using the **un-clamped** global `offs_cm`/`offs_cn`,
recomputed fresh from `pid_m`/`pid_n`). -/
def cOffset (s0 : BlockState) (M N BM BN GM scm scn : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  scm * rowGlobal s0 M N BM BN GM idx.1 + scn * colGlobal s0 M N BM BN GM idx.2.1

/-- The boundary predicate `(row < M) & (col < N)` for tile lane `(i,j)`. -/
def active (s0 : BlockState) (M N BM BN GM : Nat) (idx : TileIndex [BM, BN]) : Prop :=
  rowGlobal s0 M N BM BN GM idx.1 < M ∧ colGlobal s0 M N BM BN GM idx.2.1 < N

instance activeDecidable (s0 : BlockState) (M N BM BN GM : Nat)
    (idx : TileIndex [BM, BN]) : Decidable (active s0 M N BM BN GM idx) := by
  unfold active; infer_instance

/-- `offs_cm` / `offs_cn` eval (the **un-clamped** global index, no `tl.where`). -/
theorem offscm_eval (s : BlockState) (M BM : Nat) (pm : Nat) (pidReg : RegName)
    (hpm : s.regs .nat [] pidReg = some (Tile.scalar pm)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat BM)) (Op.arange M)) s
      = some (Tile.vec (fun i : Fin M => pm * BM + i.val)) := by
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, evalOp_arange, hpm,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

theorem evalOp_boolAnd {a b shape} (bc : Broadcast a b shape)
    (x : Op .bool a) (y : Op .bool b) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop (· && ·) bc vx vy)) := by
  simp [evalOp]

/-- `c_mask` eval: the `(offs_cm < M) & (offs_cn < N)` boolean tile. -/
theorem cmask_eval (s : BlockState) (M N BM BN : Nat) (gm : Fin BM → Nat) (gn : Fin BN → Nat)
    (hm : s.regs .nat [BM] "offs_cm" = some (Tile.vec gm))
    (hn : s.regs .nat [BN] "offs_cn" = some (Tile.vec gn)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))) s
      = some ⟨fun idx : TileIndex [BM, BN] => (decide (gm idx.1 < M) && decide (gn idx.2.1 < N))⟩ := by
  rw [evalOp_boolAnd, evalOp_lt, evalOp_lt, evalOp_expandDim_one_nat, evalOp_expandDim_zero_nat]
  simp only [hm, hn, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp [Tile.bop_data, Tile.cop_data, Tile.expandDim, Tile.vec, Tile.scalar_data, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex]

/-! ### od-generic float-store bridges

The two epilogue arms differ only in the store grid (`.fp16` vs `.f8e4`), so
the store-side lemmas are stated once over `od : FloatDType`; the exact
`writeMemTyped`-scatter readback is transported from the library's generic
`R`-scatter readback at `R := .triv`. -/

/-- `writeMemTyped` at a float tag is `writeMemAs` (every float channel of the
`writeMemTyped` match delegates). -/
private theorem writeMemTyped_float (od : FloatDType) (s : BlockState) (region : RegionName)
    (offset : Nat) (v : TileCarrier od.toTileDType) :
    s.writeMemTyped od.toTileDType region offset v = s.writeMemAs od region offset v := by
  cases od <;> rfl

/-- `writeMemTypedR` at the two epilogue tags is `writeMemAsR` (rounding-event
store site). -/
private theorem writeMemTypedR_float (R : RoundingModel) (od : FloatDType)
    (hod : od = FloatDType.fp16 ∨ od = FloatDType.f8e4)
    (s : BlockState) (region : RegionName) (offset : Nat) (v : TileCarrier od.toTileDType) :
    s.writeMemTypedR R od.toTileDType region offset v = s.writeMemAsR R od region offset v := by
  rcases hod with rfl | rfl <;> rfl

/-- Readback of a `P`-masked `writeMemTyped` float scatter store at lane `i`'s
offset: the stored `od` cell if `P i`, else the prior contents, given injective
offsets (the od-generic sibling of the library's
`scatter_memcell_fp16_prop_masked_nd`, via the `R := .triv` degeneration of
`scatter_memcell_R_prop_masked_nd`). -/
private theorem scatter_memcell_float_prop_masked_nd (od : FloatDType) {region : RegionName}
    {shape : TileShape} (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier od.toTileDType)
    (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped od.toTileDType region (offsetFn k) (valueFn k) else acc)
       s).mem region (offsetFn i)
    = if P i then
        MemCell.of od.toTileDType (od.ofReal (od.storeValue (valueFn i)))
      else s.mem region (offsetFn i) := by
  have h := BlockState.scatter_memcell_R_prop_masked_nd RoundingModel.triv od
    (region := region) s offsetFn valueFn P h_inj i
  have hfn : (fun (acc : BlockState) k =>
        if P k then acc.writeMemAsR RoundingModel.triv od region (offsetFn k) (valueFn k) else acc)
      = (fun (acc : BlockState) k =>
        if P k then acc.writeMemTyped od.toTileDType region (offsetFn k) (valueFn k) else acc) := by
    funext acc k
    by_cases hk : P k
    · rw [if_pos hk, if_pos hk, BlockState.writeMemAsR_triv]
      exact (writeMemTyped_float od acc region (offsetFn k) (valueFn k)).symm
    · rw [if_neg hk, if_neg hk]
  rw [hfn] at h
  rw [h]
  by_cases hPi : P i
  · rw [if_pos hPi, if_pos hPi]
    simp only [RoundingModel.triv_storeValue]
  · rw [if_neg hPi, if_neg hPi]

set_option maxHeartbeats 4000000 in
/-- **postLoop** (od-generic over the two epilogue arms): from the invariant at
`numKBlocks` blocks, the `od` cast + masked store writes the genuine closed form
`od( Σ_k A·B )` at every active output lane (given the output-offset map is
injective). -/
theorem matmul_postLoop (od : FloatDType)
    (hod : od = FloatDType.fp16 ∨ od = FloatDType.f8e4)
    (A B C : RegionName) (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks : Nat)
    (hInj : Function.Injective (cOffset s0 M N BM BN GM scm scn))
    (st : BlockState)
    (hinv : matmulInvariant A B s0 M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks numKBlocks st) :
    ∃ sfin, stepStmts (matmulStoreTail od C M N scm scn BM BN) st = some sfin
      ∧ ∀ idx : TileIndex [BM, BN],
          sfin.mem C (cOffset s0 M N BM BN GM scm scn idx)
            = if active s0 M N BM BN GM idx then
                MemCell.of od.toTileDType
                  (FloatDType.real.cast od
                    (some (matmulSpec s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks idx.1 idx.2.1)))
              else
                st.mem C (cOffset s0 M N BM BN GM scm scn idx) := by
  simp only [matmulInvariant] at hinv
  obtain ⟨hpids, hcle, hpm, hpn, hz, hm, hn, hk, hap, hbp, hundef, hmem⟩ := hinv
  set accT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 numKBlocks)⟩
      with haccT
  set cT : Tile od.toTileDType [BM, BN] :=
    ⟨fun idx => FloatDType.real.cast od (accT.data idx)⟩ with hcT
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (C.cast, cOffset s0 M N BM BN GM scm scn idx)⟩ with hcpT
  set cmaskT : Tile .bool [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      (decide (rowGlobal s0 M N BM BN GM idx.1 < M) && decide (colGlobal s0 M N BM BN GM idx.2.1 < N))⟩
      with hcmaskT
  unfold matmulStoreTail
  -- c = cast(accumulator, od)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.castFloat FloatDType.real od (Op.ref .real [BM, BN] "accumulator")) st
          = some cT from by rw [evalOp_castFloat]; simp [evalOp_ref, hz, hcT, haccT]))]
  -- offs_cm
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (offscm_eval _ BM BM (pidM (s0.pids 0) M N BM BN GM) "pid_m" (by simp [hpm])))]
  -- offs_cn
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (offscm_eval _ BN BN (pidN (s0.pids 0) M N BM BN GM) "pid_n" (by simp [hpn])))]
  -- c_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.mul .nat Broadcast.scalarL (Op.constNat scm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
              (Op.mul .nat Broadcast.scalarL (Op.constNat scn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) _
          = some cpT from by
          rw [cptrs_eval _ C BM BN scm scn (fun i => rowGlobal s0 M N BM BN GM i) (fun j => colGlobal s0 M N BM BN GM j)
                (by simp [rowGlobal]) (by simp [colGlobal])]
          rfl))]
  -- c_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))) _
          = some cmaskT from by
          rw [cmask_eval _ M N BM BN (fun i => rowGlobal s0 M N BM BN GM i) (fun j => colGlobal s0 M N BM BN GM j)
                (by simp [rowGlobal]) (by simp [colGlobal])]))]
  -- abstract the post-assign state
  generalize hst4 : (((((st.setReg "c" od.toTileDType [BM, BN] cT).setReg "offs_cm" .nat [BM]
        (Tile.vec fun i : Fin BM => pidM (s0.pids 0) M N BM BN GM * BM + i.val)).setReg "offs_cn" .nat [BN]
        (Tile.vec fun j : Fin BN => pidN (s0.pids 0) M N BM BN GM * BN + j.val)).setReg "c_ptrs" .ptr [BM, BN] cpT).setReg
        "c_mask" .bool [BM, BN] cmaskT) = st4
  have hc4 : st4.regs od.toTileDType [BM, BN] "c" = some cT := by rw [← hst4]; simp
  have hcp4 : st4.regs .ptr [BM, BN] "c_ptrs" = some cpT := by rw [← hst4]; simp
  have hcm4 : st4.regs .bool [BM, BN] "c_mask" = some cmaskT := by rw [← hst4]; simp
  have hmem4 : st4.mem = st.mem := by
    rw [← hst4]; funext region offset; simp only [BlockState.setReg_mem]
  -- the masked store
  have hstore : stepStmt (Stmt.store od.toTileDType [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
        (Op.ref od.toTileDType [BM, BN] "c") (.mask (Op.ref .bool [BM, BN] "c_mask"))) st4
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc i =>
            if cmaskT.data i then
              acc.writeMemTyped od.toTileDType C (cOffset s0 M N BM BN GM scm scn i) (cT.data i)
            else acc) st4) := by
    simp only [stepStmt]
    rw [show evalOp (Op.ref od.toTileDType [BM, BN] "c") st4 = some cT from by rw [evalOp_ref, hc4]]
    rw [show evalOp (Op.ref .ptr [BM, BN] "c_ptrs") st4 = some cpT from by rw [evalOp_ref, hcp4]]
    rw [show evalOp (Op.ref .bool [BM, BN] "c_mask") st4 = some cmaskT from by rw [evalOp_ref, hcm4]]
    simp only [bind, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ _ (fun acc i _ => ?_))
    by_cases hmask : cmaskT.data i
    · simp only [hmask, if_true, cpT, hcpT, Region.cast_id]
    · simp only [hmask, Bool.false_eq_true, if_false]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  rw [scatter_memcell_float_prop_masked_nd od (region := C) (s := st4)
        (offsetFn := cOffset s0 M N BM BN GM scm scn)
        (valueFn := fun i => cT.data i)
        (P := fun i => cmaskT.data i = Bool.true) hInj idx]
  by_cases hact : active s0 M N BM BN GM idx
  · have hmasktrue : cmaskT.data idx = Bool.true := by
      simp only [hcmaskT]
      obtain ⟨hr, hcc⟩ := hact
      simp [hr, hcc]
    simp only [hmasktrue, if_true, if_pos hact]
    -- collapse the od round-trip and rewrite acc → matmulSpec
    rcases hod with rfl | rfl <;>
      simp only [hcT, haccT, matmulSpec, accPartial, Nat.mul_comm numKBlocks BLOCK_K,
        FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue, FloatDType.real_toWithBot,
        FloatDType.fp16_ofWithBot, FloatDType.fp16_toWithBot,
        FloatDType.f8e4_ofWithBot, FloatDType.f8e4_toWithBot, WithBot.unbotD_some]
  · have hmaskfalse : ¬ (cmaskT.data idx = Bool.true) := by
      simp only [hcmaskT, Bool.and_eq_true, decide_eq_true_eq, not_and]
      intro hr hcc
      exact hact ⟨hr, hcc⟩
    rw [if_neg hmaskfalse, if_neg hact, hmem4]

/-! ## Composition: full exec closed form -/

set_option maxHeartbeats 4000000 in
/-- **Top exec reduction** (od-generic): composes `preLoop` + `matmul_step`
(driven by `forRange_inv`) + `matmul_postLoop` into the full lowered-body
stepping result. Every active output lane's memory cell equals the genuine
closed form `od( Σ_k A·B )`. -/
theorem matmul_exec_closed_form (od : FloatDType)
    (hod : od = FloatDType.fp16 ∨ od = FloatDType.f8e4)
    (A B C : RegionName) (s : BlockState)
    (M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks : Nat) (K : Nat)
    (hK : K = BLOCK_K * numKBlocks)
    (hInj : Function.Injective (cOffset s M N BM BN GM scm scn))
    (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [BM, BN]) :
    (match stepStmts (matmulBody od A B C M N K sam sak sbk sbn scm scn BM BN BLOCK_K GM
        numKBlocks) s with
      | some s' => s'.mem C (cOffset s M N BM BN GM scm scn idx)
      | none => (0 : MemCell)) =
      (if active s M N BM BN GM idx then
        MemCell.of od.toTileDType
          (FloatDType.real.cast od
            (some (matmulSpec s A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks idx.1 idx.2.1)))
      else
        s.mem C (cOffset s M N BM BN GM scm scn idx)) := by
  -- preLoop establishes P 0
  obtain ⟨s0, hpre_eq, hP0⟩ := preLoop A B s M N BM BN BLOCK_K sam sak sbk sbn GM
    numKBlocks hundef
  -- drive the K-loop
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "kk") (start := 0) (stop := numKBlocks) (step := 1)
      (by omega) hP0
      (fun c st hlt hinv => by
        have := matmul_step A B s M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks K hK c st hlt hinv
        simpa using this)
  -- loop exit: final = numKBlocks
  have hfinalEq : final = numKBlocks := by
    have hle : final ≤ numKBlocks := by
      simp only [matmulInvariant] at hPLoop
      exact hPLoop.2.1
    exact le_antisymm hle hfinal
  rw [hfinalEq] at hPLoop
  -- postLoop reads off the closed form
  obtain ⟨sfin, hTail, hpost⟩ :=
    matmul_postLoop od hod A B C s M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks
      hInj sLoop hPLoop
  have hexec : stepStmts (matmulBody od A B C M N K sam sak sbk sbn scm scn BM BN BLOCK_K GM
      numKBlocks) s = some sfin := by
    unfold matmulBody
    rw [stepStmts.append_some hpre_eq, stepStmts.cons_some hLoopStmt]
    exact hTail
  rw [hexec]
  -- postLoop's `st.mem` is `sLoop.mem`, which equals `s.mem` (invariant)
  have hsloopmem : sLoop.mem = s.mem := by
    simp only [matmulInvariant] at hPLoop
    exact hPLoop.2.2.2.2.2.2.2.2.2.2.2
  have := hpost idx
  rw [hsloopmem] at this
  exact this

/-- **Closed-form correctness for `triton_matmul` (fp16 epilogue arm, general
statement).**

For arbitrary linear program id `pid`, tile dims `BM`/`BN`, K-block size
`BLOCK_K`, and K-block count `numKBlocks` (so the contracted dimension is
`K = BLOCK_K · numKBlocks`), every **active** output cell of the computed
`BM × BN` tile equals `fp16( Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j] )` —
the genuine matrix product (over ℝ) of the loaded `A`/`B` tiles, cast to
float16 — **not** the kernel's own executed value; inactive lanes are left
untouched.

Layout: `A[i,k]` at `A + offs_am(i)·stride_am + k·stride_ak`, `B[k,j]` at
`B + k·stride_bk + offs_bn(j)·stride_bn`, `C[i,j]` at
`C + stride_cm·offs_cm(i) + stride_cn·offs_cn(j)`, with `pid_m`/`pid_n` derived
by the kernel's L2-grouping schedule,
`offs_am(i) = if pid_m·BM + i < M then pid_m·BM + i else 0` and
`offs_bn(j) = if pid_n·BN + j < N then pid_n·BN + j else 0` (the row-major
pointer arithmetic with the kernel's `tl.where` index clamp). Preconditions:
output-offset injectivity and clean initial `undef`. -/
specification triton_matmul_f16_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks : Nat) (K : Nat)
    (hK : K = BLOCK_K * numKBlocks)
    (hcn : scn = 1) (hbnle : BN ≤ scm)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_matmul_f16_surface A B C M N K sam sak sbk sbn scm scn
        BM BN BLOCK_K GM numKBlocks)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN GM)
        (fun idx => (C, cOffset s M N BM BN GM scm scn idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (matmulSpec s A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks idx.1 idx.2.1)))) := by
  subst hcn
  -- output-offset injectivity from the row-major bound `BN ≤ scm` (col stride 1)
  have hInj : Function.Injective (cOffset s M N BM BN GM scm 1) := by
    have heq : cOffset s M N BM BN GM scm 1
        = fun idx : TileIndex [BM, BN] =>
            (scm * (pidM (s.pids 0) M N BM BN GM * BM) + pidN (s.pids 0) M N BM BN GM * BN)
              + idx.1.val * scm + idx.2.1.val := by
      funext idx; simp only [cOffset, rowGlobal, colGlobal]; ring
    rw [heq]; exact rowMajor2D_inj _ scm hbnle
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_matmul_f16_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx hActive
  have hmain := matmul_exec_closed_form FloatDType.fp16 (Or.inl rfl) A B C s0 M N BM BN GM
    sam sak sbk sbn scm 1 BLOCK_K numKBlocks K hK hInj hundef idx
  have hExec2 : stepStmts (matmulBody FloatDType.fp16 A B C M N K sam sak sbk sbn scm 1
      BM BN BLOCK_K GM numKBlocks) s0 = some s' := by
    rw [← triton_matmul_f16_body_eq A B C M N K sam sak sbk sbn scm 1 BM BN BLOCK_K GM numKBlocks]
    exact hExec
  rw [hExec2] at hmain
  rw [if_pos hActive] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_memcell] using hmain

/-- **Closed-form correctness for `triton_matmul` (fp8 epilogue arm, general
statement)**: identical to `triton_matmul_f16_closed_form_correct` except the
epilogue grid — every active output cell holds
`f8e4( Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j] )`, the genuine matrix
product cast to `float8e4nv` (the `c_ptr.dtype.element_ty == tl.float8e4nv`
constexpr arm). -/
specification triton_matmul_f8_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks : Nat) (K : Nat)
    (hK : K = BLOCK_K * numKBlocks)
    (hcn : scn = 1) (hbnle : BN ≤ scm)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_matmul_f8_surface A B C M N K sam sak sbk sbn scm scn
        BM BN BLOCK_K GM numKBlocks)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN GM)
        (fun idx => (C, cOffset s M N BM BN GM scm scn idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .f8e4
          (FloatDType.real.cast FloatDType.f8e4
            (some (matmulSpec s A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks idx.1 idx.2.1)))) := by
  subst hcn
  have hInj : Function.Injective (cOffset s M N BM BN GM scm 1) := by
    have heq : cOffset s M N BM BN GM scm 1
        = fun idx : TileIndex [BM, BN] =>
            (scm * (pidM (s.pids 0) M N BM BN GM * BM) + pidN (s.pids 0) M N BM BN GM * BN)
              + idx.1.val * scm + idx.2.1.val := by
      funext idx; simp only [cOffset, rowGlobal, colGlobal]; ring
    rw [heq]; exact rowMajor2D_inj _ scm hbnle
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_matmul_f8_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx hActive
  have hmain := matmul_exec_closed_form FloatDType.f8e4 (Or.inr rfl) A B C s0 M N BM BN GM
    sam sak sbk sbn scm 1 BLOCK_K numKBlocks K hK hInj hundef idx
  have hExec2 : stepStmts (matmulBody FloatDType.f8e4 A B C M N K sam sak sbk sbn scm 1
      BM BN BLOCK_K GM numKBlocks) s0 = some s' := by
    rw [← triton_matmul_f8_body_eq A B C M N K sam sak sbk sbn scm 1 BM BN BLOCK_K GM numKBlocks]
    exact hExec
  rw [hExec2] at hmain
  rw [if_pos hActive] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_memcell] using hmain

/-! ## The `⊨[R]` streaming headlines (wave-5 S1 fold genre)

Everything below is purely additive; the exact surfaces above are untouched.
Structure of the `execR R` story: each arm's only two rounding events live in
`matmulStoreTail od` — the `.to(tl.float16)` / `.to(tl.float8e4nv)` cast
(`evalOpR` site 1) and the `od`-typed masked `tl.store` (`writeMemAsR`
site 2). The prologue (including the `tl.where` index clamps — nat ops) and
the whole K-loop (masked loads with `other=0.0` carry no `castFloat`) are
cast-free, so under `execR R` they collapse verbatim onto the exact stepper
and the proven `preLoop` / `matmul_step` / `forRange_inv` stack above is
reused unchanged; only the 6-statement store tail is re-proved on the `R`
side (`matmul_postLoopR`, od-generic). `round_idem` (via
`RoundingModel.storeValue_cast`) collapses the tail's double round into the
single boundary `R.round od` the skin's readback contract states. -/

open scoped VeriTile.Triton.StreamMasked2DKernelIO₂

/-! ### Cast-free collapses -/

set_option maxHeartbeats 1000000 in
/-- The prologue is cast-free (the schedule scalars, `tl.where` clamps, and
pointer seeds are nat/ptr ops): it steps identically under `stepStmtsR R`. -/
private theorem matmulPrologue_castFree (R : RoundingModel) (A B : RegionName)
    (M N BM BN BK GM sam sak sbk sbn : Nat) (t : BlockState) :
    stepStmtsR R (matmulPrologue A B M N BM BN BK GM sam sak sbk sbn) t
      = stepStmts (matmulPrologue A B M N BM BN BK GM sam sak sbk sbn) t := by
  simp only [matmulPrologue, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

set_option maxHeartbeats 1000000 in
/-- The K-loop body is cast-free: the masked loads' `mask=`/`other=` operands
are nat comparisons and a real constant (no `castFloat`), the fused
`tl.dot`+add and the pointer advances are exact ops, and the body has no
store — so it steps identically under `stepStmtsR R` and the exact invariant
stack transports to `execR`. -/
private theorem matmulLoopBody_castFree (R : RoundingModel)
    (K BM BN BK sak sbk : Nat) (t : BlockState) :
    stepStmtsR R (matmulLoopBody K BM BN BK sak sbk) t
      = stepStmts (matmulLoopBody K BM BN BK sak sbk) t := by
  simp only [matmulLoopBody, aMaskOp, bMaskOp, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-! ### Cast-free op collapses and the two rounding-event sites -/

/-- The `offs_c*` index-vector op is cast-free. -/
private theorem evalR_offsc (R : RoundingModel) (Mv BMc : Nat) (pidReg : RegName)
    (s : BlockState) :
    evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat BMc))
        (Op.arange Mv)) s
      = evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat BMc))
        (Op.arange Mv)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- `offs_cm` / `offs_cn` eval under `R` (via the cast-free collapse). -/
private theorem offscm_evalR (R : RoundingModel) (s : BlockState) (Mv BMc : Nat)
    (pm : Nat) (pidReg : RegName)
    (hpm : s.regs .nat [] pidReg = some (Tile.scalar pm)) :
    evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] pidReg) (Op.constNat BMc))
        (Op.arange Mv)) s
      = some (Tile.vec (fun i : Fin Mv => pm * BMc + i.val)) := by
  rw [evalR_offsc]
  exact offscm_eval s Mv BMc pm pidReg hpm

/-- The `c_ptrs` pointer op (general `stride_cm`/`stride_cn`) is cast-free. -/
private theorem evalR_cptrs (R : RoundingModel) (Creg : RegionName)
    (BMv BNv scm scn : Nat) (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Creg)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat scm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BMv] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat scn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BNv] "offs_cn"))))) s
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Creg)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarL (Op.constNat scm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BMv] "offs_cm")))
          (Op.mul .nat Broadcast.scalarL (Op.constNat scn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BNv] "offs_cn"))))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `c_mask` boolean op is cast-free. -/
private theorem evalR_cmask (R : RoundingModel) (Mv Nv BMv BNv : Nat) (s : BlockState) :
    evalOpR R (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BMv] "offs_cm")) (Op.constNat Mv))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BNv] "offs_cn")) (Op.constNat Nv))) s
      = evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BMv] "offs_cm")) (Op.constNat Mv))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BNv] "offs_cn")) (Op.constNat Nv))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- `c = tl.cast(accumulator, od)` eval under `R`: rounding-event site 1,
`R.cast` applied lane-wise to the accumulator tile. Proven by direct `eq_def`
unfolding (a `rw` through `evalOpR_castFloat` trips over the
`FloatDType.real.toTileDType` vs `TileDType.real` spelling). -/
private theorem castAcc_evalR (R : RoundingModel) (od : FloatDType) (Mv Nv : Nat)
    (s : BlockState) (zT : Tile .real [Mv, Nv])
    (hz : s.regs .real [Mv, Nv] "accumulator" = some zT) :
    evalOpR R (Op.castFloat FloatDType.real od (Op.ref .real [Mv, Nv] "accumulator")) s
      = some ⟨fun idx => RoundingModel.cast R FloatDType.real od (zT.data idx)⟩ := by
  have hz' : s.regs FloatDType.real.toTileDType [Mv, Nv] "accumulator" = some zT := hz
  simp only [evalOpR.eq_def, hz']
  rfl

/-! ### The tail under `execR R` -/

set_option maxHeartbeats 4000000 in
/-- **R-postLoop** (od-generic over the two epilogue arms): from the exact
invariant at `numKBlocks` blocks, the `execR R` store tail terminates and
writes, at every **active** output lane, the cell
`od(R.round od (matmulSpec …))` — the ideal GEMM value rounded **once** at the
arm's grid (`R.cast` site + `R.storeValue` site collapsed by `round_idem`);
inactive lanes' cells and every cell not hit by an active lane are untouched
(the masked-store frame). -/
private theorem matmul_postLoopR (R : RoundingModel) (od : FloatDType)
    (hod : od = FloatDType.fp16 ∨ od = FloatDType.f8e4)
    (A B C : RegionName) (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks : Nat)
    (hInj : Function.Injective (cOffset s0 M N BM BN GM scm scn))
    (st : BlockState)
    (hinv : matmulInvariant A B s0 M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks numKBlocks st) :
    ∃ sfin, stepStmtsR R (matmulStoreTail od C M N scm scn BM BN) st = some sfin
      ∧ (∀ idx : TileIndex [BM, BN],
          sfin.mem C (cOffset s0 M N BM BN GM scm scn idx)
            = if active s0 M N BM BN GM idx then
                MemCell.of od.toTileDType
                  (od.ofReal
                    (R.round od
                      (matmulSpec s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks idx.1 idx.2.1)))
              else st.mem C (cOffset s0 M N BM BN GM scm scn idx))
      ∧ (∀ r o, (r ≠ C ∨ ∀ idx : TileIndex [BM, BN],
            active s0 M N BM BN GM idx → o ≠ cOffset s0 M N BM BN GM scm scn idx) →
          sfin.mem r o = st.mem r o) := by
  simp only [matmulInvariant] at hinv
  obtain ⟨hpids, hcle, hpm, hpn, hz, hm, hn, hk, hap, hbp, hundef, hmem⟩ := hinv
  set zT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 numKBlocks)⟩
      with hzT
  set cT : Tile od.toTileDType [BM, BN] :=
    ⟨fun idx => RoundingModel.cast R FloatDType.real od (zT.data idx)⟩ with hcT
  set cpT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (C.cast, cOffset s0 M N BM BN GM scm scn idx)⟩ with hcpT
  set cmaskT : Tile .bool [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] =>
      (decide (rowGlobal s0 M N BM BN GM idx.1 < M) && decide (colGlobal s0 M N BM BN GM idx.2.1 < N))⟩
      with hcmaskT
  unfold matmulStoreTail
  -- c = cast(accumulator, od): rounding-event site 1 (`R.cast`)
  erw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (show evalOpR R (Op.castFloat FloatDType.real od
              (Op.ref .real [BM, BN] "accumulator")) st = some cT
          from castAcc_evalR R od BM BN st zT hz))]
  -- offs_cm / offs_cn (cast-free)
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (offscm_evalR R _ BM BM (pidM (s0.pids 0) M N BM BN GM) "pid_m" (by simp [hpm])))]
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (offscm_evalR R _ BN BN (pidN (s0.pids 0) M N BM BN GM) "pid_n" (by simp [hpn])))]
  -- c_ptrs (cast-free)
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (show evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase C)
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.mul .nat Broadcast.scalarL (Op.constNat scm) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")))
              (Op.mul .nat Broadcast.scalarL (Op.constNat scn) (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn"))))) _
          = some cpT from by
          rw [evalR_cptrs,
            cptrs_eval _ C BM BN scm scn (fun i => rowGlobal s0 M N BM BN GM i) (fun j => colGlobal s0 M N BM BN GM j)
              (by simp [rowGlobal]) (by simp [colGlobal])]
          rfl))]
  -- c_mask (cast-free)
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
        (show evalOpR R (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_cm")) (Op.constNat M))
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_cn")) (Op.constNat N))) _
          = some cmaskT from by
          rw [evalR_cmask,
            cmask_eval _ M N BM BN (fun i => rowGlobal s0 M N BM BN GM i) (fun j => colGlobal s0 M N BM BN GM j)
              (by simp [rowGlobal]) (by simp [colGlobal])]))]
  -- abstract the post-assign state
  generalize hst4 : (((((st.setReg "c" od.toTileDType [BM, BN] cT).setReg "offs_cm" .nat [BM]
        (Tile.vec fun i : Fin BM => pidM (s0.pids 0) M N BM BN GM * BM + i.val)).setReg "offs_cn" .nat [BN]
        (Tile.vec fun j : Fin BN => pidN (s0.pids 0) M N BM BN GM * BN + j.val)).setReg "c_ptrs" .ptr [BM, BN] cpT).setReg
        "c_mask" .bool [BM, BN] cmaskT) = st4
  have hc4 : st4.regs od.toTileDType [BM, BN] "c" = some cT := by rw [← hst4]; simp
  have hcp4 : st4.regs .ptr [BM, BN] "c_ptrs" = some cpT := by rw [← hst4]; simp
  have hcm4 : st4.regs .bool [BM, BN] "c_mask" = some cmaskT := by rw [← hst4]; simp
  have hmem4 : st4.mem = st.mem := by
    rw [← hst4]; funext region offset; simp only [BlockState.setReg_mem]
  -- the masked od store: rounding-event site 2 (`writeMemAsR`)
  have hstore : stepStmtR R (Stmt.store od.toTileDType [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "c_ptrs"))
        (Op.ref od.toTileDType [BM, BN] "c") (.mask (Op.ref .bool [BM, BN] "c_mask"))) st4
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc i =>
            if cmaskT.data i then
              acc.writeMemAsR R od C (cOffset s0 M N BM BN GM scm scn i) (cT.data i)
            else acc) st4) := by
    simp only [stepStmtR]
    rw [show evalOpR R (Op.ref od.toTileDType [BM, BN] "c") st4 = some cT from by rw [evalOpR_ref, hc4]]
    rw [show evalOpR R (Op.ref .bool [BM, BN] "c_mask") st4 = some cmaskT from by rw [evalOpR_ref, hcm4]]
    rw [show evalOpR R (Op.ref .ptr [BM, BN] "c_ptrs") st4 = some cpT from by rw [evalOpR_ref, hcp4]]
    simp only [bind, Option.map_some, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ _ (fun acc i _ => ?_))
    by_cases hmask : cmaskT.data i
    · simp only [hmask, if_true, hcpT, Region.cast_id]
      rw [writeMemTypedR_float R od hod]
    · simp only [hmask, Bool.false_eq_true, if_false]
  rw [stepStmtsR_cons_some hstore, stepStmtsR_nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · intro idx
    rw [BlockState.scatter_memcell_R_prop_masked_nd R od (region := C) st4
          (cOffset s0 M N BM BN GM scm scn) (fun i => cT.data i)
          (fun i => cmaskT.data i = Bool.true) hInj idx]
    by_cases hact : active s0 M N BM BN GM idx
    · have hmasktrue : cmaskT.data idx = Bool.true := by
        simp only [hcmaskT]
        obtain ⟨hr, hcc⟩ := hact
        simp [hr, hcc]
      rw [if_pos hmasktrue, if_pos hact]
      have hdata : cT.data idx = RoundingModel.cast R FloatDType.real od
          (some (accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 numKBlocks)) := rfl
      rw [hdata, RoundingModel.storeValue_cast]
      have hspec : matmulSpec s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks idx.1 idx.2.1
          = accPartial s0 A B M N BM BN GM sam sak sbk sbn BLOCK_K idx.1 idx.2.1 numKBlocks := by
        unfold matmulSpec accPartial
        rw [Nat.mul_comm]
      rw [hspec]
    · have hmaskfalse : ¬ (cmaskT.data idx = Bool.true) := by
        simp only [hcmaskT, Bool.and_eq_true, decide_eq_true_eq, not_and]
        intro hr hcc
        exact hact ⟨hr, hcc⟩
      rw [if_neg hmaskfalse, if_neg hact, hmem4]
  · intro r o hcond
    by_cases hr : r = C
    · subst hr
      have hno : ∀ idx : TileIndex [BM, BN],
          active s0 M N BM BN GM idx → o ≠ cOffset s0 M N BM BN GM scm scn idx := by
        rcases hcond with h | h
        · exact absurd rfl h
        · exact h
      rw [BlockState.foldl_writeMemAsR_preserve_masked_prop R od
            (cOffset s0 M N BM BN GM scm scn) (fun i => cT.data i)
            (fun i => cmaskT.data i = Bool.true) o (TileShape.allIndices [BM, BN])
            (fun k _ hk => by
              have hkact : active s0 M N BM BN GM k := by
                simp only [hcmaskT, Bool.and_eq_true, decide_eq_true_eq] at hk
                exact ⟨hk.1, hk.2⟩
              exact fun heq => hno k hkact heq.symm) st4, hmem4]
    · rw [BlockState.foldl_writeMemAsR_preserve_other_region R od
            (cOffset s0 M N BM BN GM scm scn) (fun i => cT.data i)
            (fun i => cmaskT.data i = Bool.true) r hr o (TileShape.allIndices [BM, BN]) st4,
          hmem4]

/-! ### Safety-walk invariant (weak shape half of `matmulInvariant`) -/

/-- Safety-walk loop invariant: the *shape* half of `matmulInvariant`
(`pid_m`/`pid_n` seeded from the L2 schedule, *some* accumulator tile,
`offs_k`, and the exact `a_ptrs`/`b_ptrs` address shapes) with no
`undef`/`mem`/value pins. Needed because the `⊨[R]` skin's `hts` obligation
quantifies over arbitrary launch states, so the safety walk cannot assume
the clean-`undef` precondition that `preLoop`'s full invariant needs.
`offs_k` is carried because the loop body's **masked** loads must evaluate
their `offs_k`-based masks to step at all. -/
private def matmulSafeInv (A B : RegionName) (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn BK T : Nat) (c : Nat) (s : BlockState) : Prop :=
  c ≤ T ∧
  (s.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s0.pids 0) M N BM BN GM))) ∧
  (s.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s0.pids 0) M N BM BN GM))) ∧
  (∃ zT : Tile .real [BM, BN], s.regs .real [BM, BN] "accumulator" = some zT) ∧
  (s.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val))) ∧
  (s.regs .ptr [BM, BK] "a_ptrs" = some ⟨fun idx : TileIndex [BM, BK] =>
      (A.cast, rowIndex s0 M N BM BN GM idx.1 * sam + idx.2.1.val * sak + c * BK * sak)⟩) ∧
  (s.regs .ptr [BK, BN] "b_ptrs" = some ⟨fun idx : TileIndex [BK, BN] =>
      (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BK * sbk)⟩)

set_option maxHeartbeats 1000000 in
/-- Weak `preLoop`: from an **arbitrary** state the prologue steps to a state
satisfying `matmulSafeInv … 0` (no clean-`undef` hypothesis; the value half
of `preLoop` is dropped). -/
private theorem matmul_preLoopW (A B : RegionName) (s : BlockState)
    (M N BM BN BK GM sam sak sbk sbn T : Nat) :
    ∃ s', stepStmts (matmulPrologue A B M N BM BN BK GM sam sak sbk sbn) s = some s'
      ∧ matmulSafeInv A B s M N BM BN GM sam sak sbk sbn BK T 0 s' := by
  obtain ⟨s18, h18, hpids, hpm, hpn, hm, hn, hk, huf, hmem⟩ := preLoop_scalars s M N BM BN BK GM
  rw [show matmulPrologue A B M N BM BN BK GM sam sak sbk sbn
      = [ Stmt.assign .nat [] "pid" (Op.programId 0),
          Stmt.assign .nat [] "num_pid_m"
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
              (Op.constNat BM)),
          Stmt.assign .nat [] "num_pid_n"
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
              (Op.constNat BN)),
          Stmt.assign .nat [] "num_pid_in_group"
            (Op.mul .nat Broadcast.nil (Op.constNat GM) (Op.ref .nat [] "num_pid_n")),
          Stmt.assign .nat [] "group_id"
            (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group")),
          Stmt.assign .nat [] "first_pid_m"
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM)),
          Stmt.assign .nat [] "group_size_m"
            ((Op.lt ComparableDType.nat Broadcast.nil
                (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
                (Op.constNat GM)).where
              (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m") (Op.ref .nat [] "first_pid_m"))
              (Op.constNat GM)),
          Stmt.assign .nat [] "pid_m"
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
              (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "group_size_m"))),
          Stmt.assign .nat [] "pid_n"
            (Op.floorDiv .nat Broadcast.nil
              (Op.mod .nat Broadcast.nil (Op.ref .nat [] "pid") (Op.ref .nat [] "num_pid_in_group"))
              (Op.ref .nat [] "group_size_m")),
          Stmt.assign .nat [] "start_m"
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)),
          Stmt.assign .nat [] "start_n"
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)),
          Stmt.assign .nat [BM] "offs_am"
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_m") (Op.arange BM)),
          Stmt.assign .nat [BN] "offs_bn"
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.arange BN)),
          Stmt.assign .nat [BM] "offs_am"
            ((Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BM] "offs_am") (Op.constNat M)).where
              (Op.ref .nat [BM] "offs_am") ((Op.constNat 0).broadcast [BM])),
          Stmt.assign .nat [BN] "offs_bn"
            ((Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BN] "offs_bn") (Op.constNat N)).where
              (Op.ref .nat [BN] "offs_bn") ((Op.constNat 0).broadcast [BN])),
          Stmt.assign .nat [BM] "offs_am" (Op.ref .nat [BM] "offs_am"),
          Stmt.assign .nat [BN] "offs_bn" (Op.ref .nat [BN] "offs_bn"),
          Stmt.assign .nat [BK] "offs_k" (Op.arange BK) ]
        ++ [ Stmt.assign .ptr [BM, BK] "a_ptrs"
              (Op.ptrAdd Broadcast.scalarL (Op.ptrBase A)
                (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                  (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_am")) (Op.constNat sam))
                  (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sak)))),
            Stmt.assign .ptr [BK, BN] "b_ptrs"
              (Op.ptrAdd Broadcast.scalarL (Op.ptrBase B)
                (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
                  (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offs_k")) (Op.constNat sbk))
                  (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_bn")) (Op.constNat sbn)))),
            Stmt.assign .real [BM, BN] "accumulator" (Op.full [BM, BN] (Op.const 0)) ] from rfl,
    stepStmts.append_some h18,
    stepStmts.cons_some (stepStmt_assign_eq_some
      (aptrs_eval s18 A BM BK sam sak (fun i => rowIndex s M N BM BN GM i) (by simpa using hm) (by simpa using hk))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (bptrs_eval _ B BK BN sbk sbn (fun j => colIndex s M N BM BN GM j) (by simp [hk]) (by simp [hn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval _ BM BN)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨Nat.zero_le T, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hpm]
  · simp [hpn]
  · refine ⟨⟨fun _ => some (0 : ℝ)⟩, ?_⟩
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp [hk]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp [Nat.zero_mul]

set_option maxHeartbeats 2000000 in
/-- Weak step lemma: one body iteration from `matmulSafeInv c` steps
successfully (exact stepper; the body is cast-free, and at exact-multiple
`K` the load masks evaluate all-true) and re-establishes the invariant at
`c + 1` — the shape half of `matmul_step`, valid from arbitrary launch
states. -/
private theorem matmul_stepW (A B : RegionName) (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn BK T K : Nat) (hK : K = BK * T)
    (c : Nat) (s : BlockState) (hclt : c < T)
    (hinv : matmulSafeInv A B s0 M N BM BN GM sam sak sbk sbn BK T c s) :
    ∃ s', stepStmts (matmulLoopBody K BM BN BK sak sbk)
        (s.setReg "kk" .nat [] (Tile.scalar c)) = some s'
      ∧ matmulSafeInv A B s0 M N BM BN GM sam sak sbk sbn BK T (c + 1) s' := by
  obtain ⟨hcle, hpm, hpn, ⟨zT, hz⟩, hk, hap, hbp⟩ := hinv
  set apT : Tile .ptr [BM, BK] :=
    ⟨fun idx : TileIndex [BM, BK] =>
      (A.cast, rowIndex s0 M N BM BN GM idx.1 * sam + idx.2.1.val * sak + c * BK * sak)⟩ with hapT
  set bpT : Tile .ptr [BK, BN] :=
    ⟨fun idx : TileIndex [BK, BN] =>
      (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BK * sbk)⟩ with hbpT
  set sk := s.setReg "kk" .nat [] (Tile.scalar c) with hsk
  have hapk : sk.regs .ptr [BM, BK] "a_ptrs" = some apT := by simp [hsk, hap, hapT]
  have hbpk : sk.regs .ptr [BK, BN] "b_ptrs" = some bpT := by simp [hsk, hbp, hbpT]
  have hzk : sk.regs .real [BM, BN] "accumulator" = some zT := by simp [hsk, hz]
  have hkk : sk.regs .nat [] "kk" = some (Tile.scalar c) := by simp [hsk]
  have hkofk : sk.regs .nat [BK] "offs_k" = some (Tile.vec (fun e : Fin BK => e.val)) := by
    simp [hsk, hk]
  set asub : Tile .real [BM, BK] :=
    ⟨fun idx => some (sk.readMem (apT.data idx).1 (apT.data idx).2)⟩ with hasub
  set bsub : Tile .real [BK, BN] :=
    ⟨fun idx => some (sk.readMem (bpT.data idx).1 (bpT.data idx).2)⟩ with hbsub
  unfold matmulLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_a_eval sk K BM BK T c apT hK hclt hapk hkofk hkk))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_b_eval _ K BK BN T c bpT hK hclt
          (by simp [hbpk]) (by simp [hkofk]) (by simp [hkk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (accdot_op_eval BM BK BN _ zT asub bsub
          (by simp [hzk, hasub, hbsub, BlockState.setReg_readMem])
          (by simp [hasub, BlockState.setReg_readMem])
          (by simp [hbsub, BlockState.setReg_readMem])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (aptr_adv_eval _ BM BK BK sak apT (by simp [hapk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (bptr_adv_eval _ BK BN BK sbk bpT (by simp [hbpk])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by omega, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [hsk, hpm]
  · simp [hsk, hpn]
  · refine ⟨Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.dot [] asub bsub) zT, ?_⟩
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp [hsk, hk]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hapT, NumericDType.add]
    ring
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hbpT, NumericDType.add]
    ring

set_option maxHeartbeats 1000000 in
/-- Per-iteration `TraceSafeListR` for the K-loop body: the two **masked**
loads' addresses are the invariant's pointer shapes, in bounds at *every*
lane by the all-lane bound groups (active lanes are a subset, so the masks
never need evaluating for safety); the `mask=`/`other=` operands are
register-only ops; the three remaining assigns are unconditionally safe. -/
private theorem matmul_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (A B : RegionName) (s0 : BlockState)
    (M N BM BN GM sam sak sbk sbn BK T K : Nat) (c : Nat) (hc : c < T)
    (sk : BlockState)
    (hap : sk.regs .ptr [BM, BK] "a_ptrs" = some ⟨fun idx : TileIndex [BM, BK] =>
        (A.cast, rowIndex s0 M N BM BN GM idx.1 * sam + idx.2.1.val * sak + c * BK * sak)⟩)
    (hbp : sk.regs .ptr [BK, BN] "b_ptrs" = some ⟨fun idx : TileIndex [BK, BN] =>
        (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BK * sbk)⟩)
    (hbA : ∀ (t : Fin T) (j : Fin (BM * BK)),
      clampIdx (pidM (s0.pids 0) M N BM BN GM * BM + j.val / BK) M * sam
        + (t.val * BK + j.val % BK) * sak < bounds A)
    (hbB : ∀ (t : Fin T) (j : Fin (BK * BN)),
      (t.val * BK + j.val / BN) * sbk
        + clampIdx (pidN (s0.pids 0) M N BM BN GM * BN + j.val % BN) N * sbn < bounds B) :
    Stmt.TraceSafeListR R bounds (matmulLoopBody K BM BN BK sak sbk) sk := by
  unfold matmulLoopBody
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · -- load a (masked)
    simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, aMaskOp, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, by simp, ?_⟩
    intro ptrs hptrs i _
    rw [evalOpR_ref, hap] at hptrs
    obtain rfl := Option.some.inj hptrs
    show rowIndex s0 M N BM BN GM i.1 * sam + i.2.1.val * sak + c * BK * sak
        < bounds (Region.cast A)
    have h' := hbA ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
    rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
    simp only [Region.cast_id]
    calc rowIndex s0 M N BM BN GM i.1 * sam + i.2.1.val * sak + c * BK * sak
        = clampIdx (pidM (s0.pids 0) M N BM BN GM * BM + i.1.val) M * sam
            + (c * BK + i.2.1.val) * sak := by
          unfold rowIndex rowGlobal; ring
      _ < bounds A := h'
  · intro s1 h1
    obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- load b (masked)
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, bMaskOp, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
      refine ⟨trivial, by simp, ?_⟩
      intro ptrs hptrs i _
      rw [evalOpR_ref] at hptrs
      rw [show (sk.setReg "a" .real [BM, BK] v1).regs .ptr [BK, BN] "b_ptrs"
          = some (⟨fun idx : TileIndex [BK, BN] =>
            (B.cast, idx.1.val * sbk + colIndex s0 M N BM BN GM idx.2.1 * sbn + c * BK * sbk)⟩ :
              Tile .ptr [BK, BN]) from by simp [hbp]] at hptrs
      obtain rfl := Option.some.inj hptrs
      show i.1.val * sbk + colIndex s0 M N BM BN GM i.2.1 * sbn + c * BK * sbk
          < bounds (Region.cast B)
      have h' := hbB ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
      rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
      simp only [Region.cast_id]
      calc i.1.val * sbk + colIndex s0 M N BM BN GM i.2.1 * sbn + c * BK * sbk
          = (c * BK + i.1.val) * sbk
              + clampIdx (pidN (s0.pids 0) M N BM BN GM * BN + i.2.1.val) N * sbn := by
            unfold colIndex colGlobal; ring
        _ < bounds B := h'
    · intro s2 h2
      obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv h2
      refine Stmt.TraceSafeListR.of_forall _ _ ?_
      intro stx hstx s'
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hstx
      rcases hstx with rfl | rfl | rfl <;>
        simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]

set_option maxHeartbeats 1000000 in
/-- `TraceSafeListR` for the store tail (od-generic): the five assigns are
register-only (the cast is not a memory event) and the single **masked**
`od`-typed store's active lanes are exactly the `(row<M)&(col<N)` window, so
their addresses are the skin's `writeMask`-gated `write` bounds. -/
private theorem matmul_tailSafeR (R : RoundingModel) (bounds : RegionBounds)
    (od : FloatDType) (C : RegionName) (s0 : BlockState)
    (M N scm scn BM BN GM : Nat) (st : BlockState)
    (hpm : st.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s0.pids 0) M N BM BN GM)))
    (hpn : st.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s0.pids 0) M N BM BN GM)))
    (hzE : ∃ zT : Tile .real [BM, BN], st.regs .real [BM, BN] "accumulator" = some zT)
    (hbC : ∀ j : Fin (BM * BN),
      (pidM (s0.pids 0) M N BM BN GM * BM + j.val / BN < M ∧
        pidN (s0.pids 0) M N BM BN GM * BN + j.val % BN < N) →
      scm * (pidM (s0.pids 0) M N BM BN GM * BM + j.val / BN)
        + scn * (pidN (s0.pids 0) M N BM BN GM * BN + j.val % BN) < bounds C) :
    Stmt.TraceSafeListR R bounds (matmulStoreTail od C M N scm scn BM BN) st := by
  obtain ⟨zT, hz⟩ := hzE
  unfold matmulStoreTail
  -- 1. c = cast(accumulator, od)
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s1 h1
  obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
  -- 2. offs_cm
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s2 h2
  obtain ⟨v2, hv2, rfl⟩ := stepStmtR_assign_inv h2
  rw [offscm_evalR R _ BM BM (pidM (s0.pids 0) M N BM BN GM) "pid_m" (by simp [hpm])] at hv2
  obtain rfl := Option.some.inj hv2
  -- 3. offs_cn
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s3 h3
  obtain ⟨v3, hv3, rfl⟩ := stepStmtR_assign_inv h3
  rw [offscm_evalR R _ BN BN (pidN (s0.pids 0) M N BM BN GM) "pid_n" (by simp [hpn])] at hv3
  obtain rfl := Option.some.inj hv3
  -- 4. c_ptrs
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s4 h4
  obtain ⟨v4, hv4, rfl⟩ := stepStmtR_assign_inv h4
  rw [evalR_cptrs,
    cptrs_eval _ C BM BN scm scn
      (fun i => pidM (s0.pids 0) M N BM BN GM * BM + i.val)
      (fun j => pidN (s0.pids 0) M N BM BN GM * BN + j.val) (by simp) (by simp)] at hv4
  obtain rfl := Option.some.inj hv4
  -- 5. c_mask
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s5 h5
  obtain ⟨v5, hv5, rfl⟩ := stepStmtR_assign_inv h5
  rw [evalR_cmask,
    cmask_eval _ M N BM BN
      (fun i => pidM (s0.pids 0) M N BM BN GM * BM + i.val)
      (fun j => pidN (s0.pids 0) M N BM BN GM * BN + j.val) (by simp) (by simp)] at hv5
  obtain rfl := Option.some.inj hv5
  -- 6. the masked store
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s' _ => Stmt.TraceSafeListR.nil_intro)
  simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR.eq_def,
    MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
  refine ⟨trivial, trivial, trivial, ?_⟩
  intro ptrs hptrs i hactive
  rw [evalOpR_ref] at hptrs
  simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
    not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff] at hptrs
  obtain rfl := Option.some.inj hptrs
  obtain ⟨masks, hmasks, hmi⟩ := hactive
  rw [evalOpR_ref] at hmasks
  simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
    not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff] at hmasks
  obtain rfl := Option.some.inj hmasks
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hmi
  show scm * (pidM (s0.pids 0) M N BM BN GM * BM + i.1.val)
      + scn * (pidN (s0.pids 0) M N BM BN GM * BN + i.2.1.val) < bounds (Region.cast C)
  have h' := hbC (Lane2D.encode (i.1, i.2.1, PUnit.unit))
    (by rw [Lane2D.encode_div, Lane2D.encode_mod]; exact hmi)
  rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
  simpa only [Region.cast_id] using h'

set_option maxHeartbeats 1000000 in
/-- **The `TraceSafeR` walk for the whole lowered body** (od-generic), driven
by `Stmt.forRangeTraceSafeR_inv` over the weak `matmulSafeInv`. The three
bound groups are the skin's `read1`/`read2` windows (widened to all lanes via
`hK`, since the load masks are all-true at exact-multiple `K`) and the
`writeMask`-gated `write` window. -/
private theorem matmul_traceSafeR_body (R : RoundingModel) (bounds : RegionBounds)
    (od : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat)
    (hK : K = BK * numKBlocks) (s : BlockState)
    (hbA : ∀ (t : Fin numKBlocks) (j : Fin (BM * BK)),
      clampIdx (pidM (s.pids 0) M N BM BN GM * BM + j.val / BK) M * sam
        + (t.val * BK + j.val % BK) * sak < bounds A)
    (hbB : ∀ (t : Fin numKBlocks) (j : Fin (BK * BN)),
      (t.val * BK + j.val / BN) * sbk
        + clampIdx (pidN (s.pids 0) M N BM BN GM * BN + j.val % BN) N * sbn < bounds B)
    (hbC : ∀ j : Fin (BM * BN),
      (pidM (s.pids 0) M N BM BN GM * BM + j.val / BN < M ∧
        pidN (s.pids 0) M N BM BN GM * BN + j.val % BN < N) →
      scm * (pidM (s.pids 0) M N BM BN GM * BM + j.val / BN)
        + scn * (pidN (s.pids 0) M N BM BN GM * BN + j.val % BN) < bounds C) :
    Stmt.TraceSafeListR R bounds
      (matmulBody od A B C M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks) s := by
  unfold matmulBody
  have hstep : ∀ c s', c < numKBlocks →
      matmulSafeInv A B s M N BM BN GM sam sak sbk sbn BK numKBlocks c s' →
      Stmt.TraceSafeListR R bounds (matmulLoopBody K BM BN BK sak sbk)
        (s'.setReg "kk" .nat [] (Tile.scalar c)) ∧
      ∃ s'', stepStmtsR R (matmulLoopBody K BM BN BK sak sbk)
          (s'.setReg "kk" .nat [] (Tile.scalar c)) = some s'' ∧
        matmulSafeInv A B s M N BM BN GM sam sak sbk sbn BK numKBlocks (c + 1) s'' := by
    intro c s' hcx hP
    obtain ⟨hcle, hpm, hpn, hzE, hkx, hapx, hbpx⟩ := hP
    refine ⟨matmul_bodySafeR R bounds A B s M N BM BN GM sam sak sbk sbn BK
        numKBlocks K c hcx _ (by simp [hapx]) (by simp [hbpx]) hbA hbB, ?_⟩
    obtain ⟨s'', hs'', hP''⟩ := matmul_stepW A B s M N BM BN GM sam sak sbk sbn BK
      numKBlocks K hK c s' hcx ⟨hcle, hpm, hpn, hzE, hkx, hapx, hbpx⟩
    exact ⟨s'', by rw [matmulLoopBody_castFree]; exact hs'', hP''⟩
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- prologue: register-only assigns, safe at every state
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro st hst s'
    simp only [matmulPrologue, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    obtain ⟨s1x, hpre, hP0⟩ :=
      matmul_preLoopW A B s M N BM BN BK GM sam sak sbk sbn numKBlocks
    rw [matmulPrologue_castFree R A B M N BM BN BK GM sam sak sbk sbn s, hpre] at hs1
    obtain rfl := Option.some.inj hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- the K-loop is trace-safe (invariant principle over the weak invariant)
      simp only [Stmt.TraceSafeR]
      exact Stmt.forRangeTraceSafeR_inv R bounds "kk" numKBlocks 1
        (matmulLoopBody K BM BN BK sak sbk)
        (matmulSafeInv A B s M N BM BN GM sam sak sbk sbn BK numKBlocks)
        hstep 0 s1x hP0
    · intro s2 hs2
      obtain ⟨final, sLoop, hLoopStmt, hfinal, hPL⟩ :=
        forRange_inv (idx := "kk") (start := 0) (stop := numKBlocks) (step := 1)
          (body := matmulLoopBody K BM BN BK sak sbk)
          (by omega) hP0
          (fun c st hlt hinv => matmul_stepW A B s M N BM BN GM sam sak sbk sbn BK
            numKBlocks K hK c st hlt hinv)
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _ (matmulLoopBody_castFree R K BM BN BK sak sbk) "kk",
        ← stepForRangeAux.forRange_unfold, hLoopStmt] at hs2
      obtain rfl := Option.some.inj hs2
      obtain ⟨-, hpmL, hpnL, hzL, -, -, -⟩ := hPL
      exact matmul_tailSafeR R bounds od C s M N scm scn BM BN GM sLoop hpmL hpnL hzL hbC

/-- The lowered body (masked loads, `tl.where` clamps, `od` cast, masked
store) sits inside the flat-memory bridge's covered fragment (`FlattenOk`),
for either epilogue arm. -/
private theorem matmul_flattenOk_body (od : FloatDType) (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) :
    StmtList.FlattenOk
      (matmulBody od A B C M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks) := by
  simp [matmulBody, matmulPrologue, matmulLoopBody, matmulStoreTail, aMaskOp, bMaskOp,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### IO signature, lane bridges, spec bridge -/

/-- The `A`-stream lane feeding output lane `l` at inner key `e`: the row of
`l` (row-major over the `[BM, BN]` output tile) paired with `e` over the
`[BM, BLOCK_K]` per-step `A`-tile, both via the shared `Lane2D` bridge. -/
def aLane (BM BN BK : Nat) (l : Fin (BM * BN)) (e : Fin BK) : Fin (BM * BK) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)

/-- The `B`-stream lane feeding output lane `l` at inner key `e`: `e` paired
with the column of `l` over the `[BLOCK_K, BN]` per-step `B`-tile. -/
def bLane (BM BN BK : Nat) (l : Fin (BM * BN)) (e : Fin BK) : Fin (BK * BN) :=
  Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit)

/-- Under `K = BK · numKBlocks`, every in-tile inner key is inside the
kernel's `offs_k < K - kk·BK` load window: `t·BK + e < K` for `e < BK`. -/
private theorem stream_mask_lt (K BK numKBlocks : Nat) (hK : K = BK * numKBlocks)
    (t : Fin numKBlocks) {e : Nat} (he : e < BK) : t.val * BK + e < K := by
  have h1 : (t.val + 1) * BK ≤ numKBlocks * BK :=
    Nat.mul_le_mul_right BK (Nat.succ_le_of_lt t.isLt)
  have h2 : t.val * BK + e < (t.val + 1) * BK := by
    rw [Nat.succ_mul]; omega
  calc t.val * BK + e < (t.val + 1) * BK := h2
    _ ≤ numKBlocks * BK := h1
    _ = BK * numKBlocks := Nat.mul_comm _ _
    _ = K := hK.symm

/-- Under the two stream pins, `matmulSpec` at the decoded output lane **is**
the skin-level double fold `∑ t, ∑ e, xs · ys` (`gemmSum_blocks` +
address-identity of the windows with the invariant's pointer shapes; the
pins' `mask1`/`mask2` guards are discharged by `stream_mask_lt` from `hK`). -/
private theorem matmulSpec_eq_streamSum (A B : RegionName) (s₀ : BlockState)
    (M N BM BN GM sam sak sbk sbn BK numKBlocks K : Nat)
    (hK : K = BK * numKBlocks)
    (xs : Fin numKBlocks → Fin (BM * BK) → ℝ) (ys : Fin numKBlocks → Fin (BK * BN) → ℝ)
    (hx : ∀ (t : Fin numKBlocks) (j : Fin (BM * BK)), t.val * BK + j.val % BK < K →
      s₀.readMem A (clampIdx (pidM (s₀.pids 0) M N BM BN GM * BM + j.val / BK) M * sam
          + (t.val * BK + j.val % BK) * sak)
        = xs t j)
    (hy : ∀ (t : Fin numKBlocks) (j : Fin (BK * BN)), t.val * BK + j.val / BN < K →
      s₀.readMem B ((t.val * BK + j.val / BN) * sbk
          + clampIdx (pidN (s₀.pids 0) M N BM BN GM * BN + j.val % BN) N * sbn)
        = ys t j)
    (l : Fin (BM * BN)) :
    matmulSpec s₀ A B M N BM BN GM sam sak sbk sbn BK numKBlocks
        (Lane2D.decode l).1 (Lane2D.decode l).2.1
      = ∑ t : Fin numKBlocks, ∑ e : Fin BK,
          xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e) := by
  unfold matmulSpec
  rw [Nat.mul_comm BK numKBlocks, gemmSum_blocks]
  refine Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun e _ => ?_
  have hxa : aElem s₀ A M N BM BN GM sam sak (Lane2D.decode l).1 (t.val * BK + e.val)
      = xs t (aLane BM BN BK l e) := by
    rw [← hx t (aLane BM BN BK l e)
        (by simpa [aLane, Lane2D.encode_mod] using stream_mask_lt K BK numKBlocks hK t e.isLt)]
    simp only [aLane, Lane2D.encode_div, Lane2D.encode_mod]
    rfl
  have hyb : bElem s₀ B M N BM BN GM sbk sbn (Lane2D.decode l).2.1 (t.val * BK + e.val)
      = ys t (bLane BM BN BK l e) := by
    rw [← hy t (bLane BM BN BK l e)
        (by simpa [bLane, Lane2D.encode_div] using stream_mask_lt K BK numKBlocks hK t e.isLt)]
    simp only [bLane, Lane2D.encode_div, Lane2D.encode_mod]
    rfl
  rw [hxa, hyb]

/-- Tag-exact readback of a stored float cell through `readMemAs od`, at the
two epilogue tags (the od-generic sibling of the library's
`readMemAs_fp16_of_cell`): the `storeValue ∘ ofReal` round trip is the
identity on a defined real. -/
private theorem readMemAs_float_of_cell (od : FloatDType)
    (hod : od = FloatDType.fp16 ∨ od = FloatDType.f8e4)
    {s : BlockState} {region : RegionName} {offset : Nat} {x : ℝ}
    (h : s.mem region offset = MemCell.of od.toTileDType (od.ofReal x)) :
    s.readMemAs od region offset = od.ofReal x := by
  rcases hod with rfl | rfl <;>
    simp [BlockState.readMemAs, h, FloatDType.storeValue, FloatDType.ofReal]

set_option maxHeartbeats 4000000 in
/-- **The rounded region-model Hoare triple** (od-generic over the two
epilogue arms, at unit column stride): termination under the R-stepper on the
lowered body, per-active-lane `readMemAs od` readback of the ideal GEMM fold
quantized once at `od`, and the single-output frame. -/
private theorem matmul_runR (R : RoundingModel) (od : FloatDType)
    (hod : od = FloatDType.fp16 ∨ od = FloatDType.f8e4)
    (A B C : RegionName)
    (M N K sam sak sbk sbn scm BM BN BK GM numKBlocks : Nat)
    (hK : K = BK * numKBlocks) (hBN : BN ≤ scm)
    (s₀ : BlockState)
    (xs : Fin numKBlocks → Fin (BM * BK) → ℝ) (ys : Fin numKBlocks → Fin (BK * BN) → ℝ)
    (hundef : s₀.undef = fun _ _ => 0)
    (hx : ∀ (t : Fin numKBlocks) (j : Fin (BM * BK)), t.val * BK + j.val % BK < K →
      s₀.readMem A (clampIdx (pidM (s₀.pids 0) M N BM BN GM * BM + j.val / BK) M * sam
          + (t.val * BK + j.val % BK) * sak) = xs t j)
    (hy : ∀ (t : Fin numKBlocks) (j : Fin (BK * BN)), t.val * BK + j.val / BN < K →
      s₀.readMem B ((t.val * BK + j.val / BN) * sbk
          + clampIdx (pidN (s₀.pids 0) M N BM BN GM * BN + j.val % BN) N * sbn) = ys t j) :
    ∃ sfin, stepStmtsR R
        (matmulBody od A B C M N K sam sak sbk sbn scm 1 BM BN BK GM numKBlocks) s₀
        = some sfin
      ∧ (∀ l : Fin (BM * BN),
          (pidM (s₀.pids 0) M N BM BN GM * BM + l.val / BN < M ∧
            pidN (s₀.pids 0) M N BM BN GM * BN + l.val % BN < N) →
          sfin.readMemAs od C
              (scm * (pidM (s₀.pids 0) M N BM BN GM * BM + l.val / BN)
                + 1 * (pidN (s₀.pids 0) M N BM BN GM * BN + l.val % BN))
            = od.ofReal (R.round od
                (∑ t : Fin numKBlocks, ∑ e : Fin BK,
                  xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e))))
      ∧ (∀ r o,
          (r ≠ C ∨ ∀ l : Fin (BM * BN),
            (pidM (s₀.pids 0) M N BM BN GM * BM + l.val / BN < M ∧
              pidN (s₀.pids 0) M N BM BN GM * BN + l.val % BN < N) →
            o ≠ scm * (pidM (s₀.pids 0) M N BM BN GM * BM + l.val / BN)
              + 1 * (pidN (s₀.pids 0) M N BM BN GM * BN + l.val % BN)) →
          sfin.mem r o = s₀.mem r o) := by
  have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hundef]
  have hInj : Function.Injective (cOffset s₀ M N BM BN GM scm 1) := by
    have heq : cOffset s₀ M N BM BN GM scm 1
        = fun idx : TileIndex [BM, BN] =>
            (scm * (pidM (s₀.pids 0) M N BM BN GM * BM) + pidN (s₀.pids 0) M N BM BN GM * BN)
              + idx.1.val * scm + idx.2.1.val := by
      funext idx; simp only [cOffset, rowGlobal, colGlobal]; ring
    rw [heq]; exact rowMajor2D_inj _ scm hBN
  -- exact preLoop + K-loop (cast-free, so they are the `execR` run too)
  obtain ⟨s1, hpre, hP0⟩ := preLoop A B s₀ M N BM BN BK sam sak sbk sbn GM numKBlocks hundef'
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "kk") (start := 0) (stop := numKBlocks) (step := 1)
      (by omega) hP0
      (fun c stx hlt hinv => by
        simpa using matmul_step A B s₀ M N BM BN GM sam sak sbk sbn BK numKBlocks K hK
          c stx hlt hinv)
  have hfinalEq : final = numKBlocks := by
    have hle : final ≤ numKBlocks := by
      simp only [matmulInvariant] at hPLoop
      exact hPLoop.2.1
    exact le_antisymm hle hfinal
  rw [hfinalEq] at hPLoop
  have hmem0 : sLoop.mem = s₀.mem := by
    have h := hPLoop
    simp only [matmulInvariant] at h
    exact h.2.2.2.2.2.2.2.2.2.2.2
  -- R-side store tail
  obtain ⟨sfin, hTailR, hval, hframe⟩ :=
    matmul_postLoopR R od hod A B C s₀ M N BM BN GM sam sak sbk sbn scm 1 BK
      numKBlocks hInj sLoop hPLoop
  have hLoopR : stepStmtR R (Stmt.forRange "kk" 0 numKBlocks 1
        (matmulLoopBody K BM BN BK sak sbk)) s1
      = some sLoop := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (matmulLoopBody_castFree R K BM BN BK sak sbk) "kk",
      ← stepForRangeAux.forRange_unfold]
    exact hLoopStmt
  refine ⟨sfin, ?_, ?_, ?_⟩
  · unfold matmulBody
    rw [stepStmtsR_append,
      matmulPrologue_castFree R A B M N BM BN BK GM sam sak sbk sbn s₀, hpre,
      Option.bind_some, stepStmtsR_cons_some hLoopR]
    exact hTailR
  · intro l hj
    have hact : active s₀ M N BM BN GM (Lane2D.decode l) := hj
    have hcell := hval (Lane2D.decode l)
    rw [if_pos hact] at hcell
    have haddr : scm * (pidM (s₀.pids 0) M N BM BN GM * BM + l.val / BN)
          + 1 * (pidN (s₀.pids 0) M N BM BN GM * BN + l.val % BN)
        = cOffset s₀ M N BM BN GM scm 1 (Lane2D.decode l) := rfl
    rw [haddr, readMemAs_float_of_cell od hod hcell,
      matmulSpec_eq_streamSum A B s₀ M N BM BN GM sam sak sbk sbn BK numKBlocks K
        hK xs ys hx hy l]
  · intro r o hcond
    have hcond' : r ≠ C ∨ ∀ idx : TileIndex [BM, BN],
        active s₀ M N BM BN GM idx → o ≠ cOffset s₀ M N BM BN GM scm 1 idx := by
      rcases hcond with hne | hno
      · exact Or.inl hne
      · refine Or.inr fun idx hidx => ?_
        have h := hno (Lane2D.encode idx)
          (by
            rw [Lane2D.encode_div, Lane2D.encode_mod]
            exact hidx)
        rw [Lane2D.encode_div, Lane2D.encode_mod] at h
        exact h
    rw [hframe r o hcond', hmem0]

/-! ### The io signatures and headlines -/

/-- **Streaming IO signature** of `triton_matmul`'s **fp16 epilogue arm** on
the two-stream fold skin (S1: fold + terminal store). Step `t` of the K-loop
reads the `[BM, BLOCK_K]` `A`-tile and the `[BLOCK_K, BN]` `B`-tile; after
the loop one `[BM, BN]` output tile is stored at the **fp16** grid
(`outDType := .fp16` — this arm's `.to(tl.float16)` + fp16 store). The kernel
schedules on a **single** linear `pid` (`program_id(0)`; the skin's `pid₁`
slot is unused), so every window derives `(pid_m, pid_n)` through the
transcribed L2-grouping arithmetic `pidM`/`pidN`:

* `read1` lane `l = (i, e)` (row-major over `[BM, BLOCK_K]`), step `t`:
  `clampIdx(pid_m·BM + i, M)·sam + (t·BK + e)·sak` — the invariant's `a_ptrs`
  cell after `t` advances, through the kernel's `tl.where` row clamp.
* `read2` lane `l = (e, j)` (row-major over `[BLOCK_K, BN]`), step `t`:
  `(t·BK + e)·sbk + clampIdx(pid_n·BN + j, N)·sbn` — the `b_ptrs` cell.
* `write` lane `l = (i, j)`: `scm·(pid_m·BM + i) + scn·(pid_n·BN + j)` — the
  kernel's un-clamped `c_ptrs` (= `cOffset` in pid form).
* `mask1`/`mask2` transcribe the loads' `offs_k < K - kk·BK` windows in the
  per-lane spelling `t·BK + e < K`; `writeMask` transcribes the store's
  `(row<M) & (col<N)` boundary mask verbatim. -/
def tritonMatmulF16IO (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) :
    StreamMasked2DKernelIO₂ where
  kernel := triton_matmul_f16_surface A B C M N K sam sak sbk sbn scm scn BM BN BK GM
    numKBlocks
  inp1 := A
  inp2 := B
  out := C
  T := numKBlocks
  B1 := BM * BK
  B2 := BK * BN
  C := BM * BN
  outDType := .fp16
  read1 := fun p₀ _ t l =>
    clampIdx (pidM p₀ M N BM BN GM * BM + l.val / BK) M * sam + (t.val * BK + l.val % BK) * sak
  read2 := fun p₀ _ t l =>
    (t.val * BK + l.val / BN) * sbk + clampIdx (pidN p₀ M N BM BN GM * BN + l.val % BN) N * sbn
  write := fun p₀ _ l =>
    scm * (pidM p₀ M N BM BN GM * BM + l.val / BN) + scn * (pidN p₀ M N BM BN GM * BN + l.val % BN)
  mask1 := fun _ _ t l => t.val * BK + l.val % BK < K
  mask2 := fun _ _ t l => t.val * BK + l.val / BN < K
  writeMask := fun p₀ _ l =>
    pidM p₀ M N BM BN GM * BM + l.val / BN < M ∧ pidN p₀ M N BM BN GM * BN + l.val % BN < N

/-- **Streaming IO signature** of `triton_matmul`'s **fp8 epilogue arm**:
identical windows to `tritonMatmulF16IO`, with the fp8 surface and the
`outDType := .f8e4` boundary grid (this arm's `.to(tl.float8e4nv)` + fp8
store). -/
def tritonMatmulF8IO (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) :
    StreamMasked2DKernelIO₂ where
  kernel := triton_matmul_f8_surface A B C M N K sam sak sbk sbn scm scn BM BN BK GM
    numKBlocks
  inp1 := A
  inp2 := B
  out := C
  T := numKBlocks
  B1 := BM * BK
  B2 := BK * BN
  C := BM * BN
  outDType := .f8e4
  read1 := fun p₀ _ t l =>
    clampIdx (pidM p₀ M N BM BN GM * BM + l.val / BK) M * sam + (t.val * BK + l.val % BK) * sak
  read2 := fun p₀ _ t l =>
    (t.val * BK + l.val / BN) * sbk + clampIdx (pidN p₀ M N BM BN GM * BN + l.val % BN) N * sbn
  write := fun p₀ _ l =>
    scm * (pidM p₀ M N BM BN GM * BM + l.val / BN) + scn * (pidN p₀ M N BM BN GM * BN + l.val % BN)
  mask1 := fun _ _ t l => t.val * BK + l.val % BK < K
  mask2 := fun _ _ t l => t.val * BK + l.val / BN < K
  writeMask := fun p₀ _ l =>
    pidM p₀ M N BM BN GM * BM + l.val / BN < M ∧ pidN p₀ M N BM BN GM * BN + l.val % BN < N

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming headline, fp16 epilogue arm (wave-5 S1 fold
genre).** For every rounding model `R`, the faithful fp16-arm surface
implements, on its `StreamMasked2DKernelIO₂` signature, the **ideal ℝ GEMM
fold** over the streamed tiles: output lane `l = (i, j)` holds
`∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](e,j)` — the spec `f` is exact real
arithmetic, and the single boundary quantization is carried by the skin's
readback contract (`readMemAs .fp16` holds `fp16.ofReal (R.round .fp16 (f …))`),
where the kernel's two rounding events (the `.to(tl.float16)` cast and the
fp16-typed masked store) collapse to one `R.round .fp16` by the defining
`round_idem`.

Layer map: the prologue (with the `tl.where` index clamps) and the whole
K-loop (masked loads, `other=0.0`) are cast-free, so under `execR R` they
collapse verbatim onto the exact stepper and the proven `preLoop` /
`matmul_step` / `forRange_inv` stack above is reused unchanged; only the
6-statement store tail is re-proved on the `R` side (`matmul_postLoopR`).

Every hypothesis is truth-forced:

* `hK : K = BK · numKBlocks` — the exact-multiple contraction presentation
  shared with the exact surface: it makes the trip count
  `cdiv(K, BLOCK_K) = numKBlocks` exact and the per-step
  `offs_k < K - kk·BK` load masks all-true, which is what lets the streamed
  pins cover every lane the `tl.dot` consumes (at non-multiple `K` the spec
  sum would over-count the tail block and the statement would be false).
* `hcn : scn = 1` and `hBN : BN ≤ scm` — output-offset injectivity
  (`rowMajor2D_inj`): the column stride is the unit stride and the
  column-block width fits the row stride, so distinct output lanes hit
  distinct addresses; with colliding lanes the per-lane readback would be
  last-writer-wins and the statement false. Both hold for every valid
  row-major tiling.

Relation to the exact surface: the exact headline
`triton_matmul_f16_closed_form_correct` (`Realizes_without_Rounding`) above
is retained unchanged; this `⊨[R]` face strictly generalizes its content — at
`R := .triv` the readback contract degenerates to the exact fp16-cast cell of
the same GEMM value. Both faces are kept per the rounding-as-default
doctrine. -/
specification triton_matmul_f16_io_correctness (R : RoundingModel)
    (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat)
    (hK : K = BK * numKBlocks) (hcn : scn = 1) (hBN : BN ≤ scm) :
    tritonMatmulF16IO A B C M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks
      ⊨[R] fun _ _ xs ys l =>
        ∑ t : Fin numKBlocks, ∑ e : Fin BK,
          xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e) := by
  subst hcn
  refine StreamMasked2DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · -- FlattenOk
    show ((triton_matmul_f16_surface A B C M N K sam sak sbk sbn scm 1 BM BN BK GM
        numKBlocks).toAlgKernel).FlattenOk
    unfold Kernel.FlattenOk
    rw [triton_matmul_f16_body_eq]
    exact matmul_flattenOk_body FloatDType.fp16 A B C M N K sam sak sbk sbn scm 1 BM BN BK GM
      numKBlocks
  · -- safety walk
    intro bounds s xs ys _hx _hy hbr1 hbr2 hbw
    simp only [tritonMatmulF16IO] at hbr1 hbr2 hbw ⊢
    show ((triton_matmul_f16_surface A B C M N K sam sak sbk sbn scm 1 BM BN BK GM
        numKBlocks).toAlgKernel).TraceSafeR R bounds s
    unfold Kernel.TraceSafeR
    rw [triton_matmul_f16_body_eq]
    refine matmul_traceSafeR_body R bounds FloatDType.fp16 A B C M N K sam sak sbk sbn scm 1
      BM BN BK GM numKBlocks hK s ?_ ?_ ?_
    · intro t j
      have hBK : 0 < BK := by
        rcases Nat.eq_zero_or_pos BK with h | h
        · exact absurd j.pos (by simp [h])
        · exact h
      exact hbr1 t j (stream_mask_lt K BK numKBlocks hK t (Nat.mod_lt _ hBK))
    · intro t j
      have hdiv : j.val / BN < BK :=
        Nat.div_lt_of_lt_mul (lt_of_lt_of_eq j.isLt (Nat.mul_comm BK BN))
      exact hbr2 t j (stream_mask_lt K BK numKBlocks hK t hdiv)
    · intro j hj
      exact hbw j hj
  · -- the rounded Hoare triple
    intro s₀ xs ys hundef hx hy
    simp only [tritonMatmulF16IO] at hx hy ⊢
    obtain ⟨sfin, hstep, hval, hframe⟩ := matmul_runR R FloatDType.fp16 (Or.inl rfl) A B C
      M N K sam sak sbk sbn scm BM BN BK GM numKBlocks hK hBN s₀ xs ys hundef hx hy
    refine ⟨sfin, ?_, hval, hframe⟩
    show execR R (triton_matmul_f16_surface A B C M N K sam sak sbk sbn scm 1 BM BN BK GM
        numKBlocks).toAlgKernel s₀ = some sfin
    unfold execR
    rw [triton_matmul_f16_body_eq]
    exact hstep

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming headline, fp8 epilogue arm** — the corpus's first
fp8 matmul face. Identical ideal-ℝ GEMM fold spec to
`triton_matmul_f16_io_correctness`; the boundary grid is `.f8e4`: the output
window reads back `.f8e4`-typed cells holding
`f8e4.ofReal (R.round .f8e4 (∑ t, ∑ e, A·B))`, the kernel's
`.to(tl.float8e4nv)` cast + fp8-typed masked store collapsed to one
`R.round .f8e4` by the defining `round_idem`. Hypotheses truth-forced exactly
as in the fp16 arm. -/
specification triton_matmul_f8_io_correctness (R : RoundingModel)
    (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat)
    (hK : K = BK * numKBlocks) (hcn : scn = 1) (hBN : BN ≤ scm) :
    tritonMatmulF8IO A B C M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks
      ⊨[R] fun _ _ xs ys l =>
        ∑ t : Fin numKBlocks, ∑ e : Fin BK,
          xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e) := by
  subst hcn
  refine StreamMasked2DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · -- FlattenOk
    show ((triton_matmul_f8_surface A B C M N K sam sak sbk sbn scm 1 BM BN BK GM
        numKBlocks).toAlgKernel).FlattenOk
    unfold Kernel.FlattenOk
    rw [triton_matmul_f8_body_eq]
    exact matmul_flattenOk_body FloatDType.f8e4 A B C M N K sam sak sbk sbn scm 1 BM BN BK GM
      numKBlocks
  · -- safety walk
    intro bounds s xs ys _hx _hy hbr1 hbr2 hbw
    simp only [tritonMatmulF8IO] at hbr1 hbr2 hbw ⊢
    show ((triton_matmul_f8_surface A B C M N K sam sak sbk sbn scm 1 BM BN BK GM
        numKBlocks).toAlgKernel).TraceSafeR R bounds s
    unfold Kernel.TraceSafeR
    rw [triton_matmul_f8_body_eq]
    refine matmul_traceSafeR_body R bounds FloatDType.f8e4 A B C M N K sam sak sbk sbn scm 1
      BM BN BK GM numKBlocks hK s ?_ ?_ ?_
    · intro t j
      have hBK : 0 < BK := by
        rcases Nat.eq_zero_or_pos BK with h | h
        · exact absurd j.pos (by simp [h])
        · exact h
      exact hbr1 t j (stream_mask_lt K BK numKBlocks hK t (Nat.mod_lt _ hBK))
    · intro t j
      have hdiv : j.val / BN < BK :=
        Nat.div_lt_of_lt_mul (lt_of_lt_of_eq j.isLt (Nat.mul_comm BK BN))
      exact hbr2 t j (stream_mask_lt K BK numKBlocks hK t hdiv)
    · intro j hj
      exact hbw j hj
  · -- the rounded Hoare triple
    intro s₀ xs ys hundef hx hy
    simp only [tritonMatmulF8IO] at hx hy ⊢
    obtain ⟨sfin, hstep, hval, hframe⟩ := matmul_runR R FloatDType.f8e4 (Or.inr rfl) A B C
      M N K sam sak sbk sbn scm BM BN BK GM numKBlocks hK hBN s₀ xs ys hundef hx hy
    refine ⟨sfin, ?_, hval, hframe⟩
    show execR R (triton_matmul_f8_surface A B C M N K sam sak sbk sbn scm 1 BM BN BK GM
        numKBlocks).toAlgKernel s₀ = some sfin
    unfold execR
    rw [triton_matmul_f8_body_eq]
    exact hstep

end VeriTile.Bench.TritonBenchG.TritonMatmul
