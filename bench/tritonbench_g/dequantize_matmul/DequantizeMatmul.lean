import VeriTile.Triton

/-!
# `dequantize_matmul` — strict per-kernel correctness

`dequantize_kernel` is the dequantization prologue of an int8 matmul: program
`(k_block_idx, n_block_idx)` loads a `[BLOCK_SIZE_K, BLOCK_SIZE_N]` tile of the
int8 weight `b`, loads the per-column scale `b_scale`, and stores the
dequantized `int_b * scale_b` into `fpb`, masked by
`(k·K + offs_k < K) & (n·N + offs_n < N)`. The downstream `torch.mm(a, fp_b)` is
host code, not part of this kernel.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`dequantize_kernel[grid](...)`, the 2-D grid
`(cdiv(K, BSK), cdiv(N, BSN))`, the strides, the subsequent `torch.mm`, and the
runtime composition of per-program writes) is the *trusted boundary*. Both
program ids (`k_block_idx`, `n_block_idx`) are universally quantified, so the
per-program statement covers every program of the grid.

## Proof architecture

```
dequantize_matmul_python_<BSN>x<BSK>_output_summary    ← TOP THEOREMS (4 shapes)
  ├─ (toAlgorithm? = Except.ok _)                       surface lowers to algorithm
  └─ dequantize_matmul_python_<BSN>x<BSK>_compute_correct
       └─ dequantize_kernel_compute_correct             ComputeCorrect over the store
            ├─ dequantize_kernel_correct                algorithm-layer readback per lane
            └─ dequantize_matmul_python_..._offset_injective   no-collision lemmas
```

The four shapes (128x128, 64x256, 32x256, 256x64) mirror the `@triton.autotune`
configs exercised by the Python test. The spec is the elementwise dequantize
`int_b * scale_b`, with the masked `other=0.0` default modeled on lanes that
fall outside `K`/`N`.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float. The quantization semantics
here are the **dequantization direction only**: the proof models `int_b *
scale_b` as a plain real product, where `int_b` is the value loaded from the
int8 weight region and `scale_b` the loaded float scale. There is no rounding or
int cast on the output side, so this kernel has no `to(tl.int8)` / `llrint`
blocker — `int_b` is taken as the loaded real value and the store is exact. The
masked `other=0.0` default is modeled. `@triton.autotune` is not modeled;
instead the four representative `(BLOCK_SIZE_N, BLOCK_SIZE_K)` configs are
instantiated as separate theorems.
-/

namespace VeriTile.Bench.TritonBenchG.DequantizeMatmul

open VeriTile.Triton

/-- Faithful 1:1 transcription of `dequantize_matmul.py`'s `dequantize_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_N: tl.constexpr` / `BLOCK_SIZE_K: tl.constexpr` → Lean
  `Nat` parameters. -/
def dequantize_kernel
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = tl.arange(0, $(BLOCK_SIZE_N))
  b_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_bk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_bn)
  fpb_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_fpbk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_fpbn)
  bs_offs = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]
  n_mask = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :] < $(N)
  mask = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None] < $(K)) & n_mask
  int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
  tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)
}

/-- The Python `dequantize_kernel` surface lowers to the algorithm layer,
including the 2D masked int8 load, column scale load, multiply, and masked
`fp_b` store. The bundled Python wrapper then calls `torch.mm`; this theorem is
only about the Triton dequantization kernel. -/
theorem dequantize_kernel_surface_toAlgorithm_supported
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ∃ alg, (dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
      stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K).toAlgorithm? = Except.ok alg := by
  simp [dequantize_kernel]

def bOffset (s : BlockState) (stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_bk +
    (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_bn

def fpbOffset (s : BlockState) (stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_fpbk +
    (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_fpbn

def nOffset (s : BlockState) (BLOCK_SIZE_N : Nat) (j : Fin BLOCK_SIZE_N) : Nat :=
  s.pids 1 * BLOCK_SIZE_N + j.val

def dequantizeActive (s : BlockState) (K N BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Prop :=
  s.pids 0 * BLOCK_SIZE_K + idx.1.val < K ∧
    s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N

instance dequantizeActiveDecidable
    (s : BlockState) (K N BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) :
    Decidable (dequantizeActive s K N BLOCK_SIZE_N BLOCK_SIZE_K idx) := by
  unfold dequantizeActive
  infer_instance

/-- Exact dequantized value written at active tile lane `idx`. -/
noncomputable def dequantizeSpec
    (s : BlockState) (b_ptr b_scale_ptr : RegionName)
    (stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : ℝ :=
  s.readMem b_ptr (bOffset s stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K idx) *
    s.readMem b_scale_ptr (nOffset s BLOCK_SIZE_N idx.2.1)

private noncomputable def dequantizeStoreBase
    (s : BlockState) (b_ptr b_scale_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    BlockState :=
  ((((((((((s.setReg "k_block_idx" .nat [] (Tile.scalar (s.pids 0))).setReg
    "n_block_idx" .nat [] (Tile.scalar (s.pids 1))).setReg
    "offs_k" .nat [BLOCK_SIZE_K] (Tile.vec fun i => i.val)).setReg
    "offs_n" .nat [BLOCK_SIZE_N] (Tile.vec fun i => i.val)).setReg
    "b_offs" .nat [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i => bOffset s stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K i }).setReg
    "fpb_offs" .nat [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i => fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K i }).setReg
    "bs_offs" .nat [1, BLOCK_SIZE_N]
      { data := fun i => nOffset s BLOCK_SIZE_N i.2.1 }).setReg
    "n_mask" .bool [1, BLOCK_SIZE_N]
      { data := fun i => decide (nOffset s BLOCK_SIZE_N i.2.1 < N) }).setReg
    "mask" .bool [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i => decide (dequantizeActive s K N BLOCK_SIZE_N BLOCK_SIZE_K i) }).setReg
    "int_b" .real [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i =>
          if dequantizeActive s K N BLOCK_SIZE_N BLOCK_SIZE_K i then
            some (s.readMem b_ptr (bOffset s stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K i))
          else some (0.0 : ℝ) }).setReg
    "scale_b" .real [1, BLOCK_SIZE_N]
      { data := fun i =>
          if nOffset s BLOCK_SIZE_N i.2.1 < N then
            some (s.readMem b_scale_ptr (nOffset s BLOCK_SIZE_N i.2.1))
          else some (0.0 : ℝ) }

/-- Algorithm-layer correctness for `dequantize_kernel`. -/
theorem dequantize_kernel_correct
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K))
    (hExec : exec (dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
          stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K) s = some s') :
    ∀ idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N],
      s'.readMem fpb_ptr
          (fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K idx)
        = if dequantizeActive s K N BLOCK_SIZE_N BLOCK_SIZE_K idx then
          dequantizeSpec s b_ptr b_scale_ptr stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K idx
        else
          s.readMem fpb_ptr
            (fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K idx) := by
  intro idx
  simp [exec, dequantize_kernel, stepStmts, stepStmt, evalOp.eq_def, Option.bind,
        Tile.bop, Tile.cop, Tile.expandDim,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  let offsetFn : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N] → Nat :=
    fun idx =>
      (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_fpbk +
        (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_fpbn
  let active : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_SIZE_K + idx.1.val < K ∧
        s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N
  let valueFn : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (Option.map₂ (fun x1 x2 => x1 * x2)
          (if active idx then
            some
              (s.readMem b_ptr
                ((s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_bk +
                  (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_bn))
          else some (0.0 : ℝ))
          (if s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N then
            some
              (s.readMem b_scale_ptr
                (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val))
          else some (0.0 : ℝ)))
  have hOffsetInj : Function.Injective
      offsetFn := by
    simpa [offsetFn, fpbOffset] using hOutInj
  have hscatter := BlockState.scatter_readback_prop_masked_nd
    (region := fpb_ptr)
    (s := dequantizeStoreBase s b_ptr b_scale_ptr K N stride_bk stride_bn
      stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K)
    (offsetFn := offsetFn) (valueFn := valueFn) (P := active)
    hOffsetInj idx
  by_cases hActive : active idx
  · rcases hActive with ⟨hk, hn⟩
    simpa [offsetFn, active, valueFn, dequantizeActive, dequantizeSpec,
      bOffset, fpbOffset, nOffset, hk, hn,
      dequantizeStoreBase, TileShape.dropInsertedIndex, Bool.and_eq_true] using hscatter
  · simpa [offsetFn, active, valueFn, dequantizeActive, dequantizeSpec,
      bOffset, fpbOffset, nOffset, hActive,
      dequantizeStoreBase, TileShape.dropInsertedIndex] using hscatter

/-- Compute-facing correctness for `dequantize_kernel`. -/
theorem dequantize_kernel_compute_correct
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (dequantizeActive s K N BLOCK_SIZE_N BLOCK_SIZE_K)
          (fun idx => (fpb_ptr,
            fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K idx)))
      (expected := fun idx =>
        dequantizeSpec s b_ptr b_scale_ptr stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := dequantize_kernel_correct b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
    stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K s s' hOutInj hExec idx
  simpa [hActive] using h

/-- **Dimension-general output summary for `dequantize_matmul.py`.**

For arbitrary matrix dims `K`/`N`, strides `stride_bk`/`stride_bn`/`stride_fpbk`/
`stride_fpbn`, and block sizes `BLOCK_SIZE_N`/`BLOCK_SIZE_K` (and any program ids
in `s`), under the honest row-major output-offset injectivity side condition the
`dequantize_kernel` surface lowers to the algorithm layer and the masked store
into `fp_b` realizes the genuine elementwise dequantize `int_b * scale_b`
(`dequantizeSpec`) on every active lane (`k·BSK + i < K ∧ n·BSN + j < N`),
unchanged otherwise. The four pinned
`dequantize_matmul_python_{128x128,64x256,32x256,256x64}_output_summary` (mirroring
the `@triton.autotune` configs) are concrete instantiations of this one theorem. -/
specification dequantize_matmul_output_summary_general
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K)) :
    (∃ alg, (dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
      stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (dequantizeActive s K N BLOCK_SIZE_N BLOCK_SIZE_K)
          (fun idx => (fpb_ptr,
            fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K idx)))
      (expected := fun idx =>
        dequantizeSpec s b_ptr b_scale_ptr stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K idx) :=
  ⟨dequantize_kernel_surface_toAlgorithm_supported b_ptr b_scale_ptr fpb_ptr
      K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K,
   dequantize_kernel_compute_correct b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
      stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K s hOutInj⟩

end VeriTile.Bench.TritonBenchG.DequantizeMatmul
