import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `f8_conversion_utils` — fp8 ↔ fp16 conversion kernels

Upstream: `data/TritonBench_G_v1/f8_conversion_utils.py` (68 lines, 2 jit
kernels). `kernel_f8_to_f16` copies a masked 1-D window from an fp8 buffer
into an fp16 buffer (storing **twice** — the upstream body repeats the
store verbatim); `kernel_f16_to_f8` is the reverse direction. The hosts
allocate `int8`/`float16` tensors and pass the fp8 side through
`triton.reinterpret(x, tl.float8e5)` — the fp8-ness of a buffer lives
entirely in the pointer's element dtype; the kernel bodies are plain masked
copies with **implicit** casts at the store.

This is the first consumer of the `.f8e5` float channel (the fp8 unlock
lever): the f16→f8 headline is the corpus's first fp8 boundary
quantization — the output window holds `.f8e5`-typed cells carrying
`R.round .f8e5 (xs j)`, stated for every rounding model `R`.

Translation-surface blocker: Triton's `tl.store` implicitly casts the
value to the destination pointer's element type (here fixed by the host's
`triton.reinterpret(..., tl.float8e5)` / `torch.float16` allocations); the
DSL types a store by its **value**, so the implicit cast is spelled
explicitly — `(x).to(tl.float16)` in `kernel_f8_to_f16` and
`(x).to(tl.float8e5)` in `kernel_f16_to_f8`. Python
`BLOCK_SIZE: tl.constexpr` becomes a Lean `Nat` parameter. Registered in
`proof_blockers.md`.

## Modeling boundary

Arithmetic is over `ℝ` with the abstract `RoundingModel` interface: the
store-side cast is the modeled quantization event (cast site + typed store
site collapse to **one** boundary round by `R.round_idem`). The *input*
window is consumed as exact ℝ (`⊨[R]`'s inputs-exact convention): that the
f8-side input buffer of `kernel_f8_to_f16` physically holds f8-grid points
is a property of the trusted host pipeline, not needed by the theorem. The
host launch (grid `cdiv(numel, BLOCK_SIZE)`, the `reinterpret` calls, the
`numel` bookkeeping) is the trusted boundary.

## Proof architecture (each kernel)

```
<kernel>_io_correctness                ← TOP SPECIFICATION (io ⊨[R, grid] identity)
  ├─ <kernel>_flattenOk                bridge fragment membership
  ├─ <kernel>_traceSafeR               per-execution lane-wise safety walk under R
  └─ <kernel>_region_runR              rounded region-model masked Hoare triple
```
-/

namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Faithful transcription of `f8_conversion_utils.py`'s `kernel_f8_to_f16`
(fp8 → fp16 masked copy, double store). The store's implicit fp16 cast is
spelled `(x).to(tl.float16)` — see the preamble blocker note. -/
def kernel_f8_to_f16 (Y X : RegionName) (N BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  offs = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, (x).to(tl.float16), mask=mask)
  tl.store(Y + offs, (x).to(tl.float16), mask=mask)
}

/-- Faithful transcription of `f8_conversion_utils.py`'s `kernel_f16_to_f8`
(fp16 → fp8 masked copy). The store's implicit fp8 cast is spelled
`(x).to(tl.float8e5)` — see the preamble blocker note. -/
def kernel_f16_to_f8 (Y X : RegionName) (N BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  offs = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, (x).to(tl.float8e5), mask=mask)
}

/-! ## Shared window fact -/

private theorem window_inj (pid B : Nat) :
    Function.Injective
      (fun idx : TileIndex [B] => pid * B + idx.1.val) := by
  rintro ⟨a, _⟩ ⟨b, _⟩ hab
  obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
  rfl

/-! ## `kernel_f16_to_f8` — the fp8 headline's stack -/

/-- Rounded region-model masked Hoare triple for `kernel_f16_to_f8`:
termination under `execR R`, per-active-lane `.f8e5`-typed readback of the
copied value quantized once, frame off the active output window. -/
private theorem f16_to_f8_region_runR (R : RoundingModel)
    (Y X : RegionName) (N B : Nat)
    (s₀ : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ j : Fin B, s₀.pid * B + j.val < N →
      s₀.readMem X (s₀.pid * B + j.val) = xs j) :
    ∃ s1, execR R ((kernel_f16_to_f8 Y X N B).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin B, s₀.pid * B + j.val < N →
          s1.readMemAs .f8e5 Y (s₀.pid * B + j.val)
            = FloatDType.f8e5.ofReal (R.round .f8e5 (xs j)))
      ∧ (∀ r o,
          (r ≠ Y ∨ ∀ j : Fin B, s₀.pid * B + j.val < N →
            o ≠ s₀.pid * B + j.val) →
          s1.mem r o = s₀.mem r o) := by
  simp only [BlockState.pid_eq] at hx
  have h_inj := window_inj (s₀.pids 0) B
  simp [execR, kernel_f16_to_f8, ComputeKernel.toAlgKernel,
    stepStmtsR, stepStmtR, evalOpR.eq_def, BlockState.writeMemTypedR,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt]
  refine ⟨fun j hj => ?_, fun r o hcond => ?_⟩
  · -- value: masked R-scatter readback; the cast/store rounding pair
    -- collapses by `round_idem`
    unfold BlockState.readMemAs
    rw [BlockState.scatter_memcell_R_prop_masked_nd R .f8e5 _ _ _ _ h_inj
      (j, PUnit.unit)]
    simp [hj, MemCell.readAs_of_same, FloatDType.ofReal, FloatDType.storeValue,
      RoundingModel.storeValue, RoundingModel.cast, hx j hj, R.round_idem]
  · -- frame: the masked scatter misses (r, o)
    by_cases hr : r = Y
    · subst hr
      rcases hcond with hne | hno
      · exact absurd rfl hne
      · exact BlockState.foldl_writeMemAsR_preserve_masked_prop R .f8e5 _ _ _
          o _ (fun k _ hPk => fun heq => hno k.1 (by simpa using hPk)
            (by simpa using heq.symm)) _
    · exact BlockState.foldl_writeMemAsR_preserve_other_region R .f8e5 _ _ _
        r hr o _ _

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk under `R` for `kernel_f16_to_f8`: the masked
load and masked store address the window `pid * B + j`, active only when
`< N`. -/
private theorem f16_to_f8_traceSafeR (R : RoundingModel)
    (Y X : RegionName) (N B : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin B, s.pid * B + j.val < N →
      s.pid * B + j.val < bounds X)
    (hout : ∀ j : Fin B, s.pid * B + j.val < N →
      s.pid * B + j.val < bounds Y) :
    Kernel.TraceSafeR R bounds
      ((kernel_f16_to_f8 Y X N B).toAlgKernel) s := by
  unfold Kernel.TraceSafeR
  simp only [BlockState.pid_eq] at hx hout
  simp [kernel_f16_to_f8, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.SafeAtR,
    stepStmtR, evalOpR.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MemAccess.SafeAtR, MaskOpt.ActiveR, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hout a ha⟩

/-- The kernel sits inside the bridge's covered fragment. -/
private theorem f16_to_f8_flattenOk (Y X : RegionName) (N B : Nat) :
    ((kernel_f16_to_f8 Y X N B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [kernel_f16_to_f8, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- `kernel_f16_to_f8`'s masked IO signature: 1-D window
`[pid·B, (pid+1)·B)` on both buffers, active lanes `pid·B + j < N` (the
kernel's shared load/store mask). -/
def f16ToF8IO (Y X : RegionName) (N BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := kernel_f16_to_f8 Y X N BLOCK_SIZE
  inp := X
  out := Y
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < N

/-- **The fp8 headline**: for every rounding model `R`, `kernel_f16_to_f8`
implements the masked identity copy on its IO signature at the `.f8e5`
output grid — every active output lane reads back as an `.f8e5`-typed cell
holding `R.round .f8e5 (xs j)`: the input value quantized **once** onto the
fp8 (e5m2) grid. The corpus's first fp8 boundary quantization. -/
specification f16_to_f8_io_correctness (R : RoundingModel)
    (Y X : RegionName) (N BLOCK_SIZE : Nat) :
    f16ToF8IO Y X N BLOCK_SIZE ⊨[R, .f8e5] fun xs i => xs i := by
  refine MaskedKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
  · exact f16_to_f8_flattenOk Y X N BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact f16_to_f8_traceSafeR R Y X N BLOCK_SIZE bounds s h1 h2
  · intro s₀ xs hx
    simp only [f16ToF8IO] at hx ⊢
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      f16_to_f8_region_runR R Y X N BLOCK_SIZE s₀ xs hx
    exact ⟨s1, hexec, hval, fun r o hcond _ => hframe r o hcond⟩

/-! ## `kernel_f8_to_f16` — the reverse direction (double store) -/

/-- Rounded region-model masked Hoare triple for `kernel_f8_to_f16`. The
body stores twice; the second (outermost) masked scatter determines every
active lane's final cell, and both scatters respect the frame. -/
private theorem f8_to_f16_region_runR (R : RoundingModel)
    (Y X : RegionName) (N B : Nat)
    (s₀ : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ j : Fin B, s₀.pid * B + j.val < N →
      s₀.readMem X (s₀.pid * B + j.val) = xs j) :
    ∃ s1, execR R ((kernel_f8_to_f16 Y X N B).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin B, s₀.pid * B + j.val < N →
          s1.readMemAs .fp16 Y (s₀.pid * B + j.val)
            = FloatDType.fp16.ofReal (R.round .fp16 (xs j)))
      ∧ (∀ r o,
          (r ≠ Y ∨ ∀ j : Fin B, s₀.pid * B + j.val < N →
            o ≠ s₀.pid * B + j.val) →
          s1.mem r o = s₀.mem r o) := by
  simp only [BlockState.pid_eq] at hx
  have h_inj := window_inj (s₀.pids 0) B
  simp [execR, kernel_f8_to_f16, ComputeKernel.toAlgKernel,
    stepStmtsR, stepStmtR, evalOpR.eq_def, BlockState.writeMemTypedR,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt]
  refine ⟨fun j hj => ?_, fun r o hcond => ?_⟩
  · -- value: readback of the outer (second) masked R-scatter
    unfold BlockState.readMemAs
    rw [BlockState.scatter_memcell_R_prop_masked_nd R .fp16 _ _ _ _ h_inj
      (j, PUnit.unit)]
    simp [hj, MemCell.readAs_of_same, FloatDType.ofReal, FloatDType.storeValue,
      RoundingModel.storeValue, RoundingModel.cast, hx j hj, R.round_idem]
  · -- frame: both masked scatters miss (r, o)
    by_cases hr : r = Y
    · subst hr
      rcases hcond with hne | hno
      · exact absurd rfl hne
      · rw [BlockState.foldl_writeMemAsR_preserve_masked_prop R .fp16 _ _ _
          o _ (fun k _ hPk => fun heq => hno k.1 (by simpa using hPk)
            (by simpa using heq.symm)) _]
        exact BlockState.foldl_writeMemAsR_preserve_masked_prop R .fp16 _ _ _
          o _ (fun k _ hPk => fun heq => hno k.1 (by simpa using hPk)
            (by simpa using heq.symm)) _
    · rw [BlockState.foldl_writeMemAsR_preserve_other_region R .fp16 _ _ _
        r hr o _ _]
      exact BlockState.foldl_writeMemAsR_preserve_other_region R .fp16 _ _ _
        r hr o _ _

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk under `R` for `kernel_f8_to_f16` (both masked
stores address the same active window). -/
private theorem f8_to_f16_traceSafeR (R : RoundingModel)
    (Y X : RegionName) (N B : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin B, s.pid * B + j.val < N →
      s.pid * B + j.val < bounds X)
    (hout : ∀ j : Fin B, s.pid * B + j.val < N →
      s.pid * B + j.val < bounds Y) :
    Kernel.TraceSafeR R bounds
      ((kernel_f8_to_f16 Y X N B).toAlgKernel) s := by
  unfold Kernel.TraceSafeR
  simp only [BlockState.pid_eq] at hx hout
  simp [kernel_f8_to_f16, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.SafeAtR,
    stepStmtR, evalOpR.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MemAccess.SafeAtR, MaskOpt.ActiveR, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hout a ha⟩

/-- The kernel sits inside the bridge's covered fragment. -/
private theorem f8_to_f16_flattenOk (Y X : RegionName) (N B : Nat) :
    ((kernel_f8_to_f16 Y X N B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [kernel_f8_to_f16, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- `kernel_f8_to_f16`'s masked IO signature (same window/mask shape as the
reverse direction). -/
def f8ToF16IO (Y X : RegionName) (N BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := kernel_f8_to_f16 Y X N BLOCK_SIZE
  inp := X
  out := Y
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < N

/-- **The headline**: for every rounding model `R`, `kernel_f8_to_f16`
implements the masked identity copy on its IO signature at the `.fp16`
output grid — every active output lane reads back as an `.fp16`-typed cell
holding `R.round .fp16 (xs j)` (the upstream duplicate store is idempotent:
the second scatter rewrites the same rounded cells). -/
specification f8_to_f16_io_correctness (R : RoundingModel)
    (Y X : RegionName) (N BLOCK_SIZE : Nat) :
    f8ToF16IO Y X N BLOCK_SIZE ⊨[R, .fp16] fun xs i => xs i := by
  refine MaskedKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
  · exact f8_to_f16_flattenOk Y X N BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact f8_to_f16_traceSafeR R Y X N BLOCK_SIZE bounds s h1 h2
  · intro s₀ xs hx
    simp only [f8ToF16IO] at hx ⊢
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      f8_to_f16_region_runR R Y X N BLOCK_SIZE s₀ xs hx
    exact ⟨s1, hexec, hval, fun r o hcond _ => hframe r o hcond⟩

end VeriTile.Bench.TritonBenchG.F8ConversionUtils
