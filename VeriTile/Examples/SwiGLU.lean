/-
VeriTile.Examples.SwiGLU

SwiGLU fusion example — the headline property-P2 running example
(`documents/FusionCorrectness.md` §3, ticket #491).

The fused kernel

  out = silu(a) * b       where  silu(a) = a * sigmoid(a)

is proved **fusion-correct** against the two-stage pipeline

  silu_step :  silu = a * sigmoid(a)   (materializes the intermediate buffer)
  mul_step  :  out  = silu * b

i.e. `ComputeRefine.FusionCorrect swigluFusedKernel [siluStep, mulStep]`. The
golden reference is the stage pipeline itself, assembled with the shared
`seqCompose` combinator (#490); the eliminated intermediate store/load rounding
site is the canonical input for the P4 FP-analysis layer and is out of scope
here (ℝ-equality only, decision D3).

This material is ported (re-homed to the `VeriTile/*` layout, restated
pointwise) from `origin/feat/447-swiglu-pilot`, which is quarried — not merged —
per `documents/FusionCorrectness.md` §5.

Source Triton (`.py` reference, aligned single-block flavour; assumes
`BLOCK_SIZE` equals the logical vector length, so no boundary mask is used):

```python
@triton.jit
def swiglu_kernel(a_ptr, b_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    silu = a * tl.sigmoid(a)
    tl.store(out_ptr + offsets, silu * b)

@triton.jit
def silu_step(a_ptr, silu_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + offsets)
    tl.store(silu_ptr + offsets, a * tl.sigmoid(a))

@triton.jit
def mul_step(silu_ptr, b_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    silu = tl.load(silu_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    tl.store(out_ptr + offsets, silu * b)
```
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL
import VeriTile.Math.Activation
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile

/-! ## Kernels -/

/-- Fused SwiGLU: `out = (a * sigmoid a) * b`, no intermediate buffer. -/
def swigluFusedKernel (aReg bReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  a       := tl.load($(aReg) + offsets)
  b       := tl.load($(bReg) + offsets)
  silu    := a * tl.sigmoid(a)
  y       := silu * b
  tl.store($(outReg) + offsets, y)
}

/-- Stage 1: `silu = a * sigmoid a`, materialized into `siluReg`. -/
def siluStep (aReg siluReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  a       := tl.load($(aReg) + offsets)
  silu    := a * tl.sigmoid(a)
  tl.store($(siluReg) + offsets, silu)
}

/-- Stage 2: `out = silu * b`, reading the materialized intermediate. -/
def mulStep (siluReg bReg outReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  silu    := tl.load($(siluReg) + offsets)
  b       := tl.load($(bReg) + offsets)
  y       := silu * b
  tl.store($(outReg) + offsets, y)
}

/-- The stage pipeline executed as a state-threaded fold. Bridges the shared
`seqCompose` combinator (#490) to the FusedSiLU-style hand-rolled fold used by
the component correctness lemmas below. -/
noncomputable def execUnfusedSwiGLU
    (aReg bReg siluReg outReg : RegionName)
    (blockSize : Nat) (s : BlockState) : Option BlockState :=
  match exec (siluStep aReg siluReg blockSize) s with
  | none => none
  | some s1 => exec (mulStep siluReg bReg outReg blockSize) s1

/-- Executing the `seqCompose` stage pipeline is the two-stage state-threaded
fold. Specialization of the packaged `exec_seqCompose` (#490) to this pipeline,
so the component lemmas keep the familiar match shape. -/
theorem exec_seqComposeSwiGLU
    (aReg bReg siluReg outReg : RegionName) (blockSize : Nat) (s : BlockState) :
    exec (seqCompose [siluStep aReg siluReg blockSize,
                      mulStep siluReg bReg outReg blockSize]) s =
      execUnfusedSwiGLU aReg bReg siluReg outReg blockSize s := by
  rw [exec_seqCompose]
  cases h : exec (siluStep aReg siluReg blockSize) s <;>
    simp [List.foldlM_cons, execUnfusedSwiGLU, h]

/-! ## Per-lane specs -/

/-- Per-lane fused SwiGLU output. Delegates to `TiledActivation.swiglu`. -/
noncomputable def swigluSpec {blockSize : Nat}
    (as bs : Fin blockSize → ℝ) (i : Fin blockSize) : ℝ :=
  TiledActivation.swiglu (as i) (bs i)

/-- Per-lane intermediate: `silu = a * sigmoid a`. Delegates to
`TiledActivation.silu`. -/
noncomputable def swigluSiluSpec {blockSize : Nat}
    (as : Fin blockSize → ℝ) (i : Fin blockSize) : ℝ :=
  TiledActivation.silu (as i)

/-! ## Component correctness -/

theorem swigluFused_correct
    (aReg bReg outReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (as bs : Fin blockSize → ℝ)
    (_h_a : InputLoadedAt s aReg blockSize as)
    (_h_b : InputLoadedAt s bReg blockSize bs) :
    ∀ i : Fin blockSize,
      observeAt (exec (swigluFusedKernel aReg bReg outReg blockSize) s)
                outReg blockSize s.pid i
        = some (swigluSpec as bs i) := by
  intro i
  have h_inj :
      Function.Injective (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, swigluFusedKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        swigluSpec, TiledActivation.swiglu, TiledActivation.silu]
  unfold InputLoadedAt at _h_a _h_b
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_a, _h_b]

theorem siluStep_correct
    (aReg siluReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (as : Fin blockSize → ℝ)
    (_h_a : InputLoadedAt s aReg blockSize as) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStep aReg siluReg blockSize) s)
                siluReg blockSize s.pid i
        = some (swigluSiluSpec as i) := by
  intro i
  have h_inj :
      Function.Injective (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, siluStep, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        swigluSiluSpec, TiledActivation.silu]
  unfold InputLoadedAt at _h_a
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_a]

theorem siluStep_preserves
    (aReg siluReg keepReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (as : Fin blockSize → ℝ)
    (_h_a : InputLoadedAt s aReg blockSize as)
    (h_keep : keepReg ≠ siluReg) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStep aReg siluReg blockSize) s)
                keepReg blockSize s.pid i
        = some (s.readMem keepReg (s.pid * blockSize + i.val)) := by
  intro i
  simp [observeAt, exec, siluStep, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_a
  simp_rw [_h_a]
  rw [BlockState.scatter_preserves_other_region siluReg
        (fun k : TileIndex [blockSize] => s.pid * blockSize + k.1.val)
        (fun k : TileIndex [blockSize] => as k.1 * Real.sigmoid (as k.1))
        keepReg h_keep (s.pid * blockSize + i.val)]
  simp [BlockState.pid_eq]

theorem mulStep_correct
    (siluReg bReg outReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (silus bs : Fin blockSize → ℝ)
    (_h_silu : InputLoadedAt s siluReg blockSize silus)
    (_h_b : InputLoadedAt s bReg blockSize bs) :
    ∀ i : Fin blockSize,
      observeAt (exec (mulStep siluReg bReg outReg blockSize) s)
                outReg blockSize s.pid i
        = some (silus i * bs i) := by
  intro i
  have h_inj :
      Function.Injective (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, mulStep, stepStmts, stepStmt, evalOp,
        Tile.bop, NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_silu _h_b
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_silu, _h_b]

/-! ## Pipeline correctness -/

set_option maxHeartbeats 800000 in
theorem unfused_swiglu_correct
    (aReg bReg siluReg outReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (as bs : Fin blockSize → ℝ)
    (_h_a : InputLoadedAt s aReg blockSize as)
    (_h_b : InputLoadedAt s bReg blockSize bs)
    (_h_siluB : siluReg ≠ bReg) :
    ∀ i : Fin blockSize,
      observeAt
        (execUnfusedSwiGLU aReg bReg siluReg outReg blockSize s)
        outReg blockSize s.pid i
        = some (swigluSpec as bs i) := by
  intro i
  unfold execUnfusedSwiGLU
  cases h_silu : exec (siluStep aReg siluReg blockSize) s with
  | none =>
      have hsj := siluStep_correct aReg siluReg blockSize _hN s as _h_a i
      rw [h_silu] at hsj
      simp [observeAt] at hsj
  | some s1 =>
      have hpid1 : s1.pid = s.pid := exec_pid h_silu
      have h_siluLoaded : InputLoadedAt s1 siluReg blockSize (swigluSiluSpec as) := by
        intro j
        have hsj := siluStep_correct aReg siluReg blockSize _hN s as _h_a j
        rw [h_silu] at hsj
        simp [observeAt] at hsj
        rw [hpid1]
        exact hsj
      have h_b1 : InputLoadedAt s1 bReg blockSize bs := by
        intro j
        have hbj :=
          siluStep_preserves aReg siluReg bReg blockSize _hN s as _h_a
            (fun h => _h_siluB h.symm) j
        rw [h_silu] at hbj
        simp [observeAt] at hbj
        rw [hpid1]
        rw [hbj]
        exact _h_b j
      have hout :=
        mulStep_correct siluReg bReg outReg blockSize _hN s1
          (swigluSiluSpec as) bs h_siluLoaded h_b1 i
      rw [hpid1] at hout
      rw [hout]
      simp [swigluSpec, swigluSiluSpec, TiledActivation.swiglu, TiledActivation.silu]

/-! ## Refinement views -/

/-- The fused kernel and the stage pipeline agree cell-by-cell on the declared
output (per-lane `observeAt` form). -/
theorem swiglu_refinement
    (aReg bReg siluReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (as bs : Fin blockSize → ℝ)
    (h_a : InputLoadedAt s aReg blockSize as)
    (h_b : InputLoadedAt s bReg blockSize bs)
    (h_siluB : siluReg ≠ bReg) :
    ∀ i : Fin blockSize,
      observeAt (exec (swigluFusedKernel aReg bReg outReg blockSize) s)
                outReg blockSize s.pid i =
      observeAt
        (execUnfusedSwiGLU aReg bReg siluReg outReg blockSize s)
        outReg blockSize s.pid i := by
  intro i
  rw [swigluFused_correct aReg bReg outReg blockSize hN s as bs h_a h_b i,
      unfused_swiglu_correct
        aReg bReg siluReg outReg blockSize hN s as bs h_a h_b h_siluB i]

/-- View-level (TensorView) surface for `swiglu_refinement`. -/
theorem swiglu_refinement_exec_view
    (aReg bReg siluReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (as bs : Fin blockSize → ℝ)
    (h_a : TensorView.loaded s (programTileView s aReg blockSize)
      (fun idx : TileIndex [blockSize] => as idx.1))
    (h_b : TensorView.loaded s (programTileView s bReg blockSize)
      (fun idx : TileIndex [blockSize] => bs idx.1))
    (h_siluB : siluReg ≠ bReg) :
    ∀ idx : TileIndex [blockSize],
      TensorView.observe
          (exec (swigluFusedKernel aReg bReg outReg blockSize) s)
          (programTileView s outReg blockSize) idx =
      TensorView.observe
          (execUnfusedSwiGLU aReg bReg siluReg outReg blockSize s)
          (programTileView s outReg blockSize) idx := by
  intro idx
  have ha := inputLoadedAt_of_programTileView_loaded (s := s) (region := aReg)
    (N := blockSize) (xs := as) h_a
  have hb := inputLoadedAt_of_programTileView_loaded (s := s) (region := bReg)
    (N := blockSize) (xs := bs) h_b
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using swiglu_refinement aReg bReg siluReg outReg
      blockSize hN s as bs ha hb h_siluB idx.1

/-- **Headline property P2 (fusion correctness):** the fused SwiGLU kernel
exactly implements the `[siluStep, mulStep]` stage pipeline on the declared
output, at the ℝ layer (`documents/FusionCorrectness.md` §3, ticket #491). -/
theorem swiglu_fusion_view
    (aReg bReg siluReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (as bs : Fin blockSize → ℝ)
    (h_a : TensorView.loaded s (programTileView s aReg blockSize)
      (fun idx : TileIndex [blockSize] => as idx.1))
    (h_b : TensorView.loaded s (programTileView s bReg blockSize)
      (fun idx : TileIndex [blockSize] => bs idx.1))
    (h_siluB : siluReg ≠ bReg) :
    ComputeRefine.FusionCorrect (α := ℝ)
      (swigluFusedKernel aReg bReg outReg blockSize)
      [siluStep aReg siluReg blockSize,
       mulStep siluReg bReg outReg blockSize]
      s
      (ComputeCorrect.WriteMap.ofTensorView
        (programTileView s outReg blockSize)) := by
  rw [ComputeRefine.fusionCorrect_iff]
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro idx
  have hview := swiglu_refinement_exec_view aReg bReg siluReg outReg
    blockSize hN s as bs h_a h_b h_siluB idx
  have hSeq := exec_seqComposeSwiGLU aReg bReg siluReg outReg blockSize s
  rw [← hSeq, hR] at hview
  simpa [hL, ComputeCorrect.WriteMap.ofTensorView, TensorView.observe,
    observeTileAt] using hview

end VeriTile.Examples
