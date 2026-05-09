/-
VeriTile.Examples.FlashAttention1.ScoreVariants.Backward

Launcher-facing backward wrappers for FA-1 score variants.
-/

import VeriTile.Examples.FlashAttention1.Backward
import VeriTile.Examples.FlashAttention1.ScoreVariants.Math

namespace VeriTile.Examples

open VeriTile.Triton
open BigOperators

namespace FA1Score

/-- Generic full launcher-facing correctness for score-variant atomic FA-1
backward.  The caller supplies the trace-to-score-contribution relation and
ordinary `dK`/`dV` per-block readback for the concrete kernel. -/
theorem gridLaunchedAtomic_scoreVariant_backward_correct
    {M D Bk numKVBlocks : Nat}
    (k : Kernel) (dQReg dKReg dVReg : RegionName)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (scoreGrad : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (scale : ℝ) (s sFinal : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (g : Grid)
    (hLaunch : Kernel.GridLaunchedAtomic k g s sFinal)
    (hInitialDQ :
      ∀ idx : TileIndex [M, D],
        s.readMem dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D idx) = 0)
    (hNoOrdinaryDQ :
      ∀ idx : TileIndex [M, D],
        ¬ Kernel.GridWriteFootprint hLaunch.frames
          (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx))
    (hAtomicContrib :
      ∀ idx : TileIndex [M, D],
        hLaunch.contributors.sum
            (fun gridIdx =>
              (hLaunch.runs gridIdx).trace.atomicAddRealSum
                (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx)) =
          Finset.univ.sum
            (fun block : Fin numKVBlocks =>
              dQBlockContributionScoreVariant visible score scoreGrad
                K V dO LSE scale block idx))
    (owner : Fin numKVBlocks → GridIndex g)
    (hDKWrite : ∀ block idx,
      (hLaunch.frames (owner block)).writes
        (dKReg, Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D idx))
    (hDVWrite : ∀ block idx,
      (hLaunch.frames (owner block)).writes
        (dVReg, Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D idx))
    (hDKBlock : ∀ block, ∀ idx : TileIndex [Bk, D],
      (hLaunch.frames (owner block)).final.readMem dKReg
          (Offset.rowMajor2D (rows := Bk) (cols := D)
            (block.val * Bk * D) D idx) =
        (attentionBackwardRealScoreVariant visible score scoreGrad Q K V dO LSE scale).dK
          (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))
    (hDVBlock : ∀ block, ∀ idx : TileIndex [Bk, D],
      (hLaunch.frames (owner block)).final.readMem dVReg
          (Offset.rowMajor2D (rows := Bk) (cols := D)
            (block.val * Bk * D) D idx) =
        (attentionBackwardRealScoreVariant visible score scoreGrad Q K V dO LSE scale).dV
          (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) :
    let bw := attentionBackwardRealScoreVariant visible score scoreGrad Q K V dO LSE scale
    (∀ idx : TileIndex [M, D],
      observeTileAt
        (some sFinal)
        dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D) idx =
      some (bw.dQ idx)) ∧
    (∀ block : Fin numKVBlocks, ∀ idx : TileIndex [Bk, D],
      observeTileAt
        (some sFinal)
        dKReg (Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D) idx =
      some (bw.dK
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) ∧
    (∀ block : Fin numKVBlocks, ∀ idx : TileIndex [Bk, D],
      observeTileAt
        (some sFinal)
        dVReg (Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D) idx =
      some (bw.dV
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) := by
  constructor
  · intro idx
    simp [observeTileAt]
    rw [hLaunch.observeAtomicCell
      (region := dQReg) (offset := Offset.rowMajor2D (rows := M) (cols := D) 0 D idx)
      (hNoOrdinaryWrite := hNoOrdinaryDQ idx)]
    rw [hInitialDQ idx, hAtomicContrib idx]
    rw [dQBlockContributionScoreVariant_sum_eq_attentionBackwardRealScoreVariant
      visible score scoreGrad Q K V dO LSE scale idx]
    simp
  · constructor
    · intro block idx
      let off := Offset.rowMajor2D (rows := Bk) (cols := D)
        (block.val * Bk * D) D idx
      have hMem := hLaunch.observeOrdinaryCell (owner block) dKReg off
        (hDKWrite block idx)
      have hRead :
          sFinal.readMem dKReg off =
            (hLaunch.frames (owner block)).final.readMem dKReg off := by
        unfold BlockState.readMem
        rw [hMem]
      change some (sFinal.readMem dKReg off) = _
      rw [hRead]
      simpa [observeTileAt, off] using hDKBlock block idx
    · intro block idx
      let off := Offset.rowMajor2D (rows := Bk) (cols := D)
        (block.val * Bk * D) D idx
      have hMem := hLaunch.observeOrdinaryCell (owner block) dVReg off
        (hDVWrite block idx)
      have hRead :
          sFinal.readMem dVReg off =
            (hLaunch.frames (owner block)).final.readMem dVReg off := by
        unfold BlockState.readMem
        rw [hMem]
      change some (sFinal.readMem dVReg off) = _
      rw [hRead]
      simpa [observeTileAt, off] using hDVBlock block idx

end FA1Score

end VeriTile.Examples
