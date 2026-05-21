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
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))
    (hDVBlock : ∀ block, ∀ idx : TileIndex [Bk, D],
      (hLaunch.frames (owner block)).final.readMem dVReg
          (Offset.rowMajor2D (rows := Bk) (cols := D)
            (block.val * Bk * D) D idx) =
        (attentionBackwardRealScoreVariant visible score scoreGrad Q K V dO LSE scale).dV
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
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
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) ∧
    (∀ block : Fin numKVBlocks, ∀ idx : TileIndex [Bk, D],
      observeTileAt
        (some sFinal)
        dVReg (Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D) idx =
      some (bw.dV
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
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

/-- ALiBi specialization of `gridLaunchedAtomic_scoreVariant_backward_correct`. -/
theorem gridLaunchedAtomic_alibi_backward_correct
    {M D Bk numKVBlocks : Nat}
    (k : Kernel) (dQReg dKReg dVReg : RegionName)
    (qStart : Nat) (slope scale : ℝ) (s sFinal : BlockState)
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
              dQBlockContributionScoreVariant allVisible
                (alibiScore qStart slope Q K scale) unitScoreGrad
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
        (attentionBackwardRealAlibi qStart slope Q K V dO LSE scale).dK
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))
    (hDVBlock : ∀ block, ∀ idx : TileIndex [Bk, D],
      (hLaunch.frames (owner block)).final.readMem dVReg
          (Offset.rowMajor2D (rows := Bk) (cols := D)
            (block.val * Bk * D) D idx) =
        (attentionBackwardRealAlibi qStart slope Q K V dO LSE scale).dV
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) :
    let bw := attentionBackwardRealAlibi qStart slope Q K V dO LSE scale
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
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) ∧
    (∀ block : Fin numKVBlocks, ∀ idx : TileIndex [Bk, D],
      observeTileAt
        (some sFinal)
        dVReg (Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D) idx =
      some (bw.dV
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) := by
  simpa [attentionBackwardRealAlibi] using
    gridLaunchedAtomic_scoreVariant_backward_correct
      (k := k) (dQReg := dQReg) (dKReg := dKReg) (dVReg := dVReg)
      (visible := allVisible) (score := alibiScore qStart slope Q K scale)
      (scoreGrad := unitScoreGrad) (scale := scale) (s := s) (sFinal := sFinal)
      (Q := Q) (K := K) (V := V) (dO := dO) (LSE := LSE) (g := g)
      hLaunch hInitialDQ hNoOrdinaryDQ hAtomicContrib owner hDKWrite hDVWrite
      (by intro block idx; simpa [attentionBackwardRealAlibi] using hDKBlock block idx)
      (by intro block idx; simpa [attentionBackwardRealAlibi] using hDVBlock block idx)

/-- Sliding-window specialization of
`gridLaunchedAtomic_scoreVariant_backward_correct`. -/
theorem gridLaunchedAtomic_slidingWindow_backward_correct
    {M D Bk numKVBlocks : Nat}
    (k : Kernel) (dQReg dKReg dVReg : RegionName)
    (qStart window : Nat) (scale : ℝ) (s sFinal : BlockState)
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
              dQBlockContributionScoreVariant (slidingVisible window qStart)
                (dotScore Q K scale) unitScoreGrad K V dO LSE scale block idx))
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
        (attentionBackwardRealSlidingWindow qStart window Q K V dO LSE scale).dK
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))
    (hDVBlock : ∀ block, ∀ idx : TileIndex [Bk, D],
      (hLaunch.frames (owner block)).final.readMem dVReg
          (Offset.rowMajor2D (rows := Bk) (cols := D)
            (block.val * Bk * D) D idx) =
        (attentionBackwardRealSlidingWindow qStart window Q K V dO LSE scale).dV
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) :
    let bw := attentionBackwardRealSlidingWindow qStart window Q K V dO LSE scale
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
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) ∧
    (∀ block : Fin numKVBlocks, ∀ idx : TileIndex [Bk, D],
      observeTileAt
        (some sFinal)
        dVReg (Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D) idx =
      some (bw.dV
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) := by
  simpa [attentionBackwardRealSlidingWindow] using
    gridLaunchedAtomic_scoreVariant_backward_correct
      (k := k) (dQReg := dQReg) (dKReg := dKReg) (dVReg := dVReg)
      (visible := slidingVisible window qStart) (score := dotScore Q K scale)
      (scoreGrad := unitScoreGrad) (scale := scale) (s := s) (sFinal := sFinal)
      (Q := Q) (K := K) (V := V) (dO := dO) (LSE := LSE) (g := g)
      hLaunch hInitialDQ hNoOrdinaryDQ hAtomicContrib owner hDKWrite hDVWrite
      (by intro block idx; simpa [attentionBackwardRealSlidingWindow] using hDKBlock block idx)
      (by intro block idx; simpa [attentionBackwardRealSlidingWindow] using hDVBlock block idx)

/-- Softcap specialization of `gridLaunchedAtomic_scoreVariant_backward_correct`. -/
theorem gridLaunchedAtomic_softcap_backward_correct
    {M D Bk numKVBlocks : Nat}
    (k : Kernel) (dQReg dKReg dVReg : RegionName)
    (softcap scale : ℝ) (s sFinal : BlockState)
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
              dQBlockContributionScoreVariant allVisible
                (softcapDotScore softcap Q K scale) (softcapScoreGrad softcap Q K scale)
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
        (attentionBackwardRealSoftcap softcap Q K V dO LSE scale).dK
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))
    (hDVBlock : ∀ block, ∀ idx : TileIndex [Bk, D],
      (hLaunch.frames (owner block)).final.readMem dVReg
          (Offset.rowMajor2D (rows := Bk) (cols := D)
            (block.val * Bk * D) D idx) =
        (attentionBackwardRealSoftcap softcap Q K V dO LSE scale).dV
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) :
    let bw := attentionBackwardRealSoftcap softcap Q K V dO LSE scale
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
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) ∧
    (∀ block : Fin numKVBlocks, ∀ idx : TileIndex [Bk, D],
      observeTileAt
        (some sFinal)
        dVReg (Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D) idx =
      some (bw.dV
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) := by
  simpa [attentionBackwardRealSoftcap] using
    gridLaunchedAtomic_scoreVariant_backward_correct
      (k := k) (dQReg := dQReg) (dKReg := dKReg) (dVReg := dVReg)
      (visible := allVisible) (score := softcapDotScore softcap Q K scale)
      (scoreGrad := softcapScoreGrad softcap Q K scale) (scale := scale)
      (s := s) (sFinal := sFinal)
      (Q := Q) (K := K) (V := V) (dO := dO) (LSE := LSE) (g := g)
      hLaunch hInitialDQ hNoOrdinaryDQ hAtomicContrib owner hDKWrite hDVWrite
      (by intro block idx; simpa [attentionBackwardRealSoftcap] using hDKBlock block idx)
      (by intro block idx; simpa [attentionBackwardRealSoftcap] using hDVBlock block idx)

/-- Combined ALiBi + sliding-window + softcap specialization of
`gridLaunchedAtomic_scoreVariant_backward_correct`. -/
theorem gridLaunchedAtomic_alibiSlidingSoftcap_backward_correct
    {M D Bk numKVBlocks : Nat}
    (k : Kernel) (dQReg dKReg dVReg : RegionName)
    (qStart window : Nat) (slope softcap scale : ℝ) (s sFinal : BlockState)
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
              let baseScore := alibiScore qStart slope Q K scale
              dQBlockContributionScoreVariant (slidingVisible window qStart)
                (fun i j => softcapScore softcap (baseScore i j))
                (softcapScoreGradOf softcap baseScore)
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
        (attentionBackwardRealAlibiSlidingSoftcap qStart window slope softcap
          Q K V dO LSE scale).dK
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))
    (hDVBlock : ∀ block, ∀ idx : TileIndex [Bk, D],
      (hLaunch.frames (owner block)).final.readMem dVReg
          (Offset.rowMajor2D (rows := Bk) (cols := D)
            (block.val * Bk * D) D idx) =
        (attentionBackwardRealAlibiSlidingSoftcap qStart window slope softcap
          Q K V dO LSE scale).dV
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) :
    let bw := attentionBackwardRealAlibiSlidingSoftcap qStart window slope softcap
      Q K V dO LSE scale
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
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) ∧
    (∀ block : Fin numKVBlocks, ∀ idx : TileIndex [Bk, D],
      observeTileAt
        (some sFinal)
        dVReg (Offset.rowMajor2D (rows := Bk) (cols := D)
          (block.val * Bk * D) D) idx =
      some (bw.dV
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) := by
  simpa [attentionBackwardRealAlibiSlidingSoftcap] using
    gridLaunchedAtomic_scoreVariant_backward_correct
      (k := k) (dQReg := dQReg) (dKReg := dKReg) (dVReg := dVReg)
      (visible := slidingVisible window qStart)
      (score := fun i j =>
        softcapScore softcap (alibiScore qStart slope Q K scale i j))
      (scoreGrad := softcapScoreGradOf softcap (alibiScore qStart slope Q K scale))
      (scale := scale) (s := s) (sFinal := sFinal)
      (Q := Q) (K := K) (V := V) (dO := dO) (LSE := LSE) (g := g)
      hLaunch hInitialDQ hNoOrdinaryDQ
      (by
        intro idx
        simpa using hAtomicContrib idx)
      owner hDKWrite hDVWrite
      (by
        intro block idx
        simpa [attentionBackwardRealAlibiSlidingSoftcap] using hDKBlock block idx)
      (by
        intro block idx
        simpa [attentionBackwardRealAlibiSlidingSoftcap] using hDVBlock block idx)

end FA1Score

end VeriTile.Examples
