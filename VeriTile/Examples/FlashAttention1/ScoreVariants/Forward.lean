/-
VeriTile.Examples.FlashAttention1.ScoreVariants.Forward

Forward correctness and readout surfaces for score-variant FA-1 kernels.
-/

import VeriTile.Examples.FlashAttention1.ScoreVariants.Loop

namespace VeriTile.Examples

open VeriTile

namespace FA1Score
theorem fa1_score_blockrec_forward_raw_of_step
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (body : List Stmt)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_score_blockrec qReg kReg vReg s.pid Q K V visible score i st →
          ∃ st',
            stepStmts body (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
            P_fa1_score_blockrec qReg kReg vReg s.pid Q K V visible score
              (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (stepStmts
            (fa1ScorePreLoop qReg M D ++
              [Stmt.forLoop "n" numKVBlocks body] ++
              fa1ScorePostLoop outReg M D) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial visible score V numKVBlocks idx /
            lScoreBlockPartial visible score numKVBlocks idx.1) := by
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_score_blockrec_preLoop_correct qReg kReg vReg Q K V visible score s hQ hK hV
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv
      (P := P_fa1_score_blockrec qReg kReg vReg s.pid Q K V visible score) hP0
      hStep
  intro idx
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1ScorePreLoop qReg M D)
        (l2 := [Stmt.forLoop "n" numKVBlocks body] ++ fa1ScorePostLoop outReg M D)
        hPre]
  rw [stepStmts.append_some (l1 := ([Stmt.forLoop "n" numKVBlocks body]))
      (l2 := fa1ScorePostLoop outReg M D) ?_]
  · exact fa1_score_blockrec_postLoop_correct qReg kReg vReg outReg s.pid Q K V
      visible score sLoop hPLoop idx
  · rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil

theorem fa1_forward_softcap_blockrec_raw_of_step
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale softcap : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_score_blockrec qReg kReg vReg s.pid Q K V allVisible
          (softcapDotScore softcap Q K scale) i st →
          ∃ st',
            stepStmts (fa1ScoreLoopBodySoftcap kReg vReg M D Bk scale softcap)
              (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
            P_fa1_score_blockrec qReg kReg vReg s.pid Q K V allVisible
              (softcapDotScore softcap Q K scale) (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelSoftcap qReg kReg vReg outReg M D Bk
              numKVBlocks scale softcap) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial allVisible (softcapDotScore softcap Q K scale) V
              numKVBlocks idx /
            lScoreBlockPartial allVisible (softcapDotScore softcap Q K scale)
              numKVBlocks idx.1) := by
  intro idx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [fa1ForwardKernelSoftcap_body_eq]
  exact fa1_score_blockrec_forward_raw_of_step qReg kReg vReg outReg Q K V
    allVisible (softcapDotScore softcap Q K scale)
    (fa1ScoreLoopBodySoftcap kReg vReg M D Bk scale softcap) s hQ hK hV
    hStep idx

theorem fa1_forward_alibi_blockrec_raw_of_step
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_score_blockrec qReg kReg vReg s.pid Q K V allVisible
          (alibiScore qStart slope Q K scale) i st →
          ∃ st',
            stepStmts (fa1ScoreLoopBodyAlibi kReg vReg M D Bk scale slope)
              (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
            P_fa1_score_blockrec qReg kReg vReg s.pid Q K V allVisible
              (alibiScore qStart slope Q K scale) (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelAlibi qReg kReg vReg outReg M D Bk
              numKVBlocks scale slope) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial allVisible (alibiScore qStart slope Q K scale) V
              numKVBlocks idx /
            lScoreBlockPartial allVisible (alibiScore qStart slope Q K scale)
              numKVBlocks idx.1) := by
  intro idx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [fa1ForwardKernelAlibi_body_eq]
  exact fa1_score_blockrec_forward_raw_of_step qReg kReg vReg outReg Q K V
    allVisible (alibiScore qStart slope Q K scale)
    (fa1ScoreLoopBodyAlibi kReg vReg M D Bk scale slope) s hQ hK hV
    hStep idx

theorem fa1_forward_slidingWindow_blockrec_raw_of_step
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_score_blockrec qReg kReg vReg s.pid Q K V
          (slidingVisible window qStart) (dotScore Q K scale) i st →
          ∃ st',
            stepStmts (fa1ScoreLoopBodySlidingWindow kReg vReg M D Bk window scale)
              (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
            P_fa1_score_blockrec qReg kReg vReg s.pid Q K V
              (slidingVisible window qStart) (dotScore Q K scale) (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelSlidingWindow qReg kReg vReg outReg M D Bk
              numKVBlocks window scale) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial (slidingVisible window qStart) (dotScore Q K scale) V
              numKVBlocks idx /
            lScoreBlockPartial (slidingVisible window qStart) (dotScore Q K scale)
              numKVBlocks idx.1) := by
  intro idx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [fa1ForwardKernelSlidingWindow_body_eq]
  exact fa1_score_blockrec_forward_raw_of_step qReg kReg vReg outReg Q K V
    (slidingVisible window qStart) (dotScore Q K scale)
    (fa1ScoreLoopBodySlidingWindow kReg vReg M D Bk window scale) s hQ hK hV
    hStep idx

theorem fa1_forward_alibiSlidingSoftcap_blockrec_raw_of_step
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_score_blockrec qReg kReg vReg s.pid Q K V
          (slidingVisible window qStart)
          (fun r c => softcapScore softcap (alibiScore qStart slope Q K scale r c))
          i st →
          ∃ st',
            stepStmts
              (fa1ScoreLoopBodyAlibiSlidingSoftcap kReg vReg M D Bk window
                scale slope softcap)
              (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
            P_fa1_score_blockrec qReg kReg vReg s.pid Q K V
              (slidingVisible window qStart)
              (fun r c => softcapScore softcap (alibiScore qStart slope Q K scale r c))
              (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelAlibiSlidingSoftcap qReg kReg vReg outReg M D Bk
              numKVBlocks window scale slope softcap) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial (slidingVisible window qStart)
              (fun r c => softcapScore softcap (alibiScore qStart slope Q K scale r c))
              V numKVBlocks idx /
            lScoreBlockPartial (slidingVisible window qStart)
              (fun r c => softcapScore softcap (alibiScore qStart slope Q K scale r c))
              numKVBlocks idx.1) := by
  intro idx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [fa1ForwardKernelAlibiSlidingSoftcap_body_eq]
  exact fa1_score_blockrec_forward_raw_of_step qReg kReg vReg outReg Q K V
    (slidingVisible window qStart)
    (fun r c => softcapScore softcap (alibiScore qStart slope Q K scale r c))
    (fa1ScoreLoopBodyAlibiSlidingSoftcap kReg vReg M D Bk window scale slope softcap)
    s hQ hK hV hStep idx

theorem fa1_forward_softcap_blockrec_raw
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale softcap : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelSoftcap qReg kReg vReg outReg M D Bk
              numKVBlocks scale softcap) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial allVisible (softcapDotScore softcap Q K scale) V
              numKVBlocks idx /
            lScoreBlockPartial allVisible (softcapDotScore softcap Q K scale)
              numKVBlocks idx.1) := by
  intro idx
  exact fa1_forward_softcap_blockrec_raw_of_step qReg kReg vReg outReg Q K V
    scale softcap s hQ hK hV
    (fun i st hi hP =>
      fa1_score_loop_stepSoftcap_correct hBk qReg kReg vReg s.pid Q K V
        scale softcap i st hi hP) idx

theorem fa1_forward_alibi_blockrec_raw
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hqStart : qStart = s.pid * M)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelAlibi qReg kReg vReg outReg M D Bk
              numKVBlocks scale slope) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial allVisible (alibiScore qStart slope Q K scale) V
              numKVBlocks idx /
            lScoreBlockPartial allVisible (alibiScore qStart slope Q K scale)
              numKVBlocks idx.1) := by
  intro idx
  exact fa1_forward_alibi_blockrec_raw_of_step qReg kReg vReg outReg qStart slope
    Q K V scale s hQ hK hV
    (fun i st hi hP =>
      fa1_score_loop_stepAlibi_correct hBk qReg kReg vReg s.pid qStart slope Q K V
        scale i st hi hqStart hP) idx

theorem fa1_forward_slidingWindow_blockrec_raw
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hqStart : qStart = s.pid * M)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelSlidingWindow qReg kReg vReg outReg M D Bk
              numKVBlocks window scale) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial (slidingVisible window qStart) (dotScore Q K scale) V
              numKVBlocks idx /
            lScoreBlockPartial (slidingVisible window qStart) (dotScore Q K scale)
              numKVBlocks idx.1) := by
  intro idx
  exact fa1_forward_slidingWindow_blockrec_raw_of_step qReg kReg vReg outReg
    qStart window Q K V scale s hQ hK hV
    (fun i st hi hP =>
      fa1_score_loop_stepSlidingWindow_correct hBk qReg kReg vReg s.pid qStart
        window Q K V scale i st hi hqStart hP) idx

theorem fa1_forward_alibiSlidingSoftcap_blockrec_raw
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hqStart : qStart = s.pid * M)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelAlibiSlidingSoftcap qReg kReg vReg outReg M D Bk
              numKVBlocks window scale slope softcap) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial (slidingVisible window qStart)
              (fun r c => softcapScore softcap (alibiScore qStart slope Q K scale r c))
              V numKVBlocks idx /
            lScoreBlockPartial (slidingVisible window qStart)
              (fun r c => softcapScore softcap (alibiScore qStart slope Q K scale r c))
              numKVBlocks idx.1) := by
  intro idx
  exact fa1_forward_alibiSlidingSoftcap_blockrec_raw_of_step qReg kReg vReg outReg
    qStart window slope softcap Q K V scale s hQ hK hV
    (fun i st hi hP =>
      fa1_score_loop_stepAlibiSlidingSoftcap_correct hBk qReg kReg vReg s.pid
        qStart window slope softcap Q K V scale i st hi hqStart hP) idx

theorem fa1_forward_softcap_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale softcap : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelSoftcap qReg kReg vReg outReg M D Bk
              numKVBlocks scale softcap) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some (attentionRealSoftcap softcap Q K V scale idx) := by
  intro idx
  have hraw :=
    fa1_forward_softcap_blockrec_raw hBk qReg kReg vReg outReg Q K V
      scale softcap s hQ hK hV idx
  simpa [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealSoftcap
    softcap Q K V scale idx] using hraw

theorem fa1_forward_alibi_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hqStart : qStart = s.pid * M)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelAlibi qReg kReg vReg outReg M D Bk
              numKVBlocks scale slope) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some (attentionRealAlibi qStart slope Q K V scale idx) := by
  intro idx
  have hraw :=
    fa1_forward_alibi_blockrec_raw hBk qReg kReg vReg outReg qStart slope
      Q K V scale s hqStart hQ hK hV idx
  simpa [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealAlibi
    qStart slope Q K V scale idx] using hraw

theorem fa1_forward_slidingWindow_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hqStart : qStart = s.pid * M)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelSlidingWindow qReg kReg vReg outReg M D Bk
              numKVBlocks window scale) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some (attentionRealSlidingWindow qStart window Q K V scale idx) := by
  intro idx
  have hraw :=
    fa1_forward_slidingWindow_blockrec_raw hBk qReg kReg vReg outReg
      qStart window Q K V scale s hqStart hQ hK hV idx
  simpa [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealSlidingWindow
    qStart window Q K V scale idx] using hraw

theorem fa1_forward_alibiSlidingSoftcap_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hqStart : qStart = s.pid * M)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelAlibiSlidingSoftcap qReg kReg vReg outReg M D Bk
              numKVBlocks window scale slope softcap) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (attentionRealAlibiSlidingSoftcap qStart window slope softcap
            Q K V scale idx) := by
  intro idx
  have hraw :=
    fa1_forward_alibiSlidingSoftcap_blockrec_raw hBk qReg kReg vReg outReg
      qStart window slope softcap Q K V scale s hqStart hQ hK hV idx
  simpa [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealAlibiSlidingSoftcap
    qStart window slope softcap Q K V scale idx] using hraw

theorem P_fa1_score_readout_ratio
    {M D S : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ)
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (s : BlockState)
    (_hP : P_fa1_score qReg kReg vReg origPid Q K V visible score S s)
    (idx : TileIndex [M, D]) :
    oScoreOnline visible score V S idx /
        lScoreOnline visible score S idx.1 =
      attentionRealMaskedScore visible score V idx := by
  exact oScoreOnline_div_lScoreOnline_eq_attentionRealMaskedScore
    visible score V idx

theorem P_fa1_score_readout_alibi
    {M D S : Nat}
    (qReg kReg vReg : RegionName)
    (origPid qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ) (scale : ℝ) (s : BlockState)
    (_hP : P_fa1_score qReg kReg vReg origPid Q K V allVisible
      (alibiScore qStart slope Q K scale) S s)
    (idx : TileIndex [M, D]) :
    oScoreOnline allVisible (alibiScore qStart slope Q K scale) V S idx /
        lScoreOnline allVisible (alibiScore qStart slope Q K scale) S idx.1 =
      attentionRealAlibi qStart slope Q K V scale idx := by
  exact oScoreOnline_div_lScoreOnline_eq_attentionRealAlibi
    qStart slope Q K V scale idx

theorem P_fa1_score_readout_slidingWindow
    {M D S : Nat}
    (qReg kReg vReg : RegionName)
    (origPid qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ) (scale : ℝ) (s : BlockState)
    (_hP : P_fa1_score qReg kReg vReg origPid Q K V
      (slidingVisible window qStart) (dotScore Q K scale) S s)
    (idx : TileIndex [M, D]) :
    oScoreOnline (slidingVisible window qStart) (dotScore Q K scale) V S idx /
        lScoreOnline (slidingVisible window qStart) (dotScore Q K scale) S idx.1 =
      attentionRealSlidingWindow qStart window Q K V scale idx := by
  exact oScoreOnline_div_lScoreOnline_eq_attentionRealSlidingWindow
    qStart window Q K V scale idx

theorem P_fa1_score_readout_softcap
    {M D S : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat) (softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ) (scale : ℝ) (s : BlockState)
    (_hP : P_fa1_score qReg kReg vReg origPid Q K V allVisible
      (softcapDotScore softcap Q K scale) S s)
    (idx : TileIndex [M, D]) :
    oScoreOnline allVisible (softcapDotScore softcap Q K scale) V S idx /
        lScoreOnline allVisible (softcapDotScore softcap Q K scale) S idx.1 =
      attentionRealSoftcap softcap Q K V scale idx := by
  exact oScoreOnline_div_lScoreOnline_eq_attentionRealSoftcap
    softcap Q K V scale idx

theorem P_fa1_score_readout_alibiSlidingSoftcap
    {M D S : Nat}
    (qReg kReg vReg : RegionName)
    (origPid qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ) (scale : ℝ) (s : BlockState)
    (_hP : P_fa1_score qReg kReg vReg origPid Q K V
      (slidingVisible window qStart)
      (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
      S s)
    (idx : TileIndex [M, D]) :
    oScoreOnline (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        V S idx /
      lScoreOnline (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        S idx.1 =
      attentionRealAlibiSlidingSoftcap qStart window slope softcap Q K V scale idx := by
  exact oScoreOnline_div_lScoreOnline_eq_attentionRealAlibiSlidingSoftcap
    qStart window slope softcap Q K V scale idx

@[simp] theorem attentionRealScore_apply {M S D : Nat}
    (score : Fin M → Fin S → ℝ) (V : TileIndex [S, D] → ℝ)
    (i : Fin M) (d : Fin D) :
    attentionRealScore score V (i, d, PUnit.unit) =
      (Finset.univ.sum (fun j : Fin S =>
        Real.exp (score i j) * V (j, d, PUnit.unit))) /
      (Finset.univ.sum (fun j : Fin S => Real.exp (score i j))) := by
  rfl

@[simp] theorem attentionRealMaskedScore_apply {M S D : Nat}
    (visible : Fin M → Fin S → Bool) (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) (i : Fin M) (d : Fin D) :
    attentionRealMaskedScore visible score V (i, d, PUnit.unit) =
      (Finset.univ.sum (fun j : Fin S =>
        (if visible i j then Real.exp (score i j) else 0) *
          V (j, d, PUnit.unit))) /
      (Finset.univ.sum (fun j : Fin S =>
        if visible i j then Real.exp (score i j) else 0)) := by
  rfl

noncomputable def attentionReal4DAlibi {B H S_q S_k D : Nat}
    (slopes : Fin H → ℝ)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionRealAlibi 0 (slopes h)
      (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

noncomputable def attentionReal4DSlidingWindow {B H S_q S_k D : Nat}
    (window : Nat)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionRealSlidingWindow 0 window
      (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

noncomputable def attentionReal4DSoftcap {B H S_q S_k D : Nat}
    (softcap : ℝ)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionRealSoftcap softcap
      (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

noncomputable def attentionReal4DAlibiSlidingSoftcap {B H S_q S_k D : Nat}
    (window : Nat) (slopes : Fin H → ℝ) (softcap : ℝ)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionRealAlibiSlidingSoftcap 0 window (slopes h) softcap
      (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

@[simp] theorem attentionReal4DAlibi_slice {B H S_q S_k D : Nat}
    (slopes : Fin H → ℝ)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4DAlibi slopes Q K V scale (b, h, i, d, PUnit.unit) =
      attentionRealAlibi 0 (slopes h)
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        scale (i, d, PUnit.unit) := rfl

@[simp] theorem attentionReal4DSlidingWindow_slice {B H S_q S_k D : Nat}
    (window : Nat)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4DSlidingWindow window Q K V scale (b, h, i, d, PUnit.unit) =
      attentionRealSlidingWindow 0 window
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        scale (i, d, PUnit.unit) := rfl

@[simp] theorem attentionReal4DSoftcap_slice {B H S_q S_k D : Nat}
    (softcap : ℝ)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4DSoftcap softcap Q K V scale (b, h, i, d, PUnit.unit) =
      attentionRealSoftcap softcap
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        scale (i, d, PUnit.unit) := rfl

@[simp] theorem attentionReal4DAlibiSlidingSoftcap_slice {B H S_q S_k D : Nat}
    (window : Nat) (slopes : Fin H → ℝ) (softcap : ℝ)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4DAlibiSlidingSoftcap window slopes softcap Q K V scale
        (b, h, i, d, PUnit.unit) =
      attentionRealAlibiSlidingSoftcap 0 window (slopes h) softcap
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        scale (i, d, PUnit.unit) := rfl
end FA1Score

end VeriTile.Examples
