/-
VeriTile.Examples.FlashAttention1.Core.Forward

Split-out support for FlashAttention-1 v0/full-tile proofs.
-/

import VeriTile.Examples.FlashAttention1.Core.Steps

namespace VeriTile.Examples

open VeriTile.Triton

/-- Strided readout stage: once `P_fa1_strided numKVBlocks` holds at the
loop exit, the post-loop normalization (`out := o_acc / l_i[:, None]`)
and strided store (`tl.store(outReg + oBase + offs_m * sOM + offs_d *
sOD, out)`) realize the ℝ-level attention spec at the per-`(b, h,
q_block)` slice.

The injectivity hypothesis `hInj` is the standard tile-local
non-overlap requirement on the `[M, D]` output tile. The 4D-wrapper
corollary (issue #39 step (iv)) supplies it via `Offset.strided_inj`
applied to the global `[B, H, S_q, D]` layout — i.e. once the global
strided layout is non-overlapping, the per-instance tile-local view
inherits injectivity for free. -/
theorem fa1_postLoop_correct_strided
    {M D Bk numKVBlocks : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (sLoop : BlockState)
    (hP : P_fa1_strided qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale numKVBlocks sLoop)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (stepStmts (fa1PostLoopStrided outReg M D sOM sOD) sLoop)
          outReg
          (fun idx : TileIndex [M, D] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal Q K V scale idx) := by
  intro idx
  rcases hP with
    ⟨_hpids0, _hpids1, _hpids2,
     _hpid_qb, _hpid_h, _hpid_b,
     _hq_base, _hk_base, _hv_base, ho_base,
     hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hQ, _hK, _hV⟩
  have h_inj_store :
      Function.Injective
        (fun i : TileIndex [M, D] =>
          batch * sOB + headIdx * sOH
            + (qb * M + i.1.val) * sOM + i.2.1.val * sOD) := by
    intro a b h
    apply hInj
    simp only [Nat.add_mul, Nat.add_assoc] at h ⊢
    exact h
  simp [observeTileAt, fa1PostLoopStrided, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho, ho_base,
        Tile.bop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show batch * sOB + headIdx * sOH + qb * M * sOM
        + idx.1.val * sOM + idx.2.1.val * sOD =
      batch * sOB + headIdx * sOH
        + (qb * M + idx.1.val) * sOM + idx.2.1.val * sOD by
    simp [Nat.add_mul, Nat.add_assoc]]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj_store idx]
  simp [FA1Math.streaming_eq_attentionReal hBk Q numKVBlocks hNumKVBlocks K V scale idx
        (FA1Math.lPartial_final_ne_zero hBk Q numKVBlocks hNumKVBlocks K scale idx.1)]

/-- Causal strided readout stage, raw accumulator form. This theorem
closes the operational tail of the causal kernel: assuming the causal
loop invariant at `numKVBlocks`, the post-loop writes
`oPartial / lPartial` to the strided output tile. The final theorem
below composes this with `streaming_eq_attentionRealCausalBlock`. -/
theorem fa1_postLoop_correct_strided_causal_raw
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (sLoop : BlockState)
    (hP : P_fa1_strided_causal qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale numKVBlocks sLoop)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (stepStmts (fa1PostLoopStrided outReg M D sOM sOD) sLoop)
          outReg
          (fun idx : TileIndex [M, D] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausal.oPartial Bk (qb * M) Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathCausal.lPartial Bk (qb * M) Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  intro idx
  rcases hP with
    ⟨_hpids0, _hpids1, _hpids2,
     _hpid_qb, _hpid_h, _hpid_b,
     _hq_base, _hk_base, _hv_base, ho_base,
     hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hQ, _hK, _hV⟩
  have h_inj_store :
      Function.Injective
        (fun i : TileIndex [M, D] =>
          batch * sOB + headIdx * sOH
            + (qb * M + i.1.val) * sOM + i.2.1.val * sOD) := by
    intro a b h
    apply hInj
    simp only [Nat.add_mul, Nat.add_assoc] at h ⊢
    exact h
  simp [observeTileAt, fa1PostLoopStrided, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho, ho_base,
        Tile.bop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show batch * sOB + headIdx * sOH + qb * M * sOM
        + idx.1.val * sOM + idx.2.1.val * sOD =
      batch * sOB + headIdx * sOH
        + (qb * M + idx.1.val) * sOM + idx.2.1.val * sOD by
    simp [Nat.add_mul, Nat.add_assoc]]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj_store idx]

theorem fa1_forward_correct
    {M D Bk numKVBlocks : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (_hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (_hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (_hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernel qReg kReg vReg outReg M D Bk numKVBlocks scale) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some (attentionReal Q K V scale idx) := by
  -- Stage B: pre-loop establishes P_fa1 0.
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct qReg kReg vReg Q K V scale s _hQ _hK _hV
  -- Stage C: forLoop_inv chains fa1_step over numKVBlocks iterations.
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv (P := P_fa1 qReg kReg vReg s.pid Q K V scale) hP0
      (fun i st hi hPi => fa1_step hBk qReg kReg vReg Q K V scale s.pid i st hi hPi)
  -- Stage D: post-loop readout matches `attentionReal`.
  intro idx
  -- Reshape `exec` through body = preLoop ++ [forLoop] ++ postLoop.
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernel qReg kReg vReg outReg M D Bk numKVBlocks scale).toAlgKernel.body =
        fa1PreLoop qReg M D ++
        [Stmt.forLoop "n" numKVBlocks (fa1LoopBody kReg vReg M D Bk scale)] ++
        fa1PostLoop outReg M D from rfl]
  -- Walk preLoop: stepStmts (preLoop ++ [forLoop] ++ postLoop) s
  --             = stepStmts ([forLoop] ++ postLoop) s0
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1PreLoop qReg M D)
        (l2 := [Stmt.forLoop "n" numKVBlocks (fa1LoopBody kReg vReg M D Bk scale)] ++
          fa1PostLoop outReg M D) hPre]
  -- Walk forLoop: stepStmts ([forLoop] ++ postLoop) s0 = stepStmts postLoop sLoop.
  rw [stepStmts.append_some
        (l1 := [Stmt.forLoop "n" numKVBlocks (fa1LoopBody kReg vReg M D Bk scale)])
        (l2 := fa1PostLoop outReg M D) ?_]
  · exact fa1_postLoop_correct hBk hNumKVBlocks qReg kReg vReg outReg s.pid Q K V scale
      sLoop hPLoop idx
  · -- stepStmts [forLoop] s0 = stepStmts [] sLoop = some sLoop
    rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil

/-- Strided / 4D-aware FA-1 forward correctness — single program-instance
slice. Threads `fa1_preLoop_correct_strided`, `fa1_step_strided`, and
`fa1_postLoop_correct_strided` through `forLoop_inv` exactly the way
`fa1_forward_correct` does for the 2D kernel. The output equals
`attentionReal` on the per-`(b, h, q_block)` slice; the 4D wrapper
(issue #39 step (iv)) lifts this to `attentionReal4D` via
`attentionReal4D_slice`.

The boundary / non-overlap requirement that 4D Triton FA-1 lives or
dies on (`qb*M + (M-1) < S_q`, plus `Σ (d-1)*s < next stride` along
each axis of `[B, H, S_q, D]`) is folded entirely into the readout
injectivity hypothesis `hInj` here — kept abstract so a 4D-wrapper
caller can package it via `Offset.strided_inj` + `StridesValid`, and a
2D-equivalent caller (B = H = 1, `sOB = sOH = 0`, `sOM = D`,
`sOD = 1`) can discharge it directly. -/
theorem fa1_forward_correct_strided
    {M D Bk numKVBlocks : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (fun idx : TileIndex [M, D] =>
          s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) Q)
    (hK : InputAt s kReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStrided qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal Q K V scale idx) := by
  -- Stage B: strided pre-loop establishes P_fa1_strided 0.
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct_strided qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQ hK hV
  -- Stage C: forLoop_inv chains fa1_step_strided.
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv
      (P := P_fa1_strided qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale) hP0
      (fun i st hi hPi =>
        fa1_step_strided hBk qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st hi hPi)
  -- Stage D: strided post-loop readout.
  intro idx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernelStrided qReg kReg vReg outReg M D Bk numKVBlocks
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).toAlgKernel.body =
        fa1PreLoopStrided qReg M D sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
        [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStrided kReg vReg M D Bk sKN sKD sVN sVD scale)] ++
        fa1PostLoopStrided outReg M D sOM sOD from rfl]
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1PreLoopStrided qReg M D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH)
        (l2 := [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStrided kReg vReg M D Bk sKN sKD sVN sVD scale)] ++
          fa1PostLoopStrided outReg M D sOM sOD) hPre]
  have hLoopList :
      stepStmts [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStrided kReg vReg M D Bk sKN sKD sVN sVD scale)]
        s0 = some sLoop := by
    rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil
  rw [stepStmts.append_some hLoopList]
  · exact fa1_postLoop_correct_strided hBk hNumKVBlocks
      qReg kReg vReg outReg
      (s.pids 0) (s.pids 1) (s.pids 2)
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale sLoop hPLoop hInj idx

/-- Causal strided forward correctness in raw streaming form, parameterized
by the causal loop-step lemma. Kept as a factoring lemma for the closed
theorem `fa1_forward_correct_strided_causal_raw`, which supplies
`fa1_step_strided_causal` directly. -/
theorem fa1_forward_correct_strided_causal_raw_of_step
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (fun idx : TileIndex [M, D] =>
          s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) Q)
    (hK : InputAt s kReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD))
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_causal qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedCausal kReg vReg M D Bk sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_causal qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD Q K V scale (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausal.oPartial Bk (s.pids 0 * M) Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathCausal.lPartial Bk (s.pids 0 * M) Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct_strided_causal qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQ hK hV
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv
      (P := P_fa1_strided_causal qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale) hP0 hStep
  intro idx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).toAlgKernel.body =
        fa1PreLoopStrided qReg M D sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
        [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausal kReg vReg M D Bk sKN sKD sVN sVD scale)] ++
        fa1PostLoopStrided outReg M D sOM sOD from rfl]
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1PreLoopStrided qReg M D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH)
        (l2 := [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausal kReg vReg M D Bk sKN sKD sVN sVD scale)] ++
          fa1PostLoopStrided outReg M D sOM sOD) hPre]
  have hLoopList :
      stepStmts [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausal kReg vReg M D Bk sKN sKD sVN sVD scale)]
        s0 = some sLoop := by
    rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil
  rw [stepStmts.append_some hLoopList]
  · exact fa1_postLoop_correct_strided_causal_raw
      qReg kReg vReg outReg
      (s.pids 0) (s.pids 1) (s.pids 2)
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale sLoop hPLoop hInj idx

/-- Causal strided forward correctness in raw streaming form. This is
`fa1_forward_correct_strided_causal_raw_of_step` with the loop-step
obligation discharged by `fa1_step_strided_causal`. The remaining
math bridge to the user-facing causal attention spec is handled by
the 4D theorem layer below. -/
theorem fa1_forward_correct_strided_causal_raw
    {M D Bk numKVBlocks : Nat}
    (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (fun idx : TileIndex [M, D] =>
          s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) Q)
    (hK : InputAt s kReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausal.oPartial Bk (s.pids 0 * M) Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathCausal.lPartial Bk (s.pids 0 * M) Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  exact fa1_forward_correct_strided_causal_raw_of_step
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q K V scale s hQ hK hV hInj
    (fun i st hi hPi =>
      fa1_step_strided_causal hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale i st hi hPi)

/-- Causal strided forward correctness, stated against the local-block
causal attention spec rather than the raw streaming accumulator ratio. -/
theorem fa1_forward_correct_strided_causal
    {M D Bk numKVBlocks : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (fun idx : TileIndex [M, D] =>
          s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) Q)
    (hK : InputAt s kReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionRealCausalBlock (s.pids 0 * M) Q K V scale idx) := by
  intro idx
  rw [fa1_forward_correct_strided_causal_raw hBk
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale s hQ hK hV hInj idx]
  congr 1
  exact FA1MathCausal.streaming_eq_attentionRealCausalBlock hBk
    (s.pids 0 * M) Q numKVBlocks hNumKVBlocks K V scale idx


end VeriTile.Examples
