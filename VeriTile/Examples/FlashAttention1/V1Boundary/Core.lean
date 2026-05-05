/-
VeriTile.Examples.FlashAttention1.V1Boundary.Core

Split-out support for FlashAttention-1 v1 boundary proofs.
-/

import VeriTile.Examples.FlashAttention1.V1Boundary.Helpers

namespace VeriTile.Examples

open VeriTile.Triton

/-- Boundary-masked strided initialization stage. The Q register is loaded
with the same mask as the v1 kernel: in-bounds rows come from memory, while
out-of-bounds rows are the explicit `other=0`. After that, the loop invariant
only keeps the loaded Q tile and the K/V memory contracts. -/
theorem fa1_preLoop_correct_strided_boundary
    {M D Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V) :
    ∃ s0,
      stepStmts (fa1PreLoopStridedBoundary qReg M D S_q
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH) s = some s0 ∧
      P_fa1_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q K V scale 0 s0 := by
  let qBase : Nat := s.pids 2 * sQB + s.pids 1 * sQH
  let qPtrs : Tile .nat [M, D] :=
    ⟨fun idx => qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD⟩
  let qMask : Tile .bool [M, D] :=
    ⟨fun idx => s.pids 0 * M + idx.1.val < S_q⟩
  let qLoaded : Tile .real [M, D] :=
    ⟨fun idx =>
      if h : s.pids 0 * M + idx.1.val < S_q then
        some (s.readMem qReg
          (qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD))
      else
        some 0⟩
  let s0 :=
    (((((((((((((((s.setReg "pid_qb" .nat [] (Tile.scalar (s.pids 0)))
      ).setReg "pid_h" .nat [] (Tile.scalar (s.pids 1))
      ).setReg "pid_b" .nat [] (Tile.scalar (s.pids 2))
      ).setReg "q_base_off" .nat []
        (Tile.scalar (s.pids 2 * sQB + s.pids 1 * sQH))
      ).setReg "k_base_off" .nat []
        (Tile.scalar (s.pids 2 * sKB + s.pids 1 * sKH))
      ).setReg "v_base_off" .nat []
        (Tile.scalar (s.pids 2 * sVB + s.pids 1 * sVH))
      ).setReg "o_base_off" .nat []
        (Tile.scalar (s.pids 2 * sOB + s.pids 1 * sOH))
      ).setReg "offs_m" .nat [M]
        (Tile.vec fun i : Fin M => s.pids 0 * M + i.val)
      ).setReg "offs_d" .nat [D]
        (Tile.vec fun d : Fin D => d.val)
      ).setReg "q_ptrs" .nat [M, D] qPtrs
      ).setReg "q_mask" .bool [M, D] qMask
      ).setReg "q" .real [M, D] qLoaded
      ).setReg "m_i" .real [M] ⟨fun _ => (⊥ : WithBot ℝ)⟩
      ).setReg "l_i" .real [M] (Tile.ofReal fun _ => 0)
      ).setReg "o_acc" .real [M, D] (Tile.ofReal fun _ => 0)
  have hQ_loaded_eq : qLoaded = Tile.ofReal Q := by
    ext idx
    rw [Tile.ofReal_data]
    by_cases h : s.pids 0 * M + idx.1.val < S_q
    · simp [qLoaded, qBase, h]
      rw [show qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD =
          s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD by
          simp [qBase, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]]
      exact congrArg some (hQIn idx h)
    · simp [qLoaded, h, hQOut idx h]
  refine ⟨s0, ?_, ?_⟩
  · simp [fa1PreLoopStridedBoundary, stepStmts, stepStmt, evalOp,
      Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, Option.bind, TileShape.dropInsertedIndex, Tile.vec, Tile.ofReal, qPtrs, qMask, qLoaded, qBase, s0]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0, hQ_loaded_eq]
    · simp [s0, FA1MathBoundary.mPartial]
    · simp [s0, FA1MathBoundary.lPartial, Tile.ofReal]
    · simp [s0, FA1MathBoundary.oPartial, Tile.ofReal]
    · intro idx
      simpa [s0] using hK idx
    · intro idx
      simpa [s0] using hV idx

/-- D-tail boundary-masked strided initialization stage. The loaded Q tile is
the logical Q block padded to block width `Bd`; row-tail lanes come from
`hQOut`, and hidden-dimension tail lanes come from the D mask's `other=0`. -/
theorem fa1_preLoop_correct_strided_boundaryD
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V) :
    ∃ s0,
      stepStmts (fa1PreLoopStridedBoundaryD qReg M Bd S_q D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH) s = some s0 ∧
      P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q K V scale 0 s0 := by
  let qBase : Nat := s.pids 2 * sQB + s.pids 1 * sQH
  let qPtrs : Tile .nat [M, Bd] :=
    ⟨fun idx => qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD⟩
  let qSeqMask : Tile .bool [M, Bd] :=
    ⟨fun idx => s.pids 0 * M + idx.1.val < S_q⟩
  let qDMask : Tile .bool [M, Bd] :=
    ⟨fun idx => idx.2.1.val < D⟩
  let qMask : Tile .bool [M, Bd] :=
    ⟨fun idx => (s.pids 0 * M + idx.1.val < S_q) && (idx.2.1.val < D)⟩
  let qLoaded : Tile .real [M, Bd] :=
    ⟨fun idx =>
      if h : (s.pids 0 * M + idx.1.val < S_q) ∧ (idx.2.1.val < D) then
        some (s.readMem qReg
          (qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD))
      else
        some 0⟩
  let s0 :=
    (((((((((((((((((s.setReg "pid_qb" .nat [] (Tile.scalar (s.pids 0)))
      ).setReg "pid_h" .nat [] (Tile.scalar (s.pids 1))
      ).setReg "pid_b" .nat [] (Tile.scalar (s.pids 2))
      ).setReg "q_base_off" .nat []
        (Tile.scalar (s.pids 2 * sQB + s.pids 1 * sQH))
      ).setReg "k_base_off" .nat []
        (Tile.scalar (s.pids 2 * sKB + s.pids 1 * sKH))
      ).setReg "v_base_off" .nat []
        (Tile.scalar (s.pids 2 * sVB + s.pids 1 * sVH))
      ).setReg "o_base_off" .nat []
        (Tile.scalar (s.pids 2 * sOB + s.pids 1 * sOH))
      ).setReg "offs_m" .nat [M]
        (Tile.vec fun i : Fin M => s.pids 0 * M + i.val)
      ).setReg "offs_d" .nat [Bd]
        (Tile.vec fun d : Fin Bd => d.val)
      ).setReg "q_ptrs" .nat [M, Bd] qPtrs
      ).setReg "q_seq_mask" .bool [M, Bd] qSeqMask
      ).setReg "q_d_mask" .bool [M, Bd] qDMask
      ).setReg "q_mask" .bool [M, Bd] qMask
      ).setReg "q" .real [M, Bd] qLoaded
      ).setReg "m_i" .real [M] ⟨fun _ => (⊥ : WithBot ℝ)⟩
      ).setReg "l_i" .real [M] (Tile.ofReal fun _ => 0)
      ).setReg "o_acc" .real [M, Bd] (Tile.ofReal fun _ => 0)
  have hQ_loaded_eq : qLoaded = Tile.ofReal (padHeadD (Bd := Bd) Q) := by
    ext idx
    rw [Tile.ofReal_data]
    by_cases hD : idx.2.1.val < D
    · by_cases hRow : s.pids 0 * M + idx.1.val < S_q
      · have hBoth : (s.pids 0 * M + idx.1.val < S_q) ∧ (idx.2.1.val < D) :=
          ⟨hRow, hD⟩
        simp [qLoaded, qBase, hBoth, padHeadD]
        rw [show qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD =
            s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
              + idx.1.val * sQS + idx.2.1.val * sQD by
            simp [qBase, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]]
        exact congrArg some (hQIn (idx.1, ⟨idx.2.1.val, hD⟩, PUnit.unit) hRow)
      · have hNotBoth : ¬ ((s.pids 0 * M + idx.1.val < S_q) ∧ (idx.2.1.val < D)) := by
          intro h; exact hRow h.1
        rw [show qLoaded.data idx = some 0 by
          simp [qLoaded, hNotBoth]]
        simp [padHeadD, hD, hQOut (idx.1, ⟨idx.2.1.val, hD⟩, PUnit.unit) hRow]
    · have hNotBoth : ¬ ((s.pids 0 * M + idx.1.val < S_q) ∧ (idx.2.1.val < D)) := by
        intro h; exact hD h.2
      rw [show qLoaded.data idx = some 0 by
        simp [qLoaded, hNotBoth]]
      simp [padHeadD, hD]
  refine ⟨s0, ?_, ?_⟩
  · simp [fa1PreLoopStridedBoundaryD, stepStmts, stepStmt, evalOp,
      Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, Option.bind, TileShape.dropInsertedIndex, Tile.vec, Tile.ofReal,
      qPtrs, qSeqMask, qDMask, qMask, qLoaded, qBase, s0]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0, hQ_loaded_eq]
    · simp [s0, FA1MathBoundary.mPartial]
    · simp [s0, FA1MathBoundary.lPartial, Tile.ofReal]
    · simp [s0, FA1MathBoundary.oPartial, Tile.ofReal]
    · intro idx
      simpa [s0] using hK idx
    · intro idx
      simpa [s0] using hV idx

/-- D-tail causal-boundary initialization stage. Operationally identical to
`fa1_preLoop_correct_strided_boundaryD`; only the invariant's accumulator
namespace changes at `k = 0`. -/
theorem fa1_preLoop_correct_strided_causal_boundaryD
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V) :
    ∃ s0,
      stepStmts (fa1PreLoopStridedBoundaryD qReg M Bd S_q D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH) s = some s0 ∧
      P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk)
        (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q K V scale 0 s0 := by
  rcases fa1_preLoop_correct_strided_boundaryD (Bd := Bd) (Bk := Bk)
      (numKVBlocks := numKVBlocks)
      qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV with
    ⟨s0, hExec, hP⟩
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hK0, hV0⟩
  refine ⟨s0, hExec, ?_⟩
  refine ⟨hpids0, hpids1, hpids2,
    hpid_qb, hpid_h, hpid_b, hq_base, hk_base, hv_base, ho_base,
    hoffs_m, hoffs_d, hq, ?_, ?_, ?_, hK0, hV0⟩
  · simpa [FA1MathBoundary.mPartial, FA1MathCausalBoundary.mPartial] using hm
  · simpa [FA1MathBoundary.lPartial, FA1MathCausalBoundary.lPartial] using hl
  · simpa [FA1MathBoundary.oPartial, FA1MathCausalBoundary.oPartial] using ho

/-- Causal boundary-masked strided initialization stage. Operationally this
is the same pre-loop as the non-causal boundary kernel; only the loop
invariant's accumulator interpretation changes to `FA1MathCausalBoundary`,
whose `k = 0` state is the same `(-inf, 0, 0)` initialization. -/
theorem fa1_preLoop_correct_strided_causal_boundary
    {M D Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V) :
    ∃ s0,
      stepStmts (fa1PreLoopStridedBoundary qReg M D S_q
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH) s = some s0 ∧
      P_fa1_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q K V scale 0 s0 := by
  rcases fa1_preLoop_correct_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
      qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV with
    ⟨s0, hExec, hP⟩
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hK0, hV0⟩
  refine ⟨s0, hExec, ?_⟩
  refine ⟨hpids0, hpids1, hpids2,
    hpid_qb, hpid_h, hpid_b, hq_base, hk_base, hv_base, ho_base,
    hoffs_m, hoffs_d, hq, ?_, ?_, ?_, hK0, hV0⟩
  · simpa [FA1MathBoundary.mPartial, FA1MathCausalBoundary.mPartial] using hm
  · simpa [FA1MathBoundary.lPartial, FA1MathCausalBoundary.lPartial] using hl
  · simpa [FA1MathBoundary.oPartial, FA1MathCausalBoundary.oPartial] using ho
theorem fa1_postLoop_correct_strided_boundary_raw
    {M D Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (sLoop : BlockState)
    (hP : P_fa1_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale numKVBlocks sLoop)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      qb * M + idx.1.val < S_q →
      observeTileAt
          (stepStmts (fa1PostLoopStridedBoundary outReg M D S_q sOM sOD) sLoop)
          outReg
          (fun idx : TileIndex [M, D] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathBoundary.lPartial Bk Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  intro idx hIdx
  rcases hP with
    ⟨_hpids0, _hpids1, _hpids2,
     _hpid_qb, _hpid_h, _hpid_b,
     _hq_base, _hk_base, _hv_base, ho_base,
     hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hK, _hV⟩
  have h_inj_store :
      Function.Injective
        (fun i : TileIndex [M, D] =>
          batch * sOB + headIdx * sOH
            + (qb * M + i.1.val) * sOM + i.2.1.val * sOD) := by
    intro a b h
    apply hInj
    simp only [Nat.add_mul, Nat.add_assoc] at h ⊢
    exact h
  simp [observeTileAt, fa1PostLoopStridedBoundary, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho, ho_base,
        Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show batch * sOB + headIdx * sOH + qb * M * sOM
        + idx.1.val * sOM + idx.2.1.val * sOD =
      batch * sOB + headIdx * sOH
        + (qb * M + idx.1.val) * sOM + idx.2.1.val * sOD by
    simp [Nat.add_mul, Nat.add_assoc]]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj_store idx]
  simp [hIdx]

/-- Causal boundary strided readout stage, raw accumulator form. For
in-bounds query rows (`qb*M + i < S_q`), the masked store writes the
normalized causal-boundary streaming accumulator. -/
theorem fa1_postLoop_correct_strided_causal_boundary_raw
    {M D Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (sLoop : BlockState)
    (hP : P_fa1_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale numKVBlocks sLoop)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, D],
      qb * M + idx.1.val < S_q →
      observeTileAt
          (stepStmts (fa1PostLoopStridedBoundary outReg M D S_q sOM sOD) sLoop)
          outReg
          (fun idx : TileIndex [M, D] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausalBoundary.oPartial Bk (qb * M) Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathCausalBoundary.lPartial Bk (qb * M) Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  intro idx hIdx
  rcases hP with
    ⟨_hpids0, _hpids1, _hpids2,
     _hpid_qb, _hpid_h, _hpid_b,
     _hq_base, _hk_base, _hv_base, ho_base,
     hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hK, _hV⟩
  have h_inj_store :
      Function.Injective
        (fun i : TileIndex [M, D] =>
          batch * sOB + headIdx * sOH
            + (qb * M + i.1.val) * sOM + i.2.1.val * sOD) := by
    intro a b h
    apply hInj
    simp only [Nat.add_mul, Nat.add_assoc] at h ⊢
    exact h
  simp [observeTileAt, fa1PostLoopStridedBoundary, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho, ho_base,
        Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show batch * sOB + headIdx * sOH + qb * M * sOM
        + idx.1.val * sOM + idx.2.1.val * sOD =
      batch * sOB + headIdx * sOH
        + (qb * M + idx.1.val) * sOM + idx.2.1.val * sOD by
    simp [Nat.add_mul, Nat.add_assoc]]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj_store idx]
  simp [hIdx]

/-- D-tail boundary readout stage, raw accumulator form. A D-tail store
writes only lanes satisfying both the Q-row boundary and `d < D`. -/
theorem fa1_postLoop_correct_strided_boundaryD_raw
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (sLoop : BlockState)
    (hP : P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk)
        (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale numKVBlocks sLoop)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      qb * M + idx.1.val < S_q →
      idx.2.1.val < D →
      observeTileAt
          (stepStmts (fa1PostLoopStridedBoundaryD outReg M Bd S_q D sOM sOD) sLoop)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathBoundary.oPartial Bk
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
                scale numKVBlocks idx /
              FA1MathBoundary.lPartial Bk
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) scale numKVBlocks idx.1) := by
  intro idx hRow hD
  dsimp [P_fa1_strided_boundaryD] at hP
  rcases hP with
    ⟨_hpids0, _hpids1, _hpids2,
     _hpid_qb, _hpid_h, _hpid_b,
     _hq_base, _hk_base, _hv_base, ho_base,
     hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hK, _hV⟩
  have h_no_collision :
      ∀ k : TileIndex [M, Bd],
        (qb * M + k.1.val < S_q ∧ k.2.1.val < D) →
        batch * sOB + headIdx * sOH + (qb * M + k.1.val) * sOM
            + k.2.1.val * sOD =
          batch * sOB + headIdx * sOH + (qb * M + idx.1.val) * sOM
            + idx.2.1.val * sOD →
        k = idx := by
    intro k hk heq
    obtain ⟨ki, kd, ku⟩ := k
    obtain ⟨ii, id, iu⟩ := idx
    cases ku
    cases iu
    have hEqLogical :
        (⟨ki, ⟨kd.val, hk.2⟩, PUnit.unit⟩ : TileIndex [M, D]) =
          (⟨ii, ⟨id.val, hD⟩, PUnit.unit⟩ : TileIndex [M, D]) := by
      apply hInj
      simpa [Nat.add_mul, Nat.add_assoc] using heq
    have hRowEq : ki = ii := by
      simpa using congrArg (fun x : TileIndex [M, D] => x.1) hEqLogical
    have hDEq : kd.val = id.val := by
      simpa using congrArg (fun x : TileIndex [M, D] => x.2.1.val) hEqLogical
    cases hRowEq
    cases Fin.ext hDEq
    rfl
  simp [observeTileAt, fa1PostLoopStridedBoundaryD, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho, ho_base,
        Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show batch * sOB + headIdx * sOH + qb * M * sOM
        + idx.1.val * sOM + idx.2.1.val * sOD =
      batch * sOB + headIdx * sOH
        + (qb * M + idx.1.val) * sOM + idx.2.1.val * sOD by
    simp [Nat.add_mul, Nat.add_assoc]]
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ _ idx
        (by exact ⟨hRow, hD⟩) h_no_collision]

/-- D-tail causal-boundary readout stage, raw accumulator form. -/
theorem fa1_postLoop_correct_strided_causal_boundaryD_raw
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (sLoop : BlockState)
    (hP : P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk)
        (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale numKVBlocks sLoop)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        batch * sOB + headIdx * sOH + qb * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      qb * M + idx.1.val < S_q →
      idx.2.1.val < D →
      observeTileAt
          (stepStmts (fa1PostLoopStridedBoundaryD outReg M Bd S_q D sOM sOD) sLoop)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            batch * sOB + headIdx * sOH + qb * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausalBoundary.oPartial Bk (qb * M)
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
                scale numKVBlocks idx /
              FA1MathCausalBoundary.lPartial Bk (qb * M)
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) scale numKVBlocks idx.1) := by
  intro idx hRow hD
  dsimp [P_fa1_strided_causal_boundaryD] at hP
  rcases hP with
    ⟨_hpids0, _hpids1, _hpids2,
     _hpid_qb, _hpid_h, _hpid_b,
     _hq_base, _hk_base, _hv_base, ho_base,
     hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hK, _hV⟩
  have h_no_collision :
      ∀ k : TileIndex [M, Bd],
        (qb * M + k.1.val < S_q ∧ k.2.1.val < D) →
        batch * sOB + headIdx * sOH + (qb * M + k.1.val) * sOM
            + k.2.1.val * sOD =
          batch * sOB + headIdx * sOH + (qb * M + idx.1.val) * sOM
            + idx.2.1.val * sOD →
        k = idx := by
    intro k hk heq
    obtain ⟨ki, kd, ku⟩ := k
    obtain ⟨ii, id, iu⟩ := idx
    cases ku
    cases iu
    have hEqLogical :
        (⟨ki, ⟨kd.val, hk.2⟩, PUnit.unit⟩ : TileIndex [M, D]) =
          (⟨ii, ⟨id.val, hD⟩, PUnit.unit⟩ : TileIndex [M, D]) := by
      apply hInj
      simpa [Nat.add_mul, Nat.add_assoc] using heq
    have hRowEq : ki = ii := by
      simpa using congrArg (fun x : TileIndex [M, D] => x.1) hEqLogical
    have hDEq : kd.val = id.val := by
      simpa using congrArg (fun x : TileIndex [M, D] => x.2.1.val) hEqLogical
    cases hRowEq
    cases Fin.ext hDEq
    rfl
  simp [observeTileAt, fa1PostLoopStridedBoundaryD, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho, ho_base,
        Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show batch * sOB + headIdx * sOH + qb * M * sOM
        + idx.1.val * sOM + idx.2.1.val * sOD =
      batch * sOB + headIdx * sOH
        + (qb * M + idx.1.val) * sOM + idx.2.1.val * sOD by
    simp [Nat.add_mul, Nat.add_assoc]]
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ _ idx
        (by exact ⟨hRow, hD⟩) h_no_collision]

/-- Boundary strided forward correctness in raw streaming form, parameterized
by the boundary loop-step lemma. This is the v1 analogue of
`fa1_forward_correct_strided_causal_raw_of_step`: pre-loop establishes the
boundary invariant, `forLoop_inv` consumes the supplied step theorem, and the
masked post-loop readout gives the raw `oPartial / lPartial` value for
in-bounds query lanes. -/
theorem fa1_forward_correct_strided_boundary_raw_of_step
    {M D Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD))
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
          qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
            qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD Q K V scale (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      s.pids 0 * M + idx.1.val < S_q →
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathBoundary.lPartial Bk Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct_strided_boundary qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv
      (P := P_fa1_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale) hP0 hStep
  intro idx hIdx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernelStridedBoundary qReg kReg vReg outReg
        M D Bk numKVBlocks S_q S_k
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).toAlgKernel.body =
        fa1PreLoopStridedBoundary qReg M D S_q
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
        [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)] ++
        fa1PostLoopStridedBoundary outReg M D S_q sOM sOD from rfl]
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1PreLoopStridedBoundary qReg M D S_q
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH)
        (l2 := [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)] ++
          fa1PostLoopStridedBoundary outReg M D S_q sOM sOD) hPre]
  have hLoopList :
      stepStmts [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)]
        s0 = some sLoop := by
    rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil
  rw [stepStmts.append_some hLoopList]
  · exact fa1_postLoop_correct_strided_boundary_raw
      qReg kReg vReg outReg
      (s.pids 0) (s.pids 1) (s.pids 2)
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale sLoop hPLoop hInj idx hIdx

/-- D-tail boundary strided forward correctness in raw streaming form,
parameterized by the D-tail boundary loop-step lemma. -/
theorem fa1_forward_correct_strided_boundaryD_raw_of_step
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD))
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
          qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
            qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD Q K V scale (i + 1) st') :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
      idx.2.1.val < D →
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathBoundary.oPartial Bk
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
                scale numKVBlocks idx /
              FA1MathBoundary.lPartial Bk
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) scale numKVBlocks idx.1) := by
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct_strided_boundaryD (Bd := Bd) (Bk := Bk)
      (numKVBlocks := numKVBlocks)
      qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv
      (P := P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale) hP0 hStep
  intro idx hRow hD
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernelStridedBoundaryD qReg kReg vReg outReg
        M Bd Bk numKVBlocks S_q S_k D
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).toAlgKernel.body =
        fa1PreLoopStridedBoundaryD qReg M Bd S_q D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
        [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)] ++
        fa1PostLoopStridedBoundaryD outReg M Bd S_q D sOM sOD from rfl]
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1PreLoopStridedBoundaryD qReg M Bd S_q D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH)
        (l2 := [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)] ++
          fa1PostLoopStridedBoundaryD outReg M Bd S_q D sOM sOD) hPre]
  have hLoopList :
      stepStmts [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)]
        s0 = some sLoop := by
    rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil
  rw [stepStmts.append_some hLoopList]
  · exact fa1_postLoop_correct_strided_boundaryD_raw
      qReg kReg vReg outReg
      (s.pids 0) (s.pids 1) (s.pids 2)
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale sLoop hPLoop hInj idx hRow hD

/-- Causal boundary strided forward correctness in raw streaming form,
parameterized by the causal-boundary loop-step lemma. -/
theorem fa1_forward_correct_strided_causal_boundary_raw_of_step
    {M D Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD))
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
          qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedCausalBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
            qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD Q K V scale (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      s.pids 0 * M + idx.1.val < S_q →
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausalBoundary.oPartial Bk (s.pids 0 * M) Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathCausalBoundary.lPartial Bk (s.pids 0 * M) Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
      qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv
      (P := P_fa1_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale) hP0 hStep
  intro idx hIdx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernelStridedCausalBoundary qReg kReg vReg outReg
        M D Bk numKVBlocks S_q S_k
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).toAlgKernel.body =
        fa1PreLoopStridedBoundary qReg M D S_q
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
        [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausalBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)] ++
        fa1PostLoopStridedBoundary outReg M D S_q sOM sOD from rfl]
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1PreLoopStridedBoundary qReg M D S_q
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH)
        (l2 := [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausalBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)] ++
          fa1PostLoopStridedBoundary outReg M D S_q sOM sOD) hPre]
  have hLoopList :
      stepStmts [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausalBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)]
        s0 = some sLoop := by
    rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil
  rw [stepStmts.append_some hLoopList]
  · exact fa1_postLoop_correct_strided_causal_boundary_raw
      qReg kReg vReg outReg
      (s.pids 0) (s.pids 1) (s.pids 2)
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale sLoop hPLoop hInj idx hIdx

/-- D-tail causal-boundary strided forward correctness in raw streaming form,
parameterized by the D-tail causal-boundary loop-step lemma. -/
theorem fa1_forward_correct_strided_causal_boundaryD_raw_of_step
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD))
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
          qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedCausalBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
            qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD Q K V scale (i + 1) st') :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
      idx.2.1.val < D →
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausalBoundary.oPartial Bk (s.pids 0 * M)
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
                scale numKVBlocks idx /
              FA1MathCausalBoundary.lPartial Bk (s.pids 0 * M)
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) scale numKVBlocks idx.1) := by
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_preLoop_correct_strided_causal_boundaryD (Bd := Bd) (Bk := Bk)
      (numKVBlocks := numKVBlocks)
      qReg kReg vReg
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv
      (P := P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale) hP0 hStep
  intro idx hRow hD
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [show (fa1ForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
        M Bd Bk numKVBlocks S_q S_k D
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD scale).toAlgKernel.body =
        fa1PreLoopStridedBoundaryD qReg M Bd S_q D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH ++
        [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausalBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)] ++
        fa1PostLoopStridedBoundaryD outReg M Bd S_q D sOM sOD from rfl]
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1PreLoopStridedBoundaryD qReg M Bd S_q D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH)
        (l2 := [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausalBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)] ++
          fa1PostLoopStridedBoundaryD outReg M Bd S_q D sOM sOD) hPre]
  have hLoopList :
      stepStmts [Stmt.forLoop "n" numKVBlocks
          (fa1LoopBodyStridedCausalBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)]
        s0 = some sLoop := by
    rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil
  rw [stepStmts.append_some hLoopList]
  · exact fa1_postLoop_correct_strided_causal_boundaryD_raw
      qReg kReg vReg outReg
      (s.pids 0) (s.pids 1) (s.pids 2)
      sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
      sOB sOH sOM sOD Q K V scale sLoop hPLoop hInj idx hRow hD

/-- D-tail boundary strided forward correctness in canonical spec form,
parameterized by the D-tail boundary loop-step lemma. -/
theorem fa1_forward_correct_strided_boundaryD_of_step
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD))
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
          qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
            qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD Q K V scale (i + 1) st') :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal Q K V scale
            (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hRow hDIdx
  rw [fa1_forward_correct_strided_boundaryD_raw_of_step
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj hStep idx hRow hDIdx]
  congr 1
  have hL := FA1MathBoundary.lPartial_final_ne_zero hBk hSk
    (padHeadD (Bd := Bd) Q) numKVBlocks (padHeadD (Bd := Bd) K)
    scale hSkLe idx.1
  rw [FA1MathBoundary.streaming_eq_attentionReal hBk
      (padHeadD (Bd := Bd) Q) hSkLe
      (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V) scale idx hL]
  exact FA1Math.attentionReal_padHeadD_eq hDLe Q K V scale idx hDIdx

/-- D-tail causal-boundary strided forward correctness in canonical spec
form, parameterized by the D-tail causal-boundary loop-step lemma. -/
theorem fa1_forward_correct_strided_causal_boundaryD_of_step
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD))
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
          qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD Q K V scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedCausalBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk) (numKVBlocks := numKVBlocks)
            qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD Q K V scale (i + 1) st') :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionRealCausalBlock (s.pids 0 * M) Q K V scale
            (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hRow hDIdx
  rw [fa1_forward_correct_strided_causal_boundaryD_raw_of_step
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj hStep idx hRow hDIdx]
  congr 1
  rw [FA1MathCausalBoundary.streaming_eq_attentionRealCausalBlock hBk hSk hSkLe
      (s.pids 0 * M)
      (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K)
      (padHeadD (Bd := Bd) V) scale idx]
  exact FA1Math.attentionRealCausalBlock_padHeadD_eq hDLe (s.pids 0 * M) Q K V scale idx hDIdx


end VeriTile.Examples
