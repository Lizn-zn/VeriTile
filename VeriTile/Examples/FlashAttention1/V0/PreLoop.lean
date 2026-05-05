/-
VeriTile.Examples.FlashAttention1.V0.PreLoop

Split-out support for FlashAttention-1 v0/full-tile proofs.
-/

import VeriTile.Examples.FlashAttention1.V0.Bodies

namespace VeriTile.Examples

open VeriTile.Triton

/-- Boundary-masked strided KV block load. Valid local lanes read the logical
`[S_k, D]` tensor through `blockIndex?`; invalid padded lanes are exactly the
explicit `other = 0` value from the Triton load. -/
theorem fa1_preLoop_correct_strided
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (s : BlockState)
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
            + idx.1.val * sVN + idx.2.1.val * sVD) V) :
    ∃ s0,
      stepStmts (fa1PreLoopStrided qReg M D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH) s = some s0 ∧
      P_fa1_strided qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q K V scale 0 s0 := by
  let qBase : Nat := s.pids 2 * sQB + s.pids 1 * sQH
  let qPtrs : Tile .nat [M, D] :=
    ⟨fun idx => qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD⟩
  let qLoaded : Tile .real [M, D] :=
    ⟨fun idx => some (s.readMem qReg
      (qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD))⟩
  let s0 :=
    ((((((((((((((s.setReg "pid_qb" .nat [] (Tile.scalar (s.pids 0)))
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
      ).setReg "q" .real [M, D] qLoaded
      ).setReg "m_i" .real [M] ⟨fun _ => (⊥ : WithBot ℝ)⟩
      ).setReg "l_i" .real [M] (Tile.ofReal fun _ => 0)
      ).setReg "o_acc" .real [M, D] (Tile.ofReal fun _ => 0)
  have hQ_loaded_eq : qLoaded = Tile.ofReal Q := by
    ext idx
    simp [qLoaded, qBase, Tile.ofReal]
    rw [show qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD =
        s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + idx.1.val * sQS + idx.2.1.val * sQD by
          simp [qBase, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]]
    exact congrArg some (hQ idx)
  refine ⟨s0, ?_, ?_⟩
  · simp [fa1PreLoopStrided, stepStmts, stepStmt, evalOp, Tile.bop, Tile.expandDim,
      NumericDType.add, NumericDType.mul, Option.bind, TileShape.dropInsertedIndex, Tile.vec, Tile.ofReal, qPtrs, qLoaded, qBase, s0]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
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
    · simp [s0, FA1Math.mPartial]
    · simp [s0, FA1Math.lPartial, Tile.ofReal]
    · simp [s0, FA1Math.oPartial, Tile.ofReal]
    · intro idx
      simpa [s0] using hQ idx
    · intro idx
      simpa [s0] using hK idx
    · intro idx
      simpa [s0] using hV idx

/-- Causal strided initialization stage. The executable pre-loop is
identical to the non-causal strided pre-loop; only the target invariant
uses the causal streaming recurrence. At iteration `0`, both
recurrences initialize to `m_i = -inf`, `l_i = 0`, and `o_acc = 0`. -/
theorem fa1_preLoop_correct_strided_causal
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (s : BlockState)
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
            + idx.1.val * sVN + idx.2.1.val * sVD) V) :
    ∃ s0,
      stepStmts (fa1PreLoopStrided qReg M D
          sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH) s = some s0 ∧
      P_fa1_strided_causal qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q K V scale 0 s0 := by
  let qBase : Nat := s.pids 2 * sQB + s.pids 1 * sQH
  let qPtrs : Tile .nat [M, D] :=
    ⟨fun idx => qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD⟩
  let qLoaded : Tile .real [M, D] :=
    ⟨fun idx => some (s.readMem qReg
      (qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD))⟩
  let s0 :=
    ((((((((((((((s.setReg "pid_qb" .nat [] (Tile.scalar (s.pids 0)))
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
      ).setReg "q" .real [M, D] qLoaded
      ).setReg "m_i" .real [M] ⟨fun _ => (⊥ : WithBot ℝ)⟩
      ).setReg "l_i" .real [M] (Tile.ofReal fun _ => 0)
      ).setReg "o_acc" .real [M, D] (Tile.ofReal fun _ => 0)
  have hQ_loaded_eq : qLoaded = Tile.ofReal Q := by
    ext idx
    simp [qLoaded, qBase, Tile.ofReal]
    rw [show qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD =
        s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + idx.1.val * sQS + idx.2.1.val * sQD by
          simp [qBase, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]]
    exact congrArg some (hQ idx)
  refine ⟨s0, ?_, ?_⟩
  · simp [fa1PreLoopStrided, stepStmts, stepStmt, evalOp, Tile.bop, Tile.expandDim,
      NumericDType.add, NumericDType.mul, Option.bind, TileShape.dropInsertedIndex, Tile.vec, Tile.ofReal, qPtrs, qLoaded, qBase, s0]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
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
    · simp [s0, FA1MathCausal.mPartial]
    · simp [s0, FA1MathCausal.lPartial, Tile.ofReal]
    · simp [s0, FA1MathCausal.oPartial, Tile.ofReal]
    · intro idx
      simpa [s0] using hQ idx
    · intro idx
      simpa [s0] using hK idx
    · intro idx
      simpa [s0] using hV idx


end VeriTile.Examples
