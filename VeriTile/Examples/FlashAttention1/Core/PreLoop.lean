/-
VeriTile.Examples.FlashAttention1.Core.PreLoop

Split-out support for FlashAttention-1 v0/full-tile proofs.
-/

import VeriTile.Examples.FlashAttention1.Core.Bodies

namespace VeriTile.Examples

open VeriTile.Triton

private def fa1PreLoopStridedState (qReg : RegionName) (M D : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH : Nat) (sVB sVH : Nat)
    (sOB sOH : Nat) (s : BlockState) : BlockState :=
  let qBase : Nat := s.pids 2 * sQB + s.pids 1 * sQH
  let qPtrs : Tile .nat [M, D] :=
    ⟨fun idx => qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD⟩
  let qLoaded : Tile .real [M, D] :=
    ⟨fun idx => some (s.readMem qReg
      (qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD))⟩
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

private theorem fa1PreLoopStrided_step (qReg : RegionName) {M D : Nat}
    (sQB sQH sQS sQD : Nat) (sKB sKH : Nat) (sVB sVH : Nat)
    (sOB sOH : Nat) (s : BlockState) :
    stepStmts (fa1PreLoopStrided qReg M D
        sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH) s =
      some (fa1PreLoopStridedState qReg M D sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH s) := by
  let qBase : Nat := s.pids 2 * sQB + s.pids 1 * sQH
  let qPtrs : Tile .nat [M, D] :=
    ⟨fun idx => qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD⟩
  let qLoaded : Tile .real [M, D] :=
    ⟨fun idx => some (s.readMem qReg
      (qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD))⟩
  let s1 : BlockState :=
    BlockState.setReg s "pid_qb" .nat [] (Tile.scalar (s.pids 0))
  let s2 : BlockState :=
    BlockState.setReg s1 "pid_h" .nat [] (Tile.scalar (s.pids 1))
  let s3 : BlockState :=
    BlockState.setReg s2 "pid_b" .nat [] (Tile.scalar (s.pids 2))
  let s4 : BlockState :=
    BlockState.setReg s3 "q_base_off" .nat []
      (Tile.scalar (s.pids 2 * sQB + s.pids 1 * sQH))
  let s5 : BlockState :=
    BlockState.setReg s4 "k_base_off" .nat []
      (Tile.scalar (s.pids 2 * sKB + s.pids 1 * sKH))
  let s6 : BlockState :=
    BlockState.setReg s5 "v_base_off" .nat []
      (Tile.scalar (s.pids 2 * sVB + s.pids 1 * sVH))
  let s7 : BlockState :=
    BlockState.setReg s6 "o_base_off" .nat []
      (Tile.scalar (s.pids 2 * sOB + s.pids 1 * sOH))
  let s8 : BlockState :=
    BlockState.setReg s7 "offs_m" .nat [M]
      (Tile.vec (fun i : Fin M => s.pids 0 * M + i.val))
  let s9 : BlockState :=
    BlockState.setReg s8 "offs_d" .nat [D]
      (Tile.vec (fun d : Fin D => d.val))
  let s10 : BlockState :=
    BlockState.setReg s9 "q_ptrs" .nat [M, D] qPtrs
  let s11 : BlockState :=
    BlockState.setReg s10 "q" .real [M, D] qLoaded
  let s12 : BlockState :=
    BlockState.setReg s11 "m_i" .real [M] ⟨fun _ => (⊥ : WithBot ℝ)⟩
  let s13 : BlockState :=
    BlockState.setReg s12 "l_i" .real [M] (Tile.ofReal fun _ => 0)
  have hregm : s9.regs .nat [M] "offs_m" =
      some (Tile.vec (fun i : Fin M => s.pids 0 * M + i.val)) := by
    simp [s1, s2, s3, s4, s5, s6, s7, s8, s9, BlockState.setReg]
  have hregd : s9.regs .nat [D] "offs_d" =
      some (Tile.vec (fun d : Fin D => d.val)) := by
    simp [s1, s2, s3, s4, s5, s6, s7, s8, s9, BlockState.setReg]
  have h_qbase : s9.regs .nat [] "q_base_off" =
      some (Tile.scalar (s.pids 2 * sQB + s.pids 1 * sQH)) := by
    simp [s1, s2, s3, s4, s5, s6, s7, s8, s9, BlockState.setReg]
  have h_qptrs :
      evalOp
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL
            (Op.ref .nat [] "q_base_off")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
              (Op.constNat sQS)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
            (Op.constNat sQD))) s9 =
        some qPtrs := by
    simp only [evalOp_add, evalOp_mul, evalOp_constNat, Option.bind_eq_bind,
      Option.bind_some]
    rw [evalOp_ref, h_qbase]
    simp only [Option.bind_some]
    rw [evalOp_expandDim_one_nat, hregm]
    simp only [Option.bind_some]
    rw [evalOp_expandDim_zero_nat, hregd]
    simp [qPtrs, qBase, Tile.bop, NumericDType.add, NumericDType.mul]
  have h_load :
      evalOp
          (Op.load .real (MemAccess.region qReg (Op.ref .nat [M, D] "q_ptrs"))
            MaskOpt.none) s10 =
        some qLoaded := by
    have hmem : ∀ r o, s9.mem r o = s.mem r o := by
      intro r o
      simp [s1, s2, s3, s4, s5, s6, s7, s8, s9]
    simp [evalOp_load_region_none, s10, qLoaded, qPtrs, Region.cast]
    ext idx
    cases qReg
    simp [BlockState.readMem, hmem]
  have h1 : stepStmt (Stmt.assign .nat [] "pid_qb" (Op.programId 0)) s = some s1 := by
    simp [stepStmt, s1]
  have h2 : stepStmt (Stmt.assign .nat [] "pid_h" (Op.programId 1)) s1 = some s2 := by
    simp [stepStmt, s2, s1]
  have h3 : stepStmt (Stmt.assign .nat [] "pid_b" (Op.programId 2)) s2 = some s3 := by
    simp [stepStmt, s3, s1, s2]
  have h4 :
      stepStmt
          (Stmt.assign .nat [] "q_base_off"
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil
                (Op.ref .nat [] "pid_b") (Op.constNat sQB))
              (Op.mul .nat Broadcast.nil
                (Op.ref .nat [] "pid_h") (Op.constNat sQH)))) s3 = some s4 := by
    simp [stepStmt, s1, s2, s3, s4, Tile.bop, NumericDType.add,
      NumericDType.mul]
  have h5 :
      stepStmt
          (Stmt.assign .nat [] "k_base_off"
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil
                (Op.ref .nat [] "pid_b") (Op.constNat sKB))
              (Op.mul .nat Broadcast.nil
                (Op.ref .nat [] "pid_h") (Op.constNat sKH)))) s4 = some s5 := by
    simp [stepStmt, s1, s2, s3, s4, s5, Tile.bop, NumericDType.add,
      NumericDType.mul]
  have h6 :
      stepStmt
          (Stmt.assign .nat [] "v_base_off"
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil
                (Op.ref .nat [] "pid_b") (Op.constNat sVB))
              (Op.mul .nat Broadcast.nil
                (Op.ref .nat [] "pid_h") (Op.constNat sVH)))) s5 = some s6 := by
    simp [stepStmt, s1, s2, s3, s4, s5, s6, Tile.bop, NumericDType.add,
      NumericDType.mul]
  have h7 :
      stepStmt
          (Stmt.assign .nat [] "o_base_off"
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil
                (Op.ref .nat [] "pid_b") (Op.constNat sOB))
              (Op.mul .nat Broadcast.nil
                (Op.ref .nat [] "pid_h") (Op.constNat sOH)))) s6 = some s7 := by
    simp [stepStmt, s1, s2, s3, s4, s5, s6, s7, Tile.bop, NumericDType.add,
      NumericDType.mul]
  have h8 :
      stepStmt
          (Stmt.assign .nat [M] "offs_m"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil
                (Op.ref .nat [] "pid_qb")
                (Op.constNat M))
              (Op.arange M))) s7 = some s8 := by
    simp [stepStmt, s1, s2, s3, s4, s5, s6, s7, s8, Tile.bop,
      NumericDType.add, NumericDType.mul]
    congr 1
  have h9 : stepStmt (Stmt.assign .nat [D] "offs_d" (Op.arange D)) s8 = some s9 := by
    simp [stepStmt, s9]
  have h10 :
      stepStmt
          (Stmt.assign .nat [M, D] "q_ptrs"
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.add .nat Broadcast.scalarL
                (Op.ref .nat [] "q_base_off")
                (Op.mul .nat Broadcast.scalarR
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
                  (Op.constNat sQS)))
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
                (Op.constNat sQD)))) s9 = some s10 := by
    simp only [stepStmt, h_qptrs, Option.bind_some]
    rfl
  have h11 :
      stepStmt
          (Stmt.assign .real [M, D] "q"
            (Op.load .real (MemAccess.region qReg (Op.ref .nat [M, D] "q_ptrs"))
              MaskOpt.none)) s10 = some s11 := by
    simp [stepStmt, h_load, s11]
  have h12 :
      stepStmt (Stmt.assign .real [M] "m_i" (Op.full [M] Op.negInf)) s11 =
        some s12 := by
    simp [stepStmt, s12]
    congr 1
  have h13 :
      stepStmt (Stmt.assign .real [M] "l_i" (Op.full [M] (Op.const 0))) s12 =
        some s13 := by
    simp [stepStmt, s13, Tile.ofReal]
  have h14 :
      stepStmt (Stmt.assign .real [M, D] "o_acc" (Op.full [M, D] (Op.const 0))) s13 =
        some (fa1PreLoopStridedState qReg M D sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH s) := by
    simp [stepStmt, fa1PreLoopStridedState, s1, s2, s3, s4, s5, s6, s7, s8,
      s9, s10, s11, s12, s13, qPtrs, qLoaded, qBase, Tile.ofReal]
  rw [show fa1PreLoopStrided qReg M D sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH =
      [ Stmt.assign .nat [] "pid_qb" (Op.programId 0)
      , Stmt.assign .nat [] "pid_h"  (Op.programId 1)
      , Stmt.assign .nat [] "pid_b"  (Op.programId 2)
      , Stmt.assign .nat [] "q_base_off"
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil
              (Op.ref .nat [] "pid_b") (Op.constNat sQB))
            (Op.mul .nat Broadcast.nil
              (Op.ref .nat [] "pid_h") (Op.constNat sQH)))
      , Stmt.assign .nat [] "k_base_off"
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil
              (Op.ref .nat [] "pid_b") (Op.constNat sKB))
            (Op.mul .nat Broadcast.nil
              (Op.ref .nat [] "pid_h") (Op.constNat sKH)))
      , Stmt.assign .nat [] "v_base_off"
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil
              (Op.ref .nat [] "pid_b") (Op.constNat sVB))
            (Op.mul .nat Broadcast.nil
              (Op.ref .nat [] "pid_h") (Op.constNat sVH)))
      , Stmt.assign .nat [] "o_base_off"
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil
              (Op.ref .nat [] "pid_b") (Op.constNat sOB))
            (Op.mul .nat Broadcast.nil
              (Op.ref .nat [] "pid_h") (Op.constNat sOH)))
      , Stmt.assign .nat [M] "offs_m"
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil
              (Op.ref .nat [] "pid_qb")
              (Op.constNat M))
            (Op.arange M))
      , Stmt.assign .nat [D] "offs_d" (Op.arange D)
      , Stmt.assign .nat [M, D] "q_ptrs"
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.add .nat Broadcast.scalarL
              (Op.ref .nat [] "q_base_off")
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
                (Op.constNat sQS)))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
              (Op.constNat sQD)))
      , Stmt.assign .real [M, D] "q"
          (Op.load .real (MemAccess.region qReg (Op.ref .nat [M, D] "q_ptrs")) MaskOpt.none)
      , Stmt.assign .real [M] "m_i" (Op.full [M] Op.negInf)
      , Stmt.assign .real [M] "l_i" (Op.full [M] (Op.const 0))
      , Stmt.assign .real [M, D] "o_acc" (Op.full [M, D] (Op.const 0))
      ] by rfl]
  rw [stepStmts.cons_some h1]
  rw [stepStmts.cons_some h2]
  rw [stepStmts.cons_some h3]
  rw [stepStmts.cons_some h4]
  rw [stepStmts.cons_some h5]
  rw [stepStmts.cons_some h6]
  rw [stepStmts.cons_some h7]
  rw [stepStmts.cons_some h8]
  rw [stepStmts.cons_some h9]
  rw [stepStmts.cons_some h10]
  rw [stepStmts.cons_some h11]
  rw [stepStmts.cons_some h12]
  rw [stepStmts.cons_some h13]
  rw [stepStmts.cons_some h14]
  exact stepStmts.nil

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
  let s0 := fa1PreLoopStridedState qReg M D sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH s
  have hQ_loaded_eq : qLoaded = Tile.ofReal Q := by
    ext idx
    simp [qLoaded, qBase, Tile.ofReal]
    rw [show qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD =
        s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + idx.1.val * sQS + idx.2.1.val * sQD by
          simp [qBase, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]]
    exact congrArg some (hQ idx)
  refine ⟨s0, ?_, ?_⟩
  · simpa [s0] using
      fa1PreLoopStrided_step qReg sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH s
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simpa [s0, fa1PreLoopStridedState, qLoaded, qBase] using hQ_loaded_eq
    · simp [s0, fa1PreLoopStridedState, StreamingAccumulator.mPartial]
    · simp [s0, fa1PreLoopStridedState, StreamingAccumulator.lPartial, Tile.ofReal]
    · simp [s0, fa1PreLoopStridedState, StreamingAccumulator.oPartial, Tile.ofReal]
    · intro idx
      simpa [s0, fa1PreLoopStridedState] using hQ idx
    · intro idx
      simpa [s0, fa1PreLoopStridedState] using hK idx
    · intro idx
      simpa [s0, fa1PreLoopStridedState] using hV idx

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
  let s0 := fa1PreLoopStridedState qReg M D sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH s
  have hQ_loaded_eq : qLoaded = Tile.ofReal Q := by
    ext idx
    simp [qLoaded, qBase, Tile.ofReal]
    rw [show qBase + (s.pids 0 * M + idx.1.val) * sQS + idx.2.1.val * sQD =
        s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + idx.1.val * sQS + idx.2.1.val * sQD by
          simp [qBase, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]]
    exact congrArg some (hQ idx)
  refine ⟨s0, ?_, ?_⟩
  · simpa [s0] using
      fa1PreLoopStrided_step qReg sQB sQH sQS sQD sKB sKH sVB sVH sOB sOH s
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simp [s0, fa1PreLoopStridedState]
    · simpa [s0, fa1PreLoopStridedState, qLoaded, qBase] using hQ_loaded_eq
    · simp [s0, fa1PreLoopStridedState, FA1MathCausal.mPartial]
    · simp [s0, fa1PreLoopStridedState, FA1MathCausal.lPartial, Tile.ofReal]
    · simp [s0, fa1PreLoopStridedState, FA1MathCausal.oPartial, Tile.ofReal]
    · intro idx
      simpa [s0, fa1PreLoopStridedState] using hQ idx
    · intro idx
      simpa [s0, fa1PreLoopStridedState] using hK idx
    · intro idx
      simpa [s0, fa1PreLoopStridedState] using hV idx


end VeriTile.Examples
