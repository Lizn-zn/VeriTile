import bench.tritonbench_g.attention_fwd_triton1.Proofs.RealBranch
import bench.tritonbench_g.attention_fwd_triton1.Proofs.RealStore

/-! One complete chunk for all STORE/IFCOND combinations. H receives the
pre-update carry; the output includes the current chunk and all earlier
chunks. The memory frame covers both actual write windows. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealIteration
open VeriTile.Triton
open RealBranch
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 2400000
set_option maxRecDepth 10000

theorem load_key (s sin' : BlockState) (K : RegionName)
    (s_qh BT BD : Nat) (c : Nat)
    (hmem : ∀ off, sin'.readMem K off = s.readMem K off)
    (hpk : sin'.regs .blockPtr [BD, BT] "p_k" = some
      ⟨fun _ => BlockPtr.mk K (s.pids 0 * s_qh) [BD, NT * BT] [BD, BT] [1, BD] [0, c * BT]⟩) :
    evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BD, BT] "p_k") []) MaskOpt.none .unspecified)
        sin'
      = some (aft1BkTileG s K s_qh BT BD c) := by
  rw [aft1_load_bp_2d_ref K sin' "p_k" (s.pids 0 * s_qh) BD (NT * BT) BD BT 1 BD 0 (c * BT) hpk]
  refine congrArg some ?_
  ext idx; obtain ⟨e, tk, u⟩ := idx
  simp only [aft1BkTileG, aft1KCellG, hmem, Nat.add_zero, Nat.zero_add, Nat.mul_one]

/-- General `b_v` load = `aft1BvTileG`. -/
theorem load_value (s sin' : BlockState) (V : RegionName)
    (s_qh BT BD : Nat) (c : Nat)
    (hmem : ∀ off, sin'.readMem V off = s.readMem V off)
    (hpv : sin'.regs .blockPtr [BT, BD] "p_v" = some
      ⟨fun _ => BlockPtr.mk V (s.pids 0 * s_qh) [NT * BT, BD] [BT, BD] [BD, 1] [c * BT, 0]⟩) :
    evalOp (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BD] "p_v") []) MaskOpt.none .unspecified)
        sin'
      = some (aft1BvTileG s V s_qh BT BD c) := by
  rw [aft1_load_bp_2d_ref V sin' "p_v" (s.pids 0 * s_qh) (NT * BT) BD BT BD BD 1 (c * BT) 0 hpv]
  refine congrArg some ?_
  ext idx; obtain ⟨tk, d, u⟩ := idx
  simp only [aft1BvTileG, aft1VCellG, hmem, Nat.add_zero, Nat.zero_add, Nat.mul_one]

/-- Execute through the carry update, retaining the optional H snapshot. -/
theorem registers_ready
    (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ) (BT BD NT : Nat)
    (STORE IFCOND : Bool) (s sin : BlockState) (c : Nat)
    (hQH : Q ≠ H) (hKH : K ≠ H) (hVH : V ≠ H) (hstride : BD ≤ s_ht)
    (hmem : ∀ rg off, sin.readMem rg off = s.readMem rg off)
    (hibh : sin.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0)))
    (hi : sin.regs .nat [] "i" = some (Tile.scalar c))
    (hbh : sin.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD c)) :
    ∃ st, stepStmts ((aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT
        STORE IFCOND).take 13) sin = some st
      ∧ st.pids = sin.pids
      ∧ st.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0))
      ∧ st.regs .blockPtr [BT, BD] "p_o" = some
          (⟨fun _ => BlockPtr.mk O (s.pids 0 * s_qh) [NT * BT, BD] [BT, BD] [BD, 1]
            [c * BT, 0]⟩ : Tile .blockPtr [BT, BD])
      ∧ st.regs .real [BT, BD] "b_o" = some (aft1BoTileG s Q K V s_qh scale BT BD c)
      ∧ st.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD (c + 1))
      ∧ (STORE = Bool.true → ∀ (i j : Fin BD),
          st.mem H (s.pids 0 * s_hh + (c * BD + i.val) * s_ht + j.val) =
            MemCell.real (aft1RecStateG s K V s_qh BT BD c i j))
      ∧ (∀ rg off, (STORE = Bool.false ∨ rg ≠ H ∨ ∀ (i j : Fin BD),
          off ≠ s.pids 0 * s_hh + (c * BD + i.val) * s_ht + j.val) →
          st.mem rg off = sin.mem rg off) := by
  unfold aft1LoopBodyG
  simp only [List.take_succ_cons, List.take_zero]
  -- p_q, p_k, p_v, p_h, p_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval Q sin _ _ _ [NT * BT, BD] [BT, BD] [BD, 1]
      (s.pids 0 * s_qh) (c * BT) 0
      (aft1_mulConst_eval sin "i_bh" (s.pids 0) s_qh hibh)
      (aft1_mulConst_eval sin "i" c BT hi) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval K _ _ _ _ [BD, NT * BT] [BD, BT] [1, BD]
      (s.pids 0 * s_qh) 0 (c * BT)
      (aft1_mulConst_eval _ "i_bh" (s.pids 0) s_qh (by simp [hibh])) (by simp)
      (aft1_mulConst_eval _ "i" c BT (by simp [hi]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval V _ _ _ _ [NT * BT, BD] [BT, BD] [BD, 1]
      (s.pids 0 * s_qh) (c * BT) 0
      (aft1_mulConst_eval _ "i_bh" (s.pids 0) s_qh (by simp [hibh]))
      (aft1_mulConst_eval _ "i" c BT (by simp [hi])) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval H _ _ _ _ [NT * BD, BD] [BD, BD] [s_ht, 1]
      (s.pids 0 * s_hh) (c * BD) 0
      (aft1_mulConst_eval _ "i_bh" (s.pids 0) s_hh (by simp [hibh]))
      (aft1_mulConst_eval _ "i" c BD (by simp [hi])) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval O _ _ _ _ [NT * BT, BD] [BT, BD] [BD, 1]
      (s.pids 0 * s_qh) (c * BT) 0
      (aft1_mulConst_eval _ "i_bh" (s.pids 0) s_qh (by simp [hibh]))
      (aft1_mulConst_eval _ "i" c BT (by simp [hi])) (by simp)))]
  rw [stepStmts.cons_some (RealStore.save_executes STORE _ H "p_h" "b_h"
    (NT * BD) BD BD BD (s.pids 0 * s_hh) (c * BD) s_ht
    (aft1RecStateG s K V s_qh BT BD c) (by simp) (by simpa [aft1BhTileG] using hbh))]
  -- b_q raw load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_load_bp_2d_ref Q _ "p_q" (s.pids 0 * s_qh) (NT * BT) BD BT BD BD 1
      (c * BT) 0 (by simp)))]
  -- b_q *= scale → aft1BqTileG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BT, BD] "b_q")
        (Op.const scale)) _ = some (aft1BqTileG s Q s_qh scale BT BD c) from by
      rw [evalOp_mul]
      simp only [evalOp_ref, BlockState.setReg_same, evalOp_const, Option.bind_eq_bind,
        Option.bind_some]
      refine congrArg some ?_
      ext idx; obtain ⟨t, e, u⟩ := idx
      simp only [Tile.bop, Broadcast.scalarR, Broadcast.leftIndex, Broadcast.rightIndex,
        Tile.scalar, aft1BqTileG, aft1QCellG, NumericDType.mul, WithBot.realMul,
        Option.map₂, Option.bind, Option.map, BlockState.setReg_readMem,
        RealStore.save_read_other (hr := hQH), hmem,
        Nat.add_zero, Nat.zero_add, Nat.mul_one]))]
  -- b_k, b_v loads
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_key (NT := NT) s _ K s_qh BT BD c
      (by intro off; simp [RealStore.save_read_other, hKH, hmem]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_value (NT := NT) s _ V s_qh BT BD c
      (by intro off; simp [RealStore.save_read_other, hVH, hmem]) (by simp)))]
  -- b_s = dot(b_q, b_k)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_bs_eqG s _ Q K s_qh scale BT BD c (by simp) (by simp)))]
  -- b_o = dot(b_s, b_v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aft1_bo_local_eqG s _ Q K V s_qh scale BT BD c (by simp) (by simp)))]
  change ∃ st, stepStmts [update BT BD IFCOND] _ = some st ∧ _
  rw [stepStmts.cons_some (update_executes s _ Q K V s_qh scale BT BD c IFCOND
    (by simp [hi]) (by simp) (by simp) (by simp) (by simp) (by simp [hbh]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · simp [hibh]
  · simp
  · apply afterUpdate_output
    simp
  · simp
  · intro hs i j
    simp only [afterUpdate_mem, BlockState.setReg_mem]
    exact RealStore.save_readback_cell STORE _ H _ _ _ _ hstride hs i j
  · intro rg off hout
    simp only [afterUpdate_mem, BlockState.setReg_mem]
    rw [RealStore.save_frame STORE _ H _ _ _ _ rg off hout]
    rfl

/-- One full source iteration, including both stores. No O or H output
formula is assumed: both follow from the executed assignments and scatters. -/
theorem executes
    (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ) (BT BD NT : Nat)
    (STORE IFCOND : Bool) (s sin : BlockState) (c : Nat)
    (hQH : Q ≠ H) (hKH : K ≠ H) (hVH : V ≠ H) (hHO : H ≠ O) (hstride : BD ≤ s_ht)
    (hmem : ∀ rg off, sin.readMem rg off = s.readMem rg off)
    (hibh : sin.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0)))
    (hi : sin.regs .nat [] "i" = some (Tile.scalar c))
    (hbh : sin.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD c)) :
    ∃ st, stepStmts (aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT
        STORE IFCOND) sin = some st
      ∧ st.pids = sin.pids
      ∧ st.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0))
      ∧ st.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD (c + 1))
      ∧ (∀ (i : Fin BT) (j : Fin BD),
          st.mem O (s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val) =
            MemCell.real (aft1OutG s Q K V s_qh scale BT BD c i j))
      ∧ (STORE = Bool.true → ∀ (i j : Fin BD),
          st.mem H (s.pids 0 * s_hh + (c * BD + i.val) * s_ht + j.val) =
            MemCell.real (aft1RecStateG s K V s_qh BT BD c i j))
      ∧ (∀ rg off, (rg ≠ O ∨ ∀ (i : Fin BT) (j : Fin BD),
          off ≠ s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val) →
          (STORE = Bool.false ∨ rg ≠ H ∨ ∀ (i j : Fin BD),
          off ≠ s.pids 0 * s_hh + (c * BD + i.val) * s_ht + j.val) →
          st.mem rg off = sin.mem rg off) := by
  obtain ⟨s2, hstep, hpids, hib, hpo, hbo, hcarry, hsnapshot, hframe⟩ :=
    registers_ready Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND s sin c
      hQH hKH hVH hstride hmem hibh hi hbh
  have hsplit : aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND =
      (aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND).take 13 ++
        [.store .real [BT, BD] (.blockPtr (.ref .blockPtr [BT, BD] "p_o") [])
          (.ref .real [BT, BD] "b_o") .none] := rfl
  rw [hsplit, stepStmts.append_some hstep]
  rw [stepStmts.cons_some (RealStore.executes s2 O "p_o" "b_o"
    (NT * BT) BD BT BD (s.pids 0 * s_qh) (c * BT) BD
    (aft1OutG s Q K V s_qh scale BT BD c) hpo hbo), stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa only [RealStore.pids] using hpids
  · simpa only [RealStore.regs] using hib
  · simpa only [RealStore.regs] using hcarry
  · exact RealStore.readback_cell s2 O _ _ _ _ (Nat.le_refl BD)
  · intro hs i j
    rw [RealStore.frame s2 O _ _ _ _ H _ (Or.inl hHO)]
    exact hsnapshot hs i j
  · intro rg off hout hsnap
    rw [RealStore.frame s2 O _ _ _ _ rg off hout, hframe rg off hsnap]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealIteration
