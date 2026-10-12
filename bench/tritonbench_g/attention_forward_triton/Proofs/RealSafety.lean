import bench.tritonbench_g.attention_forward_triton.Proofs.LegacyRounding
import bench.tritonbench_g.attention_forward_triton.Proofs.RealExecution

/-! Full address-safety walk for the erased typed source. The loop invariant
tracks pointer/index registers and only the existence of value registers, so
safety does not assume any numerical output equation or masked-load default.
The six bounds groups are precisely the five input windows and output window. -/
namespace VeriTile.Bench.TritonBenchG.AttentionForwardTriton.RealSafety
open VeriTile.Triton
open VeriTile.Examples.AttentionForwardClosedForm
open VeriTile.Examples.AttentionForwardClosedForm renaming baseOffset → exBaseOffset
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- Initialization is shared with the existing pointer-layout proof. -/
theorem prologue_eq (Q K V QS KS Out : RegionName)
    (sqz sqh Z H BM BN NC HD BD HA stage : Nat) :
    RealExecution.prologue Q K V QS KS Out sqz sqh Z H BM BN NC HD BD HA stage =
      attnIOPre22 Q K V QS KS Out sqz sqh H HD (BN * NC) BM BN BD HA := rfl

/-- A successful erased iteration preserves all address and register-shape pins. -/
theorem step (K KScale V Out : RegionName) (s0 : BlockState)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N)
    (c : Nat) (s : BlockState) (hc : c < BLOCK_N * numKVBlocks)
    (hP : attnIOSafeInv K KScale V Out s0 stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
      BLOCK_DMODEL numKVBlocks c s) :
    ∃ s', stepStmts (RealLoop.attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
        (s.setReg "start_n" .nat [] (Tile.scalar c)) = some s'
      ∧ attnIOSafeInv K KScale V Out s0 stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
          BLOCK_DMODEL numKVBlocks (c + BLOCK_N) s' := by
  obtain ⟨hmod, hle, hoffsm, hoffsn, ⟨mT, hmi⟩, ⟨lT, hli⟩, ⟨aT, hacc⟩, ⟨qT, hq⟩, ⟨qsv, hqs⟩,
    hKp, hKsp, hVp, hOp⟩ := hP
  have hax : 1 < [BLOCK_M].length.succ := by simp
  have hcBN : c / BLOCK_N * BLOCK_N = c := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)
  have hcdivlt : c / BLOCK_N < numKVBlocks := by
    have h1 : c / BLOCK_N * BLOCK_N < numKVBlocks * BLOCK_N := by
      rw [hcBN, Nat.mul_comm numKVBlocks BLOCK_N]
      exact hc
    exact lt_of_mul_lt_mul_right h1 (Nat.zero_le BLOCK_N)
  have hdiv : (c + BLOCK_N) / BLOCK_N = c / BLOCK_N + 1 := by
    conv_lhs => rw [show c = c / BLOCK_N * BLOCK_N from hcBN.symm]
    rw [Nat.add_div_right _ hBN, Nat.mul_div_cancel _ hBN]
  have hle' : c + BLOCK_N ≤ BLOCK_N * numKVBlocks := by
    have h1 : (c / BLOCK_N + 1) * BLOCK_N ≤ numKVBlocks * BLOCK_N :=
      Nat.mul_le_mul_right BLOCK_N hcdivlt
    calc c + BLOCK_N = c / BLOCK_N * BLOCK_N + BLOCK_N := by rw [hcBN]
      _ = (c / BLOCK_N + 1) * BLOCK_N := by ring
      _ ≤ numKVBlocks * BLOCK_N := h1
      _ = BLOCK_N * numKVBlocks := Nat.mul_comm _ _
  set u0 : BlockState := s.setReg "start_n" .nat [] (Tile.scalar c) with hu0
  set Kptr := attnIOkPtrs s0 K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c / BLOCK_N)
    with hKptr
  set Ksptr := attnIOsclPtr KScale
    (s0.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + c / BLOCK_N) with hKsptr
  set Vptr := attnIOvPtrs s0 V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c / BLOCK_N)
    with hVptr
  set kmaskT : Tile .bool [BLOCK_DMODEL, BLOCK_N] :=
    ⟨fun idx => (decide (idx.2.1.val < BLOCK_N * numKVBlocks - c)
      && decide (idx.1.val < HEAD_ACTIVE))⟩ with hkm
  set kloadT : Tile .real [BLOCK_DMODEL, BLOCK_N] :=
    ⟨fun i => if kmaskT.data i then some (u0.readMem (Kptr.data i).1 (Kptr.data i).2)
      else some (u0.undef (Kptr.data i).1 (Kptr.data i).2)⟩ with hkl
  set ksv : ℝ := u0.readMem (Ksptr.data PUnit.unit).1 (Ksptr.data PUnit.unit).2 with hksv
  set qkT : Tile .real [BLOCK_M, BLOCK_N] := Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (⟨fun i => FloatDType.real.cast FloatDType.real ((Tile.dot [] qT kloadT).data i)⟩
          : Tile .real [BLOCK_M, BLOCK_N])
        (Tile.scalar (some qsv : WithBot ℝ))) (Tile.scalar (some ksv : WithBot ℝ)) with hqk
  obtain ⟨rmaxT, hrm⟩ :
      ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qkT = some t :=
    ⟨_, by
      unfold Tile.reduceMaxDrop
      rw [dif_pos (show 0 < TileShape.axisDim [BLOCK_M, BLOCK_N]
        (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) from hBN)]⟩
  set mijT : Tile .real [BLOCK_M] :=
    Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mT rmaxT)
      mT rmaxT with hmij
  set qk2T : Tile .real [BLOCK_M, BLOCK_N] :=
    Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT
      (Tile.expandDim ⟨1, hax⟩ mijT) with hqk2
  set pT : Tile .real [BLOCK_M, BLOCK_N] := Tile.uop WithBot.realExp2 qk2T with hpT
  set lijT : Tile .real [BLOCK_M] :=
    Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) pT with hlij
  set alphaT : Tile .real [BLOCK_M] := Tile.uop WithBot.realExp2
    (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mT mijT) with hal
  set acc1T : Tile .real [BLOCK_M, BLOCK_DMODEL] :=
    Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) aT
      (Tile.expandDim ⟨1, hax⟩ alphaT) with hacc1
  set vmaskT : Tile .bool [BLOCK_N, BLOCK_DMODEL] :=
    ⟨fun idx => (decide (idx.1.val < BLOCK_N * numKVBlocks - c)
      && decide (idx.2.1.val < HEAD_ACTIVE))⟩ with hvm
  set vloadT : Tile .real [BLOCK_N, BLOCK_DMODEL] :=
    ⟨fun i => if vmaskT.data i then some (u0.readMem (Vptr.data i).1 (Vptr.data i).2)
      else some (u0.undef (Vptr.data i).1 (Vptr.data i).2)⟩ with hvl
  simp only [RealLoop.attnLoopBody, attnLoopBody,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") u0 = some (Tile.scalar c) from by
      rw [evalOp_ref]; simp [hu0]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (kmask_eval _ BLOCK_DMODEL BLOCK_N (BLOCK_N * numKVBlocks) c HEAD_ACTIVE
      (by simp [hu0, hoffsn]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (attnIO_load_mask_realW (Op.ref .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs") _ _ Kptr kmaskT
      (by rw [evalOp_ref]; simp [hu0, hKp]) (by rw [evalOp_ref]; simp [hkm])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (attnIO_scalar_load_form _ "K_scale_ptr" Ksptr (by rw [evalOp_ref]; simp [hu0, hKsp])))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (RealLoop.qk_op_eval _ BLOCK_M BLOCK_N BLOCK_DMODEL qT kloadT qsv ksv
      (by simp [hu0, hq]) (by simp [hkl]) (by simp [hu0, hqs]) (by simp [hksv])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mij_op_eval _ BLOCK_M BLOCK_N mT qkT rmaxT (by simp [hu0, hmi]) (by simp [hqk]) hrm))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (qk2_op_eval _ BLOCK_M BLOCK_N hax qkT mijT (by simp [hqk]) (by simp [hmij])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (p_op_eval _ BLOCK_M BLOCK_N qk2T (by simp [hqk2])))]
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .real [BLOCK_M] "l_ij"
    (Op.reduceSum (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
      (Op.ref .real [BLOCK_M, BLOCK_N] "p")) _ lijT
    (lij_op_eval _ BLOCK_M BLOCK_N pT (by simp [hpT])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (alpha_op_eval _ BLOCK_M mT mijT (by simp [hu0, hmi]) (by simp [hmij])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (li_op_eval _ BLOCK_M lT alphaT lijT (by simp [hu0, hli]) (by simp [hal]) (by simp [hlij])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (acc1_op_eval _ BLOCK_M BLOCK_DMODEL hax aT alphaT (by simp [hu0, hacc]) (by simp [hal])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (attnIO_load_mask_realW (Op.ref .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs") _ _ Vptr vmaskT
      (by rw [evalOp_ref]; simp [hu0, hVp])
      (vmask_eval _ BLOCK_N BLOCK_DMODEL (BLOCK_N * numKVBlocks) c HEAD_ACTIVE
        (by simp [hu0, hoffsn]) (by simp [hu0]))))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLOCK_M, BLOCK_N] "p") _ = some pT from by
      rw [evalOp_ref]; simp [hpT]))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (RealLoop.acc2_op_eval _ BLOCK_M BLOCK_N BLOCK_DMODEL acc1T pT vloadT
      (by simp [hacc1]) (by simp [hpT]) (by simp [hvl])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLOCK_M] "m_ij") _ = some mijT from by
      rw [evalOp_ref]; simp [hmij]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (kptr_adv_eval _ BLOCK_DMODEL BLOCK_N BLOCK_N HEAD_DIM Kptr "K_ptrs" (by simp [hu0, hKp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ksptr_adv_eval _ Ksptr "K_scale_ptr" (by simp [hu0, hKsp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (kptr_adv_eval _ BLOCK_N BLOCK_DMODEL BLOCK_N HEAD_DIM Vptr "V_ptrs" (by simp [hu0, hVp])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, hle', ?_, ?_, ⟨mijT, ?_⟩,
    ⟨Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
      (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) lT alphaT) lijT, ?_⟩,
    ⟨Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      acc1T (Tile.dot [] pT vloadT), ?_⟩,
    ⟨qT, ?_⟩, ⟨qsv, ?_⟩, ?_, ?_, ?_, ?_⟩
  · rw [Nat.add_mod_right]; exact hmod
  · simp [hu0, hoffsm]
  · simp [hu0, hoffsn]
  · simp
  · simp
  · simp
  · simp [hu0, hq]
  · simp [hu0, hqs]
  · simp only [hdiv]
    rw [← attnIOkPtrs_succ s0 K H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c / BLOCK_N)]
    simp [hKptr]
  · simp only [hdiv, ← Nat.add_assoc]
    rw [← attnIOsclPtr_succ KScale
      (s0.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + c / BLOCK_N)]
    simp [hKsptr]
  · simp only [hdiv]
    rw [← attnIOvPtrs_succ s0 V H stride_qz stride_qh HEAD_DIM BLOCK_N BLOCK_DMODEL (c / BLOCK_N)]
    simp [hVptr]
  · simp [hu0, hOp]


/-- All active in-loop loads are within the supplied windows. -/
theorem body_safe (R : RoundingModel) (bounds : RegionBounds)
    (K KScale V Out : RegionName) (s0 : BlockState)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (_hBN : 0 < BLOCK_N)
    (c : Nat) (s : BlockState) (hc : c < BLOCK_N * numKVBlocks)
    (hP : attnIOSafeInv K KScale V Out s0 stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
      BLOCK_DMODEL numKVBlocks c s)
    (hbK : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_DMODEL * BLOCK_N)),
      j.val % BLOCK_N < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE →
      s0.pids 1 / H * stride_qz + s0.pids 1 % H * stride_qh
        + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM < bounds K)
    (hbKS : ∀ (t : Fin numKVBlocks) (_j : Fin 1),
      s0.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + t.val < bounds KScale)
    (hbV : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_N * BLOCK_DMODEL)),
      j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks - t.val * BLOCK_N
        ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s0.pids 1 / H * stride_qz + s0.pids 1 % H * stride_qh
        + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL < bounds V) :
    Stmt.TraceSafeListR R bounds
      (RealLoop.attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
      (s.setReg "start_n" .nat [] (Tile.scalar c)) := by
  obtain ⟨hmod, hle, hoffsm, hoffsn, ⟨mT, hmi⟩, ⟨lT, hli⟩, ⟨aT, hacc⟩, ⟨qT, hq⟩, ⟨qsv, hqs⟩,
    hKp, hKsp, hVp, hOp⟩ := hP
  have hcBN : c / BLOCK_N * BLOCK_N = c := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)
  have hcdivlt : c / BLOCK_N < numKVBlocks := by
    have h1 : c / BLOCK_N * BLOCK_N < numKVBlocks * BLOCK_N := by
      rw [hcBN, Nat.mul_comm numKVBlocks BLOCK_N]
      exact hc
    exact lt_of_mul_lt_mul_right h1 (Nat.zero_le BLOCK_N)
  simp only [RealLoop.attnLoopBody, attnLoopBody,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, List.nil_append]
  -- (0) start_n = start_n (identity)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s1 h1 => ?_)
  obtain ⟨v0, hv0, rfl⟩ := stepStmtR_assign_inv h1
  rw [evalOpR_ref, BlockState.setReg_same] at hv0
  obtain rfl := Option.some.inj hv0
  -- (1) k_mask
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s2 h2 => ?_)
  obtain ⟨vkm, hvkm, rfl⟩ := stepStmtR_assign_inv h2
  rw [attnIO_kmask_evalR R _ BLOCK_DMODEL BLOCK_N (BLOCK_N * numKVBlocks) c HEAD_ACTIVE
    (by simp [hoffsn]) (BlockState.setReg_same _ _ _ _ _)] at hvkm
  obtain rfl := Option.some.inj hvkm
  -- (2) k = tl.load(K_ptrs, mask=k_mask): the read2 window bound
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s3 h3 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs idx hact
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] at hptrs
    rw [hKp] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmasks, hidx⟩ := hact
    rw [evalOpR_ref, BlockState.setReg_same] at hmasks
    obtain rfl := Option.some.inj hmasks
    have hP2 : idx.2.1.val < BLOCK_N * numKVBlocks - c ∧ idx.1.val < HEAD_ACTIVE := by
      simpa using hidx
    have hbound := hbK ⟨c / BLOCK_N, hcdivlt⟩ (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) (by
      simp only [Lane2D.encode_div, Lane2D.encode_mod]
      rw [hcBN]
      exact ⟨hP2.1, hP2.2⟩)
    simpa [attnIOkPtrs, exBaseOffset, Lane2D.encode_div, Lane2D.encode_mod] using hbound
  obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv h3
  -- (3) k_scale = tl.load(K_scale_ptr): the read5 slot bound
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s4 h4 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs i _
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] at hptrs
    rw [hKsp] at hptrs
    obtain rfl := Option.some.inj hptrs
    have hbound := hbKS ⟨c / BLOCK_N, hcdivlt⟩ 0
    simpa [attnIOsclPtr] using hbound
  obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv h4
  -- (4)–(11): register-only assigns
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s5 h5 => ?_)
  obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv h5
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s6 h6 => ?_)
  obtain ⟨v6, -, rfl⟩ := stepStmtR_assign_inv h6
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s7 h7 => ?_)
  obtain ⟨v7, -, rfl⟩ := stepStmtR_assign_inv h7
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s8 h8 => ?_)
  obtain ⟨v8, -, rfl⟩ := stepStmtR_assign_inv h8
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s9 h9 => ?_)
  obtain ⟨v9, -, rfl⟩ := stepStmtR_assign_inv h9
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s10 h10 => ?_)
  obtain ⟨v10, -, rfl⟩ := stepStmtR_assign_inv h10
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s11 h11 => ?_)
  obtain ⟨v11, -, rfl⟩ := stepStmtR_assign_inv h11
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s12 h12 => ?_)
  obtain ⟨v12, -, rfl⟩ := stepStmtR_assign_inv h12
  -- (12) v = tl.load(V_ptrs, mask=…): the read3 window bound
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s13 h13 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, by simp, ?_⟩
    intro ptrs hptrs idx hact
    rw [evalOpR_ref] at hptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] at hptrs
    rw [hVp] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmasks, hidx⟩ := hact
    rw [attnIO_vmask_evalR R _ BLOCK_N BLOCK_DMODEL (BLOCK_N * numKVBlocks) c HEAD_ACTIVE
      (by simp [hoffsn])
      (by simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same])] at hmasks
    obtain rfl := Option.some.inj hmasks
    have hP2 : idx.1.val < BLOCK_N * numKVBlocks - c ∧ idx.2.1.val < HEAD_ACTIVE := by
      simpa using hidx
    have hbound := hbV ⟨c / BLOCK_N, hcdivlt⟩ (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) (by
      simp only [Lane2D.encode_div, Lane2D.encode_mod]
      rw [hcBN]
      exact ⟨hP2.1, hP2.2⟩)
    simpa [attnIOvPtrs, exBaseOffset, Lane2D.encode_div, Lane2D.encode_mod] using hbound
  obtain ⟨v13, -, rfl⟩ := stepStmtR_assign_inv h13
  -- (13)–(18): register-only assigns
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s14 h14 => ?_)
  obtain ⟨v14, -, rfl⟩ := stepStmtR_assign_inv h14
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s15 h15 => ?_)
  obtain ⟨v15, -, rfl⟩ := stepStmtR_assign_inv h15
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s16 h16 => ?_)
  obtain ⟨v16, -, rfl⟩ := stepStmtR_assign_inv h16
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s17 h17 => ?_)
  obtain ⟨v17, -, rfl⟩ := stepStmtR_assign_inv h17
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s18 h18 => ?_)
  obtain ⟨v18, -, rfl⟩ := stepStmtR_assign_inv h18
  exact Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (fun _ _ => Stmt.TraceSafeListR.nil_intro)


/-- Complete source safety, including both scale loads and the masked store. -/
theorem original_traceSafeR
    (bounds : RegionBounds) (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks : Nat)
    (s : BlockState) (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hbQ : ∀ (_t : Fin numKVBlocks) (j : Fin (BLOCK_M * BLOCK_DMODEL)),
      s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
        ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh
        + (s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL < bounds Q)
    (hbK : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_DMODEL * BLOCK_N)),
      j.val % BLOCK_N < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE →
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh
        + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM < bounds K)
    (hbV : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_N * BLOCK_DMODEL)),
      j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks - t.val * BLOCK_N
        ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh
        + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL < bounds V)
    (hbQS : ∀ (_t : Fin numKVBlocks) (_j : Fin 1),
      s.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_M - 1) / BLOCK_M) + s.pids 0 < bounds QScale)
    (hbKS : ∀ (t : Fin numKVBlocks) (_j : Fin 1),
      s.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + t.val < bounds KScale)
    (hbO : ∀ j : Fin (BLOCK_M * BLOCK_DMODEL),
      s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
        ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh
        + (s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
        < bounds Out) :
    (RealExecution.original Q K V QScale KScale Out stride_qz stride_qh Z H
      BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.TraceSafeR
        .triv bounds s := by
  unfold Kernel.TraceSafeR
  rw [RealExecution.body_split, prologue_eq]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- the prologue: 20 register-only assigns then the two bounded loads
    unfold attnIOPre22
    refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
    · exact Stmt.TraceSafeListR.of_forall _ _
        (attnIO_pre20_stmt_safe .triv bounds Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM
          (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N BLOCK_DMODEL)
    · intro s2 hs2
      obtain ⟨s20, h20, hpids20, hmem20, hundef20, hoffsm, hoffsn, hmi, hli, hacc,
        hqp, hqsp, hkp, hksp, hvp, hop⟩ :=
        attnIO_pre20W s Q K V QScale KScale Out stride_qz stride_qh Z H BLOCK_M BLOCK_N
          numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE
      rw [stepStmtsR_triv, h20] at hs2
      obtain rfl := Option.some.inj hs2
      unfold attnIOPreLoads
      -- the q load: read1 window bound
      refine Stmt.TraceSafeListR.cons_intro ?_ (fun s21 h21 => ?_)
      · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
          MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
        refine ⟨trivial, by simp, ?_⟩
        intro ptrs hptrs idx hact
        rw [evalOpR_ref] at hptrs
        rw [hqp] at hptrs
        obtain rfl := Option.some.inj hptrs
        obtain ⟨masks, hmasks, hidx⟩ := hact
        rw [attnIO_qmask_evalR .triv _ BLOCK_M BLOCK_DMODEL (BLOCK_N * numKVBlocks) HEAD_ACTIVE
          (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val) hoffsm] at hmasks
        obtain rfl := Option.some.inj hmasks
        have hP2 : s.pids 0 * BLOCK_M + idx.1.val < BLOCK_N * numKVBlocks
            ∧ idx.2.1.val < HEAD_ACTIVE := by simpa using hidx
        have hbound := hbQ ⟨0, hnum⟩ (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) (by
          simp only [Lane2D.encode_div, Lane2D.encode_mod]
          exact hP2)
        simpa [attnIOrowPtrs, exBaseOffset, Lane2D.encode_div, Lane2D.encode_mod] using hbound
      obtain ⟨vq, -, rfl⟩ := stepStmtR_assign_inv h21
      -- the q_scale load: read4 slot bound
      refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
      refine ⟨trivial, trivial, ?_⟩
      intro ptrs hptrs i _
      rw [evalOpR_ref] at hptrs
      rw [attnIO_regs_chain (by decide) hqsp] at hptrs
      obtain rfl := Option.some.inj hptrs
      have hbound := hbQS ⟨0, hnum⟩ 0
      simpa [attnIOsclPtr] using hbound
  · -- after the prologue: the KV loop, then the postLoop
    intro sp hsp
    obtain ⟨spW, hpreW, hinv0⟩ :=
      attnIO_pre22W s Q K V QScale KScale Out stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
        HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE
    rw [stepStmtsR_triv, hpreW] at hsp
    obtain rfl := Option.some.inj hsp
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun s3 hs3 => ?_)
    · -- the KV loop is trace-safe (invariant principle over `attnIOSafeInv`)
      show Stmt.TraceSafeR .triv bounds _ spW
      simp only [Stmt.TraceSafeR]
      refine Stmt.forRangeTraceSafeR_inv .triv bounds "start_n" (BLOCK_N * numKVBlocks) BLOCK_N
        (RealLoop.attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
        (attnIOSafeInv K KScale V Out s stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
          BLOCK_DMODEL numKVBlocks)
        ?_ 0 spW hinv0
      intro c st hc hPc
      refine ⟨body_safe .triv bounds K KScale V Out s stride_qz stride_qh H HEAD_DIM
        BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hBN c st hc hPc hbK hbKS hbV, ?_⟩
      obtain ⟨st', hstep, hPc'⟩ := step K KScale V Out s stride_qz stride_qh H
        HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hBN c st hc hPc
      exact ⟨st', by
        rw [stepStmtsR_triv]
        exact hstep, hPc'⟩
    · -- identify the post-loop state and finish on the terminal store
      obtain ⟨final, sfin, hLoop, hfinal, hPfin⟩ :=
        forRange_inv (idx := "start_n") (start := 0) (stop := BLOCK_N * numKVBlocks)
          (step := BLOCK_N)
          (body := RealLoop.attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
          (P := attnIOSafeInv K KScale V Out s stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N
            BLOCK_DMODEL numKVBlocks)
          (s_init := spW) hBN.ne' hinv0
          (fun c st hc hPc => step K KScale V Out s stride_qz stride_qh H HEAD_DIM
            BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hBN c st hc hPc)
      rw [show stepStmtR .triv (Stmt.forRange "start_n" 0 (BLOCK_N * numKVBlocks) BLOCK_N
            (RealLoop.attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)) spW
          = some sfin from by
        rw [stepStmtR_triv]
        exact hLoop] at hs3
      obtain rfl := Option.some.inj hs3
      exact attnIO_postSafeW .triv bounds K KScale V Out s stride_qz stride_qh H HEAD_DIM
        BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks final sfin hPfin hbO


/-- The reciprocal rewrite preserves the complete identity-rounding safety walk. -/
theorem reciprocal_traceSafeR
    (bounds : RegionBounds) (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks : Nat)
    (s : BlockState) (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hbQ : ∀ (_t : Fin numKVBlocks) (j : Fin (BLOCK_M * BLOCK_DMODEL)),
      s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
        ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh
        + (s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL < bounds Q)
    (hbK : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_DMODEL * BLOCK_N)),
      j.val % BLOCK_N < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE →
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh
        + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM < bounds K)
    (hbV : ∀ (t : Fin numKVBlocks) (j : Fin (BLOCK_N * BLOCK_DMODEL)),
      j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks - t.val * BLOCK_N
        ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh
        + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL < bounds V)
    (hbQS : ∀ (_t : Fin numKVBlocks) (_j : Fin 1),
      s.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_M - 1) / BLOCK_M) + s.pids 0 < bounds QScale)
    (hbKS : ∀ (t : Fin numKVBlocks) (_j : Fin 1),
      s.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + t.val < bounds KScale)
    (hbO : ∀ j : Fin (BLOCK_M * BLOCK_DMODEL),
      s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
        ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE →
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh
        + (s.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
        < bounds Out) :
    (RealExecution.reciprocal Q K V QScale KScale Out stride_qz stride_qh Z H
      BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.TraceSafeR
        .triv bounds s := by
  apply (RealNormalization.trace_safetyR_triv_agrees Q K V QScale KScale Out
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE bounds s).mp
  exact original_traceSafeR bounds Q K V QScale KScale Out stride_qz stride_qh Z H
    HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks s hBN hnum
    hbQ hbK hbV hbQS hbKS hbO

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionForwardTriton.RealSafety
