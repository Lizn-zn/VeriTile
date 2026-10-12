import bench.tritonbench_g.attention_llama.Proofs.RealSafetyLoop

/-! Complete real-program trace safety. Each actual active Q/K/V load and Out
store is bounded by its allocation window. Loop invariants are constructed by
execution, independently of output equations or experimental FP assumptions. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.RealSafety
open VeriTile.Triton RealKernel RealSafetyLoop RealSafetyPrologue
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

theorem noncausal_safe (bounds : RegionBounds) (Q K V Out : RegionName)
    (s : BlockState) (sc : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      N_HEAD H SP BM BN BD numKV : Nat)
    (hBM : 0 < BM) (hBN : 0 < BN) (hundef : ∀ rg o, s.undef rg o = 0)
    (hQ : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alBase s N_HEAD sqz sqh + alRow s BM i.1 * sqm + i.2.1.val * sqk < bounds Q)
    (hK : ∀ r, r < BN * numKV → ∀ d : Fin BD,
      alBase s N_HEAD skz skh + r * skn + d.val * skk < bounds K)
    (hV : ∀ r, r < BN * numKV → ∀ d : Fin BD,
      alBase s N_HEAD svz svh + r * svk + d.val * svn < bounds V)
    (hO : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alOutOffset s N_HEAD soz soh som son BM i < bounds Out) :
    (RealKernel.noncausal Q K V Out sc
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      N_HEAD H (BN * numKV) SP BM BN BD).toAlgKernel.TraceSafe bounds s := by
  -- preamble
  obtain ⟨sp, hpre, hsppids, hspmem, hspundef, _hsm, hbi, hoh, hoffm, hoffn, hmp, hlp, hacc,
      hqreg, hkp, hvp, hbne⟩ :=
    alPreLoop_evalG s Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H (BN * numKV) BM BN BD hundef
  -- tile transport `s → sp` (the invariant is stated over sp)
  have htileQ : alQTileG sp Q N_HEAD sqz sqh sqm sqk H BM BD
      = alQTileG s Q N_HEAD sqz sqh sqm sqk H BM BD := by
    funext idx
    simp only [alQTileG, alRow, alBase, hsppids]
    unfold BlockState.readMem
    rw [hspmem]
  have htileK : alKTileG sp K N_HEAD skz skh skn skk (BN * numKV) BD
      = alKTileG s K N_HEAD skz skh skn skk (BN * numKV) BD := by
    funext idx
    simp only [alKTileG, alBase, hsppids]
    unfold BlockState.readMem
    rw [hspmem]
  have htileV : alVTileG sp V N_HEAD svz svh svk svn (BN * numKV) BD
      = alVTileG s V N_HEAD svz svh svk svn (BN * numKV) BD := by
    funext idx
    simp only [alVTileG, alBase, hsppids]
    unfold BlockState.readMem
    rw [hspmem]
  have hbaseK : alBase sp N_HEAD skz skh = alBase s N_HEAD skz skh := by
    simp only [alBase, hsppids]
  have hbaseV : alBase sp N_HEAD svz svh = alBase s N_HEAD svz svh := by
    simp only [alBase, hsppids]
  -- invariant base case at counter 0 (over sp)
  have hinv0 : alInv Q K V sp sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H BM BN BD numKV 0 sp := by
    refine alInv_zero Q K V sp sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H BM BN BD numKV hmp hlp hacc ?_ ?_ hoffn ?_ ?_ ?_ ?_ hspundef
    · rw [hqreg, htileQ]
    · rw [hoffm, hsppids]
    · rw [hbi, hsppids]
    · rw [hoh, hsppids]
    · rw [hkp, hbaseK]
    · rw [hvp, hbaseV]
  -- the streaming loop
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    forRangeDyn_inv (idx := "start_n") (startOp := Op.constNat 0)
      (stopOp := Op.ref .nat [] "block_n_end")
      (stepOp := Op.constNat BN)
      (P := fun i st => alInv Q K V sp sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
        N_HEAD H BM BN BD numKV i st)
      (s_init := sp)
      (by rw [evalOp_constNat])
      (by rw [evalOp_ref, hbne])
      (by rw [evalOp_constNat])
      (by omega)
      hinv0
      (fun i st hi hP => al_attn_step Q K V sp sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
        N_HEAD H BM BN BD numKV hBM hBN i st hi hP)
  -- final counter = BN * numKV
  simp only [alInv] at hinvL
  obtain ⟨hpidsL, _hmodL, hleL, _hmpL, _hlpL, haccL, _hqL, _hoffmL, _hoffnL, hbiL, hohL,
    _hkpL, _hvpL, _hundefL, _hmemL⟩ := hinvL
  have hfinal : final = BN * numKV := by omega
  subst hfinal
  rw [Nat.mul_div_cancel_left _ hBN] at haccL
  have hloopSafe : Stmt.TraceSafe bounds
      (.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
        (alLoopBodyG sc skn svk (BN * numKV) BM BN BD)) sp := by
    simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp_constNat, evalOp_ref,
      hbne, Tile.scalar_data, true_and]
    apply TraceProofs.range_safe
      (P := alInv Q K V sp sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
        N_HEAD H BM BN BD numKV) _ _ 0 sp hinv0
    · intro i st hi hp
      apply noncausal_iteration bounds Q K V sp sc
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn N_HEAD H BM BN BD numKV hBN i st hi hp
      · simpa only [alBase, hsppids] using hK
      · simpa only [alBase, hsppids] using hV
    · intro i st t hi hp hr
      obtain ⟨t', he, hp'⟩ := al_attn_step Q K V sp sc
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn N_HEAD H BM BN BD numKV hBM hBN i st hi hp
      rw [he] at hr
      cases hr
      exact hp'
  change Stmt.TraceSafeList bounds _ s
  rw [al_body_split]
  apply TraceProofs.append_safe
  · exact preamble_safe bounds s Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H (BN * numKV) BM BN BD hQ
  · intro t ht
    rw [hpre] at ht
    cases ht
    apply (TraceProofs.cons_of_run_iff hloop).mpr
    refine ⟨hloopSafe, ?_⟩
    apply postamble_safe bounds Out sp soz soh som son N_HEAD H BM BD sL hpidsL hbiL hohL
    simpa only [alActive, alRow, alOutOffset, alBase, hsppids] using hO


/-- Causal allocation bounds cover exactly the streamed span. Its program-ID
and start-position conditions are local to this execution, not universal over
all possible program IDs. -/
theorem causal_safe (bounds : RegionBounds) (Q K V Out : RegionName)
    (s : BlockState) (sc : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      N_HEAD H N_CTX SP BM BN BD numC : Nat)
    (hBM : 0 < BM) (hBN : 0 < BN)
    (hspanEq : (s.pids 0 + 1) * BN + SP = BN * numC)
    (hspanle : BN * numC ≤ N_CTX) (hsp : SP ≤ s.pids 0 * BM)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hQ : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alBase s N_HEAD sqz sqh + alRow s BM i.1 * sqm + i.2.1.val * sqk < bounds Q)
    (hK : ∀ r, r < BN * numC → ∀ d : Fin BD,
      alBase s N_HEAD skz skh + r * skn + d.val * skk < bounds K)
    (hV : ∀ r, r < BN * numC → ∀ d : Fin BD,
      alBase s N_HEAD svz svh + r * svk + d.val * svn < bounds V)
    (hO : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alOutOffset s N_HEAD soz soh som son BM i < bounds Out) :
    (RealKernel.causal Q K V Out sc
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      N_HEAD H N_CTX SP BM BN BD).toAlgKernel.TraceSafe bounds s := by
  -- preamble
  obtain ⟨sp, hpre, hsppids, hspmem, hspundef, hsm, hbi, hoh, hoffm, hoffn, hmp, hlp, hacc,
      hqreg, hkp, hvp, _hbne⟩ :=
    alPreLoop_evalG s Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H N_CTX BM BN BD hundef
  -- the causal block_n_end assignment
  have hcausal : stepStmt (alBnEndCausalStmt SP BN) sp =
      some (sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC))) := by
    apply stepStmt_assign_eq_some
    show evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)) (Op.constNat BN))
        (Op.constNat SP)) sp = some (Tile.scalar (BN * numC))
    rw [evalOp_add, evalOp_mul, evalOp_add]
    simp only [evalOp_ref, hsm, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    ext u
    show (s.pids 0 + 1) * BN + SP = BN * numC
    exact hspanEq
  set spc := sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC)) with hspc
  have hspcpids : spc.pids = s.pids := by rw [hspc, BlockState.setReg_pids, hsppids]
  have hspcmem : spc.mem = s.mem := by
    rw [hspc]; funext rg o; rw [BlockState.setReg_mem]
    exact congrFun (congrFun hspmem rg) o
  have hspcundef : ∀ rg o, spc.undef rg o = 0 := by
    intro rg o; rw [hspc, BlockState.setReg_undef]; exact hspundef rg o
  -- tile transport `s → spc`
  have htileQ : alQTileG spc Q N_HEAD sqz sqh sqm sqk H BM BD
      = alQTileG s Q N_HEAD sqz sqh sqm sqk H BM BD := by
    funext idx
    simp only [alQTileG, alRow, alBase, hspcpids]
    unfold BlockState.readMem
    rw [hspcmem]
  have htileK : alKTileG spc K N_HEAD skz skh skn skk (BN * numC) BD
      = alKTileG s K N_HEAD skz skh skn skk (BN * numC) BD := by
    funext idx
    simp only [alKTileG, alBase, hspcpids]
    unfold BlockState.readMem
    rw [hspcmem]
  have htileV : alVTileG spc V N_HEAD svz svh svk svn (BN * numC) BD
      = alVTileG s V N_HEAD svz svh svk svn (BN * numC) BD := by
    funext idx
    simp only [alVTileG, alBase, hspcpids]
    unfold BlockState.readMem
    rw [hspcmem]
  have hbaseK : alBase spc N_HEAD skz skh = alBase s N_HEAD skz skh := by
    simp only [alBase, hspcpids]
  have hbaseV : alBase spc N_HEAD svz svh = alBase s N_HEAD svz svh := by
    simp only [alBase, hspcpids]
  -- invariant base case at counter 0 (over spc)
  have hinv0 : alcInv Q K V spc sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H SP BM BN BD numC 0 spc := by
    refine alcInv_zero Q K V spc sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H SP BM BN BD numC ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ hspcundef
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmp
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hlp
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hacc
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [show alQTileG (sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC))) Q
            N_HEAD sqz sqh sqm sqk H BM BD
          = alQTileG s Q N_HEAD sqz sqh sqm sqk H BM BD from by rw [← hspc]; exact htileQ]
      exact hqreg
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [show (sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC))).pids = s.pids from by
        rw [← hspc]; exact hspcpids]
      exact hoffm
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffn
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [show (sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC))).pids = s.pids from by
        rw [← hspc]; exact hspcpids]
      exact hbi
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [show (sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC))).pids = s.pids from by
        rw [← hspc]; exact hspcpids]
      exact hoh
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [show alBase (sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC))) N_HEAD skz skh
          = alBase s N_HEAD skz skh from by rw [← hspc]; exact hbaseK]
      exact hkp
    · rw [hspc, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      rw [show alBase (sp.setReg "block_n_end" .nat [] (Tile.scalar (BN * numC))) N_HEAD svz svh
          = alBase s N_HEAD svz svh from by rw [← hspc]; exact hbaseV]
      exact hvp
  -- the streaming loop
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    forRangeDyn_inv (idx := "start_n") (startOp := Op.constNat 0)
      (stopOp := Op.ref .nat [] "block_n_end")
      (stepOp := Op.constNat BN)
      (P := fun i st => alcInv Q K V spc sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
        N_HEAD H SP BM BN BD numC i st)
      (s_init := spc)
      (by rw [evalOp_constNat])
      (by rw [evalOp_ref, hspc, BlockState.setReg_same])
      (by rw [evalOp_constNat])
      (by omega)
      hinv0
      (fun i st hi hP => alc_attn_step Q K V spc sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
        N_HEAD H N_CTX SP BM BN BD numC hBM hBN hspanle
        (by rw [hspcpids]; exact hsp) i st hi hP)
  -- final counter = BN * numC
  simp only [alcInv] at hinvL
  obtain ⟨hpidsL, _hmodL, hleL, _hmpL, _hlpL, haccL, _hqL, _hoffmL, _hoffnL, hbiL, hohL,
    _hkpL, _hvpL, _hundefL, _hmemL⟩ := hinvL
  have hfinal : final = BN * numC := by omega
  subst hfinal
  rw [Nat.mul_div_cancel_left _ hBN] at haccL
  have hstop : spc.regs .nat [] "block_n_end" = some (Tile.scalar (BN * numC)) := by
    rw [hspc, BlockState.setReg_same]
  have hloopSafe : Stmt.TraceSafe bounds
      (.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
        (alLoopBodyCausalG sc skn svk N_CTX SP BM BN BD)) spc := by
    simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp_constNat, evalOp_ref,
      hstop, Tile.scalar_data, true_and]
    apply TraceProofs.range_safe
      (P := alcInv Q K V spc sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
        N_HEAD H SP BM BN BD numC) _ _ 0 spc hinv0
    · intro i st hi hp
      apply causal_iteration bounds Q K V spc sc
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn N_HEAD H N_CTX SP BM BN BD numC hBN hspanle i st hi hp
      · simpa only [alBase, hspcpids] using hK
      · simpa only [alBase, hspcpids] using hV
    · intro i st t hi hp hr
      obtain ⟨t', he, hp'⟩ := alc_attn_step Q K V spc sc
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn N_HEAD H N_CTX SP BM BN BD numC
        hBM hBN hspanle (by rw [hspcpids]; exact hsp) i st hi hp
      rw [he] at hr
      cases hr
      exact hp'
  change Stmt.TraceSafeList bounds _ s
  rw [alc_body_split]
  apply TraceProofs.append_safe
  · exact preamble_safe bounds s Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      N_HEAD H N_CTX BM BN BD hQ
  · intro t ht
    rw [hpre] at ht
    cases ht
    apply (TraceProofs.cons_of_run_iff hcausal).mpr
    refine ⟨by simp [alBnEndCausalStmt, Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
    apply (TraceProofs.cons_of_run_iff hloop).mpr
    refine ⟨hloopSafe, ?_⟩
    apply postamble_safe bounds Out spc soz soh som son N_HEAD H BM BD sL hpidsL hbiL hohL
    simpa only [alActive, alRow, alOutOffset, alBase, hspcpids] using hO

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.RealSafety
