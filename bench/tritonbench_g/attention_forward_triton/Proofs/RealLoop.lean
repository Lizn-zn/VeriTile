import VeriTile.Examples.AttentionForwardClosedForm
import bench.tritonbench_g.attention_forward_triton.Kernels
import VeriTile.Meta.StatementAudit

/-! Execution of the erased attention loop and preservation of the real
online-softmax invariant. The recurrence and mathematical tile lemmas are
reused from AttentionForwardClosedForm; the execution chain below handles
the erased casts and real register slots. -/
namespace VeriTile.Bench.TritonBenchG.AttentionForwardTriton.RealLoop
open VeriTile.Triton
open VeriTile.Examples.AttentionForwardClosedForm
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def attnLoopBody (BM BN BD HA HD NC : Nat) : List Stmt :=
  Stmt.eraseDTypeList (VeriTile.Examples.AttentionForwardClosedForm.attnLoopBody BM BN BD HA HD NC)

theorem qk_op_eval (s : BlockState) (BM BN BD : Nat)
    (qtile : Tile .real [BM, BD]) (ktile : Tile .real [BD, BN]) (qsv ksv : ℝ)
    (hq : s.regs .real [BM, BD] "q" = some qtile) (hk : s.regs .real [BD, BN] "k" = some ktile)
    (hqs : s.regs .real [] "q_scale" = some (Tile.scalar (some qsv)))
    (hks : s.regs .real [] "k_scale" = some (Tile.scalar (some ksv))) :
    evalOp (Op.mul .real Broadcast.scalarR
        (Op.mul .real Broadcast.scalarR
          (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"))
          (Op.ref .real [] "q_scale")) (Op.ref .real [] "k_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR
         (Tile.bop NumericDType.real.mul Broadcast.scalarR
           (⟨fun i => FloatDType.real.cast FloatDType.real ((Tile.dot [] qtile ktile).data i)⟩ : Tile .real [BM,BN])
           (Tile.scalar (some qsv : WithBot ℝ))) (Tile.scalar (some ksv : WithBot ℝ))) := by
  have hraw : evalOp (Op.dot (batch := []) (Op.ref .real [BM,BD] "q")
      (Op.ref .real [BD,BN] "k")) s = some (Tile.dot [] qtile ktile) := by
    rw [evalOp_dot]
    simp [hq, hk]
  have hdot : @evalOp .real [BM,BN] (Op.dot (batch := [])
      (Op.ref .real [BM,BD] "q") (Op.ref .real [BD,BN] "k")) s =
      some (Tile.dot [] qtile ktile) := hraw
  simp only [evalOp_mul, hdot, evalOp_ref, hq, hk, hqs, hks,
    Option.bind_eq_bind, Option.bind_some]
  rfl

theorem acc2_op_eval (s : BlockState) (BM BN BD : Nat)
    (acc1tile : Tile .real [BM,BD]) (ptile : Tile .real [BM,BN]) (vtile : Tile .real [BN,BD])
    (hacc : s.regs .real [BM,BD] "acc" = some acc1tile)
    (hp : s.regs .real [BM,BN] "p" = some ptile)
    (hv : s.regs .real [BN,BD] "v" = some vtile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) (Op.ref .real [BM,BD] "acc")
        (Op.dot (options := { outDType := some .fp16 }) (batch := [])
          (Op.ref .real [BM,BN] "p") (Op.ref .real [BN,BD] "v"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acc1tile
          (Tile.dot [] ptile vtile)) := by
  have hraw : evalOp (Op.dot (options := { outDType := some .fp16 }) (batch := [])
      (Op.ref .real [BM,BN] "p") (Op.ref .real [BN,BD] "v")) s =
      some (Tile.dot [] ptile vtile) := by
    rw [evalOp_dot]
    simp [hp, hv]
  have hdot : @evalOp .real [BM,BD] (Op.dot (options := { outDType := some .fp16 })
      (batch := []) (Op.ref .real [BM,BN] "p") (Op.ref .real [BN,BD] "v")) s =
      some (Tile.dot [] ptile vtile) := hraw
  simp only [evalOp_add, hdot, evalOp_ref, hacc, hp, hv,
    Option.bind_eq_bind, Option.bind_some]
  rfl

theorem attn_loopBody_steps (BM BN BD NC HA HD : Nat) (hBN : 0 < BN)
    (hax : 1 < [BM].length.succ) (sin : BlockState)
    (SN : TileCarrier .nat) (qsv : ℝ)
    (Kptr : Tile .ptr [BD, BN]) (Ksptr : Tile .ptr []) (Vptr : Tile .ptr [BN, BD])
    (qtile : Tile .real [BM, BD]) (mtile ltile : Tile .real [BM]) (acctile : Tile .real [BM, BD])
    (hoffs : sin.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hKp : sin.regs .ptr [BD, BN] "K_ptrs" = some Kptr) (hKs : sin.regs .ptr [] "K_scale_ptr" = some Ksptr)
    (hVp : sin.regs .ptr [BN, BD] "V_ptrs" = some Vptr) (hq : sin.regs .real [BM, BD] "q" = some qtile)
    (hqs : sin.regs .real [] "q_scale" = some (Tile.scalar (some qsv))) (hmi : sin.regs .real [BM] "m_i" = some mtile)
    (hli : sin.regs .real [BM] "l_i" = some ltile) (hacc : sin.regs .real [BM, BD] "acc" = some acctile)
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sF, stepStmts (attnLoopBody BM BN BD HA HD NC) (sin) = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ sF.regs .ptr [BD, BN] "K_ptrs" = some (Tile.ptrAdd Broadcast.scalarR Kptr (Tile.scalar (BN * HD)))
      ∧ sF.regs .ptr [] "K_scale_ptr" = some (Tile.ptrAdd Broadcast.nil Ksptr (Tile.scalar 1))
      ∧ sF.regs .ptr [BN, BD] "V_ptrs" = some (Tile.ptrAdd Broadcast.scalarR Vptr (Tile.scalar (BN * HD)))
      ∧ sF.regs .real [BM, BD] "q" = some qtile
      ∧ sF.regs .real [] "q_scale" = some (Tile.scalar (some qsv))
      ∧ sF.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))
      ∧ sF.regs .nat [BM] "offs_m" = sin.regs .nat [BM] "offs_m"
      ∧ sF.regs .nat [BD] "offs_k" = sin.regs .nat [BD] "offs_k"
      ∧ sF.regs .ptr [BM, BD] "O_block_ptr" = sin.regs .ptr [BM, BD] "O_block_ptr"
      ∧ ∃ (rmaxT mijT alphaT : Tile .real [BM]) (qkT pT : Tile .real [BM, BN]) (lijT : Tile .real [BM])
            (acc1T : Tile .real [BM, BD]) (vloadT : Tile .real [BN, BD]),
          qkT = (Tile.bop NumericDType.real.mul Broadcast.scalarR
                (Tile.bop NumericDType.real.mul Broadcast.scalarR
                  (⟨fun i => FloatDType.real.cast FloatDType.real ((Tile.dot [] qtile
                      (⟨fun i => some (if (⟨fun idx => (decide (idx.2.1.val < BN * NC - SN) && decide (idx.1.val < HA))⟩ : Tile .bool [BD, BN]).data i then sin.readMem (Kptr.data i).1 (Kptr.data i).2 else 0)⟩ : Tile .real [BD, BN])).data i)⟩ : Tile .real [BM,BN])
                  (Tile.scalar (some qsv : WithBot ℝ)))
                (Tile.scalar (some (sin.readMem (Ksptr.data PUnit.unit).1 (Ksptr.data PUnit.unit).2) : WithBot ℝ)))
          ∧ Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM,BN].length) qkT = some rmaxT
          ∧ mijT = Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT
          ∧ alphaT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT)
          ∧ pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkT (Tile.expandDim ⟨1, hax⟩ mijT))
          ∧ lijT = Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) pT
          ∧ acc1T = Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, hax⟩ alphaT)
          ∧ vloadT = (⟨fun i => some (if (⟨fun idx => (decide (idx.1.val < BN * NC - SN) && decide (idx.2.1.val < HA))⟩ : Tile .bool [BN, BD]).data i then sin.readMem (Vptr.data i).1 (Vptr.data i).2 else 0)⟩ : Tile .real [BN, BD])
          ∧ sF.regs .real [BM] "m_i" = some mijT
          ∧ sF.regs .real [BM] "l_i" = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
              (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile alphaT) lijT)
          ∧ sF.regs .real [BM, BD] "acc" = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              acc1T (Tile.dot [] pT vloadT)) := by
  set kmaskT : Tile .bool [BD, BN] := ⟨fun idx => (decide (idx.2.1.val < BN * NC - SN) && decide (idx.1.val < HA))⟩ with hkm
  set kloadT : Tile .real [BD, BN] := ⟨fun i => some (if kmaskT.data i then sin.readMem (Kptr.data i).1 (Kptr.data i).2 else 0)⟩ with hkl
  set ksv : ℝ := sin.readMem (Ksptr.data PUnit.unit).1 (Ksptr.data PUnit.unit).2 with hksv
  set qkT : Tile .real [BM, BN] := Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (⟨fun i => FloatDType.real.cast FloatDType.real ((Tile.dot [] qtile kloadT).data i)⟩ : Tile .real [BM,BN])
        (Tile.scalar (some qsv : WithBot ℝ))) (Tile.scalar (some ksv : WithBot ℝ)) with hqk
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM,BN].length) qkT = some t :=
    ⟨_, by unfold Tile.reduceMaxDrop; rw [dif_pos (show 0 < TileShape.axisDim [BM,BN] (⟨1, by simp⟩ : Fin [BM,BN].length) from hBN)]⟩
  set mijT : Tile .real [BM] := Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT with hmij
  set qk2T : Tile .real [BM, BN] := Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, hax⟩ mijT) with hqk2
  set pT : Tile .real [BM, BN] := Tile.uop WithBot.realExp2 qk2T with hpT
  set lijT : Tile .real [BM] := Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) pT with hlij
  set alphaT : Tile .real [BM] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT) with hal
  set acc1T : Tile .real [BM, BD] := Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, hax⟩ alphaT) with hacc1
  set vmaskT : Tile .bool [BN, BD] := ⟨fun idx => (decide (idx.1.val < BN * NC - SN) && decide (idx.2.1.val < HA))⟩ with hvm
  set vloadT : Tile .real [BN, BD] := ⟨fun i => some (if vmaskT.data i then sin.readMem (Vptr.data i).1 (Vptr.data i).2 else 0)⟩ with hvl
  simp only [attnLoopBody, VeriTile.Examples.AttentionForwardClosedForm.attnLoopBody,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType, FloatDType.toTileDType, List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.ref .nat [] "start_n") sin = some (Tile.scalar SN) from by rw [evalOp_ref]; exact hsn))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (kmask_eval _ BD BN (BN * NC) SN HA (by simp [hoffs]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_ptr_mask_real (Op.ref .ptr [BD, BN] "K_ptrs") _ _ Kptr kmaskT (by rw [evalOp_ref]; simp [hKp]) (by rw [evalOp_ref]; simp [hkm]) (by intro rg o; simp [hundef])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_ptr_none_real (Op.ref .ptr [] "K_scale_ptr") _ Ksptr (by rw [evalOp_ref]; simp [hKs])))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (qk_op_eval _ BM BN BD qtile kloadT qsv ksv (by simp [hq]) (by simp [hkl]) (by simp [hqs]) (by simp [hksv]; ext z; rfl)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mij_op_eval _ BM BN mtile qkT rmaxT (by simp [hmi]) (by simp [hqk]) hrm))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (qk2_op_eval _ BM BN hax qkT mijT (by simp [hqk]) (by simp [hmij])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (p_op_eval _ BM BN qk2T (by simp [hqk2])))]
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .real [BM] "l_ij" (Op.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM,BN] "p")) _ lijT (lij_op_eval _ BM BN pT (by simp [hpT])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (alpha_op_eval _ BM mtile mijT (by simp [hmi]) (by simp [hmij])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (li_op_eval _ BM ltile alphaT lijT (by simp [hli]) (by simp [hal]) (by simp [hlij])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (acc1_op_eval _ BM BD hax acctile alphaT (by simp [hacc]) (by simp [hal])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_ptr_mask_real (Op.ref .ptr [BN, BD] "V_ptrs") _ _ Vptr vmaskT (by rw [evalOp_ref]; simp [hVp]) (vmask_eval _ BN BD (BN * NC) SN HA (by simp [hoffs]) (by simp)) (by intro rg o; simp [hundef])))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.ref .real [BM,BN] "p") _ = some pT from by rw [evalOp_ref]; simp [hpT]))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (acc2_op_eval _ BM BN BD acc1T pT vloadT (by simp [hacc1]) (by simp [hpT]) (by simp [hvl])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.ref .real [BM] "m_ij") _ = some mijT from by rw [evalOp_ref]; simp [hmij]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (kptr_adv_eval _ BD BN BN HD Kptr "K_ptrs" (by simp [hKp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (ksptr_adv_eval _ Ksptr "K_scale_ptr" (by simp [hKs])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (kptr_adv_eval _ BN BD BN HD Vptr "V_ptrs" (by simp [hVp])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext region offset; simp
  · intro rg o; simp [hundef]
  · simp
  · simp
  · simp
  · simp [hq]
  · simp [hqs]
  · simp [hoffs]
  · simp
  · simp
  · simp
  · exact ⟨rmaxT, mijT, alphaT, qkT, pT, lijT, acc1T, vloadT, rfl, hrm, rfl, rfl, rfl, rfl, rfl, rfl,
      by simp, by simp, by simp⟩




/-- One erased iteration advances the independent streaming-attention invariant. -/
theorem attn_step (Q K V Q_scale K_scale Out : RegionName) (s0 : BlockState)
    (stride_qz stride_qh H BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE : Nat)
    (hBN : 0 < BLOCK_N) (hHA : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (i : Nat) (s : BlockState) (hilt : i < BLOCK_N * numKVBlocks)
    (hinv : attnInvariant Q K V Q_scale K_scale Out s0 stride_qz stride_qh H BLOCK_M BLOCK_N
        numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE i s) :
    ∃ s', stepStmts (attnLoopBody BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE HEAD_DIM numKVBlocks)
        (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ attnInvariant Q K V Q_scale K_scale Out s0 stride_qz stride_qh H BLOCK_M BLOCK_N
          numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE (i + BLOCK_N) s' := by
  have hax : 1 < [BLOCK_M].length.succ := by simp
  have hc : i / BLOCK_N < numKVBlocks := (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [Nat.mul_comm]; exact hilt)
  have hc1 : (i + BLOCK_N) / BLOCK_N = i / BLOCK_N + 1 := Nat.add_div_right i hBN
  simp only [attnInvariant] at hinv
  obtain ⟨hpids, hi, hcle, hmi, hli, hacc, hq, hqs, hn, hm, hk, hO, hKp, hKs, hVp, hundef, hmem⟩ := hinv
  -- streaming functions (match the invariant's lets)
  set qT := qTileMasked s0 Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE (BLOCK_N * numKVBlocks) with hqTdef
  set kT := kTile s0 K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE with hkTdef
  set vT := vTile s0 V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE with hvTdef
  set kS := keyScale s0 Q_scale K_scale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N (BLOCK_N * numKVBlocks) with hkSdef
  obtain ⟨sF, hchain, hpidsF, hmemF, hundefF, hKpF, hKsF, hVpF, hqF, hqsF, hnF, hmF, hkF, hOF,
      rmaxT, mijT, alphaT, qkT, pT, lijT, acc1T, vloadT, hqkTd, hrm, hmijd, halphad, hpTd, hlijd, hacc1d, hvloadd,
      hm_iF, hl_iF, haccF⟩ :=
    attn_loopBody_steps BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks HEAD_ACTIVE HEAD_DIM hBN hax
      (s.setReg "start_n" .nat [] (Tile.scalar i)) i
      (s0.readMem Q_scale (s0.pids 1 * VeriTile.Examples.AttentionForwardClosedForm.cdiv (BLOCK_N * numKVBlocks) BLOCK_M + s0.pids 0))
      (⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] => (K.cast, baseOffset s0 H stride_qz stride_qh + idx.1.val + idx.2.1.val * HEAD_DIM + i / BLOCK_N * BLOCK_N * HEAD_DIM)⟩)
      (Tile.scalar (K_scale.cast, s0.pids 1 * VeriTile.Examples.AttentionForwardClosedForm.cdiv (BLOCK_N * numKVBlocks) BLOCK_N + i / BLOCK_N))
      (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => (V.cast, baseOffset s0 H stride_qz stride_qh + idx.1.val * HEAD_DIM + idx.2.1.val + i / BLOCK_N * BLOCK_N * HEAD_DIM)⟩)
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (if (mIndex s0 BLOCK_M idx.1 < BLOCK_N * numKVBlocks ∧ idx.2.1.val < HEAD_ACTIVE) then s0.readMem Q (baseOffset s0 H stride_qz stride_qh + mIndex s0 BLOCK_M idx.1 * HEAD_DIM + idx.2.1.val) else 0)⟩)
      (⟨fun r : TileIndex [BLOCK_M] => mP BLOCK_N numKVBlocks qT kT kS r.1 (i / BLOCK_N)⟩)
      (⟨fun r : TileIndex [BLOCK_M] => ((lP qT kT kS r.1 (i / BLOCK_N) : ℝ) : WithBot ℝ)⟩)
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => if h : idx.2.1.val < HEAD_ACTIVE then ((oP qT kT vT kS idx.1 ⟨idx.2.1.val, h⟩ (i / BLOCK_N) : ℝ) : WithBot ℝ) else some 0⟩)
      (by simp [hn]) (by simp) (by simp [hKp]) (by simp [hKs]) (by simp [hVp]) (by simp [hq])
      (by simp [hqs]) (by simp [hmi]) (by simp [hli]) (by simp [hacc]) (by intro rg o; simp [hundef])
  refine ⟨sF, hchain, ?_⟩
  -- the loop body reads the (unchanged) memory of `s0`
  have hrmem : ∀ (R : RegionName) (o : Nat),
      (s.setReg "start_n" .nat [] (Tile.scalar i)).readMem R o = s0.readMem R o := by
    intro R o; simp only [BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  simp only [hrmem] at hqkTd hvloadd
  -- per-cell `qk = kscore` bridge (feeds m_ij / l_i)
  have hqk : ∀ (r : Fin BLOCK_M) (jL : Fin BLOCK_N),
      qkT.data (r, jL, PUnit.unit) = some (kscore BLOCK_N numKVBlocks qT kT kS r (i / BLOCK_N) hc jL) := by
    intro r jL
    rw [hqkTd]
    refine qk_eq s0 Q K Q_scale K_scale H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE numKVBlocks (i / BLOCK_N) hHA hc hBN r jL _ _ (fun a e => rfl) ?_
    intro e j
    simp only [← hi]
    by_cases hcond : (j.val < BLOCK_N * numKVBlocks - i ∧ e.val < HEAD_ACTIVE)
    · simp only [if_pos hcond, Region.cast_id]
      refine congrArg some ?_
      rw [if_pos (by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact hcond)]
    · simp only [if_neg hcond]
      refine congrArg some ?_
      rw [if_neg (by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact hcond)]
  have hmij_eq : ∀ idx : TileIndex [BLOCK_M], mijT.data idx
      = mP BLOCK_N numKVBlocks qT kT kS idx.1 (i / BLOCK_N + 1) := by
    intro idx
    rw [hmijd]
    refine mij_eq BLOCK_M BLOCK_N numKVBlocks (i / BLOCK_N) hc qT kT kS _ rmaxT idx.1 ?_ ?_
    · rfl
    · exact reduceMaxDrop_data_row BLOCK_M BLOCK_N hBN _ rmaxT hrm idx.1 _ (fun jL => hqk idx.1 jL)
  simp only [attnInvariant]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF]; exact hpids
  · rw [hc1, Nat.add_one_mul, ← hi]
  · rw [hc1]; omega
  · -- m_i = mP (c+1)
    rw [hm_iF, hc1]; refine congrArg some ?_; ext idx; exact hmij_eq idx
  · -- l_i = lP (c+1)
    rw [hl_iF, hc1]; refine congrArg some ?_; ext idx
    rw [halphad, hlijd, hpTd]
    exact li_eq BLOCK_M BLOCK_N numKVBlocks (i / BLOCK_N) hBN hc qT kT kS _ mijT _ _ idx.1
      rfl (hmij_eq idx) rfl (fun jL => hqk idx.1 jL)
  · -- acc = oP (c+1)
    rw [haccF, hc1]; refine congrArg some ?_; ext idx
    rw [hacc1d, halphad, hpTd, hvloadd]
    refine acc_eq BLOCK_M BLOCK_N BLOCK_DMODEL numKVBlocks (i / BLOCK_N) HEAD_ACTIVE hc qT kT vT kS
      _ _ _ _ idx.1 idx.2.1 ?_ ?_ ?_ ?_
    · rfl
    · exact alpha_eq BLOCK_M BLOCK_N numKVBlocks (i / BLOCK_N) qT kT kS _ mijT idx.1 rfl
        (hmij_eq (idx.1, PUnit.unit))
    · intro jL
      refine exp2_some (fun a b => kscore BLOCK_N numKVBlocks qT kT kS a (i / BLOCK_N) hc b
        - mR qT kT kS a (i / BLOCK_N + 1)) _ idx.1 jL ?_
      simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
        TileShape.dropInsertedIndex, TileShape.insertAxisIndex, hqk idx.1 jL,
        hmij_eq (idx.1, PUnit.unit),
        mP_eq_coe qT kT kS hBN idx.1 (i / BLOCK_N + 1) (Nat.le_add_left 1 (i / BLOCK_N)) hc,
        NumericDType.sub, WithBot.realSub, Option.map₂, Option.bind, Option.map]
    · intro jL
      have hjL : jL.val < BLOCK_N * numKVBlocks - i := by
        have hle : i + BLOCK_N ≤ BLOCK_N * numKVBlocks := by
          rw [hi]
          calc i / BLOCK_N * BLOCK_N + BLOCK_N = (i / BLOCK_N + 1) * BLOCK_N := by ring
            _ ≤ numKVBlocks * BLOCK_N := Nat.mul_le_mul_right _ hc
            _ = BLOCK_N * numKVBlocks := by ring
        have := jL.isLt; omega
      simp only [decide_eq_true_eq.mpr hjL, Bool.true_and, Region.cast_id]
      by_cases hd : idx.2.1.val < HEAD_ACTIVE
      · rw [if_pos (decide_eq_true_eq.mpr hd), dif_pos hd]
        refine congrArg some ?_
        show s0.readMem V _ = vTile s0 V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks)
            HEAD_ACTIVE (VeriTile.Examples.AttentionForwardClosedForm.gkey BLOCK_N numKVBlocks (i / BLOCK_N) hc jL, ⟨idx.2.1.val, hd⟩, PUnit.unit)
        unfold vTile VeriTile.Examples.AttentionForwardClosedForm.gkey
        congr 1
        ring
      · rw [if_neg (by simp only [decide_eq_true_eq]; exact hd), dif_neg hd]
  · rw [hqF]
  · rw [hqsF]
  · rw [hnF]
  · rw [hmF]; exact hm
  · rw [hkF]; exact hk
  · rw [hOF]; exact hO
  · -- K_ptrs advances c → c+1
    rw [hKpF, hc1]; refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar]
    ring
  · -- K_scale_ptr advances
    rw [hKsF, hc1]; refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar]
    ring
  · -- V_ptrs advances
    rw [hVpF, hc1]; refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar]
    ring
  · exact hundefF
  · rw [hmemF]; exact hmem

private def sourceLoop (body : List Stmt) : Option (List Stmt) :=
  body.findSome? fun
    | .forRange name _ _ _ code => if name = "start_n" then some code else none
    | _ => none

/-- The loop verified above is exactly the loop in the complete typed source.
The extractor selects its induction-variable name, not a statement offset. -/
theorem loop_matches_source
    (q k v qs ks out : RegionName)
    (sqz sqh Z H BM BN NC HD BD HA stage : Nat) :
    sourceLoop (TypedSource.original q k v qs ks out
      sqz sqh HD 1 sqz sqh HD 1 sqz sqh HD 1 sqz sqh HD 1
      Z H (BN * NC) HD BM BN BD HA stage).eraseDType.toAlgKernel.body =
      some (attnLoopBody BM BN BD HA HD NC) := by
  simp only [sourceLoop, TypedSource.original, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind, List.findSome?,
    attnLoopBody, VeriTile.Examples.AttentionForwardClosedForm.attnLoopBody]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionForwardTriton.RealLoop
