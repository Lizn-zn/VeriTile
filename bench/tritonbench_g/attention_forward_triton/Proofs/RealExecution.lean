import bench.tritonbench_g.attention_forward_triton.Proofs.RealLoop
import bench.tritonbench_g.attention_forward_triton.Proofs.RealNormalization

/-! Complete real execution of the typed attention kernels. The proof combines
initialization, the erased streaming loop and the independent attention formula,
including the final masked store and the memory frame. Address safety is a
separate obligation for the public IO contract. -/
namespace VeriTile.Bench.TritonBenchG.AttentionForwardTriton.RealExecution
open VeriTile.Triton
open VeriTile.Examples.AttentionForwardClosedForm
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

section
variable (Q K V QS KS Out : RegionName)
    (sqz sqh Z H BM BN NC HD BD HA stage : Nat)

def original : ComputeKernel :=
  (TypedSource.original Q K V QS KS Out
    sqz sqh HD 1 sqz sqh HD 1 sqz sqh HD 1 sqz sqh HD 1
    Z H (BN * NC) HD BM BN BD HA stage).eraseDType

def reciprocal : ComputeKernel :=
  (TypedSource.reciprocal Q K V QS KS Out
    sqz sqh HD 1 sqz sqh HD 1 sqz sqh HD 1 sqz sqh HD 1
    Z H (BN * NC) HD BM BN BD HA stage).eraseDType

/-- The existing initialization proof uses the historical prologue. The body
identity below checks this prologue against the current complete typed source. -/
def prologue : List Stmt :=
  (VeriTile.Examples.AttentionForwardClosedForm.attention_forward_triton_surface
    Q K V QS KS Out sqz sqh HD 1 sqz sqh HD 1 sqz sqh HD 1 sqz sqh HD 1
    Z H (BN * NC) HD BM BN BD HA stage).toAlgKernel.body.take 22

theorem body_split :
    (original Q K V QS KS Out sqz sqh Z H BM BN NC HD BD HA stage).toAlgKernel.body =
      prologue Q K V QS KS Out sqz sqh Z H BM BN NC HD BD HA stage ++
      [.forRange "start_n" 0 (BN * NC) BN (RealLoop.attnLoopBody BM BN BD HA HD NC),
       attnAccAssign BM BD, attnStoreStmt BM BD BN NC HA] := by
  simp only [original, prologue, TypedSource.original,
    VeriTile.Examples.AttentionForwardClosedForm.attention_forward_triton_surface,
    RealLoop.attnLoopBody, attnLoopBody, attnAccAssign, attnStoreStmt,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind, List.take]
  rfl

end

private def outputPointers (s0 : BlockState) (rg : RegionName)
    (H sqz sqh HD BM BD : Nat) : Tile .ptr [BM, BD] :=
  ⟨fun idx => (rg.cast, baseOffset s0 H sqz sqh
    + (s0.pids 0 * BM + idx.1.val) * HD + idx.2.1.val)⟩

private theorem scatter_frame {α : Type} (region : RegionName)
    (offFn : α → Nat) (valFn : α → ℝ) (P : α → Prop) [DecidablePred P] :
    ∀ (l : List α) (s : BlockState) (r : RegionName) (o : Nat),
      (r = region → ∀ k ∈ l, P k → offFn k ≠ o) →
      ((l.foldl (fun acc k => if P k then acc.writeMem region (offFn k) (valFn k) else acc) s).mem r o
        = s.mem r o)
  | [], _, _, _, _ => rfl
  | k :: rest, s, r, o, h => by
      rw [List.foldl_cons]
      by_cases hPk : P k
      · rw [if_pos hPk,
          scatter_frame region offFn valFn P rest _ r o
            (fun hr k' hk' hP' => h hr k' (List.mem_cons_of_mem _ hk') hP'),
          BlockState.writeMem_mem]
        rw [if_neg (fun hro => h hro.1 k List.mem_cons_self hPk hro.2.symm)]
      · rw [if_neg hPk]
        exact scatter_frame region offFn valFn P rest s r o
          (fun hr k' hk' hP' => h hr k' (List.mem_cons_of_mem _ hk') hP')

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **PostLoop run + frame**: from the loop-exit register pins the `acc /= l_i`
rescale and the masked terminal store run to completion, and every cell outside
the write-active `Out` window is untouched. -/
private theorem post_run_frame (Out : RegionName) (s0 : BlockState)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (st : BlockState)
    (hoffsm : st.regs .nat [BLOCK_M] "offs_m"
      = some (Tile.vec (fun r : Fin BLOCK_M => s0.pids 0 * BLOCK_M + r.val)))
    (hOp : st.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
      = some (outputPointers s0 Out H stride_qz stride_qh HEAD_DIM BLOCK_M BLOCK_DMODEL))
    (lT : Tile .real [BLOCK_M]) (hli : st.regs .real [BLOCK_M] "l_i" = some lT)
    (aT : Tile .real [BLOCK_M, BLOCK_DMODEL])
    (hacc : st.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some aT) :
    ∃ sfin, stepStmts [attnAccAssign BLOCK_M BLOCK_DMODEL,
        attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE] st = some sfin
      ∧ ∀ r o,
          (r = Out → ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
            (s0.pids 0 * BLOCK_M + idx.1.val < BLOCK_N * numKVBlocks
              ∧ idx.2.1.val < HEAD_ACTIVE) →
            o ≠ baseOffset s0 H stride_qz stride_qh
              + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD_DIM + idx.2.1.val) →
          sfin.mem r o = st.mem r o := by
  have hax : 1 < [BLOCK_M].length.succ := by simp
  set acc' : Tile .real [BLOCK_M, BLOCK_DMODEL] :=
    Tile.bop NumericDType.real.div (Broadcast.consSame (Broadcast.consR Broadcast.nil)) aT
      (Tile.expandDim ⟨1, hax⟩ lT) with hacc'
  have hexpN : evalOp (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "l_i")) st
      = some (Tile.expandDim ⟨1, hax⟩ lT) := by rw [evalOp_expandDim]; simp [hli]
  have hexp2 : @evalOp TileDType.real [BLOCK_M, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "l_i")) st
      = some (Tile.expandDim ⟨1, hax⟩ lT) := hexpN
  have hAccEval : evalOp (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "l_i"))) st = some acc' := by
    rw [evalOp_div]
    simp only [evalOp_ref, hacc, hexp2, Option.bind_eq_bind, Option.bind_some]
    rfl
  set st1 := st.setReg "acc" .real [BLOCK_M, BLOCK_DMODEL] acc' with hst1
  set oOffFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx => baseOffset s0 H stride_qz stride_qh
      + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD_DIM + idx.2.1.val with hoOffFn
  set P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => s0.pids 0 * BLOCK_M + idx.1.val < BLOCK_N * numKVBlocks
      ∧ idx.2.1.val < HEAD_ACTIVE with hPdef
  have hacc2 : st1.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some acc' := by
    rw [hst1]; simp only [BlockState.setReg_same]
  have hopEval : evalOp (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr") st1
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out.cast, oOffFn idx)⟩
          : Tile .ptr [BLOCK_M, BLOCK_DMODEL]) := by
    rw [evalOp_ref, hst1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hOp]
    rfl
  have hmaskEval : evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
      (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
        (Op.constNat (BLOCK_N * numKVBlocks)))
      (Op.expandDim ⟨0, by simp⟩
        (Op.lt .nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) st1
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => decide (P idx)⟩
          : Tile .bool [BLOCK_M, BLOCK_DMODEL]) := by
    rw [qmask_eval st1 BLOCK_M BLOCK_DMODEL (BLOCK_N * numKVBlocks) HEAD_ACTIVE
      (fun r : Fin BLOCK_M => s0.pids 0 * BLOCK_M + r.val)
      (by rw [hst1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsm)]
    refine congrArg some ?_
    ext idx
    simp [hPdef]
  have hstore : stepStmt (attnStoreStmt BLOCK_M BLOCK_DMODEL BLOCK_N numKVBlocks HEAD_ACTIVE) st1
      = some ((TileShape.allIndices [BLOCK_M, BLOCK_DMODEL]).foldl
          (fun acc idx => if P idx then acc.writeMem Out (oOffFn idx) ((acc'.data idx).unbotD 0)
            else acc) st1) := by
    simp only [attnStoreStmt, stepStmt, evalOp_ref, hacc2, hopEval, hmaskEval,
      Option.bind_eq_bind, Option.bind_some, Option.map_some, decide_eq_true_eq]
    refine congrArg some ?_
    refine List.foldl_ext _ _ st1 ?_
    intro acc idx _
    by_cases hk : P idx
    · simp only [if_pos hk, Region.cast_id, BlockState.writeMemTyped_real,
        FloatDType.real_storeValue]
    · simp only [if_neg hk]
  rw [attnAccAssign, stepStmts.cons_some (stepStmt_assign_eq_some hAccEval),
    stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro r o hguard
  rw [scatter_frame Out oOffFn (fun idx => (acc'.data idx).unbotD 0) P
    (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL]) st1 r o
    (fun hr k _ hPk heq => hguard hr k hPk heq.symm)]
  rw [hst1]
  simp only [BlockState.setReg_mem]


section
variable (Q K V QS KS Out : RegionName)
    (sqz sqh Z H BM BN NC HD BD HA stage : Nat)

/-- The target is the mathematical base-2, per-key-scale attention formula;
it does not mention the kernel's execution or accumulator registers. -/
noncomputable def expected (s : BlockState) (idx : TileIndex [BM, BD]) : ℝ :=
  if h : idx.2.1.val < HA then
    attentionRealBase2PerKeyScale
      (qTile s Q H sqz sqh HD BM HA)
      (kTile s K H sqz sqh HD (BN * NC) HA)
      (vTile s V H sqz sqh HD (BN * NC) HA)
      (keyScale s QS KS (BN * NC) BM BN (BN * NC))
      (idx.1, ⟨idx.2.1.val, h⟩, PUnit.unit)
  else 0

def active (s : BlockState) (idx : TileIndex [BM, BD]) : Prop :=
  s.pids 0 * BM + idx.1.val < BN * NC ∧ idx.2.1.val < HA

/-- The mathematical result and every cell outside the actual write mask. -/
def postcondition (s t : BlockState) : Prop :=
  (∀ idx : TileIndex [BM, BD], active BM BN NC BD HA s idx →
    t.readMem Out (outOffset s H sqz sqh HD 1 BM BD idx) =
      expected Q K V QS KS sqz sqh H BM BN NC HD BD HA s idx) ∧
  (∀ r o, (r = Out → ∀ idx : TileIndex [BM, BD], active BM BN NC BD HA s idx →
      o ≠ outOffset s H sqz sqh HD 1 BM BD idx) → t.mem r o = s.mem r o)

theorem original_exec_correct (s : BlockState)
    (hBN : 0 < BN) (hNC : 0 < NC) (hHA : HA ≤ BD) (hHD : HA ≤ HD)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ t, exec (original Q K V QS KS Out sqz sqh Z H BM BN NC HD BD HA stage).toAlgKernel s = some t ∧
      postcondition Q K V QS KS Out sqz sqh H BM BN NC HD BD HA s t := by
  let P := attnInvariant Q K V QS KS Out s sqz sqh H BM BN NC HD BD HA
  have hpre : ∃ t, stepStmts (prologue Q K V QS KS Out sqz sqh Z H BM BN NC HD BD HA stage) s = some t ∧ P 0 t :=
    preLoop Q K V QS KS Out s sqz sqh H BM BN NC HD BD HA stage hundef
  obtain ⟨s0, hpre, hP0⟩ := hpre
  obtain ⟨final, sLoop, hloop, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "start_n") (start := 0) (stop := BN * NC) (step := BN) hBN.ne' hP0
      (fun i st hlt hP => RealLoop.attn_step Q K V QS KS Out s sqz sqh H BM BN NC HD BD HA hBN hHA i st hlt hP)
  have hle : final ≤ BN * NC := by
    obtain ⟨_, hi, hc, _⟩ := hPLoop
    calc final = final / BN * BN := hi
      _ ≤ NC * BN := Nat.mul_le_mul_right _ hc
      _ = BN * NC := Nat.mul_comm _ _
  have hfinalEq : final = BN * NC := le_antisymm hle hfinal
  subst final
  have hpins := hPLoop
  dsimp only [P, attnInvariant] at hpins
  obtain ⟨_, _, _, _, hli, hacc, _, _, _, hoffsm, _, hop, _, _, _, _, hmem⟩ := hpins
  obtain ⟨t, htail, hframe⟩ := post_run_frame Out s sqz sqh H HD BM BN BD HA NC sLoop
    (by simpa only [mIndex] using hoffsm) (by simpa only [outputPointers, mIndex] using hop)
    _ hli _ hacc
  refine ⟨t, ?_, ?_, ?_⟩
  · rw [exec, body_split, stepStmts.append_some hpre, stepStmts.cons_some hloop, htail]
  · intro idx ha
    obtain ⟨t', htail', hvalue⟩ := attn_postLoop Q K V QS KS Out s sqz sqh H BM BN NC HD BD HA
      hBN hNC hHD idx ha sLoop hPLoop
    rw [htail] at htail'
    obtain rfl := Option.some.inj htail'
    simpa only [expected, dif_pos ha.2] using hvalue
  · intro r o hguard
    rw [hframe r o (by simpa only [active, outOffset, mIndex, mul_one] using hguard), hmem]

theorem reciprocal_exec_correct (s : BlockState)
    (hBN : 0 < BN) (hNC : 0 < NC) (hHA : HA ≤ BD) (hHD : HA ≤ HD)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ t, exec (reciprocal Q K V QS KS Out sqz sqh Z H BM BN NC HD BD HA stage).toAlgKernel s = some t ∧
      postcondition Q K V QS KS Out sqz sqh H BM BN NC HD BD HA s t := by
  have h := RealNormalization.executions_agree Q K V QS KS Out
    sqz sqh HD 1 sqz sqh HD 1 sqz sqh HD 1 sqz sqh HD 1
    Z H (BN * NC) HD BM BN BD HA stage s
  change exec (original Q K V QS KS Out sqz sqh Z H BM BN NC HD BD HA stage).toAlgKernel s =
    exec (reciprocal Q K V QS KS Out sqz sqh Z H BM BN NC HD BD HA stage).toAlgKernel s at h
  rw [← h]
  exact original_exec_correct Q K V QS KS Out sqz sqh Z H BM BN NC HD BD HA stage s hBN hNC hHA hHD hundef

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionForwardTriton.RealExecution
