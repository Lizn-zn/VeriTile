import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.FullExecution

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

theorem var_safe (bounds : RegionBounds) (s0 s : BlockState) (X Y : RegionName)
    (stride N B c : Nat) (h : VarInv s0 X Y stride N B c s)
    (hbx : ∀ k : Fin N, s0.pids 0 * stride + k.val < bounds X) :
    Stmt.TraceSafeListR .triv bounds (varBody N B)
      (s.setReg "off" .nat [] (Tile.scalar (c * B))) := by
  simp [varBody, Stmt.TraceSafeListR, Stmt.TraceSafeR,
    Op.SafeAtR.eq_def, MemAccess.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    stepStmtR_triv, evalOpR_triv, stepStmt, evalOp.eq_def,
    h.xptr, h.acc, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select, Tile.scalar, -Tile.scalar_eta,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, Region.cast]
  intro j hj
  exact hbx ⟨c * B + j.val, hj⟩

theorem write_safe (bounds : RegionBounds) (s0 s : BlockState) (X Y W : RegionName)
    (stride N B c : Nat) (eps : ℝ) (h : WriteContext s0 X Y W stride N eps s)
    (hbx : ∀ k : Fin N, s0.pids 0 * stride + k.val < bounds X)
    (hbw : ∀ k : Fin N, k.val < bounds W)
    (hby : ∀ k : Fin N, s0.pids 0 * stride + k.val < bounds Y) :
    Stmt.TraceSafeListR .triv bounds (writeBody W N B)
      (s.setReg "off" .nat [] (Tile.scalar (c * B))) := by
  simp [writeBody, Stmt.TraceSafeListR, Stmt.TraceSafeR,
    Op.SafeAtR.eq_def, MemAccess.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    stepStmtR_triv, evalOpR_triv, stepStmt, evalOp.eq_def,
    h.xptr, h.yptr, h.rstd, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, Region.cast]
  exact ⟨fun j hj => hbw ⟨c * B + j.val, hj⟩,
    fun j hj => hbx ⟨c * B + j.val, hj⟩,
    fun j hj => hby ⟨c * B + j.val, hj⟩⟩

theorem var_loop_safe (bounds : RegionBounds) (s0 s : BlockState) (X Y : RegionName)
    (stride N B : Nat) (h : VarInv s0 X Y stride N B 0 s)
    (hbx : ∀ k : Fin N, s0.pids 0 * stride + k.val < bounds X) :
    Stmt.TraceSafeR .triv bounds (.forRange "off" 0 N B (varBody N B)) s := by
  simp only [Stmt.TraceSafeR]
  apply Stmt.forRangeTraceSafeR_inv .triv bounds "off" N B (varBody N B)
    (fun off st => ∃ c, off = c * B ∧ VarInv s0 X Y stride N B c st)
    ?_ 0 s ⟨0, by simp, h⟩
  intro off st _ hst
  rcases hst with ⟨c, rfl, hst⟩
  refine ⟨var_safe bounds s0 st X Y stride N B c hst hbx,
    varState s0 st X stride N B c, ?_, c+1, ?_, var_inv s0 st X Y stride N B c hst⟩
  · rw [stepStmtsR_triv]
    exact var_exec s0 st X Y stride N B c hst
  · simp [Nat.add_mul]

theorem write_loop_safe (bounds : RegionBounds) (s0 s : BlockState) (X Y W : RegionName)
    (stride N B : Nat) (eps : ℝ) (h : WriteInv s0 X Y W stride N eps 0 s)
    (hXY : X ≠ Y) (hWY : W ≠ Y)
    (hbx : ∀ k : Fin N, s0.pids 0 * stride + k.val < bounds X)
    (hbw : ∀ k : Fin N, k.val < bounds W)
    (hby : ∀ k : Fin N, s0.pids 0 * stride + k.val < bounds Y) :
    Stmt.TraceSafeR .triv bounds (.forRange "off" 0 N B (writeBody W N B)) s := by
  simp only [Stmt.TraceSafeR]
  apply Stmt.forRangeTraceSafeR_inv .triv bounds "off" N B (writeBody W N B)
    (fun off st => ∃ c, off = c * B ∧ WriteInv s0 X Y W stride N eps off st)
    ?_ 0 s ⟨0, by simp, h⟩
  intro off st _ hst
  rcases hst with ⟨c, rfl, hst⟩
  refine ⟨write_safe bounds s0 st X Y W stride N B c eps hst.context hbx hbw hby,
    writeState s0 st X Y W stride N B c eps, ?_, c+1, ?_, ?_⟩
  · rw [stepStmtsR_triv]
    exact write_exec s0 st X Y W stride N B c eps hst.context
  · simp [Nat.add_mul]
  · simpa [Nat.add_mul] using write_inv s0 st X Y W stride N B c eps hst hXY hWY

theorem pre_safe (bounds : RegionBounds) (s : BlockState) (X Y : RegionName)
    (stride B : Nat) : Stmt.TraceSafeListR .triv bounds (pre X Y stride B) s := by
  apply Stmt.TraceSafeListR.of_forall
  intro stmt hst st
  simp only [pre, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl <;> simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]

theorem post_safe (bounds : RegionBounds) (s : BlockState) (N B : Nat) (eps : ℝ) :
    Stmt.TraceSafeListR .triv bounds (post N B eps) s := by
  apply Stmt.TraceSafeListR.of_forall
  intro stmt hst st
  simp only [post, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl <;> simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]

/-- Safety follows the exact intermediate states and both loop invariants. -/
theorem real_trace_safe (bounds : RegionBounds) (s0 : BlockState) (X Y W : RegionName)
    (stride N B : Nat) (eps : ℝ) (hB : 0 < B) (hXY : X ≠ Y) (hWY : W ≠ Y)
    (hbx : ∀ k : Fin N, s0.pids 0 * stride + k.val < bounds X)
    (hbw : ∀ k : Fin N, k.val < bounds W)
    (hby : ∀ k : Fin N, s0.pids 0 * stride + k.val < bounds Y) :
    (mathSource X Y W stride N B eps).toAlgKernel.TraceSafeR .triv bounds s0 := by
  unfold Kernel.TraceSafeR
  rw [body_decomp]
  simp only [List.append_assoc, List.cons_append, List.nil_append]
  refine Stmt.TraceSafeListR.append_intro _ _ (pre_safe bounds s0 X Y stride B) ?_
  intro sp hsp
  rw [stepStmtsR_triv, pre_exec] at hsp
  have heq := Option.some.inj hsp
  subst sp
  refine Stmt.TraceSafeListR.cons_intro
    (var_loop_safe bounds s0 _ X Y stride N B (pre_inv s0 X Y stride N B) hbx) ?_
  intro sv hsv
  rw [stepStmtR_triv] at hsv
  obtain ⟨c, sv', hv, hc, hi⟩ := var_loop s0 (preState s0 X Y stride B) X Y
    stride N B hB (pre_inv s0 X Y stride N B)
  have heq : sv = sv' := Option.some.inj (hsv.symm.trans hv)
  subst sv
  refine Stmt.TraceSafeListR.append_intro _ _ (post_safe bounds sv' N B eps) ?_
  intro sp hsp
  rw [stepStmtsR_triv, post_exec s0 sv' X Y stride N B c eps hB hc hi] at hsp
  have heq := Option.some.inj hsp
  subst sp
  have hpost : WriteInv s0 X Y W stride N eps 0 (postState s0 sv' X stride N eps) := by
    refine ⟨post_context s0 sv' X Y W stride N B c eps hi, ?_, ?_⟩
    · intro k; simp [postState, BlockState.readMem, hi.mem]
    · intro r o _; simpa [postState] using congrFun (congrFun hi.mem r) o
  refine Stmt.TraceSafeListR.cons_intro
    (write_loop_safe bounds s0 _ X Y W stride N B eps hpost hXY hWY hbx hbw hby) ?_
  intro t _
  exact Stmt.TraceSafeListR.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General
