import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealLoopContext

/-! Connect the executed signed range to the complete loop statement in
both typed sources. The checks include the lo/hi registers and BLOCK_N stride,
not only the body. The prologue must still establish the displayed entry
registers, bounds and pointer metadata before this execution theorem applies. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopSource
open VeriTile.Triton

def statement (BM BN BD context window complement even : Nat) (offset size : Int) : Stmt :=
  .forRangeInt "start_n" (.ref .int [] "lo") (.ref .int [] "hi") (.constNat BN)
    (RealBody.body BM BN BD context window complement even offset size)

def loop? (index : RegName) : List Stmt → Option Stmt
  | [] => none
  | (.forRangeInt name start stop stride body) :: rest =>
      if name = index then some (.forRangeInt name start stop stride body) else loop? index rest
  | _ :: rest => loop? index rest

theorem loop_erase (index : RegName) (code : List Stmt) :
    loop? index (Stmt.eraseDTypeList code) = (loop? index code).map Stmt.eraseDType := by
  induction code with
  | nil => simp only [Stmt.eraseDTypeList, loop?, Option.map_none]
  | cons st rest ih =>
    cases st <;> simp only [Stmt.eraseDTypeList, Stmt.eraseDType, loop?]
    all_goals try exact ih
    split <;> simp_all [Stmt.eraseDType]

theorem erased_loop (k : ComputeKernel) (index : RegName)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    loop? index k.eraseDType.toAlgKernel.body = (loop? index k.toAlgKernel.body).map Stmt.eraseDType := by
  rw [ComputeKernel.eraseDType, h]
  simp only [ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?_fromKernelBody, Kernel.eraseDType]
  exact loop_erase index k.toAlgKernel.body

section
variable (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX
      IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW : Nat)
    (sliding_window_offset sliding_window_size : Int)

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 16000 in
theorem original_loop_source :
    loop? "start_n" (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body =
      some (statement BLOCK_M BLOCK_N BLOCK_DMODEL NKV_CTX SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
        IS_EVEN_N sliding_window_offset sliding_window_size) := by
  rw [erased_loop _ _ (by rfl)]
  dsimp only [TypedSource.original, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, loop?, Option.map]
  simp only [ite_true, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    statement, RealBody.body, RealBody.scores, RealBody.probabilities, RealBody.rescale,
    RealBody.accumulate, RealBody.loadKey, RealBody.tailMask,
    RealExpressions.maskStmt, RealExpressions.distanceExpr]
  rfl

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 16000 in
theorem candidate_loop_source :
    loop? "start_n" (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body =
      some (statement BLOCK_M BLOCK_N BLOCK_DMODEL NKV_CTX SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
        IS_EVEN_N sliding_window_offset sliding_window_size) := by
  rw [erased_loop _ _ (by rfl)]
  dsimp only [TypedSource.reciprocal, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, loop?, Option.map]
  simp only [ite_true, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    statement, RealBody.body, RealBody.scores, RealBody.probabilities, RealBody.rescale,
    RealBody.accumulate, RealBody.loadKey, RealBody.tailMask,
    RealExpressions.maskStmt, RealExpressions.distanceExpr]
  rfl
end

open RealLoop RealLoopFormula RealLoopContext RealScores in
/-- Execute exactly the source's for-range statement to full-context row
formulas. Bounds are read from the actual entry registers once, as in the
interpreter; no successful-loop or kernel-output premise is assumed. -/
theorem statement_executes {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (seed : Carry BM BD) (hN : 0 < BN) (hflag : p.even ≠ 0 → p.context % BN = 0)
    (hlo : p.lo = RealKeys.lower p.startM BM BN p.window p.complement p.offset p.size)
    (hhi : p.hi.toNat = RealKeys.upper p.startM BM BN p.context p.window p.complement p.offset)
    (hloreg : s.regs .int [] "lo" = some (Tile.scalar (p.lo : Int)))
    (hhireg : s.regs .int [] "hi" = some (Tile.scalar p.hi))
    (hregs : Registers p seed 0 s) :
    ∃ s', stepStmt (statement BM BN BD p.context p.window p.complement p.even p.offset p.size) s = some s' ∧
      Registers p (fullResult s p seed) (count p) s' ∧ Frame writes s s' := by
  obtain ⟨s', hs, hr, hframe⟩ := loop_full_context s p seed hN
    (full_blocks_of_source p hN hflag hlo hhi) hlo hhi hregs
  refine ⟨s', ?_, hr, hframe⟩
  simp only [statement, stepForRangeIntAux.forRangeInt_unfold, evalOp_ref, hloreg, hhireg,
    evalOp_constNat, Option.bind_some, Tile.scalar]
  exact hs

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopSource
