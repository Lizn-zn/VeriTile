import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealEpilogue

/-! Check the complete source suffix after the signed loop, including both END
arms and all stores. Splitting at the loop preserves every source statement;
no literal statement index is used to define the verified suffix. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealEpilogueSource
open VeriTile.Triton RealPrologueSource

def afterLoop : List Stmt → List Stmt
  | .forRangeInt _ _ _ _ _ :: rest => rest
  | _ :: rest => afterLoop rest
  | [] => []

theorem after_loop_erase (body : List Stmt) :
    afterLoop (Stmt.eraseDTypeList body) = Stmt.eraseDTypeList (afterLoop body) := by
  induction body with
  | nil => simp [afterLoop, Stmt.eraseDTypeList]
  | cons st rest ih => cases st <;> simp [afterLoop, Stmt.eraseDTypeList, Stmt.eraseDType, ih]

theorem through_after_loop (body : List Stmt) : throughLoop body ++ afterLoop body = body := by
  induction body with
  | nil => rfl
  | cons st rest ih => cases st <;> simp [throughLoop, afterLoop, ih]

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
theorem original_epilogue :
    afterLoop (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body =
      RealEpilogue.code BLOCK_M BLOCK_DMODEL END Bool.false := by
  rw [RealInitSource.erased_body _ (by rfl), after_loop_erase]
  dsimp only [TypedSource.original, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, afterLoop]
  simp only [Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    RealEpilogue.code, RealEpilogue.endBranch, RealEpilogue.normalizeCode,
    RealEpilogue.denominatorStore, RealEpilogue.centerStore, RealEpilogue.outputStore,
    RealEpilogue.quotientExpr, RealEpilogue.denominatorExpr]
  rfl

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 16000 in
theorem candidate_epilogue :
    afterLoop (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body =
      RealEpilogue.code BLOCK_M BLOCK_DMODEL END Bool.true := by
  rw [RealInitSource.erased_body _ (by rfl), after_loop_erase]
  dsimp only [TypedSource.reciprocal, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, afterLoop]
  simp only [Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    RealEpilogue.code, RealEpilogue.endBranch, RealEpilogue.normalizeCode,
    RealEpilogue.denominatorStore, RealEpilogue.centerStore, RealEpilogue.outputStore,
    RealEpilogue.quotientExpr, RealEpilogue.denominatorExpr]
  rfl

end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealEpilogueSource
