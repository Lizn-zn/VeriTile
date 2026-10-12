import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealWindow

/-! Source correspondence for the entire window branch, including clipping,
alignment hint, both pointer advances and the unreachable complement arm. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealWindowSource
open VeriTile.Triton

def startsBranch : List Stmt → Bool
  | .ifThenElse _ _ _ :: _ => Bool.true
  | _ => Bool.false

def window? : List Stmt → Option Stmt
  | [] => none
  | .ifThenElse c yes no :: rest =>
      if startsBranch yes then some (.ifThenElse c yes no) else window? rest
  | _ :: rest => window? rest

theorem starts_branch_erase (code : List Stmt) :
    startsBranch (Stmt.eraseDTypeList code) = startsBranch code := by
  cases code with
  | nil => simp [startsBranch, Stmt.eraseDTypeList]
  | cons st rest => cases st <;> simp [startsBranch, Stmt.eraseDTypeList, Stmt.eraseDType]

theorem window_erase (code : List Stmt) :
    window? (Stmt.eraseDTypeList code) = (window? code).map Stmt.eraseDType := by
  induction code with
  | nil => simp [window?, Stmt.eraseDTypeList]
  | cons st rest ih =>
    cases st <;> simp only [Stmt.eraseDTypeList, Stmt.eraseDType, window?]
    all_goals try exact ih
    rw [starts_branch_erase]
    split <;> simp_all [Stmt.eraseDType]

theorem advance_key_erase (BN BD : Nat) :
    (Op.advanceBlockPtrDyn (.ref .blockPtr [BD, BN] "K_block_ptr")
      [.constInt 0, .ref .int [] "lo"]).eraseDType =
    .advanceBlockPtrDyn (.ref .blockPtr [BD, BN] "K_block_ptr") [.constInt 0, .ref .int [] "lo"] := by
  rw [Op.eraseDType.eq_def]
  change Op.advanceBlockPtrDyn (Op.ref .blockPtr [BD, BN] "K_block_ptr").eraseDType
    [(Op.constInt 0).eraseDType, (Op.ref .int [] "lo").eraseDType] = _
  simp only [Op.eraseDType.eq_def]
  rfl

theorem advance_value_erase (BN BD : Nat) :
    (Op.advanceBlockPtrDyn (.ref .blockPtr [BN, BD] "V_block_ptr")
      [.ref .int [] "lo", .constInt 0]).eraseDType =
    .advanceBlockPtrDyn (.ref .blockPtr [BN, BD] "V_block_ptr") [.ref .int [] "lo", .constInt 0] := by
  rw [Op.eraseDType.eq_def]
  change Op.advanceBlockPtrDyn (Op.ref .blockPtr [BN, BD] "V_block_ptr").eraseDType
    [(Op.ref .int [] "lo").eraseDType, (Op.constInt 0).eraseDType] = _
  simp only [Op.eraseDType.eq_def]
  rfl

theorem code_erase (BM BN BD context window complement : Nat) (offset size : Int) :
    Stmt.eraseDType (RealWindow.code BM BN BD context window complement offset size) =
      RealWindow.code BM BN BD context window complement offset size := by
  simp only [RealWindow.code, RealWindow.clampLower, RealWindow.clampUpper, RealWindow.advance,
    Stmt.eraseDType, Stmt.eraseDTypeList, List.cons_append, List.nil_append]
  rw [advance_key_erase, advance_value_erase]
  simp only [RealWindow.lowerExpr, RealWindow.upperExpr, RealWindow.complementUpperExpr,
    Op.truncDiv, Op.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType]
  rfl

theorem erased_window (k : ComputeKernel) (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    window? k.eraseDType.toAlgKernel.body = (window? k.toAlgKernel.body).map Stmt.eraseDType := by
  rw [RealInitSource.erased_body k h]
  exact window_erase _

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
theorem original_window :
    window? (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body =
      some (RealWindow.code BLOCK_M BLOCK_N BLOCK_DMODEL NKV_CTX SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
        sliding_window_offset sliding_window_size) := by
  have h : window? (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).toAlgKernel.body =
      some (RealWindow.code BLOCK_M BLOCK_N BLOCK_DMODEL NKV_CTX SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
        sliding_window_offset sliding_window_size) := by rfl
  exact (erased_window _ (by rfl)).trans
    ((congrArg (fun st => st.map Stmt.eraseDType) h).trans
      (congrArg some (code_erase BLOCK_M BLOCK_N BLOCK_DMODEL NKV_CTX SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
        sliding_window_offset sliding_window_size)))

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 16000 in
theorem candidate_window :
    window? (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body =
      some (RealWindow.code BLOCK_M BLOCK_N BLOCK_DMODEL NKV_CTX SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
        sliding_window_offset sliding_window_size) := by
  have h : window? (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).toAlgKernel.body =
      some (RealWindow.code BLOCK_M BLOCK_N BLOCK_DMODEL NKV_CTX SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
        sliding_window_offset sliding_window_size) := by rfl
  exact (erased_window _ (by rfl)).trans
    ((congrArg (fun st => st.map Stmt.eraseDType) h).trans
      (congrArg some (code_erase BLOCK_M BLOCK_N BLOCK_DMODEL NKV_CTX SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
        sliding_window_offset sliding_window_size)))

end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealWindowSource
