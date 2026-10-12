import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealInit

/-! Source correspondence for carry initialization, exact scale and Q load.
Selectors retain complete statements, including both arms of each branch.
These checks do not substitute fragments for the remaining prologue proof. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealInitSource
open VeriTile.Triton

def firstAssignment? : List Stmt → Option RegName
  | .assign _ _ name _ :: _ => some name
  | _ => none

def branch? (name : RegName) : List Stmt → Option Stmt
  | [] => none
  | .ifThenElse c yes no :: rest =>
      if firstAssignment? yes = some name ∧ firstAssignment? no = some name
      then some (.ifThenElse c yes no) else branch? name rest
  | _ :: rest => branch? name rest

def assignments (name : RegName) : List Stmt → List Stmt
  | [] => []
  | (.assign dt shape dest value) :: rest =>
      if dest = name then .assign dt shape dest value :: assignments name rest else assignments name rest
  | _ :: rest => assignments name rest

theorem first_assignment_erase (code : List Stmt) :
    firstAssignment? (Stmt.eraseDTypeList code) = firstAssignment? code := by
  cases code with
  | nil => simp [Stmt.eraseDTypeList, firstAssignment?]
  | cons st rest => cases st <;> simp [Stmt.eraseDTypeList, Stmt.eraseDType, firstAssignment?]

theorem branch_erase (name : RegName) (code : List Stmt) :
    branch? name (Stmt.eraseDTypeList code) = (branch? name code).map Stmt.eraseDType := by
  induction code with
  | nil => simp [Stmt.eraseDTypeList, branch?]
  | cons st rest ih =>
    cases st <;> simp only [Stmt.eraseDTypeList, Stmt.eraseDType, branch?]
    all_goals try exact ih
    rw [first_assignment_erase, first_assignment_erase]
    split <;> simp_all [Stmt.eraseDType]

theorem assignments_erase (name : RegName) (code : List Stmt) :
    assignments name (Stmt.eraseDTypeList code) = Stmt.eraseDTypeList (assignments name code) := by
  induction code with
  | nil => simp [Stmt.eraseDTypeList, assignments]
  | cons st rest ih =>
    cases st <;> simp only [Stmt.eraseDTypeList, Stmt.eraseDType, assignments]
    all_goals try exact ih
    split <;> simp_all [Stmt.eraseDTypeList, Stmt.eraseDType]

theorem erased_body (k : ComputeKernel) (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgKernel.body = Stmt.eraseDTypeList k.toAlgKernel.body := by
  rw [ComputeKernel.eraseDType, h]
  simp only [ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?_fromKernelBody, Kernel.eraseDType]

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
theorem original_fragments :
    let code := (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body
    branch? "m_i" code = some (RealInit.carryCode BLOCK_M BLOCK_DMODEL INIT) ∧
      assignments "qk_scale" code = RealInit.scaleCode sm_scale ∧
      branch? "q" code = some (RealInit.queryCode BLOCK_M BLOCK_DMODEL IS_EVEN_M) := by
  dsimp only
  rw [erased_body _ (by rfl), branch_erase, assignments_erase, branch_erase]
  dsimp only [TypedSource.original, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, branch?, firstAssignment?, assignments, Option.map]
  simp only [String.reduceEq, Option.some.injEq, reduceCtorEq, and_self, and_false, ite_true, ite_false,
    Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    RealInit.carryCode, RealInit.scaleCode, RealInit.queryCode]
  exact ⟨rfl, trivial⟩

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 16000 in
theorem candidate_fragments :
    let code := (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body
    branch? "m_i" code = some (RealInit.carryCode BLOCK_M BLOCK_DMODEL INIT) ∧
      assignments "qk_scale" code = RealInit.scaleCode sm_scale ∧
      branch? "q" code = some (RealInit.queryCode BLOCK_M BLOCK_DMODEL IS_EVEN_M) := by
  dsimp only
  rw [erased_body _ (by rfl), branch_erase, assignments_erase, branch_erase]
  dsimp only [TypedSource.reciprocal, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, branch?, firstAssignment?, assignments, Option.map]
  simp only [String.reduceEq, Option.some.injEq, reduceCtorEq, and_self, and_false, ite_true, ite_false,
    Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    RealInit.carryCode, RealInit.scaleCode, RealInit.queryCode]
  exact ⟨rfl, trivial⟩
end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealInitSource
