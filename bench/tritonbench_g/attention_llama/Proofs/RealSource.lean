import bench.tritonbench_g.attention_llama.Kernels
import bench.tritonbench_g.attention_llama.Proofs.RealKernel
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Memory.TraceRewrite
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Connect the complete typed source to the independently analyzed real
programs. Constant causal branches are inlined by execution and access-safety congruence;
no floating-point admission assumption participates in this bridge. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.RealSource
open VeriTile.Triton RealKernel
set_option maxRecDepth 16000
set_option maxHeartbeats 500000
set_option linter.unusedSimpArgs false

private def sourceLoop (sm_scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (IS_CAUSAL : Bool) (carryFirst : Bool := Bool.false) : List Stmt :=
  alLoopPrefixG sm_scale N BM BN BD ++
    .ifThen (.constBool IS_CAUSAL) [alCausalWhereStmt SP BM BN] ::
      alLoopTailG skn svk N BM BN BD carryFirst

section
variable (Q K V Out : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (IS_CAUSAL : Bool)

private theorem original_body :
    (TypedSource.original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).eraseDType.toAlgKernel.body =
      alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N BM BN BD ++
        .ifThen (.constBool IS_CAUSAL) [alBnEndCausalStmt SP BN] ::
        .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
          (sourceLoop sm_scale skn svk N SP BM BN BD IS_CAUSAL) :: alPostLoopG Out soz soh som son H BM BD := by
  simp only [TypedSource.original, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
    MaskOpt.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
    List.map_cons, List.map_nil]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil]
  rfl

private theorem candidate_body :
    (TypedSource.carryFirst Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).eraseDType.toAlgKernel.body =
      alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N BM BN BD ++
        .ifThen (.constBool IS_CAUSAL) [alBnEndCausalStmt SP BN] ::
        .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
          (sourceLoop sm_scale skn svk N SP BM BN BD IS_CAUSAL Bool.true) :: alPostLoopG Out soz soh som son H BM BD := by
  simp only [TypedSource.carryFirst, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
    MaskOpt.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
    List.map_cons, List.map_nil]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil]
  rfl

/-- The typed source has an actual real algorithm projection. -/
theorem original_projection_supported :
    (TypedSource.original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).eraseDType.toAlgorithm? =
      .ok (TypedSource.original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).eraseDType.toAlgKernel := by
  have h : (TypedSource.original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).toAlgorithm? =
      .ok (TypedSource.original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).toAlgKernel := rfl
  simp only [ComputeKernel.eraseDType, h]
  simp

/-- The candidate also has a real algorithm projection. -/
theorem candidate_projection_supported :
    (TypedSource.carryFirst Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).eraseDType.toAlgorithm? =
      .ok (TypedSource.carryFirst Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).eraseDType.toAlgKernel := by
  have h : (TypedSource.carryFirst Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).toAlgorithm? =
      .ok (TypedSource.carryFirst Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD IS_CAUSAL).toAlgKernel := rfl
  simp only [ComputeKernel.eraseDType, h]
  simp

end

private theorem loop_noncausal (sm_scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (s : BlockState) :
    stepStmts (sourceLoop sm_scale skn svk N SP BM BN BD Bool.false) s =
      stepStmts (alLoopBodyG sm_scale skn svk N BM BN BD) s := by
  simp only [sourceLoop, alLoopBodyG, ProgramRewrite.append,
    ProgramRewrite.constant_branch, Bool.false_eq_true, ↓reduceIte, List.nil_append]

private theorem loop_causal (sm_scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (s : BlockState) :
    stepStmts (sourceLoop sm_scale skn svk N SP BM BN BD Bool.true) s =
      stepStmts (alLoopBodyCausalG sm_scale skn svk N SP BM BN BD) s := by
  simp only [sourceLoop, alLoopBodyCausalG, ProgramRewrite.append,
    ProgramRewrite.constant_branch, ↓reduceIte, List.singleton_append]

private theorem real_add_comm (a b : WithBot ℝ) :
    WithBot.realAdd a b = WithBot.realAdd b a := by
  cases a <;> cases b
  · rfl
  · rfl
  · rfl
  · simp [WithBot.realAdd, add_comm]

private theorem vector_add_comm (a b : Op .real [BM]) (s : BlockState) :
    evalOp (.add .real Broadcast.nil.consSame a b) s =
      evalOp (.add .real Broadcast.nil.consSame b a) s := by
  simp only [evalOp_add]
  cases ha : evalOp a s <;> cases hb : evalOp b s
  all_goals simp only [Option.bind_eq_bind, Option.bind_none, Option.bind_some]
  rename_i va vb
  congr 1
  ext i
  exact real_add_comm _ _

private theorem candidate_tail (skn svk N BM BN BD : Nat) (s : BlockState) :
    stepStmts (alLoopTailG skn svk N BM BN BD Bool.true) s =
      stepStmts (alLoopTailG skn svk N BM BN BD Bool.false) s := by
  have h := vector_add_comm (Op.ref .real [BM] "l_prev")
    (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
      (Op.ref .real [BM, BN] "p"))
  simp only [alLoopTailG, Bool.false_eq_true, ↓reduceIte, stepStmts, stepStmt, h]

private theorem candidate_loop (sm_scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal : Bool) (s : BlockState) :
    stepStmts (sourceLoop sm_scale skn svk N SP BM BN BD causal Bool.true) s =
      stepStmts (sourceLoop sm_scale skn svk N SP BM BN BD causal Bool.false) s := by
  simp only [sourceLoop, ProgramRewrite.append, stepStmts, candidate_tail]

private theorem loop_noncausal_safe (bounds : RegionBounds) (sm_scale : ℝ)
    (skn svk N SP BM BN BD : Nat) (s : BlockState) :
    Stmt.TraceSafeList bounds (sourceLoop sm_scale skn svk N SP BM BN BD Bool.false) s ↔
      Stmt.TraceSafeList bounds (alLoopBodyG sm_scale skn svk N BM BN BD) s := by
  simp only [sourceLoop, alLoopBodyG, TraceRewrite.append_iff,
    TraceRewrite.constant_branch, Bool.false_eq_true, ↓reduceIte, List.nil_append]

private theorem loop_causal_safe (bounds : RegionBounds) (sm_scale : ℝ)
    (skn svk N SP BM BN BD : Nat) (s : BlockState) :
    Stmt.TraceSafeList bounds (sourceLoop sm_scale skn svk N SP BM BN BD Bool.true) s ↔
      Stmt.TraceSafeList bounds (alLoopBodyCausalG sm_scale skn svk N SP BM BN BD) s := by
  simp only [sourceLoop, alLoopBodyCausalG, TraceRewrite.append_iff,
    TraceRewrite.constant_branch, ↓reduceIte, List.singleton_append]

private theorem candidate_tail_safe (bounds : RegionBounds)
    (skn svk N BM BN BD : Nat) (s : BlockState) :
    Stmt.TraceSafeList bounds (alLoopTailG skn svk N BM BN BD Bool.true) s ↔
      Stmt.TraceSafeList bounds (alLoopTailG skn svk N BM BN BD Bool.false) s := by
  have h := vector_add_comm (Op.ref .real [BM] "l_prev")
    (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
      (Op.ref .real [BM, BN] "p"))
  simp only [alLoopTailG, Bool.false_eq_true, ↓reduceIte, Stmt.TraceSafeList,
    Stmt.TraceSafe, Op.SafeAt.eq_def, stepStmt, h, true_and]

private theorem candidate_loop_safe (bounds : RegionBounds) (sm_scale : ℝ)
    (skn svk N SP BM BN BD : Nat) (causal : Bool) (s : BlockState) :
    Stmt.TraceSafeList bounds (sourceLoop sm_scale skn svk N SP BM BN BD causal Bool.true) s ↔
      Stmt.TraceSafeList bounds (sourceLoop sm_scale skn svk N SP BM BN BD causal Bool.false) s := by
  simp only [sourceLoop, TraceRewrite.append_iff, Stmt.TraceSafeList, candidate_tail_safe]

section
variable (Q K V Out : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat)

/-- In real semantics the candidate preserves the complete execution result,
including failure, for either causal setting. This uses no FP rule. -/
theorem candidate_execution (causal : Bool) (s : BlockState) :
    exec (TypedSource.carryFirst Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel s =
      exec (TypedSource.original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel s := by
  simp only [exec, candidate_body, original_body, ProgramRewrite.append]
  congr 1
  funext t
  simp only [stepStmts, ProgramRewrite.dynamic_range_congr _ _ _ _ _ _
    (candidate_loop sm_scale skn svk N SP BM BN BD causal)]

/-- All non-causal execution effects agree with the real proof's program. -/
theorem original_noncausal_execution (s : BlockState) :
    exec (TypedSource.original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD Bool.false).eraseDType.toAlgKernel s =
      exec (RealKernel.noncausal Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD).toAlgKernel s := by
  simp only [exec, original_body, al_body_split, ProgramRewrite.append,
    ProgramRewrite.constant_branch, Bool.false_eq_true, ↓reduceIte, List.nil_append]
  congr 1
  funext t
  simp only [stepStmts, ProgramRewrite.dynamic_range_congr _ _ _ _ _ _
    (loop_noncausal sm_scale skn svk N SP BM BN BD)]

/-- All causal execution effects agree with the real proof's program. -/
theorem original_causal_execution (s : BlockState) :
    exec (TypedSource.original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD Bool.true).eraseDType.toAlgKernel s =
      exec (RealKernel.causal Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD).toAlgKernel s := by
  simp only [exec, original_body, alc_body_split, ProgramRewrite.append,
    ProgramRewrite.constant_branch, ↓reduceIte, List.singleton_append]
  congr 1
  funext t
  simp only [stepStmts, ProgramRewrite.dynamic_range_congr _ _ _ _ _ _
    (loop_causal sm_scale skn svk N SP BM BN BD)]

/-- The same real rewrite also preserves the safety of all intermediate
accesses; it is not inferred from equality of final states. -/
theorem candidate_safety (bounds : RegionBounds) (causal : Bool) (s : BlockState) :
    (TypedSource.carryFirst Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel.TraceSafe bounds s ↔
      (TypedSource.original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel.TraceSafe bounds s := by
  simp only [Kernel.TraceSafe, candidate_body, original_body, TraceRewrite.append_iff,
    Stmt.TraceSafeList,
    TraceRewrite.dynamic_range_congr bounds _ _ _ _ _ _
      (candidate_loop sm_scale skn svk N SP BM BN BD causal)
      (candidate_loop_safe bounds sm_scale skn svk N SP BM BN BD causal),
    ProgramRewrite.dynamic_range_congr _ _ _ _ _ _
      (candidate_loop sm_scale skn svk N SP BM BN BD causal)]

theorem original_noncausal_safety (bounds : RegionBounds) (s : BlockState) :
    (TypedSource.original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD Bool.false).eraseDType.toAlgKernel.TraceSafe bounds s ↔
      (RealKernel.noncausal Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD).toAlgKernel.TraceSafe bounds s := by
  simp only [Kernel.TraceSafe, original_body, al_body_split, TraceRewrite.append_iff,
    TraceRewrite.constant_branch, Bool.false_eq_true, ↓reduceIte, List.nil_append,
    Stmt.TraceSafeList,
    TraceRewrite.dynamic_range_congr bounds _ _ _ _ _ _
      (loop_noncausal sm_scale skn svk N SP BM BN BD)
      (loop_noncausal_safe bounds sm_scale skn svk N SP BM BN BD),
    ProgramRewrite.dynamic_range_congr _ _ _ _ _ _
      (loop_noncausal sm_scale skn svk N SP BM BN BD)]

theorem original_causal_safety (bounds : RegionBounds) (s : BlockState) :
    (TypedSource.original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD Bool.true).eraseDType.toAlgKernel.TraceSafe bounds s ↔
      (RealKernel.causal Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD).toAlgKernel.TraceSafe bounds s := by
  simp only [Kernel.TraceSafe, original_body, alc_body_split, TraceRewrite.append_iff,
    TraceRewrite.constant_branch, ↓reduceIte, List.singleton_append,
    Stmt.TraceSafeList,
    TraceRewrite.dynamic_range_congr bounds _ _ _ _ _ _
      (loop_causal sm_scale skn svk N SP BM BN BD)
      (loop_causal_safe bounds sm_scale skn svk N SP BM BN BD),
    ProgramRewrite.dynamic_range_congr _ _ _ _ _ _
      (loop_causal sm_scale skn svk N SP BM BN BD)]

/-- Every operation in the erased original is supported by flat-memory transport. -/
theorem original_flatten_supported (causal : Bool) :
    (TypedSource.original Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, original_body, alPreLoopG, sourceLoop, alLoopPrefixG,
    alLoopTailG, alCausalWhereStmt, alBnEndCausalStmt, alPostLoopG,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  simp only [Op.FlattenOk.eq_def, true_and, and_true]

/-- Every operation in the erased candidate is supported by flat-memory transport. -/
theorem candidate_flatten_supported (causal : Bool) :
    (TypedSource.carryFirst Q K V Out sm_scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, candidate_body, alPreLoopG, sourceLoop, alLoopPrefixG,
    alLoopTailG, alCausalWhereStmt, alBnEndCausalStmt, alPostLoopG,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  simp only [Op.FlattenOk.eq_def, true_and, and_true]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.RealSource
