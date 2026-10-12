import bench.tritonbench_g.attention_llama.Proofs.FP8RealIteration
import bench.tritonbench_g.attention_llama.Proofs.RealOutput

/-! Identify the real reference used by the byte-input simulation with the
complete typed fp16 source already proved against the independent attention
formula. Only constant control-flow is normalized; no FP rule is assumed. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealReference
open VeriTile.Triton RealKernel FP8RealIteration
set_option maxRecDepth 16000
set_option maxHeartbeats 500000
set_option linter.unusedSimpArgs false

private def sourceLoop (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal candidate : Bool) : List Stmt :=
  alLoopPrefixG scale N BM BN BD ++
    .ifThen (.constBool causal) [alCausalWhereStmt SP BM BN] ::
      alLoopTailG skn svk N BM BN BD candidate

private theorem inline_loop (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal candidate : Bool) (s : BlockState) :
    stepStmts (sourceLoop scale skn svk N SP BM BN BD causal candidate) s =
      stepStmts (referenceLoop scale skn svk N SP BM BN BD causal candidate) s := by
  simp only [sourceLoop, referenceLoop, ProgramRewrite.append, ProgramRewrite.constant_branch]
  cases stepStmts (alLoopPrefixG scale N BM BN BD) s <;> rfl

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (causal candidate : Bool)

/-- The unchanged real program with the inner constant branch inlined. -/
def body : List Stmt :=
  alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N BM BN BD ++
    .ifThen (.constBool causal) [alBnEndCausalStmt SP BN] ::
    .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
      (referenceLoop scale skn svk N SP BM BN BD causal candidate) ::
      alPostLoopG Out soz soh som son H BM BD

private theorem actual_body :
    (RealOutput.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD candidate causal).eraseDType.toAlgKernel.body =
      alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N BM BN BD ++
        .ifThen (.constBool causal) [alBnEndCausalStmt SP BN] ::
        .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
          (sourceLoop scale skn svk N SP BM BN BD causal candidate) ::
          alPostLoopG Out soz soh som son H BM BD := by
  cases candidate <;>
    simp only [RealOutput.source, TypedSource.original, TypedSource.carryFirst,
      Bool.false_eq_true, ↓reduceIte, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel,
      ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
      MaskOpt.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
      List.map_cons, List.map_nil]
  all_goals
    simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil]
    rfl

/-- The reference body is the actual source with its numerical precision
erased; its existing real correctness theorem therefore applies. -/
theorem source_execution (s : BlockState) :
    exec (RealOutput.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD candidate causal).eraseDType.toAlgKernel s =
      stepStmts (body Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate) s := by
  simp only [exec, actual_body, body, ProgramRewrite.append, stepStmts,
    ProgramRewrite.dynamic_range_congr _ _ _ _ _ _
      (inline_loop scale skn svk N SP BM BN BD causal candidate)]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealReference
