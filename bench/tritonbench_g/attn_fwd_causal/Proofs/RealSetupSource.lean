import bench.tritonbench_g.attn_fwd_causal.Proofs.RealSetup

/-! Locate the actual statements surrounding the helper ranges by loop name,
not statement offsets. These source checks retain the dynamic bound registers
and verify the second helper's reset from caller pointer bases. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSetupSource
open VeriTile.Triton RealMemory RealSetup
set_option maxRecDepth 16000
set_option maxHeartbeats 1800000
set_option linter.unusedSimpArgs false

def beforeRange? (index : RegName) : List Stmt → Option (List Stmt)
  | [] => none
  | st :: rest => match st with
    | .forRangeDyn name .. => if name = index then some [] else
        (beforeRange? index rest).map (st :: ·)
    | _ => (beforeRange? index rest).map (st :: ·)

def afterRange? (index : RegName) : List Stmt → Option (List Stmt)
  | [] => none
  | .forRangeDyn name _ _ _ _ :: rest => if name = index then some rest else afterRange? index rest
  | _ :: rest => afterRange? index rest

def range? (index : RegName) : List Stmt → Nat → Option Stmt
  | [], _ => none
  | st :: rest, occurrence => match st with
    | .forRangeDyn name .. => if name = index then match occurrence with
        | 0 => some st
        | n+1 => range? index rest n
      else range? index rest occurrence
    | _ => range? index rest occurrence

def statement (BM BN BD N HA HD : Nat) (second diagonal : Bool) : Stmt :=
  .forRangeDyn "start_n" (.ref .nat [] (lowerName second)) (.ref .nat [] (upperName second)) (.constNat BN)
    (RealLoop.sourceBody BM BN BD N HA HD second diagonal)

section
variable (Q K V QS KS Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage : Nat) (hs : stage = 2 ∨ stage = 3)

theorem second_setup_source (candidate : Bool) :
    (afterRange? "start_n" ((if candidate then TypedSource.reciprocal else TypedSource.original)
      Q K V QS KS Out sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage hs).eraseDType.toAlgKernel.body).bind (beforeRange? "start_n") =
      some (RealSetup.code BM BN BD HD Bool.true) := by
  cases candidate
  all_goals
    simp only [TypedSource.original, TypedSource.reciprocal, Bool.false_eq_true, Bool.true_eq,
      ↓reduceIte, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
    simp [afterRange?, beforeRange?, RealSetup.code, lowerName, upperName, keyBase, scaleBase, valueBase,
      keyName, scaleName, valueName, lowerExpr, upperExpr]

theorem range_source (candidate second : Bool) :
    range? "start_n" ((if candidate then TypedSource.reciprocal else TypedSource.original)
      Q K V QS KS Out sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage hs).eraseDType.toAlgKernel.body (if second then 1 else 0) =
      some (statement BM BN BD N HA HD second (decide (stage=2))) := by
  cases candidate <;> cases second
  all_goals
    simp only [TypedSource.original, TypedSource.reciprocal, Bool.false_eq_true, Bool.true_eq,
      ↓reduceIte, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
    simp [range?, statement, lowerName, upperName, RealLoop.sourceBody, RealBody.first, RealBody.second,
      RealBody.loadPrefix, RealBody.keyMask, RealBody.valueMask, RealBody.score, RealBody.scoreExpr,
      RealBody.diagonal, RealBody.center, RealBody.centerExpr, RealBody.suffix,
      RealBody.probability, RealBody.rescale, RealBody.loadValue, RealBody.accumulate, RealBody.advance,
      RealExpressions.maskExpr, RealExpressions.penaltyExpr]
end

open RealLoads RealLoop in
open AttentionFwdTriton3.RealLoopFormula (Carry) in
theorem statement_executes {BM BN BD : Nat} (s : BlockState) (c : Config)
    (lower upper : Nat) (second diagonal : Bool) (seed : Carry BM BD) (hN : 0 < BN)
    (hl : s.regs .nat [] (lowerName second) = some (Tile.scalar lower))
    (hu : s.regs .nat [] (upperName second) = some (Tile.scalar upper))
    (hr : Registers (inputs s c BM BN BD) c seed lower second s) :
    ∃ t, stepStmt (statement BM BN BD c.context c.activeHead c.headDim second diagonal) s = some t ∧
      Registers (inputs s c BM BN BD) c
        (after (inputs s c BM BN BD) c lower second diagonal seed (RealFormula.tripCount lower upper BN))
        (cursor lower BN (RealFormula.tripCount lower upper BN)) second t ∧
      AttentionFwdTriton3.RealScores.Frame (RealIteration.writes second) s t := by
  obtain ⟨t, ht, hregs, hf⟩ := loop_executes s c lower upper second diagonal seed hN hr
  refine ⟨t, ?_, hregs, hf⟩
  simp only [statement, stepForRangeAux.forRangeDyn_unfold, evalOp_ref, hl, hu,
    evalOp_constNat, Option.bind_some, Tile.scalar]
  exact ht

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSetupSource
