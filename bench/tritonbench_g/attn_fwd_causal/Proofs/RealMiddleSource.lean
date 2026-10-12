import bench.tritonbench_g.attn_fwd_causal.Proofs.RealFirstSetup

/-! Exact source connection for the complete middle sequence: local copies,
stage branch, first dynamic range, reset, and second dynamic range. Selectors
use the first_lo assignment and two start_n ranges, without numeric offsets. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMiddleSource
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 2200000
set_option linter.unusedSimpArgs false

def fromAssignment? (name : RegName) : List Stmt → Option (List Stmt)
  | [] => none
  | st :: rest => match st with
    | .assign _ _ target _ => if name = target then some (st :: rest) else fromAssignment? name rest
    | _ => fromAssignment? name rest

def throughRanges? (index : RegName) : List Stmt → Nat → Option (List Stmt)
  | _, 0 => some []
  | [], _+1 => none
  | st :: rest, n+1 => match st with
    | .forRangeDyn name .. =>
      (throughRanges? index rest (if name = index then n else n+1)).map (st :: ·)
    | _ => (throughRanges? index rest (n+1)).map (st :: ·)

section
variable (Q K V QS KS Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage : Nat) (hs : stage = 2 ∨ stage = 3)

theorem source (candidate : Bool) :
    (fromAssignment? "first_lo" ((if candidate then TypedSource.reciprocal else TypedSource.original)
      Q K V QS KS Out sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage hs).eraseDType.toAlgKernel.body).bind
        (fun rest => throughRanges? "start_n" rest 2) =
      some (RealFirstSetup.code BM BN BD HD (decide (stage=2)) ++
        RealTwoLoops.code BM BN BD N HA HD (decide (stage=2))) := by
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
    simp [fromAssignment?, throughRanges?, RealFirstSetup.code, RealFirstSetup.copies,
      RealTwoLoops.code, RealSetupSource.statement, RealLoop.sourceBody,
      RealSetup.code, RealSetup.lowerExpr, RealSetup.upperExpr, RealSetup.lowerName, RealSetup.upperName,
      RealSetup.keyBase, RealSetup.scaleBase, RealSetup.valueBase,
      RealMemory.keyName, RealMemory.scaleName, RealMemory.valueName,
      RealBody.first, RealBody.second, RealBody.loadPrefix, RealBody.keyMask, RealBody.valueMask,
      RealBody.score, RealBody.scoreExpr, RealBody.diagonal, RealBody.center, RealBody.centerExpr,
      RealBody.suffix, RealBody.probability, RealBody.rescale, RealBody.loadValue, RealBody.accumulate,
      RealBody.advance, RealExpressions.maskExpr, RealExpressions.penaltyExpr]
end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMiddleSource
