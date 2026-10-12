import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.IntegerWhereCondition
open VeriTile.Triton
noncomputable section

def choose (value : Int) : ComputeKernel := triton {
  result = tl.where($((value : Int)), 7.0, -3.0)
}

def initial : BlockState where
  mem := fun _ _ => .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 0

/-- Integer conditions use nonzero truthiness: negative and even positive
values are true, rather than testing a truncated low bit. -/
theorem evaluates (value : Int) :
    ∃ v, exec (choose value).toAlgKernel initial=some v ∧
      v.regs .real [] "result"=some (Tile.scalar (some (if value≠0 then (7 : ℝ) else -3))) := by
  refine ⟨?_,?_,?_⟩
  · exact initial.setReg "result" .real [] (Tile.scalar (some (if value≠0 then (7 : ℝ) else -3)))
  · by_cases h : value=0
    all_goals simp only [choose,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
      Except.bind,bind,exec,stepStmts,stepStmt,evalOp.eq_def,Tile.select,Tile.cop,Tile.scalar,
      ComparableDType.ne,Tile.bop,NumericDType.sub,Option.bind,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals norm_num [h]
  · simp [BlockState.setReg_same]

def chooseNat (value : Nat) : ComputeKernel := triton {
  result = tl.where($((value : Nat)), 7.0, -3.0)
}

def initialNatNat : BlockState where
  mem := fun _ _ => .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 0

/-- Natural-number conditions likewise test nonzero. -/
theorem evaluatesNat (value : Nat) :
    ∃ v, exec (chooseNat value).toAlgKernel initialNat=some v ∧
      v.regs .real [] "result"=some (Tile.scalar (some (if value≠0 then (7 : ℝ) else -3))) := by
  refine ⟨?_,?_,?_⟩
  · exact initialNat.setReg "result" .real [] (Tile.scalar (some (if value≠0 then (7 : ℝ) else -3)))
  · by_cases h : value=0
    all_goals simp only [chooseNat,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
      Except.bind,bind,exec,stepStmts,stepStmt,evalOp.eq_def,Tile.select,Tile.cop,Tile.scalar,
      ComparableDType.ne,Tile.bop,NumericDType.sub,Option.bind,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals norm_num [h]
  · simp [BlockState.setReg_same]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.IntegerWhereCondition
