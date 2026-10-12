import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Semantics.Step
import VeriTile.Meta.StatementAudit

/-! Signed values retain their type through min/max, compiler hints and
broadcast views. Both interpreters must preserve the negative values [-3,-2];
no FP atomic assumptions are involved. The multiple-of-one hint is valid for
both values, and the two entries are contiguous. -/
namespace VeriTile.Tests.SignedMinMaxViews
open VeriTile.Triton

def source : ComputeKernel := triton {
  v = tl.arange(0, $(2)) - $((3 : Int))
  lower = tl.minimum(v, $((0 : Int)))
  bounded = tl.maximum(lower, $((-4 : Int)))
  cap = min(bounded, $((-1 : Int)))
  hinted = tl.multiple_of(cap, $(1))
  dense = tl.max_contiguous(hinted, $(2))
  columns = dense[:, None]
  rows = dense[None, :]
}

set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false

theorem real_negative_views (s : BlockState) :
    (source.toAlgorithm?.toOption.bind (fun k => exec k s)).map
      (fun t => (t.regs .int [2,1] "columns", t.regs .int [1,2] "rows")) =
      some (some ⟨fun i => (i.1.val : Int)-3⟩,
        some ⟨fun i => (i.2.1.val : Int)-3⟩) := by
  simp only [source, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, Except.bind, Except.toOption,
    pure, bind, Option.bind_some, exec]
  simp [stepStmts, stepStmt, evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.uop,
    Tile.scalar, Tile.expandDim, Tile.select, Tile.cop, NumericDType.sub,
    ComparableDType.lt, ComparableDType.gt, Broadcast.leftIndex,
    Broadcast.rightIndex, -Tile.scalar_eta]
  dsimp only [Option.bind]
  simp only [BlockState.setReg]
  simp
  constructor
  · apply Tile.ext
    rintro ⟨i,j,u⟩
    fin_cases i <;> rfl
  · apply Tile.ext
    rintro ⟨i,j,u⟩
    fin_cases j <;> rfl

theorem fp_negative_views {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α) :
    (FP.Structural.exec M source s).map
      (fun t => (t.regs .int [2,1] "columns", t.regs .int [1,2] "rows")) =
      some (some (fun i => (i.1.val : Int)-3),
        some (fun i => (i.2.1.val : Int)-3)) := by
  simp [source, FP.Structural.exec, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalOp_unfold, FP.Structural.State.setReg,
    FP.Structural.bop, FP.Structural.numeric, FP.Structural.numericLt,
    FP.Structural.numericLe, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex, Option.bind]
  constructor
  · funext ⟨i,j,u⟩
    fin_cases i <;> rfl
  · funext ⟨i,j,u⟩
    fin_cases j <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.SignedMinMaxViews
