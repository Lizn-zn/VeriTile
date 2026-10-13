import bench.tritonbench_g.softmax_flaggems.Kernels
import VeriTile.Triton

/-! Concrete full-source real execution witnesses. These exercise the omitted
multi-tile branches and their reverse writeback, but do not replace the pending
symbolic full IO theorem. Backward padding is deliberately nonzero. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealExecution
open VeriTile Triton FullSource
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

private theorem option_add (a b : ℝ) :
    @HAdd.hAdd (WithBot ℝ) (WithBot ℝ) (WithBot ℝ)
      (@instHAdd _ (inferInstance : AddCommMonoid (WithBot ℝ)).toAdd) (some a) (some b) =
      some (a+b) := rfl
private theorem if_bot_lt (a : ℝ) (x y : WithBot ℝ) :
    (if @LT.lt (WithBot ℝ) _ none (some a) then x else y) = x :=
  if_pos (WithBot.bot_lt_coe a)
private theorem if_not_lt_bot (a : ℝ) (x y : WithBot ℝ) :
    (if @LT.lt (WithBot ℝ) _ (some a) none then x else y) = y :=
  if_neg (not_lt_of_ge bot_le)
private theorem if_lt_self (a x y : WithBot ℝ) :
    (if a < a then x else y) = y := if_neg (lt_irrefl a)
private theorem option_max_self (a : ℝ) :
    @max (WithBot ℝ) _ (some a) (some a) = some a := max_self _

def initial : BlockState where
  mem := fun r _ ↦ .real (if r = "Y" then 7 else 0)
  regs := fun _ _ _ ↦ none
  pids := fun _ ↦ 0

noncomputable def innerSource : ComputeKernel :=
  softmax_kernel_inner .fp32 .fp32 "Y" "X" 1 3 2 Bool.false

noncomputable def innerObservation : Option (List ℝ) :=
  (exec innerSource.toAlgKernel initial).map fun s ↦
    [s.readMem "Y" 0, s.readMem "Y" 1, s.readMem "Y" 2, s.readMem "Y" 3]

theorem inner_uniform : innerObservation = some [1/3, 1/3, 1/3, 7] := by
  simp +decide [innerObservation, innerSource, softmax_kernel_inner,
    ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    exec, stepStmts, stepStmt, evalOp.eq_def, initial,
    BlockState.setReg, BlockState.readMem, BlockState.writeMem, MemCell.readAs, MemCell.real, MemCell.of,
    ComputeDType.eraseDType, castTile,
    FloatDType.cast, FloatDType.toWithBot, FloatDType.ofReal,
    TileShape.allIndices, List.finRange_succ, Finset.univ_fin2,
    Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    TileShape.dropInsertedIndex, TileShape.replaceAxisIndex,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt, ComparableDType.gt, WithBot.realExp]
  all_goals dsimp +instances only [TileShape.axisDim, TileCarrier]
  all_goals norm_num [Fin.sum_univ_two, Option.map₂, Finset.univ_fin2, Finset.sup'_insert, Finset.sup'_singleton,
    WithBot.realExp, WithBot.realSub, WithBot.realMul, WithBot.realAdd,
    WithBot.unbotD, FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]
  all_goals repeat' first
    | rw [option_add]
    | rw [if_bot_lt]
    | rw [if_not_lt_bot]
    | rw [option_max_self]
    | rw [if_lt_self]
  all_goals norm_num [option_add, if_bot_lt, if_not_lt_bot, option_max_self, if_lt_self, WithBot.lt_def, WithBot.add, max_def, WithBot.recBotCoe, Option.map₂, Option.bind, Option.map]
  all_goals dsimp
  all_goals norm_num [WithBot.realExp, WithBot.realAdd, WithBot.realSub, WithBot.realMul]
  all_goals rw [option_add]
  all_goals norm_num



def backwardInitial : BlockState where
  mem := fun r _ ↦ .real (if r = "Y" then 7 else 1)
  regs := fun _ _ _ ↦ none
  pids := fun _ ↦ 0
  undef := fun _ _ ↦ 2

noncomputable def backwardSource : ComputeKernel :=
  softmax_backward_kernel_inner .fp32 "P" "G" "Y" 1 3 1 2 Bool.false

noncomputable def backwardObservation : Option (List ℝ) :=
  (exec backwardSource.toAlgKernel backwardInitial).map fun s ↦
    [s.readMem "Y" 0, s.readMem "Y" 1, s.readMem "Y" 2, s.readMem "Y" 3]

/-- The masked tail contributes 2*2 to scale: the actual gradient is -6,
whereas silently zeroing that padding would give -2. -/
theorem backward_nonzero_padding : backwardObservation = some [-6, -6, -6, 7] := by
  simp +decide [backwardObservation, backwardSource, softmax_backward_kernel_inner,
    ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    exec, stepStmts, stepStmt, evalOp.eq_def, backwardInitial,
    BlockState.setReg, BlockState.readMem, BlockState.writeMem, MemCell.readAs, MemCell.real, MemCell.of,
    ComputeDType.eraseDType, castTile,
    FloatDType.cast, FloatDType.toWithBot, FloatDType.ofReal,
    TileShape.allIndices, List.finRange_succ, Finset.univ_fin2, Finset.univ_unique,
    Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    TileShape.dropInsertedIndex, TileShape.replaceAxisIndex,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt, ComparableDType.gt, WithBot.realExp]
  all_goals dsimp +instances only [TileShape.axisDim, TileCarrier]
  all_goals norm_num [Fin.sum_univ_two, Option.map₂, Finset.univ_fin2, Finset.univ_unique, Finset.sup'_insert, Finset.sup'_singleton,
    WithBot.realExp, WithBot.realSub, WithBot.realMul, WithBot.realAdd,
    WithBot.unbotD, FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]
  all_goals repeat' first
    | rw [option_add]
    | rw [if_bot_lt]
    | rw [if_not_lt_bot]
    | rw [option_max_self]
    | rw [if_lt_self]
  all_goals norm_num [option_add, if_bot_lt, if_not_lt_bot, option_max_self, if_lt_self, WithBot.lt_def, WithBot.add, max_def, WithBot.recBotCoe, Option.map₂, Option.bind, Option.map]


noncomputable def nonInnerSource : ComputeKernel :=
  softmax_kernel_non_inner .fp32 .fp32 "Y" "X" 1 3 1 2 1 Bool.false

noncomputable def nonInnerObservation : Option (List ℝ) :=
  (exec nonInnerSource.toAlgKernel initial).map fun s ↦
    [s.readMem "Y" 0, s.readMem "Y" 1, s.readMem "Y" 2, s.readMem "Y" 3]

theorem non_inner_uniform : nonInnerObservation = some [1/3, 1/3, 1/3, 7] := by
  simp +decide [nonInnerObservation, nonInnerSource, softmax_kernel_non_inner,
    ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    exec, stepStmts, stepStmt, evalOp.eq_def, initial,
    BlockState.setReg, BlockState.readMem, BlockState.writeMem, MemCell.readAs, MemCell.real, MemCell.of,
    ComputeDType.eraseDType, castTile,
    FloatDType.cast, FloatDType.toWithBot, FloatDType.ofReal,
    TileShape.allIndices, List.finRange_succ, Finset.univ_fin2, Finset.univ_unique,
    Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    TileShape.dropInsertedIndex, TileShape.replaceAxisIndex,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt, ComparableDType.gt, WithBot.realExp]
  all_goals dsimp +instances only [TileShape.axisDim, TileCarrier]
  all_goals norm_num [Fin.sum_univ_two, Option.map₂, Finset.univ_fin2, Finset.univ_unique, Finset.sup'_insert, Finset.sup'_singleton,
    WithBot.realExp, WithBot.realSub, WithBot.realMul, WithBot.realAdd,
    WithBot.unbotD, FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]
  all_goals repeat' first
    | rw [option_add]
    | rw [if_bot_lt]
    | rw [if_not_lt_bot]
    | rw [option_max_self]
    | rw [if_lt_self]
  all_goals norm_num [option_add, if_bot_lt, if_not_lt_bot, option_max_self, if_lt_self, WithBot.lt_def, WithBot.add, max_def, WithBot.recBotCoe, Option.map₂, Option.bind, Option.map]
  all_goals dsimp
  all_goals norm_num [WithBot.realExp, WithBot.realAdd, WithBot.realSub, WithBot.realMul]
  all_goals rw [option_add]
  all_goals norm_num



noncomputable def backwardNonInnerSource : ComputeKernel :=
  softmax_backward_kernel_non_inner .fp32 "P" "G" "Y" 1 3 1 2 1 Bool.false

noncomputable def backwardNonInnerObservation : Option (List ℝ) :=
  (exec backwardNonInnerSource.toAlgKernel backwardInitial).map fun s ↦
    [s.readMem "Y" 0, s.readMem "Y" 1, s.readMem "Y" 2, s.readMem "Y" 3]

/-- The masked tail contributes 2*2 to scale: the actual gradient is -6,
whereas silently zeroing that padding would give -2. -/
theorem backward_non_inner_nonzero_padding : backwardNonInnerObservation = some [-6, -6, -6, 7] := by
  simp +decide [backwardNonInnerObservation, backwardNonInnerSource, softmax_backward_kernel_non_inner,
    ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    exec, stepStmts, stepStmt, evalOp.eq_def, backwardInitial,
    BlockState.setReg, BlockState.readMem, BlockState.writeMem, MemCell.readAs, MemCell.real, MemCell.of,
    ComputeDType.eraseDType, castTile,
    FloatDType.cast, FloatDType.toWithBot, FloatDType.ofReal,
    TileShape.allIndices, List.finRange_succ, Finset.univ_fin2, Finset.univ_unique,
    Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    TileShape.dropInsertedIndex, TileShape.replaceAxisIndex,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt, ComparableDType.gt, WithBot.realExp]
  all_goals dsimp +instances only [TileShape.axisDim, TileCarrier]
  all_goals norm_num [Fin.sum_univ_two, Option.map₂, Finset.univ_fin2, Finset.univ_unique, Finset.sup'_insert, Finset.sup'_singleton,
    WithBot.realExp, WithBot.realSub, WithBot.realMul, WithBot.realAdd,
    WithBot.unbotD, FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]
  all_goals repeat' first
    | rw [option_add]
    | rw [if_bot_lt]
    | rw [if_not_lt_bot]
    | rw [option_max_self]
    | rw [if_lt_self]
  all_goals norm_num [option_add, if_bot_lt, if_not_lt_bot, option_max_self, if_lt_self, WithBot.lt_def, WithBot.add, max_def, WithBot.recBotCoe, Option.map₂, Option.bind, Option.map]



end
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealExecution
