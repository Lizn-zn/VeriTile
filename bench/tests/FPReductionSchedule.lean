/- Guarded schedule rewrites, explicit padding and intermediate-domain
boundaries. These integer countermodels are not IEEE or GPU observations. -/
import VeriTile.Triton.Float.ReductionSchedule
import VeriTile.Triton.Float.WelfordConditions
import bench.examples.Welford.Comparison
import VeriTile.Meta.StatementAudit

open VeriTile.Bench.Examples.Welford.Kernels

namespace FPReductionScheduleTests
open VeriTile Triton FP.Structural FP.Guarded FP.ScalarArithmetic FP.ScalarReduction
open FP.ReductionSchedule
open _root_.VeriTile.Triton.FP.Equational (ReductionTree ReductionPlan)
open VeriTile.Bench.Examples

deriving instance DecidableEq for ReductionTree

private def M : Algebra Int where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b
    | .sub => a - b
    | .mul => a * b
    | .div => a / b
    | .max => max a b
    | .pow => a
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a + 1
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun b => b.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 777

private def bounded : Domain Int
  | .finite => fun x => |x| ≤ 7
  | .nonzero => fun x => x ≠ 0
  | .positive => fun x => 0 < x

private def row4 (i : Fin 4) : Int := if i.val < 2 then -2 else 4
private def source : ReductionPlan 4 where
  tree := .add (.add (.add (.input 1) (.input 0)) (.input 2)) (.input 3)
  padding := 0
  valid := by decide

private def generatedOperand : ReductionTree 4 := canonical [2, 3]

private def target : ReductionPlan 4 where
  tree := .add (.add (.add (.input 0) (.input 1)) (.input 2)) (.input 3)
  padding := 0
  valid := by decide

-- All original partial sums are finite, yet normalization needs the newly
-- formed sum 4+4 as an atomic associativity operand. Its guard must remain.
theorem source_is_finite : FiniteTree M bounded row4 (zero M) source.tree := by
  norm_num [FiniteTree, source, bounded, row4, value, zero, add, M]
theorem transformed_operand_is_listed : generatedOperand ∈ (normalize source.tree).operands := by decide
theorem transformed_operand_is_not_finite :
    ¬ bounded .finite (value M row4 (zero M) generatedOperand) := by
  change ¬ (|(8 : Int)| ≤ 7)
  decide

theorem finite_source_is_insufficient : ¬ (normalize source.tree).Domain M bounded row4 := by
  intro h
  exact transformed_operand_is_not_finite (h generatedOperand transformed_operand_is_listed)

theorem syntactic_conditions_keep_rewrite_operands :
    ¬ (FP.WelfordConditions.schedule M row4 source target).Holds bounded := by
  rw [FP.WelfordConditions.schedule_holds]
  exact fun h => finite_source_is_insufficient h.left

theorem finite_endpoints_do_not_check_the_path :
    FiniteTree M bounded row4 (zero M) source.tree ∧
    FiniteTree M bounded row4 (zero M) target.tree ∧
    ¬ ScheduleDomain M bounded row4 source target := by
  refine ⟨source_is_finite, ?_, fun h => finite_source_is_insufficient h.left⟩
  norm_num [FiniteTree, target, bounded, row4, value, zero, add, M]

-- A duplicate input cannot become a valid schedule merely by sorting.
example : ¬ (ReductionTree.add (.input (0 : Fin 2)) (.input 0)).leaves.Perm
    ((List.finRange 2).map some ++ List.replicate 0 none) := by decide

private def singleton : ReductionPlan 1 where
  tree := .input 0
  padding := 0
  valid := by decide

private def paddedSingleton : ReductionPlan 1 where
  tree := .add .zero (.add (.input 0) .zero)
  padding := 2
  valid := by decide

private def offsetAddition : Algebra Int :=
  { M with binary := fun p d op a b => match op with
      | .add => a + b + 1
      | _ => M.binary p d op a b }

-- Padding can disappear only through the admitted zero law. Structural
-- evaluation in an arbitrary algebra cannot assert equality of these plans.
example : value offsetAddition (fun _ : Fin 1 => 3) (zero offsetAddition) singleton.tree ≠
    value offsetAddition (fun _ : Fin 1 => 3) (zero offsetAddition) paddedSingleton.tree := by decide

-- The path is data, generated without floating operations or an oracle.
-- No rewrite path is required from the caller of the schedule theorem.
example (a b : ReductionPlan n) :
    (lanes a.tree).insertionSort (· ≤ ·) = (lanes b.tree).insertionSort (· ≤ ·) :=
  (sorted_lanes a).trans (sorted_lanes b).symm

-- Explicit fp64 reductions remain opaque. Only the fp32 schedule is expanded;
-- resolving default precision preserves both original output casts.
example : (FP.ScalarReduction.algebra M FP.Equational.seededSchedules).reduceSum (some .fp64)
    (shape := [2]) ⟨0, by decide⟩ Bool.false (fun _ => 9) PUnit.unit = 777 := rfl

example (plans : FP.Equational.Schedules) :
    (WelfordFPComparison.engine M plans).cast none .real .bf16 9 = 10 := rfl

open Lean Elab Command in
run_cmd do
  if (← getEnv).contains `VeriTile.Bench.Examples.WelfordCorrect.twopass_welford_correct then
    throwError "FP comparison imported its real correctness counterpart"

#axiomsClean FP.ReductionSchedule.Rewrite.sound
#axiomsClean FP.ReductionSchedule.plans_value
#axiomsClean FP.WelfordSchedule.statistics_schedule
#axiomsClean FP.WelfordSchedule.converted_batch_count
#axiomsClean FP.WelfordSchedule.state_batch_statistics
#axiomsClean WelfordFPComparison.original_values
#axiomsClean WelfordFPComparison.original_runs
#axiomsClean finite_source_is_insufficient

#axiomsClean syntactic_conditions_keep_rewrite_operands

end FPReductionScheduleTests
