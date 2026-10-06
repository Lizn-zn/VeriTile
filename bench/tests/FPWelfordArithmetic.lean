/- Arithmetic coverage and count-conversion boundary for Welford. The rational
countermodel is not an IEEE execution, GPU experiment or newly admitted law.
It checks the eleven guarded arithmetic equations used by the scalar library;
it makes no claim to model the unrelated transcendental admission families. -/
import VeriTile.Triton.Float.Welford
import VeriTile.Triton.Float.WelfordReduction
import VeriTile.Triton.Float.WelfordAppend
import VeriTile.Triton.Float.WelfordInit
import VeriTile.Triton.Float.WelfordInduction
import VeriTile.Triton.Float.WelfordSchedule
import VeriTile.Triton.Float.ScalarReduction
import bench.examples.Welford.Proofs.FP
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

open VeriTile.Bench.Examples.Welford.Kernels

namespace FPWelfordArithmeticTests
open VeriTile Triton FP.Structural FP.Guarded
open FP.ScalarArithmetic
open VeriTile.Bench.Examples

private def domain : Domain ℚ
  | .finite => fun _ => True
  | .nonzero => fun a => a ≠ 0
  | .positive => fun a => 0 < a

private noncomputable def model (count : Nat → ℚ) : Algebra ℚ where
  literal := fun _ _ r => if r = 0 then 0 else 1
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b
    | .sub => a - b
    | .mul => a * b
    | .div => a / b
    | .max => max a b
    | .pow => a
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a
  fromNat := fun _ => count
  fromInt := fun _ n => n
  fp32Bits := fun b => b.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ {shape} ax keepDims xs i =>
    ∑ k : Fin (TileShape.axisDim shape ax), FP.ScalarReduction.inputs shape ax keepDims xs i k

/-- Every selected arithmetic atom holds, independently of how integer
conversion is interpreted. Nonzero guards on inverse cancellation are kept. -/
theorem all_arithmetic_equations (count : Nat → ℚ) (atom : Atom) (a b c : ℚ)
    (h : Inputs domain a b c atom) :
    leftValue (model count) a b c atom = rightValue (model count) a b c atom := by
  cases atom <;>
    simp_all [Inputs, domain, leftValue, rightValue, add, sub, FP.ScalarArithmetic.mul, div, zero, FP.ScalarArithmetic.one,
      model, div_eq_mul_inv]
  all_goals ring

private def doubledCount (n : Nat) : ℚ := 2 * n
private noncomputable def M := model doubledCount
private def row : Fin 1 → ℚ := fun _ => 2

-- The initial conversion is correct and every positive count stays nonzero;
-- these weaker properties still do not supply the missing successor relation.
theorem initial_count : M.fromNat none 0 = zero M := by norm_num [M, model, doubledCount, zero]
theorem positive_count (n : Nat) (hn : 0 < n) : M.fromNat none n ≠ 0 := by
  simp only [M, model, doubledCount, ne_eq, mul_eq_zero, OfNat.ofNat_ne_zero, false_or]
  exact_mod_cast (Nat.ne_of_gt hn)

theorem successor_count_is_not_forced :
    M.fromNat none 1 ≠ M.binary none .real .add (M.fromNat none 0) (M.literal none .real 1) := by
  norm_num [M, model, doubledCount]

theorem batch_mean : WelfordFPExecution.twopassMean M row = 1 := by
  norm_num [WelfordFPExecution.twopassMean, WelfordFPExecution.sumValue,
    M, model, row, doubledCount, FP.ScalarReduction.inputs, TileShape.axisDim,
    TileShape.insertAxisIndex]

theorem online_mean : (WelfordFPExecution.recurrence M row 1).1 = 2 := by
  norm_num [WelfordFPExecution.recurrence, WelfordFPExecution.update, M, model, row, doubledCount]

/-- Welford's two actual execution expressions disagree already on a single
input, although all eleven arithmetic equations hold and denominators are
nonzero. Thus arbitrary integer conversion cannot be hidden by a ring proof. -/
theorem count_conversion_gap :
    (∀ atom a b c, Inputs domain a b c atom →
      leftValue M a b c atom = rightValue M a b c atom) ∧
    WelfordFPExecution.twopassMean M row ≠ (WelfordFPExecution.recurrence M row 1).1 := by
  refine ⟨all_arithmetic_equations doubledCount, ?_⟩
  rw [batch_mean, online_mean]
  norm_num

-- The arithmetic update cannot be applied with a zero floating denominator.
example : ¬ FP.Welford.MeanStepDomain M domain 2 0 (-1) := by
  intro h
  have hn := h.countNonzero
  norm_num [domain, FP.Welford.nextCount, add, FP.ScalarArithmetic.one, M, model] at hn

-- Choosing fp32 for implicit operations cannot override an explicit fp64
-- computation, or erase the original bf16 output conversion.
example {α : Type} [Inhabited α] (A : Algebra α) (s : State α) :
    evalComputeOp (A.withDefaultPrecision .fp32)
      (.alg .fp64 (.add .real .nil (.const 1) (.const 2))) s =
      some (fun _ => A.binary (some .fp64) .real .add
        (A.literal (some .fp64) .real 1) (A.literal (some .fp64) .real 2)) := by
  simp [evalComputeOp, evalOp_unfold, numeric, Algebra.withDefaultPrecision,
    resolvePrecision]
  rfl

example {α : Type} [Inhabited α] (A : Algebra α) (s : State α) :
    evalOp (A.withDefaultPrecision .fp32) none (.castFloat .real .bf16 (.const 3)) s =
      some (fun _ => A.cast (some .fp32) .real .bf16 (A.literal (some .fp32) .real 3)) := by
  simp [evalOp_unfold, Algebra.withDefaultPrecision, resolvePrecision, ofFloat, toFloat]

private def boundedDomain : Domain ℚ
  | .finite => fun a => |a| ≤ 7
  | .nonzero => fun a => a ≠ 0
  | .positive => fun a => 0 < a

-- A valid mean step can still produce a residual outside the permitted
-- domain. The variance step must not silently reuse only the mean guards.
example : FP.Welford.MeanStepDomain M boundedDomain (-7) 0 (-2) := by
  constructor <;>
    norm_num [boundedDomain, M, model, FP.Welford.difference, FP.Welford.nextCount,
      FP.Welford.correction, FP.Welford.nextMean, add, sub, FP.ScalarArithmetic.mul,
      div, FP.ScalarArithmetic.one]

example : ¬ FP.Welford.VarianceStepDomain M boundedDomain (-7) 0 (-2) := by
  intro h
  have hres := h.residual
  norm_num [boundedDomain, M, model, FP.Welford.residual, FP.Welford.nextMean,
    FP.Welford.correction, FP.Welford.difference, FP.Welford.nextCount,
    add, sub, div, FP.ScalarArithmetic.one] at hres

private def tripleTree : FP.Equational.ReductionTree 3 :=
  .add (.input 0) (.add (.input 1) (.input 2))
private def tripleRow (i : Fin 3) : ℚ := if i.val = 0 then -2 else 2

-- Both component trees, all paired leaves and the final paired sum are
-- finite. Nevertheless, the paired right subtree is 8 and fails its guard.
-- Reduction linearity must retain the transformed intermediate domains.
example :
    FP.ScalarReduction.FiniteTree M boundedDomain tripleRow (zero M) tripleTree ∧
    (∀ i, boundedDomain .finite (add M (tripleRow i) (tripleRow i))) ∧
    boundedDomain .finite
      (FP.ScalarReduction.value M (fun i => add M (tripleRow i) (tripleRow i)) (zero M) tripleTree) ∧
    ¬ FP.ScalarReduction.FiniteTree M boundedDomain
      (fun i => add M (tripleRow i) (tripleRow i)) (zero M) tripleTree := by
  have htwo : (2 : Fin 3) ≠ 0 := by decide
  refine ⟨?_, ?_, ?_, ?_⟩
  · norm_num [FP.ScalarReduction.FiniteTree, FP.ScalarReduction.value,
      tripleTree, tripleRow, boundedDomain, add, zero, M, model]
  · intro i
    by_cases h : i.val = 0 <;> norm_num [tripleRow, h, boundedDomain, add, M, model]
  · norm_num [FP.ScalarReduction.value, tripleTree, tripleRow, boundedDomain, add, zero, M, model, htwo]
  · norm_num [FP.ScalarReduction.FiniteTree, FP.ScalarReduction.value,
      tripleTree, tripleRow, boundedDomain, add, zero, M, model, htwo]

private def paddedSingleton : FP.Equational.ReductionTree 1 :=
  .add .zero (.add (.input 0) .zero)

-- Padding is not an extra input, and the explicit sum of ones does not
-- inherit the arbitrary fromNat interpretation from the original kernel.
example : FP.WelfordReduction.count M paddedSingleton = 1 ∧
    FP.WelfordReduction.mean M row paddedSingleton = 2 ∧
    FP.WelfordReduction.count M paddedSingleton ≠ M.fromNat (some .fp32) 1 := by
  norm_num [FP.WelfordReduction.count, FP.WelfordReduction.mean, FP.ScalarReduction.value,
    paddedSingleton, row, zero, FP.ScalarArithmetic.one, add, div, M, model, doubledCount]

-- Even with literal zero correctly represented, the tree-based mean cannot
-- be used on an empty tree without its explicit nonzero-count guard.
example : ¬ FP.WelfordReduction.MeanDomain M domain (fun _ : Fin 0 => (0 : ℚ)) .zero := by
  intro h
  have hn := h.countNonzero
  norm_num [domain, FP.WelfordReduction.count, FP.ScalarReduction.value, zero, M, model] at hn

private def offsetCount (n : Nat) : ℚ := n + 1
private noncomputable def offsetModel := model offsetCount

-- A successor law alone still leaves the conversion of zero unspecified.
theorem offset_count_successor (n : Nat) :
    offsetModel.fromNat (some .fp32) (n + 1) =
      add offsetModel (offsetModel.fromNat (some .fp32) n) (FP.ScalarArithmetic.one offsetModel) := by
  norm_num [offsetModel, model, offsetCount, add, FP.ScalarArithmetic.one, Nat.cast_add]

/-- All selected scalar equations and the count successor law hold, but
variance disagrees when conversion gives zero a nonzero weight. This checks
the initialization obligation independently of the successor obligation. -/
theorem count_initialization_gap :
    (∀ atom a b c, Inputs domain a b c atom →
      leftValue offsetModel a b c atom = rightValue offsetModel a b c atom) ∧
    offsetModel.fromNat (some .fp32) 0 ≠ zero offsetModel ∧
    WelfordFPExecution.twopassVariance offsetModel row ≠
      WelfordFPExecution.varianceValue offsetModel row := by
  refine ⟨all_arithmetic_equations offsetCount, ?_, ?_⟩
  · norm_num [offsetModel, model, offsetCount, zero]
  · norm_num [WelfordFPExecution.twopassVariance, WelfordFPExecution.twopassMean,
      WelfordFPExecution.sumValue, WelfordFPExecution.varianceValue,
      WelfordFPExecution.recurrence, WelfordFPExecution.update, offsetModel, model, offsetCount,
      row, FP.ScalarReduction.inputs, TileShape.axisDim, TileShape.insertAxisIndex]

-- Initialization uses the squared residual, not the input's square. The
-- needed domains can hold even when squaring the input leaves the domain.
example : FP.WelfordInit.InitDomain M boundedDomain 7 ∧
    ¬ boundedDomain .finite (FP.Welford.square M 7) := by
  constructor
  · constructor <;>
      norm_num [boundedDomain, M, model, zero, FP.ScalarArithmetic.one, sub,
        FP.ScalarArithmetic.mul, FP.Welford.square]
  · norm_num [boundedDomain, M, model, FP.ScalarArithmetic.mul, FP.Welford.square]

private noncomputable def ordinaryCountModel := model (fun n => n)

-- The conversion record is satisfiable, but neither existing countermodel
-- can supply it. Zero and successor are independent primitive obligations.
theorem ordinary_count_conversion (N : Nat) :
    FP.WelfordInduction.CountConversion ordinaryCountModel N := by
  constructor
  · norm_num [ordinaryCountModel, model, zero]
  · intro i _
    norm_num [ordinaryCountModel, model, add, FP.ScalarArithmetic.one, Nat.cast_add]

example : ¬ FP.WelfordInduction.CountConversion M 1 := by
  intro h
  have hs := h.successor 0 (by decide)
  norm_num [M, model, doubledCount, add, FP.ScalarArithmetic.one] at hs

example (N : Nat) : ¬ FP.WelfordInduction.CountConversion offsetModel N := by
  intro h
  have hz := h.zero
  norm_num [offsetModel, model, offsetCount, zero] at hz

private def middleBadCount (n : Nat) : ℚ := if n = 1 then 100 else n
private noncomputable def middleBadModel := model middleBadCount
private def pairValues (i : Nat) : ℚ := if i = 0 then 2 else 4

-- Conversion at zero and at the final row length is correct, and all
-- arithmetic atoms hold, but conversion at an intermediate index changes
-- both statistics. Checking the final count alone cannot justify induction.
theorem intermediate_count_gap :
    (∀ atom a b c, Inputs domain a b c atom →
      leftValue middleBadModel a b c atom = rightValue middleBadModel a b c atom) ∧
    middleBadModel.fromNat (some .fp32) 0 = zero middleBadModel ∧
    middleBadModel.fromNat (some .fp32) 2 = 2 ∧
    (FP.WelfordInduction.state middleBadModel pairValues 2).1 = 204 / 101 ∧
    (FP.WelfordInduction.statistics middleBadModel (FP.WelfordInduction.rowPrefix pairValues 2)
      (FP.WelfordInduction.tree .zero 2)).1 = 3 ∧
    div middleBadModel (FP.WelfordInduction.state middleBadModel pairValues 2).2
      (middleBadModel.fromNat (some .fp32) 2) = 200 / 101 := by
  refine ⟨all_arithmetic_equations middleBadCount, ?_⟩
  norm_num [FP.WelfordInduction.state, FP.WelfordInduction.update, FP.WelfordInduction.statistics,
    FP.WelfordInduction.tree, FP.WelfordInduction.rowPrefix, FP.WelfordAppend.appendTree,
    FP.WelfordAppend.liftTree, FP.WelfordReduction.mean, FP.WelfordReduction.count,
    FP.ScalarReduction.value, FP.Welford.nextMean, FP.Welford.correction,
    FP.Welford.difference, FP.Welford.nextCount, FP.Welford.increment, FP.Welford.residual,
    middleBadModel, model, middleBadCount, pairValues, zero, FP.ScalarArithmetic.one,
    add, sub, FP.ScalarArithmetic.mul, div]

private theorem rational_finite_tree (A : Algebra ℚ) (xs : Fin n → ℚ) (seed : ℚ)
    (t : FP.Equational.ReductionTree n) : FP.ScalarReduction.FiniteTree A domain xs seed t := by
  induction t with
  | input => trivial
  | zero => trivial
  | add a b ih₁ ih₂ => exact ⟨ih₁, ih₂, trivial⟩

-- A complete two-step domain can be discharged in the ordinary rational
-- model; the induction theorem does not hide an impossible premise.
theorem ordinary_iteration_domain :
    FP.WelfordInduction.IterationDomain ordinaryCountModel domain pairValues (.add .zero .zero) 2 := by
  constructor
  · constructor <;> trivial
  · intro i hi hn
    have he : i = 1 := by omega
    subst i
    repeat' first | exact rational_finite_tree _ _ _ _ | constructor | intro
    all_goals norm_num [domain, ordinaryCountModel, model, FP.WelfordReduction.count,
      FP.WelfordInduction.tree, FP.WelfordAppend.appendTree, FP.WelfordAppend.liftTree,
      FP.ScalarReduction.value, FP.Welford.nextCount, zero, FP.ScalarArithmetic.one, add] at *

-- Schedule domains are satisfiable in the same ordinary arithmetic model
-- for any valid trees, including permutations and different padding counts.
theorem ordinary_schedule_domain (xs : Fin N → ℚ) (a b : FP.Equational.ReductionPlan N) :
    FP.WelfordSchedule.StatisticsDomain ordinaryCountModel domain xs a b := by
  constructor <;> constructor <;> intro t _ <;> trivial

-- The first sample may meet all initialization guards even though the next
-- sample leaves the domain. The loop theorem must not inspect only its base.
example : FP.WelfordInit.InitDomain ordinaryCountModel boundedDomain 1 ∧
    ¬ FP.WelfordInduction.IterationDomain ordinaryCountModel boundedDomain
      (fun i => if i = 0 then 1 else 100) .zero 2 := by
  constructor
  · constructor <;>
      norm_num [ordinaryCountModel, model, boundedDomain, zero, FP.ScalarArithmetic.one,
        sub, FP.ScalarArithmetic.mul, FP.Welford.square]
  · intro h
    have hx := (h.steps 1 (by decide) (by decide)).step.input
    norm_num [boundedDomain] at hx

-- A nonempty padded seed survives every append and still forms a valid
-- reduction plan. No padding leaf is dropped by the recurrence induction.
private def emptyPlan : FP.Equational.ReductionPlan 0 where
  tree := .add .zero .zero
  padding := 2
  valid := by simp [FP.Equational.ReductionTree.leaves]

example : (FP.WelfordInduction.plan emptyPlan 3).padding = 2 := rfl
example (N : Nat) :
    (FP.WelfordInduction.tree emptyPlan.tree N).leaves.Perm
      ((List.finRange N).map some ++ List.replicate (FP.WelfordInduction.plan emptyPlan N).padding none) := by
  rw [← FP.WelfordInduction.plan_tree emptyPlan N]
  exact (FP.WelfordInduction.plan emptyPlan N).valid

private def pairState : State ℚ where
  mem := fun _ offset => .mk .real (pairValues offset)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private theorem pair_row (stride : Nat) :
    WelfordFPContract.rowValues pairState "x" stride = pairValues := by
  funext i
  simp [WelfordFPContract.rowValues, pairState]

-- The public syntactic contract is satisfiable for every batch schedule,
-- including a different padding count from the prefix plan. Its proof uses
-- the complete semantic domain records through the compiler's iff theorem.
theorem reified_contract_satisfiable (plans : FP.Equational.Schedules) :
    (WelfordFPContract.requirements "x" 2 2 emptyPlan plans).Holds
      ordinaryCountModel domain pairState := by
  rw [WelfordFPContract.requirements_holds, pair_row]
  exact ⟨ordinary_iteration_domain, ordinary_schedule_domain _ _ _⟩

private def badRowState : State ℚ :=
  { pairState with mem := fun _ offset => .mk .real (if offset = 0 then 1 else 100) }

-- The first sample is admissible, but an intermediate loop operand is not.
-- The reified public contract must reject this even with arbitrary schedules.
theorem reified_contract_checks_later_samples (plans : FP.Equational.Schedules) :
    ¬ (WelfordFPContract.requirements "x" 2 2 emptyPlan plans).Holds
      ordinaryCountModel boundedDomain badRowState := by
  intro h
  have hi := ((WelfordFPContract.requirements_holds
    ordinaryCountModel boundedDomain badRowState "x" 2 2 emptyPlan plans).mp h).1
  have hx := (hi.steps 1 (by decide) (by decide)).step.input
  norm_num [boundedDomain, WelfordFPContract.rowValues, badRowState, pairState] at hx

#axiomsClean WelfordFPContract.requirements_holds
#axiomsClean WelfordFPContract.original_runs_under_count
#axiomsClean reified_contract_satisfiable
#axiomsClean reified_contract_checks_later_samples

#axiomsClean FP.Welford.mean_step
#axiomsClean WelfordFPExecution.fp32_mean_step
#axiomsClean FP.ScalarArithmetic.add_right_cancel
#axiomsClean FP.ScalarArithmetic.sub_add_sub_cancel
#axiomsClean FP.ScalarArithmetic.sub_self
#axiomsClean FP.ScalarArithmetic.square_eq_of_add_eq_zero
#axiomsClean FP.ScalarReduction.value_add
#axiomsClean FP.ScalarReduction.constant_value
#axiomsClean FP.ScalarReduction.value_empty
#axiomsClean FP.ScalarReduction.value_zero
#axiomsClean FP.ScalarReduction.deviations_add_center
#axiomsClean FP.WelfordReduction.centered_sum_zero
#axiomsClean FP.WelfordReduction.cross_sum_zero
#axiomsClean FP.WelfordReduction.centered_square_shift
#axiomsClean FP.WelfordAppend.mean_append
#axiomsClean FP.WelfordAppend.shift_square
#axiomsClean FP.WelfordAppend.variance_append
#axiomsClean FP.WelfordInit.initial_statistics
#axiomsClean FP.WelfordInit.singleton_variance
#axiomsClean FP.Welford.residual_step
#axiomsClean FP.Welford.variance_step
#axiomsClean FP.Welford.square_shift
#axiomsClean FP.Welford.square_shift_tree
#axiomsClean WelfordFPExecution.fp32_variance_step
#axiomsClean all_arithmetic_equations
#axiomsClean count_conversion_gap
#axiomsClean offset_count_successor
#axiomsClean count_initialization_gap
#axiomsClean FP.WelfordInduction.converted_count
#axiomsClean FP.WelfordInduction.state_statistics
#axiomsClean FP.WelfordInduction.normalized_statistics
#axiomsClean WelfordFPExecution.fp32_recurrence_prefix
#axiomsClean WelfordFPExecution.fp32_recurrence_statistics
#axiomsClean WelfordFPExecution.fp32_online_statistics_run
#axiomsClean intermediate_count_gap
#axiomsClean ordinary_iteration_domain
#axiomsClean ordinary_schedule_domain

end FPWelfordArithmeticTests
