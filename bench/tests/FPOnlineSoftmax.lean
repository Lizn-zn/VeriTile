/- Online softmax's scalar invariant, initialization and per-iteration domains.
Logical rational fixtures below are not GPU results or rule admissions. -/
import bench.examples.OnlineSoftmax.Proofs.FP
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.NormNum

open VeriTile.Bench.Examples.OnlineSoftmax.Kernels

namespace FPOnlineSoftmaxTests
open VeriTile Triton FP.Structural
open VeriTile.Bench.Examples

private noncomputable def model : Algebra ℚ where
  literal := fun _ _ r => if r = 0 then 0 else 1
  negInf := -1
  binary := fun _ _ op a b => match op with
    | .add => a + b
    | .sub => a - b
    | .mul => a * b
    | .div => a / b
    | .max => max a b
    | .pow => a
  unary := fun _ _ _ => 1
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun b => b.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

private def domain : FP.Guarded.Domain ℚ
  | .finite => fun a => 0 ≤ a
  | .nonzero => fun a => a ≠ 0
  | .positive => fun a => 0 < a

private def initial : State ℚ where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

theorem exp_sub_holds : FP.SoftmaxShift.LibdeviceExpSub model domain := by
  constructor
  intros
  norm_num [FP.SoftmaxShift.exp, FP.ScalarArithmetic.div, model]

theorem neg_inf_is_not_finite : ¬ domain .finite model.negInf := by norm_num [domain, model]

theorem original_domain_is_satisfiable :
    (OnlineSoftmaxFPComparison.requirements "x" 2).Holds model domain initial := by
  rw [OnlineSoftmaxFPComparison.requirements_holds]
  change FP.OnlineSoftmax.IterationDomain model domain (fun _ => 0) 2
  constructor
  · constructor <;>
      norm_num [domain, FP.OnlineSoftmax.maximum, FP.SoftmaxShift.exp,
        FP.ScalarArithmetic.div, FP.ScalarArithmetic.one, FP.ScalarArithmetic.zero,
        FP.ScalarArithmetic.mul, FP.ScalarArithmetic.sub, model]
  · intro i hpos hn
    obtain rfl : i = 1 := by omega
    constructor <;>
      norm_num [domain, FP.OnlineSoftmax.state, FP.OnlineSoftmax.update,
        FP.OnlineSoftmax.maximum, FP.SoftmaxShift.exp, FP.ScalarArithmetic.div,
        FP.ScalarArithmetic.one, FP.ScalarArithmetic.zero, FP.ScalarArithmetic.add,
        FP.ScalarArithmetic.mul, FP.ScalarArithmetic.sub, model]

theorem comparison_domain_is_satisfiable :
    (OnlineSoftmaxFPContract.requirements "x" 2 FP.Equational.seededSchedules).Holds
      model domain initial := by
  rw [OnlineSoftmaxFPContract.requirements_holds]
  change OnlineSoftmaxFPContract.ComparisonDomain model domain (fun _ => 0) 2 FP.Equational.seededSchedules
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · exact (OnlineSoftmaxFPComparison.requirements_holds _ _ _ _ _).mp original_domain_is_satisfiable
  · change FP.SoftmaxShift.ShiftDomain model domain (fun _ : Fin 2 => 0) 0
      (.add (.add .zero (.input 0)) (.input 1))
    constructor <;> try norm_num [domain, FP.SoftmaxShift.exp, model]
    constructor <;>
      norm_num [domain, FP.SoftmaxShift.exponentials, FP.SoftmaxShift.exp,
        FP.SoftmaxShift.scale, FP.ScalarArithmetic.zero, FP.ScalarArithmetic.one,
        FP.ScalarArithmetic.div, FP.ScalarArithmetic.mul, FP.ScalarArithmetic.add, FP.ScalarArithmetic.sub,
        FP.ScalarReduction.value, FP.ScalarReduction.FiniteTree, model]
  · norm_num [domain, FP.OnlineSoftmax.state, FP.OnlineSoftmax.update, FP.OnlineSoftmax.maximum,
      FP.SoftmaxShift.exp, FP.ScalarArithmetic.zero, FP.ScalarArithmetic.add, FP.ScalarArithmetic.mul, model]
  · change FP.SoftmaxShift.ShiftDomain model domain (fun _ : Fin 2 => 0) 0
      (.add (.input 0) (.add (.input 1) .zero))
    constructor <;> try norm_num [domain, FP.SoftmaxShift.exp, model]
    constructor <;>
      norm_num [domain, FP.SoftmaxShift.exponentials, FP.SoftmaxShift.exp,
        FP.SoftmaxShift.scale, FP.ScalarArithmetic.zero, FP.ScalarArithmetic.one,
        FP.ScalarArithmetic.div, FP.ScalarArithmetic.mul, FP.ScalarArithmetic.add, FP.ScalarArithmetic.sub,
        FP.ScalarReduction.value, FP.ScalarReduction.FiniteTree, model]
  · have hf (t : FP.Equational.ReductionTree 2) :
        domain .finite (FP.ScalarReduction.value model (fun _ => 1) (FP.ScalarArithmetic.zero model) t) := by
      induction t with
      | input => norm_num [FP.ScalarReduction.value, domain]
      | zero => norm_num [FP.ScalarReduction.value, FP.ScalarArithmetic.zero, domain, model]
      | add a b ha hb => exact add_nonneg ha hb
    constructor <;> intro t _ <;> exact hf t
private noncomputable def nonfiniteSeed : Algebra ℚ :=
  { model with unary := fun _ _ a => if a < 0 then -1 else 1 }

-- A finite later state does not prove that the seed factor was in-domain.
-- The reified contract must retain the initial exp(-inf - newMax) check.
theorem final_step_does_not_cover_initialization :
    FP.OnlineSoftmax.StepDomain nonfiniteSeed domain 0
      (FP.OnlineSoftmax.state nonfiniteSeed (fun _ => 0) 1).1
      (FP.OnlineSoftmax.state nonfiniteSeed (fun _ => 0) 1).2 ∧
    ¬ FP.OnlineSoftmax.IterationDomain nonfiniteSeed domain (fun _ => 0) 2 := by
  constructor
  · constructor <;>
      norm_num [nonfiniteSeed, model, domain, FP.OnlineSoftmax.state, FP.OnlineSoftmax.update,
        FP.OnlineSoftmax.maximum, FP.SoftmaxShift.exp, FP.ScalarArithmetic.div,
        FP.ScalarArithmetic.one, FP.ScalarArithmetic.zero, FP.ScalarArithmetic.add,
        FP.ScalarArithmetic.mul, FP.ScalarArithmetic.sub]
  · intro h
    have bad := h.initial.seedFactor
    norm_num [nonfiniteSeed, model, domain, FP.OnlineSoftmax.maximum,
      FP.SoftmaxShift.exp, FP.ScalarArithmetic.sub] at bad

-- Syntax-level reification checks the complete iteration record, not merely
-- the final-state domain used in the preceding counterexample.
theorem reified_domain_rejects_nonfinite_seed :
    ¬ (OnlineSoftmaxFPComparison.requirements "x" 2).Holds nonfiniteSeed domain initial := by
  rw [OnlineSoftmaxFPComparison.requirements_holds]
  exact final_step_does_not_cover_initialization.2

private noncomputable def translatedAddition : Algebra ℚ :=
  { model with binary := fun p d op a b => match op with
      | .add => a + b + 1
      | _ => model.binary p d op a b }

-- The prefix tree retains its seed. Its removal requires the scalar
-- add-zero law and is not an identity of the opaque execution model.
theorem prefix_padding_is_retained :
    FP.OnlineSoftmax.prefixSum translatedAddition (fun _ => 0) 2 ≠
      FP.ScalarReduction.value translatedAddition (fun _ : Fin 2 => 1)
        (FP.ScalarArithmetic.zero translatedAddition) (.add (.input 0) (.input 1)) := by
  norm_num [FP.OnlineSoftmax.prefixSum, FP.ScalarReduction.value, FP.SoftmaxShift.exp,
    FP.ScalarArithmetic.zero, FP.ScalarArithmetic.add, translatedAddition, model]

-- The exact source recurrence is connected at symbolic size and prefix.
example {α : Type} (M : Algebra α) (xs : Nat → α) (N i : Nat) (hi : i ≤ N) :
    OnlineSoftmaxFPExecution.recurrence (OnlineSoftmaxFPComparison.engine M)
      (fun k : Fin N => xs k.val) i = FP.OnlineSoftmax.state M xs i :=
  OnlineSoftmaxFPComparison.recurrence_state M xs N i hi

-- The original source has no output port or store. Its y argument cannot
-- become an observable output merely through the numerical invariant.
example (x y z : RegionName) (N : Nat) :
    OnlineSoftmax.Kernels.onlineNormalizerKernel x y N =
      OnlineSoftmax.Kernels.onlineNormalizerKernel x z N := rfl

example {α : Type} [Inhabited α] (M : Algebra α) (N : Nat) (xs : Fin N → α) (s : State α)
    (hx : ∀ i : Fin N, (s.mem "x" (s.pids 0 * N + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec (OnlineSoftmaxFPComparison.engine M)
        (OnlineSoftmax.Kernels.onlineNormalizerKernel "x" "x" N) s = some t ∧
      t.mem = s.mem := by
  obtain ⟨t, ht, _, _, hm, _⟩ :=
    OnlineSoftmaxFPExecution.online_run (OnlineSoftmaxFPComparison.engine M) "x" "x" xs s hx
  exact ⟨t, ht, hm⟩

-- The batch side retains its real-typed row store, including an aliased
-- destination, rather than borrowing the bf16 store from SoftmaxStable.
example {α : Type} [Inhabited α] (M : Algebra α) (N : Nat) (hN : 0 < N)
    (xs : Fin N → α) (s : State α)
    (hx : ∀ i : Fin N, (s.mem "x" (s.pids 0 * N + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (OnlineSoftmax.Kernels.stableSoftmaxKernel "x" "x" N) s = some t ∧
      (∀ i : Fin N, t.mem "x" (s.pids 0 * N + i.val) = .mk .real (OnlineSoftmaxFPBatch.outputValue M xs i)) ∧
      (∀ (r : RegionName) o, (r ≠ "x" ∨ ∀ i : Fin N, o ≠ s.pids 0 * N + i.val) → t.mem r o = s.mem r o) :=
  OnlineSoftmaxFPBatch.batch_run M "x" "x" N hN xs s hx

#axiomsClean FP.OnlineSoftmax.initial_recovery
#axiomsClean FP.OnlineSoftmax.step_recovery
#axiomsClean FP.OnlineSoftmax.state_recovery
#axiomsClean FP.OnlineSoftmax.normalized_prefix
#axiomsClean FP.OnlineSoftmax.prefixSum_tree
#axiomsClean FP.OnlineSoftmaxConditions.iterations_holds
#axiomsClean FP.OnlineSoftmaxConditions.map_iterations
#axiomsClean OnlineSoftmaxFPComparison.original_recovery_run
#axiomsClean OnlineSoftmaxFPBatch.batch_run
#axiomsClean OnlineSoftmaxFPContract.original_normalization_runs
#axiomsClean exp_sub_holds
#axiomsClean neg_inf_is_not_finite
#axiomsClean original_domain_is_satisfiable
#axiomsClean comparison_domain_is_satisfiable
#axiomsClean final_step_does_not_cover_initialization
#axiomsClean reified_domain_rejects_nonfinite_seed
#axiomsClean prefix_padding_is_retained

end FPOnlineSoftmaxTests
