/- Logsumexp scalar obligations, original output layout and domain boundaries.
The rational models below are logical fixtures, not numerical admissions. -/
import bench.examples.support.StableLogSumExpContract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.NormNum

namespace FPStableLogSumExpTests
open VeriTile Triton FP.Structural
open VeriTile.Bench.Examples StableLogSumExpFPExecution
open scoped VeriTile.Spec

theorem stable_empty_fails {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    FP.Structural.exec M (stableLSEKernel "x" "y" 0) s = none := by
  simp [stableLSEKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, TileShape.axisDim, TileShape.eraseAxis, Region.cast]

-- Scalar output can alias input: the complete row was read before the store.
example {α : Type} [Inhabited α] (M : Algebra α) (B : Nat) (hB : 0 < B)
    (xs : Fin B → α) (s : State α)
    (hx : ∀ i : Fin B, (s.mem "x" (s.pids 0 * B + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (stableLSEKernel "x" "x" B) s = some t ∧
      t.mem "x" (s.pids 0) = .mk .bf16 (stableValue M xs) ∧
      (∀ (r : RegionName) o, (r ≠ "x" ∨ o ≠ s.pids 0) → t.mem r o = s.mem r o) :=
  stable_run M "x" "x" B hB xs s hx

private noncomputable def model (logarithm : ℚ → ℚ) : Algebra ℚ where
  literal := fun _ _ r => if r = 0 then 0 else 1
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b
    | .sub => a - b
    | .mul => a * b
    | .div => a / b
    | .max => max a b
    | .pow => a
  unary := fun _ op a => match op with
    | .libdeviceExp => 1
    | .log => logarithm a
    | _ => 0
  cast := fun _ _ _ a => if a = 0 then 0 else 1
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun b => b.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 1
  reduceSum := fun _ _ _ _ _ => 0

private def domain : FP.Guarded.Domain ℚ
  | .finite => fun _ => True
  | .nonzero => fun a => a ≠ 0
  | .positive => fun a => 0 < a

private def initial : State ℚ where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

theorem exp_sub_holds : FP.SoftmaxShift.LibdeviceExpSub (model fun _ => 0) domain := by
  constructor
  intros
  norm_num [FP.SoftmaxShift.exp, FP.ScalarArithmetic.div, model]

theorem log_mul_holds : FP.LogSumExpShift.IntrinsicLogMul (model fun _ => 0) domain := by
  constructor
  intros
  norm_num [FP.LogSumExpShift.log, FP.ScalarArithmetic.add, model]

theorem log_exp_still_missing : ¬ FP.LogSumExpShift.LogLibdeviceExp (model fun _ => 0) domain := by
  intro h
  have bad := h.apply 1 trivial
  norm_num [FP.LogSumExpShift.log, FP.SoftmaxShift.exp, model] at bad

theorem arbitrary_log_mul_fails : ¬ FP.LogSumExpShift.IntrinsicLogMul (model id) domain := by
  intro h
  have bad := h.apply 2 3 trivial trivial (by norm_num [domain]) (by norm_num [domain])
  norm_num [FP.LogSumExpShift.log, FP.ScalarArithmetic.mul, FP.ScalarArithmetic.add, model] at bad

theorem domain_is_satisfiable :
    (StableLogSumExpFPContract.requirements "x" 2 FP.Equational.seededSchedules).Holds
      (model fun _ => 0) domain initial := by
  rw [StableLogSumExpFPContract.requirements_holds]
  change FP.LogSumExpShift.ShiftDomain _ _ _ _ (.add (.input 0) (.add (.input 1) .zero))
  constructor <;>
    norm_num [domain, FP.SoftmaxShift.exp, FP.SoftmaxShift.exponentials,
      FP.SoftmaxShift.shifted, FP.ScalarArithmetic.zero, FP.ScalarArithmetic.add,
      FP.ScalarReduction.value, FP.ScalarReduction.FiniteTree, model]

-- EXP-SUB and LOG-MUL hold above, as does the complete domain, but the
-- original stored results differ until the LOG-EXP obligation is supplied.
theorem original_results_differ_without_log_exp :
    stableValue (SoftmaxStableFPContract.engine (model fun _ => 0) FP.Equational.seededSchedules)
        (fun _ : Fin 2 => 0) ≠
      directValue (SoftmaxStableFPContract.engine (model fun _ => 0) FP.Equational.seededSchedules)
        (fun _ : Fin 2 => 0) := by
  norm_num [stableValue, directValue, SoftmaxStableFPExecution.maximum,
    SoftmaxStableFPContract.engine, FP.Scheduled.Profile.algebra, FP.Scheduled.fp32,
    FP.ScalarReduction.algebra, Algebra.withDefaultPrecision, model]
  change (1 : ℚ) ≠ 0
  norm_num

-- Equality after a bf16 output cast cannot be used as an uncast equality
-- inside the later addition. This checks the precision-binding boundary.
theorem cast_equality_cannot_be_cancelled :
    (model id).cast (some .fp32) .real .bf16 1 = (model id).cast (some .fp32) .real .bf16 2 ∧
    (model id).cast (some .fp32) .real .bf16 ((1 : ℚ) - 1) ≠
      (model id).cast (some .fp32) .real .bf16 ((2 : ℚ) - 1) := by
  norm_num [model]

private def lhs := StableLogSumExpFPContract.stable "x" "y" 2
private def rhs := StableLogSumExpFPContract.direct "x" "y" 2

theorem scalar_output_address_is_observable :
    Spec.ProgramSyntax.signature lhs ≠ Spec.ProgramSyntax.signature
      { lhs with io := { lhs.io with write := fun pid => pid * 2 } } := by
  intro h
  have bad := congrArg (fun sig => sig.1.write 1) h
  cases bad

theorem scalar_output_neighbor_is_framed :
    ¬ IO₁Frame lhs.io initial (initial.write "y" 2 (.mk .bf16 0)) := by
  intro h
  have bad := h "y" 2 (Or.inr (by
    intro i
    change 2 ≠ 1 + i.val
    have hi : i.val < 1 := i.isLt
    omega)) (by simp [lhs, StableLogSumExpFPContract.stable, stableIO, directIO])
  simp [initial, State.write] at bad

specification opaque_logsumexp (h : FP.Scheduled.Equivalent₁ [] lhs rhs) : lhs ≡[[]] rhs :=
  Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl rfl h

-- Rebuilding a record from its projected equations must not conceal an
-- external, unadmitted scalar premise from the assumption printer.
theorem rebuilt_exp_sub {α : Type} (M : Algebra α) (D : FP.Guarded.Domain α)
    (h : FP.SoftmaxShift.LibdeviceExpSub M D) : FP.SoftmaxShift.LibdeviceExpSub M D :=
  ⟨fun a b ha hb => h.apply a b ha hb⟩

theorem rebuilt_log_mul {α : Type} (M : Algebra α) (D : FP.Guarded.Domain α)
    (h : FP.LogSumExpShift.IntrinsicLogMul M D) : FP.LogSumExpShift.IntrinsicLogMul M D :=
  ⟨fun a b ha hb hpa hpb => h.apply a b ha hb hpa hpb⟩

theorem rebuilt_log_exp {α : Type} (M : Algebra α) (D : FP.Guarded.Domain α)
    (h : FP.LogSumExpShift.LogLibdeviceExp M D) : FP.LogSumExpShift.LogLibdeviceExp M D :=
  ⟨fun a ha => h.apply a ha⟩

theorem rebuilt_count {α : Type} (M : Algebra α) (N : Nat)
    (h : FP.WelfordInduction.CountConversion M N) : FP.WelfordInduction.CountConversion M N :=
  ⟨h.zero, fun i hi => h.successor i hi⟩

structure WrappedLogMul {α : Type} (M : Algebra α) (D : FP.Guarded.Domain α) : Prop where
  law : FP.LogSumExpShift.IntrinsicLogMul M D

theorem wrapped_log_mul {α : Type} (M : Algebra α) (D : FP.Guarded.Domain α)
    (h : WrappedLogMul M D) : FP.LogSumExpShift.IntrinsicLogMul M D :=
  ⟨fun a b ha hb hpa hpb => h.law.apply a b ha hb hpa hpb⟩

#axiomsClean FP.LogSumExpShift.recover_sum
#axiomsClean FP.LogSumExpShift.shifted_result
#axiomsClean StableLogSumExpFPContract.original_runs_under_log
#axiomsClean stable_empty_fails
#axiomsClean domain_is_satisfiable
#axiomsClean exp_sub_holds
#axiomsClean log_mul_holds
#axiomsClean log_exp_still_missing
#axiomsClean arbitrary_log_mul_fails
#axiomsClean original_results_differ_without_log_exp
#axiomsClean cast_equality_cannot_be_cancelled
#axiomsClean scalar_output_neighbor_is_framed
#print_fp_assumptions opaque_logsumexp
#print_fp_assumptions rebuilt_exp_sub
#print_fp_assumptions rebuilt_log_mul
#print_fp_assumptions rebuilt_log_exp
#print_fp_assumptions rebuilt_count
#print_fp_assumptions wrapped_log_mul
#print_fp_assumptions exp_sub_holds

end FPStableLogSumExpTests
