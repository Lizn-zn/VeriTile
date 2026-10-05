import bench.examples.StableLogSumExp.FPEquiv
import bench.examples.StableLogSumExp.Correct
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.NormNum

/- Rational control-flow fixture, not a model of the admitted numerical laws.
Different log symbols and a visible bf16 cast detect accidental substitution. -/
noncomputable section
namespace FPLogSumExpCandidateTests
open VeriTile Triton FP.Structural Bench.Examples
open StableLogSumExpFPExecution
set_option maxHeartbeats 2400000

private def lt (a b : ℚ) : Bool := decide (a < b)
private def le (a b : ℚ) : Bool := decide (a ≤ b)

private def model : Algebra ℚ where
  literal := fun _ _ r => if r = 0 then 0 else if r = 1 then 1 else
    if r = 2 then 2 else if r = 80 then 80 else 1 / 2
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b
    | .sub => a - b
    | .mul => a * b
    | .div => a / b
    | _ => 0
  unary := fun _ op a => match op with
    | .libdeviceExp => 1
    | .libdeviceLog => a + 10
    | .log => a + 100
    | _ => 0
  compareLt := fun p _ => if p = some .fp32 then some lt else none
  compareLe := fun p _ => if p = some .fp32 then some le else none
  cast := fun _ _ target a => if target = .bf16 then a + 1 / 4 else a
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun b => b.bits.toNat
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => -1
  reduceSum := fun _ {_} _ _ _ _ => 0

/-- Endpoints keep the product log, even when the center could simplify. -/
theorem product_endpoints :
    FP.LogSumExpCandidate.finish model lt le (1 / 2) 1 = 201 / 2 ∧
    FP.LogSumExpCandidate.finish model lt le 2 1 = 102 ∧
    FP.LogSumExpCandidate.finish model lt le (1 / 4) 1 = 405 / 4 ∧
    FP.LogSumExpCandidate.finish model lt le 4 1 = 105 := by
  norm_num [FP.LogSumExpCandidate.finish, FP.LogExp.value, FP.LogExp.Backend.logOp, model, lt, le]

/-- The split arm retains both signs of the strict lower and inclusive upper
center threshold, as well as the near-zero and extreme-input fallbacks. -/
theorem center_endpoints :
    FP.LogSumExpCandidate.finish model lt le 4 (1 / 2) = 205 ∧
    FP.LogSumExpCandidate.finish model lt le 4 (-(1 / 2)) = 205 ∧
    FP.LogSumExpCandidate.finish model lt le 4 80 = 184 ∧
    FP.LogSumExpCandidate.finish model lt le 4 (-80) = 24 ∧
    FP.LogSumExpCandidate.finish model lt le 4 0 = 205 ∧
    FP.LogSumExpCandidate.finish model lt le 4 81 = 205 ∧
    FP.LogSumExpCandidate.finish model lt le 4 (-81) = 205 := by
  norm_num [FP.LogSumExpCandidate.finish, FP.LogExp.value, FP.LogExp.Backend.logOp, model, lt, le]

private def initial : State ℚ where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

/-- The source executes its negative-center split and keeps the observable
bf16 cast and scalar output address, including when input and output alias. -/
theorem negative_center_in_place :
    ∃ t, exec (FP.Scheduled.fp32.algebra model FP.Equational.seededSchedules)
        (StableLogSumExp.Kernels.candidateLSEKernel "x" "x" 4) initial = some t ∧
      t.mem "x" 1 = .mk .bf16 (413 / 4) ∧
      ∀ (r : RegionName) o, (r ≠ "x" ∨ o ≠ 1) → t.mem r o = initial.mem r o := by
  obtain ⟨t, ht, hv, hf⟩ := candidate_run model FP.Equational.seededSchedules lt le
    (by simp [model]) (by simp [model]) "x" "x" 4 (by decide)
    (fun _ => 0) initial (fun _ => rfl)
  refine ⟨t, ht, ?_, hf⟩
  change t.mem "x" (initial.pids 0) = _
  rw [hv]
  have hs : SoftmaxStableFPExecution.rowSum
      (FP.Scheduled.fp32.algebra model FP.Equational.seededSchedules)
      (SoftmaxStableFPExecution.shifted
        (FP.Scheduled.fp32.algebra model FP.Equational.seededSchedules) (fun _ : Fin 4 => 0)) = 4 := by
    change FP.ScalarReduction.value model (fun _ : Fin 4 => 1) (FP.ScalarArithmetic.zero model)
      (.add (.input 0) (.add (.input 1) (.add (.input 2) (.add (.input 3) .zero)))) = 4
    norm_num [FP.ScalarReduction.value, FP.ScalarArithmetic.add, FP.ScalarArithmetic.zero, model]
  simp only [candidateValue, hs]
  norm_num [FP.LogSumExpCandidate.finish, FP.LogExp.value, FP.LogExp.Backend.logOp,
    SoftmaxStableFPExecution.maximum, FP.Scheduled.Profile.algebra,
    FP.Scheduled.fp32, FP.ScalarReduction.algebra, Algebra.withDefaultPrecision,
    resolvePrecision, model, lt, le]

example (B : Nat) : (StableLogSumExpCorrect.candidateIO B).kernel =
    (candidateIO "x" "y" B).kernel.eraseDType := rfl

theorem empty_row_fails :
    exec (FP.Scheduled.fp32.algebra model FP.Equational.seededSchedules)
      (StableLogSumExp.Kernels.candidateLSEKernel "x" "y" 0) initial = none := by
  simp [StableLogSumExp.Kernels.candidateLSEKernel, FP.Structural.exec, run, step, evalExpr,
    evalOp_unfold, numeric, TileShape.axisDim, Option.bind]

/-- A numerical law does not silently supply a missing comparison operation. -/
theorem missing_comparison_fails :
    exec (FP.Scheduled.fp32.algebra { model with compareLe := fun _ _ => none }
        FP.Equational.seededSchedules)
      (StableLogSumExp.Kernels.candidateLSEKernel "x" "y" 1) initial = none := by
  simp [StableLogSumExp.Kernels.candidateLSEKernel, FP.Structural.exec, run, step, evalExpr,
    evalOp_unfold, numeric, numericLe, TileShape.axisDim,
    FP.Scheduled.Profile.algebra, FP.Scheduled.fp32, FP.ScalarReduction.algebra,
    Algebra.withDefaultPrecision]

/-- The domain permits both product branches; negative centers are valid.
This fixture only checks contract satisfiability, not numerical admission. -/
private def domain : FP.Guarded.Domain ℚ
  | .finite => fun _ => True
  | .positive => fun a => 0 < a
  | .nonzero => fun a => a ≠ 0

theorem domain_is_satisfiable :
    (StableLogSumExpFPContract.requirements "x" 2 FP.Equational.seededSchedules).Holds
      model domain initial := by
  rw [StableLogSumExpFPContract.requirements_holds]
  change FP.LogSumExpShift.ShiftDomain _ _ _ _ (.add (.input 0) (.add (.input 1) .zero))
  constructor <;>
    norm_num [domain, FP.SoftmaxShift.exp, FP.SoftmaxShift.exponentials,
      FP.SoftmaxShift.shifted, FP.ScalarArithmetic.zero, FP.ScalarArithmetic.add,
      FP.ScalarReduction.value, FP.ScalarReduction.FiniteTree, model]

theorem split_domain_is_satisfiable :
    (StableLogSumExpFPContract.requirements "x" 4 FP.Equational.seededSchedules).Holds
      model domain initial := by
  rw [StableLogSumExpFPContract.requirements_holds]
  change FP.LogSumExpShift.ShiftDomain _ _ _ _
    (.add (.input 0) (.add (.input 1) (.add (.input 2) (.add (.input 3) .zero))))
  constructor <;>
    norm_num [domain, FP.SoftmaxShift.exp, FP.SoftmaxShift.exponentials,
      FP.SoftmaxShift.shifted, FP.ScalarArithmetic.zero, FP.ScalarArithmetic.add,
      FP.ScalarReduction.value, FP.ScalarReduction.FiniteTree, model]

theorem comparison_contract_is_satisfiable (B : Nat) :
    ∀ c ∈ (StableLogSumExpFPContract.candidate "x" "y" B).comparisons,
      c.Supported model := by
  intro c hc
  simp only [StableLogSumExpFPContract.candidate, List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl <;> simp [FP.Scheduled.Comparison.Supported, model]

theorem comparison_support_is_part_of_signature (B : Nat) :
    Spec.ProgramSyntax.signature (StableLogSumExpFPContract.candidate "x" "y" B) ≠
      Spec.ProgramSyntax.signature
        { StableLogSumExpFPContract.candidate "x" "y" B with comparisons := [] } := by
  intro h
  have bad := congrArg (fun sig => sig.2.2.2) h
  cases bad

example (B : Nat) : (StableLogSumExpFPContract.original "x" "y" B).io.kernel =
    StableLogSumExp.Kernels.directLSEKernel "x" "y" B := rfl

example (B : Nat) : (StableLogSumExpFPContract.candidate "x" "y" B).io.kernel =
    StableLogSumExp.Kernels.candidateLSEKernel "x" "y" B := rfl

#guard_msgs (drop info) in
#axiomsClean StableLogSumExpFPEquiv.logsumexp_equiv

#guard_msgs (drop info) in
#axiomsClean FP.LogSumExpCandidate.finish_eq
#guard_msgs (drop info) in
#axiomsClean candidate_run
#guard_msgs (drop info) in
#axiomsClean StableLogSumExpCorrect.candidate_logsumexp_correct

end FPLogSumExpCandidateTests
