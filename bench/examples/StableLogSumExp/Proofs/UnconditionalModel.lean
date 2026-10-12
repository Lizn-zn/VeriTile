import VeriTile.Triton.Float.Countermodel
import bench.examples.StableLogSumExp.Proofs.FP

/-! A finite logical model of the selected scalar rules, not IEEE arithmetic.
The finite carrier is the two-element field, plus a separate negative-infinity
symbol. Both exp backends map finite values to one and negative infinity to
zero. Both log backends return zero. Comparisons are available and choose the
fallbacks: lt is false and le is true. These interpretations are permitted by
the opaque algebra; their purpose is to refute derivability from the admitted
relations, not to predict a GPU result or a statistical gate outcome. -/
noncomputable section
namespace VeriTile.Bench.Examples.StableLogSumExpUnconditional.Model
open VeriTile Triton FP FP.Structural FP.Guarded

/-- `some false` and `some true` encode the field's zero and one; `none` is
the separate -inf symbol. These are symbolic carrier values, not BF16 bits. -/
abbrev Number := Option Bool

def lift₂ (f : Bool → Bool → Bool) : Number → Number → Number
  | some a, some b => some (f a b)
  | _, _ => none

def algebra : Algebra Number where
  literal := fun _ _ r => some (if r = 0 then Bool.false else Bool.true)
  negInf := none
  binary := fun _ _ op => match op with
    | .add | .sub => lift₂ Bool.xor
    | .mul | .div => lift₂ Bool.and
    | .max => lift₂ Bool.or
    | .pow => fun a _ => a
  unary := fun _ op a => match op with
    | .exp | .libdeviceExp => some a.isSome
    | _ => some Bool.false
  compareLt := fun _ _ => some (fun _ _ => Bool.false)
  compareLe := fun _ _ => some (fun _ _ => Bool.true)
  cast := fun _ _ _ => id
  fromNat := fun _ n => some (decide (n % 2 = 1))
  fromInt := fun _ n => some (decide (n % 2 = 1))
  fp32Bits := fun b => some (decide (b.bits.toNat % 2 = 1))
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => some Bool.true
  reduceSum := fun _ {_} _ _ _ _ => some Bool.false

def domain : Domain Number
  | .finite => fun a => a.isSome = Bool.true
  | .nonzero | .positive => fun a => a = some Bool.true

private theorem finite_register (s : State Number) (gs : List OperandGuard)
    (hd : ScalarDomain domain gs s) (name : RegName) (hm : ⟨name, .finite⟩ ∈ gs) :
    ∃ b : Bool, s.regs .real [] name = some (fun _ => some b) := by
  obtain ⟨v, hv, hf⟩ := hd _ hm
  cases he : v PUnit.unit with
  | none => simp [domain, he] at hf
  | some b =>
      refine ⟨b, hv.trans (congrArg some ?_)⟩
      funext i
      cases i
      exact he

private theorem nonzero_register (s : State Number) (gs : List OperandGuard)
    (hd : ScalarDomain domain gs s) (name : RegName) (hm : ⟨name, .nonzero⟩ ∈ gs)
    (b : Bool) (hb : s.regs .real [] name = some (fun _ => some b)) : b = Bool.true := by
  obtain ⟨v, hv, hn⟩ := hd _ hm
  have eqv := Option.some.inj (hv.symm.trans hb)
  simpa only [domain, eqv, Option.some.injEq] using hn

set_option linter.unusedSimpArgs false

set_option maxHeartbeats 1600000 in
theorem arithmetic_atom (atom : ScalarArithmetic.Atom) :
    FragmentSound algebra domain (ScalarArithmetic.lhs atom) (ScalarArithmetic.rhs atom) := by
  refine ⟨rfl, fun s hd => ?_⟩
  change ScalarDomain domain (ScalarArithmetic.guards atom) s at hd
  cases atom
  all_goals
    obtain ⟨a, ha⟩ := finite_register s _ hd "a" (by decide)
    try
      have h : ∃ b : Bool, s.regs .real [] "b" = some (fun _ => some b) := by
        apply finite_register s _ hd
        decide
      obtain ⟨b, hb⟩ := h
    try
      have h : ∃ c : Bool, s.regs .real [] "c" = some (fun _ => some c) := by
        apply finite_register s _ hd
        decide
      obtain ⟨c, hc⟩ := h
    try have hn := nonzero_register s _ hd "b" (by decide) b hb
    try have hn := nonzero_register s _ hd "a" (by decide) a ha
    simp only [ScalarArithmetic.lhs, ScalarArithmetic.rhs, ScalarArithmetic.lhsCode,
      ScalarArithmetic.rhsCode, ScalarArithmetic.fragment, ScalarArithmetic.ref,
      ScalarArithmetic.plus, ScalarArithmetic.minus, ScalarArithmetic.times,
      ScalarArithmetic.divide, Reciprocal.lhs, Reciprocal.rhs, Reciprocal.div,
      Reciprocal.inv, Reciprocal.mul, Reciprocal.ref,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      numeric, ha, State.setReg_same]
    simp_all [State.setReg, algebra, lift₂, bop, Function.comp_def]
    all_goals try cases a
    all_goals try cases b
    all_goals try cases c
    all_goals rfl

set_option maxHeartbeats 1600000 in
theorem exponential_atom (atom : Exponential.Atom) :
    FragmentSound algebra domain atom.lhs atom.rhs := by
  refine ⟨rfl, fun s hd => ?_⟩
  change ScalarDomain domain atom.guards s at hd
  rcases atom with ⟨_ | _⟩ | ⟨_ | _⟩ | ⟨_ | _⟩
  all_goals
    try
      have h : ∃ a : Bool, s.regs .real [] "a" = some (fun _ => some a) := by
        apply finite_register s _ hd
        decide
      obtain ⟨a, ha⟩ := h
    try
      have h : ∃ b : Bool, s.regs .real [] "b" = some (fun _ => some b) := by
        apply finite_register s _ hd
        decide
      obtain ⟨b, hb⟩ := h
    simp only [Exponential.Atom.lhs, Exponential.Atom.rhs, Exponential.Backend.exp,
      ScalarArithmetic.fragment, ScalarArithmetic.ref, ScalarArithmetic.minus,
      ScalarArithmetic.divide, run, step, evalExpr, evalComputeOp, evalOp_unfold,
      ComputeDType.eraseDType, numeric, State.setReg_same]
    simp_all [State.setReg, algebra, lift₂, bop, Function.comp_def]
    all_goals rfl

set_option maxHeartbeats 1600000 in
private theorem log_split (backend : LogExp.Backend) :
    FragmentSound algebra domain (LogExp.Atom.log_mul_split backend).lhs
      (LogExp.Atom.log_mul_split backend).rhs := by
  refine ⟨rfl, fun s hd => ?_⟩
  obtain ⟨a, ha⟩ := finite_register s _ hd "a"
    (by simp [LogExp.Atom.lhs, LogExp.Atom.guards, LogExp.productGuards])
  obtain ⟨b, hb⟩ := finite_register s _ hd "b"
    (by simp [LogExp.Atom.lhs, LogExp.Atom.guards, LogExp.productGuards])
  cases backend <;>
    simp [LogExp.Atom.lhs, LogExp.Atom.rhs, LogExp.Backend.log,
      LogExp.splitProduct, LogExp.keepProduct, LogExp.input, LogExp.secondInput,
      LogExp.assignOutput, run, step, evalExpr, evalComputeOp, evalOp_unfold,
      ComputeDType.eraseDType, numeric, numericLt, numericLe, bop,
      ha, hb, algebra, lift₂, State.setReg_same, Function.comp_def]

set_option maxHeartbeats 1600000 in
private theorem log_cancel (log exp : LogExp.Backend) :
    FragmentSound algebra domain (LogExp.Atom.log_exp_cancel log exp).lhs
      (LogExp.Atom.log_exp_cancel log exp).rhs := by
  refine ⟨rfl, fun s hd => ?_⟩
  obtain ⟨a, ha⟩ := finite_register s _ hd "a"
    (by simp [LogExp.Atom.lhs, LogExp.Atom.guards, LogExp.guards])
  cases log <;> cases exp <;>
    simp [LogExp.Atom.lhs, LogExp.Atom.rhs, LogExp.Backend.log, LogExp.Backend.exp,
      LogExp.expression, LogExp.useIdentity, LogExp.absolute, LogExp.input,
      LogExp.assignOutput, run, step, evalExpr, evalComputeOp, evalOp_unfold,
      ComputeDType.eraseDType, numeric, numericLt, numericLe, bop,
      ha, algebra, lift₂, State.setReg_same, Function.comp_def]

private theorem logarithmic_atom (atom : LogExp.Atom) (selected : atom.Available) :
    FragmentSound algebra domain atom.lhs atom.rhs := by
  rcases atom with ⟨_ | _⟩ | ⟨_ | _⟩ | ⟨_ | _, _ | _⟩ | ⟨_ | _, _ | _⟩
  all_goals first
    | exact log_split _
    | exact log_cancel _ _
    | exact False.elim ((by decide : ¬ _) selected)

private theorem exponential_entries (R : Exponential.Rules) (entry)
    (member : entry ∈ R.assumptions) :
    List.Forall₂ (FragmentSound algebra domain) entry.rule.lhs entry.rule.rhs := by
  rcases List.mem_append.mp member with arithmetic | exponential
  · obtain ⟨atom, _, he⟩ := List.mem_filterMap.mp arithmetic
    unfold ScalarArithmetic.Atom.entry? at he
    cases hr : atom.report? with
    | none => simp [hr] at he
    | some row =>
        simp only [hr, Option.map_some, Option.some.injEq] at he
        subst entry
        exact .cons (arithmetic_atom atom) .nil
  · obtain ⟨atom, _, he⟩ := List.mem_filterMap.mp exponential
    unfold Exponential.Atom.entry? at he
    cases hr : atom.report? with
    | none => simp [hr] at he
    | some row =>
        simp only [hr, Option.map_some, Option.some.injEq] at he
        subst entry
        exact .cons (exponential_atom atom) .nil

theorem exponential_models (R : Exponential.Rules) : Models R.assumptions algebra domain :=
  models_of_atoms algebra domain R.assumptions (exponential_entries R)

/-- Every selected arithmetic, exp and log entry holds with its actual guards.
The resulting model covers the complete derivation closure. -/
theorem models (R : LogSumExp.Rules) : Models R.assumptions algebra domain := by
  apply models_of_atoms
  intro entry member
  rcases List.mem_append.mp member with exponential | logarithmic
  · exact exponential_entries R.exponential entry exponential
  · obtain ⟨atom, _, he⟩ := List.mem_filterMap.mp logarithmic
    unfold LogExp.Atom.entry? at he
    cases hr : atom.report? with
    | none => simp [hr] at he
    | some row =>
        simp only [hr, Option.map_some, Option.some.injEq] at he
        subst entry
        exact .cons (logarithmic_atom atom (by simp [LogExp.Atom.Available, hr])) .nil

def initial : State Number where
  mem := fun _ _ => .mk .real (some Bool.true)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- The singleton row satisfies every input and intermediate domain check. -/
theorem domain_holds :
    (StableLogSumExpFPContract.requirements "x" 1 Equational.seededSchedules).Holds
      algebra domain initial := by
  rw [StableLogSumExpFPContract.requirements_holds]
  change LogSumExpShift.ShiftDomain algebra domain (fun _ : Fin 1 => some Bool.true)
    (some Bool.true) (.add (.input 0) .zero)
  constructor <;>
    simp [domain, SoftmaxShift.exp, SoftmaxShift.exponentials, SoftmaxShift.shifted,
      SoftmaxShift.scale, ScalarArithmetic.zero, ScalarArithmetic.one, ScalarArithmetic.add,
      ScalarArithmetic.sub, ScalarArithmetic.mul, ScalarArithmetic.div,
      ScalarReduction.value, ScalarReduction.FiniteTree, LogSumExpShift.log, algebra, lift₂]

/-- Execute the shared Triton sources, including their bf16 cast and store.
Both runs succeed; the difference is observable at the declared output. -/
theorem stored_outputs :
    Scheduled.observe₁ algebra Equational.seededSchedules initial
      (StableLogSumExpFPContract.stable "x" "y" 1) 0 =
        some (.mk .bf16 (some Bool.true)) ∧
    Scheduled.observe₁ algebra Equational.seededSchedules initial
      (StableLogSumExpFPContract.direct "x" "y" 1) 0 =
        some (.mk .bf16 (some Bool.false)) := by
  let engine := SoftmaxStableFPContract.engine algebra Equational.seededSchedules
  obtain ⟨a, ha, hout, _⟩ := StableLogSumExpFPExecution.stable_run engine "x" "y" 1
    (by decide) (fun _ => some Bool.true) initial (by intro i; rfl)
  obtain ⟨b, hb, hout', _⟩ := StableLogSumExpFPExecution.direct_run engine "x" "y" 1
    (fun _ => some Bool.true) initial (by intro i; rfl)
  change a.mem "y" 0 = _ at hout
  change b.mem "y" 0 = _ at hout'
  constructor
  · change (Structural.exec engine (StableLogSumExp.Kernels.stableLSEKernel "x" "y" 1)
      initial).map (fun t => t.mem "y" 0) = _
    rw [ha]
    change some (a.mem "y" 0) = _
    rw [hout]
    rfl
  · change (Structural.exec engine (StableLogSumExp.Kernels.directLSEKernel "x" "y" 1)
      initial).map (fun t => t.mem "y" 0) = _
    rw [hb]
    change some (b.mem "y" 0) = _
    rw [hout']
    rfl

end VeriTile.Bench.Examples.StableLogSumExpUnconditional.Model
