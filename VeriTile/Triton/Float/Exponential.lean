import VeriTile.Triton.Float.ScalarArithmetic

/-! Exponential candidates retain the exact intrinsic used by the experiment.
The measured tl.exp EXP-SUB-INTRINSIC relation fails the configured bias gate
(B = 0.1608954387 ULP > 0.05). It remains a candidate but cannot supply the
libdevice.exp assumption used by the stable-softmax derivation. -/
namespace VeriTile.Triton.FP.Exponential
open Structural Guarded ScalarArithmetic
open scoped VeriTile.Spec

inductive Atom where
  | exp_sub | exp_sub_intrinsic | exp_zero | exp_neg_inf_sub
  deriving DecidableEq, Repr

/-- All operations compute in fp32. These are proposed rewrites, not facts. -/
def Atom.ruleID : Atom → String
  -- libdevice.exp(a - b) → libdevice.exp(a) / libdevice.exp(b); finite a,b.
  | .exp_sub => "EXP-SUB"
  -- tl.exp(a - b) → tl.exp(a) / tl.exp(b); finite a,b.
  | .exp_sub_intrinsic => "EXP-SUB-INTRINSIC"
  -- tl.exp(0) → 1; no variable operands.
  | .exp_zero => "EXP-ZERO"
  -- tl.exp(-inf - a) → 0; finite a. -inf is a literal, not a sampled input.
  | .exp_neg_inf_sub => "EXP-NEG-INF-SUB"

def candidates : List Atom := [.exp_sub, .exp_sub_intrinsic, .exp_zero, .exp_neg_inf_sub]

def guards : List OperandGuard := [⟨"a", .finite⟩, ⟨"b", .finite⟩]
def Atom.guards : Atom → List OperandGuard
  | .exp_sub | .exp_sub_intrinsic => VeriTile.Triton.FP.Exponential.guards
  | .exp_zero => []
  | .exp_neg_inf_sub => [⟨"a", .finite⟩]

def Atom.lhs : Atom → GuardedFragment
  | a => ⟨a.guards, fragment (match a with
    | .exp_sub => .libdeviceExp (minus (ref "a") (ref "b"))
    | .exp_sub_intrinsic => .exp (minus (ref "a") (ref "b"))
    | .exp_zero => .exp (.const 0)
    | .exp_neg_inf_sub => .exp (minus .negInf (ref "a")))⟩

def Atom.rhs : Atom → GuardedFragment
  | a => ⟨a.guards, fragment (match a with
    | .exp_sub => divide (.libdeviceExp (ref "a")) (.libdeviceExp (ref "b"))
    | .exp_sub_intrinsic => divide (.exp (ref "a")) (.exp (ref "b"))
    | .exp_zero => .const 1
    | .exp_neg_inf_sub => .const 0)⟩

/-- Match the exact relation, precision and operand domain before selecting a row. -/
def Atom.matches (a : Atom) (row : ReportedScalarRule) : Bool :=
  decide (row.report.ruleID = a.ruleID ∧ row.report.input = "fp32" ∧
    row.report.compute = "fp32" ∧ row.report.accumulator = "fp32" ∧
    row.report.output = "fp32" ∧ row.guards = a.guards)

/-- Only accepted rows in the generated table can activate a candidate. -/
def Atom.report? (a : Atom) : Option ReportedScalarRule :=
  SupplementalAdmission.all.find? a.matches

abbrev Atom.Available (a : Atom) : Prop := a.report?.isSome = true

def report (a : Atom) (h : a.Available) : ReportedScalarRule := a.report?.get h

def entry (a : Atom) (h : a.Available) : Spec.RuleEntry GuardedFragment :=
  (report a h).report.bind [Atom.lhs a] [Atom.rhs a]

def Atom.entry? (a : Atom) : Option (Spec.RuleEntry GuardedFragment) :=
  a.report?.map fun row => row.report.bind [Atom.lhs a] [Atom.rhs a]

/-- External validation is required only for candidates selected by the report. -/
structure Rules where
  arithmetic : ScalarArithmetic.Rules
  validated : ∀ (a : Atom) (h : a.Available),
    Spec.EvidenceValidated (entry a h).rule (entry a h).evidence

def Rules.assumptions (R : Rules) : Spec.Assumptions GuardedFragment :=
  R.arithmetic.assumptions ++ candidates.filterMap Atom.entry?

instance : CoeOut Rules (Spec.Assumptions GuardedFragment) := ⟨Rules.assumptions⟩

/-- Admission enables this one candidate; it asserts no other numerical law. -/
theorem admitted (R : Rules) (a : Atom) (h : a.Available) :
    Spec.Derivation R.assumptions [Atom.lhs a] [Atom.rhs a] := by
  apply Spec.Derivation.atom (entry a h)
  · apply List.mem_append_right
    apply List.mem_filterMap.mpr
    refine ⟨a, ?_, ?_⟩
    · cases a <;> simp [candidates]
    · have available : a.report?.isSome = true := h
      cases hr : a.report? with
      | none => simp [hr] at available
      | some row => simp [Atom.entry?, entry, report, hr]
  · exact (report a h).report.admit _ _ (R.validated a h)

theorem rewrite (R : Rules) (a : Atom) (h : a.Available) : [Atom.lhs a] ≡[R] [Atom.rhs a] :=
  Spec.FloatingPoint.ofDerivation rfl trivial (admitted R a h)

theorem arithmetic_derivation (R : Rules) {lhs rhs : List GuardedFragment}
    (h : Spec.Derivation R.arithmetic.assumptions lhs rhs) :
    Spec.Derivation R.assumptions lhs rhs := by
  induction h with
  | refl code => exact .refl code
  | atom e he ha => exact .atom e (List.mem_append_left _ he) ha
  | symm _ ih => exact .symm ih
  | trans _ _ ih₁ ih₂ => exact .trans ih₁ ih₂
  | frame beforeCode afterCode _ ih => exact .frame beforeCode afterCode ih

theorem arithmetic_models {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) :
    Models R.arithmetic.assumptions M D :=
  fun lhs rhs h => hM lhs rhs (arithmetic_derivation R h)

def lhs : GuardedFragment := Atom.lhs .exp_sub
def rhs : GuardedFragment := Atom.rhs .exp_sub

end VeriTile.Triton.FP.Exponential
