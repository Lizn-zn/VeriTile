import VeriTile.Triton.Float.ScalarArithmetic
import VeriTile.Triton.Float.ExpAdmission

/-! Exponential candidates retain the exact intrinsic used by the experiment.
Prefer tl.exp when the required relation is admitted. The paired fp32 report
accepts both implementations of exp-zero and exp-neg-inf-sub, but only
libdevice.exp for exp-sub. A report never equates the two implementations.
Candidates remain available to describe even when their relations are rejected. -/
namespace VeriTile.Triton.FP.Exponential
open Structural Guarded ScalarArithmetic
open scoped VeriTile.Spec

inductive Backend where
  | tl | libdevice
  deriving DecidableEq, Repr

inductive Atom where
  | exp_sub (exp : Backend := .tl)
  | exp_zero (exp : Backend := .tl)
  | exp_neg_inf_sub (exp : Backend := .tl)
  deriving DecidableEq, Repr

/-- All operations compute in fp32. These are proposed rewrites, not facts. -/
def Atom.ruleID : Atom → String
  -- exp(a - b) → exp(a) / exp(b); finite a,b; same backend on both sides.
  | .exp_sub .tl => "EXP-SUB-INTRINSIC"
  | .exp_sub .libdevice => "EXP-SUB"
  -- exp(0) → 1; no variable operands.
  | .exp_zero .tl => "EXP-ZERO"
  | .exp_zero .libdevice => "EXP-ZERO-LIBDEVICE"
  -- exp(-inf - a) → 0; finite a. -inf is a literal, not a sampled input.
  | .exp_neg_inf_sub .tl => "EXP-NEG-INF-SUB"
  | .exp_neg_inf_sub .libdevice => "EXP-NEG-INF-SUB-LIBDEVICE"

def candidates : List Atom := [.exp_sub .tl, .exp_sub .libdevice,
  .exp_zero .tl, .exp_zero .libdevice, .exp_neg_inf_sub .tl, .exp_neg_inf_sub .libdevice]

def Backend.exp : Backend → Op .real [] → Op .real []
  | .tl => .exp
  | .libdevice => .libdeviceExp

def guards : List OperandGuard := [⟨"a", .finite⟩, ⟨"b", .finite⟩]
def Atom.guards : Atom → List OperandGuard
  | .exp_sub _ => VeriTile.Triton.FP.Exponential.guards
  | .exp_zero _ => []
  | .exp_neg_inf_sub _ => [⟨"a", .finite⟩]

def Atom.lhs : Atom → GuardedFragment
  | a => ⟨a.guards, fragment (match a with
    | .exp_sub exp => exp.exp (minus (ref "a") (ref "b"))
    | .exp_zero exp => exp.exp (.const 0)
    | .exp_neg_inf_sub exp => exp.exp (minus .negInf (ref "a")))⟩

def Atom.rhs : Atom → GuardedFragment
  | a => ⟨a.guards, fragment (match a with
    | .exp_sub exp => divide (exp.exp (ref "a")) (exp.exp (ref "b"))
    | .exp_zero _ => .const 1
    | .exp_neg_inf_sub _ => .const 0)⟩

/-- Match the exact relation, precision and operand domain before selecting a row. -/
def Atom.matches (a : Atom) (row : ReportedScalarRule) : Bool :=
  decide (row.report.ruleID = a.ruleID ∧ row.report.input = "fp32" ∧
    row.report.compute = "fp32" ∧ row.report.accumulator = "fp32" ∧
    row.report.output = "fp32" ∧ row.guards = a.guards)

/-- Only accepted rows in the generated table can activate a candidate. -/
def Atom.report? (a : Atom) : Option ReportedScalarRule :=
  ExpAdmission.all.find? a.matches

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
    · rcases a with ⟨_ | _⟩ | ⟨_ | _⟩ | ⟨_ | _⟩ <;> simp [candidates]
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

/-- The stable-softmax derivation specifically requires libdevice.exp. -/
def lhs : GuardedFragment := Atom.lhs (.exp_sub .libdevice)
def rhs : GuardedFragment := Atom.rhs (.exp_sub .libdevice)

end VeriTile.Triton.FP.Exponential
