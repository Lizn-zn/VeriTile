/- The admitted count atoms operate on nonnegative int32 values embedded in
Nat. Their range fits int32, including the successor, so integer arithmetic
here has the same value as the tested integer arithmetic. Floating conversion
and addition remain opaque. No natural-to-float equality is proved from reals.

The successor's input bound is encoded in its scalar syntax: outside the
accepted range both fragments return the same zero. Extracting the actual
successor equation requires an explicit in-range proof. This avoids extending
the report to arbitrary loop indices or hiding a range condition in prose. -/
import VeriTile.Triton.Float.CountAdmission
import VeriTile.Triton.Float.ScalarArithmetic

namespace VeriTile.Triton.FP.CountConversion
open Structural Guarded ScalarArithmetic
open scoped VeriTile.Spec

inductive Atom where
  | zero | successor
  deriving DecidableEq, Repr

def Atom.ruleID : Atom → String
  -- fp32(int32(0)) → fp32(0); no variable operands.
  | .zero => "COUNT-ZERO"
  -- fp32(i + 1) → fp32(i) + fp32(1); integer 0 ≤ i < 2^24.
  -- The increment is performed in int32; the addition on the right is fp32.
  | .successor => "COUNT-SUCCESSOR"

def candidates : List Atom := [.zero, .successor]

/-- This candidate's integer domain is part of its syntax, not an experiment
shape. Selection also checks that the report used this exact upper bound. -/
def limit : Nat := 16777216

def index : Op .nat [] := .ref .nat [] "i"
def bounded (e : Op .real []) : Op .real [] :=
  .where (.lt .nat .nil index (.constNat limit)) e (.const 0)

def lhsCode : Atom → List ComputeStmt
  | .zero => fragment (.natToReal (.constNat 0))
  | .successor => fragment (bounded (.natToReal (.add .nat .nil index (.constNat 1))))

def rhsCode : Atom → List ComputeStmt
  | .zero => fragment (.const 0)
  | .successor => fragment (bounded (plus (.natToReal index) (.const 1)))

def lhs (a : Atom) : GuardedFragment := ⟨[], lhsCode a⟩
def rhs (a : Atom) : GuardedFragment := ⟨[], rhsCode a⟩

/-- Match the exact relation, precision and operand domain before selecting a row. -/
def Atom.matches (a : Atom) (row : ReportedRule) : Bool :=
  decide (row.ruleID = a.ruleID ∧ row.input = "int32" ∧
    row.compute = "fp32" ∧ row.accumulator = "fp32" ∧ row.output = "fp32" ∧
    (a = .zero ∨ CountAdmission.upperExclusive = limit))

/-- Only accepted rows in the generated table can activate a candidate. -/
def Atom.report? (a : Atom) : Option ReportedRule :=
  CountAdmission.all.find? a.matches

abbrev Atom.Available (a : Atom) : Prop := a.report?.isSome = true

def report (a : Atom) (h : a.Available) : ReportedRule := a.report?.get h

def entry (a : Atom) (h : a.Available) : Spec.RuleEntry GuardedFragment :=
  (report a h).bind [lhs a] [rhs a]

def Atom.entry? (a : Atom) : Option (Spec.RuleEntry GuardedFragment) :=
  a.report?.map fun row => row.bind [lhs a] [rhs a]

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
    Spec.Derivation R.assumptions [lhs a] [rhs a] := by
  apply Spec.Derivation.atom (entry a h)
  · apply List.mem_append_right
    apply List.mem_filterMap.mpr
    refine ⟨a, ?_, ?_⟩
    · cases a <;> simp [candidates]
    · have available : a.report?.isSome = true := h
      cases hr : a.report? with
      | none => simp [hr] at available
      | some row => simp [Atom.entry?, entry, report, hr]
  · exact (report a h).admit _ _ (R.validated a h)

theorem rewrite (R : Rules) (a : Atom) (h : a.Available) : [lhs a] ≡[R] [rhs a] :=
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

end VeriTile.Triton.FP.CountConversion
