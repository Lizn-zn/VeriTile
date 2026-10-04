import VeriTile.Triton.Float.ScalarArithmetic

/-! Scalar fp32 maximum candidates. Defining their fragments does not assert
an ordering or a numerical law for the floating carrier. Accepted report rows
select which candidates can be used as assumptions. -/
namespace VeriTile.Triton.FP.Maximum
open Structural Guarded ScalarArithmetic
open scoped VeriTile.Spec

inductive Atom where
  | max_commute | max_assoc | max_idem | max_neg_inf
  deriving DecidableEq, Repr

/-- Every max below is tl.maximum at fp32 precision. -/
def Atom.ruleID : Atom → String
  -- max(a, b) → max(b, a); finite a,b.
  | .max_commute => "MAX-COMMUTE"
  -- max(max(a, b), c) → max(a, max(b, c)); finite a,b,c.
  | .max_assoc => "MAX-ASSOC"
  -- max(a, a) → a; finite a.
  | .max_idem => "MAX-IDEM"
  -- max(-inf, a) → a; finite a. -inf is a literal, not a sampled input.
  | .max_neg_inf => "MAX-NEG-INF"

def candidates : List Atom := [.max_commute, .max_assoc, .max_idem, .max_neg_inf]

def Atom.guards : Atom → List OperandGuard
  | .max_commute => [⟨"a", .finite⟩, ⟨"b", .finite⟩]
  | .max_assoc => [⟨"a", .finite⟩, ⟨"b", .finite⟩, ⟨"c", .finite⟩]
  | .max_idem | .max_neg_inf => [⟨"a", .finite⟩]

/-- The DSL lowers tl.maximum to a comparison and selection. Retain that
exact syntax so admission can be used on translated Triton fragments. -/
def maximum (a b : Op .real []) : Op .real [] :=
  .where (.gt .real .nil a b) a b

def lhs (a : Atom) : GuardedFragment := ⟨a.guards, fragment (match a with
  | .max_commute => maximum (ref "a") (ref "b")
  | .max_assoc => maximum (maximum (ref "a") (ref "b")) (ref "c")
  | .max_idem => maximum (ref "a") (ref "a")
  | .max_neg_inf => maximum .negInf (ref "a"))⟩

def rhs (a : Atom) : GuardedFragment := ⟨a.guards, fragment (match a with
  | .max_commute => maximum (ref "b") (ref "a")
  | .max_assoc => maximum (ref "a") (maximum (ref "b") (ref "c"))
  | .max_idem | .max_neg_inf => ref "a")⟩

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
  (report a h).report.bind [lhs a] [rhs a]

def Atom.entry? (a : Atom) : Option (Spec.RuleEntry GuardedFragment) :=
  a.report?.map fun row => row.report.bind [lhs a] [rhs a]

/-- External validation is required only for candidates selected by the report. -/
structure Rules where
  validated : ∀ (a : Atom) (h : a.Available),
    Spec.EvidenceValidated (entry a h).rule (entry a h).evidence

def Rules.assumptions (_ : Rules) : Spec.Assumptions GuardedFragment :=
  candidates.filterMap Atom.entry?

instance : CoeOut Rules (Spec.Assumptions GuardedFragment) := ⟨Rules.assumptions⟩

/-- Admission enables this one candidate; it asserts no other numerical law. -/
theorem admitted (R : Rules) (a : Atom) (h : a.Available) :
    Spec.Derivation R.assumptions [lhs a] [rhs a] := by
  apply Spec.Derivation.atom (entry a h)
  · apply List.mem_filterMap.mpr
    refine ⟨a, ?_, ?_⟩
    · cases a <;> simp [candidates]
    · have available : a.report?.isSome = true := h
      cases hr : a.report? with
      | none => simp [hr] at available
      | some row => simp [Atom.entry?, entry, report, hr]
  · exact (report a h).report.admit _ _ (R.validated a h)

theorem rewrite (R : Rules) (a : Atom) (h : a.Available) : [lhs a] ≡[R] [rhs a] :=
  Spec.FloatingPoint.ofDerivation rfl trivial (admitted R a h)

end VeriTile.Triton.FP.Maximum
