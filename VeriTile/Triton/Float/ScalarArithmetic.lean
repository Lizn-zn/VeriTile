import VeriTile.Triton.Float.Reciprocal

/-! Scalar fp32 candidates are defined before admission. Their exact fragments
and operand guards remain available even if the report selects no candidates.
ScalarArithmeticLaws contains applications to the currently selected rules;
no ring instance or IEEE equality is assumed here. -/

namespace VeriTile.Triton.FP.ScalarArithmetic
open Structural Guarded
open scoped VeriTile.Spec

inductive Atom where
  | addCommute | addAssociate | mulCommute | mulAssociate | mulDistribute | cancel
  | addZero | mulOne | divOne | divMulRcp | mulRcpCancel
  deriving DecidableEq, Repr

/-- Candidate fp32 rewrites; each arrow requires its own two-gates admission. -/
def Atom.ruleID : Atom → String
  -- a + b → b + a; finite a,b.
  | .addCommute => "ADD-COMMUTE"
  -- (a + b) + c → a + (b + c); finite a,b,c.
  | .addAssociate => "ADD-ASSOC"
  -- a * b → b * a; finite a,b.
  | .mulCommute => "MUL-COMMUTE"
  -- (a * b) * c → a * (b * c); finite a,b,c.
  | .mulAssociate => "MUL-ASSOC"
  -- a * (b + c) → a * b + a * c; finite a,b,c.
  | .mulDistribute => "MUL-DISTRIB"
  -- (a - b) + b → a; finite a,b.
  | .cancel => "CANCEL"
  -- a + 0 → a; finite a.
  | .addZero => "ADD-ZERO"
  -- a * 1 → a; finite a.
  | .mulOne => "MUL-ONE"
  -- a / 1 → a; finite a.
  | .divOne => "DIV-ONE"
  -- a / b → a * (1 / b); ordinary Triton division, finite a,b and b ≠ 0.
  | .divMulRcp => "DIV-MUL-RCP"
  -- a * (1 / a) → 1; ordinary Triton division, finite nonzero a.
  | .mulRcpCancel => "MUL-RCP-CANCEL"

def candidates : List Atom := [.addCommute, .addAssociate, .mulCommute, .mulAssociate,
  .mulDistribute, .cancel, .addZero, .mulOne, .divOne, .divMulRcp, .mulRcpCancel]

def guards : Atom → List OperandGuard
  | .addCommute | .mulCommute | .cancel => [⟨"a", .finite⟩, ⟨"b", .finite⟩]
  | .addAssociate | .mulAssociate | .mulDistribute =>
      [⟨"a", .finite⟩, ⟨"b", .finite⟩, ⟨"c", .finite⟩]
  | .addZero | .mulOne | .divOne => [⟨"a", .finite⟩]
  | .divMulRcp => [⟨"a", .finite⟩, ⟨"b", .finite⟩, ⟨"b", .nonzero⟩]
  | .mulRcpCancel => [⟨"a", .finite⟩, ⟨"a", .nonzero⟩]

def ref (name : String) : Op .real [] := .ref .real [] name
def plus (a b : Op .real []) : Op .real [] := .add .real .nil a b
def minus (a b : Op .real []) : Op .real [] := .sub .real .nil a b
def times (a b : Op .real []) : Op .real [] := .mul .real .nil a b
def divide (a b : Op .real []) : Op .real [] := .div .real .nil a b
def fragment (e : Op .real []) : List ComputeStmt :=
  [.assign .real [] "out" (.compute (.alg .fp32 e))]

def lhsCode : Atom → List ComputeStmt
  | .addCommute => fragment (plus (ref "a") (ref "b"))
  | .addAssociate => fragment (plus (plus (ref "a") (ref "b")) (ref "c"))
  | .mulCommute => fragment (times (ref "a") (ref "b"))
  | .mulAssociate => fragment (times (times (ref "a") (ref "b")) (ref "c"))
  | .mulDistribute => fragment (times (ref "a") (plus (ref "b") (ref "c")))
  | .cancel => fragment (plus (minus (ref "a") (ref "b")) (ref "b"))
  | .addZero => fragment (plus (ref "a") (.const 0))
  | .mulOne => fragment (times (ref "a") (.const 1))
  | .divOne => fragment (divide (ref "a") (.const 1))
  | .divMulRcp => (Reciprocal.lhs .fp32).code
  | .mulRcpCancel => fragment (times (ref "a") (divide (.const 1) (ref "a")))

def rhsCode : Atom → List ComputeStmt
  | .addCommute => fragment (plus (ref "b") (ref "a"))
  | .addAssociate => fragment (plus (ref "a") (plus (ref "b") (ref "c")))
  | .mulCommute => fragment (times (ref "b") (ref "a"))
  | .mulAssociate => fragment (times (ref "a") (times (ref "b") (ref "c")))
  | .mulDistribute => fragment (plus (times (ref "a") (ref "b")) (times (ref "a") (ref "c")))
  | .cancel | .addZero | .mulOne | .divOne => fragment (ref "a")
  | .divMulRcp => (Reciprocal.rhs .fp32).code
  | .mulRcpCancel => fragment (.const 1)

def lhs (a : Atom) : GuardedFragment := ⟨guards a, lhsCode a⟩
def rhs (a : Atom) : GuardedFragment := ⟨guards a, rhsCode a⟩

/-- Match the exact relation, precision and operand domain before selecting a row. -/
def Atom.matches (a : Atom) (row : ReportedScalarRule) : Bool :=
  decide (row.report.ruleID = a.ruleID ∧ row.report.input = "fp32" ∧
    row.report.compute = "fp32" ∧ row.report.accumulator = "fp32" ∧
    row.report.output = "fp32" ∧ row.guards = guards a)

/-- Only accepted rows in the generated tables can activate a candidate.
The six original arithmetic probes have finite-operand domains in their
registry; the supplemental table stores its domain guards explicitly. -/
def Atom.report? (a : Atom) : Option ReportedScalarRule :=
  ((match a with
    | .addCommute | .addAssociate | .mulCommute | .mulAssociate | .mulDistribute | .cancel =>
        ReportedAdmission.all.map fun row => ⟨row, guards a⟩
    | _ => []) ++ SupplementalAdmission.all).find? a.matches

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

def add {α : Type} (M : Algebra α) := M.binary (some .fp32) .real .add
def sub {α : Type} (M : Algebra α) := M.binary (some .fp32) .real .sub
def mul {α : Type} (M : Algebra α) := M.binary (some .fp32) .real .mul
def div {α : Type} (M : Algebra α) := M.binary (some .fp32) .real .div
def zero {α : Type} (M : Algebra α) := M.literal (some .fp32) .real 0
def one {α : Type} (M : Algebra α) := M.literal (some .fp32) .real 1

def leftValue {α : Type} (M : Algebra α) (a b c : α) : Atom → α
  | .addCommute => add M a b
  | .addAssociate => add M (add M a b) c
  | .mulCommute => mul M a b
  | .mulAssociate => mul M (mul M a b) c
  | .mulDistribute => mul M a (add M b c)
  | .cancel => add M (sub M a b) b
  | .addZero => add M a (zero M)
  | .mulOne => mul M a (one M)
  | .divOne => div M a (one M)
  | .divMulRcp => div M a b
  | .mulRcpCancel => mul M a (div M (one M) a)

def rightValue {α : Type} (M : Algebra α) (a b c : α) : Atom → α
  | .addCommute => add M b a
  | .addAssociate => add M a (add M b c)
  | .mulCommute => mul M b a
  | .mulAssociate => mul M a (mul M b c)
  | .mulDistribute => add M (mul M a b) (mul M a c)
  | .cancel | .addZero | .mulOne | .divOne => a
  | .divMulRcp => mul M a (div M (one M) b)
  | .mulRcpCancel => one M

def Inputs {α : Type} (D : Domain α) (a b c : α) : Atom → Prop
  | .addCommute | .mulCommute | .cancel => D .finite a ∧ D .finite b
  | .addAssociate | .mulAssociate | .mulDistribute => D .finite a ∧ D .finite b ∧ D .finite c
  | .addZero | .mulOne | .divOne => D .finite a
  | .divMulRcp => D .finite a ∧ D .finite b ∧ D .nonzero b
  | .mulRcpCancel => D .finite a ∧ D .nonzero a

set_option maxHeartbeats 1600000 in
/-- Execute the actual scalar fragments before interpreting their admitted law.
The arbitrary surrounding state contributes no numerical premise. -/
theorem apply_atom {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (atom : Atom) (selected : atom.Available) (a b c : α) (hg : Inputs D a b c atom) :
    leftValue M a b c atom = rightValue M a b c atom := by
  let t := ((s.setReg "a" .real [] (fun _ => a)).setReg "b" .real []
    (fun _ => b)).setReg "c" .real [] (fun _ => c)
  have hd : ScalarDomain D (guards atom) t := by
    intro g hg'
    cases atom <;> simp [guards] at hg'
    all_goals
      simp only [Inputs] at hg
      rcases hg' with rfl | rfl | rfl
      all_goals simp only [t, State.setReg]; simp_all
  have h := hM (lhs atom) (rhs atom) (admitted R atom selected) t hd hd
  cases atom <;>
    simp only [lhs, rhs, lhsCode, rhsCode, fragment, Reciprocal.lhs, Reciprocal.rhs,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ref, plus, minus, times, divide,
      Reciprocal.div, Reciprocal.inv, Reciprocal.mul, Reciprocal.ref,
      ComputeDType.eraseDType, numeric, State.setReg_same, t] at h
  all_goals
    simp [State.setReg, leftValue, rightValue, add, sub, mul, div, zero, one] at h ⊢
    exact congrFun h PUnit.unit

end VeriTile.Triton.FP.ScalarArithmetic
