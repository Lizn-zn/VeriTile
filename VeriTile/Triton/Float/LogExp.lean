import VeriTile.Triton.Float.LogAdmission
import VeriTile.Triton.Float.GuardedIO
import VeriTile.Meta.StatementAudit

/-!
The fp32 log/exp candidates below exist independently of experiment outcomes.
Each candidate specifies its exact intrinsics, operand domain and two scalar
fragments. Defining a candidate supplies no numerical equality.

The generated LogAdmission table selects which candidates can be used as
assumptions. The current log report selects only the masked LOG-EXP-EXPM1
relation; the other candidates remain defined without being enabled. Refreshing
the report changes availability, not the candidate definitions. Kernel
implementations and their separate specifications are in bench/examples/LogExp/.
-/

noncomputable section
namespace VeriTile.Triton.FP.LogExp
open Structural Guarded
open scoped VeriTile.Spec

/-- Both fragments require the same finite input register `a`. -/
def guards : List OperandGuard := [⟨"a", .finite⟩]

/-- Same comparison as `tl.abs(a) <= 0.5`; no near-zero input restriction. -/
def nearZero (a : Op .real []) : Op .bool [] :=
  .le .real .nil
    (.where (.lt .real .nil a (.const 0)) (.sub .real .nil (.const 0) a) a)
    (.const (1 / 2))

/-- Both arms of `where` are evaluated. Mask the unused argument to zero,
as in the measured piecewise expression, before either libdevice call. -/
def expression (a : Op .real []) : Op .real [] :=
  let near := nearZero a
  let small_a := .where near a (.const 0)
  let other_a := .where near (.const 0) a
  let small := .libdeviceLog1p (.libdeviceExpm1 small_a)
  let other := .libdeviceLog (.libdeviceExp other_a)
  .where near small other

/-- Read one scalar register. `[]` is the scalar tile shape. -/
def input : Op .real [] := .ref .real [] "a"

/-- Write `out` with explicit fp32 computation. The AST's `.real` tag is the
floating carrier; `.compute (.alg .fp32 ...)` selects the numerical precision.
All arithmetic, comparisons and libdevice calls execute at that precision. -/
def assignOutput (e : Op .real []) : List ComputeStmt :=
  [.assign .real [] "out" (.compute (.alg .fp32 e))]

/-- The original scalar computation, with the input condition attached. -/
def piecewiseLogExp : GuardedFragment := ⟨guards, assignOutput (expression input)⟩

/-- The replacement `out = a`, with the same input condition and precision. -/
def identity : GuardedFragment := ⟨guards, assignOutput input⟩

/-- Candidate names describe exact implementations, including rejected or
inconclusive relations. All fragments in this catalog compute in fp32. -/
inductive Atom where
  | log_mul | log_mul_libdevice
  | log_exp | log_exp_libdevice | log_exp_full_libdevice | log_exp_expm1
  deriving DecidableEq, Repr

def candidates : List Atom := [.log_mul, .log_mul_libdevice, .log_exp,
  .log_exp_libdevice, .log_exp_full_libdevice, .log_exp_expm1]

def Atom.ruleID : Atom → String
  | .log_mul => "LOG-MUL"
  | .log_mul_libdevice => "LOG-MUL-LIBDEVICE"
  | .log_exp => "LOG-EXP"
  | .log_exp_libdevice => "LOG-EXP-LIBDEVICE"
  | .log_exp_full_libdevice => "LOG-EXP-FULL-LIBDEVICE"
  | .log_exp_expm1 => "LOG-EXP-EXPM1"

/-- A log-product rewrite requires positive finite operands on both sides. -/
def productGuards : List OperandGuard :=
  [⟨"a", .finite⟩, ⟨"b", .finite⟩, ⟨"a", .positive⟩, ⟨"b", .positive⟩]

def Atom.guards : Atom → List OperandGuard
  | .log_mul | .log_mul_libdevice => productGuards
  | .log_exp | .log_exp_libdevice | .log_exp_full_libdevice | .log_exp_expm1 =>
      VeriTile.Triton.FP.LogExp.guards

def secondInput : Op .real [] := .ref .real [] "b"

/-- Left-hand scalar expressions, before any experimental admission:
* log_mul: tl.log(a * b)
* log_mul_libdevice: libdevice.log(a * b)
* log_exp: tl.log(tl.exp(a))
* log_exp_libdevice: tl.log(libdevice.exp(a))
* log_exp_full_libdevice: libdevice.log(libdevice.exp(a))
* log_exp_expm1: the masked piecewise expression defined above.
The tl.exp variant remains a candidate; its identity cannot be substituted
for libdevice.exp when selecting a report. -/
def Atom.lhs (a : Atom) : GuardedFragment := ⟨a.guards, assignOutput (match a with
  | .log_mul => .log (.mul .real .nil input secondInput)
  | .log_mul_libdevice => .libdeviceLog (.mul .real .nil input secondInput)
  | .log_exp => .log (.exp input)
  | .log_exp_libdevice => .log (.libdeviceExp input)
  | .log_exp_full_libdevice => .libdeviceLog (.libdeviceExp input)
  | .log_exp_expm1 => expression input)⟩

/-- Product rules propose the sum of the corresponding logs; cancellation
rules propose the original input. Operand guards are retained on both sides. -/
def Atom.rhs (a : Atom) : GuardedFragment := ⟨a.guards, assignOutput (match a with
  | .log_mul => .add .real .nil (.log input) (.log secondInput)
  | .log_mul_libdevice => .add .real .nil (.libdeviceLog input) (.libdeviceLog secondInput)
  | .log_exp | .log_exp_libdevice | .log_exp_full_libdevice | .log_exp_expm1 => input)⟩

/-- Match an accepted report to the candidate's exact fp32 profile and domain.
Experimental shape and input distribution select the row; they do not become
extra parameters of the subsequent scalar derivation. -/
def Atom.matches (a : Atom) (row : ReportedScalarRule) : Bool :=
  decide (row.report.ruleID = a.ruleID ∧ row.report.input = "fp32" ∧
    row.report.compute = "fp32" ∧ row.report.accumulator = "fp32" ∧
    row.report.output = "fp32" ∧ row.guards = a.guards)

/-- Only the generated accepted table can activate a candidate. -/
def Atom.report? (a : Atom) : Option ReportedScalarRule :=
  LogAdmission.all.find? a.matches

/-- Availability is a report-selection condition, not the proposed equality. -/
abbrev Atom.Available (a : Atom) : Prop := a.report?.isSome = true

/-- Bind the selected report to the already-defined candidate fragments. -/
def Atom.entry (a : Atom) (h : a.Available) : Spec.RuleEntry GuardedFragment :=
  (a.report?.get h).report.bind [a.lhs] [a.rhs]

def Atom.entry? (a : Atom) : Option (Spec.RuleEntry GuardedFragment) :=
  a.report?.map fun row => row.report.bind [a.lhs] [a.rhs]

/-- External validation is required only for candidates selected by the report. -/
structure Rules where
  validated : ∀ (a : Atom) (h : a.Available),
    Spec.EvidenceValidated (a.entry h).rule (a.entry h).evidence

def Rules.assumptions (_ : Rules) : Spec.Assumptions GuardedFragment :=
  candidates.filterMap Atom.entry?

instance : CoeOut Rules (Spec.Assumptions GuardedFragment) := ⟨Rules.assumptions⟩

/-- Every candidate has the same reusable derivation; a report must select it
before this lemma can be applied. No candidate is asserted unconditionally. -/
theorem derive (R : Rules) (a : Atom) (h : a.Available) :
    Spec.Derivation R.assumptions [a.lhs] [a.rhs] := by
  apply Spec.Derivation.atom (a.entry h)
  · apply List.mem_filterMap.mpr
    refine ⟨a, ?_, ?_⟩
    · cases a <;> simp [candidates]
    · have available : a.report?.isSome = true := h
      cases hr : a.report? with
      | none => simp [hr] at available
      | some row => simp [Atom.entry?, Atom.entry, hr]
  · exact (a.report?.get h).report.admit _ _ (R.validated a h)

/-- Use any admitted candidate with the ordinary FP-equivalence notation. -/
theorem rewrite (R : Rules) (a : Atom) (h : a.Available) : [a.lhs] ≡[R] [a.rhs] :=
  Spec.FloatingPoint.ofDerivation rfl trivial (derive R a h)

/-- The piecewise kernel example uses this one selected candidate. -/
def entry (h : Atom.log_exp_expm1.Available) := Atom.entry .log_exp_expm1 h

theorem admitted (R : Rules) (h : Atom.log_exp_expm1.Available) :
    Spec.Derivation R.assumptions [piecewiseLogExp] [identity] :=
  derive R .log_exp_expm1 h

theorem scalar_equiv (R : Rules) (h : Atom.log_exp_expm1.Available) :
    [piecewiseLogExp] ≡[R] [identity] :=
  rewrite R .log_exp_expm1 h

/- Scalar execution used to instantiate the admitted relation. -/

/-- Evaluate the original piecewise computation using the FP operations in M. -/
def value {α : Type} (M : Algebra α) (lt le : α → α → Bool) (a : α) : α :=
  let z := M.literal (some .fp32) .real 0
  let half := M.literal (some .fp32) .real (1 / 2)
  let absolute := if lt a z then M.binary (some .fp32) .real .sub z a else a
  let near := le absolute half
  let small := M.unary (some .fp32) .libdeviceLog1p
    (M.unary (some .fp32) .libdeviceExpm1 (if near then a else z))
  let other := M.unary (some .fp32) .libdeviceLog
    (M.unary (some .fp32) .libdeviceExp (if near then z else a))
  if near then small else other

set_option maxHeartbeats 1600000 in
/-- Instantiate the admitted scalar rule only after executing its comparisons
and both masked branches. Unsupported comparisons cannot discharge this law. -/
theorem apply_rule {α : Type} [Inhabited α] (R : Rules)
    (selected : Atom.log_exp_expm1.Available)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (lt le : α → α → Bool)
    (hlt : M.compareLt (some .fp32) .real = some lt)
    (hle : M.compareLe (some .fp32) .real = some le)
    (a : α) (ha : D .finite a) : value M lt le a = a := by
  let t := s.setReg "a" .real [] (fun _ => a)
  have hg : ScalarDomain D guards t := by
    intro g hg
    simp only [guards, List.mem_singleton] at hg
    subst g
    exact ⟨fun _ => a, by simp [t], ha⟩
  have h := hM piecewiseLogExp identity (admitted R selected) t hg hg
  simp only [piecewiseLogExp, identity, assignOutput, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, expression, nearZero, input,
    ComputeDType.eraseDType, numeric, numericLt, numericLe, hlt, hle,
    State.setReg_same, t] at h
  simp [State.setReg, bop, value] at h ⊢
  exact congrFun h PUnit.unit

end VeriTile.Triton.FP.LogExp
