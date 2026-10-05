import VeriTile.Triton.Float.LogAdmission
import VeriTile.Triton.Float.GuardedIO
import VeriTile.Meta.StatementAudit

/-!
The fp32 log/exp candidates below exist independently of experiment outcomes.
Each candidate specifies its exact intrinsics, operand domain and two scalar
fragments. Defining a candidate supplies no numerical equality.

The generated LogAdmission table selects independently tested implementations.
The log and exp backends have distinct fragments and report IDs.
All four guarded log-exp versions keep their original log(exp(a)) reference and
change only the candidate to a guarded identity with the original fallback. Refreshing
the report changes availability, not the candidate definitions. Kernel
implementations and their separate specifications are in bench/examples/LogExp/.
-/

noncomputable section
namespace VeriTile.Triton.FP.LogExp
open Structural Guarded
open scoped VeriTile.Spec

/-- Select the actual Triton implementation, independently of the rewrite.
In `log_exp`, the log and exp implementations are selected separately. -/
inductive Backend where
  | tl | libdevice
  deriving DecidableEq, Repr

/-- Proposed fp32 relations. Splitting/cancellation retain the measured
fallback; the shorter `log_mul`/`log_exp` candidates are unconditional. -/
inductive Atom where
  | log_mul (log : Backend)
  | log_mul_split (log : Backend)
  | log_exp (log exp : Backend)
  | log_exp_cancel (log : Backend) (exp : Backend := .libdevice)
  deriving DecidableEq, Repr

/-- Candidate catalog, including relations not admitted by the current report.
The implementation parameters select exact intrinsics, never an API equality.
Experiment identifiers stay frozen even when the public Lean names improve. -/
def Atom.ruleID : Atom → String
  -- log(a*b) → log(a)+log(b), for finite positive a,b.
  | .log_mul .tl => "LOG-MUL"
  | .log_mul .libdevice => "LOG-MUL-LIBDEVICE"
  -- log(p) → if 0.5 ≤ p ≤ 2 then log(p) else log(a)+log(b), p=fp32(a*b).
  -- Same operand domain; inactive log arguments are masked to one.
  | .log_mul_split .tl => "LOG-MUL-GUARDED-INTRINSIC"
  | .log_mul_split .libdevice => "LOG-MUL-GUARDED"
  -- log(exp(a)) → a, for finite a, including negative values.
  | .log_exp .tl .tl => "LOG-EXP"
  | .log_exp .libdevice .tl => "LOG-EXP-LOG-LIBDEVICE"
  | .log_exp .tl .libdevice => "LOG-EXP-LIBDEVICE"
  | .log_exp .libdevice .libdevice => "LOG-EXP-FULL-LIBDEVICE"
  -- log(exp(a)) → if 0.5 < |a| ≤ 80 then a else the original.
  -- Finite a; the unused exp argument is masked to zero.
  | .log_exp_cancel .tl .libdevice => "LOG-EXP-GUARDED-INTRINSIC"
  | .log_exp_cancel .libdevice .libdevice => "LOG-EXP-GUARDED"
  | .log_exp_cancel .tl .tl => "LOG-EXP-GUARDED-FULL-INTRINSIC"
  | .log_exp_cancel .libdevice .tl => "LOG-EXP-GUARDED-EXP-INTRINSIC"

def candidates : List Atom := [.log_mul .tl, .log_mul .libdevice,
  .log_mul_split .tl, .log_mul_split .libdevice,
  .log_exp .tl .tl, .log_exp .libdevice .tl,
  .log_exp .tl .libdevice, .log_exp .libdevice .libdevice,
  .log_exp_cancel .tl, .log_exp_cancel .libdevice,
  .log_exp_cancel .tl .tl, .log_exp_cancel .libdevice .tl]

/-- Exact operation labels used by the numerical interpreter. -/
def Backend.logOp : Backend → Unary
  | .tl => .log
  | .libdevice => .libdeviceLog

def Backend.expOp : Backend → Unary
  | .tl => .exp
  | .libdevice => .libdeviceExp

def Backend.log : Backend → Op .real [] → Op .real []
  | .tl => .log
  | .libdevice => .libdeviceLog

def Backend.exp : Backend → Op .real [] → Op .real []
  | .tl => .exp
  | .libdevice => .libdeviceExp

/-- Both fragments require the same finite input register `a`. -/
def guards : List OperandGuard := [⟨"a", .finite⟩]

/-- FP absolute value, preserving the source comparison and subtraction. -/
def absolute (a : Op .real []) : Op .real [] :=
  .where (.lt .real .nil a (.const 0)) (.sub .real .nil (.const 0) a) a

/-- Select elimination only away from zero and within the normal exp range. -/
def useIdentity (a : Op .real []) : Op .bool [] :=
  .boolAnd .nil (.lt .real .nil (.const (1 / 2)) (absolute a))
    (.le .real .nil (absolute a) (.const 80))

/-- Scalar semantics of the candidate. The GPU additionally skips both calls
for whole tiles selecting the identity; mixed tiles mask unused arguments. -/
def expression (backend : Backend) (a : Op .real [])
    (exp : Backend := .libdevice) : Op .real [] :=
  let simplify := useIdentity a
  let fallback_a := .where simplify (.const 0) a
  let fallback := backend.log (exp.exp fallback_a)
  .where simplify a fallback

/-- Test the rounded fp32 product, including both endpoints. -/
def keepProduct (p : Op .real []) : Op .bool [] :=
  .boolAnd .nil (.ge .real .nil p (.const (1 / 2))) (.le .real .nil p (.const 2))

/-- Keep the rounded product log near one, otherwise split the two logs.
Both arms evaluate with unused arguments masked to one. -/
def splitProduct (backend : Backend) (a b : Op .real []) : Op .real [] :=
  let p := .mul .real .nil a b
  let keep := keepProduct p
  let direct := backend.log (.where keep p (.const 1))
  let split := .add .real .nil
    (backend.log (.where keep (.const 1) a))
    (backend.log (.where keep (.const 1) b))
  .where keep direct split

/-- Read one scalar register. `[]` is the scalar tile shape. -/
def input : Op .real [] := .ref .real [] "a"

/-- Write `out` with explicit fp32 computation. The AST's `.real` tag is the
floating carrier; `.compute (.alg .fp32 ...)` selects the numerical precision.
All arithmetic, comparisons and transcendental calls execute at that precision. -/
def assignOutput (e : Op .real []) : List ComputeStmt :=
  [.assign .real [] "out" (.compute (.alg .fp32 e))]

/-- The fixed reference computation, with the finite-input condition attached. -/
def originalLogExp : GuardedFragment :=
  ⟨guards, assignOutput (.libdeviceLog (.libdeviceExp input))⟩

/-- The candidate eliminates the composition on its guarded fast path. -/
def piecewiseLogExp : GuardedFragment := ⟨guards, assignOutput (expression .libdevice input)⟩

/-- A log-product rewrite requires positive finite operands on both sides. -/
def productGuards : List OperandGuard :=
  [⟨"a", .finite⟩, ⟨"b", .finite⟩, ⟨"a", .positive⟩, ⟨"b", .positive⟩]

def Atom.guards : Atom → List OperandGuard
  | .log_mul _ | .log_mul_split _ => productGuards
  | .log_exp _ _ | .log_exp_cancel _ _ => VeriTile.Triton.FP.LogExp.guards

def secondInput : Op .real [] := .ref .real [] "b"

/-- The original expression with exact log/exp implementations. -/
def Atom.lhs (a : Atom) : GuardedFragment := ⟨a.guards, assignOutput (match a with
  | .log_mul log | .log_mul_split log => log.log (.mul .real .nil input secondInput)
  | .log_exp log exp => log.log (exp.exp input)
  | .log_exp_cancel log exp => log.log (exp.exp input))⟩

/-- The proposed expression. Conditional rewrites keep their fallbacks;
unconditional rewrites require their own independent admission. -/
def Atom.rhs (a : Atom) : GuardedFragment := ⟨a.guards, assignOutput (match a with
  | .log_mul log => .add .real .nil (log.log input) (log.log secondInput)
  | .log_mul_split log => splitProduct log input secondInput
  | .log_exp _ _ => input
  | .log_exp_cancel log exp => expression log input exp)⟩

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
    · rcases a with ⟨_ | _⟩ | ⟨_ | _⟩ | ⟨_ | _, _ | _⟩ | ⟨_ | _, _ | _⟩ <;> simp [candidates]
    · have available : a.report?.isSome = true := h
      cases hr : a.report? with
      | none => simp [hr] at available
      | some row => simp [Atom.entry?, Atom.entry, hr]
  · exact (a.report?.get h).report.admit _ _ (R.validated a h)

/-- Use any admitted candidate with the ordinary FP-equivalence notation. -/
theorem rewrite (R : Rules) (a : Atom) (h : a.Available) : [a.lhs] ≡[R] [a.rhs] :=
  Spec.FloatingPoint.ofDerivation rfl trivial (derive R a h)

/-- The piecewise kernel example uses this one selected candidate. -/
def entry (h : (Atom.log_exp_cancel .libdevice).Available) := Atom.entry (.log_exp_cancel .libdevice) h

theorem admitted (R : Rules) (h : (Atom.log_exp_cancel .libdevice).Available) :
    Spec.Derivation R.assumptions [originalLogExp] [piecewiseLogExp] :=
  derive R (.log_exp_cancel .libdevice) h

theorem scalar_equiv (R : Rules) (h : (Atom.log_exp_cancel .libdevice).Available) :
    [originalLogExp] ≡[R] [piecewiseLogExp] :=
  rewrite R (.log_exp_cancel .libdevice) h

/- Scalar execution used to instantiate the admitted relation. -/

/-- Evaluate the unchanged reference using the FP operations in M. -/
def referenceValue (backend : Backend) {α : Type} (M : Algebra α) (a : α) : α :=
  M.unary (some .fp32) backend.logOp (M.unary (some .fp32) .libdeviceExp a)

/-- Evaluate the piecewise candidate using the FP operations in M. -/
def value (backend : Backend) {α : Type} (M : Algebra α) (lt le : α → α → Bool) (a : α) : α :=
  let z := M.literal (some .fp32) .real 0
  let half := M.literal (some .fp32) .real (1 / 2)
  let absolute := if lt a z then M.binary (some .fp32) .real .sub z a else a
  let upper := M.literal (some .fp32) .real 80
  let simplify := lt half absolute && le absolute upper
  let fallback := M.unary (some .fp32) backend.logOp
    (M.unary (some .fp32) .libdeviceExp (if simplify then z else a))
  if simplify then a else fallback

set_option maxHeartbeats 1600000 in
/-- Instantiate the admitted scalar rule only after executing its comparisons
and both masked branches. Unsupported comparisons cannot discharge this law. -/
theorem apply_log_exp_cancel (backend : Backend) {α : Type} [Inhabited α] (R : Rules)
    (selected : (Atom.log_exp_cancel backend).Available)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (lt le : α → α → Bool)
    (hlt : M.compareLt (some .fp32) .real = some lt)
    (hle : M.compareLe (some .fp32) .real = some le)
    (a : α) (ha : D .finite a) : referenceValue backend M a = value backend M lt le a := by
  let t := s.setReg "a" .real [] (fun _ => a)
  have hg : ScalarDomain D guards t := by
    intro g hg
    simp only [guards, List.mem_singleton] at hg
    subst g
    exact ⟨fun _ => a, by simp [t], ha⟩
  have h := hM _ _ (derive R (.log_exp_cancel backend) selected) t hg hg
  cases backend <;> simp only [Atom.lhs, Atom.rhs, assignOutput, run, step, evalExpr,
    evalComputeOp, expression, Backend.log, Backend.exp, evalOp_unfold, useIdentity, absolute, input,
    ComputeDType.eraseDType, numeric, numericLt, numericLe, hlt, hle,
    State.setReg_same, t] at h
  all_goals
    simp [State.setReg, bop, referenceValue, value, Backend.logOp] at h ⊢
    exact congrFun h PUnit.unit

/-- Interpret the conditional product expression with exact fp32 operation
labels. The masked, unused log arguments remain part of this definition. -/
def splitProductValue (backend : Backend) {α : Type} (M : Algebra α) (le : α → α → Bool) (a b : α) : α :=
  let p := M.binary (some .fp32) .real .mul a b
  let one := M.literal (some .fp32) .real 1
  let keep := le (M.literal (some .fp32) .real (1 / 2)) p &&
    le p (M.literal (some .fp32) .real 2)
  let direct := M.unary (some .fp32) backend.logOp (if keep then p else one)
  let split := M.binary (some .fp32) .real .add
    (M.unary (some .fp32) backend.logOp (if keep then one else a))
    (M.unary (some .fp32) backend.logOp (if keep then one else b))
  if keep then direct else split

set_option maxHeartbeats 1600000 in
/-- Apply the selected scalar relation to actual positive finite operands.
The conclusion keeps the product-dependent branch; unconditional splitting
does not follow. Successful fp32 comparisons are required explicitly. -/
theorem apply_log_mul_split (backend : Backend) {α : Type} [Inhabited α] (R : Rules)
    (selected : (Atom.log_mul_split backend).Available)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (le : α → α → Bool)
    (hle : M.compareLe (some .fp32) .real = some le)
    (a b : α) (ha : D .finite a) (hb : D .finite b)
    (hpa : D .positive a) (hpb : D .positive b) :
    M.unary (some .fp32) backend.logOp (M.binary (some .fp32) .real .mul a b) =
      splitProductValue backend M le a b := by
  let t := (s.setReg "a" .real [] (fun _ => a)).setReg "b" .real [] (fun _ => b)
  have hg : ScalarDomain D productGuards t := by
    intro g hg
    simp [productGuards] at hg
    rcases hg with rfl | rfl | rfl | rfl
    · exact ⟨fun _ => a, by simp [t, State.setReg], ha⟩
    · exact ⟨fun _ => b, by simp [t, State.setReg], hb⟩
    · exact ⟨fun _ => a, by simp [t, State.setReg], hpa⟩
    · exact ⟨fun _ => b, by simp [t, State.setReg], hpb⟩
  have h := hM _ _ (derive R (.log_mul_split backend) selected) t hg hg
  cases backend <;> simp only [Atom.lhs, Atom.rhs, assignOutput, run, step, evalExpr,
    evalComputeOp, splitProduct, Backend.log, evalOp_unfold, keepProduct, input, secondInput,
    ComputeDType.eraseDType, numeric, numericLe, hle, t] at h
  all_goals
    simp [State.setReg, bop, splitProductValue, Backend.logOp] at h ⊢
    exact congrFun h PUnit.unit

end VeriTile.Triton.FP.LogExp
