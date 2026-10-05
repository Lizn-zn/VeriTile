import VeriTile.Triton.Float.LogAdmission
import VeriTile.Triton.Float.GuardedIO
import VeriTile.Meta.StatementAudit

/-!
The fp32 log/exp candidates below exist independently of experiment outcomes.
Each candidate specifies its exact intrinsics, operand domain and two scalar
fragments. Defining a candidate supplies no numerical equality.

The generated LogAdmission table selects which candidates can be used as
assumptions. LOG-EXP-GUARDED keeps the original log(exp(a)) reference and changes
only the candidate to a guarded identity with the original fallback. Refreshing
the report changes availability, not the candidate definitions. Kernel
implementations and their separate specifications are in bench/examples/LogExp/.
-/

noncomputable section
namespace VeriTile.Triton.FP.LogExp
open Structural Guarded
open scoped VeriTile.Spec

/-- Candidate names describe exact implementations, including rejected or
inconclusive relations. All fragments in this catalog compute in fp32. -/
inductive Atom where
  | log_mul | log_mul_libdevice | log_mul_split
  | log_exp | log_exp_libdevice | log_exp_full_libdevice | log_exp_guarded
  deriving DecidableEq, Repr

/-- Candidate rewrites and their experiment identifiers. Every floating
operation below uses fp32. An arrow describes a proposed rewrite; using it
requires two-gates admission for that exact candidate. -/
def Atom.ruleID : Atom → String
  -- tl.log(a * b) → tl.log(a) + tl.log(b); finite a,b with a > 0 and b > 0.
  | .log_mul => "LOG-MUL"
  -- libdevice.log(a * b) → libdevice.log(a) + libdevice.log(b); same positive domain.
  | .log_mul_libdevice => "LOG-MUL-LIBDEVICE"
  -- libdevice.log(p) → if 0.5 ≤ p ≤ 2 then libdevice.log(p)
  -- else libdevice.log(a) + libdevice.log(b), where p = fp32(a*b).
  -- Finite positive a,b; the interval chooses an implementation, not a domain.
  -- Both branches evaluate with unused log arguments masked to one.
  -- Keep the frozen experiment identifier when naming the Lean candidate.
  | .log_mul_split => "LOG-MUL-GUARDED"
  -- tl.log(tl.exp(a)) → a; any finite a, including negative values.
  | .log_exp => "LOG-EXP"
  -- tl.log(libdevice.exp(a)) → a; any finite a. Only exp uses libdevice.
  | .log_exp_libdevice => "LOG-EXP-LIBDEVICE"
  -- libdevice.log(libdevice.exp(a)) → a; any finite a. Both calls use libdevice.
  | .log_exp_full_libdevice => "LOG-EXP-FULL-LIBDEVICE"
  -- libdevice.log(libdevice.exp(a)) →
  -- (if 0.5 < |a| ≤ 80 then a else libdevice.log(libdevice.exp(a))).
  -- The fallback preserves near-zero rounding and extreme-input behavior.
  | .log_exp_guarded => "LOG-EXP-GUARDED"

def candidates : List Atom := [.log_mul, .log_mul_libdevice, .log_mul_split, .log_exp,
  .log_exp_libdevice, .log_exp_full_libdevice, .log_exp_guarded]

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
def expression (a : Op .real []) : Op .real [] :=
  let simplify := useIdentity a
  let fallback_a := .where simplify (.const 0) a
  let fallback := .libdeviceLog (.libdeviceExp fallback_a)
  .where simplify a fallback

/-- Test the rounded fp32 product, including both endpoints. -/
def keepProduct (p : Op .real []) : Op .bool [] :=
  .boolAnd .nil (.ge .real .nil p (.const (1 / 2))) (.le .real .nil p (.const 2))

/-- The measured product-log candidate. Every log is libdevice.log. Keep the
original product log near one; elsewhere split the two logs. Mask arguments
before evaluating either arm, exactly as in the Triton experiment. -/
def splitProduct (a b : Op .real []) : Op .real [] :=
  let p := .mul .real .nil a b
  let keep := keepProduct p
  let direct := .libdeviceLog (.where keep p (.const 1))
  let split := .add .real .nil
    (.libdeviceLog (.where keep (.const 1) a))
    (.libdeviceLog (.where keep (.const 1) b))
  .where keep direct split

/-- Read one scalar register. `[]` is the scalar tile shape. -/
def input : Op .real [] := .ref .real [] "a"

/-- Write `out` with explicit fp32 computation. The AST's `.real` tag is the
floating carrier; `.compute (.alg .fp32 ...)` selects the numerical precision.
All arithmetic, comparisons and libdevice calls execute at that precision. -/
def assignOutput (e : Op .real []) : List ComputeStmt :=
  [.assign .real [] "out" (.compute (.alg .fp32 e))]

/-- The fixed reference computation, with the finite-input condition attached. -/
def originalLogExp : GuardedFragment :=
  ⟨guards, assignOutput (.libdeviceLog (.libdeviceExp input))⟩

/-- The candidate eliminates the composition on its guarded fast path. -/
def piecewiseLogExp : GuardedFragment := ⟨guards, assignOutput (expression input)⟩

/-- A log-product rewrite requires positive finite operands on both sides. -/
def productGuards : List OperandGuard :=
  [⟨"a", .finite⟩, ⟨"b", .finite⟩, ⟨"a", .positive⟩, ⟨"b", .positive⟩]

def Atom.guards : Atom → List OperandGuard
  | .log_mul | .log_mul_libdevice | .log_mul_split => productGuards
  | .log_exp | .log_exp_libdevice | .log_exp_full_libdevice | .log_exp_guarded =>
      VeriTile.Triton.FP.LogExp.guards

def secondInput : Op .real [] := .ref .real [] "b"

/-- Left-hand fp32 fragments for the candidate rewrites listed above.
The tl.exp variant remains a candidate; its identity cannot be substituted
for libdevice.exp when selecting a report. -/
def Atom.lhs (a : Atom) : GuardedFragment := ⟨a.guards, assignOutput (match a with
  | .log_mul => .log (.mul .real .nil input secondInput)
  | .log_mul_libdevice | .log_mul_split => .libdeviceLog (.mul .real .nil input secondInput)
  | .log_exp => .log (.exp input)
  | .log_exp_libdevice => .log (.libdeviceExp input)
  | .log_exp_full_libdevice | .log_exp_guarded => .libdeviceLog (.libdeviceExp input))⟩

/-- Product rules propose split logs; cancellation rules propose the input.
The guarded probe proposes conditional elimination with the same input domain. -/
def Atom.rhs (a : Atom) : GuardedFragment := ⟨a.guards, assignOutput (match a with
  | .log_mul => .add .real .nil (.log input) (.log secondInput)
  | .log_mul_libdevice => .add .real .nil (.libdeviceLog input) (.libdeviceLog secondInput)
  | .log_mul_split => splitProduct input secondInput
  | .log_exp | .log_exp_libdevice | .log_exp_full_libdevice => input
  | .log_exp_guarded => expression input)⟩

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
def entry (h : Atom.log_exp_guarded.Available) := Atom.entry .log_exp_guarded h

theorem admitted (R : Rules) (h : Atom.log_exp_guarded.Available) :
    Spec.Derivation R.assumptions [originalLogExp] [piecewiseLogExp] :=
  derive R .log_exp_guarded h

theorem scalar_equiv (R : Rules) (h : Atom.log_exp_guarded.Available) :
    [originalLogExp] ≡[R] [piecewiseLogExp] :=
  rewrite R .log_exp_guarded h

/- Scalar execution used to instantiate the admitted relation. -/

/-- Evaluate the unchanged reference using the FP operations in M. -/
def referenceValue {α : Type} (M : Algebra α) (a : α) : α :=
  M.unary (some .fp32) .libdeviceLog (M.unary (some .fp32) .libdeviceExp a)

/-- Evaluate the piecewise candidate using the FP operations in M. -/
def value {α : Type} (M : Algebra α) (lt le : α → α → Bool) (a : α) : α :=
  let z := M.literal (some .fp32) .real 0
  let half := M.literal (some .fp32) .real (1 / 2)
  let absolute := if lt a z then M.binary (some .fp32) .real .sub z a else a
  let upper := M.literal (some .fp32) .real 80
  let simplify := lt half absolute && le absolute upper
  let fallback := M.unary (some .fp32) .libdeviceLog
    (M.unary (some .fp32) .libdeviceExp (if simplify then z else a))
  if simplify then a else fallback

set_option maxHeartbeats 1600000 in
/-- Instantiate the admitted scalar rule only after executing its comparisons
and both masked branches. Unsupported comparisons cannot discharge this law. -/
theorem apply_rule {α : Type} [Inhabited α] (R : Rules)
    (selected : Atom.log_exp_guarded.Available)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (lt le : α → α → Bool)
    (hlt : M.compareLt (some .fp32) .real = some lt)
    (hle : M.compareLe (some .fp32) .real = some le)
    (a : α) (ha : D .finite a) : referenceValue M a = value M lt le a := by
  let t := s.setReg "a" .real [] (fun _ => a)
  have hg : ScalarDomain D guards t := by
    intro g hg
    simp only [guards, List.mem_singleton] at hg
    subst g
    exact ⟨fun _ => a, by simp [t], ha⟩
  have h := hM originalLogExp piecewiseLogExp (admitted R selected) t hg hg
  simp only [originalLogExp, piecewiseLogExp, assignOutput, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, expression, useIdentity, absolute, input,
    ComputeDType.eraseDType, numeric, numericLt, numericLe, hlt, hle,
    State.setReg_same, t] at h
  simp [State.setReg, bop, referenceValue, value] at h ⊢
  exact congrFun h PUnit.unit

/-- Interpret the conditional product expression with exact fp32 operation
labels. The masked, unused log arguments remain part of this definition. -/
def splitProductValue {α : Type} (M : Algebra α) (le : α → α → Bool) (a b : α) : α :=
  let p := M.binary (some .fp32) .real .mul a b
  let one := M.literal (some .fp32) .real 1
  let keep := le (M.literal (some .fp32) .real (1 / 2)) p &&
    le p (M.literal (some .fp32) .real 2)
  let direct := M.unary (some .fp32) .libdeviceLog (if keep then p else one)
  let split := M.binary (some .fp32) .real .add
    (M.unary (some .fp32) .libdeviceLog (if keep then one else a))
    (M.unary (some .fp32) .libdeviceLog (if keep then one else b))
  if keep then direct else split

set_option maxHeartbeats 1600000 in
/-- Apply the selected scalar relation to actual positive finite operands.
The conclusion keeps the product-dependent branch; unconditional splitting
does not follow. Successful fp32 comparisons are required explicitly. -/
theorem apply_log_mul_split {α : Type} [Inhabited α] (R : Rules)
    (selected : Atom.log_mul_split.Available)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (le : α → α → Bool)
    (hle : M.compareLe (some .fp32) .real = some le)
    (a b : α) (ha : D .finite a) (hb : D .finite b)
    (hpa : D .positive a) (hpb : D .positive b) :
    M.unary (some .fp32) .libdeviceLog (M.binary (some .fp32) .real .mul a b) =
      splitProductValue M le a b := by
  let t := (s.setReg "a" .real [] (fun _ => a)).setReg "b" .real [] (fun _ => b)
  have hg : ScalarDomain D productGuards t := by
    intro g hg
    simp [productGuards] at hg
    rcases hg with rfl | rfl | rfl | rfl
    · exact ⟨fun _ => a, by simp [t, State.setReg], ha⟩
    · exact ⟨fun _ => b, by simp [t, State.setReg], hb⟩
    · exact ⟨fun _ => a, by simp [t, State.setReg], hpa⟩
    · exact ⟨fun _ => b, by simp [t, State.setReg], hpb⟩
  have h := hM _ _ (derive R .log_mul_split selected) t hg hg
  simp only [Atom.lhs, Atom.rhs, assignOutput, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, splitProduct, keepProduct, input, secondInput,
    ComputeDType.eraseDType, numeric, numericLe, hle, t] at h
  simp [State.setReg, bop, splitProductValue] at h ⊢
  exact congrFun h PUnit.unit

end VeriTile.Triton.FP.LogExp
