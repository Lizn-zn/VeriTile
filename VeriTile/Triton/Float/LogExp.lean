import VeriTile.Triton.Float.LogAdmission
import VeriTile.Triton.Float.GuardedIO
import VeriTile.Meta.StatementAudit

/-!
The admitted scalar LOG-EXP-EXPM1 relation replaces the masked piecewise
log1p/expm1 and log/exp expression with its finite fp32 input.
The selected report supplies this atom; plain LOG-EXP failed admission and
LOG-MUL remains inconclusive. Kernel implementations and their separate real
and FP specifications are in bench/examples/LogExp/.
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

/-- Bind the accepted LOG-EXP-EXPM1 row to the two fragments written above. The
imported admission table supplies report data, not a hidden theorem. -/
def entry := LogAdmission.fp32_log_exp_expm1.bind piecewiseLogExp.code identity.code

/-- Check that the selected row is exactly this fp32 rule and input domain. -/
theorem report_matches :
    LogAdmission.fp32_log_exp_expm1.report.ruleID = "LOG-EXP-EXPM1" ∧
    LogAdmission.fp32_log_exp_expm1.report.input = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.report.compute = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.report.accumulator = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.report.output = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.guards = guards := by decide

/-- The one experiment-selected assumption used in this example. Its validity
is the premise supplied by the two-gates workflow, not proved by Lean here. -/
structure Rules where
  log_exp_expm1 : Spec.EvidenceValidated entry.rule entry.evidence

def Rules.assumptions (_ : Rules) : Spec.Assumptions GuardedFragment := [entry]

instance : CoeOut Rules (Spec.Assumptions GuardedFragment) := ⟨Rules.assumptions⟩

/-- One application of the admitted atom rewrites the original into `out = a`. -/
theorem admitted (R : Rules) : Spec.Derivation R.assumptions [piecewiseLogExp] [identity] :=
  .atom entry (by simp [Rules.assumptions])
    (LogAdmission.fp32_log_exp_expm1.admit _ _ R.log_exp_expm1)

/-- Scalar rewrite available to kernels using this admitted relation. -/
theorem scalar_equiv (R : Rules) : [piecewiseLogExp] ≡[R] [identity] :=
  Spec.FloatingPoint.ofDerivation rfl trivial (admitted R)

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
  have h := hM piecewiseLogExp identity (admitted R) t hg hg
  simp only [piecewiseLogExp, identity, assignOutput, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, expression, nearZero, input,
    ComputeDType.eraseDType, numeric, numericLt, numericLe, hlt, hle,
    State.setReg_same, t] at h
  simp [State.setReg, bop, value] at h ⊢
  exact congrFun h PUnit.unit

end VeriTile.Triton.FP.LogExp
