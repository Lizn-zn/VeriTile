import VeriTile.Triton.Float.LogAdmission
import VeriTile.Triton.Float.GuardedIO
import VeriTile.Meta.StatementAudit

/-!
Replace a piecewise fp32 log-exp computation with its input.

Input: one finite fp32 value `a`. Output: the fp32 register `out`.
The original Triton computation is:

```python
near_zero = tl.abs(a) <= 0.5
small_a = tl.where(near_zero, a, 0.0)
other_a = tl.where(near_zero, 0.0, a)
small = libdevice.log1p(libdevice.expm1(small_a))
other = libdevice.log(libdevice.exp(other_a))
out = tl.where(near_zero, small, other)
```

The replacement is simply:

```python
out = a
```

`expm1(x)` computes `exp(x) - 1`; `log1p(x)` computes `log(1 + x)`.
Their small-argument implementations avoid losing a small `a` when `exp(a)`
rounds to 1. Outside that branch, the original log-exp path is retained. Since
`tl.where` evaluates both arms, each unused argument is first set to zero.
The threshold selects the implementation; it does not restrict the input.

The specification below is `[piecewiseLogExp] ≡[R] [identity]`. Its sole
numerical assumption is `log_exp_expm1`, admitted by the PR #13 two-gates
experiment. This file defines both fragments, binds that accepted row to
their exact syntax, and proves the rewrite by using the atom once. The
experiment chooses the assumption; its array shape is not a proof parameter.

This is an accepted FP rewrite, not a claim of exact IEEE equality. In
particular, the plain `log(exp(a)) = a` relation failed admission, and
`LOG-MUL` remains inconclusive. Neither is assumed or proved here.
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
as in the measured PR #13 source, before either libdevice call. -/
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

/-- Bind the accepted PR #13 row to the two fragments written above. The
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

/-- Public specification: replace the piecewise computation with its input
under the single two-gates-accepted atomic assumption. -/
specification log_exp_expm1_equiv (R : Rules) : [piecewiseLogExp] ≡[R] [identity] :=
  Spec.FloatingPoint.ofDerivation rfl trivial (admitted R)

-- Output:
-- FP assumptions used by log_exp_expm1_equiv:
--   log_exp_expm1
#print_fp_assumptions log_exp_expm1_equiv

/-! Execution interpretation of the same rewrite.

The specification above is complete. The remaining helper lets a larger
kernel proof use it on a scalar value: `M` interprets FP operations as opaque
functions, `D` interprets the input domain, and `Models` says that successful
executions obey the selected atoms. `lt` and `le` provide the two comparisons;
no real-number arithmetic laws or additional numerical atoms are introduced.
-/

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
