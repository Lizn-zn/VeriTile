/- PR #13 admits the masked, piecewise LOG-EXP-EXPM1 expression only.
Plain tl.log/libdevice.log after libdevice.exp failed its bias gate.
LOG-MUL remains inconclusive. Neither old relation is an assumption here. -/
import VeriTile.Triton.Float.LogAdmission
import VeriTile.Triton.Float.ScalarArithmetic
import VeriTile.Meta.StatementAudit

noncomputable section
namespace VeriTile.Triton.FP.LogExp
open Structural Guarded ScalarArithmetic
open scoped VeriTile.Spec

def guards : List OperandGuard := [⟨"a", .finite⟩]

/-- Same comparison as `tl.abs(a) <= 0.5`; no near-zero input restriction. -/
def nearZero (a : Op .real []) : Op .bool [] :=
  .le .real .nil
    (.where (.lt .real .nil a (.const 0)) (minus (.const 0) a) a)
    (.const (1 / 2))

/-- Both arms of `where` are evaluated. Mask the unused argument to zero,
as in the measured PR #13 source, before either libdevice call. -/
def expression (a : Op .real []) : Op .real [] :=
  let near := nearZero a
  .where near
    (.libdeviceLog1p (.libdeviceExpm1 (.where near a (.const 0))))
    (.libdeviceLog (.libdeviceExp (.where near (.const 0) a)))

def lhsCode : List ComputeStmt := fragment (expression (ref "a"))
def rhsCode : List ComputeStmt := fragment (ref "a")
def lhs : GuardedFragment := ⟨guards, lhsCode⟩
def rhs : GuardedFragment := ⟨guards, rhsCode⟩
def entry := LogAdmission.fp32_log_exp_expm1.bind lhsCode rhsCode

theorem report_matches :
    LogAdmission.fp32_log_exp_expm1.report.ruleID = "LOG-EXP-EXPM1" ∧
    LogAdmission.fp32_log_exp_expm1.report.input = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.report.compute = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.report.accumulator = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.report.output = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.guards = guards := by decide

structure Rules where
  log_exp_expm1 : Spec.EvidenceValidated entry.rule entry.evidence

def Rules.assumptions (_ : Rules) : Spec.Assumptions GuardedFragment := [entry]

instance : CoeOut Rules (Spec.Assumptions GuardedFragment) := ⟨Rules.assumptions⟩

theorem admitted (R : Rules) : Spec.Derivation R.assumptions [lhs] [rhs] :=
  .atom entry (by simp [Rules.assumptions])
    (LogAdmission.fp32_log_exp_expm1.admit _ _ R.log_exp_expm1)

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
  have h := hM lhs rhs (admitted R) t hg hg
  simp only [lhs, rhs, lhsCode, rhsCode, fragment, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, expression, nearZero, ref, minus,
    ComputeDType.eraseDType, numeric, numericLt, numericLe, hlt, hle,
    State.setReg_same, t] at h
  simp [State.setReg, bop, value] at h ⊢
  exact congrFun h PUnit.unit

/-- FP equivalence under the single accepted atom, not bitwise IEEE equality. -/
specification log_exp_expm1_equiv (R : Rules) : [lhs] ≡[R] [rhs] :=
  Spec.FloatingPoint.ofDerivation rfl trivial (admitted R)

#print_fp_assumptions log_exp_expm1_equiv

end VeriTile.Triton.FP.LogExp
