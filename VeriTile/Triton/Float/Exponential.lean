/- Bind the admitted fp32 libdevice EXP-SUB relation to its exact scalar
syntax. The measured tl.exp EXP-SUB-INTRINSIC relation fails the configured bias gate
(B = 0.1608954387 ULP > 0.05); no assumption for that symbol is supplied. -/
import VeriTile.Triton.Float.SoftmaxShift

namespace VeriTile.Triton.FP.Exponential
open Structural Guarded ScalarArithmetic

def guards : List OperandGuard := [⟨"a", .finite⟩, ⟨"b", .finite⟩]
def lhsCode : List ComputeStmt := fragment (.libdeviceExp (minus (ref "a") (ref "b")))
def rhsCode : List ComputeStmt := fragment (divide (.libdeviceExp (ref "a")) (.libdeviceExp (ref "b")))
def lhs : GuardedFragment := ⟨guards, lhsCode⟩
def rhs : GuardedFragment := ⟨guards, rhsCode⟩
def entry := SupplementalAdmission.fp32_exp_sub.bind lhsCode rhsCode

theorem report_matches :
    SupplementalAdmission.fp32_exp_sub.report.ruleID = "EXP-SUB" ∧
    SupplementalAdmission.fp32_exp_sub.report.input = "fp32" ∧
    SupplementalAdmission.fp32_exp_sub.report.compute = "fp32" ∧
    SupplementalAdmission.fp32_exp_sub.report.accumulator = "fp32" ∧
    SupplementalAdmission.fp32_exp_sub.report.output = "fp32" ∧
    SupplementalAdmission.fp32_exp_sub.guards = guards := by decide

structure Rules where
  arithmetic : ScalarArithmetic.Rules
  exp_sub : Spec.EvidenceValidated entry.rule entry.evidence

def Rules.assumptions (R : Rules) : Spec.Assumptions GuardedFragment :=
  R.arithmetic.assumptions ++ [entry]

instance : CoeOut Rules (Spec.Assumptions GuardedFragment) := ⟨Rules.assumptions⟩

theorem admitted (R : Rules) : Spec.Derivation R.assumptions [lhs] [rhs] :=
  .atom entry (by simp [Rules.assumptions])
    (SupplementalAdmission.fp32_exp_sub.admit _ _ R.exp_sub)

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

set_option maxHeartbeats 1600000 in
theorem exp_sub {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) : SoftmaxShift.LibdeviceExpSub M D := by
  constructor
  intro a b ha hb
  let t := (s.setReg "a" .real [] (fun _ => a)).setReg "b" .real [] (fun _ => b)
  have hg : ScalarDomain D guards t := by
    intro g hg
    simp [guards] at hg
    rcases hg with rfl | rfl
    · exact ⟨fun _ => a, by simp [t], ha⟩
    · exact ⟨fun _ => b, by simp [t], hb⟩
  have h := hM lhs rhs (admitted R) t hg hg
  simp only [lhs, rhs, lhsCode, rhsCode, fragment, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, ref, minus, divide, ComputeDType.eraseDType,
    numeric, State.setReg_same, t] at h
  simp [State.setReg, SoftmaxShift.exp, sub, div] at h ⊢
  exact congrFun h PUnit.unit

end VeriTile.Triton.FP.Exponential
