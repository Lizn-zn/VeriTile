/- The admitted count atoms operate on nonnegative int32 values embedded in
Nat. Their range fits int32, including the successor, so integer arithmetic
here has the same value as the tested integer arithmetic. Floating conversion
and addition remain opaque. No natural-to-float equality is proved from reals.

The successor's input bound is encoded in its scalar syntax: outside the
accepted range both fragments return the same zero. Extracting the actual
successor equation requires an explicit in-range proof. This avoids extending
the report to arbitrary loop indices or hiding a range condition in prose. -/
import VeriTile.Triton.Float.CountAdmission
import VeriTile.Triton.Float.WelfordInduction

namespace VeriTile.Triton.FP.CountConversion
open Structural Guarded ScalarArithmetic

abbrev limit : Nat := CountAdmission.upperExclusive

def index : Op .nat [] := .ref .nat [] "i"
def bounded (e : Op .real []) : Op .real [] :=
  .where (.lt .nat .nil index (.constNat limit)) e (.const 0)

inductive Atom where
  | zero | successor
  deriving DecidableEq

def report : Atom → ReportedRule
  | .zero => CountAdmission.zero
  | .successor => CountAdmission.successor

def lhsCode : Atom → List ComputeStmt
  | .zero => fragment (.natToReal (.constNat 0))
  | .successor => fragment (bounded (.natToReal (.add .nat .nil index (.constNat 1))))

def rhsCode : Atom → List ComputeStmt
  | .zero => fragment (.const 0)
  | .successor => fragment (bounded (plus (.natToReal index) (.const 1)))

def lhs (a : Atom) : GuardedFragment := ⟨[], lhsCode a⟩
def rhs (a : Atom) : GuardedFragment := ⟨[], rhsCode a⟩
def entry (a : Atom) : Spec.RuleEntry GuardedFragment := (report a).bind [lhs a] [rhs a]

structure Rules where
  arithmetic : ScalarArithmetic.Rules
  zero : Spec.EvidenceValidated (entry .zero).rule (entry .zero).evidence
  successor : Spec.EvidenceValidated (entry .successor).rule (entry .successor).evidence

def Rules.assumptions (R : Rules) : Spec.Assumptions GuardedFragment :=
  R.arithmetic.assumptions ++ [entry .zero, entry .successor]

instance : CoeOut Rules (Spec.Assumptions GuardedFragment) := ⟨Rules.assumptions⟩

theorem admitted (R : Rules) (a : Atom) : Spec.Derivation R.assumptions [lhs a] [rhs a] := by
  apply Spec.Derivation.atom (entry a)
  · cases a <;> simp [Rules.assumptions]
  · apply ReportedRule.admit
    cases a
    · exact R.zero
    · exact R.successor

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
theorem zero {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) (s : State α) :
    M.fromNat (some .fp32) 0 = ScalarArithmetic.zero M := by
  have h := hM (lhs .zero) (rhs .zero) (admitted R .zero) s
    (by simp [ScalarDomain, lhs]) (by simp [ScalarDomain, rhs])
  simp only [lhs, rhs, lhsCode, rhsCode, fragment, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, ComputeDType.eraseDType] at h
  simp [State.setReg, ScalarArithmetic.zero] at h ⊢
  exact congrFun h PUnit.unit

set_option maxHeartbeats 1600000 in
theorem successor {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (i : Nat) (hi : i < limit) :
    M.fromNat (some .fp32) (i + 1) = add M (M.fromNat (some .fp32) i) (one M) := by
  let t := s.setReg "i" .nat [] (fun _ => i)
  have h := hM (lhs .successor) (rhs .successor) (admitted R .successor) t
    (by simp [ScalarDomain, lhs]) (by simp [ScalarDomain, rhs])
  simp only [lhs, rhs, lhsCode, rhsCode, fragment, bounded, index, plus, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType, t, State.setReg_same] at h
  simp [State.setReg, numeric, natLt, bop, hi, ScalarArithmetic.add, ScalarArithmetic.one] at h ⊢
  exact congrFun h PUnit.unit

/-- Every conversion needed by the original N-step recurrence, with the
admitted upper bound retained. This is derived from two scalar atoms. -/
theorem conversion {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (N : Nat) (hN : N ≤ limit) : WelfordInduction.CountConversion M N :=
  ⟨zero R M D hM s, fun i hi => successor R M D hM s i (lt_of_lt_of_le hi hN)⟩

end VeriTile.Triton.FP.CountConversion
