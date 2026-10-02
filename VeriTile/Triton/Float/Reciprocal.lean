/- Ordinary Triton division versus a shared reciprocal. These are scalar
fragments, including the tested output cast where the report requires one. -/
import VeriTile.Triton.Float.GuardedIO
import VeriTile.Triton.Float.SupplementalAdmission

namespace VeriTile.Triton.FP.Reciprocal
open Structural Guarded

inductive Format where
  | fp32 | fp64_fp32
  deriving DecidableEq

def guards : List OperandGuard := [⟨"a", .finite⟩, ⟨"b", .finite⟩, ⟨"b", .nonzero⟩]

def ref (s : String) : Op .real [] := .ref .real [] s

def div : Op .real [] := .div .real .nil (ref "a") (ref "b")
def inv : Op .real [] := .div .real .nil (.const 1) (ref "b")
def mul : Op .real [] := .mul .real .nil (ref "a") (ref "inv")

def lhs : Format → GuardedFragment
  | .fp32 => ⟨guards, [.assign .real [] "out" (.compute (.alg .fp32 div))]⟩
  | .fp64_fp32 => ⟨guards, [
      .assign .real [] "value" (.compute (.alg .fp64 div)),
      .assign .real [] "out" (.compute (.alg .fp32 (.castFloat .real .real (ref "value"))))]⟩

def rhs : Format → GuardedFragment
  | .fp32 => ⟨guards, [
      .assign .real [] "inv" (.compute (.alg .fp32 inv)),
      .assign .real [] "out" (.compute (.alg .fp32 mul))]⟩
  | .fp64_fp32 => ⟨guards, [
      .assign .real [] "inv" (.compute (.alg .fp64 inv)),
      .assign .real [] "value" (.compute (.alg .fp64 mul)),
      .assign .real [] "out" (.compute (.alg .fp32 (.castFloat .real .real (ref "value"))))]⟩

def report : Format → ReportedScalarRule
  | .fp32 => SupplementalAdmission.fp32_div_mul_rcp
  | .fp64_fp32 => SupplementalAdmission.fp64_fp64_fp32_div_mul_rcp

def entry (f : Format) := (report f).bind (lhs f).code (rhs f).code

structure Rules (f : Format) where
  div_mul_rcp : Spec.EvidenceValidated (entry f).rule (entry f).evidence

def Rules.assumptions {f : Format} (_ : Rules f) : Spec.Assumptions GuardedFragment := [entry f]

instance {f : Format} : CoeOut (Rules f) (Spec.Assumptions GuardedFragment) :=
  ⟨Rules.assumptions⟩

theorem admitted {f : Format} (R : Rules f) :
    Spec.Derivation R.assumptions [lhs f] [rhs f] := by
  have h := Spec.Derivation.atom (assumptions := R.assumptions) (entry f) (by simp [Rules.assumptions])
    ((report f).admit _ _ R.div_mul_rcp)
  cases f <;> exact h

def precision : Format → ComputeDType
  | .fp32 => .fp32
  | .fp64_fp32 => .fp64

def finish {α : Type} (f : Format) (M : Algebra α) (v : α) : α :=
  match f with
  | .fp32 => v
  | .fp64_fp32 => M.cast (some .fp32) .real .real v

def quotient {α : Type} (f : Format) (M : Algebra α) (a b : α) : α :=
  finish f M (M.binary (some (precision f)) .real .div a b)

def reciprocal {α : Type} (f : Format) (M : Algebra α) (a b : α) : α :=
  finish f M (M.binary (some (precision f)) .real .mul a
    (M.binary (some (precision f)) .real .div (M.literal (some (precision f)) .real 1) b))

set_option maxHeartbeats 1200000 in
theorem apply_rule {α : Type} [Inhabited α] {f : Format} (R : Rules f)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (a b : α) (ha : D .finite a) (hb : D .finite b) (hn : D .nonzero b) :
    quotient f M a b = reciprocal f M a b := by
  let t := (s.setReg "a" .real [] (fun _ => a)).setReg "b" .real [] (fun _ => b)
  have hg : ScalarDomain D guards t := by
    intro g hg
    simp [guards] at hg
    rcases hg with rfl | rfl | rfl
    · exact ⟨fun _ => a, by simp [t], ha⟩
    · exact ⟨fun _ => b, by simp [t], hb⟩
    · exact ⟨fun _ => b, by simp [t], hn⟩
  have h := hM (lhs f) (rhs f) (admitted R) t
  cases f <;> simp only [lhs, rhs] at h
  all_goals
    specialize h hg hg
    simp only [run, step, evalExpr, evalComputeOp, evalOp_unfold, div, inv, mul, ref,
      ComputeDType.eraseDType, numeric, State.setReg_same, t] at h
    -- Simplify concrete register names before selecting the two successful runs.
    simp [State.setReg, quotient, reciprocal, finish, precision] at h ⊢
    exact congrFun h PUnit.unit

end VeriTile.Triton.FP.Reciprocal
