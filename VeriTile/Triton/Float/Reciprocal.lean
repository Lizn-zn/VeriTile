/- Ordinary Triton division versus a shared reciprocal. These are scalar
fragments, including the tested output cast where the report requires one. -/
import VeriTile.Triton.Float.GuardedIO
import VeriTile.Triton.Float.SupplementalAdmission

namespace VeriTile.Triton.FP.Reciprocal
open Structural Guarded
open scoped VeriTile.Spec

/-- Ordinary division and a separately rounded reciprocal are different
computations until this candidate has been admitted. -/
inductive Atom where
  | div_mul_rcp
  deriving DecidableEq, Repr

def Atom.ruleID : Atom → String
  -- a / b → a * (1 / b); finite a,b and b ≠ 0. This is Triton `/`, not tl.div_rn.
  | .div_mul_rcp => "DIV-MUL-RCP"

def candidates : List Atom := [.div_mul_rcp]

/-- fp32 computes and returns fp32. fp64_fp32 computes in fp64 and casts
both results to fp32; it supplies no uncast fp64 equality. -/
inductive Format where
  | fp32 | fp64_fp32
  deriving DecidableEq, Repr

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

def Format.matches (f : Format) (row : ReportedScalarRule) : Bool :=
  let compute := match f with | .fp32 => "fp32" | .fp64_fp32 => "fp64"
  decide (row.report.ruleID = Atom.div_mul_rcp.ruleID ∧ row.report.input = compute ∧
    row.report.compute = compute ∧ row.report.accumulator = compute ∧
    row.report.output = "fp32" ∧ row.guards = guards)

def Format.report? (f : Format) : Option ReportedScalarRule :=
  SupplementalAdmission.all.find? f.matches

abbrev Format.Available (f : Format) : Prop := f.report?.isSome = true

def report (f : Format) (h : f.Available) : ReportedScalarRule := f.report?.get h

def entry (f : Format) (h : f.Available) : Spec.RuleEntry GuardedFragment :=
  (report f h).report.bind [lhs f] [rhs f]

def Format.entry? (f : Format) : Option (Spec.RuleEntry GuardedFragment) :=
  f.report?.map fun row => row.report.bind [lhs f] [rhs f]

structure Rules (f : Format) where
  validated : ∀ (h : f.Available),
    Spec.EvidenceValidated (entry f h).rule (entry f h).evidence

def Rules.assumptions {f : Format} (_ : Rules f) : Spec.Assumptions GuardedFragment :=
  f.entry?.toList

instance {f : Format} : CoeOut (Rules f) (Spec.Assumptions GuardedFragment) :=
  ⟨Rules.assumptions⟩

theorem admitted {f : Format} (R : Rules f) (h : f.Available) :
    Spec.Derivation R.assumptions [lhs f] [rhs f] := by
  apply Spec.Derivation.atom (entry f h)
  · have available : f.report?.isSome = true := h
    cases hr : f.report? with
    | none => simp [hr] at available
    | some row => simp [Rules.assumptions, Format.entry?, entry, report, hr]
  · exact (report f h).report.admit _ _ (R.validated h)

theorem rewrite {f : Format} (R : Rules f) (h : f.Available) : [lhs f] ≡[R] [rhs f] :=
  Spec.FloatingPoint.ofDerivation rfl trivial (admitted R h)

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
theorem apply_rule {α : Type} [Inhabited α] {f : Format} (R : Rules f) (selected : f.Available)
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
  have h := hM (lhs f) (rhs f) (admitted R selected) t
  cases f <;> simp only [lhs, rhs] at h
  all_goals
    specialize h hg hg
    simp only [run, step, evalExpr, evalComputeOp, evalOp_unfold, div, inv, mul, ref,
      ComputeDType.eraseDType, numeric, State.setReg_same, t] at h
    -- Simplify concrete register names before selecting the two successful runs.
    simp [State.setReg, quotient, reciprocal, finish, precision] at h ⊢
    exact congrFun h PUnit.unit

end VeriTile.Triton.FP.Reciprocal
