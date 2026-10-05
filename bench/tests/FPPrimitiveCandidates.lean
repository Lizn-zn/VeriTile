import VeriTile.Triton.Float.Maximum
import VeriTile.Triton.Float.Exponential
import VeriTile.Triton.Float.CountConversion
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace FPPrimitiveCandidatesTests
open VeriTile Triton FP
open scoped VeriTile.Spec

/-- Compare candidates to separately translated source expressions. -/
def expressions : ComputeKernel := triton {
  a := tl.load($(("x" : RegionName)) + 0, dtype=tl.float32)
  b := tl.load($(("y" : RegionName)) + 0, dtype=tl.float32)
  c := tl.load($(("z" : RegionName)) + 0, dtype=tl.float32)
  out := libdevice.exp(a - b)
  out := libdevice.exp(a) / libdevice.exp(b)
  out := tl.exp(a - b)
  out := tl.exp(a) / tl.exp(b)
  out := tl.exp(float("-inf") - a)
  out := tl.maximum(a, b)
  out := tl.maximum(b, a)
  out := tl.maximum(tl.maximum(a, b), c)
  out := tl.maximum(a, tl.maximum(b, c))
  out := tl.maximum(a, a)
  out := tl.maximum(float("-inf"), a)
  out := a
  out := libdevice.exp(float("-inf") - a)
}

def fragmentAt (i : Nat) := expressions.surfaceBody[i]?.toList

example : (Exponential.Atom.exp_sub .libdevice).lhs.code = fragmentAt 3 ∧
    (Exponential.Atom.exp_sub .libdevice).rhs.code = fragmentAt 4 ∧
    Exponential.Atom.exp_sub.lhs.code = fragmentAt 5 ∧
    Exponential.Atom.exp_sub.rhs.code = fragmentAt 6 ∧
    Exponential.Atom.exp_neg_inf_sub.lhs.code = fragmentAt 7 := ⟨rfl, rfl, rfl, rfl, rfl⟩

example : (Exponential.Atom.exp_neg_inf_sub .libdevice).lhs.code = fragmentAt 15 := rfl

/-- Explicit fp32 constants, as in the numerical kernels. Scalar full has the
same execution as the candidate literal without assuming any exp identity. -/
private def zeroExpressions : ComputeKernel := triton {
  out := tl.exp(tl.full([], 0, dtype=tl.float32))
  out := libdevice.exp(tl.full([], 0, dtype=tl.float32))
}

open Structural Guarded in
theorem zero_sources_match {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    run M Exponential.Atom.exp_zero.lhs.code s = run M zeroExpressions.surfaceBody[0]?.toList s ∧
    run M (Exponential.Atom.exp_zero .libdevice).lhs.code s =
      run M zeroExpressions.surfaceBody[1]?.toList s := by
  simp [Exponential.Atom.lhs, Exponential.Backend.exp, ScalarArithmetic.fragment,
    zeroExpressions, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, ComputeDType.eraseDType]

example : (Maximum.lhs .max_commute).code = fragmentAt 8 ∧
    (Maximum.rhs .max_commute).code = fragmentAt 9 ∧
    (Maximum.lhs .max_assoc).code = fragmentAt 10 ∧
    (Maximum.rhs .max_assoc).code = fragmentAt 11 ∧
    (Maximum.lhs .max_idem).code = fragmentAt 12 ∧
    (Maximum.lhs .max_neg_inf).code = fragmentAt 13 ∧
    (Maximum.rhs .max_idem).code = fragmentAt 14 ∧
    (Maximum.rhs .max_neg_inf).code = fragmentAt 14 := ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

-- An admitted libdevice identity cannot enable the rejected tl.exp candidate.
example : (Exponential.Atom.exp_sub .libdevice).Available := by decide
example : ¬ Exponential.Atom.exp_sub.Available := by decide
example : Exponential.Atom.exp_sub.matches ExpAdmission.fp32_exp_sub =
    Bool.false := by decide

-- Constants prefer tl.exp; both accepted backends remain independently usable.
example (b : Exponential.Backend) :
    (Exponential.Atom.exp_zero b).Available ∧
    (Exponential.Atom.exp_neg_inf_sub b).Available := by cases b <;> decide

example : Exponential.Atom.exp_zero.matches ExpAdmission.fp32_exp_zero_libdevice = Bool.false ∧
    (Exponential.Atom.exp_zero .libdevice).matches ExpAdmission.fp32_exp_zero = Bool.false ∧
    Exponential.Atom.exp_neg_inf_sub.matches ExpAdmission.fp32_exp_neg_inf_sub_libdevice = Bool.false ∧
    (Exponential.Atom.exp_neg_inf_sub .libdevice).matches ExpAdmission.fp32_exp_neg_inf_sub = Bool.false := by decide

example : (Exponential.candidates.filterMap Exponential.Atom.entry?).length = 5 := rfl

example (a : Maximum.Atom) : (Maximum.lhs a).guards = (Maximum.rhs a).guards := by
  cases a <;> rfl
example : Maximum.Atom.max_neg_inf.guards = [⟨"a", .finite⟩] := rfl
example : Exponential.Atom.exp_zero.guards = [] := rfl
example : Exponential.Atom.exp_neg_inf_sub.guards = [⟨"a", .finite⟩] := rfl
example (a : CountConversion.Atom) : a.Available := by cases a <;> decide
example (f : Reciprocal.Format) : f.Available := by cases f <;> decide

private def row := SupplementalAdmission.fp32_max_commute
example : Maximum.Atom.max_commute.matches { row with report := { row.report with input := "bf16" } } = Bool.false ∧
    Maximum.Atom.max_commute.matches { row with report := { row.report with compute := "fp64" } } = Bool.false ∧
    Maximum.Atom.max_commute.matches { row with report := { row.report with accumulator := "fp64" } } = Bool.false ∧
    Maximum.Atom.max_commute.matches { row with report := { row.report with output := "bf16" } } = Bool.false ∧
    Maximum.Atom.max_commute.matches { row with guards := [] } = Bool.false := by decide

example : Reciprocal.Format.fp32.matches SupplementalAdmission.fp64_fp64_fp32_div_mul_rcp = Bool.false ∧
    Reciprocal.Format.fp64_fp32.matches SupplementalAdmission.fp32_div_mul_rcp = Bool.false ∧
    Reciprocal.Format.fp32.matches { SupplementalAdmission.fp32_div_mul_rcp with guards := [] } = Bool.false := by decide
example : CountConversion.Atom.successor.matches { CountAdmission.successor with input := "fp32" } = Bool.false ∧
    CountConversion.Atom.successor.matches CountAdmission.zero = Bool.false := by decide

-- A conditional proof can be written before an experiment accepts its candidate.
theorem intrinsic_when_selected (R : Exponential.Rules)
    (h : Exponential.Atom.exp_sub.Available) :
    [Exponential.Atom.exp_sub.lhs] ≡[R] [Exponential.Atom.exp_sub.rhs] :=
  Exponential.rewrite R .exp_sub h

theorem selected_exp_zero (R : Exponential.Rules) :
    [Exponential.Atom.exp_zero.lhs] ≡[R] [Exponential.Atom.exp_zero.rhs] :=
  Exponential.rewrite R .exp_zero (by decide)

theorem selected_exp_zero_libdevice (R : Exponential.Rules) :
    [(Exponential.Atom.exp_zero .libdevice).lhs] ≡[R]
      [(Exponential.Atom.exp_zero .libdevice).rhs] :=
  Exponential.rewrite R (.exp_zero .libdevice) (by decide)

theorem selected_exp_neg_inf_sub (R : Exponential.Rules) :
    [Exponential.Atom.exp_neg_inf_sub.lhs] ≡[R] [Exponential.Atom.exp_neg_inf_sub.rhs] :=
  Exponential.rewrite R .exp_neg_inf_sub (by decide)

theorem selected_exp_neg_inf_sub_libdevice (R : Exponential.Rules) :
    [(Exponential.Atom.exp_neg_inf_sub .libdevice).lhs] ≡[R]
      [(Exponential.Atom.exp_neg_inf_sub .libdevice).rhs] :=
  Exponential.rewrite R (.exp_neg_inf_sub .libdevice) (by decide)

theorem selected_maximum (R : Maximum.Rules) :
    [Maximum.lhs .max_commute] ≡[R] [Maximum.rhs .max_commute] :=
  Maximum.rewrite R .max_commute (by decide)

#print_fp_assumptions selected_maximum
#print_fp_assumptions selected_exp_zero
#print_fp_assumptions selected_exp_zero_libdevice
#print_fp_assumptions selected_exp_neg_inf_sub
#print_fp_assumptions selected_exp_neg_inf_sub_libdevice
#axiomsClean ScalarArithmetic.admitted
#axiomsClean Reciprocal.admitted
#axiomsClean Exponential.admitted
#axiomsClean CountConversion.admitted
#axiomsClean Maximum.admitted
#axiomsClean intrinsic_when_selected

end FPPrimitiveCandidatesTests
