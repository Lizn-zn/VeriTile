import VeriTile.Triton.Float.LogExp
import VeriTile.Triton.DSL

/- Candidate syntax is checked against Triton expressions independently of
admission. The negative selection checks do not supply numerical evidence. -/
namespace FPLogCandidatesTests
open VeriTile Triton FP FP.LogExp
open scoped VeriTile.Spec

def expressions : ComputeKernel := triton {
  a := tl.load($(("x" : RegionName)) + 0, dtype=tl.float32)
  b := tl.load($(("y" : RegionName)) + 0, dtype=tl.float32)
  out := tl.log(a * b)
  out := tl.log(a) + tl.log(b)
  out := libdevice.log(a * b)
  out := libdevice.log(a) + libdevice.log(b)
  out := tl.log(tl.exp(a))
  out := tl.log(libdevice.exp(a))
  out := libdevice.log(libdevice.exp(a))
  out := a
}

def fragmentAt (i : Nat) : List ComputeStmt := expressions.surfaceBody[i]?.toList

theorem intrinsic_product_syntax :
    Atom.log_mul.lhs.code = fragmentAt 2 ∧
    Atom.log_mul.rhs.code = fragmentAt 3 := ⟨rfl, rfl⟩

theorem libdevice_product_syntax :
    Atom.log_mul_libdevice.lhs.code = fragmentAt 4 ∧
    Atom.log_mul_libdevice.rhs.code = fragmentAt 5 := ⟨rfl, rfl⟩

theorem cancellation_intrinsics :
    Atom.log_exp.lhs.code = fragmentAt 6 ∧
    Atom.log_exp_libdevice.lhs.code = fragmentAt 7 ∧
    Atom.log_exp_full_libdevice.lhs.code = fragmentAt 8 ∧
    Atom.log_exp.rhs.code = fragmentAt 9 ∧
    Atom.log_exp_libdevice.rhs.code = fragmentAt 9 ∧
    Atom.log_exp_full_libdevice.rhs.code = fragmentAt 9 := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem candidate_domains (a : Atom) : a.lhs.guards = a.rhs.guards := by
  cases a <;> rfl

theorem product_needs_positive_inputs :
    Atom.log_mul.lhs.guards =
      [⟨"a", .finite⟩, ⟨"b", .finite⟩, ⟨"a", .positive⟩, ⟨"b", .positive⟩] ∧
    Atom.log_mul_libdevice.lhs.guards = Atom.log_mul.lhs.guards := ⟨rfl, rfl⟩

theorem cancellation_allows_negative_finite_inputs :
    Atom.log_exp.lhs.guards = [⟨"a", .finite⟩] ∧
    Atom.log_exp_libdevice.lhs.guards = [⟨"a", .finite⟩] ∧
    Atom.log_exp_full_libdevice.lhs.guards = [⟨"a", .finite⟩] ∧
    Atom.log_exp_expm1.lhs.guards = [⟨"a", .finite⟩] := ⟨rfl, rfl, rfl, rfl⟩

private def row := LogAdmission.fp32_log_exp_expm1

theorem wrong_precision_is_unavailable :
    Atom.log_exp_expm1.matches { row with report := { row.report with input := "bf16" } } = Bool.false ∧
    Atom.log_exp_expm1.matches { row with report := { row.report with compute := "fp64" } } = Bool.false ∧
    Atom.log_exp_expm1.matches { row with report := { row.report with accumulator := "fp64" } } = Bool.false ∧
    Atom.log_exp_expm1.matches { row with report := { row.report with output := "bf16" } } = Bool.false := by
  decide

theorem wrong_relation_or_domain_is_unavailable :
    Atom.log_exp.matches row = Bool.false ∧
    Atom.log_exp_libdevice.matches row = Bool.false ∧
    Atom.log_exp_full_libdevice.matches row = Bool.false ∧
    Atom.log_exp_expm1.matches { row with guards := [⟨"a", .positive⟩] } = Bool.false := by
  decide

theorem selected_piecewise (R : Rules) :
    [Atom.log_exp_expm1.lhs] ≡[R] [Atom.log_exp_expm1.rhs] :=
  FP.LogExp.rewrite R .log_exp_expm1 (by decide)

#print_fp_assumptions selected_piecewise

-- A reusable conditional theorem can be written before this candidate passes.
theorem log_mul_when_selected (R : Rules) (h : Atom.log_mul.Available) :
    [Atom.log_mul.lhs] ≡[R] [Atom.log_mul.rhs] :=
  FP.LogExp.rewrite R .log_mul h

#axiomsClean FP.LogExp.derive
#axiomsClean FP.LogExp.rewrite
#axiomsClean log_mul_when_selected

end FPLogCandidatesTests
