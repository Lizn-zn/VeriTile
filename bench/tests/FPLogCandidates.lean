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
  out := tl.where((a * b >= 0.5) & (a * b <= 2.0),
    libdevice.log(tl.where((a * b >= 0.5) & (a * b <= 2.0), a * b, 1.0)),
    libdevice.log(tl.where((a * b >= 0.5) & (a * b <= 2.0), 1.0, a)) +
      libdevice.log(tl.where((a * b >= 0.5) & (a * b <= 2.0), 1.0, b)))
}

def fragmentAt (i : Nat) : List ComputeStmt := expressions.surfaceBody[i]?.toList

theorem intrinsic_product_syntax :
    Atom.log_mul.lhs.code = fragmentAt 2 ∧
    Atom.log_mul.rhs.code = fragmentAt 3 := ⟨rfl, rfl⟩

theorem libdevice_product_syntax :
    Atom.log_mul_libdevice.lhs.code = fragmentAt 4 ∧
    Atom.log_mul_libdevice.rhs.code = fragmentAt 5 := ⟨rfl, rfl⟩

-- The DSL leaves the outer where expression untagged. Its enclosing fp32
-- profile is checked against the actual assignments in FPLogProduct.lean.
theorem conditional_product_syntax :
    Atom.log_mul_split.lhs.code = fragmentAt 4 ∧
    [.assign .real [] "out" (.alg (splitProduct input secondInput))] = fragmentAt 10 := by
  have half : (0.5 : ℝ) = 1 / 2 := by norm_num
  have one : (1.0 : ℝ) = 1 := by norm_num
  have two : (2.0 : ℝ) = 2 := by norm_num
  constructor
  · rfl
  · simp only [splitProduct, keepProduct, input, secondInput,
      fragmentAt, expressions, ComputeKernel.surfaceBody, half, one, two]
    rfl

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
    Atom.log_mul_libdevice.lhs.guards = Atom.log_mul.lhs.guards ∧
    Atom.log_mul_split.lhs.guards = Atom.log_mul.lhs.guards := ⟨rfl, rfl, rfl⟩

theorem cancellation_allows_negative_finite_inputs :
    Atom.log_exp.lhs.guards = [⟨"a", .finite⟩] ∧
    Atom.log_exp_libdevice.lhs.guards = [⟨"a", .finite⟩] ∧
    Atom.log_exp_full_libdevice.lhs.guards = [⟨"a", .finite⟩] ∧
    Atom.log_exp_elim.lhs.guards = [⟨"a", .finite⟩] := ⟨rfl, rfl, rfl, rfl⟩

private def row := LogAdmission.fp32_log_exp_guarded

theorem guarded_candidate_keeps_the_reference :
    Atom.log_exp_elim.lhs = Atom.log_exp_full_libdevice.lhs ∧
    Atom.log_exp_elim.lhs = originalLogExp ∧
    Atom.log_exp_elim.rhs = piecewiseLogExp := ⟨rfl, rfl, rfl⟩

theorem wrong_precision_is_unavailable :
    Atom.log_exp_elim.matches { row with report := { row.report with input := "bf16" } } = Bool.false ∧
    Atom.log_exp_elim.matches { row with report := { row.report with compute := "fp64" } } = Bool.false ∧
    Atom.log_exp_elim.matches { row with report := { row.report with accumulator := "fp64" } } = Bool.false ∧
    Atom.log_exp_elim.matches { row with report := { row.report with output := "bf16" } } = Bool.false := by
  decide

theorem wrong_relation_or_domain_is_unavailable :
    Atom.log_exp.matches row = Bool.false ∧
    Atom.log_exp_libdevice.matches row = Bool.false ∧
    Atom.log_exp_full_libdevice.matches row = Bool.false ∧
    Atom.log_exp_elim.matches { row with guards := [⟨"a", .positive⟩] } = Bool.false := by
  decide

theorem selected_piecewise (R : Rules) :
    [Atom.log_exp_elim.lhs] ≡[R] [Atom.log_exp_elim.rhs] :=
  FP.LogExp.rewrite R .log_exp_elim (by decide)

#print_fp_assumptions selected_piecewise

theorem selected_product_split (R : Rules) :
    [Atom.log_mul_split.lhs] ≡[R] [Atom.log_mul_split.rhs] :=
  FP.LogExp.rewrite R .log_mul_split (by decide)

#print_fp_assumptions selected_product_split

-- Renaming the public atom keeps the original experimental identity and all
-- four precision fields. A conditional report cannot select a plain log rule.
private def productRow := LogAdmission.fp32_log_mul_guarded

theorem split_report_identity_and_selection :
    Atom.log_mul_split.ruleID = "LOG-MUL-GUARDED" ∧
    Atom.log_mul_split.matches productRow = Bool.true ∧
    Atom.log_mul.matches productRow = Bool.false ∧
    Atom.log_mul_libdevice.matches productRow = Bool.false := by decide

theorem split_wrong_precision_or_domain :
    Atom.log_mul_split.matches { productRow with report := { productRow.report with input := "bf16" } } = Bool.false ∧
    Atom.log_mul_split.matches { productRow with report := { productRow.report with compute := "fp64" } } = Bool.false ∧
    Atom.log_mul_split.matches { productRow with report := { productRow.report with accumulator := "fp64" } } = Bool.false ∧
    Atom.log_mul_split.matches { productRow with report := { productRow.report with output := "bf16" } } = Bool.false ∧
    Atom.log_mul_split.matches { productRow with guards := [] } = Bool.false := by decide

-- A reusable conditional theorem can be written before this candidate passes.
theorem log_mul_when_selected (R : Rules) (h : Atom.log_mul.Available) :
    [Atom.log_mul.lhs] ≡[R] [Atom.log_mul.rhs] :=
  FP.LogExp.rewrite R .log_mul h

#axiomsClean FP.LogExp.derive
#axiomsClean FP.LogExp.rewrite
#axiomsClean FP.LogExp.apply_log_mul_split
#axiomsClean selected_product_split
#axiomsClean log_mul_when_selected

end FPLogCandidatesTests
