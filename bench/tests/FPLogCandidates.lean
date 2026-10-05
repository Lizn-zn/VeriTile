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
    (Atom.log_mul .tl).lhs.code = fragmentAt 2 ∧
    (Atom.log_mul .tl).rhs.code = fragmentAt 3 := ⟨rfl, rfl⟩

theorem libdevice_product_syntax :
    (Atom.log_mul .libdevice).lhs.code = fragmentAt 4 ∧
    (Atom.log_mul .libdevice).rhs.code = fragmentAt 5 := ⟨rfl, rfl⟩

-- The DSL leaves the outer where expression untagged. Its enclosing fp32
-- profile is checked against the actual assignments in FPLogProduct.lean.
theorem conditional_product_syntax :
    (Atom.log_mul_split .libdevice).lhs.code = fragmentAt 4 ∧
    [.assign .real [] "out" (.alg (splitProduct .libdevice input secondInput))] = fragmentAt 10 := by
  have half : (0.5 : ℝ) = 1 / 2 := by norm_num
  have one : (1.0 : ℝ) = 1 := by norm_num
  have two : (2.0 : ℝ) = 2 := by norm_num
  constructor
  · rfl
  · simp only [splitProduct, Backend.log, keepProduct, input, secondInput,
      fragmentAt, expressions, ComputeKernel.surfaceBody, half, one, two]
    rfl

theorem cancellation_intrinsics :
    (Atom.log_exp .tl .tl).lhs.code = fragmentAt 6 ∧
    (Atom.log_exp .tl .libdevice).lhs.code = fragmentAt 7 ∧
    (Atom.log_exp .libdevice .libdevice).lhs.code = fragmentAt 8 ∧
    (Atom.log_exp .tl .tl).rhs.code = fragmentAt 9 ∧
    (Atom.log_exp .tl .libdevice).rhs.code = fragmentAt 9 ∧
    (Atom.log_exp .libdevice .libdevice).rhs.code = fragmentAt 9 := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem candidate_domains (a : Atom) : a.lhs.guards = a.rhs.guards := by
  cases a <;> rfl

theorem product_needs_positive_inputs :
    (Atom.log_mul .tl).lhs.guards =
      [⟨"a", .finite⟩, ⟨"b", .finite⟩, ⟨"a", .positive⟩, ⟨"b", .positive⟩] ∧
    (Atom.log_mul .libdevice).lhs.guards = (Atom.log_mul .tl).lhs.guards ∧
    (Atom.log_mul_split .libdevice).lhs.guards = (Atom.log_mul .tl).lhs.guards := ⟨rfl, rfl, rfl⟩

theorem cancellation_allows_negative_finite_inputs :
    (Atom.log_exp .tl .tl).lhs.guards = [⟨"a", .finite⟩] ∧
    (Atom.log_exp .tl .libdevice).lhs.guards = [⟨"a", .finite⟩] ∧
    (Atom.log_exp .libdevice .libdevice).lhs.guards = [⟨"a", .finite⟩] ∧
    (Atom.log_exp_cancel .libdevice).lhs.guards = [⟨"a", .finite⟩] := ⟨rfl, rfl, rfl, rfl⟩

private def row := LogAdmission.fp32_log_exp_guarded

theorem guarded_candidate_keeps_the_reference :
    (Atom.log_exp_cancel .libdevice).lhs = (Atom.log_exp .libdevice .libdevice).lhs ∧
    (Atom.log_exp_cancel .libdevice).lhs = originalLogExp ∧
    (Atom.log_exp_cancel .libdevice).rhs = piecewiseLogExp := ⟨rfl, rfl, rfl⟩

theorem wrong_precision_is_unavailable :
    (Atom.log_exp_cancel .libdevice).matches { row with report := { row.report with input := "bf16" } } = Bool.false ∧
    (Atom.log_exp_cancel .libdevice).matches { row with report := { row.report with compute := "fp64" } } = Bool.false ∧
    (Atom.log_exp_cancel .libdevice).matches { row with report := { row.report with accumulator := "fp64" } } = Bool.false ∧
    (Atom.log_exp_cancel .libdevice).matches { row with report := { row.report with output := "bf16" } } = Bool.false := by
  decide

theorem wrong_relation_or_domain_is_unavailable :
    (Atom.log_exp .tl .tl).matches row = Bool.false ∧
    (Atom.log_exp .tl .libdevice).matches row = Bool.false ∧
    (Atom.log_exp .libdevice .libdevice).matches row = Bool.false ∧
    (Atom.log_exp_cancel .libdevice).matches { row with guards := [⟨"a", .positive⟩] } = Bool.false := by
  decide

theorem selected_piecewise (R : Rules) :
    [(Atom.log_exp_cancel .libdevice).lhs] ≡[R] [(Atom.log_exp_cancel .libdevice).rhs] :=
  FP.LogExp.rewrite R (.log_exp_cancel .libdevice) (by decide)

#print_fp_assumptions selected_piecewise

theorem selected_product_split (R : Rules) :
    [(Atom.log_mul_split .libdevice).lhs] ≡[R] [(Atom.log_mul_split .libdevice).rhs] :=
  FP.LogExp.rewrite R (.log_mul_split .libdevice) (by decide)

#print_fp_assumptions selected_product_split

-- Renaming the public atom keeps the original experimental identity and all
-- four precision fields. A conditional report cannot select a plain log rule.
private def productRow := LogAdmission.fp32_log_mul_guarded

theorem split_report_identity_and_selection :
    (Atom.log_mul_split .libdevice).ruleID = "LOG-MUL-GUARDED" ∧
    (Atom.log_mul_split .libdevice).matches productRow = Bool.true ∧
    (Atom.log_mul .tl).matches productRow = Bool.false ∧
    (Atom.log_mul .libdevice).matches productRow = Bool.false := by decide

theorem split_wrong_precision_or_domain :
    (Atom.log_mul_split .libdevice).matches { productRow with report := { productRow.report with input := "bf16" } } = Bool.false ∧
    (Atom.log_mul_split .libdevice).matches { productRow with report := { productRow.report with compute := "fp64" } } = Bool.false ∧
    (Atom.log_mul_split .libdevice).matches { productRow with report := { productRow.report with accumulator := "fp64" } } = Bool.false ∧
    (Atom.log_mul_split .libdevice).matches { productRow with report := { productRow.report with output := "bf16" } } = Bool.false ∧
    (Atom.log_mul_split .libdevice).matches { productRow with guards := [] } = Bool.false := by decide

-- A reusable conditional theorem can be written before this candidate passes.
theorem log_mul_when_selected (R : Rules) (h : (Atom.log_mul .tl).Available) :
    [(Atom.log_mul .tl).lhs] ≡[R] [(Atom.log_mul .tl).rhs] :=
  FP.LogExp.rewrite R (.log_mul .tl) h

#axiomsClean FP.LogExp.derive
#axiomsClean FP.LogExp.rewrite
#axiomsClean FP.LogExp.apply_log_mul_split
#axiomsClean selected_product_split
#axiomsClean log_mul_when_selected

end FPLogCandidatesTests
