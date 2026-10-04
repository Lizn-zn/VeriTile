import VeriTile.Triton.Float.ScalarArithmetic

namespace FPScalarAdmissionTests
open VeriTile.Triton.FP.ScalarArithmetic

/-- Report refreshes must preserve the selected relation, precision and domain.
An accepted bf16 output cast cannot supply a bare fp32 law. -/
theorem report_matches (a : Atom) :
    (report a).report.ruleID = a.ruleID ∧
    (report a).report.input = "fp32" ∧ (report a).report.compute = "fp32" ∧
    (report a).report.accumulator = "fp32" ∧ (report a).report.output = "fp32" ∧
    (report a).guards = guards a := by
  cases a <;> decide

end FPScalarAdmissionTests
