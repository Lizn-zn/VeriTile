import VeriTile.Triton.Float.LogExp
import VeriTile.Triton.Float.Exponential
import VeriTile.Triton.Float.CountConversion
import bench.examples.FloatDTypeAddFPEquiv
import bench.examples.HyperConnectionsDepthFPEquiv
import bench.examples.TritonBenchVectorAdditionFPEquiv
import bench.examples.AdamUpdateGridLaunchFPEquiv
import bench.examples.HyperConnectionsWidthFPEquiv
import bench.examples.VectorAddFPEquiv
import bench.examples.FlatVectorAddFPEquiv

/- These checks detect changes in selected report metadata. They do not prove
numerical relations or supply evidence to the kernel equivalence proofs. -/
namespace FPAdmissionMetadataTests
open VeriTile.Triton.FP VeriTile.Bench.Examples

def fp32Rule (r : ReportedRule) (ruleID : String) : Prop :=
  r.ruleID = ruleID ∧ r.input = "fp32" ∧ r.compute = "fp32" ∧
  r.accumulator = "fp32" ∧ r.output = "fp32"

example : fp32Rule FloatDTypeAddFPEquiv.admitted "ADD-COMMUTE" := by unfold fp32Rule; decide
example : fp32Rule HyperConnectionsDepthFPEquiv.admitted "ADD-COMMUTE" := by unfold fp32Rule; decide
example : fp32Rule TritonBenchVectorAdditionFPEquiv.admitted "ADD-COMMUTE" := by unfold fp32Rule; decide
example : fp32Rule AdamUpdateGridLaunchFPEquiv.admitted "ADD-COMMUTE" := by unfold fp32Rule; decide
example : fp32Rule HyperConnectionsWidthFPEquiv.admitted "MUL-COMMUTE" := by unfold fp32Rule; decide
example : fp32Rule VectorAddFPEquiv.admitted "ADD-COMMUTE" := by unfold fp32Rule; decide
example : fp32Rule FlatVectorAddFPEquiv.admitted "ADD-COMMUTE" := by unfold fp32Rule; decide

theorem log_exp_report_matches :
    LogAdmission.fp32_log_exp_expm1.report.ruleID = "LOG-EXP-EXPM1" ∧
    LogAdmission.fp32_log_exp_expm1.report.input = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.report.compute = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.report.accumulator = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.report.output = "fp32" ∧
    LogAdmission.fp32_log_exp_expm1.guards = VeriTile.Triton.FP.LogExp.guards := by decide

theorem exponential_report_matches :
    SupplementalAdmission.fp32_exp_sub.report.ruleID = "EXP-SUB" ∧
    SupplementalAdmission.fp32_exp_sub.report.input = "fp32" ∧
    SupplementalAdmission.fp32_exp_sub.report.compute = "fp32" ∧
    SupplementalAdmission.fp32_exp_sub.report.accumulator = "fp32" ∧
    SupplementalAdmission.fp32_exp_sub.report.output = "fp32" ∧
    SupplementalAdmission.fp32_exp_sub.guards = Exponential.guards := by decide

open CountConversion in
theorem count_report_matches (a : Atom) (h : a.Available) :
    (report a h).ruleID = (match a with | .zero => "COUNT-ZERO" | .successor => "COUNT-SUCCESSOR") ∧
    (report a h).input = "int32" ∧ (report a h).compute = "fp32" ∧
    (report a h).accumulator = "fp32" ∧ (report a h).output = "fp32" ∧
    0 < limit ∧ limit ≤ 2^24 ∧ limit < 2^31 := by
  cases a <;> exact ⟨rfl, rfl, rfl, rfl, rfl, by decide, by decide, by decide⟩

end FPAdmissionMetadataTests
