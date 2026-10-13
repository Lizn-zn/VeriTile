import bench.tritonbench_g.rmsnorm_implementation.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation
open VeriTile.Triton

/-- Positive epsilon makes the independent full-row standard deviation nonzero;
the real contract itself keeps Lean's total division/square-root conventions. -/
theorem stream_std_pos (N B : Nat) (eps : ℝ)
    (xs : Fin (rmsNumSteps N B) → Fin B → ℝ) (heps : 0 < eps) :
    0 < Real.sqrt (rmsImplStreamSumSq N B xs / (N : ℝ) + eps) := by
  apply Real.sqrt_pos.2
  have hsum : 0 ≤ rmsImplStreamSumSq N B xs := by
    unfold rmsImplStreamSumSq
    exact Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => by
      split_ifs <;> positivity
  have hmean : 0 ≤ rmsImplStreamSumSq N B xs / (N : ℝ) := div_nonneg hsum (Nat.cast_nonneg N)
  linarith

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormImplementation
