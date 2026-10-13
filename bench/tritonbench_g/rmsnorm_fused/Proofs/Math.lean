import bench.tritonbench_g.rmsnorm_fused.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormFused
open VeriTile.Triton

/-- Positive epsilon makes the independent full-row standard deviation nonzero;
the real contract itself keeps Lean's total division/square-root conventions. -/
theorem stream_std_pos (N B : Nat) (eps : ℝ)
    (xs : Fin 1 → Fin B → ℝ) (heps : 0 < eps) :
    0 < Real.sqrt (rmsFusedStreamSumSq N B xs / (N : ℝ) + eps) := by
  apply Real.sqrt_pos.2
  have hsum : 0 ≤ rmsFusedStreamSumSq N B xs := by
    unfold rmsFusedStreamSumSq
    exact Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => by
      split_ifs <;> positivity
  have hmean : 0 ≤ rmsFusedStreamSumSq N B xs / (N : ℝ) := div_nonneg hsum (Nat.cast_nonneg N)
  linarith

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormFused
