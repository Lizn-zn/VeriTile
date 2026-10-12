import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealMath

/-! A last chunk of length three with block width two. Padded inputs are
deliberately nonzero, so forgetting either contraction-tail mask changes the
answer. The specification includes the residual only when enabled. -/
namespace BmmChunkBwdMathTests
open VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealMath
noncomputable section

private def dout (k : Nat) : ℝ := if k = 0 then 7 else if k = 1 then 11 else if k = 2 then 13 else 1000
private def a (k : Nat) : ℝ := if k = 0 then 2 else if k = 1 then 3 else if k = 2 then 5 else 1000

theorem partial_last_chunk : chunkLimit 7 4 1 = 3 ∧ blocks (chunkLimit 7 4 1) 2 = 2 := by decide

theorem full_sum_with_and_without_residual :
    formula (chunkLimit 7 4 1) dout a 17 Bool.false = 112 ∧
    formula (chunkLimit 7 4 1) dout a 17 Bool.true = 129 := by
  norm_num [formula, chunkLimit, Finset.sum_range_succ, dout, a]

theorem masked_blocks_recover_three_terms :
    partialSum (chunkLimit 7 4 1) 2 2 (fun k => dout k * a k) = 112 := by
  change partialSum 3 2 (blocks 3 2) (fun k => dout k * a k) = 112
  rw [partial_complete 3 2 (by decide)]
  norm_num [Finset.sum_range_succ, dout, a]

theorem empty_chunk_retains_only_residual :
    formula 0 dout a 17 Bool.false = 0 ∧ formula 0 dout a 17 Bool.true = 17 := by
  simp [formula]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BmmChunkBwdMathTests
