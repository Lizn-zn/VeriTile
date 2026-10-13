import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerScatter

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton
open scoped BigOperators

theorem laneState_one (s : BlockState) (X : RegionName) (row N K kbase B C : Nat)
    (i : TileIndex [B, C]) :
    laneState s X row N K kbase B C 1 i =
      (score s X row N K kbase 0 B C i,
        if i.1.val < N ∧ kbase + i.2.1.val < K then 1 else 0) := by
  by_cases hk : kbase + i.2.1.val < K <;> by_cases hn : i.1.val < N <;>
    simp [laneState, score, RealForwardModel.state, RealForwardModel.values,
      RealMath.onlineStep, hk, hn] <;> rfl

noncomputable def batchMax (s : BlockState) (X : RegionName)
    (row N K kbase B C : Nat) (hB : 0 < B) : Fin C → WithBot ℝ :=
  columnMax hB (score s X row N K kbase 0 B C)

noncomputable def batchSum (s : BlockState) (X : RegionName)
    (row N K kbase B C : Nat) (hB : 0 < B) (k : Fin C) : ℝ :=
  ∑ n : Fin B, (WithBot.realExp (WithBot.realSub
    (score s X row N K kbase 0 B C (n, k, PUnit.unit))
    (batchMax s X row N K kbase B C hB k))).unbotD 0

theorem batch_denominator (s : BlockState) (X : RegionName)
    (row N K kbase B C : Nat) (hN : 0 < N) (hB : 0 < B) (hcover : N ≤ B)
    (k : Fin C) (hk : kbase + k.val < K) :
    ∃ m : ℝ, batchMax s X row N K kbase B C hB k = some m ∧
      batchSum s X row N K kbase B C hB k =
        Real.exp (-m) * denominator s X row N K (kbase + k.val) := by
  obtain ⟨m, hm, hz⟩ := denominator_closed s X row N K kbase B C 1 hN hB
    (by omega) (by simpa only [Nat.one_mul] using hcover) k hk
  simp only [laneState_one] at hm hz
  refine ⟨m, hm, ?_⟩
  rw [← hz]
  unfold batchSum columnSum
  apply Finset.sum_congr rfl
  intro n _
  by_cases hn : n.val < N
  · simp [batchMax, hn, hk]
  · simp only [score, Nat.zero_add, hn, hk, false_and, and_true, if_false, zero_mul]
    rfl

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
