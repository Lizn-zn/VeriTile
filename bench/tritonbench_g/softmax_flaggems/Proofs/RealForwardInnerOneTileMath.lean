import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerMath
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerOneTileMath
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile
open VeriTile Triton
open scoped BigOperators

noncomputable def score (s : BlockState) (X : RegionName) (row N B : Nat)
    (i : TileIndex [B]) : WithBot ℝ :=
  if i.1.val<N then some (s.readMem X (row*N+i.1.val)) else ⊥

noncomputable def batchMax (s : BlockState) (X : RegionName) (row N B : Nat)
    (hB : 0<B) : WithBot ℝ := RealForwardInner.globalMax hB (score s X row N B)

noncomputable def batchSum (s : BlockState) (X : RegionName) (row N B : Nat)
    (hB : 0<B) : ℝ :=
  ∑ i : Fin B, (WithBot.realExp (WithBot.realSub (score s X row N B (i,PUnit.unit))
    (batchMax s X row N B hB))).unbotD 0

noncomputable def storeValue (s : BlockState) (X : RegionName) (row N B : Nat)
    (hB : 0<B) (i : TileIndex [B]) : ℝ :=
  (WithBot.realDiv (WithBot.realExp (WithBot.realSub (score s X row N B i)
    (batchMax s X row N B hB))) (some (batchSum s X row N B hB))).unbotD 0

theorem batch_denominator (s : BlockState) (X : RegionName) (row N B : Nat)
    (hN : 0<N) (hB : 0<B) (hNB : N≤B) :
    ∃ m : ℝ, batchMax s X row N B hB=some m ∧
      batchSum s X row N B hB = Real.exp (-m)*RealForwardInner.denominator s X row N := by
  simpa [batchMax, batchSum, score, RealForwardInner.globalMax, RealForwardInner.denominator,
    RealForwardNonInner.batchMax, RealForwardNonInner.batchSum, RealForwardNonInner.columnMax,
    RealForwardNonInner.score, RealForwardNonInner.denominator, Finset.sum_range] using
    RealForwardNonInner.batch_denominator s X row N 1 0 B 1 hN hB hNB
      ⟨0, by omega⟩ (by omega)

theorem storeValue_expected (s : BlockState) (X : RegionName) (row N B : Nat)
    (hN : 0<N) (hB : 0<B) (hNB : N≤B) (i : TileIndex [B]) (hi : i.1.val<N) :
    storeValue s X row N B hB i = RealForwardInner.expected s X row N i.1.val := by
  obtain ⟨m,hm,hz⟩ := batch_denominator s X row N B hN hB hNB
  unfold storeValue score RealForwardInner.expected
  rw [if_pos hi, hm, hz]
  exact RealForwardNonInner.stored_value _ _ _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile
