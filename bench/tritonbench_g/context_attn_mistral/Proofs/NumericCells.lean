import bench.tritonbench_g.context_attn_mistral.Proofs.StatsExecution
import bench.tritonbench_g.context_attn_mistral.Proofs.NormalizerExecution
import bench.tritonbench_g.context_attn_mistral.Proofs.FormulaStep
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.NumericCells
open VeriTile.Triton Formula
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericOps
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 5000

theorem maximum_finite (p : Layout) (scores : Fin p.BM → Fin p.BN → ℝ)
    (qk : Tile .real [p.BM,p.BN]) (rm : Tile .real [p.BM])
    (hBN : 0 < p.BN)
    (hq : ∀ i j, qk.data (i,j,PUnit.unit) = some (scores i j))
    (hr : Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) qk = some rm)
    (i : Fin p.BM) : rm.data (i,PUnit.unit) ≠ ⊥ := by
  rw [ctxg_reduceMaxDrop_data_row hBN qk rm hr i (fun j => ((scores i j : ℝ) : WithBot ℝ)) (hq i)]
  intro h
  have hz := Finset.le_sup (f := fun j : Fin p.BN => ((scores i j : ℝ) : WithBot ℝ))
    (Finset.mem_univ (⟨0,hBN⟩ : Fin p.BN))
  rw [h] at hz
  exact absurd (le_bot_iff.mp hz) (by simp)

theorem center_finite (p : Layout) (rm : Tile .real [p.BM]) (i : Fin p.BM)
    (h : rm.data (i,PUnit.unit) ≠ ⊥) :
    ∃ center : ℝ, (StatsExecution.center p rm).data (i,PUnit.unit) = some center := by
  obtain ⟨x,hx⟩ := WithBot.ne_bot_iff_exists.mp h
  unfold StatsExecution.center
  rw [Tile.select_data]
  split_ifs
  · exact ⟨0,rfl⟩
  · exact ⟨x,hx.symm⟩

theorem exponentials_cell (p : Layout) (qk : Tile .real [p.BM,p.BN]) (center : Tile .real [p.BM])
    (i : Fin p.BM) (j : Fin p.BN) (s c : ℝ)
    (hq : qk.data (i,j,PUnit.unit) = some s)
    (hc : center.data (i,PUnit.unit) = some c) :
    (StatsExecution.exponentials p qk center).data (i,j,PUnit.unit) = some (Real.exp (s-c)) := by
  change WithBot.realExp _ = _
  rw [ctx_qk_sub_mij_cell qk center i j s c hq hc]
  rfl

theorem denominator_cell (p : Layout) (qk : Tile .real [p.BM,p.BN]) (center : Tile .real [p.BM])
    (i : Fin p.BM) (scores : Fin p.BN → ℝ) (c : ℝ)
    (hq : ∀ j, qk.data (i,j,PUnit.unit) = some (scores j))
    (hc : center.data (i,PUnit.unit) = some c) :
    (StatsExecution.denominator p qk center).data (i,PUnit.unit) =
      some (NormalizedStep.blockSum scores c) := by
  unfold StatsExecution.denominator
  simp only [Tile.reduceSumDrop,Tile.reduceSum,TileShape.eraseAxis,TileShape.insertAxisIndex]
  change (∑ j : Fin p.BN, (StatsExecution.exponentials p qk center).data (i,j,PUnit.unit)) = _
  simp_rw [exponentials_cell p qk center i _ _ c (hq _) hc]
  exact ctxg_withBot_sum_some _

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.NumericCells
