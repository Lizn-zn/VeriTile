import bench.tritonbench_g.context_attn_nopad.Proofs.StatsExecution
import bench.tritonbench_g.context_attn_nopad.Proofs.NormalizerExecution
import bench.tritonbench_g.context_attn_nopad.Proofs.FormulaStep
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.NumericCells
open VeriTile.Triton Formula
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericOps
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 5000

theorem exp_some (x : WithBot ℝ) :
    WithBot.realExp x = some ((WithBot.realExp x).unbotD 0) := by
  cases x <;> rfl

theorem maximum_bounds (p : Layout) (scores : Fin p.BM → Fin p.BN → WithBot ℝ)
    (qk : Tile .real [p.BM,p.BN]) (rm : Tile .real [p.BM]) (hBN : 0 < p.BN)
    (hq : ∀ i j, qk.data (i,j,PUnit.unit) = scores i j)
    (hr : Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) qk = some rm)
    (i : Fin p.BM) (j : Fin p.BN) : scores i j ≤ rm.data (i,PUnit.unit) := by
  rw [ctxg_reduceMaxDrop_data_row hBN qk rm hr i (scores i) (hq i)]
  exact Finset.le_sup (Finset.mem_univ j)

theorem bottom_center (p : Layout) (scores : Fin p.BM → Fin p.BN → WithBot ℝ)
    (qk : Tile .real [p.BM,p.BN]) (rm : Tile .real [p.BM]) (hBN : 0 < p.BN)
    (hq : ∀ i j, qk.data (i,j,PUnit.unit) = scores i j)
    (hr : Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) qk = some rm)
    (i : Fin p.BM) (j : Fin p.BN) (h : rm.data (i,PUnit.unit)=⊥) : scores i j=⊥ := by
  have hb := maximum_bounds p scores qk rm hBN hq hr i j
  rw [h] at hb
  exact le_bot_iff.mp hb

theorem exponentials_cell (p : Layout) (qk : Tile .real [p.BM,p.BN]) (center : Tile .real [p.BM])
    (i : Fin p.BM) (j : Fin p.BN) (s c : WithBot ℝ)
    (hq : qk.data (i,j,PUnit.unit) = s) (hc : center.data (i,PUnit.unit) = c) :
    (StatsExecution.exponentials p qk center).data (i,j,PUnit.unit) =
      some (NormalizedStep.shifted s c) := by
  change WithBot.realExp _ = _
  simp only [Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.sub,
    Tile.expandDim,Tile.remap,TileShape.dropInsertedIndex]
  rw [hq,hc]
  exact exp_some _

theorem denominator_cell (p : Layout) (qk : Tile .real [p.BM,p.BN]) (center : Tile .real [p.BM])
    (i : Fin p.BM) (scores : Fin p.BN → WithBot ℝ) (c : WithBot ℝ)
    (hq : ∀ j, qk.data (i,j,PUnit.unit) = scores j) (hc : center.data (i,PUnit.unit)=c) :
    (StatsExecution.denominator p qk center).data (i,PUnit.unit) =
      some (∑ j,NormalizedStep.shifted (scores j) c) := by
  unfold StatsExecution.denominator
  simp only [Tile.reduceSumDrop,Tile.reduceSum,TileShape.eraseAxis,TileShape.insertAxisIndex]
  change (∑ j : Fin p.BN, (StatsExecution.exponentials p qk center).data (i,j,PUnit.unit)) = _
  simp_rw [exponentials_cell p qk center i _ _ c (hq _) hc]
  exact ctxg_withBot_sum_some _

/-- The first visited block has the unmasked key zero; later iterations already
have a finite running center. Thus an all-masked block never leaves it bottom. -/
theorem next_finite (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (c : Nat)
    (m : Fin p.BM → WithBot ℝ) (hbot : ∀ i, m i=⊥ → c=0)
    (qk : Tile .real [p.BM,p.BN]) (rm : Tile .real [p.BM]) (hBN : 0 < p.BN)
    (hq : ∀ i j, qk.data (i,j,PUnit.unit) = FormulaStep.maskedScore p l a scale i.val (c*p.BN+j.val))
    (hr : Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) qk = some rm)
    (i : Fin p.BM) : m i ⊔ rm.data (i,PUnit.unit) ≠ ⊥ := by
  intro h
  have hm : m i=⊥ := le_bot_iff.mp (h ▸ (le_sup_left : m i ≤ m i ⊔ rm.data (i,PUnit.unit)))
  have hc := hbot i hm
  have hb := maximum_bounds p _ qk rm hBN hq hr i ⟨0,hBN⟩
  have hrbot : rm.data (i,PUnit.unit)=⊥ := le_bot_iff.mp (h ▸ (le_sup_right : rm.data (i,PUnit.unit) ≤ m i ⊔ rm.data (i,PUnit.unit)))
  rw [hrbot] at hb
  simp only [hc,zero_mul,zero_add,FormulaStep.maskedScore,Nat.zero_le,if_true] at hb
  exact WithBot.coe_ne_bot (le_bot_iff.mp hb)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.NumericCells
