import bench.tritonbench_g.context_attn_mistral.Proofs.NumericCells
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.NumericNormalizer
open VeriTile.Triton Formula
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericOps
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 5000

def nextCenter (m : WithBot ℝ) (center : ℝ) : ℝ := m.elim center (fun x => max x center)

theorem maximum_cell (p : Layout) (mi center : Tile .real [p.BM]) (i : Fin p.BM)
    (m : WithBot ℝ) (c : ℝ)
    (hm : mi.data (i,PUnit.unit) = m) (hc : center.data (i,PUnit.unit) = some c) :
    (NormalizerExecution.newMaximum p mi center).data (i,PUnit.unit) = some (nextCenter m c) := by
  rw [NormalizerExecution.newMaximum,ctxg_mij_max mi center i m (some c) hm hc]
  cases m with
  | bot => rfl
  | coe x =>
    change max (x : WithBot ℝ) (c : WithBot ℝ) = (max x c : ℝ)
    exact (WithBot.coe_max x c).symm

theorem alpha_cell (p : Layout) (mi center : Tile .real [p.BM]) (i : Fin p.BM)
    (m : WithBot ℝ) (c : ℝ)
    (hm : mi.data (i,PUnit.unit) = m) (hc : center.data (i,PUnit.unit) = some c) :
    (NormalizerExecution.alpha p mi center).data (i,PUnit.unit) = some (NormalizedStep.alpha m (nextCenter m c)) := by
  change WithBot.realExp _ = _
  simp only [Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.sub]
  rw [hm,maximum_cell p mi center i m c hm hc]
  cases m <;> rfl

theorem beta_cell (p : Layout) (mi center : Tile .real [p.BM]) (i : Fin p.BM)
    (m : WithBot ℝ) (c : ℝ)
    (hm : mi.data (i,PUnit.unit) = m) (hc : center.data (i,PUnit.unit) = some c) :
    (NormalizerExecution.beta p mi center).data (i,PUnit.unit) = some (Real.exp (c-nextCenter m c)) := by
  change WithBot.realExp _ = _
  simp only [Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.sub]
  rw [hc,maximum_cell p mi center i m c hm hc]
  rfl

theorem denominator_cell (p : Layout) (mi center li lij : Tile .real [p.BM]) (i : Fin p.BM)
    (m : WithBot ℝ) (c l block : ℝ)
    (hm : mi.data (i,PUnit.unit) = m) (hc : center.data (i,PUnit.unit) = some c)
    (hl : li.data (i,PUnit.unit) = some l) (hb : lij.data (i,PUnit.unit) = some block) :
    (NormalizerExecution.newDenominator p mi center li lij).data (i,PUnit.unit) =
      some (NormalizedStep.alpha m (nextCenter m c)*l+Real.exp (c-nextCenter m c)*block) := by
  simp only [NormalizerExecution.newDenominator,Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,
    alpha_cell p mi center i m c hm hc,beta_cell p mi center i m c hm hc,hl,hb]
  rfl

theorem guard_inactive (p : Layout) (mi center li lij : Tile .real [p.BM]) (i : Fin p.BM)
    (v : ℝ) (hv : 0 < v)
    (h : (NormalizerExecution.newDenominator p mi center li lij).data (i,PUnit.unit) = some v) :
    (NormalizerExecution.guardedDenominator p mi center li lij).data (i,PUnit.unit) = some v := by
  unfold NormalizerExecution.guardedDenominator
  rw [Tile.select_data,Tile.cop_data]
  simp only [Broadcast.leftIndex,Broadcast.rightIndex,Tile.scalar_data,h,ComparableDType.eq]
  rw [if_neg]
  intro he
  have he' : (some v : WithBot ℝ) = some (0.0 : ℝ) := of_decide_eq_true he
  have hz : v = (0.0 : ℝ) := WithBot.coe_inj.mp he'
  norm_num at hz
  exact (ne_of_gt hv) hz

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.NumericNormalizer
