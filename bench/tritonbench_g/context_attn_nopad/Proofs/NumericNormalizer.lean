import bench.tritonbench_g.context_attn_nopad.Proofs.NumericCells
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.NumericNormalizer
open VeriTile.Triton Formula
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericOps
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 4000

theorem maximum_cell (p : Layout) (mi center : Tile .real [p.BM]) (i : Fin p.BM)
    (m c : WithBot ℝ) (hm : mi.data (i,PUnit.unit)=m) (hc : center.data (i,PUnit.unit)=c) :
    (NormalizerExecution.newMaximum p mi center).data (i,PUnit.unit) = m ⊔ c := by
  rw [NormalizerExecution.newMaximum,ctxg_mij_max mi center i m c hm hc]

theorem alpha_cell (p : Layout) (mi center : Tile .real [p.BM]) (i : Fin p.BM)
    (m c : WithBot ℝ) (next : ℝ)
    (hm : mi.data (i,PUnit.unit)=m) (hc : center.data (i,PUnit.unit)=c)
    (hn : m ⊔ c=some next) :
    (NormalizerExecution.alpha p mi center).data (i,PUnit.unit)=some (NormalizedStep.alpha m next) := by
  change WithBot.realExp _ = _
  simp only [Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.sub]
  rw [hm,maximum_cell p mi center i m c hm hc,hn]
  exact NumericCells.exp_some _

theorem beta_cell (p : Layout) (mi center : Tile .real [p.BM]) (i : Fin p.BM)
    (m c : WithBot ℝ) (next : ℝ)
    (hm : mi.data (i,PUnit.unit)=m) (hc : center.data (i,PUnit.unit)=c)
    (hn : m ⊔ c=some next) :
    (NormalizerExecution.beta p mi center).data (i,PUnit.unit)=some (NormalizedStep.alpha c next) := by
  change WithBot.realExp _ = _
  simp only [Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.sub]
  rw [hc,maximum_cell p mi center i m c hm hc,hn]
  exact NumericCells.exp_some _

theorem denominator_cell (p : Layout) (mi center li lij : Tile .real [p.BM]) (i : Fin p.BM)
    (m c : WithBot ℝ) (next l block : ℝ)
    (hm : mi.data (i,PUnit.unit)=m) (hc : center.data (i,PUnit.unit)=c) (hn : m ⊔ c=some next)
    (hl : li.data (i,PUnit.unit)=some l) (hb : lij.data (i,PUnit.unit)=some block) :
    (NormalizerExecution.newDenominator p mi center li lij).data (i,PUnit.unit)=
      some (NormalizedStep.alpha m next*l+NormalizedStep.alpha c next*block) := by
  simp only [NormalizerExecution.newDenominator,Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,
    alpha_cell p mi center i m c next hm hc hn,beta_cell p mi center i m c next hm hc hn,hl,hb]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.NumericNormalizer
