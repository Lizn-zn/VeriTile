import bench.tritonbench_g.context_attn_mistral.Proofs.ScalesExecution
import bench.tritonbench_g.context_attn_mistral.Proofs.CommitExecution
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.NumericScales
open VeriTile.Triton Formula
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericOps
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 4000

theorem probability_cell (p : Layout) (prob : Tile .real [p.BM,p.BN]) (beta den : Tile .real [p.BM])
    (i : Fin p.BM) (j : Fin p.BN) (v b l : ℝ)
    (hp : prob.data (i,j,PUnit.unit) = some v)
    (hb : beta.data (i,PUnit.unit) = some b) (hl : den.data (i,PUnit.unit) = some l) :
    (ScalesExecution.probabilities p prob beta den).data (i,j,PUnit.unit) = some (v*(b/l)) := by
  simp only [ScalesExecution.probabilities,ScalesExecution.probabilityScale,
    Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,Tile.expandDim_data,
    TileShape.dropInsertedIndex,TileShape.insertAxisIndex,hp,hb,hl]
  rfl

theorem accumulator_cell (p : Layout) (li den alpha : Tile .real [p.BM]) (acc : Tile .real [p.BM,p.D])
    (i : Fin p.BM) (d : Fin p.D) (l next a v : ℝ)
    (hl : li.data (i,PUnit.unit) = some l) (hn : den.data (i,PUnit.unit) = some next)
    (ha : alpha.data (i,PUnit.unit) = some a) (hv : acc.data (i,d,PUnit.unit) = some v) :
    (ScalesExecution.accumulator p li den alpha acc).data (i,d,PUnit.unit) = some (v*(l/next*a)) := by
  simp only [ScalesExecution.accumulator,ScalesExecution.rawScale,
    Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,Tile.expandDim_data,
    TileShape.dropInsertedIndex,TileShape.insertAxisIndex,hl,hn,ha,hv]
  rfl

theorem commit_cell (p : Layout) (prob : Tile .real [p.BM,p.BN]) (value : Tile .real [p.BN,p.D])
    (acc : Tile .real [p.BM,p.D]) (i : Fin p.BM) (d : Fin p.D)
    (weights values : Fin p.BN → ℝ) (before : ℝ)
    (hp : ∀ j, prob.data (i,j,PUnit.unit) = some (weights j))
    (hv : ∀ j, value.data (j,d,PUnit.unit) = some (values j))
    (ha : acc.data (i,d,PUnit.unit) = some before) :
    (CommitExecution.accumulator p prob value acc).data (i,d,PUnit.unit) =
      some (before+∑ j, weights j*values j) := by
  have hd := ctxg_dot_row prob value i d weights values hp hv
  simp only [CommitExecution.accumulator,Tile.bop_data,Broadcast.leftIndex,Broadcast.rightIndex,ha,hd]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.NumericScales
