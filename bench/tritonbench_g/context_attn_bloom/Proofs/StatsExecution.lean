import bench.tritonbench_g.context_attn_bloom.Proofs.UpdateProgram
import bench.tritonbench_g.context_attn_bloom.Proofs.NumericOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.StatsExecution
open VeriTile.Triton Math Program RealSetup PreludeProgram UpdateProgram NumericOps
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def exponentials (p : Layout) (qk : Tile .real [p.BM,p.BN]) (maximum : Tile .real [p.BM]) :=
  Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
    qk (Tile.expandDim ⟨1,by simp⟩ maximum))
def denominator (p : Layout) (qk : Tile .real [p.BM,p.BN]) (maximum : Tile .real [p.BM]) :=
  Tile.reduceSumDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) (exponentials p qk maximum)
def prepared (p : Layout) (qk : Tile .real [p.BM,p.BN]) (maximum : Tile .real [p.BM]) (u : BlockState) :=
  let u := u.setReg "m_ij" .real [p.BM] maximum
  let u := u.setReg "p" .real [p.BM,p.BN] (exponentials p qk maximum)
  u.setReg "l_ij" .real [p.BM] (denominator p qk maximum)

theorem executes (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (qk : Tile .real [p.BM,p.BN]) (maximum : Tile .real [p.BM]) (u : BlockState)
    (hq : u.regs .real [p.BM,p.BN] "qk" = some qk)
    (hm : Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) qk = some maximum) :
    stepStmts (stats p r swapped scale) u = some (prepared p qk maximum u) := by
  let s1 := u.setReg "m_ij" .real [p.BM] maximum
  let s2 := s1.setReg "p" .real [p.BM,p.BN] (exponentials p qk maximum)
  have h1 := bloomMij_eval u qk maximum hq hm
  have h2 := bloomP_eval s1 qk maximum (by simp [s1,hq]) (by simp [s1])
  change evalOp _ s1 = some (exponentials p qk maximum) at h2
  have h3 := bloomLij_eval s2 (exponentials p qk maximum) (by simp [s2])
  change evalOp _ s2 = some (denominator p qk maximum) at h3
  rw [(sources p r swapped scale).1]
  unfold statsCode mat column vec
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (shape := [p.BM]) (name := "m_ij") h1)]
  change stepStmts _ s1 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h2)]
  change stepStmts _ s2 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (shape := [p.BM]) (name := "l_ij") h3)]
  simp only [stepStmts,prepared,s1,s2]

theorem reduction_exists (p : Layout) (qk : Tile .real [p.BM,p.BN]) (hBN : 0 < p.BN) :
    ∃ maximum : Tile .real [p.BM],
      Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) qk = some maximum := by
  unfold Tile.reduceMaxDrop
  rw [dif_pos (show 0 < TileShape.axisDim [p.BM,p.BN] (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) from hBN)]
  exact ⟨_,rfl⟩

theorem memory (p : Layout) (qk : Tile .real [p.BM,p.BN]) (maximum : Tile .real [p.BM]) (u : BlockState) :
    (prepared p qk maximum u).mem = u.mem := by
  funext region off
  simp only [prepared,BlockState.setReg_mem]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.StatsExecution
