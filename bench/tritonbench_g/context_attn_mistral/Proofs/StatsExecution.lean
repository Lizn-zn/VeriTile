import bench.tritonbench_g.context_attn_mistral.Proofs.LoopProgram
import bench.tritonbench_g.context_attn_bloom.Proofs.NumericOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.StatsExecution
open VeriTile.Triton Formula Program PreludeProgram LoopProgram
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericOps
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

private theorem eval_equality (h : ComparableDType dtype) (bc : Broadcast a b out)
    (lhs : Op dtype a) (rhs : Op dtype b) (s : BlockState) :
    evalOp (.eq h bc lhs rhs) s = (do
      let x ← evalOp lhs s
      let y ← evalOp rhs s
      pure (Tile.cop h.eq bc x y)) := by rw [evalOp.eq_def] <;> rfl


private theorem eval_broadcast (e : Op dtype []) (shape : TileShape) (s : BlockState) :
    evalOp (.broadcast e shape) s = (do
      let v ← evalOp e s
      pure (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype shape)) := by rw [evalOp.eq_def] <;> rfl

def center (p : Layout) (maximum : Tile .real [p.BM]) :=
  Tile.select (Tile.cop ComparableDType.real.eq Broadcast.scalarR maximum
    (Tile.scalar (some (-1000000000 : ℝ)))) (Tile.vec (fun _ => some 0)) maximum

def exponentials (p : Layout) (qk : Tile .real [p.BM,p.BN]) (maximum : Tile .real [p.BM]) :=
  Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame
    qk (Tile.expandDim ⟨1,by simp⟩ maximum))
def denominator (p : Layout) (qk : Tile .real [p.BM,p.BN]) (maximum : Tile .real [p.BM]) :=
  Tile.reduceSumDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) (exponentials p qk maximum)
def prepared (p : Layout) (qk : Tile .real [p.BM,p.BN]) (maximum : Tile .real [p.BM]) (u : BlockState) :=
  let u := u.setReg "m_ij" .real [p.BM] maximum
  let u := u.setReg "m_ij" .real [p.BM] (center p maximum)
  let u := u.setReg "p" .real [p.BM,p.BN] (exponentials p qk (center p maximum))
  u.setReg "l_ij" .real [p.BM] (denominator p qk (center p maximum))

theorem executes (p : Layout)
    (qk : Tile .real [p.BM,p.BN]) (maximum : Tile .real [p.BM]) (u : BlockState)
    (hq : u.regs .real [p.BM,p.BN] "qk" = some qk)
    (hm : Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) qk = some maximum) :
    stepStmts (statsCode p) u = some (prepared p qk maximum u) := by
  let s1 := u.setReg "m_ij" .real [p.BM] maximum
  let sguard := s1.setReg "m_ij" .real [p.BM] (center p maximum)
  let s2 := sguard.setReg "p" .real [p.BM,p.BN] (exponentials p qk (center p maximum))
  have h1 := bloomMij_eval u qk maximum hq hm
  have hs : evalOp sentinel s1 = some (Tile.scalar (dtype := .real) (some (-1000000000 : ℝ))) := by
    norm_num [sentinel,evalOp.eq_def,Tile.bop,Tile.scalar,NumericDType.sub,WithBot.realSub,
      Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_eq_bind,Option.bind,-Tile.scalar_eta]
  have hg : evalOp (.where (.eq .real .scalarR (.ref .real [p.BM] "m_ij") sentinel)
      (.broadcast (.const (0.0 : ℝ)) [p.BM]) (.ref .real [p.BM] "m_ij")) s1 = some (center p maximum) := by
    rw [evalOp_where,eval_equality,evalOp_ref,show s1.regs .real [p.BM] "m_ij" = some maximum by simp [s1]]
    simp only [hs,Option.bind_some,eval_broadcast,evalOp_const,evalOp_ref,
      show s1.regs .real [p.BM] "m_ij" = some maximum by simp [s1]]
    norm_num only [show (0.0 : ℝ) = 0 by norm_num]
    rfl
  have h2 := bloomP_eval sguard qk (center p maximum) (by simp [sguard,s1,hq]) (by simp [sguard])
  change evalOp _ sguard = some (exponentials p qk (center p maximum)) at h2
  have h3 := bloomLij_eval s2 (exponentials p qk (center p maximum)) (by simp [s2])
  change evalOp _ s2 = some (denominator p qk (center p maximum)) at h3
  unfold statsCode mat column vec
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (shape := [p.BM]) (name := "m_ij") h1)]
  change stepStmts _ s1 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hg)]
  change stepStmts _ sguard = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h2)]
  change stepStmts _ s2 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (shape := [p.BM]) (name := "l_ij") h3)]
  simp only [stepStmts,prepared,s1,sguard,s2]

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
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.StatsExecution
