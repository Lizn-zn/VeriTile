import bench.tritonbench_g.context_attn_nopad.Proofs.LoopProgram
import bench.tritonbench_g.context_attn_bloom.Proofs.NumericOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.ScalesExecution
open VeriTile.Triton Formula Program PreludeProgram LoopProgram
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericOps
noncomputable section
set_option maxHeartbeats 900000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def probabilityScale (p : Layout) (beta lin : Tile .real [p.BM]) :=
  Tile.bop NumericDType.real.div Broadcast.nil.consSame beta lin
def probabilities (p : Layout) (ptile : Tile .real [p.BM,p.BN]) (beta lin : Tile .real [p.BM]) :=
  Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame ptile
    (Tile.expandDim ⟨1,by simp⟩ (probabilityScale p beta lin))
def rawScale (p : Layout) (li lin alpha : Tile .real [p.BM]) :=
  Tile.bop NumericDType.real.mul Broadcast.nil.consSame
    (Tile.bop NumericDType.real.div Broadcast.nil.consSame li lin) alpha
def accumulator (p : Layout) (li lin alpha : Tile .real [p.BM])
    (acc : Tile .real [p.BM,p.D]) :=
  Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acc
    (Tile.expandDim ⟨1,by simp⟩ (rawScale p li lin alpha))
def prepared (p : Layout) (li lin alpha beta : Tile .real [p.BM])
    (ptile : Tile .real [p.BM,p.BN]) (acc : Tile .real [p.BM,p.D]) (u : BlockState) :=
  let u := u.setReg "p_scale" .real [p.BM] (probabilityScale p beta lin)
  let u := u.setReg "p" .real [p.BM,p.BN] (probabilities p ptile beta lin)
  let u := u.setReg "acc_scale" .real [p.BM] (rawScale p li lin alpha)
  u.setReg "acc" .real [p.BM,p.D] (accumulator p li lin alpha acc)

theorem executes (p : Layout)
    (li lin alpha beta : Tile .real [p.BM])
    (ptile : Tile .real [p.BM,p.BN]) (acc : Tile .real [p.BM,p.D]) (u : BlockState)
    (hli : u.regs .real [p.BM] "l_i" = some li)
    (hlin : u.regs .real [p.BM] "l_i_new" = some lin)
    (ha : u.regs .real [p.BM] "alpha" = some alpha)
    (hb : u.regs .real [p.BM] "beta" = some beta)
    (hpt : u.regs .real [p.BM,p.BN] "p" = some ptile)
    (hac : u.regs .real [p.BM,p.D] "acc" = some acc) :
    stepStmts (scalesCode p) u = some (prepared p li lin alpha beta ptile acc u) := by
  let s1 := u.setReg "p_scale" .real [p.BM] (probabilityScale p beta lin)
  let s2 := s1.setReg "p" .real [p.BM,p.BN] (probabilities p ptile beta lin)
  let s3 := s2.setReg "acc_scale" .real [p.BM] (rawScale p li lin alpha)
  have h1 := bloomPscale_eval u beta lin hb hlin
  change evalOp _ u = some (probabilityScale p beta lin) at h1
  have h2 := bloomP2_eval s1 ptile (probabilityScale p beta lin) (by simp [s1,hpt]) (by simp [s1])
  change evalOp _ s1 = some (probabilities p ptile beta lin) at h2
  have h3 := bloomAccScale1_eval s2 li lin alpha (by simp [s2,s1,hli]) (by simp [s2,s1,hlin]) (by simp [s2,s1,ha])
  change evalOp _ s2 = some (rawScale p li lin alpha) at h3
  have h5 := bloomAcc1_eval s3 acc (rawScale p li lin alpha)
    (by simp [s3,s2,s1,hac]) (by simp [s3])
  change evalOp _ s3 = some (accumulator p li lin alpha acc) at h5
  unfold scalesCode mat column vec
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h1)]
  change stepStmts _ s1 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h2)]
  change stepStmts _ s2 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h3)]
  change stepStmts _ s3 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h5)]
  simp only [stepStmts,prepared,s1,s2,s3]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.ScalesExecution
