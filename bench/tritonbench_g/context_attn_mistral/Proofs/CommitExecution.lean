import bench.tritonbench_g.context_attn_mistral.Proofs.LoopProgram
import bench.tritonbench_g.context_attn_bloom.Proofs.NumericOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.CommitExecution
open VeriTile.Triton Formula Program PreludeProgram LoopProgram
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericOps
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def accumulator (p : Layout) (prob : Tile .real [p.BM,p.BN]) (value : Tile .real [p.BN,p.D]) (acc : Tile .real [p.BM,p.D]) :=
  Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame acc (Tile.dot [] prob value)
def prepared (p : Layout) (prob : Tile .real [p.BM,p.BN]) (value : Tile .real [p.BN,p.D]) (acc : Tile .real [p.BM,p.D])
    (li mi : Tile .real [p.BM]) (u : BlockState) :=
  let u := u.setReg "p" .real [p.BM,p.BN] prob
  let u := u.setReg "acc" .real [p.BM,p.D] (accumulator p prob value acc)
  let u := u.setReg "l_i" .real [p.BM] li
  u.setReg "m_i" .real [p.BM] mi

theorem executes (p : Layout)
    (prob : Tile .real [p.BM,p.BN]) (value : Tile .real [p.BN,p.D]) (acc : Tile .real [p.BM,p.D])
    (li mi : Tile .real [p.BM]) (u : BlockState)
    (hp : u.regs .real [p.BM,p.BN] "p" = some prob)
    (hv : u.regs .real [p.BN,p.D] "v" = some value)
    (ha : u.regs .real [p.BM,p.D] "acc" = some acc)
    (hl : u.regs .real [p.BM] "l_i_new" = some li)
    (hm : u.regs .real [p.BM] "m_i_new" = some mi) :
    stepStmts (commitCode p) u = some (prepared p prob value acc li mi u) := by
  let s1 := u.setReg "p" .real [p.BM,p.BN] prob
  let s2 := s1.setReg "acc" .real [p.BM,p.D] (accumulator p prob value acc)
  let s3 := s2.setReg "l_i" .real [p.BM] li
  have h1 : evalOp (.ref .real [p.BM,p.BN] "p") u = some prob := by simp [hp]
  have h2 := bloomAcc2_eval s1 acc prob value (by simp [s1,ha]) (by simp [s1]) (by simp [s1,hv])
  change evalOp _ s1 = some (accumulator p prob value acc) at h2
  have h3 : evalOp (.ref .real [p.BM] "l_i_new") s2 = some li := by simp [s2,s1,hl]
  have h4 : evalOp (.ref .real [p.BM] "m_i_new") s3 = some mi := by simp [s3,s2,s1,hm]
  unfold commitCode mat vec
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h1)]
  change stepStmts _ s1 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h2)]
  change stepStmts _ s2 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h3)]
  change stepStmts _ s3 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h4)]
  simp only [stepStmts,prepared,s1,s2,s3]

theorem memory (p : Layout) (prob : Tile .real [p.BM,p.BN]) (value : Tile .real [p.BN,p.D]) (acc : Tile .real [p.BM,p.D])
    (li mi : Tile .real [p.BM]) (u : BlockState) :
    (prepared p prob value acc li mi u).mem = u.mem := by
  funext region off
  simp only [prepared,BlockState.setReg_mem]

theorem registers (p : Layout) (prob : Tile .real [p.BM,p.BN]) (value : Tile .real [p.BN,p.D]) (acc : Tile .real [p.BM,p.D])
    (li mi : Tile .real [p.BM]) (u : BlockState) :
    let next := prepared p prob value acc li mi u
    next.regs .real [p.BM] "m_i" = some mi ∧ next.regs .real [p.BM] "l_i" = some li ∧
      next.regs .real [p.BM,p.D] "acc" = some (accumulator p prob value acc) := by
  simp [prepared]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.CommitExecution
