import bench.tritonbench_g.context_attn_nopad.Proofs.LoopProgram
import bench.tritonbench_g.context_attn_bloom.Proofs.NumericOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.NormalizerExecution
open VeriTile.Triton Formula Program PreludeProgram LoopProgram
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.NumericOps
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def newMaximum (p : Layout) (mi mij : Tile .real [p.BM]) :=
  Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mi mij) mi mij
def alpha (p : Layout) (mi mij : Tile .real [p.BM]) :=
  Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mi (newMaximum p mi mij))
def beta (p : Layout) (mi mij : Tile .real [p.BM]) :=
  Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mij (newMaximum p mi mij))
def newDenominator (p : Layout) (mi mij li lij : Tile .real [p.BM]) :=
  Tile.bop NumericDType.real.add Broadcast.nil.consSame
    (Tile.bop NumericDType.real.mul Broadcast.nil.consSame (alpha p mi mij) li)
    (Tile.bop NumericDType.real.mul Broadcast.nil.consSame (beta p mi mij) lij)
def prepared (p : Layout) (mi mij li lij : Tile .real [p.BM]) (u : BlockState) :=
  let u := u.setReg "m_i_new" .real [p.BM] (newMaximum p mi mij)
  let u := u.setReg "alpha" .real [p.BM] (alpha p mi mij)
  let u := u.setReg "beta" .real [p.BM] (beta p mi mij)
  u.setReg "l_i_new" .real [p.BM] (newDenominator p mi mij li lij)

theorem executes (p : Layout)
    (mi mij li lij : Tile .real [p.BM]) (u : BlockState)
    (hmi : u.regs .real [p.BM] "m_i" = some mi)
    (hmij : u.regs .real [p.BM] "m_ij" = some mij)
    (hli : u.regs .real [p.BM] "l_i" = some li)
    (hlij : u.regs .real [p.BM] "l_ij" = some lij) :
    stepStmts (normalizerCode p) u = some (prepared p mi mij li lij u) := by
  let s1 := u.setReg "m_i_new" .real [p.BM] (newMaximum p mi mij)
  let s2 := s1.setReg "alpha" .real [p.BM] (alpha p mi mij)
  let s3 := s2.setReg "beta" .real [p.BM] (beta p mi mij)
  let s4 := s3.setReg "l_i_new" .real [p.BM] (newDenominator p mi mij li lij)
  have h1 := bloomMiNew_eval u mi mij hmi hmij
  change evalOp _ u = some (newMaximum p mi mij) at h1
  have h2 := bloomExpSub_eval s1 "m_i" "m_i_new" mi (newMaximum p mi mij)
    (by simp [s1,hmi]) (by simp [s1])
  change evalOp _ s1 = some (alpha p mi mij) at h2
  have h3 := bloomExpSub_eval s2 "m_ij" "m_i_new" mij (newMaximum p mi mij)
    (by simp [s2,s1,hmij]) (by simp [s2,s1])
  change evalOp _ s2 = some (beta p mi mij) at h3
  have h4 := bloomLiNew_eval s3 (alpha p mi mij) li (beta p mi mij) lij
    (by simp [s3,s2]) (by simp [s3,s2,s1,hli]) (by simp [s3]) (by simp [s3,s2,s1,hlij])
  change evalOp _ s3 = some (newDenominator p mi mij li lij) at h4
  unfold normalizerCode vec
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h1)]
  change stepStmts _ s1 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h2)]
  change stepStmts _ s2 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h3)]
  change stepStmts _ s3 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h4)]
  simp only [stepStmts,prepared,s1,s2,s3,s4]

theorem memory (p : Layout) (mi mij li lij : Tile .real [p.BM]) (u : BlockState) :
    (prepared p mi mij li lij u).mem = u.mem := by
  funext region off
  simp only [prepared,BlockState.setReg_mem]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.NormalizerExecution
