import bench.tritonbench_g.context_attn_fwd.Proofs.LoopProgram
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.StatsExecution
open VeriTile.Triton Formula Program PreludeProgram LoopProgram NumericOps
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

def maximum (p : Layout) (mi rm : Tile .real [p.BM]) :=
  Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mi rm) mi rm
def shifted (p : Layout) (qk : Tile .real [p.BM,p.BN]) (mx : Tile .real [p.BM]) :=
  Tile.bop NumericDType.real.sub Broadcast.nil.consR.consSame qk (Tile.expandDim ⟨1,by simp⟩ mx)
def probabilities (p : Layout) (qk : Tile .real [p.BM,p.BN]) (mx : Tile .real [p.BM]) :=
  Tile.uop WithBot.realExp2 (shifted p qk mx)
def denominator (p : Layout) (qk : Tile .real [p.BM,p.BN]) (mx : Tile .real [p.BM]) :=
  Tile.reduceSumDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) (probabilities p qk mx)
def alpha (p : Layout) (mi mx : Tile .real [p.BM]) :=
  Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub Broadcast.nil.consSame mi mx)
def updatedDenominator (p : Layout) (li alpha lij : Tile .real [p.BM]) :=
  Tile.bop NumericDType.real.add Broadcast.nil.consSame
    (Tile.bop NumericDType.real.mul Broadcast.nil.consSame li alpha) lij
def scaledAcc (p : Layout) (acc : Tile .real [p.BM,p.D]) (alpha : Tile .real [p.BM]) :=
  Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acc (Tile.expandDim ⟨1,by simp⟩ alpha)

def prepared (p : Layout) (qk : Tile .real [p.BM,p.BN]) (mi li rm : Tile .real [p.BM])
    (acc : Tile .real [p.BM,p.D]) (u : BlockState) :=
  let mx := maximum p mi rm
  let a := alpha p mi mx
  let d := denominator p qk mx
  let u := u.setReg "m_ij" .real [p.BM] mx
  let u := u.setReg "qk" .real [p.BM,p.BN] (shifted p qk mx)
  let u := u.setReg "p" .real [p.BM,p.BN] (probabilities p qk mx)
  let u := u.setReg "l_ij" .real [p.BM] d
  let u := u.setReg "alpha" .real [p.BM] a
  let u := u.setReg "l_i" .real [p.BM] (updatedDenominator p li a d)
  u.setReg "acc" .real [p.BM,p.D] (scaledAcc p acc a)

theorem executes (p : Layout) (qk : Tile .real [p.BM,p.BN]) (mi li rm : Tile .real [p.BM])
    (acc : Tile .real [p.BM,p.D]) (u : BlockState)
    (hq : u.regs .real [p.BM,p.BN] "qk" = some qk)
    (hm : u.regs .real [p.BM] "m_i" = some mi)
    (hl : u.regs .real [p.BM] "l_i" = some li)
    (ha : u.regs .real [p.BM,p.D] "acc" = some acc)
    (hr : Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) qk = some rm) :
    stepStmts (stats p) u = some (prepared p qk mi li rm acc u) := by
  let mx := maximum p mi rm
  let s1 := u.setReg "m_ij" .real [p.BM] mx
  let s2 := s1.setReg "qk" .real [p.BM,p.BN] (shifted p qk mx)
  let s3 := s2.setReg "p" .real [p.BM,p.BN] (probabilities p qk mx)
  let s4 := s3.setReg "l_ij" .real [p.BM] (denominator p qk mx)
  let s5 := s4.setReg "alpha" .real [p.BM] (alpha p mi mx)
  let s6 := s5.setReg "l_i" .real [p.BM] (updatedDenominator p li (alpha p mi mx) (denominator p qk mx))
  have h1 := ctxMij_op_eval u mi rm qk hm hq hr
  have h2 := ctxQk2_op_eval s1 qk mx (by simp [s1,hq]) (by simp [s1])
  have h3 := ctxP_op_eval s2 (shifted p qk mx) (by simp [s2])
  have h4 := ctxLij_op_eval s3 (probabilities p qk mx) (by simp [s3])
  have h5 := ctxAlpha_op_eval s4 mi mx (by simp [s4,s3,s2,s1,hm]) (by simp [s4,s3,s2,s1])
  have h6 := ctxLi_op_eval s5 li (alpha p mi mx) (denominator p qk mx)
    (by simp [s5,s4,s3,s2,s1,hl]) (by simp [s5]) (by simp [s5,s4])
  have h7 := ctxAcc1_op_eval s6 acc (alpha p mi mx)
    (by simp [s6,s5,s4,s3,s2,s1,ha]) (by simp [s6,s5])
  unfold stats column vec mat
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (shape := [p.BM]) (name := "m_ij") h1)]
  change stepStmts _ s1 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h2)]
  change stepStmts _ s2 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h3)]
  change stepStmts _ s3 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (shape := [p.BM]) (name := "l_ij") h4)]
  change stepStmts _ s4 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h5)]
  change stepStmts _ s5 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h6)]
  change stepStmts _ s6 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h7)]
  simp only [stepStmts.nil,prepared,s6,s5,s4,s3,s2,s1,mx,scaledAcc]

theorem reduction_exists (p : Layout) (qk : Tile .real [p.BM,p.BN]) (hBN : 0 < p.BN) :
    ∃ rm : Tile .real [p.BM], Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) qk = some rm := by
  unfold Tile.reduceMaxDrop
  rw [dif_pos (show 0 < TileShape.axisDim [p.BM,p.BN] (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) from hBN)]
  exact ⟨_,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.StatsExecution
