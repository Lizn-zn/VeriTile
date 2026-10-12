import bench.tritonbench_g.context_attn_bloom.Proofs.UpdateProgram
import bench.tritonbench_g.context_attn_bloom.Proofs.NumericOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.ScalesExecution
open VeriTile.Triton Math Program RealSetup PreludeProgram UpdateProgram NumericOps
noncomputable section
set_option maxHeartbeats 900000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def guard (p : Layout) (rows : Fin p.BM → Nat) (prompt sn : Int) : Tile .bool [p.BM] :=
  Tile.vec (fun i => decide (sn ≤ rows i+prompt))

theorem guard_eval (p : Layout) (rows : Fin p.BM → Nat) (prompt sn : Int) (u : BlockState)
    (hr : u.regs .nat [p.BM] "offs_m" = some (Tile.vec rows))
    (hp : u.regs .int [] "prompt_cache_len" = some (Tile.scalar prompt))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn)) :
    evalOp (rescaleGuard p) u = some (guard p rows prompt sn) := by
  simp [rescaleGuard,guard,n,z,evalOp.eq_def,hr,hp,hs,
    Tile.bop,Tile.cop,Tile.scalar,Tile.vec,ComparableDType.ge,NumericDType.add,
    Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_eq_bind,Option.bind,-Tile.scalar_eta]

def probabilityScale (p : Layout) (beta lin : Tile .real [p.BM]) :=
  Tile.bop NumericDType.real.div Broadcast.nil.consSame beta lin
def probabilities (p : Layout) (ptile : Tile .real [p.BM,p.BN]) (beta lin : Tile .real [p.BM]) :=
  Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame ptile
    (Tile.expandDim ⟨1,by simp⟩ (probabilityScale p beta lin))
def rawScale (p : Layout) (li lin alpha : Tile .real [p.BM]) :=
  Tile.bop NumericDType.real.mul Broadcast.nil.consSame
    (Tile.bop NumericDType.real.div Broadcast.nil.consSame li lin) alpha
def accumulatorScale (p : Layout) (rows : Fin p.BM → Nat) (prompt sn : Int) (li lin alpha : Tile .real [p.BM]) :=
  Tile.select (guard p rows prompt sn) (rawScale p li lin alpha) (Tile.vec (fun _ => some (1 : ℝ)))
def accumulator (p : Layout) (rows : Fin p.BM → Nat) (prompt sn : Int) (li lin alpha : Tile .real [p.BM])
    (acc : Tile .real [p.BM,p.D]) :=
  Tile.bop NumericDType.real.mul Broadcast.nil.consR.consSame acc
    (Tile.expandDim ⟨1,by simp⟩ (accumulatorScale p rows prompt sn li lin alpha))
def prepared (p : Layout) (rows : Fin p.BM → Nat) (prompt sn : Int) (li lin alpha beta : Tile .real [p.BM])
    (ptile : Tile .real [p.BM,p.BN]) (acc : Tile .real [p.BM,p.D]) (u : BlockState) :=
  let u := u.setReg "p_scale" .real [p.BM] (probabilityScale p beta lin)
  let u := u.setReg "p" .real [p.BM,p.BN] (probabilities p ptile beta lin)
  let u := u.setReg "acc_scale" .real [p.BM] (rawScale p li lin alpha)
  let u := u.setReg "acc_scale" .real [p.BM] (accumulatorScale p rows prompt sn li lin alpha)
  u.setReg "acc" .real [p.BM,p.D] (accumulator p rows prompt sn li lin alpha acc)

theorem one_eval (p : Layout) (u : BlockState) :
    evalOp (Op.broadcast (.const (1.0 : ℝ)) [p.BM]) u = some (Tile.vec (dtype := .real) (fun _ => some 1)) := by
  norm_num [evalOp.eq_def,Tile.vec,Tile.scalar,Option.bind_eq_bind,Option.bind,-Tile.scalar_eta]

theorem executes (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (rows : Fin p.BM → Nat) (prompt sn : Int) (li lin alpha beta : Tile .real [p.BM])
    (ptile : Tile .real [p.BM,p.BN]) (acc : Tile .real [p.BM,p.D]) (u : BlockState)
    (hr : u.regs .nat [p.BM] "offs_m" = some (Tile.vec rows))
    (hp : u.regs .int [] "prompt_cache_len" = some (Tile.scalar prompt))
    (hs : u.regs .int [] "start_n" = some (Tile.scalar sn))
    (hli : u.regs .real [p.BM] "l_i" = some li)
    (hlin : u.regs .real [p.BM] "l_i_new" = some lin)
    (ha : u.regs .real [p.BM] "alpha" = some alpha)
    (hb : u.regs .real [p.BM] "beta" = some beta)
    (hpt : u.regs .real [p.BM,p.BN] "p" = some ptile)
    (hac : u.regs .real [p.BM,p.D] "acc" = some acc) :
    stepStmts (scales p r swapped scale) u = some (prepared p rows prompt sn li lin alpha beta ptile acc u) := by
  let s1 := u.setReg "p_scale" .real [p.BM] (probabilityScale p beta lin)
  let s2 := s1.setReg "p" .real [p.BM,p.BN] (probabilities p ptile beta lin)
  let s3 := s2.setReg "acc_scale" .real [p.BM] (rawScale p li lin alpha)
  let s4 := s3.setReg "acc_scale" .real [p.BM] (accumulatorScale p rows prompt sn li lin alpha)
  have h1 := bloomPscale_eval u beta lin hb hlin
  change evalOp _ u = some (probabilityScale p beta lin) at h1
  have h2 := bloomP2_eval s1 ptile (probabilityScale p beta lin) (by simp [s1,hpt]) (by simp [s1])
  change evalOp _ s1 = some (probabilities p ptile beta lin) at h2
  have h3 := bloomAccScale1_eval s2 li lin alpha (by simp [s2,s1,hli]) (by simp [s2,s1,hlin]) (by simp [s2,s1,ha])
  change evalOp _ s2 = some (rawScale p li lin alpha) at h3
  have hc := guard_eval p rows prompt sn s3 (by simp [s3,s2,s1,hr]) (by simp [s3,s2,s1,hp]) (by simp [s3,s2,s1,hs])
  have he3 : s3.regs .real [p.BM] "acc_scale" = some (rawScale p li lin alpha) := by simp [s3]
  have h4 : evalOp (.where (rescaleGuard p) (.ref .real [p.BM] "acc_scale")
      (.broadcast (.const (1.0 : ℝ)) [p.BM])) s3 = some (accumulatorScale p rows prompt sn li lin alpha) := by
    rw [evalOp_where,hc,evalOp_ref,he3,one_eval]
    rfl
  have h5 := bloomAcc1_eval s4 acc (accumulatorScale p rows prompt sn li lin alpha)
    (by simp [s4,s3,s2,s1,hac]) (by simp [s4])
  change evalOp _ s4 = some (accumulator p rows prompt sn li lin alpha acc) at h5
  rw [(sources p r swapped scale).2.2.1]
  unfold scalesCode mat column vec
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h1)]
  change stepStmts _ s1 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h2)]
  change stepStmts _ s2 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h3)]
  change stepStmts _ s3 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h4)]
  change stepStmts _ s4 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h5)]
  simp only [stepStmts,prepared,s1,s2,s3,s4]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.ScalesExecution
