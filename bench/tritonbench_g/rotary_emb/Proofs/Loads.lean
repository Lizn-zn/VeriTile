import bench.tritonbench_g.rotary_emb.Proofs.Index

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def qLoadBody (p : Parameters) (Q K Cos Sin : RegionName) : List Stmt :=
  ((mathKernel p Q K Cos Sin).body.drop 10).take 6

noncomputable def qLoadState (p : Parameters) (s : BlockState)
    (Q Cos Sin : RegionName) : BlockState :=
  ((((((qIndexState p s).setReg "q0" .real [p.blockSeq,p.blockHead,p.dimension/2]
    (p.dataValues s Q p.qBatch p.qHead p.qDim p.headsQ Bool.false)).setReg
    "q1" .real [p.blockSeq,p.blockHead,p.dimension/2]
    (p.dataValues s Q p.qBatch p.qHead p.qDim p.headsQ Bool.true)).setReg
    "cos0" .real [p.blockSeq,1,p.dimension/2] (p.trigValues s Cos Bool.false)).setReg
    "sin0" .real [p.blockSeq,1,p.dimension/2] (p.trigValues s Sin Bool.false)).setReg
    "cos1" .real [p.blockSeq,1,p.dimension/2] (p.trigValues s Cos Bool.true)).setReg
    "sin1" .real [p.blockSeq,1,p.dimension/2] (p.trigValues s Sin Bool.true)

macro "solveRotaryLoad" : tactic => `(tactic|
  simp [evalOp.eq_def, qIndexState, prefixState, Parameters.dataOffsets, Parameters.trigOffsets,
    Parameters.dataValues, Parameters.trigValues, address, trigAddress, sequence, head,
    dimension, active, range, dims, tile_elementwise, Bool.and_eq_true])

theorem q_load_run (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState) :
    stepStmts (qLoadBody p Q K Cos Sin) (qIndexState p s) =
      some (qLoadState p s Q Cos Sin) := by
  change stepStmts [_,_,_,_,_,_] (qIndexState p s) = some (qLoadState p s Q Cos Sin)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.dataValues s Q p.qBatch p.qHead p.qDim p.headsQ Bool.false) (by solveRotaryLoad))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.dataValues s Q p.qBatch p.qHead p.qDim p.headsQ Bool.true) (by solveRotaryLoad))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.trigValues s Cos Bool.false) (by solveRotaryLoad))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.trigValues s Sin Bool.false) (by solveRotaryLoad))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.trigValues s Cos Bool.true) (by solveRotaryLoad))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.trigValues s Sin Bool.true) (by solveRotaryLoad))]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
