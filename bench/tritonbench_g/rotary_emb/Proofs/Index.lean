import bench.tritonbench_g.rotary_emb.Proofs.Prefix
import bench.tritonbench_g.rotary_emb.Proofs.Tiles

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def qIndexBody (p : Parameters) (Q K Cos Sin : RegionName) : List Stmt :=
  ((mathKernel p Q K Cos Sin).body.drop 6).take 4

def qIndexState (p : Parameters) (s : BlockState) : BlockState :=
  ((((prefixState p s).setReg "off_q0" .nat [p.blockSeq,p.blockHead,p.dimension/2]
    (p.dataOffsets s p.qBatch p.qHead p.qDim Bool.false)).setReg
    "off_q1" .nat [p.blockSeq,p.blockHead,p.dimension/2]
    (p.dataOffsets s p.qBatch p.qHead p.qDim Bool.true)).setReg
    "off_dimcos_sin0" .nat [p.blockSeq,1,p.dimension/2] (p.trigOffsets s Bool.false)).setReg
    "off_dimcos_sin1" .nat [p.blockSeq,1,p.dimension/2] (p.trigOffsets s Bool.true)

theorem q_index_run (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState) :
    stepStmts (qIndexBody p Q K Cos Sin) (prefixState p s) = some (qIndexState p s) := by
  change stepStmts [_,_,_,_] (prefixState p s) = some (qIndexState p s)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.dataOffsets s p.qBatch p.qHead p.qDim Bool.false) (by
      simp only [evalOp.eq_def, prefixState, BlockState.setReg_ne_name, BlockState.setReg_same,
        String.reduceEq, not_false_eq_true, ne_eq, Option.bind_some, Option.bind_eq_bind]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.dataOffsets s p.qBatch p.qHead p.qDim Bool.true) (by
      simp only [evalOp.eq_def, prefixState, BlockState.setReg_ne_name, BlockState.setReg_same,
        String.reduceEq, not_false_eq_true, ne_eq, Option.bind_some, Option.bind_eq_bind]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := p.trigOffsets s Bool.false) (by
    simp only [evalOp.eq_def, prefixState, BlockState.setReg_ne_name, BlockState.setReg_same,
      String.reduceEq, not_false_eq_true, ne_eq, Option.bind_some, Option.bind_eq_bind]
    rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := p.trigOffsets s Bool.true) (by
    simp only [evalOp.eq_def, prefixState, BlockState.setReg_ne_name, BlockState.setReg_same,
      String.reduceEq, not_false_eq_true, ne_eq, Option.bind_some, Option.bind_eq_bind]
    rfl))]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
