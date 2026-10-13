import bench.tritonbench_g.rotary_emb.Proofs.Loads

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def qArithmeticBody (p : Parameters) (Q K Cos Sin : RegionName) : List Stmt :=
  ((mathKernel p Q K Cos Sin).body.drop 16).take 2

noncomputable def qArithmeticState (p : Parameters) (s : BlockState)
    (Q Cos Sin : RegionName) : BlockState :=
  ((qLoadState p s Q Cos Sin).setReg "out0" .real
    [p.blockSeq,p.blockHead,p.dimension/2]
    (p.resultValues s Q Cos Sin p.qBatch p.qHead p.qDim p.headsQ Bool.false)).setReg
    "out1" .real [p.blockSeq,p.blockHead,p.dimension/2]
    (p.resultValues s Q Cos Sin p.qBatch p.qHead p.qDim p.headsQ Bool.true)

@[simp] theorem q_arithmetic_mem (p : Parameters) (s : BlockState)
    (Q Cos Sin : RegionName) : (qArithmeticState p s Q Cos Sin).mem = s.mem := rfl

@[simp] theorem q_arithmetic_pids (p : Parameters) (s : BlockState)
    (Q Cos Sin : RegionName) : (qArithmeticState p s Q Cos Sin).pids = s.pids := rfl

set_option hygiene false in
macro "solveRotaryArithmetic" : tactic => `(tactic|
  (simp only [evalOp.eq_def, qLoadState, BlockState.setReg_same,
    BlockState.setReg_ne_name, String.reduceEq, not_false_eq_true, ne_eq,
    Option.bind_some, Option.map_some]
   congr 1
   apply Tile.ext
   intro i
   by_cases hs : sequence s p.blockSeq i < p.length
   all_goals
     by_cases hh : head s p.blockHead i < p.headsQ
     all_goals
       simp only [sequence, head] at hs hh
       simp [Parameters.dataValues, Parameters.trigValues, Parameters.resultValues,
         expected, active, sequence, head, trigAddress, dimension, tile_elementwise,
         WithBot.realMul, WithBot.realSub, WithBot.realAdd, hs, hh]))

/-- Both arithmetic assignments retain the distinct even and odd trigonometric
loads. Masked source lanes reduce to zero, without assumptions on input memory. -/
theorem q_arithmetic_run (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState) :
    stepStmts (qArithmeticBody p Q K Cos Sin) (qLoadState p s Q Cos Sin) =
      some (qArithmeticState p s Q Cos Sin) := by
  change stepStmts [_,_] (qLoadState p s Q Cos Sin) =
    some (qArithmeticState p s Q Cos Sin)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.resultValues s Q Cos Sin p.qBatch p.qHead p.qDim p.headsQ Bool.false)
    (by solveRotaryArithmetic))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.resultValues s Q Cos Sin p.qBatch p.qHead p.qDim p.headsQ Bool.true)
    (by solveRotaryArithmetic))]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
