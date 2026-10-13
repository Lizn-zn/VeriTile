import bench.tritonbench_g.rotary_emb.Proofs.Store

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def kIndexBody (p : Parameters) (Q K Cos Sin : RegionName) : List Stmt :=
  ((mathKernel p Q K Cos Sin).body.drop 20).take 4

def kLoadBody (p : Parameters) (Q K Cos Sin : RegionName) : List Stmt :=
  ((mathKernel p Q K Cos Sin).body.drop 24).take 6

def kArithmeticBody (p : Parameters) (Q K Cos Sin : RegionName) : List Stmt :=
  ((mathKernel p Q K Cos Sin).body.drop 30).take 2

def kStoreBody (p : Parameters) (Q K Cos Sin : RegionName) : List Stmt :=
  ((mathKernel p Q K Cos Sin).body.drop 32).take 2

/-- The decomposition retains all 34 source statements and all four stores.
The second set of trigonometric loads belongs to the K load segment. -/
theorem source_parts (p : Parameters) (Q K Cos Sin : RegionName) :
    (mathKernel p Q K Cos Sin).body =
      prefix p ++ qIndexBody p Q K Cos Sin ++ qLoadBody p Q K Cos Sin ++
      qArithmeticBody p Q K Cos Sin ++ qStoreBody p Q K Cos Sin ++
      kIndexBody p Q K Cos Sin ++ kLoadBody p Q K Cos Sin ++
      kArithmeticBody p Q K Cos Sin ++ kStoreBody p Q K Cos Sin := rfl

noncomputable def kIndexState (p : Parameters) (s : BlockState)
    (Q Cos Sin : RegionName) : BlockState :=
  ((((qStoreState p s Q Cos Sin).setReg "off_k0" .nat
    [p.blockSeq,p.blockHead,p.dimension/2]
    (p.dataOffsets s p.kBatch p.kHead p.kDim Bool.false)).setReg "off_k1" .nat
    [p.blockSeq,p.blockHead,p.dimension/2]
    (p.dataOffsets s p.kBatch p.kHead p.kDim Bool.true)).setReg "off_dimcos_sin0" .nat
    [p.blockSeq,1,p.dimension/2] (p.trigOffsets s Bool.false)).setReg "off_dimcos_sin1" .nat
    [p.blockSeq,1,p.dimension/2] (p.trigOffsets s Bool.true)

macro "solveRotaryKIndex" : tactic => `(tactic|
  (simp [evalOp.eq_def, qStoreState, qArithmeticState, qLoadState, qIndexState,
    prefixState, Scatter.pair_regs, Parameters.dataOffsets, Parameters.trigOffsets,
    address, trigAddress, sequence, head, dimension, range, dims, tile_elementwise]
   <;> rfl))

theorem k_index_run (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState) :
    stepStmts (kIndexBody p Q K Cos Sin) (qStoreState p s Q Cos Sin) =
      some (kIndexState p s Q Cos Sin) := by
  change stepStmts [_,_,_,_] (qStoreState p s Q Cos Sin) =
    some (kIndexState p s Q Cos Sin)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.dataOffsets s p.kBatch p.kHead p.kDim Bool.false) (by solveRotaryKIndex))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.dataOffsets s p.kBatch p.kHead p.kDim Bool.true) (by solveRotaryKIndex))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.trigOffsets s Bool.false) (by solveRotaryKIndex))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.trigOffsets s Bool.true) (by solveRotaryKIndex))]
  rfl

/-- The second phase really reloads trigonometric values after Q stores.
These definitions intentionally sample that intermediate memory. -/
noncomputable def kLoadState (p : Parameters) (s : BlockState)
    (Q K Cos Sin : RegionName) : BlockState :=
  let t := qStoreState p s Q Cos Sin
  ((((((kIndexState p s Q Cos Sin).setReg "k0" .real
    [p.blockSeq,p.blockHead,p.dimension/2]
    (p.dataValues t K p.kBatch p.kHead p.kDim p.headsK Bool.false)).setReg "k1" .real
    [p.blockSeq,p.blockHead,p.dimension/2]
    (p.dataValues t K p.kBatch p.kHead p.kDim p.headsK Bool.true)).setReg "cos0" .real
    [p.blockSeq,1,p.dimension/2] (p.trigValues t Cos Bool.false)).setReg "sin0" .real
    [p.blockSeq,1,p.dimension/2] (p.trigValues t Sin Bool.false)).setReg "cos1" .real
    [p.blockSeq,1,p.dimension/2] (p.trigValues t Cos Bool.true)).setReg "sin1" .real
    [p.blockSeq,1,p.dimension/2] (p.trigValues t Sin Bool.true)

macro "solveRotaryKLoad" : tactic => `(tactic|
  simp [evalOp.eq_def, kIndexState, qStoreState, qArithmeticState, qLoadState,
    qIndexState, prefixState, Parameters.dataOffsets, Parameters.trigOffsets,
    Parameters.dataValues, Parameters.trigValues, address, trigAddress, sequence,
    head, dimension, active, range, dims, tile_elementwise, Bool.and_eq_true])

theorem k_load_run (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState) :
    stepStmts (kLoadBody p Q K Cos Sin) (kIndexState p s Q Cos Sin) =
      some (kLoadState p s Q K Cos Sin) := by
  change stepStmts [_,_,_,_,_,_] (kIndexState p s Q Cos Sin) =
    some (kLoadState p s Q K Cos Sin)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.dataValues (qStoreState p s Q Cos Sin) K
      p.kBatch p.kHead p.kDim p.headsK Bool.false) (by solveRotaryKLoad))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.dataValues (qStoreState p s Q Cos Sin) K
      p.kBatch p.kHead p.kDim p.headsK Bool.true) (by solveRotaryKLoad))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.trigValues (qStoreState p s Q Cos Sin) Cos Bool.false) (by solveRotaryKLoad))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.trigValues (qStoreState p s Q Cos Sin) Sin Bool.false) (by solveRotaryKLoad))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.trigValues (qStoreState p s Q Cos Sin) Cos Bool.true) (by solveRotaryKLoad))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.trigValues (qStoreState p s Q Cos Sin) Sin Bool.true) (by solveRotaryKLoad))]
  rfl

noncomputable def kArithmeticState (p : Parameters) (s : BlockState)
    (Q K Cos Sin : RegionName) : BlockState :=
  let t := qStoreState p s Q Cos Sin
  ((kLoadState p s Q K Cos Sin).setReg "out_k0" .real
    [p.blockSeq,p.blockHead,p.dimension/2]
    (p.resultValues t K Cos Sin p.kBatch p.kHead p.kDim p.headsK Bool.false)).setReg
    "out_k1" .real [p.blockSeq,p.blockHead,p.dimension/2]
    (p.resultValues t K Cos Sin p.kBatch p.kHead p.kDim p.headsK Bool.true)

set_option hygiene false in
macro "solveRotaryKArithmetic" : tactic => `(tactic|
  (simp only [evalOp.eq_def, kLoadState, BlockState.setReg_same,
    BlockState.setReg_ne_name, String.reduceEq, not_false_eq_true, ne_eq,
    Option.bind_some, Option.map_some]
   congr 1
   apply Tile.ext
   intro i
   by_cases hs : sequence (qStoreState p s Q Cos Sin) p.blockSeq i < p.length
   all_goals
     by_cases hh : head (qStoreState p s Q Cos Sin) p.blockHead i < p.headsK
     all_goals
       simp only [sequence, head] at hs hh
       simp [Parameters.dataValues, Parameters.trigValues, Parameters.resultValues,
         expected, active, sequence, head, trigAddress, dimension, tile_elementwise,
         WithBot.realMul, WithBot.realSub, WithBot.realAdd, hs, hh]))

theorem k_arithmetic_run (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState) :
    stepStmts (kArithmeticBody p Q K Cos Sin) (kLoadState p s Q K Cos Sin) =
      some (kArithmeticState p s Q K Cos Sin) := by
  change stepStmts [_,_] (kLoadState p s Q K Cos Sin) =
    some (kArithmeticState p s Q K Cos Sin)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.resultValues (qStoreState p s Q Cos Sin) K Cos Sin
      p.kBatch p.kHead p.kDim p.headsK Bool.false) (by solveRotaryKArithmetic))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := p.resultValues (qStoreState p s Q Cos Sin) K Cos Sin
      p.kBatch p.kHead p.kDim p.headsK Bool.true) (by solveRotaryKArithmetic))]
  rfl

noncomputable def kStoreState (p : Parameters) (s : BlockState)
    (Q K Cos Sin : RegionName) : BlockState :=
  Scatter.pair K (p.kAddress s) (p.expectedK (qStoreState p s Q Cos Sin) K Cos Sin)
    (p.activeK s) (kArithmeticState p s Q K Cos Sin)

theorem k_expected_initial (p : Parameters) (s : BlockState)
    (Q K Cos Sin : RegionName) (hsep : RegionsSeparate Q K Cos Sin)
    (odd : Bool) (i : p.Lane) :
    p.expectedK (qStoreState p s Q Cos Sin) K Cos Sin odd i =
      p.expectedK s K Cos Sin odd i := by
  apply expected_congr
  · exact q_store_state_pids p s Q Cos Sin
  · exact q_store_state_read_other p s Q Cos Sin K (Ne.symm hsep.1)
  · exact q_store_state_read_other p s Q Cos Sin Cos (Ne.symm hsep.2.1)
  · exact q_store_state_read_other p s Q Cos Sin Sin (Ne.symm hsep.2.2.1)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
