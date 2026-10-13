import bench.tritonbench_g.rotary_emb.Proofs.KPhase

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

noncomputable def kEvenState (p : Parameters) (s : BlockState)
    (Q K Cos Sin : RegionName) : BlockState :=
  Scatter.one K (p.kAddress s Bool.false)
    (p.expectedK (qStoreState p s Q Cos Sin) K Cos Sin Bool.false)
    (p.activeK s) (kArithmeticState p s Q K Cos Sin)

macro "solveRotaryKStore" : tactic => `(tactic|
  simp [evalOp.eq_def, kEvenState, kArithmeticState, kLoadState, kIndexState,
    qStoreState, qArithmeticState, qLoadState, qIndexState, prefixState,
    Parameters.dataOffsets, Parameters.resultValues, Parameters.kAddress,
    Parameters.expectedK, Parameters.activeK, address, trigAddress, sequence,
    head, dimension, active, range, dims, tile_elementwise, Bool.and_eq_true])

theorem k_even_store_run (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState) :
    stepStmt (mathKernel p Q K Cos Sin).body[32]! (kArithmeticState p s Q K Cos Sin) =
      some (kEvenState p s Q K Cos Sin) := by
  classical
  first
  | apply real_masked_store_scatter K _ _ _ (p.kAddress s Bool.false)
      (p.expectedK (qStoreState p s Q Cos Sin) K Cos Sin Bool.false) (p.activeK s)
      (kArithmeticState p s Q K Cos Sin)
  | apply real_masked_ptr_store_scatter K _ _ _ (p.kAddress s Bool.false)
      (p.expectedK (qStoreState p s Q Cos Sin) K Cos Sin Bool.false) (p.activeK s)
      (kArithmeticState p s Q K Cos Sin)
  all_goals solveRotaryKStore

theorem k_odd_store_run (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState) :
    stepStmt (mathKernel p Q K Cos Sin).body[33]! (kEvenState p s Q K Cos Sin) =
      some (kStoreState p s Q K Cos Sin) := by
  classical
  first
  | apply real_masked_store_scatter K _ _ _ (p.kAddress s Bool.true)
      (p.expectedK (qStoreState p s Q Cos Sin) K Cos Sin Bool.true) (p.activeK s)
      (kEvenState p s Q K Cos Sin)
  | apply real_masked_ptr_store_scatter K _ _ _ (p.kAddress s Bool.true)
      (p.expectedK (qStoreState p s Q Cos Sin) K Cos Sin Bool.true) (p.activeK s)
      (kEvenState p s Q K Cos Sin)
  all_goals solveRotaryKStore

theorem k_store_run (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState) :
    stepStmts (kStoreBody p Q K Cos Sin) (kArithmeticState p s Q K Cos Sin) =
      some (kStoreState p s Q K Cos Sin) := by
  change stepStmts [(mathKernel p Q K Cos Sin).body[32]!,
    (mathKernel p Q K Cos Sin).body[33]!] (kArithmeticState p s Q K Cos Sin) = _
  rw [stepStmts.cons_some (k_even_store_run p Q K Cos Sin s),
    stepStmts.cons_some (k_odd_store_run p Q K Cos Sin s)] <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
