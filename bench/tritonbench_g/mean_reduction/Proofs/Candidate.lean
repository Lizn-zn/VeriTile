import bench.tritonbench_g.mean_reduction.Proofs.Source
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.MeanReduction
open VeriTile.Triton

theorem index_flat (A : FlatAlloc) (BM : Nat) (s : BlockState) :
    stepStmt (A.flattenStmt (indexStmt Bool.false BM)) s =
      stepStmt (A.flattenStmt (indexStmt Bool.true BM)) s := by
  simp [indexStmt, Bool.false_eq_true, ↓reduceIte,
    FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def,
    stepStmt, evalOp.eq_def, Tile.bop, NumericDType.mul, Nat.mul_comm]

/-- Exact row-index commutation transports the full flattened program. -/
theorem typed_flat_exec (A : FlatAlloc) (input output : Storage) (v : Bool)
    (X Mean : RegionName) (M N BM BN : Nat) (s : BlockState) :
    exec (A.flattenKernel (typedKernel input output v X Mean M N BM BN).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (mean_dim_kernel X Mean M N BM BN).toAlgKernel) s := by
  rw [typed_projection]
  cases v
  · rfl
  · change exec (A.flattenKernel (mathKernel Bool.true X Mean M N BM BN)) s =
      exec (A.flattenKernel (mathKernel Bool.false X Mean M N BM BN)) s
    simp only [exec, FlatAlloc.flattenKernel, mathKernel, FlatAlloc.flattenStmts,
      List.map_cons, stepStmts, ← index_flat]

theorem typed_flat_exec_triv (A : FlatAlloc) (input output : Storage) (v : Bool)
    (X Mean : RegionName) (M N BM BN : Nat) (s : BlockState) :
    execR RoundingModel.triv
      (A.flattenKernel (typedKernel input output v X Mean M N BM BN).eraseDType.toAlgKernel) s =
      execR RoundingModel.triv
        (A.flattenKernel (mean_dim_kernel X Mean M N BM BN).toAlgKernel) s := by
  simp only [execR_triv]
  exact typed_flat_exec A input output v X Mean M N BM BN s

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MeanReduction
