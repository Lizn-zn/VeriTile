import bench.tritonbench_g.swiglu_triton.Proofs.Source

namespace VeriTile.Bench.TritonBenchG.SwigluTriton
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- Signed row-address commutation preserves the complete real execution. -/
theorem fwd_math_flat_exec (A : FlatAlloc) (v : Bool) (a b c : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) (s : BlockState) :
    exec (A.flattenKernel (mathFwd v a b c stride n_cols BLOCK_SIZE)) s =
      exec (A.flattenKernel
        (swiglu_forward_kernel a b c stride n_cols BLOCK_SIZE).toAlgKernel) s := by
  cases v <;> simp [mathFwd, sourceMathFwd, commutedMathFwd, swiglu_forward_kernel,
      ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      exec, FlatAlloc.flattenKernel, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
      FlatAlloc.flattenOp.eq_def, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, NumericDType.mul, Int.mul_comm]

/-- Full backward mathematical execution transport preserves the order of the
two stores and all memory, including inactive output lanes. -/
theorem bwd_math_flat_exec (A : FlatAlloc) (v : Bool) (dc a b : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) (s : BlockState) :
    exec (A.flattenKernel (mathBwd v dc a b stride n_cols BLOCK_SIZE)) s =
      exec (A.flattenKernel
        (swiglu_backward_kernel dc a b stride n_cols BLOCK_SIZE).toAlgKernel) s := by
  cases v <;> simp [mathBwd, sourceMathBwd, commutedMathBwd, swiglu_backward_kernel,
      ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      exec, FlatAlloc.flattenKernel, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
      FlatAlloc.flattenOp.eq_def, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, NumericDType.mul, Int.mul_comm]

theorem fwd_flat_exec (A : FlatAlloc) (storage partner : Storage) (v : Bool)
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) (s : BlockState) :
    exec (A.flattenKernel
      (typedFwd storage partner v a b c stride n_cols BLOCK_SIZE).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel
        (swiglu_forward_kernel a b c stride n_cols BLOCK_SIZE).toAlgKernel) s := by
  rw [fwd_projection]
  exact fwd_math_flat_exec A v a b c stride n_cols BLOCK_SIZE s

theorem bwd_flat_exec (A : FlatAlloc) (storage partner gradient : Storage) (v : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) (s : BlockState) :
    exec (A.flattenKernel
      (typedBwd storage partner gradient v dc a b stride n_cols BLOCK_SIZE).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel
        (swiglu_backward_kernel dc a b stride n_cols BLOCK_SIZE).toAlgKernel) s := by
  rw [bwd_projection]
  exact bwd_math_flat_exec A v dc a b stride n_cols BLOCK_SIZE s

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluTriton
