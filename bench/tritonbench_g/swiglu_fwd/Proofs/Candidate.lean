import bench.tritonbench_g.swiglu_fwd.Proofs.Source

namespace VeriTile.Bench.TritonBenchG.SwigluFwd
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- Commuting row addresses preserves every mathematical execution and memory cell. -/
theorem math_flat_exec (A : FlatAlloc) (v : Bool) (X Y OUT : RegionName)
    (sx sy so N B : Nat) (s : BlockState) :
    exec (A.flattenKernel (mathKernel v X Y OUT sx sy so N B)) s =
      exec (A.flattenKernel (swiglu_fwd_kernel X Y OUT sx sy so N B).toAlgKernel) s := by
  cases v <;> simp [mathKernel, commutedMathKernel, swiglu_fwd_kernel,
    ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    exec, FlatAlloc.flattenKernel, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, NumericDType.mul, Nat.mul_comm]

theorem typed_flat_exec (A : FlatAlloc) (input partner output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) (s : BlockState) :
    exec (A.flattenKernel
      (typedKernel input partner output v X Y OUT sx sy so N B).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (swiglu_fwd_kernel X Y OUT sx sy so N B).toAlgKernel) s := by
  rw [typed_projection]
  exact math_flat_exec A v X Y OUT sx sy so N B s

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluFwd
