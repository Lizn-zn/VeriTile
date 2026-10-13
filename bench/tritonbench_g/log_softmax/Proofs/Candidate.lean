import bench.tritonbench_g.log_softmax.Proofs.Source

namespace VeriTile.Bench.TritonBenchG.LogSoftmax
open VeriTile.Triton
set_option maxHeartbeats 12000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- The index candidate preserves complete flat mathematical execution. -/
theorem forward_math_flat_exec (A : FlatAlloc) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat) (s : BlockState) :
    exec (A.flattenKernel (mathForward v OUT X M N K R C)) s =
      exec (A.flattenKernel (log_softmax_kernel OUT X M N K R C).toAlgKernel) s := by
  cases v <;> simp [mathForward, commutedMathForward, log_softmax_kernel,
    ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, FloatDType.toTileDType,
    exec, FlatAlloc.flattenKernel, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, Option.bind, Option.map, NumericDType.mul, Nat.mul_comm,
    FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]

theorem forward_typed_flat_exec (A : FlatAlloc) (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat) (s : BlockState) :
    exec (A.flattenKernel (typedForward input output v OUT X M N K R C).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (log_softmax_kernel OUT X M N K R C).toAlgKernel) s := by
  rw [typedForward_projection]
  exact forward_math_flat_exec A v OUT X M N K R C s

/-- The index candidate preserves complete flat mathematical execution. -/
theorem backward_math_flat_exec (A : FlatAlloc) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) (s : BlockState) :
    exec (A.flattenKernel (mathBackward v Y G DX M N K R C)) s =
      exec (A.flattenKernel (log_softmax_backward_kernel Y G DX M N K R C).toAlgKernel) s := by
  cases v <;> simp [mathBackward, commutedMathBackward, log_softmax_backward_kernel,
    ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, FloatDType.toTileDType,
    exec, FlatAlloc.flattenKernel, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, Option.bind, Option.map, NumericDType.mul, Nat.mul_comm,
    FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]

theorem backward_typed_flat_exec (A : FlatAlloc) (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) (s : BlockState) :
    exec (A.flattenKernel (typedBackward input upstream output v Y G DX M N K R C).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (log_softmax_backward_kernel Y G DX M N K R C).toAlgKernel) s := by
  rw [typedBackward_projection]
  exact backward_math_flat_exec A v Y G DX M N K R C s

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LogSoftmax
