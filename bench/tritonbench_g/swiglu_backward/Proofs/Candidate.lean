import bench.tritonbench_g.swiglu_backward.Proofs.Source
namespace VeriTile.Bench.TritonBenchG.SwigluBackward
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

theorem math_flat_exec (A : FlatAlloc) (v : Bool) (X Y DOUT OUT DX DY : RegionName)
    (sx sy sd so sdx sdy N B : Nat) (R : Bool) (s : BlockState) :
    exec (A.flattenKernel (mathKernel v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R)) s =
      exec (A.flattenKernel
        (swiglu_bwd_kernel X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).toAlgKernel) s := by
  cases v <;> cases R <;> simp [mathKernel, template, swiglu_bwd_kernel,
    ComputeKernel.eraseDType, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    Except.bind, bind, exec, FlatAlloc.flattenKernel, FlatAlloc.flattenStmts,
    FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, NumericDType.mul, Nat.mul_comm]

theorem typed_flat_exec (A : FlatAlloc) (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) (s : BlockState) :
    exec (A.flattenKernel
      (typedKernel input upstream gradient output v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel
        (swiglu_bwd_kernel X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).toAlgKernel) s := by
  rw [typed_projection]
  exact math_flat_exec A v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R s

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluBackward
