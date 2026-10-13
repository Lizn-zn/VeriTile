import bench.tritonbench_g.index_select_bwd.Proofs.Source
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.IndexSelectBwd
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

theorem candidate_prefix (A : FlatAlloc) (out idx inp : RegionName)
    (rows n c s0 s1 BI BC : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts ((mathematicalSource Bool.false out idx inp rows n c s0 s1 BI BC).toAlgKernel.body.take 3)) s =
      stepStmts (A.flattenStmts ((mathematicalSource Bool.true out idx inp rows n c s0 s1 BI BC).toAlgKernel.body.take 3)) s := by
  simp [mathematicalSource, source, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeKernel.surfaceBody, ComputeKernel.fromKernelBody, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    Except.bind, bind, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, NumericDType.mul, Nat.mul_comm, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType]

theorem source_flat_exec (A : FlatAlloc) (input output : Storage) (index : IndexStorage)
    (v : Bool) (out idx inp : RegionName) (rows n c s0 s1 BI BC : Nat) (s : BlockState) :
    exec (A.flattenKernel (source input output index v out idx inp rows n c s0 s1 BI BC).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (mathematicalSource Bool.false out idx inp rows n c s0 s1 BI BC).toAlgKernel) s := by
  rw [source_erasure]
  cases v
  · rfl
  · unfold exec FlatAlloc.flattenKernel
    rw [← List.take_append_drop 3 (mathematicalSource Bool.true out idx inp rows n c s0 s1 BI BC).toAlgKernel.body,
      ← List.take_append_drop 3 (mathematicalSource Bool.false out idx inp rows n c s0 s1 BI BC).toAlgKernel.body]
    have hs : (mathematicalSource Bool.true out idx inp rows n c s0 s1 BI BC).toAlgKernel.body.drop 3 =
        (mathematicalSource Bool.false out idx inp rows n c s0 s1 BI BC).toAlgKernel.body.drop 3 := by
      simp only [mathematicalSource, source, ComputeKernel.eraseDType,
        ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
        ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
        Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
        MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, Except.bind, bind,
        VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType]
      rfl
    simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, candidate_prefix, hs]

end VeriTile.Bench.TritonBenchG.IndexSelectBwd
