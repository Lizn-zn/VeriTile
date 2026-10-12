import bench.tritonbench_g.f8_conversion_utils.Proofs.Erasure
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Memory.FlattenTolerant
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.Candidate
open VeriTile.Triton TypedSource
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def source (doubleStore swapped : Bool) (Y X : RegionName) (N B : Nat) : ComputeKernel :=
  if doubleStore then Erasure.toHalf swapped Y X N B else Erasure.toFP8 swapped Y X N B

def indexPrefix (swapped : Bool) (B : Nat) : List Stmt :=
  [.assign .nat [] "pid" (.programId 0),
   .assign .nat [B] "offs" (.add .nat .scalarL
     (if swapped then .mul .nat .nil (.constNat B) (.ref .nat [] "pid")
     else .mul .nat .nil (.ref .nat [] "pid") (.constNat B)) (.arange B))]

theorem prefix_flat (A : FlatAlloc) (B : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts (indexPrefix Bool.false B)) u=
      stepStmts (A.flattenStmts (indexPrefix Bool.true B)) u := by
  simp only [indexPrefix,Bool.false_eq_true,↓reduceIte,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    stepStmts,stepStmt,evalOp_programId,evalOp_add,evalOp_arange,evalOp_mul,evalOp_ref,evalOp_constNat,
    bind,Option.bind_some,BlockState.setReg_same]
  congr 2
  ext j
  simp [Tile.bop,NumericDType.add,NumericDType.mul,Nat.mul_comm]

theorem body (doubleStore swapped : Bool) (Y X : RegionName) (N B : Nat) :
    (source doubleStore swapped Y X N B).toAlgKernel.body =
      indexPrefix swapped B ++ (source doubleStore Bool.false Y X N B).toAlgKernel.body.drop 2 := by
  cases doubleStore <;> cases swapped
  all_goals simp [source,Erasure.toHalf,Erasure.toFP8,indexPrefix,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,NumericDType.eraseDType,ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType,Region.cast,Except.bind,bind]

theorem flat_exec (A : FlatAlloc) (doubleStore : Bool) (Y X : RegionName) (N B : Nat) (u : BlockState) :
    exec (A.flattenKernel (source doubleStore Bool.false Y X N B).toAlgKernel) u=
      exec (A.flattenKernel (source doubleStore Bool.true Y X N B).toAlgKernel) u := by
  unfold exec FlatAlloc.flattenKernel
  rw [body doubleStore Bool.false,body doubleStore Bool.true]
  simp only [FlatAlloc.flattenStmts_append,ProgramRewrite.append,prefix_flat]

theorem to_half_flat_exec (A : FlatAlloc) (format : Format) (Y X : RegionName) (N B : Nat) (u : BlockState) :
    exec (A.flattenKernel (toHalf format Bool.false Y X N B).eraseDType.toAlgKernel) u=
      exec (A.flattenKernel (toHalf format Bool.true Y X N B).eraseDType.toAlgKernel) u := by
  simpa only [exec,FlatAlloc.flattenKernel,Erasure.to_half_body,source,↓reduceIte]
    using flat_exec A Bool.true Y X N B u

theorem to_fp8_flat_exec (A : FlatAlloc) (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) (u : BlockState) :
    exec (A.flattenKernel (toFP8 input format Bool.false Y X N B).eraseDType.toAlgKernel) u=
      exec (A.flattenKernel (toFP8 input format Bool.true Y X N B).eraseDType.toAlgKernel) u := by
  simpa only [exec,FlatAlloc.flattenKernel,Erasure.to_fp8_body,source,Bool.false_eq_true,↓reduceIte]
    using flat_exec A Bool.false Y X N B u

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.Candidate
