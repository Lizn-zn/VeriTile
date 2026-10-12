import bench.tritonbench_g.fast_rope_embedding.Proofs.Erasure
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Memory.FlattenTolerant
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Candidate
open VeriTile.Triton TypedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def source (swapped : Bool) (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) : Kernel :=
  (rope .fp32 .fp32 .fp32 swapped Q C S QR CR SR seq H heads B backwards).eraseDType.toAlgKernel

def indexStatement (swapped : Bool) : Stmt :=
  .assign .nat [] "head_start" (if swapped then
    .mul .nat .nil (.ref .nat [] "ROPE_GROUP_SIZE") (.ref .nat [] "group_head_position") else
    .mul .nat .nil (.ref .nat [] "group_head_position") (.ref .nat [] "ROPE_GROUP_SIZE"))

theorem index_flat (A : FlatAlloc) (u : BlockState) :
    stepStmts (A.flattenStmts [indexStatement Bool.false]) u=
    stepStmts (A.flattenStmts [indexStatement Bool.true]) u := by
  cases hg : u.regs .nat [] "group_head_position" <;> cases hs : u.regs .nat [] "ROPE_GROUP_SIZE"
  all_goals simp [indexStatement,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    stepStmts,stepStmt,evalOp.eq_def,Tile.bop,NumericDType.mul,hg,hs,Nat.mul_comm]

theorem decomposition (swapped : Bool) (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    (source swapped Q C S QR CR SR seq H heads B backwards).body=
      (source Bool.false Q C S QR CR SR seq H heads B backwards).body.take 9 ++
      [indexStatement swapped] ++
      (source Bool.false Q C S QR CR SR seq H heads B backwards).body.drop 10 := by
  cases swapped <;> cases backwards
  all_goals simp only [source,indexStatement,rope,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
  all_goals rfl

theorem flat_exec (A : FlatAlloc) (Q C S : RegionName) (QR CR SR seq H heads B : Nat)
    (backwards : Bool) (u : BlockState) :
    exec (A.flattenKernel (source Bool.false Q C S QR CR SR seq H heads B backwards)) u=
    exec (A.flattenKernel (source Bool.true Q C S QR CR SR seq H heads B backwards)) u := by
  unfold exec FlatAlloc.flattenKernel
  rw [decomposition Bool.false,decomposition Bool.true]
  simp only [FlatAlloc.flattenStmts_append,ProgramRewrite.append,index_flat]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Candidate
