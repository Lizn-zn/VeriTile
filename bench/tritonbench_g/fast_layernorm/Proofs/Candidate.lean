import bench.tritonbench_g.fast_layernorm.Proofs.Erasure
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Memory.FlattenTolerant
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.Candidate
open VeriTile.Triton TypedSource
set_option maxHeartbeats 500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
inductive Kind where
  | forward | backward

def source (kind : Kind) (swapped : Bool) (Y X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) : Kernel :=
  (match kind with
   | .forward => forward .fp32 .fp32 .fp32 swapped Y X W b r mu YR XR N B eps
   | .backward => backward .fp32 .fp32 .fp32 swapped Y X W b r mu YR XR N B eps).eraseDType.toAlgKernel

def indexPrefix (kind : Kind) (swapped : Bool) (Y X : RegionName) (YR XR N B : Nat) : List Stmt :=
  [.assign .nat [] "row_idx" (.programId 0),
   .assign .nat [B] "col_offsets" (.arange B),
   .assign .bool [B] "mask" (.lt .nat (.scalarR) (.ref .nat [B] "col_offsets") (.constNat N)),
   .assign .ptr [] (match kind with | .forward => "Y" | .backward => "dY")
     (.ptrAdd .nil (.ptrBase Y) (.mul .nat .nil (.ref .nat [] "row_idx") (.constNat YR))),
   .assign .ptr [] "X" (.ptrAdd .nil (.ptrBase X)
     (if swapped then .mul .nat .nil (.constNat XR) (.ref .nat [] "row_idx")
      else .mul .nat .nil (.ref .nat [] "row_idx") (.constNat XR)))]

theorem prefix_flat (A : FlatAlloc) (kind : Kind) (Y X : RegionName) (YR XR N B : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts (indexPrefix kind Bool.false Y X YR XR N B)) u =
    stepStmts (A.flattenStmts (indexPrefix kind Bool.true Y X YR XR N B)) u := by
  cases kind
  all_goals
    simp [indexPrefix,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
      stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.ptrAdd,NumericDType.mul,Nat.mul_comm]

theorem body (kind : Kind) (swapped : Bool) (Y X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) :
    (source kind swapped Y X W b r mu YR XR N B eps).body =
      indexPrefix kind swapped Y X YR XR N B ++
        (source kind Bool.false Y X W b r mu YR XR N B eps).body.drop 5 := by
  cases kind <;> cases swapped
  all_goals simp [source,indexPrefix,forward,backward,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,NumericDType.eraseDType,ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType,Region.cast,Except.bind,bind]

theorem flat_exec (A : FlatAlloc) (kind : Kind) (Y X W b r mu : RegionName)
    (YR XR N B : Nat) (eps : ℝ) (u : BlockState) :
    exec (A.flattenKernel (source kind Bool.false Y X W b r mu YR XR N B eps)) u =
    exec (A.flattenKernel (source kind Bool.true Y X W b r mu YR XR N B eps)) u := by
  unfold exec FlatAlloc.flattenKernel
  rw [body kind Bool.false,body kind Bool.true]
  simp only [FlatAlloc.flattenStmts_append,ProgramRewrite.append,prefix_flat]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastLayernorm.Candidate
