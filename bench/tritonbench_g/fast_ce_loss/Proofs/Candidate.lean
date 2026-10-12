import bench.tritonbench_g.fast_ce_loss.Proofs.Erasure
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Memory.FlattenTolerant
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.Candidate
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
inductive Kind where
  | forward | chunked | backward

def source (kind : Kind) (cap scale swapped : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B C DR : Nat) (SC SL : ℝ) : Kernel :=
  (match kind with
   | .forward => forward .fp32 cap scale swapped X Y Z L V R B SC SL
   | .chunked => chunked .fp32 cap scale swapped X Y Z L V C R B SC SL
   | .backward => backward .fp32 cap scale swapped X Y Z L V R DR B SC SL).eraseDType.toAlgKernel

def indexPrefix (kind : Kind) (swapped : Bool) (X : RegionName) (R : Nat) : List Stmt :=
  [.assign .nat [] "row_idx" (.programId 0)] ++
  (match kind with
   | .forward => []
   | .chunked => [.assign .nat [] "chunk_idx" (.programId 1)]
   | .backward => [.assign .nat [] "block_idx" (.programId 1)]) ++
  [.assign .ptr [] "logits_ptr" (.ptrAdd .nil (.ptrBase X)
    (if swapped then .mul .nat .nil (.constNat R) (.ref .nat [] "row_idx")
     else .mul .nat .nil (.ref .nat [] "row_idx") (.constNat R)))]

theorem prefix_flat (A : FlatAlloc) (kind : Kind) (X : RegionName) (R : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts (indexPrefix kind Bool.false X R)) u =
    stepStmts (A.flattenStmts (indexPrefix kind Bool.true X R)) u := by
  cases kind
  all_goals
    simp [indexPrefix,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
      stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.ptrAdd,NumericDType.mul,Nat.mul_comm]

theorem body (kind : Kind) (cap scale swapped : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B C DR : Nat) (SC SL : ℝ) :
    (source kind cap scale swapped X Y Z L V R B C DR SC SL).body =
      indexPrefix kind swapped X R ++
        (source kind cap scale Bool.false X Y Z L V R B C DR SC SL).body.drop
          (indexPrefix kind swapped X R).length := by
  cases kind <;> cases cap <;> cases scale <;> cases swapped
  all_goals simp [source,indexPrefix,forward,chunked,backward,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,NumericDType.eraseDType,ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType,Region.cast,Except.bind,bind]

theorem flat_exec (A : FlatAlloc) (kind : Kind) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V R B C DR : Nat) (SC SL : ℝ) (u : BlockState) :
    exec (A.flattenKernel (source kind cap scale Bool.false X Y Z L V R B C DR SC SL)) u =
    exec (A.flattenKernel (source kind cap scale Bool.true X Y Z L V R B C DR SC SL)) u := by
  unfold exec FlatAlloc.flattenKernel
  rw [body kind cap scale Bool.false,body kind cap scale Bool.true]
  have hp : (indexPrefix kind Bool.false X R).length=(indexPrefix kind Bool.true X R).length := by
    cases kind <;> rfl
  rw [hp]
  simp only [FlatAlloc.flattenStmts_append,ProgramRewrite.append,prefix_flat]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastCeLoss.Candidate
