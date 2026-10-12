import bench.tritonbench_g.embedding_triton_kernel.Proofs.Erasure
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Memory.FlattenTolerant
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Candidate
open VeriTile.Triton TypedSource
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def indexPrefix (swapped : Bool) (N : Nat) : List Stmt :=
  [.assign .nat [] "start_n" (if swapped then .mul .nat .nil (.constNat N) (.programId 0)
    else .mul .nat .nil (.programId 0) (.constNat N))]

theorem prefix_flat (A : FlatAlloc) (N : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts (indexPrefix Bool.false N)) u=
      stepStmts (A.flattenStmts (indexPrefix Bool.true N)) u := by
  simp only [indexPrefix,Bool.false_eq_true,↓reduceIte,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    stepStmts,stepStmt,evalOp_programId,evalOp_mul,evalOp_constNat,bind,Option.bind_some]
  congr 2
  ext j
  exact Nat.mul_comm _ _

theorem body (swapped : Bool) (weight ids out : RegionName) (first last : Int) (sw so n h D N NN : Nat) :
    (Erasure.source swapped weight ids out first last sw so n h D N NN).toAlgKernel.body =
      indexPrefix swapped N ++ (Erasure.source Bool.false weight ids out first last sw so n h D N NN).toAlgKernel.body.drop 1 := by
  cases swapped <;> rfl

theorem flat_exec (A : FlatAlloc) (storage : Storage) (weight ids out : RegionName)
    (first last : Int) (sw so n h D N NN : Nat) (u : BlockState) :
    exec (A.flattenKernel (TypedSource.source storage Bool.false weight ids out first last sw so n h D N NN).eraseDType.toAlgKernel) u =
      exec (A.flattenKernel (TypedSource.source storage Bool.true weight ids out first last sw so n h D N NN).eraseDType.toAlgKernel) u := by
  unfold exec FlatAlloc.flattenKernel
  simp only [Erasure.body]
  rw [body Bool.false,body Bool.true]
  simp only [FlatAlloc.flattenStmts_append,ProgramRewrite.append,prefix_flat]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Candidate
