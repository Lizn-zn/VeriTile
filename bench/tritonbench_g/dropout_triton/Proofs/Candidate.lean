import bench.tritonbench_g.dropout_triton.Proofs.Erasure
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Memory.FlattenTolerant
namespace VeriTile.Bench.TritonBenchG.DropoutTriton.Candidate
open VeriTile.Triton TypedSource
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def indexPrefix (swapped : Bool) (B : Nat) : List Stmt :=
  [.assign .nat [] "pid" (.programId 0),
   .assign .nat [] "block_start" (if swapped then .mul .nat .nil (.constNat B) (.ref .nat [] "pid")
     else .mul .nat .nil (.ref .nat [] "pid") (.constNat B))]

theorem prefix_flat (A : FlatAlloc) (B : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts (indexPrefix Bool.false B)) u=
      stepStmts (A.flattenStmts (indexPrefix Bool.true B)) u := by
  simp only [indexPrefix,Bool.false_eq_true,↓reduceIte,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    stepStmts,stepStmt,evalOp_programId,evalOp_mul,evalOp_ref,evalOp_constNat,
    bind,Option.bind_some,BlockState.setReg_same]
  congr 2
  ext j
  exact Nat.mul_comm _ _

theorem body (keep : KeepStorage) (swapped : Bool) (x k y : RegionName) (n B : Nat) (p : ℝ) :
    (Erasure.source keep swapped x k y n B p).toAlgKernel.body =
      indexPrefix swapped B ++ (Erasure.source keep Bool.false x k y n B p).toAlgKernel.body.drop 2 := by
  cases keep <;> cases swapped <;> rfl

theorem flat_exec (A : FlatAlloc) (storage : Storage) (keep : KeepStorage)
    (x k y : RegionName) (n B : Nat) (p : ℝ) (u : BlockState) :
    exec (A.flattenKernel (TypedSource.source storage keep Bool.false x k y n B p).eraseDType.toAlgKernel) u =
      exec (A.flattenKernel (TypedSource.source storage keep Bool.true x k y n B p).eraseDType.toAlgKernel) u := by
  unfold exec FlatAlloc.flattenKernel
  simp only [Erasure.body]
  rw [body keep Bool.false,body keep Bool.true]
  simp only [FlatAlloc.flattenStmts_append,ProgramRewrite.append,prefix_flat]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DropoutTriton.Candidate
