import bench.tritonbench_g.diag_ssm_triton.Proofs.Erasure
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Memory.FlattenTolerant
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.Candidate
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def index (swapped : Bool) (block : Nat) : Stmt :=
  .assign .nat [] "col_idx"
    (if swapped then .mul .nat .nil (.constNat block) (.programId 0)
     else .mul .nat .nil (.programId 0) (.constNat block))

theorem index_step (block : Nat) (u : BlockState) :
    stepStmt (index Bool.false block) u = stepStmt (index Bool.true block) u := by
  simp only [index,Bool.false_eq_true,↓reduceIte,stepStmt,evalOp_mul,evalOp_programId,evalOp_constNat,
    bind,Option.bind_some]
  congr 2
  ext i
  exact Nat.mul_comm _ _

theorem index_flat (A : FlatAlloc) (block : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts [index Bool.false block]) u =
      stepStmts (A.flattenStmts [index Bool.true block]) u := by
  have h : stepStmt (A.flattenStmt (index Bool.false block)) u =
      stepStmt (A.flattenStmt (index Bool.true block)) u := by
    simpa only [index,Bool.false_eq_true,↓reduceIte,FlatAlloc.flattenStmt,FlatAlloc.flattenOp] using index_step block u
  simp only [FlatAlloc.flattenStmts,stepStmts,h]


theorem forward_body (swapped : Bool) (s_ptr x_ptr lambda_ptr y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    (Erasure.forward swapped s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body =
      [index swapped BLOCK_SIZE] ++ (Erasure.forward Bool.false s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body.drop 1 := by
  cases swapped <;> rfl

theorem forward_flat_exec (A : FlatAlloc) (storage : Storage) (s_ptr x_ptr lambda_ptr y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) (u : BlockState) :
    exec (A.flattenKernel (TypedSource.forward storage Bool.false s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel) u =
      exec (A.flattenKernel (TypedSource.forward storage Bool.true s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel) u := by
  unfold exec FlatAlloc.flattenKernel
  simp only [Erasure.forward_body]
  rw [forward_body Bool.false,forward_body Bool.true]
  simp only [FlatAlloc.flattenStmts_append,ProgramRewrite.append,index_flat]

theorem backward_body (swapped : Bool) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    (Erasure.backward swapped s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body =
      [index swapped BLOCK_SIZE] ++ (Erasure.backward Bool.false s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body.drop 1 := by
  cases swapped <;> rfl

theorem backward_flat_exec (A : FlatAlloc) (storage : Storage) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) (u : BlockState) :
    exec (A.flattenKernel (TypedSource.backward storage Bool.false s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel) u =
      exec (A.flattenKernel (TypedSource.backward storage Bool.true s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel) u := by
  unfold exec FlatAlloc.flattenKernel
  simp only [Erasure.backward_body]
  rw [backward_body Bool.false,backward_body Bool.true]
  simp only [FlatAlloc.flattenStmts_append,ProgramRewrite.append,index_flat]

theorem complexForward_body (swapped : Bool) (s_ptr x_ptr y_ptr lambda_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    (Erasure.complexForward swapped s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body =
      [index swapped BLOCK_SIZE] ++ (Erasure.complexForward Bool.false s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body.drop 1 := by
  cases swapped <;> rfl

theorem complexForward_flat_exec (A : FlatAlloc) (s_ptr x_ptr y_ptr lambda_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) (u : BlockState) :
    exec (A.flattenKernel (TypedSource.complexForward Bool.false s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel) u =
      exec (A.flattenKernel (TypedSource.complexForward Bool.true s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel) u := by
  unfold exec FlatAlloc.flattenKernel
  simp only [Erasure.complexForward_body]
  rw [complexForward_body Bool.false,complexForward_body Bool.true]
  simp only [FlatAlloc.flattenStmts_append,ProgramRewrite.append,index_flat]

theorem complexBackward_body (swapped : Bool) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    (Erasure.complexBackward swapped s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body =
      [index swapped BLOCK_SIZE] ++ (Erasure.complexBackward Bool.false s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body.drop 1 := by
  cases swapped <;> rfl

theorem complexBackward_flat_exec (A : FlatAlloc) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) (u : BlockState) :
    exec (A.flattenKernel (TypedSource.complexBackward Bool.false s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel) u =
      exec (A.flattenKernel (TypedSource.complexBackward Bool.true s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel) u := by
  unfold exec FlatAlloc.flattenKernel
  simp only [Erasure.complexBackward_body]
  rw [complexBackward_body Bool.false,complexBackward_body Bool.true]
  simp only [FlatAlloc.flattenStmts_append,ProgramRewrite.append,index_flat]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.Candidate
