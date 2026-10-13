import bench.tritonbench_g.square_matrix.Proofs.Source
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.SquareMatrix
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

def indexPrefix (v : Bool) (inp : RegionName) (si : Nat) : List Stmt :=
  [.assign .nat [] "row_idx" (.programId 0),
   .assign .ptr [] "row_start_ptr" (.ptrAdd .nil (.ptrBase inp)
     (if v then .mul .nat .nil (.constNat si) (.ref .nat [] "row_idx")
      else .mul .nat .nil (.ref .nat [] "row_idx") (.constNat si)))]

theorem prefix_flat (A : FlatAlloc) (inp : RegionName) (si : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts (indexPrefix Bool.false inp si)) s =
      stepStmts (A.flattenStmts (indexPrefix Bool.true inp si)) s := by
  simp [indexPrefix, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.ptrAdd, NumericDType.mul,
    Nat.mul_comm]

theorem math_body (v : Bool) (out inp : RegionName) (si so n B : Nat) :
    (mathKernel v out inp si so n B).toAlgKernel.body =
      indexPrefix v inp si ++ (square_kernel out inp si so n B).toAlgKernel.body.drop 2 := by
  cases v <;> rfl

/-- The changed address prefix preserves even flat executions and failure. -/
theorem typed_flat_exec (A : FlatAlloc) (storage : Storage) (v : Bool)
    (out inp : RegionName) (si so n B : Nat) (s : BlockState) :
    exec (A.flattenKernel (typedKernel storage v out inp si so n B).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (square_kernel out inp si so n B).toAlgKernel) s := by
  rw [typed_projection]
  cases v
  · rfl
  · change exec (A.flattenKernel (mathKernel Bool.true out inp si so n B).toAlgKernel) s =
      exec (A.flattenKernel (mathKernel Bool.false out inp si so n B).toAlgKernel) s
    unfold exec FlatAlloc.flattenKernel
    rw [math_body Bool.true, math_body Bool.false]
    simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, prefix_flat]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SquareMatrix
