import bench.tritonbench_g.rmsnorm_fused.Proofs.Source
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.RmsnormFused
open VeriTile.Triton
set_option maxHeartbeats 2000000

def pointerPrefix (v : Bool) (Y : RegionName) (stride : Nat) : List Stmt :=
  [.assign .nat [] "row" (.programId 0),
   .assign .ptr [] "Y" (.ptrAdd .nil (.ptrBase Y)
     (if v then .mul .nat .nil (.constNat stride) (.ref .nat [] "row")
      else .mul .nat .nil (.ref .nat [] "row") (.constNat stride)))]

theorem prefix_source (v : Bool) (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (mathKernel v X Y W stride N B eps).body.take 2 = pointerPrefix v Y stride := by
  cases v <;> rfl

theorem prefix_flat (A : FlatAlloc) (Y : RegionName) (stride : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts (pointerPrefix Bool.true Y stride)) s =
      stepStmts (A.flattenStmts (pointerPrefix Bool.false Y stride)) s := by
  simp [pointerPrefix, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, NumericDType.mul, Nat.mul_comm]

/-- Full state transport changes one address product and retains both loops. -/
theorem typed_flat_exec (A : FlatAlloc) (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) (s : BlockState) :
    exec (A.flattenKernel
      (typedKernel input weight output v X Y W stride N B eps).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (rms_norm_fwd_fused X Y W stride N B eps).toAlgKernel) s := by
  rw [typed_projection]
  cases v
  · rfl
  · change exec (A.flattenKernel (mathKernel Bool.true X Y W stride N B eps)) s =
      exec (A.flattenKernel (mathKernel Bool.false X Y W stride N B eps)) s
    unfold exec FlatAlloc.flattenKernel
    rw [← List.take_append_drop 2 (mathKernel Bool.true X Y W stride N B eps).body,
      ← List.take_append_drop 2 (mathKernel Bool.false X Y W stride N B eps).body]
    have ht : (mathKernel Bool.true X Y W stride N B eps).body.drop 2 =
        (mathKernel Bool.false X Y W stride N B eps).body.drop 2 := rfl
    rw [prefix_source, prefix_source]
    simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, prefix_flat, ht]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormFused
