import bench.tritonbench_g.kldiv_compute.Proofs.Source
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.KldivCompute
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1000000

def indexPrefix (v : Bool) (B : Nat) : List Stmt :=
  [.assign .nat [] "pid" (.programId 0),
   .assign .nat [] "block_start"
     (if v then .mul .nat .nil (.constNat B) (.ref .nat [] "pid")
      else .mul .nat .nil (.ref .nat [] "pid") (.constNat B))]

theorem prefix_source (v : Bool) (X Y O : RegionName) (N B : Nat) :
    (mathKernel v X Y O N B).body.take 2 = indexPrefix v B := by
  cases v <;> rfl

theorem prefix_flat (A : FlatAlloc) (X Y O : RegionName) (N B : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts ((mathKernel Bool.false X Y O N B).body.take 2)) s =
      stepStmts (A.flattenStmts ((mathKernel Bool.true X Y O N B).body.take 2)) s := by
  rw [prefix_source, prefix_source]
  simp [indexPrefix, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, NumericDType.mul, Nat.mul_comm]

/-- Both complete source variants execute the same real KL term, including every
masked load/store, termination and all memory cells. -/
theorem typed_flat_exec (A : FlatAlloc) (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) (s : BlockState) :
    exec (A.flattenKernel (kernel x y output v X Y O N B).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (kldivergence_kernel X Y O N B).toAlgKernel) s := by
  rw [typed_projection]
  cases v
  · rfl
  · change exec (A.flattenKernel (mathKernel Bool.true X Y O N B)) s =
      exec (A.flattenKernel (mathKernel Bool.false X Y O N B)) s
    unfold exec FlatAlloc.flattenKernel
    rw [← List.take_append_drop 2 (mathKernel Bool.true X Y O N B).body,
      ← List.take_append_drop 2 (mathKernel Bool.false X Y O N B).body]
    have hs : (mathKernel Bool.true X Y O N B).body.drop 2 =
        (mathKernel Bool.false X Y O N B).body.drop 2 := rfl
    simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, prefix_flat, hs]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivCompute
