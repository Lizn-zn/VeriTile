import bench.tritonbench_g.matrix_transpose.Proofs.Source
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose
open VeriTile.Triton
set_option maxHeartbeats 100000
set_option maxRecDepth 16000

def indexPrefix (v : Bool) (M : RegionName) (sx sy m d : Nat) : List Stmt :=
  [.assign .nat [m] "size_m_arange" (.arange m),
   .assign .nat [d] "d_head_arange" (.arange d),
   .assign .ptr [m, d] "matrix_ptr" (.ptrAdd .scalarL (.ptrBase M)
     (.add .nat (.consR (.consL .nil))
       (if v then .mul .nat .scalarL (.constNat sx)
          (.expandDim ⟨1, by simp⟩ (.ref .nat [m] "size_m_arange"))
        else .mul .nat .scalarR
          (.expandDim ⟨1, by simp⟩ (.ref .nat [m] "size_m_arange")) (.constNat sx))
       (.mul .nat .scalarR
         (.expandDim ⟨0, by simp⟩ (.ref .nat [d] "d_head_arange")) (.constNat sy))))]

theorem prefix_source (v : Bool) (M Out : RegionName) (sx sy ox oy m d : Nat) :
    (mathKernel v M Out sx sy ox oy m d).toAlgKernel.body.take 3 =
      indexPrefix v M sx sy m d := by
  cases v <;> rfl

/-- The complete changed prefix is memory silent and preserves every register. -/
theorem prefix_flat (A : FlatAlloc) (M Out : RegionName) (sx sy ox oy m d : Nat)
    (s : BlockState) :
    stepStmts (A.flattenStmts ((mathKernel Bool.false M Out sx sy ox oy m d).toAlgKernel.body.take 3)) s =
      stepStmts (A.flattenStmts ((mathKernel Bool.true M Out sx sy ox oy m d).toAlgKernel.body.take 3)) s := by
  rw [prefix_source, prefix_source]
  dsimp [indexPrefix, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.ptrAdd, Tile.expandDim,
    TileShape.insertAxis, TileShape.dropInsertedIndex, NumericDType.mul]
  simp [FlatAlloc.flattenOp.eq_def, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.ptrAdd, Tile.expandDim, TileShape.insertAxis,
    TileShape.dropInsertedIndex, NumericDType.mul, Nat.mul_comm]

/-- Both source versions implement the same complete flat-memory real execution. -/
theorem typed_flat_exec (A : FlatAlloc) (v : Bool) (M Out : RegionName)
    (sx sy ox oy m d : Nat) (s : BlockState) :
    exec (A.flattenKernel (typedKernel v M Out sx sy ox oy m d).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (kernel M Out sx sy ox oy m d).toAlgKernel) s := by
  rw [typed_projection]
  cases v
  · rfl
  · change exec (A.flattenKernel (mathKernel Bool.true M Out sx sy ox oy m d).toAlgKernel) s =
      exec (A.flattenKernel (mathKernel Bool.false M Out sx sy ox oy m d).toAlgKernel) s
    unfold exec FlatAlloc.flattenKernel
    rw [← List.take_append_drop 3 (mathKernel Bool.true M Out sx sy ox oy m d).toAlgKernel.body,
      ← List.take_append_drop 3 (mathKernel Bool.false M Out sx sy ox oy m d).toAlgKernel.body]
    have hs : (mathKernel Bool.true M Out sx sy ox oy m d).toAlgKernel.body.drop 3 =
        (mathKernel Bool.false M Out sx sy ox oy m d).toAlgKernel.body.drop 3 := rfl
    simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, prefix_flat, hs]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatrixTranspose
