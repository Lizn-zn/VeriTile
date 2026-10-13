import bench.tritonbench_g.rms_norm_triton.Proofs.Source

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton

theorem base_flat (A : FlatAlloc) (r : RegionName) (reg : RegName)
    (stride : Nat) (s : BlockState) :
    stepStmt (A.flattenStmt (baseStmt Bool.false r reg stride)) s =
      stepStmt (A.flattenStmt (baseStmt Bool.true r reg stride)) s := by
  simp [baseStmt,Bool.false_eq_true,↓reduceIte,FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def,stepStmt,evalOp.eq_def,Tile.bop,NumericDType.mul,Nat.mul_comm]

theorem typed_flat_exec (A : FlatAlloc) (storage weight : Storage) (v : Bool)
    (Y X W : RegionName) (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) (s : BlockState) :
    exec (A.flattenKernel (typed storage weight v Y X W ysr ysc xsr xsc N eps B).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (rms_norm_kernel Y X W ysr ysc xsr xsc N eps B).toAlgKernel) s := by
  rw [typed_projection]
  cases v
  · rfl
  · change exec (A.flattenKernel (mathKernel Bool.true Y X W ysr ysc xsr xsc N eps B)) s =
      exec (A.flattenKernel (mathKernel Bool.false Y X W ysr ysc xsr xsc N eps B)) s
    simp only [exec,FlatAlloc.flattenKernel,mathKernel,FlatAlloc.flattenStmts,
      List.map_cons,stepStmts,← base_flat]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton
