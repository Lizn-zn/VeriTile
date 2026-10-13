import bench.tritonbench_g.l2_norm_bwd.Proofs.Source

namespace VeriTile.Bench.TritonBenchG.L2NormBwd
open VeriTile.Triton

theorem base_flat (A : FlatAlloc) (r : RegionName) (reg : RegName)
    (stride : Nat) (s : BlockState) :
    stepStmt (A.flattenStmt (baseStmt Bool.false r reg stride)) s =
      stepStmt (A.flattenStmt (baseStmt Bool.true r reg stride)) s := by
  simp [baseStmt,Bool.false_eq_true,↓reduceIte,FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def,stepStmt,evalOp.eq_def,Tile.bop,NumericDType.mul,Nat.mul_comm]

theorem bwd_flat_exec (A : FlatAlloc) (storage gradient : Storage) (v : Bool)
    (X DY DX : RegionName) (stride N : Nat) (eps : ℝ) (B : Nat) (s : BlockState) :
    exec (A.flattenKernel (typedBwd storage gradient v X DY DX stride N eps B).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (l2_norm_bwd_kernel X DY DX stride N eps B).toAlgKernel) s := by
  rw [bwd_projection]
  cases v
  · rfl
  · change exec (A.flattenKernel (mathBwd Bool.true X DY DX stride N eps B)) s =
      exec (A.flattenKernel (mathBwd Bool.false X DY DX stride N eps B)) s
    simp only [exec,FlatAlloc.flattenKernel,mathBwd,FlatAlloc.flattenStmts,
      List.map_cons,stepStmts,← base_flat]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormBwd
