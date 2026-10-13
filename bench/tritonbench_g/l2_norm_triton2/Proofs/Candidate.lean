import bench.tritonbench_g.l2_norm_triton2.Proofs.Source

namespace VeriTile.Bench.TritonBenchG.L2NormTriton2
open VeriTile.Triton

theorem base_flat (A : FlatAlloc) (r : RegionName) (reg : RegName)
    (stride : Nat) (s : BlockState) :
    stepStmt (A.flattenStmt (baseStmt Bool.false r reg stride)) s =
      stepStmt (A.flattenStmt (baseStmt Bool.true r reg stride)) s := by
  simp [baseStmt,Bool.false_eq_true,↓reduceIte,FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def,stepStmt,evalOp.eq_def,Tile.bop,NumericDType.mul,Nat.mul_comm]

theorem fwd_flat_exec (A : FlatAlloc) (storage : Storage) (v : Bool)
    (X Y : RegionName) (stride N : Nat) (eps : ℝ) (B : Nat) (s : BlockState) :
    exec (A.flattenKernel (typedFwd storage v X Y stride N eps B).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (l2_norm_fwd_1pass_kernel X Y stride N eps B).toAlgKernel) s := by
  rw [fwd_projection]
  cases v
  · rfl
  · change exec (A.flattenKernel (mathFwd Bool.true X Y stride N eps B)) s =
      exec (A.flattenKernel (mathFwd Bool.false X Y stride N eps B)) s
    simp only [exec,FlatAlloc.flattenKernel,mathFwd,FlatAlloc.flattenStmts,
      List.map_cons,stepStmts,← base_flat]

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
end VeriTile.Bench.TritonBenchG.L2NormTriton2
