import bench.tritonbench_g.max_reduction.Proofs.Source
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.MaxReduction
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

theorem stage1_prefix_flat (A : FlatAlloc) (inp out : RegionName) (M B : Nat)
    (s : BlockState) :
    stepStmts (A.flattenStmts ((mathStage1 Bool.false inp out M B).toAlgKernel.body.take 2)) s =
      stepStmts (A.flattenStmts ((mathStage1 Bool.true inp out M B).toAlgKernel.body.take 2)) s := by
  simp [mathStage1, max_kernel_1, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, FlatAlloc.flattenStmts,
    FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, NumericDType.mul, Nat.mul_comm]

theorem stage2_prefix_flat (A : FlatAlloc) (inp out : RegionName) (M B : Nat)
    (s : BlockState) :
    stepStmts (A.flattenStmts ((mathStage2 Bool.false inp out M B).toAlgKernel.body.take 3)) s =
      stepStmts (A.flattenStmts ((mathStage2 Bool.true inp out M B).toAlgKernel.body.take 3)) s := by
  have hc (i : Fin B) : (!decide (M ≤ i.val)) = decide (i.val < M) := by
    by_cases h : i.val < M
    · simp [h, Nat.not_le.mpr h]
    · simp [h, Nat.le_of_not_gt h]
  simp [mathStage2, max_kernel_2, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, FlatAlloc.flattenStmts,
    FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, Tile.cop, Tile.uop, ComparableDType.lt, ComparableDType.le,
    hc]
  rfl

theorem stage1_flat_exec (A : FlatAlloc) (storage : Storage) (v : Bool)
    (inp out : RegionName) (M B : Nat) (s : BlockState) :
    exec (A.flattenKernel (stage1 storage v inp out M B).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (max_kernel_1 inp out M B).toAlgKernel) s := by
  rw [stage1_projection]
  cases v
  · rfl
  · change exec (A.flattenKernel (mathStage1 Bool.true inp out M B).toAlgKernel) s =
      exec (A.flattenKernel (mathStage1 Bool.false inp out M B).toAlgKernel) s
    unfold exec FlatAlloc.flattenKernel
    rw [← List.take_append_drop 2 (mathStage1 Bool.true inp out M B).toAlgKernel.body,
      ← List.take_append_drop 2 (mathStage1 Bool.false inp out M B).toAlgKernel.body]
    have hs : (mathStage1 Bool.true inp out M B).toAlgKernel.body.drop 2 =
        (mathStage1 Bool.false inp out M B).toAlgKernel.body.drop 2 := rfl
    simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, stage1_prefix_flat, hs]

theorem stage2_flat_exec (A : FlatAlloc) (storage : Storage) (v : Bool)
    (inp out : RegionName) (M B : Nat) (s : BlockState) :
    exec (A.flattenKernel (stage2 storage v inp out M B).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (max_kernel_2 inp out M B).toAlgKernel) s := by
  rw [stage2_projection]
  cases v
  · rfl
  · change exec (A.flattenKernel (mathStage2 Bool.true inp out M B).toAlgKernel) s =
      exec (A.flattenKernel (mathStage2 Bool.false inp out M B).toAlgKernel) s
    unfold exec FlatAlloc.flattenKernel
    rw [← List.take_append_drop 3 (mathStage2 Bool.true inp out M B).toAlgKernel.body,
      ← List.take_append_drop 3 (mathStage2 Bool.false inp out M B).toAlgKernel.body]
    have hs : (mathStage2 Bool.true inp out M B).toAlgKernel.body.drop 3 =
        (mathStage2 Bool.false inp out M B).toAlgKernel.body.drop 3 := rfl
    simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, stage2_prefix_flat, hs]

theorem dimension_prefix_flat (A : FlatAlloc) (inp ov oi : RegionName)
    (M N K BM BN : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts ((mathDimension Bool.false inp ov oi M N K BM BN).toAlgKernel.body.take 3)) s =
      stepStmts (A.flattenStmts ((mathDimension Bool.true inp ov oi M N K BM BN).toAlgKernel.body.take 3)) s := by
  simp [mathDimension, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind,
    FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, NumericDType.mul, Nat.mul_comm]

theorem dimension_flat_exec (A : FlatAlloc) (storage : Storage) (v : Bool)
    (inp ov oi : RegionName) (M N K BM BN : Nat) (s : BlockState) :
    exec (A.flattenKernel (dimensionSource storage v inp ov oi M N K BM BN).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (mathDimension Bool.false inp ov oi M N K BM BN).toAlgKernel) s := by
  rw [dimension_projection]
  cases v
  · rfl
  · unfold exec FlatAlloc.flattenKernel
    rw [← List.take_append_drop 3 (mathDimension Bool.true inp ov oi M N K BM BN).toAlgKernel.body,
      ← List.take_append_drop 3 (mathDimension Bool.false inp ov oi M N K BM BN).toAlgKernel.body]
    have hs : (mathDimension Bool.true inp ov oi M N K BM BN).toAlgKernel.body.drop 3 =
        (mathDimension Bool.false inp ov oi M N K BM BN).toAlgKernel.body.drop 3 := rfl
    simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, dimension_prefix_flat, hs]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MaxReduction
