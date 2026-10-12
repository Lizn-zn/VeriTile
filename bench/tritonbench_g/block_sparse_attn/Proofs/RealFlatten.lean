import bench.tritonbench_g.block_sparse_attn.Proofs.RealLoopSafety

/-! The entire checked real source is supported by the shared flat-memory
bridge, including signed CSR reads/ranges, both channel blocks and all masks. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealFlatten
open VeriTile.Triton
set_option linter.unusedSimpArgs false

theorem append_ok (xs ys : List Stmt) :
    StmtList.FlattenOk (xs ++ ys) ↔ StmtList.FlattenOk xs ∧ StmtList.FlattenOk ys := by
  induction xs with
  | nil => simp [StmtList.FlattenOk]
  | cons x xs ih => simp [StmtList.FlattenOk, ih, and_assoc]

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem source_ok (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    (RealProgram.kernel swapped BM BN BD p).toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk, RealProgram.body_split, RealPrologue.source_split,
    RealStores.matches_source]
  cases swapped
  all_goals
    simp [append_ok, RealProgram.loop_split, RealScore.prefix_matches,
      RealNormalizationSource.matches_source, RealAccumulation.matches_source,
      RealSetup.code, RealQuery.code, RealQuery.branch, RealQuery.load, RealQuery.pointer, RealQuery.mask,
      RealRows.code, RealRows.pointer, RealRows.load, RealColumn.code, RealColumn.offset,
      RealScore.code, RealScore.dot, RealScore.scaling, RealScore.masking, RealScore.causal,
      RealKey.code, RealKey.pointer, RealKey.mask, RealNormalizationSource.code,
      RealAccumulation.code, RealAccumulation.loadDot, RealAccumulationSource.valueDot,
      RealAccumulationSource.commit, RealValueSource.code, RealValueSource.pointer, RealValueSource.mask,
      RealStores.code, RealStores.offsetOp, RealStores.store, RealStores.pointer,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealFlatten
