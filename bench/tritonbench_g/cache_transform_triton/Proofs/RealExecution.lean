import bench.tritonbench_g.cache_transform_triton.Proofs.RealSource
import bench.tritonbench_g.cache_transform_triton.Proofs.RealMath
import VeriTile.Triton.Semantics

/-! Successful complete region executions. The prefill maximum needs a
nonempty metadata tile; decoding retains its original full two-dimensional
tile. Access bounds and flat-memory framing are separate IO obligations. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealExecution
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false

section
variable (cos sin : RegionName) (metadata : Region .int) (cosOut sinOut : RegionName)
    (rowStride hiddenStride total H N B : Nat)

def prefillFinish (commuted : Bool) (s : BlockState) (hN : 0 < N) :
    {t : BlockState // exec
      (RealSource.prefill cos sin metadata cosOut sinOut rowStride hiddenStride total H N B commuted)
      s = some t} := by
  cases commuted
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (config := { maxSteps := 100000 }) [RealSource.prefill, TypedSource.prefill,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Except.bind, bind, exec, stepStmts, stepStmt,
      evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
      NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
      ComparableDType.le, Tile.reduceMaxInt, TileShape.discreteMaximum,
      TileShape.axisDim, hN]
    try rfl

def decodeFinish (commuted : Bool) (s : BlockState) :
    {t : BlockState // exec
      (RealSource.decode cos sin metadata cosOut sinOut rowStride hiddenStride H N B commuted)
      s = some t} := by
  cases commuted
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (config := { maxSteps := 100000 }) [RealSource.decode, TypedSource.decode,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Except.bind, bind, exec, stepStmts, stepStmt,
      evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
      NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
      ComparableDType.le]
    try rfl

theorem prefill_executes (commuted : Bool) (s : BlockState) (hN : 0 < N) :
    exec (RealSource.prefill cos sin metadata cosOut sinOut rowStride hiddenStride total H N B commuted) s =
      some (prefillFinish cos sin metadata cosOut sinOut rowStride hiddenStride total H N B commuted s hN).val :=
  (prefillFinish cos sin metadata cosOut sinOut rowStride hiddenStride total H N B commuted s hN).property

theorem decode_executes (commuted : Bool) (s : BlockState) :
    exec (RealSource.decode cos sin metadata cosOut sinOut rowStride hiddenStride H N B commuted) s =
      some (decodeFinish cos sin metadata cosOut sinOut rowStride hiddenStride H N B commuted s).val :=
  (decodeFinish cos sin metadata cosOut sinOut rowStride hiddenStride H N B commuted s).property

/-- The row chosen by the complete prefill execution is the independent signed
formula, for every nonempty metadata length and both source variants. -/
theorem prefill_selected_row (commuted : Bool) (s : BlockState) (hN : 0 < N) :
    (prefillFinish cos sin metadata cosOut sinOut rowStride hiddenStride total H N B commuted s hN).val.regs
      .int [] "ori_seq_idx" = some (Tile.scalar
        (RealMath.relativeRow hN (s.pids 0 * B + s.pids 1)
          (fun i => s.readMemValue .int metadata.cast i.val))) := by
  cases commuted <;>
    simp [prefillFinish, RealMath.relativeRow, RealMath.selectedEnd, Tile.scalar,
      _root_.mul_comm, TileShape.insertAxisIndex, -Tile.scalar_eta] <;> rfl
end

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.RealExecution
