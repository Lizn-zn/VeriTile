import bench.tritonbench_g.kldiv_compute.Proofs.Source

namespace VeriTile.Bench.TritonBenchG.KldivCompute
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1600000

/-- Actual source trace safety, including both masked loads and the final masked store. -/
theorem typed_trace_safe (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin B, s.pid * B + j.val < N → s.pid * B + j.val < bounds X)
    (hy : ∀ j : Fin B, s.pid * B + j.val < N → s.pid * B + j.val < bounds Y)
    (ho : ∀ j : Fin B, s.pid * B + j.val < N → s.pid * B + j.val < bounds O) :
    Kernel.TraceSafe bounds (kernel x y output v X Y O N B).eraseDType.toAlgKernel s := by
  rw [typed_projection]
  cases v
  · exact kldivergence_kernel_traceSafe X Y O N B bounds s hx hy ho
  · unfold Kernel.TraceSafe
    simp [mathKernel, commutedMathKernel, ComputeKernel.toAlgKernel,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
      stepStmt, evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop,
      NumericDType.add, NumericDType.mul, NumericDType.div, ComparableDType.lt,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
      MaskOpt.Active, BlockState.setReg, Nat.mul_comm]
    refine ⟨?_, ?_, ?_⟩
    · simpa only [BlockState.pid, Nat.mul_comm] using hx
    · simpa only [BlockState.pid, Nat.mul_comm] using hy
    · simpa only [BlockState.pid, Nat.mul_comm] using ho

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivCompute
