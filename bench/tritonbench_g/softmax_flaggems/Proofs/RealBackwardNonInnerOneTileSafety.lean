import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerOneTileIO
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerSafetyTransport

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem one_source_safe (bounds : RegionBounds) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) (hr : s.pids 0<M)
    (hP : M*N*K≤bounds P) (hG : M*N*K≤bounds G) (hY : M*N*K≤bounds Y) :
    (source P G Y M N K B C Bool.true).TraceSafe bounds s := by
  unfold Kernel.TraceSafe
  simp [source, softmax_backward_kernel_non_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    MaskOpt.Active, MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt, evalOp.eq_def,
    Option.bind_eq_bind, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
  constructor
  · intro i k hn hk
    exact Nat.lt_of_lt_of_le (RealAddress.non_inner_bound hr hn hk) hP
  constructor
  · intro i k hn hk
    exact Nat.lt_of_lt_of_le (RealAddress.non_inner_bound hr hn hk) hG
  · intro i k hn hk
    exact Nat.lt_of_lt_of_le (RealAddress.non_inner_bound hr hn hk) hY

theorem one_original_safe (storage : Storage) (bounds : RegionBounds) (s : BlockState)
    (P G Y : RegionName) (M N K B C : Nat) (hr : s.pids 0<M)
    (hP : M*N*K≤bounds P) (hG : M*N*K≤bounds G) (hY : M*N*K≤bounds Y) :
    (softmax_backward_kernel_non_inner storage P G Y M N K B C Bool.true).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [RealProjection.softmax_backward_kernel_non_inner_erased]
  exact one_source_safe bounds s P G Y M N K B C hr hP hG hY

private theorem one_indexPrefix_safe (bounds : RegionBounds) (kind : RealCandidate.PrefixKind)
    (swapped : Bool) (B : Nat) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealCandidate.indexPrefix kind swapped B) s := by
  cases kind <;> cases swapped <;>
    simp [RealCandidate.indexPrefix, Stmt.TraceSafeList, Stmt.TraceSafe,
      Op.SafeAt.eq_def, stepStmts, stepStmt, evalOp.eq_def]

private theorem one_indexPrefix_exec (kind : RealCandidate.PrefixKind) (B : Nat) (s : BlockState) :
    stepStmts (RealCandidate.indexPrefix kind Bool.false B) s =
      stepStmts (RealCandidate.indexPrefix kind Bool.true B) s := by
  cases kind <;> simp [RealCandidate.indexPrefix, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, NumericDType.mul, NumericDType.add, Nat.mul_comm]

theorem one_candidate_safe (storage : Storage) (bounds : RegionBounds) (s : BlockState)
    (P G Y : RegionName) (M N K B C : Nat) (hr : s.pids 0<M)
    (hP : M*N*K≤bounds P) (hG : M*N*K≤bounds G) (hY : M*N*K≤bounds Y) :
    (softmax_backward_kernel_non_inner_candidate storage P G Y M N K B C Bool.true).eraseDType.toAlgKernel.TraceSafe bounds s := by
  have h := one_original_safe storage bounds s P G Y M N K B C hr hP hG hY
  unfold Kernel.TraceSafe at h ⊢
  rw [RealCandidate.backward_non_inner_original_body, RealTrace.append_iff] at h
  rw [RealCandidate.backward_non_inner_body, RealTrace.append_iff]
  refine ⟨one_indexPrefix_safe bounds _ _ _ _, ?_⟩
  intro next hn
  apply h.2 next
  rw [one_indexPrefix_exec]
  exact hn

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
