import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerSafetyTransport
import bench.tritonbench_g.softmax_flaggems.Proofs.RealFlatFrame
import VeriTile.Triton.Memory.KernelCore

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem original_flattenOk (storage : Storage) (P G Y : RegionName) (M N K B C : Nat) :
    (softmax_backward_kernel_non_inner storage P G Y M N K B C Bool.false).eraseDType.toAlgKernel.FlattenOk := by
  rw [RealProjection.softmax_backward_kernel_non_inner_erased]
  unfold Kernel.FlattenOk
  simp [softmax_backward_kernel_non_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Flat-memory form of the successful source IO theorem. The zero-undef
premise is the explicit premise of the existing flattenState bridge; the
region theorem above retains arbitrary undefined-padding values. -/
theorem original_flat_io (storage : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (P G Y : RegionName) (M N K B C : Nat)
    (hN : 0 < N) (hB : 0 < B) (hPY : P ≠ Y) (hGY : G ≠ Y) (hr : s.pids 0 < M)
    (hP : M * N * K ≤ A.extent P) (hG : M * N * K ≤ A.extent G) (hY : M * N * K ≤ A.extent Y)
    (hYr : Y ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    ∃ final,
      exec (A.flattenKernel (softmax_backward_kernel_non_inner storage P G Y M N K B C Bool.false).eraseDType.toAlgKernel)
        (A.flattenState s) = some final ∧
      BlockState.WriteWithin
        (RealFlatFrame.footprint A (outputFootprint Y (s.pids 0) N K (s.pids 1 * C) C))
        (A.flattenState s) final ∧
      (∀ n k, n < N → k < C → s.pids 1 * C + k < K →
        final.readMem A.flat (A.addr Y (s.pids 0 * N * K + n * K + (s.pids 1 * C + k))) =
          expected s P G (s.pids 0) N K (RealAddress.rounds N B * B) n (s.pids 1 * C + k)) := by
  obtain ⟨final, hexec, hframe, hvalues, _, _, _⟩ := original_io storage s P G Y M N K B C hN hB hPY hGY
  refine ⟨A.flattenState final, ?_, RealFlatFrame.transport A _ s final hframe, ?_⟩
  · rw [A.exec_flatten hd hcov _ s
      (original_safe storage A.extent s P G Y M N K B C hN hB hPY hGY hr hP hG hY)
      (original_flattenOk storage P G Y M N K B C) hu, hexec, Option.map_some]
  · intro n k hn hk hbound
    rw [A.flattenState_readMem hd final hYr
      (Nat.lt_of_lt_of_le (RealAddress.non_inner_bound hr hn hbound) hY)]
    exact hvalues n k hn hk hbound

 theorem candidate_flat_io (storage : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (P G Y : RegionName) (M N K B C : Nat)
    (hN : 0 < N) (hB : 0 < B) (hPY : P ≠ Y) (hGY : G ≠ Y) (hr : s.pids 0 < M)
    (hP : M * N * K ≤ A.extent P) (hG : M * N * K ≤ A.extent G) (hY : M * N * K ≤ A.extent Y)
    (hYr : Y ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    ∃ final,
      exec (A.flattenKernel (softmax_backward_kernel_non_inner_candidate storage P G Y M N K B C Bool.false).eraseDType.toAlgKernel)
        (A.flattenState s) = some final ∧
      BlockState.WriteWithin
        (RealFlatFrame.footprint A (outputFootprint Y (s.pids 0) N K (s.pids 1 * C) C))
        (A.flattenState s) final ∧
      (∀ n k, n < N → k < C → s.pids 1 * C + k < K →
        final.readMem A.flat (A.addr Y (s.pids 0 * N * K + n * K + (s.pids 1 * C + k))) =
          expected s P G (s.pids 0) N K (RealAddress.rounds N B * B) n (s.pids 1 * C + k)) := by
  rw [← RealCandidate.backward_non_inner_flat_exec]
  exact original_flat_io storage A hd hcov s P G Y M N K B C hN hB hPY hGY hr hP hG hY hYr hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
