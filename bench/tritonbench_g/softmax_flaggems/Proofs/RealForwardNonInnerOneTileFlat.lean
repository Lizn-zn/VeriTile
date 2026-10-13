import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerOneTileSafetyTransport
import bench.tritonbench_g.softmax_flaggems.Proofs.RealFlatFrame
import VeriTile.Triton.Memory.KernelCore

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem oneTile_original_flattenOk (input output : Storage) (Y X : RegionName) (M N K B C : Nat) :
    (softmax_kernel_non_inner input output Y X M N K B C Bool.true).eraseDType.toAlgKernel.FlattenOk := by
  rw [RealProjection.softmax_kernel_non_inner_erased]
  unfold Kernel.FlattenOk
  simp [softmax_kernel_non_inner,
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
theorem oneTile_original_flat_io (input output : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (Y X : RegionName) (M N K B C : Nat)
    (hN : 0 < N) (hB : 0 < B) (hcover : N ≤ B) (hr : s.pids 0 < M)
    (hX : M * N * K ≤ A.extent X) (hY : M * N * K ≤ A.extent Y)
    (hYr : Y ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    ∃ final,
      exec (A.flattenKernel (softmax_kernel_non_inner input output Y X M N K B C Bool.true).eraseDType.toAlgKernel)
        (A.flattenState s) = some final ∧
      BlockState.WriteWithin
        (RealFlatFrame.footprint A (outputFootprint Y (s.pids 0) N K (s.pids 1 * C) C))
        (A.flattenState s) final ∧
      (∀ n k, n < N → k < C → s.pids 1 * C + k < K →
        final.readMem A.flat (A.addr Y (s.pids 0 * N * K + n * K + (s.pids 1 * C + k))) =
          expected s X (s.pids 0) N K n (s.pids 1 * C + k)) := by
  obtain ⟨final, hexec, hframe, hvalues, _, _, _⟩ := oneTile_original_io input output s Y X M N K B C hN hB hcover
  refine ⟨A.flattenState final, ?_, RealFlatFrame.transport A _ s final hframe, ?_⟩
  · rw [A.exec_flatten hd hcov _ s
      (oneTile_original_safe input output A.extent s Y X M N K B C hB hr hX hY)
      (oneTile_original_flattenOk input output Y X M N K B C) hu, hexec, Option.map_some]
  · intro n k hn hk hbound
    rw [A.flattenState_readMem hd final hYr
      (Nat.lt_of_lt_of_le (RealAddress.non_inner_bound hr hn hbound) hY)]
    exact hvalues n k hn hk hbound

 theorem oneTile_candidate_flat_io (input output : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (Y X : RegionName) (M N K B C : Nat)
    (hN : 0 < N) (hB : 0 < B) (hcover : N ≤ B) (hr : s.pids 0 < M)
    (hX : M * N * K ≤ A.extent X) (hY : M * N * K ≤ A.extent Y)
    (hYr : Y ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    ∃ final,
      exec (A.flattenKernel (softmax_kernel_non_inner_candidate input output Y X M N K B C Bool.true).eraseDType.toAlgKernel)
        (A.flattenState s) = some final ∧
      BlockState.WriteWithin
        (RealFlatFrame.footprint A (outputFootprint Y (s.pids 0) N K (s.pids 1 * C) C))
        (A.flattenState s) final ∧
      (∀ n k, n < N → k < C → s.pids 1 * C + k < K →
        final.readMem A.flat (A.addr Y (s.pids 0 * N * K + n * K + (s.pids 1 * C + k))) =
          expected s X (s.pids 0) N K n (s.pids 1 * C + k)) := by
  rw [← RealCandidate.forward_non_inner_flat_exec]
  exact oneTile_original_flat_io input output A hd hcov s Y X M N K B C hN hB hcover hr hX hY hYr hu

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
