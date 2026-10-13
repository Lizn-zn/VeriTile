import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerSafetyTransport
import bench.tritonbench_g.softmax_flaggems.Proofs.RealFlatFrame
import VeriTile.Triton.Memory.KernelCore

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem original_flattenOk (input output : Storage) (Y X : RegionName) (M N B : Nat) :
    (softmax_kernel_inner input output Y X M N B Bool.false).eraseDType.toAlgKernel.FlattenOk := by
  rw [RealProjection.softmax_kernel_inner_erased]
  unfold Kernel.FlattenOk
  simp [softmax_kernel_inner,
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
theorem original_flat_io (input output : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y) (hr : s.pids 0 < M)
    (hX : M * N ≤ A.extent X) (hY : M * N ≤ A.extent Y)
    (hYr : Y ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    ∃ final,
      exec (A.flattenKernel (softmax_kernel_inner input output Y X M N B Bool.false).eraseDType.toAlgKernel)
        (A.flattenState s) = some final ∧
      BlockState.WriteWithin
        (RealFlatFrame.footprint A (outputFootprint Y (s.pids 0) N))
        (A.flattenState s) final ∧
      (∀ n, n < N →
        final.readMem A.flat (A.addr Y (s.pids 0 * N + n)) =
          expected s X (s.pids 0) N n) := by
  obtain ⟨final, hexec, hframe, hvalues, _, _, _⟩ := original_io input output s Y X M N B hN hB hXY
  refine ⟨A.flattenState final, ?_, RealFlatFrame.transport A _ s final hframe, ?_⟩
  · rw [A.exec_flatten hd hcov _ s
      (original_safe input output A.extent s Y X M N B hN hB hXY hr hX hY)
      (original_flattenOk input output Y X M N B) hu, hexec, Option.map_some]
  · intro n hn
    have hb : s.pids 0 * N + n < A.extent Y := by
      apply Nat.lt_of_lt_of_le _ hY
      simpa using RealAddress.non_inner_bound hr (show 0 < 1 by decide) hn
    rw [A.flattenState_readMem hd final hYr hb]
    exact hvalues n hn

 theorem candidate_flat_io (input output : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y) (hr : s.pids 0 < M)
    (hX : M * N ≤ A.extent X) (hY : M * N ≤ A.extent Y)
    (hYr : Y ∈ A.regions) (hu : s.undef = fun _ _ ↦ 0) :
    ∃ final,
      exec (A.flattenKernel (softmax_kernel_inner_candidate input output Y X M N B Bool.false).eraseDType.toAlgKernel)
        (A.flattenState s) = some final ∧
      BlockState.WriteWithin
        (RealFlatFrame.footprint A (outputFootprint Y (s.pids 0) N))
        (A.flattenState s) final ∧
      (∀ n, n < N →
        final.readMem A.flat (A.addr Y (s.pids 0 * N + n)) =
          expected s X (s.pids 0) N n) := by
  rw [← RealCandidate.forward_inner_flat_exec]
  exact original_flat_io input output A hd hcov s Y X M N B hN hB hXY hr hX hY hYr hu

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
