import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInnerSafetyTransport
import bench.tritonbench_g.softmax_flaggems.Proofs.RealFlatFrame
import VeriTile.Triton.Memory.KernelCore

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem original_flattenOk (storage : Storage) (P G Y : RegionName) (M N B C : Nat) :
    (softmax_backward_kernel_inner storage P G Y M N C B Bool.false).eraseDType.toAlgKernel.FlattenOk := by
  rw [RealProjection.softmax_backward_kernel_inner_erased]
  unfold Kernel.FlattenOk
  simp [softmax_backward_kernel_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Flat IO uses the flattenState bridge's explicit zero-undef domain. -/
theorem original_flat_io (storage : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0) (s : BlockState) (P G Y : RegionName)
    (M N B C : Nat) (hN : 0<N) (hB : 0<B) (hPY : P≠Y) (hGY : G≠Y)
    (hP : M*N≤A.extent P) (hG : M*N≤A.extent G) (hY : M*N≤A.extent Y)
    (hYr : Y∈A.regions) (hu : s.undef=fun _ _ ↦ 0) :
    ∃ final, exec (A.flattenKernel
        (softmax_backward_kernel_inner storage P G Y M N C B Bool.false).eraseDType.toAlgKernel)
        (A.flattenState s) = some final ∧
      BlockState.WriteWithin
        (RealFlatFrame.footprint A (outputFootprint Y M N (s.pids 0*C) C))
        (A.flattenState s) final ∧
      (∀ r n, r<C → s.pids 0*C+r<M → n<N →
        final.readMem A.flat (A.addr Y (offset N (s.pids 0*C+r) n)) =
          expected s P G M N (RealAddress.rounds N B*B) (s.pids 0*C+r) n) := by
  obtain ⟨final, he, hf, hv, _, _, _⟩ := original_io storage s P G Y M N B C hN hB hPY hGY
  refine ⟨A.flattenState final, ?_, RealFlatFrame.transport A _ s final hf, ?_⟩
  · rw [A.exec_flatten hd hcov _ s
      (original_safe storage A.extent s P G Y M N B C hN hB hPY hGY hP hG hY)
      (original_flattenOk storage P G Y M N B C) hu, he, Option.map_some]
  · intro r n hrC hrM hn
    have hb : offset N (s.pids 0*C+r) n<A.extent Y := by
      apply Nat.lt_of_lt_of_le _ hY
      simpa [offset] using RealAddress.non_inner_bound hrM (show 0<1 by decide) hn
    rw [A.flattenState_readMem hd final hYr hb]
    exact hv r n hrC hrM hn

/-- Flat IO uses the flattenState bridge's explicit zero-undef domain. -/
theorem candidate_flat_io (storage : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0) (s : BlockState) (P G Y : RegionName)
    (M N B C : Nat) (hN : 0<N) (hB : 0<B) (hPY : P≠Y) (hGY : G≠Y)
    (hP : M*N≤A.extent P) (hG : M*N≤A.extent G) (hY : M*N≤A.extent Y)
    (hYr : Y∈A.regions) (hu : s.undef=fun _ _ ↦ 0) :
    ∃ final, exec (A.flattenKernel
        (softmax_backward_kernel_inner_candidate storage P G Y M N C B Bool.false).eraseDType.toAlgKernel)
        (A.flattenState s) = some final ∧
      BlockState.WriteWithin
        (RealFlatFrame.footprint A (outputFootprint Y M N (s.pids 0*C) C))
        (A.flattenState s) final ∧
      (∀ r n, r<C → s.pids 0*C+r<M → n<N →
        final.readMem A.flat (A.addr Y (offset N (s.pids 0*C+r) n)) =
          expected s P G M N (RealAddress.rounds N B*B) (s.pids 0*C+r) n) := by
  rw [←RealCandidate.backward_inner_flat_exec]
  exact original_flat_io storage A hd hcov s P G Y M N B C hN hB hPY hGY hP hG hY hYr hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
