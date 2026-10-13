import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerOneTileSafety
import bench.tritonbench_g.softmax_flaggems.Proofs.RealFlatFrame
import VeriTile.Triton.Memory.KernelCore

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem original_flattenOk (input output : Storage) (Y X : RegionName) (M N B : Nat) :
    (softmax_kernel_inner input output Y X M N B Bool.true).eraseDType.toAlgKernel.FlattenOk := by
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

/-- Full flat-memory IO; zero undefined-padding is the explicit flattenState bridge domain. -/
theorem original_flat_io (input output : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0) (s : BlockState) (Y X : RegionName)
    (M N B : Nat) (hN : 0<N) (hB : 0<B) (hNB : N≤B) (hr : s.pids 0<M)
    (hX : M*N≤A.extent X) (hY : M*N≤A.extent Y) (hYr : Y∈A.regions)
    (hu : s.undef=fun _ _ ↦ 0) :
    ∃ final, exec (A.flattenKernel
        (softmax_kernel_inner input output Y X M N B Bool.true).eraseDType.toAlgKernel)
        (A.flattenState s)=some final ∧
      BlockState.WriteWithin
        (RealFlatFrame.footprint A (RealForwardInner.outputFootprint Y (s.pids 0) N))
        (A.flattenState s) final ∧
      (∀ n, n<N → final.readMem A.flat (A.addr Y (s.pids 0*N+n)) =
        RealForwardInner.expected s X (s.pids 0) N n) := by
  obtain ⟨final,he,hf,hv,_,_,_⟩ := original_io input output s Y X M N B hN hB hNB
  refine ⟨A.flattenState final,?_,RealFlatFrame.transport A _ s final hf,?_⟩
  · rw [A.exec_flatten hd hcov _ s
      (original_safe input output A.extent s Y X M N B hB hr hX hY)
      (original_flattenOk input output Y X M N B) hu,he,Option.map_some]
  · intro n hn
    have hb : s.pids 0*N+n<A.extent Y := by
      apply Nat.lt_of_lt_of_le _ hY
      simpa using RealAddress.non_inner_bound hr (show 0<1 by decide) hn
    rw [A.flattenState_readMem hd final hYr hb]
    exact hv n hn

theorem candidate_flat_io (input output : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0) (s : BlockState) (Y X : RegionName)
    (M N B : Nat) (hN : 0<N) (hB : 0<B) (hNB : N≤B) (hr : s.pids 0<M)
    (hX : M*N≤A.extent X) (hY : M*N≤A.extent Y) (hYr : Y∈A.regions)
    (hu : s.undef=fun _ _ ↦ 0) :
    ∃ final, exec (A.flattenKernel
        (softmax_kernel_inner_candidate input output Y X M N B Bool.true).eraseDType.toAlgKernel)
        (A.flattenState s)=some final ∧
      BlockState.WriteWithin
        (RealFlatFrame.footprint A (RealForwardInner.outputFootprint Y (s.pids 0) N))
        (A.flattenState s) final ∧
      (∀ n, n<N → final.readMem A.flat (A.addr Y (s.pids 0*N+n)) =
        RealForwardInner.expected s X (s.pids 0) N n) := by
  rw [←RealCandidate.forward_inner_flat_exec]
  exact original_flat_io input output A hd hcov s Y X M N B hN hB hNB hr hX hY hYr hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile
