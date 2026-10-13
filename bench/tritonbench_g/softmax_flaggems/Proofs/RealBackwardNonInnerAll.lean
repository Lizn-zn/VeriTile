import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerOneTileFlat
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerFlat
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerFormula

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource

def paddedLength (one : Bool) (N B : Nat) : Nat :=
  if one then B else RealAddress.rounds N B*B

def ExecutionDomain (one : Bool) (P G Y : RegionName) (N B : Nat) : Prop :=
  if one then N≤B else 0<N ∧ 0<B ∧ P≠Y ∧ G≠Y

def IOResult (s final : BlockState) (P G Y : RegionName) (N K B C : Nat) (one : Bool) : Prop :=
  BlockState.WriteWithin (outputFootprint Y (s.pids 0) N K (s.pids 1*C) C) s final ∧
    (∀ n k, n<N → k<C → s.pids 1*C+k<K →
      final.readMem Y (offset (s.pids 0) N K n (s.pids 1*C+k)) =
        expected s P G (s.pids 0) N K (paddedLength one N B) n (s.pids 1*C+k)) ∧
    final.pids=s.pids ∧ final.numPids=s.numPids ∧ final.undef=s.undef

theorem original_all_io (storage : Storage) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) (one : Bool) (h : ExecutionDomain one P G Y N B) :
    ∃ final, exec (softmax_backward_kernel_non_inner storage P G Y M N K B C one).eraseDType.toAlgKernel s = some final ∧
      IOResult s final P G Y N K B C one := by
  cases one
  · obtain ⟨hN,hB,hPY,hGY⟩ := h
    exact original_io storage s P G Y M N K B C hN hB hPY hGY
  · exact one_original_io storage s P G Y M N K B C h

theorem candidate_all_io (storage : Storage) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) (one : Bool) (h : ExecutionDomain one P G Y N B) :
    ∃ final, exec (softmax_backward_kernel_non_inner_candidate storage P G Y M N K B C one).eraseDType.toAlgKernel s = some final ∧
      IOResult s final P G Y N K B C one := by
  rw [← candidate_exec]
  exact original_all_io storage s P G Y M N K B C one h

theorem original_all_safe (storage : Storage) (bounds : RegionBounds) (s : BlockState)
    (P G Y : RegionName) (M N K B C : Nat) (one : Bool)
    (h : ExecutionDomain one P G Y N B) (hr : s.pids 0<M)
    (hP : M*N*K≤bounds P) (hG : M*N*K≤bounds G) (hY : M*N*K≤bounds Y) :
    (softmax_backward_kernel_non_inner storage P G Y M N K B C one).eraseDType.toAlgKernel.TraceSafe bounds s := by
  cases one
  · obtain ⟨hN,hB,hPY,hGY⟩ := h
    exact original_safe storage bounds s P G Y M N K B C hN hB hPY hGY hr hP hG hY
  · exact one_original_safe storage bounds s P G Y M N K B C hr hP hG hY

theorem candidate_all_safe (storage : Storage) (bounds : RegionBounds) (s : BlockState)
    (P G Y : RegionName) (M N K B C : Nat) (one : Bool)
    (h : ExecutionDomain one P G Y N B) (hr : s.pids 0<M)
    (hP : M*N*K≤bounds P) (hG : M*N*K≤bounds G) (hY : M*N*K≤bounds Y) :
    (softmax_backward_kernel_non_inner_candidate storage P G Y M N K B C one).eraseDType.toAlgKernel.TraceSafe bounds s := by
  cases one
  · obtain ⟨hN,hB,hPY,hGY⟩ := h
    exact candidate_safe storage bounds s P G Y M N K B C hN hB hPY hGY hr hP hG hY
  · exact one_candidate_safe storage bounds s P G Y M N K B C hr hP hG hY

def FlatIOResult (A : FlatAlloc) (s final : BlockState) (P G Y : RegionName)
    (N K B C : Nat) (one : Bool) : Prop :=
  BlockState.WriteWithin
      (RealFlatFrame.footprint A (outputFootprint Y (s.pids 0) N K (s.pids 1*C) C))
      (A.flattenState s) final ∧
    ∀ n k, n<N → k<C → s.pids 1*C+k<K →
      final.readMem A.flat (A.addr Y (offset (s.pids 0) N K n (s.pids 1*C+k))) =
        expected s P G (s.pids 0) N K (paddedLength one N B) n (s.pids 1*C+k)

theorem original_all_flat_io (storage : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) (one : Bool) (h : ExecutionDomain one P G Y N B)
    (hr : s.pids 0<M) (hP : M*N*K≤A.extent P) (hG : M*N*K≤A.extent G)
    (hY : M*N*K≤A.extent Y) (hYr : Y∈A.regions) (hu : s.undef=fun _ _ ↦ 0) :
    ∃ final, exec (A.flattenKernel
        (softmax_backward_kernel_non_inner storage P G Y M N K B C one).eraseDType.toAlgKernel)
        (A.flattenState s) = some final ∧ FlatIOResult A s final P G Y N K B C one := by
  cases one
  · obtain ⟨hN,hB,hPY,hGY⟩ := h
    exact original_flat_io storage A hd hcov s P G Y M N K B C hN hB hPY hGY hr hP hG hY hYr hu
  · exact one_original_flat_io storage A hd hcov s P G Y M N K B C h hr hP hG hY hYr hu

theorem candidate_all_flat_io (storage : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) (one : Bool) (h : ExecutionDomain one P G Y N B)
    (hr : s.pids 0<M) (hP : M*N*K≤A.extent P) (hG : M*N*K≤A.extent G)
    (hY : M*N*K≤A.extent Y) (hYr : Y∈A.regions) (hu : s.undef=fun _ _ ↦ 0) :
    ∃ final, exec (A.flattenKernel
        (softmax_backward_kernel_non_inner_candidate storage P G Y M N K B C one).eraseDType.toAlgKernel)
        (A.flattenState s) = some final ∧ FlatIOResult A s final P G Y N K B C one := by
  rw [← RealCandidate.backward_non_inner_flat_exec]
  exact original_all_flat_io storage A hd hcov s P G Y M N K B C one h hr hP hG hY hYr hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
