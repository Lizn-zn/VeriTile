import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInnerOneTileFlat
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInnerFlat
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInnerFormula

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton FullSource

def paddedLength (one : Bool) (N B : Nat) : Nat :=
  if one then B else RealAddress.rounds N B*B

def ExecutionDomain (one : Bool) (P G Y : RegionName) (N B : Nat) : Prop :=
  if one then N≤B else 0<N ∧ 0<B ∧ P≠Y ∧ G≠Y

def IOResult (s final : BlockState) (P G Y : RegionName) (M N B C : Nat) (one : Bool) : Prop :=
  BlockState.WriteWithin (outputFootprint Y M N (s.pids 0*C) C) s final ∧
    (∀ r n, r<C → s.pids 0*C+r<M → n<N →
      final.readMem Y (offset N (s.pids 0*C+r) n) =
        expected s P G M N (paddedLength one N B) (s.pids 0*C+r) n) ∧
    final.pids=s.pids ∧ final.numPids=s.numPids ∧ final.undef=s.undef

theorem original_all_io (storage : Storage) (s : BlockState) (P G Y : RegionName)
    (M N B C : Nat) (one : Bool) (h : ExecutionDomain one P G Y N B) :
    ∃ final, exec (softmax_backward_kernel_inner storage P G Y M N C B one).eraseDType.toAlgKernel s = some final ∧
      IOResult s final P G Y M N B C one := by
  cases one
  · obtain ⟨hN,hB,hPY,hGY⟩ := h
    exact original_io storage s P G Y M N B C hN hB hPY hGY
  · exact one_original_io storage s P G Y M N B C h

theorem candidate_all_io (storage : Storage) (s : BlockState) (P G Y : RegionName)
    (M N B C : Nat) (one : Bool) (h : ExecutionDomain one P G Y N B) :
    ∃ final, exec (softmax_backward_kernel_inner_candidate storage P G Y M N C B one).eraseDType.toAlgKernel s = some final ∧
      IOResult s final P G Y M N B C one := by
  rw [←candidate_exec]
  exact original_all_io storage s P G Y M N B C one h

theorem original_all_safe (storage : Storage) (bounds : RegionBounds) (s : BlockState)
    (P G Y : RegionName) (M N B C : Nat) (one : Bool)
    (h : ExecutionDomain one P G Y N B)
    (hP : M*N≤bounds P) (hG : M*N≤bounds G) (hY : M*N≤bounds Y) :
    (softmax_backward_kernel_inner storage P G Y M N C B one).eraseDType.toAlgKernel.TraceSafe bounds s := by
  cases one
  · obtain ⟨hN,hB,hPY,hGY⟩ := h
    exact original_safe storage bounds s P G Y M N B C hN hB hPY hGY hP hG hY
  · exact one_original_safe storage bounds s P G Y M N B C hP hG hY

theorem candidate_all_safe (storage : Storage) (bounds : RegionBounds) (s : BlockState)
    (P G Y : RegionName) (M N B C : Nat) (one : Bool)
    (h : ExecutionDomain one P G Y N B)
    (hP : M*N≤bounds P) (hG : M*N≤bounds G) (hY : M*N≤bounds Y) :
    (softmax_backward_kernel_inner_candidate storage P G Y M N C B one).eraseDType.toAlgKernel.TraceSafe bounds s := by
  cases one
  · obtain ⟨hN,hB,hPY,hGY⟩ := h
    exact candidate_safe storage bounds s P G Y M N B C hN hB hPY hGY hP hG hY
  · exact one_candidate_safe storage bounds s P G Y M N B C hP hG hY

def FlatIOResult (A : FlatAlloc) (s final : BlockState) (P G Y : RegionName)
    (M N B C : Nat) (one : Bool) : Prop :=
  BlockState.WriteWithin
      (RealFlatFrame.footprint A (outputFootprint Y M N (s.pids 0*C) C))
      (A.flattenState s) final ∧
    ∀ r n, r<C → s.pids 0*C+r<M → n<N →
      final.readMem A.flat (A.addr Y (offset N (s.pids 0*C+r) n)) =
        expected s P G M N (paddedLength one N B) (s.pids 0*C+r) n

theorem original_all_flat_io (storage : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0) (s : BlockState) (P G Y : RegionName)
    (M N B C : Nat) (one : Bool) (h : ExecutionDomain one P G Y N B)
    (hP : M*N≤A.extent P) (hG : M*N≤A.extent G) (hY : M*N≤A.extent Y)
    (hYr : Y∈A.regions) (hu : s.undef=fun _ _ ↦ 0) :
    ∃ final, exec (A.flattenKernel
        (softmax_backward_kernel_inner storage P G Y M N C B one).eraseDType.toAlgKernel)
        (A.flattenState s) = some final ∧ FlatIOResult A s final P G Y M N B C one := by
  cases one
  · obtain ⟨hN,hB,hPY,hGY⟩ := h
    exact original_flat_io storage A hd hcov s P G Y M N B C hN hB hPY hGY hP hG hY hYr hu
  · exact one_original_flat_io storage A hd hcov s P G Y M N B C h hP hG hY hYr hu

theorem candidate_all_flat_io (storage : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0) (s : BlockState) (P G Y : RegionName)
    (M N B C : Nat) (one : Bool) (h : ExecutionDomain one P G Y N B)
    (hP : M*N≤A.extent P) (hG : M*N≤A.extent G) (hY : M*N≤A.extent Y)
    (hYr : Y∈A.regions) (hu : s.undef=fun _ _ ↦ 0) :
    ∃ final, exec (A.flattenKernel
        (softmax_backward_kernel_inner_candidate storage P G Y M N C B one).eraseDType.toAlgKernel)
        (A.flattenState s) = some final ∧ FlatIOResult A s final P G Y M N B C one := by
  rw [←RealCandidate.backward_inner_flat_exec]
  exact original_all_flat_io storage A hd hcov s P G Y M N B C one h hP hG hY hYr hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
