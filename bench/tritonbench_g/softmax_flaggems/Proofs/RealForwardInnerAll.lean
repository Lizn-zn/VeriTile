import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerOneTileFlat
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerFlat

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource

def ExecutionDomain (one : Bool) (X Y : RegionName) (N B : Nat) : Prop :=
  0 < N ∧ 0 < B ∧ (if one then N ≤ B else X ≠ Y)

def IOResult (s final : BlockState) (Y X : RegionName) (N : Nat) : Prop :=
  BlockState.WriteWithin (outputFootprint Y (s.pids 0) N) s final ∧
    (∀ n, n<N →
      final.readMem Y (s.pids 0 * N + n) =
        expected s X (s.pids 0) N n) ∧
    final.pids=s.pids ∧ final.numPids=s.numPids ∧ final.undef=s.undef

theorem original_all_io (input output : Storage) (s : BlockState) (Y X : RegionName)
    (M N B : Nat) (one : Bool) (h : ExecutionDomain one X Y N B) :
    ∃ final, exec (softmax_kernel_inner input output Y X M N B one).eraseDType.toAlgKernel s = some final ∧
      IOResult s final Y X N := by
  cases one
  · obtain ⟨hN,hB,hXY⟩ := h
    exact original_io input output s Y X M N B hN hB hXY
  · exact RealForwardInnerOneTile.original_io input output s Y X M N B h.1 h.2.1 h.2.2

theorem candidate_all_io (input output : Storage) (s : BlockState) (Y X : RegionName)
    (M N B : Nat) (one : Bool) (h : ExecutionDomain one X Y N B) :
    ∃ final, exec (softmax_kernel_inner_candidate input output Y X M N B one).eraseDType.toAlgKernel s = some final ∧
      IOResult s final Y X N := by
  rw [← candidate_exec]
  exact original_all_io input output s Y X M N B one h

theorem original_all_safe (input output : Storage) (bounds : RegionBounds) (s : BlockState)
    (Y X : RegionName) (M N B : Nat) (one : Bool)
    (h : ExecutionDomain one X Y N B) (hr : s.pids 0<M)
    (hX : M*N≤bounds X) (hY : M*N≤bounds Y) :
    (softmax_kernel_inner input output Y X M N B one).eraseDType.toAlgKernel.TraceSafe bounds s := by
  cases one
  · obtain ⟨hN,hB,hXY⟩ := h
    exact original_safe input output bounds s Y X M N B hN hB hXY hr hX hY
  · exact RealForwardInnerOneTile.original_safe input output bounds s Y X M N B h.2.1 hr hX hY

theorem candidate_all_safe (input output : Storage) (bounds : RegionBounds) (s : BlockState)
    (Y X : RegionName) (M N B : Nat) (one : Bool)
    (h : ExecutionDomain one X Y N B) (hr : s.pids 0<M)
    (hX : M*N≤bounds X) (hY : M*N≤bounds Y) :
    (softmax_kernel_inner_candidate input output Y X M N B one).eraseDType.toAlgKernel.TraceSafe bounds s := by
  cases one
  · obtain ⟨hN,hB,hXY⟩ := h
    exact candidate_safe input output bounds s Y X M N B hN hB hXY hr hX hY
  · exact RealForwardInnerOneTile.candidate_safe input output bounds s Y X M N B h.2.1 hr hX hY

def FlatIOResult (A : FlatAlloc) (s final : BlockState) (Y X : RegionName)
    (N : Nat) : Prop :=
  BlockState.WriteWithin
      (RealFlatFrame.footprint A (outputFootprint Y (s.pids 0) N))
      (A.flattenState s) final ∧
    ∀ n, n<N →
      final.readMem A.flat (A.addr Y (s.pids 0 * N + n)) =
        expected s X (s.pids 0) N n

theorem original_all_flat_io (input output : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0) (s : BlockState) (Y X : RegionName)
    (M N B : Nat) (one : Bool) (h : ExecutionDomain one X Y N B)
    (hr : s.pids 0<M) (hX : M*N≤A.extent X)
    (hY : M*N≤A.extent Y) (hYr : Y∈A.regions) (hu : s.undef=fun _ _ ↦ 0) :
    ∃ final, exec (A.flattenKernel
        (softmax_kernel_inner input output Y X M N B one).eraseDType.toAlgKernel)
        (A.flattenState s) = some final ∧ FlatIOResult A s final Y X N := by
  cases one
  · obtain ⟨hN,hB,hXY⟩ := h
    exact original_flat_io input output A hd hcov s Y X M N B hN hB hXY hr hX hY hYr hu
  · exact RealForwardInnerOneTile.original_flat_io input output A hd hcov s Y X M N B h.1 h.2.1 h.2.2 hr hX hY hYr hu

theorem candidate_all_flat_io (input output : Storage) (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0) (s : BlockState) (Y X : RegionName)
    (M N B : Nat) (one : Bool) (h : ExecutionDomain one X Y N B)
    (hr : s.pids 0<M) (hX : M*N≤A.extent X)
    (hY : M*N≤A.extent Y) (hYr : Y∈A.regions) (hu : s.undef=fun _ _ ↦ 0) :
    ∃ final, exec (A.flattenKernel
        (softmax_kernel_inner_candidate input output Y X M N B one).eraseDType.toAlgKernel)
        (A.flattenState s) = some final ∧ FlatIOResult A s final Y X N := by
  rw [← RealCandidate.forward_inner_flat_exec]
  exact original_all_flat_io input output A hd hcov s Y X M N B one h hr hX hY hYr hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
