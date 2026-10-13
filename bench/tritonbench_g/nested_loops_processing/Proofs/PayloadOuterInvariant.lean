import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadMiddleInvariant
import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadFinish
import bench.tritonbench_g.nested_loops_processing.Proofs.RealOuterInvariant

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem CopyInvariant.outerState {payload : TileDType} {initial current : BlockState} {src dst : RegionName}
    {sm sn i : Nat} (h : CopyInvariant payload initial src dst sm sn (12 * i) current)
    (hsep : src ≠ dst) (hi : i < 2) :
    CopyInvariant payload initial src dst sm sn (12 * i + 12)
      (PayloadReal.outerState payload current (tilePointer src sm sn (14 * i))
        (tilePointer dst sm sn (2 * (12 * i))) (2 * sn)) := by
  have hv := h.loaded hsep (14 * i)
  have hp := h.setReg "a1" payload [2,2] ⟨PayloadReal.loaded payload current (tilePointer src sm sn (14 * i))⟩
  have hl := hp.middleLoopState hsep hi
  have hf := hl.setReg "a_ptrs" .ptr [2,2]
    ⟨advances 7 (tilePointer src sm sn (14 * i)) (2 * sn)⟩
  simpa only [PayloadReal.outerState, hv] using hf

theorem CopyInvariant.outerLoopState {payload : TileDType} {initial current : BlockState} {src dst : RegionName}
    {sm sn : Nat} (h : CopyInvariant payload initial src dst sm sn 0 current) (hsep : src ≠ dst) :
    CopyInvariant payload initial src dst sm sn 24
      (PayloadReal.outerLoopState payload current (tilePointer src sm sn 0) (tilePointer dst sm sn 0) (2 * sn)) := by
  have hfirst := (h.setReg "i" .nat [] (Tile.scalar 0)).outerState
    (i := 0) hsep (by decide)
  simp only [Nat.mul_zero, Nat.zero_add] at hfirst
  have hsecond := (hfirst.setReg "i" .nat [] (Tile.scalar 1)).outerState
    (i := 1) hsep (by decide)
  simpa only [PayloadReal.outerLoopState, advances_tilePointer, Nat.mul_zero, Nat.zero_add, Nat.mul_one]
    using hsecond

theorem final_invariant (payload : TileDType) (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) :
    CopyInvariant payload s src dst sm sn 24 (finalState payload s src dst sm sn) := by
  have hp : CopyInvariant payload s src dst sm sn 0 (prefixState s src dst sm sn) :=
    ⟨rfl, fun _ _ ↦ rfl⟩
  have hf := hp.outerLoopState hsep
  have hd : (2 * (sn : Int)).toNat = 2 * sn := by omega
  simpa only [finalState, initialA_nonnegative, initialC_nonnegative, hd] using hf

theorem final_frame (payload : TileDType) (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) (r : RegionName) (address : Nat)
    (h : ¬ (r = dst ∧ CopyMath.footprint sm sn address)) :
    (finalState payload s src dst sm sn).mem r address = s.mem r address := by
  have hi := final_invariant payload s src dst sm sn hsep
  by_cases hr : r = dst
  · subst r
    rw [hi.output, CopyMath.expectedPrefix_full]
    exact CopyMath.expected_frame _ _ _ _ _ (by simpa using h)
  · exact congrFun (hi.frame r hr) address

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
