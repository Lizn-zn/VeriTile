import bench.tritonbench_g.nested_loops_processing.Proofs.RealMiddleInvariant
import bench.tritonbench_g.nested_loops_processing.Proofs.RealFinish

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem advances_tilePointer (region : RegionName) (sm sn shift count : Nat) :
    advances count (tilePointer region sm sn shift) (2 * sn) =
      tilePointer region sm sn (shift + 2 * count) := by
  induction count with
  | zero => simp [advances]
  | succ n ih =>
    rw [advances, ih, tilePointer_bump]
    congr 1

theorem CopyInvariant.outerState {initial current : BlockState} {src dst : RegionName}
    {sm sn i : Nat} (h : CopyInvariant initial src dst sm sn (12 * i) current)
    (hsep : src ≠ dst) (hi : i < 2) :
    CopyInvariant initial src dst sm sn (12 * i + 12)
      (FullReal.outerState current (tilePointer src sm sn (14 * i))
        (tilePointer dst sm sn (2 * (12 * i))) (2 * sn)) := by
  have hv := h.loaded hsep (14 * i)
  have hp := h.setReg "a1" .int [2,2] ⟨FullReal.loaded current (tilePointer src sm sn (14 * i))⟩
  have hl := hp.middleLoopState hsep hi
  have hf := hl.setReg "a_ptrs" .ptr [2,2]
    ⟨advances 7 (tilePointer src sm sn (14 * i)) (2 * sn)⟩
  simpa only [FullReal.outerState, hv] using hf

theorem CopyInvariant.outerLoopState {initial current : BlockState} {src dst : RegionName}
    {sm sn : Nat} (h : CopyInvariant initial src dst sm sn 0 current) (hsep : src ≠ dst) :
    CopyInvariant initial src dst sm sn 24
      (FullReal.outerLoopState current (tilePointer src sm sn 0) (tilePointer dst sm sn 0) (2 * sn)) := by
  have hfirst := (h.setReg "i" .nat [] (Tile.scalar 0)).outerState
    (i := 0) hsep (by decide)
  simp only [Nat.mul_zero, Nat.zero_add] at hfirst
  have hsecond := (hfirst.setReg "i" .nat [] (Tile.scalar 1)).outerState
    (i := 1) hsep (by decide)
  simpa only [FullReal.outerLoopState, advances_tilePointer, Nat.mul_zero, Nat.zero_add, Nat.mul_one]
    using hsecond

theorem initialA_nonnegative (src : RegionName) (sm sn : Nat) :
    initialA src sm sn = tilePointer src sm sn 0 := by
  funext i
  simp [initialA, tilePointer]
  norm_cast

theorem initialC_nonnegative (dst : RegionName) (sm sn : Nat) :
    initialC dst sm sn = tilePointer dst sm sn 0 := by
  funext i
  change (dst, ((sm * i.1.val + sn * i.2.1.val : Nat) : Int).toNat) =
    (dst, i.1.val * sm + (i.2.1.val + 0) * sn)
  rw [Int.toNat_natCast]
  simp [Nat.mul_comm]

theorem final_invariant (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) :
    CopyInvariant s src dst sm sn 24 (finalState s src dst sm sn) := by
  have hp : CopyInvariant s src dst sm sn 0 (prefixState s src dst sm sn) :=
    ⟨rfl, fun _ _ ↦ rfl⟩
  have hf := hp.outerLoopState hsep
  have hd : (2 * (sn : Int)).toNat = 2 * sn := by omega
  simpa only [finalState, initialA_nonnegative, initialC_nonnegative, hd] using hf

theorem final_frame (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) (r : RegionName) (address : Nat)
    (h : ¬ (r = dst ∧ CopyMath.footprint sm sn address)) :
    (finalState s src dst sm sn).mem r address = s.mem r address := by
  have hi := final_invariant s src dst sm sn hsep
  by_cases hr : r = dst
  · subst r
    rw [hi.output, CopyMath.expectedPrefix_full]
    exact CopyMath.expected_frame _ _ _ _ _ (by simpa using h)
  · exact congrFun (hi.frame r hr) address

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
