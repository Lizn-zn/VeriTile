import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadLoopInvariant
import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadMiddle

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem CopyInvariant.middleState {payload : TileDType} {initial current : BlockState} {src dst : RegionName}
    {sm sn count shift v1shift : Nat}
    (h : CopyInvariant payload initial src dst sm sn count current) (hsep : src ≠ dst)
    (h00 : CopyMath.inputShift count = v1shift)
    (h01 : CopyMath.inputShift (count + 1) = shift + 2)
    (h02 : CopyMath.inputShift (count + 2) = shift + 4)
    (h10 : CopyMath.inputShift (count + 3) = v1shift)
    (h11 : CopyMath.inputShift (count + 4) = shift + 2)
    (h12 : CopyMath.inputShift (count + 5) = shift + 6) :
    CopyInvariant payload initial src dst sm sn (count + 6)
      (PayloadReal.middleState payload current (tilePointer src sm sn shift) (tilePointer dst sm sn (2 * count))
        (inputValues payload initial src sm sn v1shift) (2 * sn)) := by
  have hv : PayloadReal.loaded payload current (tilePointer src sm sn (shift + 2)) =
      inputValues payload initial src sm sn (shift + 2) := h.loaded hsep _
  have hp : CopyInvariant payload initial src dst sm sn count
      (middlePrefixState payload current (tilePointer src sm sn shift) (2 * sn)) := by
    exact (h.setReg "a_ptrs" .ptr [2,2] ⟨bump (tilePointer src sm sn shift) (2 * sn)⟩).setReg
      "a2" payload [2,2] ⟨PayloadReal.loaded payload current (bump (tilePointer src sm sn shift) (2 * sn))⟩
  have hl := hp.innerLoopState (shift := shift + 2) hsep h00 h01
    (by omega) h10 h11 (by omega)
  simpa only [PayloadReal.middleState, tilePointer_bump, hv] using hl

theorem CopyInvariant.middleState_group {payload : TileDType} {initial current : BlockState} {src dst : RegionName}
    {sm sn i j : Nat} (h : CopyInvariant payload initial src dst sm sn (12 * i + 6 * j) current)
    (hsep : src ≠ dst) (hi : i < 2) (hj : j < 2) :
    CopyInvariant payload initial src dst sm sn (12 * i + 6 * j + 6)
      (PayloadReal.middleState payload current (tilePointer src sm sn (14 * i + 6 * j))
        (tilePointer dst sm sn (2 * (12 * i + 6 * j)))
        (inputValues payload initial src sm sn (14 * i)) (2 * sn)) := by
  have hs := CopyMath.group_schedule hi hj
  exact h.middleState hsep hs.1 hs.2.1 hs.2.2.1 hs.2.2.2.1 hs.2.2.2.2.1 hs.2.2.2.2.2

theorem CopyInvariant.middleLoopState {payload : TileDType} {initial current : BlockState} {src dst : RegionName}
    {sm sn i : Nat} (h : CopyInvariant payload initial src dst sm sn (12 * i) current)
    (hsep : src ≠ dst) (hi : i < 2) :
    CopyInvariant payload initial src dst sm sn (12 * i + 12)
      (PayloadReal.middleLoopState payload current (tilePointer src sm sn (14 * i))
        (tilePointer dst sm sn (2 * (12 * i)))
        (inputValues payload initial src sm sn (14 * i)) (2 * sn)) := by
  have hfirst := (h.setReg "j" .nat [] (Tile.scalar 0)).middleState_group
    (i := i) (j := 0) hsep hi (by decide)
  simp only [Nat.mul_zero, Nat.add_zero] at hfirst
  have hsecond := (hfirst.setReg "j" .nat [] (Tile.scalar 1)).middleState_group
    (i := i) (j := 1) hsep hi (by decide)
  have ha : bump (bump (bump (tilePointer src sm sn (14 * i)) (2 * sn)) (2 * sn))
      (2 * sn) = tilePointer src sm sn (14 * i + 6) := by
    simp only [tilePointer_bump, Nat.add_assoc]
  have hc : bump (bump (bump (bump (bump (bump (tilePointer dst sm sn (2 * (12 * i)))
      (2 * sn)) (2 * sn)) (2 * sn)) (2 * sn)) (2 * sn)) (2 * sn) =
      tilePointer dst sm sn (2 * (12 * i + 6)) := by
    simp only [tilePointer_bump_count, Nat.add_assoc]
  simpa only [PayloadReal.middleLoopState, ha, hc, Nat.mul_one, Nat.add_assoc] using hsecond

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
