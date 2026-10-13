import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadInvariant
import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadLoop

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem CopyInvariant.innerLoopState {payload : TileDType} {initial current : BlockState} {src dst : RegionName}
    {sm sn count shift v1shift v2shift : Nat}
    (h : CopyInvariant payload initial src dst sm sn count current) (hsep : src ≠ dst)
    (h00 : CopyMath.inputShift count = v1shift)
    (h01 : CopyMath.inputShift (count + 1) = v2shift)
    (h02 : shift + 2 = CopyMath.inputShift (count + 2))
    (h10 : CopyMath.inputShift (count + 3) = v1shift)
    (h11 : CopyMath.inputShift (count + 4) = v2shift)
    (h12 : shift + 4 = CopyMath.inputShift (count + 5)) :
    CopyInvariant payload initial src dst sm sn (count + 6)
      (PayloadReal.innerLoopState payload current (tilePointer src sm sn shift) (tilePointer dst sm sn (2 * count))
        (inputValues payload initial src sm sn v1shift) (inputValues payload initial src sm sn v2shift) (2 * sn)) := by
  have hfirst := (h.setReg "k" .nat [] (Tile.scalar 0)).innerState (shift := shift) hsep h02
  rw [h00, h01] at hfirst
  have hsecond := (hfirst.setReg "k" .nat [] (Tile.scalar 1)).innerState
    (shift := shift + 2) hsep (by simpa [Nat.add_assoc] using h12)
  have ha := tilePointer_bump src sm sn shift
  simpa only [PayloadReal.innerLoopState, ha, tilePointer_bump_count, Nat.add_assoc, h10, h11] using hsecond

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
