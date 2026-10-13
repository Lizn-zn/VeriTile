import bench.tritonbench_g.nested_loops_processing.Proofs.RealInvariant
import bench.tritonbench_g.nested_loops_processing.Proofs.RealLoop

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem CopyInvariant.innerLoopState {initial current : BlockState} {src dst : RegionName}
    {sm sn count shift v1shift v2shift : Nat}
    (h : CopyInvariant initial src dst sm sn count current) (hsep : src ≠ dst)
    (h00 : CopyMath.inputShift count = v1shift)
    (h01 : CopyMath.inputShift (count + 1) = v2shift)
    (h02 : shift + 2 = CopyMath.inputShift (count + 2))
    (h10 : CopyMath.inputShift (count + 3) = v1shift)
    (h11 : CopyMath.inputShift (count + 4) = v2shift)
    (h12 : shift + 4 = CopyMath.inputShift (count + 5)) :
    CopyInvariant initial src dst sm sn (count + 6)
      (FullReal.innerLoopState current (tilePointer src sm sn shift) (tilePointer dst sm sn (2 * count))
        (inputValues initial src sm sn v1shift) (inputValues initial src sm sn v2shift) (2 * sn)) := by
  have hfirst := (h.setReg "k" .nat [] (Tile.scalar 0)).innerState (shift := shift) hsep h02
  rw [h00, h01] at hfirst
  have hsecond := (hfirst.setReg "k" .nat [] (Tile.scalar 1)).innerState
    (shift := shift + 2) hsep (by simpa [Nat.add_assoc] using h12)
  have ha := tilePointer_bump src sm sn shift
  simpa only [FullReal.innerLoopState, ha, tilePointer_bump_count, Nat.add_assoc, h10, h11] using hsecond

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
