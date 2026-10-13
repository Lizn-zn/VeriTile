import bench.tritonbench_g.nested_loops_processing.Proofs.RealOuterInvariant
import VeriTile.Triton.Memory.Frame.Basic

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

/-- The exact set of source stores, including their overlapping tile destinations. -/
def outputFootprint (dst : RegionName) (sm sn : Nat) : WriteFootprint :=
  fun address ↦ address.1 = dst ∧ CopyMath.footprint sm sn address.2

/-- Independent mathematical IO: the ordered oracle retains the last writer,
while every raw memory cell outside the store footprint is unchanged. -/
def RegionCopyContract (program : Kernel) (s : BlockState) (src dst : RegionName)
    (sm sn : Nat) : Prop :=
  ∃ final, exec program s = some final ∧
    BlockState.WriteWithin (outputFootprint dst sm sn) s final ∧
    final.mem dst = CopyMath.expected
      (fun address ↦ MemCell.of .int (s.readMemValue .int src address))
      (s.mem dst) sm sn

theorem source_io (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) :
    RegionCopyContract (source src dst sm sn) s src dst sm sn := by
  refine ⟨finalState s src dst sm sn, source_run src dst sm sn s, ?_, ?_⟩
  · intro r address hnot
    exact (final_frame s src dst sm sn hsep r address hnot).symm
  · exact (final_invariant s src dst sm sn hsep).output.trans
      (CopyMath.expectedPrefix_full _ _ _ _)

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
