import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadOuterInvariant
import bench.tritonbench_g.nested_loops_processing.Proofs.RealIO
import VeriTile.Triton.Memory.Frame.Basic

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

/-- Independent mathematical IO: the ordered oracle retains the last writer,
while every raw memory cell outside the store footprint is unchanged. Floating
values use the explicitly declared ideal-real storage normalization. -/
def RegionCopyContract (payload : TileDType) (program : Kernel) (s : BlockState) (src dst : RegionName)
    (sm sn : Nat) : Prop :=
  ∃ final, exec program s = some final ∧
    BlockState.WriteWithin (outputFootprint dst sm sn) s final ∧
    final.mem dst = CopyMath.expected
      (fun address ↦ storedCell payload (s.readMemValue payload src address))
      (s.mem dst) sm sn

theorem source_io (payload : TileDType) (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) :
    RegionCopyContract payload (PayloadAlgorithm.program payload src dst sm sn) s src dst sm sn := by
  refine ⟨finalState payload s src dst sm sn, source_run payload src dst sm sn s, ?_, ?_⟩
  · intro r address hnot
    exact (final_frame payload s src dst sm sn hsep r address hnot).symm
  · exact (final_invariant payload s src dst sm sn hsep).output.trans
      (CopyMath.expectedPrefix_full _ _ _ _)

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
