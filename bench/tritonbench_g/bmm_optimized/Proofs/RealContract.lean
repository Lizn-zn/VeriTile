import bench.tritonbench_g.bmm_optimized.Proofs.RealViews
import bench.tritonbench_g.bmm_optimized.Proofs.RealIOBridge

/-! Complete original and dot-first implementations of the independent
initial-array product, including actual access safety, successful flat
execution and the full memory frame at every valid runtime grid coordinate. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealContract
open VeriTile.Triton
noncomputable section

theorem implements (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hh : RealMemory.Host BM BN BK p) :
    (RealViews.io swapped BM BN BK p).Implements RealIO.expected := by
  apply RealIO.IO.Implements.intro _ _ (RealPureSafety.flatten_ok swapped BM BN BK p)
  · intro bounds s _ hl hu _ hb
    exact RealViews.safe swapped s BM BN BK p hh hl hu bounds hb
  · intro s xs hl hu hp
    exact RealViews.region_executes swapped s BM BN BK p hh hl hu xs hp

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealContract
