import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealViews
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealIOBridge
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealSafety

/-! Both complete real sources implement the independent initial-array
contraction with signed sequence filtering and causal block skipping. The
contract includes flat execution, actual access safety and every-cell frame. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealContract
open VeriTile.Triton
noncomputable section

theorem bounds_of_views (swapped : Bool) (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (bounds : RegionBounds) (hb : RealIO.Bounds (RealViews.io swapped BM BN BK p) s bounds) :
    RealSafety.Bounds s BM BN p bounds := by
  refine ⟨?_, ?_, ?_, hb.output⟩
  · intro hactive
    refine ⟨?_, ?_⟩
    · intro i k hi hk
      exact hb.a (i, ⟨k,hk⟩, PUnit.unit) ⟨hactive, hi⟩
    · intro k j hk hj
      exact hb.b (⟨k,hk⟩, j, PUnit.unit) ⟨hactive, hj⟩
  · intro hs hactive i hi
    exact hb.rowIDs hs (i,PUnit.unit) ⟨hactive, hi⟩
  · intro hs hactive j hj
    exact hb.columnIDs hs (j,PUnit.unit) ⟨hactive, hj⟩

theorem implements (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hBK : 0 < BK) (hout : RealViews.OutputDistinct BM BN p) :
    (RealViews.io swapped BM BN BK p).Implements (RealViews.expected BM BN p) := by
  apply RealIO.IO.Implements.intro _ _ (RealSafety.flatten_ok swapped BM BN BK p)
  · intro bounds s _ _ hb
    exact RealSafety.safe swapped s BM BN BK p bounds (bounds_of_views swapped s BM BN BK p bounds hb)
  · intro s xs _ hp
    exact RealViews.region_executes swapped s BM BN BK p hBK hout xs hp

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealContract
