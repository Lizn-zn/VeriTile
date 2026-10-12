import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealViews
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealIOBridge
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealSafety

/-! Complete real correctness: initial Dout/A/residual arrays determine the
masked gradient, with flat termination, reached-address bounds and a complete
frame. Partial contraction/output tiles and all three grid axes stay symbolic. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealContract
open VeriTile.Triton
noncomputable section

theorem bounds_of_views (swapped : Bool) (s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (bounds : RegionBounds) (hb : RealIO.Bounds (RealViews.io swapped BM BN BCS p) s bounds) :
    RealSafety.Bounds s BM BN p bounds := by
  have hlen : RealSetup.limit s p ≤ p.chunk := Nat.min_le_left _ _
  refine ⟨⟨?_, ?_⟩, ?_, ?_⟩
  · intro i k hi hk
    exact hb.dout (i, ⟨k, lt_of_lt_of_le hk hlen⟩, PUnit.unit) ⟨hi, hk⟩
  · intro k j hk hj
    exact hb.a (⟨k, lt_of_lt_of_le hk hlen⟩, j, PUnit.unit) ⟨hk, hj⟩
  · exact hb.residual
  · exact hb.output

/-- Both full sources implement the independent contraction-plus-residual
formula. The live output stride map must be injective; no divisibility of the
contraction length or final output tile is required. -/
theorem implements (swapped : Bool) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (hBCS : 0 < BCS) (hout : RealViews.OutputDistinct BM BN p) :
    (RealViews.io swapped BM BN BCS p).Implements (RealViews.expected BM BN p) := by
  apply RealIO.IO.Implements.intro _ _ (RealSafety.flatten_ok swapped BM BN BCS p)
  · intro bounds s _ _ hb
    exact RealSafety.safe swapped s BM BN BCS p bounds (bounds_of_views swapped s BM BN BCS p bounds hb)
  · intro s xs _ hp
    exact RealViews.region_executes swapped s BM BN BCS p hBCS hout xs hp

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealContract
