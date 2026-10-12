import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealFrame

namespace VeriTile.Bench.Tests.ChunkBwdDqkgBounds
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ChunkBwdDqkg

private def layout : RealMath.Layout where
  T := 3
  K := 3
  V := 3
  BT := 2
  BK := 4
  BV := 2
  NT := 2
  s_k_h := 9
  s_k_t := 3
  s_v_h := 9
  s_v_t := 3
  s_h_h := 18
  s_h_t := 3
  scale := 1

private def launch : RealMath.Launch := ⟨0,0,1,2⟩
private def masked : TileIndex [layout.BT,layout.BK] := (⟨0,by decide⟩,⟨3,by decide⟩,PUnit.unit)
private def active : TileIndex [layout.BT,layout.BK] := (⟨1,by decide⟩,⟨0,by decide⟩,PUnit.unit)

/-- A BK tail can have a virtual address equal to the next row's active
address. Treating the whole tile as unmasked would reject this valid layout. -/
theorem masked_lane_aliases_active :
    ¬ RealFrame.qkActive layout launch masked ∧ RealFrame.qkActive layout launch active ∧
      RealFrame.qkAddress layout launch masked = RealFrame.qkAddress layout launch active := by
  norm_num [RealFrame.qkActive, RealFrame.qkAddress, layout, launch, masked, active]

theorem all_lanes_not_injective : ¬ Function.Injective (RealFrame.qkAddress layout launch) := by
  intro h
  have bad := h masked_lane_aliases_active.2.2
  exact (by decide : masked ≠ active) bad

/-- The contract now needs injectivity only where a store actually writes. -/
theorem active_lanes_injective :
    ∀ i j, RealFrame.qkActive layout launch i → RealFrame.qkActive layout launch j →
      RealFrame.qkAddress layout launch i = RealFrame.qkAddress layout launch j → i = j :=
  RealFrame.qk_injective_on layout launch (by decide)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Tests.ChunkBwdDqkgBounds
