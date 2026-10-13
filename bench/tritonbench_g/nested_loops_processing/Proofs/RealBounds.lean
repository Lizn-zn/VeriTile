import bench.tritonbench_g.nested_loops_processing.Proofs.RealMemory
import VeriTile.Triton.Memory.Flatten

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton
set_option maxHeartbeats 300000

/-- Concurrent lanes in one source store must address distinct cells. Overlap
between different stores is retained and resolved by source statement order. -/
def TileInjective (sm sn : Nat) : Prop :=
  Function.Injective (fun i : TileIndex [2,2] ↦ i.1.val * sm + i.2.1.val * sn)

theorem tilePointer_injective (region : RegionName) (sm sn shift : Nat)
    (h : TileInjective sm sn) :
    Function.Injective (fun i ↦ (tilePointer region sm sn shift i).2) := by
  intro i j hij
  apply h
  dsimp [tilePointer] at hij
  simp only [Nat.add_mul, ← Nat.add_assoc] at hij
  exact Nat.add_right_cancel hij

theorem tilePointer_bound (bounds : RegionBounds) (region : RegionName)
    (sm sn shift maximum : Nat) (hshift : shift ≤ maximum)
    (hbound : sm + (maximum + 1) * sn < bounds region) (lane : TileIndex [2,2]) :
    (tilePointer region sm sn shift lane).2 < bounds region := by
  have hr : lane.1.val ≤ 1 := by have := lane.1.isLt; omega
  have hc : lane.2.1.val + shift ≤ maximum + 1 := by have := lane.2.1.isLt; omega
  exact lt_of_le_of_lt
    (Nat.add_le_add (by simpa using Nat.mul_le_mul_right sm hr)
      (Nat.mul_le_mul_right sn hc)) hbound

theorem contiguous_tile_injective : TileInjective 8 1 := by
  intro ⟨i, j, u⟩ ⟨i', j', u'⟩ h
  have hi : i = i' := Fin.ext (by dsimp at h; omega)
  have hj : j = j' := Fin.ext (by dsimp at h; omega)
  subst i'; subst j'; cases u; cases u'; rfl

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
