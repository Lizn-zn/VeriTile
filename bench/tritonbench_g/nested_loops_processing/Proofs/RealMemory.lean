import bench.tritonbench_g.nested_loops_processing.Proofs.RealStep
import bench.tritonbench_g.nested_loops_processing.Proofs.SourceChecks
import bench.tritonbench_g.nested_loops_processing.Proofs.Math

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def tilePointer (region : RegionName) (sm sn shift : Nat) : PointerTile :=
  fun i ↦ (region, i.1.val * sm + (i.2.1.val + shift) * sn)

theorem tilePointer_bump (region : RegionName) (sm sn shift : Nat) :
    bump (tilePointer region sm sn shift) (2 * sn) = tilePointer region sm sn (shift + 2) := by
  funext i
  simp only [bump, tilePointer, Prod.mk.injEq, true_and]
  ring

theorem tilePointer_bump_count (region : RegionName) (sm sn count : Nat) :
    bump (tilePointer region sm sn (2 * count)) (2 * sn) =
      tilePointer region sm sn (2 * (count + 1)) := by
  rw [tilePointer_bump]
  congr 1

theorem tilePointer_offset (region : RegionName) (sm sn shift : Nat) (i : TileIndex [2,2]) :
    (tilePointer region sm sn shift i).2 =
      (CopyMath.offset sm sn shift i.1.val i.2.1.val).toNat := by
  simp [tilePointer, CopyMath.offset]
  norm_cast

theorem scatter_other_region (s : BlockState) (ptr : PointerTile) (v : IntTile)
    (r : RegionName) (h : ∀ i, r ≠ (ptr i).1) :
    (scatter s ptr v).mem r = s.mem r := by
  unfold scatter
  generalize TileShape.allIndices [2,2] = lanes
  induction lanes generalizing s with
  | nil => rfl
  | cons i lanes ih =>
    rw [List.foldl_cons, ih]
    funext address
    simp [BlockState.writeMemTyped, h i]

theorem scatter_source_preserved (s : BlockState) (dst src : RegionName) (sm sn shift : Nat)
    (v : IntTile) (h : src ≠ dst) (address : Nat) :
    (scatter s (tilePointer dst sm sn shift) v).readMemValue .int src address =
      s.readMemValue .int src address := by
  have hm := scatter_other_region s (tilePointer dst sm sn shift) v src
    (fun _ ↦ h)
  simp only [BlockState.readMemValue, BlockState.readMemTyped, hm]

/-- The concrete four-lane scatter has the same order as the independent row/column oracle. -/
theorem scatter_copyTile (s : BlockState) (input : Nat → Int)
    (dst : RegionName) (sm sn t : Nat) :
    (scatter s (tilePointer dst sm sn (2 * t))
      (fun i ↦ input ((tilePointer dst sm sn (CopyMath.inputShift t) i).2))).mem dst =
      CopyMath.copyTile (fun address ↦ MemCell.of .int (input address)) sm sn (s.mem dst) t := by
  funext address
  simp [scatter, SourceChecks.scatter_lane_order, tilePointer, CopyMath.copyTile,
    CopyMath.offset, BlockState.writeMemTyped, Function.update_apply,
    List.range_succ, List.foldl_cons, List.foldl_nil]
  norm_cast

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
