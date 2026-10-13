import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadStep
import bench.tritonbench_g.nested_loops_processing.Proofs.RealMemory

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

/-- The ideal-real memory convention stores the finite fallback of an extended
floating value; integer, boolean, and pointer channels are stored exactly. This
is a memory representation map, not a source floating-point conversion. -/
def storedCell : (dtype : TileDType) → TileCarrier dtype → MemCell
  | .real, v => MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue v))
  | .fp32, v => MemCell.of .fp32 (FloatDType.fp32.ofReal (FloatDType.fp32.storeValue v))
  | .fp16, v => MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue v))
  | .bf16, v => MemCell.of .bf16 (FloatDType.bf16.ofReal (FloatDType.bf16.storeValue v))
  | .f8e4, v => MemCell.of .f8e4 (FloatDType.f8e4.ofReal (FloatDType.f8e4.storeValue v))
  | .f8e5, v => MemCell.of .f8e5 (FloatDType.f8e5.ofReal (FloatDType.f8e5.storeValue v))
  | .int, v => MemCell.of .int v
  | .nat, v => MemCell.of .nat v
  | .bool, v => MemCell.of .bool v
  | .ptr, v => MemCell.of .ptr v
  | .blockPtr, v => MemCell.of .blockPtr v

theorem write_as_cell (dtype : TileDType) (s : BlockState) (region : RegionName)
    (address : Nat) (value : TileCarrier dtype) :
    s.writeMemTyped dtype region address value = s.writeCell region address (storedCell dtype value) := by
  cases dtype <;> rfl

/-- A loaded floating value is already finite in the ideal-real semantics, so
copying it introduces no additional normalization. -/
theorem storedCell_read (dtype : TileDType) (s : BlockState) (region : RegionName) (address : Nat) :
    storedCell dtype (s.readMemValue dtype region address) =
      MemCell.of dtype (s.readMemValue dtype region address) := by
  cases dtype
  all_goals simp only [storedCell, BlockState.readMemValue]
  all_goals unfold BlockState.readMemAs
  all_goals split <;> simp [FloatDType.ofReal, FloatDType.storeValue]

theorem scatter_other_region (dtype : TileDType) (s : BlockState) (ptr : PointerTile)
    (value : Payload dtype) (region : RegionName) (h : ∀ i, region ≠ (ptr i).1) :
    (scatter dtype s ptr value).mem region = s.mem region := by
  unfold scatter
  generalize TileShape.allIndices [2,2] = lanes
  induction lanes generalizing s with
  | nil => rfl
  | cons i lanes ih =>
    rw [List.foldl_cons, ih, write_as_cell]
    funext address
    simp [BlockState.writeCell, h i]

/-- The four lane stores have the same row-major order as the independent oracle. -/
theorem scatter_copyTile (dtype : TileDType) (s : BlockState) (input : Nat → TileCarrier dtype)
    (dst : RegionName) (sm sn count : Nat) :
    (scatter dtype s (tilePointer dst sm sn (2 * count))
      (fun i ↦ input ((tilePointer dst sm sn (CopyMath.inputShift count) i).2))).mem dst =
      CopyMath.copyTile (fun address ↦ storedCell dtype (input address))
        sm sn (s.mem dst) count := by
  funext address
  simp [scatter, SourceChecks.scatter_lane_order, tilePointer, CopyMath.copyTile,
    CopyMath.offset, write_as_cell, BlockState.writeCell, Function.update_apply,
    List.range_succ, List.foldl_cons, List.foldl_nil]
  norm_cast

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
