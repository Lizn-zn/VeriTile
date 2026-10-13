import bench.tritonbench_g.nested_loops_processing.Proofs.RealIO
import bench.tritonbench_g.nested_loops_processing.Proofs.RealBounds

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.RealExecution
open VeriTile.Triton FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

/-- A valid 8×8 input/output allocation, with a sentinel outside the store footprint. -/
def initial : BlockState where
  mem := fun region address ↦
    if region = "src" then MemCell.of .int (address : Int) else MemCell.of .int (-1)
  regs := fun _ _ _ ↦ none
  pids := fun _ ↦ 0

theorem witness_native_domain : TileInjective 8 1 ∧ 8 + 27 * 1 < 64 ∧ 8 + 47 * 1 < 64 :=
  ⟨contiguous_tile_injective, by decide, by decide⟩

/-- Actual 24-store execution exhibits cross-tile overwrite, final-tile copy,
and an unchanged sentinel. In particular the overwritten cell 8 contains 2. -/
theorem actual_overlap_witness :
    ∃ final, exec (source "src" "dst" 8 1) initial = some final ∧
      final.mem "dst" 8 = MemCell.of .int 2 ∧
      final.mem "dst" 47 = MemCell.of .int 27 ∧
      final.mem "dst" 55 = MemCell.of .int 35 ∧
      final.mem "dst" 56 = MemCell.of .int (-1) := by
  obtain ⟨final, he, _, hv⟩ := source_io initial "src" "dst" 8 1 (by decide)
  refine ⟨final, he, ?_, ?_, ?_, ?_⟩
  all_goals rw [hv]
  all_goals rfl

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.RealExecution

#guard_msgs (drop info) in
#auditModuleAxioms
