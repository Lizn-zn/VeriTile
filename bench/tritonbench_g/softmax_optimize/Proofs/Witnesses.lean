import bench.tritonbench_g.softmax_optimize.Proofs.RealFullIO
import bench.tritonbench_g.softmax_optimize.Proofs.Geometry

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open VeriTile.Triton

/-- A nonempty two-tile, masked-tail mathematical execution. This witnesses
success and the full frame, rather than assuming an execution result. -/
theorem three_columns_two_lanes (Y X : RegionName) (hne : Y ≠ X)
    (s : BlockState) :
    ∃ t, exec (softmax_kernel_online_v2_surface Y X 1 3 2).toAlgKernel s = some t ∧
      (∀ j : Fin 3, t.readMem Y (linearOffset s 3 j) = softmaxOptimizeFullSpec s X 3 j) ∧
      RowFrame Y 3 s t := by
  exact full_surface_io Y X 1 3 2 (by norm_num) (by norm_num) hne s

/-- The inactive fourth lane of the second tile leaves its raw memory cell
unchanged, together with successful execution of both online/output passes. -/
theorem three_columns_tail_frame (Y X : RegionName) (hne : Y ≠ X) (s : BlockState) :
    ∃ t, exec (softmax_kernel_online_v2_surface Y X 1 3 2).toAlgKernel s = some t ∧
      (∀ j : Fin 3, t.readMem Y (linearOffset s 3 j) = softmaxOptimizeFullSpec s X 3 j) ∧
      t.mem Y (s.pid*3+3) = s.mem Y (s.pid*3+3) := by
  obtain ⟨t, ht, hout, hframe⟩ := three_columns_two_lanes Y X hne s
  exact ⟨t, ht, hout, hframe Y (s.pid*3+3) (Or.inr (Or.inr (le_refl _)))⟩

/-- Both lanes of the zero-column signed tail have negative source offsets. -/
theorem zero_columns_two_negative_lanes :
    FullSource.prevMultipleInt 0 2 = -2 ∧
    (∀ i : Fin 2, FullSource.prevMultipleInt 0 2 + (i.val : Int) < 0) := by
  exact ⟨FullSource.prevMultipleInt_zero 2 (by norm_num),
    FullSource.zero_tail_mask 2 (by norm_num)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize
