import bench.tritonbench_g.cross_entropy1.Proofs.ForwardProjection
namespace VeriTile.Bench.TritonBenchG.CrossEntropy1.ForwardIO
open VeriTile.Triton ForwardReal
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
def sourceIO (storage : Storage) (swapped : Bool)
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool) : MetaGatherMasked2DKernelIO₂ₓ₂ where
  kernel := (forward storage swapped loss_ptr lse_ptr logits_ptr labels_ptr
    smoothing lse_square_scale ignored_index total_classes class_start_idx
    n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).eraseDType
  projection := ForwardProjection.projection storage swapped loss_ptr lse_ptr logits_ptr labels_ptr
    smoothing lse_square_scale ignored_index total_classes class_start_idx
    n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT
  mbufL := Region.cast labels_ptr
  inp := logits_ptr
  out1 := loss_ptr
  out2 := lse_ptr
  B := BLOCK_SIZE
  mwinL := fun pid₀ _ => pid₀
  read := fun pid₀ pid₁ _ j =>
    pid₀ * logits_row_stride + (pid₁ * BLOCK_SIZE + j.val)
  mask := fun _ pid₁ _ j => pid₁ * BLOCK_SIZE + j.val < n_cols
  gwin := fun pid₀ _ lab =>
    pid₀ * logits_row_stride + (lab - class_start_idx).toNat
  gmask := fun _ pid₁ lab =>
    lab ≠ ignored_index ∧
    lab - class_start_idx ≥ (↑(pid₁ * BLOCK_SIZE) : Int) ∧
    lab - class_start_idx < (↑(min n_cols ((pid₁ + 1) * BLOCK_SIZE)) : Int)
  write1 := fun pid₀ pid₁ _ => pid₁ * n_rows + pid₀
  write2 := fun pid₀ pid₁ _ => pid₁ * n_rows + pid₀

open scoped VeriTile.Triton.MetaGatherMasked2DKernelIO₂ₓ₂ in

theorem implements (storage : Storage) (swapped : Bool)
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (hB : 0 < BLOCK_SIZE)
    (hne : lse_ptr ≠ loss_ptr) (hLL : lse_ptr ≠ logits_ptr) :
    sourceIO storage swapped loss_ptr lse_ptr logits_ptr labels_ptr smoothing
        lse_square_scale ignored_index total_classes class_start_idx n_cols
        n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT ⊨
      fun _ pid₁ lab xs g =>
        if pid₁ * BLOCK_SIZE < n_cols then
          (ceLossLocal n_cols total_classes BLOCK_SIZE smoothing
             lse_square_scale ignored_index class_start_idx HAS_SMOOTHING SPLIT
             pid₁ lab xs g,
           ceBlockLSE n_cols BLOCK_SIZE pid₁ xs)
        else (0, 0) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  refine MetaGatherMasked2DKernelIO₂ₓ₂.Implements.intro _ ?_ ?_ ?_
  · apply ForwardProjection.flatten_as_base storage swapped loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing lse_square_scale ignored_index total_classes class_start_idx
      n_cols n_rows logits_row_stride (n+1) HAS_SMOOTHING SPLIT
    exact cross_entropy_fwd_flattenOk loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing lse_square_scale ignored_index total_classes class_start_idx
      n_cols n_rows logits_row_stride (n+1) HAS_SMOOTHING SPLIT
  · intro bounds s lab hlab hbL hbr hbg hbw1 hbw2
    apply ForwardProjection.safe_as_base storage swapped loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing lse_square_scale ignored_index total_classes class_start_idx
      n_cols n_rows logits_row_stride (n+1) HAS_SMOOTHING SPLIT bounds s
    exact cross_entropy_fwd_traceSafe loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing lse_square_scale ignored_index total_classes class_start_idx
      n_cols n_rows logits_row_stride n HAS_SMOOTHING SPLIT bounds s lab hlab
      hbL (fun j hj => hbr j hj) (fun h1 h2 h3 => hbg ⟨h1, h2, h3⟩)
      (hbw1 trivial) (hbw2 trivial)
  · intro s₀ lab xs g hlab hx hg
    obtain ⟨s1, hexec, hval1, hval2, hframe⟩ := cross_entropy_fwd_region_run
      loss_ptr lse_ptr logits_ptr labels_ptr smoothing lse_square_scale
      ignored_index total_classes class_start_idx n_cols n_rows
      logits_row_stride n HAS_SMOOTHING SPLIT hne hLL s₀ lab xs g hlab
      (fun j hj => hx j hj) (fun hgm => hg hgm)
    refine ⟨s1, (ForwardProjection.executes_as_base storage swapped loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing lse_square_scale ignored_index total_classes class_start_idx
      n_cols n_rows logits_row_stride (n+1) HAS_SMOOTHING SPLIT s₀).trans hexec, fun _ => ?_, fun _ => ?_, fun r o h1 h2 => ?_⟩
    · rw [show ((if s₀.pids 1 * (n+1) < n_cols then
          (ceLossLocal n_cols total_classes (n+1) smoothing lse_square_scale
             ignored_index class_start_idx HAS_SMOOTHING SPLIT (s₀.pids 1)
             lab xs g,
           ceBlockLSE n_cols (n+1) (s₀.pids 1) xs)
        else ((0:ℝ), (0:ℝ))).1 : ℝ)
        = (if s₀.pids 1 * (n+1) < n_cols then
            ceLossLocal n_cols total_classes (n+1) smoothing lse_square_scale
              ignored_index class_start_idx HAS_SMOOTHING SPLIT (s₀.pids 1)
              lab xs g
          else 0) from by split <;> rfl]
      exact hval1
    · rw [show ((if s₀.pids 1 * (n+1) < n_cols then
          (ceLossLocal n_cols total_classes (n+1) smoothing lse_square_scale
             ignored_index class_start_idx HAS_SMOOTHING SPLIT (s₀.pids 1)
             lab xs g,
           ceBlockLSE n_cols (n+1) (s₀.pids 1) xs)
        else ((0:ℝ), (0:ℝ))).2 : ℝ)
        = (if s₀.pids 1 * (n+1) < n_cols then
            ceBlockLSE n_cols (n+1) (s₀.pids 1) xs
          else 0) from by split <;> rfl]
      exact hval2
    · refine hframe r o ?_ ?_
      · rcases h1 with hner | hno
        · exact fun hc => hner hc.1.symm
        · exact fun hc => hno trivial hc.2.symm
      · rcases h2 with hner | hno
        · exact fun hc => hner hc.1.symm
        · exact fun hc => hno trivial hc.2.symm

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CrossEntropy1.ForwardIO
