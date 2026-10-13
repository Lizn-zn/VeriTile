import bench.tritonbench_g.log_softmax.Proofs.Active

namespace VeriTile.Bench.TritonBenchG.LogSoftmax.SourceIO
open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₁
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

theorem forward_region
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (s₀ : BlockState) (xs : Fin (BLOCK_M * BLOCK_N) → ℝ)
    (hK : 0 < K)
    (hx : ∀ j : Fin (BLOCK_M * BLOCK_N),
      active (s₀.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
      s₀.readMem input_ptr (outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M
        (laneIdx BLOCK_M BLOCK_N j)) = xs j) :
    ∃ s1, exec ((log_softmax_kernel output_ptr input_ptr M N K BLOCK_M
        BLOCK_N).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin (BLOCK_M * BLOCK_N),
          active (s₀.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
          s1.readMem output_ptr (outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M
            (laneIdx BLOCK_M BLOCK_N j))
            = logSoftmaxOf BLOCK_M BLOCK_N
                (logSoftmaxPureTile (s₀.pids 0) M N BLOCK_M BLOCK_N xs)
                (laneIdx BLOCK_M BLOCK_N j))
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin (BLOCK_M * BLOCK_N),
            active (s₀.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
            o ≠ outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M
              (laneIdx BLOCK_M BLOCK_N j)) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := source_forward_terminates output_ptr input_ptr M N K
    BLOCK_M BLOCK_N hBN s₀
  have hs1' : exec (log_softmax_kernel output_ptr input_ptr M N K BLOCK_M
      BLOCK_N) s₀ = some s1 := hs1
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := SourceReadback.forward_active output_ptr input_ptr M N K BLOCK_M
      BLOCK_N hK s₀ s1 hs1' (laneIdx BLOCK_M BLOCK_N j) hj
    rw [h, logSoftmaxSpec,
      log_softmax_inputTile_eq s₀ input_ptr (s₀.pids 0) (s₀.pids 1) M N K
        BLOCK_M BLOCK_N xs hx]
  · refine source_forward_frame output_ptr input_ptr M N K BLOCK_M BLOCK_N
      hBN s₀ s1 hs1 r o (fun idx hact ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno (laneOf BLOCK_M BLOCK_N idx)
        (by rw [laneIdx_laneOf]; exact hact)
        (by rw [laneIdx_laneOf]; exact ho.symm)

theorem forward_io
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (hK : 0 < K) :
    logSoftmaxIO output_ptr input_ptr M N K BLOCK_M BLOCK_N
      ⊨ fun p₀ _ xs j =>
          logSoftmaxOf BLOCK_M BLOCK_N
            (logSoftmaxPureTile p₀ M N BLOCK_M BLOCK_N xs)
            (laneIdx BLOCK_M BLOCK_N j) := by
  refine Masked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact log_softmax_kernel_flattenOk output_ptr input_ptr M N K BLOCK_M BLOCK_N
  · intro bounds s hin hout _
    exact log_softmax_kernel_traceSafe output_ptr input_ptr M N K BLOCK_M BLOCK_N
      hBN bounds s hin hout
  · intro s₀ xs hx
    obtain ⟨s1, hs1, hval, hframe⟩ :=
      forward_region output_ptr input_ptr M N K BLOCK_M BLOCK_N
        hBN s₀ xs hK hx
    exact ⟨s1, hs1, hval, fun r o hcond _ => hframe r o hcond⟩

theorem backward_region
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s₀ : BlockState) (xs ys : Fin (BLOCK_M * BLOCK_N) → ℝ)
    (hundef : s₀.undef = (fun _ _ => 0))
    (hK : 0 < K)
    (hx : ∀ j : Fin (BLOCK_M * BLOCK_N),
      active (s₀.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
      s₀.readMem out_ptr (outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M
        (laneIdx BLOCK_M BLOCK_N j)) = xs j)
    (hy : ∀ j : Fin (BLOCK_M * BLOCK_N),
      active (s₀.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
      s₀.readMem out_grad_ptr (outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M
        (laneIdx BLOCK_M BLOCK_N j)) = ys j) :
    ∃ s1, exec ((log_softmax_backward_kernel out_ptr out_grad_ptr in_grad_ptr
        M N K BLOCK_M BLOCK_N).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin (BLOCK_M * BLOCK_N),
          active (s₀.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
          s1.readMem in_grad_ptr (outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M
            (laneIdx BLOCK_M BLOCK_N j))
            = logSoftmaxBackwardOf BLOCK_M BLOCK_N
                (logSoftmaxBackwardPureOutTile (s₀.pids 0) M N BLOCK_M BLOCK_N xs)
                (logSoftmaxBackwardPureGradTile (s₀.pids 0) M N BLOCK_M BLOCK_N ys)
                (laneIdx BLOCK_M BLOCK_N j))
      ∧ (∀ r o,
          (r ≠ in_grad_ptr ∨ ∀ j : Fin (BLOCK_M * BLOCK_N),
            active (s₀.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
            o ≠ outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M
              (laneIdx BLOCK_M BLOCK_N j)) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := source_backward_terminates out_ptr out_grad_ptr
    in_grad_ptr M N K BLOCK_M BLOCK_N s₀
  have hs1' : exec (log_softmax_backward_kernel out_ptr out_grad_ptr in_grad_ptr
      M N K BLOCK_M BLOCK_N) s₀ = some s1 := hs1
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := SourceReadback.backward_active out_ptr out_grad_ptr in_grad_ptr
      M N K BLOCK_M BLOCK_N hK s₀ s1 hs1' (laneIdx BLOCK_M BLOCK_N j) hj
    rw [h, logSoftmaxBackwardSpec,
      log_softmax_backward_outTile_eq s₀ out_ptr M N K BLOCK_M BLOCK_N xs hundef hx,
      log_softmax_backward_gradTile_eq s₀ out_grad_ptr M N K BLOCK_M BLOCK_N ys hundef hy]
  · refine source_backward_frame out_ptr out_grad_ptr in_grad_ptr M N K
      BLOCK_M BLOCK_N s₀ s1 hs1 r o (fun idx hact ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno (laneOf BLOCK_M BLOCK_N idx)
        (by rw [laneIdx_laneOf]; exact hact)
        (by rw [laneIdx_laneOf]; exact ho.symm)

theorem backward_io
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (hK : 0 < K) :
    Masked2DKernelIO₂.Implements
      (logSoftmaxBackwardIO out_ptr out_grad_ptr in_grad_ptr M N K BLOCK_M BLOCK_N)
      (fun p₀ _ xs ys j =>
          logSoftmaxBackwardOf BLOCK_M BLOCK_N
            (logSoftmaxBackwardPureOutTile p₀ M N BLOCK_M BLOCK_N xs)
            (logSoftmaxBackwardPureGradTile p₀ M N BLOCK_M BLOCK_N ys)
            (laneIdx BLOCK_M BLOCK_N j)) := by
  refine Masked2DKernelIO₂.Implements.intro_undef _ ?_ ?_ ?_
  · exact log_softmax_backward_kernel_flattenOk out_ptr out_grad_ptr in_grad_ptr
      M N K BLOCK_M BLOCK_N
  · intro bounds s hin1 hin2 hout _
    exact log_softmax_backward_kernel_traceSafe out_ptr out_grad_ptr in_grad_ptr
      M N K BLOCK_M BLOCK_N bounds s hin1 hin2 hout
  · intro s₀ xs ys hundef hx hy
    obtain ⟨s1, hs1, hval, hframe⟩ :=
      backward_region out_ptr out_grad_ptr in_grad_ptr
        M N K BLOCK_M BLOCK_N s₀ xs ys hundef
        hK hx hy
    exact ⟨s1, hs1, hval, fun r o hcond _ => hframe r o hcond⟩

end VeriTile.Bench.TritonBenchG.LogSoftmax.SourceIO
