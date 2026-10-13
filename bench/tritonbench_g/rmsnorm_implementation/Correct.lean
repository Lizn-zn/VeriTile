import bench.tritonbench_g.rmsnorm_implementation.Proofs.Real
import bench.tritonbench_g.rmsnorm_implementation.Proofs.Contracts
import bench.tritonbench_g.rmsnorm_implementation.Proofs.Precision
import bench.tritonbench_g.rmsnorm_implementation.Proofs.Execution
import bench.tritonbench_g.rmsnorm_implementation.Proofs.Math
import bench.tritonbench_g.rmsnorm_implementation.Proofs.Safety

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation
open VeriTile.Triton
open scoped VeriTile.Triton.StreamEmitMasked2DKernelIO₂

/-- Per-kernel output summary for `rmsnorm_triton`: the DSL surface lowers to the
algorithm layer, and the masked store to `out_ptr` is compute-correct for
arbitrary `N_SIZE` — every output column holds the full-`N` RMSNorm spec
`rmsnormWeightedYFullNSpec`. Built on the multi-block `*_compute_fullN_correct`
result; requires `0 < BLOCK_N_SIZE`, `0 < stride_out_k`, and output/input
disjointness. -/
specification rmsnorm_implementation_output_summary
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hBlockPos : 0 < BLOCK_N_SIZE)
    (hStrideOutKPos : 0 < stride_out_k)
    (hXOutNe : x_ptr ≠ out_ptr)
    (hWOutNe : rms_w_ptr ≠ out_ptr) :
    (∃ alg, (rmsnorm_implementation x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE
        eps).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rmsnorm_implementation x_ptr rms_w_ptr out_ptr
        stride_x_batch stride_x_m stride_x_k stride_rms_w
        stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N_SIZE => True)
        (fun i => (out_ptr,
          outColOffset s stride_out_batch stride_out_m stride_out_k i.val)))
      (expected := fun i =>
        rmsnormWeightedYFullNSpec s x_ptr rms_w_ptr stride_x_batch stride_x_m
          stride_x_k stride_rms_w N_SIZE BLOCK_N_SIZE eps i) := by
  exact rmsnorm_implementation_output_summary_bridge x_ptr rms_w_ptr out_ptr stride_x_batch stride_x_m stride_x_k stride_rms_w stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps s hBlockPos hStrideOutKPos hXOutNe hWOutNe

/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `rmsnorm_implementation` surface
implements, on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ
two-pass RMS-norm** over the streamed tiles: emitted window `(t, j)` holds
`x[t,j] / √(Σ_guarded x²/N_SIZE + eps) · w[t,j]`, where the guarded double
sum folds the *entire* `x` stream — the spec `f` is exact real arithmetic
in the kernel's own division spelling. The kernel has **zero rounding
events** (loads, both passes' arithmetic and the per-step stores are all at
`.real`; the erased `.to(tl.float32)` is `.real → .real`), so the skin's
boundary quantization degenerates: the readback contract's `R.round .real`
is the identity by the model's defining `round_real`, and the `.real`
in-loop stores are exact under `execR R` — the ∀-`R` face holds via the
`RoundingModel` `.real` identity fields, not as a `.triv` special case.

Layer map: both passes are cast-free, so under `execR R` they collapse
verbatim onto the exact stepper and the proven
`rmsVarLoopContextInvariant` / `rmsStdPostLoop_step_to_out_init` /
`rmsOutLoopContextInvariant` invariant stack above is reused unchanged; the
`⊨[R]` face adds the `TraceSafeR` walk, the per-cell memory frame
(`rmsImplOutBody_step_frame`, the `mem` twin of
`rmsOutLoopBody_step_preserves_old_output`), and the stream-lane spec
bridge (`rmsImpl_sum_sq_mean` re-blocking `k ↔ (k/B, k%B)`).

All four hypotheses are truth-forced (exactly the exact headline
`rmsnorm_implementation_output_summary`'s side conditions):

* `hBlockPos : 0 < BLOCK_N_SIZE` — both loops step by `BLOCK_N_SIZE`
  (`range(0, N_SIZE, BLOCK_N_SIZE)`); at `BLOCK_N_SIZE = 0` and
  `N_SIZE > 0` neither `forRange` advances, `execR` cannot terminate the
  way the invariant stack requires, and the step index `i / B` is
  meaningless. It holds for every real launch.
* `hXOutNe : x_ptr ≠ out_ptr`, `hWOutNe : rms_w_ptr ≠ out_ptr` — pass 2
  stores into `out` **between** its re-reads of `x` and `rms_w`; if `out`
  aliased either input, later blocks would re-read already-overwritten
  values and the closed form would be false.
* `hStrideOutKPos : 0 < stride_out_k` — the per-lane write window
  `pid₀·sob + pid₁·som + k·sok` is injective over global lanes only when
  the output column stride is nonzero
  (`outColOffset_fin_injective_of_stride_out_k_pos`); with
  `stride_out_k = 0` all lanes collide and the per-lane readback would be
  last-writer-wins. Always true for a real tensor.

Relation to the exact surface: the exact headline
`rmsnorm_implementation_output_summary` (`Realizes_without_Rounding`) above
is retained unchanged; this `⊨[R]` face restates the same RMS-norm content
on the streaming emit skin, for every `R` at once (at the `.real` grid the
two faces carry the same exact cell). Both faces are kept per the
rounding-as-default doctrine. -/
specification rmsnorm_implementation_io_correctness (R : RoundingModel)
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE : Nat)
    (eps : ℝ)
    (hBlockPos : 0 < BLOCK_N_SIZE)
    (hXOutNe : x_ptr ≠ out_ptr) (hWOutNe : rms_w_ptr ≠ out_ptr)
    (hStrideOutKPos : 0 < stride_out_k) :
    rmsnormImplementationKernelIO x_ptr rms_w_ptr out_ptr stride_x_batch
        stride_x_m stride_x_k stride_rms_w stride_out_batch stride_out_m
        stride_out_k N_SIZE BLOCK_N_SIZE eps ⊨[R]
      fun _ _ xs ws t j =>
        rmsImplStreamSpec N_SIZE BLOCK_N_SIZE eps xs ws t j := by
  exact rmsnorm_implementation_io_correctness_bridge R x_ptr rms_w_ptr out_ptr stride_x_batch stride_x_m stride_x_k stride_rms_w stride_out_batch stride_out_m stride_out_k N_SIZE BLOCK_N_SIZE eps hBlockPos hXOutNe hWOutNe hStrideOutKPos

/-- Both complete typed programs realize the full-row RMSNorm formula with
successful flat execution, active bounds and the complete memory frame.
This is exact real mathematics, independent of the backend FP equivalence. -/
specification typed_real_correct (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hXO : X ≠ O) (hWO : W ≠ O) (hsok : 0 < sok) :
    VeriTile.Spec.Real
      (ExactIO (typedIO input weight output v X W O sxb sxm sxk sw sob som sok N B eps)
        (fun _ _ xs ws t j => rmsImplStreamSpec N B eps xs ws t j)) :=
  typed_exact_io input weight output v X W O sxb sxm sxk sw sob som sok N B eps
    hB hXO hWO hsok

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormImplementation
