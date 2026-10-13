import bench.tritonbench_g.rmsnorm_fused.Proofs.Real
import bench.tritonbench_g.rmsnorm_fused.Proofs.Contracts
import bench.tritonbench_g.rmsnorm_fused.Proofs.FullContracts
import bench.tritonbench_g.rmsnorm_fused.Proofs.Math
import bench.tritonbench_g.rmsnorm_fused.Proofs.Precision
import bench.tritonbench_g.rmsnorm_fused.Proofs.Execution

namespace VeriTile.Bench.TritonBenchG.RmsnormFused
open VeriTile.Triton
open scoped VeriTile.Triton.StreamEmitMasked2DKernelIO₂

/-- Per-kernel output summary for `rms_norm_fwd_fused`: the DSL surface lowers to
the algorithm layer, and the masked store to `Y` is compute-correct — every
active lane (`i.val < N`) holds the RMSNorm spec `rmsnormSpec`, out-of-bounds
lanes are preserved. Stated under the `0 < N ≤ BLOCK_SIZE` single-block launch
precondition chosen by the Python wrapper. -/
specification rms_norm_fwd_fused_output_summary
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    (∃ alg, (rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i => rmsnormSpec s X W stride N BLOCK_SIZE eps i) := by
  exact rms_norm_fwd_fused_output_summary_bridge X Y W stride N BLOCK_SIZE eps s hNpos hNle hOutInj

/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `rms_norm_fwd_fused` surface
implements, on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ
two-pass RMS-norm** over the streamed tiles: emitted window `(t, j)` holds
`x[t,j] · (√(Σ_guarded x²/N + eps))⁻¹ · w[t,j]`, where the guarded double sum
folds the *entire* `x` stream — the spec `f` is exact real arithmetic. The
full Hoare triple is the skin's: for every disjoint flat allocation of
`[X, W, Y]`, every pid pair, and every launch state with both masked input
streams pinned, `execR R` terminates, every write-active emitted cell reads
back through `readMemAs .real` as the ideal value, and every flat cell
outside the active output window is untouched.

Rounding story: the kernel has **zero rounding events**. Both masked loads,
both passes' arithmetic and the in-loop store are at `.real`, and the two
`.to(tl.float32)` casts are erased outright by the compute-to-algorithm
lowering (`fused_body_decomp` is `rfl` onto a body with no `Op.castFloat`).
The skin's boundary quantization therefore degenerates: the readback
contract's `R.round .real` is the identity by the model's defining
`round_real`, and the `.real` in-loop stores are exact under `execR R` — the
∀-`R` face holds via the `RoundingModel` `.real` identity fields, not as a
`.triv` special case.

Layer map: `fused_castFree` collapses `execR R` onto the exact `exec`
statement by statement (the two `forRange` clauses lift their cast-free
bodies through `stepForRangeAuxR_castFree`), so the proven exact readback
`rms_norm_fwd_fused_correct` is reused unchanged; the `⊨[R]` face adds the
`TraceSafeR` walk (`rms_norm_fwd_fused_traceSafeR`), termination
(`rms_norm_fwd_fused_terminates`), the per-cell memory frame
(`rms_norm_fwd_fused_frame`), and the stream-lane spec bridge
(`rmsFusedStreamSpec_eq_rmsnormSpec`).

Both hypotheses are the *exact* headline `rms_norm_fwd_fused_output_summary`'s
own launch precondition, with the same provenance:

* `hNpos : 0 < N`, `hNle : N ≤ BLOCK_SIZE` — the Python wrapper picks
  `BLOCK_SIZE ≥ N` and raises otherwise, so every real launch is in this
  single-block regime; there both `range(0, N, BLOCK_SIZE)` passes execute
  exactly the `off = 0` iteration, which is what makes the stream one step
  wide (`T := 1`) and the closed form available.

The exact headline's third side condition `hOutInj` is **not** carried here:
the write window `pid₀·stride + i` is injective in `i` outright, and this
face discharges it.

Relation to the exact surface: the exact headline
`rms_norm_fwd_fused_output_summary` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face restates the same RMS-norm content on
the streaming emit skin, for every `R` at once (at the `.real` grid the two
faces carry the same exact cell). Both faces are kept per the
rounding-as-default doctrine. -/
specification rms_norm_fwd_fused_io_correctness (R : RoundingModel)
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE) :
    rmsnormFusedKernelIO X Y W stride N BLOCK_SIZE eps ⊨[R]
      fun _ _ xs ws t j => rmsFusedStreamSpec N BLOCK_SIZE eps xs ws t j := by
  exact rms_norm_fwd_fused_io_correctness_bridge R X Y W stride N BLOCK_SIZE eps hNpos hNle

/-- Every floating storage profile and both complete source variants implement the
independent full logical-row RMS formula for arbitrary row lengths, with successful
flat execution and a full memory frame. Output/input separation protects the initial
row snapshot across multiple source iterations. -/
specification typed_real_correct (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hXY : X ≠ Y) (hWY : W ≠ Y) :
    General.FullExactIO
      (typedKernel input weight output v X Y W stride N B eps).eraseDType
      X Y W stride N eps :=
  General.typed_full_io input weight output v X Y W stride N B eps hB hXY hWY

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.RmsnormFused
