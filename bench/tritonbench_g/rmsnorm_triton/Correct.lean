import bench.tritonbench_g.rmsnorm_triton.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton
open VeriTile.Triton
open scoped VeriTile.Triton.StreamEmitMasked2DKernelIO₂

namespace Full.Final2
/-- **Full output summary**: the surface lowers to the algorithm layer, and the
masked store realizes the genuine multi-block RMS-norm closed form at every
global lane `k < N_SIZE`. Holds for arbitrary `N_SIZE` — no
`N_SIZE ≤ BLOCK_N_SIZE` hypothesis. -/
specification rmsnorm_full_output_summary
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (s : BlockState) (hB : 0 < B) (hNpos : 0 < N)
    (hox : o ≠ x) (how : o ≠ w)
    (hsok : 0 < sok) :
    (∃ alg, (VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o
        sxb sxm sxk srw sob som sok N B eps).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o
        sxb sxm sxk srw sob som sok N B eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N => True)
        (fun k => (o, outOff s sob som sok k.val)))
      (expected := fun k : Fin N => rmsSpecFull s x w sxb sxm sxk srw N eps k.val) := by
  exact rmsnorm_full_output_summary_bridge x w o sxb sxm sxk srw sob som sok N B eps s hB hNpos hox how hsok
end Full.Final2

/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `rmsnorm_triton` surface implements,
on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ two-pass
RMS-norm** over the streamed tiles: emitted window `(t, j)` holds
`x[t,j] · (√(Σ_guarded x²/N_SIZE + eps))⁻¹ · w[t,j]`, where the guarded
double sum folds the *entire* `x` stream — the spec `f` is exact real
arithmetic. The kernel has **zero rounding events** (loads, both passes'
arithmetic and the per-step stores are all at `.real`; the erased
`.to(tl.float32)`s are `.real → .real`), so the skin's boundary
quantization degenerates: the readback contract's `R.round .real` is the
identity by the model's defining `round_real`, and the `.real` in-loop
stores are exact under `execR R` — the ∀-`R` face holds via the
`RoundingModel` `.real` identity fields, not as a `.triv` special case.

Layer map: both passes are cast-free, so under `execR R` they collapse
verbatim onto the exact stepper and the proven
`preLoop` / `varInv` / `postStep` / `wbInv` invariant stack above is reused
unchanged; the `⊨[R]` face adds the `TraceSafeR` walk, the per-cell memory
frame (`rms_wbBody_step_frame`, the `mem` twin of `wbStep`), and the
stream-lane spec bridge (`sum_sq_mean` re-blocking `k ↔ (k/B, k%B)`).

All four hypotheses are truth-forced (exactly the exact headline
`rmsnorm_full_output_summary`'s side conditions):

* `hB : 0 < BLOCK_N_SIZE` — both loops step by `BLOCK_N_SIZE`
  (`range(0, N_SIZE, BLOCK_N_SIZE)`); at `BLOCK_N_SIZE = 0` and
  `N_SIZE > 0` neither `forRange` advances, `execR` cannot terminate the
  way the invariant stack requires, and the step index `i / B` is
  meaningless. It holds for every real launch.
* `hox : o ≠ x`, `how : o ≠ w` — pass 2 stores into `out` **between** its
  re-reads of `x` and `rms_w`; if `out` aliased either input, later blocks
  would re-read already-overwritten values and the closed form would be
  false.
* `hsok : 0 < stride_out_k` — the per-lane write window
  `pid₀·sob + pid₁·som + k·sok` is injective over global lanes only when
  the output column stride is nonzero (`outOff_injective_of_pos`); with
  `sok = 0` all lanes collide and the per-lane readback would be
  last-writer-wins. Always true for a real tensor.

Relation to the exact surface: the exact headline
`rmsnorm_full_output_summary` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face restates the same RMS-norm content on
the streaming emit skin, for every `R` at once (at the `.real` grid the two
faces carry the same exact cell). Both faces are kept per the
rounding-as-default doctrine. -/
specification rmsnorm_full_io_correctness (R : RoundingModel)
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hox : o ≠ x) (how : o ≠ w) (hsok : 0 < sok) :
    rmsnormKernelIO x w o sxb sxm sxk srw sob som sok N B eps ⊨[R]
      fun _ _ xs ws t j => rmsStreamSpec N B eps xs ws t j := by
  exact rmsnorm_full_io_correctness_bridge R x w o sxb sxm sxk srw sob som sok N B eps hB hox how hsok

end VeriTile.Bench.TritonBenchG.RmsnormTriton
