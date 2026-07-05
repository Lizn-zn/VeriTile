# triton_linear_activation

- Source file: `triton_linear_activation.py`
- Corpus: TritonBench-G v1
- Status: DONE — DSL port (`TritonLinearActivation.lean`), genuine
  `activation(bias + Σ_k A·B)` closed-form spec, and bundled
  `ComputeCorrect.Realizes` summary
  (`triton_linear_activation_output_summary_general`) proven sorry-free.

`kernel_fma` is the xformers-style fused linear layer
`Out = activation(A × Wᵀ + bias)`: an L2-grouped linear-pid schedule, `%M`/`%N`
index wrapping (the kernel's "trick to avoid masking on M and N"), an optional
masked bias-row seed (`HAS_BIAS`), the `acc += tl.dot(a, b)` K-loop, an
optional pre-activation spill to `ACT_INPUTS` (`SHOULD_SAVE_ACT_INPUTS`, saved
for backward), and the string-constexpr activation gates
(`tanh` / `gelu` / `fast_gelu` / `relu` / identity, with the `@triton.jit`
helpers inlined — GELU via the exact real error function
`VeriTile.Math.realErf`, fast-GELU via `Real.tanh`).

The headline is one bundled, dimension-general theorem over all shapes,
strides, block dims, group size, both `Bool` constexpr flags, and **any**
`ACTIVATION` string: conjunct (2) proves every `C` cell is the genuine
activated linear form (a `gemmSum` `Finset.sum` over input memory), and
conjunct (3) proves every `ACT_INPUTS` cell is the un-activated pre-activation
value. Honest scope: the `K_LOAD_MASK_NEEDED = True` heuristics arm
(exact-multiple `K = BLOCK_K · numKBlocks`), and per-program tile-fit
hypotheses `hFitM`/`hFitN` (always true when `BLOCK_M ∣ M ∧ BLOCK_N ∣ N`) —
the kernel masks its store on the **already-wrapped** offsets, so an
overhanging tile genuinely wraps around instead of being masked off.
