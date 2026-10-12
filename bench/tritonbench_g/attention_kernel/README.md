# attention_kernel

Relative-position-biased online attention from `attention_kernel.py`,
`_fwd_kernel_aligned`. The three public entries share the full typed source:

- `Kernels.lean`: fp16 and bf16 originals and reciprocal candidates. The sole
  candidate change is `acc / l_i[:, None]` → `acc * (1.0 / l_i[:, None])`.
- `Correct.lean`: both independent real IO specifications, including every
  input/output window and the base-2 softmax weighted-sum formula.
- `FPEquiv.lean`: full-program conditional equivalence; prints `div_mul_rcp`.

The real theorem covers positive block/head dimensions, one or more whole key
blocks (`N_CTX = BLOCK_N * nB`), `P_SEQ=0`, contiguous head rows and unit inner
stride. Head and bias strides remain symbolic. It proves successful flat-memory
execution, every output and the frame outside the output window. Allocation
bounds cover all unmasked accesses. The unused Z/H/B0_NUMEL source parameters
are filled with constants; they do not restrict execution. Broader independent
row/inner strides and partial key blocks are outside this real specification.

The FP theorem keeps all source strides and dimensions symbolic. Its domain
checks the actual post-loop fp32 accumulator lanes for finiteness and row
denominators for finiteness/nonzero. It preserves complete state and failure;
no reduction, exp2 or dot identity is assumed. Both dtypes have a checked
nonempty 2×2 execution/domain witness in `Proofs/TypedExecution.lean`, in the
same abstract model satisfying the admitted rules. This is a consistency
witness, not a GPU result or a restriction of the theorem.

Source details checked in `bench/tests/FPAttentionKernelSource.lean`:

- Q/K/V/B0/Out use the selected storage dtype. Q scaling is fp32 followed by a
  storage cast; P is cast back before the value dot. Both default dot outputs
  and the accumulator are fp32.
- Bias addition and multiplication remain at storage precision. The floating
  scalar does not promote the bias tensor; its implicit conversion is explicit
  in Lean. This accounts for a textual source-comparison exception.
- B0 supplies static b1 and a per-key-block b0. The real IO packs them into one
  stream with BLOCK_N+1 columns. Both terms occur in the independent score.
- V uses the source's K-head offset; Out uses its Q-head offset. Both bias
  windows, unchecked block pointers, exp2, casts and store remain visible.

`Proofs/RealKernel.lean` proves the cast-free mathematical execution and IO;
`RealSource.lean` connects it to both storage dtypes. `RealNormalization.lean`
proves full real execution equality for the candidate, including flat memory.
`Real.lean` assembles the contracts. These proofs do not import the FP proof.
`Proofs/Legacy.lean` and `LegacyProjection.lean` preserve the historical
cast-bearing model for reference; current correctness does not depend on them.

Validation passed: `TritonBenchGVerified`, the two focused regressions, 29
inventory/layout/shard tests, source gates, scope fingerprints and manifest.
Independent comparator replay and trust checks accepted all 13 affected files,
with 204 named theorem targets and 696 matching snapshot hashes.
Details: `../verification_round.md`; logs: `Logs/comparator-check-hfg0l07c`.
These per-program results do not establish concrete IEEE identities, host grid
composition or GPU performance.
